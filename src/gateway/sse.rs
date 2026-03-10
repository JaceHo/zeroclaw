//! Server-Sent Events (SSE) stream for real-time event delivery.
//!
//! Wraps the broadcast channel in AppState to deliver events to web dashboard clients.

use super::AppState;
use axum::{
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
};
use std::convert::Infallible;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

/// GET /api/events — SSE event stream
pub async fn handle_sse_events(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    // Auth check
    if state.pairing.require_pairing() {
        let token = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|auth| auth.strip_prefix("Bearer "))
            .unwrap_or("");

        if !state.pairing.is_authenticated(token) {
            return (
                StatusCode::UNAUTHORIZED,
                "Unauthorized — provide Authorization: Bearer <token>",
            )
                .into_response();
        }
    }

    // Enforce max SSE connection limit using atomic fetch_add + rollback to avoid
    // TOCTOU races (a plain load-then-check would allow concurrent requests to all
    // pass the check before any of them increment the counter).
    let max = state.max_sse_connections;
    let counter = Arc::clone(&state.sse_connections);
    if max > 0 {
        let prev = state.sse_connections.fetch_add(1, Ordering::AcqRel);
        if prev >= max {
            // Over limit — roll back the speculative increment.
            state.sse_connections.fetch_sub(1, Ordering::Release);
            return (StatusCode::SERVICE_UNAVAILABLE, "Too many SSE connections").into_response();
        }
    } else {
        // Unlimited — still track for observability.
        state.sse_connections.fetch_add(1, Ordering::AcqRel);
    }

    let rx = state.event_tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(
        |result: Result<
            serde_json::Value,
            tokio_stream::wrappers::errors::BroadcastStreamRecvError,
        >| {
            use tokio_stream::wrappers::errors::BroadcastStreamRecvError;
            match result {
                Ok(mut value) => {
                    // Strip internal dedup ID before delivering to clients
                    if let Some(obj) = value.as_object_mut() {
                        obj.remove("_zcid");
                    }
                    Some(Ok::<_, Infallible>(
                        Event::default().data(value.to_string()),
                    ))
                }
                Err(BroadcastStreamRecvError::Lagged(n)) => {
                    let lagged = serde_json::json!({
                        "type": "lagged",
                        "missed": n,
                    });
                    Some(Ok(Event::default().data(lagged.to_string())))
                }
            }
        },
    );

    // Wrap stream to decrement counter when the client disconnects.
    let counted_stream = SseConnectionStream {
        inner: stream,
        _guard: SseConnectionGuard(counter),
    };

    Sse::new(counted_stream)
        .keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("ping"),
        )
        .into_response()
}

/// Wrapper stream that holds an RAII guard for SSE connection counting.
struct SseConnectionStream<S> {
    inner: S,
    _guard: SseConnectionGuard,
}

impl<S: futures_util::Stream + Unpin> futures_util::Stream for SseConnectionStream<S> {
    type Item = S::Item;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::new(&mut self.inner).poll_next(cx)
    }
}

/// RAII guard that decrements the SSE connection counter when dropped.
struct SseConnectionGuard(Arc<std::sync::atomic::AtomicUsize>);

impl Drop for SseConnectionGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Release);
    }
}

/// Broadcast observer that forwards events to the SSE event bus.
pub struct BroadcastObserver {
    inner: Box<dyn crate::observability::Observer>,
    tx: std::sync::Arc<dyn super::event_bus::EventBus>,
}

impl BroadcastObserver {
    pub fn new(
        inner: Box<dyn crate::observability::Observer>,
        tx: std::sync::Arc<dyn super::event_bus::EventBus>,
    ) -> Self {
        Self { inner, tx }
    }
}

impl crate::observability::Observer for BroadcastObserver {
    fn record_event(&self, event: &crate::observability::ObserverEvent) {
        // Forward to inner observer
        self.inner.record_event(event);

        // Broadcast to SSE subscribers
        let json = match event {
            crate::observability::ObserverEvent::LlmRequest {
                provider, model, ..
            } => serde_json::json!({
                "type": "llm_request",
                "provider": provider,
                "model": model,
                "timestamp": chrono::Utc::now().to_rfc3339(),
            }),
            crate::observability::ObserverEvent::ToolCall {
                tool,
                duration,
                success,
            } => serde_json::json!({
                "type": "tool_call",
                "tool": tool,
                "duration_ms": duration.as_millis(),
                "success": success,
                "timestamp": chrono::Utc::now().to_rfc3339(),
            }),
            crate::observability::ObserverEvent::ToolCallStart { tool } => serde_json::json!({
                "type": "tool_call_start",
                "tool": tool,
                "timestamp": chrono::Utc::now().to_rfc3339(),
            }),
            crate::observability::ObserverEvent::Error { component, message } => {
                serde_json::json!({
                    "type": "error",
                    "component": component,
                    "message": message,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                })
            }
            crate::observability::ObserverEvent::AgentStart { provider, model } => {
                serde_json::json!({
                    "type": "agent_start",
                    "provider": provider,
                    "model": model,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                })
            }
            crate::observability::ObserverEvent::AgentEnd {
                provider,
                model,
                duration,
                tokens_used,
                cost_usd,
            } => serde_json::json!({
                "type": "agent_end",
                "provider": provider,
                "model": model,
                "duration_ms": duration.as_millis(),
                "tokens_used": tokens_used,
                "cost_usd": cost_usd,
                "timestamp": chrono::Utc::now().to_rfc3339(),
            }),
            crate::observability::ObserverEvent::LlmResponse {
                provider,
                model,
                duration,
                success,
                error_message,
                input_tokens,
                output_tokens,
            } => serde_json::json!({
                "type": "llm_response",
                "provider": provider,
                "model": model,
                "duration_ms": duration.as_millis(),
                "success": success,
                "error_message": error_message,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "timestamp": chrono::Utc::now().to_rfc3339(),
            }),
            crate::observability::ObserverEvent::TurnComplete => serde_json::json!({
                "type": "turn_complete",
                "timestamp": chrono::Utc::now().to_rfc3339(),
            }),
            crate::observability::ObserverEvent::ChannelMessage {
                channel,
                direction,
            } => serde_json::json!({
                "type": "channel_message",
                "channel": channel,
                "direction": direction,
                "timestamp": chrono::Utc::now().to_rfc3339(),
            }),
            _ => return, // Skip HeartbeatTick and other internal events
        };

        self.tx.publish(json);
    }

    fn record_metric(&self, metric: &crate::observability::traits::ObserverMetric) {
        self.inner.record_metric(metric);
    }

    fn flush(&self) {
        self.inner.flush();
    }

    fn name(&self) -> &str {
        "broadcast"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        // Delegate to inner observer so downcast_ref::<PrometheusObserver>()
        // works correctly when the inner observer is Prometheus.
        self.inner.as_any()
    }
}
