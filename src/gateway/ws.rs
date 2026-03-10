//! WebSocket agent chat handler with full tool execution.
//!
//! Uses the same `process_message` agent loop as Telegram/Discord/WhatsApp,
//! giving the web UI access to tools, memory, security policy, and hardware.
//!
//! Protocol:
//! ```text
//! Client -> Server: {"type":"message","content":"Hello"}
//! Server -> Client: {"type":"chunk","content":"<token>"}   (streaming delta)
//! Server -> Client: {"type":"done","full_response":"..."}
//! Server -> Client: {"type":"error","message":"..."}
//! ```

use super::AppState;
use axum::{
    extract::{
        ws::{Message, WebSocket},
        Query, State, WebSocketUpgrade,
    },
    response::IntoResponse,
};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

/// Server-side ping interval for keepalive / half-open detection.
const PING_INTERVAL: Duration = Duration::from_secs(30);
/// If no Pong is received within this duration after a Ping, the connection
/// is considered dead and will be closed.
const PONG_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Deserialize)]
pub struct WsQuery {
    pub token: Option<String>,
}

/// GET /ws/chat — WebSocket upgrade for agent chat
pub async fn handle_ws_chat(
    State(state): State<AppState>,
    Query(params): Query<WsQuery>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    // Auth via query param (browser WebSocket limitation)
    if state.pairing.require_pairing() {
        let token = params.token.as_deref().unwrap_or("");
        if !state.pairing.is_authenticated(token) {
            return (
                axum::http::StatusCode::UNAUTHORIZED,
                "Unauthorized — provide ?token=<bearer_token>",
            )
                .into_response();
        }
    }

    // Enforce max WS connection limit using atomic fetch_add + rollback to avoid
    // TOCTOU races (a plain load-then-check would allow concurrent upgrades to all
    // pass the check before any of them increment the counter).
    let max = state.max_ws_connections;
    if max > 0 {
        let prev = state.ws_connections.fetch_add(1, Ordering::AcqRel);
        if prev >= max {
            // Over limit — roll back the speculative increment.
            state.ws_connections.fetch_sub(1, Ordering::Release);
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "Too many WebSocket connections",
            )
                .into_response();
        }
        // Slot reserved — pass the guard to handle_socket so it decrements on drop.
        let guard = WsConnectionGuard(Arc::clone(&state.ws_connections));
        return ws
            .on_upgrade(move |socket| handle_socket_with_guard(socket, state, guard))
            .into_response();
    }

    // Unlimited — still track for observability so /metrics can report active WS count.
    let guard = WsConnectionGuard(Arc::clone(&state.ws_connections));
    state.ws_connections.fetch_add(1, Ordering::AcqRel);
    ws.on_upgrade(move |socket| handle_socket_with_guard(socket, state, guard))
        .into_response()
}

/// Handle socket when the connection slot was already reserved by the caller.
async fn handle_socket_with_guard(socket: WebSocket, state: AppState, guard: WsConnectionGuard) {
    handle_socket_inner(socket, state, Some(guard)).await;
}

async fn handle_socket_inner(
    socket: WebSocket,
    state: AppState,
    _guard: Option<WsConnectionGuard>,
) {
    let (ws_sender, mut receiver) = socket.split();
    let ws_sender = Arc::new(Mutex::new(ws_sender));

    // --- CN-005: Server-side ping interval ---
    // Track the last time we received a Pong so we can detect stale connections.
    let last_pong = Arc::new(Mutex::new(std::time::Instant::now()));

    // Connection-level cancellation token: cancelled when the client disconnects
    // or the ping/pong watchdog fires.
    let conn_token = CancellationToken::new();

    // Spawn a task that sends Ping frames at a fixed interval and closes the
    // connection if no Pong arrives within PONG_TIMEOUT.
    let ping_sender = Arc::clone(&ws_sender);
    let ping_pong_tracker = Arc::clone(&last_pong);
    let ping_conn_token = conn_token.clone();
    let ping_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(PING_INTERVAL);
        // The first tick completes immediately; skip it so the first real ping
        // fires after PING_INTERVAL.
        interval.tick().await;

        loop {
            tokio::select! {
                _ = interval.tick() => {}
                () = ping_conn_token.cancelled() => break,
            }

            // Send a Ping frame.
            let send_ok = ping_sender
                .lock()
                .await
                .send(Message::Ping(vec![].into()))
                .await
                .is_ok();
            if !send_ok {
                ping_conn_token.cancel();
                break;
            }

            // Check if last Pong is within the timeout window.
            let elapsed = ping_pong_tracker.lock().await.elapsed();
            if elapsed > PONG_TIMEOUT {
                tracing::warn!("WS client pong timeout ({elapsed:?}), closing connection");
                // Best-effort close frame before cancelling.
                let _ = ping_sender.lock().await.send(Message::Close(None)).await;
                ping_conn_token.cancel();
                break;
            }
        }
    });

    // Build session context once on connect — provider, tools, memory, etc.
    // are reused across all turns in this WebSocket connection.
    let config = state.config.lock().clone();
    let mut ctx = match crate::agent::AgentSessionContext::from_config(&config).await {
        Ok(ctx) => ctx,
        Err(e) => {
            let sanitized = crate::providers::sanitize_api_error(&e.to_string());
            let err = serde_json::json!({
                "type": "error",
                "message": format!("Session init failed: {sanitized}"),
            });
            let _ = ws_sender
                .lock()
                .await
                .send(Message::Text(err.to_string().into()))
                .await;
            conn_token.cancel();
            ping_task.abort();
            return;
        }
    };

    // --- CN-002: inject the gateway's broadcast observer so inner events
    // (tool-call, LLM-request, etc.) are forwarded to the SSE dashboard.
    ctx.set_observer(state.observer.clone());

    // Persistent conversation history for the lifetime of this connection.
    let mut history = ctx.initial_history();

    // --- CN-006: per-connection sliding-window rate limiter ---
    let rate_limit = state.ws_rate_limit_per_minute;
    let mut rate_window_start = std::time::Instant::now();
    let mut rate_window_count: u32 = 0;

    loop {
        let msg = tokio::select! {
            biased;
            () = conn_token.cancelled() => break,
            next = receiver.next() => next,
        };

        let msg = match msg {
            Some(Ok(Message::Text(text))) => text,
            // --- CN-005: handle Ping/Pong frames ---
            Some(Ok(Message::Ping(data))) => {
                let _ = ws_sender.lock().await.send(Message::Pong(data)).await;
                continue;
            }
            Some(Ok(Message::Pong(_))) => {
                // Update last-pong timestamp for the keepalive watchdog.
                *last_pong.lock().await = std::time::Instant::now();
                continue;
            }
            Some(Ok(Message::Close(_)) | Err(_)) | None => {
                // Client disconnected — cancel any in-flight turn.
                conn_token.cancel();
                break;
            }
            _ => continue,
        };

        // Parse incoming message
        let parsed: serde_json::Value = match serde_json::from_str(&msg) {
            Ok(v) => v,
            Err(_) => {
                let err = serde_json::json!({"type": "error", "message": "Invalid JSON"});
                let _ = ws_sender
                    .lock()
                    .await
                    .send(Message::Text(err.to_string().into()))
                    .await;
                continue;
            }
        };

        let msg_type = parsed["type"].as_str().unwrap_or("");
        if msg_type != "message" {
            continue;
        }

        let content = parsed["content"].as_str().unwrap_or("").to_string();
        if content.is_empty() {
            continue;
        }

        // --- CN-006: enforce per-connection rate limit ---
        if rate_limit > 0 {
            let now = std::time::Instant::now();
            if now.duration_since(rate_window_start) >= Duration::from_secs(60) {
                // Window expired — reset.
                rate_window_start = now;
                rate_window_count = 0;
            }
            rate_window_count += 1;
            if rate_window_count > rate_limit {
                let err = serde_json::json!({
                    "type": "error",
                    "message": format!(
                        "Rate limit exceeded: max {} messages per minute. Please wait.",
                        rate_limit
                    ),
                });
                let _ = ws_sender
                    .lock()
                    .await
                    .send(Message::Text(err.to_string().into()))
                    .await;
                continue;
            }
        }

        let provider_label = &ctx.provider_name;
        let turn_start = std::time::Instant::now();

        // Broadcast agent_start event via observer (adds timestamp + metrics)
        state
            .observer
            .record_event(&crate::observability::ObserverEvent::AgentStart {
                provider: provider_label.to_string(),
                model: state.model.clone(),
            });

        // Auto-save to memory (gateway webhook handlers do their own;
        // the WS handler uses the generic key like the CLI REPL).
        ctx.auto_save_user_message(&content).await;

        // Create streaming delta channel
        let (delta_tx, mut delta_rx) = tokio::sync::mpsc::channel::<String>(256);

        // Spawn relay task: reads token deltas and sends them as WebSocket chunk messages
        let relay_sender = Arc::clone(&ws_sender);
        let relay_task = tokio::spawn(async move {
            while let Some(delta) = delta_rx.recv().await {
                // Skip the clear sentinel — it is an internal signal for draft updaters
                if delta == crate::agent::loop_::DRAFT_CLEAR_SENTINEL {
                    continue;
                }
                let chunk = serde_json::json!({
                    "type": "chunk",
                    "content": delta,
                });
                if relay_sender
                    .lock()
                    .await
                    .send(Message::Text(chunk.to_string().into()))
                    .await
                    .is_err()
                {
                    break; // client disconnected
                }
            }
        });

        // --- CN-001: create a per-turn cancellation token as a child of conn_token ---
        // When conn_token is cancelled (client disconnect / pong timeout), the
        // child token is automatically cancelled, stopping the agent turn.
        let this_turn_token = conn_token.child_token();

        // Multi-turn agent loop with streaming: history persists across messages.
        match ctx
            .process_turn_streaming(
                &mut history,
                &content,
                Some(delta_tx),
                Some(this_turn_token),
            )
            .await
        {
            Ok(response) => {
                // Wait for relay task to finish flushing remaining chunks
                let _ = relay_task.await;

                let done = serde_json::json!({
                    "type": "done",
                    "full_response": response,
                });
                let _ = ws_sender
                    .lock()
                    .await
                    .send(Message::Text(done.to_string().into()))
                    .await;

                state
                    .observer
                    .record_event(&crate::observability::ObserverEvent::AgentEnd {
                        provider: provider_label.to_string(),
                        model: state.model.clone(),
                        duration: turn_start.elapsed(),
                        tokens_used: None,
                        cost_usd: None,
                    });
            }
            Err(e) => {
                // Abort relay on error
                relay_task.abort();

                let sanitized = crate::providers::sanitize_api_error(&e.to_string());
                let err = serde_json::json!({
                    "type": "error",
                    "message": sanitized,
                });
                let _ = ws_sender
                    .lock()
                    .await
                    .send(Message::Text(err.to_string().into()))
                    .await;

                state
                    .observer
                    .record_event(&crate::observability::ObserverEvent::Error {
                        component: "ws_chat".to_string(),
                        message: sanitized.to_string(),
                    });
            }
        }
    }

    // Clean up: cancel the ping task (idempotent if already done).
    conn_token.cancel();
    ping_task.abort();
}

/// RAII guard that decrements the WS connection counter when dropped.
struct WsConnectionGuard(Arc<std::sync::atomic::AtomicUsize>);

impl Drop for WsConnectionGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Release);
    }
}
