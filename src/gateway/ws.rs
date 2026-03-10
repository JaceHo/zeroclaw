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
use std::sync::Arc;
use tokio::sync::Mutex;

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

    ws.on_upgrade(move |socket| handle_socket(socket, state))
        .into_response()
}

async fn handle_socket(socket: WebSocket, state: AppState) {
    let (ws_sender, mut receiver) = socket.split();
    let ws_sender = Arc::new(Mutex::new(ws_sender));

    // Build session context once on connect — provider, tools, memory, etc.
    // are reused across all turns in this WebSocket connection.
    let config = state.config.lock().clone();
    let ctx = match crate::agent::AgentSessionContext::from_config(&config).await {
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
            return;
        }
    };

    // Persistent conversation history for the lifetime of this connection.
    let mut history = ctx.initial_history();

    while let Some(msg) = receiver.next().await {
        let msg = match msg {
            Ok(Message::Text(text)) => text,
            Ok(Message::Close(_)) => break,
            Err(_) => break,
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

        let provider_label = &ctx.provider_name;

        // Broadcast agent_start event via observer (adds timestamp + metrics)
        state.observer.record_event(
            &crate::observability::ObserverEvent::AgentStart {
                provider: provider_label.to_string(),
                model: state.model.clone(),
            },
        );

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

        // Multi-turn agent loop with streaming: history persists across messages.
        match ctx
            .process_turn_streaming(&mut history, &content, Some(delta_tx))
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

                state.observer.record_event(
                    &crate::observability::ObserverEvent::AgentEnd {
                        provider: provider_label.to_string(),
                        model: state.model.clone(),
                        duration: std::time::Duration::ZERO,
                        tokens_used: None,
                        cost_usd: None,
                    },
                );
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

                state.observer.record_event(
                    &crate::observability::ObserverEvent::Error {
                        component: "ws_chat".to_string(),
                        message: sanitized.to_string(),
                    },
                );
            }
        }
    }
}
