//! Redis-backed event bus for cross-process event delivery.
//!
//! `RedisEventBus` publishes events to a Redis pub/sub channel and subscribes
//! to it via a background task. Incoming Redis messages are relayed into a local
//! `tokio::sync::broadcast` channel so SSE consumers work identically to the
//! in-process `TokioBroadcastBus`.

use super::event_bus::EventBus;
use serde_json::Value;
use tokio::sync::broadcast;

/// Event bus that relays events through Redis pub/sub.
///
/// - `publish()` sends to the local broadcast channel **and** issues a Redis
///   `PUBLISH` command (fire-and-forget via `tokio::spawn`).
/// - `subscribe()` returns a receiver from the local broadcast channel.
/// - A background task subscribes to the Redis channel and relays messages into
///   the local broadcast, enabling cross-process fan-out.
pub struct RedisEventBus {
    local_tx: broadcast::Sender<Value>,
    conn: redis::aio::ConnectionManager,
    channel: String,
}

impl RedisEventBus {
    /// Create a new `RedisEventBus` and spawn the background subscriber task.
    ///
    /// `redis_url` is used to open a dedicated pub/sub connection (separate from
    /// the `ConnectionManager` used for publishing).
    pub async fn new(
        conn: redis::aio::ConnectionManager,
        redis_url: &str,
        channel: String,
        capacity: usize,
    ) -> anyhow::Result<Self> {
        let (local_tx, _) = broadcast::channel(capacity);

        // Spawn background subscriber: opens its own connection for SUBSCRIBE.
        let relay_tx = local_tx.clone();
        let sub_channel = channel.clone();
        let client = redis::Client::open(redis_url)?;
        tokio::spawn(async move {
            if let Err(e) = run_subscriber(client, &sub_channel, relay_tx).await {
                tracing::error!("Redis event bus subscriber exited: {e}");
            }
        });

        Ok(Self {
            local_tx,
            conn,
            channel,
        })
    }
}

/// Background loop: subscribes to the Redis channel and relays messages into the
/// local broadcast sender.
async fn run_subscriber(
    client: redis::Client,
    channel: &str,
    relay_tx: broadcast::Sender<Value>,
) -> anyhow::Result<()> {
    use futures_util::StreamExt;

    let mut pubsub = client.get_async_pubsub().await?;
    pubsub.subscribe(channel).await?;

    let mut stream = pubsub.on_message();
    while let Some(msg) = stream.next().await {
        let payload: String = match msg.get_payload() {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("Redis event bus: invalid payload: {e}");
                continue;
            }
        };

        match serde_json::from_str::<Value>(&payload) {
            Ok(value) => {
                let _ = relay_tx.send(value);
            }
            Err(e) => {
                tracing::warn!("Redis event bus: invalid JSON in message: {e}");
            }
        }
    }

    Ok(())
}

impl EventBus for RedisEventBus {
    fn publish(&self, event: Value) {
        // Local delivery (immediate, for same-process SSE subscribers).
        let _ = self.local_tx.send(event.clone());

        // Async Redis PUBLISH for cross-process delivery.
        let mut conn = self.conn.clone();
        let channel = self.channel.clone();
        tokio::spawn(async move {
            let payload = event.to_string();
            let result: Result<(), redis::RedisError> = redis::cmd("PUBLISH")
                .arg(&channel)
                .arg(&payload)
                .query_async(&mut conn)
                .await;
            if let Err(e) = result {
                tracing::warn!("Redis PUBLISH failed: {e}");
            }
        });
    }

    fn subscribe(&self) -> broadcast::Receiver<Value> {
        self.local_tx.subscribe()
    }
}
