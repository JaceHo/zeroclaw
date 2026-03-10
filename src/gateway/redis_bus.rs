//! Redis-backed event bus for cross-process event delivery.
//!
//! `RedisEventBus` publishes events to a Redis pub/sub channel and subscribes
//! to it via a background task. Incoming Redis messages are relayed into a local
//! `tokio::sync::broadcast` channel so SSE consumers work identically to the
//! in-process `TokioBroadcastBus`.
//!
//! ## Deduplication
//!
//! Each published event is tagged with a unique `_zcid` field. When Redis
//! PUBLISH fails, the event is delivered locally as a fallback. The background
//! subscriber ignores messages whose `_zcid` was already delivered locally,
//! preventing duplicate delivery to same-process consumers.

use super::event_bus::EventBus;
use serde_json::Value;
use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;
use tokio::task::JoinHandle;

/// Maximum number of in-flight Redis PUBLISH tasks.
const MAX_INFLIGHT_PUBLISHES: usize = 256;

/// Maximum dedup IDs retained (ring-buffer style eviction).
const MAX_DEDUP_IDS: usize = 4096;

/// Event bus that relays events through Redis pub/sub.
///
/// - `publish()` sends to Redis via a bounded task pool (capped at
///   [`MAX_INFLIGHT_PUBLISHES`]). If Redis is unreachable, events are
///   delivered locally with dedup protection.
/// - `subscribe()` returns a receiver from the local broadcast channel.
/// - A background task subscribes to the Redis channel and relays messages into
///   the local broadcast, enabling cross-process fan-out.
pub struct RedisEventBus {
    local_tx: broadcast::Sender<Value>,
    conn: redis::aio::ConnectionManager,
    channel: String,
    publish_semaphore: Arc<tokio::sync::Semaphore>,
    /// IDs of events that were delivered locally via fallback, so the subscriber
    /// can skip them when they also arrive from Redis.
    local_dedup: Arc<Mutex<DedupRing>>,
    /// Handle to the background subscriber task for graceful shutdown.
    _subscriber_handle: JoinHandle<()>,
}

/// Fixed-capacity ring buffer of event IDs for deduplication.
struct DedupRing {
    ids: HashSet<String>,
    order: VecDeque<String>,
}

impl DedupRing {
    fn new() -> Self {
        Self {
            ids: HashSet::with_capacity(MAX_DEDUP_IDS),
            order: VecDeque::with_capacity(MAX_DEDUP_IDS),
        }
    }

    /// Insert an ID. Returns `true` if the ID was new (not a duplicate).
    fn insert(&mut self, id: String) -> bool {
        if !self.ids.insert(id.clone()) {
            return false;
        }
        if self.order.len() >= MAX_DEDUP_IDS {
            // Evict oldest — O(1) with VecDeque vs O(N) with Vec
            if let Some(evicted) = self.order.pop_front() {
                self.ids.remove(&evicted);
            }
        }
        self.order.push_back(id);
        true
    }

    /// Check whether the ID is known (was locally delivered).
    fn contains(&self, id: &str) -> bool {
        self.ids.contains(id)
    }

    /// Remove an ID after the subscriber has seen it (free the slot).
    fn remove(&mut self, id: &str) {
        self.ids.remove(id);
        // order is not compacted — tolerable for bounded ring.
    }
}

impl RedisEventBus {
    /// Create a new `RedisEventBus` and spawn the background subscriber task.
    ///
    /// `redis_url` is used to open a dedicated pub/sub connection (separate from
    /// the `ConnectionManager` used for publishing).
    #[allow(clippy::unused_async)]
    pub async fn new(
        conn: redis::aio::ConnectionManager,
        redis_url: &str,
        channel: String,
        capacity: usize,
    ) -> anyhow::Result<Self> {
        let (local_tx, _) = broadcast::channel(capacity);
        let local_dedup = Arc::new(Mutex::new(DedupRing::new()));

        // Spawn background subscriber with reconnection.
        let relay_tx = local_tx.clone();
        let sub_channel = channel.clone();
        let sub_dedup = Arc::clone(&local_dedup);
        let client = redis::Client::open(redis_url)?;
        let subscriber_handle = tokio::spawn(async move {
            let mut backoff_secs = 1u64;
            const MAX_BACKOFF_SECS: u64 = 30;
            loop {
                match run_subscriber(
                    client.clone(),
                    &sub_channel,
                    relay_tx.clone(),
                    Arc::clone(&sub_dedup),
                )
                .await
                {
                    Ok(()) => {
                        // Clean stream end — connection was healthy. Reset backoff.
                        backoff_secs = 1;
                        tracing::warn!("Redis event bus subscriber stream ended, reconnecting...");
                    }
                    Err(e) => {
                        tracing::error!(
                            "Redis event bus subscriber failed: {e}, retrying in {backoff_secs}s"
                        );
                        // Exponential backoff only on errors; reset path keeps backoff at 1s.
                        backoff_secs = (backoff_secs * 2).min(MAX_BACKOFF_SECS);
                    }
                }
                tokio::time::sleep(std::time::Duration::from_secs(backoff_secs)).await;
            }
        });

        Ok(Self {
            local_tx,
            conn,
            channel,
            publish_semaphore: Arc::new(tokio::sync::Semaphore::new(MAX_INFLIGHT_PUBLISHES)),
            local_dedup,
            _subscriber_handle: subscriber_handle,
        })
    }
}

impl Drop for RedisEventBus {
    fn drop(&mut self) {
        self._subscriber_handle.abort();
    }
}

/// Background loop: subscribes to the Redis channel and relays messages into the
/// local broadcast sender. Skips messages whose `_zcid` was already delivered
/// locally (dedup).
async fn run_subscriber(
    client: redis::Client,
    channel: &str,
    relay_tx: broadcast::Sender<Value>,
    dedup: Arc<Mutex<DedupRing>>,
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
                // Dedup: if this event was already delivered locally via
                // fallback, skip it to avoid duplicate delivery.
                if let Some(zcid) = value.get("_zcid").and_then(Value::as_str) {
                    let mut ring = dedup.lock().unwrap_or_else(|e| e.into_inner());
                    if ring.contains(zcid) {
                        ring.remove(zcid);
                        continue;
                    }
                }
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
        // Publish to Redis — the background `run_subscriber` task relays
        // messages back into `local_tx`, providing both local and cross-process
        // delivery through a single path (avoids duplicate events for
        // same-process SSE subscribers).
        //
        // If Redis is unreachable, fall back to local delivery and record the
        // event ID in `local_dedup` so the subscriber skips it if it eventually
        // arrives from Redis.
        let mut conn = self.conn.clone();
        let channel = self.channel.clone();
        let local_tx = self.local_tx.clone();
        let semaphore = Arc::clone(&self.publish_semaphore);
        let dedup = Arc::clone(&self.local_dedup);

        // Tag event with a unique ID for dedup
        let zcid = uuid::Uuid::new_v4().to_string();
        let mut tagged = event;
        if let Some(obj) = tagged.as_object_mut() {
            obj.insert("_zcid".to_string(), Value::String(zcid.clone()));
        }

        tokio::spawn(async move {
            // Acquire semaphore permit to bound in-flight publishes
            let _permit = match semaphore.try_acquire() {
                Ok(permit) => permit,
                Err(_) => {
                    tracing::warn!(
                        "Redis PUBLISH backpressure: too many in-flight, delivering locally"
                    );
                    let mut ring = dedup.lock().unwrap_or_else(|e| e.into_inner());
                    ring.insert(zcid);
                    let _ = local_tx.send(tagged);
                    return;
                }
            };

            let payload = tagged.to_string();
            let mut cmd = redis::cmd("PUBLISH");
            cmd.arg(&channel).arg(&payload);
            let publish_fut = cmd.query_async::<()>(&mut conn);
            match tokio::time::timeout(std::time::Duration::from_secs(5), publish_fut).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    tracing::warn!("Redis PUBLISH failed, falling back to local delivery: {e}");
                    let mut ring = dedup.lock().unwrap_or_else(|e| e.into_inner());
                    ring.insert(zcid);
                    let _ = local_tx.send(tagged);
                }
                Err(_) => {
                    tracing::warn!("Redis PUBLISH timed out, falling back to local delivery");
                    let mut ring = dedup.lock().unwrap_or_else(|e| e.into_inner());
                    ring.insert(zcid);
                    let _ = local_tx.send(tagged);
                }
            }
        });
    }

    fn subscribe(&self) -> broadcast::Receiver<Value> {
        self.local_tx.subscribe()
    }
}
