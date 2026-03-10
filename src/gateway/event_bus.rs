//! Event bus abstraction for real-time event delivery.
//!
//! The `EventBus` trait decouples event producers (webhook handlers, WS chat)
//! from the transport layer. The default `TokioBroadcastBus` wraps a
//! `tokio::sync::broadcast` channel — identical behavior to the previous
//! hard-coded implementation.
//!
//! When compiled with `--features redis`, `RedisEventBus` can relay events
//! across processes via Redis pub/sub while still using a local broadcast
//! channel for SSE subscriber delivery.

use serde_json::Value;
use tokio::sync::broadcast;

/// Abstraction over the SSE/WS event broadcast mechanism.
///
/// Both `publish` and `subscribe` are intentionally synchronous-looking:
/// - `publish` is fire-and-forget (async transports spawn internally).
/// - `subscribe` returns a `broadcast::Receiver` for `BroadcastStream` compatibility.
pub trait EventBus: Send + Sync + 'static {
    /// Publish an event to all subscribers (local and, if configured, remote).
    fn publish(&self, event: Value);

    /// Create a new subscriber receiver. Each call returns an independent stream.
    fn subscribe(&self) -> broadcast::Receiver<Value>;
}

/// In-process event bus backed by `tokio::sync::broadcast`.
///
/// This is the default when Redis is not configured. Behavior is identical
/// to the previous `broadcast::Sender<Value>` usage.
pub struct TokioBroadcastBus {
    tx: broadcast::Sender<Value>,
}

impl TokioBroadcastBus {
    /// Create a new broadcast bus with the given channel capacity.
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }
}

impl EventBus for TokioBroadcastBus {
    fn publish(&self, event: Value) {
        // Ignore send errors (no active receivers is fine).
        let _ = self.tx.send(event);
    }

    fn subscribe(&self) -> broadcast::Receiver<Value> {
        self.tx.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokio_broadcast_bus_publish_subscribe() {
        let bus = TokioBroadcastBus::new(16);
        let mut rx = bus.subscribe();

        let event = serde_json::json!({"type": "test", "value": 42});
        bus.publish(event.clone());

        let received = rx.try_recv().unwrap();
        assert_eq!(received, event);
    }

    #[test]
    fn tokio_broadcast_bus_multiple_subscribers() {
        let bus = TokioBroadcastBus::new(16);
        let mut rx1 = bus.subscribe();
        let mut rx2 = bus.subscribe();

        let event = serde_json::json!({"type": "fanout"});
        bus.publish(event.clone());

        assert_eq!(rx1.try_recv().unwrap(), event);
        assert_eq!(rx2.try_recv().unwrap(), event);
    }

    #[test]
    fn tokio_broadcast_bus_publish_without_subscribers_does_not_panic() {
        let bus = TokioBroadcastBus::new(16);
        // No subscribers — should not panic
        bus.publish(serde_json::json!({"type": "orphan"}));
    }
}
