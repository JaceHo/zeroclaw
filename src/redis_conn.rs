//! Shared Redis connection manager.
//!
//! Creates a `redis::aio::ConnectionManager` from `RedisConfig`. The manager
//! auto-reconnects and is `Clone + Send + Sync`, suitable for sharing across
//! tasks without a separate pool.

use crate::config::schema::RedisConfig;
use anyhow::{Context, Result};

/// Connection timeout for the initial Redis connection attempt.
const CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Create a Redis `ConnectionManager` from config.
///
/// The connection manager multiplexes commands over a single TCP connection
/// and transparently reconnects on transient failures.
///
/// The initial connection is bounded by a 10-second timeout to prevent the
/// application from hanging if Redis is unreachable at startup.
pub async fn create_connection_manager(
    config: &RedisConfig,
) -> Result<redis::aio::ConnectionManager> {
    let client =
        redis::Client::open(config.url.as_str()).context("invalid Redis URL in [redis].url")?;

    let manager = tokio::time::timeout(CONNECT_TIMEOUT, redis::aio::ConnectionManager::new(client))
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "Redis connection timed out after {}s — is Redis running at {}?",
                CONNECT_TIMEOUT.as_secs(),
                config.url
            )
        })?
        .context("failed to connect to Redis")?;

    // Verify connectivity with a PING
    let mut test_conn = manager.clone();
    let pong: String = redis::cmd("PING")
        .query_async(&mut test_conn)
        .await
        .context("Redis PING failed — connection established but server not responding")?;
    if pong != "PONG" {
        anyhow::bail!("Redis PING returned unexpected response: {pong}");
    }

    Ok(manager)
}
