//! Shared Redis connection manager.
//!
//! Creates a `redis::aio::ConnectionManager` from `RedisConfig`. The manager
//! auto-reconnects and is `Clone + Send + Sync`, suitable for sharing across
//! tasks without a separate pool.

use crate::config::schema::RedisConfig;
use anyhow::{Context, Result};

/// Create a Redis `ConnectionManager` from config.
///
/// The connection manager multiplexes commands over a single TCP connection
/// and transparently reconnects on transient failures.
pub async fn create_connection_manager(
    config: &RedisConfig,
) -> Result<redis::aio::ConnectionManager> {
    let client =
        redis::Client::open(config.url.as_str()).context("invalid Redis URL in [redis].url")?;
    let manager = redis::aio::ConnectionManager::new(client)
        .await
        .context("failed to connect to Redis")?;
    Ok(manager)
}
