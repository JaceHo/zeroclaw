//! Redis-backed response cache.
//!
//! Uses Redis `SET ... EX ttl` for automatic expiration (no manual eviction needed)
//! and atomic counters for hit/token-saved statistics.

use super::response_cache::ResponseCacheBackend;
use anyhow::{Context, Result};
use redis::AsyncCommands;

/// Response cache backed by Redis.
///
/// Each cached response is stored as a key with a Redis-native TTL. Stats are
/// tracked via atomic `INCR`/`INCRBY` counters.
pub struct RedisResponseCache {
    conn: redis::aio::ConnectionManager,
    prefix: String,
    ttl_secs: u64,
}

impl RedisResponseCache {
    /// Create a new Redis response cache.
    pub fn new(conn: redis::aio::ConnectionManager, prefix: &str, ttl_minutes: u32) -> Self {
        Self {
            conn,
            prefix: prefix.to_string(),
            ttl_secs: u64::from(ttl_minutes) * 60,
        }
    }

    fn response_key(&self, key: &str) -> String {
        format!("{}cache:{}", self.prefix, key)
    }

    fn tokens_key(&self, key: &str) -> String {
        format!("{}cache_tokens:{}", self.prefix, key)
    }

    fn stats_hits_key(&self) -> String {
        format!("{}stats:hits", self.prefix)
    }

    fn stats_tokens_key(&self) -> String {
        format!("{}stats:tokens_saved", self.prefix)
    }

    /// Bridge sync trait to async Redis using `block_in_place` + current runtime.
    fn block_on<F, T>(&self, future: F) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>>,
    {
        tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(future))
    }
}

impl ResponseCacheBackend for RedisResponseCache {
    fn get(&self, key: &str) -> Result<Option<String>> {
        let rkey = self.response_key(key);
        let tkey = self.tokens_key(key);
        let hits_key = self.stats_hits_key();
        let tokens_key = self.stats_tokens_key();
        let mut conn = self.conn.clone();

        self.block_on(async {
            let response: Option<String> = conn.get(&rkey).await.context("Redis GET failed")?;

            if response.is_some() {
                // Bump stats atomically
                let token_count: u64 = conn.get(&tkey).await.unwrap_or(0);
                let _: () = conn.incr(&hits_key, 1i64).await.unwrap_or(());
                let safe_count = i64::try_from(token_count).unwrap_or(i64::MAX);
                let _: () = conn.incr(&tokens_key, safe_count).await.unwrap_or(());
            }

            Ok(response)
        })
    }

    fn put(&self, key: &str, _model: &str, response: &str, token_count: u32) -> Result<()> {
        let rkey = self.response_key(key);
        let tkey = self.tokens_key(key);
        let mut conn = self.conn.clone();
        let ttl = self.ttl_secs;

        self.block_on(async {
            // Use a pipeline to set both keys atomically (single round-trip,
            // both keys get identical TTLs from the same server timestamp).
            redis::pipe()
                .cmd("SET")
                .arg(&rkey)
                .arg(response)
                .arg("EX")
                .arg(ttl)
                .cmd("SET")
                .arg(&tkey)
                .arg(token_count)
                .arg("EX")
                .arg(ttl)
                .query_async::<()>(&mut conn)
                .await
                .context("Redis pipeline SET EX failed")?;

            Ok(())
        })
    }

    fn stats(&self) -> Result<(usize, u64, u64)> {
        let hits_key = self.stats_hits_key();
        let tokens_key = self.stats_tokens_key();
        let pattern = format!("{}cache:*", self.prefix);
        let mut conn = self.conn.clone();

        self.block_on(async {
            // Count live cache keys via SCAN instead of relying on an INCR
            // counter that drifts upward as entries expire via TTL.
            let mut live_count: usize = 0;
            let mut cursor: u64 = 0;
            loop {
                let (next_cursor, keys): (u64, Vec<String>) = redis::cmd("SCAN")
                    .arg(cursor)
                    .arg("MATCH")
                    .arg(&pattern)
                    .arg("COUNT")
                    .arg(100)
                    .query_async(&mut conn)
                    .await
                    .context("Redis SCAN failed during stats")?;
                live_count += keys.len();
                cursor = next_cursor;
                if cursor == 0 {
                    break;
                }
            }

            let hits: i64 = conn.get(&hits_key).await.unwrap_or(0);
            let tokens_saved: i64 = conn.get(&tokens_key).await.unwrap_or(0);

            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            Ok((live_count, hits.max(0) as u64, tokens_saved.max(0) as u64))
        })
    }

    fn clear(&self) -> Result<usize> {
        let hits_key = self.stats_hits_key();
        let tokens_key = self.stats_tokens_key();
        let mut conn = self.conn.clone();

        self.block_on(async {
            // Collect matching keys via SCAN (non-blocking, unlike KEYS)
            let mut all_keys = Vec::new();
            for pattern in [
                format!("{}cache:*", self.prefix),
                format!("{}cache_tokens:*", self.prefix),
            ] {
                let mut cursor: u64 = 0;
                loop {
                    let (next_cursor, keys): (u64, Vec<String>) = redis::cmd("SCAN")
                        .arg(cursor)
                        .arg("MATCH")
                        .arg(&pattern)
                        .arg("COUNT")
                        .arg(100)
                        .query_async(&mut conn)
                        .await
                        .context("Redis SCAN failed during clear")?;
                    all_keys.extend(keys);
                    cursor = next_cursor;
                    if cursor == 0 {
                        break;
                    }
                }
            }

            // Count only cache entries (not token companion keys)
            let count = all_keys
                .iter()
                .filter(|k| k.starts_with(&format!("{}cache:", self.prefix)))
                .count();

            // Also delete stats counters
            all_keys.push(hits_key);
            all_keys.push(tokens_key);

            if !all_keys.is_empty() {
                redis::cmd("DEL")
                    .arg(&all_keys)
                    .query_async::<()>(&mut conn)
                    .await
                    .context("Redis DEL failed during clear")?;
            }

            Ok(count)
        })
    }
}
