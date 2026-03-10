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

    fn stats_entries_key(&self) -> String {
        format!("{}stats:entries", self.prefix)
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
                let _: () = conn
                    .incr(&tokens_key, token_count as i64)
                    .await
                    .unwrap_or(());
            }

            Ok(response)
        })
    }

    fn put(&self, key: &str, _model: &str, response: &str, token_count: u32) -> Result<()> {
        let rkey = self.response_key(key);
        let tkey = self.tokens_key(key);
        let entries_key = self.stats_entries_key();
        let mut conn = self.conn.clone();
        let ttl = self.ttl_secs;

        self.block_on(async {
            // Check if key already exists (to track net new entries)
            let exists: bool = conn.exists(&rkey).await.unwrap_or(false);

            // Store response with TTL
            let _: () = conn
                .set_ex(&rkey, response, ttl)
                .await
                .context("Redis SET EX failed")?;

            // Store token count with same TTL
            let _: () = conn
                .set_ex(&tkey, token_count, ttl)
                .await
                .context("Redis SET EX (tokens) failed")?;

            // Track total entries (only increment for new keys)
            if !exists {
                let _: () = conn.incr(&entries_key, 1i64).await.unwrap_or(());
            }

            Ok(())
        })
    }

    fn stats(&self) -> Result<(usize, u64, u64)> {
        let entries_key = self.stats_entries_key();
        let hits_key = self.stats_hits_key();
        let tokens_key = self.stats_tokens_key();
        let mut conn = self.conn.clone();

        self.block_on(async {
            let entries: i64 = conn.get(&entries_key).await.unwrap_or(0);
            let hits: i64 = conn.get(&hits_key).await.unwrap_or(0);
            let tokens_saved: i64 = conn.get(&tokens_key).await.unwrap_or(0);

            #[allow(clippy::cast_sign_loss)]
            Ok((
                entries.max(0) as usize,
                hits.max(0) as u64,
                tokens_saved.max(0) as u64,
            ))
        })
    }

    fn clear(&self) -> Result<usize> {
        let pattern = format!("{}cache:*", self.prefix);
        let tokens_pattern = format!("{}cache_tokens:*", self.prefix);
        let entries_key = self.stats_entries_key();
        let hits_key = self.stats_hits_key();
        let tokens_key = self.stats_tokens_key();
        let mut conn = self.conn.clone();

        self.block_on(async {
            // Collect matching keys via KEYS (acceptable for rare admin ops)
            let cache_keys: Vec<String> = redis::cmd("KEYS")
                .arg(&pattern)
                .query_async(&mut conn)
                .await
                .unwrap_or_default();
            let token_keys: Vec<String> = redis::cmd("KEYS")
                .arg(&tokens_pattern)
                .query_async(&mut conn)
                .await
                .unwrap_or_default();

            let count = cache_keys.len();

            // Delete all cache and token keys
            let mut all_keys = cache_keys;
            all_keys.extend(token_keys);
            all_keys.push(entries_key);
            all_keys.push(hits_key);
            all_keys.push(tokens_key);

            if !all_keys.is_empty() {
                let _: () = redis::cmd("DEL")
                    .arg(&all_keys)
                    .query_async(&mut conn)
                    .await
                    .unwrap_or(());
            }

            Ok(count)
        })
    }
}
