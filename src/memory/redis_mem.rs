//! Redis-backed memory with HNSW vector search via Redis 8 VectorSets.
//!
//! Replaces SQLite's O(N) full-table vector scan with O(log N) approximate
//! nearest-neighbor search using native Redis VectorSet commands (VADD/VSIM).
//!
//! ## Data Model
//!
//! - **VectorSet** `{prefix}mem:vs` — HNSW index. Element = memory ID, vector =
//!   embedding, attributes = metadata JSON (key, category, session_id, timestamp).
//! - **Hash** `{prefix}mem:data` — field = memory ID, value = full entry JSON
//!   (including content, which may be too large for vectorset attributes).
//! - **Hash** `{prefix}mem:keys` — field = memory key, value = memory ID (O(1)
//!   key→ID lookup for `get` and upsert).
//!
//! ## Requirements
//!
//! - Redis 8+ with VectorSet support.
//! - A real embedding provider (not `NoopEmbedding`). Without embeddings, `recall`
//!   returns empty results — Redis memory's value proposition is HNSW search.

use super::embeddings::EmbeddingProvider;
use super::traits::{Memory, MemoryCategory, MemoryEntry};
use super::vector;
use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::Local;
use redis::AsyncCommands;
use std::sync::Arc;
use uuid::Uuid;

/// Redis-backed memory with HNSW vector search.
pub struct RedisMemory {
    conn: redis::aio::ConnectionManager,
    prefix: String,
    embedder: Arc<dyn EmbeddingProvider>,
}

impl RedisMemory {
    /// Create a new Redis memory backend.
    pub fn new(
        conn: redis::aio::ConnectionManager,
        prefix: &str,
        embedder: Arc<dyn EmbeddingProvider>,
    ) -> Self {
        Self {
            conn,
            prefix: prefix.to_string(),
            embedder,
        }
    }

    /// VectorSet key for HNSW index.
    fn vs_key(&self) -> String {
        format!("{}mem:vs", self.prefix)
    }

    /// Hash key for full entry data (ID → JSON).
    fn data_key(&self) -> String {
        format!("{}mem:data", self.prefix)
    }

    /// Hash key for key→ID mapping.
    fn keys_key(&self) -> String {
        format!("{}mem:keys", self.prefix)
    }

    fn category_to_str(cat: &MemoryCategory) -> &str {
        match cat {
            MemoryCategory::Core => "core",
            MemoryCategory::Daily => "daily",
            MemoryCategory::Conversation => "conversation",
            MemoryCategory::Custom(name) => name,
        }
    }

    fn str_to_category(s: &str) -> MemoryCategory {
        match s {
            "core" => MemoryCategory::Core,
            "daily" => MemoryCategory::Daily,
            "conversation" => MemoryCategory::Conversation,
            other => MemoryCategory::Custom(other.to_string()),
        }
    }

    /// Parse a memory entry from its stored JSON representation.
    fn parse_entry(id: &str, json: &str) -> Result<MemoryEntry> {
        let v: serde_json::Value =
            serde_json::from_str(json).context("failed to parse memory entry JSON")?;
        Ok(MemoryEntry {
            id: id.to_string(),
            key: v["key"].as_str().unwrap_or("").to_string(),
            content: v["content"].as_str().unwrap_or("").to_string(),
            category: Self::str_to_category(v["category"].as_str().unwrap_or("core")),
            timestamp: v["timestamp"].as_str().unwrap_or("").to_string(),
            session_id: v["session_id"].as_str().map(String::from),
            score: None,
        })
    }
}

#[async_trait]
impl Memory for RedisMemory {
    fn name(&self) -> &str {
        "redis"
    }

    async fn store(
        &self,
        key: &str,
        content: &str,
        category: MemoryCategory,
        session_id: Option<&str>,
    ) -> Result<()> {
        let mut conn = self.conn.clone();
        let now = Local::now().to_rfc3339();
        let cat = Self::category_to_str(&category).to_string();

        let keys_key = self.keys_key();
        let vs_key = self.vs_key();
        let data_key = self.data_key();

        // Check if key already exists (upsert semantics)
        let existing_id: Option<String> = conn.hget(&keys_key, key).await.unwrap_or(None);

        let id = if let Some(ref eid) = existing_id {
            // Remove old vector entry before re-adding
            let _: Result<(), redis::RedisError> = redis::cmd("VREM")
                .arg(&vs_key)
                .arg(eid.as_str())
                .query_async(&mut conn)
                .await;
            eid.clone()
        } else {
            Uuid::new_v4().to_string()
        };

        // Compute embedding
        let embedding = if self.embedder.dimensions() > 0 {
            self.embedder.embed_one(content).await.ok()
        } else {
            None
        };

        // Store full entry as JSON in data hash
        let entry_json = serde_json::json!({
            "key": key,
            "content": content,
            "category": cat,
            "session_id": session_id,
            "timestamp": now,
        });
        let _: () = conn
            .hset(&data_key, &id, entry_json.to_string())
            .await
            .context("Redis HSET data failed")?;

        // Store key→ID mapping
        let _: () = conn
            .hset(&keys_key, key, &id)
            .await
            .context("Redis HSET keys failed")?;

        // Add to vectorset (if we have an embedding)
        if let Some(emb) = embedding {
            let bytes = vector::vec_to_bytes(&emb);
            let attrs = serde_json::json!({
                "key": key,
                "category": cat,
                "session_id": session_id.unwrap_or(""),
                "timestamp": now,
            });

            let result: Result<(), redis::RedisError> = redis::cmd("VADD")
                .arg(&vs_key)
                .arg("FP32")
                .arg(bytes.as_slice())
                .arg(&id)
                .arg("SETATTR")
                .arg(attrs.to_string())
                .query_async(&mut conn)
                .await;

            if let Err(e) = result {
                tracing::warn!("Redis VADD failed (requires Redis 8+ with VectorSet support): {e}");
            }
        }

        Ok(())
    }

    async fn recall(
        &self,
        query: &str,
        limit: usize,
        session_id: Option<&str>,
    ) -> Result<Vec<MemoryEntry>> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let mut conn = self.conn.clone();

        // Compute query embedding
        let query_embedding = if self.embedder.dimensions() > 0 {
            self.embedder.embed_one(query).await.ok()
        } else {
            None
        };

        // Without embeddings, vector search is not possible
        let Some(emb) = query_embedding else {
            return Ok(Vec::new());
        };

        let bytes = vector::vec_to_bytes(&emb);
        let vs_key = self.vs_key();
        let data_key = self.data_key();

        // Build VSIM command with optional session filter
        let mut cmd = redis::cmd("VSIM");
        cmd.arg(&vs_key)
            .arg("FP32")
            .arg(bytes.as_slice())
            .arg("COUNT")
            .arg(limit)
            .arg("WITHSCORES");

        if let Some(sid) = session_id {
            // VSIM FILTER uses JSONPath-style expressions on attributes
            let filter = format!("$.session_id == \"{}\"", sid.replace('"', "\\\""));
            cmd.arg("FILTER").arg(filter);
        }

        // VSIM returns alternating [element, score, element, score, ...]
        let raw: Vec<redis::Value> = cmd.query_async(&mut conn).await.unwrap_or_default();

        let mut result_ids: Vec<(String, f64)> = Vec::new();
        let mut i = 0;
        while i + 1 < raw.len() {
            let id = match &raw[i] {
                redis::Value::BulkString(id_bytes) => String::from_utf8_lossy(id_bytes).to_string(),
                redis::Value::SimpleString(s) => s.clone(),
                _ => {
                    i += 2;
                    continue;
                }
            };
            #[allow(clippy::cast_precision_loss)]
            let score = match &raw[i + 1] {
                redis::Value::BulkString(s) => {
                    String::from_utf8_lossy(s).parse::<f64>().unwrap_or(0.0)
                }
                redis::Value::Double(d) => *d,
                redis::Value::Int(n) => *n as f64,
                _ => 0.0,
            };
            result_ids.push((id, score));
            i += 2;
        }

        // Fetch full entries for the result IDs
        let mut entries = Vec::with_capacity(result_ids.len());
        for (id, score) in &result_ids {
            let json: Option<String> = conn.hget(&data_key, id).await.unwrap_or(None);
            if let Some(json) = json {
                if let Ok(mut entry) = Self::parse_entry(id, &json) {
                    entry.score = Some(*score);
                    entries.push(entry);
                }
            }
        }

        Ok(entries)
    }

    async fn get(&self, key: &str) -> Result<Option<MemoryEntry>> {
        let mut conn = self.conn.clone();
        let keys_key = self.keys_key();
        let data_key = self.data_key();

        // Lookup ID by key
        let id: Option<String> = conn.hget(&keys_key, key).await.unwrap_or(None);
        let Some(id) = id else {
            return Ok(None);
        };

        // Fetch full entry
        let json: Option<String> = conn.hget(&data_key, &id).await.unwrap_or(None);
        match json {
            Some(json) => Ok(Some(Self::parse_entry(&id, &json)?)),
            None => Ok(None),
        }
    }

    async fn list(
        &self,
        category: Option<&MemoryCategory>,
        session_id: Option<&str>,
    ) -> Result<Vec<MemoryEntry>> {
        let mut conn = self.conn.clone();
        let data_key = self.data_key();

        // Get all entries from data hash
        let all: std::collections::HashMap<String, String> =
            conn.hgetall(&data_key).await.unwrap_or_default();

        let mut entries = Vec::new();
        for (id, json) in &all {
            if let Ok(entry) = Self::parse_entry(id, json) {
                // Apply filters
                if let Some(cat) = category {
                    if &entry.category != cat {
                        continue;
                    }
                }
                if let Some(sid) = session_id {
                    if entry.session_id.as_deref() != Some(sid) {
                        continue;
                    }
                }
                entries.push(entry);
            }
        }

        // Sort by timestamp descending (newest first)
        entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        Ok(entries)
    }

    async fn forget(&self, key: &str) -> Result<bool> {
        let mut conn = self.conn.clone();
        let keys_key = self.keys_key();
        let vs_key = self.vs_key();
        let data_key = self.data_key();

        // Lookup ID by key
        let id: Option<String> = conn.hget(&keys_key, key).await.unwrap_or(None);
        let Some(id) = id else {
            return Ok(false);
        };

        // Remove from vectorset
        let _: Result<(), redis::RedisError> = redis::cmd("VREM")
            .arg(&vs_key)
            .arg(&id)
            .query_async(&mut conn)
            .await;

        // Remove from data hash
        let _: () = conn.hdel(&data_key, &id).await.unwrap_or(());

        // Remove key→ID mapping
        let _: () = conn.hdel(&keys_key, key).await.unwrap_or(());

        Ok(true)
    }

    async fn count(&self) -> Result<usize> {
        let mut conn = self.conn.clone();
        let data_key = self.data_key();
        let count: usize = conn.hlen(&data_key).await.unwrap_or(0);
        Ok(count)
    }

    async fn health_check(&self) -> bool {
        let mut conn = self.conn.clone();
        let result: Result<String, redis::RedisError> =
            redis::cmd("PING").query_async(&mut conn).await;
        result.is_ok()
    }
}
