//! Statistics off the render thread: one worker thread, the latest request wins, the
//! previous one is cancelled through its flag, results of a stale generation are dropped
//! by `update`, and an LRU cache keyed by (layer, projection) avoids recomputing.

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use super::TuiError;
use crate::engine::stats::{self, TensorStats};
use crate::engine::ReadOnlyModel;

/// Default cache size: 64 (layer, projection) pairs.
pub const CACHE_CAPACITY: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: u64,
}

pub trait StatsProvider: Send + Sync {
    fn list_tensors(&self) -> Vec<TensorEntry>;
    fn tensor_stats(
        &self,
        tensor_name: &str,
        bins: usize,
        cancel: &AtomicBool,
    ) -> Result<TensorStats, TuiError>;
}

/// Every shard of a model, memory-mapped by the engine (name -> shard map).
pub struct ModelStatsProvider {
    shards: Vec<ReadOnlyModel>,
    owner: HashMap<String, usize>,
}

impl ModelStatsProvider {
    pub fn open(path: &Path) -> Result<Self, TuiError> {
        let model_error = |message: String| TuiError::Model {
            path: path.to_path_buf(),
            message,
        };
        let files = crate::hardware::compute_model_size(path)
            .map_err(|e| model_error(e.to_string()))?
            .shard_files;
        let mut shards = Vec::with_capacity(files.len());
        let mut owner = HashMap::new();
        for (i, file) in files.iter().enumerate() {
            let shard = ReadOnlyModel::open(file).map_err(|e| model_error(e.to_string()))?;
            for info in &shard.index().tensors {
                owner.entry(info.name.clone()).or_insert(i);
            }
            shards.push(shard);
        }
        Ok(Self { shards, owner })
    }
}

impl StatsProvider for ModelStatsProvider {
    fn list_tensors(&self) -> Vec<TensorEntry> {
        let mut entries: Vec<TensorEntry> = self
            .shards
            .iter()
            .flat_map(|s| s.index().tensors.iter())
            .map(|t| TensorEntry {
                name: t.name.clone(),
                dtype: t.dtype.to_string(),
                shape: t.shape.clone(),
                bytes: (t.byte_end - t.byte_start) as u64,
            })
            .collect();
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        entries.dedup_by(|a, b| a.name == b.name);
        entries
    }

    fn tensor_stats(
        &self,
        tensor_name: &str,
        bins: usize,
        cancel: &AtomicBool,
    ) -> Result<TensorStats, TuiError> {
        let missing = || TuiError::Stats(format!("tenseur introuvable : {tensor_name}"));
        let shard = self
            .shards
            .get(*self.owner.get(tensor_name).ok_or_else(missing)?)
            .ok_or_else(missing)?;
        let view = shard
            .tensor(tensor_name)
            .map_err(|e| TuiError::Stats(e.to_string()))?;
        stats::tensor_stats(view.bytes, &view.info.dtype, &view.info.shape, bins, cancel)
            .map_err(|e| TuiError::Stats(e.to_string()))
    }
}

/// Bounded least-recently-used map.
pub struct Lru<K, V> {
    capacity: usize,
    map: HashMap<K, V>,
    order: VecDeque<K>,
}

impl<K: std::hash::Hash + Eq + Clone, V: Clone> Lru<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            map: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    pub fn get(&mut self, key: &K) -> Option<V> {
        let value = self.map.get(key)?.clone();
        self.order.retain(|k| k != key);
        self.order.push_back(key.clone());
        Some(value)
    }

    pub fn put(&mut self, key: K, value: V) {
        self.order.retain(|k| k != &key);
        self.order.push_back(key.clone());
        self.map.insert(key, value);
        while self.order.len() > self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.map.remove(&oldest);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// (layer, projection index).
pub type StatsKey = (usize, usize);

#[derive(Debug, Clone)]
pub struct StatsRequest {
    pub generation: u64,
    pub key: StatsKey,
    pub tensor: String,
    pub bins: usize,
}

#[derive(Debug, Clone)]
pub struct StatsReply {
    pub generation: u64,
    pub key: StatsKey,
    pub result: Result<TensorStats, String>,
}

struct Job {
    request: StatsRequest,
    cancel: Arc<AtomicBool>,
}

pub struct Worker {
    jobs: Option<Sender<Job>>,
    replies: Receiver<StatsReply>,
    current: Mutex<Option<Arc<AtomicBool>>>,
    handle: Option<JoinHandle<()>>,
}

impl Worker {
    pub fn spawn(provider: Arc<dyn StatsProvider>, capacity: usize) -> Self {
        let (jobs, inbox) = mpsc::channel::<Job>();
        let (outbox, replies) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            let mut cache: Lru<StatsKey, TensorStats> = Lru::new(capacity);
            while let Ok(mut job) = inbox.recv() {
                // Only the latest request matters: older queued ones are already stale.
                while let Ok(newer) = inbox.try_recv() {
                    job = newer;
                }
                let StatsRequest {
                    generation,
                    key,
                    tensor,
                    bins,
                } = job.request;
                let result = match cache.get(&key) {
                    Some(hit) => Ok(hit),
                    None => provider
                        .tensor_stats(&tensor, bins, &job.cancel)
                        .map_err(|e| e.to_string()),
                };
                if job.cancel.load(Ordering::Relaxed) {
                    continue;
                }
                if let Ok(stats) = &result {
                    cache.put(key, stats.clone());
                }
                if outbox
                    .send(StatsReply {
                        generation,
                        key,
                        result,
                    })
                    .is_err()
                {
                    return;
                }
            }
        });
        Self {
            jobs: Some(jobs),
            replies,
            current: Mutex::new(None),
            handle: Some(handle),
        }
    }

    /// Queues `request` and cancels the computation in progress, if any.
    pub fn request(&self, request: StatsRequest) {
        let cancel = Arc::new(AtomicBool::new(false));
        if let Ok(mut current) = self.current.lock() {
            if let Some(previous) = current.replace(Arc::clone(&cancel)) {
                previous.store(true, Ordering::Relaxed);
            }
        }
        if let Some(jobs) = &self.jobs {
            let _ = jobs.send(Job { request, cancel });
        }
    }

    pub fn try_recv(&self) -> Option<StatsReply> {
        self.replies.try_recv().ok()
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        if let Ok(current) = self.current.lock() {
            if let Some(flag) = current.as_ref() {
                flag.store(true, Ordering::Relaxed);
            }
        }
        self.jobs = None;
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lru_evicts_the_least_recent() {
        let mut lru = Lru::new(2);
        lru.put(1, "a");
        lru.put(2, "b");
        assert_eq!(lru.get(&1), Some("a"));
        lru.put(3, "c");
        assert_eq!(
            (lru.get(&2), lru.get(&1), lru.get(&3), lru.len()),
            (None, Some("a"), Some("c"), 2)
        );
        assert!(!lru.is_empty());
    }
}
