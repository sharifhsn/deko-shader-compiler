use std::{
    collections::{BTreeMap, VecDeque},
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use sha2::{Digest, Sha256};

use crate::{Artifact, Compiler, Error, Options, PipelineConstants, Stage};

/// Version of the stable cache-key encoding.
pub const CACHE_KEY_VERSION: u32 = 1;

/// Version of the native code-generation contract.
///
/// This must be incremented whenever a backend change can alter or invalidate DKSH output without
/// a package-version change.
pub const BACKEND_ABI_VERSION: u32 = 43;

/// Default maximum number of compiled artifacts retained in process memory.
pub const DEFAULT_MEMORY_CACHE_ENTRIES: usize = 256;
/// Default maximum DKSH and reflection bytes retained in process memory.
pub const DEFAULT_MEMORY_CACHE_BYTES: usize = 64 * 1024 * 1024;

const PERSISTENT_CACHE_MAGIC: &[u8; 8] = b"DKSCv001";
const PERSISTENT_CACHE_HEADER_SIZE: usize = 8 + 32 + 8 + 32;
const MAX_PERSISTENT_ARTIFACT_SIZE: usize = 16 * 1024 * 1024;
static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Deterministic identity of one fully specified shader compilation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CacheKey([u8; 32]);

impl CacheKey {
    /// Build a key from all inputs that can affect generated native code.
    #[must_use]
    pub fn new(
        source: &str,
        stage: Stage,
        entry_point: &str,
        constants: &PipelineConstants,
        options: &Options,
    ) -> Self {
        let mut digest = Sha256::new();
        digest.update(b"deko-shader-compiler-cache\0");
        digest.update(CACHE_KEY_VERSION.to_le_bytes());
        digest.update(BACKEND_ABI_VERSION.to_le_bytes());
        put_bytes(&mut digest, env!("CARGO_PKG_VERSION").as_bytes());
        put_bytes(&mut digest, source.as_bytes());
        digest.update([match stage {
            Stage::Vertex => 0,
            Stage::Fragment => 1,
            Stage::Compute => 2,
        }]);
        put_bytes(&mut digest, entry_point.as_bytes());
        digest.update((constants.len() as u64).to_le_bytes());
        for (name, value) in constants {
            put_bytes(&mut digest, name.as_bytes());
            digest.update(value.to_bits().to_le_bytes());
        }
        digest.update([match options.target {
            crate::Target::Gm20b => 0,
        }]);
        digest.update([match options.robustness {
            crate::Robustness::Robust => 0,
            crate::Robustness::PreLowered => 1,
        }]);
        match options.multiview_mask {
            Some(mask) => {
                digest.update([1]);
                digest.update(mask.to_le_bytes());
            }
            None => digest.update([0]),
        }
        digest.update([u8::from(options.zero_initialize_workgroup_memory)]);
        let mut binding_array_sizes = options.binding_array_sizes.clone();
        binding_array_sizes.sort_unstable();
        digest.update((binding_array_sizes.len() as u64).to_le_bytes());
        for size in binding_array_sizes {
            digest.update(size.group.to_le_bytes());
            digest.update(size.binding.to_le_bytes());
            digest.update(size.count.to_le_bytes());
        }
        Self(digest.finalize().into())
    }

    /// Return the raw SHA-256 cache identity.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Return a lowercase hexadecimal filename-safe identity.
    #[must_use]
    pub fn to_hex(self) -> String {
        use std::fmt::Write as _;
        let mut result = String::with_capacity(64);
        for byte in self.0 {
            write!(&mut result, "{byte:02x}").expect("writing to String cannot fail");
        }
        result
    }
}

fn put_bytes(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

/// Thread-safe deterministic in-memory artifact cache.
#[derive(Debug)]
struct CacheState {
    entries: BTreeMap<CacheKey, Arc<Artifact>>,
    least_recently_used: VecDeque<CacheKey>,
    retained_bytes: usize,
}

/// Thread-safe deterministic artifact cache with optional persistent storage.
///
/// Persistent entries are checksummed, structurally validated as DKSH, and written through an
/// atomic rename. Missing, stale, truncated, or corrupt files are treated as cache misses and
/// regenerated; cache I/O never turns a valid shader compilation into an application failure.
#[derive(Clone, Debug)]
pub struct CompilerCache {
    state: Arc<Mutex<CacheState>>,
    max_memory_entries: usize,
    max_memory_bytes: usize,
    persistent_directory: Option<Arc<PathBuf>>,
}

/// Where a successful compiler-cache lookup obtained its artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheSource {
    /// The artifact was already retained in this process.
    Memory,
    /// The artifact was loaded and validated from persistent storage.
    Persistent,
    /// The request was compiled because neither cache contained it.
    Compiled,
}

/// Timing and cache-source information for one successful request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompileTelemetry {
    /// Cache tier that produced the artifact.
    pub source: CacheSource,
    /// Wall-clock time spent resolving the complete request.
    pub elapsed: Duration,
}

impl Default for CompilerCache {
    fn default() -> Self {
        Self::with_memory_limits(DEFAULT_MEMORY_CACHE_ENTRIES, DEFAULT_MEMORY_CACHE_BYTES)
    }
}

impl CompilerCache {
    /// Create a memory cache retaining at most `max_memory_entries` artifacts.
    ///
    /// A limit of zero disables RAM retention while still allowing a persistent cache configured
    /// with [`Self::with_persistent_directory`].
    #[must_use]
    pub fn new(max_memory_entries: usize) -> Self {
        Self::with_memory_limits(max_memory_entries, DEFAULT_MEMORY_CACHE_BYTES)
    }

    /// Create a memory cache with explicit entry and byte limits.
    #[must_use]
    pub fn with_memory_limits(max_memory_entries: usize, max_memory_bytes: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(CacheState {
                entries: BTreeMap::new(),
                least_recently_used: VecDeque::new(),
                retained_bytes: 0,
            })),
            max_memory_entries,
            max_memory_bytes,
            persistent_directory: None,
        }
    }

    /// Store and load validated cache entries below `directory`.
    #[must_use]
    pub fn with_persistent_directory(mut self, directory: impl Into<PathBuf>) -> Self {
        self.persistent_directory = Some(Arc::new(directory.into()));
        self
    }

    /// Return the configured persistent cache directory, if any.
    #[must_use]
    pub fn persistent_directory(&self) -> Option<&Path> {
        self.persistent_directory.as_deref().map(PathBuf::as_path)
    }

    /// Compile WGSL or return the previously compiled artifact for the exact same request.
    ///
    /// # Errors
    ///
    /// Returns the same typed error as [`Compiler::compile_wgsl`]. Failed compilations are not
    /// cached.
    pub fn compile_wgsl(
        &self,
        source: &str,
        stage: Stage,
        entry_point: &str,
        constants: &PipelineConstants,
        options: Options,
    ) -> Result<(CacheKey, Arc<Artifact>), Error> {
        self.compile_wgsl_with_telemetry(source, stage, entry_point, constants, options)
            .map(|(key, artifact, _)| (key, artifact))
    }

    /// Compile WGSL and report whether the artifact came from memory, disk, or code generation.
    ///
    /// # Errors
    ///
    /// Returns the same typed error as [`Compiler::compile_wgsl`]. Failed compilations are not
    /// cached.
    pub fn compile_wgsl_with_telemetry(
        &self,
        source: &str,
        stage: Stage,
        entry_point: &str,
        constants: &PipelineConstants,
        options: Options,
    ) -> Result<(CacheKey, Arc<Artifact>, CompileTelemetry), Error> {
        let started = Instant::now();
        let key = CacheKey::new(source, stage, entry_point, constants, &options);
        if let Some(artifact) = self.memory_get(key) {
            return Ok((
                key,
                artifact,
                CompileTelemetry {
                    source: CacheSource::Memory,
                    elapsed: started.elapsed(),
                },
            ));
        }
        if let Some(artifact) = self.persistent_get(key) {
            let artifact = Arc::new(artifact);
            self.memory_insert(key, artifact.clone());
            return Ok((
                key,
                artifact,
                CompileTelemetry {
                    source: CacheSource::Persistent,
                    elapsed: started.elapsed(),
                },
            ));
        }
        let artifact =
            Arc::new(Compiler.compile_wgsl(source, stage, entry_point, constants, options)?);
        self.persistent_insert(key, &artifact);
        let artifact = self.memory_insert(key, artifact);
        Ok((
            key,
            artifact,
            CompileTelemetry {
                source: CacheSource::Compiled,
                elapsed: started.elapsed(),
            },
        ))
    }

    /// Number of successfully compiled entries currently retained in RAM.
    #[must_use]
    pub fn len(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entries
            .len()
    }

    /// Whether the cache contains no artifacts.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn memory_get(&self, key: CacheKey) -> Option<Arc<Artifact>> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let artifact = state.entries.get(&key).cloned()?;
        touch(&mut state.least_recently_used, key);
        Some(artifact)
    }

    fn memory_insert(&self, key: CacheKey, artifact: Arc<Artifact>) -> Arc<Artifact> {
        let artifact_bytes = artifact_size(&artifact);
        if self.max_memory_entries == 0
            || self.max_memory_bytes == 0
            || artifact_bytes > self.max_memory_bytes
        {
            return artifact;
        }
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let artifact = if let Some(existing) = state.entries.get(&key) {
            existing.clone()
        } else {
            state.retained_bytes = state.retained_bytes.saturating_add(artifact_bytes);
            state.entries.insert(key, artifact.clone());
            artifact
        };
        touch(&mut state.least_recently_used, key);
        while state.entries.len() > self.max_memory_entries
            || state.retained_bytes > self.max_memory_bytes
        {
            let Some(evicted) = state.least_recently_used.pop_front() else {
                break;
            };
            if let Some(artifact) = state.entries.remove(&evicted) {
                state.retained_bytes = state
                    .retained_bytes
                    .saturating_sub(artifact_size(&artifact));
            }
        }
        artifact
    }

    fn persistent_get(&self, key: CacheKey) -> Option<Artifact> {
        let directory = self.persistent_directory()?;
        let path = cache_path(directory, key);
        let metadata = fs::metadata(&path).ok()?;
        let maximum_size =
            PERSISTENT_CACHE_HEADER_SIZE.checked_add(MAX_PERSISTENT_ARTIFACT_SIZE)?;
        if metadata.len() > maximum_size as u64 {
            let _ = fs::remove_file(path);
            return None;
        }
        let bytes = fs::read(&path).ok()?;
        let artifact = decode_persistent_entry(key, &bytes).ok();
        if artifact.is_none() {
            let _ = fs::remove_file(path);
        }
        artifact
    }

    fn persistent_insert(&self, key: CacheKey, artifact: &Artifact) {
        let Some(directory) = self.persistent_directory() else {
            return;
        };
        if artifact.dksh.len() > MAX_PERSISTENT_ARTIFACT_SIZE
            || fs::create_dir_all(directory).is_err()
        {
            return;
        }
        let destination = cache_path(directory, key);
        if destination.is_file() {
            return;
        }
        let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temporary = directory.join(format!(
            ".{}.tmp-{}-{sequence}",
            key.to_hex(),
            std::process::id()
        ));
        let bytes = encode_persistent_entry(key, &artifact.dksh);
        let write_result = (|| -> std::io::Result<()> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)?;
            file.write_all(&bytes)?;
            file.flush()?;
            // Some Horizon/newlib filesystems do not expose fsync. Atomic rename still prevents
            // readers from observing a partially written final entry.
            let _ = file.sync_all();
            drop(file);
            fs::rename(&temporary, &destination)
        })();
        if write_result.is_err() {
            let _ = fs::remove_file(temporary);
        }
    }
}

fn touch(order: &mut VecDeque<CacheKey>, key: CacheKey) {
    if let Some(index) = order.iter().position(|candidate| *candidate == key) {
        order.remove(index);
    }
    order.push_back(key);
}

fn artifact_size(artifact: &Artifact) -> usize {
    artifact.dksh.len().saturating_add(
        artifact
            .bindings
            .len()
            .saturating_mul(core::mem::size_of::<deko_dksh::Binding>()),
    )
}

fn cache_path(directory: &Path, key: CacheKey) -> PathBuf {
    directory.join(format!("{}.dksc", key.to_hex()))
}

fn encode_persistent_entry(key: CacheKey, dksh: &[u8]) -> Vec<u8> {
    let checksum = Sha256::digest(dksh);
    let mut bytes = Vec::with_capacity(PERSISTENT_CACHE_HEADER_SIZE + dksh.len());
    bytes.extend_from_slice(PERSISTENT_CACHE_MAGIC);
    bytes.extend_from_slice(key.as_bytes());
    bytes.extend_from_slice(&(dksh.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&checksum);
    bytes.extend_from_slice(dksh);
    bytes
}

fn decode_persistent_entry(key: CacheKey, bytes: &[u8]) -> Result<Artifact, ()> {
    let header = bytes.get(..PERSISTENT_CACHE_HEADER_SIZE).ok_or(())?;
    if &header[..8] != PERSISTENT_CACHE_MAGIC || &header[8..40] != key.as_bytes() {
        return Err(());
    }
    let encoded_len = u64::from_le_bytes(header[40..48].try_into().map_err(|_| ())?);
    let encoded_len = usize::try_from(encoded_len).map_err(|_| ())?;
    if encoded_len > MAX_PERSISTENT_ARTIFACT_SIZE
        || PERSISTENT_CACHE_HEADER_SIZE.checked_add(encoded_len) != Some(bytes.len())
    {
        return Err(());
    }
    let dksh = &bytes[PERSISTENT_CACHE_HEADER_SIZE..];
    if Sha256::digest(dksh).as_slice() != &header[48..80] {
        return Err(());
    }
    let parsed = deko_dksh::parse(dksh).map_err(|_| ())?;
    Ok(Artifact {
        dksh: dksh.to_vec(),
        bindings: parsed.bindings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_DIRECTORY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn source(value: u32) -> String {
        format!("@compute @workgroup_size(1) fn main() {{ let value = {value}u; _ = value; }}")
    }

    fn compile(cache: &CompilerCache, source: &str) -> (CacheKey, Arc<Artifact>) {
        cache
            .compile_wgsl(
                source,
                Stage::Compute,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap()
    }

    fn test_directory(name: &str) -> PathBuf {
        let sequence = TEST_DIRECTORY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "deko-shader-compiler-{name}-{}-{sequence}",
            std::process::id()
        ))
    }

    #[test]
    fn memory_cache_enforces_entry_and_byte_limits() {
        let cache = CompilerCache::with_memory_limits(2, usize::MAX);
        let (first_key, _) = compile(&cache, &source(1));
        let (second_key, _) = compile(&cache, &source(2));
        let (third_key, _) = compile(&cache, &source(3));
        assert_eq!(cache.len(), 2);
        let state = cache.state.lock().unwrap();
        assert!(!state.entries.contains_key(&first_key));
        assert!(state.entries.contains_key(&second_key));
        assert!(state.entries.contains_key(&third_key));
        drop(state);

        let unlimited = CompilerCache::default();
        let (_, artifact) = compile(&unlimited, &source(4));
        let too_small = CompilerCache::with_memory_limits(8, artifact_size(&artifact) - 1);
        compile(&too_small, &source(4));
        assert!(too_small.is_empty());
    }

    #[test]
    fn telemetry_distinguishes_compilation_memory_and_persistent_hits() {
        let directory = test_directory("telemetry");
        let source = source(6);
        let writer = CompilerCache::new(1).with_persistent_directory(&directory);

        let (key, expected, cold) = writer
            .compile_wgsl_with_telemetry(
                &source,
                Stage::Compute,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        assert_eq!(cold.source, CacheSource::Compiled);

        let (_, warm, memory) = writer
            .compile_wgsl_with_telemetry(
                &source,
                Stage::Compute,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        assert_eq!(memory.source, CacheSource::Memory);
        assert!(Arc::ptr_eq(&expected, &warm));

        let reader = CompilerCache::new(1).with_persistent_directory(&directory);
        let (loaded_key, loaded, persistent) = reader
            .compile_wgsl_with_telemetry(
                &source,
                Stage::Compute,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        assert_eq!(persistent.source, CacheSource::Persistent);
        assert_eq!(loaded_key, key);
        assert_eq!(loaded.dksh, expected.dksh);

        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn persistent_cache_round_trips_and_rejects_corruption() {
        let directory = test_directory("persistent");
        let source = source(7);
        let writer = CompilerCache::new(1).with_persistent_directory(&directory);
        let (key, expected) = compile(&writer, &source);
        let path = cache_path(&directory, key);
        assert!(path.is_file());

        let reader = CompilerCache::new(1).with_persistent_directory(&directory);
        let loaded = reader.persistent_get(key).unwrap();
        assert_eq!(&loaded, expected.as_ref());

        let mut bytes = fs::read(&path).unwrap();
        let last = bytes.last_mut().unwrap();
        *last ^= 0x80;
        fs::write(&path, bytes).unwrap();
        assert!(reader.persistent_get(key).is_none());
        assert!(!path.exists());
        let (_, regenerated) = compile(&reader, &source);
        assert_eq!(regenerated, expected);
        assert!(path.is_file());

        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn persistent_envelope_rejects_wrong_key_and_truncation() {
        let cache = CompilerCache::default();
        let (key, artifact) = compile(&cache, &source(11));
        let other_key = CacheKey::new(
            &source(12),
            Stage::Compute,
            "main",
            &PipelineConstants::new(),
            &Options::default(),
        );
        let encoded = encode_persistent_entry(key, &artifact.dksh);
        assert!(decode_persistent_entry(other_key, &encoded).is_err());
        assert!(decode_persistent_entry(key, &encoded[..encoded.len() - 1]).is_err());
    }
}
