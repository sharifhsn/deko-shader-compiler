use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use sha2::{Digest, Sha256};

use crate::{Artifact, Compiler, Error, Options, PipelineConstants, Stage};

/// Version of the stable cache-key encoding.
pub const CACHE_KEY_VERSION: u32 = 1;

/// Version of the native code-generation contract.
///
/// This must be incremented whenever a backend change can alter or invalidate DKSH output without
/// a package-version change.
pub const BACKEND_ABI_VERSION: u32 = 9;

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
#[derive(Clone, Debug, Default)]
pub struct CompilerCache {
    entries: Arc<Mutex<HashMap<CacheKey, Arc<Artifact>>>>,
}

impl CompilerCache {
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
        let key = CacheKey::new(source, stage, entry_point, constants, &options);
        if let Some(artifact) = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&key)
            .cloned()
        {
            return Ok((key, artifact));
        }
        let artifact =
            Arc::new(Compiler.compile_wgsl(source, stage, entry_point, constants, options)?);
        let artifact = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(key)
            .or_insert_with(|| artifact.clone())
            .clone();
        Ok((key, artifact))
    }

    /// Number of successfully compiled entries currently retained in RAM.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// Whether the cache contains no artifacts.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
