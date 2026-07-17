//! Public Naga-facing API for the Deko shader compiler.
//!
//! WGSL parsing and validation already work. Native lowering remains unavailable until
//! the Mesa NAK extraction is connected, and is reported as a typed error rather than
//! falling back to a host compiler or embedded artifact.

use std::collections::BTreeMap;

use thiserror::Error;

/// Pipeline override values after wgpu resolves their names.
pub type PipelineConstants = BTreeMap<String, f64>;

/// Switch compiler target.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Target {
    /// Nintendo Switch Tegra X1 GM20B.
    #[default]
    Gm20b,
}

/// Policy for out-of-bounds memory accesses.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Robustness {
    /// Match wgpu's robust-access requirements.
    #[default]
    Robust,
    /// Caller has already inserted the required checks.
    PreLowered,
}

/// Options that affect generated native code and therefore the cache key.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Options {
    /// Hardware target.
    pub target: Target,
    /// Robust-access policy.
    pub robustness: Robustness,
    /// Optional multiview mask for vertex-stage compilation.
    pub multiview_mask: Option<u32>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            target: Target::Gm20b,
            robustness: Robustness::Robust,
            multiview_mask: None,
        }
    }
}

/// A validated Naga module plus pipeline-specific compilation state.
pub struct ModuleRequest<'a> {
    /// Naga module validated by the caller or [`Compiler::compile_wgsl`].
    pub module: &'a naga::Module,
    /// Validation metadata corresponding exactly to `module`.
    pub info: &'a naga::valid::ModuleInfo,
    /// Selected shader stage.
    pub stage: naga::ShaderStage,
    /// Selected entry-point name.
    pub entry_point: &'a str,
    /// Pipeline override values.
    pub constants: &'a PipelineConstants,
    /// Target and lowering policy.
    pub options: Options,
}

/// One compiled native shader and its resource reflection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Artifact {
    /// Complete DKSH bytes accepted by `Deko3D`.
    pub dksh: Vec<u8>,
    /// Resource bindings encoded in the DKSH extension.
    pub bindings: Vec<deko_dksh::Binding>,
}

/// Shader compilation failure.
#[derive(Debug, Error)]
pub enum Error {
    /// WGSL parsing failed.
    #[error("WGSL parsing failed: {0}")]
    Parse(String),
    /// Naga validation failed.
    #[error("shader validation failed: {0}")]
    Validation(String),
    /// The requested entry point and stage do not exist together.
    #[error("{stage:?} entry point '{entry_point}' does not exist")]
    MissingEntryPoint {
        /// Requested stage.
        stage: naga::ShaderStage,
        /// Requested entry-point name.
        entry_point: String,
    },
    /// Native backend is not connected yet.
    #[error(transparent)]
    Backend(#[from] deko_nak::Error),
    /// DKSH packaging failed.
    #[error(transparent)]
    Dksh(#[from] deko_dksh::Error),
}

/// Stateless Deko shader compiler.
#[derive(Clone, Copy, Debug, Default)]
pub struct Compiler;

impl Compiler {
    /// Parse and validate WGSL, then compile one selected pipeline entry point.
    ///
    /// # Errors
    ///
    /// Returns a parse, validation, entry-point, backend, or DKSH packaging error.
    pub fn compile_wgsl(
        self,
        source: &str,
        stage: naga::ShaderStage,
        entry_point: &str,
        constants: &PipelineConstants,
        options: Options,
    ) -> Result<Artifact, Error> {
        let module = naga::front::wgsl::parse_str(source)
            .map_err(|error| Error::Parse(error.emit_to_string(source)))?;
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let info = validator
            .validate(&module)
            .map_err(|error| Error::Validation(error.to_string()))?;
        self.compile_module(&ModuleRequest {
            module: &module,
            info: &info,
            stage,
            entry_point,
            constants,
            options,
        })
    }

    /// Compile one entry point from an already validated Naga module.
    ///
    /// # Errors
    ///
    /// Returns an entry-point, backend, or DKSH packaging error.
    pub fn compile_module(self, request: &ModuleRequest<'_>) -> Result<Artifact, Error> {
        if !request
            .module
            .entry_points
            .iter()
            .any(|entry| entry.stage == request.stage && entry.name == request.entry_point)
        {
            return Err(Error::MissingEntryPoint {
                stage: request.stage,
                entry_point: request.entry_point.to_owned(),
            });
        }
        let _ = request.info;
        let _ = request.constants;
        let _ = request.options;
        deko_nak::validate_target(deko_nak::Target::GM20B)?;
        Err(deko_nak::Error::BackendUnavailable.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMPUTE: &str = r"
        @compute @workgroup_size(1)
        fn main() {}
    ";

    #[test]
    fn invalid_wgsl_is_a_parse_error() {
        let error = Compiler
            .compile_wgsl(
                "this is not WGSL",
                naga::ShaderStage::Compute,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap_err();
        assert!(matches!(error, Error::Parse(_)));
    }

    #[test]
    fn missing_entry_point_is_distinct_from_backend_progress() {
        let missing = Compiler
            .compile_wgsl(
                COMPUTE,
                naga::ShaderStage::Compute,
                "missing",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap_err();
        assert!(matches!(missing, Error::MissingEntryPoint { .. }));

        let backend = Compiler
            .compile_wgsl(
                COMPUTE,
                naga::ShaderStage::Compute,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap_err();
        assert!(matches!(
            backend,
            Error::Backend(deko_nak::Error::BackendUnavailable)
        ));
    }
}
