//! Public Naga-facing API for the Deko shader compiler.
//!
//! WGSL parsing and validation already work. Native lowering remains unavailable until
//! the Mesa NAK extraction is connected, and is reported as a typed error rather than
//! falling back to a host compiler or embedded artifact.

use std::collections::BTreeMap;

use thiserror::Error;

mod lower;

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
    /// The validated module uses a feature that has not reached the native lowering yet.
    #[error("unsupported shader feature: {0}")]
    UnsupportedFeature(String),
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
        let entry = lower::entry_point(request.module, request.stage, request.entry_point)
            .ok_or_else(|| Error::MissingEntryPoint {
                stage: request.stage,
                entry_point: request.entry_point.to_owned(),
            })?;
        let _ = request.info;
        let _ = request.constants;
        let _ = request.options;
        deko_nak::validate_target(deko_nak::Target::GM20B)?;

        let sm = deko_nak::ir::ShaderModelInfo::new(53, 64);
        let shader = lower::lower_entry_point(entry, &sm)?;
        let binary = deko_nak::compile_ir(shader, None)?;

        let (program_type, entrypoint, payload, code) = match request.stage {
            naga::ShaderStage::Compute => {
                let block_dimensions = entry.workgroup_size;
                (
                    deko_dksh::ProgramType::Compute,
                    0,
                    deko_dksh::StagePayload::Compute {
                        block_dimensions,
                        shared_memory_size: 0,
                        local_positive_memory_size: binary.local_memory_size,
                        local_negative_memory_size: 0,
                        crs_size: binary.crs_size,
                        num_barriers: binary.num_control_barriers,
                    },
                    binary.code,
                )
            }
            naga::ShaderStage::Vertex => (
                deko_dksh::ProgramType::Vertex,
                0x30,
                deko_dksh::StagePayload::Vertex {
                    alternate_entrypoint: 0,
                    alternate_num_gprs: 0,
                },
                graphics_code_image(&binary),
            ),
            naga::ShaderStage::Fragment => (
                deko_dksh::ProgramType::Fragment,
                0x30,
                deko_dksh::StagePayload::Fragment {
                    has_table_3d1: false,
                    early_fragment_tests: entry.early_depth_test.is_some(),
                    post_depth_coverage: false,
                    per_sample_invocation: false,
                    table_3d1: [0, 0, 0, 0x0860_7f80],
                    param_d8: 0,
                    param_65b: 0,
                    param_489: 0,
                },
                graphics_code_image(&binary),
            ),
            stage => return Err(Error::UnsupportedFeature(format!("{stage:?} stage"))),
        };
        let bindings = Vec::new();
        let dksh = deko_dksh::encode(
            deko_dksh::Program {
                program_type,
                entrypoint,
                num_gprs: binary.num_gprs,
                constbuf1: None,
                per_warp_scratch_size: binary.per_warp_scratch_size,
                payload,
            },
            &code,
            &bindings,
        )?;
        Ok(Artifact { dksh, bindings })
    }
}

fn graphics_code_image(binary: &deko_nak::ShaderBinary) -> Vec<u8> {
    const PREFIX_SIZE: usize = 0x30;
    const SPHV3_WORDS: usize = 20;

    let mut code = vec![0; PREFIX_SIZE];
    code.extend(
        binary.sph[..SPHV3_WORDS]
            .iter()
            .flat_map(|word| word.to_le_bytes()),
    );
    code.extend_from_slice(&binary.code);
    code
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
    fn missing_entry_point_is_distinct_from_compilation() {
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

        let artifact = Compiler
            .compile_wgsl(
                COMPUTE,
                naga::ShaderStage::Compute,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        let container = deko_dksh::parse(&artifact.dksh).unwrap();
        assert_eq!(
            container.program.program_type,
            deko_dksh::ProgramType::Compute
        );
        assert_eq!(container.program.entrypoint, 0);
        assert!(container.program.num_gprs >= 4);
        assert_eq!(container.program.per_warp_scratch_size, 0x800);
        assert!(artifact.bindings.is_empty());
        assert!(container.code.iter().any(|byte| *byte != 0));
    }

    #[test]
    fn compute_workgroup_metadata_and_output_are_deterministic() {
        let source = "@compute @workgroup_size(8, 4, 2) fn main() {}";
        let compile = || {
            Compiler
                .compile_wgsl(
                    source,
                    naga::ShaderStage::Compute,
                    "main",
                    &PipelineConstants::new(),
                    Options::default(),
                )
                .unwrap()
        };
        let first = compile();
        let second = compile();
        assert_eq!(first, second);

        let container = deko_dksh::parse(&first.dksh).unwrap();
        assert_eq!(
            container.program.payload,
            deko_dksh::StagePayload::Compute {
                block_dimensions: [8, 4, 2],
                shared_memory_size: 0,
                local_positive_memory_size: 0,
                local_negative_memory_size: 0,
                crs_size: 0x800,
                num_barriers: 0,
            }
        );
    }

    #[test]
    fn constant_vertex_and_fragment_compile_to_graphics_dksh() {
        let vertex = Compiler
            .compile_wgsl(
                "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }",
                naga::ShaderStage::Vertex,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        let vertex = deko_dksh::parse(&vertex.dksh).unwrap();
        assert_eq!(vertex.program.program_type, deko_dksh::ProgramType::Vertex);
        assert_eq!(vertex.program.entrypoint, 0x30);
        assert_eq!(&vertex.code[..0x30], &[0; 0x30]);
        assert_eq!(
            u32::from_le_bytes(vertex.code[0x30..0x34].try_into().unwrap()) & 0x3fff,
            0x0461
        );
        assert!(vertex.code[0x80..].iter().any(|byte| *byte != 0));

        let fragment = Compiler
            .compile_wgsl(
                "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 0.0, 0.0, 1.0); }",
                naga::ShaderStage::Fragment,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        let fragment = deko_dksh::parse(&fragment.dksh).unwrap();
        assert_eq!(
            fragment.program.program_type,
            deko_dksh::ProgramType::Fragment
        );
        assert_eq!(fragment.program.entrypoint, 0x30);
        assert_eq!(
            u32::from_le_bytes(fragment.code[0x30..0x34].try_into().unwrap()) & 0x3fff,
            0x1462
        );
        assert!(fragment.code[0x80..].iter().any(|byte| *byte != 0));
    }
}
