//! Public Naga-facing API for the Deko shader compiler.
//!
//! WGSL is parsed and validated with Naga, lowered directly to the extracted Mesa NAK backend,
//! and packaged as DKSH. Unsupported language and pipeline features fail with a typed error.

use std::collections::BTreeMap;

use thiserror::Error;

mod cache;
mod lower;

pub use cache::{BACKEND_ABI_VERSION, CACHE_KEY_VERSION, CacheKey, CompilerCache};

/// Pipeline override values after wgpu resolves their names.
pub type PipelineConstants = BTreeMap<String, f64>;

/// Programmable pipeline stage selected for compilation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Stage {
    /// Vertex stage.
    Vertex,
    /// Fragment stage.
    Fragment,
    /// Compute stage.
    Compute,
}

impl From<Stage> for naga::ShaderStage {
    fn from(stage: Stage) -> Self {
        match stage {
            Stage::Vertex => Self::Vertex,
            Stage::Fragment => Self::Fragment,
            Stage::Compute => Self::Compute,
        }
    }
}

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
    /// Whether workgroup-scoped memory must be initialized to zero.
    pub zero_initialize_workgroup_memory: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            target: Target::Gm20b,
            robustness: Robustness::Robust,
            multiview_mask: None,
            zero_initialize_workgroup_memory: true,
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
    /// Pipeline-overridable constants could not be resolved for the selected entry point.
    #[error("pipeline constants failed: {0}")]
    PipelineConstants(String),
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
        stage: Stage,
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
            stage: stage.into(),
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
        let constants = request
            .constants
            .iter()
            .map(|(name, value)| (name.clone(), *value))
            .collect::<naga::back::PipelineConstants>();
        let (module, info) = naga::back::pipeline_constants::process_overrides(
            request.module,
            request.info,
            Some((request.stage, request.entry_point)),
            &constants,
        )
        .map_err(|error| Error::PipelineConstants(error.to_string()))?;
        let entry =
            lower::entry_point(&module, request.stage, request.entry_point).ok_or_else(|| {
                Error::MissingEntryPoint {
                    stage: request.stage,
                    entry_point: request.entry_point.to_owned(),
                }
            })?;
        let _ = info;
        if request.options.multiview_mask.is_some() {
            return Err(Error::UnsupportedFeature("multiview".to_owned()));
        }
        match request.options.target {
            Target::Gm20b => deko_nak::validate_target(deko_nak::Target::GM20B)?,
        }

        let sm = deko_nak::ir::ShaderModelInfo::new(53, 64);
        let lowered = lower::lower_entry_point(&module, entry, &sm)?;
        let binary = deko_nak::compile_ir(lowered.shader, None)?;

        let (program_type, entrypoint, payload, code) = match request.stage {
            naga::ShaderStage::Compute => {
                let block_dimensions = entry.workgroup_size;
                (
                    deko_dksh::ProgramType::Compute,
                    0,
                    deko_dksh::StagePayload::Compute {
                        block_dimensions,
                        shared_memory_size: binary.shared_memory_size,
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
        let bindings = lowered.bindings;
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
                Stage::Compute,
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
                Stage::Compute,
                "missing",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap_err();
        assert!(matches!(missing, Error::MissingEntryPoint { .. }));

        let artifact = Compiler
            .compile_wgsl(
                COMPUTE,
                Stage::Compute,
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
                    Stage::Compute,
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
    fn compute_entry_points_lower_local_arithmetic() {
        let artifact = Compiler
            .compile_wgsl(
                "@compute @workgroup_size(4) fn main() { var value = vec4<f32>(1.0); value = value * 0.5 + vec4<f32>(0.25); }",
                Stage::Compute,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        let container = deko_dksh::parse(&artifact.dksh).unwrap();
        assert_eq!(
            container.program.payload,
            deko_dksh::StagePayload::Compute {
                block_dimensions: [4, 1, 1],
                shared_memory_size: 0,
                local_positive_memory_size: 0,
                local_negative_memory_size: 0,
                crs_size: 0x800,
                num_barriers: 0,
            }
        );
    }

    #[test]
    fn compute_storage_buffers_lower_to_global_memory_and_binding_metadata() {
        let source = r"
            struct Data {
                values: array<u32, 4>,
            }

            @group(2) @binding(3)
            var<storage, read_write> data: Data;

            @compute @workgroup_size(4)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let index = id.x % 4u;
                let value = data.values[index];
                data.values[index] = value + 1u;
            }
        ";
        let artifact = Compiler
            .compile_wgsl(
                source,
                Stage::Compute,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        assert_eq!(
            artifact.bindings,
            vec![deko_dksh::Binding {
                group: 2,
                binding: 3,
                target: 0,
                kind: deko_dksh::BindingKind::Storage,
            }]
        );
        let container = deko_dksh::parse(&artifact.dksh).unwrap();
        assert_eq!(container.bindings, artifact.bindings);
        assert!(container.code.iter().any(|byte| *byte != 0));
    }

    #[test]
    fn compute_storage_atomics_compile() {
        let source = r"
            struct Counters {
                values: array<atomic<u32>, 4>,
            }

            @group(0) @binding(0)
            var<storage, read_write> counters: Counters;

            @compute @workgroup_size(4)
            fn main(@builtin(local_invocation_index) index: u32) {
                let previous = atomicAdd(&counters.values[index], 1u);
                atomicMax(&counters.values[0], previous);
                storageBarrier();
                workgroupBarrier();
            }
        ";
        let artifact = Compiler
            .compile_wgsl(
                source,
                Stage::Compute,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        assert_eq!(artifact.bindings[0].kind, deko_dksh::BindingKind::Storage);
        assert!(deko_dksh::parse(&artifact.dksh).is_ok());
    }

    #[test]
    fn compute_workgroup_memory_and_barriers_compile() {
        let source = r"
            @group(0) @binding(0)
            var<storage, read_write> output: array<u32, 4>;
            var<workgroup> scratch: array<u32, 4>;
            var<workgroup> counter: atomic<u32>;

            @compute @workgroup_size(4)
            fn main(@builtin(local_invocation_index) index: u32) {
                scratch[index] = index * 2u;
                if index == 0u {
                    atomicStore(&counter, 0u);
                }
                workgroupBarrier();
                let previous = atomicAdd(&counter, 1u);
                output[index] = scratch[3u - index] + previous;
            }
        ";
        let artifact = Compiler
            .compile_wgsl(
                source,
                Stage::Compute,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        let container = deko_dksh::parse(&artifact.dksh).unwrap();
        assert_eq!(
            container.program.payload,
            deko_dksh::StagePayload::Compute {
                block_dimensions: [4, 1, 1],
                shared_memory_size: 20,
                local_positive_memory_size: 0,
                local_negative_memory_size: 0,
                crs_size: 0x800,
                num_barriers: 0,
            }
        );
    }

    #[test]
    fn runtime_storage_array_length_compiles() {
        let source = r"
            struct Input {
                header: u32,
                values: array<u32>,
            }

            @group(0) @binding(0) var<storage, read> input: Input;
            @group(0) @binding(1) var<storage, read_write> output: array<u32, 1>;

            @compute @workgroup_size(1)
            fn main() {
                output[0] = arrayLength(&input.values);
            }
        ";
        let artifact = Compiler
            .compile_wgsl(
                source,
                Stage::Compute,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        assert_eq!(artifact.bindings.len(), 2);
        assert!(deko_dksh::parse(&artifact.dksh).is_ok());
    }

    #[test]
    fn constant_vertex_and_fragment_compile_to_graphics_dksh() {
        let vertex = Compiler
            .compile_wgsl(
                "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }",
                Stage::Vertex,
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
                Stage::Fragment,
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

    #[test]
    fn location_inputs_flow_through_vertex_and_fragment_stages() {
        let vertex = Compiler
            .compile_wgsl(
                "@vertex fn main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(position, 1.0); }",
                Stage::Vertex,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        let vertex = deko_dksh::parse(&vertex.dksh).unwrap();
        assert_eq!(vertex.program.program_type, deko_dksh::ProgramType::Vertex);
        assert!(vertex.code[0x80..].iter().any(|byte| *byte != 0));

        let fragment = Compiler
            .compile_wgsl(
                "@fragment fn main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> { return color; }",
                Stage::Fragment,
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
        assert!(fragment.code[0x80..].iter().any(|byte| *byte != 0));
    }

    #[test]
    fn float_vector_arithmetic_lowers_to_maxwell_code() {
        let artifact = Compiler
            .compile_wgsl(
                "@fragment fn main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> { return color * 0.5 + vec4<f32>(0.1, 0.2, 0.3, 0.0); }",
                Stage::Fragment,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        let container = deko_dksh::parse(&artifact.dksh).unwrap();
        assert_eq!(
            container.program.program_type,
            deko_dksh::ProgramType::Fragment
        );
        assert!(container.program.num_gprs >= 4);
        assert!(container.code[0x80..].iter().any(|byte| *byte != 0));
    }

    #[test]
    fn vertex_struct_outputs_position_and_varying() {
        let source = r"
            struct VertexOut {
                @builtin(position) position: vec4<f32>,
                @location(0) color: vec4<f32>,
            }
            @vertex
            fn main(
                @location(0) position: vec3<f32>,
                @location(1) color: vec4<f32>,
            ) -> VertexOut {
                return VertexOut(vec4<f32>(position, 1.0), color);
            }
        ";
        let artifact = Compiler
            .compile_wgsl(
                source,
                Stage::Vertex,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        let container = deko_dksh::parse(&artifact.dksh).unwrap();
        assert_eq!(
            container.program.program_type,
            deko_dksh::ProgramType::Vertex
        );
        assert!(container.code[0x80..].iter().any(|byte| *byte != 0));
    }

    #[test]
    fn matrix_vector_multiply_lowers_vertex_transforms() {
        let source = r"
            @vertex
            fn main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
                let transform = mat4x4<f32>(
                    vec4<f32>(2.0, 0.0, 0.0, 0.0),
                    vec4<f32>(0.0, 3.0, 0.0, 0.0),
                    vec4<f32>(0.0, 0.0, 4.0, 0.0),
                    vec4<f32>(1.0, 2.0, 3.0, 1.0),
                );
                return transform * vec4<f32>(position, 1.0);
            }
        ";
        let artifact = Compiler
            .compile_wgsl(
                source,
                Stage::Vertex,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        let container = deko_dksh::parse(&artifact.dksh).unwrap();
        assert_eq!(
            container.program.program_type,
            deko_dksh::ProgramType::Vertex
        );
        assert!(container.code[0x80..].iter().any(|byte| *byte != 0));
    }

    #[test]
    fn uniform_matrix_loads_emit_binding_metadata() {
        let source = r"
            struct View { transform: mat4x4<f32> }
            @group(2) @binding(7) var<uniform> view: View;
            @vertex
            fn main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
                return view.transform * vec4<f32>(position, 1.0);
            }
        ";
        let artifact = Compiler
            .compile_wgsl(
                source,
                Stage::Vertex,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        assert_eq!(
            artifact.bindings,
            vec![deko_dksh::Binding {
                group: 2,
                binding: 7,
                target: 0,
                kind: deko_dksh::BindingKind::Uniform,
            }]
        );
        let container = deko_dksh::parse(&artifact.dksh).unwrap();
        assert_eq!(container.bindings, artifact.bindings);
        assert!(container.code[0x80..].iter().any(|byte| *byte != 0));
    }

    #[test]
    fn splat_swizzle_zero_and_negate_lower_componentwise() {
        let artifact = Compiler
            .compile_wgsl(
                "@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> { let z = vec4<f32>(); return -z + vec4<f32>(uv.yx + vec2<f32>(uv[1], uv[0]), 0.5, 1.0); }",
                Stage::Fragment,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        let container = deko_dksh::parse(&artifact.dksh).unwrap();
        assert_eq!(
            container.program.program_type,
            deko_dksh::ProgramType::Fragment
        );
        assert!(container.code[0x80..].iter().any(|byte| *byte != 0));
    }

    #[test]
    fn pure_helper_functions_are_inlined() {
        let source = r"
            fn tint(value: vec4<f32>, factor: f32) -> vec4<f32> {
                return value * factor + vec4<f32>(0.1, 0.0, 0.0, 0.0);
            }
            @fragment
            fn main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
                return tint(color, 0.5);
            }
        ";
        let artifact = Compiler
            .compile_wgsl(
                source,
                Stage::Fragment,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        let container = deko_dksh::parse(&artifact.dksh).unwrap();
        assert_eq!(
            container.program.program_type,
            deko_dksh::ProgramType::Fragment
        );
        assert!(container.code[0x80..].iter().any(|byte| *byte != 0));
    }

    #[test]
    fn mutable_function_locals_lower_to_ssa_values() {
        let source = r"
            @fragment
            fn main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
                var result = vec4<f32>();
                result = color * 0.5;
                { result = result + vec4<f32>(0.1, 0.0, 0.0, 0.0); }
                return result;
            }
        ";
        let artifact = Compiler
            .compile_wgsl(
                source,
                Stage::Fragment,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        assert!(deko_dksh::parse(&artifact.dksh).is_ok());
    }

    #[test]
    fn sampled_ui_fragment_control_flow_and_math_compile() {
        let source = r"
            @group(1) @binding(0) var image: texture_2d<f32>;
            @group(1) @binding(1) var image_sampler: sampler;

            fn enabled(flags: u32, mask: u32) -> bool {
                return (flags & mask) != 0u;
            }

            fn border_active(point: vec2<f32>, flags: u32) -> bool {
                var selected: bool;
                if (flags & 6u) == 6u { return true; }
                let distance = min(abs(point.x), abs(point.y));
                if enabled(flags, 2u) {
                    selected = distance == abs(point.x);
                } else {
                    selected = false;
                }
                return selected;
            }

            @fragment
            fn main(
                @location(0) uv: vec2<f32>,
                @location(1) @interpolate(flat) flags: u32,
                @location(2) color: vec4<f32>,
            ) -> @location(0) vec4<f32> {
                let texel = textureSample(image, image_sampler, uv);
                let q = abs(uv) - vec2<f32>(0.5);
                let distance = length(max(q, vec2<f32>(0.0))) / 2.0;
                let alpha = saturate(1.0 - step(0.0, distance));
                let selected = border_active(q, flags);
                if enabled(flags, 1u) {
                    return select(color, color * texel, selected);
                } else {
                    return vec4<f32>(color.xyz, clamp(color.w * alpha, 0.0, 1.0));
                }
            }
        ";
        let artifact = Compiler
            .compile_wgsl(
                source,
                Stage::Fragment,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        assert_eq!(
            artifact.bindings,
            vec![
                deko_dksh::Binding {
                    group: 1,
                    binding: 0,
                    target: 0,
                    kind: deko_dksh::BindingKind::Texture,
                },
                deko_dksh::Binding {
                    group: 1,
                    binding: 1,
                    target: 0,
                    kind: deko_dksh::BindingKind::Sampler,
                },
            ]
        );
        let container = deko_dksh::parse(&artifact.dksh).unwrap();
        assert_eq!(container.bindings, artifact.bindings);
        assert_eq!(
            container.program.program_type,
            deko_dksh::ProgramType::Fragment
        );
        assert!(container.code[0x80..].iter().any(|byte| *byte != 0));
    }

    #[test]
    fn bevy_style_dynamic_uniform_vertex_features_compile() {
        let source = r"
            struct Mesh { world_from_local: mat3x4<f32> }
            struct VertexIn {
                @builtin(instance_index) instance: u32,
                @builtin(vertex_index) vertex: u32,
                @location(0) position: vec3<f32>,
                @location(1) normal: vec3<f32>,
            }
            @group(2) @binding(0) var<uniform> meshes: array<Mesh, 8>;

            @vertex
            fn main(input: VertexIn) -> @builtin(position) vec4<f32> {
                let transform = meshes[input.instance].world_from_local;
                let transformed_normal = normalize((transpose(transform) * vec4<f32>(input.normal, 0.0)).xyz);
                let nonzero = any(transformed_normal != vec3<f32>());
                return vec4<f32>(input.position + transformed_normal * f32(nonzero), 1.0 + f32(input.vertex & 0u));
            }
        ";
        let artifact = Compiler
            .compile_wgsl(
                source,
                Stage::Vertex,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        assert_eq!(
            artifact.bindings,
            vec![deko_dksh::Binding {
                group: 2,
                binding: 0,
                target: 0,
                kind: deko_dksh::BindingKind::Uniform,
            }]
        );
        let container = deko_dksh::parse(&artifact.dksh).unwrap();
        assert_eq!(container.bindings, artifact.bindings);
        assert!(container.code[0x80..].iter().any(|byte| *byte != 0));
    }

    #[test]
    fn bevy_style_fragment_builtins_bias_sampling_and_math_compile() {
        let source = r"
            @group(1) @binding(0) var image: texture_2d<f32>;
            @group(1) @binding(1) var image_sampler: sampler;

            @fragment
            fn main(
                @builtin(position) position: vec4<f32>,
                @builtin(front_facing) front: bool,
                @location(0) uv: vec2<f32>,
            ) -> @location(0) vec4<f32> {
                let sampled = textureSampleBias(image, image_sampler, uv, fract(position.x) - 0.5);
                let facing = f32(front);
                let shaped = smoothstep(vec3<f32>(0.0), vec3<f32>(1.0), normalize(sampled.xyz));
                return vec4<f32>(mix(sampled.xyz, shaped, facing), 1.0);
            }
        ";
        let artifact = Compiler
            .compile_wgsl(
                source,
                Stage::Fragment,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        assert_eq!(artifact.bindings.len(), 2);
        let container = deko_dksh::parse(&artifact.dksh).unwrap();
        assert_eq!(container.bindings, artifact.bindings);
        assert!(container.code[0x80..].iter().any(|byte| *byte != 0));
    }

    #[test]
    fn structured_loops_mixed_aggregates_and_depth_queries_compile() {
        let source = r"
            struct Mixed {
                enabled: bool,
                color: vec3<f32>,
            }

            @group(0) @binding(0) var shadow: texture_depth_2d_array;
            @group(0) @binding(1) var shadow_sampler: sampler_comparison;

            fn find_index(limit: u32, wanted: u32) -> u32 {
                var index = 0u;
                loop {
                    if index < limit {
                    } else {
                        break;
                    }
                    if index == wanted {
                        return index;
                    }
                    continuing {
                        index = index + 1u;
                    }
                }
                return limit;
            }

            @fragment
            fn main(
                @builtin(front_facing) front: bool,
                @location(0) uv: vec2<f32>,
            ) -> @location(0) vec4<f32> {
                var mixed: Mixed;
                mixed.enabled = front;
                mixed.color = vec3<f32>(0.25, 0.5, 0.75);
                let dimensions = textureDimensions(shadow);
                let index = find_index(4u, u32(dimensions.x) & 3u);
                let candidates = array<vec2<f32>, 2>(uv, vec2<f32>(1.0) - uv);
                let coordinate = candidates[index & 1u];
                let visibility = textureSampleCompareLevel(
                    shadow,
                    shadow_sampler,
                    coordinate,
                    i32(index & 1u),
                    0.5,
                );
                return vec4<f32>(mixed.color * visibility * f32(mixed.enabled), 1.0);
            }
        ";
        let artifact = Compiler
            .compile_wgsl(
                source,
                Stage::Fragment,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        assert_eq!(artifact.bindings.len(), 2);
        let container = deko_dksh::parse(&artifact.dksh).unwrap();
        assert_eq!(container.bindings, artifact.bindings);
        assert!(container.program.num_gprs >= 4);
        assert!(container.code[0x80..].iter().any(|byte| *byte != 0));
    }

    #[test]
    fn pipeline_overrides_are_resolved_and_multiview_fails_explicitly() {
        let source = "override scale: f32 = 1.0; @fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(scale, 0.0, 0.0, 1.0); }";
        let compile = |scale| {
            let mut constants = PipelineConstants::new();
            constants.insert("scale".to_owned(), scale);
            Compiler
                .compile_wgsl(
                    source,
                    Stage::Fragment,
                    "main",
                    &constants,
                    Options::default(),
                )
                .unwrap()
        };
        assert_ne!(compile(0.25), compile(0.75));

        let multiview_error = Compiler
            .compile_wgsl(
                "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(); }",
                Stage::Vertex,
                "main",
                &PipelineConstants::new(),
                Options {
                    multiview_mask: Some(1),
                    ..Options::default()
                },
            )
            .unwrap_err();
        assert!(matches!(multiview_error, Error::UnsupportedFeature(_)));
    }

    #[test]
    fn cache_key_covers_every_codegen_input_and_cache_reuses_artifacts() {
        let source = "@compute @workgroup_size(1) fn main() {}";
        let constants = PipelineConstants::new();
        let options = Options::default();
        let base = CacheKey::new(source, Stage::Compute, "main", &constants, &options);
        assert_eq!(base.to_hex().len(), 64);
        assert_eq!(
            base,
            CacheKey::new(source, Stage::Compute, "main", &constants, &options)
        );
        assert_ne!(
            base,
            CacheKey::new(
                " @compute @workgroup_size(1) fn main() {}",
                Stage::Compute,
                "main",
                &constants,
                &options,
            )
        );
        assert_ne!(
            base,
            CacheKey::new(source, Stage::Vertex, "main", &constants, &options)
        );
        assert_ne!(
            base,
            CacheKey::new(source, Stage::Compute, "other", &constants, &options)
        );
        assert_ne!(
            base,
            CacheKey::new(
                source,
                Stage::Compute,
                "main",
                &constants,
                &Options {
                    zero_initialize_workgroup_memory: false,
                    ..options.clone()
                },
            )
        );

        let cache = CompilerCache::default();
        let (first_key, first) = cache
            .compile_wgsl(source, Stage::Compute, "main", &constants, options.clone())
            .unwrap();
        let (second_key, second) = cache
            .compile_wgsl(source, Stage::Compute, "main", &constants, options)
            .unwrap();
        assert_eq!(first_key, second_key);
        assert!(std::sync::Arc::ptr_eq(&first, &second));
        assert_eq!(cache.len(), 1);
    }
}
