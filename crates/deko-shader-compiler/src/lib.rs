//! Public Naga-facing API for the Deko shader compiler.
//!
//! WGSL is parsed and validated with Naga, lowered directly to the extracted Mesa NAK backend,
//! and packaged as DKSH. Unsupported language and pipeline features fail with a typed error.

use std::collections::BTreeMap;

use thiserror::Error;

mod cache;
mod lower;

pub use cache::{
    BACKEND_ABI_VERSION, CACHE_KEY_VERSION, CacheKey, CacheSource, CompileTelemetry, CompilerCache,
    DEFAULT_MEMORY_CACHE_BYTES, DEFAULT_MEMORY_CACHE_ENTRIES,
};

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

/// Concrete descriptor count for a runtime-sized WGSL resource binding array.
///
/// WGSL deliberately omits the size of `binding_array<T>`. The size is supplied by the
/// pipeline's bind-group layout, so callers compiling such a module must forward that layout
/// information here.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct BindingArraySize {
    /// Bind-group index.
    pub group: u32,
    /// Binding index within the group.
    pub binding: u32,
    /// Number of descriptors in the binding array.
    pub count: u32,
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
    /// Pipeline-layout sizes for runtime-sized resource binding arrays.
    pub binding_array_sizes: Vec<BindingArraySize>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            target: Target::Gm20b,
            robustness: Robustness::Robust,
            multiview_mask: None,
            zero_initialize_workgroup_memory: true,
            binding_array_sizes: Vec::new(),
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
    /// Resolve wgpu-style automatic entry-point selection for WGSL.
    ///
    /// The requested name wins when present. Otherwise, a module with exactly one entry point for
    /// the requested stage selects that entry point, matching wgpu's omitted-entry behavior.
    ///
    /// # Errors
    ///
    /// Returns a parse error or [`Error::MissingEntryPoint`] when selection is ambiguous or absent.
    pub fn resolve_wgsl_entry_point(
        self,
        source: &str,
        stage: Stage,
        requested: &str,
    ) -> Result<String, Error> {
        let module = naga::front::wgsl::parse_str(source)
            .map_err(|error| Error::Parse(error.emit_to_string(source)))?;
        let stage = naga::ShaderStage::from(stage);
        if module
            .entry_points
            .iter()
            .any(|entry| entry.stage == stage && entry.name == requested)
        {
            return Ok(requested.to_owned());
        }
        let mut candidates = module
            .entry_points
            .iter()
            .filter(|entry| entry.stage == stage)
            .map(|entry| entry.name.as_str());
        let Some(candidate) = candidates.next() else {
            return Err(Error::MissingEntryPoint {
                stage,
                entry_point: requested.to_owned(),
            });
        };
        if candidates.next().is_some() {
            return Err(Error::MissingEntryPoint {
                stage,
                entry_point: requested.to_owned(),
            });
        }
        Ok(candidate.to_owned())
    }

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
        if let Some(mask) = request.options.multiview_mask {
            if mask == 0 {
                return Err(Error::UnsupportedFeature("zero multiview mask".to_owned()));
            }
            if request.stage == naga::ShaderStage::Compute {
                return Err(Error::UnsupportedFeature(
                    "multiview compute pipeline".to_owned(),
                ));
            }
        }
        match request.options.target {
            Target::Gm20b => deko_nak::validate_target(deko_nak::Target::GM20B)?,
        }

        let sm = deko_nak::ir::ShaderModelInfo::new(53, 64);
        let lowered = lower::lower_entry_point(&module, entry, &sm, &request.options)?;
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

    fn lowered_ir(source: &str, stage: naga::ShaderStage, entry_point: &str) -> String {
        lowered_ir_with_options(source, stage, entry_point, &Options::default())
    }

    fn lowered_ir_with_options(
        source: &str,
        stage: naga::ShaderStage,
        entry_point: &str,
        options: &Options,
    ) -> String {
        let module = naga::front::wgsl::parse_str(source).unwrap();
        let entry = lower::entry_point(&module, stage, entry_point).unwrap();
        let sm = deko_nak::ir::ShaderModelInfo::new(53, 64);
        let lowered = lower::lower_entry_point(&module, entry, &sm, options).unwrap();
        lowered.shader.to_string()
    }

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
    fn sole_stage_entry_point_can_be_selected_automatically() {
        let source = "@vertex fn vertex_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }";
        assert_eq!(
            Compiler
                .resolve_wgsl_entry_point(source, Stage::Vertex, "main")
                .unwrap(),
            "vertex_main"
        );
        let ambiguous = concat!(
            "@compute @workgroup_size(1) fn first() {}",
            "@compute @workgroup_size(1) fn second() {}",
        );
        assert!(matches!(
            Compiler.resolve_wgsl_entry_point(ambiguous, Stage::Compute, "main"),
            Err(Error::MissingEntryPoint { .. })
        ));
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
    fn compute_storage_texture_atomics_compile() {
        let source = r"
            @group(0) @binding(0)
            var unsigned_image: texture_storage_2d<r32uint, atomic>;
            @group(0) @binding(1)
            var signed_image: texture_storage_2d<r32sint, atomic>;

            @compute @workgroup_size(2)
            fn main(@builtin(local_invocation_id) id: vec3<u32>) {
                let coordinate = vec2<i32>(id.xy);
                textureAtomicMax(unsigned_image, coordinate, 1u);
                textureAtomicMin(unsigned_image, coordinate, 1u);
                textureAtomicAdd(unsigned_image, coordinate, 1u);
                textureAtomicAnd(unsigned_image, coordinate, 1u);
                textureAtomicOr(unsigned_image, coordinate, 1u);
                textureAtomicXor(unsigned_image, coordinate, 1u);
                textureAtomicMax(signed_image, coordinate, 1i);
                textureAtomicMin(signed_image, coordinate, 1i);
                textureAtomicAdd(signed_image, coordinate, 1i);
                textureAtomicAnd(signed_image, coordinate, 1i);
                textureAtomicOr(signed_image, coordinate, 1i);
                textureAtomicXor(signed_image, coordinate, 1i);
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
        assert!(
            artifact
                .bindings
                .iter()
                .all(|binding| binding.kind == deko_dksh::BindingKind::StorageTexture)
        );
        assert!(deko_dksh::parse(&artifact.dksh).is_ok());

        let ir = lowered_ir(source, naga::ShaderStage::Compute, "main");
        assert_eq!(ir.matches("suatom").count(), 12, "{ir}");
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
    fn texture_load_and_fragment_depth_compile() {
        let source = r"
            struct FragmentOutput {
                @builtin(frag_depth) depth: f32,
            }

            @group(0) @binding(0) var image: texture_2d<u32>;

            @fragment
            fn fragment(@builtin(position) position: vec4<f32>) -> FragmentOutput {
                let texel = textureLoad(image, vec2<i32>(position.xy), 0);
                return FragmentOutput(f32(texel.x) / 255.0);
            }
        ";
        let artifact = Compiler
            .compile_wgsl(
                source,
                Stage::Fragment,
                "fragment",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        assert_eq!(artifact.bindings.len(), 1);
        let container = deko_dksh::parse(&artifact.dksh).unwrap();
        assert_eq!(
            container.program.program_type,
            deko_dksh::ProgramType::Fragment
        );
        assert!(container.code[0x80..].iter().any(|byte| *byte != 0));
    }

    #[test]
    fn duplicate_resource_aliases_share_one_deko_target() {
        let source = r"
            @group(0) @binding(0) var image_alias: texture_2d<f32>;
            @group(0) @binding(1) var sampler_alias: sampler;
            @group(0) @binding(0) var image: texture_2d<f32>;
            @group(0) @binding(1) var image_sampler: sampler;

            @fragment
            fn fragment(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
                return textureSample(image, image_sampler, uv);
            }
        ";
        let artifact = Compiler
            .compile_wgsl(
                source,
                Stage::Fragment,
                "fragment",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        assert_eq!(
            artifact.bindings,
            vec![
                deko_dksh::Binding {
                    group: 0,
                    binding: 0,
                    target: 0,
                    kind: deko_dksh::BindingKind::Texture,
                },
                deko_dksh::Binding {
                    group: 0,
                    binding: 1,
                    target: 0,
                    kind: deko_dksh::BindingKind::Sampler,
                },
            ]
        );
    }

    #[test]
    fn shared_cross_group_sampler_uses_one_bindless_descriptor() {
        let source = r"
            @group(0) @binding(5) var first: texture_2d<f32>;
            @group(0) @binding(1) var second: texture_2d<f32>;
            @group(2) @binding(7) var shared_sampler: sampler;

            @fragment
            fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
                return textureSample(first, shared_sampler, uv) + textureSample(second, shared_sampler, uv);
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
        assert_eq!(artifact.bindings.len(), 3);
        assert_eq!(
            artifact
                .bindings
                .iter()
                .filter(|binding| binding.kind == deko_dksh::BindingKind::Sampler)
                .map(|binding| (binding.group, binding.binding, binding.target))
                .collect::<Vec<_>>(),
            vec![(2, 7, 0)]
        );
        assert_eq!(
            artifact
                .bindings
                .iter()
                .filter(|binding| binding.kind == deko_dksh::BindingKind::Texture)
                .map(|binding| (binding.group, binding.binding, binding.target))
                .collect::<Vec<_>>(),
            vec![(0, 1, 0), (0, 5, 1)]
        );
    }

    #[test]
    fn dynamic_texture_and_sampler_binding_arrays_compile_bindlessly() {
        let source = r"
            @group(0) @binding(0) var samplers: binding_array<sampler>;
            @group(0) @binding(1) var images: binding_array<texture_2d<f32>>;

            @fragment
            fn main(@location(0) uv: vec2<f32>, @location(1) index: u32) -> @location(0) vec4<f32> {
                return textureSample(images[index], samplers[index + 1u], uv);
            }
        ";
        let artifact = Compiler
            .compile_wgsl(
                source,
                Stage::Fragment,
                "main",
                &PipelineConstants::new(),
                Options {
                    binding_array_sizes: vec![
                        BindingArraySize {
                            group: 0,
                            binding: 0,
                            count: 16,
                        },
                        BindingArraySize {
                            group: 0,
                            binding: 1,
                            count: 16,
                        },
                    ],
                    ..Options::default()
                },
            )
            .unwrap();
        assert_eq!(
            artifact.bindings,
            vec![
                deko_dksh::Binding {
                    group: 0,
                    binding: 1,
                    target: 0,
                    kind: deko_dksh::BindingKind::Texture,
                },
                deko_dksh::Binding {
                    group: 0,
                    binding: 0,
                    target: 0,
                    kind: deko_dksh::BindingKind::Sampler,
                },
            ]
        );
    }

    #[test]
    fn cross_determinant_integer_clamp_and_signed_negate_compile() {
        let source = r"
            const OFFSETS: vec3<u32> = vec3<u32>(4u);

            @vertex
            fn main(@location(0) position: vec3<f32>, @location(1) value: i32) -> @builtin(position) vec4<f32> {
                let basis = mat3x3<f32>(
                    vec3<f32>(1.0, 2.0, 3.0),
                    vec3<f32>(0.0, 1.0, 4.0),
                    vec3<f32>(5.0, 6.0, 0.0),
                );
                let normal = cross(position, basis[0]);
                let bounded = clamp(-value, -8, 8);
                return vec4<f32>(normal * determinant(basis), f32(bounded + i32(OFFSETS.x)));
            }
        ";
        Compiler
            .compile_wgsl(
                source,
                Stage::Vertex,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
    }

    #[test]
    fn bevy_mip_compute_primitives_compile_together() {
        let source = r"
            @group(0) @binding(0)
            var mip: texture_storage_2d_array<rgba16float, read_write>;

            fn write_mip(index: u32, value: vec4<f32>) {
                var values: array<vec4<f32>, 4>;
                values[index & 3u] = value;
                let selected = values[index & 3u];
                switch index & 1u {
                    case 0u: { textureStore(mip, vec2<u32>(index, 0u), 0u, selected); }
                    default: { textureStore(mip, vec2<u32>(0u, index), 0u, selected); }
                }
            }

            @compute @workgroup_size(4)
            fn main(@builtin(local_invocation_index) index: u32) {
                let loaded = textureLoad(mip, vec2<u32>(0u), 0u);
                let shuffled = quadSwapX(loaded);
                let bits = insertBits(extractBits(index, 1u, 3u), index, 0u, 1u);
                let unsigned_math = (bits / 3u) + (bits % 3u);
                let signed = i32(index) - 2;
                let signed_math = (signed / 3) + (signed % 3);
                write_mip(index, shuffled + vec4<f32>(f32(unsigned_math + u32(signed_math + 3))));
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
        assert_eq!(artifact.bindings.len(), 1);
        assert_eq!(
            artifact.bindings[0].kind,
            deko_dksh::BindingKind::StorageTexture
        );
        assert!(deko_dksh::parse(&artifact.dksh).unwrap().code.len() > 0x80);
    }

    #[test]
    fn gather_offsets_packed_formats_and_storage_queries_compile() {
        let source = r"
            @group(0) @binding(0) var input: texture_2d<u32>;
            @group(0) @binding(1) var input_sampler: sampler;
            @group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;

            @compute @workgroup_size(1)
            fn main() {
                let gathered = textureGather(0, input, input_sampler, vec2<f32>(0.5), vec2<i32>(1, -1));
                let packed = reverseBits(gathered.x) ^ countLeadingZeros(gathered.y) ^ countTrailingZeros(gathered.z);
                let color = unpack4x8unorm(packed);
                let size = textureDimensions(output);
                textureStore(output, min(size - vec2<u32>(1u), vec2<u32>(0u)), color);
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
        assert!(
            artifact
                .bindings
                .iter()
                .any(|binding| binding.kind == deko_dksh::BindingKind::StorageTexture)
        );
    }

    #[test]
    fn void_helpers_and_fragment_discard_compile() {
        let source = r"
            fn reject_negative(value: f32) {
                if value < 0.0 { discard; }
            }

            @fragment
            fn main(@location(0) value: f32) -> @location(0) vec4<f32> {
                reject_negative(value);
                return vec4<f32>(value);
            }
        ";
        Compiler
            .compile_wgsl(
                source,
                Stage::Fragment,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
    }

    #[test]
    fn pointer_arguments_write_back_nested_struct_members() {
        let source = r"
            struct Material { color: vec4<f32>, roughness: f32 }
            struct Input { material: Material, normal: vec3<f32> }

            fn update(input: ptr<function, Input>) {
                (*input).material.color = vec4<f32>(0.25);
                (*input).material.roughness = 0.5;
                (*input).normal = normalize(vec3<f32>(1.0, 2.0, 3.0));
            }

            @fragment
            fn main() -> @location(0) vec4<f32> {
                var input: Input;
                update(&input);
                return input.material.color + vec4<f32>(input.normal * input.material.roughness, 0.0);
            }
        ";
        Compiler
            .compile_wgsl(
                source,
                Stage::Fragment,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
    }

    #[test]
    fn divergent_pointer_argument_writes_merge_per_invocation() {
        let source = r"
            @group(0) @binding(0) var<storage, read_write> output: array<u32>;

            fn choose(destination: ptr<function, u32>, lane: u32) {
                if lane == 0u {
                    *destination = 11u;
                    return;
                } else {
                    *destination = 22u;
                }
            }

            @compute @workgroup_size(4)
            fn main(@builtin(local_invocation_index) lane: u32) {
                var value = 0u;
                choose(&value, lane);
                output[lane] = value;
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
        assert!(artifact.dksh.starts_with(b"DKSH"));

        let ir = lowered_ir(source, naga::ShaderStage::Compute, "main");
        assert!(ir.contains("sel"), "{ir}");
        assert_eq!(ir.matches("st.global").count(), 1, "{ir}");
    }

    #[test]
    fn value_helpers_write_back_pointer_arguments() {
        let source = r"
            @group(0) @binding(0) var<storage, read_write> output: array<u32>;

            fn choose(destination: ptr<function, u32>, lane: u32) -> u32 {
                if lane == 0u {
                    *destination = 11u;
                    return 1u;
                }
                *destination = 22u;
                return 2u;
            }

            @compute @workgroup_size(4)
            fn main(@builtin(local_invocation_index) lane: u32) {
                var value = 0u;
                let selected = choose(&value, lane);
                output[lane] = value + selected;
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
        assert!(artifact.dksh.starts_with(b"DKSH"));

        let ir = lowered_ir(source, naga::ShaderStage::Compute, "main");
        assert!(ir.matches("sel").count() >= 2, "{ir}");
        assert_eq!(ir.matches("st.global").count(), 1, "{ir}");
    }

    #[test]
    fn multi_selector_switch_cases_compile() {
        let source = r"
            @group(0) @binding(0) var<storage, read_write> output: array<u32>;

            @compute @workgroup_size(4)
            fn main(@builtin(local_invocation_index) lane: u32) {
                switch lane {
                    case 0u, 1u: {
                        output[lane] = 10u;
                    }
                    default: {
                        output[lane] = 20u;
                    }
                }
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
        assert!(artifact.dksh.starts_with(b"DKSH"));

        let ir = lowered_ir(source, naga::ShaderStage::Compute, "main");
        let stores = ir
            .lines()
            .filter(|line| line.contains("st.global"))
            .collect::<Vec<_>>();
        assert_eq!(stores.len(), 2, "{ir}");
        assert!(stores.iter().all(|line| line.contains('@')), "{ir}");
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
    fn resource_arguments_texture_queries_and_exact_offsets_compile() {
        let source = r"
            @group(0) @binding(0) var image: texture_2d<f32>;
            @group(0) @binding(1) var image_sampler: sampler;
            @group(0) @binding(2) var multisampled: texture_multisampled_2d<f32>;
            @group(0) @binding(3) var array_image: texture_2d_array<f32>;
            @group(0) @binding(4) var cube_array: texture_cube_array<f32>;

            fn dimensions(value: texture_2d<f32>) -> vec2<u32> {
                return textureDimensions(value, 0);
            }

            @fragment
            fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
                let size = dimensions(image);
                let samples = textureNumSamples(multisampled);
                let layers = textureNumLayers(array_image) + textureNumLayers(cube_array);
                let color = textureSampleLevel(image, image_sampler, uv, 0.0, vec2<i32>(1, 0));
                return color + vec4<f32>(f32(size.x + samples + layers));
            }
        ";
        Compiler
            .compile_wgsl(
                source,
                Stage::Fragment,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
    }

    #[test]
    fn atan_float_modulo_and_screen_derivatives_compile() {
        let source = r"
            @fragment
            fn main(@location(0) value: vec2<f32>) -> @location(0) vec4<f32> {
                let angle = atan2(value.y, value.x) + atan(value.x);
                let derivatives = vec3<f32>(dpdx(value.x), dpdy(value.y), fwidth(value.x));
                return vec4<f32>(derivatives % vec3<f32>(2.0), angle);
            }
        ";
        Compiler
            .compile_wgsl(
                source,
                Stage::Fragment,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
    }

    #[test]
    fn conditional_continue_with_remaining_loop_body_compiles() {
        let source = r"
            @compute @workgroup_size(1)
            fn main() {
                var total = 0u;
                for (var index = 0u; index < 8u; index += 1u) {
                    if (index & 1u) == 0u {
                        total += index;
                        continue;
                    }
                    total += index * 2u;
                }
                if total == 0xffffffffu { return; }
            }
        ";
        Compiler
            .compile_wgsl(
                source,
                Stage::Compute,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
    }

    #[test]
    fn loop_continues_and_early_struct_return_compile() {
        let source = r"
            struct Result { value: f32, found: bool }

            fn search(limit: u32) -> Result {
                var result: Result;
                result.value = 0.0;
                result.found = false;
                for (var index = 0u; index < limit; index += 1u) {
                    if (index & 1u) == 0u { continue; }
                    let weight = f32(index) * 0.25;
                    if weight == 0.0 { continue; }
                    result.value = weight;
                    result.found = true;
                    return result;
                }
                return result;
            }

            @fragment
            fn main() -> @location(0) vec4<f32> {
                let result = search(8u);
                return vec4<f32>(result.value * f32(result.found));
            }
        ";
        Compiler
            .compile_wgsl(
                source,
                Stage::Fragment,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
    }

    #[test]
    fn pipeline_overrides_and_multiview_are_lowered() {
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

        let mut workgroup_constants = PipelineConstants::new();
        workgroup_constants.insert("width".to_owned(), 8.0);
        let workgroup_artifact = Compiler
            .compile_wgsl(
                "override width: u32 = 1u; @compute @workgroup_size(width, 2, 1) fn main() {}",
                Stage::Compute,
                "main",
                &workgroup_constants,
                Options::default(),
            )
            .unwrap();
        assert!(matches!(
            deko_dksh::parse(&workgroup_artifact.dksh)
                .unwrap()
                .program
                .payload,
            deko_dksh::StagePayload::Compute {
                block_dimensions: [8, 2, 1],
                ..
            }
        ));

        let multiview_options = Options {
            multiview_mask: Some(0b101),
            ..Options::default()
        };
        let vertex_source = r"
            struct Output {
                @builtin(position) position: vec4<f32>,
                @location(0) view: u32,
            }
            @vertex fn main(@builtin(view_index) view: u32) -> Output {
                return Output(vec4<f32>(), view);
            }
        ";
        let vertex = Compiler
            .compile_wgsl(
                vertex_source,
                Stage::Vertex,
                "main",
                &PipelineConstants::new(),
                multiview_options.clone(),
            )
            .unwrap();
        assert!(vertex.dksh.starts_with(b"DKSH"));
        let vertex_ir = lowered_ir_with_options(
            vertex_source,
            naga::ShaderStage::Vertex,
            "main",
            &multiview_options,
        );
        assert!(vertex_ir.contains("c[0x10]"), "{vertex_ir}");
        assert!(vertex_ir.contains("a[0x64]"), "{vertex_ir}");

        let fragment_source = r"
            @fragment fn main(@builtin(view_index) view: u32) -> @location(0) vec4<f32> {
                return vec4<f32>(f32(view));
            }
        ";
        let fragment = Compiler
            .compile_wgsl(
                fragment_source,
                Stage::Fragment,
                "main",
                &PipelineConstants::new(),
                multiview_options.clone(),
            )
            .unwrap();
        assert!(fragment.dksh.starts_with(b"DKSH"));
        let fragment_ir = lowered_ir_with_options(
            fragment_source,
            naga::ShaderStage::Fragment,
            "main",
            &multiview_options,
        );
        assert!(fragment_ir.contains("ipa.constant"), "{fragment_ir}");
        assert!(fragment_ir.contains("a[0x64]"), "{fragment_ir}");
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn gradient_sampling_and_subgroup_barriers_compile() {
        let gradient_shader = Compiler
            .compile_wgsl(
                r"
                    @group(0) @binding(0) var image: texture_2d<f32>;
                    @group(0) @binding(1) var image_sampler: sampler;
                    @fragment fn main() -> @location(0) vec4<f32> {
                        return textureSampleGrad(
                            image,
                            image_sampler,
                            vec2<f32>(0.5),
                            vec2<f32>(1.0, 0.0),
                            vec2<f32>(0.0, 1.0),
                        );
                    }
                ",
                Stage::Fragment,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        assert!(gradient_shader.dksh.starts_with(b"DKSH"));

        Compiler
            .compile_wgsl(
                r"
                    @group(0) @binding(0) var image: texture_2d_array<f32>;
                    @group(0) @binding(1) var image_sampler: sampler;
                    @fragment fn main() -> @location(0) vec4<f32> {
                        return textureSampleGrad(
                            image,
                            image_sampler,
                            vec2<f32>(0.5),
                            2,
                            vec2<f32>(1.0, 0.0),
                            vec2<f32>(0.0, 1.0),
                            vec2<i32>(1, -1),
                        );
                    }
                ",
                Stage::Fragment,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();

        let gradient_3d_shader = Compiler
            .compile_wgsl(
                r"
                    @group(0) @binding(0) var image: texture_3d<f32>;
                    @group(0) @binding(1) var image_sampler: sampler;
                    @fragment fn main() -> @location(0) vec4<f32> {
                        return textureSampleGrad(
                            image,
                            image_sampler,
                            vec3<f32>(0.5),
                            vec3<f32>(1.0, 0.0, 0.0),
                            vec3<f32>(0.0, 1.0, 0.0),
                            vec3<i32>(1, 0, -1),
                        );
                    }
                ",
                Stage::Fragment,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        assert!(gradient_3d_shader.dksh.starts_with(b"DKSH"));

        let gradient_cube_shader = Compiler
            .compile_wgsl(
                r"
                    @group(0) @binding(0) var image: texture_cube<f32>;
                    @group(0) @binding(1) var image_sampler: sampler;
                    @fragment fn main() -> @location(0) vec4<f32> {
                        return textureSampleGrad(
                            image,
                            image_sampler,
                            normalize(vec3<f32>(0.5, 0.25, 1.0)),
                            vec3<f32>(1.0, 0.0, 0.0),
                            vec3<f32>(0.0, 1.0, 0.0),
                        );
                    }
                ",
                Stage::Fragment,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        assert!(gradient_cube_shader.dksh.starts_with(b"DKSH"));

        Compiler
            .compile_wgsl(
                r"
                    @group(0) @binding(0) var image: texture_cube_array<f32>;
                    @group(0) @binding(1) var image_sampler: sampler;
                    @fragment fn main() -> @location(0) vec4<f32> {
                        return textureSampleGrad(
                            image,
                            image_sampler,
                            normalize(vec3<f32>(0.5, 0.25, 1.0)),
                            2,
                            vec3<f32>(1.0, 0.0, 0.0),
                            vec3<f32>(0.0, 1.0, 0.0),
                        );
                    }
                ",
                Stage::Fragment,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();

        let subgroup_shader = Compiler
            .compile_wgsl(
                "@compute @workgroup_size(1) fn main() { subgroupBarrier(); }",
                Stage::Compute,
                "main",
                &PipelineConstants::new(),
                Options::default(),
            )
            .unwrap();
        assert!(subgroup_shader.dksh.starts_with(b"DKSH"));

        let subgroup_ir = lowered_ir(
            "@compute @workgroup_size(32) fn main() { subgroupBarrier(); }",
            naga::ShaderStage::Compute,
            "main",
        );
        assert!(subgroup_ir.contains("membar"), "{subgroup_ir}");
        assert!(subgroup_ir.contains(".cta"), "{subgroup_ir}");
        assert!(!subgroup_ir.contains("bar.sync"), "{subgroup_ir}");

        let broadcast_first_ir = lowered_ir(
            "@compute @workgroup_size(32) fn main(@builtin(local_invocation_index) lane: u32) { _ = subgroupBroadcastFirst(lane); }",
            naga::ShaderStage::Compute,
            "main",
        );
        assert!(
            broadcast_first_ir.contains("vote.any"),
            "{broadcast_first_ir}"
        );
        assert!(
            broadcast_first_ir.contains("shfl.idx"),
            "{broadcast_first_ir}"
        );

        let vote_ir = lowered_ir(
            "@compute @workgroup_size(32) fn main(@builtin(local_invocation_index) lane: u32) { let predicate = lane < 16u; _ = subgroupAll(predicate); _ = subgroupAny(predicate); _ = subgroupBallot(predicate); _ = subgroupBallot(); }",
            naga::ShaderStage::Compute,
            "main",
        );
        assert!(vote_ir.contains("vote.all"), "{vote_ir}");
        assert!(vote_ir.matches("vote.any").count() >= 3, "{vote_ir}");
    }

    #[test]
    fn subgroup_compute_builtins_compile() {
        let source = r"
            @compute @workgroup_size(40, 1, 1)
            fn main(
                @builtin(subgroup_invocation_id) lane: u32,
                @builtin(subgroup_size) size: u32,
                @builtin(subgroup_id) subgroup: u32,
                @builtin(num_subgroups) count: u32,
            ) {
                _ = lane + size + subgroup + count;
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
        assert!(artifact.dksh.starts_with(b"DKSH"));

        let ir = lowered_ir(source, naga::ShaderStage::Compute, "main");
        assert!(ir.contains("s2r"), "{ir}");
        assert!(ir.contains("shr"), "{ir}");
    }

    #[test]
    fn subgroup_arithmetic_collectives_compile() {
        let functions = [
            "subgroupAdd(value)",
            "subgroupMul(value)",
            "subgroupMin(value)",
            "subgroupMax(value)",
            "subgroupAnd(value)",
            "subgroupOr(value)",
            "subgroupXor(value)",
            "subgroupExclusiveAdd(value)",
            "subgroupInclusiveAdd(value)",
            "subgroupExclusiveMul(value)",
            "subgroupInclusiveMul(value)",
        ];
        for function in functions {
            let source = format!(
                "@compute @workgroup_size(40) fn main(@builtin(local_invocation_index) lane: u32) {{ let value = lane + 1u; _ = {function}; }}"
            );
            let artifact = Compiler
                .compile_wgsl(
                    &source,
                    Stage::Compute,
                    "main",
                    &PipelineConstants::new(),
                    Options::default(),
                )
                .unwrap_or_else(|error| panic!("{function}: {error}"));
            assert!(artifact.dksh.starts_with(b"DKSH"), "{function}");
        }

        for expression in [
            "subgroupAdd(value)",
            "subgroupMul(value)",
            "subgroupMin(value)",
            "subgroupMax(value)",
            "subgroupExclusiveAdd(value)",
            "subgroupInclusiveMul(value)",
        ] {
            let source = format!(
                "@compute @workgroup_size(7) fn main(@builtin(local_invocation_index) lane: u32) {{ let value = f32(lane) + 1.0; _ = {expression}; }}"
            );
            Compiler
                .compile_wgsl(
                    &source,
                    Stage::Compute,
                    "main",
                    &PipelineConstants::new(),
                    Options::default(),
                )
                .unwrap_or_else(|error| panic!("{expression}: {error}"));
        }

        for expression in [
            "subgroupAdd(vec2<u32>(lane + 1u, lane + 2u))",
            "subgroupMin(vec4<f32>(f32(lane), 2.0, 3.0, 4.0))",
            "subgroupInclusiveAdd(vec2<i32>(i32(lane), -1))",
        ] {
            let source = format!(
                "@compute @workgroup_size(7) fn main(@builtin(local_invocation_index) lane: u32) {{ _ = {expression}; }}"
            );
            Compiler
                .compile_wgsl(
                    &source,
                    Stage::Compute,
                    "main",
                    &PipelineConstants::new(),
                    Options::default(),
                )
                .unwrap_or_else(|error| panic!("{expression}: {error}"));
        }

        let reduction_ir = lowered_ir(
            "@compute @workgroup_size(7) fn main(@builtin(local_invocation_index) lane: u32) { _ = subgroupAdd(lane + 1u); }",
            naga::ShaderStage::Compute,
            "main",
        );
        assert!(reduction_ir.contains("vote.any"), "{reduction_ir}");
        assert!(reduction_ir.contains("shfl.idx"), "{reduction_ir}");

        let scan_ir = lowered_ir(
            "@compute @workgroup_size(7) fn main(@builtin(local_invocation_index) lane: u32) { _ = subgroupExclusiveAdd(lane + 1u); }",
            naga::ShaderStage::Compute,
            "main",
        );
        assert!(scan_ir.contains("s2r"), "{scan_ir}");
        assert!(scan_ir.contains("shfl.idx"), "{scan_ir}");
    }

    #[test]
    fn workgroup_uniform_load_compiles() {
        let source = r"
            struct Pair { first: u32, second: vec2<u32> }
            var<workgroup> wg_value: Pair;
            var<workgroup> atomic_value: atomic<u32>;

            @compute @workgroup_size(4)
            fn main(@builtin(local_invocation_index) lane: u32) {
                if lane == 0u {
                    wg_value.first = 42u;
                    wg_value.second = vec2<u32>(7u, 9u);
                    atomicStore(&atomic_value, 11u);
                }
                let pair = workgroupUniformLoad(&wg_value);
                let atomic_result = workgroupUniformLoad(&atomic_value);
                _ = pair.first + pair.second.x + atomic_result;
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
        assert!(artifact.dksh.starts_with(b"DKSH"));

        let ir = lowered_ir(source, naga::ShaderStage::Compute, "main");
        assert!(ir.matches("bar.sync").count() >= 4, "{ir}");
        assert!(ir.matches("membar").count() >= 4, "{ir}");
        assert!(ir.contains("ld.shared"), "{ir}");
        assert!(
            ir.lines()
                .filter(|line| line.contains("st.shared"))
                .all(|line| line.contains('@')),
            "{ir}"
        );
    }

    #[test]
    fn conditional_side_effects_are_predicated() {
        let source = r"
            struct Output { value: u32 }
            @group(0) @binding(0) var<storage, read_write> output: Output;

            @compute @workgroup_size(4)
            fn main(@builtin(local_invocation_index) lane: u32) {
                if lane == 0u {
                    output.value = 1u;
                } else {
                    output.value = 2u;
                }
            }
        ";
        let ir = lowered_ir(source, naga::ShaderStage::Compute, "main");
        let stores = ir
            .lines()
            .filter(|line| line.contains("st.global"))
            .collect::<Vec<_>>();
        assert_eq!(stores.len(), 2, "{ir}");
        assert!(stores.iter().all(|line| line.contains('@')), "{ir}");
        assert!(stores.iter().any(|line| line.contains('!')), "{ir}");
    }

    #[test]
    fn divergent_branch_with_loop_compiles() {
        let source = r"
            @group(0) @binding(0) var<storage, read_write> output: array<u32>;

            @compute @workgroup_size(4)
            fn main(@builtin(local_invocation_index) lane: u32) {
                var sum = 0u;
                if (lane & 1u) == 0u {
                    for (var index = 0u; index < 4u; index += 1u) {
                        sum += index;
                    }
                    output[lane] = sum;
                } else {
                    output[lane] = 99u;
                }
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
        assert!(artifact.dksh.starts_with(b"DKSH"));

        let ir = lowered_ir(source, naga::ShaderStage::Compute, "main");
        let stores = ir
            .lines()
            .filter(|line| line.contains("st.global"))
            .collect::<Vec<_>>();
        assert_eq!(stores.len(), 2, "{ir}");
        assert!(stores.iter().all(|line| line.contains('@')), "{ir}");
    }

    #[test]
    fn divergent_early_return_masks_later_side_effects() {
        let source = r"
            @group(0) @binding(0) var<storage, read_write> output: array<u32>;

            @compute @workgroup_size(4)
            fn main(@builtin(local_invocation_index) lane: u32) {
                if lane == 0u {
                    return;
                }
                output[lane] = 7u;
            }
        ";
        let ir = lowered_ir(source, naga::ShaderStage::Compute, "main");
        let stores = ir
            .lines()
            .filter(|line| line.contains("st.global"))
            .collect::<Vec<_>>();
        assert_eq!(stores.len(), 1, "{ir}");
        assert!(stores[0].contains('@'), "{ir}");
        assert!(stores[0].contains('!'), "{ir}");
    }

    #[test]
    fn loop_early_return_masks_side_effects_after_the_loop() {
        let source = r"
            @group(0) @binding(0) var<storage, read_write> output: array<u32>;

            fn write_if_not_returned(lane: u32) {
                var iteration = 0u;
                loop {
                    if iteration == lane {
                        return;
                    }
                    iteration += 1u;
                    if iteration == 2u {
                        break;
                    }
                }
                output[lane] = 99u;
            }

            @compute @workgroup_size(4)
            fn main(@builtin(local_invocation_index) lane: u32) {
                write_if_not_returned(lane);
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
        assert!(artifact.dksh.starts_with(b"DKSH"));

        let ir = lowered_ir(source, naga::ShaderStage::Compute, "main");
        let stores = ir
            .lines()
            .filter(|line| line.contains("st.global"))
            .collect::<Vec<_>>();
        assert_eq!(stores.len(), 1, "{ir}");
        assert!(stores[0].contains('@'), "{ir}");
        assert!(stores[0].contains('!'), "{ir}");
    }

    #[test]
    fn nested_early_return_preserves_only_live_invocations() {
        let source = r"
            @group(0) @binding(0) var<storage, read_write> output: array<u32>;

            @compute @workgroup_size(4)
            fn main(@builtin(local_invocation_index) lane: u32) {
                if lane < 2u {
                    if lane == 0u {
                        return;
                    }
                }
                output[lane] = 9u;
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
        assert!(artifact.dksh.starts_with(b"DKSH"));

        let ir = lowered_ir(source, naga::ShaderStage::Compute, "main");
        let store = ir
            .lines()
            .find(|line| line.contains("st.global"))
            .unwrap_or_else(|| panic!("{ir}"));
        assert!(store.contains('@'), "{ir}");
    }

    #[test]
    fn sequential_value_returns_keep_prior_return_lanes() {
        let source = r"
            @group(0) @binding(0) var<storage, read_write> output: array<u32>;

            fn choose(lane: u32) -> u32 {
                if lane == 0u {
                    return 11u;
                }
                if lane == 1u {
                    return 22u;
                }
                return 33u;
            }

            @compute @workgroup_size(4)
            fn main(@builtin(local_invocation_index) lane: u32) {
                output[lane] = choose(lane);
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
        assert!(artifact.dksh.starts_with(b"DKSH"));

        let ir = lowered_ir(source, naga::ShaderStage::Compute, "main");
        assert!(ir.matches("sel").count() >= 2, "{ir}");
    }

    #[test]
    fn gradient_lowering_selects_native_and_rewritten_paths() {
        let two_dimensional = lowered_ir(
            r"
                @group(0) @binding(0) var image: texture_2d<f32>;
                @group(0) @binding(1) var image_sampler: sampler;
                @fragment fn main() -> @location(0) vec4<f32> {
                    return textureSampleGrad(image, image_sampler, vec2<f32>(0.5),
                        vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));
                }
            ",
            naga::ShaderStage::Fragment,
            "main",
        );
        assert!(two_dimensional.contains("txd"), "{two_dimensional}");

        let three_dimensional = lowered_ir(
            r"
                @group(0) @binding(0) var image: texture_3d<f32>;
                @group(0) @binding(1) var image_sampler: sampler;
                @fragment fn main() -> @location(0) vec4<f32> {
                    return textureSampleGrad(image, image_sampler, vec3<f32>(0.5),
                        vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0));
                }
            ",
            naga::ShaderStage::Fragment,
            "main",
        );
        assert!(three_dimensional.contains("txq"), "{three_dimensional}");
        assert!(three_dimensional.contains(".ll"), "{three_dimensional}");
        assert!(!three_dimensional.contains("txd"), "{three_dimensional}");

        let cube = lowered_ir(
            r"
                @group(0) @binding(0) var image: texture_cube<f32>;
                @group(0) @binding(1) var image_sampler: sampler;
                @fragment fn main() -> @location(0) vec4<f32> {
                    return textureSampleGrad(image, image_sampler, vec3<f32>(0.5, 0.25, 1.0),
                        vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0));
                }
            ",
            naga::ShaderStage::Fragment,
            "main",
        );
        assert!(cube.contains("txq"), "{cube}");
        assert!(cube.contains(".ll"), "{cube}");
        assert!(!cube.contains("txd"), "{cube}");
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
