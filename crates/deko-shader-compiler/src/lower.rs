use deko_nak::ir::{
    AtomCmpSrc, AtomOp, AtomType, BasicBlock, CBuf, CBufRef, ChannelMask, Dst, FRndMode, FSwzAddOp,
    FloatCmpOp, FloatType, Function, HasRegFile, ImageAccess, ImageDim, Instr, IntCmpOp,
    IntCmpType, IntType, InterpFreq, InterpLoc, Label, LabelAllocator, LdcMode, LogicOp2,
    MemAccess, MemAddrType, MemEvictionPriority, MemOrder, MemScope, MemSpace, MemType, MuFuOp,
    OffsetStride, Op, OpALd, OpASt, OpAtom, OpBar, OpBfe, OpBrk, OpCont, OpExit, OpF2I, OpFAdd,
    OpFMnMx, OpFMul, OpFSetP, OpFSwzAdd, OpFlo, OpI2F, OpIAdd2, OpIAdd2X, OpIMad, OpIMnMx, OpIMul,
    OpISetP, OpIpa, OpKill, OpLd, OpLdc, OpLop2, OpMemBar, OpMov, OpMuFu, OpPBk, OpPCnt, OpPSetP,
    OpPhiDsts, OpPhiSrcs, OpPrmt, OpRegOut, OpS2R, OpSel, OpShfl, OpShl, OpShr, OpSt, OpSuAtom,
    OpSuLd, OpSuSt, OpTex, OpTld, OpTld4, OpTxd, OpTxq, OpVote, Phi, Pred, PredRef, PredSetOp,
    PrmtMode, RegFile, SSARef, SSAValue, Shader, ShaderInfo, ShaderIoInfo, ShaderModelInfo,
    ShaderStageInfo, ShflOp, Src, SrcMod, SrcRef, SrcSwizzle, TexDerivMode, TexDim, TexLodMode,
    TexOffsetMode, TexQuery, TexRef, VoteOp,
};
use deko_nak::sph::PixelImap;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::Error;

// Deko reserves user uniform target 14 for wgpu's emulated multiview index and target 15
// for immediate data. Maxwell's shader-visible constant-buffer numbering adds c0/c1.
const MULTIVIEW_UNIFORM_TARGET: u8 = 14;
const LAYER_ATTRIBUTE_ADDRESS: u16 = 0x064;

pub(crate) struct LoweredShader<'sm> {
    pub shader: Shader<'sm>,
    pub bindings: Vec<deko_dksh::Binding>,
}

#[derive(Default)]
struct ResourceMap {
    uniforms: HashMap<naga::Handle<naga::GlobalVariable>, u8>,
    storages: HashMap<naga::Handle<naga::GlobalVariable>, u8>,
    storage_descriptor_base: u16,
    workgroups: HashMap<naga::Handle<naga::GlobalVariable>, u32>,
    workgroup_memory_size: u32,
    textures: HashMap<naga::Handle<naga::GlobalVariable>, ResourceRange>,
    samplers: HashMap<naga::Handle<naga::GlobalVariable>, ResourceRange>,
    storage_textures: HashMap<naga::Handle<naga::GlobalVariable>, u16>,
    bindings: Vec<deko_dksh::Binding>,
}

#[derive(Clone, Copy)]
struct ResourceRange {
    target: u16,
    count: u16,
}

type DynamicLocalPointer = (
    naga::Handle<naga::LocalVariable>,
    naga::Handle<naga::Type>,
    usize,
    usize,
    Value,
);

pub(crate) fn entry_point<'module>(
    module: &'module naga::Module,
    stage: naga::ShaderStage,
    name: &str,
) -> Option<&'module naga::EntryPoint> {
    module
        .entry_points
        .iter()
        .find(|entry| entry.stage == stage && entry.name == name)
}

pub(crate) fn lower_entry_point<'sm>(
    module: &naga::Module,
    entry: &naga::EntryPoint,
    sm: &'sm ShaderModelInfo,
    options: &crate::Options,
) -> Result<LoweredShader<'sm>, Error> {
    let resources = resource_map(module, entry, options)?;
    let shader = match entry.stage {
        naga::ShaderStage::Compute => lower_compute(module, entry, sm, &resources),
        naga::ShaderStage::Vertex => lower_vertex(
            module,
            entry,
            sm,
            &resources,
            options.multiview_mask.is_some(),
        ),
        naga::ShaderStage::Fragment => lower_fragment(
            module,
            entry,
            sm,
            &resources,
            options.multiview_mask.is_some(),
        ),
        stage => Err(Error::UnsupportedFeature(format!("{stage:?} stage"))),
    }?;
    Ok(LoweredShader {
        shader,
        bindings: resources.bindings,
    })
}

#[allow(clippy::too_many_lines)]
fn resource_map(
    module: &naga::Module,
    entry: &naga::EntryPoint,
    options: &crate::Options,
) -> Result<ResourceMap, Error> {
    let reachable_functions = reachable_functions(module, &entry.function);
    let used_globals = used_globals(module, &entry.function, &reachable_functions);
    let stage = entry.stage;
    // Deko3D places 16-byte storage descriptors in its driver constant buffer. Graphics stages
    // share GraphicsDriverCbuf and compute uses the smaller ComputeDriverCbuf layout.
    let storage_descriptor_base = match stage {
        naga::ShaderStage::Vertex => 0x0b0,
        naga::ShaderStage::Fragment => 0x730,
        naga::ShaderStage::Compute => 0x140,
        stage => {
            return Err(Error::UnsupportedFeature(format!(
                "storage descriptor ABI for {stage:?} stage"
            )));
        }
    };
    let mut resources = ResourceMap {
        storage_descriptor_base,
        ..ResourceMap::default()
    };
    let mut layouter = naga::proc::Layouter::default();
    layouter
        .update(module.to_ctx())
        .map_err(|error| Error::UnsupportedFeature(format!("workgroup memory layout: {error}")))?;
    for (handle, variable) in module.global_variables.iter() {
        if !used_globals.contains(&handle) || variable.space != naga::AddressSpace::WorkGroup {
            continue;
        }
        let layout = layouter[variable.ty];
        resources.workgroup_memory_size =
            layout.alignment.round_up(resources.workgroup_memory_size);
        resources
            .workgroups
            .insert(handle, resources.workgroup_memory_size);
        resources.workgroup_memory_size = resources
            .workgroup_memory_size
            .checked_add(layout.size)
            .ok_or_else(|| {
                Error::UnsupportedFeature("workgroup memory size overflow".to_owned())
            })?;
    }
    let mut uniforms = module
        .global_variables
        .iter()
        .filter_map(|(handle, variable)| {
            (used_globals.contains(&handle) && variable.space == naga::AddressSpace::Uniform)
                .then_some((handle, variable.binding.as_ref()))
        })
        .collect::<Vec<_>>();
    uniforms.sort_by_key(|(_, binding)| binding.map(|binding| (binding.group, binding.binding)));
    for (target, (handle, binding)) in uniforms.into_iter().enumerate() {
        let binding = binding.ok_or_else(|| {
            Error::UnsupportedFeature("uniform global without a resource binding".to_owned())
        })?;
        let target = u8::try_from(target)
            .map_err(|_| Error::UnsupportedFeature("too many uniform buffers".to_owned()))?;
        if target >= 14 {
            return Err(Error::UnsupportedFeature(
                "more than 14 user uniform buffers".to_owned(),
            ));
        }
        resources.uniforms.insert(handle, target);
        resources.bindings.push(deko_dksh::Binding {
            group: binding.group,
            binding: binding.binding,
            target: u32::from(target),
            kind: deko_dksh::BindingKind::Uniform,
        });
    }
    let mut storages = module
        .global_variables
        .iter()
        .filter_map(|(handle, variable)| {
            (used_globals.contains(&handle)
                && matches!(variable.space, naga::AddressSpace::Storage { .. }))
            .then_some((handle, variable.binding.as_ref()))
        })
        .collect::<Vec<_>>();
    storages.sort_by_key(|(_, binding)| binding.map(|binding| (binding.group, binding.binding)));
    for (target, (handle, binding)) in storages.into_iter().enumerate() {
        let binding = binding.ok_or_else(|| {
            Error::UnsupportedFeature("storage global without a resource binding".to_owned())
        })?;
        let target = u8::try_from(target)
            .map_err(|_| Error::UnsupportedFeature("too many storage buffers".to_owned()))?;
        if target >= 16 {
            return Err(Error::UnsupportedFeature(
                "more than 16 storage buffers".to_owned(),
            ));
        }
        resources.storages.insert(handle, target);
        resources.bindings.push(deko_dksh::Binding {
            group: binding.group,
            binding: binding.binding,
            target: u32::from(target),
            kind: deko_dksh::BindingKind::Storage,
        });
    }
    let mut textures = module
        .global_variables
        .iter()
        .filter_map(|(handle, variable)| {
            (used_globals.contains(&handle)
                && sampled_image_count(module, handle, options).is_some())
            .then_some((handle, variable.binding.as_ref()))
        })
        .collect::<Vec<_>>();
    textures.sort_by_key(|(_, binding)| binding.map(|binding| (binding.group, binding.binding)));
    let mut texture_targets = HashMap::<(u32, u32), ResourceRange>::default();
    let mut next_image_target = 0_u16;
    for (handle, binding) in textures {
        let count = sampled_image_count(module, handle, options).expect("filtered above");
        let binding = binding.ok_or_else(|| {
            Error::UnsupportedFeature("texture global without a resource binding".to_owned())
        })?;
        let key = (binding.group, binding.binding);
        let (range, first_alias) = if let Some(range) = texture_targets.get(&key) {
            (*range, false)
        } else {
            let range = allocate_resource_range(&mut next_image_target, count, "sampled textures")?;
            texture_targets.insert(key, range);
            (range, true)
        };
        resources.textures.insert(handle, range);
        if first_alias {
            resources.bindings.push(deko_dksh::Binding {
                group: binding.group,
                binding: binding.binding,
                target: u32::from(range.target),
                kind: deko_dksh::BindingKind::Texture,
            });
        }
    }
    let mut samplers = module
        .global_variables
        .iter()
        .filter_map(|(handle, variable)| {
            (used_globals.contains(&handle)
                && sampler_binding_count(module, handle, options).is_some())
            .then_some((handle, variable.binding.as_ref()))
        })
        .collect::<Vec<_>>();
    samplers.sort_by_key(|(_, binding)| binding.map(|binding| (binding.group, binding.binding)));
    let mut sampler_targets = HashMap::<(u32, u32), ResourceRange>::default();
    let mut next_sampler_target = 0_u16;
    for (handle, binding) in samplers {
        let count = sampler_binding_count(module, handle, options).expect("filtered above");
        let binding = binding.ok_or_else(|| {
            Error::UnsupportedFeature("sampler global without a resource binding".to_owned())
        })?;
        let key = (binding.group, binding.binding);
        let (range, first_alias) = if let Some(range) = sampler_targets.get(&key) {
            (*range, false)
        } else {
            let range = allocate_resource_range(&mut next_sampler_target, count, "samplers")?;
            sampler_targets.insert(key, range);
            (range, true)
        };
        resources.samplers.insert(handle, range);
        if first_alias {
            resources.bindings.push(deko_dksh::Binding {
                group: binding.group,
                binding: binding.binding,
                target: u32::from(range.target),
                kind: deko_dksh::BindingKind::Sampler,
            });
        }
    }
    let mut storage_texture_targets = HashMap::default();
    for (handle, variable) in module.global_variables.iter() {
        if !used_globals.contains(&handle)
            || !matches!(
                module.types[variable.ty].inner,
                naga::TypeInner::Image {
                    class: naga::ImageClass::Storage { .. },
                    ..
                }
            )
        {
            continue;
        }
        let binding = variable.binding.as_ref().ok_or_else(|| {
            Error::UnsupportedFeature("storage texture without a resource binding".to_owned())
        })?;
        let key = (binding.group, binding.binding);
        let next_target = u16::try_from(storage_texture_targets.len())
            .map_err(|_| Error::UnsupportedFeature("too many storage textures".to_owned()))?;
        if next_target >= 64 {
            return Err(Error::UnsupportedFeature(
                "more than 64 storage textures".to_owned(),
            ));
        }
        let (target, first_alias) = if let Some(target) = storage_texture_targets.get(&key) {
            (*target, false)
        } else {
            storage_texture_targets.insert(key, next_target);
            (next_target, true)
        };
        resources.storage_textures.insert(handle, target);
        if first_alias {
            resources.bindings.push(deko_dksh::Binding {
                group: binding.group,
                binding: binding.binding,
                target: u32::from(target),
                kind: deko_dksh::BindingKind::StorageTexture,
            });
        }
    }
    Ok(resources)
}

fn binding_resource_type(
    module: &naga::Module,
    global: naga::Handle<naga::GlobalVariable>,
    options: &crate::Options,
) -> Result<(naga::Handle<naga::Type>, u16), Error> {
    let ty = module.global_variables[global].ty;
    let naga::TypeInner::BindingArray { base, size } = module.types[ty].inner else {
        return Ok((ty, 1));
    };
    let count = match size {
        naga::ArraySize::Constant(size) => u16::try_from(size.get()).map_err(|_| {
            Error::UnsupportedFeature("resource binding array exceeds u16".to_owned())
        })?,
        naga::ArraySize::Dynamic => {
            let binding = module.global_variables[global]
                .binding
                .as_ref()
                .ok_or_else(|| {
                    Error::UnsupportedFeature("resource binding array without a binding".to_owned())
                })?;
            let count = options
                .binding_array_sizes
                .iter()
                .rev()
                .find(|size| size.group == binding.group && size.binding == binding.binding)
                .ok_or_else(|| {
                    Error::UnsupportedFeature(format!(
                        "runtime-sized binding array @group({}) @binding({}) requires its pipeline-layout descriptor count",
                        binding.group, binding.binding
                    ))
                })?
                .count;
            u16::try_from(count).map_err(|_| {
                Error::UnsupportedFeature("resource binding array exceeds u16".to_owned())
            })?
        }
        naga::ArraySize::Pending(_) => {
            return Err(Error::UnsupportedFeature(
                "unresolved resource binding-array size".to_owned(),
            ));
        }
    };
    Ok((base, count))
}

fn binding_resource_base_type(
    module: &naga::Module,
    global: naga::Handle<naga::GlobalVariable>,
) -> naga::Handle<naga::Type> {
    let ty = module.global_variables[global].ty;
    match module.types[ty].inner {
        naga::TypeInner::BindingArray { base, .. } => base,
        _ => ty,
    }
}

fn sampled_image_count(
    module: &naga::Module,
    global: naga::Handle<naga::GlobalVariable>,
    options: &crate::Options,
) -> Option<u16> {
    let (ty, count) = binding_resource_type(module, global, options).ok()?;
    matches!(
        module.types[ty].inner,
        naga::TypeInner::Image {
            class: naga::ImageClass::Sampled { .. } | naga::ImageClass::Depth { .. },
            ..
        }
    )
    .then_some(count)
}

fn sampler_binding_count(
    module: &naga::Module,
    global: naga::Handle<naga::GlobalVariable>,
    options: &crate::Options,
) -> Option<u16> {
    let (ty, count) = binding_resource_type(module, global, options).ok()?;
    matches!(module.types[ty].inner, naga::TypeInner::Sampler { .. }).then_some(count)
}

fn allocate_resource_range(
    next: &mut u16,
    count: u16,
    description: &str,
) -> Result<ResourceRange, Error> {
    let end = next
        .checked_add(count)
        .ok_or_else(|| Error::UnsupportedFeature(format!("too many {description}")))?;
    if end > 64 {
        return Err(Error::UnsupportedFeature(format!(
            "more than 64 {description}"
        )));
    }
    let range = ResourceRange {
        target: *next,
        count,
    };
    *next = end;
    Ok(range)
}

fn reachable_functions(
    module: &naga::Module,
    entry: &naga::Function,
) -> HashSet<naga::Handle<naga::Function>> {
    fn visit(
        module: &naga::Module,
        block: &naga::Block,
        reachable: &mut HashSet<naga::Handle<naga::Function>>,
    ) {
        for statement in block {
            match statement {
                naga::Statement::Call { function, .. } => {
                    if reachable.insert(*function) {
                        visit(module, &module.functions[*function].body, reachable);
                    }
                }
                naga::Statement::Block(block) => visit(module, block, reachable),
                naga::Statement::If { accept, reject, .. } => {
                    visit(module, accept, reachable);
                    visit(module, reject, reachable);
                }
                naga::Statement::Switch { cases, .. } => {
                    for case in cases {
                        visit(module, &case.body, reachable);
                    }
                }
                naga::Statement::Loop {
                    body, continuing, ..
                } => {
                    visit(module, body, reachable);
                    visit(module, continuing, reachable);
                }
                _ => {}
            }
        }
    }

    let mut reachable = HashSet::default();
    visit(module, &entry.body, &mut reachable);
    reachable
}

fn used_globals(
    module: &naga::Module,
    entry: &naga::Function,
    reachable_functions: &HashSet<naga::Handle<naga::Function>>,
) -> HashSet<naga::Handle<naga::GlobalVariable>> {
    let mut globals = HashSet::default();
    let mut collect = |function: &naga::Function| {
        globals.extend(function.expressions.iter().filter_map(|(_, expression)| {
            if let naga::Expression::GlobalVariable(global) = expression {
                Some(*global)
            } else {
                None
            }
        }));
    };
    collect(entry);
    for function in reachable_functions {
        collect(&module.functions[*function]);
    }
    globals
}

fn lower_compute<'sm>(
    module: &naga::Module,
    entry: &naga::EntryPoint,
    sm: &'sm ShaderModelInfo,
    resources: &ResourceMap,
) -> Result<Shader<'sm>, Error> {
    if entry.function.result.is_some() {
        return Err(Error::UnsupportedFeature(
            "compute entry-point return values".to_owned(),
        ));
    }
    let [x, y, z] = entry.workgroup_size;
    let dimension = |value| {
        u16::try_from(value)
            .map_err(|_| Error::UnsupportedFeature("workgroup dimension exceeds u16".to_owned()))
    };
    let local_size = [dimension(x)?, dimension(y)?, dimension(z)?];
    let shared_memory_size = u16::try_from(resources.workgroup_memory_size).map_err(|_| {
        Error::UnsupportedFeature("workgroup memory exceeds Maxwell limit".to_owned())
    })?;

    let mut lowerer = FunctionLowerer::new(module, &entry.function, resources, Vec::new());
    bind_compute_arguments(module, &mut lowerer, entry.workgroup_size)?;
    let body = entry.function.body.clone();
    let returned = lowerer
        .execute_statements(&body)?
        .unwrap_or_else(Value::void);
    if !lowerer.finalize_return(returned)?.is_void() {
        return Err(Error::UnsupportedFeature(
            "compute entry point returned a value".to_owned(),
        ));
    }
    Ok(Shader {
        sm,
        info: ShaderInfo::compute(local_size, shared_memory_size),
        functions: vec![lowerer.finish()],
    })
}

fn bind_compute_arguments(
    module: &naga::Module,
    lowerer: &mut FunctionLowerer<'_>,
    workgroup_size: [u32; 3],
) -> Result<(), Error> {
    let arguments = lowerer.source.arguments.clone();
    for argument in &arguments {
        let fields = if let Some(binding) = &argument.binding {
            vec![(binding.clone(), argument.ty)]
        } else if let naga::TypeInner::Struct { ref members, .. } = module.types[argument.ty].inner
        {
            members
                .iter()
                .map(|member| {
                    Ok((
                        member.binding.clone().ok_or_else(|| {
                            Error::UnsupportedFeature(
                                "unbound compute input struct member".to_owned(),
                            )
                        })?,
                        member.ty,
                    ))
                })
                .collect::<Result<Vec<_>, Error>>()?
        } else {
            return Err(Error::UnsupportedFeature(
                "unbound non-struct compute argument".to_owned(),
            ));
        };
        let mut components = Vec::new();
        for (binding, ty) in fields {
            let naga::Binding::BuiltIn(builtin) = binding else {
                return Err(Error::UnsupportedFeature(format!(
                    "compute input binding {binding:?}"
                )));
            };
            let value = bind_compute_builtin(module, lowerer, builtin, ty, workgroup_size)?;
            components.extend(value.components);
        }
        lowerer.arguments.push(Value {
            components,
            kind: naga::ScalarKind::Uint,
        });
    }
    Ok(())
}

fn read_compute_system_value(lowerer: &mut FunctionLowerer<'_>, indices: &[u8]) -> Vec<Src> {
    indices
        .iter()
        .map(|index| {
            let dst = lowerer.target.ssa_alloc.alloc(RegFile::GPR);
            lowerer.emit(Instr::new(OpS2R {
                dst: Dst::from(dst),
                idx: *index,
            }));
            Src::from(dst)
        })
        .collect()
}

fn local_invocation_index(lowerer: &mut FunctionLowerer<'_>, workgroup_size: [u32; 3]) -> Src {
    let local = read_compute_system_value(lowerer, &[32, 33, 34]);
    let xy = lowerer.target.ssa_alloc.alloc(RegFile::GPR);
    lowerer.emit(Instr::new(OpIMad {
        dst: Dst::from(xy),
        srcs: [
            local[1].clone(),
            Src::from(workgroup_size[0]),
            local[0].clone(),
        ],
        signed: false,
    }));
    let index = lowerer.target.ssa_alloc.alloc(RegFile::GPR);
    lowerer.emit(Instr::new(OpIMad {
        dst: Dst::from(index),
        srcs: [
            local[2].clone(),
            Src::from(workgroup_size[0].saturating_mul(workgroup_size[1])),
            Src::from(xy),
        ],
        signed: false,
    }));
    Src::from(index)
}

fn bind_compute_builtin(
    module: &naga::Module,
    lowerer: &mut FunctionLowerer<'_>,
    builtin: naga::BuiltIn,
    ty: naga::Handle<naga::Type>,
    workgroup_size: [u32; 3],
) -> Result<Value, Error> {
    let (components, kind) = type_shape(module, ty)?;
    if kind != naga::ScalarKind::Uint {
        return Err(Error::UnsupportedFeature(format!(
            "compute builtin {builtin:?} must be unsigned integer"
        )));
    }
    let value = match builtin {
        naga::BuiltIn::LocalInvocationId if components == 3 => {
            read_compute_system_value(lowerer, &[32, 33, 34])
        }
        naga::BuiltIn::WorkGroupId if components == 3 => {
            read_compute_system_value(lowerer, &[37, 38, 39])
        }
        naga::BuiltIn::NumWorkGroups if components == 3 => {
            read_compute_system_value(lowerer, &[43, 44, 45])
        }
        naga::BuiltIn::WorkGroupSize if components == 3 => {
            workgroup_size.into_iter().map(Src::from).collect()
        }
        naga::BuiltIn::GlobalInvocationId if components == 3 => {
            let local = read_compute_system_value(lowerer, &[32, 33, 34]);
            let group = read_compute_system_value(lowerer, &[37, 38, 39]);
            let mut global = Vec::with_capacity(3);
            for component in 0..3 {
                let dst = lowerer.target.ssa_alloc.alloc(RegFile::GPR);
                lowerer.emit(Instr::new(OpIMad {
                    dst: Dst::from(dst),
                    srcs: [
                        group[component].clone(),
                        Src::from(workgroup_size[component]),
                        local[component].clone(),
                    ],
                    signed: false,
                }));
                global.push(Src::from(dst));
            }
            global
        }
        naga::BuiltIn::LocalInvocationIndex if components == 1 => {
            vec![local_invocation_index(lowerer, workgroup_size)]
        }
        naga::BuiltIn::SubgroupInvocationId if components == 1 => {
            read_compute_system_value(lowerer, &[0])
        }
        naga::BuiltIn::SubgroupSize if components == 1 => vec![Src::from(32_u32)],
        naga::BuiltIn::NumSubgroups if components == 1 => {
            let threads = workgroup_size.into_iter().fold(1_u32, u32::saturating_mul);
            vec![Src::from(threads.div_ceil(32))]
        }
        naga::BuiltIn::SubgroupId if components == 1 => {
            let local_index = local_invocation_index(lowerer, workgroup_size);
            let subgroup = lowerer.target.ssa_alloc.alloc(RegFile::GPR);
            lowerer.emit(Instr::new(OpShr {
                dst: Dst::from(subgroup),
                src: local_index,
                shift: Src::from(5_u32),
                signed: false,
                wrap: true,
            }));
            vec![Src::from(subgroup)]
        }
        _ => {
            return Err(Error::UnsupportedFeature(format!(
                "compute input builtin {builtin:?} with {components} components"
            )));
        }
    };
    Ok(Value {
        components: value,
        kind,
    })
}

#[derive(Clone, PartialEq)]
struct Value {
    components: Vec<Src>,
    kind: naga::ScalarKind,
}

impl Value {
    fn void() -> Self {
        Self {
            components: Vec::new(),
            kind: naga::ScalarKind::Uint,
        }
    }

    fn is_void(&self) -> bool {
        self.components.is_empty()
    }
}

type UniformPointer = (
    naga::Handle<naga::GlobalVariable>,
    naga::Handle<naga::Type>,
    u32,
    Option<Src>,
);

type StoragePointer = (
    naga::Handle<naga::GlobalVariable>,
    naga::Handle<naga::Type>,
    naga::StorageAccess,
    u32,
    Option<Src>,
);

type WorkgroupPointer = (
    naga::Handle<naga::GlobalVariable>,
    naga::Handle<naga::Type>,
    u32,
    Option<Src>,
);

struct LoopContext {
    exit_label: Label,
    continue_label: Label,
    carried_locals: Vec<naga::Handle<naga::LocalVariable>>,
    exit_locals: Vec<naga::Handle<naga::LocalVariable>>,
    entry_locals: HashMap<naga::Handle<naga::LocalVariable>, Value>,
    break_edges: Vec<LoopBreakEdge>,
    continue_edges: Vec<LoopContinueEdge>,
}

struct LoopBreakEdge {
    block: usize,
    returned: Option<Value>,
    locals: HashMap<naga::Handle<naga::LocalVariable>, Value>,
}

#[derive(Clone)]
struct LoopContinueEdge {
    block: usize,
    locals: HashMap<naga::Handle<naga::LocalVariable>, Value>,
}

struct LoopPhi {
    local: naga::Handle<naga::LocalVariable>,
    phis: Vec<Phi>,
    header_value: Value,
}

struct FunctionLowerer<'function> {
    module: &'function naga::Module,
    source: &'function naga::Function,
    resources: &'function ResourceMap,
    target: Function,
    blocks: Vec<BasicBlock>,
    edges: Vec<(usize, usize)>,
    current_block: usize,
    execution_predicate: Pred,
    labels: LabelAllocator,
    loops: Vec<LoopContext>,
    loop_base_depth: usize,
    values: HashMap<naga::Handle<naga::Expression>, Value>,
    locals: HashMap<naga::Handle<naga::LocalVariable>, Value>,
    arguments: Vec<Value>,
    resource_arguments: HashMap<u32, naga::Handle<naga::GlobalVariable>>,
    early_returns: Vec<(Src, Value)>,
}

impl<'function> FunctionLowerer<'function> {
    fn new(
        module: &'function naga::Module,
        source: &'function naga::Function,
        resources: &'function ResourceMap,
        arguments: Vec<Value>,
    ) -> Self {
        let mut labels = LabelAllocator::new();
        let entry = BasicBlock {
            label: labels.alloc(),
            uniform: false,
            instrs: Vec::new(),
        };
        Self {
            module,
            source,
            resources,
            target: Function::single_block(Vec::new()),
            blocks: vec![entry],
            edges: Vec::new(),
            current_block: 0,
            execution_predicate: true.into(),
            labels,
            loops: Vec::new(),
            loop_base_depth: 0,
            values: HashMap::default(),
            locals: HashMap::default(),
            arguments,
            resource_arguments: HashMap::default(),
            early_returns: Vec::new(),
        }
    }

    fn emit(&mut self, mut instruction: Instr) {
        let has_branch_side_effect = matches!(
            instruction.op,
            Op::St(_)
                | Op::StSCheckUnlock(_)
                | Op::Atom(_)
                | Op::ASt(_)
                | Op::SuSt(_)
                | Op::SuAtom(_)
                | Op::Kill(_)
        );
        if has_branch_side_effect && !self.execution_predicate.is_true() {
            instruction.pred = self.combine_predicates(self.execution_predicate, instruction.pred);
        }
        self.blocks[self.current_block].instrs.push(instruction);
    }

    fn combine_predicates(&mut self, left: Pred, right: Pred) -> Pred {
        if left.is_false() || right.is_false() {
            return false.into();
        }
        if left.is_true() {
            return right;
        }
        if right.is_true() {
            return left;
        }
        let destination = self.target.ssa_alloc.alloc(RegFile::Pred);
        self.blocks[self.current_block]
            .instrs
            .push(Instr::new(OpPSetP {
                dsts: [Dst::from(destination), Dst::None],
                ops: [PredSetOp::And, PredSetOp::And],
                srcs: [Src::from(left), true.into(), Src::from(right)],
            }));
        Pred::from(destination)
    }

    fn combine_predicates_or(&mut self, left: Pred, right: Pred) -> Pred {
        if left.is_true() || right.is_true() {
            return true.into();
        }
        if left.is_false() {
            return right;
        }
        if right.is_false() {
            return left;
        }
        let destination = self.target.ssa_alloc.alloc(RegFile::Pred);
        self.blocks[self.current_block]
            .instrs
            .push(Instr::new(OpPSetP {
                dsts: [Dst::from(destination), Dst::None],
                ops: [PredSetOp::And, PredSetOp::Or],
                srcs: [Src::from(left), true.into(), Src::from(right)],
            }));
        Pred::from(destination)
    }

    fn allocate_label(&mut self) -> Label {
        self.labels.alloc()
    }

    fn append_block(&mut self, label: Label) -> usize {
        let index = self.blocks.len();
        self.blocks.push(BasicBlock {
            label,
            uniform: false,
            instrs: Vec::new(),
        });
        index
    }

    fn add_edge(&mut self, from: usize, to: usize) {
        self.edges.push((from, to));
    }

    fn predicate(source: &Src) -> Result<Pred, Error> {
        if source.src_swizzle != SrcSwizzle::None {
            return Err(Error::UnsupportedFeature(
                "swizzled branch predicate".to_owned(),
            ));
        }
        let inverted = match source.src_mod {
            SrcMod::None => false,
            SrcMod::BNot => true,
            _ => {
                return Err(Error::UnsupportedFeature(
                    "non-boolean branch predicate modifier".to_owned(),
                ));
            }
        };
        let predicate = match &source.src_ref {
            SrcRef::True => Pred::from(true),
            SrcRef::False => Pred::from(false),
            SrcRef::SSA(ssa) if ssa.len() == 1 && ssa.is_predicate() => Pred {
                pred_ref: PredRef::SSA(ssa[0]),
                pred_inv: false,
            },
            _ => {
                return Err(Error::UnsupportedFeature(
                    "branch condition is not a predicate".to_owned(),
                ));
            }
        };
        Ok(if inverted {
            predicate.bnot()
        } else {
            predicate
        })
    }

    #[allow(clippy::too_many_lines)]
    fn expression(&mut self, handle: naga::Handle<naga::Expression>) -> Result<Value, Error> {
        if let Some(value) = self.values.get(&handle) {
            return Ok(value.clone());
        }
        let value = match &self.source.expressions[handle] {
            naga::Expression::Literal(literal) => Value {
                components: vec![literal_source(*literal)?],
                kind: literal_kind(*literal)?,
            },
            naga::Expression::Constant(constant) => {
                global_value(self.module, self.module.constants[*constant].init)?
            }
            naga::Expression::ZeroValue(ty) => zero_value(self.module, *ty)?,
            naga::Expression::Compose { ty, components } => {
                let mut flattened = Vec::new();
                let kind = flat_type_kind(self.module, *ty)?;
                for component in components {
                    let mut value = self.expression(*component)?;
                    if value.kind == naga::ScalarKind::Bool && kind != naga::ScalarKind::Bool {
                        value.components = self
                            .materialize_loop_components(&value)?
                            .into_iter()
                            .map(Src::from)
                            .collect();
                    }
                    flattened.extend(value.components);
                }
                Value {
                    components: flattened,
                    kind,
                }
            }
            naga::Expression::FunctionArgument(index) => self
                .arguments
                .get(*index as usize)
                .cloned()
                .ok_or_else(|| Error::UnsupportedFeature(format!("argument {index}")))?,
            naga::Expression::Load { pointer } => {
                if self.pointer_is_argument(*pointer) {
                    self.load_argument_pointer(*pointer)?
                } else if self.pointer_is_local(*pointer) {
                    self.load_local_pointer(*pointer)?
                } else if self.pointer_is_storage(*pointer) {
                    self.load_storage(*pointer)?
                } else if self.pointer_is_workgroup(*pointer) {
                    self.load_workgroup(*pointer)?
                } else {
                    self.load_uniform(*pointer)?
                }
            }
            naga::Expression::ArrayLength(pointer) => self.storage_array_length(*pointer)?,
            naga::Expression::Splat { size, value } => self.splat(*size, *value)?,
            naga::Expression::Swizzle {
                size,
                vector,
                pattern,
            } => self.swizzle(*size, *vector, *pattern)?,
            naga::Expression::Access { base, index } => self.access(*base, *index)?,
            naga::Expression::AccessIndex { base, index } => self.access_index(*base, *index)?,
            naga::Expression::Unary { op, expr } => self.unary(*op, *expr)?,
            naga::Expression::Binary { op, left, right } => {
                let left_matrix = self.expression_matrix_shape(*left);
                let right_matrix = self.expression_matrix_shape(*right);
                let left = self.expression(*left)?;
                let right = self.expression(*right)?;
                self.binary_with_shapes(*op, &left, &right, left_matrix, right_matrix)?
            }
            naga::Expression::Select {
                condition,
                accept,
                reject,
            } => {
                let condition = self.expression(*condition)?;
                let accept = self.expression(*accept)?;
                let reject = self.expression(*reject)?;
                self.select(&condition, accept, reject)?
            }
            naga::Expression::ImageSample {
                image,
                sampler,
                gather,
                coordinate,
                array_index,
                offset,
                level,
                depth_ref,
                clamp_to_edge,
            } => self.image_sample(
                *image,
                *sampler,
                *gather,
                *coordinate,
                *array_index,
                *offset,
                *level,
                *depth_ref,
                *clamp_to_edge,
            )?,
            naga::Expression::ImageLoad {
                image,
                coordinate,
                array_index,
                sample,
                level,
            } => self.image_load(*image, *coordinate, *array_index, *sample, *level)?,
            naga::Expression::ImageQuery { image, query } => self.image_query(*image, *query)?,
            naga::Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => self.math(*fun, *arg, *arg1, *arg2, *arg3)?,
            naga::Expression::Derivative { axis, expr, .. } => self.derivative(*axis, *expr)?,
            naga::Expression::Relational { fun, argument } => self.relational(*fun, *argument)?,
            naga::Expression::As {
                expr,
                kind,
                convert,
            } => self.cast(*expr, *kind, *convert)?,
            expression => {
                return Err(Error::UnsupportedFeature(format!(
                    "expression {handle:?} {expression:?}"
                )));
            }
        };
        self.values.insert(handle, value.clone());
        Ok(value)
    }

    fn splat(
        &mut self,
        size: naga::VectorSize,
        handle: naga::Handle<naga::Expression>,
    ) -> Result<Value, Error> {
        let value = self.expression(handle)?;
        if value.components.len() != 1 {
            return Err(Error::UnsupportedFeature(
                "splat of a non-scalar value".to_owned(),
            ));
        }
        Ok(Value {
            components: vec![value.components[0].clone(); vector_size(size)],
            kind: value.kind,
        })
    }

    fn select(&mut self, condition: &Value, accept: Value, reject: Value) -> Result<Value, Error> {
        if condition.kind != naga::ScalarKind::Bool || condition.components.is_empty() {
            return Err(Error::UnsupportedFeature(
                "select condition is not boolean".to_owned(),
            ));
        }
        if accept.kind != reject.kind || accept.components.len() != reject.components.len() {
            return Err(Error::UnsupportedFeature(
                "select branch type mismatch".to_owned(),
            ));
        }
        if condition.components.len() != 1 && condition.components.len() != accept.components.len()
        {
            return Err(Error::UnsupportedFeature(
                "select condition width mismatch".to_owned(),
            ));
        }
        let kind = accept.kind;
        let mut components = Vec::with_capacity(accept.components.len());
        for (index, (accept, reject)) in accept
            .components
            .into_iter()
            .zip(reject.components)
            .enumerate()
        {
            let condition = &condition.components[if condition.components.len() == 1 {
                0
            } else {
                index
            }];
            components.push(if kind == naga::ScalarKind::Bool {
                let accepted = self.target.ssa_alloc.alloc(RegFile::Pred);
                self.emit(Instr::new(OpPSetP {
                    dsts: [Dst::from(accepted), Dst::None],
                    ops: [PredSetOp::And, PredSetOp::And],
                    srcs: [condition.clone(), accept, true.into()],
                }));
                let dst = self.target.ssa_alloc.alloc(RegFile::Pred);
                self.emit(Instr::new(OpPSetP {
                    dsts: [Dst::from(dst), Dst::None],
                    ops: [PredSetOp::And, PredSetOp::Or],
                    srcs: [condition.clone().bnot(), reject, Src::from(accepted)],
                }));
                Src::from(dst)
            } else {
                let accept = self.materialize_select_source(accept, kind)?;
                let reject = self.materialize_select_source(reject, kind)?;
                let dst = self.target.ssa_alloc.alloc(RegFile::GPR);
                self.emit(Instr::new(OpSel {
                    dst: Dst::from(dst),
                    cond: condition.clone(),
                    srcs: [accept, reject],
                }));
                Src::from(dst)
            });
        }
        Ok(Value { components, kind })
    }

    fn materialize_select_source(
        &mut self,
        source: Src,
        kind: naga::ScalarKind,
    ) -> Result<Src, Error> {
        if source.is_unmodified() {
            return Ok(source);
        }
        let destination = self.target.ssa_alloc.alloc(RegFile::GPR);
        let instruction = match kind {
            naga::ScalarKind::Float => Instr::new(OpFMul {
                dst: Dst::from(destination),
                srcs: [source, Src::from(1.0_f32)],
                saturate: false,
                rnd_mode: FRndMode::NearestEven,
                ftz: false,
                dnz: false,
            }),
            naga::ScalarKind::Sint | naga::ScalarKind::Uint => Instr::new(OpIAdd2 {
                dst: Dst::from(destination),
                carry_out: Dst::None,
                srcs: [source, Src::ZERO],
            }),
            _ => {
                return Err(Error::UnsupportedFeature(
                    "modified non-numeric select source".to_owned(),
                ));
            }
        };
        self.emit(instruction);
        Ok(Src::from(destination))
    }

    #[allow(clippy::too_many_lines)]
    fn image_query(
        &mut self,
        image: naga::Handle<naga::Expression>,
        query: naga::ImageQuery,
    ) -> Result<Value, Error> {
        let (image, binding_index) = self.sampled_resource(image, "texture query")?;
        let (texture_reference, bindless_target) =
            if let Some(range) = self.resources.textures.get(&image).copied() {
                (
                    TexRef::Bindless,
                    Some(self.resource_array_target(range, binding_index)?),
                )
            } else {
                let target = self
                    .resources
                    .storage_textures
                    .get(&image)
                    .copied()
                    .ok_or_else(|| {
                        Error::UnsupportedFeature("queried texture has no Deko target".to_owned())
                    })?;
                (TexRef::Bound(target), None)
            };
        let image_type = binding_resource_base_type(self.module, image);
        let naga::TypeInner::Image {
            dim: dimension,
            arrayed,
            ..
        } = self.module.types[image_type].inner
        else {
            return Err(Error::UnsupportedFeature(
                "queried resource is not an image".to_owned(),
            ));
        };
        let (level, components, native_query, channel_mask, layer_divisor) = match query {
            naga::ImageQuery::Size { level } => {
                let level = match level {
                    Some(level) => self.expression(level)?,
                    None => Value {
                        components: vec![Src::ZERO],
                        kind: naga::ScalarKind::Uint,
                    },
                };
                if level.components.len() != 1
                    || !matches!(level.kind, naga::ScalarKind::Sint | naga::ScalarKind::Uint)
                {
                    return Err(Error::UnsupportedFeature(
                        "texture query LOD must be an integer scalar".to_owned(),
                    ));
                }
                let components = match dimension {
                    naga::ImageDimension::D1 => 1,
                    naga::ImageDimension::D2 | naga::ImageDimension::Cube => 2,
                    naga::ImageDimension::D3 => 3,
                };
                (
                    level,
                    components,
                    TexQuery::Dimension,
                    ChannelMask::for_comps(components),
                    None,
                )
            }
            naga::ImageQuery::NumSamples => (
                Value {
                    components: vec![Src::ZERO],
                    kind: naga::ScalarKind::Uint,
                },
                1,
                TexQuery::TextureType,
                ChannelMask::new(1 << 2),
                None,
            ),
            naga::ImageQuery::NumLevels => (
                Value {
                    components: vec![Src::ZERO],
                    kind: naga::ScalarKind::Uint,
                },
                1,
                TexQuery::Dimension,
                ChannelMask::new(1 << 3),
                None,
            ),
            naga::ImageQuery::NumLayers if arrayed => (
                Value {
                    components: vec![Src::ZERO],
                    kind: naga::ScalarKind::Uint,
                },
                1,
                TexQuery::Dimension,
                ChannelMask::new(match dimension {
                    naga::ImageDimension::D1 => 1 << 1,
                    naga::ImageDimension::D2 | naga::ImageDimension::Cube => 1 << 2,
                    naga::ImageDimension::D3 => {
                        return Err(Error::UnsupportedFeature(
                            "arrayed 3D texture layer query".to_owned(),
                        ));
                    }
                }),
                (dimension == naga::ImageDimension::Cube).then_some(6_u32),
            ),
            naga::ImageQuery::NumLayers => {
                return Err(Error::UnsupportedFeature(
                    "layer query on a non-array texture".to_owned(),
                ));
            }
        };
        let mut source = bindless_target.unwrap_or(Value {
            components: Vec::new(),
            kind: naga::ScalarKind::Uint,
        });
        source.components.extend(level.components);
        let source = self.materialize(source)?;
        let destination = self.target.ssa_alloc.alloc_vec(RegFile::GPR, components);
        self.emit(Instr::new(OpTxq {
            dsts: [Dst::from(destination.clone()), Dst::None],
            tex: texture_reference,
            src: Src::from(source),
            query: native_query,
            nodep: false,
            channel_mask,
        }));
        let result = Value {
            components: destination.iter().copied().map(Src::from).collect(),
            kind: naga::ScalarKind::Uint,
        };
        if let Some(divisor) = layer_divisor {
            self.binary(
                naga::BinaryOperator::Divide,
                &result,
                &Value {
                    components: vec![Src::from(divisor)],
                    kind: naga::ScalarKind::Uint,
                },
                None,
            )
        } else {
            Ok(result)
        }
    }

    fn image_store(
        &mut self,
        image: naga::Handle<naga::Expression>,
        coordinate: naga::Handle<naga::Expression>,
        array_index: Option<naga::Handle<naga::Expression>>,
        value: naga::Handle<naga::Expression>,
    ) -> Result<(), Error> {
        let image = self.global_expression(image, "texture store")?;
        let target = *self.resources.storage_textures.get(&image).ok_or_else(|| {
            Error::UnsupportedFeature("stored texture has no Deko target".to_owned())
        })?;
        let variable = &self.module.global_variables[image];
        let naga::TypeInner::Image {
            dim,
            arrayed,
            class: naga::ImageClass::Storage { access, format: _ },
        } = self.module.types[variable.ty].inner
        else {
            return Err(Error::UnsupportedFeature(
                "texture store resource is not a storage image".to_owned(),
            ));
        };
        if !access.contains(naga::StorageAccess::STORE) {
            return Err(Error::UnsupportedFeature(
                "texture store to a read-only image".to_owned(),
            ));
        }
        let (image_dim, coordinate_components) = match (dim, arrayed) {
            (naga::ImageDimension::D1, false) => (ImageDim::_1D, 1),
            (naga::ImageDimension::D1, true) => (ImageDim::_1DArray, 1),
            (naga::ImageDimension::D2, false) => (ImageDim::_2D, 2),
            (naga::ImageDimension::D2, true) => (ImageDim::_2DArray, 2),
            (naga::ImageDimension::D3, false) => (ImageDim::_3D, 3),
            (naga::ImageDimension::D3, true) => {
                return Err(Error::UnsupportedFeature(
                    "arrayed 3D storage texture".to_owned(),
                ));
            }
            (naga::ImageDimension::Cube, _) => {
                return Err(Error::UnsupportedFeature("cube storage texture".to_owned()));
            }
        };
        let mut coordinate = self.expression(coordinate)?;
        if !matches!(
            coordinate.kind,
            naga::ScalarKind::Sint | naga::ScalarKind::Uint
        ) || coordinate.components.len() != coordinate_components
            || arrayed != array_index.is_some()
        {
            return Err(Error::UnsupportedFeature(
                "texture store coordinate shape mismatch".to_owned(),
            ));
        }
        if let Some(array_index) = array_index {
            let array_index = self.expression(array_index)?;
            if !matches!(
                array_index.kind,
                naga::ScalarKind::Sint | naga::ScalarKind::Uint
            ) || array_index.components.len() != 1
            {
                return Err(Error::UnsupportedFeature(
                    "texture store array index must be an integer scalar".to_owned(),
                ));
            }
            coordinate.components.extend(array_index.components);
        }
        let value = self.expression(value)?;
        if value.components.len() != 4
            || !matches!(
                value.kind,
                naga::ScalarKind::Float | naga::ScalarKind::Sint | naga::ScalarKind::Uint
            )
        {
            return Err(Error::UnsupportedFeature(
                "texture store value must have four components".to_owned(),
            ));
        }
        let coordinate = self.materialize(coordinate)?;
        let data = self.materialize(value)?;
        let handle = self.materialize(Value {
            components: vec![Src::from(u32::from(target))],
            kind: naga::ScalarKind::Uint,
        })?;
        self.emit(Instr::new(OpSuSt {
            image_access: ImageAccess::Formatted(ChannelMask::for_comps(4)),
            image_dim,
            mem_order: MemOrder::Strong(MemScope::GPU),
            mem_eviction_priority: MemEvictionPriority::Normal,
            handle: Src::from(handle),
            coord: Src::from(coordinate),
            data: Src::from(data),
        }));
        Ok(())
    }

    fn image_atomic(
        &mut self,
        image: naga::Handle<naga::Expression>,
        coordinate: naga::Handle<naga::Expression>,
        array_index: Option<naga::Handle<naga::Expression>>,
        fun: naga::AtomicFunction,
        value: naga::Handle<naga::Expression>,
    ) -> Result<(), Error> {
        let image = self.global_expression(image, "texture atomic")?;
        let target = *self.resources.storage_textures.get(&image).ok_or_else(|| {
            Error::UnsupportedFeature("atomic texture has no Deko target".to_owned())
        })?;
        let variable = &self.module.global_variables[image];
        let naga::TypeInner::Image {
            dim,
            arrayed,
            class: naga::ImageClass::Storage { access, format },
        } = self.module.types[variable.ty].inner
        else {
            return Err(Error::UnsupportedFeature(
                "texture atomic resource is not a storage image".to_owned(),
            ));
        };
        if !access.contains(naga::StorageAccess::ATOMIC) {
            return Err(Error::UnsupportedFeature(
                "texture atomic on a non-atomic image".to_owned(),
            ));
        }
        let (atom_type, kind) = match format {
            naga::StorageFormat::R32Uint => (AtomType::U32, naga::ScalarKind::Uint),
            naga::StorageFormat::R32Sint => (AtomType::I32, naga::ScalarKind::Sint),
            other => {
                return Err(Error::UnsupportedFeature(format!(
                    "texture atomic format {other:?}"
                )));
            }
        };
        let (image_dim, coordinate_components) = match (dim, arrayed) {
            (naga::ImageDimension::D1, false) => (ImageDim::_1D, 1),
            (naga::ImageDimension::D1, true) => (ImageDim::_1DArray, 1),
            (naga::ImageDimension::D2, false) => (ImageDim::_2D, 2),
            (naga::ImageDimension::D2, true) => (ImageDim::_2DArray, 2),
            (naga::ImageDimension::D3, false) => (ImageDim::_3D, 3),
            (naga::ImageDimension::D3, true) => {
                return Err(Error::UnsupportedFeature(
                    "arrayed 3D atomic texture".to_owned(),
                ));
            }
            (naga::ImageDimension::Cube, _) => {
                return Err(Error::UnsupportedFeature("cube atomic texture".to_owned()));
            }
        };
        let mut coordinate = self.expression(coordinate)?;
        if coordinate.kind != naga::ScalarKind::Sint
            || coordinate.components.len() != coordinate_components
            || arrayed != array_index.is_some()
        {
            return Err(Error::UnsupportedFeature(
                "texture atomic coordinate shape mismatch".to_owned(),
            ));
        }
        if let Some(array_index) = array_index {
            let array_index = self.expression(array_index)?;
            if array_index.kind != naga::ScalarKind::Sint || array_index.components.len() != 1 {
                return Err(Error::UnsupportedFeature(
                    "texture atomic array index must be a signed integer scalar".to_owned(),
                ));
            }
            coordinate.components.extend(array_index.components);
        }
        let value = self.expression(value)?;
        if value.kind != kind || value.components.len() != 1 {
            return Err(Error::UnsupportedFeature(
                "texture atomic value type mismatch".to_owned(),
            ));
        }
        let atom_op = Self::image_atomic_operation(fun)?;
        let coordinate = self.materialize(coordinate)?;
        let data = self.materialize(value)?;
        let handle = self.materialize(Value {
            components: vec![Src::from(u32::from(target))],
            kind: naga::ScalarKind::Uint,
        })?;
        self.emit(Instr::new(OpSuAtom {
            dst: Dst::None,
            fault: Dst::None,
            image_dim,
            atom_op,
            atom_type,
            mem_order: MemOrder::Strong(MemScope::GPU),
            mem_eviction_priority: MemEvictionPriority::Normal,
            handle: Src::from(handle),
            coord: Src::from(coordinate),
            data: Src::from(data),
        }));
        Ok(())
    }

    fn image_atomic_operation(fun: naga::AtomicFunction) -> Result<AtomOp, Error> {
        match fun {
            naga::AtomicFunction::Add => Ok(AtomOp::Add),
            naga::AtomicFunction::And => Ok(AtomOp::And),
            naga::AtomicFunction::ExclusiveOr => Ok(AtomOp::Xor),
            naga::AtomicFunction::InclusiveOr => Ok(AtomOp::Or),
            naga::AtomicFunction::Min => Ok(AtomOp::Min),
            naga::AtomicFunction::Max => Ok(AtomOp::Max),
            other => Err(Error::UnsupportedFeature(format!(
                "texture atomic operation {other:?}"
            ))),
        }
    }

    fn subgroup_gather(
        &mut self,
        mode: naga::GatherMode,
        argument: naga::Handle<naga::Expression>,
        result: naga::Handle<naga::Expression>,
    ) -> Result<(), Error> {
        let argument = self.expression(argument)?;
        let sources = self.materialize_loop_components(&argument)?;
        let (lane, c, op) = match mode {
            naga::GatherMode::BroadcastFirst => {
                let active_lanes = self.target.ssa_alloc.alloc(RegFile::GPR);
                self.emit(Instr::new(OpVote {
                    op: VoteOp::Any,
                    ballot: Dst::from(active_lanes),
                    vote: Dst::None,
                    pred: Src::from(SrcRef::True),
                }));
                let reversed = self.reverse_bits(Value {
                    components: vec![Src::from(active_lanes)],
                    kind: naga::ScalarKind::Uint,
                })?;
                let first_active = self.count_leading_zeros(reversed)?;
                (
                    Src::from(self.materialize(first_active)?),
                    Src::from(0x1f_u32),
                    ShflOp::Idx,
                )
            }
            naga::GatherMode::Broadcast(index) | naga::GatherMode::Shuffle(index) => {
                (self.subgroup_lane(index)?, Src::from(0x1f_u32), ShflOp::Idx)
            }
            naga::GatherMode::ShuffleDown(index) => (
                self.subgroup_lane(index)?,
                Src::from(0x1f_u32),
                ShflOp::Down,
            ),
            naga::GatherMode::ShuffleUp(index) => {
                (self.subgroup_lane(index)?, Src::ZERO, ShflOp::Up)
            }
            naga::GatherMode::ShuffleXor(index) => (
                self.subgroup_lane(index)?,
                Src::from(0x1f_u32),
                ShflOp::Bfly,
            ),
            naga::GatherMode::QuadBroadcast(index) => (
                self.subgroup_lane(index)?,
                Src::from(0x1c03_u32),
                ShflOp::Idx,
            ),
            naga::GatherMode::QuadSwap(direction) => (
                Src::from(match direction {
                    naga::Direction::X => 1_u32,
                    naga::Direction::Y => 2_u32,
                    naga::Direction::Diagonal => 3_u32,
                }),
                Src::from(0x1c03_u32),
                ShflOp::Bfly,
            ),
        };
        let mut components = Vec::with_capacity(sources.len());
        for source in sources {
            let destination = self.target.ssa_alloc.alloc(RegFile::GPR);
            self.emit(Instr::new(OpShfl {
                dst: Dst::from(destination),
                in_bounds: Dst::None,
                src: Src::from(source),
                lane: lane.clone(),
                c: c.clone(),
                op,
            }));
            components.push(Src::from(destination));
        }
        self.values.insert(
            result,
            Value {
                components,
                kind: argument.kind,
            },
        );
        Ok(())
    }

    fn subgroup_ballot(
        &mut self,
        predicate: Option<naga::Handle<naga::Expression>>,
        result: naga::Handle<naga::Expression>,
    ) -> Result<(), Error> {
        let predicate = match predicate {
            Some(predicate) => {
                let value = self.expression(predicate)?;
                if value.kind != naga::ScalarKind::Bool || value.components.len() != 1 {
                    return Err(Error::UnsupportedFeature(
                        "subgroup ballot predicate must be a scalar boolean".to_owned(),
                    ));
                }
                Self::predicate(&value.components[0])?
            }
            None => Pred::from(true),
        };
        let ballot = self.target.ssa_alloc.alloc(RegFile::GPR);
        self.emit(Instr::new(OpVote {
            op: VoteOp::Any,
            ballot: Dst::from(ballot),
            vote: Dst::None,
            pred: Src::from(predicate),
        }));
        self.values.insert(
            result,
            Value {
                components: vec![Src::from(ballot), Src::ZERO, Src::ZERO, Src::ZERO],
                kind: naga::ScalarKind::Uint,
            },
        );
        Ok(())
    }

    fn subgroup_collective(
        &mut self,
        op: naga::SubgroupOperation,
        collective_op: naga::CollectiveOperation,
        argument: naga::Handle<naga::Expression>,
        result: naga::Handle<naga::Expression>,
    ) -> Result<(), Error> {
        let argument = self.expression(argument)?;
        if let Some(vote_op) = match (op, collective_op) {
            (naga::SubgroupOperation::All, naga::CollectiveOperation::Reduce) => Some(VoteOp::All),
            (naga::SubgroupOperation::Any, naga::CollectiveOperation::Reduce) => Some(VoteOp::Any),
            _ => None,
        } {
            if argument.kind != naga::ScalarKind::Bool || argument.components.len() != 1 {
                return Err(Error::UnsupportedFeature(
                    "subgroup vote argument must be a scalar boolean".to_owned(),
                ));
            }
            let vote = self.target.ssa_alloc.alloc(RegFile::Pred);
            self.emit(Instr::new(OpVote {
                op: vote_op,
                ballot: Dst::None,
                vote: Dst::from(vote),
                pred: Src::from(Self::predicate(&argument.components[0])?),
            }));
            self.values.insert(
                result,
                Value {
                    components: vec![Src::from(vote)],
                    kind: naga::ScalarKind::Bool,
                },
            );
            return Ok(());
        }

        if matches!(
            op,
            naga::SubgroupOperation::All | naga::SubgroupOperation::Any
        ) {
            return Err(Error::UnsupportedFeature(format!(
                "subgroup collective {collective_op:?} {op:?}"
            )));
        }
        if !matches!(
            argument.kind,
            naga::ScalarKind::Float | naga::ScalarKind::Sint | naga::ScalarKind::Uint
        ) {
            return Err(Error::UnsupportedFeature(format!(
                "subgroup {op:?} for {:?}",
                argument.kind
            )));
        }

        let sources = self.materialize_loop_components(&argument)?;
        let ballot = self.target.ssa_alloc.alloc(RegFile::GPR);
        self.emit(Instr::new(OpVote {
            op: VoteOp::Any,
            ballot: Dst::from(ballot),
            vote: Dst::None,
            pred: Src::from(SrcRef::True),
        }));
        let ballot = Value {
            components: vec![Src::from(ballot)],
            kind: naga::ScalarKind::Uint,
        };
        let current_lane = if collective_op == naga::CollectiveOperation::Reduce {
            None
        } else {
            Some(Value {
                components: read_compute_system_value(self, &[0]),
                kind: naga::ScalarKind::Uint,
            })
        };
        let mut accumulator = Self::subgroup_identity(op, argument.kind, sources.len())?;
        for lane in 0..32_u32 {
            let include = self.subgroup_collective_include(
                &ballot,
                current_lane.as_ref(),
                collective_op,
                lane,
            )?;
            let mut components = Vec::with_capacity(sources.len());
            for source in &sources {
                let destination = self.target.ssa_alloc.alloc(RegFile::GPR);
                self.emit(Instr::new(OpShfl {
                    dst: Dst::from(destination),
                    in_bounds: Dst::None,
                    src: Src::from(*source),
                    lane: Src::from(lane),
                    c: Src::from(0x1f_u32),
                    op: ShflOp::Idx,
                }));
                components.push(Src::from(destination));
            }
            let shuffled = Value {
                components,
                kind: argument.kind,
            };
            let combined = self.subgroup_combine(op, &accumulator, &shuffled)?;
            accumulator = self.select(&include, combined, accumulator)?;
        }
        self.values.insert(result, accumulator);
        Ok(())
    }

    fn subgroup_collective_include(
        &mut self,
        ballot: &Value,
        current_lane: Option<&Value>,
        collective_op: naga::CollectiveOperation,
        lane: u32,
    ) -> Result<Value, Error> {
        let lane_value = Value {
            components: vec![Src::from(lane)],
            kind: naga::ScalarKind::Uint,
        };
        let shifted = self.binary(naga::BinaryOperator::ShiftRight, ballot, &lane_value, None)?;
        let active = self.binary(
            naga::BinaryOperator::And,
            &shifted,
            &Value {
                components: vec![Src::from(1_u32)],
                kind: naga::ScalarKind::Uint,
            },
            None,
        )?;
        let include = self.binary(
            naga::BinaryOperator::NotEqual,
            &active,
            &Value {
                components: vec![Src::ZERO],
                kind: naga::ScalarKind::Uint,
            },
            None,
        )?;
        let Some(current_lane) = current_lane else {
            return Ok(include);
        };
        let comparison = match collective_op {
            naga::CollectiveOperation::InclusiveScan => naga::BinaryOperator::GreaterEqual,
            naga::CollectiveOperation::ExclusiveScan => naga::BinaryOperator::Greater,
            naga::CollectiveOperation::Reduce => unreachable!(),
        };
        let in_prefix = self.binary(comparison, current_lane, &lane_value, None)?;
        Ok(Value {
            components: vec![self.emit_predicate_binary(
                PredSetOp::And,
                include.components[0].clone(),
                in_prefix.components[0].clone(),
            )],
            kind: naga::ScalarKind::Bool,
        })
    }

    fn subgroup_identity(
        op: naga::SubgroupOperation,
        kind: naga::ScalarKind,
        width: usize,
    ) -> Result<Value, Error> {
        let component = match (op, kind) {
            (naga::SubgroupOperation::Add, naga::ScalarKind::Float) => Src::from(0.0_f32),
            (
                naga::SubgroupOperation::Add
                | naga::SubgroupOperation::Or
                | naga::SubgroupOperation::Xor,
                naga::ScalarKind::Sint | naga::ScalarKind::Uint,
            )
            | (naga::SubgroupOperation::Max, naga::ScalarKind::Uint) => Src::ZERO,
            (naga::SubgroupOperation::Mul, naga::ScalarKind::Float) => Src::from(1.0_f32),
            (naga::SubgroupOperation::Mul, naga::ScalarKind::Sint | naga::ScalarKind::Uint) => {
                Src::from(1_u32)
            }
            (naga::SubgroupOperation::Min, naga::ScalarKind::Float) => Src::from(f32::INFINITY),
            (naga::SubgroupOperation::Max, naga::ScalarKind::Float) => Src::from(f32::NEG_INFINITY),
            (naga::SubgroupOperation::Min, naga::ScalarKind::Sint) => {
                Src::from(i32::MAX.cast_unsigned())
            }
            (naga::SubgroupOperation::Max, naga::ScalarKind::Sint) => {
                Src::from(i32::MIN.cast_unsigned())
            }
            (naga::SubgroupOperation::Min, naga::ScalarKind::Uint) => Src::from(u32::MAX),
            (naga::SubgroupOperation::And, naga::ScalarKind::Sint | naga::ScalarKind::Uint) => {
                Src::from(u32::MAX)
            }
            _ => {
                return Err(Error::UnsupportedFeature(format!(
                    "subgroup {op:?} for {kind:?}"
                )));
            }
        };
        Ok(Value {
            components: vec![component; width],
            kind,
        })
    }

    fn subgroup_combine(
        &mut self,
        op: naga::SubgroupOperation,
        left: &Value,
        right: &Value,
    ) -> Result<Value, Error> {
        match op {
            naga::SubgroupOperation::Add => {
                self.binary(naga::BinaryOperator::Add, left, right, None)
            }
            naga::SubgroupOperation::Mul => {
                self.binary(naga::BinaryOperator::Multiply, left, right, None)
            }
            naga::SubgroupOperation::Min if left.kind == naga::ScalarKind::Float => {
                self.float_minmax(left, right, true)
            }
            naga::SubgroupOperation::Max if left.kind == naga::ScalarKind::Float => {
                self.float_minmax(left, right, false)
            }
            naga::SubgroupOperation::Min => self.integer_minmax(left, right, true),
            naga::SubgroupOperation::Max => self.integer_minmax(left, right, false),
            naga::SubgroupOperation::And => {
                self.binary(naga::BinaryOperator::And, left, right, None)
            }
            naga::SubgroupOperation::Or => {
                self.binary(naga::BinaryOperator::InclusiveOr, left, right, None)
            }
            naga::SubgroupOperation::Xor => {
                self.binary(naga::BinaryOperator::ExclusiveOr, left, right, None)
            }
            naga::SubgroupOperation::All | naga::SubgroupOperation::Any => Err(
                Error::UnsupportedFeature(format!("non-boolean subgroup {op:?}")),
            ),
        }
    }

    fn subgroup_lane(&mut self, index: naga::Handle<naga::Expression>) -> Result<Src, Error> {
        let index = self.expression(index)?;
        if index.components.len() != 1
            || !matches!(index.kind, naga::ScalarKind::Sint | naga::ScalarKind::Uint)
        {
            return Err(Error::UnsupportedFeature(
                "subgroup lane is not an integer scalar".to_owned(),
            ));
        }
        Ok(Src::from(self.materialize(index)?))
    }

    #[allow(clippy::too_many_lines)]
    fn image_load(
        &mut self,
        image: naga::Handle<naga::Expression>,
        coordinate: naga::Handle<naga::Expression>,
        array_index: Option<naga::Handle<naga::Expression>>,
        sample: Option<naga::Handle<naga::Expression>>,
        level: Option<naga::Handle<naga::Expression>>,
    ) -> Result<Value, Error> {
        let image = self.global_expression(image, "texture load")?;
        let variable = &self.module.global_variables[image];
        let naga::TypeInner::Image {
            dim,
            arrayed,
            class,
        } = self.module.types[variable.ty].inner
        else {
            return Err(Error::UnsupportedFeature(
                "texture load resource is not an image".to_owned(),
            ));
        };
        if let naga::ImageClass::Storage { format, access } = class {
            return self.storage_image_load(
                image,
                dim,
                arrayed,
                format,
                access,
                coordinate,
                array_index,
                sample,
                level,
            );
        }
        let range = *self.resources.textures.get(&image).ok_or_else(|| {
            Error::UnsupportedFeature("loaded texture has no Deko target".to_owned())
        })?;
        let (kind, output_components, multisampled) = match class {
            naga::ImageClass::Sampled { kind, multi } => (kind, 4, multi),
            naga::ImageClass::Depth { multi } => (naga::ScalarKind::Float, 1, multi),
            other => {
                return Err(Error::UnsupportedFeature(format!(
                    "texture load image class {other:?}"
                )));
            }
        };
        let tex_dim = match (dim, arrayed) {
            (naga::ImageDimension::D1, false) => TexDim::_1D,
            (naga::ImageDimension::D1, true) => TexDim::Array1D,
            (naga::ImageDimension::D2, false) => TexDim::_2D,
            (naga::ImageDimension::D2, true) => TexDim::Array2D,
            (naga::ImageDimension::D3, false) => TexDim::_3D,
            (naga::ImageDimension::Cube, false) => TexDim::Cube,
            (naga::ImageDimension::Cube, true) => TexDim::ArrayCube,
            (naga::ImageDimension::D3, true) => {
                return Err(Error::UnsupportedFeature(
                    "arrayed 3D texture load".to_owned(),
                ));
            }
        };
        let expected_coordinates = match dim {
            naga::ImageDimension::D1 => 1,
            naga::ImageDimension::D2 => 2,
            naga::ImageDimension::D3 | naga::ImageDimension::Cube => 3,
        };
        let mut source = self.expression(coordinate)?;
        if !matches!(source.kind, naga::ScalarKind::Sint | naga::ScalarKind::Uint)
            || source.components.len() != expected_coordinates
        {
            return Err(Error::UnsupportedFeature(
                "texture load coordinate shape mismatch".to_owned(),
            ));
        }
        if arrayed != array_index.is_some() {
            return Err(Error::UnsupportedFeature(
                "texture load array index mismatch".to_owned(),
            ));
        }
        if let Some(array_index) = array_index {
            let array_index = self.expression(array_index)?;
            if !matches!(
                array_index.kind,
                naga::ScalarKind::Sint | naga::ScalarKind::Uint
            ) || array_index.components.len() != 1
            {
                return Err(Error::UnsupportedFeature(
                    "texture load array index must be an integer scalar".to_owned(),
                ));
            }
            source.components.extend(array_index.components);
        }
        let extra = match (multisampled, sample, level) {
            (true, Some(sample), None) => sample,
            (false, None, Some(level)) => level,
            _ => {
                return Err(Error::UnsupportedFeature(
                    "texture load sample/level mismatch".to_owned(),
                ));
            }
        };
        let extra = self.expression(extra)?;
        if !matches!(extra.kind, naga::ScalarKind::Sint | naga::ScalarKind::Uint)
            || extra.components.len() != 1
        {
            return Err(Error::UnsupportedFeature(
                "texture load sample or level must be an integer scalar".to_owned(),
            ));
        }
        source.components.extend(extra.components);
        let source = self.materialize(source)?;
        let handle = self.resource_array_target(range, None)?;
        let handle = self.materialize(handle)?;
        let destination = self
            .target
            .ssa_alloc
            .alloc_vec(RegFile::GPR, output_components);
        self.emit(Instr::new(OpTld {
            dsts: [Dst::from(destination.clone()), Dst::None],
            fault: Dst::None,
            tex: TexRef::Bindless,
            srcs: [Src::from(source), Src::from(handle)],
            dim: tex_dim,
            is_ms: multisampled,
            lod_mode: if multisampled {
                TexLodMode::Zero
            } else {
                TexLodMode::Lod
            },
            offset_mode: TexOffsetMode::None,
            mem_eviction_priority: MemEvictionPriority::Normal,
            nodep: false,
            channel_mask: ChannelMask::for_comps(output_components),
            scalar: false,
        }));
        Ok(Value {
            components: destination.iter().copied().map(Src::from).collect(),
            kind,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn storage_image_load(
        &mut self,
        image: naga::Handle<naga::GlobalVariable>,
        dim: naga::ImageDimension,
        arrayed: bool,
        format: naga::StorageFormat,
        access: naga::StorageAccess,
        coordinate: naga::Handle<naga::Expression>,
        array_index: Option<naga::Handle<naga::Expression>>,
        sample: Option<naga::Handle<naga::Expression>>,
        level: Option<naga::Handle<naga::Expression>>,
    ) -> Result<Value, Error> {
        if !access.contains(naga::StorageAccess::LOAD) {
            return Err(Error::UnsupportedFeature(
                "texture load from a write-only storage image".to_owned(),
            ));
        }
        if sample.is_some() || level.is_some() {
            return Err(Error::UnsupportedFeature(
                "storage texture load with a sample or mip level".to_owned(),
            ));
        }
        let target = *self.resources.storage_textures.get(&image).ok_or_else(|| {
            Error::UnsupportedFeature("loaded storage texture has no Deko target".to_owned())
        })?;
        let (image_dim, coordinate_components) = match (dim, arrayed) {
            (naga::ImageDimension::D1, false) => (ImageDim::_1D, 1),
            (naga::ImageDimension::D1, true) => (ImageDim::_1DArray, 1),
            (naga::ImageDimension::D2, false) => (ImageDim::_2D, 2),
            (naga::ImageDimension::D2, true) => (ImageDim::_2DArray, 2),
            (naga::ImageDimension::D3, false) => (ImageDim::_3D, 3),
            (naga::ImageDimension::D3, true) => {
                return Err(Error::UnsupportedFeature(
                    "arrayed 3D storage texture".to_owned(),
                ));
            }
            (naga::ImageDimension::Cube, _) => {
                return Err(Error::UnsupportedFeature("cube storage texture".to_owned()));
            }
        };
        let mut coordinate = self.expression(coordinate)?;
        if !matches!(
            coordinate.kind,
            naga::ScalarKind::Sint | naga::ScalarKind::Uint
        ) || coordinate.components.len() != coordinate_components
            || arrayed != array_index.is_some()
        {
            return Err(Error::UnsupportedFeature(
                "storage texture load coordinate shape mismatch".to_owned(),
            ));
        }
        if let Some(array_index) = array_index {
            let array_index = self.expression(array_index)?;
            if !matches!(
                array_index.kind,
                naga::ScalarKind::Sint | naga::ScalarKind::Uint
            ) || array_index.components.len() != 1
            {
                return Err(Error::UnsupportedFeature(
                    "storage texture array index must be an integer scalar".to_owned(),
                ));
            }
            coordinate.components.extend(array_index.components);
        }
        let coordinate = self.materialize(coordinate)?;
        let destination = self.target.ssa_alloc.alloc_vec(RegFile::GPR, 4);
        let handle = self.materialize(Value {
            components: vec![Src::from(u32::from(target))],
            kind: naga::ScalarKind::Uint,
        })?;
        self.emit(Instr::new(OpSuLd {
            dst: Dst::from(destination.clone()),
            fault: Dst::None,
            image_access: ImageAccess::Formatted(ChannelMask::for_comps(4)),
            image_dim,
            mem_order: MemOrder::Strong(MemScope::GPU),
            mem_eviction_priority: MemEvictionPriority::Normal,
            handle: Src::from(handle),
            coord: Src::from(coordinate),
        }));
        let scalar: naga::Scalar = format.into();
        Ok(Value {
            components: destination.iter().copied().map(Src::from).collect(),
            kind: scalar.kind,
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    fn image_sample(
        &mut self,
        image: naga::Handle<naga::Expression>,
        sampler: naga::Handle<naga::Expression>,
        gather: Option<naga::SwizzleComponent>,
        coordinate: naga::Handle<naga::Expression>,
        array_index: Option<naga::Handle<naga::Expression>>,
        offset: Option<naga::Handle<naga::Expression>>,
        level: naga::SampleLevel,
        depth_ref: Option<naga::Handle<naga::Expression>>,
        clamp_to_edge: bool,
    ) -> Result<Value, Error> {
        let image_expression = image;
        if clamp_to_edge || (gather.is_some() && !matches!(level, naga::SampleLevel::Zero)) {
            return Err(Error::UnsupportedFeature(format!(
                "texture sample options gather={gather:?} array={} offset={} level={level:?} depth={} clamp={clamp_to_edge}",
                array_index.is_some(),
                offset.is_some(),
                depth_ref.is_some()
            )));
        }
        let (image, image_binding_index) = self.sampled_resource(image, "texture")?;
        let (sampler, sampler_binding_index) = self.sampled_resource(sampler, "sampler")?;
        let image_range = *self.resources.textures.get(&image).ok_or_else(|| {
            Error::UnsupportedFeature("sampled image has no Deko target".to_owned())
        })?;
        let sampler_range =
            *self.resources.samplers.get(&sampler).ok_or_else(|| {
                Error::UnsupportedFeature("sampler has no Deko target".to_owned())
            })?;
        let handle = if image_binding_index.is_none() && sampler_binding_index.is_none() {
            Value {
                components: vec![Src::from(
                    u32::from(image_range.target) | (u32::from(sampler_range.target) << 20),
                )],
                kind: naga::ScalarKind::Uint,
            }
        } else {
            let image_target = self.resource_array_target(image_range, image_binding_index)?;
            let sampler_target =
                self.resource_array_target(sampler_range, sampler_binding_index)?;
            let sampler_bits = self.binary(
                naga::BinaryOperator::ShiftLeft,
                &sampler_target,
                &Value {
                    components: vec![Src::from(20_u32)],
                    kind: naga::ScalarKind::Uint,
                },
                None,
            )?;
            self.binary(
                naga::BinaryOperator::InclusiveOr,
                &image_target,
                &sampler_bits,
                None,
            )?
        };
        let texture_reference = TexRef::Bindless;
        let bindless_handle = Some(Src::from(self.materialize(handle)?));
        let (dim, kind) = match module_image_type(self.module, image)? {
            (naga::ImageDimension::D1, false, kind) => (TexDim::_1D, kind),
            (naga::ImageDimension::D1, true, kind) => (TexDim::Array1D, kind),
            (naga::ImageDimension::D2, false, kind) => (TexDim::_2D, kind),
            (naga::ImageDimension::D2, true, kind) => (TexDim::Array2D, kind),
            (naga::ImageDimension::D3, false, kind) => (TexDim::_3D, kind),
            (naga::ImageDimension::Cube, false, kind) => (TexDim::Cube, kind),
            (naga::ImageDimension::Cube, true, kind) => (TexDim::ArrayCube, kind),
            (naga::ImageDimension::D3, true, _) => {
                return Err(Error::UnsupportedFeature(
                    "arrayed 3D texture sample".to_owned(),
                ));
            }
        };
        let mut coordinate = self.expression(coordinate)?;
        let expected = match dim {
            TexDim::_1D | TexDim::Array1D => 1,
            TexDim::_2D | TexDim::Array2D => 2,
            TexDim::_3D | TexDim::Cube | TexDim::ArrayCube => 3,
        };
        if coordinate.kind != naga::ScalarKind::Float || coordinate.components.len() != expected {
            return Err(Error::UnsupportedFeature(
                "texture coordinate shape mismatch".to_owned(),
            ));
        }
        let mut auxiliary = bindless_handle.clone().into_iter().collect::<Vec<_>>();
        let array_index = if let Some(array_index) = array_index {
            let array_index = self.expression(array_index)?;
            if array_index.components.len() != 1
                || !matches!(
                    array_index.kind,
                    naga::ScalarKind::Sint | naga::ScalarKind::Uint
                )
            {
                return Err(Error::UnsupportedFeature(
                    "texture array index must be an integer scalar".to_owned(),
                ));
            }
            Some(array_index.components[0].clone())
        } else {
            None
        };
        let depth_reference = if let Some(depth_ref) = depth_ref {
            let depth_ref = self.expression(depth_ref)?;
            if depth_ref.kind != naga::ScalarKind::Float || depth_ref.components.len() != 1 {
                return Err(Error::UnsupportedFeature(
                    "texture depth reference must be a float scalar".to_owned(),
                ));
            }
            Some(depth_ref.components[0].clone())
        } else {
            None
        };
        let mut rewritten_gradient_lod = None;
        if let naga::SampleLevel::Gradient { x, y } = level {
            if depth_reference.is_some() {
                return Err(Error::UnsupportedFeature(
                    "explicit-gradient depth comparison sampling".to_owned(),
                ));
            }

            let derivative_x = self.expression(x)?;
            let derivative_y = self.expression(y)?;
            if derivative_x.kind != naga::ScalarKind::Float
                || derivative_y.kind != naga::ScalarKind::Float
                || derivative_x.components.len() != expected
                || derivative_y.components.len() != expected
            {
                return Err(Error::UnsupportedFeature(
                    "texture gradients must match the floating-point coordinate shape".to_owned(),
                ));
            }

            if expected > 2 {
                rewritten_gradient_lod = Some(self.explicit_gradient_lod(
                    image_expression,
                    dim,
                    &coordinate,
                    &derivative_x,
                    &derivative_y,
                )?);
            } else {
                // SM50 TXD has a different source contract from TEX. Mesa's NAK lowering
                // packs the bindless handle and coordinates into source zero, then interleaves
                // the explicit derivatives in source one. Array indices occupy the final
                // source-zero component; when an offset is present, PRMT combines the low
                // 16-bit array index with the packed offset in the high half.
                let mut primary = bindless_handle.into_iter().collect::<Vec<_>>();
                primary.extend(coordinate.components);
                let offset_mode = if let Some(offset) = offset {
                    let packed_offset = self.pack_texture_offset(offset)?;
                    let packed_array_offset = self.target.ssa_alloc.alloc(RegFile::GPR);
                    self.emit(Instr::new(OpPrmt {
                        dst: Dst::from(packed_array_offset),
                        srcs: [packed_offset, array_index.clone().unwrap_or(Src::ZERO)],
                        sel: Src::from(0x1054_u32),
                        mode: PrmtMode::Index,
                    }));
                    primary.push(Src::from(packed_array_offset));
                    TexOffsetMode::AddOffI
                } else {
                    primary.extend(array_index);
                    TexOffsetMode::None
                };
                if primary.len() > 4 {
                    return Err(Error::UnsupportedFeature(
                        "explicit-gradient texture source exceeds the SM50 TXD register tuple"
                            .to_owned(),
                    ));
                }

                let mut derivatives = Vec::with_capacity(expected * 2);
                for (dx, dy) in derivative_x
                    .components
                    .into_iter()
                    .zip(derivative_y.components)
                {
                    derivatives.push(dx);
                    derivatives.push(dy);
                }

                let sources = [
                    Src::from(self.materialize(Value {
                        components: primary,
                        kind: naga::ScalarKind::Uint,
                    })?),
                    Src::from(self.materialize(Value {
                        components: derivatives,
                        kind: naga::ScalarKind::Float,
                    })?),
                ];
                let destination = self.target.ssa_alloc.alloc_vec(RegFile::GPR, 4);
                self.emit(Instr::new(OpTxd {
                    dsts: [Dst::from(destination.clone()), Dst::None],
                    fault: Dst::None,
                    tex: texture_reference,
                    srcs: sources,
                    dim,
                    offset_mode,
                    mem_eviction_priority: MemEvictionPriority::Normal,
                    nodep: false,
                    channel_mask: ChannelMask::for_comps(4),
                }));
                return Ok(Value {
                    components: destination.iter().copied().map(Src::from).collect(),
                    kind,
                });
            }
        }

        if let Some(array_index) = array_index {
            coordinate.components.insert(0, array_index);
        }
        if let Some(component) = gather {
            let component = match component {
                naga::SwizzleComponent::X => 0,
                naga::SwizzleComponent::Y => 1,
                naga::SwizzleComponent::Z => 2,
                naga::SwizzleComponent::W => 3,
            };
            if let Some(offset) = offset {
                auxiliary.push(self.pack_texture_offset(offset)?);
            }
            auxiliary.extend(depth_reference.iter().cloned());
            let sources =
                self.texture_instruction_sources(texture_reference, coordinate, auxiliary)?;
            let destination = self.target.ssa_alloc.alloc_vec(RegFile::GPR, 4);
            self.emit(Instr::new(OpTld4 {
                dsts: [Dst::from(destination.clone()), Dst::None],
                fault: Dst::None,
                tex: texture_reference,
                srcs: sources,
                dim,
                comp: component,
                offset_mode: TexOffsetMode::None,
                z_cmpr: depth_ref.is_some(),
                mem_eviction_priority: MemEvictionPriority::Normal,
                nodep: false,
                channel_mask: ChannelMask::for_comps(4),
                scalar: false,
            }));
            return Ok(Value {
                components: destination.iter().copied().map(Src::from).collect(),
                kind,
            });
        }
        let lod_mode = match level {
            naga::SampleLevel::Auto => TexLodMode::Auto,
            naga::SampleLevel::Zero => TexLodMode::Zero,
            naga::SampleLevel::Bias(level) => {
                let level = self.expression(level)?;
                if level.kind != naga::ScalarKind::Float || level.components.len() != 1 {
                    return Err(Error::UnsupportedFeature(
                        "texture bias must be a float scalar".to_owned(),
                    ));
                }
                auxiliary.extend(level.components);
                TexLodMode::Bias
            }
            naga::SampleLevel::Exact(level) => {
                let level = self.expression(level)?;
                if level.kind != naga::ScalarKind::Float || level.components.len() != 1 {
                    return Err(Error::UnsupportedFeature(
                        "texture LOD must be a float scalar".to_owned(),
                    ));
                }
                auxiliary.extend(level.components);
                TexLodMode::Lod
            }
            naga::SampleLevel::Gradient { .. } => {
                auxiliary.extend(
                    rewritten_gradient_lod
                        .expect("3D/cube gradients are rewritten above")
                        .components,
                );
                TexLodMode::Lod
            }
        };
        let offset_mode = if let Some(offset) = offset {
            auxiliary.push(self.pack_texture_offset(offset)?);
            TexOffsetMode::AddOffI
        } else {
            TexOffsetMode::None
        };
        auxiliary.extend(depth_reference);
        let sources = self.texture_instruction_sources(texture_reference, coordinate, auxiliary)?;
        let output_components = if depth_ref.is_some() { 1 } else { 4 };
        let dst = self
            .target
            .ssa_alloc
            .alloc_vec(RegFile::GPR, output_components);
        self.emit(Instr::new(OpTex {
            dsts: [Dst::from(dst.clone()), Dst::None],
            fault: Dst::None,
            tex: texture_reference,
            srcs: sources,
            dim,
            lod_mode,
            deriv_mode: TexDerivMode::Auto,
            z_cmpr: depth_ref.is_some(),
            offset_mode,
            mem_eviction_priority: MemEvictionPriority::Normal,
            nodep: false,
            channel_mask: ChannelMask::for_comps(output_components),
            scalar: false,
        }));
        Ok(Value {
            components: dst.iter().copied().map(Src::from).collect(),
            kind,
        })
    }

    fn explicit_gradient_lod(
        &mut self,
        image: naga::Handle<naga::Expression>,
        dim: TexDim,
        coordinate: &Value,
        derivative_x: &Value,
        derivative_y: &Value,
    ) -> Result<Value, Error> {
        let size = self.image_query(image, naga::ImageQuery::Size { level: None })?;
        let size = self.uint_to_float(size)?;
        if matches!(dim, TexDim::Cube | TexDim::ArrayCube) {
            return self.cube_gradient_lod(coordinate, derivative_x, derivative_y, &size);
        }

        if dim != TexDim::_3D || size.components.len() != 3 {
            return Err(Error::UnsupportedFeature(format!(
                "gradient-to-LOD rewrite for texture dimension {dim}"
            )));
        }
        let scaled_x = self.binary(naga::BinaryOperator::Multiply, derivative_x, &size, None)?;
        let scaled_y = self.binary(naga::BinaryOperator::Multiply, derivative_y, &size, None)?;
        let length_x = self.float_length(scaled_x)?;
        let length_y = self.float_length(scaled_y)?;
        let rho = self.float_minmax(&length_x, &length_y, false)?;
        self.float_mufu(rho, MuFuOp::Log2)
    }

    #[allow(clippy::too_many_lines, clippy::similar_names)]
    fn cube_gradient_lod(
        &mut self,
        coordinate: &Value,
        derivative_x: &Value,
        derivative_y: &Value,
        size: &Value,
    ) -> Result<Value, Error> {
        if coordinate.components.len() != 3
            || derivative_x.components.len() != 3
            || derivative_y.components.len() != 3
            || size.components.len() != 2
        {
            return Err(Error::UnsupportedFeature(
                "cube gradient-to-LOD operand shape mismatch".to_owned(),
            ));
        }

        let swizzle = |value: &Value, indices: [usize; 3]| Value {
            components: indices
                .into_iter()
                .map(|index| value.components[index].clone())
                .collect(),
            kind: value.kind,
        };
        let scalar = |value: &Value, index: usize| Value {
            components: vec![value.components[index].clone()],
            kind: value.kind,
        };
        let pair = |value: &Value| Value {
            components: value.components[..2].to_vec(),
            kind: value.kind,
        };

        let absolute = Value {
            components: coordinate
                .components
                .iter()
                .cloned()
                .map(Src::fabs)
                .collect(),
            kind: naga::ScalarKind::Float,
        };
        let abs_x = scalar(&absolute, 0);
        let abs_y = scalar(&absolute, 1);
        let abs_z = scalar(&absolute, 2);
        let max_xy = self.float_minmax(&abs_x, &abs_y, false)?;
        let max_xz = self.float_minmax(&abs_x, &abs_z, false)?;
        let condition_z = self.binary(naga::BinaryOperator::GreaterEqual, &abs_z, &max_xy, None)?;
        let condition_y = self.binary(naga::BinaryOperator::GreaterEqual, &abs_y, &max_xz, None)?;

        let choose_face = |this: &mut Self, value: &Value| -> Result<Value, Error> {
            let y_face = swizzle(value, [0, 2, 1]);
            let x_face = swizzle(value, [1, 2, 0]);
            let not_z = this.select(&condition_y, y_face, x_face)?;
            this.select(&condition_z, value.clone(), not_z)
        };
        let q = choose_face(self, coordinate)?;
        let dqdx = choose_face(self, derivative_x)?;
        let dqdy = choose_face(self, derivative_y)?;

        let reciprocal = self.float_mufu(scalar(&q, 2), MuFuOp::Rcp)?;
        let normalized_xy =
            self.binary(naga::BinaryOperator::Multiply, &pair(&q), &reciprocal, None)?;
        let quotient_derivative = |this: &mut Self, derivative: &Value| {
            let scaled_normalized = this.binary(
                naga::BinaryOperator::Multiply,
                &normalized_xy,
                &scalar(derivative, 2),
                None,
            )?;
            let difference = this.binary(
                naga::BinaryOperator::Subtract,
                &pair(derivative),
                &scaled_normalized,
                None,
            )?;
            this.binary(
                naga::BinaryOperator::Multiply,
                &difference,
                &reciprocal,
                None,
            )
        };
        let dx = quotient_derivative(self, &dqdx)?;
        let dy = quotient_derivative(self, &dqdy)?;
        let dx_squared = self.float_dot(&dx, &dx)?;
        let dy_squared = self.float_dot(&dy, &dy)?;
        let magnitude_squared = self.float_minmax(&dx_squared, &dy_squared, false)?;

        let edge = scalar(size, 0);
        let edge_squared = self.binary(naga::BinaryOperator::Multiply, &edge, &edge, None)?;
        let scaled_magnitude = self.binary(
            naga::BinaryOperator::Multiply,
            &edge_squared,
            &magnitude_squared,
            None,
        )?;
        let logarithm = self.float_mufu(scaled_magnitude, MuFuOp::Log2)?;
        let half_logarithm = self.binary(
            naga::BinaryOperator::Multiply,
            &logarithm,
            &Value {
                components: vec![Src::from(0.5_f32)],
                kind: naga::ScalarKind::Float,
            },
            None,
        )?;
        self.binary(
            naga::BinaryOperator::Add,
            &half_logarithm,
            &Value {
                components: vec![Src::from(-1.0_f32)],
                kind: naga::ScalarKind::Float,
            },
            None,
        )
    }

    fn uint_to_float(&mut self, value: Value) -> Result<Value, Error> {
        if value.kind != naga::ScalarKind::Uint {
            return Err(Error::UnsupportedFeature(
                "texture dimensions are not unsigned integers".to_owned(),
            ));
        }
        let mut components = Vec::with_capacity(value.components.len());
        for source in value.components {
            let destination = self.target.ssa_alloc.alloc(RegFile::GPR);
            self.emit(Instr::new(OpI2F {
                dst: Dst::from(destination),
                src: source,
                dst_type: FloatType::F32,
                src_type: IntType::U32,
                rnd_mode: FRndMode::NearestEven,
            }));
            components.push(Src::from(destination));
        }
        Ok(Value {
            components,
            kind: naga::ScalarKind::Float,
        })
    }

    fn texture_instruction_sources(
        &mut self,
        _texture_reference: TexRef,
        coordinate: Value,
        auxiliary: Vec<Src>,
    ) -> Result<[Src; 2], Error> {
        // Maxwell (SM50+) keeps coordinates/array index in source zero. For bindless
        // instructions the packed image/sampler handle is the first component of source one,
        // followed by LOD, offset, comparison, and multisample operands. This is deliberately
        // different from the single packed vec4/vec8 source used by SM30-SM40.
        let coordinate = Src::from(self.materialize(coordinate)?);
        let auxiliary = if auxiliary.is_empty() {
            Src::ZERO
        } else {
            Src::from(self.materialize(Value {
                components: auxiliary,
                kind: naga::ScalarKind::Uint,
            })?)
        };
        Ok([coordinate, auxiliary])
    }

    fn pack_texture_offset(
        &mut self,
        offset: naga::Handle<naga::Expression>,
    ) -> Result<Src, Error> {
        let offset = self.expression(offset)?;
        if offset.components.is_empty()
            || offset.components.len() > 4
            || !matches!(offset.kind, naga::ScalarKind::Sint | naga::ScalarKind::Uint)
        {
            return Err(Error::UnsupportedFeature(
                "texture offset must be a one-to-four component integer vector".to_owned(),
            ));
        }
        let mut packed = Src::ZERO;
        for (index, component) in offset.components.into_iter().enumerate() {
            let masked = self.target.ssa_alloc.alloc(RegFile::GPR);
            self.emit(Instr::new(OpLop2 {
                dst: Dst::from(masked),
                srcs: [component, Src::from(0xff_u32)],
                op: LogicOp2::And,
            }));
            let shifted = if index == 0 {
                masked
            } else {
                let shift = u32::try_from(index * 8).map_err(|_| {
                    Error::UnsupportedFeature("texture offset shift overflow".to_owned())
                })?;
                let shifted = self.target.ssa_alloc.alloc(RegFile::GPR);
                self.emit(Instr::new(OpShl {
                    dst: Dst::from(shifted),
                    src: Src::from(masked),
                    shift: Src::from(shift),
                    wrap: true,
                }));
                shifted
            };
            if index == 0 {
                packed = Src::from(shifted);
            } else {
                let combined = self.target.ssa_alloc.alloc(RegFile::GPR);
                self.emit(Instr::new(OpLop2 {
                    dst: Dst::from(combined),
                    srcs: [packed, Src::from(shifted)],
                    op: LogicOp2::Or,
                }));
                packed = Src::from(combined);
            }
        }
        Ok(packed)
    }

    fn global_expression(
        &self,
        handle: naga::Handle<naga::Expression>,
        description: &str,
    ) -> Result<naga::Handle<naga::GlobalVariable>, Error> {
        match self.source.expressions[handle] {
            naga::Expression::GlobalVariable(global) => Ok(global),
            naga::Expression::FunctionArgument(index) => {
                self.resource_arguments.get(&index).copied().ok_or_else(|| {
                    Error::UnsupportedFeature(format!(
                        "{description} function argument {index} has no resource identity"
                    ))
                })
            }
            ref expression => Err(Error::UnsupportedFeature(format!(
                "{description} expression {expression:?}"
            ))),
        }
    }

    fn sampled_resource(
        &mut self,
        handle: naga::Handle<naga::Expression>,
        description: &str,
    ) -> Result<(naga::Handle<naga::GlobalVariable>, Option<Value>), Error> {
        match self.source.expressions[handle] {
            naga::Expression::GlobalVariable(global) => Ok((global, None)),
            naga::Expression::FunctionArgument(index) => self
                .resource_arguments
                .get(&index)
                .copied()
                .map(|global| (global, None))
                .ok_or_else(|| {
                    Error::UnsupportedFeature(format!(
                        "{description} function argument {index} has no resource identity"
                    ))
                }),
            naga::Expression::Access { base, index } => {
                let (global, previous) = self.sampled_resource(base, description)?;
                if previous.is_some() {
                    return Err(Error::UnsupportedFeature(format!(
                        "nested {description} binding-array access"
                    )));
                }
                let index = self.expression(index)?;
                if index.components.len() != 1
                    || !matches!(index.kind, naga::ScalarKind::Uint | naga::ScalarKind::Sint)
                {
                    return Err(Error::UnsupportedFeature(format!(
                        "{description} binding-array index is not an integer scalar"
                    )));
                }
                Ok((global, Some(index)))
            }
            naga::Expression::AccessIndex { base, index } => {
                let (global, previous) = self.sampled_resource(base, description)?;
                if previous.is_some() {
                    return Err(Error::UnsupportedFeature(format!(
                        "nested {description} binding-array access"
                    )));
                }
                Ok((
                    global,
                    Some(Value {
                        components: vec![Src::from(index)],
                        kind: naga::ScalarKind::Uint,
                    }),
                ))
            }
            ref expression => Err(Error::UnsupportedFeature(format!(
                "{description} expression {expression:?}"
            ))),
        }
    }

    fn resource_array_target(
        &mut self,
        range: ResourceRange,
        index: Option<Value>,
    ) -> Result<Value, Error> {
        let Some(mut index) = index else {
            return Ok(Value {
                components: vec![Src::from(u32::from(range.target))],
                kind: naga::ScalarKind::Uint,
            });
        };
        index.kind = naga::ScalarKind::Uint;
        let maximum = Value {
            components: vec![Src::from(u32::from(range.count - 1))],
            kind: naga::ScalarKind::Uint,
        };
        let index = self.integer_minmax(&index, &maximum, true)?;
        self.binary(
            naga::BinaryOperator::Add,
            &index,
            &Value {
                components: vec![Src::from(u32::from(range.target))],
                kind: naga::ScalarKind::Uint,
            },
            None,
        )
    }

    #[allow(clippy::too_many_lines)]
    fn math(
        &mut self,
        fun: naga::MathFunction,
        arg: naga::Handle<naga::Expression>,
        arg1: Option<naga::Handle<naga::Expression>>,
        arg2: Option<naga::Handle<naga::Expression>>,
        arg3: Option<naga::Handle<naga::Expression>>,
    ) -> Result<Value, Error> {
        let matrix_shape = self.expression_matrix_shape(arg);
        let value = self.expression(arg)?;
        match fun {
            naga::MathFunction::Abs if value.kind == naga::ScalarKind::Float => Ok(Value {
                components: value.components.into_iter().map(Src::fabs).collect(),
                kind: value.kind,
            }),
            naga::MathFunction::Abs if value.kind == naga::ScalarKind::Sint => {
                let components = value
                    .components
                    .into_iter()
                    .map(|component| {
                        let sign =
                            self.emit_shift_right(component.clone(), Src::from(31_u32), true);
                        self.emit_abs_source(component, sign)
                    })
                    .collect();
                Ok(Value {
                    components,
                    kind: value.kind,
                })
            }
            naga::MathFunction::Sign if value.kind == naga::ScalarKind::Float => {
                let width = value.components.len();
                let zero = Value {
                    components: vec![Src::from(0.0_f32); width],
                    kind: naga::ScalarKind::Float,
                };
                let positive = self.binary(naga::BinaryOperator::Greater, &value, &zero, None)?;
                let negative = self.binary(naga::BinaryOperator::Less, &value, &zero, None)?;
                let one = Value {
                    components: vec![Src::from(1.0_f32); width],
                    kind: naga::ScalarKind::Float,
                };
                let minus_one = Value {
                    components: vec![Src::from(-1.0_f32); width],
                    kind: naga::ScalarKind::Float,
                };
                let non_positive = self.select(&negative, minus_one, zero)?;
                self.select(&positive, one, non_positive)
            }
            naga::MathFunction::Min | naga::MathFunction::Max => {
                let other = self.math_argument(fun, arg1, 1)?;
                if value.kind == naga::ScalarKind::Float {
                    self.float_minmax(&value, &other, fun == naga::MathFunction::Min)
                } else {
                    self.integer_minmax(&value, &other, fun == naga::MathFunction::Min)
                }
            }
            naga::MathFunction::Clamp => {
                let low = self.math_argument(fun, arg1, 1)?;
                let high = self.math_argument(fun, arg2, 2)?;
                if value.kind == naga::ScalarKind::Float {
                    let value = self.float_minmax(&value, &low, false)?;
                    self.float_minmax(&value, &high, true)
                } else {
                    let value = self.integer_minmax(&value, &low, false)?;
                    self.integer_minmax(&value, &high, true)
                }
            }
            naga::MathFunction::Mix => {
                let other = self.math_argument(fun, arg1, 1)?;
                let factor = self.math_argument(fun, arg2, 2)?;
                let one = Value {
                    components: vec![Src::from(1.0_f32)],
                    kind: naga::ScalarKind::Float,
                };
                let inverse = self.binary(naga::BinaryOperator::Subtract, &one, &factor, None)?;
                let left = self.binary(naga::BinaryOperator::Multiply, &value, &inverse, None)?;
                let right = self.binary(naga::BinaryOperator::Multiply, &other, &factor, None)?;
                self.binary(naga::BinaryOperator::Add, &left, &right, None)
            }
            naga::MathFunction::Saturate => {
                let zero = Value {
                    components: vec![Src::from(0.0_f32)],
                    kind: naga::ScalarKind::Float,
                };
                let one = Value {
                    components: vec![Src::from(1.0_f32)],
                    kind: naga::ScalarKind::Float,
                };
                let value = self.float_minmax(&value, &zero, false)?;
                self.float_minmax(&value, &one, true)
            }
            naga::MathFunction::Length => self.float_length(value),
            naga::MathFunction::Dot => {
                let other = self.math_argument(fun, arg1, 1)?;
                self.float_dot(&value, &other)
            }
            naga::MathFunction::Cross => {
                let other = self.math_argument(fun, arg1, 1)?;
                self.float_cross(&value, &other)
            }
            naga::MathFunction::Determinant => {
                let (columns, rows) = matrix_shape.ok_or_else(|| {
                    Error::UnsupportedFeature("determinant of a non-matrix value".to_owned())
                })?;
                if columns != rows {
                    return Err(Error::UnsupportedFeature(
                        "determinant of a non-square matrix".to_owned(),
                    ));
                }
                self.float_determinant(&value, columns)
            }
            naga::MathFunction::Floor => self.float_round(value, FRndMode::NegInf),
            naga::MathFunction::Fract => {
                let floor = self.float_round(value.clone(), FRndMode::NegInf)?;
                self.binary(naga::BinaryOperator::Subtract, &value, &floor, None)
            }
            naga::MathFunction::Ceil => self.float_round(value, FRndMode::PosInf),
            naga::MathFunction::Round => self.float_round(value, FRndMode::NearestEven),
            naga::MathFunction::Trunc => self.float_round(value, FRndMode::Zero),
            naga::MathFunction::Sqrt => self.float_mufu(value, MuFuOp::Sqrt),
            naga::MathFunction::InverseSqrt => self.float_mufu(value, MuFuOp::Rsq),
            naga::MathFunction::Sin => self.float_mufu(value, MuFuOp::Sin),
            naga::MathFunction::Cos => self.float_mufu(value, MuFuOp::Cos),
            naga::MathFunction::Exp2 => self.float_mufu(value, MuFuOp::Exp2),
            naga::MathFunction::Log2 => self.float_mufu(value, MuFuOp::Log2),
            naga::MathFunction::Exp => {
                let scale = Value {
                    components: vec![Src::from(std::f32::consts::LOG2_E)],
                    kind: naga::ScalarKind::Float,
                };
                let exponent = self.binary(naga::BinaryOperator::Multiply, &value, &scale, None)?;
                self.float_mufu(exponent, MuFuOp::Exp2)
            }
            naga::MathFunction::Log => {
                let logarithm = self.float_mufu(value, MuFuOp::Log2)?;
                let scale = Value {
                    components: vec![Src::from(std::f32::consts::LN_2)],
                    kind: naga::ScalarKind::Float,
                };
                self.binary(naga::BinaryOperator::Multiply, &logarithm, &scale, None)
            }
            naga::MathFunction::Atan => self.float_atan(&value),
            naga::MathFunction::Atan2 => {
                let x = self.math_argument(fun, arg1, 1)?;
                self.float_atan2(&value, &x)
            }
            naga::MathFunction::Pow => {
                let exponent = self.math_argument(fun, arg1, 1)?;
                let logarithm = self.float_mufu(value, MuFuOp::Log2)?;
                let product =
                    self.binary(naga::BinaryOperator::Multiply, &logarithm, &exponent, None)?;
                self.float_mufu(product, MuFuOp::Exp2)
            }
            naga::MathFunction::Normalize => {
                let length = self.float_length(value.clone())?;
                self.binary(naga::BinaryOperator::Divide, &value, &length, None)
            }
            naga::MathFunction::Reflect => {
                let normal = self.math_argument(fun, arg1, 1)?;
                let dot = self.float_dot(&normal, &value)?;
                let two = Value {
                    components: vec![Src::from(2.0_f32)],
                    kind: naga::ScalarKind::Float,
                };
                let scale = self.binary(naga::BinaryOperator::Multiply, &two, &dot, None)?;
                let projection =
                    self.binary(naga::BinaryOperator::Multiply, &normal, &scale, None)?;
                self.binary(naga::BinaryOperator::Subtract, &value, &projection, None)
            }
            naga::MathFunction::Step => {
                let x = self.math_argument(fun, arg1, 1)?;
                self.float_step(&value, &x)
            }
            naga::MathFunction::SmoothStep => {
                let edge1 = self.math_argument(fun, arg1, 1)?;
                let x = self.math_argument(fun, arg2, 2)?;
                let numerator = self.binary(naga::BinaryOperator::Subtract, &x, &value, None)?;
                let denominator =
                    self.binary(naga::BinaryOperator::Subtract, &edge1, &value, None)?;
                let t =
                    self.binary(naga::BinaryOperator::Divide, &numerator, &denominator, None)?;
                let zero = Value {
                    components: vec![Src::from(0.0_f32)],
                    kind: naga::ScalarKind::Float,
                };
                let one = Value {
                    components: vec![Src::from(1.0_f32)],
                    kind: naga::ScalarKind::Float,
                };
                let t = self.float_minmax(&t, &zero, false)?;
                let t = self.float_minmax(&t, &one, true)?;
                let square = self.binary(naga::BinaryOperator::Multiply, &t, &t, None)?;
                let two = Value {
                    components: vec![Src::from(2.0_f32)],
                    kind: naga::ScalarKind::Float,
                };
                let three = Value {
                    components: vec![Src::from(3.0_f32)],
                    kind: naga::ScalarKind::Float,
                };
                let twice = self.binary(naga::BinaryOperator::Multiply, &two, &t, None)?;
                let curve = self.binary(naga::BinaryOperator::Subtract, &three, &twice, None)?;
                self.binary(naga::BinaryOperator::Multiply, &square, &curve, None)
            }
            naga::MathFunction::ReverseBits => self.reverse_bits(value),
            naga::MathFunction::CountLeadingZeros => self.count_leading_zeros(value),
            naga::MathFunction::CountTrailingZeros => {
                let reversed = self.reverse_bits(value)?;
                self.count_leading_zeros(reversed)
            }
            naga::MathFunction::Unpack4x8unorm => self.unpack_4x8_unorm(&value),
            naga::MathFunction::ExtractBits => {
                let offset = self.math_argument(fun, arg1, 1)?;
                let count = self.math_argument(fun, arg2, 2)?;
                self.extract_bits(&value, &offset, &count)
            }
            naga::MathFunction::InsertBits => {
                let insert = self.math_argument(fun, arg1, 1)?;
                let offset = self.math_argument(fun, arg2, 2)?;
                let count = self.math_argument(fun, arg3, 3)?;
                self.insert_bits(&value, &insert, &offset, &count)
            }
            naga::MathFunction::Transpose => {
                let (columns, rows) = matrix_shape.ok_or_else(|| {
                    Error::UnsupportedFeature("transpose of a non-matrix value".to_owned())
                })?;
                let mut components = Vec::with_capacity(columns * rows);
                for output_column in 0..rows {
                    for output_row in 0..columns {
                        components
                            .push(value.components[output_row * rows + output_column].clone());
                    }
                }
                Ok(Value {
                    components,
                    kind: value.kind,
                })
            }
            _ => Err(Error::UnsupportedFeature(format!("math function {fun:?}"))),
        }
    }

    fn derivative(
        &mut self,
        axis: naga::DerivativeAxis,
        expression: naga::Handle<naga::Expression>,
    ) -> Result<Value, Error> {
        let value = self.expression(expression)?;
        if value.kind != naga::ScalarKind::Float || value.components.is_empty() {
            return Err(Error::UnsupportedFeature(
                "derivative argument is not a float value".to_owned(),
            ));
        }
        let sources = self.materialize_components(value);
        let mut components = Vec::with_capacity(sources.len());
        for source in sources {
            let mut emit_axis = |lane, ops| {
                let shuffled = self.target.ssa_alloc.alloc(RegFile::GPR);
                self.emit(Instr::new(OpShfl {
                    dst: Dst::from(shuffled),
                    in_bounds: Dst::None,
                    src: Src::from(source),
                    lane: Src::from(lane),
                    c: Src::from(0x3_u32 | (0x1c_u32 << 8)),
                    op: ShflOp::Bfly,
                }));
                let derivative = self.target.ssa_alloc.alloc(RegFile::GPR);
                self.emit(Instr::new(OpFSwzAdd {
                    dst: Dst::from(derivative),
                    srcs: [Src::from(shuffled), Src::from(source)],
                    rnd_mode: FRndMode::NearestEven,
                    ftz: false,
                    deriv_mode: TexDerivMode::Auto,
                    ops,
                }));
                derivative
            };
            let horizontal = || {
                [
                    FSwzAddOp::SubLeft,
                    FSwzAddOp::SubRight,
                    FSwzAddOp::SubLeft,
                    FSwzAddOp::SubRight,
                ]
            };
            let vertical = || {
                [
                    FSwzAddOp::SubLeft,
                    FSwzAddOp::SubLeft,
                    FSwzAddOp::SubRight,
                    FSwzAddOp::SubRight,
                ]
            };
            let result = match axis {
                naga::DerivativeAxis::X => emit_axis(1_u32, horizontal()),
                naga::DerivativeAxis::Y => emit_axis(2_u32, vertical()),
                naga::DerivativeAxis::Width => {
                    let dx = emit_axis(1_u32, horizontal());
                    let dy = emit_axis(2_u32, vertical());
                    let width = self.target.ssa_alloc.alloc(RegFile::GPR);
                    self.emit(Instr::new(OpFAdd {
                        dst: Dst::from(width),
                        srcs: [Src::from(dx).fabs(), Src::from(dy).fabs()],
                        saturate: false,
                        rnd_mode: FRndMode::NearestEven,
                        ftz: false,
                    }));
                    width
                }
            };
            components.push(Src::from(result));
        }
        Ok(Value {
            components,
            kind: naga::ScalarKind::Float,
        })
    }

    fn relational(
        &mut self,
        fun: naga::RelationalFunction,
        argument: naga::Handle<naga::Expression>,
    ) -> Result<Value, Error> {
        let argument = self.expression(argument)?;
        if argument.kind != naga::ScalarKind::Bool || argument.components.is_empty() {
            return Err(Error::UnsupportedFeature(format!(
                "relational function {fun:?} on a non-boolean value"
            )));
        }
        if !matches!(
            fun,
            naga::RelationalFunction::Any | naga::RelationalFunction::All
        ) {
            return Err(Error::UnsupportedFeature(format!(
                "relational function {fun:?}"
            )));
        }
        let mut components = argument.components.into_iter();
        let mut result = components.next().expect("argument is non-empty");
        for component in components {
            let dst = self.target.ssa_alloc.alloc(RegFile::Pred);
            let (second, third) = if fun == naga::RelationalFunction::All {
                (component, false.into())
            } else {
                (true.into(), component)
            };
            self.emit(Instr::new(OpPSetP {
                dsts: [Dst::from(dst), Dst::None],
                ops: [PredSetOp::And, PredSetOp::Or],
                srcs: [result, second, third],
            }));
            result = Src::from(dst);
        }
        Ok(Value {
            components: vec![result],
            kind: naga::ScalarKind::Bool,
        })
    }

    fn cast(
        &mut self,
        expression: naga::Handle<naga::Expression>,
        kind: naga::ScalarKind,
        convert: Option<naga::Bytes>,
    ) -> Result<Value, Error> {
        let value = self.expression(expression)?;
        if convert.is_none() {
            return Ok(Value {
                components: value.components,
                kind,
            });
        }
        if convert != Some(4) {
            return Err(Error::UnsupportedFeature(format!(
                "conversion to {kind:?} with width {convert:?}"
            )));
        }
        let mut components = Vec::with_capacity(value.components.len());
        for source in value.components {
            let dst = self.target.ssa_alloc.alloc(RegFile::GPR);
            let instruction = match (value.kind, kind) {
                (naga::ScalarKind::Uint, naga::ScalarKind::Float) => Instr::new(OpI2F {
                    dst: Dst::from(dst),
                    src: source,
                    dst_type: FloatType::F32,
                    src_type: IntType::U32,
                    rnd_mode: FRndMode::NearestEven,
                }),
                (naga::ScalarKind::Sint, naga::ScalarKind::Float) => Instr::new(OpI2F {
                    dst: Dst::from(dst),
                    src: source,
                    dst_type: FloatType::F32,
                    src_type: IntType::I32,
                    rnd_mode: FRndMode::NearestEven,
                }),
                (naga::ScalarKind::Bool, naga::ScalarKind::Float) => Instr::new(OpSel {
                    dst: Dst::from(dst),
                    cond: source,
                    srcs: [Src::from(1.0_f32), Src::from(0.0_f32)],
                }),
                (
                    naga::ScalarKind::Uint | naga::ScalarKind::Sint,
                    naga::ScalarKind::Uint | naga::ScalarKind::Sint,
                ) => Instr::new(OpMov {
                    dst: Dst::from(dst),
                    src: source,
                    quad_lanes: 0xf,
                }),
                (naga::ScalarKind::Float, naga::ScalarKind::Uint) => Instr::new(OpF2I {
                    dst: Dst::from(dst),
                    src: source,
                    src_type: FloatType::F32,
                    dst_type: IntType::U32,
                    rnd_mode: FRndMode::Zero,
                    ftz: false,
                }),
                (naga::ScalarKind::Float, naga::ScalarKind::Sint) => Instr::new(OpF2I {
                    dst: Dst::from(dst),
                    src: source,
                    src_type: FloatType::F32,
                    dst_type: IntType::I32,
                    rnd_mode: FRndMode::Zero,
                    ftz: false,
                }),
                (source_kind, target_kind) if source_kind == target_kind => Instr::new(OpMov {
                    dst: Dst::from(dst),
                    src: source,
                    quad_lanes: 0xf,
                }),
                (source_kind, target_kind) => {
                    return Err(Error::UnsupportedFeature(format!(
                        "conversion from {source_kind:?} to {target_kind:?}"
                    )));
                }
            };
            self.emit(instruction);
            components.push(Src::from(dst));
        }
        Ok(Value { components, kind })
    }

    fn math_argument(
        &mut self,
        fun: naga::MathFunction,
        argument: Option<naga::Handle<naga::Expression>>,
        index: usize,
    ) -> Result<Value, Error> {
        let handle = argument.ok_or_else(|| {
            Error::UnsupportedFeature(format!("missing argument {index} for {fun:?}"))
        })?;
        self.expression(handle)
    }

    fn reverse_bits(&mut self, value: Value) -> Result<Value, Error> {
        if !matches!(value.kind, naga::ScalarKind::Uint | naga::ScalarKind::Sint) {
            return Err(Error::UnsupportedFeature(
                "reverseBits argument is not an integer".to_owned(),
            ));
        }
        let components = value
            .components
            .into_iter()
            .map(|source| {
                let destination = self.target.ssa_alloc.alloc(RegFile::GPR);
                self.emit(Instr::new(OpBfe {
                    dst: Dst::from(destination),
                    base: source,
                    range: Src::from(32_u32 << 8),
                    signed: false,
                    reverse: true,
                }));
                Src::from(destination)
            })
            .collect();
        Ok(Value {
            components,
            kind: value.kind,
        })
    }

    fn count_leading_zeros(&mut self, value: Value) -> Result<Value, Error> {
        if !matches!(value.kind, naga::ScalarKind::Uint | naga::ScalarKind::Sint) {
            return Err(Error::UnsupportedFeature(
                "countLeadingZeros argument is not an integer".to_owned(),
            ));
        }
        let components = value
            .components
            .into_iter()
            .map(|source| {
                let destination = self.target.ssa_alloc.alloc(RegFile::GPR);
                self.emit(Instr::new(OpFlo {
                    dst: Dst::from(destination),
                    src: source,
                    signed: false,
                    return_shift_amount: true,
                }));
                Src::from(destination)
            })
            .collect();
        Ok(Value {
            components,
            kind: value.kind,
        })
    }

    fn unpack_4x8_unorm(&mut self, value: &Value) -> Result<Value, Error> {
        if value.kind != naga::ScalarKind::Uint || value.components.len() != 1 {
            return Err(Error::UnsupportedFeature(
                "unpack4x8unorm argument is not a u32 scalar".to_owned(),
            ));
        }
        let source = value.components[0].clone();
        let mut components = Vec::with_capacity(4);
        for index in 0..4_u32 {
            let extracted = self.target.ssa_alloc.alloc(RegFile::GPR);
            self.emit(Instr::new(OpBfe {
                dst: Dst::from(extracted),
                base: source.clone(),
                range: Src::from((8_u32 << 8) | (index * 8)),
                signed: false,
                reverse: false,
            }));
            let converted = self.target.ssa_alloc.alloc(RegFile::GPR);
            self.emit(Instr::new(OpI2F {
                dst: Dst::from(converted),
                src: Src::from(extracted),
                dst_type: FloatType::F32,
                src_type: IntType::U32,
                rnd_mode: FRndMode::NearestEven,
            }));
            let normalized = self.target.ssa_alloc.alloc(RegFile::GPR);
            self.emit(Instr::new(OpFMul {
                dst: Dst::from(normalized),
                srcs: [Src::from(converted), Src::from(1.0_f32 / 255.0_f32)],
                saturate: false,
                rnd_mode: FRndMode::NearestEven,
                ftz: false,
                dnz: false,
            }));
            components.push(Src::from(normalized));
        }
        Ok(Value {
            components,
            kind: naga::ScalarKind::Float,
        })
    }

    fn clamped_bit_range(
        &mut self,
        offset: &Value,
        count: &Value,
    ) -> Result<(Value, Value), Error> {
        if offset.kind != naga::ScalarKind::Uint || count.kind != naga::ScalarKind::Uint {
            return Err(Error::UnsupportedFeature(
                "bitfield offset and count must be unsigned integers".to_owned(),
            ));
        }
        let width = Value {
            components: vec![Src::from(32_u32)],
            kind: naga::ScalarKind::Uint,
        };
        let offset = self.integer_minmax(offset, &width, true)?;
        let remaining = self.binary(naga::BinaryOperator::Subtract, &width, &offset, None)?;
        let count = self.integer_minmax(count, &remaining, true)?;
        Ok((offset, count))
    }

    fn extract_bits(
        &mut self,
        value: &Value,
        offset: &Value,
        count: &Value,
    ) -> Result<Value, Error> {
        if !matches!(value.kind, naga::ScalarKind::Uint | naga::ScalarKind::Sint) {
            return Err(Error::UnsupportedFeature(
                "extractBits from a non-integer value".to_owned(),
            ));
        }
        let (offset, count) = self.clamped_bit_range(offset, count)?;
        let count_bytes = self.binary(
            naga::BinaryOperator::ShiftLeft,
            &count,
            &Value {
                components: vec![Src::from(8_u32)],
                kind: naga::ScalarKind::Uint,
            },
            None,
        )?;
        let range = self.binary(
            naga::BinaryOperator::InclusiveOr,
            &offset,
            &count_bytes,
            None,
        )?;
        let width = value.components.len().max(range.components.len());
        if (value.components.len() != 1 && value.components.len() != width)
            || (range.components.len() != 1 && range.components.len() != width)
        {
            return Err(Error::UnsupportedFeature(
                "extractBits operand width mismatch".to_owned(),
            ));
        }
        let mut components = Vec::with_capacity(width);
        for index in 0..width {
            let dst = self.target.ssa_alloc.alloc(RegFile::GPR);
            self.emit(Instr::new(OpBfe {
                dst: Dst::from(dst),
                base: value.components[if value.components.len() == 1 {
                    0
                } else {
                    index
                }]
                .clone(),
                range: range.components[if range.components.len() == 1 {
                    0
                } else {
                    index
                }]
                .clone(),
                signed: value.kind == naga::ScalarKind::Sint,
                reverse: false,
            }));
            components.push(Src::from(dst));
        }
        let extracted = Value {
            components,
            kind: value.kind,
        };
        let nonzero = self.binary(
            naga::BinaryOperator::NotEqual,
            &count,
            &Value {
                components: vec![Src::ZERO],
                kind: naga::ScalarKind::Uint,
            },
            None,
        )?;
        self.select(
            &nonzero,
            extracted,
            Value {
                components: vec![Src::ZERO; width],
                kind: value.kind,
            },
        )
    }

    #[allow(clippy::too_many_lines)]
    fn insert_bits(
        &mut self,
        base: &Value,
        insert: &Value,
        offset: &Value,
        count: &Value,
    ) -> Result<Value, Error> {
        if base.kind != insert.kind
            || !matches!(base.kind, naga::ScalarKind::Uint | naga::ScalarKind::Sint)
        {
            return Err(Error::UnsupportedFeature(
                "insertBits requires matching integer values".to_owned(),
            ));
        }
        let (offset, count) = self.clamped_bit_range(offset, count)?;
        let one = Value {
            components: vec![Src::from(1_u32)],
            kind: naga::ScalarKind::Uint,
        };
        let shifted = self.binary(naga::BinaryOperator::ShiftLeft, &one, &count, None)?;
        let raw_mask = self.binary(naga::BinaryOperator::Subtract, &shifted, &one, None)?;
        let full = self.binary(
            naga::BinaryOperator::Equal,
            &count,
            &Value {
                components: vec![Src::from(32_u32)],
                kind: naga::ScalarKind::Uint,
            },
            None,
        )?;
        let mask = self.select(
            &full,
            Value {
                components: vec![Src::from(u32::MAX)],
                kind: naga::ScalarKind::Uint,
            },
            raw_mask,
        )?;
        let mask = self.binary(naga::BinaryOperator::ShiftLeft, &mask, &offset, None)?;
        let mask = Value {
            components: mask.components,
            kind: base.kind,
        };
        let inverse_mask = Value {
            components: mask.components.iter().cloned().map(Src::bnot).collect(),
            kind: base.kind,
        };
        let kept = self.binary(naga::BinaryOperator::And, base, &inverse_mask, None)?;
        let shifted_insert = self.binary(naga::BinaryOperator::ShiftLeft, insert, &offset, None)?;
        let inserted = self.binary(naga::BinaryOperator::And, &shifted_insert, &mask, None)?;
        self.binary(naga::BinaryOperator::InclusiveOr, &kept, &inserted, None)
    }

    fn float_minmax(&mut self, left: &Value, right: &Value, min: bool) -> Result<Value, Error> {
        if left.kind != naga::ScalarKind::Float || right.kind != naga::ScalarKind::Float {
            return Err(Error::UnsupportedFeature("non-float min/max".to_owned()));
        }
        let width = left.components.len().max(right.components.len());
        if (left.components.len() != 1 && left.components.len() != width)
            || (right.components.len() != 1 && right.components.len() != width)
        {
            return Err(Error::UnsupportedFeature(
                "min/max operands with incompatible widths".to_owned(),
            ));
        }
        let mut components = Vec::with_capacity(width);
        for index in 0..width {
            let left = left.components[if left.components.len() == 1 { 0 } else { index }].clone();
            let right = right.components[if right.components.len() == 1 {
                0
            } else {
                index
            }]
            .clone();
            let dst = self.target.ssa_alloc.alloc(RegFile::GPR);
            self.emit(Instr::new(OpFMnMx {
                dst: Dst::from(dst),
                srcs: [left, right],
                min: min.into(),
                ftz: false,
            }));
            components.push(Src::from(dst));
        }
        Ok(Value {
            components,
            kind: naga::ScalarKind::Float,
        })
    }

    fn integer_minmax(&mut self, left: &Value, right: &Value, min: bool) -> Result<Value, Error> {
        if left.kind != right.kind
            || !matches!(left.kind, naga::ScalarKind::Uint | naga::ScalarKind::Sint)
        {
            return Err(Error::UnsupportedFeature(
                "non-integer integer min/max".to_owned(),
            ));
        }
        let width = left.components.len().max(right.components.len());
        if (left.components.len() != 1 && left.components.len() != width)
            || (right.components.len() != 1 && right.components.len() != width)
        {
            return Err(Error::UnsupportedFeature(
                "integer min/max operands with incompatible widths".to_owned(),
            ));
        }
        let cmp_type = if left.kind == naga::ScalarKind::Uint {
            IntCmpType::U32
        } else {
            IntCmpType::I32
        };
        let mut components = Vec::with_capacity(width);
        for index in 0..width {
            let left = left.components[if left.components.len() == 1 { 0 } else { index }].clone();
            let right = right.components[if right.components.len() == 1 {
                0
            } else {
                index
            }]
            .clone();
            let dst = self.target.ssa_alloc.alloc(RegFile::GPR);
            self.emit(Instr::new(OpIMnMx {
                dst: Dst::from(dst),
                cmp_type,
                srcs: [left, right],
                min: min.into(),
            }));
            components.push(Src::from(dst));
        }
        Ok(Value {
            components,
            kind: left.kind,
        })
    }

    fn float_length(&mut self, value: Value) -> Result<Value, Error> {
        if value.kind != naga::ScalarKind::Float || value.components.is_empty() {
            return Err(Error::UnsupportedFeature("non-float length".to_owned()));
        }
        let mut sum = None;
        for component in value.components {
            let square = self.target.ssa_alloc.alloc(RegFile::GPR);
            self.emit(Instr::new(OpFMul {
                dst: Dst::from(square),
                srcs: [component.clone(), component],
                saturate: false,
                rnd_mode: FRndMode::NearestEven,
                ftz: false,
                dnz: false,
            }));
            sum = Some(match sum {
                None => Src::from(square),
                Some(previous) => {
                    let dst = self.target.ssa_alloc.alloc(RegFile::GPR);
                    self.emit(Instr::new(OpFAdd {
                        dst: Dst::from(dst),
                        srcs: [previous, Src::from(square)],
                        saturate: false,
                        rnd_mode: FRndMode::NearestEven,
                        ftz: false,
                    }));
                    Src::from(dst)
                }
            });
        }
        let dst = self.target.ssa_alloc.alloc(RegFile::GPR);
        self.emit(Instr::new(OpMuFu {
            dst: Dst::from(dst),
            op: MuFuOp::Sqrt,
            src: sum.expect("length input is non-empty"),
            op_type: FloatType::F32,
        }));
        Ok(Value {
            components: vec![Src::from(dst)],
            kind: naga::ScalarKind::Float,
        })
    }

    fn float_dot(&mut self, left: &Value, right: &Value) -> Result<Value, Error> {
        if left.kind != naga::ScalarKind::Float
            || right.kind != naga::ScalarKind::Float
            || left.components.is_empty()
            || left.components.len() != right.components.len()
        {
            return Err(Error::UnsupportedFeature(format!(
                "dot operands must be equally-sized float vectors (left {:?}/{}, right {:?}/{})",
                left.kind,
                left.components.len(),
                right.kind,
                right.components.len()
            )));
        }
        let mut sum = None;
        for (left, right) in left.components.iter().zip(&right.components) {
            let product = self.target.ssa_alloc.alloc(RegFile::GPR);
            self.emit(Instr::new(OpFMul {
                dst: Dst::from(product),
                srcs: [left.clone(), right.clone()],
                saturate: false,
                rnd_mode: FRndMode::NearestEven,
                ftz: false,
                dnz: false,
            }));
            sum = Some(match sum {
                None => Src::from(product),
                Some(previous) => {
                    let dst = self.target.ssa_alloc.alloc(RegFile::GPR);
                    self.emit(Instr::new(OpFAdd {
                        dst: Dst::from(dst),
                        srcs: [previous, Src::from(product)],
                        saturate: false,
                        rnd_mode: FRndMode::NearestEven,
                        ftz: false,
                    }));
                    Src::from(dst)
                }
            });
        }
        Ok(Value {
            components: vec![sum.expect("dot operands are non-empty")],
            kind: naga::ScalarKind::Float,
        })
    }

    fn float_cross(&mut self, left: &Value, right: &Value) -> Result<Value, Error> {
        if left.kind != naga::ScalarKind::Float
            || right.kind != naga::ScalarKind::Float
            || left.components.len() != 3
            || right.components.len() != 3
        {
            return Err(Error::UnsupportedFeature(
                "cross operands must be three-component float vectors".to_owned(),
            ));
        }
        let product = |left: Src, right: Src| Value {
            components: vec![left, right],
            kind: naga::ScalarKind::Float,
        };
        let pairs = [(1, 2, 2, 1), (2, 0, 0, 2), (0, 1, 1, 0)];
        let mut components = Vec::with_capacity(3);
        for (left_a, right_a, left_b, right_b) in pairs {
            let a = product(
                left.components[left_a].clone(),
                right.components[right_a].clone(),
            );
            let a = self.binary_component(
                naga::BinaryOperator::Multiply,
                naga::ScalarKind::Float,
                a.components[0].clone(),
                a.components[1].clone(),
            )?;
            self.emit(a.0);
            let b = product(
                left.components[left_b].clone(),
                right.components[right_b].clone(),
            );
            let b = self.binary_component(
                naga::BinaryOperator::Multiply,
                naga::ScalarKind::Float,
                b.components[0].clone(),
                b.components[1].clone(),
            )?;
            self.emit(b.0);
            let difference = self.binary_component(
                naga::BinaryOperator::Subtract,
                naga::ScalarKind::Float,
                a.1,
                b.1,
            )?;
            self.emit(difference.0);
            components.push(difference.1);
        }
        Ok(Value {
            components,
            kind: naga::ScalarKind::Float,
        })
    }

    fn float_determinant(&mut self, matrix: &Value, size: usize) -> Result<Value, Error> {
        if matrix.kind != naga::ScalarKind::Float
            || !(2..=4).contains(&size)
            || matrix.components.len() != size * size
        {
            return Err(Error::UnsupportedFeature(
                "determinant operand must be a square float matrix".to_owned(),
            ));
        }
        self.float_determinant_components(&matrix.components, size)
    }

    fn float_determinant_components(
        &mut self,
        components: &[Src],
        size: usize,
    ) -> Result<Value, Error> {
        if size == 1 {
            return Ok(Value {
                components: vec![components[0].clone()],
                kind: naga::ScalarKind::Float,
            });
        }
        let mut result = None;
        for column in 0..size {
            let mut minor = Vec::with_capacity((size - 1) * (size - 1));
            for minor_column in 0..size {
                if minor_column == column {
                    continue;
                }
                for row in 1..size {
                    minor.push(components[minor_column * size + row].clone());
                }
            }
            let minor = self.float_determinant_components(&minor, size - 1)?;
            let coefficient = Value {
                components: vec![components[column * size].clone()],
                kind: naga::ScalarKind::Float,
            };
            let term = self.binary(naga::BinaryOperator::Multiply, &coefficient, &minor, None)?;
            result = Some(match result {
                None => term,
                Some(accumulator) => self.binary(
                    if column % 2 == 0 {
                        naga::BinaryOperator::Add
                    } else {
                        naga::BinaryOperator::Subtract
                    },
                    &accumulator,
                    &term,
                    None,
                )?,
            });
        }
        Ok(result.expect("matrix size is nonzero"))
    }

    fn float_round(&mut self, value: Value, rnd_mode: FRndMode) -> Result<Value, Error> {
        if value.kind != naga::ScalarKind::Float {
            return Err(Error::UnsupportedFeature(
                "rounding a non-float value".to_owned(),
            ));
        }
        let mut components = Vec::with_capacity(value.components.len());
        for source in value.components {
            let integer = self.target.ssa_alloc.alloc(RegFile::GPR);
            self.emit(Instr::new(OpF2I {
                dst: Dst::from(integer),
                src: source,
                src_type: FloatType::F32,
                dst_type: IntType::I32,
                rnd_mode,
                ftz: false,
            }));
            let dst = self.target.ssa_alloc.alloc(RegFile::GPR);
            self.emit(Instr::new(OpI2F {
                dst: Dst::from(dst),
                src: Src::from(integer),
                dst_type: FloatType::F32,
                src_type: IntType::I32,
                rnd_mode: FRndMode::NearestEven,
            }));
            components.push(Src::from(dst));
        }
        Ok(Value {
            components,
            kind: naga::ScalarKind::Float,
        })
    }

    fn float_atan(&mut self, value: &Value) -> Result<Value, Error> {
        self.float_atan2(
            value,
            &Value {
                components: vec![Src::from(1.0_f32)],
                kind: naga::ScalarKind::Float,
            },
        )
    }

    fn float_atan2(&mut self, y: &Value, x: &Value) -> Result<Value, Error> {
        if y.kind != naga::ScalarKind::Float || x.kind != naga::ScalarKind::Float {
            return Err(Error::UnsupportedFeature(
                "atan2 arguments are not float values".to_owned(),
            ));
        }
        let zero = Value {
            components: vec![Src::from(0.0_f32)],
            kind: naga::ScalarKind::Float,
        };
        let abs_x = Value {
            components: x.components.iter().cloned().map(Src::fabs).collect(),
            kind: naga::ScalarKind::Float,
        };
        let abs_y = Value {
            components: y.components.iter().cloned().map(Src::fabs).collect(),
            kind: naga::ScalarKind::Float,
        };
        let minimum = self.float_minmax(&abs_x, &abs_y, true)?;
        let maximum = self.float_minmax(&abs_x, &abs_y, false)?;
        let ratio = self.binary(naga::BinaryOperator::Divide, &minimum, &maximum, None)?;
        let ratio_squared = self.binary(naga::BinaryOperator::Multiply, &ratio, &ratio, None)?;

        // Rajan et al.'s minimax atan approximation on [0, 1], followed by
        // the standard atan2 quadrant reconstruction.
        let mut polynomial = Value {
            components: vec![Src::from(-0.046_496_473_f32)],
            kind: naga::ScalarKind::Float,
        };
        for coefficient in [0.159_314_22_f32, -0.327_622_77_f32] {
            polynomial = self.binary(
                naga::BinaryOperator::Multiply,
                &polynomial,
                &ratio_squared,
                None,
            )?;
            polynomial = self.binary(
                naga::BinaryOperator::Add,
                &polynomial,
                &Value {
                    components: vec![Src::from(coefficient)],
                    kind: naga::ScalarKind::Float,
                },
                None,
            )?;
        }
        polynomial = self.binary(
            naga::BinaryOperator::Multiply,
            &polynomial,
            &ratio_squared,
            None,
        )?;
        polynomial = self.binary(naga::BinaryOperator::Multiply, &polynomial, &ratio, None)?;
        let mut angle = self.binary(naga::BinaryOperator::Add, &polynomial, &ratio, None)?;

        let y_dominant = self.binary(naga::BinaryOperator::Greater, &abs_y, &abs_x, None)?;
        let complement = self.binary(
            naga::BinaryOperator::Subtract,
            &Value {
                components: vec![Src::from(std::f32::consts::FRAC_PI_2)],
                kind: naga::ScalarKind::Float,
            },
            &angle,
            None,
        )?;
        angle = self.select(&y_dominant, complement, angle)?;
        let x_negative = self.binary(naga::BinaryOperator::Less, x, &zero, None)?;
        let opposite = self.binary(
            naga::BinaryOperator::Subtract,
            &Value {
                components: vec![Src::from(std::f32::consts::PI)],
                kind: naga::ScalarKind::Float,
            },
            &angle,
            None,
        )?;
        angle = self.select(&x_negative, opposite, angle)?;
        let y_negative = self.binary(naga::BinaryOperator::Less, y, &zero, None)?;
        let negated = Value {
            components: angle.components.iter().cloned().map(Src::fneg).collect(),
            kind: naga::ScalarKind::Float,
        };
        self.select(&y_negative, negated, angle)
    }

    fn float_mufu(&mut self, value: Value, op: MuFuOp) -> Result<Value, Error> {
        if value.kind != naga::ScalarKind::Float {
            return Err(Error::UnsupportedFeature(
                "transcendental operation on a non-float value".to_owned(),
            ));
        }
        let mut components = Vec::with_capacity(value.components.len());
        for source in value.components {
            let dst = self.target.ssa_alloc.alloc(RegFile::GPR);
            self.emit(Instr::new(OpMuFu {
                dst: Dst::from(dst),
                op,
                src: source,
                op_type: FloatType::F32,
            }));
            components.push(Src::from(dst));
        }
        Ok(Value {
            components,
            kind: naga::ScalarKind::Float,
        })
    }

    fn float_step(&mut self, edge: &Value, x: &Value) -> Result<Value, Error> {
        if edge.kind != naga::ScalarKind::Float || x.kind != naga::ScalarKind::Float {
            return Err(Error::UnsupportedFeature("non-float step".to_owned()));
        }
        let width = edge.components.len().max(x.components.len());
        if (edge.components.len() != 1 && edge.components.len() != width)
            || (x.components.len() != 1 && x.components.len() != width)
        {
            return Err(Error::UnsupportedFeature(
                "step operands with incompatible widths".to_owned(),
            ));
        }
        let mut components = Vec::with_capacity(width);
        for index in 0..width {
            let edge = edge.components[if edge.components.len() == 1 { 0 } else { index }].clone();
            let x = x.components[if x.components.len() == 1 { 0 } else { index }].clone();
            let (comparison, condition) =
                self.float_comparison(naga::BinaryOperator::Less, x, edge)?;
            self.emit(comparison);
            let dst = self.target.ssa_alloc.alloc(RegFile::GPR);
            self.emit(Instr::new(OpSel {
                dst: Dst::from(dst),
                cond: condition,
                srcs: [Src::from(0.0_f32), Src::from(1.0_f32)],
            }));
            components.push(Src::from(dst));
        }
        Ok(Value {
            components,
            kind: naga::ScalarKind::Float,
        })
    }

    fn expression_type(
        &self,
        handle: naga::Handle<naga::Expression>,
    ) -> Option<naga::Handle<naga::Type>> {
        match self.source.expressions[handle] {
            naga::Expression::Compose { ty, .. } | naga::Expression::ZeroValue(ty) => Some(ty),
            naga::Expression::GlobalVariable(global) => {
                Some(self.module.global_variables[global].ty)
            }
            naga::Expression::FunctionArgument(index) => self
                .source
                .arguments
                .get(index as usize)
                .map(|argument| argument.ty),
            naga::Expression::CallResult(function) => self.module.functions[function]
                .result
                .as_ref()
                .map(|result| result.ty),
            naga::Expression::AccessIndex { base, index } => {
                let base = self.expression_type(base)?;
                match self.module.types[base].inner {
                    naga::TypeInner::Struct { ref members, .. } => {
                        members.get(index as usize).map(|member| member.ty)
                    }
                    naga::TypeInner::Array { base, .. }
                    | naga::TypeInner::BindingArray { base, .. } => Some(base),
                    naga::TypeInner::Matrix { rows, scalar, .. } => {
                        self.vector_type_handle(rows, scalar)
                    }
                    naga::TypeInner::Vector { scalar, .. } => self.scalar_type_handle(scalar),
                    _ => None,
                }
            }
            naga::Expression::Access { base, .. } => {
                let base = self.expression_type(base)?;
                match self.module.types[base].inner {
                    naga::TypeInner::Array { base, .. }
                    | naga::TypeInner::BindingArray { base, .. } => Some(base),
                    naga::TypeInner::Matrix { rows, scalar, .. } => {
                        self.vector_type_handle(rows, scalar)
                    }
                    naga::TypeInner::Vector { scalar, .. } => self.scalar_type_handle(scalar),
                    _ => None,
                }
            }
            naga::Expression::Load { pointer } if self.pointer_is_argument(pointer) => {
                self.argument_pointer(pointer).ok().map(|(_, ty, _)| ty)
            }
            naga::Expression::Load { pointer } if self.pointer_is_local(pointer) => {
                self.local_pointer(pointer).ok().map(|(_, ty, _)| ty)
            }
            naga::Expression::Load { pointer } => self.pointer_type(pointer),
            naga::Expression::Math {
                fun: naga::MathFunction::Transpose,
                arg,
                ..
            } => {
                let input = self.expression_type(arg)?;
                let naga::TypeInner::Matrix {
                    columns,
                    rows,
                    scalar,
                } = self.module.types[input].inner
                else {
                    return None;
                };
                self.module.types.iter().find_map(|(handle, candidate)| {
                    matches!(
                        candidate.inner,
                        naga::TypeInner::Matrix {
                            columns: candidate_columns,
                            rows: candidate_rows,
                            scalar: candidate_scalar,
                        } if candidate_columns == rows
                            && candidate_rows == columns
                            && candidate_scalar == scalar
                    )
                    .then_some(handle)
                })
            }
            _ => None,
        }
    }

    fn expression_is_resource(&self, handle: naga::Handle<naga::Expression>) -> bool {
        self.expression_type(handle).is_some_and(|ty| {
            matches!(
                self.module.types[ty].inner,
                naga::TypeInner::Image { .. }
                    | naga::TypeInner::Sampler { .. }
                    | naga::TypeInner::BindingArray { .. }
            )
        })
    }

    fn pointer_type(
        &self,
        handle: naga::Handle<naga::Expression>,
    ) -> Option<naga::Handle<naga::Type>> {
        match self.source.expressions[handle] {
            naga::Expression::FunctionArgument(index) => {
                let argument = self.source.arguments.get(index as usize)?;
                match self.module.types[argument.ty].inner {
                    naga::TypeInner::Pointer { base, .. } => Some(base),
                    _ => None,
                }
            }
            naga::Expression::GlobalVariable(global) => {
                Some(self.module.global_variables[global].ty)
            }
            naga::Expression::LocalVariable(local) => Some(self.source.local_variables[local].ty),
            naga::Expression::Access { base, .. } => {
                let base = self.pointer_type(base)?;
                match self.module.types[base].inner {
                    naga::TypeInner::Array { base, .. } => Some(base),
                    naga::TypeInner::Matrix { rows, scalar, .. } => {
                        self.vector_type_handle(rows, scalar)
                    }
                    naga::TypeInner::Vector { scalar, .. } => self.scalar_type_handle(scalar),
                    _ => None,
                }
            }
            naga::Expression::AccessIndex { base, index } => {
                let base = self.pointer_type(base)?;
                match self.module.types[base].inner {
                    naga::TypeInner::Struct { ref members, .. } => {
                        members.get(index as usize).map(|member| member.ty)
                    }
                    naga::TypeInner::Array { base, .. } => Some(base),
                    naga::TypeInner::Matrix { rows, scalar, .. } => {
                        self.vector_type_handle(rows, scalar)
                    }
                    naga::TypeInner::Vector { scalar, .. } => self.scalar_type_handle(scalar),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn vector_type_handle(
        &self,
        size: naga::VectorSize,
        scalar: naga::Scalar,
    ) -> Option<naga::Handle<naga::Type>> {
        self.module.types.iter().find_map(|(handle, candidate)| {
            matches!(
                candidate.inner,
                naga::TypeInner::Vector {
                    size: candidate_size,
                    scalar: candidate_scalar,
                } if candidate_size == size && candidate_scalar == scalar
            )
            .then_some(handle)
        })
    }

    fn scalar_type_handle(&self, scalar: naga::Scalar) -> Option<naga::Handle<naga::Type>> {
        self.module.types.iter().find_map(|(handle, candidate)| {
            matches!(
                candidate.inner,
                naga::TypeInner::Scalar(candidate_scalar) if candidate_scalar == scalar
            )
            .then_some(handle)
        })
    }

    fn access_index(
        &mut self,
        base_handle: naga::Handle<naga::Expression>,
        index: u32,
    ) -> Result<Value, Error> {
        let base = self.expression(base_handle)?;
        let Some(ty) = self.expression_type(base_handle) else {
            let component = base
                .components
                .get(index as usize)
                .cloned()
                .ok_or_else(|| {
                    Error::UnsupportedFeature("inferred vector index out of bounds".to_owned())
                })?;
            return Ok(Value {
                components: vec![component],
                kind: base.kind,
            });
        };
        let (offset, count, kind) = match self.module.types[ty].inner {
            naga::TypeInner::Vector { size, scalar } => {
                if index as usize >= vector_size(size) {
                    return Err(Error::UnsupportedFeature(
                        "vector index out of bounds".to_owned(),
                    ));
                }
                (index as usize, 1, scalar.kind)
            }
            naga::TypeInner::Matrix { rows, scalar, .. } => {
                let rows = vector_size(rows);
                (index as usize * rows, rows, scalar.kind)
            }
            naga::TypeInner::Array {
                base: element,
                size: naga::ArraySize::Constant(size),
                ..
            } => {
                if index >= size.get() {
                    return Err(Error::UnsupportedFeature(
                        "array index out of bounds".to_owned(),
                    ));
                }
                let count = flat_type_components(self.module, element)?;
                (
                    index as usize * count,
                    count,
                    flat_type_kind(self.module, element)?,
                )
            }
            naga::TypeInner::Struct { ref members, .. } => {
                let member = members.get(index as usize).ok_or_else(|| {
                    Error::UnsupportedFeature("struct index out of bounds".to_owned())
                })?;
                let mut offset = 0;
                for previous in &members[..index as usize] {
                    offset += flat_type_components(self.module, previous.ty)?;
                }
                (
                    offset,
                    flat_type_components(self.module, member.ty)?,
                    flat_type_kind(self.module, member.ty)?,
                )
            }
            ref inner => {
                return Err(Error::UnsupportedFeature(format!(
                    "indexed value type {inner:?}"
                )));
            }
        };
        let end = offset + count;
        let components = base
            .components
            .get(offset..end)
            .ok_or_else(|| Error::UnsupportedFeature("indexed value shape".to_owned()))?
            .to_vec();
        Ok(self.decode_aggregate_value(components, kind))
    }

    fn access(
        &mut self,
        base_handle: naga::Handle<naga::Expression>,
        index_handle: naga::Handle<naga::Expression>,
    ) -> Result<Value, Error> {
        let base = self.expression(base_handle)?;
        let index = self.expression(index_handle)?;
        if index.components.len() != 1
            || !matches!(index.kind, naga::ScalarKind::Uint | naga::ScalarKind::Sint)
        {
            return Err(Error::UnsupportedFeature(
                "dynamic index is not a scalar integer".to_owned(),
            ));
        }
        let (element_count, element_width) = match self
            .expression_type(base_handle)
            .map(|ty| self.module.types[ty].inner.clone())
        {
            Some(naga::TypeInner::Vector { size, .. }) => (vector_size(size), 1),
            Some(naga::TypeInner::Matrix { columns, rows, .. }) => {
                (vector_size(columns), vector_size(rows))
            }
            Some(naga::TypeInner::Array {
                base: element,
                size: naga::ArraySize::Constant(size),
                ..
            }) => (
                size.get() as usize,
                flat_type_components(self.module, element)?,
            ),
            Some(inner) => {
                return Err(Error::UnsupportedFeature(format!(
                    "dynamic indexed value type {inner:?}"
                )));
            }
            None => (base.components.len(), 1),
        };
        if element_count == 0 || base.components.len() != element_count * element_width {
            return Err(Error::UnsupportedFeature(
                "dynamic indexed value shape".to_owned(),
            ));
        }
        let mut result = Value {
            components: base.components[..element_width].to_vec(),
            kind: base.kind,
        };
        for candidate in 1..element_count {
            let candidate_u32 = u32::try_from(candidate).map_err(|_| {
                Error::UnsupportedFeature("dynamic index candidate exceeds u32".to_owned())
            })?;
            let candidate_index = Value {
                components: vec![match index.kind {
                    naga::ScalarKind::Uint => Src::from(candidate_u32),
                    naga::ScalarKind::Sint => Src::from(
                        i32::try_from(candidate)
                            .map_err(|_| {
                                Error::UnsupportedFeature(
                                    "dynamic index candidate exceeds i32".to_owned(),
                                )
                            })?
                            .cast_unsigned(),
                    ),
                    _ => unreachable!(),
                }],
                kind: index.kind,
            };
            let condition =
                self.binary(naga::BinaryOperator::Equal, &index, &candidate_index, None)?;
            let offset = candidate * element_width;
            let value = Value {
                components: base.components[offset..offset + element_width].to_vec(),
                kind: base.kind,
            };
            result = self.select(&condition, value, result)?;
        }
        Ok(result)
    }

    fn swizzle(
        &mut self,
        size: naga::VectorSize,
        handle: naga::Handle<naga::Expression>,
        pattern: [naga::SwizzleComponent; 4],
    ) -> Result<Value, Error> {
        let vector = self.expression(handle)?;
        let mut components = Vec::with_capacity(vector_size(size));
        for component in &pattern[..vector_size(size)] {
            let index = match component {
                naga::SwizzleComponent::X => 0,
                naga::SwizzleComponent::Y => 1,
                naga::SwizzleComponent::Z => 2,
                naga::SwizzleComponent::W => 3,
            };
            components.push(vector.components.get(index).cloned().ok_or_else(|| {
                Error::UnsupportedFeature(format!(
                    "swizzle component {component:?} out of bounds for expression {handle:?} with {} components",
                    vector.components.len()
                ))
            })?);
        }
        Ok(Value {
            components,
            kind: vector.kind,
        })
    }

    fn unary(
        &mut self,
        op: naga::UnaryOperator,
        handle: naga::Handle<naga::Expression>,
    ) -> Result<Value, Error> {
        let mut value = self.expression(handle)?;
        if op == naga::UnaryOperator::Negate && value.kind == naga::ScalarKind::Float {
            for component in &mut value.components {
                *component = component.clone().fneg();
            }
            return Ok(value);
        }
        if op == naga::UnaryOperator::Negate && value.kind == naga::ScalarKind::Sint {
            value.components = value
                .components
                .into_iter()
                .map(|component| self.emit_iadd_source(component.ineg(), Src::ZERO))
                .collect();
            return Ok(value);
        }
        if op == naga::UnaryOperator::LogicalNot && value.kind == naga::ScalarKind::Bool {
            for component in &mut value.components {
                *component = component.clone().bnot();
            }
            return Ok(value);
        }
        Err(Error::UnsupportedFeature(format!(
            "unary operator {op:?} for {:?}",
            value.kind
        )))
    }

    fn expression_matrix_shape(
        &self,
        handle: naga::Handle<naga::Expression>,
    ) -> Option<(usize, usize)> {
        if let naga::Expression::Math {
            fun: naga::MathFunction::Transpose,
            arg,
            ..
        } = self.source.expressions[handle]
        {
            let (columns, rows) = self.expression_matrix_shape(arg)?;
            return Some((rows, columns));
        }
        let ty = self.expression_type(handle)?;
        let naga::TypeInner::Matrix { columns, rows, .. } = self.module.types[ty].inner else {
            return None;
        };
        Some((vector_size(columns), vector_size(rows)))
    }

    #[allow(clippy::too_many_lines)]
    fn uniform_pointer(
        &mut self,
        handle: naga::Handle<naga::Expression>,
    ) -> Result<UniformPointer, Error> {
        match self.source.expressions[handle] {
            naga::Expression::GlobalVariable(global) => {
                let variable = &self.module.global_variables[global];
                if !matches!(
                    variable.space,
                    naga::AddressSpace::Uniform | naga::AddressSpace::Immediate
                ) {
                    return Err(Error::UnsupportedFeature(format!(
                        "load from {:?} address space",
                        variable.space
                    )));
                }
                Ok((global, variable.ty, 0, None))
            }
            naga::Expression::Access { base, index } => {
                let (global, ty, offset, dynamic_offset) = self.uniform_pointer(base)?;
                let (element, stride) = match self.module.types[ty].inner {
                    naga::TypeInner::Array {
                        base: element,
                        stride,
                        ..
                    } => (element, stride),
                    naga::TypeInner::Vector { scalar, .. } => (
                        self.scalar_type_handle(scalar).ok_or_else(|| {
                            Error::UnsupportedFeature(
                                "dynamic uniform vector scalar type is absent".to_owned(),
                            )
                        })?,
                        u32::from(scalar.width),
                    ),
                    naga::TypeInner::Matrix { rows, scalar, .. } => {
                        let row_bytes = u32::from(scalar.width)
                            * u32::try_from(vector_size(rows))
                                .expect("vectors have at most 4 rows");
                        let alignment = if rows == naga::VectorSize::Bi { 8 } else { 16 };
                        (
                            self.vector_type_handle(rows, scalar).ok_or_else(|| {
                                Error::UnsupportedFeature(
                                    "dynamic uniform matrix column type is absent".to_owned(),
                                )
                            })?,
                            row_bytes.div_ceil(alignment) * alignment,
                        )
                    }
                    ref inner => {
                        return Err(Error::UnsupportedFeature(format!(
                            "dynamic uniform access into {inner:?}"
                        )));
                    }
                };
                let index = self.expression(index)?;
                if index.components.len() != 1
                    || !matches!(index.kind, naga::ScalarKind::Sint | naga::ScalarKind::Uint)
                {
                    return Err(Error::UnsupportedFeature(
                        "uniform array index must be an integer scalar".to_owned(),
                    ));
                }
                let scaled = self.target.ssa_alloc.alloc(RegFile::GPR);
                self.emit(Instr::new(OpIMad {
                    dst: Dst::from(scaled),
                    srcs: [
                        index.components[0].clone(),
                        Src::from(stride),
                        dynamic_offset.unwrap_or(Src::ZERO),
                    ],
                    signed: false,
                }));
                Ok((global, element, offset, Some(Src::from(scaled))))
            }
            naga::Expression::AccessIndex { base, index } => {
                let (global, ty, offset, dynamic_offset) = self.uniform_pointer(base)?;
                let (element, field_offset) = match self.module.types[ty].inner {
                    naga::TypeInner::Struct { ref members, .. } => {
                        let member = members.get(index as usize).ok_or_else(|| {
                            Error::UnsupportedFeature(
                                "uniform member index out of bounds".to_owned(),
                            )
                        })?;
                        (member.ty, member.offset)
                    }
                    naga::TypeInner::Array {
                        base: element,
                        stride,
                        ..
                    } => (
                        element,
                        index.checked_mul(stride).ok_or_else(|| {
                            Error::UnsupportedFeature("uniform array offset overflow".to_owned())
                        })?,
                    ),
                    naga::TypeInner::Matrix {
                        columns,
                        rows,
                        scalar,
                    } => {
                        if index as usize >= vector_size(columns) {
                            return Err(Error::UnsupportedFeature(
                                "uniform matrix column index out of bounds".to_owned(),
                            ));
                        }
                        let row_bytes = u32::from(scalar.width)
                            * u32::try_from(vector_size(rows))
                                .expect("vectors have at most 4 rows");
                        let alignment = if rows == naga::VectorSize::Bi { 8 } else { 16 };
                        let stride = row_bytes.div_ceil(alignment) * alignment;
                        let element = self.vector_type_handle(rows, scalar).ok_or_else(|| {
                            Error::UnsupportedFeature(
                                "uniform matrix column type is absent".to_owned(),
                            )
                        })?;
                        (element, index * stride)
                    }
                    naga::TypeInner::Vector { size, scalar } => {
                        if index as usize >= vector_size(size) {
                            return Err(Error::UnsupportedFeature(
                                "uniform vector index out of bounds".to_owned(),
                            ));
                        }
                        let element = self.scalar_type_handle(scalar).ok_or_else(|| {
                            Error::UnsupportedFeature(
                                "uniform vector scalar type is absent".to_owned(),
                            )
                        })?;
                        (element, index * u32::from(scalar.width))
                    }
                    ref inner => {
                        return Err(Error::UnsupportedFeature(format!(
                            "uniform access into {inner:?}"
                        )));
                    }
                };
                let offset = offset.checked_add(field_offset).ok_or_else(|| {
                    Error::UnsupportedFeature("uniform offset overflow".to_owned())
                })?;
                Ok((global, element, offset, dynamic_offset))
            }
            ref pointer => Err(Error::UnsupportedFeature(format!(
                "load pointer {pointer:?}"
            ))),
        }
    }

    fn load_uniform(&mut self, pointer: naga::Handle<naga::Expression>) -> Result<Value, Error> {
        let (global, ty, base_offset, dynamic_offset) = self.uniform_pointer(pointer)?;
        let target = if self.module.global_variables[global].space == naga::AddressSpace::Immediate
        {
            // wgpu-hal binds immediate data at Deko uniform slot 15. Maxwell's
            // shader-visible constant-buffer index includes Deko's c0/c1
            // driver reservation, hence c17 here.
            15
        } else {
            *self.resources.uniforms.get(&global).ok_or_else(|| {
                Error::UnsupportedFeature("uniform has no allocated Deko slot".to_owned())
            })?
        };
        let (offsets, kind) = uniform_component_offsets(self.module, ty, base_offset)?;
        let mut components = Vec::with_capacity(offsets.len());
        for offset in offsets {
            let offset = u16::try_from(offset).map_err(|_| {
                Error::UnsupportedFeature(
                    "uniform offset exceeds Maxwell constant buffer".to_owned(),
                )
            })?;
            let dst = self.target.ssa_alloc.alloc(RegFile::GPR);
            self.emit(Instr::new(OpLdc {
                dst: Dst::from(dst),
                cb: Src::from(CBufRef {
                    // Deko reserves c0 and c1 for driver state. User uniform target zero is c2.
                    buf: CBuf::Binding(target + 2),
                    offset,
                }),
                offset: dynamic_offset.clone().unwrap_or(Src::ZERO),
                mode: LdcMode::Indexed,
                mem_type: MemType::B32,
            }));
            components.push(Src::from(dst));
        }
        Ok(Value { components, kind })
    }

    #[allow(clippy::too_many_lines)]
    fn storage_pointer(
        &mut self,
        handle: naga::Handle<naga::Expression>,
    ) -> Result<StoragePointer, Error> {
        match self.source.expressions[handle] {
            naga::Expression::GlobalVariable(global) => {
                let variable = &self.module.global_variables[global];
                let naga::AddressSpace::Storage { access } = variable.space else {
                    return Err(Error::UnsupportedFeature(format!(
                        "storage pointer rooted in {:?} address space",
                        variable.space
                    )));
                };
                Ok((global, variable.ty, access, 0, None))
            }
            naga::Expression::Access { base, index } => {
                let (global, ty, access, offset, dynamic_offset) = self.storage_pointer(base)?;
                let (element, stride) = match self.module.types[ty].inner {
                    naga::TypeInner::Array {
                        base: element,
                        stride,
                        ..
                    } => (element, stride),
                    naga::TypeInner::Vector { scalar, .. } => (
                        self.scalar_type_handle(scalar).ok_or_else(|| {
                            Error::UnsupportedFeature(
                                "dynamic storage vector scalar type is absent".to_owned(),
                            )
                        })?,
                        u32::from(scalar.width),
                    ),
                    naga::TypeInner::Matrix { rows, scalar, .. } => {
                        let row_bytes = u32::from(scalar.width)
                            * u32::try_from(vector_size(rows))
                                .expect("vectors have at most 4 rows");
                        let alignment = if rows == naga::VectorSize::Bi { 8 } else { 16 };
                        (
                            self.vector_type_handle(rows, scalar).ok_or_else(|| {
                                Error::UnsupportedFeature(
                                    "dynamic storage matrix column type is absent".to_owned(),
                                )
                            })?,
                            row_bytes.div_ceil(alignment) * alignment,
                        )
                    }
                    ref inner => {
                        return Err(Error::UnsupportedFeature(format!(
                            "dynamic storage access into {inner:?}"
                        )));
                    }
                };
                let index = self.expression(index)?;
                if index.components.len() != 1
                    || !matches!(index.kind, naga::ScalarKind::Sint | naga::ScalarKind::Uint)
                {
                    return Err(Error::UnsupportedFeature(
                        "storage array index must be an integer scalar".to_owned(),
                    ));
                }
                let scaled = self.target.ssa_alloc.alloc(RegFile::GPR);
                self.emit(Instr::new(OpIMad {
                    dst: Dst::from(scaled),
                    srcs: [
                        index.components[0].clone(),
                        Src::from(stride),
                        dynamic_offset.unwrap_or(Src::ZERO),
                    ],
                    signed: false,
                }));
                Ok((global, element, access, offset, Some(Src::from(scaled))))
            }
            naga::Expression::AccessIndex { base, index } => {
                let (global, ty, access, offset, dynamic_offset) = self.storage_pointer(base)?;
                let (element, field_offset) = match self.module.types[ty].inner {
                    naga::TypeInner::Struct { ref members, .. } => {
                        let member = members.get(index as usize).ok_or_else(|| {
                            Error::UnsupportedFeature(
                                "storage member index out of bounds".to_owned(),
                            )
                        })?;
                        (member.ty, member.offset)
                    }
                    naga::TypeInner::Array {
                        base: element,
                        stride,
                        size,
                    } => {
                        if let naga::ArraySize::Constant(count) = size
                            && index >= count.get()
                        {
                            return Err(Error::UnsupportedFeature(
                                "storage array index out of bounds".to_owned(),
                            ));
                        }
                        (
                            element,
                            index.checked_mul(stride).ok_or_else(|| {
                                Error::UnsupportedFeature(
                                    "storage array offset overflow".to_owned(),
                                )
                            })?,
                        )
                    }
                    naga::TypeInner::Matrix {
                        columns,
                        rows,
                        scalar,
                    } => {
                        if index as usize >= vector_size(columns) {
                            return Err(Error::UnsupportedFeature(
                                "storage matrix column index out of bounds".to_owned(),
                            ));
                        }
                        let row_bytes = u32::from(scalar.width)
                            * u32::try_from(vector_size(rows))
                                .expect("vectors have at most 4 rows");
                        let alignment = if rows == naga::VectorSize::Bi { 8 } else { 16 };
                        let stride = row_bytes.div_ceil(alignment) * alignment;
                        let element = self.vector_type_handle(rows, scalar).ok_or_else(|| {
                            Error::UnsupportedFeature(
                                "storage matrix column type is absent".to_owned(),
                            )
                        })?;
                        (element, index * stride)
                    }
                    naga::TypeInner::Vector { size, scalar } => {
                        if index as usize >= vector_size(size) {
                            return Err(Error::UnsupportedFeature(
                                "storage vector index out of bounds".to_owned(),
                            ));
                        }
                        let element = self.scalar_type_handle(scalar).ok_or_else(|| {
                            Error::UnsupportedFeature(
                                "storage vector scalar type is absent".to_owned(),
                            )
                        })?;
                        (element, index * u32::from(scalar.width))
                    }
                    ref inner => {
                        return Err(Error::UnsupportedFeature(format!(
                            "storage access into {inner:?}"
                        )));
                    }
                };
                let offset = offset.checked_add(field_offset).ok_or_else(|| {
                    Error::UnsupportedFeature("storage offset overflow".to_owned())
                })?;
                Ok((global, element, access, offset, dynamic_offset))
            }
            ref pointer => Err(Error::UnsupportedFeature(format!(
                "storage pointer {pointer:?}"
            ))),
        }
    }

    fn storage_address(
        &mut self,
        global: naga::Handle<naga::GlobalVariable>,
        dynamic_offset: Option<Src>,
    ) -> Result<SSARef, Error> {
        let target = *self.resources.storages.get(&global).ok_or_else(|| {
            Error::UnsupportedFeature("storage buffer has no allocated Deko slot".to_owned())
        })?;
        let descriptor = self
            .resources
            .storage_descriptor_base
            .checked_add(u16::from(target) * 0x10)
            .ok_or_else(|| Error::UnsupportedFeature("storage descriptor overflow".to_owned()))?;
        let base = self.target.ssa_alloc.alloc_vec(RegFile::GPR, 2);
        for (component, offset) in [descriptor, descriptor + 4].into_iter().enumerate() {
            self.emit(Instr::new(OpLdc {
                dst: Dst::from(base[component]),
                cb: Src::from(CBufRef {
                    buf: CBuf::Binding(0),
                    offset,
                }),
                offset: Src::ZERO,
                mode: LdcMode::Indexed,
                mem_type: MemType::B32,
            }));
        }
        let Some(dynamic_offset) = dynamic_offset else {
            return Ok(base);
        };
        let address = self.target.ssa_alloc.alloc_vec(RegFile::GPR, 2);
        let carry = self.target.ssa_alloc.alloc(RegFile::Carry);
        self.emit(Instr::new(OpIAdd2 {
            dst: Dst::from(address[0]),
            carry_out: Dst::from(carry),
            srcs: [Src::from(base[0]), dynamic_offset],
        }));
        self.emit(Instr::new(OpIAdd2X {
            dst: Dst::from(address[1]),
            carry_out: Dst::None,
            srcs: [Src::from(base[1]), Src::ZERO],
            carry_in: Src::from(carry),
        }));
        Ok(address)
    }

    fn storage_buffer_size(
        &mut self,
        global: naga::Handle<naga::GlobalVariable>,
    ) -> Result<Src, Error> {
        let target = *self.resources.storages.get(&global).ok_or_else(|| {
            Error::UnsupportedFeature("storage buffer has no allocated Deko slot".to_owned())
        })?;
        let offset = self
            .resources
            .storage_descriptor_base
            .checked_add(u16::from(target) * 0x10 + 8)
            .ok_or_else(|| Error::UnsupportedFeature("storage descriptor overflow".to_owned()))?;
        let dst = self.target.ssa_alloc.alloc(RegFile::GPR);
        self.emit(Instr::new(OpLdc {
            dst: Dst::from(dst),
            cb: Src::from(CBufRef {
                buf: CBuf::Binding(0),
                offset,
            }),
            offset: Src::ZERO,
            mode: LdcMode::Indexed,
            mem_type: MemType::B32,
        }));
        Ok(Src::from(dst))
    }

    fn storage_array_length(
        &mut self,
        pointer: naga::Handle<naga::Expression>,
    ) -> Result<Value, Error> {
        let (global, ty, _, base_offset, dynamic_offset) = self.storage_pointer(pointer)?;
        let naga::TypeInner::Array {
            size: naga::ArraySize::Dynamic,
            stride,
            ..
        } = self.module.types[ty].inner
        else {
            return Err(Error::UnsupportedFeature(
                "arrayLength pointer does not reference a runtime array".to_owned(),
            ));
        };
        if !stride.is_power_of_two() {
            return Err(Error::UnsupportedFeature(format!(
                "runtime array stride {stride} is not a power of two"
            )));
        }
        let mut bytes = self.storage_buffer_size(global)?;
        if base_offset != 0 {
            let dst = self.target.ssa_alloc.alloc(RegFile::GPR);
            self.emit(Instr::new(OpIAdd2 {
                dst: Dst::from(dst),
                carry_out: Dst::None,
                srcs: [bytes, Src::from(base_offset).ineg()],
            }));
            bytes = Src::from(dst);
        }
        if let Some(dynamic_offset) = dynamic_offset {
            let dst = self.target.ssa_alloc.alloc(RegFile::GPR);
            self.emit(Instr::new(OpIAdd2 {
                dst: Dst::from(dst),
                carry_out: Dst::None,
                srcs: [bytes, dynamic_offset.ineg()],
            }));
            bytes = Src::from(dst);
        }
        let dst = self.target.ssa_alloc.alloc(RegFile::GPR);
        self.emit(Instr::new(OpShr {
            dst: Dst::from(dst),
            src: bytes,
            shift: Src::from(stride.trailing_zeros()),
            wrap: true,
            signed: false,
        }));
        Ok(Value {
            components: vec![Src::from(dst)],
            kind: naga::ScalarKind::Uint,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn workgroup_pointer(
        &mut self,
        handle: naga::Handle<naga::Expression>,
    ) -> Result<WorkgroupPointer, Error> {
        match self.source.expressions[handle] {
            naga::Expression::GlobalVariable(global) => {
                let variable = &self.module.global_variables[global];
                if variable.space != naga::AddressSpace::WorkGroup {
                    return Err(Error::UnsupportedFeature(format!(
                        "workgroup pointer rooted in {:?} address space",
                        variable.space
                    )));
                }
                Ok((global, variable.ty, 0, None))
            }
            naga::Expression::Access { base, index } => {
                let (global, ty, offset, dynamic_offset) = self.workgroup_pointer(base)?;
                let (element, stride) = match self.module.types[ty].inner {
                    naga::TypeInner::Array {
                        base: element,
                        stride,
                        ..
                    } => (element, stride),
                    naga::TypeInner::Vector { scalar, .. } => {
                        (self.scalar_type(scalar)?, u32::from(scalar.width))
                    }
                    ref inner => {
                        return Err(Error::UnsupportedFeature(format!(
                            "dynamic workgroup access into {inner:?}"
                        )));
                    }
                };
                let index = self.expression(index)?;
                if index.components.len() != 1
                    || !matches!(index.kind, naga::ScalarKind::Sint | naga::ScalarKind::Uint)
                {
                    return Err(Error::UnsupportedFeature(
                        "workgroup array index must be an integer scalar".to_owned(),
                    ));
                }
                let scaled = self.target.ssa_alloc.alloc(RegFile::GPR);
                self.emit(Instr::new(OpIMad {
                    dst: Dst::from(scaled),
                    srcs: [
                        index.components[0].clone(),
                        Src::from(stride),
                        dynamic_offset.unwrap_or(Src::ZERO),
                    ],
                    signed: false,
                }));
                Ok((global, element, offset, Some(Src::from(scaled))))
            }
            naga::Expression::AccessIndex { base, index } => {
                let (global, ty, offset, dynamic_offset) = self.workgroup_pointer(base)?;
                let (element, field_offset) = match self.module.types[ty].inner {
                    naga::TypeInner::Struct { ref members, .. } => {
                        let member = members.get(index as usize).ok_or_else(|| {
                            Error::UnsupportedFeature(
                                "workgroup member index out of bounds".to_owned(),
                            )
                        })?;
                        (member.ty, member.offset)
                    }
                    naga::TypeInner::Array {
                        base: element,
                        stride,
                        size,
                    } => {
                        if let naga::ArraySize::Constant(count) = size
                            && index >= count.get()
                        {
                            return Err(Error::UnsupportedFeature(
                                "workgroup array index out of bounds".to_owned(),
                            ));
                        }
                        (
                            element,
                            index.checked_mul(stride).ok_or_else(|| {
                                Error::UnsupportedFeature(
                                    "workgroup array offset overflow".to_owned(),
                                )
                            })?,
                        )
                    }
                    naga::TypeInner::Vector { size, scalar } => {
                        if index as usize >= vector_size(size) {
                            return Err(Error::UnsupportedFeature(
                                "workgroup vector index out of bounds".to_owned(),
                            ));
                        }
                        (self.scalar_type(scalar)?, index * u32::from(scalar.width))
                    }
                    ref inner => {
                        return Err(Error::UnsupportedFeature(format!(
                            "workgroup access into {inner:?}"
                        )));
                    }
                };
                let offset = offset.checked_add(field_offset).ok_or_else(|| {
                    Error::UnsupportedFeature("workgroup offset overflow".to_owned())
                })?;
                Ok((global, element, offset, dynamic_offset))
            }
            ref pointer => Err(Error::UnsupportedFeature(format!(
                "workgroup pointer {pointer:?}"
            ))),
        }
    }

    fn load_workgroup(&mut self, pointer: naga::Handle<naga::Expression>) -> Result<Value, Error> {
        let (global, ty, base_offset, dynamic_offset) = self.workgroup_pointer(pointer)?;
        let global_offset = *self.resources.workgroups.get(&global).ok_or_else(|| {
            Error::UnsupportedFeature("workgroup global has no shared-memory offset".to_owned())
        })?;
        let base_offset = global_offset.checked_add(base_offset).ok_or_else(|| {
            Error::UnsupportedFeature("workgroup load offset overflow".to_owned())
        })?;
        let (offsets, kind) = uniform_component_offsets(self.module, ty, base_offset)?;
        let mut components = Vec::with_capacity(offsets.len());
        for offset in offsets {
            let offset = i32::try_from(offset).map_err(|_| {
                Error::UnsupportedFeature("workgroup load offset exceeds Maxwell range".to_owned())
            })?;
            let dst = self.target.ssa_alloc.alloc(RegFile::GPR);
            self.emit(Instr::new(OpLd {
                dst: Dst::from(dst),
                addr: dynamic_offset.clone().unwrap_or(Src::ZERO),
                uniform_addr: Src::ZERO,
                pred: Src::new_imm_bool(true),
                offset,
                stride: OffsetStride::X1,
                access: MemAccess {
                    mem_type: MemType::B32,
                    space: MemSpace::Shared,
                    order: MemOrder::Strong(MemScope::CTA),
                    eviction_priority: MemEvictionPriority::Normal,
                },
            }));
            components.push(Src::from(dst));
        }
        Ok(self.decode_aggregate_value(components, kind))
    }

    fn store_workgroup(
        &mut self,
        pointer: naga::Handle<naga::Expression>,
        value: &Value,
    ) -> Result<(), Error> {
        let (global, ty, base_offset, dynamic_offset) = self.workgroup_pointer(pointer)?;
        let global_offset = *self.resources.workgroups.get(&global).ok_or_else(|| {
            Error::UnsupportedFeature("workgroup global has no shared-memory offset".to_owned())
        })?;
        let base_offset = global_offset.checked_add(base_offset).ok_or_else(|| {
            Error::UnsupportedFeature("workgroup store offset overflow".to_owned())
        })?;
        let (offsets, _) = uniform_component_offsets(self.module, ty, base_offset)?;
        if value.components.len() != offsets.len() {
            return Err(Error::UnsupportedFeature(format!(
                "workgroup store shape mismatch: expected {}, got {}",
                offsets.len(),
                value.components.len()
            )));
        }
        let data = self.materialize_loop_components(value)?;
        for (offset, data) in offsets.into_iter().zip(data) {
            let offset = i32::try_from(offset).map_err(|_| {
                Error::UnsupportedFeature("workgroup store offset exceeds Maxwell range".to_owned())
            })?;
            self.emit(Instr::new(OpSt {
                addr: dynamic_offset.clone().unwrap_or(Src::ZERO),
                data: Src::from(data),
                uniform_addr: Src::ZERO,
                offset,
                stride: OffsetStride::X1,
                access: MemAccess {
                    mem_type: MemType::B32,
                    space: MemSpace::Shared,
                    order: MemOrder::Strong(MemScope::CTA),
                    eviction_priority: MemEvictionPriority::Normal,
                },
            }));
        }
        Ok(())
    }

    fn storage_access(mem_type: MemType) -> MemAccess {
        MemAccess {
            mem_type,
            space: MemSpace::Global(MemAddrType::A64),
            order: MemOrder::Strong(MemScope::GPU),
            eviction_priority: MemEvictionPriority::Normal,
        }
    }

    fn load_storage(&mut self, pointer: naga::Handle<naga::Expression>) -> Result<Value, Error> {
        let (global, ty, access, base_offset, dynamic_offset) = self.storage_pointer(pointer)?;
        if !access.contains(naga::StorageAccess::LOAD) {
            return Err(Error::UnsupportedFeature(
                "load from write-only storage buffer".to_owned(),
            ));
        }
        let address = self.storage_address(global, dynamic_offset)?;
        let (offsets, kind) = uniform_component_offsets(self.module, ty, base_offset)?;
        let mut components = Vec::with_capacity(offsets.len());
        for offset in offsets {
            let offset = i32::try_from(offset).map_err(|_| {
                Error::UnsupportedFeature("storage load offset exceeds Maxwell range".to_owned())
            })?;
            let dst = self.target.ssa_alloc.alloc(RegFile::GPR);
            self.emit(Instr::new(OpLd {
                dst: Dst::from(dst),
                addr: Src::from(address.clone()),
                uniform_addr: Src::ZERO,
                pred: Src::new_imm_bool(true),
                offset,
                stride: OffsetStride::X1,
                access: Self::storage_access(MemType::B32),
            }));
            components.push(Src::from(dst));
        }
        Ok(self.decode_aggregate_value(components, kind))
    }

    fn store_storage(
        &mut self,
        pointer: naga::Handle<naga::Expression>,
        value: &Value,
    ) -> Result<(), Error> {
        let (global, ty, access, base_offset, dynamic_offset) = self.storage_pointer(pointer)?;
        if !access.contains(naga::StorageAccess::STORE) {
            return Err(Error::UnsupportedFeature(
                "store to read-only storage buffer".to_owned(),
            ));
        }
        let (offsets, _) = uniform_component_offsets(self.module, ty, base_offset)?;
        if value.components.len() != offsets.len() {
            return Err(Error::UnsupportedFeature(format!(
                "storage store shape mismatch: expected {}, got {}",
                offsets.len(),
                value.components.len()
            )));
        }
        let data = self.materialize_loop_components(value)?;
        let address = self.storage_address(global, dynamic_offset)?;
        for (offset, data) in offsets.into_iter().zip(data) {
            let offset = i32::try_from(offset).map_err(|_| {
                Error::UnsupportedFeature("storage store offset exceeds Maxwell range".to_owned())
            })?;
            self.emit(Instr::new(OpSt {
                addr: Src::from(address.clone()),
                data: Src::from(data),
                uniform_addr: Src::ZERO,
                offset,
                stride: OffsetStride::X1,
                access: Self::storage_access(MemType::B32),
            }));
        }
        Ok(())
    }

    fn storage_atomic(
        &mut self,
        pointer: naga::Handle<naga::Expression>,
        fun: naga::AtomicFunction,
        value: naga::Handle<naga::Expression>,
        result: Option<naga::Handle<naga::Expression>>,
    ) -> Result<(), Error> {
        let (global, ty, access, base_offset, dynamic_offset) = self.storage_pointer(pointer)?;
        if !access.intersects(naga::StorageAccess::STORE | naga::StorageAccess::ATOMIC) {
            return Err(Error::UnsupportedFeature(
                "atomic operation on non-atomic storage binding".to_owned(),
            ));
        }
        let naga::TypeInner::Atomic(scalar) = self.module.types[ty].inner else {
            return Err(Error::UnsupportedFeature(format!(
                "atomic operation on {:?}",
                self.module.types[ty].inner
            )));
        };
        if scalar.width != 4
            || !matches!(scalar.kind, naga::ScalarKind::Sint | naga::ScalarKind::Uint)
        {
            return Err(Error::UnsupportedFeature(format!(
                "atomic scalar {scalar:?}"
            )));
        }
        let mut value = self.expression(value)?;
        if value.components.len() != 1 || value.kind != scalar.kind {
            return Err(Error::UnsupportedFeature(
                "atomic value type mismatch".to_owned(),
            ));
        }
        let mut compare = Src::ZERO;
        let atom_op = match fun {
            naga::AtomicFunction::Add => AtomOp::Add,
            naga::AtomicFunction::Subtract => {
                value.components[0] = value.components[0].clone().ineg();
                AtomOp::Add
            }
            naga::AtomicFunction::And => AtomOp::And,
            naga::AtomicFunction::ExclusiveOr => AtomOp::Xor,
            naga::AtomicFunction::InclusiveOr => AtomOp::Or,
            naga::AtomicFunction::Min => AtomOp::Min,
            naga::AtomicFunction::Max => AtomOp::Max,
            naga::AtomicFunction::Exchange { compare: None } => AtomOp::Exch,
            naga::AtomicFunction::Exchange {
                compare: Some(handle),
            } => {
                let comparison = self.expression(handle)?;
                if comparison.components.len() != 1 || comparison.kind != scalar.kind {
                    return Err(Error::UnsupportedFeature(
                        "atomic compare-exchange comparison type mismatch".to_owned(),
                    ));
                }
                compare = comparison.components[0].clone();
                AtomOp::CmpExch(AtomCmpSrc::Separate)
            }
        };
        let data = self.materialize_loop_components(&value)?;
        let address = self.storage_address(global, dynamic_offset)?;
        let destination = result.map(|_| self.target.ssa_alloc.alloc(RegFile::GPR));
        let offset = i32::try_from(base_offset).map_err(|_| {
            Error::UnsupportedFeature("storage atomic offset exceeds Maxwell range".to_owned())
        })?;
        self.emit(Instr::new(OpAtom {
            dst: Dst::from(destination),
            addr: Src::from(address),
            uniform_address: Src::ZERO,
            cmpr: compare,
            data: Src::from(data[0]),
            atom_op,
            atom_type: if scalar.kind == naga::ScalarKind::Uint {
                AtomType::U32
            } else {
                AtomType::I32
            },
            addr_offset: offset,
            addr_stride: OffsetStride::X1,
            mem_space: MemSpace::Global(MemAddrType::A64),
            mem_order: MemOrder::Strong(MemScope::GPU),
            mem_eviction_priority: MemEvictionPriority::Normal,
        }));
        if let (Some(handle), Some(destination)) = (result, destination) {
            self.values.insert(
                handle,
                Value {
                    components: vec![Src::from(destination)],
                    kind: scalar.kind,
                },
            );
        }
        Ok(())
    }

    fn workgroup_atomic(
        &mut self,
        pointer: naga::Handle<naga::Expression>,
        fun: naga::AtomicFunction,
        value: naga::Handle<naga::Expression>,
        result: Option<naga::Handle<naga::Expression>>,
    ) -> Result<(), Error> {
        let (global, ty, base_offset, dynamic_offset) = self.workgroup_pointer(pointer)?;
        let naga::TypeInner::Atomic(scalar) = self.module.types[ty].inner else {
            return Err(Error::UnsupportedFeature(format!(
                "workgroup atomic operation on {:?}",
                self.module.types[ty].inner
            )));
        };
        if scalar.width != 4
            || !matches!(scalar.kind, naga::ScalarKind::Sint | naga::ScalarKind::Uint)
        {
            return Err(Error::UnsupportedFeature(format!(
                "workgroup atomic scalar {scalar:?}"
            )));
        }
        let mut value = self.expression(value)?;
        if value.components.len() != 1 || value.kind != scalar.kind {
            return Err(Error::UnsupportedFeature(
                "workgroup atomic value type mismatch".to_owned(),
            ));
        }
        let mut compare = Src::ZERO;
        let atom_op = match fun {
            naga::AtomicFunction::Add => AtomOp::Add,
            naga::AtomicFunction::Subtract => {
                value.components[0] = value.components[0].clone().ineg();
                AtomOp::Add
            }
            naga::AtomicFunction::And => AtomOp::And,
            naga::AtomicFunction::ExclusiveOr => AtomOp::Xor,
            naga::AtomicFunction::InclusiveOr => AtomOp::Or,
            naga::AtomicFunction::Min => AtomOp::Min,
            naga::AtomicFunction::Max => AtomOp::Max,
            naga::AtomicFunction::Exchange { compare: None } => AtomOp::Exch,
            naga::AtomicFunction::Exchange {
                compare: Some(handle),
            } => {
                let comparison = self.expression(handle)?;
                if comparison.components.len() != 1 || comparison.kind != scalar.kind {
                    return Err(Error::UnsupportedFeature(
                        "workgroup atomic comparison type mismatch".to_owned(),
                    ));
                }
                compare = comparison.components[0].clone();
                AtomOp::CmpExch(AtomCmpSrc::Separate)
            }
        };
        let data = self.materialize_loop_components(&value)?;
        let global_offset = *self.resources.workgroups.get(&global).ok_or_else(|| {
            Error::UnsupportedFeature("workgroup global has no shared-memory offset".to_owned())
        })?;
        let offset = global_offset
            .checked_add(base_offset)
            .and_then(|offset| i32::try_from(offset).ok())
            .ok_or_else(|| {
                Error::UnsupportedFeature(
                    "workgroup atomic offset exceeds Maxwell range".to_owned(),
                )
            })?;
        let destination = result.map(|_| self.target.ssa_alloc.alloc(RegFile::GPR));
        self.emit(Instr::new(OpAtom {
            dst: Dst::from(destination),
            addr: dynamic_offset.unwrap_or(Src::ZERO),
            uniform_address: Src::ZERO,
            cmpr: compare,
            data: Src::from(data[0]),
            atom_op,
            atom_type: if scalar.kind == naga::ScalarKind::Uint {
                AtomType::U32
            } else {
                AtomType::I32
            },
            addr_offset: offset,
            addr_stride: OffsetStride::X1,
            mem_space: MemSpace::Shared,
            mem_order: MemOrder::Strong(MemScope::CTA),
            mem_eviction_priority: MemEvictionPriority::Normal,
        }));
        if let (Some(handle), Some(destination)) = (result, destination) {
            self.values.insert(
                handle,
                Value {
                    components: vec![Src::from(destination)],
                    kind: scalar.kind,
                },
            );
        }
        Ok(())
    }

    fn local_value(&mut self, handle: naga::Handle<naga::LocalVariable>) -> Result<Value, Error> {
        if let Some(value) = self.locals.get(&handle) {
            return Ok(value.clone());
        }
        let local = &self.source.local_variables[handle];
        let value = match local.init {
            Some(init) => self.expression(init)?,
            None => zero_value(self.module, local.ty)?,
        };
        self.locals.insert(handle, value.clone());
        Ok(value)
    }

    fn pointer_is_local(&self, handle: naga::Handle<naga::Expression>) -> bool {
        match self.source.expressions[handle] {
            naga::Expression::LocalVariable(_) => true,
            naga::Expression::Access { base, .. } | naga::Expression::AccessIndex { base, .. } => {
                self.pointer_is_local(base)
            }
            _ => false,
        }
    }

    fn pointer_is_storage(&self, handle: naga::Handle<naga::Expression>) -> bool {
        match self.source.expressions[handle] {
            naga::Expression::GlobalVariable(global) => matches!(
                self.module.global_variables[global].space,
                naga::AddressSpace::Storage { .. }
            ),
            naga::Expression::Access { base, .. } | naga::Expression::AccessIndex { base, .. } => {
                self.pointer_is_storage(base)
            }
            _ => false,
        }
    }

    fn pointer_is_workgroup(&self, handle: naga::Handle<naga::Expression>) -> bool {
        match self.source.expressions[handle] {
            naga::Expression::GlobalVariable(global) => {
                self.module.global_variables[global].space == naga::AddressSpace::WorkGroup
            }
            naga::Expression::Access { base, .. } | naga::Expression::AccessIndex { base, .. } => {
                self.pointer_is_workgroup(base)
            }
            _ => false,
        }
    }

    fn pointer_is_argument(&self, handle: naga::Handle<naga::Expression>) -> bool {
        match self.source.expressions[handle] {
            naga::Expression::FunctionArgument(index) => self
                .source
                .arguments
                .get(index as usize)
                .is_some_and(|argument| {
                    matches!(
                        self.module.types[argument.ty].inner,
                        naga::TypeInner::Pointer { .. }
                    )
                }),
            naga::Expression::Access { base, .. } | naga::Expression::AccessIndex { base, .. } => {
                self.pointer_is_argument(base)
            }
            _ => false,
        }
    }

    fn argument_pointer(
        &self,
        handle: naga::Handle<naga::Expression>,
    ) -> Result<(u32, naga::Handle<naga::Type>, usize), Error> {
        match self.source.expressions[handle] {
            naga::Expression::FunctionArgument(index) => {
                let argument = self.source.arguments.get(index as usize).ok_or_else(|| {
                    Error::UnsupportedFeature(format!("pointer argument {index}"))
                })?;
                let naga::TypeInner::Pointer { base, space } = self.module.types[argument.ty].inner
                else {
                    return Err(Error::UnsupportedFeature(format!(
                        "non-pointer argument type {:?}",
                        self.module.types[argument.ty].inner
                    )));
                };
                if space != naga::AddressSpace::Function {
                    return Err(Error::UnsupportedFeature(format!(
                        "pointer argument in {space:?} address space"
                    )));
                }
                Ok((index, base, 0))
            }
            naga::Expression::Access { base, .. } => {
                let (argument, ty, component_offset) = self.argument_pointer(base)?;
                let naga::TypeInner::Array {
                    base: element,
                    size: naga::ArraySize::Constant(size),
                    ..
                } = self.module.types[ty].inner
                else {
                    return Err(Error::UnsupportedFeature(
                        "dynamic pointer-argument access into a non-array".to_owned(),
                    ));
                };
                if size.get() != 1 {
                    return Err(Error::UnsupportedFeature(
                        "dynamic pointer-argument arrays larger than one element".to_owned(),
                    ));
                }
                Ok((argument, element, component_offset))
            }
            naga::Expression::AccessIndex { base, index } => {
                let (argument, ty, component_offset) = self.argument_pointer(base)?;
                let (element, preceding) = match self.module.types[ty].inner {
                    naga::TypeInner::Struct { ref members, .. } => {
                        let member = members.get(index as usize).ok_or_else(|| {
                            Error::UnsupportedFeature(
                                "pointer-argument member index out of bounds".to_owned(),
                            )
                        })?;
                        let preceding = members[..index as usize]
                            .iter()
                            .map(|member| flat_type_components(self.module, member.ty))
                            .try_fold(0_usize, |sum, count| {
                                sum.checked_add(count?).ok_or_else(|| {
                                    Error::UnsupportedFeature(
                                        "pointer-argument component offset overflow".to_owned(),
                                    )
                                })
                            })?;
                        (member.ty, preceding)
                    }
                    naga::TypeInner::Array { base: element, .. } => {
                        let count = flat_type_components(self.module, element)?;
                        (element, index as usize * count)
                    }
                    naga::TypeInner::Vector { scalar, .. } => {
                        (self.scalar_type(scalar)?, index as usize)
                    }
                    naga::TypeInner::Matrix { rows, scalar, .. } => {
                        let rows = vector_size(rows);
                        (
                            self.vector_type_handle(
                                match rows {
                                    2 => naga::VectorSize::Bi,
                                    3 => naga::VectorSize::Tri,
                                    4 => naga::VectorSize::Quad,
                                    _ => unreachable!(),
                                },
                                scalar,
                            )
                            .ok_or_else(|| {
                                Error::UnsupportedFeature(
                                    "pointer-argument matrix column type is absent".to_owned(),
                                )
                            })?,
                            index as usize * rows,
                        )
                    }
                    ref inner => {
                        return Err(Error::UnsupportedFeature(format!(
                            "pointer-argument access into {inner:?}"
                        )));
                    }
                };
                Ok((argument, element, component_offset + preceding))
            }
            ref pointer => Err(Error::UnsupportedFeature(format!(
                "pointer argument expression {pointer:?}"
            ))),
        }
    }

    fn load_argument_pointer(
        &mut self,
        pointer: naga::Handle<naga::Expression>,
    ) -> Result<Value, Error> {
        let (argument, ty, offset) = self.argument_pointer(pointer)?;
        let count = flat_type_components(self.module, ty)?;
        let value = self.arguments.get(argument as usize).ok_or_else(|| {
            Error::UnsupportedFeature(format!("missing pointer argument {argument}"))
        })?;
        let end = offset.checked_add(count).ok_or_else(|| {
            Error::UnsupportedFeature("pointer-argument range overflow".to_owned())
        })?;
        let components = value
            .components
            .get(offset..end)
            .ok_or_else(|| Error::UnsupportedFeature("pointer-argument value shape".to_owned()))?
            .to_vec();
        Ok(self.decode_aggregate_value(components, flat_type_kind(self.module, ty)?))
    }

    fn store_argument_pointer(
        &mut self,
        pointer: naga::Handle<naga::Expression>,
        mut value: Value,
    ) -> Result<(), Error> {
        let (argument, ty, offset) = self.argument_pointer(pointer)?;
        let expected = flat_type_components(self.module, ty)?;
        if value.components.len() != expected {
            return Err(Error::UnsupportedFeature(format!(
                "pointer-argument store shape mismatch: expected {expected}, got {}",
                value.components.len()
            )));
        }
        let argument_kind = self
            .arguments
            .get(argument as usize)
            .ok_or_else(|| {
                Error::UnsupportedFeature(format!("missing pointer argument {argument}"))
            })?
            .kind;
        if value.kind == naga::ScalarKind::Bool && argument_kind != naga::ScalarKind::Bool {
            value.components = self
                .materialize_loop_components(&value)?
                .into_iter()
                .map(Src::from)
                .collect();
        }
        let argument_value = self.arguments.get_mut(argument as usize).ok_or_else(|| {
            Error::UnsupportedFeature(format!("missing pointer argument {argument}"))
        })?;
        let end = offset.checked_add(expected).ok_or_else(|| {
            Error::UnsupportedFeature("pointer-argument store range overflow".to_owned())
        })?;
        let destination = argument_value
            .components
            .get_mut(offset..end)
            .ok_or_else(|| {
                Error::UnsupportedFeature("pointer-argument store value shape".to_owned())
            })?;
        destination.clone_from_slice(&value.components);
        Ok(())
    }

    fn local_pointer(
        &self,
        handle: naga::Handle<naga::Expression>,
    ) -> Result<
        (
            naga::Handle<naga::LocalVariable>,
            naga::Handle<naga::Type>,
            usize,
        ),
        Error,
    > {
        match self.source.expressions[handle] {
            naga::Expression::LocalVariable(local) => {
                Ok((local, self.source.local_variables[local].ty, 0))
            }
            naga::Expression::AccessIndex { base, index } => {
                let (local, ty, component_offset) = self.local_pointer(base)?;
                match self.module.types[ty].inner {
                    naga::TypeInner::Struct { ref members, .. } => {
                        let member = members.get(index as usize).ok_or_else(|| {
                            Error::UnsupportedFeature("local member index out of bounds".to_owned())
                        })?;
                        let mut preceding = 0_usize;
                        for previous in &members[..index as usize] {
                            preceding = preceding
                                .checked_add(flat_type_components(self.module, previous.ty)?)
                                .ok_or_else(|| {
                                    Error::UnsupportedFeature(
                                        "local component offset overflow".to_owned(),
                                    )
                                })?;
                        }
                        Ok((local, member.ty, component_offset + preceding))
                    }
                    naga::TypeInner::Vector { size, scalar } => {
                        if index as usize >= vector_size(size) {
                            return Err(Error::UnsupportedFeature(
                                "local vector index out of bounds".to_owned(),
                            ));
                        }
                        let scalar_ty = self.scalar_type(scalar)?;
                        Ok((local, scalar_ty, component_offset + index as usize))
                    }
                    naga::TypeInner::Array {
                        base: element,
                        size,
                        ..
                    } => {
                        if let naga::ArraySize::Constant(count) = size {
                            if index >= count.get() {
                                return Err(Error::UnsupportedFeature(
                                    "local array index out of bounds".to_owned(),
                                ));
                            }
                        }
                        let element_components = flat_type_components(self.module, element)?;
                        let preceding = (index as usize)
                            .checked_mul(element_components)
                            .ok_or_else(|| {
                                Error::UnsupportedFeature(
                                    "local array component offset overflow".to_owned(),
                                )
                            })?;
                        Ok((local, element, component_offset + preceding))
                    }
                    ref inner => Err(Error::UnsupportedFeature(format!(
                        "local access into {inner:?}"
                    ))),
                }
            }
            ref pointer => Err(Error::UnsupportedFeature(format!(
                "local pointer {pointer:?}"
            ))),
        }
    }

    fn scalar_type(&self, scalar: naga::Scalar) -> Result<naga::Handle<naga::Type>, Error> {
        self.module
            .types
            .iter()
            .find_map(|(handle, ty)| {
                matches!(ty.inner, naga::TypeInner::Scalar(candidate) if candidate == scalar)
                    .then_some(handle)
            })
            .ok_or_else(|| Error::UnsupportedFeature(format!("missing scalar type {scalar:?}")))
    }

    fn load_local_pointer(
        &mut self,
        pointer: naga::Handle<naga::Expression>,
    ) -> Result<Value, Error> {
        if let Some((local, ty, offset, element_count, index)) =
            self.dynamic_local_pointer(pointer)?
        {
            let element_width = flat_type_components(self.module, ty)?;
            let value = self.local_value(local)?;
            let mut result = Value {
                components: value
                    .components
                    .get(offset..offset + element_width)
                    .ok_or_else(|| {
                        Error::UnsupportedFeature("dynamic local value shape mismatch".to_owned())
                    })?
                    .to_vec(),
                kind: value.kind,
            };
            for candidate in 1..element_count {
                let condition = self.dynamic_index_condition(&index, candidate)?;
                let start = offset + candidate * element_width;
                let candidate_value = Value {
                    components: value
                        .components
                        .get(start..start + element_width)
                        .ok_or_else(|| {
                            Error::UnsupportedFeature(
                                "dynamic local value shape mismatch".to_owned(),
                            )
                        })?
                        .to_vec(),
                    kind: value.kind,
                };
                result = self.select(&condition, candidate_value, result)?;
            }
            return Ok(
                self.decode_aggregate_value(result.components, flat_type_kind(self.module, ty)?)
            );
        }
        let (local, ty, offset) = self.local_pointer(pointer)?;
        let count = flat_type_components(self.module, ty)?;
        let value = self.local_value(local)?;
        let end = offset.checked_add(count).ok_or_else(|| {
            Error::UnsupportedFeature("local component range overflow".to_owned())
        })?;
        let components = value
            .components
            .get(offset..end)
            .ok_or_else(|| Error::UnsupportedFeature("local value shape mismatch".to_owned()))?
            .to_vec();
        Ok(self.decode_aggregate_value(components, flat_type_kind(self.module, ty)?))
    }

    fn dynamic_local_pointer(
        &mut self,
        pointer: naga::Handle<naga::Expression>,
    ) -> Result<Option<DynamicLocalPointer>, Error> {
        let naga::Expression::Access { base, index } = self.source.expressions[pointer] else {
            return Ok(None);
        };
        let (local, container, offset) = self.local_pointer(base)?;
        let (element, element_count) = match self.module.types[container].inner {
            naga::TypeInner::Vector { size, scalar } => {
                (self.scalar_type(scalar)?, vector_size(size))
            }
            naga::TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } => {
                let element = self.vector_type_handle(rows, scalar).ok_or_else(|| {
                    Error::UnsupportedFeature(
                        "dynamic local matrix column type is absent".to_owned(),
                    )
                })?;
                (element, vector_size(columns))
            }
            naga::TypeInner::Array {
                base: element,
                size: naga::ArraySize::Constant(size),
                ..
            } => (element, size.get() as usize),
            ref inner => {
                return Err(Error::UnsupportedFeature(format!(
                    "dynamic local access into {inner:?}"
                )));
            }
        };
        let index = self.expression(index)?;
        if index.components.len() != 1
            || !matches!(index.kind, naga::ScalarKind::Uint | naga::ScalarKind::Sint)
        {
            return Err(Error::UnsupportedFeature(
                "dynamic local index is not a scalar integer".to_owned(),
            ));
        }
        if element_count == 0 {
            return Err(Error::UnsupportedFeature(
                "dynamic local access into an empty container".to_owned(),
            ));
        }
        Ok(Some((local, element, offset, element_count, index)))
    }

    fn dynamic_index_condition(&mut self, index: &Value, candidate: usize) -> Result<Value, Error> {
        let candidate = match index.kind {
            naga::ScalarKind::Uint => u32::try_from(candidate).map_err(|_| {
                Error::UnsupportedFeature("dynamic index candidate exceeds u32".to_owned())
            })?,
            naga::ScalarKind::Sint => i32::try_from(candidate)
                .map_err(|_| {
                    Error::UnsupportedFeature("dynamic index candidate exceeds i32".to_owned())
                })?
                .cast_unsigned(),
            _ => unreachable!(),
        };
        self.binary(
            naga::BinaryOperator::Equal,
            index,
            &Value {
                components: vec![Src::from(candidate)],
                kind: index.kind,
            },
            None,
        )
    }

    fn store_local_pointer(
        &mut self,
        pointer: naga::Handle<naga::Expression>,
        mut value: Value,
    ) -> Result<(), Error> {
        if let Some((local, ty, offset, element_count, index)) =
            self.dynamic_local_pointer(pointer)?
        {
            let element_width = flat_type_components(self.module, ty)?;
            if value.components.len() != element_width {
                return Err(Error::UnsupportedFeature(format!(
                    "dynamic local store shape mismatch: expected {element_width}, got {}",
                    value.components.len()
                )));
            }
            let mut local_value = self.local_value(local)?;
            if value.kind == naga::ScalarKind::Bool && local_value.kind != naga::ScalarKind::Bool {
                value.components = self
                    .materialize_loop_components(&value)?
                    .into_iter()
                    .map(Src::from)
                    .collect();
                value.kind = local_value.kind;
            }
            for candidate in 0..element_count {
                let condition = self.dynamic_index_condition(&index, candidate)?;
                let start = offset + candidate * element_width;
                let old = Value {
                    components: local_value.components[start..start + element_width].to_vec(),
                    kind: local_value.kind,
                };
                let selected = self.select(&condition, value.clone(), old)?;
                local_value.components[start..start + element_width]
                    .clone_from_slice(&selected.components);
            }
            self.locals.insert(local, local_value);
            return Ok(());
        }

        let (local, ty, offset) = self.local_pointer(pointer)?;
        let expected = flat_type_components(self.module, ty)?;
        if value.components.len() != expected {
            return Err(Error::UnsupportedFeature(format!(
                "local store shape mismatch: expected {expected}, got {} for pointer {:?}",
                value.components.len(),
                self.source.expressions[pointer]
            )));
        }
        let mut local_value = self.local_value(local)?;
        if value.kind == naga::ScalarKind::Bool && local_value.kind != naga::ScalarKind::Bool {
            value.components = self
                .materialize_loop_components(&value)?
                .into_iter()
                .map(Src::from)
                .collect();
        }
        let end = offset + expected;
        local_value.components[offset..end].clone_from_slice(&value.components);
        self.locals.insert(local, local_value);
        Ok(())
    }

    fn decode_aggregate_value(&mut self, components: Vec<Src>, kind: naga::ScalarKind) -> Value {
        if kind != naga::ScalarKind::Bool {
            return Value { components, kind };
        }
        let components = components
            .into_iter()
            .map(|source| {
                if Self::predicate(&source).is_ok() {
                    return source;
                }
                let destination = self.target.ssa_alloc.alloc(RegFile::Pred);
                self.emit(Instr::new(OpISetP {
                    dst: Dst::from(destination),
                    set_op: PredSetOp::And,
                    cmp_op: IntCmpOp::Ne,
                    cmp_type: IntCmpType::U32,
                    ex: false,
                    srcs: [source, Src::ZERO],
                    accum: true.into(),
                    low_cmp: true.into(),
                }));
                Src::from(destination)
            })
            .collect();
        Value { components, kind }
    }

    fn binary(
        &mut self,
        op: naga::BinaryOperator,
        left: &Value,
        right: &Value,
        left_matrix: Option<(usize, usize)>,
    ) -> Result<Value, Error> {
        self.binary_with_shapes(op, left, right, left_matrix, None)
    }

    fn binary_with_shapes(
        &mut self,
        op: naga::BinaryOperator,
        left: &Value,
        right: &Value,
        left_matrix: Option<(usize, usize)>,
        right_matrix: Option<(usize, usize)>,
    ) -> Result<Value, Error> {
        let mixed_integer_shift =
            matches!(
                op,
                naga::BinaryOperator::ShiftLeft | naga::BinaryOperator::ShiftRight
            ) && matches!(left.kind, naga::ScalarKind::Sint | naga::ScalarKind::Uint)
                && matches!(right.kind, naga::ScalarKind::Sint | naga::ScalarKind::Uint);
        if left.kind != right.kind && !mixed_integer_shift {
            return Err(Error::UnsupportedFeature(format!(
                "{op:?} for {:?} and {:?}",
                left.kind, right.kind
            )));
        }
        if op == naga::BinaryOperator::Multiply {
            match (left_matrix, right_matrix) {
                (Some((columns, rows)), None) if right.components.len() == columns => {
                    return self.multiply_matrix_vector(left, right, columns, rows);
                }
                (None, Some((columns, rows))) if left.components.len() == rows => {
                    return self.multiply_vector_matrix(left, right, columns, rows);
                }
                (Some((left_columns, left_rows)), Some((right_columns, right_rows)))
                    if left_columns == right_rows =>
                {
                    return self.multiply_matrices(
                        left,
                        right,
                        left_columns,
                        left_rows,
                        right_columns,
                    );
                }
                _ => {}
            }
        }
        if op == naga::BinaryOperator::Modulo && left.kind == naga::ScalarKind::Float {
            let quotient = self.binary(naga::BinaryOperator::Divide, left, right, left_matrix)?;
            let quotient = self.float_round(quotient, FRndMode::Zero)?;
            let product = self.binary(naga::BinaryOperator::Multiply, &quotient, right, None)?;
            return self.binary(naga::BinaryOperator::Subtract, left, &product, left_matrix);
        }
        let width = left.components.len().max(right.components.len());
        if (left.components.len() != 1 && left.components.len() != width)
            || (right.components.len() != 1 && right.components.len() != width)
        {
            return Err(Error::UnsupportedFeature(format!(
                "{op:?} operands with incompatible widths {} and {} (left matrix {left_matrix:?})",
                left.components.len(),
                right.components.len()
            )));
        }
        let mut components = Vec::with_capacity(width);
        for index in 0..width {
            let lhs = left.components[if left.components.len() == 1 { 0 } else { index }].clone();
            let rhs = right.components[if right.components.len() == 1 {
                0
            } else {
                index
            }]
            .clone();
            let (instruction, component) = self.binary_component(op, left.kind, lhs, rhs)?;
            self.emit(instruction);
            components.push(component);
        }
        let kind = if matches!(
            op,
            naga::BinaryOperator::Equal
                | naga::BinaryOperator::NotEqual
                | naga::BinaryOperator::Less
                | naga::BinaryOperator::LessEqual
                | naga::BinaryOperator::Greater
                | naga::BinaryOperator::GreaterEqual
        ) {
            naga::ScalarKind::Bool
        } else {
            left.kind
        };
        Ok(Value { components, kind })
    }

    #[allow(clippy::too_many_lines)]
    fn binary_component(
        &mut self,
        op: naga::BinaryOperator,
        kind: naga::ScalarKind,
        lhs: Src,
        rhs: Src,
    ) -> Result<(Instr, Src), Error> {
        let dst = self.target.ssa_alloc.alloc(RegFile::GPR);
        let unsigned_modulo_mask = rhs
            .is_unmodified()
            .then(|| rhs.src_ref.as_u32())
            .flatten()
            .filter(|divisor| divisor.is_power_of_two())
            .map(|divisor| divisor - 1);
        let instruction = match (kind, op) {
            (naga::ScalarKind::Float, naga::BinaryOperator::Add) => Instr::new(OpFAdd {
                dst: Dst::from(dst),
                srcs: [lhs, rhs],
                saturate: false,
                rnd_mode: FRndMode::NearestEven,
                ftz: false,
            }),
            (naga::ScalarKind::Float, naga::BinaryOperator::Subtract) => Instr::new(OpFAdd {
                dst: Dst::from(dst),
                srcs: [lhs, rhs.fneg()],
                saturate: false,
                rnd_mode: FRndMode::NearestEven,
                ftz: false,
            }),
            (naga::ScalarKind::Float, naga::BinaryOperator::Multiply) => Instr::new(OpFMul {
                dst: Dst::from(dst),
                srcs: [lhs, rhs],
                saturate: false,
                rnd_mode: FRndMode::NearestEven,
                ftz: false,
                dnz: false,
            }),
            (naga::ScalarKind::Float, naga::BinaryOperator::Divide) => {
                let reciprocal = self.target.ssa_alloc.alloc(RegFile::GPR);
                self.emit(Instr::new(OpMuFu {
                    dst: Dst::from(reciprocal),
                    op: MuFuOp::Rcp,
                    src: rhs,
                    op_type: FloatType::F32,
                }));
                Instr::new(OpFMul {
                    dst: Dst::from(dst),
                    srcs: [lhs, Src::from(reciprocal)],
                    saturate: false,
                    rnd_mode: FRndMode::NearestEven,
                    ftz: false,
                    dnz: false,
                })
            }
            (
                naga::ScalarKind::Float,
                naga::BinaryOperator::Equal
                | naga::BinaryOperator::NotEqual
                | naga::BinaryOperator::Less
                | naga::BinaryOperator::LessEqual
                | naga::BinaryOperator::Greater
                | naga::BinaryOperator::GreaterEqual,
            ) => return self.float_comparison(op, lhs, rhs),
            (
                naga::ScalarKind::Uint | naga::ScalarKind::Sint,
                naga::BinaryOperator::Add | naga::BinaryOperator::Subtract,
            ) => Instr::new(OpIAdd2 {
                dst: Dst::from(dst),
                carry_out: Dst::None,
                srcs: [
                    lhs,
                    if op == naga::BinaryOperator::Subtract {
                        rhs.ineg()
                    } else {
                        rhs
                    },
                ],
            }),
            (naga::ScalarKind::Uint | naga::ScalarKind::Sint, naga::BinaryOperator::Multiply) => {
                Instr::new(OpIMul {
                    dst: Dst::from(dst),
                    srcs: [lhs, rhs],
                    signed: [kind == naga::ScalarKind::Sint; 2],
                    high: false,
                })
            }
            (naga::ScalarKind::Uint, naga::BinaryOperator::Modulo)
                if unsigned_modulo_mask.is_some() =>
            {
                Instr::new(OpLop2 {
                    dst: Dst::from(dst),
                    srcs: [lhs, Src::from(unsigned_modulo_mask.expect("guarded above"))],
                    op: LogicOp2::And,
                })
            }
            (
                naga::ScalarKind::Uint | naga::ScalarKind::Sint,
                naga::BinaryOperator::Divide | naga::BinaryOperator::Modulo,
            ) => return self.integer_div_mod(kind, op, lhs, rhs),
            (naga::ScalarKind::Uint | naga::ScalarKind::Sint, naga::BinaryOperator::ShiftLeft) => {
                Instr::new(OpShl {
                    dst: Dst::from(dst),
                    src: lhs,
                    shift: rhs,
                    wrap: true,
                })
            }
            (naga::ScalarKind::Uint | naga::ScalarKind::Sint, naga::BinaryOperator::ShiftRight) => {
                Instr::new(OpShr {
                    dst: Dst::from(dst),
                    src: lhs,
                    shift: rhs,
                    wrap: true,
                    signed: kind == naga::ScalarKind::Sint,
                })
            }
            (
                naga::ScalarKind::Uint | naga::ScalarKind::Sint,
                naga::BinaryOperator::And
                | naga::BinaryOperator::InclusiveOr
                | naga::BinaryOperator::ExclusiveOr,
            ) => {
                let logic = match op {
                    naga::BinaryOperator::And => LogicOp2::And,
                    naga::BinaryOperator::InclusiveOr => LogicOp2::Or,
                    naga::BinaryOperator::ExclusiveOr => LogicOp2::Xor,
                    _ => unreachable!(),
                };
                Instr::new(OpLop2 {
                    dst: Dst::from(dst),
                    srcs: [lhs, rhs],
                    op: logic,
                })
            }
            (
                naga::ScalarKind::Uint | naga::ScalarKind::Sint,
                naga::BinaryOperator::Equal
                | naga::BinaryOperator::NotEqual
                | naga::BinaryOperator::Less
                | naga::BinaryOperator::LessEqual
                | naga::BinaryOperator::Greater
                | naga::BinaryOperator::GreaterEqual,
            ) => return self.integer_comparison(op, kind, lhs, rhs),
            _ => return Err(Error::UnsupportedFeature(format!("binary operator {op:?}"))),
        };
        Ok((instruction, Src::from(dst)))
    }

    fn float_comparison(
        &mut self,
        op: naga::BinaryOperator,
        lhs: Src,
        rhs: Src,
    ) -> Result<(Instr, Src), Error> {
        let dst = self.target.ssa_alloc.alloc(RegFile::Pred);
        let cmp_op = match op {
            naga::BinaryOperator::Equal => FloatCmpOp::OrdEq,
            naga::BinaryOperator::NotEqual => FloatCmpOp::UnordNe,
            naga::BinaryOperator::Less => FloatCmpOp::OrdLt,
            naga::BinaryOperator::LessEqual => FloatCmpOp::OrdLe,
            naga::BinaryOperator::Greater => FloatCmpOp::OrdGt,
            naga::BinaryOperator::GreaterEqual => FloatCmpOp::OrdGe,
            _ => {
                return Err(Error::UnsupportedFeature(format!(
                    "float comparison {op:?}"
                )));
            }
        };
        Ok((
            Instr::new(OpFSetP {
                dst: Dst::from(dst),
                set_op: PredSetOp::And,
                cmp_op,
                srcs: [lhs, rhs],
                accum: true.into(),
                ftz: false,
            }),
            Src::from(dst),
        ))
    }

    #[allow(clippy::needless_pass_by_value, clippy::too_many_lines)]
    fn integer_div_mod(
        &mut self,
        kind: naga::ScalarKind,
        op: naga::BinaryOperator,
        lhs: Src,
        rhs: Src,
    ) -> Result<(Instr, Src), Error> {
        let signed = kind == naga::ScalarKind::Sint;
        let zero_divisor = self.emit_integer_comparison_source(
            naga::BinaryOperator::Equal,
            naga::ScalarKind::Uint,
            rhs.clone(),
            Src::ZERO,
        )?;
        let exceptional = if signed {
            let min_numerator = self.emit_integer_comparison_source(
                naga::BinaryOperator::Equal,
                naga::ScalarKind::Uint,
                lhs.clone(),
                Src::from(i32::MIN.cast_unsigned()),
            )?;
            let negative_one_divisor = self.emit_integer_comparison_source(
                naga::BinaryOperator::Equal,
                naga::ScalarKind::Uint,
                rhs.clone(),
                Src::from((-1_i32).cast_unsigned()),
            )?;
            let overflow =
                self.emit_predicate_binary(PredSetOp::And, min_numerator, negative_one_divisor);
            self.emit_predicate_binary(PredSetOp::Or, zero_divisor, overflow)
        } else {
            zero_divisor
        };
        let safe_divisor =
            self.emit_select_source(exceptional.clone(), Src::from(1_u32), rhs.clone());

        let (unsigned_lhs, unsigned_rhs, lhs_negative, quotient_negative) = if signed {
            let lhs_sign = self.emit_shift_right(lhs.clone(), Src::from(31_u32), true);
            let rhs_sign = self.emit_shift_right(safe_divisor.clone(), Src::from(31_u32), true);
            let unsigned_lhs = self.emit_abs_source(lhs.clone(), lhs_sign.clone());
            let unsigned_rhs = self.emit_abs_source(safe_divisor.clone(), rhs_sign.clone());
            let lhs_negative = self.emit_integer_comparison_source(
                naga::BinaryOperator::Less,
                naga::ScalarKind::Sint,
                lhs.clone(),
                Src::ZERO,
            )?;
            let signs = self.emit_logic_source(LogicOp2::Xor, lhs.clone(), safe_divisor);
            let quotient_negative = self.emit_integer_comparison_source(
                naga::BinaryOperator::Less,
                naga::ScalarKind::Sint,
                signs,
                Src::ZERO,
            )?;
            (
                unsigned_lhs,
                unsigned_rhs,
                Some(lhs_negative),
                Some(quotient_negative),
            )
        } else {
            (lhs.clone(), safe_divisor, None, None)
        };

        let denominator_float = self.target.ssa_alloc.alloc(RegFile::GPR);
        let rf = self.target.ssa_alloc.alloc(RegFile::GPR);
        self.emit(Instr::new(OpI2F {
            dst: Dst::from(denominator_float),
            src: unsigned_rhs.clone(),
            dst_type: FloatType::F32,
            src_type: IntType::U32,
            rnd_mode: FRndMode::NearestEven,
        }));
        self.emit(Instr::new(OpMuFu {
            dst: Dst::from(rf),
            op: MuFuOp::Rcp,
            src: Src::from(denominator_float),
            op_type: FloatType::F32,
        }));
        let scaled_reciprocal = self.target.ssa_alloc.alloc(RegFile::GPR);
        self.emit(Instr::new(OpFMul {
            dst: Dst::from(scaled_reciprocal),
            srcs: [Src::from(rf), Src::from(4_294_966_784.0_f32)],
            saturate: false,
            rnd_mode: FRndMode::NearestEven,
            ftz: false,
            dnz: false,
        }));
        let reciprocal = self.target.ssa_alloc.alloc(RegFile::GPR);
        self.emit(Instr::new(OpF2I {
            dst: Dst::from(reciprocal),
            src: Src::from(scaled_reciprocal),
            src_type: FloatType::F32,
            dst_type: IntType::U32,
            rnd_mode: FRndMode::Zero,
            ftz: false,
        }));

        let negative_denominator = self.emit_iadd_source(unsigned_rhs.clone().ineg(), Src::ZERO);
        let negative_product =
            self.emit_multiply_source(Src::from(reciprocal), negative_denominator, false);
        let correction = self.emit_multiply_source(Src::from(reciprocal), negative_product, true);
        let reciprocal = self.emit_iadd_source(Src::from(reciprocal), correction);
        let mut quotient = self.emit_multiply_source(unsigned_lhs.clone(), reciprocal, true);
        let product = self.emit_multiply_source(quotient.clone(), unsigned_rhs.clone(), false);
        let mut remainder = self.emit_iadd_source(unsigned_lhs, product.ineg());

        for _ in 0..2 {
            let remainder_ge_denominator = self.emit_integer_comparison_source(
                naga::BinaryOperator::GreaterEqual,
                naga::ScalarKind::Uint,
                remainder.clone(),
                unsigned_rhs.clone(),
            )?;
            if op == naga::BinaryOperator::Divide {
                let incremented = self.emit_iadd_source(quotient.clone(), Src::from(1_u32));
                quotient = self.emit_select_source(
                    remainder_ge_denominator.clone(),
                    incremented,
                    quotient,
                );
            }
            let reduced = self.emit_iadd_source(remainder.clone(), unsigned_rhs.clone().ineg());
            remainder = self.emit_select_source(remainder_ge_denominator, reduced, remainder);
        }

        let normal = if op == naga::BinaryOperator::Divide {
            if let Some(negative) = quotient_negative {
                let negated = self.emit_iadd_source(quotient.clone().ineg(), Src::ZERO);
                self.emit_select_source(negative, negated, quotient)
            } else {
                quotient
            }
        } else {
            if let Some(negative) = lhs_negative {
                let negated = self.emit_iadd_source(remainder.clone().ineg(), Src::ZERO);
                self.emit_select_source(negative, negated, remainder)
            } else {
                remainder
            }
        };
        let destination = self.target.ssa_alloc.alloc(RegFile::GPR);
        Ok((
            Instr::new(OpSel {
                dst: Dst::from(destination),
                cond: exceptional,
                srcs: [
                    if op == naga::BinaryOperator::Divide {
                        lhs
                    } else {
                        Src::ZERO
                    },
                    normal,
                ],
            }),
            Src::from(destination),
        ))
    }

    fn emit_integer_comparison_source(
        &mut self,
        op: naga::BinaryOperator,
        kind: naga::ScalarKind,
        lhs: Src,
        rhs: Src,
    ) -> Result<Src, Error> {
        let (instruction, result) = self.integer_comparison(op, kind, lhs, rhs)?;
        self.emit(instruction);
        Ok(result)
    }

    fn emit_predicate_binary(&mut self, op: PredSetOp, lhs: Src, rhs: Src) -> Src {
        let destination = self.target.ssa_alloc.alloc(RegFile::Pred);
        self.emit(Instr::new(OpPSetP {
            dsts: [Dst::from(destination), Dst::None],
            ops: [PredSetOp::And, op],
            srcs: [lhs, true.into(), rhs],
        }));
        Src::from(destination)
    }

    fn emit_select_source(&mut self, condition: Src, accept: Src, reject: Src) -> Src {
        let destination = self.target.ssa_alloc.alloc(RegFile::GPR);
        self.emit(Instr::new(OpSel {
            dst: Dst::from(destination),
            cond: condition,
            srcs: [accept, reject],
        }));
        Src::from(destination)
    }

    fn emit_iadd_source(&mut self, lhs: Src, rhs: Src) -> Src {
        let destination = self.target.ssa_alloc.alloc(RegFile::GPR);
        self.emit(Instr::new(OpIAdd2 {
            dst: Dst::from(destination),
            carry_out: Dst::None,
            srcs: [lhs, rhs],
        }));
        Src::from(destination)
    }

    fn emit_multiply_source(&mut self, lhs: Src, rhs: Src, high: bool) -> Src {
        let destination = self.target.ssa_alloc.alloc(RegFile::GPR);
        self.emit(Instr::new(OpIMul {
            dst: Dst::from(destination),
            srcs: [lhs, rhs],
            signed: [false; 2],
            high,
        }));
        Src::from(destination)
    }

    fn emit_logic_source(&mut self, op: LogicOp2, lhs: Src, rhs: Src) -> Src {
        let destination = self.target.ssa_alloc.alloc(RegFile::GPR);
        self.emit(Instr::new(OpLop2 {
            dst: Dst::from(destination),
            srcs: [lhs, rhs],
            op,
        }));
        Src::from(destination)
    }

    fn emit_shift_right(&mut self, value: Src, shift: Src, signed: bool) -> Src {
        let destination = self.target.ssa_alloc.alloc(RegFile::GPR);
        self.emit(Instr::new(OpShr {
            dst: Dst::from(destination),
            src: value,
            shift,
            wrap: true,
            signed,
        }));
        Src::from(destination)
    }

    fn emit_abs_source(&mut self, value: Src, sign: Src) -> Src {
        let toggled = self.emit_logic_source(LogicOp2::Xor, value, sign.clone());
        self.emit_iadd_source(toggled, sign.ineg())
    }

    fn integer_comparison(
        &mut self,
        op: naga::BinaryOperator,
        kind: naga::ScalarKind,
        lhs: Src,
        rhs: Src,
    ) -> Result<(Instr, Src), Error> {
        let dst = self.target.ssa_alloc.alloc(RegFile::Pred);
        let cmp_op = match op {
            naga::BinaryOperator::Equal => IntCmpOp::Eq,
            naga::BinaryOperator::NotEqual => IntCmpOp::Ne,
            naga::BinaryOperator::Less => IntCmpOp::Lt,
            naga::BinaryOperator::LessEqual => IntCmpOp::Le,
            naga::BinaryOperator::Greater => IntCmpOp::Gt,
            naga::BinaryOperator::GreaterEqual => IntCmpOp::Ge,
            _ => {
                return Err(Error::UnsupportedFeature(format!(
                    "integer comparison {op:?}"
                )));
            }
        };
        let cmp_type = match kind {
            naga::ScalarKind::Uint => IntCmpType::U32,
            naga::ScalarKind::Sint => IntCmpType::I32,
            _ => {
                return Err(Error::UnsupportedFeature(format!(
                    "integer comparison for {kind:?}"
                )));
            }
        };
        Ok((
            Instr::new(OpISetP {
                dst: Dst::from(dst),
                set_op: PredSetOp::And,
                cmp_op,
                cmp_type,
                ex: false,
                srcs: [lhs, rhs],
                accum: true.into(),
                low_cmp: true.into(),
            }),
            Src::from(dst),
        ))
    }

    fn multiply_matrix_vector(
        &mut self,
        matrix: &Value,
        vector: &Value,
        columns: usize,
        rows: usize,
    ) -> Result<Value, Error> {
        if matrix.kind != naga::ScalarKind::Float
            || vector.kind != naga::ScalarKind::Float
            || matrix.components.len() != columns * rows
            || vector.components.len() != columns
        {
            return Err(Error::UnsupportedFeature(format!(
                "matrix-vector multiply shape: matrix {} components ({columns}x{rows}), vector {} components",
                matrix.components.len(),
                vector.components.len()
            )));
        }
        let mut result = Vec::with_capacity(rows);
        for row in 0..rows {
            let mut accumulator = None;
            for column in 0..columns {
                let product = self.target.ssa_alloc.alloc(RegFile::GPR);
                self.emit(Instr::new(OpFMul {
                    dst: Dst::from(product),
                    srcs: [
                        matrix.components[column * rows + row].clone(),
                        vector.components[column].clone(),
                    ],
                    saturate: false,
                    rnd_mode: FRndMode::NearestEven,
                    ftz: false,
                    dnz: false,
                }));
                accumulator = Some(match accumulator {
                    None => Src::from(product),
                    Some(previous) => {
                        let sum = self.target.ssa_alloc.alloc(RegFile::GPR);
                        self.emit(Instr::new(OpFAdd {
                            dst: Dst::from(sum),
                            srcs: [previous, Src::from(product)],
                            saturate: false,
                            rnd_mode: FRndMode::NearestEven,
                            ftz: false,
                        }));
                        Src::from(sum)
                    }
                });
            }
            result.push(accumulator.expect("validated matrices have at least two columns"));
        }
        Ok(Value {
            components: result,
            kind: naga::ScalarKind::Float,
        })
    }

    fn multiply_vector_matrix(
        &mut self,
        vector: &Value,
        matrix: &Value,
        columns: usize,
        rows: usize,
    ) -> Result<Value, Error> {
        if matrix.kind != naga::ScalarKind::Float
            || vector.kind != naga::ScalarKind::Float
            || matrix.components.len() != columns * rows
            || vector.components.len() != rows
        {
            return Err(Error::UnsupportedFeature(
                "vector-matrix multiply shape".to_owned(),
            ));
        }
        let mut result = Vec::with_capacity(columns);
        for column in 0..columns {
            let mut accumulator = None;
            for row in 0..rows {
                let product = self.target.ssa_alloc.alloc(RegFile::GPR);
                self.emit(Instr::new(OpFMul {
                    dst: Dst::from(product),
                    srcs: [
                        vector.components[row].clone(),
                        matrix.components[column * rows + row].clone(),
                    ],
                    saturate: false,
                    rnd_mode: FRndMode::NearestEven,
                    ftz: false,
                    dnz: false,
                }));
                accumulator = Some(match accumulator {
                    None => Src::from(product),
                    Some(previous) => {
                        let sum = self.target.ssa_alloc.alloc(RegFile::GPR);
                        self.emit(Instr::new(OpFAdd {
                            dst: Dst::from(sum),
                            srcs: [previous, Src::from(product)],
                            saturate: false,
                            rnd_mode: FRndMode::NearestEven,
                            ftz: false,
                        }));
                        Src::from(sum)
                    }
                });
            }
            result.push(accumulator.expect("matrices have at least two rows"));
        }
        Ok(Value {
            components: result,
            kind: naga::ScalarKind::Float,
        })
    }

    fn multiply_matrices(
        &mut self,
        left: &Value,
        right: &Value,
        inner: usize,
        rows: usize,
        columns: usize,
    ) -> Result<Value, Error> {
        if left.kind != naga::ScalarKind::Float
            || right.kind != naga::ScalarKind::Float
            || left.components.len() != inner * rows
            || right.components.len() != columns * inner
        {
            return Err(Error::UnsupportedFeature(
                "matrix-matrix multiply shape".to_owned(),
            ));
        }
        let mut result = Vec::with_capacity(columns * rows);
        for column in 0..columns {
            for row in 0..rows {
                let mut accumulator = None;
                for index in 0..inner {
                    let product = self.target.ssa_alloc.alloc(RegFile::GPR);
                    self.emit(Instr::new(OpFMul {
                        dst: Dst::from(product),
                        srcs: [
                            left.components[index * rows + row].clone(),
                            right.components[column * inner + index].clone(),
                        ],
                        saturate: false,
                        rnd_mode: FRndMode::NearestEven,
                        ftz: false,
                        dnz: false,
                    }));
                    accumulator = Some(match accumulator {
                        None => Src::from(product),
                        Some(previous) => {
                            let sum = self.target.ssa_alloc.alloc(RegFile::GPR);
                            self.emit(Instr::new(OpFAdd {
                                dst: Dst::from(sum),
                                srcs: [previous, Src::from(product)],
                                saturate: false,
                                rnd_mode: FRndMode::NearestEven,
                                ftz: false,
                            }));
                            Src::from(sum)
                        }
                    });
                }
                result.push(accumulator.expect("matrices have at least two inner components"));
            }
        }
        Ok(Value {
            components: result,
            kind: naga::ScalarKind::Float,
        })
    }

    fn return_value(&mut self) -> Result<Value, Error> {
        let body = self.source.body.clone();
        let value = self
            .execute_statements(&body)?
            .ok_or_else(|| Error::UnsupportedFeature("missing return value".to_owned()))?;
        self.finalize_return(value)
    }

    fn call_argument(&mut self, argument: naga::Handle<naga::Expression>) -> Result<Value, Error> {
        if self.resource_argument(argument).is_some() {
            // Native resource identity is carried separately from SSA values.
            Ok(Value {
                components: vec![Src::ZERO],
                kind: naga::ScalarKind::Uint,
            })
        } else if self.pointer_is_argument(argument) {
            self.load_argument_pointer(argument)
        } else if self.pointer_is_local(argument) {
            self.load_local_pointer(argument)
        } else {
            self.expression(argument)
        }
    }

    fn resource_argument(
        &self,
        argument: naga::Handle<naga::Expression>,
    ) -> Option<naga::Handle<naga::GlobalVariable>> {
        match self.source.expressions[argument] {
            naga::Expression::GlobalVariable(global) => Some(global),
            naga::Expression::FunctionArgument(index) => {
                self.resource_arguments.get(&index).copied()
            }
            _ => None,
        }
    }

    fn call_resource_arguments(
        &self,
        function: naga::Handle<naga::Function>,
        arguments: &[naga::Handle<naga::Expression>],
    ) -> HashMap<u32, naga::Handle<naga::GlobalVariable>> {
        arguments
            .iter()
            .zip(&self.module.functions[function].arguments)
            .enumerate()
            .filter_map(|(index, (argument, parameter))| {
                matches!(
                    self.module.types[parameter.ty].inner,
                    naga::TypeInner::Image { .. }
                        | naga::TypeInner::Sampler { .. }
                        | naga::TypeInner::BindingArray { .. }
                )
                .then(|| {
                    u32::try_from(index)
                        .ok()
                        .zip(self.resource_argument(*argument))
                })
                .flatten()
            })
            .collect()
    }

    fn break_prefix(block: &naga::Block) -> Option<naga::Block> {
        Self::loop_control_prefix(block, true)
    }

    fn continue_prefix(block: &naga::Block) -> Option<naga::Block> {
        Self::loop_control_prefix(block, false)
    }

    fn loop_control_prefix(block: &naga::Block, is_break: bool) -> Option<naga::Block> {
        let control_index = block
            .iter()
            .rposition(|statement| !matches!(statement, naga::Statement::Emit(_)))?;
        let mut prefix = block[..control_index].to_vec();
        match &block[control_index] {
            naga::Statement::Break if is_break => {}
            naga::Statement::Continue if !is_break => {}
            naga::Statement::Block(nested) => {
                let nested_prefix = Self::loop_control_prefix(nested, is_break)?;
                if !nested_prefix.is_empty() {
                    prefix.push(naga::Statement::Block(nested_prefix));
                }
            }
            _ => return None,
        }
        Some(naga::Block::from_vec(prefix))
    }

    fn block_return_value(block: &naga::Block) -> Option<naga::Handle<naga::Expression>> {
        let mut statements = block.iter().peekable();
        while let Some(statement) = statements.next() {
            match statement {
                naga::Statement::Emit(_) => {}
                naga::Statement::Return { value: Some(value) } if statements.peek().is_none() => {
                    return Some(*value);
                }
                _ => return None,
            }
        }
        None
    }

    fn emit_conditional_break(
        &mut self,
        condition: &Value,
        break_when_true: bool,
        returned: Option<Value>,
    ) -> Result<(), Error> {
        if condition.kind != naga::ScalarKind::Bool || condition.components.len() != 1 {
            return Err(Error::UnsupportedFeature(
                "loop break condition is not a scalar boolean".to_owned(),
            ));
        }
        let exit_label = self
            .loops
            .last()
            .ok_or_else(|| Error::UnsupportedFeature("break outside a loop".to_owned()))?
            .exit_label;
        let mut predicate = Self::predicate(&condition.components[0])?;
        if !break_when_true {
            predicate = predicate.bnot();
        }
        if predicate.is_false() {
            return Ok(());
        }
        let branch_block = self.current_block;
        let locals = self.locals.clone();
        let mut instruction = Instr::new(OpBrk { target: exit_label });
        instruction.pred = predicate;
        self.emit(instruction);

        let continuation_label = self.allocate_label();
        let continuation = self.append_block(continuation_label);
        self.add_edge(branch_block, continuation);
        self.loops
            .last_mut()
            .expect("loop context checked above")
            .break_edges
            .push(LoopBreakEdge {
                block: branch_block,
                returned,
                locals,
            });
        self.current_block = continuation;
        Ok(())
    }

    fn emit_conditional_continue(
        &mut self,
        condition: &Value,
        continue_when_true: bool,
    ) -> Result<(), Error> {
        if condition.kind != naga::ScalarKind::Bool || condition.components.len() != 1 {
            return Err(Error::UnsupportedFeature(
                "loop continue condition is not a scalar boolean".to_owned(),
            ));
        }
        let continue_label = self
            .loops
            .last()
            .ok_or_else(|| Error::UnsupportedFeature("continue outside a loop".to_owned()))?
            .continue_label;
        let mut predicate = Self::predicate(&condition.components[0])?;
        if !continue_when_true {
            predicate = predicate.bnot();
        }
        if predicate.is_false() {
            return Ok(());
        }
        let branch_block = self.current_block;
        let mut instruction = Instr::new(OpCont {
            target: continue_label,
        });
        instruction.pred = predicate;
        self.emit(instruction);

        let continuation_label = self.allocate_label();
        let continuation = self.append_block(continuation_label);
        self.add_edge(branch_block, continuation);
        self.loops
            .last_mut()
            .expect("loop context checked above")
            .continue_edges
            .push(LoopContinueEdge {
                block: branch_block,
                locals: self.locals.clone(),
            });
        self.current_block = continuation;
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn lower_loop(
        &mut self,
        body: &naga::Block,
        continuing: &naga::Block,
        break_if: Option<naga::Handle<naga::Expression>>,
    ) -> Result<(), Error> {
        let header_label = self.allocate_label();
        let continue_label = self.allocate_label();
        let exit_label = self.allocate_label();
        let preheader = self.current_block;

        let mut written_locals = HashSet::default();
        let mut carried_locals = HashSet::default();
        self.collect_loop_carried_locals(body, &mut written_locals, &mut carried_locals);
        self.collect_loop_carried_locals(continuing, &mut written_locals, &mut carried_locals);
        let mut exit_locals = HashSet::default();
        self.collect_loop_written_locals(body, &mut exit_locals);
        self.collect_loop_written_locals(continuing, &mut exit_locals);
        let mut exit_local_handles = exit_locals.into_iter().collect::<Vec<_>>();
        exit_local_handles.sort_by_key(|handle| handle.index());
        for &local in &exit_local_handles {
            self.local_value(local)?;
        }
        let mut local_handles = carried_locals.into_iter().collect::<Vec<_>>();
        local_handles.sort_by_key(|handle| handle.index());
        let mut preheader_sources = OpPhiSrcs::new();
        let mut header_destinations = OpPhiDsts::new();
        let mut loop_phis = Vec::new();
        for &local in &local_handles {
            let value = self.local_value(local)?;
            let materialized = self.materialize_loop_components(&value)?;

            let destinations = (0..materialized.len())
                .map(|_| self.target.ssa_alloc.alloc(RegFile::GPR))
                .collect::<Vec<_>>();
            let mut phis = Vec::with_capacity(materialized.len());
            for (source, destination) in materialized.iter().zip(destinations.iter()) {
                let phi = self.target.phi_alloc.alloc();
                preheader_sources.srcs.push(phi, Src::from(*source));
                header_destinations.dsts.push(phi, Dst::from(*destination));
                phis.push(phi);
            }
            loop_phis.push(LoopPhi {
                local,
                phis,
                header_value: Value {
                    components: destinations.iter().copied().map(Src::from).collect(),
                    kind: value.kind,
                },
            });
        }

        self.emit(Instr::new(OpPBk { target: exit_label }));
        if !preheader_sources.srcs.is_empty() {
            self.emit(Instr::new(preheader_sources));
        }
        let header = self.append_block(header_label);
        self.add_edge(preheader, header);
        self.current_block = header;
        if !header_destinations.dsts.is_empty() {
            self.emit(Instr::new(header_destinations));
        }
        self.emit(Instr::new(OpPCnt {
            target: continue_label,
        }));
        for phi in &mut loop_phis {
            if phi.header_value.kind == naga::ScalarKind::Bool {
                let registers = phi
                    .header_value
                    .components
                    .iter()
                    .map(|source| match &source.src_ref {
                        SrcRef::SSA(ssa) if ssa.len() == 1 => Ok(ssa[0]),
                        _ => Err(Error::UnsupportedFeature(
                            "boolean loop phi is not a scalar register".to_owned(),
                        )),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                phi.header_value = self.boolean_loop_value(&registers);
            }
            self.locals.insert(phi.local, phi.header_value.clone());
        }

        self.loops.push(LoopContext {
            exit_label,
            continue_label,
            carried_locals: local_handles,
            exit_locals: exit_local_handles,
            entry_locals: self.locals.clone(),
            break_edges: Vec::new(),
            continue_edges: Vec::new(),
        });
        let break_prefix = Self::break_prefix(body);
        let continue_prefix = Self::continue_prefix(body);
        let executable_body = break_prefix
            .as_ref()
            .or(continue_prefix.as_ref())
            .unwrap_or(body);
        let returned_from_body = self.execute_statements(executable_body)?;
        let mut body_falls_through = returned_from_body.is_none();
        let body_end = self.current_block;
        if let Some(returned) = returned_from_body {
            self.emit(Instr::new(OpBrk { target: exit_label }));
            self.loops
                .last_mut()
                .expect("loop context was pushed")
                .break_edges
                .push(LoopBreakEdge {
                    block: body_end,
                    returned: Some(returned),
                    locals: self.locals.clone(),
                });
        } else if break_prefix.is_some() {
            let exit_label = self
                .loops
                .last()
                .expect("loop context was pushed")
                .exit_label;
            self.emit(Instr::new(OpBrk { target: exit_label }));
            self.loops
                .last_mut()
                .expect("loop context was pushed")
                .break_edges
                .push(LoopBreakEdge {
                    block: body_end,
                    returned: None,
                    locals: self.locals.clone(),
                });
            body_falls_through = false;
        }
        let continuing_block = self.append_block(continue_label);
        let mut continue_edges = self
            .loops
            .last()
            .expect("loop context was pushed")
            .continue_edges
            .clone();
        if body_falls_through {
            continue_edges.push(LoopContinueEdge {
                block: body_end,
                locals: self.locals.clone(),
            });
        }
        for edge in &continue_edges {
            self.add_edge(edge.block, continuing_block);
        }
        self.current_block = continuing_block;
        self.merge_continue_locals(&continue_edges, continuing_block)?;
        if self.execute_statements(continuing)?.is_some() {
            return Err(Error::UnsupportedFeature(
                "return from a loop continuing block".to_owned(),
            ));
        }
        if let Some(condition) = break_if {
            let condition = self.expression(condition)?;
            self.emit_conditional_break(&condition, true, None)?;
        }

        let mut backedge_sources = OpPhiSrcs::new();
        for phi in &loop_phis {
            let value = self.local_value(phi.local)?;
            let materialized = self.materialize_loop_components(&value)?;
            if materialized.len() != phi.phis.len() {
                return Err(Error::UnsupportedFeature(
                    "loop-carried local changed shape".to_owned(),
                ));
            }
            for (phi, source) in phi.phis.iter().zip(materialized.iter()) {
                backedge_sources.srcs.push(*phi, Src::from(*source));
            }
        }
        if !backedge_sources.srcs.is_empty() {
            self.emit(Instr::new(backedge_sources));
        }
        let backedge = self.current_block;
        self.emit(Instr::new(OpCont {
            target: header_label,
        }));
        self.add_edge(backedge, header);

        let context = self.loops.pop().expect("loop context was pushed");
        let exit = self.append_block(exit_label);
        for edge in &context.break_edges {
            self.add_edge(edge.block, exit);
        }
        self.current_block = exit;
        self.merge_break_locals(&context, exit)?;
        self.merge_loop_returns(&context)?;
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn merge_break_locals(
        &mut self,
        context: &LoopContext,
        exit_block: usize,
    ) -> Result<(), Error> {
        if context
            .break_edges
            .iter()
            .any(|edge| edge.returned.is_some())
        {
            self.locals.clone_from(&context.entry_locals);
            return Ok(());
        }
        let Some(first) = context.break_edges.first() else {
            self.locals.clone_from(&context.entry_locals);
            return Ok(());
        };
        let mut local_handles = Vec::new();
        for &handle in &context.exit_locals {
            if context
                .break_edges
                .iter()
                .filter(|edge| edge.returned.is_none())
                .any(|edge| edge.locals.get(&handle) != context.entry_locals.get(&handle))
            {
                local_handles.push(handle);
            }
        }
        local_handles.sort_by_key(|handle| handle.index());
        let mut merges = Vec::with_capacity(local_handles.len());
        for handle in local_handles {
            let template = first
                .locals
                .get(&handle)
                .ok_or_else(|| Error::UnsupportedFeature("missing break-edge local".to_owned()))?;
            let phis = (0..template.components.len())
                .map(|_| self.target.phi_alloc.alloc())
                .collect::<Vec<_>>();
            let destinations = (0..template.components.len())
                .map(|_| self.target.ssa_alloc.alloc(RegFile::GPR))
                .collect::<Vec<_>>();
            merges.push((
                handle,
                template.kind,
                template.components.len(),
                phis,
                destinations,
            ));
        }

        for edge in &context.break_edges {
            self.current_block = edge.block;
            let branch = self.blocks[edge.block].instrs.pop().ok_or_else(|| {
                Error::UnsupportedFeature("loop break block has no branch".to_owned())
            })?;
            if !branch.op.is_branch() {
                return Err(Error::UnsupportedFeature(
                    "loop break block does not end in a branch".to_owned(),
                ));
            }
            let mut sources = OpPhiSrcs::new();
            for (handle, kind, component_count, phis, _) in &merges {
                let values = if edge.returned.is_some() {
                    &context.entry_locals
                } else {
                    &edge.locals
                };
                let value = values.get(handle).ok_or_else(|| {
                    Error::UnsupportedFeature("missing break-edge local".to_owned())
                })?;
                if value.kind != *kind || value.components.len() != *component_count {
                    return Err(Error::UnsupportedFeature(
                        "break-edge local changed shape".to_owned(),
                    ));
                }
                let materialized = self.materialize_loop_components(value)?;
                for (phi, source) in phis.iter().zip(materialized) {
                    sources.srcs.push(*phi, Src::from(source));
                }
            }
            if !sources.srcs.is_empty() {
                self.emit(Instr::new(sources));
            }
            self.emit(branch);
        }

        self.current_block = exit_block;
        let mut destinations = OpPhiDsts::new();
        for (_, _, _, phis, registers) in &merges {
            for (phi, register) in phis.iter().zip(registers) {
                destinations.dsts.push(*phi, Dst::from(*register));
            }
        }
        if !destinations.dsts.is_empty() {
            self.emit(Instr::new(destinations));
        }

        self.locals.clone_from(&context.entry_locals);
        for (handle, kind, _, _, registers) in merges {
            let value = if kind == naga::ScalarKind::Bool {
                self.boolean_loop_value(&registers)
            } else {
                Value {
                    components: registers.into_iter().map(Src::from).collect(),
                    kind,
                }
            };
            self.locals.insert(handle, value);
        }
        Ok(())
    }

    fn merge_continue_locals(
        &mut self,
        edges: &[LoopContinueEdge],
        continuing_block: usize,
    ) -> Result<(), Error> {
        let Some(first) = edges.first() else {
            return Ok(());
        };
        let local_handles = self
            .loops
            .last()
            .expect("loop context was pushed")
            .carried_locals
            .clone();
        let mut merges = Vec::with_capacity(local_handles.len());
        for handle in local_handles {
            let template = first.locals.get(&handle).ok_or_else(|| {
                Error::UnsupportedFeature("missing continue-edge local".to_owned())
            })?;
            let phis = (0..template.components.len())
                .map(|_| self.target.phi_alloc.alloc())
                .collect::<Vec<_>>();
            let destinations = (0..template.components.len())
                .map(|_| self.target.ssa_alloc.alloc(RegFile::GPR))
                .collect::<Vec<_>>();
            merges.push((
                handle,
                template.kind,
                template.components.len(),
                phis,
                destinations,
            ));
        }

        for edge in edges {
            self.current_block = edge.block;
            let branch = self.blocks[edge.block]
                .instrs
                .last()
                .is_some_and(|instruction| instruction.op.is_branch())
                .then(|| self.blocks[edge.block].instrs.pop().expect("branch exists"));
            let mut phi_sources = OpPhiSrcs::new();
            for (handle, kind, component_count, phis, _) in &merges {
                let value = edge.locals.get(handle).ok_or_else(|| {
                    Error::UnsupportedFeature("missing continue-edge local".to_owned())
                })?;
                if value.kind != *kind || value.components.len() != *component_count {
                    return Err(Error::UnsupportedFeature(
                        "continue-edge local changed shape".to_owned(),
                    ));
                }
                let sources = self.materialize_loop_components(value)?;
                for (phi, source) in phis.iter().zip(sources) {
                    phi_sources.srcs.push(*phi, Src::from(source));
                }
            }
            self.emit(Instr::new(phi_sources));
            if let Some(branch) = branch {
                self.emit(branch);
            }
        }

        self.current_block = continuing_block;
        let mut phi_destinations = OpPhiDsts::new();
        for (_, _, _, phis, destinations) in &merges {
            for (phi, destination) in phis.iter().zip(destinations) {
                phi_destinations.dsts.push(*phi, Dst::from(*destination));
            }
        }
        self.emit(Instr::new(phi_destinations));

        let mut merged_locals = self
            .loops
            .last()
            .expect("loop context was pushed")
            .entry_locals
            .clone();
        for (handle, kind, _, _, destinations) in merges {
            let value = if kind == naga::ScalarKind::Bool {
                self.boolean_loop_value(&destinations)
            } else {
                Value {
                    components: destinations.into_iter().map(Src::from).collect(),
                    kind,
                }
            };
            merged_locals.insert(handle, value);
        }
        self.current_block = continuing_block;
        self.locals = merged_locals;
        Ok(())
    }

    fn collect_loop_carried_locals(
        &self,
        block: &naga::Block,
        written: &mut HashSet<naga::Handle<naga::LocalVariable>>,
        carried: &mut HashSet<naga::Handle<naga::LocalVariable>>,
    ) {
        for statement in block {
            match statement {
                naga::Statement::Emit(range) => {
                    for expression in range.clone() {
                        if let naga::Expression::Load { pointer } =
                            self.source.expressions[expression]
                            && let Ok((local, _, _)) = self.local_pointer(pointer)
                            && !written.contains(&local)
                        {
                            carried.insert(local);
                        }
                    }
                }
                naga::Statement::Store { pointer, .. } => {
                    if let Ok((local, _, _)) = self.local_pointer(*pointer) {
                        written.insert(local);
                    }
                }
                naga::Statement::Atomic { pointer, .. } => {
                    if let Ok((local, _, _)) = self.local_pointer(*pointer) {
                        if !written.contains(&local) {
                            carried.insert(local);
                        }
                        written.insert(local);
                    }
                }
                naga::Statement::Call { arguments, .. } => {
                    for argument in arguments {
                        if let Ok((local, _, _)) = self.local_pointer(*argument) {
                            if !written.contains(&local) {
                                carried.insert(local);
                            }
                            written.insert(local);
                        }
                    }
                }
                naga::Statement::Block(block) => {
                    self.collect_loop_carried_locals(block, written, carried);
                }
                naga::Statement::If { accept, reject, .. } => {
                    let mut accept_written = written.clone();
                    let mut reject_written = written.clone();
                    self.collect_loop_carried_locals(accept, &mut accept_written, carried);
                    self.collect_loop_carried_locals(reject, &mut reject_written, carried);
                    *written = accept_written
                        .intersection(&reject_written)
                        .copied()
                        .collect();
                }
                naga::Statement::Switch { cases, .. } => {
                    let mut all_written = None::<HashSet<_>>;
                    for case in cases {
                        let mut case_written = written.clone();
                        self.collect_loop_carried_locals(&case.body, &mut case_written, carried);
                        all_written = Some(match all_written {
                            Some(previous) => {
                                previous.intersection(&case_written).copied().collect()
                            }
                            None => case_written,
                        });
                    }
                    if let Some(all_written) = all_written {
                        *written = all_written;
                    }
                }
                naga::Statement::Loop {
                    body, continuing, ..
                } => {
                    let mut nested_written = written.clone();
                    self.collect_loop_carried_locals(body, &mut nested_written, carried);
                    self.collect_loop_carried_locals(continuing, &mut nested_written, carried);
                }
                _ => {}
            }
        }
    }

    fn collect_loop_written_locals(
        &self,
        block: &naga::Block,
        written: &mut HashSet<naga::Handle<naga::LocalVariable>>,
    ) {
        for statement in block {
            match statement {
                naga::Statement::Store { pointer, .. }
                | naga::Statement::Atomic { pointer, .. } => {
                    if let Ok((local, _, _)) = self.local_pointer(*pointer) {
                        written.insert(local);
                    }
                }
                naga::Statement::Call { arguments, .. } => {
                    for argument in arguments {
                        if let Ok((local, _, _)) = self.local_pointer(*argument) {
                            written.insert(local);
                        }
                    }
                }
                naga::Statement::Block(block) => {
                    self.collect_loop_written_locals(block, written);
                }
                naga::Statement::If { accept, reject, .. } => {
                    self.collect_loop_written_locals(accept, written);
                    self.collect_loop_written_locals(reject, written);
                }
                naga::Statement::Switch { cases, .. } => {
                    for case in cases {
                        self.collect_loop_written_locals(&case.body, written);
                    }
                }
                naga::Statement::Loop {
                    body, continuing, ..
                } => {
                    self.collect_loop_written_locals(body, written);
                    self.collect_loop_written_locals(continuing, written);
                }
                _ => {}
            }
        }
    }

    fn merge_loop_returns(&mut self, context: &LoopContext) -> Result<(), Error> {
        let Some(template) = context
            .break_edges
            .iter()
            .find_map(|edge| edge.returned.clone())
        else {
            return Ok(());
        };
        if context.break_edges.iter().any(|edge| {
            edge.returned.as_ref().is_some_and(|value| {
                value.kind != template.kind || value.components.len() != template.components.len()
            })
        }) {
            return Err(Error::UnsupportedFeature(
                "loop returns with incompatible value types".to_owned(),
            ));
        }

        let flag_phi = self.target.phi_alloc.alloc();
        let flag_destination = self.target.ssa_alloc.alloc(RegFile::GPR);
        let value_phis = (0..template.components.len())
            .map(|_| self.target.phi_alloc.alloc())
            .collect::<Vec<_>>();
        let value_destinations = (0..template.components.len())
            .map(|_| self.target.ssa_alloc.alloc(RegFile::GPR))
            .collect::<Vec<_>>();
        let exit = self.current_block;

        for edge in &context.break_edges {
            self.current_block = edge.block;
            let branch = self.blocks[edge.block].instrs.pop().ok_or_else(|| {
                Error::UnsupportedFeature("loop break block has no branch".to_owned())
            })?;
            if !branch.op.is_branch() {
                return Err(Error::UnsupportedFeature(
                    "loop break block does not end in a branch".to_owned(),
                ));
            }
            let returned = edge.returned.clone().unwrap_or_else(|| Value {
                components: vec![
                    match template.kind {
                        naga::ScalarKind::Float => Src::from(0.0_f32),
                        naga::ScalarKind::Bool => Src::new_imm_bool(false),
                        _ => Src::ZERO,
                    };
                    template.components.len()
                ],
                kind: template.kind,
            });
            let flag = Value {
                components: vec![Src::new_imm_bool(edge.returned.is_some())],
                kind: naga::ScalarKind::Bool,
            };
            let flag = self.materialize_loop_components(&flag)?;
            let returned = self.materialize_loop_components(&returned)?;
            let mut sources = OpPhiSrcs::new();
            sources.srcs.push(flag_phi, Src::from(flag[0]));
            for (phi, source) in value_phis.iter().zip(returned) {
                sources.srcs.push(*phi, Src::from(source));
            }
            self.emit(Instr::new(sources));
            self.emit(branch);
        }

        self.current_block = exit;
        let mut destinations = OpPhiDsts::new();
        destinations
            .dsts
            .push(flag_phi, Dst::from(flag_destination));
        for (phi, destination) in value_phis.iter().zip(&value_destinations) {
            destinations.dsts.push(*phi, Dst::from(*destination));
        }
        self.emit(Instr::new(destinations));
        let flag = self.boolean_loop_value(&[flag_destination]);
        let returned = if template.kind == naga::ScalarKind::Bool {
            self.boolean_loop_value(&value_destinations)
        } else {
            Value {
                components: value_destinations.into_iter().map(Src::from).collect(),
                kind: template.kind,
            }
        };
        let returned_predicate = Self::predicate(&flag.components[0])?;
        self.early_returns
            .push((flag.components[0].clone(), returned));
        self.execution_predicate =
            self.combine_predicates(self.execution_predicate, returned_predicate.bnot());
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn execute_statements(&mut self, body: &naga::Block) -> Result<Option<Value>, Error> {
        for statement in body {
            match statement {
                naga::Statement::Return {
                    value: Some(handle),
                } => {
                    let value = self.expression(*handle)?;
                    return Ok(Some(value));
                }
                naga::Statement::Call {
                    function,
                    arguments,
                    result: Some(call_result),
                } => {
                    let resource_arguments = self.call_resource_arguments(*function, arguments);
                    let argument_values = arguments
                        .iter()
                        .map(|argument| self.call_argument(*argument))
                        .collect::<Result<Vec<_>, _>>()?;
                    let (value, updated_arguments) =
                        self.inline_call(*function, argument_values, resource_arguments)?;
                    self.write_back_call_pointer_arguments(
                        *function,
                        arguments,
                        updated_arguments,
                    )?;
                    self.values.insert(*call_result, value);
                }
                naga::Statement::Call {
                    function,
                    arguments,
                    result: None,
                } => {
                    let resource_arguments = self.call_resource_arguments(*function, arguments);
                    let argument_values = arguments
                        .iter()
                        .map(|argument| self.call_argument(*argument))
                        .collect::<Result<Vec<_>, _>>()?;
                    let updated_arguments =
                        self.inline_void_call(*function, argument_values, resource_arguments)?;
                    self.write_back_call_pointer_arguments(
                        *function,
                        arguments,
                        updated_arguments,
                    )?;
                }
                naga::Statement::Kill => self.emit(Instr::new(OpKill {})),
                naga::Statement::ControlBarrier(flags) => {
                    if flags.contains(naga::Barrier::SUB_GROUP) {
                        // GM20B warps execute in lockstep. A CTA memory fence supplies the
                        // subgroup memory ordering required by Naga without using BAR.SYNC,
                        // which would incorrectly wait for every warp in the workgroup.
                        self.emit(Instr::new(OpMemBar {
                            scope: MemScope::CTA,
                        }));
                    } else if flags.intersects(naga::Barrier::STORAGE | naga::Barrier::TEXTURE) {
                        self.emit(Instr::new(OpMemBar {
                            scope: MemScope::GPU,
                        }));
                    } else if flags.contains(naga::Barrier::WORK_GROUP) {
                        self.emit(Instr::new(OpMemBar {
                            scope: MemScope::CTA,
                        }));
                    }
                    if !flags.contains(naga::Barrier::SUB_GROUP) {
                        self.emit(Instr::new(OpBar {}));
                    }
                }
                naga::Statement::MemoryBarrier(flags) => {
                    let scope = if flags.intersects(naga::Barrier::STORAGE | naga::Barrier::TEXTURE)
                    {
                        MemScope::GPU
                    } else {
                        MemScope::CTA
                    };
                    self.emit(Instr::new(OpMemBar { scope }));
                }
                naga::Statement::WorkGroupUniformLoad { pointer, result } => {
                    if !self.pointer_is_workgroup(*pointer) {
                        return Err(Error::UnsupportedFeature(
                            "workgroup uniform load requires a workgroup pointer".to_owned(),
                        ));
                    }
                    self.emit(Instr::new(OpMemBar {
                        scope: MemScope::CTA,
                    }));
                    self.emit(Instr::new(OpBar {}));
                    let value = self.load_workgroup(*pointer)?;
                    self.emit(Instr::new(OpMemBar {
                        scope: MemScope::CTA,
                    }));
                    self.emit(Instr::new(OpBar {}));
                    self.values.insert(*result, value);
                }
                naga::Statement::ImageStore {
                    image,
                    coordinate,
                    array_index,
                    value,
                } => self.image_store(*image, *coordinate, *array_index, *value)?,
                naga::Statement::ImageAtomic {
                    image,
                    coordinate,
                    array_index,
                    fun,
                    value,
                } => self.image_atomic(*image, *coordinate, *array_index, *fun, *value)?,
                naga::Statement::SubgroupGather {
                    mode,
                    argument,
                    result,
                } => self.subgroup_gather(*mode, *argument, *result)?,
                naga::Statement::SubgroupBallot { result, predicate } => {
                    self.subgroup_ballot(*predicate, *result)?;
                }
                naga::Statement::SubgroupCollectiveOperation {
                    op,
                    collective_op,
                    argument,
                    result,
                } => self.subgroup_collective(*op, *collective_op, *argument, *result)?,
                naga::Statement::Store { pointer, value } => {
                    if self.pointer_is_argument(*pointer) {
                        let value = self.expression(*value)?;
                        self.store_argument_pointer(*pointer, value)?;
                        continue;
                    }
                    if self.pointer_is_storage(*pointer) {
                        let value = self.expression(*value)?;
                        self.store_storage(*pointer, &value)?;
                        continue;
                    }
                    if self.pointer_is_workgroup(*pointer) {
                        let value = self.expression(*value)?;
                        self.store_workgroup(*pointer, &value)?;
                        continue;
                    }
                    if !self.pointer_is_local(*pointer) {
                        return Err(Error::UnsupportedFeature(format!(
                            "store pointer {:?}",
                            self.source.expressions[*pointer]
                        )));
                    }
                    let value = self.expression(*value)?;
                    self.store_local_pointer(*pointer, value)?;
                }
                naga::Statement::Atomic {
                    pointer,
                    fun,
                    value,
                    result,
                } => {
                    if self.pointer_is_storage(*pointer) {
                        self.storage_atomic(*pointer, *fun, *value, *result)?;
                    } else if self.pointer_is_workgroup(*pointer) {
                        self.workgroup_atomic(*pointer, *fun, *value, *result)?;
                    } else {
                        return Err(Error::UnsupportedFeature(format!(
                            "atomic pointer {:?}",
                            self.source.expressions[*pointer]
                        )));
                    }
                }
                naga::Statement::Block(block) => {
                    if let Some(value) = self.execute_statements(block)? {
                        return Ok(Some(value));
                    }
                }
                naga::Statement::If {
                    condition,
                    accept,
                    reject,
                } => {
                    let condition = self.expression(*condition)?;
                    if !self.loops.is_empty()
                        && reject.is_empty()
                        && let Some(prefix) = Self::continue_prefix(accept)
                    {
                        if let Some(value) =
                            self.conditional(&condition, &prefix, &naga::Block::new())?
                        {
                            return Ok(Some(value));
                        }
                        self.emit_conditional_continue(&condition, true)?;
                        continue;
                    }
                    if !self.loops.is_empty()
                        && accept.is_empty()
                        && let Some(prefix) = Self::continue_prefix(reject)
                    {
                        if let Some(value) =
                            self.conditional(&condition, &naga::Block::new(), &prefix)?
                        {
                            return Ok(Some(value));
                        }
                        self.emit_conditional_continue(&condition, false)?;
                        continue;
                    }
                    if self.loops.len() > self.loop_base_depth && reject.is_empty() {
                        if let Some(value) = Self::block_return_value(accept) {
                            let value = self.expression(value)?;
                            self.emit_conditional_break(&condition, true, Some(value))?;
                            continue;
                        }
                    }
                    if self.loops.len() > self.loop_base_depth && accept.is_empty() {
                        if let Some(value) = Self::block_return_value(reject) {
                            let value = self.expression(value)?;
                            self.emit_conditional_break(&condition, false, Some(value))?;
                            continue;
                        }
                    }
                    if !self.loops.is_empty()
                        && accept.is_empty()
                        && let Some(prefix) = Self::break_prefix(reject)
                    {
                        if let Some(value) =
                            self.conditional(&condition, &naga::Block::new(), &prefix)?
                        {
                            return Ok(Some(value));
                        }
                        self.emit_conditional_break(&condition, false, None)?;
                        continue;
                    }
                    if !self.loops.is_empty()
                        && reject.is_empty()
                        && let Some(prefix) = Self::break_prefix(accept)
                    {
                        if let Some(value) =
                            self.conditional(&condition, &prefix, &naga::Block::new())?
                        {
                            return Ok(Some(value));
                        }
                        self.emit_conditional_break(&condition, true, None)?;
                        continue;
                    }
                    if let Some(value) = self.conditional(&condition, accept, reject)? {
                        return Ok(Some(value));
                    }
                }
                naga::Statement::Loop {
                    body,
                    continuing,
                    break_if,
                } => self.lower_loop(body, continuing, *break_if)?,
                naga::Statement::Switch { selector, cases } => {
                    if let Some(value) = self.lower_switch(*selector, cases)? {
                        return Ok(Some(value));
                    }
                }
                naga::Statement::Emit(range) => {
                    for expression in range.clone() {
                        if self.pointer_type(expression).is_some()
                            || self.expression_is_resource(expression)
                            || matches!(
                                self.source.expressions[expression],
                                naga::Expression::FunctionArgument(index)
                                    if self.source.arguments.get(index as usize).is_some_and(
                                        |argument| matches!(
                                            self.module.types[argument.ty].inner,
                                            naga::TypeInner::Pointer { .. }
                                        )
                                    )
                            )
                        {
                            continue;
                        }
                        self.expression(expression)?;
                    }
                }
                naga::Statement::Return { value: None } => {
                    return Ok(Some(Value::void()));
                }
                other => {
                    return Err(Error::UnsupportedFeature(format!("statement {other:?}")));
                }
            }
        }
        Ok(None)
    }

    fn merge_conditional_pointer_arguments(
        &mut self,
        condition: &Value,
        arguments: Vec<Value>,
        accepted_arguments: &[Value],
        rejected_arguments: &[Value],
    ) -> Result<(), Error> {
        self.arguments = arguments;
        let pointer_arguments = self
            .source
            .arguments
            .iter()
            .enumerate()
            .filter_map(|(index, argument)| {
                matches!(
                    self.module.types[argument.ty].inner,
                    naga::TypeInner::Pointer { .. }
                )
                .then_some(index)
            })
            .collect::<Vec<_>>();
        for index in pointer_arguments {
            let accepted = accepted_arguments.get(index).ok_or_else(|| {
                Error::UnsupportedFeature(format!("missing accept-branch argument {index}"))
            })?;
            let rejected = rejected_arguments.get(index).ok_or_else(|| {
                Error::UnsupportedFeature(format!("missing reject-branch argument {index}"))
            })?;
            let merged = self.select(condition, accepted.clone(), rejected.clone())?;
            self.arguments[index] = merged;
        }
        Ok(())
    }

    fn conditional(
        &mut self,
        condition: &Value,
        accept: &naga::Block,
        reject: &naga::Block,
    ) -> Result<Option<Value>, Error> {
        if condition.kind != naga::ScalarKind::Bool || condition.components.len() != 1 {
            return Err(Error::UnsupportedFeature(
                "if condition is not a scalar boolean".to_owned(),
            ));
        }
        let local_handles = self
            .source
            .local_variables
            .iter()
            .map(|(handle, _)| handle)
            .collect::<Vec<_>>();
        for handle in &local_handles {
            self.local_value(*handle)?;
        }
        let parent_predicate = self.execution_predicate;
        let condition_predicate = Self::predicate(&condition.components[0])?;
        let locals = self.locals.clone();
        let arguments = self.arguments.clone();
        let accepted_predicate = self.combine_predicates(parent_predicate, condition_predicate);
        self.execution_predicate = accepted_predicate;
        let early_returns_before_accept = self.early_returns.len();
        let accepted = self.execute_statements(accept)?;
        let accepted_has_partial_return =
            accepted.is_none() && self.early_returns.len() != early_returns_before_accept;
        let accepted_locals = self.locals.clone();
        let accepted_arguments = self.arguments.clone();
        let accepted_continuation = if accepted.is_some() {
            false.into()
        } else {
            self.execution_predicate
        };
        self.locals.clone_from(&locals);
        self.arguments.clone_from(&arguments);
        let rejected_predicate =
            self.combine_predicates(parent_predicate, condition_predicate.bnot());
        self.execution_predicate = rejected_predicate;
        let early_returns_before_reject = self.early_returns.len();
        let rejected = self.execute_statements(reject)?;
        let rejected_has_partial_return =
            rejected.is_none() && self.early_returns.len() != early_returns_before_reject;
        let rejected_locals = self.locals.clone();
        let rejected_arguments = self.arguments.clone();
        let rejected_continuation = if rejected.is_some() {
            false.into()
        } else {
            self.execution_predicate
        };
        self.merge_conditional_pointer_arguments(
            condition,
            arguments,
            &accepted_arguments,
            &rejected_arguments,
        )?;
        match (accepted, rejected) {
            (Some(accepted), Some(rejected)) => {
                self.execution_predicate = parent_predicate;
                Ok(Some(self.select(condition, accepted, rejected)?))
            }
            (Some(accepted), None) => {
                self.early_returns
                    .push((Src::from(accepted_predicate), accepted));
                self.locals = rejected_locals;
                self.execution_predicate = rejected_continuation;
                Ok(None)
            }
            (None, Some(rejected)) => {
                self.early_returns
                    .push((Src::from(rejected_predicate), rejected));
                self.locals = accepted_locals;
                self.execution_predicate = accepted_continuation;
                Ok(None)
            }
            (None, None) => {
                self.locals.clear();
                for handle in local_handles {
                    let accepted = accepted_locals.get(&handle).ok_or_else(|| {
                        Error::UnsupportedFeature("missing accept-branch local".to_owned())
                    })?;
                    let rejected = rejected_locals.get(&handle).ok_or_else(|| {
                        Error::UnsupportedFeature("missing reject-branch local".to_owned())
                    })?;
                    let merged = self.select(condition, accepted.clone(), rejected.clone())?;
                    self.locals.insert(handle, merged);
                }
                self.execution_predicate =
                    if !accepted_has_partial_return && !rejected_has_partial_return {
                        parent_predicate
                    } else {
                        self.combine_predicates_or(accepted_continuation, rejected_continuation)
                    };
                Ok(None)
            }
        }
    }

    fn lower_switch(
        &mut self,
        selector: naga::Handle<naga::Expression>,
        cases: &[naga::SwitchCase],
    ) -> Result<Option<Value>, Error> {
        let selector = self.expression(selector)?;
        if selector.components.len() != 1
            || !matches!(
                selector.kind,
                naga::ScalarKind::Uint | naga::ScalarKind::Sint
            )
        {
            return Err(Error::UnsupportedFeature(
                "switch selector must be an integer scalar".to_owned(),
            ));
        }
        if cases.iter().all(|case| !case.fall_through) {
            return self.lower_simple_switch(&selector, cases);
        }
        let false_value = || Value {
            components: vec![Src::new_imm_bool(false)],
            kind: naga::ScalarKind::Bool,
        };
        let mut groups = Vec::new();
        let mut group_values = Vec::new();
        for case in cases {
            group_values.push(case.value);
            if case.fall_through {
                if !case.body.is_empty() {
                    return Err(Error::UnsupportedFeature(
                        "non-empty fall-through switch case".to_owned(),
                    ));
                }
            } else {
                groups.push((std::mem::take(&mut group_values), &case.body));
            }
        }
        if !group_values.is_empty() {
            return Err(Error::UnsupportedFeature(
                "trailing fall-through switch case".to_owned(),
            ));
        }

        let mut matched = false_value();
        for value in cases.iter().filter_map(|case| match case.value {
            naga::SwitchValue::Default => None,
            value => Some(value),
        }) {
            let condition = self.switch_value_condition(&selector, value)?;
            matched = self.boolean_or(&matched, &condition)?;
        }
        let default_condition = Value {
            components: vec![matched.components[0].clone().bnot()],
            kind: naga::ScalarKind::Bool,
        };
        let empty = naga::Block::new();
        for (values, body) in groups {
            let mut condition = false_value();
            for value in values {
                let value_condition = match value {
                    naga::SwitchValue::Default => default_condition.clone(),
                    value => self.switch_value_condition(&selector, value)?,
                };
                condition = self.boolean_or(&condition, &value_condition)?;
            }
            if let Some(value) = self.conditional(&condition, body, &empty)? {
                return Ok(Some(value));
            }
        }
        Ok(None)
    }

    fn lower_simple_switch(
        &mut self,
        selector: &Value,
        cases: &[naga::SwitchCase],
    ) -> Result<Option<Value>, Error> {
        let mut matched = Value {
            components: vec![Src::new_imm_bool(false)],
            kind: naga::ScalarKind::Bool,
        };
        let empty = naga::Block::new();
        let mut default = None;
        for case in cases {
            let literal = match case.value {
                naga::SwitchValue::I32(value) => Value {
                    components: vec![Src::from(value.cast_unsigned())],
                    kind: naga::ScalarKind::Sint,
                },
                naga::SwitchValue::U32(value) => Value {
                    components: vec![Src::from(value)],
                    kind: naga::ScalarKind::Uint,
                },
                naga::SwitchValue::Default => {
                    default = Some(&case.body);
                    continue;
                }
            };
            let condition = self.binary(naga::BinaryOperator::Equal, selector, &literal, None)?;
            if let Some(value) = self.conditional(&condition, &case.body, &empty)? {
                return Ok(Some(value));
            }
            let destination = self.target.ssa_alloc.alloc(RegFile::Pred);
            self.emit(Instr::new(OpPSetP {
                dsts: [Dst::from(destination), Dst::None],
                ops: [PredSetOp::And, PredSetOp::Or],
                srcs: [
                    matched.components[0].clone(),
                    true.into(),
                    condition.components[0].clone(),
                ],
            }));
            matched.components[0] = Src::from(destination);
        }
        if let Some(default) = default {
            matched.components[0] = matched.components[0].clone().bnot();
            if let Some(value) = self.conditional(&matched, default, &empty)? {
                return Ok(Some(value));
            }
        }
        Ok(None)
    }

    fn boolean_or(&mut self, left: &Value, right: &Value) -> Result<Value, Error> {
        if left.kind != naga::ScalarKind::Bool
            || right.kind != naga::ScalarKind::Bool
            || left.components.len() != 1
            || right.components.len() != 1
        {
            return Err(Error::UnsupportedFeature(
                "logical OR requires scalar booleans".to_owned(),
            ));
        }
        let left = Self::predicate(&left.components[0])?;
        let right = Self::predicate(&right.components[0])?;
        Ok(Value {
            components: vec![Src::from(self.combine_predicates_or(left, right))],
            kind: naga::ScalarKind::Bool,
        })
    }

    fn switch_value_condition(
        &mut self,
        selector: &Value,
        value: naga::SwitchValue,
    ) -> Result<Value, Error> {
        let literal = match value {
            naga::SwitchValue::I32(value) => Value {
                components: vec![Src::from(value.cast_unsigned())],
                kind: naga::ScalarKind::Sint,
            },
            naga::SwitchValue::U32(value) => Value {
                components: vec![Src::from(value)],
                kind: naga::ScalarKind::Uint,
            },
            naga::SwitchValue::Default => {
                return Err(Error::UnsupportedFeature(
                    "default switch value used as a literal".to_owned(),
                ));
            }
        };
        self.binary(naga::BinaryOperator::Equal, selector, &literal, None)
    }

    fn finalize_return(&mut self, mut value: Value) -> Result<Value, Error> {
        let early_returns = std::mem::take(&mut self.early_returns);
        for (condition, early) in early_returns.into_iter().rev() {
            let condition = Value {
                components: vec![condition],
                kind: naga::ScalarKind::Bool,
            };
            value = self.select(&condition, early, value)?;
        }
        Ok(value)
    }

    fn write_back_call_pointer_arguments(
        &mut self,
        function: naga::Handle<naga::Function>,
        arguments: &[naga::Handle<naga::Expression>],
        updated_arguments: Vec<Value>,
    ) -> Result<(), Error> {
        for ((argument, parameter), value) in arguments
            .iter()
            .zip(&self.module.functions[function].arguments)
            .zip(updated_arguments)
        {
            if !matches!(
                self.module.types[parameter.ty].inner,
                naga::TypeInner::Pointer { .. }
            ) {
                continue;
            }
            if self.pointer_is_argument(*argument) {
                self.store_argument_pointer(*argument, value)?;
            } else if self.pointer_is_local(*argument) {
                self.store_local_pointer(*argument, value)?;
            } else {
                return Err(Error::UnsupportedFeature(format!(
                    "call pointer argument {:?}",
                    self.source.expressions[*argument]
                )));
            }
        }
        Ok(())
    }

    fn inline_call(
        &mut self,
        function: naga::Handle<naga::Function>,
        arguments: Vec<Value>,
        resource_arguments: HashMap<u32, naga::Handle<naga::GlobalVariable>>,
    ) -> Result<(Value, Vec<Value>), Error> {
        let target = std::mem::replace(&mut self.target, Function::single_block(Vec::new()));
        let loop_base_depth = self.loops.len();
        let mut callee = Self {
            module: self.module,
            source: &self.module.functions[function],
            resources: self.resources,
            target,
            blocks: std::mem::take(&mut self.blocks),
            edges: std::mem::take(&mut self.edges),
            current_block: self.current_block,
            execution_predicate: self.execution_predicate,
            labels: std::mem::replace(&mut self.labels, LabelAllocator::new()),
            loops: std::mem::take(&mut self.loops),
            loop_base_depth,
            values: HashMap::default(),
            locals: HashMap::default(),
            arguments,
            resource_arguments,
            early_returns: Vec::new(),
        };
        let result = callee.return_value().map_err(|error| match error {
            Error::UnsupportedFeature(message) => Error::UnsupportedFeature(format!(
                "in function {}: {message}",
                callee.source.name.as_deref().unwrap_or("<unnamed>")
            )),
            error => error,
        });
        let updated_arguments = callee.arguments.clone();
        self.target = callee.target;
        self.blocks = callee.blocks;
        self.edges = callee.edges;
        self.current_block = callee.current_block;
        self.labels = callee.labels;
        self.loops = callee.loops;
        result.map(|value| (value, updated_arguments))
    }

    fn inline_void_call(
        &mut self,
        function: naga::Handle<naga::Function>,
        arguments: Vec<Value>,
        resource_arguments: HashMap<u32, naga::Handle<naga::GlobalVariable>>,
    ) -> Result<Vec<Value>, Error> {
        let target = std::mem::replace(&mut self.target, Function::single_block(Vec::new()));
        let loop_base_depth = self.loops.len();
        let mut callee = Self {
            module: self.module,
            source: &self.module.functions[function],
            resources: self.resources,
            target,
            blocks: std::mem::take(&mut self.blocks),
            edges: std::mem::take(&mut self.edges),
            current_block: self.current_block,
            execution_predicate: self.execution_predicate,
            labels: std::mem::replace(&mut self.labels, LabelAllocator::new()),
            loops: std::mem::take(&mut self.loops),
            loop_base_depth,
            values: HashMap::default(),
            locals: HashMap::default(),
            arguments,
            resource_arguments,
            early_returns: Vec::new(),
        };
        let body = callee.source.body.clone();
        let result = (|| {
            let value = callee
                .execute_statements(&body)?
                .unwrap_or_else(Value::void);
            let value = callee.finalize_return(value)?;
            if value.is_void() {
                Ok(callee.arguments.clone())
            } else {
                Err(Error::UnsupportedFeature(
                    "void function returned a value".to_owned(),
                ))
            }
        })();
        let result = result.map_err(|error| match error {
            Error::UnsupportedFeature(message) => Error::UnsupportedFeature(format!(
                "in function {}: {message}",
                callee.source.name.as_deref().unwrap_or("<unnamed>")
            )),
            error => error,
        });
        self.target = callee.target;
        self.blocks = callee.blocks;
        self.edges = callee.edges;
        self.current_block = callee.current_block;
        self.labels = callee.labels;
        self.loops = callee.loops;
        result
    }

    fn materialize(&mut self, value: Value) -> Result<SSARef, Error> {
        let components = u8::try_from(value.components.len()).map_err(|_| {
            Error::UnsupportedFeature("values wider than 255 components".to_owned())
        })?;
        if components == 0 {
            return Err(Error::UnsupportedFeature("empty value".to_owned()));
        }
        let ssa = self.target.ssa_alloc.alloc_vec(RegFile::GPR, components);
        let kind = value.kind;
        for (dst, src) in ssa.iter().zip(value.components) {
            self.materialize_component(*dst, src, kind);
        }
        Ok(ssa)
    }

    fn materialize_components(&mut self, value: Value) -> Vec<SSAValue> {
        let kind = value.kind;
        value
            .components
            .into_iter()
            .map(|source| {
                let destination = self.target.ssa_alloc.alloc(RegFile::GPR);
                self.materialize_component(destination, source, kind);
                destination
            })
            .collect()
    }

    fn materialize_component(
        &mut self,
        destination: SSAValue,
        source: Src,
        kind: naga::ScalarKind,
    ) {
        let instruction = match (kind, source.src_mod) {
            (naga::ScalarKind::Float, SrcMod::FAbs | SrcMod::FNeg | SrcMod::FNegAbs) => {
                Instr::new(OpFAdd {
                    dst: Dst::from(destination),
                    srcs: [source, Src::from(0.0_f32)],
                    saturate: false,
                    rnd_mode: FRndMode::NearestEven,
                    ftz: false,
                })
            }
            (naga::ScalarKind::Sint, SrcMod::INeg) => Instr::new(OpIAdd2 {
                dst: Dst::from(destination),
                carry_out: Dst::None,
                srcs: [source, Src::ZERO],
            }),
            (
                naga::ScalarKind::Uint | naga::ScalarKind::Sint | naga::ScalarKind::Bool,
                SrcMod::BNot,
            ) => Instr::new(OpLop2 {
                dst: Dst::from(destination),
                srcs: [Src::ZERO, source],
                op: LogicOp2::PassB,
            }),
            _ => Instr::new(OpMov {
                dst: Dst::from(destination),
                src: source,
                quad_lanes: 0xf,
            }),
        };
        self.emit(instruction);
    }

    fn materialize_loop_components(&mut self, value: &Value) -> Result<Vec<SSAValue>, Error> {
        if value.kind != naga::ScalarKind::Bool {
            return Ok(self.materialize_components(value.clone()));
        }
        value
            .components
            .iter()
            .map(|source| {
                let condition = Self::predicate(source)?;
                let destination = self.target.ssa_alloc.alloc(RegFile::GPR);
                self.emit(Instr::new(OpSel {
                    dst: Dst::from(destination),
                    cond: condition.into(),
                    srcs: [Src::from(1_u32), Src::ZERO],
                }));
                Ok(destination)
            })
            .collect()
    }

    fn boolean_loop_value(&mut self, registers: &[SSAValue]) -> Value {
        let components = registers
            .iter()
            .map(|register| {
                let destination = self.target.ssa_alloc.alloc(RegFile::Pred);
                self.emit(Instr::new(OpISetP {
                    dst: Dst::from(destination),
                    set_op: PredSetOp::And,
                    cmp_op: IntCmpOp::Ne,
                    cmp_type: IntCmpType::U32,
                    ex: false,
                    srcs: [Src::from(*register), Src::ZERO],
                    accum: true.into(),
                    low_cmp: true.into(),
                }));
                Src::from(destination)
            })
            .collect();
        Value {
            components,
            kind: naga::ScalarKind::Bool,
        }
    }

    fn finish(mut self) -> Function {
        self.emit(Instr::new(OpExit {}));
        self.target.replace_blocks(self.blocks, &self.edges);
        self.target
    }
}

fn literal_source(literal: naga::Literal) -> Result<Src, Error> {
    match literal {
        naga::Literal::F32(value) => Ok(Src::from(value)),
        naga::Literal::U32(value) => Ok(Src::from(value)),
        naga::Literal::I32(value) => Ok(Src::from(value.cast_unsigned())),
        naga::Literal::Bool(value) => Ok(Src::new_imm_bool(value)),
        other => Err(Error::UnsupportedFeature(format!("literal {other:?}"))),
    }
}

fn module_image_type(
    module: &naga::Module,
    global: naga::Handle<naga::GlobalVariable>,
) -> Result<(naga::ImageDimension, bool, naga::ScalarKind), Error> {
    let ty = binding_resource_base_type(module, global);
    let naga::TypeInner::Image {
        dim,
        arrayed,
        class,
    } = module.types[ty].inner
    else {
        return Err(Error::UnsupportedFeature(
            "sampled resource is not an image".to_owned(),
        ));
    };
    match class {
        naga::ImageClass::Sampled { kind, multi: false } => Ok((dim, arrayed, kind)),
        naga::ImageClass::Depth { multi: false } => Ok((dim, arrayed, naga::ScalarKind::Float)),
        other => Err(Error::UnsupportedFeature(format!(
            "sampled image class {other:?}"
        ))),
    }
}

fn literal_kind(literal: naga::Literal) -> Result<naga::ScalarKind, Error> {
    match literal {
        naga::Literal::F32(_) => Ok(naga::ScalarKind::Float),
        naga::Literal::U32(_) => Ok(naga::ScalarKind::Uint),
        naga::Literal::I32(_) => Ok(naga::ScalarKind::Sint),
        naga::Literal::Bool(_) => Ok(naga::ScalarKind::Bool),
        other => Err(Error::UnsupportedFeature(format!("literal {other:?}"))),
    }
}

fn vector_size(size: naga::VectorSize) -> usize {
    match size {
        naga::VectorSize::Bi => 2,
        naga::VectorSize::Tri => 3,
        naga::VectorSize::Quad => 4,
    }
}

fn zero_value(module: &naga::Module, ty: naga::Handle<naga::Type>) -> Result<Value, Error> {
    let components = flat_type_components(module, ty)?;
    let kind = flat_type_kind(module, ty)?;
    let source = match kind {
        naga::ScalarKind::Float => Src::from(0.0_f32),
        naga::ScalarKind::Sint | naga::ScalarKind::Uint => Src::ZERO,
        naga::ScalarKind::Bool => Src::new_imm_bool(false),
        other => {
            return Err(Error::UnsupportedFeature(format!(
                "zero value scalar kind {other:?}"
            )));
        }
    };
    Ok(Value {
        components: vec![source; components],
        kind,
    })
}

fn flat_type_components(
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
) -> Result<usize, Error> {
    match module.types[ty].inner {
        naga::TypeInner::Scalar(_) => Ok(1),
        naga::TypeInner::Vector { size, .. } => Ok(vector_size(size)),
        naga::TypeInner::Matrix { columns, rows, .. } => {
            Ok(vector_size(columns) * vector_size(rows))
        }
        naga::TypeInner::Array { base, size, .. } => {
            let naga::ArraySize::Constant(size) = size else {
                return Err(Error::UnsupportedFeature(
                    "runtime-sized local array".to_owned(),
                ));
            };
            flat_type_components(module, base)?
                .checked_mul(size.get() as usize)
                .ok_or_else(|| Error::UnsupportedFeature("local array size overflow".to_owned()))
        }
        naga::TypeInner::Struct { ref members, .. } => members
            .iter()
            .map(|member| flat_type_components(module, member.ty))
            .try_fold(0_usize, |sum, count| {
                sum.checked_add(count?).ok_or_else(|| {
                    Error::UnsupportedFeature("local struct size overflow".to_owned())
                })
            }),
        ref inner => Err(Error::UnsupportedFeature(format!(
            "local value type {inner:?}"
        ))),
    }
}

fn flat_type_kind(
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
) -> Result<naga::ScalarKind, Error> {
    match module.types[ty].inner {
        naga::TypeInner::Scalar(scalar)
        | naga::TypeInner::Vector { scalar, .. }
        | naga::TypeInner::Matrix { scalar, .. } => Ok(scalar.kind),
        naga::TypeInner::Array { base, .. } => flat_type_kind(module, base),
        naga::TypeInner::Struct { ref members, .. } => {
            let mut kinds = members
                .iter()
                .map(|member| flat_type_kind(module, member.ty));
            let first = kinds
                .next()
                .ok_or_else(|| Error::UnsupportedFeature("empty local struct".to_owned()))??;
            for kind in kinds {
                if kind? != first {
                    return Ok(naga::ScalarKind::Float);
                }
            }
            Ok(first)
        }
        ref inner => Err(Error::UnsupportedFeature(format!(
            "local scalar kind for type {inner:?}"
        ))),
    }
}

fn global_value(
    module: &naga::Module,
    handle: naga::Handle<naga::Expression>,
) -> Result<Value, Error> {
    match &module.global_expressions[handle] {
        naga::Expression::Literal(literal) => Ok(Value {
            components: vec![literal_source(*literal)?],
            kind: literal_kind(*literal)?,
        }),
        naga::Expression::ZeroValue(ty) => zero_value(module, *ty),
        naga::Expression::Splat { size, value } => {
            let value = global_value(module, *value)?;
            if value.components.len() != 1 {
                return Err(Error::UnsupportedFeature(
                    "constant splat of a non-scalar value".to_owned(),
                ));
            }
            Ok(Value {
                components: vec![value.components[0].clone(); vector_size(*size)],
                kind: value.kind,
            })
        }
        naga::Expression::Compose { components, .. } => {
            let values = components
                .iter()
                .map(|component| global_value(module, *component))
                .collect::<Result<Vec<_>, _>>()?;
            let kind = values
                .first()
                .ok_or_else(|| Error::UnsupportedFeature("empty constant compose".to_owned()))?
                .kind;
            if values.iter().any(|value| value.kind != kind) {
                return Err(Error::UnsupportedFeature(
                    "mixed scalar kinds in constant compose".to_owned(),
                ));
            }
            Ok(Value {
                components: values
                    .into_iter()
                    .flat_map(|value| value.components)
                    .collect(),
                kind,
            })
        }
        expression => Err(Error::UnsupportedFeature(format!(
            "constant expression {expression:?}"
        ))),
    }
}

fn type_shape(
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
) -> Result<(u8, naga::ScalarKind), Error> {
    match module.types[ty].inner {
        naga::TypeInner::Scalar(scalar) => Ok((1, scalar.kind)),
        naga::TypeInner::Vector { size, scalar } => {
            let components = match size {
                naga::VectorSize::Bi => 2,
                naga::VectorSize::Tri => 3,
                naga::VectorSize::Quad => 4,
            };
            Ok((components, scalar.kind))
        }
        ref inner => Err(Error::UnsupportedFeature(format!(
            "entry-point IO type {inner:?}"
        ))),
    }
}

fn uniform_component_offsets(
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
    base: u32,
) -> Result<(Vec<u32>, naga::ScalarKind), Error> {
    match module.types[ty].inner {
        naga::TypeInner::Scalar(scalar) | naga::TypeInner::Atomic(scalar) if scalar.width == 4 => {
            Ok((vec![base], scalar.kind))
        }
        naga::TypeInner::Vector { size, scalar } if scalar.width == 4 => Ok((
            (0..vector_size(size))
                .map(|component| base + u32::try_from(component * 4).expect("vector is small"))
                .collect(),
            scalar.kind,
        )),
        naga::TypeInner::Matrix {
            columns,
            rows,
            scalar,
        } if scalar.width == 4 => {
            let columns = vector_size(columns);
            let rows = vector_size(rows);
            let row_bytes = u32::try_from(rows * 4).expect("matrix is small");
            let alignment = if rows == 2 { 8 } else { 16 };
            let stride = row_bytes.div_ceil(alignment) * alignment;
            let mut offsets = Vec::with_capacity(columns * rows);
            for column in 0..columns {
                for row in 0..rows {
                    offsets.push(
                        base + u32::try_from(column).expect("matrix is small") * stride
                            + u32::try_from(row * 4).expect("matrix is small"),
                    );
                }
            }
            Ok((offsets, scalar.kind))
        }
        naga::TypeInner::Array {
            base: element,
            size: naga::ArraySize::Constant(size),
            stride,
        } => {
            let mut offsets = Vec::new();
            let mut kind = None;
            for index in 0..size.get() {
                let element_base = base
                    .checked_add(index.checked_mul(stride).ok_or_else(|| {
                        Error::UnsupportedFeature("uniform array offset overflow".to_owned())
                    })?)
                    .ok_or_else(|| {
                        Error::UnsupportedFeature("uniform array offset overflow".to_owned())
                    })?;
                let (element_offsets, element_kind) =
                    uniform_component_offsets(module, element, element_base)?;
                kind = Some(match kind {
                    Some(kind) if kind != element_kind => naga::ScalarKind::Float,
                    Some(kind) => kind,
                    None => element_kind,
                });
                offsets.extend(element_offsets);
            }
            let kind =
                kind.ok_or_else(|| Error::UnsupportedFeature("empty uniform array".to_owned()))?;
            Ok((offsets, kind))
        }
        naga::TypeInner::Struct { ref members, .. } => {
            let mut offsets = Vec::new();
            let mut kind = None;
            for member in members {
                let member_base = base.checked_add(member.offset).ok_or_else(|| {
                    Error::UnsupportedFeature("uniform struct offset overflow".to_owned())
                })?;
                let (member_offsets, member_kind) =
                    uniform_component_offsets(module, member.ty, member_base)?;
                kind = Some(match kind {
                    Some(kind) if kind != member_kind => naga::ScalarKind::Float,
                    Some(kind) => kind,
                    None => member_kind,
                });
                offsets.extend(member_offsets);
            }
            let kind =
                kind.ok_or_else(|| Error::UnsupportedFeature("empty uniform struct".to_owned()))?;
            Ok((offsets, kind))
        }
        ref inner => Err(Error::UnsupportedFeature(format!(
            "uniform load type {inner:?}"
        ))),
    }
}

fn location_address(location: u32) -> Result<u16, Error> {
    let address = 0x80_u32
        .checked_add(location.checked_mul(16).ok_or_else(|| {
            Error::UnsupportedFeature("shader location address overflow".to_owned())
        })?)
        .ok_or_else(|| Error::UnsupportedFeature("shader location address overflow".to_owned()))?;
    u16::try_from(address)
        .map_err(|_| Error::UnsupportedFeature("shader location exceeds Maxwell IO".to_owned()))
}

struct OutputField {
    binding: naga::Binding,
    range: std::ops::Range<usize>,
}

fn output_fields(
    module: &naga::Module,
    result: &naga::FunctionResult,
) -> Result<Vec<OutputField>, Error> {
    if let Some(binding) = &result.binding {
        let (components, _) = type_shape(module, result.ty)?;
        return Ok(vec![OutputField {
            binding: binding.clone(),
            range: 0..usize::from(components),
        }]);
    }
    let naga::TypeInner::Struct { members, .. } = &module.types[result.ty].inner else {
        return Err(Error::UnsupportedFeature(
            "unbound non-struct entry-point result".to_owned(),
        ));
    };
    let mut offset = 0;
    let mut fields = Vec::with_capacity(members.len());
    for member in members {
        let binding = member.binding.clone().ok_or_else(|| {
            Error::UnsupportedFeature("unbound entry-point result member".to_owned())
        })?;
        let (components, _) = type_shape(module, member.ty)?;
        let end = offset + usize::from(components);
        fields.push(OutputField {
            binding,
            range: offset..end,
        });
        offset = end;
    }
    Ok(fields)
}

fn load_multiview_index(lowerer: &mut FunctionLowerer<'_>) -> Src {
    let dst = lowerer.target.ssa_alloc.alloc(RegFile::GPR);
    lowerer.emit(Instr::new(OpLdc {
        dst: Dst::from(dst),
        cb: Src::from(CBufRef {
            buf: CBuf::Binding(MULTIVIEW_UNIFORM_TARGET + 2),
            offset: 0,
        }),
        offset: Src::ZERO,
        mode: LdcMode::Indexed,
        mem_type: MemType::B32,
    }));
    Src::from(dst)
}

fn bind_vertex_arguments(
    module: &naga::Module,
    lowerer: &mut FunctionLowerer<'_>,
    info: &mut ShaderInfo,
    multiview: bool,
) -> Result<(), Error> {
    let arguments = lowerer.source.arguments.clone();
    for argument in &arguments {
        let fields = if let Some(binding) = &argument.binding {
            vec![(binding.clone(), argument.ty)]
        } else if let naga::TypeInner::Struct { ref members, .. } = module.types[argument.ty].inner
        {
            members
                .iter()
                .map(|member| {
                    Ok((
                        member.binding.clone().ok_or_else(|| {
                            Error::UnsupportedFeature(
                                "unbound vertex input struct member".to_owned(),
                            )
                        })?,
                        member.ty,
                    ))
                })
                .collect::<Result<Vec<_>, Error>>()?
        } else {
            return Err(Error::UnsupportedFeature(
                "unbound non-struct vertex argument".to_owned(),
            ));
        };
        let mut value = Vec::new();
        let mut kinds = Vec::new();
        for (binding, ty) in fields {
            let field = bind_vertex_field(module, lowerer, info, &binding, ty, multiview)?;
            kinds.push(field.kind);
            value.extend(field.components);
        }
        let kind = kinds
            .first()
            .copied()
            .ok_or_else(|| Error::UnsupportedFeature("empty vertex input struct".to_owned()))?;
        lowerer.arguments.push(Value {
            components: value,
            kind: if kinds.iter().all(|candidate| *candidate == kind) {
                kind
            } else {
                naga::ScalarKind::Float
            },
        });
    }
    Ok(())
}

fn bind_vertex_field(
    module: &naga::Module,
    lowerer: &mut FunctionLowerer<'_>,
    info: &mut ShaderInfo,
    binding: &naga::Binding,
    ty: naga::Handle<naga::Type>,
    multiview: bool,
) -> Result<Value, Error> {
    let (components, kind) = type_shape(module, ty)?;
    if kind == naga::ScalarKind::Bool {
        return Err(Error::UnsupportedFeature("boolean vertex input".to_owned()));
    }
    if matches!(binding, naga::Binding::BuiltIn(naga::BuiltIn::ViewIndex))
        && components == 1
        && kind == naga::ScalarKind::Uint
        && multiview
    {
        return Ok(Value {
            components: vec![load_multiview_index(lowerer)],
            kind,
        });
    }
    let addr = match binding {
        naga::Binding::Location { location, .. } => location_address(*location)?,
        naga::Binding::BuiltIn(naga::BuiltIn::InstanceIndex) if components == 1 => 0x2f8,
        naga::Binding::BuiltIn(naga::BuiltIn::VertexIndex) if components == 1 => 0x2fc,
        binding @ naga::Binding::BuiltIn(_) => {
            return Err(Error::UnsupportedFeature(format!(
                "vertex input binding {binding:?}"
            )));
        }
    };
    let ssa = lowerer.target.ssa_alloc.alloc_vec(RegFile::GPR, components);
    lowerer.emit(Instr::new(OpALd {
        dst: Dst::from(ssa.clone()),
        vtx: Src::ZERO,
        offset: Src::ZERO,
        addr,
        comps: components,
        patch: false,
        output: false,
        phys: false,
    }));
    let ShaderIoInfo::Vtg(io) = &mut info.io else {
        unreachable!();
    };
    io.mark_attrs_read(addr..addr + u16::from(components) * 4);
    Ok(Value {
        components: ssa.iter().copied().map(Src::from).collect(),
        kind,
    })
}

fn bind_fragment_arguments(
    module: &naga::Module,
    lowerer: &mut FunctionLowerer<'_>,
    info: &mut ShaderInfo,
    multiview: bool,
) -> Result<(), Error> {
    let arguments = lowerer.source.arguments.clone();
    for (argument_index, argument) in arguments.iter().enumerate() {
        let fields = if let Some(binding) = &argument.binding {
            vec![(binding.clone(), argument.ty, true)]
        } else if let naga::TypeInner::Struct { ref members, .. } = module.types[argument.ty].inner
        {
            members
                .iter()
                .enumerate()
                .map(|(member_index, member)| {
                    let binding = member.binding.clone().ok_or_else(|| {
                        Error::UnsupportedFeature("unbound fragment input struct member".to_owned())
                    })?;
                    Ok((
                        binding,
                        member.ty,
                        argument_member_used(lowerer.source, argument_index, member_index),
                    ))
                })
                .collect::<Result<Vec<_>, Error>>()?
        } else {
            return Err(Error::UnsupportedFeature(
                "unbound non-struct fragment argument".to_owned(),
            ));
        };
        let mut components = Vec::new();
        let mut kinds = Vec::new();
        for (binding, ty, used) in fields {
            let value = bind_fragment_field(module, lowerer, info, &binding, ty, used, multiview)?;
            kinds.push(value.kind);
            components.extend(value.components);
        }
        let kind = kinds
            .first()
            .copied()
            .ok_or_else(|| Error::UnsupportedFeature("empty fragment input struct".to_owned()))?;
        lowerer.arguments.push(Value {
            components,
            kind: if kinds.iter().all(|candidate| *candidate == kind) {
                kind
            } else {
                naga::ScalarKind::Float
            },
        });
    }
    Ok(())
}

fn argument_member_used(source: &naga::Function, argument: usize, member: usize) -> bool {
    source.expressions.iter().any(|(_, expression)| {
        let naga::Expression::AccessIndex { base, index } = expression else {
            return false;
        };
        *index as usize == member
            && matches!(
                source.expressions[*base],
                naga::Expression::FunctionArgument(index) if index as usize == argument
            )
    })
}

fn bind_fragment_builtin(
    lowerer: &mut FunctionLowerer<'_>,
    info: &mut ShaderInfo,
    builtin: naga::BuiltIn,
    components: u8,
    kind: naga::ScalarKind,
    used: bool,
    multiview: bool,
) -> Result<Value, Error> {
    if matches!(builtin, naga::BuiltIn::ViewIndex)
        && components == 1
        && kind == naga::ScalarKind::Uint
        && multiview
    {
        if !used {
            return Ok(Value {
                components: vec![Src::ZERO],
                kind,
            });
        }
        let ssa = lowerer.target.ssa_alloc.alloc(RegFile::GPR);
        lowerer.emit(Instr::new(OpIpa {
            dst: Dst::from(ssa),
            addr: LAYER_ATTRIBUTE_ADDRESS,
            freq: InterpFreq::Constant,
            loc: InterpLoc::Default,
            inv_w: Src::ZERO,
            offset: Src::ZERO,
        }));
        let ShaderIoInfo::Fragment(io) = &mut info.io else {
            unreachable!();
        };
        io.mark_attr_read(LAYER_ATTRIBUTE_ADDRESS, PixelImap::Constant);
        return Ok(Value {
            components: vec![Src::from(ssa)],
            kind,
        });
    }
    if matches!(builtin, naga::BuiltIn::FrontFacing)
        && components == 1
        && kind == naga::ScalarKind::Bool
    {
        let raw = lowerer.target.ssa_alloc.alloc(RegFile::GPR);
        lowerer.emit(Instr::new(OpIpa {
            dst: Dst::from(raw),
            addr: 0x3fc,
            freq: InterpFreq::Constant,
            loc: InterpLoc::Default,
            inv_w: Src::ZERO,
            offset: Src::ZERO,
        }));
        let predicate = lowerer.target.ssa_alloc.alloc(RegFile::Pred);
        lowerer.emit(Instr::new(OpISetP {
            dst: Dst::from(predicate),
            set_op: PredSetOp::And,
            cmp_op: IntCmpOp::Ne,
            cmp_type: IntCmpType::U32,
            ex: false,
            srcs: [Src::from(raw), Src::ZERO],
            accum: true.into(),
            low_cmp: true.into(),
        }));
        return Ok(Value {
            components: vec![Src::from(predicate)],
            kind,
        });
    }
    if matches!(builtin, naga::BuiltIn::Position { .. }) && components == 4 {
        if !used {
            return Ok(Value {
                components: vec![Src::ZERO; 4],
                kind,
            });
        }
        let mut value = Vec::with_capacity(4);
        for component in 0..4_u16 {
            let addr = 0x70 + component * 4;
            let ssa = lowerer.target.ssa_alloc.alloc(RegFile::GPR);
            lowerer.emit(Instr::new(OpIpa {
                dst: Dst::from(ssa),
                addr,
                freq: InterpFreq::Pass,
                loc: InterpLoc::Default,
                inv_w: Src::ZERO,
                offset: Src::ZERO,
            }));
            value.push(Src::from(ssa));
            let ShaderIoInfo::Fragment(io) = &mut info.io else {
                unreachable!();
            };
            io.mark_attr_read(addr, PixelImap::ScreenLinear);
        }
        return Ok(Value {
            components: value,
            kind,
        });
    }
    Err(Error::UnsupportedFeature(format!(
        "fragment input builtin {builtin:?}"
    )))
}

fn bind_fragment_field(
    module: &naga::Module,
    lowerer: &mut FunctionLowerer<'_>,
    info: &mut ShaderInfo,
    binding: &naga::Binding,
    ty: naga::Handle<naga::Type>,
    used: bool,
    multiview: bool,
) -> Result<Value, Error> {
    let (components, kind) = type_shape(module, ty)?;
    if let naga::Binding::BuiltIn(builtin) = binding {
        return bind_fragment_builtin(lowerer, info, *builtin, components, kind, used, multiview);
    }
    let naga::Binding::Location {
        location,
        interpolation,
        sampling,
        ..
    } = binding
    else {
        unreachable!("builtins returned above");
    };
    if kind != naga::ScalarKind::Float && *interpolation != Some(naga::Interpolation::Flat) {
        return Err(Error::UnsupportedFeature(
            "non-float fragment varying without flat interpolation".to_owned(),
        ));
    }
    let (imap, freq) = match interpolation.unwrap_or(naga::Interpolation::Perspective) {
        naga::Interpolation::Perspective => (PixelImap::Perspective, InterpFreq::Pass),
        naga::Interpolation::Linear => (PixelImap::ScreenLinear, InterpFreq::Pass),
        naga::Interpolation::Flat => (PixelImap::Constant, InterpFreq::Constant),
        naga::Interpolation::PerVertex => {
            return Err(Error::UnsupportedFeature(
                "per-vertex fragment interpolation".to_owned(),
            ));
        }
    };
    let loc = match sampling.unwrap_or(naga::Sampling::Center) {
        naga::Sampling::Center => InterpLoc::Default,
        naga::Sampling::Centroid => InterpLoc::Centroid,
        sampling => {
            return Err(Error::UnsupportedFeature(format!(
                "fragment sampling {sampling:?}"
            )));
        }
    };
    let base = location_address(*location)?;
    let mut value = Vec::with_capacity(usize::from(components));
    for component in 0..components {
        let addr = base + u16::from(component) * 4;
        let ssa = lowerer.target.ssa_alloc.alloc(RegFile::GPR);
        lowerer.emit(Instr::new(OpIpa {
            dst: Dst::from(ssa),
            addr,
            freq,
            loc,
            inv_w: Src::ZERO,
            offset: Src::ZERO,
        }));
        value.push(Src::from(ssa));
        let ShaderIoInfo::Fragment(io) = &mut info.io else {
            unreachable!();
        };
        io.mark_attr_read(addr, imap);
    }
    Ok(Value {
        components: value,
        kind,
    })
}

fn lower_vertex<'sm>(
    module: &naga::Module,
    entry: &naga::EntryPoint,
    sm: &'sm ShaderModelInfo,
    resources: &ResourceMap,
    multiview: bool,
) -> Result<Shader<'sm>, Error> {
    let result = entry.function.result.as_ref().ok_or_else(|| {
        Error::UnsupportedFeature("vertex entry point without a result".to_owned())
    })?;
    let mut info = ShaderInfo::vertex();
    let mut lowerer = FunctionLowerer::new(module, &entry.function, resources, Vec::new());
    bind_vertex_arguments(module, &mut lowerer, &mut info, multiview)?;
    let value = lowerer.return_value()?;
    let mut wrote_position = false;
    for field in output_fields(module, result)? {
        if field.range.end > value.components.len() {
            return Err(Error::UnsupportedFeature(
                "vertex result shape mismatch".to_owned(),
            ));
        }
        let components = field.range.len();
        let addr = match field.binding {
            naga::Binding::BuiltIn(naga::BuiltIn::Position { .. }) if components == 4 => {
                wrote_position = true;
                0x70
            }
            naga::Binding::Location { location, .. } if components <= 4 => {
                location_address(location)?
            }
            binding @ (naga::Binding::BuiltIn(_) | naga::Binding::Location { .. }) => {
                return Err(Error::UnsupportedFeature(format!(
                    "vertex result binding {binding:?}"
                )));
            }
        };
        let field_value = Value {
            components: value.components[field.range].to_vec(),
            kind: value.kind,
        };
        let ssa = lowerer.materialize(field_value)?;
        lowerer.emit(Instr::new(OpASt {
            vtx: Src::ZERO,
            offset: Src::ZERO,
            data: Src::from(ssa),
            addr,
            comps: u8::try_from(components).expect("vertex field has at most four components"),
            patch: false,
            phys: false,
        }));
        let end = addr + u16::try_from(components * 4).expect("vertex field is small");
        let ShaderIoInfo::Vtg(io) = &mut info.io else {
            unreachable!();
        };
        io.mark_attrs_written(addr..end);
        io.mark_store_req(addr..end);
    }
    if !wrote_position {
        return Err(Error::UnsupportedFeature(
            "vertex result does not write @builtin(position)".to_owned(),
        ));
    }
    if multiview {
        let view_index = load_multiview_index(&mut lowerer);
        lowerer.emit(Instr::new(OpASt {
            vtx: Src::ZERO,
            offset: Src::ZERO,
            data: view_index,
            addr: LAYER_ATTRIBUTE_ADDRESS,
            comps: 1,
            patch: false,
            phys: false,
        }));
        let ShaderIoInfo::Vtg(io) = &mut info.io else {
            unreachable!();
        };
        io.mark_attrs_written(LAYER_ATTRIBUTE_ADDRESS..LAYER_ATTRIBUTE_ADDRESS + 4);
        io.mark_store_req(LAYER_ATTRIBUTE_ADDRESS..LAYER_ATTRIBUTE_ADDRESS + 4);
    }
    Ok(Shader {
        sm,
        info,
        functions: vec![lowerer.finish()],
    })
}

fn lower_fragment<'sm>(
    module: &naga::Module,
    entry: &naga::EntryPoint,
    sm: &'sm ShaderModelInfo,
    resources: &ResourceMap,
    multiview: bool,
) -> Result<Shader<'sm>, Error> {
    let result = entry.function.result.as_ref();
    let mut info = ShaderInfo::fragment(entry.early_depth_test.is_some(), false, false);
    if block_uses_kill(&entry.function.body)
        || module
            .functions
            .iter()
            .any(|(_, function)| block_uses_kill(&function.body))
    {
        let ShaderStageInfo::Fragment(stage) = &mut info.stage else {
            unreachable!();
        };
        stage.uses_kill = true;
    }
    let mut lowerer = FunctionLowerer::new(module, &entry.function, resources, Vec::new());
    bind_fragment_arguments(module, &mut lowerer, &mut info, multiview)?;
    let mut outputs = Vec::new();
    let mut writes_color = 0_u32;
    let mut depth = None;
    if let Some(result) = result {
        let value = lowerer.return_value()?;
        for field in output_fields(module, result)? {
            if field.range.end > value.components.len() {
                return Err(Error::UnsupportedFeature(
                    "fragment output shape".to_owned(),
                ));
            }
            let field_value = Value {
                components: value.components[field.range].to_vec(),
                kind: value.kind,
            };
            match field.binding {
                naga::Binding::Location { location, .. } => {
                    if location >= 8 || field_value.components.len() > 4 {
                        return Err(Error::UnsupportedFeature(
                            "fragment color output shape".to_owned(),
                        ));
                    }
                    let ssa = lowerer.materialize(field_value)?;
                    let mut target = ssa.iter().copied().map(Src::from).collect::<Vec<_>>();
                    writes_color |= ((1_u32 << target.len()) - 1) << (location * 4);
                    target.resize(4, Src::ZERO);
                    outputs.extend(target);
                }
                naga::Binding::BuiltIn(naga::BuiltIn::FragDepth) => {
                    if field_value.kind != naga::ScalarKind::Float
                        || field_value.components.len() != 1
                        || depth.is_some()
                    {
                        return Err(Error::UnsupportedFeature(
                            "fragment depth output shape".to_owned(),
                        ));
                    }
                    depth = Some(Src::from(lowerer.materialize(field_value)?));
                }
                binding @ naga::Binding::BuiltIn(_) => {
                    return Err(Error::UnsupportedFeature(format!(
                        "fragment result binding {binding:?}"
                    )));
                }
            }
        }
    } else {
        let body = entry.function.body.clone();
        let returned = lowerer
            .execute_statements(&body)?
            .unwrap_or_else(Value::void);
        if !lowerer.finalize_return(returned)?.is_void() {
            return Err(Error::UnsupportedFeature(
                "void fragment entry point returned a value".to_owned(),
            ));
        }
    }
    let writes_depth = depth.is_some();
    if let Some(depth) = depth {
        // Maxwell places the sample mask and depth after the packed color outputs. The two ABI
        // slots travel together even when only depth is written.
        outputs.push(Src::ZERO);
        outputs.push(depth);
    }
    lowerer.emit(Instr::new(OpRegOut { srcs: outputs }));
    let ShaderIoInfo::Fragment(io) = &mut info.io else {
        unreachable!();
    };
    io.writes_color = writes_color;
    io.writes_depth = writes_depth;
    Ok(Shader {
        sm,
        info,
        functions: vec![lowerer.finish()],
    })
}

fn block_uses_kill(block: &naga::Block) -> bool {
    block.iter().any(|statement| match statement {
        naga::Statement::Kill => true,
        naga::Statement::Block(block) => block_uses_kill(block),
        naga::Statement::If { accept, reject, .. } => {
            block_uses_kill(accept) || block_uses_kill(reject)
        }
        naga::Statement::Switch { cases, .. } => {
            cases.iter().any(|case| block_uses_kill(&case.body))
        }
        naga::Statement::Loop {
            body, continuing, ..
        } => block_uses_kill(body) || block_uses_kill(continuing),
        _ => false,
    })
}
