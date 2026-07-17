use deko_nak::ir::{
    BasicBlock, CBuf, CBufRef, ChannelMask, Dst, FRndMode, FloatCmpOp, FloatType, Function,
    HasRegFile, Instr, IntCmpOp, IntCmpType, IntType, InterpFreq, InterpLoc, Label, LabelAllocator,
    LdcMode, LogicOp2, MemEvictionPriority, MemType, MuFuOp, OpALd, OpASt, OpBrk, OpCont, OpExit,
    OpF2I, OpFAdd, OpFMnMx, OpFMul, OpFSetP, OpI2F, OpIAdd2, OpIMad, OpIMnMx, OpIMul, OpISetP,
    OpIpa, OpLdc, OpLop2, OpMov, OpMuFu, OpPBk, OpPCnt, OpPSetP, OpPhiDsts, OpPhiSrcs, OpRegOut,
    OpSel, OpShl, OpShr, OpTex, OpTxq, Phi, Pred, PredRef, PredSetOp, RegFile, SSARef, SSAValue,
    Shader, ShaderInfo, ShaderIoInfo, ShaderModelInfo, Src, SrcMod, SrcRef, SrcSwizzle,
    TexDerivMode, TexDim, TexLodMode, TexOffsetMode, TexQuery, TexRef,
};
use deko_nak::sph::PixelImap;
use std::collections::HashMap;

use crate::Error;

pub(crate) struct LoweredShader<'sm> {
    pub shader: Shader<'sm>,
    pub bindings: Vec<deko_dksh::Binding>,
}

#[derive(Default)]
struct ResourceMap {
    uniforms: HashMap<naga::Handle<naga::GlobalVariable>, u8>,
    textures: HashMap<naga::Handle<naga::GlobalVariable>, u16>,
    samplers: HashMap<naga::Handle<naga::GlobalVariable>, u16>,
    bindings: Vec<deko_dksh::Binding>,
}

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
) -> Result<LoweredShader<'sm>, Error> {
    let resources = resource_map(module)?;
    let shader = match entry.stage {
        naga::ShaderStage::Compute => lower_compute(module, entry, sm, &resources),
        naga::ShaderStage::Vertex => lower_vertex(module, entry, sm, &resources),
        naga::ShaderStage::Fragment => lower_fragment(module, entry, sm, &resources),
        stage => Err(Error::UnsupportedFeature(format!("{stage:?} stage"))),
    }?;
    Ok(LoweredShader {
        shader,
        bindings: resources.bindings,
    })
}

fn resource_map(module: &naga::Module) -> Result<ResourceMap, Error> {
    let mut resources = ResourceMap::default();
    let mut uniforms = module
        .global_variables
        .iter()
        .filter_map(|(handle, variable)| {
            (variable.space == naga::AddressSpace::Uniform)
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
    let mut textures = module
        .global_variables
        .iter()
        .filter_map(|(handle, variable)| {
            matches!(
                module.types[variable.ty].inner,
                naga::TypeInner::Image { .. }
            )
            .then_some((handle, variable.binding.as_ref()))
        })
        .collect::<Vec<_>>();
    textures.sort_by_key(|(_, binding)| binding.map(|binding| (binding.group, binding.binding)));
    for (target, (handle, binding)) in textures.into_iter().enumerate() {
        let binding = binding.ok_or_else(|| {
            Error::UnsupportedFeature("texture global without a resource binding".to_owned())
        })?;
        let target = u16::try_from(target)
            .map_err(|_| Error::UnsupportedFeature("too many sampled textures".to_owned()))?;
        if target >= 64 {
            return Err(Error::UnsupportedFeature(
                "more than 64 sampled textures".to_owned(),
            ));
        }
        resources.textures.insert(handle, target);
        resources.bindings.push(deko_dksh::Binding {
            group: binding.group,
            binding: binding.binding,
            target: u32::from(target),
            kind: deko_dksh::BindingKind::Texture,
        });
    }
    for (handle, variable) in module.global_variables.iter() {
        if !matches!(
            module.types[variable.ty].inner,
            naga::TypeInner::Sampler { .. }
        ) {
            continue;
        }
        let binding = variable.binding.as_ref().ok_or_else(|| {
            Error::UnsupportedFeature("sampler global without a resource binding".to_owned())
        })?;
        let target = resources
            .textures
            .iter()
            .find_map(|(texture, target)| {
                let texture_binding = module.global_variables[*texture].binding.as_ref()?;
                (texture_binding.group == binding.group
                    && texture_binding.binding.checked_add(1) == Some(binding.binding))
                .then_some(*target)
            })
            .ok_or_else(|| {
                Error::UnsupportedFeature(format!(
                    "sampler @group({}) @binding({}) is not paired after a texture",
                    binding.group, binding.binding
                ))
            })?;
        resources.samplers.insert(handle, target);
        resources.bindings.push(deko_dksh::Binding {
            group: binding.group,
            binding: binding.binding,
            target: u32::from(target),
            kind: deko_dksh::BindingKind::Sampler,
        });
    }
    Ok(resources)
}

fn lower_compute<'sm>(
    module: &naga::Module,
    entry: &naga::EntryPoint,
    sm: &'sm ShaderModelInfo,
    resources: &ResourceMap,
) -> Result<Shader<'sm>, Error> {
    if !entry.function.arguments.is_empty() || entry.function.result.is_some() {
        return Err(Error::UnsupportedFeature(
            "compute entry-point parameters or return values".to_owned(),
        ));
    }
    if entry.workgroup_size_overrides.is_some() {
        return Err(Error::UnsupportedFeature(
            "overridden compute workgroup sizes".to_owned(),
        ));
    }
    let [x, y, z] = entry.workgroup_size;
    let dimension = |value| {
        u16::try_from(value)
            .map_err(|_| Error::UnsupportedFeature("workgroup dimension exceeds u16".to_owned()))
    };
    let local_size = [dimension(x)?, dimension(y)?, dimension(z)?];

    let mut lowerer = FunctionLowerer::new(module, &entry.function, resources, Vec::new());
    let body = entry.function.body.clone();
    if lowerer.execute_statements(&body)?.is_some() {
        return Err(Error::UnsupportedFeature(
            "compute entry point returned a value".to_owned(),
        ));
    }
    Ok(Shader {
        sm,
        info: ShaderInfo::compute(local_size, 0),
        functions: vec![lowerer.finish()],
    })
}

#[derive(Clone)]
struct Value {
    components: Vec<Src>,
    kind: naga::ScalarKind,
}

type UniformPointer = (
    naga::Handle<naga::GlobalVariable>,
    naga::Handle<naga::Type>,
    u32,
    Option<Src>,
);

struct LoopContext {
    exit_label: Label,
    break_edges: Vec<LoopBreakEdge>,
}

struct LoopBreakEdge {
    block: usize,
    returned: Option<Value>,
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
    labels: LabelAllocator,
    loops: Vec<LoopContext>,
    loop_base_depth: usize,
    values: HashMap<naga::Handle<naga::Expression>, Value>,
    locals: HashMap<naga::Handle<naga::LocalVariable>, Value>,
    arguments: Vec<Value>,
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
            labels,
            loops: Vec::new(),
            loop_base_depth: 0,
            values: HashMap::new(),
            locals: HashMap::new(),
            arguments,
            early_returns: Vec::new(),
        }
    }

    fn emit(&mut self, instruction: Instr) {
        self.blocks[self.current_block].instrs.push(instruction);
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
            naga::Expression::Compose { components, .. } => {
                let mut flattened = Vec::new();
                let mut kind = None;
                for component in components {
                    let value = self.expression(*component)?;
                    if kind.is_some_and(|existing| existing != value.kind) {
                        return Err(Error::UnsupportedFeature(
                            "mixed scalar kinds in compose".to_owned(),
                        ));
                    }
                    kind = Some(value.kind);
                    flattened.extend(value.components);
                }
                Value {
                    components: flattened,
                    kind: kind
                        .ok_or_else(|| Error::UnsupportedFeature("empty compose".to_owned()))?,
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
                } else {
                    self.load_uniform(*pointer)?
                }
            }
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
            naga::Expression::ImageQuery { image, query } => self.image_query(*image, *query)?,
            naga::Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => self.math(*fun, *arg, *arg1, *arg2, *arg3)?,
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

    fn image_query(
        &mut self,
        image: naga::Handle<naga::Expression>,
        query: naga::ImageQuery,
    ) -> Result<Value, Error> {
        let image = self.global_expression(image, "texture query")?;
        let target = *self.resources.textures.get(&image).ok_or_else(|| {
            Error::UnsupportedFeature("queried texture has no Deko target".to_owned())
        })?;
        let (dimension, _, _) = module_image_type(self.module, image)?;
        let (level, components) = match query {
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
                (level, components)
            }
            query => {
                return Err(Error::UnsupportedFeature(format!(
                    "texture query {query:?}"
                )));
            }
        };
        let source = self.materialize(level)?;
        let destination = self.target.ssa_alloc.alloc_vec(RegFile::GPR, components);
        self.emit(Instr::new(OpTxq {
            dsts: [Dst::from(destination.clone()), Dst::None],
            tex: TexRef::Bound(target),
            src: Src::from(source),
            query: TexQuery::Dimension,
            nodep: false,
            channel_mask: ChannelMask::for_comps(components),
        }));
        Ok(Value {
            components: destination.iter().copied().map(Src::from).collect(),
            kind: naga::ScalarKind::Uint,
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
        if gather.is_some()
            || offset.is_some()
            || clamp_to_edge
            || matches!(level, naga::SampleLevel::Gradient { .. })
        {
            return Err(Error::UnsupportedFeature(format!(
                "texture sample options gather={gather:?} array={} offset={} level={level:?} depth={} clamp={clamp_to_edge}",
                array_index.is_some(),
                offset.is_some(),
                depth_ref.is_some()
            )));
        }
        let image = self.global_expression(image, "texture")?;
        let sampler = self.global_expression(sampler, "sampler")?;
        let target = *self.resources.textures.get(&image).ok_or_else(|| {
            Error::UnsupportedFeature("sampled texture has no Deko target".to_owned())
        })?;
        if self.resources.samplers.get(&sampler) != Some(&target) {
            return Err(Error::UnsupportedFeature(
                "texture and sampler do not share a Deko target".to_owned(),
            ));
        }
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
        if kind != naga::ScalarKind::Float {
            return Err(Error::UnsupportedFeature(format!(
                "sampled texture scalar kind {kind:?}"
            )));
        }
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
        if let Some(array_index) = array_index {
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
            coordinate.components.extend(array_index.components);
        }
        if let Some(depth_ref) = depth_ref {
            let depth_ref = self.expression(depth_ref)?;
            if depth_ref.kind != naga::ScalarKind::Float || depth_ref.components.len() != 1 {
                return Err(Error::UnsupportedFeature(
                    "texture depth reference must be a float scalar".to_owned(),
                ));
            }
            coordinate.components.extend(depth_ref.components);
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
                coordinate.components.extend(level.components);
                TexLodMode::Bias
            }
            naga::SampleLevel::Exact(level) => {
                let level = self.expression(level)?;
                if level.kind != naga::ScalarKind::Float || level.components.len() != 1 {
                    return Err(Error::UnsupportedFeature(
                        "texture LOD must be a float scalar".to_owned(),
                    ));
                }
                coordinate.components.extend(level.components);
                TexLodMode::Lod
            }
            naga::SampleLevel::Gradient { .. } => unreachable!("rejected above"),
        };
        let coordinate = self.materialize(coordinate)?;
        let output_components = if depth_ref.is_some() { 1 } else { 4 };
        let dst = self
            .target
            .ssa_alloc
            .alloc_vec(RegFile::GPR, output_components);
        self.emit(Instr::new(OpTex {
            dsts: [Dst::from(dst.clone()), Dst::None],
            fault: Dst::None,
            tex: TexRef::Bound(target),
            srcs: [Src::from(coordinate), Src::ZERO],
            dim,
            lod_mode,
            deriv_mode: TexDerivMode::Auto,
            z_cmpr: depth_ref.is_some(),
            offset_mode: TexOffsetMode::None,
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

    fn global_expression(
        &self,
        handle: naga::Handle<naga::Expression>,
        description: &str,
    ) -> Result<naga::Handle<naga::GlobalVariable>, Error> {
        match self.source.expressions[handle] {
            naga::Expression::GlobalVariable(global) => Ok(global),
            ref expression => Err(Error::UnsupportedFeature(format!(
                "{description} expression {expression:?}"
            ))),
        }
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
        if arg3.is_some() {
            return Err(Error::UnsupportedFeature(format!(
                "four-argument math function {fun:?}"
            )));
        }
        let matrix_shape = self.expression_matrix_shape(arg);
        let value = self.expression(arg)?;
        match fun {
            naga::MathFunction::Abs if value.kind == naga::ScalarKind::Float => Ok(Value {
                components: value.components.into_iter().map(Src::fabs).collect(),
                kind: value.kind,
            }),
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
                let value = self.float_minmax(&value, &low, false)?;
                self.float_minmax(&value, &high, true)
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
            return Err(Error::UnsupportedFeature(
                "dot operands must be equally-sized float vectors".to_owned(),
            ));
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
                    naga::TypeInner::Array { base, .. } => Some(base),
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
                    naga::TypeInner::Array { base, .. } => Some(base),
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
            _ => None,
        }
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
                Error::UnsupportedFeature("swizzle component out of bounds".to_owned())
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
                if variable.space != naga::AddressSpace::Uniform {
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
        let target = *self.resources.uniforms.get(&global).ok_or_else(|| {
            Error::UnsupportedFeature("uniform has no allocated Deko slot".to_owned())
        })?;
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
            naga::Expression::AccessIndex { base, .. } => self.pointer_is_local(base),
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
        if left.kind != right.kind {
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
        self.execute_statements(&body)?
            .ok_or_else(|| Error::UnsupportedFeature("missing return value".to_owned()))
    }

    fn call_argument(&mut self, argument: naga::Handle<naga::Expression>) -> Result<Value, Error> {
        if self.pointer_is_argument(argument) {
            self.load_argument_pointer(argument)
        } else if self.pointer_is_local(argument) {
            self.load_local_pointer(argument)
        } else {
            self.expression(argument)
        }
    }

    fn block_is_single_break(block: &naga::Block) -> bool {
        let mut statements = block.iter();
        matches!(statements.next(), Some(naga::Statement::Break)) && statements.next().is_none()
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
        let branch_block = self.current_block;
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
        let exit_label = self.allocate_label();
        let preheader = self.current_block;

        let local_handles = self
            .source
            .local_variables
            .iter()
            .map(|(handle, _)| handle)
            .collect::<Vec<_>>();
        let mut preheader_sources = OpPhiSrcs::new();
        let mut header_destinations = OpPhiDsts::new();
        let mut loop_phis = Vec::new();
        for local in local_handles {
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
            target: header_label,
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
            break_edges: Vec::new(),
        });
        if self.execute_statements(body)?.is_some() {
            return Err(Error::UnsupportedFeature(
                "return from inside a loop".to_owned(),
            ));
        }
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
        self.merge_loop_returns(&context)?;
        for phi in loop_phis {
            self.locals.insert(phi.local, phi.header_value);
        }
        Ok(())
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
        self.early_returns
            .push((flag.components[0].clone(), returned));
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
                    return Ok(Some(self.finalize_return(value)?));
                }
                naga::Statement::Call {
                    function,
                    arguments,
                    result: Some(call_result),
                } => {
                    let arguments = arguments
                        .iter()
                        .map(|argument| self.call_argument(*argument))
                        .collect::<Result<Vec<_>, _>>()?;
                    let value = self.inline_call(*function, arguments)?;
                    self.values.insert(*call_result, value);
                }
                naga::Statement::Store { pointer, value } => {
                    if !self.pointer_is_local(*pointer) {
                        return Err(Error::UnsupportedFeature(format!(
                            "store pointer {:?}",
                            self.source.expressions[*pointer]
                        )));
                    }
                    let (local, ty, offset) = self.local_pointer(*pointer)?;
                    let mut value = self.expression(*value)?;
                    let expected = flat_type_components(self.module, ty)?;
                    if value.components.len() != expected {
                        return Err(Error::UnsupportedFeature(format!(
                            "local store shape mismatch: expected {expected}, got {} for pointer {:?}",
                            value.components.len(),
                            self.source.expressions[*pointer]
                        )));
                    }
                    let mut local_value = self.local_value(local)?;
                    if value.kind == naga::ScalarKind::Bool
                        && local_value.kind != naga::ScalarKind::Bool
                    {
                        value.components = self
                            .materialize_loop_components(&value)?
                            .into_iter()
                            .map(Src::from)
                            .collect();
                    }
                    let end = offset + expected;
                    local_value.components[offset..end].clone_from_slice(&value.components);
                    self.locals.insert(local, local_value);
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
                        && Self::block_is_single_break(reject)
                    {
                        self.emit_conditional_break(&condition, false, None)?;
                        continue;
                    }
                    if !self.loops.is_empty()
                        && reject.is_empty()
                        && Self::block_is_single_break(accept)
                    {
                        self.emit_conditional_break(&condition, true, None)?;
                        continue;
                    }
                    if let Some(value) = self.conditional(&condition, accept, reject)? {
                        return Ok(Some(self.finalize_return(value)?));
                    }
                }
                naga::Statement::Loop {
                    body,
                    continuing,
                    break_if,
                } => self.lower_loop(body, continuing, *break_if)?,
                naga::Statement::Emit(range) => {
                    for expression in range.clone() {
                        if self.pointer_type(expression).is_some()
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
                naga::Statement::Return { value: None } => {}
                other => {
                    return Err(Error::UnsupportedFeature(format!("statement {other:?}")));
                }
            }
        }
        Ok(None)
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
        let locals = self.locals.clone();
        let accepted = self.execute_statements(accept)?;
        let accepted_locals = self.locals.clone();
        self.locals.clone_from(&locals);
        let rejected = self.execute_statements(reject)?;
        let rejected_locals = self.locals.clone();
        match (accepted, rejected) {
            (Some(accepted), Some(rejected)) => {
                Ok(Some(self.select(condition, accepted, rejected)?))
            }
            (Some(accepted), None) => {
                self.early_returns
                    .push((condition.components[0].clone(), accepted));
                self.locals = rejected_locals;
                Ok(None)
            }
            (None, Some(rejected)) => {
                self.early_returns
                    .push((condition.components[0].clone().bnot(), rejected));
                self.locals = accepted_locals;
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
                Ok(None)
            }
        }
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

    fn inline_call(
        &mut self,
        function: naga::Handle<naga::Function>,
        arguments: Vec<Value>,
    ) -> Result<Value, Error> {
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
            labels: std::mem::replace(&mut self.labels, LabelAllocator::new()),
            loops: std::mem::take(&mut self.loops),
            loop_base_depth,
            values: HashMap::new(),
            locals: HashMap::new(),
            arguments,
            early_returns: Vec::new(),
        };
        let result = callee.return_value().map_err(|error| match error {
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
        for (dst, src) in ssa.iter().zip(value.components) {
            self.emit(Instr::new(OpMov {
                dst: Dst::from(*dst),
                src,
                quad_lanes: 0xf,
            }));
        }
        Ok(ssa)
    }

    fn materialize_components(&mut self, value: Value) -> Vec<SSAValue> {
        value
            .components
            .into_iter()
            .map(|source| {
                let destination = self.target.ssa_alloc.alloc(RegFile::GPR);
                self.emit(Instr::new(OpMov {
                    dst: Dst::from(destination),
                    src: source,
                    quad_lanes: 0xf,
                }));
                destination
            })
            .collect()
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
    let variable = &module.global_variables[global];
    let naga::TypeInner::Image {
        dim,
        arrayed,
        class,
    } = module.types[variable.ty].inner
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
        naga::TypeInner::Scalar(scalar) if scalar.width == 4 => Ok((vec![base], scalar.kind)),
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

fn bind_vertex_arguments(
    module: &naga::Module,
    lowerer: &mut FunctionLowerer<'_>,
    info: &mut ShaderInfo,
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
            let field = bind_vertex_field(module, lowerer, info, &binding, ty)?;
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
) -> Result<Value, Error> {
    let (components, kind) = type_shape(module, ty)?;
    if kind == naga::ScalarKind::Bool {
        return Err(Error::UnsupportedFeature("boolean vertex input".to_owned()));
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
            let value = bind_fragment_field(module, lowerer, info, &binding, ty, used)?;
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
) -> Result<Value, Error> {
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
) -> Result<Value, Error> {
    let (components, kind) = type_shape(module, ty)?;
    if let naga::Binding::BuiltIn(builtin) = binding {
        return bind_fragment_builtin(lowerer, info, *builtin, components, kind, used);
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
) -> Result<Shader<'sm>, Error> {
    let result = entry.function.result.as_ref().ok_or_else(|| {
        Error::UnsupportedFeature("vertex entry point without a result".to_owned())
    })?;
    let mut info = ShaderInfo::vertex();
    let mut lowerer = FunctionLowerer::new(module, &entry.function, resources, Vec::new());
    bind_vertex_arguments(module, &mut lowerer, &mut info)?;
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
            binding => {
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
) -> Result<Shader<'sm>, Error> {
    let result = entry.function.result.as_ref().ok_or_else(|| {
        Error::UnsupportedFeature("fragment entry point without a result".to_owned())
    })?;
    let mut info = ShaderInfo::fragment(entry.early_depth_test.is_some(), false, false);
    let mut lowerer = FunctionLowerer::new(module, &entry.function, resources, Vec::new());
    bind_fragment_arguments(module, &mut lowerer, &mut info)?;
    let value = lowerer.return_value()?;
    let mut outputs = Vec::new();
    let mut writes_color = 0_u32;
    for field in output_fields(module, result)? {
        let naga::Binding::Location { location, .. } = field.binding else {
            return Err(Error::UnsupportedFeature(format!(
                "fragment result binding {:?}",
                field.binding
            )));
        };
        if location >= 8 || field.range.len() > 4 || field.range.end > value.components.len() {
            return Err(Error::UnsupportedFeature(
                "fragment color output shape".to_owned(),
            ));
        }
        let field_value = Value {
            components: value.components[field.range].to_vec(),
            kind: value.kind,
        };
        let ssa = lowerer.materialize(field_value)?;
        let mut target = ssa.iter().copied().map(Src::from).collect::<Vec<_>>();
        writes_color |= ((1_u32 << target.len()) - 1) << (location * 4);
        target.resize(4, Src::ZERO);
        outputs.extend(target);
    }
    lowerer.emit(Instr::new(OpRegOut { srcs: outputs }));
    let ShaderIoInfo::Fragment(io) = &mut info.io else {
        unreachable!();
    };
    io.writes_color = writes_color;
    Ok(Shader {
        sm,
        info,
        functions: vec![lowerer.finish()],
    })
}
