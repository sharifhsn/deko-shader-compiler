use deko_nak::ir::{
    CBuf, CBufRef, Dst, FRndMode, Function, Instr, InterpFreq, InterpLoc, LdcMode, MemType, OpALd,
    OpASt, OpExit, OpFAdd, OpFMul, OpIpa, OpLdc, OpMov, OpRegOut, RegFile, SSARef, Shader,
    ShaderInfo, ShaderIoInfo, ShaderModelInfo, Src,
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

struct FunctionLowerer<'function> {
    module: &'function naga::Module,
    source: &'function naga::Function,
    resources: &'function ResourceMap,
    target: Function,
    values: HashMap<naga::Handle<naga::Expression>, Value>,
    locals: HashMap<naga::Handle<naga::LocalVariable>, Value>,
    arguments: Vec<Value>,
}

impl<'function> FunctionLowerer<'function> {
    fn new(
        module: &'function naga::Module,
        source: &'function naga::Function,
        resources: &'function ResourceMap,
        arguments: Vec<Value>,
    ) -> Self {
        Self {
            module,
            source,
            resources,
            target: Function::single_block(Vec::new()),
            values: HashMap::new(),
            locals: HashMap::new(),
            arguments,
        }
    }

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
                if self.pointer_is_local(*pointer) {
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
            naga::Expression::AccessIndex { base, index } if self.expression_is_vector(*base) => {
                let base = self.expression(*base)?;
                Value {
                    components: vec![base.components.get(*index as usize).cloned().ok_or_else(
                        || Error::UnsupportedFeature("vector index out of bounds".to_owned()),
                    )?],
                    kind: base.kind,
                }
            }
            naga::Expression::Unary { op, expr } => self.unary(*op, *expr)?,
            naga::Expression::Binary { op, left, right } => {
                let left_matrix = self.expression_matrix_shape(*left);
                let left = self.expression(*left)?;
                let right = self.expression(*right)?;
                self.binary(*op, &left, &right, left_matrix)?
            }
            expression => {
                return Err(Error::UnsupportedFeature(format!(
                    "expression {expression:?}"
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
        Err(Error::UnsupportedFeature(format!(
            "unary operator {op:?} for {:?}",
            value.kind
        )))
    }

    fn expression_is_vector(&self, handle: naga::Handle<naga::Expression>) -> bool {
        match &self.source.expressions[handle] {
            naga::Expression::FunctionArgument(index) => self
                .source
                .arguments
                .get(*index as usize)
                .is_some_and(|argument| {
                    matches!(
                        self.module.types[argument.ty].inner,
                        naga::TypeInner::Vector { .. }
                    )
                }),
            naga::Expression::Compose { ty, .. } | naga::Expression::ZeroValue(ty) => {
                matches!(self.module.types[*ty].inner, naga::TypeInner::Vector { .. })
            }
            naga::Expression::Splat { .. } | naga::Expression::Swizzle { .. } => true,
            naga::Expression::Binary { left, .. } => self.expression_is_vector(*left),
            naga::Expression::Unary { expr, .. } => self.expression_is_vector(*expr),
            naga::Expression::Load { pointer } => {
                self.uniform_pointer(*pointer).is_ok_and(|(_, ty, _)| {
                    matches!(self.module.types[ty].inner, naga::TypeInner::Vector { .. })
                })
            }
            _ => false,
        }
    }

    fn expression_matrix_shape(
        &self,
        handle: naga::Handle<naga::Expression>,
    ) -> Option<(usize, usize)> {
        let ty = match self.source.expressions[handle] {
            naga::Expression::Compose { ty, .. } | naga::Expression::ZeroValue(ty) => ty,
            naga::Expression::FunctionArgument(index) => {
                self.source.arguments.get(index as usize)?.ty
            }
            naga::Expression::Load { pointer } => self.uniform_pointer(pointer).ok()?.1,
            _ => return None,
        };
        let naga::TypeInner::Matrix { columns, rows, .. } = self.module.types[ty].inner else {
            return None;
        };
        Some((vector_size(columns), vector_size(rows)))
    }

    fn uniform_pointer(
        &self,
        handle: naga::Handle<naga::Expression>,
    ) -> Result<
        (
            naga::Handle<naga::GlobalVariable>,
            naga::Handle<naga::Type>,
            u32,
        ),
        Error,
    > {
        match self.source.expressions[handle] {
            naga::Expression::GlobalVariable(global) => {
                let variable = &self.module.global_variables[global];
                if variable.space != naga::AddressSpace::Uniform {
                    return Err(Error::UnsupportedFeature(format!(
                        "load from {:?} address space",
                        variable.space
                    )));
                }
                Ok((global, variable.ty, 0))
            }
            naga::Expression::AccessIndex { base, index } => {
                let (global, ty, offset) = self.uniform_pointer(base)?;
                let naga::TypeInner::Struct { ref members, .. } = self.module.types[ty].inner
                else {
                    return Err(Error::UnsupportedFeature(
                        "uniform access into a non-struct value".to_owned(),
                    ));
                };
                let member = members.get(index as usize).ok_or_else(|| {
                    Error::UnsupportedFeature("uniform member index out of bounds".to_owned())
                })?;
                let offset = offset.checked_add(member.offset).ok_or_else(|| {
                    Error::UnsupportedFeature("uniform offset overflow".to_owned())
                })?;
                Ok((global, member.ty, offset))
            }
            ref pointer => Err(Error::UnsupportedFeature(format!(
                "load pointer {pointer:?}"
            ))),
        }
    }

    fn load_uniform(&mut self, pointer: naga::Handle<naga::Expression>) -> Result<Value, Error> {
        let (global, ty, base_offset) = self.uniform_pointer(pointer)?;
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
            self.target.blocks[0].instrs.push(Instr::new(OpLdc {
                dst: Dst::from(dst),
                cb: Src::from(CBufRef {
                    buf: CBuf::Binding(target),
                    offset,
                }),
                offset: Src::ZERO,
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
                let naga::TypeInner::Struct { ref members, .. } = self.module.types[ty].inner
                else {
                    return Err(Error::UnsupportedFeature(
                        "local access into a non-struct value".to_owned(),
                    ));
                };
                let member = members.get(index as usize).ok_or_else(|| {
                    Error::UnsupportedFeature("local member index out of bounds".to_owned())
                })?;
                let mut preceding = 0_usize;
                for previous in &members[..index as usize] {
                    preceding = preceding
                        .checked_add(flat_type_components(self.module, previous.ty)?)
                        .ok_or_else(|| {
                            Error::UnsupportedFeature("local component offset overflow".to_owned())
                        })?;
                }
                Ok((local, member.ty, component_offset + preceding))
            }
            ref pointer => Err(Error::UnsupportedFeature(format!(
                "local pointer {pointer:?}"
            ))),
        }
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
        Ok(Value {
            components: value
                .components
                .get(offset..end)
                .ok_or_else(|| Error::UnsupportedFeature("local value shape mismatch".to_owned()))?
                .to_vec(),
            kind: flat_type_kind(self.module, ty)?,
        })
    }

    fn binary(
        &mut self,
        op: naga::BinaryOperator,
        left: &Value,
        right: &Value,
        left_matrix: Option<(usize, usize)>,
    ) -> Result<Value, Error> {
        if left.kind != right.kind || left.kind != naga::ScalarKind::Float {
            return Err(Error::UnsupportedFeature(format!(
                "{op:?} for {:?} and {:?}",
                left.kind, right.kind
            )));
        }
        if op == naga::BinaryOperator::Multiply {
            if let Some((columns, rows)) = left_matrix {
                return self.multiply_matrix_vector(left, right, columns, rows);
            }
        }
        let width = left.components.len().max(right.components.len());
        if (left.components.len() != 1 && left.components.len() != width)
            || (right.components.len() != 1 && right.components.len() != width)
        {
            return Err(Error::UnsupportedFeature(
                "binary operands with incompatible widths".to_owned(),
            ));
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
            let dst = self.target.ssa_alloc.alloc(RegFile::GPR);
            let instruction = match op {
                naga::BinaryOperator::Add => Instr::new(OpFAdd {
                    dst: Dst::from(dst),
                    srcs: [lhs, rhs],
                    saturate: false,
                    rnd_mode: FRndMode::NearestEven,
                    ftz: false,
                }),
                naga::BinaryOperator::Subtract => Instr::new(OpFAdd {
                    dst: Dst::from(dst),
                    srcs: [lhs, rhs.fneg()],
                    saturate: false,
                    rnd_mode: FRndMode::NearestEven,
                    ftz: false,
                }),
                naga::BinaryOperator::Multiply => Instr::new(OpFMul {
                    dst: Dst::from(dst),
                    srcs: [lhs, rhs],
                    saturate: false,
                    rnd_mode: FRndMode::NearestEven,
                    ftz: false,
                    dnz: false,
                }),
                _ => {
                    return Err(Error::UnsupportedFeature(format!("binary operator {op:?}")));
                }
            };
            self.target.blocks[0].instrs.push(instruction);
            components.push(Src::from(dst));
        }
        Ok(Value {
            components,
            kind: left.kind,
        })
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
            return Err(Error::UnsupportedFeature(
                "matrix-vector multiply shape".to_owned(),
            ));
        }
        let mut result = Vec::with_capacity(rows);
        for row in 0..rows {
            let mut accumulator = None;
            for column in 0..columns {
                let product = self.target.ssa_alloc.alloc(RegFile::GPR);
                self.target.blocks[0].instrs.push(Instr::new(OpFMul {
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
                        self.target.blocks[0].instrs.push(Instr::new(OpFAdd {
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

    fn return_value(&mut self) -> Result<Value, Error> {
        let body = self.source.body.clone();
        self.execute_statements(&body)?
            .ok_or_else(|| Error::UnsupportedFeature("missing return value".to_owned()))
    }

    fn execute_statements(&mut self, body: &naga::Block) -> Result<Option<Value>, Error> {
        for statement in body {
            match statement {
                naga::Statement::Return {
                    value: Some(handle),
                } => return Ok(Some(self.expression(*handle)?)),
                naga::Statement::Call {
                    function,
                    arguments,
                    result: Some(call_result),
                } => {
                    let arguments = arguments
                        .iter()
                        .map(|argument| self.expression(*argument))
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
                    let value = self.expression(*value)?;
                    let expected = flat_type_components(self.module, ty)?;
                    if value.components.len() != expected {
                        return Err(Error::UnsupportedFeature(
                            "local store shape mismatch".to_owned(),
                        ));
                    }
                    let mut local_value = self.local_value(local)?;
                    let end = offset + expected;
                    local_value.components[offset..end].clone_from_slice(&value.components);
                    self.locals.insert(local, local_value);
                }
                naga::Statement::Block(block) => {
                    if let Some(value) = self.execute_statements(block)? {
                        return Ok(Some(value));
                    }
                }
                naga::Statement::Emit(_) | naga::Statement::Return { value: None } => {}
                other => {
                    return Err(Error::UnsupportedFeature(format!("statement {other:?}")));
                }
            }
        }
        Ok(None)
    }

    fn inline_call(
        &mut self,
        function: naga::Handle<naga::Function>,
        arguments: Vec<Value>,
    ) -> Result<Value, Error> {
        let target = std::mem::replace(&mut self.target, Function::single_block(Vec::new()));
        let mut callee = Self {
            module: self.module,
            source: &self.module.functions[function],
            resources: self.resources,
            target,
            values: HashMap::new(),
            locals: HashMap::new(),
            arguments,
        };
        let result = callee.return_value();
        self.target = callee.target;
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
            self.target.blocks[0].instrs.push(Instr::new(OpMov {
                dst: Dst::from(*dst),
                src,
                quad_lanes: 0xf,
            }));
        }
        Ok(ssa)
    }

    fn finish(mut self) -> Function {
        self.target.blocks[0].instrs.push(Instr::new(OpExit {}));
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
    for argument in &lowerer.source.arguments {
        let Some(naga::Binding::Location { location, .. }) = argument.binding.as_ref() else {
            return Err(Error::UnsupportedFeature(format!(
                "vertex argument binding {:?}",
                argument.binding
            )));
        };
        let (components, kind) = type_shape(module, argument.ty)?;
        if kind == naga::ScalarKind::Bool {
            return Err(Error::UnsupportedFeature(
                "boolean vertex attributes".to_owned(),
            ));
        }
        let addr = location_address(*location)?;
        let ssa = lowerer.target.ssa_alloc.alloc_vec(RegFile::GPR, components);
        lowerer.target.blocks[0].instrs.push(Instr::new(OpALd {
            dst: Dst::from(ssa.clone()),
            vtx: Src::ZERO,
            offset: Src::ZERO,
            addr,
            comps: components,
            patch: false,
            output: false,
            phys: false,
        }));
        lowerer.arguments.push(Value {
            components: ssa.iter().copied().map(Src::from).collect(),
            kind,
        });
        let ShaderIoInfo::Vtg(io) = &mut info.io else {
            unreachable!();
        };
        io.mark_attrs_read(addr..addr + u16::from(components) * 4);
    }
    Ok(())
}

fn bind_fragment_arguments(
    module: &naga::Module,
    lowerer: &mut FunctionLowerer<'_>,
    info: &mut ShaderInfo,
) -> Result<(), Error> {
    for argument in &lowerer.source.arguments {
        let Some(naga::Binding::Location {
            location,
            interpolation,
            sampling,
            ..
        }) = argument.binding.as_ref()
        else {
            return Err(Error::UnsupportedFeature(format!(
                "fragment argument binding {:?}",
                argument.binding
            )));
        };
        let (components, kind) = type_shape(module, argument.ty)?;
        if kind != naga::ScalarKind::Float {
            return Err(Error::UnsupportedFeature(
                "non-float fragment varyings".to_owned(),
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
            lowerer.target.blocks[0].instrs.push(Instr::new(OpIpa {
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
        lowerer.arguments.push(Value {
            components: value,
            kind,
        });
    }
    Ok(())
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
        lowerer.target.blocks[0].instrs.push(Instr::new(OpASt {
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
    lowerer.target.blocks[0]
        .instrs
        .push(Instr::new(OpRegOut { srcs: outputs }));
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
