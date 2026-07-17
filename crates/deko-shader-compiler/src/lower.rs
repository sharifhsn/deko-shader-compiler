use deko_nak::ir::{
    Dst, FRndMode, Function, Instr, InterpFreq, InterpLoc, OpALd, OpASt, OpExit, OpFAdd, OpFMul,
    OpIpa, OpMov, OpRegOut, RegFile, SSARef, Shader, ShaderInfo, ShaderIoInfo, ShaderModelInfo,
    Src,
};
use deko_nak::sph::PixelImap;
use std::collections::HashMap;

use crate::Error;

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
) -> Result<Shader<'sm>, Error> {
    match entry.stage {
        naga::ShaderStage::Compute => lower_compute(entry, sm),
        naga::ShaderStage::Vertex => lower_vertex(module, entry, sm),
        naga::ShaderStage::Fragment => lower_fragment(module, entry, sm),
        stage => Err(Error::UnsupportedFeature(format!("{stage:?} stage"))),
    }
}

fn lower_compute<'sm>(
    entry: &naga::EntryPoint,
    sm: &'sm ShaderModelInfo,
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
    for statement in &entry.function.body {
        match statement {
            naga::Statement::Return { value: None } => {}
            naga::Statement::Emit(range) if range.clone().next().is_none() => {}
            other => {
                return Err(Error::UnsupportedFeature(format!(
                    "compute statement {other:?}"
                )));
            }
        }
    }

    let [x, y, z] = entry.workgroup_size;
    let dimension = |value| {
        u16::try_from(value)
            .map_err(|_| Error::UnsupportedFeature("workgroup dimension exceeds u16".to_owned()))
    };
    let local_size = [dimension(x)?, dimension(y)?, dimension(z)?];

    Ok(Shader {
        sm,
        info: ShaderInfo::compute(local_size, 0),
        functions: vec![Function::single_block(vec![Instr::new(OpExit {})])],
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
    target: Function,
    values: HashMap<naga::Handle<naga::Expression>, Value>,
    arguments: Vec<Value>,
}

impl<'function> FunctionLowerer<'function> {
    fn new(
        module: &'function naga::Module,
        source: &'function naga::Function,
        arguments: Vec<Value>,
    ) -> Self {
        Self {
            module,
            source,
            target: Function::single_block(Vec::new()),
            values: HashMap::new(),
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
                let left = self.expression(*left)?;
                let right = self.expression(*right)?;
                self.binary(*op, &left, &right)?
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
            _ => false,
        }
    }

    fn binary(
        &mut self,
        op: naga::BinaryOperator,
        left: &Value,
        right: &Value,
    ) -> Result<Value, Error> {
        if left.kind != right.kind || left.kind != naga::ScalarKind::Float {
            return Err(Error::UnsupportedFeature(format!(
                "{op:?} for {:?} and {:?}",
                left.kind, right.kind
            )));
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

    fn return_value(&mut self) -> Result<Value, Error> {
        let mut result = None;
        let body = self.source.body.clone();
        for statement in &body {
            match statement {
                naga::Statement::Return {
                    value: Some(handle),
                } if result.is_none() => {
                    result = Some(self.expression(*handle)?);
                }
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
                naga::Statement::Emit(_) | naga::Statement::Return { value: None } => {}
                other => {
                    return Err(Error::UnsupportedFeature(format!("statement {other:?}")));
                }
            }
        }
        result.ok_or_else(|| Error::UnsupportedFeature("missing return value".to_owned()))
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
            target,
            values: HashMap::new(),
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
    let (components, kind) = type_shape(module, ty)?;
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
        components: vec![source; usize::from(components)],
        kind,
    })
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
) -> Result<Shader<'sm>, Error> {
    let result = entry.function.result.as_ref().ok_or_else(|| {
        Error::UnsupportedFeature("vertex entry point without a result".to_owned())
    })?;
    let mut info = ShaderInfo::vertex();
    let mut lowerer = FunctionLowerer::new(module, &entry.function, Vec::new());
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
) -> Result<Shader<'sm>, Error> {
    let result = entry.function.result.as_ref().ok_or_else(|| {
        Error::UnsupportedFeature("fragment entry point without a result".to_owned())
    })?;
    let mut info = ShaderInfo::fragment(entry.early_depth_test.is_some(), false, false);
    let mut lowerer = FunctionLowerer::new(module, &entry.function, Vec::new());
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
