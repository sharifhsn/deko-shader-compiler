use deko_nak::ir::{
    Dst, Function, Instr, OpASt, OpExit, OpMov, OpRegOut, RegFile, SSARef, Shader, ShaderInfo,
    ShaderIoInfo, ShaderModelInfo, Src,
};
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
    entry: &naga::EntryPoint,
    sm: &'sm ShaderModelInfo,
) -> Result<Shader<'sm>, Error> {
    match entry.stage {
        naga::ShaderStage::Compute => lower_compute(entry, sm),
        naga::ShaderStage::Vertex => lower_vertex(entry, sm),
        naga::ShaderStage::Fragment => lower_fragment(entry, sm),
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
}

struct FunctionLowerer<'function> {
    source: &'function naga::Function,
    target: Function,
    values: HashMap<naga::Handle<naga::Expression>, Value>,
}

impl<'function> FunctionLowerer<'function> {
    fn new(source: &'function naga::Function) -> Self {
        Self {
            source,
            target: Function::single_block(Vec::new()),
            values: HashMap::new(),
        }
    }

    fn expression(&mut self, handle: naga::Handle<naga::Expression>) -> Result<Value, Error> {
        if let Some(value) = self.values.get(&handle) {
            return Ok(value.clone());
        }
        let value = match &self.source.expressions[handle] {
            naga::Expression::Literal(literal) => Value {
                components: vec![literal_source(*literal)?],
            },
            naga::Expression::Compose { components, .. } => {
                let mut flattened = Vec::new();
                for component in components {
                    flattened.extend(self.expression(*component)?.components);
                }
                Value {
                    components: flattened,
                }
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

    fn return_value(&mut self) -> Result<Value, Error> {
        let mut result = None;
        for statement in &self.source.body {
            match statement {
                naga::Statement::Return {
                    value: Some(handle),
                } if result.is_none() => {
                    result = Some(self.expression(*handle)?);
                }
                naga::Statement::Emit(_) | naga::Statement::Return { value: None } => {}
                other => {
                    return Err(Error::UnsupportedFeature(format!("statement {other:?}")));
                }
            }
        }
        result.ok_or_else(|| Error::UnsupportedFeature("missing return value".to_owned()))
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

fn lower_vertex<'sm>(
    entry: &naga::EntryPoint,
    sm: &'sm ShaderModelInfo,
) -> Result<Shader<'sm>, Error> {
    if !entry.function.arguments.is_empty() {
        return Err(Error::UnsupportedFeature(
            "vertex entry-point arguments".to_owned(),
        ));
    }
    let result = entry.function.result.as_ref().ok_or_else(|| {
        Error::UnsupportedFeature("vertex entry point without a result".to_owned())
    })?;
    if !matches!(
        result.binding,
        Some(naga::Binding::BuiltIn(naga::BuiltIn::Position { .. }))
    ) {
        return Err(Error::UnsupportedFeature(
            "vertex result other than @builtin(position)".to_owned(),
        ));
    }

    let mut lowerer = FunctionLowerer::new(&entry.function);
    let value = lowerer.return_value()?;
    if value.components.len() != 4 {
        return Err(Error::UnsupportedFeature(
            "vertex position must contain four components".to_owned(),
        ));
    }
    let ssa = lowerer.materialize(value)?;
    lowerer.target.blocks[0].instrs.push(Instr::new(OpASt {
        vtx: Src::ZERO,
        offset: Src::ZERO,
        data: Src::from(ssa),
        addr: 0x70,
        comps: 4,
        patch: false,
        phys: false,
    }));

    let mut info = ShaderInfo::vertex();
    let ShaderIoInfo::Vtg(io) = &mut info.io else {
        unreachable!();
    };
    io.mark_attrs_written(0x70..0x80);
    io.mark_store_req(0x70..0x80);
    Ok(Shader {
        sm,
        info,
        functions: vec![lowerer.finish()],
    })
}

fn lower_fragment<'sm>(
    entry: &naga::EntryPoint,
    sm: &'sm ShaderModelInfo,
) -> Result<Shader<'sm>, Error> {
    if !entry.function.arguments.is_empty() {
        return Err(Error::UnsupportedFeature(
            "fragment entry-point arguments".to_owned(),
        ));
    }
    let result = entry.function.result.as_ref().ok_or_else(|| {
        Error::UnsupportedFeature("fragment entry point without a result".to_owned())
    })?;
    if !matches!(
        result.binding,
        Some(naga::Binding::Location { location: 0, .. })
    ) {
        return Err(Error::UnsupportedFeature(
            "fragment result other than @location(0)".to_owned(),
        ));
    }

    let mut lowerer = FunctionLowerer::new(&entry.function);
    let value = lowerer.return_value()?;
    if value.components.len() != 4 {
        return Err(Error::UnsupportedFeature(
            "fragment color must contain four components".to_owned(),
        ));
    }
    let ssa = lowerer.materialize(value)?;
    lowerer.target.blocks[0].instrs.push(Instr::new(OpRegOut {
        srcs: ssa.iter().copied().map(Src::from).collect(),
    }));

    let mut info = ShaderInfo::fragment(entry.early_depth_test.is_some(), false, false);
    let ShaderIoInfo::Fragment(io) = &mut info.io else {
        unreachable!();
    };
    io.writes_color = 0xf;
    Ok(Shader {
        sm,
        info,
        functions: vec![lowerer.finish()],
    })
}
