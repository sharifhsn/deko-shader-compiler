use deko_nak::ir::{Function, Instr, OpExit, Shader, ShaderInfo, ShaderModelInfo};

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
        stage => Err(Error::UnsupportedFeature(format!(
            "{stage:?} lowering is not implemented yet"
        ))),
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
