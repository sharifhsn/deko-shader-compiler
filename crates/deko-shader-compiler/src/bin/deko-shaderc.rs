//! Minimal command-line WGSL to DKSH compiler.

use std::{env, fs, process::ExitCode};

use deko_shader_compiler::{Compiler, Options, PipelineConstants, Stage};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("deko-shaderc: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut arguments = env::args().skip(1);
    let input = arguments.next().ok_or_else(usage)?;
    let output = arguments.next().ok_or_else(usage)?;
    let stage = match arguments.next().as_deref() {
        Some("vertex") => Stage::Vertex,
        Some("fragment") => Stage::Fragment,
        Some("compute") => Stage::Compute,
        _ => return Err(usage()),
    };
    let entry_point = arguments.next().unwrap_or_else(|| "main".to_owned());
    if arguments.next().is_some() {
        return Err(usage());
    }
    let source = fs::read_to_string(&input)
        .map_err(|error| format!("failed to read WGSL '{input}': {error}"))?;
    let artifact = Compiler
        .compile_wgsl(
            &source,
            stage,
            &entry_point,
            &PipelineConstants::new(),
            Options::default(),
        )
        .map_err(|error| error.to_string())?;
    fs::write(&output, &artifact.dksh)
        .map_err(|error| format!("failed to write DKSH '{output}': {error}"))?;
    eprintln!(
        "compiled {input} {stage:?}/{entry_point} -> {output} ({} bytes, {} bindings)",
        artifact.dksh.len(),
        artifact.bindings.len()
    );
    Ok(())
}

fn usage() -> String {
    "usage: deko-shaderc <input.wgsl> <output.dksh> <vertex|fragment|compute> [entry-point]"
        .to_owned()
}
