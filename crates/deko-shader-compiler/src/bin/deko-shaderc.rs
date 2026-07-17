//! Minimal command-line WGSL to DKSH compiler.

use std::{env, fs, process::ExitCode};

use deko_shader_compiler::{BindingArraySize, Compiler, Options, PipelineConstants, Stage};

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
    let mut binding_array_sizes = Vec::new();
    for argument in arguments {
        let (binding, count) = argument.split_once('=').ok_or_else(usage)?;
        let (group, binding) = binding.split_once(':').ok_or_else(usage)?;
        binding_array_sizes.push(BindingArraySize {
            group: group
                .parse()
                .map_err(|_| format!("invalid bind-group index in '{argument}'"))?,
            binding: binding
                .parse()
                .map_err(|_| format!("invalid binding index in '{argument}'"))?,
            count: count
                .parse()
                .map_err(|_| format!("invalid binding-array count in '{argument}'"))?,
        });
    }
    let source = fs::read_to_string(&input)
        .map_err(|error| format!("failed to read WGSL '{input}': {error}"))?;
    let artifact = Compiler
        .compile_wgsl(
            &source,
            stage,
            &entry_point,
            &PipelineConstants::new(),
            Options {
                binding_array_sizes,
                ..Options::default()
            },
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
    "usage: deko-shaderc <input.wgsl> <output.dksh> <vertex|fragment|compute> [entry-point] [group:binding=count ...]".to_owned()
}
