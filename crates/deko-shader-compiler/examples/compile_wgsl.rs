//! Compile a minimal WGSL vertex shader into a DKSH artifact.

use deko_shader_compiler::{Compiler, Options, PipelineConstants, Stage};

const WGSL: &str = r"
@vertex
fn main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vertex_index) - 1);
    let y = select(-1.0, 1.0, vertex_index == 2u);
    return vec4<f32>(x, y, 0.0, 1.0);
}
";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let artifact = Compiler.compile_wgsl(
        WGSL,
        Stage::Vertex,
        "main",
        &PipelineConstants::new(),
        Options::default(),
    )?;
    println!(
        "compiled {} DKSH bytes with {} reflected bindings",
        artifact.dksh.len(),
        artifact.bindings.len()
    );
    Ok(())
}
