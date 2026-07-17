#![no_main]

use deko_shader_compiler::{Compiler, Options, PipelineConstants, Stage};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|bytes: &[u8]| {
    if bytes.len() < 9 {
        return;
    }
    let stage = match bytes[0] % 3 {
        0 => Stage::Vertex,
        1 => Stage::Fragment,
        _ => Stage::Compute,
    };
    let operation = ["+", "-", "*", "^", "&", "|"][usize::from(bytes[1] % 6)];
    let left = u32::from_le_bytes(bytes[1..5].try_into().unwrap());
    let right = u32::from_le_bytes(bytes[5..9].try_into().unwrap());
    let source = match stage {
        Stage::Vertex => format!(
            "@vertex fn main(@builtin(vertex_index) seed: u32) -> @builtin(position) vec4<f32> {{ let left = seed ^ {left}u; let right = seed ^ {right}u; let value = left {operation} right; return vec4<f32>(f32(value & 1u)); }}"
        ),
        Stage::Fragment => format!(
            "@fragment fn main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {{ let seed = u32(position.x); let left = seed ^ {left}u; let right = seed ^ {right}u; let value = left {operation} right; return vec4<f32>(f32(value & 255u) / 255.0); }}"
        ),
        Stage::Compute => format!(
            "@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {{ let left = id.x ^ {left}u; let right = id.x ^ {right}u; let value = left {operation} right; _ = value; }}"
        ),
    };
    Compiler
        .compile_wgsl(
            &source,
            stage,
            "main",
            &PipelineConstants::new(),
            Options::default(),
        )
        .expect("generated lowering fixture must compile");
});
