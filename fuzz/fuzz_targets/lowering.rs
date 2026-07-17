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
    let multiview = matches!(stage, Stage::Vertex | Stage::Fragment) && bytes[2] % 6 == 5;
    let source = match stage {
        Stage::Vertex if multiview =>
            "@vertex fn main(@builtin(view_index) view: u32) -> @builtin(position) vec4<f32> { return vec4<f32>(f32(view)); }".to_owned(),
        Stage::Vertex => format!(
            "@vertex fn main(@builtin(vertex_index) seed: u32) -> @builtin(position) vec4<f32> {{ let left = seed ^ {left}u; let right = seed ^ {right}u; let value = left {operation} right; return vec4<f32>(f32(value & 1u)); }}"
        ),
        Stage::Fragment => match bytes[2] % 6 {
            0 => format!(
                "@fragment fn main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {{ let seed = u32(position.x); let left = seed ^ {left}u; let right = seed ^ {right}u; let value = left {operation} right; return vec4<f32>(f32(value & 255u) / 255.0); }}"
            ),
            1 => "@group(0) @binding(0) var image: texture_2d_array<f32>; @group(0) @binding(1) var image_sampler: sampler; @fragment fn main() -> @location(0) vec4<f32> { return textureSampleGrad(image, image_sampler, vec2<f32>(0.5), 2, vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0), vec2<i32>(1, -1)); }".to_owned(),
            2 => "@group(0) @binding(0) var image: texture_3d<f32>; @group(0) @binding(1) var image_sampler: sampler; @fragment fn main() -> @location(0) vec4<f32> { return textureSampleGrad(image, image_sampler, vec3<f32>(0.5), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<i32>(1, 0, -1)); }".to_owned(),
            3 => "@group(0) @binding(0) var image: texture_cube_array<f32>; @group(0) @binding(1) var image_sampler: sampler; @fragment fn main() -> @location(0) vec4<f32> { return textureSampleGrad(image, image_sampler, normalize(vec3<f32>(0.5, 0.25, 1.0)), 2, vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0)); }".to_owned(),
            4 => "@group(0) @binding(0) var image: texture_cube_array<f32>; @fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(f32(textureNumLayers(image))); }".to_owned(),
            _ => "@fragment fn main(@builtin(view_index) view: u32) -> @location(0) vec4<f32> { return vec4<f32>(f32(view)); }".to_owned(),
        },
        Stage::Compute if bytes[2] % 4 == 0 => format!(
            "@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {{ let left = id.x ^ {left}u; let right = id.x ^ {right}u; let value = left {operation} right; _ = value; }}"
        ),
        Stage::Compute if bytes[2] % 4 == 1 =>
            "@compute @workgroup_size(32) fn main() { subgroupBarrier(); }".to_owned(),
        Stage::Compute if bytes[2] % 4 == 2 => "@compute @workgroup_size(32) fn main(@builtin(local_invocation_index) lane: u32) { _ = subgroupBroadcastFirst(lane); }".to_owned(),
        Stage::Compute => "@compute @workgroup_size(40) fn main(@builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) size: u32, @builtin(subgroup_id) subgroup: u32, @builtin(num_subgroups) count: u32) { _ = lane + size + subgroup + count; }".to_owned(),
    };
    Compiler
        .compile_wgsl(
            &source,
            stage,
            "main",
            &PipelineConstants::new(),
            Options {
                multiview_mask: multiview.then_some(0b101),
                ..Options::default()
            },
        )
        .expect("generated lowering fixture must compile");
});
