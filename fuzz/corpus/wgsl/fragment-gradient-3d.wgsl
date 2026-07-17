@group(0) @binding(0) var image: texture_3d<f32>;
@group(0) @binding(1) var image_sampler: sampler;

@fragment
fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSampleGrad(
        image,
        image_sampler,
        vec3<f32>(uv, 0.5),
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<i32>(1, 0, -1),
    );
}
