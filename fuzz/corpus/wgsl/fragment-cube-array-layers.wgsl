@group(0) @binding(0) var image: texture_cube_array<f32>;

@fragment
fn main() -> @location(0) vec4<f32> {
    return vec4<f32>(f32(textureNumLayers(image)));
}
