@group(0) @binding(0)
var image: texture_storage_2d<r32uint, atomic>;

@compute @workgroup_size(4)
fn main(@builtin(local_invocation_index) lane: u32) {
    textureAtomicAdd(image, vec2<i32>(i32(lane), 0), 1u);
}
