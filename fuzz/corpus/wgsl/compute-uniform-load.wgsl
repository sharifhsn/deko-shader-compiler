var<workgroup> value: u32;

@compute @workgroup_size(4)
fn main(@builtin(local_invocation_index) lane: u32) {
    if lane == 0u {
        value = 42u;
    }
    _ = workgroupUniformLoad(&value);
}
