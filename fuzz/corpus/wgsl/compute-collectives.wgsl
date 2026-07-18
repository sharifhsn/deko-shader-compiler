@compute @workgroup_size(7)
fn main(@builtin(local_invocation_index) lane: u32) {
    let value = lane + 1u;
    _ = subgroupAdd(value);
    _ = subgroupXor(value);
    _ = subgroupExclusiveAdd(value);
    _ = subgroupInclusiveMul(value);
}
