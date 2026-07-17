@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) lane: u32) {
    _ = subgroupBroadcastFirst(lane);
}
