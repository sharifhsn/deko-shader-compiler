@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) lane: u32) {
    let predicate = lane < 16u;
    _ = subgroupAll(predicate);
    _ = subgroupAny(predicate);
    _ = subgroupBallot(predicate);
}
