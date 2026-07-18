@group(0) @binding(0) var<storage, read_write> output: array<u32>;

fn choose(destination: ptr<function, u32>, lane: u32) -> u32 {
    if lane == 0u {
        *destination = 11u;
        return 1u;
    }
    *destination = 22u;
    return 2u;
}

@compute @workgroup_size(4)
fn main(@builtin(local_invocation_index) lane: u32) {
    var value = 0u;
    output[lane] = value + choose(&value, lane);
}
