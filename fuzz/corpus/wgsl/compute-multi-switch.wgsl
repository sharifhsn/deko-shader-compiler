@group(0) @binding(0) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(4)
fn main(@builtin(local_invocation_index) lane: u32) {
    switch lane {
        case 0u, 1u: {
            output[lane] = 10u;
        }
        default: {
            output[lane] = 20u;
        }
    }
}
