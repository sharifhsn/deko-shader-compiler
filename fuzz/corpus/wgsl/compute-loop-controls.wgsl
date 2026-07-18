@group(0) @binding(0) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(4)
fn main(@builtin(local_invocation_index) lane: u32) {
    var value = lane;
    loop {
        {
            value += 1u;
            if lane == 0u {
                value += 2u;
                break;
            }
            continue;
        }
        continuing {
            break if value == lane + 3u;
        }
    }
    output[lane] = value;
}
