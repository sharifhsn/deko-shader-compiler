@group(0) @binding(0)
var<storage, read_write> output: array<u32>;

fn write_if_live(lane: u32) {
    var iteration = 0u;
    loop {
        if iteration == lane {
            return;
        }
        iteration += 1u;
        if iteration == 2u {
            break;
        }
    }
    output[lane] = 1u;
}

@compute @workgroup_size(4)
fn main(@builtin(local_invocation_index) lane: u32) {
    write_if_live(lane);
}
