#![no_main]

use deko_shader_compiler::{Compiler, Options, PipelineConstants, Stage};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|bytes: &[u8]| {
    let Ok(source) = core::str::from_utf8(bytes) else {
        return;
    };
    for stage in [Stage::Vertex, Stage::Fragment, Stage::Compute] {
        let _ = Compiler.compile_wgsl(
            source,
            stage,
            "main",
            &PipelineConstants::new(),
            Options::default(),
        );
    }
});
