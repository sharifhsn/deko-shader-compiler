# Roadmap

1. Extract the minimal NAK SM50 IR, optimizer, register allocator, scheduler, encoder,
   and shader-program-header implementation without Mesa/NIR or C runtime dependencies.
2. Prove NAK-generated vertex, fragment, and compute DKSH on physical Switch hardware.
3. Implement direct validated Naga IR to NAK IR lowering for the complete
   Switch-supported wgpu shader surface.
4. Integrate at wgpu pipeline creation, where the selected entry point, override
   constants, pipeline layout, and validated Naga module are all available.
5. Add deterministic RAM and persistent caches, structured diagnostics, fuzzing,
   differential tests, Bevy/Warbell closure, and physical-hardware soak tests.

The complete acceptance contract lives in Warbell's
`docs/deko-shader-compiler-plan.md` while the repositories are developed together.

