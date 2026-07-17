# NAK porting boundary

The files under `upstream/mesa` are byte-identical inputs, not compiled modules. They
make the extraction reviewable and allow every adaptation to be compared with a pinned
upstream revision.

Port in this order:

1. Replace `nak_bindings` with `src/bindings.rs` and generated SPH constants with a
   local bit-range module.
2. Adapt `ir.rs`, `builder.rs`, `ssa_value.rs`, and `union_find.rs` while pruning the
   compile-time SM20/SM32/SM70 model variants.
3. Add the debug-flag shim required by upstream passes.
4. Adapt liveness, register tracking, constant tracking, legalization, optimization,
   allocation, parallel-copy lowering, scheduling, and dependency calculation.
5. Adapt `sm50.rs` and `sm30_instr_latencies.rs`, then expose one deterministic
   synthetic-IR compile path.
6. Adapt `sph.rs` separately. NAK SPH words are shader IO metadata and are not the
   36-byte stage-specific union in `DkshProgramHeader`.

`from_nir.rs`, QMD generation, Mesa's C API, Nouveau winsys code, and non-SM50
encoders are intentionally outside the runtime closure.

