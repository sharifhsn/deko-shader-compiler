# NAK porting boundary

The files under `upstream/mesa` are byte-identical inputs. The compiled extraction lives
under `src`; `ACTIVE_FILES.toml` records which active modules remain byte-identical and
which have standalone adaptations. This keeps every change reviewable against the pinned
upstream revision.

Completed extraction boundaries:

1. `nak_bindings` is replaced by `src/bindings.rs`; Mesa's C API is replaced by the
   deterministic debug shim in `src/debug.rs`.
2. The IR dispatcher accepts Maxwell shader models only. SM20, SM32, SM70, and later
   encoders are not compiled.
3. Core IR/SSA, legalization, optimization, allocation, copy lowering, scheduling,
   dependency calculation, and the SM50 encoder compile as native Rust modules.
4. The SPH type and bitfield boundary is local Rust. NAK SPH words remain distinct from
   the 36-byte stage-specific union in `DkshProgramHeader`.

Next extraction boundary:

1. Expose a deterministic synthetic-IR compile path through the complete SM50 pass
   pipeline.
2. Lower validated Naga IR into NAK IR, starting with vertex/fragment WGSL fixtures.
3. Finish stage-specific SPH and DKSH metadata translation, then add compute.

`from_nir.rs`, QMD generation, Mesa's C API, Nouveau winsys code, and non-SM50
encoders are intentionally outside the runtime closure.
