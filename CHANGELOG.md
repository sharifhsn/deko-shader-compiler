# Changelog

## Unreleased

- Create a publishable, SDK-independent compiler workspace targeting the Nintendo
  Switch GM20B GPU.
- Add direct WGSL/Naga lowering for vertex, fragment, and compute shaders into the
  extracted Mesa NAK SM50 backend.
- Add structured control flow, numeric and composite operations, functions and
  pointers, graphics and compute built-ins, uniform/storage/workgroup memory,
  atomics, barriers, sampled and storage textures, queries, derivatives, and
  pipeline override constants required by the captured Bevy 0.19 workload.
- Add independent sampled-texture and sampler targets, statically and dynamically
  indexed binding arrays, and the packed Maxwell bindless handle ABI.
- Add native Maxwell TXD lowering for explicit 1D/2D texture gradients, including
  array-layer and offset operand packing.
- Add derivative-to-LOD lowering for explicit 3D and cube texture gradients, including
  the cube-face selection and quotient-rule calculation used by Mesa NIR.
- Advance the backend ABI cache namespace for the explicit-gradient codegen changes.
- Lower subgroup barriers to CTA-scoped memory fences on lockstep GM20B warps without
  introducing a whole-workgroup synchronization point.
- Lower `subgroupBroadcastFirst` using a Maxwell active-lane ballot, first-set-lane
  selection, and shuffle.
- Lower subgroup invocation ID, subgroup size, subgroup ID, and subgroup count compute
  built-ins from GM20B lane state and pipeline-specialized workgroup geometry.
- Lower boolean subgroup all/any reductions and subgroup ballots directly through
  Maxwell vote instructions, preserving active-lane semantics.
- Lower arithmetic and bitwise subgroup reductions plus add/multiply inclusive and
  exclusive scans for scalar and vector operands. Active ballots preserve partial-warp
  and divergent-lane semantics instead of reading inactive shuffle lanes.
- Lower `workgroupUniformLoad` for scalar, atomic, and aggregate workgroup pointers
  with the required workgroup barrier-load-barrier sequence.
- Predicate side-effecting instructions emitted by divergent `if` arms, so storage,
  workgroup, image, atomic, and discard effects occur only in invocations that selected
  that arm. Pure calculations and structured-control operations remain unconditional,
  preserving NAK SSA lifetime and scheduling invariants through if-converted loops.
- Track the live invocation mask through nested early returns, including void returns,
  so lanes that have returned cannot execute later side effects.
- Preserve that live invocation mask across loop exits so lanes returning from inside a
  loop cannot execute side effects that follow the loop.
- Snapshot and merge pointer arguments across divergent helper-function arms, and write
  updated pointer values back from both void and value-returning calls.
- Lower WGSL atomic operations on `r32uint` and `r32sint` storage textures to native
  Maxwell `SUATOM` instructions.
- Group WGSL multi-selector switch markers into a shared predicate and body while
  continuing to reject non-empty source-IR fall-through cases. Preserve the proven
  direct lowering path for ordinary switches that contain no such markers.
- Lower array texture layer-count queries, including the Maxwell cube-face count to
  cube-layer conversion used by Mesa NIR.
- Accept pipeline-specialized compute workgroup-size overrides and preserve their
  resolved dimensions in DKSH metadata.
- Lower emulated multiview `view_index` inputs through wgpu's reserved Deko uniform
  slot and emit the Maxwell layer output used by replayed vertex draws and fragment input.
- Add a safe deterministic DKSH encoder/parser with typed stage payloads and
  versioned binding metadata consumed by wgpu-hal.
- Add the explicit GM20B target descriptor, register allocation, scheduling, SPH
  generation, and machine-code encoding derived from Mesa NAK.
- Add a versioned deterministic cache key and bounded thread-safe in-memory compiler
  cache.
- Add the `deko-shaderc` command-line compiler and a public Rust API that accepts
  either WGSL or an already validated Naga module.
- Add typed diagnostics for parse, validation, specialization, unsupported-feature,
  backend, and DKSH errors.
- Add deterministic unit and regression coverage for the compiler pipeline. The
  captured Bevy corpus currently compiles 111 of 111 entry points.
- Integrate the compiler into the Deko3D wgpu backend and validate runtime WGSL
  rendering in Ryujinx without game-specific DKSH overrides. Physical Switch
  execution remains an explicit release gate.
