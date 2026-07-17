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
  captured Bevy corpus currently compiles 110 of 110 entry points.
- Integrate the compiler into the Deko3D wgpu backend and validate runtime WGSL
  rendering in Ryujinx without game-specific DKSH overrides. Physical Switch
  execution remains an explicit release gate.
