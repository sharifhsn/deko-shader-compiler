# deko-shader-compiler

An open Rust compiler for translating validated Naga shaders into native Maxwell
machine code and Deko3D's DKSH container format, initially targeting the Nintendo
Switch's GM20B GPU.

The project is under active development. It compiles validated WGSL
vertex, fragment, and compute entry points through a Rust extraction of Mesa NAK's SM50 backend
and packages the resulting GM20B machine code as DKSH. The host suite covers representative
Bevy UI, mesh, PBR, image-processing, and compute shader families; the application-level
Ryujinx probe currently exercises 29 distinct runtime-compiled artifacts. Compute shaders
support the core invocation built-ins,
read/write storage buffers with static or dynamically indexed host-shareable data, storage-buffer,
storage-texture, and workgroup atomics, barriers, workgroup memory, and runtime storage-array
lengths. Unsupported
features return a typed error instead of silently changing shader semantics.
Subgroup support includes barriers, gathers, invocation and geometry built-ins, boolean
all/any reductions, ballots, arithmetic and bitwise reductions, and add/multiply scans.
Collectives use an active-lane ballot so partial and sparse GM20B warps do not contribute
undefined shuffle values.
`workgroupUniformLoad` supports constructible and atomic workgroup values with two
workgroup synchronization points. Divergent structured branches predicate their memory,
image, atomic, and discard side effects while pure calculations remain safely if-converted.
Nested value and void returns remove completed lanes from later side effects.
Returns taken from inside loops also remove those lanes from effects after the loop.
Terminal unconditional loop controls preserve values written before `break` and route
`continue` through the WGSL continuing block. Ordinary conditional loop exits retain the
established loop-header value path, while backward local-liveness analysis limits changed
break-edge exit phis to values actually read after the loop.
Pointer arguments preserve per-invocation writes across divergent void and value helpers.
WGSL switch cases with multiple selectors lower to one shared conditional body while ordinary
switches retain their direct lowering path.
Native Maxwell TXD lowering supports explicit WGSL gradients for 1D/2D sampled textures,
including array layers and constant offsets. 3D and cube gradients use Mesa's mathematically
equivalent derivative-to-LOD rewrite before the ordinary Maxwell texture instruction.
Multiview pipelines use wgpu's Deko draw-replay ABI: the compiler loads the current view from
reserved uniform target 14, writes the Maxwell layer output, and exposes the layer as fragment
`view_index`.

The compiler is integrated into the Deko3D backend of the accompanying wgpu fork.
An override-free Ryujinx run has compiled ordinary Bevy WGSL at runtime and rendered
the Switch probe through frame 60 with three passing visual captures. Emulator proof
does not replace the physical-Switch execution gate.

The compiler does not require Nintendo's proprietary SDK, Mesa at runtime, or
the host-side UAM executable. Mesa's MIT-licensed NAK compiler is the machine-backend
foundation. Imported upstream source retains its original notices and is recorded in
`THIRD_PARTY.toml`.

## Workspace

- `deko-shader-compiler`: public Naga-facing compiler API
- `deko-nak`: GM20B machine backend derived from Mesa NAK
- `deko-dksh`: safe DKSH container and binding-metadata model
- `deko-shader-compiler-macros`: proc macros required by the extracted NAK IR

`deko-shader-compiler` also exposes a versioned `CacheKey` and thread-safe `CompilerCache`.
Cache identities include the exact WGSL, stage, entry point, canonical pipeline constants, every
code-generation option, the package version, and a manually versioned backend ABI. The RAM cache
has configurable entry and byte limits with deterministic LRU eviction. Call
`with_persistent_directory` to opt into checksummed, structurally validated DKSH files written by
atomic rename; corrupt or stale files are removed and regenerated without making compilation fail.

Runtime-sized WGSL `binding_array<T>` declarations take their descriptor count from
`Options::binding_array_sizes`; wgpu forwards these values from the explicit pipeline layout. The
CLI accepts the same information as one or more trailing `group:binding=count` arguments:

```sh
deko-shaderc shader.wgsl shader.dksh fragment main 0:3=16 0:4=16
```

## Development

```sh
cargo fmt --all --check
python3 tools/check_provenance.py
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p deko-shader-compiler --example compile_wgsl
cargo package --workspace --no-verify --list
cargo +nightly fuzz run dksh -- -max_total_time=30
cargo +nightly fuzz run wgsl -- -max_total_time=30
cargo +nightly fuzz run lowering -- -max_total_time=30
```

`cargo package --workspace --no-verify --list` verifies the exact contents of all
six unpublished workspace crates without requiring those path dependencies to exist
on crates.io already. Publish dependency crates in topological order before running a
full registry-backed package verification.

The public support boundary is the operations exercised by the tests and Bevy corpus,
not all of WGSL. Native execution and performance claims require physical Switch evidence.

The three fuzz targets cover the untrusted DKSH parser, arbitrary WGSL bytes at the
Naga boundary, and a valid generated vertex/fragment/compute family that reaches the
Naga-to-NAK lowering and native encoder.

## License

MIT. See `LICENSE`, `THIRD_PARTY.toml`, and `THIRD_PARTY_NOTICES.md`.
