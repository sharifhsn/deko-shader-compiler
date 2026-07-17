# deko-shader-compiler

An open Rust compiler for translating validated Naga shaders into native Maxwell
machine code and Deko3D's DKSH container format, initially targeting the Nintendo
Switch's GM20B GPU.

The project is under active development. It compiles validated WGSL
vertex, fragment, and compute entry points through a Rust extraction of Mesa NAK's SM50 backend
and packages the resulting GM20B machine code as DKSH. The complete 110-entry captured Bevy 0.19
UI, mesh, PBR, image-processing, and compute corpus is an end-to-end regression suite. Compute
shaders support the core invocation built-ins,
read/write storage buffers with static or dynamically indexed host-shareable data, storage and
workgroup atomics, barriers, workgroup memory, and runtime storage-array lengths. Unsupported
features return a typed error instead of silently changing shader semantics.

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
code-generation option, the package version, and a manually versioned backend ABI.

Runtime-sized WGSL `binding_array<T>` declarations take their descriptor count from
`Options::binding_array_sizes`; wgpu forwards these values from the explicit pipeline layout. The
CLI accepts the same information as one or more trailing `group:binding=count` arguments:

```sh
deko-shaderc shader.wgsl shader.dksh fragment main 0:3=16 0:4=16
```

## Development

```sh
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo package -p deko-dksh --allow-dirty
```

## License

MIT. See `LICENSE` and `THIRD_PARTY.toml`.
