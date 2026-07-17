# deko-shader-compiler

An open Rust compiler for translating validated Naga shaders into native Maxwell
machine code and Deko3D's DKSH container format, initially targeting the Nintendo
Switch's GM20B GPU.

The project is under active development. Its first end-to-end slice compiles validated WGSL
vertex, fragment, and compute entry points through a Rust extraction of Mesa NAK's SM50 backend
and packages the resulting GM20B machine code as DKSH. The currently supported language surface
is deliberately small; unsupported features return a typed error instead of silently changing
shader semantics.

The finished compiler will not require Nintendo's proprietary SDK, Mesa at runtime, or
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

## Development

```sh
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo package -p deko-dksh --allow-dirty
```

## License

MIT. See `LICENSE` and `THIRD_PARTY.toml`.
