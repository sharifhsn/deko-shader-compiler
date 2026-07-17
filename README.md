# deko-shader-compiler

An open Rust compiler for translating validated Naga shaders into native Maxwell
machine code and Deko3D's DKSH container format, initially targeting the Nintendo
Switch's GM20B GPU.

The project is under active development and is not yet capable of compiling a shader.
The first implemented component is the safe `deko-dksh` container encoder/parser; the
NAK-derived machine backend and Naga lowering are tracked in `ROADMAP.md`.

The finished compiler will not require Nintendo's proprietary SDK, Mesa at runtime, or
the host-side UAM executable. Mesa's MIT-licensed NAK compiler is the machine-backend
foundation. Imported upstream source retains its original notices and is recorded in
`THIRD_PARTY.toml`.

## Workspace

- `deko-shader-compiler`: public Naga-facing compiler API
- `deko-nak`: GM20B machine backend derived from Mesa NAK
- `deko-dksh`: safe DKSH container and binding-metadata model
- `deko-shader-compiler-macros`: proc macros required by the extracted NAK IR

## Development

```sh
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo package -p deko-dksh --allow-dirty
```

## License

MIT. See `LICENSE` and `THIRD_PARTY.toml`.

