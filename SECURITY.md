# Security policy

## Reporting a vulnerability

Please report memory-safety problems, malformed-shader crashes, GPU-hang inputs, or
other security-sensitive issues through a private GitHub security advisory for
`sharifhsn/deko-shader-compiler`. Do not open a public issue until a fix is available.

Include the smallest WGSL or Naga reproducer you can share, the compiler revision,
target, and whether the failure occurred on host, Ryujinx, or physical hardware.
Never attach Nintendo SDK files, keys, firmware, or other proprietary material.

## Scope

The compiler treats shader source and cached artifacts as untrusted input. A report
is in scope when that input can cause memory unsafety, an uncontrolled process abort,
an invalid DKSH container to cross the validation boundary, or a repeatable GPU hang.
Unsupported WGSL that returns a typed error is expected behavior.
