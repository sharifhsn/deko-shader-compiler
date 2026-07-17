# Third-party notices

This project contains Rust source extracted from Mesa's NAK compiler at commit
`99bf1b953cba11d41dbee519ae1b53387477df9e`. Those files retain their original
copyright and `SPDX-License-Identifier: MIT` headers. The exact source-to-local
mapping is recorded in `THIRD_PARTY.toml`, `crates/deko-nak/UPSTREAM_FILES.txt`, and
`crates/deko-nak/ACTIVE_FILES.toml`.

The DKSH parser and serializer are independently written from the public Deko3D
format and loader behavior at commit `350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db`.
Deko3D is distributed under the zlib license. No Deko3D implementation source is
included in this repository.

`tools/check_provenance.py` verifies that declared local files exist, imported MIT
files retain their SPDX headers, and files classified as exact still match their
staged upstream snapshots.
