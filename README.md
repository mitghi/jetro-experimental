# jetro-experimental

Structural-index substrate for fast JSON queries.  Standalone stage-1
scanner + Mison-style key-bitmap layer.  Built to slot under
[jetro](https://github.com/mitghi/jetro) for `$..find(k == lit)` shapes
without materialising the value tree.

## What it does

Given a JSON byte buffer, build:

1. **Stage-1 columns** from [simd-json](https://github.com/simd-lite/simd-json) byte offset, kind, depth for every structural
   character (`{`, `}`, `[`, `]`, `"`, `:`, `,`, scalar starts).  AVX2
   fast path; scalar/`memchr` fallback elsewhere.
2. **Sidecar** — `parent[]`, `close_of[]` for tree navigation.
3. **Mison key bitmaps**  interned key dictionary with a Roaring bitmap
   per key over token positions.  Compound predicates via bitmap-AND.

Built once, reusable.  Queries like `$..find(x == "test")` use the
bitmap to skip every JSON object that doesn't contain key `x`, and a
SIMD byte compare to validate the value.  No `Val` tree allocation.

## Why

**Mison** (Li et al., VLDB 2017) — speculative key-bitmap indices
outperform tree walks on selective queries by 10-100×.  jetro-experimental
is a self-contained Mison-style substrate that builds the structural
columns and key bitmaps directly from raw JSON in a single pass.

## Architecture

```
                   ┌──────────────────────────────┐
                   │ JSON bytes (&[u8])           │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
              ┌──────────────────────────────────────────┐
              │ stage-1 SIMD/scalar scan                 │
              │  • emit offset/kind/depth columns        │
              └──────────────────────────────────────────┘
                                  │
                                  ▼
              ┌──────────────────────────────────────┐
              │ index build (one fused walk)         │
              │  • parent / close_of                 │
              │  • role classification               │
              │  • key interner + Roaring bitmaps    │
              └──────────────────────────────────────┘
                                  │
                                  ▼
                    StructuralIndex  +  KeyBitmaps
                                  │
                                  ▼
              ┌─────────┬────────┬──────────┬─────────┐
              │ find_eq │ count  │ ancestors│ slice() │
              │  (Mison)│   (1)  │ (parents)│  raw    │
              └─────────┴────────┴──────────┴─────────┘
```

## Quick start

```rust
use jetro_experimental::{from_bytes, find_eq, count_key};

let bytes = br#"{"a":{"x":"test"},"b":{"x":"nope"},"c":{"x":"test"}}"#;
let idx = from_bytes(bytes).unwrap();

// Mison-style: every object whose `x` key equals "test".
for obj_tok in find_eq(&idx, bytes, "x", b"test") {
    let span = idx.byte_span(obj_tok);
    let raw = &bytes[span.start as usize..span.end as usize];
    println!("{}", std::str::from_utf8(raw).unwrap());
}
// → {"x":"test"}
// → {"x":"test"}

// Popcount only — no value walk.
let n = count_key(&idx, "x");
assert_eq!(n, 3);
```

## Public API

### Entry points

```rust
fn from_bytes(buf: &[u8]) -> Result<StructuralIndex, Error>;
fn from_bytes_with(buf: &[u8], opts: BuildOptions) -> Result<StructuralIndex, Error>;
```

### Per-token

```rust
impl StructuralIndex {
    fn kind(&self, tok: TokenId) -> TokenKind;
    fn depth(&self, tok: TokenId) -> u16;
    fn byte_offset(&self, tok: TokenId) -> u32;
    fn byte_span(&self, tok: TokenId) -> ByteSpan;
    fn parent(&self, tok: TokenId) -> Option<TokenId>;
    fn close_of(&self, container: TokenId) -> Option<TokenId>;
    fn slice<'a>(&self, bytes: &'a [u8], tok: TokenId) -> &'a [u8];
}
```

### Byte-position lookup

```rust
fn container_at_byte(&self, pos: u32) -> Option<TokenId>;
fn ancestors(&self, tok: TokenId) -> Ancestors<'_>;
```

### Mison key layer

```rust
fn keys_named(&self, name: &str, depth: Option<u16>) -> KeyHits<'_>;
fn has_key(&self, name: &str) -> bool;
fn keys_seen(&self) -> impl Iterator<Item = &str> + '_;
fn value_for_key(&self, key_tok: TokenId) -> Option<TokenId>;
```

### KeyHits — lazy composable

```rust
impl KeyHits<'_> {
    fn at_depth(self, d: u16) -> Self;
    fn and(self, other: Self) -> Self;        // bitmap intersect
    fn or(self, other: Self) -> Self;
    fn count(self) -> u64;                    // O(1) popcount
    fn first(self) -> Option<TokenId>;        // short-circuit
    fn last(self) -> Option<TokenId>;
}
impl Iterator for KeyHits<'_> { type Item = TokenId; }
```

### Fused query primitives

```rust
fn find_eq<'a>(idx, bytes, key, literal) -> impl Iterator<Item = TokenId>;
fn find_eq_compound<'a>(idx, bytes, conds) -> impl Iterator<Item = TokenId>;
fn count_key(idx, key) -> u64;
```

### SIMD helpers

```rust
fn json_string_eq(value: &[u8], literal: &[u8]) -> bool;     // memchr escape probe
fn json_number_eq(value: &[u8], literal: &[u8]) -> bool;     // numeric semantics
fn parse_f64(bytes: &[u8]) -> Option<f64>;                   // fast-float when "fast-numbers"
fn parse_i64(bytes: &[u8]) -> Option<i64>;
```

## Feature flags

| feature           | adds                                                            |
|-------------------|-----------------------------------------------------------------|
| `fast-numbers`    | `fast-float` SIMD f64 parser (~3× over `str::parse`)            |
| `multi-key`       | Aho-Corasick + Teddy multi-pattern matcher for raw byte scans   |
| `validate-utf8`   | `simdutf8` (~5-10 GB/s)                                         |

Default: minimal.  No external SIMD-JSON dependency — stage-1 of [simd-json](https://github.com/simd-lite/simd-json) is fully
self-contained.

## Status

Pre-1.0.  Public API is stable (`StructuralIndex`, `TokenId`, fused
helpers).  Internals (`Stage1`, `KeyBitmaps`, `StructIndex`) are
`#[doc(hidden)]` and may change without semver bumps.

## Roadmap

- [x] Stage-1 SIMD scanner (AVX2 + scalar/memchr fallback)
- [x] StructuralIndex + Mison KeyBitmaps
- [x] Public API facade with opaque types
- [x] SIMD helpers: `json_string_eq`, `parse_f64`, multi-key, utf8 validation
- [ ] NEON stage-1 port (currently scalar+memchr fallback on aarch64)
- [ ] Streaming `IndexBuilder` for NDJSON / huge inputs

## License

This repository contain direct copy of Stage 1 from [simd-json](https://github.com/simd-lite/simd-json) project.

MIT/Apache-2.0.
