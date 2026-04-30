# jetro-experimental

Structural-index substrate for fast JSON queries.  Pairs the SIMD-accelerated
stage-1 scanner of [simd-json](https://github.com/simd-lite/simd-json) with a 
Mison-style key-bitmap layer.  Built to slot under [jetro](https://github.com/mitghi/jetro) 
for `$..find(k == lit)` shapes without materialising the value tree.

## What it does

Given a JSON byte buffer, build:

1. **SIMD-JSON Stage-1 columns** — byte offset, kind, depth for every structural
   character (`{`, `}`, `[`, `]`, `"`, `:`, `,`, scalar starts).
2. **Sidecar** — `parent[]`, `close_of[]`, `tape_of[]` for tree navigation
   and tape alignment.
3. **Mison key bitmaps** — interned key dictionary with a Roaring bitmap
   per key over token positions.  Enables compound predicates by
   bitmap-AND.

Built once, reusable.  Queries like `$..find(x == "test")` use the bitmap
to skip every JSON object that doesn't contain key `x`, and a SIMD byte
compare to validate the value.  No `Val` tree allocation.

## Why

Two pillars:

- **Mison** (Li et al., VLDB 2017) — speculative key-bitmap indices
  outperform tree walks on selective queries by 10-100×.
- **simd-json** (Lemire et al.) — SIMD JSON parsing at ~25 GB/s.  Stage-1
  already produces every byte offset we need; we reuse the result via a
  vendored fork that exposes `Buffers::structural_indexes()`.

The two compose: simd-json walks once, jetro-experimental zips its output
into structural columns + key bitmaps in a fused single pass.

## Architecture

```
                   ┌──────────────────────────────┐
                   │ JSON bytes (mut [u8])        │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
              ┌──────────────────────────────────────────┐
              │ simd-json::to_tape_with_buffers          │
              │  • stage-1: SIMD structural scan         │
              │  • stage-2: Tape<Node> construction      │
              │  → Tape  +  Buffers::structural_indexes │
              └──────────────────────────────────────────┘
                                  │
                                  ▼
              ┌──────────────────────────────────────┐
              │ from_simdjson  (one fused walk)      │
              │  • emit offset/kind/depth columns    │
              │  • parent / close_of / tape_of       │
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
use jetro_experimental::{parse, find_eq, count_key};

let bytes = br#"{"a":{"x":"test"},"b":{"x":"nope"},"c":{"x":"test"}}"#.to_vec();
let parsed = parse(bytes).unwrap();

// Mison-style: every object whose `x` key equals "test".
for obj_tok in find_eq(&parsed.index, &parsed.bytes, "x", b"test") {
    let span = parsed.index.byte_span(obj_tok);
    let raw = &parsed.bytes[span.start as usize..span.end as usize];
    println!("{}", std::str::from_utf8(raw).unwrap());
}
// → {"x":"test"}
// → {"x":"test"}

// Popcount only — no value walk.
let n = count_key(&parsed.index, "x");
assert_eq!(n, 3);
```

## Public API

### Entry points

```rust
fn from_bytes(buf: &[u8]) -> Result<StructuralIndex, Error>;
fn from_bytes_with(buf: &[u8], opts: BuildOptions) -> Result<StructuralIndex, Error>;

// feature = "simd-json"
fn parse(bytes: Vec<u8>) -> Result<Parsed, Error>;
fn from_simdjson(tape: &Tape, bytes: &[u8], structurals: &[u32], opts: BuildOptions)
    -> Result<StructuralIndex, Error>;
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
    fn tape_index(&self, tok: TokenId) -> Option<u32>;
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
| `simd-json`       | Vendored simd-json fork (`vendor-simd-json/`) + tape bridge     |
| `fast-numbers`    | `fast-float` SIMD f64 parser (~3× over `str::parse`)            |
| `multi-key`       | Aho-Corasick + Teddy multi-pattern matcher for raw byte scans   |
| `validate-utf8`   | `simdutf8` (~5-10 GB/s)                                         |

Default: minimal (no SIMD-json bridge).  Enable `simd-json` for the
`from_simdjson` / `parse` entry points and the `sj::node` bridge.

## Bench numbers

Apple M-series, 1.7 MB doc, single thread:

| operation                                  | latency    |
|---------------------------------------------|------------|
| `parse_with_index_fused`                    | 9.6 ms     |
| `from_bytes_separate_pass` (no fork)        | 15.5 ms    |
| `from_bytes` minimal (no key bitmaps)       | 4.2 ms     |
| `count_key` (popcount)                      | 570 ns     |
| `keys_named.first()` (short-circuit)        | 637 ns     |
| `container_at_byte`                         | 26 ns      |
| `json_string_eq` 32-byte no-escape          | 8 ns       |
| `parse_f64` (fast-numbers)                  | 56 ns      |
| `parse_f64` (std::parse fallback)           | 159 ns     |

`cargo bench --bench substrate --features simd-json,fast-numbers`

## Examples

- `examples/structural_columns.rs` — derive byte spans for every value
  via `Buffers::structural_indexes()`; demonstrate byte-position →
  enclosing-object lookup.

  ```bash
  cargo run --release --example structural_columns --features simd-json
  ```

- `src/bin/jetro_bridge_demo.rs` — Mison `$..find(x == "test")` end-to-end
  with simd-json tape alignment.

## Status

Pre-1.0.  Public API is stable (`StructuralIndex`, `TokenId`, fused
helpers).  Internals (`Stage1`, `KeyBitmaps`, `StructIndex`) are
`#[doc(hidden)]` and may change without semver bumps.

## Roadmap

- [x] Stage-1 SIMD scanner (AVX2 + scalar/memchr fallback)
- [x] StructuralIndex + Mison KeyBitmaps
- [x] `from_simdjson` fused single-pass build
- [x] Public API facade with opaque types
- [x] SIMD helpers: `json_string_eq`, `parse_f64`, multi-key, utf8 validation
- [x] Criterion benches
- [ ] Streaming `IndexBuilder` for NDJSON / huge inputs
- [ ] NEON stage-1 port (currently scalar+memchr fallback on aarch64)
- [ ] Drop vendored fork once upstream simd-json exposes
      `Buffers::structural_indexes()`

## Vendored simd-json

`vendor-simd-json/` is a fork of simd-json 0.13.10 with a single
1-line patch adding `Buffers::structural_indexes()` so we can read
stage-1's structural offsets without an extra SIMD pass.

## License

MIT/Apache-2.0 (matching simd-json).
