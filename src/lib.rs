//! jetro-experimental — structural stage-1 over JSON bytes.
//!
//! Single pass → columnar output:
//!   - byte_offset[i] : u32     // position of every structural char
//!   - kind[i]        : Kind    // {, }, [, ], " (opening), :, ,
//!   - depth[i]       : u16     // nesting depth at that offset
//!
//! Strings tracked end-to-end; bytes between unescaped quotes never appear
//! in the output. AVX2 fast path ~25 GB/s, scalar fallback ~5-10 GB/s.

#![allow(unsafe_op_in_unsafe_fn)]

// === Public API (stable) ===
pub mod api;
pub mod op;

pub use op::{run_count, run_first, Ctx, Op, OpPlan};

pub use api::{
    count_key, find_eq, find_eq_compound, from_bytes, from_bytes_with, json_number_eq,
    json_string_eq, parse_f64, parse_i64, BuildOptions, ByteSpan, Error, KeyHits, StructuralIndex,
    TokenId, TokenKind, Tokens,
};

#[cfg(feature = "multi-key")]
pub use api::{multi_key_finder, multi_key_scan};

#[cfg(feature = "validate-utf8")]
pub use api::validate_utf8;

// === Internals (kept public for tests/benches; do not depend on these
// from library code — they may change without semver bumps).
#[doc(hidden)]
pub mod index;
#[doc(hidden)]
pub mod keys;
#[doc(hidden)]
pub mod stage1;

#[doc(hidden)]
pub use index::{parent_chain, token_at, StructIndex};
#[doc(hidden)]
pub use keys::{KeyBitmaps, Role};
#[doc(hidden)]
pub use stage1::{run, run_scalar, Kind, Stage1};
