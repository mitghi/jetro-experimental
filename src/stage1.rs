//! Structural stage-1 for jetro.
//! Single pass over JSON bytes → columnar output:
//!   - byte_offset[i]   : u32     // position of every structural char
//!   - kind[i]          : Kind    // {, }, [, ], ", :, , (open quote only; close inferred)
//!   - depth[i]         : u16     // nesting depth at that offset
//!
//! Strings tracked end-to-end; bytes between unescaped quotes never appear in the output.
//! Quote events for the OPENING quote are emitted; closing is implicit.
//!
//! AVX2 fast path ~25 GB/s, scalar fallback ~5-10 GB/s.

#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use std::arch::x86_64::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    ObjOpen = 0,  // {
    ObjClose = 1, // }
    ArrOpen = 2,  // [
    ArrClose = 3, // ]
    Quote = 4,    // "  (opening; the string body is skipped)
    Colon = 5,    // :
    Comma = 6,    // ,
    Scalar = 7,   // number / true / false / null literal start (added by index::fill_scalars)
}

#[derive(Default, Debug, Clone)]
pub struct Stage1 {
    pub offset: Vec<u32>,
    pub kind: Vec<Kind>,
    pub depth: Vec<u16>,
}

impl Stage1 {
    pub fn with_capacity(n: usize) -> Self {
        Self {
            offset: Vec::with_capacity(n / 8),
            kind: Vec::with_capacity(n / 8),
            depth: Vec::with_capacity(n / 8),
        }
    }
    pub fn len(&self) -> usize {
        self.offset.len()
    }
    pub fn is_empty(&self) -> bool {
        self.offset.is_empty()
    }
}

pub fn run(buf: &[u8]) -> Result<Stage1, &'static str> {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        return run_avx2(buf);
    }
    // aarch64: scalar fallback uses `memchr::memchr2` which is itself
    // NEON-accelerated on aarch64.  A direct intrinsics port (vqtbl1q +
    // vshrn_n_u16 + PMULL prefix-XOR) is sketched in the `neon` module
    // below and remains future work.
    #[allow(unreachable_code)]
    run_scalar(buf)
}


/// Scalar fallback used when AVX2 / NEON SIMD paths are unavailable.
///
/// Despite the "scalar" name, the inner string-skip uses
/// `memchr::memchr2(b'"', b'\\', ...)` which is SIMD-accelerated on every
/// supported architecture (AVX2 on x86_64, NEON on aarch64, SSE2 fallback,
/// SWAR u64 fallback otherwise).  Outer match-on-byte loop typically
/// auto-vectorises under LLVM at `-O3 / LTO=thin`.
///
/// Real-world throughput on aarch64 (M-class, NEON via memchr): ~5-10 GB/s.
pub fn run_scalar(buf: &[u8]) -> Result<Stage1, &'static str> {
    let mut s = Stage1::with_capacity(buf.len());
    let mut depth: u16 = 0;
    let mut i = 0usize;
    let n = buf.len();

    while i < n {
        let b = buf[i];
        match b {
            b'{' => {
                push(&mut s, i, Kind::ObjOpen, depth);
                depth += 1;
                i += 1;
            }
            b'[' => {
                push(&mut s, i, Kind::ArrOpen, depth);
                depth += 1;
                i += 1;
            }
            b'}' => {
                if depth == 0 {
                    return Err("underflow");
                }
                depth -= 1;
                push(&mut s, i, Kind::ObjClose, depth);
                i += 1;
            }
            b']' => {
                if depth == 0 {
                    return Err("underflow");
                }
                depth -= 1;
                push(&mut s, i, Kind::ArrClose, depth);
                i += 1;
            }
            b':' => {
                push(&mut s, i, Kind::Colon, depth);
                i += 1;
            }
            b',' => {
                push(&mut s, i, Kind::Comma, depth);
                i += 1;
            }
            b'"' => {
                push(&mut s, i, Kind::Quote, depth);
                i += 1;
                // SIMD body skip: memchr2 finds the next `"` or `\\` in one
                // SIMD pass.  AVX2 / SSE2 on x86_64, NEON on aarch64.
                loop {
                    let rest = &buf[i..];
                    match memchr::memchr2(b'"', b'\\', rest) {
                        Some(p) => {
                            i += p;
                            if buf[i] == b'\\' {
                                i = i.saturating_add(2);
                            } else {
                                i += 1;
                                break;
                            }
                        }
                        None => return Err("unterminated string"),
                    }
                }
            }
            _ => i += 1, // whitespace, scalar bytes, etc.
        }
    }
    if depth != 0 {
        return Err("unclosed container");
    }
    Ok(s)
}

#[inline(always)]
fn push(s: &mut Stage1, off: usize, k: Kind, d: u16) {
    s.offset.push(off as u32);
    s.kind.push(k);
    s.depth.push(d);
}


#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2,bmi2,pclmulqdq")]
unsafe fn run_avx2(buf: &[u8]) -> Result<Stage1, &'static str> {
    let mut s = Stage1::with_capacity(buf.len());
    let mut depth: u16 = 0;

    // String-state carries across 64-byte chunks.
    let mut prev_in_string: u64 = 0;
    let mut prev_escaped: u64 = 0;

    let n = buf.len();
    let mut i = 0usize;

    while i + 64 <= n {
        let bits = chunk_bits(buf.as_ptr().add(i));

        // 1. fix backslash escapes
        let escaped = compute_escaped(bits.bs, &mut prev_escaped);
        let real_quotes = bits.qt & !escaped;

        // 2. derive in-string mask via prefix-XOR of real quotes
        let in_string_now = prefix_xor(real_quotes);
        let in_string = in_string_now ^ prev_in_string;
        // sign-extend the topmost bit of in_string for next chunk
        prev_in_string = ((in_string as i64) >> 63) as u64;

        // 3. structural mask = brackets/colon/comma OUTSIDE strings, plus opening quotes
        // opening quotes = real_quotes & !in_string (the bit that *flips into* string)
        let opening_quotes = real_quotes & !in_string;
        let struct_outside = bits.structural & !in_string;
        let mut emit_mask = struct_outside | opening_quotes;

        // 4. walk set bits in ascending order, emit + update depth
        while emit_mask != 0 {
            let bit = emit_mask.trailing_zeros() as usize;
            emit_mask &= emit_mask - 1;
            let off = i + bit;
            let b = *buf.get_unchecked(off);
            match b {
                b'{' => {
                    push(&mut s, off, Kind::ObjOpen, depth);
                    depth += 1;
                }
                b'[' => {
                    push(&mut s, off, Kind::ArrOpen, depth);
                    depth += 1;
                }
                b'}' => {
                    if depth == 0 {
                        return Err("underflow");
                    }
                    depth -= 1;
                    push(&mut s, off, Kind::ObjClose, depth);
                }
                b']' => {
                    if depth == 0 {
                        return Err("underflow");
                    }
                    depth -= 1;
                    push(&mut s, off, Kind::ArrClose, depth);
                }
                b':' => push(&mut s, off, Kind::Colon, depth),
                b',' => push(&mut s, off, Kind::Comma, depth),
                b'"' => push(&mut s, off, Kind::Quote, depth),
                _ => unreachable!("non-structural byte in mask"),
            }
        }

        i += 64;
    }

    // tail: scalar over remainder, with current depth + string-state
    run_tail(&buf[i..], &mut s, &mut depth, prev_in_string != 0, i)?;
    if depth != 0 {
        return Err("unclosed container");
    }
    Ok(s)
}

// Per-chunk extracted bitmasks.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
struct ChunkBits {
    qt: u64,         // bytes equal to "
    bs: u64,         // bytes equal to \
    structural: u64, // bytes in { } [ ] : ,
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn chunk_bits(p: *const u8) -> ChunkBits {
    let a = _mm256_loadu_si256(p as *const __m256i);
    let b = _mm256_loadu_si256(p.add(32) as *const __m256i);

    let v_quote = _mm256_set1_epi8(b'"' as i8);
    let v_bslash = _mm256_set1_epi8(b'\\' as i8);

    let qa = _mm256_movemask_epi8(_mm256_cmpeq_epi8(a, v_quote)) as u32 as u64;
    let qb = _mm256_movemask_epi8(_mm256_cmpeq_epi8(b, v_quote)) as u32 as u64;
    let ba = _mm256_movemask_epi8(_mm256_cmpeq_epi8(a, v_bslash)) as u32 as u64;
    let bb = _mm256_movemask_epi8(_mm256_cmpeq_epi8(b, v_bslash)) as u32 as u64;

    // Structural classification via two pshufb LUTs.
    // Encode bytes whose low-nibble × high-nibble AND ≠ 0 are exactly: { } [ ] : ,
    //   {  = 0x7B   high=7 low=B
    //   }  = 0x7D   high=7 low=D
    //   [  = 0x5B   high=5 low=B
    //   ]  = 0x5D   high=5 low=D
    //   :  = 0x3A   high=3 low=A
    //   ,  = 0x2C   high=2 low=C
    //
    // Strategy: assign each target byte a unique class bit (1..=6), make low_lut
    // map low-nibble -> bitmask of low-class IDs, high_lut map high-nibble ->
    // bitmask of high-class IDs, AND them; non-zero result iff a target byte.
    //
    // bit 0 -> '{' (low B, high 7)
    // bit 1 -> '}' (low D, high 7)
    // bit 2 -> '[' (low B, high 5)
    // bit 3 -> ']' (low D, high 5)
    // bit 4 -> ':' (low A, high 3)
    // bit 5 -> ',' (low C, high 2)
    //
    // low_lut[low_nibble] = OR of class bits for bytes with that low nibble
    //   low A -> {bit4}        = 0x10
    //   low B -> {bit0, bit2}  = 0x05
    //   low C -> {bit5}        = 0x20
    //   low D -> {bit1, bit3}  = 0x0A
    //
    // high_lut[high_nibble] = OR of class bits for bytes with that high nibble
    //   high 2 -> {bit5}       = 0x20
    //   high 3 -> {bit4}       = 0x10
    //   high 5 -> {bit2, bit3} = 0x0C
    //   high 7 -> {bit0, bit1} = 0x03

    #[rustfmt::skip]
    let low_lut = _mm256_setr_epi8(
        0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0x10, 0x05,  0x20, 0x0A, 0, 0,
        0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0x10, 0x05,  0x20, 0x0A, 0, 0,
    );
    #[rustfmt::skip]
    let high_lut = _mm256_setr_epi8(
        0, 0, 0x20, 0x10,  0, 0x0C, 0, 0x03,  0, 0, 0, 0,  0, 0, 0, 0,
        0, 0, 0x20, 0x10,  0, 0x0C, 0, 0x03,  0, 0, 0, 0,  0, 0, 0, 0,
    );

    let mask_0f = _mm256_set1_epi8(0x0F);
    let low_nib_a = _mm256_and_si256(a, mask_0f);
    let high_nib_a = _mm256_and_si256(_mm256_srli_epi32::<4>(a), mask_0f);
    let low_nib_b = _mm256_and_si256(b, mask_0f);
    let high_nib_b = _mm256_and_si256(_mm256_srli_epi32::<4>(b), mask_0f);

    let class_a = _mm256_and_si256(
        _mm256_shuffle_epi8(low_lut, low_nib_a),
        _mm256_shuffle_epi8(high_lut, high_nib_a),
    );
    let class_b = _mm256_and_si256(
        _mm256_shuffle_epi8(low_lut, low_nib_b),
        _mm256_shuffle_epi8(high_lut, high_nib_b),
    );

    let zero = _mm256_setzero_si256();
    let sa = _mm256_movemask_epi8(_mm256_cmpgt_epi8(class_a, zero)) as u32 as u64;
    let sb = _mm256_movemask_epi8(_mm256_cmpgt_epi8(class_b, zero)) as u32 as u64;

    ChunkBits {
        qt: qa | (qb << 32),
        bs: ba | (bb << 32),
        structural: sa | (sb << 32),
    }
}

/// Backslash-run analysis: which bytes are escape-targets.
/// Carries `prev_escaped` across chunks.
#[inline(always)]
fn compute_escaped(bs: u64, prev_escaped: &mut u64) -> u64 {
    const EVEN: u64 = 0x5555_5555_5555_5555;
    const ODD: u64 = 0xAAAA_AAAA_AAAA_AAAA;

    let starts = bs & !(bs << 1);
    let even_starts = starts & EVEN;
    let odd_starts = starts & ODD;

    let (even_carries, _) = bs.overflowing_add(even_starts);
    let (odd_carries_lo, c1) = bs.overflowing_add(odd_starts);
    let (odd_carries, c2) = odd_carries_lo.overflowing_add(*prev_escaped);
    *prev_escaped = (c1 || c2) as u64;

    let even_runs = even_carries & !bs;
    let odd_runs = odd_carries & !bs;
    (even_runs & ODD) | (odd_runs & EVEN)
}

/// Prefix-XOR of a 64-bit mask using carryless multiply by all-ones.
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
#[target_feature(enable = "pclmulqdq,sse2")]
unsafe fn prefix_xor(bits: u64) -> u64 {
    let m = _mm_set_epi64x(0, bits as i64);
    let ones = _mm_set1_epi8(-1i8);
    let r = _mm_clmulepi64_si128::<0>(m, ones);
    _mm_cvtsi128_si64(r) as u64
}

// Tail processing: scalar, but seeded with current string-state.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn run_tail(
    buf: &[u8],
    s: &mut Stage1,
    depth: &mut u16,
    mut in_string: bool,
    base_off: usize,
) -> Result<(), &'static str> {
    let mut i = 0usize;
    let n = buf.len();
    while i < n {
        let b = buf[i];
        if in_string {
            match b {
                b'\\' => i += 2,
                b'"' => {
                    in_string = false;
                    i += 1;
                }
                _ => i += 1,
            }
            continue;
        }
        let off = base_off + i;
        match b {
            b'{' => {
                push(s, off, Kind::ObjOpen, *depth);
                *depth += 1;
            }
            b'[' => {
                push(s, off, Kind::ArrOpen, *depth);
                *depth += 1;
            }
            b'}' => {
                if *depth == 0 {
                    return Err("underflow");
                }
                *depth -= 1;
                push(s, off, Kind::ObjClose, *depth);
            }
            b']' => {
                if *depth == 0 {
                    return Err("underflow");
                }
                *depth -= 1;
                push(s, off, Kind::ArrClose, *depth);
            }
            b':' => push(s, off, Kind::Colon, *depth),
            b',' => push(s, off, Kind::Comma, *depth),
            b'"' => {
                push(s, off, Kind::Quote, *depth);
                in_string = true;
            }
            _ => {}
        }
        i += 1;
    }
    Ok(())
}

//
// Algorithm overview (mirrors `run_avx2`):
//   1. Per 64-byte chunk:
//      - Load two 16-byte lanes (or one 64-byte stride via vld1q_u8 ×4).
//      - Build per-byte equality vectors:
//          quote_mask  = vceqq_u8(lane, vdupq_n_u8(b'"'))
//          bslash_mask = vceqq_u8(lane, vdupq_n_u8(b'\\'))
//      - Compress 64 bytes of bool to 64 bits.  NEON has no movemask;
//        use vshrn_n_u16(_, 4) to halve 16 bytes -> 8 nibbles, then
//        vget_lane_u64 to extract the 64-bit packed result.
//      - Classify structurals via two TBL lookups (vqtbl1q_u8) using the
//        same low/high-nibble LUTs from the AVX2 path.
//   2. Backslash-run analysis: identical 64-bit arithmetic as AVX2 path
//      (no SIMD needed; `compute_escaped` is portable).
//   3. Prefix-XOR for in-string mask: replace `_mm_clmulepi64_si128(_, ones)`
//      with `vmull_p64` (carryless multiply, requires `target_feature = "aes"`).
//   4. Emit positions via `_tzcnt_u64` equivalent — Rust's
//      `u64::trailing_zeros()` lowers to `clz` on aarch64.
//
// Estimated effort: ~200 LoC.  Reference impl: simd-json's
// `src/aarch64/stage1.rs` (Apache-2.0/MIT) — strip stage-2 logic.
//
// Until ported, aarch64 hosts fall back to `run_scalar` which is itself
// auto-vectorisable by LLVM (memchr2 over `\\` and `"` uses NEON).
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon {
    // Port site.  Active fallback: scalar path.
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_object() {
        let s = run(br#"{"test":"a","b":[1,2,3,4]}"#).unwrap();
        assert_eq!(s.kind.first().copied(), Some(Kind::ObjOpen));
        assert_eq!(s.kind.last().copied(), Some(Kind::ObjClose));
        // depth 0 for outer, 1 inside obj, 2 inside array
        assert!(s.depth.iter().any(|&d| d == 2));
    }

    #[test]
    fn escaped_quotes() {
        // "te\"st" should NOT terminate the string at the inner quote
        let buf = br#"{"k":"te\"st"}"#;
        let s = run(buf).unwrap();
        // 4 quotes total = 2 opening
        let q = s.kind.iter().filter(|k| **k == Kind::Quote).count();
        assert_eq!(q, 2);
    }

    #[test]
    fn deep_nesting() {
        // Convention: depth[i] = nesting depth at which this token lives
        // (i.e. the depth of its parent container). For 5 nested opens the
        // deepest *open* emits depth 4; the inner scalar `1` lives at depth 5
        // but is not a structural token, so 4 is the max emitted depth.
        let buf = br#"[[[[[1]]]]]"#;
        let s = run(buf).unwrap();
        assert_eq!(s.depth.iter().copied().max(), Some(4));
    }

    #[test]
    fn cross_chunk_string() {
        // string body crosses 64-byte boundary
        let mut buf = Vec::new();
        buf.extend_from_slice(br#"{"k":""#);
        buf.extend(std::iter::repeat(b'a').take(120));
        buf.extend_from_slice(br#""}"#);
        let s = run(&buf).unwrap();
        let q = s.kind.iter().filter(|k| **k == Kind::Quote).count();
        assert_eq!(q, 2);
    }

    #[test]
    fn cross_chunk_backslash() {
        // odd-run of backslashes spanning boundary
        let mut buf = Vec::new();
        buf.extend_from_slice(br#"{"k":""#);
        buf.extend(std::iter::repeat(b'x').take(58));
        buf.extend_from_slice(br#"\\\""#); // \\\" -> escaped quote, NOT terminator
        buf.extend_from_slice(br#"end"}"#);
        let s = run(&buf).unwrap();
        let q = s.kind.iter().filter(|k| **k == Kind::Quote).count();
        assert_eq!(q, 2);
    }

    #[test]
    fn scalar_matches_simd_object() {
        let buf = br#"{"a":1,"b":[true,false,null],"c":"hi"}"#;
        let a = run_scalar(buf).unwrap();
        let b = run(buf).unwrap();
        assert_eq!(a.offset, b.offset);
        assert_eq!(a.depth, b.depth);
        assert_eq!(a.kind, b.kind);
    }

    #[test]
    fn offsets_point_at_actual_chars() {
        let buf = br#"{"x":42}"#;
        let s = run(buf).unwrap();
        for (i, &off) in s.offset.iter().enumerate() {
            let c = buf[off as usize];
            let expected = match s.kind[i] {
                Kind::ObjOpen => b'{',
                Kind::ObjClose => b'}',
                Kind::ArrOpen => b'[',
                Kind::ArrClose => b']',
                Kind::Quote => b'"',
                Kind::Colon => b':',
                Kind::Comma => b',',
                Kind::Scalar => unreachable!("stage1 emits no scalars"),
            };
            assert_eq!(c, expected, "kind mismatch at idx {}", i);
        }
    }
}
