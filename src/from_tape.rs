//! Build a `StructIndex` directly from a parsed `simd_json::tape::Tape`,
//! using the vendored simd-json fork's `Buffers::structural_indexes()`
//! accessor (added in `vendor-simd-json/`).
//!
//! Why this exists: jetro-core already calls `simd_json::to_tape` once to
//! build its `TapeData`.  Walking the tape a *second* time inside
//! jetro-experimental to build columns wastes ~3-5ms / MB.  This module
//! lets the caller emit both data structures in one walk.
//!
//! Two entry points:
//!
//!  - [`build_from_tape_and_structurals`] — caller supplies the parsed
//!    `Tape`, the original bytes, and the structural-index slice from
//!    `Buffers::structural_indexes()`.  Single fused walk produces a
//!    full `StructuralIndex` (offset/kind/depth/parent/close_of/tape_of)
//!    plus key bitmaps.
//!
//!  - [`parse_with_index`] — full convenience: parses bytes into a
//!    Tape via the vendored simd-json fork and builds the index in one
//!    call.  Useful for jetro-experimental users that don't already
//!    have a Tape.
//!
//! Tape numbering convention here matches simd-json's `Tape.0` index
//! exactly: 1 entry per Object/Array/String/Static node, NO close entries.
//! `tape_of[i]` for a stage1 token is the index into `Tape.0` if the token
//! is value-bearing, else `u32::MAX`.

use std::collections::HashMap;

use croaring::Bitmap;
use simd_json::{Buffers, Node};

use crate::index::StructIndex;
use crate::keys::{KeyBitmaps, Role};
use crate::stage1::{Kind, Stage1};

/// Result of a fused tape-walk: tape, structural index, key bitmaps.
pub struct ParsedWithIndex {
    /// Owning byte buffer.  Strings borrowed by the index/keys reference this.
    pub bytes: Vec<u8>,
    /// Stage1 + parent/close_of + tape_of columns.
    pub index: StructIndex,
    /// Mison key bitmaps.  Built during the same walk.
    pub keys: KeyBitmaps,
}

/// Parse `bytes` and build the structural index in one fused walk.
///
/// Internally:
///   1. `simd_json::to_tape_with_buffers` — runs stage-1 SIMD + stage-2 walk.
///   2. `Buffers::structural_indexes()` (fork accessor) — gives us byte
///      offsets of every structural char from stage-1.
///   3. Single zip of `tape.0.iter()` against the structural-indexes slice
///      → emits all jetro-experimental columns inline.
///
/// `bytes` is consumed (simd-json mutates input in place during escape
/// decode).  Returned as part of `ParsedWithIndex` so the caller can reuse
/// it as the owning buffer for borrowed `&str` slices.
pub fn parse_with_index(mut bytes: Vec<u8>) -> Result<ParsedWithIndex, String> {
    let mut buffers = Buffers::new(bytes.len() + 256);

    // SAFETY: we need shared access to `bytes` while `tape` is alive (for
    // structural-byte lookup during the walk).  The compiler reads the
    // surface signature of `to_tape_with_buffers` as a long-lived `&mut`,
    // but in practice simd-json only mutates `bytes` *during* that call
    // (escape decoding); afterwards the buffer is read-only and all tape
    // string slices are immutable views into it.  We therefore hand the
    // builder a fresh `&[u8]` materialised from the same allocation.
    let bytes_ptr = bytes.as_ptr();
    let bytes_len = bytes.len();

    let (index, keys) = {
        let tape = simd_json::to_tape_with_buffers(&mut bytes, &mut buffers)
            .map_err(|e| e.to_string())?;
        let structurals: &[u32] = buffers.structural_indexes();
        let bytes_view: &[u8] = unsafe { std::slice::from_raw_parts(bytes_ptr, bytes_len) };
        build_from_tape_and_structurals(&tape, bytes_view, structurals)?
    };

    Ok(ParsedWithIndex {
        bytes,
        index,
        keys,
    })
}

/// Build `StructIndex` + `KeyBitmaps` from a borrowed tape + bytes + structurals.
///
/// `structurals` must be the slice returned by `Buffers::structural_indexes()`
/// after `to_tape_with_buffers`.  Mismatched (tape, structurals) pairs produce
/// nonsense indices.
pub fn build_from_tape_and_structurals<'input>(
    tape: &simd_json::tape::Tape<'input>,
    bytes: &[u8],
    structurals: &[u32],
) -> Result<(StructIndex, KeyBitmaps), String> {
    let n_nodes = tape.0.len();

    // Pre-allocate stage1 columns. Worst-case length = n_nodes (opens/strings/scalars)
    // + close-tokens we synthesise as we walk = up to 2 * n_nodes.
    let mut s1 = Stage1 {
        offset: Vec::with_capacity(n_nodes * 2),
        kind: Vec::with_capacity(n_nodes * 2),
        depth: Vec::with_capacity(n_nodes * 2),
    };

    // Sidecar columns built in lockstep.
    let mut parent: Vec<i32> = Vec::with_capacity(n_nodes * 2);
    let mut close_of: Vec<i32> = Vec::with_capacity(n_nodes * 2);
    let mut tape_of: Vec<u32> = Vec::with_capacity(n_nodes * 2);

    // Mison columns.
    let mut role: Vec<Role> = Vec::with_capacity(n_nodes * 2);
    let mut key_id_of: Vec<u32> = Vec::with_capacity(n_nodes * 2);
    let mut dict: Vec<std::sync::Arc<str>> = Vec::new();
    let mut by_name: HashMap<std::sync::Arc<str>, u32> = HashMap::new();
    let mut bitmaps: Vec<Bitmap> = Vec::new();

    // State across the walk.
    let mut depth_cur: u16 = 0;
    let mut stack: Vec<u32> = Vec::with_capacity(16); // tok indices of open containers
    // For each open container on stack: how many of its `len` entries (object
    // pairs OR array elements) we've consumed.  Used to detect close points
    // since simd-json tape doesn't emit close nodes.
    let mut consumed: Vec<u32> = Vec::with_capacity(16);
    let mut limit: Vec<u32> = Vec::with_capacity(16); // total entries before close
    let mut struct_cursor: usize = 0;
    let mut next_is_key = false; // inside-object state

    // Helper: advance struct_cursor until we hit a byte equal to one of `targets`.
    // Returns the offset of the first match.  Bytes is the original input.
    let find_next_struct = |cursor: &mut usize, structurals: &[u32], bytes: &[u8],
                            targets: &[u8]|
     -> Option<u32> {
        while *cursor < structurals.len() {
            let off = structurals[*cursor] as usize;
            *cursor += 1;
            if off >= bytes.len() {
                return None;
            }
            if targets.iter().any(|&t| t == bytes[off]) {
                return Some(off as u32);
            }
        }
        None
    };

    let intern = |name: &str,
                  dict: &mut Vec<std::sync::Arc<str>>,
                  by_name: &mut HashMap<std::sync::Arc<str>, u32>,
                  bitmaps: &mut Vec<Bitmap>|
     -> u32 {
        if let Some(&id) = by_name.get(name) {
            return id;
        }
        let id = dict.len() as u32;
        let key: std::sync::Arc<str> = std::sync::Arc::from(name);
        dict.push(std::sync::Arc::clone(&key));
        by_name.insert(key, id);
        bitmaps.push(Bitmap::new());
        id
    };

    for (tape_idx, node) in tape.0.iter().enumerate() {
        // Before pushing the next opener / value, close any containers whose
        // entry count is satisfied (cascade-close for `]}` runs).
        while let Some(&cnt) = consumed.last() {
            let lim = *limit.last().unwrap();
            if cnt < lim {
                break;
            }
            // Container at top of stack is full → emit close.
            let opener_tok = stack.pop().unwrap();
            consumed.pop();
            limit.pop();
            depth_cur -= 1;
            let close_targets: &[u8] = match s1.kind[opener_tok as usize] {
                Kind::ObjOpen => b"}",
                Kind::ArrOpen => b"]",
                _ => unreachable!("non-container on stack"),
            };
            let close_off = find_next_struct(&mut struct_cursor, structurals, bytes, close_targets)
                .ok_or_else(|| "unmatched close".to_string())?;
            let tok_idx = s1.offset.len() as u32;
            s1.offset.push(close_off);
            s1.kind.push(if close_targets[0] == b'}' {
                Kind::ObjClose
            } else {
                Kind::ArrClose
            });
            s1.depth.push(depth_cur);
            parent.push(parent[opener_tok as usize]);
            close_of.push(-1);
            close_of[opener_tok as usize] = tok_idx as i32;
            tape_of.push(u32::MAX); // close has no tape entry
            role.push(Role::None);
            key_id_of.push(u32::MAX);
            // Tell parent (if any) that this container counted as one entry.
            if let Some(top) = consumed.last_mut() {
                *top += 1;
            }
            // Restore parent's expected-next-token state so subsequent values
            // are classified correctly.
            match stack.last().map(|t| s1.kind[*t as usize]) {
                Some(Kind::ObjOpen) => next_is_key = true,
                Some(Kind::ArrOpen) => next_is_key = false,
                _ => {}
            }
        }

        match *node {
            Node::Object { len, .. } => {
                let off = find_next_struct(&mut struct_cursor, structurals, bytes, b"{")
                    .ok_or_else(|| "missing `{` in structurals".to_string())?;
                let tok_idx = s1.offset.len() as u32;
                let p = stack.last().copied().map(|x| x as i32).unwrap_or(-1);
                let inside_obj_parent = stack
                    .last()
                    .map(|t| matches!(s1.kind[*t as usize], Kind::ObjOpen))
                    .unwrap_or(false);
                let role_here = if inside_obj_parent && next_is_key {
                    Role::Value
                } else if stack
                    .last()
                    .map(|t| matches!(s1.kind[*t as usize], Kind::ArrOpen))
                    .unwrap_or(false)
                {
                    Role::Elem
                } else if inside_obj_parent {
                    Role::Value
                } else {
                    Role::None
                };

                s1.offset.push(off);
                s1.kind.push(Kind::ObjOpen);
                s1.depth.push(depth_cur);
                parent.push(p);
                close_of.push(-1);
                tape_of.push(tape_idx as u32);
                role.push(role_here);
                key_id_of.push(u32::MAX);

                stack.push(tok_idx);
                consumed.push(0);
                // For an Object, each entry consumes 2 nodes: key (String) +
                // value (any). simd-json's `len` here is number of pairs.
                limit.push(len as u32);
                depth_cur += 1;
                next_is_key = true;
            }

            Node::Array { len, .. } => {
                let off = find_next_struct(&mut struct_cursor, structurals, bytes, b"[")
                    .ok_or_else(|| "missing `[` in structurals".to_string())?;
                let tok_idx = s1.offset.len() as u32;
                let p = stack.last().copied().map(|x| x as i32).unwrap_or(-1);
                let role_here = match stack.last() {
                    Some(t) if matches!(s1.kind[*t as usize], Kind::ArrOpen) => Role::Elem,
                    Some(t) if matches!(s1.kind[*t as usize], Kind::ObjOpen) => Role::Value,
                    _ => Role::None,
                };
                s1.offset.push(off);
                s1.kind.push(Kind::ArrOpen);
                s1.depth.push(depth_cur);
                parent.push(p);
                close_of.push(-1);
                tape_of.push(tape_idx as u32);
                role.push(role_here);
                key_id_of.push(u32::MAX);

                stack.push(tok_idx);
                consumed.push(0);
                limit.push(len as u32);
                depth_cur += 1;
                next_is_key = false;
            }

            Node::String(s) => {
                // Byte offset of the opening quote — derive from slice ptr if
                // it's inside `bytes`, else fall back to scanning structurals.
                let off = if let Some(off) = string_offset_in_bytes(bytes, s) {
                    // Skip cursor past the opening-quote structural for this string.
                    advance_cursor_past(&mut struct_cursor, structurals, off);
                    off
                } else {
                    // String body lives outside the input buffer (escape-decode
                    // moved it).  Scan structurals for the next " at our cursor.
                    find_next_struct(&mut struct_cursor, structurals, bytes, b"\"")
                        .ok_or_else(|| "missing `\"` in structurals".to_string())?
                };
                let tok_idx = s1.offset.len() as u32;
                let parent_tok = stack.last().copied().map(|x| x as i32).unwrap_or(-1);
                let inside_obj = stack
                    .last()
                    .map(|t| matches!(s1.kind[*t as usize], Kind::ObjOpen))
                    .unwrap_or(false);
                let inside_arr = stack
                    .last()
                    .map(|t| matches!(s1.kind[*t as usize], Kind::ArrOpen))
                    .unwrap_or(false);

                let role_here = if inside_obj && next_is_key {
                    Role::Key
                } else if inside_obj {
                    Role::Value
                } else if inside_arr {
                    Role::Elem
                } else {
                    Role::Value
                };

                s1.offset.push(off);
                s1.kind.push(Kind::Quote);
                s1.depth.push(depth_cur);
                parent.push(parent_tok);
                close_of.push(-1);
                tape_of.push(tape_idx as u32);
                role.push(role_here);

                if role_here == Role::Key {
                    let id = intern(s, &mut dict, &mut by_name, &mut bitmaps);
                    bitmaps[id as usize].add(tok_idx);
                    key_id_of.push(id);
                    next_is_key = false;
                    // Step past the `:` that separates key from value so the
                    // scalar/string value branch reads the correct structural.
                    if struct_cursor < structurals.len() {
                        let off = structurals[struct_cursor] as usize;
                        if off < bytes.len() && bytes[off] == b':' {
                            struct_cursor += 1;
                        }
                    }
                } else {
                    key_id_of.push(u32::MAX);
                    if inside_obj {
                        // Value just consumed → pair complete; next entry is a key.
                        next_is_key = true;
                        if let Some(top) = consumed.last_mut() {
                            *top += 1;
                        }
                    } else if inside_arr {
                        if let Some(top) = consumed.last_mut() {
                            *top += 1;
                        }
                    }
                }
            }

            Node::Static(_) => {
                // Numbers / true / false / null. Byte offset = next non-WS byte
                // after the previous structural (`,` `:` `[` etc).  Simplest:
                // scan structurals for next `,` `[` or `:` and step past WS.
                let off = scan_scalar_offset(bytes, structurals, &mut struct_cursor)
                    .ok_or_else(|| "scalar offset".to_string())?;
                let tok_idx = s1.offset.len() as u32;
                let parent_tok = stack.last().copied().map(|x| x as i32).unwrap_or(-1);
                let inside_obj = stack
                    .last()
                    .map(|t| matches!(s1.kind[*t as usize], Kind::ObjOpen))
                    .unwrap_or(false);
                let inside_arr = stack
                    .last()
                    .map(|t| matches!(s1.kind[*t as usize], Kind::ArrOpen))
                    .unwrap_or(false);
                let role_here = if inside_arr {
                    Role::Elem
                } else if inside_obj {
                    Role::Value
                } else {
                    Role::Value
                };
                s1.offset.push(off);
                s1.kind.push(Kind::Scalar);
                s1.depth.push(depth_cur);
                parent.push(parent_tok);
                close_of.push(-1);
                tape_of.push(tape_idx as u32);
                role.push(role_here);
                key_id_of.push(u32::MAX);
                if inside_obj {
                    next_is_key = true;
                    if let Some(top) = consumed.last_mut() {
                        *top += 1;
                    }
                } else if inside_arr {
                    if let Some(top) = consumed.last_mut() {
                        *top += 1;
                    }
                }
            }
        }
    }

    // Final close-cascade for any containers still open.
    while let Some(opener_tok) = stack.pop() {
        consumed.pop();
        limit.pop();
        depth_cur -= 1;
        let close_targets: &[u8] = match s1.kind[opener_tok as usize] {
            Kind::ObjOpen => b"}",
            Kind::ArrOpen => b"]",
            _ => unreachable!(),
        };
        let close_off = find_next_struct(&mut struct_cursor, structurals, bytes, close_targets)
            .ok_or_else(|| "trailing close not found".to_string())?;
        let tok_idx = s1.offset.len() as u32;
        s1.offset.push(close_off);
        s1.kind.push(if close_targets[0] == b'}' {
            Kind::ObjClose
        } else {
            Kind::ArrClose
        });
        s1.depth.push(depth_cur);
        parent.push(parent[opener_tok as usize]);
        close_of.push(-1);
        close_of[opener_tok as usize] = tok_idx as i32;
        tape_of.push(u32::MAX);
        role.push(Role::None);
        key_id_of.push(u32::MAX);
    }

    // Optimise key bitmaps.
    for b in bitmaps.iter_mut() {
        b.run_optimize();
    }

    // Build depth bitmaps from final depth column.
    let max_depth = s1.depth.iter().copied().max().unwrap_or(0) as usize;
    let mut depth_bitmaps: Vec<Bitmap> = (0..=max_depth).map(|_| Bitmap::new()).collect();
    for (i, &d) in s1.depth.iter().enumerate() {
        depth_bitmaps[d as usize].add(i as u32);
    }
    for b in depth_bitmaps.iter_mut() {
        b.run_optimize();
    }

    let n_tokens = s1.offset.len();
    let index = StructIndex {
        stage1: s1,
        parent,
        close_of,
        tape_of,
    };
    let keys = KeyBitmaps {
        dict,
        by_name,
        bitmaps,
        depth_bitmaps,
        role,
        key_id_of,
        n_tokens,
    };
    Ok((index, keys))
}

/// Map a `&str` from a `Node::String` back to an offset in the original bytes.
/// Returns None if the slice is not a sub-slice of `bytes` (escape-decoded
/// strings are written into a separate scratch buffer).
fn string_offset_in_bytes(bytes: &[u8], s: &str) -> Option<u32> {
    let base = bytes.as_ptr() as usize;
    let p = s.as_ptr() as usize;
    if p >= base && p < base + bytes.len() {
        // Offset of the *body*; we want the opening quote one byte earlier.
        let body = (p - base) as u32;
        if body == 0 {
            None
        } else {
            Some(body - 1)
        }
    } else {
        None
    }
}

fn advance_cursor_past(cursor: &mut usize, structurals: &[u32], target_off: u32) {
    while *cursor < structurals.len() && structurals[*cursor] < target_off {
        *cursor += 1;
    }
    // Step past the matching opener.
    if *cursor < structurals.len() && structurals[*cursor] == target_off {
        *cursor += 1;
    }
}

/// Read the byte offset of the next scalar literal directly from the
/// structurals slice.  simd-json's stage-1 emits an entry pointing at
/// the first byte of every value (number/true/false/null included), so
/// we just consume `structurals[cursor]` and step past it.
fn scan_scalar_offset(
    bytes: &[u8],
    structurals: &[u32],
    cursor: &mut usize,
) -> Option<u32> {
    let _ = bytes;
    if *cursor >= structurals.len() {
        return None;
    }
    let off = structurals[*cursor];
    *cursor += 1;
    Some(off)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::stage1::Kind;

    #[test]
    fn smoke_simple_object() {
        let bytes = br#"{"a":1,"b":"hi","c":[1,2,3]}"#.to_vec();
        let parsed = parse_with_index(bytes).unwrap();

        // Key bitmap dict has a, b, c.
        let mut keys: Vec<String> = parsed.keys.dict.iter().map(|s| s.to_string()).collect();
        keys.sort();
        assert_eq!(keys, vec!["a".to_string(), "b".to_string(), "c".to_string()]);

        // tape_of for ObjOpen[0] = 0 (root); ArrOpen for "c" should map to a tape
        // index that's a Node::Array.
        let mut found_arr = false;
        for i in 0..parsed.index.stage1.len() {
            if parsed.index.stage1.kind[i] == Kind::ArrOpen {
                found_arr = true;
                let t = parsed.index.tape_of[i];
                assert_ne!(t, u32::MAX);
            }
        }
        assert!(found_arr);
    }

    #[test]
    fn key_bitmap_lookup() {
        let bytes = br#"{"x":"test","y":1,"x":"again"}"#.to_vec();
        // Note: simd-json may dedupe duplicate keys; for normal input this works.
        let parsed = parse_with_index(bytes).unwrap();
        let xs = parsed.keys.find_key("x", None);
        assert!(!xs.is_empty(), "expected at least one match for 'x'");
        for tok in xs {
            assert_eq!(parsed.keys.role[tok as usize], Role::Key);
        }
    }

    #[test]
    fn tape_of_aligns_with_simd_json_nodes() {
        let bytes = br#"{"a":[1,2,{"b":"v"}],"c":true}"#.to_vec();
        let mut owned = bytes.clone();
        let owned_ptr = owned.as_ptr();
        let owned_len = owned.len();
        let mut buffers = Buffers::new(owned.len() + 256);
        let tape = simd_json::to_tape_with_buffers(&mut owned, &mut buffers).unwrap();
        let structurals: Vec<u32> = buffers.structural_indexes().to_vec();
        // SAFETY: simd-json no longer mutates `owned` after to_tape returns.
        let owned_view: &[u8] = unsafe { std::slice::from_raw_parts(owned_ptr, owned_len) };
        let (idx, _keys) =
            build_from_tape_and_structurals(&tape, owned_view, &structurals).unwrap();

        // Every value-bearing token (Open/String/Scalar) has a non-MAX tape_of.
        for i in 0..idx.stage1.len() {
            let needs_tape = matches!(
                idx.stage1.kind[i],
                Kind::ObjOpen | Kind::ArrOpen | Kind::Quote | Kind::Scalar
            );
            if needs_tape {
                assert_ne!(
                    idx.tape_of[i],
                    u32::MAX,
                    "value-bearing token {} has no tape_of",
                    i
                );
                assert!((idx.tape_of[i] as usize) < tape.0.len());
            } else {
                assert_eq!(idx.tape_of[i], u32::MAX);
            }
        }
    }

    #[test]
    fn close_of_pairs_with_open() {
        let bytes = br#"{"a":[1,2,3]}"#.to_vec();
        let parsed = parse_with_index(bytes).unwrap();
        let s = &parsed.index.stage1;

        // Find ObjOpen and ArrOpen, ensure each has a matching close at greater idx.
        for i in 0..s.len() {
            if matches!(s.kind[i], Kind::ObjOpen | Kind::ArrOpen) {
                let close = parsed.index.close_of[i];
                assert!(close > 0, "open at {} has no close", i);
                assert!((close as usize) < s.len());
                let close_kind = s.kind[close as usize];
                let expected = match s.kind[i] {
                    Kind::ObjOpen => Kind::ObjClose,
                    Kind::ArrOpen => Kind::ArrClose,
                    _ => unreachable!(),
                };
                assert_eq!(close_kind, expected);
            }
        }
    }

    #[test]
    fn structural_indexes_accessor_works() {
        // Minimal fork-accessor smoke test.
        let mut bytes = br#"{"x":1}"#.to_vec();
        let mut buffers = Buffers::new(bytes.len() + 256);
        let _ = simd_json::to_tape_with_buffers(&mut bytes, &mut buffers).unwrap();
        let s = buffers.structural_indexes();
        assert!(!s.is_empty(), "structural_indexes should be populated");
        // First should be 0 (`{`), and indices should be ascending.
        for w in s.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }
}
