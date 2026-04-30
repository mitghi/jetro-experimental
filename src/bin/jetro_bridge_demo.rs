//! Demonstrates how jetro-experimental's structural index + Mison key bitmaps
//! would plug into a jetro-core-style architecture (TapeData + TapeView).
//!
//! Build/run:
//!   cargo run --bin jetro_bridge_demo --features simd-json --quiet
//!
//! No dependency on jetro-core itself. We parse with simd-json directly to
//! match jetro's `TapeData` layout (1 entry per Static/String/Object/Array;
//! NO close markers — children flow inline; len/count drives navigation).

use jetro_experimental::keys::{value_for_key, KeyBitmaps};
use jetro_experimental::stage1::Kind;
use jetro_experimental::StructIndex;

use simd_json::{Node, StaticNode};

fn main() {
    let src_str = r#"{"a":{"x":"test"},"b":{"x":"nope"},"c":{"y":"test"},"items":[1,2,3,4]}"#;
    println!("input: {}", src_str);
    println!();

    // --- 1. Build jetro-experimental's StructIndex + KeyBitmaps over raw bytes ---
    let buf = src_str.as_bytes();
    let idx = StructIndex::build(buf).expect("structural index");
    let kb = KeyBitmaps::build(&idx, buf);

    println!("interned keys: {:?}", kb.dict);
    println!();

    // --- 2. Parse the same bytes with simd-json's tape (mirrors jetro's TapeData) ---
    // simd-json mutates input in place during escape decoding.
    let mut owned = src_str.as_bytes().to_vec();
    let tape = simd_json::to_tape(&mut owned).expect("simd-json parse");

    println!("simd-json tape (length {}):", tape.0.len());
    for (i, n) in tape.0.iter().enumerate() {
        println!("  [{:>2}] {:?}", i, n);
    }
    println!();

    // --- 3. Build a "jetro-compatible" tape_of[] mapping ---
    // jetro-core's TapeData.nodes layout = simd-json Node sequence (no Closes).
    // Our default tape_of allocates a slot for ObjClose/ArrClose; recompute
    // a compat variant that skips those.
    let tape_of_compat = build_tape_of_jetro_compat(&idx);
    println!("our_tok -> jetro_tape_idx (compat):");
    for i in 0..idx.stage1.len() {
        let off = idx.stage1.offset[i];
        let kind = idx.stage1.kind[i];
        let t = tape_of_compat[i];
        let t_str = if t == u32::MAX { "-".into() } else { format!("{}", t) };
        println!(
            "  tok={:>2} off={:>2} kind={:<10?}  jetro_tape={}",
            i, off, kind, t_str
        );
    }
    println!();

    // --- 4. Use case A — Mison-style $..find(x == "test") ---
    println!("=== A. $..find(x == \"test\")  via Mison key bitmap ===");
    println!();
    for k_tok in kb.find_key("x", None) {
        let v_tok = value_for_key(&idx, k_tok).unwrap();
        let v_off = idx.stage1.offset[v_tok as usize] as usize;

        // Probe value bytes directly for "test" (skip materialisation).
        let v_bytes = value_byte_slice(buf, v_off, idx.stage1.kind[v_tok as usize]);
        let is_hit = v_bytes == b"\"test\"";

        if !is_hit {
            continue;
        }

        // Enclosing object in our stage1 space.
        let obj_tok = idx.parent[k_tok as usize];
        let obj_tape_idx = tape_of_compat[obj_tok as usize] as usize;

        // Materialise via simd-json's tape — this is what TapeView::materialize
        // does in jetro-core, just dressed in our types.
        let mat = materialize_node(&tape, obj_tape_idx);

        println!(
            "HIT: key_tok={} value_tok={} -> jetro_tape_node[{}]:",
            k_tok, v_tok, obj_tape_idx
        );
        println!("     materialised: {}", mat);
        println!();
    }

    // --- 5. Use case B — byte position -> simd-json Node ---
    println!("=== B. byte position -> simd-json tape Node ===");
    println!();
    // Pick a few byte offsets and walk to enclosing container then to tape.
    for &pos in &[3u32, 9, 11, 22, 60, 62] {
        let cont = match idx.container_at(pos) {
            Some(t) => t,
            None => {
                println!("  pos={:<2}  (no container)", pos);
                continue;
            }
        };
        let tape_idx = tape_of_compat[cont as usize];
        let parents: Vec<u32> = idx
            .parent_chain_tokens(cont)
            .map(|t| tape_of_compat[t as usize])
            .filter(|&t| t != u32::MAX)
            .collect();

        let kind = idx.stage1.kind[cont as usize];
        let node = &tape.0[tape_idx as usize];
        println!(
            "  pos={:<2} -> stage1_tok={:<2} ({:?}, byte_off={}) -> jetro_tape[{}] = {:?}",
            pos, cont, kind, idx.stage1.offset[cont as usize], tape_idx, node
        );
        println!("           parent jetro_tape chain (innermost->root): {:?}", parents);
    }
    println!();

    // --- 6. Use case C — popcount-only count of an aggregate ---
    println!("=== C. count_if(x exists) ===");
    let n = kb.find_key("x", None).len();
    println!("  number of objects with key 'x' = {}  (no value walk, pure bitmap popcount)", n);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Like `index::build_tape_of` but mirrors jetro-core's `TapeData.nodes`:
/// no entry for ObjClose/ArrClose. After this, our token idx maps cleanly
/// onto simd-json `Node` indices (modulo simd-json version pinning).
fn build_tape_of_jetro_compat(idx: &StructIndex) -> Vec<u32> {
    let n = idx.stage1.len();
    let mut tape_of = vec![u32::MAX; n];
    let mut t: u32 = 0;
    for i in 0..n {
        match idx.stage1.kind[i] {
            // 1 entry per opener / scalar / string in simd-json's tape; closes skipped.
            Kind::ObjOpen | Kind::ArrOpen | Kind::Quote | Kind::Scalar => {
                tape_of[i] = t;
                t += 1;
            }
            _ => {}
        }
    }
    tape_of
}

/// Slice the raw byte span of a value token (Quote or Scalar). For strings
/// the slice INCLUDES the surrounding quotes — handy for direct `b"\"test\""`
/// equality probes that skip escape decoding for the common no-escape case.
fn value_byte_slice<'a>(buf: &'a [u8], off: usize, kind: Kind) -> &'a [u8] {
    match kind {
        Kind::Quote => {
            // Walk to closing quote, honouring escapes.
            let mut i = off + 1;
            while i < buf.len() {
                match buf[i] {
                    b'\\' => i += 2,
                    b'"' => {
                        return &buf[off..i + 1];
                    }
                    _ => i += 1,
                }
            }
            &buf[off..]
        }
        Kind::Scalar => {
            let mut i = off;
            while i < buf.len() {
                match buf[i] {
                    b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r' => break,
                    _ => i += 1,
                }
            }
            &buf[off..i]
        }
        _ => &buf[off..off],
    }
}

/// Materialise the simd-json node at index `tape_idx` to a Rust string.
/// Mirrors `value_view::TapeView::materialize` in jetro-core.
fn materialize_node(tape: &simd_json::tape::Tape<'_>, tape_idx: usize) -> String {
    let mut idx = tape_idx;
    walk(tape, &mut idx)
}

fn walk(tape: &simd_json::tape::Tape<'_>, idx: &mut usize) -> String {
    let here = tape.0[*idx];
    *idx += 1;
    match here {
        Node::Static(StaticNode::Null) => "null".into(),
        Node::Static(StaticNode::Bool(b)) => format!("{}", b),
        Node::Static(StaticNode::I64(n)) => format!("{}", n),
        Node::Static(StaticNode::U64(n)) => format!("{}", n),
        Node::Static(StaticNode::F64(f)) => format!("{}", f),
        Node::String(s) => format!("\"{}\"", s),
        Node::Array { len, .. } => {
            let mut out = String::from("[");
            for i in 0..len {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&walk(tape, idx));
            }
            out.push(']');
            out
        }
        Node::Object { len, .. } => {
            let mut out = String::from("{");
            for i in 0..len {
                if i > 0 {
                    out.push(',');
                }
                let key = match tape.0[*idx] {
                    Node::String(k) => k,
                    _ => "?",
                };
                *idx += 1;
                out.push('"');
                out.push_str(key);
                out.push_str("\":");
                out.push_str(&walk(tape, idx));
            }
            out.push('}');
            out
        }
    }
}
