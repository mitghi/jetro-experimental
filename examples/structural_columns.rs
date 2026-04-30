//! Demonstrate that `Buffers::structural_indexes()` (the new accessor in
//! the vendored simd-json fork) gives us enough information to compute
//! the byte span (start..end) of every JSON value in a document.
//!
//! Run:
//!   cargo run --release --example structural_columns --features simd-json
//!
//! What we prove:
//!   1. The accessor returns a slice of byte offsets, one per structural
//!      character (`{ } [ ] " : ,`).
//!   2. Pairing offsets with the parsed `Tape` lets us derive:
//!        - object start `{` and matching end `}`
//!        - array start `[` and matching end `]`
//!        - string start `"` and matching end `"`
//!        - scalar (number / bool / null) start and end
//!   3. The byte spans round-trip: slicing the input `&bytes[start..end]`
//!      reproduces the original JSON for that value.

use simd_json::{Buffers, Node};

const DOC: &[u8] = br#"{"name":"alice","age":30,"tags":["x","y","z"],"profile":{"city":"NYC","zip":"10001"},"score":99.5,"active":true,"note":null}"#;

#[derive(Debug, Clone, Copy)]
enum ValueKind {
    Object,
    Array,
    String,
    Scalar,
}

#[derive(Debug, Clone)]
struct Span {
    kind: ValueKind,
    /// Optional key when this value sits at depth 1 inside the root object.
    key: Option<String>,
    start: usize,
    end: usize,
}

fn main() {
    println!("Input ({} bytes):", DOC.len());
    println!("  {}", std::str::from_utf8(DOC).unwrap());
    println!();

    // 1. Parse with `to_tape_with_buffers` so we can reach the buffers afterwards.
    let mut bytes = DOC.to_vec();
    let mut buffers = Buffers::new(bytes.len() + 256);
    let bytes_ptr = bytes.as_ptr();
    let bytes_len = bytes.len();

    let tape = simd_json::to_tape_with_buffers(&mut bytes, &mut buffers).expect("parse");

    // 2. Read the structural_indexes via the new accessor.
    let structurals: &[u32] = buffers.structural_indexes();
    println!("Stage-1 structural indexes ({} entries):", structurals.len());
    print!("  ");
    for (i, &off) in structurals.iter().enumerate() {
        // SAFETY: bytes is the same allocation simd-json parsed; we read
        // it now that to_tape has returned.
        let c = unsafe { *bytes_ptr.add(off as usize) } as char;
        print!("[{i}: {off}={c:?}] ");
        if i % 6 == 5 {
            println!();
            print!("  ");
        }
    }
    println!("\n");

    // 3. Walk the tape + structurals in lockstep to compute byte spans.
    let bytes_view: &[u8] = unsafe { std::slice::from_raw_parts(bytes_ptr, bytes_len) };
    let spans = derive_spans(&tape, bytes_view, structurals);

    // 4. Print byte spans + show that round-tripping via &bytes[start..end] reproduces the value.
    println!("Derived byte spans for every value:");
    for (i, span) in spans.iter().enumerate() {
        let key_label = match &span.key {
            Some(k) => format!("key=\"{k}\" "),
            None => String::new(),
        };
        let raw = std::str::from_utf8(&bytes_view[span.start..span.end]).unwrap_or("<non-utf8>");
        println!(
            "  [{i:>2}] {kind:?} {key}bytes [{s}..{e}) = {raw}",
            kind = span.kind,
            key = key_label,
            s = span.start,
            e = span.end,
            raw = raw,
        );
    }
    println!();

    // 5. Demonstrate practical use: extract the raw bytes of `tags` and `profile`.
    println!("=== Practical: zero-copy passthrough output ===");
    for span in &spans {
        if let Some(key) = &span.key {
            if key == "tags" || key == "profile" {
                let raw = std::str::from_utf8(&bytes_view[span.start..span.end]).unwrap();
                println!("  raw value of \"{key}\": {raw}");
                println!("    span = [{}, {}), length = {}", span.start, span.end, span.end - span.start);
            }
        }
    }
    println!();

    // 6. Byte-position → enclosing object lookup.
    //
    // Use case: external byte index (sourcemap entry, log span, error
    // position, fuzzer crash offset) → which JSON object holds it?
    //
    // Algorithm: among Object spans whose [start, end) covers the position,
    // pick the one with the *largest start* — that's the innermost. Linear
    // scan over spans is fine for any reasonable doc; for huge corpora
    // sort spans by start and binary-search.
    println!("=== Byte-position → enclosing object ===");
    for &pos in &[3u32, 11, 23, 35, 60, 78, 100, 120] {
        let pos = pos as usize;
        let context = &bytes_view[pos..pos.min(bytes_view.len())];
        let preview: String = context.iter().take(8).map(|&b| b as char).collect();
        println!(
            "  byte {pos:>3} ({preview:?}…) → {}",
            describe_enclosing_object(&spans, pos),
        );
    }
}

fn describe_enclosing_object(spans: &[Span], pos: usize) -> String {
    let mut best: Option<&Span> = None;
    for s in spans {
        if !matches!(s.kind, ValueKind::Object) {
            continue;
        }
        if pos >= s.start && pos < s.end {
            // Prefer deeper (later-starting) match.
            if best.map_or(true, |b| s.start > b.start) {
                best = Some(s);
            }
        }
    }
    match best {
        Some(s) => {
            let key_part = match &s.key {
                Some(k) => format!("key=\"{k}\" "),
                None => String::from("(root) "),
            };
            format!(
                "Object {key}span [{a}..{b}) len {n}",
                key = key_part,
                a = s.start,
                b = s.end,
                n = s.end - s.start
            )
        }
        None => String::from("no enclosing object"),
    }
}

/// Walk tape nodes; for each, peek at structurals to derive byte ranges.
fn derive_spans(tape: &simd_json::tape::Tape<'_>, bytes: &[u8], structurals: &[u32]) -> Vec<Span> {
    let mut out: Vec<Span> = Vec::new();
    let mut cursor = 0usize; // index into structurals
    let mut stack: Vec<(ValueKind, usize)> = Vec::new(); // (kind, span_index)
    let mut consumed: Vec<u32> = Vec::new();
    let mut limit: Vec<u32> = Vec::new();
    let mut next_is_key = false;
    let mut last_key: Option<String> = None;

    for node in tape.0.iter() {
        // Cascade-close fully-consumed containers.
        while let Some(&cnt) = consumed.last() {
            if cnt < *limit.last().unwrap() {
                break;
            }
            let (kind, span_idx) = stack.pop().unwrap();
            consumed.pop();
            limit.pop();
            let close_byte = match kind {
                ValueKind::Object => b'}',
                ValueKind::Array => b']',
                _ => unreachable!(),
            };
            let close_off = scan_for(bytes, structurals, &mut cursor, close_byte);
            out[span_idx].end = close_off + 1; // inclusive of close char
            if let Some(top) = consumed.last_mut() {
                *top += 1;
            }
            // The container we just closed counted as one entry of its parent.
            // Restore the parent's expected-next-token state.
            match stack.last().map(|(k, _)| *k) {
                Some(ValueKind::Object) => next_is_key = true,
                Some(ValueKind::Array) => next_is_key = false,
                _ => {}
            }
        }

        // Determine the parent context for this value.
        let parent_kind = stack.last().map(|(k, _)| *k);
        let key_for_this = if matches!(parent_kind, Some(ValueKind::Object)) && !next_is_key {
            last_key.take()
        } else {
            None
        };

        match *node {
            Node::Object { len, .. } => {
                let open_off = scan_for(bytes, structurals, &mut cursor, b'{');
                let span_idx = out.len();
                out.push(Span {
                    kind: ValueKind::Object,
                    key: key_for_this,
                    start: open_off,
                    end: 0, // filled in on close
                });
                stack.push((ValueKind::Object, span_idx));
                consumed.push(0);
                limit.push(len as u32);
                next_is_key = true;
            }
            Node::Array { len, .. } => {
                let open_off = scan_for(bytes, structurals, &mut cursor, b'[');
                let span_idx = out.len();
                out.push(Span {
                    kind: ValueKind::Array,
                    key: key_for_this,
                    start: open_off,
                    end: 0,
                });
                stack.push((ValueKind::Array, span_idx));
                consumed.push(0);
                limit.push(len as u32);
                next_is_key = false;
            }
            Node::String(s) => {
                let start = string_offset(bytes, s).unwrap_or_else(|| {
                    scan_for(bytes, structurals, &mut cursor, b'"')
                });
                // String span: opening quote .. closing quote (inclusive).
                let end = start + s.len() + 2; // 2 quotes
                advance_cursor_to(structurals, &mut cursor, start as u32);

                let inside_obj = matches!(parent_kind, Some(ValueKind::Object));
                if inside_obj && next_is_key {
                    last_key = Some(s.to_string());
                    next_is_key = false;
                    // Key consumed; also step over the `:` that follows so the
                    // next scalar/value branch can anchor correctly.
                    skip_one_of(bytes, structurals, &mut cursor, b':');
                    // Key string itself isn't an emitted "value span" — skip.
                } else {
                    out.push(Span {
                        kind: ValueKind::String,
                        key: key_for_this,
                        start,
                        end,
                    });
                    if inside_obj {
                        next_is_key = true;
                        if let Some(top) = consumed.last_mut() {
                            *top += 1;
                        }
                    } else if matches!(parent_kind, Some(ValueKind::Array)) {
                        if let Some(top) = consumed.last_mut() {
                            *top += 1;
                        }
                    }
                }
            }
            Node::Static(_) => {
                // simd-json's structural index already points at the first byte
                // of every value (number/true/false/null inclusive). So we just
                // read `structurals[cursor]` and use it as the literal's start.
                let start = if cursor < structurals.len() {
                    let off = structurals[cursor] as usize;
                    cursor += 1;
                    off
                } else {
                    bytes.len()
                };
                let end = next_scalar_end(bytes, start);
                out.push(Span {
                    kind: ValueKind::Scalar,
                    key: key_for_this,
                    start,
                    end,
                });
                if matches!(parent_kind, Some(ValueKind::Object)) {
                    next_is_key = true;
                    if let Some(top) = consumed.last_mut() {
                        *top += 1;
                    }
                } else if matches!(parent_kind, Some(ValueKind::Array)) {
                    if let Some(top) = consumed.last_mut() {
                        *top += 1;
                    }
                }
            }
        }
    }

    // Flush remaining opens.
    while let Some((kind, span_idx)) = stack.pop() {
        consumed.pop();
        limit.pop();
        let close_byte = match kind {
            ValueKind::Object => b'}',
            ValueKind::Array => b']',
            _ => unreachable!(),
        };
        let close_off = scan_for(bytes, structurals, &mut cursor, close_byte);
        out[span_idx].end = close_off + 1;
    }

    out
}

/// Advance cursor through structurals until we hit a structural byte equal
/// to `target`. Return its byte offset.
fn scan_for(bytes: &[u8], structurals: &[u32], cursor: &mut usize, target: u8) -> usize {
    while *cursor < structurals.len() {
        let off = structurals[*cursor] as usize;
        *cursor += 1;
        if off < bytes.len() && bytes[off] == target {
            return off;
        }
    }
    bytes.len() // sentinel
}

/// Skip cursor forward until structurals[cursor] >= target, then optionally past it.
fn advance_cursor_to(structurals: &[u32], cursor: &mut usize, target: u32) {
    while *cursor < structurals.len() && structurals[*cursor] < target {
        *cursor += 1;
    }
    if *cursor < structurals.len() && structurals[*cursor] == target {
        *cursor += 1;
    }
}

/// If `structurals[cursor]` references a byte equal to `expected`, advance past it.
fn skip_one_of(bytes: &[u8], structurals: &[u32], cursor: &mut usize, expected: u8) {
    if *cursor < structurals.len() {
        let off = structurals[*cursor] as usize;
        if off < bytes.len() && bytes[off] == expected {
            *cursor += 1;
        }
    }
}

/// For `Node::String`: derive byte offset of the opening quote in the input.
fn string_offset(bytes: &[u8], s: &str) -> Option<usize> {
    let base = bytes.as_ptr() as usize;
    let p = s.as_ptr() as usize;
    if p >= base && p < base + bytes.len() {
        let body = p - base;
        if body == 0 { None } else { Some(body - 1) }
    } else {
        None
    }
}

/// Skip whitespace after the previous structural to find scalar start.
fn next_scalar_start(bytes: &[u8], structurals: &[u32], cursor: usize) -> usize {
    if cursor == 0 {
        return 0;
    }
    let anchor = structurals[cursor - 1] as usize;
    let mut i = anchor + 1;
    while i < bytes.len() && matches!(bytes[i], b' ' | b'\t' | b'\n' | b'\r') {
        i += 1;
    }
    i
}

/// Walk forward from `start` until we hit a structural delimiter.
fn next_scalar_end(bytes: &[u8], start: usize) -> usize {
    let mut i = start;
    while i < bytes.len() {
        match bytes[i] {
            b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r' => break,
            _ => i += 1,
        }
    }
    i
}
