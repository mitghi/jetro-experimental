//! NDJSON `select(.key == "literal")` over a file.
//!
//! Pipeline:
//!   mmap → split at \n → 1 MB chunks → rayon par_iter → per chunk:
//!     for each line:
//!         build StructuralIndex via parse_with_index
//!         if any key match satisfies predicate, emit raw line bytes
//!   concat outputs in order to stdout.
//!
//! Usage:
//!   ndjson_select <key> <literal> <file.ndjson>
//!
//! Output: matching lines as-is (zero-copy passthrough), one per line.
//!
//! Reference: qj's `process_ndjson_file_mmap` algorithm.

use std::io::Write;
use std::path::Path;

use memmap2::Mmap;
use rayon::prelude::*;

use jetro_experimental::parse;

const CHUNK_TARGET: usize = 1024 * 1024; // 1 MB

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: {} <key> <literal> <file.ndjson>", args[0]);
        std::process::exit(2);
    }
    let key = &args[1];
    let literal = args[2].as_bytes();
    let path = Path::new(&args[3]);

    let file = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let buf: &[u8] = &mmap;

    let chunks = split_chunks(buf, CHUNK_TARGET);

    let outputs: Vec<Vec<u8>> = chunks
        .par_iter()
        .map(|chunk| process_chunk(chunk, key, literal))
        .collect();

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for chunk_out in outputs {
        out.write_all(&chunk_out)?;
    }
    Ok(())
}

/// Split `buf` into chunks ≤ `target` bytes, each ending at `\n`.
fn split_chunks(buf: &[u8], target: usize) -> Vec<&[u8]> {
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < buf.len() {
        let end = (start + target).min(buf.len());
        if end == buf.len() {
            chunks.push(&buf[start..]);
            break;
        }
        // Walk back to last \n so we don't split a line.
        let mut split = end;
        while split > start && buf[split - 1] != b'\n' {
            split -= 1;
        }
        if split == start {
            // No \n in this window — emit whole window (line longer than target).
            chunks.push(&buf[start..end]);
            start = end;
        } else {
            chunks.push(&buf[start..split]);
            start = split;
        }
    }
    chunks
}

/// Walk lines of a chunk; emit any matching the predicate.
fn process_chunk(chunk: &[u8], key: &str, literal: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(chunk.len() / 4);
    let mut start = 0usize;
    let mut i = 0usize;
    while i < chunk.len() {
        if chunk[i] == b'\n' {
            let line = &chunk[start..i];
            if line_matches(line, key, literal) {
                out.extend_from_slice(line);
                out.push(b'\n');
            }
            start = i + 1;
        }
        i += 1;
    }
    if start < chunk.len() {
        let line = &chunk[start..];
        if line_matches(line, key, literal) {
            out.extend_from_slice(line);
            out.push(b'\n');
        }
    }
    out
}

fn line_matches(line: &[u8], key: &str, literal: &[u8]) -> bool {
    let line_owned = line.to_vec();
    let parsed = match parse(line_owned) {
        Ok(p) => p,
        Err(_) => return false,
    };
    let bytes = &parsed.bytes;
    for tok in parsed.index.keys_named(key, None) {
        if let Some(v_tok) = parsed.index.value_for_key(tok) {
            let span = parsed.index.byte_span_in(v_tok, bytes);
            let v = &bytes[span.start as usize..span.end as usize];
            if jetro_experimental::json_string_eq(v, literal) {
                return true;
            }
        }
    }
    false
}
