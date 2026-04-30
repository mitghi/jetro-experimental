//! Streaming structural index builder for NDJSON / huge JSON inputs.
//!
//! Two consumption patterns:
//!
//! 1. **Per-line NDJSON** — `LineBuilder` parses one JSON value per line
//!    independently, building a fresh `StructuralIndex` per line.  Reuses
//!    a single internal `simd_json::Buffers` across lines so allocator
//!    pressure stays bounded.  Sink demand can stop iteration after the
//!    first hit / first K hits etc.
//!
//! 2. **Chunk-fed single document** — `ChunkBuilder` accepts contiguous
//!    chunks of one logical JSON document (still requires the whole doc
//!    to be parseable; doesn't yet support cross-chunk parser state).
//!    Useful when the caller has the bytes in many slices.
//!
//! Both consume from a `BufRead` source.  `LineBuilder` short-circuits
//! cleanly: once the consumer drops the iterator, the underlying reader
//! stops being polled.

#![cfg(feature = "simd-json")]

use std::io::BufRead;

use crate::api::{from_simdjson, BuildOptions, Error, StructuralIndex};
use simd_json::Buffers;

/// One result yielded by `LineBuilder`: the parsed line bytes (consumed
/// in place by simd-json) plus the `StructuralIndex` over them.
pub struct LineParsed {
    /// Bytes the index references.  Owned; safe to keep beyond the
    /// reader's lifetime.
    pub bytes: Vec<u8>,
    pub index: StructuralIndex,
}

/// Streams JSONL: one JSON value per `\n`-delimited line.
///
/// Holds an internal `simd_json::Buffers` for amortised parse setup; the
/// buffer's `Vec`s grow to the largest line seen and reuse capacity on
/// subsequent calls.
pub struct LineBuilder<R> {
    reader: R,
    line_buf: Vec<u8>,
    buffers: Buffers,
    opts: BuildOptions,
    done: bool,
}

impl<R: BufRead> LineBuilder<R> {
    pub fn new(reader: R) -> Self {
        Self::with_options(reader, BuildOptions::default())
    }

    pub fn with_options(reader: R, opts: BuildOptions) -> Self {
        Self {
            reader,
            line_buf: Vec::with_capacity(4096),
            buffers: Buffers::new(4096),
            opts,
            done: false,
        }
    }

    /// Build an index for the next JSONL line.  Returns `Ok(None)` at EOF.
    pub fn next_line(&mut self) -> Result<Option<LineParsed>, Error> {
        if self.done {
            return Ok(None);
        }
        self.line_buf.clear();
        let n = self
            .reader
            .read_until(b'\n', &mut self.line_buf)
            .map_err(|e| Error::Parse(e.to_string()))?;
        if n == 0 {
            self.done = true;
            return Ok(None);
        }
        // Trim trailing newline + CR.
        while matches!(self.line_buf.last().copied(), Some(b'\n') | Some(b'\r')) {
            self.line_buf.pop();
        }
        // Skip empty lines.
        if self
            .line_buf
            .iter()
            .all(|&b| matches!(b, b' ' | b'\t'))
        {
            return self.next_line();
        }

        // simd-json mutates input in place; clone into a per-line owned Vec
        // so we can also keep a borrow for the index build.  Buffers reused
        // across lines amortises Vec capacity allocations.
        let mut bytes = self.line_buf.clone();
        let bytes_ptr = bytes.as_ptr();
        let bytes_len = bytes.len();

        let tape = simd_json::to_tape_with_buffers(&mut bytes, &mut self.buffers)
            .map_err(|e| Error::Parse(e.to_string()))?;
        let structurals = self.buffers.structural_indexes();
        // SAFETY: simd-json no longer mutates bytes after to_tape returns.
        let bytes_view: &[u8] = unsafe { std::slice::from_raw_parts(bytes_ptr, bytes_len) };
        let index = from_simdjson(&tape, bytes_view, structurals, self.opts.clone())?;
        drop(tape);

        Ok(Some(LineParsed { bytes, index }))
    }

    /// Convert into a streaming iterator that yields `Result<LineParsed, Error>`.
    /// The iterator stops after the first error (which it returns) and
    /// after EOF.
    pub fn into_iter(self) -> LineIter<R> {
        LineIter { inner: self }
    }
}

pub struct LineIter<R> {
    inner: LineBuilder<R>,
}

impl<R: BufRead> Iterator for LineIter<R> {
    type Item = Result<LineParsed, Error>;
    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next_line() {
            Ok(Some(p)) => Some(Ok(p)),
            Ok(None) => None,
            Err(e) => {
                self.inner.done = true;
                Some(Err(e))
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const NDJSON: &[u8] = b"\
{\"id\":1,\"x\":\"a\"}
{\"id\":2,\"x\":\"b\"}
{\"id\":3,\"x\":\"a\"}
";

    #[test]
    fn line_builder_iterates_each_line() {
        let mut b = LineBuilder::new(Cursor::new(NDJSON));
        let p1 = b.next_line().unwrap().unwrap();
        assert!(p1.index.has_key("id"));
        assert!(p1.index.has_key("x"));
        let p2 = b.next_line().unwrap().unwrap();
        assert!(p2.index.has_key("x"));
        let p3 = b.next_line().unwrap().unwrap();
        assert!(p3.index.has_key("x"));
        // EOF
        assert!(b.next_line().unwrap().is_none());
    }

    #[test]
    fn line_iter_collects_all() {
        let b = LineBuilder::new(Cursor::new(NDJSON));
        let lines: Vec<_> = b.into_iter().filter_map(|r| r.ok()).collect();
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn skips_blank_lines() {
        let buf: &[u8] = b"\
{\"a\":1}


{\"b\":2}
";
        let b = LineBuilder::new(Cursor::new(buf));
        let lines: Vec<_> = b.into_iter().filter_map(|r| r.ok()).collect();
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn short_circuits_via_take() {
        let b = LineBuilder::new(Cursor::new(NDJSON));
        // Stop after 2 lines.
        let lines: Vec<_> = b.into_iter().take(2).filter_map(|r| r.ok()).collect();
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn cross_line_filter_active_only() {
        let buf: &[u8] = b"\
{\"id\":1,\"active\":true}
{\"id\":2,\"active\":false}
{\"id\":3,\"active\":true}
";
        let b = LineBuilder::new(Cursor::new(buf));
        let n_active = b
            .into_iter()
            .filter_map(|r| r.ok())
            .filter(|p| {
                let mut hit = false;
                for tok in p.index.keys_named("active", None) {
                    if let Some(v_tok) = p.index.value_for_key(tok) {
                        let span = p.index.byte_span_in(v_tok, &p.bytes);
                        let v = &p.bytes[span.start as usize..span.end as usize];
                        if v == b"true" {
                            hit = true;
                            break;
                        }
                    }
                }
                hit
            })
            .count();
        assert_eq!(n_active, 2);
    }

    #[test]
    fn build_options_propagates() {
        let mut b = LineBuilder::with_options(Cursor::new(NDJSON), BuildOptions::minimal());
        let p = b.next_line().unwrap().unwrap();
        // minimal disables key bitmaps
        assert!(!p.index.has_keys());
    }
}
