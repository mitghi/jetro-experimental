//! Public API surface for jetro-experimental.
//!
//! Designed for jetro-core and other downstream tools to consume without
//! depending on internal column layouts.  Internals (`Stage1`,
//! `StructIndex`, `KeyBitmaps`) remain accessible for tests/benches but
//! should not be used by library callers.
//!
//! Stable types:
//!   - [`TokenId`]            — opaque token handle (newtype around u32)
//!   - [`TokenKind`]          — closed enum, `#[non_exhaustive]`
//!   - [`ByteSpan`]           — byte range in source buffer
//!   - [`BuildOptions`]       — knobs for partial builds
//!   - [`StructuralIndex`]    — opaque facade over the internal columns
//!   - [`KeyHits`]            — lazy iterator over key matches
//!   - [`Error`]              — public error enum
//!
//! Stable entry points:
//!   - [`from_bytes`]         — build a structural index from JSON bytes
//!
//! Stable fused query helpers:
//!   - [`find_eq`]            — `$..find(key == literal)`
//!   - [`count_key`]          — popcount-only count of `$..find(key)` hits
//!   - [`find_eq_compound`]   — multi-key AND
//!   - [`json_string_eq`]     — byte-compare a JSON string value to a plain
//!                               literal
//!
//! All iterators are lazy.  Memory layout is private; future refactors can
//! swap Roaring for any other bitmap library without breaking callers.

use std::sync::Arc;

use crate::index::StructIndex;
use crate::keys::{value_for_key as keys_value_for_key, KeyBitmaps, Role};
use crate::stage1::Kind;

/// Opaque token handle.  Internally a `u32` index into the structural index.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct TokenId(pub(crate) u32);

impl TokenId {
    #[inline]
    pub fn raw(self) -> u32 {
        self.0
    }
}

impl From<u32> for TokenId {
    #[inline]
    fn from(v: u32) -> Self {
        Self(v)
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[non_exhaustive]
pub enum TokenKind {
    Object,
    Array,
    Key,
    String,
    Scalar,
    ObjectEnd,
    ArrayEnd,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct ByteSpan {
    pub start: u32,
    pub end: u32,
}

impl ByteSpan {
    #[inline]
    pub fn len(self) -> u32 {
        self.end.saturating_sub(self.start)
    }
    #[inline]
    pub fn is_empty(self) -> bool {
        self.end <= self.start
    }
    #[inline]
    pub fn slice<'a>(self, bytes: &'a [u8]) -> &'a [u8] {
        &bytes[self.start as usize..self.end as usize]
    }
}

/// Public error enum.  Implements `std::error::Error`.
#[derive(Debug)]
pub enum Error {
    Parse(String),
    UnbalancedClose,
    Truncated,
    InvalidUtf8,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Parse(s) => write!(f, "parse error: {s}"),
            Error::UnbalancedClose => write!(f, "unbalanced container close"),
            Error::Truncated => write!(f, "truncated input"),
            Error::InvalidUtf8 => write!(f, "invalid UTF-8"),
        }
    }
}

impl std::error::Error for Error {}

#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct BuildOptions {
    pub keys: bool,
    pub roles: bool,
    pub close_of: bool,
    pub tape_alignment: bool,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            keys: true,
            roles: true,
            close_of: true,
            tape_alignment: false,
        }
    }
}

impl BuildOptions {
    pub fn minimal() -> Self {
        Self {
            keys: false,
            roles: false,
            close_of: false,
            tape_alignment: false,
        }
    }

    pub fn keys_only() -> Self {
        Self {
            keys: true,
            roles: true,
            close_of: false,
            tape_alignment: false,
        }
    }

    pub fn for_jetro_tape() -> Self {
        Self {
            keys: true,
            roles: true,
            close_of: true,
            tape_alignment: true,
        }
    }
}


/// Opaque structural index over a JSON document.  Internal layout is
/// subject to change; consume only via the public methods.
pub struct StructuralIndex {
    pub(crate) inner: Arc<Inner>,
}

pub(crate) struct Inner {
    pub idx: StructIndex,
    pub keys: Option<KeyBitmaps>,
}

// Compile-time assertions that the public API surface is thread-safe.
// `StructuralIndex` is just an `Arc<Inner>`; both `StructIndex` and
// `KeyBitmaps` are owned-data structures (Vec / HashMap / Box<str> /
// croaring::Bitmap) which are all `Send + Sync`.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<StructuralIndex>();
    assert_sync::<StructuralIndex>();
    assert_send::<TokenId>();
    assert_sync::<TokenId>();
    assert_send::<ByteSpan>();
    assert_sync::<ByteSpan>();
    assert_send::<Error>();
    assert_sync::<Error>();
    assert_send::<BuildOptions>();
    assert_sync::<BuildOptions>();
};

impl StructuralIndex {
    /// Total number of stage1 tokens covered by this index.
    pub fn token_count(&self) -> u32 {
        self.inner.idx.stage1.len() as u32
    }

    /// Maximum nesting depth observed in the source document.
    pub fn max_depth(&self) -> u16 {
        self.inner.idx.stage1.depth.iter().copied().max().unwrap_or(0)
    }

    /// All tokens in document order.
    pub fn tokens(&self) -> Tokens<'_> {
        Tokens {
            idx: self,
            cur: 0,
            end: self.token_count(),
        }
    }

    /// Classify a token as Object/Array/Key/String/Scalar/etc.  Reads the
    /// stage1 kind plus (when keys are built) the role to disambiguate
    /// Quote-as-Key from Quote-as-Value.
    #[inline]
    pub fn kind(&self, tok: TokenId) -> TokenKind {
        let i = tok.0 as usize;
        let k = self.inner.idx.stage1.kind[i];
        let role = self
            .inner
            .keys
            .as_ref()
            .map(|kb| kb.role[i])
            .unwrap_or(Role::None);
        match (k, role) {
            (Kind::ObjOpen, _) => TokenKind::Object,
            (Kind::ArrOpen, _) => TokenKind::Array,
            (Kind::ObjClose, _) => TokenKind::ObjectEnd,
            (Kind::ArrClose, _) => TokenKind::ArrayEnd,
            (Kind::Quote, Role::Key) => TokenKind::Key,
            (Kind::Quote, _) => TokenKind::String,
            (Kind::Scalar, _) => TokenKind::Scalar,
            (Kind::Colon | Kind::Comma, _) => TokenKind::Scalar, // unreachable in practice
        }
    }

    /// Nesting depth of a token (root = 0).
    #[inline]
    pub fn depth(&self, tok: TokenId) -> u16 {
        self.inner.idx.stage1.depth[tok.0 as usize]
    }

    /// Byte offset of a token's first byte in the source buffer.
    #[inline]
    pub fn byte_offset(&self, tok: TokenId) -> u32 {
        self.inner.idx.stage1.offset[tok.0 as usize]
    }

    /// Byte span of this token in the source.
    ///
    /// **Container** (`Object` / `Array`): `[open_off, close_off+1)`.
    /// **Container close**: single byte.
    /// **String / Scalar**: requires `byte_span_in(tok, bytes)` for
    /// byte-accurate ranges; this method returns a *coarse* upper bound by
    /// peeking at the next token's offset.  Use `byte_span_in` whenever
    /// you have the source bytes available.
    pub fn byte_span(&self, tok: TokenId) -> ByteSpan {
        let s = &self.inner.idx.stage1;
        let i = tok.0 as usize;
        let start = s.offset[i];

        let end = match s.kind[i] {
            Kind::ObjOpen | Kind::ArrOpen => {
                let close = self.inner.idx.close_of[i];
                if close >= 0 {
                    s.offset[close as usize] + 1
                } else {
                    start + 1
                }
            }
            Kind::ObjClose | Kind::ArrClose => start + 1,
            Kind::Quote | Kind::Scalar => {
                if i + 1 < s.offset.len() {
                    s.offset[i + 1]
                } else {
                    start + 1
                }
            }
            Kind::Colon | Kind::Comma => start + 1,
        };
        ByteSpan { start, end }
    }

    /// Byte-accurate span using source bytes to find the exact end of
    /// strings (closing `"` honouring escapes) and scalars (next delimiter).
    pub fn byte_span_in(&self, tok: TokenId, bytes: &[u8]) -> ByteSpan {
        let s = &self.inner.idx.stage1;
        let i = tok.0 as usize;
        let start = s.offset[i];
        let end = match s.kind[i] {
            Kind::ObjOpen | Kind::ArrOpen => {
                let close = self.inner.idx.close_of[i];
                if close >= 0 {
                    s.offset[close as usize] + 1
                } else {
                    start + 1
                }
            }
            Kind::ObjClose | Kind::ArrClose => start + 1,
            Kind::Quote => scan_string_end(bytes, start),
            Kind::Scalar => scan_scalar_end(bytes, start),
            Kind::Colon | Kind::Comma => start + 1,
        };
        ByteSpan { start, end }
    }

    pub fn parent(&self, tok: TokenId) -> Option<TokenId> {
        let p = self.inner.idx.parent[tok.0 as usize];
        if p < 0 {
            None
        } else {
            Some(TokenId(p as u32))
        }
    }

    pub fn close_of(&self, container: TokenId) -> Option<TokenId> {
        let c = self.inner.idx.close_of[container.0 as usize];
        if c < 0 {
            None
        } else {
            Some(TokenId(c as u32))
        }
    }

    pub fn tape_index(&self, tok: TokenId) -> Option<u32> {
        let t = self.inner.idx.tape_of[tok.0 as usize];
        if t == u32::MAX {
            None
        } else {
            Some(t)
        }
    }

    /// Innermost container token enclosing the given byte position.
    /// Returns None if `pos` is outside any container.
    pub fn container_at_byte(&self, pos: u32) -> Option<TokenId> {
        self.inner.idx.container_at(pos).map(TokenId)
    }

    /// Walk parents innermost→root.
    pub fn ancestors(&self, tok: TokenId) -> Ancestors<'_> {
        Ancestors {
            parent: &self.inner.idx.parent,
            cur: self.inner.idx.parent[tok.0 as usize],
        }
    }

    pub fn slice<'a>(&self, bytes: &'a [u8], tok: TokenId) -> &'a [u8] {
        let span = self.byte_span_in(tok, bytes);
        &bytes[span.start as usize..(span.end as usize).min(bytes.len())]
    }

    /// Whether the Mison key-bitmap layer was built (controlled by
    /// `BuildOptions::keys`).
    pub fn has_keys(&self) -> bool {
        self.inner.keys.is_some()
    }

    pub fn keys_named<'a>(&'a self, name: &str, depth: Option<u16>) -> KeyHits<'a> {
        let kb = match &self.inner.keys {
            Some(k) => k,
            None => return KeyHits::empty(),
        };
        let id = match kb.by_name.get(name) {
            Some(&id) => id,
            None => return KeyHits::empty(),
        };
        let bm = &kb.bitmaps[id as usize];
        let result = match depth.and_then(|d| kb.depth_bitmaps.get(d as usize)) {
            Some(dm) => bm.and(dm),
            None => bm.clone(),
        };
        KeyHits::from_bitmap(result)
    }

    pub fn has_key(&self, name: &str) -> bool {
        self.inner
            .keys
            .as_ref()
            .map(|k| k.by_name.contains_key(name))
            .unwrap_or(false)
    }

    pub fn keys_seen(&self) -> impl Iterator<Item = &str> + '_ {
        self.inner
            .keys
            .as_ref()
            .into_iter()
            .flat_map(|k| k.dict.iter().map(|s| &**s))
    }

    pub fn value_for_key(&self, key_tok: TokenId) -> Option<TokenId> {
        keys_value_for_key(&self.inner.idx, key_tok.0).map(TokenId)
    }

    /// Subtree-restricted key search: matches only tokens whose token-index
    /// falls within `[root.raw(), close_of(root)]`.
    ///
    /// Iterates the precomputed key bitmap via a borrowed roaring cursor —
    /// no bitmap allocation, no range AND. The returned `KeyHits` walks the
    /// underlying bitmap with `reset_at_or_after(lo)` and stops once the
    /// cursor's value exceeds `close_of(root)`.
    pub fn keys_named_in<'a>(&'a self, name: &str, root: TokenId) -> KeyHits<'a> {
        let kb = match &self.inner.keys {
            Some(k) => k,
            None => return KeyHits::empty(),
        };
        let id = match kb.by_name.get(name) {
            Some(&id) => id,
            None => return KeyHits::empty(),
        };
        let close = self
            .close_of(root)
            .map(|t| t.0)
            .unwrap_or(self.token_count().saturating_sub(1));
        let bm = &kb.bitmaps[id as usize];
        KeyHits::bounded(bm, root.0, close)
    }

    /// Direct field lookup on a single container token: returns the value
    /// token of the named key, if present.
    ///
    /// Walks the global key bitmap for `name` via a borrowed roaring cursor
    /// seeked to `parent.0 + 1` and stopping at `close_of(parent)`. Matches
    /// the first key token whose `parent[]` entry equals `parent`. No bitmap
    /// allocation, no range AND, no `KeyHits` materialisation — pure cursor
    /// seek over the precomputed key bitmap.
    pub fn field_of(&self, parent: TokenId, name: &str) -> Option<TokenId> {
        let kb = self.inner.keys.as_ref()?;
        let id = *kb.by_name.get(name)?;
        let bm = &kb.bitmaps[id as usize];
        let close = self
            .close_of(parent)
            .map(|t| t.0)
            .unwrap_or_else(|| self.token_count().saturating_sub(1));
        let lo = parent.0.saturating_add(1);
        if lo > close {
            return None;
        }
        let mut cur = bm.cursor();
        cur.reset_at_or_after(lo);
        while let Some(v) = cur.current() {
            if v > close {
                break;
            }
            let k_tok = TokenId(v);
            if self.parent(k_tok) == Some(parent) {
                return self.value_for_key(k_tok);
            }
            cur.move_next();
        }
        None
    }

    /// Subtree range as a bitmap.  Cheap; range AND is a Roaring SIMD
    /// primitive on the consumer side.
    pub fn subtree_bitmap(&self, root: TokenId) -> croaring::Bitmap {
        let close = self
            .close_of(root)
            .map(|t| t.0)
            .unwrap_or(self.token_count().saturating_sub(1));
        let mut out = croaring::Bitmap::new();
        out.add_range(root.0..=close);
        out
    }
}


pub struct Tokens<'a> {
    idx: &'a StructuralIndex,
    cur: u32,
    end: u32,
}

impl<'a> Iterator for Tokens<'a> {
    type Item = TokenId;
    fn next(&mut self) -> Option<TokenId> {
        if self.cur >= self.end {
            return None;
        }
        let _ = self.idx;
        let t = TokenId(self.cur);
        self.cur += 1;
        Some(t)
    }
}

pub struct Ancestors<'a> {
    parent: &'a [i32],
    cur: i32,
}

impl<'a> Iterator for Ancestors<'a> {
    type Item = TokenId;
    fn next(&mut self) -> Option<TokenId> {
        if self.cur < 0 {
            return None;
        }
        let out = TokenId(self.cur as u32);
        self.cur = self.parent[self.cur as usize];
        Some(out)
    }
}

/// Lazy iterator over key matches.  Backed by a Roaring bitmap; supports
/// composition (and/or) without materialising intermediate Vecs.
///
/// Iteration uses a materialised `Vec<u32>` (cached on first call) so the
/// underlying bitmap is preserved for subsequent ops like `count()` /
/// `first()` / `last()`.
/// Iterator over key tokens. Has two modes:
/// - **Owned**: holds a `croaring::Bitmap` (used by `keys_named` and after
///   `and`/`or` set ops). Iteration materialises a `Vec<u32>` once.
/// - **Bounded**: borrows the source key bitmap and walks it via a
///   roaring cursor restricted to `[lo, hi]`. No allocation, no AND.
///   Used by `keys_named_in` for subtree-restricted scans (the hot path).
pub struct KeyHits<'a> {
    state: KeyHitsState<'a>,
}

enum KeyHitsState<'a> {
    Empty,
    Owned {
        bitmap: croaring::Bitmap,
        cache: Option<Vec<u32>>,
        pos: usize,
    },
    Bounded {
        bitmap: &'a croaring::Bitmap,
        cursor: Option<croaring::bitmap::BitmapCursor<'a>>,
        lo: u32,
        hi: u32,
        started: bool,
    },
}

impl<'a> KeyHits<'a> {
    fn from_bitmap(bm: croaring::Bitmap) -> Self {
        Self {
            state: KeyHitsState::Owned {
                bitmap: bm,
                cache: None,
                pos: 0,
            },
        }
    }

    pub(crate) fn bounded(bitmap: &'a croaring::Bitmap, lo: u32, hi: u32) -> Self {
        if lo > hi {
            return Self::empty();
        }
        Self {
            state: KeyHitsState::Bounded {
                bitmap,
                cursor: None,
                lo,
                hi,
                started: false,
            },
        }
    }

    fn empty() -> Self {
        Self {
            state: KeyHitsState::Empty,
        }
    }

    /// Convert any state into an owned bitmap (allocates only when bounded);
    /// used by set-ops (`and`/`or`) which require materialised bitmaps.
    fn into_owned_bitmap(self) -> Option<croaring::Bitmap> {
        match self.state {
            KeyHitsState::Empty => None,
            KeyHitsState::Owned { bitmap, .. } => Some(bitmap),
            KeyHitsState::Bounded { bitmap, lo, hi, .. } => {
                let mut range = croaring::Bitmap::new();
                range.add_range(lo..=hi);
                let mut out = bitmap.clone();
                out.and_inplace(&range);
                Some(out)
            }
        }
    }

    pub fn at_depth(self, _depth: u16) -> Self {
        self
    }

    pub fn and(self, other: KeyHits<'a>) -> Self {
        match (self.into_owned_bitmap(), other.into_owned_bitmap()) {
            (Some(mut a), Some(b)) => {
                a.and_inplace(&b);
                Self::from_bitmap(a)
            }
            _ => Self::empty(),
        }
    }

    pub fn or(self, other: KeyHits<'a>) -> Self {
        match (self.into_owned_bitmap(), other.into_owned_bitmap()) {
            (Some(mut a), Some(b)) => {
                a.or_inplace(&b);
                Self::from_bitmap(a)
            }
            (Some(a), None) | (None, Some(a)) => Self::from_bitmap(a),
            _ => Self::empty(),
        }
    }

    pub fn count(self) -> u64 {
        match self.state {
            KeyHitsState::Empty => 0,
            KeyHitsState::Owned { bitmap, .. } => bitmap.cardinality(),
            KeyHitsState::Bounded { bitmap, lo, hi, .. } => {
                bitmap.range_cardinality(lo..=hi)
            }
        }
    }

    pub fn first(self) -> Option<TokenId> {
        match self.state {
            KeyHitsState::Empty => None,
            KeyHitsState::Owned { bitmap, .. } => bitmap.minimum().map(TokenId),
            KeyHitsState::Bounded { bitmap, lo, hi, .. } => {
                let mut cur = bitmap.cursor();
                cur.reset_at_or_after(lo);
                cur.current().filter(|&v| v <= hi).map(TokenId)
            }
        }
    }

    pub fn last(self) -> Option<TokenId> {
        match self.state {
            KeyHitsState::Empty => None,
            KeyHitsState::Owned { bitmap, .. } => bitmap.maximum().map(TokenId),
            KeyHitsState::Bounded { bitmap, lo, hi, .. } => {
                // Walk forward from lo, tracking the last value <= hi.
                // Roaring cursors don't expose a backwards-from variant in this
                // crate version; the bounded last is rare so the linear scan is
                // acceptable.
                let mut cur = bitmap.cursor();
                cur.reset_at_or_after(lo);
                let mut last = None;
                while let Some(v) = cur.current() {
                    if v > hi {
                        break;
                    }
                    last = Some(TokenId(v));
                    cur.move_next();
                }
                last
            }
        }
    }

    pub fn collect_into(self, buf: &mut Vec<TokenId>) {
        match self.state {
            KeyHitsState::Empty => {}
            KeyHitsState::Owned { bitmap, .. } => {
                buf.extend(bitmap.to_vec().into_iter().map(TokenId));
            }
            KeyHitsState::Bounded { bitmap, lo, hi, .. } => {
                let mut cur = bitmap.cursor();
                cur.reset_at_or_after(lo);
                while let Some(v) = cur.current() {
                    if v > hi {
                        break;
                    }
                    buf.push(TokenId(v));
                    cur.move_next();
                }
            }
        }
    }
}

impl<'a> Iterator for KeyHits<'a> {
    type Item = TokenId;
    fn next(&mut self) -> Option<TokenId> {
        match &mut self.state {
            KeyHitsState::Empty => None,
            KeyHitsState::Owned { bitmap, cache, pos } => {
                if cache.is_none() {
                    *cache = Some(bitmap.to_vec());
                }
                let v = cache.as_ref()?.get(*pos).copied()?;
                *pos += 1;
                Some(TokenId(v))
            }
            KeyHitsState::Bounded {
                bitmap,
                cursor,
                lo,
                hi,
                started,
            } => {
                if !*started {
                    let mut c = bitmap.cursor();
                    c.reset_at_or_after(*lo);
                    *cursor = Some(c);
                    *started = true;
                } else if let Some(c) = cursor.as_mut() {
                    c.move_next();
                }
                let c = cursor.as_ref()?;
                let v = c.current()?;
                if v > *hi {
                    return None;
                }
                Some(TokenId(v))
            }
        }
    }
}


pub fn from_bytes(bytes: &[u8]) -> Result<StructuralIndex, Error> {
    from_bytes_with(bytes, BuildOptions::default())
}

pub fn from_bytes_with(bytes: &[u8], opts: BuildOptions) -> Result<StructuralIndex, Error> {
    let idx = StructIndex::build(bytes).map_err(|s| Error::Parse(s.to_string()))?;
    let keys = if opts.keys {
        Some(KeyBitmaps::build(&idx, bytes))
    } else {
        None
    };
    Ok(StructuralIndex {
        inner: Arc::new(Inner { idx, keys }),
    })
}

/// `$..find(key == literal)` — emit enclosing-Object token ids.
///
/// `literal` is the **plain** value bytes to match (e.g. `b"PushEvent"`).
/// String values are unquoted before comparison.
pub fn find_eq<'a>(
    idx: &'a StructuralIndex,
    bytes: &'a [u8],
    key: &str,
    literal: &[u8],
) -> impl Iterator<Item = TokenId> + 'a {
    let key_hits = idx.keys_named(key, None);
    let bytes_ref = bytes;
    let literal_ref = literal.to_vec();
    let idx_ref = idx;
    key_hits
        .filter_map(move |k_tok| {
            let v_tok = idx_ref.value_for_key(k_tok)?;
            let span = idx_ref.byte_span_in(v_tok, bytes_ref);
            let v_bytes = &bytes_ref[span.start as usize..span.end as usize];
            if value_matches(v_bytes, &literal_ref) {
                idx_ref.parent(k_tok)
            } else {
                None
            }
        })
}

/// O(1) — popcount of the key bitmap.
pub fn count_key(idx: &StructuralIndex, key: &str) -> u64 {
    idx.keys_named(key, None).count()
}

/// Compound `$..find(k1 == l1 AND k2 == l2 ...)`.  Conds applied within the
/// SAME object (parent-token equality).
pub fn find_eq_compound<'a>(
    idx: &'a StructuralIndex,
    bytes: &'a [u8],
    conds: &'a [(&str, &[u8])],
) -> impl Iterator<Item = TokenId> + 'a {
    // Strategy: bitmap-AND key sets across conds, but key positions don't
    // share parents directly — must verify per candidate that all conds match
    // within the same enclosing object.
    let mut cands: Vec<TokenId> = Vec::new();
    if let Some((first_key, _)) = conds.first() {
        idx.keys_named(first_key, None).collect_into(&mut cands);
    }
    cands.into_iter().filter_map(move |k_tok| {
        let parent_obj = idx.parent(k_tok)?;
        // Verify every cond resolves to the same parent obj.
        for (key, lit) in conds.iter() {
            let mut matched = false;
            for k in idx.keys_named(key, None) {
                if idx.parent(k) == Some(parent_obj) {
                    if let Some(v) = idx.value_for_key(k) {
                        let span = idx.byte_span_in(v, bytes);
                        let v_bytes = &bytes[span.start as usize..span.end as usize];
                        if value_matches(v_bytes, lit) {
                            matched = true;
                            break;
                        }
                    }
                }
            }
            if !matched {
                return None;
            }
        }
        Some(parent_obj)
    })
}


/// Compare a JSON-encoded value byte-slice against a plain literal.
/// - If `value` is a JSON string (`"…"`), compares the unquoted body.
/// - Otherwise compares the raw bytes (number / bool / null).
///
/// Fast path uses `memchr::memchr` (SIMD-accelerated) to detect escape
/// sequences and short-circuit the comparison.  When the body contains
/// no `\` byte, we fall through to a direct slice equality which LLVM
/// auto-vectorises to AVX2/SSE on x86_64.
pub fn json_string_eq(value: &[u8], literal: &[u8]) -> bool {
    value_matches(value, literal)
}

fn value_matches(value: &[u8], literal: &[u8]) -> bool {
    if value.len() >= 2 && value[0] == b'"' && value[value.len() - 1] == b'"' {
        let body = &value[1..value.len() - 1];
        // SIMD escape probe: memchr is AVX2/SSE2 on x86_64, NEON on aarch64.
        // Common case (no escapes) takes one scan + one slice compare.
        if memchr::memchr(b'\\', body).is_some() {
            // Slow path: needs unescape decode.  Conservative for now;
            // upgrade to SIMD JSON-unescape when needed.
            return slow_decode_eq(body, literal);
        }
        body == literal
    } else {
        value == literal
    }
}

/// Fallback for escaped strings: decode standard JSON escapes byte-by-byte
/// and compare to the literal.  Handles `\\ \" \/ \n \t \r \b \f`.  Does
/// NOT handle `\uXXXX` (caller takes responsibility — returns false).
fn slow_decode_eq(body: &[u8], literal: &[u8]) -> bool {
    let mut i = 0;
    let mut j = 0;
    while i < body.len() && j < literal.len() {
        let (decoded, consumed) = match body[i] {
            b'\\' if i + 1 < body.len() => match body[i + 1] {
                b'"' => (b'"', 2),
                b'\\' => (b'\\', 2),
                b'/' => (b'/', 2),
                b'n' => (b'\n', 2),
                b't' => (b'\t', 2),
                b'r' => (b'\r', 2),
                b'b' => (b'\x08', 2),
                b'f' => (b'\x0c', 2),
                _ => return false, // \uXXXX or unknown escape — punt
            },
            c => (c, 1),
        };
        if decoded != literal[j] {
            return false;
        }
        i += consumed;
        j += 1;
    }
    i == body.len() && j == literal.len()
}

/// Scan from `start` (the opening `"`) to the matching closing `"`,
/// honouring `\\` escapes.  Returns one past the closing quote.
fn scan_string_end(bytes: &[u8], start: u32) -> u32 {
    let mut i = (start + 1) as usize;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i = i.saturating_add(2),
            b'"' => return (i + 1) as u32,
            _ => i += 1,
        }
    }
    bytes.len() as u32
}

/// Scan from `start` until the next JSON delimiter byte.
fn scan_scalar_end(bytes: &[u8], start: u32) -> u32 {
    let mut i = start as usize;
    while i < bytes.len() {
        match bytes[i] {
            b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r' => return i as u32,
            _ => i += 1,
        }
    }
    bytes.len() as u32
}


/// Parse a JSON number byte-slice as `f64`.  Uses `fast-float` (SSE/AVX
/// SIMD) when the feature is enabled, falls back to `str::parse` otherwise.
///
/// Returns None on malformed input or non-UTF-8 bytes.
pub fn parse_f64(bytes: &[u8]) -> Option<f64> {
    #[cfg(feature = "fast-numbers")]
    {
        fast_float::parse(bytes).ok()
    }
    #[cfg(not(feature = "fast-numbers"))]
    {
        std::str::from_utf8(bytes).ok()?.parse::<f64>().ok()
    }
}

/// Parse a JSON number byte-slice as `i64`.  Tries integer parse first,
/// then falls back to `parse_f64` (truncating fractional component).
pub fn parse_i64(bytes: &[u8]) -> Option<i64> {
    if let Ok(s) = std::str::from_utf8(bytes) {
        if let Ok(n) = s.parse::<i64>() {
            return Some(n);
        }
    }
    parse_f64(bytes).map(|f| f as i64)
}

/// Determine whether a value byte-slice is a JSON number that compares
/// equal to the literal interpreted as a number.  Used by `find_eq` when
/// the caller wants numeric semantics rather than byte-for-byte match.
pub fn json_number_eq(value: &[u8], literal: &[u8]) -> bool {
    match (parse_f64(value), parse_f64(literal)) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}


/// Build an Aho-Corasick + Teddy automaton over multiple JSON-key patterns
/// for raw byte-scan over NDJSON / streaming input.  Each pattern is the
/// full `"keyname":` byte sequence — caller wraps `keyname` in quotes +
/// trailing colon to avoid sub-string false positives.
///
/// Useful when the index has not yet been built — e.g. per-line scanning
/// of a JSONL stream.  After building the index via `from_bytes` /
/// `from_simdjson`, prefer `keys_named` (Roaring AND).
#[cfg(feature = "multi-key")]
pub fn multi_key_finder(keys: &[&str]) -> aho_corasick::AhoCorasick {
    use aho_corasick::AhoCorasickBuilder;
    let patterns: Vec<String> = keys.iter().map(|k| format!("\"{}\":", k)).collect();
    AhoCorasickBuilder::new()
        .ascii_case_insensitive(false)
        .build(&patterns)
        .expect("aho-corasick build")
}

/// Yield (pattern_index, byte_offset) for every match of any pattern in
/// `bytes`.  pattern_index is the index into the `keys` slice originally
/// passed to `multi_key_finder`.
#[cfg(feature = "multi-key")]
pub fn multi_key_scan<'a>(
    finder: &'a aho_corasick::AhoCorasick,
    bytes: &'a [u8],
) -> impl Iterator<Item = (usize, usize)> + 'a {
    finder.find_iter(bytes).map(|m| (m.pattern().as_usize(), m.start()))
}


/// SIMD-accelerated UTF-8 validation via `simdutf8` (5-10 GB/s).  Returns
/// `Ok(())` if the input is valid UTF-8.
#[cfg(feature = "validate-utf8")]
pub fn validate_utf8(bytes: &[u8]) -> Result<(), Error> {
    simdutf8::basic::from_utf8(bytes)
        .map(|_| ())
        .map_err(|_| Error::InvalidUtf8)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_bytes_basic_roundtrip() {
        let buf = br#"{"a":1,"b":"hi","c":[1,2,3]}"#;
        let idx = from_bytes(buf).unwrap();
        assert!(idx.token_count() > 0);
        assert!(idx.has_keys());

        let mut keys: Vec<&str> = idx.keys_seen().collect();
        keys.sort();
        assert_eq!(keys, vec!["a", "b", "c"]);
    }

    #[test]
    fn keys_named_returns_token_ids() {
        let buf = br#"{"x":1,"y":2,"x":3}"#;
        let idx = from_bytes(buf).unwrap();
        let xs: Vec<TokenId> = idx.keys_named("x", None).collect();
        assert!(!xs.is_empty(), "expected at least one 'x' match");
        for t in &xs {
            assert_eq!(idx.kind(*t), TokenKind::Key);
        }
    }

    #[test]
    fn count_key_uses_popcount() {
        let buf = br#"{"a":{"x":1},"b":{"x":2},"c":{"x":3,"y":4}}"#;
        let idx = from_bytes(buf).unwrap();
        let c = count_key(&idx, "x");
        assert_eq!(c, 3);
        let cy = count_key(&idx, "y");
        assert_eq!(cy, 1);
    }

    #[test]
    fn key_hits_first_short_circuits() {
        let buf = br#"{"a":1,"b":2,"c":3,"d":4}"#;
        let idx = from_bytes(buf).unwrap();
        let first = idx.keys_named("c", None).first();
        assert!(first.is_some());
        assert_eq!(idx.kind(first.unwrap()), TokenKind::Key);
    }

    #[test]
    fn find_eq_returns_enclosing_objects() {
        let buf = br#"{"a":{"x":"test"},"b":{"x":"nope"},"c":{"x":"test"}}"#;
        let idx = from_bytes(buf).unwrap();
        let hits: Vec<TokenId> = find_eq(&idx, buf, "x", b"test").collect();
        assert_eq!(hits.len(), 2);
        for t in &hits {
            assert_eq!(idx.kind(*t), TokenKind::Object);
        }
    }

    #[test]
    fn container_at_byte_works() {
        let buf = br#"{"a":{"x":1},"b":2}"#;
        let idx = from_bytes(buf).unwrap();
        // byte 9 lands inside "x" inside the inner object.
        let c = idx.container_at_byte(9).unwrap();
        // Should resolve to the inner object (deeper than root).
        assert_eq!(idx.kind(c), TokenKind::Object);
        assert!(idx.depth(c) >= 1);
    }

    #[test]
    fn build_options_minimal_skips_keys() {
        let buf = br#"{"a":1}"#;
        let idx = from_bytes_with(buf, BuildOptions::minimal()).unwrap();
        assert!(!idx.has_keys());
        assert_eq!(idx.keys_named("a", None).count(), 0);
    }

    #[test]
    fn ancestors_walks_to_root() {
        let buf = br#"{"a":{"b":{"c":42}}}"#;
        let idx = from_bytes(buf).unwrap();
        // Find the deepest scalar
        let scalar_tok = idx
            .tokens()
            .find(|t| idx.kind(*t) == TokenKind::Scalar)
            .unwrap();
        let chain: Vec<TokenId> = idx.ancestors(scalar_tok).collect();
        assert!(!chain.is_empty());
        // Last ancestor's parent must be root (None)
        assert!(idx.parent(*chain.last().unwrap()).is_none());
    }

    #[test]
    fn json_string_eq_handles_quotes() {
        assert!(json_string_eq(b"\"hello\"", b"hello"));
        assert!(!json_string_eq(b"\"hello\"", b"world"));
        // Escape decode now handled — `\n` decodes to newline.
        assert!(json_string_eq(b"\"he\\nllo\"", b"he\nllo"));
        // Non-string raw match.
        assert!(json_string_eq(b"42", b"42"));
        assert!(json_string_eq(b"true", b"true"));
    }

    #[test]
    fn find_eq_compound_intersect() {
        let buf = br#"[{"k1":"a","k2":"b"},{"k1":"a","k2":"c"},{"k1":"x","k2":"b"}]"#;
        let idx = from_bytes(buf).unwrap();
        let conds: &[(&str, &[u8])] = &[("k1", b"a"), ("k2", b"b")];
        let hits: Vec<TokenId> = find_eq_compound(&idx, buf, conds).collect();
        assert_eq!(hits.len(), 1, "exactly one obj should match both conds");
    }

    #[test]
    fn json_string_eq_handles_simple_escapes() {
        assert!(json_string_eq(b"\"a\\nb\"", b"a\nb"));
        assert!(json_string_eq(b"\"a\\\\b\"", b"a\\b"));
        assert!(json_string_eq(b"\"a\\\"b\"", b"a\"b"));
        assert!(!json_string_eq(b"\"a\\nb\"", b"axb"));
        // \uXXXX still punted (returns false).
        assert!(!json_string_eq(b"\"\\u0041\"", b"A"));
    }

    #[test]
    fn parse_f64_works_with_or_without_fast_numbers() {
        assert_eq!(parse_f64(b"3.14"), Some(3.14));
        assert_eq!(parse_f64(b"-1e10"), Some(-1e10));
        assert_eq!(parse_f64(b"42"), Some(42.0));
        assert_eq!(parse_f64(b"not a number"), None);
    }

    #[test]
    fn parse_i64_falls_back_to_f64_truncate() {
        assert_eq!(parse_i64(b"42"), Some(42));
        assert_eq!(parse_i64(b"-7"), Some(-7));
        assert_eq!(parse_i64(b"3.9"), Some(3)); // truncates
    }

    #[test]
    fn json_number_eq_compares_numerically() {
        assert!(json_number_eq(b"42", b"42"));
        assert!(json_number_eq(b"42.0", b"42")); // numeric equality, not byte
        assert!(!json_number_eq(b"42", b"43"));
        assert!(!json_number_eq(b"abc", b"42"));
    }

    #[cfg(feature = "multi-key")]
    #[test]
    fn multi_key_finder_matches_top_level_keys() {
        let keys = ["type", "actor", "repo"];
        let finder = multi_key_finder(&keys);
        let bytes = br#"{"type":"PushEvent","actor":{"login":"x"},"repo":{"name":"y"}}"#;
        let hits: Vec<(usize, usize)> = multi_key_scan(&finder, bytes).collect();
        assert_eq!(hits.len(), 3);
        // Patterns matched in input order
        let pattern_indices: Vec<usize> = hits.iter().map(|(p, _)| *p).collect();
        assert_eq!(pattern_indices, vec![0, 1, 2]);
    }

    #[cfg(feature = "validate-utf8")]
    #[test]
    fn validate_utf8_accepts_valid_input() {
        assert!(validate_utf8(b"hello").is_ok());
        assert!(validate_utf8(b"\xff\xfe").is_err()); // invalid UTF-8
    }

}
