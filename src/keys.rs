//! Mison-style key-bitmap layer on top of `StructIndex`.
//!
//! For each interned key, store a Roaring bitmap over stage1 token indices,
//! with bit `i` set iff token `i` is a Quote acting as a *Key* with that key id.
//! Plus per-depth Roaring bitmaps for path-aware lookup.
//!
//! Roaring chosen over `Vec<u64>` because key bitmaps are sparse: most keys
//! appear at <1% of tokens. Raw `K * n_tokens / 8` would explode for K large;
//! Roaring scales with set-bit count instead.
//!
//! Query path:
//!   1. `find_key("x", Some(2))` → AND key bitmap with depth bitmap, iterate set bits
//!   2. for each matching Key token, `value_for_key(key_tok)` → token idx of the value
//!   3. compare value bytes directly (or recurse via container_at / parent_chain)
//!
//! Reference: Li et al., "Mison: A Fast JSON Parser for Data Analytics",
//! VLDB 2017. The two big ideas — SIMD structural index + speculative key
//! lookup — already split across `stage1.rs` and this module.

use std::collections::HashMap;
use std::sync::Arc;

use croaring::Bitmap;

use crate::index::StructIndex;
use crate::stage1::Kind;

/// Role of each stage1 token within its parent container.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Role {
    /// String acting as an object key (Quote followed by Colon, parent is ObjOpen).
    Key = 0,
    /// Value bound to a key inside an object (Quote/Scalar/ObjOpen/ArrOpen after Colon).
    Value = 1,
    /// Array element (Quote/Scalar/ObjOpen/ArrOpen with parent ArrOpen).
    Elem = 2,
    /// Container open/close, separators, or anything else.
    None = 3,
}

#[derive(Debug, Clone)]
pub struct KeyBitmaps {
    /// Interned key strings.  `Arc<str>` shared with `by_name` so each
    /// unique key is heap-allocated exactly once.
    pub dict: Vec<Arc<str>>,
    /// Reverse map: key string -> id.  Borrows the same `Arc<str>` as
    /// `dict[id]`; lookups via `&str` work through `Borrow<str>`.
    pub by_name: HashMap<Arc<str>, u32>,
    /// `bitmaps[key_id]` is a Roaring bitmap over stage1 token indices.
    pub bitmaps: Vec<Bitmap>,
    /// `depth_bitmaps[d]` has bit `i` set iff `stage1.depth[i] == d`.
    pub depth_bitmaps: Vec<Bitmap>,
    /// Role per token; `role.len() == stage1.len()`.
    pub role: Vec<Role>,
    /// `key_id_of[i]` is the dict id for token `i` if `role[i] == Key`,
    /// else `u32::MAX`.
    pub key_id_of: Vec<u32>,
    /// Total number of stage1 tokens covered.
    pub n_tokens: usize,
}

impl KeyBitmaps {
    /// Build the bitmap layer.
    ///
    /// Walks the stage1 token stream once:
    ///   - classifies each Quote as Key vs Value (using parent kind + next-non-separator-is-Colon)
    ///   - decodes Key string bytes, interns, sets the appropriate bitmap bits
    ///   - fills depth bitmaps in lockstep
    pub fn build(idx: &StructIndex, buf: &[u8]) -> Self {
        let n = idx.stage1.len();
        let mut role = vec![Role::None; n];
        let mut key_id_of = vec![u32::MAX; n];
        let mut dict: Vec<Arc<str>> = Vec::new();
        let mut by_name: HashMap<Arc<str>, u32> = HashMap::new();
        let mut bitmaps: Vec<Bitmap> = Vec::new();
        // Max depth observed; used to size depth_bitmaps.
        let max_depth = idx.stage1.depth.iter().copied().max().unwrap_or(0) as usize;
        let mut depth_bitmaps: Vec<Bitmap> = (0..=max_depth).map(|_| Bitmap::new()).collect();

        for i in 0..n {
            let d = idx.stage1.depth[i] as usize;
            depth_bitmaps[d].add(i as u32);
        }

        for i in 0..n {
            let k = idx.stage1.kind[i];
            // Container parent in raw stage1 terms (could be -1 for root)
            let pidx = idx.parent[i];
            let parent_kind = if pidx >= 0 {
                Some(idx.stage1.kind[pidx as usize])
            } else {
                None
            };

            // Determine role.
            let r = match k {
                Kind::Quote => {
                    if parent_kind == Some(Kind::ObjOpen) && next_is_colon(idx, i) {
                        Role::Key
                    } else if parent_kind == Some(Kind::ObjOpen) && prev_is_colon(idx, i) {
                        Role::Value
                    } else if parent_kind == Some(Kind::ArrOpen) {
                        Role::Elem
                    } else {
                        // Root-level quote (single string doc) — treat as Value
                        Role::Value
                    }
                }
                Kind::Scalar => {
                    if parent_kind == Some(Kind::ArrOpen) {
                        Role::Elem
                    } else {
                        Role::Value
                    }
                }
                Kind::ObjOpen | Kind::ArrOpen => {
                    if parent_kind == Some(Kind::ArrOpen) {
                        Role::Elem
                    } else if parent_kind == Some(Kind::ObjOpen) {
                        Role::Value
                    } else {
                        Role::None
                    }
                }
                _ => Role::None,
            };
            role[i] = r;

            if r == Role::Key {
                // Decode key body.
                let off = idx.stage1.offset[i] as usize;
                let key_bytes = decode_string_body(buf, off);
                // Single allocation: build Arc<str> once, share between dict
                // and by_name.  Cloning Arc<str> is a refcount bump, not a
                // buffer copy.
                let id = match by_name.get(key_bytes.as_str()) {
                    Some(&id) => id,
                    None => {
                        let id = dict.len() as u32;
                        let key: Arc<str> = Arc::from(key_bytes.as_str());
                        dict.push(Arc::clone(&key));
                        by_name.insert(key, id);
                        bitmaps.push(Bitmap::new());
                        id
                    }
                };
                key_id_of[i] = id;
                bitmaps[id as usize].add(i as u32);
            }
        }

        // Compact run-length-encoded bitmap storage.
        for b in bitmaps.iter_mut() {
            b.run_optimize();
        }
        for b in depth_bitmaps.iter_mut() {
            b.run_optimize();
        }

        KeyBitmaps {
            dict,
            by_name,
            bitmaps,
            depth_bitmaps,
            role,
            key_id_of,
            n_tokens: n,
        }
    }

    /// Find all token indices where `name` appears as an object Key.
    /// If `depth` is supplied, restricts to that depth.
    pub fn find_key(&self, name: &str, depth: Option<u16>) -> Vec<u32> {
        let id = match self.by_name.get(name) {
            Some(&id) => id,
            None => return Vec::new(),
        };
        let bm = &self.bitmaps[id as usize];
        match depth.and_then(|d| self.depth_bitmaps.get(d as usize)) {
            Some(dm) => bm.and(dm).iter().collect(),
            None => bm.iter().collect(),
        }
    }

    /// Number of tokens covered.
    pub fn len(&self) -> usize {
        self.n_tokens
    }

    /// Was `name` ever seen as a key in this document? O(1).
    pub fn contains_key(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }
}

/// Given a Key token index, return the token index of its bound value
/// (the next non-Colon token at the same parent).
pub fn value_for_key(idx: &StructIndex, key_tok: u32) -> Option<u32> {
    let parent = idx.parent[key_tok as usize];
    let n = idx.stage1.len();
    let mut i = key_tok as usize + 1;
    while i < n {
        let same_parent = idx.parent[i] == parent;
        if same_parent {
            match idx.stage1.kind[i] {
                Kind::Colon | Kind::Comma => {}
                Kind::Quote
                | Kind::Scalar
                | Kind::ObjOpen
                | Kind::ArrOpen
                | Kind::ObjClose
                | Kind::ArrClose => return Some(i as u32),
            }
        } else {
            // Stepped into a child container — that container open IS the value.
            return Some(i as u32);
        }
        i += 1;
    }
    None
}


/// Is the next non-separator token after `i` a Colon under the same parent?
fn next_is_colon(idx: &StructIndex, i: usize) -> bool {
    let parent = idx.parent[i];
    let n = idx.stage1.len();
    let mut j = i + 1;
    while j < n {
        if idx.parent[j] != parent {
            return false;
        }
        match idx.stage1.kind[j] {
            Kind::Colon => return true,
            Kind::Comma => return false,
            _ => return false,
        }
    }
    false
}

/// Is the previous non-separator token before `i` a Colon under the same parent?
fn prev_is_colon(idx: &StructIndex, i: usize) -> bool {
    let parent = idx.parent[i];
    if i == 0 {
        return false;
    }
    let j = i - 1;
    if idx.parent[j] != parent {
        return false;
    }
    matches!(idx.stage1.kind[j], Kind::Colon)
}

/// Decode a JSON string starting at byte `quote_off` (the opening `"`).
/// Honours `\\`, `\"`, `\n`, `\t`, `\r`, `\/`, `\b`, `\f`, and `\uXXXX`.
/// Returns the decoded string body (without surrounding quotes).
fn decode_string_body(buf: &[u8], quote_off: usize) -> String {
    debug_assert_eq!(buf[quote_off], b'"');
    let mut out = String::new();
    let mut i = quote_off + 1;
    while i < buf.len() {
        match buf[i] {
            b'"' => break,
            b'\\' if i + 1 < buf.len() => {
                let esc = buf[i + 1];
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000C}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        if i + 6 <= buf.len() {
                            let hex = &buf[i + 2..i + 6];
                            if let Ok(s) = std::str::from_utf8(hex) {
                                if let Ok(code) = u32::from_str_radix(s, 16) {
                                    if let Some(c) = char::from_u32(code) {
                                        out.push(c);
                                    }
                                }
                            }
                            i += 6;
                            continue;
                        }
                    }
                    _ => out.push(esc as char),
                }
                i += 2;
                continue;
            }
            other => {
                // Raw bytes; collect into out as utf8.
                // Walk forward until next " or \ ; push slice.
                let start = i;
                let _ = other;
                while i < buf.len() && buf[i] != b'"' && buf[i] != b'\\' {
                    i += 1;
                }
                out.push_str(std::str::from_utf8(&buf[start..i]).unwrap_or(""));
                continue;
            }
        }
    }
    out
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_roles_basic() {
        let buf = br#"{"a":1,"b":"two","c":[3,4]}"#;
        let idx = StructIndex::build(buf).unwrap();
        let kb = KeyBitmaps::build(&idx, buf);
        // Roles for the three keys
        let mut keys = Vec::new();
        for (i, &r) in kb.role.iter().enumerate() {
            if r == Role::Key {
                keys.push(i);
            }
        }
        assert_eq!(keys.len(), 3);
        // dict has a, b, c (order = insertion)
        let dict_strs: Vec<&str> = kb.dict.iter().map(|s| &**s).collect();
        assert_eq!(dict_strs, vec!["a", "b", "c"]);
    }

    #[test]
    fn find_key_returns_correct_tokens() {
        let buf = br#"{"a":1,"b":2}"#;
        let idx = StructIndex::build(buf).unwrap();
        let kb = KeyBitmaps::build(&idx, buf);
        let tok_a = kb.find_key("a", None);
        assert_eq!(tok_a.len(), 1);
        let t = tok_a[0] as usize;
        assert_eq!(idx.stage1.kind[t], Kind::Quote);
        assert_eq!(kb.role[t], Role::Key);
    }

    #[test]
    fn key_at_depth_filters() {
        let buf = br#"{"x":{"x":42}}"#;
        let idx = StructIndex::build(buf).unwrap();
        let kb = KeyBitmaps::build(&idx, buf);
        let all = kb.find_key("x", None);
        let d1 = kb.find_key("x", Some(1));
        let d2 = kb.find_key("x", Some(2));
        assert_eq!(all.len(), 2);
        assert_eq!(d1.len(), 1);
        assert_eq!(d2.len(), 1);
        // depth 1 key precedes depth 2 key in token order
        assert!(d1[0] < d2[0]);
    }

    #[test]
    fn key_with_same_text_as_value_not_classified_as_key() {
        // "x" appears as both key and value
        let buf = br#"{"k":"x","x":1}"#;
        let idx = StructIndex::build(buf).unwrap();
        let kb = KeyBitmaps::build(&idx, buf);
        let xs = kb.find_key("x", None);
        // only one match — the actual key, not the string value
        assert_eq!(xs.len(), 1);
        let t = xs[0] as usize;
        assert_eq!(kb.role[t], Role::Key);
        // The other "x" Quote (the string value) should be classified as Value
        let value_xs: Vec<usize> = (0..idx.stage1.len())
            .filter(|&i| idx.stage1.kind[i] == Kind::Quote && kb.role[i] == Role::Value)
            .collect();
        assert!(!value_xs.is_empty());
    }

    #[test]
    fn value_for_key_string() {
        let buf = br#"{"k":"hit"}"#;
        let idx = StructIndex::build(buf).unwrap();
        let kb = KeyBitmaps::build(&idx, buf);
        let k = kb.find_key("k", None)[0];
        let v = value_for_key(&idx, k).unwrap();
        assert_eq!(idx.stage1.kind[v as usize], Kind::Quote);
        assert_eq!(kb.role[v as usize], Role::Value);
        // value's offset is at the opening quote of "hit"
        let off = idx.stage1.offset[v as usize] as usize;
        assert_eq!(buf[off], b'"');
        assert_eq!(&buf[off + 1..off + 4], b"hit");
    }

    #[test]
    fn value_for_key_object() {
        let buf = br#"{"k":{"x":1}}"#;
        let idx = StructIndex::build(buf).unwrap();
        let kb = KeyBitmaps::build(&idx, buf);
        let k = kb.find_key("k", None)[0];
        let v = value_for_key(&idx, k).unwrap();
        assert_eq!(idx.stage1.kind[v as usize], Kind::ObjOpen);
    }

    #[test]
    fn value_for_key_number() {
        let buf = br#"{"k":42}"#;
        let idx = StructIndex::build(buf).unwrap();
        let kb = KeyBitmaps::build(&idx, buf);
        let k = kb.find_key("k", None)[0];
        let v = value_for_key(&idx, k).unwrap();
        assert_eq!(idx.stage1.kind[v as usize], Kind::Scalar);
    }

    #[test]
    fn mison_style_query_x_eq_test() {
        // Full pipeline: $..find(x == "test")
        let buf = br#"{"a":{"x":"test"},"b":{"x":"nope"},"c":{"y":"test"}}"#;
        let idx = StructIndex::build(buf).unwrap();
        let kb = KeyBitmaps::build(&idx, buf);

        let mut hits = Vec::new();
        for k_tok in kb.find_key("x", None) {
            let v = value_for_key(&idx, k_tok).unwrap();
            if idx.stage1.kind[v as usize] != Kind::Quote {
                continue;
            }
            // Compare value string body byte-for-byte
            let off = idx.stage1.offset[v as usize] as usize;
            // body between quotes; find closing quote
            let mut e = off + 1;
            while e < buf.len() && buf[e] != b'"' {
                if buf[e] == b'\\' {
                    e += 2;
                } else {
                    e += 1;
                }
            }
            if &buf[off + 1..e] == b"test" {
                // Walk to enclosing object container
                let parent_obj = idx.parent[k_tok as usize];
                hits.push(parent_obj as u32);
            }
        }
        assert_eq!(hits.len(), 1);
        // The one hit is the inner obj containing {"x":"test"}
        let h = hits[0] as usize;
        assert_eq!(idx.stage1.kind[h], Kind::ObjOpen);
        assert_eq!(idx.stage1.offset[h], 5); // byte offset of inner '{' for "a"
    }

    #[test]
    fn empty_doc() {
        let buf = br#"{}"#;
        let idx = StructIndex::build(buf).unwrap();
        let kb = KeyBitmaps::build(&idx, buf);
        assert_eq!(kb.dict.len(), 0);
        assert_eq!(kb.find_key("anything", None), Vec::<u32>::new());
    }

    #[test]
    fn escaped_key() {
        let buf = br#"{"a\nb":42}"#;
        let idx = StructIndex::build(buf).unwrap();
        let kb = KeyBitmaps::build(&idx, buf);
        // Decoded key contains a real newline
        let dict_strs: Vec<&str> = kb.dict.iter().map(|s| &**s).collect();
        assert_eq!(dict_strs, vec!["a\nb"]);
        let hits = kb.find_key("a\nb", None);
        assert_eq!(hits.len(), 1);
    }
}
