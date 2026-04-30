//! Structural index over a `Stage1` token stream.
//!
//! Builds three sidecar columns:
//!   - `parent[i]`   : index of enclosing container open (-1 for root)
//!   - `close_of[i]` : for ObjOpen/ArrOpen, index of matching close (-1 otherwise)
//!   - `tape_of[i]`  : tape index assigned to value-bearing tokens (Open/Close/Quote/Scalar)
//!
//! Plus query helpers: `token_at(pos)`, `container_at(pos)`, `parent_chain(...)`,
//! `value_token_at(pos)`, `parents_as_tape(...)`.
//!
//! Tape numbering convention used here:
//!   ObjOpen  -> 1 entry
//!   ObjClose -> 1 entry
//!   ArrOpen  -> 1 entry
//!   ArrClose -> 1 entry
//!   Quote    -> 1 entry  (each opening quote is a key OR string-value node)
//!   Scalar   -> 1 entry  (number/true/false/null)
//!   Colon, Comma -> no tape entry
//!
//! This matches simd-json-rs `Tape<'input> = Vec<Node<'input>>` Node-level indexing
//! closely; verify against your simd-json version before trusting tape mapping cross-tool.

use crate::stage1::{Kind, Stage1};

#[derive(Debug, Clone)]
pub struct StructIndex {
    pub stage1: Stage1,
    pub parent: Vec<i32>,
    pub close_of: Vec<i32>,
    pub tape_of: Vec<u32>,
}

impl StructIndex {
    /// Build the index in three passes:
    ///   1. stage1 over the raw bytes (already columnar)
    ///   2. fill_scalars: insert Scalar tokens for number/true/false/null literals
    ///   3. parent + close_of via stack walk; tape_of via running counter
    pub fn build(buf: &[u8]) -> Result<Self, &'static str> {
        let mut stage1 = crate::stage1::run(buf)?;
        fill_scalars(&mut stage1, buf);
        let (parent, close_of) = build_parent_close(&stage1)?;
        let tape_of = build_tape_of(&stage1);
        Ok(Self {
            stage1,
            parent,
            close_of,
            tape_of,
        })
    }

    /// Innermost container token enclosing `pos`. Walks parent chain until the
    /// container's [open_offset, close_offset] covers pos.
    pub fn container_at(&self, pos: u32) -> Option<u32> {
        let mut cur = token_at(&self.stage1, pos)? as i32;
        while cur >= 0 {
            let k = self.stage1.kind[cur as usize];
            if matches!(k, Kind::ObjOpen | Kind::ArrOpen) {
                let close = self.close_of[cur as usize];
                if close >= 0 {
                    let start = self.stage1.offset[cur as usize];
                    let end = self.stage1.offset[close as usize];
                    if pos >= start && pos <= end {
                        return Some(cur as u32);
                    }
                }
            }
            cur = self.parent[cur as usize];
        }
        None
    }

    /// Tape index for any token. `u32::MAX` for tokens with no tape entry
    /// (Colon, Comma).
    pub fn tape(&self, tok: u32) -> u32 {
        self.tape_of[tok as usize]
    }

    /// Tape indices of parents from innermost to root.
    pub fn parents_as_tape(&self, tok: u32) -> Vec<u32> {
        self.parent_chain_tokens(tok)
            .map(|t| self.tape_of[t as usize])
            .collect()
    }

    /// Iterator over parent container tokens (innermost first), as stage1 indices.
    pub fn parent_chain_tokens<'a>(&'a self, tok: u32) -> ParentChain<'a> {
        ParentChain {
            parent: &self.parent,
            cur: self.parent[tok as usize],
        }
    }

    /// Token of the value-bearing structural item closest to `pos`.
    /// - On a Quote byte             -> that Quote token (key or string value)
    /// - Inside a string body        -> the opening Quote
    /// - On a Scalar literal byte    -> that Scalar token
    /// - Returns None if pos is on a separator (`:` `,`) or a container bracket.
    pub fn value_token_at(&self, pos: u32) -> Option<u32> {
        let tok = token_at(&self.stage1, pos)? as usize;
        match self.stage1.kind[tok] {
            Kind::Quote | Kind::Scalar => Some(tok as u32),
            _ => None,
        }
    }
}

/// Iterator yielding parent container tokens, innermost first.
pub struct ParentChain<'a> {
    parent: &'a [i32],
    cur: i32,
}

impl<'a> Iterator for ParentChain<'a> {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        if self.cur < 0 {
            return None;
        }
        let out = self.cur as u32;
        self.cur = self.parent[out as usize];
        Some(out)
    }
}

/// Free-standing parent-chain iterator over a foreign parent[] column.
pub fn parent_chain<'a>(parent: &'a [i32], start: u32) -> ParentChain<'a> {
    ParentChain {
        parent,
        cur: parent[start as usize],
    }
}

/// Binary-search the structural-token whose byte_offset <= pos < next_offset.
/// stage1.offset is sorted ascending by construction.
pub fn token_at(s: &Stage1, pos: u32) -> Option<u32> {
    let p = s.offset.partition_point(|&o| o <= pos);
    if p == 0 {
        None
    } else {
        Some((p - 1) as u32)
    }
}


/// Scan for number/true/false/null literals adjacent to Colon, Comma, ArrOpen.
/// Inserts a `Kind::Scalar` token at the literal's first byte, with the same
/// `depth` as the trigger token.
pub fn fill_scalars(s: &mut Stage1, buf: &[u8]) {
    let n = s.len();
    let mut new_off = Vec::with_capacity(n + n / 4);
    let mut new_kind = Vec::with_capacity(new_off.capacity());
    let mut new_dep = Vec::with_capacity(new_off.capacity());

    for i in 0..n {
        new_off.push(s.offset[i]);
        new_kind.push(s.kind[i]);
        new_dep.push(s.depth[i]);

        if !matches!(s.kind[i], Kind::Colon | Kind::Comma | Kind::ArrOpen) {
            continue;
        }

        let next_struct = if i + 1 < n {
            s.offset[i + 1] as usize
        } else {
            buf.len()
        };
        let mut p = (s.offset[i] as usize) + 1;
        while p < next_struct && matches!(buf[p], b' ' | b'\t' | b'\n' | b'\r') {
            p += 1;
        }
        if p >= next_struct {
            continue;
        }
        let b = buf[p];
        let is_scalar_start = matches!(
            b,
            b'-' | b'0'..=b'9' | b't' | b'f' | b'n'
        );
        if is_scalar_start {
            // For ArrOpen, scalar lives one level deeper than the open's depth.
            // Open at depth D contains content at depth D+1. So the scalar's
            // depth is `depth_of_trigger + 1` for ArrOpen, but `depth_of_trigger`
            // for Colon and Comma (those are siblings of the literal).
            let d = match s.kind[i] {
                Kind::ArrOpen => s.depth[i] + 1,
                _ => s.depth[i],
            };
            new_off.push(p as u32);
            new_kind.push(Kind::Scalar);
            new_dep.push(d);
        }
    }

    s.offset = new_off;
    s.kind = new_kind;
    s.depth = new_dep;
}


fn build_parent_close(s: &Stage1) -> Result<(Vec<i32>, Vec<i32>), &'static str> {
    let n = s.len();
    let mut parent = vec![-1i32; n];
    let mut close_of = vec![-1i32; n];
    let mut stack: Vec<u32> = Vec::with_capacity(16);

    for i in 0..n {
        match s.kind[i] {
            Kind::ObjOpen | Kind::ArrOpen => {
                parent[i] = stack.last().copied().map(|x| x as i32).unwrap_or(-1);
                stack.push(i as u32);
            }
            Kind::ObjClose | Kind::ArrClose => {
                let open = stack.pop().ok_or("unbalanced close")? as usize;
                close_of[open] = i as i32;
                // parent of a close is the same as parent of its matching open
                parent[i] = parent[open];
            }
            _ => {
                parent[i] = stack.last().copied().map(|x| x as i32).unwrap_or(-1);
            }
        }
    }
    if !stack.is_empty() {
        return Err("unclosed container");
    }
    Ok((parent, close_of))
}


fn build_tape_of(s: &Stage1) -> Vec<u32> {
    let n = s.len();
    let mut tape_of = vec![u32::MAX; n];
    let mut t: u32 = 0;
    for i in 0..n {
        match s.kind[i] {
            Kind::ObjOpen
            | Kind::ObjClose
            | Kind::ArrOpen
            | Kind::ArrClose
            | Kind::Quote
            | Kind::Scalar => {
                tape_of[i] = t;
                t += 1;
            }
            Kind::Colon | Kind::Comma => {
                // not a tape entry
            }
        }
    }
    tape_of
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fills_number_scalars() {
        let buf = br#"{"a":1,"b":[2,3,4]}"#;
        let mut s = crate::stage1::run(buf).unwrap();
        let before = s.len();
        fill_scalars(&mut s, buf);
        // 4 scalars: 1, 2, 3, 4
        assert_eq!(s.len() - before, 4);
        let n_scalar = s.kind.iter().filter(|k| **k == Kind::Scalar).count();
        assert_eq!(n_scalar, 4);
    }

    #[test]
    fn fills_keyword_scalars() {
        let buf = br#"{"a":true,"b":false,"c":null}"#;
        let mut s = crate::stage1::run(buf).unwrap();
        fill_scalars(&mut s, buf);
        let n_scalar = s.kind.iter().filter(|k| **k == Kind::Scalar).count();
        assert_eq!(n_scalar, 3);
    }

    #[test]
    fn parent_close_basic() {
        let buf = br#"{"a":[1,2]}"#;
        let idx = StructIndex::build(buf).unwrap();
        // Token order with scalars filled:
        //  0  {       depth 0  parent -1
        //  1  "       depth 1  parent 0
        //  2  :       depth 1  parent 0
        //  3  [       depth 1  parent 0  close_of=8
        //  4  Scalar  depth 2  parent 3
        //  5  ,       depth 2  parent 3
        //  6  Scalar  depth 2  parent 3
        //  7  ]       depth 1  parent 0
        //  8  }       depth 0  parent -1
        assert_eq!(idx.stage1.kind[0], Kind::ObjOpen);
        assert_eq!(idx.parent[0], -1);
        assert_eq!(idx.close_of[0], 8);
        assert_eq!(idx.stage1.kind[3], Kind::ArrOpen);
        assert_eq!(idx.parent[3], 0);
        assert_eq!(idx.close_of[3], 7);
        // scalars under array
        assert_eq!(idx.stage1.kind[4], Kind::Scalar);
        assert_eq!(idx.parent[4], 3);
    }

    #[test]
    fn tape_of_monotonic() {
        let buf = br#"{"a":1,"b":[2,3]}"#;
        let idx = StructIndex::build(buf).unwrap();
        // every value-bearing token has a tape, separators don't
        let mut last: i64 = -1;
        for i in 0..idx.stage1.len() {
            let t = idx.tape_of[i];
            let needs_tape = !matches!(idx.stage1.kind[i], Kind::Colon | Kind::Comma);
            if needs_tape {
                assert_ne!(t, u32::MAX);
                assert!(t as i64 > last);
                last = t as i64;
            } else {
                assert_eq!(t, u32::MAX);
            }
        }
    }

    #[test]
    fn container_at_byte_inside_array() {
        let buf = br#"{"a":[1,2,3]}"#;
        let idx = StructIndex::build(buf).unwrap();
        // [ at byte 5, ] at byte 11 — pos 8 is the literal '2' inside the array
        let c = idx.container_at(8).unwrap();
        assert_eq!(idx.stage1.kind[c as usize], Kind::ArrOpen);
        assert_eq!(idx.stage1.offset[c as usize], 5);
    }

    #[test]
    fn container_at_byte_inside_string() {
        let buf = br#"{"key":"value"}"#;
        let idx = StructIndex::build(buf).unwrap();
        // 'a' inside "value" is at byte 9; outermost obj is the only container
        let c = idx.container_at(9).unwrap();
        assert_eq!(idx.stage1.kind[c as usize], Kind::ObjOpen);
    }

    #[test]
    fn nested_innermost_wins() {
        let buf = br#"{"a":{"x":"hit"}}"#;
        // Layout (offsets):
        //  0   { (outer)
        //  1   "a"
        //  4   :
        //  5   { (inner)
        //  6   "x"
        //  9   :
        //  10  "hit"
        //  15  }
        //  16  }
        let idx = StructIndex::build(buf).unwrap();
        // pos 11 is inside "hit" — innermost should be the inner obj at byte 5
        let c = idx.container_at(11).unwrap();
        assert_eq!(idx.stage1.offset[c as usize], 5);
        assert_eq!(idx.stage1.kind[c as usize], Kind::ObjOpen);
    }

    #[test]
    fn parent_chain_walks_to_root() {
        let buf = br#"{"a":{"b":{"c":42}}}"#;
        let idx = StructIndex::build(buf).unwrap();
        // pos of '4' in 42
        let pos = buf.iter().position(|&b| b == b'4').unwrap() as u32;
        let inner = idx.container_at(pos).unwrap();
        let chain: Vec<u32> = idx.parent_chain_tokens(inner).collect();
        // inner is the {"c":...} obj; chain is its ancestors (not itself)
        assert!(!chain.is_empty());
        for &t in &chain {
            assert!(matches!(
                idx.stage1.kind[t as usize],
                Kind::ObjOpen | Kind::ArrOpen
            ));
        }
        // last in chain must be root
        let last = *chain.last().unwrap();
        assert_eq!(idx.parent[last as usize], -1);
    }

    #[test]
    fn value_token_at_string_body() {
        let buf = br#"{"k":"hit"}"#;
        let idx = StructIndex::build(buf).unwrap();
        // 'h' of "hit" is at byte 6; opening quote of "hit" is at 5
        let v = idx.value_token_at(6).unwrap();
        assert_eq!(idx.stage1.kind[v as usize], Kind::Quote);
        assert_eq!(idx.stage1.offset[v as usize], 5);
    }

    #[test]
    fn value_token_at_number() {
        let buf = br#"{"k":42}"#;
        let idx = StructIndex::build(buf).unwrap();
        // 42 is at byte 5..7; pos 6 is on '2' but token_at returns the Scalar at 5
        let v = idx.value_token_at(6).unwrap();
        assert_eq!(idx.stage1.kind[v as usize], Kind::Scalar);
        assert_eq!(idx.stage1.offset[v as usize], 5);
    }

    #[test]
    fn parents_as_tape_is_increasing_outwards() {
        let buf = br#"{"a":{"b":[1]}}"#;
        let idx = StructIndex::build(buf).unwrap();
        let pos = buf.iter().position(|&b| b == b'1').unwrap() as u32;
        let inner = idx.container_at(pos).unwrap();
        let parents = idx.parents_as_tape(inner);
        // tape indices walk outward — each parent's tape index must be lower
        // than the inner container's (since parents open before children)
        let inner_tape = idx.tape_of[inner as usize];
        for &p in &parents {
            assert!(p < inner_tape);
        }
    }

    #[test]
    fn empty_container() {
        let buf = br#"{"a":{},"b":[]}"#;
        let idx = StructIndex::build(buf).unwrap();
        // Find the inner empty obj ({}) and array ([])
        for i in 0..idx.stage1.len() {
            let k = idx.stage1.kind[i];
            if matches!(k, Kind::ObjOpen | Kind::ArrOpen) && idx.stage1.depth[i] == 1 {
                let close = idx.close_of[i];
                assert_eq!(close as usize, i + 1);
            }
        }
    }
}
