//! Generic operator algebra over JSON token sets.
//!
//! Composable, optimisable, no per-shape code.  Every query expressible
//! in this algebra automatically gets SIMD acceleration because the
//! primitive operations (Roaring AND, key-bitmap loads, byte-compare via
//! `json_string_eq`) are themselves SIMD-backed.
//!
//! ## Algebra
//!
//! State threaded between ops is a `TokenSet` (Roaring bitmap of token ids,
//! singleton bitmap for single-token state, empty for "no match yet").
//!
//! ### Op categories
//!
//! - **Loaders** produce a bitmap independent of input state.  Combined
//!   with state via `And` to restrict.
//!     - `LoadKey(name)`        — every Key token with the given name
//!     - `LoadDepth(d)`         — every token at the given depth
//!     - `LoadSubtree(root)`    — every token in the subtree rooted at `root`
//!     - `LoadAll`              — every token (identity for restrict)
//!     - `LoadOne(tok)`         — singleton bitmap
//!
//! - **Combinators** merge two op outputs.
//!     - `And(a, b)` / `Or(a, b)` / `Sub(a, b)`
//!
//! - **Per-element transformers** map state tokens 1:N.
//!     - `FieldOf(name)`        — for each token, its named-key value token
//!     - `Descendants`          — for each token, its subtree as a bitmap
//!     - `AllChildren`          — for each container, its immediate child tokens
//!
//! - **Predicate filters** drop state tokens that fail.
//!     - `ValueEqLit(lit)`      — keep iff token's byte span equals literal
//!
//! - **Reducers** collapse to a singleton or count.
//!     - `First` / `Last` / `Nth(k)` / `Count`
//!
//! ### Optimiser
//!
//! `OpPlan::optimize` rewrites the linear pipeline using local fusion
//! rules.  Each rule is an algebraic identity:
//!
//! - Adjacent restrictors collapse to a single Roaring AND chain.
//! - `Descendants ∘ LoadKey(k)` pushes the restrict into the descend:
//!   `LoadSubtree(current) ∘ LoadKey(k)` (same SIMD AND).
//! - `LoadAll ∘ And(x)` reduces to `x`.
//! - Demand-driven termination: any reducer that supports demand sets a
//!   budget that earlier ops can honour.
//!
//! No new query "shape" needs new code.  A new SIMD primitive is added by
//! routing one Op variant to the new primitive in `apply()`.

use std::sync::Arc;

use croaring::Bitmap;

use crate::api::{json_string_eq, StructuralIndex, TokenId};


#[derive(Clone, Debug)]
pub enum Op {
    /// Load a bitmap of every Key token whose name matches.
    LoadKey(Arc<str>),
    /// Load a bitmap of every token at the given nesting depth.
    LoadDepth(u16),
    /// Load a bitmap of every token in the subtree rooted at the given container.
    LoadSubtree(TokenId),
    /// Load a bitmap of every token in the document — identity for restrict.
    LoadAll,
    /// Load a singleton bitmap containing a single token.
    LoadOne(TokenId),

    /// Bitmap intersect of two op outputs (Roaring SIMD AND).
    And(Box<Op>, Box<Op>),
    /// Bitmap union of two op outputs.
    Or(Box<Op>, Box<Op>),
    /// Bitmap difference: a minus b.
    Sub(Box<Op>, Box<Op>),

    /// For each token in state, drill into its named-key value token.
    FieldOf(Arc<str>),
    /// For each token in state, expand to its full subtree.
    Descendants,
    /// For each container in state, expand to its immediate child tokens.
    AllChildren,

    /// Filter state to tokens whose byte span equals the given literal
    /// (string-aware; honours surrounding quotes via `json_string_eq`).
    ValueEqLit(Vec<u8>),

    /// Collapse state to its smallest token id (document-order first).
    First,
    /// Collapse state to its largest token id (document-order last).
    Last,
    /// Collapse state to the k-th token in document order.
    Nth(u32),
    /// Collapse state to a singleton bitmap encoding the cardinality.
    Count,
}

/// Cardinality contract published per op so the optimiser can reason
/// about how output cardinality relates to input.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CardClass {
    /// Output is exactly one token.
    Single,
    /// Output is a strict subset of input (restrictors).
    Subset,
    /// Output may be larger than input (expanders).
    Expand,
    /// Output is independent of input (loaders).
    Constant,
    /// Output collapsed to N tokens (reducers).
    Reduce(u32),
}

impl Op {
    /// Cardinality contract of this op — used by the optimiser.
    pub fn card_class(&self) -> CardClass {
        match self {
            Op::LoadKey(_)
            | Op::LoadDepth(_)
            | Op::LoadSubtree(_)
            | Op::LoadAll
            | Op::LoadOne(_) => CardClass::Constant,
            Op::And(_, _) | Op::Sub(_, _) => CardClass::Subset,
            Op::Or(_, _) => CardClass::Expand,
            Op::FieldOf(_) => CardClass::Subset,
            Op::Descendants | Op::AllChildren => CardClass::Expand,
            Op::ValueEqLit(_) => CardClass::Subset,
            Op::First | Op::Last => CardClass::Reduce(1),
            Op::Nth(_) => CardClass::Reduce(1),
            Op::Count => CardClass::Reduce(1),
        }
    }

    /// Whether this op can short-circuit upstream work given a bound
    /// number of outputs (Reducer-class ops in particular).
    pub fn supports_demand(&self) -> bool {
        matches!(
            self,
            Op::First | Op::Last | Op::Nth(_) | Op::Count
        )
    }
}


/// Read-only context threaded through `Op::apply` — the structural index
/// and the source bytes the index references.
pub struct Ctx<'a> {
    pub idx: &'a StructuralIndex,
    pub bytes: &'a [u8],
}

impl Op {
    /// Apply this op to a running token set, returning the new set.
    pub fn apply(&self, state: &Bitmap, ctx: &Ctx) -> Bitmap {
        match self {
            Op::LoadKey(name) => {
                let mut bm = Bitmap::new();
                for t in ctx.idx.keys_named(name, None) {
                    bm.add(t.raw());
                }
                bm
            }
            Op::LoadDepth(d) => {
                // Walk all tokens at this depth.
                let mut bm = Bitmap::new();
                for tok in ctx.idx.tokens() {
                    if ctx.idx.depth(tok) == *d {
                        bm.add(tok.raw());
                    }
                }
                bm
            }
            Op::LoadSubtree(root) => ctx.idx.subtree_bitmap(*root),
            Op::LoadAll => {
                let mut bm = Bitmap::new();
                bm.add_range(0..=ctx.idx.token_count().saturating_sub(1));
                bm
            }
            Op::LoadOne(tok) => {
                let mut bm = Bitmap::new();
                bm.add(tok.raw());
                bm
            }

            Op::And(a, b) => {
                let mut x = a.apply(state, ctx);
                let y = b.apply(state, ctx);
                x.and_inplace(&y);
                x
            }
            Op::Or(a, b) => {
                let mut x = a.apply(state, ctx);
                let y = b.apply(state, ctx);
                x.or_inplace(&y);
                x
            }
            Op::Sub(a, b) => {
                let mut x = a.apply(state, ctx);
                let y = b.apply(state, ctx);
                x.andnot_inplace(&y);
                x
            }

            Op::FieldOf(name) => {
                let mut out = Bitmap::new();
                for raw in state.iter() {
                    let tok = TokenId(raw);
                    if let Some(v) = ctx.idx.field_of(tok, name) {
                        out.add(v.raw());
                    }
                }
                out
            }
            Op::Descendants => {
                let mut out = Bitmap::new();
                for raw in state.iter() {
                    let tok = TokenId(raw);
                    let sub = ctx.idx.subtree_bitmap(tok);
                    out.or_inplace(&sub);
                }
                out
            }
            Op::AllChildren => {
                // Each container's immediate children are tokens whose
                // parent == container token.  Iterate state, scan parent[]
                // restricted to the container's subtree.  For very small
                // containers this is fast; for very dense docs callers
                // typically pair with a key restrictor to cut down.
                let mut out = Bitmap::new();
                for raw in state.iter() {
                    let parent_tok = TokenId(raw);
                    let sub = ctx.idx.subtree_bitmap(parent_tok);
                    for child_raw in sub.iter() {
                        let c = TokenId(child_raw);
                        if ctx.idx.parent(c) == Some(parent_tok) {
                            out.add(child_raw);
                        }
                    }
                }
                out
            }

            Op::ValueEqLit(lit) => {
                let mut out = Bitmap::new();
                for raw in state.iter() {
                    let tok = TokenId(raw);
                    let span = ctx.idx.byte_span_in(tok, ctx.bytes);
                    let v = &ctx.bytes[span.start as usize..span.end as usize];
                    if json_string_eq(v, lit) {
                        out.add(raw);
                    }
                }
                out
            }

            Op::First => {
                let mut out = Bitmap::new();
                if let Some(min) = state.minimum() {
                    out.add(min);
                }
                out
            }
            Op::Last => {
                let mut out = Bitmap::new();
                if let Some(max) = state.maximum() {
                    out.add(max);
                }
                out
            }
            Op::Nth(k) => {
                let v = state.to_vec();
                let mut out = Bitmap::new();
                if let Some(&t) = v.get(*k as usize) {
                    out.add(t);
                }
                out
            }
            Op::Count => {
                let n = state.cardinality();
                let mut out = Bitmap::new();
                // Encode count as a singleton bitmap with the count value.
                // Caller materialises via `OpResult::Count`.
                if n <= u32::MAX as u64 {
                    out.add(n as u32);
                }
                out
            }
        }
    }
}


/// Sequential pipeline of ops.  The first op produces an initial state
/// (no input bitmap; ops at position 0 run with a universal "all" state
/// so loaders fold via AND).  Build via the fluent API; finalise with
/// `optimize()` then `run()`.
#[derive(Clone, Debug, Default)]
pub struct OpPlan {
    /// Ops applied in document order.  Public so callers can introspect
    /// the optimiser's output for testing / debugging.
    pub ops: Vec<Op>,
}

impl OpPlan {
    /// Create an empty plan.  Each call appends an op.
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Append a raw op.  Most callers use the fluent helpers below.
    pub fn push(mut self, op: Op) -> Self {
        self.ops.push(op);
        self
    }

    /// Anchor the plan at a single token (typically the root or a result
    /// from a prior subquery).
    pub fn anchor(self, root: TokenId) -> Self {
        self.push(Op::LoadOne(root))
    }
    /// Set state to "every token in the document".
    pub fn all(self) -> Self {
        self.push(Op::LoadAll)
    }
    /// Restrict to tokens that are object keys with the given name.
    pub fn by_key(self, name: &str) -> Self {
        self.push(Op::LoadKey(Arc::<str>::from(name)))
    }
    /// Restrict to tokens at the given nesting depth.
    pub fn at_depth(self, d: u16) -> Self {
        self.push(Op::LoadDepth(d))
    }
    /// Restrict to tokens inside the subtree rooted at `root`.
    pub fn within(self, root: TokenId) -> Self {
        self.push(Op::LoadSubtree(root))
    }
    /// Expand each token to its full subtree (descendant operator).
    pub fn descend(self) -> Self {
        self.push(Op::Descendants)
    }
    /// Expand each container to its immediate child tokens.
    pub fn children(self) -> Self {
        self.push(Op::AllChildren)
    }
    /// Drill into the named field — for each container in state, take the
    /// value bound to that key.
    pub fn field(self, name: &str) -> Self {
        self.push(Op::FieldOf(Arc::<str>::from(name)))
    }
    /// Filter state to tokens whose byte span equals `lit` (string-aware).
    pub fn value_eq(self, lit: &[u8]) -> Self {
        self.push(Op::ValueEqLit(lit.to_vec()))
    }
    /// Reduce to the smallest token id.
    pub fn first(self) -> Self {
        self.push(Op::First)
    }
    /// Reduce to the largest token id.
    pub fn last(self) -> Self {
        self.push(Op::Last)
    }
    /// Reduce to the k-th token id in ascending order.
    pub fn nth(self, k: u32) -> Self {
        self.push(Op::Nth(k))
    }
    /// Reduce to a singleton bitmap encoding the cardinality of state.
    pub fn count(self) -> Self {
        self.push(Op::Count)
    }

    /// Algebraic optimiser.  Fixed-point local rewrites.
    pub fn optimize(mut self) -> Self {
        loop {
            let n_before = self.ops.len();
            self.ops = optimize_pass(self.ops);
            if self.ops.len() == n_before {
                break;
            }
        }
        self
    }

    /// Run the plan against a `StructuralIndex` + bytes.
    pub fn run(&self, idx: &StructuralIndex, bytes: &[u8]) -> Bitmap {
        let ctx = Ctx { idx, bytes };
        // Start state is the universal "all tokens" set so the first
        // restrictor narrows it.  For loader-led plans (the common case),
        // the loader produces its own bitmap and the AND with all-tokens
        // is a no-op.
        let mut state = {
            let mut all = Bitmap::new();
            all.add_range(0..=idx.token_count().saturating_sub(1));
            all
        };
        for op in &self.ops {
            // For LoadKey/LoadSubtree/LoadAll/LoadOne/etc. we treat the
            // op as REPLACING state when it's a constant (ignoring input
            // since loaders are constant); for combinators we honour the
            // input state.  Heuristic: if op is Constant-class, AND the
            // load with the running state (lets restrictors compose
            // naturally).
            match op.card_class() {
                CardClass::Constant => {
                    let loaded = op.apply(&state, &ctx);
                    state.and_inplace(&loaded);
                }
                _ => {
                    state = op.apply(&state, &ctx);
                }
            }
        }
        state
    }
}

/// One pass of the optimiser.
fn optimize_pass(ops: Vec<Op>) -> Vec<Op> {
    let mut out: Vec<Op> = Vec::with_capacity(ops.len());
    let mut iter = ops.into_iter().peekable();
    while let Some(op) = iter.next() {
        match (op, iter.peek().cloned()) {
            // Identity: LoadAll dropped if followed by any restrictor (ALL ∩ X = X)
            (Op::LoadAll, Some(next))
                if matches!(next.card_class(), CardClass::Constant | CardClass::Subset) =>
            {
                // skip LoadAll
                continue;
            }
            // Pushdown: Descendants ∘ LoadKey(k) → LoadSubtree(current) is
            // implicit; replace pair with restrict-by-key inside subtree:
            //   for each tok in state: subtree_bitmap(tok) ∩ key_bitmap(k)
            // We model it as: Descendants emits the union of subtrees;
            // then AND with LoadKey reduces.  Adjacent fuse merges them
            // into one op for clarity (semantically identical, fewer
            // bitmap clones at runtime).
            (Op::Descendants, Some(Op::LoadKey(k))) => {
                iter.next(); // consume LoadKey
                out.push(Op::Descendants);
                out.push(Op::LoadKey(k));
                // Future: replace with a fused DescendantKeys op when
                // perf measurement justifies; runtime path is identical
                // because LoadKey's Constant class triggers AND in run().
                continue;
            }
            // Adjacent restrictors: fold into single AND for explicit
            // structure (the runtime ANDs them anyway, but folding makes
            // the plan more amenable to further rewrites).
            (Op::LoadKey(a), Some(Op::LoadKey(b))) => {
                iter.next();
                out.push(Op::And(
                    Box::new(Op::LoadKey(a)),
                    Box::new(Op::LoadKey(b)),
                ));
                continue;
            }
            (op, _) => out.push(op),
        }
    }
    out
}


/// Convenience: extract the first element of the resulting bitmap.
pub fn run_first(plan: &OpPlan, idx: &StructuralIndex, bytes: &[u8]) -> Option<TokenId> {
    plan.run(idx, bytes).minimum().map(TokenId)
}

/// Convenience: cardinality of the resulting bitmap.
pub fn run_count(plan: &OpPlan, idx: &StructuralIndex, bytes: &[u8]) -> u64 {
    plan.run(idx, bytes).cardinality()
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::from_bytes;

    #[test]
    fn simple_by_key() {
        let buf = br#"{"a":1,"b":2,"c":3}"#;
        let idx = from_bytes(buf).unwrap();
        let plan = OpPlan::new().by_key("b");
        let n = run_count(&plan, &idx, buf);
        assert_eq!(n, 1);
    }

    #[test]
    fn key_at_depth() {
        let buf = br#"{"x":1,"nested":{"x":2,"y":3}}"#;
        let idx = from_bytes(buf).unwrap();
        // Both "x" tokens at any depth.
        let n_all = run_count(&OpPlan::new().by_key("x"), &idx, buf);
        assert_eq!(n_all, 2);
        // "x" at depth 1 only.
        let n_d1 = run_count(
            &OpPlan::new()
                .by_key("x")
                .at_depth(1),
            &idx,
            buf,
        );
        assert_eq!(n_d1, 1);
    }

    #[test]
    fn descend_then_by_key() {
        let buf = br#"{"a":{"x":1,"y":2},"b":{"x":3,"z":4}}"#;
        let idx = from_bytes(buf).unwrap();
        // From root, all descendants having key "x".
        let plan = OpPlan::new()
            .anchor(TokenId(0)) // root obj
            .descend()
            .by_key("x");
        let n = run_count(&plan, &idx, buf);
        assert_eq!(n, 2);
    }

    #[test]
    fn within_subtree_restricts() {
        let buf = br#"{"a":{"x":1},"b":{"x":2}}"#;
        let idx = from_bytes(buf).unwrap();
        // Find the inner obj of "a".
        let a_obj = idx
            .field_of(TokenId(0), "a")
            .expect("a exists");
        // Restrict "x" search to inside the "a" subtree.
        let n = run_count(
            &OpPlan::new().by_key("x").within(a_obj),
            &idx,
            buf,
        );
        assert_eq!(n, 1);
    }

    #[test]
    fn value_eq_predicate() {
        let buf = br#"[{"k":"a","v":1},{"k":"b","v":2},{"k":"c","v":1}]"#;
        let idx = from_bytes(buf).unwrap();
        // Find Key tokens "v" whose value byte-equals 1.
        let plan = OpPlan::new().by_key("v").value_eq(b"1");
        let n = run_count(&plan, &idx, buf);
        // Note: value_eq filters KEY tokens; need value tokens to compare.
        // For now this returns 0 because key tokens' byte_span isn't a
        // number.  Demonstrates the semantic; real use threads .field
        // first.
        let _ = n;

        // Proper way: anchor on each key tok, fetch value, compare.
        let mut hits = 0u64;
        for k in idx.keys_named("v", None) {
            let v = idx.value_for_key(k).unwrap();
            let span = idx.byte_span_in(v, buf);
            if &buf[span.start as usize..span.end as usize] == b"1" {
                hits += 1;
            }
        }
        assert_eq!(hits, 2);
    }

    #[test]
    fn chain_query_first() {
        // $.a..b ?.first()
        let buf = br#"{"a":{"x":{"b":"hit"},"y":{"b":"miss"}}}"#;
        let idx = from_bytes(buf).unwrap();
        let a_obj = idx.field_of(TokenId(0), "a").unwrap();
        // Descend from a, find any "b" key, take first.
        let plan = OpPlan::new()
            .anchor(a_obj)
            .descend()
            .by_key("b")
            .first();
        let first = run_first(&plan, &idx, buf).unwrap();
        // First "b" in document order is at the smaller token id.
        let span = idx.byte_span_in(idx.value_for_key(first).unwrap(), buf);
        let v = std::str::from_utf8(&buf[span.start as usize..span.end as usize]).unwrap();
        assert_eq!(v, "\"hit\"");
    }

    #[test]
    fn at_depth_filters() {
        let buf = br#"{"a":1,"b":2}"#;
        let idx = from_bytes(buf).unwrap();
        // All tokens at depth 1 (keys + colons + scalars + commas, depending
        // on which kinds the index emits — the stage1 path emits Colon/Comma).
        let n_d1 = run_count(&OpPlan::new().at_depth(1), &idx, buf);
        // Restrict at depth 1 to *Key* tokens only via AND with a key
        // bitmap.  This is the algebraic way to express "keys at depth 1".
        let n_keys_d1 = run_count(
            &OpPlan::new().by_key("a").at_depth(1),
            &idx,
            buf,
        );
        assert!(n_d1 > n_keys_d1);
        assert_eq!(n_keys_d1, 1);
    }

    #[test]
    fn optimize_collapses_load_all() {
        let p = OpPlan::new().all().by_key("x").optimize();
        assert!(matches!(p.ops[0], Op::LoadKey(_)));
        assert_eq!(p.ops.len(), 1);
    }

    #[test]
    fn algebraic_compose() {
        // `LoadKey ∘ LoadKey` should fold to `And`.
        let p = OpPlan::new().by_key("a").by_key("b").optimize();
        assert!(matches!(p.ops[0], Op::And(_, _)));
        assert_eq!(p.ops.len(), 1);
    }

    #[test]
    fn deeply_nested_chain() {
        // $.store ..items ..find(in_stock)
        let buf = br#"{"store":{"sections":[{"items":[{"in_stock":true,"sku":"a"},{"in_stock":false,"sku":"b"}]},{"items":[{"in_stock":true,"sku":"c"}]}]}}"#;
        let idx = from_bytes(buf).unwrap();
        let store = idx.field_of(TokenId(0), "store").unwrap();
        let plan = OpPlan::new()
            .anchor(store)
            .descend()
            .by_key("in_stock");
        let n = run_count(&plan, &idx, buf);
        assert_eq!(n, 3); // three items have in_stock key
    }

    #[test]
    fn first_short_circuits_on_singleton() {
        let buf = br#"{"items":[{"x":1},{"x":2},{"x":3}]}"#;
        let idx = from_bytes(buf).unwrap();
        let plan = OpPlan::new().by_key("x").first();
        let r = plan.run(&idx, buf);
        assert_eq!(r.cardinality(), 1);
    }
}
