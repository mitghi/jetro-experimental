use jetro_experimental::keys::{value_for_key, KeyBitmaps};
use jetro_experimental::{run, Kind, Role, StructIndex};

fn main() {
    let src = br#"{"test":"a","b":[1,2,3,4]}"#;
    let s = run(src).unwrap();
    println!("input: {}", std::str::from_utf8(src).unwrap());
    println!("len: {} structurals", s.len());
    println!();
    println!("idx  off  depth  kind        char");
    println!("---  ---  -----  ----------  ----");
    for i in 0..s.len() {
        let off = s.offset[i];
        let kind = s.kind[i];
        let d = s.depth[i];
        let c = src[off as usize] as char;
        println!("{:>3}  {:>3}  {:>5}  {:<10?}  {}", i, off, d, kind, c);
    }
    println!();

    // small demo: count number of objects
    let n_obj = s.kind.iter().filter(|k| **k == Kind::ObjOpen).count();
    let n_arr = s.kind.iter().filter(|k| **k == Kind::ArrOpen).count();
    let n_str = s.kind.iter().filter(|k| **k == Kind::Quote).count();
    println!("objects: {}  arrays: {}  strings: {}", n_obj, n_arr, n_str);

    println!();
    println!("=== StructIndex ===");
    let idx = StructIndex::build(src).unwrap();
    println!("tokens (with scalars): {}", idx.stage1.len());
    println!();
    println!("idx  off  depth  kind        tape  parent  close_of");
    println!("---  ---  -----  ----------  ----  ------  --------");
    for i in 0..idx.stage1.len() {
        let off = idx.stage1.offset[i];
        let kind = idx.stage1.kind[i];
        let d = idx.stage1.depth[i];
        let t = idx.tape_of[i];
        let p = idx.parent[i];
        let c = idx.close_of[i];
        let t_str = if t == u32::MAX { "-".into() } else { format!("{}", t) };
        let c_str = if c < 0 { "-".into() } else { format!("{}", c) };
        println!(
            "{:>3}  {:>3}  {:>5}  {:<10?}  {:>4}  {:>6}  {:>8}",
            i, off, d, kind, t_str, p, c_str
        );
    }
    println!();

    // Demo lookup: find object containing byte position 18 (inside the array)
    println!("=== lookups ===");
    for pos in [3u32, 9, 18, 22] {
        let cont = idx.container_at(pos);
        match cont {
            Some(t) => {
                let chain: Vec<u32> = idx.parent_chain_tokens(t).collect();
                let tape = idx.tape_of[t as usize];
                let parents_tape = idx.parents_as_tape(t);
                println!(
                    "pos={:<2}  innermost_tok={}  tape={}  parent_chain_tokens={:?}  parents_as_tape={:?}",
                    pos, t, tape, chain, parents_tape
                );
            }
            None => println!("pos={:<2}  no container", pos),
        }
    }

    // ---------------- Mison-style key bitmap demo ----------------
    println!();
    println!("=== Mison key bitmaps ===");
    let mison_src = br#"{"a":{"x":"test"},"b":{"x":"nope"},"c":{"y":"test"}}"#;
    println!("input: {}", std::str::from_utf8(mison_src).unwrap());
    let m_idx = StructIndex::build(mison_src).unwrap();
    let kb = KeyBitmaps::build(&m_idx, mison_src);
    println!("dict: {:?}", kb.dict);
    println!("'x' tokens at any depth: {:?}", kb.find_key("x", None));
    println!("'x' tokens at depth 2:   {:?}", kb.find_key("x", Some(2)));
    println!();
    println!("--- $..find(x == \"test\") via Mison-style scan ---");
    for k_tok in kb.find_key("x", None) {
        let v = value_for_key(&m_idx, k_tok).unwrap() as usize;
        if m_idx.stage1.kind[v] != Kind::Quote {
            continue;
        }
        let off = m_idx.stage1.offset[v] as usize;
        let mut e = off + 1;
        while e < mison_src.len() && mison_src[e] != b'"' {
            if mison_src[e] == b'\\' { e += 2 } else { e += 1 }
        }
        let body = &mison_src[off + 1..e];
        let _ = Role::Key; // ensure re-export usable
        let parent = m_idx.parent[k_tok as usize];
        let parent_off = if parent >= 0 { m_idx.stage1.offset[parent as usize] } else { u32::MAX };
        println!(
            "  key_tok={} value=\"{}\" {} -> enclosing obj at byte {}",
            k_tok,
            std::str::from_utf8(body).unwrap_or("?"),
            if body == b"test" { "[HIT]" } else { "" },
            parent_off
        );
    }
}
