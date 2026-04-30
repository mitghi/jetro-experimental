//! Criterion benches for jetro-experimental substrate operations.
//!
//! Run:
//!   cargo bench --bench substrate --features simd-json,fast-numbers
//!
//! Bench groups:
//!   1. parse_with_index   — fused tape + index build vs separate passes
//!   2. find_eq            — Mison key bitmap query
//!   3. count_key          — popcount-only
//!   4. key_hits_iter      — iter cache vs old destructive iter
//!   5. parse_f64          — fast-float vs str::parse
//!   6. json_string_eq     — SIMD escape probe vs naive

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use jetro_experimental::{
    count_key, find_eq, from_bytes, json_string_eq, parse, parse_f64, BuildOptions, TokenId,
};

const N: usize = 5_000;

fn build_doc(n: usize) -> Vec<u8> {
    let mut s = String::from(r#"{"data":["#);
    let cities = ["NYC", "SF", "LA", "Boston", "Seattle", "Austin", "Miami", "Chicago"];
    for i in 0..n {
        if i > 0 {
            s.push(',');
        }
        let city = cities[i % cities.len()];
        let mut items = String::from("[");
        let item_count = 3 + (i % 5);
        for k in 0..item_count {
            if k > 0 {
                items.push(',');
            }
            items.push_str(&format!(
                r#"{{"sku":"S{i}_{k}","qty":{},"price":{}}}"#,
                (k + 1) * 2,
                ((i * 7 + k * 13) % 200) as f64 + 9.99
            ));
        }
        items.push(']');
        s.push_str(&format!(
            r#"{{"id":{i},"user":{{"name":"user_{i}","city":"{city}"}},"items":{items},"active":{},"score":{}}}"#,
            if i % 3 == 0 { "true" } else { "false" },
            (i * 37) % 1000,
        ));
    }
    s.push_str("]}");
    s.into_bytes()
}

fn bench_parse(c: &mut Criterion) {
    let doc = build_doc(N);
    let mut g = c.benchmark_group("parse");
    g.throughput(criterion::Throughput::Bytes(doc.len() as u64));

    g.bench_function("parse_with_index_fused", |b| {
        b.iter(|| {
            let p = parse(black_box(doc.clone())).unwrap();
            black_box(p.index.token_count());
        });
    });

    g.bench_function("from_bytes_separate_pass", |b| {
        b.iter(|| {
            let idx = from_bytes(black_box(&doc)).unwrap();
            black_box(idx.token_count());
        });
    });

    g.bench_function("from_bytes_minimal_no_keys", |b| {
        b.iter(|| {
            let idx = jetro_experimental::from_bytes_with(
                black_box(&doc),
                BuildOptions::minimal(),
            )
            .unwrap();
            black_box(idx.token_count());
        });
    });

    g.finish();
}

fn bench_queries(c: &mut Criterion) {
    let doc = build_doc(N);
    let idx = from_bytes(&doc).unwrap();
    let mut g = c.benchmark_group("queries");

    g.bench_function("count_key_score", |b| {
        b.iter(|| {
            black_box(count_key(black_box(&idx), "score"));
        });
    });

    g.bench_function("find_eq_active_true", |b| {
        b.iter(|| {
            let n = find_eq(black_box(&idx), &doc, "active", b"true").count();
            black_box(n);
        });
    });

    g.bench_function("keys_named_score_collect", |b| {
        b.iter(|| {
            let mut buf: Vec<TokenId> = Vec::new();
            idx.keys_named("score", None).collect_into(&mut buf);
            black_box(buf.len());
        });
    });

    g.bench_function("keys_named_score_first", |b| {
        b.iter(|| {
            black_box(idx.keys_named("score", None).first());
        });
    });

    g.bench_function("container_at_byte_mid", |b| {
        let pos = (doc.len() / 2) as u32;
        b.iter(|| {
            black_box(idx.container_at_byte(black_box(pos)));
        });
    });

    g.finish();
}

fn bench_compare(c: &mut Criterion) {
    let mut g = c.benchmark_group("compare");

    let small = b"\"PushEvent\"".to_vec();
    let small_lit = b"PushEvent".to_vec();
    let escaped = b"\"with\\nescape\"".to_vec();
    let escaped_lit = b"with\nescape".to_vec();
    let large = format!("\"{}\"", "x".repeat(256)).into_bytes();
    let large_lit = "x".repeat(256).into_bytes();

    g.bench_function("string_eq_small_no_escape", |b| {
        b.iter(|| {
            black_box(json_string_eq(black_box(&small), black_box(&small_lit)));
        });
    });

    g.bench_function("string_eq_with_escape", |b| {
        b.iter(|| {
            black_box(json_string_eq(black_box(&escaped), black_box(&escaped_lit)));
        });
    });

    g.bench_function("string_eq_large_no_escape", |b| {
        b.iter(|| {
            black_box(json_string_eq(black_box(&large), black_box(&large_lit)));
        });
    });

    g.finish();
}

fn bench_numbers(c: &mut Criterion) {
    let mut g = c.benchmark_group("numbers");

    let nums: &[&[u8]] = &[
        b"42",
        b"3.14159",
        b"1e10",
        b"-1234.5678",
        b"99.99999999999999",
    ];

    g.bench_function("parse_f64", |b| {
        b.iter(|| {
            for n in nums {
                black_box(parse_f64(black_box(n)));
            }
        });
    });

    g.bench_function("std_parse_f64", |b| {
        b.iter(|| {
            for n in nums {
                let s = std::str::from_utf8(n).unwrap();
                black_box(s.parse::<f64>().ok());
            }
        });
    });

    g.finish();
}

criterion_group!(benches, bench_parse, bench_queries, bench_compare, bench_numbers);
criterion_main!(benches);
