//! Contract tests: every mutation class must REJECT; valid and subtle
//! schedules must ACCEPT. Fixed seeds throughout.

use serde_json::{json, Value};
use validator::gen::{gen_value, mutate, Rng, MUTATIONS};
use validator::{check_str, ViolationKind};

fn accept(text: &Value) -> bool {
    check_str(&serde_json::to_string(text).unwrap())
        .unwrap()
        .accepted()
}

fn kinds(text: &Value) -> Vec<ViolationKind> {
    check_str(&serde_json::to_string(text).unwrap())
        .unwrap()
        .violations
        .iter()
        .map(|v| v.kind.clone())
        .collect()
}

#[test]
fn valid_generated_schedules_accept() {
    for &(layers, blocks) in &[(1, 1), (2, 4), (3, 8), (28, 36), (28, 72), (4, 96)] {
        let v = gen_value(layers, blocks);
        let verdict = check_str(&serde_json::to_string(&v).unwrap()).unwrap();
        assert!(
            verdict.accepted(),
            "layers={layers} blocks={blocks}: {:?}",
            verdict.violations
        );
    }
}

#[test]
fn every_mutation_class_rejects() {
    // several seeds; each class must reject every time it applies
    for seed in 0..20u64 {
        let mut rng = Rng::new(0xC0FFEE ^ seed.wrapping_mul(0x9E3779B97F4A7C15));
        for name in MUTATIONS {
            for _ in 0..4 {
                let mut v = gen_value(3, 8);
                if !mutate(&mut v, name, &mut rng, 3, 8) {
                    continue;
                }
                let verdict = check_str(&serde_json::to_string(&v).unwrap()).unwrap();
                assert!(!verdict.accepted(), "mutation {name} accepted (seed {seed})");
            }
        }
    }
}

#[test]
fn mutation_kinds_are_as_labeled() {
    // spot-check the violation kind matches the corruption class
    let mut rng = Rng::new(42);
    let expect = [
        ("delete_set", ViolationKind::InexactFlag),
        ("delete_wait", ViolationKind::Race),
        ("lower_wait", ViolationKind::InexactFlag),
        ("raise_add", ViolationKind::InexactFlag),
        ("cyclic_wait_cross_block", ViolationKind::Deadlock),
        ("cyclic_wait_same_block", ViolationKind::Deadlock),
        ("overlap_writes", ViolationKind::Race),
        ("raw_race", ViolationKind::Race),
        ("oob_range", ViolationKind::Structural),
        ("dup_order", ViolationKind::Structural),
        ("undeclared_ref", ViolationKind::Structural),
    ];
    for (name, kind) in expect {
        let mut v = gen_value(3, 8);
        assert!(mutate(&mut v, name, &mut rng, 3, 8), "{name} did not apply");
        let ks = kinds(&v);
        assert!(ks.contains(&kind), "{name}: got {ks:?}, want {kind:?}");
    }
}

// ---------------------------------------------------------- subtle cases

fn mini(tasks: Value) -> Value {
    json!({
        "version": 1, "model": "t", "blocks": 4,
        "buffers": [{"id": "x", "bytes": 64}, {"id": "y", "bytes": 64}],
        "flags": [{"id": "f1"}, {"id": "f2"}, {"id": "f3"}, {"id": "g"}],
        "tasks": tasks,
    })
}

fn acc(buf: &str, b: u64, e: u64) -> Value {
    json!({"buffer": buf, "begin": b, "end": e})
}
fn wait(f: &str) -> Value {
    json!({"flag": f, "value": 1})
}
fn set(f: &str) -> Value {
    json!({"flag": f, "add": 1})
}
fn w2(f: &str, v: u32) -> Value {
    json!({"flag": f, "value": v})
}

macro_rules! tk {
    ($id:expr, $blk:expr, $ord:expr, $r:expr, $w:expr, $wt:expr, $s:expr) => {
        json!({"id": $id, "block": $blk, "order": $ord,
               "reads": $r, "writes": $w, "waits": $wt, "sets": $s})
    };
}

/// race ordered through a 3-hop cross-block chain: a --f1--> b --f2--> c
/// --f3--> d; a and d both write x[0,8) in different blocks -> must ACCEPT.
#[test]
fn three_hop_chain_orders_race_accept() {
    let v = mini(json!([
        tk!("a", 0, 0, json!([]), json!([acc("x", 0, 8)]), json!([]), json!([set("f1")])),
        tk!("b", 1, 0, json!([]), json!([acc("y", 0, 8)]), json!([wait("f1")]), json!([set("f2")])),
        tk!("c", 2, 0, json!([]), json!([acc("y", 8, 16)]), json!([wait("f2")]), json!([set("f3")])),
        tk!("d", 3, 0, json!([]), json!([acc("x", 0, 8)]), json!([wait("f3")]), json!([])),
    ]));
    assert!(accept(&v), "{:?}", kinds(&v));
}

/// same schedule minus the f2 wait on c: chain broken, d unordered vs a.
#[test]
fn broken_chain_race_rejects() {
    let v = mini(json!([
        tk!("a", 0, 0, json!([]), json!([acc("x", 0, 8)]), json!([]), json!([set("f1")])),
        tk!("b", 1, 0, json!([]), json!([acc("y", 0, 8)]), json!([wait("f1")]), json!([set("f2")])),
        tk!("c", 2, 0, json!([]), json!([acc("y", 8, 16)]), json!([]), json!([set("f3")])),
        tk!("d", 3, 0, json!([]), json!([acc("x", 0, 8)]), json!([wait("f3")]), json!([])),
    ]));
    let ks = kinds(&v);
    assert!(ks.contains(&ViolationKind::Race), "{ks:?}");
}

/// half-open ranges: [0,8) and [8,16) do NOT overlap -> ACCEPT even unordered.
#[test]
fn halfopen_boundary_no_overlap_accept() {
    let v = mini(json!([
        tk!("a", 0, 0, json!([]), json!([acc("x", 0, 8)]), json!([]), json!([])),
        tk!("b", 1, 0, json!([]), json!([acc("x", 8, 16)]), json!([]), json!([])),
    ]));
    assert!(accept(&v), "{:?}", kinds(&v));
}

/// one byte of overlap at the boundary: [0,8) and [7,16) -> REJECT.
#[test]
fn one_byte_boundary_overlap_rejects() {
    let v = mini(json!([
        tk!("a", 0, 0, json!([]), json!([acc("x", 0, 8)]), json!([]), json!([])),
        tk!("b", 1, 0, json!([]), json!([acc("x", 7, 16)]), json!([]), json!([])),
    ]));
    assert!(kinds(&v).contains(&ViolationKind::Race));
}

/// same-block overlapping writes are ordered by program order -> ACCEPT.
#[test]
fn same_block_overlap_ordered_accept() {
    let v = mini(json!([
        tk!("a", 0, 0, json!([]), json!([acc("x", 0, 8)]), json!([]), json!([])),
        tk!("b", 0, 1, json!([]), json!([acc("x", 0, 8)]), json!([]), json!([])),
    ]));
    assert!(accept(&v), "{:?}", kinds(&v));
}

/// read-read overlaps never race, even unordered cross-block -> ACCEPT.
#[test]
fn read_read_overlap_accept() {
    let v = mini(json!([
        tk!("a", 0, 0, json!([acc("x", 0, 8)]), json!([]), json!([]), json!([])),
        tk!("b", 1, 0, json!([acc("x", 0, 8)]), json!([]), json!([]), json!([])),
    ]));
    assert!(accept(&v), "{:?}", kinds(&v));
}

/// wait on a flag nobody sets -> inexact.
#[test]
fn wait_on_unset_flag_rejects() {
    let v = mini(json!([
        tk!("a", 0, 0, json!([]), json!([acc("x", 0, 8)]), json!([wait("g")]), json!([])),
    ]));
    assert!(kinds(&v).contains(&ViolationKind::InexactFlag));
}

/// task that sets and waits the same flag (total 1) -> 2-cycle -> deadlock.
#[test]
fn self_flag_cycle_rejects() {
    let v = mini(json!([
        tk!("a", 0, 0, json!([]), json!([acc("x", 0, 8)]), json!([wait("g")]), json!([set("g")])),
    ]));
    assert!(kinds(&v).contains(&ViolationKind::Deadlock), "{:?}", kinds(&v));
}

/// empty ranges conflict with nothing -> ACCEPT.
#[test]
fn empty_ranges_accept() {
    let v = mini(json!([
        tk!("a", 0, 0, json!([]), json!([acc("x", 8, 8)]), json!([]), json!([])),
        tk!("b", 1, 0, json!([]), json!([acc("x", 0, 64)]), json!([]), json!([])),
    ]));
    assert!(accept(&v), "{:?}", kinds(&v));
}

/// barrier fan-in: v = number of setters, multiple waiters -> ACCEPT.
#[test]
fn barrier_fan_in_accept() {
    let v = mini(json!([
        tk!("w0", 0, 0, json!([]), json!([acc("x", 0, 8)]), json!([]), json!([set("f1")])),
        tk!("w1", 1, 0, json!([]), json!([acc("x", 8, 16)]), json!([]), json!([set("f1")])),
        tk!("r", 2, 0, json!([acc("x", 0, 16)]), json!([]), json!([w2("f1", 2)]), json!([])),
        tk!("r2", 3, 0, json!([acc("x", 0, 16)]), json!([]), json!([w2("f1", 2)]), json!([])),
    ]));
    assert!(accept(&v), "{:?}", kinds(&v));
}

/// duplicate task id -> structural reject.
#[test]
fn duplicate_task_id_rejects() {
    let v = mini(json!([
        tk!("a", 0, 0, json!([]), json!([]), json!([]), json!([])),
        tk!("a", 1, 0, json!([]), json!([]), json!([]), json!([])),
    ]));
    assert!(kinds(&v).contains(&ViolationKind::Structural));
}

/// inconsistent wait values on one flag -> inexact reject.
#[test]
fn inconsistent_wait_values_reject() {
    let v = mini(json!([
        tk!("s", 0, 0, json!([]), json!([]), json!([]), json!([set("f1")])),
        tk!("a", 1, 0, json!([]), json!([]), json!([wait("f1")]), json!([])),
        tk!("b", 2, 0, json!([]), json!([]), json!([w2("f1", 2)]), json!([])),
    ]));
    assert!(kinds(&v).contains(&ViolationKind::InexactFlag));
}
