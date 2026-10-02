//! Schedule generator + adversarial mutations for LM-05 fuzzing.
//!
//! Generated schedules mimic a Qwen3-0.6B decode step: `layers` x phases
//! [norm1, qkv, attn, o, norm2, gate_up, down] then one lm_head phase. Every
//! phase contributes exactly one task per block; phase P's task in every block
//! waits on flag `f{P-1}` with value = blocks, and every phase-(P-1) task sets
//! `f{P-1}` += 1 (total = blocks, so exactness holds by construction). Each
//! task writes a private slice of its phase's output buffer and reads the
//! whole input buffer, so all cross-phase conflicts on shared buffers are
//! ordered through the flag chain; within a phase, write slices are disjoint.
//!
//! Layout per layer: norm1 reads x writes xn; qkv reads xn writes qkv; attn
//! reads qkv writes attn; o reads attn writes obuf; norm2 reads obuf writes
//! xn2; gate_up reads xn2 writes mlp; down reads mlp writes x (residual);
//! lm_head reads x writes logits.

use serde_json::{json, Map, Value};

/// splitmix64/xorshift64* — deterministic, no external deps.
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

pub const PHASES: [&str; 7] = ["norm1", "qkv", "attn", "o", "norm2", "gate_up", "down"];

/// buffer name -> bytes per block-slice (total bytes = blocks * per_block)
const BUF_SPC: [(&str, u64); 8] = [
    ("x", 32),
    ("xn", 32),
    ("qkv", 64),
    ("attn", 32),
    ("obuf", 32),
    ("xn2", 32),
    ("mlp", 96),
    ("logits", 128),
];

/// phase -> (reads buffer, writes buffer)
fn phase_io(phase: &str) -> (&'static str, &'static str) {
    match phase {
        "norm1" => ("x", "xn"),
        "qkv" => ("xn", "qkv"),
        "attn" => ("qkv", "attn"),
        "o" => ("attn", "obuf"),
        "norm2" => ("obuf", "xn2"),
        "gate_up" => ("xn2", "mlp"),
        "down" => ("mlp", "x"),
        "lm_head" => ("x", "logits"),
        _ => unreachable!(),
    }
}

/// total number of phases for `layers` layers (incl. lm_head)
pub fn n_phases(layers: u32) -> u32 {
    layers * PHASES.len() as u32 + 1
}

/// phase index -> (layer, phase name); last phase index is lm_head
pub fn phase_name(p: u32, layers: u32) -> String {
    if p == layers * PHASES.len() as u32 {
        "lm_head".to_string()
    } else {
        let l = p / PHASES.len() as u32;
        format!("L{}.{}", l, PHASES[(p % PHASES.len() as u32) as usize])
    }
}

fn acc(buf: &str, begin: u64, end: u64) -> Value {
    json!({"buffer": buf, "begin": begin, "end": end})
}

/// Generate a valid schedule. blocks >= 1, layers >= 1.
pub fn gen_value(layers: u32, blocks: u32) -> Value {
    let np = n_phases(layers);
    let mut buffers = Vec::new();
    for (name, spc) in BUF_SPC {
        buffers.push(json!({"id": name, "bytes": spc * blocks as u64}));
    }
    let spc = |name: &str| -> u64 {
        BUF_SPC.iter().find(|(n, _)| *n == name).unwrap().1
    };

    let mut flags = Vec::new();
    let mut tasks = Vec::new();
    // flag f{p} is set by phase p and waited by phase p+1 (p = 0..np-2)
    for p in 0..np - 1 {
        flags.push(json!({"id": format!("f{}", p)}));
    }
    for p in 0..np {
        let pname = phase_name(p, layers);
        let (rbuf, wbuf) = phase_io(if p == np - 1 { "lm_head" } else { PHASES[(p % 7) as usize] });
        let (rspc, wspc) = (spc(rbuf), spc(wbuf));
        for b in 0..blocks {
            let mut t = Map::new();
            t.insert("id".into(), json!(format!("{}.b{}", pname, b)));
            t.insert("block".into(), json!(b));
            t.insert("order".into(), json!(p * blocks + b));
            t.insert(
                "reads".into(),
                json!([acc(rbuf, 0, rspc * blocks as u64)]),
            );
            t.insert(
                "writes".into(),
                json!([acc(wbuf, wspc * b as u64, wspc * (b as u64 + 1))]),
            );
            if p > 0 {
                t.insert("waits".into(), json!([{"flag": format!("f{}", p - 1), "value": blocks}]));
            }
            if p < np - 1 {
                t.insert("sets".into(), json!([{"flag": format!("f{}", p), "add": 1}]));
            }
            tasks.push(Value::Object(t));
        }
    }
    json!({
        "version": 1,
        "model": "qwen3-0.6b",
        "blocks": blocks,
        "buffers": buffers,
        "flags": flags,
        "tasks": tasks,
    })
}

// ------------------------------------------------------------ mutations
// Each takes a valid generated schedule and corrupts it so that check()
// MUST reject. All operate on serde_json::Value.

pub const MUTATIONS: [&str; 11] = [
    "delete_set",
    "delete_wait",
    "lower_wait",
    "raise_add",
    "cyclic_wait_cross_block",
    "cyclic_wait_same_block",
    "overlap_writes",
    "raw_race",
    "oob_range",
    "dup_order",
    "undeclared_ref",
];

/// index of the unique task of phase p in block b inside tasks[] (tasks are
/// emitted phase-major: task p*blocks + b)
fn tidx(p: usize, b: usize, blocks: usize) -> usize {
    p * blocks + b
}

fn tasks_mut(s: &mut Value) -> &mut Vec<Value> {
    s["tasks"].as_array_mut().unwrap()
}
/// Apply mutation `name`. Returns true if applied; caller may retry with a
/// different name if false (kept simple: all apply when layers >= 2, blocks >= 2).
pub fn mutate(s: &mut Value, name: &str, rng: &mut Rng, layers: u32, blocks: u32) -> bool {
    let np = n_phases(layers) as usize;
    let b = blocks as usize;
    match name {
        // drop one set edge: flag total no longer reaches the wait value
        "delete_set" => {
            // any task of phases 0..np-2 sets a flag
            let p = rng.below(np - 1);
            let i = tidx(p, rng.below(b), b);
            let sets = tasks_mut(s)[i]["sets"].as_array_mut().unwrap();
            sets.remove(rng.below(sets.len()));
            true
        }
        // remove the only synchronising edge into one block for a flag:
        // the phase-p task in block b loses its wait on f{p-1}, so it races
        // every cross-block setter of f{p-1} on the shared input buffer.
        "delete_wait" => {
            let p = 1 + rng.below(np - 1); // phases 1..np-1 have waits
            let i = tidx(p, rng.below(b), b);
            tasks_mut(s)[i].as_object_mut().unwrap().insert("waits".into(), json!([]));
            true
        }
        // wait value lower than the flag total -> inexact
        "lower_wait" => {
            if blocks < 2 {
                return false; // need value >= 2 to lower
            }
            let p = 1 + rng.below(np - 1);
            let i = tidx(p, rng.below(b), b);
            let w = &mut tasks_mut(s)[i]["waits"][0];
            let v = w["value"].as_u64().unwrap();
            w.as_object_mut()
                .unwrap()
                .insert("value".into(), json!(v - 1));
            true
        }
        // add larger than needed: flag total exceeds the wait value -> inexact
        "raise_add" => {
            let p = rng.below(np - 1);
            let i = tidx(p, rng.below(b), b);
            let st = &mut tasks_mut(s)[i]["sets"][0];
            let a = st["add"].as_u64().unwrap();
            st.as_object_mut().unwrap().insert("add".into(), json!(a + 1));
            true
        }
        // S (phase p-1) waits on G (set by phase p, value = blocks): every
        // phase-p task hb-before S via f{p-1}, and hb-after S through G.
        "cyclic_wait_cross_block" => {
            if np < 4 {
                return false;
            }
            let p = 1 + rng.below(np - 2); // 1..np-2: needs p-1 >= 0 and flag f{p} exists (p <= np-2)
            let i = tidx(p - 1, rng.below(b), b);
            let t = &mut tasks_mut(s)[i];
            let arr = if t["waits"].is_array() {
                t["waits"].as_array_mut().unwrap()
            } else {
                t.as_object_mut().unwrap().insert("waits".into(), json!([]));
                t["waits"].as_array_mut().unwrap()
            };
            arr.push(json!({"flag": format!("f{}", p), "value": blocks}));
            true
        }
        // T (phase p) waits on f{p+1} which is set by the phase-(p+1) task in
        // the same block — later in program order: 2-cycle inside the block.
        "cyclic_wait_same_block" => {
            if np < 4 {
                return false;
            }
            let p = rng.below(np - 2); // 0..np-3 so that p+1 <= np-2 sets a flag
            let i = tidx(p, rng.below(b), b);
            let t = &mut tasks_mut(s)[i];
            let arr = if t["waits"].is_array() {
                t["waits"].as_array_mut().unwrap()
            } else {
                t.as_object_mut().unwrap().insert("waits".into(), json!([]));
                t["waits"].as_array_mut().unwrap()
            };
            arr.push(json!({"flag": format!("f{}", p + 1), "value": blocks}));
            true
        }
        // two fresh tasks in different blocks write overlapping ranges of x,
        // unordered (appended at both chains' ends: hb-after everything,
        // unordered between themselves)
        "overlap_writes" => {
            if b < 2 {
                return false;
            }
            let (b1, b2) = (0usize, 1usize);
            for (k, blk) in [b1, b2].iter().enumerate() {
                tasks_mut(s).push(json!({
                    "id": format!("mut.ww{}", k),
                    "block": blk,
                    "order": 4_000_000_000u32 + k as u32,
                    "reads": [],
                    "writes": [acc("x", 0, 64)],
                }));
            }
            true
        }
        // fresh reader in one block vs fresh writer in another, same bytes
        "raw_race" => {
            if b < 2 {
                return false;
            }
            tasks_mut(s).push(json!({
                "id": "mut.rd", "block": 0, "order": 4_000_000_100u32,
                "reads": [acc("x", 8, 24)], "writes": [],
            }));
            tasks_mut(s).push(json!({
                "id": "mut.wr", "block": 1, "order": 4_000_000_100u32,
                "reads": [], "writes": [acc("x", 16, 40)],
            }));
            true
        }
        // extend a write past the buffer end
        "oob_range" => {
            let i = rng.below(tasks_mut(s).len());
            let bufname = tasks_mut(s)[i]["writes"][0]["buffer"]
                .as_str()
                .unwrap()
                .to_string();
            let bytes = s["buffers"]
                .as_array()
                .unwrap()
                .iter()
                .find(|bb| bb["id"] == bufname)
                .unwrap()["bytes"]
                .as_u64()
                .unwrap();
            let w = &mut tasks_mut(s)[i]["writes"][0];
            w.as_object_mut()
                .unwrap()
                .insert("end".into(), json!(bytes + 64));
            true
        }
        // duplicate the order of the next task in the same block
        "dup_order" => {
            // pick a phase p < np-1 and block b: task (p,b) and (p+1,b) share
            // block b; set (p+1,b).order = (p,b).order
            let p = rng.below(np - 1);
            let bi = rng.below(b);
            let o = tasks_mut(s)[tidx(p, bi, b)]["order"].as_u64().unwrap();
            let j = tidx(p + 1, bi, b);
            tasks_mut(s)[j]
                .as_object_mut()
                .unwrap()
                .insert("order".into(), json!(o));
            true
        }
        // reference a buffer or flag that was never declared
        "undeclared_ref" => {
            let i = rng.below(tasks_mut(s).len());
            let t = &mut tasks_mut(s)[i];
            if rng.next() % 2 == 0 {
                t["reads"].as_array_mut().unwrap().push(acc("__nope_buf", 0, 8));
            } else {
                let arr = if t["waits"].is_array() {
                    t["waits"].as_array_mut().unwrap()
                } else {
                    t.as_object_mut().unwrap().insert("waits".into(), json!([]));
                    t["waits"].as_array_mut().unwrap()
                };
                arr.push(json!({"flag": "__nope_flag", "value": 1}));
            }
            true
        }
        _ => false,
    }
}
