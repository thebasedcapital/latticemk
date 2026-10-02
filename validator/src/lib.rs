//! LM-05: static validator for Schedule IR v1.
//!
//! Checks, per schedule file:
//!   structural  — task ids unique; (block, order) unique; referenced
//!                 buffers/flags declared; ranges within buffer bounds;
//!                 block < blocks; set add >= 1; wait value >= 1; version == 1.
//!   exactness   — for every flag F: sum of `add` over all sets of F == v for
//!                 every wait (F, v), and every wait on F uses the same v. A
//!                 wait on a flag nobody sets is inexact (total 0 != v >= 1).
//!   deadlock    — the program-order + setter→waiter graph is acyclic.
//!   race        — every pair of tasks accessing overlapping byte ranges of
//!                 the same buffer with at least one write is ordered by the
//!                 happens-before transitive closure.
//!
//! Complexity (N tasks, B blocks, F flags, A accesses, K conflicting pairs):
//!   structural   O(N + A)
//!   exactness    O(N)
//!   graph        The setter→waiter relation is expanded through one virtual
//!     node per flag (setter→F, F→waiter): O(N) edges instead of O(S·W).
//!     Kahn's algorithm gives a topo order and detects cycles in O(N + E).
//!     Reachability for race queries uses a matrix hb[node][block] = 1 + the
//!     program position of the latest task of `block` that happens-before
//!     `node` (0 = none). Rows merge by componentwise max along edges, so the
//!     build is O((N+F)·B + E·B); memory (N+F)·B·4 bytes. Each block is a
//!     chain, so a single u32 per (node, block) exactly captures cross-block
//!     happens-before. If the matrix would exceed HB_MATRIX_MAX_BYTES the
//!     checker falls back to per-pair DFS reachability, O(N+E) per query.
//!   race         Per buffer, accesses are grouped into classes with
//!     identical (begin, end, write). Classes are interval-swept (sort +
//!     + L) for C classes and L overlapping class pairs. A conflicting access
//!     pair exists iff a pair of classes overlaps, so this is exact. For a
//!     class pair (A, B) and block pair (c1 != c2), an unordered access pair
//!     (a in A@c1, b in B@c2) exists iff exists a,b with pos_a >= hb[b][c1]
//!     AND pos_b >= hb[a][c2]; checked by sorting A@c1 by pos and keeping a
//!     suffix-min of hb[a][c2], O(|A| log |A| + |B|) per block pair. Same-
//!     block pairs are skipped (chains are total orders). Individual
//!     violations are enumerated up to MAX_PAIR_VIOLATIONS, then counted.
use serde::Deserialize;
use std::collections::HashMap;

/// Cap on the (tasks+flags) x blocks reachability matrix (512 MiB).
pub const HB_MATRIX_MAX_BYTES: usize = 512 * 1024 * 1024;

pub mod gen;



#[derive(Debug, Deserialize)]
pub struct Schedule {
    pub version: u32,
    #[serde(default)]
    pub model: String,
    pub blocks: u32,
    #[serde(default)]
    pub buffers: Vec<BufferDecl>,
    #[serde(default)]
    pub flags: Vec<FlagDecl>,
    #[serde(default)]
    pub tasks: Vec<Task>,
}

#[derive(Debug, Deserialize)]
pub struct BufferDecl {
    pub id: String,
    pub bytes: u64,
}

#[derive(Debug, Deserialize)]
pub struct FlagDecl {
    pub id: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Task {
    pub id: String,
    pub block: u32,
    pub order: u32,
    #[serde(default)]
    pub reads: Vec<Access>,
    #[serde(default)]
    pub writes: Vec<Access>,
    #[serde(default)]
    pub waits: Vec<Wait>,
    #[serde(default)]
    pub sets: Vec<Set>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Access {
    pub buffer: String,
    pub begin: u64,
    pub end: u64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Wait {
    pub flag: String,
    pub value: u32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Set {
    pub flag: String,
    pub add: u32,
}

// ---------------------------------------------------------------- violations

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViolationKind {
    Structural,
    InexactFlag,
    Deadlock,
    Race,
}

impl std::fmt::Display for ViolationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ViolationKind::Structural => "structural",
            ViolationKind::InexactFlag => "inexact-flag",
            ViolationKind::Deadlock => "deadlock",
            ViolationKind::Race => "race",
        };
        f.write_str(s)
    }
}

#[derive(Debug, Clone)]
pub struct Violation {
    pub kind: ViolationKind,
    /// One-line self-contained description (task ids, buffer/flag, ranges).
    pub detail: String,
}

impl Violation {
    fn new(kind: ViolationKind, detail: String) -> Self {
        Violation { kind, detail }
    }
}

#[derive(Debug)]
pub struct Verdict {
    pub violations: Vec<Violation>,
}

impl Verdict {
    pub fn accepted(&self) -> bool {
        self.violations.is_empty()
    }
}

/// Parse + validate one schedule from JSON text.
pub fn check_str(text: &str) -> Result<Verdict, String> {
    let sched: Schedule = serde_json::from_str(text).map_err(|e| e.to_string())?;
    Ok(check(&sched))
}

// ---------------------------------------------------------------- checker

pub fn check(s: &Schedule) -> Verdict {
    let prof = std::env::var_os("LM05_PROFILE").is_some();
    let t0 = std::time::Instant::now();
    let mark = |stage: &str| {
        if prof {
            eprintln!("  {stage}: {:?}", t0.elapsed());
        }
    };
    let mut v: Vec<Violation> = Vec::new();
    if s.version != 1 {
        v.push(Violation::new(
            ViolationKind::Structural,
            format!("version {} != 1", s.version),
        ));
    }
    if s.blocks == 0 {
        v.push(Violation::new(
            ViolationKind::Structural,
            "blocks == 0".to_string(),
        ));
    }

    let mut buf_bytes: HashMap<&str, u64> = HashMap::with_capacity(s.buffers.len());
    for b in &s.buffers {
        if buf_bytes.insert(b.id.as_str(), b.bytes).is_some() {
            v.push(Violation::new(
                ViolationKind::Structural,
                format!("duplicate buffer id '{}'", b.id),
            ));
        }
    }
    let mut flag_ids: HashMap<&str, ()> = HashMap::with_capacity(s.flags.len());
    for f in &s.flags {
        if flag_ids.insert(f.id.as_str(), ()).is_some() {
            v.push(Violation::new(
                ViolationKind::Structural,
                format!("duplicate flag id '{}'", f.id),
            ));
        }
    }

    let n = s.tasks.len();
    {
        let mut task_ids: HashMap<&str, ()> = HashMap::with_capacity(n);
        for t in &s.tasks {
            if task_ids.insert(t.id.as_str(), ()).is_some() {
                v.push(Violation::new(
                    ViolationKind::Structural,
                    format!("duplicate task id '{}'", t.id),
                ));
            }
        }
    }
    {
        // duplicate (block, order): sort a (block, order, idx) list
        let mut bo: Vec<(u32, u32)> = s.tasks.iter().map(|t| (t.block, t.order)).collect();
        bo.sort_unstable();
        for w in bo.windows(2) {
            if w[0] == w[1] {
                v.push(Violation::new(
                    ViolationKind::Structural,
                    format!("duplicate order {} within block {}", w[0].1, w[0].0),
                ));
            }
        }
    }
    for t in &s.tasks {
        if t.block >= s.blocks {
            v.push(Violation::new(
                ViolationKind::Structural,
                format!("task '{}' block {} >= blocks {}", t.id, t.block, s.blocks),
            ));
        }
        for (is_write, a) in t
            .reads
            .iter()
            .map(|a| (false, a))
            .chain(t.writes.iter().map(|a| (true, a)))
        {
            match buf_bytes.get(a.buffer.as_str()) {
                None => v.push(Violation::new(
                    ViolationKind::Structural,
                    format!(
                        "task '{}' {} undeclared buffer '{}'",
                        t.id,
                        if is_write { "writes" } else { "reads" },
                        a.buffer
                    ),
                )),
                Some(&bytes) => {
                    if a.begin > a.end || a.end > bytes {
                        v.push(Violation::new(
                            ViolationKind::Structural,
                            format!(
                                "task '{}' {} [{}, {}) out of bounds of buffer '{}' ({} bytes)",
                                t.id,
                                if is_write { "writes" } else { "reads" },
                                a.begin,
                                a.end,
                                a.buffer,
                                bytes
                            ),
                        ));
                    }
                }
            }
        }
        for w in &t.waits {
            if !flag_ids.contains_key(w.flag.as_str()) {
                v.push(Violation::new(
                    ViolationKind::Structural,
                    format!("task '{}' waits on undeclared flag '{}'", t.id, w.flag),
                ));
            }
            if w.value < 1 {
                v.push(Violation::new(
                    ViolationKind::Structural,
                    format!(
                        "task '{}' waits on flag '{}' with value {} < 1",
                        t.id, w.flag, w.value
                    ),
                ));
            }
        }
        for st in &t.sets {
            if !flag_ids.contains_key(st.flag.as_str()) {
                v.push(Violation::new(
                    ViolationKind::Structural,
                    format!("task '{}' sets undeclared flag '{}'", t.id, st.flag),
                ));
            }
            if st.add < 1 {
                v.push(Violation::new(
                    ViolationKind::Structural,
                    format!(
                        "task '{}' sets flag '{}' with add {} < 1",
                        t.id, st.flag, st.add
                    ),
                ));
            }
        }
    }

    mark("structural");

    // ---- exactness -------------------------------------------------------
    let mut total: HashMap<&str, u64> = HashMap::new();
    let mut wait_val: HashMap<&str, u32> = HashMap::new();
    let mut wait_inconsistent: HashMap<&str, ()> = HashMap::new();
    for t in &s.tasks {
        for st in &t.sets {
            *total.entry(st.flag.as_str()).or_insert(0) += st.add as u64;
        }
        for w in &t.waits {
            match wait_val.get(w.flag.as_str()) {
                None => {
                    wait_val.insert(w.flag.as_str(), w.value);
                }
                Some(&prev) if prev != w.value => {
                    if wait_inconsistent.insert(w.flag.as_str(), ()).is_none() {
                        v.push(Violation::new(
                            ViolationKind::InexactFlag,
                            format!(
                                "flag '{}': inconsistent wait values {} and {}",
                                w.flag, prev, w.value
                            ),
                        ));
                    }
                }
                _ => {}
            }
        }
    }
    let flag_exact = |flag: &str| -> bool {
        flag_ids.contains_key(flag)
            && !wait_inconsistent.contains_key(flag)
            && match wait_val.get(flag) {
                Some(&val) => total.get(flag).copied().unwrap_or(0) == val as u64,
                None => true, // set-only flag: no waits, nothing to satisfy
            }
    };
    for (flag, &val) in &wait_val {
        let sum = total.get(flag).copied().unwrap_or(0);
        if sum != val as u64 {
            v.push(Violation::new(
                ViolationKind::InexactFlag,
                format!("flag '{}': sum of adds {} != wait value {}", flag, sum, val),
            ));
        }
    }

    mark("exactness");

    // ---- graph: program-order chains + one virtual node per exact flag ---
    // node ids: tasks 0..n-1; flag f -> node n + flag_idx. setter->f plus
    // f->waiter edges encode all setter->waiter hb pairs in O(S+W) edges.
    let b_eff = (s.blocks as usize).max(1);
    let mut pos_in_block: Vec<u32> = vec![0; n];
    let mut block_sorted: Vec<Vec<usize>> = vec![Vec::new(); b_eff];
    for (i, t) in s.tasks.iter().enumerate() {
        if t.block < s.blocks {
            block_sorted[t.block as usize].push(i);
        }
    }
    for chain in block_sorted.iter_mut() {
        chain.sort_by_key(|&i| s.tasks[i].order);
        for (pos, &i) in chain.iter().enumerate() {
            pos_in_block[i] = pos as u32;
        }
    }

    let mut flag_node: HashMap<&str, usize> = HashMap::new();
    for (flag, _) in &wait_val {
        if flag_exact(flag) {
            flag_node.insert(flag, n + flag_node.len());
        }
    }
    let nnodes = n + flag_node.len();

    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); nnodes];
    let mut indeg: Vec<u32> = vec![0; nnodes];
    let mut add_edge = |from: usize, to: usize| {
        succ[from].push(to);
        indeg[to] += 1;
    };
    for chain in &block_sorted {
        for w in chain.windows(2) {
            add_edge(w[0], w[1]);
        }
    }
    for (i, t) in s.tasks.iter().enumerate() {
        for st in &t.sets {
            if let Some(&fnode) = flag_node.get(st.flag.as_str()) {
                add_edge(i, fnode);
            }
        }
        for w in &t.waits {
            if let Some(&fnode) = flag_node.get(w.flag.as_str()) {
                add_edge(fnode, i);
            }
        }
    }
    // a task that sets and waits the same flag yields i->F->i: a 2-cycle,
    // caught by Kahn. A flag that is its own only setter+waiter with v=1 is
    // a deadlock exactly as the spec's exactness+acyclicity rule requires.

    mark("graph-build");

    // ---- Kahn topo + happens-before matrix --------------------------------
    // hb[node][b] = 1 + pos of the latest task in block b that happens-before
    // node. Seed each task's own cell before propagation so chains carry it.
    let use_matrix = nnodes
        .checked_mul(b_eff)
        .map(|c| c.saturating_mul(4) <= HB_MATRIX_MAX_BYTES)
        .unwrap_or(false);

    let mut topo_len = 0usize;
    let mut hb: Vec<u32> = if use_matrix {
        vec![0u32; nnodes * b_eff]
    } else {
        Vec::new()
    };
    if use_matrix {
        for (i, t) in s.tasks.iter().enumerate() {
            if t.block < s.blocks {
                hb[i * b_eff + t.block as usize] = pos_in_block[i] + 1;
            }
        }
    }
    {
        let mut stack: Vec<usize> = (0..nnodes).filter(|&u| indeg[u] == 0).collect();
        while let Some(u) = stack.pop() {
            topo_len += 1;
            if use_matrix {
                let ru = u * b_eff;
                for k in 0..succ[u].len() {
                    let w = succ[u][k];
                    let rw = w * b_eff;
                    for b in 0..b_eff {
                        if hb[ru + b] > hb[rw + b] {
                            hb[rw + b] = hb[ru + b];
                        }
                    }
                    indeg[w] -= 1;
                    if indeg[w] == 0 {
                        stack.push(w);
                    }
                }
            } else {
                for k in 0..succ[u].len() {
                    let w = succ[u][k];
                    indeg[w] -= 1;
                    if indeg[w] == 0 {
                        stack.push(w);
                    }
                }
            }
        }
    }

    mark("kahn+hb");

    // ---- deadlock ---------------------------------------------------------
    if topo_len < nnodes {
        let cyc: Vec<String> = (0..nnodes)
            .filter(|&u| indeg[u] > 0)
            .take(32)
            .map(|u| {
                if u < n {
                    format!("task '{}'", s.tasks[u].id)
                } else {
                    let name = flag_node
                        .iter()
                        .find(|&(_, &fi)| fi == u)
                        .map(|(k, _)| *k)
                        .unwrap_or("?");
                    format!("flag '{}'", name)
                }
            })
            .collect();
        v.push(Violation::new(
            ViolationKind::Deadlock,
            format!(
                "cycle: {} of {} nodes never released, e.g. [{}]",
                nnodes - topo_len,
                nnodes,
                cyc.join(", ")
            ),
        ));
        // On a cyclic graph happens-before is not a DAG; race verdicts would
        // be meaningless. The schedule is already rejected.
        return Verdict { violations: v };
    }

    mark("deadlock");

    // ---- races ------------------------------------------------------------
    // Enumerating every conflicting access pair is O(K) with K potentially
    // >> N (buffers reused across many phases give ~quadratic K in layers).
    // Instead, group accesses into identical-range classes. Every conflicting
    // pair of accesses belongs to a pair of overlapping classes, so checking
    // all class pairs is exact. For a class pair (A, B) and a block pair
    // (c1, c2), an unordered pair (a in A@c1, b in B@c2) exists iff
    //     exists a: pos_a >= hb_depth[b][c1] AND pos_b >= hb_depth[a][c2]
    // where hb_depth[t][c] = hb[t][c] (1 + latest hb-predecessor pos, 0=none).
    // For fixed c2, sort A@c1 by pos descending and sweep B@c2 by pos
    // descending; "exists a with pos_a >= hb[b] and hb[a] <= pos_b" becomes:
    // among a's with pos_a >= hb[b], the minimum hb[a][c2] must be <= pos_b.
    // Precompute suffix-min of hb[a][c2] over pos-sorted A@c1 -> O(log) per b.
    // A violation is reported once per (buffer, class pair, block pair) with
    // a witness task pair and the unordered-pair count bound.
    struct Class {
        begin: u64,
        end: u64,
        write: bool,
        /// (task_idx, pos_in_block) grouped by block
        members: HashMap<u32, Vec<(usize, u32)>>,
    }
    let mut per_buf: HashMap<&str, HashMap<(u64, u64, bool), Class>> = HashMap::new();
    for (i, t) in s.tasks.iter().enumerate() {
        if t.block >= s.blocks {
            continue; // out-of-bounds block already reported structurally
        }
        let pos = pos_in_block[i];
        for (is_write, a) in t
            .reads
            .iter()
            .map(|a| (false, a))
            .chain(t.writes.iter().map(|a| (true, a)))
        {
            if a.end > a.begin && buf_bytes.contains_key(a.buffer.as_str()) {
                per_buf
                    .entry(a.buffer.as_str())
                    .or_default()
                    .entry((a.begin, a.end, is_write))
                    .or_insert_with(|| Class {
                        begin: a.begin,
                        end: a.end,
                        write: is_write,
                        members: HashMap::new(),
                    })
                    .members
                    .entry(t.block)
                    .or_default()
                    .push((i, pos));
            }
        }
    }

    /// DFS reachability used only when the hb matrix exceeds the memory cap.
    fn reach(from: usize, to: usize, succ: &Vec<Vec<usize>>, nnodes: usize) -> bool {
        let mut seen = vec![false; nnodes];
        let mut st = vec![from];
        seen[from] = true;
        while let Some(u) = st.pop() {
            if u == to {
                return true;
            }
            for &w in &succ[u] {
                if !seen[w] {
                    seen[w] = true;
                    st.push(w);
                }
            }
        }
        false
    }

    let hb_depth = |t: usize, c: usize| -> u32 {
        if use_matrix {
            hb[t * b_eff + c]
        } else {
            // DFS fallback: is any task of block c at pos p hb-before t?
            // equivalent reach check on the per-query path; exact but slower.
            // We compute the same value: 1 + max pos of block-c tasks that
            // reach t. DFS from each block-c task is O(chain). Rare path.
            let chain = &block_sorted[c];
            let mut best = 0u32;
            for &u in chain {
                // stop early: chain positions ascend; a later member still may
                // reach t — cannot stop. Keep it simple and exact.
                if reach(u, t, &succ, nnodes) {
                    best = pos_in_block[u] + 1;
                }
            }
            best
        }
    };

    const MAX_PAIR_VIOLATIONS: usize = 4096;
    for (buf, classes) in per_buf.iter() {
        // class-granularity interval sweep: sort classes by begin, maintain a
        // min-heap on end; only overlapping class pairs are visited.
        let mut cls: Vec<&Class> = classes.values().collect();
        cls.sort_by_key(|c| c.begin);
        let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(u64, usize)>> =
            std::collections::BinaryHeap::new();
        for j in 0..cls.len() {
            let cur_b = cls[j].begin;
            while let Some(&std::cmp::Reverse((e, _))) = heap.peek() {
                if e <= cur_b {
                    heap.pop();
                } else {
                    break;
                }
            }
            // overlapping class pairs: (j,j) if write class, else (i,j)
            let mut pairs: Vec<(&Class, &Class)> = Vec::with_capacity(heap.len() + 1);
            for &std::cmp::Reverse((_, i)) in heap.iter() {
                pairs.push((cls[i], cls[j]));
            }
            if cls[j].write {
                pairs.push((cls[j], cls[j]));
            }
            heap.push(std::cmp::Reverse((cls[j].end, j)));
            for (a, b) in pairs {
                if !(a.write || b.write) {
                    continue;
                }
                for (&c1, pa) in &a.members {
                    for (&c2, pb) in &b.members {
                        if c1 == c2 {
                            continue; // program order covers it
                        }
                        let mut pa_sorted: Vec<(usize, u32)> = pa.clone();
                        pa_sorted.sort_unstable_by_key(|&(_, p)| p);
                        let m = pa_sorted.len();
                        let mut suffix_min = vec![u32::MAX; m + 1];
                        for k in (0..m).rev() {
                            let d = hb_depth(pa_sorted[k].0, c2 as usize);
                            suffix_min[k] = suffix_min[k + 1].min(d);
                        }
                        let mut emitted = 0usize;
                        let mut hidden = 0usize;
                        'outer: for &(tb, pos_b) in pb {
                            let need = hb_depth(tb, c1 as usize);
                            let idx =
                                pa_sorted.partition_point(|&(_, p)| p < need);
                            if idx >= m || suffix_min[idx] > pos_b {
                                continue;
                            }
                            for &(ta, _) in &pa_sorted[idx..] {
                                if hb_depth(ta, c2 as usize) <= pos_b {
                                    if v.len() < MAX_PAIR_VIOLATIONS {
                                        emitted += 1;
                                        v.push(Violation::new(
                                            ViolationKind::Race,
                                            format!(
                                                "tasks '{}' (block {}, order {}) \
                                                 and '{}' (block {}, order {}): \
                                                 {} [{}, {}) x {} [{}, {}) on \
                                                 buffer '{}' are unordered",
                                                s.tasks[ta].id, s.tasks[ta].block,
                                                s.tasks[ta].order,
                                                s.tasks[tb].id, s.tasks[tb].block,
                                                s.tasks[tb].order,
                                                if a.write { "W" } else { "R" },
                                                a.begin, a.end,
                                                if b.write { "W" } else { "R" },
                                                b.begin, b.end,
                                                buf
                                            ),
                                        ));
                                    } else {
                                        hidden += 1;
                                    }
                                    continue 'outer;
                                }
                            }
                        }
                        if hidden > 0 {
                            v.push(Violation::new(
                                ViolationKind::Race,
                                format!(
                                    "... and >= {} further unordered pairs on \
                                     buffer '{}' ({} [{}, {}) x {} [{}, {}), \
                                     blocks {} x {})",
                                    hidden, buf,
                                    if a.write { "W" } else { "R" }, a.begin, a.end,
                                    if b.write { "W" } else { "R" }, b.begin, b.end,
                                    c1, c2,
                                ),
                            ));
                        }
                        let _ = emitted;
                    }
                }
            }
        }
    }

    mark("races");

    Verdict { violations: v }
}
