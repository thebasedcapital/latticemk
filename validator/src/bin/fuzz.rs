//! fuzz — adversarial fuzzer for the Schedule IR validator.
//!
//! usage:
//!   fuzz [--cases N] [--seed S] [--layers L] [--blocks B]
//!   fuzz --emit FILE [--layers L] [--blocks B]     write one valid schedule
//!
//! Each case generates a valid Qwen3-like schedule, checks it (false-reject
//! counter), applies one labeled mutation, checks again (false-accept if the
//! mutated schedule is ACCEPTed). Summary: cases, false accepts, false rejects,
//! per-mutation counts. Exit 0 iff both counters are 0.

use std::process::ExitCode;
use validator::gen::{mutate, gen_value, Rng, MUTATIONS};

fn main() -> ExitCode {
    let mut cases: usize = 10_000;
    let mut seed: u64 = 0x5EED_1234;
    let mut layers: u32 = 3;
    let mut blocks: u32 = 8;
    let mut emit: Option<String> = None;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--cases" => cases = it.next().unwrap().parse().unwrap(),
            "--seed" => seed = it.next().unwrap().parse().unwrap(),
            "--layers" => layers = it.next().unwrap().parse().unwrap(),
            "--blocks" => blocks = it.next().unwrap().parse().unwrap(),
            "--emit" => emit = Some(it.next().unwrap()),
            _ => {
                eprintln!("unknown arg {a}");
                return ExitCode::from(2);
            }
        }
    }

    if let Some(path) = emit {
        let v = gen_value(layers, blocks);
        std::fs::write(&path, serde_json::to_string(&v).unwrap()).unwrap();
        let ntasks = v["tasks"].as_array().unwrap().len();
        eprintln!("wrote {path}: layers={layers} blocks={blocks} tasks={ntasks}");
        return ExitCode::SUCCESS;
    }

    let mut rng = Rng::new(seed);
    let mut false_accepts = 0usize;
    let mut false_rejects = 0usize;
    let mut per_mut: Vec<(String, usize)> = MUTATIONS.iter().map(|m| (m.to_string(), 0)).collect();
    let mut skipped = 0usize;

    for case in 0..cases {
        let layers_i = layers + (rng.next() % 2) as u32; // slight size variation
        let mut v = gen_value(layers_i, blocks);

        // valid control
        let verdict = validator::check_str(&serde_json::to_string(&v).unwrap()).unwrap();
        if !verdict.accepted() {
            false_rejects += 1;
            eprintln!("FALSE REJECT case {case}: {:?}", verdict.violations.first());
            continue;
        }

        // apply one mutation (retry another class if it doesn't apply)
        let mut applied = None;
        for _ in 0..MUTATIONS.len() {
            let name = MUTATIONS[rng.below(MUTATIONS.len())];
            if mutate(&mut v, name, &mut rng, layers_i, blocks) {
                applied = Some(name);
                break;
            }
        }
        let Some(name) = applied else {
            skipped += 1;
            continue;
        };
        let verdict = validator::check_str(&serde_json::to_string(&v).unwrap()).unwrap();
        if verdict.accepted() {
            false_accepts += 1;
            eprintln!("FALSE ACCEPT case {case} mutation {name}");
        } else {
            per_mut.iter_mut().find(|(m, _)| m == name).unwrap().1 += 1;
        }
    }

    println!("fuzz summary: seed={seed} cases={cases} blocks={blocks}");
    println!("  false accepts (mutated but ACCEPTed): {false_accepts}");
    println!("  false rejects (valid but REJECTed):   {false_rejects}");
    println!("  skipped (no mutation applied):        {skipped}");
    for (m, c) in &per_mut {
        println!("  {m}: {c} rejected");
    }
    if false_accepts == 0 && false_rejects == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
