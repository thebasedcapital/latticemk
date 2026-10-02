//! schedcheck FILE.json... — validate Schedule IR v1 files.
//! Prints `ACCEPT <file>` or `REJECT <file>` plus every violation.
//! Exit code 0 iff every file parsed and validated clean.

use std::process::ExitCode;
use validator::check_str;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: schedcheck FILE.json...");
        return ExitCode::from(2);
    }
    let mut all_ok = true;
    for path in &args {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                all_ok = false;
                println!("REJECT {path}");
                println!("  io-error: {e}");
                continue;
            }
        };
        match check_str(&text) {
            Err(e) => {
                all_ok = false;
                println!("REJECT {path}");
                println!("  parse-error: {e}");
            }
            Ok(verdict) if verdict.accepted() => {
                println!("ACCEPT {path}");
            }
            Ok(verdict) => {
                all_ok = false;
                println!("REJECT {path}");
                for v in &verdict.violations {
                    println!("  {}: {}", v.kind, v.detail);
                }
            }
        }
    }
    if all_ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
