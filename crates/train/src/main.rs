//! `chess-train` CLI: self-play data generation and neural training in Rust.
//!
//! Usage:
//!   chess-train selfplay --net <net.csnn> --games N [--iters N] [--threads N]
//!       [--temp-drop N] [--max-plies N] --out data.bin
//!   chess-train train --data data.bin [--init net.csnn] --out net.csnn
//!       [--epochs N] [--batch-size N] [--lr F] [--lambda-policy F]

use train::gen::{generate, GenOptions};
use train::train::{train, TrainOptions};

fn main() {
    engine::init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage();
        return;
    }
    let cmd = args[1].as_str();
    let opts = parse_opts(&args[2..]);
    let res = match cmd {
        "selfplay" => {
            let net = get(&opts, "--net");
            let games = get_u32(&opts, "--games", 1000);
            let out = get(&opts, "--out");
            match (net, out) {
                (Some(n), Some(o)) => generate(&GenOptions {
                    net_path: n,
                    games,
                    iters: get_u32(&opts, "--iters", 200),
                    threads: get_usize(&opts, "--threads", 0),
                    temp_drop: get_u32(&opts, "--temp-drop", 12),
                    max_plies: get_u32(&opts, "--max-plies", 400),
                    out_path: o,
                }),
                _ => {
                    usage();
                    return;
                }
            }
        }
        "train" => {
            let data = get(&opts, "--data");
            let out = get(&opts, "--out");
            match (data, out) {
                (Some(d), Some(o)) => {
                    let samples = match train::data::read_samples(&d) {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!("error reading {}: {}", d, e);
                            return;
                        }
                    };
                    println!("loaded {} positions from {}", samples.len(), d);
                    let t = TrainOptions {
                        init_net: get(&opts, "--init").unwrap_or_default(),
                        out_net: o,
                        epochs: get_u32(&opts, "--epochs", 2),
                        batch_size: get_usize(&opts, "--batch-size", 1024),
                        lr: get_f32(&opts, "--lr", 1e-3),
                        lambda_policy: get_f32(&opts, "--lambda-policy", 1.0),
                        seed: 0x1234_5678,
                    };
                    train(&samples, &t).map(|_| ())
                }
                _ => {
                    usage();
                    return;
                }
            }
        }
        _ => {
            usage();
            std::process::exit(0);
        }
    };
    if let Err(e) = res {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
}

fn usage() {
    println!(
        "chess-train
  selfplay --net <net.csnn> --games N [--iters N] [--threads N] [--temp-drop N] [--max-plies N] --out data.bin
  train --data data.bin [--init net.csnn] --out net.csnn [--epochs N] [--batch-size N] [--lr F] [--lambda-policy F]"
    );
}

fn parse_opts(args: &[String]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut it = args.iter();
    while let Some(k) = it.next() {
        if let Some(v) = it.next() {
            out.push((k.clone(), v.clone()));
        }
    }
    out
}

fn get(opts: &[(String, String)], key: &str) -> Option<String> {
    opts.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone())
}

fn get_u32(opts: &[(String, String)], key: &str, def: u32) -> u32 {
    get(opts, key).and_then(|v| v.parse().ok()).unwrap_or(def)
}

fn get_usize(opts: &[(String, String)], key: &str, def: usize) -> usize {
    get(opts, key).and_then(|v| v.parse().ok()).unwrap_or(def)
}

fn get_f32(opts: &[(String, String)], key: &str, def: f32) -> f32 {
    get(opts, key).and_then(|v| v.parse().ok()).unwrap_or(def)
}