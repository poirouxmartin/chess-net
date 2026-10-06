//! Supervised training loop: shuffle, minibatches, Adam, periodic export.
//!
//! LEGACY track: unflipped from*64+to 4096 policy mapping (see data.rs).
//! The active pipeline is the Python AlphaZero loop (python/chessnet),
//! which uses STM-oriented 4288 indices. Nets trained here are NOT
//! compatible with the current engine policy mapping (their policy head
//! is ignored: engine requires policy_size == 4288).

use crate::data::Sample;
use crate::net::TrainNet;
use crate::Rng;

pub struct TrainOptions {
    /// Start from this CSNN net, or fresh-init if empty.
    pub init_net: String,
    pub out_net: String,
    pub epochs: u32,
    pub batch_size: usize,
    pub lr: f32,
    pub lambda_policy: f32,
    pub seed: u64,
}

pub fn train(samples: &[Sample], opts: &TrainOptions) -> Result<TrainNet, String> {
    let mut net = if opts.init_net.is_empty() {
        TrainNet::init(2, 768, 256, 32, 4096, &mut Rng::new(opts.seed))
    } else {
        nn::load(&opts.init_net)?;
        let loaded = nn::loaded().expect("net loaded");
        if loaded.policy_size != 4096 {
            return Err(format!(
                "init net has policy_size {} (want 4096): this legacy trainer uses the \
                 unflipped from*64+to mapping; v1 (value-only) and 4288 nets are rejected",
                loaded.policy_size
            ));
        }
        let net = TrainNet::from_nn(&loaded);
        println!(
            "loaded {} (feat={}, {}->{}->{}, policy={})",
            opts.init_net, net.feat, net.feat_count, net.l0, net.l1, net.policy_size
        );
        net
    };
    if opts.batch_size == 0 || samples.is_empty() {
        return Err("need samples and a non-zero batch size".to_string());
    }
    // Trust boundary: the .bin format carries no feat_count, so validate
    // indices against this net (OOB would panic in forward/backward) and
    // drop non-finite/out-of-range results (BCE poison).
    let feat_count = net.feat_count;
    let kept: Vec<Sample> = samples
        .iter()
        .filter(|s| {
            s.result.is_finite()
                && (0.0..=1.0).contains(&s.result)
                && s.active.iter().all(|&f| (f as usize) < feat_count)
        })
        .cloned()
        .collect();
    if kept.len() < samples.len() {
        println!(
            "filtered {}/{} corrupt samples (bad features/result)",
            samples.len() - kept.len(),
            samples.len()
        );
    }
    if kept.is_empty() {
        return Err("no valid samples after filtering".to_string());
    }
    let samples = &kept[..];

    let mut idx: Vec<usize> = (0..samples.len()).collect();
    let mut rng = Rng::new(opts.seed ^ 0xabcdef);
    let batches_per_epoch = samples.len() / opts.batch_size;

    for ep in 0..opts.epochs {
        // Fisher-Yates shuffle.
        for i in (1..idx.len()).rev() {
            let j = rng.range(i as u32 + 1) as usize;
            idx.swap(i, j);
        }
        let mut lv_sum = 0.0f32;
        let mut lp_sum = 0.0f32;
        for bi in 0..batches_per_epoch {
            let batch: Vec<&Sample> = idx[bi * opts.batch_size..(bi + 1) * opts.batch_size]
                .iter()
                .map(|&i| &samples[i])
                .collect();
            let (lv, lp) = net.train_batch(&batch, opts.lr, opts.lambda_policy);
            lv_sum += lv;
            lp_sum += lp;
        }
        let n = batches_per_epoch.max(1) as f32;
        let mean_lv = lv_sum / n;
        let mean_lp = lp_sum / n;
        println!(
            "epoch {}: value_loss={:.5} policy_loss={:.5} ({} batches)",
            ep + 1,
            mean_lv,
            mean_lp,
            batches_per_epoch
        );

        // Export after each epoch so an interrupted run still produces a net.
        let nn = net.to_nn();
        nn::write_csnn(&nn, &opts.out_net)?;
    }
    Ok(net)
}