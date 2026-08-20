//! Supervised training loop: shuffle, minibatches, Adam, periodic export.

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
        let net = TrainNet::from_nn(nn::loaded().expect("net loaded"));
        println!(
            "loaded {} (feat={}, {}->{}->{}, policy={})",
            opts.init_net, net.feat, net.feat_count, net.l0, net.l1, net.policy_size
        );
        net
    };
    if opts.batch_size == 0 || samples.is_empty() {
        return Err("need samples and a non-zero batch size".to_string());
    }

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