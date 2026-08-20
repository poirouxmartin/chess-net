//! Trainable neural net: same architecture as `nn::Net`, plus Adam state and
//! a batched forward/backward. One forward computes the value and the policy
//! logits for legal moves only (cheaper than a full 4096-wide policy pass).
//!
//! Loss per sample (matching the Python trainer):
//!   l = BCE(value_logit, result) + lambda_policy * CE(policy, visit_targets)

use crate::data::policy_target;
use crate::data::Sample;
use crate::Rng;

pub struct TrainNet {
    pub feat: u32,
    pub feat_count: usize,
    pub l0: usize,
    pub l1: usize,
    pub policy_size: usize,
    pub fw: Vec<f32>, // [feat_count][l0]
    pub fb: Vec<f32>, // l0
    pub w1: Vec<f32>, // [l1][l0]
    pub b1: Vec<f32>, // l1
    pub wo: Vec<f32>, // l1
    pub bo: f32,
    pub wp: Vec<f32>, // [policy_size][l1]
    pub bp: Vec<f32>, // policy_size
    adam_m: Vec<f32>,
    adam_v: Vec<f32>,
    step: u32,
}

impl TrainNet {
    pub fn from_nn(net: &nn::Net) -> Self {
        let total = Self::param_total(net.feat_count, net.l0, net.l1, net.policy_size);
        Self {
            feat: net.feat,
            feat_count: net.feat_count,
            l0: net.l0,
            l1: net.l1,
            policy_size: net.policy_size,
            fw: net.fw.clone(),
            fb: net.fb.clone(),
            w1: net.w1.clone(),
            b1: net.b1.clone(),
            wo: net.wo.clone(),
            bo: net.bo,
            wp: net.wp.clone(),
            bp: net.bp.clone(),
            adam_m: vec![0.0; total],
            adam_v: vec![0.0; total],
            step: 0,
        }
    }

    /// Fresh init matching the Python trainer:
    /// fw ~ N(0, 0.05), w1/b1 uniform +/-1/sqrt(l0), wo uniform +/-1/sqrt(l1),
    /// wp ~ N(0, 0.02), biases zero.
    pub fn init(feat: u32, feat_count: usize, l0: usize, l1: usize, policy_size: usize, rng: &mut Rng) -> Self {
        let total = Self::param_total(feat_count, l0, l1, policy_size);
        let fw = (0..feat_count * l0).map(|_| rng.normal() * 0.05).collect();
        let b1_scale = 1.0 / (l0 as f32).sqrt();
        let wo_scale = 1.0 / (l1 as f32).sqrt();
        let w1 = (0..l1 * l0).map(|_| (rng.f32() * 2.0 - 1.0) * b1_scale).collect();
        let b1 = (0..l1).map(|_| (rng.f32() * 2.0 - 1.0) * b1_scale).collect();
        let wo = (0..l1).map(|_| (rng.f32() * 2.0 - 1.0) * wo_scale).collect();
        let wp = (0..policy_size * l1).map(|_| rng.normal() * 0.02).collect();
        Self {
            feat,
            feat_count,
            l0,
            l1,
            policy_size,
            fw,
            fb: vec![0.0; l0],
            w1,
            b1,
            wo,
            bo: 0.0,
            wp,
            bp: vec![0.0; policy_size],
            adam_m: vec![0.0; total],
            adam_v: vec![0.0; total],
            step: 0,
        }
    }

    pub fn to_nn(&self) -> nn::Net {
        nn::Net {
            feat: self.feat,
            l0: self.l0,
            l1: self.l1,
            feat_count: self.feat_count,
            fw: self.fw.clone(),
            fb: self.fb.clone(),
            w1: self.w1.clone(),
            b1: self.b1.clone(),
            wo: self.wo.clone(),
            bo: self.bo,
            policy_size: self.policy_size,
            wp: self.wp.clone(),
            bp: self.bp.clone(),
        }
    }

    fn param_total(feat_count: usize, l0: usize, l1: usize, policy_size: usize) -> usize {
        feat_count * l0 + l0 + l1 * l0 + l1 + l1 + 1 + policy_size * l1 + policy_size
    }

    /// One forward/backward pass over `batch`, then Adam updates.
    /// Returns (mean_value_loss, mean_policy_loss).
    pub fn train_batch(&mut self, batch: &[&Sample], lr: f32, lambda_policy: f32) -> (f32, f32) {
        let b = batch.len();
        let (l0, l1, ps) = (self.l0, self.l1, self.policy_size);
        let inv_b = 1.0 / b as f32;

        // Scratch.
        let mut acc = vec![0.0f32; b * l0];
        let mut s1 = vec![0.0f32; b * l1];
        let mut h1 = vec![0.0f32; b * l1];
        let mut d_h1 = vec![0.0f32; b * l1];
        let mut vl = vec![0.0f32; b];
        let mut g_fw = vec![0.0f32; self.fw.len()];
        let mut g_fb = vec![0.0f32; l0];
        let mut g_w1 = vec![0.0f32; self.w1.len()];
        let mut g_b1 = vec![0.0f32; l1];
        let mut g_wo = vec![0.0f32; l1];
        let mut g_bo = 0.0f32;
        let mut g_wp = vec![0.0f32; self.wp.len()];
        let mut g_bp = vec![0.0f32; ps];

        // ---- Forward: accumulator ----
        for bi in 0..b {
            let s = batch[bi];
            let ab = &mut acc[bi * l0..(bi + 1) * l0];
            ab.copy_from_slice(&self.fb);
            for &f in &s.active {
                let fw = &self.fw[f as usize * l0..(f as usize + 1) * l0];
                for j in 0..l0 {
                    ab[j] += fw[j];
                }
            }
        }

        // ---- Forward: hidden + heads ----
        let mut lv_total = 0.0f32;
        let mut lp_total = 0.0f32;
        for bi in 0..b {
            let a = &acc[bi * l0..(bi + 1) * l0];
            let sh = &mut s1[bi * l1..(bi + 1) * l1];
            let hh = &mut h1[bi * l1..(bi + 1) * l1];
            for k in 0..l1 {
                let w = &self.w1[k * l0..(k + 1) * l0];
                let mut sum = self.b1[k];
                for j in 0..l0 {
                    let x = a[j];
                    sum += w[j] * if x > 0.0 { x } else { 0.0 };
                }
                sh[k] = sum;
                hh[k] = if sum > 0.0 { sum } else { 0.0 };
            }

            let mut v = self.bo;
            for k in 0..l1 {
                v += self.wo[k] * hh[k];
            }
            vl[bi] = v;
            let sig = sigmoid(v);
            let r = batch[bi].result;
            lv_total += -(r * sig.ln().clamp(-50.0, 50.0)
                + (1.0 - r) * (1.0 - sig).ln().clamp(-50.0, 50.0));

            // Policy logits for distinct legal indices.
            let tgt = policy_target(&batch[bi].moves, &batch[bi].visits);
            let total_v: u32 = tgt.iter().map(|(_, v)| *v).sum();
            let mut logits = Vec::with_capacity(tgt.len());
            let mut maxl = f32::MIN;
            for (idx, _) in &tgt {
                let mut l = self.bp[*idx];
                for k in 0..l1 {
                    l += self.wp[idx * l1 + k] * hh[k];
                }
                if l > maxl {
                    maxl = l;
                }
                logits.push(l);
            }
            let mut denom = 0.0f32;
            for l in &logits {
                denom += (l - maxl).exp();
            }
            for i in 0..tgt.len() {
                let p = ((logits[i] - maxl).exp()) / denom;
                let t = tgt[i].1 as f32 / total_v.max(1) as f32;
                lp_total += -(t * p.ln().clamp(-50.0, 50.0));
            }

            // ---- Backward: value ----
            let d = (sig - r) * inv_b;
            g_bo += d;
            for k in 0..l1 {
                g_wo[k] += d * hh[k];
                d_h1[bi * l1 + k] += d * self.wo[k];
            }

            // ---- Backward: policy ----
            for i in 0..tgt.len() {
                let p = ((logits[i] - maxl).exp()) / denom;
                let t = tgt[i].1 as f32 / total_v.max(1) as f32;
                let d = (p - t) * inv_b * lambda_policy;
                let idx = tgt[i].0;
                g_bp[idx] += d;
                for k in 0..l1 {
                    g_wp[idx * l1 + k] += d * hh[k];
                    d_h1[bi * l1 + k] += d * self.wp[idx * l1 + k];
                }
            }
        }

        // ---- Backward: hidden layer ----
        for bi in 0..b {
            let a = &acc[bi * l0..(bi + 1) * l0];
            let sh = &s1[bi * l1..(bi + 1) * l1];
            let dh = &mut d_h1[bi * l1..(bi + 1) * l1];
            for k in 0..l1 {
                let ds = dh[k] * if sh[k] > 0.0 { 1.0 } else { 0.0 };
                dh[k] = ds;
                g_b1[k] += ds;
                for j in 0..l0 {
                    let x = a[j];
                    g_w1[k * l0 + j] += ds * if x > 0.0 { x } else { 0.0 };
                }
            }
        }

        // ---- Backward: accumulator layer ----
        for bi in 0..b {
            let s = batch[bi];
            let a = &acc[bi * l0..(bi + 1) * l0];
            let dh = &d_h1[bi * l1..(bi + 1) * l1];
            for j in 0..l0 {
                let mut da = 0.0f32;
                for k in 0..l1 {
                    da += dh[k] * self.w1[k * l0 + j];
                }
                da *= if a[j] > 0.0 { 1.0 } else { 0.0 };
                g_fb[j] += da;
                for &f in &s.active {
                    g_fw[f as usize * l0 + j] += da;
                }
            }
        }

        self.step += 1;
        let t = self.step;
        let (n_fw, n_fb, n_w1, n_b1, n_wo, n_wp, n_bp) =
            (self.fw.len(), l0, self.w1.len(), l1, l1, self.wp.len(), ps);
        let mut off = 0;
        adam_update(&mut self.fw, &g_fw, &mut self.adam_m[off..off + n_fw], &mut self.adam_v[off..off + n_fw], lr, t);
        off += n_fw;
        adam_update(&mut self.fb, &g_fb, &mut self.adam_m[off..off + n_fb], &mut self.adam_v[off..off + n_fb], lr, t);
        off += n_fb;
        adam_update(&mut self.w1, &g_w1, &mut self.adam_m[off..off + n_w1], &mut self.adam_v[off..off + n_w1], lr, t);
        off += n_w1;
        adam_update(&mut self.b1, &g_b1, &mut self.adam_m[off..off + n_b1], &mut self.adam_v[off..off + n_b1], lr, t);
        off += n_b1;
        adam_update(&mut self.wo, &g_wo, &mut self.adam_m[off..off + n_wo], &mut self.adam_v[off..off + n_wo], lr, t);
        off += n_wo;
        {
            let bo = std::slice::from_mut(&mut self.bo);
            let g = [g_bo];
            adam_update(bo, &g, &mut self.adam_m[off..off + 1], &mut self.adam_v[off..off + 1], lr, t);
        }
        off += 1;
        adam_update(&mut self.wp, &g_wp, &mut self.adam_m[off..off + n_wp], &mut self.adam_v[off..off + n_wp], lr, t);
        off += n_wp;
        adam_update(&mut self.bp, &g_bp, &mut self.adam_m[off..off + n_bp], &mut self.adam_v[off..off + n_bp], lr, t);

        (lv_total * inv_b, lp_total * inv_b)
    }

    /// Value output (white POV, unscaled logit) for one sample.
    pub fn predict_value(&self, active: &[u32]) -> f32 {
        let (l0, l1) = (self.l0, self.l1);
        let mut a = vec![0.0f32; l0];
        a.copy_from_slice(&self.fb);
        for &f in active {
            let fw = &self.fw[f as usize * l0..(f as usize + 1) * l0];
            for j in 0..l0 {
                a[j] += fw[j];
            }
        }
        let mut h = vec![0.0f32; l1];
        for k in 0..l1 {
            let w = &self.w1[k * l0..(k + 1) * l0];
            let mut sum = self.b1[k];
            for j in 0..l0 {
                let x = a[j];
                sum += w[j] * if x > 0.0 { x } else { 0.0 };
            }
            h[k] = if sum > 0.0 { sum } else { 0.0 };
        }
        let mut v = self.bo;
        for k in 0..l1 {
            v += self.wo[k] * h[k];
        }
        v
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn adam_update(p: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32], lr: f32, t: u32) {
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let b1t = b1.powi(t as i32);
    let b2t = b2.powi(t as i32);
    let m_hat = 1.0 / (1.0 - b1t);
    let v_hat = 1.0 / (1.0 - b2t);
    for i in 0..p.len() {
        m[i] = b1 * m[i] + (1.0 - b1) * g[i];
        v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
        p[i] -= lr * (m[i] * m_hat) / ((v[i] * v_hat).sqrt() + eps);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Sample;
    use crate::Rng;
    use engine::move_::Move;

    fn sample(active: Vec<u32>, moves: Vec<Move>, visits: Vec<u32>, result: f32) -> Sample {
        Sample { active, moves, visits, result }
    }

    #[test]
    fn value_loss_trains_toward_result() {
        let mut rng = Rng::new(1);
        let mut net = TrainNet::init(2, 768, 32, 16, 4096, &mut rng);
        // All samples: white wins, same features, same few legal moves.
        let s = sample(vec![0, 5, 100], vec![Move(0), Move(65)], vec![8, 2], 1.0);
        let samples: Vec<&Sample> = vec![&s; 64];
        let lr = 0.01;
        let l0 = net.predict_value(&s.active);
        let mut last = 0.0f32;
        for _ in 0..200 {
            let (lv, lp) = net.train_batch(&samples, lr, 1.0);
            last = lv;
            assert!(lv.is_finite() && lp.is_finite(), "loss not finite: {lv} {lp}");
        }
        let l1 = net.predict_value(&s.active);
        // BCE dropped, value logit moved toward +inf (white win).
        assert!(last < 0.4, "loss still high: {last}");
        assert!(l1 > l0, "value did not rise: {l0} -> {l1}");
    }

    #[test]
    fn roundtrip_to_nn() {
        let mut rng = Rng::new(2);
        let net = TrainNet::init(2, 768, 32, 16, 4096, &mut rng);
        let nn = net.to_nn();
        assert_eq!(nn.feat_count, 768);
        assert_eq!(nn.policy_size, 4096);
        assert_eq!(nn.wp.len(), 4096 * 16);
    }
}