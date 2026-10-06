//! GPU-accelerated ONNX inference for the ResNet+SE network.
//!
//! Uses ONNX Runtime (via the `ort` crate) with CUDA execution provider
//! for high-throughput batched inference. The model takes 112×8×8 planes
//! and outputs policy logits (4288) + tanh value scalar (1, STM frame).

use crate::policy_index;

use ort::ep;
use ort::session::Session;
use ort::value::TensorRef;
use std::sync::Mutex;

use engine::movegen::MoveList;
use engine::position::Position;

use crate::features_lc0::{encode_position, NUM_PLANES};

pub const POLICY_SIZE: usize = crate::POLICY_SIZE; // 4288: from*64+to + underpromo slices

/// Evaluation result for a single position.
#[derive(Clone, Debug)]
pub struct EvalResult {
    /// Centipawn score from side-to-move perspective.
    pub cp: i32,
    /// Raw policy logits, one per move in `legal` (same order).
    pub logits: Vec<f32>,
}

/// GPU-backed ONNX inference session.
pub struct OnnxEvaluator {
    session: Session,
    max_batch: usize,
}

impl OnnxEvaluator {
    /// Maximum batch size this session was created with.
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }
}

static SESSION: Mutex<Option<OnnxEvaluator>> = Mutex::new(None);

/// Lock the global session (poison-tolerant: a previous panic must not
/// wedge all later analysis).
fn lock_session() -> std::sync::MutexGuard<'static, Option<OnnxEvaluator>> {
    SESSION.lock().unwrap_or_else(|e| e.into_inner())
}

impl OnnxEvaluator {
    /// Initialize the global GPU evaluator. Call once at startup.
    pub fn init(model_path: &str, max_batch: usize) -> Result<(), String> {
        // CUDA and ONNX Runtime DLLs come from the Python wheels (nvidia-*,
        // onnxruntime-gpu). `CHESSNET_SITE_PACKAGES` points at that
        // site-packages; default is the per-user Python 3.12 install.
        let site = std::env::var("CHESSNET_SITE_PACKAGES").unwrap_or_else(|_| {
            let local = std::env::var("LOCALAPPDATA").unwrap_or_default();
            format!(r"{local}\Programs\Python\Python312\Lib\site-packages")
        });

        // Add CUDA DLL paths to PATH so ONNX Runtime can find cuDNN/cuBLAS
        let python_base = format!(r"{site}\nvidia");
        let cuda_dirs = [
            format!("{python_base}\\cudnn\\bin"),
            format!("{python_base}\\cublas\\bin"),
            format!("{python_base}\\cuda_nvrtc\\bin"),
        ];
        let extra = cuda_dirs.join(";");
        let current = std::env::var("PATH").unwrap_or_default();
        if !current.contains(&extra) {
            std::env::set_var("PATH", format!("{extra};{current}"));
        }

        // Load ONNX Runtime from Python's onnxruntime installation
        let ort_dll = std::env::var("ORT_DYLIB_PATH")
            .unwrap_or_else(|_| format!(r"{site}\onnxruntime\capi\onnxruntime.dll"));
        ort::init_from(&ort_dll)
            .map_err(|e| format!("init ORT from {}: {e}", ort_dll))?;

        let cuda = ep::CUDA::default()
            .with_device_id(0)
            .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Heuristic)
            .build();

        let mut builder = Session::builder()
            .map_err(|e| format!("session builder: {e}"))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .unwrap_or_else(|e| e.recover())
            .with_intra_threads(1)
            .unwrap_or_else(|e| e.recover())
            .with_execution_providers([cuda])
            .unwrap_or_else(|e| e.recover());

        let mut session = builder
            .commit_from_file(model_path)
            .map_err(|e| format!("commit: {e}"))?;

        // Validate the model I/O before accepting it: a wrong-architecture
        // .onnx must fail here with a message, not later as a worker panic
        // (panic=abort kills the whole app mid-analysis).
        {
            let zeros = vec![0.0f32; NUM_PLANES * 8 * 8];
            let input = TensorRef::from_array_view(([1, NUM_PLANES, 8, 8], zeros.as_slice()))
                .map_err(|e| format!("validation input: {e}"))?;
            let outputs = session
                .run(ort::inputs![input])
                .map_err(|e| format!("validation inference: {e}"))?;
            if outputs.len() < 2 {
                return Err("incompatible model: want 2 outputs (policy, value)".into());
            }
            let (_, policy): (_, &[f32]) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("policy output: {e}"))?;
            let (_, value): (_, &[f32]) = outputs[1]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("value output: {e}"))?;
            if policy.len() != POLICY_SIZE || value.len() != 1 {
                return Err(format!(
                    "incompatible model outputs: policy={} (want 4288), value={} (want 1)",
                    policy.len(),
                    value.len()
                ));
            }
        }

        log::info!(
            "ONNX session loaded: {} inputs, {} outputs, max_batch={}",
            session.inputs().len(),
            session.outputs().len(),
            max_batch
        );

        let eval = OnnxEvaluator { session, max_batch };
        // Replaceable (not OnceCell): picking another net must switch.
        *lock_session() = Some(eval);
        Ok(())
    }

    /// Whether a model is currently loaded.
    pub fn is_ready() -> bool {
        lock_session().is_some()
    }

    /// Drop the current session so a different backend (e.g. CSNN) takes
    /// over cleanly. Without this, switching EvalFile away from ONNX would
    /// leave the old session answering batched MCTS calls.
    pub fn unload() {
        *lock_session() = None;
    }

    /// Run `f` on the global evaluator, if one is loaded.
    fn with_eval<R>(f: impl FnOnce(&mut OnnxEvaluator) -> R) -> Option<R> {
        lock_session().as_mut().map(f)
    }

    /// Batch-evaluate multiple positions on GPU.
    pub fn batch_evaluate(
        &mut self,
        positions: &[(Position, &MoveList)],
    ) -> Vec<EvalResult> {
        let batch_size = positions.len();
        assert!(
            batch_size <= self.max_batch,
            "batch {} exceeds max {}",
            batch_size,
            self.max_batch
        );

        // Build input tensor: [B, 112, 8, 8]
        let mut input_data = vec![0.0f32; batch_size * NUM_PLANES * 8 * 8];
        for (i, (pos, _)) in positions.iter().enumerate() {
            let planes = encode_position(pos);
            let offset = i * NUM_PLANES * 8 * 8;
            input_data[offset..offset + NUM_PLANES * 8 * 8].copy_from_slice(&planes);
        }

        let input_tensor = TensorRef::from_array_view((
            [batch_size, NUM_PLANES, 8, 8],
            input_data.as_slice(),
        ))
        .expect("create input tensor");

        // Run inference
        let outputs = self
            .session
            .run(ort::inputs![input_tensor])
            .expect("inference");

        // Extract policy logits from output 0: [B, 4288]
        let (_, policy_data): (_, &[f32]) = outputs[0]
            .try_extract_tensor::<f32>()
            .expect("policy data");

        // Extract value scalar from output 1: [B, 1] tanh in [-1, 1],
        // side-to-move frame (AlphaZero convention).
        let (_, value_data): (_, &[f32]) = outputs[1]
            .try_extract_tensor::<f32>()
            .expect("value data");

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            // Policy logits
            let base_p = i * POLICY_SIZE;
            let mut logits = Vec::with_capacity(POLICY_SIZE);
            logits.extend_from_slice(&policy_data[base_p..base_p + POLICY_SIZE]);

            // tanh value -> centipawns, side-to-move frame (no flip:
            // the net already speaks STM).
            let v = value_data[i].clamp(-1.0, 1.0);
            let cp = (v * 400.0).round() as i32;

            results.push(EvalResult { cp, logits });
        }

        results
    }

    /// Single-position evaluation (convenience wrapper).
    pub fn evaluate(&mut self, pos: &Position, legal: &MoveList) -> EvalResult {
        let results = self.batch_evaluate(&[(pos.clone(), legal)]);
        results.into_iter().next().expect("at least one result")
    }
}

/// Evaluate from the MCTS ValuePolicyFn signature.
/// This calls the global ONNX evaluator.
pub fn evaluate_onnx(pos: &Position, legal: &MoveList) -> (i32, Vec<f32>) {
    OnnxEvaluator::with_eval(|eval| {
        let result = eval.evaluate(pos, legal);
        (result.cp, result.logits)
    })
    .expect("ONNX evaluator not initialized")
}

/// `BatchEvalFn`-compatible adapter for MCTS: evaluates a batch of positions
/// in as few GPU calls as needed and returns per-position `(cp, logits)`
/// with the policy logits filtered to the given legal moves (side-to-move
/// oriented mapping, so MCTS sees one logit per legal move, in order).
pub fn batch_evaluate_onnx(positions: &[Position], legal_lists: &[&MoveList]) -> Vec<(i32, Vec<f32>)> {
    assert_eq!(positions.len(), legal_lists.len());
    let mut out = Vec::with_capacity(positions.len());
    if positions.is_empty() {
        return out;
    }
    let mut eval = lock_session();
    let eval = eval.as_mut().expect("ONNX evaluator not initialized");
    let pairs: Vec<(Position, &MoveList)> = positions
        .iter()
        .copied()
        .zip(legal_lists.iter().copied())
        .collect();
    for chunk in pairs.chunks(eval.max_batch().max(1)) {
        for (result, (pos, legal)) in eval.batch_evaluate(chunk).iter().zip(chunk.iter()) {
            let mut filtered = Vec::with_capacity(legal.len);
            for i in 0..legal.len {
                let mv = legal.moves[i];
                filtered.push(result.logits[policy_index(pos.side, mv)]);
            }
            out.push((result.cp, filtered));
        }
    }
    out
}

/// EvalFn adapter for alpha-beta search (no policy needed).
pub fn eval_fn_onnx(pos: &Position) -> i32 {
    OnnxEvaluator::with_eval(|eval| {
        let empty = MoveList::new();
        eval.evaluate(pos, &empty).cp
    })
    .expect("ONNX evaluator not initialized")
}
