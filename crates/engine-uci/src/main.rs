//! chess-net UCI binary. Wires the engine to the NN eval backend.

use engine::movegen::MoveList;
use engine::position::Position;

fn main() {
    engine::init();
    engine::uci::run_with_hook(Some(&on_option));
}

/// Batch ONNX evaluation: evaluates multiple positions in a single GPU call.
/// Delegates to the shared adapter (4096 policy logits filtered to legal
/// moves, side-to-move cp) so UCI and GUI search see identical values.
fn batch_eval_onnx(positions: &[Position], legal: &[&MoveList]) -> Vec<(i32, Vec<f32>)> {
    nn::onnx::batch_evaluate_onnx(positions, legal)
}

/// Expose the batch eval to the engine UCI loop.
pub fn batch_eval_fn() -> engine::mcts::BatchEvalFn {
    batch_eval_onnx as engine::mcts::BatchEvalFn
}

fn on_option(name: &str, value: &str) -> bool {
    if name == "EvalFile" {
        // GUIs may quote paths or use odd case; normalize for dispatch
        // but load the raw (trimmed) path.
        let path = value.trim().trim_matches('"');
        if path.is_empty() || path == "<empty>" {
            nn::onnx::OnnxEvaluator::unload();
            engine::uci::set_eval(engine::evaluate::evaluate);
            return true;
        }
        if path.to_ascii_lowercase().ends_with(".onnx") {
            // ONNX ResNet model — GPU inference with policy head
            match nn::onnx::OnnxEvaluator::init(path, 128) {
                Ok(_) => {
                    engine::uci::set_eval(nn::onnx::eval_fn_onnx);
                    // Register the ONNX policy head for single-threaded MCTS
                    engine::mcts::set_mcts_value_policy(nn::onnx::evaluate_onnx);
                    // Register batch eval for batched MCTS
                    engine::mcts::set_batch_eval(batch_eval_onnx);
                    println!("info string ONNX eval loaded: {path}");
                    true
                }
                Err(e) => {
                    println!("info string ONNX load failed: {e}");
                    true
                }
            }
        } else {
            // CSNN model — CPU inference
            match nn::load(path) {
                Ok(_) => {
                    // Take ONNX out of the loop so the old session can't
                    // keep answering (it would mix stale NN priors with
                    // the CSNN value = franken-eval).
                    nn::onnx::OnnxEvaluator::unload();
                    engine::uci::set_eval(nn::evaluate_loaded_stm);
                    engine::mcts::set_mcts_value_policy(nn::evaluate_loaded_combined);
                    println!("info string eval file loaded: {path}");
                    true
                }
                Err(e) => {
                    println!("info string eval load failed: {e}");
                    true
                }
            }
        }
    } else {
        false
    }
}