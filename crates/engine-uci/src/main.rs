//! chess-net UCI binary. Wires the engine to the NN eval backend.

fn main() {
    engine::init();
    engine::uci::run_with_hook(Some(&on_option));
}

fn on_option(name: &str, value: &str) -> bool {
    if name == "EvalFile" {
        if value.is_empty() || value == "<empty>" {
            engine::uci::set_eval(engine::evaluate::evaluate);
            return true;
        }
        match nn::load(value) {
            Ok(_) => {
                engine::uci::set_eval(nn::evaluate_loaded);
                println!("info string eval file loaded: {value}");
                true
            }
            Err(e) => {
                println!("info string eval load failed: {e}");
                true
            }
        }
    } else {
        false
    }
}