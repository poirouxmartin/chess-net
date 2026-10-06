"""Export a tiny 105-plane test net for the Rust onnx_4288 test."""
import os
import torch

from chessnet.resnet import create_model
from chessnet.train_alpha import export_onnx

torch.manual_seed(7)
model = create_model(channels=32, num_blocks=2)
out = os.path.join(os.path.dirname(__file__),
                   "../crates/nn/tests/data/net105_4288.onnx")
os.makedirs(os.path.dirname(out), exist_ok=True)
export_onnx(model, out, fp16=False)
print("wrote", out, os.path.getsize(out), "bytes")
