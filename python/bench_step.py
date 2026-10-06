import time
import torch
from chessnet.resnet import create_model
from chessnet.train_alpha import train_step

assert torch.cuda.is_available()
model = create_model(channels=256, num_blocks=20).to("cuda")
opt = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9,
                      weight_decay=1e-4)
scaler = torch.amp.GradScaler("cuda")
B = 4096
x = torch.randn(B, 105, 8, 8, device="cuda")
z = torch.rand(B, device="cuda")
pi = torch.zeros(B, 8, dtype=torch.long, device="cuda")
pv = torch.zeros(B, 8, device="cuda")
pm = torch.zeros(B, 8, dtype=torch.bool, device="cuda")
pi[:, 0] = 7
pv[:, 0] = 1.0
pm[:, 0] = True
# warmup (cudnn autotune)
train_step(model, opt, x, z, pi, pv, pm, scaler)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(5):
    out = train_step(model, opt, x, z, pi, pv, pm, scaler)
torch.cuda.synchronize()
dt = (time.time() - t0) / 5
print(f"isolated train_step batch4096: {dt:.2f}s/step loss={out[0]:.4f}")
