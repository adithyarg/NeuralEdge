"""
quantize_thermal.py — Export thermal MLP weights as $readmemh hex files.

Run AFTER train_thermal.py:
    python training/thermal/train_thermal.py
    python training/thermal/quantize_thermal.py

Fixed-point formats (same as MNIST accelerator):
    Weights : Q4.12  signed 16-bit   scale = 4096
    Biases  : Q8.8   signed 16-bit   scale = 256

Output: rtl/hex/thermal/*.hex
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn


class ThermalMLP(nn.Module):
    def __init__(self, in_features=1024, hidden1=64, hidden2=32, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def to_fixed16(arr, frac_bits):
    scale   = 2 ** frac_bits
    clipped = np.clip(arr * scale, -(2**15), 2**15 - 1)
    return clipped.round().astype(np.int16)


def write_hex(arr_i16, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for v in arr_i16.flatten():
            f.write(f"{int(v) & 0xFFFF:04x}\n")
    print(f"  {path}  ({arr_i16.size} entries)")


def main():
    pt_file = "training/thermal/weights/mlp_thermal.pt"
    hex_dir = "rtl/hex/thermal"

    if not os.path.exists(pt_file):
        print(f"ERROR: {pt_file} not found — run train_thermal.py first")
        sys.exit(1)

    model = ThermalMLP()
    model.load_state_dict(torch.load(pt_file, map_location="cpu"))
    model.eval()

    layers = [
        (model.fc1.weight, model.fc1.bias, "l1"),
        (model.fc2.weight, model.fc2.bias, "l2"),
        (model.fc3.weight, model.fc3.bias, "l3"),
    ]

    print("Exporting thermal hex files...")
    for W_t, b_t, tag in layers:
        W = W_t.detach().numpy().astype(np.float64)
        b = b_t.detach().numpy().astype(np.float64)
        write_hex(to_fixed16(W, 12), f"{hex_dir}/weight_{tag}.hex")
        write_hex(to_fixed16(b,  8), f"{hex_dir}/bias_{tag}.hex")

    print(f"\nDone. Hex files in {hex_dir}/")
    print("\nTo use in Vivado, instantiate top_basys3 with:")
    print("  .IMG_PIXELS(1024),")
    print("  .L1_IN(1024), .L1_OUT(64),")
    print("  .L2_IN(64),   .L2_OUT(32),")
    print("  .L3_IN(32),   .L3_OUT(2),")
    print('  .W1_HEX("hex/thermal/weight_l1.hex"),')
    print('  .W2_HEX("hex/thermal/weight_l2.hex"),')
    print('  .W3_HEX("hex/thermal/weight_l3.hex"),')
    print('  .B1_HEX("hex/thermal/bias_l1.hex"),')
    print('  .B2_HEX("hex/thermal/bias_l2.hex"),')
    print('  .B3_HEX("hex/thermal/bias_l3.hex")')


if __name__ == "__main__":
    main()
