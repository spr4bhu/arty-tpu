#!/usr/bin/env python3
"""
mnist_fpga.py — Handwritten-digit recognition running on the FPGA TPU.

A linear (softmax) classifier is trained in pure numpy, quantized to signed
int8, and its forward pass `logits = image @ W` is executed **on the Arty
A7-35T** through the UART matmul service (host/tpu_uart.py). Each test digit is
drawn as ASCII art next to the chip's prediction.

This is a visual demo on top of the rigorous numpy self-test in tpu_uart.py.

Pipeline:
  28x28 uint8 image -> 14x14 downsample -> quantize to int8 (0..127)
  Xq (B x 196) @ Wq (196 x 10)  --> int32 logits  [computed on the FPGA]
  + quantized bias, argmax -> predicted digit

Usage:
  python host/mnist_fpga.py                 # train (cached) + classify on FPGA
  python host/mnist_fpga.py --images 12     # number of test digits to show
  python host/mnist_fpga.py --cpu           # skip FPGA, run model on CPU only
"""

import argparse
import gzip
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(os.path.dirname(__file__), "mnist_data")
MODEL_NPZ = os.path.join(DATA_DIR, "linear_model.npz")
MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}


# ---------------------------------------------------------------- data loading
def _download(fname):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        return path
    last = None
    for base in MIRRORS:
        try:
            print(f"  downloading {fname} ...")
            urllib.request.urlretrieve(base + fname, path)
            return path
        except Exception as e:  # try next mirror
            last = e
    raise RuntimeError(f"could not download {fname}: {last}")


def _read_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[:4], "big")
    ndim = magic & 0xFF
    dims = [int.from_bytes(data[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim)]
    arr = np.frombuffer(data[4 + 4 * ndim:], dtype=np.uint8)
    return arr.reshape(dims)


def load_mnist():
    paths = {k: _download(v) for k, v in FILES.items()}
    return (_read_idx(paths["train_x"]), _read_idx(paths["train_y"]),
            _read_idx(paths["test_x"]), _read_idx(paths["test_y"]))


# ------------------------------------------------------------- preprocessing
def downsample(imgs):
    """28x28 -> 14x14 by 2x2 average pooling. imgs: (N,28,28) uint8."""
    n = imgs.shape[0]
    x = imgs.reshape(n, 14, 2, 14, 2).mean(axis=(2, 4))
    return x  # (N,14,14) float 0..255


def to_features(imgs):
    """Flatten downsampled images to (N,196) normalized to [0,1]."""
    return downsample(imgs).reshape(imgs.shape[0], -1) / 255.0


# ----------------------------------------------------------------- training
def train_linear(Xtr, ytr, epochs=30, lr=0.5, batch=256, seed=0):
    """Softmax-regression trained with mini-batch gradient descent (numpy)."""
    rng = np.random.default_rng(seed)
    n, d = Xtr.shape
    W = np.zeros((d, 10)); b = np.zeros(10)
    Y = np.eye(10)[ytr]
    for ep in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            bi = idx[s:s + batch]
            x, y = Xtr[bi], Y[bi]
            z = x @ W + b
            z -= z.max(axis=1, keepdims=True)
            p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
            g = (p - y) / len(bi)
            W -= lr * (x.T @ g)
            b -= lr * g.sum(axis=0)
    return W, b


def get_model(retrain=False):
    if os.path.exists(MODEL_NPZ) and not retrain:
        d = np.load(MODEL_NPZ)
        return d["W"], d["b"], float(d["acc"])
    print("Training linear classifier (numpy) ...")
    trx, tr_y, tex, tey = load_mnist()
    Xtr, Xte = to_features(trx), to_features(tex)
    W, b = train_linear(Xtr, tr_y)
    acc = float((np.argmax(Xte @ W + b, axis=1) == tey).mean())
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(MODEL_NPZ, W=W, b=b, acc=acc)
    print(f"  float model test accuracy: {acc*100:.1f}%")
    return W, b, acc


# --------------------------------------------------------------- quantization
def quantize_weights(W):
    wmax = np.abs(W).max()
    Wq = np.round(W / wmax * 127).astype(np.int64)
    return Wq, wmax


def quantize_features(X01):
    """[0,1] features -> int8 in 0..127."""
    return np.round(X01 * 127).astype(np.int64)


# ------------------------------------------------------------------ ascii art
def ascii_digit(img28):
    ramp = " .:-=+*#%@"
    out = []
    for row in img28:
        out.append("".join(ramp[min(9, int(v) * 10 // 256)] for v in row))
    return out


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="MNIST inference on the FPGA TPU")
    ap.add_argument("--port", default="/dev/ttyUSB1")
    ap.add_argument("--images", type=int, default=12, help="test digits to show")
    ap.add_argument("--cpu", action="store_true", help="run on CPU, skip FPGA")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    W, b, float_acc = get_model(retrain=args.retrain)
    Wq, wmax = quantize_weights(W)
    bias_score = b * (127.0 * 127.0 / wmax)   # bias in the int-accumulator domain
    logit_scale = wmax / (127.0 * 127.0)      # rescale int acc -> float logit

    # Pick random test digits.
    _, _, tex, tey = load_mnist()
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(len(tex), size=args.images, replace=False)
    imgs28 = tex[sel]
    labels = tey[sel]
    Xq = quantize_features(to_features(imgs28))   # (B,196) int8

    # Compute logits: on the FPGA (default) or CPU.
    if args.cpu:
        acc_int = Xq @ Wq
        backend = "CPU (numpy)"
    else:
        from host.tpu_uart import TPUUart
        tpu = TPUUart(port=args.port)
        print(f"Running inference on FPGA ({len(sel)} digits, batched) ...")
        # Batch in groups of N (=4): the M dimension pads to N anyway, so a
        # batch of 4 images costs the same as 1 — showcases the array width.
        N = tpu.N
        chunks = []
        t0 = time.time()
        for s in range(0, len(Xq), N):
            chunk = Xq[s:s + N]
            res = tpu.matmul(chunk, Wq,
                             progress=lambda d, t: print(f"\r  tiles {d}/{t}",
                                                         end="", flush=True))
            chunks.append(res)
        print(f"\r  FPGA matmul done in {time.time()-t0:.1f}s" + " " * 12)
        tpu.close()
        acc_int = np.vstack(chunks)
        backend = "FPGA (Arty A7-35T systolic array)"

    scores = acc_int.astype(np.float64) + bias_score
    preds = np.argmax(scores, axis=1)
    # softmax confidence for display — rescale to the float-logit domain first,
    # else the ~16000x-larger integer gaps saturate exp() to a constant 100%.
    z = scores * logit_scale
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    conf = p[np.arange(len(preds)), preds]

    # ---- visual report ----
    print("\n" + "=" * 60)
    print(f"  MNIST inference — logits computed on: {backend}")
    print("=" * 60)
    correct = 0
    for i in range(len(sel)):
        art = ascii_digit(imgs28[i])
        mark = "OK " if preds[i] == labels[i] else "XX "
        correct += int(preds[i] == labels[i])
        print()
        for r, line in enumerate(art):
            tag = ""
            if r == 10:
                tag = f"   {mark} predicted: {preds[i]}   (true {labels[i]})"
            elif r == 12:
                tag = f"       confidence: {conf[i]*100:.0f}%"
            print("   " + line + tag)
    print("\n" + "-" * 60)
    print(f"  Demo accuracy: {correct}/{len(sel)} "
          f"= {correct/len(sel)*100:.0f}%   "
          f"(quantized model full-test float acc ~{float_acc*100:.0f}%)")

    # If on FPGA, cross-check against CPU int math (must match exactly).
    if not args.cpu:
        cpu_int = Xq @ Wq
        if np.array_equal(cpu_int, acc_int):
            print("  FPGA vs CPU integer logits: EXACT MATCH ✓")
        else:
            n_bad = int((cpu_int != acc_int).sum())
            print(f"  WARNING: FPGA/CPU logits differ in {n_bad} entries")
    print("-" * 60)


if __name__ == "__main__":
    main()
