#!/usr/bin/env python3
"""
mnist_draw.py — Draw a digit with the mouse; the FPGA classifies it.

A tkinter canvas captures a hand-drawn digit, applies MNIST-style centering and
scaling, quantizes to int8, and runs the linear classifier's forward pass
`logits = image @ W` on the Arty A7-35T systolic array (host/tpu_uart.py). The
prediction and top-3 confidences update live.

The classifier weights come from host/mnist_fpga.py (trained + cached on first
run there, or trained here if missing).

Usage:
  python host/mnist_draw.py                  # classify on the FPGA
  python host/mnist_draw.py --cpu            # classify on CPU (no board)
  python host/mnist_draw.py --port /dev/ttyUSB1
"""

import argparse
import os
import sys
import threading

import numpy as np
import tkinter as tk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from host.mnist_fpga import get_model, quantize_weights, downsample

CANVAS = 280          # on-screen canvas size (px)
GRID = 28             # internal ink buffer resolution
CELL = CANVAS // GRID  # px per ink cell
BRUSH = 1.4           # stroke radius in grid cells


# ----------------------------------------------------- preprocessing helpers
def _resize(img, oh, ow):
    """Bilinear resize a 2-D array to (oh, ow)."""
    h, w = img.shape
    if h == 0 or w == 0:
        return np.zeros((oh, ow))
    yi = np.linspace(0, h - 1, oh); xi = np.linspace(0, w - 1, ow)
    y0 = np.floor(yi).astype(int); x0 = np.floor(xi).astype(int)
    y1 = np.minimum(y0 + 1, h - 1); x1 = np.minimum(x0 + 1, w - 1)
    wy = (yi - y0)[:, None]; wx = (xi - x0)[None, :]
    Ia = img[np.ix_(y0, x0)]; Ib = img[np.ix_(y0, x1)]
    Ic = img[np.ix_(y1, x0)]; Id = img[np.ix_(y1, x1)]
    top = Ia * (1 - wx) + Ib * wx
    bot = Ic * (1 - wx) + Id * wx
    return top * (1 - wy) + bot * wy


def center_and_scale(ink):
    """MNIST-style normalize: crop to ink, scale (keeping aspect) into a 20x20
    box, center by center-of-mass in a 28x28 frame. Matches how MNIST itself was
    built and makes hand drawings robust to position and size."""
    ys, xs = np.where(ink > 0.05)
    if len(xs) == 0:
        return np.zeros((28, 28))
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = ink[y0:y1, x0:x1]
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    small = _resize(crop, nh, nw)
    out = np.zeros((28, 28))
    if (small > 0.05).any():
        cy, cx = np.array(np.where(small > 0.05)).mean(axis=1)
    else:
        cy, cx = nh / 2, nw / 2
    top = int(round(14 - cy)); left = int(round(14 - cx))
    top = min(max(top, 0), 28 - nh); left = min(max(left, 0), 28 - nw)
    out[top:top + nh, left:left + nw] = small
    return out


def features_from_ink(ink):
    """Hand-drawn 28x28 ink (0..1) -> (1,196) int8 in 0..127 for the FPGA."""
    img28 = center_and_scale(ink) * 255.0                 # (28,28) 0..255
    x14 = downsample(img28[None, ...]).reshape(1, -1)     # (1,196) 0..255
    x01 = x14 / 255.0
    return np.round(x01 * 127).astype(np.int64)


# --------------------------------------------------------------------- app
class DrawApp:
    def __init__(self, root, port, use_cpu):
        self.root = root
        self.port = port
        self.ink = np.zeros((GRID, GRID))
        self.use_cpu = use_cpu
        self.tpu = None

        W, b, self.float_acc = get_model()
        self.Wq, wmax = quantize_weights(W)
        self.bias_score = b * (127.0 * 127.0 / wmax)
        # The integer accumulator is (127*127/wmax)x larger than the float
        # logit; divide back before softmax so confidences are meaningful
        # (without this, the huge integer gaps saturate exp() to ~100%).
        self.logit_scale = wmax / (127.0 * 127.0)

        root.title("Draw a digit")

        # Drawing surface (black bg / white pen matches MNIST).
        self.canvas = tk.Canvas(root, width=CANVAS, height=CANVAS,
                                bg="black", highlightthickness=1)
        self.canvas.grid(row=0, column=0, padx=8, pady=8)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        # Plain-text readout column.
        right = tk.Frame(root)
        right.grid(row=0, column=1, padx=8, sticky="n")
        mono = ("monospace", 11)
        self.result = tk.Label(right, text="prediction: -", font=("monospace", 13),
                               anchor="w", justify="left")
        self.result.pack(anchor="w", pady=(4, 8))
        self.detail = tk.Label(right, text="", font=mono, anchor="w",
                               justify="left")
        self.detail.pack(anchor="w")

        btns = tk.Frame(root)
        btns.grid(row=1, column=0, columnspan=2, pady=(0, 6))
        self.go = tk.Button(btns, text="Classify", command=self.classify, width=10)
        self.go.pack(side="left", padx=4)
        tk.Button(btns, text="Clear", command=self.clear, width=8).pack(side="left",
                                                                        padx=4)

        backend = "cpu" if use_cpu else f"fpga {port}"
        self.status = tk.Label(root, text=f"draw a digit  ·  backend: {backend}",
                               font=mono, anchor="w")
        self.status.grid(row=2, column=0, columnspan=2, sticky="we", padx=8,
                         pady=(0, 6))

    # ---- drawing ----
    def paint(self, ev):
        gx, gy = ev.x / CELL, ev.y / CELL
        r = int(BRUSH) + 1
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                x, y = int(gx) + dx, int(gy) + dy
                if 0 <= x < GRID and 0 <= y < GRID:
                    d = ((gx - x) ** 2 + (gy - y) ** 2) ** 0.5
                    val = max(0.0, 1.0 - d / BRUSH)
                    if val > self.ink[y, x]:
                        self.ink[y, x] = val
        self.canvas.create_oval(ev.x - 9, ev.y - 9, ev.x + 9, ev.y + 9,
                                fill="white", outline="white")

    def clear(self):
        self.ink[:] = 0
        self.canvas.delete("all")
        self.result.config(text="prediction: -")
        self.detail.config(text="")
        self.status.config(text="cleared")

    # ---- inference ----
    def classify(self):
        if self.ink.max() < 0.05:
            self.status.config(text="canvas is empty")
            return
        Xq = features_from_ink(self.ink)
        self.go.config(state="disabled")
        self.status.config(text="classifying on " +
                           ("cpu ..." if self.use_cpu else "fpga ..."))
        threading.Thread(target=self._infer, args=(Xq,), daemon=True).start()

    def _infer(self, Xq):
        try:
            if self.use_cpu:
                acc = Xq @ self.Wq
            else:
                if self.tpu is None:           # lazy, robust connect
                    from host.tpu_uart import TPUUart
                    self.tpu = TPUUart(port=self.port)
                acc = self.tpu.matmul(Xq, self.Wq)
            self.root.after(0, self._show, acc)
        except Exception as e:
            self.root.after(0, self._error, str(e))

    def _error(self, msg):
        self.status.config(text=f"error: {msg}")
        self.go.config(state="normal")

    def _show(self, acc):
        scores = acc.astype(np.float64)[0] + self.bias_score
        # back to the float-logit domain so softmax confidences are real
        z = scores * self.logit_scale
        z -= z.max()
        p = np.exp(z); p /= p.sum()
        pred = int(np.argmax(scores))
        order = np.argsort(scores)[::-1][:3]
        self.result.config(text=f"prediction: {pred}   ({p[pred]*100:.0f}%)")
        self.detail.config(text="\n".join(f"  {d}: {p[d]*100:5.1f}%" for d in order))
        self.status.config(text="done")
        self.go.config(state="normal")


def main():
    ap = argparse.ArgumentParser(description="Draw a digit, classify on the FPGA")
    ap.add_argument("--port", default="/dev/ttyUSB1")
    ap.add_argument("--cpu", action="store_true", help="classify on CPU, no board")
    args = ap.parse_args()

    root = tk.Tk()
    DrawApp(root, args.port, args.cpu)
    root.mainloop()


if __name__ == "__main__":
    main()
