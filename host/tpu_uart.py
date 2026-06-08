#!/usr/bin/env python3
"""
tpu_uart.py — Host driver for the uart_matmul FPGA bitstream (Basys 3).

Speaks the fixed UART protocol implemented by rtl/uart_matmul.v:

  Host -> FPGA : 0xA5 sync, then (2N-1) A slices, then (2N-1) B slices,
                 each slice = N*DATA_WIDTH/8 bytes, little-endian, lane0 first.
  FPGA -> Host : N*N result words, ACC_WIDTH/8 bytes each, little-endian,
                 row-major signed.

The A/B skewing/packing is shared with the simulation via tiling.skew, so the
bytes the hardware receives are identical to what the Cocotb tests verify.

Examples:
  python host/tpu_uart.py --self-test            # random matmuls vs numpy
  python host/tpu_uart.py --port /dev/ttyUSB1    # custom serial port
"""

import argparse
import os
import sys
import time

import numpy as np
import serial  # pyserial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tiling.skew import compute_bram_slices, to_signed


class TPUUart:
    """Drives one 4x4 (or NxN) signed-int matmul tile over UART."""

    SYNC = 0xA5

    def __init__(self, port="/dev/ttyUSB1", baud=115200,
                 N=4, data_width=8, acc_width=32, timeout=2.0):
        self.N = N
        self.data_width = data_width
        self.acc_width = acc_width
        self.slice_bytes = (N * data_width) // 8
        self.nslices = 2 * N - 1
        self.cbytes = acc_width // 8
        self.cwords = N * N
        self.timeout = timeout
        self.ser = serial.Serial(port, baud, timeout=timeout)
        # Absorb any line glitch the FPGA's TX pin emits as it settles after
        # configuration (these land in the OS buffer and would desync the
        # first transaction).
        time.sleep(0.1)
        self._drain(soak=True)

    def close(self):
        self.ser.close()

    def _drain(self, soak=False):
        """Clear pending input. With soak=True, also spend a moment absorbing
        in-flight bytes (used once at connect for the post-config glitch); the
        plain reset is instant and used between in-sync transactions."""
        self.ser.reset_input_buffer()
        if soak:
            old = self.ser.timeout
            self.ser.timeout = 0.02
            while self.ser.read(64):
                pass
            self.ser.timeout = old

    def _frame(self, A, B):
        """Build the host->FPGA byte frame for tiles A, B (NxN int arrays)."""
        sa, sb = compute_bram_slices(A, B, self.N, self.data_width)
        out = bytearray([self.SYNC])
        for w in sa:
            out += int(w).to_bytes(self.slice_bytes, "little")
        for w in sb:
            out += int(w).to_bytes(self.slice_bytes, "little")
        return bytes(out)

    def matmul_tile(self, A, B):
        """Run one NxN tile on the FPGA; return the (N,N) int64 result."""
        A = np.asarray(A)
        B = np.asarray(B)
        assert A.shape == (self.N, self.N) and B.shape == (self.N, self.N)

        self._drain()
        self.ser.write(self._frame(A, B))
        self.ser.flush()

        nbytes = self.cwords * self.cbytes
        resp = self.ser.read(nbytes)
        if len(resp) != nbytes:
            raise TimeoutError(
                f"expected {nbytes} bytes from FPGA, got {len(resp)}")

        C = np.zeros((self.N, self.N), dtype=np.int64)
        for idx in range(self.cwords):
            word = resp[idx * self.cbytes:(idx + 1) * self.cbytes]
            val = int.from_bytes(word, "little")
            C[idx // self.N][idx % self.N] = to_signed(val, self.acc_width)
        return C

    def matmul(self, A, B, progress=None):
        """Compute C = A @ B for arbitrary shapes by tiling over the NxN core.

        A : (M, K) int, B : (K, P) int  ->  C : (M, P) int64.
        Pads each dimension to a multiple of N, runs every NxN tile on the
        FPGA, and accumulates partial products in software (int64). Mirrors
        tiling/tiling_engine.py but synchronous over UART.

        `progress(done, total)` is called after each tile if provided.
        """
        N = self.N
        A = np.asarray(A, dtype=np.int64)
        B = np.asarray(B, dtype=np.int64)
        M, K = A.shape
        K2, P = B.shape
        assert K == K2, f"inner dims must match: {A.shape} @ {B.shape}"

        def ceil_div(a, b):
            return (a + b - 1) // b

        Mp, Kp, Pp = ceil_div(M, N) * N, ceil_div(K, N) * N, ceil_div(P, N) * N
        Ap = np.zeros((Mp, Kp), dtype=np.int64); Ap[:M, :K] = A
        Bp = np.zeros((Kp, Pp), dtype=np.int64); Bp[:K, :P] = B
        Cp = np.zeros((Mp, Pp), dtype=np.int64)

        tiles_m, tiles_k, tiles_p = Mp // N, Kp // N, Pp // N
        total = tiles_m * tiles_k * tiles_p
        done = 0
        for ti in range(tiles_m):
            for tj in range(tiles_p):
                part = np.zeros((N, N), dtype=np.int64)
                for tk in range(tiles_k):
                    a = Ap[ti*N:(ti+1)*N, tk*N:(tk+1)*N]
                    b = Bp[tk*N:(tk+1)*N, tj*N:(tj+1)*N]
                    part += self.matmul_tile(a, b)
                    done += 1
                    if progress:
                        progress(done, total)
                Cp[ti*N:(ti+1)*N, tj*N:(tj+1)*N] = part
        return Cp[:M, :P]


def self_test(port, trials=10, seed=0):
    """Drive random int8 matmuls and compare the FPGA result to numpy."""
    rng = np.random.default_rng(seed)
    tpu = TPUUart(port=port)
    N = tpu.N
    try:
        # Deterministic edge cases first, then random trials.
        cases = [
            ("identity", np.eye(N, dtype=np.int64),
                         rng.integers(-128, 128, size=(N, N))),
            ("negatives", rng.integers(-128, 0, size=(N, N)),
                          rng.integers(-128, 0, size=(N, N))),
        ]
        for _ in range(trials):
            cases.append(("random",
                          rng.integers(-128, 128, size=(N, N)),
                          rng.integers(-128, 128, size=(N, N))))

        failures = 0
        for name, A, B in cases:
            got = tpu.matmul_tile(A, B)
            exp = A.astype(np.int64) @ B.astype(np.int64)
            ok = np.array_equal(got, exp)
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
            if not ok:
                failures += 1
                print("    A=\n", A, "\n    B=\n", B)
                print("    got=\n", got, "\n    exp=\n", exp)

        total = len(cases)
        print(f"\n{total - failures}/{total} cases passed")
        return failures == 0
    finally:
        tpu.close()


def main():
    ap = argparse.ArgumentParser(description="UART host for the FPGA TPU")
    ap.add_argument("--port", default="/dev/ttyUSB1", help="serial device")
    ap.add_argument("--self-test", action="store_true",
                    help="run random matmuls and compare against numpy")
    ap.add_argument("--trials", type=int, default=10,
                    help="number of random trials in --self-test")
    args = ap.parse_args()

    if args.self_test:
        ok = self_test(args.port, trials=args.trials)
        sys.exit(0 if ok else 1)

    # Default: one demonstration matmul.
    tpu = TPUUart(port=args.port)
    try:
        A = np.arange(1, 17).reshape(4, 4)
        B = np.eye(4, dtype=np.int64)
        print("A @ I =\n", tpu.matmul_tile(A, B))
    finally:
        tpu.close()


if __name__ == "__main__":
    main()
