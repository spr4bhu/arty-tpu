"""
skew.py — Pure (framework-free) helpers for packing matrices into the
diagonal-skewed BRAM layout the TPU expects, and for interpreting results.

These functions are the single source of truth shared by:
  - the Cocotb simulation driver (tiling/tpu_driver.py)
  - the on-hardware UART host (host/tpu_uart.py)

Keeping them here guarantees the bytes a real FPGA receives are skewed and
packed identically to what the simulation verifies.
"""


def pack_signed(values, data_width=8):
    """Pack a list of signed integers into one wide unsigned word (LSB lane first).

    Lane i occupies bits [i*data_width +: data_width]. Matches the packing the
    systolic_array unpacks via a_in[i*DATA_WIDTH +: DATA_WIDTH].
    """
    mask = (1 << data_width) - 1
    result = 0
    for i, v in enumerate(values):
        if v < 0:
            v = (1 << data_width) + v
        result |= (v & mask) << (i * data_width)
    return result


def to_signed(val, acc_width=32):
    """Reinterpret an unsigned integer as an acc_width-bit signed value."""
    if val >= (1 << (acc_width - 1)):
        val -= (1 << acc_width)
    return val


def compute_bram_slices(A, B, N=4, data_width=8):
    """Pre-compute the diagonal-skewed BRAM words for one NxN tile pair.

    BRAM_A[t]: lane i = A[i][t-i]  if valid, else 0
    BRAM_B[t]: lane j = B[t-j][j]  if valid, else 0

    Returns (bram_a, bram_b): two lists of length 2*N-1 of packed integer words.
    """
    stream_len = 2 * N - 1
    bram_a, bram_b = [], []
    for t in range(stream_len):
        a_vals = [int(A[i][t - i]) if 0 <= (t - i) < N else 0 for i in range(N)]
        b_vals = [int(B[t - j][j]) if 0 <= (t - j) < N else 0 for j in range(N)]
        bram_a.append(pack_signed(a_vals, data_width))
        bram_b.append(pack_signed(b_vals, data_width))
    return bram_a, bram_b
