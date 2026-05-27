"""
Cocotb testbench for tpu_top.

The host interface:
  - Write pre-skewed diagonal slices to BRAM_A and BRAM_B
  - Assert start; poll until done
  - Read N*N results from BRAM_C (async, row-major)
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ClockCycles
import numpy as np


def pack_signed(values, width):
    """Pack a list of signed integers into a single wide unsigned value (LSB-first)."""
    mask = (1 << width) - 1
    result = 0
    for i, v in enumerate(values):
        if v < 0:
            v = (1 << width) + v
        result |= (v & mask) << (i * width)
    return result


def to_signed(val, width):
    """Reinterpret an unsigned integer as a signed integer of the given width."""
    if val >= (1 << (width - 1)):
        val -= (1 << width)
    return val


def compute_bram_slices(A, B, N, DATA_WIDTH):
    """
    Pre-compute the diagonal-skewed BRAM contents for matrices A and B.

    Returns two lists of length STREAM_LEN = 2*N-1, each element being
    a packed integer word (N lanes of DATA_WIDTH bits).

    BRAM_A[t]: lane i = A[i][t-i]  if 0 <= t-i < N, else 0
    BRAM_B[t]: lane j = B[t-j][j]  if 0 <= t-j < N, else 0
    """
    stream_len = 2 * N - 1
    bram_a = []
    bram_b = []
    for t in range(stream_len):
        a_vals = [int(A[i][t - i]) if 0 <= (t - i) < N else 0 for i in range(N)]
        b_vals = [int(B[t - j][j]) if 0 <= (t - j) < N else 0 for j in range(N)]
        bram_a.append(pack_signed(a_vals, DATA_WIDTH))
        bram_b.append(pack_signed(b_vals, DATA_WIDTH))
    return bram_a, bram_b


async def reset_dut(dut):
    dut.rst.value     = 1
    dut.start.value   = 0
    dut.wr_en_a.value = 0
    dut.wr_en_b.value = 0
    dut.rd_addr_c.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def load_bram(dut, slices, wr_en, wr_addr, wr_data):
    """Write pre-skewed slice words into BRAM_A or BRAM_B."""
    for t, word in enumerate(slices):
        wr_en.value   = 1
        wr_addr.value = t
        wr_data.value = word
        await RisingEdge(dut.clk)
    wr_en.value = 0
    await RisingEdge(dut.clk)


async def run_tpu(dut):
    """Pulse start and wait for done."""
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    # Poll until done pulses
    for _ in range(500):
        await RisingEdge(dut.clk)
        if dut.done.value == 1:
            return
    raise TimeoutError("TPU did not assert done within timeout")


async def read_result(dut, N, ACC_WIDTH):
    """
    Read N*N results from BRAM_C via the async read port.
    Sets rd_addr_c and waits 1 ns for combinatorial logic to settle.
    """
    C = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(N):
            dut.rd_addr_c.value = i * N + j
            await Timer(1, units="ns")
            C[i][j] = to_signed(int(dut.rd_data_c.value), ACC_WIDTH)
    return C


def get_params(dut):
    N          = int(dut.N.value)
    DATA_WIDTH = int(dut.DATA_WIDTH.value)
    ACC_WIDTH  = int(dut.ACC_WIDTH.value)
    return N, DATA_WIDTH, ACC_WIDTH


@cocotb.test()
async def test_identity_matrix(dut):
    """A x I = A"""
    N, DATA_WIDTH, ACC_WIDTH = get_params(dut)
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    await reset_dut(dut)

    A = np.array([[i * N + j + 1 for j in range(N)] for i in range(N)], dtype=np.int64)
    I = np.eye(N, dtype=np.int64)
    expected = A @ I

    bram_a_slices, bram_b_slices = compute_bram_slices(A, I, N, DATA_WIDTH)
    await load_bram(dut, bram_a_slices, dut.wr_en_a, dut.wr_addr_a, dut.wr_data_a)
    await load_bram(dut, bram_b_slices, dut.wr_en_b, dut.wr_addr_b, dut.wr_data_b)

    await run_tpu(dut)

    C = await read_result(dut, N, ACC_WIDTH)
    dut._log.info(f"A:\n{A}\nExpected:\n{expected}\nGot:\n{C}")
    assert np.array_equal(C, expected), f"Identity FAILED\nGot:\n{C}\nExpected:\n{expected}"
    dut._log.info("test_identity_matrix PASSED")


@cocotb.test()
async def test_full_matrix(dut):
    """Full NxN multiply with the reference test vectors from the riscv-gpu repo."""
    N, DATA_WIDTH, ACC_WIDTH = get_params(dut)
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    await reset_dut(dut)

    A = np.array([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
    ], dtype=np.int64)[:N, :N]

    B = np.array([
        [5,  6,  7,  8],
        [6,  7,  8,  9],
        [7,  8,  9, 10],
        [8,  9, 10, 11],
    ], dtype=np.int64)[:N, :N]

    expected = A @ B

    bram_a_slices, bram_b_slices = compute_bram_slices(A, B, N, DATA_WIDTH)
    await load_bram(dut, bram_a_slices, dut.wr_en_a, dut.wr_addr_a, dut.wr_data_a)
    await load_bram(dut, bram_b_slices, dut.wr_en_b, dut.wr_addr_b, dut.wr_data_b)

    await run_tpu(dut)

    C = await read_result(dut, N, ACC_WIDTH)
    dut._log.info(f"A:\n{A}\nB:\n{B}\nExpected:\n{expected}\nGot:\n{C}")
    assert np.array_equal(C, expected), f"Full matrix FAILED\nGot:\n{C}\nExpected:\n{expected}"
    dut._log.info("test_full_matrix PASSED")


@cocotb.test()
async def test_negative_values(dut):
    """Signed arithmetic with negative entries."""
    N, DATA_WIDTH, ACC_WIDTH = get_params(dut)
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    await reset_dut(dut)

    A = np.zeros((N, N), dtype=np.int64)
    B = np.zeros((N, N), dtype=np.int64)
    A[:2, :2] = [[ 3, -2], [-1,  4]]
    B[:2, :2] = [[-5,  6], [ 7, -8]]
    expected = A @ B

    bram_a_slices, bram_b_slices = compute_bram_slices(A, B, N, DATA_WIDTH)
    await load_bram(dut, bram_a_slices, dut.wr_en_a, dut.wr_addr_a, dut.wr_data_a)
    await load_bram(dut, bram_b_slices, dut.wr_en_b, dut.wr_addr_b, dut.wr_data_b)

    await run_tpu(dut)

    C = await read_result(dut, N, ACC_WIDTH)
    dut._log.info(f"Expected:\n{expected}\nGot:\n{C}")
    assert np.array_equal(C, expected), f"Negative values FAILED\nGot:\n{C}\nExpected:\n{expected}"
    dut._log.info("test_negative_values PASSED")


@cocotb.test()
async def test_back_to_back(dut):
    """Two consecutive multiplications without a full reset between them."""
    N, DATA_WIDTH, ACC_WIDTH = get_params(dut)
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    await reset_dut(dut)

    # First multiply: all-2s  x  all-3s
    A1 = np.full((N, N), 2, dtype=np.int64)
    B1 = np.full((N, N), 3, dtype=np.int64)
    exp1 = A1 @ B1

    s_a, s_b = compute_bram_slices(A1, B1, N, DATA_WIDTH)
    await load_bram(dut, s_a, dut.wr_en_a, dut.wr_addr_a, dut.wr_data_a)
    await load_bram(dut, s_b, dut.wr_en_b, dut.wr_addr_b, dut.wr_data_b)
    await run_tpu(dut)
    C1 = await read_result(dut, N, ACC_WIDTH)
    assert np.array_equal(C1, exp1), f"Back-to-back 1st FAILED\nGot:\n{C1}\nExpected:\n{exp1}"
    dut._log.info(f"Back-to-back 1st PASSED: result=\n{C1}")

    # Second multiply (new data, no reset)
    A2 = np.diag(np.arange(1, N + 1, dtype=np.int64))
    B2 = np.array([[i + j for j in range(N)] for i in range(N)], dtype=np.int64)
    exp2 = A2 @ B2

    s_a, s_b = compute_bram_slices(A2, B2, N, DATA_WIDTH)
    await load_bram(dut, s_a, dut.wr_en_a, dut.wr_addr_a, dut.wr_data_a)
    await load_bram(dut, s_b, dut.wr_en_b, dut.wr_addr_b, dut.wr_data_b)
    await run_tpu(dut)
    C2 = await read_result(dut, N, ACC_WIDTH)
    assert np.array_equal(C2, exp2), f"Back-to-back 2nd FAILED\nGot:\n{C2}\nExpected:\n{exp2}"
    dut._log.info(f"Back-to-back 2nd PASSED: result=\n{C2}")

    dut._log.info("test_back_to_back PASSED")


@cocotb.test()
async def test_random_matrices(dut):
    """Property-based test: random signed matrices verified against NumPy."""
    N, DATA_WIDTH, ACC_WIDTH = get_params(dut)
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    rng = np.random.default_rng(seed=42)
    max_val = (1 << (DATA_WIDTH - 1)) - 1  # 127 for 8-bit

    for trial in range(5):
        await reset_dut(dut)

        A = rng.integers(-max_val, max_val + 1, size=(N, N)).astype(np.int64)
        B = rng.integers(-max_val, max_val + 1, size=(N, N)).astype(np.int64)
        expected = A @ B

        s_a, s_b = compute_bram_slices(A, B, N, DATA_WIDTH)
        await load_bram(dut, s_a, dut.wr_en_a, dut.wr_addr_a, dut.wr_data_a)
        await load_bram(dut, s_b, dut.wr_en_b, dut.wr_addr_b, dut.wr_data_b)
        await run_tpu(dut)
        C = await read_result(dut, N, ACC_WIDTH)

        dut._log.info(f"Trial {trial}: expected=\n{expected}\ngot=\n{C}")
        assert np.array_equal(C, expected), (
            f"Random trial {trial} FAILED\nA:\n{A}\nB:\n{B}\nGot:\n{C}\nExpected:\n{expected}"
        )

    dut._log.info("test_random_matrices PASSED (5 trials)")
