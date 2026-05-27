"""
Cocotb testbench for parameterized NxN systolic array.

Tests matrix multiplication C = A x B by feeding inputs in the
diagonal-skewed pattern required by the systolic dataflow.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import numpy as np


def get_params(dut):
    """Extract array parameters from the DUT."""
    N = int(dut.N.value)
    DATA_WIDTH = int(dut.DATA_WIDTH.value)
    ACC_WIDTH = int(dut.ACC_WIDTH.value)
    return N, DATA_WIDTH, ACC_WIDTH


def pack_signed(values, width):
    """Pack a list of signed integers into a single wide value (LSB-first per index)."""
    mask = (1 << width) - 1
    result = 0
    for i, v in enumerate(values):
        # Convert to two's complement
        if v < 0:
            v = (1 << width) + v
        result |= (v & mask) << (i * width)
    return result


def unpack_signed(packed, n_elements, width):
    """Unpack a wide value into a list of signed integers."""
    mask = (1 << width) - 1
    sign_bit = 1 << (width - 1)
    values = []
    for i in range(n_elements):
        v = (int(packed) >> (i * width)) & mask
        if v & sign_bit:
            v -= (1 << width)
        values.append(v)
    return values


async def reset_dut(dut):
    """Assert reset for a few cycles."""
    dut.rst.value = 1
    dut.clear.value = 0
    dut.a_in.value = 0
    dut.b_in.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def clear_accumulators(dut):
    """Pulse clear to zero all PE accumulators."""
    dut.clear.value = 1
    await RisingEdge(dut.clk)
    dut.clear.value = 0
    await RisingEdge(dut.clk)


async def feed_matrices(dut, A, B, N, DATA_WIDTH):
    """
    Feed matrices A and B into the systolic array with diagonal skewing.

    For an NxN systolic array, data is fed over 2*N - 1 cycles.
    At cycle t (0-indexed):
      - Row i feeds A[i][t-i] if 0 <= t-i < N, else 0
      - Col j feeds B[t-j][j] if 0 <= t-j < N, else 0
    """
    total_cycles = 2 * N - 1

    for t in range(total_cycles):
        a_vals = []
        b_vals = []
        for i in range(N):
            k = t - i
            if 0 <= k < N:
                a_vals.append(int(A[i][k]))
            else:
                a_vals.append(0)
        for j in range(N):
            k = t - j
            if 0 <= k < N:
                b_vals.append(int(B[k][j]))
            else:
                b_vals.append(0)

        dut.a_in.value = pack_signed(a_vals, DATA_WIDTH)
        dut.b_in.value = pack_signed(b_vals, DATA_WIDTH)
        await RisingEdge(dut.clk)

    # Zero inputs and flush pipeline
    dut.a_in.value = 0
    dut.b_in.value = 0
    await ClockCycles(dut.clk, N)


def read_result(dut, N, ACC_WIDTH):
    """Read the C matrix from the packed output."""
    all_vals = unpack_signed(dut.c_out.value, N * N, ACC_WIDTH)
    C = np.zeros((N, N), dtype=np.int64)
    for row in range(N):
        for col in range(N):
            C[row][col] = all_vals[row * N + col]
    return C


@cocotb.test()
async def test_identity_matrix(dut):
    """Test multiplication with identity matrix: A x I = A."""
    N, DATA_WIDTH, ACC_WIDTH = get_params(dut)
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)
    await clear_accumulators(dut)

    A = np.array([[i * N + j + 1 for j in range(N)] for i in range(N)], dtype=np.int64)
    I = np.eye(N, dtype=np.int64)

    expected = A @ I

    await feed_matrices(dut, A, I, N, DATA_WIDTH)

    C = read_result(dut, N, ACC_WIDTH)
    dut._log.info(f"A:\n{A}")
    dut._log.info(f"Result:\n{C}")
    dut._log.info(f"Expected:\n{expected}")

    assert np.array_equal(C, expected), f"Identity test FAILED!\nGot:\n{C}\nExpected:\n{expected}"
    dut._log.info("Identity matrix test PASSED!")


@cocotb.test()
async def test_simple_2x2_in_4x4(dut):
    """
    Test the reference 2x2 example embedded in NxN:
    A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    Expected C = [[19, 22], [43, 50]]
    (padded to NxN with zeros)
    """
    N, DATA_WIDTH, ACC_WIDTH = get_params(dut)
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)
    await clear_accumulators(dut)

    A = np.zeros((N, N), dtype=np.int64)
    B = np.zeros((N, N), dtype=np.int64)
    A[0:2, 0:2] = [[1, 2], [3, 4]]
    B[0:2, 0:2] = [[5, 6], [7, 8]]

    expected = A @ B

    await feed_matrices(dut, A, B, N, DATA_WIDTH)

    C = read_result(dut, N, ACC_WIDTH)
    dut._log.info(f"Result:\n{C}")
    dut._log.info(f"Expected:\n{expected}")

    assert np.array_equal(C, expected), f"2x2 embedded test FAILED!\nGot:\n{C}\nExpected:\n{expected}"
    dut._log.info("Simple 2x2 embedded test PASSED!")


@cocotb.test()
async def test_full_matrix(dut):
    """
    Full NxN multiplication matching the reference repo's 4x4 test:
    A = [[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]]
    B = [[5,6,7,8],[6,7,8,9],[7,8,9,10],[8,9,10,11]]
    Expected C = [[70,80,90,100],[96,110,124,138],[122,140,158,176],[148,170,192,214]]
    """
    N, DATA_WIDTH, ACC_WIDTH = get_params(dut)
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)
    await clear_accumulators(dut)

    A = np.array([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7]
    ], dtype=np.int64)[:N, :N]

    B = np.array([
        [5, 6, 7, 8],
        [6, 7, 8, 9],
        [7, 8, 9, 10],
        [8, 9, 10, 11]
    ], dtype=np.int64)[:N, :N]

    expected = A @ B

    await feed_matrices(dut, A, B, N, DATA_WIDTH)

    C = read_result(dut, N, ACC_WIDTH)
    dut._log.info(f"A:\n{A}")
    dut._log.info(f"B:\n{B}")
    dut._log.info(f"Result:\n{C}")
    dut._log.info(f"Expected:\n{expected}")

    assert np.array_equal(C, expected), f"Full matrix test FAILED!\nGot:\n{C}\nExpected:\n{expected}"
    dut._log.info("Full matrix test PASSED!")


@cocotb.test()
async def test_negative_values(dut):
    """Test with negative matrix entries."""
    N, DATA_WIDTH, ACC_WIDTH = get_params(dut)
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)
    await clear_accumulators(dut)

    max_val = (1 << (DATA_WIDTH - 1)) - 1  # 127 for 8-bit

    A = np.zeros((N, N), dtype=np.int64)
    B = np.zeros((N, N), dtype=np.int64)
    A[0:2, 0:2] = [[3, -2], [-1, 4]]
    B[0:2, 0:2] = [[-5, 6], [7, -8]]

    expected = A @ B

    await feed_matrices(dut, A, B, N, DATA_WIDTH)

    C = read_result(dut, N, ACC_WIDTH)
    dut._log.info(f"A:\n{A}")
    dut._log.info(f"B:\n{B}")
    dut._log.info(f"Result:\n{C}")
    dut._log.info(f"Expected:\n{expected}")

    assert np.array_equal(C, expected), f"Negative values test FAILED!\nGot:\n{C}\nExpected:\n{expected}"
    dut._log.info("Negative values test PASSED!")


@cocotb.test()
async def test_back_to_back(dut):
    """Test two consecutive multiplications using clear between them."""
    N, DATA_WIDTH, ACC_WIDTH = get_params(dut)
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # First multiplication
    await clear_accumulators(dut)
    A1 = np.ones((N, N), dtype=np.int64) * 2
    B1 = np.ones((N, N), dtype=np.int64) * 3
    expected1 = A1 @ B1

    await feed_matrices(dut, A1, B1, N, DATA_WIDTH)
    C1 = read_result(dut, N, ACC_WIDTH)
    dut._log.info(f"First result:\n{C1}")
    assert np.array_equal(C1, expected1), f"Back-to-back test (1st) FAILED!\nGot:\n{C1}\nExpected:\n{expected1}"

    # Second multiplication (after clear)
    await clear_accumulators(dut)
    A2 = np.eye(N, dtype=np.int64) * 5
    B2 = np.array([[i + j for j in range(N)] for i in range(N)], dtype=np.int64)
    expected2 = A2 @ B2

    await feed_matrices(dut, A2, B2, N, DATA_WIDTH)
    C2 = read_result(dut, N, ACC_WIDTH)
    dut._log.info(f"Second result:\n{C2}")
    assert np.array_equal(C2, expected2), f"Back-to-back test (2nd) FAILED!\nGot:\n{C2}\nExpected:\n{expected2}"

    dut._log.info("Back-to-back test PASSED!")


@cocotb.test()
async def test_random_matrices(dut):
    """Test with random matrices, verified against numpy."""
    N, DATA_WIDTH, ACC_WIDTH = get_params(dut)
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    rng = np.random.default_rng(seed=42)
    max_val = (1 << (DATA_WIDTH - 1)) - 1

    for trial in range(3):
        await reset_dut(dut)
        await clear_accumulators(dut)

        A = rng.integers(-max_val, max_val + 1, size=(N, N)).astype(np.int64)
        B = rng.integers(-max_val, max_val + 1, size=(N, N)).astype(np.int64)
        expected = A @ B

        await feed_matrices(dut, A, B, N, DATA_WIDTH)

        C = read_result(dut, N, ACC_WIDTH)
        dut._log.info(f"Random trial {trial}: A=\n{A}\nB=\n{B}\nC=\n{C}\nExpected=\n{expected}")

        assert np.array_equal(C, expected), (
            f"Random trial {trial} FAILED!\nA:\n{A}\nB:\n{B}\nGot:\n{C}\nExpected:\n{expected}"
        )

    dut._log.info("All random matrix tests PASSED!")
