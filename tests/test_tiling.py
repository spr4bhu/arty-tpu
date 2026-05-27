"""
Cocotb testbench for the tiling engine.

Each test uses the full tpu_top DUT and the TilingEngine to multiply
matrices larger than the native 4x4 tile size, verifying results against
numpy for all shapes and sizes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import numpy as np

from tiling.tpu_driver import TPUDriver
from tiling.tiling_engine import TilingEngine

def get_params(dut):
    return int(dut.N.value), int(dut.DATA_WIDTH.value), int(dut.ACC_WIDTH.value)

def make_engine(dut):
    N, DATA_WIDTH, ACC_WIDTH = get_params(dut)
    driver = TPUDriver(dut, N=N, DATA_WIDTH=DATA_WIDTH, ACC_WIDTH=ACC_WIDTH)
    return TilingEngine(driver), driver

@cocotb.test()
async def test_single_tile_4x4(dut):
    """4×4 matrix — exactly one tile, one TPU invocation."""
    N, DATA_WIDTH, _ = get_params(dut)
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    engine, driver = make_engine(dut)
    await driver.reset()

    rng = np.random.default_rng(0)
    max_val = (1 << (DATA_WIDTH - 1)) - 1
    A = rng.integers(-max_val, max_val + 1, (4, 4)).astype(np.int64)
    B = rng.integers(-max_val, max_val + 1, (4, 4)).astype(np.int64)
    expected = A @ B

    C = await engine.matmul(A, B)
    assert np.array_equal(C, expected), f"FAILED\nExpected:\n{expected}\nGot:\n{C}"
    dut._log.info("test_single_tile_4x4 PASSED")

@cocotb.test()
async def test_8x8(dut):
    """8×8 matrix — 2×2 tiling, 8 TPU runs total."""
    N, DATA_WIDTH, _ = get_params(dut)
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    engine, driver = make_engine(dut)
    await driver.reset()

    rng = np.random.default_rng(1)
    max_val = (1 << (DATA_WIDTH - 1)) - 1
    A = rng.integers(-max_val, max_val + 1, (8, 8)).astype(np.int64)
    B = rng.integers(-max_val, max_val + 1, (8, 8)).astype(np.int64)
    expected = A @ B

    invocations = engine.tpu_invocations(8, 8, 8)
    dut._log.info(f"8×8 matmul: {invocations} TPU tile invocations")

    C = await engine.matmul(A, B)
    assert np.array_equal(C, expected), f"FAILED\nExpected:\n{expected}\nGot:\n{C}"
    dut._log.info("test_8x8 PASSED")

@cocotb.test()
async def test_12x12(dut):
    """12×12 matrix — 3×3 tiling, 27 TPU runs total."""
    N, DATA_WIDTH, _ = get_params(dut)
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    engine, driver = make_engine(dut)
    await driver.reset()

    rng = np.random.default_rng(2)
    max_val = (1 << (DATA_WIDTH - 1)) - 1
    A = rng.integers(-max_val, max_val + 1, (12, 12)).astype(np.int64)
    B = rng.integers(-max_val, max_val + 1, (12, 12)).astype(np.int64)
    expected = A @ B

    invocations = engine.tpu_invocations(12, 12, 12)
    dut._log.info(f"12×12 matmul: {invocations} TPU tile invocations")

    C = await engine.matmul(A, B)
    assert np.array_equal(C, expected), f"FAILED\nExpected:\n{expected}\nGot:\n{C}"
    dut._log.info("test_12x12 PASSED")

@cocotb.test()
async def test_non_multiple_6x6(dut):
    """6×6 matrix — not a multiple of 4, padded to 8×8 internally."""
    N, DATA_WIDTH, _ = get_params(dut)
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    engine, driver = make_engine(dut)
    await driver.reset()

    rng = np.random.default_rng(3)
    max_val = (1 << (DATA_WIDTH - 1)) - 1
    A = rng.integers(-max_val, max_val + 1, (6, 6)).astype(np.int64)
    B = rng.integers(-max_val, max_val + 1, (6, 6)).astype(np.int64)
    expected = A @ B

    C = await engine.matmul(A, B)
    assert C.shape == (6, 6), f"Shape mismatch: got {C.shape}"
    assert np.array_equal(C, expected), f"FAILED\nExpected:\n{expected}\nGot:\n{C}"
    dut._log.info("test_non_multiple_6x6 PASSED")

@cocotb.test()
async def test_non_multiple_5x7(dut):
    """5×7 matrix — both dimensions non-multiples of 4."""
    N, DATA_WIDTH, _ = get_params(dut)
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    engine, driver = make_engine(dut)
    await driver.reset()

    rng = np.random.default_rng(4)
    max_val = (1 << (DATA_WIDTH - 1)) - 1
    A = rng.integers(-max_val, max_val + 1, (5, 7)).astype(np.int64)
    B = rng.integers(-max_val, max_val + 1, (7, 5)).astype(np.int64)
    expected = A @ B

    C = await engine.matmul(A, B)
    assert C.shape == (5, 5), f"Shape mismatch: got {C.shape}"
    assert np.array_equal(C, expected), f"FAILED\nExpected:\n{expected}\nGot:\n{C}"
    dut._log.info("test_non_multiple_5x7 PASSED")

@cocotb.test()
async def test_rectangular_tall_wide(dut):
    """(8×4) × (4×8) → 8×8 result. Tests non-square output tiles."""
    N, DATA_WIDTH, _ = get_params(dut)
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    engine, driver = make_engine(dut)
    await driver.reset()

    rng = np.random.default_rng(5)
    max_val = (1 << (DATA_WIDTH - 1)) - 1
    A = rng.integers(-max_val, max_val + 1, (8, 4)).astype(np.int64)
    B = rng.integers(-max_val, max_val + 1, (4, 8)).astype(np.int64)
    expected = A @ B

    invocations = engine.tpu_invocations(8, 4, 8)
    dut._log.info(f"(8×4)×(4×8): {invocations} TPU tile invocations")

    C = await engine.matmul(A, B)
    assert C.shape == (8, 8), f"Shape mismatch: got {C.shape}"
    assert np.array_equal(C, expected), f"FAILED\nExpected:\n{expected}\nGot:\n{C}"
    dut._log.info("test_rectangular_tall_wide PASSED")

@cocotb.test()
async def test_rectangular_wide_tall(dut):
    """(4×12) × (12×4) → 4×4 result. Inner dimension requires 3 k-tiles."""
    N, DATA_WIDTH, _ = get_params(dut)
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    engine, driver = make_engine(dut)
    await driver.reset()

    rng = np.random.default_rng(6)
    max_val = (1 << (DATA_WIDTH - 1)) - 1
    A = rng.integers(-max_val, max_val + 1, (4, 12)).astype(np.int64)
    B = rng.integers(-max_val, max_val + 1, (12, 4)).astype(np.int64)
    expected = A @ B

    invocations = engine.tpu_invocations(4, 12, 4)
    dut._log.info(f"(4×12)×(12×4): {invocations} TPU tile invocations")

    C = await engine.matmul(A, B)
    assert C.shape == (4, 4), f"Shape mismatch: got {C.shape}"
    assert np.array_equal(C, expected), f"FAILED\nExpected:\n{expected}\nGot:\n{C}"
    dut._log.info("test_rectangular_wide_tall PASSED")

@cocotb.test()
async def test_stress_16x16(dut):
    """16×16 random signed matrices — 4×4 tiling, 64 TPU invocations."""
    N, DATA_WIDTH, _ = get_params(dut)
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    engine, driver = make_engine(dut)
    await driver.reset()

    rng = np.random.default_rng(42)
    max_val = (1 << (DATA_WIDTH - 1)) - 1
    A = rng.integers(-max_val, max_val + 1, (16, 16)).astype(np.int64)
    B = rng.integers(-max_val, max_val + 1, (16, 16)).astype(np.int64)
    expected = A @ B

    invocations = engine.tpu_invocations(16, 16, 16)
    dut._log.info(f"16×16 matmul: {invocations} TPU tile invocations")

    C = await engine.matmul(A, B)
    assert np.array_equal(C, expected), f"FAILED\nExpected:\n{expected}\nGot:\n{C}"
    dut._log.info("test_stress_16x16 PASSED")
