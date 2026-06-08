"""
tpu_driver.py — Low-level async driver for the tpu_top hardware interface.

Wraps all DUT signal interactions so the rest of the software never
touches raw Cocotb signals directly.
"""

import numpy as np
from cocotb.triggers import RisingEdge, Timer, ClockCycles

from tiling.skew import pack_signed, to_signed, compute_bram_slices


class TPUDriver:
    """Async driver that speaks to one tpu_top DUT instance."""

    def __init__(self, dut, N=4, DATA_WIDTH=8, ACC_WIDTH=32):
        self.dut = dut
        self.N = N
        self.DATA_WIDTH = DATA_WIDTH
        self.ACC_WIDTH = ACC_WIDTH

    def _pack_signed(self, values):
        """Pack a list of signed integers into a single wide unsigned value (LSB-first)."""
        return pack_signed(values, self.DATA_WIDTH)

    def _to_signed(self, val):
        """Reinterpret an unsigned integer as ACC_WIDTH-bit signed."""
        return to_signed(val, self.ACC_WIDTH)

    def _compute_bram_slices(self, A, B):
        """
        Pre-compute the diagonal-skewed BRAM words for one NxN tile pair.

        Delegates to tiling.skew so simulation and hardware stay byte-identical.
        Returns two lists of length 2*N-1, each a packed integer word.
        """
        return compute_bram_slices(A, B, self.N, self.DATA_WIDTH)

    async def reset(self):
        """Hard reset the DUT."""
        dut = self.dut
        dut.rst.value       = 1
        dut.start.value     = 0
        dut.wr_en_a.value   = 0
        dut.wr_en_b.value   = 0
        dut.rd_addr_c.value = 0
        await ClockCycles(dut.clk, 3)
        dut.rst.value = 0
        await RisingEdge(dut.clk)

    async def load_tile(self, A_tile, B_tile):
        """
        Pre-skew A_tile and B_tile and write the resulting words
        to BRAM_A and BRAM_B simultaneously (one write per clock).
        """
        dut = self.dut
        slices_a, slices_b = self._compute_bram_slices(A_tile, B_tile)
        for t, (wa, wb) in enumerate(zip(slices_a, slices_b)):
            dut.wr_en_a.value   = 1
            dut.wr_addr_a.value = t
            dut.wr_data_a.value = wa
            dut.wr_en_b.value   = 1
            dut.wr_addr_b.value = t
            dut.wr_data_b.value = wb
            await RisingEdge(dut.clk)
        dut.wr_en_a.value = 0
        dut.wr_en_b.value = 0
        await RisingEdge(dut.clk)

    async def run(self):
        """Pulse start and block until the done signal fires."""
        dut = self.dut
        dut.start.value = 1
        await RisingEdge(dut.clk)
        dut.start.value = 0
        for _ in range(500):
            await RisingEdge(dut.clk)
            if dut.done.value == 1:
                return
        raise TimeoutError("TPU did not assert done within 500 cycles")

    async def read_result(self):
        """
        Read all N*N results from BRAM_C via the async read port.
        Returns an (N, N) int64 numpy array.
        """
        dut = self.dut
        N = self.N
        C = np.zeros((N, N), dtype=np.int64)
        for i in range(N):
            for j in range(N):
                dut.rd_addr_c.value = i * N + j
                await Timer(1, units="ns")
                C[i][j] = self._to_signed(int(dut.rd_data_c.value))
        return C

    async def matmul_tile(self, A_tile, B_tile):
        """Run a complete single NxN tile multiply: load → run → read."""
        await self.load_tile(A_tile, B_tile)
        await self.run()
        return await self.read_result()
