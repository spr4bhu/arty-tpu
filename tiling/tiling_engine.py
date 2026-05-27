"""
tiling_engine.py — Tiles arbitrary-sized matrix multiplications over the NxN TPU.

Algorithm (C = A @ B, A is M×K, B is K×P):
  1. Pad A and B to the nearest multiple of N in every dimension.
  2. For every output tile position (ti, tj):
       partial = zeros(N, N)
       For every k-tile tk:
           partial += TPU.matmul_tile(A[ti,tk], B[tk,tj])
       C[ti, tj] = partial
  3. Trim padding from the result.

Partial sums are accumulated in software (numpy int64) because the TPU
controller always clears accumulators at the start of each run.
"""

import math
import numpy as np


def _ceil_div(a, b):
    return (a + b - 1) // b


class TilingEngine:
    """
    Breaks arbitrary M×K and K×P matrix multiplies into NxN tiles
    and drives the TPU hardware via a TPUDriver instance.
    """

    def __init__(self, driver):
        self.driver = driver
        self.N = driver.N

    async def matmul(self, A, B):
        """
        Compute C = A @ B for matrices of any size.

        A : (M, K)  numpy int64
        B : (K, P)  numpy int64
        returns C : (M, P)  numpy int64

        Internally pads to multiples of N, tiles the computation,
        and trims the result before returning.
        """
        N = self.N
        M, K  = A.shape
        K2, P = B.shape
        assert K == K2, f"Inner dimensions must match: A={A.shape}, B={B.shape}"

        # pad to multiples of N
        M_pad = _ceil_div(M, N) * N
        K_pad = _ceil_div(K, N) * N
        P_pad = _ceil_div(P, N) * N

        A_pad = np.zeros((M_pad, K_pad), dtype=np.int64)
        B_pad = np.zeros((K_pad, P_pad), dtype=np.int64)
        A_pad[:M, :K] = A
        B_pad[:K, :P] = B

        C_pad = np.zeros((M_pad, P_pad), dtype=np.int64)

        tiles_m = M_pad // N
        tiles_k = K_pad // N
        tiles_p = P_pad // N

        # iterate over output tile positions, accumulating k-tile partial products
        for ti in range(tiles_m):
            for tj in range(tiles_p):
                partial = np.zeros((N, N), dtype=np.int64)

                for tk in range(tiles_k):
                    A_tile = A_pad[ti*N:(ti+1)*N, tk*N:(tk+1)*N]
                    B_tile = B_pad[tk*N:(tk+1)*N, tj*N:(tj+1)*N]

                    tile_result = await self.driver.matmul_tile(A_tile, B_tile)
                    partial += tile_result

                C_pad[ti*N:(ti+1)*N, tj*N:(tj+1)*N] = partial

        return C_pad[:M, :P]

    def tpu_invocations(self, M, K, P):
        """Return the number of TPU tile runs a given matmul will require."""
        N = self.N
        return (_ceil_div(M, N) * _ceil_div(K, N) * _ceil_div(P, N))
