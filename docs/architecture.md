# Architecture Notes

## Systolic array dataflow

An N×N systolic array computes C = A × B in a **weight-stationary** style where
both operands stream through the array:

```
Cycle  a_in lanes (row 0..3)        b_in lanes (col 0..3)
------  ──────────────────────────   ───────────────────────────────
  0    [A00, 0,   0,   0  ]          [B00, 0,   0,   0  ]
  1    [A01, A10, 0,   0  ]          [B10, B01, 0,   0  ]
  2    [A02, A11, A20, 0  ]          [B20, B11, B02, 0  ]
  3    [A03, A12, A21, A30]          [B30, B21, B12, B03]
  4    [0,   A13, A22, A31]          [0,   B31, B22, B13]
  5    [0,   0,   A23, A32]          [0,   0,   B32, B23]
  6    [0,   0,   0,   A33]          [0,   0,   0,   B33]
```

After cycle 6 (= 2N−2), all PEs have received their full inner-product
contribution. N more zero-flush cycles drain the last pipeline stage so
PE[N-1][N-1] accumulates its final product.

## Controller timing (N=4)

```
Cycle   State     sa_clear  bram_rd_addr  sa_a_in / sa_b_in
------  ────────  ────────  ────────────  ──────────────────
  0     CLEAR     1         0             0
  1     STREAM    0         1             mem[0]
  2     STREAM    0         2             mem[1]
  3     STREAM    0         3             mem[2]
  4     STREAM    0         4             mem[3]
  5     STREAM    0         5             mem[4]
  6     STREAM    0         6             mem[5]
  7     STREAM    0         6 (hold)      mem[6]   ← last slice
  8     FLUSH     0         —             0
  9     FLUSH     0         —             0
 10     FLUSH     0         —             0
 11     FLUSH     0         —             0        ← PE[3][3] done
 12-27  WRITE     —         —             (write 16 words to BRAM_C)
 28     DONE      —         —             done=1
```

Total latency for one NxN tile: `1 + (2N-1) + N + N² + 1 = 3N + N² + 1` cycles.
For N=4: 3×4 + 16 + 1 = **29 cycles**.

## BRAM read pipeline offset

`bram.v` has an async (combinatorial) read port. The controller advances
`bram_a_rd_addr` one cycle *before* it latches `bram_a_rd_data` into `sa_a_in`.
This means:

- During ST_CLEAR: addr ← 0 (data will appear combinatorially)
- First STREAM cycle: latch data[0], advance addr to 1
- ...

This is why `STREAM_LEN = 2N-1` cycles feeds exactly `2N-1` slices
into the SA even though BRAM is read asynchronously.

## Tiling algorithm

For a C = A @ B matmul with A of shape (M, K) and B of shape (K, P):

```python
M_pad = ceil(M/N)*N;  K_pad = ceil(K/N)*N;  P_pad = ceil(P/N)*N
A_pad = pad(A, M_pad, K_pad)
B_pad = pad(B, K_pad, P_pad)

for ti in range(M_pad//N):
    for tj in range(P_pad//N):
        partial = zeros(N, N)
        for tk in range(K_pad//N):
            partial += TPU.matmul_tile(A_pad[ti,tk], B_pad[tk,tj])
        C_pad[ti,tj] = partial

return C_pad[:M, :P]
```

Partial sums accumulate in software (int64) because each TPU run clears
its own accumulators. The TPU handles overflow avoidance internally via
the 32-bit accumulator for 8-bit inputs (max product: 127×127×N = 64,516×N;
for N=16 that is 1,032,256, well within int32).

## FPGA synthesis notes

- The three memories are tiny (BRAM_A/B = 8×32 b, BRAM_C = 16×32 b at N=4), so
  `bram.v`'s async-read register file maps cleanly to **distributed LUT-RAM** —
  no need to swap in a vendor BRAM primitive or change the controller timing.
  (For much larger N you would switch to registered block RAM and add a read
  latency cycle in `tpu_controller.v`.)
- The systolic array `generate` block synthesizes into a regular array of DSP
  slices: the PE MAC `c + a_in*b_in` infers a DSP48.

## Running on hardware (Arty A7-35T)

`tpu_top` exposes wide parallel buses that can't reach pins, so `uart_matmul.v`
wraps it in a UART "matmul service":

```
PC (host/tpu_uart.py)  ──UART──►  uart_rx ─► loader FSM ─► tpu_top
                       ◄──UART──   uart_tx ◄─ unloader FSM ◄┘
```

- **FSM** (`uart_matmul.v`): `IDLE` (wait `0xA5`) → `RX` (load BRAM_A/B) →
  `START` → `WAIT` (done) → `TX` (stream result), plus a small power-on reset.
- **Host pre-skews** A/B with `tiling/skew.py` — the same helpers the Cocotb
  tests use — so on-chip results are byte-identical to simulation.
- **Wire protocol** (N=4): host→FPGA 57 bytes (`0xA5` + 7 A-slices + 7 B-slices,
  4 B each, little-endian); FPGA→host 64 bytes (16 result words, row-major).
- **Timing**: at N=4 the design closes 100 MHz comfortably (WNS ≈ +1.3 ns); the
  UART link, not the array, is the throughput bottleneck.

Flows: `make build-fpga` (bitstream) · `make program-fpga` (JTAG, volatile) ·
`make flash-fpga` (QSPI, persists across power cycles). See the README for the
full host/demo commands.
