# systolic-tpu

A parameterized, simulation-ready **Tensor Processing Unit** built around an NxN weight-stationary systolic array. Written in synthesizable Verilog with a Python/Cocotb testbench stack and a software tiling engine that tiles arbitrary-size matrix multiplications over the fixed hardware tile.

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/<your-username>/systolic-tpu.git
cd systolic-tpu

# 2. Install Python dependencies (requires Icarus Verilog installed separately)
pip install -r requirements.txt

# 3. Run all tests
make test
```

For individual suites:

```bash
make test-sa       # systolic array unit tests
make test-tpu      # full tpu_top integration tests
make test-tiling   # arbitrary-size matrix tiling tests
```

---

## What it does

The hardware computes **C = A × B** for signed 8-bit integer matrices in `2N − 1` streaming cycles, using diagonal (wavefront) input skewing so every Processing Element stays busy every cycle. A software `TilingEngine` wraps the hardware to support matrices of any shape by padding, tiling, and accumulating partial products.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        tpu_top                           │
│                                                          │
│  ┌─────────┐   a_in    ┌──────────────────┐             │
│  │ BRAM_A  │──────────►│                  │             │
│  └─────────┘           │  systolic_array  │ c_out       │
│                        │    (N × N PEs)   │────────────►│ BRAM_C
│  ┌─────────┐   b_in    │                  │             │
│  │ BRAM_B  │──────────►│  PE  PE  PE  PE  │             │
│  └─────────┘           │  PE  PE  PE  PE  │             │
│                        │  PE  PE  PE  PE  │             │
│  ┌───────────────┐     │  PE  PE  PE  PE  │             │
│  │ tpu_controller│────►└──────────────────┘             │
│  │   (FSM)       │                                      │
│  └───────────────┘                                      │
└──────────────────────────────────────────────────────────┘

Processing Element (PE):
   a_in ──►[ c += a × b ]──► a_out   (horizontal pass-through)
   b_in ──►[             ]──► b_out   (vertical pass-through)
              │
              ▼
             c  (accumulated into BRAM_C after computation)
```

### Controller FSM

```
IDLE ──start──► CLEAR ──► STREAM (2N-1 cycles) ──► FLUSH (N cycles)
                                                         │
                    IDLE ◄── DONE ◄── WRITE (N² cycles) ◄┘
```

| State  | Action |
|--------|--------|
| IDLE   | Wait for `start` pulse from host |
| CLEAR  | Assert `sa_clear` to zero all PE accumulators |
| STREAM | Feed `2N−1` pre-skewed diagonal slices from BRAM_A/B into the array |
| FLUSH  | Feed N zero cycles to drain the pipeline |
| WRITE  | Write all N² accumulator values to BRAM_C, row-major |
| DONE   | Pulse `done` for one cycle, return to IDLE |

---

## Repository layout

```
systolic-tpu/
├── rtl/
│   ├── pe.v               Processing Element — multiply-accumulate + pass-through
│   ├── systolic_array.v   Parameterized N×N PE grid with generate blocks
│   ├── bram.v             Synchronous write / async read register file
│   ├── tpu_controller.v   FSM: IDLE→CLEAR→STREAM→FLUSH→WRITE→DONE
│   └── tpu_top.v          Top-level: ties BRAM, systolic array, controller together
│
├── tiling/
│   ├── tpu_driver.py      Async Cocotb driver — all DUT signal access lives here
│   └── tiling_engine.py   Software tiler for arbitrary M×K × K×P matmuls
│
├── tests/
│   ├── test_systolic_array.py   Unit tests for systolic_array (6 tests)
│   ├── test_tpu.py              Integration tests for tpu_top (5 tests)
│   ├── test_tiling.py           End-to-end tiling tests, up to 16×16 (8 tests)
│   ├── Makefile                 Runs test_systolic_array against systolic_array
│   ├── Makefile.tpu             Runs test_tpu against tpu_top
│   └── Makefile.tiling          Runs test_tiling against tpu_top
│
└── docs/
    └── architecture.md    Detailed design notes and timing diagrams
```

---

## Parameters

All modules are parameterized via Verilog parameters:

| Parameter    | Default | Description |
|--------------|---------|-------------|
| `N`          | 4       | Array dimension (N×N PEs, N-wide data lanes) |
| `DATA_WIDTH` | 8       | Input element width in bits (signed) |
| `ACC_WIDTH`  | 32      | Accumulator width in bits (signed) |

Change them in the Makefile `COMPILE_ARGS` lines:

```makefile
COMPILE_ARGS += -Ptpu_top.N=8
COMPILE_ARGS += -Ptpu_top.DATA_WIDTH=8
COMPILE_ARGS += -Ptpu_top.ACC_WIDTH=32
```

---

## Dependencies

| Tool | Purpose |
|------|---------|
| [Icarus Verilog](https://github.com/steveicarus/iverilog) ≥ 12 | Verilog simulation |
| [Cocotb](https://www.cocotb.org/) ≥ 1.8 | Python testbench framework |
| [NumPy](https://numpy.org/) | Reference matrix arithmetic in tests |
| Python ≥ 3.9 | Test runner language |

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Running tests

All targets are driven from the repo root via the unified `Makefile`.

```bash
make test-sa       # unit tests — systolic_array only
make test-tpu      # integration tests — full tpu_top
make test-tiling   # end-to-end — arbitrary matrix sizes via tiling engine
make test          # run all three suites in sequence
make clean         # remove sim_build/, __pycache__, .pyc files
```

Override the simulator or array size with variables:

```bash
SIM=verilator make test-sa
N=8 make test
```

---

## Test coverage

| Test file | Tests | What is covered |
|-----------|-------|-----------------|
| `test_systolic_array.py` | 6 | Identity multiply, 2×2 embedded in 4×4, full 4×4, negative values, back-to-back (clear), 3× random |
| `test_tpu.py`            | 5 | Identity, full 4×4, negative values, back-to-back (no reset), 5× random |
| `test_tiling.py`         | 8 | 4×4 (1 tile), 8×8 (8 tiles), 12×12 (27 tiles), 6×6 non-multiple, 5×7 non-multiple, tall×wide, wide×tall, 16×16 stress (64 tiles) |

---

## Host interface (tpu_top ports)

```
clk, rst                          — clock and active-high reset

wr_en_a, wr_addr_a, wr_data_a    — write pre-skewed A slices to BRAM_A
wr_en_b, wr_addr_b, wr_data_b    — write pre-skewed B slices to BRAM_B

rd_addr_c, rd_data_c              — async read result from BRAM_C

start                             — pulse high for 1 cycle to begin computation
busy                              — high while computation is in progress
done                              — pulses high for 1 cycle when result is ready
```

### Usage sequence

```
1. Write 2N−1 pre-skewed diagonal words to BRAM_A and BRAM_B
2. Pulse start for one clock cycle
3. Wait until done pulses (or poll busy == 0)
4. Read N×N words from BRAM_C at addresses 0 .. N²−1 (row-major)
```

The `TPUDriver` Python class handles steps 1–4 automatically.

---

## Input skewing

The systolic array expects inputs to arrive in a diagonal wavefront pattern. For an NxN multiply at timestep `t` (0-indexed):

- **BRAM_A lane `i`** carries `A[i][t−i]` if `0 ≤ t−i < N`, else `0`
- **BRAM_B lane `j`** carries `B[t−j][j]` if `0 ≤ t−j < N`, else `0`

`TPUDriver._compute_bram_slices()` pre-computes and packs these words before loading BRAM.

---

## Design notes

- **Weight-stationary dataflow** — in this implementation both A and B stream through; accumulators stay resident in each PE.
- **No external skew registers** — skewing is pre-computed by the host and stored in BRAM, keeping the RTL simple.
- **BRAM is async-read** — the controller advances the read address one cycle ahead so data is available on the combinatorial output by the time the SA clock edge fires.
- The BRAM model in `bram.v` is intentionally simple (register file). For FPGA targets, replace with vendor BRAM primitives and adjust the read-latency timing in `tpu_controller.v` accordingly.
