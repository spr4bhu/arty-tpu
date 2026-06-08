# arty-tpu

A weight-stationary systolic-array TPU running on a Digilent Arty A7-35T FPGA. The N×N array computes signed int8 matrix multiplications in `2N−1` streaming cycles. A Python host tiles arbitrary-size matmuls over the fixed hardware tile via UART.

---

## Simulation

```bash
pip install -r requirements.txt   # numpy, cocotb, pyserial
make test                         # run all three test suites
make test-sa                      # systolic array unit tests
make test-tpu                     # tpu_top integration tests
make test-tiling                  # arbitrary-size tiling tests
```

---

## FPGA (Arty A7-35T)

| Item      | Value |
|-----------|-------|
| Board     | Digilent Arty A7-35T |
| FPGA      | Artix-7 `xc7a35ticsg324-1L` |
| Toolchain | Vivado 2024.2 |
| Host link | USB-UART 115200 8N1, `/dev/ttyUSB1` |

```bash
make build-fpga      # synth + impl → build/uart_matmul.bit
make program-fpga    # load over JTAG (volatile)
make flash-fpga      # write to QSPI flash (persists across power cycles)
make fpga-selftest   # compare FPGA matmuls against numpy
make fpga-mnist      # MNIST digit inference on the FPGA (ASCII output)
make fpga-draw       # draw a digit with the mouse, classified live on FPGA
```

Override port or Vivado path if needed:

```bash
PORT=/dev/ttyUSB0 make fpga-selftest
VIVADO=/opt/Xilinx/Vivado/2024.2/bin/vivado make build-fpga
```

### UART protocol (N=4)

| Direction   | Bytes | Layout |
|-------------|-------|--------|
| Host → FPGA | 57    | `0xA5` sync + 7 A-slices × 4 B + 7 B-slices × 4 B (little-endian, lane0 = bits[7:0]) |
| FPGA → Host | 64    | 16 result words × 4 B, row-major `C[0][0]..C[3][3]`, signed |

---

## Repository layout

```
arty-tpu/
├── rtl/
│   ├── pe.v               Multiply-accumulate processing element
│   ├── systolic_array.v   N×N PE grid
│   ├── bram.v             Register file (sync write, async read)
│   ├── tpu_controller.v   FSM: IDLE→CLEAR→STREAM→FLUSH→WRITE→DONE
│   ├── tpu_top.v          Core: BRAM + array + controller
│   ├── uart_rx.v          8N1 UART receiver
│   ├── uart_tx.v          8N1 UART transmitter
│   └── uart_matmul.v      FPGA top — UART shell wrapping tpu_top
├── tiling/
│   ├── skew.py            Skew/pack helpers (shared by sim and hardware)
│   ├── tpu_driver.py      Cocotb async driver
│   └── tiling_engine.py   Software tiler for arbitrary-shape matmuls
├── host/
│   ├── tpu_uart.py        Serial driver + tiling + numpy self-test
│   ├── mnist_fpga.py      MNIST inference on the FPGA
│   └── mnist_draw.py      Draw-a-digit GUI, classified live on the FPGA
├── constr/
│   └── arty_a35t.xdc      Pin constraints
├── scripts/
│   ├── build_arty.tcl     Vivado build
│   ├── program_arty.tcl   JTAG program
│   └── flash_arty.tcl     QSPI flash
├── tests/
│   ├── test_systolic_array.py
│   ├── test_tpu.py
│   └── test_tiling.py
└── docs/
    └── architecture.md
```

---

## Parameters

| Parameter    | Default | Description |
|--------------|---------|-------------|
| `N`          | 4       | Array dimension (N×N PEs) |
| `DATA_WIDTH` | 8       | Input width in bits (signed) |
| `ACC_WIDTH`  | 32      | Accumulator width in bits |
