# Unified Makefile for systolic-tpu
#
# Targets:
#   make test-sa      — unit tests for systolic_array only
#   make test-tpu     — integration tests for tpu_top
#   make test-tiling  — end-to-end tiling tests (arbitrary matrix sizes)
#   make test         — run all three suites in sequence
#   make clean        — remove all simulation artifacts
#
# Simulator: icarus (default). Override with SIM=verilator make test-sa
# Array size: N=4 (default). Override with N=8 make test-sa

SIM ?= icarus
N   ?= 4

# FPGA (Arty A7-35T) settings
VIVADO ?= vivado
PORT   ?= /dev/ttyUSB1

.PHONY: test-sa test-tpu test-tiling test clean build-fpga program-fpga flash-fpga fpga-selftest fpga-mnist fpga-draw

# Each suite uses a distinct SIM_BUILD so the shared tests/ dir doesn't reuse a
# stale sim binary built for a different TOPLEVEL (e.g. test-sa's systolic_array
# build leaking into test-tpu).
test-sa:
	$(MAKE) -C tests -f Makefile \
		SIM=$(SIM) SIM_BUILD=sim_build_sa \
		COMPILE_ARGS="-Psystolic_array.N=$(N) -Psystolic_array.DATA_WIDTH=8 -Psystolic_array.ACC_WIDTH=32"

test-tpu:
	$(MAKE) -C tests -f Makefile.tpu \
		SIM=$(SIM) SIM_BUILD=sim_build_tpu \
		COMPILE_ARGS="-Ptpu_top.N=$(N) -Ptpu_top.DATA_WIDTH=8 -Ptpu_top.ACC_WIDTH=32"

test-tiling:
	$(MAKE) -C tests -f Makefile.tiling \
		SIM=$(SIM) SIM_BUILD=sim_build_tiling \
		COMPILE_ARGS="-Ptpu_top.N=$(N) -Ptpu_top.DATA_WIDTH=8 -Ptpu_top.ACC_WIDTH=32"

test: test-sa test-tpu test-tiling

# ---- FPGA flow (Digilent Arty A7-35T, xc7a35ticsg324-1L) ----------------
# build-fpga      synthesize/implement uart_matmul -> build/uart_matmul.bit
# program-fpga    program the bitstream onto the board over JTAG
# fpga-selftest   drive random matmuls over UART and check against numpy
build-fpga:
	$(VIVADO) -mode batch -source scripts/build_arty.tcl

program-fpga:
	$(VIVADO) -mode batch -source scripts/program_arty.tcl

# Write the design to QSPI flash so it auto-boots on power-up (persistent).
flash-fpga:
	$(VIVADO) -mode batch -source scripts/flash_arty.tcl

fpga-selftest:
	python3 host/tpu_uart.py --self-test --port $(PORT)

# MNIST handwritten-digit recognition with inference running on the FPGA
fpga-mnist:
	python3 host/mnist_fpga.py --port $(PORT)

# Interactive GUI: draw a digit with the mouse, classified live on the FPGA
fpga-draw:
	python3 host/mnist_draw.py --port $(PORT)

clean:
	$(MAKE) -C tests -f Makefile        clean 2>/dev/null || true
	$(MAKE) -C tests -f Makefile.tpu    clean 2>/dev/null || true
	$(MAKE) -C tests -f Makefile.tiling clean 2>/dev/null || true
	rm -rf tests/sim_build_sa tests/sim_build_tpu tests/sim_build_tiling
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
