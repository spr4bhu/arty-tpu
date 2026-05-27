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

.PHONY: test-sa test-tpu test-tiling test clean

test-sa:
	$(MAKE) -C tests -f Makefile \
		SIM=$(SIM) \
		COMPILE_ARGS="-Psystolic_array.N=$(N) -Psystolic_array.DATA_WIDTH=8 -Psystolic_array.ACC_WIDTH=32"

test-tpu:
	$(MAKE) -C tests -f Makefile.tpu \
		SIM=$(SIM) \
		COMPILE_ARGS="-Ptpu_top.N=$(N) -Ptpu_top.DATA_WIDTH=8 -Ptpu_top.ACC_WIDTH=32"

test-tiling:
	$(MAKE) -C tests -f Makefile.tiling \
		SIM=$(SIM) \
		COMPILE_ARGS="-Ptpu_top.N=$(N) -Ptpu_top.DATA_WIDTH=8 -Ptpu_top.ACC_WIDTH=32"

test: test-sa test-tpu test-tiling

clean:
	$(MAKE) -C tests -f Makefile        clean 2>/dev/null || true
	$(MAKE) -C tests -f Makefile.tpu    clean 2>/dev/null || true
	$(MAKE) -C tests -f Makefile.tiling clean 2>/dev/null || true
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
