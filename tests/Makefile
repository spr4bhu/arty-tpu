TOPLEVEL_LANG = verilog

# Verilog sources
VERILOG_SOURCES = $(PWD)/../rtl/pe.v $(PWD)/../rtl/systolic_array.v

# Top-level module
TOPLEVEL = systolic_array

# Cocotb test module
MODULE = test_systolic_array

# Simulator
SIM ?= icarus

# Parameters (change N here to test different array sizes)
COMPILE_ARGS += -Psystolic_array.N=4
COMPILE_ARGS += -Psystolic_array.DATA_WIDTH=8
COMPILE_ARGS += -Psystolic_array.ACC_WIDTH=32

include $(shell cocotb-config --makefiles)/Makefile.sim
