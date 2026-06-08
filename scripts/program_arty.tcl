# program_arty.tcl — Program build/uart_matmul.bit onto the Arty A7-35T over JTAG.
#
# Requires a running hw_server (Vivado starts one automatically via
# connect_hw_server if hw_server is on PATH).
#
# Usage (from repo root):
#   vivado -mode batch -source scripts/program_arty.tcl

set repo_root [file normalize [file dirname [info script]]/..]
set bit       $repo_root/build/uart_matmul.bit

if {![file exists $bit]} {
    puts "ERROR: $bit not found. Run scripts/build_arty.tcl first."
    exit 1
}

open_hw_manager
connect_hw_server -url localhost:3121
open_hw_target

# Target the Artix-7 on the board (xc7a35t).
set dev [lindex [get_hw_devices xc7a35t*] 0]
current_hw_device $dev
refresh_hw_device -update_hw_probes false $dev

set_property PROGRAM.FILE $bit $dev
program_hw_devices $dev
refresh_hw_device $dev

puts "==== Programmed $bit onto $dev ===="

close_hw_target
disconnect_hw_server
close_hw_manager
exit 0
