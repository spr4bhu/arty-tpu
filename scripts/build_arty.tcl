# build_arty.tcl — Non-project Vivado flow for uart_matmul on the Arty A7-35T.
#
# Usage (from repo root):
#   vivado -mode batch -source scripts/build_arty.tcl
#
# Produces build/uart_matmul.bit and timing/utilization reports in build/.

set part      xc7a35ticsg324-1L
set top       uart_matmul
set repo_root [file normalize [file dirname [info script]]/..]
set out_dir   $repo_root/build

file mkdir $out_dir

# ---- Read sources -------------------------------------------------------
read_verilog [glob $repo_root/rtl/*.v]
read_xdc     $repo_root/constr/arty_a35t.xdc

# ---- Synthesis ----------------------------------------------------------
synth_design -top $top -part $part
write_checkpoint -force $out_dir/post_synth.dcp
report_utilization -file $out_dir/post_synth_util.rpt

# ---- Implementation -----------------------------------------------------
opt_design
place_design
phys_opt_design
route_design
write_checkpoint -force $out_dir/post_route.dcp

# ---- Reports ------------------------------------------------------------
report_timing_summary -file $out_dir/timing_summary.rpt
report_utilization    -file $out_dir/post_route_util.rpt

# Fail loudly if timing is not met.
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "==== Worst Negative Slack (setup): $wns ns ===="
if {$wns < 0} {
    puts "ERROR: Timing NOT met (WNS = $wns). Bitstream not written."
    exit 1
}

# ---- Bitstream ----------------------------------------------------------
write_bitstream -force $out_dir/uart_matmul.bit
puts "==== Bitstream written: $out_dir/uart_matmul.bit ===="
exit 0
