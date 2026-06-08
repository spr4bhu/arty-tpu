# flash_arty.tcl — Write uart_matmul to the Arty A7-35T's QSPI flash so the
# design auto-boots on power-up (survives power cycles, unlike volatile JTAG
# configuration).
#
# Two phases in one run:
#   1. From the routed checkpoint, regenerate a bitstream configured to boot
#      from SPI x4, then pack it into an .mcs image.
#   2. Program that .mcs into the on-board QSPI flash over JTAG.
#
# Flash part: Micron MT25QL128 (16 MB) — detected on this Arty A7-35T. Older
# board revisions ship a Spansion S25FL127S; if flashing reports a different
# detected part, set cfgpart to the matching entry from get_cfgmem_parts.
#
# Usage (from repo root):
#   vivado -mode batch -source scripts/flash_arty.tcl

set repo_root [file normalize [file dirname [info script]]/..]
set out_dir   $repo_root/build
set dcp       $out_dir/post_route.dcp
set spi_bit   $out_dir/uart_matmul_spi.bit
set mcs       $out_dir/uart_matmul.mcs
set cfgpart   mt25ql128-spi-x1_x2_x4

if {![file exists $dcp]} {
    puts "ERROR: $dcp not found. Run scripts/build_arty.tcl first."
    exit 1
}

# ---- Phase 1: SPI-boot bitstream + .mcs ---------------------------------
open_checkpoint $dcp
set_property CONFIG_MODE SPIx4                       [current_design]
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4         [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE 33          [current_design]
set_property BITSTREAM.CONFIG.SPI_FALL_EDGE YES      [current_design]
set_property BITSTREAM.GENERAL.COMPRESS TRUE         [current_design]
write_bitstream -force $spi_bit

write_cfgmem -force -format mcs -interface spix4 -size 16 \
    -loadbit "up 0x0 $spi_bit" -file $mcs
puts "==== Wrote $mcs ===="
close_design

# ---- Phase 2: program the QSPI flash ------------------------------------
open_hw_manager
connect_hw_server -url localhost:3121
open_hw_target
set dev [lindex [get_hw_devices xc7a35t*] 0]
current_hw_device $dev
refresh_hw_device -update_hw_probes false $dev

create_hw_cfgmem -hw_device $dev [lindex [get_cfgmem_parts $cfgpart] 0]
set cfgmem [get_property PROGRAM.HW_CFGMEM $dev]
set_property PROGRAM.FILES         [list $mcs] $cfgmem
set_property PROGRAM.ADDRESS_RANGE {use_file}  $cfgmem
set_property PROGRAM.BLANK_CHECK   0           $cfgmem
set_property PROGRAM.ERASE         1           $cfgmem
set_property PROGRAM.CFG_PROGRAM   1           $cfgmem
set_property PROGRAM.VERIFY        1           $cfgmem
set_property PROGRAM.CHECKSUM      0           $cfgmem

# Load the indirect-programming helper bitstream into the FPGA, then flash.
create_hw_bitstream -hw_device $dev [get_property PROGRAM.HW_CFGMEM_BITFILE $dev]
program_hw_devices $dev
program_hw_cfgmem -hw_cfgmem $cfgmem

puts "==== QSPI flash programmed. Press PROG (or power-cycle) to boot from flash. ===="

close_hw_target
disconnect_hw_server
close_hw_manager
exit 0
