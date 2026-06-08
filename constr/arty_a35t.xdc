# arty_a35t.xdc — Constraints for uart_matmul on the Digilent Arty A7-35T
# FPGA: xc7a35ticsg324-1L (Artix-7)
# Pin assignments from the Digilent Arty A7-35 master XDC.

# ---- Configuration bank voltage -----------------------------------------
set_property CFGBVS VCCO        [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]

# ---- 100 MHz system clock (E3) ------------------------------------------
set_property -dict { PACKAGE_PIN E3  IOSTANDARD LVCMOS33 } [get_ports clk]
create_clock -name sys_clk -period 10.000 [get_ports clk]

# ---- Reset: push button BTN0 (D9), active high --------------------------
set_property -dict { PACKAGE_PIN D9  IOSTANDARD LVCMOS33 } [get_ports btn_rst]

# ---- USB-UART bridge (FT2232H channel B) --------------------------------
# uart_txd_in  = data from PC into FPGA (FPGA input)  -> A9
# uart_rxd_out = data from FPGA to PC  (FPGA output)  -> D10
set_property -dict { PACKAGE_PIN A9  IOSTANDARD LVCMOS33 } [get_ports uart_rx_pin]
set_property -dict { PACKAGE_PIN D10 IOSTANDARD LVCMOS33 } [get_ports uart_tx_pin]

# ---- Status LEDs --------------------------------------------------------
# led[0] = LD4 (H5) busy, led[1] = LD5 (J5) done latched
set_property -dict { PACKAGE_PIN H5  IOSTANDARD LVCMOS33 } [get_ports {led[0]}]
set_property -dict { PACKAGE_PIN J5  IOSTANDARD LVCMOS33 } [get_ports {led[1]}]

# ---- Relax timing on async UART/button I/O ------------------------------
# (Single-bit signals are synchronized in RTL.)
set_false_path -from [get_ports uart_rx_pin]
set_false_path -from [get_ports btn_rst]
set_false_path -to   [get_ports uart_tx_pin]
set_false_path -to   [get_ports {led[*]}]
