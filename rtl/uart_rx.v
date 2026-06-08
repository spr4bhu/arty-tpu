// uart_rx.v — 8N1 UART receiver
//
// Receives one byte at a time at BAUD on a CLK_FREQ clock.
// rx is the asynchronous serial input (double-flopped internally).
// rx_valid pulses high for exactly one clock when rx_data is fresh.

`timescale 1ns / 1ps

module uart_rx #(
    parameter CLK_FREQ = 100_000_000,
    parameter BAUD     = 115_200
)(
    input  wire       clk,
    input  wire       rst,
    input  wire       rx,           // async serial in
    output reg  [7:0] rx_data,
    output reg        rx_valid      // 1-cycle strobe
);

    localparam integer CLKS_PER_BIT = CLK_FREQ / BAUD;   // 868 @ 100MHz/115200
    localparam integer HALF_BIT     = CLKS_PER_BIT / 2;

    // Synchronize the async input to clk (avoid metastability)
    reg rx_meta, rx_sync;
    always @(posedge clk) begin
        rx_meta <= rx;
        rx_sync <= rx_meta;
    end

    localparam S_IDLE  = 2'd0;
    localparam S_START = 2'd1;
    localparam S_DATA  = 2'd2;
    localparam S_STOP  = 2'd3;

    reg [1:0]  state;
    reg [15:0] clk_cnt;
    reg [2:0]  bit_idx;
    reg [7:0]  shifter;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state    <= S_IDLE;
            clk_cnt  <= 0;
            bit_idx  <= 0;
            shifter  <= 0;
            rx_data  <= 0;
            rx_valid <= 1'b0;
        end else begin
            rx_valid <= 1'b0;   // default; pulsed in S_STOP

            case (state)
                S_IDLE: begin
                    clk_cnt <= 0;
                    bit_idx <= 0;
                    if (rx_sync == 1'b0)   // start bit detected
                        state <= S_START;
                end

                // Wait to the middle of the start bit; confirm still low
                S_START: begin
                    if (clk_cnt == HALF_BIT) begin
                        if (rx_sync == 1'b0) begin
                            clk_cnt <= 0;
                            state   <= S_DATA;
                        end else begin
                            state <= S_IDLE;   // false start
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end
                end

                // Sample each data bit at its center, LSB first
                S_DATA: begin
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt          <= 0;
                        shifter[bit_idx] <= rx_sync;
                        if (bit_idx == 3'd7) begin
                            bit_idx <= 0;
                            state   <= S_STOP;
                        end else begin
                            bit_idx <= bit_idx + 1'b1;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end
                end

                // One stop bit; emit byte at its center
                S_STOP: begin
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        rx_data  <= shifter;
                        rx_valid <= 1'b1;
                        clk_cnt  <= 0;
                        state    <= S_IDLE;
                    end else begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
