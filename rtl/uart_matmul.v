// uart_matmul.v — Physical top-level for the Arty A7-35T (Artix-7 xc7a35t)
//
// Wraps tpu_top in a UART "matmul service": a Python host streams pre-skewed
// A/B slices in, the FSM loads them into the core's BRAMs, runs one tile, and
// streams the N*N result words back.  All host I/O is over the on-board
// USB-UART bridge; no parallel pins are exposed.
//
// Wire protocol (little-endian, see host/tpu_uart.py):
//   Host -> FPGA : 0xA5 sync, then (2N-1) A slices, then (2N-1) B slices,
//                  each slice = N*DATA_WIDTH/8 bytes, byte0 = lane0 = bits[7:0].
//   FPGA -> Host : N*N result words, each ACC_WIDTH/8 bytes, little-endian,
//                  row-major (C[0][0], C[0][1], ... C[N-1][N-1]).
//
// For the default N=4/8b/32b build: 1 + 28 + 28 = 57 bytes in, 64 bytes out.

`timescale 1ns / 1ps

module uart_matmul #(
    parameter N          = 4,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32,
    parameter CLK_FREQ   = 100_000_000,
    parameter BAUD       = 115_200
)(
    input  wire       clk,          // 100 MHz (Arty A7 E3)
    input  wire       btn_rst,      // active-high reset button (BTN0)
    input  wire       uart_rx_pin,  // host -> FPGA serial
    output wire       uart_tx_pin,  // FPGA -> host serial
    output wire [1:0] led           // [0]=busy, [1]=done latched
);

    // ---- Derived sizes ---------------------------------------------------
    localparam AWORD_W = N * DATA_WIDTH;       // packed slice width (bits)
    localparam ABYTES  = AWORD_W / 8;          // bytes per A/B slice
    localparam NSLICES = 2 * N - 1;            // STREAM_LEN
    localparam A_BYTES = NSLICES * ABYTES;     // bytes for all A slices
    localparam PAYLOAD = 2 * A_BYTES;          // A + B input bytes
    localparam CBYTES  = ACC_WIDTH / 8;        // bytes per result word
    localparam C_WORDS = N * N;                // result words
    localparam TX_BYTES = C_WORDS * CBYTES;    // result bytes out

    // ---- Power-on reset (the Arty A7 has no POR pin) ---------------------
    // por_cnt is free-running with an init value; holds reset high ~16 cycles
    // after configuration, then OR with the user button.
    reg [4:0] por_cnt = 5'd0;
    always @(posedge clk) begin
        if (!por_cnt[4])
            por_cnt <= por_cnt + 1'b1;
    end
    wire rst = (~por_cnt[4]) | btn_rst;

    // ---- UART RX / TX ----------------------------------------------------
    wire [7:0] rx_data;
    wire       rx_valid;
    reg  [7:0] tx_data;
    reg        tx_start;
    wire       tx_busy;

    uart_rx #(.CLK_FREQ(CLK_FREQ), .BAUD(BAUD)) u_rx (
        .clk      (clk),
        .rst      (rst),
        .rx       (uart_rx_pin),
        .rx_data  (rx_data),
        .rx_valid (rx_valid)
    );

    uart_tx #(.CLK_FREQ(CLK_FREQ), .BAUD(BAUD)) u_tx (
        .clk      (clk),
        .rst      (rst),
        .tx_start (tx_start),
        .tx_data  (tx_data),
        .tx       (uart_tx_pin),
        .tx_busy  (tx_busy)
    );

    // ---- TPU core --------------------------------------------------------
    reg                        wr_en_a, wr_en_b;
    reg  [$clog2(2*N)-1:0]      wr_addr_a, wr_addr_b;
    reg  [AWORD_W-1:0]          wr_data_a, wr_data_b;
    reg                        core_start;
    wire                       core_busy, core_done;
    wire [ACC_WIDTH-1:0]       rd_data_c;
    wire [$clog2(N*N)-1:0]     rd_addr_c;

    tpu_top #(
        .N(N), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)
    ) core (
        .clk       (clk),
        .rst       (rst),
        .wr_en_a   (wr_en_a),
        .wr_addr_a (wr_addr_a),
        .wr_data_a (wr_data_a),
        .wr_en_b   (wr_en_b),
        .wr_addr_b (wr_addr_b),
        .wr_data_b (wr_data_b),
        .rd_addr_c (rd_addr_c),
        .rd_data_c (rd_data_c),
        .start     (core_start),
        .busy      (core_busy),
        .done      (core_done)
    );

    // ---- Loader / unloader FSM ------------------------------------------
    localparam S_IDLE        = 3'd0;
    localparam S_RX          = 3'd1;
    localparam S_START       = 3'd2;
    localparam S_WAIT        = 3'd3;
    localparam S_TX_SET      = 3'd4;
    localparam S_TX_WAITBUSY = 3'd5;
    localparam S_TX_WAITDONE = 3'd6;

    reg [2:0]                    state;
    reg [$clog2(PAYLOAD+1)-1:0]  byte_cnt;   // input byte index
    reg [$clog2(TX_BYTES+1)-1:0] out_cnt;    // output byte index
    reg [AWORD_W-1:0]            word_asm;   // byte assembler
    reg                          done_latch;

    // Byte being assembled, with the freshly received byte shifted into the
    // top so that the first byte of a word lands in bits [7:0] (lane 0 / LSB).
    wire [AWORD_W-1:0] full_word = {rx_data, word_asm[AWORD_W-1:8]};
    wire               word_done = ((byte_cnt % ABYTES) == (ABYTES - 1));

    // Combinational result read: out_cnt selects word and byte.
    assign rd_addr_c = out_cnt / CBYTES;
    wire [7:0] c_byte = rd_data_c[(out_cnt % CBYTES) * 8 +: 8];

    assign led[0] = core_busy;
    assign led[1] = done_latch;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state      <= S_IDLE;
            byte_cnt   <= 0;
            out_cnt    <= 0;
            word_asm   <= 0;
            wr_en_a    <= 1'b0;
            wr_en_b    <= 1'b0;
            wr_addr_a  <= 0;
            wr_addr_b  <= 0;
            wr_data_a  <= 0;
            wr_data_b  <= 0;
            core_start <= 1'b0;
            tx_start   <= 1'b0;
            tx_data    <= 8'd0;
            done_latch <= 1'b0;
        end else begin
            // Default: de-assert single-cycle pulses.
            wr_en_a    <= 1'b0;
            wr_en_b    <= 1'b0;
            core_start <= 1'b0;
            tx_start   <= 1'b0;

            case (state)

                // Wait for the 0xA5 sync byte.
                S_IDLE: begin
                    byte_cnt <= 0;
                    if (rx_valid && rx_data == 8'hA5) begin
                        done_latch <= 1'b0;
                        state      <= S_RX;
                    end
                end

                // Collect PAYLOAD bytes; assemble each ABYTES into one slice
                // word and write it to BRAM_A (first half) or BRAM_B (second).
                S_RX: begin
                    if (rx_valid) begin
                        word_asm <= full_word;
                        if (word_done) begin
                            if (byte_cnt < A_BYTES) begin
                                wr_en_a   <= 1'b1;
                                wr_addr_a <= byte_cnt / ABYTES;
                                wr_data_a <= full_word;
                            end else begin
                                wr_en_b   <= 1'b1;
                                wr_addr_b <= (byte_cnt - A_BYTES) / ABYTES;
                                wr_data_b <= full_word;
                            end
                        end

                        if (byte_cnt == PAYLOAD - 1)
                            state <= S_START;
                        else
                            byte_cnt <= byte_cnt + 1'b1;
                    end
                end

                // Kick off one tile computation.
                S_START: begin
                    core_start <= 1'b1;
                    out_cnt    <= 0;
                    state      <= S_WAIT;
                end

                // Wait for the core to finish.
                S_WAIT: begin
                    if (core_done) begin
                        done_latch <= 1'b1;
                        state      <= S_TX_SET;
                    end
                end

                // Present the next result byte and start its UART transmission.
                S_TX_SET: begin
                    tx_data  <= c_byte;
                    tx_start <= 1'b1;
                    state    <= S_TX_WAITBUSY;
                end

                // Wait until the transmitter has accepted the byte...
                S_TX_WAITBUSY: begin
                    if (tx_busy)
                        state <= S_TX_WAITDONE;
                end

                // ...then until it has finished, before advancing.
                S_TX_WAITDONE: begin
                    if (!tx_busy) begin
                        if (out_cnt == TX_BYTES - 1)
                            state <= S_IDLE;
                        else begin
                            out_cnt <= out_cnt + 1'b1;
                            state   <= S_TX_SET;
                        end
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
