// tpu_controller.v — FSM Controller for the TPU
//
// Orchestrates one matrix multiplication tile:
//   IDLE → CLEAR → STREAM → FLUSH → WRITE → DONE → IDLE
//
// The host pre-computes diagonal-skewed input sequences and writes them to
// BRAM_A and BRAM_B before asserting 'start'.  After 'done' pulses, results
// are available in BRAM_C.
//
// Timing (N=4 example):
//   CLEAR  : 1 cycle  — sa_clear=1, pre-load BRAM addr=0
//   STREAM : 7 cycles — feeds mem[0..6] into SA (1-cycle pipeline offset means
//                        SA sees mem[0] in STREAM cycle 1; last element mem[6]
//                        is seen by the SA in the first FLUSH cycle)
//   FLUSH  : 4 cycles — zeros fed in; pipeline drains; all PE accumulators final
//   WRITE  : 16 cycles — c_out written word-by-word to BRAM_C
//   DONE   : 1 cycle  — done=1 pulse

`timescale 1ns / 1ps

module tpu_controller #(
    parameter N          = 4,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire clk,
    input  wire rst,

    // Host control / status
    input  wire start,
    output reg  busy,
    output reg  done,

    // Systolic array interface
    output reg                       sa_clear,
    output reg  [N*DATA_WIDTH-1:0]   sa_a_in,
    output reg  [N*DATA_WIDTH-1:0]   sa_b_in,
    input  wire [N*N*ACC_WIDTH-1:0]  sa_c_out,

    // BRAM_A read (asynchronous: addr registered, data combinatorial)
    output reg  [$clog2(2*N)-1:0]    bram_a_rd_addr,
    input  wire [N*DATA_WIDTH-1:0]   bram_a_rd_data,

    // BRAM_B read (asynchronous)
    output reg  [$clog2(2*N)-1:0]    bram_b_rd_addr,
    input  wire [N*DATA_WIDTH-1:0]   bram_b_rd_data,

    // BRAM_C write
    output reg                       bram_c_wr_en,
    output reg  [$clog2(N*N)-1:0]    bram_c_wr_addr,
    output reg  [ACC_WIDTH-1:0]      bram_c_wr_data
);

    // FSM state encoding
    localparam ST_IDLE   = 3'd0;
    localparam ST_CLEAR  = 3'd1;
    localparam ST_STREAM = 3'd2;
    localparam ST_FLUSH  = 3'd3;
    localparam ST_WRITE  = 3'd4;
    localparam ST_DONE   = 3'd5;

    // Timing constants for one NxN tile
    localparam STREAM_LEN = 2 * N - 1;   // diagonal slices: 7 for N=4
    localparam FLUSH_LEN  = N;            // pipeline drain cycles: 4 for N=4
    localparam C_LEN      = N * N;        // result elements: 16 for N=4

    reg [2:0]                 state;
    reg [$clog2(2*N)-1:0]     stream_cnt;  // counts 0 .. STREAM_LEN-1
    reg [$clog2(N)-1:0]       flush_cnt;   // counts 0 .. FLUSH_LEN-1
    reg [$clog2(N*N)-1:0]     write_cnt;   // counts 0 .. C_LEN-1

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state          <= ST_IDLE;
            busy           <= 1'b0;
            done           <= 1'b0;
            sa_clear       <= 1'b0;
            sa_a_in        <= 0;
            sa_b_in        <= 0;
            bram_a_rd_addr <= 0;
            bram_b_rd_addr <= 0;
            bram_c_wr_en   <= 1'b0;
            bram_c_wr_addr <= 0;
            bram_c_wr_data <= 0;
            stream_cnt     <= 0;
            flush_cnt      <= 0;
            write_cnt      <= 0;
        end else begin
            // Default: de-assert single-cycle pulses
            sa_clear     <= 1'b0;
            bram_c_wr_en <= 1'b0;
            done         <= 1'b0;

            case (state)

                ST_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy  <= 1'b1;
                        state <= ST_CLEAR;
                    end
                end

                // sa_clear=1 zeros accumulators on the next edge;
                // pre-loading addr=0 ensures BRAM data is ready when STREAM starts.
                ST_CLEAR: begin
                    sa_clear       <= 1'b1;
                    sa_a_in        <= 0;
                    sa_b_in        <= 0;
                    bram_a_rd_addr <= 0;
                    bram_b_rd_addr <= 0;
                    stream_cnt     <= 0;
                    state          <= ST_STREAM;
                end

                // BRAM read is async: addr is advanced one cycle ahead so data
                // is combinatorially available when latched into sa_a_in/sa_b_in.
                // mem[STREAM_LEN-1] is therefore seen by the SA in FLUSH cycle 0,
                // which is why FLUSH_LEN = N (not N-1).
                ST_STREAM: begin
                    // Capture data that was placed on the BRAM output last cycle
                    sa_a_in <= bram_a_rd_data;
                    sa_b_in <= bram_b_rd_data;

                    if (stream_cnt == STREAM_LEN - 1) begin
                        // Last slice fetched; do not advance addr further
                        flush_cnt <= 0;
                        state     <= ST_FLUSH;
                    end else begin
                        // Advance BRAM address so data is available next cycle
                        bram_a_rd_addr <= stream_cnt + 1;
                        bram_b_rd_addr <= stream_cnt + 1;
                        stream_cnt     <= stream_cnt + 1;
                    end
                end

                // N zero-cycles drain the pipeline: cycle 0 processes the last
                // real slice, cycles 1..N-1 propagate it to PE[N-1][N-1].
                ST_FLUSH: begin
                    sa_a_in <= 0;
                    sa_b_in <= 0;

                    if (flush_cnt == FLUSH_LEN - 1) begin
                        write_cnt <= 0;
                        state     <= ST_WRITE;
                    end else begin
                        flush_cnt <= flush_cnt + 1;
                    end
                end

                // c_out is stable; drain all N*N words into BRAM_C (row-major).
                ST_WRITE: begin
                    bram_c_wr_en   <= 1'b1;
                    bram_c_wr_addr <= write_cnt;
                    bram_c_wr_data <= sa_c_out[write_cnt * ACC_WIDTH +: ACC_WIDTH];

                    if (write_cnt == C_LEN - 1) begin
                        state <= ST_DONE;
                    end else begin
                        write_cnt <= write_cnt + 1;
                    end
                end

                ST_DONE: begin
                    done  <= 1'b1;
                    busy  <= 1'b0;
                    state <= ST_IDLE;
                end

                default: state <= ST_IDLE;

            endcase
        end
    end

endmodule
