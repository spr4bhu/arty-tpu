// tpu_top.v — Top-level TPU wrapper
//
// Integrates:
//   BRAM_A   — host loads pre-skewed matrix A diagonal slices
//   BRAM_B   — host loads pre-skewed matrix B diagonal slices
//   BRAM_C   — controller writes results; host reads results
//   systolic_array — NxN weight-stationary MAC array
//   tpu_controller — FSM sequencing the computation
//
// Host workflow:
//   1. Write pre-skewed A slices to BRAM_A (STREAM_LEN words of N*DATA_WIDTH bits)
//   2. Write pre-skewed B slices to BRAM_B (same format)
//   3. Assert start for one cycle
//   4. Poll busy/done; wait for done pulse
//   5. Read N*N words from BRAM_C (row-major, each word is ACC_WIDTH bits signed)
//
// BRAM_A / BRAM_B addressing:
//   addr t (0 .. 2N-2) holds the packed diagonal slice for timestep t.
//   For row i: A_lane_i = A[i][t-i] if valid, else 0.
//   For col j: B_lane_j = B[t-j][j] if valid, else 0.
//
// BRAM_C addressing:
//   addr = row*N + col  →  C[row][col]  (32-bit signed)

`timescale 1ns / 1ps

module tpu_top #(
    parameter N          = 4,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire clk,
    input  wire rst,

    // ---- Host → BRAM_A (pre-skewed matrix A slices) ----------------------
    input  wire                        wr_en_a,
    input  wire [$clog2(2*N)-1:0]      wr_addr_a,
    input  wire [N*DATA_WIDTH-1:0]     wr_data_a,

    // ---- Host → BRAM_B (pre-skewed matrix B slices) ----------------------
    input  wire                        wr_en_b,
    input  wire [$clog2(2*N)-1:0]      wr_addr_b,
    input  wire [N*DATA_WIDTH-1:0]     wr_data_b,

    // ---- Host ← BRAM_C (result matrix C, async read) --------------------
    input  wire [$clog2(N*N)-1:0]      rd_addr_c,
    output wire [ACC_WIDTH-1:0]        rd_data_c,

    // ---- Control ---------------------------------------------------------
    input  wire start,
    output wire busy,
    output wire done
);

    // Internal wires

    // Controller → SA
    wire                       sa_clear;
    wire [N*DATA_WIDTH-1:0]    sa_a_in;
    wire [N*DATA_WIDTH-1:0]    sa_b_in;
    wire [N*N*ACC_WIDTH-1:0]   sa_c_out;

    // Controller ↔ BRAM_A
    wire [$clog2(2*N)-1:0]     bram_a_rd_addr;
    wire [N*DATA_WIDTH-1:0]    bram_a_rd_data;

    // Controller ↔ BRAM_B
    wire [$clog2(2*N)-1:0]     bram_b_rd_addr;
    wire [N*DATA_WIDTH-1:0]    bram_b_rd_data;

    // Controller → BRAM_C
    wire                       bram_c_wr_en;
    wire [$clog2(N*N)-1:0]     bram_c_wr_addr;
    wire [ACC_WIDTH-1:0]       bram_c_wr_data;

    // BRAM_A  (host writes; controller reads)
    // Depth = 2*N  (holds STREAM_LEN = 2N-1 slices; slot 2N-1 is unused pad)
    // Width = N * DATA_WIDTH  (one packed lane per row)
    bram #(
        .WIDTH (N * DATA_WIDTH),
        .DEPTH (2 * N)
    ) bram_a (
        .clk     (clk),
        .wr_en   (wr_en_a),
        .wr_addr (wr_addr_a),
        .wr_data (wr_data_a),
        .rd_addr (bram_a_rd_addr),
        .rd_data (bram_a_rd_data)
    );

    // BRAM_B  (host writes; controller reads)
    bram #(
        .WIDTH (N * DATA_WIDTH),
        .DEPTH (2 * N)
    ) bram_b (
        .clk     (clk),
        .wr_en   (wr_en_b),
        .wr_addr (wr_addr_b),
        .wr_data (wr_data_b),
        .rd_addr (bram_b_rd_addr),
        .rd_data (bram_b_rd_data)
    );

    // BRAM_C  (controller writes; host reads)
    // Depth = N*N  (one 32-bit word per output element, row-major)
    bram #(
        .WIDTH (ACC_WIDTH),
        .DEPTH (N * N)
    ) bram_c (
        .clk     (clk),
        .wr_en   (bram_c_wr_en),
        .wr_addr (bram_c_wr_addr),
        .wr_data (bram_c_wr_data),
        .rd_addr (rd_addr_c),
        .rd_data (rd_data_c)
    );

    // Systolic Array
    systolic_array #(
        .N          (N),
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH)
    ) sa (
        .clk   (clk),
        .rst   (rst),
        .clear (sa_clear),
        .a_in  (sa_a_in),
        .b_in  (sa_b_in),
        .c_out (sa_c_out)
    );

    // TPU Controller
    tpu_controller #(
        .N          (N),
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH)
    ) ctrl (
        .clk            (clk),
        .rst            (rst),
        .start          (start),
        .busy           (busy),
        .done           (done),
        .sa_clear       (sa_clear),
        .sa_a_in        (sa_a_in),
        .sa_b_in        (sa_b_in),
        .sa_c_out       (sa_c_out),
        .bram_a_rd_addr (bram_a_rd_addr),
        .bram_a_rd_data (bram_a_rd_data),
        .bram_b_rd_addr (bram_b_rd_addr),
        .bram_b_rd_data (bram_b_rd_data),
        .bram_c_wr_en   (bram_c_wr_en),
        .bram_c_wr_addr (bram_c_wr_addr),
        .bram_c_wr_data (bram_c_wr_data)
    );

endmodule
