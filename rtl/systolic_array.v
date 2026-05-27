// Parameterized NxN Systolic Array for matrix multiplication
// Computes C = A x B using weight-stationary dataflow
//
// Inputs are fed in a skewed (diagonal wavefront) fashion:
//   - Row i of A is delayed by i cycles
//   - Column j of B is delayed by j cycles
//
// After 2*N-1 compute cycles (plus pipeline latency), the result
// matrix C is available in the PE accumulators.

`timescale 1ns / 1ps

module systolic_array #(
    parameter N          = 4,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire                                    clk,
    input  wire                                    rst,
    input  wire                                    clear,
    input  wire signed [N*DATA_WIDTH-1:0]          a_in,  // packed row inputs
    input  wire signed [N*DATA_WIDTH-1:0]          b_in,  // packed col inputs
    output wire signed [N*N*ACC_WIDTH-1:0]         c_out  // packed result matrix
);

    // Internal wires: horizontal a-links and vertical b-links
    // a_wire[i][j] connects PE(i,j) a_out -> PE(i,j+1) a_in
    // b_wire[i][j] connects PE(i,j) b_out -> PE(i+1,j) b_in
    wire signed [DATA_WIDTH-1:0] a_wire [0:N-1][0:N]; // N rows, N+1 columns (includes input)
    wire signed [DATA_WIDTH-1:0] b_wire [0:N][0:N-1]; // N+1 rows, N columns (includes input)
    wire signed [ACC_WIDTH-1:0]  c_wire [0:N-1][0:N-1];

    // Connect external inputs to the left edge (a) and top edge (b)
    genvar gi;
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : gen_inputs
            assign a_wire[gi][0] = a_in[gi*DATA_WIDTH +: DATA_WIDTH];
            assign b_wire[0][gi] = b_in[gi*DATA_WIDTH +: DATA_WIDTH];
        end
    endgenerate

    // Instantiate NxN grid of PEs
    genvar row, col;
    generate
        for (row = 0; row < N; row = row + 1) begin : gen_row
            for (col = 0; col < N; col = col + 1) begin : gen_col
                pe #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH(ACC_WIDTH)
                ) pe_inst (
                    .clk   (clk),
                    .rst   (rst),
                    .clear (clear),
                    .a_in  (a_wire[row][col]),
                    .b_in  (b_wire[row][col]),
                    .a_out (a_wire[row][col+1]),
                    .b_out (b_wire[row+1][col]),
                    .c     (c_wire[row][col])
                );

                // Pack output: c_out[(row*N + col)*ACC_WIDTH +: ACC_WIDTH]
                assign c_out[(row*N + col)*ACC_WIDTH +: ACC_WIDTH] = c_wire[row][col];
            end
        end
    endgenerate

endmodule
