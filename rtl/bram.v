// bram.v — Simple dual-port register file
//
// One synchronous write port, one asynchronous (combinatorial) read port.
// Suitable for simulation; replace with FPGA BRAM primitives for synthesis.

`timescale 1ns / 1ps

module bram #(
    parameter WIDTH = 32,
    parameter DEPTH = 16
)(
    input  wire                      clk,

    // Write port (synchronous)
    input  wire                      wr_en,
    input  wire [$clog2(DEPTH)-1:0]  wr_addr,
    input  wire [WIDTH-1:0]          wr_data,

    // Read port (asynchronous / combinatorial)
    input  wire [$clog2(DEPTH)-1:0]  rd_addr,
    output wire [WIDTH-1:0]          rd_data
);

    reg [WIDTH-1:0] mem [0:DEPTH-1];

    integer k;
    initial begin
        for (k = 0; k < DEPTH; k = k + 1)
            mem[k] = 0;
    end

    always @(posedge clk) begin
        if (wr_en)
            mem[wr_addr] <= wr_data;
    end

    assign rd_data = mem[rd_addr];

endmodule
