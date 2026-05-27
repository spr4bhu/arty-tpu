// Processing Element (PE) for systolic array
// Performs multiply-accumulate: c += a_in * b_in
// Passes a and b to neighboring PEs

module pe #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire                         clk,
    input  wire                         rst,
    input  wire                         clear,
    input  wire signed [DATA_WIDTH-1:0] a_in,
    input  wire signed [DATA_WIDTH-1:0] b_in,
    output reg  signed [DATA_WIDTH-1:0] a_out,
    output reg  signed [DATA_WIDTH-1:0] b_out,
    output reg  signed [ACC_WIDTH-1:0]  c
);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            a_out <= 0;
            b_out <= 0;
            c     <= 0;
        end else if (clear) begin
            c     <= 0;
            a_out <= a_in;
            b_out <= b_in;
        end else begin
            c     <= c + (a_in * b_in);
            a_out <= a_in;
            b_out <= b_in;
        end
    end

endmodule
