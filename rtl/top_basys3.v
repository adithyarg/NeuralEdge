// top_basys3.v — Basys3 top-level (100 MHz, 115200 baud)
`default_nettype none

module top_basys3 #(
    parameter CLK_HZ = 100_000_000,
    parameter BAUD   = 115_200
)(
    input  wire       clk,      // 100 MHz onboard oscillator (W5)
    input  wire       btnC,     // centre button = reset
    input  wire       RsRx,     // USB-UART RX (pin B18)
    output wire       RsTx,     // USB-UART TX (pin A18)
    output wire [3:0] led       // LD0-LD3 show predicted digit in binary
);
    wire rst = btnC;

    wire [7:0] rx_data;
    wire       rx_valid;

    uart_rx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_rx (
        .clk(clk), .rst(rst),
        .rx_pin(RsRx),
        .rx_data(rx_data), .rx_valid(rx_valid)
    );

    reg [7:0] pix_buf [0:783];
    reg [9:0] wr_ptr;
    reg       start;

    always @(posedge clk) begin
        start <= 0;
        if (rst) wr_ptr <= 0;
        else if (rx_valid) begin
            pix_buf[wr_ptr] <= rx_data;
            if (wr_ptr == 783) begin wr_ptr <= 0; start <= 1; end
            else wr_ptr <= wr_ptr + 1;
        end
    end

    wire [9:0] pix_addr;
    wire [3:0] result;
    wire       done;

    reg [7:0] pix_data;
    always @(posedge clk) pix_data <= pix_buf[pix_addr];

    mlp_accel u_mlp (
        .clk(clk), .rst(rst),
        .start(start), .done(done),
        .result(result),
        .pix_addr(pix_addr),
        .pix_data(pix_data)
    );

    reg [3:0] result_r;
    reg       tx_start_r;
    always @(posedge clk) begin
        tx_start_r <= 0;
        if (done) begin result_r <= result; tx_start_r <= 1; end
    end

    wire tx_busy;
    uart_tx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_tx (
        .clk(clk), .rst(rst),
        .tx_data({4'b0, result_r}),
        .tx_start(tx_start_r),
        .tx_busy(tx_busy),
        .tx_pin(RsTx)
    );

    assign led = result_r;

endmodule

`default_nettype wire
