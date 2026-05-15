// top_basys3.v — General-purpose Basys3 top-level
// Parameterized for any input size and network topology.
//
// Retargeting to a new application:
//   1. Change IMG_PIXELS to match your input size (e.g. 1024 for 32x32 thermal)
//   2. Update mlp_accel parameters to match your trained network
//   3. Flash new hex weight files
//   4. Rebuild bitstream — no other RTL changes needed
`default_nettype none

module top_basys3 #(
    // ── Platform ──────────────────────────────────────────────────────────────
    parameter CLK_HZ     = 100_000_000,  // Basys3 onboard oscillator
    parameter BAUD       = 115_200,

    // ── Application ───────────────────────────────────────────────────────────
    parameter IMG_PIXELS = 784,          // bytes to receive before inference
                                         // 784 = MNIST 28x28
                                         // 1024 = thermal 32x32

    // ── Network topology (must match trained model) ───────────────────────────
    parameter L1_IN      = 784,
    parameter L1_OUT     = 64,
    parameter L2_IN      = 64,
    parameter L2_OUT     = 32,
    parameter L3_IN      = 32,
    parameter L3_OUT     = 10,

    // ── Weight / bias hex files ───────────────────────────────────────────────
    parameter W1_HEX     = "hex/weight_l1.hex",
    parameter W2_HEX     = "hex/weight_l2.hex",
    parameter W3_HEX     = "hex/weight_l3.hex",
    parameter B1_HEX     = "hex/bias_l1.hex",
    parameter B2_HEX     = "hex/bias_l2.hex",
    parameter B3_HEX     = "hex/bias_l3.hex"
)(
    input  wire       clk,      // 100 MHz onboard oscillator (W5)
    input  wire       btnC,     // centre button = reset (active-high)
    input  wire       RsRx,     // USB-UART RX (B18)
    output wire       RsTx,     // USB-UART TX (A18)
    output wire [3:0] led       // LD0-LD3: result in binary
);
    wire rst = btnC;

    // ── UART RX ───────────────────────────────────────────────────────────────
    wire [7:0] rx_data;
    wire       rx_valid;

    uart_rx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_rx (
        .clk(clk), .rst(rst),
        .rx_pin(RsRx),
        .rx_data(rx_data), .rx_valid(rx_valid)
    );

    // ── Pixel buffer — sized by IMG_PIXELS parameter ──────────────────────────
    reg [7:0]                        pix_buf [0:IMG_PIXELS-1];
    reg [$clog2(IMG_PIXELS)-1:0]     wr_ptr;
    reg                              start;

    always @(posedge clk) begin
        start <= 0;
        if (rst) wr_ptr <= 0;
        else if (rx_valid) begin
            pix_buf[wr_ptr] <= rx_data;
            if (wr_ptr == IMG_PIXELS-1) begin
                wr_ptr <= 0;
                start  <= 1;
            end else begin
                wr_ptr <= wr_ptr + 1;
            end
        end
    end

    // ── MLP accelerator ───────────────────────────────────────────────────────
    wire [$clog2(L1_IN)-1:0] pix_addr;
    wire [3:0]                result;
    wire                      done;

    reg [7:0] pix_data;
    always @(posedge clk) pix_data <= pix_buf[pix_addr];

    mlp_accel #(
        .L1_IN (L1_IN),  .L1_OUT(L1_OUT),
        .L2_IN (L2_IN),  .L2_OUT(L2_OUT),
        .L3_IN (L3_IN),  .L3_OUT(L3_OUT),
        .W1_HEX(W1_HEX), .W2_HEX(W2_HEX), .W3_HEX(W3_HEX),
        .B1_HEX(B1_HEX), .B2_HEX(B2_HEX), .B3_HEX(B3_HEX)
    ) u_mlp (
        .clk(clk), .rst(rst),
        .start(start), .done(done),
        .result(result),
        .pix_addr(pix_addr),
        .pix_data(pix_data)
    );

    // ── Latch result and trigger TX ───────────────────────────────────────────
    reg [3:0] result_r;
    reg       tx_start_r;

    always @(posedge clk) begin
        tx_start_r <= 0;
        if (done) begin
            result_r   <= result;
            tx_start_r <= 1;
        end
    end

    // ── UART TX ───────────────────────────────────────────────────────────────
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
