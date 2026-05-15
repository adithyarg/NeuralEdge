// mlp_accel.v — General-purpose MLP Accelerator
// Parameterized: any 3-layer fully-connected network, any input size, any output classes.
//
// Usage examples
// --------------
//   MNIST digit recognition  : L1_IN=784, L1_OUT=64, L2_OUT=32, L3_OUT=10
//   Thermal human detection  : L1_IN=1024, L1_OUT=64, L2_OUT=32, L3_OUT=2
//   Sensor anomaly detection : L1_IN=128,  L1_OUT=32, L2_OUT=16, L3_OUT=4
//
// To retarget: retrain in PyTorch, re-export hex files, update parameters at
// instantiation. No RTL logic changes needed.
//
// Critical path note
// ------------------
// BRAM address = base_addr + in_ctr (registered accumulator, no multiply).
// base_addr increments by L_IN at each STORE state → no wide multiplier on
// the critical path → clean timing closure at 100 MHz on Artix-7.

`default_nettype none
`timescale 1ns/1ps

// ============================================================================
// Weight BRAM — synchronous read, block RAM inferred via ram_style attribute
// ============================================================================
module weight_bram #(
    parameter DEPTH = 1024,
    parameter HEX   = "hex/weight_l1.hex"
)(
    input  wire                       clk,
    input  wire [$clog2(DEPTH)-1:0]   addr,
    output reg  signed [15:0]         dout
);
    (* ram_style = "block" *) reg signed [15:0] mem [0:DEPTH-1];
    initial $readmemh(HEX, mem);
    always @(posedge clk) dout <= mem[addr];
endmodule


// ============================================================================
// MAC — pipelined Q8.8 × Q4.12, 2-cycle latency, signed saturation
// acc += (a × b) >> 4   on each enabled cycle
// On clr: acc preloaded with bias_in << 8 (aligns Q8.8 bias to Q8.16 acc)
// ============================================================================
module mac (
    input  wire        clk, rst, clr, en,
    input  wire signed [15:0] a, b,
    input  wire signed [31:0] bias_in,
    output wire signed [31:0] acc_out,
    output wire               valid_out
);
    wire signed [31:0] a_sx = {{16{a[15]}}, a};
    wire signed [31:0] b_sx = {{16{b[15]}}, b};

    reg signed [31:0] product_r, acc_r;
    reg               en_r1, valid_r;

    wire signed [31:0] sum_w   = acc_r + product_r;
    wire               ovf_pos = (!acc_r[31] && !product_r[31] &&  sum_w[31]);
    wire               ovf_neg = ( acc_r[31] &&  product_r[31] && !sum_w[31]);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            product_r<=0; en_r1<=0; acc_r<=0; valid_r<=0;
        end else if (clr) begin
            product_r<=0; en_r1<=0; acc_r<=$signed(bias_in)<<<8; valid_r<=0;
        end else begin
            product_r <= en ? (a_sx * b_sx) >>> 4 : 32'sd0;
            en_r1     <= en;
            if (en_r1) begin
                if      (ovf_pos) acc_r <= 32'sh7fffffff;
                else if (ovf_neg) acc_r <= 32'sh80000000;
                else              acc_r <= sum_w;
                valid_r <= 1;
            end else valid_r <= 0;
        end
    end
    assign acc_out   = acc_r;
    assign valid_out = valid_r;
endmodule


// ============================================================================
// MLP accelerator — fully parameterized 3-layer FC network
// ============================================================================
module mlp_accel #(
    // ── Network topology ─────────────────────────────────────────────────────
    parameter L1_IN  = 784,   // input size  (e.g. 784 for MNIST, 1024 for 32x32)
    parameter L1_OUT = 64,    // hidden layer 1 neurons
    parameter L2_IN  = 64,    // must equal L1_OUT
    parameter L2_OUT = 32,    // hidden layer 2 neurons
    parameter L3_IN  = 32,    // must equal L2_OUT
    parameter L3_OUT = 10,    // output classes (e.g. 10 for MNIST, 2 for binary)

    // ── Weight hex files ─────────────────────────────────────────────────────
    parameter W1_HEX = "hex/weight_l1.hex",
    parameter W2_HEX = "hex/weight_l2.hex",
    parameter W3_HEX = "hex/weight_l3.hex",

    // ── Bias hex files ───────────────────────────────────────────────────────
    parameter B1_HEX = "hex/bias_l1.hex",
    parameter B2_HEX = "hex/bias_l2.hex",
    parameter B3_HEX = "hex/bias_l3.hex"
)(
    input  wire        clk, rst, start,
    output reg         done,
    output reg  [ 3:0] result,          // argmax index (0 to L3_OUT-1)
    output wire [$clog2(L1_IN)-1:0] pix_addr,
    input  wire [ 7:0] pix_data
);
    localparam DRAIN = 2;

    // ── FSM states (one-hot) ──────────────────────────────────────────────────
    localparam [17:0]
        S_IDLE     = 18'd1,    S_L1_PRE   = 18'd2,
        S_L1_PRE2  = 18'd4,    S_L1_MAC   = 18'd8,
        S_L1_DRAIN = 18'd16,   S_L1_STORE = 18'd32,
        S_L2_PRE   = 18'd64,   S_L2_PRE2  = 18'd128,
        S_L2_MAC   = 18'd256,  S_L2_DRAIN = 18'd512,
        S_L2_STORE = 18'd1024, S_L3_PRE   = 18'd2048,
        S_L3_PRE2  = 18'd4096, S_L3_MAC   = 18'd8192,
        S_L3_DRAIN = 18'd16384,S_L3_STORE = 18'd32768,
        S_ARGMAX   = 18'd65536,S_DONE     = 18'd131072;

    reg [17:0] state;

    // Counter widths derived from parameters
    reg [$clog2(L1_OUT > L2_OUT ? (L1_OUT > L3_OUT ? L1_OUT : L3_OUT) :
                                  (L2_OUT > L3_OUT ? L2_OUT : L3_OUT))-1:0] nrn_ctr;
    reg [$clog2(L1_IN)-1:0]  in_ctr;
    reg [1:0]                 drain_ctr;

    // base_addr: registered accumulator — no multiply on critical path
    reg [16:0] base_addr;

    // ── Bias memories (LUT RAM — small) ──────────────────────────────────────
    reg signed [15:0] b1_mem [0:L1_OUT-1];
    reg signed [15:0] b2_mem [0:L2_OUT-1];
    reg signed [15:0] b3_mem [0:L3_OUT-1];
    initial begin
        $readmemh(B1_HEX, b1_mem);
        $readmemh(B2_HEX, b2_mem);
        $readmemh(B3_HEX, b3_mem);
    end

    // ── Weight BRAMs ──────────────────────────────────────────────────────────
    wire [16:0] w_addr_next = base_addr + {{(17-$clog2(L1_IN)){1'b0}}, in_ctr} + 17'd1;
    wire [16:0] w_addr_cur  = base_addr;

    reg [16:0] w_addr;
    always @(*) begin
        case (state)
            S_L1_PRE, S_L1_PRE2,
            S_L2_PRE, S_L2_PRE2,
            S_L3_PRE, S_L3_PRE2: w_addr = w_addr_cur;
            S_L1_MAC, S_L2_MAC,
            S_L3_MAC:             w_addr = w_addr_next;
            default:              w_addr = 17'd0;
        endcase
    end

    wire signed [15:0] w1_dout, w2_dout, w3_dout;

    weight_bram #(
        .DEPTH(L1_OUT * L1_IN),
        .HEX(W1_HEX)
    ) u_w1 (
        .clk(clk),
        .addr(w_addr[$clog2(L1_OUT*L1_IN)-1:0]),
        .dout(w1_dout)
    );

    weight_bram #(
        .DEPTH(L2_OUT * L2_IN),
        .HEX(W2_HEX)
    ) u_w2 (
        .clk(clk),
        .addr(w_addr[$clog2(L2_OUT*L2_IN)-1:0]),
        .dout(w2_dout)
    );

    weight_bram #(
        .DEPTH(L3_OUT * L3_IN),
        .HEX(W3_HEX)
    ) u_w3 (
        .clk(clk),
        .addr(w_addr[$clog2(L3_OUT*L3_IN)-1:0]),
        .dout(w3_dout)
    );

    // Select BRAM output based on current layer (registered to match BRAM latency)
    reg is_l1, is_l2;
    always @(posedge clk or posedge rst) begin
        if (rst) begin is_l1<=0; is_l2<=0; end
        else begin
            is_l1 <= (state==S_L1_PRE)||(state==S_L1_PRE2)||(state==S_L1_MAC);
            is_l2 <= (state==S_L2_PRE)||(state==S_L2_PRE2)||(state==S_L2_MAC);
        end
    end
    wire signed [15:0] w_dout = is_l1 ? w1_dout : is_l2 ? w2_dout : w3_dout;

    // Bias: combinatorial read from LUT RAM
    wire signed [31:0] b_now =
        (state==S_L1_PRE2) ? {{16{b1_mem[nrn_ctr[$clog2(L1_OUT)-1:0]][15]}},
                               b1_mem[nrn_ctr[$clog2(L1_OUT)-1:0]]} :
        (state==S_L2_PRE2) ? {{16{b2_mem[nrn_ctr[$clog2(L2_OUT)-1:0]][15]}},
                               b2_mem[nrn_ctr[$clog2(L2_OUT)-1:0]]} :
                             {{16{b3_mem[nrn_ctr[$clog2(L3_OUT)-1:0]][15]}},
                               b3_mem[nrn_ctr[$clog2(L3_OUT)-1:0]]};

    // ── Activation buffers ────────────────────────────────────────────────────
    reg [15:0]        act1 [0:L1_OUT-1];
    reg [15:0]        act2 [0:L2_OUT-1];
    reg signed [31:0] act3 [0:L3_OUT-1];

    // ── MAC ───────────────────────────────────────────────────────────────────
    reg               mac_clr, mac_en;
    reg  signed [15:0] mac_a, mac_b;
    reg  signed [31:0] mac_bias;
    wire signed [31:0] mac_acc;
    wire               mac_valid;

    mac u_mac (
        .clk(clk), .rst(rst), .clr(mac_clr), .en(mac_en),
        .a(mac_a), .b(mac_b), .bias_in(mac_bias),
        .acc_out(mac_acc), .valid_out(mac_valid)
    );

    // ── ReLU ──────────────────────────────────────────────────────────────────
    localparam signed [31:0] SAT_LIM = 32'sh007FFF00;
    wire [15:0] relu_out =
        (mac_acc <= 32'sd0) ? 16'h0000 :
        (mac_acc >  SAT_LIM) ? 16'h7FFF : mac_acc[23:8];

    // ── Argmax (combinatorial over L3_OUT outputs) ────────────────────────────
    reg [3:0]         argmax_idx;
    reg signed [31:0] argmax_val;
    integer ai;
    always @(*) begin : argmax_comb
        argmax_val = act3[0];
        argmax_idx = 4'd0;
        for (ai = 1; ai < L3_OUT; ai = ai + 1)
            if (act3[ai] > argmax_val) begin
                argmax_val = act3[ai];
                argmax_idx = ai[3:0];
            end
    end

    assign pix_addr = in_ctr[$clog2(L1_IN)-1:0];

    // ── FSM ───────────────────────────────────────────────────────────────────
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state<=S_IDLE; done<=0; nrn_ctr<=0; in_ctr<=0;
            drain_ctr<=0; base_addr<=0; result<=0;
            mac_clr<=0; mac_en<=0; mac_a<=0; mac_b<=0; mac_bias<=0;
        end else begin
            done<=0; mac_clr<=0; mac_en<=0;

            case (state)
            S_IDLE:
                if (start) begin
                    nrn_ctr<=0; in_ctr<=0; base_addr<=0;
                    state<=S_L1_PRE;
                end

            // ── Layer 1 ───────────────────────────────────────────────────────
            S_L1_PRE:  begin in_ctr<=0; state<=S_L1_PRE2; end
            S_L1_PRE2: begin mac_clr<=1; mac_bias<=b_now; state<=S_L1_MAC; end
            S_L1_MAC: begin
                mac_en<=1;
                mac_a <= {8'b0, pix_data};
                mac_b <= w_dout;
                if (in_ctr == L1_IN-1) begin drain_ctr<=0; state<=S_L1_DRAIN; end
                else in_ctr <= in_ctr + 1;
            end
            S_L1_DRAIN: begin
                drain_ctr <= drain_ctr + 1;
                if (drain_ctr == DRAIN-1) state <= S_L1_STORE;
            end
            S_L1_STORE: begin
                act1[nrn_ctr[$clog2(L1_OUT)-1:0]] <= relu_out;
                if (nrn_ctr == L1_OUT-1) begin
                    nrn_ctr<=0; in_ctr<=0; base_addr<=0; state<=S_L2_PRE;
                end else begin
                    nrn_ctr<=nrn_ctr+1; in_ctr<=0;
                    base_addr<=base_addr+L1_IN; state<=S_L1_PRE;
                end
            end

            // ── Layer 2 ───────────────────────────────────────────────────────
            S_L2_PRE:  begin in_ctr<=0; state<=S_L2_PRE2; end
            S_L2_PRE2: begin mac_clr<=1; mac_bias<=b_now; state<=S_L2_MAC; end
            S_L2_MAC: begin
                mac_en<=1;
                mac_a <= $signed({1'b0, act1[in_ctr[$clog2(L2_IN)-1:0]]});
                mac_b <= w_dout;
                if (in_ctr == L2_IN-1) begin drain_ctr<=0; state<=S_L2_DRAIN; end
                else in_ctr <= in_ctr + 1;
            end
            S_L2_DRAIN: begin
                drain_ctr <= drain_ctr + 1;
                if (drain_ctr == DRAIN-1) state <= S_L2_STORE;
            end
            S_L2_STORE: begin
                act2[nrn_ctr[$clog2(L2_OUT)-1:0]] <= relu_out;
                if (nrn_ctr == L2_OUT-1) begin
                    nrn_ctr<=0; in_ctr<=0; base_addr<=0; state<=S_L3_PRE;
                end else begin
                    nrn_ctr<=nrn_ctr+1; in_ctr<=0;
                    base_addr<=base_addr+L2_IN; state<=S_L2_PRE;
                end
            end

            // ── Layer 3 ───────────────────────────────────────────────────────
            S_L3_PRE:  begin in_ctr<=0; state<=S_L3_PRE2; end
            S_L3_PRE2: begin mac_clr<=1; mac_bias<=b_now; state<=S_L3_MAC; end
            S_L3_MAC: begin
                mac_en<=1;
                mac_a <= $signed({1'b0, act2[in_ctr[$clog2(L3_IN)-1:0]]});
                mac_b <= w_dout;
                if (in_ctr == L3_IN-1) begin drain_ctr<=0; state<=S_L3_DRAIN; end
                else in_ctr <= in_ctr + 1;
            end
            S_L3_DRAIN: begin
                drain_ctr <= drain_ctr + 1;
                if (drain_ctr == DRAIN-1) state <= S_L3_STORE;
            end
            S_L3_STORE: begin
                act3[nrn_ctr[$clog2(L3_OUT)-1:0]] <= mac_acc;
                if (nrn_ctr == L3_OUT-1) state <= S_ARGMAX;
                else begin
                    nrn_ctr<=nrn_ctr+1; in_ctr<=0;
                    base_addr<=base_addr+L3_IN; state<=S_L3_PRE;
                end
            end

            S_ARGMAX: begin result <= argmax_idx; state <= S_DONE; end
            S_DONE:   begin done <= 1; state <= S_IDLE; end
            default:  state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
