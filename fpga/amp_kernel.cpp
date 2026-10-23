/**
 * AURA-G FPGA kernel: DDSP amplifier (simplified)
 * Target: Xilinx Zynq / Alveo, Vitis HLS
 */
#include <hls_math.h>
#include <ap_fixed.h>
#include <hls_stream.h>

typedef ap_fixed<32, 18, AP_RND, AP_SAT> sample_t;
typedef ap_fixed<32, 8, AP_RND, AP_SAT>  coeff_t;

#define BLOCK_SIZE 256

struct BiquadCoeffs { coeff_t b0, b1, b2, a1, a2; };
struct BiquadState  { sample_t x1, x2, y1, y2; };

void biquad_process(BiquadCoeffs &c, BiquadState &s,
                    sample_t in, sample_t &out) {
    #pragma HLS INLINE
    sample_t w = in - c.a1 * s.y1 - c.a2 * s.y2;
    out = c.b0 * w + c.b1 * s.x1 + c.b2 * s.x2;
    s.x2 = s.x1; s.x1 = w;
    s.y2 = s.y1; s.y1 = out;
}

void preamp(sample_t in, sample_t drive, sample_t &out,
            BiquadCoeffs &bc, BiquadCoeffs &mc, BiquadCoeffs &tc,
            BiquadState &bs, BiquadState &ms, BiquadState &ts) {
    #pragma HLS INLINE
    sample_t x = in * drive;
    sample_t x3 = x * x * x;
    sample_t tx = x - x3 * 0.333333;
    if (tx > 1.0) tx = 1.0;
    if (tx < -1.0) tx = -1.0;
    sample_t t1, t2, t3;
    biquad_process(bc, bs, tx, t1);
    biquad_process(mc, ms, t1, t2);
    biquad_process(tc, ts, t2, out);
}

void power_amp(sample_t in, sample_t &out, sample_t &env,
               sample_t sag, sample_t sat) {
    #pragma HLS INLINE
    sample_t abs_in = (in > 0) ? in : -in;
    env = 0.999 * env + 0.001 * abs_in;
    sample_t g = 1.0 / (1.0 + sag * env);
    sample_t x = in * g * sat;
    sample_t x3 = x * x * x;
    out = x - x3 * 0.333333;
    if (out > 1.0) out = 1.0;
    if (out < -1.0) out = -1.0;
}

void aura_g_amp(
    hls::stream<sample_t> &in_stream,
    hls::stream<sample_t> &out_stream,
    sample_t drive, sample_t sag, sample_t saturation) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=drive
    #pragma HLS INTERFACE s_axilite port=sag
    #pragma HLS INTERFACE s_axilite port=saturation

    BiquadCoeffs bc = {1.0, 0.0, 0.0, 0.0, 0.0};
    BiquadCoeffs mc = {1.0, 0.0, 0.0, 0.0, 0.0};
    BiquadCoeffs tc = {1.0, 0.0, 0.0, 0.0, 0.0};
    BiquadState bs = {0, 0, 0, 0};
    BiquadState ms = {0, 0, 0, 0};
    BiquadState ts = {0, 0, 0, 0};
    sample_t env = 0.0;

    for (int i = 0; i < BLOCK_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        sample_t x = in_stream.read();
        sample_t pre, amp;
        preamp(x, drive, pre, bc, mc, tc, bs, ms, ts);
        power_amp(pre, amp, env, sag, saturation);
        out_stream.write(amp);
    }
}
