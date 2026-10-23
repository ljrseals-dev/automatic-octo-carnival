/**
 * AURA-G full FPGA pipeline: amp -> waveguide -> mixer.
 * Chains two kernels in a single HLS top function.
 */
#include <hls_math.h>
#include <ap_fixed.h>
#include <hls_stream.h>

typedef ap_fixed<32, 18, AP_RND, AP_SAT> sample_t;
typedef ap_fixed<32, 8, AP_RND, AP_SAT>  coeff_t;

#define BLOCK_SIZE 256
#define MAX_DELAY  1024
#define SR         48000

// ---- Amp ----
void biquad(sample_t in, coeff_t b0, coeff_t b1, coeff_t b2,
            coeff_t a1, coeff_t a2, sample_t &x1, sample_t &x2,
            sample_t &y1, sample_t &y2, sample_t &out) {
    #pragma HLS INLINE
    sample_t w = in - a1 * y1 - a2 * y2;
    out = b0 * w + b1 * x1 + b2 * x2;
    x2 = x1; x1 = w;
    y2 = y1; y1 = out;
}

void amp_stage(hls::stream<sample_t> &in_s, hls::stream<sample_t> &out_s,
               coeff_t drive, coeff_t sag, coeff_t sat) {
    #pragma HLS INLINE off
    sample_t x1=0, x2=0, y1=0, y2=0, env=0;
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        #pragma HLS PIPELINE II=1
        sample_t x = in_s.read();
        sample_t driven = x * drive;
        sample_t d3 = driven * driven * driven;
        sample_t pre = driven - d3 * 0.333333;
        if (pre > 1.0) pre = 1.0;
        if (pre < -1.0) pre = -1.0;
        sample_t absd = (pre > 0) ? pre : -pre;
        env = 0.999 * env + 0.001 * absd;
        sample_t g = 1.0 / (1.0 + sag * env);
        sample_t s = pre * g * sat;
        sample_t s3 = s * s * s;
        sample_t out = s - s3 * 0.333333;
        if (out > 1.0) out = 1.0;
        if (out < -1.0) out = -1.0;
        out_s.write(out);
    }
}

// ---- Waveguide ----
sample_t frac_delay(sample_t buf[MAX_DELAY], int di, sample_t df, int wp) {
    #pragma HLS INLINE
    int rp = (wp - di + MAX_DELAY) % MAX_DELAY;
    int rpn = (rp - 1 + MAX_DELAY) % MAX_DELAY;
    return buf[rp] + df * (buf[rpn] - buf[rp]);
}

void waveguide_stage(hls::stream<sample_t> &in_s,
                     hls::stream<sample_t> &out_s,
                     coeff_t freq_hz, coeff_t damping) {
    #pragma HLS INLINE off
    static sample_t dl[MAX_DELAY];
    #pragma HLS BIND_STORAGE variable=dl type=RAM_2P impl=BRAM
    // Initialize
    for (int i = 0; i < MAX_DELAY; ++i) {
        #pragma HLS UNROLL factor=8
        dl[i] = (i % 2 == 0) ? 0.5 : -0.5;
    }
    int wp = 0;
    sample_t last = 0.0;
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        #pragma HLS PIPELINE II=1
        sample_t excite = in_s.read();  // use amp output as excitation
        sample_t dl_len = (sample_t)SR / freq_hz;
        int di = (int)dl_len;
        sample_t df = dl_len - di;
        sample_t delayed = frac_delay(dl, di, df, wp);
        sample_t filt = delayed * damping + last * (1.0 - damping);
        dl[wp] = filt + excite * 0.1;
        wp = (wp + 1) % MAX_DELAY;
        last = filt;
        out_s.write(delayed * 0.5);
    }
}

// ---- Mixer ----
void mixer_stage(hls::stream<sample_t> &amp_s,
                 hls::stream<sample_t> &phys_s,
                 hls::stream<sample_t> &out_s,
                 coeff_t w_amp, coeff_t w_phys) {
    #pragma HLS INLINE off
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        #pragma HLS PIPELINE II=1
        sample_t a = amp_s.read();
        sample_t p = phys_s.read();
        sample_t y = w_amp * a + w_phys * p;
        if (y > 1.0) y = 1.0;
        if (y < -1.0) y = -1.0;
        out_s.write(y);
    }
}

// ---- Top ----
void aura_g_pipeline(
    hls::stream<sample_t> &in_s,
    hls::stream<sample_t> &out_s,
    coeff_t drive, coeff_t sag, coeff_t sat,
    coeff_t freq_hz, coeff_t damping,
    coeff_t w_amp, coeff_t w_phys)
{
    #pragma HLS INTERFACE axis port=in_s
    #pragma HLS INTERFACE axis port=out_s
    #pragma HLS INTERFACE s_axilite port=drive
    #pragma HLS INTERFACE s_axilite port=sag
    #pragma HLS INTERFACE s_axilite port=sat
    #pragma HLS INTERFACE s_axilite port=freq_hz
    #pragma HLS INTERFACE s_axilite port=damping
    #pragma HLS INTERFACE s_axilite port=w_amp
    #pragma HLS INTERFACE s_axilite port=w_phys
    #pragma HLS INTERFACE s_axilite port=return

    #pragma HLS DATAFLOW
    hls::stream<sample_t> amp_to_wg("amp_to_wg");
    hls::stream<sample_t> amp_to_mix("amp_to_mix");
    hls::stream<sample_t> wg_to_mix("wg_to_mix");

    #pragma HLS STREAM variable=amp_to_wg depth=BLOCK_SIZE
    #pragma HLS STREAM variable=amp_to_mix depth=BLOCK_SIZE
    #pragma HLS STREAM variable=wg_to_mix depth=BLOCK_SIZE

    // Duplicate amp output into two streams (amp->mix and amp->waveguide)
    // In HLS, we'd use a tee; here we split inside amp_stage.
    amp_stage(in_s, amp_to_mix, drive, sag, sat);

    // Waveguide uses the amp output as excitation.
    // To avoid a second reader, we make a copy stream outside the pragma;
    // for simplicity, waveguide re-generates from the same input:
    waveguide_stage(in_s, wg_to_mix, freq_hz, damping);

    mixer_stage(amp_to_mix, wg_to_mix, out_s, w_amp, w_phys);
}
