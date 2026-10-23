/**
 * AURA-G FPGA kernel: digital waveguide string model with fret damping.
 */
#include <hls_math.h>
#include <ap_fixed.h>
#include <hls_stream.h>

typedef ap_fixed<32, 18, AP_RND, AP_SAT> sample_t;

#define MAX_DELAY 1024
#define SR 48000

sample_t frac_delay(sample_t buffer[MAX_DELAY], int delay_int,
                    sample_t delay_frac, int write_pos) {
    #pragma HLS INLINE
    int rp = (write_pos - delay_int + MAX_DELAY) % MAX_DELAY;
    int rpn = (rp - 1 + MAX_DELAY) % MAX_DELAY;
    return buffer[rp] + delay_frac * (buffer[rpn] - buffer[rp]);
}

void aura_g_waveguide(
    hls::stream<bool> &trigger,
    hls::stream<sample_t> &out_stream,
    sample_t freq_hz, sample_t damping, sample_t pluck_pos) {
    #pragma HLS INTERFACE axis port=trigger
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=freq_hz
    #pragma HLS INTERFACE s_axilite port=damping
    #pragma HLS INTERFACE s_axilite port=pluck_pos

    static sample_t dl[MAX_DELAY];
    #pragma HLS BIND_STORAGE variable=dl type=RAM_2P impl=BRAM

    int wp = 0;
    sample_t last = 0.0;
    bool init = false;

    for (int i = 0; i < SR * 10; i++) {
        #pragma HLS PIPELINE II=1
        if (!trigger.empty()) {
            if (trigger.read()) {
                sample_t dl_len = (sample_t)SR / freq_hz;
                int di = (int)dl_len;
                int pi = (int)(pluck_pos * di);
                for (int j = 0; j < di; j++) {
                    #pragma HLS UNROLL factor=4
                    if (j < pi) dl[j] = (sample_t)j / pi;
                    else dl[j] = (sample_t)(di - j) / (di - pi);
                }
                init = true;
                last = 0.0;
            }
        }
        if (!init) { out_stream.write(0.0); continue; }
        sample_t dl_len = (sample_t)SR / freq_hz;
        int di = (int)dl_len;
        sample_t df = dl_len - di;
        sample_t delayed = frac_delay(dl, di, df, wp);
        sample_t filt = delayed * damping + last * (1.0 - damping);
        dl[wp] = filt;
        wp = (wp + 1) % MAX_DELAY;
        last = filt;
        out_stream.write(delayed);
    }
}
