/** Host for the chained pipeline kernel. */
#include <iostream>
#include <vector>
#include <algorithm>
#include <sndfile.h>
#include "xcl2.hpp"

#define BLOCK_SIZE 256

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.wav> <output.wav>\n";
        return 1;
    }
    SF_INFO si;
    SNDFILE* in = sf_open(argv[1], SFM_READ, &si);
    std::vector<float> input(si.frames);
    sf_read_float(in, input.data(), si.frames);
    sf_close(in);

    auto devs = xcl::get_xil_devices();
    cl::Context ctx(devs[0]);
    cl::CommandQueue q(ctx, devs[0], CL_QUEUE_PROFILING_ENABLE);
    auto buf = xcl::read_binary_file("aura_g_pipeline.xclbin");
    cl::Program::Binaries bins{{buf.data(), buf.size()}};
    cl::Program prog(ctx, {devs[0]}, bins);
    cl::Kernel k(prog, "aura_g_pipeline");

    cl::Buffer in_b(ctx, CL_MEM_READ_ONLY, BLOCK_SIZE * sizeof(float));
    cl::Buffer out_b(ctx, CL_MEM_WRITE_ONLY, BLOCK_SIZE * sizeof(float));

    k.setArg(0, in_b);  k.setArg(1, out_b);
    k.setArg(2, 1.0f);  k.setArg(3, 0.3f);  k.setArg(4, 0.7f);
    k.setArg(5, 196.0f); k.setArg(6, 0.995f);
    k.setArg(7, 0.5f);  k.setArg(8, 0.5f);

    std::vector<float> output(si.frames);
    size_t nb = (si.frames + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (size_t b = 0; b < nb; ++b) {
        size_t s = b * BLOCK_SIZE;
        size_t len = std::min((size_t)BLOCK_SIZE, si.frames - s);
        std::vector<float> ih(BLOCK_SIZE, 0);
        for (size_t i = 0; i < len; ++i) ih[i] = input[s + i];
        q.enqueueWriteBuffer(in_b, CL_TRUE, 0, BLOCK_SIZE*sizeof(float), ih.data());
        q.enqueueTask(k);
        q.finish();
        std::vector<float> oh(BLOCK_SIZE);
        q.enqueueReadBuffer(out_b, CL_TRUE, 0, BLOCK_SIZE*sizeof(float), oh.data());
        for (size_t i = 0; i < len; ++i) output[s + i] = oh[i];
    }
    SF_INFO so = si;
    SNDFILE* out = sf_open(argv[2], SFM_WRITE, &so);
    sf_write_float(out, output.data(), output.size());
    sf_close(out);
    std::cout << "Wrote " << argv[2] << "\n";
    return 0;
}
