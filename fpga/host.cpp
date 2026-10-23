#include <iostream>
#include <fstream>
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
    SF_INFO sfinfo;
    SNDFILE* infile = sf_open(argv[1], SFM_READ, &sfinfo);
    if (!infile) { std::cerr << "Cannot open input\n"; return 1; }
    std::vector<float> input(sfinfo.frames);
    sf_read_float(infile, input.data(), sfinfo.frames);
    sf_close(infile);

    auto devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);
    auto fileBuf = xcl::read_binary_file("aura_g_amp.xclbin");
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    cl::Program program(context, {device}, bins);
    cl::Kernel kernel(program, "aura_g_amp");

    size_t in_size = BLOCK_SIZE * sizeof(float);
    size_t out_size = BLOCK_SIZE * sizeof(float);
    cl::Buffer in_buf(context, CL_MEM_READ_ONLY, in_size);
    cl::Buffer out_buf(context, CL_MEM_WRITE_ONLY, out_size);

    kernel.setArg(0, in_buf);
    kernel.setArg(1, out_buf);
    kernel.setArg(2, 1.0f);
    kernel.setArg(3, 0.3f);
    kernel.setArg(4, 0.7f);

    std::vector<float> output(sfinfo.frames);
    size_t nb = (sfinfo.frames + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (size_t b = 0; b < nb; b++) {
        size_t s = b * BLOCK_SIZE;
        size_t len = std::min((size_t)BLOCK_SIZE, sfinfo.frames - s);
        std::vector<float> ih(BLOCK_SIZE, 0);
        for (size_t i = 0; i < len; i++) ih[i] = input[s + i];
        q.enqueueWriteBuffer(in_buf, CL_TRUE, 0, in_size, ih.data());
        q.enqueueTask(kernel);
        q.finish();
        std::vector<float> oh(BLOCK_SIZE);
        q.enqueueReadBuffer(out_buf, CL_TRUE, 0, out_size, oh.data());
        for (size_t i = 0; i < len; i++) output[s + i] = oh[i];
    }

    SF_INFO oi = sfinfo;
    SNDFILE* outfile = sf_open(argv[2], SFM_WRITE, &oi);
    sf_write_float(outfile, output.data(), output.size());
    sf_close(outfile);
    std::cout << "Done: " << argv[2] << "\n";
    return 0;
}
