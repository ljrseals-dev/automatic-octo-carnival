TARGET   := hw
PLATFORM := xilinx_zcu104_base_202220_1
XCLBIN   := aura_g_amp.xclbin
HOST     := host

.PHONY: all clean xo xclbin host run

all: host xclbin

xo:
	v++ -c -t $(TARGET) --platform $(PLATFORM) --config cfg/amp.cfg \
		-k aura_g_amp fpga/amp_kernel.cpp -o aura_g_amp.xo

xclbin: xo
	v++ -l -t $(TARGET) --platform $(PLATFORM) aura_g_amp.xo -o $(XCLBIN)

host:
	g++ -std=c++17 -O3 -I$(XILINX_XRT)/include -I/usr/include \
		fpga/host.cpp -L$(XILINX_XRT)/lib \
		-lxrt_coreutil -lOpenCL -lsndfile -lpthread -o $(HOST)

run: all
	./$(HOST) input.wav output_fpga.wav

clean:
	rm -f *.xo *.xclbin $(HOST) *.log
