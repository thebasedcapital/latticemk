NVCC ?= $(HOME)/.local/cuda-13.3/bin/nvcc
# Only gcc 16 is installed; nvcc 13.3 officially stops at gcc 15. Correctness is covered by gemv_bench's checks.
NVFLAGS = -O3 -arch=sm_75 -std=c++17 -allow-unsupported-compiler -lineinfo -L$(dir $(NVCC))../lib

all: build/libvqenc.so build/libgemv.so

build/libvqenc.so: kernels/vqenc.cu
	@mkdir -p build
	$(NVCC) $(NVFLAGS) -Xcompiler -fPIC -shared -cudart static -o $@ $<

build/libgemv.so: kernels/gemv.cu
	@mkdir -p build
	$(NVCC) $(NVFLAGS) -Xptxas -v -Xcompiler -fPIC -shared -cudart static -o $@ $< 2>build/ptxas.log

clean:
	rm -rf build

.PHONY: all clean
