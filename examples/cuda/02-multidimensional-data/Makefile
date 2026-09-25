.PHONY: all clean

# Anvil has libpng but no module for it in the GPU tree; elsewhere this is empty.
LIBPNG_HOME ?= $(wildcard /apps/spack/anvilgpu/apps/libpng/1.6.37-gcc-8.4.1-zbpsn2w)
PNG_FLAGS = $(if $(LIBPNG_HOME),-I$(LIBPNG_HOME)/include -L$(LIBPNG_HOME)/lib -Xlinker -rpath=$(LIBPNG_HOME)/lib) -lpng -lz

all: 2.1_multidimensional_indices 2.2_rgb_to_grayscale \
	2.3_image_blur 2.4_matrix_multiplication 2.5_vector_add overall

2.1_multidimensional_indices: 2.1_multidimensional_indices.cu ../common/cuda_helpers.h
	nvcc -O3 -std=c++17 2.1_multidimensional_indices.cu -o 2.1_multidimensional_indices

2.2_rgb_to_grayscale: 2.2_rgb_to_grayscale.cu ../common/cuda_helpers.h
	nvcc -O3 -std=c++17 2.2_rgb_to_grayscale.cu -o 2.2_rgb_to_grayscale

2.3_image_blur: 2.3_image_blur.cu png_io.h ../common/cuda_helpers.h
	nvcc -O3 -std=c++17 2.3_image_blur.cu $(PNG_FLAGS) -o 2.3_image_blur

2.4_matrix_multiplication: 2.4_matrix_multiplication.cu ../common/cuda_helpers.h
	nvcc -O3 -std=c++17 2.4_matrix_multiplication.cu -o 2.4_matrix_multiplication

2.5_vector_addup: 2.5_vector_addup.cu
	nvcc -O3 -std=c++17 2.5_vector_addup.cu -o 2.5_vector_addup

overall: overall.cu png_io.h ../common/cuda_helpers.h
	nvcc -O3 -std=c++17 overall.cu $(PNG_FLAGS) -o overall

clean:
	rm -f 2.1_multidimensional_indices 2.2_rgb_to_grayscale \
		2.3_image_blur 2.4_matrix_multiplication 2.5_vector_addup overall \
		bird_grayscale.png bird_blurred.png \
		overall_bird_grayscale.png overall_bird_blurred.png
