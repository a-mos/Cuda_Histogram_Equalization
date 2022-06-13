#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define NBINS 256
#define BLOCKSIZE 1024
#define CHANNELS 3
#define SHARED_NUM_OPS 32

__device__ unsigned char clamp(float x, unsigned char min, unsigned char max) {
	if (x >= max)
		return max;
	else if (x <= min)
		return min;
	else
		return x;
}

__global__ void forwardTransform(unsigned int* histogram, unsigned char* input_rgb, unsigned char* output_grayscale, unsigned int size) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x < size) {
		int pos_rgb = x * CHANNELS;
		output_grayscale[x] = input_rgb[pos_rgb] * 0.2126f +
							  input_rgb[pos_rgb + 1] * 0.7152f +
							  input_rgb[pos_rgb + 2] * 0.0722f;
	}
}

__global__ void histogramCalc(unsigned int* histogram, unsigned char* grayscale, unsigned int size) {
	
	__shared__ unsigned int local_hist[NBINS];

	local_hist[threadIdx.x] = 0;

	__syncthreads();
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	while (x < size) {
		atomicAdd(&local_hist[grayscale[x]], 1);
		x += blockDim.x * gridDim.x;
	}
	__syncthreads();

	atomicAdd(&histogram[threadIdx.x], local_hist[threadIdx.x]);
}

__global__ void compute_cdf_parallel(unsigned int* histogram, unsigned char* output_cdf, unsigned int size) {
	__shared__ float scan[NBINS];

	int x = threadIdx.x;
	scan[x] = histogram[x];
	scan[x + blockDim.x] = histogram[x + blockDim.x];

	__syncthreads();

	for (unsigned int stride = 1; stride <= NBINS / 2; stride <<= 1) {
		unsigned int index = (x + 1) * stride * 2 - 1;
		if (index < NBINS) {
			scan[index] += scan[index - stride];
		}
		__syncthreads();
	}

	for (unsigned int stride = NBINS / 4; stride > 0; stride >>= 1) {
		__syncthreads();
		unsigned int index = (x + 1) * stride * 2 - 1;
		if (index + stride < NBINS) {
			scan[index + stride] += scan[index];
		}
	}

	__syncthreads();
	output_cdf[x] = 255 * scan[x] / size;
	output_cdf[x + blockDim.x] = 255 * scan[x + blockDim.x] / size;
}

// For benchmarking only
__global__ void compute_cdf_simple(unsigned int* histogram, unsigned char* output_cdf, unsigned int size) {
	unsigned int min = histogram[0];
	output_cdf[0] = 0;
	for (int i = 1; i < NBINS; ++i) {
		histogram[i] += histogram[i - 1];
		output_cdf[i] = ((NBINS - 1) * (histogram[i] - min) / (size - min));
	}
}

__global__ void backwardTransform(unsigned char* cdf, unsigned char* grayscale, unsigned char* image, unsigned int size) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x < size) {
		int pos_rgb = x * CHANNELS;

		float y = cdf[grayscale[x]];
		float u = image[pos_rgb] * -0.114572f + image[pos_rgb + 1] * -0.385428f + image[pos_rgb + 2] * 0.5f;
		float v = image[pos_rgb] * 0.5f + image[pos_rgb + 1] * -0.454153f + image[pos_rgb + 2] * -0.045847f;

		float r = y + v * 1.5748f;
		float g = y + u * -0.187324f + v * -0.468124f;
		float b = y + u * 1.8556f;

		image[pos_rgb] = clamp(r, 0, 255);
		image[pos_rgb+1] = clamp(g, 0, 255);
		image[pos_rgb+2] = clamp(b, 0, 255);
	}
}

unsigned char* filter_gpu(unsigned char* data, int envmap_w, int envmap_h) {
	unsigned int spatial_size = envmap_w * envmap_h;

	const dim3 dimGrid((int)ceil(spatial_size / (float)BLOCKSIZE));
	const dim3 dimBlock(BLOCKSIZE);

	unsigned char *image_gpu, *grayscale_gpu, *cdf_gpu;
	unsigned int *histogram;

	unsigned char* output_cpu = (unsigned char*)malloc(sizeof(unsigned char) * spatial_size * CHANNELS);
	cudaMalloc(&image_gpu, spatial_size * CHANNELS * sizeof(unsigned char));
	cudaMalloc(&grayscale_gpu, spatial_size * sizeof(unsigned char));
	cudaMalloc(&histogram, NBINS * sizeof(unsigned int));
	cudaMalloc(&cdf_gpu, NBINS * sizeof(unsigned char));
	cudaMemset(histogram, 0, NBINS * sizeof(unsigned int));

	const auto start_with_copy = std::chrono::high_resolution_clock::now();
	cudaMemcpy(image_gpu, data, spatial_size * CHANNELS * sizeof(unsigned char), cudaMemcpyHostToDevice);
	const auto start_filter = std::chrono::high_resolution_clock::now();
	
	cudaDeviceSynchronize();
	forwardTransform<<<dimGrid, dimBlock>>> (histogram, image_gpu, grayscale_gpu, spatial_size);
	histogramCalc << <spatial_size / (SHARED_NUM_OPS * NBINS), NBINS>>> (histogram, grayscale_gpu, spatial_size);
	compute_cdf_parallel <<<1, NBINS / 2 >>> (histogram, cdf_gpu, spatial_size);
	backwardTransform<<<dimGrid, dimBlock>>> (cdf_gpu, grayscale_gpu, image_gpu, spatial_size);
	cudaDeviceSynchronize();

	const auto stop_filter = std::chrono::high_resolution_clock::now();
	cudaMemcpy(output_cpu, image_gpu, sizeof(unsigned char) * CHANNELS * spatial_size, cudaMemcpyDeviceToHost);
	const auto end_with_copy = std::chrono::high_resolution_clock::now();

	cudaFree(image_gpu);
	cudaFree(grayscale_gpu);
	cudaFree(histogram);
	cudaFree(cdf_gpu);
	
	double elapsed_time_filter_ms = std::chrono::duration<double, std::milli>(stop_filter - start_filter).count();
	double elapsed_time_total_ms = std::chrono::duration<double, std::milli>(end_with_copy - start_with_copy).count();

	std::cout << "Filter GPU time (without copy): " << elapsed_time_filter_ms << " ms\n";
	std::cout << "Filter GPU time (with copy): " << elapsed_time_total_ms << " ms\n";
	std::cout << "GPU Copy time: " << elapsed_time_total_ms - elapsed_time_filter_ms << " ms\n";
	return output_cpu;
}
