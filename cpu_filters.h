#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <omp.h>

#define NBINS 256
#define BLOCKSIZE 1024
#define CHANNELS 3

unsigned char clamp_cpu(float x, unsigned char min, unsigned char max) {
    if (x >= max)
        return max;
    else if (x <= min)
        return min;
    else
        return x;
}

unsigned char* filter_cpu(unsigned char* data, int envmap_w, int envmap_h, int num_threads) {
    unsigned int spatial_size = envmap_w * envmap_h;
    unsigned char* image = (unsigned char*)malloc(spatial_size * CHANNELS * sizeof(unsigned char));
    unsigned char* grayscale = (unsigned char*)malloc(spatial_size * sizeof(unsigned char));
    unsigned int* histogram = (unsigned int*)malloc(NBINS * sizeof(unsigned int));
    unsigned char* cdf = (unsigned char*)malloc(NBINS * sizeof(unsigned char));

    memset(histogram, 0, NBINS * sizeof(unsigned int));
    memset(cdf, 0, NBINS * sizeof(unsigned char));
    omp_set_num_threads(num_threads);

    const auto start_filter = std::chrono::high_resolution_clock::now();

    # pragma omp parallel for
    for (int x = 0; x < spatial_size; ++x) {
        int pos_rgb = x * CHANNELS;
        unsigned char value = data[pos_rgb] * 0.2126f +
                              data[pos_rgb + 1] * 0.7152f +
                              data[pos_rgb + 2] * 0.0722f;
        grayscale[x] = value;
        
        # pragma omp atomic
        histogram[value]++;
    }

    unsigned int min = histogram[0];
    for (int i = 1; i < NBINS; ++i) {
        histogram[i] += histogram[i - 1];
        cdf[i] = ((NBINS - 1) * (histogram[i] - min) / (spatial_size - min));
    }

    # pragma omp parallel for
    for (int x = 0; x < spatial_size; ++x) {
        int pos_rgb = x * CHANNELS;
        float y = cdf[grayscale[x]];
        float u = data[pos_rgb] * -0.114572f + data[pos_rgb + 1] * -0.385428f + data[pos_rgb + 2] * 0.5f;
        float v = data[pos_rgb] * 0.5f + data[pos_rgb + 1] * -0.454153f + data[pos_rgb + 2] * -0.045847f;

        float r = y + v * 1.5748f;
        float g = y + u * -0.187324f + v * -0.468124f;
        float b = y + u * 1.8556f;

        image[pos_rgb] = clamp_cpu(r, 0, 255);
        image[pos_rgb + 1] = clamp_cpu(g, 0, 255);
        image[pos_rgb + 2] = clamp_cpu(b, 0, 255);
    }

    const auto end_filter = std::chrono::high_resolution_clock::now();

    double elapsed_time = std::chrono::duration<double, std::milli>(end_filter - start_filter).count();
    std::cout << "Filter CPU time (num threads-" << num_threads << "): " << elapsed_time << " ms\n";

    free(grayscale);
    free(histogram);
    free(cdf);
    return image;
}
