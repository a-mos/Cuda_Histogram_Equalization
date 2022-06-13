#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

unsigned char* filter_gpu(unsigned char *data, int envmap_w, int envmap_h);