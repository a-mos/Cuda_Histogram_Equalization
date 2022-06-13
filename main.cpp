#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "kernel.h"
#include "cpu_filters.h"
#include "string.h"

int main(int argc, const char** argv) {

	if (argc < 2) {
                std::cout << "Usage: ./filter {input_img_path}" << std::endl;
		return 1;
	}

	int envmap_w, envmap_h, channels;
	auto env_map_path = argv[1];
	unsigned char* pixmap = stbi_load(env_map_path, &envmap_w, &envmap_h, &channels, 3);

	std::cout << "Img shape: " << envmap_h << "x" << envmap_w << "\n";
	unsigned char* output_cpu_1_thread = filter_cpu(pixmap, envmap_w, envmap_h, 1);
	unsigned char* output_cpu_4_thread = filter_cpu(pixmap, envmap_w, envmap_h, 4);
	unsigned char* output_cpu_8_thread = filter_cpu(pixmap, envmap_w, envmap_h, 8);
	unsigned char* output_cuda = filter_gpu(pixmap, envmap_w, envmap_h);

	stbi_write_png("out_gpu.png", envmap_w, envmap_h, 3, output_cuda, envmap_w * 3);
	stbi_write_png("out_cpu.png", envmap_w, envmap_h, 3, output_cpu_1_thread, envmap_w * 3);

	stbi_image_free(pixmap);
	free(output_cuda);
	free(output_cpu_1_thread);
	free(output_cpu_4_thread);
	free(output_cpu_8_thread);
}
