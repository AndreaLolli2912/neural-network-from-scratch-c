#include "dataset.h"

void convert_img_to_float (unsigned char *mem_char, float *mem_float, int nSamples)
{
    for (int i = 0; i < IMAGE_SIZE * nSamples; i++) {
        mem_float[i] = mem_char[i];
    }
}

