#include "dataset.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>

const char* INITIALIZATION = "HE";

typedef struct {
    float *weights, *biases;
    int input_size, output_size;
} Layer;

void init_layer (Layer *layer, int in_size, int out_size)
{
    int n = in_size * out_size;

    // weights memory allocation
    layer->weights = malloc(n * sizeof(float));
    layer->biases  = malloc(out_size * sizeof(float));

    // memory allocation failure check
    if (!layer->weights || !layer->biases) {
        fprintf(stderr, "init_layer: malloc failed\n");
        exit(1);
    }

    // store in and out sizes
    layer->input_size  = in_size;
    layer->output_size = out_size;

    // weights initialization
    if (strcmp(INITIALIZATION, "HE") == 0) {
        float scale = sqrtf(2.0f / (float)in_size);
        for (int i = 0; i < n; ++i) {
            float u = (float)rand() / (float)RAND_MAX; // [0,1]
            layer->weights[i] = (u - 0.5f) * 2.0f * scale; // [-scale, +scale]
        }
        for (int o = 0; o < out_size; ++o) layer->biases[o] = 0.0f;
    }
}
