#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "network.h"


const char* INITIALIZATION = "HE";


/* ================ ACTIVATION FUNCTIONS ================ */
void relu (float *x, int n)
{
    for (int i = 0; i < n; i++)
        x[i] = x[i] > 0 ? x[i] : 0.f;
}

void softmax (const float *logits, float *probs, int n)
{
    if (n <= 0) return;

    float exp_log;
    float max = logits[0];
    float sum = 0.f;

    // find maximum to prevent overflow
    for (int i = 0; i < n; i++)
        if (logits[i] > max) max = logits[i];

    // compute bounded sum, and update probs
    for (int i = 0; i < n; i++) {
        exp_log = expf(logits[i] - max);
        sum = sum + exp_log;
        probs[i] = exp_log;
    }
    if (sum == 0.0f) return;

    // normalize probs, with softmax formula
    for (int i = 0; i < n; i++)
        probs[i] = probs[i] / sum;
}

/* ================ STRUCTURES ================ */
/*
typedef struct Layer {
    float *weights, *biases;
    int input_size, output_size;
} Layer;

typedef struct Net {
    Layer *layers;
    int n_layers;
    int capacity;
} Net;
*/
/* ================ NET PRIVATE METHODS ================ */
int net_reserve(Net *net, int cap) {
    if (net->capacity >= cap) return 1;
    Layer *tmp = realloc(net->layers, cap * sizeof *tmp);  // store return value
    if (!tmp) return 0;                                    // leave net unchanged on fail
    net->layers   = tmp;
    net->capacity = cap;
    return 1;
}


/* ================ NET PUBLIC METHODS ================ */
int net_size(const Net *self) { return self->n_layers; }

Layer *net_get_layer(Net *self, int i) {
    if (i < 0 || i >= self->n_layers) return NULL;
    return &self->layers[i];
}

const Layer *net_get_layer_const(const Net *self, int i) {
    if (i < 0 || i >= self->n_layers) return NULL;
    return &self->layers[i];
}

int net_add_layer(Net *net, int in_size, int out_size)
{
    if (!net) return 0;

    // shape check (if not the first layer)
    if (net->n_layers > 0) {
        int prev_out = net->layers[net->n_layers - 1].output_size;
        if (in_size != prev_out) {
            printf("net_add_layer: shape mismatch (got %d, expected %d)\n", in_size, prev_out);
            return 0;
        }
    }

    // ensure space
    if (net->n_layers == net->capacity) {
        int new_cap = net->capacity ? net->capacity * 2 : 4;
        if (!net_reserve(net, new_cap)) {
            printf("net_add_layer: out of memory while growing to %d\n", new_cap);
            return 0;
        }
    }

    // place-construct layer in the next free slot
    Layer *slot = &net->layers[net->n_layers];
    init_layer(slot, in_size, out_size);

    net->n_layers++;
    return 1;
}
/* ================ CONSTRUCTORS ================ */
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

void init_net(Net *self)
{
    self->layers = NULL;
    self->n_layers = 0;
    self->capacity = 0;
}

/* ================ DE-CONSTRUCTORS ================ */
void free_layer (Layer *self)
{
    free(self->biases);
    free(self->weights);
    self->biases = self->weights = NULL;
    self->input_size = self->output_size = 0;

}

void free_net(Net *self)
{
    if (!self) return;

    for (int i = 0; i < self->n_layers; ++i) {
        free_layer(&self->layers[i]);   // pass pointer
    }

    free(self->layers);
    self->layers   = NULL;
    self->n_layers = 0;
    self->capacity = 0;
}


/* ================ FORWARD PASS ================*/

void layer_affine_forward(const Layer *L, const float *x, float *out)
{
    const int in  = L->input_size;
    const int outn = L->output_size;
    for (int i = 0; i < outn; i++) {
        float sum = L->biases[i];
        const float *wrow = &L->weights[i * in];
        for (int j = 0; j < in; j++) {
            sum += wrow[j] * x[j];
        }
        out[i] = sum;
    }
}

int net_forward(const Net *net, const float *x, float *out)
{
    if (!net || net->n_layers == 0 || !x || !out) return 0;

    // Find the largest layer width to size temporary buffers
    int maxw = 0;
    for (int k = 0; k < net->n_layers; ++k) {
        if (net->layers[k].output_size > maxw) maxw = net->layers[k].output_size;
    }
    if (maxw <= 0) return 0;

    float *bufA = malloc((size_t)maxw * sizeof *bufA);
    float *bufB = malloc((size_t)maxw * sizeof *bufB);
    if (!bufA || !bufB) { free(bufA); free(bufB); return 0; }

    const float *cur_in = x;
    float *cur_out = (net->n_layers == 1) ? out : bufA;

    for (int k = 0; k < net->n_layers; ++k) {
        const Layer *L = &net->layers[k];

        // Basic shape guard for the very first layer
        if (k == 0 && L->input_size <= 0) { free(bufA); free(bufB); return 0; }

        // Affine: out = W x + b
        layer_affine_forward(L, cur_in, cur_out);

        const int is_last = (k == net->n_layers - 1);
        if (!is_last) {
            // Hidden activation
            relu(cur_out, L->output_size);

            // Next layer input/output selection (ping-pong buffers)
            cur_in = cur_out;
            cur_out = (k == net->n_layers - 2) ? out : (cur_out == bufA ? bufB : bufA);
        }
    }

    free(bufA);
    free(bufB);
    return 1;
}
