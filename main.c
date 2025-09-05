#include "dataset.h"
#include "utils.h"
#include "scaler.h"
#include "network.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>


static int feq(float a, float b, float eps) { return fabsf(a-b) <= eps; }

static void test_affine_known() {
    Layer L;
    init_layer(&L, 3, 2);

    // row-major by output: row 0 then row 1
    // W = [[1, 2, 3],
    //      [-1, 0.5, 0]]
    L.weights[0] = 1.0f;  L.weights[1] = 2.0f;  L.weights[2] = 3.0f;
    L.weights[3] = -1.0f; L.weights[4] = 0.5f;  L.weights[5] = 0.0f;
    L.biases[0] = 0.1f; L.biases[1] = -0.2f;

    float x[3] = {1.0f, 2.0f, -1.0f};
    float y[2] = {0};
    layer_affine_forward(&L, x, y);
    // expected: [2.1, -0.2]
    assert(feq(y[0],  2.1f, 1e-5f));
    assert(feq(y[1], -0.2f, 1e-5f));
    free_layer(&L);
    printf("[OK] affine_known\n");
}

static void test_relu_softmax() {
    float v[4] = {-2.0f, 0.0f, 3.0f, -0.1f};
    relu(v, 4);
    assert(feq(v[0], 0.0f, 0)); assert(feq(v[1], 0.0f, 0));
    assert(feq(v[2], 3.0f, 0)); assert(feq(v[3], 0.0f, 0));

    float logits[3] = {1.0f, 2.0f, 3.0f}, p[3];
    softmax(logits, p, 3);
    float s = p[0]+p[1]+p[2];
    assert(fabsf(s - 1.0f) < 1e-5f);
    assert(p[2] > p[1] && p[1] > p[0]); // preserves order
    printf("[OK] relu_softmax\n");
}

static void test_builder_and_shapes() {
    Net net; init_net(&net);
    assert(net_size(&net) == 0);

    // add consistent layers
    assert(net_add_layer(&net, 9, 16) == 1);
    assert(net_add_layer(&net, 16, 10) == 1);
    assert(net_size(&net) == 2);

    // mismatch must fail
    assert(net_add_layer(&net, 99, 2) == 0);

    free_net(&net);
    printf("[OK] builder_shapes\n");
}

static void test_forward_single_layer() {
    Net net; init_net(&net);
    assert(net_add_layer(&net, 4, 3) == 1);

    // zero weights, non-zero biases → output == biases
    Layer *L = net_get_layer(&net, 0);
    for (int i = 0; i < L->input_size * L->output_size; ++i) L->weights[i] = 0.0f;
    L->biases[0] = 0.5f; L->biases[1] = -1.0f; L->biases[2] = 2.0f;

    float x[4] = {10, -3, 7, 0.25f}; // should be ignored (weights zero)
    float logits[3] = {0};
    assert(net_forward(&net, x, logits) == 1);
    assert(feq(logits[0], 0.5f, 1e-6f));
    assert(feq(logits[1], -1.0f, 1e-6f));
    assert(feq(logits[2], 2.0f, 1e-6f));

    free_net(&net);
    printf("[OK] forward_single\n");
}

static void test_forward_two_layers_pingpong() {
    Net net; init_net(&net);
    assert(net_add_layer(&net, 3, 2) == 1);
    assert(net_add_layer(&net, 2, 2) == 1);

    // L0: output = ReLU(bias) since weights=0
    Layer *L0 = net_get_layer(&net, 0);
    for (int i = 0; i < L0->input_size * L0->output_size; ++i) L0->weights[i] = 0.0f;
    L0->biases[0] = 1.0f; L0->biases[1] = -2.0f; // after ReLU → [1, 0]

    // L1: logits = [[2,0],[0,3]] * [1,0] + [0,0] = [2,0]
    Layer *L1 = net_get_layer(&net, 1);
    // row0: [2,0], row1: [0,3]
    L1->weights[0]=2.0f; L1->weights[1]=0.0f;
    L1->weights[2]=0.0f; L1->weights[3]=3.0f;
    L1->biases[0]=0.0f;  L1->biases[1]=0.0f;

    float x[3] = {42, -7, 0.5f}; // ignored by L0 weights=0
    float logits[2] = {0};
    assert(net_forward(&net, x, logits) == 1);
    assert(feq(logits[0], 2.0f, 1e-6f));
    assert(feq(logits[1], 0.0f, 1e-6f));

    // Optional: softmax check
    float p[2]; softmax(logits, p, 2);
    float s = p[0]+p[1];
    assert(fabsf(s - 1.0f) < 1e-6f);
    assert(p[0] > p[1]);

    free_net(&net);
    printf("[OK] forward_two_layers\n");
}

int main(void) {
    test_affine_known();
    test_relu_softmax();
    test_builder_and_shapes();
    test_forward_single_layer();
    test_forward_two_layers_pingpong();
    printf("All tests passed ✅\n");
    return 0;
}
