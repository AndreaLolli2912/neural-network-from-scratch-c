#include "dataset.h"
#include "utils.h"
#include "scaler.h"
#include "network.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(void) {
    Net net; init_net(&net);
    net_add_layer(&net, 9, 16);   // hidden (ReLU applied inside net_forward)
    net_add_layer(&net, 16, 10);  // output (logits)

    float x[9] = {0}; x[0] = 1.0f;
    float logits[10];
    if (!net_forward(&net, x, logits)) printf("forward failed\n");

    // (optional) turn logits into probabilities:
    float probs[10];
    softmax(logits, probs, 10);

    free_net(&net);
}
