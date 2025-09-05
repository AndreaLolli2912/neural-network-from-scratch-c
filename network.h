#ifndef NETWORK_H
#define NETWORK_H


/* ================ ACTIVATION FUNCTIONS ================ */
void relu (float *x, int n);
void softmax (const float *logits, float *probs, int n);

/* ================ STRUCTURES ================ */
typedef struct Layer {
    float *weights, *biases;
    int input_size, output_size;
} Layer;

typedef struct Net {
    Layer *layers;
    int n_layers;
    int capacity;
} Net;

/* ================ NET PRIVATE METHODS ================ */
int net_reserve(Net *net, int cap);

/* ================ NET PUBLIC METHODS ================ */
int net_size(const Net *self);
Layer *net_get_layer(Net *self, int i);
const Layer *net_get_layer_const(const Net *self, int i);
int net_add_layer(Net *net, int in_size, int out_size);

/* ================ CONSTRUCTORS ================ */
void init_layer (Layer *layer, int in_size, int out_size);
void init_net(Net *self);

/* ================ DE-CONSTRUCTORS ================ */
void free_layer (Layer *self);
void free_net(Net *self);

/* ================ FORWARD PASS ================ */
void layer_affine_forward(const Layer *L, const float *x, float *out);
int net_forward(const Net *net, const float *x, float *out);

#endif // NETWORK_H
