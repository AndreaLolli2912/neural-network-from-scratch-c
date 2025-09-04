#include <stdlib.h>
#include <stdio.h>

// Private struct definitions
typedef struct Scaler {
    void (*fit)(void* self, const float* data, int n);
    void (*transform)(void* self, const float* input, float* output, int n);
    void (*fit_transform)(void* self, const float* input, float* output, int n);
} Scaler;

typedef struct MinMaxScaler {
    float __min, __max;
    Scaler base;
} MinMaxScaler;

// Private static functions
static void __MinMaxScaler_fit(void* self, const float* data, int n) {
    MinMaxScaler* sc = (MinMaxScaler*)self;
    sc->__min = sc->__max = data[0];
    for (int i = 0; i < n; i++) {
        if (data[i] < sc->__min) sc->__min = data[i];
        if (data[i] > sc->__max) sc->__max = data[i];
    }
}

static void __MinMaxScaler_transform(void* self, const float* input, float* output, int n) {
    MinMaxScaler* sc = (MinMaxScaler*)self;
    for (int i = 0; i < n; i++) {
        output[i] = (input[i] - sc->__min) / (sc->__max - sc->__min);
    }
}

static void __MinMaxScaler_fit_transform(void* self, const float* input, float* output, int n) {
    __MinMaxScaler_fit(self, input, n);
    __MinMaxScaler_transform(self, input, output, n);
}

// Constructor / Destructor
MinMaxScaler* MinMaxScaler_new(void) {
    MinMaxScaler* sc = malloc(sizeof(*sc));
    if (!sc) return NULL;

    sc->__min = 0.0f;
    sc->__max = 0.0f;

    sc->base.fit = &__MinMaxScaler_fit;
    sc->base.transform = &__MinMaxScaler_transform;
    sc->base.fit_transform = &__MinMaxScaler_fit_transform;

    return sc;
}

void MinMaxScaler_del(void* self) {
    free(self);
}

// Public wrapper methods
void MinMaxScaler_fit(MinMaxScaler* self, const float* data, int n) {
    self->base.fit(self, data, n);
}

void MinMaxScaler_transform(MinMaxScaler* self, const float* input, float* output, int n) {
    self->base.transform(self, input, output, n);
}

void MinMaxScaler_fit_transform(MinMaxScaler* self, const float* input, float* output, int n) {
    self->base.fit_transform(self, input, output, n);
}
