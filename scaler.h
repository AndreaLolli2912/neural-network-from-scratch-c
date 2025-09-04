#ifndef SCALER_H
#define SCALER_H

typedef struct MinMaxScaler MinMaxScaler;

// Constructor / Destructor
MinMaxScaler* MinMaxScaler_new(void);
void MinMaxScaler_del(void* self);

// Public methods
void MinMaxScaler_fit(MinMaxScaler* self, const float* data, int n);
void MinMaxScaler_transform(MinMaxScaler* self, const float* input, float* output, int n);
void MinMaxScaler_fit_transform(MinMaxScaler* self, const float* input, float* output, int n);

#endif // SCALER_H
