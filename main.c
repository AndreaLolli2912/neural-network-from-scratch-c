#include "dataset.h"
#include "utils.h"
#include "scaler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
Dataset generation,
We will fill allocated arrays.
*/

int STANDARDIZE = 1;


void test_dataset_conversion_and_scaling(unsigned char *images_uc, int nSamples) {
    printf("=== Testing Dataset Conversion & Scaling ===\n");

    // Allocate float memory
    float *images_float = malloc(IMAGE_SIZE * nSamples * sizeof(*images_float));
    if (!images_float) {
        printf("Memory allocation failed for images_float\n");
        return;
    }

    // Convert to float
    convert_img_to_float(images_uc, images_float, nSamples);

    // Print first image (raw float values)
    printf("\nFirst image as float (raw):\n");
    for (int i = 0; i < IMAGE_SIZE; i++) {
        printf("%6.1f ", images_float[i]);
        if ((i+1) % 9 == 0) printf("\n"); // adjust for small IMAGE_SIZE
    }

    // Apply MinMaxScaler
    float *images_std = malloc(IMAGE_SIZE * nSamples * sizeof(*images_std));
    if (!images_std) {
        printf("Memory allocation failed for images_std\n");
        free(images_float);
        return;
    }

    MinMaxScaler *scaler = MinMaxScaler_new();
    MinMaxScaler_fit_transform(scaler, images_float, images_std, nSamples);

    // Print first image after scaling
    printf("\nFirst image after MinMax scaling [0,1]:\n");
    for (int i = 0; i < IMAGE_SIZE; i++) {
        printf("%0.3f ", images_std[i]);
        if ((i+1) % 9 == 0) printf("\n");
    }


    // Free memory
    free(images_float);
    free(images_std);
    MinMaxScaler_del(scaler);
    printf("\n=== Test Complete ===\n");
}

int main() {
    int nSamples = DATASET_SIZE;
    srand(time(NULL));

    unsigned char *images = malloc(IMAGE_SIZE * nSamples * sizeof(*images));
    unsigned char *labels = malloc(             nSamples * sizeof(*labels));
    generate_random_mnist(images, labels, nSamples);

    // Run the test
    test_dataset_conversion_and_scaling(images, nSamples);

    // Clean up
    free(images);
    free(labels);
    return 0;
}



