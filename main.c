#include "dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
Dataset generation,
We will fill allocated arrays.
*/

int main()
{
    // define vars
    int nSamples = DATASET_SIZE;

    // set random seed
    srand(time(NULL));

    // allocate dataset memory
    unsigned char *images = malloc(IMAGE_SIZE * nSamples * sizeof(*images));
    unsigned char *labels = malloc(             nSamples * sizeof(*labels));

    // handle memory allocation failures
    if (!images || !labels) {
        printf("Memory allocation failed");
        exit(1);
    }

    // fill allocated memory
    generate_random_mnist(images, labels, nSamples);

    /*
    MATRICES CONTENT TESTS
    printf("images@%p, labels@%p\n", (void *)images, (void *)labels);
    printf("images 1st item = %d\n", *images);
    printf("labels 1st item = %d\n", *labels);

    printf("First image:\n");
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            printf("%3d ", images[row * 3 + col]);
        }
        printf("\n");
    }
    */

    /* NEVER FORGET TO FREE ALLOCATED MEMORY */
    free(images);
    free(labels);
}



