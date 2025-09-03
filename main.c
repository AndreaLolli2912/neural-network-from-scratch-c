#include <stdio.h>
#include <stdlib.h>

#define IMAGE_SIZE 9   // 28 x 28 pixels
#define NUM_CLASSES 2   // digits 0–9
#define DATASET_SIZE 10 // how many images you want

int main()
{
    int i, j;

    // allocate memory
    unsigned char *images = malloc(IMAGE_SIZE * DATASET_SIZE * sizeof(*images));
    unsigned char *labels = malloc(DATASET_SIZE * sizeof(*labels));
    if (!images || !labels) {
        printf("Failed memory allocation for images OR labels\n");
        exit(1);
    }

    // fill with rand values
    for (i = 0; i < DATASET_SIZE; i++) {
        labels[i] = rand() % NUM_CLASSES;
        for (j = 0; j < IMAGE_SIZE; j++) {
            images[ i * IMAGE_SIZE + j] = rand() % 256;
        }
    }

    printf("images@%p, labels@%p\n", (void *)images, (void *)labels);
    printf("images 1st item = %d\n", *images);
    printf("labels 1st item = %d\n", *labels);

    /* NEVER FORGET TO FREE ALLOCATED MEMORY */
    free(images);
    free(labels);
}



