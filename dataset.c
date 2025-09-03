/* main file entry for the dataset generation functions */
#include "dataset.h"

#include <stdlib.h>
#include <time.h>

/* generate random images */
void generate_random_images(unsigned char** images, int *nSamples)
{
    *nSamples = DATASET_SIZE; // number of images
    *images   = malloc((*nSamples) * IMAGE_SIZE); // allocate memory

    if (!*images) {
        printf("Memory allocation failed for images\n");
        exit(1);
    }

    srand(time(NULL)); // seed RNG

    for (int i = 0; i < (*nSamples) * IMAGE_SIZE; i++) {
        (*images)[i] = rand() % 256; // pixel values 0-255
    }
}

void generate_random_labels(unsigned char **labels, int *nSamples)
{
    *nSamples = DATASET_SIZE; // number of labels
    *labels = malloc(*nSamples); //allocate memory

    if (!*labels) {
        printf("Memory allocation failed for labels\n");
        exit(1);
    }

    for (int i = 0, i < (*nSamples); i++)
        (*labels)[i] = rand() % NUM_CLASSES;
}
