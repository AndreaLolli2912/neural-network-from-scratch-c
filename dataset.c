/*
Main file entry for the dataset generation functions.
User must allocate memory autonomously and pass pointer.
Functions access memory allocation and fill with rand values.
*/
#include "dataset.h"
#include <stdio.h>
#include <stdlib.h>

void generate_random_images(unsigned char* images, int nSamples)
{
    for (int i = 0; i < nSamples * IMAGE_SIZE; i++)
        images[i] = rand() % 256; // values bounded [0, 255]
}

void generate_random_labels(unsigned char* labels, int nSamples)
{
    for (int i = 0; i < nSamples; i++)
        labels[i] = rand() % NUM_CLASSES;
}

void generate_random_mnist(unsigned char *images, unsigned char *labels, int nSamples)
{
    // handle memory allocation failures
    if (!images || !labels) {
        printf("Memory allocation failed");
        exit(1);
    }

    // fill images
    generate_random_images(images, nSamples);
    generate_random_labels(labels, nSamples);
}
