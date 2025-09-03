#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Include guard:
// Prevents this header from being included multiple times in the same file,
// avoiding duplicate declarations and compiler errors.
#ifndef DATASET_H
#define DATASET_H

#define IMAGE_SIZE 784   // 28 x 28 pixels
#define NUM_CLASSES 10   // digits 0–9
#define DATASET_SIZE 1000 // how many images you want

/* forward declarations */
void generate_random_images (unsigned char **images, int *nSamples); // TODO: investigate 'double *'
void generate_random_labels (unsigned char **labels, int *nSamples);


#endif // DATASET_H
