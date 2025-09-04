// Include guard:
// Prevents this header from being included multiple times in the same file,
// avoiding duplicate declarations and compiler errors.
#ifndef DATASET_H
#define DATASET_H

#define IMAGE_SIZE 9   // 28 x 28 pixels (784)
#define NUM_CLASSES 10   // digits 0–9
#define DATASET_SIZE 10 // how many images you want (100)

/* forward declarations */
void generate_random_images(unsigned char* images, int nSamples);
void generate_random_labels(unsigned char* labels, int nSamples);
void generate_random_mnist (unsigned char* images, unsigned char* labels, int nSamples);

#endif // DATASET_H
