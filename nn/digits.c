#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include "nn.h"


int main(void)
{
    const char *train_images_path = "./datasets/Digits/train-images.idx3-ubyte";
    const char *train_labels_path = "./datasets/Digits/train-labels.idx1-ubyte";
    const char *test_images_path = "./datasets/Digits/t10k-images.idx3-ubyte";
    const char *test_labels_path = "./datasets/Digits/t10k-labels.idx1-ubyte";

    Dataset train_dataset = load_dataset(train_images_path, train_labels_path);
    Dataset test_dataset = load_dataset(test_images_path, test_labels_path);

    size_t num_layers = 3;
    size_t input_size = 784; // MNIST input size
    size_t output_size = 10; // MNIST output size (0-9 digits)
    size_t layer_sizes[] = {128, 16, output_size};
    ActivationType activations[] = {RELU, RELU, SOFTMAX};

    NeuralNetwork *nn = build_nn(num_layers, input_size, output_size, layer_sizes, activations);
    train(nn, &train_dataset);

    size_t *confusion_matrix = (size_t *)calloc(nn->output_size*nn->output_size, sizeof(size_t));
    test(nn, &test_dataset, confusion_matrix);

    print_confusion_matrix(confusion_matrix, nn->output_size);
    calculate_metrics(confusion_matrix, nn->output_size);

    free_nn(nn);
    free_dataset(&train_dataset);
    free_dataset(&test_dataset);

    return 0;
}