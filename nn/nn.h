// Header File: nn.h
#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

#define MAX_EPOCHS 1
#define BATCH_SIZE 64

// Activation Functions
typedef enum { RELU, SIGMOID, SOFTMAX } ActivationType;
float relu(float x);
float relu_derivative(float x);
float sigmoid(float x);
float sigmoid_derivative(float x);

// Layer Type
typedef enum { INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER } LayerType;

// Neuron Structure
typedef struct {
    float *weights;
    float bias;
    float output;
    float delta;
    size_t num_weights;
} Neuron;

Neuron *build_neuron(size_t num_weights);
void free_neuron(Neuron *n);

// Layer Structure
typedef struct Layer
{
    size_t num_neurons;
    LayerType type;
    ActivationType activation;
    Neuron **neurons;
    float *outputs;
} Layer;

Layer *build_layer(size_t num_neurons, size_t num_weights, LayerType type, ActivationType activation);
void free_layer(Layer *l);

// Neural Network Structure
typedef struct {
    Layer **layers;
    size_t num_layers;
    float learning_rate;
    size_t input_size;
    size_t output_size;
} NeuralNetwork;

// Dataset Structure
typedef struct {
    float *data;
    size_t width;
    size_t height;
    unsigned char label;
} DigitImage;

typedef struct {
    DigitImage *images;
    size_t num_images;
} Dataset;

// Neural Network Functions
NeuralNetwork *build_nn(size_t num_layers, size_t input_size, size_t output_size, size_t *layer_sizes, ActivationType *activations);
void free_nn(NeuralNetwork *nn);
void forward_propagation(NeuralNetwork *nn, float *input);
void backward_propagation(NeuralNetwork *nn, float *expected_output);
void update_weights(NeuralNetwork *nn, float *input, float learning_rate);
void train(NeuralNetwork *nn, Dataset *dataset);
void train_and_log_loss(NeuralNetwork *nn, Dataset *dataset, const char *loss_file);
void test(NeuralNetwork *nn, Dataset *dataset, size_t *confusion_matrix);
void calculate_metrics(size_t *confusion_matrix, size_t num_classes);
void print_confusion_matrix(size_t *confusion_matrix, size_t num_classes);

// Utility Functions
size_t get_predicted_label(NeuralNetwork *nn);
void swap_images(DigitImage *a, DigitImage *b);
void shuffle_dataset(Dataset *dataset);

// Initialization Functions
float xavier_init(size_t fan_in, size_t fan_out);

// Activation Functions
float relu(float x);
float relu_derivative(float x);
float sigmoid(float x);
float sigmoid_derivative(float x);

// Dataset Functions
Dataset load_dataset(const char *images_path, const char *labels_path);
void load_labels(const char *file_path, DigitImage *images, size_t num_images);
void load_images(const char *file_path, DigitImage **images, size_t *num_images, size_t *width, size_t *height);
void free_dataset(Dataset *dataset);

#endif // NN_H
