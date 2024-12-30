#include "nn.h"

float random_float()
{
    return (float)rand() / (float)RAND_MAX;
}

float relu(float x)
{
    return x > 0 ? x : 0;
}

float relu_derivative(float x)
{
    return x > 0 ? 1.0f : 0.0f;
}

float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

float sigmoid_derivative(float x)
{
    return x * (1.0f - x);
}

float softmax(float *outputs, size_t size, size_t index)
{
    float max_output = outputs[0];
    for (size_t i = 1; i < size; i++)
    {
        if (outputs[i] > max_output)
        {
            max_output = outputs[i];
        }
    }

    float sum = 0.0f;
    for (size_t i = 0; i < size; i++)
    {
        sum += exp(outputs[i] - max_output);
    }

    return exp(outputs[index] - max_output) / sum;
}

float xavier_init(size_t fan_in, size_t fan_out)
{
    float limit = sqrtf(6.0f / (fan_in + fan_out));
    return (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * limit;
}

Neuron *build_neuron(size_t num_weights)
{
    Neuron *n = (Neuron *)malloc(sizeof(Neuron));
    n->weights = (float *)malloc(sizeof(float) * num_weights);
    n->num_weights = num_weights;
    n->bias = 0.0f;
    n->output = 0.0f;
    n->delta = 0.0f;

    for (size_t i = 0; i < num_weights; i++)
    {
        n->weights[i] = xavier_init(num_weights, num_weights);
    }

    return n;
}

void free_neuron(Neuron *n)
{
    free(n->weights);
    free(n);
}

Layer *build_layer(size_t num_neurons, size_t num_weights, LayerType type, ActivationType activation)
{
    Layer *l = (Layer *)malloc(sizeof(Layer));
    l->num_neurons = num_neurons;
    l->type = type;
    l->activation = activation;
    l->neurons = (Neuron **)malloc(sizeof(Neuron *) * num_neurons);
    l->outputs = (float *)malloc(sizeof(float) * num_neurons);

    for (size_t i = 0; i < num_neurons; i++)
    {
        l->neurons[i] = build_neuron(num_weights);
    }

    return l;
}

void free_layer(Layer *l)
{
    for (size_t i = 0; i < l->num_neurons; i++)
    {
        free_neuron(l->neurons[i]);
    }
    free(l->neurons);
    free(l->outputs);
    free(l);
}

NeuralNetwork *build_nn(size_t num_layers, size_t input_size, size_t output_size, size_t *layer_sizes, ActivationType *activations)
{
    NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    nn->num_layers = num_layers;
    nn->learning_rate = 0.01f;
    nn->input_size = input_size;
    nn->output_size = output_size;
    nn->layers = (Layer **)malloc(sizeof(Layer *) * num_layers);

    for (size_t i = 0; i < num_layers; i++)
    {
        size_t num_weights = (i == 0) ? input_size : layer_sizes[i - 1];
        nn->layers[i] = build_layer(layer_sizes[i], num_weights, (i == 0) ? INPUT_LAYER : ((i == num_layers - 1) ? OUTPUT_LAYER : HIDDEN_LAYER), activations[i]);
    }

    return nn;
}

void free_nn(NeuralNetwork *nn)
{
    for (size_t i = 0; i < nn->num_layers; i++)
    {
        free_layer(nn->layers[i]);
    }
    free(nn->layers);
    free(nn);
}

void forward_propagation(NeuralNetwork *nn, float *input)
{
    for (size_t l = 0; l < nn->num_layers; l++)
    {
        Layer *layer = nn->layers[l];

        float *prev_outputs = NULL;

        if (l == 0)
        {
            // The first layer sees the input
            prev_outputs = input;
        }
        else
        {
            // The next layers see the outputs from the previous layer
            prev_outputs = nn->layers[l - 1]->outputs;
        }

        // 1) Compute raw sums
        for (size_t n = 0; n < layer->num_neurons; n++)
        {
            Neuron *neuron = layer->neurons[n];
            float sum = 0.0f;
            // Weighted sum of the previous outputs
            for (size_t w = 0; w < neuron->num_weights; w++)
            {
                sum += neuron->weights[w] * prev_outputs[w];
            }
            sum += neuron->bias;
            layer->outputs[n] = sum;
        }

        // 2) Apply activation function
        if (layer->activation == SOFTMAX)
        {
            float max_output = layer->outputs[0];
            for (size_t n = 1; n < layer->num_neurons; n++)
            {
                if (layer->outputs[n] > max_output)
                {
                    max_output = layer->outputs[n];
                }
            }

            float sum_exp = 0.0f;
            for (size_t n = 0; n < layer->num_neurons; n++)
            {
                layer->outputs[n] = expf(layer->outputs[n] - max_output);
                sum_exp += layer->outputs[n];
            }

            for (size_t n = 0; n < layer->num_neurons; n++)
            {
                float val = layer->outputs[n] / sum_exp;
                layer->outputs[n] = val;
                layer->neurons[n]->output = val;
            }
        }
        else
        {
            for (size_t n = 0; n < layer->num_neurons; n++)
            {
                float val = layer->outputs[n];
                switch (layer->activation)
                {
                case RELU:
                    val = relu(val);
                    break;
                case SIGMOID:
                    val = sigmoid(val);
                    break;
                default:
                    break;
                }
                layer->outputs[n] = val;
                layer->neurons[n]->output = val;
            }
        }
    }
}

void backward_propagation(NeuralNetwork *nn, float *expected_output)
{
    for (size_t l = nn->num_layers; l-- > 0;)
    {
        Layer *layer = nn->layers[l];
        Layer *next_layer = (l == nn->num_layers - 1) ? NULL : nn->layers[l + 1];

        for (size_t n = 0; n < layer->num_neurons; n++)
        {
            Neuron *neuron = layer->neurons[n];

            if (l == nn->num_layers - 1)
            {
                if (layer->activation == SOFTMAX)
                {
                    neuron->delta = neuron->output - expected_output[n];
                }
                else
                {
                    float error = expected_output[n] - neuron->output;
                    if (layer->activation == SIGMOID)
                    {
                        neuron->delta = error * sigmoid_derivative(neuron->output);
                    }
                    else if (layer->activation == RELU)
                    {
                        neuron->delta = error * relu_derivative(neuron->output);
                    }
                }
            }
            else
            {
                float error = 0.0f;
                for (size_t k = 0; k < next_layer->num_neurons; k++)
                {
                    error += next_layer->neurons[k]->weights[n] * next_layer->neurons[k]->delta;
                }
                if (layer->activation == SIGMOID)
                {
                    neuron->delta = error * sigmoid_derivative(neuron->output);
                }
                else if (layer->activation == RELU)
                {
                    neuron->delta = error * relu_derivative(neuron->output);
                }
            }
        }
    }
}

void update_weights(NeuralNetwork *nn, float *input, float learning_rate)
{
    for (size_t l = 0; l < nn->num_layers; l++)
    {
        Layer *layer = nn->layers[l];

        float *prev_outputs = NULL;
        if (l == 0)
        {
            // The first layer sees the raw input
            prev_outputs = input;
        }
        else
        {
            prev_outputs = nn->layers[l - 1]->outputs;
        }

        for (size_t n = 0; n < layer->num_neurons; n++)
        {
            Neuron *neuron = layer->neurons[n];
            for (size_t w = 0; w < neuron->num_weights; w++)
            {
                // Gradient descent step
                neuron->weights[w] -= learning_rate * neuron->delta * prev_outputs[w];
            }
            neuron->bias -= learning_rate * neuron->delta;
        }
    }
}

void load_images(const char *file_path, DigitImage **images, size_t *num_images, size_t *width, size_t *height)
{
    FILE *file = fopen(file_path, "rb");
    if (!file)
    {
        fprintf(stderr, "Error while opening file %s:\n", file_path);
        exit(EXIT_FAILURE);
    }

    int magic_number;
    if (fread(&magic_number, sizeof(int), 1, file) != 1)
    {
        fprintf(stderr, "Error reading magic number from file %s.\n", file_path);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    magic_number = __builtin_bswap32(magic_number);

    if (magic_number != 0x00000803)
    {
        fprintf(stderr, "Wrong magic number for images: %d\n", magic_number);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    if (fread(num_images, sizeof(int), 1, file) != 1 ||
        fread(width, sizeof(int), 1, file) != 1 ||
        fread(height, sizeof(int), 1, file) != 1)
    {
        fprintf(stderr, "Error reading header from file %s.\n", file_path);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    *num_images = __builtin_bswap32(*num_images);
    *width = __builtin_bswap32(*width);
    *height = __builtin_bswap32(*height);

    size_t image_size = (*width) * (*height);
    *images = (DigitImage *)malloc((*num_images) * sizeof(DigitImage));
    if (!*images)
    {
        fprintf(stderr, "Error allocating memory for images.\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < *num_images; ++i)
    {
        (*images)[i].data = (float *)malloc(image_size * sizeof(float));
        if (!(*images)[i].data)
        {
            fprintf(stderr, "Error allocating memory for image data.\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }

        unsigned char *buffer = (unsigned char *)malloc(image_size);
        if (!buffer)
        {
            fprintf(stderr, "Error allocating memory for image buffer.\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }

        if (fread(buffer, sizeof(unsigned char), image_size, file) != image_size)
        {
            fprintf(stderr, "Error reading image data from file %s.\n", file_path);
            fclose(file);
            exit(EXIT_FAILURE);
        }

        for (size_t j = 0; j < image_size; ++j)
        {
            (*images)[i].data[j] = buffer[j] / 255.0f;
        }
        (*images)[i].width = *width;
        (*images)[i].height = *height;

        free(buffer);
    }

    fclose(file);
    printf("INFO: Loaded %zu images of size %zux%zu from %s.\n", *num_images, *width, *height, file_path);
}

void load_labels(const char *file_path, DigitImage *images, size_t num_images)
{
    FILE *file = fopen(file_path, "rb");
    if (!file)
    {
        fprintf(stderr, "ERROR: Could not open file %s.\n", file_path);
        exit(EXIT_FAILURE);
    }

    int magic_number;
    fread(&magic_number, sizeof(int), 1, file);
    magic_number = __builtin_bswap32(magic_number);

    if (magic_number != 0x00000801)
    {
        fprintf(stderr, "Wrong magic number for labels: %d\n", magic_number);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    int num_labels_int;
    if (fread(&num_labels_int, sizeof(int), 1, file) != 1)
    {
        fprintf(stderr, "Error reading number of labels.\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    num_labels_int = __builtin_bswap32(num_labels_int);

    for (size_t i = 0; i < num_images; ++i)
    {
        fread(&images[i].label, sizeof(unsigned char), 1, file);
    }

    fclose(file);
}

Dataset load_dataset(const char *images_path, const char *labels_path)
{
    Dataset dataset;
    size_t width, height;

    load_images(images_path, &dataset.images, &dataset.num_images, &width, &height);
    load_labels(labels_path, dataset.images, dataset.num_images);

    return dataset;
}

void free_dataset(Dataset *dataset)
{
    for (size_t i = 0; i < dataset->num_images; i++)
    {
        free(dataset->images[i].data);
    }
    free(dataset->images);
}

size_t get_predicted_label(NeuralNetwork *nn)
{
    Layer *output_layer = nn->layers[nn->num_layers - 1];
    size_t predicted_label = 0;
    float max_output = output_layer->neurons[0]->output;

    for (size_t i = 1; i < output_layer->num_neurons; i++)
    {
        if (output_layer->neurons[i]->output > max_output)
        {
            max_output = output_layer->neurons[i]->output;
            predicted_label = i;
        }
    }

    return predicted_label;
}

void swap_images(DigitImage *a, DigitImage *b)
{
    DigitImage temp = *a;
    *a = *b;
    *b = temp;
}

void shuffle_dataset(Dataset *dataset)
{
    srand((unsigned int)time(NULL));
    for (size_t i = 0; i < dataset->num_images - 1; i++)
    {
        // Pick a random index from i..(num_images-1)
        size_t j = i + rand() / (RAND_MAX / (dataset->num_images - i) + 1);
        swap_images(&dataset->images[i], &dataset->images[j]);
    }
}

void train(NeuralNetwork *nn, Dataset *dataset)
{
    for (size_t epoch = 0; epoch < MAX_EPOCHS; epoch++)
    {
        printf("INFO: Epoch %zu started.\n", epoch);

        shuffle_dataset(dataset);

        float total_loss = 0.0f;

        for (size_t i = 0; i < dataset->num_images; i++)
        {
            DigitImage *image = &dataset->images[i];

            // 1) Forward pass
            forward_propagation(nn, image->data);

            // Construct the "one-hot" expected output
            float expected_output[10] = {0.0f};
            expected_output[image->label] = 1.0f;

            // 2) Backward pass
            backward_propagation(nn, expected_output);

            // 3) Calculate the loss for logging
            Layer *output_layer = nn->layers[nn->num_layers - 1];
            float sample_loss = 0.0f;
            for (int c = 0; c < 10; c++)
            {
                if (expected_output[c] > 0.0f)
                {
                    sample_loss -= logf(output_layer->neurons[c]->output + 1e-7f);
                }
            }
            total_loss += sample_loss;

            // 4) Update weights immediately for this sample (SGD step)
            update_weights(nn, image->data, nn->learning_rate);
        }

        // Average loss across the entire dataset for this epoch
        total_loss /= (float)dataset->num_images;
        printf("INFO: Epoch %zu completed. Loss: %.6f\n", epoch, total_loss);
        total_loss = 0.0f;
    }
}

void train_and_log_loss(NeuralNetwork *nn, Dataset *dataset, const char *loss_file)
{
    FILE *file = fopen(loss_file, "w");
    if (!file)
    {
        fprintf(stderr, "ERROR: Unable to open file %s for writing.\n", loss_file);
        return;
    }

    shuffle_dataset(dataset);
    float total_loss = 0.0f;

    for (size_t i = 0; i < dataset->num_images; i++)
    {
        DigitImage *image = &dataset->images[i];

        // 1) Forward pass
        forward_propagation(nn, image->data);

        // Construct the "one-hot" expected output
        float expected_output[10] = {0.0f};
        expected_output[image->label] = 1.0f;

        // 2) Backward pass
        backward_propagation(nn, expected_output);

        // 3) Calculate the loss for this image
        Layer *output_layer = nn->layers[nn->num_layers - 1];
        float sample_loss = 0.0f;
        for (int c = 0; c < 10; c++)
        {
            if (expected_output[c] > 0.0f)
            {
                sample_loss -= logf(output_layer->neurons[c]->output + 1e-7f);
            }
        }
        total_loss += sample_loss;

        // 4) Update weights
        update_weights(nn, image->data, nn->learning_rate);

        // Log the loss after this image
        fprintf(file, "%zu\t%.6f\n", i, sample_loss);

        printf("INFO: Image %zu completed. Loss: %.6f\n", i, sample_loss);
    }

    fclose(file);
    printf("INFO: Loss values saved to %s.\n", loss_file);
}

void test(NeuralNetwork *nn, Dataset *dataset, size_t *confusion_matrix)
{
    size_t correct = 0;

    if (!confusion_matrix)
    {
        fprintf(stderr, "Error: Unable to allocate memory for confusion matrix.\n");
        return;
    }

    for (size_t i = 0; i < dataset->num_images; i++)
    {
        DigitImage *image = &dataset->images[i];
        forward_propagation(nn, image->data);

        // Get the predicted label (index of the maximum output value)
        size_t predicted_label = 0;
        float max_output = nn->layers[nn->num_layers - 1]->neurons[0]->output;

        for (size_t j = 1; j < nn->output_size; j++)
        {
            if (nn->layers[nn->num_layers - 1]->neurons[j]->output > max_output)
            {
                max_output = nn->layers[nn->num_layers - 1]->neurons[j]->output;
                predicted_label = j;
            }
        }

        // Update the confusion matrix
        confusion_matrix[image->label * 10 + predicted_label]++;

        if (predicted_label == image->label)
        {
            correct++;
        }
    }
}

void print_confusion_matrix(size_t *confusion_matrix, size_t num_classes)
{
    // Print header row
    printf("Confusion Matrix:\n   ");
    for (size_t i = 0; i < num_classes; i++)
    {
        printf("%4zu", i);
    }
    printf("\n");

    for (size_t i = 0; i < num_classes; i++)
    {
        // Print row label
        printf("%2zu ", i);
        for (size_t j = 0; j < num_classes; j++)
        {
            printf("%4zu", confusion_matrix[i * num_classes + j]);
        }
        printf("\n");
    }
}

void calculate_metrics(size_t *confusion_matrix, size_t num_classes)
{
    printf("Class\tAccuracy\tPrecision\tRecall\t\tF1-score\n");
    for (size_t i = 0; i < num_classes; i++)
    {
        size_t true_positives = confusion_matrix[i * num_classes + i];
        size_t false_positives = 0;
        size_t false_negatives = 0;

        for (size_t j = 0; j < num_classes; j++)
        {
            if (j != i)
            {
                false_positives += confusion_matrix[j * num_classes + i];
                false_negatives += confusion_matrix[i * num_classes + j];
            }
        }

        float accuracy = (float)true_positives / (true_positives + false_positives + false_negatives + 1e-7f);
        float precision = (float)true_positives / (true_positives + false_positives + 1e-7f);
        float recall = (float)true_positives / (true_positives + false_negatives + 1e-7f);
        float f1_score = 2.0f * (precision * recall) / (precision + recall + 1e-7f);

        printf("%zu\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n", i, accuracy, precision, recall, f1_score);
    }
}
