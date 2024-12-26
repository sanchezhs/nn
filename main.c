#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <raylib.h>

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600

#define MAX_EPOCHS 1000
#define NUM_SAMPLES 4
#define NUM_INPUTS 2
#define NUM_OUTPUTS 1

typedef struct
{
    float *inputs;
    size_t num_inputs;
    float *outputs;
    int num_outputs;
    float *weights;
    float bias;
    // Fields for training state
    int epoch;
    int max_epochs;
    int sample_index;
    int training_done;
    int missed;
} Perceptron;

static Color MY_GREEN = {
    .r = 20,
    .g = 215,
    .b = 96,
    .a = 255};

int and_matrix[4][2] = {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}};

int and_labels[] = {
    0, 0, 0, 1
    };

float random_float(int min, int max)
{
    return min + ((float)rand() / RAND_MAX) * (max - min);
}

void draw_axis()
{
    float x = WINDOW_WIDTH * 0.1;
    float y = WINDOW_HEIGHT * 0.9;

    // y-axis
    Vector2 y_start_pos = {
        .x = x,
        .y = WINDOW_HEIGHT - y};
    Vector2 y_end_pos = {
        .x = x,
        .y = y};
    DrawLineEx(y_start_pos, y_end_pos, 1.0f, MY_GREEN);

    // x-axis
    Vector2 x_end_pos = {
        .x = WINDOW_WIDTH - x,
        .y = y_end_pos.y};
    DrawLineEx(y_end_pos, x_end_pos, 1.0f, MY_GREEN);

    // Legend
    const char *y_axis = "Y";
    const char *x_axis = "X";
    int font_size = 18;
    int text_width = MeasureText(y_axis, font_size);
    DrawText(y_axis, x - text_width * 2, y_start_pos.y, font_size, MY_GREEN);
    DrawText(x_axis, x_end_pos.x - text_width * 2, x_end_pos.y + text_width, font_size, MY_GREEN);
    const char *zero = "(0,0)";
    text_width = MeasureText(zero, font_size);
    DrawText(zero, y_start_pos.x - text_width - 5, y_end_pos.y, font_size, MY_GREEN);
}

void draw_points(Vector2 *points, size_t num_points, bool show_legend, bool draw_points, Color line_color)
{
    if (num_points % 2 != 0)
    {
        fprintf(stderr, "ERROR: Number of points must be even in order to draw the lines\n");
        return;
    }

    float origin_x = WINDOW_WIDTH * 0.1;
    float origin_y = WINDOW_HEIGHT * 0.9;

    float point_size = 3.0f;

    // Scale factor to make the points more visible
    float scale = 400.0f;

    for (size_t i = 0; i < num_points; i += 2)
    {
        // Lines
        Vector2 a = points[i];
        Vector2 b = points[i + 1];

        Vector2 screen_a = {
            .x = origin_x + (a.x * scale),
            .y = origin_y - (a.y * scale)};

        Vector2 screen_b = {
            .x = origin_x + (b.x * scale),
            .y = origin_y - (b.y * scale)};
        DrawLineEx(screen_a, screen_b, 2.5f, line_color);

        // Points
        if (draw_points)
        {
            DrawCircleV(screen_b, point_size, WHITE);
            DrawCircleV(screen_a, point_size, WHITE);
        }

        // Legend
        if (show_legend)
        {
            char point_text[32];
            snprintf(point_text, sizeof(point_text), "(%.1f, %.1f)", points[i].x, points[i].y);
            int text_width = MeasureText(point_text, 18);
            DrawText(point_text, screen_a.x - text_width, screen_a.x + text_width * 0.7, 18, RED);
        }
    }
}

void initialize(Perceptron *p)
{
    p->num_inputs = NUM_INPUTS;
    p->weights = (float *)malloc(sizeof(float) * p->num_inputs);
    if (p->weights == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for weights\n");
        exit(1);
    }
    p->bias = random_float(-1, 1);
    for (size_t i = 0; i < p->num_inputs; i++)
    {
        p->weights[i] = random_float(-1, 1);
    }
    p->epoch = 0;
    p->max_epochs = MAX_EPOCHS;
    p->sample_index = 0;
    p->training_done = 0;
    p->missed = 0;
}

int classify(Perceptron *p, int *data, int label)
{
    float res = p->bias;
    for (size_t i = 0; i < p->num_inputs; i++)
    {
        res += p->weights[i] * data[i];
    }
    return res > 0 ? 1 : 0;
}

void train_step(Perceptron *p)
{
    if (p->training_done)
        return;

    int i = p->sample_index;

    int label = and_labels[i];
    int *row = and_matrix[i];
    float y = classify(p, row, label);

    if (y != label)
    {
        float learning_rate = 0.1f;
        p->bias += learning_rate * (label - y);
        for (size_t w = 0; w < p->num_inputs; w++)
        {
            p->weights[w] += learning_rate * (label - y) * row[w];
        }
        p->missed++;
    }

    if (p->num_inputs == 2)
    {
        float w0 = p->weights[0];
        float w1 = p->weights[1];
        float b = p->bias;

        Vector2 points[2];
        points[0] = (Vector2){0, -b / w1};
        points[1] = (Vector2){-b / w0, 0};
        draw_points(points, 2, false, false, BLUE);
    }

    p->sample_index++;
    if (p->sample_index >= NUM_SAMPLES)
    {
        if (p->missed == 0)
        {
            p->training_done = 1;
            printf("Training complete at epoch %d\n", p->epoch);
        }
        else if (p->epoch >= p->max_epochs)
        {
            p->training_done = 1;
            printf("Reached max epochs. Training stopped.\n");
        }
        else
        {
            p->epoch++;
            p->sample_index = 0;
            p->missed = 0;
        }
    }
}

void train(Perceptron *p)
{
    int max_epochs = 1000;
    int epoch = 0;

    while (epoch < max_epochs)
    {
        int missed = 0;
        for (size_t i = 0; i < NUM_SAMPLES; i++)
        {
            int label = and_labels[i];
            int *row = and_matrix[i];
            float y = classify(p, row, label);
            if (y != label)
            {
                float learning_rate = 0.1f;
                p->bias += learning_rate * (label - y);
                for (size_t w = 0; w < p->num_inputs; w++)
                {
                    p->weights[w] += learning_rate * (label - y) * row[w];
                }
                missed++;
            }

            if (p->num_inputs == 2)
            {
                float w0 = p->weights[0];
                float w1 = p->weights[1];
                float b = p->bias;

                Vector2 points[2];
                points[0] = (Vector2){0, -b / w1};
                points[1] = (Vector2){-b / w0, 0};
                draw_points(points, 2, false, false, BLUE);
            }
        }
        if (missed == 0)
        {
            break;
        }
        epoch++;
    }
}

int main(void)
{
    srand(time(0));
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Perceptron");

    Perceptron p = {0};
    initialize(&p);

    size_t num_points = 4;
    Vector2 and_points[] = {
        (Vector2){0, 1},
        (Vector2){1, 1},

        (Vector2){1, 1},
        (Vector2){1, 0},
    };

    while (!WindowShouldClose())
    {
        if (IsKeyPressed(KEY_SPACE) && !p.training_done)
        {
            train_step(&p);
        }

        BeginDrawing();
        ClearBackground((Color){0x18, 0x18, 0x18, 0xff});
        draw_axis();
        draw_points(and_points, num_points, true, true, RED);

        if (p.num_inputs == 2)
        {
            float w0 = p.weights[0];
            float w1 = p.weights[1];
            float b = p.bias;

            if (w1 != 0 && w0 != 0)
            {
                Vector2 points[2];
                points[0] = (Vector2){0, -b / w1};
                points[1] = (Vector2){-b / w0, 0};
                draw_points(points, 2, false, false, BLUE);
            }
        }

        char info[128];
        snprintf(info, sizeof(info), "Epoch: %d\nPress SPACE to train step", p.epoch);
        DrawText(info, 10, 10, 20, MY_GREEN);

        if (p.training_done)
        {
            DrawText("Training Completed", 10, 60, 20, RED);
        }
        EndDrawing();
    }
    CloseWindow();

    free(p.weights);

    return 0;
}