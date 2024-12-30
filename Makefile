CC = gcc
CFLAGS = -Wall -Wextra
LDFLAGS = -lraylib -lm

NN_DIR = nn
BIN_DIR = bin

DIGITS_SRC = $(NN_DIR)/digits.c
NN_SRC = $(NN_DIR)/nn.c
NN_HEADER = $(NN_DIR)/nn.h
PERCEPTRON_SRC = $(NN_DIR)/perceptron.c

DIGITS_OUT = $(BIN_DIR)/digits
PERCEPTRON_OUT = $(BIN_DIR)/perceptron

all: $(BIN_DIR) $(DIGITS_OUT) $(PERCEPTRON_OUT)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(DIGITS_OUT): $(DIGITS_SRC) $(NN_SRC) $(NN_HEADER) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(DIGITS_SRC) $(NN_SRC) -o $(DIGITS_OUT) $(LDFLAGS)

$(PERCEPTRON_OUT): $(PERCEPTRON_SRC) $(NN_HEADER) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(PERCEPTRON_SRC) -o $(PERCEPTRON_OUT) $(LDFLAGS)

clean:
	rm -rf $(BIN_DIR)