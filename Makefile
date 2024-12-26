CC = gcc
CFLAGS = -Wall -Wextra
LDFLAGS = -lraylib -lm

main:
	$(CC) $(CFLAGS) main.c -o main $(LDFLAGS)