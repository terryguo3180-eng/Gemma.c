CC = gcc
CFLAGS = -Wall -Wextra -Ofast -fopenmp -static -march=native
SOURCES = gemma.c
OBJECTS = $(SOURCES:.c=.o)

.PHONY: all clean

ifeq ($(OS),Windows_NT)
    RM = del /Q
    RMDIR = rmdir /S /Q
    NULL = 2>nul
	TARGET = gemma.exe
else
    RM = rm -f
    RMDIR = rm -rf
    NULL = 2>/dev/null
	TARGET = gemma
endif

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ -lmsvcrt

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJECTS) $(TARGET) $(NULL)
