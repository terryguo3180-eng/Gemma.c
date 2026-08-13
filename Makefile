CC = gcc
CFLAGS = -Wall -Wextra -Ofast -fopenmp -flto -march=native -mtune=native
LDFLAGS = -fopenmp -flto -march=native -mtune=native
LDLIBS = -lgomp -lpthread -lm
SOURCES = gemma.c
OBJECTS = $(SOURCES:.c=.o)

.PHONY: all clean

$(info $(OS))

ifeq ($(OS),Windows_NT)
	RM = del /Q
	RMDIR = rmdir /S /Q
	NULL = 2>nul
	TARGET = gemma.exe
	LDFLAGS += -static -static-libgcc -static-libstdc++
	LDLIBS += -lmingw32 -lmingwex -lucrtbase -lkernel32 -lmsvcrt
else
	RM = rm -f
	RMDIR = rm -rf
	NULL = 2>/dev/null
	TARGET = gemma
	LDFLAGS += -static -static-libgcc -static-libstdc++
endif

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) $^ -o $@ -Wl,-Bstatic $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJECTS) $(TARGET) $(NULL)
	$(RMDIR) $(NULL) 2>nul || true
