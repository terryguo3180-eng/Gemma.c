# ---------------------------------------------------------------------------
#
# Makefile for gemma.c
#
# Just run `make`. Works out of the box with both GCC and Clang, on
# Linux, macOS, and Windows (MinGW-w64 / MSYS2).
#
# Useful overrides:
#
#   make CC=clang           # force a specific compiler
#   make DTYPE=BF16         # compile weights as bfloat16 instead of fp16
#   make debug              # -O0 -g build for use with gdb/lldb
#
# ---------------------------------------------------------------------------

TARGET := gemma
SRC    := gemma.c

CC ?= cc

# `cc` resolves to whatever the system default is (gcc on Linux, clang
# on macOS/BSD). We still need to know which one we ended up with, because
# a couple of flags below only make sense for one of the two.
CC_VERSION_STRING := $(shell $(CC) --version 2>/dev/null)
IS_CLANG := $(if $(findstring clang,$(CC_VERSION_STRING)),1,)

# -Ofast completely breaks the code in Clang and I'm not sure why
ifeq ($(IS_CLANG),1)
  OPT := -O3
else
  OPT := -Ofast
endif

# -march=native isn't understood everywhere (Apple Silicon Clang, some
# distro-patched cross-compilers, emulated CI runners), so probe for it
# instead of hard-coding it and breaking the build on those targets.
MARCH_NATIVE := $(shell $(CC) -march=native -E -x c /dev/null >/dev/null 2>&1 && echo -march=native)

# Same story for OpenMP: not every Clang install ships libomp out of the
# box (notably Xcode's clang on macOS), so probe rather than assume, and
# fall back to a single-threaded build with a warning instead of failing.
OPENMP := $(shell $(CC) -fopenmp -E -x c /dev/null >/dev/null 2>&1 && echo -fopenmp)
ifeq ($(OPENMP),)
  $(warning OpenMP not found for '$(CC)', building single-threaded.)
  $(warning Install gcc's libgomp, or on macOS 'brew install libomp', to enable multithreading.)
endif

WARNFLAGS := -Wall -Wextra -Wno-unused-parameter
CFLAGS    := -std=gnu11 $(WARNFLAGS) $(OPT) $(MARCH_NATIVE) $(OPENMP)
LDLIBS    := -lm

ifneq ($(strip $(DTYPE)),)
  CFLAGS += -DDTYPE=$(DTYPE)
endif

ifeq ($(OS),Windows_NT)
  TARGET := gemma.exe
  # Statically link the runtime so the .exe doesn't need libgcc /
  # libwinpthread DLLs sitting next to it to run on a machine without MinGW
  # installed.
  LDFLAGS += -static
else
  ifeq ($(OPENMP),-fopenmp)
    LDLIBS += -lgomp
  endif
endif

.PHONY: all debug clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) $< -o $@ $(LDLIBS)

# Unoptimized build with debug symbols and no OpenMP (makes stepping through
# with gdb/lldb sane, -Ofast + threads turns every backtrace into soup ;D).
debug: CFLAGS := -std=gnu11 $(WARNFLAGS) -O0 -g -fsanitize=address,undefined
debug: LDFLAGS := -fsanitize=address,undefined
debug: LDLIBS := -lm
debug: $(TARGET)

clean:
	rm -f gemma gemma.exe
