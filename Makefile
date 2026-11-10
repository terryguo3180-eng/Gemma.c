# ---------------------------------------------------------------------------
#
# Makefile for gemma.c
#
# Just run `make`. Works out of the box with GCC, Clang, and MSVC (cl), on
# Linux, macOS, and Windows (MinGW-w64 / MSYS2 / Developer Command Prompt).
#
# Useful overrides:
#
#   make CC=clang           # force a specific compiler
#   make CC=cl              # force MSVC (from an MSVC developer shell)
#   make DTYPE=BF16         # compile weights as bfloat16 instead of float32
#   make debug              # -O0 -g build for use with gdb/lldb (GCC/Clang)
#
# MSVC example (x64 Developer Command Prompt / vcvarsall x64):
#
#   make CC=cl DTYPE=FP32
#
# ---------------------------------------------------------------------------

TARGET := gemma
SRC    := gemma.c

CC ?= cc

# Detect MSVC: CC is cl / cl.exe, or the binary reports Microsoft's banner.
CC_BASENAME := $(notdir $(CC))
CC_VERSION_STRING := $(shell $(CC) 2>&1)
IS_MSVC := $(if $(filter cl cl.exe,$(CC_BASENAME)),1,$(if $(findstring Microsoft,$(CC_VERSION_STRING)),1,))
IS_CLANG := $(if $(findstring clang,$(shell $(CC) --version 2>/dev/null)),1,)

ifeq ($(OS),Windows_NT)
  TARGET := gemma.exe
endif

# ---------------------------------------------------------------------------
# MSVC
# ---------------------------------------------------------------------------
ifeq ($(IS_MSVC),1)

  # MSVC has no native _Float16 / __bf16; default to FP32 unless the user
  # overrides DTYPE on the command line.
  ifeq ($(strip $(DTYPE)),)
    DTYPE := FP32
  endif

  STATIC ?= 1
  ifeq ($(STATIC),1)
    CRTFLAG := /MT
  else
    CRTFLAG := /MD
  endif

  # MSYS2 rewrites arguments that start with "/" into paths, which breaks
  # every cl switch; turn that off for the recipes below.
  export MSYS2_ARG_CONV_EXCL := *

  # MSVC has no -march=native, so find out whether this CPU can run AVX2 by
  # compiling a probe and running it: pre-2014 Intel parts and pre-2017 AMD
  # parts fault on AVX2, which make sees as a failed probe.
  AVX2_PROBE := gemma_avx2_probe
  $(file >$(AVX2_PROBE).c,#include <immintrin.h>)
  $(file >>$(AVX2_PROBE).c,int main(void) { __m256i z = _mm256_setzero_si256();)
  $(file >>$(AVX2_PROBE).c,  return _mm256_extract_epi32(_mm256_add_epi32(z, z), 0); })
  AVX2_FLAG  := $(shell $(CC) /nologo /O1 /arch:AVX2 /Fe:$(AVX2_PROBE).exe \
                  $(AVX2_PROBE).c > $(AVX2_PROBE).log 2>&1 && \
                cmd /c $(AVX2_PROBE).exe && echo /arch:AVX2)
  $(shell rm -f $(AVX2_PROBE).c $(AVX2_PROBE).exe $(AVX2_PROBE).obj \
          $(AVX2_PROBE).log 2>/dev/null || \
          del /Q $(AVX2_PROBE).c $(AVX2_PROBE).exe $(AVX2_PROBE).obj \
          $(AVX2_PROBE).log 2>NUL)

  # /arch:AVX2 is what enables the vectorized GEMM kernels (MSVC doesn't
  # define __AVX2__ / __FMA__ without it)
  CFLAGS  := /nologo /O2 /std:c11 /fp:fast $(AVX2_FLAG) \
             /openmp:experimental $(CRTFLAG)
  CFLAGS  += /DDTYPE=$(DTYPE)
  LDFLAGS :=
  LDLIBS  := shell32.lib

  .PHONY: all debug clean

  all: $(TARGET)

  $(TARGET): $(SRC)
	$(CC) $(CFLAGS) $< /Fe:$@ /link $(LDLIBS)

  # MSVC debug: /Od /Zi, no OpenMP so single-stepping stays sane.
  debug: CFLAGS := /nologo /Od /Zi /std:c11 /DDTYPE=$(DTYPE) $(CRTFLAG)
  debug: LDLIBS := shell32.lib
  debug: $(TARGET)

  clean:
	-del /Q gemma.exe gemma.obj gemma.pdb gemma.ilk 2>NUL
	-rm -f gemma gemma.exe gemma.obj gemma.pdb gemma.ilk

# ---------------------------------------------------------------------------
# GCC / Clang (Linux, macOS, MinGW)
# ---------------------------------------------------------------------------
else

  OPT := -O3 -ffast-math

  # -march=native & -mtune=native isn't understood everywhere, so probe for it
  MARCH_NATIVE := $(shell $(CC) -march=native -E -x c /dev/null >/dev/null 2>&1 && echo -march=native)
  MTUNE_NATIVE := $(shell $(CC) -mtune=native -E -x c /dev/null >/dev/null 2>&1 && echo -mtune=native)

  # Same story for OpenMP
  OPENMP := $(shell $(CC) -fopenmp -E -x c /dev/null >/dev/null 2>&1 && echo -fopenmp)
  ifeq ($(OPENMP),)
    $(warning OpenMP not found for '$(CC)', building single-threaded.)
    $(warning Install gcc's libgomp, or on macOS 'brew install libomp', to enable multithreading.)
  endif

  WARNFLAGS := -Wall -Wextra
  CFLAGS    := -std=c11 $(WARNFLAGS) $(OPT) $(MARCH_NATIVE) $(MTUNE_NATIVE) $(OPENMP)
  
  ifeq ($(OS),Windows_NT)
    STATIC ?= 1
  else
    STATIC ?= 0
  endif

  ifeq ($(STATIC),1)
    LDLIBS := -static
  else
    LDLIBS :=
  endif

  ifneq ($(strip $(DTYPE)),)
    CFLAGS += -DDTYPE=$(DTYPE)
  endif

  ifeq ($(OS),Windows_NT)
    ifeq ($(IS_MSVC),1)
      LDLIBS += shell32.lib
    else
      LDLIBS += -lshell32
    endif
  else
    LDLIBS += -lm
    ifeq ($(IS_CLANG),1)
      LDLIBS += -fopenmp=libgomp
    else
      LDLIBS += -lgomp
    endif
  endif

  .PHONY: all debug clean

  all: $(TARGET)

  $(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) $< -o $@ $(LDLIBS)

  # Unoptimized build with debug symbols and no OpenMP
  debug: CFLAGS := -std=gnu11 $(WARNFLAGS) -O0 -g -fsanitize=address,undefined
  debug: LDFLAGS := -fsanitize=address,undefined
  debug: LDLIBS := -lm
  debug: $(TARGET)
    
  ifeq ($(OS),Windows_NT)
  clean:
		-del /Q gemma gemma.exe 2>NUL
		-rm -f gemma gemma.exe 2>NUL
  else
  clean:
		rm -f gemma gemma.exe
  endif

endif
