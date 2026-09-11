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
#   make DTYPE=BF16         # compile weights as bfloat16 instead of fp16
#   make DTYPE=FP32         # recommended default when building with MSVC
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

  # /openmp:experimental enables omp simd; /openmp:llvm (VS 2022+) is better
  # when available but experimental is the widely-working choice you used.
  CFLAGS  := /nologo /O2 /std:c11 /openmp:experimental /D_CRT_SECURE_NO_WARNINGS
  CFLAGS  += /DDTYPE=$(DTYPE)
  LDFLAGS :=
  LDLIBS  := shell32.lib

  .PHONY: all debug clean

  all: $(TARGET)

  $(TARGET): $(SRC)
	$(CC) $(CFLAGS) $< /Fe:$@ /link $(LDLIBS)

  # MSVC debug: /Od /Zi, no OpenMP so single-stepping stays sane.
  debug: CFLAGS := /nologo /Od /Zi /std:c11 /D_CRT_SECURE_NO_WARNINGS /DDTYPE=$(DTYPE)
  debug: LDLIBS := shell32.lib
  debug: $(TARGET)

  clean:
	-del /Q gemma.exe gemma.obj gemma.pdb gemma.ilk 2>NUL
	-rm -f gemma gemma.exe gemma.obj gemma.pdb gemma.ilk

# ---------------------------------------------------------------------------
# GCC / Clang (Linux, macOS, MinGW)
# ---------------------------------------------------------------------------
else

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

endif