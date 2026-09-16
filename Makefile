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

  STATIC ?= 1
  ifeq ($(STATIC),1)
    CRTFLAG := /MT
  else
    CRTFLAG := /MD
  endif

  CFLAGS  := /nologo /O2 /std:c11 /fp:fast /openmp:experimental $(CRTFLAG)
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

  # -march=native & -mtune=native isn't understood everywhere, so probe for
  # it instead of hard-coding it and breaking the build on those targets.
  MARCH_NATIVE := $(shell $(CC) -march=native -E -x c /dev/null >/dev/null 2>&1 && echo -march=native)
  MTUNE_NATIVE := $(shell $(CC) -mtune=native -E -x c /dev/null >/dev/null 2>&1 && echo -mtune=native)

  # Same story for OpenMP: not every Clang install ships libomp out of the
  # box (notably Xcode's clang on macOS), so probe rather than assume, and
  # fall back to a single-threaded build with a warning instead of failing.
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

  # Unoptimized build with debug symbols and no OpenMP (makes stepping through
  # with gdb/lldb sane, -Ofast + threads turns every backtrace into soup ;D).
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
