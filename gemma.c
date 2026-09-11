/* Gemma 1, 2 and 3 implemented in a single file of pure C.
 *
 * This is a from-scratch inference engine for Google's Gemma family of
 * language models: Gemma 1, 2 and 3, text-only and multimodal (SigLIP vision
 * tower included), in one .c file, no third-party dependencies, builds with
 * a single gcc/clang invocation. It is not a wrapper around llama.cpp or
 * ggml, the transformer, the attention masks, the RoPE tables, the int8
 * kernels, the BPE tokenizer, and the image decoder / resizer are all
 * self-included here. I did this mostly to actually understand how Gemma
 * works end to end, and the by-product is a codebase that's small enough to
 * read in an afternoon. It's inspired by karpathy/llama2.c, which is the
 * same idea applied to Llama.
 *
 * Features:
 *
 *   - Gemma 1 / 2 / 3, text-only and multimodal, single file, no deps
 *   - Hand-written GEMM/GEMV kernels (fpx and int8), packed/tiled, OpenMP
 *     across heads/rows
 *   - W8A8 quantization (per-channel weights, per-token activations) with a
 *     one-shot `-q` export flag
 *   - Hybrid sliding-window / full attention (Gemma 2/3) with a
 *     FlashAttention-style causal tiling scheme
 *   - Full SigLIP vision encoder with bidirectional image-token attention,
 *     plus a pan & scan crop utility for high-aspect-ratio images
 *   - mmap'd weight loading (falls back to a plain read on `--disable-mmap`)
 *   - KV-cache streaming generation, chunked prefill
 *   - Temperature / top-k / top-p / repetition-penalty sampling
 *   - float32 / float16 / bfloat16 weights, selectable at compile time
 *   - A from-scratch BPE tokenizer, weights embedded straight into the
 *     `.bin` file by the export script
 *   - Runs on Linux, macOS, and Windows (MinGW), gcc or clang
 *
 * Model files are a custom binary format produced by the accompanying
 * `export.py`, which converts a HuggingFace Gemma checkpoint (weights,
 * config, tokenizer vocab + BPE merges) into this inference-ready layout,
 * applying the activation rescaling described above and, optionally, W8A8
 * quantization.
 *
 * Build it:
 *
 *   ```bash
 *   make
 *   ```
 *
 * `make` auto-detects gcc vs. clang and picks safe flags for each (see the
 * long comment about -ffast-math above for why that distinction matters).
 * If you'd rather invoke the compiler directly:
 *
 *   ```bash
 *   gcc   -Ofast -march=native -fopenmp gemma.c -lm -o gemma   # gcc
 *   clang -O3    -march=native -fopenmp gemma.c -lm -o gemma   # clang
 *   ```
 *
 * Weight dtype defaults to float16; override with -DDTYPE at compile time
 * (FP32 / FP16 / BF16), or `make DTYPE=BF16`:
 *
 *   ```bash
 *   gcc -Ofast -march=native -fopenmp -DDTYPE=BF16 gemma.c -lm -o gemma
 *   ```
 *
 * (This has to match the dtype the .bin file was exported with.)
 *
 * Basic usage:
 *
 *   ```bash
 *   ./gemma model.bin --prompt "Hello I'm a language model, "
 *   ```
 *
 * Interactive chat, maintaining history across turns with Gemma's own
 * <start_of_turn>/<end_of_turn> template:
 *
 *   ```bash
 *   ./gemma model.bin --chat
 *   ```
 *
 * Multimodal models take images inline via @image{...}:
 *
 *   ```bash
 *   ./gemma model.bin --prompt "Looking at @image{photo.jpg}, we can see"
 *   ```
 *
 *   ```
 *   > Describe what you see in @image{photo.jpg}.
 *   ```
 *
 * The image gets decoded, resized to the model's expected input size, run
 * through the vision encoder, and the resulting soft tokens get spliced into
 * the text sequence between <start_of_image>/<end_of_image>. Pass
 * `--disable-mm` to skip loading the vision tower entirely and save memory on
 * a text-only workload.
 *
 * For very wide/tall images, @image_pas{...} runs Gemma 3's pan & scan crop
 * utility first, so the model gets both the full downsized image and a few
 * higher-resolution crops of it:
 *
 *   ```
 *   > What's happening at the bottom of @image_pas{long_screenshot.png}?
 *   ```
 *
 * `--seqlen`, `--temperature`, `--topk`, `--topp`, `--rpen`, `--seed` and
 * friends cover the usual sampling knobs; run `--help` / `-?` for the full
 * list.
 *
 * Code layout, roughly top to bottom: cross-platform shims (mmap emulation,
 * UTF-8 console handling) -> embedded stb_image(_resize2) blob -> config
 * structs and the .bin (de)serialization -> BPE tokenizer -> GEMM/GEMV
 * kernels (fpx and int8) -> SigLIP vision forward pass -> Gemma transformer
 * forward pass (single-token decode, chunked prefill, and the combined
 * text+vision path) -> sampling -> chat template + CLI + main loop. The
 * matmul kernels in particular are written to be swappable in isolation if
 * you want to drop in something faster (BLAS, a hand-tuned SIMD kernel,
 * whatever) without touching the model code around them.
 *
 * v1.0  (09/11/2026)
 *
 * @TerryGuo (https://github.com/terryguo3180-eng | terry.guo2021@outlook.com)
 */

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

// A bunch of hairy cross-platform code, unrelated to the inference engine
#ifdef _WIN32
#  include <io.h>
#  include <time.h>
#  include <windows.h>

// Global interruption flag
static volatile LONG g_interrupted = 0;

static inline void
set_interrupted(void)
{
  InterlockedExchange(&g_interrupted, 1);
}

static inline int
is_interrupted(void)
{
  return InterlockedCompareExchange(&g_interrupted, 0, 0) != 0;
}

// mmap support on Windows

#  define O_RDONLY     _O_RDONLY
#  define O_WRONLY     _O_WRONLY
#  define O_RDWR       _O_RDWR
#  define O_APPEND     _O_APPEND
#  define O_CREAT      _O_CREAT
#  define O_TRUNC      _O_TRUNC
#  define O_EXCL       _O_EXCL
#  define O_TEXT       _O_TEXT
#  define O_BINARY     _O_BINARY
#  define O_RAW        _O_BINARY
#  define O_TEMPORARY  _O_TEMPORARY
#  define O_NOINHERIT  _O_NOINHERIT
#  define O_SEQUENTIAL _O_SEQUENTIAL
#  define O_RANDOM     _O_RANDOM

#  define open   _open
#  define close  _close
#  define strdup _strdup

#  define PROT_NONE  0
#  define PROT_READ  1
#  define PROT_WRITE 2
#  define PROT_EXEC  4

#  define MAP_FILE      0
#  define MAP_SHARED    1
#  define MAP_PRIVATE   2
#  define MAP_TYPE      0xf
#  define MAP_FIXED     0x10
#  define MAP_ANONYMOUS 0x20
#  define MAP_FAILED    ((void *)-1)

#  ifndef FILE_MAP_EXECUTE
#    define FILE_MAP_EXECUTE 0x0020
#  endif

/* */
static uint32_t
__map_mmap_prot_page(const int prot)
{
  uint32_t protect = 0;
  if (prot == PROT_NONE) return protect;
  if ((prot & PROT_EXEC) != 0)
  {
    protect =
      ((prot & PROT_WRITE) != 0) ? PAGE_EXECUTE_READWRITE : PAGE_EXECUTE_READ;
  }
  else
  {
    protect = ((prot & PROT_WRITE) != 0) ? PAGE_READWRITE : PAGE_READONLY;
  }
  return protect;
}
/* */
static uint32_t
__map_mmap_prot_file(const int prot)
{
  uint32_t desired_acc = 0;
  if (prot == PROT_NONE)
  {
    return desired_acc;
  }
  if ((prot & PROT_READ) != 0)
  {
    desired_acc |= FILE_MAP_READ;
  }
  if ((prot & PROT_WRITE) != 0)
  {
    desired_acc |= FILE_MAP_WRITE;
  }
  if ((prot & PROT_EXEC) != 0)
  {
    desired_acc |= FILE_MAP_EXECUTE;
  }
  return desired_acc;
}
/* */
void *
mmap(void *addr, size_t len, int prot, int flags, int fildes, int64_t off)
{
  (void)addr;

  HANDLE fm, h;
  void  *map = MAP_FAILED;

#  ifdef _MSC_VER
#    pragma warning(push)
#    pragma warning(disable : 4293)
#  endif

  const uint32_t dw_fileoff_low  = (uint32_t)(off & 0xFFFFFFFFL);
  const uint32_t dw_fileoff_high = (uint32_t)((off >> 32) & 0xFFFFFFFFL);
  const uint32_t protect         = __map_mmap_prot_page(prot);
  const uint32_t desired_acc     = __map_mmap_prot_file(prot);
  const int64_t  maxsize         = off + (int64_t)len;
  const uint32_t dw_maxsize_low  = (uint32_t)(maxsize & 0xFFFFFFFFL);
  const uint32_t dw_maxsize_high = (uint32_t)((maxsize >> 32) & 0xFFFFFFFFL);

#  ifdef _MSC_VER
#    pragma warning(pop)
#  endif

  errno = 0;
  if (len == 0
      // Unsupported flag combinations
      || (flags & MAP_FIXED) != 0
      // Unsupported protection combinations
      || prot == PROT_EXEC)
  {
    errno = EINVAL;
    return MAP_FAILED;
  }
  h = ((flags & MAP_ANONYMOUS) == 0) ? (HANDLE)_get_osfhandle(fildes)
                                     : INVALID_HANDLE_VALUE;
  if ((flags & MAP_ANONYMOUS) == 0 && h == INVALID_HANDLE_VALUE)
  {
    errno = EBADF;
    return MAP_FAILED;
  }
  fm =
    CreateFileMapping(h, NULL, protect, dw_maxsize_high, dw_maxsize_low, NULL);
  if (fm == NULL)
  {
    errno = GetLastError();
    return MAP_FAILED;
  }
  map = MapViewOfFile(fm, desired_acc, dw_fileoff_high, dw_fileoff_low, len);
  CloseHandle(fm);
  if (map == NULL)
  {
    errno = GetLastError();
    return MAP_FAILED;
  }
  return map;
}
/* */
int
munmap(void *addr, size_t len)
{
  (void)len;

  if (UnmapViewOfFile(addr)) return 0;
  errno = GetLastError();
  return -1;
}
/* */
int
mprotect(void *addr, size_t len, int prot)
{
  uint32_t new_prot = __map_mmap_prot_page(prot);
  DWORD    old_prot = 0;
  if (VirtualProtect(addr, len, new_prot, &old_prot)) return 0;
  errno = GetLastError();
  return -1;
}
/* */
int
msync(void *addr, size_t len, int flags)
{
  (void)flags;

  if (FlushViewOfFile(addr, len)) return 0;
  errno = GetLastError();
  return -1;
}
/* */
int
mlock(const void *addr, size_t len)
{
  if (VirtualLock((LPVOID)addr, len)) return 0;
  errno = GetLastError();
  return -1;
}
/* */
int
munlock(const void *addr, size_t len)
{
  if (VirtualUnlock((LPVOID)addr, len)) return 0;
  errno = GetLastError();
  return -1;
}

// UTF-8 console encoding support on Windows

/* */
static void
set_utf8_console(void)
{
  // Set encoding to UTF-8
  SetConsoleOutputCP(65001);
  SetConsoleCP(65001);
}

/* Convert Windows command line to UTF-8 argc/argv */
static char **
get_utf8_argv(int *argc_out)
{
  wchar_t **wargv = CommandLineToArgvW(GetCommandLineW(), argc_out);
  if (!wargv) return NULL;

  char **argv = malloc((*argc_out + 1) * sizeof(char *));
  if (!argv)
  {
    LocalFree(wargv);
    return NULL;
  }

  for (int i = 0; i < *argc_out; i++)
  {
    int size =
      WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, NULL, 0, NULL, NULL);
    argv[i] = malloc(size);
    if (!argv[i])
    {
      for (int j = 0; j < i; j++)
      {
        free(argv[j]);
      }

      free(argv);
      LocalFree(wargv);
      return NULL;
    }
    WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, argv[i], size, NULL, NULL);
  }
  argv[*argc_out] = NULL;

  LocalFree(wargv);
  return argv;
}

/* */
static void
free_utf8_argv(char **argv, int argc)
{
  if (argv == NULL) return;
  for (int i = 0; i < argc; i++)
  {
    free(argv[i]);
  }
  free(argv);
}

// Interruption handling on Windows

#  define SLEEP_SEC(sec) Sleep((sec) * 1000)
#  define GETPID()       GetCurrentProcessId()
/* */
BOOL WINAPI
console_handler_(DWORD dwCtrlType)
{
  switch (dwCtrlType)
  {
    case CTRL_C_EVENT:         // Ctrl+C
    case CTRL_BREAK_EVENT:     // Ctrl+Break
    case CTRL_CLOSE_EVENT:     // Closed console window
    case CTRL_LOGOFF_EVENT:    // User logoff
    case CTRL_SHUTDOWN_EVENT:  // System shutdown
      set_interrupted();
      return TRUE;
    default:
      return FALSE;
  }
}

#else
#  undef _POSIX_C_SOURCE
#  define _POSIX_C_SOURCE 199309L

#  include <sys/mman.h>
#  include <time.h>
#  include <unistd.h>

#  define SLEEP_SEC(sec) sleep(sec)
#  define GETPID()       getpid()

static volatile sig_atomic_t g_interrupted = 0;

static inline void
set_interrupted(void)
{
  g_interrupted = 1;
}

static inline int
is_interrupted(void)
{
  return g_interrupted != 0;
}

/* */
static void
set_utf8_console(void)
{
}

/* */
static char **
get_utf8_argv(int *argc_out)
{
  (void)argc_out;
  return NULL;
}

/* */
static void
free_utf8_argv(char **argv, int argc)
{
  (void)argv;
  (void)argc;
}
#endif  // _WIN32

/* */
static void
signal_handler(int signum)
{
  (void)signum;
  set_interrupted();
}

/* */
void
setup_signal_handler(void)
{
#ifdef _WIN32
  if (!SetConsoleCtrlHandler(console_handler_, TRUE))
  {
    // Should almost never happen
    signal(SIGINT, signal_handler);
  }
#else
  struct sigaction sa;
  sa.sa_handler = signal_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;

  if (sigaction(SIGINT, &sa, NULL) == -1)
  {
    perror("sigaction(SIGINT)");
  }
  if (sigaction(SIGTERM, &sa, NULL) == -1)
  {
    perror("sigaction(SIGTERM)");
  }
#endif  // _WIN32
}

// Cross-platform wall time

/* */
static double
now_sec(void)
{
#ifdef _WIN32
  static LARGE_INTEGER freq;
  static int           freq_init = 0;
  LARGE_INTEGER        counter;

  if (!freq_init)
  {
    QueryPerformanceFrequency(&freq);
    freq_init = 1;
  }
  QueryPerformanceCounter(&counter);
  return (double)counter.QuadPart / (double)freq.QuadPart;
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

/* This giant blob is a self-contained, pre-processed implementation of
 * "stb_image" and "stb_image_resize2". :D
 *
 * I want to keep this project in a single .c file, to avoid requiring users to
 * obtain the original stb headers, and to keep the source code free of tens of
 * thousands of lines of third-party code, I just took the source code of the
 * two libraries, compressed / minified them into a "code blob" and embedded
 * directly here. The functions declared `extern` remain visible. (This blob
 * polluted the global namespace quite seriously, but I'm good with that ;-)
 *
 * The blob is equivilant to the following code if you have the two headers:
 *
 * ```c
 * #define STBI_NO_SIMD
 * #define STBI_ONLY_JPEG
 * #define STBI_ONLY_PNG
 * #define STBI_WINDOWS_UTF8
 * #define STB_IMAGE_IMPLEMENTATION
 * #include "stb_image.h"
 *
 * #define STBIR_NO_SIMD
 * #define STBIR_USE_FMA
 * #define STB_IMAGE_RESIZE_IMPLEMENTATION
 * #include "stb_image_resize2.h"
 * ```
 *
 * The blob is treated as a black box, please do not edit it. Skip this part if
 * you are reading the code.
 */

// NOLINTBEGIN  // Tell clang-tidy to shutup
// Tell clang-format to ignore this blob
// clang-format off
// Ignore all the warnings inside this blob

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-else"
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"

enum{STBI_default=0,STBI_grey=1,STBI_grey_alpha=2,STBI_rgb=3,STBI_rgb_alpha=4};
typedef unsigned char A;typedef unsigned short B;typedef struct{int(*A)(void*,//
char*,int);void(*B)(void*,int);int(*C)(void*);}C;typedef uint16_t D;typedef/////
int16_t E;typedef uint32_t F;typedef int32_t G;typedef unsigned char H[1];//////
typedef struct{F A,B;int C,D;C E;void*F;int G;int H;A I[128];int J;A*K,*L;A*M,*N
;}I;static void J(I*Gw);static void K(I*Gw,A const*Gx,int Gy){Gw->E.A=0;Gw->G=0;
Gw->J=0;Gw->K=Gw->M=(A*)Gx;Gw->L=Gw->N=(A*)Gx+Gy;}static void L(I*Gw,C*Gx,void*
Gy){Gw->E=*Gx;Gw->F=Gy;Gw->H=128;Gw->G=1;Gw->J=0;Gw->K=Gw->M=Gw->I;J(Gw);Gw->N=
Gw->L;}static int M(void*Gw,char*Gx,int Gy){return(int)fread(Gx,1,Gy,(FILE*)Gw);
}static void N(void*Gw,int Gx){int Gy;fseek((FILE*)Gw,Gx,1);Gy=fgetc((FILE*)Gw);
if(Gy!=1)ungetc(Gy,(FILE*)Gw);}static int O(void*Gw){return feof((FILE*)Gw)||///
ferror((FILE*)Gw);}static C P={M,N,O,};static void Q(I*Gw,FILE*Gx){L(Gw,&P,(void
*)Gx);}static void R(I*Gw){Gw->K=Gw->M;Gw->L=Gw->N;}enum{STBI_ORDER_RGB,////////
STBI_ORDER_BGR};typedef struct{int A;int B;int C;}S;static int T(I*Gw);static///
void*U(I*Gw,int*Gx,int*Gy,int*Gz,int G0,S*G1);static int V(I*Gw,int*Gx,int*Gy,//
int*Gz);static int W(I*Gw);static void*X(I*Gw,int*Gx,int*Gy,int*Gz,int G0,S*G1);
static int Y(I*Gw,int*Gx,int*Gy,int*Gz);static int Z(I*Gw);static _Thread_local
const char*a;extern const char*stbi_failure_reason(void){return a;}static int b(
const char*Gw){a=Gw;return 0;}static void*c(size_t Gw){return malloc(Gw);}static
int d(int Gw,int Gx){if(Gx<0)return 0;return Gw<=INT_MAX-Gx;}static int e(int Gw
,int Gx){if(Gw<0||Gx<0)return 0;if(Gx==0)return 1;return Gw<=INT_MAX/Gx;}static
int f(int Gw,int Gx,int Gy){return e(Gw,Gx)&&d(Gw*Gx,Gy);}static int g(int Gw,//
int Gx,int Gy,int Gz){return e(Gw,Gx)&&e(Gw*Gx,Gy)&&d(Gw*Gx*Gy,Gz);}static int h
(int Gw,int Gx,int Gy,int Gz,int G0){return e(Gw,Gx)&&e(Gw*Gx,Gy)&&e(Gw*Gx*Gy,Gz
)&&d(Gw*Gx*Gy*Gz,G0);}static void*i(int Gw,int Gx,int Gy){if(!f(Gw,Gx,Gy))return
0;return c(Gw*Gx+Gy);}static void*j(int Gw,int Gx,int Gy,int Gz){if(!g(Gw,Gx,Gy,
Gz))return 0;return c(Gw*Gx*Gy+Gz);}static void*k(int Gw,int Gx,int Gy,int Gz,//
int G0){if(!h(Gw,Gx,Gy,Gz,G0))return 0;return c(Gw*Gx*Gy*Gz+G0);}static int l(//
int Gw,int Gx){if((Gw>=0)!=(Gx>=0))return 1;if(Gw<0&&Gx<0)return Gw>=1-Gx;return
Gw<=INT_MAX-Gx;}static int m(int Gw,int Gx){if(Gx==0||Gx==-1)return 1;if((Gw>=0)
==(Gx>=0))return Gw<=32767/Gx;if(Gx<0)return Gw<=32768/Gx;return Gw>=32768/Gx;}
extern void stbi_image_free(void*Gw){free(Gw);}static float*n(A*Gw,int Gx,int Gy
,int Gz);static int o=0;extern void stbi_set_flip_vertically_on_load(int Gw){o=
Gw;}static _Thread_local int p,q;extern void////////////////////////////////////
stbi_set_flip_vertically_on_load_thread(int Gw){p=Gw;q=1;}static void*r(I*Gw,int
*Gx,int*Gy,int*Gz,int G0,S*G1,int G2){memset(G1,0,12);G1->A=8;G1->C=////////////
STBI_ORDER_RGB;G1->B=0;if(W(Gw))return X(Gw,Gx,Gy,Gz,G0,G1);if(T(Gw))return U(Gw
,Gx,Gy,Gz,G0,G1);return((unsigned char*)(size_t)(b("unknown image type")?0:0));}
static A*s(D*Gw,int Gx,int Gy,int Gz){int G0;int G1=Gx*Gy*Gz;A*G2;G2=(A*)c(G1);
if(G2==0)return(unsigned char*)(size_t)(b("outofmem")?0:0);for(G0=0;G0<G1;++G0)
G2[G0]=(A)((Gw[G0]>>8)&0xFF);free(Gw);return G2;}static D*t(A*Gw,int Gx,int Gy,
int Gz){int G0;int G1=Gx*Gy*Gz;D*G2;G2=(D*)c(G1*2);if(G2==0)return(D*)((unsigned
char*)(size_t)(b("outofmem")?0:0));for(G0=0;G0<G1;++G0)G2[G0]=(D)((Gw[G0]<<8)+Gw
[G0]);free(Gw);return G2;}static void u(void*Gw,int Gx,int Gy,int Gz){int G0;///
size_t G1=(size_t)Gx*Gz;A G2[2048];A*G3=(A*)Gw;for(G0=0;G0<(Gy>>1);G0++){A*G4=G3
+G0*G1;A*G5=G3+(Gy-G0-1)*G1;size_t G6=G1;while(G6){size_t G7=(G6<2048)?G6:2048;
memcpy(G2,G4,G7);memcpy(G4,G5,G7);memcpy(G5,G2,G7);G4+=G7;G5+=G7;G6-=G7;}}}/////
static unsigned char*v(I*Gw,int*Gx,int*Gy,int*Gz,int G0){S G1;void*G2=r(Gw,Gx,Gy
,Gz,G0,&G1,8);if(G2==0)return 0;if(G1.A!=8){G2=s((D*)G2,*Gx,*Gy,G0==0?*Gz:G0);G1
.A=8;}if(q?p:o){int G3=G0?G0:*Gz;u(G2,*Gx,*Gy,G3*1);}return(unsigned char*)G2;}
static D*w(I*Gw,int*Gx,int*Gy,int*Gz,int G0){S G1;void*G2=r(Gw,Gx,Gy,Gz,G0,&G1,
16);if(G2==0)return 0;if(G1.A!=16){G2=t((A*)G2,*Gx,*Gy,G0==0?*Gz:G0);G1.A=16;}if
(q?p:o){int G3=G0?G0:*Gz;u(G2,*Gx,*Gy,G3*2);}return(D*)G2;}static FILE*x(char///
const*Gw,char const*Gx){FILE*Gy;Gy=fopen(Gw,Gx);return Gy;}extern A*////////////
stbi_load_from_file(FILE*Gw,int*Gx,int*Gy,int*Gz,int G0){unsigned char*G1;I G2;Q
(&G2,Gw);G1=v(&G2,Gx,Gy,Gz,G0);if(G1)fseek(Gw,-(int)(G2.L-G2.K),1);return G1;}//
extern A*stbi_load(char const*Gw,int*Gx,int*Gy,int*Gz,int G0){FILE*G1=x(Gw,"rb")
;unsigned char*G2;if(!G1)return(unsigned char*)(size_t)(b("can't fopen")?0:0);G2
=stbi_load_from_file(G1,Gx,Gy,Gz,G0);fclose(G1);return G2;}extern D*////////////
stbi_load_from_file_16(FILE*Gw,int*Gx,int*Gy,int*Gz,int G0){D*G1;I G2;Q(&G2,Gw);
G1=w(&G2,Gx,Gy,Gz,G0);if(G1)fseek(Gw,-(int)(G2.L-G2.K),1);return G1;}extern B*//
stbi_load_16(char const*Gw,int*Gx,int*Gy,int*Gz,int G0){FILE*G1=x(Gw,"rb");D*G2;
if(!G1)return(B*)((unsigned char*)(size_t)(b("can't fopen")?0:0));G2=///////////
stbi_load_from_file_16(G1,Gx,Gy,Gz,G0);fclose(G1);return G2;}extern B*//////////
stbi_load_16_from_memory(A const*Gw,int Gx,int*Gy,int*Gz,int*G0,int G1){I G2;K(&
G2,Gw,Gx);return w(&G2,Gy,Gz,G0,G1);}extern B*stbi_load_16_from_callbacks(C/////
const*Gw,void*Gx,int*Gy,int*Gz,int*G0,int G1){I G2;L(&G2,(C*)Gw,Gx);return w(&G2
,Gy,Gz,G0,G1);}extern A*stbi_load_from_memory(A const*Gw,int Gx,int*Gy,int*Gz,//
int*G0,int G1){I G2;K(&G2,Gw,Gx);return v(&G2,Gy,Gz,G0,G1);}extern A*///////////
stbi_load_from_callbacks(C const*Gw,void*Gx,int*Gy,int*Gz,int*G0,int G1){I G2;L(
&G2,(C*)Gw,Gx);return v(&G2,Gy,Gz,G0,G1);}static float*y(I*Gw,int*Gx,int*Gy,int*
Gz,int G0){unsigned char*G1;G1=v(Gw,Gx,Gy,Gz,G0);if(G1)return n(G1,*Gx,*Gy,G0?G0
:*Gz);return(float*)(size_t)(b("unknown image type")?0:0);}extern float*////////
stbi_loadf_from_memory(A const*Gw,int Gx,int*Gy,int*Gz,int*G0,int G1){I G2;K(&G2
,Gw,Gx);return y(&G2,Gy,Gz,G0,G1);}extern float*stbi_loadf_from_callbacks(C/////
const*Gw,void*Gx,int*Gy,int*Gz,int*G0,int G1){I G2;L(&G2,(C*)Gw,Gx);return y(&G2
,Gy,Gz,G0,G1);}extern float*stbi_loadf_from_file(FILE*Gw,int*Gx,int*Gy,int*Gz,//
int G0){I G1;Q(&G1,Gw);return y(&G1,Gx,Gy,Gz,G0);}extern float*stbi_loadf(char//
const*Gw,int*Gx,int*Gy,int*Gz,int G0){float*G1;FILE*G2=x(Gw,"rb");if(!G2)return(
float*)(size_t)(b("can't fopen")?0:0);G1=stbi_loadf_from_file(G2,Gx,Gy,Gz,G0);//
fclose(G2);return G1;}extern int stbi_is_hdr_from_memory(A const*Gw,int Gx){////
return 0;}extern int stbi_is_hdr_from_file(FILE*Gw){(void)Gw;return 0;}extern///
int stbi_is_hdr(char const*Gw){FILE*Gx=x(Gw,"rb");int Gy=0;if(Gx){Gy=///////////
stbi_is_hdr_from_file(Gx);fclose(Gx);}return Gy;}extern int/////////////////////
stbi_is_hdr_from_callbacks(C const*Gw,void*Gx){(void)Gw;(void)Gx;return 0;}/////
static float z=2.2f,_=1.0f;extern void stbi_ldr_to_hdr_gamma(float Gw){z=Gw;}///
extern void stbi_ldr_to_hdr_scale(float Gw){_=Gw;}static float AA=4.545e-01,AB=
1.0f;extern void stbi_hdr_to_ldr_gamma(float Gw){AA=1/Gw;}extern void///////////
stbi_hdr_to_ldr_scale(float Gw){AB=1/Gw;}enum{STBI__SCAN_load=0,STBI__SCAN_type,
STBI__SCAN_header};static void J(I*Gx){int Gy=Gx->E.A(Gx->F,(char*)Gx->I,Gx->H);
Gx->J+=(int)(Gx->K-Gx->M);if(Gy==0){Gx->G=0;Gx->K=Gx->I;Gx->L=Gx->I+1;*Gx->K=0;}
else{Gx->K=Gx->I;Gx->L=Gx->I+Gy;}}static A AC(I*Gw){if(Gw->K<Gw->L)return*Gw->K
++;if(Gw->G){J(Gw);return*Gw->K++;}return 0;}static int AD(I*Gw){if(Gw->E.A){if(
!Gw->E.C(Gw->F))return 0;if(Gw->G==0)return 1;}return Gw->K>=Gw->L;}static void
AE(I*Gw,int Gx){if(Gx==0)return;if(Gx<0){Gw->K=Gw->L;return;}if(Gw->E.A){int Gy=
(int)(Gw->L-Gw->K);if(Gy<Gx){Gw->K=Gw->L;Gw->E.B(Gw->F,Gx-Gy);return;}}Gw->K+=Gx
;}static int AF(I*Gw,A*Gx,int Gy){if(Gw->E.A){int Gz=(int)(Gw->L-Gw->K);if(Gz<Gy
){int G0,G1;memcpy(Gx,Gw->K,Gz);G1=Gw->E.A(Gw->F,(char*)Gx+Gz,Gy-Gz);G0=(G1==(Gy
-Gz));Gw->K=Gw->L;return G0;}}if(Gw->K+Gy<=Gw->L){memcpy(Gx,Gw->K,Gy);Gw->K+=Gy;
return 1;}else return 0;}static int AG(I*Gw){int Gx=AC(Gw);return(Gx<<8)+AC(Gw);
}static F AH(I*Gw){F Gx=AG(Gw);return(Gx<<16)+AG(Gw);}static A AI(int Gw,int Gx,
int Gy){return(A)(((Gw*77)+(Gx*150)+(29*Gy))>>8);}static unsigned char*AJ(//////
unsigned char*Gw,int Gx,int Gy,unsigned int Gz,unsigned int G0){int G1,G2;//////
unsigned char*G3;if(Gy==Gx)return Gw;G3=(unsigned char*)j(Gy,Gz,G0,0);if(G3==0){
free(Gw);return(unsigned char*)(size_t)(b("outofmem")?0:0);}for(G2=0;G2<(int)G0;
++G2){unsigned char*G4=Gw+G2*Gz*Gx;unsigned char*G5=G3+G2*Gz*Gy;switch((Gx*8+Gy)
){case 10:for(G1=Gz-1;G1>=0;--G1,G4+=1,G5+=2){G5[0]=G4[0];G5[1]=255;}break;case
11:for(G1=Gz-1;G1>=0;--G1,G4+=1,G5+=3)G5[0]=G5[1]=G5[2]=G4[0];break;case 12:for(
G1=Gz-1;G1>=0;--G1,G4+=1,G5+=4){G5[0]=G5[1]=G5[2]=G4[0];G5[3]=255;}break;case 17
:for(G1=Gz-1;G1>=0;--G1,G4+=2,G5+=1)G5[0]=G4[0];break;case 19:for(G1=Gz-1;G1>=0;
--G1,G4+=2,G5+=3)G5[0]=G5[1]=G5[2]=G4[0];break;case 20:for(G1=Gz-1;G1>=0;--G1,G4
+=2,G5+=4){G5[0]=G5[1]=G5[2]=G4[0];G5[3]=G4[1];}break;case 28:for(G1=Gz-1;G1>=0;
--G1,G4+=3,G5+=4){G5[0]=G4[0];G5[1]=G4[1];G5[2]=G4[2];G5[3]=255;}break;case 25:
for(G1=Gz-1;G1>=0;--G1,G4+=3,G5+=1)G5[0]=AI(G4[0],G4[1],G4[2]);break;case 26:for
(G1=Gz-1;G1>=0;--G1,G4+=3,G5+=2){G5[0]=AI(G4[0],G4[1],G4[2]);G5[1]=255;}break;//
case 33:for(G1=Gz-1;G1>=0;--G1,G4+=4,G5+=1)G5[0]=AI(G4[0],G4[1],G4[2]);break;///
case 34:for(G1=Gz-1;G1>=0;--G1,G4+=4,G5+=2){G5[0]=AI(G4[0],G4[1],G4[2]);G5[1]=G4
[3];}break;case 35:for(G1=Gz-1;G1>=0;--G1,G4+=4,G5+=3){G5[0]=G4[0];G5[1]=G4[1];
G5[2]=G4[2];}break;default:free(Gw);free(G3);return(unsigned char*)(size_t)(b(//
"unsupported")?0:0);}}free(Gw);return G3;}static D AK(int Gw,int Gx,int Gy){////
return(D)(((Gw*77)+(Gx*150)+(29*Gy))>>8);}static D*AL(D*Gw,int Gx,int Gy,///////
unsigned int Gz,unsigned int G0){int G1,G2;D*G3;if(Gy==Gx)return Gw;G3=(D*)c(Gy*
Gz*G0*2);if(G3==0){free(Gw);return(D*)((unsigned char*)(size_t)(b("outofmem")?0:
0));}for(G2=0;G2<(int)G0;++G2){D*G4=Gw+G2*Gz*Gx;D*G5=G3+G2*Gz*Gy;switch((Gx*8+Gy
)){case 10:for(G1=Gz-1;G1>=0;--G1,G4+=1,G5+=2){G5[0]=G4[0];G5[1]=0xffff;}break;
case 11:for(G1=Gz-1;G1>=0;--G1,G4+=1,G5+=3)G5[0]=G5[1]=G5[2]=G4[0];break;case 12
:for(G1=Gz-1;G1>=0;--G1,G4+=1,G5+=4){G5[0]=G5[1]=G5[2]=G4[0];G5[3]=0xffff;}break
;case 17:for(G1=Gz-1;G1>=0;--G1,G4+=2,G5+=1)G5[0]=G4[0];break;case 19:for(G1=Gz-
1;G1>=0;--G1,G4+=2,G5+=3)G5[0]=G5[1]=G5[2]=G4[0];break;case 20:for(G1=Gz-1;G1>=0
;--G1,G4+=2,G5+=4){G5[0]=G5[1]=G5[2]=G4[0];G5[3]=G4[1];}break;case 28:for(G1=Gz-
1;G1>=0;--G1,G4+=3,G5+=4){G5[0]=G4[0];G5[1]=G4[1];G5[2]=G4[2];G5[3]=0xffff;}////
break;case 25:for(G1=Gz-1;G1>=0;--G1,G4+=3,G5+=1)G5[0]=AK(G4[0],G4[1],G4[2]);///
break;case 26:for(G1=Gz-1;G1>=0;--G1,G4+=3,G5+=2){G5[0]=AK(G4[0],G4[1],G4[2]);G5
[1]=0xffff;}break;case 33:for(G1=Gz-1;G1>=0;--G1,G4+=4,G5+=1)G5[0]=AK(G4[0],G4[1
],G4[2]);break;case 34:for(G1=Gz-1;G1>=0;--G1,G4+=4,G5+=2){G5[0]=AK(G4[0],G4[1],
G4[2]);G5[1]=G4[3];}break;case 35:for(G1=Gz-1;G1>=0;--G1,G4+=4,G5+=3){G5[0]=G4[0
];G5[1]=G4[1];G5[2]=G4[2];}break;default:free(Gw);free(G3);return(D*)((unsigned
char*)(size_t)(b("unsupported")?0:0));}}free(Gw);return G3;}static float*n(A*G0,
int G1,int G2,int G3){int G4,G5,G6;float*G7;if(!G0)return 0;G7=(float*)k(G1,G2,
G3,4,0);if(G7==0){free(G0);return(float*)(size_t)(b("outofmem")?0:0);}if(G3&1)G6
=G3;else G6=G3-1;for(G4=0;G4<G1*G2;++G4)for(G5=0;G5<G6;++G5)G7[G4*G3+G5]=(float)
(pow(G0[G4*G3+G5]/255.0f,z)*_);if(G6<G3)for(G4=0;G4<G1*G2;++G4)G7[G4*G3+G6]=G0[
G4*G3+G6]/255.0f;free(G0);return G7;}typedef struct{A A[512];D B[256];A C[256];A
D[257];unsigned int E[18];int F[17];}AM;typedef struct{I*A;AM B[4];AM C[4];D D[4
][64];E E[4][512];int F,G;int H,I;int J,K;struct{int A;int B,C;int D;int E,F;int
G;int H,I,J,K;A*L;void*M,*N;A*O;short*P;int Q,R;}L[4];F M;int N;unsigned char O;
int P;int Q;int R;int S;int T;int U;int V;int W;int X;int Y;int Z,a[4];int b,c;
void(*d)(A*,int,short[64]);void(*e)(A*,const A*,const A*,const A*,int,int);A*(*f
)(A*,A*,A*,int,int);}AN;static int AO(AM*Gw,int*Gx){int Gy,Gz,G0=0;unsigned int
G1;for(Gy=0;Gy<16;++Gy)for(Gz=0;Gz<Gx[Gy];++Gz){Gw->D[G0++]=(A)(Gy+1);if(G0>=257
)return b("bad size list");}Gw->D[G0]=0;G1=0;G0=0;for(Gz=1;Gz<=16;++Gz){Gw->F[Gz
]=G0-G1;if(Gw->D[G0]==Gz){while(Gw->D[G0]==Gz)Gw->B[G0++]=(D)(G1++);if(G1-1>=(1u
<<Gz))return b("bad code lengths");}Gw->E[Gz]=G1<<(16-Gz);G1<<=1;}Gw->E[Gz]=////
0xffffffff;memset(Gw->A,255,512);for(Gy=0;Gy<G0;++Gy){int G2=Gw->D[Gy];if(G2<=9)
{int G3=Gw->B[Gy]<<(9-G2);int G4=1<<(9-G2);for(Gz=0;Gz<G4;++Gz)Gw->A[G3+Gz]=(A)
Gy;}}return 1;}static void AP(E*Gw,AM*Gx){int Gy;for(Gy=0;Gy<512;++Gy){A Gz=Gx->
A[Gy];Gw[Gy]=0;if(Gz<255){int G0=Gx->C[Gz];int G1=(G0>>4)&15;int G2=G0&15;int G3
=Gx->D[Gz];if(G2&&G3+G2<=9){int G4=((Gy<<G3)&511)>>(9-G2);int G5=1<<(G2-1);if(G4
<G5)G4+=(~0U<<G2)+1;if(G4>=-128&&G4<=127)Gw[Gy]=(E)((G4*256)+(G1*16)+(G3+G2));}}
}}static void AQ(AN*Gw){do{unsigned int Gx=Gw->P?0:AC(Gw->A);if(Gx==0xff){int Gy
=AC(Gw->A);while(Gy==0xff)Gy=AC(Gw->A);if(Gy!=0){Gw->O=(unsigned char)Gy;Gw->P=1
;return;}}Gw->M|=Gx<<(24-Gw->N);Gw->N+=8;}while(Gw->N<=24);}static const F AR[17
]={0,1,3,7,15,31,63,127,255,511,1023,2047,4095,8191,16383,32767,65535};static///
int AS(AN*Gw,AM*Gx){unsigned int Gy;int Gz,G0;if(Gw->N<16)AQ(Gw);Gz=(Gw->M>>23)&
511;G0=Gx->A[Gz];if(G0<255){int G1=Gx->D[G0];if(G1>Gw->N)return-1;Gw->M<<=G1;Gw
->N-=G1;return Gx->C[G0];}Gy=Gw->M>>16;for(G0=10;;++G0)if(Gy<Gx->E[G0])break;if(
G0==17){Gw->N-=16;return-1;}if(G0>Gw->N)return-1;Gz=((Gw->M>>(32-G0))&AR[G0])+Gx
->F[G0];if(Gz<0||Gz>=256)return-1;Gw->N-=G0;Gw->M<<=G0;return Gx->C[Gz];}static
const int AT[16]={0,-1,-3,-7,-15,-31,-63,-127,-255,-511,-1023,-2047,-4095,-8191,
-16383,-32767};static int AU(AN*Gw,int Gx){unsigned int Gy;int Gz;if(Gw->N<Gx)AQ
(Gw);if(Gw->N<Gx)return 0;Gz=Gw->M>>31;Gy=((Gw->M<<Gx)|(Gw->M>>(-Gx&31)));Gw->M=
Gy&~AR[Gx];Gy&=AR[Gx];Gw->N-=Gx;return Gy+(AT[Gx]&(Gz-1));}static int AV(AN*Gw,
int Gx){unsigned int Gy;if(Gw->N<Gx)AQ(Gw);if(Gw->N<Gx)return 0;Gy=((Gw->M<<Gx)|
(Gw->M>>(-Gx&31)));Gw->M=Gy&~AR[Gx];Gy&=AR[Gx];Gw->N-=Gx;return Gy;}static int//
AW(AN*Gw){unsigned int Gx;if(Gw->N<1)AQ(Gw);if(Gw->N<1)return 0;Gx=Gw->M;Gw->M//
<<=1;--Gw->N;return Gx&0x80000000;}static const A AX[79]={0,1,8,16,9,2,3,10,17,
24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,35,42,49,56,57,
50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63
,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63};static int AY(AN*Gw,short Gx[64],
AM*Gy,AM*Gz,E*G0,int G1,D*G2){int G3,G4,G5;int G6;if(Gw->N<16)AQ(Gw);G6=AS(Gw,Gy
);if(G6<0||G6>15)return b("bad huffman code");memset(Gx,0,128);G3=G6?AU(Gw,G6):0
;if(!l(Gw->L[G1].G,G3))return b("bad delta");G4=Gw->L[G1].G+G3;Gw->L[G1].G=G4;if
(!m(G4,G2[0]))return b("can't merge dc and ac");Gx[0]=(short)(G4*G2[0]);G5=1;do{
unsigned int G7;int G8,G9,G_;if(Gw->N<16)AQ(Gw);G8=(Gw->M>>23)&511;G9=G0[G8];if(
G9){G5+=(G9>>4)&15;G_=G9&15;if(G_>Gw->N)return b("bad huffman code");Gw->M<<=G_;
Gw->N-=G_;G7=AX[G5++];Gx[G7]=(short)((G9>>8)*G2[G7]);}else{int HA=AS(Gw,Gz);if(
HA<0)return b("bad huffman code");G_=HA&15;G9=HA>>4;if(G_==0){if(HA!=0xf0)break;
G5+=16;}else{G5+=G9;G7=AX[G5++];Gx[G7]=(short)(AU(Gw,G_)*G2[G7]);}}}while(G5<64)
;return 1;}static int AZ(AN*Gw,short Gx[64],AM*Gy,int Gz){int G0,G1;int G2;if(Gw
->S!=0)return b("can't merge dc and ac");if(Gw->N<16)AQ(Gw);if(Gw->T==0){memset(
Gx,0,128);G2=AS(Gw,Gy);if(G2<0||G2>15)return b("can't merge dc and ac");G0=G2?AU
(Gw,G2):0;if(!l(Gw->L[Gz].G,G0))return b("bad delta");G1=Gw->L[Gz].G+G0;Gw->L[Gz
].G=G1;if(!m(G1,1<<Gw->U))return b("can't merge dc and ac");Gx[0]=(short)(G1*(1
<<Gw->U));}else if(AW(Gw))Gx[0]+=(short)(1<<Gw->U);return 1;}static int Aa(AN*Gw
,short Gx[64],AM*Gy,E*Gz){int G0;if(Gw->R==0)return b("can't merge dc and ac");
if(Gw->T==0){int G1=Gw->U;if(Gw->V){--Gw->V;return 1;}G0=Gw->R;do{unsigned int//
G2;int G3,G4,G5;if(Gw->N<16)AQ(Gw);G3=(Gw->M>>23)&511;G4=Gz[G3];if(G4){G0+=(G4>>
4)&15;G5=G4&15;if(G5>Gw->N)return b("bad huffman code");Gw->M<<=G5;Gw->N-=G5;G2=
AX[G0++];Gx[G2]=(short)((G4>>8)*(1<<G1));}else{int G6=AS(Gw,Gy);if(G6<0)return b
("bad huffman code");G5=G6&15;G4=G6>>4;if(G5==0){if(G4<15){Gw->V=(1<<G4);if(G4)
Gw->V+=AV(Gw,G4);--Gw->V;break;}G0+=16;}else{G0+=G4;G2=AX[G0++];Gx[G2]=(short)(
AU(Gw,G5)*(1<<G1));}}}while(G0<=Gw->S);}else{short G1=(short)(1<<Gw->U);if(Gw->V
){--Gw->V;for(G0=Gw->R;G0<=Gw->S;++G0){short*G2=&Gx[AX[G0]];if(*G2!=0)if(AW(Gw))
if((*G2&G1)==0)if(*G2>0)*G2+=G1;else*G2-=G1;}}else{G0=Gw->R;do{int G2,G3;int G4=
AS(Gw,Gy);if(G4<0)return b("bad huffman code");G3=G4&15;G2=G4>>4;if(G3==0){if(G2
<15){Gw->V=(1<<G2)-1;if(G2)Gw->V+=AV(Gw,G2);G2=64;}else{}}else{if(G3!=1)return b
("bad huffman code");if(AW(Gw))G3=G1;else G3=-G1;}while(G0<=Gw->S){short*G5=&Gx[
AX[G0++]];if(*G5!=0){if(AW(Gw))if((*G5&G1)==0)if(*G5>0)*G5+=G1;else*G5-=G1;}else
{if(G2==0){*G5=(short)G3;break;}--G2;}}}while(G0<=Gw->S);}}return 1;}static A Ab
(int Gw){if((unsigned int)Gw>255){if(Gw<0)return 0;if(Gw>255)return 255;}return(
A)Gw;}static void Ac(A*Gw,int Gx,short Gy[64]){int Gz,G0[64],*G1=G0;A*G2;short*
G3=Gy;for(Gz=0;Gz<8;++Gz,++G3,++G1)if(G3[8]==0&&G3[16]==0&&G3[24]==0&&G3[32]==0
&&G3[40]==0&&G3[48]==0&&G3[56]==0){int G4=G3[0]*4;G1[0]=G1[8]=G1[16]=G1[24]=G1[
32]=G1[40]=G1[48]=G1[56]=G4;}else{int G4,G5,G6,G7,G8,G9,G_,HA,HB,HC,HD,HE,HF;G9=
G3[16];G_=G3[48];G8=(G9+G_)*0.5;G6=G8+G_*-4095.5;G7=G8+G9*0.5;G9=G3[0];G_=G3[32]
;G4=((G9+G_)*4096);G5=((G9-G_)*4096);HC=G4+G7;HF=G4-G7;HD=G5+G6;HE=G5-G6;G4=G3[
56];G5=G3[40];G6=G3[24];G7=G3[8];G_=G4+G6;HA=G5+G7;G8=G4+G7;G9=G5+G6;HB=(G_+HA)*
4096.5;G4=G4*0.5;G5=G5*8192.5;G6=G6*12288.5;G7=G7*4096.5;G8=HB+G8*0.5;G9=HB+G9*-
8191.5;G_=G_*-4095.5;HA=HA*0.5;G7+=G8+HA;G6+=G9+G_;G5+=G9+HA;G4+=G8+G_;HC+=512;
HD+=512;HE+=512;HF+=512;G1[0]=(HC+G7)>>10;G1[56]=(HC-G7)>>10;G1[8]=(HD+G6)>>10;
G1[48]=(HD-G6)>>10;G1[16]=(HE+G5)>>10;G1[40]=(HE-G5)>>10;G1[24]=(HF+G4)>>10;G1[
32]=(HF-G4)>>10;}for(Gz=0,G1=G0,G2=Gw;Gz<8;++Gz,G1+=8,G2+=Gx){int G4,G5,G6,G7,G8
,G9,G_,HA,HB,HC,HD,HE,HF;G9=G1[2];G_=G1[6];G8=(G9+G_)*0.5;G6=G8+G_*-4095.5;G7=G8
+G9*0.5;G9=G1[0];G_=G1[4];G4=((G9+G_)*4096);G5=((G9-G_)*4096);HC=G4+G7;HF=G4-G7;
HD=G5+G6;HE=G5-G6;G4=G1[7];G5=G1[5];G6=G1[3];G7=G1[1];G_=G4+G6;HA=G5+G7;G8=G4+G7
;G9=G5+G6;HB=(G_+HA)*4096.5;G4=G4*0.5;G5=G5*8192.5;G6=G6*12288.5;G7=G7*4096.5;G8
=HB+G8*0.5;G9=HB+G9*-8191.5;G_=G_*-4095.5;HA=HA*0.5;G7+=G8+HA;G6+=G9+G_;G5+=G9+
HA;G4+=G8+G_;HC+=16842752;HD+=16842752;HE+=16842752;HF+=16842752;G2[0]=Ab((HC+G7
)>>17);G2[7]=Ab((HC-G7)>>17);G2[1]=Ab((HD+G6)>>17);G2[6]=Ab((HD-G6)>>17);G2[2]=
Ab((HE+G5)>>17);G2[5]=Ab((HE-G5)>>17);G2[3]=Ab((HF+G4)>>17);G2[4]=Ab((HF-G4)>>17
);}}static A Ad(AN*Gw){A Gx;if(Gw->O!=0xff){Gx=Gw->O;Gw->O=0xff;return Gx;}Gx=AC
(Gw->A);if(Gx!=0xff)return 0xff;while(Gx==0xff)Gx=AC(Gw->A);return Gx;}static///
void Ae(AN*Gw){Gw->N=0;Gw->M=0;Gw->P=0;Gw->L[0].G=Gw->L[1].G=Gw->L[2].G=Gw->L[3]
.G=0;Gw->O=0xff;Gw->c=Gw->b?Gw->b:0x7fffffff;Gw->V=0;}static int Af(AN*Gw){Ae(Gw
);if(!Gw->Q){if(Gw->Z==1){int Gx,Gy;short Gz[64];int G0=Gw->a[0];int G1=(Gw->L[
G0].H+7)>>3;int G2=(Gw->L[G0].I+7)>>3;for(Gy=0;Gy<G2;++Gy)for(Gx=0;Gx<G1;++Gx){
int G3=Gw->L[G0].F;if(!AY(Gw,Gz,Gw->B+Gw->L[G0].E,Gw->C+G3,Gw->E[G3],G0,Gw->D[Gw
->L[G0].D]))return 0;Gw->d(Gw->L[G0].L+Gw->L[G0].J*Gy*8+Gx*8,Gw->L[G0].J,Gz);if(
--Gw->c<=0){if(Gw->N<24)AQ(Gw);if(!(Gw->O>=0xd0&&Gw->O<=0xd7))return 1;Ae(Gw);}}
return 1;}else{int Gx,Gy,Gz,G0,G1;short G2[64];for(Gy=0;Gy<Gw->I;++Gy)for(Gx=0;
Gx<Gw->H;++Gx){for(Gz=0;Gz<Gw->Z;++Gz){int G3=Gw->a[Gz];for(G1=0;G1<Gw->L[G3].C;
++G1)for(G0=0;G0<Gw->L[G3].B;++G0){int G4=(Gx*Gw->L[G3].B+G0)*8;int G5=(Gy*Gw->L
[G3].C+G1)*8;int G6=Gw->L[G3].F;if(!AY(Gw,G2,Gw->B+Gw->L[G3].E,Gw->C+G6,Gw->E[G6
],G3,Gw->D[Gw->L[G3].D]))return 0;Gw->d(Gw->L[G3].L+Gw->L[G3].J*G5+G4,Gw->L[G3].
J,G2);}}if(--Gw->c<=0){if(Gw->N<24)AQ(Gw);if(!(Gw->O>=0xd0&&Gw->O<=0xd7))return
1;Ae(Gw);}}return 1;}}else if(Gw->Z==1){int Gx,Gy;int Gz=Gw->a[0];int G0=(Gw->L[
Gz].H+7)>>3;int G1=(Gw->L[Gz].I+7)>>3;for(Gy=0;Gy<G1;++Gy)for(Gx=0;Gx<G0;++Gx){
short*G2=Gw->L[Gz].P+64*(Gx+Gy*Gw->L[Gz].Q);if(Gw->R==0){if(!AZ(Gw,G2,&Gw->B[Gw
->L[Gz].E],Gz))return 0;}else{int G3=Gw->L[Gz].F;if(!Aa(Gw,G2,&Gw->C[G3],Gw->E[
G3]))return 0;}if(--Gw->c<=0){if(Gw->N<24)AQ(Gw);if(!(Gw->O>=0xd0&&Gw->O<=0xd7))
return 1;Ae(Gw);}}return 1;}else{int Gx,Gy,Gz,G0,G1;for(Gy=0;Gy<Gw->I;++Gy)for(
Gx=0;Gx<Gw->H;++Gx){for(Gz=0;Gz<Gw->Z;++Gz){int G2=Gw->a[Gz];for(G1=0;G1<Gw->L[
G2].C;++G1)for(G0=0;G0<Gw->L[G2].B;++G0){int G3=Gx*Gw->L[G2].B+G0;int G4=Gy*Gw->
L[G2].C+G1;short*G5=Gw->L[G2].P+64*(G3+G4*Gw->L[G2].Q);if(!AZ(Gw,G5,&Gw->B[Gw->L
[G2].E],G2))return 0;}}if(--Gw->c<=0){if(Gw->N<24)AQ(Gw);if(!(Gw->O>=0xd0&&Gw->O
<=0xd7))return 1;Ae(Gw);}}return 1;}}static void Ag(short*Gw,D*Gx){int Gy;for(Gy
=0;Gy<64;++Gy)Gw[Gy]*=Gx[Gy];}static void Ah(AN*Gw){if(Gw->Q){int Gx,Gy,Gz;for(
Gz=0;Gz<Gw->A->C;++Gz){int G0=(Gw->L[Gz].H+7)>>3;int G1=(Gw->L[Gz].I+7)>>3;for(
Gy=0;Gy<G1;++Gy)for(Gx=0;Gx<G0;++Gx){short*G2=Gw->L[Gz].P+64*(Gx+Gy*Gw->L[Gz].Q)
;Ag(G2,Gw->D[Gw->L[Gz].D]);Gw->d(Gw->L[Gz].L+Gw->L[Gz].J*Gy*8+Gx*8,Gw->L[Gz].J,
G2);}}}}static int Ai(AN*Gw,int Gx){int Gy;switch(Gx){case 0xff:return b(///////
"expected marker");case 0xDD:if(AG(Gw->A)!=4)return b("bad DRI len");Gw->b=AG(Gw
->A);return 1;case 0xDB:Gy=AG(Gw->A)-2;while(Gy>0){int Gz=AC(Gw->A);int G0=Gz>>4
,G1=G0!=0;int G2=Gz&15,G3;if(G0!=0&&G0!=1)return b("bad DQT type");if(G2>3)/////
return b("bad DQT table");for(G3=0;G3<64;++G3)Gw->D[G2][AX[G3]]=(D)(G1?AG(Gw->A)
:AC(Gw->A));Gy-=(G1?129:65);}return Gy==0;case 0xC4:Gy=AG(Gw->A)-2;while(Gy>0){A
*Gz;int G0[16],G1,G2=0;int G3=AC(Gw->A);int G4=G3>>4;int G5=G3&15;if(G4>1||G5>3)
return b("bad DHT header");for(G1=0;G1<16;++G1){G0[G1]=AC(Gw->A);G2+=G0[G1];}if(
G2>256)return b("bad DHT header");Gy-=17;if(G4==0){if(!AO(Gw->B+G5,G0))return 0;
Gz=Gw->B[G5].C;}else{if(!AO(Gw->C+G5,G0))return 0;Gz=Gw->C[G5].C;}for(G1=0;G1<G2
;++G1)Gz[G1]=AC(Gw->A);if(G4!=0)AP(Gw->E[G5],Gw->C+G5);Gy-=G2;}return Gy==0;}if(
(Gx>=0xE0&&Gx<=0xEF)||Gx==0xFE){Gy=AG(Gw->A);if(Gy<2)if(Gx==0xFE)return b(//////
"bad COM len");else return b("bad APP len");Gy-=2;if(Gx==0xE0&&Gy>=5){static////
const unsigned char Gz[5]={'J','F','I','F','\0'};int G0=1;int G1;for(G1=0;G1<5;
++G1)if(AC(Gw->A)!=Gz[G1])G0=0;Gy-=5;if(G0)Gw->W=1;}else if(Gx==0xEE&&Gy>=12){//
static const unsigned char Gz[6]={'A','d','o','b','e','\0'};int G0=1;int G1;for(
G1=0;G1<6;++G1)if(AC(Gw->A)!=Gz[G1])G0=0;Gy-=6;if(G0){AC(Gw->A);AG(Gw->A);AG(Gw
->A);Gw->X=AC(Gw->A);Gy-=6;}}AE(Gw->A,Gy);return 1;}return b("unknown marker");}
static int Aj(AN*Gw){int Gx;int Gy=AG(Gw->A);Gw->Z=AC(Gw->A);if(Gw->Z<1||Gw->Z>4
||Gw->Z>(int)Gw->A->C)return b("bad SOS component count");if(Gy!=6+2*Gw->Z)/////
return b("bad SOS len");for(Gx=0;Gx<Gw->Z;++Gx){int Gz=AC(Gw->A),G0;int G1=AC(Gw
->A);for(G0=0;G0<Gw->A->C;++G0)if(Gw->L[G0].A==Gz)break;if(G0==Gw->A->C)return 0
;Gw->L[G0].E=G1>>4;if(Gw->L[G0].E>3)return b("bad DC huff");Gw->L[G0].F=G1&15;if
(Gw->L[G0].F>3)return b("bad AC huff");Gw->a[Gx]=G0;}{int Gz;Gw->R=AC(Gw->A);Gw
->S=AC(Gw->A);Gz=AC(Gw->A);Gw->T=(Gz>>4);Gw->U=(Gz&15);if(Gw->Q){if(Gw->R>63||Gw
->S>63||Gw->R>Gw->S||Gw->T>13||Gw->U>13)return b("bad SOS");}else{if(Gw->R!=0)//
return b("bad SOS");if(Gw->T!=0||Gw->U!=0)return b("bad SOS");Gw->S=63;}}return
1;}static int Ak(AN*Gw,int Gx,int Gy){int Gz;for(Gz=0;Gz<Gx;++Gz){if(Gw->L[Gz].M
){free(Gw->L[Gz].M);Gw->L[Gz].M=0;Gw->L[Gz].L=0;}if(Gw->L[Gz].N){free(Gw->L[Gz].
N);Gw->L[Gz].N=0;Gw->L[Gz].P=0;}if(Gw->L[Gz].O){free(Gw->L[Gz].O);Gw->L[Gz].O=0;
}}return Gy;}static int Al(AN*Gw,int Gx){I*Gy=Gw->A;int Gz,G0,G1,G2,G3=1,G4=1,G5
;Gz=AG(Gy);if(Gz<11)return b("bad SOF len");G0=AC(Gy);if(G0!=8)return b(////////
"only 8-bit");Gy->B=AG(Gy);if(Gy->B==0)return b("no header height");Gy->A=AG(Gy)
;if(Gy->A==0)return b("0 width");if(Gy->B>16777216)return b("too large");if(Gy->
A>16777216)return b("too large");G5=AC(Gy);if(G5!=3&&G5!=1&&G5!=4)return b(/////
"bad component count");Gy->C=G5;for(G1=0;G1<G5;++G1){Gw->L[G1].L=0;Gw->L[G1].O=0
;}if(Gz!=8+3*Gy->C)return b("bad SOF len");Gw->Y=0;for(G1=0;G1<Gy->C;++G1){/////
static const unsigned char G6[3]={'R','G','B'};Gw->L[G1].A=AC(Gy);if(Gy->C==3&&
Gw->L[G1].A==G6[G1])++Gw->Y;G2=AC(Gy);Gw->L[G1].B=(G2>>4);if(!Gw->L[G1].B||Gw->L
[G1].B>4)return b("bad H");Gw->L[G1].C=G2&15;if(!Gw->L[G1].C||Gw->L[G1].C>4)////
return b("bad V");Gw->L[G1].D=AC(Gy);if(Gw->L[G1].D>3)return b("bad TQ");}if(Gx
!=STBI__SCAN_load)return 1;if(!g(Gy->A,Gy->B,Gy->C,0))return b("too large");for(
G1=0;G1<Gy->C;++G1){if(Gw->L[G1].B>G3)G3=Gw->L[G1].B;if(Gw->L[G1].C>G4)G4=Gw->L[
G1].C;}for(G1=0;G1<Gy->C;++G1){if(G3%Gw->L[G1].B!=0)return b("bad H");if(G4%Gw->
L[G1].C!=0)return b("bad V");}Gw->F=G3;Gw->G=G4;Gw->J=G3*8;Gw->K=G4*8;Gw->H=(Gy
->A+Gw->J-1)/Gw->J;Gw->I=(Gy->B+Gw->K-1)/Gw->K;for(G1=0;G1<Gy->C;++G1){Gw->L[G1]
.H=(Gy->A*Gw->L[G1].B+G3-1)/G3;Gw->L[G1].I=(Gy->B*Gw->L[G1].C+G4-1)/G4;Gw->L[G1]
.J=Gw->H*Gw->L[G1].B*8;Gw->L[G1].K=Gw->I*Gw->L[G1].C*8;Gw->L[G1].P=0;Gw->L[G1].N
=0;Gw->L[G1].O=0;Gw->L[G1].M=i(Gw->L[G1].J,Gw->L[G1].K,15);if(Gw->L[G1].M==0)///
return Ak(Gw,G1+1,b("outofmem"));Gw->L[G1].L=(A*)(((size_t)Gw->L[G1].M+15)&~15);
if(Gw->Q){Gw->L[G1].Q=Gw->L[G1].J/8;Gw->L[G1].R=Gw->L[G1].K/8;Gw->L[G1].N=j(Gw->
L[G1].J,Gw->L[G1].K,2,15);if(Gw->L[G1].N==0)return Ak(Gw,G1+1,b("outofmem"));Gw
->L[G1].P=(short*)(((size_t)Gw->L[G1].N+15)&~15);}}return 1;}static int Am(AN*Gw
,int Gx){int Gy;Gw->W=0;Gw->X=-1;Gw->O=0xff;Gy=Ad(Gw);if(!(Gy==0xd8))return b(//
"no SOI");if(Gx==STBI__SCAN_type)return 1;Gy=Ad(Gw);while(!(Gy==0xc0||Gy==0xc1||
Gy==0xc2)){if(!Ai(Gw,Gy))return 0;Gy=Ad(Gw);while(Gy==0xff){if(AD(Gw->A))return
b("no SOF");Gy=Ad(Gw);}}Gw->Q=(Gy==0xc2);if(!Al(Gw,Gx))return 0;return 1;}static
A An(AN*Gw){while(!AD(Gw->A)){A Gx=AC(Gw->A);while(Gx==0xff){if(AD(Gw->A))return
0xff;Gx=AC(Gw->A);if(Gx!=0&&Gx!=0xff)return Gx;}}return 0xff;}static int Ao(AN*
Gw){int Gx;for(Gx=0;Gx<4;Gx++){Gw->L[Gx].M=0;Gw->L[Gx].N=0;}Gw->b=0;if(!Am(Gw,//
STBI__SCAN_load))return 0;Gx=Ad(Gw);while(!(Gx==0xd9))if(Gx==0xda){if(!Aj(Gw))//
return 0;if(!Af(Gw))return 0;if(Gw->O==0xff)Gw->O=An(Gw);Gx=Ad(Gw);if(Gx>=0xd0&&
Gx<=0xd7)Gx=Ad(Gw);}else if(Gx==0xdc){int Gy=AG(Gw->A);F Gz=AG(Gw->A);if(Gy!=4)
return b("bad DNL len");if(Gz!=Gw->A->B)return b("bad DNL height");Gx=Ad(Gw);}//
else{if(!Ai(Gw,Gx))return 1;Gx=Ad(Gw);}if(Gw->Q)Ah(Gw);return 1;}typedef A*(*Ap)
(A*out,A*in0,A*in1,int w,int hs);static A*Aq(A*Gw,A*Gx,A*Gy,int Gz,int G0){/////
return Gx;}static A*Ar(A*Gw,A*Gx,A*Gy,int Gz,int G0){int G1;for(G1=0;G1<Gz;++G1)
Gw[G1]=((A)((3*Gx[G1]+Gy[G1]+2)>>2));return Gw;}static A*As(A*Gw,A*Gx,A*Gy,int//
Gz,int G0){int G1;A*G2=Gx;if(Gz==1){Gw[0]=Gw[1]=G2[0];return Gw;}Gw[0]=G2[0];Gw[
1]=((A)((G2[0]*3+G2[1]+2)>>2));for(G1=1;G1<Gz-1;++G1){int G3=3*G2[G1]+2;Gw[G1*2+
0]=((A)((G3+G2[G1-1])>>2));Gw[G1*2+1]=((A)((G3+G2[G1+1])>>2));}Gw[G1*2+0]=((A)((
G2[Gz-2]*3+G2[Gz-1]+2)>>2));Gw[G1*2+1]=G2[Gz-1];return Gw;}static A*At(A*Gw,A*Gx
,A*Gy,int Gz,int G0){int G1,G2,G3;if(Gz==1){Gw[0]=Gw[1]=((A)((3*Gx[0]+Gy[0]+2)>>
2));return Gw;}G3=3*Gx[0]+Gy[0];Gw[0]=((A)((G3+2)>>2));for(G1=1;G1<Gz;++G1){G2=
G3;G3=3*Gx[G1]+Gy[G1];Gw[G1*2-1]=((A)((3*G2+G3+8)>>4));Gw[G1*2]=((A)((3*G3+G2+8)
>>4));}Gw[Gz*2-1]=((A)((G3+2)>>2));return Gw;}static A*Au(A*Gw,A*Gx,A*Gy,int Gz,
int G0){int G1,G2;for(G1=0;G1<Gz;++G1)for(G2=0;G2<G0;++G2)Gw[G1*G0+G2]=Gx[G1];//
return Gw;}static void Av(A*Gw,const A*Gx,const A*Gy,const A*Gz,int G0,int G1){
int G2;for(G2=0;G2<G0;++G2){int G3=(Gx[G2]<<20)+524288;int G4,G5,G6;int G7=Gz[G2
]-128;int G8=Gy[G2]-128;G4=G3+G7*1470208;G5=G3+(G7*-748800)+((G8*-360960)&//////
0xffff0000);G6=G3+G8*1858048;G4>>=20;G5>>=20;G6>>=20;if((unsigned)G4>255)if(G4<0
)G4=0;else G4=255;if((unsigned)G5>255)if(G5<0)G5=0;else G5=255;if((unsigned)G6>
255)if(G6<0)G6=0;else G6=255;Gw[0]=(A)G4;Gw[1]=(A)G5;Gw[2]=(A)G6;Gw[3]=255;Gw+=
G1;}}static void Aw(AN*Gw){Gw->d=Ac;Gw->e=Av;Gw->f=At;}static void Ax(AN*Gw){Ak(
Gw,Gw->A->C,0);}typedef struct{Ap A;A*B,*C;int D,E;int F;int G;int H;}Ay;static
A Az(A Gw,A Gx){unsigned int Gy=Gw*Gx+128;return(A)((Gy+(Gy>>8))>>8);}static A*
A0(AN*Gw,int*Gx,int*Gy,int*Gz,int G0){int G1,G2,G3;Gw->A->C=0;if(G0<0||G0>4)////
return(unsigned char*)(size_t)(b("bad req_comp")?0:0);if(!Ao(Gw)){Ax(Gw);return
0;}G1=G0?G0:Gw->A->C>=3?3:1;G3=Gw->A->C==3&&(Gw->Y==3||(Gw->X==0&&!Gw->W));if(Gw
->A->C==3&&G1<3&&!G3)G2=1;else G2=Gw->A->C;if(G2<=0){Ax(Gw);return 0;}{int G4;//
unsigned int G5,G6;A*G7;A*G8[4]={0,0,0,0};Ay G9[4];for(G4=0;G4<G2;++G4){Ay*G_=&
G9[G4];Gw->L[G4].O=(A*)c(Gw->A->A+3);if(!Gw->L[G4].O){Ax(Gw);return(unsigned////
char*)(size_t)(b("outofmem")?0:0);}G_->D=Gw->F/Gw->L[G4].B;G_->E=Gw->G/Gw->L[G4]
.C;G_->G=G_->E>>1;G_->F=(Gw->A->A+G_->D-1)/G_->D;G_->H=0;G_->B=G_->C=Gw->L[G4].L
;if(G_->D==1&&G_->E==1)G_->A=Aq;else if(G_->D==1&&G_->E==2)G_->A=Ar;else if(G_->
D==2&&G_->E==1)G_->A=As;else if(G_->D==2&&G_->E==2)G_->A=Gw->f;else G_->A=Au;}G7
=(A*)j(G1,Gw->A->A,Gw->A->B,1);if(!G7){Ax(Gw);return(unsigned char*)(size_t)(b(
"outofmem")?0:0);}for(G6=0;G6<Gw->A->B;++G6){A*G_=G7+G1*Gw->A->A*G6;for(G4=0;G4<
G2;++G4){Ay*HA=&G9[G4];int HB=HA->G>=(HA->E>>1);G8[G4]=HA->A(Gw->L[G4].O,HB?HA->
C:HA->B,HB?HA->B:HA->C,HA->F,HA->D);if(++HA->G>=HA->E){HA->G=0;HA->B=HA->C;if(++
HA->H<Gw->L[G4].I)HA->C+=Gw->L[G4].J;}}if(G1>=3){A*HA=G8[0];if(Gw->A->C==3){if(
G3)for(G5=0;G5<Gw->A->A;++G5){G_[0]=HA[G5];G_[1]=G8[1][G5];G_[2]=G8[2][G5];G_[3]
=255;G_+=G1;}else Gw->e(G_,HA,G8[1],G8[2],Gw->A->A,G1);}else if(Gw->A->C==4){if(
Gw->X==0)for(G5=0;G5<Gw->A->A;++G5){A HB=G8[3][G5];G_[0]=Az(G8[0][G5],HB);G_[1]=
Az(G8[1][G5],HB);G_[2]=Az(G8[2][G5],HB);G_[3]=255;G_+=G1;}else if(Gw->X==2){Gw->
e(G_,HA,G8[1],G8[2],Gw->A->A,G1);for(G5=0;G5<Gw->A->A;++G5){A HB=G8[3][G5];G_[0]
=Az(255-G_[0],HB);G_[1]=Az(255-G_[1],HB);G_[2]=Az(255-G_[2],HB);G_+=G1;}}else Gw
->e(G_,HA,G8[1],G8[2],Gw->A->A,G1);}else for(G5=0;G5<Gw->A->A;++G5){G_[0]=G_[1]=
G_[2]=HA[G5];G_[3]=255;G_+=G1;}}else if(G3){if(G1==1)for(G5=0;G5<Gw->A->A;++G5)*
G_++ =AI(G8[0][G5],G8[1][G5],G8[2][G5]);else for(G5=0;G5<Gw->A->A;++G5,G_+=2){G_
[0]=AI(G8[0][G5],G8[1][G5],G8[2][G5]);G_[1]=255;}}else if(Gw->A->C==4&&Gw->X==0)
for(G5=0;G5<Gw->A->A;++G5){A HA=G8[3][G5];A HB=Az(G8[0][G5],HA);A HC=Az(G8[1][G5
],HA);A HD=Az(G8[2][G5],HA);G_[0]=AI(HB,HC,HD);G_[1]=255;G_+=G1;}else if(Gw->A->
C==4&&Gw->X==2)for(G5=0;G5<Gw->A->A;++G5){G_[0]=Az(255-G8[0][G5],G8[3][G5]);G_[1
]=255;G_+=G1;}else{A*HA=G8[0];if(G1==1)for(G5=0;G5<Gw->A->A;++G5)G_[G5]=HA[G5];
else for(G5=0;G5<Gw->A->A;++G5){*G_++ =HA[G5];*G_++ =255;}}}Ax(Gw);*Gx=Gw->A->A;
*Gy=Gw->A->B;if(Gz)*Gz=Gw->A->C>=3?3:1;return G7;}}static void*U(I*G2,int*G3,int
*G4,int*G5,int G6,S*G7){unsigned char*G8;AN*G9=(AN*)c(18568);if(!G9)return((////
unsigned char*)(size_t)(b("outofmem")?0:0));memset(G9,0,18568);G9->A=G2;Aw(G9);
G8=A0(G9,G3,G4,G5,G6);free(G9);return G8;}static int T(I*Gx){int Gy;AN*Gz=(AN*)c
(18568);if(!Gz)return b("outofmem");memset(Gz,0,18568);Gz->A=Gx;Aw(Gz);Gy=Am(Gz,
STBI__SCAN_type);R(Gx);free(Gz);return Gy;}static int A1(AN*Gw,int*Gx,int*Gy,int
*Gz){if(!Am(Gw,STBI__SCAN_header)){R(Gw->A);return 0;}if(Gx)*Gx=Gw->A->A;if(Gy)*
Gy=Gw->A->B;if(Gz)*Gz=Gw->A->C>=3?3:1;return 1;}static int V(I*G0,int*G1,int*G2,
int*G3){int G4;AN*G5=(AN*)c(18568);if(!G5)return b("outofmem");memset(G5,0,18568
);G5->A=G0;G4=A1(G5,G1,G2,G3);free(G5);return G4;}typedef struct{D A[512];D B[16
];int C[17];D D[16];A E[288];D F[288];}A2;static int A3(int Gw){Gw=((Gw&0xAAAA)
>>1)|((Gw&0x5555)<<1);Gw=((Gw&0xCCCC)>>2)|((Gw&0x3333)<<2);Gw=((Gw&0xF0F0)>>4)|(
(Gw&3855)<<4);Gw=((Gw&0xFF00)>>8)|((Gw&255)<<8);return Gw;}static int A4(int Gw,
int Gx){return A3(Gw)>>(16-Gx);}static int A5(A2*Gw,const A*Gx,int Gy){int Gz,G0
=0;int G1,G2[16],G3[17];memset(G3,0,68);memset(Gw->A,0,1024);for(Gz=0;Gz<Gy;++Gz
)++G3[Gx[Gz]];G3[0]=0;for(Gz=1;Gz<16;++Gz)if(G3[Gz]>(1<<Gz))return b("bad sizes"
);G1=0;for(Gz=1;Gz<16;++Gz){G2[Gz]=G1;Gw->B[Gz]=(D)G1;Gw->D[Gz]=(D)G0;G1=(G1+G3[
Gz]);if(G3[Gz])if(G1-1>=(1<<Gz))return b("bad codelengths");Gw->C[Gz]=G1<<(16-Gz
);G1<<=1;G0+=G3[Gz];}Gw->C[16]=65536;for(Gz=0;Gz<Gy;++Gz){int G4=Gx[Gz];if(G4){
int G5=G2[G4]-Gw->B[G4]+Gw->D[G4];D G6=(D)((G4<<9)|Gz);Gw->E[G5]=(A)G4;Gw->F[G5]
=(D)Gz;if(G4<=9){int G7=A4(G2[G4],G4);while(G7<512){Gw->A[G7]=G6;G7+=(1<<G4);}}
++G2[G4];}}return 1;}typedef struct{A*A,*B;int C;int D;F E;char*F;char*G;char*H;
int I;A2 J,K;}A6;static int A7(A6*Gw){return Gw->A>=Gw->B;}static A A8(A6*Gw){//
return A7(Gw)?0:*Gw->A++;}static void A9(A6*Gw){do{if(Gw->E>=(1U<<Gw->C)){Gw->A=
Gw->B;return;}Gw->E|=(unsigned int)A8(Gw)<<Gw->C;Gw->C+=8;}while(Gw->C<=24);}///
static unsigned int A_(A6*Gw,int Gx){unsigned int Gy;if(Gw->C<Gx)A9(Gw);Gy=Gw->E
&((1<<Gx)-1);Gw->E>>=Gx;Gw->C-=Gx;return Gy;}static int BA(A6*Gw,A2*Gx){int Gy,
Gz,G0;G0=A4(Gw->E,16);for(Gz=10;;++Gz)if(G0<Gx->C[Gz])break;if(Gz>=16)return-1;
Gy=(G0>>(16-Gz))-Gx->B[Gz]+Gx->D[Gz];if(Gy>=288)return-1;if(Gx->E[Gy]!=Gz)return
-1;Gw->E>>=Gz;Gw->C-=Gz;return Gx->F[Gy];}static int BB(A6*Gw,A2*Gx){int Gy,Gz;
if(Gw->C<16)if(A7(Gw)){if(!Gw->D){Gw->D=1;Gw->C+=16;}else return-1;}else A9(Gw);
Gy=Gx->A[Gw->E&511];if(Gy){Gz=Gy>>9;Gw->E>>=Gz;Gw->C-=Gz;return Gy&511;}return//
BA(Gw,Gx);}static int BC(A6*Gw,char*Gx,int Gy){char*Gz;unsigned int G0,G1,G2;Gw
->F=Gx;if(!Gw->I)return b("output buffer limit");G0=(unsigned int)(Gw->F-Gw->G);
G1=G2=(unsigned)(Gw->H-Gw->G);if(UINT_MAX-G0<(unsigned)Gy)return b("outofmem");
while(G0+Gy>G1){if(G1>2147483647)return b("outofmem");G1*=2;}Gz=(char*)realloc(
Gw->G,G1);if(Gz==0)return b("outofmem");Gw->G=Gz;Gw->F=Gz+G0;Gw->H=Gz+G1;return
1;}static const int BD[31]={3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59
,67,83,99,115,131,163,195,227,258,0,0};static const int BE[31]={0,0,0,0,0,0,0,0,
1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0,0,0};static const int BF[32]={1,2,3,4,
5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145
,8193,12289,16385,24577,0,0};static const int BG[32]={0,0,0,0,1,1,2,2,3,3,4,4,5,
5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};static int BH(A6*Gw){char*Gx=Gw->F;//
for(;;){int Gy=BB(Gw,&Gw->J);if(Gy<256){if(Gy<0)return b("bad huffman code");if(
Gx>=Gw->H){if(!BC(Gw,Gx,1))return 0;Gx=Gw->F;}*Gx++ =(char)Gy;}else{A*Gz;int G0,
G1;if(Gy==256){Gw->F=Gx;if(Gw->D&&Gw->C<16)return b("unexpected end");return 1;}
if(Gy>=286)return b("bad huffman code");Gy-=257;G0=BD[Gy];if(BE[Gy])G0+=A_(Gw,BE
[Gy]);Gy=BB(Gw,&Gw->K);if(Gy<0||Gy>=30)return b("bad huffman code");G1=BF[Gy];if
(BG[Gy])G1+=A_(Gw,BG[Gy]);if(Gx-Gw->G<G1)return b("bad dist");if(G0>Gw->H-Gx){if
(!BC(Gw,Gx,G0))return 0;Gx=Gw->F;}Gz=(A*)(Gx-G1);if(G1==1){A G2=*Gz;if(G0)do*Gx
++ =G2;while(--G0);}else if(G0)do*Gx++ =*Gz++;while(--G0);}}}static int BI(A6*Gw
){static const A Gx[19]={16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};A2 Gy;A
Gz[455];A G0[19];int G1,G2;int G3=A_(Gw,5)+257;int G4=A_(Gw,5)+1;int G5=A_(Gw,4)
+4;int G6=G3+G4;memset(G0,0,19);for(G1=0;G1<G5;++G1){int G7=A_(Gw,3);G0[Gx[G1]]=
(A)G7;}if(!A5(&Gy,G0,19))return 0;G2=0;while(G2<G6){int G7=BB(Gw,&Gy);if(G7<0||
G7>=19)return b("bad codelengths");if(G7<16)Gz[G2++]=(A)G7;else{A G8=0;if(G7==16
){G7=A_(Gw,2)+3;if(G2==0)return b("bad codelengths");G8=Gz[G2-1];}else if(G7==17
)G7=A_(Gw,3)+3;else if(G7==18)G7=A_(Gw,7)+11;else return b("bad codelengths");if
(G6-G2<G7)return b("bad codelengths");memset(Gz+G2,G8,G7);G2+=G7;}}if(G2!=G6)///
return b("bad codelengths");if(!A5(&Gw->J,Gz,G3))return 0;if(!A5(&Gw->K,Gz+G3,G4
))return 0;return 1;}static int BJ(A6*Gw){A Gx[4];int Gy,Gz,G0;if(Gw->C&7)A_(Gw,
Gw->C&7);G0=0;while(Gw->C>0){Gx[G0++]=(A)(Gw->E&255);Gw->E>>=8;Gw->C-=8;}if(Gw->
C<0)return b("zlib corrupt");while(G0<4)Gx[G0++]=A8(Gw);Gy=Gx[1]*256+Gx[0];Gz=Gx
[3]*256+Gx[2];if(Gz!=(Gy^0xffff))return b("zlib corrupt");if(Gw->A+Gy>Gw->B)////
return b("read past buffer");if(Gw->F+Gy>Gw->H)if(!BC(Gw,Gw->F,Gy))return 0;////
memcpy(Gw->F,Gw->A,Gy);Gw->A+=Gy;Gw->F+=Gy;return 1;}static int BK(A6*Gw){int Gx
=A8(Gw);int Gy=Gx&15;int Gz=A8(Gw);if(A7(Gw))return b("bad zlib header");if((Gx*
256+Gz)%31!=0)return b("bad zlib header");if(Gz&32)return b("no preset dict");if
(Gy!=8)return b("bad compression");return 1;}static const A BL[288]={8,8,8,8,8,8
,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8
,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8
,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8
,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9
,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9
,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9
,9,9,9,9,9,9,9,9,9,9,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8
,8,8};static const A BM[32]={5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5
,5,5,5,5,5,5};static int BN(A6*Gw,int Gx){int Gy,Gz;if(Gx)if(!BK(Gw))return 0;Gw
->C=0;Gw->E=0;Gw->D=0;do{Gy=A_(Gw,1);Gz=A_(Gw,2);if(Gz==0){if(!BJ(Gw))return 0;}
else if(Gz==3)return 0;else{if(Gz==1){if(!A5(&Gw->J,BL,288))return 0;if(!A5(&Gw
->K,BM,32))return 0;}else if(!BI(Gw))return 0;if(!BH(Gw))return 0;}}while(!Gy);
return 1;}static int BO(A6*Gw,char*Gx,int Gy,int Gz,int G0){Gw->G=Gx;Gw->F=Gx;Gw
->H=Gx+Gy;Gw->I=Gz;return BN(Gw,G0);}extern char*///////////////////////////////
stbi_zlib_decode_malloc_guesssize(const char*Gw,int Gx,int Gy,int*Gz){A6 G0;char
*G1=(char*)c(Gy);if(G1==0)return 0;G0.A=(A*)Gw;G0.B=(A*)Gw+Gx;if(BO(&G0,G1,Gy,1,
1)){if(Gz)*Gz=(int)(G0.F-G0.G);return G0.G;}else{free(G0.G);return 0;}}extern///
char*stbi_zlib_decode_malloc(char const*Gw,int Gx,int*Gy){return////////////////
stbi_zlib_decode_malloc_guesssize(Gw,Gx,16384,Gy);}extern char*/////////////////
stbi_zlib_decode_malloc_guesssize_headerflag(const char*Gw,int Gx,int Gy,int*Gz,
int G0){A6 G1;char*G2=(char*)c(Gy);if(G2==0)return 0;G1.A=(A*)Gw;G1.B=(A*)Gw+Gx;
if(BO(&G1,G2,Gy,1,G0)){if(Gz)*Gz=(int)(G1.F-G1.G);return G1.G;}else{free(G1.G);
return 0;}}extern int stbi_zlib_decode_buffer(char*Gw,int Gx,char const*Gy,int//
Gz){A6 G0;G0.A=(A*)Gy;G0.B=(A*)Gy+Gz;if(BO(&G0,Gw,Gx,0,1))return(int)(G0.F-G0.G)
;else return-1;}extern char*stbi_zlib_decode_noheader_malloc(char const*Gw,int//
Gx,int*Gy){A6 Gz;char*G0=(char*)c(16384);if(G0==0)return 0;Gz.A=(A*)Gw;Gz.B=(A*)
Gw+Gx;if(BO(&Gz,G0,16384,1,0)){if(Gy)*Gy=(int)(Gz.F-Gz.G);return Gz.G;}else{free
(Gz.G);return 0;}}extern int stbi_zlib_decode_noheader_buffer(char*Gw,int Gx,///
const char*Gy,int Gz){A6 G0;G0.A=(A*)Gy;G0.B=(A*)Gy+Gz;if(BO(&G0,Gw,Gx,0,0))////
return(int)(G0.F-G0.G);else return-1;}typedef struct{F A;F B;}BP;static BP BQ(I*
Gw){BP Gx;Gx.A=AH(Gw);Gx.B=AH(Gw);return Gx;}static int BR(I*Gw){static const A
Gx[8]={137,80,78,71,13,10,26,10};int Gy;for(Gy=0;Gy<8;++Gy)if(AC(Gw)!=Gx[Gy])///
return b("bad png sig");return 1;}typedef struct{I*A;A*B,*C,*D;int E;}BS;enum{//
STBI__F_none=0,STBI__F_sub=1,STBI__F_up=2,STBI__F_avg=3,STBI__F_paeth=4,////////
STBI__F_avg_first};static A BT[5]={STBI__F_none,STBI__F_sub,STBI__F_none,///////
STBI__F_avg_first,STBI__F_sub};static int BU(int Gw,int Gx,int Gy){int Gz=Gy*3-(
Gw+Gx);int G0=Gw<Gx?Gw:Gx;int G1=Gw<Gx?Gx:Gw;int G2=(G1<=Gz)?G0:Gy;int G3=(Gz<=
G0)?G1:G2;return G3;}static const A BV[9]={0,0xff,85,0,17,0,0,0,1};static void//
BW(A*Gw,A*Gx,F Gy,int Gz){int G0;if(Gz==1)for(G0=Gy-1;G0>=0;--G0){Gw[G0*2+1]=255
;Gw[G0*2+0]=Gx[G0];}else for(G0=Gy-1;G0>=0;--G0){Gw[G0*4+3]=255;Gw[G0*4+2]=Gx[G0
*3+2];Gw[G0*4+1]=Gx[G0*3+1];Gw[G0*4+0]=Gx[G0*3+0];}}static int BX(BS*Gw,A*Gx,F//
Gy,int Gz,F G0,F G1,int G2,int G3){int G4=G2==16?2:1;I*G5=Gw->A;F G6,G7,G8=G0*Gz
*G4;F G9,G_;A*HA;int HB=1;int HC;int HD=G5->C;int HE=Gz*G4;int HF=HD*G4;int HG=
G0;Gw->D=(A*)j(G0,G1,HE,0);if(!Gw->D)return b("outofmem");if(!g(HD,G0,G2,7))////
return b("too large");G_=(((HD*G0*G2)+7)>>3);if(!f(G_,G1,G_))return b(//////////
"too large");G9=(G_+1)*G1;if(Gy<G9)return b("not enough pixels");HA=(A*)i(G_,2,0
);if(!HA)return b("outofmem");if(G2<8){HF=1;HG=G_;}for(G7=0;G7<G1;++G7){A*HH=HA+
(G7&1)*G_;A*HI=HA+(~G7&1)*G_;A*HJ=Gw->D+G8*G7;int HK=HG*HF;int HL=*Gx++;if(HL>4)
{HB=b("invalid filter");break;}if(G7==0)HL=BT[HL];switch(HL){case STBI__F_none:
memcpy(HH,Gx,HK);break;case STBI__F_sub:memcpy(HH,Gx,HF);for(HC=HF;HC<HK;++HC)HH
[HC]=((A)((Gx[HC]+HH[HC-HF])&255));break;case STBI__F_up:for(HC=0;HC<HK;++HC)HH[
HC]=((A)((Gx[HC]+HI[HC])&255));break;case STBI__F_avg:for(HC=0;HC<HF;++HC)HH[HC]
=((A)((Gx[HC]+(HI[HC]>>1))&255));for(HC=HF;HC<HK;++HC)HH[HC]=((A)((Gx[HC]+((HI[
HC]+HH[HC-HF])>>1))&255));break;case STBI__F_paeth:for(HC=0;HC<HF;++HC)HH[HC]=((
A)((Gx[HC]+HI[HC])&255));for(HC=HF;HC<HK;++HC)HH[HC]=((A)((Gx[HC]+BU(HH[HC-HF],
HI[HC],HI[HC-HF]))&255));break;case STBI__F_avg_first:memcpy(HH,Gx,HF);for(HC=HF
;HC<HK;++HC)HH[HC]=((A)((Gx[HC]+(HH[HC-HF]>>1))&255));break;}Gx+=HK;if(G2<8){A//
HM=(G3==0)?BV[G2]:1;A*HN=HH;A*HO=HJ;A HP=0;F HQ=G0*HD;if(G2==4)for(G6=0;G6<HQ;++
G6){if((G6&1)==0)HP=*HN++;*HO++ =HM*(HP>>4);HP<<=4;}else if(G2==2)for(G6=0;G6<HQ
;++G6){if((G6&3)==0)HP=*HN++;*HO++ =HM*(HP>>6);HP<<=2;}else for(G6=0;G6<HQ;++G6)
{if((G6&7)==0)HP=*HN++;*HO++ =HM*(HP>>7);HP<<=1;}if(HD!=Gz)BW(HJ,HJ,G0,HD);}else
if(G2==8){if(HD==Gz)memcpy(HJ,HH,G0*HD);else BW(HJ,HH,G0,HD);}else if(G2==16){D*
HM=(D*)HJ;F HN=G0*HD;if(HD==Gz)for(G6=0;G6<HN;++G6,++HM,HH+=2)*HM=(HH[0]<<8)|HH[
1];else if(HD==1)for(G6=0;G6<G0;++G6,HM+=2,HH+=2){HM[0]=(HH[0]<<8)|HH[1];HM[1]=
0xffff;}else for(G6=0;G6<G0;++G6,HM+=4,HH+=6){HM[0]=(HH[0]<<8)|HH[1];HM[1]=(HH[2
]<<8)|HH[3];HM[2]=(HH[4]<<8)|HH[5];HM[3]=0xffff;}}}free(HA);if(!HB)return 0;////
return 1;}static int BY(BS*Gw,A*Gx,F Gy,int Gz,int G0,int G1,int G2){int G3=G0==
16?2:1;int G4=Gz*G3;A*G5;int G6;if(!G2)return BX(Gw,Gx,Gy,Gz,Gw->A->A,Gw->A->B,
G0,G1);G5=(A*)j(Gw->A->A,Gw->A->B,G4,0);if(!G5)return b("outofmem");for(G6=0;G6<
7;++G6){int G7[]={0,4,0,2,0,1,0};int G8[]={0,0,4,0,2,0,1};int G9[]={8,8,4,4,2,2,
1};int G_[]={8,8,8,4,4,2,2};int HA,HB,HC,HD;HC=(Gw->A->A-G7[G6]+G9[G6]-1)/G9[G6]
;HD=(Gw->A->B-G8[G6]+G_[G6]-1)/G_[G6];if(HC&&HD){F HE=((((Gw->A->C*HC*G0)+7)>>3)
+1)*HD;if(!BX(Gw,Gx,Gy,Gz,HC,HD,G0,G1)){free(G5);return 0;}for(HB=0;HB<HD;++HB)
for(HA=0;HA<HC;++HA){int HF=HB*G_[G6]+G8[G6];int HG=HA*G9[G6]+G7[G6];memcpy(G5+
HF*Gw->A->A*G4+HG*G4,Gw->D+(HB*HC+HA)*G4,G4);}free(Gw->D);Gx+=HE;Gy-=HE;}}Gw->D=
G5;return 1;}static int BZ(BS*Gw,A Gx[3],int Gy){I*Gz=Gw->A;F G0,G1=Gz->A*Gz->B;
A*G2=Gw->D;if(Gy==2)for(G0=0;G0<G1;++G0){G2[1]=(G2[0]==Gx[0]?0:255);G2+=2;}else
for(G0=0;G0<G1;++G0){if(G2[0]==Gx[0]&&G2[1]==Gx[1]&&G2[2]==Gx[2])G2[3]=0;G2+=4;}
return 1;}static int Ba(BS*Gw,D Gx[3],int Gy){I*Gz=Gw->A;F G0,G1=Gz->A*Gz->B;D*
G2=(D*)Gw->D;if(Gy==2)for(G0=0;G0<G1;++G0){G2[1]=(G2[0]==Gx[0]?0:65535);G2+=2;}
else for(G0=0;G0<G1;++G0){if(G2[0]==Gx[0]&&G2[1]==Gx[1]&&G2[2]==Gx[2])G2[3]=0;G2
+=4;}return 1;}static int Bb(BS*Gw,A*Gx,int Gy,int Gz){F G0,G1=Gw->A->A*Gw->A->B
;A*G2,*G3,*G4=Gw->D;G2=(A*)i(G1,Gz,0);if(G2==0)return b("outofmem");G3=G2;if(Gz
==3)for(G0=0;G0<G1;++G0){int G5=G4[G0]*4;G2[0]=Gx[G5];G2[1]=Gx[G5+1];G2[2]=Gx[G5
+2];G2+=3;}else for(G0=0;G0<G1;++G0){int G5=G4[G0]*4;G2[0]=Gx[G5];G2[1]=Gx[G5+1]
;G2[2]=Gx[G5+2];G2[3]=Gx[G5+3];G2+=4;}free(Gw->D);Gw->D=G3;return 1;}static int
Bc=0;static int Bd=0;extern void stbi_set_unpremultiply_on_load(int Gw){Bc=Gw;}
extern void stbi_convert_iphone_png_to_rgb(int Gw){Bd=Gw;}static _Thread_local//
int Be,Bf;static _Thread_local int Bg,Bh;extern void////////////////////////////
stbi_set_unpremultiply_on_load_thread(int Gw){Be=Gw;Bf=1;}extern void///////////
stbi_convert_iphone_png_to_rgb_thread(int Gw){Bg=Gw;Bh=1;}static void Bi(BS*Gw){
I*Gx=Gw->A;F Gy,Gz=Gx->A*Gx->B;A*G0=Gw->D;if(Gx->D==3)for(Gy=0;Gy<Gz;++Gy){A G1=
G0[0];G0[0]=G0[2];G0[2]=G1;G0+=3;}else if(Bf?Be:Bc)for(Gy=0;Gy<Gz;++Gy){A G1=G0[
3];A G2=G0[0];if(G1){A G3=G1/2;G0[0]=(G0[2]*255+G3)/G1;G0[1]=(G0[1]*255+G3)/G1;
G0[2]=(G2*255+G3)/G1;}else{G0[0]=G0[2];G0[2]=G2;}G0+=4;}else for(Gy=0;Gy<Gz;++Gy
){A G1=G0[0];G0[0]=G0[2];G0[2]=G1;G0+=4;}}static int Bj(BS*Gw,int Gx,int Gy){A//
Gz[1024],G0=0;A G1=0,G2[3]={0};D G3[3];F G4=0,G5=0,G6,G7=0;int G8=1,G9,G_=0,HA=0
,HB=0;I*HC=Gw->A;Gw->C=0;Gw->B=0;Gw->D=0;if(!BR(HC))return 0;if(Gx==////////////
STBI__SCAN_type)return 1;for(;;){BP HD=BQ(HC);switch(HD.B){case 1130840649:HB=1;
AE(HC,HD.A);break;case 1229472850:{int HE,HF;if(!G8)return b("multiple IHDR");G8
=0;if(HD.A!=13)return b("bad IHDR len");HC->A=AH(HC);HC->B=AH(HC);if(HC->B>/////
16777216)return b("too large");if(HC->A>16777216)return b("too large");Gw->E=AC(
HC);if(Gw->E!=1&&Gw->E!=2&&Gw->E!=4&&Gw->E!=8&&Gw->E!=16)return b(//////////////
"1/2/4/8/16-bit only");HA=AC(HC);if(HA>6)return b("bad ctype");if(HA==3&&Gw->E==
16)return b("bad ctype");if(HA==3)G0=3;else if(HA&1)return b("bad ctype");HE=AC(
HC);if(HE)return b("bad comp method");HF=AC(HC);if(HF)return b(/////////////////
"bad filter method");G_=AC(HC);if(G_>1)return b("bad interlace method");if(!HC->
A||!HC->B)return b("0-pixel image");if(!G0){HC->C=(HA&2?3:1)+(HA&4?1:0);if((1<<
30)/HC->A/HC->C<HC->B)return b("too large");}else{HC->C=1;if((1<<30)/HC->A/4<HC
->B)return b("too large");}break;}case 1347179589:{if(G8)return b(//////////////
"first not IHDR");if(HD.A>768)return b("invalid PLTE");G7=HD.A/3;if(G7*3!=HD.A)
return b("invalid PLTE");for(G6=0;G6<G7;++G6){Gz[G6*4+0]=AC(HC);Gz[G6*4+1]=AC(HC
);Gz[G6*4+2]=AC(HC);Gz[G6*4+3]=255;}break;}case 1951551059:{if(G8)return b(/////
"first not IHDR");if(Gw->B)return b("tRNS after IDAT");if(G0){if(Gx==///////////
STBI__SCAN_header){HC->C=4;return 1;}if(G7==0)return b("tRNS before PLTE");if(HD
.A>G7)return b("bad tRNS len");G0=4;for(G6=0;G6<HD.A;++G6)Gz[G6*4+3]=AC(HC);}///
else{if(!(HC->C&1))return b("tRNS with alpha");if(HD.A!=(F)HC->C*2)return b(////
"bad tRNS len");G1=1;if(Gx==STBI__SCAN_header){++HC->C;return 1;}if(Gw->E==16)//
for(G9=0;G9<HC->C&&G9<3;++G9)G3[G9]=(D)AG(HC);else for(G9=0;G9<HC->C&&G9<3;++G9)
G2[G9]=(A)(AG(HC)&255)*BV[Gw->E];}break;}case 1229209940:{if(G8)return b(///////
"first not IHDR");if(G0&&!G7)return b("no PLTE");if(Gx==STBI__SCAN_header){if(G0
)HC->C=G0;return 1;}if(HD.A>(1u<<30))return b("IDAT size limit");if((int)(G4+HD.
A)<(int)G4)return 0;if(G4+HD.A>G5){F HE=G5;A*HF;if(G5==0)G5=HD.A>4096?HD.A:4096;
while(G4+HD.A>G5)G5*=2;HF=(A*)realloc(Gw->B,G5);if(HF==0)return b("outofmem");Gw
->B=HF;}if(!AF(HC,Gw->B+G4,HD.A))return b("outofdata");G4+=HD.A;break;}case/////
1229278788:{F HE,HF;if(G8)return b("first not IHDR");if(Gx!=STBI__SCAN_load)////
return 1;if(Gw->B==0)return b("no IDAT");HF=(HC->A*Gw->E+7)/8;HE=HF*HC->B*HC->C+
HC->B;Gw->C=(A*)stbi_zlib_decode_malloc_guesssize_headerflag((char*)Gw->B,G4,HE,
(int*)&HE,!HB);if(Gw->C==0)return 0;free(Gw->B);Gw->B=0;if((Gy==HC->C+1&&Gy!=3&&
!G0)||G1)HC->D=HC->C+1;else HC->D=HC->C;if(!BY(Gw,Gw->C,HE,HC->D,Gw->E,HA,G_))//
return 0;if(G1)if(Gw->E==16){if(!Ba(Gw,G3,HC->D))return 0;}else if(!BZ(Gw,G2,HC
->D))return 0;if(HB&&(Bh?Bg:Bd)&&HC->D>2)Bi(Gw);if(G0){HC->C=G0;HC->D=G0;if(Gy>=
3)HC->D=Gy;if(!Bb(Gw,Gz,G7,HC->D))return 0;}else if(G1)++HC->C;free(Gw->C);Gw->C
=0;AH(HC);return 1;}default:if(G8)return b("first not IHDR");if((HD.B&(1<<29))==
0){static char HE[]="XXXX PNG chunk not known";HE[0]=((A)((HD.B>>24)&255));HE[1]
=((A)((HD.B>>16)&255));HE[2]=((A)((HD.B>>8)&255));HE[3]=((A)((HD.B>>0)&255));///
return b(HE);}AE(HC,HD.A);break;}AH(HC);}}static void*Bk(BS*Gw,int*Gx,int*Gy,int
*Gz,int G0,S*G1){void*G2=0;if(G0<0||G0>4)return((unsigned char*)(size_t)(b(/////
"bad req_comp")?0:0));if(Bj(Gw,STBI__SCAN_load,G0)){if(Gw->E<=8)G1->A=8;else if(
Gw->E==16)G1->A=16;else return((unsigned char*)(size_t)(b("bad bits_per_channel"
)?0:0));G2=Gw->D;Gw->D=0;if(G0&&G0!=Gw->A->D){if(G1->A==8)G2=AJ((unsigned char*)
G2,Gw->A->D,G0,Gw->A->A,Gw->A->B);else G2=AL((D*)G2,Gw->A->D,G0,Gw->A->A,Gw->A->
B);Gw->A->D=G0;if(G2==0)return G2;}*Gx=Gw->A->A;*Gy=Gw->A->B;if(Gz)*Gz=Gw->A->C;
}free(Gw->D);Gw->D=0;free(Gw->C);Gw->C=0;free(Gw->B);Gw->B=0;return G2;}static//
void*X(I*G2,int*G3,int*G4,int*G5,int G6,S*G7){BS G8;G8.A=G2;return Bk(&G8,G3,G4,
G5,G6,G7);}static int W(I*Gx){int Gy;Gy=BR(Gx);R(Gx);return Gy;}static int Bl(BS
*Gw,int*Gx,int*Gy,int*Gz){if(!Bj(Gw,STBI__SCAN_header,0)){R(Gw->A);return 0;}if(
Gx)*Gx=Gw->A->A;if(Gy)*Gy=Gw->A->B;if(Gz)*Gz=Gw->A->C;return 1;}static int Y(I*
G0,int*G1,int*G2,int*G3){BS G4;G4.A=G0;return Bl(&G4,G1,G2,G3);}static int Z(I*
Gx){BS Gy;Gy.A=Gx;if(!Bl(&Gy,0,0,0))return 0;if(Gy.E!=16){R(Gy.A);return 0;}////
return 1;}static int Bm(I*Gw,int*Gx,int*Gy,int*Gz){if(V(Gw,Gx,Gy,Gz))return 1;if
(Y(Gw,Gx,Gy,Gz))return 1;return b("unknown image type");}static int Bn(I*Gw){if(
Z(Gw))return 1;return 0;}extern int stbi_info_from_file(FILE*Gw,int*Gx,int*Gy,//
int*Gz){int G0;I G1;long G2=ftell(Gw);Q(&G1,Gw);G0=Bm(&G1,Gx,Gy,Gz);fseek(Gw,G2,
0);return G0;}extern int stbi_info(char const*Gw,int*Gx,int*Gy,int*Gz){FILE*G0=x
(Gw,"rb");int G1;if(!G0)return b("can't fopen");G1=stbi_info_from_file(G0,Gx,Gy,
Gz);fclose(G0);return G1;}extern int stbi_is_16_bit_from_file(FILE*Gw){int Gx;I
Gy;long Gz=ftell(Gw);Q(&Gy,Gw);Gx=Bn(&Gy);fseek(Gw,Gz,0);return Gx;}extern int//
stbi_is_16_bit(char const*Gw){FILE*Gx=x(Gw,"rb");int Gy;if(!Gx)return b(////////
"can't fopen");Gy=stbi_is_16_bit_from_file(Gx);fclose(Gx);return Gy;}extern int
stbi_info_from_memory(A const*Gw,int Gx,int*Gy,int*Gz,int*G0){I G1;K(&G1,Gw,Gx);
return Bm(&G1,Gy,Gz,G0);}extern int stbi_info_from_callbacks(C const*Gw,void*Gx,
int*Gy,int*Gz,int*G0){I G1;L(&G1,(C*)Gw,Gx);return Bm(&G1,Gy,Gz,G0);}extern int
stbi_is_16_bit_from_memory(A const*Gw,int Gx){I Gy;K(&Gy,Gw,Gx);return Bn(&Gy);}
extern int stbi_is_16_bit_from_callbacks(C const*Gw,void*Gx){I Gy;L(&Gy,(C*)Gw,
Gx);return Bn(&Gy);}typedef uint8_t Bo;typedef uint16_t Bp;typedef uint32_t Bq;
typedef uint64_t Br;typedef enum{STBIR_1CHANNEL=1,STBIR_2CHANNEL=2,STBIR_RGB=3,
STBIR_BGR=0,STBIR_4CHANNEL=5,STBIR_RGBA=4,STBIR_BGRA=6,STBIR_ARGB=7,STBIR_ABGR=8
,STBIR_RA=9,STBIR_AR=10,STBIR_RGBA_PM=11,STBIR_BGRA_PM=12,STBIR_ARGB_PM=13,/////
STBIR_ABGR_PM=14,STBIR_RA_PM=15,STBIR_AR_PM=16,STBIR_RGBA_NO_AW=11,/////////////
STBIR_BGRA_NO_AW=12,STBIR_ARGB_NO_AW=13,STBIR_ABGR_NO_AW=14,STBIR_RA_NO_AW=15,//
STBIR_AR_NO_AW=16,}Bs;typedef enum{STBIR_EDGE_CLAMP=0,STBIR_EDGE_REFLECT=1,/////
STBIR_EDGE_WRAP=2,STBIR_EDGE_ZERO=3,}Bt;typedef enum{STBIR_FILTER_DEFAULT=0,////
STBIR_FILTER_BOX=1,STBIR_FILTER_TRIANGLE=2,STBIR_FILTER_CUBICBSPLINE=3,/////////
STBIR_FILTER_CATMULLROM=4,STBIR_FILTER_MITCHELL=5,STBIR_FILTER_POINT_SAMPLE=6,//
STBIR_FILTER_OTHER=7,}Bu;typedef enum{STBIR_TYPE_UINT8=0,STBIR_TYPE_UINT8_SRGB=1
,STBIR_TYPE_UINT8_SRGB_ALPHA=2,STBIR_TYPE_UINT16=3,STBIR_TYPE_FLOAT=4,//////////
STBIR_TYPE_HALF_FLOAT=5}Bv;typedef void const*Bw(void*optional_output,void const
*input_ptr,int num_pixels,int x,int y,void*context);typedef void Bx(void const*
output_ptr,int num_pixels,int y,void*context);typedef float By(float x,float////
scale,void*user_data);typedef float Bz(float scale,void*user_data);typedef//////
struct A B0;typedef struct B{void*A;void const*B;int C,D;double E,F,G,H;Bw*I;///
void*J;int K,L;int M,N,O,P;Bx*Q;int R;int S;int T;int U;int V;int W;Bs X;Bs Y;Bv
Z;Bv a;Bu b,c;Bt d,e;By*f;Bz*g;By*h;Bz*i;B0*j;}B1;typedef enum{STBIRI_1CHANNEL=0
,STBIRI_2CHANNEL=1,STBIRI_RGB=2,STBIRI_BGR=3,STBIRI_4CHANNEL=4,STBIRI_RGBA=5,///
STBIRI_BGRA=6,STBIRI_ARGB=7,STBIRI_ABGR=8,STBIRI_RA=9,STBIRI_AR=10,/////////////
STBIRI_RGBA_PM=11,STBIRI_BGRA_PM=12,STBIRI_ARGB_PM=13,STBIRI_ABGR_PM=14,////////
STBIRI_RA_PM=15,STBIRI_AR_PM=16,}B2;static unsigned char B3[]={1,1,1,2,4,2};////
typedef struct{int A;int B;}B4;typedef struct{int A;int B;int C;}B5;typedef/////
struct{int A;int B;int C;}B6;typedef struct C{int A;int B;float C;float D;float
E;int F;Bq G,H;}B7;typedef struct{B4*A;float*B;B4*C;float*D;B7 E;float F;Bu G;By
*H;Bz*I;Bt J;int K;int L;int M;int N;int O;int P;B5 Q;int R;int S;int T;int U;//
int V;}B8;typedef struct{B4 A;int B[2];B6 C[2];}B9;typedef struct{float*A;int B;
int C;int D;int E,F;int G,H;float*I;float*J;char K[64];}B_;typedef float*CA(////
float*,int,void const*);typedef void CB(float*,int);typedef void CC(float*,/////
unsigned int,float const*,B4 const*,float const*,int);typedef void CD(float*,int
);typedef void CE(void*,int,float const*);struct A{B8 A;B8 B;void const*C;void*D
;int E;int F;int G;int H;Bv I;Bv J;Bw*K;void*L;Bx*M;B9 N;void*O;B_*P;CA*Q;CB*R;
CC*S;CD*T;CE*U;int V;int W;B2 X;B2 Y;int Z;int a,b;int c;int d;int e;size_t f;};
static inline int CF(int Gw,int Gx){return Gw<Gx?Gw:Gx;}static inline int CG(int
Gw,int Gx){return Gw>Gx?Gw:Gx;}static float CH[256]={0.0,0.000304f,0.000607f,///
0.000911f,0.001214f,0.001518f,0.001821f,0.002125f,0.002428f,0.002732f,0.003035f,
0.003347f,0.003677f,0.004025f,0.004391f,0.004777f,0.005182f,0.005605f,0.006049f,
0.006512f,0.006995f,0.007499f,0.008023f,0.008568f,0.009134f,0.009721f,0.010330f,
0.010960f,0.011612f,0.012286f,0.012983f,0.013702f,0.014444f,0.015209f,0.016,////
0.016807f,0.017642f,0.018500f,0.019382f,0.020289f,0.021219f,0.022174f,0.023153f,
0.024158f,0.025187f,0.026241f,0.027321f,0.028426f,0.029557f,0.030713f,0.031896f,
0.033105f,0.034340f,0.035601f,0.036889f,0.038204f,0.039546f,0.040915f,0.042311f,
0.043735f,0.045186f,0.046665f,0.048172f,0.049707f,0.051269f,0.052861f,0.054480f,
0.056128f,0.057805f,0.059511f,0.061246f,0.063010f,0.064803f,0.066626f,0.068478f,
0.070360f,0.072272f,0.074214f,0.076185f,0.078187f,0.080220f,0.082283f,0.084376f,
0.086500f,0.088656f,0.090842f,0.093059f,0.095307f,0.097587f,0.099899f,0.102242f,
0.104616f,0.107023f,0.109462f,0.111932f,0.114435f,0.116971f,0.119538f,0.122139f,
0.124772f,0.127438f,0.130136f,0.132868f,0.135633f,0.138432f,0.141263f,0.144128f,
0.147027f,1.5e-01,0.152926f,0.155926f,0.158961f,0.162029f,0.165132f,0.168269f,//
0.171441f,0.174647f,0.177888f,0.181164f,0.184475f,0.187821f,0.191202f,0.194618f,
0.198069f,0.201556f,0.205079f,0.208637f,0.212231f,0.215861f,0.219526f,0.223228f,
0.226966f,0.230740f,0.234551f,0.238398f,0.242281f,0.246201f,0.250158f,0.254152f,
0.258183f,0.262251f,0.266356f,0.270498f,0.274677f,0.278894f,0.283149f,0.287441f,
0.291771f,0.296138f,0.300544f,0.304987f,0.309469f,0.313989f,0.318547f,0.323143f,
0.327778f,0.332452f,0.337164f,0.341914f,0.346704f,0.351533f,0.356400f,0.361307f,
0.366253f,0.371238f,0.376262f,0.381326f,0.386430f,0.391573f,0.396755f,0.401978f,
0.407240f,0.412543f,0.417885f,0.423268f,0.428691f,0.434154f,0.439657f,0.445201f,
0.450786f,0.456411f,0.462077f,0.467784f,0.473532f,0.479320f,0.485150f,0.491021f,
0.496933f,0.502887f,0.508881f,0.514918f,0.520996f,0.527115f,0.533276f,0.539480f,
0.545725f,0.552011f,0.558340f,0.564712f,0.571125f,0.577581f,0.584078f,0.590619f,
0.597202f,0.603827f,0.610496f,0.617207f,0.623960f,0.630757f,0.637597f,0.644480f,
0.651406f,0.658375f,0.665387f,0.672443f,0.679543f,0.686685f,0.693872f,0.701102f,
0.708376f,0.715694f,0.723055f,0.730461f,0.737911f,0.745404f,0.752942f,0.760525f,
0.768151f,0.775822f,0.783538f,0.791298f,0.799103f,0.806952f,0.814847f,0.822786f,
0.830770f,0.838799f,0.846873f,0.854993f,0.863157f,0.871367f,0.879622f,0.887923f,
0.896269f,0.904661f,0.913099f,0.921582f,0.930111f,0.938686f,0.947307f,0.955974f,
0.964686f,0.973445f,0.982251f,0.991102f,1.0f};typedef union{unsigned int A;float
B;}CI;static const Bq CJ[104]={7536653,7995405,8388621,8847373,9240589,9699341,
10092557,10551309,10944538,11796506,12648474,13500442,14286874,15138842,15990810
,16842778,17694771,19398707,21037107,22741043,24444979,26148915,27787315,///////
29491251,31195239,34537575,37945447,41287783,44695655,48037991,51445863,54788199
,58196174,64946382,71696590,78446798,85197006,91947205,98369724,0x063b00b5,/////
0x06970158,0x07420142,0x07e30130,0x087b0120,0x090b0112,0x09940106,0x0a1700fc,///
0x0a9500f2,0x0b0f01cb,0x0bf401ae,0x0ccb0195,0x0d950180,0x0e56016e,0x0f0d015e,///
0x0fbc0150,0x10630143,0x11070264,0x1238023e,0x1357021d,0x14660201,0x156601e9,///
0x165a01d3,0x174401c0,0x182401af,0x18fe0331,0x1a9602fe,0x1c1502d2,0x1d7e02ad,///
0x1ed4028d,0x201a0270,0x21520256,0x227d0240,0x239f0443,0x25c003fe,0x27bf03c4,///
0x29a10392,0x2b6a0367,0x2d1d0341,0x2ebe031f,0x304d0300,0x31d105b0,0x34a80555,///
0x37520507,0x39d504c5,0x3c37048b,0x3e7c0458,0x40a8042a,0x42bd0401,0x44c20798,///
0x488e071e,0x4c1c06b6,0x4f76065d,0x52a50610,0x55ac05cc,0x5892058f,0x5b590559,///
0x5e0c0a23,0x631c0980,0x67db08f6,0x6c55087f,0x70940818,0x74a007bd,0x787d076c,///
0x7c330723,};static inline Bo CK(float Gw){static const CI Gx={0x3f7fffff};/////
static const CI Gy={956301312};Bq Gz,G0,G1,G2;CI G3;if(!(Gw>Gy.B))return 0;if(Gw
>Gx.B)return 255;G3.B=Gw;Gz=CJ[(G3.A-Gy.A)>>20];G0=(Gz>>16)<<9;G1=Gz&0xffff;G2=(
G3.A>>12)&0xff;return(unsigned char)((G0+G1*G2)>>16);}typedef union D{unsigned//
short A;}CL;static inline float CM(CL Gw){static const CI Gx={2004877312};static
const CI Gy={1199570944};CI Gz;Gz.A=(Gw.A&0x7fff)<<13;Gz.B*=Gx.B;if(Gz.B>=Gy.B)
Gz.A|=255<<23;Gz.A|=(Gw.A&0x8000)<<16;return Gz.B;}static inline CL CN(float Gw)
{CI Gx={255<<23};CI Gy={1199570944};CI Gz={1056964608};unsigned int G0=/////////
0x80000000u;CL G1={0};CI G2;unsigned int G3;G2.B=Gw;G3=G2.A&G0;G2.A^=G3;if(G2.A
>=Gy.A)G1.A=(G2.A>Gx.A)?0x7e00:0x7c00;else if(G2.A<947912704){G2.B+=Gz.B;G1.A=(
unsigned short)(G2.A-Gz.A);}else{unsigned int G4=(G2.A>>13)&1;G2.A=G2.A+////////
3355443200u+0xfff;G2.A+=G4;G1.A=(unsigned short)(G2.A>>13);}G1.A|=G3>>16;return
G1;}static void CO(void*Gw,void const*Gx,size_t Gy){char*restrict Gz=(char*)Gx;
char*restrict G0=((char*)Gx)+Gy;ptrdiff_t G1=(char*)Gw-(char*)Gx;if(G1>=8){char*
restrict G2=((char*)Gx)+(Gy&~7);if(((((ptrdiff_t)Gw)|((ptrdiff_t)Gx))&7)==0)////
_Pragma("GCC unroll 1")_Pragma("GCC novector")do{asm(""::"r"(Gz));*(Br*)(Gz+G1)=
*(Br*)Gz;Gz+=8;}while(Gz<G2);else _Pragma("GCC unroll 1")_Pragma("GCC novector")
do{int G3,G4;asm(""::"r"(Gz));G3=((int*)Gz)[0];G4=((int*)Gz)[1];((int*)(Gz+G1))[
0]=G3;((int*)(Gz+G1))[1]=G4;Gz+=8;}while(Gz<G2);if(Gz==G0)return;}_Pragma(//////
"GCC unroll 1")_Pragma("GCC novector")do{asm(""::"r"(Gz));*(int*)(Gz+G1)=*(int*)
Gz;Gz+=4;}while(Gz<G0);}static float CP(float Gw,float Gx,void*Gy){float Gz=Gx/2
;float G0=0.5f+Gz;if(Gw<0.0f)Gw=-Gw;if(Gw>=G0)return 0.0f;else{float G1=0.5f-Gz;
if(Gw<=G1)return 1.0f;else return(G0-Gw)/Gx;}}static float CQ(float Gw,void*Gx){
return 0.5f+Gw/2.0f;}static float CR(float Gw,float Gx,void*Gy){if(Gw<0.0f)Gw= -
Gw;if(Gw<=1.0f)return 1.0f-Gw;else return 0.0f;}static float CS(float Gw,float//
Gx,void*Gy){return 1.0f;}static float CT(float Gw,float Gx,void*Gy){if(Gw<0.0f)
Gw=-Gw;if(Gw<1.0f)return(4.0f+Gw*Gw*(3.0f*Gw-6.0f))/6.0f;else if(Gw<2.0f)return(
8.0f+Gw*(-12.0+Gw*(6.0f-Gw)))/6.0f;return 0.0f;}static float CU(float Gw,float//
Gx,void*Gy){if(Gw<0.0f)Gw=-Gw;if(Gw<1.0f)return 1.0f-Gw*Gw*(2.5f-1.5f*Gw);else//
if(Gw<2.0f)return 2.0f-Gw*(4.0f+Gw*(0.5f*Gw-2.5f));return 0.0f;}static float CV(
float Gw,float Gx,void*Gy){if(Gw<0.0f)Gw=-Gw;if(Gw<1.0f)return(16.0f+Gw*Gw*(////
21.0f*Gw-36.0f))/18.0f;else if(Gw<2.0f)return(32.0f+Gw*(-60.0+Gw*(36.0f-7.0f*Gw)
))/18.0f;return 0.0f;}static float CW(float Gw,void*Gx){return 0.5f;}static/////
float CX(float Gw,void*Gx){return 1;}static float CZ(float Gw,void*Gx){return 2;
}static int Ca(Bz*Gw,float Gx,void*Gy){if(Gx>=1.0)return(int)((float)ceil((float
)(Gw(1.0f/Gx,Gy)*2.0f)));else return(int)((float)ceil((float)(Gw(Gx,Gy)*2.0f/Gx)
));}static int Cb(B8*Gw,int Gx,void*Gy){float Gz=Gw->E.C;Bz*G0=Gw->I;switch(Gx){
case 1:return(int)((float)ceil((float)(G0(1.0f/Gz,Gy)*2.0f)));case 2:return(int)
((float)ceil((float)(G0(Gz,Gy)*2.0f/Gz)));case 0:return(int)((float)ceil((float)
(G0(Gz,Gy)*2.0f)));default:return 0;}}static int Cc(B8*Gw,int Gx){if(Gx)return//
Gw->E.B;else return Gw->E.A+Gw->M*2;}static int Cd(int Gw,int Gx){return 0;}////
static int Ce(int Gw,int Gx){if(Gw<0)return 0;if(Gw>=Gx)return Gx-1;return Gw;}
static int Cf(int Gw,int Gx){if(Gw<0)if(Gw>-Gx)return-Gw;else return Gx-1;if(Gw
>=Gx){int Gy=Gx*2;if(Gw>=Gy)return 0;else return Gy-Gw-1;}return Gw;}static int
Cg(int Gw,int Gx){if(Gw>=0)return Gw%Gx;else{int Gy=(-Gw)%Gx;if(Gy!=0)Gy=Gx-Gy;
return Gy;}}typedef int Ch(int n,int max);static Ch*Ci[]={Ce,Cf,Cg,Cd,};inline//
static int Cj(Bt Gw,int Gx,int Gy){if(Gx>=0&&Gx<Gy)return Gx;return Ci[Gw](Gx,Gy
);}static void Ck(B8*Gw,B9*Gx){int Gy,Gz;int G0,G1;int G2=0x7fffffff,G3=-///////
0x7fffffff;int G4=0x7fffffff,G5=-0x7fffffff;int G6=0x7fffffff,G7=-0x7fffffff;Bt
G8=Gw->J;B4*G9=Gw->A;int G_=Gw->E.B;int HA=Gw->E.A;int HB=Gw->M;Gz=G_;for(Gy=0;
Gy<Gz;Gy++)if(G9[Gy].A<G2){G2=G9[Gy].A;Gz=Gy+HB;if(Gz>G_)Gz=G_;}Gz=0;for(Gy=G_-1
;Gy>=Gz;Gy--)if(G9[Gy].B>G3){G3=G9[Gy].B;Gz=Gy-HB;if(Gz<0)Gz=0;}G0=0;if(G2<0){G0
=-G2;G2=0;}G1=0;if(G3>=HA){G1=G3-HA+1;G3=HA-1;}Gx->B[0]=G0;Gx->B[1]=G1;Gx->C[0].
A=G2;Gx->C[0].B=G3;Gx->C[0].C=G2;Gx->C[1].A=0;Gx->C[1].B=-1;Gx->C[1].C=0;if(G8==
STBIR_EDGE_ZERO)return;for(Gy=-G0;Gy<0;Gy++){int HC=Cj(G8,Gy,HA);if(HC<G4)G4=HC;
if(HC>G5)G5=HC;}for(Gy=HA;Gy<(HA+G1);Gy++){int HC=Cj(G8,Gy,HA);if(HC<G6)G6=HC;if
(HC>G7)G7=HC;}if(G4!=0x7fffffff)if(((G4<=G2)&&((G5+16)>=G2))||((G2<=G4)&&((G3+16
)>=G5))){Gx->C[0].A=G2=CF(G2,G4);Gx->C[0].B=G3=CG(G3,G5);Gx->C[0].C=G2;G0=0;}if(
G6!=0x7fffffff)if(((G6<=G2)&&((G7+16)>=G2))||((G2<=G6)&&((G3+16)>=G7))){Gx->C[0]
.A=G2=CF(G2,G6);Gx->C[0].B=G3=CG(G3,G7);Gx->C[0].C=G2;G1=0;}if(G0&&(G4!=////////
0x7fffffff)){B6*HC=Gx->C+1;if(G4<Gx->C[0].A){Gx->C[1].C=Gx->C[0].A;Gx->C[1].A=Gx
->C[0].A;Gx->C[1].B=Gx->C[0].B;--HC;}HC->C=G4;HC->A=-G0;HC->B=(G5-G4)-G0;Gx->B[0
]=0;}else if(G1&&(G6!=0x7fffffff)){B6*HC=Gx->C+1;if(G6<Gx->C[0].A){Gx->C[1].C=Gx
->C[0].A;Gx->C[1].A=Gx->C[0].A;Gx->C[1].B=Gx->C[0].B;--HC;}HC->C=G6;HC->A=Gx->C[
1].B+1;HC->B=Gx->C[1].B+1+(G7-G6);Gx->B[1]=0;}if((Gx->C[1].B>Gx->C[1].A)&&(Gx->C
[0].A>Gx->C[1].A)){B6 HC=Gx->C[0];Gx->C[0]=Gx->C[1];Gx->C[1]=HC;}}static void Cl
(int*Gw,int*Gx,float Gy,float Gz,float G0,float G1,int G2,Bt G3){int G4,G5;float
G6=Gy-Gz;float G7=Gy+Gz;float G8=(G6+G1)*G0;float G9=(G7+G1)*G0;G4=(int)(float)
floor((float)(G8+0.5f));G5=(int)(float)floor((float)(G9-0.5f));if(G5<G4)G5=G4;if
(G3==STBIR_EDGE_WRAP){if(G4<-G2)G4=-G2;if(G5>=(G2*2))G5=(G2*2)-1;}*Gw=G4;*Gx=G5;
}static void Cm(float Gw,By*Gx,B7*Gy,int Gz,B4*G0,float*G1,int G2,Bt G3,void*G4)
{int G5,G6;float G7=Gy->D;float G8=Gy->E;int G9=Gy->A;int G_=Gy->G;int HA=Gy->F
&&(G_<Gz);G6=Gz;if(HA)G6=G_;for(G5=0;G5<G6;G5++){int HB;int HC;float HD=(float)
G5+0.5f;float HE=(HD+G8)*G7;int HF,HG;Cl(&HF,&HG,HD,Gw,G7,G8,G9,G3);if((HG-HF+1)
>G2)HG=HF+G2-1;HC=-1;for(HB=0;HB<=HG-HF;HB++){float HH=(float)(HB+HF)+0.5f;float
HI=Gx(HE-HH,G7,G4);if((HI<7.523e-37)&&(HI>-7.523e-37)){if(HB==0){++HF;HB--;/////
continue;}HI=0;}else HC=HB;G1[HB]=HI;}HG=HC+HF;G0->A=HF;G0->B=HG;++G0;G1+=G2;}}
static void Cn(B4*Gw,float*Gx,int Gy,float Gz,int G0){if(Gw->B<Gw->A){Gw->A=Gw->
B=Gy;Gx[0]=Gz;}else if(Gy<=Gw->B){if(Gy<Gw->A){if((Gw->B-Gy+1)<=G0){int G1,G2=Gw
->A-Gy;for(G1=Gw->B-Gw->A;G1>=0;G1--)Gx[G1+G2]=Gx[G1];for(G1=1;G1<G2;G1++)Gx[G1]
=0;Gx[0]=Gz;Gw->A=Gy;}}else Gx[Gy-Gw->A]+=Gz;}else if((Gy-Gw->A+1)<=G0){int G1,
G2=Gy-Gw->A;for(G1=(Gw->B-Gw->A)+1;G1<G2;G1++)Gx[G1]=0;Gx[G2]=Gz;Gw->B=Gy;}}////
static void Co(int*Gw,int*Gx,float Gy,float Gz,float G0,float G1,int G2){float//
G3=Gy-Gz;float G4=Gy+Gz;float G5=G3*G0-G1;float G6=G4*G0-G1;int G7=(int)(float)
floor((float)(G5+0.5f));int G8=(int)(float)floor((float)(G6-0.5f));if(G7<0)G7=0;
if(G8>=G2)G8=G2-1;*Gw=G7;*Gx=G8;}static void Cp(int Gw,int Gx,float Gy,By*Gz,B7*
G0,int G1,int G2,B4*G3,float*G4,void*G5){int G6;int G7;int G8=-1;float G9=G0->C;
float G_=G0->E;int HA=G0->B;int HB=G0->G;int HC=G0->F&&(HB<HA);for(G6=Gw;G6<Gx;
G6++){float HD=(float)G6+0.5f;float HE=HD*G9-G_;int HF,HG;Co(&HF,&HG,HD,Gy,G9,G_
,HA);if(HF>HG)continue;if(HC){if(HF==HB)break;if(HG>=HB)HG=HB-1;}for(G7=0;G7<=HG
-HF;G7++){float HH=(float)(G7+HF)+0.5f;float HI=HH-HE;float HJ=Gz(HI,G9,G5)*G9;
if((HJ<7.523e-37)&&(HJ>-7.523e-37))HJ=0.0f;{int HK=G7+HF;float*HL=G4+HK*G1;B4*HM
=G3+HK;if(HK>G8){G8=HK;HM->A=G6;HM->B=G6;HL[0]=HJ;}else{if(HL[0]==0.0f)HM->A=G6;
HM->B=G6;HL[G6-HM->A]=HJ;}}}}}static void Cq(Bt Gw,B5*Gx,B7*Gy,int Gz,B4*G0,////
float*G1,int G2){int G3=Gy->A;int G4=G3-1;int G5,G6;int G7=0x7fffffff;int G8=-//
0x7fffffff;int G9=-1;int G_=Gy->G;int HA=Gy->H;int HB=Gy->F&&(G_<Gz);float*HC;B4
*HD;HC=G1;HD=G0;G6=Gz;if(HB)G6=G_;for(G5=0;G5<G6;G5++){int HE;double HF,HG=0;int
HH;HH=HD->B-HD->A;for(HE=0;HE<=HH;HE++)HG+=(double)HC[HE];if((HG<7.523e-37)&&(HG
>-7.523e-37)){HD->B=HD->A;HC[0]=0.0f;}else if((HG<1.0)||(HG>1.0)){HF=1.0/HG;for(
HE=0;HE<=HH;HE++)HC[HE]=(float)(HC[HE]*HF);}++HD;HC+=G2;}if(HB){B4*HE=G0;B4*HF=
G0+G_;for(G5=G_;G5<Gz;G5++){HF->A=HE->A+HA;HF->B=HE->B+HA;++HF;++HE;}CO(G1+G_*G2
,G1,(Gz-G_)*G2*4);}HC=G1;HD=G0;for(G5=0;G5<Gz;G5++){int HE;if(Gw==//////////////
STBIR_EDGE_ZERO){if(HD->B>G4)HD->B=G4;if(HD->A<0){int HF,HG,HH=0;HH=-HD->A;HD->A
=0;HG=HD->B-HD->A+1;if(HG>0)for(HF=0;HF<HG;HF++)HC[HF]=HC[HF+HH];}}else if((Gw==
STBIR_EDGE_CLAMP)||(Gw==STBIR_EDGE_REFLECT)){if(HD->B>G4){int HF=HD->A;int HG=HD
->B;HD->B=G4;for(HE=G3;HE<=HG;HE++)Cn(HD,HC,Ci[Gw](HE,G3),HC[HE-HF],G2);}if(HD->
A<0){int HF;float HG;float*HH=HC-(HD->A+1);for(HE=-1;HE>HD->A;HE--)Cn(HD,HC,Ci[
Gw](HE,G3),*HH--,G2);HF=HD->A;HG=HH[0];HD->A=0;for(HE=0;HE<=HD->B;HE++)HC[HE]=HC
[HE-HF];Cn(HD,HC,Ci[Gw](HF,G3),HG,G2);}}if(HD->A<=HD->B){int HF=HD->B-HD->A+1;//
while(HF&&(HC[HF-1]==0.0f))--HF;HD->B=HD->A+HF-1;if(HD->A<=HD->B){if(HD->A<G7)G7
=HD->A;if(HD->B>G8)G8=HD->B;if(HF>G9)G9=HF;}for(HE=HF;HE<G2;HE++)HC[HE]=0.0f;}++
HD;HC+=G2;}Gx->A=G7;Gx->B=G8;Gx->C=G9;}static int Cr(int Gw,B4*Gx,float*Gy,int//
Gz,int G0,int G1,int G2){int G3=G2+1;if(Gz!=G0){float*G4=Gy;float*G5=Gy;float*G6
=Gy+Gw*G0;switch(G0){case 1:_Pragma("GCC unroll 1")_Pragma("GCC novector")do{{//
asm(""::"r"(G4));((Bq*)G4)[0]=((Bq*)G5)[0];}++G4;G5+=Gz;}while(G4<G6);break;case
2:_Pragma("GCC unroll 1")_Pragma("GCC novector")do{{asm(""::"r"(G4));((Br*)G4)[0
]=((Br*)G5)[0];}G4+=2;G5+=Gz;}while(G4<G6);break;case 3:_Pragma("GCC unroll 1")
_Pragma("GCC novector")do{{asm(""::"r"(G4));((Br*)G4)[0]=((Br*)G5)[0];}{asm(""::
"r"(G4+2));((Bq*)(G4+2))[0]=((Bq*)(G5+2))[0];}G4+=3;G5+=Gz;}while(G4<G6);break;
case 4:_Pragma("GCC unroll 1")_Pragma("GCC novector")do{{asm(""::"r"(G4));((Br*)
G4)[0]=((Br*)G5)[0];((Br*)G4)[1]=((Br*)G5)[1];}G4+=4;G5+=Gz;}while(G4<G6);break;
case 5:_Pragma("GCC unroll 1")_Pragma("GCC novector")do{{asm(""::"r"(G4));((Br*)
G4)[0]=((Br*)G5)[0];((Br*)G4)[1]=((Br*)G5)[1];}{asm(""::"r"(G4+4));((Bq*)(G4+4))
[0]=((Bq*)(G5+4))[0];}G4+=5;G5+=Gz;}while(G4<G6);break;case 6:_Pragma(//////////
"GCC unroll 1")_Pragma("GCC novector")do{{asm(""::"r"(G4));((Br*)G4)[0]=((Br*)G5
)[0];((Br*)G4)[1]=((Br*)G5)[1];}{asm(""::"r"(G4+4));((Br*)(G4+4))[0]=((Br*)(G5+4
))[0];}G4+=6;G5+=Gz;}while(G4<G6);break;case 7:_Pragma("GCC unroll 1")_Pragma(//
"GCC novector")do{{asm(""::"r"(G4));((Br*)G4)[0]=((Br*)G5)[0];((Br*)G4)[1]=((Br*
)G5)[1];}{asm(""::"r"(G4+4));((Br*)(G4+4))[0]=((Br*)(G5+4))[0];}{asm(""::"r"(G4+
6));((Bq*)(G4+6))[0]=((Bq*)(G5+6))[0];}G4+=7;G5+=Gz;}while(G4<G6);break;case 8:
_Pragma("GCC unroll 1")_Pragma("GCC novector")do{{asm(""::"r"(G4));((Br*)G4)[0]=
((Br*)G5)[0];((Br*)G4)[1]=((Br*)G5)[1];}{asm(""::"r"(G4+4));((Br*)(G4+4))[0]=((
Br*)(G5+4))[0];((Br*)(G4+4))[1]=((Br*)(G5+4))[1];}G4+=8;G5+=Gz;}while(G4<G6);///
break;case 9:_Pragma("GCC unroll 1")_Pragma("GCC novector")do{{asm(""::"r"(G4));
((Br*)G4)[0]=((Br*)G5)[0];((Br*)G4)[1]=((Br*)G5)[1];}{asm(""::"r"(G4+4));((Br*)(
G4+4))[0]=((Br*)(G5+4))[0];((Br*)(G4+4))[1]=((Br*)(G5+4))[1];}{asm(""::"r"(G4+8)
);((Bq*)(G4+8))[0]=((Bq*)(G5+8))[0];}G4+=9;G5+=Gz;}while(G4<G6);break;case 10://
_Pragma("GCC unroll 1")_Pragma("GCC novector")do{{asm(""::"r"(G4));((Br*)G4)[0]=
((Br*)G5)[0];((Br*)G4)[1]=((Br*)G5)[1];}{asm(""::"r"(G4+4));((Br*)(G4+4))[0]=((
Br*)(G5+4))[0];((Br*)(G4+4))[1]=((Br*)(G5+4))[1];}{asm(""::"r"(G4+8));((Br*)(G4+
8))[0]=((Br*)(G5+8))[0];}G4+=10;G5+=Gz;}while(G4<G6);break;case 11:_Pragma(/////
"GCC unroll 1")_Pragma("GCC novector")do{{asm(""::"r"(G4));((Br*)G4)[0]=((Br*)G5
)[0];((Br*)G4)[1]=((Br*)G5)[1];}{asm(""::"r"(G4+4));((Br*)(G4+4))[0]=((Br*)(G5+4
))[0];((Br*)(G4+4))[1]=((Br*)(G5+4))[1];}{asm(""::"r"(G4+8));((Br*)(G4+8))[0]=((
Br*)(G5+8))[0];}{asm(""::"r"(G4+10));((Bq*)(G4+10))[0]=((Bq*)(G5+10))[0];}G4+=11
;G5+=Gz;}while(G4<G6);break;case 12:_Pragma("GCC unroll 1")_Pragma(/////////////
"GCC novector")do{{asm(""::"r"(G4));((Br*)G4)[0]=((Br*)G5)[0];((Br*)G4)[1]=((Br*
)G5)[1];}{asm(""::"r"(G4+4));((Br*)(G4+4))[0]=((Br*)(G5+4))[0];((Br*)(G4+4))[1]=
((Br*)(G5+4))[1];}{asm(""::"r"(G4+8));((Br*)(G4+8))[0]=((Br*)(G5+8))[0];((Br*)(
G4+8))[1]=((Br*)(G5+8))[1];}G4+=12;G5+=Gz;}while(G4<G6);break;default:_Pragma(//
"GCC unroll 1")_Pragma("GCC novector")do{float*G7=G4+G0-4;float*G8=G5;do{asm(""
::"r"(G4));{asm(""::"r"(G4));((Br*)G4)[0]=((Br*)G8)[0];((Br*)G4)[1]=((Br*)G8)[1]
;}G4+=4;G8+=4;}while(G4<=G7);G7+=4;_Pragma("GCC unroll 1")_Pragma("GCC novector"
)while(G4<G7){{asm(""::"r"(G4));((Bq*)G4)[0]=((Bq*)G8)[0];}++G4;++G8;}G5+=Gz;}//
while(G4<G6);break;}}Gy[G0*Gw]=8888.0f;{B4*G4=Gx+Gw-1;float*G5=Gy+G0*(Gw-1);////
while((G4>=Gx)&&((G4->A+G0*2)>=G3)){if((G4->A+G0)>G3){int G6=G0;if(G0>12){int G7
;G7=G0&3;G6=(((G4->B-G4->A+1)-G7+3)&~3)+G7;if(G6<(8+G7))G6=8+G7;}if((G4->A+G6)>
G3){int G7=G3-G6;int G8=G4->B-G4->A+1;int G9=G4->A-G7;float*G_=G5+G8-1;float*HA=
G_+G9;while(G8){*HA-- =*G_--;--G8;}while(HA>=G5)*HA-- =0;G4->A=G7;if(G0>12){int
HB;HB=G0&3;G6=(((G4->B-G4->A+1)-HB+3)&~3)+HB;if(G6<(8+HB))G6=8+HB;}}}--G4;G5-=G0
;}}return G0;}static void Cs(B8*Gw,B8*Gx,void*Gy){int Gz;float G0=Gw->E.C;By*G1=
Gw->H;Bz*G2=Gw->I;float G3=Gw->E.D;int G4=Gw->E.A;int G5=Gw->N;B4*G6=Gw->A;float
*G7=Gw->B;int G8=Gw->K;switch(Gw->R){case 1:{float G9=G2(G3,Gy)*G0;Cm(G9,G1,&Gw
->E,G5,G6,G7,G8,Gw->J,Gy);Cq(Gw->J,&Gw->Q,&Gw->E,G5,G6,G7,G8);}break;case 0:case
2:{float G9=G2(G0,Gy)*G3;int G_=Gw->M;int HA=G4+G_;if(!Gw->R){if(Gx){G6=Gx->A;G7
=Gx->B;G8=Gx->K;G5=Gx->N;Gw->Q.A=Gx->Q.A;Gw->Q.B=Gx->Q.B;Gw->Q.C=Gx->Q.C;goto///
jump_right_to_pivot;}G6=Gw->C;G7=Gw->D;G8=Gw->T;G5=Gw->S;}Cp(-G_,HA,G9,G1,&Gw->E
,G8,G5,G6,G7,Gy);Cq(Gw->J,&Gw->Q,&Gw->E,G5,G6,G7,G8);if(!Gw->R){B4*HB;int HC;///
jump_right_to_pivot:HC=(-G_)-1;for(Gz=0;Gz<G5;Gz++){int HD;int HE=G6->A,HF=G6->B
;int HG=Gw->K;float*HH=Gw->B+(HE+G_)*HG;float*HI=G7;HB=Gw->A+(HE+G_);for(HD=HE;
HD<=HF;HD++){float HJ=*HI++;if((HJ>=7.523e-37)||(HJ<=-7.523e-37))if((HD>HC)||(HB
->A>HB->B)){{B4*HK=Gw->A+(HC+G_+1);while(HK<HB){HK->A=0;HK->B=-1;++HK;}}HB->A=Gz
;HB->B=Gz;HH[0]=HJ;HC=HD;}else Cn(HB,HH,Gz,HJ,HG);++HB;HH+=HG;}++G6;G7+=G8;}{B4*
HD=Gw->A+(HC+G_+1);B4*HE=Gw->A+Gw->N;while(HD<HE){HD->A=0;HD->B=-1;++HD;}}}}////
break;}}static float*Ct(float*Gw,int Gx,void const*Gy){float*restrict Gz=Gw;////
float*G0=(float*)Gz+Gx;unsigned char const*G1=(unsigned char const*)Gy;Gz+=4;///
while(Gz<=G0){Gz[-4]=((float)G1[0])*3.922e-03;Gz[-3]=((float)G1[1])*3.922e-03;Gz
[-2]=((float)G1[2])*3.922e-03;Gz[-1]=((float)G1[3])*3.922e-03;Gz+=4;G1+=4;}Gz-=4
;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(Gz<G0){asm(""::"r"(Gz));Gz[
0]=((float)G1[0])*3.922e-03;Gz+=1;G1+=1;}return G0;}static void Cu(void*Gw,int//
Gx,float const*Gy){unsigned char*restrict Gz=(unsigned char*)Gw;unsigned char*G0
=((unsigned char*)Gz)+Gx;Gz+=4;while(Gz<=G0){float G1;G1=Gy[0]*255.0f+0.5f;for(;
;){if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[-4]=(unsigned char)G1;G1=Gy[1]*255.0f
+0.5f;for(;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[-3]=(unsigned char)G1;G1=
Gy[2]*255.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[-2]=(unsigned
char)G1;G1=Gy[3]*255.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[-1]
=(unsigned char)G1;Gz+=4;Gy+=4;}Gz-=4;_Pragma("GCC unroll 1")_Pragma(///////////
"GCC novector")while(Gz<G0){float G1;asm(""::"r"(Gy));G1=Gy[0]*255.0f+0.5f;for(;
;){if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[0]=(unsigned char)G1;Gz+=1;Gy+=1;}}//
static float*Cv(float*Gw,int Gx,void const*Gy){float*restrict Gz=Gw;float*G0=(//
float*)Gz+Gx;unsigned char const*G1=(unsigned char const*)Gy;Gz+=4;while(Gz<=G0)
{Gz[-4]=((float)G1[0]);Gz[-3]=((float)G1[1]);Gz[-2]=((float)G1[2]);Gz[-1]=((////
float)G1[3]);Gz+=4;G1+=4;}Gz-=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")//
while(Gz<G0){asm(""::"r"(Gz));Gz[0]=((float)G1[0]);Gz+=1;G1+=1;}return G0;}/////
static void Cw(void*Gw,int Gx,float const*Gy){unsigned char*restrict Gz=(///////
unsigned char*)Gw;unsigned char*G0=((unsigned char*)Gz)+Gx;Gz+=4;while(Gz<=G0){
float G1;G1=Gy[0]+0.5f;for(;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[-4]=(////
unsigned char)G1;G1=Gy[1]+0.5f;for(;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[-
3]=(unsigned char)G1;G1=Gy[2]+0.5f;for(;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}
Gz[-2]=(unsigned char)G1;G1=Gy[3]+0.5f;for(;;){if(G1<0)G1=0;if(G1>255)G1=255;///
break;}Gz[-1]=(unsigned char)G1;Gz+=4;Gy+=4;}Gz-=4;_Pragma("GCC unroll 1")//////
_Pragma("GCC novector")while(Gz<G0){float G1;asm(""::"r"(Gy));G1=Gy[0]+0.5f;for(
;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[0]=(unsigned char)G1;Gz+=1;Gy+=1;}}
static float*Cx(float*Gw,int Gx,void const*Gy){float*restrict Gz=Gw;float*G0=(//
float*)Gz+Gx;unsigned char const*G1=(unsigned char const*)Gy;Gz+=4;while(Gz<=G0)
{Gz[-4]=CH[G1[0]];Gz[-3]=CH[G1[1]];Gz[-2]=CH[G1[2]];Gz[-1]=CH[G1[3]];Gz+=4;G1+=4
;}Gz-=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(Gz<G0){asm(""::"r"(
Gz));Gz[0]=CH[G1[0]];Gz+=1;G1+=1;}return G0;}static void Cy(void*Gw,int Gx,float
const*Gy){unsigned char*restrict Gz=(unsigned char*)Gw;unsigned char*G0=((//////
unsigned char*)Gz)+Gx;Gz+=4;while(Gz<=G0){Gz[-4]=CK(Gy[0]);Gz[-3]=CK(Gy[1]);Gz[-
2]=CK(Gy[2]);Gz[-1]=CK(Gy[3]);Gz+=4;Gy+=4;}Gz-=4;_Pragma("GCC unroll 1")_Pragma(
"GCC novector")while(Gz<G0){asm(""::"r"(Gy));Gz[0]=CK(Gy[0]);Gz+=1;Gy+=1;}}/////
static float*Cz(float*Gw,int Gx,void const*Gy){float*restrict Gz=Gw;float*G0=(//
float*)Gz+Gx;unsigned char const*G1=(unsigned char const*)Gy;do{Gz[0]=CH[G1[0]];
Gz[1]=CH[G1[1]];Gz[2]=CH[G1[2]];Gz[3]=((float)G1[3])*3.922e-03;G1+=4;Gz+=4;}////
while(Gz<G0);return G0;}static void C0(void*Gw,int Gx,float const*Gy){unsigned//
char*restrict Gz=(unsigned char*)Gw;unsigned char*G0=((unsigned char*)Gz)+Gx;do{
float G1;Gz[0]=CK(Gy[0]);Gz[1]=CK(Gy[1]);Gz[2]=CK(Gy[2]);G1=Gy[3]*255.0f+0.5f;//
for(;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[3]=(unsigned char)G1;Gz+=4;Gy+=4
;}while(Gz<G0);}static float*C1(float*Gw,int Gx,void const*Gy){float*restrict Gz
=Gw;float*G0=(float*)Gz+Gx;unsigned char const*G1=(unsigned char const*)Gy;Gz+=4
;while(Gz<=G0){Gz[-4]=CH[G1[0]];Gz[-3]=((float)G1[1])*3.922e-03;Gz[-2]=CH[G1[2]]
;Gz[-1]=((float)G1[3])*3.922e-03;G1+=4;Gz+=4;}Gz-=4;if(Gz<G0){Gz[0]=CH[G1[0]];Gz
[1]=((float)G1[1])*3.922e-03;}return G0;}static void C2(void*Gw,int Gx,float////
const*Gy){unsigned char*restrict Gz=(unsigned char*)Gw;unsigned char*G0=((//////
unsigned char*)Gz)+Gx;do{float G1;Gz[0]=CK(Gy[0]);G1=Gy[1]*255.0f+0.5f;for(;;){
if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[1]=(unsigned char)G1;Gz+=2;Gy+=2;}while(
Gz<G0);}static float*C3(float*Gw,int Gx,void const*Gy){float*restrict Gz=Gw;////
float*G0=(float*)Gz+Gx;unsigned short const*G1=(unsigned short const*)Gy;Gz+=4;
while(Gz<=G0){Gz[-4]=((float)G1[0])*1.526e-05;Gz[-3]=((float)G1[1])*1.526e-05;Gz
[-2]=((float)G1[2])*1.526e-05;Gz[-1]=((float)G1[3])*1.526e-05;Gz+=4;G1+=4;}Gz-=4
;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(Gz<G0){asm(""::"r"(Gz));Gz[
0]=((float)G1[0])*1.526e-05;Gz+=1;G1+=1;}return G0;}static void C4(void*Gw,int//
Gx,float const*Gy){unsigned short*restrict Gz=(unsigned short*)Gw;unsigned short
*G0=((unsigned short*)Gz)+Gx;Gz+=4;while(Gz<=G0){float G1;G1=Gy[0]*65535.0f+0.5f
;for(;;){if(G1<0)G1=0;if(G1>65535)G1=65535;break;}Gz[-4]=(unsigned short)G1;G1=
Gy[1]*65535.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>65535)G1=65535;break;}Gz[-3]=(///
unsigned short)G1;G1=Gy[2]*65535.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>65535)G1=///
65535;break;}Gz[-2]=(unsigned short)G1;G1=Gy[3]*65535.0f+0.5f;for(;;){if(G1<0)G1
=0;if(G1>65535)G1=65535;break;}Gz[-1]=(unsigned short)G1;Gz+=4;Gy+=4;}Gz-=4;////
_Pragma("GCC unroll 1")_Pragma("GCC novector")while(Gz<G0){float G1;asm(""::"r"(
Gy));G1=Gy[0]*65535.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>65535)G1=65535;break;}Gz[
0]=(unsigned short)G1;Gz+=1;Gy+=1;}}static float*C5(float*Gw,int Gx,void const*
Gy){float*restrict Gz=Gw;float*G0=(float*)Gz+Gx;unsigned short const*G1=(///////
unsigned short const*)Gy;Gz+=4;while(Gz<=G0){Gz[-4]=((float)G1[0]);Gz[-3]=((////
float)G1[1]);Gz[-2]=((float)G1[2]);Gz[-1]=((float)G1[3]);Gz+=4;G1+=4;}Gz-=4;////
_Pragma("GCC unroll 1")_Pragma("GCC novector")while(Gz<G0){asm(""::"r"(Gz));Gz[0
]=((float)G1[0]);Gz+=1;G1+=1;}return G0;}static void C6(void*Gw,int Gx,float////
const*Gy){unsigned short*restrict Gz=(unsigned short*)Gw;unsigned short*G0=((///
unsigned short*)Gz)+Gx;Gz+=4;while(Gz<=G0){float G1;G1=Gy[0]+0.5f;for(;;){if(G1<
0)G1=0;if(G1>65535)G1=65535;break;}Gz[-4]=(unsigned short)G1;G1=Gy[1]+0.5f;for(;
;){if(G1<0)G1=0;if(G1>65535)G1=65535;break;}Gz[-3]=(unsigned short)G1;G1=Gy[2]+
0.5f;for(;;){if(G1<0)G1=0;if(G1>65535)G1=65535;break;}Gz[-2]=(unsigned short)G1;
G1=Gy[3]+0.5f;for(;;){if(G1<0)G1=0;if(G1>65535)G1=65535;break;}Gz[-1]=(unsigned
short)G1;Gz+=4;Gy+=4;}Gz-=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(
Gz<G0){float G1;asm(""::"r"(Gy));G1=Gy[0]+0.5f;for(;;){if(G1<0)G1=0;if(G1>65535)
G1=65535;break;}Gz[0]=(unsigned short)G1;Gz+=1;Gy+=1;}}static float*C7(float*Gw,
int Gx,void const*Gy){float*restrict Gz=Gw;float*G0=(float*)Gz+Gx;CL const*G1=(
CL const*)Gy;Gz+=4;while(Gz<=G0){Gz[-4]=CM(G1[0]);Gz[-3]=CM(G1[1]);Gz[-2]=CM(G1[
2]);Gz[-1]=CM(G1[3]);Gz+=4;G1+=4;}Gz-=4;_Pragma("GCC unroll 1")_Pragma(/////////
"GCC novector")while(Gz<G0){asm(""::"r"(Gz));Gz[0]=CM(G1[0]);Gz+=1;G1+=1;}return
G0;}static void C8(void*Gw,int Gx,float const*Gy){CL*restrict Gz=(CL*)Gw;CL*G0=(
(CL*)Gz)+Gx;Gz+=4;while(Gz<=G0){Gz[-4]=CN(Gy[0]);Gz[-3]=CN(Gy[1]);Gz[-2]=CN(Gy[2
]);Gz[-1]=CN(Gy[3]);Gz+=4;Gy+=4;}Gz-=4;_Pragma("GCC unroll 1")_Pragma(//////////
"GCC novector")while(Gz<G0){asm(""::"r"(Gz));Gz[0]=CN(Gy[0]);Gz+=1;Gy+=1;}}/////
static float*C9(float*Gw,int Gx,void const*Gy){if((void*)Gw!=Gy)memcpy(Gw,Gy,Gx*
4);return Gw+Gx;}static void C_(void*Gw,int Gx,float const*Gy){if((void*)Gw!=(//
void*)Gy)memcpy(Gw,Gy,Gx*4);}static float*DA(float*Gw,int Gx,void const*Gy){////
float*restrict Gz=Gw;float*G0=(float*)Gz+Gx;unsigned char const*G1=(unsigned////
char const*)Gy;Gz+=4;while(Gz<=G0){Gz[-4]=((float)G1[2])*3.922e-03;Gz[-3]=((////
float)G1[1])*3.922e-03;Gz[-2]=((float)G1[0])*3.922e-03;Gz[-1]=((float)G1[3])*///
3.922e-03;Gz+=4;G1+=4;}Gz-=4;return G0;}static void DB(void*Gw,int Gx,float/////
const*Gy){unsigned char*restrict Gz=(unsigned char*)Gw;unsigned char*G0=((//////
unsigned char*)Gz)+Gx;Gz+=4;while(Gz<=G0){float G1;G1=Gy[2]*255.0f+0.5f;for(;;){
if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[-4]=(unsigned char)G1;G1=Gy[1]*255.0f+//
0.5f;for(;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[-3]=(unsigned char)G1;G1=Gy
[0]*255.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[-2]=(unsigned///
char)G1;G1=Gy[3]*255.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[-1]
=(unsigned char)G1;Gz+=4;Gy+=4;}Gz-=4;}static float*DC(float*Gw,int Gx,void/////
const*Gy){float*restrict Gz=Gw;float*G0=(float*)Gz+Gx;unsigned char const*G1=(//
unsigned char const*)Gy;Gz+=4;while(Gz<=G0){Gz[-4]=((float)G1[2]);Gz[-3]=((float
)G1[1]);Gz[-2]=((float)G1[0]);Gz[-1]=((float)G1[3]);Gz+=4;G1+=4;}Gz-=4;return G0
;}static void DD(void*Gw,int Gx,float const*Gy){unsigned char*restrict Gz=(/////
unsigned char*)Gw;unsigned char*G0=((unsigned char*)Gz)+Gx;Gz+=4;while(Gz<=G0){
float G1;G1=Gy[2]+0.5f;for(;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[-4]=(////
unsigned char)G1;G1=Gy[1]+0.5f;for(;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[-
3]=(unsigned char)G1;G1=Gy[0]+0.5f;for(;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}
Gz[-2]=(unsigned char)G1;G1=Gy[3]+0.5f;for(;;){if(G1<0)G1=0;if(G1>255)G1=255;///
break;}Gz[-1]=(unsigned char)G1;Gz+=4;Gy+=4;}Gz-=4;}static float*DE(float*Gw,int
Gx,void const*Gy){float*restrict Gz=Gw;float*G0=(float*)Gz+Gx;unsigned char/////
const*G1=(unsigned char const*)Gy;Gz+=4;while(Gz<=G0){Gz[-4]=CH[G1[2]];Gz[-3]=CH
[G1[1]];Gz[-2]=CH[G1[0]];Gz[-1]=CH[G1[3]];Gz+=4;G1+=4;}Gz-=4;return G0;}static//
void DF(void*Gw,int Gx,float const*Gy){unsigned char*restrict Gz=(unsigned char*
)Gw;unsigned char*G0=((unsigned char*)Gz)+Gx;Gz+=4;while(Gz<=G0){Gz[-4]=CK(Gy[2]
);Gz[-3]=CK(Gy[1]);Gz[-2]=CK(Gy[0]);Gz[-1]=CK(Gy[3]);Gz+=4;Gy+=4;}Gz-=4;}static
float*DG(float*Gw,int Gx,void const*Gy){float*restrict Gz=Gw;float*G0=(float*)Gz
+Gx;unsigned char const*G1=(unsigned char const*)Gy;do{Gz[0]=CH[G1[2]];Gz[1]=CH[
G1[1]];Gz[2]=CH[G1[0]];Gz[3]=((float)G1[3])*3.922e-03;G1+=4;Gz+=4;}while(Gz<G0);
return G0;}static void DH(void*Gw,int Gx,float const*Gy){unsigned char*restrict
Gz=(unsigned char*)Gw;unsigned char*G0=((unsigned char*)Gz)+Gx;do{float G1;Gz[2]
=CK(Gy[0]);Gz[1]=CK(Gy[1]);Gz[0]=CK(Gy[2]);G1=Gy[3]*255.0f+0.5f;for(;;){if(G1<0)
G1=0;if(G1>255)G1=255;break;}Gz[3]=(unsigned char)G1;Gz+=4;Gy+=4;}while(Gz<G0);}
static float*DI(float*Gw,int Gx,void const*Gy){float*restrict Gz=Gw;float*G0=(//
float*)Gz+Gx;unsigned short const*G1=(unsigned short const*)Gy;Gz+=4;while(Gz<=
G0){Gz[-4]=((float)G1[2])*1.526e-05;Gz[-3]=((float)G1[1])*1.526e-05;Gz[-2]=((///
float)G1[0])*1.526e-05;Gz[-1]=((float)G1[3])*1.526e-05;Gz+=4;G1+=4;}Gz-=4;return
G0;}static void DJ(void*Gw,int Gx,float const*Gy){unsigned short*restrict Gz=(//
unsigned short*)Gw;unsigned short*G0=((unsigned short*)Gz)+Gx;Gz+=4;while(Gz<=G0
){float G1;G1=Gy[2]*65535.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>65535)G1=65535;////
break;}Gz[-4]=(unsigned short)G1;G1=Gy[1]*65535.0f+0.5f;for(;;){if(G1<0)G1=0;if(
G1>65535)G1=65535;break;}Gz[-3]=(unsigned short)G1;G1=Gy[0]*65535.0f+0.5f;for(;;
){if(G1<0)G1=0;if(G1>65535)G1=65535;break;}Gz[-2]=(unsigned short)G1;G1=Gy[3]*//
65535.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>65535)G1=65535;break;}Gz[-1]=(unsigned
short)G1;Gz+=4;Gy+=4;}Gz-=4;}static float*DK(float*Gw,int Gx,void const*Gy){////
float*restrict Gz=Gw;float*G0=(float*)Gz+Gx;unsigned short const*G1=(unsigned///
short const*)Gy;Gz+=4;while(Gz<=G0){Gz[-4]=((float)G1[2]);Gz[-3]=((float)G1[1]);
Gz[-2]=((float)G1[0]);Gz[-1]=((float)G1[3]);Gz+=4;G1+=4;}Gz-=4;return G0;}static
void DL(void*Gw,int Gx,float const*Gy){unsigned short*restrict Gz=(unsigned/////
short*)Gw;unsigned short*G0=((unsigned short*)Gz)+Gx;Gz+=4;while(Gz<=G0){float//
G1;G1=Gy[2]+0.5f;for(;;){if(G1<0)G1=0;if(G1>65535)G1=65535;break;}Gz[-4]=(//////
unsigned short)G1;G1=Gy[1]+0.5f;for(;;){if(G1<0)G1=0;if(G1>65535)G1=65535;break;
}Gz[-3]=(unsigned short)G1;G1=Gy[0]+0.5f;for(;;){if(G1<0)G1=0;if(G1>65535)G1=///
65535;break;}Gz[-2]=(unsigned short)G1;G1=Gy[3]+0.5f;for(;;){if(G1<0)G1=0;if(G1>
65535)G1=65535;break;}Gz[-1]=(unsigned short)G1;Gz+=4;Gy+=4;}Gz-=4;}static float
*DM(float*Gw,int Gx,void const*Gy){float*restrict Gz=Gw;float*G0=(float*)Gz+Gx;
CL const*G1=(CL const*)Gy;Gz+=4;while(Gz<=G0){Gz[-4]=CM(G1[2]);Gz[-3]=CM(G1[1]);
Gz[-2]=CM(G1[0]);Gz[-1]=CM(G1[3]);Gz+=4;G1+=4;}Gz-=4;return G0;}static void DN(
void*Gw,int Gx,float const*Gy){CL*restrict Gz=(CL*)Gw;CL*G0=((CL*)Gz)+Gx;Gz+=4;
while(Gz<=G0){Gz[-4]=CN(Gy[2]);Gz[-3]=CN(Gy[1]);Gz[-2]=CN(Gy[0]);Gz[-1]=CN(Gy[3]
);Gz+=4;Gy+=4;}Gz-=4;}static float*DO(float*Gw,int Gx,void const*Gy){float*/////
restrict Gz=Gw;float*G0=(float*)Gz+Gx;float const*G1=(float const*)Gy;Gz+=4;////
while(Gz<=G0){Gz[-4]=G1[2];Gz[-3]=G1[1];Gz[-2]=G1[0];Gz[-1]=G1[3];Gz+=4;G1+=4;}
Gz-=4;return G0;}static void DP(void*Gw,int Gx,float const*Gy){float*restrict Gz
=(float*)Gw;float*G0=((float*)Gz)+Gx;Gz+=4;while(Gz<=G0){float G1;G1=Gy[2];Gz[-4
]=G1;G1=Gy[1];Gz[-3]=G1;G1=Gy[0];Gz[-2]=G1;G1=Gy[3];Gz[-1]=G1;Gz+=4;Gy+=4;}Gz-=4
;}static float*DQ(float*Gw,int Gx,void const*Gy){float*restrict Gz=Gw;float*G0=(
float*)Gz+Gx;unsigned char const*G1=(unsigned char const*)Gy;Gz+=4;while(Gz<=G0)
{Gz[-4]=((float)G1[1])*3.922e-03;Gz[-3]=((float)G1[2])*3.922e-03;Gz[-2]=((float)
G1[3])*3.922e-03;Gz[-1]=((float)G1[0])*3.922e-03;Gz+=4;G1+=4;}Gz-=4;return G0;}
static void DR(void*Gw,int Gx,float const*Gy){unsigned char*restrict Gz=(///////
unsigned char*)Gw;unsigned char*G0=((unsigned char*)Gz)+Gx;Gz+=4;while(Gz<=G0){
float G1;G1=Gy[3]*255.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[-4
]=(unsigned char)G1;G1=Gy[0]*255.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>255)G1=255;
break;}Gz[-3]=(unsigned char)G1;G1=Gy[1]*255.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>
255)G1=255;break;}Gz[-2]=(unsigned char)G1;G1=Gy[2]*255.0f+0.5f;for(;;){if(G1<0)
G1=0;if(G1>255)G1=255;break;}Gz[-1]=(unsigned char)G1;Gz+=4;Gy+=4;}Gz-=4;}static
float*DS(float*Gw,int Gx,void const*Gy){float*restrict Gz=Gw;float*G0=(float*)Gz
+Gx;unsigned char const*G1=(unsigned char const*)Gy;Gz+=4;while(Gz<=G0){Gz[-4]=(
(float)G1[1]);Gz[-3]=((float)G1[2]);Gz[-2]=((float)G1[3]);Gz[-1]=((float)G1[0]);
Gz+=4;G1+=4;}Gz-=4;return G0;}static void DT(void*Gw,int Gx,float const*Gy){////
unsigned char*restrict Gz=(unsigned char*)Gw;unsigned char*G0=((unsigned char*)
Gz)+Gx;Gz+=4;while(Gz<=G0){float G1;G1=Gy[3]+0.5f;for(;;){if(G1<0)G1=0;if(G1>255
)G1=255;break;}Gz[-4]=(unsigned char)G1;G1=Gy[0]+0.5f;for(;;){if(G1<0)G1=0;if(G1
>255)G1=255;break;}Gz[-3]=(unsigned char)G1;G1=Gy[1]+0.5f;for(;;){if(G1<0)G1=0;
if(G1>255)G1=255;break;}Gz[-2]=(unsigned char)G1;G1=Gy[2]+0.5f;for(;;){if(G1<0)
G1=0;if(G1>255)G1=255;break;}Gz[-1]=(unsigned char)G1;Gz+=4;Gy+=4;}Gz-=4;}static
float*DU(float*Gw,int Gx,void const*Gy){float*restrict Gz=Gw;float*G0=(float*)Gz
+Gx;unsigned char const*G1=(unsigned char const*)Gy;Gz+=4;while(Gz<=G0){Gz[-4]=
CH[G1[1]];Gz[-3]=CH[G1[2]];Gz[-2]=CH[G1[3]];Gz[-1]=CH[G1[0]];Gz+=4;G1+=4;}Gz-=4;
return G0;}static void DV(void*Gw,int Gx,float const*Gy){unsigned char*restrict
Gz=(unsigned char*)Gw;unsigned char*G0=((unsigned char*)Gz)+Gx;Gz+=4;while(Gz<=
G0){Gz[-4]=CK(Gy[3]);Gz[-3]=CK(Gy[0]);Gz[-2]=CK(Gy[1]);Gz[-1]=CK(Gy[2]);Gz+=4;Gy
+=4;}Gz-=4;}static float*DW(float*Gw,int Gx,void const*Gy){float*restrict Gz=Gw;
float*G0=(float*)Gz+Gx;unsigned char const*G1=(unsigned char const*)Gy;do{Gz[0]=
CH[G1[1]];Gz[1]=CH[G1[2]];Gz[2]=CH[G1[3]];Gz[3]=((float)G1[0])*3.922e-03;G1+=4;
Gz+=4;}while(Gz<G0);return G0;}static void DX(void*Gw,int Gx,float const*Gy){///
unsigned char*restrict Gz=(unsigned char*)Gw;unsigned char*G0=((unsigned char*)
Gz)+Gx;do{float G1;Gz[1]=CK(Gy[0]);Gz[2]=CK(Gy[1]);Gz[3]=CK(Gy[2]);G1=Gy[3]*////
255.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[0]=(unsigned char)G1
;Gz+=4;Gy+=4;}while(Gz<G0);}static float*DY(float*Gw,int Gx,void const*Gy){float
*restrict Gz=Gw;float*G0=(float*)Gz+Gx;unsigned short const*G1=(unsigned short//
const*)Gy;Gz+=4;while(Gz<=G0){Gz[-4]=((float)G1[1])*1.526e-05;Gz[-3]=((float)G1[
2])*1.526e-05;Gz[-2]=((float)G1[3])*1.526e-05;Gz[-1]=((float)G1[0])*1.526e-05;Gz
+=4;G1+=4;}Gz-=4;return G0;}static void DZ(void*Gw,int Gx,float const*Gy){//////
unsigned short*restrict Gz=(unsigned short*)Gw;unsigned short*G0=((unsigned/////
short*)Gz)+Gx;Gz+=4;while(Gz<=G0){float G1;G1=Gy[3]*65535.0f+0.5f;for(;;){if(G1<
0)G1=0;if(G1>65535)G1=65535;break;}Gz[-4]=(unsigned short)G1;G1=Gy[0]*65535.0f+
0.5f;for(;;){if(G1<0)G1=0;if(G1>65535)G1=65535;break;}Gz[-3]=(unsigned short)G1;
G1=Gy[1]*65535.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>65535)G1=65535;break;}Gz[-2]=(
unsigned short)G1;G1=Gy[2]*65535.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>65535)G1=///
65535;break;}Gz[-1]=(unsigned short)G1;Gz+=4;Gy+=4;}Gz-=4;}static float*Da(float
*Gw,int Gx,void const*Gy){float*restrict Gz=Gw;float*G0=(float*)Gz+Gx;unsigned//
short const*G1=(unsigned short const*)Gy;Gz+=4;while(Gz<=G0){Gz[-4]=((float)G1[1
]);Gz[-3]=((float)G1[2]);Gz[-2]=((float)G1[3]);Gz[-1]=((float)G1[0]);Gz+=4;G1+=4
;}Gz-=4;return G0;}static void Db(void*Gw,int Gx,float const*Gy){unsigned short*
restrict Gz=(unsigned short*)Gw;unsigned short*G0=((unsigned short*)Gz)+Gx;Gz+=4
;while(Gz<=G0){float G1;G1=Gy[3]+0.5f;for(;;){if(G1<0)G1=0;if(G1>65535)G1=65535;
break;}Gz[-4]=(unsigned short)G1;G1=Gy[0]+0.5f;for(;;){if(G1<0)G1=0;if(G1>65535)
G1=65535;break;}Gz[-3]=(unsigned short)G1;G1=Gy[1]+0.5f;for(;;){if(G1<0)G1=0;if(
G1>65535)G1=65535;break;}Gz[-2]=(unsigned short)G1;G1=Gy[2]+0.5f;for(;;){if(G1<0
)G1=0;if(G1>65535)G1=65535;break;}Gz[-1]=(unsigned short)G1;Gz+=4;Gy+=4;}Gz-=4;}
static float*Dc(float*Gw,int Gx,void const*Gy){float*restrict Gz=Gw;float*G0=(//
float*)Gz+Gx;CL const*G1=(CL const*)Gy;Gz+=4;while(Gz<=G0){Gz[-4]=CM(G1[1]);Gz[-
3]=CM(G1[2]);Gz[-2]=CM(G1[3]);Gz[-1]=CM(G1[0]);Gz+=4;G1+=4;}Gz-=4;return G0;}///
static void Dd(void*Gw,int Gx,float const*Gy){CL*restrict Gz=(CL*)Gw;CL*G0=((CL*
)Gz)+Gx;Gz+=4;while(Gz<=G0){Gz[-4]=CN(Gy[3]);Gz[-3]=CN(Gy[0]);Gz[-2]=CN(Gy[1]);
Gz[-1]=CN(Gy[2]);Gz+=4;Gy+=4;}Gz-=4;}static float*De(float*Gw,int Gx,void const*
Gy){float*restrict Gz=Gw;float*G0=(float*)Gz+Gx;float const*G1=(float const*)Gy;
Gz+=4;while(Gz<=G0){Gz[-4]=G1[1];Gz[-3]=G1[2];Gz[-2]=G1[3];Gz[-1]=G1[0];Gz+=4;G1
+=4;}Gz-=4;return G0;}static void Df(void*Gw,int Gx,float const*Gy){float*//////
restrict Gz=(float*)Gw;float*G0=((float*)Gz)+Gx;Gz+=4;while(Gz<=G0){float G1;G1=
Gy[3];Gz[-4]=G1;G1=Gy[0];Gz[-3]=G1;G1=Gy[1];Gz[-2]=G1;G1=Gy[2];Gz[-1]=G1;Gz+=4;
Gy+=4;}Gz-=4;}static float*Dg(float*Gw,int Gx,void const*Gy){float*restrict Gz=
Gw;float*G0=(float*)Gz+Gx;unsigned char const*G1=(unsigned char const*)Gy;Gz+=4;
while(Gz<=G0){Gz[-4]=((float)G1[3])*3.922e-03;Gz[-3]=((float)G1[2])*3.922e-03;Gz
[-2]=((float)G1[1])*3.922e-03;Gz[-1]=((float)G1[0])*3.922e-03;Gz+=4;G1+=4;}Gz-=4
;return G0;}static void Dh(void*Gw,int Gx,float const*Gy){unsigned char*restrict
Gz=(unsigned char*)Gw;unsigned char*G0=((unsigned char*)Gz)+Gx;Gz+=4;while(Gz<=
G0){float G1;G1=Gy[3]*255.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}
Gz[-4]=(unsigned char)G1;G1=Gy[2]*255.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>255)G1=
255;break;}Gz[-3]=(unsigned char)G1;G1=Gy[1]*255.0f+0.5f;for(;;){if(G1<0)G1=0;if
(G1>255)G1=255;break;}Gz[-2]=(unsigned char)G1;G1=Gy[0]*255.0f+0.5f;for(;;){if(
G1<0)G1=0;if(G1>255)G1=255;break;}Gz[-1]=(unsigned char)G1;Gz+=4;Gy+=4;}Gz-=4;}
static float*Di(float*Gw,int Gx,void const*Gy){float*restrict Gz=Gw;float*G0=(//
float*)Gz+Gx;unsigned char const*G1=(unsigned char const*)Gy;Gz+=4;while(Gz<=G0)
{Gz[-4]=((float)G1[3]);Gz[-3]=((float)G1[2]);Gz[-2]=((float)G1[1]);Gz[-1]=((////
float)G1[0]);Gz+=4;G1+=4;}Gz-=4;return G0;}static void Dj(void*Gw,int Gx,float//
const*Gy){unsigned char*restrict Gz=(unsigned char*)Gw;unsigned char*G0=((//////
unsigned char*)Gz)+Gx;Gz+=4;while(Gz<=G0){float G1;G1=Gy[3]+0.5f;for(;;){if(G1<0
)G1=0;if(G1>255)G1=255;break;}Gz[-4]=(unsigned char)G1;G1=Gy[2]+0.5f;for(;;){if(
G1<0)G1=0;if(G1>255)G1=255;break;}Gz[-3]=(unsigned char)G1;G1=Gy[1]+0.5f;for(;;)
{if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[-2]=(unsigned char)G1;G1=Gy[0]+0.5f;for
(;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[-1]=(unsigned char)G1;Gz+=4;Gy+=4;}
Gz-=4;}static float*Dk(float*Gw,int Gx,void const*Gy){float*restrict Gz=Gw;float
*G0=(float*)Gz+Gx;unsigned char const*G1=(unsigned char const*)Gy;Gz+=4;while(Gz
<=G0){Gz[-4]=CH[G1[3]];Gz[-3]=CH[G1[2]];Gz[-2]=CH[G1[1]];Gz[-1]=CH[G1[0]];Gz+=4;
G1+=4;}Gz-=4;return G0;}static void Dl(void*Gw,int Gx,float const*Gy){unsigned//
char*restrict Gz=(unsigned char*)Gw;unsigned char*G0=((unsigned char*)Gz)+Gx;Gz
+=4;while(Gz<=G0){Gz[-4]=CK(Gy[3]);Gz[-3]=CK(Gy[2]);Gz[-2]=CK(Gy[1]);Gz[-1]=CK(
Gy[0]);Gz+=4;Gy+=4;}Gz-=4;}static float*Dm(float*Gw,int Gx,void const*Gy){float*
restrict Gz=Gw;float*G0=(float*)Gz+Gx;unsigned char const*G1=(unsigned char/////
const*)Gy;do{Gz[0]=CH[G1[3]];Gz[1]=CH[G1[2]];Gz[2]=CH[G1[1]];Gz[3]=((float)G1[0]
)*3.922e-03;G1+=4;Gz+=4;}while(Gz<G0);return G0;}static void Dn(void*Gw,int Gx,
float const*Gy){unsigned char*restrict Gz=(unsigned char*)Gw;unsigned char*G0=((
unsigned char*)Gz)+Gx;do{float G1;Gz[3]=CK(Gy[0]);Gz[2]=CK(Gy[1]);Gz[1]=CK(Gy[2]
);G1=Gy[3]*255.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[0]=(/////
unsigned char)G1;Gz+=4;Gy+=4;}while(Gz<G0);}static float*Do(float*Gw,int Gx,void
const*Gy){float*restrict Gz=Gw;float*G0=(float*)Gz+Gx;unsigned short const*G1=(
unsigned short const*)Gy;Gz+=4;while(Gz<=G0){Gz[-4]=((float)G1[3])*1.526e-05;Gz[
-3]=((float)G1[2])*1.526e-05;Gz[-2]=((float)G1[1])*1.526e-05;Gz[-1]=((float)G1[0
])*1.526e-05;Gz+=4;G1+=4;}Gz-=4;return G0;}static void Dp(void*Gw,int Gx,float//
const*Gy){unsigned short*restrict Gz=(unsigned short*)Gw;unsigned short*G0=((///
unsigned short*)Gz)+Gx;Gz+=4;while(Gz<=G0){float G1;G1=Gy[3]*65535.0f+0.5f;for(;
;){if(G1<0)G1=0;if(G1>65535)G1=65535;break;}Gz[-4]=(unsigned short)G1;G1=Gy[2]*
65535.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>65535)G1=65535;break;}Gz[-3]=(unsigned
short)G1;G1=Gy[1]*65535.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>65535)G1=65535;break;
}Gz[-2]=(unsigned short)G1;G1=Gy[0]*65535.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>///
65535)G1=65535;break;}Gz[-1]=(unsigned short)G1;Gz+=4;Gy+=4;}Gz-=4;}static float
*Dq(float*Gw,int Gx,void const*Gy){float*restrict Gz=Gw;float*G0=(float*)Gz+Gx;
unsigned short const*G1=(unsigned short const*)Gy;Gz+=4;while(Gz<=G0){Gz[-4]=((
float)G1[3]);Gz[-3]=((float)G1[2]);Gz[-2]=((float)G1[1]);Gz[-1]=((float)G1[0]);
Gz+=4;G1+=4;}Gz-=4;return G0;}static void Dr(void*Gw,int Gx,float const*Gy){////
unsigned short*restrict Gz=(unsigned short*)Gw;unsigned short*G0=((unsigned/////
short*)Gz)+Gx;Gz+=4;while(Gz<=G0){float G1;G1=Gy[3]+0.5f;for(;;){if(G1<0)G1=0;if
(G1>65535)G1=65535;break;}Gz[-4]=(unsigned short)G1;G1=Gy[2]+0.5f;for(;;){if(G1<
0)G1=0;if(G1>65535)G1=65535;break;}Gz[-3]=(unsigned short)G1;G1=Gy[1]+0.5f;for(;
;){if(G1<0)G1=0;if(G1>65535)G1=65535;break;}Gz[-2]=(unsigned short)G1;G1=Gy[0]+
0.5f;for(;;){if(G1<0)G1=0;if(G1>65535)G1=65535;break;}Gz[-1]=(unsigned short)G1;
Gz+=4;Gy+=4;}Gz-=4;}static float*Ds(float*Gw,int Gx,void const*Gy){float*///////
restrict Gz=Gw;float*G0=(float*)Gz+Gx;CL const*G1=(CL const*)Gy;Gz+=4;while(Gz<=
G0){Gz[-4]=CM(G1[3]);Gz[-3]=CM(G1[2]);Gz[-2]=CM(G1[1]);Gz[-1]=CM(G1[0]);Gz+=4;G1
+=4;}Gz-=4;return G0;}static void Dt(void*Gw,int Gx,float const*Gy){CL*restrict
Gz=(CL*)Gw;CL*G0=((CL*)Gz)+Gx;Gz+=4;while(Gz<=G0){Gz[-4]=CN(Gy[3]);Gz[-3]=CN(Gy[
2]);Gz[-2]=CN(Gy[1]);Gz[-1]=CN(Gy[0]);Gz+=4;Gy+=4;}Gz-=4;}static float*Du(float*
Gw,int Gx,void const*Gy){float*restrict Gz=Gw;float*G0=(float*)Gz+Gx;float const
*G1=(float const*)Gy;Gz+=4;while(Gz<=G0){Gz[-4]=G1[3];Gz[-3]=G1[2];Gz[-2]=G1[1];
Gz[-1]=G1[0];Gz+=4;G1+=4;}Gz-=4;return G0;}static void Dv(void*Gw,int Gx,float//
const*Gy){float*restrict Gz=(float*)Gw;float*G0=((float*)Gz)+Gx;Gz+=4;while(Gz<=
G0){float G1;G1=Gy[3];Gz[-4]=G1;G1=Gy[2];Gz[-3]=G1;G1=Gy[1];Gz[-2]=G1;G1=Gy[0];
Gz[-1]=G1;Gz+=4;Gy+=4;}Gz-=4;}static float*Dw(float*Gw,int Gx,void const*Gy){///
float*restrict Gz=Gw;float*G0=(float*)Gz+Gx;unsigned char const*G1=(unsigned////
char const*)Gy;Gz+=4;while(Gz<=G0){Gz[-4]=((float)G1[1])*3.922e-03;Gz[-3]=((////
float)G1[0])*3.922e-03;Gz[-2]=((float)G1[3])*3.922e-03;Gz[-1]=((float)G1[2])*///
3.922e-03;Gz+=4;G1+=4;}Gz-=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while
(Gz<G0){asm(""::"r"(Gz));Gz[0]=((float)G1[1])*3.922e-03;Gz[1]=((float)G1[0])*///
3.922e-03;Gz+=2;G1+=2;}return G0;}static void Dx(void*Gw,int Gx,float const*Gy){
unsigned char*restrict Gz=(unsigned char*)Gw;unsigned char*G0=((unsigned char*)
Gz)+Gx;Gz+=4;while(Gz<=G0){float G1;G1=Gy[1]*255.0f+0.5f;for(;;){if(G1<0)G1=0;if
(G1>255)G1=255;break;}Gz[-4]=(unsigned char)G1;G1=Gy[0]*255.0f+0.5f;for(;;){if(
G1<0)G1=0;if(G1>255)G1=255;break;}Gz[-3]=(unsigned char)G1;G1=Gy[3]*255.0f+0.5f;
for(;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[-2]=(unsigned char)G1;G1=Gy[2]*
255.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[-1]=(unsigned char)
G1;Gz+=4;Gy+=4;}Gz-=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(Gz<G0)
{float G1;asm(""::"r"(Gy));G1=Gy[1]*255.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>255)
G1=255;break;}Gz[0]=(unsigned char)G1;G1=Gy[0]*255.0f+0.5f;for(;;){if(G1<0)G1=0;
if(G1>255)G1=255;break;}Gz[1]=(unsigned char)G1;Gz+=2;Gy+=2;}}static float*Dy(//
float*Gw,int Gx,void const*Gy){float*restrict Gz=Gw;float*G0=(float*)Gz+Gx;/////
unsigned char const*G1=(unsigned char const*)Gy;Gz+=4;while(Gz<=G0){Gz[-4]=((///
float)G1[1]);Gz[-3]=((float)G1[0]);Gz[-2]=((float)G1[3]);Gz[-1]=((float)G1[2]);
Gz+=4;G1+=4;}Gz-=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(Gz<G0){//
asm(""::"r"(Gz));Gz[0]=((float)G1[1]);Gz[1]=((float)G1[0]);Gz+=2;G1+=2;}return//
G0;}static void Dz(void*Gw,int Gx,float const*Gy){unsigned char*restrict Gz=(///
unsigned char*)Gw;unsigned char*G0=((unsigned char*)Gz)+Gx;Gz+=4;while(Gz<=G0){
float G1;G1=Gy[1]+0.5f;for(;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[-4]=(////
unsigned char)G1;G1=Gy[0]+0.5f;for(;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[-
3]=(unsigned char)G1;G1=Gy[3]+0.5f;for(;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}
Gz[-2]=(unsigned char)G1;G1=Gy[2]+0.5f;for(;;){if(G1<0)G1=0;if(G1>255)G1=255;///
break;}Gz[-1]=(unsigned char)G1;Gz+=4;Gy+=4;}Gz-=4;_Pragma("GCC unroll 1")//////
_Pragma("GCC novector")while(Gz<G0){float G1;asm(""::"r"(Gy));G1=Gy[1]+0.5f;for(
;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[0]=(unsigned char)G1;G1=Gy[0]+0.5f;
for(;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[1]=(unsigned char)G1;Gz+=2;Gy+=2
;}}static float*D0(float*Gw,int Gx,void const*Gy){float*restrict Gz=Gw;float*G0=
(float*)Gz+Gx;unsigned char const*G1=(unsigned char const*)Gy;Gz+=4;while(Gz<=G0
){Gz[-4]=CH[G1[1]];Gz[-3]=CH[G1[0]];Gz[-2]=CH[G1[3]];Gz[-1]=CH[G1[2]];Gz+=4;G1+=
4;}Gz-=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(Gz<G0){asm(""::"r"(
Gz));Gz[0]=CH[G1[1]];Gz[1]=CH[G1[0]];Gz+=2;G1+=2;}return G0;}static void D1(void
*Gw,int Gx,float const*Gy){unsigned char*restrict Gz=(unsigned char*)Gw;unsigned
char*G0=((unsigned char*)Gz)+Gx;Gz+=4;while(Gz<=G0){Gz[-4]=CK(Gy[1]);Gz[-3]=CK(
Gy[0]);Gz[-2]=CK(Gy[3]);Gz[-1]=CK(Gy[2]);Gz+=4;Gy+=4;}Gz-=4;_Pragma(////////////
"GCC unroll 1")_Pragma("GCC novector")while(Gz<G0){asm(""::"r"(Gy));Gz[0]=CK(Gy[
1]);Gz[1]=CK(Gy[0]);Gz+=2;Gy+=2;}}static float*D2(float*Gw,int Gx,void const*Gy)
{float*restrict Gz=Gw;float*G0=(float*)Gz+Gx;unsigned char const*G1=(unsigned///
char const*)Gy;Gz+=4;while(Gz<=G0){Gz[-4]=CH[G1[1]];Gz[-3]=((float)G1[0])*//////
3.922e-03;Gz[-2]=CH[G1[3]];Gz[-1]=((float)G1[2])*3.922e-03;G1+=4;Gz+=4;}Gz-=4;if
(Gz<G0){Gz[0]=CH[G1[1]];Gz[1]=((float)G1[0])*3.922e-03;}return G0;}static void//
D3(void*Gw,int Gx,float const*Gy){unsigned char*restrict Gz=(unsigned char*)Gw;
unsigned char*G0=((unsigned char*)Gz)+Gx;do{float G1;Gz[1]=CK(Gy[0]);G1=Gy[1]*//
255.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>255)G1=255;break;}Gz[0]=(unsigned char)G1
;Gz+=2;Gy+=2;}while(Gz<G0);}static float*D4(float*Gw,int Gx,void const*Gy){float
*restrict Gz=Gw;float*G0=(float*)Gz+Gx;unsigned short const*G1=(unsigned short//
const*)Gy;Gz+=4;while(Gz<=G0){Gz[-4]=((float)G1[1])*1.526e-05;Gz[-3]=((float)G1[
0])*1.526e-05;Gz[-2]=((float)G1[3])*1.526e-05;Gz[-1]=((float)G1[2])*1.526e-05;Gz
+=4;G1+=4;}Gz-=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(Gz<G0){asm(
""::"r"(Gz));Gz[0]=((float)G1[1])*1.526e-05;Gz[1]=((float)G1[0])*1.526e-05;Gz+=2
;G1+=2;}return G0;}static void D5(void*Gw,int Gx,float const*Gy){unsigned short*
restrict Gz=(unsigned short*)Gw;unsigned short*G0=((unsigned short*)Gz)+Gx;Gz+=4
;while(Gz<=G0){float G1;G1=Gy[1]*65535.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>65535)
G1=65535;break;}Gz[-4]=(unsigned short)G1;G1=Gy[0]*65535.0f+0.5f;for(;;){if(G1<0
)G1=0;if(G1>65535)G1=65535;break;}Gz[-3]=(unsigned short)G1;G1=Gy[3]*65535.0f+//
0.5f;for(;;){if(G1<0)G1=0;if(G1>65535)G1=65535;break;}Gz[-2]=(unsigned short)G1;
G1=Gy[2]*65535.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>65535)G1=65535;break;}Gz[-1]=(
unsigned short)G1;Gz+=4;Gy+=4;}Gz-=4;_Pragma("GCC unroll 1")_Pragma(////////////
"GCC novector")while(Gz<G0){float G1;asm(""::"r"(Gy));G1=Gy[1]*65535.0f+0.5f;for
(;;){if(G1<0)G1=0;if(G1>65535)G1=65535;break;}Gz[0]=(unsigned short)G1;G1=Gy[0]*
65535.0f+0.5f;for(;;){if(G1<0)G1=0;if(G1>65535)G1=65535;break;}Gz[1]=(unsigned//
short)G1;Gz+=2;Gy+=2;}}static float*D6(float*Gw,int Gx,void const*Gy){float*////
restrict Gz=Gw;float*G0=(float*)Gz+Gx;unsigned short const*G1=(unsigned short///
const*)Gy;Gz+=4;while(Gz<=G0){Gz[-4]=((float)G1[1]);Gz[-3]=((float)G1[0]);Gz[-2]
=((float)G1[3]);Gz[-1]=((float)G1[2]);Gz+=4;G1+=4;}Gz-=4;_Pragma("GCC unroll 1")
_Pragma("GCC novector")while(Gz<G0){asm(""::"r"(Gz));Gz[0]=((float)G1[1]);Gz[1]=
((float)G1[0]);Gz+=2;G1+=2;}return G0;}static void D7(void*Gw,int Gx,float const
*Gy){unsigned short*restrict Gz=(unsigned short*)Gw;unsigned short*G0=((unsigned
short*)Gz)+Gx;Gz+=4;while(Gz<=G0){float G1;G1=Gy[1]+0.5f;for(;;){if(G1<0)G1=0;if
(G1>65535)G1=65535;break;}Gz[-4]=(unsigned short)G1;G1=Gy[0]+0.5f;for(;;){if(G1<
0)G1=0;if(G1>65535)G1=65535;break;}Gz[-3]=(unsigned short)G1;G1=Gy[3]+0.5f;for(;
;){if(G1<0)G1=0;if(G1>65535)G1=65535;break;}Gz[-2]=(unsigned short)G1;G1=Gy[2]+
0.5f;for(;;){if(G1<0)G1=0;if(G1>65535)G1=65535;break;}Gz[-1]=(unsigned short)G1;
Gz+=4;Gy+=4;}Gz-=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(Gz<G0){//
float G1;asm(""::"r"(Gy));G1=Gy[1]+0.5f;for(;;){if(G1<0)G1=0;if(G1>65535)G1=////
65535;break;}Gz[0]=(unsigned short)G1;G1=Gy[0]+0.5f;for(;;){if(G1<0)G1=0;if(G1>
65535)G1=65535;break;}Gz[1]=(unsigned short)G1;Gz+=2;Gy+=2;}}static float*D8(///
float*Gw,int Gx,void const*Gy){float*restrict Gz=Gw;float*G0=(float*)Gz+Gx;CL///
const*G1=(CL const*)Gy;Gz+=4;while(Gz<=G0){Gz[-4]=CM(G1[1]);Gz[-3]=CM(G1[0]);Gz[
-2]=CM(G1[3]);Gz[-1]=CM(G1[2]);Gz+=4;G1+=4;}Gz-=4;_Pragma("GCC unroll 1")_Pragma
("GCC novector")while(Gz<G0){asm(""::"r"(Gz));Gz[0]=CM(G1[1]);Gz[1]=CM(G1[0]);Gz
+=2;G1+=2;}return G0;}static void D9(void*Gw,int Gx,float const*Gy){CL*restrict
Gz=(CL*)Gw;CL*G0=((CL*)Gz)+Gx;Gz+=4;while(Gz<=G0){Gz[-4]=CN(Gy[1]);Gz[-3]=CN(Gy[
0]);Gz[-2]=CN(Gy[3]);Gz[-1]=CN(Gy[2]);Gz+=4;Gy+=4;}Gz-=4;_Pragma("GCC unroll 1")
_Pragma("GCC novector")while(Gz<G0){asm(""::"r"(Gz));Gz[0]=CN(Gy[1]);Gz[1]=CN(Gy
[0]);Gz+=2;Gy+=2;}}static float*D_(float*Gw,int Gx,void const*Gy){float*restrict
Gz=Gw;float*G0=(float*)Gz+Gx;float const*G1=(float const*)Gy;Gz+=4;while(Gz<=G0)
{Gz[-4]=G1[1];Gz[-3]=G1[0];Gz[-2]=G1[3];Gz[-1]=G1[2];Gz+=4;G1+=4;}Gz-=4;_Pragma(
"GCC unroll 1")_Pragma("GCC novector")while(Gz<G0){asm(""::"r"(Gz));Gz[0]=G1[1];
Gz[1]=G1[0];Gz+=2;G1+=2;}return G0;}static void EA(void*Gw,int Gx,float const*Gy
){float*restrict Gz=(float*)Gw;float*G0=((float*)Gz)+Gx;Gz+=4;while(Gz<=G0){////
float G1;G1=Gy[1];Gz[-4]=G1;G1=Gy[0];Gz[-3]=G1;G1=Gy[3];Gz[-2]=G1;G1=Gy[2];Gz[-1
]=G1;Gz+=4;Gy+=4;}Gz-=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(Gz<
G0){float G1;asm(""::"r"(Gy));G1=Gy[1];Gz[0]=G1;G1=Gy[0];Gz[1]=G1;Gz+=2;Gy+=2;}}
static void EB(float*Gw,int Gx){float*restrict Gy=Gw;float const*Gz=Gw+(Gx/4)*7;
float*restrict G0=(float*)Gz-Gx;while(G0<Gz){float G1=G0[0],G2=G0[1],G3=G0[2],G4
=G0[3];Gy[0]=G1;Gy[1]=G2;Gy[2]=G3;Gy[3]=G4;Gy[4]=G1*G4;Gy[5]=G2*G4;Gy[6]=G3*G4;
Gy+=7;G0+=4;}}static void EC(float*Gw,int Gx){float*restrict Gy=Gw;float const*
Gz=Gw+(Gx/2)*3;float*restrict G0=(float*)Gz-Gx;while(G0<Gz){float G1=G0[0],G2=G0
[1];Gy[0]=G1;Gy[1]=G2;Gy[2]=G1*G2;Gy+=3;G0+=2;}}static void ED(float*Gw,int Gx){
float*restrict Gy=Gw;float*restrict Gz=Gw;float const*G0=Gw+Gx;do{float G1=Gz[3]
;if(G1<7.523e-37){Gy[0]=Gz[0];Gy[1]=Gz[1];Gy[2]=Gz[2];}else{float G2=1.0f/G1;Gy[
0]=Gz[4]*G2;Gy[1]=Gz[5]*G2;Gy[2]=Gz[6]*G2;}Gy[3]=G1;Gz+=7;Gy+=4;}while(Gy<G0);}
static void EE(float*Gw,int Gx){float*restrict Gy=Gw;float*restrict Gz=Gw;float
const*G0=Gw+Gx;do{float G1=Gz[1];Gy[0]=Gz[0];if(G1>=7.523e-37)Gy[0]=Gz[2]/G1;Gy[
1]=G1;Gz+=3;Gy+=2;}while(Gy<G0);}static void EF(float*Gw,int Gx){float*restrict
Gy=Gw;float const*Gz=Gw+Gx;while(Gy<Gz){float G0=Gy[3];Gy[0]*=G0;Gy[1]*=G0;Gy[2]
*=G0;Gy+=4;}}static void EG(float*Gw,int Gx){float*restrict Gy=Gw;float const*Gz
=Gw+Gx;while(Gy<Gz){float G0=Gy[1];Gy[0]*=G0;Gy+=2;}}static void EH(float*Gw,int
Gx){float*restrict Gy=Gw;float const*Gz=Gw+Gx;do{float G0=Gy[3];if(G0>=7.523e-37
){float G1=1.0f/G0;Gy[0]*=G1;Gy[1]*=G1;Gy[2]*=G1;}Gy+=4;}while(Gy<Gz);}static///
void EI(float*Gw,int Gx){float*restrict Gy=Gw;float const*Gz=Gw+Gx;do{float G0=
Gy[1];if(G0>=7.523e-37)Gy[0]/=G0;Gy+=2;}while(Gy<Gz);}static void EJ(float*Gw,//
int Gx){float*restrict Gy=Gw;float const*Gz=Gw+Gx;Gz-=12;_Pragma("GCC unroll 1")
_Pragma("GCC novector")while(Gy<=Gz){float G0,G1,G2,G3;asm(""::"r"(Gy));G0=Gy[0]
;G1=Gy[3];G2=Gy[6];G3=Gy[9];Gy[0]=Gy[2];Gy[3]=Gy[5];Gy[6]=Gy[8];Gy[9]=Gy[11];Gy[
2]=G0;Gy[5]=G1;Gy[8]=G2;Gy[11]=G3;Gy+=12;}Gz+=12;_Pragma("GCC unroll 1")_Pragma(
"GCC novector")while(Gy<Gz){float G0=Gy[0];asm(""::"r"(Gy));Gy[0]=Gy[2];Gy[2]=G0
;Gy+=3;}}static void EK(B0 const*Gw,int Gx,float*Gy){int Gz=Gw->d;int G0=Gw->e;
int G1=B3[Gw->I]*Gz;Bt G2=Gw->A.J;Bt G3=Gw->B.J;int G4=Cj(G3,Gx,Gw->B.E.A);const
void*G5=((char*)Gw->C)+(size_t)G4*(size_t)Gw->E;B6 const*G6=Gw->N.C;float*G7=Gy-
Gw->N.A.A*G0;float*G8=0;do{float*G9;void const*G_;float*HA;int HB;int HC;if(G6->
B<G6->A)break;HC=G6->B+1-G6->A;G9=G7+G6->A*G0;HA=G7+(G6->B+1)*G0;HB=HC*Gz;G_=((
char*)G5)+G6->C*G1;if(Gw->K)G_=Gw->K(((char*)HA)-(HC*G1)+((Gw->I!=//////////////
STBIR_TYPE_FLOAT)?12:0),G5,HC,G6->C,G4,Gw->L);G8=Gw->Q((float*)HA-HB,HB,G_);if(
Gw->R)Gw->R(G9,HB);++G6;}while(G6<=(&Gw->N.C[1]));if((G2==STBIR_EDGE_WRAP)&&(Gw
->N.B[0]|Gw->N.B[1])){int G9,G_[2];int HA=Gw->A.E.A;G_[0]=-Gw->N.B[0];G_[1]=HA;
for(G9=0;G9<2;G9++){int HB=Gw->N.B[G9];if(HB){int HC=G_[G9];float*HD=G7+HC*G0;//
float const*HE=G7+Cj(G2,HC,HA)*G0;memcpy(HD,HE,HB*G0*4);if(G9==1)G8=HD+HB*G0;}}}
G8[0]=0.0f;G8[1]=0.0f;}static void EL(float*Gw,unsigned int Gx,float const*Gy,B4
const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*1;float*restrict G3=Gw;do{
float const*G4=Gy+Gz->A*1;float const*G5=G0;float G6;G6=G4[0]*G5[0];G3[0]=G6;G0
+=G1;++Gz;G3+=1;}while(G3<G2);}static void EM(float*Gw,unsigned int Gx,float////
const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*1;float*///////
restrict G3=Gw;do{float const*G4=Gy+Gz->A*1;float const*G5=G0;float G6;G6=G4[0]*
G5[0];G6+=G4[1]*G5[1];G3[0]=G6;G0+=G1;++Gz;G3+=1;}while(G3<G2);}static void EN(
float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz,float const*G0,int G1){float
const*G2=Gw+Gx*1;float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*1;float const*
G5=G0;float G6;G6=G4[0]*G5[0];G6+=G4[1]*G5[1];G6+=G4[2]*G5[2];G3[0]=G6;G0+=G1;++
Gz;G3+=1;}while(G3<G2);}static void EO(float*Gw,unsigned int Gx,float const*Gy,
B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*1;float*restrict G3=Gw;
do{float const*G4=Gy+Gz->A*1;float const*G5=G0;float G6,G7,G8,G9;G6=G4[0]*G5[0];
G7=G4[1]*G5[1];G8=G4[2]*G5[2];G9=G4[3]*G5[3];G3[0]=(G6+G8)+(G7+G9);G0+=G1;++Gz;
G3+=1;}while(G3<G2);}static void EP(float*Gw,unsigned int Gx,float const*Gy,B4//
const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*1;float*restrict G3=Gw;do{
float const*G4=Gy+Gz->A*1;float const*G5=G0;float G6,G7,G8,G9;G6=G4[0]*G5[0];G7=
G4[1]*G5[1];G8=G4[2]*G5[2];G9=G4[3]*G5[3];G6+=G4[4]*G5[4];G3[0]=(G6+G8)+(G7+G9);
G0+=G1;++Gz;G3+=1;}while(G3<G2);}static void EQ(float*Gw,unsigned int Gx,float//
const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*1;float*///////
restrict G3=Gw;do{float const*G4=Gy+Gz->A*1;float const*G5=G0;float G6,G7,G8,G9;
G6=G4[0]*G5[0];G7=G4[1]*G5[1];G8=G4[2]*G5[2];G9=G4[3]*G5[3];G6+=G4[4]*G5[4];G7+=
G4[5]*G5[5];G3[0]=(G6+G8)+(G7+G9);G0+=G1;++Gz;G3+=1;}while(G3<G2);}static void//
ER(float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz,float const*G0,int G1){//
float const*G2=Gw+Gx*1;float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*1;float//
const*G5=G0;float G6,G7,G8,G9;G6=G4[0]*G5[0];G7=G4[1]*G5[1];G8=G4[2]*G5[2];G9=G4
[3]*G5[3];G6+=G4[4]*G5[4];G7+=G4[5]*G5[5];G8+=G4[6]*G5[6];G3[0]=(G6+G8)+(G7+G9);
G0+=G1;++Gz;G3+=1;}while(G3<G2);}static void ES(float*Gw,unsigned int Gx,float//
const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*1;float*///////
restrict G3=Gw;do{float const*G4=Gy+Gz->A*1;float const*G5=G0;float G6,G7,G8,G9;
G6=G4[0]*G5[0];G7=G4[1]*G5[1];G8=G4[2]*G5[2];G9=G4[3]*G5[3];G6+=G4[4]*G5[4];G7+=
G4[5]*G5[5];G8+=G4[6]*G5[6];G9+=G4[7]*G5[7];G3[0]=(G6+G8)+(G7+G9);G0+=G1;++Gz;G3
+=1;}while(G3<G2);}static void ET(float*Gw,unsigned int Gx,float const*Gy,B4////
const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*1;float*restrict G3=Gw;do{
float const*G4=Gy+Gz->A*1;float const*G5=G0;float G6,G7,G8,G9;G6=G4[0]*G5[0];G7=
G4[1]*G5[1];G8=G4[2]*G5[2];G9=G4[3]*G5[3];G6+=G4[4]*G5[4];G7+=G4[5]*G5[5];G8+=G4
[6]*G5[6];G9+=G4[7]*G5[7];G6+=G4[8]*G5[8];G3[0]=(G6+G8)+(G7+G9);G0+=G1;++Gz;G3+=
1;}while(G3<G2);}static void EU(float*Gw,unsigned int Gx,float const*Gy,B4 const
*Gz,float const*G0,int G1){float const*G2=Gw+Gx*1;float*restrict G3=Gw;do{float
const*G4=Gy+Gz->A*1;float const*G5=G0;float G6,G7,G8,G9;G6=G4[0]*G5[0];G7=G4[1]*
G5[1];G8=G4[2]*G5[2];G9=G4[3]*G5[3];G6+=G4[4]*G5[4];G7+=G4[5]*G5[5];G8+=G4[6]*G5
[6];G9+=G4[7]*G5[7];G6+=G4[8]*G5[8];G7+=G4[9]*G5[9];G3[0]=(G6+G8)+(G7+G9);G0+=G1
;++Gz;G3+=1;}while(G3<G2);}static void EV(float*Gw,unsigned int Gx,float const*
Gy,B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*1;float*restrict G3=
Gw;do{float const*G4=Gy+Gz->A*1;float const*G5=G0;float G6,G7,G8,G9;G6=G4[0]*G5[
0];G7=G4[1]*G5[1];G8=G4[2]*G5[2];G9=G4[3]*G5[3];G6+=G4[4]*G5[4];G7+=G4[5]*G5[5];
G8+=G4[6]*G5[6];G9+=G4[7]*G5[7];G6+=G4[8]*G5[8];G7+=G4[9]*G5[9];G8+=G4[10]*G5[10
];G3[0]=(G6+G8)+(G7+G9);G0+=G1;++Gz;G3+=1;}while(G3<G2);}static void EW(float*Gw
,unsigned int Gx,float const*Gy,B4 const*Gz,float const*G0,int G1){float const*
G2=Gw+Gx*1;float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*1;float const*G5=G0;
float G6,G7,G8,G9;G6=G4[0]*G5[0];G7=G4[1]*G5[1];G8=G4[2]*G5[2];G9=G4[3]*G5[3];G6
+=G4[4]*G5[4];G7+=G4[5]*G5[5];G8+=G4[6]*G5[6];G9+=G4[7]*G5[7];G6+=G4[8]*G5[8];G7
+=G4[9]*G5[9];G8+=G4[10]*G5[10];G9+=G4[11]*G5[11];G3[0]=(G6+G8)+(G7+G9);G0+=G1;
++Gz;G3+=1;}while(G3<G2);}static void EX(float*Gw,unsigned int Gx,float const*Gy
,B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*1;float*restrict G3=Gw;
do{float const*G4=Gy+Gz->A*1;int G5=((Gz->B-Gz->A+1)-4+3)>>2;float const*G6=G0;
float G7,G8,G9,G_;G7=G4[0]*G6[0];G8=G4[1]*G6[1];G9=G4[2]*G6[2];G_=G4[3]*G6[3];do
{G6+=4;G4+=4;G7+=G4[0]*G6[0];G8+=G4[1]*G6[1];G9+=G4[2]*G6[2];G_+=G4[3]*G6[3];--
G5;}while(G5>0);G3[0]=(G7+G9)+(G8+G_);G0+=G1;++Gz;G3+=1;}while(G3<G2);}static///
void EY(float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz,float const*G0,int//
G1){float const*G2=Gw+Gx*1;float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*1;int
G5=((Gz->B-Gz->A+1)-5+3)>>2;float const*G6=G0;float G7,G8,G9,G_;G7=G4[0]*G6[0];
G8=G4[1]*G6[1];G9=G4[2]*G6[2];G_=G4[3]*G6[3];do{G6+=4;G4+=4;G7+=G4[0]*G6[0];G8+=
G4[1]*G6[1];G9+=G4[2]*G6[2];G_+=G4[3]*G6[3];--G5;}while(G5>0);G7+=G4[4]*G6[4];G3
[0]=(G7+G9)+(G8+G_);G0+=G1;++Gz;G3+=1;}while(G3<G2);}static void EZ(float*Gw,///
unsigned int Gx,float const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2
=Gw+Gx*1;float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*1;int G5=((Gz->B-Gz->A+
1)-6+3)>>2;float const*G6=G0;float G7,G8,G9,G_;G7=G4[0]*G6[0];G8=G4[1]*G6[1];G9=
G4[2]*G6[2];G_=G4[3]*G6[3];do{G6+=4;G4+=4;G7+=G4[0]*G6[0];G8+=G4[1]*G6[1];G9+=G4
[2]*G6[2];G_+=G4[3]*G6[3];--G5;}while(G5>0);G7+=G4[4]*G6[4];G8+=G4[5]*G6[5];G3[0
]=(G7+G9)+(G8+G_);G0+=G1;++Gz;G3+=1;}while(G3<G2);}static void Ea(float*Gw,/////
unsigned int Gx,float const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2
=Gw+Gx*1;float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*1;int G5=((Gz->B-Gz->A+
1)-7+3)>>2;float const*G6=G0;float G7,G8,G9,G_;G7=G4[0]*G6[0];G8=G4[1]*G6[1];G9=
G4[2]*G6[2];G_=G4[3]*G6[3];do{G6+=4;G4+=4;G7+=G4[0]*G6[0];G8+=G4[1]*G6[1];G9+=G4
[2]*G6[2];G_+=G4[3]*G6[3];--G5;}while(G5>0);G7+=G4[4]*G6[4];G8+=G4[5]*G6[5];G9+=
G4[6]*G6[6];G3[0]=(G7+G9)+(G8+G_);G0+=G1;++Gz;G3+=1;}while(G3<G2);}static CC*Eb[
4]={EX,EY,EZ,Ea,};static CC*Ec[12]={EL,EM,EN,EO,EP,EQ,ER,ES,ET,EU,EV,EW,};static
void Ed(float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz,float const*G0,int//
G1){float const*G2=Gw+Gx*2;float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*2;///
float const*G5=G0;float G6,G7,G8;G8=G5[0];G6=G4[0]*G8;G7=G4[1]*G8;G3[0]=G6;G3[1]
=G7;G0+=G1;++Gz;G3+=2;}while(G3<G2);}static void Ee(float*Gw,unsigned int Gx,///
float const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*2;float*
restrict G3=Gw;do{float const*G4=Gy+Gz->A*2;float const*G5=G0;float G6,G7,G8;G8=
G5[0];G6=G4[0]*G8;G7=G4[1]*G8;G8=G5[1];G6+=G4[2]*G8;G7+=G4[3]*G8;G3[0]=G6;G3[1]=
G7;G0+=G1;++Gz;G3+=2;}while(G3<G2);}static void Ef(float*Gw,unsigned int Gx,////
float const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*2;float*
restrict G3=Gw;do{float const*G4=Gy+Gz->A*2;float const*G5=G0;float G6,G7,G8;G8=
G5[0];G6=G4[0]*G8;G7=G4[1]*G8;G8=G5[2];G6+=G4[4]*G8;G7+=G4[5]*G8;G8=G5[1];G6+=G4
[2]*G8;G7+=G4[3]*G8;G3[0]=G6;G3[1]=G7;G0+=G1;++Gz;G3+=2;}while(G3<G2);}static///
void Eg(float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz,float const*G0,int//
G1){float const*G2=Gw+Gx*2;float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*2;///
float const*G5=G0;float G6,G7,G8,G9,G_,HA,HB,HC,HD;HD=G5[0];G6=G4[0]*HD;G_=G4[1]
*HD;HD=G5[1];G7=G4[2]*HD;HA=G4[3]*HD;HD=G5[2];G8=G4[4]*HD;HB=G4[5]*HD;HD=G5[3];
G9=G4[6]*HD;HC=G4[7]*HD;G3[0]=(G6+G8)+(G7+G9);G3[1]=(G_+HB)+(HA+HC);G0+=G1;++Gz;
G3+=2;}while(G3<G2);}static void Eh(float*Gw,unsigned int Gx,float const*Gy,B4//
const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*2;float*restrict G3=Gw;do{
float const*G4=Gy+Gz->A*2;float const*G5=G0;float G6,G7,G8,G9,G_,HA,HB,HC,HD;HD=
G5[0];G6=G4[0]*HD;G_=G4[1]*HD;HD=G5[1];G7=G4[2]*HD;HA=G4[3]*HD;HD=G5[2];G8=G4[4]
*HD;HB=G4[5]*HD;HD=G5[3];G9=G4[6]*HD;HC=G4[7]*HD;HD=G5[4];G6+=G4[8]*HD;G_+=G4[9]
*HD;G3[0]=(G6+G8)+(G7+G9);G3[1]=(G_+HB)+(HA+HC);G0+=G1;++Gz;G3+=2;}while(G3<G2);
}static void Ei(float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz,float const*
G0,int G1){float const*G2=Gw+Gx*2;float*restrict G3=Gw;do{float const*G4=Gy+Gz->
A*2;float const*G5=G0;float G6,G7,G8,G9,G_,HA,HB,HC,HD;HD=G5[0];G6=G4[0]*HD;G_=
G4[1]*HD;HD=G5[1];G7=G4[2]*HD;HA=G4[3]*HD;HD=G5[2];G8=G4[4]*HD;HB=G4[5]*HD;HD=G5
[3];G9=G4[6]*HD;HC=G4[7]*HD;HD=G5[4];G6+=G4[8]*HD;G_+=G4[9]*HD;HD=G5[5];G7+=G4[
10]*HD;HA+=G4[11]*HD;G3[0]=(G6+G8)+(G7+G9);G3[1]=(G_+HB)+(HA+HC);G0+=G1;++Gz;G3
+=2;}while(G3<G2);}static void Ej(float*Gw,unsigned int Gx,float const*Gy,B4////
const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*2;float*restrict G3=Gw;do{
float const*G4=Gy+Gz->A*2;float const*G5=G0;float G6,G7,G8,G9,G_,HA,HB,HC,HD;HD=
G5[0];G6=G4[0]*HD;G_=G4[1]*HD;HD=G5[1];G7=G4[2]*HD;HA=G4[3]*HD;HD=G5[2];G8=G4[4]
*HD;HB=G4[5]*HD;HD=G5[3];G9=G4[6]*HD;HC=G4[7]*HD;HD=G5[4];G6+=G4[8]*HD;G_+=G4[9]
*HD;HD=G5[5];G7+=G4[10]*HD;HA+=G4[11]*HD;HD=G5[6];G8+=G4[12]*HD;HB+=G4[13]*HD;G3
[0]=(G6+G8)+(G7+G9);G3[1]=(G_+HB)+(HA+HC);G0+=G1;++Gz;G3+=2;}while(G3<G2);}/////
static void Ek(float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz,float const*
G0,int G1){float const*G2=Gw+Gx*2;float*restrict G3=Gw;do{float const*G4=Gy+Gz->
A*2;float const*G5=G0;float G6,G7,G8,G9,G_,HA,HB,HC,HD;HD=G5[0];G6=G4[0]*HD;G_=
G4[1]*HD;HD=G5[1];G7=G4[2]*HD;HA=G4[3]*HD;HD=G5[2];G8=G4[4]*HD;HB=G4[5]*HD;HD=G5
[3];G9=G4[6]*HD;HC=G4[7]*HD;HD=G5[4];G6+=G4[8]*HD;G_+=G4[9]*HD;HD=G5[5];G7+=G4[
10]*HD;HA+=G4[11]*HD;HD=G5[6];G8+=G4[12]*HD;HB+=G4[13]*HD;HD=G5[7];G9+=G4[14]*HD
;HC+=G4[15]*HD;G3[0]=(G6+G8)+(G7+G9);G3[1]=(G_+HB)+(HA+HC);G0+=G1;++Gz;G3+=2;}//
while(G3<G2);}static void El(float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz
,float const*G0,int G1){float const*G2=Gw+Gx*2;float*restrict G3=Gw;do{float////
const*G4=Gy+Gz->A*2;float const*G5=G0;float G6,G7,G8,G9,G_,HA,HB,HC,HD;HD=G5[0];
G6=G4[0]*HD;G_=G4[1]*HD;HD=G5[1];G7=G4[2]*HD;HA=G4[3]*HD;HD=G5[2];G8=G4[4]*HD;HB
=G4[5]*HD;HD=G5[3];G9=G4[6]*HD;HC=G4[7]*HD;HD=G5[4];G6+=G4[8]*HD;G_+=G4[9]*HD;HD
=G5[5];G7+=G4[10]*HD;HA+=G4[11]*HD;HD=G5[6];G8+=G4[12]*HD;HB+=G4[13]*HD;HD=G5[7]
;G9+=G4[14]*HD;HC+=G4[15]*HD;HD=G5[8];G6+=G4[16]*HD;G_+=G4[17]*HD;G3[0]=(G6+G8)+
(G7+G9);G3[1]=(G_+HB)+(HA+HC);G0+=G1;++Gz;G3+=2;}while(G3<G2);}static void Em(//
float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz,float const*G0,int G1){float
const*G2=Gw+Gx*2;float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*2;float const*
G5=G0;float G6,G7,G8,G9,G_,HA,HB,HC,HD;HD=G5[0];G6=G4[0]*HD;G_=G4[1]*HD;HD=G5[1]
;G7=G4[2]*HD;HA=G4[3]*HD;HD=G5[2];G8=G4[4]*HD;HB=G4[5]*HD;HD=G5[3];G9=G4[6]*HD;
HC=G4[7]*HD;HD=G5[4];G6+=G4[8]*HD;G_+=G4[9]*HD;HD=G5[5];G7+=G4[10]*HD;HA+=G4[11]
*HD;HD=G5[6];G8+=G4[12]*HD;HB+=G4[13]*HD;HD=G5[7];G9+=G4[14]*HD;HC+=G4[15]*HD;HD
=G5[8];G6+=G4[16]*HD;G_+=G4[17]*HD;HD=G5[9];G7+=G4[18]*HD;HA+=G4[19]*HD;G3[0]=(
G6+G8)+(G7+G9);G3[1]=(G_+HB)+(HA+HC);G0+=G1;++Gz;G3+=2;}while(G3<G2);}static////
void En(float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz,float const*G0,int//
G1){float const*G2=Gw+Gx*2;float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*2;///
float const*G5=G0;float G6,G7,G8,G9,G_,HA,HB,HC,HD;HD=G5[0];G6=G4[0]*HD;G_=G4[1]
*HD;HD=G5[1];G7=G4[2]*HD;HA=G4[3]*HD;HD=G5[2];G8=G4[4]*HD;HB=G4[5]*HD;HD=G5[3];
G9=G4[6]*HD;HC=G4[7]*HD;HD=G5[4];G6+=G4[8]*HD;G_+=G4[9]*HD;HD=G5[5];G7+=G4[10]*
HD;HA+=G4[11]*HD;HD=G5[6];G8+=G4[12]*HD;HB+=G4[13]*HD;HD=G5[7];G9+=G4[14]*HD;HC
+=G4[15]*HD;HD=G5[8];G6+=G4[16]*HD;G_+=G4[17]*HD;HD=G5[9];G7+=G4[18]*HD;HA+=G4[
19]*HD;HD=G5[10];G8+=G4[20]*HD;HB+=G4[21]*HD;G3[0]=(G6+G8)+(G7+G9);G3[1]=(G_+HB)
+(HA+HC);G0+=G1;++Gz;G3+=2;}while(G3<G2);}static void Eo(float*Gw,unsigned int//
Gx,float const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*2;////
float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*2;float const*G5=G0;float G6,G7,
G8,G9,G_,HA,HB,HC,HD;HD=G5[0];G6=G4[0]*HD;G_=G4[1]*HD;HD=G5[1];G7=G4[2]*HD;HA=G4
[3]*HD;HD=G5[2];G8=G4[4]*HD;HB=G4[5]*HD;HD=G5[3];G9=G4[6]*HD;HC=G4[7]*HD;HD=G5[4
];G6+=G4[8]*HD;G_+=G4[9]*HD;HD=G5[5];G7+=G4[10]*HD;HA+=G4[11]*HD;HD=G5[6];G8+=G4
[12]*HD;HB+=G4[13]*HD;HD=G5[7];G9+=G4[14]*HD;HC+=G4[15]*HD;HD=G5[8];G6+=G4[16]*
HD;G_+=G4[17]*HD;HD=G5[9];G7+=G4[18]*HD;HA+=G4[19]*HD;HD=G5[10];G8+=G4[20]*HD;HB
+=G4[21]*HD;HD=G5[11];G9+=G4[22]*HD;HC+=G4[23]*HD;G3[0]=(G6+G8)+(G7+G9);G3[1]=(
G_+HB)+(HA+HC);G0+=G1;++Gz;G3+=2;}while(G3<G2);}static void Ep(float*Gw,unsigned
int Gx,float const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*2;
float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*2;int G5=((Gz->B-Gz->A+1)-4+3)>>
2;float const*G6=G0;float G7,G8,G9,G_,HA,HB,HC,HD,HE;HE=G6[0];G7=G4[0]*HE;HA=G4[
1]*HE;HE=G6[1];G8=G4[2]*HE;HB=G4[3]*HE;HE=G6[2];G9=G4[4]*HE;HC=G4[5]*HE;HE=G6[3]
;G_=G4[6]*HE;HD=G4[7]*HE;do{G6+=4;G4+=8;HE=G6[0];G7+=G4[0]*HE;HA+=G4[1]*HE;HE=G6
[1];G8+=G4[2]*HE;HB+=G4[3]*HE;HE=G6[2];G9+=G4[4]*HE;HC+=G4[5]*HE;HE=G6[3];G_+=G4
[6]*HE;HD+=G4[7]*HE;--G5;}while(G5>0);G3[0]=(G7+G9)+(G8+G_);G3[1]=(HA+HC)+(HB+HD
);G0+=G1;++Gz;G3+=2;}while(G3<G2);}static void Eq(float*Gw,unsigned int Gx,float
const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*2;float*///////
restrict G3=Gw;do{float const*G4=Gy+Gz->A*2;int G5=((Gz->B-Gz->A+1)-5+3)>>2;////
float const*G6=G0;float G7,G8,G9,G_,HA,HB,HC,HD,HE;HE=G6[0];G7=G4[0]*HE;HA=G4[1]
*HE;HE=G6[1];G8=G4[2]*HE;HB=G4[3]*HE;HE=G6[2];G9=G4[4]*HE;HC=G4[5]*HE;HE=G6[3];
G_=G4[6]*HE;HD=G4[7]*HE;do{G6+=4;G4+=8;HE=G6[0];G7+=G4[0]*HE;HA+=G4[1]*HE;HE=G6[
1];G8+=G4[2]*HE;HB+=G4[3]*HE;HE=G6[2];G9+=G4[4]*HE;HC+=G4[5]*HE;HE=G6[3];G_+=G4[
6]*HE;HD+=G4[7]*HE;--G5;}while(G5>0);HE=G6[4];G7+=G4[8]*HE;HA+=G4[9]*HE;G3[0]=(
G7+G9)+(G8+G_);G3[1]=(HA+HC)+(HB+HD);G0+=G1;++Gz;G3+=2;}while(G3<G2);}static////
void Er(float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz,float const*G0,int//
G1){float const*G2=Gw+Gx*2;float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*2;int
G5=((Gz->B-Gz->A+1)-6+3)>>2;float const*G6=G0;float G7,G8,G9,G_,HA,HB,HC,HD,HE;
HE=G6[0];G7=G4[0]*HE;HA=G4[1]*HE;HE=G6[1];G8=G4[2]*HE;HB=G4[3]*HE;HE=G6[2];G9=G4
[4]*HE;HC=G4[5]*HE;HE=G6[3];G_=G4[6]*HE;HD=G4[7]*HE;do{G6+=4;G4+=8;HE=G6[0];G7+=
G4[0]*HE;HA+=G4[1]*HE;HE=G6[1];G8+=G4[2]*HE;HB+=G4[3]*HE;HE=G6[2];G9+=G4[4]*HE;
HC+=G4[5]*HE;HE=G6[3];G_+=G4[6]*HE;HD+=G4[7]*HE;--G5;}while(G5>0);HE=G6[4];G7+=
G4[8]*HE;HA+=G4[9]*HE;HE=G6[5];G8+=G4[10]*HE;HB+=G4[11]*HE;G3[0]=(G7+G9)+(G8+G_)
;G3[1]=(HA+HC)+(HB+HD);G0+=G1;++Gz;G3+=2;}while(G3<G2);}static void Es(float*Gw,
unsigned int Gx,float const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2
=Gw+Gx*2;float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*2;int G5=((Gz->B-Gz->A+
1)-7+3)>>2;float const*G6=G0;float G7,G8,G9,G_,HA,HB,HC,HD,HE;HE=G6[0];G7=G4[0]*
HE;HA=G4[1]*HE;HE=G6[1];G8=G4[2]*HE;HB=G4[3]*HE;HE=G6[2];G9=G4[4]*HE;HC=G4[5]*HE
;HE=G6[3];G_=G4[6]*HE;HD=G4[7]*HE;do{G6+=4;G4+=8;HE=G6[0];G7+=G4[0]*HE;HA+=G4[1]
*HE;HE=G6[1];G8+=G4[2]*HE;HB+=G4[3]*HE;HE=G6[2];G9+=G4[4]*HE;HC+=G4[5]*HE;HE=G6[
3];G_+=G4[6]*HE;HD+=G4[7]*HE;--G5;}while(G5>0);HE=G6[4];G7+=G4[8]*HE;HA+=G4[9]*
HE;HE=G6[5];G8+=G4[10]*HE;HB+=G4[11]*HE;HE=G6[6];G9+=G4[12]*HE;HC+=G4[13]*HE;G3[
0]=(G7+G9)+(G8+G_);G3[1]=(HA+HC)+(HB+HD);G0+=G1;++Gz;G3+=2;}while(G3<G2);}static
CC*Et[4]={Ep,Eq,Er,Es,};static CC*Eu[12]={Ed,Ee,Ef,Eg,Eh,Ei,Ej,Ek,El,Em,En,Eo,};
static void Ev(float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz,float const*
G0,int G1){float const*G2=Gw+Gx*3;float*restrict G3=Gw;do{float const*G4=Gy+Gz->
A*3;float const*G5=G0;float G6,G7,G8,G9;G9=G5[0];G6=G4[0]*G9;G7=G4[1]*G9;G8=G4[2
]*G9;G3[0]=G6;G3[1]=G7;G3[2]=G8;G0+=G1;++Gz;G3+=3;}while(G3<G2);}static void Ew(
float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz,float const*G0,int G1){float
const*G2=Gw+Gx*3;float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*3;float const*
G5=G0;float G6,G7,G8,G9;G9=G5[0];G6=G4[0]*G9;G7=G4[1]*G9;G8=G4[2]*G9;G9=G5[1];G6
+=G4[3]*G9;G7+=G4[4]*G9;G8+=G4[5]*G9;G3[0]=G6;G3[1]=G7;G3[2]=G8;G0+=G1;++Gz;G3+=
3;}while(G3<G2);}static void Ex(float*Gw,unsigned int Gx,float const*Gy,B4 const
*Gz,float const*G0,int G1){float const*G2=Gw+Gx*3;float*restrict G3=Gw;do{float
const*G4=Gy+Gz->A*3;float const*G5=G0;float G6,G7,G8,G9;G9=G5[0];G6=G4[0]*G9;G7=
G4[1]*G9;G8=G4[2]*G9;G9=G5[1];G6+=G4[3]*G9;G7+=G4[4]*G9;G8+=G4[5]*G9;G9=G5[2];G6
+=G4[6]*G9;G7+=G4[7]*G9;G8+=G4[8]*G9;G3[0]=G6;G3[1]=G7;G3[2]=G8;G0+=G1;++Gz;G3+=
3;}while(G3<G2);}static void Ey(float*Gw,unsigned int Gx,float const*Gy,B4 const
*Gz,float const*G0,int G1){float const*G2=Gw+Gx*3;float*restrict G3=Gw;do{float
const*G4=Gy+Gz->A*3;float const*G5=G0;float G6,G7,G8,G9,G_,HA,HB,HC,HD,HE,HF,HG,
HH;HH=G5[0];G6=G4[0]*HH;G7=G4[1]*HH;G8=G4[2]*HH;HH=G5[1];G9=G4[3]*HH;G_=G4[4]*HH
;HA=G4[5]*HH;HH=G5[2];HB=G4[6]*HH;HC=G4[7]*HH;HD=G4[8]*HH;HH=G5[3];HE=G4[9]*HH;
HF=G4[10]*HH;HG=G4[11]*HH;G3[0]=(G6+HB)+(G9+HE);G3[1]=(G7+HC)+(G_+HF);G3[2]=(G8+
HD)+(HA+HG);G0+=G1;++Gz;G3+=3;}while(G3<G2);}static void Ez(float*Gw,unsigned///
int Gx,float const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*3;
float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*3;float const*G5=G0;float G6,G7,
G8,G9,G_,HA,HB,HC,HD,HE,HF,HG,HH;HH=G5[0];G6=G4[0]*HH;G7=G4[1]*HH;G8=G4[2]*HH;HH
=G5[1];G9=G4[3]*HH;G_=G4[4]*HH;HA=G4[5]*HH;HH=G5[2];HB=G4[6]*HH;HC=G4[7]*HH;HD=
G4[8]*HH;HH=G5[3];HE=G4[9]*HH;HF=G4[10]*HH;HG=G4[11]*HH;HH=G5[4];G6+=G4[12]*HH;
G7+=G4[13]*HH;G8+=G4[14]*HH;G3[0]=(G6+HB)+(G9+HE);G3[1]=(G7+HC)+(G_+HF);G3[2]=(
G8+HD)+(HA+HG);G0+=G1;++Gz;G3+=3;}while(G3<G2);}static void E0(float*Gw,unsigned
int Gx,float const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*3;
float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*3;float const*G5=G0;float G6,G7,
G8,G9,G_,HA,HB,HC,HD,HE,HF,HG,HH;HH=G5[0];G6=G4[0]*HH;G7=G4[1]*HH;G8=G4[2]*HH;HH
=G5[1];G9=G4[3]*HH;G_=G4[4]*HH;HA=G4[5]*HH;HH=G5[2];HB=G4[6]*HH;HC=G4[7]*HH;HD=
G4[8]*HH;HH=G5[3];HE=G4[9]*HH;HF=G4[10]*HH;HG=G4[11]*HH;HH=G5[4];G6+=G4[12]*HH;
G7+=G4[13]*HH;G8+=G4[14]*HH;HH=G5[5];G9+=G4[15]*HH;G_+=G4[16]*HH;HA+=G4[17]*HH;
G3[0]=(G6+HB)+(G9+HE);G3[1]=(G7+HC)+(G_+HF);G3[2]=(G8+HD)+(HA+HG);G0+=G1;++Gz;G3
+=3;}while(G3<G2);}static void E1(float*Gw,unsigned int Gx,float const*Gy,B4////
const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*3;float*restrict G3=Gw;do{
float const*G4=Gy+Gz->A*3;float const*G5=G0;float G6,G7,G8,G9,G_,HA,HB,HC,HD,HE,
HF,HG,HH;HH=G5[0];G6=G4[0]*HH;G7=G4[1]*HH;G8=G4[2]*HH;HH=G5[1];G9=G4[3]*HH;G_=G4
[4]*HH;HA=G4[5]*HH;HH=G5[2];HB=G4[6]*HH;HC=G4[7]*HH;HD=G4[8]*HH;HH=G5[3];HE=G4[9
]*HH;HF=G4[10]*HH;HG=G4[11]*HH;HH=G5[4];G6+=G4[12]*HH;G7+=G4[13]*HH;G8+=G4[14]*
HH;HH=G5[5];G9+=G4[15]*HH;G_+=G4[16]*HH;HA+=G4[17]*HH;HH=G5[6];HB+=G4[18]*HH;HC
+=G4[19]*HH;HD+=G4[20]*HH;G3[0]=(G6+HB)+(G9+HE);G3[1]=(G7+HC)+(G_+HF);G3[2]=(G8+
HD)+(HA+HG);G0+=G1;++Gz;G3+=3;}while(G3<G2);}static void E2(float*Gw,unsigned///
int Gx,float const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*3;
float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*3;float const*G5=G0;float G6,G7,
G8,G9,G_,HA,HB,HC,HD,HE,HF,HG,HH;HH=G5[0];G6=G4[0]*HH;G7=G4[1]*HH;G8=G4[2]*HH;HH
=G5[1];G9=G4[3]*HH;G_=G4[4]*HH;HA=G4[5]*HH;HH=G5[2];HB=G4[6]*HH;HC=G4[7]*HH;HD=
G4[8]*HH;HH=G5[3];HE=G4[9]*HH;HF=G4[10]*HH;HG=G4[11]*HH;HH=G5[4];G6+=G4[12]*HH;
G7+=G4[13]*HH;G8+=G4[14]*HH;HH=G5[5];G9+=G4[15]*HH;G_+=G4[16]*HH;HA+=G4[17]*HH;
HH=G5[6];HB+=G4[18]*HH;HC+=G4[19]*HH;HD+=G4[20]*HH;HH=G5[7];HE+=G4[21]*HH;HF+=G4
[22]*HH;HG+=G4[23]*HH;G3[0]=(G6+HB)+(G9+HE);G3[1]=(G7+HC)+(G_+HF);G3[2]=(G8+HD)+
(HA+HG);G0+=G1;++Gz;G3+=3;}while(G3<G2);}static void E3(float*Gw,unsigned int Gx
,float const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*3;float*
restrict G3=Gw;do{float const*G4=Gy+Gz->A*3;float const*G5=G0;float G6,G7,G8,G9,
G_,HA,HB,HC,HD,HE,HF,HG,HH;HH=G5[0];G6=G4[0]*HH;G7=G4[1]*HH;G8=G4[2]*HH;HH=G5[1]
;G9=G4[3]*HH;G_=G4[4]*HH;HA=G4[5]*HH;HH=G5[2];HB=G4[6]*HH;HC=G4[7]*HH;HD=G4[8]*
HH;HH=G5[3];HE=G4[9]*HH;HF=G4[10]*HH;HG=G4[11]*HH;HH=G5[4];G6+=G4[12]*HH;G7+=G4[
13]*HH;G8+=G4[14]*HH;HH=G5[5];G9+=G4[15]*HH;G_+=G4[16]*HH;HA+=G4[17]*HH;HH=G5[6]
;HB+=G4[18]*HH;HC+=G4[19]*HH;HD+=G4[20]*HH;HH=G5[7];HE+=G4[21]*HH;HF+=G4[22]*HH;
HG+=G4[23]*HH;HH=G5[8];G6+=G4[24]*HH;G7+=G4[25]*HH;G8+=G4[26]*HH;G3[0]=(G6+HB)+(
G9+HE);G3[1]=(G7+HC)+(G_+HF);G3[2]=(G8+HD)+(HA+HG);G0+=G1;++Gz;G3+=3;}while(G3<
G2);}static void E4(float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz,float///
const*G0,int G1){float const*G2=Gw+Gx*3;float*restrict G3=Gw;do{float const*G4=
Gy+Gz->A*3;float const*G5=G0;float G6,G7,G8,G9,G_,HA,HB,HC,HD,HE,HF,HG,HH;HH=G5[
0];G6=G4[0]*HH;G7=G4[1]*HH;G8=G4[2]*HH;HH=G5[1];G9=G4[3]*HH;G_=G4[4]*HH;HA=G4[5]
*HH;HH=G5[2];HB=G4[6]*HH;HC=G4[7]*HH;HD=G4[8]*HH;HH=G5[3];HE=G4[9]*HH;HF=G4[10]*
HH;HG=G4[11]*HH;HH=G5[4];G6+=G4[12]*HH;G7+=G4[13]*HH;G8+=G4[14]*HH;HH=G5[5];G9+=
G4[15]*HH;G_+=G4[16]*HH;HA+=G4[17]*HH;HH=G5[6];HB+=G4[18]*HH;HC+=G4[19]*HH;HD+=
G4[20]*HH;HH=G5[7];HE+=G4[21]*HH;HF+=G4[22]*HH;HG+=G4[23]*HH;HH=G5[8];G6+=G4[24]
*HH;G7+=G4[25]*HH;G8+=G4[26]*HH;HH=G5[9];G9+=G4[27]*HH;G_+=G4[28]*HH;HA+=G4[29]*
HH;G3[0]=(G6+HB)+(G9+HE);G3[1]=(G7+HC)+(G_+HF);G3[2]=(G8+HD)+(HA+HG);G0+=G1;++Gz
;G3+=3;}while(G3<G2);}static void E5(float*Gw,unsigned int Gx,float const*Gy,B4
const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*3;float*restrict G3=Gw;do{
float const*G4=Gy+Gz->A*3;float const*G5=G0;float G6,G7,G8,G9,G_,HA,HB,HC,HD,HE,
HF,HG,HH;HH=G5[0];G6=G4[0]*HH;G7=G4[1]*HH;G8=G4[2]*HH;HH=G5[1];G9=G4[3]*HH;G_=G4
[4]*HH;HA=G4[5]*HH;HH=G5[2];HB=G4[6]*HH;HC=G4[7]*HH;HD=G4[8]*HH;HH=G5[3];HE=G4[9
]*HH;HF=G4[10]*HH;HG=G4[11]*HH;HH=G5[4];G6+=G4[12]*HH;G7+=G4[13]*HH;G8+=G4[14]*
HH;HH=G5[5];G9+=G4[15]*HH;G_+=G4[16]*HH;HA+=G4[17]*HH;HH=G5[6];HB+=G4[18]*HH;HC
+=G4[19]*HH;HD+=G4[20]*HH;HH=G5[7];HE+=G4[21]*HH;HF+=G4[22]*HH;HG+=G4[23]*HH;HH=
G5[8];G6+=G4[24]*HH;G7+=G4[25]*HH;G8+=G4[26]*HH;HH=G5[9];G9+=G4[27]*HH;G_+=G4[28
]*HH;HA+=G4[29]*HH;HH=G5[10];HB+=G4[30]*HH;HC+=G4[31]*HH;HD+=G4[32]*HH;G3[0]=(G6
+HB)+(G9+HE);G3[1]=(G7+HC)+(G_+HF);G3[2]=(G8+HD)+(HA+HG);G0+=G1;++Gz;G3+=3;}////
while(G3<G2);}static void E6(float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz
,float const*G0,int G1){float const*G2=Gw+Gx*3;float*restrict G3=Gw;do{float////
const*G4=Gy+Gz->A*3;float const*G5=G0;float G6,G7,G8,G9,G_,HA,HB,HC,HD,HE,HF,HG,
HH;HH=G5[0];G6=G4[0]*HH;G7=G4[1]*HH;G8=G4[2]*HH;HH=G5[1];G9=G4[3]*HH;G_=G4[4]*HH
;HA=G4[5]*HH;HH=G5[2];HB=G4[6]*HH;HC=G4[7]*HH;HD=G4[8]*HH;HH=G5[3];HE=G4[9]*HH;
HF=G4[10]*HH;HG=G4[11]*HH;HH=G5[4];G6+=G4[12]*HH;G7+=G4[13]*HH;G8+=G4[14]*HH;HH=
G5[5];G9+=G4[15]*HH;G_+=G4[16]*HH;HA+=G4[17]*HH;HH=G5[6];HB+=G4[18]*HH;HC+=G4[19
]*HH;HD+=G4[20]*HH;HH=G5[7];HE+=G4[21]*HH;HF+=G4[22]*HH;HG+=G4[23]*HH;HH=G5[8];
G6+=G4[24]*HH;G7+=G4[25]*HH;G8+=G4[26]*HH;HH=G5[9];G9+=G4[27]*HH;G_+=G4[28]*HH;
HA+=G4[29]*HH;HH=G5[10];HB+=G4[30]*HH;HC+=G4[31]*HH;HD+=G4[32]*HH;HH=G5[11];HE+=
G4[33]*HH;HF+=G4[34]*HH;HG+=G4[35]*HH;G3[0]=(G6+HB)+(G9+HE);G3[1]=(G7+HC)+(G_+HF
);G3[2]=(G8+HD)+(HA+HG);G0+=G1;++Gz;G3+=3;}while(G3<G2);}static void E7(float*Gw
,unsigned int Gx,float const*Gy,B4 const*Gz,float const*G0,int G1){float const*
G2=Gw+Gx*3;float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*3;int G5=((Gz->B-Gz->
A+1)-4+3)>>2;float const*G6=G0;float G7,G8,G9,G_,HA,HB,HC,HD,HE,HF,HG,HH,HI;HI=
G6[0];G7=G4[0]*HI;G8=G4[1]*HI;G9=G4[2]*HI;HI=G6[1];G_=G4[3]*HI;HA=G4[4]*HI;HB=G4
[5]*HI;HI=G6[2];HC=G4[6]*HI;HD=G4[7]*HI;HE=G4[8]*HI;HI=G6[3];HF=G4[9]*HI;HG=G4[
10]*HI;HH=G4[11]*HI;do{G6+=4;G4+=12;HI=G6[0];G7+=G4[0]*HI;G8+=G4[1]*HI;G9+=G4[2]
*HI;HI=G6[1];G_+=G4[3]*HI;HA+=G4[4]*HI;HB+=G4[5]*HI;HI=G6[2];HC+=G4[6]*HI;HD+=G4
[7]*HI;HE+=G4[8]*HI;HI=G6[3];HF+=G4[9]*HI;HG+=G4[10]*HI;HH+=G4[11]*HI;--G5;}////
while(G5>0);G3[0]=(G7+HC)+(G_+HF);G3[1]=(G8+HD)+(HA+HG);G3[2]=(G9+HE)+(HB+HH);G0
+=G1;++Gz;G3+=3;}while(G3<G2);}static void E8(float*Gw,unsigned int Gx,float////
const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*3;float*///////
restrict G3=Gw;do{float const*G4=Gy+Gz->A*3;int G5=((Gz->B-Gz->A+1)-5+3)>>2;////
float const*G6=G0;float G7,G8,G9,G_,HA,HB,HC,HD,HE,HF,HG,HH,HI;HI=G6[0];G7=G4[0]
*HI;G8=G4[1]*HI;G9=G4[2]*HI;HI=G6[1];G_=G4[3]*HI;HA=G4[4]*HI;HB=G4[5]*HI;HI=G6[2
];HC=G4[6]*HI;HD=G4[7]*HI;HE=G4[8]*HI;HI=G6[3];HF=G4[9]*HI;HG=G4[10]*HI;HH=G4[11
]*HI;do{G6+=4;G4+=12;HI=G6[0];G7+=G4[0]*HI;G8+=G4[1]*HI;G9+=G4[2]*HI;HI=G6[1];G_
+=G4[3]*HI;HA+=G4[4]*HI;HB+=G4[5]*HI;HI=G6[2];HC+=G4[6]*HI;HD+=G4[7]*HI;HE+=G4[8
]*HI;HI=G6[3];HF+=G4[9]*HI;HG+=G4[10]*HI;HH+=G4[11]*HI;--G5;}while(G5>0);HI=G6[4
];G7+=G4[12]*HI;G8+=G4[13]*HI;G9+=G4[14]*HI;G3[0]=(G7+HC)+(G_+HF);G3[1]=(G8+HD)+
(HA+HG);G3[2]=(G9+HE)+(HB+HH);G0+=G1;++Gz;G3+=3;}while(G3<G2);}static void E9(//
float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz,float const*G0,int G1){float
const*G2=Gw+Gx*3;float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*3;int G5=((Gz->
B-Gz->A+1)-6+3)>>2;float const*G6=G0;float G7,G8,G9,G_,HA,HB,HC,HD,HE,HF,HG,HH,
HI;HI=G6[0];G7=G4[0]*HI;G8=G4[1]*HI;G9=G4[2]*HI;HI=G6[1];G_=G4[3]*HI;HA=G4[4]*HI
;HB=G4[5]*HI;HI=G6[2];HC=G4[6]*HI;HD=G4[7]*HI;HE=G4[8]*HI;HI=G6[3];HF=G4[9]*HI;
HG=G4[10]*HI;HH=G4[11]*HI;do{G6+=4;G4+=12;HI=G6[0];G7+=G4[0]*HI;G8+=G4[1]*HI;G9
+=G4[2]*HI;HI=G6[1];G_+=G4[3]*HI;HA+=G4[4]*HI;HB+=G4[5]*HI;HI=G6[2];HC+=G4[6]*HI
;HD+=G4[7]*HI;HE+=G4[8]*HI;HI=G6[3];HF+=G4[9]*HI;HG+=G4[10]*HI;HH+=G4[11]*HI;--
G5;}while(G5>0);HI=G6[4];G7+=G4[12]*HI;G8+=G4[13]*HI;G9+=G4[14]*HI;HI=G6[5];G_+=
G4[15]*HI;HA+=G4[16]*HI;HB+=G4[17]*HI;G3[0]=(G7+HC)+(G_+HF);G3[1]=(G8+HD)+(HA+HG
);G3[2]=(G9+HE)+(HB+HH);G0+=G1;++Gz;G3+=3;}while(G3<G2);}static void E_(float*Gw
,unsigned int Gx,float const*Gy,B4 const*Gz,float const*G0,int G1){float const*
G2=Gw+Gx*3;float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*3;int G5=((Gz->B-Gz->
A+1)-7+3)>>2;float const*G6=G0;float G7,G8,G9,G_,HA,HB,HC,HD,HE,HF,HG,HH,HI;HI=
G6[0];G7=G4[0]*HI;G8=G4[1]*HI;G9=G4[2]*HI;HI=G6[1];G_=G4[3]*HI;HA=G4[4]*HI;HB=G4
[5]*HI;HI=G6[2];HC=G4[6]*HI;HD=G4[7]*HI;HE=G4[8]*HI;HI=G6[3];HF=G4[9]*HI;HG=G4[
10]*HI;HH=G4[11]*HI;do{G6+=4;G4+=12;HI=G6[0];G7+=G4[0]*HI;G8+=G4[1]*HI;G9+=G4[2]
*HI;HI=G6[1];G_+=G4[3]*HI;HA+=G4[4]*HI;HB+=G4[5]*HI;HI=G6[2];HC+=G4[6]*HI;HD+=G4
[7]*HI;HE+=G4[8]*HI;HI=G6[3];HF+=G4[9]*HI;HG+=G4[10]*HI;HH+=G4[11]*HI;--G5;}////
while(G5>0);HI=G6[4];G7+=G4[12]*HI;G8+=G4[13]*HI;G9+=G4[14]*HI;HI=G6[5];G_+=G4[
15]*HI;HA+=G4[16]*HI;HB+=G4[17]*HI;HI=G6[6];HC+=G4[18]*HI;HD+=G4[19]*HI;HE+=G4[
20]*HI;G3[0]=(G7+HC)+(G_+HF);G3[1]=(G8+HD)+(HA+HG);G3[2]=(G9+HE)+(HB+HH);G0+=G1;
++Gz;G3+=3;}while(G3<G2);}static CC*FA[4]={E7,E8,E9,E_,};static CC*FB[12]={Ev,Ew
,Ex,Ey,Ez,E0,E1,E2,E3,E4,E5,E6,};static void FC(float*Gw,unsigned int Gx,float//
const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*4;float*///////
restrict G3=Gw;do{float const*G4=Gy+Gz->A*4;float const*G5=G0;float G6,G7,G8,G9,
G_;G_=G5[0];G6=G4[0]*G_;G7=G4[1]*G_;G8=G4[2]*G_;G9=G4[3]*G_;G3[0]=G6;G3[1]=G7;G3
[2]=G8;G3[3]=G9;G0+=G1;++Gz;G3+=4;}while(G3<G2);}static void FD(float*Gw,///////
unsigned int Gx,float const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2
=Gw+Gx*4;float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*4;float const*G5=G0;///
float G6,G7,G8,G9,G_;G_=G5[0];G6=G4[0]*G_;G7=G4[1]*G_;G8=G4[2]*G_;G9=G4[3]*G_;G_
=G5[1];G6+=G4[4]*G_;G7+=G4[5]*G_;G8+=G4[6]*G_;G9+=G4[7]*G_;G3[0]=G6;G3[1]=G7;G3[
2]=G8;G3[3]=G9;G0+=G1;++Gz;G3+=4;}while(G3<G2);}static void FE(float*Gw,unsigned
int Gx,float const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*4;
float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*4;float const*G5=G0;float G6,G7,
G8,G9,G_;G_=G5[0];G6=G4[0]*G_;G7=G4[1]*G_;G8=G4[2]*G_;G9=G4[3]*G_;G_=G5[1];G6+=
G4[4]*G_;G7+=G4[5]*G_;G8+=G4[6]*G_;G9+=G4[7]*G_;G_=G5[2];G6+=G4[8]*G_;G7+=G4[9]*
G_;G8+=G4[10]*G_;G9+=G4[11]*G_;G3[0]=G6;G3[1]=G7;G3[2]=G8;G3[3]=G9;G0+=G1;++Gz;
G3+=4;}while(G3<G2);}static void FF(float*Gw,unsigned int Gx,float const*Gy,B4//
const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*4;float*restrict G3=Gw;do{
float const*G4=Gy+Gz->A*4;float const*G5=G0;float G6,G7,G8,G9,G_,HA,HB,HC,HD;HD=
G5[0];G6=G4[0]*HD;G7=G4[1]*HD;G8=G4[2]*HD;G9=G4[3]*HD;HD=G5[1];G_=G4[4]*HD;HA=G4
[5]*HD;HB=G4[6]*HD;HC=G4[7]*HD;HD=G5[2];G6+=G4[8]*HD;G7+=G4[9]*HD;G8+=G4[10]*HD;
G9+=G4[11]*HD;HD=G5[3];G_+=G4[12]*HD;HA+=G4[13]*HD;HB+=G4[14]*HD;HC+=G4[15]*HD;
G3[0]=G6+G_;G3[1]=G7+HA;G3[2]=G8+HB;G3[3]=G9+HC;G0+=G1;++Gz;G3+=4;}while(G3<G2);
}static void FG(float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz,float const*
G0,int G1){float const*G2=Gw+Gx*4;float*restrict G3=Gw;do{float const*G4=Gy+Gz->
A*4;float const*G5=G0;float G6,G7,G8,G9,G_,HA,HB,HC,HD;HD=G5[0];G6=G4[0]*HD;G7=
G4[1]*HD;G8=G4[2]*HD;G9=G4[3]*HD;HD=G5[1];G_=G4[4]*HD;HA=G4[5]*HD;HB=G4[6]*HD;HC
=G4[7]*HD;HD=G5[2];G6+=G4[8]*HD;G7+=G4[9]*HD;G8+=G4[10]*HD;G9+=G4[11]*HD;HD=G5[3
];G_+=G4[12]*HD;HA+=G4[13]*HD;HB+=G4[14]*HD;HC+=G4[15]*HD;HD=G5[4];G6+=G4[16]*HD
;G7+=G4[17]*HD;G8+=G4[18]*HD;G9+=G4[19]*HD;G3[0]=G6+G_;G3[1]=G7+HA;G3[2]=G8+HB;
G3[3]=G9+HC;G0+=G1;++Gz;G3+=4;}while(G3<G2);}static void FH(float*Gw,unsigned///
int Gx,float const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*4;
float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*4;float const*G5=G0;float G6,G7,
G8,G9,G_,HA,HB,HC,HD;HD=G5[0];G6=G4[0]*HD;G7=G4[1]*HD;G8=G4[2]*HD;G9=G4[3]*HD;HD
=G5[1];G_=G4[4]*HD;HA=G4[5]*HD;HB=G4[6]*HD;HC=G4[7]*HD;HD=G5[2];G6+=G4[8]*HD;G7
+=G4[9]*HD;G8+=G4[10]*HD;G9+=G4[11]*HD;HD=G5[3];G_+=G4[12]*HD;HA+=G4[13]*HD;HB+=
G4[14]*HD;HC+=G4[15]*HD;HD=G5[4];G6+=G4[16]*HD;G7+=G4[17]*HD;G8+=G4[18]*HD;G9+=
G4[19]*HD;HD=G5[5];G_+=G4[20]*HD;HA+=G4[21]*HD;HB+=G4[22]*HD;HC+=G4[23]*HD;G3[0]
=G6+G_;G3[1]=G7+HA;G3[2]=G8+HB;G3[3]=G9+HC;G0+=G1;++Gz;G3+=4;}while(G3<G2);}////
static void FI(float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz,float const*
G0,int G1){float const*G2=Gw+Gx*4;float*restrict G3=Gw;do{float const*G4=Gy+Gz->
A*4;float const*G5=G0;float G6,G7,G8,G9,G_,HA,HB,HC,HD;HD=G5[0];G6=G4[0]*HD;G7=
G4[1]*HD;G8=G4[2]*HD;G9=G4[3]*HD;HD=G5[1];G_=G4[4]*HD;HA=G4[5]*HD;HB=G4[6]*HD;HC
=G4[7]*HD;HD=G5[2];G6+=G4[8]*HD;G7+=G4[9]*HD;G8+=G4[10]*HD;G9+=G4[11]*HD;HD=G5[3
];G_+=G4[12]*HD;HA+=G4[13]*HD;HB+=G4[14]*HD;HC+=G4[15]*HD;HD=G5[4];G6+=G4[16]*HD
;G7+=G4[17]*HD;G8+=G4[18]*HD;G9+=G4[19]*HD;HD=G5[5];G_+=G4[20]*HD;HA+=G4[21]*HD;
HB+=G4[22]*HD;HC+=G4[23]*HD;HD=G5[6];G6+=G4[24]*HD;G7+=G4[25]*HD;G8+=G4[26]*HD;
G9+=G4[27]*HD;G3[0]=G6+G_;G3[1]=G7+HA;G3[2]=G8+HB;G3[3]=G9+HC;G0+=G1;++Gz;G3+=4;
}while(G3<G2);}static void FJ(float*Gw,unsigned int Gx,float const*Gy,B4 const*
Gz,float const*G0,int G1){float const*G2=Gw+Gx*4;float*restrict G3=Gw;do{float//
const*G4=Gy+Gz->A*4;float const*G5=G0;float G6,G7,G8,G9,G_,HA,HB,HC,HD;HD=G5[0];
G6=G4[0]*HD;G7=G4[1]*HD;G8=G4[2]*HD;G9=G4[3]*HD;HD=G5[1];G_=G4[4]*HD;HA=G4[5]*HD
;HB=G4[6]*HD;HC=G4[7]*HD;HD=G5[2];G6+=G4[8]*HD;G7+=G4[9]*HD;G8+=G4[10]*HD;G9+=G4
[11]*HD;HD=G5[3];G_+=G4[12]*HD;HA+=G4[13]*HD;HB+=G4[14]*HD;HC+=G4[15]*HD;HD=G5[4
];G6+=G4[16]*HD;G7+=G4[17]*HD;G8+=G4[18]*HD;G9+=G4[19]*HD;HD=G5[5];G_+=G4[20]*HD
;HA+=G4[21]*HD;HB+=G4[22]*HD;HC+=G4[23]*HD;HD=G5[6];G6+=G4[24]*HD;G7+=G4[25]*HD;
G8+=G4[26]*HD;G9+=G4[27]*HD;HD=G5[7];G_+=G4[28]*HD;HA+=G4[29]*HD;HB+=G4[30]*HD;
HC+=G4[31]*HD;G3[0]=G6+G_;G3[1]=G7+HA;G3[2]=G8+HB;G3[3]=G9+HC;G0+=G1;++Gz;G3+=4;
}while(G3<G2);}static void FK(float*Gw,unsigned int Gx,float const*Gy,B4 const*
Gz,float const*G0,int G1){float const*G2=Gw+Gx*4;float*restrict G3=Gw;do{float//
const*G4=Gy+Gz->A*4;float const*G5=G0;float G6,G7,G8,G9,G_,HA,HB,HC,HD;HD=G5[0];
G6=G4[0]*HD;G7=G4[1]*HD;G8=G4[2]*HD;G9=G4[3]*HD;HD=G5[1];G_=G4[4]*HD;HA=G4[5]*HD
;HB=G4[6]*HD;HC=G4[7]*HD;HD=G5[2];G6+=G4[8]*HD;G7+=G4[9]*HD;G8+=G4[10]*HD;G9+=G4
[11]*HD;HD=G5[3];G_+=G4[12]*HD;HA+=G4[13]*HD;HB+=G4[14]*HD;HC+=G4[15]*HD;HD=G5[4
];G6+=G4[16]*HD;G7+=G4[17]*HD;G8+=G4[18]*HD;G9+=G4[19]*HD;HD=G5[5];G_+=G4[20]*HD
;HA+=G4[21]*HD;HB+=G4[22]*HD;HC+=G4[23]*HD;HD=G5[6];G6+=G4[24]*HD;G7+=G4[25]*HD;
G8+=G4[26]*HD;G9+=G4[27]*HD;HD=G5[7];G_+=G4[28]*HD;HA+=G4[29]*HD;HB+=G4[30]*HD;
HC+=G4[31]*HD;HD=G5[8];G6+=G4[32]*HD;G7+=G4[33]*HD;G8+=G4[34]*HD;G9+=G4[35]*HD;
G3[0]=G6+G_;G3[1]=G7+HA;G3[2]=G8+HB;G3[3]=G9+HC;G0+=G1;++Gz;G3+=4;}while(G3<G2);
}static void FL(float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz,float const*
G0,int G1){float const*G2=Gw+Gx*4;float*restrict G3=Gw;do{float const*G4=Gy+Gz->
A*4;float const*G5=G0;float G6,G7,G8,G9,G_,HA,HB,HC,HD;HD=G5[0];G6=G4[0]*HD;G7=
G4[1]*HD;G8=G4[2]*HD;G9=G4[3]*HD;HD=G5[1];G_=G4[4]*HD;HA=G4[5]*HD;HB=G4[6]*HD;HC
=G4[7]*HD;HD=G5[2];G6+=G4[8]*HD;G7+=G4[9]*HD;G8+=G4[10]*HD;G9+=G4[11]*HD;HD=G5[3
];G_+=G4[12]*HD;HA+=G4[13]*HD;HB+=G4[14]*HD;HC+=G4[15]*HD;HD=G5[4];G6+=G4[16]*HD
;G7+=G4[17]*HD;G8+=G4[18]*HD;G9+=G4[19]*HD;HD=G5[5];G_+=G4[20]*HD;HA+=G4[21]*HD;
HB+=G4[22]*HD;HC+=G4[23]*HD;HD=G5[6];G6+=G4[24]*HD;G7+=G4[25]*HD;G8+=G4[26]*HD;
G9+=G4[27]*HD;HD=G5[7];G_+=G4[28]*HD;HA+=G4[29]*HD;HB+=G4[30]*HD;HC+=G4[31]*HD;
HD=G5[8];G6+=G4[32]*HD;G7+=G4[33]*HD;G8+=G4[34]*HD;G9+=G4[35]*HD;HD=G5[9];G_+=G4
[36]*HD;HA+=G4[37]*HD;HB+=G4[38]*HD;HC+=G4[39]*HD;G3[0]=G6+G_;G3[1]=G7+HA;G3[2]=
G8+HB;G3[3]=G9+HC;G0+=G1;++Gz;G3+=4;}while(G3<G2);}static void FM(float*Gw,/////
unsigned int Gx,float const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2
=Gw+Gx*4;float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*4;float const*G5=G0;///
float G6,G7,G8,G9,G_,HA,HB,HC,HD;HD=G5[0];G6=G4[0]*HD;G7=G4[1]*HD;G8=G4[2]*HD;G9
=G4[3]*HD;HD=G5[1];G_=G4[4]*HD;HA=G4[5]*HD;HB=G4[6]*HD;HC=G4[7]*HD;HD=G5[2];G6+=
G4[8]*HD;G7+=G4[9]*HD;G8+=G4[10]*HD;G9+=G4[11]*HD;HD=G5[3];G_+=G4[12]*HD;HA+=G4[
13]*HD;HB+=G4[14]*HD;HC+=G4[15]*HD;HD=G5[4];G6+=G4[16]*HD;G7+=G4[17]*HD;G8+=G4[
18]*HD;G9+=G4[19]*HD;HD=G5[5];G_+=G4[20]*HD;HA+=G4[21]*HD;HB+=G4[22]*HD;HC+=G4[
23]*HD;HD=G5[6];G6+=G4[24]*HD;G7+=G4[25]*HD;G8+=G4[26]*HD;G9+=G4[27]*HD;HD=G5[7]
;G_+=G4[28]*HD;HA+=G4[29]*HD;HB+=G4[30]*HD;HC+=G4[31]*HD;HD=G5[8];G6+=G4[32]*HD;
G7+=G4[33]*HD;G8+=G4[34]*HD;G9+=G4[35]*HD;HD=G5[9];G_+=G4[36]*HD;HA+=G4[37]*HD;
HB+=G4[38]*HD;HC+=G4[39]*HD;HD=G5[10];G6+=G4[40]*HD;G7+=G4[41]*HD;G8+=G4[42]*HD;
G9+=G4[43]*HD;G3[0]=G6+G_;G3[1]=G7+HA;G3[2]=G8+HB;G3[3]=G9+HC;G0+=G1;++Gz;G3+=4;
}while(G3<G2);}static void FN(float*Gw,unsigned int Gx,float const*Gy,B4 const*
Gz,float const*G0,int G1){float const*G2=Gw+Gx*4;float*restrict G3=Gw;do{float//
const*G4=Gy+Gz->A*4;float const*G5=G0;float G6,G7,G8,G9,G_,HA,HB,HC,HD;HD=G5[0];
G6=G4[0]*HD;G7=G4[1]*HD;G8=G4[2]*HD;G9=G4[3]*HD;HD=G5[1];G_=G4[4]*HD;HA=G4[5]*HD
;HB=G4[6]*HD;HC=G4[7]*HD;HD=G5[2];G6+=G4[8]*HD;G7+=G4[9]*HD;G8+=G4[10]*HD;G9+=G4
[11]*HD;HD=G5[3];G_+=G4[12]*HD;HA+=G4[13]*HD;HB+=G4[14]*HD;HC+=G4[15]*HD;HD=G5[4
];G6+=G4[16]*HD;G7+=G4[17]*HD;G8+=G4[18]*HD;G9+=G4[19]*HD;HD=G5[5];G_+=G4[20]*HD
;HA+=G4[21]*HD;HB+=G4[22]*HD;HC+=G4[23]*HD;HD=G5[6];G6+=G4[24]*HD;G7+=G4[25]*HD;
G8+=G4[26]*HD;G9+=G4[27]*HD;HD=G5[7];G_+=G4[28]*HD;HA+=G4[29]*HD;HB+=G4[30]*HD;
HC+=G4[31]*HD;HD=G5[8];G6+=G4[32]*HD;G7+=G4[33]*HD;G8+=G4[34]*HD;G9+=G4[35]*HD;
HD=G5[9];G_+=G4[36]*HD;HA+=G4[37]*HD;HB+=G4[38]*HD;HC+=G4[39]*HD;HD=G5[10];G6+=
G4[40]*HD;G7+=G4[41]*HD;G8+=G4[42]*HD;G9+=G4[43]*HD;HD=G5[11];G_+=G4[44]*HD;HA+=
G4[45]*HD;HB+=G4[46]*HD;HC+=G4[47]*HD;G3[0]=G6+G_;G3[1]=G7+HA;G3[2]=G8+HB;G3[3]=
G9+HC;G0+=G1;++Gz;G3+=4;}while(G3<G2);}static void FO(float*Gw,unsigned int Gx,
float const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*4;float*
restrict G3=Gw;do{float const*G4=Gy+Gz->A*4;int G5=((Gz->B-Gz->A+1)-4+3)>>2;////
float const*G6=G0;float G7,G8,G9,G_,HA,HB,HC,HD,HE;HE=G6[0];G7=G4[0]*HE;G8=G4[1]
*HE;G9=G4[2]*HE;G_=G4[3]*HE;HE=G6[1];HA=G4[4]*HE;HB=G4[5]*HE;HC=G4[6]*HE;HD=G4[7
]*HE;HE=G6[2];G7+=G4[8]*HE;G8+=G4[9]*HE;G9+=G4[10]*HE;G_+=G4[11]*HE;HE=G6[3];HA
+=G4[12]*HE;HB+=G4[13]*HE;HC+=G4[14]*HE;HD+=G4[15]*HE;do{G6+=4;G4+=16;HE=G6[0];
G7+=G4[0]*HE;G8+=G4[1]*HE;G9+=G4[2]*HE;G_+=G4[3]*HE;HE=G6[1];HA+=G4[4]*HE;HB+=G4
[5]*HE;HC+=G4[6]*HE;HD+=G4[7]*HE;HE=G6[2];G7+=G4[8]*HE;G8+=G4[9]*HE;G9+=G4[10]*
HE;G_+=G4[11]*HE;HE=G6[3];HA+=G4[12]*HE;HB+=G4[13]*HE;HC+=G4[14]*HE;HD+=G4[15]*
HE;--G5;}while(G5>0);G3[0]=G7+HA;G3[1]=G8+HB;G3[2]=G9+HC;G3[3]=G_+HD;G0+=G1;++Gz
;G3+=4;}while(G3<G2);}static void FP(float*Gw,unsigned int Gx,float const*Gy,B4
const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*4;float*restrict G3=Gw;do{
float const*G4=Gy+Gz->A*4;int G5=((Gz->B-Gz->A+1)-5+3)>>2;float const*G6=G0;////
float G7,G8,G9,G_,HA,HB,HC,HD,HE;HE=G6[0];G7=G4[0]*HE;G8=G4[1]*HE;G9=G4[2]*HE;G_
=G4[3]*HE;HE=G6[1];HA=G4[4]*HE;HB=G4[5]*HE;HC=G4[6]*HE;HD=G4[7]*HE;HE=G6[2];G7+=
G4[8]*HE;G8+=G4[9]*HE;G9+=G4[10]*HE;G_+=G4[11]*HE;HE=G6[3];HA+=G4[12]*HE;HB+=G4[
13]*HE;HC+=G4[14]*HE;HD+=G4[15]*HE;do{G6+=4;G4+=16;HE=G6[0];G7+=G4[0]*HE;G8+=G4[
1]*HE;G9+=G4[2]*HE;G_+=G4[3]*HE;HE=G6[1];HA+=G4[4]*HE;HB+=G4[5]*HE;HC+=G4[6]*HE;
HD+=G4[7]*HE;HE=G6[2];G7+=G4[8]*HE;G8+=G4[9]*HE;G9+=G4[10]*HE;G_+=G4[11]*HE;HE=
G6[3];HA+=G4[12]*HE;HB+=G4[13]*HE;HC+=G4[14]*HE;HD+=G4[15]*HE;--G5;}while(G5>0);
HE=G6[4];G7+=G4[16]*HE;G8+=G4[17]*HE;G9+=G4[18]*HE;G_+=G4[19]*HE;G3[0]=G7+HA;G3[
1]=G8+HB;G3[2]=G9+HC;G3[3]=G_+HD;G0+=G1;++Gz;G3+=4;}while(G3<G2);}static void FQ
(float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz,float const*G0,int G1){////
float const*G2=Gw+Gx*4;float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*4;int G5=
((Gz->B-Gz->A+1)-6+3)>>2;float const*G6=G0;float G7,G8,G9,G_,HA,HB,HC,HD,HE;HE=
G6[0];G7=G4[0]*HE;G8=G4[1]*HE;G9=G4[2]*HE;G_=G4[3]*HE;HE=G6[1];HA=G4[4]*HE;HB=G4
[5]*HE;HC=G4[6]*HE;HD=G4[7]*HE;HE=G6[2];G7+=G4[8]*HE;G8+=G4[9]*HE;G9+=G4[10]*HE;
G_+=G4[11]*HE;HE=G6[3];HA+=G4[12]*HE;HB+=G4[13]*HE;HC+=G4[14]*HE;HD+=G4[15]*HE;
do{G6+=4;G4+=16;HE=G6[0];G7+=G4[0]*HE;G8+=G4[1]*HE;G9+=G4[2]*HE;G_+=G4[3]*HE;HE=
G6[1];HA+=G4[4]*HE;HB+=G4[5]*HE;HC+=G4[6]*HE;HD+=G4[7]*HE;HE=G6[2];G7+=G4[8]*HE;
G8+=G4[9]*HE;G9+=G4[10]*HE;G_+=G4[11]*HE;HE=G6[3];HA+=G4[12]*HE;HB+=G4[13]*HE;HC
+=G4[14]*HE;HD+=G4[15]*HE;--G5;}while(G5>0);HE=G6[4];G7+=G4[16]*HE;G8+=G4[17]*HE
;G9+=G4[18]*HE;G_+=G4[19]*HE;HE=G6[5];HA+=G4[20]*HE;HB+=G4[21]*HE;HC+=G4[22]*HE;
HD+=G4[23]*HE;G3[0]=G7+HA;G3[1]=G8+HB;G3[2]=G9+HC;G3[3]=G_+HD;G0+=G1;++Gz;G3+=4;
}while(G3<G2);}static void FR(float*Gw,unsigned int Gx,float const*Gy,B4 const*
Gz,float const*G0,int G1){float const*G2=Gw+Gx*4;float*restrict G3=Gw;do{float//
const*G4=Gy+Gz->A*4;int G5=((Gz->B-Gz->A+1)-7+3)>>2;float const*G6=G0;float G7,
G8,G9,G_,HA,HB,HC,HD,HE;HE=G6[0];G7=G4[0]*HE;G8=G4[1]*HE;G9=G4[2]*HE;G_=G4[3]*HE
;HE=G6[1];HA=G4[4]*HE;HB=G4[5]*HE;HC=G4[6]*HE;HD=G4[7]*HE;HE=G6[2];G7+=G4[8]*HE;
G8+=G4[9]*HE;G9+=G4[10]*HE;G_+=G4[11]*HE;HE=G6[3];HA+=G4[12]*HE;HB+=G4[13]*HE;HC
+=G4[14]*HE;HD+=G4[15]*HE;do{G6+=4;G4+=16;HE=G6[0];G7+=G4[0]*HE;G8+=G4[1]*HE;G9
+=G4[2]*HE;G_+=G4[3]*HE;HE=G6[1];HA+=G4[4]*HE;HB+=G4[5]*HE;HC+=G4[6]*HE;HD+=G4[7
]*HE;HE=G6[2];G7+=G4[8]*HE;G8+=G4[9]*HE;G9+=G4[10]*HE;G_+=G4[11]*HE;HE=G6[3];HA
+=G4[12]*HE;HB+=G4[13]*HE;HC+=G4[14]*HE;HD+=G4[15]*HE;--G5;}while(G5>0);HE=G6[4]
;G7+=G4[16]*HE;G8+=G4[17]*HE;G9+=G4[18]*HE;G_+=G4[19]*HE;HE=G6[5];HA+=G4[20]*HE;
HB+=G4[21]*HE;HC+=G4[22]*HE;HD+=G4[23]*HE;HE=G6[6];G7+=G4[24]*HE;G8+=G4[25]*HE;
G9+=G4[26]*HE;G_+=G4[27]*HE;G3[0]=G7+HA;G3[1]=G8+HB;G3[2]=G9+HC;G3[3]=G_+HD;G0+=
G1;++Gz;G3+=4;}while(G3<G2);}static CC*FS[4]={FO,FP,FQ,FR,};static CC*FT[12]={FC
,FD,FE,FF,FG,FH,FI,FJ,FK,FL,FM,FN,};static void FU(float*Gw,unsigned int Gx,////
float const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*7;float*
restrict G3=Gw;do{float const*G4=Gy+Gz->A*7;float const*G5=G0;float G6,G7,G8,G9,
G_,HA,HB,HC;HC=G5[0];G6=G4[0]*HC;G7=G4[1]*HC;G8=G4[2]*HC;G9=G4[3]*HC;G_=G4[4]*HC
;HA=G4[5]*HC;HB=G4[6]*HC;G3[0]=G6;G3[1]=G7;G3[2]=G8;G3[3]=G9;G3[4]=G_;G3[5]=HA;
G3[6]=HB;G0+=G1;++Gz;G3+=7;}while(G3<G2);}static void FV(float*Gw,unsigned int//
Gx,float const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*7;////
float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*7;float const*G5=G0;float G6,G7,
G8,G9,G_,HA,HB,HC;HC=G5[0];G6=G4[0]*HC;G7=G4[1]*HC;G8=G4[2]*HC;G9=G4[3]*HC;G_=G4
[4]*HC;HA=G4[5]*HC;HB=G4[6]*HC;HC=G5[1];G6+=G4[7]*HC;G7+=G4[8]*HC;G8+=G4[9]*HC;
G9+=G4[10]*HC;G_+=G4[11]*HC;HA+=G4[12]*HC;HB+=G4[13]*HC;G3[0]=G6;G3[1]=G7;G3[2]=
G8;G3[3]=G9;G3[4]=G_;G3[5]=HA;G3[6]=HB;G0+=G1;++Gz;G3+=7;}while(G3<G2);}static//
void FW(float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz,float const*G0,int//
G1){float const*G2=Gw+Gx*7;float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*7;///
float const*G5=G0;float G6,G7,G8,G9,G_,HA,HB,HC;HC=G5[0];G6=G4[0]*HC;G7=G4[1]*HC
;G8=G4[2]*HC;G9=G4[3]*HC;G_=G4[4]*HC;HA=G4[5]*HC;HB=G4[6]*HC;HC=G5[1];G6+=G4[7]*
HC;G7+=G4[8]*HC;G8+=G4[9]*HC;G9+=G4[10]*HC;G_+=G4[11]*HC;HA+=G4[12]*HC;HB+=G4[13
]*HC;HC=G5[2];G6+=G4[14]*HC;G7+=G4[15]*HC;G8+=G4[16]*HC;G9+=G4[17]*HC;G_+=G4[18]
*HC;HA+=G4[19]*HC;HB+=G4[20]*HC;G3[0]=G6;G3[1]=G7;G3[2]=G8;G3[3]=G9;G3[4]=G_;G3[
5]=HA;G3[6]=HB;G0+=G1;++Gz;G3+=7;}while(G3<G2);}static void FX(float*Gw,unsigned
int Gx,float const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*7;
float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*7;float const*G5=G0;float G6,G7,
G8,G9,G_,HA,HB,HC,HD,HE,HF,HG,HH,HI,HJ;HJ=G5[0];G6=G4[0]*HJ;G7=G4[1]*HJ;G8=G4[2]
*HJ;G9=G4[3]*HJ;G_=G4[4]*HJ;HA=G4[5]*HJ;HB=G4[6]*HJ;HJ=G5[1];HC=G4[7]*HJ;HD=G4[8
]*HJ;HE=G4[9]*HJ;HF=G4[10]*HJ;HG=G4[11]*HJ;HH=G4[12]*HJ;HI=G4[13]*HJ;HJ=G5[2];G6
+=G4[14]*HJ;G7+=G4[15]*HJ;G8+=G4[16]*HJ;G9+=G4[17]*HJ;G_+=G4[18]*HJ;HA+=G4[19]*
HJ;HB+=G4[20]*HJ;HJ=G5[3];HC+=G4[21]*HJ;HD+=G4[22]*HJ;HE+=G4[23]*HJ;HF+=G4[24]*
HJ;HG+=G4[25]*HJ;HH+=G4[26]*HJ;HI+=G4[27]*HJ;G3[0]=G6+HC;G3[1]=G7+HD;G3[2]=G8+HE
;G3[3]=G9+HF;G3[4]=G_+HG;G3[5]=HA+HH;G3[6]=HB+HI;G0+=G1;++Gz;G3+=7;}while(G3<G2)
;}static void FY(float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz,float const
*G0,int G1){float const*G2=Gw+Gx*7;float*restrict G3=Gw;do{float const*G4=Gy+Gz
->A*7;float const*G5=G0;float G6,G7,G8,G9,G_,HA,HB,HC,HD,HE,HF,HG,HH,HI,HJ;HJ=G5
[0];G6=G4[0]*HJ;G7=G4[1]*HJ;G8=G4[2]*HJ;G9=G4[3]*HJ;G_=G4[4]*HJ;HA=G4[5]*HJ;HB=
G4[6]*HJ;HJ=G5[1];HC=G4[7]*HJ;HD=G4[8]*HJ;HE=G4[9]*HJ;HF=G4[10]*HJ;HG=G4[11]*HJ;
HH=G4[12]*HJ;HI=G4[13]*HJ;HJ=G5[2];G6+=G4[14]*HJ;G7+=G4[15]*HJ;G8+=G4[16]*HJ;G9
+=G4[17]*HJ;G_+=G4[18]*HJ;HA+=G4[19]*HJ;HB+=G4[20]*HJ;HJ=G5[3];HC+=G4[21]*HJ;HD
+=G4[22]*HJ;HE+=G4[23]*HJ;HF+=G4[24]*HJ;HG+=G4[25]*HJ;HH+=G4[26]*HJ;HI+=G4[27]*
HJ;HJ=G5[4];G6+=G4[28]*HJ;G7+=G4[29]*HJ;G8+=G4[30]*HJ;G9+=G4[31]*HJ;G_+=G4[32]*
HJ;HA+=G4[33]*HJ;HB+=G4[34]*HJ;G3[0]=G6+HC;G3[1]=G7+HD;G3[2]=G8+HE;G3[3]=G9+HF;
G3[4]=G_+HG;G3[5]=HA+HH;G3[6]=HB+HI;G0+=G1;++Gz;G3+=7;}while(G3<G2);}static void
FZ(float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz,float const*G0,int G1){//
float const*G2=Gw+Gx*7;float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*7;float//
const*G5=G0;float G6,G7,G8,G9,G_,HA,HB,HC,HD,HE,HF,HG,HH,HI,HJ;HJ=G5[0];G6=G4[0]
*HJ;G7=G4[1]*HJ;G8=G4[2]*HJ;G9=G4[3]*HJ;G_=G4[4]*HJ;HA=G4[5]*HJ;HB=G4[6]*HJ;HJ=
G5[1];HC=G4[7]*HJ;HD=G4[8]*HJ;HE=G4[9]*HJ;HF=G4[10]*HJ;HG=G4[11]*HJ;HH=G4[12]*HJ
;HI=G4[13]*HJ;HJ=G5[2];G6+=G4[14]*HJ;G7+=G4[15]*HJ;G8+=G4[16]*HJ;G9+=G4[17]*HJ;
G_+=G4[18]*HJ;HA+=G4[19]*HJ;HB+=G4[20]*HJ;HJ=G5[3];HC+=G4[21]*HJ;HD+=G4[22]*HJ;
HE+=G4[23]*HJ;HF+=G4[24]*HJ;HG+=G4[25]*HJ;HH+=G4[26]*HJ;HI+=G4[27]*HJ;HJ=G5[4];
G6+=G4[28]*HJ;G7+=G4[29]*HJ;G8+=G4[30]*HJ;G9+=G4[31]*HJ;G_+=G4[32]*HJ;HA+=G4[33]
*HJ;HB+=G4[34]*HJ;HJ=G5[5];HC+=G4[35]*HJ;HD+=G4[36]*HJ;HE+=G4[37]*HJ;HF+=G4[38]*
HJ;HG+=G4[39]*HJ;HH+=G4[40]*HJ;HI+=G4[41]*HJ;G3[0]=G6+HC;G3[1]=G7+HD;G3[2]=G8+HE
;G3[3]=G9+HF;G3[4]=G_+HG;G3[5]=HA+HH;G3[6]=HB+HI;G0+=G1;++Gz;G3+=7;}while(G3<G2)
;}static void Fa(float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz,float const
*G0,int G1){float const*G2=Gw+Gx*7;float*restrict G3=Gw;do{float const*G4=Gy+Gz
->A*7;float const*G5=G0;float G6,G7,G8,G9,G_,HA,HB,HC,HD,HE,HF,HG,HH,HI,HJ;HJ=G5
[0];G6=G4[0]*HJ;G7=G4[1]*HJ;G8=G4[2]*HJ;G9=G4[3]*HJ;G_=G4[4]*HJ;HA=G4[5]*HJ;HB=
G4[6]*HJ;HJ=G5[1];HC=G4[7]*HJ;HD=G4[8]*HJ;HE=G4[9]*HJ;HF=G4[10]*HJ;HG=G4[11]*HJ;
HH=G4[12]*HJ;HI=G4[13]*HJ;HJ=G5[2];G6+=G4[14]*HJ;G7+=G4[15]*HJ;G8+=G4[16]*HJ;G9
+=G4[17]*HJ;G_+=G4[18]*HJ;HA+=G4[19]*HJ;HB+=G4[20]*HJ;HJ=G5[3];HC+=G4[21]*HJ;HD
+=G4[22]*HJ;HE+=G4[23]*HJ;HF+=G4[24]*HJ;HG+=G4[25]*HJ;HH+=G4[26]*HJ;HI+=G4[27]*
HJ;HJ=G5[4];G6+=G4[28]*HJ;G7+=G4[29]*HJ;G8+=G4[30]*HJ;G9+=G4[31]*HJ;G_+=G4[32]*
HJ;HA+=G4[33]*HJ;HB+=G4[34]*HJ;HJ=G5[5];HC+=G4[35]*HJ;HD+=G4[36]*HJ;HE+=G4[37]*
HJ;HF+=G4[38]*HJ;HG+=G4[39]*HJ;HH+=G4[40]*HJ;HI+=G4[41]*HJ;HJ=G5[6];G6+=G4[42]*
HJ;G7+=G4[43]*HJ;G8+=G4[44]*HJ;G9+=G4[45]*HJ;G_+=G4[46]*HJ;HA+=G4[47]*HJ;HB+=G4[
48]*HJ;G3[0]=G6+HC;G3[1]=G7+HD;G3[2]=G8+HE;G3[3]=G9+HF;G3[4]=G_+HG;G3[5]=HA+HH;
G3[6]=HB+HI;G0+=G1;++Gz;G3+=7;}while(G3<G2);}static void Fb(float*Gw,unsigned///
int Gx,float const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*7;
float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*7;float const*G5=G0;float G6,G7,
G8,G9,G_,HA,HB,HC,HD,HE,HF,HG,HH,HI,HJ;HJ=G5[0];G6=G4[0]*HJ;G7=G4[1]*HJ;G8=G4[2]
*HJ;G9=G4[3]*HJ;G_=G4[4]*HJ;HA=G4[5]*HJ;HB=G4[6]*HJ;HJ=G5[1];HC=G4[7]*HJ;HD=G4[8
]*HJ;HE=G4[9]*HJ;HF=G4[10]*HJ;HG=G4[11]*HJ;HH=G4[12]*HJ;HI=G4[13]*HJ;HJ=G5[2];G6
+=G4[14]*HJ;G7+=G4[15]*HJ;G8+=G4[16]*HJ;G9+=G4[17]*HJ;G_+=G4[18]*HJ;HA+=G4[19]*
HJ;HB+=G4[20]*HJ;HJ=G5[3];HC+=G4[21]*HJ;HD+=G4[22]*HJ;HE+=G4[23]*HJ;HF+=G4[24]*
HJ;HG+=G4[25]*HJ;HH+=G4[26]*HJ;HI+=G4[27]*HJ;HJ=G5[4];G6+=G4[28]*HJ;G7+=G4[29]*
HJ;G8+=G4[30]*HJ;G9+=G4[31]*HJ;G_+=G4[32]*HJ;HA+=G4[33]*HJ;HB+=G4[34]*HJ;HJ=G5[5
];HC+=G4[35]*HJ;HD+=G4[36]*HJ;HE+=G4[37]*HJ;HF+=G4[38]*HJ;HG+=G4[39]*HJ;HH+=G4[
40]*HJ;HI+=G4[41]*HJ;HJ=G5[6];G6+=G4[42]*HJ;G7+=G4[43]*HJ;G8+=G4[44]*HJ;G9+=G4[
45]*HJ;G_+=G4[46]*HJ;HA+=G4[47]*HJ;HB+=G4[48]*HJ;HJ=G5[7];HC+=G4[49]*HJ;HD+=G4[
50]*HJ;HE+=G4[51]*HJ;HF+=G4[52]*HJ;HG+=G4[53]*HJ;HH+=G4[54]*HJ;HI+=G4[55]*HJ;G3[
0]=G6+HC;G3[1]=G7+HD;G3[2]=G8+HE;G3[3]=G9+HF;G3[4]=G_+HG;G3[5]=HA+HH;G3[6]=HB+HI
;G0+=G1;++Gz;G3+=7;}while(G3<G2);}static void Fc(float*Gw,unsigned int Gx,float
const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*7;float*///////
restrict G3=Gw;do{float const*G4=Gy+Gz->A*7;float const*G5=G0;float G6,G7,G8,G9,
G_,HA,HB,HC,HD,HE,HF,HG,HH,HI,HJ;HJ=G5[0];G6=G4[0]*HJ;G7=G4[1]*HJ;G8=G4[2]*HJ;G9
=G4[3]*HJ;G_=G4[4]*HJ;HA=G4[5]*HJ;HB=G4[6]*HJ;HJ=G5[1];HC=G4[7]*HJ;HD=G4[8]*HJ;
HE=G4[9]*HJ;HF=G4[10]*HJ;HG=G4[11]*HJ;HH=G4[12]*HJ;HI=G4[13]*HJ;HJ=G5[2];G6+=G4[
14]*HJ;G7+=G4[15]*HJ;G8+=G4[16]*HJ;G9+=G4[17]*HJ;G_+=G4[18]*HJ;HA+=G4[19]*HJ;HB
+=G4[20]*HJ;HJ=G5[3];HC+=G4[21]*HJ;HD+=G4[22]*HJ;HE+=G4[23]*HJ;HF+=G4[24]*HJ;HG
+=G4[25]*HJ;HH+=G4[26]*HJ;HI+=G4[27]*HJ;HJ=G5[4];G6+=G4[28]*HJ;G7+=G4[29]*HJ;G8
+=G4[30]*HJ;G9+=G4[31]*HJ;G_+=G4[32]*HJ;HA+=G4[33]*HJ;HB+=G4[34]*HJ;HJ=G5[5];HC
+=G4[35]*HJ;HD+=G4[36]*HJ;HE+=G4[37]*HJ;HF+=G4[38]*HJ;HG+=G4[39]*HJ;HH+=G4[40]*
HJ;HI+=G4[41]*HJ;HJ=G5[6];G6+=G4[42]*HJ;G7+=G4[43]*HJ;G8+=G4[44]*HJ;G9+=G4[45]*
HJ;G_+=G4[46]*HJ;HA+=G4[47]*HJ;HB+=G4[48]*HJ;HJ=G5[7];HC+=G4[49]*HJ;HD+=G4[50]*
HJ;HE+=G4[51]*HJ;HF+=G4[52]*HJ;HG+=G4[53]*HJ;HH+=G4[54]*HJ;HI+=G4[55]*HJ;HJ=G5[8
];G6+=G4[56]*HJ;G7+=G4[57]*HJ;G8+=G4[58]*HJ;G9+=G4[59]*HJ;G_+=G4[60]*HJ;HA+=G4[
61]*HJ;HB+=G4[62]*HJ;G3[0]=G6+HC;G3[1]=G7+HD;G3[2]=G8+HE;G3[3]=G9+HF;G3[4]=G_+HG
;G3[5]=HA+HH;G3[6]=HB+HI;G0+=G1;++Gz;G3+=7;}while(G3<G2);}static void Fd(float*
Gw,unsigned int Gx,float const*Gy,B4 const*Gz,float const*G0,int G1){float const
*G2=Gw+Gx*7;float*restrict G3=Gw;do{float const*G4=Gy+Gz->A*7;float const*G5=G0;
float G6,G7,G8,G9,G_,HA,HB,HC,HD,HE,HF,HG,HH,HI,HJ;HJ=G5[0];G6=G4[0]*HJ;G7=G4[1]
*HJ;G8=G4[2]*HJ;G9=G4[3]*HJ;G_=G4[4]*HJ;HA=G4[5]*HJ;HB=G4[6]*HJ;HJ=G5[1];HC=G4[7
]*HJ;HD=G4[8]*HJ;HE=G4[9]*HJ;HF=G4[10]*HJ;HG=G4[11]*HJ;HH=G4[12]*HJ;HI=G4[13]*HJ
;HJ=G5[2];G6+=G4[14]*HJ;G7+=G4[15]*HJ;G8+=G4[16]*HJ;G9+=G4[17]*HJ;G_+=G4[18]*HJ;
HA+=G4[19]*HJ;HB+=G4[20]*HJ;HJ=G5[3];HC+=G4[21]*HJ;HD+=G4[22]*HJ;HE+=G4[23]*HJ;
HF+=G4[24]*HJ;HG+=G4[25]*HJ;HH+=G4[26]*HJ;HI+=G4[27]*HJ;HJ=G5[4];G6+=G4[28]*HJ;
G7+=G4[29]*HJ;G8+=G4[30]*HJ;G9+=G4[31]*HJ;G_+=G4[32]*HJ;HA+=G4[33]*HJ;HB+=G4[34]
*HJ;HJ=G5[5];HC+=G4[35]*HJ;HD+=G4[36]*HJ;HE+=G4[37]*HJ;HF+=G4[38]*HJ;HG+=G4[39]*
HJ;HH+=G4[40]*HJ;HI+=G4[41]*HJ;HJ=G5[6];G6+=G4[42]*HJ;G7+=G4[43]*HJ;G8+=G4[44]*
HJ;G9+=G4[45]*HJ;G_+=G4[46]*HJ;HA+=G4[47]*HJ;HB+=G4[48]*HJ;HJ=G5[7];HC+=G4[49]*
HJ;HD+=G4[50]*HJ;HE+=G4[51]*HJ;HF+=G4[52]*HJ;HG+=G4[53]*HJ;HH+=G4[54]*HJ;HI+=G4[
55]*HJ;HJ=G5[8];G6+=G4[56]*HJ;G7+=G4[57]*HJ;G8+=G4[58]*HJ;G9+=G4[59]*HJ;G_+=G4[
60]*HJ;HA+=G4[61]*HJ;HB+=G4[62]*HJ;HJ=G5[9];HC+=G4[63]*HJ;HD+=G4[64]*HJ;HE+=G4[
65]*HJ;HF+=G4[66]*HJ;HG+=G4[67]*HJ;HH+=G4[68]*HJ;HI+=G4[69]*HJ;G3[0]=G6+HC;G3[1]
=G7+HD;G3[2]=G8+HE;G3[3]=G9+HF;G3[4]=G_+HG;G3[5]=HA+HH;G3[6]=HB+HI;G0+=G1;++Gz;
G3+=7;}while(G3<G2);}static void Fe(float*Gw,unsigned int Gx,float const*Gy,B4//
const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*7;float*restrict G3=Gw;do{
float const*G4=Gy+Gz->A*7;float const*G5=G0;float G6,G7,G8,G9,G_,HA,HB,HC,HD,HE,
HF,HG,HH,HI,HJ;HJ=G5[0];G6=G4[0]*HJ;G7=G4[1]*HJ;G8=G4[2]*HJ;G9=G4[3]*HJ;G_=G4[4]
*HJ;HA=G4[5]*HJ;HB=G4[6]*HJ;HJ=G5[1];HC=G4[7]*HJ;HD=G4[8]*HJ;HE=G4[9]*HJ;HF=G4[
10]*HJ;HG=G4[11]*HJ;HH=G4[12]*HJ;HI=G4[13]*HJ;HJ=G5[2];G6+=G4[14]*HJ;G7+=G4[15]*
HJ;G8+=G4[16]*HJ;G9+=G4[17]*HJ;G_+=G4[18]*HJ;HA+=G4[19]*HJ;HB+=G4[20]*HJ;HJ=G5[3
];HC+=G4[21]*HJ;HD+=G4[22]*HJ;HE+=G4[23]*HJ;HF+=G4[24]*HJ;HG+=G4[25]*HJ;HH+=G4[
26]*HJ;HI+=G4[27]*HJ;HJ=G5[4];G6+=G4[28]*HJ;G7+=G4[29]*HJ;G8+=G4[30]*HJ;G9+=G4[
31]*HJ;G_+=G4[32]*HJ;HA+=G4[33]*HJ;HB+=G4[34]*HJ;HJ=G5[5];HC+=G4[35]*HJ;HD+=G4[
36]*HJ;HE+=G4[37]*HJ;HF+=G4[38]*HJ;HG+=G4[39]*HJ;HH+=G4[40]*HJ;HI+=G4[41]*HJ;HJ=
G5[6];G6+=G4[42]*HJ;G7+=G4[43]*HJ;G8+=G4[44]*HJ;G9+=G4[45]*HJ;G_+=G4[46]*HJ;HA+=
G4[47]*HJ;HB+=G4[48]*HJ;HJ=G5[7];HC+=G4[49]*HJ;HD+=G4[50]*HJ;HE+=G4[51]*HJ;HF+=
G4[52]*HJ;HG+=G4[53]*HJ;HH+=G4[54]*HJ;HI+=G4[55]*HJ;HJ=G5[8];G6+=G4[56]*HJ;G7+=
G4[57]*HJ;G8+=G4[58]*HJ;G9+=G4[59]*HJ;G_+=G4[60]*HJ;HA+=G4[61]*HJ;HB+=G4[62]*HJ;
HJ=G5[9];HC+=G4[63]*HJ;HD+=G4[64]*HJ;HE+=G4[65]*HJ;HF+=G4[66]*HJ;HG+=G4[67]*HJ;
HH+=G4[68]*HJ;HI+=G4[69]*HJ;HJ=G5[10];G6+=G4[70]*HJ;G7+=G4[71]*HJ;G8+=G4[72]*HJ;
G9+=G4[73]*HJ;G_+=G4[74]*HJ;HA+=G4[75]*HJ;HB+=G4[76]*HJ;G3[0]=G6+HC;G3[1]=G7+HD;
G3[2]=G8+HE;G3[3]=G9+HF;G3[4]=G_+HG;G3[5]=HA+HH;G3[6]=HB+HI;G0+=G1;++Gz;G3+=7;}
while(G3<G2);}static void Ff(float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz
,float const*G0,int G1){float const*G2=Gw+Gx*7;float*restrict G3=Gw;do{float////
const*G4=Gy+Gz->A*7;float const*G5=G0;float G6,G7,G8,G9,G_,HA,HB,HC,HD,HE,HF,HG,
HH,HI,HJ;HJ=G5[0];G6=G4[0]*HJ;G7=G4[1]*HJ;G8=G4[2]*HJ;G9=G4[3]*HJ;G_=G4[4]*HJ;HA
=G4[5]*HJ;HB=G4[6]*HJ;HJ=G5[1];HC=G4[7]*HJ;HD=G4[8]*HJ;HE=G4[9]*HJ;HF=G4[10]*HJ;
HG=G4[11]*HJ;HH=G4[12]*HJ;HI=G4[13]*HJ;HJ=G5[2];G6+=G4[14]*HJ;G7+=G4[15]*HJ;G8+=
G4[16]*HJ;G9+=G4[17]*HJ;G_+=G4[18]*HJ;HA+=G4[19]*HJ;HB+=G4[20]*HJ;HJ=G5[3];HC+=
G4[21]*HJ;HD+=G4[22]*HJ;HE+=G4[23]*HJ;HF+=G4[24]*HJ;HG+=G4[25]*HJ;HH+=G4[26]*HJ;
HI+=G4[27]*HJ;HJ=G5[4];G6+=G4[28]*HJ;G7+=G4[29]*HJ;G8+=G4[30]*HJ;G9+=G4[31]*HJ;
G_+=G4[32]*HJ;HA+=G4[33]*HJ;HB+=G4[34]*HJ;HJ=G5[5];HC+=G4[35]*HJ;HD+=G4[36]*HJ;
HE+=G4[37]*HJ;HF+=G4[38]*HJ;HG+=G4[39]*HJ;HH+=G4[40]*HJ;HI+=G4[41]*HJ;HJ=G5[6];
G6+=G4[42]*HJ;G7+=G4[43]*HJ;G8+=G4[44]*HJ;G9+=G4[45]*HJ;G_+=G4[46]*HJ;HA+=G4[47]
*HJ;HB+=G4[48]*HJ;HJ=G5[7];HC+=G4[49]*HJ;HD+=G4[50]*HJ;HE+=G4[51]*HJ;HF+=G4[52]*
HJ;HG+=G4[53]*HJ;HH+=G4[54]*HJ;HI+=G4[55]*HJ;HJ=G5[8];G6+=G4[56]*HJ;G7+=G4[57]*
HJ;G8+=G4[58]*HJ;G9+=G4[59]*HJ;G_+=G4[60]*HJ;HA+=G4[61]*HJ;HB+=G4[62]*HJ;HJ=G5[9
];HC+=G4[63]*HJ;HD+=G4[64]*HJ;HE+=G4[65]*HJ;HF+=G4[66]*HJ;HG+=G4[67]*HJ;HH+=G4[
68]*HJ;HI+=G4[69]*HJ;HJ=G5[10];G6+=G4[70]*HJ;G7+=G4[71]*HJ;G8+=G4[72]*HJ;G9+=G4[
73]*HJ;G_+=G4[74]*HJ;HA+=G4[75]*HJ;HB+=G4[76]*HJ;HJ=G5[11];HC+=G4[77]*HJ;HD+=G4[
78]*HJ;HE+=G4[79]*HJ;HF+=G4[80]*HJ;HG+=G4[81]*HJ;HH+=G4[82]*HJ;HI+=G4[83]*HJ;G3[
0]=G6+HC;G3[1]=G7+HD;G3[2]=G8+HE;G3[3]=G9+HF;G3[4]=G_+HG;G3[5]=HA+HH;G3[6]=HB+HI
;G0+=G1;++Gz;G3+=7;}while(G3<G2);}static void Fg(float*Gw,unsigned int Gx,float
const*Gy,B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*7;float*///////
restrict G3=Gw;do{float const*G4=Gy+Gz->A*7;int G5=((Gz->B-Gz->A+1)-4+3)>>2;////
float const*G6=G0;float G7,G8,G9,G_,HA,HB,HC,HD,HE,HF,HG,HH,HI,HJ,HK;HK=G6[0];G7
=G4[0]*HK;G8=G4[1]*HK;G9=G4[2]*HK;G_=G4[3]*HK;HA=G4[4]*HK;HB=G4[5]*HK;HC=G4[6]*
HK;HK=G6[1];HD=G4[7]*HK;HE=G4[8]*HK;HF=G4[9]*HK;HG=G4[10]*HK;HH=G4[11]*HK;HI=G4[
12]*HK;HJ=G4[13]*HK;HK=G6[2];G7+=G4[14]*HK;G8+=G4[15]*HK;G9+=G4[16]*HK;G_+=G4[17
]*HK;HA+=G4[18]*HK;HB+=G4[19]*HK;HC+=G4[20]*HK;HK=G6[3];HD+=G4[21]*HK;HE+=G4[22]
*HK;HF+=G4[23]*HK;HG+=G4[24]*HK;HH+=G4[25]*HK;HI+=G4[26]*HK;HJ+=G4[27]*HK;do{G6
+=4;G4+=28;HK=G6[0];G7+=G4[0]*HK;G8+=G4[1]*HK;G9+=G4[2]*HK;G_+=G4[3]*HK;HA+=G4[4
]*HK;HB+=G4[5]*HK;HC+=G4[6]*HK;HK=G6[1];HD+=G4[7]*HK;HE+=G4[8]*HK;HF+=G4[9]*HK;
HG+=G4[10]*HK;HH+=G4[11]*HK;HI+=G4[12]*HK;HJ+=G4[13]*HK;HK=G6[2];G7+=G4[14]*HK;
G8+=G4[15]*HK;G9+=G4[16]*HK;G_+=G4[17]*HK;HA+=G4[18]*HK;HB+=G4[19]*HK;HC+=G4[20]
*HK;HK=G6[3];HD+=G4[21]*HK;HE+=G4[22]*HK;HF+=G4[23]*HK;HG+=G4[24]*HK;HH+=G4[25]*
HK;HI+=G4[26]*HK;HJ+=G4[27]*HK;--G5;}while(G5>0);G3[0]=G7+HD;G3[1]=G8+HE;G3[2]=
G9+HF;G3[3]=G_+HG;G3[4]=HA+HH;G3[5]=HB+HI;G3[6]=HC+HJ;G0+=G1;++Gz;G3+=7;}while(
G3<G2);}static void Fh(float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz,float
const*G0,int G1){float const*G2=Gw+Gx*7;float*restrict G3=Gw;do{float const*G4=
Gy+Gz->A*7;int G5=((Gz->B-Gz->A+1)-5+3)>>2;float const*G6=G0;float G7,G8,G9,G_,
HA,HB,HC,HD,HE,HF,HG,HH,HI,HJ,HK;HK=G6[0];G7=G4[0]*HK;G8=G4[1]*HK;G9=G4[2]*HK;G_
=G4[3]*HK;HA=G4[4]*HK;HB=G4[5]*HK;HC=G4[6]*HK;HK=G6[1];HD=G4[7]*HK;HE=G4[8]*HK;
HF=G4[9]*HK;HG=G4[10]*HK;HH=G4[11]*HK;HI=G4[12]*HK;HJ=G4[13]*HK;HK=G6[2];G7+=G4[
14]*HK;G8+=G4[15]*HK;G9+=G4[16]*HK;G_+=G4[17]*HK;HA+=G4[18]*HK;HB+=G4[19]*HK;HC
+=G4[20]*HK;HK=G6[3];HD+=G4[21]*HK;HE+=G4[22]*HK;HF+=G4[23]*HK;HG+=G4[24]*HK;HH
+=G4[25]*HK;HI+=G4[26]*HK;HJ+=G4[27]*HK;do{G6+=4;G4+=28;HK=G6[0];G7+=G4[0]*HK;G8
+=G4[1]*HK;G9+=G4[2]*HK;G_+=G4[3]*HK;HA+=G4[4]*HK;HB+=G4[5]*HK;HC+=G4[6]*HK;HK=
G6[1];HD+=G4[7]*HK;HE+=G4[8]*HK;HF+=G4[9]*HK;HG+=G4[10]*HK;HH+=G4[11]*HK;HI+=G4[
12]*HK;HJ+=G4[13]*HK;HK=G6[2];G7+=G4[14]*HK;G8+=G4[15]*HK;G9+=G4[16]*HK;G_+=G4[
17]*HK;HA+=G4[18]*HK;HB+=G4[19]*HK;HC+=G4[20]*HK;HK=G6[3];HD+=G4[21]*HK;HE+=G4[
22]*HK;HF+=G4[23]*HK;HG+=G4[24]*HK;HH+=G4[25]*HK;HI+=G4[26]*HK;HJ+=G4[27]*HK;--
G5;}while(G5>0);HK=G6[4];G7+=G4[28]*HK;G8+=G4[29]*HK;G9+=G4[30]*HK;G_+=G4[31]*HK
;HA+=G4[32]*HK;HB+=G4[33]*HK;HC+=G4[34]*HK;G3[0]=G7+HD;G3[1]=G8+HE;G3[2]=G9+HF;
G3[3]=G_+HG;G3[4]=HA+HH;G3[5]=HB+HI;G3[6]=HC+HJ;G0+=G1;++Gz;G3+=7;}while(G3<G2);
}static void Fi(float*Gw,unsigned int Gx,float const*Gy,B4 const*Gz,float const*
G0,int G1){float const*G2=Gw+Gx*7;float*restrict G3=Gw;do{float const*G4=Gy+Gz->
A*7;int G5=((Gz->B-Gz->A+1)-6+3)>>2;float const*G6=G0;float G7,G8,G9,G_,HA,HB,HC
,HD,HE,HF,HG,HH,HI,HJ,HK;HK=G6[0];G7=G4[0]*HK;G8=G4[1]*HK;G9=G4[2]*HK;G_=G4[3]*
HK;HA=G4[4]*HK;HB=G4[5]*HK;HC=G4[6]*HK;HK=G6[1];HD=G4[7]*HK;HE=G4[8]*HK;HF=G4[9]
*HK;HG=G4[10]*HK;HH=G4[11]*HK;HI=G4[12]*HK;HJ=G4[13]*HK;HK=G6[2];G7+=G4[14]*HK;
G8+=G4[15]*HK;G9+=G4[16]*HK;G_+=G4[17]*HK;HA+=G4[18]*HK;HB+=G4[19]*HK;HC+=G4[20]
*HK;HK=G6[3];HD+=G4[21]*HK;HE+=G4[22]*HK;HF+=G4[23]*HK;HG+=G4[24]*HK;HH+=G4[25]*
HK;HI+=G4[26]*HK;HJ+=G4[27]*HK;do{G6+=4;G4+=28;HK=G6[0];G7+=G4[0]*HK;G8+=G4[1]*
HK;G9+=G4[2]*HK;G_+=G4[3]*HK;HA+=G4[4]*HK;HB+=G4[5]*HK;HC+=G4[6]*HK;HK=G6[1];HD
+=G4[7]*HK;HE+=G4[8]*HK;HF+=G4[9]*HK;HG+=G4[10]*HK;HH+=G4[11]*HK;HI+=G4[12]*HK;
HJ+=G4[13]*HK;HK=G6[2];G7+=G4[14]*HK;G8+=G4[15]*HK;G9+=G4[16]*HK;G_+=G4[17]*HK;
HA+=G4[18]*HK;HB+=G4[19]*HK;HC+=G4[20]*HK;HK=G6[3];HD+=G4[21]*HK;HE+=G4[22]*HK;
HF+=G4[23]*HK;HG+=G4[24]*HK;HH+=G4[25]*HK;HI+=G4[26]*HK;HJ+=G4[27]*HK;--G5;}////
while(G5>0);HK=G6[4];G7+=G4[28]*HK;G8+=G4[29]*HK;G9+=G4[30]*HK;G_+=G4[31]*HK;HA
+=G4[32]*HK;HB+=G4[33]*HK;HC+=G4[34]*HK;HK=G6[5];HD+=G4[35]*HK;HE+=G4[36]*HK;HF
+=G4[37]*HK;HG+=G4[38]*HK;HH+=G4[39]*HK;HI+=G4[40]*HK;HJ+=G4[41]*HK;G3[0]=G7+HD;
G3[1]=G8+HE;G3[2]=G9+HF;G3[3]=G_+HG;G3[4]=HA+HH;G3[5]=HB+HI;G3[6]=HC+HJ;G0+=G1;
++Gz;G3+=7;}while(G3<G2);}static void Fj(float*Gw,unsigned int Gx,float const*Gy
,B4 const*Gz,float const*G0,int G1){float const*G2=Gw+Gx*7;float*restrict G3=Gw;
do{float const*G4=Gy+Gz->A*7;int G5=((Gz->B-Gz->A+1)-7+3)>>2;float const*G6=G0;
float G7,G8,G9,G_,HA,HB,HC,HD,HE,HF,HG,HH,HI,HJ,HK;HK=G6[0];G7=G4[0]*HK;G8=G4[1]
*HK;G9=G4[2]*HK;G_=G4[3]*HK;HA=G4[4]*HK;HB=G4[5]*HK;HC=G4[6]*HK;HK=G6[1];HD=G4[7
]*HK;HE=G4[8]*HK;HF=G4[9]*HK;HG=G4[10]*HK;HH=G4[11]*HK;HI=G4[12]*HK;HJ=G4[13]*HK
;HK=G6[2];G7+=G4[14]*HK;G8+=G4[15]*HK;G9+=G4[16]*HK;G_+=G4[17]*HK;HA+=G4[18]*HK;
HB+=G4[19]*HK;HC+=G4[20]*HK;HK=G6[3];HD+=G4[21]*HK;HE+=G4[22]*HK;HF+=G4[23]*HK;
HG+=G4[24]*HK;HH+=G4[25]*HK;HI+=G4[26]*HK;HJ+=G4[27]*HK;do{G6+=4;G4+=28;HK=G6[0]
;G7+=G4[0]*HK;G8+=G4[1]*HK;G9+=G4[2]*HK;G_+=G4[3]*HK;HA+=G4[4]*HK;HB+=G4[5]*HK;
HC+=G4[6]*HK;HK=G6[1];HD+=G4[7]*HK;HE+=G4[8]*HK;HF+=G4[9]*HK;HG+=G4[10]*HK;HH+=
G4[11]*HK;HI+=G4[12]*HK;HJ+=G4[13]*HK;HK=G6[2];G7+=G4[14]*HK;G8+=G4[15]*HK;G9+=
G4[16]*HK;G_+=G4[17]*HK;HA+=G4[18]*HK;HB+=G4[19]*HK;HC+=G4[20]*HK;HK=G6[3];HD+=
G4[21]*HK;HE+=G4[22]*HK;HF+=G4[23]*HK;HG+=G4[24]*HK;HH+=G4[25]*HK;HI+=G4[26]*HK;
HJ+=G4[27]*HK;--G5;}while(G5>0);HK=G6[4];G7+=G4[28]*HK;G8+=G4[29]*HK;G9+=G4[30]*
HK;G_+=G4[31]*HK;HA+=G4[32]*HK;HB+=G4[33]*HK;HC+=G4[34]*HK;HK=G6[5];HD+=G4[35]*
HK;HE+=G4[36]*HK;HF+=G4[37]*HK;HG+=G4[38]*HK;HH+=G4[39]*HK;HI+=G4[40]*HK;HJ+=G4[
41]*HK;HK=G6[6];G7+=G4[42]*HK;G8+=G4[43]*HK;G9+=G4[44]*HK;G_+=G4[45]*HK;HA+=G4[
46]*HK;HB+=G4[47]*HK;HC+=G4[48]*HK;G3[0]=G7+HD;G3[1]=G8+HE;G3[2]=G9+HF;G3[3]=G_+
HG;G3[4]=HA+HH;G3[5]=HB+HI;G3[6]=HC+HJ;G0+=G1;++Gz;G3+=7;}while(G3<G2);}static//
CC*Fk[4]={Fg,Fh,Fi,Fj,};static CC*Fl[12]={FU,FV,FW,FX,FY,FZ,Fa,Fb,Fc,Fd,Fe,Ff,};
static void Fm(float**Gw,float const*Gx,float const*Gy,float const*Gz){float*///
restrict G0=Gw[0];float G1=Gx[0];_Pragma("GCC unroll 1")_Pragma("GCC novector")
while(((char*)Gz-(char*)Gy)>=16){float G2,G3,G4,G5;asm(""::"r"(Gy));G2=Gy[0],G3=
Gy[1],G4=Gy[2],G5=Gy[3];G0[0]=(G2*G1);G0[1]=(G3*G1);G0[2]=(G4*G1);G0[3]=(G5*G1);
Gy+=4;G0+=4;}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(Gy<Gz){float G2
=Gy[0];asm(""::"r"(G0));G0[0]=(G2*G1);++Gy;++G0;}}static void Fn(float*Gw,float
const*Gx,float const**Gy,float const*Gz){float*restrict G0=Gw;float const*G1=Gy[
0];float G2=Gx[0];if((G2>=1e00)&&(G2<=1e00)){memcpy(G0,G1,(char*)Gz-(char*)G1);
return;}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char*)Gz-(char*)G1
)>=16){float G3,G4,G5,G6;asm(""::"r"(G0));G3=G1[0]*G2;G4=G1[1]*G2;G5=G1[2]*G2;G6
=G1[3]*G2;G0[0]=G3;G0[1]=G4;G0[2]=G5;G0[3]=G6;G0+=4;G1+=4;}_Pragma(/////////////
"GCC unroll 1")_Pragma("GCC novector")while(G1<Gz){float G3;asm(""::"r"(G0));G3=
G1[0]*G2;G0[0]=G3;++G0;++G1;}}static void Fo(float**Gw,float const*Gx,float/////
const*Gy,float const*Gz){float*restrict G0=Gw[0];float G1=Gx[0];_Pragma(////////
"GCC unroll 1")_Pragma("GCC novector")while(((char*)Gz-(char*)Gy)>=16){float G2,
G3,G4,G5;asm(""::"r"(Gy));G2=Gy[0],G3=Gy[1],G4=Gy[2],G5=Gy[3];G0[0]+=(G2*G1);G0[
1]+=(G3*G1);G0[2]+=(G4*G1);G0[3]+=(G5*G1);Gy+=4;G0+=4;}_Pragma("GCC unroll 1")//
_Pragma("GCC novector")while(Gy<Gz){float G2=Gy[0];asm(""::"r"(G0));G0[0]+=(G2*
G1);++Gy;++G0;}}static void Fp(float*Gw,float const*Gx,float const**Gy,float////
const*Gz){float*restrict G0=Gw;float const*G1=Gy[0];float G2=Gx[0];_Pragma(/////
"GCC unroll 1")_Pragma("GCC novector")while(((char*)Gz-(char*)G1)>=16){float G3,
G4,G5,G6;asm(""::"r"(G0));G3=G0[0]+G1[0]*G2;G4=G0[1]+G1[1]*G2;G5=G0[2]+G1[2]*G2;
G6=G0[3]+G1[3]*G2;G0[0]=G3;G0[1]=G4;G0[2]=G5;G0[3]=G6;G0+=4;G1+=4;}_Pragma(/////
"GCC unroll 1")_Pragma("GCC novector")while(G1<Gz){float G3;asm(""::"r"(G0));G3=
G0[0]+G1[0]*G2;G0[0]=G3;++G0;++G1;}}static void Fq(float**Gw,float const*Gx,////
float const*Gy,float const*Gz){float*restrict G0=Gw[0];float G1=Gx[0];float*////
restrict G2=Gw[1];float G3=Gx[1];_Pragma("GCC unroll 1")_Pragma("GCC novector")
while(((char*)Gz-(char*)Gy)>=16){float G4,G5,G6,G7;asm(""::"r"(Gy));G4=Gy[0],G5=
Gy[1],G6=Gy[2],G7=Gy[3];G0[0]=(G4*G1);G0[1]=(G5*G1);G0[2]=(G6*G1);G0[3]=(G7*G1);
G2[0]=(G4*G3);G2[1]=(G5*G3);G2[2]=(G6*G3);G2[3]=(G7*G3);Gy+=4;G0+=4;G2+=4;}/////
_Pragma("GCC unroll 1")_Pragma("GCC novector")while(Gy<Gz){float G4=Gy[0];asm(""
::"r"(G0));G0[0]=(G4*G1);G2[0]=(G4*G3);++Gy;++G0;++G2;}}static void Fr(float*Gw,
float const*Gx,float const**Gy,float const*Gz){float*restrict G0=Gw;float const*
G1=Gy[0];float G2=Gx[0];float const*G3=Gy[1];float G4=Gx[1];_Pragma(////////////
"GCC unroll 1")_Pragma("GCC novector")while(((char*)Gz-(char*)G1)>=16){float G5,
G6,G7,G8;asm(""::"r"(G0));G5=G1[0]*G2;G6=G1[1]*G2;G7=G1[2]*G2;G8=G1[3]*G2;G5+=G3
[0]*G4;G6+=G3[1]*G4;G7+=G3[2]*G4;G8+=G3[3]*G4;G0[0]=G5;G0[1]=G6;G0[2]=G7;G0[3]=
G8;G0+=4;G1+=4;G3+=4;}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(G1<Gz)
{float G5;asm(""::"r"(G0));G5=G1[0]*G2;G5+=G3[0]*G4;G0[0]=G5;++G0;++G1;++G3;}}//
static void Fs(float**Gw,float const*Gx,float const*Gy,float const*Gz){float*///
restrict G0=Gw[0];float G1=Gx[0];float*restrict G2=Gw[1];float G3=Gx[1];_Pragma(
"GCC unroll 1")_Pragma("GCC novector")while(((char*)Gz-(char*)Gy)>=16){float G4,
G5,G6,G7;asm(""::"r"(Gy));G4=Gy[0],G5=Gy[1],G6=Gy[2],G7=Gy[3];G0[0]+=(G4*G1);G0[
1]+=(G5*G1);G0[2]+=(G6*G1);G0[3]+=(G7*G1);G2[0]+=(G4*G3);G2[1]+=(G5*G3);G2[2]+=(
G6*G3);G2[3]+=(G7*G3);Gy+=4;G0+=4;G2+=4;}_Pragma("GCC unroll 1")_Pragma(////////
"GCC novector")while(Gy<Gz){float G4=Gy[0];asm(""::"r"(G0));G0[0]+=(G4*G1);G2[0]
+=(G4*G3);++Gy;++G0;++G2;}}static void Ft(float*Gw,float const*Gx,float const**
Gy,float const*Gz){float*restrict G0=Gw;float const*G1=Gy[0];float G2=Gx[0];////
float const*G3=Gy[1];float G4=Gx[1];_Pragma("GCC unroll 1")_Pragma(/////////////
"GCC novector")while(((char*)Gz-(char*)G1)>=16){float G5,G6,G7,G8;asm(""::"r"(G0
));G5=G0[0]+G1[0]*G2;G6=G0[1]+G1[1]*G2;G7=G0[2]+G1[2]*G2;G8=G0[3]+G1[3]*G2;G5+=
G3[0]*G4;G6+=G3[1]*G4;G7+=G3[2]*G4;G8+=G3[3]*G4;G0[0]=G5;G0[1]=G6;G0[2]=G7;G0[3]
=G8;G0+=4;G1+=4;G3+=4;}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(G1<Gz
){float G5;asm(""::"r"(G0));G5=G0[0]+G1[0]*G2;G5+=G3[0]*G4;G0[0]=G5;++G0;++G1;++
G3;}}static void Fu(float**Gw,float const*Gx,float const*Gy,float const*Gz){////
float*restrict G0=Gw[0];float G1=Gx[0];float*restrict G2=Gw[1];float G3=Gx[1];//
float*restrict G4=Gw[2];float G5=Gx[2];_Pragma("GCC unroll 1")_Pragma(//////////
"GCC novector")while(((char*)Gz-(char*)Gy)>=16){float G6,G7,G8,G9;asm(""::"r"(Gy
));G6=Gy[0],G7=Gy[1],G8=Gy[2],G9=Gy[3];G0[0]=(G6*G1);G0[1]=(G7*G1);G0[2]=(G8*G1)
;G0[3]=(G9*G1);G2[0]=(G6*G3);G2[1]=(G7*G3);G2[2]=(G8*G3);G2[3]=(G9*G3);G4[0]=(G6
*G5);G4[1]=(G7*G5);G4[2]=(G8*G5);G4[3]=(G9*G5);Gy+=4;G0+=4;G2+=4;G4+=4;}_Pragma(
"GCC unroll 1")_Pragma("GCC novector")while(Gy<Gz){float G6=Gy[0];asm(""::"r"(G0
));G0[0]=(G6*G1);G2[0]=(G6*G3);G4[0]=(G6*G5);++Gy;++G0;++G2;++G4;}}static void//
Fv(float*Gw,float const*Gx,float const**Gy,float const*Gz){float*restrict G0=Gw;
float const*G1=Gy[0];float G2=Gx[0];float const*G3=Gy[1];float G4=Gx[1];float///
const*G5=Gy[2];float G6=Gx[2];_Pragma("GCC unroll 1")_Pragma("GCC novector")////
while(((char*)Gz-(char*)G1)>=16){float G7,G8,G9,G_;asm(""::"r"(G0));G7=G1[0]*G2;
G8=G1[1]*G2;G9=G1[2]*G2;G_=G1[3]*G2;G7+=G3[0]*G4;G8+=G3[1]*G4;G9+=G3[2]*G4;G_+=
G3[3]*G4;G7+=G5[0]*G6;G8+=G5[1]*G6;G9+=G5[2]*G6;G_+=G5[3]*G6;G0[0]=G7;G0[1]=G8;
G0[2]=G9;G0[3]=G_;G0+=4;G1+=4;G3+=4;G5+=4;}_Pragma("GCC unroll 1")_Pragma(//////
"GCC novector")while(G1<Gz){float G7;asm(""::"r"(G0));G7=G1[0]*G2;G7+=G3[0]*G4;
G7+=G5[0]*G6;G0[0]=G7;++G0;++G1;++G3;++G5;}}static void Fw(float**Gw,float const
*Gx,float const*Gy,float const*Gz){float*restrict G0=Gw[0];float G1=Gx[0];float*
restrict G2=Gw[1];float G3=Gx[1];float*restrict G4=Gw[2];float G5=Gx[2];_Pragma(
"GCC unroll 1")_Pragma("GCC novector")while(((char*)Gz-(char*)Gy)>=16){float G6,
G7,G8,G9;asm(""::"r"(Gy));G6=Gy[0],G7=Gy[1],G8=Gy[2],G9=Gy[3];G0[0]+=(G6*G1);G0[
1]+=(G7*G1);G0[2]+=(G8*G1);G0[3]+=(G9*G1);G2[0]+=(G6*G3);G2[1]+=(G7*G3);G2[2]+=(
G8*G3);G2[3]+=(G9*G3);G4[0]+=(G6*G5);G4[1]+=(G7*G5);G4[2]+=(G8*G5);G4[3]+=(G9*G5
);Gy+=4;G0+=4;G2+=4;G4+=4;}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(
Gy<Gz){float G6=Gy[0];asm(""::"r"(G0));G0[0]+=(G6*G1);G2[0]+=(G6*G3);G4[0]+=(G6*
G5);++Gy;++G0;++G2;++G4;}}static void Fx(float*Gw,float const*Gx,float const**Gy
,float const*Gz){float*restrict G0=Gw;float const*G1=Gy[0];float G2=Gx[0];float
const*G3=Gy[1];float G4=Gx[1];float const*G5=Gy[2];float G6=Gx[2];_Pragma(//////
"GCC unroll 1")_Pragma("GCC novector")while(((char*)Gz-(char*)G1)>=16){float G7,
G8,G9,G_;asm(""::"r"(G0));G7=G0[0]+G1[0]*G2;G8=G0[1]+G1[1]*G2;G9=G0[2]+G1[2]*G2;
G_=G0[3]+G1[3]*G2;G7+=G3[0]*G4;G8+=G3[1]*G4;G9+=G3[2]*G4;G_+=G3[3]*G4;G7+=G5[0]*
G6;G8+=G5[1]*G6;G9+=G5[2]*G6;G_+=G5[3]*G6;G0[0]=G7;G0[1]=G8;G0[2]=G9;G0[3]=G_;G0
+=4;G1+=4;G3+=4;G5+=4;}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(G1<Gz
){float G7;asm(""::"r"(G0));G7=G0[0]+G1[0]*G2;G7+=G3[0]*G4;G7+=G5[0]*G6;G0[0]=G7
;++G0;++G1;++G3;++G5;}}static void Fy(float**Gw,float const*Gx,float const*Gy,//
float const*Gz){float*restrict G0=Gw[0];float G1=Gx[0];float*restrict G2=Gw[1];
float G3=Gx[1];float*restrict G4=Gw[2];float G5=Gx[2];float*restrict G6=Gw[3];//
float G7=Gx[3];_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char*)Gz-(
char*)Gy)>=16){float G8,G9,G_,HA;asm(""::"r"(Gy));G8=Gy[0],G9=Gy[1],G_=Gy[2],HA=
Gy[3];G0[0]=(G8*G1);G0[1]=(G9*G1);G0[2]=(G_*G1);G0[3]=(HA*G1);G2[0]=(G8*G3);G2[1
]=(G9*G3);G2[2]=(G_*G3);G2[3]=(HA*G3);G4[0]=(G8*G5);G4[1]=(G9*G5);G4[2]=(G_*G5);
G4[3]=(HA*G5);G6[0]=(G8*G7);G6[1]=(G9*G7);G6[2]=(G_*G7);G6[3]=(HA*G7);Gy+=4;G0+=
4;G2+=4;G4+=4;G6+=4;}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(Gy<Gz){
float G8=Gy[0];asm(""::"r"(G0));G0[0]=(G8*G1);G2[0]=(G8*G3);G4[0]=(G8*G5);G6[0]=
(G8*G7);++Gy;++G0;++G2;++G4;++G6;}}static void Fz(float*Gw,float const*Gx,float
const**Gy,float const*Gz){float*restrict G0=Gw;float const*G1=Gy[0];float G2=Gx[
0];float const*G3=Gy[1];float G4=Gx[1];float const*G5=Gy[2];float G6=Gx[2];float
const*G7=Gy[3];float G8=Gx[3];_Pragma("GCC unroll 1")_Pragma("GCC novector")////
while(((char*)Gz-(char*)G1)>=16){float G9,G_,HA,HB;asm(""::"r"(G0));G9=G1[0]*G2;
G_=G1[1]*G2;HA=G1[2]*G2;HB=G1[3]*G2;G9+=G3[0]*G4;G_+=G3[1]*G4;HA+=G3[2]*G4;HB+=
G3[3]*G4;G9+=G5[0]*G6;G_+=G5[1]*G6;HA+=G5[2]*G6;HB+=G5[3]*G6;G9+=G7[0]*G8;G_+=G7
[1]*G8;HA+=G7[2]*G8;HB+=G7[3]*G8;G0[0]=G9;G0[1]=G_;G0[2]=HA;G0[3]=HB;G0+=4;G1+=4
;G3+=4;G5+=4;G7+=4;}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(G1<Gz){
float G9;asm(""::"r"(G0));G9=G1[0]*G2;G9+=G3[0]*G4;G9+=G5[0]*G6;G9+=G7[0]*G8;G0[
0]=G9;++G0;++G1;++G3;++G5;++G7;}}static void F0(float**Gw,float const*Gx,float//
const*Gy,float const*Gz){float*restrict G0=Gw[0];float G1=Gx[0];float*restrict//
G2=Gw[1];float G3=Gx[1];float*restrict G4=Gw[2];float G5=Gx[2];float*restrict G6
=Gw[3];float G7=Gx[3];_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char
*)Gz-(char*)Gy)>=16){float G8,G9,G_,HA;asm(""::"r"(Gy));G8=Gy[0],G9=Gy[1],G_=Gy[
2],HA=Gy[3];G0[0]+=(G8*G1);G0[1]+=(G9*G1);G0[2]+=(G_*G1);G0[3]+=(HA*G1);G2[0]+=(
G8*G3);G2[1]+=(G9*G3);G2[2]+=(G_*G3);G2[3]+=(HA*G3);G4[0]+=(G8*G5);G4[1]+=(G9*G5
);G4[2]+=(G_*G5);G4[3]+=(HA*G5);G6[0]+=(G8*G7);G6[1]+=(G9*G7);G6[2]+=(G_*G7);G6[
3]+=(HA*G7);Gy+=4;G0+=4;G2+=4;G4+=4;G6+=4;}_Pragma("GCC unroll 1")_Pragma(//////
"GCC novector")while(Gy<Gz){float G8=Gy[0];asm(""::"r"(G0));G0[0]+=(G8*G1);G2[0]
+=(G8*G3);G4[0]+=(G8*G5);G6[0]+=(G8*G7);++Gy;++G0;++G2;++G4;++G6;}}static void//
F1(float*Gw,float const*Gx,float const**Gy,float const*Gz){float*restrict G0=Gw;
float const*G1=Gy[0];float G2=Gx[0];float const*G3=Gy[1];float G4=Gx[1];float///
const*G5=Gy[2];float G6=Gx[2];float const*G7=Gy[3];float G8=Gx[3];_Pragma(//////
"GCC unroll 1")_Pragma("GCC novector")while(((char*)Gz-(char*)G1)>=16){float G9,
G_,HA,HB;asm(""::"r"(G0));G9=G0[0]+G1[0]*G2;G_=G0[1]+G1[1]*G2;HA=G0[2]+G1[2]*G2;
HB=G0[3]+G1[3]*G2;G9+=G3[0]*G4;G_+=G3[1]*G4;HA+=G3[2]*G4;HB+=G3[3]*G4;G9+=G5[0]*
G6;G_+=G5[1]*G6;HA+=G5[2]*G6;HB+=G5[3]*G6;G9+=G7[0]*G8;G_+=G7[1]*G8;HA+=G7[2]*G8
;HB+=G7[3]*G8;G0[0]=G9;G0[1]=G_;G0[2]=HA;G0[3]=HB;G0+=4;G1+=4;G3+=4;G5+=4;G7+=4;
}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(G1<Gz){float G9;asm(""::"r"
(G0));G9=G0[0]+G1[0]*G2;G9+=G3[0]*G4;G9+=G5[0]*G6;G9+=G7[0]*G8;G0[0]=G9;++G0;++
G1;++G3;++G5;++G7;}}static void F2(float**Gw,float const*Gx,float const*Gy,float
const*Gz){float*restrict G0=Gw[0];float G1=Gx[0];float*restrict G2=Gw[1];float//
G3=Gx[1];float*restrict G4=Gw[2];float G5=Gx[2];float*restrict G6=Gw[3];float G7
=Gx[3];float*restrict G8=Gw[4];float G9=Gx[4];_Pragma("GCC unroll 1")_Pragma(///
"GCC novector")while(((char*)Gz-(char*)Gy)>=16){float G_,HA,HB,HC;asm(""::"r"(Gy
));G_=Gy[0],HA=Gy[1],HB=Gy[2],HC=Gy[3];G0[0]=(G_*G1);G0[1]=(HA*G1);G0[2]=(HB*G1)
;G0[3]=(HC*G1);G2[0]=(G_*G3);G2[1]=(HA*G3);G2[2]=(HB*G3);G2[3]=(HC*G3);G4[0]=(G_
*G5);G4[1]=(HA*G5);G4[2]=(HB*G5);G4[3]=(HC*G5);G6[0]=(G_*G7);G6[1]=(HA*G7);G6[2]
=(HB*G7);G6[3]=(HC*G7);G8[0]=(G_*G9);G8[1]=(HA*G9);G8[2]=(HB*G9);G8[3]=(HC*G9);
Gy+=4;G0+=4;G2+=4;G4+=4;G6+=4;G8+=4;}_Pragma("GCC unroll 1")_Pragma(////////////
"GCC novector")while(Gy<Gz){float G_=Gy[0];asm(""::"r"(G0));G0[0]=(G_*G1);G2[0]=
(G_*G3);G4[0]=(G_*G5);G6[0]=(G_*G7);G8[0]=(G_*G9);++Gy;++G0;++G2;++G4;++G6;++G8;
}}static void F3(float*Gw,float const*Gx,float const**Gy,float const*Gz){float*
restrict G0=Gw;float const*G1=Gy[0];float G2=Gx[0];float const*G3=Gy[1];float G4
=Gx[1];float const*G5=Gy[2];float G6=Gx[2];float const*G7=Gy[3];float G8=Gx[3];
float const*G9=Gy[4];float G_=Gx[4];_Pragma("GCC unroll 1")_Pragma(/////////////
"GCC novector")while(((char*)Gz-(char*)G1)>=16){float HA,HB,HC,HD;asm(""::"r"(G0
));HA=G1[0]*G2;HB=G1[1]*G2;HC=G1[2]*G2;HD=G1[3]*G2;HA+=G3[0]*G4;HB+=G3[1]*G4;HC
+=G3[2]*G4;HD+=G3[3]*G4;HA+=G5[0]*G6;HB+=G5[1]*G6;HC+=G5[2]*G6;HD+=G5[3]*G6;HA+=
G7[0]*G8;HB+=G7[1]*G8;HC+=G7[2]*G8;HD+=G7[3]*G8;HA+=G9[0]*G_;HB+=G9[1]*G_;HC+=G9
[2]*G_;HD+=G9[3]*G_;G0[0]=HA;G0[1]=HB;G0[2]=HC;G0[3]=HD;G0+=4;G1+=4;G3+=4;G5+=4;
G7+=4;G9+=4;}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(G1<Gz){float HA
;asm(""::"r"(G0));HA=G1[0]*G2;HA+=G3[0]*G4;HA+=G5[0]*G6;HA+=G7[0]*G8;HA+=G9[0]*
G_;G0[0]=HA;++G0;++G1;++G3;++G5;++G7;++G9;}}static void F4(float**Gw,float const
*Gx,float const*Gy,float const*Gz){float*restrict G0=Gw[0];float G1=Gx[0];float*
restrict G2=Gw[1];float G3=Gx[1];float*restrict G4=Gw[2];float G5=Gx[2];float*//
restrict G6=Gw[3];float G7=Gx[3];float*restrict G8=Gw[4];float G9=Gx[4];_Pragma(
"GCC unroll 1")_Pragma("GCC novector")while(((char*)Gz-(char*)Gy)>=16){float G_,
HA,HB,HC;asm(""::"r"(Gy));G_=Gy[0],HA=Gy[1],HB=Gy[2],HC=Gy[3];G0[0]+=(G_*G1);G0[
1]+=(HA*G1);G0[2]+=(HB*G1);G0[3]+=(HC*G1);G2[0]+=(G_*G3);G2[1]+=(HA*G3);G2[2]+=(
HB*G3);G2[3]+=(HC*G3);G4[0]+=(G_*G5);G4[1]+=(HA*G5);G4[2]+=(HB*G5);G4[3]+=(HC*G5
);G6[0]+=(G_*G7);G6[1]+=(HA*G7);G6[2]+=(HB*G7);G6[3]+=(HC*G7);G8[0]+=(G_*G9);G8[
1]+=(HA*G9);G8[2]+=(HB*G9);G8[3]+=(HC*G9);Gy+=4;G0+=4;G2+=4;G4+=4;G6+=4;G8+=4;}
_Pragma("GCC unroll 1")_Pragma("GCC novector")while(Gy<Gz){float G_=Gy[0];asm(""
::"r"(G0));G0[0]+=(G_*G1);G2[0]+=(G_*G3);G4[0]+=(G_*G5);G6[0]+=(G_*G7);G8[0]+=(
G_*G9);++Gy;++G0;++G2;++G4;++G6;++G8;}}static void F5(float*Gw,float const*Gx,//
float const**Gy,float const*Gz){float*restrict G0=Gw;float const*G1=Gy[0];float
G2=Gx[0];float const*G3=Gy[1];float G4=Gx[1];float const*G5=Gy[2];float G6=Gx[2]
;float const*G7=Gy[3];float G8=Gx[3];float const*G9=Gy[4];float G_=Gx[4];_Pragma
("GCC unroll 1")_Pragma("GCC novector")while(((char*)Gz-(char*)G1)>=16){float HA
,HB,HC,HD;asm(""::"r"(G0));HA=G0[0]+G1[0]*G2;HB=G0[1]+G1[1]*G2;HC=G0[2]+G1[2]*G2
;HD=G0[3]+G1[3]*G2;HA+=G3[0]*G4;HB+=G3[1]*G4;HC+=G3[2]*G4;HD+=G3[3]*G4;HA+=G5[0]
*G6;HB+=G5[1]*G6;HC+=G5[2]*G6;HD+=G5[3]*G6;HA+=G7[0]*G8;HB+=G7[1]*G8;HC+=G7[2]*
G8;HD+=G7[3]*G8;HA+=G9[0]*G_;HB+=G9[1]*G_;HC+=G9[2]*G_;HD+=G9[3]*G_;G0[0]=HA;G0[
1]=HB;G0[2]=HC;G0[3]=HD;G0+=4;G1+=4;G3+=4;G5+=4;G7+=4;G9+=4;}_Pragma(///////////
"GCC unroll 1")_Pragma("GCC novector")while(G1<Gz){float HA;asm(""::"r"(G0));HA=
G0[0]+G1[0]*G2;HA+=G3[0]*G4;HA+=G5[0]*G6;HA+=G7[0]*G8;HA+=G9[0]*G_;G0[0]=HA;++G0
;++G1;++G3;++G5;++G7;++G9;}}static void F6(float**Gw,float const*Gx,float const*
Gy,float const*Gz){float*restrict G0=Gw[0];float G1=Gx[0];float*restrict G2=Gw[1
];float G3=Gx[1];float*restrict G4=Gw[2];float G5=Gx[2];float*restrict G6=Gw[3];
float G7=Gx[3];float*restrict G8=Gw[4];float G9=Gx[4];float*restrict G_=Gw[5];//
float HA=Gx[5];_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char*)Gz-(
char*)Gy)>=16){float HB,HC,HD,HE;asm(""::"r"(Gy));HB=Gy[0],HC=Gy[1],HD=Gy[2],HE=
Gy[3];G0[0]=(HB*G1);G0[1]=(HC*G1);G0[2]=(HD*G1);G0[3]=(HE*G1);G2[0]=(HB*G3);G2[1
]=(HC*G3);G2[2]=(HD*G3);G2[3]=(HE*G3);G4[0]=(HB*G5);G4[1]=(HC*G5);G4[2]=(HD*G5);
G4[3]=(HE*G5);G6[0]=(HB*G7);G6[1]=(HC*G7);G6[2]=(HD*G7);G6[3]=(HE*G7);G8[0]=(HB*
G9);G8[1]=(HC*G9);G8[2]=(HD*G9);G8[3]=(HE*G9);G_[0]=(HB*HA);G_[1]=(HC*HA);G_[2]=
(HD*HA);G_[3]=(HE*HA);Gy+=4;G0+=4;G2+=4;G4+=4;G6+=4;G8+=4;G_+=4;}_Pragma(///////
"GCC unroll 1")_Pragma("GCC novector")while(Gy<Gz){float HB=Gy[0];asm(""::"r"(G0
));G0[0]=(HB*G1);G2[0]=(HB*G3);G4[0]=(HB*G5);G6[0]=(HB*G7);G8[0]=(HB*G9);G_[0]=(
HB*HA);++Gy;++G0;++G2;++G4;++G6;++G8;++G_;}}static void F7(float*Gw,float const*
Gx,float const**Gy,float const*Gz){float*restrict G0=Gw;float const*G1=Gy[0];///
float G2=Gx[0];float const*G3=Gy[1];float G4=Gx[1];float const*G5=Gy[2];float G6
=Gx[2];float const*G7=Gy[3];float G8=Gx[3];float const*G9=Gy[4];float G_=Gx[4];
float const*HA=Gy[5];float HB=Gx[5];_Pragma("GCC unroll 1")_Pragma(/////////////
"GCC novector")while(((char*)Gz-(char*)G1)>=16){float HC,HD,HE,HF;asm(""::"r"(G0
));HC=G1[0]*G2;HD=G1[1]*G2;HE=G1[2]*G2;HF=G1[3]*G2;HC+=G3[0]*G4;HD+=G3[1]*G4;HE
+=G3[2]*G4;HF+=G3[3]*G4;HC+=G5[0]*G6;HD+=G5[1]*G6;HE+=G5[2]*G6;HF+=G5[3]*G6;HC+=
G7[0]*G8;HD+=G7[1]*G8;HE+=G7[2]*G8;HF+=G7[3]*G8;HC+=G9[0]*G_;HD+=G9[1]*G_;HE+=G9
[2]*G_;HF+=G9[3]*G_;HC+=HA[0]*HB;HD+=HA[1]*HB;HE+=HA[2]*HB;HF+=HA[3]*HB;G0[0]=HC
;G0[1]=HD;G0[2]=HE;G0[3]=HF;G0+=4;G1+=4;G3+=4;G5+=4;G7+=4;G9+=4;HA+=4;}_Pragma(
"GCC unroll 1")_Pragma("GCC novector")while(G1<Gz){float HC;asm(""::"r"(G0));HC=
G1[0]*G2;HC+=G3[0]*G4;HC+=G5[0]*G6;HC+=G7[0]*G8;HC+=G9[0]*G_;HC+=HA[0]*HB;G0[0]=
HC;++G0;++G1;++G3;++G5;++G7;++G9;++HA;}}static void F8(float**Gw,float const*Gx,
float const*Gy,float const*Gz){float*restrict G0=Gw[0];float G1=Gx[0];float*////
restrict G2=Gw[1];float G3=Gx[1];float*restrict G4=Gw[2];float G5=Gx[2];float*//
restrict G6=Gw[3];float G7=Gx[3];float*restrict G8=Gw[4];float G9=Gx[4];float*//
restrict G_=Gw[5];float HA=Gx[5];_Pragma("GCC unroll 1")_Pragma("GCC novector")
while(((char*)Gz-(char*)Gy)>=16){float HB,HC,HD,HE;asm(""::"r"(Gy));HB=Gy[0],HC=
Gy[1],HD=Gy[2],HE=Gy[3];G0[0]+=(HB*G1);G0[1]+=(HC*G1);G0[2]+=(HD*G1);G0[3]+=(HE*
G1);G2[0]+=(HB*G3);G2[1]+=(HC*G3);G2[2]+=(HD*G3);G2[3]+=(HE*G3);G4[0]+=(HB*G5);
G4[1]+=(HC*G5);G4[2]+=(HD*G5);G4[3]+=(HE*G5);G6[0]+=(HB*G7);G6[1]+=(HC*G7);G6[2]
+=(HD*G7);G6[3]+=(HE*G7);G8[0]+=(HB*G9);G8[1]+=(HC*G9);G8[2]+=(HD*G9);G8[3]+=(HE
*G9);G_[0]+=(HB*HA);G_[1]+=(HC*HA);G_[2]+=(HD*HA);G_[3]+=(HE*HA);Gy+=4;G0+=4;G2
+=4;G4+=4;G6+=4;G8+=4;G_+=4;}_Pragma("GCC unroll 1")_Pragma("GCC novector")while
(Gy<Gz){float HB=Gy[0];asm(""::"r"(G0));G0[0]+=(HB*G1);G2[0]+=(HB*G3);G4[0]+=(HB
*G5);G6[0]+=(HB*G7);G8[0]+=(HB*G9);G_[0]+=(HB*HA);++Gy;++G0;++G2;++G4;++G6;++G8;
++G_;}}static void F9(float*Gw,float const*Gx,float const**Gy,float const*Gz){//
float*restrict G0=Gw;float const*G1=Gy[0];float G2=Gx[0];float const*G3=Gy[1];//
float G4=Gx[1];float const*G5=Gy[2];float G6=Gx[2];float const*G7=Gy[3];float G8
=Gx[3];float const*G9=Gy[4];float G_=Gx[4];float const*HA=Gy[5];float HB=Gx[5];
_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char*)Gz-(char*)G1)>=16){
float HC,HD,HE,HF;asm(""::"r"(G0));HC=G0[0]+G1[0]*G2;HD=G0[1]+G1[1]*G2;HE=G0[2]+
G1[2]*G2;HF=G0[3]+G1[3]*G2;HC+=G3[0]*G4;HD+=G3[1]*G4;HE+=G3[2]*G4;HF+=G3[3]*G4;
HC+=G5[0]*G6;HD+=G5[1]*G6;HE+=G5[2]*G6;HF+=G5[3]*G6;HC+=G7[0]*G8;HD+=G7[1]*G8;HE
+=G7[2]*G8;HF+=G7[3]*G8;HC+=G9[0]*G_;HD+=G9[1]*G_;HE+=G9[2]*G_;HF+=G9[3]*G_;HC+=
HA[0]*HB;HD+=HA[1]*HB;HE+=HA[2]*HB;HF+=HA[3]*HB;G0[0]=HC;G0[1]=HD;G0[2]=HE;G0[3]
=HF;G0+=4;G1+=4;G3+=4;G5+=4;G7+=4;G9+=4;HA+=4;}_Pragma("GCC unroll 1")_Pragma(//
"GCC novector")while(G1<Gz){float HC;asm(""::"r"(G0));HC=G0[0]+G1[0]*G2;HC+=G3[0
]*G4;HC+=G5[0]*G6;HC+=G7[0]*G8;HC+=G9[0]*G_;HC+=HA[0]*HB;G0[0]=HC;++G0;++G1;++G3
;++G5;++G7;++G9;++HA;}}static void F_(float**Gw,float const*Gx,float const*Gy,//
float const*Gz){float*restrict G0=Gw[0];float G1=Gx[0];float*restrict G2=Gw[1];
float G3=Gx[1];float*restrict G4=Gw[2];float G5=Gx[2];float*restrict G6=Gw[3];//
float G7=Gx[3];float*restrict G8=Gw[4];float G9=Gx[4];float*restrict G_=Gw[5];//
float HA=Gx[5];float*restrict HB=Gw[6];float HC=Gx[6];_Pragma("GCC unroll 1")///
_Pragma("GCC novector")while(((char*)Gz-(char*)Gy)>=16){float HD,HE,HF,HG;asm(""
::"r"(Gy));HD=Gy[0],HE=Gy[1],HF=Gy[2],HG=Gy[3];G0[0]=(HD*G1);G0[1]=(HE*G1);G0[2]
=(HF*G1);G0[3]=(HG*G1);G2[0]=(HD*G3);G2[1]=(HE*G3);G2[2]=(HF*G3);G2[3]=(HG*G3);
G4[0]=(HD*G5);G4[1]=(HE*G5);G4[2]=(HF*G5);G4[3]=(HG*G5);G6[0]=(HD*G7);G6[1]=(HE*
G7);G6[2]=(HF*G7);G6[3]=(HG*G7);G8[0]=(HD*G9);G8[1]=(HE*G9);G8[2]=(HF*G9);G8[3]=
(HG*G9);G_[0]=(HD*HA);G_[1]=(HE*HA);G_[2]=(HF*HA);G_[3]=(HG*HA);HB[0]=(HD*HC);HB
[1]=(HE*HC);HB[2]=(HF*HC);HB[3]=(HG*HC);Gy+=4;G0+=4;G2+=4;G4+=4;G6+=4;G8+=4;G_+=
4;HB+=4;}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(Gy<Gz){float HD=Gy[
0];asm(""::"r"(G0));G0[0]=(HD*G1);G2[0]=(HD*G3);G4[0]=(HD*G5);G6[0]=(HD*G7);G8[0
]=(HD*G9);G_[0]=(HD*HA);HB[0]=(HD*HC);++Gy;++G0;++G2;++G4;++G6;++G8;++G_;++HB;}}
static void GA(float*Gw,float const*Gx,float const**Gy,float const*Gz){float*///
restrict G0=Gw;float const*G1=Gy[0];float G2=Gx[0];float const*G3=Gy[1];float G4
=Gx[1];float const*G5=Gy[2];float G6=Gx[2];float const*G7=Gy[3];float G8=Gx[3];
float const*G9=Gy[4];float G_=Gx[4];float const*HA=Gy[5];float HB=Gx[5];float///
const*HC=Gy[6];float HD=Gx[6];_Pragma("GCC unroll 1")_Pragma("GCC novector")////
while(((char*)Gz-(char*)G1)>=16){float HE,HF,HG,HH;asm(""::"r"(G0));HE=G1[0]*G2;
HF=G1[1]*G2;HG=G1[2]*G2;HH=G1[3]*G2;HE+=G3[0]*G4;HF+=G3[1]*G4;HG+=G3[2]*G4;HH+=
G3[3]*G4;HE+=G5[0]*G6;HF+=G5[1]*G6;HG+=G5[2]*G6;HH+=G5[3]*G6;HE+=G7[0]*G8;HF+=G7
[1]*G8;HG+=G7[2]*G8;HH+=G7[3]*G8;HE+=G9[0]*G_;HF+=G9[1]*G_;HG+=G9[2]*G_;HH+=G9[3
]*G_;HE+=HA[0]*HB;HF+=HA[1]*HB;HG+=HA[2]*HB;HH+=HA[3]*HB;HE+=HC[0]*HD;HF+=HC[1]*
HD;HG+=HC[2]*HD;HH+=HC[3]*HD;G0[0]=HE;G0[1]=HF;G0[2]=HG;G0[3]=HH;G0+=4;G1+=4;G3
+=4;G5+=4;G7+=4;G9+=4;HA+=4;HC+=4;}_Pragma("GCC unroll 1")_Pragma("GCC novector"
)while(G1<Gz){float HE;asm(""::"r"(G0));HE=G1[0]*G2;HE+=G3[0]*G4;HE+=G5[0]*G6;HE
+=G7[0]*G8;HE+=G9[0]*G_;HE+=HA[0]*HB;HE+=HC[0]*HD;G0[0]=HE;++G0;++G1;++G3;++G5;
++G7;++G9;++HA;++HC;}}static void GB(float**Gw,float const*Gx,float const*Gy,///
float const*Gz){float*restrict G0=Gw[0];float G1=Gx[0];float*restrict G2=Gw[1];
float G3=Gx[1];float*restrict G4=Gw[2];float G5=Gx[2];float*restrict G6=Gw[3];//
float G7=Gx[3];float*restrict G8=Gw[4];float G9=Gx[4];float*restrict G_=Gw[5];//
float HA=Gx[5];float*restrict HB=Gw[6];float HC=Gx[6];_Pragma("GCC unroll 1")///
_Pragma("GCC novector")while(((char*)Gz-(char*)Gy)>=16){float HD,HE,HF,HG;asm(""
::"r"(Gy));HD=Gy[0],HE=Gy[1],HF=Gy[2],HG=Gy[3];G0[0]+=(HD*G1);G0[1]+=(HE*G1);G0[
2]+=(HF*G1);G0[3]+=(HG*G1);G2[0]+=(HD*G3);G2[1]+=(HE*G3);G2[2]+=(HF*G3);G2[3]+=(
HG*G3);G4[0]+=(HD*G5);G4[1]+=(HE*G5);G4[2]+=(HF*G5);G4[3]+=(HG*G5);G6[0]+=(HD*G7
);G6[1]+=(HE*G7);G6[2]+=(HF*G7);G6[3]+=(HG*G7);G8[0]+=(HD*G9);G8[1]+=(HE*G9);G8[
2]+=(HF*G9);G8[3]+=(HG*G9);G_[0]+=(HD*HA);G_[1]+=(HE*HA);G_[2]+=(HF*HA);G_[3]+=(
HG*HA);HB[0]+=(HD*HC);HB[1]+=(HE*HC);HB[2]+=(HF*HC);HB[3]+=(HG*HC);Gy+=4;G0+=4;
G2+=4;G4+=4;G6+=4;G8+=4;G_+=4;HB+=4;}_Pragma("GCC unroll 1")_Pragma(////////////
"GCC novector")while(Gy<Gz){float HD=Gy[0];asm(""::"r"(G0));G0[0]+=(HD*G1);G2[0]
+=(HD*G3);G4[0]+=(HD*G5);G6[0]+=(HD*G7);G8[0]+=(HD*G9);G_[0]+=(HD*HA);HB[0]+=(HD
*HC);++Gy;++G0;++G2;++G4;++G6;++G8;++G_;++HB;}}static void GC(float*Gw,float////
const*Gx,float const**Gy,float const*Gz){float*restrict G0=Gw;float const*G1=Gy[
0];float G2=Gx[0];float const*G3=Gy[1];float G4=Gx[1];float const*G5=Gy[2];float
G6=Gx[2];float const*G7=Gy[3];float G8=Gx[3];float const*G9=Gy[4];float G_=Gx[4]
;float const*HA=Gy[5];float HB=Gx[5];float const*HC=Gy[6];float HD=Gx[6];_Pragma
("GCC unroll 1")_Pragma("GCC novector")while(((char*)Gz-(char*)G1)>=16){float HE
,HF,HG,HH;asm(""::"r"(G0));HE=G0[0]+G1[0]*G2;HF=G0[1]+G1[1]*G2;HG=G0[2]+G1[2]*G2
;HH=G0[3]+G1[3]*G2;HE+=G3[0]*G4;HF+=G3[1]*G4;HG+=G3[2]*G4;HH+=G3[3]*G4;HE+=G5[0]
*G6;HF+=G5[1]*G6;HG+=G5[2]*G6;HH+=G5[3]*G6;HE+=G7[0]*G8;HF+=G7[1]*G8;HG+=G7[2]*
G8;HH+=G7[3]*G8;HE+=G9[0]*G_;HF+=G9[1]*G_;HG+=G9[2]*G_;HH+=G9[3]*G_;HE+=HA[0]*HB
;HF+=HA[1]*HB;HG+=HA[2]*HB;HH+=HA[3]*HB;HE+=HC[0]*HD;HF+=HC[1]*HD;HG+=HC[2]*HD;
HH+=HC[3]*HD;G0[0]=HE;G0[1]=HF;G0[2]=HG;G0[3]=HH;G0+=4;G1+=4;G3+=4;G5+=4;G7+=4;
G9+=4;HA+=4;HC+=4;}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(G1<Gz){//
float HE;asm(""::"r"(G0));HE=G0[0]+G1[0]*G2;HE+=G3[0]*G4;HE+=G5[0]*G6;HE+=G7[0]*
G8;HE+=G9[0]*G_;HE+=HA[0]*HB;HE+=HC[0]*HD;G0[0]=HE;++G0;++G1;++G3;++G5;++G7;++G9
;++HA;++HC;}}static void GD(float**Gw,float const*Gx,float const*Gy,float const*
Gz){float*restrict G0=Gw[0];float G1=Gx[0];float*restrict G2=Gw[1];float G3=Gx[1
];float*restrict G4=Gw[2];float G5=Gx[2];float*restrict G6=Gw[3];float G7=Gx[3];
float*restrict G8=Gw[4];float G9=Gx[4];float*restrict G_=Gw[5];float HA=Gx[5];//
float*restrict HB=Gw[6];float HC=Gx[6];float*restrict HD=Gw[7];float HE=Gx[7];//
_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char*)Gz-(char*)Gy)>=16){
float HF,HG,HH,HI;asm(""::"r"(Gy));HF=Gy[0],HG=Gy[1],HH=Gy[2],HI=Gy[3];G0[0]=(HF
*G1);G0[1]=(HG*G1);G0[2]=(HH*G1);G0[3]=(HI*G1);G2[0]=(HF*G3);G2[1]=(HG*G3);G2[2]
=(HH*G3);G2[3]=(HI*G3);G4[0]=(HF*G5);G4[1]=(HG*G5);G4[2]=(HH*G5);G4[3]=(HI*G5);
G6[0]=(HF*G7);G6[1]=(HG*G7);G6[2]=(HH*G7);G6[3]=(HI*G7);G8[0]=(HF*G9);G8[1]=(HG*
G9);G8[2]=(HH*G9);G8[3]=(HI*G9);G_[0]=(HF*HA);G_[1]=(HG*HA);G_[2]=(HH*HA);G_[3]=
(HI*HA);HB[0]=(HF*HC);HB[1]=(HG*HC);HB[2]=(HH*HC);HB[3]=(HI*HC);HD[0]=(HF*HE);HD
[1]=(HG*HE);HD[2]=(HH*HE);HD[3]=(HI*HE);Gy+=4;G0+=4;G2+=4;G4+=4;G6+=4;G8+=4;G_+=
4;HB+=4;HD+=4;}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(Gy<Gz){float
HF=Gy[0];asm(""::"r"(G0));G0[0]=(HF*G1);G2[0]=(HF*G3);G4[0]=(HF*G5);G6[0]=(HF*G7
);G8[0]=(HF*G9);G_[0]=(HF*HA);HB[0]=(HF*HC);HD[0]=(HF*HE);++Gy;++G0;++G2;++G4;++
G6;++G8;++G_;++HB;++HD;}}static void GE(float*Gw,float const*Gx,float const**Gy,
float const*Gz){float*restrict G0=Gw;float const*G1=Gy[0];float G2=Gx[0];float//
const*G3=Gy[1];float G4=Gx[1];float const*G5=Gy[2];float G6=Gx[2];float const*G7
=Gy[3];float G8=Gx[3];float const*G9=Gy[4];float G_=Gx[4];float const*HA=Gy[5];
float HB=Gx[5];float const*HC=Gy[6];float HD=Gx[6];float const*HE=Gy[7];float HF
=Gx[7];_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char*)Gz-(char*)G1)
>=16){float HG,HH,HI,HJ;asm(""::"r"(G0));HG=G1[0]*G2;HH=G1[1]*G2;HI=G1[2]*G2;HJ=
G1[3]*G2;HG+=G3[0]*G4;HH+=G3[1]*G4;HI+=G3[2]*G4;HJ+=G3[3]*G4;HG+=G5[0]*G6;HH+=G5
[1]*G6;HI+=G5[2]*G6;HJ+=G5[3]*G6;HG+=G7[0]*G8;HH+=G7[1]*G8;HI+=G7[2]*G8;HJ+=G7[3
]*G8;HG+=G9[0]*G_;HH+=G9[1]*G_;HI+=G9[2]*G_;HJ+=G9[3]*G_;HG+=HA[0]*HB;HH+=HA[1]*
HB;HI+=HA[2]*HB;HJ+=HA[3]*HB;HG+=HC[0]*HD;HH+=HC[1]*HD;HI+=HC[2]*HD;HJ+=HC[3]*HD
;HG+=HE[0]*HF;HH+=HE[1]*HF;HI+=HE[2]*HF;HJ+=HE[3]*HF;G0[0]=HG;G0[1]=HH;G0[2]=HI;
G0[3]=HJ;G0+=4;G1+=4;G3+=4;G5+=4;G7+=4;G9+=4;HA+=4;HC+=4;HE+=4;}_Pragma(////////
"GCC unroll 1")_Pragma("GCC novector")while(G1<Gz){float HG;asm(""::"r"(G0));HG=
G1[0]*G2;HG+=G3[0]*G4;HG+=G5[0]*G6;HG+=G7[0]*G8;HG+=G9[0]*G_;HG+=HA[0]*HB;HG+=HC
[0]*HD;HG+=HE[0]*HF;G0[0]=HG;++G0;++G1;++G3;++G5;++G7;++G9;++HA;++HC;++HE;}}////
static void GF(float**Gw,float const*Gx,float const*Gy,float const*Gz){float*///
restrict G0=Gw[0];float G1=Gx[0];float*restrict G2=Gw[1];float G3=Gx[1];float*//
restrict G4=Gw[2];float G5=Gx[2];float*restrict G6=Gw[3];float G7=Gx[3];float*//
restrict G8=Gw[4];float G9=Gx[4];float*restrict G_=Gw[5];float HA=Gx[5];float*//
restrict HB=Gw[6];float HC=Gx[6];float*restrict HD=Gw[7];float HE=Gx[7];_Pragma(
"GCC unroll 1")_Pragma("GCC novector")while(((char*)Gz-(char*)Gy)>=16){float HF,
HG,HH,HI;asm(""::"r"(Gy));HF=Gy[0],HG=Gy[1],HH=Gy[2],HI=Gy[3];G0[0]+=(HF*G1);G0[
1]+=(HG*G1);G0[2]+=(HH*G1);G0[3]+=(HI*G1);G2[0]+=(HF*G3);G2[1]+=(HG*G3);G2[2]+=(
HH*G3);G2[3]+=(HI*G3);G4[0]+=(HF*G5);G4[1]+=(HG*G5);G4[2]+=(HH*G5);G4[3]+=(HI*G5
);G6[0]+=(HF*G7);G6[1]+=(HG*G7);G6[2]+=(HH*G7);G6[3]+=(HI*G7);G8[0]+=(HF*G9);G8[
1]+=(HG*G9);G8[2]+=(HH*G9);G8[3]+=(HI*G9);G_[0]+=(HF*HA);G_[1]+=(HG*HA);G_[2]+=(
HH*HA);G_[3]+=(HI*HA);HB[0]+=(HF*HC);HB[1]+=(HG*HC);HB[2]+=(HH*HC);HB[3]+=(HI*HC
);HD[0]+=(HF*HE);HD[1]+=(HG*HE);HD[2]+=(HH*HE);HD[3]+=(HI*HE);Gy+=4;G0+=4;G2+=4;
G4+=4;G6+=4;G8+=4;G_+=4;HB+=4;HD+=4;}_Pragma("GCC unroll 1")_Pragma(////////////
"GCC novector")while(Gy<Gz){float HF=Gy[0];asm(""::"r"(G0));G0[0]+=(HF*G1);G2[0]
+=(HF*G3);G4[0]+=(HF*G5);G6[0]+=(HF*G7);G8[0]+=(HF*G9);G_[0]+=(HF*HA);HB[0]+=(HF
*HC);HD[0]+=(HF*HE);++Gy;++G0;++G2;++G4;++G6;++G8;++G_;++HB;++HD;}}static void//
GG(float*Gw,float const*Gx,float const**Gy,float const*Gz){float*restrict G0=Gw;
float const*G1=Gy[0];float G2=Gx[0];float const*G3=Gy[1];float G4=Gx[1];float///
const*G5=Gy[2];float G6=Gx[2];float const*G7=Gy[3];float G8=Gx[3];float const*G9
=Gy[4];float G_=Gx[4];float const*HA=Gy[5];float HB=Gx[5];float const*HC=Gy[6];
float HD=Gx[6];float const*HE=Gy[7];float HF=Gx[7];_Pragma("GCC unroll 1")//////
_Pragma("GCC novector")while(((char*)Gz-(char*)G1)>=16){float HG,HH,HI,HJ;asm(""
::"r"(G0));HG=G0[0]+G1[0]*G2;HH=G0[1]+G1[1]*G2;HI=G0[2]+G1[2]*G2;HJ=G0[3]+G1[3]*
G2;HG+=G3[0]*G4;HH+=G3[1]*G4;HI+=G3[2]*G4;HJ+=G3[3]*G4;HG+=G5[0]*G6;HH+=G5[1]*G6
;HI+=G5[2]*G6;HJ+=G5[3]*G6;HG+=G7[0]*G8;HH+=G7[1]*G8;HI+=G7[2]*G8;HJ+=G7[3]*G8;
HG+=G9[0]*G_;HH+=G9[1]*G_;HI+=G9[2]*G_;HJ+=G9[3]*G_;HG+=HA[0]*HB;HH+=HA[1]*HB;HI
+=HA[2]*HB;HJ+=HA[3]*HB;HG+=HC[0]*HD;HH+=HC[1]*HD;HI+=HC[2]*HD;HJ+=HC[3]*HD;HG+=
HE[0]*HF;HH+=HE[1]*HF;HI+=HE[2]*HF;HJ+=HE[3]*HF;G0[0]=HG;G0[1]=HH;G0[2]=HI;G0[3]
=HJ;G0+=4;G1+=4;G3+=4;G5+=4;G7+=4;G9+=4;HA+=4;HC+=4;HE+=4;}_Pragma(/////////////
"GCC unroll 1")_Pragma("GCC novector")while(G1<Gz){float HG;asm(""::"r"(G0));HG=
G0[0]+G1[0]*G2;HG+=G3[0]*G4;HG+=G5[0]*G6;HG+=G7[0]*G8;HG+=G9[0]*G_;HG+=HA[0]*HB;
HG+=HC[0]*HD;HG+=HE[0]*HF;G0[0]=HG;++G0;++G1;++G3;++G5;++G7;++G9;++HA;++HC;++HE;
}}typedef void GH(float*output,float const*coeffs,float const**inputs,float/////
const*input0_end);static GH*GI[8]={Fn,Fr,Fv,Fz,F3,F7,GA,GE};static GH*GJ[8]={Fp,
Ft,Fx,F1,F5,F9,GC,GG};typedef void GK(float**outputs,float const*coeffs,float///
const*input,float const*input_end);static GK*GL[8]={Fm,Fq,Fu,Fy,F2,F6,F_,GD};///
static GK*GM[8]={Fo,Fs,Fw,F0,F4,F8,GB,GF};static void GN(B0 const*Gw,void*Gx,///
float*Gy,int Gz){int G0=Gw->A.E.B;int G1=Gw->d;int G2=G0*G1;void*G3;if(Gw->T)Gw
->T(Gy,G2);G3=Gx;if(Gw->M)G3=Gy;Gw->U(G3,G2,Gy);if(Gw->M)Gw->M(G3,G0,Gz,Gw->L);}
static float*GO(B0 const*Gw,B_ const*Gx,int Gy){return(float*)(((char*)Gx->I)+(
Gy*Gw->G));}static float*GP(B0 const*Gw,B_ const*Gx,int Gy){int Gz=(Gx->D+(Gy-Gx
->B))%Gw->H;return GO(Gw,Gx,Gz);}static void GQ(B0 const*Gw,float*Gx,float const
*Gy){float const*Gz=Gy-(Gw->N.A.A*Gw->e);if((Gw->A.G==STBIR_FILTER_POINT_SAMPLE)
&&(Gw->A.E.C==1.0f))memcpy(Gx,Gy,Gw->A.E.B*4*Gw->e);else Gw->S(Gx,Gw->A.E.B,Gz,
Gw->A.A,Gw->A.B,Gw->A.K);}static void GR(B0 const*Gw,B_*Gx,int Gy,int Gz,int G0,
float const*G1){float*G2=Gx->J;float*G3=Gx->A;int G4=Gw->c;int G5=G4?(Gw->N.A.B-
Gw->N.A.A+1):Gw->A.E.B;int G6=Gw->e*G5;{int G7=0,G8=G0-Gz+1;do{float const*G9[8]
;int G_,HA=G8;if(HA>8)HA=8;for(G_=0;G_<HA;G_++)G9[G_]=GP(Gw,Gx,G7+G_+Gz);((G7==0
)?GI:GJ)[HA-1](G4?G3:G2,G1+G7,G9,G9[0]+G6);G7+=HA;G8-=HA;}while(G8);}if(G4){G3[
G6]=0.0f;G3[G6+1]=0.0f;GQ(Gw,G2,G3);}GN(Gw,((char*)Gw->D)+((size_t)Gy*(size_t)Gw
->F),G2,Gy);}static void GS(B0 const*Gw,B_*Gx,int Gy){int Gz;float*G0;EK(Gw,Gy,
Gx->A);Gx->C=Gy;Gz=(Gx->D+(Gx->C-Gx->B))%Gw->H;G0=GO(Gw,Gx,Gz);GQ(Gw,G0,Gx->A);}
static void GT(B0 const*Gw,B_*Gx,int Gy){int Gz,G0,G1;B4*G2=Gw->B.A;float const*
G3=Gw->B.B;G0=Gx->E;G1=Gx[Gy-1].F;G2+=G0;G3+=G0*Gw->B.K;Gx->D=0;Gx->B=G2->A;Gx->
C=Gx->B-1;for(Gz=G0;Gz<G1;Gz++){int G4,G5;G4=G2->A;G5=G2->B;while(G5>Gx->C){if((
Gx->C-Gx->B+1)==Gw->H){Gx->B++;Gx->D++;}if(Gw->c){float*G6=GP(Gw,Gx,++Gx->C);EK(
Gw,Gx->C,G6);}else GS(Gw,Gx,Gx->C+1);}GR(Gw,Gx,Gz,G4,G5,G3);++G2;G3+=Gw->B.K;}}
static void GU(B0 const*Gw,B_*Gx){float*Gy=GO(Gw,Gx,Gx->D);GN(Gw,((char*)Gw->D)+
((size_t)Gx->B*(size_t)Gw->F),Gy,Gx->B);Gy[0]=3e38;Gx->B++;if(++Gx->D==Gw->H)Gx
->D=0;}static void GV(B0 const*Gw,B_*Gx){float*Gy=GO(Gw,Gx,Gx->D);GQ(Gw,Gx->J,Gy
);GN(Gw,((char*)Gw->D)+((size_t)Gx->B*(size_t)Gw->F),Gx->J,Gx->B);Gy[0]=3e38;Gx
->B++;if(++Gx->D==Gw->H)Gx->D=0;}static void GW(B0 const*Gw,B_*Gx,int Gy,int Gz,
float const*G0,float const*G1,float const*G2){{int G3=0,G4=Gz-Gy+1;do{float*G5[8
];int G6,G7=G4;if(G7>8)G7=8;for(G6=0;G6<G7;G6++){G5[G6]=GP(Gw,Gx,G3+G6+Gy);if(G6
&&((G5[G6][0]==3e38f)!=(G5[0][0]==3e38f))){G7=G6;break;}}(G5[0][0]==3e38f?GL:GM)
[G7-1](G5,G0+G3,G1,G2);G3+=G7;G4-=G7;}while(G4);}}typedef void GX(B0 const*/////
stbir_info,B_*split_info);static void GY(B0 const*Gw,B_*Gx,int Gy){int Gz,G0,G1,
G2,G3;B4*G4=Gw->B.A;float const*G5=Gw->B.B;GX*G6;void*G7;void*G8;int G9,G_;int//
HA=Gw->c?(Gw->N.A.B-Gw->N.A.A+1):Gw->A.E.B;int HB=Gw->e*HA;G0=Gx->E;G1=Gx[Gy-1].
F;G2=Gx->G;G3=Gx[Gy-1].H;Gz=G2+Gw->B.M;G4+=Gz;G5+=Gw->B.K*Gz;if(Gw->c){G6=GV;G7=
Gx->A;G8=((char*)G7)+4*Gw->e*(Gw->N.A.B-Gw->N.A.A+1);}else{G6=GU;G7=Gx->J;G8=((
char*)G7)+4*Gw->e*Gw->A.E.B;}Gx->B=G0;Gx->C=-1;Gx->D=-1;for(Gz=0;Gz<Gw->H;Gz++){
float*HC=GO(Gw,Gx,Gz);HC[HB]=0.0f;HC[HB+1]=0.0f;HC[0]=3e38;}G9=1;G_=G2;for(Gz=G2
;Gz<G3;Gz++){int HC,HD;HC=G4->A;HD=G4->B;if((HD>=HC)&&(((HC>=G0)&&(HC<G1))||((HD
>=G0)&&(HD<G1)))){float const*HE=G5;G_=Gz;if(G9&&(Gz>G2))Gx->G=Gz;G9=0;if(HC<G0)
{HE+=G0-HC;HC=G0;}if(HD>=G1)HD=G1-1;if(Gx->D<0)Gx->D=HC-G0;EK(Gw,Gz,Gx->A);if(!
Gw->c)GQ(Gw,Gx->J,Gx->A);if(((Gx->C-Gx->B+1)==Gw->H)&&(HD>Gx->C))G6(Gw,Gx);GW(Gw
,Gx,HC,HD,HE,(float*)G7,(float*)G8);if(HD>Gx->C)Gx->C=HD;}++G4;G5+=Gw->B.K;}////
while(Gx->B<G1)G6(Gw,Gx);++G_;for(Gz=0;Gz<Gy;Gz++)if(Gx[Gz].H>G_)Gx[Gz].H=G_;}//
static By*GZ[]={0,CP,CR,CT,CU,CV,CS};static Bz*Ga[]={0,CQ,CX,CZ,CZ,CZ,CW};static
void Gb(B8*Gw,Bu Gx,By*Gy,Bz*Gz,Bt G0,B7*G1,int G2,void*G3){if(Gx==0){Gx=///////
STBIR_FILTER_MITCHELL;if(G1->C>=1.0)if((G1->C<=1.0)&&(((float)ceil((float)G1->E)
)==G1->E))Gx=STBIR_FILTER_POINT_SAMPLE;else Gx=STBIR_FILTER_CATMULLROM;}Gw->G=Gx
;Gw->H=GZ[Gx];Gw->I=Ga[Gx];if(Gy&&Gz){Gw->H=Gy;Gw->I=Gz;Gw->G=STBIR_FILTER_OTHER
;}Gw->J=G0;Gw->L=Ca(Gw->I,G1->C,G3);Gw->R=0;if(G1->C>=1.0)Gw->R=1;else if(G2||(
Gw->L<=32))Gw->R=2;Gw->K=Cb(Gw,Gw->R,G3);if(G0==STBIR_EDGE_WRAP)if(Gw->L>(G1->A*
3))Gw->L=G1->A*3;Gw->M=Gw->L/2;if(G0==STBIR_EDGE_WRAP)if(Gw->M>G1->A)Gw->M=G1->A
;Gw->N=Cc(Gw,Gw->R);Gw->O=Gw->N*8;Gw->P=Gw->N*Gw->K*4+12;Gw->C=0;Gw->D=0;if(Gw->
R==0){Gw->T=Gw->L;Gw->S=Cc(Gw,2);Gw->U=Gw->S*8;Gw->V=Gw->S*Gw->T*4;}}static void
Gc(B8*Gw,B4*Gx,void*Gy){float Gz=Gw->E.C;float G0=Gw->E.E;Bz*G1=Gw->I;int G2=Gw
->E.A;Bt G3=Gw->J;float G4=Gw->E.D;if(Gw->R==1){int G5,G6;float G7=G1(G4,Gy)*Gz;
Cl(&G5,&G6,0.5,G7,G4,G0,G2,G3);Gx->A=G5;Cl(&G5,&G6,((float)(Gw->E.B-1))+0.5f,G7,
G4,G0,G2,G3);Gx->B=G6;}else if(Gw->R==2){float G5=G1(Gz,Gy)*G4;int G6=Gw->M;int
G7=Gw->E.B;int G8;int G9;int G_,HA;Cl(&G_,&HA,0,0,G4,G0,G2,G3);Gx->A=G_;Cl(&G_,&
HA,(float)G7,0,G4,G0,G2,G3);Gx->B=HA;G9=Gx->A+1;G8=-G6;while(G9>=G8){int HB,HC;
Co(&HB,&HC,((float)G9)+0.5f,G5,Gz,G0,G7);if(HB>HC)break;if((HB<G7)||(HC>=0))Gx->
A=G9;--G9;}G9=Gx->B-1;G8=G9+1+G6;while(G9<=G8){int HB,HC;Co(&HB,&HC,((float)G9)+
0.5f,G5,Gz,G0,G7);if(HB>HC)break;if((HB<G7)||(HC>=0))Gx->B=G9;++G9;}}if(Gw->J==
STBIR_EDGE_WRAP){if((Gx->A>0)&&(Gx->B>=G2)){int G5=Gx->B-G2+1;if((G5+16)>=Gx->A)
Gx->A=0;}if((Gx->A<0)&&(Gx->B<(G2-1))){int G5=-Gx->A;if((G2-G5-16-1)<=Gx->B)Gx->
B=G2-1;}}else{if(Gx->A<0)Gx->A=0;if(Gx->B>=G2)Gx->B=G2-1;}}static void Gd(B_*Gw,
int Gx,int Gy,int Gz,int G0,int G1,B4*G2){int G3,G4;int G5=Gy;G4=0;for(G3=0;G3<
Gx;G3++){int G6;Gw[G3].E=G4;G6=G5/(Gx-G3);Gw[G3].F=G4+G6;if(G1&&G3){B4*G7;int G8
,G9,G_,HA;B4*HB=G2+G4;G_=Gz*3;if(G6<G_)G_=G6;G9=0;G7=HB;HA=G7->A;for(G8=1;G8<=G_
;G8++){++HB;if(HB->A>HA)break;if(HB->A<G7->A){G7=HB;G9=G8;}}Gw[G3-1].F+=G9;Gw[G3
].E+=G9;}G4+=G6;G5-=G6;Gw[G3].G=-Gz;Gw[G3].H=G0+Gz;}}static void Ge(B0*Gw){if(Gw
)if(Gw->O){void*Gx=Gw->O;Gw->O=0;((void)Gw->L,free(Gx));}}static int Gf(int Gw,
int Gx){int Gy;int Gz=0;for(Gy=0;Gy<Gw;Gy++){int G0=Gx/(Gw-Gy);if(G0>Gz)Gz=G0;Gx
-=G0;}return Gz;}static CC**Gg[8]={0,Eb,Et,FA,FS,0,0,Fk};static CC**Gh[8]={0,Ec,
Eu,FB,FT,0,0,Fl};static float Gi[5][8][4]={{{1.0,1.0,0.3125,1.0},{0.5625,///////
0.59375f,0.0,0.96875f},{1.0,0.0625,0.0,1.0},{0.0,0.09375f,1.0,1.0},{1.0,1.0,////
0.3125,1.0},{0.03125f,0.125,1.0,1.0},{1.0,1.0,0.0625,1.0},{0.0,1.0,0.0,0.03125f}
,},{{0.0,0.84375f,0.0,0.03125f},{0.09375f,0.9375,0.0,0.78125f},{0.875,0.21875f,
0.0,0.96875f},{0.09375f,0.09375f,1.0,1.0},{0.0,0.84375f,0.0,0.03125f},{0.03125f,
0.125,1.0,1.0},{1.0,1.0,0.0625,1.0},{0.0,1.0,0.0,0.53125f},},{{0.0,0.53125f,0.0,
0.03125f},{0.0625,0.96875f,0.0,0.53125f},{0.875,0.1875,0.0,0.9375},{0.0,0.09375f
,1.0,1.0},{0.0,0.53125f,0.0,0.03125f},{0.03125f,0.125,1.0,1.0},{1.0,1.0,0.0625,
1.0},{0.0,1.0,0.0,0.5625},},{{0.0,0.5,0.0,0.71875f},{0.0625,0.84375f,0.0,0.875},
{1.0,0.5,0.5,0.96875f},{1.0,0.09375f,0.3125,0.5},{0.0,0.5,0.0,0.71875f},{1.0,///
0.03125f,0.03125f,0.53125f},{1.0,1.0,0.0625,1.0},{0.0,1.0,0.03125f,0.1875},},{{
0.0,0.59375f,0.0,0.96875f},{0.0625,0.8125,0.0625,0.59375f},{0.75,0.4375,0.125,//
0.96875f},{0.875,0.0625,0.1875,0.4375},{0.0,0.59375f,0.0,0.96875f},{0.15625f,///
0.125,1.0,1.0},{1.0,1.0,0.0625,1.0},{0.0,1.0,0.03125f,0.34375f},}};typedef//////
struct E{double A,B;int C;int D;int E;int F;}Gj;static int Gk(float Gw[8][4],int
Gx,float Gy,int Gz,int G0,float G1,int G2,int G3,Gj*G4){double G5,G6;float*G7;//
int G8;int G9;if((G2<=4)||(Gz<=4))G9=(G2<Gz)?6:7;else if((!G3)&&((G2<=16)||(Gz<=
16)))G9=4;else if(G1<=1.0f)G9=G3?1:0;else if(G1<=2.0f)G9=2;else if(G1<=3.0f)G9=3
;else G9=5;G7=Gw[G9];G6=(float)Gx*G7[0]+Gy*(float)G0*G7[1];G5=(float)G0*G7[2]+G1
*(float)Gx*G7[3];G8=(G5<=G6)?1:0;if(G4){G4->B=G6;G4->A=G5;G4->E=G9;G4->D=G8;G4->
F=G3;}if(G4&&G4->C)G8=(G4->C==2)?1:0;return G8;}static unsigned char Gl[]={1,2,3
,3,4,4,4,4,4,2,2,4,4,4,4,2,2,};static B2 Gm[]={STBIRI_BGR,STBIRI_1CHANNEL,//////
STBIRI_2CHANNEL,STBIRI_RGB,STBIRI_RGBA,STBIRI_4CHANNEL,STBIRI_BGRA,STBIRI_ARGB,
STBIRI_ABGR,STBIRI_RA,STBIRI_AR,STBIRI_RGBA_PM,STBIRI_BGRA_PM,STBIRI_ARGB_PM,///
STBIRI_ABGR_PM,STBIRI_RA_PM,STBIRI_AR_PM,};static B0*Gn(B8*Gw,B8*Gx,B4*Gy,Bs Gz,
Bs G0,int G1,int G2,int G3,int G4,void*G5){static char G6[8]={9,0,1,2,3,9,9,4};
B0*G7=0;void*G8=0;size_t G9=0;int G_;size_t HA,HB,HC,HD;int HE;int HF=0;int HG=
Gf(G1,Gx->E.B);B2 HH=Gm[Gz];B2 HI=Gm[G0];int HJ=Gl[HH];int HK=HJ;if((Gw->G!=////
STBIR_FILTER_POINT_SAMPLE)||(Gx->G!=STBIR_FILTER_POINT_SAMPLE))if((HH>=/////////
STBIRI_RGBA)&&(HH<=STBIRI_AR)&&(HI>=STBIRI_RGBA)&&(HI<=STBIRI_AR)){if(G4)HF=4;//
else{static int HL[6]={7,7,7,7,3,3};HF=2;HK=HL[HH-STBIRI_RGBA];}}else if((HH>=//
STBIRI_RGBA_PM)&&(HH<=STBIRI_AR_PM)&&(HI>=STBIRI_RGBA)&&(HI<=STBIRI_AR))HF=3;///
else if((HH>=STBIRI_RGBA)&&(HH<=STBIRI_AR)&&(HI>=STBIRI_RGBA_PM)&&(HI<=/////////
STBIRI_AR_PM))HF=1;if(HJ!=Gl[HI])return 0;G_=Gk(Gi[(int)G6[HK]],Gw->L,Gw->E.C,Gw
->E.B,Gx->L,Gx->E.C,Gx->E.B,Gx->R,0);HA=(Gy->B-Gy->A+1)*HK*4+12;HB=(size_t)Gw->E
.B*(size_t)HK*4+12;if(G_)HB=(HA+15)&~15;if((HB&4095)==0)HB+=192;HE=Gx->L+1;if((!
Gx->R)&&(HE>HG))HE=HG;HC=(size_t)HE*(size_t)HB;HD=(size_t)Gw->E.B*(size_t)HK*4+4
;for(;;){int HL;void*HM=G8;int HN=0;B8*HO=0;HM=(void*)((((size_t)HM)+15)&~15);if
(G8)G7=(B0*)HM;HM=(char*)(((size_t)HM)+512);HM=(void*)((((size_t)HM)+15)&~15);if
(G8)G7->P=(B_*)HM;HM=(char*)(((size_t)HM)+(120*G1));if(G7){static CB*HP[6]={EB,
EB,EB,EB,EC,EC};static CD*HQ[6]={ED,ED,ED,ED,EE,EE};static CB*HR[6]={EF,EF,EF,EF
,EG,EG};static CD*HS[6]={EH,EH,EH,EH,EI,EI};G7->O=G8;G7->f=G9;G7->d=HJ;G7->e=HK;
G7->a=G2;G7->b=G3;G7->V=(int)HE;G7->H=0;G7->G=(int)HB;G7->W=G1;G7->c=G_;G7->X=HH
;G7->Y=HI;G7->R=0;G7->T=0;if(HF==2){G7->R=HP[HH-STBIRI_RGBA];G7->T=HQ[HI-///////
STBIRI_RGBA];}else if(HF==4){G7->R=HR[HH-STBIRI_RGBA];G7->T=HS[HI-STBIRI_RGBA];}
else if(HF==1)G7->R=HR[HH-STBIRI_RGBA];else if(HF==3)G7->T=HS[HI-STBIRI_RGBA];if
(((HH==STBIRI_RGB)&&(HI==STBIRI_BGR))||((HH==STBIRI_BGR)&&(HI==STBIRI_RGB)))if(
Gw->E.C<1.0f)G7->T=EJ;else G7->R=EJ;}for(HL=0;HL<G1;HL++){HM=(void*)((((size_t)
HM)+15)&~15);if(G8)G7->P[HL].A=(float*)HM;HM=(char*)(((size_t)HM)+HA);HM=(void*)
((((size_t)HM)+15)&~15);if(G8)G7->P[HL].I=(float*)HM;HM=(char*)(((size_t)HM)+HC)
;HM=(void*)((((size_t)HM)+15)&~15);if(G8)G7->P[HL].J=(float*)HM;HM=(char*)(((///
size_t)HM)+HD);}if(Gx->R==0){size_t HP;size_t HQ;HP=(size_t)Gx->U+(size_t)Gx->V;
HQ=(size_t)(HA+HC+HD)*(size_t)G1;if(HQ>=HP){if(G7){Gx->C=(B4*)G7->P[0].A;Gx->D=(
float*)(((char*)G7->P[0].A)+Gx->U);}}else{HM=(void*)((((size_t)HM)+15)&~15);if(
G8)Gx->C=(B4*)HM;HM=(char*)(((size_t)HM)+Gx->U);HM=(void*)((((size_t)HM)+15)&~15
);if(G8)Gx->D=(float*)HM;HM=(char*)(((size_t)HM)+Gx->V);}}HM=(void*)((((size_t)
HM)+15)&~15);if(G8)Gw->A=(B4*)HM;HM=(char*)(((size_t)HM)+Gw->O);HM=(void*)((((//
size_t)HM)+15)&~15);if(G8)Gw->B=(float*)HM;HM=(char*)(((size_t)HM)+Gw->P);if((Gw
->H==Gx->H)&&(Gw->I==Gx->I)&&(Gw->J==Gx->J)&&(Gw->E.B==Gx->E.B)){float HP=Gw->E.
C-Gx->E.C;float HQ=Gw->E.E-Gx->E.E;if(HP<0.0f)HP=-HP;if(HQ<0.0f)HQ=-HQ;if((HP<=
7.523e-37)&&(HQ<=7.523e-37)){if(Gw->R==Gx->R){HN=1;goto no_vert_alloc;}HO=Gw;}}
HM=(void*)((((size_t)HM)+15)&~15);if(G8)Gx->A=(B4*)HM;HM=(char*)(((size_t)HM)+Gx
->O);HM=(void*)((((size_t)HM)+15)&~15);if(G8)Gx->B=(float*)HM;HM=(char*)(((/////
size_t)HM)+Gx->P);no_vert_alloc:if(G7){Cs(Gw,0,G5);G7->S=Gg[HK][Gw->Q.C&3];if(Gw
->Q.C<=12)G7->S=Gh[HK][Gw->Q.C-1];G7->N.A.A=Gy->A;G7->N.A.B=Gy->B;Ck(Gw,&G7->N);
Gw->K=Cr(Gw->N,Gw->A,Gw->B,Gw->K,Gw->Q.C,G7->N.A.A,G7->N.A.B);memcpy(&G7->A,Gw,
152);if(HN)memcpy(&G7->B,Gw,152);else{Cs(Gx,HO,G5);memcpy(&G7->B,Gx,152);}Gd(G7
->P,G7->W,G7->B.E.B,G7->B.M,G7->B.E.A,G7->B.R,G7->B.A);G7->H=G7->B.Q.C;if((!G7->
B.R)&&(G7->H>HG))G7->H=HG;}if(G7==0){G9=(15+(size_t)HM);G8=((void)G5,malloc(G9))
;if(G8==0)return 0;}else return G7;}}static int Go(B0 const*Gw,int Gx,int Gy){B_
*Gz=Gw->P+Gx;if(Gw->B.R)GT(Gw,Gz,Gy);else GY(Gw,Gz,Gy);return 1;}static void Gp(
B0*Gw,B1*Gx){static CA*Gy[5]={Cx,Cx,0,C9,C7,};static CA*Gz[6][5]={{Cz,Cx,0,C9,C7
},{DG,DE,0,DO,DM},{DW,DU,0,De,Dc},{Dm,Dk,0,Du,Ds},{C1,Cx,0,C9,C7},{D2,D0,0,D_,D8
},};static CA*G0[2][2]={{Ct,Cv},{C3,C5},};static CA*G1[6][2][2]={{{Ct,Cv},{C3,C5
}},{{DA,DC},{DI,DK}},{{DQ,DS},{DY,Da}},{{Dg,Di},{Do,Dq}},{{Ct,Cv},{C3,C5}},{{Dw,
Dy},{D4,D6}}};static CE*G2[5]={Cy,Cy,0,C_,C8,};static CE*G3[6][5]={{C0,Cy,0,C_,
C8},{DH,DF,0,DP,DN},{DX,DV,0,Df,Dd},{Dn,Dl,0,Dv,Dt},{C2,Cy,0,C_,C8},{D3,D1,0,EA,
D9}};static CE*G4[2][2]={{Cu,Cw},{C4,C6},};static CE*G5[6][2][2]={{{Cu,Cw},{C4,
C6}},{{DB,DD},{DJ,DL}},{{DR,DT},{DZ,Db}},{{Dh,Dj},{Dp,Dr}},{{Cu,Cw},{C4,C6}},{{
Dx,Dz},{D5,D7}}};CA*G6=0;CE*G7=0;Bv G8,G9;G8=Gx->Z;G9=Gx->a;Gw->C=Gx->B;Gw->E=Gx
->R;Gw->F=Gx->S;if((Gw->A.G==STBIR_FILTER_POINT_SAMPLE)&&(Gw->B.G==/////////////
STBIR_FILTER_POINT_SAMPLE))if(((G8==STBIR_TYPE_UINT8_SRGB)||(G8==///////////////
STBIR_TYPE_UINT8_SRGB_ALPHA))&&((G9==STBIR_TYPE_UINT8_SRGB)||(G9==//////////////
STBIR_TYPE_UINT8_SRGB_ALPHA))){G8=STBIR_TYPE_UINT8;G9=STBIR_TYPE_UINT8;}if(Gw->E
==0)Gw->E=Gw->d*Gw->A.E.A*B3[G8];if(Gw->F==0)Gw->F=Gw->d*Gw->A.E.B*B3[G9];Gw->D=
((char*)Gx->J)+((size_t)Gw->b*(size_t)Gx->S)+(Gw->a*Gw->d*B3[G9]);Gw->K=Gx->I;Gw
->L=Gx->A;Gw->M=Gx->Q;if((G8==STBIR_TYPE_UINT8)||(G8==STBIR_TYPE_UINT16)){int G_
=0;if((!Gw->R)&&(!Gw->T))if(((G8==STBIR_TYPE_UINT8)&&(G9==STBIR_TYPE_UINT8))||((
G8==STBIR_TYPE_UINT16)&&(G9==STBIR_TYPE_UINT16)))G_=1;if(Gw->X<=STBIRI_4CHANNEL)
G6=G0[G8==STBIR_TYPE_UINT16][G_];else G6=G1[(Gw->X-STBIRI_RGBA)%6][G8==/////////
STBIR_TYPE_UINT16][G_];}else if(Gw->X<=STBIRI_4CHANNEL)G6=Gy[G8-////////////////
STBIR_TYPE_UINT8_SRGB];else G6=Gz[(Gw->X-STBIRI_RGBA)%6][G8-////////////////////
STBIR_TYPE_UINT8_SRGB];if((G9==STBIR_TYPE_UINT8)||(G9==STBIR_TYPE_UINT16)){int//
G_=0;if((!Gw->R)&&(!Gw->T))if(((G8==STBIR_TYPE_UINT8)&&(G9==STBIR_TYPE_UINT8))||
((G8==STBIR_TYPE_UINT16)&&(G9==STBIR_TYPE_UINT16)))G_=1;if(Gw->Y<=//////////////
STBIRI_4CHANNEL)G7=G4[G9==STBIR_TYPE_UINT16][G_];else G7=G5[(Gw->Y-STBIRI_RGBA)%
6][G9==STBIR_TYPE_UINT16][G_];}else if(Gw->Y<=STBIRI_4CHANNEL)G7=G2[G9-/////////
STBIR_TYPE_UINT8_SRGB];else G7=G3[(Gw->Y-STBIRI_RGBA)%6][G9-////////////////////
STBIR_TYPE_UINT8_SRGB];Gw->I=G8;Gw->J=G9;Gw->Q=G6;Gw->U=G7;}static void Gq(int*
Gw,int*Gx,int Gy,double*Gz,double*G0){double G1,G2;int G3;if(*Gw<0){G1=((double)
*Gw)/((double)*Gx);G2=G1*(*G0-*Gz);*Gz-=G2;*Gw=0;}G3=Gy-(*Gw+*Gx);if(G3<0){G1=((
double)G3)/((double)*Gx);G2=G1*(*G0-*Gz);*G0+=G2;*Gx=Gy-*Gw;}}static int Gr(////
double Gw,Bq Gx,Bq*Gy,Bq*Gz,int G0){double G1;Br G2,G3;Br G4=0;Br G5=1;Br G6=1;
Br G7=0;G2=(Br)(Gw*33554432.0);G3=1<<25;for(;;){Br G8,G9;if((G0?G7:G6)>=Gx)break
;if(G7){G1=((double)G6/(double)G7)-Gw;if(G1<0.0)G1=-G1;if(G1<5.96e-08){*Gy=(Bq)
G6;*Gz=(Bq)G7;return 1;}}if(G3==0)break;G8=G2/G3;G9=G2%G3;G2=G3;G3=G9;G9=G8*G7+
G5;G5=G7;G7=G9;G9=G8*G6+G4;G4=G6;G6=G9;}if(G0){G6=(Br)(Gw*(double)Gx+0.5);G7=Gx;
}else{G6=Gx;G7=(Br)(((double)Gx/Gw)+0.5);}*Gy=(Bq)G6;*Gz=(Bq)G7;G1=G7?(((double)
(Bq)G6/(double)(Bq)G7)-Gw):1.0;if(G1<0.0)G1=-G1;return(G1<5.96e-08)?1:0;}static
int Gs(B7*Gw,int Gx,int*Gy,int Gz,int G0,double G1,double G2){double G3,G4,G5,G6
,G7,G8;G6=G2-G1;if((Gx==0)||(G0==0)||(Gz==0)||(G6<=7.523e-37))return 0;if((*Gy>=
Gx)||((*Gy+Gz)<=0)||(G1>=1.0)||(G2<=7.523e-37))return 0;G3=(double)Gx;G4=(double
)G0;G5=((double)Gz)/G3;G7=G5/G6;G8=(G3/G4)*G7;Gw->C=(float)G8;Gw->D=(float)(1.0/ 
G8);Gq(Gy,&Gz,Gx,&G1,&G2);G6=G2-G1;if(G6<=7.523e-37)return 0;Gw->E=(float)(G1*G7
*G3);Gw->F=Gr(G8,(G8<=1.0)?Gx:G0,&Gw->G,&Gw->H,G8>=1.0);Gw->A=G0;Gw->B=Gz;return
1;}static void Gt(B1*Gw,Bs Gx,Bv Gy){Gw->I=0;Gw->Q=0;Gw->A=Gw;Gw->j=0;Gw->W=0;Gw
->b=STBIR_FILTER_DEFAULT;Gw->f=0;Gw->g=0;Gw->c=STBIR_FILTER_DEFAULT;Gw->h=0;Gw->
i=0;Gw->d=STBIR_EDGE_CLAMP;Gw->e=STBIR_EDGE_CLAMP;Gw->E=0;Gw->F=0;Gw->G=1;Gw->H=
1;Gw->M=0;Gw->N=0;Gw->O=Gw->K;Gw->P=Gw->L;Gw->Z=Gy;Gw->a=Gy;Gw->X=Gx;Gw->Y=Gx;Gw
->V=1;}extern void stbir_resize_init(B1*Gw,const void*Gx,int Gy,int Gz,int G0,//
void*G1,int G2,int G3,int G4,Bs G5,Bv G6){Gw->B=Gx;Gw->C=Gy;Gw->D=Gz;Gw->R=G0;Gw
->J=G1;Gw->K=G2;Gw->L=G3;Gw->S=G4;Gw->U=0;Gt(Gw,G5,G6);}extern void/////////////
stbir_set_datatypes(B1*Gw,Bv Gx,Bv Gy){Gw->Z=Gx;Gw->a=Gy;if(Gw->j&&(!Gw->V))Gp(
Gw->j,Gw);}extern void stbir_set_pixel_callbacks(B1*Gw,Bw*Gx,Bx*Gy){Gw->I=Gx;Gw
->Q=Gy;if(Gw->j&&(!Gw->V)){Gw->j->K=Gx;Gw->j->M=Gy;}}extern void////////////////
stbir_set_user_data(B1*Gw,void*Gx){Gw->A=Gx;if(Gw->j&&(!Gw->V))Gw->j->L=Gx;}////
extern void stbir_set_buffer_ptrs(B1*Gw,const void*Gx,int Gy,void*Gz,int G0){Gw
->B=Gx;Gw->R=Gy;Gw->J=Gz;Gw->S=G0;if(Gw->j&&(!Gw->V))Gp(Gw->j,Gw);}extern int///
stbir_set_edgemodes(B1*Gw,Bt Gx,Bt Gy){Gw->d=Gx;Gw->e=Gy;Gw->V=1;return 1;}/////
extern int stbir_set_filters(B1*Gw,Bu Gx,Bu Gy){Gw->b=Gx;Gw->c=Gy;Gw->V=1;return
1;}extern int stbir_set_filter_callbacks(B1*Gw,By*Gx,Bz*Gy,By*Gz,Bz*G0){Gw->f=Gx
;Gw->g=Gy;Gw->h=Gz;Gw->i=G0;Gw->V=1;return 1;}extern int stbir_set_pixel_layouts
(B1*Gw,Bs Gx,Bs Gy){Gw->X=Gx;Gw->Y=Gy;Gw->V=1;return 1;}extern int//////////////
stbir_set_non_pm_alpha_speed_over_quality(B1*Gw,int Gx){Gw->U=Gx;Gw->V=1;return
1;}extern int stbir_set_input_subrect(B1*Gw,double Gx,double Gy,double Gz,double
G0){Gw->E=Gx;Gw->F=Gy;Gw->G=Gz;Gw->H=G0;Gw->V=1;if((Gz<7.523e-37)||((Gz-Gx)<////
7.523e-37)||(G0<7.523e-37)||((G0-Gy)<7.523e-37)||(Gx>1.0)||(Gy>1.0))return 0;///
return 1;}extern int stbir_set_output_pixel_subrect(B1*Gw,int Gx,int Gy,int Gz,
int G0){Gw->M=Gx;Gw->N=Gy;Gw->O=Gz;Gw->P=G0;Gw->V=1;if((Gx>=Gw->K)||((Gx+Gz)<=0)
||(Gy>=Gw->L)||((Gy+G0)<=0)||(Gz==0)||(G0==0))return 0;return 1;}extern int/////
stbir_set_pixel_subrect(B1*Gw,int Gx,int Gy,int Gz,int G0){double G1,G2,G3,G4;G1
=((double)Gx)/((double)Gw->K);G2=((double)Gy)/((double)Gw->L);G3=((double)(Gx+Gz
))/((double)Gw->K);G4=((double)(Gy+G0))/((double)Gw->L);Gw->E=G1;Gw->F=G2;Gw->G=
G3;Gw->H=G4;Gw->M=Gx;Gw->N=Gy;Gw->O=Gz;Gw->P=G0;Gw->V=1;if((Gx>=Gw->K)||((Gx+Gz)
<=0)||(Gy>=Gw->L)||((Gy+G0)<=0)||(Gz==0)||(G0==0))return 0;return 1;}static int
Gu(B1*Gw,int Gx){B4 Gy={0,0};B8 Gz,G0;int G1,G2;B0*G3;if(Gw->j)return 0;if((////
unsigned)Gw->b>=STBIR_FILTER_OTHER)return 0;if((unsigned)Gw->c>=////////////////
STBIR_FILTER_OTHER)return 0;if(Gx<=0)return 0;G1=Gw->M;G2=Gw->N;if(!Gs(&Gz.E,Gw
->K,&G1,Gw->O,Gw->C,Gw->E,Gw->G))return 0;if(!Gs(&G0.E,Gw->L,&G2,Gw->P,Gw->D,Gw
->F,Gw->H))return 0;if((Gz.E.B==0)||(G0.E.B==0))return 0;Gb(&Gz,Gw->b,Gw->f,Gw->
g,Gw->d,&Gz.E,1,Gw->A);Gc(&Gz,&Gy,Gw->A);Gb(&G0,Gw->c,Gw->h,Gw->i,Gw->e,&G0.E,0,
Gw->A);if((G0.E.B/Gx)<4){Gx=G0.E.B/4;if(Gx==0)Gx=1;}G3=Gn(&Gz,&G0,&Gy,Gw->X,Gw->
Y,Gx,G1,G2,Gw->U,Gw->A);if(G3){Gw->T=Gx;Gw->j=G3;Gw->V=0;Gp(G3,Gw);return Gx;}//
return 0;}extern void stbir_free_samplers(B1*Gw){if(Gw->j){Ge(Gw->j);Gw->j=0;Gw
->W=0;}}extern int stbir_build_samplers_with_splits(B1*Gw,int Gx){if((Gw->j==0)
||Gw->V){if(Gw->j)stbir_free_samplers(Gw);Gw->W=1;return Gu(Gw,Gx);}return 1;}//
extern int stbir_build_samplers(B1*Gw){return stbir_build_samplers_with_splits(
Gw,1);}extern int stbir_resize_extended(B1*Gw){int Gx;if((Gw->j==0)||Gw->V){int
Gy=Gw->W;if(Gw->j){Ge(Gw->j);Gw->j=0;}if(!stbir_build_samplers(Gw))return 0;Gw->
W=Gy;if(Gw->j==0)return 1;}else{}Gx=Go(Gw->j,0,Gw->T);if(!Gw->W){///////////////
stbir_free_samplers(Gw);Gw->j=0;}return Gx;}extern int//////////////////////////
stbir_resize_extended_split(B1*Gw,int Gx,int Gy){if((Gx==-1)||((Gx==0)&&(Gy==Gw
->T)))return stbir_resize_extended(Gw);if((Gw->j==0)||Gw->V)return 0;if((Gx>=Gw
->T)||(Gx<0)||((Gx+Gy)>Gw->T)||(Gy<=0))return 0;return Go(Gw->j,Gx,Gy);}static//
void*Gv(const void*Gw,int Gx,int Gy,int Gz,void*G0,int G1,int G2,int G3,Bs G4,Bv
G5,Bt G6,Bu G7){B1 G8;int G9;int G_;void*HA;void*HB;G9=G1*B3[G5]*Gl[Gm[G4]];if(
G9==0)return 0;if(G3==0)G3=G9;G_=G3;if(G_<0)G_=-G_;if(G_<G9)return 0;HA=G0;HB=0;
if(G0==0){size_t HC;char*HD;HC=(size_t)G_*(size_t)G2;if(HC==0)return 0;HD=(char*
)((void)0,malloc(HC));if(HD==0)return 0;HB=HD;if(G3<0)HA=HD+((size_t)G_*(size_t)
(G2-1));else HA=HD;}stbir_resize_init(&G8,Gw,Gx,Gy,Gz,HA,G1,G2,G3,G4,G5);G8.d=G6
;G8.e=G6;G8.b=G7;G8.c=G7;if(!stbir_resize_extended(&G8)){if(HB)(void)0,free(HB);
return 0;}return HB?HB:HA;}extern unsigned char*stbir_resize_uint8_linear(const
unsigned char*Gw,int Gx,int Gy,int Gz,unsigned char*G0,int G1,int G2,int G3,Bs//
G4){return(unsigned char*)Gv(Gw,Gx,Gy,Gz,G0,G1,G2,G3,G4,STBIR_TYPE_UINT8,///////
STBIR_EDGE_CLAMP,STBIR_FILTER_DEFAULT);}extern unsigned char*///////////////////
stbir_resize_uint8_srgb(const unsigned char*Gw,int Gx,int Gy,int Gz,unsigned////
char*G0,int G1,int G2,int G3,Bs G4){return(unsigned char*)Gv(Gw,Gx,Gy,Gz,G0,G1,
G2,G3,G4,STBIR_TYPE_UINT8_SRGB,STBIR_EDGE_CLAMP,STBIR_FILTER_DEFAULT);}extern///
float*stbir_resize_float_linear(const float*Gw,int Gx,int Gy,int Gz,float*G0,int
G1,int G2,int G3,Bs G4){return(float*)Gv(Gw,Gx,Gy,Gz,G0,G1,G2,G3,G4,////////////
STBIR_TYPE_FLOAT,STBIR_EDGE_CLAMP,STBIR_FILTER_DEFAULT);}extern void*///////////
stbir_resize(const void*Gw,int Gx,int Gy,int Gz,void*G0,int G1,int G2,int G3,Bs
G4,Bv G5,Bt G6,Bu G7){return(void*)Gv(Gw,Gx,Gy,Gz,G0,G1,G2,G3,G4,G5,G6,G7);}////

#pragma GCC diagnostic pop
// clang-format on
// NOLINTEND

// The actual code starts here :D

// Default values for cli
#define DEFAULT_SEQLEN      16384
#define DEFAULT_TOPK        0
#define DEFAULT_CHUNK_SIZE  1024
#define DEFAULT_TEMPERATURE 1.0
#define DEFAULT_TOPP        1.0
#define DEFAULT_RPEN        1.0
#define DEFAULT_PROMPT      "Once upon a time"

// Matrix multiplication block sizes
#define GEMM_NT_MR 8
#define GEMM_NT_NR 8
#define GEMM_NN_MR 8
#define GEMM_I8_MR 8
#define GEMM_I8_NR 8

// If the number of prefill tokens is below this number, fallback to the decode
// implementation
#define PREFILL_FALLBACK_THRESHOLD 8

// Block size of blockwise causal masking
#define QK_BLOCK_SIZE 64

// Prompt used in pan & scan
// google/gemma_pytorch/blob/main/gemma/gemma3_preprocessor.py
#define CROPPED_IMAGE_PREFIX "here is the original image"
#define CROPPED_IMAGE_FILTER "and here are some crops to help you see better"

// Default parameters used in pan & scan
// google/gemma_pytorch/blob/main/gemma/siglip_vision/pan_and_scan.py
#define MIN_CROP_SIZE 256
#define MAX_NUM_CROPS 4

// Macros for different dtypes
// TODO: dispatch dtype at runtime using the `#include __FILE__` trick
#define FP16 1
#define BF16 2
#define FP32 3

#ifndef DTYPE
// Default dtype
#  define DTYPE FP16
#endif

// clang-format off
#if DTYPE == FP16
#  define floatx     _Float16
#  define DTYPE_MAGIC  "f16"
#  define FLOATX_MAX (((union {floatx f; uint16_t b; }){.b = 0x00007BFF}).f)
#elif DTYPE == BF16
#  define floatx     __bf16
#  define DTYPE_MAGIC  "b16"
#  define FLOATX_MAX (((union {floatx f; uint16_t b; }){.b = 0x00007F7F}).f)
#elif DTYPE == FP32
#  define floatx     float
#  define DTYPE_MAGIC  "f32"
#  define FLOATX_MAX (((union {floatx f; uint32_t b; }){.b = 0x7F7FFFFF}).f)
#else
#  error "unsupported DTYPE"
#endif
// clang-format on

#ifdef min
#  undef min
#endif
#ifdef max
#  undef max
#endif

#define GENERATOR_EXIT 10000

/* */
static inline int
max(int a, int b)
{
  return a > b ? a : b;
}

/* */
static inline int
min(int a, int b)
{
  return a < b ? a : b;
}

#define TOSTRING(x) STRINGIFY_(x)
// I usually use a trailing underscore to indicate a variable/function/macro is
// temporary
#define STRINGIFY_(x) #x

/* Poor man's memory wrappers, I just really dislike the redundency of writing
 * ```c
 * if ((ptr = malloc(count * sizeof(*ptr))) == NULL)
 * {
 *   fprintf(stderr, "error: ...");
 *   goto fail;
 * }
 * ```
 * in every memory allocation. So I wrapped them into the following macros, now
 * I only need to type:
 * ```c
 * MALLOC(ptr, count, "name_of_the_buffer", goto fail;);
 * ```
 * (Same for other operations like calloc or fread)
 * This is much more readable and less redundent for me. The implementations of
 * the macros themselves are indeed error-prone and hard to read, but once
 * they're complete I barely need to touch them. I guess that worked for me :D
 */

#define MALLOC(ptr, count, name, ...)                                          \
  do                                                                           \
  {                                                                            \
    ptr = malloc((count) * sizeof(*(ptr)));                                    \
    if (ptr == NULL)                                                           \
    {                                                                          \
      fprintf(                                                                 \
        stderr, "error: memory allocation failed: %s (size: %zu)\n", (name),   \
        (size_t)(count) * sizeof(*(ptr))                                       \
      );                                                                       \
      __VA_ARGS__                                                              \
    }                                                                          \
  }                                                                            \
  while (0)

#define CALLOC(ptr, count, name, ...)                                          \
  do                                                                           \
  {                                                                            \
    ptr = calloc((count), sizeof(*(ptr)));                                     \
    if (ptr == NULL)                                                           \
    {                                                                          \
      fprintf(                                                                 \
        stderr, "error: memory allocation failed: %s (size: %zu)\n", (name),   \
        (size_t)(count) * sizeof(*(ptr))                                       \
      );                                                                       \
      __VA_ARGS__                                                              \
    }                                                                          \
  }                                                                            \
  while (0)

#define FGETC(var, fp, name, ...)                                              \
  do                                                                           \
  {                                                                            \
    var = fgetc(fp);                                                           \
    if (var == EOF)                                                            \
    {                                                                          \
      fprintf(stderr, "error: file read failed: %s", (name));                  \
      __VA_ARGS__                                                              \
    }                                                                          \
  }                                                                            \
  while (0)

#define FSEEK(fp, offset, base, name, ...)                                     \
  do                                                                           \
  {                                                                            \
    if (fseek((fp), (offset), (base)) == EOF)                                  \
    {                                                                          \
      fprintf(stderr, "error: file seek failed: %s", (name));                  \
      __VA_ARGS__                                                              \
    }                                                                          \
  }                                                                            \
  while (0)

#define FREAD(ptr, count, fp, name, ...)                                       \
  do                                                                           \
  {                                                                            \
    size_t fread_buf_ = fread((ptr), sizeof(*(ptr)), (count), (fp));           \
    if (fread_buf_ != (size_t)(count))                                         \
    {                                                                          \
      fprintf(                                                                 \
        stderr, "error: file read failed: %s (expected %zu, got %zu)\n",       \
        (name), (size_t)(count), fread_buf_                                    \
      );                                                                       \
      __VA_ARGS__                                                              \
    }                                                                          \
  }                                                                            \
  while (0)

#define READ_UINT16(var, fp, name, ...)                                        \
  do                                                                           \
  {                                                                            \
    /* Big-endian */                                                           \
    unsigned char ru16_buf_[2];                                                \
    FREAD(ru16_buf_, 2, (fp), (name), __VA_ARGS__);                            \
    var = ((int)ru16_buf_[0] << 8) | (int)ru16_buf_[1];                        \
  }                                                                            \
  while (0)

#define READ_UINT32(var, fp, name, ...)                                        \
  do                                                                           \
  {                                                                            \
    /* Big-endian */                                                           \
    unsigned char ru32_buf_[4];                                                \
    FREAD(ru32_buf_, 4, (fp), (name), __VA_ARGS__);                            \
    var = ((uint32_t)ru32_buf_[0] << 24) | ((uint32_t)ru32_buf_[1] << 16) |    \
          ((uint32_t)ru32_buf_[2] << 8) | ((uint32_t)ru32_buf_[3]);            \
  }                                                                            \
  while (0)

#define READ_FP32(var, fp, name, ...)                                          \
  do                                                                           \
  {                                                                            \
    int rfp32_bits_;                                                           \
    READ_UINT32(rfp32_bits_, (fp), (name), __VA_ARGS__);                       \
    memcpy(&var, &rfp32_bits_, sizeof(float));                                 \
  }                                                                            \
  while (0)

#define READ_TENSOR(ptr, count, fp, name, ...)                                 \
  do                                                                           \
  {                                                                            \
    MALLOC((ptr), (count), (name), __VA_ARGS__);                               \
    FREAD((ptr), (count), (fp), (name), __VA_ARGS__);                          \
  }                                                                            \
  while (0)

// Pascal-style string: first byte is length, then that many chars
#define READ_STR(var, fp, data, offset, name, ...)                             \
  do                                                                           \
  {                                                                            \
    char rstr_len_name_[128];                                                  \
    snprintf(rstr_len_name_, 128, "%s.length", (name));                        \
    int rstr_len_;                                                             \
    FGETC(rstr_len_, fp, rstr_len_name_, __VA_ARGS__);                         \
    var = (data) + *(offset);                                                  \
    FREAD(var, rstr_len_, (fp), (name), __VA_ARGS__);                          \
    (data)[*(offset) + rstr_len_] = '\0';                                      \
    *(offset) += rstr_len_ + 1;                                                \
  }                                                                            \
  while (0)

/* Peek at how many bytes a sequence of pascal strings will occupy */
static inline int
get_strarr_bytes(FILE *fp, int count)
{
  // Read a sequence of pascal-stype strings
  int  offset = 0;
  long pos    = ftell(fp);
  if (pos == -1L)
  {
    perror("ftell failed");
    return -1;
  }
  // Get the total number of bytes
  for (int i = 0; i < count; i++)
  {
    int len;
    FGETC(len, fp, "<str.length>", return -1;);
    if (fseek(fp, len, SEEK_CUR) != 0)
    {
      perror("fseek failed");
      return -1;
    }
    offset += len + 1;
  }
  // Resume position
  if (fseek(fp, pos, SEEK_SET) != 0)
  {
    perror("fseek failed");
    return -1;
  }
  return offset;
}

/* Configuration for the SigLIP vision encoder */
typedef struct
{
  int   n_layers;    // ViT layers
  int   image_size;  // Pixels per image side
  int   patch_size;  // Patches per image side
  int   hidden_dim;  // ViT dimension
  int   n_heads;     // Attention heads
  int   mlp_dim;     // Intermediate size in FFN
  float eps;         // LayerNorm epsilon
} VisionConfig;

/* */
static VisionConfig *
read_vision_config(FILE *fp)
{
  VisionConfig *vcfg;

  const char *vcfg_name = "model.encoder.config";
  const char *nl_name   = "model.encoder.config.n_layers";
  const char *nh_name   = "model.encoder.config.n_heads";
  const char *cm_name   = "model.encoder.config.mlp_dim";
  const char *c_name    = "model.encoder.config.hidden_dim";
  const char *is_name   = "model.encoder.config.image_size";
  const char *ps_name   = "model.encoder.config.patch_size";
  const char *eps_name  = "model.encoder.config.eps";

  CALLOC(vcfg, 1, vcfg_name, goto fail;);

  FGETC(vcfg->n_layers, fp, nl_name, goto fail;);
  FGETC(vcfg->n_heads, fp, nh_name, goto fail;);
  READ_UINT16(vcfg->mlp_dim, fp, cm_name, goto fail;);
  READ_UINT16(vcfg->hidden_dim, fp, c_name, goto fail;);
  READ_UINT16(vcfg->image_size, fp, is_name, goto fail;);
  READ_UINT16(vcfg->patch_size, fp, ps_name, goto fail;);
  READ_FP32(vcfg->eps, fp, eps_name, goto fail;);

  return vcfg;

fail:
  free(vcfg);
  return NULL;
}

/* Configuration for the main Gemma text encoder */
typedef struct
{
  int   n_layers;       // Transformer layers
  int   n_heads;        // Attention heads
  int   n_kv_heads;     // Key/value heads (GQA)
  int   head_dim;       // Dim per head
  int   embed_dim;      // Model dimension
  int   mlp_dim;        // Intermediate size in FFN
  int   q_scale;        // Scale applied to queries before attention
  int   slide_len;      // Sliding-window size
  int   image_toks;     // Number of soft tokens per image
  int   max_seqlen;     // Max number of position embeddings
  int   vocab_size;     // Number of possible tokens
  float local_theta;    // RoPE base for local (sliding) attention
  float global_theta;   // RoPE base for full attention
  float eps;            // RMSNorm epsilon
  float att_softcap;    // tanh softcap on attention scores (0 = off)
  float logit_softcap;  // tanh softcap on final logits (0 = off)
  bool *att_layers;     // true = use sliding window for that layer
  bool  qk_norm;        // query/key RMSNorm
  bool  pre_mlp_norm;
  bool  pst_mlp_norm;
} TextConfig;

/* */
static void
free_text_config(TextConfig *cfg)
{
  if (cfg == NULL) return;
  free(cfg->att_layers);
  free(cfg);
}

/* */
static TextConfig *
read_text_config(FILE *fp, bool *support_mm)
{
  TextConfig *cfg            = NULL;
  char       *att_layers_buf = NULL;

  const char *cfg_name = "model.decoder.config";
  const char *nl_name  = "model.decoder.config.n_layers";
  const char *nh_name  = "model.decoder.config.n_heads";
  const char *nkv_name = "model.decoder.config.n_kv_heads";
  const char *ch_name  = "model.decoder.config.head_dim";
  const char *c_name   = "model.decoder.config.embed_dim";
  const char *cm_name  = "model.decoder.config.mlp_dim";
  const char *qs_name  = "model.decoder.config.q_scale";
  const char *sl_name  = "model.decoder.config.slide_len";
  const char *it_name  = "model.decoder.config.image_toks";
  const char *msq_name = "model.decoder.config.max_seqlen";
  const char *vs_name  = "model.decoder.config.vocab_size";
  const char *lt_name  = "model.decoder.config.local_theta";
  const char *gt_name  = "model.decoder.config.global_theta";
  const char *eps_name = "model.decoder.config.eps";
  const char *asc_name = "model.decoder.config.att_softcap";
  const char *lgc_name = "model.decoder.config.logit_softcap";
  const char *al_name  = "model.decoder.config.att_layers";
  const char *ext_name = "model.decoder.config.extra_bytes";

  CALLOC(cfg, 1, cfg_name, goto fail;);

  FGETC(cfg->n_layers, fp, nl_name, goto fail;);
  FGETC(cfg->n_heads, fp, nh_name, goto fail;);
  FGETC(cfg->n_kv_heads, fp, nkv_name, goto fail;);

  READ_UINT16(cfg->head_dim, fp, ch_name, goto fail;);
  READ_UINT16(cfg->embed_dim, fp, c_name, goto fail;);
  READ_UINT16(cfg->mlp_dim, fp, cm_name, goto fail;);
  READ_UINT16(cfg->q_scale, fp, qs_name, goto fail;);
  READ_UINT16(cfg->slide_len, fp, sl_name, goto fail;);
  READ_UINT16(cfg->image_toks, fp, it_name, goto fail;);
  READ_UINT32(cfg->max_seqlen, fp, msq_name, goto fail;);
  READ_UINT32(cfg->vocab_size, fp, vs_name, goto fail;);

  READ_FP32(cfg->local_theta, fp, lt_name, goto fail;);
  READ_FP32(cfg->global_theta, fp, gt_name, goto fail;);
  READ_FP32(cfg->eps, fp, eps_name, goto fail;);
  READ_FP32(cfg->att_softcap, fp, asc_name, goto fail;);
  READ_FP32(cfg->logit_softcap, fp, lgc_name, goto fail;);

  // Packed bit-field of which layers use sliding-window attention
  // A terrible terrible idea, wish I didn't do this
  int n_bytes;
  FGETC(n_bytes, fp, al_name, goto fail;);
  if (n_bytes * 8 < cfg->n_layers)
  {
    fprintf(stderr, "error: insufficient att_layers bytes\n");
    goto fail;
  }
  MALLOC(att_layers_buf, n_bytes, al_name, goto fail;);
  MALLOC(cfg->att_layers, cfg->n_layers, al_name, goto fail;);
  FREAD(att_layers_buf, n_bytes, fp, al_name, goto fail;);
  for (int i = 0; i < cfg->n_layers; i++)
  {
    int pos            = i;
    int byte_idx       = pos / 8;
    int bit_idx        = 7 - (pos % 8);
    cfg->att_layers[i] = (att_layers_buf[byte_idx] >> bit_idx) & 1;
  }
  free(att_layers_buf);
  att_layers_buf = NULL;

  // Extra feature flags packed into one byte
  int extra_flags;
  FGETC(extra_flags, fp, ext_name, goto fail;);
  *support_mm       = (extra_flags & 8) == 8;
  cfg->qk_norm      = (extra_flags & 4) == 4;
  cfg->pre_mlp_norm = (extra_flags & 2) == 2;
  cfg->pst_mlp_norm = (extra_flags & 1) == 1;

  return cfg;

fail:
  free_text_config(cfg);
  free(att_layers_buf);
  return NULL;
}

// Tokenizer implementation

/* */
typedef struct
{
  char *val;
  int   idx;
} Token;
typedef struct
{
  char *str1;
  char *str2;
  int   rank;
} Merge;

/* */
static int
cmp_token(const void *a, const void *b)
{
  return strcmp(((Token *)a)->val, ((Token *)b)->val);
}

/* */
static int
cmp_merge(const void *a, const void *b)
{
  int ret = strcmp(((Merge *)a)->str1, ((Merge *)b)->str1);
  if (ret != 0)
  {
    return ret;
  }
  return strcmp(((Merge *)a)->str2, ((Merge *)b)->str2);
}

/* */
typedef struct
{
  int n_merges;
  int vocab_size;
  int bos;  // beginning of sequence
  int eos;  // end of sequence
  int sot;  // start of turn
  int eot;  // end of turn
  int soi;  // start of image
  int eoi;  // end of image
  int ist;  // image soft token

  char  *vocab_data;
  char  *merge_data;
  char **vocab;
  Token *vocab_sorted;  // sorted for binary search
  Merge *ranks;         // sorted merges for BPE
} GemmaTokenizer;

/* Look up a string in the sorted vocab -> token id (or -1) */
static int
get_token_idx(GemmaTokenizer *tok, char *str)
{
  // Get idx_to_vocab[string]
  Token  key = {.val = str};
  Token *val = bsearch(
    &key, tok->vocab_sorted, tok->vocab_size, sizeof(tok->vocab_sorted[0]),
    cmp_token
  );
  if (val == NULL)
  {
    return -1;
  }
  return val->idx;
}

/* Look up a (str1, str2) pair in the merges table */
static Merge *
get_merge_rec(GemmaTokenizer *tok, char *str1, char *str2)
{
  // Get merges[(str1, str2)]
  Merge  key = {.str1 = str1, .str2 = str2};
  Merge *val =
    bsearch(&key, tok->ranks, tok->n_merges, sizeof(tok->ranks[0]), cmp_merge);
  return val;  // NULL if not found
}

/* */
static void
free_tokenizer(GemmaTokenizer *tok)
{
  if (tok == NULL) return;
  free(tok->vocab_data);
  free(tok->merge_data);
  free(tok->vocab);
  free(tok->vocab_sorted);
  free(tok->ranks);
  free(tok);
}

/* */
static GemmaTokenizer *
read_tokenizer(FILE *fp, TextConfig *cfg, bool support_mm)
{
  GemmaTokenizer *tok;
  CALLOC(tok, 1, "model.tokenizer", goto fail;);

  tok->vocab_size = cfg->vocab_size;
  if (support_mm)
  {
    tok->vocab_size++;
  }  // ++ for the <image_soft_token>
  int vocab_data_bytes = get_strarr_bytes(fp, tok->vocab_size);
  if (vocab_data_bytes == -1) goto fail;

  const char *vd_name = "model.tokenizer.vocab_data";
  const char *vc_name = "model.tokenizer.vocab";
  const char *vt_name = "model.tokenizer.vocab_sorted";

  MALLOC(tok->vocab_data, vocab_data_bytes, vd_name, goto fail;);
  MALLOC(tok->vocab, tok->vocab_size, vc_name, goto fail;);
  MALLOC(tok->vocab_sorted, tok->vocab_size, vt_name, goto fail;);

  int offset = 0;
  for (int i = 0; i < tok->vocab_size; i++)
  {
    char name[64];
    snprintf(name, sizeof(name), "%s.%d", vd_name, i);
    char *str;
    READ_STR(str, fp, tok->vocab_data, &offset, name, goto fail;);
    tok->vocab[i]            = str;
    tok->vocab_sorted[i].idx = i;
    tok->vocab_sorted[i].val = str;
  }
  qsort(
    tok->vocab_sorted, tok->vocab_size, sizeof(tok->vocab_sorted[0]), cmp_token
  );

  // Special tokens
  tok->bos = get_token_idx(tok, "<bos>");
  tok->eos = get_token_idx(tok, "<eos>");
  tok->sot = get_token_idx(tok, "<start_of_turn>");
  tok->eot = get_token_idx(tok, "<end_of_turn>");
  tok->soi = get_token_idx(tok, "<start_of_image>");
  tok->eoi = get_token_idx(tok, "<end_of_image>");
  tok->ist = get_token_idx(tok, "<image_soft_token>");

  const char *nm_name = "model.tokenizer.n_merges";
  const char *rk_name = "model.tokenizer.ranks";
  const char *md_name = "model.tokenizer.merge_data";

  // Build merges
  READ_UINT32(tok->n_merges, fp, nm_name, goto fail;);
  MALLOC(tok->ranks, tok->n_merges, rk_name, goto fail;);

  int merge_bytes = get_strarr_bytes(fp, tok->n_merges * 2);
  if (merge_bytes == -1) goto fail;
  MALLOC(tok->merge_data, merge_bytes, md_name, goto fail;);

  offset = 0;
  for (int i = 0; i < tok->n_merges; i++)
  {
    char name0[64], name1[64];
    snprintf(name0, sizeof(name0), "%s.%d.0", md_name, i);
    snprintf(name1, sizeof(name1), "%s.%d.1", md_name, i);

    char *str1, *str2;
    READ_STR(str1, fp, tok->merge_data, &offset, name0, goto fail;);
    READ_STR(str2, fp, tok->merge_data, &offset, name1, goto fail;);

    tok->ranks[i].rank = i;
    tok->ranks[i].str1 = str1;
    tok->ranks[i].str2 = str2;
  }
  qsort(tok->ranks, tok->n_merges, sizeof(tok->ranks[0]), cmp_merge);
  return tok;

fail:
  free_tokenizer(tok);
  return NULL;
}

/* Minimal byte-pair encoding algorithm implementation */
void
encode(
  GemmaTokenizer *tok,
  const char     *text,
  int             len,
  int            *tokens,
  int             spos,
  int            *n_tokens)
{
  unsigned char *ustr = (unsigned char *)text;

  // Convert UTF-8 string to tokens of individual codepoints
  int i     = 0;
  int tok_i = 0;
  tokens += spos;

  while (i < len)
  {
    /* table from https://zh.wikipedia.org/wiki/UTF-8
     *
     * U+00000-U+00007F  1  0xxxxxxx
     * U+00080-U+0007FF  2  110xxxxx  10xxxxxx
     * U+00800-U+00FFFF  3  1110xxxx  10xxxxxx  10xxxxxx
     * U+10000-U+1FFFFF  4  11110xxx  10xxxxxx  10xxxxxx  10xxxxxx
     */
    int n_bytes = 0;
    int start   = i;

    if (ustr[i] >> 7 == 0)
    {
      n_bytes = 1;
    }
    else if (i + 1 < len && ustr[i] >> 5 == 6 && ustr[i + 1] >> 6 == 2)
    {
      n_bytes = 2;
    }
    else if (i + 2 < len && ustr[i] >> 4 == 14 && ustr[i + 1] >> 6 == 2 &&
             ustr[i + 2] >> 6 == 2)
    {
      n_bytes = 3;
    }
    else if (i + 3 < len && ustr[i] >> 3 == 30 && ustr[i + 1] >> 6 == 2 &&
             ustr[i + 2] >> 6 == 2 && ustr[i + 3] >> 6 == 2)
    {
      n_bytes = 4;
    }
    else
    {
      n_bytes = 1;
    }

    char cstr[5];
    for (int b = 0; b < n_bytes; b++)
    {
      cstr[b] = (char)ustr[i++];
    }
    cstr[n_bytes] = '\0';

    int token = get_token_idx(tok, cstr);
    if (token == -1)
    {
      // Fallback to byte level tokens
      char bstr[10];
      for (int b = start; b < i; b++)
      {
        snprintf(bstr, 10, "<0x%02X>", (unsigned char)ustr[b]);
        tokens[tok_i++] = get_token_idx(tok, bstr);
      }
    }
    else
    {
      tokens[tok_i++] = token;
    }
  }

  // Keep merging the highest-rank pair until nothing left
  for (;;)
  {
    int    best_rank = 2147483647;
    Merge *best_pair = NULL;

    // Find the merge with best rank
    for (int i = 0; i < tok_i - 1; i++)
    {
      char *str1 = tok->vocab[tokens[i]];
      char *str2 = tok->vocab[tokens[i + 1]];

      Merge *merge = get_merge_rec(tok, str1, str2);
      if (merge != NULL && merge->rank < best_rank)
      {
        best_rank = merge->rank;
        best_pair = merge;
      }
    }

    if (best_pair == NULL) break;  // No more merges

    int i = 0;
    while (i < tok_i - 1)
    {
      Merge pair = {
        .str1 = tok->vocab[tokens[i]], .str2 = tok->vocab[tokens[i + 1]]
      };
      if (cmp_merge(&pair, best_pair) == 0)
      {
        // Merge the pair, left shift all the tokens on its right side
        char merged[128];
        snprintf(
          merged, sizeof(merged), "%s%s", best_pair->str1, best_pair->str2
        );
        tokens[i] = get_token_idx(tok, merged);
        for (int j = i + 1; j < tok_i - 1; j++)
        {
          tokens[j] = tokens[j + 1];
        }
        tok_i--;
      }
      i++;
    }
  }

  *n_tokens += tok_i;
}

/* */
const char *
decode(GemmaTokenizer *tok, int id, char *byte_buf)
{
  char *s = tok->vocab[id];
  // Byte-fallback tokens look like <0xAB>, turn them back into a raw byte
  size_t len = strlen(s);
  if (len == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>')
  {
    unsigned int byte_val = 0;
    if (sscanf(s + 3, "%2x", &byte_val) == 1)
    {
      byte_buf[0] = (char)byte_val;
      byte_buf[1] = '\0';
      return byte_buf;
    }
  }
  return s;
}

// Weight storage

/* */
typedef enum
{
  DTYPE_FPX,
  DTYPE_INT8
} WeightDType;

/* A linear layer, either dense fpx or int8 + scales */
typedef struct
{
  WeightDType dtype;
  union
  {
    floatx *fpx;
    struct
    {
      int8_t *q;
      floatx *scales;
    } i8;
  };
} Linear;

/* */
static inline size_t
linear_size(int m, int n, bool quant)
{
  if (!quant)
  {
    return (size_t)m * n * sizeof(floatx);
  }
  else
  {
    return (size_t)m * n * sizeof(int8_t) + (size_t)n * sizeof(floatx);
  }
}

// Linear layer weights: either plain floatx or int8 + per-channel scales
#define READ_LINEAR(w, fp, m, n, quant, name, ...)                             \
  do                                                                           \
  {                                                                            \
    MALLOC((w), 1, (name), __VA_ARGS__);                                       \
    if (!(quant))                                                              \
    {                                                                          \
      (w)->dtype = DTYPE_FPX;                                                  \
      char rlinear_fpx_name_[128];                                             \
      snprintf(rlinear_fpx_name_, 128, "%s.fpx", (name));                      \
      READ_TENSOR(                                                             \
        (w)->fpx, (size_t)(m) * (size_t)(n), (fp), rlinear_fpx_name_,          \
        __VA_ARGS__                                                            \
      );                                                                       \
    }                                                                          \
    else                                                                       \
    {                                                                          \
      (w)->dtype = DTYPE_INT8;                                                 \
      char rlinear_i8q_name_[128];                                             \
      char rlinear_i8scales_name_[128];                                        \
      snprintf(rlinear_i8q_name_, 128, "%s.i8.q", (name));                     \
      snprintf(rlinear_i8scales_name_, 128, "%s.i8.scales", (name));           \
      READ_TENSOR(                                                             \
        (w)->i8.q, (size_t)(m) * (size_t)(n), (fp), rlinear_i8q_name_,         \
        __VA_ARGS__                                                            \
      );                                                                       \
      READ_TENSOR(                                                             \
        (w)->i8.scales, (n), (fp), rlinear_i8scales_name_, __VA_ARGS__         \
      );                                                                       \
    }                                                                          \
  }                                                                            \
  while (0)

/* */
static inline void
mmap_linear(Linear *w, uint8_t *base, size_t *off, int m, int n, bool quant)
{
  if (!quant)
  {
    w->dtype = DTYPE_FPX;
    w->fpx   = (floatx *)(base + *off);
    *off += (size_t)m * n * sizeof(floatx);
  }
  else
  {
    w->dtype = DTYPE_INT8;
    w->i8.q  = (int8_t *)(base + *off);
    *off += (size_t)m * n * sizeof(int8_t);
    w->i8.scales = (floatx *)(base + *off);
    *off += (size_t)n * sizeof(floatx);
  }
}

/* */
static void
free_linear(Linear *l)
{
  if (l == NULL) return;
  if (l->dtype == DTYPE_FPX)
  {
    free(l->fpx);
  }
  else
  {
    free(l->i8.q);
    free(l->i8.scales);
  }
  free(l);
}

/* One Gemma transformer block */
typedef struct
{
  // Attention weights
  // 2D weights are stored transposed for better GEMV cache use

  Linear *wq;  // (embed_dim, n_heads * head_dim).T
  Linear *wk;  // (embed_dim, n_kv_heads * head_dim).T
  Linear *wv;  // (embed_dim, n_kv_heads * head_dim).T
  Linear *wo;  // (n_heads * head_dim, embed_dim).T

  // Feedforward weights
  Linear *w1;  // (embed_dim, mlp_dim).T
  Linear *w2;  // (embed_dim, mlp_dim).T
  Linear *w3;  // (mlp_dim, embed_dim).T

  // RMSNorm weights (Gemma adds 1.0 to the weight)
  floatx *nq;  // (head_dim,)
  floatx *nk;  // (head_dim,)
  floatx *n1;  // (embed_dim,)
  floatx *n2;  // (embed_dim,)
  floatx *n3;  // (embed_dim,)
  floatx *n4;  // (embed_dim,)
} TextDecoderLayer;

/* */
static void
free_text_layer(TextDecoderLayer *layer)
{
  if (layer == NULL) return;
  free_linear(layer->wq);
  free_linear(layer->wk);
  free_linear(layer->wv);
  free_linear(layer->wo);
  free_linear(layer->w1);
  free_linear(layer->w2);
  free_linear(layer->w3);
  free(layer->n1);
  free(layer->n2);
  free(layer->nq);
  free(layer->nk);
  free(layer->n3);
  free(layer->n4);
  free(layer);
}

/* */
static void
free_text_layer_wrapper(TextDecoderLayer *layer)
{
  // Used in munmap, only free the container, not the mapped pointers
  if (layer == NULL) return;
  free(layer->wq);
  free(layer->wk);
  free(layer->wv);
  free(layer->wo);
  free(layer->w1);
  free(layer->w2);
  free(layer->w3);
  free(layer);
}

/* One SigLIP (vision) encoder block */
typedef struct
{
  // ViT attention weights
  Linear *wq;  // (hidden_dim, hidden_dim).T
  Linear *wk;  // (hidden_dim, hidden_dim).T
  Linear *wv;  // (hidden_dim, hidden_dim).T
  Linear *wo;  // (hidden_dim, hidden_dim).T

  // ViT attention biases
  floatx *bq;  // (hidden_dim,)
  floatx *bk;  // (hidden_dim,)
  floatx *bv;  // (hidden_dim,)
  floatx *bo;  // (hidden_dim,)

  // ViT feedforward weights
  Linear *w1;  // (hidden_dim, mlp_dim).T
  Linear *w2;  // (hidden_dim, mlp_dim).T

  // ViT feedforward biases
  floatx *b1;  // (mlp_dim,)
  floatx *b2;  // (mlp_dim,)

  // ViT layernorm weights
  floatx *n1;  // (hidden_dim,)
  floatx *n2;  // (hidden_dim,)

  // ViT layernorm biases
  floatx *n1_b;  // (hidden_dim,)
  floatx *n2_b;  // (hidden_dim,)
} VisionEncoderLayer;

/* */
static void
free_vision_layer(VisionEncoderLayer *layer)
{
  if (layer == NULL) return;
  free_linear(layer->wq);
  free_linear(layer->wk);
  free_linear(layer->wv);
  free_linear(layer->wo);
  free(layer->bq);
  free(layer->bk);
  free(layer->bv);
  free(layer->bo);
  free_linear(layer->w1);
  free_linear(layer->w2);
  free(layer->b1);
  free(layer->b2);
  free(layer->n1);
  free(layer->n2);
  free(layer->n1_b);
  free(layer->n2_b);
  free(layer);
}

/* */
static void
free_vision_layer_wrapper(VisionEncoderLayer *layer)
{
  if (layer == NULL) return;
  free(layer->wq);
  free(layer->wk);
  free(layer->wv);
  free(layer->wo);
  free(layer->w1);
  free(layer->w2);
  free(layer);
}

/* Text decoder container */
typedef struct
{
  TextConfig *config;
  // (vocab_size, embed_dim), shared with lm_head (tied weights)
  Linear            *embedding;
  TextDecoderLayer **layers;
  floatx            *final_norm;  // (embed_dim,)
} TextDecoder;

/* */
size_t
get_text_decoder_size(const TextConfig *cfg, bool quant)
{
  size_t size = 0;
  int    C    = cfg->embed_dim;
  int    CM   = cfg->mlp_dim;
  int    Cq   = cfg->n_heads * cfg->head_dim;
  int    Ckv  = cfg->n_kv_heads * cfg->head_dim;
  int    vs   = cfg->vocab_size;

  // embedding: linear (C -> vs)
  size += linear_size(C, vs, quant);

  for (int l = 0; l < cfg->n_layers; l++)
  {
    // wq, wk, wv, wo
    size += linear_size(C, Cq, quant);
    size += linear_size(C, Ckv, quant);
    size += linear_size(C, Ckv, quant);
    size += linear_size(Cq, C, quant);

    // optional nq, nk
    if (cfg->qk_norm)
    {
      size += (size_t)cfg->head_dim * sizeof(floatx);
      size += (size_t)cfg->head_dim * sizeof(floatx);
    }

    // w1, w2, w3
    size += linear_size(C, CM, quant);
    size += linear_size(C, CM, quant);
    size += linear_size(CM, C, quant);

    // n1, n2
    size += (size_t)C * sizeof(floatx);
    size += (size_t)C * sizeof(floatx);

    // optional n3, n4
    if (cfg->pre_mlp_norm)
    {
      size += (size_t)C * sizeof(floatx);
    }
    if (cfg->pst_mlp_norm)
    {
      size += (size_t)C * sizeof(floatx);
    }
  }

  // final_norm
  size += (size_t)C * sizeof(floatx);

  return size;
}

/* */
void
free_text_decoder(TextDecoder *dec)
{
  if (dec == NULL) return;

  free_linear(dec->embedding);
  if (dec->layers != NULL && dec->config != NULL)
  {
    for (int l = 0; l < dec->config->n_layers; l++)
    {
      free_text_layer(dec->layers[l]);
    }
    free(dec->layers);
  }
  free(dec->final_norm);
  free_text_config(dec->config);
  free(dec);
}

/* */
void
free_text_decoder_wrapper(TextDecoder *dec)
{
  if (dec == NULL) return;

  free(dec->embedding);
  if (dec->layers != NULL && dec->config != NULL)
  {
    for (int l = 0; l < dec->config->n_layers; l++)
    {
      free_text_layer_wrapper(dec->layers[l]);
    }
    free(dec->layers);
  }
  free_text_config(dec->config);
  free(dec);
}

/* */
static TextDecoder *
read_text_decoder(FILE *fp, TextConfig *cfg, bool quant)
{
  TextDecoder *dec;
  CALLOC(dec, 1, "model.decoder", goto fail;);
  dec->config = cfg;

  int C   = cfg->embed_dim;
  int CM  = cfg->mlp_dim;
  int Cq  = cfg->n_heads * cfg->head_dim;
  int Ckv = cfg->n_kv_heads * cfg->head_dim;
  int vs  = cfg->vocab_size;

  const char *emb_name = "model.decoder.embedding";
  const char *lrs_name = "model.decoder.layers";

  /* The embedding shape is (vocab_size, embed_dim), but it uses per-tensor
   * quantization rather than per-channel like other weights. Gemma uses tied
   * weights, which means the final lm_head shares the same weights with the
   * embedding table, but transposed. So it becomes per-channel quantization in
   * the final lm_head.
   */

  READ_LINEAR(dec->embedding, fp, C, vs, quant, emb_name, goto fail;);
  CALLOC(dec->layers, cfg->n_layers, lrs_name, goto fail;);  // NOLINT

  // Read all the layers
  for (int l = 0; l < cfg->n_layers; l++)
  {
    char layer_name[64];
    snprintf(layer_name, sizeof(layer_name), "%s.%d", lrs_name, l);
    TextDecoderLayer *layer = NULL;
    CALLOC(layer, 1, layer_name, goto fail;);

    char wq_name[64], wk_name[64], wv_name[64], wo_name[64];
    snprintf(wq_name, sizeof(wq_name), "%s.%d.wq", lrs_name, l);
    snprintf(wk_name, sizeof(wk_name), "%s.%d.wk", lrs_name, l);
    snprintf(wv_name, sizeof(wv_name), "%s.%d.wv", lrs_name, l);
    snprintf(wo_name, sizeof(wo_name), "%s.%d.wo", lrs_name, l);

    // Attention weights
    READ_LINEAR(layer->wq, fp, C, Cq, quant, wq_name, goto fail;);
    READ_LINEAR(layer->wk, fp, C, Ckv, quant, wk_name, goto fail;);
    READ_LINEAR(layer->wv, fp, C, Ckv, quant, wv_name, goto fail;);
    READ_LINEAR(layer->wo, fp, Cq, C, quant, wo_name, goto fail;);

    if (cfg->qk_norm)
    {
      char nq_name[64], nk_name[64];
      snprintf(nq_name, sizeof(nq_name), "%s.%d.nq", lrs_name, l);
      snprintf(nk_name, sizeof(nk_name), "%s.%d.nk", lrs_name, l);
      READ_TENSOR(layer->nq, cfg->head_dim, fp, nq_name, goto fail;);
      READ_TENSOR(layer->nk, cfg->head_dim, fp, nk_name, goto fail;);
    }
    else
    {
      layer->nq = NULL;
      layer->nk = NULL;
    }

    char w1_name[64], w2_name[64], w3_name[64];
    snprintf(w1_name, sizeof(w1_name), "%s.%d.w1", lrs_name, l);
    snprintf(w2_name, sizeof(w2_name), "%s.%d.w2", lrs_name, l);
    snprintf(w3_name, sizeof(w3_name), "%s.%d.w3", lrs_name, l);

    // Feedforward weights
    READ_LINEAR(layer->w1, fp, C, CM, quant, w1_name, goto fail;);
    READ_LINEAR(layer->w2, fp, C, CM, quant, w2_name, goto fail;);
    READ_LINEAR(layer->w3, fp, CM, C, quant, w3_name, goto fail;);

    char n1_name[64], n2_name[64];
    snprintf(n1_name, sizeof(n1_name), "%s.%d.n1", lrs_name, l);
    snprintf(n2_name, sizeof(n2_name), "%s.%d.n2", lrs_name, l);

    // RMSNorm weights
    READ_TENSOR(layer->n1, C, fp, n1_name, goto fail;);
    READ_TENSOR(layer->n2, C, fp, n2_name, goto fail;);

    if (cfg->pre_mlp_norm)
    {
      char n3_name[64];
      snprintf(n3_name, sizeof(n3_name), "%s.%d.n3", lrs_name, l);
      READ_TENSOR(layer->n3, C, fp, n3_name, goto fail;);
    }
    else
    {
      layer->n3 = NULL;
    }
    if (cfg->pst_mlp_norm)
    {
      char n4_name[64];
      snprintf(n4_name, sizeof(n4_name), "%s.%d.n4", lrs_name, l);
      READ_TENSOR(layer->n4, C, fp, n4_name, goto fail;);
    }
    else
    {
      layer->n4 = NULL;
    }
    dec->layers[l] = layer;
  }
  READ_TENSOR(dec->final_norm, C, fp, "model.decoder.final_norm", goto fail;);

  return dec;

fail:
  free_text_decoder(dec);
  return NULL;
}

/* */
static TextDecoder *
mmap_text_decoder(void *data, TextConfig *cfg, size_t *offset, bool quant)
{
  TextDecoder *dec;
  CALLOC(dec, 1, "model.decoder", goto fail;);
  dec->config = cfg;

  uint8_t *base = (uint8_t *)data;

  int C   = cfg->embed_dim;
  int CM  = cfg->mlp_dim;
  int Cq  = cfg->n_heads * cfg->head_dim;
  int Ckv = cfg->n_kv_heads * cfg->head_dim;
  int vs  = cfg->vocab_size;

  const char *emb_name = "model.decoder.embedding";
  const char *lrs_name = "model.decoder.layers";

  // Embedding: linear (C -> vs)
  CALLOC(dec->embedding, 1, emb_name, goto fail;);
  mmap_linear(dec->embedding, base, offset, C, vs, quant);

  // Layers array
  CALLOC(dec->layers, cfg->n_layers, lrs_name, goto fail;);  // NOLINT

  for (int l = 0; l < cfg->n_layers; l++)
  {
    char layer_name[64];
    snprintf(layer_name, sizeof(layer_name), "%s.%d", lrs_name, l);
    TextDecoderLayer *layer = NULL;
    CALLOC(layer, 1, layer_name, goto fail;);

    // Attention weights
    char wq_name[64], wk_name[64], wv_name[64], wo_name[64];
    snprintf(wq_name, sizeof(wq_name), "%s.%d.wq", lrs_name, l);
    snprintf(wk_name, sizeof(wk_name), "%s.%d.wk", lrs_name, l);
    snprintf(wv_name, sizeof(wv_name), "%s.%d.wv", lrs_name, l);
    snprintf(wo_name, sizeof(wo_name), "%s.%d.wo", lrs_name, l);

    CALLOC(layer->wq, 1, wq_name, goto fail;);
    mmap_linear(layer->wq, base, offset, C, Cq, quant);
    CALLOC(layer->wk, 1, wk_name, goto fail;);
    mmap_linear(layer->wk, base, offset, C, Ckv, quant);
    CALLOC(layer->wv, 1, wv_name, goto fail;);
    mmap_linear(layer->wv, base, offset, C, Ckv, quant);
    CALLOC(layer->wo, 1, wo_name, goto fail;);
    mmap_linear(layer->wo, base, offset, Cq, C, quant);

    // QK normalization (optional)
    if (cfg->qk_norm)
    {
      layer->nq = (floatx *)(base + *offset);
      *offset += (size_t)cfg->head_dim * sizeof(floatx);
      layer->nk = (floatx *)(base + *offset);
      *offset += (size_t)cfg->head_dim * sizeof(floatx);
    }
    else
    {
      layer->nq = NULL;
      layer->nk = NULL;
    }

    // Feedforward weights
    char w1_name[64], w2_name[64], w3_name[64];
    snprintf(w1_name, sizeof(w1_name), "%s.%d.w1", lrs_name, l);
    snprintf(w2_name, sizeof(w2_name), "%s.%d.w2", lrs_name, l);
    snprintf(w3_name, sizeof(w3_name), "%s.%d.w3", lrs_name, l);

    CALLOC(layer->w1, 1, w1_name, goto fail;);
    mmap_linear(layer->w1, base, offset, C, CM, quant);
    CALLOC(layer->w2, 1, w2_name, goto fail;);
    mmap_linear(layer->w2, base, offset, C, CM, quant);
    CALLOC(layer->w3, 1, w3_name, goto fail;);
    mmap_linear(layer->w3, base, offset, CM, C, quant);

    // RMSNorm weights
    layer->n1 = (floatx *)(base + *offset);
    *offset += (size_t)C * sizeof(floatx);
    layer->n2 = (floatx *)(base + *offset);
    *offset += (size_t)C * sizeof(floatx);

    // Optional pre/post MLP norms
    if (cfg->pre_mlp_norm)
    {
      layer->n3 = (floatx *)(base + *offset);
      *offset += (size_t)C * sizeof(floatx);
    }
    else
    {
      layer->n3 = NULL;
    }

    if (cfg->pst_mlp_norm)
    {
      layer->n4 = (floatx *)(base + *offset);
      *offset += (size_t)C * sizeof(floatx);
    }
    else
    {
      layer->n4 = NULL;
    }

    dec->layers[l] = layer;
  }

  // Final norm
  dec->final_norm = (floatx *)(base + *offset);
  *offset += (size_t)C * sizeof(floatx);

  return dec;

fail:
  free_text_decoder(dec);
  return NULL;
}

/* Vision encoder container */
typedef struct
{
  VisionConfig *config;

  floatx *patch_emb;      // (hidden_dim, 3, patch_size, patch_size)
  floatx *patch_emb_b;    // (hidden_dim,)
  Linear *pos_embedding;  // ((image_size / patch_size)^2, hidden_dim)

  VisionEncoderLayer **layers;

  floatx *post_norm;    // (hidden_dim,)
  floatx *post_norm_b;  // (hidden_dim,)
  floatx *norm;         // (hidden_dim,)
  Linear *proj;         // (hidden_dim, embed_dim).T
} VisionEncoder;

/* */
size_t
get_vision_encoder_size(
  const VisionConfig *vcfg, const TextConfig *cfg, bool quant)
{
  size_t size = 0;
  int    P    = vcfg->patch_size;
  int    C    = vcfg->hidden_dim;
  int    CM   = vcfg->mlp_dim;
  int    N    = vcfg->image_size / P;
  N *= N;

  // patch_emb
  size += (size_t)C * 3 * P * P * sizeof(floatx);
  // patch_emb_b
  size += (size_t)C * sizeof(floatx);

  // pos_embedding: linear (C -> N)
  size += linear_size(C, N, quant);

  // each layer
  for (int l = 0; l < vcfg->n_layers; l++)
  {
    // n1, n1_b
    size += (size_t)C * sizeof(floatx);
    size += (size_t)C * sizeof(floatx);

    // wq, wk, wv, wo: linear (C -> C)
    size += linear_size(C, C, quant);
    size += linear_size(C, C, quant);
    size += linear_size(C, C, quant);
    size += linear_size(C, C, quant);

    // bq, bk, bv, bo
    size += (size_t)C * sizeof(floatx);
    size += (size_t)C * sizeof(floatx);
    size += (size_t)C * sizeof(floatx);
    size += (size_t)C * sizeof(floatx);

    // n2, n2_b
    size += (size_t)C * sizeof(floatx);
    size += (size_t)C * sizeof(floatx);

    // w1: linear (C -> CM), w2: linear (CM -> C)
    size += linear_size(C, CM, quant);
    size += linear_size(CM, C, quant);

    // b1 (CM), b2 (C)
    size += (size_t)CM * sizeof(floatx);
    size += (size_t)C * sizeof(floatx);
  }

  // post_norm, post_norm_b
  size += (size_t)C * sizeof(floatx);
  size += (size_t)C * sizeof(floatx);

  // norm
  size += (size_t)C * sizeof(floatx);

  // proj: linear (C -> cfg->embed_dim)
  size += linear_size(C, cfg->embed_dim, quant);

  return size;
}

/* */
void
free_vision_encoder(VisionEncoder *enc)
{
  if (enc == NULL) return;

  free(enc->patch_emb);
  free(enc->patch_emb_b);
  free_linear(enc->pos_embedding);
  if (enc->layers != NULL && enc->config != NULL)
  {
    for (int l = 0; l < enc->config->n_layers; l++)
    {
      free_vision_layer(enc->layers[l]);
    }
    free(enc->layers);
  }
  free(enc->config);
  free(enc->post_norm);
  free(enc->post_norm_b);
  free(enc->norm);
  free_linear(enc->proj);
  free(enc);
}

/* */
void
free_vision_encoder_wrapper(VisionEncoder *enc)
{
  if (enc == NULL) return;
  free(enc->pos_embedding);
  if (enc->layers != NULL && enc->config != NULL)
  {
    for (int l = 0; l < enc->config->n_layers; l++)
    {
      free_vision_layer_wrapper(enc->layers[l]);
    }
    free(enc->layers);
  }
  free(enc->config);
  free(enc->proj);
  free(enc);
}

/* */
static VisionEncoder *
read_vision_encoder(FILE *fp, TextConfig *cfg, VisionConfig *vcfg, bool quant)
{
  VisionEncoder *enc;
  CALLOC(enc, 1, "model.encoder", goto fail;);
  enc->config = vcfg;

  int P  = vcfg->patch_size;
  int C  = vcfg->hidden_dim;
  int CM = vcfg->mlp_dim;
  int N  = vcfg->image_size / P;
  N *= N;

  const char *pem_name = "model.encoder.patch_emb";
  const char *peb_name = "model.encoder.patch_emb_b";
  const char *psm_name = "model.encoder.pos_embedding";
  const char *lrs_name = "model.encoder.layers";
  const char *pnm_name = "model.encoder.post_norm";
  const char *pnb_name = "model.encoder.post_norm_b";
  const char *nrm_name = "model.encoder.norm";
  const char *prj_name = "model.encoder.proj";

  READ_TENSOR(enc->patch_emb, C * 3 * P * P, fp, pem_name, goto fail;);
  READ_TENSOR(enc->patch_emb_b, C, fp, peb_name, goto fail;);

  // Same as here, the real shape is (N, C)
  READ_LINEAR(enc->pos_embedding, fp, C, N, quant, psm_name, goto fail;);
  CALLOC(enc->layers, vcfg->n_layers, lrs_name, goto fail;);  // NOLINT

  // Read all the layers of ViT
  for (int l = 0; l < vcfg->n_layers; l++)
  {
    char layer_name[64];
    snprintf(layer_name, sizeof(layer_name), "%s.%d", lrs_name, l);
    VisionEncoderLayer *layer = NULL;
    CALLOC(layer, 1, layer_name, goto fail;);

    // First layernorm
    char n1_name[64], n1b_name[64];
    snprintf(n1_name, sizeof(n1_name), "%s.%d.n1", lrs_name, l);
    snprintf(n1b_name, sizeof(n1b_name), "%s.%d.n1_b", lrs_name, l);
    READ_TENSOR(layer->n1, C, fp, n1_name, goto fail;);
    READ_TENSOR(layer->n1_b, C, fp, n1b_name, goto fail;);

    // Attention weights
    char wq_name[64], wk_name[64], wv_name[64], wo_name[64];
    snprintf(wq_name, sizeof(wq_name), "%s.%d.wq", lrs_name, l);
    snprintf(wk_name, sizeof(wk_name), "%s.%d.wk", lrs_name, l);
    snprintf(wv_name, sizeof(wv_name), "%s.%d.wv", lrs_name, l);
    snprintf(wo_name, sizeof(wo_name), "%s.%d.wo", lrs_name, l);

    READ_LINEAR(layer->wq, fp, C, C, quant, wq_name, goto fail;);
    READ_LINEAR(layer->wk, fp, C, C, quant, wk_name, goto fail;);
    READ_LINEAR(layer->wv, fp, C, C, quant, wv_name, goto fail;);
    READ_LINEAR(layer->wo, fp, C, C, quant, wo_name, goto fail;);

    // Attention biases
    char bq_name[64], bk_name[64], bv_name[64], bo_name[64];
    snprintf(bq_name, sizeof(bq_name), "%s.%d.bq", lrs_name, l);
    snprintf(bk_name, sizeof(bk_name), "%s.%d.bk", lrs_name, l);
    snprintf(bv_name, sizeof(bv_name), "%s.%d.bv", lrs_name, l);
    snprintf(bo_name, sizeof(bo_name), "%s.%d.bo", lrs_name, l);

    READ_TENSOR(layer->bq, C, fp, bq_name, goto fail;);
    READ_TENSOR(layer->bk, C, fp, bk_name, goto fail;);
    READ_TENSOR(layer->bv, C, fp, bv_name, goto fail;);
    READ_TENSOR(layer->bo, C, fp, bo_name, goto fail;);

    // Second layernorm
    char n2_name[64], n2b_name[64];
    snprintf(n2_name, sizeof(n2_name), "%s.%d.n2", lrs_name, l);
    snprintf(n2b_name, sizeof(n2b_name), "%s.%d.n2_b", lrs_name, l);

    READ_TENSOR(layer->n2, C, fp, n2_name, goto fail;);
    READ_TENSOR(layer->n2_b, C, fp, n2b_name, goto fail;);

    // Feedforward weights
    char w1_name[64], w2_name[64];
    snprintf(w1_name, sizeof(w1_name), "%s.%d.w1", lrs_name, l);
    snprintf(w2_name, sizeof(w2_name), "%s.%d.w2", lrs_name, l);
    READ_LINEAR(layer->w1, fp, C, CM, quant, w1_name, goto fail;);
    READ_LINEAR(layer->w2, fp, CM, C, quant, w2_name, goto fail;);

    // Feedforward biases
    char b1_name[64], b2_name[64];
    snprintf(b1_name, sizeof(b1_name), "%s.%d.b1", lrs_name, l);
    snprintf(b2_name, sizeof(b2_name), "%s.%d.b2", lrs_name, l);
    READ_TENSOR(layer->b1, CM, fp, b1_name, goto fail;);
    READ_TENSOR(layer->b2, C, fp, b2_name, goto fail;);

    enc->layers[l] = layer;
  }

  // Post layernorm
  READ_TENSOR(enc->post_norm, C, fp, pnm_name, goto fail;);
  READ_TENSOR(enc->post_norm_b, C, fp, pnb_name, goto fail;);

  // Soft embedding RMSNorm
  READ_TENSOR(enc->norm, C, fp, nrm_name, goto fail;);
  // Final projection
  READ_LINEAR(enc->proj, fp, C, cfg->embed_dim, quant, prj_name, goto fail;);

  return enc;

fail:
  free_vision_encoder(enc);
  return NULL;
}

/* */
static VisionEncoder *
mmap_vision_encoder(
  void *data, TextConfig *cfg, VisionConfig *vcfg, size_t *offset, bool quant)
{
  VisionEncoder *enc;
  CALLOC(enc, 1, "model.encoder", goto fail;);
  enc->config = vcfg;

  uint8_t *base = (uint8_t *)data;

  int P  = vcfg->patch_size;
  int C  = vcfg->hidden_dim;
  int CM = vcfg->mlp_dim;
  int N  = vcfg->image_size / P;
  N *= N;

  const char *psm_name = "model.encoder.pos_embedding";
  const char *lrs_name = "model.encoder.layers";
  const char *prj_name = "model.encoder.proj";

  // patch_emb: C * 3 * P * P floats
  enc->patch_emb = (floatx *)(base + *offset);
  *offset += (size_t)C * 3 * P * P * sizeof(floatx);

  // patch_emb_b: C floats
  enc->patch_emb_b = (floatx *)(base + *offset);
  *offset += (size_t)C * sizeof(floatx);

  // pos_embedding: linear (C -> N)
  CALLOC(enc->pos_embedding, 1, psm_name, goto fail;);
  mmap_linear(enc->pos_embedding, base, offset, C, N, quant);

  // layers array
  CALLOC(enc->layers, vcfg->n_layers, lrs_name, goto fail;);  // NOLINT

  for (int l = 0; l < vcfg->n_layers; l++)
  {
    char layer_name[64];
    snprintf(layer_name, sizeof(layer_name), "%s.%d", lrs_name, l);
    VisionEncoderLayer *layer = NULL;
    CALLOC(layer, 1, layer_name, goto fail;);

    // n1, n1_b
    layer->n1 = (floatx *)(base + *offset);
    *offset += (size_t)C * sizeof(floatx);
    layer->n1_b = (floatx *)(base + *offset);
    *offset += (size_t)C * sizeof(floatx);

    // attention weights: wq, wk, wv, wo (linear C -> C)
    char wq_name[64], wk_name[64], wv_name[64], wo_name[64];
    snprintf(wq_name, sizeof(wq_name), "%s.%d.wq", lrs_name, l);
    snprintf(wk_name, sizeof(wk_name), "%s.%d.wk", lrs_name, l);
    snprintf(wv_name, sizeof(wv_name), "%s.%d.wv", lrs_name, l);
    snprintf(wo_name, sizeof(wo_name), "%s.%d.wo", lrs_name, l);

    CALLOC(layer->wq, 1, wq_name, goto fail;);
    mmap_linear(layer->wq, base, offset, C, C, quant);
    CALLOC(layer->wk, 1, wk_name, goto fail;);
    mmap_linear(layer->wk, base, offset, C, C, quant);
    CALLOC(layer->wv, 1, wv_name, goto fail;);
    mmap_linear(layer->wv, base, offset, C, C, quant);
    CALLOC(layer->wo, 1, wo_name, goto fail;);
    mmap_linear(layer->wo, base, offset, C, C, quant);

    // attention biases: bq, bk, bv, bo
    layer->bq = (floatx *)(base + *offset);
    *offset += (size_t)C * sizeof(floatx);
    layer->bk = (floatx *)(base + *offset);
    *offset += (size_t)C * sizeof(floatx);
    layer->bv = (floatx *)(base + *offset);
    *offset += (size_t)C * sizeof(floatx);
    layer->bo = (floatx *)(base + *offset);
    *offset += (size_t)C * sizeof(floatx);

    // n2, n2_b
    layer->n2 = (floatx *)(base + *offset);
    *offset += (size_t)C * sizeof(floatx);
    layer->n2_b = (floatx *)(base + *offset);
    *offset += (size_t)C * sizeof(floatx);

    // feedforward weights: w1 (C -> CM), w2 (CM -> C)
    char w1_name[64], w2_name[64];
    snprintf(w1_name, sizeof(w1_name), "%s.%d.w1", lrs_name, l);
    snprintf(w2_name, sizeof(w2_name), "%s.%d.w2", lrs_name, l);

    CALLOC(layer->w1, 1, w1_name, goto fail;);
    mmap_linear(layer->w1, base, offset, C, CM, quant);
    CALLOC(layer->w2, 1, w2_name, goto fail;);
    mmap_linear(layer->w2, base, offset, CM, C, quant);

    // feedforward biases: b1 (CM), b2 (C)
    layer->b1 = (floatx *)(base + *offset);
    *offset += (size_t)CM * sizeof(floatx);
    layer->b2 = (floatx *)(base + *offset);
    *offset += (size_t)C * sizeof(floatx);

    enc->layers[l] = layer;
  }

  // post_norm, post_norm_b
  enc->post_norm = (floatx *)(base + *offset);
  *offset += (size_t)C * sizeof(floatx);
  enc->post_norm_b = (floatx *)(base + *offset);
  *offset += (size_t)C * sizeof(floatx);

  // norm
  enc->norm = (floatx *)(base + *offset);
  *offset += (size_t)C * sizeof(floatx);

  // proj: linear (C -> embed_dim)
  CALLOC(enc->proj, 1, prj_name, goto fail;);
  mmap_linear(enc->proj, base, offset, C, cfg->embed_dim, quant);

  return enc;

fail:
  free_vision_encoder(enc);
  return NULL;
}

/* Top-level model container */
typedef struct
{
  TextDecoder    *decoder;
  VisionEncoder  *encoder;
  GemmaTokenizer *tokenizer;
  bool            quant;  // W8A8
  bool            support_mm;
  void           *mmap_data;
  size_t          mmap_size;
} GemmaModel;

/* */
void
free_gemma_model(GemmaModel *model)
{
  if (model == NULL) return;
  free_tokenizer(model->tokenizer);
  free_vision_encoder(model->encoder);
  free_text_decoder(model->decoder);
  free(model);
}

/* */
void
munmap_gemma_model(GemmaModel *model)
{
  if (model == NULL) return;
  free_tokenizer(model->tokenizer);
  munmap(model->mmap_data, model->mmap_size);
  free_vision_encoder_wrapper(model->encoder);
  free_text_decoder_wrapper(model->decoder);
  free(model);
}

/* Check the file header and return a flag that indicates whether the model
 * is quantized. Return -1 if failed. */
static int
check_head(FILE *fp, const char *filename)
{
  // Magic header
  char hdr[8];
  FREAD(hdr, 8, fp, "header", return -1;);
  if (memcmp(hdr, "GEMA", 4) != 0)
  {
    fprintf(stderr, "error: not a valid gemma model file: %s\n", filename);
    return -1;
  }
  // dtype
  if (memcmp(hdr + 4, DTYPE_MAGIC, 3) != 0)
  {
    fprintf(
      stderr, "error: unsupported dtype '%c%c%c' in model file\n", hdr[4],
      hdr[5], hdr[6]
    );
    return -1;
  }
  // W8A8 flag ('Q' / 'U')
  if (hdr[7] == 'Q')
  {
    return 1;
  }
  else if (hdr[7] != 'U')
  {
    fprintf(stderr, "error: not a valid gemma model file: %s\n", filename);
    return -1;
  }

  return 0;
}

/* Map the entire text & vision model into virtual memory */
GemmaModel *
mmap_gemma_model(const char *filename, bool enable_mm)
{
  FILE       *fp    = fopen(filename, "rb");
  GemmaModel *model = NULL;

  if (fp == NULL)
  {
    fprintf(stderr, "error: failed to open file: %s\n", filename);
    goto fail;
  }
  CALLOC(model, 1, "model", goto fail;);

  // Header
  int quant = check_head(fp, filename);
  if (quant == -1) goto fail;
  model->quant = (bool)quant;

  // Text config
  TextConfig *cfg = read_text_config(fp, &model->support_mm);
  if (cfg == NULL) goto fail;

  bool use_mm = model->support_mm && enable_mm;

  // Vision config
  VisionConfig *vcfg = NULL;
  if (use_mm)
  {
    vcfg = read_vision_config(fp);
    if (vcfg == NULL) goto fail;
  }
  else if (model->support_mm)
  {
    // Skip vision config if user didn't ask for multimodal
    if (read_vision_config(fp) == NULL) goto fail;
  }

  // Tokenizer
  GemmaTokenizer *tok = read_tokenizer(fp, cfg, model->support_mm);
  if (tok == NULL) goto fail;
  model->tokenizer = tok;

  size_t offset = ftell(fp);
  fclose(fp);

  int fd = open(filename, O_RDONLY);  // read-only
  if (fd == -1)
  {
    fprintf(stderr, "error: failed to open file: %s\n", filename);
    goto fail;
  }

  // Get the total amount of bytes for the weights
  size_t mmap_size = offset + get_text_decoder_size(cfg, model->quant);
  if (use_mm)
  {
    mmap_size += get_vision_encoder_size(vcfg, cfg, model->quant);
  }

  void *data = mmap(NULL, mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (data == MAP_FAILED)
  {
    fprintf(stderr, "error: memory mapping failed: %s\n", filename);
    goto fail;
  }
  model->mmap_data = data;
  model->mmap_size = mmap_size;
  close(fd);

  // Text decoder
  TextDecoder *dec = mmap_text_decoder(data, cfg, &offset, model->quant);
  if (dec == NULL) goto fail;
  model->decoder = dec;

  // Vision encoder
  if (use_mm)
  {
    VisionEncoder *enc =
      mmap_vision_encoder(data, cfg, vcfg, &offset, model->quant);
    if (enc == NULL) goto fail;
    model->encoder = enc;
  }

  return model;

fail:
  if (fp != NULL) fclose(fp);
  free(model);
  return NULL;
}

/* Load the entire text & vision model into RAM */
GemmaModel *
read_gemma_model(const char *filename, bool enable_mm)
{
  FILE       *fp    = fopen(filename, "rb");
  GemmaModel *model = NULL;

  if (fp == NULL)
  {
    fprintf(stderr, "error: failed to open file: %s\n", filename);
    goto fail;
  }
  CALLOC(model, 1, "model", goto fail;);

  // Header
  int quant = check_head(fp, filename);
  if (quant == -1) goto fail;
  model->quant = (bool)quant;

  // Text config
  TextConfig *cfg = read_text_config(fp, &model->support_mm);
  if (cfg == NULL) goto fail;

  bool use_mm = model->support_mm && enable_mm;

  // Vision config
  VisionConfig *vcfg = NULL;
  if (use_mm)
  {
    vcfg = read_vision_config(fp);
    if (vcfg == NULL) goto fail;
  }
  else if (model->support_mm)
  {
    // Skip vision config if user didn't ask for multimodal
    if (read_vision_config(fp) == NULL) goto fail;
  }

  // Tokenizer
  GemmaTokenizer *tok = read_tokenizer(fp, cfg, model->support_mm);
  if (tok == NULL) goto fail;
  model->tokenizer = tok;

  // Text decoder
  TextDecoder *dec = read_text_decoder(fp, cfg, model->quant);
  if (dec == NULL) goto fail;
  model->decoder = dec;

  // Vision encoder
  if (use_mm)
  {
    VisionEncoder *enc = read_vision_encoder(fp, cfg, vcfg, model->quant);
    if (enc == NULL) goto fail;
    model->encoder = enc;
  }

  fclose(fp);
  return model;

fail:
  if (fp != NULL) fclose(fp);
  free(model);
  return NULL;
}

// Runtime buffers (allocated once, reused every step)

/**
 * SigLIP vision model runtime buffer
 * N       = n_patches
 * C       = hidden_dim
 * CM      = mlp_dim
 * tpi     = image_toks
 * C_embed = embed_dim
 */
typedef struct
{
  int8_t *x_i8;        // (N, C)
  floatx *x_scales;    // (N,)
  int8_t *mlp_i8;      // (N, CM)
  floatx *mlp_scales;  // (N)

  floatx *x;           // (N, C)
  floatx *resid;       // (N, C)
  floatx *xq;          // (N, C)
  floatx *xk;          // (N, C)
  floatx *xv;          // (N, C)
  floatx *att_out;     // (N, C)
  floatx *mlp_hidden;  // (N, CM)
  floatx *scores;      // (NH, N, N)
} VisionBuffer;

/* */
void
free_vision_buffer(VisionBuffer *buf, bool quant)
{
  if (buf == NULL) return;
  if (quant)
  {
    free(buf->x_i8);
    free(buf->x_scales);
    free(buf->mlp_i8);
    free(buf->mlp_scales);
  }
  free(buf->x);
  free(buf->resid);
  free(buf->xq);
  free(buf->xk);
  free(buf->xv);
  free(buf->att_out);
  free(buf->mlp_hidden);
  free(buf->scores);
  free(buf);
}

/* */
VisionBuffer *
malloc_vision_buffer(VisionConfig *vcfg, bool quant)
{
  VisionBuffer *buf = NULL;
  CALLOC(buf, 1, "vbuf", goto fail;);

  size_t C   = (size_t)vcfg->hidden_dim;
  size_t ppi = (size_t)vcfg->image_size / vcfg->patch_size;
  size_t N   = (size_t)ppi * ppi;
  size_t CM  = (size_t)vcfg->mlp_dim;
  size_t NH  = (size_t)vcfg->n_heads;

  if (quant)
  {
    // Quantization buffers
    MALLOC(buf->x_i8, N * C, "vbuf.x_i8", goto fail;);
    MALLOC(buf->mlp_i8, N * CM, "vbuf.mlp_i8", goto fail;);
    MALLOC(buf->x_scales, N, "vbuf.x_scales", goto fail;);
    MALLOC(buf->mlp_scales, N, "vbuf.mlp_scales", goto fail;);
  }

  MALLOC(buf->x, N * C, "vbuf.x", goto fail;);
  MALLOC(buf->resid, N * C, "vbuf.resid", goto fail;);
  MALLOC(buf->xq, N * C, "vbuf.q", goto fail;);
  MALLOC(buf->xk, N * C, "vbuf.k", goto fail;);
  MALLOC(buf->xv, N * C, "vbuf.v", goto fail;);
  MALLOC(buf->att_out, N * C, "vbuf.att_out", goto fail;);
  MALLOC(buf->mlp_hidden, N * CM, "vbuf.mlp_hidden", goto fail;);
  MALLOC(buf->scores, NH * N * N, "vbuf.scores", goto fail;);

  return buf;

fail:
  free_vision_buffer(buf, quant);
  return NULL;
}

/**
 * Gemma language model runtime buffer
 * T     = number of input tokens (used in prefilling)
 * L     = n_layers
 * C     = embed_dim
 * CM    = mlp_dim
 * NH    = n_heads
 * NH_kv = n_kv_heads
 */
typedef struct
{
  int cache_len;

  // Temporary quantized activations (when quant=true)
  int8_t *x_i8;       // ([T], C,)
  floatx *x_scales;   // ([T],)
  int8_t *xo_i8;      // (NH, [T], CH)
  floatx *xo_scales;  // ([T],)
  int8_t *xg_i8;      // ([T], CM,)
  floatx *xg_scales;  // ([T],)

  // Pre-computed cos/sin for RoPE
  floatx *csfreqs_slid;  // ([T], CH / 2, 2)
  floatx *csfreqs_full;  // ([T], CH / 2, 2)

  // Residual stream
  floatx *x;      // ([T], C,)
  floatx *resid;  // ([T], C,)

  // Attention buffers
  floatx *xq;        // (NH, [T], CH)
  floatx *xk;        // (NH_kv, [T], CH)
  floatx *xv;        // (NH_kv, CH, [T])
  floatx *xo;        // ([T], NH, CH)
  floatx *att;       // (NH, [T], cache_len)
  floatx *kv_cache;  // (L, 2, NH_kv, cache_len, CH)

  // MLP buffers
  floatx *xg;      // ([T], CM,)
  floatx *xu;      // ([T], CM,)
  floatx *logits;  // (vocab_size,)
} TextBuffer;

/* */
void
free_text_buffer(TextBuffer *buf, bool quant)
{
  if (buf == NULL) return;

  free(buf->x);
  free(buf->resid);
  free(buf->xq);
  free(buf->xk);
  free(buf->csfreqs_slid);
  free(buf->csfreqs_full);
  free(buf->xv);
  free(buf->xo);
  free(buf->att);
  free(buf->kv_cache);
  free(buf->xg);
  free(buf->xu);
  free(buf->logits);
  if (quant)
  {
    free(buf->x_i8);
    free(buf->x_scales);
    free(buf->xo_i8);
    free(buf->xo_scales);
    free(buf->xg_i8);
    free(buf->xg_scales);
  }
  free(buf);
}

/* */
TextBuffer *
malloc_text_buffer(
  TextConfig   *cfg,
  VisionConfig *vcfg,
  int           cache_len,
  int           chunk_size,
  bool          use_mm,
  bool          quant)
{
  TextBuffer *buf = NULL;
  CALLOC(buf, 1, "buf", goto fail;);  // Init to all NULL

  buf->cache_len = cache_len;

  int C  = cfg->embed_dim;
  int L  = cfg->n_layers;
  int CH = cfg->head_dim;
  int NH = cfg->n_heads;
  int CM = cfg->mlp_dim;

  int Cq  = NH * CH;
  int Ckv = cfg->n_kv_heads * CH;

  MALLOC(buf->kv_cache, (size_t)L * 2 * (size_t)cache_len * (size_t)Ckv,
    "buf.kv_cache", goto fail;);
  MALLOC(buf->logits, cfg->vocab_size, "buf.logits", goto fail;);

  int mult = chunk_size;

  // Multimodal models need space for a whole image worth of tokens
  if (use_mm && vcfg != NULL)
  {
    mult = max(mult, cfg->image_toks);
  }

  if (quant)
  {
    // Quantization buffers
    MALLOC(buf->x_i8, mult * C, "buf.x_i8", goto fail;);
    MALLOC(buf->xo_i8, mult * Cq, "buf.xo_i8", goto fail;);
    MALLOC(buf->xg_i8, mult * CM, "buf.xg_i8", goto fail;);
    MALLOC(buf->x_scales, mult, "buf.x_scales", goto fail;);
    MALLOC(buf->xo_scales, mult, "buf.xo_scales", goto fail;);
    MALLOC(buf->xg_scales, mult, "buf.xg_scales", goto fail;);
  }

  MALLOC(buf->x, mult * C, "buf.x", goto fail;);
  MALLOC(buf->resid, mult * C, "buf.resid", goto fail;);
  MALLOC(buf->xq, mult * Cq, "buf.xq", goto fail;);
  MALLOC(buf->xk, mult * Ckv, "buf.xk", goto fail;);
  MALLOC(buf->csfreqs_slid, mult * CH, "buf.csfreqs_slid", goto fail;);
  MALLOC(buf->csfreqs_full, mult * CH, "buf.csfreqs_full", goto fail;);
  MALLOC(buf->xv, mult * Ckv, "buf.xv", goto fail;);
  MALLOC(buf->xo, mult * Cq, "buf.xo", goto fail;);
  MALLOC(buf->att, mult * NH * cache_len, "buf.att", goto fail;);
  MALLOC(buf->xg, mult * CM, "buf.xg", goto fail;);
  MALLOC(buf->xu, mult * CM, "buf.xu", goto fail;);

  return buf;

fail:
  free_text_buffer(buf, quant);
  return NULL;
}

// Math primitives

// Make sure the clamping is not optimized by compilers
#if defined(__GNUC__) && !defined(__clang__)  // GCC
#  define CLAMP_NO_FAST_MATH __attribute__((optimize("no-fast-math")))
#elif defined(__clang__)  // clang
#  define CLAMP_NO_FAST_MATH
#elif defined(_MSC_VER)  // MSVC
#  define CLAMP_NO_FAST_MATH
#else
#  define CLAMP_NO_FAST_MATH
#endif

#if defined(__clang__)
#  pragma float_control(precise, on, push)
#elif defined(_MSC_VER)
#  pragma float_control(push)
#  pragma float_control(precise, on)
#endif

/* */
static inline floatx CLAMP_NO_FAST_MATH
clamp_fpx(floatx v)
{
  return (floatx)fminf(FLOATX_MAX, fmaxf((float)-FLOATX_MAX, (float)v));
}

#if defined(__clang__) || defined(_MSC_VER)
#  pragma float_control(pop)
#endif

/* Gemma-style RMSNorm: (x * rsqrt(mean(x^2) + eps)) * (weight + 1) */
static void
rmsnorm(
  floatx *dst, const floatx *src, const floatx *weight, int dim, float eps)
{
  float sqsum = 0.0f;
  #pragma omp simd reduction(+ : sqsum)
  for (int i = 0; i < dim; i++)
  {
    sqsum += (float)src[i] * (float)src[i];
  }
  float rms = 1.0f / sqrtf(sqsum / (float)dim + eps);
  #pragma omp simd
  for (int i = 0; i < dim; i++)
  {
    dst[i] = (floatx)((float)src[i] * rms * (float)(weight[i] + 1));
  }
}

/* Classic LayerNorm used inside the SigLIP vision tower */
static void
layernorm(
  floatx       *dst,
  const floatx *src,
  const floatx *weight,
  const floatx *bias,
  int           dim,
  float         eps)
{
  float mean = 0.0f;
  #pragma omp simd reduction(+ : mean)
  for (int i = 0; i < dim; i++)
  {
    mean += (float)src[i];
  }
  mean /= (float)dim;

  float var = 0.0f;
  #pragma omp simd reduction(+ : var)
  for (int i = 0; i < dim; i++)
  {
    float diff = (float)src[i] - mean;
    var += diff * diff;
  }
  var /= (float)dim;

  float inv_std = 1.0f / sqrtf(var + eps);
  #pragma omp simd
  for (int i = 0; i < dim; i++)
  {
    dst[i] = (floatx)(((float)src[i] - mean) * inv_std * (float)weight[i] +
                      (float)bias[i]);
  }
}

/* Symmetric per-vector quantization into int8 [-127, 127]
 * Q(fpx src (dim,)) ~= int8 dst (dim,) * fpx vec_scale (1,) */
static floatx
quantize_act(int8_t *dst, const floatx *vec, int dim)
{
  floatx amax = 0.0f;

  #pragma omp simd reduction(max : amax)
  for (int d = 0; d < dim; d++)
  {
    floatx av = vec[d] >= 0 ? vec[d] : -vec[d];
    if (av > amax)
    {
      amax = av;
    }
  }

  float vec_scale = (float)amax > 0.0f ? (float)amax / 127.0f : 1.0f;

  for (int d = 0; d < dim; d++)
  {
    int q = (int)roundf((float)vec[d] / vec_scale);
    if (q > 127)
    {
      q = 127;
    }
    else if (q < -127)
    {
      q = -127;
    }
    dst[d] = (int8_t)q;
  }

  return (floatx)vec_scale;
}

/* Symmetric int8 quantization for a matrix of rows
 * Q(fpx src (m, n)) ~= int8 dst (m, n) * fpx dst_scales (m,) */
static void
quantize_acts(
  int8_t *restrict       dst,
  floatx *restrict       dst_scales,
  const floatx *restrict src,
  int                    src_stride,
  int                    m,
  int                    n,
  bool                   omp)
{
  if (src_stride == 0)
  {
    src_stride = n;
  }

  if (omp)
  {
    #pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
      dst_scales[i] = quantize_act(dst + i * n, src + i * src_stride, n);
    }
    return;
  }
  for (int i = 0; i < m; i++)
  {
    dst_scales[i] = quantize_act(dst + i * n, src + i * src_stride, n);
  }
}

/* */
static inline floatx
gemv_fpx_row(
  const floatx *restrict vec, const floatx *restrict mat, int n, int i)
{
  float sum = 0;
  #pragma omp simd reduction(+ : sum)
  for (int j = 0; j < n; j++)
  {
    sum += (float)mat[i * n + j] * (float)vec[j];
  }
  return (floatx)sum;
}

/* fpx matrix-vector multiply (NT)
 * fpx vec (n,) @ fpx mat (m, n).T = fpx dst (m,) */
static void
gemv_fpx(
  floatx *restrict       dst,
  const floatx *restrict mat,
  const floatx *restrict vec,
  int                    m,
  int                    n,
  bool                   omp)
{
  if (omp)
  {
    #pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
      dst[i] = gemv_fpx_row(vec, mat, n, i);
    }
    return;
  }
  for (int i = 0; i < m; i++)
  {
    dst[i] = gemv_fpx_row(vec, mat, n, i);
  }
}

/* */
static inline floatx
gemv_int8_row(
  const int8_t *restrict vec,
  floatx                 vec_scale,
  const int8_t *restrict mat,
  const floatx *restrict mat_scales,
  int                    n,
  int                    i)
{
  int32_t sum = 0;
  #pragma omp simd reduction(+ : sum)
  for (int j = 0; j < n; j++)
  {
    sum += (int32_t)mat[i * n + j] * (int32_t)vec[j];
  }
  return (floatx)((float)sum * (float)vec_scale * (float)mat_scales[i]);
}

/* int8 matrix-vector multiply + dequant (NT)
 *   (int8 vec (n,)   * fpx vec_scale  (1,))
 * @ (int8 mat (m, n) * fpx mat_scales (m,)).T = fpx dst (m,) */
static void
gemv_int8(
  floatx *restrict       dst,
  const int8_t *restrict mat,
  const floatx *restrict mat_scales,
  const int8_t *restrict vec,
  floatx                 vec_scale,
  int                    m,
  int                    n,
  bool                   omp)
{
  if (omp)
  {
    #pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
      dst[i] = gemv_int8_row(vec, vec_scale, mat, mat_scales, n, i);
    }
    return;
  }

  for (int i = 0; i < m; i++)
  {
    dst[i] = gemv_int8_row(vec, vec_scale, mat, mat_scales, n, i);
  }
}

/* fpx matrix-vector multiply (NN)
 * fpx vec (n,) @ fpx mat (n, m) = fpx dst (m,) */
static void
gemv_fpx_nn(
  floatx *restrict       dst,
  const floatx *restrict vec,
  const floatx *restrict mat,
  int                    mat_stride,
  int                    n,
  int                    m)
{
  if (mat_stride == 0) mat_stride = m;

  float acc[m];
  for (int j = 0; j < m; j++)
  {
    acc[j] = 0.0f;
  }

  for (int i = 0; i < n; i++)
  {
    float         v       = (float)vec[i];
    const floatx *mat_row = mat + i * mat_stride;
    #pragma omp simd
    for (int j = 0; j < m; j++)
    {
      acc[j] += v * (float)mat_row[j];
    }
  }

  for (int j = 0; j < m; j++)
  {
    dst[j] = (floatx)acc[j];
  }
}

/* Copy `rows` rows (stride `row_stride`, length k each) of `src` into a
 * contiguous, already-upcast-to-float buffer laid out as `packed[l*rows + r]`,
 * so the micro-kernel can read all `rows` values for a fixed `l` with one
 * contiguous load instead of `rows` separate strided reads. */
static inline void
pack_panel(
  float *restrict        packed,
  const floatx *restrict src,
  int                    row_stride,
  int                    rows,
  int                    k)
{
  for (int r = 0; r < rows; r++)
  {
    const floatx *row = src + r * row_stride;
    #pragma omp simd
    for (int l = 0; l < k; l++)
    {
      packed[l * rows + r] = (float)row[l];
    }
  }
}

/* Pack every R-row panel of a (total_rows x k) matrix, back to back, into one
 * contiguous buffer. Parallelized because for large prefill chunks this alone
 * touches every element of `mat` once. */
static void
pack_all(
  float *restrict        packed_full,
  const floatx *restrict mat,
  int                    stride,
  int                    total_rows,
  int                    R,
  int                    k,
  bool                   omp)
{
  if (omp)
  {
    #pragma omp parallel for
    for (int base = 0; base < total_rows; base += R)
    {
      pack_panel(
        packed_full + (size_t)base * k, mat + (size_t)base * stride, stride, R,
        k
      );
    }
  }
  else
  {
    for (int base = 0; base < total_rows; base += R)
    {
      pack_panel(
        packed_full + (size_t)base * k, mat + (size_t)base * stride, stride, R,
        k
      );
    }
  }
}

/* Compute one 8x8 tile of dst = src @ mat.T from PACKED, contiguous
 * MR/NR-major panels (see pack_panel/pack_all). The 64 FMAs are spelled out by
 * hand rather than as a nested i/j loop. GCC vectorizes the nested-loop form
 * just fine, but Clang's optimizer can only produces good code once the
 * accumulation is fully unrolled with compile-time-constant indices :'D. So I
 * just hardcoded for GEMM_NT_MR == GEMM_NT_NR == 8. */
static inline void
gemm_fpx_kernel(
  floatx *restrict      dst,
  int                   dst_stride,
  const float *restrict Bp,
  const float *restrict Ap,
  int                   k)
{
  float acc[64] = {0};

  for (int l = 0; l < k; l++)
  {
    float a0 = Ap[l * 8 + 0];
    float a1 = Ap[l * 8 + 1];
    float a2 = Ap[l * 8 + 2];
    float a3 = Ap[l * 8 + 3];
    float a4 = Ap[l * 8 + 4];
    float a5 = Ap[l * 8 + 5];
    float a6 = Ap[l * 8 + 6];
    float a7 = Ap[l * 8 + 7];
    float b0 = Bp[l * 8 + 0];
    float b1 = Bp[l * 8 + 1];
    float b2 = Bp[l * 8 + 2];
    float b3 = Bp[l * 8 + 3];
    float b4 = Bp[l * 8 + 4];
    float b5 = Bp[l * 8 + 5];
    float b6 = Bp[l * 8 + 6];
    float b7 = Bp[l * 8 + 7];

    acc[ 0] += a0*b0; acc[ 1] += a0*b1;
    acc[ 2] += a0*b2; acc[ 3] += a0*b3;
    acc[ 4] += a0*b4; acc[ 5] += a0*b5;
    acc[ 6] += a0*b6; acc[ 7] += a0*b7;
    acc[ 8] += a1*b0; acc[ 9] += a1*b1;
    acc[10] += a1*b2; acc[11] += a1*b3;
    acc[12] += a1*b4; acc[13] += a1*b5;
    acc[14] += a1*b6; acc[15] += a1*b7;
    acc[16] += a2*b0; acc[17] += a2*b1;
    acc[18] += a2*b2; acc[19] += a2*b3;
    acc[20] += a2*b4; acc[21] += a2*b5;
    acc[22] += a2*b6; acc[23] += a2*b7;
    acc[24] += a3*b0; acc[25] += a3*b1;
    acc[26] += a3*b2; acc[27] += a3*b3;
    acc[28] += a3*b4; acc[29] += a3*b5;
    acc[30] += a3*b6; acc[31] += a3*b7;
    acc[32] += a4*b0; acc[33] += a4*b1;
    acc[34] += a4*b2; acc[35] += a4*b3;
    acc[36] += a4*b4; acc[37] += a4*b5;
    acc[38] += a4*b6; acc[39] += a4*b7;
    acc[40] += a5*b0; acc[41] += a5*b1;
    acc[42] += a5*b2; acc[43] += a5*b3;
    acc[44] += a5*b4; acc[45] += a5*b5;
    acc[46] += a5*b6; acc[47] += a5*b7;
    acc[48] += a6*b0; acc[49] += a6*b1;
    acc[50] += a6*b2; acc[51] += a6*b3;
    acc[52] += a6*b4; acc[53] += a6*b5;
    acc[54] += a6*b6; acc[55] += a6*b7;
    acc[56] += a7*b0; acc[57] += a7*b1;
    acc[58] += a7*b2; acc[59] += a7*b3;
    acc[60] += a7*b4; acc[61] += a7*b5;
    acc[62] += a7*b6; acc[63] += a7*b7;
  }

  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++)
      dst[i * dst_stride + j] = (floatx)acc[i * 8 + j];
}

/* Scalar fallback for the remainder tiles. */
static inline void
gemm_fpx_scalar(
  floatx *restrict       dst,
  int                    dst_stride,
  const floatx *restrict mat,
  int                    mat_stride,
  const floatx *restrict src,
  int                    src_stride,
  int                    m,
  int                    n,
  int                    k)
{
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
    {
      float         sum     = 0.0f;
      const floatx *src_row = src + i * src_stride;
      const floatx *w_row   = mat + j * mat_stride;
      #pragma omp simd reduction(+ : sum)
      for (int l = 0; l < k; l++)
        sum += (float)src_row[l] * (float)w_row[l];
      dst[i * dst_stride + j] = (floatx)sum;
    }
}

// _Thread_local is critical here since these will be used in openmp threads
static _Thread_local float *gemm_fpx_pack_scratch     = NULL;
static _Thread_local size_t gemm_fpx_pack_scratch_cap = 0;   // in floats

/* */
static float *
gemm_pack_scratch_get(size_t needed_floats)
{
  if (needed_floats > gemm_fpx_pack_scratch_cap)
  {
    float *tmp = realloc(gemm_fpx_pack_scratch, needed_floats * sizeof(float));
    if (tmp == NULL) return NULL;
    gemm_fpx_pack_scratch     = tmp;
    gemm_fpx_pack_scratch_cap = needed_floats;
  }
  return gemm_fpx_pack_scratch;
}

/* fpx matrix-matrix multiply (NT)
 * fpx src (m, k) @ fpx mat.T (k, n) = fpx dst (m, n) */
static void
gemm_fpx(
  floatx *restrict       dst,
  int                    dst_stride,
  const floatx *restrict mat,
  int                    mat_stride,
  const floatx *restrict src,
  int                    src_stride,
  int                    m,
  int                    n,
  int                    k,
  bool                   omp)
{
  if (dst_stride == 0) dst_stride = n;
  if (mat_stride == 0) mat_stride = k;
  if (src_stride == 0) src_stride = k;

  int m_full = (m / GEMM_NT_MR) * GEMM_NT_MR;
  int n_full = (n / GEMM_NT_NR) * GEMM_NT_NR;

  if (m_full > 0 && n_full > 0)
  {
    size_t needed = (size_t)(m_full + n_full) * k;
    float *scratch = gemm_pack_scratch_get(needed);
    if (scratch == NULL)
    {
      fprintf(stderr, "error: gemm packing scratch allocation failed\n");
      return;
    }
    float *Ap_full = scratch;
    float *Bp_full = scratch + (size_t)m_full * k;

    pack_all(Ap_full, src, src_stride, m_full, GEMM_NT_MR, k, omp);
    pack_all(Bp_full, mat, mat_stride, n_full, GEMM_NT_NR, k, omp);

    if (omp)
    {
      #pragma omp parallel for collapse(2)
      for (int jb = 0; jb < n_full; jb += GEMM_NT_NR)
        for (int ib = 0; ib < m_full; ib += GEMM_NT_MR)
        {
          gemm_fpx_kernel(
            dst + ib * dst_stride + jb, dst_stride, Bp_full + (size_t)jb * k,
            Ap_full + (size_t)ib * k, k
          );
        }
    }
    else
    {
      for (int jb = 0; jb < n_full; jb += GEMM_NT_NR)
        for (int ib = 0; ib < m_full; ib += GEMM_NT_MR)
        {
          gemm_fpx_kernel(
            dst + ib * dst_stride + jb, dst_stride, Bp_full + (size_t)jb * k,
            Ap_full + (size_t)ib * k, k
          );
        }
    }
  }

  // Remainder: leftover rows (full width) + leftover columns (remaining
  // height only, so the (m_full:m, n_full:n) corner isn't done twice)
  if (m_full < m)
  {
    gemm_fpx_scalar(
      dst + m_full * dst_stride, dst_stride, mat, mat_stride,
      src + m_full * src_stride, src_stride, m - m_full, n, k
    );
  }
  if (n_full < n && m_full > 0)
  {
    gemm_fpx_scalar(
      dst + n_full, dst_stride, mat + n_full * mat_stride, mat_stride, src,
      src_stride, m_full, n - n_full, k
    );
  }
}

/* Compute an `mr`x`n` (mr <= GEMM_NN_MR) block of dst = src @ mat.
 * `mat` is converted from floatx -> float once per `l` (into `row`) and
 * then reused across all `mr` accumulator rows, instead of being
 * re-read/re-converted once per row like the naive version. */
static inline void
gemm_fpx_nn_kernel(
  floatx *restrict       dst,
  int                    dst_stride,
  const floatx *restrict mat,
  int                    mat_stride,
  const floatx *restrict src,
  int                    src_stride,
  int                    mr,
  int                    n,
  int                    k)
{
  float acc[GEMM_NN_MR][n];
  float row[n];
  for (int ii = 0; ii < mr; ii++)
    memset(acc[ii], 0, n * sizeof(float));

  for (int l = 0; l < k; l++)
  {
    const floatx *mat_row = mat + l * mat_stride;
    #pragma omp simd
    for (int j = 0; j < n; j++)
    {
      row[j] = (float)mat_row[j];
    }

    for (int ii = 0; ii < mr; ii++)
    {
      float a = (float)src[ii * src_stride + l];
      #pragma omp simd
      for (int j = 0; j < n; j++)
      {
        acc[ii][j] += a * row[j];
      }
    }
  }

  for (int ii = 0; ii < mr; ii++)
  {
    floatx *dst_row = dst + ii * dst_stride;
    for (int j = 0; j < n; j++)
    {
      dst_row[j] = (floatx)acc[ii][j];
    }
  }
}

/* fpx matrix-matrix multiply (NN)
 * fpx src (m, k) @ fpx mat (k, n) = fpx dst (m, n) */
static void
gemm_fpx_nn(
  floatx *restrict       dst,
  int                    dst_stride,
  const floatx *restrict mat,
  int                    mat_stride,
  const floatx *restrict src,
  int                    src_stride,
  int                    m,
  int                    n,
  int                    k,
  bool                   omp)
{
  if (dst_stride == 0) dst_stride = n;
  if (mat_stride == 0) mat_stride = n;
  if (src_stride == 0) src_stride = k;

  int m_full = (m / GEMM_NN_MR) * GEMM_NN_MR;

  if (omp)
  {
    #pragma omp parallel for
    for (int ib = 0; ib < m_full; ib += GEMM_NN_MR)
    {
      gemm_fpx_nn_kernel(
        dst + ib * dst_stride, dst_stride, mat, mat_stride,
        src + ib * src_stride, src_stride, GEMM_NN_MR, n, k
      );
    }
  }
  else
  {
    for (int ib = 0; ib < m_full; ib += GEMM_NN_MR)
    {
      gemm_fpx_nn_kernel(
        dst + ib * dst_stride, dst_stride, mat, mat_stride,
        src + ib * src_stride, src_stride, GEMM_NN_MR, n, k
      );
    }
  }

  // Remainder: leftover rows (m % GEMM_NN_MR), still full width
  if (m_full < m)
  {
    gemm_fpx_nn_kernel(
      dst + m_full * dst_stride, dst_stride, mat, mat_stride,
      src + m_full * src_stride, src_stride, m - m_full, n, k
    );
  }
}

static _Thread_local int8_t *gemm_i8_pack_scratch     = NULL;
static _Thread_local size_t  gemm_i8_pack_scratch_cap = 0;

/* */
static int8_t *
gemm_i8_pack_scratch_get(size_t needed_bytes)
{
  if (needed_bytes > gemm_i8_pack_scratch_cap)
  {
    int8_t *tmp = realloc(gemm_i8_pack_scratch, needed_bytes);
    if (tmp == NULL) return NULL;
    gemm_i8_pack_scratch     = tmp;
    gemm_i8_pack_scratch_cap = needed_bytes;
  }
  return gemm_i8_pack_scratch;
}

/* Same idea as pack_panel/pack_all for fpx, but source/dest are int8_t, no
 * upcast needed, this is a pure gather-to-contiguous transpose. */
static inline void
pack_panel_i8(
  int8_t *restrict       packed,
  const int8_t *restrict src,
  int                    row_stride,
  int                    rows,
  int                    k)
{
  for (int r = 0; r < rows; r++)
  {
    const int8_t *row = src + r * row_stride;
    #pragma omp simd
    for (int l = 0; l < k; l++)
    {
      packed[l * rows + r] = row[l];
    }
  }
}

/* */
static void
pack_all_i8(
  int8_t *restrict       packed_full,
  const int8_t *restrict mat,
  int                    stride,
  int                    total_rows,
  int                    R,
  int                    k,
  bool                   omp)
{
  if (omp)
  {
    #pragma omp parallel for
    for (int base = 0; base < total_rows; base += R)
    {
      pack_panel_i8(
        packed_full + (size_t)base * k, mat + (size_t)base * stride, stride, R, k
      );
    }
  }
  else
  {
    for (int base = 0; base < total_rows; base += R)
    {
      pack_panel_i8(
        packed_full + (size_t)base * k, mat + (size_t)base * stride, stride, R, k
      );
    }
  }
}

/* Hardcoded for GEMM_I8_MR == GEMM_I8_NR == 8 */
static inline void
gemm_int8_kernel(
  floatx *restrict dst, int dst_stride, const int8_t *restrict Bp,
  const floatx *restrict mat_scales, const int8_t *restrict Ap,
  const floatx *restrict src_scales, int k
)
{
  int32_t acc[64] = {0};

  for (int l = 0; l < k; l++)
  {
    int32_t a0 = Ap[l*8+0];
    int32_t a1 = Ap[l*8+1];
    int32_t a2 = Ap[l*8+2];
    int32_t a3 = Ap[l*8+3];
    int32_t a4 = Ap[l*8+4];
    int32_t a5 = Ap[l*8+5];
    int32_t a6 = Ap[l*8+6];
    int32_t a7 = Ap[l*8+7];
    int32_t b0 = Bp[l*8+0];
    int32_t b1 = Bp[l*8+1];
    int32_t b2 = Bp[l*8+2];
    int32_t b3 = Bp[l*8+3];
    int32_t b4 = Bp[l*8+4];
    int32_t b5 = Bp[l*8+5];
    int32_t b6 = Bp[l*8+6];
    int32_t b7 = Bp[l*8+7];

    acc[ 0] += a0*b0; acc[ 1] += a0*b1;
    acc[ 2] += a0*b2; acc[ 3] += a0*b3;
    acc[ 4] += a0*b4; acc[ 5] += a0*b5;
    acc[ 6] += a0*b6; acc[ 7] += a0*b7;
    acc[ 8] += a1*b0; acc[ 9] += a1*b1;
    acc[10] += a1*b2; acc[11] += a1*b3;
    acc[12] += a1*b4; acc[13] += a1*b5;
    acc[14] += a1*b6; acc[15] += a1*b7;
    acc[16] += a2*b0; acc[17] += a2*b1;
    acc[18] += a2*b2; acc[19] += a2*b3;
    acc[20] += a2*b4; acc[21] += a2*b5;
    acc[22] += a2*b6; acc[23] += a2*b7;
    acc[24] += a3*b0; acc[25] += a3*b1;
    acc[26] += a3*b2; acc[27] += a3*b3;
    acc[28] += a3*b4; acc[29] += a3*b5;
    acc[30] += a3*b6; acc[31] += a3*b7;
    acc[32] += a4*b0; acc[33] += a4*b1;
    acc[34] += a4*b2; acc[35] += a4*b3;
    acc[36] += a4*b4; acc[37] += a4*b5;
    acc[38] += a4*b6; acc[39] += a4*b7;
    acc[40] += a5*b0; acc[41] += a5*b1;
    acc[42] += a5*b2; acc[43] += a5*b3;
    acc[44] += a5*b4; acc[45] += a5*b5;
    acc[46] += a5*b6; acc[47] += a5*b7;
    acc[48] += a6*b0; acc[49] += a6*b1;
    acc[50] += a6*b2; acc[51] += a6*b3;
    acc[52] += a6*b4; acc[53] += a6*b5;
    acc[54] += a6*b6; acc[55] += a6*b7;
    acc[56] += a7*b0; acc[57] += a7*b1;
    acc[58] += a7*b2; acc[59] += a7*b3;
    acc[60] += a7*b4; acc[61] += a7*b5;
    acc[62] += a7*b6; acc[63] += a7*b7;
  }

  for (int i = 0; i < 8; i++)
  {
    float fscale = (float)src_scales[i];
    for (int j = 0; j < 8; j++)
    {
      dst[i * dst_stride + j] = (floatx)((float)acc[i*8+j] * fscale * (float)mat_scales[j]);
    }
  }
}

/* Scalar fallback for the remainder tiles. */
static inline void
gemm_int8_scalar(
  floatx *restrict       dst,
  int                    dst_stride,
  const int8_t *restrict mat,
  int                    mat_stride,
  const floatx *restrict mat_scales,
  const int8_t *restrict src,
  int                    src_stride,
  const floatx *restrict src_scales,
  int                    m,
  int                    n,
  int                    k)
{
  for (int i = 0; i < m; i++)
  {
    float fscale = (float)src_scales[i];
    for (int j = 0; j < n; j++)
    {
      int32_t       sum     = 0;
      const int8_t *src_row = src + i * src_stride;
      const int8_t *mat_row = mat + j * mat_stride;
      #pragma omp simd reduction(+ : sum)
      for (int l = 0; l < k; l++)
      {
        sum += (int32_t)src_row[l] * (int32_t)mat_row[l];
      }
      float val               = (float)sum * fscale * (float)mat_scales[j];
      dst[i * dst_stride + j] = (floatx)val;
    }
  }
}


/* int8 matrix-matrix multiply (NT) + dequant
 *
 *   (int8 src (m, k) * fpx src_scales (m,))
 * @ (int8 mat (n, k) * fpx mat_scales (n,)).T = (fpx dst (m, n)) */
static void
gemm_int8(
  floatx *restrict       dst,
  int                    dst_stride,
  const int8_t *restrict mat,
  int                    mat_stride,
  const floatx *restrict mat_scales,
  const int8_t *restrict src,
  int                    src_stride,
  const floatx *restrict src_scales,
  int                    m,
  int                    n,
  int                    k,
  bool                   omp)
{
  if (dst_stride == 0) dst_stride = n;
  if (mat_stride == 0) mat_stride = k;
  if (src_stride == 0) src_stride = k;

  int m_full = (m / GEMM_I8_MR) * GEMM_I8_MR;
  int n_full = (n / GEMM_I8_NR) * GEMM_I8_NR;

  if (m_full > 0 && n_full > 0)
  {
    int8_t *scratch = gemm_i8_pack_scratch_get((size_t)(m_full + n_full) * k);
    if (scratch == NULL)
    {
      fprintf(stderr, "error: gemm_int8 packing scratch allocation failed\n");
      return;
    }
    int8_t *Ap_full = scratch;
    int8_t *Bp_full = scratch + (size_t)m_full * k;

    pack_all_i8(Ap_full, src, src_stride, m_full, GEMM_I8_MR, k, omp);
    pack_all_i8(Bp_full, mat, mat_stride, n_full, GEMM_I8_NR, k, omp);

    if (omp)
    {
      #pragma omp parallel for collapse(2)
      for (int jb = 0; jb < n_full; jb += GEMM_I8_NR)
        for (int ib = 0; ib < m_full; ib += GEMM_I8_MR)
        {
          gemm_int8_kernel(
            dst + ib * dst_stride + jb, dst_stride, Bp_full + (size_t)jb * k,
            mat_scales + jb, Ap_full + (size_t)ib * k, src_scales + ib, k
          );
        }
    }
    else
    {
      for (int jb = 0; jb < n_full; jb += GEMM_I8_NR)
        for (int ib = 0; ib < m_full; ib += GEMM_I8_MR)
        {
          gemm_int8_kernel(
            dst + ib * dst_stride + jb, dst_stride, Bp_full + (size_t)jb * k,
            mat_scales + jb, Ap_full + (size_t)ib * k, src_scales + ib, k
          );
        }
    }
  }

  // Remainder: leftover rows (full width) + leftover columns (remaining
  // height only, so the (m_full:m, n_full:n) corner isn't done twice)
  if (m_full < m)
  {
    gemm_int8_scalar(
      dst + m_full * dst_stride, dst_stride, mat, mat_stride, mat_scales,
      src + m_full * src_stride, src_stride, src_scales + m_full, m - m_full, n,
      k
    );
  }
  if (n_full < n)
  {
    gemm_int8_scalar(
      dst + n_full, dst_stride, mat + n_full * mat_stride, mat_stride,
      mat_scales + n_full, src, src_stride, src_scales, m_full, n - n_full, k
    );
  }
}

/* */
static void
softmax(floatx *dst, const floatx *src, int dim)
{
  floatx max = -FLOATX_MAX;
  for (int i = 0; i < dim; i++)
  {
    if (src[i] > max)
    {
      max = src[i];
    }
  }

  float expsum = 0.0f;
  for (int i = 0; i < dim; i++)
  {
    float val = expf((float)(src[i] - max));
    dst[i]    = (floatx)val;
    expsum += val;
  }

  #pragma omp simd
  for (int i = 0; i < dim; i++)
  {
    dst[i] = (floatx)((float)dst[i] / expsum);
  }
}

/* */
floatx *
prepare_image(const char *path, int image_size)
{
  // Load the image
  int            rows, cols, channels;
  unsigned char *raw = stbi_load(path, &cols, &rows, &channels, 3);
  if (raw == NULL)
  {
    return NULL;
  }

  // Resize the image
  unsigned char *rsz = stbir_resize(
    raw, cols, rows, 0,
    NULL, image_size, image_size, 0,
    STBIR_RGB, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM
  );
  stbi_image_free(raw);
  if (rsz == NULL)
  {
    return NULL;
  }

  floatx *out;
  MALLOC(out, image_size * image_size * 3, "image_buf", {
    free(rsz);
    return NULL;
  });

  for (int i = 0; i < image_size * image_size * 3; i++)
  {
    out[i] = (rsz[i] / 255.0f - 0.5f) / 0.5f;
  }

  free(rsz);
  return out;
}

/* */
floatx *
prepare_image_pas(
  const char *path,
  int         image_size,
  int         min_crop_size,
  int         max_crops,
  int        *n_crops)
{ 
  int             rows, cols, channels;
  unsigned char  *raw = NULL;
  unsigned char  *rsz = NULL;
  floatx         *out = NULL;

  size_t epi = (size_t)image_size * image_size * 3;  // Elements per image

  *n_crops = 0;

  // Load the full image
  raw = stbi_load(path, &cols, &rows, &channels, 3);
  if (raw == NULL) goto fail;

  // Compute the number of crops (W / H)
  int wn_crops = 1;
  int hn_crops = 1;
  if (cols >= rows)
  {
    if ((float)cols / rows >= 1.5)
    {
      wn_crops = (int)floor((double)cols / rows + 0.5);
      int cap  = (int)floor((double)cols / min_crop_size);
      wn_crops = min(max(min(cap, wn_crops), 2), max_crops);
    }
  }
  else
  {
    if ((float)rows / cols >= 1.5)
    {
      hn_crops = (int)floor((double)rows / cols + 0.5);
      int cap  = (int)floor((double)rows / min_crop_size);
      hn_crops = min(max(min(cap, hn_crops), 2), max_crops);
    }
  }

  // Size of each crop
  int cw = (cols + wn_crops - 1) / wn_crops;
  int ch = (rows + hn_crops - 1) / hn_crops;

  // Fallback to full size if too small
  if (cw < min_crop_size || ch < min_crop_size)
  {
    wn_crops = hn_crops = 0;
  }

  *n_crops = wn_crops * hn_crops;

  // Allocate resize buffer
  MALLOC(rsz, (size_t)(*n_crops + 1) * epi, "pas_resize", goto fail;);
  // Allocate out container
  CALLOC(out, (size_t)(*n_crops + 1) * epi, "pas_crops", goto fail;);

  // The first entry represents the full image (resized)
  unsigned char *rsz_r = stbir_resize(
    raw, cols, rows, cols * 3,
    rsz, image_size, image_size, 0,
    STBIR_RGB, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM
  );
  if (rsz_r == NULL) goto fail;
  for (size_t i = 0; i < epi; i++)
  {
    out[i] = (rsz[i] / 255.0f - 0.5f) / 0.5f;
  }

  for (int i = 0; i < hn_crops; i++)
    for (int j = 0; j < wn_crops; j++)
    {
      int idx = i * wn_crops + j + 1;  // +1 to skip over the first entry

      int px = j * cw;
      int py = i * ch;
      int pw = cw;
      int ph = ch;

      pw = min(pw, cols - px);
      ph = min(ph, rows - py);
      
      // Resize crop
      unsigned char *rsz_crop = rsz + (size_t)idx * epi;
      unsigned char *raw_crop = raw + ((size_t)py * cols + px) * 3;

      rsz_r = stbir_resize(
        raw_crop, pw, ph, cols * 3,
        rsz_crop, image_size, image_size, 0,
        STBIR_RGB, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM
      );
      if (rsz_r == NULL) goto fail;

      // Normalize into out
      floatx *dst = out + (size_t)idx * epi;
      for (size_t k = 0; k < epi; k++)
      {
        dst[k] = (rsz_crop[k] / 255.0f - 0.5f) / 0.5f;
      }
    }

  (*n_crops)++;
  stbi_image_free(raw);
  free(rsz);
  return out;

fail:
  stbi_image_free(raw);
  free(rsz);
  free(out);
  *n_crops = 0;
  return NULL;
}

// Forward passes

/* Vision forward pass (SigLIP)
 * img: (img_sz, img_sz, 3) */
int
forward_vision(
  VisionEncoder *enc,
  TextConfig    *cfg,
  TextBuffer    *buf,
  VisionBuffer  *vbuf,
  const floatx  *img,
  bool           quant)
{
  VisionConfig *vcfg = enc->config;

  int C        = vcfg->hidden_dim;
  int P        = vcfg->patch_size;
  int img_sz   = vcfg->image_size;
  int ppi      = img_sz / P;
  int N        = ppi * ppi;
  int tpi      = cfg->image_toks;
  int side_len = (int)roundf(sqrtf((float)tpi));
  int K        = (img_sz / vcfg->patch_size) / side_len;

  int CH     = C / vcfg->n_heads;
  int CM     = vcfg->mlp_dim;
  int in_dim = 3 * P * P;

  // Patch Embedding

  // Iterate over all the patch
  // Can be further optimized by reusing GEMM, but this part is executed only
  // once per call, the cost is acceptable
  #pragma omp parallel for collapse(2)
  for (int oy = 0; oy < ppi; oy++)
    for (int ox = 0; ox < ppi; ox++)
    {
      int patch_idx = oy * ppi + ox;

      // Compute the embed vector for the current patch
      for (int oc = 0; oc < C; oc++)
      {
        float sum = 0.0f;

        // equivalent to Conv2d(
        //   in_channels=3, out_channels=C, kernal_size=P, stride=P, bias=True)
        #pragma omp simd collapse(3)
        for (int py = 0; py < P; py++)
          for (int px = 0; px < P; px++)
            for (int c = 0; c < 3; c++)
            {
              // img[c, oy*P + py, ox*P + px]
              int in_idx = ((oy * P + py) * img_sz + (ox * P + px)) * 3 + c;
              // patch_emb[oc, c, py, px]
              int w_idx = oc * in_dim + c * P * P + py * P + px;
              sum += (float)enc->patch_emb[w_idx] * (float)img[in_idx];
            }

        vbuf->x[patch_idx * C + oc] =
          clamp_fpx((floatx)(sum + (float)enc->patch_emb_b[oc]));
      }
    }

  if (is_interrupted()) return 1;

  // Position Embedding
  if (!quant)
  {
    #pragma omp simd
    for (int d = 0; d < N * C; d++)
    {
      vbuf->x[d] = clamp_fpx(vbuf->x[d] + enc->pos_embedding->fpx[d]);
    }
  }
  else
  {
    // Dequantize per row
    #pragma omp simd collapse(2)
    for (int i = 0; i < N; i++)
      for (int j = 0; j < C; j++)
      {
        float scale        = (float)enc->pos_embedding->i8.scales[i];
        float val          = (float)enc->pos_embedding->i8.q[i * C + j] * scale;
        vbuf->x[i * C + j] = clamp_fpx(vbuf->x[i * C + j] + (floatx)val);
      }
  }

  if (is_interrupted()) return 1;

  // Encoder Layers
  for (int l = 0; l < vcfg->n_layers; l++)
  {
    VisionEncoderLayer *layer = enc->layers[l];

    memcpy(vbuf->resid, vbuf->x, N * C * sizeof(floatx));
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
      floatx *row = vbuf->x + i * C;
      layernorm(row, row, layer->n1, layer->n1_b, C, vcfg->eps);
    }
    if (is_interrupted()) return 1;

    // QKV projections
    if (!quant)
    {
      gemm_fpx(vbuf->xq, 0, layer->wq->fpx, 0, vbuf->x, 0, N, C, C, true);
      gemm_fpx(vbuf->xk, 0, layer->wk->fpx, 0, vbuf->x, 0, N, C, C, true);
      gemm_fpx(vbuf->xv, 0, layer->wv->fpx, 0, vbuf->x, 0, N, C, C, true);
    }
    else
    {
      quantize_acts(vbuf->x_i8, vbuf->x_scales, vbuf->x, 0, N, C, true);
      gemm_int8(
        vbuf->xq, 0, layer->wq->i8.q, 0, layer->wq->i8.scales, vbuf->x_i8, 0,
        vbuf->x_scales, N, C, C, true
      );
      gemm_int8(
        vbuf->xk, 0, layer->wk->i8.q, 0, layer->wk->i8.scales, vbuf->x_i8, 0,
        vbuf->x_scales, N, C, C, true
      );
      gemm_int8(
        vbuf->xv, 0, layer->wv->i8.q, 0, layer->wv->i8.scales, vbuf->x_i8, 0,
        vbuf->x_scales, N, C, C, true
      );
    }
    if (is_interrupted()) return 1;

    // Add biases
    #pragma omp simd collapse(2)
    for (int i = 0; i < N; i++)
      for (int j = 0; j < C; j++)
      {
        int idx = i * C + j;
        vbuf->xq[idx] += layer->bq[j];
        vbuf->xk[idx] += layer->bk[j];
        vbuf->xv[idx] += layer->bv[j];
      }
    if (is_interrupted()) return 1;

    // Attention
    memset(vbuf->att_out, 0, N * C * sizeof(floatx));
    float scale = 1.0f / sqrtf((float)CH);

    #pragma omp parallel for
    for (int h = 0; h < vcfg->n_heads; h++)
    {
      floatx *scores = vbuf->scores + h * N * N;  // (N, N) for this head

      // scores = Q @ K^T * scale
      gemm_fpx(
        scores, /*dst_stride=*/N, vbuf->xk + h * CH, /*mat_stride=*/C,
        vbuf->xq + h * CH, /*src_stride=*/C, N, N, CH, false
      );

      // Apply scale and softmax per row
      for (int i = 0; i < N; i++)
      {
        floatx *row = scores + i * N;
        #pragma omp simd
        for (int j = 0; j < N; j++)
        {
          row[j] *= scale;
        }
        softmax(row, row, N);
      }

      // Weighted sum of values: out = scores @ V
      gemm_fpx_nn(
        vbuf->att_out + h * CH, /*dst_stride=*/C, vbuf->xv + h * CH,
        /*mat_stride=*/C, scores, /*src_stride=*/N, N, CH, N, false
      );
    }
    if (is_interrupted()) return 1;

    // Output projection
    if (!quant)
    {
      gemm_fpx(vbuf->x, 0, layer->wo->fpx, 0, vbuf->att_out, 0, N, C, C, true);
    }
    else
    {
      quantize_acts(vbuf->x_i8, vbuf->x_scales, vbuf->att_out, 0, N, C, true);
      gemm_int8(
        vbuf->x, 0, layer->wo->i8.q, 0, layer->wo->i8.scales, vbuf->x_i8, 0,
        vbuf->x_scales, N, C, C, true
      );
    }
    if (is_interrupted()) return 1;

    // Add output bias
    #pragma omp simd collapse(2)
    for (int i = 0; i < N; i++)
      for (int j = 0; j < C; j++)
      {
        vbuf->x[i * C + j] += layer->bo[j];
      }
    if (is_interrupted()) return 1;

    // Residual connection
    for (int i = 0; i < N * C; i++)
    {
      vbuf->x[i] = clamp_fpx(vbuf->x[i] + vbuf->resid[i]);
    }
    if (is_interrupted()) return 1;

    memcpy(vbuf->resid, vbuf->x, N * C * sizeof(floatx));
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
      floatx *row = vbuf->x + i * C;
      layernorm(row, row, layer->n2, layer->n2_b, C, vcfg->eps);
    }
    if (is_interrupted()) return 1;

    // x @ fc1 = mlp_hidden
    if (!quant)
    {
      gemm_fpx(
        vbuf->mlp_hidden, 0, layer->w1->fpx, 0, vbuf->x, 0, N, CM, C, true
      );
    }
    else
    {
      quantize_acts(vbuf->x_i8, vbuf->x_scales, vbuf->x, 0, N, C, true);
      gemm_int8(
        vbuf->mlp_hidden, 0, layer->w1->i8.q, 0, layer->w1->i8.scales,
        vbuf->x_i8, 0, vbuf->x_scales, N, CM, C, true
      );
    }
    if (is_interrupted()) return 1;

    #pragma omp simd collapse(2)
    for (int i = 0; i < N; i++)
      for (int j = 0; j < CM; j++)
      {
        // Apply fc1 biases
        float val = (float)vbuf->mlp_hidden[i * CM + j] + (float)layer->b1[j];
        // GELU tanh approximation
        float c = 0.79788456080287f;
        val =
          0.5f * val * (1.0f + tanhf(c * (val + 0.044715f * val * val * val)));
        vbuf->mlp_hidden[i * CM + j] = (floatx)val;
      }
    if (is_interrupted()) return 1;

    // mlp_hidden @ fc2 = x
    if (!quant)
    {
      gemm_fpx(
        vbuf->x, 0, layer->w2->fpx, 0, vbuf->mlp_hidden, 0, N, C, CM, true
      );
    }
    else
    {
      quantize_acts(
        vbuf->mlp_i8, vbuf->mlp_scales, vbuf->mlp_hidden, 0, N, CM, true
      );
      gemm_int8(
        vbuf->x, 0, layer->w2->i8.q, 0, layer->w2->i8.scales, vbuf->mlp_i8, 0,
        vbuf->mlp_scales, N, C, CM, true
      );
    }
    if (is_interrupted()) return 1;

    // x += b2
    #pragma omp simd collapse(2)
    for (int i = 0; i < N; i++)
      for (int j = 0; j < C; j++)
      {
        vbuf->x[i * C + j] += layer->b2[j];
      }

    if (is_interrupted()) return 1;

    // Residual connection
    #pragma omp parallel for
    for (int i = 0; i < N * C; i++)
    {
      vbuf->x[i] = clamp_fpx(vbuf->x[i] + vbuf->resid[i]);
    }
    if (is_interrupted()) return 1;
  }

  // Post-norm + average pooling down to image_toks tokens
  #pragma omp parallel for
  for (int i = 0; i < N; i++)
  {
    floatx *row = vbuf->x + i * C;
    layernorm(row, row, enc->post_norm, enc->post_norm_b, C, vcfg->eps);
  }
  if (is_interrupted()) return 1;

  // Average pooling (AvgPool2d(kernel_size=K, stride=K))

  /* Be careful that this loop must remain serial, in-place pooling writes
   * results to the front of vbuf->x, which overlaps with input data needed by
   * other output positions. Parallelizing this loop introduces a read/write
   * race condition. */
  for (int oy = 0; oy < side_len; oy++)
    for (int ox = 0; ox < side_len; ox++)
    {
      int out_idx = (oy * side_len + ox) * C;
      for (int d = 0; d < C; d++)
      {
        float sum = 0.0f;
        for (int ky = 0; ky < K; ky++)
        {
          #pragma omp simd reduction(+ : sum)
          for (int kx = 0; kx < K; kx++)
          {
            int py        = oy * K + ky;
            int px        = ox * K + kx;
            int token_idx = (py * ppi + px) * C + d;
            sum += (float)vbuf->x[token_idx];
          }
        }
        vbuf->x[out_idx + d] = (floatx)(sum / (K * K));
      }
    }

  if (is_interrupted()) return 1;
  // buf->x now becomes (tpi, C)

  // Final RMSNorm
  #pragma omp parallel for
  for (int i = 0; i < tpi; i++)
  {
    floatx *row = vbuf->x + i * C;
    rmsnorm(row, row, enc->norm, C, vcfg->eps);
  }
  if (is_interrupted()) return 1;

  // Final projection into language model embedding space
  // x (tpi, embed_dim) = x (tpi, C) @ proj (C, embed_dim)
  if (!quant)
  {
    gemm_fpx(
      buf->x, 0, enc->proj->fpx, 0, vbuf->x, 0, tpi, cfg->embed_dim, C, true
    );
  }
  else
  {
    quantize_acts(vbuf->x_i8, vbuf->x_scales, vbuf->x, 0, tpi, C, true);
    gemm_int8(
      buf->x, 0, enc->proj->i8.q, 0, enc->proj->i8.scales, vbuf->x_i8, 0,
      vbuf->x_scales, tpi, cfg->embed_dim, C, true
    );
  }
  if (is_interrupted()) return 1;

  return 0;
}

/* Language model forward (one token) */
int
forward_text_decode(
  TextDecoder *dec, TextBuffer *buf, int pos, bool quant, bool compute_logits)
{
  TextConfig *cfg = dec->config;

  int C     = cfg->embed_dim;
  int NH    = cfg->n_heads;
  int NH_kv = cfg->n_kv_heads;
  int CH    = cfg->head_dim;
  int Cq    = NH * CH;
  int Ckv   = NH_kv * CH;

  int CH_half = CH / 2;

  if (pos >= buf->cache_len)
  {
    fprintf(stderr, "\nerror: KV Cache is full\n");
    return 1;
  }

  // Precompute cos & sin for all frequencies (used in RoPE)
  for (int d = 0; d < CH_half; d++)
  {
    float freq;
    float e = (float)(-2 * d) / (float)CH;  // exponent

    // Rotation angles for sliding window attentions
    freq                         = powf(cfg->local_theta, e);
    buf->csfreqs_slid[d * 2]     = (floatx)cosf(freq * (float)pos);
    buf->csfreqs_slid[d * 2 + 1] = (floatx)sinf(freq * (float)pos);

    // Rotation angles for full attentions
    freq                         = powf(cfg->global_theta, e);
    buf->csfreqs_full[d * 2]     = (floatx)cosf(freq * (float)pos);
    buf->csfreqs_full[d * 2 + 1] = (floatx)sinf(freq * (float)pos);
  }
  if (is_interrupted()) return 1;

  // Forward all the layers
  for (int l = 0; l < cfg->n_layers; l++)
  {
    TextDecoderLayer *layer = dec->layers[l];

    memcpy(buf->resid, buf->x, C * sizeof(*buf->x));

    rmsnorm(buf->x, buf->x, layer->n1, C, cfg->eps);
    if (is_interrupted()) return 1;

    // The attention block
    if (!quant)
    {
      gemv_fpx(buf->xq, layer->wq->fpx, buf->x, Cq, C, true);  // (NH, CH)
      // (NH_kv, CH)
      gemv_fpx(buf->xk, layer->wk->fpx, buf->x, Ckv, C, true);
      gemv_fpx(buf->xv, layer->wv->fpx, buf->x, Ckv, C, true);
    }
    else
    {
      floatx x_scale = quantize_act(buf->x_i8, buf->x, C);
      gemv_int8(
        buf->xq, layer->wq->i8.q, layer->wq->i8.scales, buf->x_i8, x_scale, Cq,
        C, true
      );
      gemv_int8(
        buf->xk, layer->wk->i8.q, layer->wk->i8.scales, buf->x_i8, x_scale, Ckv,
        C, true
      );
      gemv_int8(
        buf->xv, layer->wv->i8.q, layer->wv->i8.scales, buf->x_i8, x_scale, Ckv,
        C, true
      );
    }
    if (is_interrupted()) return 1;

    if (cfg->qk_norm)
    {
      // Query RMSNorm
      for (int h = 0; h < NH; h++)
      {
        floatx *xq_head = buf->xq + h * CH;
        // Use the non-threading version here since we are running this over
        // every head
        rmsnorm(xq_head, xq_head, layer->nq, CH, cfg->eps);
      }
      // Key RMSNorm
      for (int h = 0; h < NH_kv; h++)
      {
        floatx *xk_head = buf->xk + h * CH;
        rmsnorm(xk_head, xk_head, layer->nk, CH, cfg->eps);
      }
    }
    if (is_interrupted()) return 1;

    bool    is_local = cfg->att_layers[l];
    floatx *freqs_cs = is_local ? buf->csfreqs_slid : buf->csfreqs_full;

    // Apply RoPE to queries & keys
    for (int idx = 0; idx < NH + NH_kv; idx++)
    {
      floatx *data;
      if (idx < NH)
      {
        data = buf->xq + idx * CH;  // Apply to queries
      }
      else
      {
        data = buf->xk + (idx - NH) * CH;  // Apply to keys
      }

      for (int d = 0; d < CH_half; d++)
      {
        float cfr = (float)freqs_cs[2 * d];
        float sfr = (float)freqs_cs[2 * d + 1];
        float a   = (float)data[d];            // Index in the first half vector
        float b   = (float)data[d + CH_half];  // ... second half vector

        data[d]           = (floatx)(a * cfr - b * sfr);
        data[d + CH_half] = (floatx)(a * sfr + b * cfr);
      }
    }
    if (is_interrupted()) return 1;

    // (NH_kv, cache_len, CH)
    floatx *k_cache = buf->kv_cache + l * 2 * buf->cache_len * Ckv;
    floatx *v_cache = k_cache + buf->cache_len * Ckv;

    // Write to kv_cache
    for (int h = 0; h < NH_kv; h++)
    {
      floatx *xk_head = k_cache + h * buf->cache_len * CH + pos * CH;
      floatx *xv_head = v_cache + h * buf->cache_len * CH + pos * CH;
      memcpy(xk_head, buf->xk + h * CH, CH * sizeof(*buf->xk));
      memcpy(xv_head, buf->xv + h * CH, CH * sizeof(*buf->xv));
    }
    if (is_interrupted()) return 1;

    // Sliding-window (true) or full attention (false)?
    bool local_att = is_local && pos >= cfg->slide_len;
    // Starting position of kv_cache
    int spos   = local_att ? (pos + 1 - cfg->slide_len) : 0;
    int attlen = pos + 1 - spos;  // Include the current pos

    floatx att_scale = (floatx)(1.0f / sqrtf((float)cfg->q_scale));

    // Iterate over all the attention heads
    #pragma omp parallel for
    for (int h = 0; h < NH; h++)
    {
      int h_kv = h * NH_kv / NH;  // GQA mapping

      floatx *xq_head = buf->xq + h * CH;  // xq[h, :]
      // k_cache[h_kv, spos:, :]
      floatx *xk_head = k_cache + h_kv * buf->cache_len * CH + spos * CH;
      // att[h, spos:]
      floatx *att_head = buf->att + h * attlen + spos;

      // Compute dot product of the current query across all the keys
      gemv_fpx(att_head, xk_head, xq_head, attlen, CH, false);
      for (int t = 0; t < attlen; t++)
      {
        att_head[t] *= att_scale;
      }

      // Attention score softcapping
      if (cfg->att_softcap != 0.0f)
      {
        for (int t = 0; t < attlen; t++)
        {
          float val   = (float)att_head[t] / cfg->att_softcap;
          att_head[t] = (floatx)(tanhf(val) * cfg->att_softcap);
        }
      }

      // Softmax
      softmax(att_head, att_head, attlen);

      // Compute output as weighted sum of values
      // v_cache[h_kv, spos:, :]
      floatx *xv_head = v_cache + h_kv * buf->cache_len * CH + spos * CH;
      floatx *xo_head = buf->xo + h * CH;

      // xo_head (CH,) = att_head (attlen,) @ xv_head (attlen, CH)
      gemv_fpx_nn(xo_head, att_head, xv_head, 0, attlen, CH);
    }
    if (is_interrupted()) return 1;

    // Output projection maps xo back to x
    // x (C,) = xo (CH,) @ wo.T (CH, C)
    if (!quant)
    {
      gemv_fpx(buf->x, layer->wo->fpx, buf->xo, C, Cq, true);
    }
    else
    {
      floatx xo_scale = quantize_act(buf->xo_i8, buf->xo, Cq);
      gemv_int8(
        buf->x, layer->wo->i8.q, layer->wo->i8.scales, buf->xo_i8, xo_scale, C,
        Cq, true
      );
    }
    if (is_interrupted()) return 1;

    rmsnorm(buf->x, buf->x, layer->n2, C, cfg->eps);
    if (is_interrupted()) return 1;

    // Combine the residual stream
    floatx *restrict x     = buf->x;
    floatx *restrict resid = buf->resid;
    for (int d = 0; d < C; d++)
    {
      /* Sometimes the residual stream accumulates huge values on certain
       * channels, especially in pretrained/bigger models (Sun et al., 2024,
       * https://arxiv.org/abs/2402.17762). It works fine in fp32 or bf16, but
       * it can easily overflow fp16 and become inf, causing all the activations
       * turning into nan after the next RMSNorm, so we need to clamp it. */

      /* NOTE: Actually this should never trigger now since I added activation
       * scalers afterwards (see export.py), the clamp here is more of a
       * last-resort safety net. */

      buf->x[d] = clamp_fpx(x[d] + resid[d]);
    }
    if (is_interrupted()) return 1;

    memcpy(buf->resid, buf->x, C * sizeof(*buf->x));

    // Pre feedforward RMSNorm
    if (cfg->pre_mlp_norm)
    {
      rmsnorm(buf->x, buf->x, layer->n3, C, cfg->eps);
    }
    if (is_interrupted()) return 1;

    // MLP feedforward layer (SwiGLU-style)
    if (!quant)
    {
      gemv_fpx(buf->xu, layer->w1->fpx, buf->x, cfg->mlp_dim, C, true);
      gemv_fpx(buf->xg, layer->w2->fpx, buf->x, cfg->mlp_dim, C, true);
    }
    else
    {
      floatx x_scale = quantize_act(buf->x_i8, buf->x, C);
      gemv_int8(
        buf->xg, layer->w2->i8.q, layer->w2->i8.scales, buf->x_i8, x_scale,
        cfg->mlp_dim, C, true
      );
      gemv_int8(
        buf->xu, layer->w1->i8.q, layer->w1->i8.scales, buf->x_i8, x_scale,
        cfg->mlp_dim, C, true
      );
    }
    if (is_interrupted()) return 1;

    // GELU gate
    #pragma omp parallel for
    for (int d = 0; d < cfg->mlp_dim; d++)
    {
      // Tanh approximation of GELU
      float x    = (float)buf->xg[d];
      float c    = 0.79788456080287f;  // = sqrt(2 / pi)
      x          = 0.5 * x * (1 + tanhf(c * (x + 0.044715 * x * x * x)));
      buf->xg[d] = (floatx)x;
      buf->xg[d] *= buf->xu[d];  // Fuse xg * xu into xg
    }
    if (is_interrupted()) return 1;

    if (!quant)
    {
      gemv_fpx(buf->x, layer->w3->fpx, buf->xg, C, cfg->mlp_dim, true);
    }
    else
    {
      floatx xscale = quantize_act(buf->xg_i8, buf->xg, cfg->mlp_dim);
      gemv_int8(
        buf->x, layer->w3->i8.q, layer->w3->i8.scales, buf->xg_i8, xscale, C,
        cfg->mlp_dim, true
      );
    }
    if (is_interrupted()) return 1;

    // Post feedforward RMSNorm
    if (cfg->pst_mlp_norm)
    {
      rmsnorm(buf->x, buf->x, layer->n4, C, cfg->eps);
    }
    if (is_interrupted()) return 1;

    // Second residual
    x     = buf->x;
    resid = buf->resid;
    for (int d = 0; d < C; d++)
    {
      buf->x[d] = clamp_fpx(x[d] + resid[d]);
    }
    if (is_interrupted()) return 1;
  }

  // Final RMSNorm
  rmsnorm(buf->x, buf->x, dec->final_norm, C, cfg->eps);
  if (is_interrupted()) return 1;

  // Compute logits (tied embedding)
  if (compute_logits)
  {
    if (!quant)
    {
      gemv_fpx(
        buf->logits, dec->embedding->fpx, buf->x, cfg->vocab_size, C, true
      );
    }
    else
    {
      floatx xscale = quantize_act(buf->x_i8, buf->x, C);
      gemv_int8(
        buf->logits, dec->embedding->i8.q, dec->embedding->i8.scales, buf->x_i8,
        xscale, cfg->vocab_size, C, true
      );
    }
    if (is_interrupted()) return 1;

    // Optional logit softcapping
    if (cfg->logit_softcap != 0.0f)
    {
      for (int d = 0; d < cfg->vocab_size; d++)
      {
        float val      = (float)buf->logits[d] / cfg->logit_softcap;
        buf->logits[d] = (floatx)(tanhf(val) * cfg->logit_softcap);
      }
    }
    if (is_interrupted()) return 1;
  }
  return 0;
}

/* Language model forward (a chunk of tokens) */
static int
forward_text_chunk(
  TextDecoder *dec,
  TextBuffer  *buf,
  int          spos,
  int          T,
  bool         mask,
  bool         quant,
  bool         compute_logits)
{
  TextConfig *cfg = dec->config;

  if (quant && (buf->x_scales == NULL || buf->xo_scales == NULL ||
                 buf->xg_scales == NULL))
  {
    fprintf(stderr, "\nerror: scale buffers are not allocated\n");
    return 1;
  }

  int C       = cfg->embed_dim;
  int NH      = cfg->n_heads;
  int NH_kv   = cfg->n_kv_heads;
  int CH      = cfg->head_dim;
  int Cq      = NH * CH;
  int Ckv     = NH_kv * CH;
  int CH_half = CH / 2;

  int epos = spos + T - 1;

  if (epos >= buf->cache_len)
  {
    fprintf(stderr, "\nerror: KV Cache is full\n");
    return 1;
  }

  // Precompute RoPE angles
  #pragma omp parallel for
  for (int t = 0; t < T; t++)
  {
    int pos = spos + t;
    int off = t * CH;

    for (int d = 0; d < CH_half; d++)
    {
      float freq;
      float e = (float)(-2 * d) / (float)CH;

      // Sliding window angles
      freq                               = powf(cfg->local_theta, e);
      buf->csfreqs_slid[off + d * 2]     = (floatx)cosf(freq * (float)pos);
      buf->csfreqs_slid[off + d * 2 + 1] = (floatx)sinf(freq * (float)pos);

      // Full attention angles
      freq                               = powf(cfg->global_theta, e);
      buf->csfreqs_full[off + d * 2]     = (floatx)cosf(freq * (float)pos);
      buf->csfreqs_full[off + d * 2 + 1] = (floatx)sinf(freq * (float)pos);
    }
  }
  if (is_interrupted()) return 1;

  floatx att_scale = (floatx)(1.0f / sqrtf((float)cfg->q_scale));

  for (int l = 0; l < cfg->n_layers; l++)
  {
    TextDecoderLayer *layer = dec->layers[l];

    memcpy(buf->resid, buf->x, T * C * sizeof(*buf->x));

    #pragma omp parallel for
    for (int t = 0; t < T; t++)
    {
      floatx *x_row = buf->x + t * C;
      rmsnorm(x_row, x_row, layer->n1, C, cfg->eps);
    }
    if (is_interrupted()) return 1;

    // The attention block

    if (quant)
    {
      quantize_acts(buf->x_i8, buf->x_scales, buf->x, 0, T, C, true);
    }

    // Compute xq & xk & xv
    #pragma omp parallel for
    for (int h = 0; h < NH; h++)
    {
      floatx *xq_head = buf->xq + h * T * CH;  // (T, CH)
      if (!quant)
      {
        gemm_fpx(
          xq_head, 0, layer->wq->fpx + h * CH * C, 0, buf->x, 0, T, CH, C, false
        );
      }
      else
      {
        gemm_int8(
          xq_head, 0, layer->wq->i8.q + h * CH * C, 0,
          layer->wq->i8.scales + h * CH, buf->x_i8, 0, buf->x_scales, T, CH, C,
          false
        );
      }
      if (h >= NH_kv) continue;

      floatx *xk_head = buf->xk + h * T * CH;
      floatx *xv_head = buf->xv + h * T * CH;

      if (!quant)
      {
        // (T, NH_kv, CH)
        gemm_fpx(
          xk_head, 0, layer->wk->fpx + h * CH * C, 0, buf->x, 0, T, CH, C, false
        );
        gemm_fpx(
          xv_head, 0, layer->wv->fpx + h * CH * C, 0, buf->x, 0, T, CH, C, false
        );
      }
      else
      {
        gemm_int8(
          xk_head, 0, layer->wk->i8.q + h * CH * C, 0,
          layer->wk->i8.scales + h * CH, buf->x_i8, 0, buf->x_scales, T, CH, C,
          false
        );
        gemm_int8(
          xv_head, 0, layer->wv->i8.q + h * CH * C, 0,
          layer->wv->i8.scales + h * CH, buf->x_i8, 0, buf->x_scales, T, CH, C,
          false
        );
      }
    }
    if (is_interrupted()) return 1;

    // Optional q & k norm
    if (cfg->qk_norm)
    {
      #pragma omp parallel for collapse(2)
      for (int h = 0; h < NH; h++)
      {
        for (int t = 0; t < T; t++)
        {
          // Q norm
          floatx *xq_head = buf->xq + h * T * CH + t * CH;
          rmsnorm(xq_head, xq_head, layer->nq, CH, cfg->eps);

          if (h < NH_kv)
          {
            // K norm
            floatx *xk_head = buf->xk + h * T * CH + t * CH;
            rmsnorm(xk_head, xk_head, layer->nk, CH, cfg->eps);
          }
        }
      }
    }
    if (is_interrupted()) return 1;

    bool    is_local = cfg->att_layers[l];
    floatx *freqs_cs = is_local ? buf->csfreqs_slid : buf->csfreqs_full;

    // RoPE
    #pragma omp parallel for collapse(2)
    for (int h = 0; h < NH + NH_kv; h++)
      for (int t = 0; t < T; t++)
      {
        floatx *data;
        if (h < NH)
        {
          data = buf->xq + h * T * CH + t * CH;  // Apply to queries
        }
        else
        {
          data = buf->xk + (h - NH) * T * CH + t * CH;  // Apply to keys
        }

        for (int d = 0; d < CH_half; d++)
        {
          float cfr = (float)freqs_cs[2 * d + t * CH];
          float sfr = (float)freqs_cs[2 * d + 1 + t * CH];
          float a   = (float)data[d];  // Index in the first half vector
          float b   = (float)data[d + CH_half];  // ... second half vector
          float r0  = a * cfr - b * sfr;
          float r1  = a * sfr + b * cfr;

          // Apply att_scale beforehand in this step, mathematically equivilant,
          // but avoided scaling the entire T * max_k attention matrix
          if (h < NH)
          {
            r0 *= (float)att_scale;
            r1 *= (float)att_scale;
          }

          data[d]           = (floatx)r0;
          data[d + CH_half] = (floatx)r1;
        }
      }

    if (is_interrupted()) return 1;

    // (NH_kv, cache_len, CH)
    floatx *k_cache = buf->kv_cache + l * 2 * buf->cache_len * Ckv;
    floatx *v_cache = k_cache + buf->cache_len * Ckv;

    // Write to kv_cache
    for (int h = 0; h < NH_kv; h++)
    {
      floatx *xk_head = k_cache + h * buf->cache_len * CH + spos * CH;
      floatx *xv_head = v_cache + h * buf->cache_len * CH + spos * CH;
      memcpy(xk_head, buf->xk + h * T * CH, T * CH * sizeof(*buf->xk));
      memcpy(xv_head, buf->xv + h * T * CH, T * CH * sizeof(*buf->xv));
    }
    if (is_interrupted()) return 1;

    int max_k = epos + 1;

    #pragma omp parallel for
    for (int h = 0; h < NH; h++)
    {
      int h_kv = h * NH_kv / NH;  // GQA mapping

      // (T, max_k)
      floatx *att_head = buf->att + h * T * max_k;  // att[h, :, :]
      // (cache_len, CH)
      floatx *xv_head = v_cache + h_kv * buf->cache_len * CH;
      floatx *xo_head = buf->xo + h * CH;  // (T, CH), column stride Cq

      /* Blockwise causal masking, the tiling algorithm used in the
       * FlashAttention paper (Dao et al., 2022,
       * https://arxiv.org/abs/2205.14135).
       *
       * The original algorithm was designed for GPUs to reduce HBM access,
       * which is not a problem for CPUs, but the idea can also be used on CPUs
       * to provide better performance for causal masking.
       *
       * Naively computing causal attention means either (a) materializing the
       * full [T, attlen] score matrix and masking out the upper triangle
       * afterwards (wastes ~half the compute), or (b) looping token by token
       * with a triangular schedule (correct FLOP count, but ragged inner loop
       * length basically kills vectorization across threads). So I tile both
       * the query and key range into blocks and classify each (qb, kb) pair
       * into five cases:
       *
       * kb > qb -> skip, every key in this block lies in the future of queries.
       * kb < qb -> full GEMM, every key lies in the past of queries.
       * kb = qb -> full GEMM & mask the upper triangle.
       * kb < spos -> skip, every key in lies before the sliding window.
       * kb = spos -> full GEMM & mask the lower triangle.
       */

      if (mask)  // Only apply tiling if mask=true
      {
        for (int q_start = 0; q_start < T; q_start += QK_BLOCK_SIZE)
        {
          // xq[h, q_start:q_end, :] (q_end - q_start, CH)
          floatx *qb          = buf->xq + h * T * CH + q_start * CH;
          int     q_end       = min(q_start + QK_BLOCK_SIZE, T);
          int     abs_q_end   = spos + q_end;
          int     abs_q_start = spos + q_start;
          if (is_local && abs_q_start >= cfg->slide_len)
          {
            abs_q_start = abs_q_start + 1 - cfg->slide_len;
          }
          else
          {
            abs_q_start = 0;
          }
          int spos_b = abs_q_start / QK_BLOCK_SIZE * QK_BLOCK_SIZE;

          for (int k_start = spos_b; k_start < abs_q_end;
            k_start += QK_BLOCK_SIZE)
          {
            // k_cache[h_kv, k_start:k_end, :] (k_end - k_start, CH)
            floatx *kb    = k_cache + h_kv * buf->cache_len * CH + k_start * CH;
            int     k_end = min(k_start + QK_BLOCK_SIZE, max_k);
            floatx *att_b = att_head + q_start * max_k + k_start;

            // Typically people don't quantize this
            gemm_fpx(
              att_b, max_k, kb, 0, qb, 0, q_end - q_start, k_end - k_start, CH,
              false
            );

            // Apply causal mask & sliding window mask
            for (int qi = q_start; qi < q_end; qi++)
            {
              int pos_i  = spos + qi;
              int spos_i = (is_local && pos_i >= cfg->slide_len)
                             ? (pos_i + 1 - cfg->slide_len)
                             : 0;
              for (int ki = k_start; ki < k_end; ki++)
              {
                if (ki > pos_i || ki < spos_i)
                {
                  att_head[qi * max_k + ki] = 0.0f;
                }
              }
            }
          }
          // Softmax & softcap for the rows
          for (int qi = q_start; qi < q_end; qi++)
          {
            floatx *att_row = att_head + qi * max_k;
            int     pos_i   = spos + qi;
            int     spos_i  = (is_local && pos_i >= cfg->slide_len)
                                ? (pos_i + 1 - cfg->slide_len)
                                : 0;

            // Optional tanh softcapping
            if (cfg->att_softcap != 0.0f)
            {
              for (int t = spos_i; t <= pos_i; t++)
              {
                float val  = (float)att_row[t] / cfg->att_softcap;
                att_row[t] = (floatx)(tanhf(val) * cfg->att_softcap);
              }
            }
            // only softmax in the range [spos_i, pos_i]
            softmax(att_row + spos_i, att_row + spos_i, pos_i - spos_i + 1);
          }

          // xo_block (qb_len, CH) = att_block (qb_len, qb_len)
          //                       @ v_block   (qb_len, CH)
          floatx *xo_block  = xo_head + q_start * Cq;
          floatx *att_block = att_head + q_start * max_k + spos_b;
          floatx *v_block   = xv_head + spos_b * CH;

          gemm_fpx_nn(
            xo_block, /*dst_stride=*/Cq, v_block, /*mat_stride=*/0, att_block,
            /*src_stride=*/max_k, q_end - q_start, CH, abs_q_end - spos_b, false
          );
        }
      }
      else  // mask=false, take the dense path
      {
        // (T, CH)
        floatx *xq_head = buf->xq + h * T * CH;  // xq[h, :, :]
        // k_cache[h_kv, :, :] (cache_len, CH)
        floatx *xk_head = k_cache + h_kv * buf->cache_len * CH;
        // att_head = xq_head @ xk_head.T
        gemm_fpx(att_head, max_k, xk_head, 0, xq_head, 0, T, max_k, CH, false);

        for (int qi = 0; qi < T; qi++)
        {
          floatx *att_row = att_head + qi * max_k;

          // Optional tanh softcapping
          for (int t = 0; t < max_k; t++)
          {
            if (cfg->att_softcap != 0.0f)
            {
              float val  = (float)att_row[t] / cfg->att_softcap;
              att_row[t] = (floatx)(tanhf(val) * cfg->att_softcap);
            }
          }
          // Softmax
          softmax(att_row, att_row, max_k);
        }
        // xo_head (T, CH) = att_head (T, max_k) @ xv_head[:max_k] (max_k, CH)
        gemm_fpx_nn(
          xo_head, /*dst_stride=*/Cq, xv_head, /*mat_stride=*/0, att_head,
          /*src_stride=*/max_k, T, CH, max_k, false
        );
      }
    }
    if (is_interrupted()) return 1;
    // x (T, C) = xo_head (T, CH) @ wo.T (CH, C)
    if (!quant)
    {
      gemm_fpx(buf->x, 0, layer->wo->fpx, 0, buf->xo, Cq, T, C, Cq, true);
    }
    else
    {
      quantize_acts(buf->xo_i8, buf->xo_scales, buf->xo, Cq, T, Cq, true);
      gemm_int8(
        buf->x, 0, layer->wo->i8.q, 0, layer->wo->i8.scales, buf->xo_i8, 0,
        buf->xo_scales, T, C, Cq, true
      );
    }
    if (is_interrupted()) return 1;

    #pragma omp parallel for
    for (int t = 0; t < T; t++)
    {
      floatx *x_row = buf->x + t * C;
      rmsnorm(x_row, x_row, layer->n2, C, cfg->eps);
    }
    if (is_interrupted()) return 1;

    floatx *restrict x     = buf->x;
    floatx *restrict resid = buf->resid;
    #pragma omp parallel for
    for (int d = 0; d < T * C; d++)
    {
      // Combine the residual stream
      buf->x[d] = clamp_fpx(x[d] + resid[d]);
    }
    if (is_interrupted()) return 1;

    memcpy(buf->resid, buf->x, T * C * sizeof(*buf->x));

    if (cfg->pre_mlp_norm)
    {
      #pragma omp parallel for
      for (int t = 0; t < T; t++)
      {
        floatx *x_row = buf->x + t * C;
        rmsnorm(x_row, x_row, layer->n3, C, cfg->eps);
      }
    }
    if (is_interrupted()) return 1;

    // MLP
    if (!quant)
    {
      gemm_fpx(
        buf->xu, 0, layer->w1->fpx, 0, buf->x, 0, T, cfg->mlp_dim, C, true
      );
      gemm_fpx(
        buf->xg, 0, layer->w2->fpx, 0, buf->x, 0, T, cfg->mlp_dim, C, true
      );
    }
    else
    {
      quantize_acts(buf->x_i8, buf->x_scales, buf->x, 0, T, C, true);
      gemm_int8(
        buf->xu, 0, layer->w1->i8.q, 0, layer->w1->i8.scales, buf->x_i8, 0,
        buf->x_scales, T, cfg->mlp_dim, C, true
      );
      gemm_int8(
        buf->xg, 0, layer->w2->i8.q, 0, layer->w2->i8.scales, buf->x_i8, 0,
        buf->x_scales, T, cfg->mlp_dim, C, true
      );
    }
    if (is_interrupted()) return 1;

    // GELU gate
    #pragma omp parallel for
    for (int d = 0; d < T * cfg->mlp_dim; d++)
    {
      // Tanh approximation of GELU
      float x    = (float)buf->xg[d];
      float c    = 0.79788456080287f;  // = sqrt(2 / pi)
      x          = 0.5 * x * (1 + tanhf(c * (x + 0.044715 * x * x * x)));
      buf->xg[d] = (floatx)x;
      buf->xg[d] *= buf->xu[d];  // Fuse xg * xu into xg
    }
    if (is_interrupted()) return 1;

    // Down projection
    if (!quant)
    {
      gemm_fpx(
        buf->x, 0, layer->w3->fpx, 0, buf->xg, 0, T, C, cfg->mlp_dim, true
      );
    }
    else
    {
      quantize_acts(
        buf->xg_i8, buf->xg_scales, buf->xg, 0, T, cfg->mlp_dim, true
      );
      gemm_int8(
        buf->x, 0, layer->w3->i8.q, 0, layer->w3->i8.scales, buf->xg_i8, 0,
        buf->xg_scales, T, C, cfg->mlp_dim, true
      );
    }
    if (is_interrupted()) return 1;

    if (cfg->pst_mlp_norm)
    {
      #pragma omp parallel for
      for (int t = 0; t < T; t++)
      {
        floatx *x_row = buf->x + t * C;
        rmsnorm(x_row, x_row, layer->n4, C, cfg->eps);
      }
    }
    if (is_interrupted()) return 1;

    x     = buf->x;
    resid = buf->resid;
    #pragma omp parallel for
    for (int d = 0; d < T * C; d++)
    {
      // Second residual
      buf->x[d] = clamp_fpx(x[d] + resid[d]);
    }
    if (is_interrupted()) return 1;
  }

  // Final RMSNorm
  #pragma omp parallel for
  for (int t = 0; t < T; t++)
  {
    floatx *x_row = buf->x + t * C;
    rmsnorm(x_row, x_row, dec->final_norm, C, cfg->eps);
  }

  // Compute logits (tied embedding)
  if (compute_logits)
  {
    floatx *last_x = buf->x + (T - 1) * C;
    if (!quant)
    {
      gemv_fpx(
        buf->logits, dec->embedding->fpx, last_x, cfg->vocab_size, C, true
      );
    }
    else
    {
      floatx xscale = quantize_act(buf->x_i8, last_x, C);
      gemv_int8(
        buf->logits, dec->embedding->i8.q, dec->embedding->i8.scales, buf->x_i8,
        xscale, cfg->vocab_size, C, true
      );
    }
    if (is_interrupted()) return 1;

    // Optional logit softcapping
    if (cfg->logit_softcap != 0.0f)
    {
      for (int d = 0; d < cfg->vocab_size; d++)
      {
        float val      = (float)buf->logits[d] / cfg->logit_softcap;
        buf->logits[d] = (floatx)(tanhf(val) * cfg->logit_softcap);
      }
    }
    if (is_interrupted()) return 1;
  }
  return 0;
}

/* Language + vision model forward */
int
forward_gemma_decode(GemmaModel *model, TextBuffer *buf, int token, int pos)
{
  TextDecoder *dec = model->decoder;

  /* Embedding lookup.
   * The export script already applied 1/sqrt(embed_dim), so the usual
   * Gemma sqrt(embed_dim) scale cancels out to 1.0 here. */
  floatx embed_scale =
    1.0f;  // equivalent to sqrt(embed_dim) * (1/sqrt(embed_dim))
  if (model->quant)
  {
    // Dequantize
    embed_scale *= dec->embedding->i8.scales[token];
  }

  int C = dec->config->embed_dim;

  // x = embedding[tok] * embed_scale
  #pragma omp parallel for
  for (int d = 0; d < C; d++)
  {
    if (!model->quant)
    {
      buf->x[d] = dec->embedding->fpx[token * C + d] * embed_scale;
    }
    else
    {
      buf->x[d] = (floatx)dec->embedding->i8.q[token * C + d] * embed_scale;
    }
  }
  if (is_interrupted()) return 1;
  return forward_text_decode(model->decoder, buf, pos, model->quant, true);
}

/* Language model prefill */
int
forward_gemma_prefill(
  GemmaModel *model,
  TextBuffer *buf,
  int        *tokens,
  int         T,
  int        *pos,
  int         chunk_size,
  bool       *rpen_visited,
  bool        compute_logits)
{
  TextDecoder *dec = model->decoder;
  int          C   = dec->config->embed_dim;

  // Prefill by chunks
  for (int off = 0; off < T; off += chunk_size)
  {
    int  cur_len       = min(chunk_size, T - off);
    bool is_last_chunk = (off + cur_len == T);

    // Embedding lookup
    for (int t = 0; t < cur_len; t++)
    {
      int token   = tokens[off + t];
      int emb_off = t * C;

      if (rpen_visited != NULL)
      {
        rpen_visited[token] = true;
      }

      for (int d = 0; d < C; d++)
      {
        if (!model->quant)
        {
          buf->x[emb_off + d] = dec->embedding->fpx[token * C + d];  // * 1.0f
        }
        else
        {
          buf->x[emb_off + d] = (floatx)dec->embedding->i8.q[token * C + d] *
                                dec->embedding->i8.scales[token];
        }
      }
    }

    int rc = forward_text_chunk(
      dec, buf, *pos + off, cur_len, true, model->quant,
      is_last_chunk && compute_logits
    );

    if (rc != 0) return rc;
  }

  *pos += T;
  return 0;
}

/* Vision model forward + embedding prefill */
int
forward_gemma_image(
  GemmaModel   *model,
  TextBuffer   *buf,
  VisionBuffer *vbuf,
  const floatx *image,
  int          *pos,
  bool          compute_logits)
{
  VisionEncoder *enc = model->encoder;
  if (enc == NULL)
  {
    return 1;
  }

  TextDecoder *dec = model->decoder;
  if (forward_vision(enc, dec->config, buf, vbuf, image, model->quant) == 1)
  {
    return 1;
  }

  int image_toks = dec->config->image_toks;

  int suc = forward_text_chunk(
    // Gemma 3 models uses bi-directional attention for vision tokens
    dec, buf, *pos, image_toks, false, model->quant, compute_logits
  );

  *pos += image_toks;
  return suc;
}

// Sampling

/* */
static int
argmax(floatx *logits, int vocab_size)
{
  // Pick the index with the max value
  int    max_idx = -1;
  floatx max_val = -FLOATX_MAX;
  #pragma omp parallel
  {
    int    local_idx = -1;
    floatx local_val = -FLOATX_MAX;
    #pragma omp for nowait
    for (int i = 0; i < vocab_size; i++)
    {
      if (logits[i] > local_val)
      {
        local_val = logits[i];
        local_idx = i;
      }
    }
    #pragma omp critical
    {
      // Only one thread is able to run this at a time
      if (local_val > max_val)
      {
        max_val = local_val;
        max_idx = local_idx;
      }
    }
  }
  return max_idx;
}

/* */
typedef struct
{
  floatx val;
  int    idx;
} FloatIdx;

/* */
static inline void
swap_fi(FloatIdx *a, FloatIdx *b)
{
  FloatIdx t = *a;

  *a = *b;
  *b = t;
}

/* Tiny xorshift for pivot selection */
static uint32_t qs_rand_state = 3418323524;
static inline uint32_t
qs_rand(void)
{
  qs_rand_state ^= qs_rand_state << 13;
  qs_rand_state ^= qs_rand_state >> 7;
  qs_rand_state ^= qs_rand_state << 17;
  return (uint32_t)qs_rand_state;
}

/* */
static int
partition_desc(FloatIdx *arr, int lo, int hi)
{
  // Pick the pivot randomly (use a seperate rand sequence)
  uint32_t range = (uint32_t)(hi - lo + 1);
  int      r     = lo + (int)(qs_rand() % range);
  swap_fi(&arr[r], &arr[hi]);

  floatx pivot = arr[hi].val;

  int i = lo;
  for (int j = lo; j < hi; j++)
  {
    if (arr[j].val > pivot)
    {
      // Put the greater one on the left
      swap_fi(&arr[i++], &arr[j]);
    }
  }
  swap_fi(&arr[i], &arr[hi]);
  return i;
}

/* Quickselect to find the top-k elements (descending) */
static void
quickselect_topk(FloatIdx *arr, int lo, int hi, int k_idx)
{
  while (lo < hi)
  {
    int p = partition_desc(arr, lo, hi);
    if (p == k_idx) return;
    else if (p < k_idx)
    {
      lo = p + 1;
    }
    else
    {
      hi = p - 1;
    }
  }
}

/* */
static void
apply_topk(floatx *logits, FloatIdx *logit_indices, int vocab_size, int k)
{
  if (k <= 0)
  {
    k = 1;
  }
  if (k > vocab_size)
  {
    k = vocab_size;
  }

  // Record index info
  #pragma omp parallel for
  for (int i = 0; i < vocab_size; i++)
  {
    logit_indices[i].idx = i;
    logit_indices[i].val = logits[i];
  }
  quickselect_topk(logit_indices, 0, vocab_size - 1, k - 1);

  // Keep the top k channels
  #pragma omp parallel for
  for (int i = 0; i < vocab_size; i++)
  {
    logits[i] = -FLOATX_MAX;
  }
  for (int i = 0; i < k; i++)
  {
    logits[logit_indices[i].idx] = logit_indices[i].val;
  }
}

/* Max-heap helpers for top-p */
static void
sift_down(FloatIdx *arr, int n, int i)
{
  // Make sure the parent node arr[i] is greater than its children in
  // the heap
  for (;;)
  {
    // l & r are the two children node
    int l       = 2 * i + 1;
    int r       = 2 * i + 2;
    int largest = i;
    if (l < n && arr[l].val > arr[largest].val)
    {
      largest = l;
    }
    if (r < n && arr[r].val > arr[largest].val)
    {
      largest = r;
    }
    if (largest == i) break;
    swap_fi(&arr[i], &arr[largest]);
    i = largest;
  }
}

/* */
static void
build_heap(FloatIdx *arr, int n)
{
  for (int i = n / 2 - 1; i >= 0; i--)
  {
    sift_down(arr, n, i);
  }
}

/* */
static void
apply_topp(
  floatx   *logits,
  floatx   *fpbuf,
  FloatIdx *logit_indices,
  int       vocab_size,
  int       k,
  float     p)
{
  if (k > vocab_size)
  {
    k = vocab_size;
  }
  // Softmax to get the probs, store in fpbuf
  softmax(fpbuf, logits, vocab_size);
  int heap_size = (k == 0) ? vocab_size : k;

  if (k == 0)
  {
    #pragma omp parallel for
    for (int i = 0; i < vocab_size; i++)
    {
      logit_indices[i].idx = i;
      logit_indices[i].val = fpbuf[i];
    }
  }
  else
  {
    for (int i = 0; i < k; i++)
    {
      int idx = logit_indices[i].idx;  // Reuse the candidates from topk
      logit_indices[i].val = fpbuf[idx];
    }
  }
  build_heap(logit_indices, heap_size);  // O(k)

  // fpbuf is now a copy of the original logits
  memcpy(fpbuf, logits, vocab_size * sizeof(floatx));

  // Set logits to -inf
  #pragma omp parallel for
  for (int i = 0; i < vocab_size; i++)
  {
    logits[i] = -FLOATX_MAX;
  }

  float cum = 0.0f;  // Cumulative prob

  while (heap_size > 0)
  {
    // Pop the current max prob
    FloatIdx top     = logit_indices[0];
    logit_indices[0] = logit_indices[--heap_size];  // Put the last element to
                                                    // the top
    sift_down(logit_indices, heap_size, 0);         // O(log(vocab_size))

    logits[top.idx] = fpbuf[top.idx];
    cum += (float)top.val;
    if (cum >= p) break;
  }
}

/* Simple repetition penalty */
void
apply_rpen(floatx *logits, bool *visited, int vocab_size, float rpen)
{
  // rpen short for Repetition Penalty
  #pragma omp parallel for
  for (int i = 0; i < vocab_size; i++)
  {
    if (!visited[i]) continue;
    float val = (float)logits[i];
    if (val > 0.0f)
    {
      logits[i] = (floatx)(val / rpen);
    }
    else
    {
      logits[i] = (floatx)(val * rpen);
    }
  }
}

/* */
typedef enum
{
  INJECT_NONE,
  INJECT_TEXT,
  INJECT_IMAG,
  INJECT_QUIT,
  INJECT_DONE,
} InjectDataType;

/* */
typedef struct
{
  InjectDataType type;
  union
  {
    struct
    {
      int *tokens;
      int  n_tokens;
    };
    floatx *image;
  };
} InjectData;

/* Sample the next token from the logits */
int
sample_from_logits(
  floatx   *logits,
  floatx   *probs,
  FloatIdx *logit_indices,
  bool     *visited,
  int       vocab_size,
  float     temperature,
  int       topk,
  float     topp,
  float     rpen)
{
  bool dosample = temperature != 0 && topk != 1;

  // Manipulate the logits & sample the next token
  if (!dosample)
  {
    // Argmax sampling
    return argmax(logits, vocab_size);
  }

  bool use_topk = dosample && topk != 0;
  bool use_topp = dosample && topp < 1.0f;
  bool use_rpen = dosample && rpen > 1.0f;

  // Apply the temperature
  #pragma omp parallel for
  for (int d = 0; d < vocab_size; d++)
  {
    logits[d] /= (floatx)temperature;
  }

  if (use_topk)
  {
    apply_topk(logits, logit_indices, vocab_size, topk);
  }
  if (use_topp)
  {
    apply_topp(logits, probs, logit_indices, vocab_size, topk, topp);
  }
  if (use_rpen)
  {
    apply_rpen(logits, visited, vocab_size, rpen);
  }

  // Softmax to get the probs
  softmax(probs, logits, vocab_size);

  // Multinomial sample from probs
  float r   = (float)((float)rand() / (RAND_MAX + 1.0));
  float sum = 0.0f;

  int token = vocab_size - 1;
  for (int d = 0; d < vocab_size; d++)
  {
    sum += (float)probs[d];
    if (r < sum)
    {
      token = d;
      break;
    }
  }
  return token;
}

/* The main sampling loop */
void
sample(
  GemmaModel   *model,
  TextBuffer   *buf,
  VisionBuffer *vbuf,
  int           seqlen,
  int           chunk_size,
  float         temperature,
  int           topk,
  float         topp,
  float         rpen,
  bool          use_mm,
  void         *inject_ctx,
  InjectData (*inject_callback)(
    int token, GemmaModel *model, bool use_mm, void *ctx
  ))
{
  TextConfig *cfg = model->decoder->config;
  int         vs  = cfg->vocab_size;

  bool    dosample        = temperature != 0 && topk != 1;
  bool    use_topk        = dosample && topk != 0;
  bool    use_topp        = dosample && topp < 1.0f;
  bool    use_rpen        = dosample && rpen > 1.0f;
  double  prefill_start   = 0.0;
  double  prefill_end     = 0.0;
  double  prefill_elapsed = 0.0;
  double  gen_start       = 0.0;
  double  gen_end         = 0.0;
  double  gen_elapsed     = 0.0;
  int     prompt_toks     = 0;
  int     gen_toks        = 0;
  int     pos             = 0;
  int     token           = 0;
  floatx *probs           = NULL;

  FloatIdx *logit_indices = NULL;
  bool     *visited       = NULL;

  if (use_rpen)
  {
    CALLOC(visited, vs, "visited", goto end;);
  }
  if (dosample)
  {
    MALLOC(probs, vs, "probs", goto end;);
  }
  if (use_topk || use_topp)
  {
    MALLOC(logit_indices, vs, "logit_indices", goto end;);
  }

  // `<bos>` is always the very first token of the sequence
  {
    int bos_tok = model->tokenizer->bos;
    if (forward_gemma_prefill(
          model, buf, &bos_tok, 1, &pos, chunk_size, visited, false
        ) == 1)
    {
      goto end;
    }
    prompt_toks = 1;
  }

  InjectData injected = inject_callback(EOF, model, use_mm, inject_ctx);

  while (pos < seqlen && injected.type != INJECT_QUIT)
  {
    if (injected.type == INJECT_NONE)
    {
      // pure auto-regressive step
      gen_start = now_sec();
      if (forward_gemma_decode(model, buf, token, pos) == 1) goto end;
      pos++;
      gen_end = now_sec();
      gen_elapsed += gen_end - gen_start;

      if (is_interrupted()) break;
      // Fall through to common sample
    }
    else if (injected.type == INJECT_DONE)
    {
      break;
    }
    else
    {
      // Data injection (text / image)
      InjectData next    = inject_callback(EOF, model, use_mm, inject_ctx);
      bool       is_last = (next.type == INJECT_DONE);

      int prev_pos  = pos;
      prefill_start = now_sec();

      if (injected.type == INJECT_TEXT)
      {
        if (forward_gemma_prefill(
              model, buf, injected.tokens, injected.n_tokens, &pos, chunk_size,
              visited, is_last
            ) == 1)
        {
          goto end;
        }
      }
      else if (injected.type == INJECT_IMAG)
      {
        if (forward_gemma_image(
              model, buf, vbuf, injected.image, &pos, is_last
            ) == 1)
        {
          goto end;
        }
      }

      prefill_end = now_sec();
      prefill_elapsed += prefill_end - prefill_start;
      prompt_toks += pos - prev_pos;

      if (is_interrupted()) break;

      if (!is_last)
      {
        injected = next;
        continue;
      }
      // is_last: fall through to common sample
      gen_start = now_sec();
    }

    // Common sample path (both Case A and last-injection)
    token = sample_from_logits(
      buf->logits, probs, logit_indices, visited, vs, temperature, topk, topp,
      rpen
    );
    if (use_rpen) visited[token] = true;
    gen_toks++;

    // only the is_last path needs to close the gen timer here
    // the pure-generation path already closed it before the fall-through
    if (injected.type != INJECT_NONE)
    {
      gen_end = now_sec();
      gen_elapsed += gen_end - gen_start;
    }

    injected = inject_callback(token, model, use_mm, inject_ctx);
  }

end:
  if (prefill_elapsed > 0.0)
  {
    printf(
      "\n\nprompt processed %d tokens in %.2f seconds (%.2f tok/s)\n",
      prompt_toks, prefill_elapsed, prompt_toks / prefill_elapsed
    );
  }
  else
  {
    printf("\n\nprompt processed %d tokens instantly\n", prompt_toks);
  }
  if (gen_elapsed > 0.0)
  {
    printf(
      "generated %d tokens in %.2f seconds (%.2f tok/s)\n", gen_toks,
      gen_elapsed, gen_toks / gen_elapsed
    );
  }
  else
  {
    printf("generated %d tokens instantly\n", gen_toks);
  }

  free(probs);
  if (use_rpen) free(visited);
  if (use_topk || use_topp) free(logit_indices);
}

/* */
typedef struct
{
  int         state;
  floatx     *img;
  const char *arg;
  int         arg_len;

  // Used in pan & scan
  char    path[4096];
  floatx *crops;
  int     n_crops;
  int     crop_i;
} CommandContext;

/* */
typedef struct InjectContext InjectContext;

/* */
typedef struct
{
  const char *name;
  enum
  {
    COMMAND_IMAGE = 0,
    COMMAND_IMAGE_PAS,
    COMMAND_TOTAL,
  } id;
  bool (*should_ignore)(GemmaModel *model, bool use_mm);
  InjectData (*inject_next)(GemmaModel *model, InjectContext *ctx, bool use_mm);
} CommandType;

/* */
typedef enum
{
  SCAN_TEXT,
  SCAN_COMMAND,
  SCAN_DONE,
} ScanType;

/* */
typedef struct
{
  CommandType type;
  char       *raw;
  int         raw_len;
  char       *arg;
  int         arg_len;
} CommandRecord;

/* */
typedef struct
{
  ScanType type;
  union
  {
    struct
    {
      const char *text;
      int         text_len;
    };
    CommandRecord cmd;
  };
  char *remaining;
} ScanResult;

/* */
struct InjectContext
{
  int            state;
  const char    *text;
  int            text_len;
  ScanResult     event;
  CommandContext cmd_ctx;
  int           *tokens_buf;
  int            tokens_len;
  int            tokens_cap;
};

/* */
typedef struct
{
  int            state;
  char          *line_buf;
  int            line_cap;
  InjectContext *jctx;
} ChatContext;

/* */
bool
image_should_ignore(GemmaModel *model, bool use_mm)
{
  (void)model;
  return !use_mm;
}

/* */
InjectData
image_inject_next(GemmaModel *model, InjectContext *ctx, bool use_mm)
{
  /* Template:
   * \n\n<start_of_image>[image soft tokens]<end_of_image>\n\n */

  VisionEncoder  *enc     = model->encoder;
  GemmaTokenizer *tok     = model->tokenizer;
  CommandContext *cmd_ctx = &ctx->cmd_ctx;

  int spos, epos;

  /* A classic Duff-style hack that fakes a generator/coroutine using an
   * explicit state machine, each call resumes at the `case` matching the
   * current state, does one step, and returns an `InjectData` value as a sort
   * of pseudo-`yield`. The caller loops until it receives INJECT_DONE (or
   * INJECT_QUIT on error). The `state` field is effectively a saved program
   * counter that lets the sequence survive across multiple invocations. */

  switch (cmd_ctx->state)
  {
    case 0:
      // Inject header ("\n\n<start_of_image>")
      spos = ctx->tokens_len;
      encode(tok, "\n\n", 2, ctx->tokens_buf, spos, &ctx->tokens_len);
      ctx->tokens_buf[ctx->tokens_len++] = tok->soi;
      epos = ctx->tokens_len;

      cmd_ctx->state = 1;
      return (InjectData){
        .type     = INJECT_TEXT,
        .tokens   = ctx->tokens_buf + spos,
        .n_tokens = epos - spos,
      };

    case 1:
      // Inject image
      ;  // "label followed by a declaration is a C23 extension"
      int  img_sz     = (enc && use_mm) ? enc->config->image_size : 0;
      char path[4096] = {0};
      if ((size_t)cmd_ctx->arg_len >= sizeof(path))
      {
        fprintf(stderr, "\nerror: image path too long\n");
        cmd_ctx->state = GENERATOR_EXIT;
        return (InjectData){.type = INJECT_QUIT};
      }
      memcpy(path, cmd_ctx->arg, cmd_ctx->arg_len);
      cmd_ctx->img = prepare_image(path, img_sz);
      if (cmd_ctx->img == NULL)
      {
        fprintf(stderr, "\nerror: failed to prepare image: '%s'\n", path);
        cmd_ctx->state = GENERATOR_EXIT;
        return (InjectData){.type = INJECT_QUIT};
      }

      cmd_ctx->state = 2;
      return (InjectData){.type = INJECT_IMAG, .image = cmd_ctx->img};

    case 2:
      // Inject trailer ("<end_of_image>\n\n")
      spos = ctx->tokens_len;
      ctx->tokens_buf[ctx->tokens_len++] = tok->eoi;
      encode(
        tok, "\n\n", 2, ctx->tokens_buf, ctx->tokens_len, &ctx->tokens_len
      );
      epos = ctx->tokens_len;

      cmd_ctx->state = 3;
      return (InjectData){
        .type     = INJECT_TEXT,
        .tokens   = ctx->tokens_buf + spos,
        .n_tokens = epos - spos,
      };

    default:
      free(cmd_ctx->img);
      cmd_ctx->state = GENERATOR_EXIT;
      return (InjectData){.type = INJECT_DONE};
  }
}

/* */
bool
image_pas_should_ignore(GemmaModel *model, bool use_mm)
{
  (void)model;
  return !use_mm;
}

/* */
InjectData
image_pas_inject_next(GemmaModel *model, InjectContext *ctx, bool use_mm)
{
  // A funny workaround for long images, crop the full image and send the crops
  // to the model

  /* Template:
   * here is the original image
   * \n\n<start_of_image>
   * [full image soft tokens]
   * <end_of_image>\n\n
   * and here are some crops to help you see better
   * \n\n<start_of_image>
   * [crop 1 soft tokens]
   * <end_of_image>\n\n
   * \n\n<start_of_image>
   * [crop 2 soft tokens]
   * <end_of_image>\n\n
   * ...
   */

  VisionEncoder  *enc     = model->encoder;
  GemmaTokenizer *tok     = model->tokenizer;
  CommandContext *cmd_ctx = &ctx->cmd_ctx;

  int spos, epos;
  int img_sz = (enc && use_mm) ? enc->config->image_size : 0;

  switch (cmd_ctx->state)
  {
    case 0:
      // Inject prefix prompt
      spos = ctx->tokens_len;
      encode(
        tok, CROPPED_IMAGE_PREFIX, strlen(CROPPED_IMAGE_PREFIX),
        ctx->tokens_buf, spos, &ctx->tokens_len
      );
      // Inject image header ("\n\n<start_of_image>")
      encode(
        tok, "\n\n", 2, ctx->tokens_buf, ctx->tokens_len, &ctx->tokens_len
      );
      ctx->tokens_buf[ctx->tokens_len++] = tok->soi;
      epos = ctx->tokens_len;

      cmd_ctx->state = 1;
      return (InjectData){
        .type     = INJECT_TEXT,
        .tokens   = ctx->tokens_buf + spos,
        .n_tokens = epos - spos,
      };
    
    case 1:
      if ((size_t)cmd_ctx->arg_len >= sizeof(cmd_ctx->path))
      {
        fprintf(stderr, "\nerror: image path too long\n");
        cmd_ctx->state = GENERATOR_EXIT;
        return (InjectData){.type = INJECT_QUIT};
      }
      memcpy(cmd_ctx->path, cmd_ctx->arg, cmd_ctx->arg_len);
      cmd_ctx->path[cmd_ctx->arg_len] = '\0';

      // Save the image to cmd_ctx->crops
      cmd_ctx->crops = prepare_image_pas(
        cmd_ctx->path, img_sz, MIN_CROP_SIZE, MAX_NUM_CROPS, &cmd_ctx->n_crops
      );
      if (cmd_ctx->crops == NULL)
      {
        fprintf(stderr, "\nerror: failed to prepare image: '%s'\n", cmd_ctx->path);
        cmd_ctx->state = GENERATOR_EXIT;
        return (InjectData){.type = INJECT_QUIT};
      }

      cmd_ctx->state = 2;
      /* Inject the full image.
       * Passing in cmd_ctx->crops is safe here since `sample()` will only take
       * the first `image_size * image_size * 3` elements. */
      return (InjectData){.type = INJECT_IMAG, .image = cmd_ctx->crops};
    
    case 2:
      // Inject crop prompt ("and here are some crops to help you see better")
      spos = ctx->tokens_len;
      encode(
        tok, CROPPED_IMAGE_FILTER, strlen(CROPPED_IMAGE_FILTER),
        ctx->tokens_buf, spos, &ctx->tokens_len
      );
      // Inject image header of the first crop
      encode(
        tok, "\n\n", 2, ctx->tokens_buf, ctx->tokens_len, &ctx->tokens_len
      );
      ctx->tokens_buf[ctx->tokens_len++] = tok->soi;
      epos = ctx->tokens_len;

      cmd_ctx->state = 3;
      return (InjectData){
        .type     = INJECT_TEXT,
        .tokens   = ctx->tokens_buf + spos,
        .n_tokens = epos - spos,
      };
    
    case 3:
      // Inject all the crops
      for (
        cmd_ctx->crop_i = 0;
        cmd_ctx->crop_i < cmd_ctx->n_crops;
        cmd_ctx->crop_i++)
      {
        if (cmd_ctx->crop_i != 0)
        {
          // Inject crop header
          spos = ctx->tokens_len;
          encode(tok, "\n\n", 2, ctx->tokens_buf, spos, &ctx->tokens_len);
          ctx->tokens_buf[ctx->tokens_len++] = tok->soi;
          epos = ctx->tokens_len;

          cmd_ctx->state = 4;
          return (InjectData){
            .type     = INJECT_TEXT,
            .tokens   = ctx->tokens_buf + spos,
            .n_tokens = epos - spos,
          };
        }
    
    case 4:
        // Inject image crop
        cmd_ctx->state = 5;
        return (InjectData){
          .type  = INJECT_IMAG,
          .image = cmd_ctx->crops +
                   (size_t)cmd_ctx->crop_i * img_sz * img_sz * 3,
        };
    
    case 5:
        // Inject crop trailer
        spos = ctx->tokens_len;
        ctx->tokens_buf[ctx->tokens_len++] = tok->eoi;
        encode(
          tok, "\n\n", 2, ctx->tokens_buf, ctx->tokens_len, &ctx->tokens_len
        );
        epos = ctx->tokens_len;

        cmd_ctx->state = 6;
        return (InjectData){
          .type     = INJECT_TEXT,
          .tokens   = ctx->tokens_buf + spos,
          .n_tokens = epos - spos,
        };
    
    case 6:
        ;
      }

    __attribute__((fallthrough));
    default:
      free(cmd_ctx->crops);
      cmd_ctx->state = GENERATOR_EXIT;
      return (InjectData){.type = INJECT_DONE};
  }
}

/* */
CommandType command_types[COMMAND_TOTAL] = {
  (CommandType){
    .name          = "image",
    .id            = COMMAND_IMAGE,
    .should_ignore = image_should_ignore,
    .inject_next   = image_inject_next,
  },
  (CommandType){
    .name          = "image_pas",
    .id            = COMMAND_IMAGE_PAS,
    .should_ignore = image_pas_should_ignore,
    .inject_next   = image_pas_inject_next,
  },
};

/* */
ScanResult
scan_next_event(const char *text)
{
  if (text[0] == '\0')
  {
    return (ScanResult){.type = SCAN_DONE};
  }

  CommandRecord first_cmd = {.raw = NULL};
  char         *closing;

  for (int i = 0; i < COMMAND_TOTAL; i++)
  {
    CommandType cmd_type = command_types[i];

    char pattern[64] = {0};
    sprintf(pattern, "@%s{", cmd_type.name);
    char *cmd = strstr(text, pattern);

    if (cmd != NULL && (first_cmd.raw == NULL || cmd < first_cmd.raw))
    {
      // Look for closing '}'
      closing = strchr(cmd, (int)'}');
      if (closing != NULL)
      {
        int raw_len = closing + 1 - cmd;
        int arg_off = strlen(cmd_type.name) + 2;
        int arg_len = raw_len - arg_off - 1;

        first_cmd = (CommandRecord){
          .type    = cmd_type,
          .raw     = cmd,
          .raw_len = raw_len,
          .arg     = cmd + arg_off,
          .arg_len = arg_len,
        };
      }
    }
  }

  if (first_cmd.raw == NULL)
  {
    size_t len = strlen(text);
    // No command found, return the whole text
    return (ScanResult){
      .type      = SCAN_TEXT,
      .text      = text,
      .text_len  = (int)len,
      .remaining = (char *)(text + len),
    };
  }
  if (first_cmd.raw > text)
  {
    // Text before command
    return (ScanResult){
      .type      = SCAN_TEXT,
      .text      = text,
      .text_len  = first_cmd.raw - text,
      .remaining = first_cmd.raw,
    };
  }
  // Starts with command
  return (ScanResult){
    .type      = SCAN_COMMAND,
    .cmd       = first_cmd,
    .remaining = closing + 1,
  };
}

/* */
InjectData
inject_next(GemmaModel *model, InjectContext *ctx, bool use_mm)
{
  GemmaTokenizer *tok = model->tokenizer;

  int spos, epos;

  switch (ctx->state)
  {
    while (ctx->text_len > 0)
    {
    case 0:
      ctx->event = scan_next_event(ctx->text);

      if (ctx->event.type == SCAN_DONE) break;
      if (ctx->event.type == SCAN_TEXT)
      {
        spos = ctx->tokens_len;
        encode(
          tok, ctx->event.text, ctx->event.text_len, ctx->tokens_buf, spos,
          &ctx->tokens_len
        );
        epos = ctx->tokens_len;

        ctx->state = 1;
        return (InjectData){
          .type     = INJECT_TEXT,
          .tokens   = ctx->tokens_buf + spos,
          .n_tokens = epos - spos,
        };

    case 1:
        ctx->text_len -= ctx->event.remaining - ctx->text;
        ctx->text  = ctx->event.remaining;
        ctx->state = 0;
        continue;
      }

      char *raw     = ctx->event.cmd.raw;
      int   raw_len = ctx->event.cmd.raw_len;

      if (ctx->event.cmd.type.should_ignore(model, use_mm))
      {
        spos = ctx->tokens_len;
        encode(tok, raw, raw_len, ctx->tokens_buf, spos, &ctx->tokens_len);
        epos = ctx->tokens_len;

        ctx->state = 2;
        return (InjectData){
          .type     = INJECT_TEXT,
          .tokens   = ctx->tokens_buf + spos,
          .n_tokens = epos - spos,
        };

    case 2:
        ctx->text_len -= ctx->event.remaining - ctx->text;
        ctx->text = ctx->event.remaining;
        continue;
      }

      ctx->cmd_ctx = (CommandContext){
        .state   = 0,
        .arg     = ctx->event.cmd.arg,
        .arg_len = ctx->event.cmd.arg_len,
      };

      InjectData r;
      for (;;)
      {
        r = ctx->event.cmd.type.inject_next(model, ctx, use_mm);
        if (r.type != INJECT_DONE)
        {
          ctx->state = 3;
          return r;
    case 3:
          ;
        }
        else
        {
          ctx->state = 4;  // done
          break;
        }
      }

      ctx->text_len -= ctx->event.remaining - ctx->text;
      ctx->text = ctx->event.remaining;
    }

    __attribute__((fallthrough));
    default:
      ctx->state = GENERATOR_EXIT;
      return (InjectData){.type = INJECT_DONE};
  }
}

/* */
InjectData
generate_inject_callback(int token, GemmaModel *model, bool use_mm, void *ctx)
{
  GemmaTokenizer *tok = model->tokenizer;
  if (token == EOF)
  {
    // Request for prefilling
    return inject_next(model, ctx, use_mm);
  }
  if (token == tok->eos || token == tok->eot)
  {
    return (InjectData){.type = INJECT_QUIT};
  }

  // Prints the token to stdout
  char byte_buf[2];
  printf("%s", decode(tok, token, byte_buf));
  fflush(stdout);

  return (InjectData){.type = INJECT_NONE};
}

/* */
void
generate(
  GemmaModel   *model,
  TextBuffer   *buf,
  VisionBuffer *vbuf,
  const char   *prompt,
  int           seqlen,
  int           chunk_size,
  float         temperature,
  int           topk,
  float         topp,
  float         rpen,
  bool          enable_mm)
{
  printf("%s", prompt);

  int *tokens_buf;
  int  tokens_cap = seqlen * 50;
  MALLOC(tokens_buf, tokens_cap, "tokens_buf", return;);

  InjectContext ctx = {
    .state      = 0,
    .text       = prompt,
    .text_len   = strlen(prompt),
    .tokens_buf = tokens_buf,
    .tokens_len = 0,
    .tokens_cap = tokens_cap,
  };

  sample(
    model, buf, vbuf, seqlen, chunk_size, temperature, topk, topp, rpen,
    enable_mm, &ctx, generate_inject_callback
  );

  free(tokens_buf);
}

/* Build the Gemma chat template */
static InjectData
new_turn(GemmaModel *model, bool use_mm, ChatContext *ctx)
{
  /* Template (from https://ai.google.dev/gemma/docs/core/prompt-structure):
   * <start_of_turn>user\n
   * What is Cramer's Rule?<end_of_turn>\n
   * <start_of_turn>model */

  GemmaTokenizer *tok  = model->tokenizer;
  InjectContext  *jctx = ctx->jctx;

  int        spos, epos;
  InjectData r;

  switch (ctx->state)
  {
    case 0:
      // Get a fresh input from stdin
      printf("\n> ");

      if (fgets(ctx->line_buf, ctx->line_cap, stdin) == NULL)
      {
        if (!is_interrupted())
        {
#ifdef _WIN32
          // Wait 10ms for the console handler to set the flag
          Sleep(10);
#endif
        }
        if (!is_interrupted())
        {
          fprintf(stderr, "\nerror: failed to read user input\n");
        }
        goto fail;
      }
      if (strchr(ctx->line_buf, '\n') == NULL)
      {
        // Input is truncated (too long)
        fprintf(stderr, "\nerror: input too long\n");
        goto fail;
      }
      if (is_interrupted()) goto fail;

      // Set string end
      ctx->line_buf[strcspn(ctx->line_buf, "\n")] = '\0';

      spos                                 = jctx->tokens_len;
      jctx->tokens_buf[jctx->tokens_len++] = tok->sot;  // <start_of_turn>
      encode(
        tok, "user\n", strlen("user\n"), jctx->tokens_buf, spos,
        &jctx->tokens_len
      );
      epos = jctx->tokens_len;

      ctx->state = 1;
      return (InjectData){
        .type     = INJECT_TEXT,
        .tokens   = jctx->tokens_buf + spos,
        .n_tokens = epos - spos,
      };

    case 1:
      jctx->text     = ctx->line_buf;
      jctx->text_len = (int)strlen(ctx->line_buf);
      jctx->state    = 0;

      for (;;)
      {
        r = inject_next(model, jctx, use_mm);
        if (r.type != INJECT_DONE)
        {
          ctx->state = 2;
          return r;

    case 2:
          ;
        }
        else
        {
          break;
        }
      }

      int spos = jctx->tokens_len;

      // No more command ahead, encode whatever plain text remains and wrap up
      // the turn with the chat template's closing tokens
      if (jctx->text_len != 0)
      {
        encode(
          tok, jctx->text, jctx->text_len, jctx->tokens_buf, spos,
          &jctx->tokens_len
        );
      }

      jctx->tokens_buf[jctx->tokens_len++] = tok->eot;
      jctx->tokens_buf[jctx->tokens_len++] = get_token_idx(tok, "\n");
      jctx->tokens_buf[jctx->tokens_len++] = tok->sot;

      encode(
        tok, "model\n", strlen("model\n"), jctx->tokens_buf, jctx->tokens_len,
        &jctx->tokens_len
      );

      int epos = jctx->tokens_len;

      ctx->state = 3;
      return (InjectData){
        .type     = INJECT_TEXT,
        .tokens   = jctx->tokens_buf + spos,
        .n_tokens = epos - spos,
      };

    default:
      ctx->state = GENERATOR_EXIT;
      return (InjectData){.type = INJECT_DONE};

    fail:
      ctx->state = GENERATOR_EXIT;
      return (InjectData){.type = INJECT_QUIT};
  }
}

/* */
static InjectData
chat_inject_callback(int token, GemmaModel *model, bool use_mm, void *ctx)
{
  GemmaTokenizer *tok = model->tokenizer;
  ChatContext    *cc  = (ChatContext *)ctx;

  if (token == tok->eos || token == tok->eot)
  {
    // Model completed the turn
    cc->state = 0;
    return new_turn(model, use_mm, ctx);
  }
  if (token == EOF)
  {
    // Requesting more injection data
    return new_turn(model, use_mm, ctx);
  }

  char byte_buf[2];
  printf("%s", decode(tok, token, byte_buf));
  fflush(stdout);
  return (InjectData){.type = INJECT_NONE};
}

/* */
int
chat(
  GemmaModel   *model,
  TextBuffer   *buf,
  VisionBuffer *vbuf,
  int           seqlen,
  int           chunk_size,
  float         temperature,
  int           topk,
  float         topp,
  float         rpen,
  bool          use_mm)
{
  int  *tokens_buf = NULL;
  char *line_buf   = NULL;

  int tokens_cap = seqlen * 50;
  int line_cap   = seqlen * 50;
  MALLOC(tokens_buf, tokens_cap, "tokens_buf", goto fail;);
  MALLOC(line_buf, line_cap, "line_buf", goto fail;);

  InjectContext jctx = (InjectContext){
    .tokens_buf = tokens_buf,
    .tokens_len = 0,
    .tokens_cap = tokens_cap,
  };
  ChatContext ctx = (ChatContext){
    .state    = 0,
    .line_buf = line_buf,
    .line_cap = line_cap,
    .jctx     = &jctx,
  };

  sample(
    model, buf, vbuf, seqlen, chunk_size, temperature, topk, topp, rpen, use_mm,
    &ctx, chat_inject_callback
  );

fail:
  free(tokens_buf);
  free(line_buf);
  return 1;
}

// CLI

/* */
static inline bool
safe_atoui(const char *str, unsigned int *result)
{
  if (str == NULL) return false;

  errno = 0;

  char     *endptr = NULL;
  long long val    = strtoll(str, &endptr, 10);

  // Invalid number
  if (endptr == str) return false;
  // Extra characters at the end
  if (*endptr != '\0') return false;
  // long overflow
  if (errno == ERANGE) return false;
  // uint overflow
  if (val < 0 || val > (long long)UINT_MAX) return false;

  // All passed
  *result = (unsigned int)val;
  return true;
}

/* */
static inline bool
safe_atof(const char *str, float *result)
{
  if (str == NULL) return false;

  errno = 0;

  char *endptr = NULL;
  float val    = strtof(str, &endptr);

  // Invalid number
  if (endptr == str) return false;
  // Extra characters at the end
  if (*endptr != '\0') return false;
  // Overflow
  if (errno == ERANGE) return false;
  // inf / nan
  if (isinf(val) || isnan(val)) return false;

  // All passed
  *result = val;
  return true;
}

/* Pretty-print of the loaded model */
void
print_model_config(
  GemmaModel *model, int seqlen, int chunk_size, bool enable_mm)
{
  const int       width  = 20;
  bool            use_mm = model->support_mm && enable_mm;
  VisionEncoder  *enc    = model->encoder;
  TextDecoder    *dec    = model->decoder;
  GemmaTokenizer *tok    = model->tokenizer;
  TextConfig     *cfg    = dec->config;

  printf("\n========== model configuration ==========\n");
  printf("architecture:\n");

  // Integer fields
  printf("  %-*s: %d\n", width, "n_layers", cfg->n_layers);
  printf("  %-*s: %d\n", width, "n_heads", cfg->n_heads);
  printf("  %-*s: %d\n", width, "n_kv_heads", cfg->n_kv_heads);
  printf("  %-*s: %d\n", width, "head_dim", cfg->head_dim);
  printf("  %-*s: %d\n", width, "embed_dim", cfg->embed_dim);
  printf("  %-*s: %d\n", width, "mlp_dim", cfg->mlp_dim);
  printf("  %-*s: %d\n", width, "q_scale", cfg->q_scale);
  printf("  %-*s: %d\n", width, "slide_len", cfg->slide_len);
  printf("  %-*s: %d\n", width, "image_toks", cfg->image_toks);
  printf("  %-*s: %d\n", width, "max_seqlen", cfg->max_seqlen);
  printf("  %-*s: %d\n", width, "vocab_size", cfg->vocab_size);

  // Float fields
  printf("  %-*s: %.6f\n", width, "local_theta", cfg->local_theta);
  printf("  %-*s: %.6f\n", width, "global_theta", cfg->global_theta);
  printf("  %-*s: %.6f\n", width, "eps", cfg->eps);
  printf("  %-*s: %.6f\n", width, "att_softcap", cfg->att_softcap);
  printf("  %-*s: %.6f\n", width, "logit_softcap", cfg->logit_softcap);

  // Array fields
  printf("  %-*s: ", width, "att_layers");
  for (int i = 0; i < cfg->n_layers; i++)
  {
    printf("%d", cfg->att_layers[i] ? 1 : 0);
    if ((i + 1) % (width - 1) == 0 && i + 1 < cfg->n_layers)
    {
      printf("\n");
      for (int j = 0; j < width + 4; j++)
      {
        printf(" ");
      }
    }
  }
  printf("\n");

  // Boolean fields
  printf("  %-*s: %d\n", width, "qk_norm", cfg->qk_norm);
  printf("  %-*s: %d\n", width, "pre_mlp_norm", cfg->pre_mlp_norm);
  printf("  %-*s: %d\n", width, "pst_mlp_norm", cfg->pst_mlp_norm);

  // Vision encoder architecture (if available and enabled)
  if (use_mm)
  {
    VisionConfig *vcfg = enc->config;
    printf("\nvision:\n");
    printf("  %-*s: %d\n", width, "n_layers", vcfg->n_layers);
    printf("  %-*s: %d\n", width, "image_size", vcfg->image_size);
    printf("  %-*s: %d\n", width, "patch_size", vcfg->patch_size);
    printf("  %-*s: %d\n", width, "hidden_dim", vcfg->hidden_dim);
    printf("  %-*s: %d\n", width, "n_heads", vcfg->n_heads);
    printf("  %-*s: %d\n", width, "mlp_dim", vcfg->mlp_dim);
    printf("  %-*s: %.6f\n", width, "eps", vcfg->eps);
  }

  // Runtime flags
  printf("\nruntime:\n");
  printf("  %-*s: %d\n", width, "quant", model->quant);
  printf("  %-*s: %d\n", width, "support_mm", model->support_mm);

  // Tokenizer
  printf("\ntokenizer:\n");
  printf("  %-*s: %d\n", width, "vocab_size", tok->vocab_size);
  printf("  %-*s: %d\n", width, "n_merges", tok->n_merges);
  printf("  %-*s: %d\n", width, "bos", tok->bos);
  printf("  %-*s: %d\n", width, "eos", tok->eos);
  printf("  %-*s: %d\n", width, "sot", tok->sot);
  printf("  %-*s: %d\n", width, "eot", tok->eot);
  printf("  %-*s: %d\n", width, "soi", tok->soi);
  printf("  %-*s: %d\n", width, "eoi", tok->eoi);
  printf("  %-*s: %d\n", width, "ist", tok->ist);

  printf("\nmemory footprint (estimated):\n");

  int   C   = cfg->embed_dim;
  int   L   = cfg->n_layers;
  int   CH  = cfg->head_dim;
  int   NH  = cfg->n_heads;
  int   Cq  = NH * CH;
  int   Ckv = cfg->n_kv_heads * CH;
  int   CM  = cfg->mlp_dim;
  int   vs  = cfg->vocab_size;
  float GB  = 1024.0 * 1024.0 * 1024.0;

  // Text buffer
  size_t decB = 0;

  int           ppi  = 0;
  VisionConfig *vcfg = NULL;
  if (use_mm)
  {
    vcfg = enc->config;
    ppi  = vcfg->image_size / vcfg->patch_size;
  }
  int mult = use_mm ? max(ppi * ppi, chunk_size) : chunk_size;

  // Quantized buffers
  if (model->quant)
  {
    decB += C * sizeof(int8_t);   // x_i8
    decB += Cq * sizeof(int8_t);  // xo_i8
    decB += CM * sizeof(int8_t);  // xg_i8
  }

  // Main buffers
  decB += L * 2 * seqlen * Ckv * sizeof(floatx);  // kv_cache
  decB += vs   * sizeof(floatx);                  // logits
  decB += mult * C   * sizeof(floatx);            // x
  decB += mult * C   * sizeof(floatx);            // resid
  decB += mult * Cq  * sizeof(floatx);            // xq
  decB += mult * Ckv * sizeof(floatx);            // xk
  decB += mult * CH  * sizeof(floatx);            // csfreqs_slid
  decB += mult * CH  * sizeof(floatx);            // csfreqs_full
  decB += mult * Ckv * sizeof(floatx);            // xv
  decB += mult * Cq  * sizeof(floatx);            // xo
  decB += mult * NH  * seqlen * sizeof(floatx);   // att
  decB += mult * CM  * sizeof(floatx);            // xg
  decB += mult * CM  * sizeof(floatx);            // xu

  printf("  %-*s: %.2f GB\n", width, "decoder buffer", (float)decB / GB);

  // Vision buffer (if applicable)
  if (use_mm)
  {
    size_t encB = 0;

    int C  = vcfg->hidden_dim;
    int CM = vcfg->mlp_dim;
    int N  = ppi * ppi;
    int NH = vcfg->n_heads;

    // Quantized buffers
    if (model->quant)
    {
      encB += N * C * sizeof(int8_t);   // x_i8
      encB += N * sizeof(floatx);       // x_scales
      encB += N * CM * sizeof(int8_t);  // mlp_i8
      encB += N * sizeof(floatx);       // mlp_scales
    }

    // Main buffers
    encB += N * C  * sizeof(floatx);       // x
    encB += N * C  * sizeof(floatx);       // resid
    encB += N * C  * sizeof(floatx);       // xq
    encB += N * C  * sizeof(floatx);       // xk
    encB += N * C  * sizeof(floatx);       // xv
    encB += N * C  * sizeof(floatx);       // att_out
    encB += N * CM * sizeof(floatx);       // mlp_hidden
    encB += N * N  * NH * sizeof(floatx);  // scores

    printf("  %-*s: %.2f GB\n", width, "encoder buffer", (float)encB / GB);
  }

  // Weights
  if (use_mm)
  {
    float encGB_w = get_vision_encoder_size(vcfg, cfg, model->quant) / GB;
    printf("  %-*s: %.2f GB\n", width, "encoder weights", encGB_w);
  }
  float decGB_w = get_text_decoder_size(cfg, model->quant) / GB;
  printf("  %-*s: %.2f GB\n", width, "decoder weights", decGB_w);

  // KV Cache
  float kvcGB = (float)cfg->n_layers * 2 * Ckv * sizeof(floatx) / 1024.0;
  printf("  %-*s: %.2f KB\n", width, "kv cache (tok)", kvcGB);
  printf("=========================================\n\n");
}

/* */
void
print_usage(void)
{
  // clang-format off
  printf(
  "usage:\n"
  "  ./gemma <modelfile> [options]\n"
  "\n"
  "arguments:\n"
  "  modelfile          path to the model file\n"
  "\n"
  "options:\n"
  "  --seqlen <N>       set sequence length"
                            " (default: " TOSTRING(DEFAULT_SEQLEN) ")\n"
  "  --topk <N>         set top-k sampling value"
                            " (default: " TOSTRING(DEFAULT_TOPK) ")\n"
  "  --seed <N>         set random seed"
                            " (default: current time)\n"
  "  --chunk <N>        set prefilling chunk size, must be >= 1"
                            " (default: " TOSTRING(DEFAULT_CHUNK_SIZE) ")\n"
  "  --temperature <F>  set temperature value, must be >= 0.0"
                            " (default: " TOSTRING(DEFAULT_TEMPERATURE) ")\n"
  "  --topp <F>         set top-p sampling value, must be 0.0 < p <= 1.0"
                            " (default: " TOSTRING(DEFAULT_TOPP) ")\n"
  "  --rpen <F>         set repetition penalty, must be >= 1.0"
                            " (default: " TOSTRING(DEFAULT_RPEN) ")\n"
  "  --prompt <S>       set input prompt, ignored if chat mode is enabled"
                            " (default: \"" DEFAULT_PROMPT "\")\n"
  "  --chat             enable chat mode\n"
  "  --disable-mm       disable multimodal capability\n"
  "  --disable-mmap     disable mmap (memory mapped file)\n"
  "  --verbose          print model info\n"
  "  --help, -?         display this help message\n"
  "\n"
  "controls:\n"
  "  Ctrl+C             gracefully interrupt generation and exit\n"
  "\n"
  "examples:\n"
  "  ./gemma model.bin -l 2048 -t 0.8 --chat\n"
  "  ./gemma model.bin -i \"Hello I'm a language model,\""
          " --seqlen 4096 --topk 50 --seed 12345\n");
  // clang-format on
}

/* */
static const char *
safe_get_arg(int i, int argc, char **argv)
{
  if (i + 1 >= argc)
  {
    print_usage();
    fprintf(stderr, "error: option '%s' requires an argument.\n", argv[i]);
    return NULL;
  }
  return argv[i + 1];
}

/* */
int
main(int argc, char **argv)
{
  set_utf8_console();
  setup_signal_handler();

  unsigned int seqlen      = DEFAULT_SEQLEN;
  unsigned int topk        = DEFAULT_TOPK;
  unsigned int seed        = (unsigned int)time(NULL);
  unsigned int chunk_size  = DEFAULT_CHUNK_SIZE;
  float        temperature = DEFAULT_TEMPERATURE;
  float        topp        = DEFAULT_TOPP;
  float        rpen        = DEFAULT_RPEN;
  const char  *prompt      = DEFAULT_PROMPT;
  bool         chatmode    = false;
  bool         enable_mm   = true;
  bool         enable_mmap = true;
  bool         print_cfg   = false;

  GemmaModel   *model = NULL;
  TextConfig   *cfg   = NULL;
  VisionConfig *vcfg  = NULL;
  TextBuffer   *buf   = NULL;
  VisionBuffer *vbuf  = NULL;

  // On Windows, get UTF-8 encoded command line arguments
  char **utf8_argv = get_utf8_argv(&argc);
  if (utf8_argv != NULL)
  {
    argv = utf8_argv;
  }

  if (argc < 2)
  {
    print_usage();
    fprintf(stderr, "\nerror: model filename is not provided\n");
    goto fail;
  }

  char *modelfile = argv[1];
  if (strcmp(modelfile, "--help") == 0 || strcmp(modelfile, "-?") == 0)
  {
    print_usage();
    goto end;
  }

  const char *val;

  // Parse the command line arguments
  for (int i = 2; i < argc; i++)
  {
    const char *arg = argv[i];

    if (strcmp(arg, "--seqlen") == 0)
    {
      val = safe_get_arg(i++, argc, argv);
      if (!val) goto fail;
      if (!safe_atoui(val, &seqlen))
      {
        fprintf(stderr, "error: invalid number for --seqlen: %s\n", val);
        goto fail;
      }
    }
    else if (strcmp(arg, "--topk") == 0)
    {
      val = safe_get_arg(i++, argc, argv);
      if (!val) goto fail;
      if (!safe_atoui(val, &topk))
      {
        fprintf(stderr, "error: invalid number for --topk: %s\n", val);
        goto fail;
      }
    }
    else if (strcmp(arg, "--seed") == 0)
    {
      val = safe_get_arg(i++, argc, argv);
      if (!val) goto fail;
      if (!safe_atoui(val, &seed))
      {
        fprintf(stderr, "error: invalid number for --seed: %s\n", val);
        goto fail;
      }
    }
    else if (strcmp(arg, "--chunk") == 0)
    {
      val = safe_get_arg(i++, argc, argv);
      if (!val) goto fail;
      if (!safe_atoui(val, &chunk_size) || chunk_size == 0)
      {
        fprintf(stderr, "error: invalid number for --chunk: %s\n", val);
        goto fail;
      }
    }
    else if (strcmp(arg, "--temperature") == 0)
    {
      val = safe_get_arg(i++, argc, argv);
      if (!val) goto fail;
      if (!safe_atof(val, &temperature) || temperature < 0.0f)
      {
        fprintf(stderr, "error: invalid temperature: %s (must be >= 0)\n", val);
        goto fail;
      }
    }
    else if (strcmp(arg, "--topp") == 0)
    {
      val = safe_get_arg(i++, argc, argv);
      if (!val) goto fail;
      if (!safe_atof(val, &topp) || topp <= 0.0f || topp > 1.0f)
      {
        fprintf(stderr, "error: invalid top-p: %s (must be 0 < p <= 1)\n", val);
        goto fail;
      }
    }
    else if (strcmp(arg, "--rpen") == 0)
    {
      val = safe_get_arg(i++, argc, argv);
      if (!val) goto fail;
      if (!safe_atof(val, &rpen) || rpen < 1.0f)
      {
        fprintf(
          stderr, "error: invalid repetition penalty: %s (must be >= 1)\n", val
        );
        goto fail;
      }
    }
    else if (strcmp(arg, "--prompt") == 0)
    {
      val = safe_get_arg(i++, argc, argv);
      if (!val) goto fail;
      prompt = val;
    }
    else if (strcmp(arg, "--chat") == 0)
    {
      chatmode = true;
    }
    else if (strcmp(arg, "--disable-mm") == 0)
    {
      enable_mm = false;
    }
    else if (strcmp(arg, "--disable-mmap") == 0)
    {
      enable_mmap = false;
    }
    else if (strcmp(arg, "--verbose") == 0)
    {
      print_cfg = true;
    }
    else if (strcmp(arg, "--help") == 0 || strcmp(arg, "-?") == 0)
    {
      print_usage();
      goto end;
    }
    else
    {
      fprintf(stderr, "error: unknown option: %s\n", arg);
      print_usage();
      goto fail;
    }
  }

  srand(seed);

  // Read / mmap model
  if (enable_mmap)
  {
    model = mmap_gemma_model(modelfile, enable_mm);
  }
  else
  {
    model = read_gemma_model(modelfile, enable_mm);
  }

  if (model == NULL) goto fail;

  // Text config
  cfg = model->decoder->config;
  // Vision config
  if (model->encoder != NULL)
  {
    vcfg = model->encoder->config;
  }

  bool use_mm = enable_mm && model->support_mm;

  // Text buffer
  buf = malloc_text_buffer(
    cfg, vcfg, (int)seqlen, (int)chunk_size, use_mm, model->quant
  );
  if (buf == NULL) goto fail;

  // Vision buffer
  if (use_mm)
  {
    vbuf = malloc_vision_buffer(vcfg, model->quant);
    if (vbuf == NULL) goto fail;
  }

  if (print_cfg)
  {
    print_model_config(model, (int)seqlen, (int)chunk_size, enable_mm);
  }

  if (chatmode)
  {
    chat(
      model, buf, vbuf, (int)seqlen, (int)chunk_size, temperature, (int)topk,
      topp, rpen, use_mm
    );
  }
  else
  {
    generate(
      model, buf, vbuf, prompt, (int)seqlen, (int)chunk_size, temperature,
      (int)topk, topp, rpen, use_mm
    );
  }

  if (is_interrupted())
  {
    printf("\n");
  }

  int r;
end:
  r = 0;
  goto cleanup;
fail:
  r = 1;
  goto cleanup;

cleanup:
  free_utf8_argv(utf8_argv, argc);
  if (model != NULL)
  {
    free_text_buffer(buf, model->quant);
    free_vision_buffer(vbuf, model->quant);
    if (enable_mmap)
    {
      munmap_gemma_model(model);
    }
    else
    {
      free_gemma_model(model);
    }
  }
  return r;
}