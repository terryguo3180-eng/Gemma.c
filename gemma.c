/* Gemma 1 & 2 & 3 implemented in a single file of pure C.
 *
 * One big self-contained inference engine. No external deps beyond the C
 * standard library + OpenMP for the parallel bits.
 * Supports W8A8 quantization, sliding-window + full attention, optional
 * vision encoder for multimodal models, and a simple BPE tokenizer.
 */

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#ifdef _WIN32
#  include <time.h>
#  include <windows.h>
#else
#  undef _POSIX_C_SOURCE
#  define _POSIX_C_SOURCE 199309L
#  include <time.h>
#endif

// Default values for cli
#define DEFAULT_SEQLEN      16384
#define DEFAULT_TOPK        0
#define DEFAULT_CHUNK_SIZE  64
#define DEFAULT_TEMPERATURE 1.0
#define DEFAULT_TOPP        1.0
#define DEFAULT_RPEN        1.0
#define DEFAULT_PROMPT      "Once upon a time"

#define TOSTRING(x) STRINGIFY_(x)
// I usually use a trailing underscore to indicate a variable/function/macro is
// temporary
#define STRINGIFY_(x) #x

// Matrix multiplication block sizes
#define GEMM_NT_MR 4
#define GEMM_NT_NR 4
#define GEMM_NN_MR 4
#define GEMM_I8_MR 4
#define GEMM_I8_NR 4

// To compile the code for a different dtype, simply append -DDTYPE=... (e.g.
// -DTYPE=DTYPE_BF16)
#define DTYPE_FP16 1
#define DTYPE_BF16 2
#define DTYPE_FP32 3

#ifndef DTYPE
#  define DTYPE DTYPE_FP16
#endif

// clang-format off
#if DTYPE == DTYPE_FP16
#  define floatx     _Float16
#  define DTYPE_STR  "float16"
#  define DTYPE_CODE 1
#  define FLOATX_MAX (((union {floatx f; uint16_t b; }){.b = 0x00007BFF}).f)
#elif DTYPE == DTYPE_BF16
#  define floatx     __bf16
#  define DTYPE_STR  "bfloat16"
#  define DTYPE_CODE 2
#  define FLOATX_MAX (((union {floatx f; uint16_t b; }){.b = 0x00007F7F}).f)
#elif DTYPE == DTYPE_FP32
#  define floatx     float
#  define DTYPE_STR  "float32"
#  define DTYPE_CODE 3
#  define FLOATX_MAX (((union {floatx f; uint32_t b; }){.b = 0x7F7FFFFF}).f)
#else
#  error "unsupported DTYPE"
#endif

// thread_local looks much cleaner, much nicer :D
#define thread_local _Thread_local

// User prompt/tokens buffer
char thread_local *g_prompt_buf     = NULL;
int  thread_local  g_prompt_buf_len = 0;
int  thread_local *g_tokens_buf     = NULL;
int  thread_local  g_tokens_buf_len = 0;
// clang-format on

// Block size of blockwise causal masking
static thread_local unsigned int qk_block_size = 64;

// Global interruption flag
static volatile sig_atomic_t g_interrupted = 0;

// safe MIN/MAX macros

// Statement expressions: ({ }), an extension of GNU C. Not supported in MSVC
#define MAX(a, b)                      \
  ({                                   \
    __typeof__(a) _max_a = (a);        \
    __typeof__(a) _max_b = (b);        \
    _max_a > _max_b ? _max_a : _max_b; \
  })

#define MIN(a, b)                      \
  ({                                   \
    __typeof__(a) _max_a = (a);        \
    __typeof__(a) _max_b = (b);        \
    _max_a < _max_b ? _max_a : _max_b; \
  })

/* ------------------------------------------------------------------ */
/* Cross-platform Interruption handling                               */
/* ------------------------------------------------------------------ */

/* */
static void
signal_handler(int signum)
{
  (void)signum;
  g_interrupted = 1;
}

#ifdef _WIN32
#  include <windows.h>
#  define SLEEP_SEC(sec) Sleep((sec) * 1000)
#  define GETPID()       GetCurrentProcessId()
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
      g_interrupted = 1;
      return TRUE;
    default:
      return FALSE;
  }
}
#else
#  include <unistd.h>
#  define SLEEP_SEC(sec) sleep(sec)
#  define GETPID()       getpid()
#endif

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
#endif
}

/* ------------------------------------------------------------------ */
/* Cross-platform wall time                                           */
/* ------------------------------------------------------------------ */

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

/* ------------------------------------------------------------------ */
/* Memory wrappers, fail fast and print a useful message              */
/* ------------------------------------------------------------------ */

#define MALLOC(ptr, count, name, fail)                                     \
  do                                                                       \
  {                                                                        \
    ptr = malloc((count) * sizeof(*(ptr)));                                \
    if (ptr == NULL)                                                       \
    {                                                                      \
      fprintf(stderr, "error: memory allocation failed: %s (size: %zu)\n", \
          (name), (size_t)(count) * sizeof(*(ptr)));                       \
      fail                                                                 \
    }                                                                      \
  }                                                                        \
  while (0)

#define CALLOC(ptr, count, name, fail)                                     \
  do                                                                       \
  {                                                                        \
    ptr = calloc((count), sizeof(*(ptr)));                                 \
    if (ptr == NULL)                                                       \
    {                                                                      \
      fprintf(stderr, "error: memory allocation failed: %s (size: %zu)\n", \
          (name), (size_t)(count) * sizeof(*(ptr)));                       \
      fail                                                                 \
    }                                                                      \
  }                                                                        \
  while (0)

/* ------------------------------------------------------------------ */
/* File-reading helpers (big-endian integers + pascal strings)        */
/* ------------------------------------------------------------------ */

#define FGETC(fp, name, fail)                                 \
  ({                                                          \
    int fgetc_ret_ = fgetc(fp);                               \
    if (fgetc_ret_ == EOF)                                    \
    {                                                         \
      fprintf(stderr, "error: File read failed: %s", (name)); \
      fail                                                    \
    }                                                         \
    fgetc_ret_;                                               \
  })

#define FREAD(ptr, count, fp, name, fail)                                      \
  do                                                                           \
  {                                                                            \
    size_t fread_buf_ = fread((ptr), sizeof(*(ptr)), (count), (fp));           \
    if (fread_buf_ != (size_t)(count))                                         \
    {                                                                          \
      fprintf(stderr, "error: file read failed: %s (expected %zu, got %zu)\n", \
          (name), (size_t)(count), fread_buf_);                                \
      fail                                                                     \
    }                                                                          \
  }                                                                            \
  while (0)

#define READ_UINT16(fp, name, fail)               \
  ({                                              \
    /* Big-endian */                              \
    unsigned char ru16_buf_[2];                   \
    FREAD(ru16_buf_, 2, (fp), (name), fail);      \
    ((int)ru16_buf_[0] << 8) | (int)ru16_buf_[1]; \
  })

#define READ_UINT32(fp, name, fail)                                   \
  ({                                                                  \
    /* Big-endian */                                                  \
    unsigned char ru32_buf_[4];                                       \
    FREAD(ru32_buf_, 4, (fp), (name), fail);                          \
    ((uint32_t)ru32_buf_[0] << 24) | ((uint32_t)ru32_buf_[1] << 16) | \
        ((uint32_t)ru32_buf_[2] << 8) | ((uint32_t)ru32_buf_[3]);     \
  })

#define READ_FP32(fp, name, fail)                           \
  ({                                                        \
    uint32_t rfp32_bits_ = READ_UINT32((fp), (name), fail); \
    float    rfp32_val_;                                    \
    memcpy(&rfp32_val_, &rfp32_bits_, sizeof(float));       \
    rfp32_val_;                                             \
  })

// Pascal-style string: first byte is length, then that many chars
#define READ_STR(fp, data, offset, name, fail)          \
  ({                                                    \
    char rstr_len_name_[128];                           \
    snprintf(rstr_len_name_, 128, "%s.length", (name)); \
    int   rstr_len_ = FGETC(fp, rstr_len_name_, fail);  \
    char *rstr_ptr_ = (data) + *(offset);               \
    FREAD(rstr_ptr_, rstr_len_, (fp), (name), fail);    \
    (data)[*(offset) + rstr_len_] = '\0';               \
    *(offset) += rstr_len_ + 1;                         \
    rstr_ptr_;                                          \
  })

#define READ_TENSOR(ptr, count, fp, name, fail) \
  do                                            \
  {                                             \
    MALLOC((ptr), (count), (name), fail);       \
    FREAD((ptr), (count), (fp), (name), fail);  \
  }                                             \
  while (0)

// Linear layer weights: either plain floatx or int8 + per-channel scales
#define READ_LINEAR(w, fp, m, n, quant, name, fail)                            \
  do                                                                           \
  {                                                                            \
    MALLOC((w), 1, (name), fail);                                              \
    if (!(quant))                                                              \
    {                                                                          \
      (w)->dtype = DTYPE_FPX;                                                  \
      char rlinear_fpx_name_[128];                                             \
      snprintf(rlinear_fpx_name_, 128, "%s.fpx", (name));                      \
      READ_TENSOR(                                                             \
          (w)->fpx, (size_t)(m) * (size_t)(n), (fp), rlinear_fpx_name_, fail); \
    }                                                                          \
    else                                                                       \
    {                                                                          \
      (w)->dtype = DTYPE_INT8;                                                 \
      char rlinear_i8q_name_[128];                                             \
      char rlinear_i8scales_name_[128];                                        \
      snprintf(rlinear_i8q_name_, 128, "%s.i8.q", (name));                     \
      snprintf(rlinear_i8scales_name_, 128, "%s.i8.scales", (name));           \
      READ_TENSOR((w)->i8.q, (size_t)(m) * (size_t)(n), (fp),                  \
          rlinear_i8q_name_, fail);                                            \
      READ_TENSOR((w)->i8.scales, (n), (fp), rlinear_i8scales_name_, fail);    \
    }                                                                          \
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
    int len = FGETC(fp, "<str.length>", return -1;);
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

/* ------------------------------------------------------------------ */
/* Model configuration structs                                        */
/* ------------------------------------------------------------------ */

/* */
typedef struct
{
  int   n_layers;
  int   image_size;
  int   patch_size;
  int   hidden_dim;
  int   n_heads;
  int   mlp_dim;
  float eps;
} VisionConfig;

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
  bool  support_mm;     // model has a vision tower
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

/* ------------------------------------------------------------------ */
/* Tokenizer                                                          */
/* ------------------------------------------------------------------ */

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

/* Look up a string in the sorted vocab -> token id (or -1) */
static int
get_token_idx(GemmaTokenizer *tok, char *str)
{
  // Get idx_to_vocab[string]
  Token  key = {.val = str};
  Token *val = bsearch(&key, tok->vocab_sorted, tok->vocab_size,
      sizeof(tok->vocab_sorted[0]), cmp_token);
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
  Merge *val = bsearch(
      &key, tok->ranks, tok->n_merges, sizeof(tok->ranks[0]), cmp_merge);
  return val;  // NULL if not found
}

/* Minimal byte-pair encoding algorithm implementation */
int *
encode(
    GemmaTokenizer *tok, const char *sstr, int len, int *tokens, int *n_tokens)
{
  unsigned char *str = (unsigned char *)sstr;

  // Convert UTF-8 string to tokens of individual codepoints
  int i     = 0;
  int tok_i = 0;

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

    if (str[i] >> 7 == 0)
    {
      n_bytes = 1;
    }
    else if (i + 1 < len && str[i] >> 5 == 6 && str[i + 1] >> 6 == 2)
    {
      n_bytes = 2;
    }
    else if (i + 2 < len && str[i] >> 4 == 14 && str[i + 1] >> 6 == 2 &&
             str[i + 2] >> 6 == 2)
    {
      n_bytes = 3;
    }
    else if (i + 3 < len && str[i] >> 3 == 30 && str[i + 1] >> 6 == 2 &&
             str[i + 2] >> 6 == 2 && str[i + 3] >> 6 == 2)
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
      cstr[b] = (char)str[i++];
    }
    cstr[n_bytes] = '\0';

    int token = get_token_idx(tok, cstr);
    if (token == -1)
    {
      // Fallback to byte level tokens
      char bstr[10];
      for (int b = start; b < i; b++)
      {
        snprintf(bstr, 10, "<0x%02X>", (unsigned char)str[b]);
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
          .str1 = tok->vocab[tokens[i]], .str2 = tok->vocab[tokens[i + 1]]};
      if (cmp_merge(&pair, best_pair) == 0)
      {
        // Merge the pair, left shift all the tokens on its right side
        char merged[128];
        snprintf(
            merged, sizeof(merged), "%s%s", best_pair->str1, best_pair->str2);
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
  return tokens;
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

/* ------------------------------------------------------------------ */
/* Weight storage                                                     */
/* ------------------------------------------------------------------ */

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
void
free_vision_encoder(VisionEncoder *enc)
{
  if (enc == NULL) return;

  free(enc->patch_emb);
  free(enc->patch_emb_b);
  free_linear(enc->pos_embedding);
  if (enc->layers != NULL)
  {
    for (int i = 0; i < enc->config->n_layers; i++)
    {
      free_vision_layer(enc->layers[i]);
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
typedef struct
{
  TextConfig     *config;
  GemmaTokenizer *tokenizer;
  // (vocab_size, embed_dim), shared with lm_head (tied weights)
  Linear            *embedding;
  TextDecoderLayer **layers;
  floatx            *final_norm;  // (embed_dim,)
} TextDecoder;

/* */
void
free_text_decoder(TextDecoder *dec)
{
  if (dec == NULL) return;

  free_tokenizer(dec->tokenizer);
  free_linear(dec->embedding);
  if (dec->layers != NULL && dec->config != NULL)
  {
    for (int i = 0; i < dec->config->n_layers; i++)
    {
      free_text_layer(dec->layers[i]);
    }
    free(dec->layers);
  }
  free(dec->final_norm);
  free_text_config(dec->config);
  free(dec);
}

/* Top-level model container */
typedef struct
{
  TextDecoder   *decoder;
  VisionEncoder *encoder;
  bool           quant;  // W8A8
} GemmaModel;

/* */
void
free_gemma_model(GemmaModel *model)
{
  if (model == NULL) return;
  free_vision_encoder(model->encoder);
  free_text_decoder(model->decoder);
  free(model);
}

/* ------------------------------------------------------------------ */
/* Runtime buffers (allocated once, reused every step)                */
/* ------------------------------------------------------------------ */

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
malloc_text_buffer(TextConfig *cfg,
    VisionConfig              *vcfg,

    int  cache_len,
    int  prefill_chunk,
    bool enable_mm,
    bool quant)
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

  int mult = prefill_chunk;

  // Multimodal models need space for a whole image worth of tokens
  if (cfg->support_mm && enable_mm && vcfg != NULL)
  {
    mult = MAX(mult, cfg->image_toks);
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

/* ------------------------------------------------------------------ */
/* Model loading                                                      */
/* ------------------------------------------------------------------ */

/* */
GemmaModel *
read_model(const char *filename, bool enable_mm)
{
  FILE           *fp    = NULL;
  GemmaTokenizer *tok   = NULL;
  TextConfig     *cfg   = NULL;
  VisionEncoder  *enc   = NULL;
  VisionConfig   *vcfg  = NULL;
  TextDecoder    *dec   = NULL;
  GemmaModel     *model = NULL;

  char *att_layers_buf = NULL;

  fp = fopen(filename, "rb");
  if (fp == NULL)
  {
    fprintf(stderr, "error: failed to open file: %s\n", filename);
    goto fail;
  }

  CALLOC(tok, 1, "model.decoder.tokenizer", goto fail;);
  CALLOC(cfg, 1, "model.decoder.config", goto fail;);
  // Build the text model
  CALLOC(dec, 1, "model.decoder", goto fail;);
  dec->config    = cfg;
  dec->tokenizer = tok;
  // Build the full model
  CALLOC(model, 1, "model", goto fail;);
  model->decoder = dec;

  // Read the configs
  cfg->n_layers   = FGETC(fp, "model.decoder.config.n_layers", goto fail;);
  cfg->n_heads    = FGETC(fp, "model.decoder.config.n_heads", goto fail;);
  cfg->n_kv_heads = FGETC(fp, "model.decoder.config.n_kv_heads", goto fail;);

  cfg->head_dim = READ_UINT16(fp, "model.decoder.config.head_dim", goto fail;);
  cfg->embed_dim =
      READ_UINT16(fp, "model.decoder.config.embed_dim", goto fail;);
  cfg->mlp_dim = READ_UINT16(fp, "model.decoder.config.mlp_dim", goto fail;);
  cfg->q_scale = READ_UINT16(fp, "model.decoder.config.q_scale", goto fail;);
  cfg->slide_len =
      READ_UINT16(fp, "model.decoder.config.slide_len", goto fail;);
  cfg->image_toks =
      READ_UINT16(fp, "model.decoder.config.image_toks", goto fail;);
  cfg->max_seqlen =
      READ_UINT32(fp, "model.decoder.config.max_seqlen", goto fail;);
  cfg->vocab_size =
      READ_UINT32(fp, "model.decoder.config.vocab_size", goto fail;);

  cfg->local_theta =
      READ_FP32(fp, "model.decoder.config.local_theta", goto fail;);
  cfg->global_theta =
      READ_FP32(fp, "model.decoder.config.global_theta", goto fail;);
  cfg->eps = READ_FP32(fp, "model.decoder.config.eps", goto fail;);
  cfg->att_softcap =
      READ_FP32(fp, "model.decoder.config.att_softcap", goto fail;);
  cfg->logit_softcap =
      READ_FP32(fp, "model.decoder.config.logit_softcap", goto fail;);

  // Packed bit-field of which layers use sliding-window attention
  // A terrible terrible idea, wish I didn't do this
  int n_bytes = FGETC(fp, "model.decoder.config.att_layers", goto fail;);
  if (n_bytes * 8 < cfg->n_layers)
  {
    fprintf(stderr, "error: insufficient att_layers bytes\n");
    goto fail;
  }
  MALLOC(
      att_layers_buf, n_bytes, "model.decoder.config.att_layers", goto fail;);
  MALLOC(cfg->att_layers, cfg->n_layers, "model.decoder.config.att_layers",
         goto fail;);
  FREAD(att_layers_buf, n_bytes, fp, "model.decoder.config.att_layers",
        goto fail;);
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
  int extra_flags   = FGETC(fp, "model.decoder.config.extra_flags", goto fail;);
  cfg->support_mm   = (extra_flags & 16) == 16;
  cfg->qk_norm      = (extra_flags & 8) == 8;
  cfg->pre_mlp_norm = (extra_flags & 4) == 4;
  cfg->pst_mlp_norm = (extra_flags & 2) == 2;
  model->quant      = (extra_flags & 1);

  bool use_mm = cfg->support_mm && enable_mm;

  if (use_mm)
  {
    // Read vision config
    CALLOC(enc, 1, "model.encoder", goto fail;);
    CALLOC(vcfg, 1, "model.encoder.config", goto fail;);
    vcfg->n_layers = FGETC(fp, "model.encoder.config.n_layers", goto fail;);
    vcfg->n_heads  = FGETC(fp, "model.encoder.config.n_heads", goto fail;);
    vcfg->mlp_dim = READ_UINT16(fp, "model.encoder.config.mlp_dim", goto fail;);
    vcfg->hidden_dim =
        READ_UINT16(fp, "model.encoder.config.hidden_dim", goto fail;);
    vcfg->image_size =
        READ_UINT16(fp, "model.encoder.config.image_size", goto fail;);
    vcfg->patch_size =
        READ_UINT16(fp, "model.encoder.config.patch_size", goto fail;);
    vcfg->eps      = READ_FP32(fp, "model.encoder.config.eps", goto fail;);
    enc->config    = vcfg;
    model->encoder = enc;
  }
  else if (cfg->support_mm)
  {
    // Skip vision config if user didn't ask for multimodal
    FGETC(fp, "model.encoder.config.n_layers", goto fail;);
    FGETC(fp, "model.encoder.config.n_heads", goto fail;);
    READ_UINT16(fp, "model.encoder.config.mlp_dim", goto fail;);
    READ_UINT16(fp, "model.encoder.config.hidden_dim", goto fail;);
    READ_UINT16(fp, "model.encoder.config.image_size", goto fail;);
    READ_UINT16(fp, "model.encoder.config.patch_size", goto fail;);
    READ_FP32(fp, "model.encoder.config.eps", goto fail;);
  }

  // dtype
  int  offset = 0;
  char dtype[10];
  READ_STR(fp, dtype, &offset, "dtype", goto fail;);
  if (strcmp(dtype, DTYPE_STR) != 0)
  {
    printf("dtype '%s' not supported\n", dtype);
    goto fail;
  }

  // Build vocabulary
  offset = 0;

  tok->vocab_size = cfg->vocab_size;
  if (cfg->support_mm)
  {
    tok->vocab_size++;
  }  // ++ for the <image_soft_token>
  int vocab_data_bytes = get_strarr_bytes(fp, tok->vocab_size);
  if (vocab_data_bytes == -1) goto fail;
  MALLOC(tok->vocab_data, vocab_data_bytes,
         "model.decoder.tokenizer.vocab_data", goto fail;);
  MALLOC(
      tok->vocab, tok->vocab_size, "model.decoder.tokenizer.vocab", goto fail;);
  MALLOC(tok->vocab_sorted, tok->vocab_size,
         "model.decoder.tokenizer.vocab_sorted", goto fail;);
  for (int i = 0; i < tok->vocab_size; i++)
  {
    char name[64];
    snprintf(name, sizeof(name), "model.decoder.tokenizer.vocab_data.%d", i);
    char *str     = READ_STR(fp, tok->vocab_data, &offset, name, goto fail;);
    tok->vocab[i] = str;
    tok->vocab_sorted[i].idx = i;
    tok->vocab_sorted[i].val = str;
  }
  qsort(tok->vocab_sorted, tok->vocab_size, sizeof(tok->vocab_sorted[0]),
      cmp_token);

  // Special tokens
  tok->bos = get_token_idx(tok, "<bos>");
  tok->eos = get_token_idx(tok, "<eos>");
  tok->sot = get_token_idx(tok, "<start_of_turn>");
  tok->eot = get_token_idx(tok, "<end_of_turn>");
  tok->soi = get_token_idx(tok, "<start_of_image>");
  tok->eoi = get_token_idx(tok, "<end_of_image>");
  tok->ist = get_token_idx(tok, "<image_soft_token>");

  // Build merges
  tok->n_merges =
      (int)READ_UINT32(fp, "model.decoder.tokenizer.n_merges", goto fail;);
  MALLOC(
      tok->ranks, tok->n_merges, "model.decoder.tokenizer.ranks", goto fail;);
  int merge_bytes = get_strarr_bytes(fp, tok->n_merges * 2);
  if (merge_bytes == -1) goto fail;
  MALLOC(tok->merge_data, merge_bytes, "model.decoder.tokenizer.merge_data",
         goto fail;);

  offset = 0;
  for (int i = 0; i < tok->n_merges; i++)
  {
    char name0[64], name1[64];
    snprintf(
        name0, sizeof(name0), "model.decoder.tokenizer.merge_data.%d.0", i);
    snprintf(
        name1, sizeof(name1), "model.decoder.tokenizer.merge_data.%d.1", i);
    char *str1 = READ_STR(fp, tok->merge_data, &offset, name0, goto fail;);
    char *str2 = READ_STR(fp, tok->merge_data, &offset, name1, goto fail;);
    tok->ranks[i].rank = i;
    tok->ranks[i].str1 = str1;
    tok->ranks[i].str2 = str2;
  }
  qsort(tok->ranks, tok->n_merges, sizeof(tok->ranks[0]), cmp_merge);

  /* The embedding shape is (vocab_size, embed_dim), but it uses per-tensor
   * quantization rather than per-channel like other weights. Gemma uses tied
   * weights, which means the final lm_head shares the same weights with the
   * embedding table, but transposed. So it becomes per-channel quantization in
   * the final lm_head.
   */
  int C = cfg->embed_dim;

  READ_LINEAR(dec->embedding, fp, C, cfg->vocab_size, model->quant,
              "model.decoder.embedding", goto fail;);
  CALLOC(dec->layers, cfg->n_layers, "model.decoder.layers",  // NOLINT
         goto fail;);

  int Cq  = cfg->n_heads * cfg->head_dim;
  int Ckv = cfg->n_kv_heads * cfg->head_dim;

  // Read all the layers
  for (int l = 0; l < cfg->n_layers; l++)
  {
    char layer_name[64];
    snprintf(layer_name, sizeof(layer_name), "model.decoder.layers.%d", l);
    TextDecoderLayer *layer = NULL;
    CALLOC(layer, 1, layer_name, goto fail;);

    char wq_name[64], wk_name[64], wv_name[64], wo_name[64];
    snprintf(wq_name, sizeof(wq_name), "model.decoder.layers.%d.wq", l);
    snprintf(wk_name, sizeof(wk_name), "model.decoder.layers.%d.wk", l);
    snprintf(wv_name, sizeof(wv_name), "model.decoder.layers.%d.wv", l);
    snprintf(wo_name, sizeof(wo_name), "model.decoder.layers.%d.wo", l);

    // Attention weights
    READ_LINEAR(layer->wq, fp, C, Cq, model->quant, wq_name, goto fail;);
    READ_LINEAR(layer->wk, fp, C, Ckv, model->quant, wk_name, goto fail;);
    READ_LINEAR(layer->wv, fp, C, Ckv, model->quant, wv_name, goto fail;);
    READ_LINEAR(layer->wo, fp, Cq, C, model->quant, wo_name, goto fail;);

    if (cfg->qk_norm)
    {
      char nq_name[64], nk_name[64];
      snprintf(nq_name, sizeof(nq_name), "model.decoder.layers.%d.nq", l);
      snprintf(nk_name, sizeof(nk_name), "model.decoder.layers.%d.nk", l);
      READ_TENSOR(layer->nq, cfg->head_dim, fp, nq_name, goto fail;);
      READ_TENSOR(layer->nk, cfg->head_dim, fp, nk_name, goto fail;);
    }
    else
    {
      layer->nq = NULL;
      layer->nk = NULL;
    }

    char w1_name[64], w2_name[64], w3_name[64];
    snprintf(w1_name, sizeof(w1_name), "model.decoder.layers.%d.w1", l);
    snprintf(w2_name, sizeof(w2_name), "model.decoder.layers.%d.w2", l);
    snprintf(w3_name, sizeof(w3_name), "model.decoder.layers.%d.w3", l);

    // Feedforward weights
    READ_LINEAR(
        layer->w1, fp, C, cfg->mlp_dim, model->quant, w1_name, goto fail;);
    READ_LINEAR(
        layer->w2, fp, C, cfg->mlp_dim, model->quant, w2_name, goto fail;);
    READ_LINEAR(
        layer->w3, fp, cfg->mlp_dim, C, model->quant, w3_name, goto fail;);

    char n1_name[64], n2_name[64];
    snprintf(n1_name, sizeof(n1_name), "model.decoder.layers.%d.n1", l);
    snprintf(n2_name, sizeof(n2_name), "model.decoder.layers.%d.n2", l);

    // RMSNorm weights
    READ_TENSOR(layer->n1, C, fp, n1_name, goto fail;);
    READ_TENSOR(layer->n2, C, fp, n2_name, goto fail;);

    if (cfg->pre_mlp_norm)
    {
      char n3_name[64];
      snprintf(n3_name, sizeof(n3_name), "model.decoder.layers.%d.n3", l);
      READ_TENSOR(layer->n3, C, fp, n3_name, goto fail;);
    }
    else
    {
      layer->n3 = NULL;
    }
    if (cfg->pst_mlp_norm)
    {
      char n4_name[64];
      snprintf(n4_name, sizeof(n4_name), "model.decoder.layers.%d.n4", l);
      READ_TENSOR(layer->n4, C, fp, n4_name, goto fail;);
    }
    else
    {
      layer->n4 = NULL;
    }
    dec->layers[l] = layer;
  }
  READ_TENSOR(dec->final_norm, C, fp, "model.decoder.final_norm", goto fail;);

  if (use_mm)
  {
    int P  = vcfg->patch_size;
    int VC = vcfg->hidden_dim;

    READ_TENSOR(enc->patch_emb, VC * 3 * P * P, fp, "model.encoder.patch_emb",
                goto fail;);
    READ_TENSOR(
        enc->patch_emb_b, VC, fp, "model.encoder.patch_emb_b", goto fail;);
    int n_patches = vcfg->image_size / P;
    n_patches *= n_patches;
    // Same as here, the real shape is (n_patches, VC)
    READ_LINEAR(enc->pos_embedding, fp, VC, n_patches, model->quant,
                "model.encoder.pos_embedding", goto fail;);
    CALLOC(enc->layers, vcfg->n_layers, "model.encoder.layers",  // NOLINT
           goto fail;);

    // Read all the layers of ViT
    for (int l = 0; l < vcfg->n_layers; l++)
    {
      char layer_name[64];
      snprintf(layer_name, sizeof(layer_name), "model.encoder.layers.%d", l);
      VisionEncoderLayer *layer = NULL;
      CALLOC(layer, 1, layer_name, goto fail;);

      // First layernorm
      char n1_name[64], n1b_name[64];
      snprintf(n1_name, sizeof(n1_name), "model.encoder.layers.%d.n1", l);
      snprintf(n1b_name, sizeof(n1b_name), "model.encoder.layers.%d.n1_b", l);
      READ_TENSOR(layer->n1, VC, fp, n1_name, goto fail;);
      READ_TENSOR(layer->n1_b, VC, fp, n1b_name, goto fail;);

      // Attention weights
      char wq_name[64], wk_name[64], wv_name[64], wo_name[64];
      snprintf(wq_name, sizeof(wq_name), "model.encoder.layers.%d.wq", l);
      snprintf(wk_name, sizeof(wk_name), "model.encoder.layers.%d.wk", l);
      snprintf(wv_name, sizeof(wv_name), "model.encoder.layers.%d.wv", l);
      snprintf(wo_name, sizeof(wo_name), "model.encoder.layers.%d.wo", l);

      READ_LINEAR(layer->wq, fp, VC, VC, model->quant, wq_name, goto fail;);
      READ_LINEAR(layer->wk, fp, VC, VC, model->quant, wk_name, goto fail;);
      READ_LINEAR(layer->wv, fp, VC, VC, model->quant, wv_name, goto fail;);
      READ_LINEAR(layer->wo, fp, VC, VC, model->quant, wo_name, goto fail;);

      // Attention biases
      char bq_name[64], bk_name[64], bv_name[64], bo_name[64];
      snprintf(bq_name, sizeof(bq_name), "model.encoder.layers.%d.bq", l);
      snprintf(bk_name, sizeof(bk_name), "model.encoder.layers.%d.bk", l);
      snprintf(bv_name, sizeof(bv_name), "model.encoder.layers.%d.bv", l);
      snprintf(bo_name, sizeof(bo_name), "model.encoder.layers.%d.bo", l);

      READ_TENSOR(layer->bq, VC, fp, bq_name, goto fail;);
      READ_TENSOR(layer->bk, VC, fp, bk_name, goto fail;);
      READ_TENSOR(layer->bv, VC, fp, bv_name, goto fail;);
      READ_TENSOR(layer->bo, VC, fp, bo_name, goto fail;);

      // Second layernorm
      char n2_name[64], n2b_name[64];
      snprintf(n2_name, sizeof(n2_name), "model.encoder.layers.%d.n2", l);
      snprintf(n2b_name, sizeof(n2b_name), "model.encoder.layers.%d.n2_b", l);

      READ_TENSOR(layer->n2, VC, fp, n2_name, goto fail;);
      READ_TENSOR(layer->n2_b, VC, fp, n2b_name, goto fail;);

      // Feedforward weights
      char w1_name[64], w2_name[64];
      snprintf(w1_name, sizeof(w1_name), "model.encoder.layers.%d.w1", l);
      snprintf(w2_name, sizeof(w2_name), "model.encoder.layers.%d.w2", l);
      READ_LINEAR(
          layer->w1, fp, VC, vcfg->mlp_dim, model->quant, w1_name, goto fail;);
      READ_LINEAR(
          layer->w2, fp, vcfg->mlp_dim, VC, model->quant, w2_name, goto fail;);

      // Feedforward biases
      char b1_name[64], b2_name[64];
      snprintf(b1_name, sizeof(b1_name), "model.encoder.layers.%d.b1", l);
      snprintf(b2_name, sizeof(b2_name), "model.encoder.layers.%d.b2", l);
      READ_TENSOR(layer->b1, vcfg->mlp_dim, fp, b1_name, goto fail;);
      READ_TENSOR(layer->b2, VC, fp, b2_name, goto fail;);

      enc->layers[l] = layer;
    }

    // Post layernorm
    READ_TENSOR(enc->post_norm, VC, fp, "model.encoder.post_norm", goto fail;);
    READ_TENSOR(
        enc->post_norm_b, VC, fp, "model.encoder.post_norm_b", goto fail;);

    // Soft embedding RMSNorm
    READ_TENSOR(enc->norm, VC, fp, "model.encoder.norm", goto fail;);
    // Final projection
    READ_LINEAR(
        enc->proj, fp, VC, C, model->quant, "model.encoder.proj", goto fail;);
  }

  fclose(fp);
  return model;

fail:
  if (fp != NULL) fclose(fp);
  free(att_layers_buf);
  if (model != NULL)
  {
    free_gemma_model(model);
  }
  else
  {
    free_text_decoder(dec);
    free_vision_encoder(enc);
    free_text_config(cfg);
    free_tokenizer(tok);
  }
  return NULL;
}

/* ------------------------------------------------------------------ */
/* Math primitives                                                    */
/* ------------------------------------------------------------------ */

// Make sure the clamping is not optimized by compilers
#ifdef __GNUC__
__attribute__((optimize("no-fast-math")))
#endif
#ifdef __clang__
#  pragma float_control(precise, on, push)
#endif
/* */
static inline floatx
clamp_fpx(floatx v)
{
  return (floatx)fminf(FLOATX_MAX, fmaxf((float)-FLOATX_MAX, (float)v));
}
#ifdef __clang__
#  pragma float_control(pop)
#endif

/* Gemma-style RMSNorm: (x * rsqrt(mean(x²) + eps)) * (weight + 1) */
static void
rmsnorm(floatx   *dst,
    const floatx *src,
    const floatx *weight,

    int   dim,
    float eps)
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
layernorm(floatx *dst,
    const floatx *src,
    const floatx *weight,
    const floatx *bias,

    int   dim,
    float eps)
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
quantize_acts(int8_t *restrict dst,
    floatx *restrict           dst_scales,
    const floatx *restrict     src,
    int                        src_stride,

    int m,
    int n,

    bool omp)
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
gemv_fpx_row(floatx *restrict dst,
    const floatx *restrict    vec,
    const floatx *restrict    mat,

    int n,
    int i)
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
gemv_fpx(floatx *restrict  dst,
    const floatx *restrict mat,
    const floatx *restrict vec,

    int m,
    int n,

    bool omp)
{
  if (omp)
  {
    #pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
      dst[i] = gemv_fpx_row(dst, vec, mat, n, i);
    }
    return;
  }
  for (int i = 0; i < m; i++)
  {
    dst[i] = gemv_fpx_row(dst, vec, mat, n, i);
  }
}

/* */
static inline floatx
gemv_int8_row(const int8_t *restrict vec,
    floatx                           vec_scale,
    const int8_t *restrict           mat,
    const floatx *restrict           mat_scales,

    int n,
    int i)
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
gemv_int8(floatx *restrict dst,
    const int8_t *restrict mat,
    const floatx *restrict mat_scales,
    const int8_t *restrict vec,
    floatx                 vec_scale,

    int m,
    int n,

    bool omp)
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
gemv_fpx_nn(floatx *restrict dst,
    const floatx *restrict   vec,
    const floatx *restrict   mat,
    int                      mat_stride,

    int n,
    int m)
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

/* Compute one MRxNR tile of dst = src @ mat.T, dot-product form.
 * `mat`/`src` rows are read with their natural strides; the tile itself
 * is dense (GEMM_NT_MR x GEMM_NT_NR), never partial. */
static inline void
gemm_fpx_kernel_4x4(floatx *restrict dst,
    int                              dst_stride,
    const floatx *restrict           mat,
    int                              mat_stride,
    const floatx *restrict           src,
    int                              src_stride,

    int k)
{
  float acc[GEMM_NT_MR][GEMM_NT_NR] = {{0}};

  const floatx *s0 = src + 0 * src_stride, *s1 = src + 1 * src_stride;
  const floatx *s2 = src + 2 * src_stride, *s3 = src + 3 * src_stride;
  const floatx *w0 = mat + 0 * mat_stride, *w1 = mat + 1 * mat_stride;
  const floatx *w2 = mat + 2 * mat_stride, *w3 = mat + 3 * mat_stride;

  for (int l = 0; l < k; l++)
  {
    float a0 = (float)s0[l], a1 = (float)s1[l];
    float a2 = (float)s2[l], a3 = (float)s3[l];
    float b0 = (float)w0[l], b1 = (float)w1[l];
    float b2 = (float)w2[l], b3 = (float)w3[l];

    acc[0][0] += a0 * b0;
    acc[0][1] += a0 * b1;
    acc[0][2] += a0 * b2;
    acc[0][3] += a0 * b3;
    acc[1][0] += a1 * b0;
    acc[1][1] += a1 * b1;
    acc[1][2] += a1 * b2;
    acc[1][3] += a1 * b3;
    acc[2][0] += a2 * b0;
    acc[2][1] += a2 * b1;
    acc[2][2] += a2 * b2;
    acc[2][3] += a2 * b3;
    acc[3][0] += a3 * b0;
    acc[3][1] += a3 * b1;
    acc[3][2] += a3 * b2;
    acc[3][3] += a3 * b3;
  }

  for (int ii = 0; ii < GEMM_NT_MR; ii++)
    for (int jj = 0; jj < GEMM_NT_NR; jj++)
      dst[ii * dst_stride + jj] = (floatx)acc[ii][jj];
}

/* Scalar fallback for the m%4 / n%4 remainder tiles (and for m or n < 4
 * outright, e.g. tiny prefill chunks). Same math as the original loop. */
static inline void
gemm_fpx_scalar(floatx *restrict dst,
    int                          dst_stride,
    const floatx *restrict       mat,
    int                          mat_stride,
    const floatx *restrict       src,
    int                          src_stride,

    int m,
    int n,
    int k)
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

/* fpx matrix-matrix multiply (NT)
 * fpx src (m, k) @ fpx mat.T (k, n) = fpx dst (m, n) */
static void
gemm_fpx(floatx *restrict  dst,
    int                    dst_stride,
    const floatx *restrict mat,
    int                    mat_stride,
    const floatx *restrict src,
    int                    src_stride,

    int m,
    int n,
    int k,

    bool omp)
{
  if (dst_stride == 0) dst_stride = n;
  if (mat_stride == 0) mat_stride = k;
  if (src_stride == 0) src_stride = k;

  int m_full = (m / GEMM_NT_MR) * GEMM_NT_MR;
  int n_full = (n / GEMM_NT_NR) * GEMM_NT_NR;

  // Loop order is jb (column panel of `mat`) outer, ib (row block of
  // `src`) inner: every 4x4 tile computed while jb is fixed shares the
  // same 4 rows of `mat`, which is what actually needs to stay resident.
  if (omp)
  {
    #pragma omp parallel for collapse(2)
    for (int jb = 0; jb < n_full; jb += GEMM_NT_NR)
      for (int ib = 0; ib < m_full; ib += GEMM_NT_MR)
      {
        gemm_fpx_kernel_4x4(dst + ib * dst_stride + jb, dst_stride,
            mat + jb * mat_stride, mat_stride, src + ib * src_stride,
            src_stride, k);
      }
  }
  else
  {
    for (int jb = 0; jb < n_full; jb += GEMM_NT_NR)
      for (int ib = 0; ib < m_full; ib += GEMM_NT_MR)
      {
        gemm_fpx_kernel_4x4(dst + ib * dst_stride + jb, dst_stride,
            mat + jb * mat_stride, mat_stride, src + ib * src_stride,
            src_stride, k);
      }
  }

  // Remainder: leftover rows (full width) + leftover columns (remaining
  // height only, so the (m_full:m, n_full:n) corner isn't done twice)
  if (m_full < m)
  {
    gemm_fpx_scalar(dst + m_full * dst_stride, dst_stride, mat, mat_stride,
        src + m_full * src_stride, src_stride, m - m_full, n, k);
  }
  if (n_full < n)
  {
    gemm_fpx_scalar(dst + n_full, dst_stride, mat + n_full * mat_stride,
        mat_stride, src, src_stride, m_full, n - n_full, k);
  }
}

/* Compute an `mr`x`n` (mr <= GEMM_NN_MR) block of dst = src @ mat.
 * `mat` is converted from floatx -> float once per `l` (into `row`) and
 * then reused across all `mr` accumulator rows, instead of being
 * re-read/re-converted once per row like the naive version. */
static inline void
gemm_fpx_nn_kernel(floatx *restrict dst,
    int                             dst_stride,
    const floatx *restrict          mat,
    int                             mat_stride,
    const floatx *restrict          src,
    int                             src_stride,
    int                             mr,

    int n,
    int k)
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
gemm_fpx_nn(floatx *restrict dst,
    int                      dst_stride,
    const floatx *restrict   mat,
    int                      mat_stride,
    const floatx *restrict   src,
    int                      src_stride,

    int m,
    int n,
    int k,

    bool omp)
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
      gemm_fpx_nn_kernel(dst + ib * dst_stride, dst_stride, mat, mat_stride,
          src + ib * src_stride, src_stride, GEMM_NN_MR, n, k);
    }
  }
  else
  {
    for (int ib = 0; ib < m_full; ib += GEMM_NN_MR)
    {
      gemm_fpx_nn_kernel(dst + ib * dst_stride, dst_stride, mat, mat_stride,
          src + ib * src_stride, src_stride, GEMM_NN_MR, n, k);
    }
  }

  // Remainder: leftover rows (m % GEMM_NN_MR), still full width
  if (m_full < m)
  {
    gemm_fpx_nn_kernel(dst + m_full * dst_stride, dst_stride, mat, mat_stride,
        src + m_full * src_stride, src_stride, m - m_full, n, k);
  }
}

static inline void
gemm_int8_kernel_4x4(floatx *restrict dst,
    int                               dst_stride,
    const int8_t *restrict            mat,
    int                               mat_stride,
    const floatx *restrict            mat_scales,
    const int8_t *restrict            src,
    int                               src_stride,
    const floatx *restrict            src_scales,

    int k)
{
  int32_t acc[GEMM_I8_MR][GEMM_I8_NR] = {{0}};

  const int8_t *s0 = src + 0 * src_stride, *s1 = src + 1 * src_stride;
  const int8_t *s2 = src + 2 * src_stride, *s3 = src + 3 * src_stride;
  const int8_t *w0 = mat + 0 * mat_stride, *w1 = mat + 1 * mat_stride;
  const int8_t *w2 = mat + 2 * mat_stride, *w3 = mat + 3 * mat_stride;

  for (int l = 0; l < k; l++)
  {
    int32_t a0 = s0[l], a1 = s1[l], a2 = s2[l], a3 = s3[l];
    int32_t b0 = w0[l], b1 = w1[l], b2 = w2[l], b3 = w3[l];

    acc[0][0] += a0 * b0;
    acc[0][1] += a0 * b1;
    acc[0][2] += a0 * b2;
    acc[0][3] += a0 * b3;
    acc[1][0] += a1 * b0;
    acc[1][1] += a1 * b1;
    acc[1][2] += a1 * b2;
    acc[1][3] += a1 * b3;
    acc[2][0] += a2 * b0;
    acc[2][1] += a2 * b1;
    acc[2][2] += a2 * b2;
    acc[2][3] += a2 * b3;
    acc[3][0] += a3 * b0;
    acc[3][1] += a3 * b1;
    acc[3][2] += a3 * b2;
    acc[3][3] += a3 * b3;
  }

  for (int ii = 0; ii < GEMM_I8_MR; ii++)
  {
    float fscale = (float)src_scales[ii];
    for (int jj = 0; jj < GEMM_I8_NR; jj++)
    {
      float val = (float)acc[ii][jj] * fscale * (float)mat_scales[jj];
      dst[ii * dst_stride + jj] = (floatx)val;
    }
  }
}

/* Scalar fallback for the m%4 / n%4 remainder tiles. */
static inline void
gemm_int8_scalar(floatx *restrict dst,
    int                           dst_stride,
    const int8_t *restrict        mat,
    int                           mat_stride,
    const floatx *restrict        mat_scales,
    const int8_t *restrict        src,
    int                           src_stride,
    const floatx *restrict        src_scales,

    int m,
    int n,
    int k)
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
gemm_int8(floatx *restrict dst,
    int                    dst_stride,
    const int8_t *restrict mat,
    int                    mat_stride,
    const floatx *restrict mat_scales,
    const int8_t *restrict src,
    int                    src_stride,
    const floatx *restrict src_scales,

    int m,
    int n,
    int k,

    bool omp)
{
  if (dst_stride == 0) dst_stride = n;
  if (mat_stride == 0) mat_stride = k;
  if (src_stride == 0) src_stride = k;

  int m_full = (m / GEMM_I8_MR) * GEMM_I8_MR;
  int n_full = (n / GEMM_I8_NR) * GEMM_I8_NR;

  if (omp)
  {
    #pragma omp parallel for collapse(2)
    for (int jb = 0; jb < n_full; jb += GEMM_I8_NR)
      for (int ib = 0; ib < m_full; ib += GEMM_I8_MR)
      {
        gemm_int8_kernel_4x4(dst + ib * dst_stride + jb, dst_stride,
            mat + jb * mat_stride, mat_stride, mat_scales + jb,
            src + ib * src_stride, src_stride, src_scales + ib, k);
      }
  }
  else
  {
    for (int jb = 0; jb < n_full; jb += GEMM_I8_NR)
      for (int ib = 0; ib < m_full; ib += GEMM_I8_MR)
      {
        gemm_int8_kernel_4x4(dst + ib * dst_stride + jb, dst_stride,
            mat + jb * mat_stride, mat_stride, mat_scales + jb,
            src + ib * src_stride, src_stride, src_scales + ib, k);
      }
  }

  // Remainder: leftover rows (full width) + leftover columns (remaining
  // height only, so the (m_full:m, n_full:n) corner isn't done twice)
  if (m_full < m)
  {
    gemm_int8_scalar(dst + m_full * dst_stride, dst_stride, mat, mat_stride,
        mat_scales, src + m_full * src_stride, src_stride, src_scales + m_full,
        m - m_full, n, k);
  }
  if (n_full < n)
  {
    gemm_int8_scalar(dst + n_full, dst_stride, mat + n_full * mat_stride,
        mat_stride, mat_scales + n_full, src, src_stride, src_scales, m_full,
        n - n_full, k);
  }
}

/* */
static void
softmax(floatx *dst, const floatx *src, int dim)
{
  floatx max = -(floatx)INFINITY;
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

/* ------------------------------------------------------------------ */
/* Image preprocessing                                                */
/* ------------------------------------------------------------------ */

/* */
floatx *
prepare_image(const char *path, int image_size)
{
  // Load the image
  int            rows, cols, channels;
  unsigned char *img = stbi_load(path, &cols, &rows, &channels, 3);
  if (img == NULL)
  {
    return NULL;
  }

  // Resize the image
  unsigned char *rsz = stbir_resize_uint8_linear(
      img, cols, rows, 0, NULL, image_size, image_size, 0, STBIR_RGB);
  stbi_image_free(img);
  if (rsz == NULL)
  {
    return NULL;
  }

  floatx *out;
  MALLOC(out, image_size * image_size * 3, path, {
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

/* ------------------------------------------------------------------ */
/* Forward functions                                                  */
/* ------------------------------------------------------------------ */

/* Vision forward pass (SigLIP)
 * img: (img_sz, img_sz, 3) */
int
forward_vision(VisionEncoder *enc,

    TextConfig   *cfg,
    TextBuffer   *buf,
    VisionBuffer *vbuf,
    const floatx *img,
    bool          quant)
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

  if (g_interrupted) return 1;

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

  if (g_interrupted) return 1;

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
    if (g_interrupted) return 1;

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
      gemm_int8(vbuf->xq, 0, layer->wq->i8.q, 0, layer->wq->i8.scales,
          vbuf->x_i8, 0, vbuf->x_scales, N, C, C, true);
      gemm_int8(vbuf->xk, 0, layer->wk->i8.q, 0, layer->wk->i8.scales,
          vbuf->x_i8, 0, vbuf->x_scales, N, C, C, true);
      gemm_int8(vbuf->xv, 0, layer->wv->i8.q, 0, layer->wv->i8.scales,
          vbuf->x_i8, 0, vbuf->x_scales, N, C, C, true);
    }
    if (g_interrupted) return 1;

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
    if (g_interrupted) return 1;

    // Attention
    memset(vbuf->att_out, 0, N * C * sizeof(floatx));
    float scale = 1.0f / sqrtf((float)CH);

    #pragma omp parallel for
    for (int h = 0; h < vcfg->n_heads; h++)
    {
      floatx *scores = vbuf->scores + h * N * N;  // (N, N) for this head

      // scores = Q @ K^T * scale
      gemm_fpx(scores, /*dst_stride=*/N, vbuf->xk + h * CH, /*mat_stride=*/C,
          vbuf->xq + h * CH, /*src_stride=*/C, N, N, CH, false);

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
      gemm_fpx_nn(vbuf->att_out + h * CH, /*dst_stride=*/C, vbuf->xv + h * CH,
          /*mat_stride=*/C, scores, /*src_stride=*/N, N, CH, N, false);
    }
    if (g_interrupted) return 1;

    // Output projection
    if (!quant)
    {
      gemm_fpx(vbuf->x, 0, layer->wo->fpx, 0, vbuf->att_out, 0, N, C, C, true);
    }
    else
    {
      quantize_acts(vbuf->x_i8, vbuf->x_scales, vbuf->att_out, 0, N, C, true);
      gemm_int8(vbuf->x, 0, layer->wo->i8.q, 0, layer->wo->i8.scales,
          vbuf->x_i8, 0, vbuf->x_scales, N, C, C, true);
    }
    if (g_interrupted) return 1;

    // Add output bias
    #pragma omp simd collapse(2)
    for (int i = 0; i < N; i++)
      for (int j = 0; j < C; j++)
      {
        vbuf->x[i * C + j] += layer->bo[j];
      }
    if (g_interrupted) return 1;

    // Residual connection
    for (int i = 0; i < N * C; i++)
    {
      vbuf->x[i] = clamp_fpx(vbuf->x[i] + vbuf->resid[i]);
    }
    if (g_interrupted) return 1;

    memcpy(vbuf->resid, vbuf->x, N * C * sizeof(floatx));
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
      floatx *row = vbuf->x + i * C;
      layernorm(row, row, layer->n2, layer->n2_b, C, vcfg->eps);
    }
    if (g_interrupted) return 1;

    // x @ fc1 = mlp_hidden
    if (!quant)
    {
      gemm_fpx(
          vbuf->mlp_hidden, 0, layer->w1->fpx, 0, vbuf->x, 0, N, CM, C, true);
    }
    else
    {
      quantize_acts(vbuf->x_i8, vbuf->x_scales, vbuf->x, 0, N, C, true);
      gemm_int8(vbuf->mlp_hidden, 0, layer->w1->i8.q, 0, layer->w1->i8.scales,
          vbuf->x_i8, 0, vbuf->x_scales, N, CM, C, true);
    }
    if (g_interrupted) return 1;

    #pragma omp simd collapse(2)
    for (int i = 0; i < N; i++)
      for (int j = 0; j < CM; j++)
      {
        // Apply fc1 biases
        float val = (float)vbuf->mlp_hidden[i * CM + j] + (float)layer->b1[j];
        // GELU tanh approximation
        float c = 0.79788456080287f;
        val     = 0.5f * val *
              (1.0f + tanhf(c * (val + 0.044715f * val * val * val)));
        vbuf->mlp_hidden[i * CM + j] = (floatx)val;
      }
    if (g_interrupted) return 1;

    // mlp_hidden @ fc2 = x
    if (!quant)
    {
      gemm_fpx(
          vbuf->x, 0, layer->w2->fpx, 0, vbuf->mlp_hidden, 0, N, C, CM, true);
    }
    else
    {
      quantize_acts(
          vbuf->mlp_i8, vbuf->mlp_scales, vbuf->mlp_hidden, 0, N, CM, true);
      gemm_int8(vbuf->x, 0, layer->w2->i8.q, 0, layer->w2->i8.scales,
          vbuf->mlp_i8, 0, vbuf->mlp_scales, N, C, CM, true);
    }
    if (g_interrupted) return 1;

    // x += b2
    #pragma omp simd collapse(2)
    for (int i = 0; i < N; i++)
      for (int j = 0; j < C; j++)
      {
        vbuf->x[i * C + j] += layer->b2[j];
      }

    if (g_interrupted) return 1;

    // Residual connection
    #pragma omp parallel for
    for (int i = 0; i < N * C; i++)
    {
      vbuf->x[i] = clamp_fpx(vbuf->x[i] + vbuf->resid[i]);
    }
    if (g_interrupted) return 1;
  }

  // Post-norm + average pooling down to image_toks tokens
  #pragma omp parallel for
  for (int i = 0; i < N; i++)
  {
    floatx *row = vbuf->x + i * C;
    layernorm(row, row, enc->post_norm, enc->post_norm_b, C, vcfg->eps);
  }
  if (g_interrupted) return 1;

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

  if (g_interrupted) return 1;
  // buf->x now becomes (tpi, C)

  // RMSNorm
  #pragma omp parallel for
  for (int i = 0; i < tpi; i++)
  {
    floatx *row = vbuf->x + i * C;
    rmsnorm(row, row, enc->norm, C, vcfg->eps);
  }
  if (g_interrupted) return 1;

  // Final projection into language model embedding space
  // x (tpi, embed_dim) = x (tpi, C) @ proj (C, embed_dim)
  if (!quant)
  {
    gemm_fpx(
        buf->x, 0, enc->proj->fpx, 0, vbuf->x, 0, tpi, cfg->embed_dim, C, true);
  }
  else
  {
    quantize_acts(vbuf->x_i8, vbuf->x_scales, vbuf->x, 0, tpi, C, true);
    gemm_int8(buf->x, 0, enc->proj->i8.q, 0, enc->proj->i8.scales, vbuf->x_i8,
        0, vbuf->x_scales, tpi, cfg->embed_dim, C, true);
  }
  if (g_interrupted) return 1;

  return 0;
}

/* Language model forward (one token) */
int
forward_text_decode(TextDecoder *dec, TextBuffer *buf, int pos, bool quant)
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
  if (g_interrupted) return 1;

  // Forward all the layers
  for (int l = 0; l < cfg->n_layers; l++)
  {
    TextDecoderLayer *layer = dec->layers[l];

    memcpy(buf->resid, buf->x, C * sizeof(*buf->x));

    rmsnorm(buf->x, buf->x, layer->n1, C, cfg->eps);
    if (g_interrupted) return 1;

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
      gemv_int8(buf->xq, layer->wq->i8.q, layer->wq->i8.scales, buf->x_i8,
          x_scale, Cq, C, true);
      gemv_int8(buf->xk, layer->wk->i8.q, layer->wk->i8.scales, buf->x_i8,
          x_scale, Ckv, C, true);
      gemv_int8(buf->xv, layer->wv->i8.q, layer->wv->i8.scales, buf->x_i8,
          x_scale, Ckv, C, true);
    }
    if (g_interrupted) return 1;

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
    if (g_interrupted) return 1;

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
    if (g_interrupted) return 1;

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
    if (g_interrupted) return 1;

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
    if (g_interrupted) return 1;

    // Output projection maps xo back to x
    // x (C,) = xo (CH,) @ wo.T (CH, C)
    if (!quant)
    {
      gemv_fpx(buf->x, layer->wo->fpx, buf->xo, C, Cq, true);
    }
    else
    {
      floatx xo_scale = quantize_act(buf->xo_i8, buf->xo, Cq);
      gemv_int8(buf->x, layer->wo->i8.q, layer->wo->i8.scales, buf->xo_i8,
          xo_scale, C, Cq, true);
    }
    if (g_interrupted) return 1;

    rmsnorm(buf->x, buf->x, layer->n2, C, cfg->eps);
    if (g_interrupted) return 1;

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
    if (g_interrupted) return 1;

    memcpy(buf->resid, buf->x, C * sizeof(*buf->x));

    if (cfg->pre_mlp_norm)
    {
      rmsnorm(buf->x, buf->x, layer->n3, C, cfg->eps);
    }
    if (g_interrupted) return 1;

    // MLP feedforward layer (SwiGLU-style)
    if (!quant)
    {
      gemv_fpx(buf->xu, layer->w1->fpx, buf->x, cfg->mlp_dim, C, true);
      gemv_fpx(buf->xg, layer->w2->fpx, buf->x, cfg->mlp_dim, C, true);
    }
    else
    {
      floatx x_scale = quantize_act(buf->x_i8, buf->x, C);
      gemv_int8(buf->xg, layer->w2->i8.q, layer->w2->i8.scales, buf->x_i8,
          x_scale, cfg->mlp_dim, C, true);
      gemv_int8(buf->xu, layer->w1->i8.q, layer->w1->i8.scales, buf->x_i8,
          x_scale, cfg->mlp_dim, C, true);
    }
    if (g_interrupted) return 1;

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
    if (g_interrupted) return 1;

    if (!quant)
    {
      gemv_fpx(buf->x, layer->w3->fpx, buf->xg, C, cfg->mlp_dim, true);
    }
    else
    {
      floatx xscale = quantize_act(buf->xg_i8, buf->xg, cfg->mlp_dim);
      gemv_int8(buf->x, layer->w3->i8.q, layer->w3->i8.scales, buf->xg_i8,
          xscale, C, cfg->mlp_dim, true);
    }
    if (g_interrupted) return 1;

    if (cfg->pst_mlp_norm)
    {
      rmsnorm(buf->x, buf->x, layer->n4, C, cfg->eps);
    }
    if (g_interrupted) return 1;

    // Second residual
    x     = buf->x;
    resid = buf->resid;
    for (int d = 0; d < C; d++)
    {
      buf->x[d] = clamp_fpx(x[d] + resid[d]);
    }
    if (g_interrupted) return 1;
  }

  // Final RMSNorm
  rmsnorm(buf->x, buf->x, dec->final_norm, C, cfg->eps);
  if (g_interrupted) return 1;

  // Compute logits (tied embedding)
  if (!quant)
  {
    gemv_fpx(
        buf->logits, dec->embedding->fpx, buf->x, cfg->vocab_size, C, true);
  }
  else
  {
    floatx xscale = quantize_act(buf->x_i8, buf->x, C);
    gemv_int8(buf->logits, dec->embedding->i8.q, dec->embedding->i8.scales,
        buf->x_i8, xscale, cfg->vocab_size, C, true);
  }
  if (g_interrupted) return 1;

  // Optional logit softcapping
  if (cfg->logit_softcap != 0.0f)
  {
    for (int d = 0; d < cfg->vocab_size; d++)
    {
      float val      = (float)buf->logits[d] / cfg->logit_softcap;
      buf->logits[d] = (floatx)(tanhf(val) * cfg->logit_softcap);
    }
  }
  if (g_interrupted) return 1;
  return 0;
}

/* Language model forward (a chunk of tokens) */
static int
forward_text_chunk(TextDecoder *dec,
    TextBuffer                 *buf,

    int  spos,
    int  T,
    bool mask,
    bool quant,
    bool compute_logits)
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
  if (g_interrupted) return 1;

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
    if (g_interrupted) return 1;

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
        gemm_fpx(xq_head, 0, layer->wq->fpx + h * CH * C, 0, buf->x, 0, T, CH,
            C, false);
      }
      else
      {
        gemm_int8(xq_head, 0, layer->wq->i8.q + h * CH * C, 0,
            layer->wq->i8.scales + h * CH, buf->x_i8, 0, buf->x_scales, T, CH,
            C, false);
      }
      if (h >= NH_kv) continue;

      floatx *xk_head = buf->xk + h * T * CH;
      floatx *xv_head = buf->xv + h * T * CH;

      if (!quant)
      {
        // (T, NH_kv, CH)
        gemm_fpx(xk_head, 0, layer->wk->fpx + h * CH * C, 0, buf->x, 0, T, CH,
            C, false);
        gemm_fpx(xv_head, 0, layer->wv->fpx + h * CH * C, 0, buf->x, 0, T, CH,
            C, false);
      }
      else
      {
        gemm_int8(xk_head, 0, layer->wk->i8.q + h * CH * C, 0,
            layer->wk->i8.scales + h * CH, buf->x_i8, 0, buf->x_scales, T, CH,
            C, false);
        gemm_int8(xv_head, 0, layer->wv->i8.q + h * CH * C, 0,
            layer->wv->i8.scales + h * CH, buf->x_i8, 0, buf->x_scales, T, CH,
            C, false);
      }
    }
    if (g_interrupted) return 1;

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
    if (g_interrupted) return 1;

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

    if (g_interrupted) return 1;

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
    if (g_interrupted) return 1;

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
        for (int q_start = 0; q_start < T; q_start += qk_block_size)
        {
          // xq[h, q_start:q_end, :] (q_end - q_start, CH)
          floatx *qb          = buf->xq + h * T * CH + q_start * CH;
          int     q_end       = MIN(q_start + qk_block_size, T);
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
          int spos_b = abs_q_start / qk_block_size * qk_block_size;

          for (int k_start = spos_b; k_start < abs_q_end;
               k_start += qk_block_size)
          {
            // k_cache[h_kv, k_start:k_end, :] (k_end - k_start, CH)
            floatx *kb    = k_cache + h_kv * buf->cache_len * CH + k_start * CH;
            int     k_end = MIN(k_start + qk_block_size, max_k);
            floatx *att_b = att_head + q_start * max_k + k_start;

            // Typically people don't quantize this
            gemm_fpx(att_b, max_k, kb, 0, qb, 0, q_end - q_start,
                k_end - k_start, CH, false);

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

          gemm_fpx_nn(xo_block, /*dst_stride=*/Cq, v_block, /*mat_stride=*/0,
              att_block,
              /*src_stride=*/max_k, q_end - q_start, CH, abs_q_end - spos_b,
              false);
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
          int     pos     = spos + qi;
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
        gemm_fpx_nn(xo_head, /*dst_stride=*/Cq, xv_head, /*mat_stride=*/0,
            att_head, /*src_stride=*/max_k, T, CH, max_k, false);
      }
    }
    if (g_interrupted) return 1;
    // x (T, C) = xo_head (T, CH) @ wo.T (CH, C)
    if (!quant)
    {
      gemm_fpx(buf->x, 0, layer->wo->fpx, 0, buf->xo, Cq, T, C, Cq, true);
    }
    else
    {
      quantize_acts(buf->xo_i8, buf->xo_scales, buf->xo, Cq, T, Cq, true);
      gemm_int8(buf->x, 0, layer->wo->i8.q, 0, layer->wo->i8.scales, buf->xo_i8,
          0, buf->xo_scales, T, C, Cq, true);
    }
    if (g_interrupted) return 1;

    #pragma omp parallel for
    for (int t = 0; t < T; t++)
    {
      floatx *x_row = buf->x + t * C;
      rmsnorm(x_row, x_row, layer->n2, C, cfg->eps);
    }
    if (g_interrupted) return 1;

    floatx *restrict x     = buf->x;
    floatx *restrict resid = buf->resid;
    #pragma omp parallel for
    for (int d = 0; d < T * C; d++)
    {
      // Combine the residual stream
      buf->x[d] = clamp_fpx(x[d] + resid[d]);
    }
    if (g_interrupted) return 1;

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
    if (g_interrupted) return 1;

    // MLP
    if (!quant)
    {
      gemm_fpx(
          buf->xu, 0, layer->w1->fpx, 0, buf->x, 0, T, cfg->mlp_dim, C, true);
      gemm_fpx(
          buf->xg, 0, layer->w2->fpx, 0, buf->x, 0, T, cfg->mlp_dim, C, true);
    }
    else
    {
      quantize_acts(buf->x_i8, buf->x_scales, buf->x, 0, T, C, true);
      gemm_int8(buf->xu, 0, layer->w1->i8.q, 0, layer->w1->i8.scales, buf->x_i8,
          0, buf->x_scales, T, cfg->mlp_dim, C, true);
      gemm_int8(buf->xg, 0, layer->w2->i8.q, 0, layer->w2->i8.scales, buf->x_i8,
          0, buf->x_scales, T, cfg->mlp_dim, C, true);
    }
    if (g_interrupted) return 1;

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
    if (g_interrupted) return 1;

    // Down projection
    if (!quant)
    {
      gemm_fpx(
          buf->x, 0, layer->w3->fpx, 0, buf->xg, 0, T, C, cfg->mlp_dim, true);
    }
    else
    {
      quantize_acts(
          buf->xg_i8, buf->xg_scales, buf->xg, 0, T, cfg->mlp_dim, true);
      gemm_int8(buf->x, 0, layer->w3->i8.q, 0, layer->w3->i8.scales, buf->xg_i8,
          0, buf->xg_scales, T, C, cfg->mlp_dim, true);
    }
    if (g_interrupted) return 1;

    if (cfg->pst_mlp_norm)
    {
      #pragma omp parallel for
      for (int t = 0; t < T; t++)
      {
        floatx *x_row = buf->x + t * C;
        rmsnorm(x_row, x_row, layer->n4, C, cfg->eps);
      }
    }
    if (g_interrupted) return 1;

    x     = buf->x;
    resid = buf->resid;
    #pragma omp parallel for
    for (int d = 0; d < T * C; d++)
    {
      // Second residual
      buf->x[d] = clamp_fpx(x[d] + resid[d]);
    }
    if (g_interrupted) return 1;
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
          buf->logits, dec->embedding->fpx, last_x, cfg->vocab_size, C, true);
    }
    else
    {
      floatx xscale = quantize_act(buf->x_i8, last_x, C);
      gemv_int8(buf->logits, dec->embedding->i8.q, dec->embedding->i8.scales,
          buf->x_i8, xscale, cfg->vocab_size, C, true);
    }
    if (g_interrupted) return 1;

    // Optional logit softcapping
    if (cfg->logit_softcap != 0.0f)
    {
      for (int d = 0; d < cfg->vocab_size; d++)
      {
        float val      = (float)buf->logits[d] / cfg->logit_softcap;
        buf->logits[d] = (floatx)(tanhf(val) * cfg->logit_softcap);
      }
    }
    if (g_interrupted) return 1;
  }
  return 0;
}

/* Language model forward (multiple tokens) */
int
forward_text_prefill(TextDecoder *dec,
    TextBuffer                   *buf,

    int  spos,
    int  T,
    int  chunk_size,
    bool mask,
    bool quant,
    bool compute_logits)
{
  if (T <= 0)
  {
    fprintf(stderr, "\nerror: no tokens to prefill\n");
    return 1;
  }

  if (chunk_size <= 0)
  {
    fprintf(stderr, "\nerror: invalid chunk_size\n");
    return 1;
  }

  for (int off = 0; off < T; off += chunk_size)
  {
    int  cur_len       = MIN(chunk_size, T - off);
    bool is_last_chunk = (off + cur_len == T);

    int rc = forward_text_chunk(dec, buf, spos + off, cur_len, mask, quant,
        is_last_chunk && compute_logits);
    if (rc != 0) return rc;
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
  if (g_interrupted) return 1;
  return forward_text_decode(model->decoder, buf, pos, model->quant);
}

int
forward_gemma_prefill(GemmaModel *model,
    TextBuffer                   *buf,

    int  *tokens,
    int  *pos,
    int   chunk_size,
    bool *rpen_visited,
    bool  compute_logits)
{
  TextDecoder *dec = model->decoder;
  int          T   = 0;
  for (int *t = tokens; *t != EOF; t++)
  {
    if (rpen_visited != NULL)
    {
      rpen_visited[*t] = true;
    }
    T++;
  }

  int C = dec->config->embed_dim;

  // Embedding lookup
  #pragma omp parallel for
  for (int t = 0; t < T; t++)
  {
    int token = tokens[t];
    int off   = t * C;

    for (int d = 0; d < C; d++)
    {
      if (!model->quant)
      {
        buf->x[off + d] = dec->embedding->fpx[token * C + d];  // * 1.0f
      }
      else
      {
        buf->x[off + d] = (floatx)dec->embedding->i8.q[token * C + d] *
                          dec->embedding->i8.scales[token];
      }
    }
  }
  if (g_interrupted) return 1;
  int suc = forward_text_prefill(model->decoder, buf, *pos, T, chunk_size,
      /*mask=*/true, model->quant, compute_logits);

  *pos += T;
  return suc;
}

int
forward_gemma_image(GemmaModel *model,
    TextBuffer                 *buf,
    VisionBuffer               *vbuf,

    const floatx *image,
    int          *pos,
    int           chunk_size,
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

  int suc = forward_text_prefill(
      // Gemma 3 models uses bi-directional attention for vision tokens
      dec, buf, *pos, image_toks, MAX(chunk_size, image_toks), /*mask=*/false,
      model->quant, compute_logits);

  *pos += image_toks;
  return suc;
}

/* ------------------------------------------------------------------ */
/* Sampling helpers                                                   */
/* ------------------------------------------------------------------ */

/* */
static int
argmax(floatx *logits, int vocab_size)
{
  // Pick the index with the max value
  int    max_idx = -1;
  floatx max_val = -(floatx)INFINITY;
  #pragma omp parallel
  {
    int    local_idx = -1;
    floatx local_val = -(floatx)INFINITY;
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
  int r = (int)lo + (int)qs_rand() % (hi - lo + 1);
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
    logits[i] = -(floatx)INFINITY;
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
    int l = 2 * i + 1, r = 2 * i + 2, largest = i;
    if (l < n && arr[l].val > arr[largest].val) largest = l;
    if (r < n && arr[r].val > arr[largest].val) largest = r;
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
apply_topp(floatx *logits,
    floatx        *fpbuf,
    FloatIdx      *logit_indices,

    int   vocab_size,
    int   k,
    float p)
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
    logits[i] = -(floatx)INFINITY;
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

/* ------------------------------------------------------------------ */
/* Generation loop                                                    */
/* ------------------------------------------------------------------ */

typedef enum
{
  INJECT_NONE,
  INJECT_TEXT,
  INJECT_IMAG
} InjectType;

typedef struct
{
  InjectType type;
  bool       prefill_finished;
  bool       quit;
  union
  {
    int    *tokens;
    floatx *image;
  };
} InjectData;

/* */
void
sample(GemmaModel *model,
    TextBuffer    *buf,
    VisionBuffer  *vbuf,
    InjectData     init_data,

    int   seqlen,
    int   chunk_size,
    float temperature,
    int   topk,
    float topp,
    float rpen,
    bool  enable_mm,

    InjectData (*token_callback)(int token, GemmaModel *model, bool enable_mm))
{
  TextConfig *cfg = model->decoder->config;
  int         vs  = cfg->vocab_size;

  // Boolean flags
  bool dosample = temperature != 0 && topk != 1;
  bool use_topk = dosample && topk != 0;
  bool use_topp = dosample && topp < 1.0f;
  bool use_rpen = dosample && rpen > 1.0f;

  double prefill_start   = 0.0f;
  double prefill_end     = 0.0f;
  double prefill_elapsed = 0.0f;
  double gen_start       = 0.0f;
  double gen_end         = 0.0f;
  double gen_elapsed     = 0.0f;

  int     prompt_toks = 0;
  int     gen_toks    = 0;
  int     pos         = 0;
  int     token       = 0;
  floatx *probs       = NULL;

  // Injected data (text chunk / images)
  InjectData injected = init_data;
  // Logit indices for topk & topp
  FloatIdx *logit_indices = NULL;

  // bool array indicating which tokens have already been processed
  // Used in rpen (repetition penalty)
  bool *visited = NULL;
  if (use_rpen)
  {
    CALLOC(visited, vs, "visited", goto end;);
  }

  // Only allocate probs if needed
  if (temperature != 0.0f)
  {
    MALLOC(probs, vs, "probs", goto end;);
  }

  if (use_topk || use_topp)
  {
    MALLOC(logit_indices, vs, "logit_indices", goto end;);
  }

  // Record tok/s
  gen_toks = 0;

  bool prefill_finished = init_data.prefill_finished;

  while (pos < seqlen && !injected.quit)
  {
    if (injected.type == INJECT_NONE)
    {
      gen_start = now_sec();
      // Decode to get the next logits
      if (forward_gemma_decode(model, buf, token, pos) == 1) goto end;
      prefill_finished = true;
      pos++;
      gen_end = now_sec();
      gen_elapsed += gen_end - gen_start;
    }
    else
    {
      int prev_pos     = pos;
      prefill_start    = now_sec();
      prefill_finished = injected.prefill_finished;

      if (injected.type == INJECT_TEXT)
      {
        // Injected a token array (ends with EOF), prefill all the tokens
        if (forward_gemma_prefill(model, buf, injected.tokens, &pos, chunk_size,
                visited, prefill_finished) == 1)
          goto end;
      }
      else if (injected.type == INJECT_IMAG)
      {
        // Injected a image (floatx pixel values)
        if (forward_gemma_image(model, buf, vbuf, injected.image, &pos,
                chunk_size, prefill_finished) == 1)
          goto end;
        free(injected.image);
      }

      prefill_end = now_sec();
      prefill_elapsed += prefill_end - prefill_start;
      prompt_toks += pos - prev_pos;
    }

    if (g_interrupted) break;

    int img_size = (model->encoder && model->encoder->config)
                       ? model->encoder->config->image_size
                       : 0;

    if (prefill_finished)
    {
      gen_start = now_sec();

      // Manipulate the logits & sample the next token
      if (g_interrupted) break;
      if (!dosample)
      {
        // Argmax sampling
        token = argmax(buf->logits, vs);
      }
      else
      {
        // Apply the temperature
        #pragma omp parallel for
        for (int d = 0; d < vs; d++)
        {
          buf->logits[d] /= (floatx)temperature;
        }

        if (use_topk)
        {
          apply_topk(buf->logits, logit_indices, vs, topk);
        }
        if (use_topp)
        {
          apply_topp(buf->logits, probs, logit_indices, vs, topk, topp);
        }
        if (use_rpen)
        {
          apply_rpen(buf->logits, visited, vs, rpen);
        }

        // Softmax to get the probs
        softmax(probs, buf->logits, vs);

        // Multinomial sample from probs
        float r   = (float)((float)rand() / (RAND_MAX + 1.0));
        float sum = 0.0f;

        token = vs - 1;
        for (int d = 0; d < vs; d++)
        {
          sum += (float)probs[d];
          if (r < sum)
          {
            token = d;
            break;
          }
        }
      }
      gen_toks++;
      gen_end = now_sec();
      gen_elapsed += gen_end - gen_start;

      injected = token_callback(token, model, enable_mm);
    }
    else
    {
      // Injection not complete, send EOF to request the next chunk of data
      injected = token_callback(EOF, model, enable_mm);
    }
  }

end:
  // Print prefilling speed
  if (prefill_elapsed > 0.0)
  {
    printf("\nPrompt processed %d tokens in %.2f seconds (%.2f tok/s)\n",
        prompt_toks, prefill_elapsed, prompt_toks / prefill_elapsed);
  }
  else
  {
    printf("\nPrompt processed %d tokens instantly\n", prompt_toks);
  }
  // Print generation speed
  if (gen_elapsed > 0.0)
  {
    printf("Generated %d tokens in %.2f seconds (%.2f tok/s)\n", gen_toks,
        gen_elapsed, gen_toks / gen_elapsed);
  }
  else
  {
    printf("Generated %d tokens instantly\n", gen_toks);
  }

  free(probs);
  if (use_rpen)
  {
    free(visited);
  }
  if (use_topk || use_topp)
  {
    free(logit_indices);
  }
}

/* ------------------------------------------------------------------ */
/* High-level generate / chat                                         */
/* ------------------------------------------------------------------ */

/* */
static InjectData
generate_callback(int token, GemmaModel *model, bool enable_mm)
{
  GemmaTokenizer *tok = model->decoder->tokenizer;

  if (token == tok->eos || token == tok->eot)
  {
    return (InjectData){.type = INJECT_NONE, .quit = true};
  }
  char byte_buf[2];
  printf("%s", decode(tok, token, byte_buf));
  fflush(stdout);
  return (InjectData){.type = INJECT_NONE, .quit = false};
}

/* */
int
generate(GemmaModel *model,
    TextBuffer      *buf,
    VisionBuffer    *vbuf,

    const char *prompt,
    int         seqlen,
    int         chunk_size,
    float       temperature,
    int         topk,
    float       topp,
    float       rpen,
    bool        enable_mm)
{
  GemmaTokenizer *tok = model->decoder->tokenizer;

  printf("%s", prompt);

  int n_tokens    = 1;
  g_tokens_buf[0] = tok->bos;
  encode(tok, prompt, strlen(prompt), g_tokens_buf + 1, &n_tokens);
  if (n_tokens >= g_tokens_buf_len)
  {
    fprintf(stderr, "error: prompt too long\n");
    return 1;
  }
  g_tokens_buf[n_tokens] = EOF;

  sample(model, buf, vbuf,
      (InjectData){.type    = INJECT_TEXT,
          .tokens           = g_tokens_buf,
          .prefill_finished = true,
          .quit             = false},
      seqlen, chunk_size, temperature, topk, topp, rpen, enable_mm,
      generate_callback);

  return 0;
}

/* Build the Gemma chat template */
static InjectData
new_turn(GemmaTokenizer *tok, GemmaModel *model, bool enable_mm, bool bos)
{
  /* Template (from https://ai.google.dev/gemma/docs/core/prompt-structure):
   * <start_of_turn>user\n
   * What is Cramer's Rule?<end_of_turn>\n
   * <start_of_turn>model */

  // clang-format off
#define RETURN_TEXT_INJECT(prefill_finished_val, quit_val) \
  do {                                                     \
    int start_             = token_start;                  \
    token_start            = n_tokens;                     \
    g_tokens_buf[n_tokens] = EOF;                          \
    return (InjectData){.type = INJECT_TEXT,               \
        .tokens               = g_tokens_buf + start_,     \
        .prefill_finished     = (prefill_finished_val),    \
        .quit                 = (quit_val)};               \
  }                                                        \
  while (0)
  // clang-format on

  // A pretty messy finite state machine
  static thread_local bool turn_finished = true;
  static thread_local bool pending_image = false;
  static thread_local bool was_image     = false;
  static thread_local int  prompt_idx    = 0;
  static thread_local int  token_start   = 0;
  static thread_local int  n_tokens      = 0;
  static thread_local char pending_path[4096];

  VisionEncoder *enc = model->encoder;

  if (turn_finished)
  {
    // Get a fresh input from stdin
    if (bos)
    {
      printf("User: ");
    }
    else
    {
      printf("\nUser: ");
    }

    if (fgets(g_prompt_buf, g_prompt_buf_len, stdin) == NULL)
    {
      if (!g_interrupted)
      {
        fprintf(stderr, "\nerror: failed to read user input\n");
      }
      goto fail;
    }
    if (strchr(g_prompt_buf, '\n') == NULL)
    {
      // Input is truncated (too long)
      fprintf(stderr, "\nerror: input too long\n");
      goto fail;
    }
    if (g_interrupted)
    {
      goto fail;
    }
    // Set string end
    g_prompt_buf[strcspn(g_prompt_buf, "\n")] = '\0';

    turn_finished = false;
    prompt_idx    = 0;
    token_start   = 0;
    n_tokens      = 0;

    if (bos)
    {
      g_tokens_buf[n_tokens++] = tok->bos;
    }
    g_tokens_buf[n_tokens++] = tok->sot;
    encode(tok, "user\n", strlen("user\n"), g_tokens_buf + n_tokens, &n_tokens);
    printf("Model: ");
  }

  if (pending_image)
  {
    pending_image = false;
    // Load the saved image path
    int     img_size = (enc && enable_mm) ? enc->config->image_size : 0;
    floatx *img      = prepare_image(pending_path, img_size);
    if (img == NULL)
    {
      fprintf(stderr, "error: failed to load image: '%s'\n", pending_path);
      goto fail;
    }
    was_image = true;
    return (InjectData){.type = INJECT_IMAG,
        .image                = img,
        .prefill_finished     = false,
        .quit                 = false};
  }

  if (was_image)
  {
    was_image                = false;
    g_tokens_buf[n_tokens++] = tok->eoi;  // <end_of_image>
    encode(tok, "\n\n", 2, g_tokens_buf + n_tokens, &n_tokens);
  }

  char *prompt = g_prompt_buf + prompt_idx;
  if (prompt[0] == '\0') goto epilogue;  // Empty string

  // Use @image{<path>} to insert an image
  char *image_cmd = strstr(prompt, "@image{");
  if (enc == NULL || !enable_mm)
  {
    image_cmd = NULL;
  }

  if (image_cmd == NULL)
  {
    // Pure text input
    goto finish_turn;
  }
  int spos = image_cmd - prompt;
  // Match the closing brace
  char *closing = strchr(image_cmd, (int)'}');
  int   epos    = -1;
  if (closing != NULL)
  {
    epos = closing - prompt;
  }
  if (epos == -1)
  {
    // Pure text chunk
    goto finish_turn;
  }
  if (spos == 0)
  {
    // Image at the start of chunk
    const char *path = image_cmd + strlen("@image{");
    char path_buf[4096];  // 4096 is the max number of characters allowed in
                          // path strings
    int path_len = epos - strlen("@image{");
    if (path_len < 0 || path_len >= sizeof(path_buf))
    {
      fprintf(stderr, "error: invalid image path\n");
      goto fail;
    }
    memcpy(path_buf, path, path_len);
    path_buf[path_len] = '\0';
    strcpy(pending_path, path_buf);

    pending_image = true;

    encode(tok, "\n\n", 2, g_tokens_buf + n_tokens, &n_tokens);
    g_tokens_buf[n_tokens++] = tok->soi;  // Insert <start_of_image>
    prompt_idx += epos + 1;
    RETURN_TEXT_INJECT(false, false);
  }
  // Matched a complete image command! Encode & return the text part first,
  // handle the image part in the next iteration
  encode(tok, prompt, spos, g_tokens_buf + n_tokens, &n_tokens);
  // No need to insert <start_of_image> here, leave it to the next iteration
  prompt_idx += spos;  // Next iteration directly starts with "@image"
  RETURN_TEXT_INJECT(false, false);

finish_turn:
  encode(tok, prompt, strlen(prompt), g_tokens_buf + n_tokens, &n_tokens);

epilogue:
  turn_finished            = true;
  g_tokens_buf[n_tokens++] = tok->eot;
  g_tokens_buf[n_tokens++] = get_token_idx(tok, "\n");
  g_tokens_buf[n_tokens++] = tok->sot;
  encode(tok, "model\n", strlen("model\n"), g_tokens_buf + n_tokens, &n_tokens);
  RETURN_TEXT_INJECT(true, false);

fail:
  return (InjectData){.type = INJECT_NONE, .quit = true};
}

/* */
static InjectData
chat_callback(int token, GemmaModel *model, bool enable_mm)
{
  VisionEncoder  *enc = model->encoder;
  TextDecoder    *dec = model->decoder;
  GemmaTokenizer *tok = dec->tokenizer;

  if (token == EOF)
  {
    // EOF: continue the injection process
    return new_turn(tok, model, enable_mm, false);
  }
  if (token == tok->eos || token == tok->eot)
  {
    // Start a new injection
    return new_turn(tok, model, enable_mm, false);
  }
  char byte_buf[2];
  printf("%s", decode(tok, token, byte_buf));
  fflush(stdout);
  return (InjectData){.type = INJECT_NONE, .quit = false};
}

/* */
int
chat(GemmaModel  *model,
    TextBuffer   *buf,
    VisionBuffer *vbuf,

    int   seqlen,
    int   chunk_size,
    float temperature,
    int   topk,
    float topp,
    float rpen,
    bool  enable_mm)
{
  VisionEncoder *enc = model->encoder;
  InjectData     first_turn =
      new_turn(model->decoder->tokenizer, model, enable_mm, true);
  if (first_turn.quit)
  {
    return 1;
  }
  sample(model, buf, vbuf, first_turn, seqlen, chunk_size, temperature, topk,
      topp, rpen, enable_mm, chat_callback);
  return 0;
}

/* ------------------------------------------------------------------ */
/* CLI helpers                                                        */
/* ------------------------------------------------------------------ */

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

#ifdef _WIN32
#  include <windows.h>
// The default console encoding is kinda weird on Windows
/* */
static void
set_utf8_console(void)
{
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

#else
// No problem with POSIX though
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
#endif

/* Pretty-print of the loaded model */
void
print_model_config(GemmaModel *model, int seqlen, bool enable_mm)
{
  const int       width = 20;
  VisionEncoder  *enc   = model->encoder;
  TextDecoder    *dec   = model->decoder;
  TextConfig     *cfg   = dec->config;
  GemmaTokenizer *tok   = dec->tokenizer;

  printf("\n========== Model Configuration ==========\n");
  printf("Architecture:\n");

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
  printf(
      "  %-*s: %s\n", width, "support_mm", cfg->support_mm ? "true" : "false");
  printf("  %-*s: %s\n", width, "qk_norm", cfg->qk_norm ? "true" : "false");
  printf("  %-*s: %s\n", width, "pre_mlp_norm",
      cfg->pre_mlp_norm ? "true" : "false");
  printf("  %-*s: %s\n", width, "pst_mlp_norm",
      cfg->pst_mlp_norm ? "true" : "false");
  printf("  %-*s: %s\n", width, "quant", model->quant ? "true" : "false");

  // Tokenizer
  printf("\nTokenizer:\n");
  printf("  %-*s: %d\n", width, "vocab_size", tok->vocab_size);
  printf("  %-*s: %d\n", width, "n_merges", tok->n_merges);
  printf("  %-*s: %d\n", width, "bos", tok->bos);
  printf("  %-*s: %d\n", width, "eos", tok->eos);
  printf("  %-*s: %d\n", width, "sot", tok->sot);
  printf("  %-*s: %d\n", width, "eot", tok->eot);
  printf("  %-*s: %d\n", width, "soi", tok->soi);
  printf("  %-*s: %d\n", width, "eoi", tok->eoi);
  printf("  %-*s: %d\n", width, "ist", tok->ist);

  printf("\nMemory Footprint (estimated):\n");

  size_t total = 0;

  // Embedding
  if (!model->quant)
  {
    total += cfg->embed_dim * cfg->vocab_size * sizeof(floatx);
  }
  else
  {
    total += cfg->embed_dim * cfg->vocab_size * sizeof(int8_t);
    total += cfg->vocab_size * sizeof(floatx);  // scales
  }

  // Weights per layer
  for (int l = 0; l < cfg->n_layers; l++)
  {
    int C   = cfg->embed_dim;
    int Cq  = cfg->n_heads * cfg->head_dim;
    int Ckv = cfg->n_kv_heads * cfg->head_dim;
    int CM  = cfg->mlp_dim;

    int layer_params = C * Cq + C * Ckv + C * Ckv + Cq * C + C * CM * 3;

    if (!model->quant)
    {
      total += layer_params * sizeof(floatx);
    }
    else
    {
      total += layer_params * sizeof(int8_t);
      total += (Cq + Ckv + Ckv + C * 2 + CM * 2) * sizeof(floatx);  // scales
    }

    // Norm layers
    total += C * sizeof(floatx);  // n1
    total += C * sizeof(floatx);  // n2
    if (cfg->qk_norm)
    {
      total += 2 * cfg->head_dim * sizeof(floatx);
    }
    if (cfg->pre_mlp_norm)
    {
      total += C * sizeof(floatx);
    }
    if (cfg->pst_mlp_norm)
    {
      total += C * sizeof(floatx);
    }
  }

  // Final norm
  total += cfg->embed_dim * sizeof(floatx);

  printf("  %-*s: %.2f GB\n", width, "Weights",
      (float)total / (1024.0 * 1024.0 * 1024.0));

  // KV Cache
  int    Ckv            = cfg->n_kv_heads * cfg->head_dim;
  size_t kv_cache_bytes = cfg->n_layers * 2 * Ckv * sizeof(floatx);
  printf("  %-*s: %.2f KB\n", width, "KV Cache (tok)",
      (float)kv_cache_bytes / 1024.0);

  // Gemma Buffer
  int           ppi  = 0;
  VisionConfig *vcfg = NULL;
  if (cfg->support_mm && enable_mm && enc != NULL)
  {
    vcfg = enc->config;
    ppi  = vcfg->image_size / vcfg->patch_size;
  }
  int Cq   = cfg->n_heads * cfg->head_dim;
  int CM   = cfg->mlp_dim;
  int mult = (cfg->support_mm && enable_mm && enc != NULL) ? ppi * ppi : 1;

  size_t dec_bytes = 0;

  // Quantized buffers
  if (model->quant)
  {
    dec_bytes += cfg->embed_dim * sizeof(int8_t);  // x_i8
    dec_bytes += Cq * sizeof(int8_t);              // xo_i8
    dec_bytes += CM * sizeof(int8_t);              // xg_i8
  }

  // Main buffers
  dec_bytes += cfg->n_layers * 2 * seqlen * Ckv * sizeof(floatx);  // kv_cache
  dec_bytes += cfg->vocab_size * sizeof(floatx);                   // logits
  dec_bytes += mult * cfg->embed_dim * sizeof(floatx);             // x
  dec_bytes += mult * cfg->embed_dim * sizeof(floatx);             // resid
  dec_bytes += mult * Cq * sizeof(floatx);                         // xq
  dec_bytes += mult * Ckv * sizeof(floatx);                        // xk
  dec_bytes += mult * cfg->head_dim * sizeof(floatx);          // csfreqs_slid
  dec_bytes += mult * cfg->head_dim * sizeof(floatx);          // csfreqs_full
  dec_bytes += mult * Ckv * sizeof(floatx);                    // xv
  dec_bytes += mult * Cq * sizeof(floatx);                     // xo
  dec_bytes += mult * cfg->n_heads * seqlen * sizeof(floatx);  // att
  dec_bytes += mult * CM * sizeof(floatx);                     // xg
  dec_bytes += mult * CM * sizeof(floatx);                     // xu

  printf("  %-*s: %.2f MB\n", width, "Gemma Buffer",
      (float)dec_bytes / (1024.0 * 1024.0));

  // SigLIP Buffer (if applicable)
  if (cfg->support_mm && enable_mm && enc != NULL)
  {
    int C         = vcfg->hidden_dim;
    int CM        = vcfg->mlp_dim;
    int ppi_local = vcfg->image_size / vcfg->patch_size;
    int N         = ppi_local * ppi_local;
    int NH        = vcfg->n_heads;

    size_t enc_bytes = 0;

    // Quantized buffers
    if (model->quant)
    {
      enc_bytes += N * C * sizeof(int8_t);   // x_i8
      enc_bytes += N * sizeof(floatx);       // x_scales
      enc_bytes += N * CM * sizeof(int8_t);  // mlp_i8
      enc_bytes += N * sizeof(floatx);       // mlp_scales
    }

    // Main buffers
    enc_bytes += N * C * sizeof(floatx);       // x
    enc_bytes += N * C * sizeof(floatx);       // resid
    enc_bytes += N * C * sizeof(floatx);       // xq
    enc_bytes += N * C * sizeof(floatx);       // xk
    enc_bytes += N * C * sizeof(floatx);       // xv
    enc_bytes += N * C * sizeof(floatx);       // att_out
    enc_bytes += N * CM * sizeof(floatx);      // mlp_hidden
    enc_bytes += NH * N * N * sizeof(floatx);  // scores

    printf("  %-*s: %.2f MB\n", width, "SigLIP Buffer",
        (float)enc_bytes / (1024.0 * 1024.0));
  }

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
  "  modelfile              path to the model file\n"
  "\n"
  "options:\n"
  "  -l, --seqlen <N>       set sequence length"
                            " (default: " TOSTRING(DEFAULT_SEQLEN) ")\n"
  "  -k, --topk <N>         set top-k sampling value"
                            " (default: " TOSTRING(DEFAULT_TOPK) ")\n"
  "  -s, --seed <N>         set random seed"
                            " (default: current time)\n"
  "  -u, --chunk <N>        set prefilling chunk size, must be >= 1"
                            " (default: " TOSTRING(DEFAULT_CHUNK_SIZE) ")\n"
  "  -t, --temperature <F>  set temperature value, must be >= 0.0"
                            " (default: " TOSTRING(DEFAULT_TEMPERATURE) ")\n"
  "  -p, --topp <F>         set top-p sampling value, must be 0.0 < p <= 1.0"
                            " (default: " TOSTRING(DEFAULT_TOPP) ")\n"
  "  -r, --rpen <F>         set repetition penalty, must be >= 1.0"
                            " (default: " TOSTRING(DEFAULT_RPEN) ")\n"
  "  -i, --prompt <S>       set input prompt, ignored if chat mode is enabled"
                            " (default: \"" DEFAULT_PROMPT "\")\n"
  "  -c, --chat             enable chat mode\n"
  "  -m, --enable-mm        enable multimodal capability, if supported by the model\n"
  "  -v, --verbose          print model info\n"
  "  -h, --help, -?         display this help message\n"
  "\n"
  "controls:\n"
  "  Ctrl+C                 gracefully interrupt generation and exit\n"
  "\n"
  "examples:\n"
  "  ./gemma model.bin -l 2048 -t 0.8 -c\n"
  "  ./gemma model.bin -i \"Hello I'm a language model,\""
          "--seqlen 4096 --topk 50 --seed 12345\n");
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

/* ------------------------------------------------------------------ */
/* main                                                               */
/* ------------------------------------------------------------------ */

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
  bool         enable_mm   = false;
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
    printf("\nerror: model filename is not provided\n");
    if (utf8_argv != NULL)
    {
      free_utf8_argv(utf8_argv, argc);
    }
    goto fail;
  }

  char *modelfile = argv[1];
  if (!strcmp(modelfile, "-h") || !strcmp(modelfile, "--help"))
  {
    print_usage();
    goto end;
  }

  const char *val;

  // Set attention block sizes from environment variables
  val = getenv("GEMMAC_QK_BLOCK_SIZE");
  if (val != NULL)
  {
    if (!safe_atoui(val, &qk_block_size) || qk_block_size == 0)
    {
      fprintf(stderr, "error: GEMMAC_QK_BLOCK_SIZE must be greater than 0\n");
    }
  }

  // Parse the command line arguments
  for (int i = 2; i < argc; i++)
  {
    const char *arg = argv[i];

    if (strcmp(arg, "-l") == 0 || strcmp(arg, "--seqlen") == 0)
    {
      val = safe_get_arg(i++, argc, argv);
      if (!val) goto fail;
      if (!safe_atoui(val, &seqlen))
      {
        fprintf(stderr, "error: invalid number for --seqlen: %s\n", val);
        goto fail;
      }
    }
    else if (strcmp(arg, "-k") == 0 || strcmp(arg, "--topk") == 0)
    {
      val = safe_get_arg(i++, argc, argv);
      if (!val) goto fail;
      if (!safe_atoui(val, &topk))
      {
        fprintf(stderr, "error: invalid number for --topk: %s\n", val);
        goto fail;
      }
    }
    else if (strcmp(arg, "-s") == 0 || strcmp(arg, "--seed") == 0)
    {
      val = safe_get_arg(i++, argc, argv);
      if (!val) goto fail;
      if (!safe_atoui(val, &seed))
      {
        fprintf(stderr, "error: invalid number for --seed: %s\n", val);
        goto fail;
      }
    }
    else if (strcmp(arg, "-u") == 0 || strcmp(arg, "--chunk") == 0)
    {
      val = safe_get_arg(i++, argc, argv);
      if (!val) goto fail;
      if (!safe_atoui(val, &chunk_size))
      {
        fprintf(stderr, "error: invalid number for --chunk: %s\n", val);
        goto fail;
      }
    }
    else if (strcmp(arg, "-t") == 0 || strcmp(arg, "--temperature") == 0)
    {
      val = safe_get_arg(i++, argc, argv);
      if (!val) goto fail;
      if (!safe_atof(val, &temperature) || temperature < 0.0f)
      {
        fprintf(stderr, "error: invalid temperature: %s (must be >= 0)\n", val);
        goto fail;
      }
    }
    else if (strcmp(arg, "-p") == 0 || strcmp(arg, "--topp") == 0)
    {
      val = safe_get_arg(i++, argc, argv);
      if (!val) goto fail;
      if (!safe_atof(val, &topp) || topp <= 0.0f || topp > 1.0f)
      {
        fprintf(stderr, "error: invalid top-p: %s (must be 0 < p <= 1)\n", val);
        goto fail;
      }
    }
    else if (strcmp(arg, "-r") == 0 || strcmp(arg, "--rpen") == 0)
    {
      val = safe_get_arg(i++, argc, argv);
      if (!val) goto fail;
      if (!safe_atof(val, &rpen) || rpen < 1.0f)
      {
        fprintf(stderr,
            "error: invalid repetition penalty: %s (must be >= 1)\n", val);
        goto fail;
      }
    }
    else if (strcmp(arg, "-i") == 0 || strcmp(arg, "--prompt") == 0)
    {
      val = safe_get_arg(i++, argc, argv);
      if (!val) goto fail;
      prompt = val;
    }
    else if (strcmp(arg, "-c") == 0 || strcmp(arg, "--chat") == 0)
    {
      chatmode = true;
    }
    else if (strcmp(arg, "-m") == 0 || strcmp(arg, "--enable-mm") == 0)
    {
      enable_mm = true;
    }
    else if (strcmp(arg, "-v") == 0 || strcmp(arg, "--verbose") == 0)
    {
      print_cfg = true;
    }
    else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0 ||
             strcmp(arg, "-?") == 0)
    {
      print_usage();
      goto fail;
    }
    else
    {
      fprintf(stderr, "error: unknown option: %s\n", arg);
      print_usage();
      goto fail;
    }
  }

  srand(seed);

  // Read model
  model = read_model(modelfile, enable_mm);
  if (model == NULL) goto fail;

  // Text config
  cfg = model->decoder->config;
  if (enable_mm && !cfg->support_mm)
  {
    fprintf(stderr, "warning: model file '%s' does not support multimodal\n",
        modelfile);
  }

  // Vision config
  if (model->encoder != NULL)
  {
    vcfg = model->encoder->config;
  }

  // Text buffer
  buf = malloc_text_buffer(
      cfg, vcfg, (int)seqlen, (int)chunk_size, enable_mm, model->quant);
  if (buf == NULL) goto fail;

  // Vision buffer
  if (enable_mm && cfg->support_mm)
  {
    vbuf = malloc_vision_buffer(vcfg, model->quant);
    if (vbuf == NULL) goto fail;
  }

  if (print_cfg)
  {
    print_model_config(model, (int)seqlen, enable_mm);
  }

  g_prompt_buf_len = seqlen;
  MALLOC(g_prompt_buf, seqlen, "g_prompt_buf", goto fail;);
  // Token buffer needs extra space for special tokens like <start_of_turn>
  g_tokens_buf_len = seqlen + 128;
  MALLOC(g_tokens_buf, seqlen + 128, "g_tokens_buf", goto fail;);

  if (chatmode)
  {
    chat(model, buf, vbuf, (int)seqlen, (int)chunk_size, temperature, (int)topk,
        topp, rpen, enable_mm);
  }
  else
  {
    generate(model, buf, vbuf, prompt, (int)seqlen, (int)chunk_size,
        temperature, (int)topk, topp, rpen, enable_mm);
  }

  if (g_interrupted)
  {
    printf("\ninterrupted by user\n");
  }

end:
  free_utf8_argv(utf8_argv, argc);
  free(g_prompt_buf);
  if (cfg != NULL)
  {
    free_text_buffer(buf, model->quant);
    free_vision_buffer(vbuf, model->quant);
  }
  free_gemma_model(model);
  return 0;

fail:
  free_utf8_argv(utf8_argv, argc);
  free(g_prompt_buf);
  if (cfg != NULL)
  {
    free_text_buffer(buf, model->quant);
    free_vision_buffer(vbuf, model->quant);
  }
  free_gemma_model(model);
  return 1;
}
