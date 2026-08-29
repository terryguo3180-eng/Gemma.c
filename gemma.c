/* Gemma 1 & 2 & 3 & 3n implemented in a single file of pure C.
 *
 * One big self-contained inference engine. No external deps beyond
 * the C standard library + OpenMP for the parallel bits.
 * Supports FP16 and int8 quantized weights, sliding-window + full
 * attention, optional SigLIP vision encoder for multimodal models (not
 * finished), and a simple BPE tokenizer.
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
#include <time.h>

/* Max value for _Float16 so we can clamp safely */
#define FP16_MAX   \
  (((union {       \
    _Float16 f;    \
    uint16_t b;    \
  }){.b = 0x7BFF}) \
          .f)

#ifdef DEBUG
#  pragma message("Debug mode on")
#endif

/* ------------------------------------------------------------------ */
/* Global interruption handling (Ctrl+C)                              */
/* ------------------------------------------------------------------ */

static volatile bool g_interrupted = false;

static void
signal_handler(int signum)
{
  (void)signum;
  g_interrupted = true;
}

void
setup_signal_handler(void)
{
  signal(SIGINT, signal_handler);
#ifdef SIGTERM
  signal(SIGTERM, signal_handler);
#endif
}

/* ------------------------------------------------------------------ */
/* Memory wrappers, fail fast and print a useful message              */
/* ------------------------------------------------------------------ */

#define MALLOC(ptr, count, name, fail)                                      \
  do {                                                                      \
    ptr = malloc((count) * sizeof(*(ptr)));                                 \
    if (ptr == NULL)                                                        \
    {                                                                       \
      fprintf(stderr, "Memory allocation failed: %s (size: %zu)\n", (name), \
          (count) * sizeof(*(ptr)));                                        \
      fail                                                                  \
    }                                                                       \
  }                                                                         \
  while (0)

#define CALLOC(ptr, count, name, fail)                                      \
  do {                                                                      \
    ptr = calloc((count), sizeof(*(ptr)));                                  \
    if (ptr == NULL)                                                        \
    {                                                                       \
      fprintf(stderr, "Memory allocation failed: %s (size: %zu)\n", (name), \
          (count) * sizeof(*(ptr)));                                        \
      fail                                                                  \
    }                                                                       \
  }                                                                         \
  while (0)

/* ------------------------------------------------------------------ */
/* File-reading helpers (big-endian integers + pascal strings)        */
/* ------------------------------------------------------------------ */

// Statement expressions: ({ ... }), an extension of GNU C. Not supported in
// MSVC
#define FGETC(fp, name, fail)                          \
  ({                                                   \
    int _fgetc_ret = fgetc(fp);                        \
    if (_fgetc_ret == EOF)                             \
    {                                                  \
      fprintf(stderr, "File read failed: %s", (name)); \
      fail                                             \
    }                                                  \
    _fgetc_ret;                                        \
  })

#define FREAD(ptr, count, fp, name, fail)                                      \
  do {                                                                         \
    size_t _fread_buf = fread((ptr), sizeof(*(ptr)), (count), (fp));           \
    if (_fread_buf != (count))                                                 \
    {                                                                          \
      fprintf(stderr, "File read failed: %s (expected %d, got %zu)\n", (name), \
          (count), _fread_buf);                                                \
      fail                                                                     \
    }                                                                          \
  }                                                                            \
  while (0)

#define READ_UINT16(fp, name, fail)               \
  ({                                              \
    /* Big-endian */                              \
    unsigned char _ru16_buf[2];                   \
    FREAD(_ru16_buf, 2, (fp), (name), fail);      \
    ((int)_ru16_buf[0] << 8) | (int)_ru16_buf[1]; \
  })

#define READ_UINT32(fp, name, fail)                                   \
  ({                                                                  \
    /* Big-endian */                                                  \
    unsigned char _ru32_buf[4];                                       \
    FREAD(_ru32_buf, 4, (fp), (name), fail);                          \
    ((uint32_t)_ru32_buf[0] << 24) | ((uint32_t)_ru32_buf[1] << 16) | \
        ((uint32_t)_ru32_buf[2] << 8) | ((uint32_t)_ru32_buf[3]);     \
  })

#define READ_FP32(fp, name, fail)                           \
  ({                                                        \
    uint32_t _rfp32_bits = READ_UINT32((fp), (name), fail); \
    float    _rfp32_val;                                    \
    memcpy(&_rfp32_val, &_rfp32_bits, sizeof(float));       \
    _rfp32_val;                                             \
  })

/* Pascal-style string: first byte is length, then that many chars */
#define READ_STR(fp, data, offset, name, fail)          \
  ({                                                    \
    char _rstr_len_name[128];                           \
    snprintf(_rstr_len_name, 128, "%s.length", (name)); \
    int   _rstr_len = FGETC(fp, _rstr_len_name, fail);  \
    char *_rstr_ptr = (data) + *(offset);               \
    FREAD(_rstr_ptr, _rstr_len, (fp), (name), fail);    \
    (data)[*(offset) + _rstr_len] = '\0';               \
    *(offset) += _rstr_len + 1;                         \
    _rstr_ptr;                                          \
  })

#define READ_TENSOR(ptr, count, fp, name, fail) \
  do {                                          \
    MALLOC((ptr), (count), (name), fail);       \
    FREAD((ptr), (count), (fp), (name), fail);  \
  }                                             \
  while (0)

/* Linear layer weights: either plain FP16 or int8 + per-channel scales */
#define READ_LINEAR(w, fp, m, n, quant, name, fail)                         \
  do {                                                                      \
    MALLOC((w), 1, (name), fail);                                           \
    if (!(quant))                                                           \
    {                                                                       \
      (w)->dtype = DTYPE_FP16;                                              \
      char _rlinear_fp16_name[128];                                         \
      snprintf(_rlinear_fp16_name, 128, "%s.fp16", (name));                 \
      READ_TENSOR((w)->fp16, (m) * (n), (fp), _rlinear_fp16_name, fail);    \
    }                                                                       \
    else                                                                    \
    {                                                                       \
      (w)->dtype = DTYPE_INT8;                                              \
      char _rlinear_i8q_name[128];                                          \
      char _rlinear_i8scales_name[128];                                     \
      snprintf(_rlinear_i8q_name, 128, "%s.i8.q", (name));                  \
      snprintf(_rlinear_i8scales_name, 128, "%s.i8.scales", (name));        \
      READ_TENSOR((w)->i8.q, (m) * (n), (fp), _rlinear_i8q_name, fail);     \
      READ_TENSOR((w)->i8.scales, (n), (fp), _rlinear_i8scales_name, fail); \
    }                                                                       \
  }                                                                         \
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
} SigLIPConfig;

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
  bool  post_mlp_norm;
  bool  quant;  // int8 weights
} GemmaConfig;

static void
free_config(GemmaConfig *conf)
{
  if (conf == NULL) return;
  free(conf->att_layers);
  free(conf);
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

static int
cmp_token(const void *a, const void *b)
{
  return strcmp(((Token *)a)->val, ((Token *)b)->val);
}

static int
cmp_merge(const void *a, const void *b)
{
  int ret = strcmp(((Merge *)a)->str1, ((Merge *)b)->str1);
  if (ret != 0) { return ret; }
  return strcmp(((Merge *)a)->str2, ((Merge *)b)->str2);
}

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
  if (val == NULL) { return -1; }
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
encode(GemmaTokenizer *tok, const char *sstr, int *tokens, int *n_tokens)
{
  unsigned char *str = (unsigned char *)sstr;
  int            len = (int)strlen((char *)str);

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

    if (str[i] >> 7 == 0) { n_bytes = 1; }
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
    else { n_bytes = 1; }

    char cstr[5];
    for (int b = 0; b < n_bytes; b++) { cstr[b] = (char)str[i++]; }
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
    else { tokens[tok_i++] = token; }
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
        for (int j = i + 1; j < tok_i - 1; j++) { tokens[j] = tokens[j + 1]; }
        tok_i--;
      }
      i++;
    }
  }

  *n_tokens += tok_i;
  return tokens;
}

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
  DTYPE_FP16,
  DTYPE_INT8
} WeightDType;

/* A linear layer, either dense FP16 or int8 + scales */
typedef struct
{
  WeightDType dtype;
  union
  {
    _Float16 *fp16;
    struct
    {
      int8_t   *q;
      _Float16 *scales;
    } i8;
  };
} Linear;

static void
free_linear(Linear *l)
{
  if (l == NULL) return;
  if (l->dtype == DTYPE_FP16) { free(l->fp16); }
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
  _Float16 *nq;  // (head_dim,)
  _Float16 *nk;  // (head_dim,)
  _Float16 *n1;  // (embed_dim,)
  _Float16 *n2;  // (embed_dim,)
  _Float16 *n3;  // (embed_dim,)
  _Float16 *n4;  // (embed_dim,)
} GemmaDecoderLayer;

static void
free_gemma_layer(GemmaDecoderLayer *layer)
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
  _Float16 *bq;  // (hidden_dim,)
  _Float16 *bk;  // (hidden_dim,)
  _Float16 *bv;  // (hidden_dim,)
  _Float16 *bo;  // (hidden_dim,)

  // ViT feedforward weights
  Linear *w1;  // (hidden_dim, mlp_dim).T
  Linear *w2;  // (hidden_dim, mlp_dim).T

  // ViT feedforward biases
  _Float16 *b1;  // (mlp_dim,)
  _Float16 *b2;  // (mlp_dim,)

  // ViT layernorm weights
  _Float16 *n1;  // (hidden_dim,)
  _Float16 *n2;  // (hidden_dim,)

  // ViT layernorm biases
  _Float16 *n1_b;  // (hidden_dim,)
  _Float16 *n2_b;  // (hidden_dim,)
} SigLIPEncoderLayer;

static void
free_siglip_layer(SigLIPEncoderLayer *layer)
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

typedef struct
{
  SigLIPConfig *config;

  _Float16 *patch_emb;      // (hidden_dim, 3, patch_size, patch_size)
  _Float16 *patch_emb_b;    // (hidden_dim,)
  Linear   *pos_embedding;  // ((image_size / patch_size)^2, hidden_dim)

  SigLIPEncoderLayer **layers;

  _Float16 *post_norm;    // (hidden_dim,)
  _Float16 *post_norm_b;  // (hidden_dim,)
  _Float16 *norm;         // (hidden_dim,)
  Linear   *proj;         // (hidden_dim, embed_dim).T
} SigLIPVisionEncoder;

void
free_siglip(SigLIPVisionEncoder *enc)
{
  if (enc == NULL) return;

  free(enc->patch_emb);
  free(enc->patch_emb_b);
  free_linear(enc->pos_embedding);
  if (enc->layers != NULL)
  {
    for (int i = 0; i < enc->config->n_layers; i++)
    {
      free_siglip_layer(enc->layers[i]);
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

/* Top-level model container */
typedef struct
{
  GemmaConfig         *config;
  GemmaTokenizer      *tokenizer;
  SigLIPVisionEncoder *vision_enc;
  Linear *
      embedding;  // (vocab_size, embed_dim), shared with lm_head (tied weights)
  GemmaDecoderLayer **layers;
  _Float16           *final_norm;  // (embed_dim,)
} GemmaModel;

void
free_model(GemmaModel *model)
{
  if (model == NULL) return;

  free_tokenizer(model->tokenizer);
  free_siglip(model->vision_enc);
  free_linear(model->embedding);
  if (model->layers != NULL && model->config != NULL)
  {
    for (int i = 0; i < model->config->n_layers; i++)
    {
      free_gemma_layer(model->layers[i]);
    }
    free(model->layers);
  }
  free(model->final_norm);
  free_config(model->config);
  free(model);
}

/* ------------------------------------------------------------------ */
/* Runtime buffers (allocated once, reused every step)                */
/* ------------------------------------------------------------------ */

/* */
typedef struct
{
  int8_t   *x_i8;        // (n_patches, hidden_dim)
  _Float16 *x_scales;    // (n_patches,)
  int8_t   *mlp_i8;      // (n_patches, mlp_dim)
  _Float16 *mlp_scales;  // (n_patches)

  _Float16 *x;           // (n_patches, hidden_dim)
  _Float16 *resid;       // (n_patches, hidden_dim)
  _Float16 *xq;          // (n_patches, hidden_dim)
  _Float16 *xk;          // (n_patches, hidden_dim)
  _Float16 *xv;          // (n_patches, hidden_dim)
  _Float16 *att_out;     // (n_patches, hidden_dim)
  _Float16 *mlp_hidden;  // (n_patches, mlp_dim)
  _Float16 *scores;
} SigLIPBuffer;

void
free_siglip_buffer(SigLIPBuffer *buf, bool quant)
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

SigLIPBuffer *
malloc_siglip_buffer(SigLIPConfig *vconf, bool quant)
{
  SigLIPBuffer *buf = NULL;
  CALLOC(buf, 1, "sbuf", goto fail;);

  int C   = vconf->hidden_dim;
  int ppi = vconf->image_size / vconf->patch_size;
  int N   = ppi * ppi;
  int CM  = vconf->mlp_dim;
  int NH  = vconf->n_heads;

  if (quant)
  {
    MALLOC(buf->x_i8, N * C, "sbuf.x_i8", goto fail;);
    MALLOC(buf->x_scales, N, "sbuf.x_scales", goto fail;);
    MALLOC(buf->mlp_i8, N * CM, "sbuf.mlp_i8", goto fail;);
    MALLOC(buf->mlp_scales, N, "sbuf.mlp_scales", goto fail;);
  }

  MALLOC(buf->x, N * C, "sbuf.x", goto fail;);
  MALLOC(buf->resid, N * C, "sbuf.resid", goto fail;);
  MALLOC(buf->xq, N * C, "sbuf.q", goto fail;);
  MALLOC(buf->xk, N * C, "sbuf.k", goto fail;);
  MALLOC(buf->xv, N * C, "sbuf.v", goto fail;);
  MALLOC(buf->att_out, N * C, "sbuf.attn_out", goto fail;);
  MALLOC(buf->mlp_hidden, N * CM, "sbuf.mlp_hidden", goto fail;);
  MALLOC(buf->scores, NH * N * N, "sbuf.scores", goto fail;);

  return buf;

fail:
  free_siglip_buffer(buf, quant);
  return NULL;
}

typedef struct
{
  int cache_len;

  // Temporary quantized activations (when quant=true)
  int8_t *x_i8;   // ([n_patches], embed_dim,)
  int8_t *xo_i8;  // (n_heads, [n_patches], head_dim)
  int8_t *xg_i8;  // ([n_patches], mlp_dim,)

  // Pre-computed cos/sin for RoPE
  _Float16 *csfreqs_slid;  // ([n_patches], head_dim / 2, 2)
  _Float16 *csfreqs_full;  // ([n_patches], head_dim / 2, 2)

  // Residual stream
  _Float16 *x;      // ([n_patches], embed_dim,)
  _Float16 *resid;  // ([n_patches], embed_dim,)

  // Attention buffers
  _Float16 *xq;        // (n_heads, [n_patches], head_dim)
  _Float16 *xk;        // (n_kv_heads, [n_patches], head_dim)
  _Float16 *xv;        // (n_kv_heads, [n_patches], head_dim)
  _Float16 *xo;        // (n_heads, [n_patches], head_dim)
  _Float16 *att;       // (n_heads, [n_patches], cache_len)
  _Float16 *kv_cache;  // (n_layers, 2, cache_len, n_kv_heads, head_dim)

  // MLP buffers
  _Float16 *xg;      // ([n_patches], mlp_dim,)
  _Float16 *xu;      // ([n_patches], mlp_dim,)
  _Float16 *logits;  // (vocab_size,)
} GemmaBuffer;

void
free_buffer(GemmaBuffer *buf, bool quant)
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
    free(buf->xo_i8);
    free(buf->xg_i8);
  }
  free(buf);
}

GemmaBuffer *
malloc_buffer(
    GemmaConfig *conf, SigLIPConfig *vconf, int cache_len, bool enable_mm)
{
  GemmaBuffer *buf = NULL;
  CALLOC(buf, 1, "buf", goto fail;);

  buf->cache_len = cache_len;

  int C  = conf->embed_dim;
  int L  = conf->n_layers;
  int CH = conf->head_dim;
  int NH = conf->n_heads;
  int CM = conf->mlp_dim;

  int Cq  = NH * CH;
  int Ckv = conf->n_kv_heads * CH;

  if (conf->quant)
  {
    MALLOC(buf->x_i8, C, "buf.x_i8", goto fail;);
    MALLOC(buf->xo_i8, Cq, "buf.xo_i8", goto fail;);
    MALLOC(buf->xg_i8, conf->mlp_dim, "buf.xg_i8", goto fail;);
  }

  MALLOC(buf->kv_cache, L * 2 * cache_len * Ckv, "buf.kv_cache", goto fail;);
  MALLOC(buf->logits, conf->vocab_size, "buf.logits", goto fail;);

  // Multimodal models need space for a whole image worth of tokens
  int mult = 1;
  if (conf->support_mm && enable_mm && vconf != NULL)
  {
    int ppi = vconf->image_size / vconf->patch_size;

    mult = ppi * ppi;
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
  free_buffer(buf, conf->quant);
  return NULL;
}

/* ------------------------------------------------------------------ */
/* Model loading                                                      */
/* ------------------------------------------------------------------ */

/* */
GemmaModel *
read_model(const char *filename, bool enable_mm)
{
  FILE                *fp    = NULL;
  GemmaTokenizer      *tok   = NULL;
  GemmaConfig         *conf  = NULL;
  SigLIPVisionEncoder *enc   = NULL;
  SigLIPConfig        *vconf = NULL;
  GemmaModel          *model = NULL;

  char *att_layers_buf = NULL;

  fp = fopen(filename, "rb");
  if (fp == NULL)
  {
    fprintf(stderr, "Failed to open file: %s\n", filename);
    goto fail;
  }

  CALLOC(tok, 1, "model.tokenizer", goto fail;);
  CALLOC(conf, 1, "model.config", goto fail;);

  // Read the configs
  conf->n_layers   = FGETC(fp, "model.config.n_layers", goto fail;);
  conf->n_heads    = FGETC(fp, "model.config.n_heads", goto fail;);
  conf->n_kv_heads = FGETC(fp, "model.config.n_kv_heads", goto fail;);

  conf->head_dim   = READ_UINT16(fp, "model.config.head_dim", goto fail;);
  conf->embed_dim  = READ_UINT16(fp, "model.config.embed_dim", goto fail;);
  conf->mlp_dim    = READ_UINT16(fp, "model.config.mlp_dim", goto fail;);
  conf->q_scale    = READ_UINT16(fp, "model.config.q_scale", goto fail;);
  conf->slide_len  = READ_UINT16(fp, "model.config.slide_len", goto fail;);
  conf->image_toks = READ_UINT16(fp, "model.config.image_toks", goto fail;);
  conf->max_seqlen = READ_UINT32(fp, "model.config.max_seqlen", goto fail;);
  conf->vocab_size = READ_UINT32(fp, "model.config.vocab_size", goto fail;);

  conf->local_theta   = READ_FP32(fp, "model.config.local_theta", goto fail;);
  conf->global_theta  = READ_FP32(fp, "model.config.global_theta", goto fail;);
  conf->eps           = READ_FP32(fp, "model.config.eps", goto fail;);
  conf->att_softcap   = READ_FP32(fp, "model.config.att_softcap", goto fail;);
  conf->logit_softcap = READ_FP32(fp, "model.config.logit_softcap", goto fail;);

  // Packed bit-field of which layers use sliding-window attention
  // A terrible terrible idea, wish I didn't do this
  int n_bytes = FGETC(fp, "model.config.att_layers", goto fail;);
  MALLOC(att_layers_buf, n_bytes, "model.config.att_layers", goto fail;);
  MALLOC(
      conf->att_layers, conf->n_layers, "model.config.att_layers", goto fail;);
  FREAD(att_layers_buf, n_bytes, fp, "model.config.att_layers", goto fail;);
  for (int i = 0; i < conf->n_layers; i++)
  {
    int pos             = i;
    int byte_idx        = pos / 8;
    int bit_idx         = 7 - (pos % 8);
    conf->att_layers[i] = (att_layers_buf[byte_idx] >> bit_idx) & 1;
  }
  free(att_layers_buf);
  att_layers_buf = NULL;

  // Extra feature flags packed into one byte
  int extra_flags     = FGETC(fp, "model.config.extra_flags", goto fail;);
  conf->support_mm    = (extra_flags & 16) == 16;
  conf->qk_norm       = (extra_flags & 8) == 8;
  conf->pre_mlp_norm  = (extra_flags & 4) == 4;
  conf->post_mlp_norm = (extra_flags & 2) == 2;
  conf->quant         = (extra_flags & 1);

  bool use_mm = conf->support_mm && enable_mm;

  if (use_mm)
  {
    // Read vision config
    CALLOC(enc, 1, "model.vision_enc", goto fail;);
    CALLOC(vconf, 1, "model.vision_enc.config", goto fail;);
    vconf->n_layers   = FGETC(fp, "model.config.n_layers", goto fail;);
    vconf->n_heads    = FGETC(fp, "model.config.n_heads", goto fail;);
    vconf->mlp_dim    = READ_UINT16(fp, "model.config.mlp_dim", goto fail;);
    vconf->hidden_dim = READ_UINT16(fp, "model.config.hidden_dim", goto fail;);
    vconf->image_size = READ_UINT16(fp, "model.config.image_size", goto fail;);
    vconf->patch_size = READ_UINT16(fp, "model.config.patch_size", goto fail;);
    vconf->eps        = READ_FP32(fp, "model.config.eps", goto fail;);
    enc->config       = vconf;
  }
  else if (conf->support_mm)
  {
    // Skip vision config if user didn't ask for multimodal
    FGETC(fp, "model.config.n_layers", goto fail;);
    FGETC(fp, "model.config.n_heads", goto fail;);
    READ_UINT16(fp, "model.config.mlp_dim", goto fail;);
    READ_UINT16(fp, "model.config.hidden_dim", goto fail;);
    READ_UINT16(fp, "model.config.image_size", goto fail;);
    READ_UINT16(fp, "model.config.patch_size", goto fail;);
    READ_FP32(fp, "model.config.eps", goto fail;);
  }

  // dtype (only supports float16)
  int  offset = 0;
  char dtype[10];
  READ_STR(fp, dtype, &offset, "dtype", goto fail;);
  if (strcmp(dtype, "float16") != 0)
  {
    printf("dtype '%s' not supported\n", dtype);
    goto fail;
  }

  // Build vocabulary
  offset = 0;

  tok->vocab_size = conf->vocab_size;
  if (conf->support_mm) { tok->vocab_size++; }  // ++ for the <image_soft_token>
  int vocab_data_bytes = get_strarr_bytes(fp, tok->vocab_size);
  if (vocab_data_bytes == -1) { goto fail; }
  MALLOC(tok->vocab_data, vocab_data_bytes, "model.tokenizer.vocab_data",
         goto fail;);
  MALLOC(tok->vocab, tok->vocab_size, "model.tokenizer.vocab", goto fail;);
  MALLOC(tok->vocab_sorted, tok->vocab_size, "model.tokenizer.vocab_sorted",
         goto fail;);
  for (int i = 0; i < tok->vocab_size; i++)
  {
    char name[64];
    snprintf(name, sizeof(name), "model.tokenizer.vocab_data.%d", i);
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
  tok->n_merges = (int)READ_UINT32(fp, "model.tokenizer.n_merges", goto fail;);
  MALLOC(tok->ranks, tok->n_merges, "model.tokenizer.ranks", goto fail;);
  MALLOC(tok->merge_data, get_strarr_bytes(fp, tok->n_merges * 2),
         "model.tokenizer.merge_data", goto fail;);

  offset = 0;
  for (int i = 0; i < tok->n_merges; i++)
  {
    char name0[64], name1[64];
    snprintf(name0, sizeof(name0), "model.tokenizer.merge_data.%d.0", i);
    snprintf(name1, sizeof(name1), "model.tokenizer.merge_data.%d.1", i);
    char *str1 = READ_STR(fp, tok->merge_data, &offset, name0, goto fail;);
    char *str2 = READ_STR(fp, tok->merge_data, &offset, name1, goto fail;);
    tok->ranks[i].rank = i;
    tok->ranks[i].str1 = str1;
    tok->ranks[i].str2 = str2;
  }
  qsort(tok->ranks, tok->n_merges, sizeof(tok->ranks[0]), cmp_merge);

  // Build the model
  CALLOC(model, 1, "model", goto fail;);
  model->vision_enc = enc;

  // Read the weights
  model->config    = conf;
  model->tokenizer = tok;

  // The shape is actually (vocab_size, embed_dim), but it uses per-tensor
  // quantization rather than per-channel like other weights. Gemma uses tied
  // weights, which means the final lm_head shares the same weights with the
  // embedding table, but transposed. So it becomes per-channel quantization in
  // the final lm_head
  int C = conf->embed_dim;

  READ_LINEAR(model->embedding, fp, C, conf->vocab_size, conf->quant,
              "model.embedding", goto fail;);
  CALLOC(model->layers, conf->n_layers, "model.layers", goto fail;);  // NOLINT

  int Cq  = conf->n_heads * conf->head_dim;
  int Ckv = conf->n_kv_heads * conf->head_dim;

  // Read all the layers
  for (int l = 0; l < conf->n_layers; l++)
  {
    char layer_name[64];
    snprintf(layer_name, sizeof(layer_name), "model.layers.%d", l);
    GemmaDecoderLayer *layer = NULL;
    CALLOC(layer, 1, layer_name, goto fail;);

    char wq_name[64], wk_name[64], wv_name[64], wo_name[64];
    snprintf(wq_name, sizeof(wq_name), "model.layers.%d.wq", l);
    snprintf(wk_name, sizeof(wk_name), "model.layers.%d.wk", l);
    snprintf(wv_name, sizeof(wv_name), "model.layers.%d.wv", l);
    snprintf(wo_name, sizeof(wo_name), "model.layers.%d.wo", l);

    // Attention weights
    READ_LINEAR(layer->wq, fp, C, Cq, conf->quant, wq_name, goto fail;);
    READ_LINEAR(layer->wk, fp, C, Ckv, conf->quant, wk_name, goto fail;);
    READ_LINEAR(layer->wv, fp, C, Ckv, conf->quant, wv_name, goto fail;);
    READ_LINEAR(layer->wo, fp, Cq, C, conf->quant, wo_name, goto fail;);

    if (conf->qk_norm)
    {
      char nq_name[64], nk_name[64];
      snprintf(nq_name, sizeof(nq_name), "model.layers.%d.nq", l);
      snprintf(nk_name, sizeof(nk_name), "model.layers.%d.nk", l);
      READ_TENSOR(layer->nq, conf->head_dim, fp, nq_name, goto fail;);
      READ_TENSOR(layer->nk, conf->head_dim, fp, nk_name, goto fail;);
    }
    else
    {
      layer->nq = NULL;
      layer->nk = NULL;
    }

    char w1_name[64], w2_name[64], w3_name[64];
    snprintf(w1_name, sizeof(w1_name), "model.layers.%d.w1", l);
    snprintf(w2_name, sizeof(w2_name), "model.layers.%d.w2", l);
    snprintf(w3_name, sizeof(w3_name), "model.layers.%d.w3", l);

    // Feedforward weights
    READ_LINEAR(
        layer->w1, fp, C, conf->mlp_dim, conf->quant, w1_name, goto fail;);
    READ_LINEAR(
        layer->w2, fp, C, conf->mlp_dim, conf->quant, w2_name, goto fail;);
    READ_LINEAR(
        layer->w3, fp, conf->mlp_dim, C, conf->quant, w3_name, goto fail;);

    char n1_name[64], n2_name[64];
    snprintf(n1_name, sizeof(n1_name), "model.layers.%d.n1", l);
    snprintf(n2_name, sizeof(n2_name), "model.layers.%d.n2", l);

    // RMSNorm weights
    READ_TENSOR(layer->n1, C, fp, n1_name, goto fail;);
    READ_TENSOR(layer->n2, C, fp, n2_name, goto fail;);

    if (conf->pre_mlp_norm)
    {
      char n3_name[64];
      snprintf(n3_name, sizeof(n3_name), "model.layers.%d.n3", l);
      READ_TENSOR(layer->n3, C, fp, n3_name, goto fail;);
    }
    else { layer->n3 = NULL; }
    if (conf->post_mlp_norm)
    {
      char n4_name[64];
      snprintf(n4_name, sizeof(n4_name), "model.layers.%d.n4", l);
      READ_TENSOR(layer->n4, C, fp, n4_name, goto fail;);
    }
    else { layer->n4 = NULL; }
    model->layers[l] = layer;
  }
  READ_TENSOR(model->final_norm, C, fp, "model.final_norm", goto fail;);

  if (use_mm)
  {
    int P  = vconf->patch_size;
    int VC = vconf->hidden_dim;

    READ_TENSOR(enc->patch_emb, VC * 3 * P * P, fp,
                "model.vision_enc.patch_emb", goto fail;);
    READ_TENSOR(
        enc->patch_emb_b, VC, fp, "model.vision_enc.patch_emb_b", goto fail;);
    int n_patches = vconf->image_size / P;
    n_patches *= n_patches;
    // Same as here, the real shape is (n_patches, VC)
    READ_LINEAR(enc->pos_embedding, fp, VC, n_patches, conf->quant,
                "model.vision_enc.pos_embedding", goto fail;);
    CALLOC(enc->layers, vconf->n_layers, "model.vision_enc.layers",  // NOLINT
           goto fail;);

    // Read all the layers of ViT
    for (int l = 0; l < vconf->n_layers; l++)
    {
      char layer_name[64];
      snprintf(layer_name, sizeof(layer_name), "model.vision_enc.layers.%d", l);
      SigLIPEncoderLayer *layer = NULL;
      CALLOC(layer, 1, layer_name, goto fail;);

      // First layernorm
      char n1_name[64], n1b_name[64];
      snprintf(n1_name, sizeof(n1_name), "model.vision_enc.layers.%d.n1", l);
      snprintf(
          n1b_name, sizeof(n1b_name), "model.vision_enc.layers.%d.n1_b", l);
      READ_TENSOR(layer->n1, VC, fp, n1_name, goto fail;);
      READ_TENSOR(layer->n1_b, VC, fp, n1b_name, goto fail;);

      // Attention weights
      char wq_name[64], wk_name[64], wv_name[64], wo_name[64];
      snprintf(wq_name, sizeof(wq_name), "model.vision_enc.layers.%d.wq", l);
      snprintf(wk_name, sizeof(wk_name), "model.vision_enc.layers.%d.wk", l);
      snprintf(wv_name, sizeof(wv_name), "model.vision_enc.layers.%d.wv", l);
      snprintf(wo_name, sizeof(wo_name), "model.vision_enc.layers.%d.wo", l);

      READ_LINEAR(layer->wq, fp, VC, VC, conf->quant, wq_name, goto fail;);
      READ_LINEAR(layer->wk, fp, VC, VC, conf->quant, wk_name, goto fail;);
      READ_LINEAR(layer->wv, fp, VC, VC, conf->quant, wv_name, goto fail;);
      READ_LINEAR(layer->wo, fp, VC, VC, conf->quant, wo_name, goto fail;);

      // Attention biases
      char bq_name[64], bk_name[64], bv_name[64], bo_name[64];
      snprintf(bq_name, sizeof(bq_name), "model.vision_enc.layers.%d.bq", l);
      snprintf(bk_name, sizeof(bk_name), "model.vision_enc.layers.%d.bk", l);
      snprintf(bv_name, sizeof(bv_name), "model.vision_enc.layers.%d.bv", l);
      snprintf(bo_name, sizeof(bo_name), "model.vision_enc.layers.%d.bo", l);

      READ_TENSOR(layer->bq, VC, fp, bq_name, goto fail;);
      READ_TENSOR(layer->bk, VC, fp, bk_name, goto fail;);
      READ_TENSOR(layer->bv, VC, fp, bv_name, goto fail;);
      READ_TENSOR(layer->bo, VC, fp, bo_name, goto fail;);

      // Second layernorm
      char n2_name[64], n2b_name[64];
      snprintf(n2_name, sizeof(n2_name), "model.vision_enc.layers.%d.n2", l);
      snprintf(
          n2b_name, sizeof(n2b_name), "model.vision_enc.layers.%d.n2_b", l);

      READ_TENSOR(layer->n2, VC, fp, n2_name, goto fail;);
      READ_TENSOR(layer->n2_b, VC, fp, n2b_name, goto fail;);

      // Feedforward weights
      char w1_name[64], w2_name[64];
      snprintf(w1_name, sizeof(w1_name), "model.vision_enc.layers.%d.w1", l);
      snprintf(w2_name, sizeof(w2_name), "model.vision_enc.layers.%d.w2", l);
      READ_LINEAR(
          layer->w1, fp, VC, vconf->mlp_dim, conf->quant, w1_name, goto fail;);
      READ_LINEAR(
          layer->w2, fp, vconf->mlp_dim, VC, conf->quant, w2_name, goto fail;);

      // Feedforward biases
      char b1_name[64], b2_name[64];
      snprintf(b1_name, sizeof(b1_name), "model.vision_enc.layers.%d.b1", l);
      snprintf(b2_name, sizeof(b2_name), "model.vision_enc.layers.%d.b2", l);
      READ_TENSOR(layer->b1, vconf->mlp_dim, fp, b1_name, goto fail;);
      READ_TENSOR(layer->b2, VC, fp, b2_name, goto fail;);

      enc->layers[l] = layer;
    }

    // Post layernorm
    READ_TENSOR(
        enc->post_norm, VC, fp, "model.vision_enc.post_norm", goto fail;);
    READ_TENSOR(
        enc->post_norm_b, VC, fp, "model.vision_enc.post_norm_b", goto fail;);

    // Soft embedding RMSNorm
    READ_TENSOR(enc->norm, VC, fp, "model.vision_enc.norm", goto fail;);
    // Final projection
    READ_LINEAR(
        enc->proj, fp, VC, C, conf->quant, "model.vision_enc.proj", goto fail;);
  }

  fclose(fp);
  return model;

fail:
  if (fp != NULL) fclose(fp);
  free(att_layers_buf);
  if (model != NULL) { free_model(model); }
  else
  {
    free_siglip(enc);
    free_config(conf);
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
static inline _Float16
clamp_fp16(_Float16 v)
{
  return (_Float16)fminf(FP16_MAX, fmaxf((float)-FP16_MAX, (float)v));
}
#ifdef __clang__
#  pragma float_control(pop)
#endif

/* Gemma-style RMSNorm: (x * rsqrt(mean(x²) + eps)) * (weight + 1) */
static void
rmsnorm(_Float16 *dst, _Float16 *src, _Float16 *weight, int dim, float eps)
{
  float sqsum = 0.0f;
  #pragma omp simd reduction(+ : sqsum)  // Only using SIMD here
  for (int i = 0; i < dim; i++) { sqsum += (float)src[i] * (float)src[i]; }
  float rms = 1.0f / sqrtf(sqsum / (float)dim + eps);
  #pragma omp simd
  for (int i = 0; i < dim; i++)
  {
    // Gemma uses (weight + 1) instead of (weight)
    dst[i] = (_Float16)((float)src[i] * rms * (float)(weight[i] + 1));
  }
}

/* Gemma-style RMSNorm with OpenMP parallel reduction for larger dims */
static void
rmsnorm_omp(_Float16 *dst, _Float16 *src, _Float16 *weight, int dim, float eps)
{
  // OpenMP parallelized version of RMSNorm
  float sqsum = 0.0f;
  #pragma omp parallel for reduction(+ : sqsum)
  for (int i = 0; i < dim; i++) { sqsum += (float)src[i] * (float)src[i]; }
  float rms = 1.0f / sqrtf(sqsum / (float)dim + eps);
  #pragma omp simd
  for (int i = 0; i < dim; i++)
  {
    // Gemma uses (weight + 1) instead of (weight)
    dst[i] = (_Float16)((float)src[i] * rms * (float)(weight[i] + 1));
  }
}

/* Classic LayerNorm used inside the SigLIP vision tower */
static void
layernorm(_Float16 *dst,
    _Float16       *src,
    _Float16       *weight,
    _Float16       *bias,

    int   dim,
    float eps)
{
  float mean = 0.0f;
  for (int i = 0; i < dim; i++) { mean += (float)src[i]; }
  mean /= (float)dim;

  float var = 0.0f;
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
    dst[i] = (_Float16)(((float)src[i] - mean) * inv_std * (float)weight[i] +
                        (float)bias[i]);
  }
}

/* Symmetric per-vector quantization into int8 [-127, 127] */
static _Float16
quantize_act(int8_t *dst, _Float16 *vec, int dim)
{
  _Float16 amax = 0.0f;  // Find the abs max in vec
                         #pragma omp parallel for reduction(max : amax)
  for (int d = 0; d < dim; d++)
  {
    _Float16 av = vec[d] >= 0 ? vec[d] : -vec[d];
    if (av > amax) { amax = av; }
  }

  // 1.0f just to make sure we are not dividing by zero
  float qscale = (float)amax > 0.0f ? (float)amax / 127.0f : 1.0f;

  // Quantize vec into qbuf
  #pragma omp simd
  for (int d = 0; d < dim; d++)
  {
    int q = (int)roundf((float)vec[d] / qscale);
    if (q > 127) { q = 127; }
    else if (q < -127) { q = -127; }
    dst[d] = (int8_t)q;
  }

  return (_Float16)qscale;
}

/* Symmetric int8 quantization for a matrix of rows (used in vision GEMMs) */
static void
quantize_act_rows(int8_t *restrict dst,
    const _Float16 *restrict src,

    int m,
    int n,

    _Float16 *restrict scales_rows)
{
  #pragma omp parallel for
  for (int r = 0; r < m; r++)
  {
    const _Float16 *row  = src + r * n;
    _Float16        amax = 0.0f;
    for (int c = 0; c < n; c++)
    {
      _Float16 av = row[c] >= 0 ? row[c] : -row[c];
      if (av > amax) { amax = av; }
    }
    float qscale   = (float)amax > 0.0f ? (float)amax / 127.0f : 1.0f;
    scales_rows[r] = (_Float16)qscale;

    #pragma omp simd
    for (int c = 0; c < n; c++)
    {
      int q = (int)roundf((float)row[c] / qscale);
      if (q > 127) { q = 127; }
      else if (q < -127) { q = -127; }
      dst[r * n + c] = (int8_t)q;
    }
  }
}

/* fp16 matrix-vector multiply */
static void
gemv_fp16(_Float16 *restrict dst,
    const Linear *restrict mat,
    _Float16 *restrict vec,

    int m,
    int n)
{
  // fp16 mat (m, n) @ fp16 vec (n,) = fp16 dst (m,)
  #pragma omp parallel for
  for (int i = 0; i < m; i++)
  {
    float sum = 0;
    for (int j = 0; j < n; j++)
    {
      sum += (float)mat->fp16[i * n + j] * (float)vec[j];
    }
    dst[i] = clamp_fp16((_Float16)sum);
  }
}

/* int8 matrix-vector multiply + dequant */
static void
gemv_int8(_Float16 *restrict dst,
    const Linear *restrict mat,
    int8_t *restrict vec,

    int m,
    int n,

    _Float16 qscale)
{
  float fqscale = (float)qscale;

  // int8 mat (m, n) @ int8 vec (n,) = fp16 dst (m,)
  #pragma omp parallel for
  for (int i = 0; i < m; i++)
  {
    int32_t sum = 0;
    for (int j = 0; j < n; j++)
    {
      sum += (int32_t)mat->i8.q[i * n + j] * (int32_t)vec[j];
    }
    // Dequantize
    float val = (float)sum * fqscale * (float)mat->i8.scales[i];
    dst[i]    = clamp_fp16((_Float16)val);
  }
}

/* fp16 matrix-matrix (used by vision tower) */
static void
gemm_fp16(_Float16 *dst, const Linear *mat, _Float16 *src, int m, int n, int k)
{
  // fp16 src (m, k) @ (fp16 mat (n, k)).T = fp16 dst (m, n)
  if (k <= 0) return;

  #pragma omp parallel for
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
    {
      float           sum     = 0.0f;
      const _Float16 *src_row = src + i * k;
      const _Float16 *w_row   = mat->fp16 + j * k;
      #pragma omp simd reduction(+ : sum)
      for (int l = 0; l < k; l++)
      {
        sum += (float)src_row[l] * (float)w_row[l];
      }
      dst[i * n + j] = clamp_fp16((_Float16)sum);
    }
}

/* int8 matrix-matrix + dequant */
static void
gemm_int8(_Float16 *restrict dst,
    const Linear *restrict mat,
    const int8_t *restrict src,

    int m,
    int n,
    int k,

    const _Float16 *restrict scales_rows)
{
  // int8 src (m, k) @ (int8 mat (n, k)).T = fp16 dst (m, n)
  if (k <= 0) return;

  #pragma omp parallel for
  for (int i = 0; i < m; i++)
  {
    float fscale = (float)scales_rows[i];
    for (int j = 0; j < n; j++)
    {
      int32_t       sum     = 0;
      const int8_t *src_row = src + i * k;
      const int8_t *w_row   = mat->i8.q + j * k;

      #pragma omp simd reduction(+ : sum)
      for (int l = 0; l < k; l++)
      {
        sum += (int32_t)src_row[l] * (int32_t)w_row[l];
      }

      // Dequantize
      float val      = (float)sum * fscale * (float)mat->i8.scales[j];
      dst[i * n + j] = clamp_fp16((_Float16)val);
    }
  }
}

static _Float16
dot(_Float16 *v1, _Float16 *v2, int dim)
{
  // Must use float instead of _Float16
  float sum = 0.0f;
  for (int i = 0; i < dim; i++) { sum += (float)v1[i] * (float)v2[i]; }
  return (_Float16)sum;
}

static void
softmax(_Float16 *dst, _Float16 *src, int dim)
{
  if (dim <= 0) return;

  _Float16 max = -(_Float16)INFINITY;
  for (int i = 0; i < dim; i++)
  {
    if (src[i] > max) { max = src[i]; }
  }

  float expsum = 0.0f;
  for (int i = 0; i < dim; i++)
  {
    float val = expf((float)(src[i] - max));
    dst[i]    = (_Float16)val;
    expsum += val;
  }

  #pragma omp simd
  for (int i = 0; i < dim; i++) { dst[i] = (_Float16)((float)dst[i] / expsum); }
}

static void
softmax_omp(_Float16 *dst, _Float16 *src, int dim)
{
  // OpenMP parallelized version of softmax
  _Float16 max = -(_Float16)INFINITY;
  #pragma omp parallel for reduction(max : max)
  for (int i = 0; i < dim; i++)
  {
    if (src[i] > max) { max = src[i]; }
  }

  float expsum = 0.0f;
  #pragma omp parallel for reduction(+ : expsum)
  for (int i = 0; i < dim; i++)
  {
    float val = expf((float)(src[i] - max));
    dst[i]    = (_Float16)val;
    expsum += val;
  }

  #pragma omp parallel for
  for (int i = 0; i < dim; i++) { dst[i] = (_Float16)((float)dst[i] / expsum); }
}

/* Optional debug helper, prints stats when activations look suspicious */
static void
warn_stats(const char *name,
    const _Float16    *data,

    int len,
    int layer,
    int pos,
    int freq)
{
  // To keep the compiler happy
  (void)name;
  (void)data;
  (void)len;
  (void)layer;
  (void)pos;
  (void)freq;

#ifdef DEBUG
  if (freq <= 0) return;
  if (layer % freq != 0) return;

  float  min = INFINITY, max = -INFINITY;
  double sum = 0.0, sumsq = 0.0;
  int    inf_cnt = 0, nan_cnt = 0;

  for (int i = 0; i < len; i++)
  {
    float v = (float)data[i];
    if (isinf(v)) { inf_cnt++; }
    if (isnan(v)) { nan_cnt++; }
    if (v < min) { min = v; }
    if (v > max) { max = v; }
    sum += v;
    sumsq += (double)v * v;
  }

  if (max >= 0.8 * FP16_MAX || min <= -0.8 * FP16_MAX || inf_cnt >= 1 ||
      nan_cnt >= 1)
  {
    double mean = sum / len;
    double var  = sumsq / len - mean * mean;
    double std  = sqrt(var > 0 ? var : 0);

    printf(
        "\nWARNING: [L%02d P%04d] %-20s: min=%9.3f max=%9.3f mean=%9.3f "
        "std=%9.3f inf=%d nan=%d\n",
        layer, pos, name, min, max, mean, std, inf_cnt, nan_cnt);
  }
#endif
}

/* ------------------------------------------------------------------ */
/* Forward functions                                                  */
/* ------------------------------------------------------------------ */

/* Vision forward pass (SigLIP) */
void
forward_siglip(SigLIPVisionEncoder *enc,

    GemmaConfig  *conf,
    SigLIPBuffer *buf,
    _Float16     *img,
    _Float16     *out)
{
  // TODO: Integrate this into the language model (tons and tons of work to
  // do...)
  SigLIPConfig *vconf = enc->config;

  int C        = vconf->hidden_dim;
  int P        = vconf->patch_size;
  int ppi      = vconf->image_size / P;
  int N        = ppi * ppi;
  int tpi      = conf->image_toks;
  int side_len = (int)roundf(sqrtf((float)tpi));
  int K        = (vconf->image_size / vconf->patch_size) / side_len;

  int CH = C / vconf->n_heads;
  int CM = vconf->mlp_dim;

  // Patch Embedding
  int in_dim = 3 * P * P;

  for (int oy = 0; oy < ppi; oy++)
    for (int ox = 0; ox < ppi; ox++)
    {
      int token_idx = oy * ppi + ox;
      for (int oc = 0; oc < C; oc++)
      {
        float sum = 0.0f;
        for (int c = 0; c < 3; c++)
          for (int py = 0; py < P; py++)
            for (int px = 0; px < P; px++)
            {
              int in_idx = (c * vconf->image_size * vconf->image_size +
                            (oy * P + py) * vconf->image_size + (ox * P + px));
              int w_idx  = oc * in_dim + c * P * P + py * P + px;
              sum += (float)enc->patch_emb[w_idx] * (float)img[in_idx];
            }
        buf->x[token_idx * C + oc] =
            clamp_fp16((_Float16)(sum + (float)enc->patch_emb_b[oc]));
      }
    }

  warn_stats("patch_emb", buf->x, N * C, 0, 0, 0);

  // Position Embedding
  if (!conf->quant)
  {
    for (int i = 0; i < N; i++)
    {
      _Float16 *pos_vec = enc->pos_embedding->fp16 + i * C;
      for (int j = 0; j < C; j++)
      {
        buf->x[i * C + j] = clamp_fp16(buf->x[i * C + j] + pos_vec[j]);
      }
    }
  }
  else
  {
    // Dequantize per row
    for (int i = 0; i < N; i++)
    {
      float scale = (float)enc->pos_embedding->i8.scales[i];
      for (int j = 0; j < C; j++)
      {
        float val         = (float)enc->pos_embedding->i8.q[i * C + j] * scale;
        buf->x[i * C + j] = clamp_fp16(buf->x[i * C + j] + (_Float16)val);
      }
    }
  }
  warn_stats("pos_emb", buf->x, N * C, 0, 0, 0);

  // Encoder Layers
  for (int l = 0; l < vconf->n_layers; l++)
  {
    SigLIPEncoderLayer *layer = enc->layers[l];

    memcpy(buf->resid, buf->x, N * C * sizeof(_Float16));
    layernorm(buf->x, buf->x, layer->n1, layer->n1_b, C, vconf->eps);
    warn_stats("ln1", buf->x, N * C, l, 0, 8);

    // QKV projections
    if (!conf->quant)
    {
      gemm_fp16(buf->xq, layer->wq, buf->x, N, C, C);
      gemm_fp16(buf->xk, layer->wk, buf->x, N, C, C);
      gemm_fp16(buf->xv, layer->wv, buf->x, N, C, C);
    }
    else
    {
      quantize_act_rows(buf->x_i8, buf->x, N, C, buf->x_scales);
      gemm_int8(buf->xq, layer->wq, buf->x_i8, N, C, C, buf->x_scales);
      gemm_int8(buf->xk, layer->wk, buf->x_i8, N, C, C, buf->x_scales);
      gemm_int8(buf->xv, layer->wv, buf->x_i8, N, C, C, buf->x_scales);
    }
    warn_stats("q", buf->xq, N * C, l, 0, 8);
    warn_stats("k", buf->xk, N * C, l, 0, 8);
    warn_stats("v", buf->xv, N * C, l, 0, 8);

    // Add biases
    for (int i = 0; i < N; i++)
      for (int j = 0; j < C; j++)
      {
        int idx = i * C + j;
        buf->xq[idx] += layer->bq[j];
        buf->xk[idx] += layer->bk[j];
        buf->xv[idx] += layer->bv[j];
      }

    // Attention
    memset(buf->att_out, 0, N * C * sizeof(_Float16));
    float scale = 1.0f / sqrtf((float)CH);

    #pragma omp parallel for
    for (int h = 0; h < vconf->n_heads; h++)
    {
      _Float16 *scores = buf->scores + h * N * N;  // (N, N) for this head

      // Compute scores
      for (int i = 0; i < N; i++)
      {
        _Float16 *q_vec = buf->xq + i * C + h * CH;
        for (int j = 0; j < N; j++)
        {
          _Float16 *k_vec = buf->xk + j * C + h * CH;
          float     dot   = 0.0f;
          for (int d = 0; d < CH; d++)
          {
            dot += (float)q_vec[d] * (float)k_vec[d];
          }
          scores[i * N + j] = (_Float16)(dot * scale);
        }
        // Softmax over j
        softmax(scores + i * N, scores + i * N, N);
      }

      // Weighted sum of values
      for (int i = 0; i < N; i++)
      {
        _Float16 *out_vec = buf->att_out + i * C + h * CH;
        for (int d = 0; d < CH; d++)
        {
          float sum = 0.0f;
          for (int j = 0; j < N; j++)
          {
            _Float16 *v_vec = buf->xv + j * C + h * CH;
            sum += (float)v_vec[d] * (float)scores[i * N + j];
          }
          out_vec[d] = (_Float16)sum;
        }
      }
    }

    // Output projection
    if (!conf->quant) { gemm_fp16(buf->x, layer->wo, buf->att_out, N, C, C); }
    else
    {
      quantize_act_rows(buf->x_i8, buf->att_out, N, C, buf->x_scales);
      gemm_int8(buf->x, layer->wo, buf->x_i8, N, C, C, buf->x_scales);
    }
    // Add output bias
    for (int i = 0; i < N; i++)
      for (int j = 0; j < C; j++) { buf->x[i * C + j] += layer->bo[j]; }
    warn_stats("att_out", buf->x, N * C, l, 0, 8);

    // Residual connection
    for (int i = 0; i < N * C; i++)
    {
      buf->x[i] = clamp_fp16(buf->x[i] + buf->resid[i]);
    }
    warn_stats("resid1", buf->x, N * C, l, 0, 8);

    memcpy(buf->resid, buf->x, N * C * sizeof(_Float16));
    layernorm(buf->x, buf->x, layer->n2, layer->n2_b, C, vconf->eps);
    warn_stats("ln2", buf->x, N * C, l, 0, 8);

    // x @ fc1 = mlp_hidden
    if (!conf->quant)
    {
      gemm_fp16(buf->mlp_hidden, layer->w1, buf->x, N, CM, C);
    }
    else
    {
      quantize_act_rows(buf->x_i8, buf->x, N, C, buf->x_scales);
      gemm_int8(buf->mlp_hidden, layer->w1, buf->x_i8, N, CM, C, buf->x_scales);
    }

    for (int i = 0; i < N; i++)
      for (int j = 0; j < CM; j++)
      {
        // Apply fc1 biases
        float val = (float)buf->mlp_hidden[i * CM + j] + (float)layer->b1[j];
        // GELU tanh approximation
        float c = 0.79788456080287f;
        val     = 0.5f * val *
              (1.0f + tanhf(c * (val + 0.044715f * val * val * val)));
        buf->mlp_hidden[i * CM + j] = (_Float16)val;
      }
    warn_stats("mlp_hidden", buf->mlp_hidden, N * CM, l, 0, 8);

    // mlp_hidden @ fc2 = x
    if (!conf->quant)
    {
      gemm_fp16(buf->x, layer->w2, buf->mlp_hidden, N, C, CM);
    }
    else
    {
      quantize_act_rows(buf->mlp_i8, buf->mlp_hidden, N, CM, buf->mlp_scales);
      gemm_int8(buf->x, layer->w2, buf->mlp_i8, N, C, CM, buf->mlp_scales);
    }
    // x += b2
    for (int i = 0; i < N; i++)
      for (int j = 0; j < C; j++) { buf->x[i * C + j] += layer->b2[j]; }
    warn_stats("mlp_out", buf->x, N * C, l, 0, 8);
    // Residual connection
    for (int i = 0; i < N * C; i++)
    {
      buf->x[i] = clamp_fp16(buf->x[i] + buf->resid[i]);
    }
    warn_stats("resid2", buf->x, N * C, l, 0, 8);
  }

  // Post-norm + average pooling down to image_toks tokens
  layernorm(buf->x, buf->x, enc->post_norm, enc->post_norm_b, C, vconf->eps);
  warn_stats("post_ln", buf->x, N * C, 0, 0, 0);

  // Average pooling
  for (int oy = 0; oy < side_len; oy++)
    for (int ox = 0; ox < side_len; ox++)
    {
      int out_idx = (oy * side_len + ox) * C;
      for (int d = 0; d < C; d++)
      {
        float sum = 0.0f;
        for (int ky = 0; ky < K; ky++)
          for (int kx = 0; kx < K; kx++)
          {
            int py        = oy * K + ky;
            int px        = ox * K + kx;
            int token_idx = (py * ppi + px) * C + d;
            sum += (float)buf->x[token_idx];
          }
        buf->x[out_idx + d] = (_Float16)(sum / (K * K));
      }
    }
  warn_stats("avg_pool", buf->x, tpi * C, 0, 0, 0);
  // buf->x now becomes (tpi, C)

  // RMSNorm
  rmsnorm(buf->x, buf->x, enc->norm, C, vconf->eps);
  warn_stats("rmsnorm", buf->x, tpi * C, 0, 0, 0);

  // Final projection into language-model embedding space
  if (!conf->quant)
  {
    gemm_fp16(out, enc->proj, buf->x, tpi, conf->embed_dim, C);
  }
  else
  {
    quantize_act_rows(buf->x_i8, buf->x, tpi, C, buf->x_scales);
    gemm_int8(
        out, enc->proj, buf->x_i8, tpi, conf->embed_dim, C, buf->x_scales);
  }
  warn_stats("proj", out, tpi * conf->embed_dim, 0, 0, 0);
}

/* Language model forward (one token) */
int
forward_gemma(GemmaModel *model, GemmaBuffer *buf, int tok, int pos)
{
  GemmaConfig *conf = model->config;

  int C   = conf->embed_dim;
  int NH  = conf->n_heads;
  int CH  = conf->head_dim;
  int Cq  = NH * CH;
  int Ckv = conf->n_kv_heads * CH;

  int CH_half = CH / 2;

  if (pos >= buf->cache_len)
  {
    fprintf(stderr, "\nError: KV Cache is full");
    return 1;
  }

  /* Embedding lookup.
   * The export script already applied 1/sqrt(embed_dim), so the usual
   * Gemma sqrt(embed_dim) scale cancels out to 1.0 here. */
  _Float16 embed_scale =
      1.0f;  // equivalent to sqrt(embed_dim) * (1/sqrt(embed_dim))
  if (conf->quant)
  {
    // Dequantize
    embed_scale *= model->embedding->i8.scales[tok];
  }

  // x = embedding[tok] * embed_scale
  #pragma omp parallel for
  for (int i = 0; i < C; i++)
  {
    if (!conf->quant)
    {
      buf->x[i] = model->embedding->fp16[tok * C + i] * embed_scale;
    }
    else
    {
      buf->x[i] = (_Float16)model->embedding->i8.q[tok * C + i] * embed_scale;
    }
  }

  warn_stats("embedding", buf->x, C, 0, pos, 0);

  // Precompute cos & sin for all frequencies (used in RoPE)
  #pragma omp simd
  for (int d = 0; d < CH_half; d++)
  {
    float freq = 0.0f;
    float e    = (float)(-2 * d) / (float)CH;

    // Rotation angles for sliding window attentions
    freq                         = powf(conf->local_theta, e);
    buf->csfreqs_slid[d * 2]     = (_Float16)cosf(freq * (float)pos);
    buf->csfreqs_slid[d * 2 + 1] = (_Float16)sinf(freq * (float)pos);

    // Rotation angles for full attentions
    freq                         = powf(conf->global_theta, e);
    buf->csfreqs_full[d * 2]     = (_Float16)cosf(freq * (float)pos);
    buf->csfreqs_full[d * 2 + 1] = (_Float16)sinf(freq * (float)pos);
  }

  _Float16 att_scale = (_Float16)(1.0f / sqrtf((float)conf->q_scale));

  for (int l = 0; l < conf->n_layers; l++)
  {
    GemmaDecoderLayer *layer = model->layers[l];

    memcpy(buf->resid, buf->x, C * sizeof(_Float16));

    rmsnorm_omp(buf->x, buf->x, layer->n1, C, conf->eps);
    warn_stats("norm1", buf->x, C, l, pos, 8);

    // The attention block
    if (!conf->quant)
    {
      gemv_fp16(buf->xq, layer->wq, buf->x, Cq, C);   // (n_heads, head_dim)
      gemv_fp16(buf->xk, layer->wk, buf->x, Ckv, C);  // (n_kv_heads, head_dim)
      gemv_fp16(buf->xv, layer->wv, buf->x, Ckv, C);  // (n_kv_heads, head_dim)
    }
    else
    {
      float xscale = (float)quantize_act(buf->x_i8, buf->x, C);
      gemv_int8(buf->xq, layer->wq, buf->x_i8, Cq, C, xscale);
      gemv_int8(buf->xk, layer->wk, buf->x_i8, Ckv, C, xscale);
      gemv_int8(buf->xv, layer->wv, buf->x_i8, Ckv, C, xscale);
    }

    if (conf->qk_norm)
    {
      // Query RMSNorm
      for (int h = 0; h < NH; h++)
      {
        _Float16 *xq_head = buf->xq + h * CH;
        // Use the non-threading version here since we are running this over
        // every head
        rmsnorm(xq_head, xq_head, layer->nq, CH, conf->eps);
      }
      // Key RMSNorm
      for (int h = 0; h < conf->n_kv_heads; h++)
      {
        _Float16 *xk_head = buf->xk + h * CH;
        rmsnorm(xk_head, xk_head, layer->nk, CH, conf->eps);
      }
    }

    bool      is_local = conf->att_layers[l];
    _Float16 *freqs_cs = is_local ? buf->csfreqs_slid : buf->csfreqs_full;

    // These used to be a single merged loop, but I splitted it into two
    // separate loops for higher performance

    // Apply RoPE to queries
    #pragma omp parallel for
    for (int h = 0; h < NH; h++)
    {
      int       o  = h * CH;  // Offset for current head
      _Float16 *xq = buf->xq + o;
      #pragma omp simd
      for (int i = 0; i < CH_half; i++)
      {
        float cfr = (float)freqs_cs[2 * i];
        float sfr = (float)freqs_cs[2 * i + 1];
        float a   = (float)xq[i];            // Index in the first half vector
        float b   = (float)xq[i + CH_half];  // Index in the second half vector

        xq[i]           = (_Float16)(a * cfr - b * sfr);
        xq[i + CH_half] = (_Float16)(a * sfr + b * cfr);
      }
    }

    // Apply RoPE to keys
    #pragma omp parallel for
    for (int h = 0; h < conf->n_kv_heads; h++)
    {
      int       o  = h * CH;  // Offset for current head
      _Float16 *xk = buf->xk + o;
      #pragma omp simd
      for (int i = 0; i < CH_half; i++)
      {
        float cfr = (float)freqs_cs[2 * i];
        float sfr = (float)freqs_cs[2 * i + 1];
        float a   = (float)xk[i];            // Index in the first half vector
        float b   = (float)xk[i + CH_half];  // Index in the second half vector

        xk[i]           = (_Float16)(a * cfr - b * sfr);
        xk[i + CH_half] = (_Float16)(a * sfr + b * cfr);
      }
    }

    int entry_sz = conf->n_kv_heads * CH;
    int layer_sz = 2 * buf->cache_len * entry_sz;

    // (cache_len, n_kv_heads, head_dim)
    _Float16 *k_cache = buf->kv_cache + l * layer_sz;
    _Float16 *v_cache = k_cache + buf->cache_len * entry_sz;

    // Write to key & value cache
    memcpy(k_cache + pos * entry_sz, buf->xk, entry_sz * sizeof(_Float16));
    memcpy(v_cache + pos * entry_sz, buf->xv, entry_sz * sizeof(_Float16));

    // Sliding-window (true) or full attention (false)?
    bool local_att = is_local && pos >= conf->slide_len;
    // Starting position of attention
    int spos   = local_att ? (pos + 1 - conf->slide_len) : 0;
    int attlen = pos + 1 - spos;  // Include the current pos

    // Iterate over all the attention heads
    #pragma omp parallel for
    for (int h = 0; h < NH; h++)
    {
      int       h_kv    = h * conf->n_kv_heads / NH;  // GQA mapping
      _Float16 *xq_head = buf->xq + h * CH;           // xq[h, :]
      _Float16 *xk_head =
          k_cache + spos * entry_sz + h_kv * CH;  // k_cache[spos:, h_kv, :]
      _Float16 *att_head =
          buf->att + h * buf->cache_len + spos;  // att[h, spos:]

      // Compute dot product of the current query across all the keys
      for (int t = 0; t < attlen; t++)
      {
        att_head[t] = dot(xq_head, xk_head + t * entry_sz, CH) * att_scale;
      }

      // Attention score softcapping
      if (conf->att_softcap != 0.0f)
      {
        for (int t = 0; t < attlen; t++)
        {
          float val   = (float)att_head[t] / conf->att_softcap;
          att_head[t] = (_Float16)(tanhf(val) * conf->att_softcap);
        }
      }

      // Softmax
      softmax(att_head, att_head, attlen);

      // Compute output as weighted sum of values
      _Float16 *xv_head =
          v_cache + spos * entry_sz + h_kv * CH;  // v_cache[spos:, h_kv, :]
      _Float16 *xo_head = buf->xo + h * CH;
      // xo_head (head_dim,) = att_head (attlen,) @ xv_head (attlen,
      // head_dim)
      for (int d = 0; d < CH; d++)
      {
        float sum = 0.0f;
        #pragma omp simd reduction(+ : sum)
        for (int t = 0; t < attlen; t++)
        {
          sum += (float)(xv_head + t * entry_sz)[d] * (float)att_head[t];
        }
        xo_head[d] = sum;
      }
    }

    // Output projection maps xo back to x
    if (!conf->quant) { gemv_fp16(buf->x, layer->wo, buf->xo, C, Cq); }
    else
    {
      float xoscale = (float)quantize_act(buf->xo_i8, buf->xo, Cq);
      gemv_int8(buf->x, layer->wo, buf->xo_i8, C, Cq, xoscale);
    }
    warn_stats("attn_out", buf->x, C, l, pos, 8);

    rmsnorm_omp(buf->x, buf->x, layer->n2, C, conf->eps);
    warn_stats("norm2", buf->x, C, l, pos, 8);

    // Combine the residual stream
    _Float16 *restrict x     = buf->x;
    _Float16 *restrict resid = buf->resid;
    #pragma omp simd
    for (int d = 0; d < C; d++)
    {
      /* Sometimes the residual stream accumulates huge values on certain
       * channels, especially in pretrained/bigger models (Sun et al.
       * https://arxiv.org/abs/2402.17762) It works fine in fp32 or bf16, but it
       * can easily overflow fp16 and become inf, causing all the activations
       * turning into nan after the next RMSNorm, so we need to clamp it */

      /* NOTE: Actually this should never trigger now since I added activation
       * scalers afterwards (see export.py), the clamp here is more of a
       * last-resort safety net */
      buf->x[d] = clamp_fp16(x[d] + resid[d]);
    }
    warn_stats("resid1", buf->x, C, l, pos, 8);

    memcpy(buf->resid, buf->x, C * sizeof(_Float16));

    if (conf->pre_mlp_norm)
    {
      rmsnorm_omp(buf->x, buf->x, layer->n3, C, conf->eps);
    }

    // MLP feedforward layer (SwiGLU-style)
    if (!conf->quant)
    {
      gemv_fp16(buf->xg, layer->w2, buf->x, conf->mlp_dim, C);
      gemv_fp16(buf->xu, layer->w1, buf->x, conf->mlp_dim, C);
    }
    else
    {
      float xscale = (float)quantize_act(buf->x_i8, buf->x, C);
      gemv_int8(buf->xg, layer->w2, buf->x_i8, conf->mlp_dim, C, xscale);
      gemv_int8(buf->xu, layer->w1, buf->x_i8, conf->mlp_dim, C, xscale);
    }

    // GELU gate
    #pragma omp parallel for
    for (int d = 0; d < conf->mlp_dim; d++)
    {
      // Tanh approximation of GELU
      float x    = (float)buf->xg[d];
      float c    = 0.79788456080287f;  // = sqrt(2 / pi)
      x          = 0.5 * x * (1 + tanhf(c * (x + 0.044715 * x * x * x)));
      buf->xg[d] = (_Float16)x;
      buf->xg[d] *= buf->xu[d];  // Fuse xg * xu into xg
    }

    if (!conf->quant)
    {
      gemv_fp16(buf->x, layer->w3, buf->xg, C, conf->mlp_dim);
    }
    else
    {
      float xscale = (float)quantize_act(buf->xg_i8, buf->xg, conf->mlp_dim);
      gemv_int8(buf->x, layer->w3, buf->xg_i8, C, conf->mlp_dim, xscale);
    }
    warn_stats("down_proj", buf->x, C, l, pos, 8);

    if (conf->post_mlp_norm)
    {
      rmsnorm_omp(buf->x, buf->x, layer->n4, C, conf->eps);
      warn_stats("norm4", buf->x, C, l, pos, 8);
    }

    // Second residual
    x     = buf->x;
    resid = buf->resid;
    #pragma omp simd
    for (int d = 0; d < C; d++) { buf->x[d] = clamp_fp16(x[d] + resid[d]); }
    warn_stats("resid2", buf->x, C, l, pos, 8);
  }

  // Final RMSNorm (logits are computed later in the sampling loop)
  rmsnorm_omp(buf->x, buf->x, model->final_norm, C, conf->eps);
  return 0;
}

/* ------------------------------------------------------------------ */
/* Sampling helpers                                                   */
/* ------------------------------------------------------------------ */

static int
argmax(_Float16 *logits, int vocab_size)
{
  // Pick the index with the max value
  int      max_idx = -1;
  _Float16 max_val = -(_Float16)INFINITY;
  #pragma omp parallel
  {
    int      local_idx = -1;
    _Float16 local_val = -(_Float16)INFINITY;
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

typedef struct
{
  _Float16 val;
  int      idx;
} FloatIdx;

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

static int
partition_desc(FloatIdx *arr, int lo, int hi)
{
  // Pick the pivot randomly (use a seperate rand sequence)
  int r = (int)lo + (int)qs_rand() % (hi - lo + 1);
  swap_fi(&arr[r], &arr[hi]);

  _Float16 pivot = arr[hi].val;

  int i = lo;
  for (int j = lo; j < hi; j++)
  {
    // Put the greater one on the left
    if (arr[j].val > pivot) { swap_fi(&arr[i++], &arr[j]); }
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
    else if (p < k_idx) { lo = p + 1; }
    else { hi = p - 1; }
  }
}

static void
apply_topk(_Float16 *logits, FloatIdx *logit_indices, int vocab_size, int k)
{
  if (k <= 0) { k = 1; }
  if (k > vocab_size) { k = vocab_size; }

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
  for (int i = 0; i < vocab_size; i++) { logits[i] = -(_Float16)INFINITY; }
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

static void
build_heap(FloatIdx *arr, int n)
{
  for (int i = n / 2 - 1; i >= 0; i--) { sift_down(arr, n, i); }
}

static void
apply_topp(_Float16 *logits,
    _Float16        *fpbuf,
    FloatIdx        *logit_indices,

    int   vocab_size,
    int   k,
    float p)
{
  if (k > vocab_size) { k = vocab_size; }
  // Softmax to get the probs, store in fpbuf
  softmax_omp(fpbuf, logits, vocab_size);
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
    // topk typically uses values less than 100, no need to use omp
    // here
    for (int i = 0; i < k; i++)
    {
      int idx = logit_indices[i].idx;  // Reuse the candidates from topk
      logit_indices[i].val = fpbuf[idx];
    }
  }
  build_heap(logit_indices, heap_size);  // O(k)

  // fpbuf is now a copy of the original logits
  memcpy(fpbuf, logits, vocab_size * sizeof(_Float16));

  // Set logits to -inf
  #pragma omp parallel for
  for (int i = 0; i < vocab_size; i++) { logits[i] = -(_Float16)INFINITY; }

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
apply_rpen(_Float16 *logits, bool *visited, int vocab_size, float rpen)
{
  // rpen short for Repetition Penalty
  #pragma omp parallel for
  for (int i = 0; i < vocab_size; i++)
  {
    if (!visited[i]) continue;
    float val = (float)logits[i];
    if (val > 0.0f) { logits[i] = (_Float16)(val / rpen); }
    else { logits[i] = (_Float16)(val * rpen); }
  }
}

/* ------------------------------------------------------------------ */
/* Generation loop                                                    */
/* ------------------------------------------------------------------ */

/* */
void
sample(GemmaModel *model,
    GemmaBuffer   *buf,

    int  *tokens,
    int   seqlen,
    float temperature,
    int   topk,
    float topp,
    float rpen,

    int *(*token_callback)(int, GemmaTokenizer *, bool *))
{
  GemmaConfig *conf = model->config;

  int  vs   = conf->vocab_size;
  bool quit = false;

  // Boolean flags
  bool dosample = temperature != 0 && topk != 1;
  bool use_topk = dosample && topk != 0;
  bool use_topp = dosample && topp < 1.0f;
  bool use_rpen = dosample && rpen > 1.0f;

  clock_t prefill_start = 0;
  clock_t prefill_end   = 0;
  clock_t gen_start     = 0;

  int prompt_toks = 0;
  int gen_toks    = 0;
  int pos         = 0;
  int token       = 0;

  // bool array indicating which tokens have already been processed
  // Used in rpen (repetition penalty)
  bool *visited = NULL;
  if (use_rpen)
  {
    // Will expand to "... return ; ...". No return values
    CALLOC(visited, vs, "visited", goto end;);
  }

  // Prefill all the prompt tokens except the last one
  prefill_start = clock();
  pos           = 0;
  for (int *t = tokens; *t != EOF && *(t + 1) != EOF; t++)
  {
    if (g_interrupted)
    {
      if (use_rpen) { free(visited); }
      return;
    }
    if (use_rpen) { visited[*t] = true; }
    if (forward_gemma(model, buf, *t, pos++) == 1)
    {
      goto end;
    }  // forward with the current pos
  }
  prefill_end = clock();
  double prefill_elapsed =
      (double)(prefill_end - prefill_start) / CLOCKS_PER_SEC;
  prompt_toks = pos;

  token           = tokens[pos];
  _Float16 *probs = NULL;

  // Only allocate probs if needed
  if (temperature != 0.0f) { MALLOC(probs, vs, "probs", goto end;); }

  // Logit indices for topk & topp
  FloatIdx *logit_indices = NULL;
  if (use_topk || use_topp)
  {
    MALLOC(logit_indices, vs, "logit_indices", goto end;);
  }

  // Record tok/s
  gen_start = clock();
  gen_toks  = 0;

  for (; pos < seqlen; pos++)
  {
    if (g_interrupted) break;

    if (use_rpen) visited[token] = true;
    if (forward_gemma(model, buf, token, pos) == 1) { goto end; }

    // Compute logits from the final residual (tied embedding)
    if (!conf->quant)
    {
      gemv_fp16(buf->logits, model->embedding, buf->x, vs, conf->embed_dim);
    }
    else
    {
      float xscale = (float)quantize_act(buf->x_i8, buf->x, conf->embed_dim);
      gemv_int8(buf->logits, model->embedding, buf->x_i8, vs, conf->embed_dim,
          xscale);
    }

    if (!dosample)
    {
      // Argmax sampling
      token = argmax(buf->logits, vs);
    }
    else
    {
      // Logit softcapping
      if (conf->logit_softcap != 0.0f)
      {
        for (int d = 0; d < conf->vocab_size; d++)
        {
          float val      = (float)buf->logits[d] / conf->logit_softcap;
          buf->logits[d] = (_Float16)(tanhf(val) * conf->logit_softcap);
        }
      }

      // Apply the temperature
      #pragma omp parallel for
      for (int d = 0; d < vs; d++) { buf->logits[d] /= (_Float16)temperature; }

      if (use_topk) { apply_topk(buf->logits, logit_indices, vs, topk); }
      if (use_topp)
      {
        apply_topp(buf->logits, probs, logit_indices, vs, topk, topp);
      }
      if (use_rpen) { apply_rpen(buf->logits, visited, vs, rpen); }

      // Softmax to get the probs
      softmax_omp(probs, buf->logits, vs);

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
    int *ret = token_callback(token, model->tokenizer, &quit);

    if (quit) break;
    else if (ret != NULL)
    {
      // Injected a token array (ends with EOF)
      // Prefill all the tokens except the last one
      int i = 0;
      for (; (token = ret[i]) != EOF && ret[i + 1] != EOF; i++)
      {
        if (g_interrupted)
        {
          free(ret);
          break;
        }
        if (use_rpen) { visited[token] = true; }
        // Forward with the next pos
        if (forward_gemma(model, buf, token, ++pos) == 1)
        {
          free(ret);
          goto end;
        }
      }
      if (g_interrupted)
      {
        free(ret);
        break;
      }
      token = ret[i];  // The last element
      free(ret);
    }
    // ret == NULL: do nothing
  }

end:
  clock_t gen_end     = clock();
  double  gen_elapsed = (double)(gen_end - gen_start) / CLOCKS_PER_SEC;

  // Print prefilling speed
  if (prefill_elapsed > 0.0)
  {
    printf("\nPrompt processed %d tokens in %.2f seconds (%.2f tok/s)\n",
        prompt_toks, prefill_elapsed, prompt_toks / prefill_elapsed);
  }
  else { printf("\nPrompt processed %d tokens instantly\n", prompt_toks); }
  // Print generation speed
  if (gen_elapsed > 0.0)
  {
    printf("Generated %d tokens in %.2f seconds (%.2f tok/s)\n", gen_toks,
        gen_elapsed, gen_toks / gen_elapsed);
  }
  else { printf("Generated %d tokens instantly\n", gen_toks); }

  free(probs);
  if (use_rpen) free(visited);
  if (use_topk || use_topp) free(logit_indices);
}

/* ------------------------------------------------------------------ */
/* High-level generate / chat                                         */
/* ------------------------------------------------------------------ */

static int *
generate_callback(int token, GemmaTokenizer *tok, bool *quit)
{
  if (token == tok->eos || token == tok->eot) { *quit = true; }
  char byte_buf[2];
  printf("%s", decode(tok, token, byte_buf));
  fflush(stdout);
  return NULL;
}

int
generate(GemmaModel *model,
    GemmaBuffer     *buf,

    const char *prompt,
    int         seqlen,
    float       temperature,
    int         topk,
    float       topp,
    float       rpen)
{
  GemmaTokenizer *tok = model->tokenizer;

  printf("%s", prompt);

  int n_tokens = 1;
  int size     = (int)strlen(prompt) + 1;
  int tokens[size + 1];
  tokens[0] = tok->bos;
  encode(tok, prompt, tokens + 1, &n_tokens);
  tokens[n_tokens] = EOF;

  sample(model, buf, tokens, seqlen, temperature, topk, topp, rpen,
      generate_callback);

  return 0;
}

/* Build the Gemma chat template */
static int *
new_turn(GemmaTokenizer *tok, bool bos, bool *quit)
{
  if (bos) { printf("User: "); }
  else { printf("\nUser: "); }

  char user_prompt[65536];
  if (fgets(user_prompt, sizeof(user_prompt), stdin) == NULL)
  {
    fprintf(stderr, "Failed to read user input\n");
    *quit = true;
  }
  user_prompt[strcspn(user_prompt, "\n")] = '\0';

  /* Template:
   * [<bos>]<start_of_turn>user\n
   * {user text}<end_of_turn>\n
   * <start_of_turn>model\n */

  int n_tokens = 0;
  int size =
      (int)strlen("user") + (int)strlen(user_prompt) + (int)strlen("model") + 6;
  if (bos) size++;
  int *tokens = NULL;
  MALLOC(tokens, size + 1, "tokens", {
    *quit = true;
    return NULL;
  });

  if (bos) tokens[n_tokens++] = tok->bos;
  tokens[n_tokens++] = tok->sot;
  encode(tok, "user\n", tokens + n_tokens, &n_tokens);
  encode(tok, user_prompt, tokens + n_tokens, &n_tokens);
  tokens[n_tokens++] = tok->eot;
  tokens[n_tokens++] = get_token_idx(tok, "\n");
  tokens[n_tokens++] = tok->sot;
  encode(tok, "model\n", tokens + n_tokens, &n_tokens);
  tokens[n_tokens++] = EOF;

  printf("Model: ");
  return tokens;
}

static int *
chat_callback(int token, GemmaTokenizer *tok, bool *quit)
{
  if (token == tok->eos || token == tok->eot)
  {
    return new_turn(tok, false, quit);
  }
  char byte_buf[2];
  printf("%s", decode(tok, token, byte_buf));
  fflush(stdout);
  return NULL;
}

int
chat(GemmaModel *model,
    GemmaBuffer *buf,

    int   seqlen,
    float temperature,
    int   topk,
    float topp,
    float rpen)
{
  bool quit       = false;
  int *first_turn = new_turn(model->tokenizer, true, &quit);
  if (quit) { return 1; }
  sample(model, buf, first_turn, seqlen, temperature, topk, topp, rpen,
      chat_callback);
  free(first_turn);
  return 0;
}

/* ------------------------------------------------------------------ */
/* CLI helpers                                                        */
/* ------------------------------------------------------------------ */

static inline bool
safe_atoui(const char *str, unsigned int *result)
{
  if (str == NULL) { return false; }

  errno = 0;

  char     *endptr = NULL;
  long long val    = strtoll(str, &endptr, 10);

  // Invalid number
  if (endptr == str) { return false; }
  // Extra characters at the end
  if (*endptr != '\0') { return false; }
  // long overflow
  if (errno == ERANGE) { return false; }
  // uint overflow
  if (val < 0 || val > (long long)UINT_MAX) { return false; }

  // All passed
  *result = (unsigned int)val;
  return true;
}

static inline bool
safe_atof(const char *str, float *result)
{
  if (str == NULL) { return false; }

  errno = 0;

  char *endptr = NULL;
  float val    = strtof(str, &endptr);

  // Invalid number
  if (endptr == str) { return false; }
  // Extra characters at the end
  if (*endptr != '\0') { return false; }
  // Overflow
  if (errno == ERANGE) { return false; }
  // inf / nan
  if (isinf(val) || isnan(val)) { return false; }

  // All passed
  *result = val;
  return true;
}

void
print_usage(void)
{
  // clang-format off
  printf(
  "Usage:\n"
  "  ./gemma <modelfile> [options]\n"
  "\n"
  "Arguments:\n"
  "  modelfile              Path to the model file\n"
  "\n"
  "Options:\n"
  "  -l, --seqlen <N>       Set sequence length (default: 16384)\n"
  "  -k, --topk <N>         Set top-k sampling value (default: 0)\n"
  "  -s, --seed <N>         Set random seed\n"
  "                         (default: current time)\n"
  "  -t, --temperature <F>  Set temperature value, must be >= 0.0\n"
  "                         (default: 1.0)\n"
  "  -p, --topp <F>         Set top-p sampling value, must be 0.0 < p <= 1.0\n"
  "                         (default: 1.0)\n"
  "  -r, --rpen <F>         Set repetition penalty, must be >= 1.0\n"
  "                         (default: 1.0)\n"
  "  -i, --prompt <S>       Set input prompt, ignored if chat mode is enabled\n"
  "                         (default: \"Once upon a time\")\n"
  "  -c, --chat             Enable chat mode\n"
  "  -m, --enable-mm        Enable multi-modal capability, if supported by the model\n"
  "  -h, --help             Display this help message\n"
  "\n"
  "Controls:\n"
  "  Ctrl+C                 Gracefully interrupt generation and exit\n"
  "\n"
  "Examples:\n"
  "  ./gemma model.bin -l 2048 -t 0.8 -c\n"
  "  ./gemma model.bin -i \"Hello I'm a language model,\"\\\n"
  "      --seqlen 4096 --topk 50 --seed 12345\n");
  // clang-format on
}

#ifdef _WIN32
#  include <windows.h>
// The default console encoding is kinda weird in Windows
static void
set_utf8_console(void)
{
  SetConsoleOutputCP(65001);
  SetConsoleCP(65001);
}

// Convert Windows command line to UTF-8 argc/argv
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
      for (int j = 0; j < i; j++) { free(argv[j]); }
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

static void
free_utf8_argv(char **argv, int argc)
{
  if (argv == NULL) return;
  for (int i = 0; i < argc; i++) { free(argv[i]); }
  free(argv);
}

#else
// No problem with POSIX though
static void
set_utf8_console(void)
{
}

static char **
get_utf8_argv(int *argc_out)
{
  (void)argc_out;
  return NULL;
}

static void
free_utf8_argv(char **argv, int argc)
{
  (void)argv;
  (void)argc;
}
#endif

static const char *
safe_get_arg(int i, int argc, char **argv)
{
  if (i + 1 >= argc)
  {
    fprintf(stderr, "Error: Option '%s' requires an argument.\n", argv[i]);
    return NULL;
  }
  return argv[i + 1];
}

/* Debug-only pretty-print of the loaded model (only compiled with -DDEBUG) */
void
print_model_config(GemmaModel *model, int seqlen, bool enable_mm)
{
  (void)model;
  (void)seqlen;
  (void)enable_mm;

#ifdef DEBUG
  const int       width = 20;
  GemmaConfig    *conf  = model->config;
  GemmaTokenizer *tok   = model->tokenizer;

  printf("\n========== Model Configuration ==========\n");
  printf("Architecture:\n");

  // Integer fields
  printf("  %-*s: %d\n", width, "n_layers", conf->n_layers);
  printf("  %-*s: %d\n", width, "n_heads", conf->n_heads);
  printf("  %-*s: %d\n", width, "n_kv_heads", conf->n_kv_heads);
  printf("  %-*s: %d\n", width, "head_dim", conf->head_dim);
  printf("  %-*s: %d\n", width, "embed_dim", conf->embed_dim);
  printf("  %-*s: %d\n", width, "mlp_dim", conf->mlp_dim);
  printf("  %-*s: %d\n", width, "q_scale", conf->q_scale);
  printf("  %-*s: %d\n", width, "slide_len", conf->slide_len);
  printf("  %-*s: %d\n", width, "tpi", conf->tpi);
  printf("  %-*s: %d\n", width, "max_seqlen", conf->max_seqlen);
  printf("  %-*s: %d\n", width, "vocab_size", conf->vocab_size);

  // Float fields
  printf("  %-*s: %.6f\n", width, "local_theta", conf->local_theta);
  printf("  %-*s: %.6f\n", width, "global_theta", conf->global_theta);
  printf("  %-*s: %.6f\n", width, "eps", conf->eps);
  printf("  %-*s: %.6f\n", width, "att_softcap", conf->att_softcap);
  printf("  %-*s: %.6f\n", width, "logit_softcap", conf->logit_softcap);

  // Array fields
  printf("  %-*s: ", width, "att_layers");
  for (int i = 0; i < conf->n_layers; i++)
  {
    printf("%d", conf->att_layers[i] ? 1 : 0);
    if ((i + 1) % (width - 1) == 0 && i + 1 < conf->n_layers)
    {
      printf("\n");
      for (int j = 0; j < width + 4; j++) { printf(" "); }
    }
  }
  printf("\n");

  // Boolean fields
  printf(
      "  %-*s: %s\n", width, "support_mm", conf->support_mm ? "true" : "false");
  printf("  %-*s: %s\n", width, "qk_norm", conf->qk_norm ? "true" : "false");
  printf("  %-*s: %s\n", width, "pre_mlp_norm",
      conf->pre_mlp_norm ? "true" : "false");
  printf("  %-*s: %s\n", width, "post_mlp_norm",
      conf->post_mlp_norm ? "true" : "false");
  printf("  %-*s: %s\n", width, "quant", conf->quant ? "true" : "false");

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
  if (!conf->quant)
  {
    total += conf->embed_dim * conf->vocab_size * sizeof(_Float16);
  }
  else
  {
    total += conf->embed_dim * conf->vocab_size * sizeof(int8_t);
    total += conf->vocab_size * sizeof(_Float16);  // scales
  }

  // Weights per layer
  for (int l = 0; l < conf->n_layers; l++)
  {
    int C   = conf->embed_dim;
    int Cq  = conf->n_heads * conf->head_dim;
    int Ckv = conf->n_kv_heads * conf->head_dim;
    int CM  = conf->mlp_dim;

    int layer_params = C * Cq + C * Ckv + C * Ckv + Cq * C + C * CM * 3;

    if (!conf->quant) { total += layer_params * sizeof(_Float16); }
    else
    {
      total += layer_params * sizeof(int8_t);
      total += (Cq + Ckv + Ckv + C * 2 + CM * 2) * sizeof(_Float16);  // scales
    }

    // Norm layers
    total += C * sizeof(_Float16);  // n1
    total += C * sizeof(_Float16);  // n2
    if (conf->qk_norm) { total += 2 * conf->head_dim * sizeof(_Float16); }
    if (conf->pre_mlp_norm) { total += C * sizeof(_Float16); }
    if (conf->post_mlp_norm) { total += C * sizeof(_Float16); }
  }

  // Final norm
  total += conf->embed_dim * sizeof(_Float16);

  printf("  %-*s: %.2f GB\n", width, "Weights",
      (float)total / (1024.0 * 1024.0 * 1024.0));

  // KV Cache
  int    Ckv            = conf->n_kv_heads * conf->head_dim;
  size_t kv_cache_bytes = conf->n_layers * 2 * Ckv * sizeof(_Float16);
  printf("  %-*s: %.2f KB\n", width, "KV Cache (tok)",
      (float)kv_cache_bytes / 1024.0);

  // Gemma Buffer
  int           ppi   = 0;
  SigLIPConfig *vconf = NULL;
  if (conf->support_mm && enable_mm && model->vision_enc != NULL)
  {
    vconf = model->vision_enc->config;
    ppi   = vconf->image_size / vconf->patch_size;
  }
  int Cq   = conf->n_heads * conf->head_dim;
  int CM   = conf->mlp_dim;
  int mult = (conf->support_mm && enable_mm && model->vision_enc != NULL)
                 ? ppi * ppi
                 : 1;

  size_t buf_bytes = 0;

  // Quantized buffers
  if (conf->quant)
  {
    buf_bytes += conf->embed_dim * sizeof(int8_t);  // x_i8
    buf_bytes += Cq * sizeof(int8_t);               // xo_i8
    buf_bytes += CM * sizeof(int8_t);               // xg_i8
  }

  // Main buffers
  buf_bytes +=
      conf->n_layers * 2 * seqlen * Ckv * sizeof(_Float16);  // kv_cache
  buf_bytes += conf->vocab_size * sizeof(_Float16);          // logits
  buf_bytes += mult * conf->embed_dim * sizeof(_Float16);    // x
  buf_bytes += mult * conf->embed_dim * sizeof(_Float16);    // resid
  buf_bytes += mult * Cq * sizeof(_Float16);                 // xq
  buf_bytes += mult * Ckv * sizeof(_Float16);                // xk
  buf_bytes += mult * conf->head_dim * sizeof(_Float16);     // csfreqs_slid
  buf_bytes += mult * conf->head_dim * sizeof(_Float16);     // csfreqs_full
  buf_bytes += mult * Ckv * sizeof(_Float16);                // xv
  buf_bytes += mult * Cq * sizeof(_Float16);                 // xo
  buf_bytes += mult * conf->n_heads * seqlen * sizeof(_Float16);  // att
  buf_bytes += mult * CM * sizeof(_Float16);                      // xg
  buf_bytes += mult * CM * sizeof(_Float16);                      // xu

  printf("  %-*s: %.2f MB\n", width, "Gemma Buffer",
      (float)buf_bytes / (1024.0 * 1024.0));

  // SigLIP Buffer (if applicable)
  if (conf->support_mm && enable_mm && model->vision_enc != NULL)
  {
    SigLIPConfig *vconf_local = model->vision_enc->config;

    int C         = vconf_local->hidden_dim;
    int mlp_dim   = vconf_local->mlp_dim;
    int ppi_local = vconf_local->image_size / vconf_local->patch_size;
    int N         = ppi_local * ppi_local;
    int n_heads   = vconf_local->n_heads;

    size_t siglip_bytes = 0;

    // Quantized buffers
    if (conf->quant)
    {
      siglip_bytes += N * C * sizeof(int8_t);        // x_i8
      siglip_bytes += N * sizeof(_Float16);          // x_scales
      siglip_bytes += N * mlp_dim * sizeof(int8_t);  // mlp_i8
      siglip_bytes += N * sizeof(_Float16);          // mlp_scales
    }

    // Main buffers
    siglip_bytes += N * C * sizeof(_Float16);            // x
    siglip_bytes += N * C * sizeof(_Float16);            // resid
    siglip_bytes += N * C * sizeof(_Float16);            // xq
    siglip_bytes += N * C * sizeof(_Float16);            // xk
    siglip_bytes += N * C * sizeof(_Float16);            // xv
    siglip_bytes += N * C * sizeof(_Float16);            // att_out
    siglip_bytes += N * mlp_dim * sizeof(_Float16);      // mlp_hidden
    siglip_bytes += n_heads * N * N * sizeof(_Float16);  // scores

    printf("  %-*s: %.2f MB\n", width, "SigLIP Buffer",
        (float)siglip_bytes / (1024.0 * 1024.0));
  }

  printf("=========================================\n\n");
#endif
}

/* ------------------------------------------------------------------ */
/* main                                                               */
/* ------------------------------------------------------------------ */

int
main(int argc, char **argv)
{
  set_utf8_console();
  setup_signal_handler();

  // On Windows, get UTF-8 encoded command line arguments
  char **utf8_argv = get_utf8_argv(&argc);
  if (utf8_argv != NULL) { argv = utf8_argv; }

  if (argc < 2)
  {
    print_usage();
    printf("\nError: model filename is not provided\n");
    if (utf8_argv != NULL) { free_utf8_argv(utf8_argv, argc); }
    return 1;
  }

  char *modelfile = argv[1];
  if (!strcmp(modelfile, "-h") || !strcmp(modelfile, "--help"))
  {
    print_usage();
    return 0;
  }

  unsigned int seqlen      = 16384;
  unsigned int topk        = 0;
  unsigned int seed        = (unsigned int)time(NULL);
  float        temperature = 1.0;
  float        topp        = 1.0;
  float        rpen        = 1.0;
  const char  *prompt      = "Once upon a time";
  bool         chatmode    = false;
  bool         enable_mm   = false;

  // Parse the command line arguments
  for (int i = 2; i < argc; i++)
  {
    const char *arg = argv[i];

    if (strcmp(arg, "-l") == 0 || strcmp(arg, "--seqlen") == 0)
    {
      const char *val = safe_get_arg(i++, argc, argv);
      if (!val) { return 1; }
      if (!safe_atoui(val, &seqlen))
      {
        fprintf(stderr, "Invalid number for --seqlen: %s\n", val);
        return 1;
      }
    }
    else if (strcmp(arg, "-k") == 0 || strcmp(arg, "--topk") == 0)
    {
      const char *val = safe_get_arg(i++, argc, argv);
      if (!val) { return 1; }
      if (!safe_atoui(val, &topk))
      {
        fprintf(stderr, "Invalid number for --topk: %s\n", val);
        return 1;
      }
    }
    else if (strcmp(arg, "-s") == 0 || strcmp(arg, "--seed") == 0)
    {
      const char *val = safe_get_arg(i++, argc, argv);
      if (!val) { return 1; }
      if (!safe_atoui(val, &seed))
      {
        fprintf(stderr, "Invalid number for --seed: %s\n", val);
        return 1;
      }
    }
    else if (strcmp(arg, "-t") == 0 || strcmp(arg, "--temperature") == 0)
    {
      const char *val = safe_get_arg(i++, argc, argv);
      if (!val) { return 1; }
      if (!safe_atof(val, &temperature) || temperature < 0.0f)
      {
        fprintf(stderr, "Invalid temperature: %s (must be >= 0)\n", val);
        return 1;
      }
    }
    else if (strcmp(arg, "-p") == 0 || strcmp(arg, "--topp") == 0)
    {
      const char *val = safe_get_arg(i++, argc, argv);
      if (!val) { return 1; }
      if (!safe_atof(val, &topp) || topp <= 0.0f || topp > 1.0f)
      {
        fprintf(stderr, "Invalid top-p: %s (must be 0 < p <= 1)\n", val);
        return 1;
      }
    }
    else if (strcmp(arg, "-r") == 0 || strcmp(arg, "--rpen") == 0)
    {
      const char *val = safe_get_arg(i++, argc, argv);
      if (!val) { return 1; }
      if (!safe_atof(val, &rpen) || rpen < 1.0f)
      {
        fprintf(stderr, "Invalid repetition penalty: %s (must be >= 1)\n", val);
        return 1;
      }
    }
    else if (strcmp(arg, "-i") == 0 || strcmp(arg, "--prompt") == 0)
    {
      const char *val = safe_get_arg(i++, argc, argv);
      if (!val) { return 1; }
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
    else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0)
    {
      print_usage();
      return 0;
    }
    else
    {
      fprintf(stderr, "Unknown option: %s\n", arg);
      print_usage();
      return 1;
    }
  }

  srand(seed);

  GemmaModel   *model = read_model(modelfile, enable_mm);
  GemmaConfig  *conf  = NULL;
  SigLIPConfig *vconf = NULL;
  GemmaBuffer  *buf   = NULL;
  SigLIPBuffer *sbuf  = NULL;

  if (model == NULL) { goto end; }

  conf = model->config;
  if (model->vision_enc != NULL) { vconf = model->vision_enc->config; }

  buf = malloc_buffer(conf, vconf, (int)seqlen, enable_mm);
  if (buf == NULL) { goto end; }

  // malloc buffer for SigLIP
  if (enable_mm && conf->support_mm)
  {
    sbuf = malloc_siglip_buffer(vconf, conf->quant);
    if (sbuf == NULL) { goto end; }
  }
  print_model_config(model, (int)seqlen, enable_mm);

  if (chatmode)
  {
    chat(model, buf, (int)seqlen, temperature, (int)topk, topp, rpen);
  }
  else
  {
    generate(
        model, buf, prompt, (int)seqlen, temperature, (int)topk, topp, rpen);
  }

  if (g_interrupted) { printf("\n\nInterrupted by user\n"); }

end:
  free_utf8_argv(utf8_argv, argc);
  if (conf != NULL)
  {
    free_buffer(buf, conf->quant);
    free_siglip_buffer(sbuf, conf->quant);
  }
  free_model(model);
  return 0;
}
