#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <signal.h>

#define FP16_MAX      (((union {_Float16 f; uint16_t b;}){.b = 0x7BFF}).f)
#define EOT_SENTINEL  (-1)  // Sentinel for end of token array
#define SAMPLE_ABORT  ((int *)(intptr_t)-1)

#ifdef DEBUG
#pragma message("Debug mode on")
#endif

// Global state for interruption handling
static volatile bool g_interrupted = false;

void signal_handler(int signum) {
    (void)signum;
    g_interrupted = true;
}

void setup_signal_handler(void) {
    signal(SIGINT, signal_handler);
#ifdef SIGTERM
    signal(SIGTERM, signal_handler);
#endif
}

// Safe memory operation wrappers
void *safe_malloc(size_t size, const char *context) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(
            stderr, "Memory allocation failed: %s (size: %zu)\n",
            context, size
        ); exit(1);
    }
    return ptr;
}

void *safe_calloc(size_t count, size_t size, const char *context) {
    void *ptr = calloc(count, size);
    if (ptr == NULL) {
        fprintf(
            stderr, "Memory allocation failed: %s (count: %zu, size: %zu)\n",
            context, count, size
        ); exit(1);
    }
    return ptr;
}

void safe_fread(void *ptr, size_t size, size_t count, FILE *fp, const char *context) {
    size_t read = fread(ptr, size, count, fp);
    if (read != count) {
        fprintf(
            stderr, "File read failed: %s (expected %zu, got %zu)\n",
            context, count, read
        ); exit(1);
    }
}

typedef struct {
    int n_layers;
    int image_size;
    int patch_size;
    int hidden_dim;
    int n_heads;
    int mlp_hidden_dim;
    float eps;
} SigLIPConfig;

typedef struct {
    int n_layers;         // Number of transformer layers
    int n_heads;          // Number of attention heads
    int n_kv_heads;       // Number of key & value heads (Grouped Query Attention)
    int head_dim;         // Dimensions of the attention heads
    int embed_dim;        // Dimensions of the embedding vectors
    int mlp_hidden_dim;  // Dimensions of the MLP hidden layers
    // The scale applied to query vectors before computing attention scores
    int q_pre_attn_scalar;
    int sliding_window;   // Context size in sliding window attention
    int tokens_per_image; // Number of embedding vectors per image in vision models
    int max_seq_len;      // Max number of positional embeddings
    int vocab_size;       // Number of tokens in the vocabulary
    float local_theta;    // RoPE wavelength in sliding window attention
    float global_theta;   // RoPE wavelength in full attention
    float eps;            // Epsilon in RMSNorm layers
    // Tanh softcapping scalar to the attention scores before softmax (0.0 means disabled)
    float attn_softcapping;
    // Tanh softcapping scalar to the final logits (0.0 means disabled)
    float logit_softcapping;
    // Bool array that specifies which layers should use sliding window attention
    bool *attn_local_layers;

    bool support_mm;          // Whether the model supports multi-modal
    bool use_qk_norm;     // Whether query & key normalization is applied
    bool pre_ffwd_norm;   // Whether pre-feedforward normalization is applied
    bool post_ffwd_norm;  // Whether post-feedforward normalization is applied
    bool quant;           // Whether int8 quantization is enabled
} GemmaConfig;

void free_config(GemmaConfig *conf) {
    free(conf->attn_local_layers);
    free(conf);
}

typedef struct { char *val; int idx; } Token;
typedef struct { char *str1; char *str2; int rank; } Merge;

// Functions for qsort / bsearch

int cmp_token(const void *a, const void *b) {
    return strcmp(((Token *)a)->val, ((Token *)b)->val);
}

int cmp_merge(const void *a, const void *b) {
    int ret = strcmp(((Merge *)a)->str1, ((Merge *)b)->str1);
    if (ret != 0) { return ret; }
    return strcmp(((Merge *)a)->str2, ((Merge *)b)->str2);
}

typedef struct {
    int n_merges;
    int vocab_size;
    int bos;  // Beginning of sequence
    int eos;  // End of sequence
    int sot;  // Start of turn
    int eot;  // End of turn
    int soi;  // Start of image
    int eoi;  // End of image
    int ist;  // Image soft token
    char *vocab_data;
    char *merge_data;
    char **vocab;
    Token *vocab_sorted;
    Merge *ranks;
} GemmaTokenizer;

void free_tokenizer(GemmaTokenizer *tok) {
    free(tok->vocab_data);
    free(tok->merge_data);
    free(tok->vocab);
    free(tok->vocab_sorted);
    free(tok->ranks);
    free(tok);
}

// These kind of act as dictionaries in python

int get_token_idx(GemmaTokenizer *tok, char *str) {
    // Get idx_to_vocab[string]
    Token key = { .val = str };
    Token *val = bsearch(
        &key, tok->vocab_sorted, tok->vocab_size, sizeof(tok->vocab_sorted[0]), cmp_token
    );
    if (val == NULL) { return -1; }
    return val->idx;
}

Merge *get_merge_rec(GemmaTokenizer *tok, char *str1, char *str2) {
    // Get merges[(str1, str2)]
    Merge key = { .str1 = str1, .str2 = str2 };
    Merge *val = bsearch(
        &key, tok->ranks, tok->n_merges, sizeof(tok->ranks[0]), cmp_merge
    );
    return val;  // NULL if not found
}

typedef enum { DTYPE_FP16, DTYPE_INT8 } WeightDType;

// Linear weights
typedef struct {
    WeightDType dtype;
    union {
        _Float16 *fp16;
        struct { int8_t *q; _Float16 *scales; } i8;
    };
} Linear;

void free_linear(Linear l) {
    if (l.dtype == DTYPE_FP16) { free(l.fp16); }
    else { free(l.i8.q); free(l.i8.scales); }
}

typedef struct {
    // Attention weights
    // 2D weights are transposed for higher CPU cache hits in GEMV
    Linear wq;  // (embed_dim, n_heads * head_dim).T
    Linear wk;  // (embed_dim, n_kv_heads * head_dim).T
    Linear wv;  // (embed_dim, n_kv_heads * head_dim).T
    Linear wo;  // (n_heads * head_dim, embed_dim).T
    // Feedforward weights
    Linear w1;  // (embed_dim, mlp_hidden_dim).T
    Linear w2;  // (embed_dim, mlp_hidden_dim).T
    Linear w3;  // (mlp_hidden_dim, embed_dim).T
    // RMSNorm weights
    _Float16 *nq;  // (head_dim,)
    _Float16 *nk;  // (head_dim,)
    _Float16 *n1;  // (embed_dim,)
    _Float16 *n2;  // (embed_dim,)
    _Float16 *n3;  // (embed_dim,)
    _Float16 *n4;  // (embed_dim,)
} GemmaDecoderLayer;

void free_gemma_layer(GemmaDecoderLayer *layer) {
    free_linear(layer->wq);
    free_linear(layer->wk);
    free_linear(layer->wv);
    free_linear(layer->wo);
    free_linear(layer->w1);
    free_linear(layer->w2);
    free_linear(layer->w3);
    free(layer->n1);
    free(layer->n2);
    if (layer->nq != NULL) { free(layer->nq); }
    if (layer->nk != NULL) { free(layer->nk); }
    if (layer->n3 != NULL) { free(layer->n3); }
    if (layer->n4 != NULL) { free(layer->n4); }
    free(layer);
}

typedef struct {
    // ViT attention weights
    Linear wq;  // (hidden_dim, hidden_dim).T
    Linear wk;  // (hidden_dim, hidden_dim).T
    Linear wv;  // (hidden_dim, hidden_dim).T
    Linear wo;  // (hidden_dim, hidden_dim).T
    // ViT attention biases
    _Float16 *bq;  // (hidden_dim,)
    _Float16 *bk;  // (hidden_dim,)
    _Float16 *bv;  // (hidden_dim,)
    _Float16 *bo;  // (hidden_dim,)
    // ViT feedforward weights
    Linear w1;  // (hidden_dim, mlp_hidden_dim).T
    Linear w2;  // (hidden_dim, mlp_hidden_dim).T
    // ViT feedforward biases
    _Float16 *b1;  // (mlp_hidden_dim,)
    _Float16 *b2;  // (mlp_hidden_dim,)
    // ViT layernorm weights
    _Float16 *n1;  // (hidden_dim,)
    _Float16 *n2;  // (hidden_dim,)
    // ViT layernorm biases
    _Float16 *n1_b;  // (hidden_dim,)
    _Float16 *n2_b;  // (hidden_dim,)
} SigLIPEncoderLayer;

void free_siglip_layer(SigLIPEncoderLayer *layer) {
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
}

typedef struct {
    SigLIPConfig *config;
    _Float16 *patch_emb;   // (hidden_dim, 3, patch_size, patch_size)
    _Float16 *patch_emb_b;  // (hidden_dim,)
    Linear pos_embedding;  // ((image_size / patch_size)^2, hidden_dim)
    SigLIPEncoderLayer **layers;
    _Float16 *post_norm;  // (hidden_dim,)
    _Float16 *post_norm_b;  // (hidden_dim,)
    _Float16 *norm;  // (hidden_dim,)
    Linear proj;  // (hidden_dim, embed_dim).T
} SigLIPVisionEncoder;

void free_siglip(SigLIPVisionEncoder *enc) {
    free(enc->patch_emb);
    free(enc->patch_emb_b);
    free_linear(enc->pos_embedding);
    for (int i = 0; i < enc->config->n_layers; i++) {
        free_siglip_layer(enc->layers[i]);
    }
    free(enc->post_norm);
    free(enc->post_norm_b);
    free(enc->norm);
    free_linear(enc->proj);
    free(enc);
}

typedef struct {
    GemmaConfig *config;
    GemmaTokenizer *tokenizer;
    SigLIPVisionEncoder *vision_enc;
    Linear embedding;   // (vocab_size, embed_dim)
    GemmaDecoderLayer **layers;
    _Float16 *final_norm;  // (embed_dim,)
} GemmaModel;

void free_model(GemmaModel *model) {
    free_tokenizer(model->tokenizer);
    if (model->vision_enc != NULL) {
        free_siglip(model->vision_enc);
    }
    free_linear(model->embedding);
    for (int i = 0; i < model->config->n_layers; i++) {
        free_gemma_layer(model->layers[i]);
    }
    free(model->final_norm);
    free_config(model->config);
    free(model->layers);
    free(model);
}

typedef struct {
    int8_t *x_i8;           // (n_patches, hidden_dim)
    _Float16 *x_scales;     // (n_patches,)
    int8_t *mlp_hidden_i8;  // (n_patches, mlp_hidden_dim)
    _Float16 *mlp_hidden_scales;  // (n_patches)

    _Float16 *x;            // (n_patches, hidden_dim)
    _Float16 *resid;        // (n_patches, hidden_dim)
    _Float16 *xq;           // (n_patches, hidden_dim)
    _Float16 *xk;           // (n_patches, hidden_dim)
    _Float16 *xv;           // (n_patches, hidden_dim)
    _Float16 *att_out;      // (n_patches, hidden_dim)
    _Float16 *mlp_hidden;   // (n_patches, mlp_hidden_dim)
    _Float16 *scores;
} SigLIPBuffer;

SigLIPBuffer *malloc_siglip_buffer(SigLIPConfig *vconf, bool quant) {
    int C = vconf->hidden_dim;
    int ppi = vconf->image_size / vconf->patch_size;
    int NP = ppi * ppi;
    int CM = vconf->mlp_hidden_dim;
    int NH = vconf->n_heads;

    SigLIPBuffer *buf = safe_malloc(sizeof(*buf), "SigLIPBuffer");

    if (quant) {
        buf->x_i8 = safe_malloc(NP * C * sizeof(int8_t), "Siglip x_i8");
        buf->x_scales = safe_malloc(NP * sizeof(_Float16), "Siglip x_scales");
    }

    buf->x = safe_malloc(NP * C * sizeof(_Float16), "SigLIP x");
    buf->resid = safe_malloc(NP * C * sizeof(_Float16), "SigLIP resid");
    buf->xq = safe_malloc(NP * C * sizeof(_Float16), "SigLIP q");
    buf->xk = safe_malloc(NP * C * sizeof(_Float16), "SigLIP k");
    buf->xv = safe_malloc(NP * C * sizeof(_Float16), "SigLIP v");
    buf->att_out = safe_malloc(NP * C * sizeof(_Float16), "SigLIP attn_out");
    buf->mlp_hidden = safe_malloc(NP * CM * sizeof(_Float16), "SigLIP mlp_hidden");
    buf->scores = safe_malloc(NH * NP * NP * sizeof(_Float16), "SigLIP scores");

    return buf;
}

void free_siglip_buffer(SigLIPBuffer *buf, bool quant) {
    if (quant) {
        free(buf->x_i8);
        free(buf->x_scales);
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

typedef struct {
    int cache_len;

    int8_t *x_i8;            // (embed_dim,)
    int8_t *xo_i8;           // (n_heads, head_dim)
    int8_t *xg_i8;           // (mlp_hidden_dim,)

    _Float16 *x;             // (embed_dim,)
    _Float16 *resid;         // (embed_dim,)
    _Float16 *xq;            // (n_heads, head_dim)
    _Float16 *xk;            // (n_kv_heads, head_dim)
    _Float16 *csfreqs_slid;  // (head_dim / 2, 2)
    _Float16 *csfreqs_full;  // (head_dim / 2, 2)
    _Float16 *xv;            // (n_kv_heads, head_dim)
    _Float16 *xo;            // (n_heads, head_dim)
    _Float16 *att;           // (n_heads, cache_len)
    _Float16 *kv_cache;      // (n_layers, 2, cache_len, n_kv_heads, head_dim)
    _Float16 *xg;            // (mlp_hidden_dim,)
    _Float16 *xu;            // (mlp_hidden_dim,)
    _Float16 *logits;        // (vocab_size,)
} GemmaBuffer;

GemmaBuffer *malloc_buffer(GemmaConfig *conf, int cache_len, bool enable_mm) {
    GemmaBuffer *buf = safe_malloc(sizeof(*buf), "GemmaBuffer");
    buf->cache_len = cache_len;

    int C = conf->embed_dim;
    int CH = conf->head_dim;
    int NH = conf->n_heads;
    int CM = conf->mlp_hidden_dim;
    int NL = conf->n_layers;

    int q_size = NH * CH;
    int kv_size = conf->n_kv_heads * CH;
    
    if (conf->quant) {
        buf->x_i8 = safe_malloc(C * sizeof(int8_t), "buffer x_i8");
        buf->xo_i8 = safe_malloc(q_size * sizeof(int8_t), "buffer xo_i8");
        buf->xg_i8 = safe_malloc(conf->mlp_hidden_dim * sizeof(int8_t), "buffer xg_i8");
    }

    buf->x = safe_malloc(C * sizeof(*buf->x), "buffer x");
    buf->resid = safe_malloc(C * sizeof(*buf->resid), "buffer resid");

    buf->xq = safe_malloc(q_size * sizeof(*buf->xq), "buffer xq");
    buf->xk = safe_malloc(kv_size * sizeof(*buf->xk), "buffer xk");
    buf->csfreqs_slid = safe_malloc(CH * sizeof(*buf->csfreqs_slid), "buffer csfreqs_slid");
    buf->csfreqs_full = safe_malloc(CH * sizeof(*buf->csfreqs_full), "buffer csfreqs_full");
    buf->xv = safe_malloc(kv_size * sizeof(*buf->xv), "buffer xv");
    buf->xo = safe_malloc(q_size * sizeof(*buf->xo), "buffer xo");
    buf->att = safe_malloc(NH * cache_len * sizeof(*buf->att), "buffer att");
    buf->kv_cache = safe_malloc(NL * 2 * cache_len * kv_size * sizeof(*buf->kv_cache), "buffer kv_cache");
    buf->xg = safe_malloc(CM * sizeof(*buf->xg), "buffer xg");
    buf->xu = safe_malloc(CM * sizeof(*buf->xu), "buffer xu");
    buf->logits = safe_malloc(conf->vocab_size * sizeof(*buf->logits), "buffer logits");

    return buf;
}

void free_buffer(GemmaBuffer *buf, bool quant) {
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
    if (quant) { free(buf->x_i8); free(buf->xo_i8); free(buf->xg_i8); }
    free(buf);
}

int read_uint16(FILE *fp) {
    unsigned char bytes[2];
    safe_fread(bytes, 1, 2, fp, "uint16");
    // Big-endian ordering
    return ((int)bytes[0] << 8) | (int)bytes[1];
}

uint32_t read_uint32(FILE *fp) {
    unsigned char bytes[4];
    safe_fread(bytes, 1, 4, fp, "uint32");
    // Big-endian ordering
    return (
        ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) |
        ((uint32_t)bytes[2] <<  8) | ((uint32_t)bytes[3])
    );
}

float read_float32(FILE *fp) {
    uint32_t buf = read_uint32(fp);
    float f;
    memcpy(&f, &buf, sizeof(float));
    return f;
}

char *read_str(FILE *fp, char *data, int *offset) {
    // Read a pascal-style string, the first byte indicates the length
    int len = fgetc(fp);
    if (len == EOF) {
        fprintf(stderr, "File read failed: reading string length\n"); exit(1);
    }
    char *str = data + *offset;
    safe_fread(str, sizeof(char), len, fp, "string data");
    data[*offset + len] = '\0';  // Convert to C-style
    *offset += len + 1;  // Advance offset
    return str;
}

int get_strarr_bytes(FILE *fp, int count) {
    // Read a sequence of pascal-stype strings
    int offset = 0;
    long pos = ftell(fp);
    if (pos == -1L) {
        perror("ftell failed"); exit(1);
    }
    // Get the total number of bytes
    for (int i = 0; i < count; i++) {
        int len = fgetc(fp);
        if (len == EOF) {
            fprintf(stderr, "File read failed: reading string length\n"); exit(1);
        }
        if (fseek(fp, len, SEEK_CUR) != 0) {
            perror("fseek failed"); exit(1);
        }
        offset += len + 1;
    }
    // Resume position
    if (fseek(fp, pos, SEEK_SET) != 0) {
        perror("fseek failed"); exit(1);
    }
    return offset;
}

_Float16 *read_tensor_fp16(FILE *fp, int size) {
    _Float16 *tensor = safe_malloc(size * sizeof(_Float16), "fp16 tensor");
    safe_fread(tensor, sizeof(_Float16), size, fp, "fp16 tensor");
    return tensor;
}

int8_t *read_tensor_int8(FILE *fp, int size) {
    int8_t *tensor = safe_malloc(size * sizeof(int8_t), "int8 tensor");
    safe_fread(tensor, sizeof(int8_t), size, fp, "int8 tensor");
    return tensor;
}

Linear read_linear(FILE *fp, int m, int n, bool quant) {
    Linear w;
    if (!quant) {
        // fp16 precision
        w.dtype = DTYPE_FP16;
        w.fp16 = read_tensor_fp16(fp, m * n);
    } else {
        // int8 quantization
        w.dtype = DTYPE_INT8;
        w.i8.q = read_tensor_int8(fp, m * n);
        w.i8.scales = read_tensor_fp16(fp, n);
    }
    return w;
}

// Python-like repr() for error printing
char *repr(const char *str) {
    int len = strlen(str);
    char *result = malloc(len * 4 + 3);
    result[0] = '"';
    int offset = 1;
    for (int i = 0; i < len; i++) {
        switch (str[i]) {
        case '\n':
            result[offset++] = '\\';
            result[offset++] = 'n';
            break;
        case '\t':
            result[offset++] = '\\';
            result[offset++] = 't';
            break;
        case '\r':
            result[offset++] = '\\';
            result[offset++] = 'r';
            break;
        case '\\':
            result[offset++] = '\\';
            result[offset++] = '\\';
            break;
        case '"':
            result[offset++] = '\\';
            result[offset++] = '"';
            break;
        default:
            result[offset++] = str[i];
        }
    }
    result[offset++] = '"';
    result[offset++] = '\0';
    return result;
}

GemmaModel *read_model(const char *filename, bool enable_mm) {
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) { perror(filename); exit(1); }

    GemmaTokenizer *tok = safe_malloc(sizeof(*tok), "GemmaTokenizer");
    GemmaConfig *conf = safe_malloc(sizeof(*conf), "GemmaConfig");

    // Read the configs
    conf->n_layers = fgetc(fp);
    conf->n_heads = fgetc(fp);
    conf->n_kv_heads = fgetc(fp);
    conf->head_dim = read_uint16(fp);
    conf->embed_dim = read_uint16(fp);
    conf->mlp_hidden_dim = read_uint16(fp);
    conf->q_pre_attn_scalar = read_uint16(fp);
    conf->sliding_window = read_uint16(fp);
    conf->tokens_per_image = read_uint16(fp);
    conf->max_seq_len = (int)read_uint32(fp);
    conf->vocab_size = (int)read_uint32(fp);
    conf->local_theta = read_float32(fp);
    conf->global_theta = read_float32(fp);
    conf->eps = read_float32(fp);
    conf->attn_softcapping = read_float32(fp);
    conf->logit_softcapping = read_float32(fp);

    // attn_local_layers
    int n_bytes = fgetc(fp);
    if (n_bytes == EOF) {
        fprintf(stderr, "File read failed: reading attn_local_layers size\n"); exit(1);
    }
    char *buf = safe_malloc(n_bytes, "attn_local_layers buffer");
    conf->attn_local_layers = safe_malloc(conf->n_layers * sizeof(bool), "attn_local_layers");
    safe_fread(buf, 1, n_bytes, fp, "attn_local_layers data");
    for (int i = 0; i < conf->n_layers; i++) {
        int pos = i;
        int byte_idx = pos / 8;
        int bit_idx = 7 - (pos % 8);
        conf->attn_local_layers[i] = (buf[byte_idx] >> bit_idx) & 1;
    }
    free(buf);

    // Additional flags
    int extra_flags = fgetc(fp);
    if (extra_flags == EOF) {
        fprintf(stderr, "File read failed: reading extra flags\n"); exit(1);
    }
    conf->support_mm = (extra_flags & 16) == 16;
    conf->use_qk_norm = (extra_flags & 8) == 8;
    conf->pre_ffwd_norm = (extra_flags & 4) == 4;
    conf->post_ffwd_norm = (extra_flags & 2) == 2;
    conf->quant = extra_flags & 1;

    bool use_mm = conf->support_mm && enable_mm;

    SigLIPVisionEncoder *enc = NULL;
    SigLIPConfig *vconf = NULL;
    if (use_mm) {
        // Read vision config
        enc = malloc(sizeof(*enc));
        vconf = malloc(sizeof(*vconf));
        vconf->n_layers = fgetc(fp);
        vconf->n_heads = fgetc(fp);
        vconf->mlp_hidden_dim = read_uint16(fp);
        vconf->hidden_dim = read_uint16(fp);
        vconf->image_size = read_uint16(fp);
        vconf->patch_size = read_uint16(fp);
        vconf->eps = read_float32(fp);
        enc->config = vconf;
    }

    // dtype (only supports float16)
    int offset = 0;
    char dtype[10];
    read_str(fp, dtype, &offset);
    if (strcmp(dtype, "float16") != 0) {
        char *repr_dtype = repr(dtype);
        printf("dtype %s not supported\n", repr_dtype);
        free(repr_dtype);
        exit(1);
    }

    // Build vocabulary
    offset = 0;
    tok->vocab_size = conf->vocab_size;
    if (use_mm) { tok->vocab_size++; }  // ++ for the <image_soft_token>
    tok->vocab_data = safe_malloc(get_strarr_bytes(fp, tok->vocab_size), "vocab_data");
    tok->vocab = safe_malloc(tok->vocab_size * sizeof(*tok->vocab), "vocab");
    tok->vocab_sorted = safe_malloc(tok->vocab_size * sizeof(*tok->vocab_sorted), "vocab_sorted");
    for (int i = 0; i < tok->vocab_size; i++) {
        char *str = read_str(fp, tok->vocab_data, &offset);
        tok->vocab[i] = str;
        tok->vocab_sorted[i].idx = i;
        tok->vocab_sorted[i].val = str;
    }
    qsort(tok->vocab_sorted, tok->vocab_size, sizeof(tok->vocab_sorted[0]), cmp_token);
    // Special tokens
    tok->bos = get_token_idx(tok, "<bos>");
    tok->eos = get_token_idx(tok, "<eos>");
    tok->sot = get_token_idx(tok, "<start_of_turn>");
    tok->eot = get_token_idx(tok, "<end_of_turn>");
    tok->soi = get_token_idx(tok, "<start_of_image>");
    tok->eoi = get_token_idx(tok, "<end_of_image>");
    tok->ist = get_token_idx(tok, "<image_soft_token>");

    // Build merges
    tok->n_merges = read_uint32(fp);
    tok->ranks = safe_malloc(tok->n_merges * sizeof(*tok->ranks), "ranks");
    tok->merge_data = safe_malloc(get_strarr_bytes(fp, tok->n_merges * 2), "merge_data");

    offset = 0;
    for (int i = 0; i < tok->n_merges; i++) {
        char *str1 = read_str(fp, tok->merge_data, &offset);
        char *str2 = read_str(fp, tok->merge_data, &offset);
        tok->ranks[i].rank = i;
        tok->ranks[i].str1 = str1;
        tok->ranks[i].str2 = str2;
    }
    qsort(tok->ranks, tok->n_merges, sizeof(tok->ranks[0]), cmp_merge);

    // Build the model
    GemmaModel *model = safe_malloc(sizeof(*model), "GemmaModel");
    model->vision_enc = enc;

    // Read the weights
    model->config = conf;
    model->tokenizer = tok;

    // The shape is actually (vocab_size, embed_dim), but it uses per-tensor quantization
    // rather than per-channel like other weights. Gemma uses tied weights, which means
    // the final lm_head shares the same weights with the embedding table, but transposed.
    // So it becomes per-channel quantization in the final lm_head
    model->embedding = read_linear(fp, conf->embed_dim, conf->vocab_size, conf->quant);
    model->layers = safe_malloc(conf->n_layers * sizeof(*model->layers), "model layers");

    int q_size = conf->n_heads * conf->head_dim;
    int kv_size = conf->n_kv_heads * conf->head_dim;

    // Read all the layers
    for (int l = 0; l < conf->n_layers; l++) {
        GemmaDecoderLayer *layer = safe_malloc(sizeof(*layer), "GemmaDecoderLayer");

        // Attention weights
        layer->wq = read_linear(fp, conf->embed_dim, q_size, conf->quant);
        layer->wk = read_linear(fp, conf->embed_dim, kv_size, conf->quant);
        layer->wv = read_linear(fp, conf->embed_dim, kv_size, conf->quant);
        layer->wo = read_linear(fp, q_size, conf->embed_dim, conf->quant);

        if (conf->use_qk_norm) {
            layer->nq = read_tensor_fp16(fp, conf->head_dim);
            layer->nk = read_tensor_fp16(fp, conf->head_dim);
        } else {
            layer->nq = NULL;
            layer->nk = NULL;
        }

        // Feedforward weights
        layer->w1 = read_linear(fp, conf->embed_dim, conf->mlp_hidden_dim, conf->quant);
        layer->w2 = read_linear(fp, conf->embed_dim, conf->mlp_hidden_dim, conf->quant);
        layer->w3 = read_linear(fp, conf->mlp_hidden_dim, conf->embed_dim, conf->quant);

        // RMSNorm weights
        layer->n1 = read_tensor_fp16(fp, conf->embed_dim);
        layer->n2 = read_tensor_fp16(fp, conf->embed_dim);

        if (conf->pre_ffwd_norm) {
            layer->n3 = read_tensor_fp16(fp, conf->embed_dim);
        } else {
            layer->n3 = NULL;
        }
        if (conf->post_ffwd_norm) {
            layer->n4 = read_tensor_fp16(fp, conf->embed_dim);
        } else {
            layer->n4 = NULL;
        }
        model->layers[l] = layer;
    }
    model->final_norm = read_tensor_fp16(fp, conf->embed_dim);

    if (use_mm) {
        enc->patch_emb = read_tensor_fp16(fp, vconf->hidden_dim * 3 * vconf->patch_size * vconf->patch_size);
        enc->patch_emb_b = read_tensor_fp16(fp, vconf->hidden_dim);
        int n_patches = vconf->image_size / vconf->patch_size;
        n_patches *= n_patches;
        // Same as here, the real shape is (n_patches, vconf->hidden_dim)
        enc->pos_embedding = read_linear(fp, vconf->hidden_dim, n_patches, conf->quant);
        enc->layers = malloc(vconf->n_layers * sizeof(*enc->layers));

        // Read all the layers of ViT
        for (int l = 0; l < vconf->n_layers; l++) {
            SigLIPEncoderLayer *layer = safe_malloc(sizeof(*layer), "SigLIPEncoderLayer");
            
            // First layernorm
            layer->n1 = read_tensor_fp16(fp, vconf->hidden_dim);
            layer->n1_b = read_tensor_fp16(fp, vconf->hidden_dim);

            // Attention weights
            layer->wq = read_linear(fp, vconf->hidden_dim, vconf->hidden_dim, conf->quant);
            layer->wk = read_linear(fp, vconf->hidden_dim, vconf->hidden_dim, conf->quant);
            layer->wv = read_linear(fp, vconf->hidden_dim, vconf->hidden_dim, conf->quant);
            layer->wo = read_linear(fp, vconf->hidden_dim, vconf->hidden_dim, conf->quant);
            // Attention biases
            layer->bq = read_tensor_fp16(fp, vconf->hidden_dim);
            layer->bk = read_tensor_fp16(fp, vconf->hidden_dim);
            layer->bv = read_tensor_fp16(fp, vconf->hidden_dim);
            layer->bo = read_tensor_fp16(fp, vconf->hidden_dim);

            // Second layernorm
            layer->n2 = read_tensor_fp16(fp, vconf->hidden_dim);
            layer->n2_b = read_tensor_fp16(fp, vconf->hidden_dim);

            // Feedforward weights
            layer->w1 = read_linear(fp, vconf->hidden_dim, vconf->mlp_hidden_dim, conf->quant);
            layer->w2 = read_linear(fp, vconf->mlp_hidden_dim, vconf->hidden_dim, conf->quant);
            // Feedforward biases
            layer->b1 = read_tensor_fp16(fp, vconf->mlp_hidden_dim);
            layer->b2 = read_tensor_fp16(fp, vconf->hidden_dim);
            
            enc->layers[l] = layer;
        }

        // Post layernorm
        enc->post_norm = read_tensor_fp16(fp, vconf->hidden_dim);
        enc->post_norm_b = read_tensor_fp16(fp, vconf->hidden_dim);

        // Soft embedding RMSNorm
        enc->norm = read_tensor_fp16(fp, vconf->hidden_dim);
        // Final projection
        enc->proj = read_linear(fp, vconf->hidden_dim, conf->embed_dim, conf->quant);
    }

    fclose(fp);
    return model;
}

// Make sure the clamping is not optimized by compilers
#ifdef __GNUC__
__attribute__((optimize("no-fast-math")))
#endif
static inline _Float16 clamp_fp16(_Float16 v) {
#ifdef __clang__
#pragma float_control(precise, on, push)
#endif
    return fminf(FP16_MAX, fmaxf(-FP16_MAX, v));
#ifdef __clang__
#pragma float_control(pop)
#endif
}

void rmsnorm(_Float16 *dst, _Float16 *src, _Float16 *weight, int dim, float eps) {
    float sqsum = 0.0f;
    #pragma omp simd reduction(+:sqsum)  // Only using SIMD here
    for (int i = 0; i < dim; i++) {
        sqsum += (float)src[i] * (float)src[i];
    }
    float rms = 1.0f / sqrtf(sqsum / dim + eps);
    #pragma omp simd
    for (int i = 0; i < dim; i++) {
        // Gemma uses (weight + 1) instead of (weight)
        dst[i] = (_Float16)((float)src[i] * rms * (weight[i] + 1));
    }
}

void rmsnorm_omp(_Float16 *dst, _Float16 *src, _Float16 *weight, int dim, float eps) {
    // OpenMP parallelized version of RMSNorm
    float sqsum = 0.0f;
    #pragma omp parallel for reduction(+:sqsum)
    for (int i = 0; i < dim; i++) {
        sqsum += (float)src[i] * (float)src[i];
    }
    float rms = 1.0f / sqrtf(sqsum / dim + eps);
    #pragma omp simd
    for (int i = 0; i < dim; i++) {
        // Gemma uses (weight + 1) instead of (weight)
        dst[i] = (_Float16)((float)src[i] * rms * (weight[i] + 1));
    }
}

// LayerNorm used in SigLIP
void layernorm(_Float16 *dst, _Float16 *src, _Float16 *weight, _Float16 *bias, int dim, float eps) {
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) {
        mean += (float)src[i];
    }
    mean /= dim;

    float var = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = (float)src[i] - mean;
        var += diff * diff;
    }
    var /= dim;

    float inv_std = 1.0f / sqrtf(var + eps);
    #pragma omp simd
    for (int i = 0; i < dim; i++) {
        dst[i] = (_Float16)(((float)src[i] - mean) * inv_std * (float)weight[i] + (float)bias[i]);
    }
}

_Float16 quantize_act(int8_t *dst, _Float16 *vec, int dim) {
    // Symmetricly quantize the activations into [-127, 127]
    _Float16 amax = 0.0f;  // Find the abs max in vec
    #pragma omp parallel for reduction(max:amax)
    for (int d = 0; d < dim; d++) {
        _Float16 av = vec[d] >= 0 ? vec[d] : -vec[d];
        if (av > amax) { amax = av; }
    }

    // 1.0f just to make sure we are not dividing by zero
    float qscale = amax > 0.0f ? amax / 127.0f : 1.0f;

    // Quantize vec into qbuf
    #pragma omp simd
    for (int d = 0; d < dim; d++) {
        int q = (int)roundf((float)vec[d] / qscale);
        if (q > 127) { q = 127; }
        else if (q < -127) { q = -127; }
        dst[d] = (int8_t)q;
    }

    return (_Float16)qscale;
}

void quantize_act_rows(
    int8_t *restrict dst,
    const _Float16 *restrict src,
    int m, int n,
    _Float16 *restrict scales_rows
) {
    #pragma omp parallel for
    for (int r = 0; r < m; r++) {
        const _Float16 *row = src + r * n;
        _Float16 amax = 0.0f;
        for (int c = 0; c < n; c++) {
            _Float16 av = row[c] >= 0 ? row[c] : -row[c];
            if (av > amax) { amax = av; }
        }
        float qscale = amax > 0.0f ? (float)amax / 127.0f : 1.0f;
        scales_rows[r] = (_Float16)qscale;

        #pragma omp simd
        for (int c = 0; c < n; c++) {
            int q = (int)roundf((float)row[c] / qscale);
            if (q > 127) { q = 127; }
            else if (q < -127) { q = -127; }
            dst[r * n + c] = (int8_t)q;
        }
    }
}

void gemv_fp16(
    _Float16 *restrict dst,
    const Linear *restrict mat,
    _Float16 *restrict vec,
    int m, int n
) {
    // fp16 mat (m, n) @ fp16 vec (n,) = fp16 dst (m,)
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        float sum = 0;
        for (int j = 0; j < n; j++) {
            sum += (float)mat->fp16[i*n + j] * (float)vec[j];
        }
        dst[i] = clamp_fp16((_Float16)sum);
    }
}

void gemv_int8(
    _Float16 *restrict dst,
    const Linear *restrict mat,
    int8_t *restrict vec,
    int m, int n,
    _Float16 qscale
) {
    float fqscale = (float)qscale;

    // int8 mat (m, n) @ int8 vec (n,) = fp16 dst (m,)
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        int32_t sum = 0;
        for (int j = 0; j < n; j++) {
            sum += (int32_t)mat->i8.q[i*n + j] * (int32_t)vec[j];
        }
        // Dequantize
        float val = (float)sum * fqscale * (float)mat->i8.scales[i];
        dst[i] = clamp_fp16((_Float16)val);
    }
}

void gemm_fp16(_Float16 *dst, const Linear *mat, _Float16 *src, int m, int n, int k) {
    // fp16 src (m, k) @ (fp16 mat (n, k)).T = fp16 dst (m, n)
    #pragma omp parallel for
    for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
        float sum = 0.0f;
        const _Float16 *src_row = src + i * k;
        const _Float16 *w_row = mat->fp16 + j * k;
        #pragma omp simd reduction(+:sum)
        for (int l = 0; l < k; l++) {
            sum += (float)src_row[l] * (float)w_row[l];
        }
        dst[i * n + j] = clamp_fp16((_Float16)sum);
    }
}

void gemm_int8(
    _Float16 *restrict dst,
    const Linear *restrict mat,
    const int8_t *restrict src,
    int m, int n, int k,
    const _Float16 *restrict scales_rows
) {
    // int8 src (m, k) @ (int8 mat (n, k)).T = fp16 dst (m, n)
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        float fscale = (float)scales_rows[i];
        for (int j = 0; j < n; j++) {
            int32_t sum = 0;
            const int8_t *src_row = src + i * k;
            const int8_t *w_row = mat->i8.q + j * k;

            #pragma omp simd reduction(+:sum)
            for (int l = 0; l < k; l++) {
                sum += (int32_t)src_row[l] * (int32_t)w_row[l];
            }

            // Dequantize
            float val = (float)sum * fscale * (float)mat->i8.scales[j];
            dst[i * n + j] = clamp_fp16((_Float16)val);
        }
    }
}

_Float16 dot(_Float16 *v1, _Float16 *v2, int dim) {
    // Must use float instead of _Float16
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum += (float)v1[i] * (float)v2[i];
    }
    return (_Float16)sum;
}

void softmax(_Float16 *dst, _Float16 *src, int dim) {
    _Float16 max = -(_Float16)INFINITY;
    for (int i = 0; i < dim; i++) {
        if (src[i] > max) { max = src[i]; }
    }

    float expsum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float val = expf((float)(src[i] - max));
        dst[i] = (_Float16)val;
        expsum += val;
    }

    #pragma omp simd
    for (int i = 0; i < dim; i++) {
        dst[i] = (_Float16)((float)dst[i] / expsum);
    }
}

void softmax_omp(_Float16 *dst, _Float16 *src, int dim) {
    // OpenMP parallelized version of softmax
    _Float16 max = -(_Float16)INFINITY;
    #pragma omp parallel for reduction(max:max)
    for (int i = 0; i < dim; i++) {
        if (src[i] > max) { max = src[i]; }
    }

    float expsum = 0.0f;
    #pragma omp parallel for reduction(+:expsum)
    for (int i = 0; i < dim; i++) {
        float val = expf((float)(src[i] - max));
        dst[i] = (_Float16)val;
        expsum += val;
    }

    #pragma omp parallel for
    for (int i = 0; i < dim; i++) {
        dst[i] = (_Float16)((float)dst[i] / expsum);
    }
}

static void warn_stats(const char *name, const _Float16 *data, int len, int layer, int pos, int freq) {
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

    float min = INFINITY, max = -INFINITY;
    double sum = 0.0, sumsq = 0.0;
    int inf_cnt = 0, nan_cnt = 0;

    for (int i = 0; i < len; i++) {
        float v = (float)data[i];
        if (isinf(v)) { inf_cnt++; }
        if (isnan(v)) { nan_cnt++; }
        if (v < min) { min = v; }
        if (v > max) { max = v; }
        sum += v;
        sumsq += (double)v * v;
    }

    if (max >= 0.8 * FP16_MAX || min <= -0.8 * FP16_MAX || inf_cnt >= 1 || nan_cnt >= 1) {        
        double mean = sum / len;
        double var = sumsq / len - mean * mean;
        double std = sqrt(var > 0 ? var : 0);

        printf(
            "\nWARNING: [L%02d P%04d] %-20s: min=%9.3f max=%9.3f mean=%9.3f std=%9.3f inf=%d nan=%d\n",
            layer, pos, name, min, max, mean, std, inf_cnt, nan_cnt
        );
    }
#endif
}

void forward_siglip(
    SigLIPVisionEncoder *enc, GemmaConfig *conf, SigLIPConfig *vconf, SigLIPBuffer *buf,
    _Float16 *img, _Float16 *out
) {
    SigLIPConfig *vconf = enc->config;

    int C = vconf->hidden_dim;
    int patch = vconf->patch_size;
    int ppi = vconf->image_size / patch;
    int N = ppi * ppi;
    int side_len = (int)roundf(sqrtf((float)conf->tokens_per_image));
    int kernel_size = (vconf->image_size / vconf->patch_size) / side_len;

    int head_dim = C / vconf->n_heads;
    int mlp_dim = vconf->mlp_hidden_dim;

    if (conf->quant) {
        fprintf(stderr, "Error: SigLIP int8 quantization not supported in this build.\n");
        exit(1);
    }

    // Patch Embedding
    int in_dim = 3 * patch * patch;
    for (int oy = 0; oy < ppi; oy++)
    for (int ox = 0; ox < ppi; ox++) {
        int token_idx = oy * ppi + ox;
        for (int oc = 0; oc < C; oc++) {
            float sum = 0.0f;
            for (int c = 0; c < 3; c++)
            for (int py = 0; py < patch; py++)
            for (int px = 0; px < patch; px++) {
                int in_idx = (
                    c * vconf->image_size * vconf->image_size
                    + (oy * patch + py) * vconf->image_size
                    + (ox * patch + px)
                );
                int w_idx = oc * in_dim + c * patch * patch + py * patch + px;
                sum += (float)enc->patch_emb[w_idx] * (float)img[in_idx];
            }
            buf->x[token_idx * C + oc] = clamp_fp16((_Float16)(sum + (float)enc->patch_emb_b[oc]));
        }
    }
    warn_stats("patch_emb", buf->x, N * C, 0, 0, 0);

    // Position Embedding
    for (int i = 0; i < N; i++) {
        _Float16 *pos_vec = enc->pos_embedding.fp16 + i * C;
        for (int j = 0; j < C; j++) {
            buf->x[i * C + j] = clamp_fp16(buf->x[i * C + j] + pos_vec[j]);
        }
    }
    warn_stats("pos_emb", buf->x, N * C, 0, 0, 0);

    // Encoder Layers
    for (int l = 0; l < vconf->n_layers; l++) {
        SigLIPEncoderLayer *layer = enc->layers[l];

        memcpy(buf->resid, buf->x, N * C * sizeof(_Float16));
        layernorm(buf->x, buf->x, layer->n1, layer->n1_b, C, vconf->eps);
        warn_stats("ln1", buf->x, N * C, l, 0, 8);

        // QKV projections
        if (!conf->quant) {
            gemm_fp16(buf->xq, &layer->wq, buf->x, N, C, C);
            gemm_fp16(buf->xk, &layer->wk, buf->x, N, C, C);
            gemm_fp16(buf->xv, &layer->wv, buf->x, N, C, C);
        } else {
            quantize_act_rows(buf->x_i8, buf->x, N, C, buf->x_scales);
            gemm_int8(buf->xq, &layer->wq, buf->x_i8, N, C, C, buf->x_scales);
            gemm_int8(buf->xk, &layer->wk, buf->x_i8, N, C, C, buf->x_scales);
            gemm_int8(buf->xv, &layer->wv, buf->x_i8, N, C, C, buf->x_scales);
        }
        warn_stats("q", buf->xq, N * C, l, 0, 8);
        warn_stats("k", buf->xk, N * C, l, 0, 8);
        warn_stats("v", buf->xv, N * C, l, 0, 8);

        // Add biases
        for (int i = 0; i < N; i++)
        for (int j = 0; j < C; j++) {
            int idx = i * C + j;
            buf->xq[idx] += layer->bq[j];
            buf->xk[idx] += layer->bk[j];
            buf->xv[idx] += layer->bv[j];
        }

        // Attention
        memset(buf->att_out, 0, N * C * sizeof(_Float16));
        float scale = 1.0f / sqrtf((float)head_dim);

        #pragma omp parallel for
        for (int h = 0; h < vconf->n_heads; h++) {
            _Float16 *scores = buf->scores + h * N * N;  // (N, N) for this head

            // Compute scores
            for (int i = 0; i < N; i++) {
                _Float16 *q_vec = buf->xq + i * C + h * head_dim;
                for (int j = 0; j < N; j++) {
                    _Float16 *k_vec = buf->xk + j * C + h * head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        dot += (float)q_vec[d] * (float)k_vec[d];
                    }
                    scores[i * N + j] = (_Float16)(dot * scale);
                }
                // Softmax over j
                softmax(scores + i * N, scores + i * N, N);
            }

            // Weighted sum of values
            for (int i = 0; i < N; i++) {
                _Float16 *out_vec = buf->att_out + i * C + h * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    float sum = 0.0f;
                    for (int j = 0; j < N; j++) {
                        _Float16 *v_vec = buf->xv + j * C + h * head_dim;
                        sum += (float)v_vec[d] * (float)scores[i * N + j];
                    }
                    out_vec[d] = (_Float16)sum;
                }
            }
        }

        // Output projection
        if (!conf->quant) {
            gemm_fp16(buf->x, &layer->wo, buf->att_out, N, C, C);
        } else {
            quantize_act_rows(buf->x_i8, buf->x, N, C, buf->x_scales);
            gemm_int8(buf->x, &layer->wo, buf->att_out, N, C, C, buf->x_scales);
        }
        // Add output bias
        for (int i = 0; i < N; i++)
        for (int j = 0; j < C; j++) {
            buf->x[i * C + j] += layer->bo[j];
        }
        warn_stats("att_out", buf->x, N * C, l, 0, 8);

        // Residual connection
        for (int i = 0; i < N * C; i++) {
            buf->x[i] = clamp_fp16(buf->x[i] + buf->resid[i]);
        }
        warn_stats("resid1", buf->x, N * C, l, 0, 8);

        memcpy(buf->resid, buf->x, N * C * sizeof(_Float16));
        layernorm(buf->x, buf->x, layer->n2, layer->n2_b, C, vconf->eps);
        warn_stats("ln2", buf->x, N * C, l, 0, 8);

        // x @ fc1 = mlp_hidden
        if (!conf->quant) {
            gemm_fp16(buf->mlp_hidden, &layer->w1, buf->x, N, mlp_dim, C);
        } else {
            quantize_act_rows(buf->x_i8, buf->x, N, C, buf->x_scales);
            gemm_int8(buf->mlp_hidden, &layer->w1, buf->x_i8, N, mlp_dim, C, buf->x_scales);
        }

        for (int i = 0; i < N; i++)
        for (int j = 0; j < mlp_dim; j++) {
            // Apply fc1 biases
            float val = (float)buf->mlp_hidden[i * mlp_dim + j] + (float)layer->b1[j];
            // GELU tanh approximation
            float c = 0.79788456080287f;
            val = 0.5f * val * (1.0f + tanhf(c * (val + 0.044715f * val * val * val)));
            buf->mlp_hidden[i * mlp_dim + j] = (_Float16)val;
        }
        warn_stats("mlp_hidden", buf->mlp_hidden, N * mlp_dim, l, 0, 8);

        // mlp_hidden @ fc2 = x
        if (!conf->quant) {
            gemm_fp16(buf->x, &layer->w2, buf->mlp_hidden, N, C, mlp_dim);
        } else {
            quantize_act_rows(buf->mlp_hidden_i8, buf->mlp_hidden, N, mlp_dim, buf->mlp_hidden_scales);
            gemm_int8(buf->x, &layer->w2, buf->mlp_hidden_i8, N, C, mlp_dim, buf->mlp_hidden_scales);
        }
        // x += b2
        for (int i = 0; i < N; i++)
        for (int j = 0; j < C; j++) {
            buf->x[i * C + j] += layer->b2[j];
        }
        warn_stats("mlp_out", buf->x, N * C, l, 0, 8);
        // Residual connection
        for (int i = 0; i < N * C; i++) {
            buf->x[i] = clamp_fp16(buf->x[i] + buf->resid[i]);
        }
        warn_stats("resid2", buf->x, N * C, l, 0, 8);
    }

    // Post layernorm
    layernorm(buf->x, buf->x, enc->post_norm, enc->post_norm_b, C, vconf->eps);
    warn_stats("post_ln", buf->x, N * C, 0, 0, 0);

    // Average pooling
    int tokens_per_image = side_len * side_len;
    for (int oy = 0; oy < side_len; oy++)
    for (int ox = 0; ox < side_len; ox++) {
        int out_idx = (oy * side_len + ox) * C;
        for (int d = 0; d < C; d++) {
            float sum = 0.0f;
            for (int ky = 0; ky < kernel_size; ky++)
            for (int kx = 0; kx < kernel_size; kx++) {
                int py = oy * kernel_size + ky;
                int px = ox * kernel_size + kx;
                int token_idx = (py * ppi + px) * C + d;
                sum += (float)buf->x[token_idx];
            }
            buf->x[out_idx + d] = (_Float16)(sum / (kernel_size * kernel_size));
        }
    }
    warn_stats("avg_pool", buf->x, tokens_per_image * C, 0, 0, 0);

    // RMSNorm
    rmsnorm(buf->x, buf->x, enc->norm, C, vconf->eps);
    warn_stats("rmsnorm", buf->x, tokens_per_image * C, 0, 0, 0);

    // Final projection
    gemm_fp16(out, &enc->proj, buf->x, tokens_per_image, conf->embed_dim, C);
    warn_stats("proj", out, tokens_per_image * conf->embed_dim, 0, 0, 0);
}

void forward_gemma(GemmaModel *model, GemmaBuffer *buf, int tok, int pos) {
    GemmaConfig *conf = model->config;
    
    if (pos >= buf->cache_len) {
        printf("\nKV Cache is full, exiting...");
        free_buffer(buf, conf->quant);
        free_model(model);
        exit(1);
    }

    // I can't directly scale the embedding weight beforehand (see export.py),
    // so I have to scale it here at run time. The activation scaler that I used
    // in export.py was 1/sqrt(embed_dim), and the original Gemma training uses
    // sqrt(embed_dim) as embed_scale, they canceled out to 1.0f

    _Float16 embed_scale = 1.0f;  // equivalent to sqrt(embed_dim) * (1/sqrt(embed_dim))
    if (conf->quant) {
        // Dequantize
        embed_scale *= model->embedding.i8.scales[tok];
    }

    // x = embedding[tok] * embed_scale
    #pragma omp parallel for
    for (int i = 0; i < conf->embed_dim; i++) {
        if (!conf->quant) {
            buf->x[i] = model->embedding.fp16[tok*conf->embed_dim + i] * embed_scale;
        } else {
            buf->x[i] = model->embedding.i8.q[tok*conf->embed_dim + i] * embed_scale;
        }
    }

    warn_stats("embedding", buf->x, conf->embed_dim, 0, pos, 0);
    
    int q_size = conf->n_heads * conf->head_dim;
    int kv_size = conf->n_kv_heads * conf->head_dim;
    int hd_half = conf->head_dim / 2;

    // Precompute cos & sin for all frequencies (used in RoPE)
    #pragma omp simd
    for (int d = 0; d < hd_half; d++) {
        float freq;
        float e = (float)(-2*d) / conf->head_dim;

        // Rotation angles for sliding window attentions
        freq = powf(conf->local_theta, e);
        buf->csfreqs_slid[d*2] = (_Float16)cosf(freq * pos);
        buf->csfreqs_slid[d*2 + 1] = (_Float16)sinf(freq * pos);

        // Rotation angles for full attentions
        freq = powf(conf->global_theta, e);
        buf->csfreqs_full[d*2] = (_Float16)cosf(freq * pos);
        buf->csfreqs_full[d*2 + 1] = (_Float16)sinf(freq * pos);
    }

    _Float16 att_scale = (_Float16)(1.0f / sqrtf((float)conf->q_pre_attn_scalar));

    for (int l = 0; l < conf->n_layers; l++) {
        GemmaDecoderLayer *layer = model->layers[l];

        memcpy(buf->resid, buf->x, conf->embed_dim*sizeof(_Float16));

        rmsnorm_omp(buf->x, buf->x, layer->n1, conf->embed_dim, conf->eps);
        warn_stats("norm1", buf->x, conf->embed_dim, l, pos, 8);

        // The attention block
        if (!conf->quant) {
            gemv_fp16(buf->xq, &layer->wq, buf->x, q_size, conf->embed_dim);  // (n_heads, head_dim)
            gemv_fp16(buf->xk, &layer->wk, buf->x, kv_size, conf->embed_dim);  // (n_kv_heads, head_dim)
            gemv_fp16(buf->xv, &layer->wv, buf->x, kv_size, conf->embed_dim);  // (n_kv_heads, head_dim)
        } else {
            float xscale = quantize_act(buf->x_i8, buf->x, conf->embed_dim);
            gemv_int8(buf->xq, &layer->wq, buf->x_i8, q_size, conf->embed_dim, xscale);
            gemv_int8(buf->xk, &layer->wk, buf->x_i8, kv_size, conf->embed_dim, xscale);
            gemv_int8(buf->xv, &layer->wv, buf->x_i8, kv_size, conf->embed_dim, xscale);
        }

        if (conf->use_qk_norm) {
            // Query RMSNorm
            for (int h = 0; h < conf->n_heads; h++) {
                _Float16 *xq_head = buf->xq + h*conf->head_dim;
                // Use the non-threading version here since we are running this over every head
                rmsnorm(xq_head, xq_head, layer->nq, conf->head_dim, conf->eps);
            }
            // Key RMSNorm
            for (int h = 0; h < conf->n_kv_heads; h++) {
                _Float16 *xk_head = buf->xk + h*conf->head_dim;
                rmsnorm(xk_head, xk_head, layer->nk, conf->head_dim, conf->eps);
            }
        }

        bool is_local = conf->attn_local_layers[l];
        _Float16 *freqs_cs = is_local ? buf->csfreqs_slid : buf->csfreqs_full;
        
        // These used to be a single merged loop, but I splitted it into two separate loops for higher
        // performance

        // Apply RoPE to queries
        #pragma omp parallel for
        for (int h = 0; h < conf->n_heads; h++) {
            int o = h * conf->head_dim;  // Offset for current head
            _Float16 *xq = buf->xq + o;
            #pragma omp simd
            for (int i = 0; i < hd_half; i++) {
                float cfr = (float)freqs_cs[2*i];
                float sfr = (float)freqs_cs[2*i + 1];
                float a = (float)xq[i];  // Index in the first half vector
                float b = (float)xq[i + hd_half];  // Index in the second half vector
                xq[i] = (_Float16)(a * cfr - b * sfr);
                xq[i + hd_half] = (_Float16)(a * sfr + b * cfr);
            }
        }

        // Apply RoPE to keys
        #pragma omp parallel for
        for (int h = 0; h < conf->n_kv_heads; h++) {
            int o = h * conf->head_dim;  // Offset for current head
            _Float16 *xk = buf->xk + o;
            #pragma omp simd
            for (int i = 0; i < hd_half; i++) {
                float cfr = (float)freqs_cs[2*i];
                float sfr = (float)freqs_cs[2*i + 1];
                float a = (float)xk[i];  // Index in the first half vector
                float b = (float)xk[i + hd_half];  // Index in the second half vector
                xk[i] = (_Float16)(a * cfr - b * sfr);
                xk[i + hd_half] = (_Float16)(a * sfr + b * cfr);
            }
        }

        int entry_sz = conf->n_kv_heads * conf->head_dim;
        int layer_sz = 2*buf->cache_len * entry_sz;

        // (cache_len, n_kv_heads, head_dim)
        _Float16 *k_cache = buf->kv_cache + l*layer_sz;
        _Float16 *v_cache = k_cache + buf->cache_len * entry_sz;

        // Write to key & value cache
        memcpy(k_cache + pos*entry_sz, buf->xk, entry_sz*sizeof(_Float16));
        memcpy(v_cache + pos*entry_sz, buf->xv, entry_sz*sizeof(_Float16));

        // Whether the current layer uses sliding window attention
        bool local_att = is_local && pos >= conf->sliding_window;
        // Starting position of attention
        int spos = local_att ? (pos + 1 - conf->sliding_window) : 0;
        int attlen = pos + 1 - spos;  // Include the current pos

        // Iterate over all the attention heads
        #pragma omp parallel for
        for (int h = 0; h < conf->n_heads; h++) {
            int h_kv = h * conf->n_kv_heads / conf->n_heads;
            _Float16 *xq_head = buf->xq + h*conf->head_dim;  // xq[h, :]
            _Float16 *xk_head = k_cache + spos*entry_sz + h_kv*conf->head_dim;  // k_cache[spos:, h_kv, :]
            _Float16 *att_head = buf->att + h*buf->cache_len + spos;  // att[h, spos:]

            // Compute dot product of the current query across all the keys
            for (int t = 0; t < attlen; t++) {
                att_head[t] = dot(xq_head, xk_head + t*entry_sz, conf->head_dim) * att_scale;
            }
    
            // Attention score softcapping
            if (conf->attn_softcapping != 0.0f) {
                for (int t = 0; t < attlen; t++) {
                    float val = (float)att_head[t] / conf->attn_softcapping;
                    att_head[t] = (_Float16)(tanhf(val) * conf->attn_softcapping);
                }
            }

            // Softmax
            softmax(att_head, att_head, attlen);

            // Compute output
            _Float16 *xv_head = v_cache + spos*entry_sz + h_kv*conf->head_dim;  // v_cache[spos:, h_kv, :]
            _Float16 *xo_head = buf->xo + h*conf->head_dim;
            // xo_head (head_dim,) = att_head (attlen,) @ xv_head (attlen, head_dim)
            for (int d = 0; d < conf->head_dim; d++) {
                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for (int t = 0; t < attlen; t++) {
                    sum += (xv_head + t*entry_sz)[d] * att_head[t];
                }
                xo_head[d] = sum;
            }
        }

        // Map xo back to x
        if (!conf->quant) {
            gemv_fp16(buf->x, &layer->wo, buf->xo, conf->embed_dim, q_size);
        } else {
            float xoscale = quantize_act(buf->xo_i8, buf->xo, q_size);
            gemv_int8(buf->x, &layer->wo, buf->xo_i8, conf->embed_dim, q_size, xoscale);
        }
        warn_stats("attn_out", buf->x, conf->embed_dim, l, pos, 8);

        rmsnorm_omp(buf->x, buf->x, layer->n2, conf->embed_dim, conf->eps);
        warn_stats("norm2", buf->x, conf->embed_dim, l, pos, 8);

        // Combine the residual stream
        _Float16 *restrict x = buf->x;
        _Float16 *restrict resid = buf->resid;
        #pragma omp simd
        for (int d = 0; d < conf->embed_dim; d++) {
            // Sometimes the residual stream accumulates huge values on certain channels,
            // especially in pretrained/bigger models (Sun et al. https://arxiv.org/abs/2402.17762)
            // It works fine in fp32 or bf16, but it can easily overflow fp16 and become
            // inf, causing all the activations turning into nan after the next RMSNorm,
            // so we need to clamp it

            // NOTE: Actually this should never trigger now since I added activation scalers
            // afterwards (see export.py), the clamp here is more of a last-resort safety net
            buf->x[d] = clamp_fp16(x[d] + resid[d]);
        }
        warn_stats("resid1", buf->x, conf->embed_dim, l, pos, 8);

        memcpy(buf->resid, buf->x, conf->embed_dim * sizeof(_Float16));

        if (conf->pre_ffwd_norm) {
            rmsnorm_omp(buf->x, buf->x, layer->n3, conf->embed_dim, conf->eps);
        }

        // MLP feedforward layer
        if (!conf->quant) {
            gemv_fp16(buf->xg, &layer->w2, buf->x, conf->mlp_hidden_dim, conf->embed_dim);
            gemv_fp16(buf->xu, &layer->w1, buf->x, conf->mlp_hidden_dim, conf->embed_dim);
        } else {
            float xscale = quantize_act(buf->x_i8, buf->x, conf->embed_dim);
            gemv_int8(buf->xg, &layer->w2, buf->x_i8, conf->mlp_hidden_dim, conf->embed_dim, xscale);
            gemv_int8(buf->xu, &layer->w1, buf->x_i8, conf->mlp_hidden_dim, conf->embed_dim, xscale);
        }

        // GELU layer
        #pragma omp parallel for
        for (int d = 0; d < conf->mlp_hidden_dim; d++) {
            // Tanh approximation of GELU
            float x = (float)buf->xg[d];
            float c = 0.79788456080287f;  // sqrt(2 / pi)
            x = 0.5*x * (1 + tanhf(c * (x + 0.044715 * x*x*x)));
            buf->xg[d] = (_Float16)x;
            // Fuse xg * xu into xg
            buf->xg[d] *= buf->xu[d];
        }
        
        if (!conf->quant) {
            gemv_fp16(buf->x, &layer->w3, buf->xg, conf->embed_dim, conf->mlp_hidden_dim);
        } else {
            float xscale = quantize_act(buf->xg_i8, buf->xg, conf->mlp_hidden_dim);
            gemv_int8(buf->x, &layer->w3, buf->xg_i8, conf->embed_dim, conf->mlp_hidden_dim, xscale);
        }
        warn_stats("down_proj", buf->x, conf->embed_dim, l, pos, 8);

        if (conf->post_ffwd_norm) {
            rmsnorm_omp(buf->x, buf->x, layer->n4, conf->embed_dim, conf->eps);
            warn_stats("norm4", buf->x, conf->embed_dim, l, pos, 8);
        }

        // Combine the residual stream
        x = buf->x;
        resid = buf->resid;
        #pragma omp simd
        for (int d = 0; d < conf->embed_dim; d++) {
            buf->x[d] = clamp_fp16(x[d] + resid[d]);
        }
        warn_stats("resid2", buf->x, conf->embed_dim, l, pos, 8);
    }

    // Final RMSNorm
    rmsnorm_omp(buf->x, buf->x, model->final_norm, conf->embed_dim, conf->eps);
}

int argmax(_Float16 *logits, int vocab_size) {
    // Pick the index with the max value
    int max_idx = -1;
    _Float16 max_val = -(_Float16)INFINITY;
    #pragma omp parallel
    {
        int local_idx = -1;
        _Float16 local_val = -(_Float16)INFINITY;
        #pragma omp for nowait
        for (int i = 0; i < vocab_size; i++) {
            if (logits[i] > local_val) { local_val = logits[i]; local_idx = i; }
        }
        #pragma omp critical
        {
            // Only one thread is able to run this at a time
            if (local_val > max_val) { max_val = local_val; max_idx = local_idx; }
        }
    }
    return max_idx;
}

typedef struct { _Float16 val; int idx; } FloatIdx;

static inline void swap_fi(FloatIdx *a, FloatIdx *b) {
    FloatIdx t = *a; *a = *b; *b = t;
}

static uint32_t qs_rand_state = 3418323524;
static inline uint32_t qs_rand(void) {
    qs_rand_state ^= qs_rand_state << 13;
    qs_rand_state ^= qs_rand_state >> 7;
    qs_rand_state ^= qs_rand_state << 17;
    return (uint32_t)qs_rand_state;
}

int partition_desc(FloatIdx *arr, int lo, int hi) {
    // Pick the pivot randomly (use a seperate rand sequence)
    int r = lo + qs_rand() % (hi - lo + 1);
    swap_fi(&arr[r], &arr[hi]);

    _Float16 pivot = arr[hi].val;
    int i = lo;
    for (int j = lo; j < hi; j++) {
        // Put the greater one on the left
        if (arr[j].val > pivot) { swap_fi(&arr[i++], &arr[j]); }
    }
    swap_fi(&arr[i], &arr[hi]);
    return i;
}

void quickselect_topk(FloatIdx *arr, int lo, int hi, int k_idx) {
    while (lo < hi) {
        int p = partition_desc(arr, lo, hi);
        if (p == k_idx) return;
        else if (p < k_idx) { lo = p + 1; }
        else { hi = p - 1; }
    }
}

void apply_topk(_Float16 *logits, FloatIdx *logit_indices, int vocab_size, int k) {
    if (k <= 0) { k = 1; }
    if (k > vocab_size) { k = vocab_size; }

    // Record index info
    #pragma omp parallel for
    for (int i = 0; i < vocab_size; i++) {
        logit_indices[i].idx = i;
        logit_indices[i].val = logits[i];
    }
    quickselect_topk(logit_indices, 0, vocab_size - 1, k - 1);

    // Keep the top k channels
    #pragma omp parallel for
    for (int i = 0; i < vocab_size; i++) { logits[i] = -(_Float16)INFINITY; }
    for (int i = 0; i < k; i++) { logits[logit_indices[i].idx] = logit_indices[i].val; }
}

void sift_down(FloatIdx *arr, int n, int i) {
    // Make sure the parent node arr[i] is greater than its children in the heap
    for (;;) {
        // l & r are the two children node
        int l = 2*i + 1, r = 2*i + 2, largest = i;
        if (l < n && arr[l].val > arr[largest].val) largest = l;
        if (r < n && arr[r].val > arr[largest].val) largest = r;
        if (largest == i) break;
        swap_fi(&arr[i], &arr[largest]);
        i = largest;
    }
}

void build_heap(FloatIdx *arr, int n) {
    for (int i = n/2 - 1; i >= 0; i--) { sift_down(arr, n, i); }
}

void apply_topp(
    _Float16 *logits, _Float16 *fpbuf, FloatIdx *logit_indices,
    int vocab_size, int k, float p
) {
    // Softmax to get the probs, store in fpbuf
    softmax_omp(fpbuf, logits, vocab_size);
    int heap_size = (k == 0) ? vocab_size : k;

    if (k == 0) {
        #pragma omp parallel for
        for (int i = 0; i < vocab_size; i++) {
            logit_indices[i].idx = i;
            logit_indices[i].val = fpbuf[i];
        }
    } else {
        // topk typically uses values less than 100, no need to use omp here
        for (int i = 0; i < k; i++) {
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

    while (heap_size > 0) {
        // Pop the current max prob
        FloatIdx top = logit_indices[0];
        logit_indices[0] = logit_indices[--heap_size];  // Put the last element to the top
        sift_down(logit_indices, heap_size, 0);  // O(log(vocab_size))

        logits[top.idx] = fpbuf[top.idx];
        cum += (float)top.val;
        if (cum >= p) break;
    }
}

void apply_rpen(_Float16* logits, bool *visited, int vocab_size, float rpen) {
    // rpen short for Repetition Penalty
    #pragma omp parallel for
    for (int i = 0; i < vocab_size; i++) {
        if (!visited[i]) continue;
        _Float16 val = logits[i];
        if (val > 0.0f) { logits[i] = val / rpen; }
        else { logits[i] = val * rpen; }
    }
}

void sample(
    GemmaModel *model,
    GemmaBuffer *buf,
    int *tokens,
    int seqlen,
    float temperature,
    int topk,
    float topp,
    float rpen,
    int *(*token_callback)(int, GemmaTokenizer *)
) {
    if (tokens == SAMPLE_ABORT) return;

    GemmaConfig *conf = model->config;
    int vs = conf->vocab_size;

    // Boolean flags
    bool dosample = temperature != 0 && topk != 1;
    bool use_topk = dosample && topk != 0;
    bool use_topp = dosample && topp < 1.0f;
    bool use_rpen = dosample && rpen > 1.0f;

    // bool array indicating which tokens have already been processed
    // Used in rpen (repetition penalty)
    bool *visited = NULL;
    if (use_rpen) {
        visited = safe_calloc(vs, sizeof(bool), "visited tokens");
    }
    int pos, token;

    // Prefill all the prompt tokens except the last one
    clock_t prefill_start = clock();
    pos = 0;
    for (int *t = tokens; *t != EOT_SENTINEL && *(t+1) != EOT_SENTINEL; t++) {
        if (g_interrupted) {
            if (use_rpen) { free(visited); }
            return;
        }
        if (use_rpen) { visited[*t] = true; }
        forward_gemma(model, buf, *t, pos++);  // forward in the current pos
    }
    clock_t prefill_end = clock();
    double prefill_elapsed = (double)(prefill_end - prefill_start) / CLOCKS_PER_SEC;
    int prefill_tokens = pos;

    token = tokens[pos];
    _Float16 *probs = NULL;

    // Only allocate probs if needed
    if (temperature != 0.0f) {
        probs = safe_malloc(vs * sizeof(*probs), "probs");
    }

    // Logit indices for topk & topp
    FloatIdx *logit_indices = NULL;
    if (use_topk || use_topp) {
        logit_indices = safe_malloc(vs * sizeof(*logit_indices), "FloatIdx array");
    }

    // Record tok/s
    clock_t gen_start = clock();
    int gen_tokens = 0;

    for (; pos < seqlen; pos++) {
        if (g_interrupted) break;

        if (use_rpen) visited[token] = true;
        forward_gemma(model, buf, token, pos);

        // Compute logits
        if (!conf->quant) {
            gemv_fp16(buf->logits, &model->embedding, buf->x, vs, conf->embed_dim);
        } else {
            float xscale = quantize_act(buf->x_i8, buf->x, conf->embed_dim);
            gemv_int8(buf->logits, &model->embedding, buf->x_i8, vs, conf->embed_dim, xscale);
        }

        if (!dosample) {
            // Argmax sampling
            token = argmax(buf->logits, vs);
        } else {
            // Logit softcapping
            if (conf->logit_softcapping != 0.0f) {
                for (int d = 0; d < conf->vocab_size; d++) {
                    float val = (float)buf->logits[d] / conf->logit_softcapping;
                    buf->logits[d] = (_Float16)(tanhf(val) * conf->logit_softcapping);
                }
            }

            // Apply the temperature
            #pragma omp parallel for
            for (int d = 0; d < vs; d++) {
                buf->logits[d] /= (_Float16)temperature;
            }

            if (use_topk) {
                apply_topk(buf->logits, logit_indices, vs, topk);
            }
            if (use_topp) {
                apply_topp(buf->logits, probs, logit_indices, vs, topk, topp);
            }
            if (use_rpen) {
                apply_rpen(buf->logits, visited, vs, rpen);
            }

            // Softmax to get the probs
            softmax_omp(probs, buf->logits, vs);

            // Sample from probs
            float r = (float)rand() / (RAND_MAX + 1.0);
            float sum = 0.0;

            token = vs - 1;
            for (int d = 0; d < vs; d++) {
                sum += (float)probs[d];
                if (r < sum) { token = d; break; }
            }
        }

        gen_tokens++;
        int *ret = token_callback(token, model->tokenizer);

        if (ret == SAMPLE_ABORT) break;
        else if (ret != NULL) {
            // Injected a token array (ends with EOT_SENTINEL)
            // Prefill all the tokens except the last one
            int i;
            for (i = 0; (token = ret[i]) != EOT_SENTINEL && ret[i+1] != EOT_SENTINEL; i++) {
                if (g_interrupted) break;
                if (use_rpen) { visited[token] = true; }
                forward_gemma(model, buf, token, ++pos);  // forward in the next pos
            }
            if (g_interrupted) break;
            token = ret[i];  // The last element
        }
        // ret == NULL: do nothing
    }

    clock_t gen_end = clock();
    double gen_elapsed = (double)(gen_end - gen_start) / CLOCKS_PER_SEC;

    // Print prefilling speed
    if (prefill_elapsed > 0.0) {
        printf(
            "\nPrompt processed %d tokens in %.2f seconds (%.2f tok/s)\n",
            prefill_tokens, prefill_elapsed, prefill_tokens / prefill_elapsed
        );
    } else {
        printf("\nPrompt processed %d tokens instantly\n", prefill_tokens);
    }
    // Print generation speed
    if (gen_elapsed > 0.0) {
        printf(
            "Generated %d tokens in %.2f seconds (%.2f tok/s)\n",
            gen_tokens, gen_elapsed, gen_tokens / gen_elapsed
        );
    } else {
        printf("Generated %d tokens instantly\n", gen_tokens);
    }

    free(probs);
    if (use_rpen) free(visited);
    if (use_topk || use_topp) free(logit_indices);
}

int *encode(GemmaTokenizer *tok, const char *sstr, int *tokens, int *n_tokens) {
    unsigned char *str = (unsigned char *)sstr;
    int len = strlen((char *)str);

    // Convert UTF-8 string to tokens of individual codepoints
    int i = 0;
    int tok_i = 0;

    while (i < len) {
        // table from https://zh.wikipedia.org/wiki/UTF-8

        // U+00000-U+00007F  1  0xxxxxxx
        // U+00080-U+0007FF  2  110xxxxx  10xxxxxx
        // U+00800-U+00FFFF  3  1110xxxx  10xxxxxx  10xxxxxx
        // U+10000-U+1FFFFF  4  11110xxx  10xxxxxx  10xxxxxx  10xxxxxx

        int n_bytes;
        int start = i;

        if (str[i] >> 7 == 0) {
            n_bytes = 1;
        } else if (
            i + 1 < len && str[i] >> 5 == 6
            && str[i+1] >> 6 == 2
        ) {
            n_bytes = 2;
        } else if (
            i + 2 < len && str[i] >> 4 == 14
            && str[i+1] >> 6 == 2 && str[i+2] >> 6 == 2
        ) {
            n_bytes = 3;
        } else if (
            i + 3 < len && str[i] >> 3 == 30
            && str[i+1] >> 6 == 2 && str[i+2] >> 6 == 2 && str[i+3] >> 6 == 2
        ) {
            n_bytes = 4;
        } else {
            n_bytes = 1;
        }

        char cstr[5];
        for (int b = 0; b < n_bytes; b++) cstr[b] = str[i++];
        cstr[n_bytes] = '\0';

        int token = get_token_idx(tok, cstr);
        if (token == -1) {
            // Fallback to byte level tokens
            char bstr[10];
            for (int b = start; b < i; b++) {
                snprintf(bstr, 10, "<0x%02X>", (unsigned char)str[b]);
                tokens[tok_i++] = get_token_idx(tok, bstr);
            }
        } else {
            tokens[tok_i++] = token;
        }
    }

    for (;;) {
        int best_rank = 2147483647;
        Merge *best_pair = NULL;

        // Find the merge with best rank
        for (int i = 0; i < tok_i - 1; i++) {
            char *str1 = tok->vocab[tokens[i]];
            char *str2 = tok->vocab[tokens[i + 1]];

            Merge *merge = get_merge_rec(tok, str1, str2);
            if (merge != NULL && merge->rank < best_rank) {
                best_rank = merge->rank;
                best_pair = merge;
            }
        }
        
        if (best_pair == NULL) break;  // No more merges
        
        int i = 0;
        while (i < tok_i - 1) {
            Merge pair = {
                .str1 = tok->vocab[tokens[i]],
                .str2 = tok->vocab[tokens[i + 1]]
            };
            if (cmp_merge(&pair, best_pair) == 0) {
                // Merge the pair, left shift all the tokens on its right side
                char merged[128];
                snprintf(merged, sizeof(merged), "%s%s", best_pair->str1, best_pair->str2);
                tokens[i] = get_token_idx(tok, merged);
                for (int j = i + 1; j < tok_i - 1; j++)
                tokens[j] = tokens[j + 1];
                tok_i--;
            }
            i++;
        }
    }

    *n_tokens += tok_i;
    return tokens;
}

const char *decode(GemmaTokenizer *tok, int id, char *byte_buf) {
    char *s = tok->vocab[id];
    // Byte-fallback token, decode back to the raw byte it represents
    size_t len = strlen(s);
    if (len == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>') {
        unsigned int byte_val;
        if (sscanf(s + 3, "%2x", &byte_val) == 1) {
            byte_buf[0] = (char)byte_val;
            byte_buf[1] = '\0';
            return byte_buf;
        }
    }
    return s;
}

int *generate_callback(int token, GemmaTokenizer *tok) {
    if (token == tok->eos || token == tok->eot) {
        return SAMPLE_ABORT;
    }
    char byte_buf[2];
    printf("%s", decode(tok, token, byte_buf));
    fflush(stdout);
    return NULL;
}

void generate(
    GemmaModel *model, GemmaBuffer *buf,
    const char *prompt,
    int seqlen, float temperature, int topk, float topp, float rpen
) {
    GemmaTokenizer *tok = model->tokenizer;

    printf("%s", prompt);

    int n_tokens = 1;
    int size = strlen(prompt) + 1;
    int tokens[size + 1];
    tokens[0] = tok->bos;
    encode(tok, prompt, tokens + 1, &n_tokens);
    tokens[n_tokens] = EOT_SENTINEL;

    sample(model, buf, tokens, seqlen, temperature, topk, topp, rpen, generate_callback);
}

int *new_turn(GemmaTokenizer *tok, bool bos) {
    if (bos) {
        printf("User: ");
    } else {
        printf("\nUser: ");
    }

    char user_prompt[65536];
    if (fgets(user_prompt, sizeof(user_prompt), stdin) == NULL) {
        fprintf(stderr, "Failed to read user input\n");
        exit(1);
    }
    user_prompt[strcspn(user_prompt, "\n")] = '\0';

    // Apply the gemma chat template
    //
    // [<bos>]<start_of_turn>user\n
    // Hello! Introduce yourself.<end_of_turn>\n
    // <start_of_turn>model\n
    // ...

    int n_tokens = 0;
    int size = strlen("user") + strlen(user_prompt) + strlen("model") + 6;
    if (bos) size++;
    int *tokens = safe_malloc((size + 1) * sizeof(*tokens), "chat tokens");

    if (bos) tokens[n_tokens++] = tok->bos;
    tokens[n_tokens++] = tok->sot;
    encode(tok, "user\n", tokens + n_tokens, &n_tokens);
    encode(tok, user_prompt, tokens + n_tokens, &n_tokens);
    tokens[n_tokens++] = tok->eot;
    tokens[n_tokens++] = get_token_idx(tok, "\n");
    tokens[n_tokens++] = tok->sot;
    encode(tok, "model\n", tokens + n_tokens, &n_tokens);
    tokens[n_tokens++] = EOT_SENTINEL;

    printf("Model: ");
    return tokens;
}

int *chat_callback(int token, GemmaTokenizer *tok) {
    if (token == tok->eos || token == tok->eot) {
        return new_turn(tok, false);
    }
    char byte_buf[2];
    printf("%s", decode(tok, token, byte_buf));
    fflush(stdout);
    return NULL;
}

void chat(
    GemmaModel *model, GemmaBuffer *buf,
    int seqlen, float temperature, int topk, float topp, float rpen
) {
    sample(
        model, buf, new_turn(model->tokenizer, true),
        seqlen, temperature, topk, topp, rpen, chat_callback
    );
}

bool safe_atoui(const char *str, unsigned int *result) {
    if (str == NULL) { return false; }

    char *endptr;
    errno = 0;
    long long val = strtoll(str, &endptr, 10);

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

bool safe_atof(const char *str, float *result) {
    if (str == NULL) { return false; }
    
    char *endptr;
    errno = 0;
    float val = strtof(str, &endptr);
    
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

void print_usage(void) {
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
    "  -s, --seed <N>         Set random seed (default: current time)\n"
    "  -t, --temperature <F>  Set temperature value, must be >= 0.0 (default: 1.0)\n"
    "  -p, --topp <F>         Set top-p sampling value, must be 0.0 < p <= 1.0 (default: 1.0)\n"
    "  -r, --rpen <F>         Set repetition penalty, must be >= 1.0 (default: 1.0)\n"
    "  -i, --prompt <S>       Set input prompt, ignored if chat mode is enabled (default: \"Once upon a time\")\n"
    "  -c, --chat             Enable chat mode\n"
    "  -m, --enable-mm        Enable multi-modal capability, if supported by the model\n"
    "  -h, --help             Display this help message\n"
    "\n"
    "Controls:\n"
    "  Ctrl+C                 Gracefully interrupt generation and exit\n"
    "\n"
    "Examples:\n"
    "  ./gemma model.bin -l 2048 -t 0.8 -c\n"
    "  ./gemma model.bin -i \"Hello I'm a language model,\" --seqlen 4096 --topk 50 --seed 12345\n"
    );
}


#ifdef _WIN32
#include <windows.h>
// The default console encoding is kinda weird in Windows
static void set_utf8_console() {
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
}

// Convert Windows command line to UTF-8 argc/argv
static char** get_utf8_argv(int *argc_out) {
    wchar_t **wargv = CommandLineToArgvW(GetCommandLineW(), argc_out);
    if (!wargv) return NULL;

    char **argv = malloc((*argc_out + 1) * sizeof(char*));
    if (!argv) {
        LocalFree(wargv);
        return NULL;
    }

    for (int i = 0; i < *argc_out; i++) {
        int size = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, NULL, 0, NULL, NULL);
        argv[i] = malloc(size);
        if (!argv[i]) {
            for (int j = 0; j < i; j++) free(argv[j]);
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

static void free_utf8_argv(char **argv, int argc) {
    if (!argv) return;
    for (int i = 0; i < argc; i++) free(argv[i]);
    free(argv);
}
#else
// No problem with POSIX though
static void set_utf8_console() {}
static char** get_utf8_argv(int *argc_out) { (void)argc_out; return NULL; }
static void free_utf8_argv(char **argv, int argc) { (void)argv; (void)argc; }
#endif

static const char* safe_get_arg(int i, int argc, char **argv) {
    if (i + 1 >= argc) {
        fprintf(stderr, "Error: Option '%s' requires an argument.\n", argv[i]);
        return NULL;
    }
    return argv[i + 1];
}

void print_model_config(GemmaModel *model) {
    (void)model;
#ifdef DEBUG
    GemmaConfig *conf = model->config;
    GemmaTokenizer *tok = model->tokenizer;
    
    printf("\n========== Model Configuration ==========\n");
    printf("Architecture:\n");
    printf("  n_layers:           %d\n", conf->n_layers);
    printf("  n_heads:            %d\n", conf->n_heads);
    printf("  n_kv_heads:         %d\n", conf->n_kv_heads);
    printf("  head_dim:           %d\n", conf->head_dim);
    printf("  embed_dim:          %d\n", conf->embed_dim);
    printf("  mlp_hidden_dim:     %d\n", conf->mlp_hidden_dim);
    printf("  vocab_size:         %d\n", conf->vocab_size);
    printf("  max_seq_len:        %d\n", conf->max_seq_len);
    printf("  sliding_window:     %d\n", conf->sliding_window);
    
    printf("\nAttention:\n");
    printf("  q_pre_attn_scalar:  %d\n", conf->q_pre_attn_scalar);
    printf("  attn_softcapping:   %.6f\n", conf->attn_softcapping);
    printf("  use_qk_norm:        %s\n", conf->use_qk_norm ? "true" : "false");
    
    printf("\nRoPE:\n");
    printf("  local_theta:        %.6f\n", conf->local_theta);
    printf("  global_theta:       %.6f\n", conf->global_theta);
    
    printf("\nNormalization:\n");
    printf("  eps:                %.6f\n", conf->eps);
    printf("  pre_ffwd_norm:      %s\n", conf->pre_ffwd_norm ? "true" : "false");
    printf("  post_ffwd_norm:     %s\n", conf->post_ffwd_norm ? "true" : "false");
    
    printf("\nQuantization:\n");
    printf("  quant:              %s\n", conf->quant ? "true (int8)" : "false (fp16)");
    
    printf("\nLogits:\n");
    printf("  logit_softcapping:  %.6f\n", conf->logit_softcapping);
    
    printf("\nAttention Local Layers:\n");
    printf("  ");
    for (int i = 0; i < conf->n_layers; i++) {
        printf("%d", conf->attn_local_layers[i] ? 1 : 0);
        if ((i + 1) % 32 == 0 && i + 1 < conf->n_layers) { printf("\n  "); }
    }
    printf("\n");
    
    printf("\nTokenizer:\n");
    printf("  vocab_size:         %d\n", tok->vocab_size);
    printf("  n_merges:           %d\n", tok->n_merges);
    printf("  bos token:          %d\n", tok->bos);
    printf("  eos token:          %d\n", tok->eos);
    printf("  sot token:          %d\n", tok->sot);
    printf("  eot token:          %d\n", tok->eot);
    printf("  soi token:          %d\n", tok->soi);
    printf("  eoi token:          %d\n", tok->eoi);
    printf("  ist token:          %d\n", tok->ist);
    
    printf("\nMemory Footprint (estimated):\n");
    
    size_t total_bytes = 0;
    int q_size = conf->n_heads * conf->head_dim;
    int kv_size = conf->n_kv_heads * conf->head_dim;
    
    // Embedding
    if (!conf->quant) {
        total_bytes += conf->embed_dim * conf->vocab_size * sizeof(_Float16);
    } else {
        total_bytes += conf->embed_dim * conf->vocab_size * sizeof(int8_t);
        total_bytes += conf->vocab_size * sizeof(_Float16);  // scales
    }
    
    // Each layer
    for (int l = 0; l < conf->n_layers; l++) {
        int n_params = (
            conf->embed_dim * q_size +   // wq
            conf->embed_dim * kv_size +  // wk
            conf->embed_dim * kv_size +  // wv
            q_size * conf->embed_dim +   // wo
            conf->embed_dim * conf->mlp_hidden_dim +  // w1
            conf->embed_dim * conf->mlp_hidden_dim +  // w2
            conf->mlp_hidden_dim * conf->embed_dim    // w3
        )
        if (!conf->quant) {
            total_bytes += n_params * sizeof(_Float16);
        } else {
            total_bytes += n_params * sizeof(int8_t);
            // Add scales
            total_bytes += (
                q_size + kv_size + kv_size + conf->embed_dim +
                conf->mlp_hidden_dim + conf->mlp_hidden_dim + conf->embed_dim
            ) * sizeof(_Float16);
        }
        // Norm weights
        total_bytes += (conf->embed_dim + conf->embed_dim) * sizeof(_Float16);  // n1, n2
        if (conf->use_qk_norm) {
            total_bytes += 2 * conf->head_dim * sizeof(_Float16);  // nq, nk
        }
        if (conf->pre_ffwd_norm) {
            total_bytes += conf->embed_dim * sizeof(_Float16);  // n3
        }
        if (conf->post_ffwd_norm) {
            total_bytes += conf->embed_dim * sizeof(_Float16);  // n4
        }
    }
    total_bytes += conf->embed_dim * sizeof(_Float16);  // final_norm
    
    printf("  Weights:            %.2f GB\n", total_bytes / (1024.0 * 1024.0 * 1024.0));
    
    // KV Cache (per token)
    size_t kv_cache_bytes = conf->n_layers * 2 * kv_size * sizeof(_Float16);
    printf("  KV Cache (tok):     %.2f KB\n", kv_cache_bytes / 1024.0);
    
    printf("=========================================\n\n");
#endif
}

int main(int argc, char **argv) {
    set_utf8_console();
    setup_signal_handler();

    // On Windows, get UTF-8 encoded command line arguments
    char **utf8_argv = get_utf8_argv(&argc);
    if (utf8_argv != NULL) { argv = utf8_argv; }

    if (argc < 2) {
        print_usage();
        printf("\nError: model filename is not provided\n");
        if (utf8_argv != NULL) { free_utf8_argv(utf8_argv, argc); }
        return 1;
    }

    char *modelfile = argv[1];
    if (!strcmp(modelfile, "-h") || !strcmp(modelfile, "--help")) {
        print_usage();
        return 0;
    }

    unsigned int seqlen = 16384;
    unsigned int topk = 0;
    unsigned int seed = (unsigned int)time(NULL);
    float temperature = 1.0;
    float topp = 1.0;
    float rpen = 1.0;
    const char *prompt = "Once upon a time";
    bool chatmode = false;
    bool enable_mm = false;

    // Parse the command line arguments
    for (int i = 2; i < argc; i++) {
        const char *arg = argv[i];

        if (strcmp(arg, "-l") == 0 || strcmp(arg, "--seqlen") == 0) {
            const char *val = safe_get_arg(i++, argc, argv);
            if (!val) { return 1; }
            if (!safe_atoui(val, &seqlen)) {
                fprintf(stderr, "Invalid number for --seqlen: %s\n", val);
                return 1;
            }
        } else if (strcmp(arg, "-k") == 0 || strcmp(arg, "--topk") == 0) {
            const char *val = safe_get_arg(i++, argc, argv);
            if (!val) { return 1; }
            if (!safe_atoui(val, &topk)) {
                fprintf(stderr, "Invalid number for --topk: %s\n", val);
                return 1;
            }
        } else if (strcmp(arg, "-s") == 0 || strcmp(arg, "--seed") == 0) {
            const char *val = safe_get_arg(i++, argc, argv);
            if (!val) { return 1; }
            if (!safe_atoui(val, &seed)) {
                fprintf(stderr, "Invalid number for --seed: %s\n", val);
                return 1;
            }
        } else if (strcmp(arg, "-t") == 0 || strcmp(arg, "--temperature") == 0) {
            const char *val = safe_get_arg(i++, argc, argv);
            if (!val) { return 1; }
            if (!safe_atof(val, &temperature) || temperature < 0.0f) {
                fprintf(stderr, "Invalid temperature: %s (must be >= 0)\n", val);
                return 1;
            }
        } else if (strcmp(arg, "-p") == 0 || strcmp(arg, "--topp") == 0) {
            const char *val = safe_get_arg(i++, argc, argv);
            if (!val) { return 1; }
            if (!safe_atof(val, &topp) || topp <= 0.0f || topp > 1.0f) {
                fprintf(stderr, "Invalid top-p: %s (must be 0 < p <= 1)\n", val);
                return 1;
            }
        } else if (strcmp(arg, "-r") == 0 || strcmp(arg, "--rpen") == 0) {
            const char *val = safe_get_arg(i++, argc, argv);
            if (!val) { return 1; }
            if (!safe_atof(val, &rpen) || rpen < 1.0f) {
                fprintf(stderr, "Invalid repetition penalty: %s (must be >= 1)\n", val);
                return 1;
            }
        } else if (strcmp(arg, "-i") == 0 || strcmp(arg, "--prompt") == 0) {
            const char *val = safe_get_arg(i++, argc, argv);
            if (!val) { return 1; }
            prompt = val;
        } else if (strcmp(arg, "-c") == 0 || strcmp(arg, "--chat") == 0) {
            chatmode = true;
        } else if (strcmp(arg, "-m") == 0 || strcmp(arg, "--enable-mm") == 0) {
            enable_mm = true;
        } else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            print_usage();
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", arg);
            print_usage();
            return 1;
        }
    }

    srand(seed);

    GemmaModel *model = read_model(modelfile, enable_mm);
    GemmaBuffer *buf = malloc_buffer(model->config, seqlen, enable_mm);
    // malloc buffer for SigLIP
    SigLIPBuffer *sbuf = NULL;
    if (enable_mm && model->config->support_mm) {
        sbuf = malloc_siglip_buffer(model->vision_enc, model->config->quant);
    }
    print_model_config(model);

    if (chatmode) {
        chat(model, buf, seqlen, temperature, topk, topp, rpen);
    } else {
        generate(model, buf, prompt, seqlen, temperature, topk, topp, rpen);
    }

    if (g_interrupted) {
        printf("\n\nInterrupted by user\n");
    }

    if (utf8_argv != NULL) { free_utf8_argv(utf8_argv, argc); }
    free_buffer(buf, model->config->quant);
    if (sbuf != NULL) { free(sbuf); }
    free_model(model);
    return 0;
}
