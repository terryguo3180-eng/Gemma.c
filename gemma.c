/* Gemma 1 & 2 & 3 & 3n implemented in a single file of pure C */

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

#define FP16_MAX (((union {_Float16 f; uint16_t b;}){.b = 0x7BFF}).f)

#ifdef DEBUG
#pragma message("Debug mode on")
#endif

// Global state for interruption handling
static volatile bool g_interrupted = false;

static void signal_handler(int signum) {
    (void)signum;
    g_interrupted = true;
}

void setup_signal_handler(void) {
    signal(SIGINT, signal_handler);
#ifdef SIGTERM
    signal(SIGTERM, signal_handler);
#endif
}

// Memory operation wrappers
// You can pass in basically arbitrary text into macros, including spaces. So inside
// void functions simply use `MALLOC(ptr, count, name, )`, and it will be automatically
// expanded to `... fprintf(...); return ; ...`
#define MALLOC(ptr, count, name, fail) do { \
    ptr = malloc((count) * sizeof(*(ptr)));  /* NOLINT */ \
    if (ptr == NULL) { \
        fprintf( \
            stderr, "Memory allocation failed: %s (size: %zu)\n", \
            (name), (count) * sizeof(*(ptr)));  /* NOLINT */\
        return fail; \
    } \
} while(0)

#define CALLOC(ptr, count, name, fail) do { \
    ptr = calloc((count), sizeof(*(ptr)));  /* NOLINT */ \
    if (ptr == NULL) { \
        fprintf( \
            stderr, "Memory allocation failed: %s (count: %zu, size: %zu)\n", \
            (name), (count), sizeof(*(ptr)));  /* NOLINT */ \
        return fail; \
    } \
} while(0)

// Statement expressions: ({ ... }), an extension of GNU C. Not supported in MSVC
#define FGETC(fp, name, fail) ({ \
    int __FGETC_read = fgetc(fp); \
    if (__FGETC_read == EOF) { \
        fprintf(stderr, "File read failed: %s", (name)); \
        return fail; \
    } \
    __FGETC_read; \
})

#define FREAD(ptr, count, fp, name, fail) do { \
    size_t __FREAD_read = fread((ptr), sizeof(*(ptr)), (count), (fp)); \
    if (__FREAD_read != (count)) { \
        fprintf( \
            stderr, "File read failed: %s (expected %d, got %zu)\n", \
            (name), (count), __FREAD_read); \
        return fail; \
    } \
} while(0)

#define READ_UINT16(fp, fail) ({ \
    /* Big-endian */ \
    unsigned char __READ_UINT16_bytes[2]; \
    FREAD(__READ_UINT16_bytes, 2, (fp), "uint16", fail); \
    ((int)__READ_UINT16_bytes[0] << 8) | (int)__READ_UINT16_bytes[1]; \
})

#define READ_UINT32(fp, fail) ({ \
    /* Big-endian */ \
    unsigned char __READ_UINT32_bytes[4]; \
    FREAD(__READ_UINT32_bytes, 4, (fp), "uint32", fail); \
    ((uint32_t)__READ_UINT32_bytes[0] << 24) | \
    ((uint32_t)__READ_UINT32_bytes[1] << 16) | \
    ((uint32_t)__READ_UINT32_bytes[2] <<  8) | \
    ((uint32_t)__READ_UINT32_bytes[3]); \
})

#define READ_FP32(fp, fail) ({ \
    uint32_t __READ_FP32_buf = READ_UINT32((fp), fail); \
    float __READ_FP32_f; \
    memcpy(&__READ_FP32_f, &__READ_FP32_buf, sizeof(float)); \
    __READ_FP32_f; \
})

#define READ_STR(fp, data, offset, name, fail) ({ \
    /* Read a pascal-style string, the first byte indicates the length */ \
    char __READ_STR_len_name[128]; \
    snprintf(__READ_STR_len_name, 128, "%s.length", (name)); \
    int __READ_STR_len = FGETC(fp, __READ_STR_len_name, fail); \
    char *__READ_STR_str = (data) + *(offset); \
    FREAD(__READ_STR_str, __READ_STR_len, (fp), (name), fail); \
    (data)[*(offset) + __READ_STR_len] = '\0'; \
    *(offset) += __READ_STR_len + 1;\
    __READ_STR_str; \
})

#define READ_TENSOR(ptr, count, fp, name, fail) do { \
    MALLOC((ptr), (count), (name), fail); \
    FREAD((ptr), (count), (fp), (name), fail); \
} while(0)

#define READ_LINEAR(w, fp, m, n, quant, name, fail) do { \
    MALLOC((w), 1, (name), fail); \
    if (!(quant)) { \
        (w)->dtype = DTYPE_FP16; \
        char __READ_LINEAR_fp16_name[128]; \
        snprintf(__READ_LINEAR_fp16_name, 128, "%s.fp16", (name)); \
        READ_TENSOR((w)->fp16, (m) * (n), (fp), __READ_LINEAR_fp16_name, fail); \
    } else { \
        (w)->dtype = DTYPE_INT8; \
        char __READ_LINEAR_i8q_name[128]; \
        char __READ_LINEAR_i8scales_name[128]; \
        snprintf(__READ_LINEAR_i8q_name, 128, "%s.i8.q", (name)); \
        snprintf(__READ_LINEAR_i8scales_name, 128, "%s.i8.scales", (name)); \
        READ_TENSOR((w)->i8.q, (m) * (n), (fp), __READ_LINEAR_i8q_name, fail); \
        READ_TENSOR((w)->i8.scales, (n), (fp), __READ_LINEAR_i8scales_name, fail); \
    } \
} while(0)

static inline int get_strarr_bytes(FILE *fp, int count) {
    // Read a sequence of pascal-stype strings
    int offset = 0;
    long pos = ftell(fp);
    if (pos == -1L) {
        perror("ftell failed");
        return -1;
    }
    // Get the total number of bytes
    for (int i = 0; i < count; i++) {
        int len = FGETC(fp, "<str.length>", -1);
        if (fseek(fp, len, SEEK_CUR) != 0) {
            perror("fseek failed");
            return -1;
        }
        offset += len + 1;
    }
    // Resume position
    if (fseek(fp, pos, SEEK_SET) != 0) {
        perror("fseek failed");
        return -1;
    }
    return offset;
}

typedef struct {
    int n_layers;
    int image_size;
    int patch_size;
    int hidden_dim;
    int n_heads;
    int mlp_dim;
    float eps;
} SigLIPConfig;

typedef struct {
    int n_layers;         // Number of transformer layers
    int n_heads;          // Number of attention heads
    int n_kv_heads;       // Number of key & value heads (Grouped Query Attention)
    int head_dim;         // Dimensions of the attention heads
    int embed_dim;        // Dimensions of the embedding vectors
    int mlp_dim;          // Dimensions of the MLP hidden layers
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

    bool support_mm;      // Whether the model supports multi-modal
    bool use_qk_norm;     // Whether query & key normalization is applied
    bool pre_ffwd_norm;   // Whether pre-feedforward normalization is applied
    bool post_ffwd_norm;  // Whether post-feedforward normalization is applied
    bool quant;           // Whether int8 quantization is enabled
} GemmaConfig;

static void free_config(GemmaConfig *conf) {
    if (conf == NULL) return;
    if (conf->attn_local_layers != NULL) { free(conf->attn_local_layers); }
    free(conf);
}

typedef struct { char *val; int idx; } Token;
typedef struct { char *str1; char *str2; int rank; } Merge;

// Functions for qsort / bsearch

static int cmp_token(const void *a, const void *b) {
    return strcmp(((Token *)a)->val, ((Token *)b)->val);
}

static int cmp_merge(const void *a, const void *b) {
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

static void free_tokenizer(GemmaTokenizer *tok) {
    if (tok == NULL) return;
    if (tok->vocab_data != NULL) { free(tok->vocab_data); }
    if (tok->merge_data != NULL) { free(tok->merge_data); }
    if (tok->vocab != NULL) { free(tok->vocab); }
    if (tok->vocab_sorted != NULL) { free(tok->vocab_sorted); }
    if (tok->ranks != NULL) { free(tok->ranks); }
    free(tok);
}

static int get_token_idx(GemmaTokenizer *tok, char *str) {
    // Get idx_to_vocab[string]
    Token key = { .val = str };
    Token *val = bsearch(
        &key, tok->vocab_sorted, tok->vocab_size, sizeof(tok->vocab_sorted[0]), cmp_token
    );
    if (val == NULL) { return -1; }
    return val->idx;
}

static Merge *get_merge_rec(GemmaTokenizer *tok, char *str1, char *str2) {
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

static void free_linear(Linear *l) {
    if (l == NULL) return;
    if (l->dtype == DTYPE_FP16 && l->fp16 != NULL) { free(l->fp16); }
    else {
        if (l->i8.q != NULL) { free(l->i8.q); }
        if (l->i8.scales != NULL) { free(l->i8.scales); }
    }
}

typedef struct {
    // Attention weights
    // 2D weights are transposed for higher CPU cache hits in GEMV
    Linear *wq;    // (embed_dim, n_heads * head_dim).T
    Linear *wk;    // (embed_dim, n_kv_heads * head_dim).T
    Linear *wv;    // (embed_dim, n_kv_heads * head_dim).T
    Linear *wo;    // (n_heads * head_dim, embed_dim).T
    // Feedforward weights
    Linear *w1;    // (embed_dim, mlp_dim).T
    Linear *w2;    // (embed_dim, mlp_dim).T
    Linear *w3;    // (mlp_dim, embed_dim).T
    // RMSNorm weights
    _Float16 *nq;  // (head_dim,)
    _Float16 *nk;  // (head_dim,)
    _Float16 *n1;  // (embed_dim,)
    _Float16 *n2;  // (embed_dim,)
    _Float16 *n3;  // (embed_dim,)
    _Float16 *n4;  // (embed_dim,)
} GemmaDecoderLayer;

static void free_gemma_layer(GemmaDecoderLayer *layer) {
    if (layer == NULL) return;
    free_linear(layer->wq);
    free_linear(layer->wk);
    free_linear(layer->wv);
    free_linear(layer->wo);
    free_linear(layer->w1);
    free_linear(layer->w2);
    free_linear(layer->w3);
    if (layer->n1 != NULL) { free(layer->n1); }
    if (layer->n2 != NULL) { free(layer->n2); }
    if (layer->nq != NULL) { free(layer->nq); }
    if (layer->nk != NULL) { free(layer->nk); }
    if (layer->n3 != NULL) { free(layer->n3); }
    if (layer->n4 != NULL) { free(layer->n4); }
    free(layer);
}

typedef struct {
    // ViT attention weights
    Linear *wq;      // (hidden_dim, hidden_dim).T
    Linear *wk;      // (hidden_dim, hidden_dim).T
    Linear *wv;      // (hidden_dim, hidden_dim).T
    Linear *wo;      // (hidden_dim, hidden_dim).T
    // ViT attention biases
    _Float16 *bq;    // (hidden_dim,)
    _Float16 *bk;    // (hidden_dim,)
    _Float16 *bv;    // (hidden_dim,)
    _Float16 *bo;    // (hidden_dim,)
    // ViT feedforward weights
    Linear *w1;      // (hidden_dim, mlp_dim).T
    Linear *w2;      // (hidden_dim, mlp_dim).T
    // ViT feedforward biases
    _Float16 *b1;    // (mlp_dim,)
    _Float16 *b2;    // (mlp_dim,)
    // ViT layernorm weights
    _Float16 *n1;    // (hidden_dim,)
    _Float16 *n2;    // (hidden_dim,)
    // ViT layernorm biases
    _Float16 *n1_b;  // (hidden_dim,)
    _Float16 *n2_b;  // (hidden_dim,)
} SigLIPEncoderLayer;

static void free_siglip_layer(SigLIPEncoderLayer *layer) {
    if (layer == NULL) return;
    free_linear(layer->wq);
    free_linear(layer->wk);
    free_linear(layer->wv);
    free_linear(layer->wo);
    if (layer->bq != NULL) { free(layer->bq); }
    if (layer->bk != NULL) { free(layer->bk); }
    if (layer->bv != NULL) { free(layer->bv); }
    if (layer->bo != NULL) { free(layer->bo); }
    free_linear(layer->w1);
    free_linear(layer->w2);
    if (layer->b1 != NULL) { free(layer->b1); }
    if (layer->b2 != NULL) { free(layer->b2); }
    if (layer->n1 != NULL) { free(layer->n1); }
    if (layer->n2 != NULL) { free(layer->n2); }
    if (layer->n1_b != NULL) { free(layer->n1_b); }
    if (layer->n2_b != NULL) { free(layer->n2_b); }
}

typedef struct {
    SigLIPConfig *config;
    _Float16 *patch_emb;    // (hidden_dim, 3, patch_size, patch_size)
    _Float16 *patch_emb_b;  // (hidden_dim,)
    Linear *pos_embedding;  // ((image_size / patch_size)^2, hidden_dim)
    SigLIPEncoderLayer **layers;
    _Float16 *post_norm;    // (hidden_dim,)
    _Float16 *post_norm_b;  // (hidden_dim,)
    _Float16 *norm;         // (hidden_dim,)
    Linear *proj;           // (hidden_dim, embed_dim).T
} SigLIPVisionEncoder;

void free_siglip(SigLIPVisionEncoder *enc) {
    if (enc == NULL) return;

    if (enc->patch_emb != NULL) { free(enc->patch_emb); }
    if (enc->patch_emb_b != NULL) { free(enc->patch_emb_b); }
    if (enc->pos_embedding != NULL) { free_linear(enc->pos_embedding); }
    if (enc->layers != NULL) {
        for (int i = 0; i < enc->config->n_layers; i++) {
            if (enc->layers[i] != NULL) { free_siglip_layer(enc->layers[i]); }
        }
    }
    if (enc->post_norm != NULL) { free(enc->post_norm); }
    if (enc->post_norm_b != NULL) { free(enc->post_norm_b); }
    if (enc->norm != NULL) { free(enc->norm); }
    if (enc->proj != NULL) { free_linear(enc->proj); } 
    free(enc);
}

typedef struct {
    GemmaConfig *config;
    GemmaTokenizer *tokenizer;
    SigLIPVisionEncoder *vision_enc;
    Linear *embedding;      // (vocab_size, embed_dim)
    GemmaDecoderLayer **layers;
    _Float16 *final_norm;  // (embed_dim,)
} GemmaModel;

void free_model(GemmaModel *model) {
    if (model == NULL) return;

    if (model->tokenizer != NULL) { free_tokenizer(model->tokenizer); }
    if (model->vision_enc != NULL) { free_siglip(model->vision_enc); }
    free_linear(model->embedding);
    if (model->layers != NULL) {
        for (int i = 0; i < model->config->n_layers; i++) {
            if (model->layers[i] != NULL) { free_gemma_layer(model->layers[i]); }
        }
    }
    if (model->final_norm != NULL) { free(model->final_norm); }
    if (model->config != NULL) { free_config(model->config); }
    if (model->layers != NULL) { free(model->layers); }
    free(model);
}

typedef struct {
    int8_t *x_i8;           // (n_patches, hidden_dim)
    _Float16 *x_scales;     // (n_patches,)
    int8_t *mlp_i8;         // (n_patches, mlp_dim)
    _Float16 *mlp_scales;   // (n_patches)

    _Float16 *x;            // (n_patches, hidden_dim)
    _Float16 *resid;        // (n_patches, hidden_dim)
    _Float16 *xq;           // (n_patches, hidden_dim)
    _Float16 *xk;           // (n_patches, hidden_dim)
    _Float16 *xv;           // (n_patches, hidden_dim)
    _Float16 *att_out;      // (n_patches, hidden_dim)
    _Float16 *mlp_hidden;   // (n_patches, mlp_dim)
    _Float16 *scores;
} SigLIPBuffer;

SigLIPBuffer *malloc_siglip_buffer(SigLIPConfig *vconf, bool quant) {
    SigLIPBuffer *buf;
    MALLOC(buf, 1, "SigLIPBuffer", NULL);

    int C = vconf->hidden_dim;
    int ppi = vconf->image_size / vconf->patch_size;
    int N = ppi * ppi;
    int CM = vconf->mlp_dim;
    int NH = vconf->n_heads;

    if (quant) {
        MALLOC(buf->x_i8, N, "SigLIP x_i8", NULL);
        MALLOC(buf->x_scales, N, "SigLIP x_scales", NULL);
    }

    MALLOC(buf->x, N * C, "SigLIP x", NULL);
    MALLOC(buf->resid, N * C, "SigLIP resid", NULL);
    MALLOC(buf->xq, N * C, "SigLIP q", NULL);
    MALLOC(buf->xk, N * C, "SigLIP k", NULL);
    MALLOC(buf->xv, N * C, "SigLIP v", NULL);
    MALLOC(buf->att_out, N * C, "SigLIP attn_out", NULL);
    MALLOC(buf->mlp_hidden, N * CM, "SigLIP mlp_hidden", NULL);
    MALLOC(buf->scores, NH * N * N, "SigLIP scores", NULL);

    return buf;
}

void free_siglip_buffer(SigLIPBuffer *buf, bool quant) {
    if (buf == NULL) return;
    if (quant) {
        if (buf->x_i8 != NULL) { free(buf->x_i8); }
        if (buf->x_scales != NULL) { free(buf->x_scales); }
    }
    if (buf->x != NULL) { free(buf->x); }
    if (buf->resid != NULL) { free(buf->resid); }
    if (buf->xq != NULL) { free(buf->xq); }
    if (buf->xk != NULL) { free(buf->xk); }
    if (buf->xv != NULL) { free(buf->xv); }
    if (buf->att_out != NULL) { free(buf->att_out); }
    if (buf->mlp_hidden != NULL) { free(buf->mlp_hidden); }
    if (buf->scores != NULL) { free(buf->scores); }
    free(buf);
}

typedef struct {
    int cache_len;

    int8_t *x_i8;            // ([n_patches], embed_dim,)
    int8_t *xo_i8;           // (n_heads, [n_patches], head_dim)
    int8_t *xg_i8;           // ([n_patches], mlp_dim,)

    _Float16 *x;             // ([n_patches], embed_dim,)
    _Float16 *resid;         // ([n_patches], embed_dim,)
    _Float16 *xq;            // (n_heads, [n_patches], head_dim)
    _Float16 *xk;            // (n_kv_heads, [n_patches], head_dim)
    _Float16 *csfreqs_slid;  // ([n_patches], head_dim / 2, 2)
    _Float16 *csfreqs_full;  // ([n_patches], head_dim / 2, 2)
    _Float16 *xv;            // (n_kv_heads, [n_patches], head_dim)
    _Float16 *xo;            // (n_heads, [n_patches], head_dim)
    _Float16 *att;           // (n_heads, [n_patches], cache_len)
    _Float16 *kv_cache;      // (n_layers, 2, cache_len, n_kv_heads, head_dim)
    _Float16 *xg;            // ([n_patches], mlp_dim,)
    _Float16 *xu;            // ([n_patches], mlp_dim,)
    _Float16 *logits;        // (vocab_size,)
} GemmaBuffer;

GemmaBuffer *malloc_buffer(GemmaConfig *conf, SigLIPConfig *vconf, int cache_len, bool enable_mm) {
    GemmaBuffer *buf;
    MALLOC(buf, 1, "GemmaBuffer", NULL);

    buf->cache_len = cache_len;

    int C = conf->embed_dim;
    int L = conf->n_layers;
    int CH = conf->head_dim;
    int NH = conf->n_heads;
    int CM = conf->mlp_dim;

    int q_size = NH * CH;
    int kv_size = conf->n_kv_heads * CH;

    if (conf->quant) {
        MALLOC(buf->x_i8, C, "buf.x_i8", NULL);
        MALLOC(buf->xo_i8, q_size, "buf.xo_i8", NULL);
        MALLOC(buf->xg_i8, conf->mlp_dim, "buf.xg_i8", NULL);
    }

    MALLOC(buf->kv_cache, L * 2 * cache_len * kv_size, "buf.kv_cache", NULL);
    MALLOC(buf->logits, conf->vocab_size, "buf.logits", NULL);

    int mult = 1;
    if (conf->support_mm && enable_mm && vconf != NULL) {
        int ppi = vconf->image_size / vconf->patch_size;
        mult = ppi * ppi;
    }

    MALLOC(buf->x, mult * C, "buf.x", NULL);
    MALLOC(buf->resid, mult * C, "buf.resid", NULL);
    MALLOC(buf->xq, mult * q_size, "buf.xq", NULL);
    MALLOC(buf->xk, mult * kv_size, "buf.xk", NULL);
    MALLOC(buf->csfreqs_slid, mult * CH, "buf.csfreqs_slid", NULL);
    MALLOC(buf->csfreqs_full, mult * CH, "buf.csfreqs_full", NULL);
    MALLOC(buf->xv, mult * kv_size, "buf.xv", NULL);
    MALLOC(buf->xo, mult * q_size, "buf.xo", NULL);
    MALLOC(buf->att, mult * NH * cache_len, "buf.att", NULL);
    MALLOC(buf->xg, mult * CM, "buf.xg", NULL);
    MALLOC(buf->xu, mult * CM, "buf.xu", NULL);

    return buf;
}

void free_buffer(GemmaBuffer *buf, bool quant) {
    if (buf->x != NULL) { free(buf->x); }
    if (buf->resid != NULL) { free(buf->resid); }
    if (buf->xq != NULL) { free(buf->xq); }
    if (buf->xk != NULL) { free(buf->xk); }
    if (buf->csfreqs_slid != NULL) { free(buf->csfreqs_slid); }
    if (buf->csfreqs_full != NULL) { free(buf->csfreqs_full); }
    if (buf->xv != NULL) { free(buf->xv); }
    if (buf->xo != NULL) { free(buf->xo); }
    if (buf->att != NULL) { free(buf->att); }
    if (buf->kv_cache != NULL) { free(buf->kv_cache); }
    if (buf->xg != NULL) { free(buf->xg); }
    if (buf->xu != NULL) { free(buf->xu); }
    if (buf->logits != NULL) { free(buf->logits); }
    if (quant) {
        if (buf->x_i8 != NULL) { free(buf->x_i8); }
        if (buf->xo_i8 != NULL) { free(buf->xo_i8); }
        if (buf->xg_i8 != NULL) { free(buf->xg_i8); }
    }
    free(buf);
}

GemmaModel *read_model(const char *filename, bool enable_mm) {
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) { perror(filename); return NULL; }

    GemmaTokenizer *tok;
    MALLOC(tok, 1, "GemmaTokenizer", NULL);
    GemmaConfig *conf;
    MALLOC(conf, 1, "GemmaConfig", NULL);

    // Read the configs
    conf->n_layers = FGETC(fp, "model.config.n_layers", NULL);
    conf->n_heads = FGETC(fp, "model.config.n_heads", NULL);
    conf->n_kv_heads = FGETC(fp, "model.config.n_kv_heads", NULL);
    conf->head_dim = READ_UINT16(fp, NULL);
    conf->embed_dim = READ_UINT16(fp, NULL);
    conf->mlp_dim = READ_UINT16(fp, NULL);
    conf->q_pre_attn_scalar = READ_UINT16(fp, NULL);
    conf->sliding_window = READ_UINT16(fp, NULL);
    conf->tokens_per_image = READ_UINT16(fp, NULL);
    conf->max_seq_len = (int)READ_UINT32(fp, NULL);
    conf->vocab_size = (int)READ_UINT32(fp, NULL);
    conf->local_theta = READ_FP32(fp, NULL);
    conf->global_theta = READ_FP32(fp, NULL);
    conf->eps = READ_FP32(fp, NULL);
    conf->attn_softcapping = READ_FP32(fp, NULL);
    conf->logit_softcapping = READ_FP32(fp, NULL);

    // attn_local_layers
    int n_bytes = FGETC(fp, "model.config.attn_local_layers", NULL);
    char *buf;
    MALLOC(buf, n_bytes, "model.config.attn_local_layers", NULL);
    MALLOC(conf->attn_local_layers, conf->n_layers, "model.config.attn_local_layers", NULL);
    FREAD(buf, n_bytes, fp, "model.config.attn_local_layers", NULL);
    for (int i = 0; i < conf->n_layers; i++) {
        int pos = i;
        int byte_idx = pos / 8;
        int bit_idx = 7 - (pos % 8);
        conf->attn_local_layers[i] = (buf[byte_idx] >> bit_idx) & 1;
    }
    free(buf);

    // Additional flags
    int extra_flags = FGETC(fp, "model.config.extra_flags", NULL);
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
        MALLOC(enc, 1, "model.vision_enc", NULL);
        MALLOC(vconf, 1, "model.vision_enc.config", NULL);
        vconf->n_layers = fgetc(fp);
        if (vconf->n_layers == EOF) { return NULL; }
        vconf->n_heads = fgetc(fp);
        if (vconf->n_heads == EOF) { return NULL; }
        vconf->mlp_dim = READ_UINT16(fp, NULL);
        vconf->hidden_dim = READ_UINT16(fp, NULL);
        vconf->image_size = READ_UINT16(fp, NULL);
        vconf->patch_size = READ_UINT16(fp, NULL);
        vconf->eps = READ_FP32(fp, NULL);
        enc->config = vconf;
    } else if (conf->support_mm) {
        for (int j = 0; j < 2; j++) { if (fgetc(fp) == EOF) { return NULL; }; }
        for (int j = 0; j < 4; j++) { READ_UINT16(fp, NULL); }
        READ_FP32(fp, NULL);
    }

    // dtype (only supports float16)
    int offset = 0;
    char dtype[10];
    READ_STR(fp, dtype, &offset, "dtype", NULL);
    if (strcmp(dtype, "float16") != 0) {
        printf("dtype '%s' not supported\n", dtype);
        return NULL;
    }

    // Build vocabulary
    offset = 0;
    tok->vocab_size = conf->vocab_size;
    if (conf->support_mm) { tok->vocab_size++; }  // ++ for the <image_soft_token>
    MALLOC(tok->vocab_data, get_strarr_bytes(fp, tok->vocab_size), "model.tokenizer.vocab_data", NULL);
    MALLOC(tok->vocab, tok->vocab_size, "model.tokenizer.vocab", NULL);
    MALLOC(tok->vocab_sorted, tok->vocab_size, "model.tokenizer.vocab_sorted", NULL);
    for (int i = 0; i < tok->vocab_size; i++) {
        char name[64];
        snprintf(name, sizeof(name), "model.tokenizer.vocab_data.%d", i);
        char *str = READ_STR(fp, tok->vocab_data, &offset, name, NULL);
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
    tok->n_merges = (int)READ_UINT32(fp, NULL);
    MALLOC(tok->ranks, tok->n_merges, "model.tokenizer.ranks", NULL);
    MALLOC(tok->merge_data, get_strarr_bytes(fp, tok->n_merges * 2), "model.tokenizer.merge_data", NULL);

    offset = 0;
    for (int i = 0; i < tok->n_merges; i++) {
        char name0[64], name1[64];
        snprintf(name0, sizeof(name0), "model.tokenizer.merge_data.%d.0", i);
        snprintf(name1, sizeof(name1), "model.tokenizer.merge_data.%d.1", i);
        char *str1 = READ_STR(fp, tok->merge_data, &offset, name0, NULL);
        char *str2 = READ_STR(fp, tok->merge_data, &offset, name1, NULL);
        tok->ranks[i].rank = i;
        tok->ranks[i].str1 = str1;
        tok->ranks[i].str2 = str2;
    }
    qsort(tok->ranks, tok->n_merges, sizeof(tok->ranks[0]), cmp_merge);

    // Build the model
    GemmaModel *model;
    MALLOC(model, 1, "model", NULL);
    model->vision_enc = enc;

    // Read the weights
    model->config = conf;
    model->tokenizer = tok;

    // The shape is actually (vocab_size, embed_dim), but it uses per-tensor quantization
    // rather than per-channel like other weights. Gemma uses tied weights, which means
    // the final lm_head shares the same weights with the embedding table, but transposed.
    // So it becomes per-channel quantization in the final lm_head
    int C = conf->embed_dim;

    READ_LINEAR(model->embedding, fp, C, conf->vocab_size, conf->quant, "model.embedding", NULL);
    MALLOC(model->layers, conf->n_layers, "model.layers", NULL);

    int q_size = conf->n_heads * conf->head_dim;
    int kv_size = conf->n_kv_heads * conf->head_dim;

    // Read all the layers
    for (int l = 0; l < conf->n_layers; l++) {
        char layer_name[64];
        snprintf(layer_name, sizeof(layer_name), "model.layers.%d", l);
        GemmaDecoderLayer *layer;
        MALLOC(layer, 1, layer_name, NULL);

        char wq_name[64], wk_name[64], wv_name[64], wo_name[64];
        snprintf(wq_name, sizeof(wq_name), "model.layers.%d.wq", l);
        snprintf(wk_name, sizeof(wk_name), "model.layers.%d.wk", l);
        snprintf(wv_name, sizeof(wv_name), "model.layers.%d.wv", l);
        snprintf(wo_name, sizeof(wo_name), "model.layers.%d.wo", l);

        // Attention weights
        READ_LINEAR(layer->wq, fp, C, q_size, conf->quant, wq_name, NULL);
        READ_LINEAR(layer->wk, fp, C, kv_size, conf->quant, wk_name, NULL);
        READ_LINEAR(layer->wv, fp, C, kv_size, conf->quant, wv_name, NULL);
        READ_LINEAR(layer->wo, fp, q_size, C, conf->quant, wo_name, NULL);

        if (conf->use_qk_norm) {
            char nq_name[64], nk_name[64];
            snprintf(nq_name, sizeof(nq_name), "model.layers.%d.nq", l);
            snprintf(nk_name, sizeof(nk_name), "model.layers.%d.nk", l);
            READ_TENSOR(layer->nq, conf->head_dim, fp, nq_name, NULL);
            READ_TENSOR(layer->nk, conf->head_dim, fp, nk_name, NULL);
        } else {
            layer->nq = NULL;
            layer->nk = NULL;
        }

        char w1_name[64], w2_name[64], w3_name[64];
        snprintf(w1_name, sizeof(w1_name), "model.layers.%d.w1", l);
        snprintf(w2_name, sizeof(w2_name), "model.layers.%d.w2", l);
        snprintf(w3_name, sizeof(w3_name), "model.layers.%d.w3", l);

        // Feedforward weights
        READ_LINEAR(layer->w1, fp, C, conf->mlp_dim, conf->quant, w1_name, NULL);
        READ_LINEAR(layer->w2, fp, C, conf->mlp_dim, conf->quant, w2_name, NULL);
        READ_LINEAR(layer->w3, fp, conf->mlp_dim, C, conf->quant, w3_name, NULL);

        char n1_name[64], n2_name[64];
        snprintf(n1_name, sizeof(n1_name), "model.layers.%d.n1", l);
        snprintf(n2_name, sizeof(n2_name), "model.layers.%d.n2", l);

        // RMSNorm weights
        READ_TENSOR(layer->n1, C, fp, n1_name, NULL);
        READ_TENSOR(layer->n2, C, fp, n2_name, NULL);

        if (conf->pre_ffwd_norm) {
            char n3_name[64];
            snprintf(n3_name, sizeof(n3_name), "model.layers.%d.n3", l);
            READ_TENSOR(layer->n3, C, fp, n3_name, NULL);
        } else {
            layer->n3 = NULL;
        }
        if (conf->post_ffwd_norm) {
            char n4_name[64];
            snprintf(n4_name, sizeof(n4_name), "model.layers.%d.n4", l);
            READ_TENSOR(layer->n4, C, fp, n4_name, NULL);
        } else {
            layer->n4 = NULL;
        }
        model->layers[l] = layer;
    }
    READ_TENSOR(model->final_norm, C, fp, "model.final_norm", NULL);

    if (use_mm) {
        int P = vconf->patch_size;
        int VC = vconf->hidden_dim;

        READ_TENSOR(enc->patch_emb, VC * 3 * P * P, fp, "model.vision_enc.patch_emb", NULL);
        READ_TENSOR(enc->patch_emb_b, VC, fp, "model.vision_enc.patch_emb_b", NULL);
        int n_patches = vconf->image_size / P;
        n_patches *= n_patches;
        // Same as here, the real shape is (n_patches, VC)
        READ_LINEAR(enc->pos_embedding, fp, VC, n_patches, conf->quant, "model.vision_enc.pos_embedding", NULL);
        MALLOC(enc->layers, vconf->n_layers, "model.vision_enc.layers", NULL);

        // Read all the layers of ViT
        for (int l = 0; l < vconf->n_layers; l++) {
            char layer_name[64];
            snprintf(layer_name, sizeof(layer_name), "model.vision_enc.layers.%d", l);
            SigLIPEncoderLayer *layer;
            MALLOC(layer, 1, layer_name, NULL);
            
            // First layernorm
            char n1_name[64], n1b_name[64];
            snprintf(n1_name, sizeof(n1_name), "model.vision_enc.layers.%d.n1", l);
            snprintf(n1b_name, sizeof(n1b_name), "model.vision_enc.layers.%d.n1_b", l);
            READ_TENSOR(layer->n1, VC, fp, n1_name, NULL);
            READ_TENSOR(layer->n1_b, VC, fp, n1b_name, NULL);

            // Attention weights
            char wq_name[64], wk_name[64], wv_name[64], wo_name[64];
            snprintf(wq_name, sizeof(wq_name), "model.vision_enc.layers.%d.wq", l);
            snprintf(wk_name, sizeof(wk_name), "model.vision_enc.layers.%d.wk", l);
            snprintf(wv_name, sizeof(wv_name), "model.vision_enc.layers.%d.wv", l);
            snprintf(wo_name, sizeof(wo_name), "model.vision_enc.layers.%d.wo", l);
            READ_LINEAR(layer->wq, fp, VC, VC, conf->quant, wq_name, NULL);
            READ_LINEAR(layer->wk, fp, VC, VC, conf->quant, wk_name, NULL);
            READ_LINEAR(layer->wv, fp, VC, VC, conf->quant, wv_name, NULL);
            READ_LINEAR(layer->wo, fp, VC, VC, conf->quant, wo_name, NULL);
            
            // Attention biases
            char bq_name[64], bk_name[64], bv_name[64], bo_name[64];
            snprintf(bq_name, sizeof(bq_name), "model.vision_enc.layers.%d.bq", l);
            snprintf(bk_name, sizeof(bk_name), "model.vision_enc.layers.%d.bk", l);
            snprintf(bv_name, sizeof(bv_name), "model.vision_enc.layers.%d.bv", l);
            snprintf(bo_name, sizeof(bo_name), "model.vision_enc.layers.%d.bo", l);
            READ_TENSOR(layer->bq, VC, fp, bq_name, NULL);
            READ_TENSOR(layer->bk, VC, fp, bk_name, NULL);
            READ_TENSOR(layer->bv, VC, fp, bv_name, NULL);
            READ_TENSOR(layer->bo, VC, fp, bo_name, NULL);

            // Second layernorm
            char n2_name[64], n2b_name[64];
            snprintf(n2_name, sizeof(n2_name), "model.vision_enc.layers.%d.n2", l);
            snprintf(n2b_name, sizeof(n2b_name), "model.vision_enc.layers.%d.n2_b", l);
            READ_TENSOR(layer->n2, VC, fp, n2_name, NULL);
            READ_TENSOR(layer->n2_b, VC, fp, n2b_name, NULL);

            // Feedforward weights
            char w1_name[64], w2_name[64];
            snprintf(w1_name, sizeof(w1_name), "model.vision_enc.layers.%d.w1", l);
            snprintf(w2_name, sizeof(w2_name), "model.vision_enc.layers.%d.w2", l);
            READ_LINEAR(layer->w1, fp, VC, vconf->mlp_dim, conf->quant, w1_name, NULL);
            READ_LINEAR(layer->w2, fp, vconf->mlp_dim, VC, conf->quant, w2_name, NULL);
            
            // Feedforward biases
            char b1_name[64], b2_name[64];
            snprintf(b1_name, sizeof(b1_name), "model.vision_enc.layers.%d.b1", l);
            snprintf(b2_name, sizeof(b2_name), "model.vision_enc.layers.%d.b2", l);
            READ_TENSOR(layer->b1, vconf->mlp_dim, fp, b1_name, NULL);
            READ_TENSOR(layer->b2, VC, fp, b2_name, NULL);
            
            enc->layers[l] = layer;
        }

        // Post layernorm
        READ_TENSOR(enc->post_norm, VC, fp, "model.vision_enc.post_norm", NULL);
        READ_TENSOR(enc->post_norm_b, VC, fp, "model.vision_enc.post_norm_b", NULL);

        // Soft embedding RMSNorm
        READ_TENSOR(enc->norm, VC, fp, "model.vision_enc.norm", NULL);
        // Final projection
        READ_LINEAR(enc->proj, fp, VC, C, conf->quant, "model.vision_enc.proj", NULL);
    }

    fclose(fp);
    return model;
}

// Make sure the clamping is not optimized by compilers
#ifdef __GNUC__
__attribute__((optimize("no-fast-math")))
#endif
#ifdef __clang__
#pragma float_control(precise, on, push)
#endif
static inline _Float16 clamp_fp16(_Float16 v) {
    return fminf(FP16_MAX, fmaxf(-FP16_MAX, v));
}
#ifdef __clang__
#pragma float_control(pop)
#endif

static void rmsnorm(_Float16 *dst, _Float16 *src, _Float16 *weight, int dim, float eps) {
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

static void rmsnorm_omp(_Float16 *dst, _Float16 *src, _Float16 *weight, int dim, float eps) {
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
static void layernorm(_Float16 *dst, _Float16 *src, _Float16 *weight, _Float16 *bias, int dim, float eps) {
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

static _Float16 quantize_act(int8_t *dst, _Float16 *vec, int dim) {
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

static void quantize_act_rows(
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

static void gemv_fp16(
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

static void gemv_int8(
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

static void gemm_fp16(_Float16 *dst, const Linear *mat, _Float16 *src, int m, int n, int k) {
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

static void gemm_int8(
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

static _Float16 dot(_Float16 *v1, _Float16 *v2, int dim) {
    // Must use float instead of _Float16
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum += (float)v1[i] * (float)v2[i];
    }
    return (_Float16)sum;
}

static void softmax(_Float16 *dst, _Float16 *src, int dim) {
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

static void softmax_omp(_Float16 *dst, _Float16 *src, int dim) {
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

// TODO: Integrate this into the language model (tons and tons of work to do...)
void forward_siglip(
    SigLIPVisionEncoder *enc, GemmaConfig *conf, SigLIPBuffer *buf,
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
    int mlp_dim = vconf->mlp_dim;

    // Patch Embedding
    int in_dim = 3 * patch * patch;

    // NOTE: The consecutive "naked" for-loops (no braces) stacked right on top of
    // each other are intentional here, I like to write it in this way. Think of
    // them as one combined loop nest, like nested sigma notation. Only the innermost
    // statement gets braces
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
            // Clangd love to remind me about my unconventional style, have to put `// NOLINT`
            // everywhere to shut it up
            buf->x[token_idx * C + oc] = clamp_fp16((_Float16)(sum + (float)enc->patch_emb_b[oc]));  // NOLINT
        }
    }

    warn_stats("patch_emb", buf->x, N * C, 0, 0, 0);  // NOLINT

    // Position Embedding
    for (int i = 0; i < N; i++) {
        _Float16 *pos_vec = enc->pos_embedding->fp16 + i * C;
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
            gemm_fp16(buf->xq, layer->wq, buf->x, N, C, C);
            gemm_fp16(buf->xk, layer->wk, buf->x, N, C, C);
            gemm_fp16(buf->xv, layer->wv, buf->x, N, C, C);
        } else {
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
        for (int j = 0; j < C; j++) {
            int idx = i * C + j;
            buf->xq[idx] += layer->bq[j];
            buf->xk[idx] += layer->bk[j];
            buf->xv[idx] += layer->bv[j];
        }

        // Attention
        memset(buf->att_out, 0, N * C * sizeof(_Float16));  // NOLINT
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
            gemm_fp16(buf->x, layer->wo, buf->att_out, N, C, C);
        } else {
            quantize_act_rows(buf->x_i8, buf->att_out, N, C, buf->x_scales);
            gemm_int8(buf->x, layer->wo, buf->x_i8, N, C, C, buf->x_scales);
        }
        // Add output bias
        for (int i = 0; i < N; i++)
        for (int j = 0; j < C; j++) {
            buf->x[i * C + j] += layer->bo[j];
        }
        warn_stats("att_out", buf->x, N * C, l, 0, 8);  // NOLINT

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
            gemm_fp16(buf->mlp_hidden, layer->w1, buf->x, N, mlp_dim, C);
        } else {
            quantize_act_rows(buf->x_i8, buf->x, N, C, buf->x_scales);
            gemm_int8(buf->mlp_hidden, layer->w1, buf->x_i8, N, mlp_dim, C, buf->x_scales);
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
        warn_stats("mlp_hidden", buf->mlp_hidden, N * mlp_dim, l, 0, 8);  // NOLINT

        // mlp_hidden @ fc2 = x
        if (!conf->quant) {
            gemm_fp16(buf->x, layer->w2, buf->mlp_hidden, N, C, mlp_dim);
        } else {
            quantize_act_rows(buf->mlp_i8, buf->mlp_hidden, N, mlp_dim, buf->mlp_scales);
            gemm_int8(buf->x, layer->w2, buf->mlp_i8, N, C, mlp_dim, buf->mlp_scales);
        }
        // x += b2
        for (int i = 0; i < N; i++)
        for (int j = 0; j < C; j++) {
            buf->x[i * C + j] += layer->b2[j];
        }
        warn_stats("mlp_out", buf->x, N * C, l, 0, 8);  // NOLINT
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
            buf->x[out_idx + d] = (_Float16)(sum / (kernel_size * kernel_size));  // NOLINT
        }
    }
    warn_stats("avg_pool", buf->x, tokens_per_image * C, 0, 0, 0);  // NOLINT

    // RMSNorm
    rmsnorm(buf->x, buf->x, enc->norm, C, vconf->eps);
    warn_stats("rmsnorm", buf->x, tokens_per_image * C, 0, 0, 0);

    // Final projection
    gemm_fp16(out, enc->proj, buf->x, tokens_per_image, conf->embed_dim, C);
    warn_stats("proj", out, tokens_per_image * conf->embed_dim, 0, 0, 0);
}

int forward_gemma(GemmaModel *model, GemmaBuffer *buf, int tok, int pos) {
    GemmaConfig *conf = model->config;
    
    int C = conf->embed_dim;
    
    if (pos >= buf->cache_len) {
        fprintf(stderr, "\nError: KV Cache is full");
        return 1;
    }

    // I can't directly scale the embedding weight beforehand (see export.py),
    // so I have to scale it here at run time. The activation scaler that I used
    // in export.py was 1/sqrt(embed_dim), and the original Gemma training uses
    // sqrt(embed_dim) as embed_scale, they canceled out to 1.0f

    _Float16 embed_scale = 1.0f;  // equivalent to sqrt(embed_dim) * (1/sqrt(embed_dim))
    if (conf->quant) {
        // Dequantize
        embed_scale *= model->embedding->i8.scales[tok];
    }

    // x = embedding[tok] * embed_scale
    #pragma omp parallel for
    for (int i = 0; i < C; i++) {
        if (!conf->quant) {
            buf->x[i] = model->embedding->fp16[tok*C + i] * embed_scale;
        } else {
            buf->x[i] = model->embedding->i8.q[tok*C + i] * embed_scale;
        }
    }

    warn_stats("embedding", buf->x, C, 0, pos, 0);
    
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

        memcpy(buf->resid, buf->x, C*sizeof(_Float16));

        rmsnorm_omp(buf->x, buf->x, layer->n1, C, conf->eps);
        warn_stats("norm1", buf->x, C, l, pos, 8);

        // The attention block
        if (!conf->quant) {
            gemv_fp16(buf->xq, layer->wq, buf->x, q_size, C);  // (n_heads, head_dim)
            gemv_fp16(buf->xk, layer->wk, buf->x, kv_size, C);  // (n_kv_heads, head_dim)
            gemv_fp16(buf->xv, layer->wv, buf->x, kv_size, C);  // (n_kv_heads, head_dim)
        } else {
            float xscale = quantize_act(buf->x_i8, buf->x, C);
            gemv_int8(buf->xq, layer->wq, buf->x_i8, q_size, C, xscale);
            gemv_int8(buf->xk, layer->wk, buf->x_i8, kv_size, C, xscale);
            gemv_int8(buf->xv, layer->wv, buf->x_i8, kv_size, C, xscale);
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
            gemv_fp16(buf->x, layer->wo, buf->xo, C, q_size);
        } else {
            float xoscale = quantize_act(buf->xo_i8, buf->xo, q_size);
            gemv_int8(buf->x, layer->wo, buf->xo_i8, C, q_size, xoscale);
        }
        warn_stats("attn_out", buf->x, C, l, pos, 8);

        rmsnorm_omp(buf->x, buf->x, layer->n2, C, conf->eps);
        warn_stats("norm2", buf->x, C, l, pos, 8);

        // Combine the residual stream
        _Float16 *restrict x = buf->x;
        _Float16 *restrict resid = buf->resid;
        #pragma omp simd
        for (int d = 0; d < C; d++) {
            // Sometimes the residual stream accumulates huge values on certain channels,
            // especially in pretrained/bigger models (Sun et al. https://arxiv.org/abs/2402.17762)
            // It works fine in fp32 or bf16, but it can easily overflow fp16 and become
            // inf, causing all the activations turning into nan after the next RMSNorm,
            // so we need to clamp it

            // NOTE: Actually this should never trigger now since I added activation scalers
            // afterwards (see export.py), the clamp here is more of a last-resort safety net
            buf->x[d] = clamp_fp16(x[d] + resid[d]);
        }
        warn_stats("resid1", buf->x, C, l, pos, 8);

        memcpy(buf->resid, buf->x, C * sizeof(_Float16));

        if (conf->pre_ffwd_norm) {
            rmsnorm_omp(buf->x, buf->x, layer->n3, C, conf->eps);
        }

        // MLP feedforward layer
        if (!conf->quant) {
            gemv_fp16(buf->xg, layer->w2, buf->x, conf->mlp_dim, C);
            gemv_fp16(buf->xu, layer->w1, buf->x, conf->mlp_dim, C);
        } else {
            float xscale = quantize_act(buf->x_i8, buf->x, C);
            gemv_int8(buf->xg, layer->w2, buf->x_i8, conf->mlp_dim, C, xscale);
            gemv_int8(buf->xu, layer->w1, buf->x_i8, conf->mlp_dim, C, xscale);
        }

        // GELU layer
        #pragma omp parallel for
        for (int d = 0; d < conf->mlp_dim; d++) {
            // Tanh approximation of GELU
            float x = (float)buf->xg[d];
            float c = 0.79788456080287f;  // sqrt(2 / pi)
            x = 0.5*x * (1 + tanhf(c * (x + 0.044715 * x*x*x)));
            buf->xg[d] = (_Float16)x;
            // Fuse xg * xu into xg
            buf->xg[d] *= buf->xu[d];
        }
        
        if (!conf->quant) {
            gemv_fp16(buf->x, layer->w3, buf->xg, C, conf->mlp_dim);
        } else {
            float xscale = quantize_act(buf->xg_i8, buf->xg, conf->mlp_dim);
            gemv_int8(buf->x, layer->w3, buf->xg_i8, C, conf->mlp_dim, xscale);
        }
        warn_stats("down_proj", buf->x, C, l, pos, 8);

        if (conf->post_ffwd_norm) {
            rmsnorm_omp(buf->x, buf->x, layer->n4, C, conf->eps);
            warn_stats("norm4", buf->x, C, l, pos, 8);
        }

        // Combine the residual stream
        x = buf->x;
        resid = buf->resid;
        #pragma omp simd
        for (int d = 0; d < C; d++) {
            buf->x[d] = clamp_fp16(x[d] + resid[d]);
        }
        warn_stats("resid2", buf->x, C, l, pos, 8);
    }

    // Final RMSNorm
    rmsnorm_omp(buf->x, buf->x, model->final_norm, C, conf->eps);
    return 0;
}

static int argmax(_Float16 *logits, int vocab_size) {
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

static int partition_desc(FloatIdx *arr, int lo, int hi) {
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

static void quickselect_topk(FloatIdx *arr, int lo, int hi, int k_idx) {
    while (lo < hi) {
        int p = partition_desc(arr, lo, hi);
        if (p == k_idx) return;
        else if (p < k_idx) { lo = p + 1; }
        else { hi = p - 1; }
    }
}

static void apply_topk(_Float16 *logits, FloatIdx *logit_indices, int vocab_size, int k) {
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

static void sift_down(FloatIdx *arr, int n, int i) {
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

static void build_heap(FloatIdx *arr, int n) {
    for (int i = n/2 - 1; i >= 0; i--) { sift_down(arr, n, i); }
}

static void apply_topp(
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
    int *(*token_callback)(int, GemmaTokenizer *, bool *)
) {
    GemmaConfig *conf = model->config;
    int vs = conf->vocab_size;
    bool quit = false;

    // Boolean flags
    bool dosample = temperature != 0 && topk != 1;
    bool use_topk = dosample && topk != 0;
    bool use_topp = dosample && topp < 1.0f;
    bool use_rpen = dosample && rpen > 1.0f;

    // bool array indicating which tokens have already been processed
    // Used in rpen (repetition penalty)
    bool *visited = NULL;
    if (use_rpen) {
        bool *visited;
        // Will expand to "... return ; ...". No return values
        CALLOC(visited, vs, "visited", );
    }
    int pos, token;

    // Prefill all the prompt tokens except the last one
    clock_t prefill_start = clock();
    pos = 0;
    for (int *t = tokens; *t != EOF && *(t+1) != EOF; t++) {
        if (g_interrupted) {
            if (use_rpen) { free(visited); }
            return;
        }
        if (use_rpen) { visited[*t] = true; }
        if (forward_gemma(model, buf, *t, pos++) == 1) { goto end; }  // forward in the current pos
    }
    clock_t prefill_end = clock();
    double prefill_elapsed = (double)(prefill_end - prefill_start) / CLOCKS_PER_SEC;
    int prefill_tokens = pos;

    token = tokens[pos];
    _Float16 *probs = NULL;

    // Only allocate probs if needed
    if (temperature != 0.0f) { MALLOC(probs, vs, "probs", ); }

    // Logit indices for topk & topp
    FloatIdx *logit_indices = NULL;
    if (use_topk || use_topp) { MALLOC(logit_indices, vs, "logit_indices", ); }

    // Record tok/s
    clock_t gen_start = clock();
    int gen_tokens = 0;

    for (; pos < seqlen; pos++) {
        if (g_interrupted) break;

        if (use_rpen) visited[token] = true;
        if (forward_gemma(model, buf, token, pos) == 1) { goto end; }

        // Compute logits
        if (!conf->quant) {
            gemv_fp16(buf->logits, model->embedding, buf->x, vs, conf->embed_dim);
        } else {
            float xscale = quantize_act(buf->x_i8, buf->x, conf->embed_dim);
            gemv_int8(buf->logits, model->embedding, buf->x_i8, vs, conf->embed_dim, xscale);
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
        int *ret = token_callback(token, model->tokenizer, &quit);

        if (quit) break;
        else if (ret != NULL) {
            // Injected a token array (ends with EOF)
            // Prefill all the tokens except the last one
            int i;
            for (i = 0; (token = ret[i]) != EOF && ret[i+1] != EOF; i++) {
                if (g_interrupted) break;
                if (use_rpen) { visited[token] = true; }
                if (forward_gemma(model, buf, token, ++pos) == 1) { goto end; }  // Forward in the next pos
            }
            if (g_interrupted) break;
            token = ret[i];  // The last element
        }
        // ret == NULL: do nothing
    }

end:
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
                for (int j = i + 1; j < tok_i - 1; j++) {
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

static int *generate_callback(int token, GemmaTokenizer *tok, bool *quit) {
    if (token == tok->eos || token == tok->eot) { *quit = true; }
    char byte_buf[2];
    printf("%s", decode(tok, token, byte_buf));
    fflush(stdout);
    return NULL;
}

int generate(
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
    tokens[n_tokens] = EOF;

    sample(model, buf, tokens, seqlen, temperature, topk, topp, rpen, generate_callback);

    return 0;
}

static int *new_turn(GemmaTokenizer *tok, bool bos, bool *quit) {
    if (bos) { printf("User: "); }
    else { printf("\nUser: "); }

    char user_prompt[65536];
    if (fgets(user_prompt, sizeof(user_prompt), stdin) == NULL) {
        fprintf(stderr, "Failed to read user input\n");
        *quit = true;
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
    int *tokens;
    MALLOC(tokens, size + 1, "tokens", (*quit = true, NULL));

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

static int *chat_callback(int token, GemmaTokenizer *tok, bool *quit) {
    if (token == tok->eos || token == tok->eot) { return new_turn(tok, false, quit); }
    char byte_buf[2];
    printf("%s", decode(tok, token, byte_buf));
    fflush(stdout);
    return NULL;
}

int chat(
    GemmaModel *model, GemmaBuffer *buf,
    int seqlen, float temperature, int topk, float topp, float rpen
) {
    bool quit = false;
    int *first_turn = new_turn(model->tokenizer, true, &quit);
    if (quit) { return 1; }
    sample(model, buf, first_turn, seqlen, temperature, topk, topp, rpen, chat_callback);
    return 0;
}

static inline bool safe_atoui(const char *str, unsigned int *result) {
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

static inline bool safe_atof(const char *str, float *result) {
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
    if (argv == NULL) return;
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

// Auto-generated by gen_print.py
void print_model_config(GemmaModel *model, int seqlen, bool enable_mm) {
    (void)model;
    (void)seqlen;
    (void)enable_mm;
#ifdef DEBUG
    GemmaConfig *conf = model->config;
    GemmaTokenizer *tok = model->tokenizer;

    printf("\n========== Model Configuration ==========\n");
    printf("Architecture:\n");

    printf("  %-*s: %d\n", 19, "n_layers", conf->n_layers);
    printf("  %-*s: %d\n", 19, "n_heads", conf->n_heads);
    printf("  %-*s: %d\n", 19, "n_kv_heads", conf->n_kv_heads);
    printf("  %-*s: %d\n", 19, "head_dim", conf->head_dim);
    printf("  %-*s: %d\n", 19, "embed_dim", conf->embed_dim);
    printf("  %-*s: %d\n", 19, "mlp_dim", conf->mlp_dim);
    printf("  %-*s: %d\n", 19, "q_pre_attn_scalar", conf->q_pre_attn_scalar);
    printf("  %-*s: %d\n", 19, "sliding_window", conf->sliding_window);
    printf("  %-*s: %d\n", 19, "tokens_per_image", conf->tokens_per_image);
    printf("  %-*s: %d\n", 19, "max_seq_len", conf->max_seq_len);
    printf("  %-*s: %d\n", 19, "vocab_size", conf->vocab_size);
    printf("  %-*s: %.6f\n", 19, "local_theta", conf->local_theta);
    printf("  %-*s: %.6f\n", 19, "global_theta", conf->global_theta);
    printf("  %-*s: %.6f\n", 19, "eps", conf->eps);
    printf("  %-*s: %.6f\n", 19, "attn_softcapping", conf->attn_softcapping);
    printf("  %-*s: %.6f\n", 19, "logit_softcapping", conf->logit_softcapping);
    printf("  %-*s: ", 19, "attn_local_layers");
    for (int i = 0; i < conf->n_layers; i++) {
        printf("%d", conf->attn_local_layers[i] ? 1 : 0);
        if ((i + 1) % 18 == 0 && i + 1 < conf->n_layers) { printf("\n                       "); }
    }
    printf("\n");
    printf("  %-*s: %s\n", 19, "support_mm", conf->support_mm ? "true" : "false");
    printf("  %-*s: %s\n", 19, "use_qk_norm", conf->use_qk_norm ? "true" : "false");
    printf("  %-*s: %s\n", 19, "pre_ffwd_norm", conf->pre_ffwd_norm ? "true" : "false");
    printf("  %-*s: %s\n", 19, "post_ffwd_norm", conf->post_ffwd_norm ? "true" : "false");
    printf("  %-*s: %s\n", 19, "quant", conf->quant ? "true" : "false");

    printf("\nTokenizer:\n");

    printf("  %-*s: %d\n", 19, "vocab_size", tok->vocab_size);
    printf("  %-*s: %d\n", 19, "n_merges", tok->n_merges);
    printf("  %-*s: %d\n", 19, "bos", tok->bos);
    printf("  %-*s: %d\n", 19, "eos", tok->eos);
    printf("  %-*s: %d\n", 19, "sot", tok->sot);
    printf("  %-*s: %d\n", 19, "eot", tok->eot);
    printf("  %-*s: %d\n", 19, "soi", tok->soi);
    printf("  %-*s: %d\n", 19, "eoi", tok->eoi);
    printf("  %-*s: %d\n", 19, "ist", tok->ist);

    printf("\nMemory Footprint (estimated):\n");

    size_t total_bytes = 0;

    // Embedding
    if (!(conf->quant)) {
        total_bytes += conf->embed_dim * conf->vocab_size * sizeof(_Float16);
    } else {
        total_bytes += conf->embed_dim * conf->vocab_size * sizeof(int8_t);
        total_bytes += conf->vocab_size * sizeof(_Float16);  // scales
    }

    // Each layer
    for (int l = 0; l < conf->n_layers; l++) {
        int q_size  = conf->n_heads * conf->head_dim;
        int kv_size = conf->n_kv_heads * conf->head_dim;

        int layer_params = conf->embed_dim * q_size
            + conf->embed_dim * kv_size
            + conf->embed_dim * kv_size
            + q_size * conf->embed_dim
            + conf->embed_dim * conf->mlp_dim
            + conf->embed_dim * conf->mlp_dim
            + conf->mlp_dim * conf->embed_dim;

        if (!conf->quant) {
            total_bytes += layer_params * sizeof(_Float16);
        } else {
            total_bytes += layer_params * sizeof(int8_t);
            total_bytes += (
                q_size
                + kv_size
                + kv_size
                + conf->embed_dim
                + conf->mlp_dim
                + conf->mlp_dim
                + conf->embed_dim
            ) * sizeof(_Float16);  // scales
        }

        total_bytes += conf->embed_dim * sizeof(_Float16);  // n1
        total_bytes += conf->embed_dim * sizeof(_Float16);  // n2
        if (conf->use_qk_norm ? 2 * conf->head_dim : 0) {
            total_bytes += conf->use_qk_norm ? 2 * conf->head_dim : 0 * sizeof(_Float16);  // qk_norm
        }
        if (conf->pre_ffwd_norm ? conf->embed_dim : 0) {
            total_bytes += conf->pre_ffwd_norm ? conf->embed_dim : 0 * sizeof(_Float16);  // pre_ffwd_norm
        }
        if (conf->post_ffwd_norm ? conf->embed_dim : 0) {
            total_bytes += conf->post_ffwd_norm ? conf->embed_dim : 0 * sizeof(_Float16);  // post_ffwd_norm
        }
    }

    // Final norm
    total_bytes += conf->embed_dim * sizeof(_Float16);

    printf("  %-*s: %.2f GB\n", 19, "Weights", total_bytes / (1024.0 * 1024.0 * 1024.0));

    // KV Cache per token
    int kv_size = conf->n_kv_heads * conf->head_dim;
    size_t kv_cache_bytes = conf->n_layers * 2 * kv_size * sizeof(_Float16);
    printf("  %-*s: %.2f KB\n", 19, "KV Cache (tok)", kv_cache_bytes / 1024.0);

    // GemmaBuffer

    int C = conf->embed_dim;
    int L = conf->n_layers;
    int CH = conf->head_dim;
    int NH = conf->n_heads;
    int CM = conf->mlp_dim;
    int ppi = model->vision_enc->config->image_size / model->vision_enc->config->patch_size;
    int q_size = NH * CH;
    int mult = conf->support_mm && enable_mm && model->vision_enc != NULL? ppi * ppi : 1;

    size_t buf_bytes = 0;

    if (conf->quant) {
        buf_bytes += C * sizeof(int8_t);  // x_i8
    }
    if (conf->quant) {
        buf_bytes += q_size * sizeof(int8_t);  // xo_i8
    }
    if (conf->quant) {
        buf_bytes += CM * sizeof(int8_t);  // xg_i8
    }
    buf_bytes += L * 2 * seqlen * kv_size * sizeof(_Float16);  // kv_cache
    buf_bytes += conf->vocab_size * sizeof(_Float16);  // logits
    buf_bytes += mult * C * sizeof(_Float16);  // x
    buf_bytes += mult * C * sizeof(_Float16);  // resid
    buf_bytes += mult * q_size * sizeof(_Float16);  // xq
    buf_bytes += mult * kv_size * sizeof(_Float16);  // xk
    buf_bytes += mult * CH * sizeof(_Float16);  // csfreqs_slid
    buf_bytes += mult * CH * sizeof(_Float16);  // csfreqs_full
    buf_bytes += mult * kv_size * sizeof(_Float16);  // xv
    buf_bytes += mult * q_size * sizeof(_Float16);  // xo
    buf_bytes += mult * NH * seqlen * sizeof(_Float16);  // att
    buf_bytes += mult * CM * sizeof(_Float16);  // xg
    buf_bytes += mult * CM * sizeof(_Float16);  // xu

    printf("  %-*s: %.2f MB\n", 19, "Gemma Buffer", buf_bytes / (1024.0 * 1024.0));

    if (enable_mm && model->vision_enc != NULL) {
        SigLIPConfig *vconf = model->vision_enc->config;

        int C = vconf->hidden_dim;
        int mlp_dim = vconf->mlp_dim;
        int ppi = vconf->image_size / vconf->patch_size;
        int n_patches = ppi * ppi;
        int n_heads = vconf->n_heads;

        size_t siglip_bytes = 0;

        if (conf->quant) {
            siglip_bytes += n_patches * C * sizeof(int8_t);  // x_i8
        }
        if (conf->quant) {
            siglip_bytes += n_patches * sizeof(_Float16);  // x_scales
        }
        if (conf->quant) {
            siglip_bytes += n_patches * mlp_dim * sizeof(int8_t);  // mlp_i8
        }
        if (conf->quant) {
            siglip_bytes += n_patches * sizeof(_Float16);  // mlp_scales
        }
        siglip_bytes += n_patches * C * sizeof(_Float16);  // x
        siglip_bytes += n_patches * C * sizeof(_Float16);  // resid
        siglip_bytes += n_patches * C * sizeof(_Float16);  // xq
        siglip_bytes += n_patches * C * sizeof(_Float16);  // xk
        siglip_bytes += n_patches * C * sizeof(_Float16);  // xv
        siglip_bytes += n_patches * C * sizeof(_Float16);  // att_out
        siglip_bytes += n_patches * mlp_dim * sizeof(_Float16);  // mlp_hidden
        siglip_bytes += n_heads * n_patches * n_patches * sizeof(_Float16);  // scores

        printf("  %-*s: %.2f MB\n", 19, "SigLIP Buffer", siglip_bytes / (1024.0 * 1024.0));
    }

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
    if (model == NULL) { goto end; }

    GemmaConfig *conf = model->config;
    SigLIPConfig *vconf = NULL;
    if (model->vision_enc != NULL) { vconf = model->vision_enc->config; }

    GemmaBuffer *buf = malloc_buffer(conf, vconf, seqlen, enable_mm);
    if (buf == NULL) { goto end; }

    // malloc buffer for SigLIP
    SigLIPBuffer *sbuf = NULL;
    if (enable_mm && conf->support_mm) {
        sbuf = malloc_siglip_buffer(vconf, conf->quant);
        if (sbuf == NULL) { goto end; }
    }
    print_model_config(model, seqlen, enable_mm);

    if (chatmode) { chat(model, buf, seqlen, temperature, topk, topp, rpen); }
    else { generate(model, buf, prompt, seqlen, temperature, topk, topp, rpen); }

    if (g_interrupted) { printf("\n\nInterrupted by user\n"); }

end:
    free_utf8_argv(utf8_argv, argc);
    free_buffer(buf, conf->quant);
    free_siglip_buffer(sbuf, conf->quant);
    free_model(model);
    return 0;
}
