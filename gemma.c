#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <signal.h>

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
#include <termios.h>
#include <unistd.h>
// No problem with POSIX though
static void set_utf8_console() {}
static char** get_utf8_argv(int *argc_out) { (void)argc_out; return NULL; }
static void free_utf8_argv(char **argv, int argc) { (void)argv; (void)argc; }
#endif


typedef struct {
    int n_layers;         // Number of transformer layers
    int n_heads;          // Number of attention heads
    int n_kv_heads;       // Number of key & value heads (Grouped Query Attention)
    int head_dim;         // Dimensions of the attention heads
    int embed_dim;        // Dimensions of the embedding vectors
    int mlp_hidden_size;  // Dimensions of the MLP hidden layers
    // The scale applied to query vectors before computing attention scores
    int q_pre_attn_scalar;
    int sliding_window;   // Context size in sliding window attention
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
    bool use_qk_norm;     // Whether query & key normalization is applied
    bool pre_ffwd_norm;   // Whether pre-feedforward normalization is applied
    bool post_ffwd_norm;  // Whether post-feedforward normalization is applied
} GemmaConfig;

typedef struct { char *val; int idx; } Token;
typedef struct { char *str1; char *str2; int rank; } Merge;

// Functions for qsort / bsearch

int cmp_token(const void *a, const void *b) {
    return strcmp(((Token *)a)->val, ((Token *)b)->val);
}

int cmp_merge(const void *a, const void *b) {
    int ret = strcmp(((Merge *)a)->str1, ((Merge *)b)->str1);
    if (ret != 0)
        return ret;
    return strcmp(((Merge *)a)->str2, ((Merge *)b)->str2);
}

typedef struct {
    int n_merges;
    int vocab_size;
    int bos;
    int eos;
    int sot;
    int eot;
    char *vocab_data;
    char *merge_data;
    char **vocab;
    Token *vocab_sorted;
    Merge *ranks;
} GemmaTokenizer;

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

typedef struct {
    // Attention weights
    // 2D weights are transposed for higher CPU cache hits in GEMV
    _Float16 *wq;  // (embed_dim, n_heads * head_dim).T
    _Float16 *wk;  // (embed_dim, n_kv_heads * head_dim).T
    _Float16 *wv;  // (embed_dim, n_kv_heads * head_dim).T
    _Float16 *wo;  // (n_heads * head_dim, embed_dim).T
    // Feedforward weights
    _Float16 *w1;  // (embed_dim, mlp_hidden_size).T
    _Float16 *w2;  // (embed_dim, mlp_hidden_size).T
    _Float16 *w3;  // (mlp_hidden_size, embed_dim).T
    // RMSNorm weights
    _Float16 *nq;  // (head_dim,)
    _Float16 *nk;  // (head_dim,)
    _Float16 *n1;  // (embed_dim,)
    _Float16 *n2;  // (embed_dim,)
    _Float16 *n3;  // (embed_dim,)
    _Float16 *n4;  // (embed_dim,)
} GemmaDecoderLayer;

typedef struct {
    GemmaConfig *config;
    GemmaTokenizer *tokenizer;
    _Float16 *embedding;   // (vocab_size, embed_dim)
    GemmaDecoderLayer **layers;
    _Float16 *final_norm;  // (embed_dim,)
} GemmaModel;

typedef struct {
    int cache_len;

    _Float16 *x;             // (embed_dim,)
    _Float16 *resid;         // (embed_dim,)
    _Float16 *xq;            // (n_heads, head_dim)
    _Float16 *xk;            // (n_kv_heads, head_dim)
    _Float16 *xq_buf;        // (n_heads, head_dim)
    _Float16 *xk_buf;        // (n_kv_heads, head_dim)
    _Float16 *csfreqs_slid;  // (head_dim / 2, 2)
    _Float16 *csfreqs_full;  // (head_dim / 2, 2)
    _Float16 *xv;            // (n_kv_heads, head_dim)
    _Float16 *xo;            // (n_heads, head_dim)
    _Float16 *att;           // (n_heads, cache_len)
    _Float16 *kv_cache;      // (n_layers, 2, cache_len, n_kv_heads, head_dim)
    _Float16 *xg;            // (mlp_hidden_size,)
    _Float16 *xu;            // (mlp_hidden_size,)
    _Float16 *logits;        // (vocab_size,)
} ModelBuffer;


#define FP16_MAX      (((union {_Float16 f; uint16_t b;}){.b = 0x7BFF}).f)
#define EOT_SENTINEL  (-1)  // Sentinel for end of token array
#define SAMPLE_ABORT  ((int *)(intptr_t)-1)

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

// Forward declarations
char *decode(GemmaTokenizer *tok, int id);

// Safe memory operation wrappers
void *safe_malloc(size_t size, const char *context) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(
            stderr, "Memory allocation failed: %s (size: %zu bytes)\n",
            context, size
        );
        exit(1);
    }
    return ptr;
}

void *safe_calloc(size_t count, size_t size, const char *context) {
    void *ptr = calloc(count, size);
    if (ptr == NULL) {
        fprintf(
            stderr, "Memory allocation failed: %s (count: %zu, size: %zu)\n",
            context, count, size
        );
        exit(1);
    }
    return ptr;
}

void safe_fread(void *ptr, size_t size, size_t count, FILE *fp, const char *context) {
    size_t read = fread(ptr, size, count, fp);
    if (read != count) {
        fprintf(
            stderr, "File read failed: %s (expected %zu, got %zu)\n",
            context, count, read
        );
        exit(1);
    }
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
        fprintf(stderr, "File read failed: reading string length\n");
        exit(1);
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
        perror("ftell failed");
        exit(1);
    }
    // Get the total number of bytes
    for (int i = 0; i < count; i++) {
        int len = fgetc(fp);
        if (len == EOF) {
            fprintf(stderr, "File read failed: reading string length in array\n");
            exit(1);
        }
        if (fseek(fp, len, SEEK_CUR) != 0) {
            perror("fseek failed");
            exit(1);
        }
        offset += len + 1;
    }
    // Resume position
    if (fseek(fp, pos, SEEK_SET) != 0) {
        perror("fseek failed");
        exit(1);
    }
    return offset;
}

_Float16 *read_tensor(FILE *fp, int size) {
    _Float16 *tensor = safe_malloc(size * sizeof(_Float16), "tensor");
    safe_fread(tensor, sizeof(_Float16), size, fp, "tensor data");
    return tensor;
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
            if (str[i] >= 32 && str[i] < 127) {
                result[offset++] = str[i];
            } else {
                result[offset++] = '\\';
                result[offset++] = 'x';
                result[offset++] = "0123456789abcdef"[(unsigned char)str[i] >> 4];
                result[offset++] = "0123456789abcdef"[(unsigned char)str[i] & 15];
            }
        }
    }
    result[offset++] = '"';
    result[offset++] = '\0';
    return result;
}

GemmaModel *read_model(const char *filename) {
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
    conf->mlp_hidden_size = read_uint16(fp);
    conf->q_pre_attn_scalar = read_uint16(fp);
    conf->sliding_window = read_uint16(fp);
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
        fprintf(stderr, "File read failed: reading attn_local_layers size\n");
        exit(1);
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
        fprintf(stderr, "File read failed: reading extra flags\n");
        exit(1);
    }
    conf->use_qk_norm = (extra_flags & 8) == 8;
    conf->pre_ffwd_norm = (extra_flags & 4) == 4;
    conf->post_ffwd_norm = (extra_flags & 2) == 2;
    if (extra_flags & 1) {
        printf("int8 quantization not supported\n");
        exit(1);
    }

    // dtype (only supports float16)
    int offset = 0;
    char dtype[10];
    read_str(fp, dtype, &offset);
    if (strcmp(dtype, "float16") != 0) {
        printf("dtype %s not supported\n", repr(dtype));
        exit(1);
    }

    // Build vocabulary
    offset = 0;
    tok->vocab_size = conf->vocab_size;
    tok->vocab_data = safe_malloc(get_strarr_bytes(fp, conf->vocab_size), "vocab_data");
    tok->vocab = safe_malloc(conf->vocab_size * sizeof(*tok->vocab), "vocab");
    tok->vocab_sorted = safe_malloc(conf->vocab_size * sizeof(*tok->vocab_sorted), "vocab_sorted");
    for (int i = 0; i < conf->vocab_size; i++) {
        char *str = read_str(fp, tok->vocab_data, &offset);
        tok->vocab[i] = str;
        tok->vocab_sorted[i].idx = i;
        tok->vocab_sorted[i].val = str;
    }
    qsort(tok->vocab_sorted, conf->vocab_size, sizeof(tok->vocab_sorted[0]), cmp_token);
    // Special tokens
    tok->bos = get_token_idx(tok, "<bos>");
    tok->eos = get_token_idx(tok, "<eos>");
    tok->sot = get_token_idx(tok, "<start_of_turn>");
    tok->eot = get_token_idx(tok, "<end_of_turn>");

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
    model->config = conf;
    model->tokenizer = tok;

    // Read the weights
    model->embedding = read_tensor(fp, conf->vocab_size * conf->embed_dim);
    model->layers = safe_malloc(conf->n_layers * sizeof(*model->layers), "model layers");

    int q_size = conf->n_heads * conf->head_dim;
    int kv_size = conf->n_kv_heads * conf->head_dim;

    // Read all the layers
    for (int l = 0; l < conf->n_layers; l++) {
        GemmaDecoderLayer *layer = safe_malloc(sizeof(*layer), "GemmaDecoderLayer");

        // Attention weights
        layer->wq = read_tensor(fp, conf->embed_dim * q_size);
        layer->wk = read_tensor(fp, conf->embed_dim * kv_size);
        layer->wv = read_tensor(fp, conf->embed_dim * kv_size);
        layer->wo = read_tensor(fp, q_size * conf->embed_dim);

        if (conf->use_qk_norm) {
            layer->nq = read_tensor(fp, conf->head_dim);
            layer->nk = read_tensor(fp, conf->head_dim);
        }

        // Feedforward weights
        layer->w1 = read_tensor(fp, conf->embed_dim * conf->mlp_hidden_size);
        layer->w2 = read_tensor(fp, conf->embed_dim * conf->mlp_hidden_size);
        layer->w3 = read_tensor(fp, conf->mlp_hidden_size * conf->embed_dim);

        // RMSNorm weights
        layer->n1 = read_tensor(fp, conf->embed_dim);
        layer->n2 = read_tensor(fp, conf->embed_dim);

        if (conf->pre_ffwd_norm)
            layer->n3 = read_tensor(fp, conf->embed_dim);
        if (conf->post_ffwd_norm)
            layer->n4 = read_tensor(fp, conf->embed_dim);

        model->layers[l] = layer;
    }
    model->final_norm = read_tensor(fp, conf->embed_dim);

    fclose(fp);
    return model;
}

void free_model(GemmaModel *model) {
    GemmaConfig *conf = model->config;
    GemmaTokenizer *tok = model->tokenizer;

    free(conf->attn_local_layers);
    free(tok->vocab_data);
    free(tok->vocab);
    free(tok->merge_data);
    free(tok->ranks);
    free(model->embedding);

    for (int i = 0; i < conf->n_layers; i++) {
        GemmaDecoderLayer *layer = model->layers[i];
        free(layer->wq);
        free(layer->wk);
        free(layer->wv);
        free(layer->wo);
        if (conf->use_qk_norm) {
            free(layer->nq);
            free(layer->nk);
        }
        free(layer->w1);
        free(layer->w2);
        free(layer->w3);
        free(layer->n1);
        free(layer->n2);
        if (conf->pre_ffwd_norm)
            free(layer->n3);
        if (conf->post_ffwd_norm)
            free(layer->n4);
        free(layer);
    }
    free(model->final_norm);

    free(tok);
    free(conf);
    free(model->layers);
    free(model);
}

ModelBuffer *malloc_buffer(GemmaConfig *conf, int cache_len) {
    ModelBuffer *buf = safe_malloc(sizeof(*buf), "ModelBuffer");
    buf->cache_len = cache_len;
    buf->x = safe_malloc(conf->embed_dim * sizeof(_Float16), "buffer x");
    buf->resid = safe_malloc(conf->embed_dim * sizeof(_Float16), "buffer resid");

    int q_size = conf->n_heads * conf->head_dim;
    int kv_size = conf->n_kv_heads * conf->head_dim;

    buf->xq = safe_malloc(q_size * sizeof(_Float16), "buffer xq");
    buf->xk = safe_malloc(kv_size * sizeof(_Float16), "buffer xk");
    buf->xq_buf = safe_malloc(q_size * sizeof(_Float16), "buffer xq_buf");
    buf->xk_buf = safe_malloc(kv_size * sizeof(_Float16), "buffer xk_buf");
    buf->csfreqs_slid = safe_malloc(conf->head_dim * sizeof(_Float16), "buffer csfreqs_slid");
    buf->csfreqs_full = safe_malloc(conf->head_dim * sizeof(_Float16), "buffer csfreqs_full");
    buf->xv = safe_malloc(kv_size * sizeof(_Float16), "buffer xv");
    buf->xo = safe_malloc(q_size * sizeof(_Float16), "buffer xo");
    buf->att = safe_malloc(conf->n_heads * cache_len * sizeof(_Float16), "buffer att");
    buf->kv_cache = safe_malloc(
        conf->n_layers * 2 * cache_len * kv_size * sizeof(_Float16),
        "buffer kv_cache"
    );
    buf->xg = safe_malloc(conf->mlp_hidden_size * sizeof(_Float16), "buffer xg");
    buf->xu = safe_malloc(conf->mlp_hidden_size * sizeof(_Float16), "buffer xu");
    buf->logits = safe_malloc(conf->vocab_size * sizeof(_Float16), "buffer logits");

    return buf;
}

void free_buffer(ModelBuffer *buf) {
    free(buf->x);
    free(buf->resid);
    free(buf->xq);
    free(buf->xk);
    free(buf->xq_buf);
    free(buf->xk_buf);
    free(buf->csfreqs_slid);
    free(buf->csfreqs_full);
    free(buf->xv);
    free(buf->xo);
    free(buf->att);
    free(buf->kv_cache);
    free(buf->xg);
    free(buf->xu);
    free(buf->logits);
    free(buf);
}

void rmsnorm(_Float16 *dst, _Float16 *src, _Float16 *weight, int dim, float eps) {
    float sqsum = 0.0f;
    #pragma omp parallel for reduction(+:sqsum)
    for (int i = 0; i < dim; i++) {
        sqsum += (float)src[i] * (float)src[i];
    }
    float rms = 1.0f / sqrtf(sqsum / dim + eps);
    #pragma omp parallel for
    for (int i = 0; i < dim; i++) {
        // Gemma uses (weight + 1) instead of (weight)
        dst[i] = (_Float16)((float)src[i] * rms * (weight[i] + 1));
    }
}

void gemv(
    _Float16 *restrict dst,
    _Float16 *restrict mat,
    _Float16 *restrict vec,
    int m, int n
) {
    // mat (m, n) @ vec (n,) = dst (m,)
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        float sum = 0;
        for (int j = 0; j < n; j++)
            sum += (float)mat[i * n + j] * (float)vec[j];
        dst[i] = (_Float16)sum;
    }
}

_Float16 dot(_Float16 *v1, _Float16 *v2, int dim) {
    _Float16 sum = 0.0f;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < dim; i++)
        sum += v1[i] * v2[i];
    return sum;
}

void softmax(_Float16 *dst, _Float16 *src, int dim) {
    _Float16 max = -1e10f;
    for (int i = 0; i < dim; i++)
        if (src[i] > max) max = src[i];

    float expsum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float val = expf((float)(src[i] - max));
        expsum += val;
        dst[i] = (_Float16)val;
    }

    for (int i = 0; i < dim; i++)
        dst[i] = (_Float16)((float)dst[i] / expsum);
}

__attribute__((optimize("no-fast-math")))  // Not sure if this works for other compilers
static inline float clamp_fp16(float v) {
    return fminf(FP16_MAX, fmaxf(-FP16_MAX, v));
}

void forward(GemmaModel *model, ModelBuffer *buf, int tok, int pos) {
    GemmaConfig *conf = model->config;

    // x = embedding[tok] * embed_dim**0.5
    _Float16 embed_scale = (_Float16)sqrtf((float)conf->embed_dim);
    for (int i = 0; i < conf->embed_dim; i++)
        buf->x[i] = model->embedding[tok*conf->embed_dim + i] * embed_scale;
    
    int q_size = conf->n_heads * conf->head_dim;
    int kv_size = conf->n_kv_heads * conf->head_dim;
    int hd_half = conf->head_dim / 2;

    // Precompute cos & sin for all frequencies (used in RoPE)
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
        rmsnorm(buf->x, buf->x, layer->n1, conf->embed_dim, conf->eps);

        // The attention block
        gemv(buf->xq, layer->wq, buf->x, q_size, conf->embed_dim);  // (n_heads, head_dim)
        gemv(buf->xk, layer->wk, buf->x, kv_size, conf->embed_dim);  // (n_kv_heads, head_dim)
        gemv(buf->xv, layer->wv, buf->x, kv_size, conf->embed_dim);  // (n_kv_heads, head_dim)

        if (conf->use_qk_norm) {
            // Query RMSNorm
            for (int h = 0; h < conf->n_heads; h++) {
                _Float16 *xq_head = buf->xq + h*conf->head_dim;
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
        
        // Apply RoPE to queries & keys
        #pragma omp parallel for
        for (int d = 0; d < conf->n_heads * conf->head_dim; d++) {
            int a = d % hd_half;  // Index in the current half vector
            int b = a + hd_half;  // Index in the current half vector
            int i = d % conf->head_dim;  // Index in the current vector
            int h = d / conf->head_dim;  // Current head
            int o = h * conf->head_dim;  // Offset for current head
            
            float cfr = freqs_cs[a*2];
            float sfr = freqs_cs[a*2 + 1];
            
            // Apply to queries
            if (i < hd_half)
                buf->xq_buf[o+i] = (_Float16)(buf->xq[o+a]*cfr - buf->xq[o+b]*sfr);
            else
                buf->xq_buf[o+i] = (_Float16)(buf->xq[o+a]*sfr + buf->xq[o+b]*cfr);

            if (h >= conf->n_kv_heads) continue;
            
            // Apply to keys
            if (i < hd_half)
                buf->xk_buf[o+i] = (_Float16)(buf->xk[o+a]*cfr - buf->xk[o+b]*sfr);
            else
                buf->xk_buf[o+i] = (_Float16)(buf->xk[o+a]*sfr + buf->xk[o+b]*cfr);
        }

        memcpy(buf->xq, buf->xq_buf, conf->n_heads * conf->head_dim * sizeof(_Float16));
        memcpy(buf->xk, buf->xk_buf, conf->n_kv_heads * conf->head_dim * sizeof(_Float16));

        if (pos >= buf->cache_len) {
            printf("\nKV Cache is full, exiting...");
            free_buffer(buf);
            free_model(model);
            exit(1);
        }
        int entry_size = conf->n_kv_heads * conf->head_dim;
        int layer_size = 2*buf->cache_len * entry_size;

        // (cache_len, n_kv_heads, head_dim)
        _Float16 *k_cache = buf->kv_cache + l*layer_size;
        _Float16 *v_cache = k_cache + buf->cache_len * entry_size;

        // Write to key & value cache
        memcpy(k_cache + pos*entry_size, buf->xk, entry_size*sizeof(_Float16));
        memcpy(v_cache + pos*entry_size, buf->xv, entry_size*sizeof(_Float16));

        // Iterate over all the attention heads
        #pragma omp parallel for
        for (int h = 0; h < conf->n_heads; h++) {
            int h_kv = h * conf->n_kv_heads / conf->n_heads;
            _Float16 *xq_head = buf->xq + h*conf->head_dim;  // xq[h, :]
            _Float16 *xk_head = k_cache + h_kv*conf->head_dim;  // k_cache[:, h_kv, :]
            _Float16 *att_head = buf->att + h*buf->cache_len;  // att[h, :]

            // Compute dot product of the current query across all the keys
            for (int t = 0; t <= pos; t++)
                att_head[t] = dot(xq_head, xk_head + t*entry_size, conf->head_dim) * att_scale;
    
            // Attention score softcapping
            if (conf->attn_softcapping != 0.0f)
                for (int t = 0; t <= pos; t++) {
                    float val = (float)att_head[t] / conf->attn_softcapping;
                    att_head[t] = (_Float16)(tanhf(val) * conf->attn_softcapping);
                }

            // Sliding window attention, discard attention scores at the beginning
            if (is_local && pos >= conf->sliding_window)
                for (int i = 0; i <= pos - conf->sliding_window; i++)
                    att_head[i] = -(_Float16)INFINITY;
            
            // Softmax
            softmax(att_head, att_head, pos + 1);

            // Compute output
            _Float16 *xv_head = v_cache + h_kv*conf->head_dim;  // v_cache[:, h_kv, :]
            _Float16 *xo_head = buf->xo + h*conf->head_dim;
            for (int d = 0; d < conf->head_dim; d++) {
                float sum = 0.0f;
                for (int t = 0; t <= pos; t++)
                    sum += (xv_head + t*entry_size)[d] * att_head[t];
                xo_head[d] = sum;
            }
        }

        gemv(buf->x, layer->wo, buf->xo, conf->embed_dim, q_size);
        rmsnorm(buf->x, buf->x, layer->n2, conf->embed_dim, conf->eps);

        // Combine the residual stream
        for (int d = 0; d < conf->embed_dim; d++) {
            buf->x[d] += buf->resid[d];
            // Sometimes the residual stream accumulates huge values on certain channels,
            // especially in pretrained models (Sun et al. https://arxiv.org/abs/2402.17762)
            // It works fine in fp32 or bf16, but it can easily overflow fp16
            // So we need to clamp it
            buf->x[d] = clamp_fp16(buf->x[d]);
        }

        memcpy(buf->resid, buf->x, conf->embed_dim * sizeof(_Float16));

        if (conf->pre_ffwd_norm)
            rmsnorm(buf->x, buf->x, layer->n3, conf->embed_dim, conf->eps);

        // MLP feedforward layer
        gemv(buf->xg, layer->w2, buf->x, conf->mlp_hidden_size, conf->embed_dim);
        gemv(buf->xu, layer->w1, buf->x, conf->mlp_hidden_size, conf->embed_dim);

        // GELU layer using tanh approximation
        #pragma omp parallel for
        for (int d = 0; d < conf->mlp_hidden_size; d++) {
            float x = (float)buf->xg[d];
            float c = 0.79788456080287f;  // sqrt(2 / pi)
            x = 0.5*x * (1 + tanhf(c * (x + 0.044715 * x*x*x)));
            buf->xg[d] = (_Float16)x;
        }
        // Fuse xg * xu into xg
        for (int d = 0; d < conf->mlp_hidden_size; d++)
            buf->xg[d] *= buf->xu[d];

        gemv(buf->x, layer->w3, buf->xg, conf->embed_dim, conf->mlp_hidden_size);

        if (conf->post_ffwd_norm)
            rmsnorm(buf->x, buf->x, layer->n4, conf->embed_dim, conf->eps);

        // Combine the residual stream
        for (int d = 0; d < conf->embed_dim; d++) {
            buf->x[d] += buf->resid[d];
            buf->x[d] = clamp_fp16(buf->x[d]);
        }
    }

    // Final RMSNorm
    rmsnorm(buf->x, buf->x, model->final_norm, conf->embed_dim, conf->eps);

    // Logit softcapping
    for (int d = 0; d < conf->vocab_size; d++) {
        float val = (float)buf->logits[d] / conf->logit_softcapping;
        buf->logits[d] = (_Float16)(tanhf(val) * conf->logit_softcapping);
    }
}

int argmax(_Float16 *logits, int vocab_size) {
    // Pick the index with the max value
    int max_idx = -1;
    _Float16 max_val = -(_Float16)INFINITY;
    for (int i = 0; i < vocab_size; i++)
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    return max_idx;
}

typedef struct { _Float16 val; int idx; } FloatIdx;

int cmp_floatidx(const void *a, const void *b) {
    FloatIdx *pa = (FloatIdx *)a;
    FloatIdx *pb = (FloatIdx *)b;
    if (pb->val > pa->val) return 1;
    else if (pb->val < pa->val) return -1;
    return 0;
}

void apply_topk(_Float16 *logits, FloatIdx *fis, int vocab_size, int k) {
    if (k <= 0) k = 1;
    if (k > vocab_size) k = vocab_size;

    // Record index info
    #pragma omp parallel for
    for (int i = 0; i < vocab_size; i++) {
        fis[i].val = logits[i];
        fis[i].idx = i;
    }
    qsort(fis, vocab_size, sizeof(FloatIdx), cmp_floatidx);

    // Keep the top k channels, set the rest to -inf
    #pragma omp parallel for
    for (int i = 0; i < vocab_size; i++)
        logits[i] = -(_Float16)INFINITY;

    for (int i = 0; i < k; i++) {
        int orig_i = fis[i].idx;
        logits[orig_i] = fis[i].val;
    }
}

void apply_topp(
    _Float16 *logits, _Float16 *fpbuf, FloatIdx *fis,
    int vocab_size, float p
) {
    // Softmax to get the probs, store in fpbuf
    softmax(fpbuf, logits, vocab_size);

    // Record index info
    #pragma omp parallel for
    for (int i = 0; i < vocab_size; i++) {
        fis[i].val = fpbuf[i];
        fis[i].idx = i;
    }
    qsort(fis, vocab_size, sizeof(FloatIdx), cmp_floatidx);

    // Find the threshold where cumulative probability exceeds p
    float cum = 0.0f;
    int cutoff = 0;
    for (; cutoff < vocab_size; cutoff++) {
        cum += (float)fis[cutoff].val;
        if (cum >= p) { cutoff++; break; }
    }
    if (cutoff <= 0) cutoff = 1;

    // fpbuf is now a copy of the original logits
    memcpy(fpbuf, logits, vocab_size * sizeof(_Float16));

    #pragma omp parallel for
    for (int i = 0; i < vocab_size; i++)
        logits[i] = -(_Float16)INFINITY;

    for (int i = 0; i < cutoff; i++) {
        int orig_i = fis[i].idx;
        logits[orig_i] = fpbuf[orig_i];
    }
}

void apply_rpen(_Float16* logits, bool *visited, int vocab_size, float rpen) {
    #pragma omp parallel for
    for (int i = 0; i < vocab_size; i++) {
        if (!visited[i]) continue;
        _Float16 val = logits[i];
        if (val > 0.0f)
            logits[i] = val / rpen;
        else
            logits[i] = val * rpen;
    }
}

void sample(
    GemmaModel *model,
    ModelBuffer *buf,
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

    // Boolean flags
    bool use_topk = topk != 0;
    bool use_topp = topp < 1.0f;
    bool use_rpen = rpen > 1.0f;

    // bool array indicating which tokens have already been processed
    // Used in rpen (repetition penalty)
    bool *visited = NULL;
    if (use_rpen)
        visited = safe_calloc(conf->vocab_size, sizeof(bool), "visited tokens");

    int pos, token;

    // Prefill all the prompt tokens except the last one
    pos = 0;
    for (int *t = tokens; *t != EOT_SENTINEL && *(t+1) != EOT_SENTINEL; t++) {
        if (g_interrupted) {
            if (use_rpen) free(visited);
            return;
        }
        if (use_rpen) visited[*t] = true;
        forward(model, buf, *t, pos++);
    }

    token = tokens[pos];
    _Float16 *probs = NULL;

    // Only allocate probs if needed
    if (temperature != 0.0f)
        probs = safe_malloc(conf->vocab_size * sizeof(*probs), "probs");

    // Float buffer for topk & topp
    FloatIdx *fis = NULL;
    if (use_topk || use_topp)
        fis = safe_malloc(conf->vocab_size * sizeof(*fis), "FloatIdx array");

    for (; pos < seqlen; pos++) {
        if (g_interrupted) break;

        if (use_rpen) visited[token] = true;
        forward(model, buf, token, pos);

        // Compute logits
        gemv(buf->logits, model->embedding, buf->x, conf->vocab_size, conf->embed_dim);

        if (temperature == 0.0f) {
            // Argmax sampling
            token = argmax(buf->logits, conf->vocab_size);
        } else {
            // Apply the temperature
            #pragma omp parallel for
            for (int d = 0; d < conf->vocab_size; d++)
                buf->logits[d] /= (_Float16)temperature;

            if (use_topk)
                apply_topk(buf->logits, fis, conf->vocab_size, topk);
            if (use_topp)
                apply_topp(buf->logits, probs, fis, conf->vocab_size, topp);
            if (use_rpen)
                apply_rpen(buf->logits, visited, conf->vocab_size, rpen);

            // Softmax to get the probs
            softmax(probs, buf->logits, conf->vocab_size);

            // Sample from probs
            float r = (float)rand() / (RAND_MAX + 1.0);
            float sum = 0.0;

            token = conf->vocab_size - 1;
            for (int d = 0; d < conf->vocab_size; d++) {
                sum += (float)probs[d];
                if (r < sum) { token = d; break; }
            }
        }

        int *ret = token_callback(token, model->tokenizer);

        if (ret == SAMPLE_ABORT) break;
        else if (ret != NULL) {
            // Injected a token array (ends with EOT_SENTINEL)
            // Prefill all the tokens except the last one
            int i;
            for (i = 0; (token = ret[i]) != EOT_SENTINEL && ret[i+1] != EOT_SENTINEL; i++) {
                if (g_interrupted) break;
                if (use_rpen) visited[token] = true;
                forward(model, buf, token, pos++);
            }
            if (g_interrupted) break;
            token = ret[i];  // The last element
        }
        // ret == NULL: do nothing
    }

    free(probs);
    if (use_rpen) free(visited);
    if (use_topk || use_topp) free(fis);
}

int *encode(GemmaTokenizer *tok, char *sstr, int *tokens, int *n_tokens) {
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

        if (str[i] >> 7 == 0)
            n_bytes = 1;
        else if (
            i + 1 < len && str[i] >> 5 == 6
            && str[i+1] >> 6 == 2
        )
            n_bytes = 2;
        else if (
            i + 2 < len && str[i] >> 4 == 14
            && str[i+1] >> 6 == 2 && str[i+2] >> 6 == 2
        )
            n_bytes = 3;
        else if (
            i + 3 < len && str[i] >> 3 == 30
            && str[i+1] >> 6 == 2 && str[i+2] >> 6 == 2 && str[i+3] >> 6 == 2
        )
            n_bytes = 4;
        else
            n_bytes = 1;

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
        } else
            tokens[tok_i++] = token;
    }

    while (true) {
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

char *decode(GemmaTokenizer *tok, int id) { return tok->vocab[id]; }


int *generate_callback(int token, GemmaTokenizer *tok) {
    if (token == tok->eos)
        return SAMPLE_ABORT;

    printf("%s", decode(tok, token));
    fflush(stdout);

    return NULL;
}

void generate(
    GemmaModel *model, ModelBuffer *buf,
    char *prompt,
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
    if (bos)
        printf("User: ");
    else
        printf("\nUser: ");

    char user_prompt[65536];
    if (fgets(user_prompt, sizeof(user_prompt), stdin) == NULL) {
        fprintf(stderr, "Failed to read user input\n");
        exit(1);
    }
    user_prompt[strcspn(user_prompt, "\n")] = '\0';

    /*
     * Apply the gemma chat template
     *
     * [<bos>]<start_of_turn>user\n
     * Hello! Introduce yourself.<end_of_turn>\n
     * <start_of_turn>model\n
     * ...
    */
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
    if (token == tok->eot)
        return new_turn(tok, false);

    printf("%s", decode(tok, token));
    fflush(stdout);
    return NULL;
}

void chat(
    GemmaModel *model, ModelBuffer *buf,
    int seqlen, float temperature, int topk, float topp, float rpen
) {
    sample(
        model, buf, new_turn(model->tokenizer, true),
        seqlen, temperature, topk, topp, rpen, chat_callback
    );
}

bool safe_atoui(const char *str, unsigned int *result) {
 
    if (str == NULL) return false;

    char *endptr;
    long long val = strtoll(str, &endptr, 10);

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

bool safe_atof(const char *str, float *result) {
    if (str == NULL) return false;
    
    char *endptr;
    float val = strtof(str, &endptr);
    
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

int main(int argc, char **argv) {
    set_utf8_console();
    setup_signal_handler();

#ifdef _WIN32
    // On Windows, get UTF-8 encoded command line arguments
    char **utf8_argv = get_utf8_argv(&argc);
    if (utf8_argv)
        argv = utf8_argv;
#endif

    if (argc < 2) {
        print_usage();
        printf("\nError: model filename is not provided\n");
#ifdef _WIN32
        if (utf8_argv) free_utf8_argv(utf8_argv, argc);
#endif
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
    char *prompt = "Once upon a time";
    bool chatmode = false;

    // Parse the command line arguments
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-l") || !strcmp(argv[i], "--seqlen")) {
            char *l = argv[++i];
            if (!safe_atoui(l, &seqlen)) {
                print_usage();
                printf("\nError: Invalid argument for -l / --seqlen: '%s'\n", l);
                return 1;
            }
        } else if (!strcmp(argv[i], "-k") || !strcmp(argv[i], "--topk")) {
            char *k = argv[++i];
            if (!safe_atoui(k, &topk)) {
                print_usage();
                printf("\nError: Invalid argument for -k / --topk: '%s'\n", k);
                return 1;
            }
        } else if (!strcmp(argv[i], "-s") || !strcmp(argv[i], "--seed")) {
            char *s = argv[++i];
            if (!safe_atoui(s, &seed)) {
                print_usage();
                printf("\nError: Invalid argument for -s / --seed: '%s'\n", s);
                return 1;
            }
        } else if (!strcmp(argv[i], "-t") || !strcmp(argv[i], "--temperature")) {
            char *t = argv[++i];
            if (!safe_atof(t, &temperature) || temperature < 0.0) {
                print_usage();
                printf("\nError: Invalid argument for -t / --temperature: '%s'\n", t);
                return 1;
            }
        } else if (!strcmp(argv[i], "-p") || !strcmp(argv[i], "--topp")) {
            char *p = argv[++i];
            if (!safe_atof(p, &topp) || topp > 1.0 || topp <= 0.0) {
                print_usage();
                printf("\nError: Invalid argument for -p / --topp: '%s'\n", p);
                return 1;
            }
        } else if (!strcmp(argv[i], "-r") || !strcmp(argv[i], "--rpen")) {
            char *r = argv[++i];
            if (!safe_atof(r, &rpen) || rpen < 1.0) {
                print_usage();
                printf("\nError: Invalid argument for -r / --rpen: '%s'\n", r);
                return 1;
            }
        } else if (!strcmp(argv[i], "-i") || !strcmp(argv[i], "--prompt")) {
            prompt = argv[++i];
        } else if (!strcmp(argv[i], "-c") || !strcmp(argv[i], "--chat")) {
            chatmode = true;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            print_usage();
            return 0;
        } else {
            print_usage();
            printf("\nError: Invalid argument: '%s'\n", argv[i]);
            return 1;
        }
    }

    srand(seed);

    GemmaModel *model = read_model(modelfile);
    ModelBuffer *buf = malloc_buffer(model->config, seqlen);

    if (chatmode)
        chat(model, buf, seqlen, temperature, topk, topp, rpen);
    else
        generate(model, buf, prompt, seqlen, temperature, topk, topp, rpen);

    if (g_interrupted)
        printf("\n\nInterrupted by user\n");

    free_buffer(buf);
    free_model(model);
    return 0;
}
