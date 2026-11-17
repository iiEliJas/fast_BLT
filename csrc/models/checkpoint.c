#include "models/checkpoint.h"

#include <stdio.h>
#include <string.h>

#include "core/backend.h"
#include "core/platform.h"
#include "models/local_common.h"

// Stable ordering:
//   enc.embed, enc.ngram<k>, enc.L<i>.<self 7>, enc.L<i>.<cross 5>,
//   glob.L<i>.<self 7>, dec.L<i>.<self 7>, dec.L<i>.<cross 5>,
//   dec.lm_head, dec.d0_embed

static const char *SELF_NAMES[] = {
    "norm1", "attn_qkv", "attn_proj", "norm2", "ffn_up", "ffn_gate", "ffn_down",
};

static const char *CROSS_NAMES[] = {
    "cross_norm", "cross_q", "cross_k", "cross_v", "cross_proj",
};

static blt_tensor *self_tensor(blt_local_layer_storage *l, size_t j) {
    blt_tensor *all[] = {
        &l->norm1_weight, &l->attn_qkv_w, &l->attn_proj_w, &l->norm2_weight,
        &l->ffn_up_w,     &l->ffn_gate_w, &l->ffn_down_w,
    };
    return all[j];
}

static blt_tensor *cross_tensor(blt_local_layer_storage *l, size_t j) {
    blt_tensor *all[] = {
        &l->cross_norm_weight, &l->cross_weight_q, &l->cross_weight_k, &l->cross_weight_v, &l->cross_weight_proj,
    };
    return all[j];
}

static blt_tensor *global_self_tensor(blt_transformer_layer_storage *l, size_t j) {
    // Same field order as the local self-attention block.
    return self_tensor((blt_local_layer_storage *)l, j);
}

size_t blt_model_num_tensors(const blt_model *model) {
    const size_t L = model->config.encoder_config.num_layers;
    const size_t G = model->config.global_config.num_layers;
    const size_t D = model->config.decoder_config.num_layers;
    const size_t NT = model->encoder->ngram_weights.num_tables;
    return 1 + NT + (L + D) * 12 + G * 7 + 2;
}

static void emit_name(char *buf, size_t cap, const char *prefix, size_t layer, const char *leaf) {
    snprintf(buf, cap, "%s.L%zu.%s", prefix, layer, leaf);
}

void blt_model_tensor_at(const blt_model *model, size_t index, const char **name, blt_tensor **tensor) {
    static char name_buf[64];
    const size_t L = model->config.encoder_config.num_layers;
    const size_t G = model->config.global_config.num_layers;
    const size_t D = model->config.decoder_config.num_layers;
    const size_t NT = model->encoder->ngram_weights.num_tables;

    size_t idx = 0;
    if (index == idx) {
        *name = "enc.embed";
        *tensor = &model->encoder->byte_embedding_weight;
        return;
    }
    idx++;

    for (size_t k = 0; k < NT; k++, idx++) {
        if (index == idx) {
            snprintf(name_buf, sizeof(name_buf), "enc.ngram%zu", k);
            *name = name_buf;
            *tensor = &model->encoder->ngram_weights.tables[k];
            return;
        }
    }

    for (size_t i = 0; i < L; i++) {
        for (size_t j = 0; j < 7; j++, idx++) {
            if (index == idx) {
                emit_name(name_buf, sizeof(name_buf), "enc", i, SELF_NAMES[j]);
                *name = name_buf;
                *tensor = self_tensor(&model->encoder->layers[i], j);
                return;
            }
        }
        for (size_t j = 0; j < 5; j++, idx++) {
            if (index == idx) {
                emit_name(name_buf, sizeof(name_buf), "enc", i, CROSS_NAMES[j]);
                *name = name_buf;
                *tensor = cross_tensor(&model->encoder->layers[i], j);
                return;
            }
        }
    }

    for (size_t i = 0; i < G; i++) {
        for (size_t j = 0; j < 7; j++, idx++) {
            if (index == idx) {
                emit_name(name_buf, sizeof(name_buf), "glob", i, SELF_NAMES[j]);
                *name = name_buf;
                *tensor = global_self_tensor(&model->global->stack.layer_storage[i], j);
                return;
            }
        }
    }

    for (size_t i = 0; i < D; i++) {
        for (size_t j = 0; j < 7; j++, idx++) {
            if (index == idx) {
                emit_name(name_buf, sizeof(name_buf), "dec", i, SELF_NAMES[j]);
                *name = name_buf;
                *tensor = self_tensor(&model->decoder->layers[i], j);
                return;
            }
        }
        for (size_t j = 0; j < 5; j++, idx++) {
            if (index == idx) {
                emit_name(name_buf, sizeof(name_buf), "dec", i, CROSS_NAMES[j]);
                *name = name_buf;
                *tensor = cross_tensor(&model->decoder->layers[i], j);
                return;
            }
        }
    }

    if (index == idx) {
        *name = "dec.lm_head";
        *tensor = &model->decoder->lm_head_weight;
        return;
    }
    idx++;
    if (index == idx) {
        *name = "dec.d0_embed";
        *tensor = &model->decoder->d0_embed_weight;
        return;
    }

    BLT_FATAL("checkpoint: tensor index %zu out of range (num=%zu)", index, blt_model_num_tensors(model));
}

static uint32_t tensor_ndim(const blt_tensor *t) { return (uint32_t)t->ndim; }

// Host staging for tensor payload I/O: device-resident models cannot fread/
// fwrite their storage directly. CPU tensors take the memcpy paths inside
// upload/download, so behavior there is unchanged.
static void tensor_payload_write(const blt_tensor *t, FILE *f, const char *name, const char *path) {
    if (t->backend == BLT_BACKEND_CPU) {
        if (fwrite(t->data, sizeof(float), t->numel, f) != t->numel)
            BLT_FATAL("checkpoint save: write failed at '%s' (%s)", name, path);
        return;
    }
    float *stage = (float *)malloc(t->numel * sizeof(float));
    BLT_REQUIRE(stage != NULL, "checkpoint save: staging alloc failed");
    blt_tensor_download(t, stage, blt_tensor_bytes(t));
    const size_t wrote = fwrite(stage, sizeof(float), t->numel, f);
    free(stage);
    if (wrote != t->numel) BLT_FATAL("checkpoint save: write failed at '%s' (%s)", name, path);
}

static void tensor_payload_read(blt_tensor *t, FILE *f, const char *name) {
    if (t->backend == BLT_BACKEND_CPU) {
        if (fread(t->data, sizeof(float), t->numel, f) != t->numel)
            BLT_FATAL("checkpoint load: '%s' truncated payload", name);
        return;
    }
    float *stage = (float *)malloc(t->numel * sizeof(float));
    BLT_REQUIRE(stage != NULL, "checkpoint load: staging alloc failed");
    if (fread(stage, sizeof(float), t->numel, f) != t->numel) {
        free(stage);
        BLT_FATAL("checkpoint load: '%s' truncated payload", name);
    }
    blt_tensor_upload(t, stage, blt_tensor_bytes(t));
    free(stage);
}

void blt_model_save(const blt_model *model, const char *path) { blt_model_save_ex(model, path, NULL); }

void blt_model_save_ex(const blt_model *model, const char *path, const char *meta_json) {
    char tmp[4096];
    FILE *f = blt_atomic_begin(path, tmp, sizeof(tmp));
    if (!f) BLT_FATAL("checkpoint save: cannot open '%s'", path);

    const char *meta = meta_json ? meta_json : "{}";
    const uint32_t meta_len = (uint32_t)strlen(meta);
    const uint32_t version = 2;
    const uint32_t num = (uint32_t)blt_model_num_tensors(model);

    if (fwrite("FBLT", 1, 4, f) != 4 || fwrite(&version, sizeof(uint32_t), 1, f) != 1 ||
        fwrite(&meta_len, sizeof(uint32_t), 1, f) != 1 || fwrite(meta, 1, meta_len, f) != meta_len ||
        fwrite(&num, sizeof(uint32_t), 1, f) != 1) {
        BLT_FATAL("checkpoint save: header write failed (%s)", path);
    }

    for (uint32_t i = 0; i < num; i++) {
        const char *name;
        blt_tensor *t;
        blt_model_tensor_at(model, i, &name, &t);

        const uint16_t name_len = (uint16_t)strlen(name);
        const uint32_t ndim = tensor_ndim(t);
        if (fwrite(&name_len, sizeof(uint16_t), 1, f) != 1 || fwrite(name, 1, name_len, f) != name_len ||
            fwrite(&ndim, sizeof(uint32_t), 1, f) != 1 || fwrite(t->shape, sizeof(size_t), ndim, f) != ndim) {
            BLT_FATAL("checkpoint save: write failed at '%s' (%s)", name, path);
        }
        tensor_payload_write(t, f, name, path);
    }

    if (blt_atomic_end(f, tmp, path) != 0) BLT_FATAL("checkpoint save: cannot finalize '%s'", path);
}

void blt_entropy_lm_save(const blt_entropy_lm *lm, const char *path) {
    char tmp[4096];
    FILE *f = blt_atomic_begin(path, tmp, sizeof(tmp));
    if (!f) BLT_FATAL("entropy lm save: cannot open '%s'", path);

    const uint32_t version = 1;
    const uint32_t num = 2 + lm->stack.num_layers * 7;

    if (fwrite("FBLT", 1, 4, f) != 4 || fwrite(&version, sizeof(uint32_t), 1, f) != 1 ||
        fwrite(&num, sizeof(uint32_t), 1, f) != 1)
        BLT_FATAL("entropy lm save: header write failed (%s)", path);

    for (uint32_t i = 0; i < num; i++) {
        blt_tensor *t;
        char namebuf[32];
        const char *name;
        if (i == 0) {
            name = "ent.embed";
            t = (blt_tensor *)&lm->embedding_weight;
        } else if (i == num - 1) {
            name = "ent.lm_head";
            t = (blt_tensor *)&lm->lm_head_weight;
        } else {
            static const char *NAMES[] = {
                "norm1", "attn_qkv", "attn_proj", "norm2", "ffn_up", "ffn_gate", "ffn_down",
            };
            const size_t layer = (i - 1) / 7;
            snprintf(namebuf, sizeof(namebuf), "ent.L%zu.%s", layer, NAMES[(i - 1) % 7]);
            name = namebuf;
            blt_transformer_layer_storage *l = (blt_transformer_layer_storage *)&lm->stack.layer_storage[layer];
            blt_tensor *all[] = {
                &l->norm1_weight, &l->attn_qkv_w, &l->attn_proj_w, &l->norm2_weight,
                &l->ffn_up_w,     &l->ffn_gate_w, &l->ffn_down_w,
            };
            t = all[(i - 1) % 7];
        }

        const uint16_t name_len = (uint16_t)strlen(name);
        const uint32_t ndim = (uint32_t)t->ndim;
        if (fwrite(&name_len, sizeof(uint16_t), 1, f) != 1 || fwrite(name, 1, name_len, f) != name_len ||
            fwrite(&ndim, sizeof(uint32_t), 1, f) != 1 || fwrite(t->shape, sizeof(size_t), ndim, f) != ndim)
            BLT_FATAL("entropy lm save: write failed at '%s'", name);
        tensor_payload_write(t, f, name, path);
    }
    if (blt_atomic_end(f, tmp, path) != 0) BLT_FATAL("entropy lm save: cannot finalize '%s'", path);
}

void blt_entropy_lm_load(blt_entropy_lm *lm, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) BLT_FATAL("entropy lm load: cannot open '%s'", path);

    char magic[4];
    uint32_t version = 0, num = 0;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "FBLT", 4) != 0)
        BLT_FATAL("entropy lm load: bad magic in '%s'", path);
    if (fread(&version, sizeof(uint32_t), 1, f) != 1 || version != 1)
        BLT_FATAL("entropy lm load: unsupported version %u", version);
    const uint32_t want = 2 + lm->stack.num_layers * 7;
    if (fread(&num, sizeof(uint32_t), 1, f) != 1 || num != want)
        BLT_FATAL("entropy lm load: tensor count %u != %u", num, want);

    for (uint32_t i = 0; i < num; i++) {
        blt_tensor *t;
        char want_name[32];
        if (i == 0) {
            strcpy(want_name, "ent.embed");
            t = &lm->embedding_weight;
        } else if (i == num - 1) {
            strcpy(want_name, "ent.lm_head");
            t = &lm->lm_head_weight;
        } else {
            static const char *NAMES[] = {
                "norm1", "attn_qkv", "attn_proj", "norm2", "ffn_up", "ffn_gate", "ffn_down",
            };
            snprintf(want_name, sizeof(want_name), "ent.L%zu.%s", (size_t)(i - 1) / 7, NAMES[(i - 1) % 7]);
            blt_transformer_layer_storage *l = &lm->stack.layer_storage[(i - 1) / 7];
            blt_tensor *all[] = {
                &l->norm1_weight, &l->attn_qkv_w, &l->attn_proj_w, &l->norm2_weight,
                &l->ffn_up_w,     &l->ffn_gate_w, &l->ffn_down_w,
            };
            t = all[(i - 1) % 7];
        }

        uint16_t name_len = 0;
        char name[256];
        uint32_t ndim = 0;
        if (fread(&name_len, sizeof(uint16_t), 1, f) != 1 || fread(name, 1, name_len, f) != name_len ||
            fread(&ndim, sizeof(uint32_t), 1, f) != 1)
            BLT_FATAL("entropy lm load: truncated entry %u", i);
        name[name_len] = '\0';
        if (strcmp(name, want_name) != 0)
            BLT_FATAL("entropy lm load: entry %u is '%s', expected '%s'", i, name, want_name);
        if (ndim != t->ndim || ndim > BLT_MAX_NDIM) BLT_FATAL("entropy lm load: '%s' ndim mismatch", name);
        for (uint32_t d = 0; d < ndim; d++) {
            size_t dim = 0;
            if (fread(&dim, sizeof(size_t), 1, f) != 1 || dim != t->shape[d])
                BLT_FATAL("entropy lm load: '%s' shape mismatch", name);
        }
        tensor_payload_read(t, f, name);
    }
    fclose(f);
}

void blt_model_load(blt_model *model, const char *path) { blt_model_load_ex(model, path, NULL, 0); }

int blt_model_read_meta(const char *path, char *meta_out, size_t meta_cap) {
    FILE *f = fopen(path, "rb");
    if (!f) BLT_FATAL("checkpoint read_meta: cannot open '%s'", path);

    char magic[4];
    uint32_t version = 0, meta_len = 0;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "FBLT", 4) != 0) {
        fclose(f);
        BLT_FATAL("checkpoint read_meta: bad magic in '%s'", path);
    }
    if (fread(&version, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        BLT_FATAL("checkpoint read_meta: truncated version in '%s'", path);
    }
    if (version == 1) {
        fclose(f);
        if (meta_out && meta_cap > 0) meta_out[0] = '\\0';
        return 0;
    }
    if (version != 2) {
        fclose(f);
        BLT_FATAL("checkpoint read_meta: unsupported version %u in '%s'", version, path);
    }
    if (fread(&meta_len, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        BLT_FATAL("checkpoint read_meta: truncated meta_len in '%s'", path);
    }
    if (meta_len > 1024 * 1024) {
        fclose(f);
        BLT_FATAL("checkpoint read_meta: meta_len too large in '%s'", path);
    }
    char *buf = (char *)malloc(meta_len + 1);
    BLT_REQUIRE(buf != NULL, "checkpoint read_meta: alloc failed");
    if (meta_len > 0 && fread(buf, 1, meta_len, f) != meta_len) {
        free(buf);
        fclose(f);
        BLT_FATAL("checkpoint read_meta: truncated meta in '%s'", path);
    }
    buf[meta_len] = '\0';
    if (meta_out && meta_cap > 0) {
        if (meta_len + 1 > meta_cap) {
            free(buf);
            fclose(f);
            BLT_FATAL("checkpoint read_meta: meta too large for buffer");
        }
        memcpy(meta_out, buf, meta_len + 1);
    }
    free(buf);
    fclose(f);
    return 1;
}

void blt_model_load_ex(blt_model *model, const char *path, char *meta_out, size_t meta_cap) {
    FILE *f = fopen(path, "rb");
    if (!f) BLT_FATAL("checkpoint load: cannot open '%s'", path);

    char magic[4];
    uint32_t version = 0;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "FBLT", 4) != 0)
        BLT_FATAL("checkpoint load: bad magic in '%s'", path);
    if (fread(&version, sizeof(uint32_t), 1, f) != 1) BLT_FATAL("checkpoint load: truncated version in '%s'", path);

    uint32_t num = 0;
    if (version == 1) {
        if (fread(&num, sizeof(uint32_t), 1, f) != 1) BLT_FATAL("checkpoint load: truncated count in '%s'", path);
        if (meta_out && meta_cap > 0) meta_out[0] = '\0';
    } else if (version == 2) {
        uint32_t meta_len = 0;
        if (fread(&meta_len, sizeof(uint32_t), 1, f) != 1)
            BLT_FATAL("checkpoint load: truncated meta_len in '%s'", path);
        if (meta_len > 1024 * 1024) BLT_FATAL("checkpoint load: meta_len too large in '%s'", path);
        char *buf = (char *)malloc(meta_len + 1);
        BLT_REQUIRE(buf != NULL, "checkpoint load: meta alloc failed");
        if (meta_len > 0 && fread(buf, 1, meta_len, f) != meta_len) {
            free(buf);
            BLT_FATAL("checkpoint load: truncated meta in '%s'", path);
        }
        buf[meta_len] = '\0';
        if (meta_out && meta_cap > 0) {
            if (meta_len + 1 > meta_cap) {
                free(buf);
                BLT_FATAL("checkpoint load: meta too large for buffer");
            }
            memcpy(meta_out, buf, meta_len + 1);
        }
        free(buf);
        if (fread(&num, sizeof(uint32_t), 1, f) != 1) BLT_FATAL("checkpoint load: truncated count in '%s'", path);
    } else {
        BLT_FATAL("checkpoint load: unsupported version %u in '%s'", version, path);
    }

    if (num != (uint32_t)blt_model_num_tensors(model))
        BLT_FATAL("checkpoint load: tensor count %u != expected %zu ('%s')", num, blt_model_num_tensors(model), path);

    for (uint32_t i = 0; i < num; i++) {
        const char *want_name;
        blt_tensor *t;
        blt_model_tensor_at(model, i, &want_name, &t);

        uint16_t name_len = 0;
        char name[65536];
        uint32_t ndim = 0;
        if (fread(&name_len, sizeof(uint16_t), 1, f) != 1 || fread(name, 1, name_len, f) != name_len ||
            fread(&ndim, sizeof(uint32_t), 1, f) != 1)
            BLT_FATAL("checkpoint load: truncated entry %u ('%.64s')", i, path);
        name[name_len] = '\0';

        if (strcmp(name, want_name) != 0)
            BLT_FATAL("checkpoint load: entry %u is '%s', expected '%s' ('%.64s')", i, name, want_name, path);
        if (ndim != t->ndim || ndim > BLT_MAX_NDIM)
            BLT_FATAL("checkpoint load: '%s' ndim %u != %u", name, ndim, (uint32_t)t->ndim);
        for (uint32_t d = 0; d < ndim; d++) {
            size_t dim = 0;
            if (fread(&dim, sizeof(size_t), 1, f) != 1) BLT_FATAL("checkpoint load: '%s' truncated dims", name);
            if (dim != t->shape[d])
                BLT_FATAL("checkpoint load: '%s' dim[%u] %zu != %zu (config mismatch?)", name, d, dim, t->shape[d]);
        }
        tensor_payload_read(t, f, want_name);
    }

    fclose(f);
}
