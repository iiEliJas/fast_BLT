#include "train_data.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/backend.h"

static void push_window(size_t **offsets, size_t *count, size_t *cap, size_t offset) {
    if (*count == *cap) {
        const size_t next_cap = (*cap == 0) ? 64 : (*cap * 2);
        size_t *grown = (size_t *)realloc(*offsets, next_cap * sizeof(size_t));
        BLT_REQUIRE(grown != NULL, "train_data: out of memory growing the window offset table");
        *offsets = grown;
        *cap = next_cap;
    }
    (*offsets)[(*count)++] = offset;
}

// Windows per entry: length - 1 bytes are usable because the last window still
// needs its target byte at offset + n * window to stay inside the document.
static size_t windows_in_doc(size_t length, size_t window) { return (length > window) ? (length - 1) / window : 0; }

void blt_data_windows_flat(size_t corpus_size, size_t window, blt_data_windows *out) {
    BLT_REQUIRE(out != NULL, "train_data: output is NULL");
    BLT_REQUIRE(window > 0, "train_data: window must be >= 1");
    memset(out, 0, sizeof(*out));
    out->window = window;
    if (corpus_size <= window) {
        BLT_FATAL("train_data: corpus (%zu bytes) does not exceed one window (%zu bytes)", corpus_size, window);
    }
    out->count = (corpus_size - window) / window;
    out->offsets = NULL;
    out->bytes_indexed = corpus_size;
    out->bytes_covered = out->count * window;
    out->tail_bytes = corpus_size - out->bytes_covered;
}

static void check_declared_size(int64_t declared, const char *key, size_t corpus_size) {
    if (declared < 0) return;
    BLT_REQUIRE((uint64_t)declared == (uint64_t)corpus_size,
                "train_data: manifest %s=%lld does not match the corpus (%zu bytes)", key, (long long)declared,
                corpus_size);
}

static void emit_doc_windows(blt_data_windows *out, size_t offset, size_t length, size_t *cap) {
    out->docs_total++;
    out->bytes_indexed += length;
    const size_t n = windows_in_doc(length, out->window);
    for (size_t j = 0; j < n; j++) push_window(&out->offsets, &out->count, cap, offset + j * out->window);
    if (n > 0) out->docs_used++;
    else out->docs_dropped++;
}

static void build_from_index(const blt_json_value *manifest, const char *split, const blt_json_value *entries,
                             size_t corpus_size, blt_data_windows *out) {
    const blt_json_value *bytes = blt_json_get(manifest, "bytes");
    const blt_json_value *split_bytes = blt_json_get(bytes, split);
    char key[160];
    snprintf(key, sizeof(key), "bytes.%s.size_bytes", split);
    check_declared_size(blt_json_get_int(split_bytes, "size_bytes", -1), key, corpus_size);

    const size_t n = blt_json_array_size(entries);
    size_t cap = 0;
    size_t prev_end = 0;
    for (size_t i = 0; i < n; i++) {
        const blt_json_value *entry = blt_json_array_at(entries, i);
        BLT_REQUIRE(entry != NULL && entry->type == BLT_JSON_OBJECT, "train_data: index.%s[%zu] is not an object",
                    split, i);
        const int64_t raw_off = blt_json_get_int(entry, "offset", -1);
        const int64_t raw_len = blt_json_get_int(entry, "length", -1);
        BLT_REQUIRE(raw_off >= 0 && raw_len >= 0, "train_data: index.%s[%zu] needs integer offset and length", split,
                    i);
        BLT_REQUIRE(raw_len > 0, "train_data: index.%s[%zu] has non-positive length %lld", split, i,
                    (long long)raw_len);
        const size_t offset = (size_t)raw_off;
        const size_t length = (size_t)raw_len;
        BLT_REQUIRE((uint64_t)raw_off + (uint64_t)raw_len <= (uint64_t)corpus_size,
                    "train_data: index.%s[%zu] (offset %zu length %zu) runs past the corpus (%zu bytes)", split, i,
                    offset, length, corpus_size);
        BLT_REQUIRE(offset >= prev_end,
                    "train_data: index.%s[%zu] starts at %zu, before the previous entry ends at %zu", split, i, offset,
                    prev_end);
        prev_end = offset + length;
        emit_doc_windows(out, offset, length, &cap);
    }
}

// <split>_files carries sizes only; the corpus joins the documents with a
// single '\n', so offsets are cumulative and the separator byte is checkable.
static void build_from_files(const blt_json_value *manifest, const char *split, const char *files_key,
                             const blt_json_value *files, const uint8_t *corpus, size_t corpus_size,
                             blt_data_windows *out) {
    char size_key[160];
    snprintf(size_key, sizeof(size_key), "%s_size_bytes", split);
    check_declared_size(blt_json_get_int(manifest, size_key, -1), size_key, corpus_size);

    const size_t n = blt_json_array_size(files);
    size_t cap = 0;
    size_t offset = 0;
    for (size_t i = 0; i < n; i++) {
        const blt_json_value *entry = blt_json_array_at(files, i);
        BLT_REQUIRE(entry != NULL && entry->type == BLT_JSON_OBJECT, "train_data: %s[%zu] is not an object", files_key,
                    i);
        const int64_t raw_size = blt_json_get_int(entry, "size", -1);
        BLT_REQUIRE(raw_size > 0, "train_data: %s[%zu] needs a positive integer size", files_key, i);
        const size_t length = (size_t)raw_size;
        BLT_REQUIRE((uint64_t)offset + (uint64_t)length <= (uint64_t)corpus_size,
                    "train_data: %s[%zu] (offset %zu length %zu) runs past the corpus (%zu bytes)", files_key, i,
                    offset, length, corpus_size);
        if (i + 1 < n) {
            BLT_REQUIRE(offset + length < corpus_size && corpus[offset + length] == '\n',
                        "train_data: %s[%zu] at offset %zu is not followed by a '\\n' separator; the manifest does not "
                        "describe this corpus",
                        files_key, i, offset);
        }
        emit_doc_windows(out, offset, length, &cap);
        offset += length + 1;
    }
}

void blt_data_windows_manifest(const blt_json_value *manifest, const char *split, const uint8_t *corpus,
                               size_t corpus_size, size_t window, blt_data_windows *out) {
    BLT_REQUIRE(manifest != NULL, "train_data: manifest is NULL");
    BLT_REQUIRE(split != NULL, "train_data: split is NULL");
    BLT_REQUIRE(corpus != NULL, "train_data: corpus is NULL");
    BLT_REQUIRE(window > 0, "train_data: window must be >= 1");
    memset(out, 0, sizeof(*out));
    out->window = window;

    const blt_json_value *index = blt_json_get(manifest, "index");
    const blt_json_value *entries = blt_json_get(index, split);
    char files_key[160];
    snprintf(files_key, sizeof(files_key), "%s_files", split);
    const blt_json_value *files = blt_json_get(manifest, files_key);

    if (entries != NULL && entries->type == BLT_JSON_ARRAY) {
        build_from_index(manifest, split, entries, corpus_size, out);
    } else if (files != NULL && files->type == BLT_JSON_ARRAY) {
        build_from_files(manifest, split, files_key, files, corpus, corpus_size, out);
    } else {
        BLT_FATAL("train_data: manifest has neither index.%s nor %s", split, files_key);
    }

    out->bytes_covered = out->count * window;
    out->tail_bytes = out->bytes_indexed - out->bytes_covered;
}

size_t blt_data_window_offset(const blt_data_windows *w, size_t index) {
    BLT_REQUIRE(w != NULL, "train_data: window table is NULL");
    BLT_REQUIRE(index < w->count, "train_data: window %zu is out of range (%zu windows)", index, w->count);
    return w->offsets ? w->offsets[index] : index * w->window;
}

void blt_data_windows_free(blt_data_windows *w) {
    if (w == NULL) return;
    free(w->offsets);
    memset(w, 0, sizeof(*w));
}
