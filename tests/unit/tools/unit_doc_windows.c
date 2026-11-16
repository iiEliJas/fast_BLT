#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_helpers.h"
#include "test_suite.h"

#include "core/allocator.h"
#include "core/json.h"
#include "train_data.h"

#define MANIFEST_ARENA_BYTES (1u << 20)

static blt_json_value *parse_manifest(blt_arena *arena, const char *text) {
    blt_arena_reset(arena);
    return blt_json_parse(arena, text);
}

// Legacy slicing: count must stay (corpus_size - window) / window because the
// trainer also uploads the next-byte target at start+window.
static int test_flat_windows(void) {
    blt_data_windows w;
    blt_data_windows_flat(100, 32, &w);
    TEST_ASSERT(w.count == 2);
    TEST_ASSERT(w.offsets == NULL);
    TEST_ASSERT(w.window == 32);
    TEST_ASSERT(w.bytes_indexed == 100);
    TEST_ASSERT(w.bytes_covered == 64);
    TEST_ASSERT(w.tail_bytes == 36);
    TEST_ASSERT(w.docs_total == 0 && w.docs_used == 0 && w.docs_dropped == 0);
    TEST_ASSERT(blt_data_window_offset(&w, 0) == 0);
    TEST_ASSERT(blt_data_window_offset(&w, 1) == 32);
    blt_data_windows_free(&w);

    printf("    flat: 100B corpus / 32B window -> 2 windows, no offsets\n");
    return 1;
}

// index.<split> mode: every emitted window (target byte included) must stay
// inside one document span, and two spans must never overlap.
static int test_manifest_alignment(void) {
    static const size_t spans[3][2] = {{0, 101}, {102, 40}, {143, 200}};
    const size_t corpus_size = 343;
    const size_t window = 32;
    const char *json = "{\"bytes\":{\"train\":{\"size_bytes\":343}},"
                       "\"index\":{\"train\":["
                       "{\"offset\":0,\"length\":101},"
                       "{\"offset\":102,\"length\":40},"
                       "{\"offset\":143,\"length\":200}]}}";

    blt_arena *arena = blt_arena_create(MANIFEST_ARENA_BYTES, BLT_BACKEND_CPU);
    TEST_ASSERT(arena != NULL);
    uint8_t *corpus = (uint8_t *)calloc(corpus_size, 1);
    TEST_ASSERT(corpus != NULL);

    blt_data_windows w;
    blt_data_windows_manifest(parse_manifest(arena, json), "train", corpus, corpus_size, window, &w);

    TEST_ASSERT(w.count == 10); // (101-1)/32 + (40-1)/32 + (200-1)/32
    TEST_ASSERT(w.offsets != NULL);
    TEST_ASSERT(w.docs_total == 3);
    TEST_ASSERT(w.docs_used == 3);
    TEST_ASSERT(w.docs_dropped == 0);
    TEST_ASSERT(w.bytes_indexed == 341);
    TEST_ASSERT(w.bytes_covered == 320);
    TEST_ASSERT(w.tail_bytes == 21);

    for (size_t i = 1; i < 3; i++) {
        TEST_ASSERT(spans[i][0] >= spans[i - 1][0] + spans[i - 1][1]);
    }
    for (size_t i = 0; i < w.count; i++) {
        const size_t start = blt_data_window_offset(&w, i);
        const size_t end = start + window;
        int inside = 0;
        for (size_t s = 0; s < 3; s++) {
            // The target byte at `end` must also stay inside the document.
            if (start >= spans[s][0] && end < spans[s][0] + spans[s][1]) inside = 1;
        }
        TEST_ASSERT(inside);
    }

    blt_data_windows_free(&w);
    free(corpus);
    blt_arena_destroy(arena);
    printf("    mask: 3 docs -> 10 windows, every window+target inside one doc\n");
    return 1;
}

// A document must be strictly longer than the window to contribute anything:
// a window that fills the document has no room for its target byte.
static int test_short_doc_dropped(void) {
    const size_t corpus_size = 133;
    const size_t window = 32;
    const char *json = "{\"index\":{\"train\":["
                       "{\"offset\":0,\"length\":32},"
                       "{\"offset\":33,\"length\":100}]}}";

    blt_arena *arena = blt_arena_create(MANIFEST_ARENA_BYTES, BLT_BACKEND_CPU);
    TEST_ASSERT(arena != NULL);
    uint8_t *corpus = (uint8_t *)calloc(corpus_size, 1);
    TEST_ASSERT(corpus != NULL);

    blt_data_windows w;
    blt_data_windows_manifest(parse_manifest(arena, json), "train", corpus, corpus_size, window, &w);

    TEST_ASSERT(w.docs_total == 2);
    TEST_ASSERT(w.docs_used == 1);
    TEST_ASSERT(w.docs_dropped == 1);
    TEST_ASSERT(w.count == 3); // (100-1)/32
    TEST_ASSERT(w.bytes_indexed == 132);
    for (size_t i = 0; i < w.count; i++) {
        const size_t start = blt_data_window_offset(&w, i);
        TEST_ASSERT(start >= 33 && start + window <= 133);
    }

    blt_data_windows_free(&w);
    free(corpus);
    blt_arena_destroy(arena);
    printf("    mask: window-sized doc contributes 0 windows (docs_dropped=1)\n");
    return 1;
}

// <split>_files reconstructs offsets from sizes and the single '\n' separator
// between documents, which is the layout of data/sample_corpus_manifest.json.
static int test_files_layout(void) {
    static const size_t sizes[3] = {10, 20, 5};
    const size_t corpus_size = 37;
    const size_t window = 4;
    const char *json = "{\"train_size_bytes\":37,\"train_files\":[{\"size\":10},{\"size\":20},{\"size\":5}]}";

    blt_arena *arena = blt_arena_create(MANIFEST_ARENA_BYTES, BLT_BACKEND_CPU);
    TEST_ASSERT(arena != NULL);
    uint8_t *corpus = (uint8_t *)malloc(corpus_size);
    TEST_ASSERT(corpus != NULL);
    memset(corpus, 'x', corpus_size);
    size_t off = 0;
    for (size_t i = 0; i < 3; i++) {
        off += sizes[i];
        if (i + 1 < 3) corpus[off++] = '\n';
    }
    TEST_ASSERT(off == corpus_size);

    blt_data_windows w;
    blt_data_windows_manifest(parse_manifest(arena, json), "train", corpus, corpus_size, window, &w);

    TEST_ASSERT(w.count == 7); // (10-1)/4 + (20-1)/4 + (5-1)/4
    TEST_ASSERT(w.docs_total == 3);
    TEST_ASSERT(w.docs_used == 3);
    TEST_ASSERT(w.docs_dropped == 0);
    TEST_ASSERT(w.bytes_indexed == 35);
    TEST_ASSERT(w.bytes_covered == 28);
    TEST_ASSERT(w.tail_bytes == 7);

    size_t starts[3];
    size_t ends[3];
    starts[0] = 0;
    ends[0] = sizes[0];
    starts[1] = sizes[0] + 1;
    ends[1] = starts[1] + sizes[1];
    starts[2] = ends[1] + 1;
    ends[2] = starts[2] + sizes[2];
    for (size_t i = 0; i < w.count; i++) {
        const size_t start = blt_data_window_offset(&w, i);
        const size_t end = start + window;
        int inside = 0;
        for (size_t s = 0; s < 3; s++) {
            if (start >= starts[s] && end < ends[s]) inside = 1;
        }
        TEST_ASSERT(inside);
    }

    blt_data_windows_free(&w);
    free(corpus);
    blt_arena_destroy(arena);
    printf("    files: sizes{10,20,5} -> 7 windows across the 3 joined files\n");
    return 1;
}

// An empty split yields zero windows; the trainer is responsible for rejecting
// the run, so the builder must not fabricate a window.
static int test_empty_split(void) {
    const char *json = "{\"index\":{\"train\":[]}}";
    blt_arena *arena = blt_arena_create(MANIFEST_ARENA_BYTES, BLT_BACKEND_CPU);
    TEST_ASSERT(arena != NULL);
    uint8_t corpus[64];
    memset(corpus, 0, sizeof(corpus));

    blt_data_windows w;
    blt_data_windows_manifest(parse_manifest(arena, json), "train", corpus, sizeof(corpus), 16, &w);
    TEST_ASSERT(w.count == 0);
    TEST_ASSERT(w.offsets == NULL);
    TEST_ASSERT(w.docs_total == 0);
    TEST_ASSERT(w.bytes_indexed == 0);
    TEST_ASSERT(w.bytes_covered == 0);
    TEST_ASSERT(w.tail_bytes == 0);
    blt_data_windows_free(&w);

    blt_arena_destroy(arena);
    printf("    mask: empty index.<split> -> 0 windows\n");
    return 1;
}

int run_doc_window_tests(void) {
    int ok = 1;
    ok &= test_flat_windows();
    ok &= test_manifest_alignment();
    ok &= test_short_doc_dropped();
    ok &= test_files_layout();
    ok &= test_empty_split();
    return ok;
}
