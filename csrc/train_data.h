#ifndef BLT_TRAIN_DATA_H
#define BLT_TRAIN_DATA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#include "core/json.h"

// Training-window table. A window is `window` consecutive bytes plus a
// next-byte target at start+window, so a window needs window+1 bytes of room.
// offsets is NULL in flat mode, where window i starts at i * window.
typedef struct {
    size_t count;         // number of training windows
    size_t *offsets;      // malloc'd window start offsets; NULL in flat mode
    size_t window;        // bytes per window
    size_t docs_total;    // manifest entries considered
    size_t docs_used;     // entries that contributed >= 1 window
    size_t docs_dropped;  // entries too short to yield a window
    size_t bytes_indexed; // sum of manifest entry lengths (excludes separators)
    size_t bytes_covered; // count * window
    size_t tail_bytes;    // bytes_indexed - bytes_covered
} blt_data_windows;

// Legacy flat-stream slicing: window i covers [i*window, (i+1)*window) with no
// document alignment. Fatal when the corpus cannot hold one window.
void blt_data_windows_flat(size_t corpus_size, size_t window, blt_data_windows *out);

// Document-aligned slicing from a corpus manifest. Accepts an index.<split>
// array of {offset, length} objects, or a <split>_files array of {size}
// objects whose entries are joined by single '\n' separators in the corpus.
// Fatal on a manifest/corpus mismatch or an invalid entry.
void blt_data_windows_manifest(const blt_json_value *manifest, const char *split, const uint8_t *corpus,
                               size_t corpus_size, size_t window, blt_data_windows *out);

// Start offset of window `index`: manifest offsets when present, otherwise
// index * window.
size_t blt_data_window_offset(const blt_data_windows *w, size_t index);

void blt_data_windows_free(blt_data_windows *w);

#ifdef __cplusplus
}
#endif

#endif // BLT_TRAIN_DATA_H
