// Cross-platform abstractions for Windows/POSIX port
//
//   blt_time_sec() monotonic wall-clock seconds (QPC on Win32, clock_gettime on POSIX)
//   blt_get_temp_dir() platform temp directory path
//   blt_atomic_begin() / blt_atomic_end() write-then-rename file replacement
//   BLT_ACCESS / BLT_F_OK / BLT_R_OK / BLT_X_OK portable access() shim
//   BLT_DEVNULL is "/dev/null" or "NUL"

#ifndef BLT_PLATFORM_H
#define BLT_PLATFORM_H

#include <stdio.h>
#include <string.h>

#ifdef _WIN32

// -- Windows ------------------------
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#include <direct.h>

// access() shim — _access() on MSVC, same signature.
#define BLT_ACCESS _access
#define BLT_F_OK 0
#define BLT_R_OK 4
#define BLT_X_OK 0 // exec-bit meaningless on Windows; 0 = file exists
#define BLT_DEVNULL "NUL"

static inline double blt_time_sec(void) {
    static LARGE_INTEGER freq = {0};
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart;
}

static inline int blt_get_temp_dir(char *buf, size_t len) {
    DWORD n = GetTempPathA((DWORD)len, buf);
    return (n > 0 && n < len) ? 0 : -1;
}

#else

// -- POSIX (Linux, macOS, BSDs) ------------------------
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define BLT_ACCESS access
#define BLT_F_OK F_OK
#define BLT_R_OK R_OK
#define BLT_X_OK X_OK
#define BLT_DEVNULL "/dev/null"

static inline double blt_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static inline int blt_get_temp_dir(char *buf, size_t len) {
    const char *tmp = getenv("TMPDIR");
    if (!tmp || tmp[0] == '\0') tmp = "/tmp";
    size_t n = strlen(tmp);
    if (n >= len) return -1;
    memcpy(buf, tmp, n + 1);
    return 0;
}

#endif // _WIN32

// Atomic file replacement: write the whole payload to "<path>.tmp", flush it to
// the OS, then rename over the target. A process killed mid-save leaves the
// previous file intact instead of truncating it.
//
// blt_atomic_begin returns NULL when the caller's buffer cannot hold the temp
// name; blt_atomic_end returns 0 on success and -1 after unlinking the temp
// file when anything fails.
static inline FILE *blt_atomic_begin(const char *path, char *tmp, size_t cap) {
    const size_t n = strlen(path);
    if (n + 5 > cap) return NULL;
    memcpy(tmp, path, n);
    memcpy(tmp + n, ".tmp", 5);
    return fopen(tmp, "wb");
}

static inline int blt_atomic_end(FILE *f, const char *tmp, const char *path) {
    int ok = (fflush(f) == 0);
#ifdef _WIN32
    if (ok && _commit(_fileno(f)) != 0) ok = 0;
#else
    if (ok && fsync(fileno(f)) != 0) ok = 0;
#endif
    if (fclose(f) != 0) ok = 0;
    if (ok) {
#ifdef _WIN32
        // MSVC rename() refuses an existing target.
        remove(path);
#endif
        if (rename(tmp, path) != 0) ok = 0;
    }
    if (!ok) remove(tmp);
    return ok ? 0 : -1;
}

#endif // BLT_PLATFORM_H
