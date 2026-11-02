#ifndef BLT_DEVICE_INFO_H
#define BLT_DEVICE_INFO_H

#include <stddef.h>

// Reference (vendor) peak numbers for known GPUs. These assume boost clocks
// and may not match a laptop / thermally limited device; treat as reference peaks.
typedef struct {
    const char *name; // substring matched against cudaGetDeviceProperties.name
    double fp32_tflops;
    double bf16_tflops;
    double fp8_tflops;
    double hbm_gbps;
} blt_device_spec;

// Returns a pointer to the table entry whose `name` is a substring of the
// supplied device name, or NULL if no entry matches (unknown device).
const blt_device_spec *blt_device_lookup(const char *device_name);

// Number of entries in the table.
size_t blt_device_table_size(void);

// Direct access to table entries (for tests / enumeration).
const blt_device_spec *blt_device_table_entry(size_t i);

// Formula helpers (pure, testable).
// MFU = achieved_tflops / peak_tflops  (achieved in TFLOP/s).
double blt_mfu(double achieved_tflops, double peak_tflops);
// MBU = bytes_moved_gb / (time_s * bandwidth_gbps)  (i.e. fraction of peak BW).
double blt_mbu(double bytes_moved_gb, double time_s, double bandwidth_gbps);

#endif
