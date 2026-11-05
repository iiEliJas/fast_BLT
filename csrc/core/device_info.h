#ifndef BLT_DEVICE_INFO_H
#define BLT_DEVICE_INFO_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Reference (vendor) peak numbers for known GPUs. These assume boost clocks
// and may not match a laptop / thermally limited device; treat as reference peaks.
typedef struct {
    const char *name; // substring matched against cudaGetDeviceProperties.name
    double fp32_tflops;
    double bf16_tflops;
    double fp8_tflops;
    double hbm_gbps;
} blt_device_spec;

// Dtype column to read from the table when no explicit override is supplied.
typedef enum { BLT_PEAK_FP32 = 0, BLT_PEAK_BF16 = 1, BLT_PEAK_FP8 = 2 } blt_peak_dtype;

// Resolved peaks for a device. `spec` is NULL when the name was unknown.
typedef struct {
    const blt_device_spec *spec; // matched table entry, NULL when unknown
    double tflops;               // explicit override wins, else dtype peak
    double bandwidth_gbps;       // explicit override wins, else table HBM
} blt_device_peaks;

// Returns the table entry whose `name` is the LONGEST substring of the supplied
// device name, or NULL if no entry matches (unknown device). Longest-match keeps
// variants such as "RTX 4060 Ti", "H100 PCIe" and "L40S" from collapsing onto
// their shorter base names.
const blt_device_spec *blt_device_lookup(const char *device_name);

// Number of entries in the table.
size_t blt_device_table_size(void);

// Direct access to table entries (for tests / enumeration).
const blt_device_spec *blt_device_table_entry(size_t i);

// Resolves the effective peaks for `device_name`. A positive override always
// wins; otherwise the matched entry's dtype column / HBM bandwidth is used.
// Unknown devices with no overrides resolve to {NULL, 0, 0}.
blt_device_peaks blt_device_resolve(const char *device_name, blt_peak_dtype dtype, double tflops_override,
                                    double bandwidth_override);

// Formula helpers (pure, testable).
// MFU = achieved_tflops / peak_tflops  (achieved in TFLOP/s).
double blt_mfu(double achieved_tflops, double peak_tflops);
// MBU = bytes_moved_gb / (time_s * bandwidth_gbps)  (i.e. fraction of peak BW).
double blt_mbu(double bytes_moved_gb, double time_s, double bandwidth_gbps);

#ifdef __cplusplus
}
#endif

#endif
