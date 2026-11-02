#include "device_info.h"

#include <string.h>

// Reference peaks. Sources: NVIDIA product pages / datasheets (boost clocks).
// Values are theoretical peaks; real sustained throughput is lower.
static const blt_device_spec BLT_DEVICE_TABLE[] = {
    // name substring, fp32 TFLOP/s, bf16 TFLOP/s, fp8 TFLOP/s, HBM GB/s
    {"NVIDIA GeForce RTX 4060", 15.1, 30.2, 60.4, 272.0},
    {"NVIDIA GeForce RTX 4060 Ti", 22.1, 44.1, 88.2, 288.0},
    {"NVIDIA GeForce RTX 4070", 29.1, 58.3, 116.5, 504.0},
    {"NVIDIA GeForce RTX 4070 Ti", 40.1, 80.1, 160.1, 504.0},
    {"NVIDIA GeForce RTX 4080", 48.7, 97.5, 194.9, 716.8},
    {"NVIDIA GeForce RTX 4090", 82.6, 165.2, 330.3, 1008.0},
    {"NVIDIA A100", 19.5, 312.0, 624.0, 1555.0}, // SXM, bf16 tensor
    {"NVIDIA A100-PCIE", 19.5, 312.0, 624.0, 1555.0},
    {"NVIDIA H100", 67.0, 989.0, 1979.0, 3350.0}, // SXM
    {"NVIDIA H100 PCIe", 51.0, 756.0, 1513.0, 2000.0},
    {"NVIDIA L4", 30.3, 121.0, 242.0, 300.0},
    {"NVIDIA L40", 90.5, 181.0, 362.0, 864.0},
    {"NVIDIA T4", 8.1, 65.0, 130.0, 300.0},
};

const blt_device_spec *blt_device_lookup(const char *device_name) {
    if (!device_name) return NULL;
    size_t n = sizeof(BLT_DEVICE_TABLE) / sizeof(BLT_DEVICE_TABLE[0]);
    for (size_t i = 0; i < n; i++) {
        if (strstr(device_name, BLT_DEVICE_TABLE[i].name) != NULL) {
            return &BLT_DEVICE_TABLE[i];
        }
    }
    return NULL;
}

size_t blt_device_table_size(void) { return sizeof(BLT_DEVICE_TABLE) / sizeof(BLT_DEVICE_TABLE[0]); }

const blt_device_spec *blt_device_table_entry(size_t i) {
    size_t n = sizeof(BLT_DEVICE_TABLE) / sizeof(BLT_DEVICE_TABLE[0]);
    if (i >= n) return NULL;
    return &BLT_DEVICE_TABLE[i];
}

double blt_mfu(double achieved_tflops, double peak_tflops) {
    if (peak_tflops <= 0.0) return 0.0;
    return achieved_tflops / peak_tflops;
}

double blt_mbu(double bytes_moved_gb, double time_s, double bandwidth_gbps) {
    if (time_s <= 0.0 || bandwidth_gbps <= 0.0) return 0.0;
    double achieved_gbps = bytes_moved_gb / time_s;
    return achieved_gbps / bandwidth_gbps;
}
