#include "device_info.h"

#include <string.h>

// Reference peaks. Values are theoretical dense peaks at boost clock; real
// sustained throughput is lower (power, thermal and MIG/vGPU partitioning).
// Sources:
//   - NVIDIA Ada product pages / whitepapers (RTX 4060 .. 4090)
//   - NVIDIA A100 datasheet (SXM 40/80 GB, PCIe 40/80 GB variants)
//   - NVIDIA H100 datasheet (SXM HBM3, PCIe HBM2e)
//   - NVIDIA L4 / L40 / L40S product briefs, NVIDIA T4 datasheet
// Per-device links live in docs/BENCHMARKS.md; the table below only carries the
// numbers so C callers do not need the docs.
static const blt_device_spec BLT_DEVICE_TABLE[] = {
    // name substring, fp32 TFLOP/s, bf16 TFLOP/s, fp8 TFLOP/s, HBM GB/s
    {"NVIDIA GeForce RTX 4060", 15.1, 30.2, 60.4, 272.0},
    {"NVIDIA GeForce RTX 4060 Ti", 22.1, 44.1, 88.2, 288.0},
    {"NVIDIA GeForce RTX 4070", 29.1, 58.3, 116.5, 504.0},
    {"NVIDIA GeForce RTX 4070 Ti", 40.1, 80.1, 160.1, 504.0},
    {"NVIDIA GeForce RTX 4080", 48.7, 97.5, 194.9, 716.8},
    {"NVIDIA GeForce RTX 4090", 82.6, 165.2, 330.3, 1008.0},
    // A100 SXM4-40GB is the generic "A100" HBM2 value; the 80 GB SXM module
    // doubles capacity and bandwidth (2039 GB/s). PCIe 40 GB keeps 1555 and
    // PCIe 80 GB gets 1935.
    {"NVIDIA A100", 19.5, 312.0, 624.0, 1555.0},
    {"NVIDIA A100-SXM4-40GB", 19.5, 312.0, 624.0, 1555.0},
    {"NVIDIA A100-SXM4-80GB", 19.5, 312.0, 624.0, 2039.0},
    {"NVIDIA A100-PCIE", 19.5, 312.0, 624.0, 1935.0},
    {"NVIDIA A100-PCIE-40GB", 19.5, 312.0, 624.0, 1555.0},
    {"NVIDIA A100-PCIE-80GB", 19.5, 312.0, 624.0, 1935.0},
    {"NVIDIA A100 80GB PCIe", 19.5, 312.0, 624.0, 1935.0},
    {"NVIDIA H100", 67.0, 989.0, 1979.0, 3350.0}, // SXM, HBM3
    {"NVIDIA H100 PCIe", 51.0, 756.0, 1513.0, 2000.0},
    {"NVIDIA L4", 30.3, 121.0, 242.0, 300.0},
    {"NVIDIA L40", 90.5, 181.0, 362.0, 864.0},
    {"NVIDIA L40S", 91.6, 366.0, 733.0, 864.0},
    {"NVIDIA T4", 8.1, 65.0, 0.0, 300.0}, // Turing: no FP8 tensor cores
};

const blt_device_spec *blt_device_lookup(const char *device_name) {
    if (!device_name || device_name[0] == '\0') return NULL;
    const blt_device_spec *best = NULL;
    size_t best_len = 0;
    size_t n = sizeof(BLT_DEVICE_TABLE) / sizeof(BLT_DEVICE_TABLE[0]);
    for (size_t i = 0; i < n; i++) {
        const char *name = BLT_DEVICE_TABLE[i].name;
        if (strstr(device_name, name) == NULL) continue;
        size_t len = strlen(name);
        if (len > best_len) {
            best = &BLT_DEVICE_TABLE[i];
            best_len = len;
        }
    }
    return best;
}

size_t blt_device_table_size(void) { return sizeof(BLT_DEVICE_TABLE) / sizeof(BLT_DEVICE_TABLE[0]); }

const blt_device_spec *blt_device_table_entry(size_t i) {
    size_t n = sizeof(BLT_DEVICE_TABLE) / sizeof(BLT_DEVICE_TABLE[0]);
    if (i >= n) return NULL;
    return &BLT_DEVICE_TABLE[i];
}

blt_device_peaks blt_device_resolve(const char *device_name, blt_peak_dtype dtype, double tflops_override,
                                    double bandwidth_override) {
    blt_device_peaks p;
    p.spec = blt_device_lookup(device_name);
    p.tflops = 0.0;
    p.bandwidth_gbps = 0.0;

    if (tflops_override > 0.0) {
        p.tflops = tflops_override;
    } else if (p.spec) {
        switch (dtype) {
        case BLT_PEAK_FP8:
            p.tflops = p.spec->fp8_tflops;
            break;
        case BLT_PEAK_BF16:
            p.tflops = p.spec->bf16_tflops;
            break;
        case BLT_PEAK_FP32:
        default:
            p.tflops = p.spec->fp32_tflops;
            break;
        }
    }

    if (bandwidth_override > 0.0) {
        p.bandwidth_gbps = bandwidth_override;
    } else if (p.spec) {
        p.bandwidth_gbps = p.spec->hbm_gbps;
    }
    return p;
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
