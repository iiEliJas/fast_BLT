#include "test_helpers.h"
#include "core/device_info.h"

#include <string.h>

int run_device_info_tests(void) {
    const blt_device_spec *s = blt_device_lookup("NVIDIA GeForce RTX 4060 Laptop GPU");
    TEST_ASSERT(s != NULL);
    TEST_ASSERT(s->hbm_gbps == 272.0);

    TEST_ASSERT(blt_device_lookup("Some Unknown GPU 9000") == NULL);
    TEST_ASSERT(blt_device_lookup(NULL) == NULL);
    TEST_ASSERT(blt_device_lookup("") == NULL);

    TEST_ASSERT(blt_device_table_size() > 0);

    // Longest-match: Ti and non-Ti SKUs share a prefix, so a first-hit lookup
    // would report the base card's peaks.
    s = blt_device_lookup("NVIDIA GeForce RTX 4060 Ti");
    TEST_ASSERT(s != NULL);
    TEST_ASSERT(strcmp(s->name, "NVIDIA GeForce RTX 4060 Ti") == 0);
    TEST_ASSERT(s->hbm_gbps == 288.0);

    s = blt_device_lookup("NVIDIA GeForce RTX 4060");
    TEST_ASSERT(s != NULL);
    TEST_ASSERT(strcmp(s->name, "NVIDIA GeForce RTX 4060") == 0);
    TEST_ASSERT(s->hbm_gbps == 272.0);

    s = blt_device_lookup("NVIDIA GeForce RTX 4070 Ti");
    TEST_ASSERT(s != NULL);
    TEST_ASSERT(strcmp(s->name, "NVIDIA GeForce RTX 4070 Ti") == 0);

    s = blt_device_lookup("NVIDIA H100 PCIe");
    TEST_ASSERT(s != NULL);
    TEST_ASSERT(strcmp(s->name, "NVIDIA H100 PCIe") == 0);
    TEST_ASSERT(s->hbm_gbps == 2000.0);

    s = blt_device_lookup("NVIDIA H100 80GB HBM3");
    TEST_ASSERT(s != NULL);
    TEST_ASSERT(strcmp(s->name, "NVIDIA H100") == 0);
    TEST_ASSERT(s->hbm_gbps == 3350.0);

    // L4 is a prefix of L40/L40S.
    s = blt_device_lookup("NVIDIA L40");
    TEST_ASSERT(s != NULL);
    TEST_ASSERT(strcmp(s->name, "NVIDIA L40") == 0);

    s = blt_device_lookup("NVIDIA L40S");
    TEST_ASSERT(s != NULL);
    TEST_ASSERT(strcmp(s->name, "NVIDIA L40S") == 0);
    TEST_ASSERT(s->hbm_gbps == 864.0);

    s = blt_device_lookup("NVIDIA L4");
    TEST_ASSERT(s != NULL);
    TEST_ASSERT(strcmp(s->name, "NVIDIA L4") == 0);

    // A100 PCIe has more HBM bandwidth than the generic SXM entry.
    s = blt_device_lookup("NVIDIA A100-PCIE");
    TEST_ASSERT(s != NULL);
    TEST_ASSERT(strcmp(s->name, "NVIDIA A100-PCIE") == 0);
    TEST_ASSERT(s->hbm_gbps == 1935.0);

    s = blt_device_lookup("NVIDIA A100-SXM4-40GB");
    TEST_ASSERT(s != NULL);
    TEST_ASSERT(strcmp(s->name, "NVIDIA A100-SXM4-40GB") == 0);
    TEST_ASSERT(s->hbm_gbps == 1555.0);

    // Resolver: dtype columns.
    blt_device_peaks p = blt_device_resolve("NVIDIA GeForce RTX 4060", BLT_PEAK_FP32, 0.0, 0.0);
    TEST_ASSERT(p.spec != NULL);
    TEST_ASSERT(p.tflops == 15.1);
    TEST_ASSERT(p.bandwidth_gbps == 272.0);

    p = blt_device_resolve("NVIDIA GeForce RTX 4060", BLT_PEAK_BF16, 0.0, 0.0);
    TEST_ASSERT(p.tflops == 30.2);

    p = blt_device_resolve("NVIDIA GeForce RTX 4060", BLT_PEAK_FP8, 0.0, 0.0);
    TEST_ASSERT(p.tflops == 60.4);

    // Turing T4 has no FP8, and the resolver must not fall back to bf16.
    p = blt_device_resolve("NVIDIA T4", BLT_PEAK_FP8, 0.0, 0.0);
    TEST_ASSERT(p.spec != NULL);
    TEST_ASSERT(p.tflops == 0.0);

    // Overrides win over the table and work for unknown names too.
    p = blt_device_resolve("NVIDIA GeForce RTX 4060", BLT_PEAK_BF16, 123.0, 456.0);
    TEST_ASSERT(p.spec != NULL);
    TEST_ASSERT(p.tflops == 123.0);
    TEST_ASSERT(p.bandwidth_gbps == 456.0);

    p = blt_device_resolve("Some Unknown GPU 9000", BLT_PEAK_BF16, 123.0, 456.0);
    TEST_ASSERT(p.spec == NULL);
    TEST_ASSERT(p.tflops == 123.0);
    TEST_ASSERT(p.bandwidth_gbps == 456.0);

    // Unknown with no overrides resolves to all-zero peaks.
    p = blt_device_resolve("Some Unknown GPU 9000", BLT_PEAK_BF16, 0.0, 0.0);
    TEST_ASSERT(p.spec == NULL);
    TEST_ASSERT(p.tflops == 0.0);
    TEST_ASSERT(p.bandwidth_gbps == 0.0);

    p = blt_device_resolve(NULL, BLT_PEAK_BF16, 0.0, 0.0);
    TEST_ASSERT(p.spec == NULL);
    TEST_ASSERT(p.tflops == 0.0);
    TEST_ASSERT(p.bandwidth_gbps == 0.0);

    p = blt_device_resolve("", BLT_PEAK_BF16, 0.0, 0.0);
    TEST_ASSERT(p.spec == NULL);

    TEST_ASSERT(blt_mfu(10.0, 20.0) == 0.5);
    TEST_ASSERT(blt_mfu(5.0, 0.0) == 0.0);

    TEST_ASSERT(blt_mbu(1.0, 1.0, 100.0) == 0.01);
    TEST_ASSERT(blt_mbu(1.0, 0.0, 100.0) == 0.0);

    return 1;
}
