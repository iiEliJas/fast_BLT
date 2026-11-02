#include "test_helpers.h"
#include "core/device_info.h"

int run_device_info_tests(void) {
    const blt_device_spec *s = blt_device_lookup("NVIDIA GeForce RTX 4060 Laptop GPU");
    TEST_ASSERT(s != NULL);
    TEST_ASSERT(s->hbm_gbps == 272.0);

    TEST_ASSERT(blt_device_lookup("Some Unknown GPU 9000") == NULL);

    TEST_ASSERT(blt_device_table_size() > 0);

    TEST_ASSERT(blt_mfu(10.0, 20.0) == 0.5);
    TEST_ASSERT(blt_mfu(5.0, 0.0) == 0.0);

    TEST_ASSERT(blt_mbu(1.0, 1.0, 100.0) == 0.01);
    TEST_ASSERT(blt_mbu(1.0, 0.0, 100.0) == 0.0);

    return 1;
}
