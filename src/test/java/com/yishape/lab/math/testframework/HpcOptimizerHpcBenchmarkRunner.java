package com.yishape.lab.math.testframework;

import com.yishape.lab.math.compute.hpc.HpcSwitch;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Disabled;

/**
 * Runs OptimizerHpcBenchmarkTest with HPC enabled.
 * Measures native gosh-lbfgs performance via Rust/FFM.
 */
@TestMethodOrder(MethodOrderer.DisplayName.class)
// @Disabled("优化器性能基准测试，耗时长，仅在需要时手动启用")
public class HpcOptimizerHpcBenchmarkRunner extends OptimizerHpcBenchmarkTest {

    @BeforeAll
    static void enableHpc() {
        HpcSwitch.enable();
        System.out.println("=== HPC ENABLED ===");
    }
}
