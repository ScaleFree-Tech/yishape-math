package com.yishape.lab.math.testframework;

import com.yishape.lab.math.compute.hpc.HpcSwitch;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.MethodOrderer;

/**
 * Runs OptimizerHpcBenchmarkTest with HPC disabled (pure Java fallback path).
 * Verifies that RustLBFGS/RustOWLQN fallback matches RereLBFGS performance.
 */
@TestMethodOrder(MethodOrderer.DisplayName.class)
public class PureJavaOptimizerHpcBenchmarkRunner extends OptimizerHpcBenchmarkTest {

    @BeforeAll
    static void disableHpc() {
        HpcSwitch.disable();
        System.out.println("=== HPC DISABLED (Pure Java) ===");
    }
}
