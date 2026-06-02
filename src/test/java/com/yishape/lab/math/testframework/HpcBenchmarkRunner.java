package com.yishape.lab.math.testframework;

import com.yishape.lab.math.compute.hpc.HpcSwitch;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.MethodOrderer;

/**
 * Runs PerformanceBenchmarkTest with HPC enabled.
 */
@TestMethodOrder(MethodOrderer.DisplayName.class)
public class HpcBenchmarkRunner extends PerformanceBenchmarkTest {

    @BeforeAll
    static void enableHpc() {
        HpcSwitch.enable();
        System.out.println("=== HPC ENABLED ===");
    }
}
