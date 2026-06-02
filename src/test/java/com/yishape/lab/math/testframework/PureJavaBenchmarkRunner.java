package com.yishape.lab.math.testframework;

import com.yishape.lab.math.compute.hpc.HpcSwitch;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.MethodOrderer;

/**
 * Runs PerformanceBenchmarkTest with HPC disabled.
 */
@TestMethodOrder(MethodOrderer.DisplayName.class)
public class PureJavaBenchmarkRunner extends PerformanceBenchmarkTest {

    @BeforeAll
    static void disableHpc() {
        HpcSwitch.disable();
        System.out.println("=== HPC DISABLED (Pure Java) ===");
    }
}
