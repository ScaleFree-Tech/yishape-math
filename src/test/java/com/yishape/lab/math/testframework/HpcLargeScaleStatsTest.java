package com.yishape.lab.math.testframework;

import com.yishape.lab.math.compute.hpc.HpcSwitch;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Disabled;

/**
 * Runs LargeScaleStatsTest with HPC enabled.
 */
@Disabled("大规模统计性能基准，默认跳过；需要时去掉本注解或单独 -Dtest=... 运行")
public class HpcLargeScaleStatsTest extends LargeScaleStatsTest {
    @BeforeAll
    static void enableHpc() {
        HpcSwitch.enable();
        System.out.println("=== LargeScaleStatsTest: HPC ENABLED ===");
    }
}
