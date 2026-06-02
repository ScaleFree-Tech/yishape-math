package com.yishape.lab.math.compute.hpc;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * HpcHealthChecker 测试类
 */
@DisplayName("HPC 健康检查器测试 / HPC Health Checker Tests")
public class HpcHealthCheckerTest {

    @Test
    @DisplayName("check() 不抛异常并返回非空报告 / check() returns non-null report without exception")
    public void testCheckDoesNotThrow() {
        HpcHealthReport report = HpcHealthChecker.check();
        assertNotNull(report, "报告不应为 null");
        // 无论 HPC 是否可用，报告都应包含基本状态
        assertNotNull(report.getDetails(), "details 不应为 null");
        assertFalse(report.getDetails().isEmpty(), "details 不应为空");
    }

    @Test
    @DisplayName("toString() 输出包含关键字段 / toString() contains key fields")
    public void testReportToString() {
        HpcHealthReport report = HpcHealthChecker.check();
        String s = report.toString();
        assertTrue(s.contains("HPC Health Report"), "应包含标题");
        assertTrue(s.contains("Extension present"), "应包含扩展状态");
        assertTrue(s.contains("Native runtime available"), "应包含原生运行时状态");
        assertTrue(s.contains("Overall healthy"), "应包含整体健康状态");
    }

    @Test
    @DisplayName("HPC 不可用时 overallHealthy 为 false / overallHealthy false when HPC unavailable")
    public void testOverallHealthyWhenUnavailable() {
        HpcHealthReport report = HpcHealthChecker.check();
        if (!report.isExtensionPresent() || !report.isNativeRuntimeAvailable()) {
            assertFalse(report.isOverallHealthy(),
                "扩展或原生不可用时，overallHealthy 应为 false");
        }
    }

    @Test
    @DisplayName("HPC 可用时各子项状态一致 / sub-item consistency when HPC available")
    public void testSubItemConsistency() {
        HpcHealthReport report = HpcHealthChecker.check();
        if (!report.isExtensionPresent() || !report.isNativeRuntimeAvailable()) {
            // HPC 不可用，跳过
            return;
        }
        // 若 overallHealthy 为 true，则所有子项必须为 true
        if (report.isOverallHealthy()) {
            assertTrue(report.isMatMulHealthy(), "overallHealthy 时 MatMul 应为 true");
            assertTrue(report.isLbfgsHealthy(), "overallHealthy 时 L-BFGS 应为 true");
            assertTrue(report.isCholeskyHealthy(), "overallHealthy 时 Cholesky 应为 true");
            assertTrue(report.isSvdHealthy(), "overallHealthy 时 SVD 应为 true");
        }
    }
}
