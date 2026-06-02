package com.yishape.lab.math.compute.gpu;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link GpuConfig} and {@link GpuSwitch} — configuration and runtime switching.
 */
public class GpuConfigTest {

    @AfterEach
    void restoreSwitch() {
        GpuSwitch.enable();
    }

    // ==================== GpuSwitch ====================

    @Test
    void testSwitchDefaultEnabled() {
        GpuSwitch.enable();
        assertTrue(GpuSwitch.isEnabled());
    }

    @Test
    void testSwitchDisable() {
        GpuSwitch.disable();
        assertFalse(GpuSwitch.isEnabled());
    }

    @Test
    void testSwitchToggle() {
        GpuSwitch.enable();
        boolean newState = GpuSwitch.toggle();
        assertFalse(newState);
        assertFalse(GpuSwitch.isEnabled());

        newState = GpuSwitch.toggle();
        assertTrue(newState);
        assertTrue(GpuSwitch.isEnabled());
    }

    @Test
    void testSwitchRunWith() {
        GpuSwitch.enable();
        GpuSwitch.runWith(false, () -> {
            assertFalse(GpuSwitch.isEnabled());
        });
        // Should restore after runWith
        assertTrue(GpuSwitch.isEnabled());
    }

    @Test
    void testSwitchRunWithNestedRestore() {
        GpuSwitch.enable();
        GpuSwitch.runWith(false, () -> {
            assertFalse(GpuSwitch.isEnabled());
            // Nested runWith
            GpuSwitch.runWith(false, () -> {
                assertFalse(GpuSwitch.isEnabled());
            });
            // Inner runWith restores to outer's value (false)
            assertFalse(GpuSwitch.isEnabled());
        });
        // Outer runWith restores to original (true)
        assertTrue(GpuSwitch.isEnabled());
    }

    // ==================== GpuConfig.allowAttempts ====================

    @Test
    void testAllowAttemptsDefault() {
        // Default: GpuSwitch enabled + no system property override
        GpuSwitch.enable();
        assertTrue(GpuConfig.allowAttempts());
    }

    @Test
    void testAllowAttemptsDisabledBySwitch() {
        GpuSwitch.disable();
        assertFalse(GpuConfig.allowAttempts());
    }

    @Test
    void testAllowAttemptsDisabledBySystemProperty() {
        String orig = System.getProperty("yishape.gpu");
        try {
            System.setProperty("yishape.gpu", "false");
            GpuSwitch.enable();
            assertFalse(GpuConfig.allowAttempts());
        } finally {
            if (orig == null) System.clearProperty("yishape.gpu");
            else System.setProperty("yishape.gpu", orig);
        }
    }

    @Test
    void testAllowAttemptsExplicitlyEnabledByProperty() {
        String orig = System.getProperty("yishape.gpu");
        try {
            System.setProperty("yishape.gpu", "true");
            GpuSwitch.enable();
            assertTrue(GpuConfig.allowAttempts());
        } finally {
            if (orig == null) System.clearProperty("yishape.gpu");
            else System.setProperty("yishape.gpu", orig);
        }
    }

    @Test
    void testAllowAttemptsSwitchOverridesProperty() {
        String orig = System.getProperty("yishape.gpu");
        try {
            System.setProperty("yishape.gpu", "true");
            GpuSwitch.disable();
            // Switch disabled takes precedence
            assertFalse(GpuConfig.allowAttempts());
        } finally {
            if (orig == null) System.clearProperty("yishape.gpu");
            else System.setProperty("yishape.gpu", orig);
        }
    }

    // ==================== GpuConfig Thresholds ====================

    @Test
    void testGemmMinFlopsDefault() {
        assertEquals(1_000_000, GpuConfig.gemmMinFlops());
    }

    @Test
    void testElementwiseMinElementsDefault() {
        assertEquals(100_000, GpuConfig.elementwiseMinElements());
    }

    @Test
    void testActivationMinElementsDefault() {
        assertEquals(100_000, GpuConfig.activationMinElements());
    }

    @Test
    void testReduceMinElementsDefault() {
        assertEquals(10_000, GpuConfig.reduceMinElements());
    }

    @Test
    void testLayernormMinElementsDefault() {
        assertEquals(10_000, GpuConfig.layernormMinElements());
    }

    @Test
    void testBatchnormMinElementsDefault() {
        assertEquals(10_000, GpuConfig.batchnormMinElements());
    }

    @Test
    void testThresholdOverrideViaSystemProperty() {
        String orig = System.getProperty("yishape.gpu.gemm.minFlops");
        try {
            System.setProperty("yishape.gpu.gemm.minFlops", "500000");
            assertEquals(500_000, GpuConfig.gemmMinFlops());
        } finally {
            if (orig == null) System.clearProperty("yishape.gpu.gemm.minFlops");
            else System.setProperty("yishape.gpu.gemm.minFlops", orig);
        }
    }

    @Test
    void testThresholdInvalidValueFallsBackToDefault() {
        String orig = System.getProperty("yishape.gpu.gemm.minFlops");
        try {
            System.setProperty("yishape.gpu.gemm.minFlops", "not_a_number");
            // Should fall back to default (1_000_000)
            assertEquals(1_000_000, GpuConfig.gemmMinFlops());
        } finally {
            if (orig == null) System.clearProperty("yishape.gpu.gemm.minFlops");
            else System.setProperty("yishape.gpu.gemm.minFlops", orig);
        }
    }

    @Test
    void testThresholdNegativeValueClampedToZero() {
        String orig = System.getProperty("yishape.gpu.elementwise.minElements");
        try {
            System.setProperty("yishape.gpu.elementwise.minElements", "-100");
            assertEquals(0, GpuConfig.elementwiseMinElements());
        } finally {
            if (orig == null) System.clearProperty("yishape.gpu.elementwise.minElements");
            else System.setProperty("yishape.gpu.elementwise.minElements", orig);
        }
    }
}
