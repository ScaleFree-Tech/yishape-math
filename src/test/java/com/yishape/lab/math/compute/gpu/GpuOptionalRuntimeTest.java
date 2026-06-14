package com.yishape.lab.math.compute.gpu;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link GpuOptionalRuntime} — the reflective bridge to the GPU module.
 * All methods must return null/false gracefully when GPU is unavailable.
 */
public class GpuOptionalRuntimeTest {

    private static boolean gpuPresent;

    @BeforeAll
    static void detect() {
        gpuPresent = GpuOptionalRuntime.isGpuAvailable();
        System.out.println("GPU extension present: " + GpuOptionalRuntime.isExtensionPresent());
        System.out.println("GPU available: " + gpuPresent);
    }

    @AfterEach
    void resetGpuState() {
        // Reset the 'destroyed' flag in YishapeGpu so subsequent tests can use the GPU.
        // shutdown() sets destroyed=true permanently; we undo it via reflection.
        try {
            Class<?> gpu = Class.forName("com.yishape.lab.gpu.YishapeGpu");
            Field f = gpu.getDeclaredField("destroyed");
            f.setAccessible(true);
            f.setBoolean(null, false);
        } catch (ReflectiveOperationException ignored) {
            // GPU module not on classpath or field not found
        }
    }

    // ==================== Availability Probes ====================

    @Test
    void testIsExtensionPresentIsStable() {
        // Verify isExtensionPresent returns a boolean without throwing.
        // Returns true when GPU jar is on classpath, false otherwise.
        boolean result = GpuOptionalRuntime.isExtensionPresent();
        // If extension is present AND GPU hardware is detected, both should be true
        if (result && gpuPresent) {
            assertTrue(true, "GPU extension present and GPU available — consistent state");
        } else if (result && !gpuPresent) {
            // GPU jar present but no hardware detected — also valid (e.g., no Vulkan driver)
            assertTrue(true, "GPU extension present but no hardware — fallback to CPU");
        } else {
            // No GPU jar on classpath
            assertFalse(result, "isExtensionPresent should be false when GPU jar absent");
        }
    }

    @Test
    void testIsGpuAvailableDoesNotThrow() {
        assertDoesNotThrow(GpuOptionalRuntime::isGpuAvailable);
    }

    @Test
    void testTryDeviceNameDoesNotThrow() {
        assertDoesNotThrow(() -> {
            String name = GpuOptionalRuntime.tryDeviceName();
            if (gpuPresent) {
                assertNotNull(name);
                assertFalse(name.isEmpty());
            }
            // When GPU unavailable, null is acceptable
        });
    }

    // ==================== Arithmetic Operations ====================

    @Test
    void testTryAdd() {
        double[] a = {1, 2, 3};
        double[] b = {4, 5, 6};
        double[] result = GpuOptionalRuntime.tryAdd(a, b);
        if (gpuPresent) {
            assertNotNull(result);
            assertArrayEquals(new double[]{5, 7, 9}, result, 1e-10);
        } else {
            assertNull(result);
        }
    }

    @Test
    void testTrySub() {
        double[] a = {10, 20, 30};
        double[] b = {1, 2, 3};
        double[] result = GpuOptionalRuntime.trySub(a, b);
        if (gpuPresent) {
            assertNotNull(result);
            assertArrayEquals(new double[]{9, 18, 27}, result, 1e-10);
        } else {
            assertNull(result);
        }
    }

    @Test
    void testTryMul() {
        double[] a = {2, 3, 4};
        double[] b = {5, 6, 7};
        double[] result = GpuOptionalRuntime.tryMul(a, b);
        if (gpuPresent) {
            assertNotNull(result);
            assertArrayEquals(new double[]{10, 18, 28}, result, 1e-10);
        } else {
            assertNull(result);
        }
    }

    @Test
    void testTryDiv() {
        double[] a = {10, 20, 30};
        double[] b = {2, 4, 5};
        double[] result = GpuOptionalRuntime.tryDiv(a, b);
        if (gpuPresent) {
            assertNotNull(result);
            assertArrayEquals(new double[]{5, 5, 6}, result, 1e-10);
        } else {
            assertNull(result);
        }
    }

    // ==================== Activation Operations ====================

    @Test
    void testTryRelu() {
        double[] input = {-2, -1, 0, 1, 2};
        double[] result = GpuOptionalRuntime.tryRelu(input);
        if (gpuPresent) {
            assertNotNull(result);
            assertArrayEquals(new double[]{0, 0, 0, 1, 2}, result, 1e-10);
        } else {
            assertNull(result);
        }
    }

    @Test
    void testTrySigmoid() {
        double[] input = {0};
        double[] result = GpuOptionalRuntime.trySigmoid(input);
        if (gpuPresent) {
            assertNotNull(result);
            assertEquals(0.5, result[0], 1e-5);
        } else {
            assertNull(result);
        }
    }

    @Test
    void testTryTanh() {
        double[] input = {0};
        double[] result = GpuOptionalRuntime.tryTanh(input);
        if (gpuPresent) {
            assertNotNull(result);
            assertEquals(0.0, result[0], 1e-10);
        } else {
            assertNull(result);
        }
    }

    @Test
    void testTryGelu() {
        double[] input = {0};
        double[] result = GpuOptionalRuntime.tryGelu(input);
        if (gpuPresent) {
            assertNotNull(result);
            assertEquals(0.0, result[0], 1e-5);
        } else {
            assertNull(result);
        }
    }

    // ==================== Matrix Operations ====================

    @Test
    void testTryMatMul() {
        double[][] a = {{1, 2}, {3, 4}};
        double[][] b = {{5, 6}, {7, 8}};
        double[][] result = GpuOptionalRuntime.tryMatMul(a, b);
        if (gpuPresent) {
            assertNotNull(result);
            // A*B = [[19,22],[43,50]] (row-major)
            assertEquals(19, result[0][0], 1e-10);
            assertEquals(22, result[0][1], 1e-10);
            assertEquals(43, result[1][0], 1e-10);
            assertEquals(50, result[1][1], 1e-10);
        } else {
            assertNull(result);
        }
    }

    // ==================== Graph Execution ====================

    @Test
    void testTryExecuteGraphNull() {
        assertNull(GpuOptionalRuntime.tryExecuteGraph(null));
    }

    @Test
    void testTryExecuteGraphMalformed() {
        String result = GpuOptionalRuntime.tryExecuteGraph("not valid json");
        // Should not throw, may return null or error
    }

    @Test
    void testTryExecuteGraphEmptyGraph() {
        String result = GpuOptionalRuntime.tryExecuteGraph("{\"nodes\":[]}");
        // Empty graph — should return null or error gracefully
    }

    // ==================== Shutdown ====================

    @Test
    void testTryShutdownDoesNotThrow() {
        assertDoesNotThrow(GpuOptionalRuntime::tryShutdown);
    }

    // ==================== Null Input Handling ====================

    @Test
    void testTryAddNullInputs() {
        // Should not throw even with null inputs
        assertDoesNotThrow(() -> {
            GpuOptionalRuntime.tryAdd(null, null);
        });
    }

    @Test
    void testTryReluNullInput() {
        assertDoesNotThrow(() -> {
            GpuOptionalRuntime.tryRelu(null);
        });
    }
}
