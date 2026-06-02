package com.yishape.lab.math.compute;

/**
 *
 * @author lteb2
 */
public class ComputerConfig {
    /**
     * 是否尝试使用 SIMD（Vector API）。默认 true；若 JVM 未 {@code --add-modules jdk.incubator.vector}，
     * 探测会失败并自动回退到 SISD。可通过 {@code -Dyishape.math.use.simd=false} 强制标量路径。
     */
    public static final boolean USE_SIMD = Boolean.parseBoolean(
            System.getProperty("yishape.math.use.simd", "true"));
    public static final boolean USE_GPU = Boolean.parseBoolean(
            System.getProperty("yishape.math.use.gpu", "true"));
    //矩阵或者向量元素多于此数就使用GPU
    public static final int GPU_THRESHOLD = 10000000; // 10M elements - scalar operations need much higher threshold due to overhead

    // Operation-specific thresholds based on performance analysis
    public static final int GPU_MATRIX_SCALAR_THRESHOLD = 50000000; // 50M elements - scalar ops have high overhead
    public static final int GPU_MATRIX_MULTIPLY_THRESHOLD = 1000000;  // 1M elements - complex ops benefit more
    public static final int GPU_VECTOR_THRESHOLD = 10000000;          // 10M elements - vector ops overhead

    // Minimum thresholds for any GPU operation
    public static final int MIN_GPU_ELEMENTS = 1000000; // Never use GPU below 1M elements

    // Cached SIMD/GPU support detection — computed once at class load
    private static volatile Boolean simdSupported = null;
    private static volatile Boolean gpuSupported = null;

    /**
     * 检测SIMD是否被支持（结果缓存，避免重复反射）
     * @return
     */
    public static boolean checkIfSIMDSupported(){
        Boolean cached = simdSupported;
        if (cached != null) return cached;
        try {
            Class<?> simdClass = Class.forName("com.yishape.lab.math.compute.SIMDDoubleComputer");
            cached = (Boolean) simdClass.getMethod("checkIfSupport").invoke(null);
        } catch (Throwable t) {
            cached = false;
        }
        simdSupported = cached;
        return cached;
    }

    /**
     * 检测GPU是否被支持（结果缓存，避免重复反射）
     * @return
     */
    public static boolean checkIfGPUSupported(){
        Boolean cached = gpuSupported;
        if (cached != null) return cached;
        try {
            Class<?> gpuRuntimeClass = Class.forName("com.yishape.lab.math.compute.gpu.GpuOptionalRuntime");
            cached = (Boolean) gpuRuntimeClass.getMethod("isGpuAvailable").invoke(null);
        } catch (Throwable t) {
            try {
                Class<?> gpuClass = Class.forName("com.yishape.lab.math.compute.GPUDoubleComputer");
                cached = (Boolean) gpuClass.getMethod("isGPUAvailable").invoke(null);
            } catch (Throwable t2) {
                cached = false;
            }
        }
        gpuSupported = cached;
        return cached;
    }
}