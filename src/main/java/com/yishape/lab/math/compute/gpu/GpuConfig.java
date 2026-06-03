package com.yishape.lab.math.compute.gpu;

/**
 * Threshold configuration for GPU operations via system properties.
 * Analogous to {@link com.yishape.lab.math.compute.hpc.HpcConfig}.
 */
public final class GpuConfig {
    private GpuConfig() {}

    public static boolean allowAttempts() {
        return GpuSwitch.isEnabled()
            && Boolean.parseBoolean(System.getProperty("yishape.gpu", "true"));
    }

    public static long gemmMinFlops() {
        return parseLong("yishape.gpu.gemm.minFlops", 1_000_000L);
    }

    public static long elementwiseMinElements() {
        return parseLong("yishape.gpu.elementwise.minElements", 100_000L);
    }

    public static long activationMinElements() {
        return parseLong("yishape.gpu.activation.minElements", 100_000L);
    }

    public static long reduceMinElements() {
        return parseLong("yishape.gpu.reduce.minElements", 10_000L);
    }

    public static long layernormMinElements() {
        return parseLong("yishape.gpu.layernorm.minElements", 10_000L);
    }

    public static long batchnormMinElements() {
        return parseLong("yishape.gpu.batchnorm.minElements", 10_000L);
    }

    public static long im2colMinElements() {
        return parseLong("yishape.gpu.im2col.minElements", 100_000L);
    }

    private static long parseLong(String key, long def) {
        try {
            return Math.max(0, Long.parseLong(System.getProperty(key, String.valueOf(def))));
        } catch (NumberFormatException e) {
            return def;
        }
    }
}
