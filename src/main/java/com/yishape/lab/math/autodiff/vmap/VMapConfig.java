package com.yishape.lab.math.autodiff.vmap;

/**
 * Threshold configuration for vmap operations via system properties.
 * Analogous to {@link com.yishape.lab.math.compute.gpu.GpuConfig}.
 */
final class VMapConfig {

    private VMapConfig() {}

    /** Minimum total elements before GPU vmap path is attempted. */
    static long gpuVMapThreshold() {
        return parseLong("yishape.vmap.gpu.threshold", 1_000_000L);
    }

    /** Minimum total elements before HPC element-wise path is attempted in vmap. */
    static long hpcElementwiseThreshold() {
        return parseLong("yishape.vmap.hpc.elementwiseThreshold", 1_000_000L);
    }

    private static long parseLong(String key, long def) {
        try {
            return Math.max(0, Long.parseLong(System.getProperty(key, String.valueOf(def))));
        } catch (NumberFormatException e) {
            return def;
        }
    }
}
