package com.yishape.lab.math.compute.gpu;

/**
 * Threshold configuration for GPU operations via system properties.
 * Analogous to {@link com.yishape.lab.math.compute.hpc.HpcConfig}.
 *
 * <h2>Memory budget</h2>
 * Default: {@code -Dyishape.gpu.maxMemoryBytes=N} where N is in bytes.
 * 0 or unset = auto-detect via platform-specific query (DXGI on Windows, sysfs on Linux).
 * Environment variable {@code YISHAPE_GPU_MAX_MEM_BYTES} takes priority over the system property
 * (it's read directly in Rust before Java's override is applied).
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

    /**
     * Maximum number of f64 elements per GPU buffer. GPU stores data as f32
     * (4 bytes/element), so this translates to maxBufferElements × 4 bytes.
     * <p>
     * The real limit is {@code max_storage_buffer_binding_size} (typically
     * 128 MB = 134,217,728 bytes on consumer GPUs), NOT {@code max_buffer_size}
     * (256 MB). A buffer that fits in allocation may still be rejected at
     * bind-group creation time if it exceeds the storage binding cap.
     * <p>
     * Default 32M elements × 4 bytes = 128 MB, right at the typical storage
     * binding limit. Set via {@code -Dyishape.gpu.maxBufferElements=N}.
     */
    public static long maxBufferElements() {
        return parseLong("yishape.gpu.maxBufferElements", 32_000_000L);
    }

    /**
     * Returns the user-configured GPU memory budget in bytes from system property.
     * 0 = auto-detect (platform-specific VRAM query in Rust).
     * Set via {@code -Dyishape.gpu.maxMemoryBytes=N}.
     * Environment variable {@code YISHAPE_GPU_MAX_MEM_BYTES} overrides this.
     */
    public static long maxMemoryBytes() {
        return parseLong("yishape.gpu.maxMemoryBytes", 0L);
    }

    /**
     * Returns a human-readable diagnostic string about the GPU memory budget.
     * For logging / debugging purposes.
     */
    public static String memoryBudgetInfo() {
        long fromProp = maxMemoryBytes();
        String envVal = System.getenv("YISHAPE_GPU_MAX_MEM_BYTES");
        StringBuilder sb = new StringBuilder();
        sb.append("GPU memory budget config: ");
        if (envVal != null && !envVal.isEmpty()) {
            sb.append("env YISHAPE_GPU_MAX_MEM_BYTES=").append(envVal);
            if (fromProp > 0) {
                sb.append(" (system property yishape.gpu.maxMemoryBytes=")
                  .append(fromProp).append(" present but env takes priority in Rust)");
            }
        } else if (fromProp > 0) {
            sb.append("system property yishape.gpu.maxMemoryBytes=").append(fromProp);
        } else {
            sb.append("auto-detect (default 2 GiB fallback)");
        }
        return sb.toString();
    }

    private static long parseLong(String key, long def) {
        try {
            return Math.max(0, Long.parseLong(System.getProperty(key, String.valueOf(def))));
        } catch (NumberFormatException e) {
            return def;
        }
    }
}
