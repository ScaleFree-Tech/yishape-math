package com.yishape.lab.math.compute.gpu;

/**
 * Runtime toggle for GPU acceleration. Analogous to {@link com.yishape.lab.math.compute.hpc.HpcSwitch}.
 */
public final class GpuSwitch {
    private static volatile boolean enabled = true;

    private GpuSwitch() {}

    public static void enable() { enabled = true; }
    public static void disable() { enabled = false; }
    public static boolean toggle() { enabled = !enabled; return enabled; }
    public static boolean isEnabled() { return enabled; }

    public static void runWith(boolean state, Runnable task) {
        synchronized (GpuSwitch.class) {
            boolean prev = enabled;
            try {
                enabled = state;
                task.run();
            } finally {
                enabled = prev;
            }
        }
    }
}
