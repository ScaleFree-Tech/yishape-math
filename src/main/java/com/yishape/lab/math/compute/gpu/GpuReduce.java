package com.yishape.lab.math.compute.gpu;

/**
 * GPU reduce dispatch: sum, mean, max, min, prod, variance.
 * STANDARD_DEVIATION is computed on the Java side as sqrt(variance).
 */
public final class GpuReduce {
    public static final int SUM = 0;
    public static final int MEAN = 1;
    public static final int MAX = 2;
    public static final int MIN = 3;
    public static final int PROD = 4;
    public static final int VARIANCE = 5;

    private GpuReduce() {}

    public static double[] tryReduce(int op, double[] input, int outer, int inner) {
        if (input == null || outer <= 0 || inner <= 0) return null;
        if ((long) outer * inner < GpuConfig.reduceMinElements()) return null;
        if (!GpuConfig.allowAttempts()) return null;
        if (!GpuOptionalRuntime.isGpuAvailable()) return null;
        try {
            return GpuOptionalRuntime.tryReduce(op, input, outer, inner);
        } catch (Throwable t) {
            return null;
        }
    }
}
