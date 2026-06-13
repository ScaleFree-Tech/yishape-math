package com.yishape.lab.math.compute.hpc;

/**
 * HPC-accelerated cross product (a × b for last-dim=3 vectors).
 * Inputs must be pre-broadcasted to common shape before calling.
 */
public final class HpcCross {

    private HpcCross() {
    }

    /**
     * Attempt HPC cross product forward.
     *
     * @param a           [numTriplets * 3] pre-broadcasted
     * @param b           [numTriplets * 3] pre-broadcasted
     * @param numTriplets number of 3D vector triplets
     * @param output      [numTriplets * 3] (pre-allocated)
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryCrossForward(double[] a, double[] b, int numTriplets, double[] output) {
        if (a == null || b == null || output == null) {
            return false;
        }
        long total = (long) numTriplets * 3;
        if (total < HpcConfig.crossMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try { int rc = com.yishape.lab.math.hpc.YishapeHpc.crossForward(a, b, numTriplets, output); return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK; } catch (Throwable t) { return false; }
    }

    /**
     * Attempt HPC cross product backward. Works on pre-broadcasted shapes.
     * Caller must un-broadcast (sum-reduce) gradA/gradB to original input shapes.
     *
     * @param gradOutput  [numTriplets * 3] upstream gradient
     * @param a           [numTriplets * 3] pre-broadcasted (saved from forward)
     * @param b           [numTriplets * 3] pre-broadcasted (saved from forward)
     * @param numTriplets number of 3D vector triplets
     * @param gradA       [numTriplets * 3] (pre-allocated, will be zeroed)
     * @param gradB       [numTriplets * 3] (pre-allocated, will be zeroed)
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryCrossBackward(double[] gradOutput, double[] a, double[] b,
                                            int numTriplets, double[] gradA, double[] gradB) {
        if (gradOutput == null || a == null || b == null || gradA == null || gradB == null) {
            return false;
        }
        long total = (long) numTriplets * 3;
        if (total < HpcConfig.crossMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try { int rc = com.yishape.lab.math.hpc.YishapeHpc.crossBackward(
                gradOutput, a, b, numTriplets, gradA, gradB); return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK; } catch (Throwable t) { return false; }
    }
}
