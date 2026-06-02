package com.yishape.lab.math.compute.hpc;

/**
 * HPC-accelerated Mamba selective scan for SSM sequence modeling.
 * Follows the same four-gate optional-native pattern as {@link HpcIm2col}.
 */
public final class HpcMamba {

    private HpcMamba() {
    }

    /**
     * Attempt HPC selective scan forward.
     *
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean trySelectiveScan(
            double[] input, double[] dt, double[] aLog, double[] B, double[] C,
            int L, int innerDim, int N, double[] output, double[] hCache) {
        if (input == null || dt == null || aLog == null || B == null || C == null
                || output == null || hCache == null) {
            return false;
        }
        long total = (long) L * innerDim;
        if (total < HpcConfig.mambaMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        int rc = com.yishape.lab.math.hpc.YishapeHpc.mambaSelectiveScan(
                input, dt, aLog, B, C, L, innerDim, N, output, hCache);
        return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK;
    }

    /**
     * Attempt HPC selective scan backward.
     *
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryScanBackward(
            double[] gradOutput, double[] input, double[] dt, double[] aLog,
            double[] B, double[] C, double[] hCache,
            int L, int innerDim, int N,
            double[] gradInput, double[] gradDt, double[] gradALog,
            double[] gradB, double[] gradC) {
        if (gradOutput == null || input == null || dt == null || aLog == null
                || B == null || C == null || hCache == null
                || gradInput == null || gradDt == null || gradALog == null
                || gradB == null || gradC == null) {
            return false;
        }
        long total = (long) L * innerDim;
        if (total < HpcConfig.mambaMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        int rc = com.yishape.lab.math.hpc.YishapeHpc.mambaScanBackward(
                gradOutput, input, dt, aLog, B, C, hCache,
                L, innerDim, N, gradInput, gradDt, gradALog, gradB, gradC);
        return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK;
    }
}
