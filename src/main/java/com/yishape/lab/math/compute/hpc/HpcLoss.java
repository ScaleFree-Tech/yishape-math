package com.yishape.lab.math.compute.hpc;

/**
 * HPC-accelerated loss functions (CTC, etc.).
 * Attempts the native HPC call only — returns boolean.
 * Java fallback is handled by the caller (DiffTensorDecomp).
 */
public final class HpcLoss {

    private HpcLoss() {
    }

    /**
     * Attempt HPC fused CTC forward + backward.
     *
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryCtcForwardBackward(double[] logProbs, int[] labels, int labelLen,
                                                int T, int C, double[] loss, double[] grad) {
        if (logProbs == null || labels == null || loss == null || grad == null) {
            return false;
        }
        long total = (long) T * C;
        if (total < HpcConfig.ctcMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try {
            int rc = com.yishape.lab.math.hpc.YishapeHpc.ctcForwardBackward(
                    logProbs, labels, labelLen, T, C, loss, grad);
            return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK;
        } catch (Throwable t) {
            return false;
        }
    }
}
