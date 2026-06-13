package com.yishape.lab.math.compute.hpc;

/**
 * HPC-accelerated trapezoidal selective scan (Mamba SSM core).
 */
public final class HpcTrapezoidalScan {

    private HpcTrapezoidalScan() {
    }

    /**
     * Attempt HPC trapezoidal scan forward.
     *
     * @param u              [B * L * D] input sequence
     * @param delta          [B * L * D] or [B * D] time deltas
     * @param a              [D] or [B * D] state transition params
     * @param b              [B * L * D] input projection
     * @param c              [B * L * D] output projection
     * @param d              [1] or [D] skip connection
     * @param aIsVec         1 if A has shape [D], 0 if [B, D]
     * @param dIsScalar      1 if D is scalar [1], 0 if [D]
     * @param deltaBroadcast 1 if delta has shape [B, 1, D], 0 if [B, L, D]
     * @param output         [B * L * D] (pre-allocated)
     * @param savedH         [B * L * D] hidden states for backward (pre-allocated)
     * @param savedABar      [B * L * D] a_bar for backward (pre-allocated)
     * @param savedBBarU     [B * L * D] b_bar*u for backward (pre-allocated)
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryTrapezoidalScanForward(
            double[] u, double[] delta, double[] a, double[] b, double[] c, double[] d,
            int B, int L, int D, int aIsVec, int dIsScalar, int deltaBroadcast,
            double[] output, double[] savedH, double[] savedABar, double[] savedBBarU) {
        if (u == null || delta == null || a == null || b == null || c == null || d == null
                || output == null || savedH == null || savedABar == null || savedBBarU == null) {
            return false;
        }
        long total = (long) B * L * D;
        if (total < HpcConfig.trapezoidalScanMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try { int rc = com.yishape.lab.math.hpc.YishapeHpc.trapezoidalScanForward(
                u, delta, a, b, c, d, B, L, D, aIsVec, dIsScalar, deltaBroadcast,
                output, savedH, savedABar, savedBBarU); return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK; } catch (Throwable t) { return false; }
    }

    /**
     * Attempt HPC trapezoidal scan backward.
     *
     * @param gradOutput     [B * L * D] upstream gradient
     * @param u              [B * L * D] forward input (saved)
     * @param delta          [B * L * D] or [B * D] forward delta (saved)
     * @param a              [D] or [B * D] forward A (saved)
     * @param b              [B * L * D] forward B (saved)
     * @param c              [B * L * D] forward C (saved)
     * @param d              [1] or [D] forward D (saved)
     * @param savedH         [B * L * D] hidden states from forward
     * @param savedABar      [B * L * D] a_bar from forward
     * @param savedBBarU     [B * L * D] b_bar*u from forward
     * @param gradU          [B * L * D] (pre-allocated)
     * @param gradDelta      [B * L * D] or [B * D] (pre-allocated)
     * @param gradA          [D] or [B * D] (pre-allocated)
     * @param gradB          [B * L * D] (pre-allocated)
     * @param gradC          [B * L * D] (pre-allocated)
     * @param gradD          [1] or [D] (pre-allocated)
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryTrapezoidalScanBackward(
            double[] gradOutput, double[] u, double[] delta, double[] a, double[] b,
            double[] c, double[] d, double[] savedH, double[] savedABar, double[] savedBBarU,
            int B, int L, int D, int aIsVec, int dIsScalar, int deltaBroadcast,
            double[] gradU, double[] gradDelta, double[] gradA, double[] gradB,
            double[] gradC, double[] gradD) {
        if (gradOutput == null || u == null || delta == null || a == null || b == null
                || c == null || d == null || savedH == null || savedABar == null || savedBBarU == null
                || gradU == null || gradDelta == null || gradA == null || gradB == null
                || gradC == null || gradD == null) {
            return false;
        }
        long total = (long) B * L * D;
        if (total < HpcConfig.trapezoidalScanMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try { int rc = com.yishape.lab.math.hpc.YishapeHpc.trapezoidalScanBackward(
                gradOutput, u, delta, a, b, c, d, savedH, savedABar, savedBBarU,
                B, L, D, aIsVec, dIsScalar, deltaBroadcast,
                gradU, gradDelta, gradA, gradB, gradC, gradD); return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK; } catch (Throwable t) { return false; }
    }
}
