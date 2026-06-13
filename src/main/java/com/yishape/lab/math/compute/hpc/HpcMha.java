package com.yishape.lab.math.compute.hpc;

/**
 * HPC-accelerated multi-head attention for inference.
 * Operates on pre-projected Q, K, V arrays (no QKV projection or output projection).
 */
public final class HpcMha {

    private HpcMha() {
    }

    /**
     * Attempt HPC multi-head attention forward.
     *
     * @param q         [seqLen × dModel] Q from QKV projection (row-major flat)
     * @param k         [seqLen × kvSize] K from QKV projection (kvSize = numKvHeads × headDim)
     * @param v         [seqLen × kvSize] V from QKV projection
     * @param seqLen    sequence length
     * @param dModel    model dimension (numHeads × headDim)
     * @param numHeads  number of query heads
     * @param numKvHeads number of key/value heads (== numHeads for standard MHA)
     * @param causal    true to apply causal mask
     * @param output    [seqLen × dModel] attention output (pre-allocated)
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryAttention(double[] q, double[] k, double[] v,
                                        int seqLen, int dModel, int numHeads, int numKvHeads,
                                        boolean causal, double[] output) {
        if (q == null || k == null || v == null || output == null) {
            return false;
        }
        long total = (long) seqLen * dModel;
        if (total < HpcConfig.mhaMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try { int rc = com.yishape.lab.math.hpc.YishapeHpc.mhaAttention(
                q, k, v, seqLen, dModel, numHeads, numKvHeads, causal ? 1 : 0, output); return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK; } catch (Throwable t) { return false; }
    }
}
