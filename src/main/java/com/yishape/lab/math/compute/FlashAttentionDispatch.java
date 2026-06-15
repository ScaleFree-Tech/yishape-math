package com.yishape.lab.math.compute;

/**
 * Flash Attention dispatch: HPC native → caller fallback.
 *
 * <p>Implements the tiled online-softmax attention algorithm (Dao et al., 2022).
 * When native HPC kernels are available, all heads are processed in a single native call.
 * On fallback, returns {@code null} — the caller should iterate per-head using
 * the Java tiled implementation.</p>
 *
 * <p>All arrays are row-major flat. Q is {@code [seqLen, numHeads * headDim]},
 * K and V are {@code [seqLen, numKVHeads * headDim]} (GQA-compatible).</p>
 *
 * <p>GPU dispatch is handled at the yishape-dl level via {@code MultiheadAttention}.</p>
 */
public final class FlashAttentionDispatch {

    private FlashAttentionDispatch() {}

    /** Check whether HPC native Flash Attention backend is available. */
    public static boolean isNativeAvailable() {
        try {
            return com.yishape.lab.math.hpc.YishapeHpc.isFlashAttentionAvailable();
        } catch (LinkageError | RuntimeException e) {
            return false;
        }
    }

    // ---- Forward result ----

    public record NativeFlashResult(
        double[] O,    // output [seqLen * numHeads * headDim]
        double[] L,    // row-wise normalizer [numHeads * seqLen]
        double[] M     // row-wise max [numHeads * seqLen]
    ) {}

    /**
     * Flash Attention forward pass over all heads via HPC native backend.
     *
     * @param Q         query, row-major [seqLen * (numHeads * headDim)]
     * @param K         key, row-major [seqLen * (numKVHeads * headDim)]
     * @param V         value, row-major [seqLen * (numKVHeads * headDim)]
     * @param seqLen    sequence length
     * @param numHeads  number of query heads
     * @param numKVHeads number of key/value heads (&le; numHeads for GQA)
     * @param headDim   dimension per head
     * @param causal    whether to apply causal mask
     * @return native flash result, or null if HPC backend unavailable/failed
     */
    public static NativeFlashResult flashForward(
            double[] Q, double[] K, double[] V,
            int seqLen, int numHeads, int numKVHeads, int headDim,
            boolean causal) {

        try {
            if (!com.yishape.lab.math.hpc.YishapeHpc.isFlashAttentionAvailable()) return null;
            int dModel = numHeads * headDim;
            double[] O = new double[seqLen * dModel];
            double[] L = new double[numHeads * seqLen];
            double[] M = new double[numHeads * seqLen];

            int rc = com.yishape.lab.math.hpc.YishapeHpc.flashAttnFwd(
                Q, K, V, O, L, M, seqLen, dModel, numHeads, numKVHeads, causal ? 1 : 0);
            if (rc != com.yishape.lab.math.hpc.YishapeHpcStatus.OK) return null;
            return new NativeFlashResult(O, L, M);
        } catch (LinkageError | RuntimeException e) {
            return null;
        }
    }

    /**
     * Flash Attention backward pass over all heads via HPC native backend.
     *
     * @param dO        upstream gradient [seqLen * (numHeads * headDim)]
     * @param Q, K, V   forward inputs (same shapes as forward)
     * @param L, M      forward row-wise normalizer and max (from {@link NativeFlashResult})
     * @return gradients {dQ, dK, dV}, or null if HPC backend unavailable/failed
     */
    public static double[][] flashBackward(
            double[] dO, double[] Q, double[] K, double[] V,
            double[] L, double[] M,
            int seqLen, int numHeads, int numKVHeads, int headDim,
            boolean causal) {

        try {
            if (!com.yishape.lab.math.hpc.YishapeHpc.isFlashAttentionAvailable()) return null;
            int dModel = numHeads * headDim;
            int kvDim = numKVHeads * headDim;
            double[] O = new double[seqLen * dModel]; // forward output placeholder
            double[] dQ = new double[seqLen * dModel];
            double[] dK = new double[seqLen * kvDim];
            double[] dV = new double[seqLen * kvDim];

            int rc = com.yishape.lab.math.hpc.YishapeHpc.flashAttnBwd(
                Q, K, V, O, dO, L, M, dQ, dK, dV,
                seqLen, dModel, numHeads, numKVHeads, causal ? 1 : 0);
            if (rc != com.yishape.lab.math.hpc.YishapeHpcStatus.OK) return null;
            return new double[][]{dQ, dK, dV};
        } catch (LinkageError | RuntimeException e) {
            return null;
        }
    }
}
