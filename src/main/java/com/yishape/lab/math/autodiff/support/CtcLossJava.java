package com.yishape.lab.math.autodiff.support;

/**
 * Pure-Java (SISD) CTC Loss forward-backward implementation.
 * Used as fallback when HPC native runtime is unavailable.
 *
 * <p>Implements the standard CTC algorithm (Graves et al., 2006):
 * forward-backward dynamic programming over extended label sequence
 * with blank-augmented state space. The loops here are structural DP
 * iterations over the (T × L') state lattice with conditional transitions;
 * each state depends on 2-3 predecessor/successor states. These are NOT
 * element-wise numerical loops — they cannot be vectorized via
 * DoubleVectorComputer or equivalent.</p>
 *
 * <p>Input {@code logProbs} must be log-probabilities (e.g. after log-softmax).
 * Gradients returned in {@code grad} are w.r.t. the log-probability inputs.</p>
 */
public final class CtcLossJava {

    private CtcLossJava() { /* utility class */ }

    /** Clamp threshold to prevent log(0) and division by zero. */
    private static final double CTC_EPS = 1e-300;

    /**
     * CTC forward-backward on a single batch element.
     *
     * @param logProbs  [T * C] flat row-major log-probabilities (logProbs[t*C + c])
     * @param labels    [labelLen] integer class labels (1-based, blank = 0)
     * @param labelLen  number of labels in {@code labels}
     * @param T         number of time steps
     * @param C         number of classes (including blank, which is class 0)
     * @param loss      [1] output: negative log-likelihood of the label sequence
     * @param grad      [T * C] output: gradient of loss w.r.t. logProbs (same layout)
     * @return true on success, false if inputs are invalid
     */
    public static boolean tryForwardBackward(
            double[] logProbs, int[] labels, int labelLen,
            int T, int C, double[] loss, double[] grad) {

        // --- input validation ---
        if (logProbs == null || labels == null || loss == null || grad == null) {
            return false;
        }
        if (T <= 0 || C <= 0 || labelLen <= 0 || labelLen > T) {
            return false;
        }
        int tc = T * C;
        if (logProbs.length < tc || loss.length < 1 || grad.length < tc) {
            return false;
        }

        // --- extended labels: [blank, l0, blank, l1, blank, l2, ...] ---
        int lpLen = 2 * labelLen + 1;
        int[] ext = new int[lpLen];
        ext[0] = 0;
        for (int i = 0; i < labelLen; i++) {
            ext[2 * i + 1] = labels[i];
            ext[2 * i + 2] = 0;
        }

        // --- y = exp(logProbs), pre-computed once ---
        // Structural: per-element exp is unavoidable for the DP table setup;
        // GpuActivation.tryExp could be used here but the overhead of the GPU
        // dispatch dominates for typical CTC batch sizes. Each batch element
        // undergoes this only once.
        double[] y = new double[tc];
        for (int i = 0; i < tc; i++) {
            y[i] = Math.exp(logProbs[i]);
        }

        // --- forward (α) ---
        // Structural DP: α[t][s] depends on α[t-1][s], α[t-1][s-1], α[t-1][s-2]
        // with conditional skip logic. Cannot be vectorized.
        double[] alpha = new double[T * lpLen];

        // Initialize: α[0][ext[0]] = y[0][ext[0]]
        alpha[ext[0]] = y[ext[0]];
        if (lpLen > 1) {
            alpha[lpLen + ext[1]] = y[ext[1]]; // α[0][ext[1]]
        }

        for (int t = 1; t < T; t++) {
            int off = t * lpLen;
            int offPrev = (t - 1) * lpLen;
            int yOff = t * C;
            for (int s = 0; s < lpLen; s++) {
                double sum = alpha[offPrev + s];               // stay
                if (s >= 1) {
                    sum += alpha[offPrev + s - 1];              // advance one
                }
                if (s >= 2 && ext[s] != ext[s - 2]) {
                    sum += alpha[offPrev + s - 2];              // skip (only if labels differ)
                }
                alpha[off + s] = sum * y[yOff + ext[s]];
            }
        }

        // --- backward (β) ---
        // Structural DP: β[t][s] depends on β[t+1][s], β[t+1][s+1], β[t+1][s+2]
        // with conditional skip logic. Cannot be vectorized.
        double[] beta = new double[T * lpLen];

        // Initialize: β[T-1][s] = y[T-1][ext[s]]
        int offLast = (T - 1) * lpLen;
        int yOffLast = (T - 1) * C;
        for (int s = 0; s < lpLen; s++) {
            beta[offLast + s] = y[yOffLast + ext[s]];
        }

        for (int t = T - 2; t >= 0; t--) {
            int off = t * lpLen;
            int offNext = (t + 1) * lpLen;
            int yOff = t * C;
            for (int s = 0; s < lpLen; s++) {
                double sum = beta[offNext + s];                  // stay
                if (s + 1 < lpLen) {
                    sum += beta[offNext + s + 1];                // advance one
                }
                if (s + 2 < lpLen && ext[s] != ext[s + 2]) {
                    sum += beta[offNext + s + 2];                // skip (only if labels differ)
                }
                beta[off + s] = sum * y[yOff + ext[s]];
            }
        }

        // --- loss = -ln(Z), Z = sum_s α[T-1][s] ---
        double z = 0;
        for (int s = 0; s < lpLen; s++) {
            z += alpha[offLast + s];
        }
        double zClamped = Math.max(z, CTC_EPS);
        loss[0] = -Math.log(zClamped);

        // --- gradient ---
        // Input is log-probabilities x[t][k]; y[t][k] = exp(x[t][k]).
        // ∂L/∂x[t][k] = -(1/Z) * Σ_{s:ext[s]=k} α[t][s]·β[t][s] / y[t][k]
        // = -sum_ab / (Z * y[t][k])
        // Structural: per-class accumulation of α·β products across the state lattice.
        double invZ = 1.0 / zClamped;
        for (int t = 0; t < T; t++) {
            int off = t * lpLen;
            int yOff = t * C;
            int gradOff = t * C;
            for (int k = 0; k < C; k++) {
                double sumAb = 0;
                for (int s = 0; s < lpLen; s++) {
                    if (ext[s] == k) {
                        sumAb += alpha[off + s] * beta[off + s];
                    }
                }
                double yVal = y[yOff + k];
                grad[gradOff + k] = -sumAb * invZ / Math.max(yVal, CTC_EPS);
            }
        }

        return true;
    }
}
