package com.yishape.lab.math.linalg.decomposition.impl.support;

import com.yishape.lab.math.util.RerePrecision;

/**
 * Maps a matrix condition number estimate to a recommended epsilon
 * for numerical comparisons: zero-tests, rank determination, and
 * convergence criteria.
 *
 * <p>LAPACK guideline: tol = ||A|| * epsilon_machine * max(m,n) * cond_estimate.
 * In practice we simplify to a discretized mapping for robustness:</p>
 * <ul>
 *   <li>cond &le; 10: epsilon = 1e-14 (well-conditioned, high precision)</li>
 *   <li>cond &le; 10^4: epsilon = 1e-12 (the historical safe default)</li>
 *   <li>cond &le; 10^8: epsilon = 1e-10 (moderately ill-conditioned)</li>
 *   <li>cond &gt; 10^8: epsilon = 1e-8 (very ill-conditioned)</li>
 * </ul>
 */
public final class AdaptiveEpsilon {

    private static final double EPS_WELL = 1e-14;
    private static final double EPS_MODERATE = 1e-12;
    private static final double EPS_ILL = 1e-10;
    private static final double EPS_VERY_ILL = 1e-8;

    /** Minimum matrix dimension for Hager estimation (smaller uses heuristic). */
    private static final int MIN_DIM_FOR_HAGER = 100;

    private AdaptiveEpsilon() {}

    /**
     * Recommended epsilon for a matrix given its estimated 1-norm condition number.
     *
     * @param condEstimate estimated condition number (e.g. from Hager estimator)
     * @return recommended epsilon for zero-comparison
     */
    public static double epsilonForCondition(double condEstimate) {
        if (condEstimate <= 10.0) {
            return EPS_WELL;
        } else if (condEstimate <= 1e4) {
            return EPS_MODERATE;
        } else if (condEstimate <= 1e8) {
            return EPS_ILL;
        } else {
            return EPS_VERY_ILL;
        }
    }

    /**
     * Quick heuristic: returns epsilon based on matrix scale alone.
     * Used when condition estimation is unavailable.
     *
     * @param frobeniusNorm ||A||_F as a rough scale indicator
     * @param maxDim        max(rows, cols)
     * @return heuristic epsilon
     */
    public static double heuristicEpsilon(double frobeniusNorm, int maxDim) {
        double machineEps = RerePrecision.getMachineEpsilon();
        // Scale epsilon by matrix norm and dimension
        double tol = machineEps * Math.max(1.0, frobeniusNorm) * maxDim;
        // Clamp to reasonable range
        if (tol < EPS_WELL) return EPS_WELL;
        if (tol > EPS_VERY_ILL) return EPS_VERY_ILL;
        return tol;
    }

    /**
     * Convenience: runs Hager estimator on the given raw data array,
     * then returns epsilonForCondition(). Falls back to heuristicEpsilon()
     * for small matrices where Hager overhead is not worth it.
     *
     * @param data square double[][] matrix data (dense row-major)
     * @return recommended epsilon
     */
    public static double computeAdaptiveEpsilon(double[][] data) {
        int n = data.length;
        if (n < MIN_DIM_FOR_HAGER) {
            double frobNorm = frobeniusNorm(data);
            return heuristicEpsilon(frobNorm, n);
        }
        double cond = HagerConditionEstimator.estimateCondition(data);
        if (Double.isInfinite(cond) || cond > 1e16) {
            return EPS_VERY_ILL;
        }
        return epsilonForCondition(cond);
    }

    /** Compute Frobenius norm of a double[][]. */
    private static double frobeniusNorm(double[][] A) {
        double sumSq = 0.0;
        for (double[] row : A) {
            for (double v : row) {
                sumSq += v * v;
            }
        }
        return Math.sqrt(sumSq);
    }
}
