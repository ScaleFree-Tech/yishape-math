package com.yishape.lab.math.linalg.decomposition;

/**
 * LAPACK-style decomposition error estimate.
 *
 * <h3>Definitions (LAPACK convention):</h3>
 * <ul>
 *   <li><b>Backward error</b>: How close the computed decomposition is to an exact
 *       decomposition of a nearby matrix. For SVD: {@code ||A - U*S*V^T||_F / ||A||_F}</li>
 *   <li><b>Forward error bound</b>: Upper bound on the error in eigenvalues/singular
 *       values: {@code backward_error * condition_number}</li>
 *   <li><b>Relative residual</b>: For solve-based contexts:
 *       {@code ||A*x - b|| / (||A|| * ||x|| + ||b||)}</li>
 * </ul>
 *
 * @param backwardError       ||reconstructed - original|| / ||original|| (Frobenius norm)
 * @param forwardErrorBound   worst-case upper bound: backward_error * condition_number
 * @param relativeResidual    residual norm (solver context; 0 if not applicable)
 * @param conditionNumberUsed the condition number used to compute the forward bound
 * @param computed            whether the estimate was actually computed
 */
public record DecompositionErrorEstimate(
    double backwardError,
    double forwardErrorBound,
    double relativeResidual,
    double conditionNumberUsed,
    boolean computed
) {
    /** Sentinel: no estimate available. */
    public static final DecompositionErrorEstimate NOT_AVAILABLE =
        new DecompositionErrorEstimate(Double.NaN, Double.NaN, Double.NaN, Double.NaN, false);

    @Override
    public String toString() {
        if (!computed) {
            return "DecompositionErrorEstimate[not available]";
        }
        return String.format(
            "backward=%.2e forward=%.2e residual=%.2e cond=%.2e",
            backwardError, forwardErrorBound, relativeResidual, conditionNumberUsed);
    }
}
