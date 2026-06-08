package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.ISchurDecomposition;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.solver.SchurDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.NonSquareMatrixException;
import com.yishape.lab.math.linalg.decomposition.DecompositionFailedException;
import com.yishape.lab.util.Tuple2;

/**
 * Implementation of Schur decomposition using the Francis QR algorithm with double shift.
 * <p>
 * This implementation computes the real Schur decomposition of a square matrix,
 * which decomposes the matrix A as: A = U T U^T where U is orthogonal and T
 * is quasi-upper triangular (block upper triangular with 1x1 and 2x2 blocks).
 * </p>
 * 
 * <p>
 * The algorithm uses the Francis QR step with double shift for better convergence:
 * <ol>
 *   <li>Reduce matrix to upper Hessenberg form</li>
 *   <li>Apply Francis QR steps with double shift until convergence</li>
 *   <li>Accumulate orthogonal transformations to form U</li>
 * </ol>
 * </p>
 * 
 * <h3>References</h3>
 * <ul>
 *   <li>Golub, G. H., &amp; Van Loan, C. F. (2013). Matrix computations (4th ed.). Johns Hopkins University Press.</li>
 *   <li>Hessenberg, K. (1940). Behandlung linearer Eigenwertaufgaben mit Hilfe der Hamilton-Cayleyschen Gleichung.</li>
 *   <li>Francis, J. G. F. (1961). The QR Transformation A Unitary Analogue to the LR Transformation—Part 1.</li>
 *   <li>Watkins, D. S. (2007). The matrix eigenvalue problem: GR and Krylov subspace methods. SIAM.</li>
 * </ul>
 *
 * @author RereMouse
 * @version 1.0
 * @since 2.0
 */
public class RereSchurDecomposition implements ISchurDecomposition {
    
    /** Cached value of U. */
    private IMatrix<Double> cachedU;
    /** Cached value of T. */
    private IMatrix<Double> cachedT;
    /** Cached value of UT. */
    private IMatrix<Double> cachedUT;
    /** The matrix data in Hessenberg form. */
    private double[][] hData;
    /** The orthogonal transformation matrix. */
    private double[][] uData;
    /** Epsilon for numerical comparisons. */
    private double epsilon;
    /** Maximum number of iterations for the QR algorithm. */
    private int maxIterations = 1000;
    
    /**
     * Default constructor with default epsilon and maxIterations.
     */
    public RereSchurDecomposition() {
        this.epsilon = DEFAULT_EPSILON;
        this.maxIterations = DEFAULT_MAX_ITERATIONS;
    }

    /**
     * Constructor with configurable epsilon.
     *
     * @param epsilon threshold for considering an element as zero
     */
    public RereSchurDecomposition(double epsilon) {
        this.epsilon = epsilon;
        this.maxIterations = DEFAULT_MAX_ITERATIONS;
    }

    /**
     * Constructor with configurable epsilon and max iterations.
     *
     * @param epsilon       threshold for considering an element as zero
     * @param maxIterations maximum number of QR iterations
     */
    public RereSchurDecomposition(double epsilon, int maxIterations) {
        this.epsilon = epsilon;
        this.maxIterations = maxIterations;
    }
    
    @Override
    public Tuple2<IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix) {
        return decompose(matrix, epsilon);
    }
    
    @Override
    public Tuple2<IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix, double epsilon) {
        // Reset cached values
        cachedU = null;
        cachedT = null;
        cachedUT = null;
        this.epsilon = epsilon;
        
        // Get matrix dimensions
        int rows = matrix.rows();
        int cols = matrix.cols();
        
        if (rows != cols) {
            throw new NonSquareMatrixException(
                "Only square matrices can be decomposed using Schur decomposition",
                "Schur Decomposition", 
                "Matrix " + rows + "x" + cols,
                rows, cols);
        }
        
        int n = rows;
        
        // Copy data
        hData = matrix.toDoubleArray();
        uData = new double[n][n];
        
        // Initialize U as identity matrix
        for (int i = 0; i < n; i++) {
            uData[i][i] = 1.0;
        }
        
        // Reduce to upper Hessenberg form
        reduceToHessenberg();
        
        // Apply Francis QR algorithm with double shift
        francisQRStep();
        
        // Create result matrices
        cachedT = Linalg.matrix(hData);
        cachedU = Linalg.matrix(uData);
        cachedUT = cachedU.transpose();
        
        return new Tuple2<>(cachedU, cachedT);
    }
    
    /**
     * Reduce matrix to upper Hessenberg form using Householder reflections.
     */
    private void reduceToHessenberg() {
        int n = hData.length;
        double[] colBuf = new double[n];
        // Pre-allocated Householder vector buffer reused across iterations.
        double[] h = new double[n];

        for (int k = 0; k < n - 2; k++) {
            int len = n - k - 1;

            // Extract column k into h and compute norm
            double norm = 0.0;
            for (int i = 0; i < len; i++) {
                double val = hData[k + 1 + i][k];
                h[i] = val;
                norm += val * val;
            }

            if (norm > epsilon) {
                norm = Math.sqrt(norm);
                if (h[0] != 0) {
                    norm *= Math.signum(h[0]);
                }

                // Normalize the Householder vector
                h[0] += norm;
                double hNorm = 0.0;
                for (int i = 0; i < len; i++) {
                    hNorm += h[i] * h[i];
                }
                hNorm = Math.sqrt(hNorm);

                if (hNorm > epsilon) {
                    for (int i = 0; i < len; i++) {
                        h[i] /= hNorm;
                    }

                    // Apply the Householder reflection from the right
                    for (int j = k + 1; j < n; j++) {
                        double dotProduct = 0.0;
                        for (int i = 0; i < len; i++) {
                            dotProduct += hData[j][k + 1 + i] * h[i];
                        }
                        for (int i = 0; i < len; i++) {
                            hData[j][k + 1 + i] -= 2.0 * dotProduct * h[i];
                        }
                    }

                    // Apply the Householder reflection from the left
                    // Column extraction into colBuf: hData[j][i] with inner j varying
                    // is stride-n.  Buffer in colBuf for stride-1 arithmetic.
                    for (int i = 0; i < n; i++) {
                        double dotProduct = 0.0;
                        for (int j = 0; j < len; j++) {
                            double val = hData[k + 1 + j][i];
                            colBuf[j] = val;
                            dotProduct += val * h[j];
                        }
                        double twoDot = 2.0 * dotProduct;
                        for (int j = 0; j < len; j++) {
                            hData[k + 1 + j][i] = colBuf[j] - twoDot * h[j];
                        }
                    }

                    // Accumulate the transformation in U
                    for (int i = 0; i < n; i++) {
                        double dotProduct = 0.0;
                        for (int j = 0; j < len; j++) {
                            dotProduct += uData[i][k + 1 + j] * h[j];
                        }
                        for (int j = 0; j < len; j++) {
                            uData[i][k + 1 + j] -= 2.0 * dotProduct * h[j];
                        }
                    }
                }
            }
        }
    }
    
    /**
     * Apply Francis QR step with double shift and proper deflation.
     *
     * <p>The algorithm maintains an active subproblem [0, p] that shrinks as
     * subdiagonal elements converge to zero (deflation). Only the active
     * subproblem participates in QR steps, reducing per-iteration work from
     * O(n²) to O(p²).</p>
     *
     * <p>Convergence tolerance uses scaled comparison:
     * |h[i][i-1]| &le; &epsilon; &times; (|h[i-1][i-1]| + |h[i][i]|)</p>
     *
     * <p>Exceptional shifts are applied every 10 iterations per subproblem
     * to break convergence cycles.</p>
     */
    // GUARD: Deflation must shrink p from the bottom as subdiagonals converge.
    // The QR step operates only on [q, p] where q is found by scanning for a
    // negligible subdiagonal from p downward. Exceptional shifts at iter%10==0
    // break cycles. Scaled tolerance avoids mistaking large-magnitude entries
    // as converged. See: decomposition-pitfalls.md § SchurDeflation
    private void francisQRStep() {
        int n = hData.length;
        int totalIter = 0;
        int p = n - 1;

        while (p > 0 && totalIter < maxIterations) {
            // Deflate from bottom: zero out negligible subdiagonals at position p
            while (p > 0) {
                double tol = epsilon * (Math.abs(hData[p - 1][p - 1]) + Math.abs(hData[p][p]));
                if (Math.abs(hData[p][p - 1]) > tol) {
                    break;
                }
                hData[p][p - 1] = 0.0;
                p--;
            }
            if (p <= 0) break;

            // Find the top of the lowest active subproblem [q, p]
            // Scan downward from p to find a negligible subdiagonal
            int q = p;
            while (q > 0) {
                double tol = epsilon * (Math.abs(hData[q - 1][q - 1]) + Math.abs(hData[q][q]));
                if (Math.abs(hData[q][q - 1]) <= tol) {
                    hData[q][q - 1] = 0.0;
                    break;
                }
                q--;
            }
            // Subproblem is [q, p]; q can be 0 if no internal split found

            // Apply Francis QR steps to subproblem [q, p]
            int subIter = 0;
            while (subIter < 100 && totalIter < maxIterations) {
                // Check for deflation at the bottom of the subproblem
                double tol = epsilon * (Math.abs(hData[p - 1][p - 1]) + Math.abs(hData[p][p]));
                if (Math.abs(hData[p][p - 1]) <= tol) {
                    hData[p][p - 1] = 0.0;
                    p--;
                    break;
                }

                // Compute double shift from trailing 2x2 of subproblem [q, p]
                double s, t;
                if (subIter > 0 && subIter % 10 == 0) {
                    // Exceptional shift: use magnitude of subdiagonals to break cycles
                    double excShift = Math.abs(hData[p][p - 1]);
                    if (p > q) {
                        excShift += Math.abs(hData[p - 1][p - 2]);
                    }
                    s = excShift;
                    t = excShift;
                } else {
                    double hpp = hData[p][p];
                    double hpm = hData[p][p - 1];
                    double hmp = hData[p - 1][p];
                    double hmm = hData[p - 1][p - 1];
                    double tr = hmm + hpp;
                    double det = hmm * hpp - hpm * hmp;
                    double disc = tr * tr - 4.0 * det;
                    if (disc >= 0) {
                        double sqrtDisc = Math.sqrt(disc);
                        s = 0.5 * (tr + sqrtDisc);
                        t = 0.5 * (tr - sqrtDisc);
                    } else {
                        s = 0.5 * tr;
                        t = s;
                    }
                }

                qrStepWithShift(s, t, q, p);
                subIter++;
                totalIter++;
            }
        }

        if (totalIter >= maxIterations) {
            throw new DecompositionFailedException(
                "Schur decomposition failed to converge after " + maxIterations + " iterations",
                "Schur Decomposition",
                "Matrix " + n + "x" + n,
                totalIter,
                epsilon);
        }
    }

    /**
     * Apply a Francis QR step with double shift to the Hessenberg submatrix H[start..end].
     *
     * <p>Chases the bulge introduced by (H - sI)(H - tI) from position start down
     * through the subdiagonal to position end-1.</p>
     *
     * @param s     first shift value
     * @param t     second shift value
     * @param start first index of the active subproblem (inclusive)
     * @param end   last index of the active subproblem (inclusive)
     */
    private void qrStepWithShift(double s, double t, int start, int end) {
        int n = hData.length;

        // First column of (H - sI)(H - tI) restricted to subproblem [start, end]
        double x = hData[start][start] * hData[start][start]
                 + hData[start][start + 1] * hData[start + 1][start]
                 - (s + t) * hData[start][start] + s * t;
        double y = hData[start + 1][start] * (hData[start][start] + hData[start + 1][start + 1] - s - t);
        // z is only used for the initial bulge computation; not needed after first rotation

        for (int k = start; k < end; k++) {
            double r = Math.hypot(x, y);
            if (r < epsilon) {
                // Near-zero rotation: skip and grab next bulge values
                x = hData[k + 1][k];
                y = (k + 2 <= end) ? hData[k + 2][k] : 0.0;
                continue;
            }

            double c = x / r;
            double sd = y / r;

            // Left multiply G * H: rotate rows k and k+1, columns k..end
            for (int j = k; j <= end; j++) {
                double tmp = c * hData[k][j] + sd * hData[k + 1][j];
                hData[k + 1][j] = -sd * hData[k][j] + c * hData[k + 1][j];
                hData[k][j] = tmp;
            }

            // Right multiply H * G^T: rotate columns k and k+1, rows 0..min(k+2,end)
            // (Hessenberg structure: rows beyond k+2 are zero in columns k, k+1)
            int rightEnd = Math.min(k + 2, end);
            for (int i = 0; i <= rightEnd; i++) {
                double tmp = c * hData[i][k] + sd * hData[i][k + 1];
                hData[i][k + 1] = -sd * hData[i][k] + c * hData[i][k + 1];
                hData[i][k] = tmp;
            }

            // Accumulate U * G^T: rotate columns k and k+1 of U, all rows 0..n-1
            for (int i = 0; i < n; i++) {
                double tmp = c * uData[i][k] + sd * uData[i][k + 1];
                uData[i][k + 1] = -sd * uData[i][k] + c * uData[i][k + 1];
                uData[i][k] = tmp;
            }

            // Chase the bulge: next rotation zeroes hData[k+2][k]
            x = hData[k + 1][k];
            y = (k + 2 <= end) ? hData[k + 2][k] : 0.0;
        }
    }
    
    @Override
    public IMatrix<Double> getU() {
        if (cachedU == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        return cachedU;
    }
    
    @Override
    public IMatrix<Double> getT() {
        if (cachedT == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        return cachedT;
    }
    
    @Override
    public IMatrix<Double> getUT() {
        if (cachedUT == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        return cachedUT;
    }
    
    @Override
    public IDecompositionSolver getSolver() {
        if (hData == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        // Return the standalone solver
        return new SchurDecompositionSolver(Linalg.matrix(hData), Linalg.matrix(uData), epsilon);
    }
    
    @Override
    public double getEpsilon() {
        return epsilon;
    }

    /**
     * Perform Schur decomposition with configurable parameters.
     * 
     * @param matrix The matrix to decompose (must be square)
     * @param epsilon Threshold for considering an element as zero
     * @param maxIterations Maximum number of iterations for iterative algorithms
     * @return A tuple containing U (orthogonal matrix) and T (quasi-upper triangular matrix)
     * @throws NonSquareMatrixException if the matrix is not square
     * @throws DecompositionFailedException if the decomposition fails to converge
     */
    @Override
    public Tuple2<IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix, double epsilon, int maxIterations) {
        this.maxIterations = maxIterations;
        return decompose(matrix, epsilon);
    }

    /**
     * Calculate the determinant of the matrix.
     * 
     * @return determinant of the matrix
     * @throws IllegalStateException if decomposition has not been performed
     */
    @Override
    public double getDeterminant() {
        if (hData == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        
        // For Schur decomposition A = U T U^T, det(A) = det(T)
        // Since T is quasi-upper triangular, its determinant is the product of diagonal elements
        // For 2x2 blocks, we need to compute the determinant of each block
        double det = 1.0;
        int n = hData.length;
        
        for (int i = 0; i < n; i++) {
            if (i < n - 1 && Math.abs(hData[i+1][i]) > epsilon) {
                // 2x2 block
                double a = hData[i][i];
                double b = hData[i][i+1];
                double c = hData[i+1][i];
                double d = hData[i+1][i+1];
                det *= (a * d - b * c);
                i++; // Skip the next element as it's part of this block
            } else {
                // 1x1 block
                det *= hData[i][i];
            }
        }
        
        return det;
    }

    /**
     * Check if the decomposed matrix is non-singular.
     * 
     * @return true if the decomposed matrix is non-singular
     * @throws IllegalStateException if decomposition has not been performed
     */
    @Override
    public boolean isNonSingular() {
        return Math.abs(getDeterminant()) > epsilon;
    }

    /**
     * Get the condition number of the matrix.
     * 
     * @return condition number of the matrix
     * @throws IllegalStateException if decomposition has not been performed
     */
    @Override
    public double getConditionNumber() {
        if (hData == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        
        // Estimate the condition number from the Schur form
        // The condition number is the ratio of the largest to smallest singular values
        // For a triangular matrix, we can estimate from the diagonal elements
        double maxDiag = 0.0;
        double minDiag = Double.POSITIVE_INFINITY;
        
        for (int i = 0; i < hData.length; i++) {
            double diagAbs = Math.abs(hData[i][i]);
            maxDiag = Math.max(maxDiag, diagAbs);
            if (diagAbs > epsilon) {
                minDiag = Math.min(minDiag, diagAbs);
            }
        }
        
        if (minDiag == Double.POSITIVE_INFINITY || minDiag < epsilon) {
            return Double.POSITIVE_INFINITY; // Singular matrix
        }
        
        return maxDiag / minDiag;
    }

    /**
     * Get the rank of the matrix.
     * 
     * @return rank of the matrix
     * @throws IllegalStateException if decomposition has not been performed
     */
    @Override
    public int getRank() {
        if (hData == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        
        // Count non-zero diagonal elements in the Schur form
        int rank = 0;
        for (int i = 0; i < hData.length; i++) {
            if (Math.abs(hData[i][i]) > epsilon) {
                rank++;
            }
        }
        
        return rank;
    }

    /**
     * Get the maximum number of iterations allowed.
     * 
     * @return the maximum number of iterations
     */
    @Override
    public int getMaxIterations() {
        return maxIterations;
    }
}