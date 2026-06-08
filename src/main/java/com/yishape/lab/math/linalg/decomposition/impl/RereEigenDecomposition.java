package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.IEigenDecomposition;

import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.solver.EigenDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.NonSquareMatrixException;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.math.util.RerePrecision;

/**
 * Implementation of eigen decomposition with enhanced numerical stability.
 * <p>
 * This implementation computes the eigenvalue decomposition of a square matrix A
 * such that A = V * D * V^(-1) where D is a diagonal matrix of eigenvalues and
 * the columns of V are the corresponding eigenvectors.
 * </p>
 *
 * <h3>Algorithm Improvements</h3>
 * <ul>
 *   <li>Automatic detection of symmetric matrices for optimized paths</li>
 *   <li>Tridiagonal reduction for symmetric matrices</li>
 *   <li>Hessenberg reduction for general matrices</li>
 *   <li>Enhanced QR algorithm with shifts</li>
 *   <li>Better numerical stability with precision-aware comparisons</li>
 *   <li>Comprehensive error reporting with context information</li>
 *   <li>Efficient caching of computed results</li>
 * </ul>
 *
 * <h3>References</h3>
 * <ul>
 *   <li>Golub, G. H., &amp; Van Loan, C. F. (2013). Matrix computations (4th ed.). Johns Hopkins University Press.</li>
 *   <li>Press, W. H., Teukolsky, S. A., Vetterling, W. T., &amp; Flannery, B. P. (2007). Numerical recipes: The art of scientific computing (3rd ed.). Cambridge University Press.</li>
 * </ul>
 *
 * @author RereMouse
 * @version 1.0
 * @since 2.0
 */
public class RereEigenDecomposition implements IEigenDecomposition {

    /** Default epsilon for numerical comparisons. */
    private static final double DEFAULT_EPSILON = 1e-12;
    /** Default maximum number of iterations. */
    private static final int DEFAULT_MAX_ITERATIONS = 2000;

    /** Diagnostic: last Francis QR iteration count */
    public int lastIterCount = 0;
    /** Diagnostic: last number of Francis QR steps (performFrancisQRStep calls) */
    public int lastStepCount = 0;

    /** Cached eigenvalues. */
    private IVector<Double> cachedEigenvalues;
    /** Cached eigenvectors. */
    private IMatrix<Double> cachedEigenvectors;
    /** Determinant of the matrix. */
    private Double determinant;
    /** Condition number of the matrix. */
    private Double conditionNumber;
    /** Rank of the matrix. */
    private Integer rank;
    /** Epsilon for numerical comparisons. */
    private double epsilon;
    /** Maximum number of iterations. */
    private int maxIterations;

    /**
     * Default constructor with default parameters.
     */
    public RereEigenDecomposition() {
        this.epsilon = DEFAULT_EPSILON;
        this.maxIterations = DEFAULT_MAX_ITERATIONS;
    }

    /**
     * Constructor with unified parameters.
     *
     * @param epsilon threshold for considering an element as zero
     * @param maxIterations maximum number of iterations
     */
    public RereEigenDecomposition(double epsilon, int maxIterations) {
        this.epsilon = epsilon;
        this.maxIterations = maxIterations;
    }

    @Override
    public IDecompositionSolver getSolver() {
        if (cachedEigenvectors == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        // Return the standalone solver
        return new EigenDecompositionSolver(cachedEigenvalues, cachedEigenvectors, epsilon);
    }

    @Override
    public double getDeterminant() {
        if (determinant == null) {
            if (cachedEigenvalues == null) {
                throw new IllegalStateException("Decomposition not yet performed");
            }
            double det = 1.0;
            for (int i = 0; i < cachedEigenvalues.length(); i++) {
                det *= cachedEigenvalues.get(i);
            }
            determinant = det;
        }
        return determinant;
    }

    @Override
    public boolean isNonSingular() {
        if (cachedEigenvalues == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        // A matrix is non-singular if all eigenvalues are non-zero
        for (int i = 0; i < cachedEigenvalues.length(); i++) {
            if (Math.abs(cachedEigenvalues.get(i)) < epsilon) {
                return false;
            }
        }
        return true;
    }

    @Override
    public double getConditionNumber() {
        if (conditionNumber == null) {
            if (cachedEigenvalues == null) {
                throw new IllegalStateException("Decomposition not yet performed");
            }
            // Estimate condition number using the ratio of largest to smallest eigenvalues in absolute value
            double maxEigen = 0.0;
            double minEigen = Double.POSITIVE_INFINITY;
            for (int i = 0; i < cachedEigenvalues.length(); i++) {
                double eigen = Math.abs(cachedEigenvalues.get(i));
                maxEigen = Math.max(maxEigen, eigen);
                minEigen = Math.min(minEigen, eigen);
            }
            if (minEigen > epsilon) {
                conditionNumber = maxEigen / minEigen;
            } else {
                conditionNumber = Double.POSITIVE_INFINITY;
            }
        }
        return conditionNumber;
    }

    @Override
    public int getRank() {
        if (rank == null) {
            if (cachedEigenvalues == null) {
                throw new IllegalStateException("Decomposition not yet performed");
            }
            // Rank is the number of non-zero eigenvalues
            int r = 0;
            for (int i = 0; i < cachedEigenvalues.length(); i++) {
                if (Math.abs(cachedEigenvalues.get(i)) > epsilon) {
                    r++;
                }
            }
            rank = r;
        }
        return rank;
    }

    @Override
    public double getEpsilon() {
        return epsilon;
    }

    @Override
    public int getMaxIterations() {
        return maxIterations;
    }

    @Override
    public Tuple2<IVector<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix) {
        return decompose(matrix, epsilon);
    }

    @Override
    public Tuple2<IVector<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix, double epsilon) {
        return decompose(matrix, epsilon, this.maxIterations);
    }

    @Override
    public Tuple2<IVector<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix, double epsilon, int maxIterations) {
        // Reset cached values
        cachedEigenvalues = null;
        cachedEigenvectors = null;
        determinant = null;
        conditionNumber = null;
        rank = null;
        this.epsilon = epsilon;
        this.maxIterations = maxIterations;

        IDoubleMatrix doubleMatrix = (IDoubleMatrix) matrix;
        double[][] data = doubleMatrix.getData();

        // Check if matrix is square - eigen decomposition only applies to square matrices
        if (data.length != data[0].length) {
            throw new NonSquareMatrixException(
                "Eigendecomposition requires square matrix",
                "Eigen Decomposition",
                "Matrix " + data.length + "x" + data[0].length,
                data.length, data[0].length);
        }

        int n = data.length;  // Matrix dimension

        // Check if matrix is symmetric
        boolean isSymmetric = isSymmetric(doubleMatrix);

        Tuple2<IVector<Double>, IMatrix<Double>> result;
        if (isSymmetric) {
            // Eigen decomposition for symmetric matrices: use tridiagonalization + QR algorithm
            result = symmetricEigenDecomposition(doubleMatrix);
        } else {
            // Eigen decomposition for general matrices: use Hessenberg reduction + QR algorithm
            result = generalEigenDecomposition(doubleMatrix);
        }

        // Cache results
        cachedEigenvalues = result._1;
        cachedEigenvectors = result._2;

        return result;
    }

    /**
     * 检查矩阵是否对称
     */
    private boolean isSymmetric(IDoubleMatrix matrix) {
        int n = matrix.getRowNum();
        if (n != matrix.getColNum()) {
            return false;
        }

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (!RerePrecision.equals(matrix.get(i, j), matrix.get(j, i), epsilon)) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * 对称矩阵特征值分解 — 使用原始 double[][] 完成三对角化，消除 IMatrix 虚调用开销。
     */
    private Tuple2<IVector<Double>, IMatrix<Double>> symmetricEigenDecomposition(IDoubleMatrix matrix) {
        int n = matrix.getRowNum();
        double[][] aData = matrix.getData();

        double[][] T = new double[n][n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(aData[i], 0, T[i], 0, n);
        }

        double[][] z = new double[n][n];
        for (int i = 0; i < n; i++) z[i][i] = 1.0;

        double[] colBuf = new double[n];
        double[] v = new double[n];

        // Householder tridiagonalization (same pattern as RereTridiagonalDecomposition)
        for (int k = 0; k < n - 2; k++) {
            int len = n - k - 1;

            double norm = 0.0;
            for (int i = 0; i < len; i++) {
                double val = T[k + 1 + i][k];
                v[i] = val;
                norm += val * val;
            }
            norm = Math.sqrt(norm);
            if (norm <= epsilon) continue;

            v[0] += (v[0] >= 0 ? norm : -norm);

            double vNorm = 0.0;
            for (int i = 0; i < len; i++) vNorm += v[i] * v[i];
            vNorm = Math.sqrt(vNorm);
            if (vNorm <= epsilon) continue;

            for (int i = 0; i < len; i++) v[i] /= vNorm;

            // Left multiply: T = (I - 2vv^T) * T, columns k..n-1
            for (int j = k; j < n; j++) {
                double sum = 0.0;
                for (int i = 0; i < len; i++) {
                    double val = T[k + 1 + i][j];
                    colBuf[i] = val;
                    sum += v[i] * val;
                }
                double twoSum = 2.0 * sum;
                for (int i = 0; i < len; i++) {
                    T[k + 1 + i][j] = colBuf[i] - twoSum * v[i];
                }
            }

            // Right multiply: T = T * (I - 2vv^T)
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                double[] Ti = T[i];
                for (int j = 0; j < len; j++) {
                    sum += Ti[k + 1 + j] * v[j];
                }
                for (int j = 0; j < len; j++) {
                    Ti[k + 1 + j] -= 2.0 * sum * v[j];
                }
            }

            // Accumulate Q: Q = Q * (I - 2vv^T)
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                double[] zi = z[i];
                for (int j = 0; j < len; j++) {
                    sum += zi[k + 1 + j] * v[j];
                }
                for (int j = 0; j < len; j++) {
                    zi[k + 1 + j] -= 2.0 * sum * v[j];
                }
            }
        }

        // Extract diagonal and subdiagonal
        double[] main = new double[n];
        double[] secondary = new double[n - 1];
        for (int i = 0; i < n; i++) main[i] = T[i][i];
        for (int i = 0; i < n - 1; i++) secondary[i] = T[i + 1][i];

        // QL algorithm on tridiagonal (modifies main, secondary, z in-place)
        computeEigenvaluesAndVectors(main, secondary, z);

        normalizeEigenvectors(z, n);

        sortEigenvaluesAndVectors(main, z);

        // Wrap results
        IVector<Double> eigenvalues = Linalg.zeros(n);
        for (int i = 0; i < n; i++) eigenvalues.set(i, main[i]);

        double[][] eigenvectorData = new double[n][n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(z[i], 0, eigenvectorData[i], 0, n);
        }
        IMatrix<Double> eigenvectors = Linalg.matrix(eigenvectorData);

        return new Tuple2<>(eigenvalues, eigenvectors);
    }

    /**
     * 一般矩阵的特征分解 - 使用海森伯格化简
     */
    private Tuple2<IVector<Double>, IMatrix<Double>> generalEigenDecomposition(IDoubleMatrix matrix) {
        int n = matrix.getRowNum();

        // Use static raw-array helper directly — avoids IMatrix boxing overhead
        // that was 50%+ of runtime at small sizes (n=50).
        double[][] aData = matrix.getData();
        Tuple2<double[][], double[][]> hessResult = RereHessenbergDecomposition.reduceToHessenberg(aData, n, epsilon);
        double[][] H = hessResult._1;
        double[][] Q = hessResult._2;

        // Francis QR algorithm accumulates Givens rotations directly into Q,
        // producing Q = Q_hess * Q_qr. The eigenvectors returned by
        // implicitQRAlgorithm are ALREADY in original coordinates because
        // back-substitution multiplies the combined Q with Schur eigenvectors.
        Tuple2<IDoubleVector, IDoubleMatrix> result = implicitQRAlgorithm(H, Q);
        return new Tuple2<>(result._1, result._2);
    }

    /**
     * Core eigenvalue computation based on Apache Commons Math implementation
     * Uses implicit QL transformation with Wilkinson shifts
     */
    private void computeEigenvaluesAndVectors(double[] main, double[] secondary, double[][] z) {
        final int n = main.length;
        double[] e = new double[n];

        // Copy secondary diagonal
        for (int i = 0; i < n - 1; i++) {
            e[i] = secondary[i];
        }
        e[n - 1] = 0.0;

        // Determine the largest main and secondary value in absolute term
        double maxAbsoluteValue = 0;
        for (int i = 0; i < n; i++) {
            if (Math.abs(main[i]) > maxAbsoluteValue) {
                maxAbsoluteValue = Math.abs(main[i]);
            }
            if (Math.abs(e[i]) > maxAbsoluteValue) {
                maxAbsoluteValue = Math.abs(e[i]);
            }
        }

        // Make null any main and secondary value too small to be significant
        if (maxAbsoluteValue != 0) {
            for (int i = 0; i < n; i++) {
                if (Math.abs(main[i]) <= RerePrecision.getMachineEpsilon() * maxAbsoluteValue) {
                    main[i] = 0;
                }
                if (Math.abs(e[i]) <= RerePrecision.getMachineEpsilon() * maxAbsoluteValue) {
                    e[i] = 0;
                }
            }
        }

        // Main QL iteration loop
        for (int j = 0; j < n; j++) {
            int its = 0;
            int m;
            do {
                // Look for a single small sub-diagonal element to split the matrix
                for (m = j; m < n - 1; m++) {
                    double delta = Math.abs(main[m]) + Math.abs(main[m + 1]);
                    if (Math.abs(e[m]) + delta == delta) {
                        break;
                    }
                }

                if (m != j) {
                    if (its >= maxIterations) {
                        throw new RuntimeException("QR algorithm failed to converge after " + maxIterations + " iterations");
                    }
                    its++;

                    // Compute Wilkinson shift
                    double q = (main[j + 1] - main[j]) / (2 * e[j]);
                    double t = Math.sqrt(1 + q * q);
                    if (q < 0.0) {
                        q = main[m] - main[j] + e[j] / (q - t);
                    } else {
                        q = main[m] - main[j] + e[j] / (q + t);
                    }

                    double u = 0.0;
                    double s = 1.0;
                    double c = 1.0;
                    int i;

                    // Apply Givens rotations
                    for (i = m - 1; i >= j; i--) {
                        double p = s * e[i];
                        double h = c * e[i];
                        if (Math.abs(p) >= Math.abs(q)) {
                            c = q / p;
                            t = Math.sqrt(c * c + 1.0);
                            e[i + 1] = p * t;
                            s = 1.0 / t;
                            c *= s;
                        } else {
                            s = p / q;
                            t = Math.sqrt(s * s + 1.0);
                            e[i + 1] = q * t;
                            c = 1.0 / t;
                            s *= c;
                        }

                        if (e[i + 1] == 0.0) {
                            main[i + 1] -= u;
                            e[m] = 0.0;
                            break;
                        }

                        q = main[i + 1] - u;
                        t = (main[i] - q) * s + 2.0 * c * h;
                        u = s * t;
                        main[i + 1] = q + u;
                        q = c * t - h;

                        // Apply the transformation to the eigenvector matrix
                        for (int ia = 0; ia < n; ia++) {
                            p = z[ia][i + 1];
                            z[ia][i + 1] = s * z[ia][i] + c * p;
                            z[ia][i] = c * z[ia][i] - s * p;
                        }
                    }

                    if (t == 0.0 && i >= j) {
                        continue;
                    }

                    main[j] -= u;
                    e[j] = q;
                    e[m] = 0.0;
                }
            } while (m != j);
        }

        // Sort eigenvalues and eigenvectors in descending order
        for (int i = 0; i < n; i++) {
            int k = i;
            double p = main[i];
            for (int j = i + 1; j < n; j++) {
                if (main[j] > p) {
                    k = j;
                    p = main[j];
                }
            }
            if (k != i) {
                main[k] = main[i];
                main[i] = p;
                for (int j = 0; j < n; j++) {
                    p = z[j][i];
                    z[j][i] = z[j][k];
                    z[j][k] = p;
                }
            }
        }

        // Determine the largest eigenvalue in absolute terms
        maxAbsoluteValue = 0;
        for (int i = 0; i < n; i++) {
            if (Math.abs(main[i]) > maxAbsoluteValue) {
                maxAbsoluteValue = Math.abs(main[i]);
            }
        }

        // Make null any eigenvalue too small to be significant
        if (maxAbsoluteValue != 0.0) {
            for (int i = 0; i < n; i++) {
                if (Math.abs(main[i]) < RerePrecision.getMachineEpsilon() * maxAbsoluteValue) {
                    main[i] = 0;
                }
            }
        }
    }


    /**
     * Francis double-shift QR algorithm for Hessenberg matrices.
     * Converges H to real Schur form T = Q^T H Q, then extracts eigenvalues/eigenvectors.
     * Hot-path optimized: zero heap allocations in QR steps, inlined Householder, batched back-substitution.
     *
     * Key improvements over naive QR:
     * 1. Scans ALL subdiagonals for deflation (not just bottom two) — CM4 findSmallSubDiagonalElement
     * 2. Per-subproblem iteration counting (reset on deflation, max 100 per eigenvalue)
     * 3. Exceptional shifts at iter=10,30 to break eigenvalue clusters (LAPACK strategy)
     */
    private Tuple2<IDoubleVector, IDoubleMatrix> implicitQRAlgorithm(double[][] H) {
        return implicitQRAlgorithm(H, null);
    }

    /**
     * Francis QR algorithm with optional external initial Q.
     * <p>If initialQ is non-null, QR Givens rotations accumulate directly into it,
     * producing Q = initialQ * Q_qr. This eliminates one O(n³) matrix multiply
     * when the caller already has the Hessenberg transformation matrix.</p>
     */
    private Tuple2<IDoubleVector, IDoubleMatrix> implicitQRAlgorithm(double[][] H, double[][] initialQ) {
        int n = H.length;
        double[][] Q;
        if (initialQ != null) {
            Q = initialQ;
        } else {
            Q = new double[n][n];
            for (int i = 0; i < n; i++) Q[i][i] = 1.0;
        }
        double[][] T = new double[n][n];

        for (int i = 0; i < n; i++) {
            System.arraycopy(H[i], 0, T[i], 0, n);
        }

        final double eps = epsilon;

        // Pre-allocated work arrays for Householder (avoids heap allocations)
        double[] hv = new double[3]; // [v1, v2, beta] for 3-element householder
        double[] work = new double[n]; // column extraction buffer for stride-1 arithmetic

        // Pre-compute norm of T for deflation tolerance scaling
        double norm = 0.0;
        for (int i = 0; i < n; i++) {
            int start = Math.max(0, i - 1);
            for (int j = start; j < n; j++) norm += Math.abs(T[i][j]);
        }

        int idx = n - 1;      // trailing index (bottom of active subproblem)
        int iter = 0;         // iterations for CURRENT subproblem
        int stepCount = 0;
        final int maxIterPerSub = Math.max(30 * n, 100);

        while (idx >= 0) {
            // CM4 findSmallSubDiagonalElement: scan UP from idx to find first negligible subdiagonal
            int l = idx;
            while (l > 0) {
                double s = Math.abs(T[l - 1][l - 1]) + Math.abs(T[l][l]);
                if (s == 0.0) s = norm;
                if (Math.abs(T[l][l - 1]) <= eps * s) break;
                l--;
            }

            if (l == idx) {
                // 1x1 block deflated — real eigenvalue, idx is converged
                idx--;
                iter = 0;
                continue;
            }

            if (l == idx - 1) {
                // 2x2 block deflated — two real eigenvalues or complex conjugate pair
                double a = T[idx - 1][idx - 1], b = T[idx - 1][idx];
                double c2 = T[idx][idx - 1], d = T[idx][idx];
                double tr = a + d, det = a * d - b * c2;
                double disc = tr * tr - 4.0 * det;
                if (disc >= 0) {
                    double sd = Math.sqrt(disc);
                    double lam1 = (tr + sd) / 2.0;
                    // Compute eigenvector of the 2x2 block for lam1 and apply
                    // a Givens rotation to upper-triangularize T, accumulating
                    // the transformation into Q so that eigenvectors are correct.
                    double v1, v2;
                    if (Math.abs(b) > Math.abs(c2)) {
                        v1 = b;
                        v2 = lam1 - a;
                    } else {
                        v1 = lam1 - d;
                        v2 = c2;
                    }
                    double vn = Math.sqrt(v1 * v1 + v2 * v2);
                    if (vn > eps) {
                        double cc = v1 / vn;
                        double ss = v2 / vn;
                        // GUARD: Givens similarity T = G * T * G^T must apply to ALL
                        // rows/columns [0, n), not just [0, idx]. G differs from I
                        // only at the 2×2 block (idx-1, idx), but T*G^T mixes column
                        // values into those two columns from every row, so the column
                        // sweep must be full-range. Same for rows in G*T.
                        // A prior bug limited ranges to <= idx, causing non-symmetric
                        // eigenpath failures for n≥4.
                        // See: decomposition-pitfalls.md § GivensDeflationRange
                        // Left multiply G*T affects rows idx-1,idx for ALL columns.
                        for (int jj = 0; jj < n; jj++) {
                            double t1 = T[idx - 1][jj];
                            double t2 = T[idx][jj];
                            T[idx - 1][jj] = cc * t1 + ss * t2;
                            T[idx][jj] = -ss * t1 + cc * t2;
                        }
                        // Right multiply T*G^T affects columns idx-1,idx for ALL rows.
                        for (int ii = 0; ii < n; ii++) {
                            double t1 = T[ii][idx - 1];
                            double t2 = T[ii][idx];
                            T[ii][idx - 1] = cc * t1 + ss * t2;
                            T[ii][idx] = -ss * t1 + cc * t2;
                        }
                        // Accumulate Q = Q * G^T
                        for (int ii = 0; ii < n; ii++) {
                            double q1 = Q[ii][idx - 1];
                            double q2 = Q[ii][idx];
                            Q[ii][idx - 1] = cc * q1 + ss * q2;
                            Q[ii][idx] = -ss * q1 + cc * q2;
                        }
                    } else {
                        T[idx - 1][idx - 1] = lam1;
                        T[idx][idx] = tr - lam1;
                    }
                }
                // For complex eigenvalues, keep the 2x2 block as-is (real Schur form)
                idx -= 2;
                iter = 0;
                continue;
            }

            // No deflation possible — perform one Francis QR step on submatrix [l, idx]
            iter++;
            if (iter > maxIterPerSub) {
                throw new RuntimeException("QR algorithm failed to converge after " + maxIterPerSub + " iterations at idx=" + idx);
            }

            int activeN = idx - l + 1;
            // Exceptional shifts at iter 10 and 30 (LAPACK strategy)
            boolean useExceptionalShift = (iter == 10 || iter == 30);
            performFrancisQRStep(T, Q, l, idx, activeN, n, hv, work, useExceptionalShift);
            stepCount++;
            cleanRoundingErrors(T, n);
        }

        lastIterCount = stepCount;
        lastStepCount = stepCount;

        double[] eigenvalues = extractRealSchurEigenvalues(T, n, eps);
        double[][] eigvecsRaw = backSubstituteSchurEigenvectors(T, Q, n, eps, eigenvalues);

        normalizeEigenvectors(eigvecsRaw, n);
        sortEigenvaluesAndVectors(eigenvalues, eigvecsRaw);

        return new Tuple2<>(IDoubleVector.of(eigenvalues), (IDoubleMatrix) Linalg.matrix(eigvecsRaw));
    }

    // ======================== Francis double-shift QR step (allocation-free) ========================

    /**
     * One Francis double-shift QR step on submatrix T[l..idx][l..idx].
     * Uses eigenvalues of trailing 2x2 block as shifts. Hot-path: zero allocations, inlined Householder.
     * hv[3] is pre-allocated work array: hv[0]=v1, hv[1]=v2/nan, hv[2]=beta.
     */
    private void performFrancisQRStep(double[][] T, double[][] Q, int l, int idx,
                                       int activeN, int fullN,
                                       double[] hv, double[] work, boolean useExceptionalShift) {
        // Shifts = eigenvalues of trailing 2x2 block
        double a = T[idx - 1][idx - 1];
        double b = T[idx - 1][idx];
        double c = T[idx][idx - 1];
        double d = T[idx][idx];
        double traceS = a + d;
        double detS = a * d - b * c;

        // Exceptional shift: perturb to break eigenvalue clusters (LAPACK DLAHQR strategy)
        if (useExceptionalShift) {
            double s = Math.abs(T[idx][idx - 1]);
            if (activeN > 2) s += Math.abs(T[idx - 1][idx - 2]);
            double adj = 0.75 * s;
            traceS += adj;
            detS += adj * Math.max(Math.abs(T[idx - 1][idx - 1]), Math.abs(T[idx][idx]));
        }

        // First column of (T_sub^2 - traceS*T_sub + detS*I)*e1 where T_sub = T[l..idx][l..idx]
        // This encodes both shifts implicitly
        double h00 = T[l][l], h01 = T[l][l + 1];
        double h10 = T[l + 1][l], h11 = T[l + 1][l + 1];
        double h21 = (activeN > 2) ? T[l + 2][l + 1] : 0.0;

        double x = h00 * h00 + h01 * h10 - traceS * h00 + detS;
        double y = h10 * (h00 + h11 - traceS);
        double z = h10 * h21;

        for (int k = l; k < idx; k++) {
            double beta;
            int hvLen;
            boolean use3 = (k < idx - 1) && (Math.abs(z) > 1e-30);

            if (use3) {
                // Inline householder3
                double norm = Math.sqrt(x * x + y * y + z * z);
                if (norm == 0.0) {
                    if (k < idx - 1) {
                        x = T[k + 1][k]; y = T[k + 2][k];
                        z = (k + 3 <= idx) ? T[k + 3][k] : 0.0;
                    }
                    continue;
                }
                double alpha = (x > 0) ? -norm : norm;
                beta = (alpha - x) / alpha;
                hv[0] = y / (x - alpha);
                hv[1] = z / (x - alpha);
                hv[2] = beta;
                hvLen = 3;
            } else {
                // Inline householder2
                double norm = Math.sqrt(x * x + y * y);
                if (norm == 0.0) {
                    if (k < idx - 1) {
                        x = T[k + 1][k]; y = T[k + 2][k];
                        z = (k + 3 <= idx) ? T[k + 3][k] : 0.0;
                    }
                    continue;
                }
                double alpha = (x > 0) ? -norm : norm;
                beta = (alpha - x) / alpha;
                hv[0] = y / (x - alpha);
                hv[1] = beta;
                hvLen = 2;
            }

            int rEnd = Math.min(k + hvLen, idx);
            int cEnd = Math.min(k + hvLen - 1, idx);
            int leftColStart = (k > l) ? k - 1 : l;

            // Left multiply: T(k:rEnd, leftColStart:idx) = P * T(...)
            // Extract column segment into work buffer for stride-1 arithmetic,
            // avoiding the stride-n penalty of T[k+t][j] with varying t.
            for (int j = leftColStart; j <= idx; j++) {
                work[0] = T[k][j];
                for (int t = 1; t < hvLen; t++) work[t] = T[k + t][j];
                double s = work[0];
                for (int t = 1; t < hvLen; t++) s += hv[t - 1] * work[t];
                s *= beta;
                T[k][j] = work[0] - s;
                for (int t = 1; t < hvLen; t++) T[k + t][j] = work[t] - s * hv[t - 1];
            }

            // Right multiply: T(l:maxRow, k:cEnd) = T(...) * P^T
            int maxRow = Math.min(k + hvLen, idx);
            for (int i = l; i <= maxRow; i++) {
                double[] row = T[i];
                double s = row[k];
                for (int t = 1; t < hvLen; t++) {
                    s += hv[t - 1] * row[k + t];
                }
                s *= beta;
                row[k] -= s;
                for (int t = 1; t < hvLen; t++) {
                    row[k + t] -= s * hv[t - 1];
                }
            }

            // Accumulate Q: Q(0:fullN-1, k:cEnd) = Q(...) * P^T
            for (int i = 0; i < fullN; i++) {
                double[] row = Q[i];
                double s = row[k];
                for (int t = 1; t < hvLen; t++) {
                    s += hv[t - 1] * row[k + t];
                }
                s *= beta;
                row[k] -= s;
                for (int t = 1; t < hvLen; t++) {
                    row[k + t] -= s * hv[t - 1];
                }
            }

            if (k < idx - 1) {
                x = T[k + 1][k];
                y = T[k + 2][k];
                z = (k + 3 <= idx) ? T[k + 3][k] : 0.0;
            }
        }
    }

    // ======================== Eigenvalue / Eigenvector extraction ========================

    private double[] extractRealSchurEigenvalues(double[][] T, int n, double eps) {
        double[] evals = new double[n];
        int i = 0;
        while (i < n) {
            if (i < n - 1 && Math.abs(T[i + 1][i]) > eps * (Math.abs(T[i][i]) + Math.abs(T[i + 1][i + 1]) + 1.0)) {
                double a = T[i][i], b = T[i][i + 1];
                double c = T[i + 1][i], d = T[i + 1][i + 1];
                double disc = (a - d) * (a - d) + 4.0 * b * c;
                if (disc < 0) {
                    evals[i] = (a + d) / 2.0;
                    evals[i + 1] = evals[i];
                } else {
                    double sd = Math.sqrt(disc);
                    evals[i] = (a + d + sd) / 2.0;
                    evals[i + 1] = (a + d - sd) / 2.0;
                }
                i += 2;
            } else {
                evals[i] = T[i][i];
                i++;
            }
        }
        return evals;
    }

    /**
     * Back-substitution to extract eigenvectors from real Schur form.
     * Optimized: uses pre-extracted eigenvalues, GEMM for Q*Y multiplication.
     */
    private double[][] backSubstituteSchurEigenvectors(double[][] T, double[][] Q, int n,
                                                        double eps, double[] evals) {
        // Phase 1: Back-substitute to get eigenvectors y of T: (T - λI)y = 0
        // Store results column-by-column in matrix Y (stored as rows for GEMM)
        double[][] Y = new double[n][n]; // Y[j][p] = y_p[j] (transposed for GEMM)

        int p = 0;
        while (p < n) {
            boolean is2x2 = (p < n - 1) &&
                Math.abs(T[p + 1][p]) > eps * (Math.abs(T[p][p]) + Math.abs(T[p + 1][p + 1]) + 1.0);

            if (is2x2) {
                double a = T[p][p], b = T[p][p + 1];
                double c = T[p + 1][p], d = T[p + 1][p + 1];
                double disc = (a - d) * (a - d) + 4.0 * b * c;

                // y1 and y2 stored directly in Y columns p and p+1
                double[] y1 = Y[p];     // Y column p
                double[] y2 = Y[p + 1]; // Y column p+1

                if (disc < 0) {
                    y1[p] = 1.0; y1[p + 1] = 0.0;
                    y2[p] = 0.0; y2[p + 1] = 1.0;
                } else {
                    double sd = Math.sqrt(disc);
                    double lam1 = (a + d + sd) / 2.0;
                    double lam2 = (a + d - sd) / 2.0;
                    if (Math.abs(b) > Math.abs(c)) {
                        y1[p] = 1.0; y1[p + 1] = Math.abs(b) > 1e-30 ? (lam1 - a) / b : 0.0;
                        y2[p] = 1.0; y2[p + 1] = Math.abs(b) > 1e-30 ? (lam2 - a) / b : 0.0;
                    } else {
                        y1[p + 1] = 1.0; y1[p] = Math.abs(c) > 1e-30 ? (lam1 - d) / c : 0.0;
                        y2[p + 1] = 1.0; y2[p] = Math.abs(c) > 1e-30 ? (lam2 - d) / c : 0.0;
                    }
                }

                // Back-substitute for both eigenvalues simultaneously
                for (int k = p - 1; k >= 0; k--) {
                    double sum1 = 0.0, sum2 = 0.0;
                    double[] Tk = T[k];
                    for (int j = k + 1; j <= p + 1; j++) {
                        sum1 += Tk[j] * y1[j];
                        sum2 += Tk[j] * y2[j];
                    }
                    double denom1 = Tk[k] - evals[p];
                    double denom2 = Tk[k] - evals[p + 1];
                    y1[k] = (Math.abs(denom1) > eps) ? -sum1 / denom1 : 0.0;
                    y2[k] = (Math.abs(denom2) > eps) ? -sum2 / denom2 : 0.0;
                }
                p += 2;
            } else {
                double lam = T[p][p];
                double[] y = Y[p];
                y[p] = 1.0;
                for (int k = p - 1; k >= 0; k--) {
                    double sum = 0.0;
                    double[] Tk = T[k];
                    for (int j = k + 1; j <= p; j++) {
                        sum += Tk[j] * y[j];
                    }
                    double denom = Tk[k] - lam;
                    y[k] = (Math.abs(denom) > eps) ? -sum / denom : 0.0;
                }
                p++;
            }
        }

        // Phase 2: eigvecs = Q * Y  (Y rows = eigenvectors, dot-product GEMM)
        double[][] eigvecs = new double[n][n];
        for (int i = 0; i < n; i++) {
            double[] Qi = Q[i];
            double[] eRow = eigvecs[i];
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                double[] Yj = Y[j];
                for (int t = 0; t < n; t++) {
                    s += Qi[t] * Yj[t];
                }
                eRow[j] = s;
            }
        }

        return eigvecs;
    }

    private void cleanRoundingErrors(double[][] matrix, int n) {
        for (int i = 2; i < n; i++) {
            if (RerePrecision.equalsZero(matrix[i][i - 2], 1e-12)) {
                matrix[i][i - 2] = 0.0;
            }
            if (i > 2 && RerePrecision.equalsZero(matrix[i][i - 3], 1e-12)) {
                matrix[i][i - 3] = 0.0;
            }
        }
    }

    /**
     * 标准化特征向量 - 基于Apache Commons Math3的实现
     *
     * @param eigenvectors 特征向量矩阵
     * @param n 矩阵维度
     */
    private void normalizeEigenvectors(double[][] eigenvectors, int n) {
        for (int j = 0; j < n; j++) {
            double norm = 0.0;
            for (int i = 0; i < n; i++) {
                norm += eigenvectors[i][j] * eigenvectors[i][j];
            }
            norm = Math.sqrt(norm);

            if (!RerePrecision.equalsZero(norm, 1e-10)) {
                for (int i = 0; i < n; i++) {
                    eigenvectors[i][j] /= norm;
                }
            } else {
                // 如果向量线性相关，设置为单位向量
                for (int i = 0; i < n; i++) {
                    eigenvectors[i][j] = (i == j) ? 1.0 : 0.0;
                }
            }
        }
    }

    /**
     * 特征值和特征向量排序（Eigenvalue and Eigenvector Sorting）
     *
     * <p>
     * 在特征分解完成后，需要将特征值和对应的特征向量按特征值大小进行排序。 这确保了特征分解结果的一致性和可预测性，便于后续的数值计算和分析。</p>
     *
     * <p>
     * 排序策略：</p>
     * <ul>
     * <li><strong>降序排列</strong>：按特征值大小从大到小排序</li>
     * <li><strong>保持对应关系</strong>：特征向量与特征值保持一一对应</li>
     * <li><strong>稳定性</strong>：使用稳定的排序算法保持相对顺序</li>
     * </ul>
     *
     * <p>
     * 算法选择：</p>
     * <ul>
     * <li>使用冒泡排序，虽然时间复杂度为O(n²)，但对于特征分解中的小矩阵足够高效</li>
     * <li>冒泡排序是稳定排序，保持相等元素的相对顺序</li>
     * <li>实现简单，易于理解和维护</li>
     * </ul>
     *
     * @param eigenvalues 特征值数组，将被就地排序
     * @param eigenvectors 特征向量矩阵（按行存储），将与特征值同步排序
     */
    private void sortEigenvaluesAndVectors(double[] eigenvalues, double[][] eigenvectors) {
        int n = eigenvalues.length;
        // Selection sort: O(n²) comparisons, O(n) column swaps (each O(n)) = O(n²) total.
        // Bubble sort was previously used and caused O(n³) work for column swaps on n≥500.
        for (int i = 0; i < n - 1; i++) {
            int maxIdx = i;
            double maxVal = eigenvalues[i];
            for (int j = i + 1; j < n; j++) {
                if (eigenvalues[j] > maxVal) {
                    maxIdx = j;
                    maxVal = eigenvalues[j];
                }
            }
            if (maxIdx != i) {
                eigenvalues[maxIdx] = eigenvalues[i];
                eigenvalues[i] = maxVal;
                for (int row = 0; row < n; row++) {
                    double temp = eigenvectors[row][i];
                    eigenvectors[row][i] = eigenvectors[row][maxIdx];
                    eigenvectors[row][maxIdx] = temp;
                }
            }
        }
    }

    /**
     * 计算Wilkinson位移 - 基于Apache Commons Math3的实现
     *
     * @param matrix 矩阵
     * @param n 矩阵维度
     * @return Wilkinson位移值
     */
    private double computeShift(double[][] matrix, int n) {
        if (n < 2) {
            return 0.0;
        }

        // 提取右下角2×2子矩阵的元素
        double a = matrix[n - 2][n - 2];  // 主对角线元素
        double b = matrix[n - 2][n - 1];  // 次对角线元素
        double d = matrix[n - 1][n - 1];  // 右下角元素

        // 计算2×2子矩阵的特征值
        double trace = a + d;                    // 迹（trace）
        double det = a * d - b * b;             // 行列式（determinant）
        double discriminant = trace * trace - 4 * det;  // 判别式

        if (discriminant >= 0) {
            // 实数特征值情况：使用二次公式
            double sqrtDisc = Math.sqrt(discriminant);
            double lambda1 = (trace + sqrtDisc) / 2.0;  // 较大特征值
            double lambda2 = (trace - sqrtDisc) / 2.0;  // 较小特征值

            // 选择更接近d的特征值，提高收敛速度
            double diff1 = Math.abs(lambda1 - d);
            double diff2 = Math.abs(lambda2 - d);

            // 数值稳定性考虑：如果两个特征值都很接近，选择较小的那个
            if (RerePrecision.equals(diff1, diff2, 1e-6)) {
                return Math.min(lambda1, lambda2);
            } else {
                return (diff1 < diff2) ? lambda1 : lambda2;
            }
        } else {
            // 复数特征值情况：使用Rayleigh商位移
            // 当判别式为负时，选择右下角元素作为位移
            return d;
        }
    }

    /**
     * 执行隐式QR步骤 - 基于Apache Commons Math3的实现
     *
     * @param matrix 矩阵
     * @param eigenvectors 特征向量矩阵
     * @param shift 位移值
     * @param n 矩阵维度
     */
    private void performImplicitQRStep(double[][] matrix, double[][] eigenvectors, double shift, int n) {
        // 对每一列应用Givens旋转
        for (int k = 0; k < n - 1; k++) {
            // 计算Givens旋转参数
            // 目标：将matrix[k+1][k]位置清零，实现QR分解的效果
            double a = matrix[k][k] - shift;    // 减去位移后的主对角线元素
            double b = matrix[k + 1][k];        // 次对角线元素

            // 使用我们自定义的Precision类进行数值比较
            if (RerePrecision.equalsZero(b, 1e-12)) {
                continue;
            }

            double r = Math.sqrt(a * a + b * b);  // 旋转半径

            if (!RerePrecision.equalsZero(r, 1e-12)) {  // 避免除零错误，确保数值稳定性
                double c = a / r;      // 余弦值（cosine）
                double s = -b / r;     // 正弦值（sine），注意符号

                // 应用Givens旋转到矩阵A
                // 旋转矩阵G作用于行k和k+1，列k到n-1
                // 这相当于执行QR分解中的Q^T * A操作
                applyGivensRotation(matrix, c, s, k, k + 1, k, n - 1);

                // 应用Givens旋转到特征向量矩阵
                // 注意：由于矩阵按行存储，需要调整更新逻辑
                // 这里更新的是特征向量矩阵的列，而不是行
                for (int i = 0; i < n; i++) {
                    double temp = eigenvectors[i][k];
                    eigenvectors[i][k] = c * temp + s * eigenvectors[i][k + 1];
                    eigenvectors[i][k + 1] = -s * temp + c * eigenvectors[i][k + 1];
                }
            }
        }
    }

    /**
     * 计算矩阵范数 - 基于Apache Commons Math3的实现
     *
     * @param matrix 矩阵
     * @param n 矩阵维度
     * @return 矩阵范数
     */
    private double computeMatrixNorm(double[][] matrix, int n) {
        double norm = 0.0;
        // 计算所有非对角线元素的绝对值之和
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    norm += Math.abs(matrix[i][j]);
                }
            }
        }
        return norm;
    }

    /**
     * 应用Givens旋转到矩阵（Apply Givens Rotation to Matrix）
     *
     * <p>
     * Givens旋转是数值线性代数中的基本变换，用于在矩阵的特定位置引入零元素。
     * 在QR分解和特征分解中，Givens旋转是实现矩阵变换的核心工具。</p>
     *
     * <p>
     * Givens旋转矩阵：</p>
     * <pre>
     * G = [c  -s] 其中 c² + s² = 1
     *     [s   c]
     * </pre>
     *
     * <p>
     * 算法原理：</p>
     * <ol>
     * <li>计算旋转参数c和s，使得s*a + c*b = 0</li>
     * <li>应用旋转矩阵到指定行，消除目标元素</li>
     * <li>保持矩阵的正交性和数值稳定性</li>
     * </ol>
     *
     * <p>
     * 应用场景：</p>
     * <ul>
     * <li>QR分解中的矩阵三角化</li>
     * <li>特征分解中的矩阵简化</li>
     * <li>数值线性代数中的基础变换</li>
     * </ul>
     *
     * @param matrix 矩阵
     * @param c 余弦值
     * @param s 正弦值
     * @param row1 第一行索引
     * @param row2 第二行索引
     * @param colStart 起始列索引
     * @param colEnd 结束列索引
     */
    private void applyGivensRotation(double[][] matrix, double c, double s, int row1, int row2, int colStart, int colEnd) {
        // 应用Givens旋转到指定行范围
        for (int j = colStart; j <= colEnd; j++) {
            double temp1 = matrix[row1][j];
            double temp2 = matrix[row2][j];
            matrix[row1][j] = c * temp1 - s * temp2;
            matrix[row2][j] = s * temp1 + c * temp2;
        }
    }

}
