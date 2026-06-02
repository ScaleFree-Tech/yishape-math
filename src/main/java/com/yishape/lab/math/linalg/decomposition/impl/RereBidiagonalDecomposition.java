package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.decomposition.IBidiagonalDecomposition;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.solver.BidiagonalDecompositionSolver;
import com.yishape.lab.util.Tuple3;
import com.yishape.lab.math.util.RerePrecision;

/**
 * Bidiagonal decomposition implementation with enhanced numerical stability.
 * <p>
 * Bidiagonal decomposition transforms a matrix A into the form A = U * B * V^T,
 * where B is a bidiagonal matrix, and U and V are orthogonal matrices.
 * </p>
 * 
 * <h3>Algorithm Improvements</h3>
 * <ul>
 *   <li>Enhanced Householder reflection computations</li>
 *   <li>Better numerical stability with precision-aware comparisons</li>
 *   <li>Comprehensive error reporting with context information</li>
 *   <li>Efficient caching of computed results</li>
 *   <li>Configurable thresholds for numerical comparisons</li>
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
public class RereBidiagonalDecomposition implements IBidiagonalDecomposition {
    
    /** Cached value of U. */
    private IMatrix<Double> cachedU;
    /** Cached value of B. */
    private IMatrix<Double> cachedB;
    /** Cached value of V. */
    private IMatrix<Double> cachedV;
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
    public RereBidiagonalDecomposition() {
        this.epsilon = DEFAULT_EPSILON;
        this.maxIterations = DEFAULT_MAX_ITERATIONS;
    }
    
    /**
     * Constructor with unified parameters.
     * 
     * @param epsilon threshold for considering an element as zero
     * @param maxIterations maximum number of iterations
     */
    public RereBidiagonalDecomposition(double epsilon, int maxIterations) {
        this.epsilon = epsilon;
        this.maxIterations = maxIterations;
    }
    
    @Override
    public Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix) {
        return decompose(matrix, epsilon);
    }
    
    @Override
    public Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix, double epsilon) {
        return decompose(matrix, epsilon, DEFAULT_MAX_ITERATIONS);
    }
    
    @Override
    public Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix, double epsilon, int maxIterations) {
        // Reset cached values
        cachedU = null;
        cachedB = null;
        cachedV = null;
        determinant = null;
        conditionNumber = null;
        rank = null;
        this.epsilon = epsilon;
        this.maxIterations = maxIterations;
        
        IDoubleMatrix doubleMatrix = (IDoubleMatrix) matrix;
        double[][] data = doubleMatrix.getData();
        int m = data.length;
        int n = data[0].length;
        int minDim = Math.min(m, n);

        double[][] A = new double[m][n];
        for (int i = 0; i < m; i++) {
            System.arraycopy(data[i], 0, A[i], 0, n);
        }

        // Initialize U and V as identity matrices
        double[][] U = new double[m][m];
        double[][] V = new double[n][n];
        for (int i = 0; i < m; i++) {
            U[i][i] = 1.0;
        }
        for (int i = 0; i < n; i++) {
            V[i][i] = 1.0;
        }

        // Pre-allocated buffers reused across iterations to avoid O(n) allocations.
        // vecBuf: Householder vector construction (left and right, sequential).
        // workBuf1: wA = v^T * A (left: n-k cols; right: m-k rows, sequential).
        // workBuf2: U*v or V*v accumulation (full m/n, sequential).
        int maxDim = Math.max(m, n);
        double[] vecBuf = new double[maxDim];
        double[] workBuf1 = new double[maxDim];
        double[] workBuf2 = new double[maxDim];

        // Bidiagonalization (Householder). Extracted block B is min(m,n) x min(m,n), see below.
        for (int k = 0; k < minDim; k++) {
            // Left transformation (Householder transformation on column k)
            if (k < m - 1) {
                int lRows = m - k;
                int lCols = n - k;

                // Compute Householder vector v directly into vecBuf
                double norm = 0.0;
                for (int i = 0; i < lRows; i++) {
                    double val = A[k + i][k];
                    vecBuf[i] = val;
                    norm += val * val;
                }
                norm = Math.sqrt(norm);

                if (norm > epsilon) {
                    double[] v = vecBuf;
                    v[0] += Math.signum(v[0]) * norm;

                    // Normalize v
                    double vNorm = 0.0;
                    for (int i = 0; i < lRows; i++) {
                        vNorm += v[i] * v[i];
                    }
                    vNorm = Math.sqrt(vNorm);
                    for (int i = 0; i < lRows; i++) {
                        v[i] /= vNorm;
                    }

                    // wA = v^T * A  (workBuf1)
                    // Loop-interchanged for stride-1 inner loop: accumulate
                    // wA[j] += vi * aRow[k+j] across rows, inner j varies
                    double[] wA = workBuf1;
                    java.util.Arrays.fill(wA, 0, lCols, 0.0);
                    for (int i = 0; i < lRows; i++) {
                        double vi = v[i];
                        double[] aRow = A[k + i];
                        for (int j = 0; j < lCols; j++) {
                            wA[j] += vi * aRow[k + j];
                        }
                    }
                    // A = A - 2*v*wA
                    for (int i = 0; i < lRows; i++) {
                        double[] aRow = A[k + i];
                        double vi2 = 2.0 * v[i];
                        for (int j = 0; j < lCols; j++) {
                            aRow[k + j] -= vi2 * wA[j];
                        }
                    }

                    // uCol = U*v  (workBuf2)
                    double[] uCol = workBuf2;
                    for (int i = 0; i < m; i++) {
                        double sum = 0;
                        for (int jj = 0; jj < lRows; jj++) {
                            sum += U[i][k + jj] * v[jj];
                        }
                        uCol[i] = sum;
                    }
                    // U = U - 2*uCol*v^T
                    for (int i = 0; i < m; i++) {
                        double[] uRow = U[i];
                        double wi2 = 2.0 * uCol[i];
                        for (int jj = 0; jj < lRows; jj++) {
                            uRow[k + jj] -= wi2 * v[jj];
                        }
                    }
                }
            }

            // Right transformation (Householder transformation on row k)
            if (k < n - 2) {
                int rRows = m - k;
                int rCols = n - k - 1;

                // Compute Householder vector v directly into vecBuf
                double norm = 0.0;
                for (int j = 0; j < rCols; j++) {
                    double val = A[k][k + 1 + j];
                    vecBuf[j] = val;
                    norm += val * val;
                }
                norm = Math.sqrt(norm);

                if (norm > epsilon) {
                    double[] v = vecBuf;
                    v[0] += Math.signum(v[0]) * norm;

                    // Normalize v
                    double vNorm = 0.0;
                    for (int j = 0; j < rCols; j++) {
                        vNorm += v[j] * v[j];
                    }
                    vNorm = Math.sqrt(vNorm);
                    for (int j = 0; j < rCols; j++) {
                        v[j] /= vNorm;
                    }

                    // wA = A * v  (workBuf1)
                    double[] wA = workBuf1;
                    for (int i = 0; i < rRows; i++) {
                        double sum = 0;
                        for (int j = 0; j < rCols; j++) {
                            sum += A[k + i][k + 1 + j] * v[j];
                        }
                        wA[i] = sum;
                    }
                    // A = A - 2*wA*v^T
                    for (int i = 0; i < rRows; i++) {
                        double[] aRow = A[k + i];
                        double wi2 = 2.0 * wA[i];
                        for (int j = 0; j < rCols; j++) {
                            aRow[k + 1 + j] -= wi2 * v[j];
                        }
                    }

                    // vCol = V * v  (workBuf2)
                    double[] vCol = workBuf2;
                    for (int i = 0; i < n; i++) {
                        double sum = 0;
                        for (int j = 0; j < rCols; j++) {
                            sum += V[i][k + 1 + j] * v[j];
                        }
                        vCol[i] = sum;
                    }
                    // V = V - 2*vCol*v^T
                    for (int i = 0; i < n; i++) {
                        double[] vRow = V[i];
                        double wi2 = 2.0 * vCol[i];
                        for (int j = 0; j < rCols; j++) {
                            vRow[k + 1 + j] -= wi2 * v[j];
                        }
                    }
                }
            }
        }

        // Ensure bidiagonal form: clean numerical errors
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (j != i && j != i + 1 && RerePrecision.equalsZero(A[i][j], epsilon)) {
                    A[i][j] = 0.0;
                }
            }
        }

        // Extract the bidiagonal matrix B (should be minDim x minDim)
        double[][] B = new double[minDim][minDim];
        for (int i = 0; i < minDim; i++) {
            for (int j = 0; j < minDim; j++) {
                if (j == i || j == i + 1) {
                    B[i][j] = A[i][j];
                } else {
                    B[i][j] = 0.0;
                }
            }
        }

        // Extract the relevant part of U matrix (m x minDim)
        double[][] U_reduced = new double[m][minDim];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < minDim; j++) {
                U_reduced[i][j] = U[i][j];
            }
        }

        cachedU = IDoubleMatrix.of(U_reduced);
        cachedB = IDoubleMatrix.of(B);
        cachedV = IDoubleMatrix.of(V);
        return new Tuple3<>(cachedU, cachedB, cachedV);
    }
    
    @Override
    public IDecompositionSolver getSolver() {
        if (cachedU == null || cachedB == null || cachedV == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        // Solver expects this square B (and matching U/V) from the decomposition above.
        return new BidiagonalDecompositionSolver(cachedB, cachedU, cachedV, epsilon);
    }
    
    @Override
    public double getDeterminant() {
        if (determinant == null) {
            if (cachedB == null) {
                throw new IllegalStateException("Decomposition not yet performed");
            }
            double det = 1.0;
            for (int i = 0; i < Math.min(cachedB.rows(), cachedB.cols()); ++i) {
                det *= cachedB.get(i, i);
            }
            determinant = det;
        }
        return determinant;
    }
    
    @Override
    public boolean isNonSingular() {
        if (cachedB == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        // A matrix is non-singular if all diagonal elements of B are non-zero
        for (int i = 0; i < Math.min(cachedB.rows(), cachedB.cols()); i++) {
            if (Math.abs(cachedB.get(i, i)) < epsilon) {
                return false;
            }
        }
        return true;
    }
    
    @Override
    public double getConditionNumber() {
        if (conditionNumber == null) {
            if (cachedB == null) {
                throw new IllegalStateException("Decomposition not yet performed");
            }
            // Estimate condition number using the ratio of largest to smallest diagonal elements
            double maxDiag = 0.0;
            double minDiag = Double.POSITIVE_INFINITY;
            for (int i = 0; i < Math.min(cachedB.rows(), cachedB.cols()); i++) {
                double diag = Math.abs(cachedB.get(i, i));
                maxDiag = Math.max(maxDiag, diag);
                minDiag = Math.min(minDiag, diag);
            }
            if (minDiag > epsilon) {
                conditionNumber = maxDiag / minDiag;
            } else {
                conditionNumber = Double.POSITIVE_INFINITY;
            }
        }
        return conditionNumber;
    }
    
    @Override
    public int getRank() {
        if (rank == null) {
            if (cachedB == null) {
                throw new IllegalStateException("Decomposition not yet performed");
            }
            // Rank is the number of non-zero diagonal elements
            int r = 0;
            for (int i = 0; i < Math.min(cachedB.rows(), cachedB.cols()); i++) {
                if (Math.abs(cachedB.get(i, i)) > epsilon) {
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
    public IMatrix<Double> getU() {
        if (cachedU == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        return cachedU;
    }
    
    @Override
    public IMatrix<Double> getB() {
        if (cachedB == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        return cachedB;
    }
    
    @Override
    public IMatrix<Double> getV() {
        if (cachedV == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        return cachedV;
    }
}