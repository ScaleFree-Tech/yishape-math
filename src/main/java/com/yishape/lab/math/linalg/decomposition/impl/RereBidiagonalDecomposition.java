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

        // Bidiagonalization (Householder). Extracted block B is min(m,n) x min(m,n), see below.
        for (int k = 0; k < minDim; k++) {
            // Left transformation (Householder transformation on column k)
            if (k < m - 1) {
                double[] x = new double[m - k];
                for (int i = k; i < m; i++) {
                    x[i - k] = A[i][k];
                }

                double norm = 0.0;
                for (double v : x) {
                    norm += v * v;
                }
                norm = Math.sqrt(norm);

                if (norm > epsilon) {
                    double[] v = new double[m - k];
                    v[0] = x[0] + Math.signum(x[0]) * norm;
                    for (int i = 1; i < v.length; i++) {
                        v[i] = x[i];
                    }

                    // Normalize v
                    double vNorm = 0.0;
                    for (double vi : v) {
                        vNorm += vi * vi;
                    }
                    vNorm = Math.sqrt(vNorm);
                    for (int i = 0; i < v.length; i++) {
                        v[i] /= vNorm;
                    }

                    // Construct Householder matrix P = I - 2*v*v^T
                    double[][] P = new double[m - k][m - k];
                    for (int i = 0; i < m - k; i++) {
                        for (int j = 0; j < m - k; j++) {
                            P[i][j] = (i == j ? 1.0 : 0.0) - 2.0 * v[i] * v[j];
                        }
                    }

                    // Apply transformation to A
                    double[][] subA = new double[m - k][n - k];
                    for (int i = 0; i < m - k; i++) {
                        for (int j = 0; j < n - k; j++) {
                            subA[i][j] = A[k + i][k + j];
                        }
                    }

                    double[][] PsubA = new double[m - k][n - k];
                    for (int i = 0; i < m - k; i++) {
                        for (int j = 0; j < n - k; j++) {
                            for (int l = 0; l < m - k; l++) {
                                PsubA[i][j] += P[i][l] * subA[l][j];
                            }
                        }
                    }

                    for (int i = 0; i < m - k; i++) {
                        for (int j = 0; j < n - k; j++) {
                            A[k + i][k + j] = PsubA[i][j];
                        }
                    }

                    // Update U
                    double[][] subU = new double[m][m - k];
                    for (int i = 0; i < m; i++) {
                        for (int j = 0; j < m - k; j++) {
                            subU[i][j] = U[i][k + j];
                        }
                    }

                    double[][] subUP = new double[m][m - k];
                    for (int i = 0; i < m; i++) {
                        for (int j = 0; j < m - k; j++) {
                            for (int l = 0; l < m - k; l++) {
                                subUP[i][j] += subU[i][l] * P[l][j];
                            }
                        }
                    }

                    for (int i = 0; i < m; i++) {
                        for (int j = 0; j < m - k; j++) {
                            U[i][k + j] = subUP[i][j];
                        }
                    }
                }
            }

            // Right transformation (Householder transformation on row k)
            if (k < n - 2) {
                double[] x = new double[n - k - 1];
                for (int j = k + 1; j < n; j++) {
                    x[j - k - 1] = A[k][j];
                }

                double norm = 0.0;
                for (double v : x) {
                    norm += v * v;
                }
                norm = Math.sqrt(norm);

                if (norm > epsilon) {
                    double[] v = new double[n - k - 1];
                    v[0] = x[0] + Math.signum(x[0]) * norm;
                    for (int i = 1; i < v.length; i++) {
                        v[i] = x[i];
                    }

                    // Normalize v
                    double vNorm = 0.0;
                    for (double vi : v) {
                        vNorm += vi * vi;
                    }
                    vNorm = Math.sqrt(vNorm);
                    for (int i = 0; i < v.length; i++) {
                        v[i] /= vNorm;
                    }

                    // Construct Householder matrix P = I - 2*v*v^T
                    double[][] P = new double[n - k - 1][n - k - 1];
                    for (int i = 0; i < n - k - 1; i++) {
                        for (int j = 0; j < n - k - 1; j++) {
                            P[i][j] = (i == j ? 1.0 : 0.0) - 2.0 * v[i] * v[j];
                        }
                    }

                    // Apply transformation to A
                    double[][] subA = new double[m - k][n - k - 1];
                    for (int i = 0; i < m - k; i++) {
                        for (int j = 0; j < n - k - 1; j++) {
                            subA[i][j] = A[k + i][k + 1 + j];
                        }
                    }

                    double[][] subAP = new double[m - k][n - k - 1];
                    for (int i = 0; i < m - k; i++) {
                        for (int j = 0; j < n - k - 1; j++) {
                            for (int l = 0; l < n - k - 1; l++) {
                                subAP[i][j] += subA[i][l] * P[l][j];
                            }
                        }
                    }

                    for (int i = 0; i < m - k; i++) {
                        for (int j = 0; j < n - k - 1; j++) {
                            A[k + i][k + 1 + j] = subAP[i][j];
                        }
                    }

                    // Update V
                    double[][] subV = new double[n][n - k - 1];
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < n - k - 1; j++) {
                            subV[i][j] = V[i][k + 1 + j];
                        }
                    }

                    double[][] subVP = new double[n][n - k - 1];
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < n - k - 1; j++) {
                            for (int l = 0; l < n - k - 1; l++) {
                                subVP[i][j] += subV[i][l] * P[l][j];
                            }
                        }
                    }

                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < n - k - 1; j++) {
                            V[i][k + 1 + j] = subVP[i][j];
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