package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.decomposition.ITridiagonalDecomposition;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.solver.TridiagonalDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.NonSquareMatrixException;
import com.yishape.lab.math.linalg.decomposition.NonSymmetricMatrixException;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.math.util.RerePrecision;

/**
 * Tridiagonal decomposition implementation with enhanced numerical stability.
 * <p>
 * Tridiagonal decomposition transforms a symmetric matrix A into the form A = Q * T * Q^T,
 * where T is a tridiagonal matrix, and Q is an orthogonal matrix.
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
public class RereTridiagonalDecomposition implements ITridiagonalDecomposition {
    
    /** Cached value of T. */
    private IMatrix<Double> cachedT;
    /** Cached value of Q. */
    private IMatrix<Double> cachedQ;
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
    public RereTridiagonalDecomposition() {
        this.epsilon = DEFAULT_EPSILON;
        this.maxIterations = DEFAULT_MAX_ITERATIONS;
    }
    
    /**
     * Constructor with unified parameters.
     * 
     * @param epsilon threshold for considering an element as zero
     * @param maxIterations maximum number of iterations
     */
    public RereTridiagonalDecomposition(double epsilon, int maxIterations) {
        this.epsilon = epsilon;
        this.maxIterations = maxIterations;
    }
    
    @Override
    public Tuple2<IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix) {
        return decompose(matrix, epsilon);
    }
    
    @Override
    public Tuple2<IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix, double epsilon) {
        return decompose(matrix, epsilon, DEFAULT_MAX_ITERATIONS);
    }
    
    @Override
    public Tuple2<IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix, double epsilon, int maxIterations) {
        // Reset cached values
        cachedT = null;
        cachedQ = null;
        determinant = null;
        conditionNumber = null;
        rank = null;
        this.epsilon = epsilon;
        this.maxIterations = maxIterations;
        
        IDoubleMatrix doubleMatrix = (IDoubleMatrix) matrix;
        double[][] data = doubleMatrix.getData();
        int n = data.length;
        
        // Check if matrix is square
        if (data.length != data[0].length) {
            throw new NonSquareMatrixException(
                "Tridiagonal decomposition requires square matrix",
                "Tridiagonal Decomposition", 
                "Matrix " + data.length + "x" + data[0].length,
                data.length, data[0].length);
        }
        
        // Check if matrix is symmetric
        double maxAsymmetry = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) { // Only check upper triangle
                double asymmetry = Math.abs(data[i][j] - data[j][i]);
                maxAsymmetry = Math.max(maxAsymmetry, asymmetry);
                if (asymmetry > epsilon) {
                    throw new NonSymmetricMatrixException(
                        "Matrix must be symmetric for tridiagonal decomposition",
                        "Tridiagonal Decomposition",
                        "Matrix " + n + "x" + n,
                        epsilon,
                        maxAsymmetry);
                }
            }
        }
        
        double[][] T = new double[n][n];
        double[][] Q = new double[n][n];

        // Copy original matrix
        for (int i = 0; i < n; i++) {
            System.arraycopy(data[i], 0, T[i], 0, n);
        }

        // Initialize Q as identity matrix
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Q[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }

        // For small matrices, return directly
        if (n <= 2) {
            cachedT = IDoubleMatrix.of(T);
            cachedQ = IDoubleMatrix.of(Q);
            return new Tuple2<>(cachedT, cachedQ);
        }

        // Column buffer for stride-1 Householder computation (see Hessenberg).
        double[] colBuf = new double[n];
        // Pre-allocated Householder vector buffer reused across iterations.
        double[] v = new double[n];

        // Apply Householder transformations to each column
        for (int k = 0; k < n - 2; k++) {
            int len = n - k - 1;

            // Compute Householder vector v directly from column k
            double norm = 0.0;
            for (int i = 0; i < len; i++) {
                double val = T[k + 1 + i][k];
                v[i] = val;
                norm += val * val;
            }
            norm = Math.sqrt(norm);

            if (RerePrecision.equalsZero(norm, epsilon)) {
                continue;
            }

            // v[0] = x[0] + sign(x[0]) * norm (the remaining v[i]=x[i] are already set)
            v[0] += (v[0] >= 0 ? norm : -norm);

            // Normalize v
            double vNorm = 0.0;
            for (int i = 0; i < len; i++) {
                vNorm += v[i] * v[i];
            }
            vNorm = Math.sqrt(vNorm);

            if (RerePrecision.equalsZero(vNorm, epsilon)) {
                continue;
            }

            for (int i = 0; i < len; i++) {
                v[i] /= vNorm;
            }

            // Efficiently apply Householder transformation: avoid constructing full matrix
            // Apply left multiplication: T = (I - 2vv^T) * T
            // Column extraction into colBuf decouples the stride-n column scan from
            // the arithmetic, enabling stride-1 dot/axpy on L1-cached data.
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

            // Apply right multiplication: T = T * (I - 2vv^T)
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int j = 0; j < len; j++) {
                    sum += T[i][k + 1 + j] * v[j];
                }
                for (int j = 0; j < len; j++) {
                    T[i][k + 1 + j] -= 2.0 * sum * v[j];
                }
            }

            // Update Q matrix: Q = Q * (I - 2vv^T)
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int j = 0; j < len; j++) {
                    sum += Q[i][k + 1 + j] * v[j];
                }
                for (int j = 0; j < len; j++) {
                    Q[i][k + 1 + j] -= 2.0 * sum * v[j];
                }
            }
        }

        // Ensure tridiagonal form: clean numerical errors
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (Math.abs(i - j) > 1 && RerePrecision.equalsZero(T[i][j], epsilon)) {
                    T[i][j] = 0.0;
                }
            }
        }

        cachedT = IDoubleMatrix.of(T);
        cachedQ = IDoubleMatrix.of(Q);
        return new Tuple2<>(cachedT, cachedQ);
    }
    
    @Override
    public IDecompositionSolver getSolver() {
        if (cachedT == null || cachedQ == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        // Return the standalone solver
        return new TridiagonalDecompositionSolver(cachedT, cachedQ, epsilon);
    }
    
    @Override
    public double getDeterminant() {
        if (determinant == null) {
            if (cachedT == null) {
                throw new IllegalStateException("Decomposition not yet performed");
            }
            double det = 1.0;
            for (int i = 0; i < cachedT.rows(); ++i) {
                det *= cachedT.get(i, i);
            }
            determinant = det;
        }
        return determinant;
    }
    
    @Override
    public boolean isNonSingular() {
        if (cachedT == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        // A matrix is non-singular if all diagonal elements of T are non-zero
        for (int i = 0; i < cachedT.rows(); i++) {
            if (Math.abs(cachedT.get(i, i)) < epsilon) {
                return false;
            }
        }
        return true;
    }
    
    @Override
    public double getConditionNumber() {
        if (conditionNumber == null) {
            if (cachedT == null) {
                throw new IllegalStateException("Decomposition not yet performed");
            }
            // Estimate condition number using the ratio of largest to smallest diagonal elements
            double maxDiag = 0.0;
            double minDiag = Double.POSITIVE_INFINITY;
            for (int i = 0; i < cachedT.rows(); i++) {
                double diag = Math.abs(cachedT.get(i, i));
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
            if (cachedT == null) {
                throw new IllegalStateException("Decomposition not yet performed");
            }
            // Rank is the number of non-zero diagonal elements
            int r = 0;
            for (int i = 0; i < cachedT.rows(); i++) {
                if (Math.abs(cachedT.get(i, i)) > epsilon) {
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
}