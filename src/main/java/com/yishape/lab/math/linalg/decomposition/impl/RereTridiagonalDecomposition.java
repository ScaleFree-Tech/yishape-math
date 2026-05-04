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

        // Apply Householder transformations to each column
        for (int k = 0; k < n - 2; k++) {
            // Compute vector to eliminate
            double[] x = new double[n - k - 1];
            for (int i = k + 1; i < n; i++) {
                x[i - k - 1] = T[i][k];
            }

            double norm = 0.0;
            for (double v : x) {
                norm += v * v;
            }
            norm = Math.sqrt(norm);

            if (RerePrecision.equalsZero(norm, epsilon)) {
                continue; // Improve numerical stability
            }
            
            // Construct Householder vector
            double[] v = new double[n - k - 1];
            v[0] = x[0] + (x[0] >= 0 ? norm : -norm); // Improve sign selection
            for (int i = 1; i < v.length; i++) {
                v[i] = x[i];
            }

            // Normalize v
            double vNorm = 0.0;
            for (double vi : v) {
                vNorm += vi * vi;
            }
            vNorm = Math.sqrt(vNorm);

            if (RerePrecision.equalsZero(vNorm, epsilon)) {
                continue;
            }

            for (int i = 0; i < v.length; i++) {
                v[i] /= vNorm;
            }

            // Efficiently apply Householder transformation: avoid constructing full matrix
            // Apply left multiplication: T = (I - 2vv^T) * T
            for (int j = k; j < n; j++) {
                double sum = 0.0;
                for (int i = 0; i < v.length; i++) {
                    sum += v[i] * T[k + 1 + i][j];
                }
                for (int i = 0; i < v.length; i++) {
                    T[k + 1 + i][j] -= 2.0 * v[i] * sum;
                }
            }

            // Apply right multiplication: T = T * (I - 2vv^T)
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int j = 0; j < v.length; j++) {
                    sum += T[i][k + 1 + j] * v[j];
                }
                for (int j = 0; j < v.length; j++) {
                    T[i][k + 1 + j] -= 2.0 * sum * v[j];
                }
            }

            // Update Q matrix: Q = Q * (I - 2vv^T)
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int j = 0; j < v.length; j++) {
                    sum += Q[i][k + 1 + j] * v[j];
                }
                for (int j = 0; j < v.length; j++) {
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