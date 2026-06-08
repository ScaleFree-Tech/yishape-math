package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.DoubleMatrixView;
import com.yishape.lab.math.linalg.RereDoubleMatrix;
import com.yishape.lab.math.linalg.decomposition.ICholeskyDecomposition;
import com.yishape.lab.math.linalg.decomposition.solver.CholeskyDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.NonSquareMatrixException;
import com.yishape.lab.math.linalg.decomposition.NonSymmetricMatrixException;
import com.yishape.lab.math.linalg.decomposition.NonPositiveDefiniteMatrixException;
import com.yishape.lab.math.linalg.decomposition.DecompositionDenseAccess;

/**
 * Implementation of Cholesky decomposition with enhanced numerical stability.
 * <p>
 * This implementation computes the Cholesky decomposition of a symmetric,
 * positive definite matrix A such that A = L * L^T where L is a lower triangular matrix.
 * </p>
 * 
 * <h3>Algorithm Improvements</h3>
 * <ul>
 *   <li>Enhanced symmetry checking with configurable thresholds</li>
 *   <li>Improved positive definiteness verification</li>
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
public class RereCholeskyDecomposition implements ICholeskyDecomposition {
    
    /** Cached value of L. */
    private IMatrix<Double> cachedL;
    /** Cached value of LT. */
    private IMatrix<Double> cachedLT;
    /** Row-oriented storage for LT matrix data. */
    private double[][] lTData;
    /** Relative symmetry threshold. */
    private double relativeSymmetryThreshold;
    /** Absolute positivity threshold. */
    private double absolutePositivityThreshold;
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
     * Default constructor with default thresholds
     */
    public RereCholeskyDecomposition() {
        this.relativeSymmetryThreshold = DEFAULT_RELATIVE_SYMMETRY_THRESHOLD;
        this.absolutePositivityThreshold = DEFAULT_ABSOLUTE_POSITIVITY_THRESHOLD;
        this.epsilon = DEFAULT_EPSILON;
        this.maxIterations = DEFAULT_MAX_ITERATIONS;
    }
    
    /**
     * Constructor with configurable thresholds
     * 
     * @param relativeSymmetryThreshold threshold above which off-diagonal
     * elements are considered too different and matrix not symmetric
     * @param absolutePositivityThreshold threshold below which diagonal
     * elements are considered null and matrix not positive definite
     */
    public RereCholeskyDecomposition(double relativeSymmetryThreshold, double absolutePositivityThreshold) {
        this.relativeSymmetryThreshold = relativeSymmetryThreshold;
        this.absolutePositivityThreshold = absolutePositivityThreshold;
        this.epsilon = DEFAULT_EPSILON;
        this.maxIterations = DEFAULT_MAX_ITERATIONS;
    }
    
    /**
     * Constructor with unified parameters
     * 
     * @param epsilon threshold for considering an element as zero
     * @param maxIterations maximum number of iterations
     */
    public RereCholeskyDecomposition(double epsilon, int maxIterations) {
        this.relativeSymmetryThreshold = epsilon;
        this.absolutePositivityThreshold = epsilon;
        this.epsilon = epsilon;
        this.maxIterations = maxIterations;
    }
    
    @Override
    public IMatrix<Double> decompose(IMatrix<Double> matrix) {
        return decompose(matrix, relativeSymmetryThreshold, absolutePositivityThreshold);
    }
    
    @Override
    public IMatrix<Double> decompose(IMatrix<Double> matrix, double epsilon) {
        return decompose(matrix, epsilon, DEFAULT_MAX_ITERATIONS);
    }
    
    @Override
    public IMatrix<Double> decompose(IMatrix<Double> matrix, double epsilon, int maxIterations) {
        return decompose(matrix, epsilon, epsilon); // Use epsilon for both thresholds
    }
    
    @Override
    public IMatrix<Double> decompose(IMatrix<Double> matrix, 
                                   double relativeSymmetryThreshold,
                                   double absolutePositivityThreshold) {
        // Reset cached values
        cachedL = null;
        cachedLT = null;
        determinant = null;
        conditionNumber = null;
        rank = null;
        this.relativeSymmetryThreshold = relativeSymmetryThreshold;
        this.absolutePositivityThreshold = absolutePositivityThreshold;
        
        // Get matrix dimensions
        int rows = matrix.rows();
        int cols = matrix.cols();
        
        if (rows != cols) {
            throw new NonSquareMatrixException(
                "Only square matrices can perform Cholesky decomposition",
                "Cholesky Decomposition", 
                "Matrix " + rows + "x" + cols,
                rows, cols);
        }

        int n = rows;
        lTData = new double[n][n];

        DecompositionDenseAccess.copyInto(matrix, lTData, n, n);
        
        // Check symmetry after all data has been copied
        double maxAsymmetry = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) { // Only check upper triangle
                // Check symmetry with configurable threshold
                double asymmetry = Math.abs(lTData[i][j] - lTData[j][i]);
                maxAsymmetry = Math.max(maxAsymmetry, asymmetry);
                if (asymmetry > 
                    relativeSymmetryThreshold * Math.max(Math.abs(lTData[i][j]), Math.abs(lTData[j][i]))) {
                    throw new NonSymmetricMatrixException(
                        "Matrix must be symmetric for Cholesky decomposition",
                        "Cholesky Decomposition",
                        "Matrix " + n + "x" + n,
                        relativeSymmetryThreshold,
                        maxAsymmetry);
                }
            }
        }

        // Right-looking in-place Cholesky on the upper triangle of lTData.
        // After completion lTData[i][j] = LT[i][j] = L[j][i] for i <= j,
        // so getLT() returns the Cholesky L^T factor directly.
        // Zero extra allocation — eliminates the left-looking L[n][n] + O(n²) transpose.
        double minDiagonal = Double.POSITIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            final double[] ltI = lTData[i];

            if (ltI[i] <= absolutePositivityThreshold) {
                throw new NonPositiveDefiniteMatrixException(
                    "Matrix is not positive definite for Cholesky decomposition",
                    "Cholesky Decomposition",
                    "Matrix " + n + "x" + n,
                    absolutePositivityThreshold,
                    ltI[i]);
            }
            minDiagonal = Math.min(minDiagonal, ltI[i]);

            ltI[i] = Math.sqrt(ltI[i]);
            if (ltI[i] <= absolutePositivityThreshold) {
                throw new NonPositiveDefiniteMatrixException(
                    "Cholesky diagonal too small after sqrt at row " + i,
                    "Cholesky Decomposition",
                    "Matrix " + n + "x" + n,
                    absolutePositivityThreshold,
                    ltI[i]);
            }
            final double inv = 1.0 / ltI[i];

            // Scale row i to the right of the diagonal: L[j][i] for j > i
            for (int q = i + 1; q < n; q++) {
                ltI[q] *= inv;
            }

            // Rank-1 update on trailing submatrix (upper triangle only)
            for (int q = i + 1; q < n; q++) {
                final double[] ltQ = lTData[q];
                final double lik = ltI[q];
                for (int p = q; p < n; p++) {
                    ltQ[p] -= lik * ltI[p];
                }
            }
        }

        // LT is upper-triangular: zero the lower triangle (below diagonal).
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                lTData[i][j] = 0.0;
            }
        }

        // Construct L from the upper triangle of lTData (which stores LT).
        // L[i][j] = LT[j][i] = lTData[j][i] for j <= i.
        double[][] L = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                L[i][j] = lTData[j][i];
            }
        }
        return IMatrix.of(L);
    }

    @Override
    public IMatrix<Double> decomposeInPlace(IMatrix<Double> matrix) {
        if (!(matrix instanceof RereDoubleMatrix dm)) {
            return decompose(matrix);
        }

        cachedL = null;
        cachedLT = null;
        determinant = null;
        conditionNumber = null;
        rank = null;

        int rows = matrix.rows();
        int cols = matrix.cols();
        if (rows != cols) {
            throw new NonSquareMatrixException(
                "Only square matrices can perform Cholesky decomposition",
                "Cholesky Decomposition",
                "Matrix " + rows + "x" + cols,
                rows, cols);
        }

        int n = rows;
        double[][] data = dm.getData();

        // Check symmetry on original data
        double maxAsymmetry = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double asymmetry = Math.abs(data[i][j] - data[j][i]);
                maxAsymmetry = Math.max(maxAsymmetry, asymmetry);
                if (asymmetry >
                    relativeSymmetryThreshold * Math.max(Math.abs(data[i][j]), Math.abs(data[j][i]))) {
                    throw new NonSymmetricMatrixException(
                        "Matrix must be symmetric for Cholesky decomposition",
                        "Cholesky Decomposition",
                        "Matrix " + n + "x" + n,
                        relativeSymmetryThreshold,
                        maxAsymmetry);
                }
            }
        }

        // Right-looking in-place Cholesky on the upper triangle of data.
        // After completion data[i][j] = L[j][i] for i <= j (i.e. LT stored upper).
        double minDiagonal = Double.POSITIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            final double[] ltI = data[i];

            if (ltI[i] <= absolutePositivityThreshold) {
                throw new NonPositiveDefiniteMatrixException(
                    "Matrix is not positive definite for Cholesky decomposition",
                    "Cholesky Decomposition",
                    "Matrix " + n + "x" + n,
                    absolutePositivityThreshold,
                    ltI[i]);
            }
            minDiagonal = Math.min(minDiagonal, ltI[i]);

            ltI[i] = Math.sqrt(ltI[i]);
            final double inv = 1.0 / ltI[i];

            for (int q = i + 1; q < n; q++) {
                ltI[q] *= inv;
            }

            for (int q = i + 1; q < n; q++) {
                final double[] ltQ = data[q];
                final double lik = ltI[q];
                for (int p = q; p < n; p++) {
                    ltQ[p] -= lik * ltI[p];
                }
            }
        }
        // Zero lower triangle (below diagonal) — LT is upper-triangular.
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                data[i][j] = 0.0;
            }
        }

        // data upper triangle already stores LT; copy directly (no transpose needed).
        lTData = new double[n][n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(data[i], 0, lTData[i], 0, n);
        }

        // Transpose L^T from upper triangle to lower triangle so the returned view IS L.
        // data[i][j] for i <= j contains L[j][i] (= L^T element).
        // Move to data[j][i] so that column i of the lower triangle stores L's column i.
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                data[j][i] = data[i][j]; // L[i][j] from L^T[j][i]
                data[i][j] = 0.0;        // zero upper triangle
            }
        }

        // View: lower triangle of the mutated input IS L
        return new DoubleMatrixView(dm, 0, 0, n, n);
    }

    @Override
    public IMatrix<Double> getL() {
        if (cachedL == null) {
            if (lTData == null) {
                throw new IllegalStateException("Decomposition not yet performed");
            }
            // Recreate L from LT (which is stored in lTData)
            int n = lTData.length;
            double[][] L = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    L[i][j] = lTData[j][i]; // Transpose of LT gives us L
                }
            }
            // Use IMatrix factory method instead of direct instantiation
            cachedL = IMatrix.of(L);
        }
        return cachedL;
    }
    
    @Override
    public IMatrix<Double> getLT() {
        if (cachedLT == null) {
            if (lTData == null) {
                throw new IllegalStateException("Decomposition not yet performed");
            }
            // Use IMatrix factory method instead of direct instantiation
            cachedLT = IMatrix.of(lTData);
        }
        return cachedLT;
    }
    
    @Override
    public double getDeterminant() {
        if (determinant == null) {
            if (lTData == null) {
                throw new IllegalStateException("Decomposition not yet performed");
            }
            double det = 1.0;
            for (int i = 0; i < lTData.length; ++i) {
                double lTii = lTData[i][i];
                det *= lTii * lTii;
            }
            determinant = det;
        }
        return determinant;
    }
    
    @Override
    public boolean isNonSingular() {
        if (lTData == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        // A matrix is non-singular if all diagonal elements of L are non-zero
        for (int i = 0; i < lTData.length; i++) {
            if (Math.abs(lTData[i][i]) < epsilon) {
                return false;
            }
        }
        return true;
    }
    
    @Override
    public double getConditionNumber() {
        if (conditionNumber == null) {
            if (lTData == null) {
                throw new IllegalStateException("Decomposition not yet performed");
            }
            // Estimate condition number using the ratio of largest to smallest diagonal elements
            double maxDiag = 0.0;
            double minDiag = Double.POSITIVE_INFINITY;
            for (int i = 0; i < lTData.length; i++) {
                double diag = Math.abs(lTData[i][i]);
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
            if (lTData == null) {
                throw new IllegalStateException("Decomposition not yet performed");
            }
            // Rank is the number of non-zero diagonal elements
            int r = 0;
            for (int i = 0; i < lTData.length; i++) {
                if (Math.abs(lTData[i][i]) > epsilon) {
                    r++;
                }
            }
            rank = r;
        }
        return rank;
    }
    
    @Override
    public IDecompositionSolver getSolver() {
        if (lTData == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        return new CholeskyDecompositionSolver(lTData);
    }
    
    @Override
    public double getRelativeSymmetryThreshold() {
        return relativeSymmetryThreshold;
    }
    
    @Override
    public double getAbsolutePositivityThreshold() {
        return absolutePositivityThreshold;
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