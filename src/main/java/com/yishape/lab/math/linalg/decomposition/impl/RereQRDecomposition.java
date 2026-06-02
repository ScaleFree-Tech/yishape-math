package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.IQRDecomposition;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.solver.QRDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.DecompositionDenseAccess;
import com.yishape.lab.math.linalg.decomposition.SegmentDoubleVector;
import com.yishape.lab.util.Tuple2;

/**
 * Implementation of QR decomposition using Householder reflections with enhanced numerical stability.
 * <p>
 * This implementation computes the QR decomposition of a matrix A such that A = Q * R where:
 * <ul>
 *   <li>Q is an orthogonal matrix (Q^T * Q = I)</li>
 *   <li>R is an upper triangular matrix</li>
 * </ul>
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
public class RereQRDecomposition implements IQRDecomposition {
    /** A packed TRANSPOSED representation of the QR decomposition. */
    protected double[][] qrt;
    /** The diagonal elements of R. */
    protected double[] rDiag;
    /** Cached value of Q. */
    private IMatrix<Double> cachedQ;
    /** Cached value of R. */
    private IMatrix<Double> cachedR;
    /** Cached value of QT. */
    private IMatrix<Double> cachedQT;
    /** Singularity threshold. */
    protected double threshold;
    /** Epsilon for numerical comparisons. */
    protected double epsilon;
    /** Maximum number of iterations. */
    private int maxIterations;
    /** Condition number of the matrix. */
    private Double conditionNumber;
    /** Rank of the matrix. */
    private Integer rank;
    /**
     * Default constructor with default threshold.
     */
    public RereQRDecomposition() {
        this.threshold = 1e-12;
        this.epsilon = DEFAULT_EPSILON;
        this.maxIterations = DEFAULT_MAX_ITERATIONS;
    }
    
    /**
     * Constructor with configurable threshold.
     * 
     * @param threshold singularity threshold
     */
    public RereQRDecomposition(double threshold) {
        this.threshold = threshold;
        this.epsilon = threshold;
        this.maxIterations = DEFAULT_MAX_ITERATIONS;
    }
    
    /**
     * Constructor with unified parameters.
     * 
     * @param epsilon threshold for considering an element as zero
     * @param maxIterations maximum number of iterations
     */
    public RereQRDecomposition(double epsilon, int maxIterations) {
        this.threshold = epsilon;
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
        cachedQ = null;
        cachedR = null;
        cachedQT = null;
        conditionNumber = null;
        rank = null;
        this.epsilon = epsilon;
        this.maxIterations = maxIterations;
        
        // Get matrix dimensions
        int rows = matrix.rows();
        int cols = matrix.cols();
        
        // Packed transposed representation: one allocation, avoid matrix.t() + second toDoubleArray() copy
        qrt = new double[cols][rows];
        if (matrix instanceof IDoubleMatrix dm) {
            double[][] src = dm.getData();
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    qrt[j][i] = src[i][j];
                }
            }
        } else {
            DecompositionDenseAccess.copyInto(matrix.transposeNew(), qrt, cols, rows);
        }
        
        rDiag = new double[Math.min(rows, cols)];

        // Perform Householder decomposition
        decompose(qrt);

        // Return Q and R matrices.
        // NOTE: Unlike CM4's QRDecomposition which builds Q/R lazily (only when getQ()/getR()
        // is called), we build them eagerly here because the API contract returns both.
        // When benchmarking against CM4, ensure the CM4 side also calls getQ()+getR()
        // inside the timed region — otherwise CM4 appears 2x faster because it only
        // measures Householder reduction (half the work).
        return new Tuple2<>(getQ(), getR());
    }
    
    /** Decompose matrix using Householder reflections. */
    protected void decompose(double[][] matrix) {
        for (int minor = 0; minor < Math.min(matrix.length, matrix[0].length); minor++) {
            performHouseholderReflection(minor, matrix);
        }
    }

    /**
     * Perform Householder reflection for a minor A(minor, minor) of A.
     *
     * <h3>PERFORMANCE NOTE — Why inline loops instead of SegmentDoubleVector</h3>
     * <p>This method previously used SegmentDoubleVector.dot()/axpy() which routes through
     * DoubleVectorComputer for SIMD dispatch. However, every dot/axpy call allocates
     * temporary contiguous arrays via toContiguous(). Inside this O(n³) loop, the number
     * of allocations scales cubically (millions for n=1000), and the segment lengths
     * decrease each iteration — too short to amortize SIMD dispatch overhead.</p>
     *
     * <p>Switching to inline double[] loops eliminated 12-24x QR decomposition overhead
     * vs CM4 (2026-05-15). The same rationale applies to getQT(). Do NOT reintroduce
     * SegmentDoubleVector or any per-iteration allocation inside these loops.</p>
     *
     * @see #getQT()
     */
    protected void performHouseholderReflection(int minor, double[][] matrix) {
        final double[] qrtMinor = matrix[minor];

        /*
         * Let x be the first column of the minor, and a^2 = |x|^2.
         * x will be in the positions qr[minor][minor] through qr[m][minor].
         * The first column of the transformed minor will be (a,0,0,..)'
         * The sign of a is chosen to be opposite to the sign of the first
         * component of x. Let's find a:
         */
        double xNormSqr = 0;
        for (int row = minor; row < qrtMinor.length; row++) {
            final double c = qrtMinor[row];
            xNormSqr += c * c;
        }
        
        // Check for zero vector
        if (xNormSqr < epsilon * epsilon) {
            rDiag[minor] = 0.0;
            return;
        }
        
        final double a = (qrtMinor[minor] > 0) ? -Math.sqrt(xNormSqr) : Math.sqrt(xNormSqr);
        rDiag[minor] = a;

        if (Math.abs(a) > epsilon) {
            /*
             * Calculate the normalized reflection vector v and transform
             * the first column. We know the norm of v beforehand: v = x-ae
             * so |v|^2 = <x-ae,x-ae> = <x,x>-2a<x,e>+a^2<e,e> =
             * a^2+a^2-2a<x,e> = 2a*(a - <x,e>).
             * Here <x, e> is now qr[minor][minor].
             * v = x-ae is stored in the column at qr:
             */
            qrtMinor[minor] -= a; // now |v|^2 = -2a*(qr[minor][minor])

            /*
             * Transform the rest of the columns of the minor:
             * They will be transformed by the matrix H = I-2vv'/|v|^2.
             * If x is a column vector of the minor, then
             * Hx = (I-2vv'/|v|^2)x = x-2vv'x/|v|^2 = x - 2<x,v>/|v|^2 v.
             * Therefore the transformation is easily calculated by
             * subtracting the column vector (2<x,v>/|v|^2)v from x.
             *
             * Let 2<x,v>/|v|^2 = alpha. From above we have
             * |v|^2 = -2a*(qr[minor][minor]), so
             * alpha = -<x,v>/(a*qr[minor][minor])
             */
            final int m = qrtMinor.length;
            final double denominator = a * qrtMinor[minor];
            final double invAbsDenom = (Math.abs(denominator) > epsilon) ? 1.0 / denominator : 0.0;
            for (int col = minor + 1; col < matrix.length; col++) {
                final double[] qrtCol = matrix[col];
                // Inline dot: alpha = -<x, v> = -sum(qrtCol[row] * qrtMinor[row])
                double dot = 0.0;
                for (int row = minor; row < m; row++) {
                    dot += qrtCol[row] * qrtMinor[row];
                }
                double alpha = -dot * invAbsDenom;
                // Inline axpy: x = x - alpha * v
                for (int row = minor; row < m; row++) {
                    qrtCol[row] -= alpha * qrtMinor[row];
                }
            }
        }
    }

    /**
     * Returns the matrix R of the decomposition.
     * <p>R is an upper-triangular matrix</p>
     * @return the R matrix
     */
    @Override
    public IMatrix<Double> getR() {
        if (cachedR == null) {
            // R is supposed to be m x n
            final int n = qrt.length;
            final int m = qrt[0].length;
            double[][] ra = new double[m][n];
            // copy the diagonal from rDiag and the upper triangle of qr
            for (int row = Math.min(m, n) - 1; row >= 0; row--) {
                ra[row][row] = rDiag[row];
                for (int col = row + 1; col < n; col++) {
                    ra[row][col] = qrt[col][row];
                }
            }
            
            // Convert to IMatrix using Linalg API
            cachedR = Linalg.matrix(ra);
        }
        return cachedR;
    }

    /**
     * Returns the matrix Q of the decomposition.
     * <p>Q is an orthogonal matrix</p>
     * @return the Q matrix
     */
    @Override
    public IMatrix<Double> getQ() {
        if (cachedQ == null) {
            IMatrix<Double> qt = getQT();
            cachedQ = qt.transpose();
        }
        return cachedQ;
    }

    /**
     * Returns the transpose of the matrix Q of the decomposition.
     * <p>Q is an orthogonal matrix</p>
     *
     * <h3>PERFORMANCE NOTE</h3>
     * <p>This method uses inline loops instead of SegmentDoubleVector for the same reason
     * as performHouseholderReflection — the O(n³) back-accumulation of Householder
     * reflectors would generate millions of temporary array allocations.</p>
     *
     * @return the transpose of the Q matrix, Q<sup>T</sup>
     */
    @Override
    public IMatrix<Double> getQT() {
        if (cachedQT == null) {
            // QT is supposed to be m x m
            final int n = qrt.length;
            final int m = qrt[0].length;
            
            double[][] qta = new double[m][m];

            /*
             * Q = Q1 Q2 ... Q_m, so Q is formed by first constructing Q_m and then
             * applying the Householder transformations Q_(m-1),Q_(m-2),...,Q1 in
             * succession to the result
             */
            for (int minor = m - 1; minor >= Math.min(m, n); minor--) {
                qta[minor][minor] = 1.0d;
            }

            for (int minor = Math.min(m, n)-1; minor >= 0; minor--){
                final double[] qrtMinor = qrt[minor];
                qta[minor][minor] = 1.0d;
                if (Math.abs(qrtMinor[minor]) > epsilon) {
                    final double denominator = rDiag[minor] * qrtMinor[minor];
                    final double invDenom = (Math.abs(denominator) > epsilon) ? 1.0 / denominator : 0.0;
                    for (int col = minor; col < m; col++) {
                        final double[] qtaCol = qta[col];
                        // Inline dot: alpha = -<x, v>
                        double dot = 0.0;
                        for (int row = minor; row < m; row++) {
                            dot += qtaCol[row] * qrtMinor[row];
                        }
                        double alpha = -dot * invDenom;
                        // Inline axpy: x = x - alpha * v
                        for (int row = minor; row < m; row++) {
                            qtaCol[row] -= alpha * qrtMinor[row];
                        }
                    }
                }
            }
            
            // Convert to IMatrix using Linalg API
            cachedQT = Linalg.matrix(qta);
        }
        return cachedQT;
    }
    
    /**
     * Get a solver for finding the A &times; X = B solution in least square sense.
     * @return a solver
     */
    @Override
    public IDecompositionSolver getSolver() {
        return new QRDecompositionSolver(qrt, rDiag, threshold);
    }
    
    /**
     * Calculate the determinant of the matrix.
     * @return determinant of the matrix
     */
    @Override
    public double getDeterminant() {
        if (rDiag == null) {
            throw new IllegalStateException("Decomposition not performed yet");
        }
        
        double determinant = 1.0;
        for (int i = 0; i < rDiag.length; i++) {
            determinant *= rDiag[i];
        }
        return determinant;
    }
    
    /**
     * Check if the decomposed matrix is non-singular.
     * @return true if the decomposed matrix is non-singular
     */
    @Override
    public boolean isNonSingular() {
        if (rDiag == null) {
            throw new IllegalStateException("Decomposition not performed yet");
        }
        
        // A matrix is non-singular if all diagonal elements of R are non-zero
        for (int i = 0; i < rDiag.length; i++) {
            if (Math.abs(rDiag[i]) < epsilon) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * Get the condition number of the matrix.
     * @return condition number of the matrix
     */
    @Override
    public double getConditionNumber() {
        if (conditionNumber == null) {
            if (rDiag == null) {
                throw new IllegalStateException("Decomposition not performed yet");
            }
            
            // Estimate condition number using the ratio of largest to smallest diagonal elements
            double maxDiag = 0.0;
            double minDiag = Double.POSITIVE_INFINITY;
            for (int i = 0; i < rDiag.length; i++) {
                double diag = Math.abs(rDiag[i]);
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
    
    /**
     * Get the rank of the matrix.
     * @return rank of the matrix
     */
    @Override
    public int getRank() {
        if (rank == null) {
            if (rDiag == null) {
                throw new IllegalStateException("Decomposition not performed yet");
            }
            
            // Rank is the number of non-zero diagonal elements
            int r = 0;
            for (int i = 0; i < rDiag.length; i++) {
                if (Math.abs(rDiag[i]) > epsilon) {
                    r++;
                }
            }
            rank = r;
        }
        return rank;
    }
    
    /**
     * Get the epsilon value used for numerical comparisons.
     * @return the epsilon value
     */
    @Override
    public double getEpsilon() {
        return epsilon;
    }
    
    /**
     * Get the maximum number of iterations allowed.
     * @return the maximum number of iterations
     */
    @Override
    public int getMaxIterations() {
        return maxIterations;
    }
}