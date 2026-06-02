package com.yishape.lab.math.linalg.decomposition.solver;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

/**
 * Solver for QR decomposition algorithms.
 *
 * @author RereMouse
 * @version 1.0
 * @since 2.0
 */
public class QRDecompositionSolver implements IDecompositionSolver {
    /** A packed TRANSPOSED representation of the QR decomposition. */
    private final double[][] qrt;
    /** The diagonal elements of R. */
    private final double[] rDiag;
    /** Singularity threshold. */
    private final double threshold;
    
    /**
     * Build a solver from decomposed matrix.
     *
     * @param qrt Packed TRANSPOSED representation of the QR decomposition.
     * @param rDiag Diagonal elements of R.
     * @param threshold Singularity threshold.
     */
    public QRDecompositionSolver(final double[][] qrt, final double[] rDiag, final double threshold) {
        this.qrt = qrt;
        this.rDiag = rDiag;
        this.threshold = threshold;
    }
    
    /**
     * Build a solver from decomposed matrix with default threshold.
     *
     * @param qrt Packed TRANSPOSED representation of the QR decomposition.
     * @param rDiag Diagonal elements of R.
     */
    public QRDecompositionSolver(final double[][] qrt, final double[] rDiag) {
        this(qrt, rDiag, 0d);
    }

    /** {@inheritDoc} */
    @Override
    public boolean isNonSingular() {
        return !checkSingular(rDiag, threshold, false);
    }

    /** {@inheritDoc} */
    @Override
    public IVector<Double> solve(IVector<Double> b) {
        // Convert IVector to double array for computation using toDoubleArray method
        double[] y = b.toDoubleArray();
        
        final int n = qrt.length;
        final int m = qrt[0].length;
        if (y.length != m) {
            throw new IllegalArgumentException("Vector dimension mismatch: expected " + m + ", got " + y.length);
        }
        checkSingular(rDiag, threshold, true);

        final double[] x = new double[n];

        // apply Householder transforms to solve Q.y = b.
        // Inline loops replace SegmentDoubleVector.dot()/axpy() to avoid per-iteration
        // toContiguous() allocations (fixed 2026-05-16). Each dot/axpy was allocating
        // two double[] copies; inside getInverse() → solve(identity), this scales
        // cubically and produces ~1 GB of garbage at n=500.
        for (int minor = 0; minor < Math.min(m, n); minor++) {
            final double[] qrtMinor = qrt[minor];
            final double denom = rDiag[minor] * qrtMinor[minor];
            if (Math.abs(denom) < threshold) {
                throw new RuntimeException("Householder denominator too small at index " + minor);
            }
            final double invDenom = 1.0 / denom;

            double dotProduct = 0.0;
            for (int row = minor; row < m; row++) {
                dotProduct += y[row] * qrtMinor[row];
            }
            final double alpha = dotProduct * invDenom;
            for (int row = minor; row < m; row++) {
                y[row] += alpha * qrtMinor[row];
            }
        }

        // solve triangular system R.x = y
        for (int row = rDiag.length - 1; row >= 0; --row) {
            y[row] /= rDiag[row];
            final double yRow = y[row];
            final double[] qrtRow = qrt[row];
            x[row] = yRow;
            if (row > 0) {
                for (int i = 0; i < row; i++) {
                    y[i] -= yRow * qrtRow[i];
                }
            }
        }

        // Convert result back to IVector using Linalg API
        return Linalg.vector(x);
    }

    /** {@inheritDoc} */
    @Override
    public IMatrix<Double> solve(IMatrix<Double> b) {
        final int n = qrt.length;
        final int m = qrt[0].length;
        final int bRows = b.rows();
        final int bCols = b.cols();
        
        if (bRows != m) {
            throw new IllegalArgumentException("Matrix row dimension mismatch: expected " + m + ", got " + bRows);
        }
        checkSingular(rDiag, threshold, true);

        // 独立工作区：避免改写调用方矩阵 toDoubleArray() 可能共享的后备存储
        double[][] bData = DecompositionRhsCopy.mutableRowMajorCopy(b);

        final int columns = bCols;
        final double[][] x = new double[n][columns];

        // apply Householder transforms to solve Q.Y = B.
        // Inline loops replace gather+SegmentDoubleVector.dot()+axpy()+scatter to avoid
        // per-column toContiguous() allocations (fixed 2026-05-16). At n=500 getInverse()
        // each minor iteration processes 500 columns, and the qrt segment and colBuf
        // allocs per column produced ~1 GB of garbage.
        for (int minor = 0; minor < Math.min(m, n); minor++) {
            final double[] qrtMinor = qrt[minor];
            final double denom = rDiag[minor] * qrtMinor[minor];
            final double invDenom = 1.0 / denom;

            for (int col = 0; col < columns; col++) {
                // Inline dot: Σ bData[row][col] * qrtMinor[row]
                double dotProduct = 0.0;
                for (int row = minor; row < m; row++) {
                    dotProduct += bData[row][col] * qrtMinor[row];
                }
                final double alpha = dotProduct * invDenom;
                // Inline axpy: bData[row][col] += alpha * qrtMinor[row]
                for (int row = minor; row < m; row++) {
                    bData[row][col] += alpha * qrtMinor[row];
                }
            }
        }

        // solve triangular system R.X = Y
        for (int row = rDiag.length - 1; row >= 0; --row) {
            final double[] qrtRow = qrt[row];
            final double invDiag = 1.0 / rDiag[row];

            for (int col = 0; col < columns; col++) {
                bData[row][col] *= invDiag;
                final double yRow = bData[row][col];
                x[row][col] = yRow;
                if (row > 0) {
                    for (int i = 0; i < row; i++) {
                        bData[i][col] -= yRow * qrtRow[i];
                    }
                }
            }
        }

        // Convert result back to IMatrix using Linalg API
        return Linalg.matrix(x);
    }

    /**
     * {@inheritDoc}
     * @throws RuntimeException if the decomposed matrix is singular.
     */
    @Override
    public IMatrix<Double> getInverse() {
        // Create identity matrix of appropriate size using Linalg API
        int size = qrt[0].length;
        IMatrix<Double> identity = Linalg.eye(size);
        
        return solve(identity);
    }

    /**
     * Check singularity.
     *
     * @param diag Diagonal elements of the R matrix.
     * @param min Singularity threshold.
     * @param raise Whether to raise an exception if any element of the diagonal fails the check.
     * @return {@code true} if any element of the diagonal is smaller or equal to {@code min}.
     * @throws RuntimeException if the matrix is singular and {@code raise} is {@code true}.
     */
    private static boolean checkSingular(double[] diag, double min, boolean raise) {
        final int len = diag.length;
        for (int i = 0; i < len; i++) {
            final double d = diag[i];
            if (Math.abs(d) <= min) {
                if (raise) {
                    throw new RuntimeException("Matrix is singular with element " + d + " at index " + i + 
                                             " which is smaller than threshold " + min);
                } else {
                    return true;
                }
            }
        }
        return false;
    }
}