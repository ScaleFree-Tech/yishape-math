package com.yishape.lab.math.linalg.decomposition.solver;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

/**
 * Solver for QR decomposition algorithms.
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
        double[] bArray = b.toDoubleArray();
        
        final int n = qrt.length;
        final int m = qrt[0].length;
        if (bArray.length != m) {
            throw new IllegalArgumentException("Vector dimension mismatch: expected " + m + ", got " + bArray.length);
        }
        checkSingular(rDiag, threshold, true);

        final double[] x = new double[n];
        final double[] y = bArray.clone();

        // apply Householder transforms to solve Q.y = b
        for (int minor = 0; minor < Math.min(m, n); minor++) {
            final double[] qrtMinor = qrt[minor];
            double dotProduct = 0;
            for (int row = minor; row < m; row++) {
                dotProduct += y[row] * qrtMinor[row];
            }
            dotProduct /= rDiag[minor] * qrtMinor[minor];

            for (int row = minor; row < m; row++) {
                y[row] += dotProduct * qrtMinor[row];
            }
        }

        // solve triangular system R.x = y
        for (int row = rDiag.length - 1; row >= 0; --row) {
            y[row] /= rDiag[row];
            final double yRow = y[row];
            final double[] qrtRow = qrt[row];
            x[row] = yRow;
            for (int i = 0; i < row; i++) {
                y[i] -= yRow * qrtRow[i];
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

        // Convert matrix to double[][] array using toDoubleArray method
        double[][] bData = b.toDoubleArray();

        final int columns = bCols;
        final double[][] x = new double[n][columns];

        // apply Householder transforms to solve Q.Y = B
        for (int minor = 0; minor < Math.min(m, n); minor++) {
            final double[] qrtMinor = qrt[minor];
            
            for (int col = 0; col < columns; col++) {
                double dotProduct = 0;
                for (int row = minor; row < m; row++) {
                    dotProduct += bData[row][col] * qrtMinor[row];
                }
                dotProduct /= rDiag[minor] * qrtMinor[minor];

                for (int row = minor; row < m; row++) {
                    bData[row][col] += dotProduct * qrtMinor[row];
                }
            }
        }

        // solve triangular system R.X = Y
        for (int row = rDiag.length - 1; row >= 0; --row) {
            final double[] qrtRow = qrt[row];
            
            for (int col = 0; col < columns; col++) {
                bData[row][col] /= rDiag[row];
                final double yRow = bData[row][col];
                x[row][col] = yRow;
                for (int i = 0; i < row; i++) {
                    bData[i][col] -= yRow * qrtRow[i];
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