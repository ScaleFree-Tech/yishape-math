package com.yishape.lab.math.linalg.decomposition.solver;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

/**
 * Solver for tridiagonal decomposition.
 * <p>
 * This solver uses the tridiagonal decomposition A = Q * T * Q^T to solve
 * linear systems. The tridiagonal form allows for more efficient solution
 * of linear systems compared to the general case.
 * </p>
 * 
 * <h3>References</h3>
 * <ul>
 *   <li>Golub, G. H., &amp; Van Loan, C. F. (2013). Matrix computations (4th ed.). Johns Hopkins University Press.</li>
 * </ul>
 *
 * @author RereMouse
 * @version 1.0
 * @since 2.0
 */
public class TridiagonalDecompositionSolver implements IDecompositionSolver {
    
    /** The tridiagonal matrix T. */
    private final IMatrix<Double> tMatrix;
    /** The orthogonal matrix Q. */
    private final IMatrix<Double> qMatrix;
    /** Epsilon for numerical comparisons. */
    private final double epsilon;
    
    /**
     * Create a solver from tridiagonal decomposition results.
     * 
     * @param tMatrix the tridiagonal matrix T
     * @param qMatrix the orthogonal matrix Q
     * @param epsilon threshold for considering an element as zero
     */
    public TridiagonalDecompositionSolver(IMatrix<Double> tMatrix, IMatrix<Double> qMatrix, double epsilon) {
        this.tMatrix = tMatrix;
        this.qMatrix = qMatrix;
        this.epsilon = epsilon;
    }
    
    @Override
    public IMatrix<Double> solve(IMatrix<Double> b) {
        // For tridiagonal decomposition A = Q * T * Q^T, to solve A * X = B:
        // 1. Compute Y = Q^T * B
        // 2. Solve T * Z = Y for Z (using Thomas algorithm for tridiagonal systems)
        // 3. Compute X = Q * Z
        
        // Step 1: Y = Q^T * B
        IMatrix<Double> y = qMatrix.transpose().mmul(b);
        
        // Step 2: Solve T * Z = Y for Z using tridiagonal solver
        IMatrix<Double> z = solveTridiagonalSystem(y);
        
        // Step 3: X = Q * Z
        return qMatrix.mmul(z);
    }
    
    @Override
    public IVector<Double> solve(IVector<Double> b) {
        // Convert vector to matrix, solve, then convert back
        IMatrix<Double> bMatrix = Linalg.matrix(new double[][]{b.toDoubleArray()}).transpose();
        IMatrix<Double> xMatrix = solve(bMatrix);
        double[] xArray = new double[xMatrix.rows()];
        for (int i = 0; i < xArray.length; i++) {
            xArray[i] = xMatrix.get(i, 0);
        }
        return Linalg.vector(xArray);
    }
    
    @Override
    public boolean isNonSingular() {
        // Check if all diagonal elements of T are non-zero
        for (int i = 0; i < tMatrix.rows(); i++) {
            if (Math.abs(tMatrix.get(i, i)) < epsilon) {
                return false;
            }
        }
        return true;
    }
    
    @Override
    public IMatrix<Double> getInverse() {
        // Create identity matrix of appropriate size
        int size = tMatrix.rows();
        IMatrix<Double> identity = Linalg.eye(size);
        
        // Solve A * X = I for X
        return solve(identity);
    }
    
    /**
     * Solve a tridiagonal linear system T * X = B using the Thomas algorithm.
     * 
     * @param b Right-hand side matrix B
     * @return Solution matrix X
     */
    private IMatrix<Double> solveTridiagonalSystem(IMatrix<Double> b) {
        int n = tMatrix.rows();
        int m = b.cols();
        
        // Extract tridiagonal elements
        double[][] tData = tMatrix.toDoubleArray();
        double[][] bData = b.toDoubleArray();
        double[][] xData = new double[n][m];
        
        // Copy b to x as working space
        for (int i = 0; i < n; i++) {
            System.arraycopy(bData[i], 0, xData[i], 0, m);
        }
        
        // Extract diagonal, subdiagonal, and superdiagonal
        double[] a = new double[n]; // subdiagonal
        double[] b_diag = new double[n]; // diagonal
        double[] c = new double[n]; // superdiagonal
        
        for (int i = 0; i < n; i++) {
            b_diag[i] = tData[i][i];
            if (i > 0) {
                a[i] = tData[i][i - 1];
            }
            if (i < n - 1) {
                c[i] = tData[i][i + 1];
            }
        }
        
        // Thomas algorithm for each column of the right-hand side
        for (int col = 0; col < m; col++) {
            // Forward elimination
            double[] cp = new double[n];
            double[] dp = new double[n];
            
            cp[0] = c[0] / b_diag[0];
            dp[0] = xData[0][col] / b_diag[0];
            
            for (int i = 1; i < n; i++) {
                double denom = b_diag[i] - a[i] * cp[i - 1];
                if (Math.abs(denom) < epsilon) {
                    throw new RuntimeException("Matrix is singular: zero denominator in Thomas algorithm at index " + i);
                }
                cp[i] = c[i] / denom;
                dp[i] = (xData[i][col] - a[i] * dp[i - 1]) / denom;
            }
            
            // Back substitution
            xData[n - 1][col] = dp[n - 1];
            for (int i = n - 2; i >= 0; i--) {
                xData[i][col] = dp[i] - cp[i] * xData[i + 1][col];
            }
        }
        
        // Convert back to IMatrix
        return Linalg.matrix(xData);
    }
    
    /**
     * Get the tridiagonal matrix T.
     * 
     * @return the tridiagonal matrix
     */
    public IMatrix<Double> getT() {
        return tMatrix;
    }
    
    /**
     * Get the orthogonal matrix Q.
     * 
     * @return the orthogonal matrix
     */
    public IMatrix<Double> getQ() {
        return qMatrix;
    }
}