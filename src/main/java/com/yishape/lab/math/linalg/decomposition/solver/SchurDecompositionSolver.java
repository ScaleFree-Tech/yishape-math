package com.yishape.lab.math.linalg.decomposition.solver;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

/**
 * Solver for Schur decomposition.
 * <p>
 * This solver uses the Schur decomposition A = U * T * U^T to solve
 * linear systems. The Schur form allows for more efficient solution
 * of linear systems compared to the general case, especially for eigenvalue problems.
 * </p>
 * 
 * <p>
 * For a linear system A * X = B where A = U * T * U^T:
 * <ol>
 *   <li>Compute Y = U^T * B</li>
 *   <li>Solve T * Z = Y for Z (solving quasi-upper triangular system)</li>
 *   <li>Compute X = U * Z</li>
 * </ol>
 * </p>
 * 
 * <h3>References</h3>
 * <ul>
 *   <li>Golub, G. H., &amp; Van Loan, C. F. (2013). Matrix computations (4th ed.). Johns Hopkins University Press.</li>
 *   <li>Horn, R. A., &amp; Johnson, C. R. (2012). Matrix analysis (2nd ed.). Cambridge University Press.</li>
 *   <li>Watkins, D. S. (2007). The matrix eigenvalue problem: GR and Krylov subspace methods. SIAM.</li>
 * </ul>
 *
 * @author RereMouse
 * @version 1.0
 * @since 2.0
 */
public class SchurDecompositionSolver implements IDecompositionSolver {
    
    /** The quasi-upper triangular matrix T. */
    private final IMatrix<Double> tMatrix;
    /** The orthogonal matrix U. */
    private final IMatrix<Double> uMatrix;
    /** Epsilon for numerical comparisons. */
    private final double epsilon;
    
    /**
     * Create a solver from Schur decomposition results.
     * 
     * @param tMatrix the quasi-upper triangular matrix T
     * @param uMatrix the orthogonal matrix U
     * @param epsilon threshold for considering an element as zero
     */
    public SchurDecompositionSolver(IMatrix<Double> tMatrix, IMatrix<Double> uMatrix, double epsilon) {
        this.tMatrix = tMatrix;
        this.uMatrix = uMatrix;
        this.epsilon = epsilon;
    }
    
    /**
     * Create a solver from raw Schur decomposition data.
     * 
     * @param tData the quasi-upper triangular matrix T data
     * @param uData the orthogonal matrix U data
     * @param epsilon threshold for considering an element as zero
     */
    public SchurDecompositionSolver(double[][] tData, double[][] uData, double epsilon) {
        this.tMatrix = Linalg.matrix(tData);
        this.uMatrix = Linalg.matrix(uData);
        this.epsilon = epsilon;
    }
    
    @Override
    public IMatrix<Double> solve(IMatrix<Double> b) {
        // For Schur decomposition A = U * T * U^T, to solve A * X = B:
        // 1. Compute Y = U^T * B
        // 2. Solve T * Z = Y for Z (solving quasi-upper triangular system)
        // 3. Compute X = U * Z
        
        // Step 1: Y = U^T * B
        IMatrix<Double> y = uMatrix.transpose().mmul(b);
        
        // Step 2: Solve T * Z = Y for Z (solving quasi-upper triangular system)
        IMatrix<Double> z = solveQuasiUpperTriangularSystem(y);
        
        // Step 3: X = U * Z
        return uMatrix.mmul(z);
    }
    
    @Override
    public IVector<Double> solve(IVector<Double> b) {
        // Convert vector to matrix, solve, then convert back
        IMatrix<Double> bMatrix = Linalg.matrix(new double[][]{b.toDoubleArray()}).transpose();
        IMatrix<Double> xMatrix = solve(bMatrix);
        return xMatrix.getColumn(0);
    }
    
    @Override
    public boolean isNonSingular() {
        // Check if all diagonal elements are non-zero
        // For quasi-upper triangular matrix, we need to check both 1x1 and 2x2 blocks
        int n = tMatrix.rows();
        for (int i = 0; i < n; i++) {
            if (Math.abs(tMatrix.get(i, i)) < epsilon) {
                // Check if this is part of a 2x2 block with zero determinant
                if (i < n - 1 && Math.abs(tMatrix.get(i + 1, i)) > epsilon) {
                    // This is a 2x2 block
                    double a = tMatrix.get(i, i);
                    double b_elem = tMatrix.get(i, i + 1);
                    double c = tMatrix.get(i + 1, i);
                    double d = tMatrix.get(i + 1, i + 1);
                    double det = a * d - b_elem * c;
                    if (Math.abs(det) < epsilon) {
                        return false;
                    }
                    i++; // Skip the next element as it's part of this 2x2 block
                } else {
                    // This is a 1x1 block (diagonal element)
                    return false;
                }
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
     * Solve a quasi-upper triangular linear system T * X = B.
     * 
     * @param b Right-hand side matrix B
     * @return Solution matrix X
     */
    private IMatrix<Double> solveQuasiUpperTriangularSystem(IMatrix<Double> b) {
        int n = tMatrix.rows();
        int m = b.cols();
        
        // Convert to arrays for efficient computation
        double[][] tData = tMatrix.toDoubleArray();
        double[][] bData = b.toDoubleArray();
        double[][] xData = new double[n][m];
        
        // Copy b to x as working space
        for (int i = 0; i < n; i++) {
            System.arraycopy(bData[i], 0, xData[i], 0, m);
        }
        
        // Backward substitution for quasi-upper triangular system
        // We process from bottom to top, handling 1x1 and 2x2 blocks
        for (int i = n - 1; i >= 0; i--) {
            if (i > 0 && Math.abs(tData[i][i - 1]) > epsilon) {
                // This is a 2x2 block with the element below
                // Solve the 2x2 system:
                // [t[i-1][i-1]  t[i-1][i]] [x[i-1][k]] = [b[i-1][k]]
                // [t[i][i-1]    t[i][i]  ] [x[i][k]  ] = [b[i][k]  ]
                
                int im1 = i - 1; // i minus 1
                double a = tData[im1][im1];
                double b_elem = tData[im1][i];
                double c = tData[i][im1];
                double d = tData[i][i];
                
                // Process each column of the right-hand side
                for (int k = 0; k < m; k++) {
                    // Subtract contributions from already solved variables (to the right)
                    double b1 = xData[im1][k];
                    double b2 = xData[i][k];
                    for (int j = i + 1; j < n; j++) {
                        b1 -= tData[im1][j] * xData[j][k];
                        b2 -= tData[i][j] * xData[j][k];
                    }
                    
                    // Solve 2x2 system
                    double det = a * d - b_elem * c;
                    if (Math.abs(det) < epsilon) {
                        throw new RuntimeException("Matrix is singular: zero determinant in 2x2 block");
                    }
                    
                    xData[im1][k] = (d * b1 - b_elem * b2) / det;
                    xData[i][k] = (a * b2 - c * b1) / det;
                }
                
                // Skip the next iteration since we've processed two rows
                i--; // This will be decremented again by the for loop
            } else {
                // This is a 1x1 block (standard diagonal element)
                double diag = tData[i][i];
                if (Math.abs(diag) < epsilon) {
                    throw new RuntimeException("Matrix is singular: zero diagonal element at index " + i);
                }
                
                // Process each column of the right-hand side
                for (int k = 0; k < m; k++) {
                    // Subtract contributions from already solved variables (to the right)
                    for (int j = i + 1; j < n; j++) {
                        xData[i][k] -= tData[i][j] * xData[j][k];
                    }
                    // Solve for this variable
                    xData[i][k] /= diag;
                }
            }
        }
        
        // Convert back to IMatrix
        return Linalg.matrix(xData);
    }
    
    /**
     * Get the quasi-upper triangular matrix T.
     * 
     * @return the T matrix
     */
    public IMatrix<Double> getT() {
        return tMatrix;
    }
    
    /**
     * Get the orthogonal matrix U.
     * 
     * @return the U matrix
     */
    public IMatrix<Double> getU() {
        return uMatrix;
    }
}