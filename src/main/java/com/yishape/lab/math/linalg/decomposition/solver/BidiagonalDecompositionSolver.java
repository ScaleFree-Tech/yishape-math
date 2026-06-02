package com.yishape.lab.math.linalg.decomposition.solver;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.DecompositionDenseAccess;

/**
 * Solver for bidiagonal decomposition.
 * <p>
 * This solver uses the bidiagonal decomposition A = U * B * V^T to solve
 * linear systems. The bidiagonal form allows for more efficient solution
 * of linear systems compared to the general case, especially for least squares problems.
 * </p>
 *
 * <p><b>Shape contract (important):</b> In {@link com.yishape.lab.math.linalg.decomposition.impl.RereBidiagonalDecomposition},
 * the extracted block {@code B} is always <b>square</b> {@code min(m,n) × min(m,n)} even when the original
 * matrix {@code A} is {@code m × n} and non-square. This class therefore expects a <b>square</b> bidiagonal
 * {@code B} consistent with that decomposition. Passing a non-square {@code B} is unsupported and indicates
 * a mismatched constructor use—not missing support for non-square {@code A}.
 * General least squares for an arbitrary rectangular bidiagonal system belongs to dedicated LS/SVD APIs, not here.
 * </p>
 * 
 * <p>
 * For a linear system A * X = B where A = U * B * V^T:
 * <ol>
 *   <li>Compute Y = U^T * B</li>
 *   <li>Solve B * Z = Y for Z (solving bidiagonal system)</li>
 *   <li>Compute X = V * Z</li>
 * </ol>
 * </p>
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
public class BidiagonalDecompositionSolver implements IDecompositionSolver {
    
    /** The bidiagonal matrix B. */
    private final IMatrix<Double> bMatrix;
    /** The orthogonal matrix U. */
    private final IMatrix<Double> uMatrix;
    /** The orthogonal matrix V. */
    private final IMatrix<Double> vMatrix;
    /** Epsilon for numerical comparisons. */
    private final double epsilon;
    
    /**
     * Create a solver from bidiagonal decomposition results.
     * 
     * @param bMatrix the bidiagonal matrix B (must be square; same shape as produced by {@code RereBidiagonalDecomposition})
     * @param uMatrix the orthogonal matrix U
     * @param vMatrix the orthogonal matrix V
     * @param epsilon threshold for considering an element as zero
     */
    public BidiagonalDecompositionSolver(IMatrix<Double> bMatrix, IMatrix<Double> uMatrix, IMatrix<Double> vMatrix, double epsilon) {
        this.bMatrix = bMatrix;
        this.uMatrix = uMatrix;
        this.vMatrix = vMatrix;
        this.epsilon = epsilon;
    }
    
    @Override
    public IMatrix<Double> solve(IMatrix<Double> B) {
        // For bidiagonal decomposition A = U * B * V^T, to solve A * X = B:
        // 1. Compute Y = U^T * B
        // 2. Solve B * Z = Y for Z (solving bidiagonal system)
        // 3. Compute X = V * Z
        
        // Step 1: Y = U^T * B
        IMatrix<Double> y = uMatrix.transpose().mmul(B);
        
        // Step 2: Solve B * Z = Y for Z
        IMatrix<Double> z = solveBidiagonalSystem(y);
        
        // Step 3: X = V * Z
        return vMatrix.mmul(z);
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
        int minDim = Math.min(bMatrix.rows(), bMatrix.cols());
        for (int i = 0; i < minDim; i++) {
            if (Math.abs(bMatrix.get(i, i)) < epsilon) {
                return false;
            }
        }
        return true;
    }
    
    @Override
    public IMatrix<Double> getInverse() {
        // Create identity matrix of appropriate size
        int size = bMatrix.rows();
        IMatrix<Double> identity = Linalg.eye(size);
        
        // Solve A * X = I for X
        return solve(identity);
    }
    
    /**
     * Solve a bidiagonal linear system B * X = Y.
     * 
     * @param y Right-hand side matrix Y
     * @return Solution matrix X
     */
    private IMatrix<Double> solveBidiagonalSystem(IMatrix<Double> y) {
        int m = bMatrix.rows();
        int n = bMatrix.cols();
        int minDim = Math.min(m, n);
        int p = y.cols();
        
        // Coefficients and RHS are only read; avoid toDoubleArray() heap churn for dense matrices
        double[][] bData = DecompositionDenseAccess.rowsForReadOnly(bMatrix);
        double[][] yData = DecompositionDenseAccess.rowsForReadOnly(y);
        double[][] xData = new double[n][p];
        
        // Square B only (matches decomposition output). Non-square B is not a "todo":
        // use LS/SVD elsewhere if you need a general rectangular bidiagonal solve.
        if (m == n) {
            // Backward substitution for upper bidiagonal system
            for (int k = 0; k < p; k++) {
                for (int i = minDim - 1; i >= 0; i--) {
                    double sum = yData[i][k];
                    if (i + 1 < n) {
                        sum -= bData[i][i + 1] * xData[i + 1][k];
                    }
                    if (Math.abs(bData[i][i]) < epsilon) {
                        throw new RuntimeException("Matrix is singular: zero diagonal element at index " + i);
                    }
                    xData[i][k] = sum / bData[i][i];
                }
            }
        } else {
            throw new IllegalArgumentException(
                    "Bidiagonal B must be square (min(m,n)×min(m,n)), as produced by RereBidiagonalDecomposition.getSolver(). "
                            + "Non-square B is not supported here; for general least squares use the appropriate LS/SVD API.");
        }
        
        // Convert back to IMatrix
        return Linalg.matrix(xData);
    }
    
    /**
     * Get the bidiagonal matrix B.
     * 
     * @return the B matrix
     */
    public IMatrix<Double> getB() {
        return bMatrix;
    }
    
    /**
     * Get the orthogonal matrix U.
     * 
     * @return the U matrix
     */
    public IMatrix<Double> getU() {
        return uMatrix;
    }
    
    /**
     * Get the orthogonal matrix V.
     * 
     * @return the V matrix
     */
    public IMatrix<Double> getV() {
        return vMatrix;
    }
}