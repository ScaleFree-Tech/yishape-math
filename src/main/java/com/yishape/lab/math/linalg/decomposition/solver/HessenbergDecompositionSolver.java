package com.yishape.lab.math.linalg.decomposition.solver;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.DecompositionDenseAccess;

/**
 * Solver for Hessenberg decomposition.
 * <p>
 * This solver uses the Hessenberg decomposition A = Q * H * Q^T to solve
 * linear systems. The Hessenberg form allows for more efficient solution
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
public class HessenbergDecompositionSolver implements IDecompositionSolver {
    
    /** The Hessenberg matrix H. */
    private final IMatrix<Double> hMatrix;
    /** The orthogonal matrix Q. */
    private final IMatrix<Double> qMatrix;
    /** Epsilon for numerical comparisons. */
    private final double epsilon;
    
    /**
     * Create a solver from Hessenberg decomposition results.
     * 
     * @param hMatrix the Hessenberg matrix H
     * @param qMatrix the orthogonal matrix Q
     * @param epsilon threshold for considering an element as zero
     */
    public HessenbergDecompositionSolver(IMatrix<Double> hMatrix, IMatrix<Double> qMatrix, double epsilon) {
        this.hMatrix = hMatrix;
        this.qMatrix = qMatrix;
        this.epsilon = epsilon;
    }
    
    @Override
    public IMatrix<Double> solve(IMatrix<Double> b) {
        // For Hessenberg decomposition A = Q * H * Q^T, to solve A * X = B:
        // 1. Compute Y = Q^T * B
        // 2. Solve H * Z = Y for Z (using backward substitution since H is upper Hessenberg)
        // 3. Compute X = Q * Z
        
        // Step 1: Y = Q^T * B
        IMatrix<Double> y = qMatrix.transpose().mmul(b);
        
        // Step 2: Solve H * Z = Y for Z using Hessenberg solver
        IMatrix<Double> z = solveHessenbergSystem(y);
        
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
        // Check if all diagonal elements of H are non-zero
        for (int i = 0; i < hMatrix.rows(); i++) {
            if (Math.abs(hMatrix.get(i, i)) < epsilon) {
                return false;
            }
        }
        return true;
    }
    
    @Override
    public IMatrix<Double> getInverse() {
        // Create identity matrix of appropriate size
        int size = hMatrix.rows();
        IMatrix<Double> identity = Linalg.eye(size);
        
        // Solve A * X = I for X
        return solve(identity);
    }
    
    /**
     * Solve a Hessenberg linear system H * X = B.
     * Uses Gaussian elimination adapted for Hessenberg structure.
     * 
     * @param b Right-hand side matrix B
     * @return Solution matrix X
     */
    private IMatrix<Double> solveHessenbergSystem(IMatrix<Double> b) {
        int n = hMatrix.rows();
        int m = b.cols();
        
        // H 与分解缓存共享存储时必须拷贝后再消元，否则多次 solve 会破坏 cached H
        double[][] hData = DecompositionRhsCopy.mutableRowMajorCopy(hMatrix);
        double[][] xData = new double[n][m];

        DecompositionDenseAccess.copyInto(b, xData, n, m);
        
        // Forward elimination (adapted for Hessenberg structure)
        for (int i = 0; i < n - 1; i++) {
            double pivot = hData[i][i];
            
            if (Math.abs(pivot) < epsilon) {
                throw new RuntimeException("Matrix is singular: zero pivot at index " + i);
            }
            
            // Eliminate elements below the pivot (at most one due to Hessenberg structure)
            if (i + 1 < n && Math.abs(hData[i + 1][i]) > epsilon) {
                double factor = hData[i + 1][i] / pivot;
                
                // Update H matrix (only the affected elements)
                for (int j = i; j < Math.min(i + 3, n); j++) { // Hessenberg property
                    hData[i + 1][j] -= factor * hData[i][j];
                }
                
                // Update right-hand side
                for (int k = 0; k < m; k++) {
                    xData[i + 1][k] -= factor * xData[i][k];
                }
            }
        }
        
        // Backward substitution
        for (int i = n - 1; i >= 0; i--) {
            for (int k = 0; k < m; k++) {
                for (int j = i + 1; j < n; j++) {
                    xData[i][k] -= hData[i][j] * xData[j][k];
                }
                xData[i][k] /= hData[i][i];
            }
        }
        
        // Convert back to IMatrix
        return Linalg.matrix(xData);
    }
    
    /**
     * Get the Hessenberg matrix H.
     * 
     * @return the Hessenberg matrix
     */
    public IMatrix<Double> getH() {
        return hMatrix;
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