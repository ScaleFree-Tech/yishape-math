package com.yishape.lab.math.linalg.decomposition.solver;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

/**
 * Solver for singular value decomposition.
 * <p>
 * This solver uses the SVD decomposition A = U * S * V^T to solve
 * linear systems. For a non-singular matrix, the solution to A * X = B is
 * X = V * S^(-1) * U^T * B.
 * </p>
 * 
 * <h3>References</h3>
 * <ul>
 *   <li>Golub, G. H., &amp; Van Loan, C. F. (2013). Matrix computations (4th ed.). Johns Hopkins University Press.</li>
 * </ul>
 * 
 * @since 2.0
 */
public class SVDDecompositionSolver implements IDecompositionSolver {
    
    /** The U matrix. */
    private final IMatrix<Double> uMatrix;
    /** The singular values. */
    private final IVector<Double> singularValues;
    /** The V^T matrix. */
    private final IMatrix<Double> vtMatrix;
    /** Epsilon for numerical comparisons. */
    private final double epsilon;
    
    /**
     * Create a solver from SVD decomposition results.
     * 
     * @param uMatrix the U matrix
     * @param singularValues the singular values
     * @param vtMatrix the V^T matrix
     * @param epsilon threshold for considering an element as zero
     */
    public SVDDecompositionSolver(IMatrix<Double> uMatrix, IVector<Double> singularValues, IMatrix<Double> vtMatrix, double epsilon) {
        this.uMatrix = uMatrix;
        this.singularValues = singularValues;
        this.vtMatrix = vtMatrix;
        this.epsilon = epsilon;
    }
    
    @Override
    public IMatrix<Double> solve(IMatrix<Double> b) {
        // For SVD decomposition A = U * S * V^T, to solve A * X = B:
        // 1. Compute Y = U^T * B
        // 2. Solve S * Z = Y for Z (element-wise division by singular values)
        // 3. Compute X = V * Z
        
        // Check for singularity first
        if (!isNonSingular()) {
            throw new RuntimeException("Matrix is singular");
        }
        
        // Step 1: Y = U^T * B
        IMatrix<Double> y = uMatrix.transpose().mmul(b);
        
        // Step 2: Solve S * Z = Y for Z (element-wise division)
        IMatrix<Double> z = solveDiagonalSystem(singularValues, y);
        
        // Step 3: X = V * Z (where V = vtMatrix^T)
        IMatrix<Double> vMatrix = vtMatrix.transpose();
        return vMatrix.mmul(z);
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
        // Check if all singular values are non-zero
        for (int i = 0; i < singularValues.length(); i++) {
            if (Math.abs(singularValues.get(i)) < epsilon) {
                return false;
            }
        }
        return true;
    }
    
    @Override
    public IMatrix<Double> getInverse() {
        // For SVD decomposition A = U * S * V^T, A^(-1) = V * S^(-1) * U^T
        // Check for singularity first
        if (!isNonSingular()) {
            throw new RuntimeException("Matrix is singular");
        }
        
        // Create diagonal matrix with inverted singular values
        int n = singularValues.length();
        IMatrix<Double> sInv = Linalg.zeros(n, n);
        for (int i = 0; i < n; i++) {
            sInv.put(i, i, 1.0 / singularValues.get(i));
        }
        
        // Compute A^(-1) = V * S^(-1) * U^T
        IMatrix<Double> vMatrix = vtMatrix.transpose();
        IMatrix<Double> uTranspose = uMatrix.transpose();
        return vMatrix.mmul(sInv).mmul(uTranspose);
    }
    
    /**
     * Solve a diagonal linear system S * X = B where S is a diagonal matrix
     * represented by its diagonal elements (singular values).
     * 
     * @param diagonal Diagonal elements (singular values)
     * @param b Right-hand side matrix B
     * @return Solution matrix X
     */
    private IMatrix<Double> solveDiagonalSystem(IVector<Double> diagonal, IMatrix<Double> b) {
        int rows = b.rows();
        int cols = b.cols();
        IMatrix<Double> x = Linalg.zeros(rows, cols);
        
        // For diagonal system, X[i,j] = B[i,j] / S[i,i]
        for (int i = 0; i < rows; i++) {
            double diagElement = diagonal.get(i);
            if (Math.abs(diagElement) < epsilon) {
                throw new RuntimeException("Matrix is singular: singular value " + diagElement + " at index " + i);
            }
            for (int j = 0; j < cols; j++) {
                x.put(i, j, b.get(i, j) / diagElement);
            }
        }
        
        return x;
    }
    
    /**
     * Get the U matrix.
     * 
     * @return the U matrix
     */
    public IMatrix<Double> getU() {
        return uMatrix;
    }
    
    /**
     * Get the singular values.
     * 
     * @return the singular values
     */
    public IVector<Double> getSingularValues() {
        return singularValues;
    }
    
    /**
     * Get the V^T matrix.
     * 
     * @return the V^T matrix
     */
    public IMatrix<Double> getVT() {
        return vtMatrix;
    }
}