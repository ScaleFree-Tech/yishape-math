package com.yishape.lab.math.linalg.decomposition.solver;

import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

/**
 * Solver for eigen decomposition.
 * <p>
 * This solver uses the eigen decomposition A = V * D * V^(-1) to solve
 * linear systems. If A is diagonalizable, the solution to A * X = B is
 * X = V * D^(-1) * V^(-1) * B.
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
public class EigenDecompositionSolver implements IDecompositionSolver {
    
    /** The eigenvalues. */
    private final IVector<Double> eigenvalues;
    /** The eigenvectors matrix V. */
    private final IMatrix<Double> eigenvectors;
    /** Epsilon for numerical comparisons. */
    private final double epsilon;
    
    /**
     * Create a solver from eigen decomposition results.
     * 
     * @param eigenvalues the eigenvalues
     * @param eigenvectors the eigenvectors matrix
     * @param epsilon threshold for considering an element as zero
     */
    public EigenDecompositionSolver(IVector<Double> eigenvalues, IMatrix<Double> eigenvectors, double epsilon) {
        this.eigenvalues = eigenvalues;
        this.eigenvectors = eigenvectors;
        this.epsilon = epsilon;
    }
    
    @Override
    public IMatrix<Double> solve(IMatrix<Double> b) {
        // For eigen decomposition A = V * D * V^(-1), to solve A * X = B:
        // 1. Compute Y = V^(-1) * B
        // 2. Solve D * Z = Y for Z (element-wise division by eigenvalues)
        // 3. Compute X = V * Z
        
        // Check for singularity first
        if (!isNonSingular()) {
            throw new RuntimeException("Matrix is singular");
        }
        
        // Step 1: Y = V^(-1) * B
        // We need to solve V * Y = B for Y, which means Y = V \ B
        IMatrix<Double> y = solveLinearSystem(eigenvectors, b);
        
        // Step 2: Solve D * Z = Y for Z (element-wise division)
        IMatrix<Double> z = solveDiagonalSystem(eigenvalues, y);
        
        // Step 3: X = V * Z
        return eigenvectors.mmul(z);
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
        // Check if all eigenvalues are non-zero
        for (int i = 0; i < eigenvalues.length(); i++) {
            if (Math.abs(eigenvalues.get(i)) < epsilon) {
                return false;
            }
        }
        return true;
    }
    
    @Override
    public IMatrix<Double> getInverse() {
        // For eigen decomposition A = V * D * V^(-1), A^(-1) = V * D^(-1) * V^(-1)
        // Check for singularity first
        if (!isNonSingular()) {
            throw new RuntimeException("Matrix is singular");
        }
        
        // Create diagonal matrix with inverted eigenvalues
        int n = eigenvalues.length();
        IMatrix<Double> dInv = Linalg.zeros(n, n);
        for (int i = 0; i < n; i++) {
            double ev = eigenvalues.get(i);
            if (Math.abs(ev) < epsilon) {
                throw new RuntimeException("Eigenvalue too small for inversion at index " + i + ": " + ev);
            }
            dInv.put(i, i, 1.0 / ev);
        }
        
        // Compute A^(-1) = V * D^(-1) * V^(-1)
        // V^(-1) is solved by V * V^(-1) = I
        IMatrix<Double> vInv = solveLinearSystem(eigenvectors, Linalg.eye(n));
        return eigenvectors.mmul(dInv).mmul(vInv);
    }
    
    /**
     * Solve a diagonal linear system D * X = B where D is a diagonal matrix
     * represented by its diagonal elements (eigenvalues).
     * 
     * @param diagonal Diagonal elements (eigenvalues)
     * @param b Right-hand side matrix B
     * @return Solution matrix X
     */
    private IMatrix<Double> solveDiagonalSystem(IVector<Double> diagonal, IMatrix<Double> b) {
        int rows = b.rows();
        int cols = b.cols();
        IMatrix<Double> x = Linalg.zeros(rows, cols);
        
        // For diagonal system, X[i,j] = B[i,j] / D[i,i]
        for (int i = 0; i < rows; i++) {
            double diagElement = diagonal.get(i);
            if (Math.abs(diagElement) < epsilon) {
                throw new RuntimeException("Matrix is singular: eigenvalue " + diagElement + " at index " + i);
            }
            for (int j = 0; j < cols; j++) {
                x.put(i, j, b.get(i, j) / diagElement);
            }
        }
        
        return x;
    }
    
    /**
     * Solve a general linear system A * X = B using Gaussian elimination with partial pivoting.
     * 
     * @param a Coefficient matrix A
     * @param b Right-hand side matrix B
     * @return Solution matrix X
     */
    private IMatrix<Double> solveLinearSystem(IMatrix<Double> a, IMatrix<Double> b) {
        int n = a.rows();
        int m = b.cols();
        
        // Build [A|B] without allocating separate full copies of A and B for dense matrices
        double[][] augmented = new double[n][n + m];
        if (a instanceof IDoubleMatrix adm && b instanceof IDoubleMatrix bdm) {
            double[][] asrc = adm.getData();
            double[][] bsrc = bdm.getData();
            for (int i = 0; i < n; i++) {
                System.arraycopy(asrc[i], 0, augmented[i], 0, n);
                System.arraycopy(bsrc[i], 0, augmented[i], n, m);
            }
        } else {
            double[][] aData = a.toDoubleArray();
            double[][] bData = b.toDoubleArray();
            for (int i = 0; i < n; i++) {
                System.arraycopy(aData[i], 0, augmented[i], 0, n);
                System.arraycopy(bData[i], 0, augmented[i], n, m);
            }
        }
        
        // Gaussian elimination with partial pivoting
        for (int i = 0; i < n; i++) {
            // Find pivot
            int maxRow = i;
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
                    maxRow = k;
                }
            }
            
            // Swap rows
            if (maxRow != i) {
                double[] temp = augmented[i];
                augmented[i] = augmented[maxRow];
                augmented[maxRow] = temp;
            }
            
            // Check for singular matrix
            if (Math.abs(augmented[i][i]) < epsilon) {
                throw new RuntimeException("Matrix is singular");
            }
            
            // Eliminate column
            for (int k = i + 1; k < n; k++) {
                double factor = augmented[k][i] / augmented[i][i];
                for (int j = i; j < n + m; j++) {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
        
        // Back substitution
        double[][] xData = new double[n][m];
        for (int i = n - 1; i >= 0; i--) {
            for (int j = 0; j < m; j++) {
                xData[i][j] = augmented[i][n + j];
                for (int k = i + 1; k < n; k++) {
                    xData[i][j] -= augmented[i][k] * xData[k][j];
                }
                xData[i][j] /= augmented[i][i];
            }
        }
        
        return Linalg.matrix(xData);
    }
    
    /**
     * Get the eigenvalues.
     * 
     * @return the eigenvalues
     */
    public IVector<Double> getEigenvalues() {
        return eigenvalues;
    }
    
    /**
     * Get the eigenvectors matrix.
     * 
     * @return the eigenvectors matrix
     */
    public IMatrix<Double> getEigenvectors() {
        return eigenvectors;
    }
}