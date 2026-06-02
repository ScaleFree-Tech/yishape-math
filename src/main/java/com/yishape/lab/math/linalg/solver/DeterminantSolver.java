package com.yishape.lab.math.linalg.solver;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.decomposition.Decomps;
import com.yishape.lab.math.linalg.decomposition.DecompositionFailedException;
import com.yishape.lab.math.linalg.decomposition.ICholeskyDecomposition;
import com.yishape.lab.math.linalg.decomposition.ILUDecomposition;
import com.yishape.lab.math.linalg.decomposition.IQRDecomposition;
import com.yishape.lab.math.linalg.decomposition.ISVDDecomposition;
import com.yishape.lab.math.linalg.decomposition.ITridiagonalDecomposition;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;

/**
 * Unified determinant calculator that automatically selects the most appropriate
 * decomposition method based on matrix properties.
 * <p>
 * This calculator analyzes the input matrix characteristics and selects the optimal
 * decomposition algorithm for computing the matrix determinant det(A). It provides
 * a unified interface for determinant calculation regardless of matrix type or condition,
 * automatically choosing the most efficient and numerically stable method available.
 * </p>
 *
 * <h3>Routing</h3>
 * <p>SPD Cholesky, symmetric tridiagonal, partial-pivot LU (standard det via U and pivot parity),
 * then SVD.</p>
 * 
 * <h3>Usage Examples</h3>
 * <pre>
 * {@code
 * // Compute the determinant of a matrix
 * IMatrix<Double> A = Linalg.matrix(new double[][]{{2, 1}, {1, 2}});
 * double det = DeterminantSolver.compute(A);
 * }
 * </pre>
 * 
 * <h3>References</h3>
 * <ul>
 *   <li>Golub, G. H., &amp; Van Loan, C. F. (2013). Matrix computations (4th ed.). Johns Hopkins University Press.</li>
 *   <li>Press, W. H., Teukolsky, S. A., Vetterling, W. T., &amp; Flannery, B. P. (2007). Numerical recipes: The art of scientific computing (3rd ed.). Cambridge University Press.</li>
 *   <li>Horn, R. A., &amp; Johnson, C. R. (2012). Matrix analysis (2nd ed.). Cambridge University Press.</li>
 * </ul>
 * 
 * @since 2.0
 * @see Decomps For creating specific decomposition instances
 * @see IDecompositionSolver For individual solver interfaces
 */
public class DeterminantSolver {
    
    /**
     * Compute the determinant of a matrix by automatically selecting the most
     * appropriate decomposition method based on matrix properties.
     * 
     * @param A The matrix to compute determinant for
     * @return The determinant of matrix A
     * @throws DecompositionFailedException if no suitable decomposition method can be applied
     * @throws IllegalArgumentException if matrix is not square
     */
    public static double compute(IMatrix<Double> A) {
        // Validate input matrix
        if (A == null) {
            throw new IllegalArgumentException("Input matrix cannot be null");
        }
        
        // Check if matrix is square
        if (A.rows() != A.cols()) {
            throw new IllegalArgumentException("Matrix must be square to compute determinant");
        }
        
        // 1. First check if matrix is symmetric positive definite
        if (isSymmetricPositiveDefinite(A)) {
            try {
                // Symmetric positive definite matrix, use Cholesky decomposition
                ICholeskyDecomposition chol = Decomps.createCholesky();
                chol.decompose(A);
                return chol.getDeterminant();
            } catch (Exception e) {
                // If Cholesky fails, fall back to other methods
            }
        }
        
        // 2. Check if matrix is symmetric
        if (isSymmetric(A)) {
            try {
                // Symmetric matrix, use Tridiagonal decomposition
                ITridiagonalDecomposition tri = Decomps.createTridiagonal();
                tri.decompose(A);
                return tri.getDeterminant();
            } catch (Exception e) {
                // If Tridiagonal fails, fall back to other methods
            }
        }
        
        // 3. General square: LU (standard for det via U diagonal and pivot parity)
        if (A.rows() == A.cols()) {
            try {
                // Square matrix, try LU decomposition first (common method for determinant)
                ILUDecomposition lu = Decomps.createLU();
                lu.decompose(A);
                return lu.getDeterminant();
            } catch (Exception e) {
                // If LU fails, try QR decomposition
                try {
                    IQRDecomposition qr = Decomps.createQR();
                    qr.decompose(A);
                    return qr.getDeterminant();
                } catch (Exception e2) {
                    // If QR also fails, fall back to other methods
                }
            }
        }
        
        // 4. Last resort: SVD
        try {
            ISVDDecomposition svd = Decomps.createSVD();
            svd.decompose(A);
            return svd.getDeterminant();
        } catch (Exception e) {
            throw new DecompositionFailedException(
                "Failed to compute matrix determinant using any decomposition method", 
                "DeterminantSolver", 
                "All decomposition methods failed", 
                -1, 
                Double.NaN, 
                e);
        }
    }
    
    /**
     * Check if a matrix is symmetric positive definite.
     * 
     * @param matrix The matrix to check
     * @return true if the matrix is symmetric positive definite, false otherwise
     */
    private static boolean isSymmetricPositiveDefinite(IMatrix<Double> matrix) {
        try {
            // First check if matrix is square
            if (matrix.rows() != matrix.cols()) {
                return false;
            }
            
            // Check if matrix is symmetric
            if (!matrix.isSymmetric()) {
                return false;
            }
            
            // Check if matrix is positive definite
            return matrix.isPositiveDefinite();
        } catch (Exception e) {
            // If any check fails, assume it's not symmetric positive definite
            return false;
        }
    }
    
    /**
     * Check if a matrix is symmetric.
     * 
     * @param matrix The matrix to check
     * @return true if the matrix is symmetric, false otherwise
     */
    private static boolean isSymmetric(IMatrix<Double> matrix) {
        try {
            // First check if matrix is square
            if (matrix.rows() != matrix.cols()) {
                return false;
            }
            
            return matrix.isSymmetric();
        } catch (Exception e) {
            // If the check fails, assume it's not symmetric
            return false;
        }
    }
}