package com.yishape.lab.math.linalg.solver;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.decomposition.Decomps;
import com.yishape.lab.math.linalg.decomposition.DecompositionFailedException;

/**
 * Unified matrix condition number calculator that automatically selects the most appropriate
 * decomposition method based on matrix properties.
 * <p>
 * This calculator analyzes the input matrix characteristics and selects the optimal
 * decomposition algorithm for computing the matrix condition number κ(A). It provides
 * a unified interface for condition number calculation regardless of matrix type or condition,
 * automatically choosing the most efficient and numerically stable method available.
 * </p>
 *
 * <h3>Routing</h3>
 * <p>Primary κ₂ from SVD when it succeeds; else SPD Cholesky, symmetric tridiagonal, square LU/QR
 * estimates, Gaussian elimination rough estimate for square {@code A}, then QR. No {@code matrix.cond()}
 * re-entry from inside this class.</p>
 * 
 * <h3>Usage Examples</h3>
 * <pre>
 * {@code
 * // Compute the condition number of a matrix
 * IMatrix<Double> A = Linalg.matrix(new double[][]{{2, 1}, {1, 2}});
 * double conditionNumber = ConditionNumberSolver.compute(A);
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
 */
public class ConditionNumberSolver {
    
    /** Default epsilon for numerical comparisons. */
    private static final double DEFAULT_EPSILON = 1e-12;
    
    /**
     * Compute the condition number of a matrix by automatically selecting the most
     * appropriate decomposition method based on matrix properties.
     * 
     * @param A The matrix to compute condition number for
     * @return The condition number of matrix A
     * @throws DecompositionFailedException if no suitable decomposition method can be applied
     */
    public static double compute(IMatrix<Double> A) {
        // Validate input matrix
        if (A == null) {
            throw new IllegalArgumentException("Input matrix cannot be null");
        }
        
        // For condition number calculation, SVD is the most accurate method
        try {
            // Try SVD decomposition first as it's the most accurate for condition number calculation
            var svd = Decomps.createSVD();
            svd.decompose(A);
            return svd.getConditionNumber();
        } catch (Exception e) {
            // If SVD fails, fall back to other methods
        }
        
        // 1. Check if matrix is symmetric positive definite
        if (isSymmetricPositiveDefinite(A)) {
            try {
                // Symmetric positive definite matrix, use Cholesky decomposition
                var chol = Decomps.createCholesky();
                chol.decompose(A);
                return chol.getConditionNumber();
            } catch (Exception e) {
                // If Cholesky fails, fall back to other methods
            }
        }
        
        // 2. Check if matrix is symmetric
        if (isSymmetric(A)) {
            try {
                // Symmetric matrix, use Tridiagonal decomposition
                var tri = Decomps.createTridiagonal();
                tri.decompose(A);
                return tri.getConditionNumber();
            } catch (Exception e) {
                // If Tridiagonal fails, fall back to other methods
            }
        }
        
        // 3. Square: LU then QR (after SVD failure)
        if (A.rows() == A.cols()) {
            try {
                // Square matrix, try LU decomposition first
                var lu = Decomps.createLU();
                lu.decompose(A);
                return lu.getConditionNumber();
            } catch (Exception e) {
                // If LU fails, try QR decomposition
                try {
                    var qr = Decomps.createQR();
                    qr.decompose(A);
                    return qr.getConditionNumber();
                } catch (Exception e2) {
                    // If QR also fails, fall back to other methods
                }
            }
        }
        
        // 4. Square: rough estimate from Gaussian elimination (no matrix.cond() — avoids re-entry)
        if (A.rows() == A.cols()) {
            try {
                return conditionNumberByGaussianElimination(A);
            } catch (Exception e) {
                // fall through
            }
        }
        
        // 5. Last resort QR
        try {
            var qr = Decomps.createQR();
            qr.decompose(A);
            return qr.getConditionNumber();
        } catch (Exception e) {
            throw new DecompositionFailedException(
                "Failed to compute matrix condition number using any decomposition method", 
                "ConditionNumberSolver", 
                "All decomposition methods failed", 
                -1, 
                Double.NaN, 
                e);
        }
    }
    
    /**
     * Compute condition number using Gaussian elimination with partial pivoting.
     * This method is particularly useful for ill-conditioned matrices.
     * 
     * @param a The matrix to compute condition number for
     * @return The condition number of matrix A
     */
    private static double conditionNumberByGaussianElimination(IMatrix<Double> a) {
        // For condition number computation using Gaussian elimination, we need to
        // compute both the norm of the matrix and the norm of its inverse
        // We'll use the approach of computing the matrix norm and estimating the inverse norm
        
        // Compute Frobenius norm of the original matrix
        double normA = a.frobeniusNorm();
        
        // If the matrix is singular, return infinity
        if (normA == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        
        // For a more accurate condition number, we would ideally compute ||A|| * ||inv(A)||
        // However, computing the inverse explicitly is expensive and potentially unstable
        // Instead, we'll estimate the condition number using the ratio of
        // largest to smallest singular values from the Gaussian elimination process
        
        int n = a.rows();
        double[][] matrix = a.copy().toDoubleArray();
        
        // Keep track of the scaling factors during elimination
        double[] diagonalElements = new double[n];
        
        // Gaussian elimination with partial pivoting to estimate condition number
        for (int i = 0; i < n; i++) {
            // Find pivot
            int maxRow = i;
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(matrix[k][i]) > Math.abs(matrix[maxRow][i])) {
                    maxRow = k;
                }
            }
            
            // Swap rows
            if (maxRow != i) {
                double[] temp = matrix[i];
                matrix[i] = matrix[maxRow];
                matrix[maxRow] = temp;
            }
            
            // Check for singular matrix
            if (Math.abs(matrix[i][i]) < DEFAULT_EPSILON) {
                return Double.POSITIVE_INFINITY;
            }
            
            diagonalElements[i] = matrix[i][i];
            
            // Eliminate column
            for (int k = i + 1; k < n; k++) {
                double factor = matrix[k][i] / matrix[i][i];
                for (int j = i; j < n; j++) {
                    matrix[k][j] -= factor * matrix[i][j];
                }
            }
        }
        
        // Estimate condition number using the ratio of largest to smallest diagonal elements
        double maxDiag = 0.0;
        double minDiag = Double.POSITIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            double diag = Math.abs(diagonalElements[i]);
            maxDiag = Math.max(maxDiag, diag);
            minDiag = Math.min(minDiag, diag);
        }
        
        if (minDiag > DEFAULT_EPSILON) {
            return maxDiag / minDiag;
        } else {
            return Double.POSITIVE_INFINITY;
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