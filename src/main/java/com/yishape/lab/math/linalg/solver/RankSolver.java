package com.yishape.lab.math.linalg.solver;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.decomposition.Decomps;
import com.yishape.lab.math.linalg.decomposition.DecompositionFailedException;
import com.yishape.lab.math.util.RerePrecision;

/**
 * Unified matrix rank calculator that automatically selects the most appropriate
 * decomposition method based on matrix properties.
 * <p>
 * This calculator analyzes the input matrix characteristics and selects the optimal
 * decomposition algorithm for computing the matrix rank rank(A). It provides
 * a unified interface for rank calculation regardless of matrix type or condition,
 * automatically choosing the most efficient and numerically stable method available.
 * </p>
 *
 * <h3>Routing</h3>
 * <p>SVD first for accurate numerical rank; then SPD Cholesky, symmetric tridiagonal, square LU/QR,
 * Gaussian elimination estimate, last-chance QR.</p>
 * 
 * <h3>Usage Examples</h3>
 * <pre>
 * {@code
 * // Compute the rank of a matrix
 * IMatrix<Double> A = Linalg.matrix(new double[][]{{2, 1}, {1, 2}});
 * int rank = RankSolver.compute(A);
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
 * @see com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver For individual solver interfaces
 */
public class RankSolver {
    
    /**
     * Compute the rank of a matrix by automatically selecting the most
     * appropriate decomposition method based on matrix properties.
     * 
     * @param A The matrix to compute rank for
     * @return The rank of matrix A
     * @throws DecompositionFailedException if no suitable decomposition method can be applied
     */
    public static int compute(IMatrix<Double> A) {
        // Validate input matrix
        if (A == null) {
            throw new IllegalArgumentException("Input matrix cannot be null");
        }
        
        // For rank calculation, SVD is the most accurate method
        try {
            // Try SVD decomposition first as it's the most accurate for rank calculation
            var svd = Decomps.createSVD();
            svd.decompose(A);
            return svd.getRank();
        } catch (Exception e) {
            // If SVD fails, fall back to other methods
        }
        
        // 1. Check if matrix is symmetric positive definite
        if (isSymmetricPositiveDefinite(A)) {
            try {
                // Symmetric positive definite matrix, use Cholesky decomposition
                var chol = Decomps.createCholesky();
                chol.decompose(A);
                return chol.getRank();
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
                return tri.getRank();
            } catch (Exception e) {
                // If Tridiagonal fails, fall back to other methods
            }
        }
        
        // 3. Square matrix: LU then QR (align with general dense practice after SVD failure)
        if (A.rows() == A.cols()) {
            try {
                // Square matrix, try LU decomposition first
                var lu = Decomps.createLU();
                lu.decompose(A);
                return lu.getRank();
            } catch (Exception e) {
                // If LU fails, try QR decomposition
                try {
                    var qr = Decomps.createQR();
                    qr.decompose(A);
                    return qr.getRank();
                } catch (Exception e2) {
                    // If QR also fails, fall back to other methods
                }
            }
        }
        
        // 4. Cheap rank estimate without another SVD / cond()
        try {
            return rankByGaussianElimination(A);
        } catch (Exception e) {
            // fall through
        }
        
        // 5. Last resort QR
        try {
            var qr = Decomps.createQR();
            qr.decompose(A);
            return qr.getRank();
        } catch (Exception e) {
            throw new DecompositionFailedException(
                "Failed to compute matrix rank using any decomposition method", 
                "RankSolver", 
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
    
    /**
     * 使用高斯消元法计算矩阵秩的辅助方法 / Helper method for computing rank using Gaussian
     * elimination
     */
    private static int rankByGaussianElimination(IMatrix<Double> matrix0) {
        double[][] matrix = matrix0.copy().toDoubleArray();
                int rows = matrix.length;
        int cols = matrix[0].length;

        int rank = 0;
        final double tolerance = 1e-10;

        for (int col = 0; col < cols && rank < rows; col++) {
            // 寻找主元
            int pivotRow = rank;
            for (int row = rank + 1; row < rows; row++) {
                if (!RerePrecision.equalsZero(matrix[row][col], 1e-12) &&
                    (RerePrecision.equalsZero(matrix[pivotRow][col], 1e-12) ||
                     Math.abs(matrix[row][col]) > Math.abs(matrix[pivotRow][col]))) {
                    pivotRow = row;
                }
            }

            if (!RerePrecision.equalsZero(matrix[pivotRow][col], tolerance)) {
                // 交换行
                if (pivotRow != rank) {
                    double[] temp = matrix[rank];
                    matrix[rank] = matrix[pivotRow];
                    matrix[pivotRow] = temp;
                }

                // 消元
                for (int row = rank + 1; row < rows; row++) {
                    double factor = matrix[row][col] / matrix[rank][col];
                    for (int c = col; c < cols; c++) {
                        matrix[row][c] -= factor * matrix[rank][c];
                    }
                }
                rank++;
            }
        }

        return rank;
    }
}