package com.yishape.lab.math.linalg.solver;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.Decomps;
import com.yishape.lab.math.linalg.decomposition.DecompositionFailedException;
import com.yishape.lab.math.linalg.decomposition.IBunchKaufmanDecomposition;
import com.yishape.lab.math.linalg.decomposition.ICholeskyDecomposition;
import com.yishape.lab.math.linalg.decomposition.ILUDecomposition;
import com.yishape.lab.math.linalg.decomposition.IQRDecomposition;
import com.yishape.lab.math.linalg.decomposition.ISVDDecomposition;
import com.yishape.lab.math.linalg.decomposition.ITridiagonalDecomposition;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.sparse.ISparseLinearSolver;
import com.yishape.lab.math.linalg.sparse.ISparseMatrix;
import com.yishape.lab.math.linalg.sparse.ISpecialSparseMatrix;
import com.yishape.lab.math.linalg.sparse.SparseILUPreconditioner;
import com.yishape.lab.math.linalg.sparse.impl.SparseConjugateGradientSolver;
import com.yishape.lab.math.linalg.sparse.impl.SparseBICGSTABSolver;
import com.yishape.lab.math.linalg.sparse.impl.SparseGMRESSolver;

import java.util.function.BiFunction;
/**
 * Unified linear system solver that automatically selects the most appropriate
 * decomposition method based on matrix properties.
 * <p>
 * This solver analyzes the input matrix characteristics and selects the optimal
 * decomposition algorithm for solving the linear system A × X = B. It provides
 * a unified interface for solving linear systems regardless of matrix type or
 * condition, automatically choosing the most efficient and numerically stable
 * method available.
 * </p>
 * 
 * <h3>Algorithm selection (square {@code A})</h3>
 * <p>SPD → Cholesky + iterative refinement. Symmetric indefinite / general symmetric →
 * Bunch–Kaufman (SYTRF 风格) + refinement first，其次 partial-pivot LU + refinement，
 * 再对称三对角 + refinement，其后可选 HPC、一般 LU、QR、SVD（秩判据）。无隐式全 SVD 条件探测。</p>
 * 
 * <h3>Performance Characteristics</h3>
 * <ul>
 *   <li><strong>Speed (typical dense square)</strong>: Cholesky (SPD) &gt; LU &gt; QR &gt; … &gt; SVD</li>
 *   <li><strong>Stability</strong>: SVD &gt; QR ≥ Cholesky &gt; LU &gt; …</li>
 *   <li><strong>Universality</strong>: SVD (any matrix) > QR/LU (square) > Cholesky (symmetric positive definite) > others (specialized)</li>
 * </ul>
 * 
 * <h3>Usage Examples</h3>
 * <pre>
 * {@code
 * // Solve a linear system with matrix right-hand side
 * IMatrix<Double> A = Linalg.matrix(new double[][]{{2, 1}, {1, 2}});
 * IMatrix<Double> B = Linalg.matrix(new double[][]{{1, 0}, {0, 1}});
 * IMatrix<Double> X = LinearSystemSolver.solve(A, B);
 * 
 * // Solve a linear system with vector right-hand side
 * IVector<Double> b = Linalg.vector(new double[]{1, 2});
 * IVector<Double> x = LinearSystemSolver.solve(A, b);
 * }
 * </pre>
 * 
 * <h3>References</h3>
 * <ul>
 *   <li>Golub, G. H., &amp; Van Loan, C. F. (2013). Matrix computations (4th ed.). Johns Hopkins University Press.</li>
 *   <li>Press, W. H., Teukolsky, S. A., Vetterling, W. T., &amp; Flannery, B. P. (2007). Numerical recipes: The art of scientific computing (3rd ed.). Cambridge University Press.</li>
 *   <li>Watkins, D. S. (2007). The matrix eigenvalue problem: GR and Krylov subspace methods. SIAM.</li>
 *   <li>Horn, R. A., &amp; Johnson, C. R. (2012). Matrix analysis (2nd ed.). Cambridge University Press.</li>
 * </ul>
 * 
 * @since 2.0
 * @see Decomps For creating specific decomposition instances
 * @see IDecompositionSolver For individual solver interfaces
 */
public class LinearSystemSolver {
    
    /**
     * Solve the linear system A × x = b by automatically selecting the most
     * appropriate decomposition method based on matrix properties.
     * 
     * @param A Coefficient matrix of the linear system
     * @param b Right-hand side vector of the linear system
     * @return Solution vector x
     * @throws DecompositionFailedException if no suitable decomposition method can be applied
     * @throws IllegalArgumentException if matrices dimensions do not match
     */
    public static IVector<Double> solve(IMatrix<Double> A, IVector<Double> b) {
        // Validate input matrices
        if (A == null || b == null) {
            throw new IllegalArgumentException("Input matrix and vector cannot be null");
        }
        
        if (A.rows() != b.size()) {
            throw new IllegalArgumentException(
                "Matrix row dimension mismatch: A has " + A.rows() + 
                " rows but b has " + b.size() + " elements");
        }
        
        double[] bd = b.toDoubleArray();
        double[][] col = new double[bd.length][1];
        for (int i = 0; i < bd.length; i++) {
            col[i][0] = bd[i];
        }
        IMatrix<Double> bMatrix = Linalg.matrix(col);
        IMatrix<Double> xMatrix = solve(A, bMatrix, null);
        
        // Convert result back to vector using getColumn method
        return xMatrix.getColumn(0);
    }
    
    /**
     * Solve the linear system A × X = B by automatically selecting the most
     * appropriate decomposition method based on matrix properties.
     * 
     * @param A Coefficient matrix of the linear system
     * @param B Right-hand side matrix of the linear system
     * @return Solution matrix X
     * @throws DecompositionFailedException if no suitable decomposition method can be applied
     * @throws IllegalArgumentException if matrices dimensions do not match
     * @see #solve(IMatrix, IMatrix, BiFunction)
     */
    public static IMatrix<Double> solve(IMatrix<Double> A, IMatrix<Double> B) {
        return solve(A, B, null);
    }

    /**
     * 与 {@link #solve(IMatrix, IMatrix)} 相同，但在「一般方阵、尝试 LU」阶段之前，可先调用
     * {@code optionalDenseSquareSingleColumnRhs}（由上层注入，例如 {@code RereDoubleMatrix} 对接 yishape-math-hpc）。
     * 传入 {@code null} 时与 {@link #solve(IMatrix, IMatrix)} 完全一致。
     *
     * @param optionalDenseSquareSingleColumnRhs 对 (A,B) 返回解矩阵或 {@code null}；仅当 A 为方阵且 B 为单列时可能调用
     */
    public static IMatrix<Double> solve(IMatrix<Double> A, IMatrix<Double> B,
            BiFunction<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> optionalDenseSquareSingleColumnRhs) {
        // Validate input matrices
        if (A == null || B == null) {
            throw new IllegalArgumentException("Input matrices cannot be null");
        }

        // solve is strictly for square matrices (determined systems)
        if (A.rows() != A.cols()) {
            throw new IllegalArgumentException(
                "solve 方法仅支持方阵（恰定方程组）。对于非方阵（超定或欠定方程组），请使用 lstsq 方法。 / " +
                "solve method is strictly for square matrices (determined systems). " +
                "For non-square matrices (overdetermined or underdetermined systems), please use lstsq method.");
        }
        
        if (A.rows() != B.rows()) {
            throw new IllegalArgumentException(
                "Matrix row dimension mismatch: A has " + A.rows() +
                " rows but B has " + B.rows() + " rows");
        }

        // 0. Sparse matrix detection: dispatch to iterative solvers
        if (A instanceof ISparseMatrix sparseA) {
            IMatrix<Double> xSparse = trySparseSolve(sparseA, B);
            if (xSparse != null) return xSparse;
        }

        // 1. Try Cholesky first (fastest for SPD): the decomposition itself validates SPD,
        //    so we skip the expensive isSymmetricPositiveDefinite() pre-check.
        //    PREVIOUS BUG (fixed 2026-05-16): isSymmetricPositiveDefinite() called
        //    isPositiveDefinite() which ran a full O(n³/6) Cholesky and discarded the result.
        //    Then the actual solve ran a second Cholesky. Removing the pre-check saves ~50%
        //    decomposition cost for SPD matrices. For non-SPD matrices, Cholesky fails at
        //    the first negative pivot (O(n) detection).
        try {
            ICholeskyDecomposition chol = Decomps.createCholesky();
            chol.decompose(A);
            var sol = chol.getSolver();
            IMatrix<Double> x = sol.solve(B);
            return DenseIterativeRefinement.refine(A, B, x, sol);
        } catch (Exception e) {
            // If Cholesky fails, fall back to other methods
        }
        
        // 2. 对称：先 Bunch–Kaufman（SYTRF/DSYTF2），再 LU，再对称三对角
        if (isSymmetric(A)) {
            try {
                IBunchKaufmanDecomposition bk = Decomps.createBunchKaufman();
                bk.decompose(A);
                if (bk.isNonSingular()) {
                    var solBk = bk.getSolver();
                    IMatrix<Double> xBk = solBk.solve(B);
                    return DenseIterativeRefinement.refine(A, B, xBk, solBk);
                }
            } catch (Exception e) {
                // fall through
            }
            try {
                ILUDecomposition luSym = Decomps.createLU();
                luSym.decompose(A);
                if (luSym.isNonSingular()) {
                    var sol = luSym.getSolver();
                    IMatrix<Double> x = sol.solve(B);
                    return DenseIterativeRefinement.refine(A, B, x, sol);
                }
            } catch (Exception e) {
                // fall through
            }
            try {
                ITridiagonalDecomposition tri = Decomps.createTridiagonal();
                tri.decompose(A);
                var sol = tri.getSolver();
                IMatrix<Double> x = sol.solve(B);
                return DenseIterativeRefinement.refine(A, B, x, sol);
            } catch (Exception e) {
                // If Tridiagonal fails, fall back to other methods
            }
        }
        
        // 3. General square matrix: optional injected dense solve (e.g. HPC) then LU / QR (Java).
        if (A.rows() == A.cols()) {
            if (optionalDenseSquareSingleColumnRhs != null && B.cols() == 1) {
                try {
                    IMatrix<Double> xNative = optionalDenseSquareSingleColumnRhs.apply(A, B);
                    if (xNative != null) {
                        return xNative;
                    }
                } catch (Exception e) {
                    // fall through to Java LU
                }
            }
            try {
                // Square matrix, try LU decomposition first (faster than QR for general case)
                ILUDecomposition lu = Decomps.createLU();
                lu.decompose(A);
                var luSol = lu.getSolver();
                IMatrix<Double> xLu = luSol.solve(B);
                return DenseIterativeRefinement.refine(A, B, xLu, luSol);
            } catch (Exception e) {
                // If LU fails, try QR decomposition
                try {
                    IQRDecomposition qr = Decomps.createQR();
                    qr.decompose(A);
                    var qrSol = qr.getSolver();
                    IMatrix<Double> xQr = qrSol.solve(B);
                    return DenseIterativeRefinement.refine(A, B, xQr, qrSol);
                } catch (Exception e2) {
                    // If QR also fails, fall back to other methods
                }
            }
        }
        
        // 4. When other methods fail or matrix is singular in LU/QR sense, use SVD
        try {
            ISVDDecomposition svd = Decomps.createSVD();
            svd.decompose(A);
            int rank = svd.getRank();
            if (rank < A.rows()) {
                throw new ArithmeticException(
                    "矩阵是奇异的（不满秩），无法保证唯一解。请使用 lstsq 方法获取最小二乘解。 / " +
                    "The matrix is singular (not full rank), unique solution cannot be guaranteed. " +
                    "Please use lstsq method to get the least squares solution.");
            }
            
            return svd.getSolver().solve(B);
        } catch (Exception e) {
            if (e instanceof ArithmeticException) {
                throw (ArithmeticException) e;
            }
            throw new DecompositionFailedException(
                "无法使用任何分解方法求解线性方程组 / Failed to solve linear system using any decomposition method",
                "LinearSystemSolver",
                "所有分解方法均失败 / All decomposition methods failed",
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

    private static IMatrix<Double> trySparseSolve(ISparseMatrix A, IMatrix<Double> B) {
        if (A instanceof ISpecialSparseMatrix.DiagonalSparseMatrix diag) {
            return solveDiagonal(diag, B);
        }
        int n = A.rows();
        double[][] xCols = new double[n][B.cols()];
        ISparseLinearSolver solver;
        if (isSymmetric(A.toDense())) {
            solver = new SparseConjugateGradientSolver(1e-8, Math.max(1000, n), (com.yishape.lab.math.linalg.sparse.ISparsePreconditioner) null);
        } else {
            solver = new SparseGMRESSolver(1e-8, Math.max(1000, n), Math.min(30, n), null);
        }
        try {
            for (int j = 0; j < B.cols(); j++) {
                IVector<Double> bj = B.getColumn(j);
                IVector<Double> xj = solver.solve(A, bj);
                for (int i = 0; i < n; i++) xCols[i][j] = xj.get(i);
            }
            return Linalg.matrix(xCols);
        } catch (Exception e) {
            return null;
        }
    }

    private static IMatrix<Double> solveDiagonal(ISpecialSparseMatrix.DiagonalSparseMatrix D, IMatrix<Double> B) {
        int n = D.rows();
        double[][] xCols = new double[n][B.cols()];
        for (int j = 0; j < B.cols(); j++) {
            for (int i = 0; i < n; i++) {
                double diag = D.get(i, i);
                xCols[i][j] = (Math.abs(diag) > 1e-15) ? B.get(i, j) / diag : 0;
            }
        }
        return Linalg.matrix(xCols);
    }
}