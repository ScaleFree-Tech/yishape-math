package com.yishape.lab.math.linalg.solver;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.Decomps;
import com.yishape.lab.math.linalg.decomposition.DecompositionFailedException;
import com.yishape.lab.math.linalg.decomposition.IQrcpDecomposition;
import com.yishape.lab.math.linalg.decomposition.QrcpRankTolerance;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.util.RerePrecision;
import com.yishape.lab.util.Tuple2;

/**
 * Unified least squares solver that automatically selects the most appropriate
 * decomposition method based on matrix properties.
 * <p>
 * This solver analyzes the input matrix characteristics and selects the optimal
 * decomposition algorithm for solving the least squares problem min||A × X - B||₂.
 * It provides a unified interface for solving least squares problems regardless
 * of matrix type or condition, automatically choosing the most efficient and
 * numerically stable method available.
 * </p>
 *
 * <h3>Computation strategy</h3>
 * <p>Very large / tall {@code A}: matrix-free CGLS. Otherwise for {@code m ≥ n}: column-pivoted Householder QR (QRCP,
 * DGEQP3+DLAQPS style); when columns are rank-deficient or the factorization is singular per threshold, an xGELSY-style
 * tolerant backsolve is selected; otherwise the strict QR solver is used. Full-rank solutions receive fixed-point
 * iterative refinement. If the relative residual ‖AX−B‖_F/‖B‖_F remains large, escalate to SVD. For small dimensions,
 * normal equations (Cholesky / LU / tridiagonal) are tried, with SVD as final fallback.</p>
 * 
 * <h3>Usage Examples</h3>
 * <pre>
 * {@code
 * // Solve a least squares problem with matrix right-hand side
 * IMatrix<Double> A = Linalg.matrix(new double[][]{{1, 2}, {3, 4}, {5, 6}});
 * IMatrix<Double> B = Linalg.matrix(new double[][]{{1}, {2}, {3}});
 * IMatrix<Double> X = LeastSquaresSolver.solve(A, B);
 * 
 * // Solve a least squares problem with vector right-hand side
 * IVector<Double> b = Linalg.vector(new double[]{1, 2, 3});
 * IVector<Double> x = LeastSquaresSolver.solve(A, b);
 * 
 * // Solve a least squares problem and get both solution and residual
 * Tuple2<IVector<Double>, Double> result = LeastSquaresSolver.solveWithResidual(A, b);
 * IVector<Double> solution = result.getFirst();
 * Double residual = result.getSecond();
 * }
 * </pre>
 * 
 * <h3>References</h3>
 * <ul>
 *   <li>Golub, G. H., &amp; Van Loan, C. F. (2013). Matrix computations (4th ed.). Johns Hopkins University Press.</li>
 *   <li>Press, W. H., Teukolsky, S. A., Vetterling, W. T., &amp; Flannery, B. P. (2007). Numerical recipes: The art of scientific computing (3rd ed.). Cambridge University Press.</li>
 *   <li>Björck, Å. (1996). Numerical methods for least squares problems. SIAM.</li>
 *   <li>Lawson, C. L., &amp; Hanson, R. J. (1995). Solving least squares problems. SIAM.</li>
 * </ul>
 * 
 * @since 2.0
 * @see Decomps For creating specific decomposition instances
 */
public class LeastSquaresSolver {
    
    /** Threshold for matrix size to determine when to use specialized algorithms. */
    private static final int LARGE_MATRIX_THRESHOLD = 100;

    private static final int DEFAULT_HUGE_ROW_THRESHOLD = 50_000;

    private static final long DEFAULT_HUGE_ELEMENT_THRESHOLD = 10_000_000L;

    /** When mn exceeds this, use conjugate gradient on ½‖Ax−b‖² (avoid forming AᵀA and dense decompositions). */
    private static long hugeElementThreshold = DEFAULT_HUGE_ELEMENT_THRESHOLD;

    /** When A has at least this many rows, use CG least squares (typical tall-data case). */
    private static int hugeRowThreshold = DEFAULT_HUGE_ROW_THRESHOLD;

    private static final double CG_TOLERANCE = 1e-6;

    private static final int CG_MAX_ITERATIONS = 10_000;

    /**
     * Override row / element thresholds that route to conjugate-gradient least squares.
     * Intended for tests or advanced tuning; call {@link #resetHugeScaleThresholds()} when done
     * (e.g. in {@code @AfterEach}) so other tests see production defaults.
     *
     * @param rowThreshold minimum row count of A to use CG (must be &gt;= 0)
     * @param elementThreshold minimum {@code rows×cols} to use CG (must be &gt;= 0)
     */
    public static void setHugeScaleThresholds(int rowThreshold, long elementThreshold) {
        if (rowThreshold < 0 || elementThreshold < 0) {
            throw new IllegalArgumentException("Thresholds must be non-negative");
        }
        hugeRowThreshold = rowThreshold;
        hugeElementThreshold = elementThreshold;
    }

    /** Restore production defaults for row and {@code rows×cols} thresholds that select the CG path. */
    public static void resetHugeScaleThresholds() {
        hugeRowThreshold = DEFAULT_HUGE_ROW_THRESHOLD;
        hugeElementThreshold = DEFAULT_HUGE_ELEMENT_THRESHOLD;
    }
    
    /**
     * Solve the least squares problem min||A × x - b||₂ by automatically selecting
     * the most appropriate decomposition method based on matrix properties.
     * 
     * @param A Coefficient matrix of the least squares problem
     * @param b Right-hand side vector of the least squares problem
     * @return Solution vector x that minimizes ||A × x - b||₂
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
        IMatrix<Double> xMatrix = solve(A, bMatrix);
        
        // Convert result back to vector using getColumn method
        return xMatrix.getColumn(0);
    }
    
    /**
     * Solve the least squares problem min||A × X - B||₂ by automatically selecting
     * the most appropriate decomposition method based on matrix properties.
     * 
     * @param A Coefficient matrix of the least squares problem
     * @param B Right-hand side matrix of the least squares problem
     * @return Solution matrix X that minimizes ||A × X - B||₂
     * @throws DecompositionFailedException if no suitable decomposition method can be applied
     * @throws IllegalArgumentException if matrices dimensions do not match
     */
    public static IMatrix<Double> solve(IMatrix<Double> A, IMatrix<Double> B) {
        // Validate input matrices
        if (A == null || B == null) {
            throw new IllegalArgumentException("Input matrices cannot be null");
        }
        
        if (A.rows() != B.rows()) {
            throw new IllegalArgumentException(
                "Matrix row dimension mismatch: A has " + A.rows() + 
                " rows but B has " + B.rows() + " rows");
        }

        if (shouldUseConjugateGradientForLargeScale(A)) {
            return solveLeastSquaresCgls(A, B);
        }
        
        // 1. Tall or square A (m ≥ n): QRCP（DGEQP3+DLAQPS）+ 秩亏时容忍最小二乘；满秩迭代修正；残差仍大则 SVD
        if (A.rows() >= A.cols()) {
            try {
                IQrcpDecomposition qr = Decomps.createQrcpDlaqps();
                qr.decompose(A);
                double rankTol = QrcpRankTolerance.forLeastSquares(qr, A.rows(), A.cols());
                IDecompositionSolver sol = selectQrcpLeastSquaresSolver(qr, A.cols(), rankTol);
                IMatrix<Double> x = sol.solve(B);
                if (sol.isNonSingular()) {
                    x = DenseIterativeRefinement.refine(A, B, x, sol);
                }
                if (leastSquaresResidualAcceptable(A, B, x)) {
                    return x;
                }
                try {
                    var svdEarly = Decomps.createSVD();
                    svdEarly.decompose(A);
                    return svdEarly.getSolver().solve(B);
                } catch (Exception e2) {
                    // fall through
                }
            } catch (Exception e) {
                // fall through
            }
        }
        
        // 2. For normal equations approach when matrix is not too large
        if (A.rows() <= LARGE_MATRIX_THRESHOLD) {
            try {
                // Form normal equations A^T × A × X = A^T × B
                IMatrix<Double> AtA = A.transpose().mmul(A);
                
                // Check if normal equations matrix is symmetric positive definite
                if (isSymmetricPositiveDefinite(AtA)) {
                    // Use Cholesky decomposition for normal equations
                    var chol = Decomps.createCholesky();
                    chol.decompose(AtA);
                    IMatrix<Double> AtB = A.transpose().mmul(B);
                    return chol.getSolver().solve(AtB);
                } else {
                    // Use LU decomposition for normal equations
                    var lu = Decomps.createLU();
                    lu.decompose(AtA);
                    IMatrix<Double> AtB = A.transpose().mmul(B);
                    return lu.getSolver().solve(AtB);
                }
            } catch (Exception e) {
                // If normal equations approach fails, fall back to other methods
            }
        }
        
        // 3. Check if normal equations matrix is symmetric
        if (isSymmetric(A.transpose().mmul(A))) {
            try {
                // Form normal equations A^T × A × X = A^T × B
                IMatrix<Double> AtA = A.transpose().mmul(A);
                IMatrix<Double> AtB = A.transpose().mmul(B);
                
                // Symmetric normal equations matrix, use Tridiagonal decomposition
                var tri = Decomps.createTridiagonal();
                tri.decompose(AtA);
                return tri.getSolver().solve(AtB);
            } catch (Exception e) {
                // If Tridiagonal fails, fall back to other methods
            }
        }
        
        // 4. Large problems: normal equations with Cholesky (if SPD) or LU — not Hessenberg
        if (A.rows() > LARGE_MATRIX_THRESHOLD) {
            try {
                IMatrix<Double> AtA = A.transpose().mmul(A);
                IMatrix<Double> AtB = A.transpose().mmul(B);
                if (isSymmetricPositiveDefinite(AtA)) {
                    var chol = Decomps.createCholesky();
                    chol.decompose(AtA);
                    return chol.getSolver().solve(AtB);
                }
                var lu = Decomps.createLU();
                lu.decompose(AtA);
                return lu.getSolver().solve(AtB);
            } catch (Exception e) {
                // If normal equations approach fails, fall back to other methods
            }
        }
        
        // 5. Robust fallback: SVD
        try {
            var svd = Decomps.createSVD();
            svd.decompose(A);
            return svd.getSolver().solve(B);
        } catch (Exception e) {
            throw new DecompositionFailedException(
                "Failed to solve least squares problem using any decomposition method", 
                "LeastSquaresSolver", 
                "All decomposition methods failed", 
                -1, 
                Double.NaN, 
                e);
        }
    }
    
    /**
     * Solve the least squares problem min||A × x - b||₂ and return both the solution
     * and the residual norm ||A × x - b||₂.
     * 
     * @param A Coefficient matrix of the least squares problem
     * @param b Right-hand side vector of the least squares problem
     * @return Tuple2 containing the solution vector x and the residual norm
     * @throws DecompositionFailedException if no suitable decomposition method can be applied
     * @throws IllegalArgumentException if matrices dimensions do not match
     */
    public static Tuple2<IVector<Double>, Double> solveWithResidual(IMatrix<Double> A, IVector<Double> b) {
        // Validate input matrices
        if (A == null || b == null) {
            throw new IllegalArgumentException("Input matrix and vector cannot be null");
        }
        
        if (A.rows() != b.size()) {
            throw new IllegalArgumentException(
                "Matrix row dimension mismatch: A has " + A.rows() + 
                " rows but b has " + b.size() + " elements");
        }
        
        // Solve the least squares problem
        IVector<Double> x = solve(A, b);
        
        // Compute the residual: ||A × x - b||₂
        IVector<Double> residualVector = A.mmul(x).sub(b);
        double residualNorm = RerePrecision.roundToDecimalPlaces(residualVector.norm2Value(),6);
        
        return new Tuple2<>(x, residualNorm);
    }
    
    /**
     * Solve the least squares problem min||A × X - B||₂ and return both the solution
     * and the residual norm ||A × X - B||₂.
     * 
     * @param A Coefficient matrix of the least squares problem
     * @param B Right-hand side matrix of the least squares problem
     * @return Tuple2 containing the solution matrix X and the residual norm
     * @throws DecompositionFailedException if no suitable decomposition method can be applied
     * @throws IllegalArgumentException if matrices dimensions do not match
     */
    public static Tuple2<IMatrix<Double>, Double> solveWithResidual(IMatrix<Double> A, IMatrix<Double> B) {
        // Validate input matrices
        if (A == null || B == null) {
            throw new IllegalArgumentException("Input matrices cannot be null");
        }
        
        if (A.rows() != B.rows()) {
            throw new IllegalArgumentException(
                "Matrix row dimension mismatch: A has " + A.rows() + 
                " rows but B has " + B.rows() + " rows");
        }
        
        // Solve the least squares problem
        IMatrix<Double> X = solve(A, B);
        
        // Compute the residual: ||A × X - B||₂
        IMatrix<Double> residualMatrix = A.mmul(X).sub(B);
        double residualNorm = residualMatrix.frobeniusNorm();
        
        return new Tuple2<>(X, residualNorm);
    }
    
    /** 秩亏或分解判奇异时用容忍 QR 最小二乘；否则严格三角回代。 */
    private static IDecompositionSolver selectQrcpLeastSquaresSolver(
            IQrcpDecomposition qr, int nCols, double rankTol) {
        if (qr.getRank() < nCols || !qr.isNonSingular()) {
            return qr.createTolerantLeastSquaresSolver(rankTol);
        }
        return qr.getSolver();
    }

    private static boolean shouldUseConjugateGradientForLargeScale(IMatrix<Double> A) {
        int m = A.rows();
        int n = A.cols();
        if (m == 0 || n == 0) {
            return false;
        }
        return m >= hugeRowThreshold || (long) m * n >= hugeElementThreshold;
    }

    private static IMatrix<Double> solveLeastSquaresCgls(IMatrix<Double> A, IMatrix<Double> B) {
        int n = A.cols();
        int k = B.cols();
        if (k == 0) {
            return IMatrix.zeros(n, 0, Double.class);
        }
        IMatrix<Double> X = IMatrix.zeros(n, k, Double.class);
        for (int j = 0; j < k; j++) {
            IVector<Double> bj = B.getColumn(j);
            X.setColumn(j, solveLeastSquaresCglsSingleRhs(A, bj));
        }
        return X;
    }

    static IVector<Double> solveLeastSquaresCglsSingleRhs(IMatrix<Double> A, IVector<Double> b) {
        IVector<Double> br = Linalg.vector(b.toDoubleArray());
        int n = A.cols();
        IVector<Double> x = Linalg.zeros(n);
        IVector<Double> r = br.sub(multiplyA(A, x));
        IVector<Double> p = multiplyTransposeA(A, r);
        double pNrm = p.norm2Value();
        double refScale = Math.max(1e-12, pNrm);
        double rho = pNrm * pNrm;
        for (int iter = 0; iter < CG_MAX_ITERATIONS; iter++) {
            IVector<Double> q = multiplyA(A, p);
            double qNrm = q.norm2Value();
            double qNormSq = qNrm * qNrm;
            if (qNormSq <= 1e-30) {
                break;
            }
            double alpha = rho / qNormSq;
            x = x.add(p.multiplyByScalar(alpha));
            r = r.sub(q.multiplyByScalar(alpha));
            IVector<Double> s = multiplyTransposeA(A, r);
            double sNrm = s.norm2Value();
            double rhoNew = sNrm * sNrm;
            if (sNrm < CG_TOLERANCE * refScale) {
                break;
            }
            double beta = rhoNew / rho;
            p = s.add(p.multiplyByScalar(beta));
            rho = rhoNew;
        }
        return x;
    }

    private static IVector<Double> multiplyA(IMatrix<Double> A, IVector<Double> p) {
        int m = A.rows();
        int n = A.cols();
        double[] pd = p.toDoubleArray();
        if (pd.length != n) {
            throw new IllegalArgumentException("A v: v length must equal A.cols()");
        }
        double[] y = new double[m];
        for (int i = 0; i < m; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                sum += A.get(i, j) * pd[j];
            }
            y[i] = sum;
        }
        return Linalg.vector(y);
    }

    /** y = Aᵀ v (explicit row/column sum; avoids matrix transpose multiply inaccuracies on some paths). */
    private static IVector<Double> multiplyTransposeA(IMatrix<Double> A, IVector<Double> v) {
        int m = A.rows();
        int n = A.cols();
        double[] vd = v.toDoubleArray();
        if (vd.length != m) {
            throw new IllegalArgumentException("A^T v: v length must equal A.rows()");
        }
        double[] y = new double[n];
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int i = 0; i < m; i++) {
                sum += A.get(i, j) * vd[i];
            }
            y[j] = sum;
        }
        return Linalg.vector(y);
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

    /** Relative residual ‖AX−B‖_F / ‖B‖_F must be below a size-scaled tolerance (post QR refinement). */
    private static boolean leastSquaresResidualAcceptable(IMatrix<Double> a, IMatrix<Double> b, IMatrix<Double> x) {
        double rel = DenseIterativeRefinement.relativeResidualFrobenius(a, b, x);
        if (Double.isNaN(rel)) {
            return false;
        }
        double tol = 1e-9 * Math.max(1.0, Math.sqrt(a.rows()));
        return rel <= tol;
    }
}