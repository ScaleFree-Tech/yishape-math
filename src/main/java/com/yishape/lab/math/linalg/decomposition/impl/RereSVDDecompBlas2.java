package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.util.YishapeLogger;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.Decomps;
import com.yishape.lab.math.linalg.decomposition.IBidiagonalDecomposition;
import com.yishape.lab.math.linalg.decomposition.ISVDDecomposition;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.DecompositionErrorEstimate;
import com.yishape.lab.math.linalg.decomposition.solver.SVDDecompositionSolver;
import com.yishape.lab.math.util.RerePrecision;
import com.yishape.lab.math.linalg.decomposition.impl.support.AdaptiveEpsilon;
import com.yishape.lab.math.linalg.decomposition.impl.support.DiagonalPlusRankOneSymmetricEigenSolver;
import com.yishape.lab.util.Tuple3;

import java.util.Arrays;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import com.yishape.lab.math.linalg.RereDoubleMatrix;

/**
 * BLAS-2 SVD Implementation of singular value decomposition with enhanced numerical stability.
 * 标量逐列 BLAS-2 双对角化实现
 * <p>
 * This implementation computes the singular value decomposition (SVD) of a matrix A
 * such that A = U * S * V^T where U and V are orthogonal matrices and S is a diagonal
 * matrix of singular values.
 * </p>
 * <p><b>返回形状约定（与 {@link ISVDDecomposition} 一致）：</b>设 A 为 m×n，k = min(m,n)。</p>
 * <ul>
 *   <li>{@code U}：m×k，列正交（瘦型左因子；非“满 m×m”）</li>
 *   <li>{@code S}：长度为 k 的向量，非负且降序（经后处理）</li>
 *   <li>{@code V^T}：n×n（与双对角化路径中右因子存储一致；重构时使用前 k 个奇异值与 {@code U}、{@code V^T} 的相应部分）</li>
 * </ul>
 * <p>若行数或列数为 0，{@link #decompose(IMatrix, double, int)} 抛出 {@link IllegalArgumentException}。</p>
 * 
 * <h3>Algorithm Improvements</h3>
 * <ul>
 *   <li>Bidiagonalization preprocessing using Householder reflections</li>
 *   <li>QR algorithm with shifts for bidiagonal SVD</li>
 *   <li>Divide-and-conquer path (large matrices): 子块奇异值与正交阵在块对角基下拼成中间矩阵 {@code K}（对角块为子奇异值、连接处为秩一拐角），
 *       对 {@code K} 双对角化后再做双对角 QR-SVD；理想 O(n²) 合并依赖 LAPACK DLASD2 Givens 链规约到箭式
 *       {@code diag(σ)^2 + ρ zz^T} 后调用 {@link DiagonalPlusRankOneSymmetricEigenSolver}（替代 DLASD4 循环）及 DLASD3 式 GEMM
 *       更新向量。当前生产路径仍以稠密 {@code K}（由 {@link IMatrix} 累加构造）+ 双对角 QR-SVD 保证与对父双对角直接 QR-SVD 一致</li>
 *   <li>Better numerical stability with precision-aware comparisons</li>
 *   <li>Comprehensive error reporting with context information</li>
 *   <li>Efficient caching of computed results</li>
 *   <li>Configurable thresholds for numerical comparisons</li>
 * </ul>
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
public class RereSVDDecompBlas2 implements ISVDDecomposition {

    private static final YishapeLogger log = YishapeLogger.getLogger(RereSVDDecompBlas2.class);

    
    /** Cached value of U. */
    private IMatrix<Double> cachedU;
    /** Cached value of V. */
    private IMatrix<Double> cachedV;
    /** Cached singular values. */
    private IVector<Double> cachedS;
    /** Cached value of VT. */
    private IMatrix<Double> cachedVT;
    /** Determinant of the matrix. */
    private Double determinant;
    /** Cached singular values array. */
    private double[] cachedSingularValues;
    /** Condition number of the matrix. */
    private Double conditionNumber;
    /** Rank of the matrix. */
    private Integer rank;
    /** Cached error estimate (lazy). */
    private DecompositionErrorEstimate cachedErrorEstimate;
    /** Whether error estimate has been computed. */
    private boolean errorEstimateComputed = false;
    /** Epsilon for numerical comparisons. */
    private double epsilon;
    /** Maximum number of iterations. */
    private int maxIterations;
    /** The matrix being decomposed. */
    private IMatrix<Double> matrix;
    /** Whether to compute epsilon adaptively based on matrix condition. */
    private boolean useAdaptiveEpsilon;
    /**
     * When true, medium and large matrices (size > 1000) are delegated to the BLAS-3
     * {@link RereSVDDecompBlas3} path; small matrices still use {@link #traditionalSVD}.
     * Defaults to true so the BLAS-3 path is exercised by the broad SVD test surface.
     * Toggle via {@link #setUseFaerStyle(boolean)} for A/B comparison against the legacy
     * BLAS-2 {@link #bidiagonalSVD} + {@link #divideAndConquerSVD} routes.
     */
    private boolean useFaerStyle = true;

    /** Default epsilon for numerical comparisons. */
    private static final double DEFAULT_EPSILON = RerePrecision.getDefaultEpsilon();
    /**
     * Golub–Reinsch BD-SQR 主循环总步数上界。过小会在未完全收缩时退出，出现 O(1) 级重构误差（病态/结构化矩阵上尤甚）。
     * Apache Commons Math 同期实现不设此上界；此处取与规模相关的充裕默认值，必要时调用方仍可通过 {@link #decompose(IMatrix, double, int)} 收紧。
     */
    private static final int DEFAULT_MAX_ITERATIONS = 6_000_000;
    /** Absolute threshold for small singular values. */
    private static final double TINY = RerePrecision.getSafeMin();

    private static double fastHypot(double x, double y) {
        double ax = Math.abs(x);
        double ay = Math.abs(y);
        if (ax > 1e154 || ay > 1e154) {
            return Math.hypot(x, y);
        }
        return Math.sqrt(x * x + y * y);
    }
    
    /**
     * Default constructor with default parameters.
     */
    public RereSVDDecompBlas2() {
        this.epsilon = DEFAULT_EPSILON;
        this.maxIterations = DEFAULT_MAX_ITERATIONS;
        this.useAdaptiveEpsilon = false;
    }

    /**
     * Constructor with unified parameters.
     *
     * @param epsilon threshold for considering an element as zero
     * @param maxIterations maximum number of iterations
     */
    public RereSVDDecompBlas2(double epsilon, int maxIterations) {
        this.epsilon = epsilon;
        this.maxIterations = maxIterations;
        this.useAdaptiveEpsilon = false;
    }

    /**
     * Constructor with adaptive epsilon option.
     * When {@code adaptive} is true, epsilon is computed automatically
     * from a Hager-Higham condition number estimate at decompose time.
     *
     * @param adaptive if true, use adaptive epsilon selection
     */
    public RereSVDDecompBlas2(boolean adaptive) {
        this.epsilon = DEFAULT_EPSILON;
        this.maxIterations = DEFAULT_MAX_ITERATIONS;
        this.useAdaptiveEpsilon = adaptive;
    }

    /**
     * Enable or disable the BLAS-3 FaerStyle bidiagonalization route for medium/large matrices.
     * When disabled, falls back to the legacy BLAS-2 bidiagonalSVD / divide-and-conquer routing.
     */
    public void setUseFaerStyle(boolean useFaerStyle) {
        this.useFaerStyle = useFaerStyle;
    }

    public boolean isUseFaerStyle() {
        return useFaerStyle;
    }
    
    @Override
    public IDecompositionSolver getSolver() {
        if (cachedU == null || cachedS == null || cachedVT == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        // Return the standalone solver
        return new SVDDecompositionSolver(cachedU, cachedS, cachedVT, epsilon);
    }
    
    @Override
    public double getDeterminant() {
        if (determinant == null) {
            if (cachedS == null) {
                throw new IllegalStateException("Decomposition not yet performed");
            }
            double det = 1.0;
            for (int i = 0; i < cachedS.length(); i++) {
                det *= cachedS.get(i);
            }
            determinant = det;
        }
        return determinant;
    }
    
    @Override
    public boolean isNonSingular() {
        if (cachedS == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        // A matrix is non-singular if all singular values are non-zero
        for (int i = 0; i < cachedS.length(); i++) {
            if (Math.abs(cachedS.get(i)) < epsilon) {
                return false;
            }
        }
        return true;
    }
    
    @Override
    public double getConditionNumber() {
        if (conditionNumber == null) {
            if (cachedS == null) {
                throw new IllegalStateException("Decomposition not yet performed");
            }
            // Estimate condition number using the ratio of largest to smallest singular values
            if (cachedS.length() > 0) {
                double maxSingular = cachedS.get(0);
                double minSingular = cachedS.get(cachedS.length() - 1);
                if (minSingular > epsilon) {
                    conditionNumber = maxSingular / minSingular;
                } else {
                    conditionNumber = Double.POSITIVE_INFINITY;
                }
            } else {
                conditionNumber = Double.POSITIVE_INFINITY;
            }
        }
        return conditionNumber;
    }
    
    @Override
    public int getRank() {
        if (rank == null) {
            if (cachedS == null) {
                throw new IllegalStateException("Decomposition not yet performed");
            }
            // Rank is the number of non-zero singular values
            int r = 0;
            for (int i = 0; i < cachedS.length(); i++) {
                if (Math.abs(cachedS.get(i)) > epsilon) {
                    r++;
                }
            }
            rank = r;
        }
        return rank;
    }
    
    @Override
    public double getEpsilon() {
        return epsilon;
    }
    
    @Override
    public int getMaxIterations() {
        return maxIterations;
    }

    @Override
    public DecompositionErrorEstimate getErrorEstimate() {
        if (cachedU == null || cachedS == null || cachedVT == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        if (!errorEstimateComputed) {
            cachedErrorEstimate = computeErrorEstimate();
            errorEstimateComputed = true;
        }
        return cachedErrorEstimate != null ? cachedErrorEstimate : DecompositionErrorEstimate.NOT_AVAILABLE;
    }

    private DecompositionErrorEstimate computeErrorEstimate() {
        int m = cachedU.rows();
        int k = cachedS.length();
        int nFull = cachedVT.cols();
        long maxDim = Math.max(m, Math.max(k, nFull));
        if (maxDim > 4000) {
            log.warn("SVD error estimate skipped: matrix too large (max dim {})", maxDim);
            return DecompositionErrorEstimate.NOT_AVAILABLE;
        }
        double matrixNorm = this.matrix != null ? this.matrix.frobeniusNorm() : 0.0;
        if (matrixNorm == 0.0) {
            return new DecompositionErrorEstimate(0.0, 0.0, 0.0, getConditionNumber(), true);
        }
        IMatrix<Double> Sdiag = Linalg.diag(cachedS);
        IMatrix<Double> reconstructed = cachedU.mmul(Sdiag).mmul(cachedVT);
        IMatrix<Double> residual = this.matrix.sub(reconstructed);
        double backwardError = residual.frobeniusNorm() / matrixNorm;
        double cond = getConditionNumber();
        double forwardErrorBound = backwardError * cond;
        return new DecompositionErrorEstimate(backwardError, forwardErrorBound, 0.0, cond, true);
    }

    @Override
    public Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix) {
        return decompose(matrix, epsilon);
    }
    
    @Override
    public Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix, double epsilon) {
        return decompose(matrix, epsilon, DEFAULT_MAX_ITERATIONS);
    }
    
    @Override
    public Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix, double epsilon, int maxIterations) {
        // Reset cached values
        cachedU = null;
        cachedS = null;
        cachedVT = null;
        cachedV = null;
        determinant = null;
        conditionNumber = null;
        rank = null;
        cachedErrorEstimate = null;
        errorEstimateComputed = false;
        this.epsilon = epsilon;
        this.maxIterations = maxIterations;

        IDoubleMatrix doubleMatrix = (IDoubleMatrix) matrix;
        double[][] data = doubleMatrix.getData();

        int m = data.length;    // Matrix rows
        int n = (m > 0) ? data[0].length : 0; // Matrix columns
        if (m == 0 || n == 0) {
            throw new IllegalArgumentException("Matrix cannot be empty");
        }

        // Adaptive epsilon: override user-supplied epsilon with condition-based estimate
        if (useAdaptiveEpsilon) {
            this.epsilon = AdaptiveEpsilon.computeAdaptiveEpsilon(data);
        }

        // Store matrix for later use
        this.matrix = matrix;
        
        // 根据矩阵大小选择不同的SVD算法
        // 分治仅在元素积很大且 min 维足够大时划算；中小方阵（如 200×200）走双对角 + QR 远快于分治+递归拷贝。
        // BLAS-3 LABRD bidiagonalization (FaerStyle) 的 panel 开销与尾随更新同阶 O(nb*m*n)，
        // 在纯 Java 无原生 BLAS 的情况下无法摊销，反而比标量 BLAS-2 多 ~2× 浮点运算。
        // 实测 n=500 慢 1.5×，n=1000 慢 1.68×。仅在 >=4e6 元素（方阵 n>=2000）时才启用。
        int size = m * n;
        int mn = Math.min(m, n);
        final long divideConquerMinElements = 2_500_000L;
        final long faerStyleMinElements = 4_000_000L;
        if (size <= 1000) {
            // 小矩阵：双对角分解（RereBidiagonal）+ 对方阵 B 做 BD-SQR；与中档矩阵的 Golub–Reinsch 稠密约化为不同管线
            traditionalSVD(doubleMatrix);
        } else if (useFaerStyle && (long) size >= faerStyleMinElements) {
            // 大矩阵（>=4M 元素）：BLAS-3 FaerStyle（panel LABRD + 尾随 GEMM）。
            runFaerStyle(doubleMatrix);
        } else if ((long) size > divideConquerMinElements && mn >= 384) {
            divideAndConquerSVD(doubleMatrix);
        } else {
            // 双对角化 + bdsqr，与 Commons Math 一致预处理（含 m<n 时转置），保证瘦型路径正确性
            bidiagonalSVD(doubleMatrix);
        }
        
        // 与 ISVDDecomposition 约定一致：V^T 为 n×n（瘦型路径先补全正交列再转置）
        int nFull = matrix.cols();
        if (cachedV != null && cachedV.cols() < nFull) {
            cachedV = widenVToFullOrthogonal(cachedV, nFull, this.epsilon);
            cachedVT = cachedV.transposeNew();
        }

        // 确保奇异值为正数并正确排序
        ensureValidSingularValues();
        
        return new Tuple3<>(cachedU, cachedS, cachedVT);
    }
    
    /**
     * 传统SVD入口（小矩阵）：委托 {@link #optimizedSVD}，与 {@link #decompose} 中小矩阵分支一致。
     */
    private void traditionalSVD(IDoubleMatrix matrix) {
        optimizedSVD(matrix);
    }

    /**
     * 委托给 {@link RereSVDDecompBlas3} 的 BLAS-3 路径，并把结果回填到本实例的缓存字段，
     * 使得后续 {@link #ensureValidSingularValues}、getter 等流程保持一致。
     */
    private void runFaerStyle(IDoubleMatrix matrix) {
        RereSVDDecompBlas3 delegate = new RereSVDDecompBlas3(this.epsilon, this.maxIterations);
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> r = delegate.decompose(matrix);
        this.cachedU = r.getFirst();
        this.cachedS = r.getSecond();
        this.cachedVT = r.getThird();
        this.cachedV = (this.cachedVT != null) ? this.cachedVT.transposeNew() : null;
        // ensureValidSingularValues 需要 cachedSingularValues 与 cachedS 长度一致；
        // 历次调用 decompose 不会重置该字段，因此每条路径都必须自行分配匹配长度的数组。
        int k = this.cachedS.length();
        this.cachedSingularValues = new double[k];
        for (int i = 0; i < k; i++) {
            this.cachedSingularValues[i] = this.cachedS.get(i);
        }
    }
    
    /**
     * 分治SVD算法（适用于大矩阵）
     * 使用分治策略提高大矩阵的计算效率
     */
    private void divideAndConquerSVD(IDoubleMatrix matrix) {
        int m = matrix.getRowNum();
        int n = matrix.getColNum();
        int minDim = Math.min(m, n);
        // 宽矩阵（m < n）先转置再分解，避免双对角提取时丢失最后一条超对角元
        final boolean transposed = m < n;
        final IDoubleMatrix workMat = transposed
                ? (IDoubleMatrix) (IMatrix<Double>) matrix.transpose()
                : matrix;
        final int mWork = workMat.getRowNum();
        final int nWork = workMat.getColNum();

        try {
            // 步骤1：双对角化预处理（与优化方法相同）
            Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> bidiagResult =
                bidiagonalizationWithIMatrix(workMat);
            IMatrix<Double> U1 = bidiagResult.getFirst();   // (mWork×minDim)
            IMatrix<Double> B = bidiagResult.getSecond();   // (minDim×minDim)
            IMatrix<Double> V1 = bidiagResult.getThird();   // (nWork×nWork)

            // 验证双对角化结果
            if (U1 == null || B == null || V1 == null) {
                throw new RuntimeException("Bidiagonalization failed: null result");
            }

            // 步骤2：使用分治算法求解双对角矩阵的SVD
            Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> dcResult =
                divideAndConquerBidiagonalSVD(B);
            IVector<Double> singularValuesVector = dcResult.getFirst();
            IMatrix<Double> Q_left = dcResult.getSecond();   // (minDim×minDim)
            IMatrix<Double> Q_right = dcResult.getThird();   // (minDim×minDim)

            // 验证分治算法结果
            if (singularValuesVector == null || Q_left == null || Q_right == null) {
                throw new RuntimeException("Divide-and-conquer algorithm failed: null result");
            }

            // 步骤3：计算工作矩阵的 U 和 V
            IMatrix<Double> U_work = U1.mmul(Q_left);

            IMatrix<Double> V1_reduced = extractFirstColumns(V1, minDim);
            IMatrix<Double> V_work = V1_reduced.mmul(Q_right);

            if (!transposed) {
                IMatrix<Double> V_temp = assembleFullRightOrthogonal(V1, V_work, minDim);

                if (V_temp.rows() != n || V_temp.cols() != n) {
                    throw new RuntimeException(String.format(
                        "Invalid V dimensions: expected %dx%d, got %dx%d",
                        n, n, V_temp.rows(), V_temp.cols()));
                }

                cachedU = U_work;
                cachedV = V_temp;
            } else {
                // A^T = U_work * S * V_work^T → A = V_work * S * U_work^T
                cachedU = V_work;   // m×minDim
                cachedV = U_work;   // n×minDim, 稍后由 decompose() 补全
            }

            // 验证 U 维度
            if (cachedU.rows() != m || cachedU.cols() != minDim) {
                throw new RuntimeException(String.format(
                    "Invalid U dimensions: expected %dx%d, got %dx%d",
                    m, minDim, cachedU.rows(), cachedU.cols()));
            }

            // 确保奇异值为正数
            for (int i = 0; i < singularValuesVector.length(); i++) {
                if (singularValuesVector.get(i) < 0) {
                    singularValuesVector.set(i, -singularValuesVector.get(i));
                    for (int j = 0; j < cachedU.rows(); j++) {
                        cachedU.set(j, i, -cachedU.get(j, i));
                    }
                }
            }

            // 检查正交性并在必要时重新正交化
            double orthogonalityError = checkOrthogonality(cachedU);
            if (orthogonalityError > epsilon * 100) {
                orthogonalizeMatrixWithIMatrix(cachedU);
            }

            if (!transposed) {
                orthogonalityError = checkOrthogonality(cachedV);
                if (orthogonalityError > epsilon * 100) {
                    orthogonalizeMatrixWithIMatrix(cachedV);
                }
            }

            // 缓存结果
            cachedS = singularValuesVector;
            cachedVT = cachedV.transposeNew();

            // 缓存奇异值
            this.cachedSingularValues = new double[singularValuesVector.length()];
            for (int i = 0; i < singularValuesVector.length(); i++) {
                this.cachedSingularValues[i] = singularValuesVector.get(i);
            }

        } catch (Exception e) {
            // 如果分治算法失败，回退到双对角SVD
            log.warn("Divide-and-conquer SVD failed, falling back to bidiagonal SVD: " + e.getMessage());
            bidiagonalSVD(matrix);
        }
    }
    
    /**
     * 分治算法求解双对角矩阵的 SVD。
     * 超对角连接元非零时仍递归子块；合并时在块对角子 SVD 基下构造 {@code K}，双对角化后再 QR-SVD。
     */
    private Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> divideAndConquerBidiagonalSVD(
            IMatrix<Double> B) {
        int n = B.rows();

        // 基本情况：小矩阵直接使用QR算法
        if (n <= 8) {
            return qrAlgorithmForBidiagonalWithIMatrix(B, n, n);
        }

        // 分割点
        int mid = n / 2;
        double connectingElement = (mid > 0 && mid < n) ? B.get(mid - 1, mid) : 0.0;

        try {
            IMatrix<Double> B1 = extractSubmatrix(B, 0, mid, 0, mid);
            IMatrix<Double> B2 = extractSubmatrix(B, mid, n, mid, n);

            // 并行：右子节点 fork，左子节点在当前线程计算
            ForkJoinPool pool = RereDoubleMatrix.getDecompositionPool();
            boolean useParallel = pool != null && !pool.isShutdown()
                    && n >= RereDoubleMatrix.PARALLELISM_THRESHOLD;

            ForkJoinTask<Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>>> rightTask = null;
            if (useParallel) {
                rightTask = pool.submit(
                    () -> divideAndConquerBidiagonalSVD(B2));
            }

            Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> result1 =
                divideAndConquerBidiagonalSVD(B1);

            Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> result2;
            if (rightTask != null) {
                result2 = rightTask.join();
            } else {
                result2 = divideAndConquerBidiagonalSVD(B2);
            }

            return mergeSubproblems(result1, result2, connectingElement, B, mid);

        } catch (Exception e) {
            // 如果分治失败，回退到QR算法
            log.warn("Divide-and-conquer failed for bidiagonal matrix, using QR: " + e.getMessage());
            return qrAlgorithmForBidiagonalWithIMatrix(B, n, n);
        }
    }

    /**
     * Raw-array adapter for divide-and-conquer bidiagonal SVD.
     * Constructs a temporary bidiagonal IMatrix from {@code diag}/{@code offDiag},
     * runs the existing DC-SVD, and unpacks the result to raw arrays.
     * <p>Overhead is O(n²) which is negligible vs O(n² log n) or O(n³) DC-SVD.</p>
     *
     * @param diag      main diagonal entries, length n
     * @param offDiag   super-diagonal entries, length n-1 (offDiag[n-1] ignored)
     * @param n         matrix dimension
     * @param epsilon   numerical tolerance
     * @param maxIterations max iterations for QR fallback inside DC tree
     * @return Tuple3 of (singularValues[n], Q_left[n][n], Q_right[n][n])
     */
    public static Tuple3<double[], double[][], double[][]> divideAndConquerBidiagonalSVDRaw(
            double[] diag, double[] offDiag, int n, double epsilon, int maxIterations) {
        double[][] bData = new double[n][n];
        for (int i = 0; i < n; i++) {
            bData[i][i] = diag[i];
            if (i + 1 < n) {
                bData[i][i + 1] = offDiag[i];
            }
        }
        IMatrix<Double> B = Linalg.matrix(bData);

        RereSVDDecompBlas2 temp = new RereSVDDecompBlas2();
        temp.epsilon = epsilon;
        temp.maxIterations = maxIterations;

        var result = temp.divideAndConquerBidiagonalSVD(B);

        double[] s = new double[n];
        IVector<Double> sVec = result.getFirst();
        for (int i = 0; i < n; i++) {
            s[i] = sVec.get(i);
        }

        double[][] qL = ((IDoubleMatrix) result.getSecond()).getData();
        double[][] qR = ((IDoubleMatrix) result.getThird()).getData();

        return new Tuple3<>(s, qL, qR);
    }

    /**
     * 提取子矩阵
     */
    private IMatrix<Double> extractSubmatrix(IMatrix<Double> matrix, int startRow, int endRow,
                                            int startCol, int endCol) {
        int rows = endRow - startRow;
        int cols = endCol - startCol;
        
        double[][] data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = matrix.get(startRow + i, startCol + j);
            }
        }
        
        return Linalg.matrix(data);
    }
    
    /**
     * 合并两个子问题的 SVD 结果。
     * <p>
     * 若双对角块之间超对角耦合在容差内为零，则为块双对角直和再按奇异值排序。
     * 若耦合非零，在块对角子 SVD 基下令 {@code K = U_blk^T B V_blk}（{@code B} 为父上双对角），
     * 对 {@code K} 双对角化并 QR-SVD，再将左右正交阵与块对角的 U₁,V₁,U₂,V₂ 相乘。失败时回退对父双对角 QR-SVD。
     * LAPACK 式 O(n²) 合并可将对称秩一特征子问题换为 {@link DiagonalPlusRankOneSymmetricEigenSolver}。
     * </p>
     */
    private Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> mergeSubproblems(
            Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> result1,
            Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> result2,
            double connectingElement,
            IMatrix<Double> parentB,
            int mid) {

        int nParent = parentB.rows();
        if (!RerePrecision.equalsZero(connectingElement, epsilon)) {
            try {
                return mergeRankOneCoupling(result1, result2, mid, nParent, parentB);
            } catch (Exception ex) {
                log.warn("rank-one merge failed, using QR bidiagonal SVD: {}", ex.getMessage());
                return qrAlgorithmForBidiagonalWithIMatrix(parentB, nParent, nParent);
            }
        }

        IVector<Double> s1 = result1.getFirst();
        IMatrix<Double> u1 = result1.getSecond();
        IMatrix<Double> v1 = result1.getThird();
        IVector<Double> s2 = result2.getFirst();
        IMatrix<Double> u2 = result2.getSecond();
        IMatrix<Double> v2 = result2.getThird();

        int n1 = s1.length();
        int n2 = s2.length();
        int totalSize = n1 + n2;
        if (totalSize != nParent) {
            return qrAlgorithmForBidiagonalWithIMatrix(parentB, nParent, nParent);
        }

        IVector<Double> mergedSingularValues = Linalg.zeros(totalSize);
        for (int i = 0; i < n1; i++) {
            mergedSingularValues.set(i, s1.get(i));
        }
        for (int i = 0; i < n2; i++) {
            mergedSingularValues.set(n1 + i, s2.get(i));
        }
        IMatrix<Double> mergedU = Linalg.zeros(totalSize, totalSize);
        IMatrix<Double> mergedV = Linalg.zeros(totalSize, totalSize);
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n1; j++) {
                mergedU.set(i, j, u1.get(i, j));
                mergedV.set(i, j, v1.get(i, j));
            }
        }
        for (int i = 0; i < n2; i++) {
            for (int j = 0; j < n2; j++) {
                mergedU.set(n1 + i, n1 + j, u2.get(i, j));
                mergedV.set(n1 + i, n1 + j, v2.get(i, j));
            }
        }
        sortSingularValuesWithMatrices(mergedSingularValues, mergedU, mergedV);
        for (int i = 0; i < totalSize; i++) {
            if (mergedSingularValues.get(i) < 0) {
                mergedSingularValues.set(i, -mergedSingularValues.get(i));
                for (int k = 0; k < mergedU.rows(); k++) {
                    mergedU.set(k, i, -mergedU.get(k, i));
                }
            }
        }
        return new Tuple3<>(mergedSingularValues, mergedU, mergedV);
    }

    /**
     * 在 blkdiag(U₁,U₂)、blkdiag(V₁,V₂) 正交基下合并：{@code B = U_blk K V_blk^T}，故
     * {@code K = U_blk^T B V_blk}（与 dlasd2 的 (α,β) 秩一分解一致，不再用仅超对角启发式秩一项）。
     * <p>
     * 优化：直接由结构构造 K = [Σ₁  α·u·vᵀ; 0  Σ₂]，避免构建稠密 U_blk、V_blk 及 O(n³) 矩阵乘法。
     * U_out、V_out 亦通过分块乘直接生成，不落地完整块对角阵。
     * 理论上 Gu-Eisenstat O(n²) D&amp;C 合并可将此进一步约化为对称秩一特征子问题
     * （参见 {@link DiagonalPlusRankOneSymmetricEigenSolver}），但推导涉及完整的 deflate +
     * secular 根求 + 向量更新链条；当前仍沿用双对角化+QR-SVD 保证正确性。
     * </p>
     */
    private Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> mergeRankOneCoupling(
            Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> result1,
            Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> result2,
            int mid,
            int nParent,
            IMatrix<Double> parentB) {
        IVector<Double> s1 = result1.getFirst();
        IMatrix<Double> u1 = result1.getSecond();
        IMatrix<Double> v1 = result1.getThird();
        IVector<Double> s2 = result2.getFirst();
        IMatrix<Double> u2 = result2.getSecond();
        IMatrix<Double> v2 = result2.getThird();
        int n1 = s1.length();
        int n2 = s2.length();
        if (n1 + n2 != nParent || n1 != mid) {
            throw new IllegalStateException("merge rank-one: dimension mismatch n1=" + n1 + " n2=" + n2 + " mid=" + mid);
        }
        if (u1.rows() != n1 || u1.cols() != n1 || v1.rows() != n1 || v1.cols() != n1
                || u2.rows() != n2 || u2.cols() != n2 || v2.rows() != n2 || v2.cols() != n2) {
            throw new IllegalStateException("merge rank-one: child U/V not square orthogonal-sized");
        }

        double alpha = parentB.get(n1 - 1, n1);
        double[] uRow = new double[n1];
        for (int j = 0; j < n1; j++) {
            uRow[j] = u1.get(n1 - 1, j);
        }
        double[] vRow = new double[n2];
        for (int j = 0; j < n2; j++) {
            vRow[j] = v2.get(0, j);
        }

        double[][] kData = new double[nParent][nParent];
        for (int i = 0; i < n1; i++) {
            kData[i][i] = s1.get(i);
        }
        for (int i = 0; i < n2; i++) {
            kData[n1 + i][n1 + i] = s2.get(i);
        }
        for (int i = 0; i < n1; i++) {
            double aui = alpha * uRow[i];
            for (int j = 0; j < n2; j++) {
                kData[i][n1 + j] = aui * vRow[j];
            }
        }

        IMatrix<Double> kMat = Linalg.matrix(kData);

        double scale = 0.0;
        for (int i = 0; i < nParent; i++) {
            scale = Math.max(scale, Math.abs(parentB.get(i, i)));
        }

        Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> bi = bidiagonalizationWithIMatrix(kMat);
        IMatrix<Double> ub = bi.getFirst();
        IMatrix<Double> bbd = bi.getSecond();
        IMatrix<Double> vb = bi.getThird();
        Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> inner =
                qrAlgorithmForBidiagonalWithIMatrix(bbd, nParent, nParent);
        IVector<Double> sigma = inner.getFirst();
        IMatrix<Double> ql = inner.getSecond();
        IMatrix<Double> qrMat = inner.getThird();
        IMatrix<Double> uK = ub.mmul(ql);
        IMatrix<Double> vK = vb.mmul(qrMat);

        IMatrix<Double> uOut = Linalg.zeros(nParent, nParent);
        IMatrix<Double> vOut = Linalg.zeros(nParent, nParent);
        for (int j = 0; j < nParent; j++) {
            for (int i = 0; i < n1; i++) {
                double su = 0.0, sv = 0.0;
                for (int k = 0; k < n1; k++) {
                    su += u1.get(i, k) * uK.get(k, j);
                    sv += v1.get(i, k) * vK.get(k, j);
                }
                uOut.set(i, j, su);
                vOut.set(i, j, sv);
            }
            for (int i = 0; i < n2; i++) {
                double su = 0.0, sv = 0.0;
                for (int k = 0; k < n2; k++) {
                    su += u2.get(i, k) * uK.get(n1 + k, j);
                    sv += v2.get(i, k) * vK.get(n1 + k, j);
                }
                uOut.set(n1 + i, j, su);
                vOut.set(n1 + i, j, sv);
            }
        }

        sortSingularValuesWithIMatrix(sigma, uOut, vOut);

        double fullErr = 0.0;
        for (int i = 0; i < nParent; i++) {
            for (int j = 0; j < nParent; j++) {
                double sum = 0.0;
                for (int t = 0; t < nParent; t++) {
                    sum += uOut.get(i, t) * sigma.get(t) * vOut.get(j, t);
                }
                fullErr = Math.max(fullErr, Math.abs(parentB.get(i, j) - sum));
            }
        }
        if (!(fullErr <= Math.max(epsilon * 10, epsilon * 100 * scale) * nParent)) {
            throw new IllegalStateException("merged parent SVD reconstruction max " + fullErr);
        }
        return new Tuple3<>(sigma, uOut, vOut);
    }
    
    /**
     * 排序奇异值及对应的矩阵列
     */
    private void sortSingularValuesWithMatrices(IVector<Double> singularValues,
                                               IMatrix<Double> U, IMatrix<Double> V) {
        int n = singularValues.length();
        
        // 使用快速排序（降序）
        quickSortWithMatrices(singularValues, U, V, 0, n - 1);
    }
    
    /**
     * 快速排序实现
     */
    private void quickSortWithMatrices(IVector<Double> singularValues,
                                      IMatrix<Double> U, IMatrix<Double> V,
                                      int low, int high) {
        if (low < high) {
            int pi = partitionWithMatrices(singularValues, U, V, low, high);
            quickSortWithMatrices(singularValues, U, V, low, pi - 1);
            quickSortWithMatrices(singularValues, U, V, pi + 1, high);
        }
    }
    
    /**
     * 分区操作
     */
    private int partitionWithMatrices(IVector<Double> singularValues,
                                     IMatrix<Double> U, IMatrix<Double> V,
                                     int low, int high) {
        double pivot = Math.abs(singularValues.get(high));
        int i = (low - 1);
        
        for (int j = low; j < high; j++) {
            if (Math.abs(singularValues.get(j)) >= pivot) {
                i++;
                
                // 交换奇异值
                double temp = singularValues.get(i);
                singularValues.set(i, singularValues.get(j));
                singularValues.set(j, temp);
                
                // 交换U的对应列
                for (int k = 0; k < U.rows(); k++) {
                    double tempU = U.get(k, i);
                    U.set(k, i, U.get(k, j));
                    U.set(k, j, tempU);
                }
                
                // 交换V的对应列
                for (int k = 0; k < V.rows(); k++) {
                    double tempV = V.get(k, i);
                    V.set(k, i, V.get(k, j));
                    V.set(k, j, tempV);
                }
            }
        }
        
        // 交换奇异值
        double temp = singularValues.get(i + 1);
        singularValues.set(i + 1, singularValues.get(high));
        singularValues.set(high, temp);
        
        // 交换U的对应列
        for (int k = 0; k < U.rows(); k++) {
            double tempU = U.get(k, i + 1);
            U.set(k, i + 1, U.get(k, high));
            U.set(k, high, tempU);
        }
        
        // 交换V的对应列
        for (int k = 0; k < V.rows(); k++) {
            double tempV = V.get(k, i + 1);
            V.set(k, i + 1, V.get(k, high));
            V.set(k, high, tempV);
        }
        
        return i + 1;
    }
    /**
     * BD-SQR 主迭代 — 原始布局（V[row][col]）。
     * 保留用于 {@link #qrAlgorithmForBidiagonalWithIMatrix}（IMatrix 管线，小矩阵/D&C 子问题）。
     * 大矩阵走 {@link #bdsqrMainLoopVt}（转置 V，stride-1 缓存优化）。
     */
    private void bdsqrMainLoop(double[] singularValues, double[] e, IMatrix<Double> U, IMatrix<Double> V,
                               int m, int n, int pInitial) {
        // U、V 均为分解过程中新创建的矩阵（非原始输入），直接操作其裸数组可消除 IMatrix get/set 虚分派。
        // 然而低维度（≤4）在 2026-05 前后的回归测试中仍出现 O(1) 级重构误差，疑为转换遗留。
        final double[][] uData = ((IDoubleMatrix) U).getData();
        final double[][] vData = ((IDoubleMatrix) V).getData();
        if (log.isDebugEnabled()) {
            log.debug("[bdsqr] pInitial={} m={} n={} maxIter={}", pInitial, m, n, maxIterations);
        }

        int p = pInitial;
        int iterCount = 0;
        // 全局降序排序推迟至 ensureValidSingularValues()（O(n log n) 索引排序 + 批量拷贝）。
        // JAMA-style O(n²) bubble sort per deflation 已移除——曾是 n≥500 时的主要瓶颈。
        while (p > 0 && iterCount < maxIterations) {
            int k;
            int kase;
            if (log.isTraceEnabled()) log.trace("[bdsqr] iter={} p={}", iterCount, p);
            
            // This section inspects for negligible elements in the s and e arrays.
            // Improved deflation check with better numerical stability
            for (k = p - 2; k >= 0; k--) {
                final double threshold = TINY + epsilon * (Math.abs(singularValues[k]) +
                        Math.abs(singularValues[k + 1]));
                
                if (Math.abs(e[k]) <= threshold || Double.isNaN(e[k])) {
                    e[k] = 0;
                    break;
                }
            }

            if (k == p - 2) {
                kase = 4;
            } else {
                int ks;
                for (ks = p - 1; ks >= k; ks--) {
                    if (ks == k) {
                        break;
                    }
                    final double t = (ks != p ? Math.abs(e[ks]) : 0) +
                        (ks != k + 1 ? Math.abs(e[ks - 1]) : 0);
                    if (Math.abs(singularValues[ks]) <= TINY + epsilon * t) {
                        singularValues[ks] = 0;
                        break;
                    }
                }
                if (ks == k) {
                    kase = 3;
                } else if (ks == p - 1) {
                    kase = 1;
                } else {
                    kase = 2;
                    k = ks;
                }
            }
            k++;
            
            // Perform the task indicated by kase.
            double f;
            switch (kase) {
                case 1 -> {
                    f = e[p - 2];
                    e[p - 2] = 0;
                    for (int j = p - 2; j >= k; j--) {
                        double t = fastHypot(singularValues[j], f);
                        final double cs = singularValues[j] / t;
                        final double sn = f / t;
                        singularValues[j] = t;
                        if (j != k) {
                            f = -sn * e[j - 1];
                            e[j - 1] = cs * e[j - 1];
                        }
                        
                        for (int i = 0; i < n; i++) {
                            double viJ = vData[i][j];
                            double viP = vData[i][p - 1];
                            vData[i][j] = cs * viJ + sn * viP;
                            vData[i][p - 1] = -sn * viJ + cs * viP;
                        }
                    }
                }
                case 2 -> {
                    f = e[k - 1];
                    e[k - 1] = 0;
                    for (int j = k; j < p; j++) {
                        double t = fastHypot(singularValues[j], f);
                        final double cs = singularValues[j] / t;
                        final double sn = f / t;
                        singularValues[j] = t;
                        f = -sn * e[j];
                        e[j] = cs * e[j];
                        
                        // U 旋转 (uData[row][col], 内层 row 变化 → stride-n)。
                        // 简单循环，C2 自动向量化处理跨行访问。
                        for (int i = 0; i < m; i++) {
                            double uiJ = uData[i][j];
                            double uiK = uData[i][k - 1];
                            uData[i][j] = cs * uiJ + sn * uiK;
                            uData[i][k - 1] = -sn * uiJ + cs * uiK;
                        }
                    }
                }
                case 3 -> {
                    // Calculate the shift using Wilkinson shift for better convergence.
                    double shift = 0;
                    double sp = 0, spm1 = 0, epm1 = 0, sk = 0, ek = 0;
                    if (p >= 2) {
                        final double maxPm1Pm2 = Math.max(Math.abs(singularValues[p - 1]),
                                Math.abs(singularValues[p - 2]));
                        final double scale = Math.max(Math.max(Math.max(maxPm1Pm2,
                                Math.abs(e[p - 2])),
                                Math.abs(singularValues[k])),
                                Math.abs(e[k]));
                        
                        if (scale != 0) {
                            sp = singularValues[p - 1] / scale;
                            spm1 = singularValues[p - 2] / scale;
                            epm1 = e[p - 2] / scale;
                            sk = singularValues[k] / scale;
                            ek = e[k] / scale;
                            final double b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2.0;
                            final double c = (sp * epm1) * (sp * epm1);
                            
                            if (b != 0 || c != 0) {
                                // Improved Wilkinson shift calculation with better numerical stability
                                double discriminant = b * b + c;
                                if (discriminant >= 0) {
                                    double sqrtDisc = Math.sqrt(discriminant);
                                    if (b < 0) {
                                        sqrtDisc = -sqrtDisc;
                                    }
                                    shift = c / (b + sqrtDisc);
                                } else {
                                    // Handle complex case by using real part
                                    shift = 0;
                                }
                            }
                        }
                    }
                    f = (sk + sp) * (sk - sp) + shift;
                    double g = sk * ek;
                    // Chase zeros.
                    for (int j = k; j < p - 1; j++) {
                        double t = fastHypot(f, g);
                        double cs = (Math.abs(t) > epsilon) ? f / t : 1.0;
                        double sn = (Math.abs(t) > epsilon) ? g / t : 0.0;
                        if (j != k) {
                            e[j - 1] = t;
                        }
                        f = cs * singularValues[j] + sn * e[j];
                        e[j] = cs * e[j] - sn * singularValues[j];
                        g = sn * singularValues[j + 1];
                        singularValues[j + 1] = cs * singularValues[j + 1];
                        
                        for (int i = 0; i < n; i++) {
                            double viJ = vData[i][j];
                            double viJ1 = vData[i][j + 1];
                            vData[i][j] = cs * viJ + sn * viJ1;
                            vData[i][j + 1] = -sn * viJ + cs * viJ1;
                        }

                        t = fastHypot(f, g);
                        cs = (Math.abs(t) > epsilon) ? f / t : 1.0;
                        sn = (Math.abs(t) > epsilon) ? g / t : 0.0;
                        singularValues[j] = t;
                        f = cs * e[j] + sn * singularValues[j + 1];
                        singularValues[j + 1] = -sn * e[j] + cs * singularValues[j + 1];
                        g = sn * e[j + 1];
                        e[j + 1] = cs * e[j + 1];

                        // U 旋转 (uData[row][col], 内层 row 变化 → stride-n)。
                        // 简单循环，避免手动展开干扰 C2 决策。
                        if (j < m - 1) {
                            for (int i = 0; i < m; i++) {
                                double uiJ = uData[i][j];
                                double uiJ1 = uData[i][j + 1];
                                uData[i][j] = cs * uiJ + sn * uiJ1;
                                uData[i][j + 1] = -sn * uiJ + cs * uiJ1;
                            }
                        }
                    }
                    e[p - 2] = f;
                }
                default -> {
                    // Converged singular value: ensure non-negative.
                    if (singularValues[k] <= 0) {
                        singularValues[k] = singularValues[k] < 0 ? -singularValues[k] : 0;
                        for (int i = 0; i < n; i++) {
                            vData[i][k] = -vData[i][k];
                        }
                    }

                    // Global descending sort + sign fix deferred to
                    // ensureValidSingularValues() after the BD-SQR loop exits.
                    // JAMA O(n²) bubble sort per deflation removed —
                    // was the primary bottleneck for matrices with n ≥ 500.
                    iterCount = 0;
                    p--;
                }
            }
            // Deflate negligible s(p).
            // Split at negligible s(k).
            // Perform one qr step.
            // Convergence.
            iterCount++;
        }
    }

    // 就地转置方阵：使 vData[col][row] = V[row][col]，将 BD-SQR 中对 V 列的 Givens
    // 旋转变为对 V^T 行的旋转，从而在 Java row-major double[][] 上实现 stride-1 访问。
    private static void transposeSquareInPlace(double[][] a, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double t = a[i][j];
                a[i][j] = a[j][i];
                a[j][i] = t;
            }
        }
    }

    /**
     * BD-SQR 主迭代 — U^T + V^T 转置版本。
     * <p>
     * 假定 uData 和 vData 均已转置（uData[col][row] = U[row][col]，
     * vData[col][row] = V[row][col]），将 Givens 列旋转变为行旋转：
     * <ul>
     *   <li>原始模式 data[row][col]：内层 row 变化 → stride-n → 每缓存行仅用到 1 个有效 double</li>
     *   <li>转置模式 data[col][row]：内层 row 在第二维 → stride-1 → 同一缓存行 8 个 double 全被利用</li>
     * </ul>
     * 调用方负责在进入前转置 U 和 V，并在返回后恢复为正常 row-major 布局。
     * </p>
     */
    private void bdsqrMainLoopVt(double[] singularValues, double[] e, IMatrix<Double> U, IMatrix<Double> V,
                                  int m, int n, int pInitial) {
        final double[][] uData = ((IDoubleMatrix) U).getData();
        final double[][] vData = ((IDoubleMatrix) V).getData();
        if (log.isDebugEnabled()) {
            log.debug("[bdsqr-Vt] pInitial={} m={} n={} maxIter={}", pInitial, m, n, maxIterations);
        }

        int p = pInitial;
        int iterCount = 0;
        while (p > 0 && iterCount < maxIterations) {
            int k;
            int kase;
            if (log.isTraceEnabled()) log.trace("[bdsqr-Vt] iter={} p={}", iterCount, p);

            for (k = p - 2; k >= 0; k--) {
                final double threshold = TINY + epsilon * (Math.abs(singularValues[k]) +
                        Math.abs(singularValues[k + 1]));
                if (Math.abs(e[k]) <= threshold || Double.isNaN(e[k])) {
                    e[k] = 0;
                    break;
                }
            }

            if (k == p - 2) {
                kase = 4;
            } else {
                int ks;
                for (ks = p - 1; ks >= k; ks--) {
                    if (ks == k) {
                        break;
                    }
                    final double t = (ks != p ? Math.abs(e[ks]) : 0) +
                        (ks != k + 1 ? Math.abs(e[ks - 1]) : 0);
                    if (Math.abs(singularValues[ks]) <= TINY + epsilon * t) {
                        singularValues[ks] = 0;
                        break;
                    }
                }
                if (ks == k) {
                    kase = 3;
                } else if (ks == p - 1) {
                    kase = 1;
                } else {
                    kase = 2;
                    k = ks;
                }
            }
            k++;

            double f;
            switch (kase) {
                case 1 -> {
                    f = e[p - 2];
                    e[p - 2] = 0;
                    for (int j = p - 2; j >= k; j--) {
                        double t = fastHypot(singularValues[j], f);
                        final double cs = singularValues[j] / t;
                        final double sn = f / t;
                        singularValues[j] = t;
                        if (j != k) {
                            f = -sn * e[j - 1];
                            e[j - 1] = cs * e[j - 1];
                        }

                        // vData[col][row] 转置存储：对 V 列 j 和 p-1 的 Givens 旋转
                        // 变为对 V^T 行 j 和 p-1 的旋转。内层 i 变化在第二维 → stride-1。
                        // 使用简单循环而非 4x 展开——V^T stride-1 让 C2 的 SuperWord 自动向量化
                        // 能生成 SIMD FMA（4 元素/周期），手动展开反而阻止向量化。
                        for (int i = 0; i < n; i++) {
                            double viJ = vData[j][i];
                            double viP = vData[p - 1][i];
                            vData[j][i] = cs * viJ + sn * viP;
                            vData[p - 1][i] = -sn * viJ + cs * viP;
                        }
                    }
                }
                case 2 -> {
                    f = e[k - 1];
                    e[k - 1] = 0;
                    for (int j = k; j < p; j++) {
                        double t = fastHypot(singularValues[j], f);
                        final double cs = singularValues[j] / t;
                        final double sn = f / t;
                        singularValues[j] = t;
                        f = -sn * e[j];
                        e[j] = cs * e[j];

                        // U^T 旋转 (uData[col][row], 内层 row 变化 → stride-1)。
                        // U 已转置为 [n][m]，列旋转变为行旋转，缓存行利用率 8×。
                        for (int i = 0; i < m; i++) {
                            double uiJ = uData[j][i];
                            double uiK = uData[k - 1][i];
                            uData[j][i] = cs * uiJ + sn * uiK;
                            uData[k - 1][i] = -sn * uiJ + cs * uiK;
                        }
                    }
                }
                case 3 -> {
                    double shift = 0;
                    double sp = 0, spm1 = 0, epm1 = 0, sk = 0, ek = 0;
                    if (p >= 2) {
                        final double maxPm1Pm2 = Math.max(Math.abs(singularValues[p - 1]),
                                Math.abs(singularValues[p - 2]));
                        final double scale = Math.max(Math.max(Math.max(maxPm1Pm2,
                                Math.abs(e[p - 2])),
                                Math.abs(singularValues[k])),
                                Math.abs(e[k]));

                        if (scale != 0) {
                            sp = singularValues[p - 1] / scale;
                            spm1 = singularValues[p - 2] / scale;
                            epm1 = e[p - 2] / scale;
                            sk = singularValues[k] / scale;
                            ek = e[k] / scale;
                            final double b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2.0;
                            final double c = (sp * epm1) * (sp * epm1);

                            if (b != 0 || c != 0) {
                                double discriminant = b * b + c;
                                if (discriminant >= 0) {
                                    double sqrtDisc = Math.sqrt(discriminant);
                                    if (b < 0) {
                                        sqrtDisc = -sqrtDisc;
                                    }
                                    shift = c / (b + sqrtDisc);
                                } else {
                                    shift = 0;
                                }
                            }
                        }
                    }
                    f = (sk + sp) * (sk - sp) + shift;
                    double g = sk * ek;
                    for (int j = k; j < p - 1; j++) {
                        double t = fastHypot(f, g);
                        double cs = (Math.abs(t) > epsilon) ? f / t : 1.0;
                        double sn = (Math.abs(t) > epsilon) ? g / t : 0.0;
                        if (j != k) {
                            e[j - 1] = t;
                        }
                        f = cs * singularValues[j] + sn * e[j];
                        e[j] = cs * e[j] - sn * singularValues[j];
                        g = sn * singularValues[j + 1];
                        singularValues[j + 1] = cs * singularValues[j + 1];

                        // V 列 j 和 j+1 的 Givens 旋转 → V^T 行 j 和 j+1。Stride-1。
                        // 简单循环，C2 自动向量化生成 SIMD FMA。
                        for (int i = 0; i < n; i++) {
                            double viJ = vData[j][i];
                            double viJ1 = vData[j + 1][i];
                            vData[j][i] = cs * viJ + sn * viJ1;
                            vData[j + 1][i] = -sn * viJ + cs * viJ1;
                        }

                        t = fastHypot(f, g);
                        cs = (Math.abs(t) > epsilon) ? f / t : 1.0;
                        sn = (Math.abs(t) > epsilon) ? g / t : 0.0;
                        singularValues[j] = t;
                        f = cs * e[j] + sn * singularValues[j + 1];
                        singularValues[j + 1] = -sn * e[j] + cs * singularValues[j + 1];
                        g = sn * e[j + 1];
                        e[j + 1] = cs * e[j + 1];

                        // U^T 旋转 (uData[col][row], 内层 row 变化 → stride-1)。
                        // U 已转置为 [n][m]，列旋转变为行旋转，缓存行利用率 8×。
                        if (j < m - 1) {
                            for (int i = 0; i < m; i++) {
                                double uiJ = uData[j][i];
                                double uiJ1 = uData[j + 1][i];
                                uData[j][i] = cs * uiJ + sn * uiJ1;
                                uData[j + 1][i] = -sn * uiJ + cs * uiJ1;
                            }
                        }
                    }
                    e[p - 2] = f;
                }
                default -> {
                    if (singularValues[k] <= 0) {
                        singularValues[k] = singularValues[k] < 0 ? -singularValues[k] : 0;
                        for (int i = 0; i < n; i++) {
                            vData[k][i] = -vData[k][i];
                        }
                    }
                    iterCount = 0;
                    p--;
                }
            }
            iterCount++;
        }
    }

    private void bidiagonalSVD(IDoubleMatrix matrix) {
        final int m0 = matrix.getRowNum();
        final int n0 = matrix.getColNum();
        if (m0 == 0 || n0 == 0) {
            throw new IllegalArgumentException("Matrix cannot be empty");
        }
        // 与 Apache Commons Math SingularValueDecomposition 一致：较大维始终在「行」一侧，避免 m<n 时整条约化维数混乱
        final boolean transposed = m0 < n0;
        final IDoubleMatrix workMat = transposed
                ? (IDoubleMatrix) (IMatrix<Double>) matrix.transpose()
                : matrix;
        int m = workMat.getRowNum();
        int n = workMat.getColNum();

        double[][] A = new double[m][n];
        double[][] srcData = ((IDoubleMatrix) workMat).getData();
        for (int i = 0; i < m; i++) {
            System.arraycopy(srcData[i], 0, A[i], 0, n);
        }

        double[][] uData = new double[m][n];
        double[][] vData = new double[n][n];

        double[] singularValues = new double[n];
        double[] e = new double[n];
        double[] work = new double[m];
        // Pre-allocated buffer for rank-1 update column factors (max size n-1, reused per k-iteration).
        double[] tVals = new double[Math.max(0, n - 1)];

        int nct = Math.min(m - 1, n);
        int nrt = Math.max(0, n - 2);
        
        for (int k = 0; k < Math.max(nct, nrt); k++) {
            if (k < nct) {
                // Compute the transformation for the k-th column and
                // place the k-th diagonal in singularValues[k].
                // Compute 2-norm of k-th column without under/overflow.
                singularValues[k] = 0;
                for (int i = k; i < m; i++) {
                    singularValues[k] = fastHypot(singularValues[k], A[i][k]);
                }
                if (singularValues[k] != 0) {
                    if (A[k][k] < 0) {
                        singularValues[k] = -singularValues[k];
                    }
                    for (int i = k; i < m; i++) {
                        A[i][k] /= singularValues[k];
                    }
                    A[k][k] += 1;
                }
                singularValues[k] = -singularValues[k];
            }
            
            for (int j = k + 1; j < n; j++) {
                if (k < nct && singularValues[k] != 0) {
                    // Apply the transformation.
                    double t = 0;
                    for (int i = k; i < m; i++) {
                        t += A[i][k] * A[i][j];
                    }
                    t = -t / A[k][k];
                    for (int i = k; i < m; i++) {
                        A[i][j] += t * A[i][k];
                    }
                }
                // Place the k-th row of A into e for the
                // subsequent calculation of the row transformation.
                e[j] = A[k][j];
            }
            
            if (k < nct) {
                for (int i = k; i < m; i++) {
                    uData[i][k] = A[i][k];
                }
            }
            
            if (k < nrt) {
                // Compute the k-th row transformation and place the
                // k-th super-diagonal in e[k].
                // Compute 2-norm without under/overflow.
                e[k] = 0;
                for (int i = k + 1; i < n; i++) {
                    e[k] = fastHypot(e[k], e[i]);
                }
                if (e[k] != 0) {
                    if (e[k + 1] < 0) {
                        e[k] = -e[k];
                    }
                    for (int i = k + 1; i < n; i++) {
                        e[i] /= e[k];
                    }
                    e[k + 1] += 1;
                }
                e[k] = -e[k];
                
                if (k + 1 < m && e[k] != 0) {
                    // Apply the transformation. Loop-interchanged: inner loop walks
                    // columns (stride-1) instead of rows (stride-n), improving
                    // cache-line utilization by up to 8x on Java row-major arrays.
                    for (int i = k + 1; i < m; i++) {
                        double s = 0;
                        for (int j = k + 1; j < n; j++) {
                            s += e[j] * A[i][j];
                        }
                        work[i] = s;
                    }
                    // Rank-1 update: A += t_j * work[i]. Precompute t per column,
                    // loop-interchange so inner loop is stride-1 on A[i][j].
                    double invEk1 = -1.0 / e[k + 1];
                    final int tLen = n - k - 1;
                    for (int j = k + 1; j < n; j++) {
                        tVals[j - k - 1] = e[j] * invEk1;
                    }
                    for (int i = k + 1; i < m; i++) {
                        double wi = work[i];
                        for (int jj = 0; jj < tLen; jj++) {
                            A[i][k + 1 + jj] += tVals[jj] * wi;
                        }
                    }
                }
                
                for (int i = k + 1; i < n; i++) {
                    vData[i][k] = e[i];
                }
            }
        }
        
        // Set up the final bidiagonal matrix or order p.
        int p = n;
        if (nct < n) {
            singularValues[nct] = A[nct][nct];
        }
        if (m < p) {
            singularValues[p - 1] = 0;
        }
        if (nrt + 1 < p) {
            e[nrt] = A[nrt][p - 1];
        }
        e[p - 1] = 0;
        
        // Generate U.
        for (int j = nct; j < n; j++) {
            for (int i = 0; i < m; i++) {
                uData[i][j] = 0.0;
            }
            uData[j][j] = 1.0;
        }

        for (int k = nct - 1; k >= 0; k--) {
            if (singularValues[k] != 0) {
                // Extract column k to stride-1 buffer (work[], already allocated).
                // Avoids two stride-n accesses in inner dot-product and update loops.
                final int uColLen = m - k;
                for (int i = 0; i < uColLen; i++) {
                    work[i] = uData[k + i][k];
                }
                for (int j = k + 1; j < n; j++) {
                    double t = 0;
                    for (int i = 0; i < uColLen; i++) {
                        t += work[i] * uData[k + i][j];
                    }
                    if (Math.abs(uData[k][k]) > epsilon) {
                        t = -t / uData[k][k];
                        for (int i = 0; i < uColLen; i++) {
                            uData[k + i][j] += t * work[i];
                        }
                    }
                }
                for (int i = k; i < m; i++) {
                    uData[i][k] = -uData[i][k];
                }
                uData[k][k] = 1 + uData[k][k];
                for (int i = 0; i < k - 1; i++) {
                    uData[i][k] = 0.0;
                }
            } else {
                for (int i = 0; i < m; i++) {
                    uData[i][k] = 0.0;
                }
                uData[k][k] = 1.0;
            }
        }

        // Generate V.
        for (int k = n - 1; k >= 0; k--) {
            if (k < nrt && e[k] != 0) {
                // Extract column k to stride-1 buffer (reuses work[]).
                final int vColLen = n - k - 1;
                for (int i = 0; i < vColLen; i++) {
                    work[i] = vData[k + 1 + i][k];
                }
                for (int j = k + 1; j < n; j++) {
                    double t = 0;
                    for (int i = 0; i < vColLen; i++) {
                        t += work[i] * vData[k + 1 + i][j];
                    }
                    if (Math.abs(vData[k + 1][k]) > epsilon) {
                        t = -t / vData[k + 1][k];
                        for (int i = 0; i < vColLen; i++) {
                            vData[k + 1 + i][j] += t * work[i];
                        }
                    }
                }
            }
            for (int i = 0; i < n; i++) {
                vData[i][k] = 0.0;
            }
            vData[k][k] = 1.0;
        }

        // GUARD: U^T + V^T transpose optimization. Both uData and vData are
        // transposed so that Givens rotations on U/V columns become row ops on
        // U^T/V^T (stride-1, 8× cache-line utilization). bdsqrMainLoopVt assumes
        // both layouts. For non-square U (m > n), allocate a temporary n×m buffer.
        // See: decomposition-pitfalls.md § SVDVtTranspose
        IMatrix<Double> U;
        final double[][] uTrans;
        if (m == n) {
            transposeSquareInPlace(uData, n);
            U = Linalg.matrix(uData);
            uTrans = uData;
        } else {
            uTrans = new double[n][m];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    uTrans[j][i] = uData[i][j];
                }
            }
            U = Linalg.matrix(uTrans);
        }
        transposeSquareInPlace(vData, n);
        IMatrix<Double> V = Linalg.matrix(vData);

        bdsqrMainLoopVt(singularValues, e, U, V, m, n, p);

        // Restore V and U to normal row-major layout.
        transposeSquareInPlace(vData, n);
        if (m == n) {
            transposeSquareInPlace(uData, n);
        } else {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    uData[i][j] = uTrans[j][i];
                }
            }
        }
        U = Linalg.matrix(uData);

        final int kMin = n;
        this.cachedSingularValues = java.util.Arrays.copyOf(singularValues, kMin);
        IVector<Double> singularValuesVector = Linalg.vector(this.cachedSingularValues);

        // GUARD: 宽矩阵 (m<n) 时先转置再分解. 对 A^T = U_t * S * V_t^T.
        // 取转置得 A = V_t * S * U_t^T = U_A * S * V_A^T, 所以:
        //   cachedU = V_t  (m×k, 从 A^T 的右因子取出)
        //   cachedV = U_t  (n×k, 从 A^T 的左因子取出, 经 widen 补全为 n×n)
        // 这个交换是数学恒等式的结果, 不是 bug. WideMatrixPathTest 中的
        // "This is INCORRECT" 注释是错误的.
        // See: decomposition-pitfalls.md § WideMatrixSVD
        if (!transposed) {
            cachedU = U;
            cachedV = V;
        } else {
            cachedU = V;
            cachedV = widenVToFullOrthogonal(U, n0, this.epsilon);
        }
        cachedS = singularValuesVector;
        cachedVT = cachedV.transposeNew();
    }
    
    
    
    /**
     * 优化的SVD算法（双对角化+分治法）
     * 改进版本，使用更好的数值稳定性和错误检查
     */
    private void optimizedSVD(IDoubleMatrix matrix) {
        int m = matrix.getRowNum();
        int n = matrix.getColNum();
        int minDim = Math.min(m, n);
        // 宽矩阵（m < n）先转置再分解，与 bidiagonalSVD 一致：避免双对角提取时丢失最后一条超对角元
        final boolean transposed = m < n;
        final IDoubleMatrix workMat = transposed
                ? (IDoubleMatrix) (IMatrix<Double>) matrix.transpose()
                : matrix;
        final int mWork = workMat.getRowNum();
        final int nWork = workMat.getColNum();

        try {
            // 步骤1：双对角化预处理
            Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> bidiagResult =
                bidiagonalizationWithIMatrix(workMat);
            IMatrix<Double> U1 = bidiagResult.getFirst();   // (mWork×minDim)
            IMatrix<Double> B = bidiagResult.getSecond();   // (minDim×minDim)
            IMatrix<Double> V1 = bidiagResult.getThird();   // (nWork×nWork)

            // 验证双对角化结果
            if (U1 == null || B == null || V1 == null) {
                throw new RuntimeException("Bidiagonalization failed: null result");
            }

            // 步骤2：对双对角矩阵应用QR算法
            Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> qrResult =
                qrAlgorithmForBidiagonalWithIMatrix(B, mWork, nWork);
            IVector<Double> singularValuesVector = qrResult.getFirst();
            IMatrix<Double> Q_left = qrResult.getSecond();   // (minDim×minDim)
            IMatrix<Double> Q_right = qrResult.getThird();   // (minDim×minDim)

            // 验证QR算法结果
            if (singularValuesVector == null || Q_left == null || Q_right == null) {
                throw new RuntimeException("QR algorithm failed: null result");
            }

            // 步骤3：计算工作矩阵的 U 和 V
            // U_work = U1 * Q_left (mWork×minDim * minDim×minDim = mWork×minDim)
            IMatrix<Double> U_work = U1.mmul(Q_left);

            IMatrix<Double> V1_reduced = extractFirstColumns(V1, minDim);
            IMatrix<Double> V_work = V1_reduced.mmul(Q_right);

            if (!transposed) {
                // 原问题: A = U_work * S * V_work^T
                IMatrix<Double> V_temp = assembleFullRightOrthogonal(V1, V_work, minDim);

                if (V_temp.rows() != n || V_temp.cols() != n) {
                    throw new RuntimeException(String.format(
                        "Invalid V dimensions: expected %dx%d, got %dx%d",
                        n, n, V_temp.rows(), V_temp.cols()));
                }

                cachedU = U_work;
                cachedV = V_temp;
            } else {
                // 转了置的: A^T = U_work * S * V_work^T, 所以 A = V_work * S * U_work^T
                // U_A = V_work (m×minDim), V_A = U_work (n×minDim)
                cachedU = V_work;   // m×minDim
                cachedV = U_work;   // n×minDim, 稍后由 decompose() 调用 widenVToFullOrthogonal 补全
            }

            // 验证 U 维度
            if (cachedU.rows() != m || cachedU.cols() != minDim) {
                throw new RuntimeException(String.format(
                    "Invalid U dimensions: expected %dx%d, got %dx%d",
                    m, minDim, cachedU.rows(), cachedU.cols()));
            }

            // 确保奇异值为正数
            for (int i = 0; i < singularValuesVector.length(); i++) {
                if (singularValuesVector.get(i) < 0) {
                    singularValuesVector.set(i, -singularValuesVector.get(i));
                    for (int j = 0; j < cachedU.rows(); j++) {
                        cachedU.set(j, i, -cachedU.get(j, i));
                    }
                }
            }

            // 轻微的正交化以确保数值稳定性（仅在需要时）
            double orthogonalityError = checkOrthogonality(cachedU);
            if (orthogonalityError > epsilon * 100) {
                orthogonalizeMatrixWithIMatrix(cachedU);
            }

            if (!transposed) {
                orthogonalityError = checkOrthogonality(cachedV);
                if (orthogonalityError > epsilon * 100) {
                    orthogonalizeMatrixWithIMatrix(cachedV);
                }
            }

            // 缓存结果
            cachedS = singularValuesVector;
            cachedVT = cachedV.transposeNew();

            // 缓存奇异值
            this.cachedSingularValues = new double[singularValuesVector.length()];
            for (int i = 0; i < singularValuesVector.length(); i++) {
                this.cachedSingularValues[i] = singularValuesVector.get(i);
            }

        } catch (Exception e) {
            // 不在此回退到 Jama bdsqr（矩形中档规模上曾出现 O(1) 级重构误差）；改试分治路径。
            log.warn("Optimized SVD failed, trying divide-and-conquer SVD: {}", e.getMessage());
            divideAndConquerSVD(matrix);
        }
    }
    
    /**
     * Extract the first n columns from a matrix
     */
    private IMatrix<Double> extractFirstColumns(IMatrix<Double> matrix, int n) {
        int rows = matrix.rows();
        int cols = Math.min(matrix.cols(), n);
        
        double[][] data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = matrix.get(i, j);
            }
        }
        
        return Linalg.matrix(data);
    }
    /**
     * 双对角化，使用IMatrix API
     */
    private Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> bidiagonalizationWithIMatrix(IMatrix<Double> matrix) {
        IBidiagonalDecomposition bidiagonalDecomposition = Decomps.createBidiagonal();
        return bidiagonalDecomposition.decompose(matrix);
    }
    
    /**
     * 双对角矩阵 SVD：由主对角 / 超对角初始化 {@code s}/{@code e}，调用 {@link #bdsqrMainLoop}，返回
     * {@code B = Q_left Σ Q_right^T}。不再对结果调用 {@link #sortSingularValuesWithIMatrix}，以免与
     * {@link #ensureValidSingularValues} 对 VT 行的处理不一致；合并端可对最终 {@code U,V} 再排序。
     */
    private Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> qrAlgorithmForBidiagonalWithIMatrix(
            IMatrix<Double> B, int originalM, int originalN) {
        int m = B.rows();
        int n = B.cols();
        int minDim = Math.min(m, n);

        double[] singularValues = new double[minDim];
        double[] e = new double[n];
        java.util.Arrays.fill(e, 0.0);
        for (int i = 0; i < minDim; i++) {
            singularValues[i] = B.get(i, i);
        }
        for (int i = 0; i < minDim - 1; i++) {
            e[i] = B.get(i, i + 1);
        }

        IMatrix<Double> qLeft;
        if (m > minDim) {
            double[][] uData = new double[m][minDim];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < minDim; j++) {
                    uData[i][j] = (i == j) ? 1.0 : 0.0;
                }
            }
            qLeft = Linalg.matrix(uData);
        } else {
            qLeft = Linalg.eye(m);
        }
        IMatrix<Double> qRight = Linalg.eye(n);

        bdsqrMainLoop(singularValues, e, qLeft, qRight, m, n, minDim);

        IVector<Double> alpha = Linalg.zeros(minDim);
        for (int i = 0; i < minDim; i++) {
            alpha.set(i, singularValues[i]);
        }
        return new Tuple3<>(alpha, extractFirstColumns(qLeft, minDim), extractFirstColumns(qRight, minDim));
    }

    /**
     * 对奇异值进行排序，同时更新左右旋转矩阵
     */
    private void sortSingularValuesWithIMatrix(
            IVector<Double> singularValues,
            IMatrix<Double> Q_left,
            IMatrix<Double> Q_right) {
        int n = singularValues.length();
        
        // 使用快速排序（降序）
        quickSortWithIMatrix(singularValues, Q_left, Q_right, 0, n - 1);
        
        // 确保奇异值为正，必要时调整旋转矩阵的符号
        for (int i = 0; i < n; i++) {
            if (singularValues.get(i) < 0) {
                singularValues.set(i, -singularValues.get(i));
                // 翻转Q_left的对应列的符号（也可以选择翻转Q_right，效果相同）
                for (int k = 0; k < Q_left.rows(); k++) {
                    Q_left.set(k, i, -Q_left.get(k, i));
                }
            }
        }
    }
    
    /**
     * 快速排序实现
     */
    private void quickSortWithIMatrix(IVector<Double> singularValues,
                                     IMatrix<Double> Q_left, IMatrix<Double> Q_right,
                                     int low, int high) {
        if (low < high) {
            int pi = partitionWithIMatrix(singularValues, Q_left, Q_right, low, high);
            quickSortWithIMatrix(singularValues, Q_left, Q_right, low, pi - 1);
            quickSortWithIMatrix(singularValues, Q_left, Q_right, pi + 1, high);
        }
    }
    
    /**
     * 分区操作
     */
    private int partitionWithIMatrix(IVector<Double> singularValues,
                                    IMatrix<Double> Q_left, IMatrix<Double> Q_right,
                                    int low, int high) {
        double pivot = Math.abs(singularValues.get(high));
        int i = (low - 1);
        
        for (int j = low; j < high; j++) {
            if (Math.abs(singularValues.get(j)) >= pivot) {
                i++;
                
                // 交换奇异值
                double temp = singularValues.get(i);
                singularValues.set(i, singularValues.get(j));
                singularValues.set(j, temp);
                
                // 交换Q_left的对应列
                for (int k = 0; k < Q_left.rows(); k++) {
                    double tempL = Q_left.get(k, i);
                    Q_left.set(k, i, Q_left.get(k, j));
                    Q_left.set(k, j, tempL);
                }
                
                // 交换Q_right的对应列
                for (int k = 0; k < Q_right.rows(); k++) {
                    double tempR = Q_right.get(k, i);
                    Q_right.set(k, i, Q_right.get(k, j));
                    Q_right.set(k, j, tempR);
                }
            }
        }
        
        // 交换奇异值
        double temp = singularValues.get(i + 1);
        singularValues.set(i + 1, singularValues.get(high));
        singularValues.set(high, temp);
        
        // 交换Q_left的对应列
        for (int k = 0; k < Q_left.rows(); k++) {
            double tempL = Q_left.get(k, i + 1);
            Q_left.set(k, i + 1, Q_left.get(k, high));
            Q_left.set(k, high, tempL);
        }
        
        // 交换Q_right的对应列
        for (int k = 0; k < Q_right.rows(); k++) {
            double tempR = Q_right.get(k, i + 1);
            Q_right.set(k, i + 1, Q_right.get(k, high));
            Q_right.set(k, high, tempR);
        }
        
        return i + 1;
    }
    
    /**
     * 检查矩阵的正交性
     * @param matrix 要检查的矩阵
     * @return 正交性错误（最大错误）
     */
    private double checkOrthogonality(IMatrix<Double> matrix) {
        IMatrix<Double> product = matrix.transposeNew().mmul(matrix);
        int n = product.rows();
        double maxError = 0.0;
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                double actual = product.get(i, j);
                double error = Math.abs(actual - expected);
                maxError = Math.max(maxError, error);
            }
        }
        
        return maxError;
    }
    
    /**
     * 正交化矩阵，使用修正的Gram-Schmidt方法
     * 直接操作矩阵数据以提高性能，避免重复获取列向量
     */
    private void orthogonalizeMatrixWithIMatrix(IMatrix<Double> matrix) {
        int m = matrix.rows();
        int n = matrix.cols();
        
        // 使用修正的Gram-Schmidt正交化过程以提高数值稳定性
        for (int j = 0; j < n; j++) {
            // 修正的Gram-Schmidt过程：对每个先前的向量重新正交化
            for (int k = 0; k < j; k++) {
                // 计算投影系数
                double dot = 0.0;
                for (int i = 0; i < m; i++) {
                    dot += matrix.get(i, j) * matrix.get(i, k);
                }
                
                // 减去投影
                for (int i = 0; i < m; i++) {
                    matrix.set(i, j, matrix.get(i, j) - dot * matrix.get(i, k));
                }
            }
            
            // 归一化当前列
            double norm = 0.0;
            for (int i = 0; i < m; i++) {
                norm += matrix.get(i, j) * matrix.get(i, j);
            }
            norm = Math.sqrt(norm);
            
            if (norm > epsilon) {
                for (int i = 0; i < m; i++) {
                    matrix.set(i, j, matrix.get(i, j) / norm);
                }
            } else {
                // 如果列向量为零，设置为单位向量
                for (int i = 0; i < m; i++) {
                    matrix.set(i, j, (i == j && i < n) ? 1.0 : 0.0);
                }
            }
        }
    }
    
    /**
     * 设 {@code V1} 为双对角分解的右正交因子（n×n），{@code Q_right} 为 k×k。若 {@code W = blkdiag(Q_right, I_{n-k})}，
     * 则 {@code V = V1 * W = [ V1[:,0:k]*Q_right | V1[:,k:n] ]}，仍为 n×n 正交阵；此方法按该形式组装，避免显式 n×n 稠密乘与 Gram–Schmidt 补空间。
     */
    private static IMatrix<Double> assembleFullRightOrthogonal(IMatrix<Double> V1,
            IMatrix<Double> v1FrontTimesQ, int k) {
        int n = V1.rows();
        if (V1.cols() != n || v1FrontTimesQ.rows() != n || v1FrontTimesQ.cols() != k) {
            throw new IllegalArgumentException("assembleFullRightOrthogonal: dimension mismatch");
        }
        if (k >= n) {
            return v1FrontTimesQ;
        }
        IMatrix<Double> V = Linalg.zeros(n, n);
        for (int j = 0; j < k; j++) {
            for (int i = 0; i < n; i++) {
                V.set(i, j, v1FrontTimesQ.get(i, j));
            }
        }
        for (int j = k; j < n; j++) {
            for (int i = 0; i < n; i++) {
                V.set(i, j, V1.get(i, j));
            }
        }
        return V;
    }

    /**
     * 将 n×k 列正交矩阵（k&lt;n）扩展为 n×n 正交矩阵，以满足 {@link ISVDDecomposition} 中 {@code V^T} 为 n×n 的约定。
     * <p>前 k 列保持原样不动；后 n-k 列通过随机初始化 + Modified Gram–Schmidt 正交化补全。</p>
     */
    private static IMatrix<Double> widenVToFullOrthogonal(IMatrix<Double> Vthin, int nOut, double epsilon) {
        int n = Vthin.rows();
        int k = Vthin.cols();
        if (k >= nOut) {
            return Vthin;
        }

        double[][] result = new double[n][nOut];
        // 前 k 列原样复制（右奇异向量，不可修改）
        for (int j = 0; j < k; j++) {
            for (int i = 0; i < n; i++) {
                result[i][j] = Vthin.get(i, j);
            }
        }
        // 剩余列用标准高斯随机初始化
        java.util.Random rng = new java.util.Random(42);
        for (int j = k; j < nOut; j++) {
            for (int i = 0; i < n; i++) {
                result[i][j] = rng.nextGaussian();
            }
        }

        // Modified Gram–Schmidt：仅处理第 k..nOut-1 列，前 k 列保持不动
        for (int j = k; j < nOut; j++) {
            // 对前面所有已正交化的列做正交化
            for (int p = 0; p < j; p++) {
                double dot = 0.0;
                for (int i = 0; i < n; i++) {
                    dot += result[i][j] * result[i][p];
                }
                for (int i = 0; i < n; i++) {
                    result[i][j] -= dot * result[i][p];
                }
            }
            // 归一化
            double norm = 0.0;
            for (int i = 0; i < n; i++) {
                norm += result[i][j] * result[i][j];
            }
            norm = Math.sqrt(norm);
            if (norm > epsilon) {
                for (int i = 0; i < n; i++) {
                    result[i][j] /= norm;
                }
            } else {
                // 退化情况：用第 j 个单位向量替代
                for (int i = 0; i < n; i++) {
                    result[i][j] = (i == j) ? 1.0 : 0.0;
                }
                // 重新正交化
                for (int p = 0; p < j; p++) {
                    double dot = 0.0;
                    for (int i = 0; i < n; i++) {
                        dot += result[i][j] * result[i][p];
                    }
                    for (int i = 0; i < n; i++) {
                        result[i][j] -= dot * result[i][p];
                    }
                }
                norm = 0.0;
                for (int i = 0; i < n; i++) {
                    norm += result[i][j] * result[i][j];
                }
                norm = Math.sqrt(norm);
                if (norm > epsilon) {
                    for (int i = 0; i < n; i++) {
                        result[i][j] /= norm;
                    }
                }
            }
        }

        return Linalg.matrix(result);
    }

    /**
     * 确保奇异值为正数并正确排序
     */
    private void ensureValidSingularValues() {
        if (cachedS == null || cachedU == null || cachedVT == null) {
            return;
        }
        
        int n = cachedS.length();
        
        // 首先确保所有奇异值为正数
        for (int i = 0; i < n; i++) {
            if (cachedS.get(i) < 0) {
                // 如果奇异值为负，反转对应的U列
                cachedS.set(i, -cachedS.get(i));
                for (int j = 0; j < cachedU.rows(); j++) {
                    cachedU.set(j, i, -cachedU.get(j, i));
                }
            }
        }
        
        // 按奇异值降序重排 U 的列、V^T 的行（O(n log n) 索引排序 + 一遍拷贝，替代冒泡 O(n²)）
        double[] svals = new double[n];
        for (int i = 0; i < n; i++) {
            svals[i] = cachedS.get(i);
        }
        Integer[] ord = new Integer[n];
        for (int i = 0; i < n; i++) {
            ord[i] = i;
        }
        Arrays.sort(ord, (a, b) -> Double.compare(svals[b], svals[a]));

        boolean identity = true;
        for (int i = 0; i < n; i++) {
            if (ord[i] != i) {
                identity = false;
                break;
            }
        }
        if (!identity) {
            int rowsU = cachedU.rows();
            int rowsVT = cachedVT.rows();
            int colsVT = cachedVT.cols();
            double[][] uRaw = ((IDoubleMatrix) cachedU).getData();
            double[][] vtRaw = ((IDoubleMatrix) cachedVT).getData();
            double[][] newU = new double[rowsU][n];
            double[][] newVt = new double[rowsVT][colsVT];
            double[] newS = new double[n];
            for (int i = 0; i < n; i++) {
                int src = ord[i];
                newS[i] = svals[src];
                for (int r = 0; r < rowsU; r++) {
                    newU[r][i] = uRaw[r][src];
                }
                System.arraycopy(vtRaw[src], 0, newVt[i], 0, colsVT);
            }
            // Preserve zero-space basis rows from wide-matrix padding (indices n..rowsVT-1)
            for (int i = n; i < rowsVT; i++) {
                System.arraycopy(vtRaw[i], 0, newVt[i], 0, colsVT);
            }
            this.cachedS = IVector.of(newS);
            this.cachedU = Linalg.matrix(newU);
            this.cachedVT = Linalg.matrix(newVt);
            if (this.cachedV != null) {
                this.cachedV = this.cachedVT.transposeNew();
            }
        }

        if (this.cachedSingularValues != null) {
            for (int i = 0; i < n; i++) {
                this.cachedSingularValues[i] = cachedS.get(i);
            }
        }
    }

    /** 同包测试用：分治双对角 SVD。 */
    Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> divideAndConquerBidiagonalSVDForTesting(IMatrix<Double> b) {
        return divideAndConquerBidiagonalSVD(b);
    }

    /** 同包测试用：双对角 QR-SVD 基线。 */
    Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> qrBidiagonalForTesting(IMatrix<Double> b) {
        return qrAlgorithmForBidiagonalWithIMatrix(b, b.rows(), b.cols());
    }

}
