package com.yishape.lab.math.linalg.decomposition.impl;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.Decomps;
import com.yishape.lab.math.linalg.decomposition.IBidiagonalDecomposition;
import com.yishape.lab.math.linalg.decomposition.ISVDDecomposition;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.solver.SVDDecompositionSolver;
import com.yishape.lab.math.util.RerePrecision;
import com.yishape.lab.math.linalg.decomposition.impl.support.DiagonalPlusRankOneSymmetricEigenSolver;
import com.yishape.lab.util.Tuple3;

/**
 * Implementation of singular value decomposition with enhanced numerical stability.
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
public class RereSVDDecomposition implements ISVDDecomposition {

    private static final Logger log = LoggerFactory.getLogger(RereSVDDecomposition.class);

    
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
    /** Epsilon for numerical comparisons. */
    private double epsilon;
    /** Maximum number of iterations. */
    private int maxIterations;
    /** The matrix being decomposed. */
    private IMatrix<Double> matrix;
    
    
    /** Default epsilon for numerical comparisons. */
    private static final double DEFAULT_EPSILON = RerePrecision.getDefaultEpsilon();
    /** Default maximum number of iterations. */
    private static final int DEFAULT_MAX_ITERATIONS = 1000;
    /** Absolute threshold for small singular values. */
    private static final double TINY = RerePrecision.getSafeMin();
    
    /**
     * Default constructor with default parameters.
     */
    public RereSVDDecomposition() {
        this.epsilon = DEFAULT_EPSILON;
        this.maxIterations = DEFAULT_MAX_ITERATIONS;
    }
    
    /**
     * Constructor with unified parameters.
     * 
     * @param epsilon threshold for considering an element as zero
     * @param maxIterations maximum number of iterations
     */
    public RereSVDDecomposition(double epsilon, int maxIterations) {
        this.epsilon = epsilon;
        this.maxIterations = maxIterations;
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
        this.epsilon = epsilon;
        this.maxIterations = maxIterations;
        
        IDoubleMatrix doubleMatrix = (IDoubleMatrix) matrix;
        double[][] data = doubleMatrix.getData();
        
        int m = data.length;    // Matrix rows
        int n = (m > 0) ? data[0].length : 0; // Matrix columns
        if (m == 0 || n == 0) {
            throw new IllegalArgumentException("Matrix cannot be empty");
        }
        
        // Store matrix for later use
        this.matrix = matrix;
        
        // 根据矩阵大小选择不同的SVD算法
        int size = m * n;
        if (size > 10000) {
            // 对于大矩阵，使用分治SVD算法以获得更好的性能
            divideAndConquerSVD(doubleMatrix);
        } else if (size > 1000) {
            // 对于中等矩阵，使用数值稳定的双对角化方法
            bidiagonalSVD(doubleMatrix);
        } else {
            // 对于小矩阵，使用传统SVD方法
            traditionalSVD(doubleMatrix);
        }
        
        // 确保奇异值为正数并正确排序
        ensureValidSingularValues();
        
        return new Tuple3<>(cachedU, cachedS, cachedVT);
    }
    
    /**
     * 传统SVD算法（适用于小矩阵）
     * 基于Householder变换的直接方法
     */
    private void traditionalSVD(IDoubleMatrix matrix) {
        // For small matrices, use the proven bidiagonal method
        // as it's already optimized and numerically stable
        bidiagonalSVD(matrix);
    }
    
    /**
     * 分治SVD算法（适用于大矩阵）
     * 使用分治策略提高大矩阵的计算效率
     */
    private void divideAndConquerSVD(IDoubleMatrix matrix) {
        int m = matrix.getRowNum();
        int n = matrix.getColNum();
        int minDim = Math.min(m, n);
        
        try {
            // 步骤1：双对角化预处理（与优化方法相同）
            Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> bidiagResult = 
                bidiagonalizationWithIMatrix(matrix);
            IMatrix<Double> U1 = bidiagResult.getFirst();   // (m×minDim)
            IMatrix<Double> B = bidiagResult.getSecond();   // (minDim×minDim)
            IMatrix<Double> V1 = bidiagResult.getThird();   // (n×n)
            
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
            
            // 步骤3：计算最终的U和V矩阵
            IMatrix<Double> U_temp = U1.mmul(Q_left);
            IMatrix<Double> V1_reduced = extractFirstColumns(V1, minDim);
            IMatrix<Double> V_temp = V1_reduced.mmul(Q_right);
            
            // 验证结果矩阵的维度
            if (U_temp.rows() != m || U_temp.cols() != minDim) {
                throw new RuntimeException(String.format(
                    "Invalid U dimensions: expected %dx%d, got %dx%d", 
                    m, minDim, U_temp.rows(), U_temp.cols()));
            }
            
            if (V_temp.rows() != n || V_temp.cols() != minDim) {
                throw new RuntimeException(String.format(
                    "Invalid V dimensions: expected %dx%d, got %dx%d", 
                    n, minDim, V_temp.rows(), V_temp.cols()));
            }
            
            // 确保奇异值为正数
            for (int i = 0; i < singularValuesVector.length(); i++) {
                if (singularValuesVector.get(i) < 0) {
                    singularValuesVector.set(i, -singularValuesVector.get(i));
                    // 反转对应的U列
                    for (int j = 0; j < U_temp.rows(); j++) {
                        U_temp.set(j, i, -U_temp.get(j, i));
                    }
                }
            }
            
            // 检查正交性并在必要时重新正交化
            double orthogonalityError = checkOrthogonality(U_temp);
            if (orthogonalityError > 1e-12) {
                orthogonalizeMatrixWithIMatrix(U_temp);
            }
            
            orthogonalityError = checkOrthogonality(V_temp);
            if (orthogonalityError > 1e-12) {
                orthogonalizeMatrixWithIMatrix(V_temp);
            }
            
            // 缓存结果
            cachedU = U_temp;
            cachedV = V_temp;
            cachedS = singularValuesVector;
            cachedVT = V_temp.transposeNew();
            
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

            Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> result1 =
                divideAndConquerBidiagonalSVD(B1);
            Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> result2 =
                divideAndConquerBidiagonalSVD(B2);

            return mergeSubproblems(result1, result2, connectingElement, B, mid);
            
        } catch (Exception e) {
            // 如果分治失败，回退到QR算法
            log.warn("Divide-and-conquer failed for bidiagonal matrix, using QR: " + e.getMessage());
            return qrAlgorithmForBidiagonalWithIMatrix(B, n, n);
        }
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
     * 对 {@code K} 双对角化 + QR-SVD，最后左乘/右乘块对角嵌入阵。
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

        IMatrix<Double> uBlk = Linalg.zeros(nParent, nParent);
        IMatrix<Double> vBlk = Linalg.zeros(nParent, nParent);
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n1; j++) {
                uBlk.set(i, j, u1.get(i, j));
                vBlk.set(i, j, v1.get(i, j));
            }
        }
        for (int i = 0; i < n2; i++) {
            for (int j = 0; j < n2; j++) {
                uBlk.set(n1 + i, n1 + j, u2.get(i, j));
                vBlk.set(n1 + i, n1 + j, v2.get(i, j));
            }
        }

        IMatrix<Double> kMat = uBlk.transposeNew().mmul(parentB).mmul(vBlk);

        IMatrix<Double> recon = uBlk.mmul(kMat).mmul(vBlk.transposeNew());
        double reconErr = 0.0;
        for (int i = 0; i < nParent; i++) {
            for (int j = 0; j < nParent; j++) {
                reconErr = Math.max(reconErr, Math.abs(recon.get(i, j) - parentB.get(i, j)));
            }
        }
        double scale = 0.0;
        for (int i = 0; i < nParent; i++) {
            scale = Math.max(scale, Math.abs(parentB.get(i, i)));
        }
        if (!(reconErr <= Math.max(1e-10, 1e-8 * scale) * nParent)) {
            throw new IllegalStateException("K merge reconstruction max error " + reconErr);
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

        IMatrix<Double> uOut = uBlk.mmul(uK);
        IMatrix<Double> vOut = vBlk.mmul(vK);
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
        if (!(fullErr <= Math.max(1e-10, 1e-8 * scale) * nParent)) {
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
     * BD-SQR 主迭代（与 {@link #bidiagonalSVD} 中 reduction 后阶段相同）。
     */
    private void bdsqrMainLoop(double[] singularValues, double[] e, IMatrix<Double> U, IMatrix<Double> V,
                               int m, int n, int pInitial) {
        int p = pInitial;
        int iterCount = 0;
        // Jama/LINPACK：pp 在第一次进入主循环前固定为 pInitial-1，收敛分支中 bubble 排序与 V 列翻符号均用此上界；
        // 若随 deflate 每轮改用 p-1，已收敛的 σ 无法交换到全局正确位置，重构误差可达 O(1e-3)。
        final int ppFixed = pInitial - 1;
        final int maxIter = Math.max(30, Math.min(m, n));
        while (p > 0 && iterCount < maxIterations) {
            int k;
            int kase;
            
            // This section inspects for negligible elements in the s and e arrays.
            // Improved deflation check with better numerical stability
            for (k = p - 2; k >= 0; k--) {
                final double threshold = TINY + epsilon * (Math.abs(singularValues[k]) +
                        Math.abs(singularValues[k + 1]));
                
                // Use RerePrecision utility for better numerical comparison
                if (RerePrecision.equalsZero(e[k], threshold) || Double.isNaN(e[k])) {
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
            
            // Check for too many iterations (deflation case)
            if (iterCount >= maxIter) {
                // Force convergence by setting remaining superdiagonal elements to zero
                for (int i = 0; i < p - 1; i++) {
                    e[i] = 0;
                }
                break;
            }
            
            // Perform the task indicated by kase.
            double f;
            switch (kase) {
                case 1 -> {
                    f = e[p - 2];
                    e[p - 2] = 0;
                    for (int j = p - 2; j >= k; j--) {
                        double t = Math.hypot(singularValues[j], f);
                        final double cs = singularValues[j] / t;
                        final double sn = f / t;
                        singularValues[j] = t;
                        if (j != k) {
                            f = -sn * e[j - 1];
                            e[j - 1] = cs * e[j - 1];
                        }
                        
                        for (int i = 0; i < n; i++) {
                            t = cs * V.get(i, j) + sn * V.get(i, p - 1);
                            V.set(i, p - 1, -sn * V.get(i, j) + cs * V.get(i, p - 1));
                            V.set(i, j, t);
                        }
                    }
                }
                case 2 -> {
                    f = e[k - 1];
                    e[k - 1] = 0;
                    for (int j = k; j < p; j++) {
                        double t = Math.hypot(singularValues[j], f);
                        final double cs = singularValues[j] / t;
                        final double sn = f / t;
                        singularValues[j] = t;
                        f = -sn * e[j];
                        e[j] = cs * e[j];
                        
                        for (int i = 0; i < m; i++) {
                            t = cs * U.get(i, j) + sn * U.get(i, k - 1);
                            U.set(i, k - 1, -sn * U.get(i, j) + cs * U.get(i, k - 1));
                            U.set(i, j, t);
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
                        double t = Math.hypot(f, g);
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
                            t = cs * V.get(i, j) + sn * V.get(i, j + 1);
                            V.set(i, j + 1, -sn * V.get(i, j) + cs * V.get(i, j + 1));
                            V.set(i, j, t);
                        }
                        
                        t = Math.hypot(f, g);
                        cs = (Math.abs(t) > epsilon) ? f / t : 1.0;
                        sn = (Math.abs(t) > epsilon) ? g / t : 0.0;
                        singularValues[j] = t;
                        f = cs * e[j] + sn * singularValues[j + 1];
                        singularValues[j + 1] = -sn * e[j] + cs * singularValues[j + 1];
                        g = sn * e[j + 1];
                        e[j + 1] = cs * e[j + 1];
                        
                        if (j < m - 1) {
                            for (int i = 0; i < m; i++) {
                                t = cs * U.get(i, j) + sn * U.get(i, j + 1);
                                U.set(i, j + 1, -sn * U.get(i, j) + cs * U.get(i, j + 1));
                                U.set(i, j, t);
                            }
                        }
                    }
                    e[p - 2] = f;
                }
                default -> {
                    // Make the singular values positive.
                    if (singularValues[k] <= 0) {
                        singularValues[k] = singularValues[k] < 0 ? -singularValues[k] : 0;
                        
                        for (int i = 0; i <= ppFixed; i++) {
                            V.set(i, k, -V.get(i, k));
                        }
                    }
                    
                    // Order the singular values.
                    while (k < ppFixed) {
                        if (singularValues[k] >= singularValues[k + 1]) {
                            break;
                        }
                        double t = singularValues[k];
                        singularValues[k] = singularValues[k + 1];
                        singularValues[k + 1] = t;
                        
                        if (k < n - 1) {
                            for (int i = 0; i < n; i++) {
                                t = V.get(i, k + 1);
                                V.set(i, k + 1, V.get(i, k));
                                V.set(i, k, t);
                            }
                        }
                        
                        if (k < m - 1) {
                            for (int i = 0; i < m; i++) {
                                t = U.get(i, k + 1);
                                U.set(i, k + 1, U.get(i, k));
                                U.set(i, k, t);
                            }
                        }
                        k++;
                    }
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

    private void bidiagonalSVD(IDoubleMatrix matrix) {
        int m = matrix.getRowNum();
        int n = matrix.getColNum();
        
        // 检查空矩阵情况
        if (m == 0 || n == 0) {
            throw new IllegalArgumentException("Matrix cannot be empty");
        }
        
        int minDim = Math.min(m, n);
        
        // For the traditional SVD, we'll implement the full algorithm directly
        // similar to the Commons Math 4 approach but using our IMatrix API
        
        // Create a copy of the matrix data for manipulation
        double[][] A = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = matrix.get(i, j);
            }
        }
        
        // Initialize U and V matrices
        IMatrix<Double> U = Linalg.eye(m);  // Start with m x m identity
        // For non-square matrices, we'll need to extract the first minDim columns later
        if (m > minDim) {
            // Extract first minDim columns for U
            double[][] uData = new double[m][minDim];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < minDim; j++) {
                    uData[i][j] = (i == j) ? 1.0 : 0.0;
                }
            }
            U = Linalg.matrix(uData);
        } else if (m < minDim) {
            // This shouldn't happen as minDim = min(m, n) and m >= minDim
            U = Linalg.eye(m);
        }
        IMatrix<Double> V = Linalg.eye(n);          // n x n
        
        // Initialize singular values array
        double[] singularValues = new double[minDim];
        double[] e = new double[n];  // e needs to be size n for super-diagonal elements
        double[] work = new double[m];
        
        // Reduce A to bidiagonal form, storing the diagonal elements
        // in singularValues and the super-diagonal elements in e.
        int nct = Math.min(m - 1, n);
        int nrt = Math.max(0, Math.min(n - 2, m));
        
        for (int k = 0; k < Math.max(nct, nrt); k++) {
            if (k < nct) {
                // Compute the transformation for the k-th column and
                // place the k-th diagonal in singularValues[k].
                // Compute 2-norm of k-th column without under/overflow.
                singularValues[k] = 0;
                for (int i = k; i < m; i++) {
                    singularValues[k] = Math.hypot(singularValues[k], A[i][k]);
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
                // Place the transformation in U for subsequent back
                // multiplication.
                for (int i = k; i < m; i++) {
                    U.set(i, k, A[i][k]);
                }
            }
            
            if (k < nrt) {
                // Compute the k-th row transformation and place the
                // k-th super-diagonal in e[k].
                // Compute 2-norm without under/overflow.
                e[k] = 0;
                for (int i = k + 1; i < n; i++) {
                    e[k] = Math.hypot(e[k], e[i]);
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
                    // Apply the transformation.
                    for (int i = k + 1; i < m; i++) {
                        work[i] = 0;
                    }
                    for (int j = k + 1; j < n; j++) {
                        for (int i = k + 1; i < m; i++) {
                            work[i] += e[j] * A[i][j];
                        }
                    }
                    for (int j = k + 1; j < n; j++) {
                        double t = -e[j] / e[k + 1];
                        for (int i = k + 1; i < m; i++) {
                            A[i][j] += t * work[i];
                        }
                    }
                }
                
                // Place the transformation in V for subsequent
                // back multiplication.
                for (int i = k + 1; i < n; i++) {
                    V.set(i, k, e[i]);
                }
            }
        }
        
        // Set up the final bidiagonal matrix or order p.
        int p = minDim;
        if (nct < n && nct < minDim) {
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
        for (int j = nct; j < minDim; j++) {
            for (int i = 0; i < m; i++) {
                U.set(i, j, 0.0);
            }
            U.set(j, j, 1.0);
        }
        
        for (int k = nct - 1; k >= 0; k--) {
            if (singularValues[k] != 0) {
                for (int j = k + 1; j < minDim; j++) {
                    double t = 0;
                    for (int i = k; i < m; i++) {
                        t += U.get(i, k) * U.get(i, j);
                    }
                    if (Math.abs(U.get(k, k)) > epsilon) {
                        t = -t / U.get(k, k);
                        for (int i = k; i < m; i++) {
                            U.set(i, j, U.get(i, j) + t * U.get(i, k));
                        }
                    }
                }
                for (int i = k; i < m; i++) {
                    U.set(i, k, -U.get(i, k));
                }
                U.set(k, k, 1 + U.get(k, k));
                for (int i = 0; i < k; i++) {
                    U.set(i, k, 0.0);
                }
            } else {
                for (int i = 0; i < m; i++) {
                    U.set(i, k, 0.0);
                }
                U.set(k, k, 1.0);
            }
        }
        
        // Generate V.
        for (int k = n - 1; k >= 0; k--) {
            if (k < nrt && e[k] != 0) {
                for (int j = k + 1; j < n; j++) {
                    double t = 0;
                    for (int i = k + 1; i < n; i++) {
                        t += V.get(i, k) * V.get(i, j);
                    }
                    if (Math.abs(V.get(k + 1, k)) > epsilon) {
                        t = -t / V.get(k + 1, k);
                        for (int i = k + 1; i < n; i++) {
                            V.set(i, j, V.get(i, j) + t * V.get(i, k));
                        }
                    }
                }
            }
            for (int i = 0; i < n; i++) {
                V.set(i, k, 0.0);
            }
            V.set(k, k, 1.0);
        }
        
        bdsqrMainLoop(singularValues, e, U, V, m, n, p);
        
        // Create IVector for singular values
        IVector<Double> singularValuesVector = Linalg.zeros(minDim);
        for (int i = 0; i < minDim; i++) {
            singularValuesVector.set(i, singularValues[i]);
        }
        
        // Cache results
        cachedU = U;
        cachedV = V;
        cachedS = singularValuesVector;
        cachedVT = V.transposeNew();
        
        // Cache singular values
        this.cachedSingularValues = new double[minDim];
        System.arraycopy(singularValues, 0, this.cachedSingularValues, 0, minDim);
    }
    
    
    
    /**
     * 优化的SVD算法（双对角化+分治法）
     * 改进版本，使用更好的数值稳定性和错误检查
     */
    private void optimizedSVD(IDoubleMatrix matrix) {
        int m = matrix.getRowNum();
        int n = matrix.getColNum();
        int minDim = Math.min(m, n);
        
        try {
            // 步骤1：双对角化预处理
            Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> bidiagResult = 
                bidiagonalizationWithIMatrix(matrix);
            IMatrix<Double> U1 = bidiagResult.getFirst();   // (m×minDim)
            IMatrix<Double> B = bidiagResult.getSecond();   // (minDim×minDim)
            IMatrix<Double> V1 = bidiagResult.getThird();   // (n×n)
            
            // 验证双对角化结果
            if (U1 == null || B == null || V1 == null) {
                throw new RuntimeException("Bidiagonalization failed: null result");
            }
            
            // 步骤2：对双对角矩阵应用QR算法
            Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> qrResult = 
                qrAlgorithmForBidiagonalWithIMatrix(B, m, n);
            IVector<Double> singularValuesVector = qrResult.getFirst();
            IMatrix<Double> Q_left = qrResult.getSecond();   // (minDim×minDim)
            IMatrix<Double> Q_right = qrResult.getThird();   // (minDim×minDim)
            
            // 验证QR算法结果
            if (singularValuesVector == null || Q_left == null || Q_right == null) {
                throw new RuntimeException("QR algorithm failed: null result");
            }
            
            // 步骤3：计算最终的U和V矩阵
            // U = U1 * Q_left (m×minDim * minDim×minDim = m×minDim)
            IMatrix<Double> U_temp = U1.mmul(Q_left);
            
            // V = V1 * Q_right (n×n * minDim×minDim = n×minDim)
            // But we need to extract only the first minDim columns of V1 for multiplication
            IMatrix<Double> V1_reduced = extractFirstColumns(V1, minDim);
            IMatrix<Double> V_temp = V1_reduced.mmul(Q_right);
            
            // 验证结果矩阵的维度
            if (U_temp.rows() != m || U_temp.cols() != minDim) {
                throw new RuntimeException(String.format(
                    "Invalid U dimensions: expected %dx%d, got %dx%d", 
                    m, minDim, U_temp.rows(), U_temp.cols()));
            }
            
            if (V_temp.rows() != n || V_temp.cols() != minDim) {
                throw new RuntimeException(String.format(
                    "Invalid V dimensions: expected %dx%d, got %dx%d", 
                    n, minDim, V_temp.rows(), V_temp.cols()));
            }
            
            // 确保奇异值为正数
            for (int i = 0; i < singularValuesVector.length(); i++) {
                if (singularValuesVector.get(i) < 0) {
                    singularValuesVector.set(i, -singularValuesVector.get(i));
                    // 反转对应的U列
                    for (int j = 0; j < U_temp.rows(); j++) {
                        U_temp.set(j, i, -U_temp.get(j, i));
                    }
                }
            }
            
            // 轻微的正交化以确保数值稳定性（仅在需要时）
            double orthogonalityError = checkOrthogonality(U_temp);
            if (orthogonalityError > 1e-12) {
                orthogonalizeMatrixWithIMatrix(U_temp);
            }
            
            orthogonalityError = checkOrthogonality(V_temp);
            if (orthogonalityError > 1e-12) {
                orthogonalizeMatrixWithIMatrix(V_temp);
            }
            
            // 缓存结果
            cachedU = U_temp;
            cachedV = V_temp;
            cachedS = singularValuesVector;
            cachedVT = V_temp.transposeNew();
            
            // 缓存奇异值
            this.cachedSingularValues = new double[singularValuesVector.length()];
            for (int i = 0; i < singularValuesVector.length(); i++) {
                this.cachedSingularValues[i] = singularValuesVector.get(i);
            }
            
        } catch (Exception e) {
            // 如果优化算法失败，回退到双对角SVD
            log.warn("Optimized SVD failed, falling back to bidiagonal SVD: " + e.getMessage());
            bidiagonalSVD(matrix);
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
            
            if (norm > 1e-12) {
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
        
        // 然后确保奇异值按降序排列
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                if (cachedS.get(i) < cachedS.get(j)) {
                    // 交换奇异值
                    double tempS = cachedS.get(i);
                    cachedS.set(i, cachedS.get(j));
                    cachedS.set(j, tempS);
                    
                    // 交换U矩阵的对应列
                    for (int k = 0; k < cachedU.rows(); k++) {
                        double tempU = cachedU.get(k, i);
                        cachedU.set(k, i, cachedU.get(k, j));
                        cachedU.set(k, j, tempU);
                    }
                    
                    // 交换VT矩阵的对应行
                    for (int k = 0; k < cachedVT.cols(); k++) {
                        double tempVT = cachedVT.get(i, k);
                        cachedVT.set(i, k, cachedVT.get(j, k));
                        cachedVT.set(j, k, tempVT);
                    }
                }
            }
        }
        
        // 更新缓存的奇异值数组
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