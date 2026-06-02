package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.util.YishapeLogger;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.ISVDDecomposition;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.solver.SVDDecompositionSolver;
import com.yishape.lab.math.util.RerePrecision;
import com.yishape.lab.util.Tuple3;

import java.util.Arrays;
import java.util.concurrent.ForkJoinPool;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Future;

/**
 * BLAS-3 SVD via blocked Householder bidiagonalization (LAPACK DGEBRD pattern).
 * <p>
 * 与 {@link RereSVDDecompBlas2}（标量逐列 BLAS-2 双对角化）并列的 BLAS-3 风格实现：
 * 把连续的 nb 个 Householder 反射通过 WY 紧凑表示（{@code I - Y T Y^T}）合并为单次矩阵乘法
 * （GEMM），尾随子矩阵更新和 U/V 构造均以 BLAS-3 完成。
 * </p>
 *
 * <h3>核心机制</h3>
 * <ul>
 *   <li>双对角化按 nb=32 列分块。每个 panel 内使用 LABRD：在线累加辅助矩阵 {@code X} 与 {@code Y}，
 *       使 panel 内每步反射的「等效尾随更新」由 X、Y 携带，panel 结束后用两次 GEMM 一次性
 *       施加到剩余子矩阵 {@code A[k+nb:m, k+nb:n]} 上。</li>
 *   <li>U/V 构造同样按块进行：每块通过 WY 表示 {@code (I - V T V^T)} 后以两次 GEMM 应用于已
 *       累积的 U/V，cache 行利用率提升 8×。</li>
 *   <li>BD-SQR 主循环复用 {@link RereSVDDecompBlas2} 的 U^T+V^T 转置布局思路，使列旋转变
 *       为 stride-1 行旋转。</li>
 * </ul>
 *
 * <p><b>shape 约定与 {@link ISVDDecomposition} / {@link RereSVDDecompBlas2} 严格一致：</b>
 * 设 A 为 m×n，k = min(m,n)。返回 U（m×k 瘦型），singular values（长度 k 降序），V^T（n×n）。</p>
 *
 * @author lteb2
 * @version 1.0
 * @since 2.0
 */
public class RereSVDDecompBlas3 implements ISVDDecomposition {

    private static final YishapeLogger log = YishapeLogger.getLogger(RereSVDDecompBlas3.class);

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
    /** Block size for blocked Householder bidiagonalization. */
    private int blockSize;

    /** Default epsilon. */
    private static final double DEFAULT_EPSILON = RerePrecision.getDefaultEpsilon();
    /** Default BD-SQR iteration cap (kept generous, same as RereSVDDecomposition). */
    private static final int DEFAULT_MAX_ITERATIONS = 6_000_000;
    /** Absolute threshold for small singular values. */
    private static final double TINY = RerePrecision.getSafeMin();
    /** Default LABRD block size. nb=32 trades panel overhead vs. cache footprint typical of dual-cache CPUs. */
    private static final int DEFAULT_BLOCK_SIZE = 32;
    /** Minimum dimension for blocked path to amortize panel setup; below this we fall back to scalar bidiag. */
    private static final int BLOCKED_PATH_MIN_DIM = 64;

    private static double fastHypot(double x, double y) {
        double ax = Math.abs(x);
        double ay = Math.abs(y);
        if (ax > 1e154 || ay > 1e154) {
            return Math.hypot(x, y);
        }
        return Math.sqrt(x * x + y * y);
    }

    public RereSVDDecompBlas3() {
        this(DEFAULT_EPSILON, DEFAULT_MAX_ITERATIONS, DEFAULT_BLOCK_SIZE);
    }

    public RereSVDDecompBlas3(double epsilon, int maxIterations) {
        this(epsilon, maxIterations, DEFAULT_BLOCK_SIZE);
    }

    public RereSVDDecompBlas3(double epsilon, int maxIterations, int blockSize) {
        this.epsilon = epsilon;
        this.maxIterations = maxIterations;
        this.blockSize = Math.max(1, blockSize);
    }

    @Override
    public IDecompositionSolver getSolver() {
        if (cachedU == null || cachedS == null || cachedVT == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        return new SVDDecompositionSolver(cachedU, cachedS, cachedVT, epsilon);
    }

    @Override
    public double getDeterminant() {
        if (determinant == null) {
            if (cachedSingularValues == null) {
                throw new IllegalStateException("Decomposition not yet performed");
            }
            double d = 1.0;
            for (double s : cachedSingularValues) d *= s;
            determinant = d;
        }
        return determinant;
    }

    @Override
    public boolean isNonSingular() {
        if (cachedSingularValues == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        for (double s : cachedSingularValues) {
            if (Math.abs(s) < epsilon) return false;
        }
        return true;
    }

    @Override
    public double getConditionNumber() {
        if (conditionNumber == null) {
            if (cachedSingularValues == null) {
                throw new IllegalStateException("Decomposition not yet performed");
            }
            double max = 0, min = Double.POSITIVE_INFINITY;
            for (double s : cachedSingularValues) {
                double a = Math.abs(s);
                if (a > max) max = a;
                if (a < min) min = a;
            }
            conditionNumber = min > epsilon ? max / min : Double.POSITIVE_INFINITY;
        }
        return conditionNumber;
    }

    @Override
    public int getRank() {
        if (rank == null) {
            if (cachedSingularValues == null) {
                throw new IllegalStateException("Decomposition not yet performed");
            }
            int r = 0;
            for (double s : cachedSingularValues) {
                if (Math.abs(s) > epsilon) r++;
            }
            rank = r;
        }
        return rank;
    }

    @Override public double getEpsilon() { return epsilon; }
    @Override public int getMaxIterations() { return maxIterations; }
    public int getBlockSize() { return blockSize; }

    @Override
    public Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix) {
        return decompose(matrix, this.epsilon, this.maxIterations);
    }

    @Override
    public Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix, double epsilon) {
        return decompose(matrix, epsilon, DEFAULT_MAX_ITERATIONS);
    }

    @Override
    public Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix, double epsilon, int maxIterations) {
        cachedU = null;
        cachedV = null;
        cachedS = null;
        cachedVT = null;
        cachedSingularValues = null;
        determinant = null;
        conditionNumber = null;
        rank = null;
        this.epsilon = epsilon;
        this.maxIterations = maxIterations;

        IDoubleMatrix doubleMatrix = (IDoubleMatrix) matrix;
        double[][] data = doubleMatrix.getData();
        int m0 = data.length;
        int n0 = (m0 > 0) ? data[0].length : 0;
        if (m0 == 0 || n0 == 0) {
            throw new IllegalArgumentException("Matrix cannot be empty");
        }

        // 与 BLAS-2 路径一致：m<n 时先转置（确保上双对角形式），最后再交换 cachedU/cachedV
        final boolean transposed = m0 < n0;
        final int m;
        final int n;
        final double[] src;  // flat row-major
        if (transposed) {
            m = n0;
            n = m0;
            src = new double[m * n];
            for (int i = 0; i < m0; i++) {
                for (int j = 0; j < n0; j++) {
                    src[j * n + i] = data[i][j];
                }
            }
        } else {
            m = m0;
            n = n0;
            src = new double[m * n];
            for (int i = 0; i < m; i++) {
                System.arraycopy(data[i], 0, src, i * n, n);
            }
        }

        // 主算法：分块双对角化 → BD-SQR / DC-SVD
        blockedBidiagonalSVD(src, m, n, n, transposed, n0);

        // ensureValidSingularValues：保证非负 + 降序
        ensureValidSingularValues();

        return new Tuple3<>(cachedU, cachedS, cachedVT);
    }

    // ============================================================
    //  Blocked bidiagonalization (LABRD/DGEBRD-style, BLAS-3)
    // ============================================================

    /**
     * 分块 LABRD + 尾随 BLAS-3 更新 + 块状 U/V 构造 + DC-SVD / 转置布局 BD-SQR。
     *
     * @param A           m×n 工作矩阵 flat row-major（in-place 修改，存储 Householder 向量）
     * @param m           行数（m >= n 由调用方保证）
     * @param n           列数
     * @param lda         A 的 leading dimension (= n)
     * @param transposed  原始 A 是否被转置（m0 < n0）
     * @param n0Orig      原始 n（用于宽矩阵 widen V）
     */
    private void blockedBidiagonalSVD(double[] A, int m, int n, int lda,
                                        boolean transposed, int n0Orig) {
        final int kMin = Math.min(m, n);  // = n since m >= n
        final double[] d = new double[n];        // 主对角 (singularValues 缓冲)
        final double[] e = new double[n];        // 超对角
        final double[] tauQ = new double[kMin];  // 左反射 tau
        final double[] tauP = new double[kMin];  // 右反射 tau

        final int nct = Math.min(m - 1, n);      // 左反射个数
        final int nrt = Math.max(0, n - 2);      // 右反射个数

        // X[m×nb], Y[n×nb] 用于尾随更新累积（flat row-major）
        final int nb = Math.min(blockSize, kMin);
        final int ldx = nb;
        final int ldy = nb;
        double[] X = new double[m * nb];
        double[] Y = new double[n * nb];

        // 预分配共享缓冲区，消除 applyLeft/RightHouseholdersBlocked 的 per-block 分配
        final int ldvbuf = nb;
        final int ldrbuf = nb;
        final int ldt = nb;
        final int ldw = Math.max(n, m);
        double[] vBuf = new double[m * nb];
        double[] uRBuf = new double[n * nb];
        double[] tBuf = new double[nb * nb];
        double[] wBuf = new double[nb * ldw];
        double[] twBuf = new double[nb * ldw];
        double[] zBuf = new double[nb];

        // 选择路径：小尺寸使用标量回退（块开销不划算）
        if (kMin < BLOCKED_PATH_MIN_DIM) {
            scalarBidiagonalize(A, m, n, lda, d, e, tauQ, tauP);
        } else {
            // 分块循环：每次处理 currNb 个 panel 列/行
            int k = 0;
            while (k < kMin) {
                int currNb = Math.min(nb, kMin - k);
                labrd(A, m, n, lda, k, currNb, d, e, tauQ, tauP, X, ldx, Y, ldy);

                if (k + currNb < m && k + currNb < n) {
                    trailingMatrixUpdate(A, m, n, lda, k, currNb, X, ldx, Y, ldy);
                }
                k += currNb;
            }
        }

        final double[] singularValues = d;

        // === U/V 构造 ===
        final int ldu = n;
        final int ldv = n;
        double[] uData = new double[m * n];
        double[] vData = new double[n * n];

        for (int j = 0; j < n; j++) {
            uData[j * ldu + j] = 1.0;
        }
        for (int j = 0; j < n; j++) {
            vData[j * ldv + j] = 1.0;
        }

        applyLeftHouseholdersBlocked(uData, ldu, A, lda, tauQ, m, n, nct,
                                     vBuf, ldvbuf, tBuf, ldt, wBuf, ldw, twBuf, ldw, zBuf);
        applyRightHouseholdersBlocked(vData, ldv, A, lda, tauP, n, nrt,
                                      uRBuf, ldrbuf, tBuf, ldt, wBuf, ldw, twBuf, ldw, zBuf);

        // === BD-SQR：转置布局（U^T + V^T 存储，stride-1 行旋转） ===
            double[][] uTrans;
            if (m == n) {
                uTrans = new double[n][n];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        uTrans[j][i] = uData[i * ldu + j];
                    }
                }
            } else {
                uTrans = new double[n][m];
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        uTrans[j][i] = uData[i * ldu + j];
                    }
                }
            }
            double[][] vTrans = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    vTrans[j][i] = vData[i * ldv + j];
                }
            }

            bdsqrMainLoopVt(singularValues, e, uTrans, vTrans, m, n, n);

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    vData[i * ldv + j] = vTrans[j][i];
                }
            }
            if (m == n) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        uData[i * ldu + j] = uTrans[j][i];
                    }
                }
            } else {
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        uData[i * ldu + j] = uTrans[j][i];
                    }
                }
            }

        // flat → double[][] for IMatrix 构造
        double[][] uData2D = new double[m][n];
        double[][] vData2D = new double[n][n];
        for (int i = 0; i < m; i++) {
            System.arraycopy(uData, i * ldu, uData2D[i], 0, n);
        }
        for (int i = 0; i < n; i++) {
            System.arraycopy(vData, i * ldv, vData2D[i], 0, n);
        }

        IMatrix<Double> U = Linalg.matrix(uData2D);
        IMatrix<Double> V = Linalg.matrix(vData2D);

        this.cachedSingularValues = Arrays.copyOf(singularValues, n);
        IVector<Double> singularValuesVector = Linalg.vector(this.cachedSingularValues);

        if (!transposed) {
            cachedU = U;
            cachedV = V;
        } else {
            cachedU = V;
            cachedV = widenVToFullOrthogonal(U, n0Orig, this.epsilon);
        }
        cachedS = singularValuesVector;
        cachedVT = cachedV.transposeNew();
    }

    // ============================================================
    //  LABRD: panel reduction with X, Y accumulation (LAPACK DLABRD)
    // ============================================================

    /**
     * 对 A 的 panel [k:m, k:k+nb] 和 [k:k+nb, k:n] 做双对角约化，累积辅助矩阵 X, Y。
     * <p>
     * 设 A_local = A[k:m, k:n]（视为 m'×n' 矩阵，m'=m-k, n'=n-k）。
     * 循环 i = 0..nb-1：
     *   1. 用累积的 X, Y 修正 A_local[i:, i]（两次 GEMV）
     *   2. 生成左 Householder Q_i 消去 A_local[i+1:, i]，存 d[k+i] 和向量 v_i 到 A_local[i:, i]
     *   3. 计算 Y[k+i+1:n, i] 用于后续修正（一系列 GEMV 链）
     *   4. 用 Y, X 修正 A_local[i, i+1:]（两次 GEMV）
     *   5. 生成右 Householder P_i 消去 A_local[i, i+2:]，存 e[k+i] 和向量 u_i
     *   6. 计算 X[k+i+1:m, i] 用于后续修正（GEMV 链）
     * </p>
     * <p>结束后，X[k:m, 0:nb] 和 Y[k:n, 0:nb] 包含累积补偿，可用于 BLAS-3 尾随更新。</p>
     */
    private void labrd(double[] A, int m, int n, int lda, int k, int nb,
                       double[] d, double[] e, double[] tauQ, double[] tauP,
                       double[] X, int ldx, double[] Y, int ldy) {
        for (int j = 0; j < nb; j++) {
            for (int i = k; i < m; i++) X[i * ldx + j] = 0.0;
            for (int i = k; i < n; i++) Y[i * ldy + j] = 0.0;
        }

        final double[] yTmp = new double[nb];
        final double[] xTmp = new double[nb];

        for (int i = 0; i < nb; i++) {
            final int gi = k + i;

            if (i > 0) {
                for (int row = gi; row < m; row++) {
                    int aRowOff = row * lda;
                    double sum = 0.0;
                    for (int l = 0; l < i; l++) {
                        sum += A[aRowOff + (k + l)] * Y[gi * ldy + l];
                    }
                    A[aRowOff + gi] -= sum;
                }
                for (int row = gi; row < m; row++) {
                    int xRowOff = row * ldx;
                    double sum = 0.0;
                    for (int l = 0; l < i; l++) {
                        sum += X[xRowOff + l] * A[(k + l) * lda + gi];
                    }
                    A[row * lda + gi] -= sum;
                }
            }

            double tauQi;
            if (gi + 1 < m) {
                double alpha = A[gi * lda + gi];
                double colNorm = 0.0;
                for (int r = gi; r < m; r++) {
                    colNorm = fastHypot(colNorm, A[r * lda + gi]);
                }
                if (colNorm == 0.0) {
                    tauQi = 0.0;
                    d[gi] = alpha;
                } else {
                    double signAlpha = (alpha >= 0) ? 1.0 : -1.0;
                    double beta = -signAlpha * colNorm;
                    tauQi = (beta - alpha) / beta;
                    double scale = 1.0 / (alpha - beta);
                    A[gi * lda + gi] = beta;
                    for (int r = gi + 1; r < m; r++) {
                        A[r * lda + gi] *= scale;
                    }
                    d[gi] = beta;
                }
            } else {
                tauQi = 0.0;
                d[gi] = A[gi * lda + gi];
            }
            tauQ[gi] = tauQi;

            if (gi < n - 1) {
                if (tauQi != 0.0) {
                    A[gi * lda + gi] = 1.0;
                }

                if (tauQi != 0.0) {
                    for (int col = gi + 1; col < n; col++) {
                        double sum = 0.0;
                        for (int r = gi; r < m; r++) {
                            int aRowOff = r * lda;
                            sum += A[aRowOff + col] * A[aRowOff + gi];
                        }
                        Y[col * ldy + i] = sum;
                    }
                    if (i > 0) {
                        for (int l = 0; l < i; l++) {
                            double sum = 0.0;
                            for (int r = gi; r < m; r++) {
                                int aRowOff = r * lda;
                                sum += A[aRowOff + (k + l)] * A[aRowOff + gi];
                            }
                            yTmp[l] = sum;
                        }
                        for (int col = gi + 1; col < n; col++) {
                            double sum = 0.0;
                            for (int l = 0; l < i; l++) {
                                sum += Y[col * ldy + l] * yTmp[l];
                            }
                            Y[col * ldy + i] -= sum;
                        }
                        for (int l = 0; l < i; l++) {
                            double sum = 0.0;
                            for (int r = gi; r < m; r++) {
                                sum += X[r * ldx + l] * A[r * lda + gi];
                            }
                            yTmp[l] = sum;
                        }
                        for (int col = gi + 1; col < n; col++) {
                            double sum = 0.0;
                            for (int l = 0; l < i; l++) {
                                sum += A[(k + l) * lda + col] * yTmp[l];
                            }
                            Y[col * ldy + i] -= sum;
                        }
                    }
                    for (int col = gi + 1; col < n; col++) {
                        Y[col * ldy + i] *= tauQi;
                    }
                }

                for (int col = gi + 1; col < n; col++) {
                    double sum = 0.0;
                    for (int l = 0; l <= i; l++) {
                        sum += A[gi * lda + (k + l)] * Y[col * ldy + l];
                    }
                    A[gi * lda + col] -= sum;
                }
                if (i > 0) {
                    for (int col = gi + 1; col < n; col++) {
                        double sum = 0.0;
                        for (int l = 0; l < i; l++) {
                            sum += X[gi * ldx + l] * A[(k + l) * lda + col];
                        }
                        A[gi * lda + col] -= sum;
                    }
                }

                double tauPi;
                if (gi + 2 < n) {
                    double alphaR = A[gi * lda + (gi + 1)];
                    double rowNorm = 0.0;
                    for (int c = gi + 1; c < n; c++) {
                        rowNorm = fastHypot(rowNorm, A[gi * lda + c]);
                    }
                    if (rowNorm == 0.0) {
                        tauPi = 0.0;
                        e[gi] = alphaR;
                    } else {
                        double signAlphaR = (alphaR >= 0) ? 1.0 : -1.0;
                        double betaR = -signAlphaR * rowNorm;
                        tauPi = (betaR - alphaR) / betaR;
                        double scaleR = 1.0 / (alphaR - betaR);
                        A[gi * lda + (gi + 1)] = betaR;
                        for (int c = gi + 2; c < n; c++) {
                            A[gi * lda + c] *= scaleR;
                        }
                        e[gi] = betaR;
                    }
                } else {
                    tauPi = 0.0;
                    e[gi] = A[gi * lda + (gi + 1)];
                }
                tauP[gi] = tauPi;

                if (gi + 1 < m && tauPi != 0.0) {
                    A[gi * lda + (gi + 1)] = 1.0;

                    for (int row = gi + 1; row < m; row++) {
                        int aRowOff = row * lda;
                        double sum = 0.0;
                        for (int c = gi + 1; c < n; c++) {
                            sum += A[aRowOff + c] * A[gi * lda + c];
                        }
                        X[row * ldx + i] = sum;
                    }
                    for (int l = 0; l <= i; l++) {
                        double sum = 0.0;
                        for (int c = gi + 1; c < n; c++) {
                            sum += Y[c * ldy + l] * A[gi * lda + c];
                        }
                        xTmp[l] = sum;
                    }
                    for (int row = gi + 1; row < m; row++) {
                        int aRowOff = row * lda;
                        double sum = 0.0;
                        for (int l = 0; l <= i; l++) {
                            sum += A[aRowOff + (k + l)] * xTmp[l];
                        }
                        X[row * ldx + i] -= sum;
                    }
                    if (i > 0) {
                        for (int l = 0; l < i; l++) {
                            double sum = 0.0;
                            for (int c = gi + 1; c < n; c++) {
                                sum += A[(k + l) * lda + c] * A[gi * lda + c];
                            }
                            xTmp[l] = sum;
                        }
                        for (int row = gi + 1; row < m; row++) {
                            int xRowOff = row * ldx;
                            double sum = 0.0;
                            for (int l = 0; l < i; l++) {
                                sum += X[xRowOff + l] * xTmp[l];
                            }
                            X[xRowOff + i] -= sum;
                        }
                    }
                    for (int row = gi + 1; row < m; row++) {
                        X[row * ldx + i] *= tauPi;
                    }
                }
            } else {
                tauP[gi] = 0.0;
                if (gi < n) e[gi] = 0.0;
            }
        }
    }

    // ============================================================
    //  Trailing matrix update (BLAS-3 GEMM)
    // ============================================================

    /**
     * panel 结束后对剩余子矩阵的 BLAS-3 修正：
     * <p>
     * A[k+nb:m, k+nb:n] := A[k+nb:m, k+nb:n] - V[k+nb:m, 0:nb] * Y[k+nb:n, 0:nb]^T
     *                                       - X[k+nb:m, 0:nb] * U[0:nb, k+nb:n]
     * </p>
     * 其中：
     * - V 是左 Householder 向量矩阵，存于 A[k+nb:m, k:k+nb]（lower part）。注意 panel 内
     *   对角元已被存为 beta 而非 1，但 v 首元（在 panel 处理时是 sentinel）已被恢复为
     *   beta，因此 V[k+l, l] = 1 是 implicit。这里使用「先用 A[k+l][k+l]=1 替换、用完再
     *   还原」的 sentinel pattern。
     * - U 是右 Householder 向量矩阵，存于 A[k:k+nb, k+nb:n]（upper part）。同样 sentinel
     *   pattern：A[gi][gi+1] 在 panel 内是 betaR，更新时需临时设为 1。
     */
    /** sentinel 临时值的共享缓冲，避免每次 trailing 调用都 new double[nb]。 */
    private static final ThreadLocal<double[]> SENTINEL_DIAG_BUF = ThreadLocal.withInitial(() -> new double[64]);
    private static final ThreadLocal<double[]> SENTINEL_SUP_BUF = ThreadLocal.withInitial(() -> new double[64]);

    /**
     * panel 结束后对剩余子矩阵的 BLAS-3 修正：
     * A[k+nb:m, k+nb:n] -= V * Y^T - X * U
     * 其中 V 与 U 均取自 A 的 panel 区域（通过 sentinel 模式临时设为 1）。
     * 对 GEMM 1/2 均做了循环交换 + tiling + 多线程（大矩阵时），cache 效率显著提升。
     */
    private void trailingMatrixUpdate(double[] A, int m, int n, int lda, int k, int nb,
                                      double[] X, int ldx, double[] Y, int ldy) {
        final int rowStart = k + nb;
        final int colStart = k + nb;
        if (rowStart >= m || colStart >= n) return;

        final double[] savedDiagL = SENTINEL_DIAG_BUF.get();
        final double[] savedSupDiag = SENTINEL_SUP_BUF.get();
        for (int l = 0; l < nb; l++) {
            savedDiagL[l] = A[(k + l) * lda + (k + l)];
            A[(k + l) * lda + (k + l)] = 1.0;
            if (k + l + 1 < n) {
                savedSupDiag[l] = A[(k + l) * lda + (k + l + 1)];
                A[(k + l) * lda + (k + l + 1)] = 1.0;
            } else {
                savedSupDiag[l] = 0.0;
            }
        }

        try {
            final int nRows = m - rowStart;
            final int nCols = n - colStart;
            final long flops = (long) nRows * nCols * nb * 2L;

            if (flops < 8_000_000L) {
                gemm1Trailing(A, m, n, lda, k, nb, rowStart, colStart, Y, ldy);
                gemm2Trailing(A, m, n, lda, k, nb, rowStart, colStart, X, ldx);
            } else {
                ForkJoinPool pool = com.yishape.lab.math.linalg.RereDoubleMatrix.getThreadPool();
                if (pool == null || nRows < 256) {
                    gemm1Trailing(A, m, n, lda, k, nb, rowStart, colStart, Y, ldy);
                    gemm2Trailing(A, m, n, lda, k, nb, rowStart, colStart, X, ldx);
                } else {
                    List<Future<?>> futures = new ArrayList<>();
                    int numThreads = pool.getParallelism();
                    int rowsPerTask = (nRows + numThreads - 1) / numThreads;
                    for (int t = 0; t < numThreads; t++) {
                        final int r0 = rowStart + t * rowsPerTask;
                        final int r1 = Math.min(r0 + rowsPerTask, m);
                        if (r0 >= r1) break;
                        futures.add(pool.submit(() -> {
                            gemm1TrailingRows(A, m, n, lda, k, nb, r0, r1, colStart, Y, ldy);
                            gemm2TrailingRows(A, m, n, lda, k, nb, r0, r1, colStart, X, ldx);
                        }));
                    }
                    for (Future<?> f : futures) {
                        try { f.get(); } catch (Exception ignored) { }
                    }
                }
            }
        } finally {
            for (int l = 0; l < nb; l++) {
                A[(k + l) * lda + (k + l)] = savedDiagL[l];
                if (k + l + 1 < n) {
                    A[(k + l) * lda + (k + l + 1)] = savedSupDiag[l];
                }
            }
        }
    }

    /** GEMM 1: A -= V * Y^T，对指定行范围，tiling + 循环交换优化。 */
    private static void gemm1TrailingRows(double[] A, int m, int n, int lda, int k, int nb,
                                          int rowStart, int rowEnd, int colStart, double[] Y, int ldy) {
        final int TILE = 128;
        for (int row = rowStart; row < rowEnd; row++) {
            final int rowOff = row * lda;
            for (int colBlock = colStart; colBlock < n; colBlock += TILE) {
                int colEnd = Math.min(colBlock + TILE, n);
                for (int l = 0; l < nb; l++) {
                    double v = A[rowOff + (k + l)];
                    for (int col = colBlock; col < colEnd; col++) {
                        A[rowOff + col] -= v * Y[col * ldy + l];
                    }
                }
            }
        }
    }

    /** GEMM 2: A -= X * U，对指定行范围，tiling + 循环交换优化。 */
    private static void gemm2TrailingRows(double[] A, int m, int n, int lda, int k, int nb,
                                          int rowStart, int rowEnd, int colStart, double[] X, int ldx) {
        final int TILE = 128;
        for (int row = rowStart; row < rowEnd; row++) {
            final int rowOff = row * lda;
            for (int colBlock = colStart; colBlock < n; colBlock += TILE) {
                int colEnd = Math.min(colBlock + TILE, n);
                for (int l = 0; l < nb; l++) {
                    double x = X[row * ldx + l];
                    final int panelOff = (k + l) * lda;
                    for (int col = colBlock; col < colEnd; col++) {
                        A[rowOff + col] -= x * A[panelOff + col];
                    }
                }
            }
        }
    }

    /** GEMM 1 全行范围包装。 */
    private static void gemm1Trailing(double[] A, int m, int n, int lda, int k, int nb,
                                      int rowStart, int colStart, double[] Y, int ldy) {
        gemm1TrailingRows(A, m, n, lda, k, nb, rowStart, m, colStart, Y, ldy);
    }

    /** GEMM 2 全行范围包装。 */
    private static void gemm2Trailing(double[] A, int m, int n, int lda, int k, int nb,
                                      int rowStart, int colStart, double[] X, int ldx) {
        gemm2TrailingRows(A, m, n, lda, k, nb, rowStart, m, colStart, X, ldx);
    }

    // ============================================================
    //  Scalar bidiagonalization fallback (used for small dims)
    // ============================================================

    /**
     * 与 RereSVDDecomposition.bidiagonalSVD 中的 BLAS-2 双对角化等价；用于 kMin &lt; 64 的回退。
     * 直接修改 A，输出 d, e, tauQ, tauP，并把 v 向量留在 A 中（与 LABRD 出口形式一致）。
     */
    private void scalarBidiagonalize(double[] A, int m, int n, int lda,
                                     double[] d, double[] e, double[] tauQ, double[] tauP) {
        final int nct = Math.min(m - 1, n);
        final int nrt = Math.max(0, n - 2);
        final int kMax = Math.max(nct, nrt);

        for (int k = 0; k < kMax; k++) {
            if (k < nct) {
                double colNorm = 0.0;
                for (int r = k; r < m; r++) {
                    colNorm = fastHypot(colNorm, A[r * lda + k]);
                }
                if (colNorm != 0.0) {
                    double alpha = A[k * lda + k];
                    double signAlpha = (alpha >= 0) ? 1.0 : -1.0;
                    double beta = -signAlpha * colNorm;
                    double tauQk = (beta - alpha) / beta;
                    double scale = 1.0 / (alpha - beta);
                    A[k * lda + k] = beta;
                    for (int r = k + 1; r < m; r++) {
                        A[r * lda + k] *= scale;
                    }
                    d[k] = beta;
                    tauQ[k] = tauQk;
                    A[k * lda + k] = 1.0;
                    for (int col = k + 1; col < n; col++) {
                        double sum = 0.0;
                        for (int r = k; r < m; r++) {
                            sum += A[r * lda + k] * A[r * lda + col];
                        }
                        double t = tauQk * sum;
                        for (int r = k; r < m; r++) {
                            A[r * lda + col] -= t * A[r * lda + k];
                        }
                    }
                    A[k * lda + k] = beta;
                } else {
                    d[k] = 0.0;
                    tauQ[k] = 0.0;
                }
            }
            if (k < nrt) {
                double rowNorm = 0.0;
                for (int c = k + 1; c < n; c++) {
                    rowNorm = fastHypot(rowNorm, A[k * lda + c]);
                }
                if (rowNorm != 0.0) {
                    double alphaR = A[k * lda + (k + 1)];
                    double signAlphaR = (alphaR >= 0) ? 1.0 : -1.0;
                    double betaR = -signAlphaR * rowNorm;
                    double tauPk = (betaR - alphaR) / betaR;
                    double scaleR = 1.0 / (alphaR - betaR);
                    A[k * lda + (k + 1)] = betaR;
                    for (int c = k + 2; c < n; c++) {
                        A[k * lda + c] *= scaleR;
                    }
                    e[k] = betaR;
                    tauP[k] = tauPk;
                    A[k * lda + (k + 1)] = 1.0;
                    if (k + 1 < m) {
                        for (int row = k + 1; row < m; row++) {
                            double sum = 0.0;
                            for (int c = k + 1; c < n; c++) {
                                sum += A[row * lda + c] * A[k * lda + c];
                            }
                            double t = tauPk * sum;
                            for (int c = k + 1; c < n; c++) {
                                A[row * lda + c] -= t * A[k * lda + c];
                            }
                        }
                    }
                    A[k * lda + (k + 1)] = betaR;
                } else {
                    e[k] = 0.0;
                    tauP[k] = 0.0;
                }
            }
        }
        if (nct == n - 1 && n > 0) {
            d[n - 1] = A[(n - 1) * lda + (n - 1)];
        }
        if (n >= 2 && nrt < n - 1) {
            e[n - 2] = (n - 2 >= 0 && nrt > n - 2) ? e[n - 2] : A[(n - 2) * lda + (n - 1)];
        }
    }

    // ============================================================
    //  U/V generation via blocked Householder application
    // ============================================================

    /**
     * 把 Q_1 Q_2 ... Q_nct 应用到 U（初始为 m×n 单位前 n 列）。
     * <p>
     * Q_i = I - tauQ[i] * v_i v_i^T, v_i 存于 A[i:m, i]（其中 A[i][i]=1 sentinel，其余 A[r][i] 为 v）。
     * </p>
     * <p>分块 BLAS-3: 每 nb 个 Householder 通过 WY 表示 (I - V T V^T) 合并，
     * 然后 U := (I - V T V^T) U 用 2 次 GEMM 实现 (W = V^T U, U -= V T W)。</p>
     * <p>所有临时矩阵均使用调用方预分配的共享缓冲区，消除 per-block GC 压力。</p>
     */
    private void applyLeftHouseholdersBlocked(double[] uData, int ldu, double[] A, int lda,
                                              double[] tauQ, int m, int n, int nct,
                                              double[] vBuf, int ldvbuf, double[] tBuf, int ldt,
                                              double[] wBuf, int ldw, double[] twBuf, int ldtw,
                                              double[] zBuf) {
        if (nct <= 0) return;

        for (int blockEnd = nct; blockEnd > 0; blockEnd -= blockSize) {
            final int blockStart = Math.max(0, blockEnd - blockSize);
            final int bb = blockEnd - blockStart;

            for (int r = 0; r < m; r++) {
                java.util.Arrays.fill(vBuf, r * ldvbuf, r * ldvbuf + bb, 0.0);
            }
            for (int l = 0; l < bb; l++) {
                final int gi = blockStart + l;
                vBuf[gi * ldvbuf + l] = 1.0;
                for (int r = gi + 1; r < m; r++) {
                    vBuf[r * ldvbuf + l] = A[r * lda + gi];
                }
            }

            for (int r = 0; r < bb; r++) {
                java.util.Arrays.fill(tBuf, r * ldt, r * ldt + bb, 0.0);
            }
            for (int i = 0; i < bb; i++) {
                final double tau_i = tauQ[blockStart + i];
                if (i == 0) {
                    tBuf[0 * ldt + 0] = tau_i;
                } else {
                    for (int l = 0; l < i; l++) {
                        double sum = 0.0;
                        for (int r = 0; r < m; r++) {
                            sum += vBuf[r * ldvbuf + l] * vBuf[r * ldvbuf + i];
                        }
                        zBuf[l] = sum;
                    }
                    for (int l = 0; l < i; l++) {
                        double sum = 0.0;
                        for (int p = l; p < i; p++) {
                            sum += tBuf[l * ldt + p] * zBuf[p];
                        }
                        tBuf[l * ldt + i] = -tau_i * sum;
                    }
                    tBuf[i * ldt + i] = tau_i;
                }
            }

            // Step 1: W = V^T * U   (bb × n) — 串行（并行会导致共享 W 缓冲区竞态）
            for (int l = 0; l < bb; l++) {
                java.util.Arrays.fill(wBuf, l * ldw, l * ldw + n, 0.0);
            }
            for (int r = 0; r < m; r++) {
                int vOff = r * ldvbuf;
                int uOff = r * ldu;
                for (int l = 0; l < bb; l++) {
                    double v = vBuf[vOff + l];
                    int wOff = l * ldw;
                    for (int col = 0; col < n; col++) {
                        wBuf[wOff + col] += v * uData[uOff + col];
                    }
                }
            }

            // Step 2: TW = T * W   (bb × n) — 上三角，循环交换（bb 很小，串行即可）
            for (int l = 0; l < bb; l++) {
                java.util.Arrays.fill(twBuf, l * ldtw, l * ldtw + n, 0.0);
            }
            for (int p = 0; p < bb; p++) {
                int wOffP = p * ldw;
                for (int l = 0; l <= p; l++) {
                    double t = tBuf[l * ldt + p];
                    int twOff = l * ldtw;
                    for (int col = 0; col < n; col++) {
                        twBuf[twOff + col] += t * wBuf[wOffP + col];
                    }
                }
            }

            // Step 3: U := U - V * TW — 循环交换，stride-1 访问，大矩阵并行
            if ((long) m * bb * n > 2_000_000L) {
                parallelRowRange(m, r0 -> {
                    for (int r = r0.start; r < r0.end; r++) {
                        int vOff = r * ldvbuf;
                        int uOff = r * ldu;
                        for (int l = 0; l < bb; l++) {
                            double v = vBuf[vOff + l];
                            int twOff = l * ldtw;
                            for (int col = 0; col < n; col++) {
                                uData[uOff + col] -= v * twBuf[twOff + col];
                            }
                        }
                    }
                });
            } else {
                for (int r = 0; r < m; r++) {
                    int vOff = r * ldvbuf;
                    int uOff = r * ldu;
                    for (int l = 0; l < bb; l++) {
                        double v = vBuf[vOff + l];
                        int twOff = l * ldtw;
                        for (int col = 0; col < n; col++) {
                            uData[uOff + col] -= v * twBuf[twOff + col];
                        }
                    }
                }
            }
        }
    }

    /**
     * 把 P_1 P_2 ... P_nrt 应用到 V（初始为 n×n 单位矩阵）。
     * <p>
     * P_i = I - tauP[i] * u_i u_i^T, u_i 存于 A[i, i+1:n]（A[i][i+1]=1 sentinel）。
     * V 是右因子，应用规则是 V := P_1 ... P_nrt * V？— 实际上对应于
     * V_full = (P_1 ... P_nrt)，但注意双对角化得到 A = U B V^T，因此右因子 V_full
     * 满足 V_full^T = P_nrt^T ... P_1^T = P_nrt ... P_1（Householder 是对称 + 自逆）。
     * </p>
     * <p>实际上与 RereSVDDecomposition 的 V 构造一致：从后向前应用 P_nrt-1, P_nrt-2, ..., P_0
     * 到累积 V 的右侧。每个 P_i 仅作用于 V 的 [i+1:n, *] 行。</p>
     * <p>所有临时矩阵均使用调用方预分配的共享缓冲区。</p>
     */
    private void applyRightHouseholdersBlocked(double[] vData, int ldv, double[] A, int lda,
                                               double[] tauP, int n, int nrt,
                                               double[] uRBuf, int ldur, double[] tBuf, int ldt,
                                               double[] wBuf, int ldw, double[] twBuf, int ldtw,
                                               double[] zBuf) {
        if (nrt <= 0) return;

        for (int blockEnd = nrt; blockEnd > 0; blockEnd -= blockSize) {
            final int blockStart = Math.max(0, blockEnd - blockSize);
            final int bb = blockEnd - blockStart;

            for (int r = 0; r < n; r++) {
                java.util.Arrays.fill(uRBuf, r * ldur, r * ldur + bb, 0.0);
            }
            for (int l = 0; l < bb; l++) {
                final int gi = blockStart + l;
                if (gi + 1 < n) {
                    uRBuf[(gi + 1) * ldur + l] = 1.0;
                    for (int r = gi + 2; r < n; r++) {
                        uRBuf[r * ldur + l] = A[gi * lda + r];
                    }
                }
            }

            for (int r = 0; r < bb; r++) {
                java.util.Arrays.fill(tBuf, r * ldt, r * ldt + bb, 0.0);
            }
            for (int i = 0; i < bb; i++) {
                final double tau_i = tauP[blockStart + i];
                if (i == 0) {
                    tBuf[0 * ldt + 0] = tau_i;
                } else {
                    for (int l = 0; l < i; l++) {
                        double sum = 0.0;
                        for (int r = 0; r < n; r++) {
                            sum += uRBuf[r * ldur + l] * uRBuf[r * ldur + i];
                        }
                        zBuf[l] = sum;
                    }
                    for (int l = 0; l < i; l++) {
                        double sum = 0.0;
                        for (int p = l; p < i; p++) {
                            sum += tBuf[l * ldt + p] * zBuf[p];
                        }
                        tBuf[l * ldt + i] = -tau_i * sum;
                    }
                    tBuf[i * ldt + i] = tau_i;
                }
            }

            // Step 1: W = U_R^T * V  (bb × n) — 串行（并行会导致共享 W 缓冲区竞态）
            for (int l = 0; l < bb; l++) {
                java.util.Arrays.fill(wBuf, l * ldw, l * ldw + n, 0.0);
            }
            for (int r = 0; r < n; r++) {
                int uOff = r * ldur;
                int vOff = r * ldv;
                for (int l = 0; l < bb; l++) {
                    double u = uRBuf[uOff + l];
                    int wOff = l * ldw;
                    for (int col = 0; col < n; col++) {
                        wBuf[wOff + col] += u * vData[vOff + col];
                    }
                }
            }

            // Step 2: TW = T * W   (bb × n) — 上三角，循环交换（bb 很小，串行即可）
            for (int l = 0; l < bb; l++) {
                java.util.Arrays.fill(twBuf, l * ldtw, l * ldtw + n, 0.0);
            }
            for (int p = 0; p < bb; p++) {
                int wOffP = p * ldw;
                for (int l = 0; l <= p; l++) {
                    double t = tBuf[l * ldt + p];
                    int twOff = l * ldtw;
                    for (int col = 0; col < n; col++) {
                        twBuf[twOff + col] += t * wBuf[wOffP + col];
                    }
                }
            }

            // Step 3: V -= U_R * TW — 循环交换，stride-1 访问，大矩阵并行
            if ((long) n * bb * n > 2_000_000L) {
                parallelRowRange(n, r0 -> {
                    for (int r = r0.start; r < r0.end; r++) {
                        int uOff = r * ldur;
                        int vOff = r * ldv;
                        for (int l = 0; l < bb; l++) {
                            double u = uRBuf[uOff + l];
                            int twOff = l * ldtw;
                            for (int col = 0; col < n; col++) {
                                vData[vOff + col] -= u * twBuf[twOff + col];
                            }
                        }
                    }
                });
            } else {
                for (int r = 0; r < n; r++) {
                    int uOff = r * ldur;
                    int vOff = r * ldv;
                    for (int l = 0; l < bb; l++) {
                        double u = uRBuf[uOff + l];
                        int twOff = l * ldtw;
                        for (int col = 0; col < n; col++) {
                            vData[vOff + col] -= u * twBuf[twOff + col];
                        }
                    }
                }
            }
        }
    }

    // ============================================================
    //  BD-SQR main loop (U^T + V^T transpose layout for stride-1)
    // ============================================================

    /**
     * 双对角 SVD 主循环（与 RereSVDDecomposition.bdsqrMainLoopVt 等价）。
     * uData 和 vData 均以转置布局传入（uData[col][row], vData[col][row]）。
     */
    private void bdsqrMainLoopVt(double[] singularValues, double[] e,
                                 double[][] uData, double[][] vData,
                                 int m, int n, int pInitial) {
        int p = pInitial;
        int iterCount = 0;
        while (p > 0 && iterCount < maxIterations) {
            int k;
            int kase;

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
                    if (ks == k) break;
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
                                    if (b < 0) sqrtDisc = -sqrtDisc;
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
                        if (j != k) e[j - 1] = t;
                        f = cs * singularValues[j] + sn * e[j];
                        e[j] = cs * e[j] - sn * singularValues[j];
                        g = sn * singularValues[j + 1];
                        singularValues[j + 1] = cs * singularValues[j + 1];

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

    // ============================================================
    //  Helpers
    // ============================================================

    /** 简单的行区间描述，供并行化辅助方法使用。 */
    private static final class RowRange {
        final int start;
        final int end;
        RowRange(int s, int e) { this.start = s; this.end = e; }
    }

    /**
     * 对按行独立的工作做 ForkJoin 粗粒度并行。
     * 只有当总行数 >= PARALLELISM_THRESHOLD 时才真正 fork，否则串行执行。
     */
    private static void parallelRowRange(int totalRows, java.util.function.Consumer<RowRange> task) {
        ForkJoinPool pool = com.yishape.lab.math.linalg.RereDoubleMatrix.getThreadPool();
        if (pool == null || totalRows < com.yishape.lab.math.linalg.RereDoubleMatrix.PARALLELISM_THRESHOLD) {
            task.accept(new RowRange(0, totalRows));
            return;
        }
        int numThreads = pool.getParallelism();
        int rowsPerTask = (totalRows + numThreads - 1) / numThreads;
        List<Future<?>> futures = new ArrayList<>();
        for (int t = 0; t < numThreads; t++) {
            final int r0 = t * rowsPerTask;
            final int r1 = Math.min(r0 + rowsPerTask, totalRows);
            if (r0 >= r1) break;
            futures.add(pool.submit(() -> task.accept(new RowRange(r0, r1))));
        }
        for (Future<?> f : futures) {
            try { f.get(); } catch (Exception ignored) { }
        }
    }

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
     * 把 m×k 瘦型正交矩阵补全为 m×nOut 完整正交（k &lt; nOut）。
     * <p>用 Modified Gram–Schmidt 在标准高斯种子上展开缺失列。与 RereSVDDecomposition 同接口同实现。</p>
     */
    private static IMatrix<Double> widenVToFullOrthogonal(IMatrix<Double> Vthin, int nOut, double epsilon) {
        int n = Vthin.rows();
        int k = Vthin.cols();
        if (k >= nOut) return Vthin;

        double[][] result = new double[n][nOut];
        for (int j = 0; j < k; j++) {
            for (int i = 0; i < n; i++) {
                result[i][j] = Vthin.get(i, j);
            }
        }
        java.util.Random rng = new java.util.Random(42);
        for (int j = k; j < nOut; j++) {
            for (int i = 0; i < n; i++) {
                result[i][j] = rng.nextGaussian();
            }
        }
        for (int j = k; j < nOut; j++) {
            for (int p = 0; p < j; p++) {
                double dot = 0.0;
                for (int i = 0; i < n; i++) dot += result[i][j] * result[i][p];
                for (int i = 0; i < n; i++) result[i][j] -= dot * result[i][p];
            }
            double norm = 0.0;
            for (int i = 0; i < n; i++) norm += result[i][j] * result[i][j];
            norm = Math.sqrt(norm);
            if (norm > epsilon) {
                for (int i = 0; i < n; i++) result[i][j] /= norm;
            } else {
                for (int i = 0; i < n; i++) result[i][j] = (i == j) ? 1.0 : 0.0;
                for (int p = 0; p < j; p++) {
                    double dot = 0.0;
                    for (int i = 0; i < n; i++) dot += result[i][j] * result[i][p];
                    for (int i = 0; i < n; i++) result[i][j] -= dot * result[i][p];
                }
                norm = 0.0;
                for (int i = 0; i < n; i++) norm += result[i][j] * result[i][j];
                norm = Math.sqrt(norm);
                if (norm > epsilon) {
                    for (int i = 0; i < n; i++) result[i][j] /= norm;
                }
            }
        }
        return Linalg.matrix(result);
    }

    private void ensureValidSingularValues() {
        if (cachedS == null || cachedU == null || cachedVT == null) return;
        int n = cachedS.length();

        for (int i = 0; i < n; i++) {
            if (cachedS.get(i) < 0) {
                cachedS.set(i, -cachedS.get(i));
                for (int j = 0; j < cachedU.rows(); j++) {
                    cachedU.set(j, i, -cachedU.get(j, i));
                }
            }
        }

        double[] svals = new double[n];
        for (int i = 0; i < n; i++) svals[i] = cachedS.get(i);
        Integer[] ord = new Integer[n];
        for (int i = 0; i < n; i++) ord[i] = i;
        Arrays.sort(ord, (a, b) -> Double.compare(svals[b], svals[a]));

        boolean identity = true;
        for (int i = 0; i < n; i++) {
            if (ord[i] != i) { identity = false; break; }
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
            for (int i = n; i < rowsVT; i++) {
                System.arraycopy(vtRaw[i], 0, newVt[i], 0, colsVT);
            }
            cachedU = Linalg.matrix(newU);
            cachedVT = Linalg.matrix(newVt);
            for (int i = 0; i < n; i++) cachedS.set(i, newS[i]);
            this.cachedSingularValues = newS;
        }
    }
}
