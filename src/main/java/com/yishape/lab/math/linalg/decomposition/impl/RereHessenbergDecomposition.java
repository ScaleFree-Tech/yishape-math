package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.decomposition.IHessenbergDecomposition;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.solver.HessenbergDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.NonSquareMatrixException;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.math.util.RerePrecision;

/**
 * Hessenberg decomposition implementation with enhanced numerical stability.
 * <p>
 * Hessenberg decomposition transforms a matrix A into the form A = Q * H * Q^T,
 * where H is a Hessenberg matrix, and Q is an orthogonal matrix.
 * </p>
 * 
 * <h3>Algorithm Improvements</h3>
 * <ul>
 *   <li>Enhanced Householder reflection computations</li>
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
public class RereHessenbergDecomposition implements IHessenbergDecomposition {
    
    /** Cached value of H. */
    private IMatrix<Double> cachedH;
    /** Cached value of Q. */
    private IMatrix<Double> cachedQ;
    /** Determinant of the matrix. */
    private Double determinant;
    /** Condition number of the matrix. */
    private Double conditionNumber;
    /** Rank of the matrix. */
    private Integer rank;
    /** Epsilon for numerical comparisons. */
    private double epsilon;
    /** Maximum number of iterations. */
    private int maxIterations;
    
    /**
     * Default constructor with default parameters.
     */
    public RereHessenbergDecomposition() {
        this.epsilon = DEFAULT_EPSILON;
        this.maxIterations = DEFAULT_MAX_ITERATIONS;
    }
    
    /**
     * Constructor with unified parameters.
     * 
     * @param epsilon threshold for considering an element as zero
     * @param maxIterations maximum number of iterations
     */
    public RereHessenbergDecomposition(double epsilon, int maxIterations) {
        this.epsilon = epsilon;
        this.maxIterations = maxIterations;
    }
    
    /**
     * Reduce a square matrix to upper Hessenberg form using Householder reflections.
     * <p>Static helper that works entirely on raw double[][] — no IMatrix boxing.
     * Callers with raw data (e.g. RereEigenDecomposition) can use this directly.</p>
     *
     * @param data square input matrix (not modified)
     * @param n matrix dimension
     * @param epsilon threshold for numerical comparisons
     * @return (H, Q) where H is upper Hessenberg and Q is orthogonal
     */
    public static Tuple2<double[][], double[][]> reduceToHessenberg(double[][] data, int n, double epsilon) {
        double[][] H = new double[n][n];
        double[][] Q = new double[n][n];

        for (int i = 0; i < n; i++) {
            System.arraycopy(data[i], 0, H[i], 0, n);
        }
        for (int i = 0; i < n; i++) Q[i][i] = 1.0;

        if (n <= 2) {
            return new Tuple2<>(H, Q);
        }

        double[] colBuf = new double[n];
        double[] v = new double[n];

        for (int k = 0; k < n - 2; k++) {
            int len = n - k - 1;

            double norm = 0.0;
            for (int i = 0; i < len; i++) {
                double val = H[k + 1 + i][k];
                v[i] = val;
                norm += val * val;
            }
            norm = Math.sqrt(norm);
            if (norm <= epsilon) continue;

            v[0] += (v[0] >= 0 ? norm : -norm);

            double vNorm = 0.0;
            for (int i = 0; i < len; i++) vNorm += v[i] * v[i];
            vNorm = Math.sqrt(vNorm);
            if (vNorm <= epsilon) continue;

            for (int i = 0; i < len; i++) v[i] /= vNorm;

            for (int j = k; j < n; j++) {
                double sum = 0.0;
                for (int i = 0; i < len; i++) {
                    double val = H[k + 1 + i][j];
                    colBuf[i] = val;
                    sum += v[i] * val;
                }
                double twoSum = 2.0 * sum;
                for (int i = 0; i < len; i++) {
                    H[k + 1 + i][j] = colBuf[i] - twoSum * v[i];
                }
            }

            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                double[] Hi = H[i];
                for (int j = 0; j < len; j++) {
                    sum += Hi[k + 1 + j] * v[j];
                }
                for (int j = 0; j < len; j++) {
                    Hi[k + 1 + j] -= 2.0 * sum * v[j];
                }
            }

            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                double[] Qi = Q[i];
                for (int j = 0; j < len; j++) {
                    sum += Qi[k + 1 + j] * v[j];
                }
                for (int j = 0; j < len; j++) {
                    Qi[k + 1 + j] -= 2.0 * sum * v[j];
                }
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (Math.abs(i - j) > 1 && Math.abs(H[i][j]) <= epsilon) {
                    H[i][j] = 0.0;
                }
            }
        }

        return new Tuple2<>(H, Q);
    }

    @Override
    public Tuple2<IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix) {
        return decompose(matrix, epsilon);
    }
    
    @Override
    public Tuple2<IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix, double epsilon) {
        return decompose(matrix, epsilon, DEFAULT_MAX_ITERATIONS);
    }
    
    @Override
    public Tuple2<IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix, double epsilon, int maxIterations) {
        // Reset cached values
        cachedH = null;
        cachedQ = null;
        determinant = null;
        conditionNumber = null;
        rank = null;
        this.epsilon = epsilon;
        this.maxIterations = maxIterations;
        
        IDoubleMatrix doubleMatrix = (IDoubleMatrix) matrix;
        double[][] data = doubleMatrix.getData();
        int n = data.length;
        
        // Check if matrix is square
        if (data.length != data[0].length) {
            throw new NonSquareMatrixException(
                "Hessenberg decomposition requires square matrix",
                "Hessenberg Decomposition", 
                "Matrix " + data.length + "x" + data[0].length,
                data.length, data[0].length);
        }
        
        double[][] H = new double[n][n];
        double[][] Q = new double[n][n];

        // Copy original matrix
        for (int i = 0; i < n; i++) {
            System.arraycopy(data[i], 0, H[i], 0, n);
        }

        // Initialize Q as identity matrix
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Q[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }

        // For small matrices, return directly
        if (n <= 2) {
            cachedH = IDoubleMatrix.of(H);
            cachedQ = IDoubleMatrix.of(Q);
            return new Tuple2<>(cachedH, cachedQ);
        }

        // Column buffer for stride-1 Householder computation.
        // The left multiply (I-2vv^T)*H has a stride-n column scan in the
        // inner loop: H[k+1+i][j] with i varying.  Extracting the column into
        // this buffer separates the memory stride-n pass from the arithmetic,
        // letting the dot product / axpy run at full L1-cached speed.
        double[] colBuf = new double[n];
        // Pre-allocated Householder vector buffer reused across iterations.
        double[] v = new double[n];

        // Apply Householder transformations to each column
        for (int k = 0; k < n - 2; k++) {
            int len = n - k - 1;

            // Compute Householder vector v directly from column k
            double norm = 0.0;
            for (int i = 0; i < len; i++) {
                double val = H[k + 1 + i][k];
                v[i] = val;
                norm += val * val;
            }
            norm = Math.sqrt(norm);

            if (RerePrecision.equalsZero(norm, epsilon)) {
                continue;
            }

            // v[0] = x[0] + sign(x[0]) * norm (the remaining v[i]=x[i] are already set)
            v[0] += (v[0] >= 0 ? norm : -norm);

            // Normalize v
            double vNorm = 0.0;
            for (int i = 0; i < len; i++) {
                vNorm += v[i] * v[i];
            }
            vNorm = Math.sqrt(vNorm);

            if (RerePrecision.equalsZero(vNorm, epsilon)) {
                continue;
            }

            for (int i = 0; i < len; i++) {
                v[i] /= vNorm;
            }

            // Efficiently apply Householder transformation: avoid constructing full matrix
            // Apply left multiplication: H = (I - 2vv^T) * H
            // Column extraction into colBuf decouples the stride-n column scan from
            // the arithmetic, enabling stride-1 dot/axpy on L1-cached data.
            for (int j = k; j < n; j++) {
                double sum = 0.0;
                for (int i = 0; i < len; i++) {
                    double val = H[k + 1 + i][j];
                    colBuf[i] = val;
                    sum += v[i] * val;
                }
                double twoSum = 2.0 * sum;
                for (int i = 0; i < len; i++) {
                    H[k + 1 + i][j] = colBuf[i] - twoSum * v[i];
                }
            }

            // Apply right multiplication: H = H * (I - 2vv^T)
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int j = 0; j < len; j++) {
                    sum += H[i][k + 1 + j] * v[j];
                }
                for (int j = 0; j < len; j++) {
                    H[i][k + 1 + j] -= 2.0 * sum * v[j];
                }
            }

            // Update Q matrix: Q = Q * (I - 2vv^T)
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int j = 0; j < len; j++) {
                    sum += Q[i][k + 1 + j] * v[j];
                }
                for (int j = 0; j < len; j++) {
                    Q[i][k + 1 + j] -= 2.0 * sum * v[j];
                }
            }
        }

        // Ensure Hessenberg form: clean numerical errors
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (Math.abs(i - j) > 1 && RerePrecision.equalsZero(H[i][j], epsilon)) {
                    H[i][j] = 0.0;
                }
            }
        }

        cachedH = IDoubleMatrix.of(H);
        cachedQ = IDoubleMatrix.of(Q);
        return new Tuple2<>(cachedH, cachedQ);
    }
    
    @Override
    public IDecompositionSolver getSolver() {
        if (cachedH == null || cachedQ == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        // Return the standalone solver
        return new HessenbergDecompositionSolver(cachedH, cachedQ, epsilon);
    }
    
    @Override
    public double getDeterminant() {
        if (determinant == null) {
            if (cachedH == null) {
                throw new IllegalStateException("Decomposition not yet performed");
            }
            double det = 1.0;
            for (int i = 0; i < cachedH.rows(); ++i) {
                det *= cachedH.get(i, i);
            }
            determinant = det;
        }
        return determinant;
    }
    
    @Override
    public boolean isNonSingular() {
        if (cachedH == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        // A matrix is non-singular if all diagonal elements of H are non-zero
        for (int i = 0; i < cachedH.rows(); i++) {
            if (Math.abs(cachedH.get(i, i)) < epsilon) {
                return false;
            }
        }
        return true;
    }
    
    @Override
    public double getConditionNumber() {
        if (conditionNumber == null) {
            if (cachedH == null) {
                throw new IllegalStateException("Decomposition not yet performed");
            }
            // Estimate condition number using the ratio of largest to smallest diagonal elements
            double maxDiag = 0.0;
            double minDiag = Double.POSITIVE_INFINITY;
            for (int i = 0; i < cachedH.rows(); i++) {
                double diag = Math.abs(cachedH.get(i, i));
                maxDiag = Math.max(maxDiag, diag);
                minDiag = Math.min(minDiag, diag);
            }
            if (minDiag > epsilon) {
                conditionNumber = maxDiag / minDiag;
            } else {
                conditionNumber = Double.POSITIVE_INFINITY;
            }
        }
        return conditionNumber;
    }
    
    @Override
    public int getRank() {
        if (rank == null) {
            if (cachedH == null) {
                throw new IllegalStateException("Decomposition not yet performed");
            }
            // Rank is the number of non-zero diagonal elements
            int r = 0;
            for (int i = 0; i < cachedH.rows(); i++) {
                if (Math.abs(cachedH.get(i, i)) > epsilon) {
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
    
    /**
     * 分块海森伯格化简 - 将矩阵转换为海森伯格形式
     *
     * <p>
     * 海森伯格化简是特征分解的关键预处理步骤，将任意矩阵A通过相似变换转换为海森伯格矩阵H，
     * 使得A = Q * H * Q^T，其中Q是正交矩阵，H是海森伯格矩阵（上三角矩阵加上一条次对角线）。</p>
     *
     * <p>
     * 分块算法的优势：</p>
     * <ul>
     * <li>提高缓存效率：分块处理减少内存访问次数</li>
     * <li>减少浮点运算：避免重复计算</li>
     * <li>支持大矩阵：内存使用更优化</li>
     * </ul>
     *
     * <p>
     * 算法原理：</p>
     * <ol>
     * <li>对每一列k，计算Householder向量v，使得H[k+1:n, k] = 0</li>
     * <li>应用Householder变换：H = (I - 2vv^T) * H * (I - 2vv^T)</li>
     * <li>分块处理：将列分成块，逐块应用变换</li>
     * </ol>
     *
     * <p>
     * 时间复杂度：O(n³)，但常数因子比标准算法小</p>
     * <p>
     * 空间复杂度：O(n²)</p>
     *
     * @return 包含海森伯格矩阵H和变换矩阵Q的元组
     */
    public Tuple2<double[][], double[][]> blockedHessenbergReduction(IDoubleMatrix matrix) {
        double[][] data = matrix.getData();
        int n = data.length;
        double[][] H = new double[n][n];
        double[][] Q = new double[n][n];

        // 步骤1：复制原矩阵到工作矩阵H
        for (int i = 0; i < n; i++) {
            System.arraycopy(data[i], 0, H[i], 0, n);
        }

        // 步骤2：初始化变换矩阵Q为单位矩阵
        // Q将记录所有Householder变换的累积效果
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Q[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }

        // 步骤3：计算分块大小 - 根据矩阵大小自适应调整
        // 分块大小影响缓存效率和计算开销的平衡
        int blockSize = Math.min(64, Math.max(16, n / 8));

        // 步骤4：分块处理每一列，应用Householder变换
        for (int k = 0; k < n - 2; k += blockSize) {
            int endK = Math.min(k + blockSize, n - 2);

            // 处理当前块内的每一列
            for (int j = k; j < endK; j++) {
                // 步骤4.1：计算当前列j的Householder向量
                // 计算H[j+1:n, j]的2-范数
                double norm = 0.0;
                for (int i = j + 1; i < n; i++) {
                    norm += H[i][j] * H[i][j];
                }
                norm = Math.sqrt(norm);

                // 只有当范数足够大时才进行变换（避免数值不稳定）
                if (norm > 1e-10) {
                    // 步骤4.2：构造Householder向量v
                    // v = H[j+1:n, j] + sign(H[j+1,j]) * ||H[j+1:n, j]|| * e1
                    double[] v = new double[n - j - 1];
                    v[0] = H[j + 1][j] + Math.signum(H[j + 1][j]) * norm;
                    for (int i = 1; i < n - j - 1; i++) {
                        v[i] = H[j + 1 + i][j];
                    }

                    // 步骤4.3：归一化Householder向量
                    // 确保v是单位向量
                    double vNorm = 0.0;
                    for (int i = 0; i < v.length; i++) {
                        vNorm += v[i] * v[i];
                    }
                    vNorm = Math.sqrt(vNorm);
                    for (int i = 0; i < v.length; i++) {
                        v[i] /= vNorm;
                    }

                    // 步骤4.4：应用Householder变换到海森伯格矩阵H
                    // H = (I - 2vv^T) * H，只影响H[j+1:n, j:n-1]部分
                    applyHouseholderToHessenberg(H, v, j + 1, n - 1, j, n - 1);

                    // 步骤4.5：应用Householder变换到变换矩阵Q
                    // Q = Q * (I - 2vv^T)，更新Q的所有行
                    applyHouseholderToHessenberg(Q, v, 0, n - 1, j + 1, n - 1);
                }
            }
        }

        return new Tuple2<>(H, Q);
    }

    /**
     * 应用Householder变换到海森伯格化简
     */
    private void applyHouseholderToHessenberg(double[][] matrix, double[] v, int startRow, int endRow, int startCol, int endCol) {
        int n = v.length;

        // 计算 w = matrix * v
        double[] w = new double[endRow - startRow + 1];
        for (int i = 0; i < w.length; i++) {
            w[i] = 0.0;
            for (int j = 0; j < n; j++) {
                w[i] += matrix[startRow + i][startCol + j] * v[j];
            }
        }

        // 计算 matrix = matrix - 2 * w * v^T
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < n; j++) {
                matrix[startRow + i][startCol + j] -= 2.0 * w[i] * v[j];
            }
        }
    }
}