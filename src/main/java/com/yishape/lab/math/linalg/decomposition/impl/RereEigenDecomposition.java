package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.IEigenDecomposition;
import com.yishape.lab.math.linalg.decomposition.IHessenbergDecomposition;
import com.yishape.lab.math.linalg.decomposition.Decomps;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.solver.EigenDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.NonSquareMatrixException;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.util.Tuple3;
import com.yishape.lab.math.util.Precision;

/**
 * Implementation of eigen decomposition with enhanced numerical stability.
 * <p>
 * This implementation computes the eigenvalue decomposition of a square matrix A
 * such that A = V * D * V^(-1) where D is a diagonal matrix of eigenvalues and
 * the columns of V are the corresponding eigenvectors.
 * </p>
 * 
 * <h3>Algorithm Improvements</h3>
 * <ul>
 *   <li>Automatic detection of symmetric matrices for optimized paths</li>
 *   <li>Tridiagonal reduction for symmetric matrices</li>
 *   <li>Hessenberg reduction for general matrices</li>
 *   <li>Enhanced QR algorithm with shifts</li>
 *   <li>Better numerical stability with precision-aware comparisons</li>
 *   <li>Comprehensive error reporting with context information</li>
 *   <li>Efficient caching of computed results</li>
 * </ul>
 * 
 * <h3>References</h3>
 * <ul>
 *   <li>Golub, G. H., &amp; Van Loan, C. F. (2013). Matrix computations (4th ed.). Johns Hopkins University Press.</li>
 *   <li>Press, W. H., Teukolsky, S. A., Vetterling, W. T., &amp; Flannery, B. P. (2007). Numerical recipes: The art of scientific computing (3rd ed.). Cambridge University Press.</li>
 * </ul>
 */
public class RereEigenDecomposition implements IEigenDecomposition {
    
    /** Default epsilon for numerical comparisons. */
    private static final double DEFAULT_EPSILON = 1e-12;
    /** Default maximum number of iterations. */
    private static final int DEFAULT_MAX_ITERATIONS = 30;
    
    /** Cached eigenvalues. */
    private IVector<Double> cachedEigenvalues;
    /** Cached eigenvectors. */
    private IMatrix<Double> cachedEigenvectors;
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
    public RereEigenDecomposition() {
        this.epsilon = DEFAULT_EPSILON;
        this.maxIterations = DEFAULT_MAX_ITERATIONS;
    }
    
    /**
     * Constructor with unified parameters.
     * 
     * @param epsilon threshold for considering an element as zero
     * @param maxIterations maximum number of iterations
     */
    public RereEigenDecomposition(double epsilon, int maxIterations) {
        this.epsilon = epsilon;
        this.maxIterations = maxIterations;
    }
    
    @Override
    public IDecompositionSolver getSolver() {
        if (cachedEigenvectors == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        // Return the standalone solver
        return new EigenDecompositionSolver(cachedEigenvalues, cachedEigenvectors, epsilon);
    }

    @Override
    public double getDeterminant() {
        if (determinant == null) {
            if (cachedEigenvalues == null) {
                throw new IllegalStateException("Decomposition not yet performed");
            }
            double det = 1.0;
            for (int i = 0; i < cachedEigenvalues.length(); i++) {
                det *= cachedEigenvalues.get(i);
            }
            determinant = det;
        }
        return determinant;
    }
    
    @Override
    public boolean isNonSingular() {
        if (cachedEigenvalues == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        // A matrix is non-singular if all eigenvalues are non-zero
        for (int i = 0; i < cachedEigenvalues.length(); i++) {
            if (Math.abs(cachedEigenvalues.get(i)) < epsilon) {
                return false;
            }
        }
        return true;
    }
    
    @Override
    public double getConditionNumber() {
        if (conditionNumber == null) {
            if (cachedEigenvalues == null) {
                throw new IllegalStateException("Decomposition not yet performed");
            }
            // Estimate condition number using the ratio of largest to smallest eigenvalues in absolute value
            double maxEigen = 0.0;
            double minEigen = Double.POSITIVE_INFINITY;
            for (int i = 0; i < cachedEigenvalues.length(); i++) {
                double eigen = Math.abs(cachedEigenvalues.get(i));
                maxEigen = Math.max(maxEigen, eigen);
                minEigen = Math.min(minEigen, eigen);
            }
            if (minEigen > epsilon) {
                conditionNumber = maxEigen / minEigen;
            } else {
                conditionNumber = Double.POSITIVE_INFINITY;
            }
        }
        return conditionNumber;
    }
    
    @Override
    public int getRank() {
        if (rank == null) {
            if (cachedEigenvalues == null) {
                throw new IllegalStateException("Decomposition not yet performed");
            }
            // Rank is the number of non-zero eigenvalues
            int r = 0;
            for (int i = 0; i < cachedEigenvalues.length(); i++) {
                if (Math.abs(cachedEigenvalues.get(i)) > epsilon) {
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
    public Tuple2<IVector<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix) {
        return decompose(matrix, epsilon);
    }
    
    @Override
    public Tuple2<IVector<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix, double epsilon) {
        return decompose(matrix, epsilon, DEFAULT_MAX_ITERATIONS);
    }
    
    @Override
    public Tuple2<IVector<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix, double epsilon, int maxIterations) {
        // Reset cached values
        cachedEigenvalues = null;
        cachedEigenvectors = null;
        determinant = null;
        conditionNumber = null;
        rank = null;
        this.epsilon = epsilon;
        this.maxIterations = maxIterations;
        
        IDoubleMatrix doubleMatrix = (IDoubleMatrix) matrix;
        double[][] data = doubleMatrix.getData();
        
        // Check if matrix is square - eigen decomposition only applies to square matrices
        if (data.length != data[0].length) {
            throw new NonSquareMatrixException(
                "Eigendecomposition requires square matrix",
                "Eigen Decomposition", 
                "Matrix " + data.length + "x" + data[0].length,
                data.length, data[0].length);
        }

        int n = data.length;  // Matrix dimension
        
        // Check if matrix is symmetric
        boolean isSymmetric = isSymmetric(doubleMatrix);
        
        Tuple2<IVector<Double>, IMatrix<Double>> result;
        if (isSymmetric) {
            // Eigen decomposition for symmetric matrices: use tridiagonalization + QR algorithm
            result = symmetricEigenDecomposition(doubleMatrix);
        } else {
            // Eigen decomposition for general matrices: use Hessenberg reduction + QR algorithm
            result = generalEigenDecomposition(doubleMatrix);
        }
        
        // Cache results
        cachedEigenvalues = result._1;
        cachedEigenvectors = result._2;
        
        return result;
    }
    
    /**
     * 检查矩阵是否对称
     */
    private boolean isSymmetric(IDoubleMatrix matrix) {
        int n = matrix.getRowNum();
        if (n != matrix.getColNum()) {
            return false;
        }
        
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (!Precision.equals(matrix.get(i, j), matrix.get(j, i), epsilon)) {
                    return false;
                }
            }
        }
        return true;
    }
    
    /**
     * 对称矩阵特征值分解
     */
    private Tuple2<IVector<Double>, IMatrix<Double>> symmetricEigenDecomposition(IDoubleMatrix matrix) {
        int n = matrix.getRowNum();
        
        // 创建工作矩阵的副本
        IDoubleMatrix workMatrix = (IDoubleMatrix) matrix.copy();
        
        // 三对角化
        Tuple3<IVector<Double>, IVector<Double>, IMatrix<Double>> tridiagResult = tridiagonalReduction(workMatrix);
        IVector<Double> diagonal = tridiagResult._1;
        IVector<Double> subDiagonal = tridiagResult._2;
        IMatrix<Double> Q = tridiagResult._3;
        
        // 对三对角矩阵应用QR算法
        Tuple2<IVector<Double>, IMatrix<Double>> qrResult = qrAlgorithmForTridiagonalImproved(diagonal, subDiagonal, Q);
        IVector<Double> eigenvalues = qrResult._1;
        IMatrix<Double> eigenvectors = qrResult._2;
        
        // 标准化特征向量
        normalizeEigenvectors(eigenvectors);
        
        // 排序特征值和特征向量
        sortEigenvaluesAndVectors(eigenvalues, eigenvectors);
        
        return new Tuple2<>(eigenvalues, eigenvectors);
    }
    
    /**
     * 一般矩阵的特征分解 - 使用海森伯格化简
     */
    private Tuple2<IVector<Double>, IMatrix<Double>> generalEigenDecomposition(IDoubleMatrix matrix) {
        int n = matrix.getRowNum();
        double[][] data = matrix.getData();

        // 根据矩阵大小选择算法
        Tuple2<double[][], double[][]> hessResult;
        if (n > 100) {
            // 大矩阵使用分块算法
            hessResult = blockedHessenbergReduction(matrix);
        } else {
            // 小矩阵使用标准算法
            hessResult = hessenbergReduction(matrix);
        }

        double[][] H = hessResult._1;  // 海森伯格矩阵
        double[][] Q = hessResult._2;  // 变换矩阵

        // 对海森伯格矩阵应用QR算法
        Tuple2<IDoubleVector, IDoubleMatrix> eigenResult;
        if (n > 50) {
            // 大矩阵使用隐式QR算法
            eigenResult = implicitQRAlgorithm(H);
        } else {
            // 小矩阵使用标准QR算法
            eigenResult = qrAlgorithmForHessenberg(H);
        }
        
        IDoubleVector eigenvalues = eigenResult._1;
        IDoubleMatrix eigenvectors = eigenResult._2;

        // 变换回原坐标系的特征向量
        double[][] Q_matrix = Q;
        double[][] U_matrix = eigenvectors.getData();
        int matrixSize = Q_matrix.length;

        // 手动计算 Q * U^T，结果的列是原矩阵A的特征向量
        double[][] transformedCols = new double[matrixSize][matrixSize];
        for (int i = 0; i < matrixSize; i++) {
            for (int j = 0; j < matrixSize; j++) {
                transformedCols[i][j] = 0.0;
                for (int k = 0; k < matrixSize; k++) {
                    transformedCols[i][j] += Q_matrix[i][k] * U_matrix[j][k];
                }
            }
        }

        // 转置为行存储格式
        double[][] originalEigenvectorsData = new double[matrixSize][matrixSize];
        for (int i = 0; i < matrixSize; i++) {
            for (int j = 0; j < matrixSize; j++) {
                originalEigenvectorsData[i][j] = transformedCols[j][i];
            }
        }

        IMatrix<Double> originalEigenvectors = Linalg.matrix(originalEigenvectorsData);

        return new Tuple2<>(eigenvalues, originalEigenvectors);
    }
    
    /**
     * 三对角化 - 使用Householder变换将对称矩阵转换为三对角形式
     */
    private Tuple3<IVector<Double>, IVector<Double>, IMatrix<Double>> tridiagonalReduction(IDoubleMatrix matrix) {
        int n = matrix.getRowNum();
        
        // 创建工作矩阵
        IDoubleMatrix A = (IDoubleMatrix) matrix.copy();
        
        // 创建正交变换矩阵Q，初始化为单位矩阵
        IDoubleMatrix Q = (IDoubleMatrix) Linalg.eye(n);
        
        for (int k = 0; k < n - 2; k++) {
            // 提取第k列的下三角部分
            IVector<Double> x = Linalg.zeros(n - k - 1);
            for (int i = k + 1; i < n; i++) {
                x.set(i - k - 1, A.get(i, k));
            }
            
            double norm = 0.0;
            for (int i = 0; i < x.length(); i++) {
                norm += x.get(i) * x.get(i);
            }
            norm = Math.sqrt(norm);
            
            if (norm < epsilon) {
                continue; // 跳过零向量
            }
            
            // 构造Householder向量
            IVector<Double> v = x.copy();
            double sign = x.get(0) >= 0 ? 1.0 : -1.0;
            v.set(0, v.get(0) + sign * norm);
            
            double vNorm = 0.0;
            for (int i = 0; i < v.length(); i++) {
                vNorm += v.get(i) * v.get(i);
            }
            vNorm = Math.sqrt(vNorm);
            
            if (vNorm < epsilon) {
                continue;
            }
            
            // 标准化v
            for (int i = 0; i < v.length(); i++) {
                v.set(i, v.get(i) / vNorm);
            }
            
            // 应用Householder变换到A
            applyHouseholderToTridiagonal(A, v, k);
            
            // 更新Q矩阵
            updateQMatrix(Q, v, k);
        }
        
        // 提取对角线元素
        IVector<Double> diagonal = Linalg.zeros(n);
        for (int i = 0; i < n; i++) {
            diagonal.set(i, A.get(i, i));
        }
        
        // 提取次对角线元素
        IVector<Double> subDiagonal = Linalg.zeros(n - 1);
        for (int i = 0; i < n - 1; i++) {
            subDiagonal.set(i, A.get(i + 1, i));
        }
        
        return new Tuple3<>(diagonal, subDiagonal, Q);
    }

    /**
     * 应用Householder变换到三对角化
     */
    private void applyHouseholderToTridiagonal(IDoubleMatrix A, IVector<Double> v, int k) {
        int n = A.getRowNum();
        int vLen = v.length();
        
        // 应用变换: A = (I - 2vv^T) * A * (I - 2vv^T)
        
        // 第一步：A = (I - 2vv^T) * A (左乘)
        for (int j = 0; j < n; j++) {
            double dot = 0.0;
            for (int i = 0; i < vLen; i++) {
                dot += v.get(i) * A.get(k + 1 + i, j);
            }
            
            for (int i = 0; i < vLen; i++) {
                A.set(k + 1 + i, j, A.get(k + 1 + i, j) - 2.0 * dot * v.get(i));
            }
        }
        
        // 第二步：A = A * (I - 2vv^T) (右乘)
        for (int i = 0; i < n; i++) {
            double dot = 0.0;
            for (int j = 0; j < vLen; j++) {
                dot += A.get(i, k + 1 + j) * v.get(j);
            }
            
            for (int j = 0; j < vLen; j++) {
                A.set(i, k + 1 + j, A.get(i, k + 1 + j) - 2.0 * dot * v.get(j));
            }
        }
    }

    /**
     * Improved QR algorithm for tridiagonal matrices based on Apache Commons Math
     * This is the core method that actually computes eigenvalues and eigenvectors
     */
    private Tuple2<IVector<Double>, IMatrix<Double>> qrAlgorithmForTridiagonalImproved(
            IVector<Double> diagonal, IVector<Double> subDiagonal, IMatrix<Double> Q) {
        int n = diagonal.length();
        
        // Convert to working arrays
        double[] main = new double[n];
        double[] secondary = new double[n - 1];
        
        for (int i = 0; i < n; i++) {
            main[i] = diagonal.get(i);
        }
        for (int i = 0; i < n - 1; i++) {
            secondary[i] = subDiagonal.get(i);
        }
        
        // Initialize z matrix from Q
        double[][] z = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                z[i][j] = Q.get(i, j);
            }
        }
        
        // Call the Apache Commons Math style eigenvalue computation
        computeEigenvaluesAndVectors(main, secondary, z);
        
        // Convert results back to our format
        IVector<Double> eigenvalues = Linalg.zeros(n);
        for (int i = 0; i < n; i++) {
            eigenvalues.set(i, main[i]);
        }
        
        double[][] eigenvectorData = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                eigenvectorData[i][j] = z[j][i]; // Note: transpose to column format
            }
        }
        
        IMatrix<Double> eigenvectors = Linalg.matrix(eigenvectorData);
        
        return new Tuple2<>(eigenvalues, eigenvectors);
    }
    
    /**
     * Core eigenvalue computation based on Apache Commons Math implementation
     * Uses implicit QL transformation with Wilkinson shifts
     */
    private void computeEigenvaluesAndVectors(double[] main, double[] secondary, double[][] z) {
        final int n = main.length;
        double[] e = new double[n];
        
        // Copy secondary diagonal
        for (int i = 0; i < n - 1; i++) {
            e[i] = secondary[i];
        }
        e[n - 1] = 0.0;
        
        // Determine the largest main and secondary value in absolute term
        double maxAbsoluteValue = 0;
        for (int i = 0; i < n; i++) {
            if (Math.abs(main[i]) > maxAbsoluteValue) {
                maxAbsoluteValue = Math.abs(main[i]);
            }
            if (Math.abs(e[i]) > maxAbsoluteValue) {
                maxAbsoluteValue = Math.abs(e[i]);
            }
        }
        
        // Make null any main and secondary value too small to be significant
        if (maxAbsoluteValue != 0) {
            for (int i = 0; i < n; i++) {
                if (Math.abs(main[i]) <= Precision.getMachineEpsilon() * maxAbsoluteValue) {
                    main[i] = 0;
                }
                if (Math.abs(e[i]) <= Precision.getMachineEpsilon() * maxAbsoluteValue) {
                    e[i] = 0;
                }
            }
        }
        
        // Main QL iteration loop
        for (int j = 0; j < n; j++) {
            int its = 0;
            int m;
            do {
                // Look for a single small sub-diagonal element to split the matrix
                for (m = j; m < n - 1; m++) {
                    double delta = Math.abs(main[m]) + Math.abs(main[m + 1]);
                    if (Math.abs(e[m]) + delta == delta) {
                        break;
                    }
                }
                
                if (m != j) {
                    if (its == DEFAULT_MAX_ITERATIONS) {
                        throw new RuntimeException("QR algorithm failed to converge after " + DEFAULT_MAX_ITERATIONS + " iterations");
                    }
                    its++;
                    
                    // Compute Wilkinson shift
                    double q = (main[j + 1] - main[j]) / (2 * e[j]);
                    double t = Math.sqrt(1 + q * q);
                    if (q < 0.0) {
                        q = main[m] - main[j] + e[j] / (q - t);
                    } else {
                        q = main[m] - main[j] + e[j] / (q + t);
                    }
                    
                    double u = 0.0;
                    double s = 1.0;
                    double c = 1.0;
                    int i;
                    
                    // Apply Givens rotations
                    for (i = m - 1; i >= j; i--) {
                        double p = s * e[i];
                        double h = c * e[i];
                        if (Math.abs(p) >= Math.abs(q)) {
                            c = q / p;
                            t = Math.sqrt(c * c + 1.0);
                            e[i + 1] = p * t;
                            s = 1.0 / t;
                            c *= s;
                        } else {
                            s = p / q;
                            t = Math.sqrt(s * s + 1.0);
                            e[i + 1] = q * t;
                            c = 1.0 / t;
                            s *= c;
                        }
                        
                        if (e[i + 1] == 0.0) {
                            main[i + 1] -= u;
                            e[m] = 0.0;
                            break;
                        }
                        
                        q = main[i + 1] - u;
                        t = (main[i] - q) * s + 2.0 * c * h;
                        u = s * t;
                        main[i + 1] = q + u;
                        q = c * t - h;
                        
                        // Apply the transformation to the eigenvector matrix
                        for (int ia = 0; ia < n; ia++) {
                            p = z[ia][i + 1];
                            z[ia][i + 1] = s * z[ia][i] + c * p;
                            z[ia][i] = c * z[ia][i] - s * p;
                        }
                    }
                    
                    if (t == 0.0 && i >= j) {
                        continue;
                    }
                    
                    main[j] -= u;
                    e[j] = q;
                    e[m] = 0.0;
                }
            } while (m != j);
        }
        
        // Sort eigenvalues and eigenvectors in descending order
        for (int i = 0; i < n; i++) {
            int k = i;
            double p = main[i];
            for (int j = i + 1; j < n; j++) {
                if (main[j] > p) {
                    k = j;
                    p = main[j];
                }
            }
            if (k != i) {
                main[k] = main[i];
                main[i] = p;
                for (int j = 0; j < n; j++) {
                    p = z[j][i];
                    z[j][i] = z[j][k];
                    z[j][k] = p;
                }
            }
        }
        
        // Determine the largest eigenvalue in absolute terms
        maxAbsoluteValue = 0;
        for (int i = 0; i < n; i++) {
            if (Math.abs(main[i]) > maxAbsoluteValue) {
                maxAbsoluteValue = Math.abs(main[i]);
            }
        }
        
        // Make null any eigenvalue too small to be significant
        if (maxAbsoluteValue != 0.0) {
            for (int i = 0; i < n; i++) {
                if (Math.abs(main[i]) < Precision.getMachineEpsilon() * maxAbsoluteValue) {
                    main[i] = 0;
                }
            }
        }
    }



    /**
     * 隐式QR算法 - 改进版本，基于Apache Commons Math3的实现
     *
     * <p>
     * 隐式QR算法是现代特征分解的核心算法，通过直接操作矩阵元素来避免显式的矩阵乘法，
     * 从而显著提高数值稳定性和计算效率。这是LAPACK等专业数值库使用的标准算法。</p>
     *
     * <p>
     * 算法优势：</p>
     * <ul>
     * <li>数值稳定性：避免显式矩阵乘法减少舍入误差累积</li>
     * <li>计算效率：直接操作矩阵元素，减少中间计算</li>
     * <li>内存友好：不需要存储完整的Q和R矩阵</li>
     * <li>收敛快速：Wilkinson位移加速收敛</li>
     * </ul>
     *
     * <p>
     * 算法原理：</p>
     * <ol>
     * <li>对海森伯格矩阵H应用Wilkinson位移：H - σI</li>
     * <li>使用Givens旋转进行隐式QR分解</li>
     * <li>计算R*Q + σI得到新的H</li>
     * <li>重复直到收敛到上三角矩阵（特征值在对角线上）</li>
     * </ol>
     *
     * <p>
     * Wilkinson位移：选择右下角2x2子矩阵的特征值中更接近H[n-1][n-1]的那个， 
     * 这能显著加速收敛，特别是对于接近收敛的情况。</p>
     *
     * <p>
     * 时间复杂度：O(n³)，但收敛速度比标准QR算法快2-3倍</p>
     * <p>
     * 空间复杂度：O(n²)</p>
     *
     * @param H 海森伯格矩阵（输入）
     * @return 包含特征值和特征向量的元组
     */
    private Tuple2<IDoubleVector, IDoubleMatrix> implicitQRAlgorithm(double[][] H) {
        int n = H.length;
        double[][] eigenvectors = new double[n][n];
        double[][] current = new double[n][n];
        
        // 复制H到current
        for (int i = 0; i < n; i++) {
            System.arraycopy(H[i], 0, current[i], 0, n);
        }

        // 初始化特征向量矩阵为单位矩阵
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                eigenvectors[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }

        final int maxIterations = 30;         // 最大迭代次数
        final double epsilon = 1e-12;         // 收敛容差

        // QR迭代
        for (int iter = 0; iter < maxIterations; iter++) {
            // 查找小的次对角元素
            int l = findSmallSubDiagonalElement(current, n, epsilon);

            // 检查收敛
            if (l == n - 1) {
                // 已经收敛
                break;
            }

            // 如果是2x2块，直接计算特征值
            if (l == n - 2) {
                compute2x2Eigenvalues(current, n);
                break;
            }

            // 计算Wilkinson位移
            double shift = computeWilkinsonShiftForTridiagonal(current, n);

            // 应用QR步骤
            performQRStepForTridiagonal(current, eigenvectors, shift, l, n);

            // 清理舍入误差
            cleanRoundingErrors(current, n);
        }

        // 提取特征值（对角线元素）
        double[] eigenvalues = new double[n];
        for (int i = 0; i < n; i++) {
            eigenvalues[i] = current[i][i];
        }

        // 标准化特征向量
        normalizeEigenvectors(eigenvectors, n);

        // 按特征值大小降序排列
        sortEigenvaluesAndVectors(eigenvalues, eigenvectors);

        // 转置特征向量矩阵，使其变成列存储格式
        double[][] eigenvectorsTransposed = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                eigenvectorsTransposed[j][i] = eigenvectors[i][j];
            }
        }

        return new Tuple2<>(IDoubleVector.of(eigenvalues), (IDoubleMatrix) Linalg.matrix(eigenvectorsTransposed));
    }

    /**
     * 查找小的次对角元素 - 基于Apache Commons Math3的实现
     * 
     * @param matrix 矩阵
     * @param n 矩阵维度
     * @param epsilon 容差
     * @return 小次对角元素的索引
     */
    private int findSmallSubDiagonalElement(double[][] matrix, int n, double epsilon) {
        int l = n - 1;
        while (l > 0) {
            double s = Math.abs(matrix[l - 1][l - 1]) + Math.abs(matrix[l][l]);
            if (Precision.equalsZero(s, 1e-12)) {
                s = 1.0;
            }
            // 使用我们自定义的Precision类进行数值比较
            if (Precision.equalsZero(matrix[l][l - 1], epsilon * s)) {
                break;
            }
            l--;
        }
        return l;
    }

    /**
     * 计算2x2块的特征值 - 基于Apache Commons Math3的实现
     * 
     * @param matrix 矩阵
     * @param n 矩阵维度
     */
    private void compute2x2Eigenvalues(double[][] matrix, int n) {
        double p = (matrix[n - 2][n - 2] - matrix[n - 1][n - 1]) / 2.0;
        double q = p * p + matrix[n - 1][n - 2] * matrix[n - 2][n - 1];
        
        if (q >= 0) {
            double z = Math.sqrt(Math.abs(q));
            if (p >= 0) {
                z = p + z;
            } else {
                z = p - z;
            }
            matrix[n - 1][n - 1] += z;
            matrix[n - 2][n - 2] -= z;
        }
    }

    /**
     * 计算Wilkinson位移 - 基于Apache Commons Math3的实现
     * 
     * @param matrix 矩阵
     * @param n 矩阵维度
     * @return Wilkinson位移值
     */
    private double computeWilkinsonShiftForTridiagonal(double[][] matrix, int n) {
        double a = matrix[n - 2][n - 2];
        double b = matrix[n - 2][n - 1];
        double d = matrix[n - 1][n - 1];

        double trace = a + d;
        double det = a * d - b * b;
        double discriminant = trace * trace - 4 * det;

        if (discriminant >= 0) {
            double sqrtDisc = Math.sqrt(discriminant);
            double lambda1 = (trace + sqrtDisc) / 2.0;
            double lambda2 = (trace - sqrtDisc) / 2.0;

            double diff1 = Math.abs(lambda1 - d);
            double diff2 = Math.abs(lambda2 - d);

            if (Precision.equals(diff1, diff2, 1e-6)) {
                return Math.min(lambda1, lambda2);
            } else {
                return (diff1 < diff2) ? lambda1 : lambda2;
            }
        } else {
            return d;
        }
    }

    /**
     * 对三对角矩阵执行QR步骤 - 基于Apache Commons Math3的实现
     * 
     * @param matrix 矩阵
     * @param eigenvectors 特征向量矩阵
     * @param shift 位移值
     * @param l 起始索引
     * @param n 矩阵维度
     */
    private void performQRStepForTridiagonal(double[][] matrix, double[][] eigenvectors, double shift, int l, int n) {
        double p = matrix[l][l] - shift;
        double q = matrix[l + 1][l];

        for (int k = l; k < n - 1; k++) {
            double r = Math.sqrt(p * p + q * q);
            if (Precision.equalsZero(r, 1e-12)) {
                continue;
            }
            
            double c = p / r;
            double s = -q / r;

            if (k > l) {
                matrix[k][k - 1] = -s * matrix[k][k - 1];
            }

            // 行修改
            for (int j = k; j < n; j++) {
                double temp1 = matrix[k][j];
                double temp2 = matrix[k + 1][j];
                matrix[k][j] = c * temp1 - s * temp2;
                matrix[k + 1][j] = s * temp1 + c * temp2;
            }

            // 列修改
            for (int i = 0; i <= Math.min(k + 2, n - 1); i++) {
                double temp1 = matrix[i][k];
                double temp2 = matrix[i][k + 1];
                matrix[i][k] = c * temp1 - s * temp2;
                matrix[i][k + 1] = s * temp1 + c * temp2;
            }

            // 更新特征向量
            for (int i = 0; i < n; i++) {
                double temp1 = eigenvectors[i][k];
                double temp2 = eigenvectors[i][k + 1];
                eigenvectors[i][k] = c * temp1 - s * temp2;
                eigenvectors[i][k + 1] = s * temp1 + c * temp2;
            }

            if (k < n - 2) {
                p = matrix[k + 1][k];
                q = matrix[k + 2][k];
            }
        }
    }

    /**
     * 清理舍入误差 - 基于Apache Commons Math3的实现
     * 
     * @param matrix 矩阵
     * @param n 矩阵维度
     */
    private void cleanRoundingErrors(double[][] matrix, int n) {
        for (int i = 2; i < n; i++) {
            if (Precision.equalsZero(matrix[i][i - 2], 1e-12)) {
                matrix[i][i - 2] = 0.0;
            }
            if (i > 2 && Precision.equalsZero(matrix[i][i - 3], 1e-12)) {
                matrix[i][i - 3] = 0.0;
            }
        }
    }

    /**
     * 标准化特征向量 - 基于Apache Commons Math3的实现
     * 
     * @param eigenvectors 特征向量矩阵
     * @param n 矩阵维度
     */
    private void normalizeEigenvectors(double[][] eigenvectors, int n) {
        for (int j = 0; j < n; j++) {
            double norm = 0.0;
            for (int i = 0; i < n; i++) {
                norm += eigenvectors[i][j] * eigenvectors[i][j];
            }
            norm = Math.sqrt(norm);

            if (!Precision.equalsZero(norm, 1e-10)) {
                for (int i = 0; i < n; i++) {
                    eigenvectors[i][j] /= norm;
                }
            } else {
                // 如果向量线性相关，设置为单位向量
                for (int i = 0; i < n; i++) {
                    eigenvectors[i][j] = (i == j) ? 1.0 : 0.0;
                }
            }
        }
    }

    /**
     * 特征值和特征向量排序（Eigenvalue and Eigenvector Sorting）
     *
     * <p>
     * 在特征分解完成后，需要将特征值和对应的特征向量按特征值大小进行排序。 这确保了特征分解结果的一致性和可预测性，便于后续的数值计算和分析。</p>
     *
     * <p>
     * 排序策略：</p>
     * <ul>
     * <li><strong>降序排列</strong>：按特征值大小从大到小排序</li>
     * <li><strong>保持对应关系</strong>：特征向量与特征值保持一一对应</li>
     * <li><strong>稳定性</strong>：使用稳定的排序算法保持相对顺序</li>
     * </ul>
     *
     * <p>
     * 算法选择：</p>
     * <ul>
     * <li>使用冒泡排序，虽然时间复杂度为O(n²)，但对于特征分解中的小矩阵足够高效</li>
     * <li>冒泡排序是稳定排序，保持相等元素的相对顺序</li>
     * <li>实现简单，易于理解和维护</li>
     * </ul>
     *
     * @param eigenvalues 特征值数组，将被就地排序
     * @param eigenvectors 特征向量矩阵（按行存储），将与特征值同步排序
     */
    private void sortEigenvaluesAndVectors(double[] eigenvalues, double[][] eigenvectors) {
        int n = eigenvalues.length;  // 特征值个数

        // 使用冒泡排序算法进行降序排列
        // 冒泡排序虽然时间复杂度较高，但对于小矩阵足够高效且稳定
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - 1 - i; j++) {
                // 比较相邻的特征值，如果前一个小于后一个则交换
                if (eigenvalues[j] < eigenvalues[j + 1]) {
                    // 交换特征值
                    double tempVal = eigenvalues[j];
                    eigenvalues[j] = eigenvalues[j + 1];
                    eigenvalues[j + 1] = tempVal;

                    // 交换对应的特征向量（行）
                    // 保持特征值和特征向量的对应关系
                    double[] tempVec = eigenvectors[j];
                    eigenvectors[j] = eigenvectors[j + 1];
                    eigenvectors[j + 1] = tempVec;
                }
            }
        }
    }

    /**
     * 计算Wilkinson位移 - 基于Apache Commons Math3的实现
     * 
     * @param matrix 矩阵
     * @param n 矩阵维度
     * @return Wilkinson位移值
     */
    private double computeShift(double[][] matrix, int n) {
        if (n < 2) {
            return 0.0;
        }

        // 提取右下角2×2子矩阵的元素
        double a = matrix[n - 2][n - 2];  // 主对角线元素
        double b = matrix[n - 2][n - 1];  // 次对角线元素
        double d = matrix[n - 1][n - 1];  // 右下角元素

        // 计算2×2子矩阵的特征值
        double trace = a + d;                    // 迹（trace）
        double det = a * d - b * b;             // 行列式（determinant）
        double discriminant = trace * trace - 4 * det;  // 判别式

        if (discriminant >= 0) {
            // 实数特征值情况：使用二次公式
            double sqrtDisc = Math.sqrt(discriminant);
            double lambda1 = (trace + sqrtDisc) / 2.0;  // 较大特征值
            double lambda2 = (trace - sqrtDisc) / 2.0;  // 较小特征值

            // 选择更接近d的特征值，提高收敛速度
            double diff1 = Math.abs(lambda1 - d);
            double diff2 = Math.abs(lambda2 - d);

            // 数值稳定性考虑：如果两个特征值都很接近，选择较小的那个
            if (Precision.equals(diff1, diff2, 1e-6)) {
                return Math.min(lambda1, lambda2);
            } else {
                return (diff1 < diff2) ? lambda1 : lambda2;
            }
        } else {
            // 复数特征值情况：使用Rayleigh商位移
            // 当判别式为负时，选择右下角元素作为位移
            return d;
        }
    }

    /**
     * 执行隐式QR步骤 - 基于Apache Commons Math3的实现
     * 
     * @param matrix 矩阵
     * @param eigenvectors 特征向量矩阵
     * @param shift 位移值
     * @param n 矩阵维度
     */
    private void performImplicitQRStep(double[][] matrix, double[][] eigenvectors, double shift, int n) {
        // 对每一列应用Givens旋转
        for (int k = 0; k < n - 1; k++) {
            // 计算Givens旋转参数
            // 目标：将matrix[k+1][k]位置清零，实现QR分解的效果
            double a = matrix[k][k] - shift;    // 减去位移后的主对角线元素
            double b = matrix[k + 1][k];        // 次对角线元素
            
            // 使用我们自定义的Precision类进行数值比较
            if (Precision.equalsZero(b, 1e-12)) {
                continue;
            }
            
            double r = Math.sqrt(a * a + b * b);  // 旋转半径

            if (!Precision.equalsZero(r, 1e-12)) {  // 避免除零错误，确保数值稳定性
                double c = a / r;      // 余弦值（cosine）
                double s = -b / r;     // 正弦值（sine），注意符号

                // 应用Givens旋转到矩阵A
                // 旋转矩阵G作用于行k和k+1，列k到n-1
                // 这相当于执行QR分解中的Q^T * A操作
                applyGivensRotation(matrix, c, s, k, k + 1, k, n - 1);

                // 应用Givens旋转到特征向量矩阵
                // 注意：由于矩阵按行存储，需要调整更新逻辑
                // 这里更新的是特征向量矩阵的列，而不是行
                for (int i = 0; i < n; i++) {
                    double temp = eigenvectors[i][k];
                    eigenvectors[i][k] = c * temp + s * eigenvectors[i][k + 1];
                    eigenvectors[i][k + 1] = -s * temp + c * eigenvectors[i][k + 1];
                }
            }
        }
    }

    /**
     * 计算矩阵范数 - 基于Apache Commons Math3的实现
     * 
     * @param matrix 矩阵
     * @param n 矩阵维度
     * @return 矩阵范数
     */
    private double computeMatrixNorm(double[][] matrix, int n) {
        double norm = 0.0;
        // 计算所有非对角线元素的绝对值之和
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    norm += Math.abs(matrix[i][j]);
                }
            }
        }
        return norm;
    }

    /**
     * 应用Givens旋转到矩阵（Apply Givens Rotation to Matrix）
     *
     * <p>
     * Givens旋转是数值线性代数中的基本变换，用于在矩阵的特定位置引入零元素。
     * 在QR分解和特征分解中，Givens旋转是实现矩阵变换的核心工具。</p>
     *
     * <p>
     * Givens旋转矩阵：</p>
     * <pre>
     * G = [c  -s] 其中 c² + s² = 1
     *     [s   c]
     * </pre>
     *
     * <p>
     * 算法原理：</p>
     * <ol>
     * <li>计算旋转参数c和s，使得s*a + c*b = 0</li>
     * <li>应用旋转矩阵到指定行，消除目标元素</li>
     * <li>保持矩阵的正交性和数值稳定性</li>
     * </ol>
     *
     * <p>
     * 应用场景：</p>
     * <ul>
     * <li>QR分解中的矩阵三角化</li>
     * <li>特征分解中的矩阵简化</li>
     * <li>数值线性代数中的基础变换</li>
     * </ul>
     *
     * @param matrix 矩阵
     * @param c 余弦值
     * @param s 正弦值
     * @param row1 第一行索引
     * @param row2 第二行索引
     * @param colStart 起始列索引
     * @param colEnd 结束列索引
     */
    private void applyGivensRotation(double[][] matrix, double c, double s, int row1, int row2, int colStart, int colEnd) {
        // 应用Givens旋转到指定行范围
        for (int j = colStart; j <= colEnd; j++) {
            double temp1 = matrix[row1][j];
            double temp2 = matrix[row2][j];
            matrix[row1][j] = c * temp1 - s * temp2;
            matrix[row2][j] = s * temp1 + c * temp2;
        }
    }

    /**
     * 分块海森伯格化简 - 将矩阵转换为海森伯格形式
     *
     * <p>
     * 海森伯格化简是特征分解的关键预处理步骤，将任意矩阵A通过相似变换转换为海森伯格矩阵H， 使得A = Q * H *
     * Q^T，其中Q是正交矩阵，H是海森伯格矩阵（上三角矩阵加上一条次对角线）。</p>
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
    private Tuple2<double[][], double[][]> blockedHessenbergReduction(IDoubleMatrix matrix) {
        // 使用Hessenberg分解实现类
        IHessenbergDecomposition hessenbergDecomposition = Decomps.createHessenberg();
        Tuple2<IMatrix<Double>, IMatrix<Double>> result = hessenbergDecomposition.decompose(matrix);
        
        // 转换为所需的double[][]格式
        IDoubleMatrix hessenbergMatrix = (IDoubleMatrix) result._1;
        IDoubleMatrix qMatrix = (IDoubleMatrix) result._2;
        
        return new Tuple2<>(hessenbergMatrix.getData(), qMatrix.getData());
    }

    /**
     * 海森伯格化简 - 将矩阵转换为海森伯格形式 海森伯格矩阵只有主对角线和次对角线非零，大大减少后续QR算法的计算量
     *
     * @return 海森伯格矩阵和变换矩阵
     */
    private Tuple2<double[][], double[][]> hessenbergReduction(IDoubleMatrix matrix) {
        // 使用Hessenberg分解实现类
        IHessenbergDecomposition hessenbergDecomposition = Decomps.createHessenberg();
        Tuple2<IMatrix<Double>, IMatrix<Double>> result = hessenbergDecomposition.decompose(matrix);
        
        // 转换为所需的double[][]格式
        IDoubleMatrix hessenbergMatrix = (IDoubleMatrix) result._1;
        IDoubleMatrix qMatrix = (IDoubleMatrix) result._2;
        
        return new Tuple2<>(hessenbergMatrix.getData(), qMatrix.getData());
    }
    
    /**
     * 海森伯格矩阵的QR算法
     * 
     * @param H 海森伯格矩阵
     * @return 特征值和特征向量
     */
    private Tuple2<IDoubleVector, IDoubleMatrix> qrAlgorithmForHessenberg(double[][] H) {
        int n = H.length;
        double[][] eigenvectors = new double[n][n];
        
        // 初始化特征向量矩阵为单位矩阵
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                eigenvectors[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }
        
        final int maxIterations = 30;
        final double epsilon = 1e-12;
        
        // QR迭代
        for (int iter = 0; iter < maxIterations; iter++) {
            // 查找小的次对角元素
            int l = findSmallSubDiagonalElement(H, n, epsilon);
            
            // 检查收敛
            if (l == n - 1) {
                break;
            }
            
            // 如果是2x2块，直接计算特征值
            if (l == n - 2) {
                compute2x2Eigenvalues(H, n);
                break;
            }
            
            // QR分解
            Tuple2<double[][], double[][]> qr = qrDecompositionForHessenberg(H);
            double[][] Q = qr._1;
            double[][] R = qr._2;
            
            // 重新组合：H = R * Q
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    H[i][j] = 0.0;
                    for (int k = 0; k < n; k++) {
                        H[i][j] += R[i][k] * Q[k][j];
                    }
                }
            }
            
            // 更新特征向量：V = V * Q
            double[][] temp = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    temp[i][j] = 0.0;
                    for (int k = 0; k < n; k++) {
                        temp[i][j] += eigenvectors[i][k] * Q[k][j];
                    }
                }
            }
            for (int i = 0; i < n; i++) {
                System.arraycopy(temp[i], 0, eigenvectors[i], 0, n);
            }
        }
        
        // 提取特征值
        double[] eigenvalues = new double[n];
        for (int i = 0; i < n; i++) {
            eigenvalues[i] = H[i][i];
        }
        
        return new Tuple2<>(IDoubleVector.of(eigenvalues), (IDoubleMatrix) Linalg.matrix(eigenvectors));
    }
    
    /**
     * 海森伯格矩阵的QR分解
     * 
     * @param matrix 海森伯格矩阵
     * @return Q和R矩阵
     */
    private Tuple2<double[][], double[][]> qrDecompositionForHessenberg(double[][] matrix) {
        int n = matrix.length;
        double[][] Q = new double[n][n];
        double[][] R = new double[n][n];
        
        // 初始化Q为单位矩阵
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Q[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }
        
        // 复制matrix到R
        for (int i = 0; i < n; i++) {
            System.arraycopy(matrix[i], 0, R[i], 0, n);
        }
        
        // Givens旋转
        for (int j = 0; j < n - 1; j++) {
            for (int i = j + 1; i < Math.min(j + 3, n); i++) {
                if (!Precision.equalsZero(R[i][j], 1e-12)) {
                    // 计算Givens旋转参数
                    double r = Math.sqrt(R[j][j] * R[j][j] + R[i][j] * R[i][j]);
                    double c = R[j][j] / r;
                    double s = -R[i][j] / r;
                    
                    // 应用Givens旋转到R
                    for (int k = j; k < n; k++) {
                        double temp = c * R[j][k] - s * R[i][k];
                        R[i][k] = s * R[j][k] + c * R[i][k];
                        R[j][k] = temp;
                    }
                    
                    // 应用Givens旋转到Q
                    for (int k = 0; k < n; k++) {
                        double temp = c * Q[j][k] - s * Q[i][k];
                        Q[i][k] = s * Q[j][k] + c * Q[i][k];
                        Q[j][k] = temp;
                    }
                }
            }
        }
        
        return new Tuple2<>(Q, R);
    }
    



    
    /**
     * 更新Q矩阵
     */
    private void updateQMatrix(IDoubleMatrix Q, IVector<Double> v, int k) {
        int n = Q.getRowNum();
        int vLen = v.length();
        
        // 应用Householder变换: Q = Q * (I - 2vv^T)
        for (int i = 0; i < n; i++) {
            double dot = 0.0;
            for (int j = 0; j < vLen; j++) {
                dot += Q.get(i, k + 1 + j) * v.get(j);
            }
            
            for (int j = 0; j < vLen; j++) {
                Q.set(i, k + 1 + j, Q.get(i, k + 1 + j) - 2.0 * dot * v.get(j));
            }
        }
    }


    

    
    /**
     * 标准化特征向量 (IMatrix版本)
     */
    private void normalizeEigenvectors(IMatrix<Double> eigenvectors) {
        int n = eigenvectors.getColNum();
        
        for (int j = 0; j < n; j++) {
            // 计算第j列的范数
            double norm = 0.0;
            for (int i = 0; i < eigenvectors.getRowNum(); i++) {
                double val = eigenvectors.get(i, j);
                norm += val * val;
            }
            norm = Math.sqrt(norm);
            
            // 标准化第j列
            if (norm > epsilon) {
                for (int i = 0; i < eigenvectors.getRowNum(); i++) {
                    eigenvectors.set(i, j, eigenvectors.get(i, j) / norm);
                }
            }
        }
    }
    
    /**
     * 排序特征值和特征向量（按特征值降序）(IMatrix/IVector版本)
     */
    private void sortEigenvaluesAndVectors(IVector<Double> eigenvalues, IMatrix<Double> eigenvectors) {
        int n = eigenvalues.length();
        
        // 创建索引数组
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }
        
        // 按特征值降序排序索引
        java.util.Arrays.sort(indices, (i, j) -> Double.compare(eigenvalues.get(j), eigenvalues.get(i)));
        
        // 创建临时数组存储排序后的结果
        double[] sortedEigenvalues = new double[n];
        double[][] sortedEigenvectors = new double[eigenvectors.getRowNum()][n];
        
        for (int i = 0; i < n; i++) {
            int originalIndex = indices[i];
            sortedEigenvalues[i] = eigenvalues.get(originalIndex);
            
            for (int j = 0; j < eigenvectors.getRowNum(); j++) {
                sortedEigenvectors[j][i] = eigenvectors.get(j, originalIndex);
            }
        }
        
        // 将排序后的结果写回原数组
        for (int i = 0; i < n; i++) {
            eigenvalues.set(i, sortedEigenvalues[i]);
            for (int j = 0; j < eigenvectors.getRowNum(); j++) {
                eigenvectors.set(j, i, sortedEigenvectors[j][i]);
            }
        }
    }
}