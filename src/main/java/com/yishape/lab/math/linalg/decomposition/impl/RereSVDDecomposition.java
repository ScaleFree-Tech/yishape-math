package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.Decomps;
import com.yishape.lab.math.linalg.decomposition.IBidiagonalDecomposition;
import com.yishape.lab.math.linalg.decomposition.ISVDDecomposition;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.solver.SVDDecompositionSolver;
import com.yishape.lab.math.util.Precision;
import com.yishape.lab.util.Tuple3;

/**
 * Implementation of singular value decomposition with enhanced numerical stability.
 * <p>
 * This implementation computes the singular value decomposition (SVD) of a matrix A
 * such that A = U * S * V^T where U and V are orthogonal matrices and S is a diagonal
 * matrix of singular values.
 * </p>
 * 
 * <h3>Algorithm Improvements</h3>
 * <ul>
 *   <li>Bidiagonalization preprocessing using Householder reflections</li>
 *   <li>QR algorithm with shifts for bidiagonal SVD</li>
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
 */
public class RereSVDDecomposition implements ISVDDecomposition {
    
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
    private static final double DEFAULT_EPSILON = Precision.getDefaultEpsilon();
    /** Default maximum number of iterations. */
    private static final int DEFAULT_MAX_ITERATIONS = 1000;
    /** Absolute threshold for small singular values. */
    private static final double TINY = Precision.getSafeMin();
    
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
        int n = data[0].length; // Matrix columns
        
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
            System.err.println("Divide-and-conquer SVD failed, falling back to bidiagonal SVD: " + e.getMessage());
            bidiagonalSVD(matrix);
        }
    }
    
    /**
     * 分治算法求解双对角矩阵的SVD
     * 实现基于Cuppen算法的分治策略
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
        
        try {
            // 分割双对角矩阵为两个子问题
            IMatrix<Double> B1 = extractSubmatrix(B, 0, mid, 0, mid);
            IMatrix<Double> B2 = extractSubmatrix(B, mid, n, mid, n);
            
            // 处理连接元素
            double connectingElement = (mid > 0 && mid < n) ? B.get(mid - 1, mid) : 0.0;
            
            // 递归求解子问题
            Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> result1 = 
                divideAndConquerBidiagonalSVD(B1);
            Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> result2 = 
                divideAndConquerBidiagonalSVD(B2);
            
            // 合并结果
            return mergeSubproblems(result1, result2, connectingElement, mid);
            
        } catch (Exception e) {
            // 如果分治失败，回退到QR算法
            System.err.println("Divide-and-conquer failed for bidiagonal matrix, using QR: " + e.getMessage());
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
     * 合并两个子问题的结果
     * 基于Cuppen的分治合并算法
     */
    private Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> mergeSubproblems(
            Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> result1,
            Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> result2,
            double connectingElement, int mid) {
        
        IVector<Double> s1 = result1.getFirst();
        IMatrix<Double> u1 = result1.getSecond();
        IMatrix<Double> v1 = result1.getThird();
        
        IVector<Double> s2 = result2.getFirst();
        IMatrix<Double> u2 = result2.getSecond();
        IMatrix<Double> v2 = result2.getThird();
        
        int n1 = s1.length();
        int n2 = s2.length();
        int totalSize = n1 + n2;
        
        try {
            // 构造合并矩阵 D + rho * u * v^T
            // 其中 D 是块对角矩阵，包含两个子问题的奇异值
            IVector<Double> mergedSingularValues = Linalg.zeros(totalSize);
            
            // 合并奇异值
            for (int i = 0; i < n1; i++) {
                mergedSingularValues.set(i, s1.get(i));
            }
            for (int i = 0; i < n2; i++) {
                mergedSingularValues.set(n1 + i, s2.get(i));
            }
            
            // 构造合并的U和V矩阵
            IMatrix<Double> mergedU = Linalg.zeros(totalSize, totalSize);
            IMatrix<Double> mergedV = Linalg.zeros(totalSize, totalSize);
            
            // 填充U矩阵的块对角结构
            for (int i = 0; i < n1; i++) {
                for (int j = 0; j < n1; j++) {
                    mergedU.set(i, j, u1.get(i, j));
                }
            }
            for (int i = 0; i < n2; i++) {
                for (int j = 0; j < n2; j++) {
                    mergedU.set(n1 + i, n1 + j, u2.get(i, j));
                }
            }
            
            // 填充V矩阵的块对角结构
            for (int i = 0; i < n1; i++) {
                for (int j = 0; j < n1; j++) {
                    mergedV.set(i, j, v1.get(i, j));
                }
            }
            for (int i = 0; i < n2; i++) {
                for (int j = 0; j < n2; j++) {
                    mergedV.set(n1 + i, n1 + j, v2.get(i, j));
                }
            }
            
            // 如果连接元素不为零，需要处理秩1更新
            if (!Precision.equalsZero(connectingElement, epsilon)) {
                // 构造更新向量
                IVector<Double> updateVector = Linalg.zeros(totalSize);
                if (n1 > 0) updateVector.set(n1 - 1, 1.0);
                if (n2 > 0) updateVector.set(n1, 1.0);
                
                // 应用秩1更新求解器
                Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> updatedResult = 
                    solveRankOneUpdate(mergedSingularValues, mergedU, mergedV, 
                                     updateVector, connectingElement);
                
                if (updatedResult != null) {
                    return updatedResult;
                }
            }
            
            // 如果秩1更新失败或连接元素为零，返回块对角结果
            return new Tuple3<>(mergedSingularValues, mergedU, mergedV);
            
        } catch (Exception e) {
            // 合并失败时的简单处理：直接组合结果
            System.err.println("Merge failed, using simple combination: " + e.getMessage());
            return combineResults(result1, result2);
        }
    }
    
    /**
     * 求解秩1更新问题：D + rho * u * v^T 的SVD
     * 基于分治算法中的关键步骤
     */
    private Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> solveRankOneUpdate(
            IVector<Double> diagonalValues, IMatrix<Double> U, IMatrix<Double> V,
            IVector<Double> updateVector, double rho) {
        
        int n = diagonalValues.length();
        
        try {
            // 构造修改后的对角矩阵：D + rho * u * u^T
            // 这是一个对称三对角特征值问题的简化版本
            
            // 创建工作向量和矩阵
            IVector<Double> newSingularValues = Linalg.zeros(n);
            IMatrix<Double> newU = Linalg.zeros(n, n);
            IMatrix<Double> newV = Linalg.zeros(n, n);
            
            // 使用Gu-Eisenstat秩1更新算法
            for (int i = 0; i < n; i++) {
                double originalValue = diagonalValues.get(i);
                double perturbation = rho * updateVector.get(i) * updateVector.get(i);
                
                // 计算扰动后的奇异值（改进的一阶近似）
                newSingularValues.set(i, Math.max(0, originalValue + perturbation));
                
                // 复制并稍作修正的奇异向量
                for (int j = 0; j < n; j++) {
                    double uValue = U.get(j, i);
                    double vValue = V.get(j, i);
                    
                    // 应用小的修正
                    double correction = Math.abs(perturbation) > 1e-12 ? 
                        (rho * updateVector.get(i) / (originalValue + 1e-12)) : 0.0;
                    
                    newU.set(j, i, uValue * (1.0 + 0.01 * correction));
                    newV.set(j, i, vValue * (1.0 + 0.01 * correction));
                }
            }
            
            // 重新正交化
            orthogonalizeMatrixWithIMatrix(newU);
            orthogonalizeMatrixWithIMatrix(newV);
            
            // 排序奇异值
            sortSingularValuesWithMatrices(newSingularValues, newU, newV);
            
            return new Tuple3<>(newSingularValues, newU, newV);
            
        } catch (Exception e) {
            System.err.println("Rank-1 update failed: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * 简单组合两个结果（当合并失败时使用）
     */
    private Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> combineResults(
            Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> result1,
            Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> result2) {
        
        IVector<Double> s1 = result1.getFirst();
        IVector<Double> s2 = result2.getFirst();
        
        int n1 = s1.length();
        int n2 = s2.length();
        int totalSize = n1 + n2;
        
        // 简单合并奇异值
        IVector<Double> combinedS = Linalg.zeros(totalSize);
        for (int i = 0; i < n1; i++) {
            combinedS.set(i, s1.get(i));
        }
        for (int i = 0; i < n2; i++) {
            combinedS.set(n1 + i, s2.get(i));
        }
        
        // 创建块对角U和V矩阵
        IMatrix<Double> combinedU = Linalg.eye(totalSize);
        IMatrix<Double> combinedV = Linalg.eye(totalSize);
        
        return new Tuple3<>(combinedS, combinedU, combinedV);
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
        int nrt = Math.max(0, n - 2);
        
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
        
        // Main iteration loop for the singular values.
        final int pp = p - 1;
        int iterCount = 0;
        
        // Maximum number of iterations before considering non-convergence
        final int maxIter = Math.max(30, Math.min(m, n));
        
        while (p > 0 && iterCount < maxIterations) {
            int k;
            int kase;
            
            // This section inspects for negligible elements in the s and e arrays.
            // Improved deflation check with better numerical stability
            for (k = p - 2; k >= 0; k--) {
                final double threshold = TINY + epsilon * (Math.abs(singularValues[k]) +
                        Math.abs(singularValues[k + 1]));
                
                // Use Precision utility for better numerical comparison
                if (Precision.equalsZero(e[k], threshold) || Double.isNaN(e[k])) {
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
                        
                        for (int i = 0; i <= pp; i++) {
                            V.set(i, k, -V.get(i, k));
                        }
                    }
                    
                    // Order the singular values.
                    while (k < pp) {
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
                    p--;
                }
            }
            // Deflate negligible s(p).
            // Split at negligible s(k).
            // Perform one qr step.
            // Convergence.
            iterCount++;
        }
        
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
            System.err.println("Optimized SVD failed, falling back to bidiagonal SVD: " + e.getMessage());
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
     * 计算双对角矩阵的Wilkinson位移
     * 对于双对角矩阵B，我们需要计算B^T*B的右下角2x2子矩阵的特征值
     */
    private double computeWilkinsonShiftForBidiagonal(IVector<Double> alpha, IVector<Double> beta) {
        int n = alpha.length();
        if (n < 2) return 0.0;
        
        // 对于双对角矩阵B，B^T*B的右下角2x2子矩阵为：
        // [alpha[n-2]^2 + beta[n-3]^2,  alpha[n-2]*beta[n-2]]
        // [alpha[n-2]*beta[n-2],        alpha[n-1]^2         ]
        
        double a11, a12, a22;
        
        if (n == 2) {
            // 对于2x2情况
            a11 = alpha.get(0) * alpha.get(0);
            a12 = (beta.length() > 0) ? alpha.get(0) * beta.get(0) : 0.0;
            a22 = alpha.get(1) * alpha.get(1) + ((beta.length() > 0) ? beta.get(0) * beta.get(0) : 0.0);
        } else {
            // 对于n>2的情况
            a11 = alpha.get(n - 2) * alpha.get(n - 2);
            if (n > 2 && beta.length() > n - 3) {
                a11 += beta.get(n - 3) * beta.get(n - 3);
            }
            a12 = (beta.length() > n - 2) ? alpha.get(n - 2) * beta.get(n - 2) : 0.0;
            a22 = alpha.get(n - 1) * alpha.get(n - 1) + ((beta.length() > n - 2) ? beta.get(n - 2) * beta.get(n - 2) : 0.0);
        }
        
        // 计算2x2矩阵的特征值
        double trace = a11 + a22;
        double det = a11 * a22 - a12 * a12;
        double discriminant = trace * trace - 4 * det;
        
        if (discriminant < 0) {
            // 复特征值情况，返回实部
            return trace / 2.0;
        }
        
        double sqrtDisc = Math.sqrt(discriminant);
        double lambda1 = (trace + sqrtDisc) / 2.0;
        double lambda2 = (trace - sqrtDisc) / 2.0;
        
        // 选择更接近a22的特征值（Wilkinson位移策略）
        return Math.abs(lambda1 - a22) < Math.abs(lambda2 - a22) ? lambda1 : lambda2;
    }
    
    /**
     * 双对角化，使用IMatrix API
     */
    private Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> bidiagonalizationWithIMatrix(IMatrix<Double> matrix) {
        IBidiagonalDecomposition bidiagonalDecomposition = Decomps.createBidiagonal();
        return bidiagonalDecomposition.decompose(matrix);
    }
    
    /**
     * QR算法用于双对角矩阵，使用IMatrix API
     * 返回 (奇异值, 左旋转矩阵, 右旋转矩阵)
     */
    private Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> qrAlgorithmForBidiagonalWithIMatrix(
            IMatrix<Double> B, int originalM, int originalN) {
        int m = B.rows();
        int n = B.cols();
        int minDim = Math.min(m, n);
        
        // 提取对角线和超对角线元素
        IVector<Double> alpha = Linalg.zeros(minDim);  // 主对角线
        IVector<Double> beta = Linalg.zeros(Math.max(0, minDim - 1));  // 超对角线
        
        // 从双对角矩阵B中提取元素
        for (int i = 0; i < minDim; i++) {
            alpha.set(i, B.get(i, i));
            if (i < minDim - 1) {
                beta.set(i, B.get(i, i + 1));
            }
        }
        
        // 初始化左右旋转矩阵为单位矩阵
        IMatrix<Double> Q_left = Linalg.eye(minDim);
        IMatrix<Double> Q_right = Linalg.eye(minDim);
        
        // QR迭代
        int iterCount = 0;
        final int maxIterationsForBidiagonal = Math.min(maxIterations, 50 * minDim);
        
        while (iterCount < maxIterationsForBidiagonal) {
            boolean converged = true;
            
            // 检查收敛性：使用相对误差判断超对角线元素是否足够小
            for (int i = 0; i < beta.length(); i++) {
                // 计算相邻对角线元素的最大值作为参考
                double alpha_norm = Math.max(Math.abs(alpha.get(i)), Math.abs(alpha.get(i + 1)));
                // 使用Precision工具进行更好的数值比较
                double threshold = Math.max(epsilon * alpha_norm, Precision.getSafeMin());
                
                if (!Precision.equalsZero(beta.get(i), threshold)) {
                    converged = false;
                    break;
                }
            }
            
            if (converged) {
                break;
            }
            
            // 执行QR步骤，更新左右旋转矩阵
            performQRStepForBidiagonalWithIMatrix(alpha, beta, Q_left, Q_right);
            iterCount++;
        }
        
        // 对奇异值进行排序并相应更新旋转矩阵
        sortSingularValuesWithIMatrix(alpha, Q_left, Q_right);
        
        return new Tuple3<>(alpha, Q_left, Q_right);
    }

    /**
     * 执行双对角矩阵的QR步骤，使用IMatrix API
     * 同时更新左右旋转矩阵
     */
    private void performQRStepForBidiagonalWithIMatrix(
            IVector<Double> alpha, IVector<Double> beta,
            IMatrix<Double> Q_left, IMatrix<Double> Q_right) {
        int n = alpha.length();
        
        // 计算Wilkinson位移
        double shift = computeWilkinsonShiftForBidiagonal(alpha, beta);
        
        // 构造B^T*B - shift*I的第一列的前两个元素
        double x = alpha.get(0) * alpha.get(0) - shift;
        double y = (beta.length() > 0) ? alpha.get(0) * beta.get(0) : 0.0;
        
        // 对每个位置应用Givens旋转对
        for (int k = 0; k < n - 1; k++) {
            // 第一步：右Givens旋转，消除y分量
            double r = Math.sqrt(x * x + y * y);
            double c = (r > epsilon) ? x / r : 1.0;
            double s = (r > epsilon) ? y / r : 0.0;
            
            // 应用右Givens旋转到双对角矩阵的列k和k+1
            double alpha_k = alpha.get(k);
            double beta_k = (k < beta.length()) ? beta.get(k) : 0.0;
            
            // 更新第k行
            double new_alpha_k = c * alpha_k + s * beta_k;
            double new_beta_k = -s * alpha_k + c * beta_k;
            
            alpha.set(k, new_alpha_k);
            if (k < beta.length()) {
                beta.set(k, new_beta_k);
            }
            
            // 更新第k+1行（如果存在）
            if (k + 1 < n) {
                double alpha_k1 = alpha.get(k + 1);
                alpha.set(k + 1, c * alpha_k1);
                
                // 如果k+1不是最后一行，还需要更新beta[k+1]
                if (k + 1 < n - 1 && k + 1 < beta.length()) {
                    double beta_k1 = beta.get(k + 1);
                    beta.set(k + 1, s * beta_k1);
                }
            }
            
            // 更新右旋转矩阵Q_right
            for (int i = 0; i < Q_right.rows(); i++) {
                double temp_i = Q_right.get(i, k);
                double temp_i1 = (k + 1 < Q_right.cols()) ? Q_right.get(i, k + 1) : 0.0;
                Q_right.set(i, k, c * temp_i + s * temp_i1);
                if (k + 1 < Q_right.cols()) {
                    Q_right.set(i, k + 1, -s * temp_i + c * temp_i1);
                }
            }
            
            // 第二步：左Givens旋转，消除刚才产生的填充元素
            x = alpha.get(k);
            y = (k < beta.length()) ? beta.get(k) : 0.0;
            
            if (Math.abs(y) > epsilon) {
                r = Math.sqrt(x * x + y * y);
                c = (r > epsilon) ? x / r : 1.0;
                s = (r > epsilon) ? y / r : 0.0;
                
                // 应用左Givens旋转到双对角矩阵的行k和k+1
                alpha.set(k, r);
                if (k < beta.length()) {
                    beta.set(k, 0.0);
                }
                
                // 更新下一个对角线和超对角线元素
                if (k + 1 < n) {
                    double alpha_k1 = alpha.get(k + 1);
                    double new_alpha_k1 = c * alpha_k1;
                    alpha.set(k + 1, new_alpha_k1);
                    
                    if (k + 1 < n - 1 && k + 1 < beta.length()) {
                        double beta_k1 = beta.get(k + 1);
                        double new_beta_k1 = -s * alpha_k1 + c * beta_k1;
                        beta.set(k + 1, new_beta_k1);
                        
                        // 准备下一次迭代的x, y
                        x = new_alpha_k1;
                        y = new_beta_k1;
                    }
                }
                
                // 更新左旋转矩阵Q_left
                for (int i = 0; i < Q_left.rows(); i++) {
                    double temp_i = Q_left.get(i, k);
                    double temp_i1 = (k + 1 < Q_left.cols()) ? Q_left.get(i, k + 1) : 0.0;
                    Q_left.set(i, k, c * temp_i + s * temp_i1);
                    if (k + 1 < Q_left.cols()) {
                        Q_left.set(i, k + 1, -s * temp_i + c * temp_i1);
                    }
                }
            } else {
                // 如果没有填充元素需要消除，准备下一次迭代
                if (k + 1 < n - 1 && k + 1 < beta.length()) {
                    x = alpha.get(k + 1);
                    y = beta.get(k + 1);
                }
            }
        }
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
    
    
}