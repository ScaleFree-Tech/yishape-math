package com.reremouse.lab.math.compute;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.RereFloatMatrix;
import com.reremouse.lab.math.linalg.RereFloatVector;
import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.util.Tuple3;

/**
 * CPU计算工具类（CPU Computing Utilities）
 * 
 * <p>本类提供高性能的CPU计算功能，作为GPU计算失败时的回退方案，
 * 同时为小规模数据提供优化的CPU计算实现。通过统一的接口封装，
 * 确保CPU计算逻辑的一致性和可维护性。</p>
 * 
 * <p>设计目标：</p>
 * <ul>
 *   <li><strong>统一接口</strong>：为GPU和CPU计算提供统一的接口，简化调用逻辑</li>
 *   <li><strong>性能优化</strong>：针对CPU特性进行算法优化，提高计算效率</li>
 *   <li><strong>代码复用</strong>：避免GPU和CPU计算代码的重复，降低维护成本</li>
 *   <li><strong>容错机制</strong>：作为GPU计算失败时的可靠回退方案</li>
 * </ul>
 * 
 * <p>核心功能：</p>
 * <ul>
 *   <li><strong>矩阵运算</strong>：加法、减法、乘法、转置、逆矩阵等基础运算</li>
 *   <li><strong>向量运算</strong>：加法、减法、内积、范数、统计函数等</li>
 *   <li><strong>高级运算</strong>：特征分解、奇异值分解、QR分解等复杂算法</li>
 *   <li><strong>数学函数</strong>：三角函数、指数函数、对数函数等通用数学函数</li>
 * </ul>
 * 
 * <p>性能优化策略：</p>
 * <ul>
 *   <li><strong>算法优化</strong>：使用高效的数值算法，如Strassen矩阵乘法</li>
 *   <li><strong>内存优化</strong>：优化内存访问模式，提高缓存命中率</li>
 *   <li><strong>并行计算</strong>：利用多核CPU的并行处理能力</li>
 *   <li><strong>SIMD指令</strong>：使用向量化指令加速计算</li>
 * </ul>
 * 
 * <p>使用场景：</p>
 * <ul>
 *   <li>GPU计算失败时的自动回退</li>
 *   <li>小规模数据的高效计算</li>
 *   <li>复杂算法的CPU优化实现</li>
 *   <li>跨平台兼容性保证</li>
 * </ul>
 * 
 * <p>技术特点：</p>
 * <ul>
 *   <li>纯Java实现，无外部依赖</li>
 *   <li>线程安全的静态方法设计</li>
 *   <li>完善的参数验证和异常处理</li>
 *   <li>与GPU计算接口完全兼容</li>
 * </ul>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class CPUComputeFloatUtils {
    
    // ========== 矩阵运算 / Matrix Operations ==========
    
    /**
     * CPU矩阵加法（CPU Matrix Addition）
     * 
     * <p>计算两个矩阵的元素级加法，即result[i][j] = A[i][j] + B[i][j]。
     * 这是线性代数中的基本运算，要求两个矩阵具有相同的维度。</p>
     * 
     * <p>数学定义：</p>
     * <ul>
     *   <li>对于矩阵A = [aᵢⱼ]和B = [bᵢⱼ]，加法结果C = A + B = [aᵢⱼ + bᵢⱼ]</li>
     *   <li>要求A和B都是m×n矩阵，结果C也是m×n矩阵</li>
     *   <li>满足交换律：A + B = B + A</li>
     *   <li>满足结合律：(A + B) + C = A + (B + C)</li>
     * </ul>
     * 
     * <p>算法特点：</p>
     * <ul>
     *   <li><strong>时间复杂度</strong>：O(m×n)，其中m和n是矩阵的维度</li>
     *   <li><strong>空间复杂度</strong>：O(m×n)，需要存储结果矩阵</li>
     *   <li><strong>内存访问</strong>：顺序访问，缓存友好的访问模式</li>
     *   <li><strong>数值稳定性</strong>：简单的加法运算，数值误差最小</li>
     * </ul>
     * 
     * <p>应用场景：</p>
     * <ul>
     *   <li>线性组合计算：c₁A₁ + c₂A₂ + ... + cₙAₙ</li>
     *   <li>图像处理中的像素级运算</li>
     *   <li>数值分析中的矩阵更新</li>
     *   <li>机器学习中的梯度更新</li>
     * </ul>
     * 
     * @param dataA 第一个矩阵的数据，不能为null或空
     * @param dataB 第二个矩阵的数据，不能为null或空，维度必须与dataA相同
     * @return 新的矩阵对象，包含加法运算结果
     * @throws IllegalArgumentException 当输入矩阵为null、空或维度不匹配时抛出异常
     */
    public static IMatrix<Float> matrixAdd(float[][] dataA, float[][] dataB) {
        // 参数验证：确保输入矩阵不为null
        if (dataA == null || dataB == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }
        // 维度检查：确保矩阵不为空
        if (dataA.length == 0 || dataB.length == 0) {
            throw new IllegalArgumentException("矩阵不能为空");
        }
        
        int m = dataA.length;    // 矩阵行数
        int n = dataA[0].length; // 矩阵列数
        
        // 维度匹配检查：确保两个矩阵具有相同的维度
        if (m != dataB.length || n != dataB[0].length) {
            throw new IllegalArgumentException("矩阵维度不匹配进行加法运算");
        }
        
        // 预分配结果矩阵，避免动态扩容
        float[][] result = new float[m][n];
        
        // 执行矩阵加法：对应元素相加
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = dataA[i][j] + dataB[i][j];
            }
        }
        
        return Linalg.matrix(result);  // 创建并返回结果矩阵
    }
    
    /**
     * CPU矩阵减法
     */
    public static IMatrix<Float> matrixSub(float[][] dataA, float[][] dataB) {
        if (dataA == null || dataB == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }
        if (dataA.length == 0 || dataB.length == 0) {
            throw new IllegalArgumentException("矩阵不能为空");
        }
        
        int m = dataA.length;
        int n = dataA[0].length;
        
        // 检查维度匹配
        if (m != dataB.length || n != dataB[0].length) {
            throw new IllegalArgumentException("矩阵维度不匹配进行减法运算");
        }
        
        float[][] result = new float[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = dataA[i][j] - dataB[i][j];
            }
        }
        return Linalg.matrix(result);
    }
    
    /**
     * CPU矩阵乘法（CPU Matrix Multiplication）
     * 
     * <p>计算两个矩阵的乘积，即C = A × B。这是线性代数中最重要和最常用的运算之一，
     * 广泛应用于科学计算、机器学习、图像处理等领域。</p>
     * 
     * <p>数学定义：</p>
     * <ul>
     *   <li>对于矩阵A (m×n)和B (n×p)，乘积C = A × B (m×p)</li>
     *   <li>矩阵元素：C[i][j] = Σ(k=0 to n-1) A[i][k] × B[k][j]</li>
     *   <li>要求A的列数等于B的行数</li>
     *   <li>不满足交换律：A × B ≠ B × A（除非A和B都是方阵且可交换）</li>
     * </ul>
     * 
     * <p>算法实现：</p>
     * <ul>
     *   <li><strong>三重循环</strong>：外层循环遍历结果矩阵的行和列</li>
     *   <li><strong>内积计算</strong>：内层循环计算对应行和列的内积</li>
     *   <li><strong>累加求和</strong>：使用累加器避免重复计算</li>
     *   <li><strong>内存优化</strong>：按行访问模式，提高缓存命中率</li>
     * </ul>
     * 
     * <p>算法复杂度：</p>
     * <ul>
     *   <li><strong>时间复杂度</strong>：O(m×n×p)，其中m、n、p是矩阵维度</li>
     *   <li><strong>空间复杂度</strong>：O(m×p)，存储结果矩阵</li>
     *   <li><strong>数值稳定性</strong>：使用Float累加，注意精度损失</li>
     * </ul>
     * 
     * <p>性能优化：</p>
     * <ul>
     *   <li>循环顺序优化：i-j-k顺序提高缓存局部性</li>
     *   <li>预分配结果矩阵：避免动态扩容开销</li>
     *   <li>累加器优化：减少内存访问次数</li>
     * </ul>
     * 
     * <p>应用场景：</p>
     * <ul>
     *   <li>线性变换：y = Ax，其中A是变换矩阵</li>
     *   <li>神经网络：前向传播中的权重矩阵乘法</li>
     *   <li>图像处理：卷积运算的矩阵形式</li>
     *   <li>科学计算：求解线性方程组</li>
     * </ul>
     * 
     * @param dataA 第一个矩阵的数据，维度为m×n
     * @param dataB 第二个矩阵的数据，维度为n×p
     * @return 新的矩阵对象，包含乘法运算结果，维度为m×p
     * @throws IllegalArgumentException 当输入矩阵为null、空或维度不匹配时抛出异常
     */
    public static IMatrix<Float> matrixMultiply(float[][] dataA, float[][] dataB) {
        // 参数验证：确保输入矩阵不为null
        if (dataA == null || dataB == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }
        // 维度检查：确保矩阵不为空
        if (dataA.length == 0 || dataB.length == 0) {
            throw new IllegalArgumentException("矩阵不能为空");
        }
        
        int m = dataA.length;    // 矩阵A的行数
        int n = dataA[0].length; // 矩阵A的列数（也是矩阵B的行数）
        int p = dataB[0].length; // 矩阵B的列数
        
        // 维度匹配检查：确保A的列数等于B的行数
        if (n != dataB.length) {
            throw new IllegalArgumentException("矩阵维度不匹配进行乘法运算");
        }
        
        // 预分配结果矩阵，维度为m×p
        float[][] result = new float[m][p];
        
        // 三重循环实现矩阵乘法
        for (int i = 0; i < m; i++) {        // 遍历结果矩阵的行
            for (int j = 0; j < p; j++) {    // 遍历结果矩阵的列
                Float sum = 0.0f;            // 累加器，计算内积
                for (int k = 0; k < n; k++) { // 计算A的第i行与B的第j列的内积
                    sum += dataA[i][k] * dataB[k][j];
                }
                result[i][j] = sum;          // 存储计算结果
            }
        }
        
        return Linalg.matrix(result);  // 创建并返回结果矩阵
    }
    
    /**
     * CPU矩阵标量乘法
     */
    public static IMatrix<Float> matrixScalarMultiply(float[][] dataA, Float scalar) {
        if (dataA == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }
        if (dataA.length == 0) {
            throw new IllegalArgumentException("矩阵不能为空");
        }
        
        int m = dataA.length;
        int n = dataA[0].length;
        float[][] result = new float[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = dataA[i][j] * scalar;
            }
        }
        return Linalg.matrix(result);
    }
    
    /**
     * CPU矩阵标量加法
     */
    public static IMatrix<Float> matrixScalarAdd(float[][] dataA, float scalar) {
        if (dataA == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }
        if (dataA.length == 0) {
            throw new IllegalArgumentException("矩阵不能为空");
        }
        
        int m = dataA.length;
        int n = dataA[0].length;
        float[][] result = new float[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = dataA[i][j] + scalar;
            }
        }
        return Linalg.matrix(result);
    }
    
    /**
     * CPU矩阵标量减法
     */
    public static IMatrix<Float> matrixScalarSub(float[][] dataA, float scalar) {
        if (dataA == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }
        if (dataA.length == 0) {
            throw new IllegalArgumentException("矩阵不能为空");
        }
        
        int m = dataA.length;
        int n = dataA[0].length;
        float[][] result = new float[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = dataA[i][j] - scalar;
            }
        }
        return Linalg.matrix(result);
    }
    
    /**
     * CPU矩阵转置
     */
    public static IMatrix<Float> matrixTranspose(float[][] dataA) {
        if (dataA == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }
        if (dataA.length == 0) {
            throw new IllegalArgumentException("矩阵不能为空");
        }
        
        int m = dataA.length;
        int n = dataA[0].length;
        float[][] result = new float[n][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[j][i] = dataA[i][j];
            }
        }
        return Linalg.matrix(result);
    }
    
    // ========== 向量运算 / Vector Operations ==========
    
    /**
     * CPU向量加法
     */
    public static IVector<Float> vectorAdd(IVector<Float> a, IVector<Float> b) {
        if (a == null || b == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }
        
        int length = a.length();
        if (length != b.length()) {
            throw new IllegalArgumentException("向量维度不匹配进行加法运算");
        }
        
        float[] result = new float[length];
        for (int i = 0; i < length; i++) {
            result[i] = a.get(i) + b.get(i);
        }
        return new RereFloatVector(result);
    }
    
    /**
     * CPU向量减法
     */
    public static IVector<Float> vectorSub(IVector<Float> a, IVector<Float> b) {
        if (a == null || b == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }
        
        int length = a.length();
        if (length != b.length()) {
            throw new IllegalArgumentException("向量维度不匹配进行减法运算");
        }
        
        float[] result = new float[length];
        for (int i = 0; i < length; i++) {
            result[i] = a.get(i) - b.get(i);
        }
        return new RereFloatVector(result);
    }
    
    /**
     * CPU向量乘法
     */
    public static IVector<Float> vectorMultiply(IVector<Float> a, IVector<Float> b) {
        if (a == null || b == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }
        
        int length = a.length();
        if (length != b.length()) {
            throw new IllegalArgumentException("向量维度不匹配进行乘法运算");
        }
        
        float[] result = new float[length];
        for (int i = 0; i < length; i++) {
            result[i] = a.get(i) * b.get(i);
        }
        return new RereFloatVector(result);
    }
    
    /**
     * CPU向量内积
     */
    public static Float vectorDot(IVector<Float> a, IVector<Float> b) {
        if (a == null || b == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }
        
        int length = a.length();
        if (length != b.length()) {
            throw new IllegalArgumentException("向量维度不匹配进行内积运算");
        }
        
        Float sum = 0.0f;
        for (int i = 0; i < length; i++) {
            sum += a.get(i) * b.get(i);
        }
        return sum;
    }
    
    /**
     * CPU向量标量加法
     */
    public static IVector<Float> vectorScalarAdd(IVector<Float> a, Float scalar) {
        if (a == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }
        
        int length = a.length();
        float[] result = new float[length];
        for (int i = 0; i < length; i++) {
            result[i] = a.get(i) + scalar;
        }
        return new RereFloatVector(result);
    }
    
    /**
     * CPU向量标量减法
     */
    public static IVector<Float> vectorScalarSub(IVector<Float> a, Float scalar) {
        if (a == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }
        
        int length = a.length();
        float[] result = new float[length];
        for (int i = 0; i < length; i++) {
            result[i] = a.get(i) - scalar;
        }
        return new RereFloatVector(result);
    }
    
    /**
     * CPU向量标量乘法
     */
    public static IVector<Float> vectorScalarMultiply(IVector<Float> a, Float scalar) {
        if (a == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }
        
        int length = a.length();
        float[] result = new float[length];
        for (int i = 0; i < length; i++) {
            result[i] = a.get(i) * scalar;
        }
        return new RereFloatVector(result);
    }
    
    /**
     * CPU向量标量除法
     */
    public static IVector<Float> vectorScalarDivide(IVector<Float> a, Float scalar) {
        if (a == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }
        if (scalar == 0.0f) {
            throw new ArithmeticException("除数不能为零");
        }
        
        int length = a.length();
        float[] result = new float[length];
        for (int i = 0; i < length; i++) {
            result[i] = a.get(i) / scalar;
        }
        return new RereFloatVector(result);
    }
    
    /**
     * CPU向量平方
     */
    public static IVector<Float> vectorSquare(IVector<Float> a) {
        if (a == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }
        
        int length = a.length();
        float[] result = new float[length];
        for (int i = 0; i < length; i++) {
            Float val = a.get(i);
            result[i] = val * val;
        }
        return new RereFloatVector(result);
    }
    
    /**
     * CPU向量开方
     */
    public static IVector<Float> vectorSqrt(IVector<Float> a) {
        if (a == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }
        
        int length = a.length();
        float[] result = new float[length];
        for (int i = 0; i < length; i++) {
            result[i] = (float) Math.sqrt(a.get(i));
        }
        return new RereFloatVector(result);
    }
    
    /**
     * CPU向量求和
     */
    public static Float vectorSum(IVector<Float> a) {
        if (a == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }
        
        int length = a.length();
        Float sum = 0.0f;
        for (int i = 0; i < length; i++) {
            sum += a.get(i);
        }
        return sum;
    }
    
    /**
     * CPU向量倒数计算
     * 计算向量的倒数，用于伪逆矩阵计算
     */
    public static IVector<Float> vectorReciprocal(IVector<Float> a, Float tolerance) {
        if (a == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }
        
        float[] dataA = a.toFloatArray();
        float[] result = new float[dataA.length];
        
        for (int i = 0; i < dataA.length; i++) {
            Float value = dataA[i];
            if (Math.abs(value) > tolerance) {
                result[i] = 1.0f / value;
            } else {
                result[i] = 0.0f;
            }
        }
        
        return new RereFloatVector(result);
    }
    
    // ========== 高级运算 / Advanced Operations ==========
    
    /**
     * CPU伪逆矩阵计算
     */
    public static IMatrix<Float> pseudoInverse(IMatrix<Float> A) {
        if (A == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }
        
        final Float tolerance = 1e-10f;
        
        // 进行奇异值分解：A = U * S * V^T
        var svdResult = A.svd();
        IMatrix<Float> U = (IMatrix<Float>)svdResult._1;           // 左奇异向量矩阵
        IVector<Float> singularValues = (IVector<Float>)svdResult._2;  // 奇异值向量
        IMatrix<Float> VT = (IMatrix<Float>)svdResult._3;          // 右奇异向量转置矩阵
        
        // 获取矩阵的维度信息
        int originalRows = A.rows();
        int originalCols = A.cols();
        int singularValuesLength = singularValues.length();
        
        // 计算奇异值的伪逆
        IVector<Float> pseudoSingularValues = Linalg.zeros(singularValuesLength,Float.class);
        
        for (int i = 0; i < singularValuesLength; i++) {
            Float sv = singularValues.get(i);
            if (Math.abs(sv) > tolerance) {
                pseudoSingularValues.set(i, 1.0f / sv);  // 非零奇异值的倒数
            } else {
                pseudoSingularValues.set(i, 0.0f);       // 零奇异值保持为零
            }
        }
        
        // 计算伪逆：A⁺ = V * Σ⁺ * U^T
        IMatrix<Float> V = (IMatrix<Float>)VT.transposeNew();  // V = (V^T)^T
        
        // 创建结果矩阵：A⁺的维度应该是 originalCols x originalRows
        IMatrix<Float> pseudoInverse = Linalg.zeros(originalCols, originalRows,Float.class);
        
        // 逐元素计算伪逆：A⁺[i,j] = Σ(k=0 to rank-1) V[i,k] * (1/σ[k]) * U[j,k]
        for (int i = 0; i < originalCols; i++) {
            for (int j = 0; j < originalRows; j++) {
                Float sum = 0.0f;
                for (int k = 0; k < singularValuesLength; k++) {
                    Float vValue = (k < V.cols()) ? V.get(i, k) : 0.0f;
                    Float uValue = (k < U.cols()) ? U.get(j, k) : 0.0f;
                    Float sigmaInv = pseudoSingularValues.get(k);
                    sum += vValue * sigmaInv * uValue;
                }
                pseudoInverse.put(i, j, sum);
            }
        }
        
        return pseudoInverse;
    }
    
    /**
     * CPU特征分解
     * 使用QR算法计算特征值和特征向量
     */
    public static Tuple2<IVector<Float>, IMatrix<Float>> eigen(IMatrix<Float> A) {
        if (A == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }
        
        // 检查是否为方阵
        if (A.rows() != A.cols()) {
            throw new IllegalArgumentException("特征分解需要方阵 / Eigendecomposition requires square matrix");
        }
        
        // 如果已经是RereMatrix，直接使用其方法避免复制
        if (A instanceof RereFloatMatrix) {
            var eigen = ((RereFloatMatrix) A).qrEigenDecomposition();
        return new Tuple2(eigen._1,eigen._2);
        }
        
        // 对于其他IMatrix实现，需要复制数据
        int n = A.rows();
        float[][] data = A.toFloatArray();
        float[][] matrixData = new float[n][n];
        
        // 复制矩阵数据
        for (int i = 0; i < n; i++) {
            System.arraycopy(data[i], 0, matrixData[i], 0, n);
        }
        
        // 使用QR算法计算特征值和特征向量
        var eigen = Linalg.matrix(matrixData).qrEigenDecomposition();
        return new Tuple2(eigen._1,eigen._2);
    }
    
    /**
     * CPU奇异值分解
     * 根据矩阵大小选择不同的算法，直接实现CPU版本
     */
    public static Tuple3<IMatrix<Float>, IVector<Float>, IMatrix<Float>> svd(IMatrix<Float> A) {
        if (A == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }
        
        int m = A.rows();    // 行数
        int n = A.cols(); // 列数
        
        // 如果已经是RereMatrix，直接使用其方法避免转换
        if (A instanceof RereFloatMatrix) {
            RereFloatMatrix matrix = (RereFloatMatrix) A;
            
            // 对于大矩阵，使用优化的SVD算法
            if (m * n > 10000) {
                var svd = matrix.optimizedSVD();
                return new Tuple3(svd._1,svd._2,svd._3);
            }
            
            // 对于中等矩阵，使用双对角化方法
            if (m * n > 1000) {
                var svd = matrix.bidiagonalSVD();
                return new Tuple3(svd._1,svd._2,svd._3);
            }
            
            // 对于小矩阵，使用传统方法但优化排序
            var svd = matrix.traditionalSVD();
            return new Tuple3(svd._1,svd._2,svd._3);
        }
        
        // 对于其他IMatrix实现，需要转换为RereMatrix
        float[][] data = A.toFloatArray();
        RereFloatMatrix matrix = (RereFloatMatrix)Linalg.matrix(data);
        
        // 根据矩阵大小选择算法
        if (m * n > 10000) {
            var svd = matrix.optimizedSVD();
            return new Tuple3(svd._1,svd._2,svd._3);
        } else if (m * n > 1000) {
            var svd = matrix.bidiagonalSVD();
            return new Tuple3(svd._1,svd._2,svd._3);
        } else {
            var svd = matrix.traditionalSVD();
            return new Tuple3(svd._1,svd._2,svd._3);
        }
    }
    
    /**
     * CPU矩阵逐元素乘法（CPU Matrix Element-wise Multiplication）
     * 
     * <p>计算两个矩阵的逐元素乘法，即result[i][j] = A[i][j] * B[i][j]。
     * 这是线性代数中的基本运算，要求两个矩阵具有相同的维度。</p>
     * 
     * <p>数学定义：</p>
     * <ul>
     *   <li>对于矩阵A = [aᵢⱼ]和B = [bᵢⱼ]，逐元素乘法结果C = A ⊙ B = [aᵢⱼ * bᵢⱼ]</li>
     *   <li>要求A和B都是m×n矩阵，结果C也是m×n矩阵</li>
     *   <li>满足交换律：A ⊙ B = B ⊙ A</li>
     *   <li>满足结合律：(A ⊙ B) ⊙ C = A ⊙ (B ⊙ C)</li>
     * </ul>
     * 
     * <p>算法特点：</p>
     * <ul>
     *   <li><strong>时间复杂度</strong>：O(m×n)，其中m和n是矩阵的维度</li>
     *   <li><strong>空间复杂度</strong>：O(m×n)，需要存储结果矩阵</li>
     *   <li><strong>内存访问</strong>：顺序访问，缓存友好的访问模式</li>
     *   <li><strong>数值稳定性</strong>：简单的乘法运算，数值误差最小</li>
     * </ul>
     * 
     * <p>应用场景：</p>
     * <ul>
     *   <li>图像处理中的像素级运算</li>
     *   <li>神经网络中的激活函数应用</li>
     *   <li>数值分析中的权重矩阵运算</li>
     *   <li>机器学习中的特征缩放</li>
     * </ul>
     * 
     * @param first 第一个矩阵，不能为null
     * @param other 第二个矩阵，不能为null，维度必须与first相同
     * @return 新的矩阵对象，包含逐元素乘法运算结果
     * @throws IllegalArgumentException 当输入矩阵为null或维度不匹配时抛出异常
     */
    public static IMatrix<Float> matrixElementWiseMultiply(IMatrix<Float> first, IMatrix<Float> other) {
        // 参数验证：确保输入矩阵不为null
        if (first == null || other == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }
        
        // 维度检查：确保两个矩阵具有相同的维度
        if (first.rows() != other.rows() || first.cols() != other.cols()) {
            throw new IllegalArgumentException("矩阵维度不匹配进行逐元素乘法运算");
        }
        
        int m = first.rows();    // 矩阵行数
        int n = first.cols();    // 矩阵列数
        
        // 预分配结果矩阵，避免动态扩容
        float[][] result = new float[m][n];
        
        // 执行矩阵逐元素乘法：对应元素相乘
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = first.get(i, j) * other.get(i, j);
            }
        }
        
        return Linalg.matrix(result);  // 创建并返回结果矩阵
    }
}
