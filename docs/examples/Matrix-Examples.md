# 矩阵运算示例 (Matrix Operation Examples)

## 概述 / Overview

本文档提供了 `IMatrix` 接口和 `RereMatrix` 实现类的详细使用示例，涵盖从基础操作到高级应用的各个方面。

This document provides detailed usage examples for the `IMatrix` interface and `RereMatrix` implementation class, covering everything from basic operations to advanced applications.

## 基础示例 / Basic Examples

### 矩阵创建和基本操作 / Matrix Creation and Basic Operations

```java
import com.reremouse.lab.math.IMatrix;
import com.reremouse.lab.math.IVector;
import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.util.Tuple3;
import java.util.Arrays;
import java.util.List;

public class MatrixBasicExample {
    public static void main(String[] args) {
        // 矩阵创建 / Matrix creation
        float[][] data = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        IMatrix matrix = IMatrix.of(data);
        
        // 从List创建 / Create from List
        List<float[]> rows = Arrays.asList(
            new float[]{1, 2, 3},
            new float[]{4, 5, 6},
            new float[]{7, 8, 9}
        );
        IMatrix matrix2 = IMatrix.of(rows);
        
        // 从Vector数组创建 / Create from Vector array
        IVector[] vectors = {IVector.of(new float[]{1, 2}), IVector.of(new float[]{3, 4})};
        IMatrix matrix3 = IMatrix.of(vectors);
        
        // 创建特殊矩阵 / Create special matrices
        IMatrix ones = IMatrix.ones(3, 3);
        IMatrix zeros = IMatrix.zeros(3, 3);
        IMatrix identity = IMatrix.eye(3);
        IMatrix random = IMatrix.rand(3, 3);
        IMatrix randomSeeded = IMatrix.rand(3, 3, 12345L);
        
        // 创建对角矩阵 / Create diagonal matrix
        IMatrix diagonal = IMatrix.diag(new float[]{1, 2, 3});
        
        // 从一维数组创建 / Create from 1D array
        IMatrix reshaped = IMatrix.fromArray(new float[]{1, 2, 3, 4}, 2, 2);
        
        // 从文件加载 / Load from file
        IMatrix loaded = IMatrix.load("matrix.txt");
        
        // 基本运算 / Basic operations
        IMatrix sum = matrix.add(ones);
        IMatrix diff = matrix.sub(ones);
        IMatrix transposed = matrix.transpose();
        
        System.out.println("矩阵: " + matrix);
        System.out.println("转置: " + transposed);
    }
}
```

### 基本数学运算 / Basic Mathematical Operations

```java
public class BasicMathOperationsExample {
    public static void main(String[] args) {
        // 创建矩阵 / Create matrices
        float[][] data1 = {{1, 2}, {3, 4}};
        float[][] data2 = {{5, 6}, {7, 8}};
        IMatrix matrix1 = IMatrix.of(data1);
        IMatrix matrix2 = IMatrix.of(data2);

        // 基本运算 / Basic operations
        IMatrix sum = matrix1.add(matrix2);                    // [[6, 8], [10, 12]]
        IMatrix diff = matrix1.sub(matrix2);                   // [[-4, -4], [-4, -4]]
        IMatrix matrixProduct = matrix1.mmul(matrix2);         // [[19, 22], [43, 50]]

        // 标量运算 / Scalar operations
        IMatrix scaled = matrix1.mmul(2.0f);                   // [[2, 4], [6, 8]]
        IMatrix shifted = matrix1.sub(10.0f);                  // [[-9, -8], [-7, -6]]
        
        // 向量点积 / Vector dot product
        float dotProduct = matrix1.dot(matrix2); // 要求都是列向量
        
        System.out.println("矩阵加法结果: " + sum);
        System.out.println("矩阵乘法结果: " + matrixProduct);
        System.out.println("标量乘法结果: " + scaled);
    }
}
```

### 矩阵变换 / Matrix Transformations

```java
public class MatrixTransformationsExample {
    public static void main(String[] args) {
        IMatrix matrix = IMatrix.of(new float[][]{{1, 2}, {3, 4}});

        // 转置 / Transpose
        IMatrix transposed = matrix.transpose();                // [[1, 3], [2, 4]]
        IMatrix transposedNew = matrix.transposeNew();         // 创建新对象
        IMatrix transposedShort = matrix.t();                  // 简写形式

        // 幂运算 / Power
        IMatrix squared = matrix.pow(2.0f);                     // 元素级幂运算

        // 开方 / Square root
        IMatrix sqrt = matrix.sqrt();                           // 元素级开方
        
        // 求逆 / Inverse
        IMatrix inverse = matrix.inv();
        
        // 伪逆 / Pseudo-inverse
        IMatrix pseudoInverse = matrix.pinv();

        System.out.println("转置: " + transposed);
        System.out.println("平方: " + squared);
        System.out.println("开方: " + sqrt);
    }
}
```

### 线性代数运算 / Linear Algebra Operations

#### 特征分解 / Eigendecomposition

```java
public class EigendecompositionExample {
    public static void main(String[] args) {
        // 特征分解 / Eigendecomposition
        float[][] data = {{4, -2}, {-2, 4}};
        IMatrix matrix = IMatrix.of(data);
        
        Tuple2<IVector, IMatrix> eigenResult = matrix.eigen();
        IVector eigenValues = eigenResult._1;
        IMatrix eigenVectors = eigenResult._2;
        
        System.out.println("特征值: " + eigenValues);
        System.out.println("特征向量: " + eigenVectors);
        
        // 验证：A * v = λ * v
        IVector eigenVector1 = eigenVectors.getColunm(0);
        IVector result = matrix.mmul(eigenVector1);
        // result ≈ 6 * eigenVector1
        
        // QR算法特征分解 / QR algorithm eigendecomposition
        Tuple2<IVector, IMatrix> qrEigenResult = matrix.qrEigenDecomposition();
    }
}
```

#### 奇异值分解 / Singular Value Decomposition

```java
public class SVDExample {
    public static void main(String[] args) {
        float[][] data = {{4, -2}, {-2, 4}};
        IMatrix matrix = IMatrix.of(data);
        
        // SVD分解 / SVD decomposition
        Tuple3<IMatrix, IVector, IMatrix> svdResult = matrix.svd();
        IMatrix U = svdResult._1;                  // 左奇异向量 / Left singular vectors
        IVector S = svdResult._2;                  // 奇异值 / Singular values
        IMatrix V = svdResult._3;                  // 右奇异向量 / Right singular vectors
        
        System.out.println("奇异值: " + S);
        System.out.println("左奇异向量: " + U);
        System.out.println("右奇异向量: " + V);
    }
}
```

#### 矩阵分解 / Matrix Decomposition

```java
public class MatrixDecompositionExample {
    public static void main(String[] args) {
        IMatrix matrix = IMatrix.of(new float[][]{{2, 1}, {1, 3}});
        
        // QR分解 / QR decomposition
        Tuple2<IMatrix, IMatrix> qrResult = matrix.qr();
        IMatrix Q = qrResult._1;                   // 正交矩阵 / Orthogonal matrix
        IMatrix R = qrResult._2;                   // 上三角矩阵 / Upper triangular matrix

        // LU分解 / LU decomposition
        Tuple2<IMatrix, IMatrix> luResult = matrix.lu();
        IMatrix L = luResult._1;                   // 下三角矩阵 / Lower triangular matrix
        IMatrix U = luResult._2;                   // 上三角矩阵 / Upper triangular matrix

        // Cholesky分解 / Cholesky decomposition
        IMatrix L_chol = matrix.cholesky();        // 下三角矩阵L，A = L * L^T
        
        System.out.println("Q矩阵: " + Q);
        System.out.println("R矩阵: " + R);
        System.out.println("L矩阵: " + L);
        System.out.println("U矩阵: " + U);
    }
}
```

#### 线性方程组求解 / Linear System Solving

```java
public class LinearSystemExample {
    public static void main(String[] args) {
        IMatrix A = IMatrix.of(new float[][]{{2, 1}, {1, 3}});
        IVector b = IVector.of(new float[]{5, 6});

        // LU分解 / LU decomposition
        Tuple2<IMatrix, IMatrix> luResult = A.lu();
        IMatrix L = luResult._1;  // 下三角矩阵
        IMatrix U = luResult._2;  // 上三角矩阵

        // 求解线性方程组 / Solve linear system
        IVector x = A.solve(b);   // 求解Ax = b

        // 验证结果 / Verify result
        IVector Ax = A.mmul(x);   // 应该等于b
        
        System.out.println("解向量x: " + x);
        System.out.println("验证Ax: " + Ax);
        System.out.println("右侧向量b: " + b);
    }
}
```

#### 行列式和迹 / Determinant and Trace

```java
public class DeterminantTraceExample {
    public static void main(String[] args) {
        IMatrix matrix = IMatrix.of(new float[][]{{2, 1}, {1, 3}});
        
        // 行列式 / Determinant
        float det = matrix.det();

        // 迹 / Trace
        float trace = matrix.trace();

        // 秩 / Rank
        int rank = matrix.rank();

        // 条件数 / Condition number
        float cond = matrix.cond();
        
        System.out.println("行列式: " + det);
        System.out.println("迹: " + trace);
        System.out.println("秩: " + rank);
        System.out.println("条件数: " + cond);
    }
}
```

### 统计运算 / Statistical Operations

```java
public class MatrixStatisticsExample {
    public static void main(String[] args) {
        float[][] data = {
            {12.5f, 15.2f, 18.7f},
            {13.9f, 17.1f, 19.4f},
            {14.8f, 16.5f, 20.1f}
        };
        
        IMatrix matrix = IMatrix.of(data);
        
        // 行统计 / Row statistics (注意：实际是按列求和)
        IVector rowSums = matrix.rowSums();        // 行和（按列求和）
        IVector rowMeans = matrix.rowMeans();      // 行均值（按列求均值）
        
        // 列统计 / Column statistics (注意：实际是按行求和)
        IVector colSums = matrix.colSums();        // 列和（按行求和）
        IVector colMeans = matrix.colMeans();      // 列均值（按行求均值）
        
        // 整体统计 / Overall statistics
        float totalSum = matrix.sum();             // 总和
        float totalMean = matrix.mean();           // 整体均值
        float variance = matrix.var();             // 整体方差
        float stdDev = matrix.std();               // 整体标准差
        float max = matrix.max();                  // 最大值
        float min = matrix.min();                  // 最小值
        
        System.out.println("行和: " + rowSums);
        System.out.println("列均值: " + colMeans);
        System.out.println("总和: " + totalSum);
        System.out.println("均值: " + totalMean);
        System.out.println("方差: " + variance);
        System.out.println("标准差: " + stdDev);
    }
}
```

### 数据访问和操作 / Data Access and Manipulation

```java
public class DataAccessExample {
    public static void main(String[] args) {
        IMatrix matrix = IMatrix.of(new float[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
        
        // 获取行 / Get row
        IVector row = matrix.getRow(0);

        // 获取列 / Get column
        IVector column = matrix.getColunm(0);
        IMatrix columnMatrix = matrix.getColumn(0);

        // 设置列 / Set column
        IVector newColumn = IVector.of(new float[]{10, 11, 12});
        matrix.putColumn(0, newColumn);

        // 获取多个列 / Get multiple columns
        IMatrix[] columns = matrix.getColumns(new int[]{0, 2});

        // 获取形状 / Get shape
        int[] shape = matrix.shape();             // [行数, 列数]
        int rows = matrix.getRowNum();            // 行数
        int cols = matrix.getColNum();            // 列数
        int rowsAlt = matrix.getRows();           // 行数（替代方法）
        int colsAlt = matrix.getColumns();        // 列数（替代方法）
        
        // 获取和设置元素 / Get and set elements
        float element = matrix.get(1, 2);
        matrix.put(1, 2, 10.0f);
        
        // 负数索引访问 / Negative indexing access
        float lastElement = matrix.get(-1, -1);  // 获取最后一个元素
        matrix.put(-1, -1, 100.0f);             // 设置最后一个元素
        
        System.out.println("第0行: " + row);
        System.out.println("第0列: " + column);
        System.out.println("矩阵形状: " + Arrays.toString(shape));
        System.out.println("最后一个元素: " + lastElement);
    }
}
```

### 矩阵切片表达式和负数索引 / Matrix Slice Expressions and Negative Indexing

```java
public class MatrixSliceExpressionExample {
    public static void main(String[] args) {
        // 创建测试矩阵 / Create test matrix
        IMatrix matrix = IMatrix.of(new float[][]{
            {1, 2, 3, 4, 5},
            {6, 7, 8, 9, 10},
            {11, 12, 13, 14, 15},
            {16, 17, 18, 19, 20},
            {21, 22, 23, 24, 25}
        });
        
        System.out.println("原始矩阵:");
        System.out.println(matrix);
        System.out.println("矩阵形状: " + Arrays.toString(matrix.shape()));
        
        System.out.println("\n=== 基本切片表达式 / Basic Slice Expressions ===");
        
        // 基本切片 / Basic slicing
        IMatrix slice1 = matrix.slice("1:3", "1:3");  // 行1-2，列1-2
        System.out.println("slice(\"1:3\", \"1:3\"):");
        System.out.println(slice1);
        
        IMatrix slice2 = matrix.slice("2:", "2:");    // 从行2到末尾，从列2到末尾
        System.out.println("slice(\"2:\", \"2:\"):");
        System.out.println(slice2);
        
        IMatrix slice3 = matrix.slice(":3", ":3");    // 从开始到行2，从开始到列2
        System.out.println("slice(\":3\", \":3\"):");
        System.out.println(slice3);
        
        IMatrix slice4 = matrix.slice("::2", "::2");  // 每隔一行一列
        System.out.println("slice(\"::2\", \"::2\"):");
        System.out.println(slice4);
        
        IMatrix slice5 = matrix.slice("1:4:2", "1:4:2"); // 行1,3，列1,3
        System.out.println("slice(\"1:4:2\", \"1:4:2\"):");
        System.out.println(slice5);
        
        System.out.println("\n=== 负数索引切片 / Negative Indexing Slicing ===");
        
        // 负数索引切片 / Negative indexing slicing
        IMatrix slice6 = matrix.slice("-2:", "-2:");  // 最后两行，最后两列
        System.out.println("slice(\"-2:\", \"-2:\"):");
        System.out.println(slice6);
        
        IMatrix slice7 = matrix.slice(":-2", ":-2");  // 除最后两行外的所有行，除最后两列外的所有列
        System.out.println("slice(\":-2\", \":-2\"):");
        System.out.println(slice7);
        
        IMatrix slice8 = matrix.slice("-3:-1", "-3:-1"); // 倒数第3-2行，倒数第3-2列
        System.out.println("slice(\"-3:-1\", \"-3:-1\"):");
        System.out.println(slice8);
        
        IMatrix slice9 = matrix.slice("-4:-1:2", "-4:-1:2"); // 倒数第4-2行，倒数第4-2列，步长为2
        System.out.println("slice(\"-4:-1:2\", \"-4:-1:2\"):");
        System.out.println(slice9);
        
        IMatrix slice10 = matrix.slice("::-1", "::-1"); // 反转向量（行列都反转）
        System.out.println("slice(\"::-1\", \"::-1\"):");
        System.out.println(slice10);
        
        System.out.println("\n=== 行切片和列切片 / Row and Column Slicing ===");
        
        // 行切片 / Row slicing
        IMatrix rowSlice1 = matrix.sliceRows("1:3");   // 行1-2
        System.out.println("sliceRows(\"1:3\"):");
        System.out.println(rowSlice1);
        
        IMatrix rowSlice2 = matrix.sliceRows("-2:");   // 最后两行
        System.out.println("sliceRows(\"-2:\"):");
        System.out.println(rowSlice2);
        
        IMatrix rowSlice3 = matrix.sliceRows("::2");   // 每隔一行
        System.out.println("sliceRows(\"::2\"):");
        System.out.println(rowSlice3);
        
        // 列切片 / Column slicing
        IMatrix colSlice1 = matrix.sliceColumns("1:3"); // 列1-2
        System.out.println("sliceColumns(\"1:3\"):");
        System.out.println(colSlice1);
        
        IMatrix colSlice2 = matrix.sliceColumns("-2:"); // 最后两列
        System.out.println("sliceColumns(\"-2:\"):");
        System.out.println(colSlice2);
        
        IMatrix colSlice3 = matrix.sliceColumns("::2"); // 每隔一列
        System.out.println("sliceColumns(\"::2\"):");
        System.out.println(colSlice3);
        
        System.out.println("\n=== 负数索引访问 / Negative Indexing Access ===");
        
        // 负数索引访问 / Negative indexing access
        System.out.println("matrix.get(0, 0): " + matrix.get(0, 0));     // 第一个元素
        System.out.println("matrix.get(-1, -1): " + matrix.get(-1, -1)); // 最后一个元素
        System.out.println("matrix.get(-1, 0): " + matrix.get(-1, 0));   // 最后一行的第一个元素
        System.out.println("matrix.get(0, -1): " + matrix.get(0, -1));   // 第一行的最后一个元素
        System.out.println("matrix.get(-2, -2): " + matrix.get(-2, -2)); // 倒数第二行倒数第二列
        
        System.out.println("\n=== 切片表达式边界情况 / Slice Expression Edge Cases ===");
        
        // 边界情况 / Edge cases
        IMatrix slice11 = matrix.slice("0:0", "0:0"); // 空切片
        System.out.println("slice(\"0:0\", \"0:0\"): " + slice11 + " (形状: " + Arrays.toString(slice11.shape()) + ")");
        
        IMatrix slice12 = matrix.slice("2:2", "2:2"); // 空切片
        System.out.println("slice(\"2:2\", \"2:2\"): " + slice12 + " (形状: " + Arrays.toString(slice12.shape()) + ")");
        
        IMatrix slice13 = matrix.slice("-1:-1", "-1:-1"); // 空切片
        System.out.println("slice(\"-1:-1\", \"-1:-1\"): " + slice13 + " (形状: " + Arrays.toString(slice13.shape()) + ")");
        
        System.out.println("\n=== 切片表达式错误处理 / Slice Expression Error Handling ===");
        
        try {
            // 无效的切片表达式 / Invalid slice expression
            matrix.slice("1:2:3:4", "1:2");  // 超过3个冒号
        } catch (IllegalArgumentException e) {
            System.out.println("捕获异常: " + e.getMessage());
        }
        
        try {
            // 无效的步长 / Invalid step
            matrix.slice("1:3:0", "1:3");    // 步长为0
        } catch (IllegalArgumentException e) {
            System.out.println("捕获异常: " + e.getMessage());
        }
        
        try {
            // 索引超出范围 / Index out of bounds
            matrix.slice("1:10", "1:10");    // 结束索引超出范围
        } catch (IllegalArgumentException e) {
            System.out.println("捕获异常: " + e.getMessage());
        }
        
        System.out.println("\n=== 实际应用示例 / Practical Application Examples ===");
        
        // 获取中心部分（去除边界） / Get center part (excluding boundaries)
        IMatrix center = matrix.slice("1:-1", "1:-1");
        System.out.println("中心部分（去除边界）:");
        System.out.println(center);
        
        // 获取四个角落 / Get four corners
        IMatrix topLeft = matrix.slice(":2", ":2");
        IMatrix topRight = matrix.slice(":2", "-2:");
        IMatrix bottomLeft = matrix.slice("-2:", ":2");
        IMatrix bottomRight = matrix.slice("-2:", "-2:");
        
        System.out.println("左上角:");
        System.out.println(topLeft);
        System.out.println("右上角:");
        System.out.println(topRight);
        System.out.println("左下角:");
        System.out.println(bottomLeft);
        System.out.println("右下角:");
        System.out.println(bottomRight);
        
        // 获取对角线元素 / Get diagonal elements
        IMatrix diagonal = matrix.slice("::1", "::1"); // 这里需要特殊处理对角线
        System.out.println("对角线（需要特殊处理）:");
        System.out.println(diagonal);
        
        // 获取奇数行和偶数列 / Get odd rows and even columns
        IMatrix oddRowsEvenCols = matrix.slice("1::2", "::2");
        System.out.println("奇数行和偶数列:");
        System.out.println(oddRowsEvenCols);
        
        // 获取最后几行和最后几列 / Get last few rows and columns
        IMatrix lastFew = matrix.slice("-3:", "-3:");
        System.out.println("最后三行三列:");
        System.out.println(lastFew);
    }
}
```

### 数据预处理 / Data Preprocessing

```java
public class DataPreprocessingExample {
    public static void main(String[] args) {
        IMatrix data = IMatrix.of(new float[][]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });

        // 中心化 / Centering
        IMatrix centered = data.center();                       // 每列减去列均值

        // 协方差矩阵 / Covariance matrix
        IMatrix covMatrix = data.covariance();                  // 从原始数据计算协方差
        IMatrix covFromCentered = data.covarianceFromCentered(); // 从中心化数据计算协方差

        // 归一化 / Normalization
        IMatrix normalizedRows = data.normalizeRows();          // L2行归一化
        IMatrix normalizedCols = data.normalizeColumns();       // L2列归一化
        
        System.out.println("中心化后: " + centered);
        System.out.println("协方差矩阵: " + covMatrix);
        System.out.println("行归一化: " + normalizedRows);
    }
}
```

### 数学函数应用 / Mathematical Functions

```java
public class MathematicalFunctionsExample {
    public static void main(String[] args) {
        IMatrix matrix = IMatrix.of(new float[][]{{1, 2}, {3, 4}});

        // 三角函数 / Trigonometric functions
        IMatrix sinResult = matrix.sin();          // 正弦
        IMatrix cosResult = matrix.cos();          // 余弦
        IMatrix tanResult = matrix.tan();          // 正切

        // 双曲函数 / Hyperbolic functions
        IMatrix sinhResult = matrix.sinh();        // 双曲正弦
        IMatrix coshResult = matrix.cosh();        // 双曲余弦
        IMatrix tanhResult = matrix.tanh();        // 双曲正切

        // 指数和对数 / Exponential and logarithm
        IMatrix expResult = matrix.exp();          // 指数
        IMatrix logResult = matrix.log();          // 自然对数

        // 绝对值和符号 / Absolute value and sign
        IMatrix absResult = matrix.abs();          // 绝对值
        IMatrix signResult = matrix.sign();        // 符号函数
        
        System.out.println("正弦: " + sinResult);
        System.out.println("指数: " + expResult);
        System.out.println("绝对值: " + absResult);
    }
}
```

### 矩阵操作 / Matrix Operations

```java
public class MatrixOperationsExample {
    public static void main(String[] args) {
        IMatrix A = IMatrix.of(new float[][]{{1, 2}, {3, 4}});
        IMatrix B = IMatrix.of(new float[][]{{5, 6}, {7, 8}});

        // 水平拼接 / Horizontal concatenation
        IMatrix hstacked = A.hstack(B);  // [[1, 2, 5, 6], [3, 4, 7, 8]]

        // 垂直拼接 / Vertical concatenation
        IMatrix vstacked = A.vstack(B);  // [[1, 2], [3, 4], [5, 6], [7, 8]]

        // 分割矩阵 / Split matrix
        IMatrix[] hsplit = hstacked.hsplit(new int[]{2});  // 在第2列处分割
        IMatrix[] vsplit = vstacked.vsplit(new int[]{2});  // 在第2行处分割

        // 矩阵重塑 / Matrix reshape
        IMatrix reshaped = A.reshape(4, 1);  // 重塑为4x1

        // 矩阵复制 / Matrix copy
        IMatrix copied = A.copy();            // 深拷贝

        // 矩阵范数 / Matrix norms
        float frobeniusNorm = A.frobeniusNorm(); // Frobenius范数
        float frobeniusDistance = A.frobeniusDistance(B); // Frobenius距离
        
        System.out.println("水平拼接: " + hstacked);
        System.out.println("垂直拼接: " + vstacked);
        System.out.println("Frobenius范数: " + frobeniusNorm);
    }
}
```

### 文件操作 / File Operations

```java
public class FileOperationsExample {
    public static void main(String[] args) {
        // 保存矩阵到文件 / Save matrix to file
        IMatrix matrix = IMatrix.of(new float[][]{{1, 2}, {3, 4}});
        matrix.save("matrix.txt");

        // 从文件加载矩阵 / Load matrix from file
        IMatrix loaded = IMatrix.load("matrix.txt");
        
        System.out.println("原始矩阵: " + matrix);
        System.out.println("加载的矩阵: " + loaded);
    }
}
```

## 高级示例 / Advanced Examples

### 链式操作 / Method Chaining

```java
public class MethodChainingExample {
    public static void main(String[] args) {
        IMatrix result = IMatrix.ones(3, 3)                     // 创建3x3全1矩阵
            .sub(5.0f)                                          // 每个元素减5
            .mmul(2.0f)                                         // 每个元素乘2
            .transpose()                                         // 转置
            .pow(2);                                            // 每个元素平方
            
        System.out.println("链式操作结果: " + result);
    }
}
```

### 矩阵分解和重构 / Matrix Decomposition and Reconstruction

```java
public class MatrixDecompositionReconstructionExample {
    public static void main(String[] args) {
        // 创建低秩矩阵 / Create low-rank matrix
        int rows = 100;
        int cols = 50;
        int rank = 10;
        
        float[][] lowRankData = new float[rows][cols];
        // 生成低秩矩阵数据 / Generate low-rank matrix data
        
        IMatrix originalMatrix = IMatrix.of(lowRankData);
        
        // SVD分解 / SVD decomposition
        Tuple3<IMatrix, IVector, IMatrix> svdResult = originalMatrix.svd();
        
        // 不同秩的重构 / Reconstruction with different ranks
        for (int testRank = 1; testRank <= 20; testRank += 5) {
            // 使用前testRank个奇异值重构
            IMatrix U = svdResult._1;
            IVector S = svdResult._2;
            IMatrix V = svdResult._3;
            
            // 重构矩阵
            IMatrix reconstructed = U.mmul(IMatrix.diag(S.toArray())).mmul(V);
            
            // 计算重构误差
            float error = originalMatrix.frobeniusDistance(reconstructed);
            
            System.out.println("秩 " + testRank + ": 重构误差 = " + error);
        }
    }
}
```

### 机器学习应用示例 / Machine Learning Application Examples

```java
public class MachineLearningExample {
    public static void main(String[] args) {
        // 特征矩阵 / Feature matrix
        IMatrix X = IMatrix.of(new float[][]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });
        
        // 标签向量 / Label vector
        IVector y = IVector.of(new float[]{1, 2, 3});
        
        // 数据预处理 / Data preprocessing
        IMatrix X_centered = X.center();                    // 中心化
        IMatrix X_normalized = X.normalizeColumns();        // 列归一化
        
        // 计算协方差矩阵 / Compute covariance matrix
        IMatrix covMatrix = X_centered.covarianceFromCentered();
        
        // 特征分解 / Eigendecomposition
        Tuple2<IVector, IMatrix> eigenResult = covMatrix.eigen();
        IVector eigenValues = eigenResult._1;
        IMatrix eigenVectors = eigenResult._2;
        
        // PCA降维 / PCA dimensionality reduction
        IMatrix X_pca = X_centered.mmul(eigenVectors);
        
        System.out.println("特征值: " + eigenValues);
        System.out.println("PCA结果: " + X_pca);
    }
}
```

### 图像处理示例 / Image Processing Examples

```java
public class ImageProcessingExample {
    public static void main(String[] args) {
        // 模拟图像矩阵 / Simulate image matrix
        int height = 100;
        int width = 100;
        float[][] imageData = new float[height][width];
        
        // 生成测试图像数据 / Generate test image data
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                imageData[i][j] = (float) Math.sin(i * 0.1) * (float) Math.cos(j * 0.1);
            }
        }
        
        IMatrix image = IMatrix.of(imageData);
        
        // 图像变换 / Image transformations
        IMatrix image_transposed = image.transpose();        // 转置
        IMatrix image_scaled = image.mmul(2.0f);            // 缩放
        IMatrix image_shifted = image.add(0.5f);            // 亮度调整
        
        // 计算图像统计信息 / Compute image statistics
        float mean = image.mean();
        float std = image.std();
        float max = image.max();
        float min = image.min();
        
        System.out.println("图像均值: " + mean);
        System.out.println("图像标准差: " + std);
        System.out.println("图像最大值: " + max);
        System.out.println("图像最小值: " + min);
    }
}
```

## 性能优化示例 / Performance Optimization Examples

### 内存优化 / Memory Optimization

```java
public class MemoryOptimizationExample {
    public static void main(String[] args) {
        // 避免创建过多临时矩阵 / Avoid creating too many temporary matrices
        IMatrix matrix = IMatrix.of(new float[][]{{1, 2}, {3, 4}});
        
        // 不好的做法 / Bad practice
        IMatrix temp1 = matrix.add(IMatrix.ones(2, 2));
        IMatrix temp2 = temp1.mmul(2.0f);
        IMatrix temp3 = temp2.transpose();
        IMatrix result1 = temp3.pow(2);
        
        // 好的做法 / Good practice
        IMatrix result2 = matrix.add(IMatrix.ones(2, 2))
                                .mmul(2.0f)
                                .transpose()
                                .pow(2);
        
        // 及时释放引用 / Release references promptly
        temp1 = null;
        temp2 = null;
        temp3 = null;
        
        System.out.println("结果1: " + result1);
        System.out.println("结果2: " + result2);
    }
}
```

### 批量操作 / Batch Operations

```java
public class BatchOperationsExample {
    public static void main(String[] args) {
        // 批量矩阵操作 / Batch matrix operations
        int batchSize = 1000;
        IMatrix[] matrices = new IMatrix[batchSize];
        
        // 创建批量矩阵 / Create batch matrices
        for (int i = 0; i < batchSize; i++) {
            matrices[i] = IMatrix.rand(10, 10);
        }
        
        // 批量处理 / Batch processing
        IMatrix[] results = new IMatrix[batchSize];
        for (int i = 0; i < batchSize; i++) {
            results[i] = matrices[i].transpose().mmul(2.0f);
        }
        
        // 计算批量统计 / Compute batch statistics
        float totalSum = 0;
        for (IMatrix result : results) {
            totalSum += result.sum();
        }
        
        System.out.println("批量处理完成，总求和: " + totalSum);
    }
}
```

## 错误处理和调试示例 / Error Handling and Debugging Examples

### 异常处理 / Exception Handling

```java
public class ExceptionHandlingExample {
    public static void main(String[] args) {
        try {
            // 尝试求逆 / Try to compute inverse
            IMatrix matrix = IMatrix.of(new float[][]{{1, 1}, {1, 1}}); // 奇异矩阵
            IMatrix inverse = matrix.inv(); // 会抛出异常
        } catch (ArithmeticException e) {
            System.out.println("矩阵不可逆: " + e.getMessage());
            // 使用伪逆 / Use pseudo-inverse instead
            IMatrix pseudoInverse = matrix.pinv();
            System.out.println("伪逆: " + pseudoInverse);
        }
        
        try {
            // 尝试对负数开方 / Try to compute square root of negative numbers
            IMatrix matrix = IMatrix.of(new float[][]{{-1, 2}, {3, -4}});
            IMatrix sqrt = matrix.sqrt(); // 会抛出异常
        } catch (ArithmeticException e) {
            System.out.println("无法对负数开方: " + e.getMessage());
            // 使用绝对值 / Use absolute values instead
            IMatrix abs = matrix.abs();
            IMatrix sqrt = abs.sqrt();
            System.out.println("绝对值开方: " + sqrt);
        }
    }
}
```

### 数值稳定性检查 / Numerical Stability Check

```java
public class NumericalStabilityExample {
    public static void main(String[] args) {
        IMatrix matrix = IMatrix.of(new float[][]{{1e-6, 1}, {1, 1e6}});
        
        // 检查条件数 / Check condition number
        float cond = matrix.cond();
        System.out.println("矩阵条件数: " + cond);
        
        if (cond > 1e10) {
            System.out.println("警告：矩阵条件数过大，可能数值不稳定");
        }
        
        // 检查行列式 / Check determinant
        float det = matrix.det();
        System.out.println("矩阵行列式: " + det);
        
        if (Math.abs(det) < 1e-10) {
            System.out.println("警告：矩阵接近奇异");
        }
    }
}
```

## 总结 / Summary

本文档展示了矩阵运算的全面使用方法，从基础操作到高级应用。建议在实际使用中：

1. **优先使用矩阵化操作而不是循环** / Prioritize matrix operations over loops
2. **合理利用矩阵分解进行数据压缩** / Reasonably use matrix decomposition for data compression
3. **注意内存使用，避免创建过多临时矩阵** / Pay attention to memory usage, avoid creating too many temporary matrices
4. **根据具体需求选择合适的矩阵操作方法** / Choose appropriate matrix operation methods based on specific requirements
5. **注意方法命名的混淆问题** / Pay attention to method naming confusion (e.g., `rowSums()` actually sums by columns)
6. **处理异常情况，确保数值稳定性** / Handle exceptional cases and ensure numerical stability
7. **使用链式操作提高代码可读性** / Use method chaining to improve code readability

---

**矩阵运算示例** - 从基础到高级，掌握矩阵操作的精髓！

**Matrix Operation Examples** - From basics to advanced, master the essence of matrix operations!
