# 矩阵运算示例 (Matrix Operation Examples)

## 概述 / Overview

本文档提供了 `IMatrix<T>` 泛型接口的详细使用示例，涵盖从基础操作到高级应用的各个方面。推荐使用 `Linalg` 工厂类来创建矩阵实例。

This document provides detailed usage examples for the `IMatrix<T>` generic interface, covering everything from basic operations to advanced applications. It is recommended to use the `Linalg` factory class to create matrix instances.

## 广播与相关 API 速查 / Quick reference: broadcasting and related APIs

与下文各节示例配合阅读，便于快速定位 `Float` / `Double` 矩阵与向量上的广播、逐元素运算、`.npy` 与实数 FFT 等入口。

Use this table alongside the sections below to find broadcasting, element-wise ops, `.npy` I/O, and real FFT entry points on `Float` / `Double` matrices and vectors.

| 能力 / Capability | 使用方式 / Usage |
|-------------------|------------------|
| 二维广播 / 2D broadcasting | **`IMatrix.broadcastShape`** / **`broadcastTo`**；逐元素运算为 **`IMatrix.broadcastElementWise`**（`double[][]` / `IMatrix<Double>` / `float[][]` 等）；两个操作数均为 **`IMatrix<Float>`** 时用 **`IFloatMatrix.broadcastElementWise`**（泛型擦除下需显式入口） / **`IMatrix.broadcastElementWise`** for `double[][]` / `IMatrix<Double>` / `float[][]`, etc.; for two **`IMatrix<Float>`** operands use **`IFloatMatrix.broadcastElementWise`** (explicit entry under type erasure). |
| 矩阵乘、转置、行和等 / Matrix multiply, transpose, row sums, etc. | `IMatrix#mmul`、`IMatrix#transpose`、`IMatrix#rowSums` 等（见 `IMatrix`） / See `IMatrix` for `#mmul`, `#transpose`, `#rowSums`, etc. |
| 一维选取、重复、布尔过滤 / 1D fancy index, repeat, boolean filter | `IVector#fancyGet`、`IVector#repeat`、`IVector#booleanGet`（见 `IVector` 与 [Vector-Examples.md](Vector-Examples.md)） / See `IVector` and [Vector-Examples.md](Vector-Examples.md). |
| 直方图、分箱、`polyfit`、`where` / Histogram, bins, `polyfit`, `where` | **`IVector`** 静态方法；`IDoubleVector` / `IFloatVector` 提供同名便捷入口；直方图结果为 **`IVector.HistogramResult`** / **`IVector`** static methods; **`IDoubleVector`** / **`IFloatVector`** convenience entry points; histogram type **`IVector.HistogramResult`**. |
| `.npy` 文件 / `.npy` files | `com.yishape.lab.math.util.NpyArrayIO` / **`NpyArrayIO`** |
| 实数 FFT / Real FFT | `RereFFT` |

```java
import com.yishape.lab.math.linalg.IFloatMatrix;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.util.NpyArrayIO;

IMatrix<Double> c = IMatrix.broadcastElementWise(a, b, Double::sum);
IMatrix<Float> cF = IFloatMatrix.broadcastElementWise(mF1, mF2, Double::sum);
IMatrix<Double> p = m1.mmul(m2);
IVector<Double> coef = (IVector<Double>) IVector.polyfit(xVec, yVec, 1);
IVector<Float> coefF = (IVector<Float>) IVector.polyfit(xVecF, yVecF, 1);
IMatrix<Double> roundTrip = NpyArrayIO.fromByteArray(NpyArrayIO.toByteArray(matrix));
```

单元测试：`src/test/java/com/yishape/lab/math/linalg/DenseDoubleArrayUtilitiesTest.java`。

Unit tests: `src/test/java/com/yishape/lab/math/linalg/DenseDoubleArrayUtilitiesTest.java`.

## 基础示例 / Basic Examples

### 矩阵创建和基本操作 / Matrix Creation and Basic Operations

```java
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;

import java.util.Arrays;
import java.util.List;

public class MatrixBasicExample {
    public static void main(String[] args) {
        // 推荐使用 Linalg 工厂类创建矩阵 / Recommended to use Linalg factory class
        double[][] data = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
        IMatrix<Double> matrix = Linalg.matrix(data);

        // 从 List<double[]> 创建（每行一个 double[]）/ Create from list of row arrays
        List<double[]> rowList = Arrays.asList(
                new double[]{1.0, 2.0, 3.0},
                new double[]{4.0, 5.0, 6.0},
                new double[]{7.0, 8.0, 9.0}
        );
        IMatrix<Double> matrix2 = Linalg.matrixFromDoubleList(rowList);

        // 从行向量数组创建：IMatrix.of 需要 IDoubleVector[]（体现与 Linalg 分工）/ Rows as IDoubleVector[]
        IDoubleVector[] rowVecs = new IDoubleVector[]{
                IDoubleVector.of(new double[]{1.0, 2.0, 3.0}),
                IDoubleVector.of(new double[]{4.0, 5.0, 6.0}),
                IDoubleVector.of(new double[]{7.0, 8.0, 9.0})
        };
        IMatrix<Double> matrixFromRows = IMatrix.of(rowVecs);

        // 创建特殊矩阵 / Create special matrices
        IMatrix<Double> ones = Linalg.ones(3, 3);
        IMatrix<Float> zeros = Linalg.zeros(3, 3, Float.class);
        IMatrix<Double> identity = Linalg.eye(3);
        IMatrix<Double> random = Linalg.rand(3, 3);
        IMatrix<Double> randomSeeded = Linalg.rand(3, 3, 12345L);
        IMatrix<Double> randomNormal = Linalg.randn(3, 3);
        IMatrix<Double> randomNormalSeeded = Linalg.randn(3, 3, 0.0, 1.0, 12345L);

        // 从包装类数组创建 / Create from wrapper arrays
        Double[][] doubleData = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
        IMatrix<Double> matrix4 = Linalg.matrix(doubleData);

        Float[][] floatData = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}};
        IMatrix<Float> matrix5 = Linalg.matrix(floatData);

        // 创建对角矩阵 / Create diagonal matrix
        IMatrix<Double> diagonal = Linalg.diag(new Double[]{1.0, 2.0, 3.0});

        // 从一维数组创建 / Create from 1D array
        IMatrix<Double> reshaped = Linalg.fromArray(new double[]{1.0, 2.0, 3.0, 4.0}, 2, 2);

        // 从文件加载（需存在可读矩阵文本文件）/ Load from file when present
        // IMatrix<Double> loaded = Linalg.load("matrix.txt");

        // 同形状矩阵逐元素平均：使用 IMatrix.average（两矩阵版本）
        // Linalg.average(多个矩阵) 面向「列向量数组」的高级用法，一般矩阵平均请用本方法
        IMatrix<Double> averaged = IMatrix.average(matrix, matrix2, Double.class);

        // 基本运算 / Basic operations
        IMatrix<Double> sum = matrix.add(ones);
        IMatrix<Double> diff = matrix.sub(ones);
        IMatrix<Double> transposed = matrix.transpose();

        System.out.println("矩阵: " + matrix);
        System.out.println("从行向量构建: " + matrixFromRows);
        System.out.println("转置: " + transposed);
        System.out.println("多矩阵平均: " + averaged);
    }
}
```

### 基本数学运算 / Basic Mathematical Operations

```java
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;

public class BasicMathOperationsExample {
    public static void main(String[] args) {
        // 创建矩阵 / Create matrices
        double[][] data1 = {{1.0, 2.0}, {3.0, 4.0}};
        double[][] data2 = {{5.0, 6.0}, {7.0, 8.0}};
        IMatrix<Double> matrix1 = Linalg.matrix(data1);
        IMatrix<Double> matrix2 = Linalg.matrix(data2);

        // 基本运算 / Basic operations
        IMatrix<Double> sum = matrix1.add(matrix2);                    // [[6.0, 8.0], [10.0, 12.0]]
        IMatrix<Double> diff = matrix1.sub(matrix2);                   // [[-4.0, -4.0], [-4.0, -4.0]]
        IMatrix<Double> matrixProduct = matrix1.mmul(matrix2);         // [[19.0, 22.0], [43.0, 50.0]]

        // 标量逐元素乘：multiplyScalar 与 mmul(标量) 等价（对齐 NumPy A*s）
        IMatrix<Double> scaled = matrix1.multiplyScalar(2.0);          // [[2.0, 4.0], [6.0, 8.0]]
        IMatrix<Double> scaled2 = matrix1.mmul(2.0);
        IMatrix<Double> shifted = matrix1.sub(10.0);                     // [[-9.0, -8.0], [-7.0, -6.0]]

        // Frobenius 内积（对应元素相乘再求和）/ Frobenius inner product
        Double frob = matrix1.dot(matrix2);

        // 元素级乘法（Hadamard）/ Element-wise product
        IMatrix<Double> hadamard = matrix1.multiply(matrix2);

        // Kronecker 积（numpy.kron）与展平外积（np.outer(ravel,ravel)）
        IMatrix<Double> kron = matrix1.kron(matrix2);
        IMatrix<Double> flatOuter = matrix1.outer(matrix2);

        System.out.println("矩阵加法结果: " + sum);
        System.out.println("矩阵乘法结果: " + matrixProduct);
        System.out.println("标量乘法结果: " + scaled);
        System.out.println("mmul(标量) 与上式相同: " + scaled2.get(0, 0));
        System.out.println("Frobenius 内积: " + frob);
        System.out.println("Hadamard 积: " + hadamard);
        System.out.println("Kronecker 形状: " + kron.rows() + "x" + kron.cols());
        System.out.println("展平外积形状: " + flatOuter.rows() + "x" + flatOuter.cols());
    }
}
```

### 矩阵变换 / Matrix Transformations

```java
public class MatrixTransformationsExample {
    public static void main(String[] args) {
        IMatrix<Double> matrix = Linalg.matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});

        // 转置 / Transpose
        IMatrix<Double> transposed = matrix.transpose();                // [[1.0, 3.0], [2.0, 4.0]]
        IMatrix<Double> transposedNew = matrix.transposeNew();         // 创建新对象
        IMatrix<Double> transposedShort = matrix.t();                  // 简写形式

        // 幂运算 / Power
        IMatrix<Double> squared = matrix.pow(2.0);                     // 元素级幂运算

        // 开方 / Square root
        IMatrix<Double> sqrt = matrix.sqrt();                           // 元素级开方
        
        // 求逆 / Inverse
        IMatrix<Double> inverse = matrix.inv();
        
        // 伪逆 / Pseudo-inverse
        IMatrix<Double> pseudoInverse = matrix.pinv();
        
        // 矩阵属性检查 / Matrix property checks
        boolean isSquare = matrix.isSquare();
        boolean isSymmetric = matrix.isSymmetric();
        boolean isPositiveDefinite = matrix.isPositiveDefinite();
        
        // 矩阵条件数和秩 / Matrix condition number and rank
        Double conditionNumber = matrix.cond();
        int rank = matrix.rank();

        System.out.println("转置: " + transposed);
        System.out.println("平方: " + squared);
        System.out.println("开方: " + sqrt);
        System.out.println("是否为方阵: " + isSquare);
        System.out.println("是否为对称矩阵: " + isSymmetric);
        System.out.println("是否为正定矩阵: " + isPositiveDefinite);
        System.out.println("条件数: " + conditionNumber);
        System.out.println("矩阵秩: " + rank);
    }
}
```

### 线性代数运算 / Linear Algebra Operations

#### 特征分解 / Eigendecomposition

```java
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.util.Tuple2;

public class EigendecompositionExample {
    public static void main(String[] args) {
        // 特征分解 / Eigendecomposition
        double[][] data = {{4.0, -2.0}, {-2.0, 4.0}};
        IMatrix<Double> matrix = Linalg.matrix(data);
        
        Tuple2<IVector<Double>, IMatrix<Double>> eigenResult = matrix.eigen();
        IVector<Double> eigenValues = eigenResult._1;
        IMatrix<Double> eigenVectors = eigenResult._2;
        
        System.out.println("特征值: " + eigenValues);
        System.out.println("特征向量: " + eigenVectors);
        
        // 验证：A * v = λ * v
        IVector<Double> eigenVector1 = eigenVectors.getColumn(0);
        IVector<Double> result = matrix.mmul(eigenVector1);
        // result ≈ λ * eigenVector1

        // qrEigen() / qrEigenDecomposition() 与 eigen() 相同；与 matrix.qr()（A=QR 因子分解）完全不同
        Tuple2<IVector<Double>, IMatrix<Double>> evShort = matrix.qrEigen();
        System.out.println("qrEigen 特征值: " + evShort._1);
        System.out.println("qrEigenDecomposition 特征值: " + matrix.qrEigenDecomposition()._1);
    }
}
```

#### 奇异值分解 / Singular Value Decomposition

```java
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.util.Tuple3;

public class SVDExample {
    public static void main(String[] args) {
        double[][] data = {{4.0, -2.0}, {-2.0, 4.0}};
        IMatrix<Double> matrix = Linalg.matrix(data);
        
        // SVD分解 / SVD decomposition
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> svdResult = matrix.svd();
        IMatrix<Double> U = svdResult._1;                  // 左奇异向量 / Left singular vectors
        IVector<Double> S = svdResult._2;                  // 奇异值 / Singular values
        IMatrix<Double> V = svdResult._3;                  // 右奇异向量 / Right singular vectors
        
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
        IMatrix<Double> matrix = Linalg.matrix(new double[][]{{2.0, 1.0}, {1.0, 3.0}});
        
        // QR 因子分解 A=QR（非特征分解）/ QR factorization, not eigendecomposition
        Tuple2<IMatrix<Double>, IMatrix<Double>> qrResult = matrix.qr();
        IMatrix<Double> Q = qrResult._1;                   // 正交矩阵 / Orthogonal matrix
        IMatrix<Double> R = qrResult._2;                   // 上三角矩阵 / Upper triangular matrix

        // LU分解 / LU decomposition
        Tuple2<IMatrix<Double>, IMatrix<Double>> luResult = matrix.lu();
        IMatrix<Double> L = luResult._1;                   // 下三角矩阵 / Lower triangular matrix
        IMatrix<Double> U = luResult._2;                   // 上三角矩阵 / Upper triangular matrix

        // Cholesky分解 / Cholesky decomposition
        IMatrix<Double> L_chol = matrix.cholesky();        // 下三角矩阵L，A = L * L^T
        
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
        IMatrix<Double> A = Linalg.matrix(new double[][]{{2.0, 1.0}, {1.0, 3.0}});
        IVector<Double> b = Linalg.vector(new double[]{5.0, 6.0});

        // LU分解 / LU decomposition
        Tuple2<IMatrix<Double>, IMatrix<Double>> luResult = A.lu();
        IMatrix<Double> L = luResult._1;  // 下三角矩阵
        IMatrix<Double> U = luResult._2;  // 上三角矩阵

        // 求解线性方程组 / Solve linear system
        IVector<Double> x = A.solve(b);   // 求解Ax = b

        // 验证结果 / Verify result
        IVector<Double> Ax = A.mmul(x);   // 应该等于b
        
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
        IMatrix<Double> matrix = Linalg.matrix(new double[][]{{2.0, 1.0}, {1.0, 3.0}});
        
        // 行列式 / Determinant
        Double det = matrix.det();

        // 迹 / Trace
        Double trace = matrix.trace();

        // 秩 / Rank
        int rank = matrix.rank();

        // 条件数 / Condition number
        Double cond = matrix.cond();
        
        // 矩阵范数 / Matrix norms
        Double frobeniusNorm = matrix.frobeniusNorm();
        Double frobeniusDistance = matrix.frobeniusDistance(matrix);
        
        // 矩阵展平和重塑 / Matrix flattening and reshaping
        IVector<Double> flattened = matrix.flatten();
        IMatrix<Double> reshaped = matrix.reshape(4, 1);
        
        System.out.println("行列式: " + det);
        System.out.println("迹: " + trace);
        System.out.println("秩: " + rank);
        System.out.println("条件数: " + cond);
        System.out.println("Frobenius范数: " + frobeniusNorm);
        System.out.println("Frobenius距离: " + frobeniusDistance);
        System.out.println("展平向量: " + flattened);
        System.out.println("重塑矩阵: " + reshaped);
    }
}
```

### 统计运算 / Statistical Operations

```java
public class MatrixStatisticsExample {
    public static void main(String[] args) {
        double[][] data = {
            {12.5, 15.2, 18.7},
            {13.9, 17.1, 19.4},
            {14.8, 16.5, 20.1}
        };
        
        IMatrix<Double> matrix = Linalg.matrix(data);
        
        // 行统计 / Row statistics (注意：实际是按列求和)
        IVector<Double> rowSums = matrix.rowSums();        // 行和（按列求和）
        IVector<Double> rowMeans = matrix.rowMeans();      // 行均值（按列求均值）
        
        // 列统计 / Column statistics (注意：实际是按行求和)
        IVector<Double> colSums = matrix.colSums();        // 列和（按行求和）
        IVector<Double> colMeans = matrix.colMeans();      // 列均值（按行求均值）
        
        // 整体统计 / Overall statistics
        Double totalSum = matrix.sum();             // 总和
        Double totalMean = matrix.mean();           // 整体均值
        Double variance = matrix.var();             // 整体方差
        Double stdDev = matrix.std();               // 整体标准差
        Double max = matrix.max();                  // 最大值
        Double min = matrix.min();                  // 最小值
        
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
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.util.Arrays;

public class DataAccessExample {
    public static void main(String[] args) {
        IMatrix<Double> matrix = Linalg.matrix(new double[][]{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
        
        // 获取行 / Get row
        IVector<Double> row = matrix.getRow(0);

        // 获取列 / Get column（向量形式 n；矩阵形式 n×1 用 getColumnMatrix / getColumnAsCloumnVector）
        IVector<Double> column = matrix.getColumn(0);
        IMatrix<Double> columnMatrix = matrix.getColumnMatrix(0);

        // 设置列 / Set column（向量用 setColumn；矩阵形式列用 putColumn）
        IVector<Double> newColumn = Linalg.vector(new double[]{10.0, 11.0, 12.0});
        matrix.setColumn(0, newColumn);

        // 获取多个列 / Get multiple columns（返回列向量数组）
        IVector<Double>[] columnVecs = matrix.getColumns(new int[]{0, 2});

        // 获取形状 / Get shape
        int[] shape = matrix.shape();             // [行数, 列数]
        int rows = matrix.rows();                 // 行数
        int cols = matrix.cols();                 // 列数
        
        // 获取和设置元素 / Get and set elements
        Double element = matrix.get(1, 2);
        matrix.put(1, 2, 10.0);
        
        // 负数索引访问 / Negative indexing access
        Double lastElement = matrix.get(-1, -1);  // 获取最后一个元素
        matrix.put(-1, -1, 100.0);               // 设置最后一个元素
        
        System.out.println("第0行: " + row);
        System.out.println("第0列(向量): " + column);
        System.out.println("第0列(n×1矩阵)行数: " + columnMatrix.rows());
        System.out.println("列向量数组长度: " + columnVecs.length);
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
        IMatrix<Double> matrix = Linalg.matrix(new double[][]{
            {1.0, 2.0, 3.0, 4.0, 5.0},
            {6.0, 7.0, 8.0, 9.0, 10.0},
            {11.0, 12.0, 13.0, 14.0, 15.0},
            {16.0, 17.0, 18.0, 19.0, 20.0},
            {21.0, 22.0, 23.0, 24.0, 25.0}
        });
        
        System.out.println("原始矩阵:");
        System.out.println(matrix);
        System.out.println("矩阵形状: " + Arrays.toString(matrix.shape()));
        
        System.out.println("\n=== 基本切片表达式 / Basic Slice Expressions ===");
        
        // 基本切片 / Basic slicing
        IMatrix<Double> slice1 = matrix.slice("1:3", "1:3");  // 行1-2，列1-2
        System.out.println("slice(\"1:3\", \"1:3\"):");
        System.out.println(slice1);
        
        IMatrix<Double> slice2 = matrix.slice("2:", "2:");    // 从行2到末尾，从列2到末尾
        System.out.println("slice(\"2:\", \"2:\"):");
        System.out.println(slice2);
        
        IMatrix<Double> slice3 = matrix.slice(":3", ":3");    // 从开始到行2，从开始到列2
        System.out.println("slice(\":3\", \":3\"):");
        System.out.println(slice3);
        
        IMatrix<Double> slice4 = matrix.slice("::2", "::2");  // 每隔一行一列
        System.out.println("slice(\"::2\", \"::2\"):");
        System.out.println(slice4);
        
        IMatrix<Double> slice5 = matrix.slice("1:4:2", "1:4:2"); // 行1,3，列1,3
        System.out.println("slice(\"1:4:2\", \"1:4:2\"):");
        System.out.println(slice5);
        
        System.out.println("\n=== 负数索引切片 / Negative Indexing Slicing ===");
        
        // 负数索引切片 / Negative indexing slicing
        IMatrix<Double> slice6 = matrix.slice("-2:", "-2:");  // 最后两行，最后两列
        System.out.println("slice(\"-2:\", \"-2:\"):");
        System.out.println(slice6);
        
        IMatrix<Double> slice7 = matrix.slice(":-2", ":-2");  // 除最后两行外的所有行，除最后两列外的所有列
        System.out.println("slice(\":-2\", \":-2\"):");
        System.out.println(slice7);
        
        IMatrix<Double> slice8 = matrix.slice("-3:-1", "-3:-1"); // 倒数第3-2行，倒数第3-2列
        System.out.println("slice(\"-3:-1\", \"-3:-1\"):");
        System.out.println(slice8);
        
        IMatrix<Double> slice9 = matrix.slice("-4:-1:2", "-4:-1:2"); // 倒数第4-2行，倒数第4-2列，步长为2
        System.out.println("slice(\"-4:-1:2\", \"-4:-1:2\"):");
        System.out.println(slice9);
        
        IMatrix<Double> slice10 = matrix.slice("::-1", "::-1"); // 反转向量（行列都反转）
        System.out.println("slice(\"::-1\", \"::-1\"):");
        System.out.println(slice10);
        
        System.out.println("\n=== 行切片和列切片 / Row and Column Slicing ===");
        
        // 行切片 / Row slicing
        IMatrix<Double> rowSlice1 = matrix.sliceRows("1:3");   // 行1-2
        System.out.println("sliceRows(\"1:3\"):");
        System.out.println(rowSlice1);
        
        IMatrix<Double> rowSlice2 = matrix.sliceRows("-2:");   // 最后两行
        System.out.println("sliceRows(\"-2:\"):");
        System.out.println(rowSlice2);
        
        IMatrix<Double> rowSlice3 = matrix.sliceRows("::2");   // 每隔一行
        System.out.println("sliceRows(\"::2\"):");
        System.out.println(rowSlice3);
        
        // 列切片 / Column slicing
        IMatrix<Double> colSlice1 = matrix.sliceColumns("1:3"); // 列1-2
        System.out.println("sliceColumns(\"1:3\"):");
        System.out.println(colSlice1);
        
        IMatrix<Double> colSlice2 = matrix.sliceColumns("-2:"); // 最后两列
        System.out.println("sliceColumns(\"-2:\"):");
        System.out.println(colSlice2);
        
        IMatrix<Double> colSlice3 = matrix.sliceColumns("::2"); // 每隔一列
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
        IMatrix<Double> slice11 = matrix.slice("0:0", "0:0"); // 空切片
        System.out.println("slice(\"0:0\", \"0:0\"): " + slice11 + " (形状: " + Arrays.toString(slice11.shape()) + ")");
        
        IMatrix<Double> slice12 = matrix.slice("2:2", "2:2"); // 空切片
        System.out.println("slice(\"2:2\", \"2:2\"): " + slice12 + " (形状: " + Arrays.toString(slice12.shape()) + ")");
        
        IMatrix<Double> slice13 = matrix.slice("-1:-1", "-1:-1"); // 空切片
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
        IMatrix<Double> center = matrix.slice("1:-1", "1:-1");
        System.out.println("中心部分（去除边界）:");
        System.out.println(center);
        
        // 获取四个角落 / Get four corners
        IMatrix<Double> topLeft = matrix.slice(":2", ":2");
        IMatrix<Double> topRight = matrix.slice(":2", "-2:");
        IMatrix<Double> bottomLeft = matrix.slice("-2:", ":2");
        IMatrix<Double> bottomRight = matrix.slice("-2:", "-2:");
        
        System.out.println("左上角:");
        System.out.println(topLeft);
        System.out.println("右上角:");
        System.out.println(topRight);
        System.out.println("左下角:");
        System.out.println(bottomLeft);
        System.out.println("右下角:");
        System.out.println(bottomRight);
        
        // 获取对角线元素 / Get diagonal elements
        IMatrix<Double> diagonal = matrix.slice("::1", "::1"); // 这里需要特殊处理对角线
        System.out.println("对角线（需要特殊处理）:");
        System.out.println(diagonal);
        
        // 获取奇数行和偶数列 / Get odd rows and even columns
        IMatrix<Double> oddRowsEvenCols = matrix.slice("1::2", "::2");
        System.out.println("奇数行和偶数列:");
        System.out.println(oddRowsEvenCols);
        
        // 获取最后几行和最后几列 / Get last few rows and columns
        IMatrix<Double> lastFew = matrix.slice("-3:", "-3:");
        System.out.println("最后三行三列:");
        System.out.println(lastFew);
    }
}
```

### 数据预处理 / Data Preprocessing

```java
public class DataPreprocessingExample {
    public static void main(String[] args) {
        IMatrix<Double> data = Linalg.matrix(new double[][]{
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0}
        });

        // 中心化 / Centering
        IMatrix<Double> centered = data.center();                       // 每列减去列均值

        // 协方差矩阵 / Covariance matrix
        IMatrix<Double> covMatrix = data.covariance();                  // 从原始数据计算协方差
        IMatrix<Double> covFromCentered = data.covarianceFromCentered(); // 从中心化数据计算协方差

        // 归一化 / Normalization
        IMatrix<Double> normalizedRows = data.normalizeRows();          // L2行归一化
        IMatrix<Double> normalizedCols = data.normalizeColumns();       // L2列归一化
        
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
        IMatrix<Double> matrix = Linalg.matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});

        // 三角函数 / Trigonometric functions
        IMatrix<Double> sinResult = matrix.sin();          // 正弦
        IMatrix<Double> cosResult = matrix.cos();          // 余弦
        IMatrix<Double> tanResult = matrix.tan();          // 正切

        // 双曲函数 / Hyperbolic functions
        IMatrix<Double> sinhResult = matrix.sinh();        // 双曲正弦
        IMatrix<Double> coshResult = matrix.cosh();        // 双曲余弦
        IMatrix<Double> tanhResult = matrix.tanh();        // 双曲正切

        // 指数和对数 / Exponential and logarithm
        IMatrix<Double> expResult = matrix.exp();          // 指数
        IMatrix<Double> logResult = matrix.log();          // 自然对数

        // 绝对值和符号 / Absolute value and sign
        IMatrix<Double> absResult = matrix.abs();          // 绝对值
        IMatrix<Double> signResult = matrix.sign();        // 符号函数
        
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
        IMatrix<Double> A = Linalg.matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        IMatrix<Double> B = Linalg.matrix(new double[][]{{5.0, 6.0}, {7.0, 8.0}});

        // 水平拼接 / Horizontal concatenation
        IMatrix<Double> hstacked = A.hstack(B);  // [[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]]

        // 垂直拼接 / Vertical concatenation
        IMatrix<Double> vstacked = A.vstack(B);  // [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]

        // 分割矩阵 / Split matrix
        IMatrix<Double>[] hsplit = hstacked.hsplit(new int[]{2});  // 在第2列处分割
        IMatrix<Double>[] vsplit = vstacked.vsplit(new int[]{2});  // 在第2行处分割

        // 矩阵重塑 / Matrix reshape
        IMatrix<Double> reshaped = A.reshape(4, 1);  // 重塑为4x1

        // 矩阵复制 / Matrix copy
        IMatrix<Double> copied = A.copy();            // 深拷贝

        // 矩阵范数 / Matrix norms
        Double frobeniusNorm = A.frobeniusNorm(); // Frobenius范数
        Double frobeniusDistance = A.frobeniusDistance(B); // Frobenius距离
        
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
        IMatrix<Double> matrix = Linalg.matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        // matrix.save("matrix.txt");  // 需要实现保存方法

        // 从文件加载矩阵 / Load matrix from file
        IMatrix<Double> loaded = Linalg.load("matrix.txt");
        
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
        IMatrix<Double> result = Linalg.ones(3, 3)                     // 创建3x3全1矩阵
            .sub(5.0)                                                  // 每个元素减5
            .multiplyScalar(2.0)                                                 // 每个元素乘2
            .transpose()                                               // 转置
            .pow(2);                                                   // 每个元素平方
            
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
        
        double[][] lowRankData = new double[rows][cols];
        // 生成低秩矩阵数据 / Generate low-rank matrix data
        
        IMatrix<Double> originalMatrix = Linalg.matrix(lowRankData);
        
        // SVD分解 / SVD decomposition
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> svdResult = originalMatrix.svd();
        
        // 不同秩的重构 / Reconstruction with different ranks
        for (int testRank = 1; testRank <= 20; testRank += 5) {
            // 使用前testRank个奇异值重构
            IMatrix<Double> U = svdResult._1;
            IVector<Double> S = svdResult._2;
            IMatrix<Double> V = svdResult._3;
            
            // 重构矩阵
            IMatrix<Double> reconstructed = U.mmul(Linalg.diag(S.toDoubleArray())).mmul(V);
            
            // 计算重构误差
            Double error = originalMatrix.frobeniusDistance(reconstructed);
            
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
        IMatrix<Double> X = Linalg.matrix(new double[][]{
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0}
        });
        
        // 标签向量 / Label vector
        IVector<Double> y = Linalg.vector(new double[]{1.0, 2.0, 3.0});
        
        // 数据预处理 / Data preprocessing
        IMatrix<Double> X_centered = X.center();                    // 中心化
        IMatrix<Double> X_normalized = X.normalizeColumns();        // 列归一化
        
        // 计算协方差矩阵 / Compute covariance matrix
        IMatrix<Double> covMatrix = X_centered.covarianceFromCentered();
        
        // 特征分解 / Eigendecomposition
        Tuple2<IVector<Double>, IMatrix<Double>> eigenResult = covMatrix.eigen();
        IVector<Double> eigenValues = eigenResult._1;
        IMatrix<Double> eigenVectors = eigenResult._2;
        
        // PCA降维 / PCA dimensionality reduction
        IMatrix<Double> X_pca = X_centered.mmul(eigenVectors);
        
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
        double[][] imageData = new double[height][width];
        
        // 生成测试图像数据 / Generate test image data
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                imageData[i][j] = Math.sin(i * 0.1) * Math.cos(j * 0.1);
            }
        }
        
        IMatrix<Double> image = Linalg.matrix(imageData);
        
        // 图像变换 / Image transformations
        IMatrix<Double> image_transposed = image.transpose();        // 转置
        IMatrix<Double> image_scaled = image.multiplyScalar(2.0);             // 缩放
        IMatrix<Double> image_shifted = image.add(0.5);             // 亮度调整
        
        // 计算图像统计信息 / Compute image statistics
        Double mean = image.mean();
        Double std = image.std();
        Double max = image.max();
        Double min = image.min();
        
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
        IMatrix<Double> matrix = Linalg.matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        
        // 不好的做法 / Bad practice
        IMatrix<Double> temp1 = matrix.add(Linalg.ones(2, 2));
        IMatrix<Double> temp2 = temp1.multiplyScalar(2.0);
        IMatrix<Double> temp3 = temp2.transpose();
        IMatrix<Double> result1 = temp3.pow(2);
        
        // 好的做法 / Good practice
        IMatrix<Double> result2 = matrix.add(Linalg.ones(2, 2))
                                        .multiplyScalar(2.0)
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
        IMatrix<Double>[] matrices = new IMatrix[batchSize];
        
        // 创建批量矩阵 / Create batch matrices
        for (int i = 0; i < batchSize; i++) {
            matrices[i] = Linalg.rand(10, 10);
        }
        
        // 批量处理 / Batch processing
        IMatrix<Double>[] results = new IMatrix[batchSize];
        for (int i = 0; i < batchSize; i++) {
            results[i] = matrices[i].transpose().multiplyScalar(2.0);
        }
        
        // 计算批量统计 / Compute batch statistics
        double totalSum = 0;
        for (IMatrix<Double> result : results) {
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
            IMatrix<Double> matrix = Linalg.matrix(new double[][]{{1.0, 1.0}, {1.0, 1.0}}); // 奇异矩阵
            IMatrix<Double> inverse = matrix.inv(); // 会抛出异常
        } catch (ArithmeticException e) {
            System.out.println("矩阵不可逆: " + e.getMessage());
            // 使用伪逆 / Use pseudo-inverse instead
            IMatrix<Double> pseudoInverse = matrix.pinv();
            System.out.println("伪逆: " + pseudoInverse);
        }
        
        try {
            // 尝试对负数开方 / Try to compute square root of negative numbers
            IMatrix<Double> matrix = Linalg.matrix(new double[][]{{-1.0, 2.0}, {3.0, -4.0}});
            IMatrix<Double> sqrt = matrix.sqrt(); // 会抛出异常
        } catch (ArithmeticException e) {
            System.out.println("无法对负数开方: " + e.getMessage());
            // 使用绝对值 / Use absolute values instead
            IMatrix<Double> abs = matrix.abs();
            IMatrix<Double> sqrt = abs.sqrt();
            System.out.println("绝对值开方: " + sqrt);
        }
    }
}
```

### 数值稳定性检查 / Numerical Stability Check

```java
public class NumericalStabilityExample {
    public static void main(String[] args) {
        IMatrix<Double> matrix = Linalg.matrix(new double[][]{{1e-6, 1.0}, {1.0, 1e6}});
        
        // 检查条件数 / Check condition number
        Double cond = matrix.cond();
        System.out.println("矩阵条件数: " + cond);
        
        if (cond > 1e10) {
            System.out.println("警告：矩阵条件数过大，可能数值不稳定");
        }
        
        // 检查行列式 / Check determinant
        Double det = matrix.det();
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
5. **使用Linalg工厂类创建矩阵实例** / Use Linalg factory class to create matrix instances
6. **注意泛型类型的使用，确保类型一致性** / Pay attention to generic type usage, ensure type consistency
7. **处理异常情况，确保数值稳定性** / Handle exceptional cases and ensure numerical stability
8. **使用链式操作提高代码可读性** / Use method chaining to improve code readability

---

**矩阵运算示例** - 从基础到高级，掌握矩阵操作的精髓！

**Matrix Operation Examples** - From basics to advanced, master the essence of matrix operations!
