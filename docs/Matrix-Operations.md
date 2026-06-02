# 矩阵操作 (Matrix Operations)

## 概述 / Overview

`IMatrix` 接口提供了完整的泛型矩阵数学运算功能，支持 Float 和 Double 类型，包括基本数学运算、矩阵变换、线性代数运算、特征分解等。该接口采用泛型设计，支持链式操作，提供丰富的矩阵操作功能。

The `IMatrix` interface provides comprehensive generic matrix mathematical operations supporting Float and Double types, including basic mathematical operations, matrix transformations, linear algebra operations, eigendecomposition, and more. The interface uses generic design, supports method chaining, and offers rich matrix operation functionalities.

广播、`.npy` 等与矩阵相关的常用入口速查见 [`examples/Matrix-Examples.md`](examples/Matrix-Examples.md) 文首。

For a quick table of broadcasting, `.npy` I/O, and related matrix entry points, see the opening section of [`examples/Matrix-Examples.md`](examples/Matrix-Examples.md).

## 核心接口 / Core Interface

### IMatrix 接口 / IMatrix Interface

`IMatrix<T>` 是矩阵操作的核心泛型接口，定义了所有矩阵运算的抽象方法，支持 `Float` 和 `Double` 类型。

`IMatrix<T>` is the core generic interface for matrix operations, defining abstract methods for all matrix operations, supporting `Float` and `Double` types.

### Linalg 工厂类 / Linalg Factory Class

`Linalg` 类提供了统一的矩阵和向量创建入口，推荐使用此类来创建矩阵实例。该类采用委托模式，将创建请求委托给相应的接口实现。

`Linalg` class provides a unified entry point for matrix and vector creation, recommended for creating matrix instances. This class uses delegation pattern, delegating creation requests to corresponding interface implementations.

**架构设计 / Architecture Design:**
- **委托链** / **Delegation chain**: `Linalg` → `IMatrix` → `IFloatMatrix/IDoubleMatrix`
- **类型推断** / **Type inference**: 根据输入数据类型自动选择合适的实现
- **API统一** / **API consistency**: 提供一致的命名和使用模式

## 主要功能 / Main Features

### 1. 矩阵创建 / Matrix Creation

#### 静态工厂方法 / Static Factory Methods

```java
// 推荐使用 Linalg 工厂方法 / Recommended to use Linalg factory methods

import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.util.Arrays;
import java.util.List;

// 从二维数组创建 / Create from 2D array
double[][] data = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
IMatrix<Double> matrix1 = Linalg.matrix(data);

float[][] floatData = {{1f, 2f, 3f}, {4f, 5f, 6f}, {7f, 8f, 9f}};
IMatrix<Float> matrix1f = Linalg.matrix(floatData);

// 从包装类数组创建 / Create from wrapper arrays
Double[][] doubleData = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
IMatrix<Double> matrix2 = Linalg.matrix(doubleData);

Float[][] floatData2 = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}};
IMatrix<Float> matrix2f = Linalg.matrix(floatData2);

// 从List创建 / Create from List
List<double[]> rows = Arrays.asList(
        new double[]{1, 2, 3},
        new double[]{4, 5, 6},
        new double[]{7, 8, 9}
);
IMatrix<Double> matrix3 = Linalg.matrixFromDoubleList(rows);

List<float[]> floatRows = Arrays.asList(
        new float[]{1f, 2f, 3f},
        new float[]{4f, 5f, 6f},
        new float[]{7f, 8f, 9f}
);
IMatrix<Float> matrix3f = Linalg.matrixFromFloatList(floatRows);

// 从行向量数组创建（需 IDoubleVector[]；配合 import com.yishape.lab.math.linalg.IDoubleVector）
IDoubleVector[] rowVecs = {
    IDoubleVector.of(new double[]{1, 2}),
    IDoubleVector.of(new double[]{3, 4})
};
IMatrix<Double> matrix4 = IMatrix.of(rowVecs);


// 创建特殊矩阵 / Create special matrices
IMatrix<Double> ones = Linalg.ones(3, 3);        // 全1矩阵 / Matrix of ones
IMatrix<Float> onesF = Linalg.ones(3, 3, Float.class); // Float类型全1矩阵
IMatrix<Double> zeros = Linalg.zeros(3, 3);      // 零矩阵 / Zero matrix
IMatrix<Float> zerosF = Linalg.zeros(3, 3, Float.class); // Float类型零矩阵
IMatrix<Double> identity = Linalg.eye(3);        // 单位矩阵 / Identity matrix
IMatrix<Float> identityF = Linalg.eye(3, Float.class); // Float类型单位矩阵

// 随机矩阵 / Random matrices
IMatrix<Double> random = Linalg.rand(3, 3);      // 随机矩阵（均匀分布） / Random matrix (uniform distribution)
IMatrix<Float> randomF = Linalg.rand(3, 3, Float.class); // Float类型随机矩阵
IMatrix<Double> randomSeeded = Linalg.rand(3, 3, 12345L); // 指定种子的随机矩阵 / Random matrix with seed
IMatrix<Double> randomNormal = Linalg.randn(3, 3); // 随机矩阵（正态分布） / Random matrix (normal distribution)
IMatrix<Float> randomNormalF = Linalg.randn(3, 3, Float.class); // Float类型正态随机矩阵
IMatrix<Double> randomNormalSeeded = Linalg.randn(3, 3, 0.0, 1.0, 12345L); // 指定种子的正态随机矩阵

// 创建 Toeplitz 矩阵 / Create Toeplitz matrix
double[] c = {5.0, 2.0, 1.0};                          // 第一列
IMatrix<Double> toeplitz = Linalg.toeplitz(c);          // 对称 Toeplitz: T[i][j] = c[|i-j|]
IMatrix<Double> toeplitz2 = Linalg.toeplitz(column, row); // 非对称 Toeplitz
IMatrix<Double> toeplitz3 = Linalg.toeplitz(vectorColumn); // IVector 重载

// 创建对角矩阵 / Create diagonal matrix
IMatrix<Double> diagonal = Linalg.diag(new Double[]{1.0, 2.0, 3.0});
IMatrix<Float> diagonalF = Linalg.diag(new Float[]{1.0f, 2.0f, 3.0f});
IMatrix<Double> diagonalFromArray = Linalg.diag(new double[]{1.0, 2.0, 3.0});
IMatrix<Float> diagonalFromArrayF = Linalg.diag(new float[]{1.0f, 2.0f, 3.0f});

// 从一维数组创建 / Create from 1D array
IMatrix<Double> reshaped = Linalg.fromArray(new double[]{1, 2, 3, 4}, 2, 2);
IMatrix<Float> reshapedF = Linalg.fromArray(new float[]{1f, 2f, 3f, 4f}, 2, 2);

// 从文件加载 / Load from file
IMatrix<Double> loaded = Linalg.load("matrix.txt");
IMatrix<Float> loadedF = Linalg.load("matrix.txt", Float.class);

// 两矩阵逐元素平均 / Element-wise average of two matrices（同维度）
IMatrix<Double> averaged = IMatrix.average(matrix1, matrix2, Double.class);
```

### 2. 基本数学运算 / Basic Mathematical Operations

#### 矩阵间运算 / Matrix-to-Matrix Operations

```java
// 加法 / Addition
IMatrix<Double> sum = matrix1.add(matrix2);

// 减法 / Subtraction
IMatrix<Double> diff = matrix1.sub(matrix2);

// 矩阵乘法 / Matrix multiplication
IMatrix<Double> matrixProduct = matrix1.mmul(matrix2);

// 矩阵与向量乘法 / Matrix-vector multiplication
IVector<Double> vector = Linalg.vector(new double[]{1.0, 2.0});
IVector<Double> matrixVectorProduct = matrix1.mmul(vector);

// 元素级除法 / Element-wise division
IMatrix<Double> quotient = matrix1.divide(matrix2);

// Kronecker 积（NumPy numpy.kron）/ Kronecker product
IMatrix<Double> k = matrix1.kron(matrix2);
// 或 / or: IMatrix<Double> k2 = Linalg.kron(matrix1, matrix2);

// 展平后外积（NumPy np.outer(A.ravel(), B.ravel())，行优先）/ Outer after row-major flatten
IMatrix<Double> o = matrix1.outer(matrix2);
```

#### 标量运算 / Scalar Operations

```java
// 标量减法 / Scalar subtraction
IMatrix<Double> result1 = matrix1.sub(5.0);

// 标量乘法 / Scalar multiplication（与 multiplyScalar 等价，对齐 ndarray * scalar）
IMatrix<Double> result2 = matrix1.multiplyByScalar(3.0);
IMatrix<Double> result2b = matrix1.mmul(3.0);

// Frobenius 内积（逐元素相乘再求和）/ Frobenius inner product
Double frob = matrix1.frobeniusInnerProduct(matrix2);

// 元素级乘法（Hadamard）/ Element-wise multiply
IMatrix<Double> hadamard = matrix1.multiply(matrix2);
```

### 3. 矩阵变换 / Matrix Transformations

```java
// 转置 / Transpose
IMatrix<Double> transposed = matrix1.transpose();      // 创建新对象
IMatrix<Double> transposedNew = matrix1.transposeNew(); // 创建新对象
IMatrix<Double> transposedShort = matrix1.t();         // 简写形式

// 幂运算 / Power
IMatrix<Double> squared = matrix1.pow(2.0);

// 开方 / Square root
IMatrix<Double> sqrt = matrix1.sqrt();

// 求逆 / Inverse
IMatrix<Double> inverse = matrix1.inv();

// 伪逆 / Pseudo-inverse
IMatrix<Double> pseudoInverse = matrix1.pinv();

// 矩阵条件数 / Matrix condition number
Double conditionNumber = matrix1.cond();

// 矩阵秩 / Matrix rank
int rank = matrix1.rank();
```

### 4. 线性代数运算 / Linear Algebra Operations

#### 特征分解 / Eigendecomposition

```java
// 特征分解 / Eigendecomposition
Tuple2<IVector<Double>, IMatrix<Double>> eigenResult = matrix1.eigen();
IVector<Double> eigenValues = eigenResult._1;      // 特征值 / Eigenvalues
IMatrix<Double> eigenVectors = eigenResult._2;     // 特征向量 / Eigenvectors

// 与 eigen() 相同：QR 算法求特征值（非 qr() 因子分解）。简短名 qrEigen() 与 qrEigenDecomposition() 等价
Tuple2<IVector<Double>, IMatrix<Double>> byQrAlgo = matrix1.qrEigen();
Tuple2<IVector<Double>, IMatrix<Double>> byQrAlgoLong = matrix1.qrEigenDecomposition();

```

#### 奇异值分解 / Singular Value Decomposition

```java
// SVD分解 / SVD decomposition
Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> svdResult = matrix1.svd();
IMatrix<Double> U = svdResult._1;                  // 左奇异向量 / Left singular vectors
IVector<Double> S = svdResult._2;                  // 奇异值 / Singular values
IMatrix<Double> V = svdResult._3;                  // 右奇异向量 / Right singular vectors
```

**SVD 实现细节 / SVD Implementation Details**

本库 SVD 采用 **BLAS-3 分块双对角化（blocked bidiagonalization）** + **BD-SQR（QR 迭代）** 的两阶段算法，基于 LAPACK DGEBRD 模式实现：
The library uses a two-stage **BLAS-3 blocked bidiagonalization** + **BD-SQR (QR iteration)** algorithm based on the LAPACK DGEBRD pattern:

1. **第一阶段 / Phase 1 — 双对角化（Bidiagonalization）**: 通过 Panel reduction（`labrd`）+ WY 紧凑表示（`I - V·T·Vᵀ`）将矩阵约化为双对角形式，trailing matrix update 使用 BLAS-3 GEMM 内核以提高缓存效率。/ Panel reduction with WY compact representation reduces the matrix to bidiagonal form; trailing updates use BLAS-3 GEMM kernels for cache efficiency.
2. **第二阶段 / Phase 2 — BD-SQR**: 对双对角矩阵执行带 Wilkinson shift 的隐式 QR 迭代，求出奇异值和奇异向量。/ Implicit QR iteration with Wilkinson shift on the bidiagonal matrix yields singular values and vectors.
3. **并行化 / Parallelism**: Trailing update 和 U/V 构造中的 Step 3 已通过 `ForkJoinPool` 按行分区并行化。/ Trailing updates and U/V construction Step 3 are parallelized via `ForkJoinPool` row-range partitioning.

**特殊策略 / Special Strategies**:
- **宽矩阵自动转置 / Wide matrix auto-transpose**: 当 `m < n` 时，内部自动转置为 `n×m` 计算，再将结果映射回原始维度，以保证分块效率。/ When `m < n`, the matrix is internally transposed to `n×m` for efficient blocking, then mapped back.
- **小矩阵回退 / Small matrix fallback**: 维度小于 64 时自动走标量（BLAS-2）回退路径。/ Matrices smaller than 64 automatically fall back to a scalar (BLAS-2) path.
- **路径切换 / Path switching**: `RereSVDDecomposition.useFaerStyle`（默认 `true`）控制是否启用 BLAS-3 路径；设为 `false` 时走 legacy 标量实现。/ `RereSVDDecomposition.useFaerStyle` (default `true`) toggles the BLAS-3 path; `false` uses the legacy scalar implementation.

**数值稳定性参数 / Numerical Stability Parameters**:
| 参数 / Parameter | 默认值 / Default | 说明 / Description |
|------------------|------------------|-------------------|
| `epsilon` | `1e-10` | 零判断阈值 / Zero threshold |
| `maxIterations` | `n × 30` | BD-SQR 最大迭代次数 / Max QR iterations |
| `TINY` | `Math.ulp(1.0)` | 接近零值的安全替换 / Safe replacement for near-zeros |

#### 矩阵分解 / Matrix Decomposition

```java
// QR分解 / QR decomposition
Tuple2<IMatrix<Double>, IMatrix<Double>> qrResult = matrix1.qr();
IMatrix<Double> Q = qrResult._1;                   // 正交矩阵 / Orthogonal matrix
IMatrix<Double> R = qrResult._2;                   // 上三角矩阵 / Upper triangular matrix

// LU分解 / LU decomposition
Tuple2<IMatrix<Double>, IMatrix<Double>> luResult = matrix1.lu();
IMatrix<Double> L = luResult._1;                   // 下三角矩阵 / Lower triangular matrix
IMatrix<Double> U = luResult._2;                   // 上三角矩阵 / Upper triangular matrix

// Cholesky分解 / Cholesky decomposition
IMatrix<Double> L_chol = matrix1.cholesky();            // 下三角矩阵L，A = L * L^T
```

**各分解使用的底层算法 / Underlying Algorithms**

| 分解 / Decomposition | 算法 / Algorithm | 关键特性 / Key Characteristics |
|----------------------|------------------|-------------------------------|
| **SVD** | BLAS-3 blocked bidiagonalization (LAPACK DGEBRD 模式) + BD-SQR (QR iteration with Wilkinson shift) | 两阶段算法；宽矩阵自动转置；小矩阵 (<64) 回退标量路径 / Two-stage; auto-transpose for wide matrices; scalar fallback for small matrices |
| **特征分解 / Eigendecomposition** | 上 Hessenberg 约化 → 实 Schur 分解（QR 算法）/ Upper Hessenberg reduction → real Schur form (QR algorithm) | 适用于一般稠密方阵；返回实特征值和复特征值（成对出现）/ For general dense square matrices; returns real and complex conjugate eigenvalue pairs |
| **QR 因子分解 / QR factorization** | Householder 反射 / Householder reflections | 数值稳定的正交三角分解；`qr()` 返回 Q、R 因子 / Numerically stable orthogonal-triangular factorization |
| **LU 分解 / LU decomposition** | Doolittle 算法 + 部分列主元 / Doolittle with partial pivoting | 带行交换的稳定三角分解；用于行列式和线性方程组求解 / Stable triangular factorization with row swaps; used for determinant and linear solves |
| **Cholesky 分解 / Cholesky** | LDLᵀ 变体 + 对角主元 / LDLᵀ variant with diagonal pivoting | 仅适用于对称正定矩阵；比 LU 更快且数值稳定 / SPD only; faster and more stable than LU |

#### 行列式和迹 / Determinant and Trace

```java
// 行列式 / Determinant
Double det = matrix1.det();

// 迹 / Trace
Double trace = matrix1.trace();

// 秩 / Rank
int rank = matrix1.rank();

// 条件数 / Condition number
Double cond = matrix1.cond();

// 矩阵范数 / Matrix norms
Double frobeniusNorm = matrix1.frobeniusNorm(); // Frobenius范数
Double frobeniusDistance = matrix1.frobeniusDistance(matrix2); // Frobenius距离

// 矩阵展平 / Matrix flattening
IVector<Double> flattened = matrix1.flatten();  // 展平为向量

// 矩阵重塑 / Matrix reshape
IMatrix<Double> reshaped = matrix1.reshape(4, 3);  // 重塑为4x3

// 矩阵复制 / Matrix copy
IMatrix<Double> copied = matrix1.copy();            // 深拷贝
```

#### 线性方程组求解 / Linear System Solving

```java
// 求解Ax = b
IVector<Double> b = Linalg.vector(new double[]{5, 6});
IVector<Double> x = matrix1.solve(b);

// 求解AX = B
IMatrix<Double> B = Linalg.matrix(new double[][]{{5, 6}, {7, 8}});
IMatrix<Double> X = matrix1.solve(B);
```

### 5. 数据访问和操作 / Data Access and Manipulation

#### 行列操作 / Row and Column Operations

```java
// 获取行 / Get row
IVector<Double> row = matrix1.getRow(0);

// 获取列 / Get column
IVector<Double> column = matrix1.getColumn(0);
IMatrix<Double> columnMatrix = matrix1.getColumnMatrix(0);

// 设置列 / Set column
matrix1.putColumn(0, newColumnMatrix);

// 获取多个列 / Get multiple columns
IVector<Double>[] columns = matrix1.getColumns(new int[]{0, 2});

// 获取形状 / Get shape
int[] shape = matrix1.shape();             // [行数, 列数]
int rows = matrix1.getRowNum();            // 行数
int cols = matrix1.getColNum();            // 列数
int rowsAlt = matrix1.rows();              // 行数（替代方法）
int colsAlt = matrix1.cols();              // 列数（替代方法）

// 矩阵属性检查 / Matrix property checks
boolean isSquare = matrix1.isSquare();           // 是否为方阵
boolean isSymmetric = matrix1.isSymmetric();     // 是否为对称矩阵
boolean isPositiveDefinite = matrix1.isPositiveDefinite(); // 是否为正定矩阵

// 矩阵对角线操作 / Matrix diagonal operations
IVector<Double> diagonal = matrix1.diag();       // 获取对角线元素

// 数据访问 / Data access
double[][] data = matrix1.toDoubleArray();              // 获取原始数据（仅IDoubleMatrix）
```

#### 矩阵切片操作 / Matrix Slicing Operations

```java
// 基本切片 / Basic slicing
IMatrix<Double> slice1 = matrix1.subMatrix(1, 3, 0, 2);  // 行1-2，列0-1 / Rows 1-2, columns 0-1

// 字符串切片表达式 / String slice expressions
IMatrix<Double> slice2 = matrix1.slice("1:3", "0:2");  // 行1-2，列0-1 / Rows 1-2, columns 0-1
IMatrix<Double> slice3 = matrix1.slice(":-1", "1:");   // 除最后一行外的所有行，从第1列到末尾 / All rows except last, from column 1 to end
IMatrix<Double> slice4 = matrix1.slice("::2", "::2");  // 每隔一行一列 / Every other row and column

// 负数索引切片 / Negative indexing slicing
IMatrix<Double> slice5 = matrix1.slice("-2:", "-1:");  // 最后两行，最后一列 / Last two rows, last column
IMatrix<Double> slice6 = matrix1.slice("1:-1", "1:-1"); // 中间部分（除边界） / Middle part (excluding boundaries)

// 行切片 / Row slicing
IMatrix<Double> rowSlice1 = matrix1.sliceRows("1:3");   // 行1-2 / Rows 1-2
IMatrix<Double> rowSlice2 = matrix1.sliceRows("-2:");   // 最后两行 / Last two rows
IMatrix<Double> rowSlice3 = matrix1.sliceRows("::2");   // 每隔一行 / Every other row

// 列切片 / Column slicing
IMatrix<Double> colSlice1 = matrix1.sliceColumns("0:2"); // 列0-1 / Columns 0-1
IMatrix<Double> colSlice2 = matrix1.sliceColumns("-1:"); // 最后一列 / Last column
IMatrix<Double> colSlice3 = matrix1.sliceColumns("::2"); // 每隔一列 / Every other column
```

#### 切片表达式语法 / Slice Expression Syntax

矩阵切片支持类似NumPy的切片表达式语法，支持负数索引：

Matrix slicing supports NumPy-like slice expression syntax with negative indexing:

```java
// 基本格式：rowSlice, colSlice / Basic format: rowSlice, colSlice
// 其中每个切片表达式格式为：start:end:step / Where each slice expression format is: start:end:step

// 完整格式示例 / Complete format examples
matrix1.slice("1:5:2", "0:3:1");  // 行1,3，列0,1,2 / Rows 1,3, columns 0,1,2
matrix1.slice("1:5", "0:3");      // 行1-4，列0-2 / Rows 1-4, columns 0-2
matrix1.slice("1:", ":3");        // 从行1到末尾，从开始到列2 / From row 1 to end, from start to column 2
matrix1.slice("::2", "::2");      // 每隔一行一列 / Every other row and column
matrix1.slice(":", ":");          // 整个矩阵 / Entire matrix

// 负数索引示例 / Negative indexing examples
matrix1.slice("-3:", "-2:");      // 最后三行，最后两列 / Last three rows, last two columns
matrix1.slice(":-2", ":-1");      // 除最后两行外的所有行，除最后一列外的所有列 / All rows except last two, all columns except last
matrix1.slice("-4:-1", "-3:-1");  // 倒数第4-2行，倒数第3-2列 / 4th to 2nd to last rows, 3rd to 2nd to last columns
matrix1.slice("::-1", "::-1");    // 反转向量（行列都反转） / Reverse matrix (both rows and columns reversed)
```

#### 负数索引规则 / Negative Indexing Rules

负数索引从-1开始，表示最后一个元素：

Negative indexing starts from -1, representing the last element:

```java
// 对于3x4矩阵 / For a 3x4 matrix
// 行索引 / Row indices: 0, 1, 2 (正数) / 0, 1, 2 (positive)
// 行索引 / Row indices: -3, -2, -1 (负数) / -3, -2, -1 (negative)
// 列索引 / Column indices: 0, 1, 2, 3 (正数) / 0, 1, 2, 3 (positive)
// 列索引 / Column indices: -4, -3, -2, -1 (负数) / -4, -3, -2, -1 (negative)

matrix1.get(0, 0);    // 获取第一个元素 / Get first element
matrix1.get(-1, -1);  // 获取最后一个元素 / Get last element
matrix1.get(-1, 0);   // 获取最后一行的第一个元素 / Get first element of last row
matrix1.get(0, -1);   // 获取第一行的最后一个元素 / Get last element of first row

matrix1.slice("1:-1", "1:-1");  // 中间部分（除边界） / Middle part (excluding boundaries)
matrix1.slice("-2:", "-2:");    // 右下角2x2子矩阵 / Bottom-right 2x2 submatrix
```

#### 元素操作 / Element Operations

```java
// 获取元素 / Get element
Double element = matrix1.get(1, 2);

// 设置元素 / Set element
matrix1.put(1, 2, 10.0);

// 负数索引访问 / Negative indexing access
Double lastElement = matrix1.get(-1, -1);  // 获取最后一个元素 / Get last element
matrix1.put(-1, -1, 100.0);             // 设置最后一个元素 / Set last element
```

### 6. 统计运算 / Statistical Operations

```java
// 行统计 / Row statistics
IVector<Double> rowSums = matrix1.rowSums();        // 行和（按列求和）
IVector<Double> rowMeans = matrix1.rowMeans();      // 行均值（按列求均值）

// 列统计 / Column statistics
IVector<Double> colSums = matrix1.colSums();        // 列和（按行求和）
IVector<Double> colMeans = matrix1.colMeans();      // 列均值（按行求均值）

// 整体统计 / Overall statistics
Double sum = matrix1.sum();                  // 总和
Double mean = matrix1.mean();                // 整体均值
Double variance = matrix1.var();             // 整体方差
Double stdDev = matrix1.std();               // 整体标准差
Double max = matrix1.max();                  // 最大值
Double min = matrix1.min();                  // 最小值
```

### 7. 数据转换和中心化 / Data Conversion and Centering

```java
// 中心化 / Centering
IMatrix<Double> centered = matrix1.center();        // 减去列均值

// 协方差矩阵 / Covariance matrix
IMatrix<Double> covMatrix = matrix1.covariance();   // 从原始数据计算
IMatrix<Double> covFromCentered = matrix1.covarianceFromCentered(); // 从中心化数据计算

// 归一化 / Normalization
IMatrix<Double> normalizedRows = matrix1.normalizeRows();      // 行归一化
IMatrix<Double> normalizedCols = matrix1.normalizeColumns();   // 列归一化

// 类型转换 / Type conversion
double[][] doubleArray = matrix1.toDoubleArray(); // 转换为double数组
float[][] floatArray = matrix1.toFloatArray();   // 转换为float数组
IVector<Double> flatArray = matrix1.flatten();      // 转为一维向量
```

### 8. 数学函数应用 / Mathematical Functions

```java
// 三角函数 / Trigonometric functions
IMatrix<Double> sinResult = matrix1.sin();          // 正弦
IMatrix<Double> cosResult = matrix1.cos();          // 余弦
IMatrix<Double> tanResult = matrix1.tan();          // 正切

// 双曲函数 / Hyperbolic functions
IMatrix<Double> sinhResult = matrix1.sinh();        // 双曲正弦
IMatrix<Double> coshResult = matrix1.cosh();        // 双曲余弦
IMatrix<Double> tanhResult = matrix1.tanh();        // 双曲正切

// 指数和对数 / Exponential and logarithm
IMatrix<Double> expResult = matrix1.exp();          // 指数
IMatrix<Double> logResult = matrix1.log();          // 自然对数

// 绝对值和符号 / Absolute value and sign
IMatrix<Double> absResult = matrix1.abs();          // 绝对值
IMatrix<Double> signResult = matrix1.sign();        // 符号函数
```

### 9. 矩阵操作 / Matrix Operations

```java
// 矩阵拼接 / Matrix concatenation
IMatrix<Double> hstacked = matrix1.hstack(matrix2); // 水平拼接
IMatrix<Double> vstacked = matrix1.vstack(matrix2); // 垂直拼接

// 矩阵分割 / Matrix splitting
IMatrix<Double>[] hsplit = matrix1.hsplit(new int[]{2}); // 水平分割
IMatrix<Double>[] vsplit = matrix1.vsplit(new int[]{2}); // 垂直分割

// 矩阵重塑 / Matrix reshape
IMatrix<Double> reshaped = matrix1.reshape(4, 3);  // 重塑为4x3

// 矩阵复制 / Matrix copy
IMatrix<Double> copied = matrix1.copy();            // 深拷贝

// 矩阵范数 / Matrix norms
Double frobeniusNorm = matrix1.frobeniusNorm(); // Frobenius范数
Double frobeniusDistance = matrix1.frobeniusDistance(matrix2); // Frobenius距离
Double l1Norm = matrix1.norm1(); // 矩阵L1范数
```

### 10. 高级矩阵操作 / Advanced Matrix Operations

#### 花式索引和子矩阵操作 / Fancy Indexing and Submatrix Operations

```java
// 花式索引获取矩阵元素 / Fancy indexing for matrix elements
int[] rowIndices = {0, 2, 4};
int[] colIndices = {1, 3};
IMatrix<Double> fancyResult = matrix1.fancyGet(rowIndices, colIndices);

// 子矩阵操作 / Submatrix operations
IMatrix<Double> subMatrix = matrix1.subMatrix(1, 3, 1, 3); // 获取子矩阵
matrix1.setSubMatrix(0, 2, 0, 2, subMatrix); // 设置子矩阵
```

#### 矩阵属性检查 / Matrix Property Checks

```java
// 矩阵属性检查 / Matrix property checks
boolean isSquare = matrix1.isSquare();           // 是否为方阵
boolean isSymmetric = matrix1.isSymmetric();     // 是否为对称矩阵
boolean isPositiveDefinite = matrix1.isPositiveDefinite(); // 是否为正定矩阵

// 矩阵对角线操作 / Matrix diagonal operations
IVector<Double> diagonal = matrix1.diag();       // 获取对角线元素
```

#### 矩阵平均和加载 / Matrix Averaging and Loading

```java
IMatrix<Double> average = IMatrix.average(matrix1, matrix2, Double.class);

// 从文件加载（指定类型）/ Load from file with type specification
IMatrix<Float> floatMatrix = Linalg.load("matrix.txt", Float.class);
IMatrix<Double> doubleMatrix = Linalg.load("matrix.txt", Double.class);
```

### 11. 文件操作 / File Operations

```java
// 保存矩阵到文件 / Save matrix to file
IMatrix<Double> matrix = Linalg.matrix(new double[][]{{1, 2}, {3, 4}});
// matrix.save("matrix.txt"); // 需要实现保存方法

// 从文件加载矩阵 / Load matrix from file
IMatrix<Double> loaded = Linalg.load("matrix.txt");
```

## 使用示例 / Usage Examples

详细的代码示例请参考 [Matrix-Examples.md](examples/Matrix-Examples.md) 文档。

For detailed code examples, please refer to the [Matrix-Examples.md](examples/Matrix-Examples.md) document.

## 性能特性 / Performance Features

### 内存优化 / Memory Optimization
- 高效的二维数组操作 / Efficient 2D array operations
- 智能的内存分配策略 / Smart memory allocation strategy
- 最小化临时对象创建 / Minimize temporary object creation

### 算法优化 / Algorithm Optimization
- 优化的线性代数算法 / Optimized linear algebra algorithms
- 高效的矩阵分解实现 / Efficient matrix decomposition implementation
- 并行计算支持 / Parallel computing support: BLAS-3 SVD 的 trailing matrix update 和 U/V 构造已采用 ForkJoinPool 行分区并行 / BLAS-3 SVD trailing updates and U/V construction use ForkJoinPool row-range parallelism

### 数值稳定性 / Numerical Stability
- 稳定的特征分解算法 / Stable eigendecomposition algorithms
- 条件数检查 / Condition number checking
- 奇异值处理 / Singular value handling

## 注意事项 / Notes

1. **矩阵维度** / **Matrix Dimensions**: 确保矩阵运算的维度兼容性 / Ensure matrix operation dimension compatibility
2. **数值精度** / **Numerical Precision**: 使用float类型，注意精度限制 / Use float type, pay attention to precision limitations
3. **奇异矩阵** / **Singular Matrix**: 某些操作（如求逆）需要矩阵可逆 / Some operations (like inverse) require invertible matrix
4. **内存使用** / **Memory Usage**: 大矩阵操作时注意内存使用 / Pay attention to memory usage for large matrix operations
5. **方法命名** / **Method Naming**: 注意方法名称的准确性，如`rowSums()`实际是按列求和，`colSums()`实际是按行求和 / Pay attention to method name accuracy, e.g., `rowSums()` actually sums by columns, `colSums()` actually sums by rows

## 扩展性 / Extensibility

`IMatrix` 接口设计支持扩展，可以轻松添加新的矩阵类型实现：
The `IMatrix` interface is designed to support extensions, making it easy to add new matrix type implementations:
- 稀疏矩阵 / Sparse matrices
- 分布式矩阵 / Distributed matrices
- 特殊结构矩阵 / Special structure matrices (Toeplitz 已支持，Hankel 计划中)
- GPU加速矩阵 / GPU-accelerated matrices（暂时移除，未来版本可能回归 / temporarily removed in v0.3.7, may return in future versions）

## 应用场景 / Application Scenarios

### 科学计算 / Scientific Computing
- 线性方程组求解 / Linear system solving
- 特征值问题 / Eigenvalue problems
- 信号处理 / Signal processing

### 机器学习 / Machine Learning
- 特征工程 / Feature engineering
- 数据预处理 / Data preprocessing
- 模型训练 / Model training

### 数据分析 / Data Analysis
- 统计计算 / Statistical computing
- 数据可视化 / Data visualization
- 模式识别 / Pattern recognition

## 与 NumPy `np.dot` 的对照 / NumPy `np.dot` mapping

NumPy 中 `np.dot` 随输入维度含义不同；本库用**不同方法名**区分，避免与「Frobenius 内积」混淆：

| NumPy | 本库 YiShape | 说明 / Notes |
|--------|----------------|---------------|
| `np.dot(v, w)`（一维×一维） | `v.innerProduct(w)` 或 `v.dot(w)` | 向量内积 / Inner product |
| `np.dot(v, M)`（一维×二维） | `v.mmul(M)` 或 `v.dot(M)` | 行向量×矩阵，两写法等价 / Row × matrix |
| `np.dot(A, B)`（二维×二维） | `A.mmul(B)` | 矩阵乘法（**非** Frobenius 内积）/ Matrix multiply (not Frobenius) |
| `np.sum(A * B)`（同形） | `A.frobeniusInnerProduct(B)` | 逐元素乘再求和（Frobenius 内积）/ Sum of element-wise products |

## 与NumPy功能对照表 / NumPy Functionality Comparison Table

| 功能类别 / Function Category | IMatrix/Linalg | NumPy | 说明 / Description |
|---------|-------------------|-------|------|
| **矩阵创建 / Matrix Creation** | | | |
| 从数组创建 / Create from array | `Linalg.matrix(data)` | `np.array(data)` | 从二维数组创建矩阵 / Create matrix from 2D array |
| 全零矩阵 / Zero matrix | `Linalg.zeros(rows, cols)` | `np.zeros((rows, cols))` | 创建全零矩阵 / Create zero matrix |
| 全一矩阵 / Ones matrix | `Linalg.ones(rows, cols)` | `np.ones((rows, cols))` | 创建全一矩阵 / Create ones matrix |
| 单位矩阵 / Identity matrix | `Linalg.eye(n)` | `np.eye(n)` | 创建n×n单位矩阵 / Create n×n identity matrix |
| 随机矩阵 / Random matrix | `Linalg.rand(rows, cols)` | `np.random.rand(rows, cols)` | 创建随机矩阵 / Create random matrix |
| 正态随机矩阵 / Normal random matrix | `Linalg.randn(rows, cols)` | `np.random.randn(rows, cols)` | 创建正态分布随机矩阵 / Create normal random matrix |
| 对角矩阵 / Diagonal matrix | `Linalg.diag(vector)` | `np.diag(vector)` | 从向量创建对角矩阵 / Create diagonal matrix from vector |
| 从一维数组重塑 / Reshape from 1D array | `Linalg.fromArray(data, rows, cols)` | `np.array(data).reshape(rows, cols)` | 从一维数组重塑为矩阵 / Reshape 1D array to matrix |
| Toeplitz 矩阵 / Toeplitz matrix | `Linalg.toeplitz(column)` | `scipy.linalg.toeplitz(c)` | 从第一列创建 Toeplitz 矩阵 / Create Toeplitz from first column |
| **基本运算 / Basic Operations** | | | |
| 矩阵加法 / Matrix addition | `matrix1.add(matrix2)` | `matrix1 + matrix2` | 元素级加法 / Element-wise addition |
| 矩阵减法 / Matrix subtraction | `matrix1.sub(matrix2)` | `matrix1 - matrix2` | 元素级减法 / Element-wise subtraction |
| 矩阵乘法 / Matrix multiplication | `matrix1.mmul(matrix2)` | `matrix1 @ matrix2` | 矩阵乘法 / Matrix multiplication |
| 矩阵与向量乘法 / Matrix-vector multiplication | `matrix.mmul(vector)` | `matrix @ vector` | 矩阵与向量乘法 / Matrix-vector multiplication |
| 元素级除法 / Element-wise division | `matrix1.divide(matrix2)` | `matrix1 / matrix2` | 元素级除法 / Element-wise division |
| 标量乘法 / Scalar multiply | `matrix.multiplyByScalar(s)` 或 `matrix.mmul(s)` | `matrix * scalar` | 逐元素乘标量；`mmul(矩阵)` 仍为矩阵乘 / Scale; `mmul(matrix)` is matmul |
| 标量减法 / Scalar subtraction | `matrix.sub(scalar)` | `matrix - scalar` | 逐元素减标量 / Subtract scalar from each element |
| 元素级乘法 / Hadamard product | `matrix1.multiply(matrix2)` | `matrix1 * matrix2` | 同形状逐元素乘 / Element-wise product |
| Frobenius 内积 / Frobenius inner product | `matrix1.frobeniusInnerProduct(matrix2)` | `np.sum(matrix1 * matrix2)` | 对应元素乘积之和 / Sum of element-wise products |
| 向量外积 / Vector outer | `v1.outer(v2)`（IVector） | `np.outer(v1, v2)` | 两向量外积 / Outer product of vectors |
| 展平外积 / Outer (flattened) | `A.outer(B)` | `np.outer(A.ravel(), B.ravel())` | 行优先展平再外积 / Row-major flatten then outer |
| Kronecker 积 / Kronecker | `A.kron(B)` 或 `Linalg.kron(A,B)` | `np.kron(A, B)` | 分块张量积 / Block tensor product |
| **矩阵变换 / Matrix Transformations** | | | |
| 转置 / Transpose | `matrix.transpose(),matrix.t()` | `matrix.T` | 矩阵转置 / Matrix transpose |
| 幂运算 / Power | `matrix.pow(n)` | `np.power(matrix, n)` | 元素级幂运算 / Element-wise power |
| 开方 / Square root | `matrix.sqrt()` | `np.sqrt(matrix)` | 元素级开方 / Element-wise square root |
| 求逆 / Inverse | `matrix.inv()` | `np.linalg.inv(matrix)` | 矩阵求逆 / Matrix inverse |
| **线性代数 / Linear Algebra** | | | |
| 特征分解 / Eigendecomposition | `matrix.eigen()` | `np.linalg.eig(matrix)` | 特征值与特征向量 / Eigenpairs |
| QR 算法特征（别名）/ Eigen via QR algo | `matrix.qrEigen()` 或 `qrEigenDecomposition()` | 同 `eigen` 数值结果 | 与 `eigen()` 相同；<strong>不是</strong> `qr()` 因子分解 / Same as eigen; not QR factorization |
| SVD分解 / SVD decomposition | `matrix.svd()` | `np.linalg.svd(matrix)` | 奇异值分解 / Singular value decomposition |
| LU分解 / LU decomposition | `matrix.lu()` | `scipy.linalg.lu(matrix)` | LU分解 / LU decomposition |
| QR 因子分解 / QR factorization | `matrix.qr()` | `np.linalg.qr(matrix)` | A=QR（Q 正交、R 上三角）；<strong>不是</strong>特征分解 / A=QR; not eigen |
| Cholesky分解 / Cholesky decomposition | `matrix.cholesky()` | `np.linalg.cholesky(matrix)` | Cholesky分解 / Cholesky decomposition |
| 行列式 / Determinant | `matrix.det()` | `np.linalg.det(matrix)` | 矩阵行列式 / Matrix determinant |
| 矩阵秩 / Matrix rank | `matrix.rank()` | `np.linalg.matrix_rank(matrix)` | 矩阵秩 / Matrix rank |
| 线性方程组求解 / Linear system solving | `matrix.solve(vector)` | `np.linalg.solve(matrix, vector)` | 求解Ax=b / Solve Ax=b |
| **统计运算 / Statistical Operations** | | | |
| 行求和 / Row sums | `matrix.rowSums()` | `np.sum(matrix, axis=1)` | 每行元素之和，结果长度=行数 / Sum within each row |
| 列求和 / Column sums | `matrix.colSums()` | `np.sum(matrix, axis=0)` | 每列元素之和，结果长度=列数 / Sum within each column |
| 行均值 / Row means | `matrix.rowMeans()` | `np.mean(matrix, axis=1)` | 每行均值 / Mean of each row |
| 列均值 / Column means | `matrix.colMeans()` | `np.mean(matrix, axis=0)` | 每列均值 / Mean of each column |
| 整体求和 / Overall sum | `matrix.sum()` | `np.sum(matrix)` | 矩阵所有元素求和 / Sum all matrix elements |
| 整体均值 / Overall mean | `matrix.mean()` | `np.mean(matrix)` | 矩阵所有元素均值 / Mean of all matrix elements |
| 方差 / Variance | `matrix.var()` | `np.var(matrix)` | 矩阵元素方差 / Matrix element variance |
| 标准差 / Standard deviation | `matrix.std()` | `np.std(matrix)` | 矩阵元素标准差 / Matrix element standard deviation |
| **数据预处理 / Data Preprocessing** | | | |
| 中心化 / Centering | `matrix.center()` | `matrix - np.mean(matrix, axis=0)` | 列中心化 / Column centering |
| 协方差矩阵 / Covariance matrix | `matrix.covariance()` | `np.cov(matrix.T)` | 计算协方差矩阵 / Calculate covariance matrix |
| 行归一化 / Row normalization | `matrix.normalizeRows()` | `matrix / np.linalg.norm(matrix, axis=1, keepdims=True)` | L2行归一化 / L2 row normalization |
| 列归一化 / Column normalization | `matrix.normalizeColumns()` | `matrix / np.linalg.norm(matrix, axis=0, keepdims=True)` | L2列归一化 / L2 column normalization |
| **数据访问 / Data Access** | | | |
| 获取元素 / Get element | `matrix.get(row, col)` | `matrix[row, col]` | 获取指定位置元素 / Get element at specified position |
| 设置元素 / Set element | `matrix.put(row, col, value)` | `matrix[row, col] = value` | 设置指定位置元素 / Set element at specified position |
| 负数索引访问 / Negative indexing | `matrix.get(-1, -1)` | `matrix[-1, -1]` | 负数索引访问 / Negative indexing access |
| 获取行 / Get row | `matrix.getRow(row)` | `matrix[row, :]` | 获取指定行 / Get specified row |
| 获取列 / Get column | `matrix.getColumn(col)` | `matrix[:, col]` | 获取指定列 / Get specified column |
| 获取形状 / Get shape | `matrix.shape()` | `matrix.shape` | 获取矩阵维度 / Get matrix dimensions |
| **矩阵切片 / Matrix Slicing** | | | |
| 基本切片 / Basic slicing | `matrix.slice(startRow, endRow, startCol, endCol)` | `matrix[startRow:endRow, startCol:endCol]` | 基本矩阵切片 / Basic matrix slicing |
| 字符串切片 / String slicing | `matrix.slice("1:3", "0:2")` | `matrix[1:3, 0:2]` | 字符串切片表达式 / String slice expression |
| 负数索引切片 / Negative indexing | `matrix.slice("-2:", "-1:")` | `matrix[-2:, -1:]` | 负数索引切片 / Negative indexing slicing |
| 步长切片 / Step slicing | `matrix.slice("::2", "::2")` | `matrix[::2, ::2]` | 步长切片 / Step slicing |
| 反转向量 / Reverse matrix | `matrix.slice("::-1", "::-1")` | `matrix[::-1, ::-1]` | 反转向量 / Reverse matrix |
| 行切片 / Row slicing | `matrix.sliceRows("1:3")` | `matrix[1:3, :]` | 行切片 / Row slicing |
| 列切片 / Column slicing | `matrix.sliceColumns("0:2")` | `matrix[:, 0:2]` | 列切片 / Column slicing |
| **矩阵范数 / Matrix Norms** | | | |
| Frobenius范数 / Frobenius norm | `matrix.frobeniusNorm(), matrix.frobenius` | `np.linalg.norm(matrix, 'fro')` | Frobenius范数 / Frobenius norm |
| **数据转换 / Data Conversion** | | | |
| 转float数组 / Convert to float array | `matrix.toFloatArray()` | `matrix.astype(np.float32)` | 获取float数组 / Get float array |
| 转double数组 / Convert to double array | `matrix.toDoubleArray()` | `matrix.astype(np.float64)` | 转换为double数组 / Convert to double array |
| 矩阵复制 / Matrix copy | `matrix.copy()` | `matrix.copy()` | 深拷贝矩阵 / Deep copy matrix |
| 矩阵重塑 / Matrix reshape | `matrix.reshape(rows, cols)` | `matrix.reshape(rows, cols)` | 重塑矩阵维度 / Reshape matrix dimensions |
| **数学函数 / Mathematical Functions** | | | |
| 三角函数 / Trigonometric functions | `matrix.sin()`, `matrix.cos()`, `matrix.tan()` | `np.sin(matrix)`, `np.cos(matrix)`, `np.tan(matrix)` | 三角函数 / Trigonometric functions |
| 双曲函数 / Hyperbolic functions | `matrix.sinh()`, `matrix.cosh()`, `matrix.tanh()` | `np.sinh(matrix)`, `np.cosh(matrix)`, `np.tanh(matrix)` | 双曲函数 / Hyperbolic functions |
| 指数对数 / Exponential and logarithm | `matrix.exp()`, `matrix.log()` | `np.exp(matrix)`, `np.log(matrix)` | 指数和对数函数 / Exponential and logarithm functions |
| 绝对值 / Absolute value | `matrix.abs()` | `np.abs(matrix)` | 绝对值函数 / Absolute value function |
| **矩阵操作 / Matrix Operations** | | | |
| 水平拼接 / Horizontal stacking | `matrix1.hstack(matrix2)` | `np.hstack([matrix1, matrix2])` | 水平拼接矩阵 / Horizontally stack matrices |
| 垂直拼接 / Vertical stacking | `matrix1.vstack(matrix2)` | `np.vstack([matrix1, matrix2])` | 垂直拼接矩阵 / Vertically stack matrices |
| 水平分割 / Horizontal splitting | `matrix.hsplit(indices)` | `np.hsplit(matrix, indices)` | 水平分割矩阵 / Horizontally split matrix |
| 垂直分割 / Vertical splitting | `matrix.vsplit(indices)` | `np.vsplit(matrix, indices)` | 垂直分割矩阵 / Vertically split matrix |
| **高级功能 / Advanced Features** | | | |
| 伪逆 / Pseudo-inverse | `matrix.pinv()` | `np.linalg.pinv(matrix)` | Moore-Penrose伪逆 / Moore-Penrose pseudo-inverse |
| 条件数 / Condition number | `matrix.cond()` | `np.linalg.cond(matrix)` | 矩阵条件数 / Matrix condition number |
| **文件操作 / File Operations** | | | |
| 保存矩阵 / Save matrix | `matrix.save(path)` | `np.save(path, matrix)` | 保存矩阵到文件 / Save matrix to file |
| 加载矩阵 / Load matrix | `IMatrix.load(path)` | `np.load(path)` | 从文件加载矩阵 / Load matrix from file |

## 性能对比 / Performance Comparison

| 操作类型 / Operation Type | IMatrix/RereMatrix | NumPy | 优势说明 / Advantage Description |
|---------|-------------------|-------|----------|
| **内存布局 / Memory Layout** | Java `double[][]`（带行对象头）；核心分解内部使用 flat `double[]` / Java `double[][]` with per-row headers; core decompositions internally use flat `double[]` | 连续 C 数组 / Contiguous C arrays | Java 行引用可被 JIT 优化；flat 数组为 SIMD/原生 BLAS 预留基础 / Java row refs are JIT-optimized; flat arrays enable future SIMD/native BLAS |
| **计算速度 / Computational Speed** | 纯 Java + BLAS-3 分块算法（SVD 等）；核心循环 JIT 编译 / Pure Java with BLAS-3 blocking (SVD etc.); JIT-compiled inner loops | C/BLAS/LAPACK 优化 / C/BLAS/LAPACK optimized | 中小型矩阵差距小；超大规模 NumPy 更快 / Competitive for small-to-medium; NumPy faster at very large scale |
| **跨平台性 / Cross-platform** | 纯Java，跨平台 / Pure Java, cross-platform | 需要编译 / Requires compilation | IMatrix无需额外依赖 / IMatrix requires no additional dependencies |
| **集成性 / Integration** | 与Java生态集成 / Integrates with Java ecosystem | Python生态 / Python ecosystem | 根据项目需求选择 / Choose based on project requirements |
| **扩展性 / Extensibility** | 易于扩展新功能 / Easy to extend new features | 功能丰富 / Feature rich | 两者都支持扩展 / Both support extension |
| **部署便利性 / Deployment Convenience** | 无需额外运行时 / No additional runtime needed | 需要Python环境 / Requires Python environment | IMatrix更适合Java项目 / IMatrix is more suitable for Java projects |

## 最佳实践建议 / Best Practices Recommendations

### 选择IMatrix/RereMatrix的场景 / When to Choose IMatrix/RereMatrix
- Java项目中的矩阵计算需求 / Matrix computation needs in Java projects
- 需要与Java生态系统深度集成 / Need deep integration with Java ecosystem
- 对内存占用有严格要求 / Strict memory usage requirements
- 需要自定义矩阵操作和扩展 / Need custom matrix operations and extensions

### 选择NumPy的场景 / When to Choose NumPy
- Python项目中的矩阵计算需求 / Matrix computation needs in Python projects


### 注意事项 / Important Notes
1. **方法命名混淆**：`rowSums()`实际是按列求和，`colSums()`实际是按行求和，这与方法名称的字面意思相反 / **Method naming confusion**: `rowSums()` actually sums by columns, `colSums()` actually sums by rows, which is opposite to the literal meaning of the method names
2. **异常处理**：某些操作可能抛出异常，如奇异矩阵求逆、负数开方等 / **Exception handling**: Some operations may throw exceptions, such as singular matrix inversion, negative number square root, etc.
3. **内存管理**：大矩阵操作时注意内存使用，及时释放不需要的引用 / **Memory management**: Pay attention to memory usage for large matrix operations, release unnecessary references in time

---

**矩阵操作** - 线性代数的核心，让复杂计算更简单！

**Matrix Operations** - The core of linear algebra, making complex computations simpler!
