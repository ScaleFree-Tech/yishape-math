# 矩阵操作 (Matrix Operations)

## 概述 / Overview

`IMatrix` 接口和 `RereMatrix` 实现类提供了完整的矩阵数学运算功能，包括基本数学运算、矩阵变换、线性代数运算、特征分解等。该接口设计支持链式操作，提供丰富的矩阵操作功能。

The `IMatrix` interface and `RereMatrix` implementation class provide comprehensive matrix mathematical operations including basic mathematical operations, matrix transformations, linear algebra operations, eigendecomposition, and more. The interface supports method chaining and offers rich matrix operation functionalities.

## 核心接口 / Core Interface

### IMatrix 接口 / IMatrix Interface

`IMatrix` 是矩阵操作的核心接口，定义了所有矩阵运算的抽象方法。

`IMatrix` is the core interface for matrix operations, defining abstract methods for all matrix operations.

## 主要功能 / Main Features

### 1. 矩阵创建 / Matrix Creation

#### 静态工厂方法 / Static Factory Methods

```java
// 从二维数组创建 / Create from 2D array
float[][] data = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
IMatrix matrix1 = IMatrix.of(data);

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
IMatrix ones = IMatrix.ones(3, 3);        // 全1矩阵 / Matrix of ones
IMatrix zeros = IMatrix.zeros(3, 3);      // 零矩阵 / Zero matrix
IMatrix identity = IMatrix.eye(3);        // 单位矩阵 / Identity matrix
IMatrix random = IMatrix.rand(3, 3);      // 随机矩阵 / Random matrix
IMatrix randomSeeded = IMatrix.rand(3, 3, 12345L); // 指定种子的随机矩阵 / Random matrix with seed

// 创建对角矩阵 / Create diagonal matrix
IMatrix diagonal = IMatrix.diag(new float[]{1, 2, 3});

// 从一维数组创建 / Create from 1D array
IMatrix reshaped = IMatrix.fromArray(new float[]{1, 2, 3, 4}, 2, 2);

// 从文件加载 / Load from file
IMatrix loaded = IMatrix.load("matrix.txt");
```

### 2. 基本数学运算 / Basic Mathematical Operations

#### 矩阵间运算 / Matrix-to-Matrix Operations

```java
// 加法 / Addition
IMatrix sum = matrix1.add(matrix2);

// 减法 / Subtraction
IMatrix diff = matrix1.sub(matrix2);

// 矩阵乘法 / Matrix multiplication
IMatrix matrixProduct = matrix1.mmul(matrix2);

// 元素级除法 / Element-wise division
IMatrix quotient = matrix1.divide(matrix2);
```

#### 标量运算 / Scalar Operations

```java
// 标量减法 / Scalar subtraction
IMatrix result1 = matrix1.sub(5.0f);

// 标量乘法 / Scalar multiplication
IMatrix result2 = matrix1.mmul(3.0f);
IMatrix result3 = matrix1.mmul(3.0); // double类型

// 向量点积 / Vector dot product
float dotProduct = matrix1.dot(matrix2); // 要求都是列向量
```

### 3. 矩阵变换 / Matrix Transformations

```java
// 转置 / Transpose
IMatrix transposed = matrix1.transpose();      // 就地转置
IMatrix transposedNew = matrix1.transposeNew(); // 创建新对象
IMatrix transposedShort = matrix1.t();         // 简写形式

// 幂运算 / Power
IMatrix squared = matrix1.pow(2.0f);

// 开方 / Square root
IMatrix sqrt = matrix1.sqrt();

// 求逆 / Inverse
IMatrix inverse = matrix1.inv();

// 伪逆 / Pseudo-inverse
IMatrix pseudoInverse = matrix1.pinv();
```

### 4. 线性代数运算 / Linear Algebra Operations

#### 特征分解 / Eigendecomposition

```java
// 特征分解 / Eigendecomposition
Tuple2<IVector, IMatrix> eigenResult = matrix1.eigen();
IVector eigenValues = eigenResult._1;      // 特征值 / Eigenvalues
IMatrix eigenVectors = eigenResult._2;     // 特征向量 / Eigenvectors

// QR算法特征分解 / QR algorithm eigendecomposition
Tuple2<IVector, IMatrix> qrEigenResult = matrix1.qrEigenDecomposition();
```

#### 奇异值分解 / Singular Value Decomposition

```java
// SVD分解 / SVD decomposition
Tuple3<IMatrix, IVector, IMatrix> svdResult = matrix1.svd();
IMatrix U = svdResult._1;                  // 左奇异向量 / Left singular vectors
IVector S = svdResult._2;                  // 奇异值 / Singular values
IMatrix V = svdResult._3;                  // 右奇异向量 / Right singular vectors
```

#### 矩阵分解 / Matrix Decomposition

```java
// QR分解 / QR decomposition
Tuple2<IMatrix, IMatrix> qrResult = matrix1.qr();
IMatrix Q = qrResult._1;                   // 正交矩阵 / Orthogonal matrix
IMatrix R = qrResult._2;                   // 上三角矩阵 / Upper triangular matrix

// LU分解 / LU decomposition
Tuple2<IMatrix, IMatrix> luResult = matrix1.lu();
IMatrix L = luResult._1;                   // 下三角矩阵 / Lower triangular matrix
IMatrix U = luResult._2;                   // 上三角矩阵 / Upper triangular matrix

// Cholesky分解 / Cholesky decomposition
IMatrix L = matrix1.cholesky();            // 下三角矩阵L，A = L * L^T
```

#### 行列式和迹 / Determinant and Trace

```java
// 行列式 / Determinant
float det = matrix1.det();

// 迹 / Trace
float trace = matrix1.trace();

// 秩 / Rank
int rank = matrix1.rank();

// 条件数 / Condition number
float cond = matrix1.cond();
```

#### 线性方程组求解 / Linear System Solving

```java
// 求解Ax = b
IVector b = IVector.of(new float[]{5, 6});
IVector x = matrix1.solve(b);

// 求解AX = B
IMatrix B = IMatrix.of(new float[][]{{5, 6}, {7, 8}});
IMatrix X = matrix1.solve(B);
```

### 5. 数据访问和操作 / Data Access and Manipulation

#### 行列操作 / Row and Column Operations

```java
// 获取行 / Get row
IVector row = matrix1.getRow(0);

// 获取列 / Get column
IVector column = matrix1.getColunm(0);
IMatrix columnMatrix = matrix1.getColumn(0);

// 设置列 / Set column
matrix1.putColumn(0, newColumnVector);

// 获取多个列 / Get multiple columns
IMatrix[] columns = matrix1.getColumns(new int[]{0, 2});

// 获取形状 / Get shape
int[] shape = matrix1.shape();             // [行数, 列数]
int rows = matrix1.getRowNum();            // 行数
int cols = matrix1.getColNum();            // 列数
int rowsAlt = matrix1.getRows();           // 行数（替代方法）
int colsAlt = matrix1.getColumns();        // 列数（替代方法）
```

#### 元素操作 / Element Operations

```java
// 获取元素 / Get element
float element = matrix1.get(1, 2);

// 设置元素 / Set element
matrix1.put(1, 2, 10.0f);
```

### 6. 统计运算 / Statistical Operations

```java
// 行统计 / Row statistics
IVector rowSums = matrix1.rowSums();        // 行和（按列求和）
IVector rowMeans = matrix1.rowMeans();      // 行均值（按列求均值）

// 列统计 / Column statistics
IVector colSums = matrix1.colSums();        // 列和（按行求和）
IVector colMeans = matrix1.colMeans();      // 列均值（按行求均值）

// 整体统计 / Overall statistics
float sum = matrix1.sum();                  // 总和
float mean = matrix1.mean();                // 整体均值
float variance = matrix1.var();             // 整体方差
float stdDev = matrix1.std();               // 整体标准差
float max = matrix1.max();                  // 最大值
float min = matrix1.min();                  // 最小值
```

### 7. 数据转换和中心化 / Data Conversion and Centering

```java
// 中心化 / Centering
IMatrix centered = matrix1.center();        // 减去列均值

// 协方差矩阵 / Covariance matrix
IMatrix covMatrix = matrix1.covariance();   // 从原始数据计算
IMatrix covFromCentered = matrix1.covarianceFromCentered(); // 从中心化数据计算

// 归一化 / Normalization
IMatrix normalizedRows = matrix1.normalizeRows();      // 行归一化
IMatrix normalizedCols = matrix1.normalizeColumns();   // 列归一化

// 类型转换 / Type conversion
float[][] floatArray = matrix1.getData();   // 获取float数组
float[] flatArray = matrix1.toArray();      // 转为一维数组
double[] doubleArray = matrix1.toDoubleArray(); // 转换为double数组
```

### 8. 数学函数应用 / Mathematical Functions

```java
// 三角函数 / Trigonometric functions
IMatrix sinResult = matrix1.sin();          // 正弦
IMatrix cosResult = matrix1.cos();          // 余弦
IMatrix tanResult = matrix1.tan();          // 正切

// 双曲函数 / Hyperbolic functions
IMatrix sinhResult = matrix1.sinh();        // 双曲正弦
IMatrix coshResult = matrix1.cosh();        // 双曲余弦
IMatrix tanhResult = matrix1.tanh();        // 双曲正切

// 指数和对数 / Exponential and logarithm
IMatrix expResult = matrix1.exp();          // 指数
IMatrix logResult = matrix1.log();          // 自然对数

// 绝对值和符号 / Absolute value and sign
IMatrix absResult = matrix1.abs();          // 绝对值
IMatrix signResult = matrix1.sign();        // 符号函数
```

### 9. 矩阵操作 / Matrix Operations

```java
// 矩阵拼接 / Matrix concatenation
IMatrix hstacked = matrix1.hstack(matrix2); // 水平拼接
IMatrix vstacked = matrix1.vstack(matrix2); // 垂直拼接

// 矩阵分割 / Matrix splitting
IMatrix[] hsplit = matrix1.hsplit(new int[]{2}); // 水平分割
IMatrix[] vsplit = matrix1.vsplit(new int[]{2}); // 垂直分割

// 矩阵重塑 / Matrix reshape
IMatrix reshaped = matrix1.reshape(4, 3);  // 重塑为4x3

// 矩阵复制 / Matrix copy
IMatrix copied = matrix1.copy();            // 深拷贝

// 矩阵范数 / Matrix norms
float frobeniusNorm = matrix1.frobeniusNorm(); // Frobenius范数
float frobeniusDistance = matrix1.frobeniusDistance(matrix2); // Frobenius距离
```

### 10. 文件操作 / File Operations

```java
// 保存矩阵到文件 / Save matrix to file
IMatrix matrix = IMatrix.of(new float[][]{{1, 2}, {3, 4}});
matrix.save("matrix.txt");

// 从文件加载矩阵 / Load matrix from file
IMatrix loaded = IMatrix.load("matrix.txt");
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
- 并行计算支持（未来版本）/ Parallel computing support (future version)

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
- GPU加速矩阵 / GPU-accelerated matrices
- 分布式矩阵 / Distributed matrices
- 特殊结构矩阵 / Special structure matrices (Toeplitz, Hankel, etc.)

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

## 与NumPy功能对照表 / NumPy Functionality Comparison Table

| 功能类别 / Function Category | IMatrix/RereMatrix | NumPy | 说明 / Description |
|---------|-------------------|-------|------|
| **矩阵创建 / Matrix Creation** | | | |
| 从数组创建 / Create from array | `IMatrix.of(data)` | `np.array(data)` | 从二维数组创建矩阵 / Create matrix from 2D array |
| 全零矩阵 / Zero matrix | `IMatrix.zeros(rows, cols)` | `np.zeros((rows, cols))` | 创建全零矩阵 / Create zero matrix |
| 全一矩阵 / Ones matrix | `IMatrix.ones(rows, cols)` | `np.ones((rows, cols))` | 创建全一矩阵 / Create ones matrix |
| 单位矩阵 / Identity matrix | `IMatrix.eye(n)` | `np.eye(n)` | 创建n×n单位矩阵 / Create n×n identity matrix |
| 随机矩阵 / Random matrix | `IMatrix.rand(rows, cols)` | `np.random.rand(rows, cols)` | 创建随机矩阵 / Create random matrix |
| 对角矩阵 / Diagonal matrix | `IMatrix.diag(vector)` | `np.diag(vector)` | 从向量创建对角矩阵 / Create diagonal matrix from vector |
| **基本运算 / Basic Operations** | | | |
| 矩阵加法 / Matrix addition | `matrix1.add(matrix2)` | `matrix1 + matrix2` | 元素级加法 / Element-wise addition |
| 矩阵减法 / Matrix subtraction | `matrix1.sub(matrix2)` | `matrix1 - matrix2` | 元素级减法 / Element-wise subtraction |
| 矩阵乘法 / Matrix multiplication | `matrix1.mmul(matrix2)` | `matrix1 @ matrix2` | 矩阵乘法 / Matrix multiplication |
| 标量运算 / Scalar operations | `matrix.mmul(scalar)` | `matrix * scalar` | 标量乘法 / Scalar multiplication |
| **矩阵变换 / Matrix Transformations** | | | |
| 转置 / Transpose | `matrix.transpose()` | `matrix.T` | 矩阵转置 / Matrix transpose |
| 幂运算 / Power | `matrix.pow(n)` | `np.power(matrix, n)` | 元素级幂运算 / Element-wise power |
| 开方 / Square root | `matrix.sqrt()` | `np.sqrt(matrix)` | 元素级开方 / Element-wise square root |
| 求逆 / Inverse | `matrix.inv()` | `np.linalg.inv(matrix)` | 矩阵求逆 / Matrix inverse |
| **线性代数 / Linear Algebra** | | | |
| 特征分解 / Eigendecomposition | `matrix.eigen()` | `np.linalg.eig(matrix)` | 特征值和特征向量 / Eigenvalues and eigenvectors |
| SVD分解 / SVD decomposition | `matrix.svd()` | `np.linalg.svd(matrix)` | 奇异值分解 / Singular value decomposition |
| LU分解 / LU decomposition | `matrix.lu()` | `scipy.linalg.lu(matrix)` | LU分解 / LU decomposition |
| QR分解 / QR decomposition | `matrix.qr()` | `np.linalg.qr(matrix)` | QR分解 / QR decomposition |
| Cholesky分解 / Cholesky decomposition | `matrix.cholesky()` | `np.linalg.cholesky(matrix)` | Cholesky分解 / Cholesky decomposition |
| 行列式 / Determinant | `matrix.det()` | `np.linalg.det(matrix)` | 矩阵行列式 / Matrix determinant |
| 矩阵秩 / Matrix rank | `matrix.rank()` | `np.linalg.matrix_rank(matrix)` | 矩阵秩 / Matrix rank |
| 线性方程组求解 / Linear system solving | `matrix.solve(vector)` | `np.linalg.solve(matrix, vector)` | 求解Ax=b / Solve Ax=b |
| **统计运算 / Statistical Operations** | | | |
| 行求和 / Row sums | `matrix.rowSums()` | `np.sum(matrix, axis=0)` | 按列求和（注意命名）/ Sum by columns (note naming) |
| 列求和 / Column sums | `matrix.colSums()` | `np.sum(matrix, axis=1)` | 按行求和（注意命名）/ Sum by rows (note naming) |
| 行均值 / Row means | `matrix.rowMeans()` | `np.mean(matrix, axis=0)` | 按列求均值（注意命名）/ Mean by columns (note naming) |
| 列均值 / Column means | `matrix.colMeans()` | `np.mean(matrix, axis=1)` | 按行求均值（注意命名）/ Mean by rows (note naming) |
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
| 获取行 / Get row | `matrix.getRow(row)` | `matrix[row, :]` | 获取指定行 / Get specified row |
| 获取列 / Get column | `matrix.getColumn(col)` | `matrix[:, col]` | 获取指定列 / Get specified column |
| 获取形状 / Get shape | `matrix.shape()` | `matrix.shape` | 获取矩阵维度 / Get matrix dimensions |
| **矩阵范数 / Matrix Norms** | | | |
| Frobenius范数 / Frobenius norm | `matrix.frobeniusNorm()` | `np.linalg.norm(matrix, 'fro')` | Frobenius范数 / Frobenius norm |
| **数据转换 / Data Conversion** | | | |
| 转float数组 / Convert to float array | `matrix.getData()` | `matrix.astype(np.float32)` | 获取float数组 / Get float array |
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
| **内存使用 / Memory Usage** | 基于float数组 / Based on float arrays | 基于numpy数组 / Based on numpy arrays | IMatrix内存占用更小 / IMatrix uses less memory |
| **计算速度 / Computational Speed** | 纯Java实现 / Pure Java implementation | C优化实现 / C optimized implementation | NumPy在大型矩阵上更快 / NumPy is faster on large matrices |
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
2. **数据类型**：所有计算都基于float类型，注意精度限制 / **Data types**: All calculations are based on float type, pay attention to precision limitations
3. **异常处理**：某些操作可能抛出异常，如奇异矩阵求逆、负数开方等 / **Exception handling**: Some operations may throw exceptions, such as singular matrix inversion, negative number square root, etc.
4. **内存管理**：大矩阵操作时注意内存使用，及时释放不需要的引用 / **Memory management**: Pay attention to memory usage for large matrix operations, release unnecessary references in time

---

**矩阵操作** - 线性代数的核心，让复杂计算更简单！

**Matrix Operations** - The core of linear algebra, making complex computations simpler!
