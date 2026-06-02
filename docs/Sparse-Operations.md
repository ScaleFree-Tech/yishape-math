# 稀疏矩阵运算 (Sparse Matrix Operations)

## 概述 / Overview

稀疏矩阵是一种高效存储和操作大规模稀疏数据的数据结构。当矩阵中大部分元素为零时，稀疏存储可以显著节省内存和计算资源。

`sparse` 包提供了 CSR、CSC、COO 三种主流稀疏矩阵格式的支持，以及多种特殊稀疏矩阵实现。

A sparse matrix is a data structure for efficiently storing and operating on large-scale sparse data. When most elements in a matrix are zero, sparse storage can significantly save memory and computing resources.

The `sparse` package provides support for three mainstream sparse matrix formats: CSR, CSC, and COO, as well as various special sparse matrix implementations.

## 核心类 / Core Classes

### ISparseMatrix 接口 / ISparseMatrix Interface

```java
import com.yishape.lab.math.linalg.sparse.ISparseMatrix;
import com.yishape.lab.math.linalg.sparse.SparseFormat;
import com.yishape.lab.math.linalg.Linalg;
```

## 主要功能 / Main Features

### 1. 从稠密矩阵创建 / Create from Dense Matrix

```java
double[][] dense = {
    {1.0, 0.0, 0.0, 0.0},
    {0.0, 2.0, 0.0, 0.0},
    {0.0, 0.0, 3.0, 0.0},
    {0.0, 0.0, 0.0, 4.0}
};

// 从稠密数组创建稀疏矩阵（自动检测非零元素）/ Create sparse matrix from dense array
ISparseMatrix sparse = Linalg.sparse(dense);

// 从稠密数组创建稀疏矩阵（指定阈值）/ Create with tolerance
ISparseMatrix sparse2 = Linalg.sparse(dense, 1e-10);
```

### 2. 特殊稀疏矩阵创建 / Special Sparse Matrix Creation

```java
// 稀疏单位矩阵 / Sparse identity matrix
ISparseMatrix identity = Linalg.sparseEye(4);  // 4x4 单位矩阵
ISparseMatrix identity2 = Linalg.sparseEye(3, 4);  // 3x4 单位矩阵

// 稀疏对角矩阵 / Sparse diagonal matrix
double[] diag = {1.0, 2.0, 3.0, 4.0};
ISparseMatrix diagSparse = Linalg.sparseDiag(diag);  // 4x4 对角矩阵
ISparseMatrix diagSparse2 = Linalg.sparseDiag(diag, 5);  // 5x5 对角矩阵

// 对角稀疏矩阵（特殊实现，高效）/ Diagonal sparse matrix (special implementation)
ISparseMatrix diagSpecial = Linalg.diagonalSparse(diag);

// 三对角稀疏矩阵 / Tridiagonal sparse matrix
double[] lower = {1.0, 2.0, 3.0};    // 下对角线
double[] main = {4.0, 5.0, 6.0, 7.0}; // 主对角线
double[] upper = {8.0, 9.0, 10.0};    // 上对角线
ISparseMatrix tridiag = Linalg.tridiagonalSparse(lower, main, upper);

// 稀疏零矩阵 / Sparse zero matrix
ISparseMatrix zero = Linalg.zeroSparse(3, 4);  // 3x4 零矩阵
```

### 3. 从压缩格式创建 / Create from Compressed Format

```java
// COO 格式 (Coordinate) / COO format
int[] rowIdx = {0, 1, 2, 3};
int[] colIdx = {0, 1, 2, 3};
double[] values = {1.0, 2.0, 3.0, 4.0};
ISparseMatrix coo = Linalg.sparseFromCOO(rowIdx, colIdx, values, 4, 4);

// CSR 格式 (Compressed Sparse Row) / CSR format
int[] rowPtr = {0, 1, 2, 3, 4};
int[] colInd = {0, 1, 2, 3};
double[] csrValues = {1.0, 2.0, 3.0, 4.0};
ISparseMatrix csr = Linalg.sparseFromCSR(rowPtr, colInd, csrValues, 4, 4);

// CSC 格式 (Compressed Sparse Column) / CSC format
int[] cscRowInd = {0, 1, 2, 3};
int[] colPtr = {0, 1, 2, 3, 4};
double[] cscValues = {1.0, 2.0, 3.0, 4.0};
ISparseMatrix csc = Linalg.sparseFromCSC(cscRowInd, colPtr, cscValues, 4, 4);
```

### 4. 格式转换 / Format Conversion

```java
// 转换为指定格式 / Convert to specified format
ISparseMatrix csr = sparse.toFormat(SparseFormat.CSR);
ISparseMatrix csc = sparse.toFormat(SparseFormat.CSC);
ISparseMatrix coo = sparse.toFormat(SparseFormat.COO);

// 获取当前格式 / Get current format
SparseFormat format = sparse.format();
```

### 5. 基本运算 / Basic Operations

```java
// 矩阵加法 / Matrix addition
ISparseMatrix sum = sparse1.add(sparse2);

// 矩阵减法 / Matrix subtraction
ISparseMatrix diff = sparse1.sub(sparse2);

// 标量乘法 / Scalar multiplication
ISparseMatrix scaled = sparse.scale(2.0);

// 矩阵乘法 / Matrix multiplication
ISparseMatrix product = sparse1.multiply(sparse2);

// 稀疏矩阵乘向量 / Sparse matrix times vector
IVector<Double> result = sparse.mmul(vector);
```

### 6. 矩阵属性 / Matrix Properties

```java
// 非零元素个数 / Number of non-zero elements
int nnz = sparse.nnz();

// 稀疏度（零元素比例）/ Sparsity (ratio of zero elements)
double sparsity = sparse.sparsity();

// Frobenius 范数 / Frobenius norm
double norm = sparse.frobeniusNorm();

// 获取单个元素 / Get single element
double value = sparse.get(row, col);

// 转置 / Transpose
ISparseMatrix transposed = sparse.transpose();
```

### 7. 转换为稠密矩阵 / Convert to Dense

```java
// 转换为稠密矩阵接口 / Convert to dense matrix interface
IMatrix<Double> dense = sparse.toDense();

// 转换为二维数组 / Convert to 2D array
double[][] denseArray = sparse.toDenseArray();
```

## 特殊稀疏矩阵 / Special Sparse Matrices

### DiagonalSparseMatrix

对角矩阵，只在主对角线上有非零元素。

```java
// 创建对角稀疏矩阵 / Create diagonal sparse matrix
double[] diag = {1.0, 2.0, 3.0};
ISpecialSparseMatrix.DiagonalSparseMatrix diagMat =
    new ISpecialSparseMatrix.DiagonalSparseMatrix(diag);

// 获取对角线元素 / Get diagonal elements
double[] diagValues = diagMat.getDiagonal();
```

### TridiagonalSparseMatrix

三对角矩阵，只有主对角线及其上下相邻对角线上有非零元素。

```java
// 创建三对角稀疏矩阵 / Create tridiagonal sparse matrix
double[] lower = {1.0, 2.0};    // 下对角线
double[] main = {4.0, 5.0, 6.0}; // 主对角线
double[] upper = {7.0, 8.0};    // 上对角线
ISpecialSparseMatrix.TridiagonalSparseMatrix triMat =
    new ISpecialSparseMatrix.TridiagonalSparseMatrix(lower, main, upper);
```

### IdentitySparseMatrix

单位矩阵的特殊稀疏实现。

```java
// 创建稀疏单位矩阵 / Create sparse identity matrix
ISpecialSparseMatrix.IdentitySparseMatrix identity =
    new ISpecialSparseMatrix.IdentitySparseMatrix(4);  // 4x4
```

### ZeroSparseMatrix

零矩阵的稀疏实现（不存储任何元素）。

```java
// 创建稀疏零矩阵 / Create sparse zero matrix
ISpecialSparseMatrix.ZeroSparseMatrix zero =
    new ISpecialSparseMatrix.ZeroSparseMatrix(3, 4);  // 3x4
```

## 稀疏格式说明 / Sparse Format Details

### COO (Coordinate)

最简单的格式存储每个非零元素的行索引、列索引和值。适合构建阶段，转换到 CSR/CSC 后更高效。

**优点**: 构建简单，容易理解
**缺点**: 存储效率较低，不适合算术运算

### CSR (Compressed Sparse Row)

按行压缩存储，非零元素按行组织。适合行优先操作和矩阵-向量乘法。

**优点**: 行操作高效，存储紧凑
**缺点**: 列操作较慢

### CSC (Compressed Sparse Column)

按列压缩存储，非零元素按列组织。适合列优先操作。

**优点**: 列操作高效，存储紧凑
**缺点**: 行操作较慢

## 性能对比 / Performance Comparison

| 操作 | 稠密矩阵 | 稀疏矩阵 (10% 密度) |
|------|---------|-------------------|
| 存储 | O(n²) | O(nnz) |
| 矩阵-向量乘 | O(n²) | O(nnz) |
| 矩阵-矩阵乘 | O(n³) | O(n² × k) |

## 与 SciPy 对照表 / SciPy Mapping

| 功能 / Function | YiShape | SciPy |
|---------------|---------|-------|
| 从稠密创建 / From dense | `Linalg.sparse(dense)` | `scipy.sparse.csr_matrix(dense)` |
| 单位矩阵 / Identity | `Linalg.sparseEye(n)` | `scipy.sparse.eye(n)` |
| 对角矩阵 / Diagonal | `Linalg.sparseDiag(v)` | `scipy.sparse.diags(v)` |
| COO 格式 / COO format | `Linalg.sparseFromCOO(...)` | `scipy.sparse.coo_matrix(...)` |
| CSR 格式 / CSR format | `Linalg.sparseFromCSR(...)` | `scipy.sparse.csr_matrix(...)` |
| 格式转换 / Convert format | `A.toFormat(CSR)` | `A.tocsr()` |
| 转密集 / To dense | `A.toDense()` | `A.toarray()` |

## 稀疏自动微分 / Sparse Autodiff

`ISparseVariable` 接口（通过 `Autodiff.sparseVariable()` 创建）支持稀疏矩阵的自动微分，梯度在反向传播中保持稀疏结构：

```java
import com.yishape.lab.math.optimize.autodiff.Autodiff;

var A = Autodiff.sparseVariable(sparseMatrix);
var y = A.matmul(x);  // sparse @ dense → IVariable
var loss = y.square().sum();
loss.backward();       // 梯度通过稀疏结构高效传播
var gradA = A.getGradient();  // 保持稀疏格式
```

## 稀疏迭代求解器 / Sparse Iterative Solvers

### 预条件子接口 / Preconditioner Interface

```java
import com.yishape.lab.math.linalg.sparse.ISparsePreconditioner;

// 预条件子接口: factor(A) 分解矩阵, apply(r) 应用 M^{-1}
public interface ISparsePreconditioner {
    IVector<Double> apply(IVector<Double> r);
    void factor(ISparseMatrix A);
}
```

### ILU(0) 预条件子 / ILU(0) Preconditioner

```java
// ILU(0) — 无填充不完全 LU 分解
ISparsePreconditioner ilu = Linalg.sparseILU();

// ILUT — 带阈值截断
ISparsePreconditioner ilut = Linalg.sparseILU(1e-4);
```

### 共轭梯度求解器 (CG) / Conjugate Gradient Solver

```java
import com.yishape.lab.math.linalg.sparse.ISparseLinearSolver;

// CG (带 ILU 预条件) / CG with ILU preconditioner
ISparseLinearSolver cg = Linalg.sparseSolverCG(1e-8, 1000, Linalg.sparseILU());
IVector<Double> x = cg.solve(A, b);  // A 须为 SPD

// CG (无预条件，默认 Jacobi) / CG without preconditioner (default Jacobi)
ISparseLinearSolver cg2 = Linalg.sparseSolverCG(1e-8, 1000, null);
```

### 双共轭梯度稳定求解器 (BiCGStab) / Biconjugate Gradient Stabilized Solver

```java
// BiCGStab (带 ILU 预条件) / BiCGStab with ILU preconditioner
ISparseLinearSolver bicgstab = Linalg.sparseSolverBiCGStab(1e-8, 1000, Linalg.sparseILU());
IVector<Double> x = bicgstab.solve(A, b);  // 适用于非对称矩阵 / For non-symmetric matrices
```

### 重启 GMRES(m) 求解器 / Restarted GMRES(m) Solver

```java
// GMRES(30) — 重启 Arnoldi + MGS 正交化 + Givens 旋转
ISparseLinearSolver gmres = Linalg.sparseSolverGMRES(1e-8, 1000, 30, Linalg.sparseILU());
IVector<Double> x = gmres.solve(A, b);

// 获取残差历史 / Get residual history
double[] residuals = ((SparseGMRESSolver) gmres).getResidualHistory();
```

### 自动求解器选择 / Automatic Solver Selection

`LinearSystemSolver.solve()` 自动检测稀疏矩阵并选择合适的求解器：

- `ISpecialSparseMatrix`（对角/三对角）→ 专用快速路径
- 对称正定 `ISparseMatrix` → CG + ILU
- 非对称 `ISparseMatrix` → GMRES + ILU
- 所有稀疏路径失败时回退到稠密求解器

```java
// 自动稀疏检测 / Automatic sparse detection
IMatrix<Double> X = Linalg.solveLinearSystem(sparseA, denseB);
```

### SciPy 求解器对照 / SciPy Solver Mapping

| 求解器 | YiShape | SciPy |
|--------|---------|-------|
| CG | `Linalg.sparseSolverCG(...)` | `scipy.sparse.linalg.cg()` |
| BiCGStab | `Linalg.sparseSolverBiCGStab(...)` | `scipy.sparse.linalg.bicgstab()` |
| GMRES | `Linalg.sparseSolverGMRES(...)` | `scipy.sparse.linalg.gmres()` |
| ILU 预条件 | `Linalg.sparseILU()` | `scipy.sparse.linalg.spilu()` |

## 注意事项 / Notes

1. **稀疏度阈值**: 非零元素检测使用容差值，默认为 1e-10
2. **格式选择**:
   - 构建阶段用 COO
   - 运算阶段用 CSR/CSC
   - 列操作用 CSC，行操作用 CSR
3. **内存**: 稀疏矩阵在密度低于 30-50% 时开始显示内存优势
4. **性能**: 稀疏矩阵-向量乘法通常比稠密快 10-100 倍（高稀疏度时）
