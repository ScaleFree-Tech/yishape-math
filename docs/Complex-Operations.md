# 复数运算 (Complex Number Operations)

## 概述 / Overview

`IComplexMatrix` 和 `IComplexVector` 接口提供了完整的复数矩阵和向量运算功能，基于 `Complex` 类实现。适用于信号处理、傅里叶变换、量子计算等领域。

`IComplexMatrix` and `IComplexVector` interfaces provide comprehensive complex matrix and vector operations, implemented based on the `Complex` class. Suitable for signal processing, Fourier transforms, quantum computing, and other fields.

## 核心类 / Core Classes

### Complex 类 / Complex Class

`com.yishape.lab.math.core.Complex` 是复数的基础类，提供复数运算功能。

```java
import com.yishape.lab.math.core.Complex;

// 创建复数 / Create complex numbers
Complex c1 = new Complex(3.0, 4.0);  // 3 + 4i
Complex c2 = Complex.of(1.0, 2.0);   // 1 + 2i

// 从极坐标创建 / Create from polar coordinates
Complex c3 = Complex.fromPolar(5.0, Math.PI / 4);  // 幅度5，相位π/4
```

### IComplexMatrix 接口 / IComplexMatrix Interface

复数矩阵接口，支持实部/虚部矩阵或极坐标创建。

```java
import com.yishape.lab.math.linalg.IComplexMatrix;
import com.yishape.lab.math.linalg.Linalg;

// 从实部和虚部创建 / Create from real and imaginary parts
double[][] real = {{1.0, 2.0}, {3.0, 4.0}};
double[][] imag = {{0.0, 1.0}, {-1.0, 0.0}};
IComplexMatrix matrix = Linalg.complexMatrix(real, imag);

// 从极坐标创建 / Create from polar coordinates
double[][] mag = {{5.0, 0.0}, {0.0, 5.0}};
double[][] phase = {{0.0, Math.PI / 2}, {-Math.PI / 2, 0.0}};
IComplexMatrix matrixPolar = Linalg.complexMatrixFromPolar(mag, phase);
```

### IComplexVector 接口 / IComplexVector Interface

复数向量接口。

```java
// 从实部和虚部创建复数向量 / Create complex vector from real and imaginary parts
double[] real = {1.0, 2.0, 3.0};
double[] imag = {0.0, 1.0, -1.0};
IComplexMatrix.IComplexVector vec = Linalg.complexVector(real, imag);
```

## 主要功能 / Main Features

### 1. 基本运算 / Basic Operations

```java
// 加法和减法 / Addition and subtraction
IComplexMatrix sum = matrix1.add(matrix2);
IComplexMatrix diff = matrix1.sub(matrix2);

// 标量乘法 / Scalar multiplication
IComplexMatrix scaled = matrix1.scale(2.0);

// 共轭 / Conjugate
IComplexMatrix conj = matrix1.conjugate();

// 转置和共轭转置 / Transpose and conjugate transpose
IComplexMatrix transposed = matrix1.transpose();
IComplexMatrix hermitian = matrix1.conjugateTranspose();
```

### 2. 矩阵乘法 / Matrix Multiplication

```java
// 复数矩阵乘法 / Complex matrix multiplication
IComplexMatrix result = matrix1.mmul(matrix2);

// 复数矩阵乘复数向量 / Complex matrix times complex vector
IComplexMatrix.IComplexVector vecResult = matrix1.mmul(vector);
```

### 3. 元素级运算 / Element-wise Operations

```java
// Hadamard 积（元素级乘法）/ Hadamard product (element-wise multiplication)
IComplexMatrix hadamard = matrix1.hadamard(matrix2);
```

### 4. 矩阵属性 / Matrix Properties

```java
// 迹 / Trace
Complex trace = matrix.trace();

// 行列式 / Determinant
Complex det = matrix.det();

// Frobenius 范数 / Frobenius norm
double norm = matrix.frobeniusNorm();

// 判断是否为 Hermitian 矩阵 / Check if Hermitian
boolean isHerm = matrix.isHermitian();
```

### 5. 求逆 / Inverse

```java
// 矩阵求逆 / Matrix inverse
IComplexMatrix inv = matrix.inv();
```

## 使用示例 / Usage Examples

### 傅里叶变换应用 / Fourier Transform Application

```java
import com.yishape.lab.math.linalg.IComplexMatrix;
import com.yishape.lab.math.linalg.Linalg;

// 创建复数矩阵模拟 FFT 结果 / Create complex matrix simulating FFT result
double[][] real = {{1.0, 0.0}, {0.5, 0.5}};
double[][] imag = {{0.0, 0.0}, {0.5, -0.5}};
IComplexMatrix fftResult = Linalg.complexMatrix(real, imag);

// 计算功率谱 / Calculate power spectrum
IComplexMatrix conj = fftResult.conjugate();
IComplexMatrix power = fftResult.hadamard(conj);

System.out.println("FFT Result: " + fftResult);
System.out.println("Power Spectrum: " + power);
```

## 与 NumPy 对照表 / NumPy Mapping

| 功能 / Function | YiShape | NumPy |
|----------------|----------|-------|
| 创建复数矩阵 / Create complex matrix | `Linalg.complexMatrix(real, imag)` | `np.array(real) + 1j * np.array(imag)` |
| 复数矩阵乘法 / Complex matmul | `A.mmul(B)` | `A @ B` |
| 共轭转置 / Conjugate transpose | `A.conjugateTranspose()` | `A.conj().T` |
| 矩阵求逆 / Matrix inverse | `A.inv()` | `np.linalg.inv(A)` |
| 行列式 / Determinant | `A.det()` | `np.linalg.det(A)` |
| Frobenius 范数 / Frobenius norm | `A.frobeniusNorm()` | `np.linalg.norm(A, 'fro')` |

## 复数自动微分 / Complex Autodiff

`IComplexVariable` 接口（通过 `Autodiff.complexVariable()` 创建）支持复数域的自动微分，使用 Wirtinger 导数：

```java
import com.yishape.lab.math.optimize.autodiff.Autodiff;
import com.yishape.lab.math.linalg.Linalg;

var z = Autodiff.complexVariable(complexVec);
var loss = z.exp().sum();   // 复数运算自动追踪梯度
loss.backward();             // Wirtinger 导数反向传播
var grad = z.getGradient();
```

## 注意事项 / Notes

1. **精度**: 复数运算可能累积误差，大型矩阵运算注意数值稳定性
2. **内存**: 复数矩阵占用实数矩阵 2 倍内存
3. **性能**: 复数乘法比实数乘法慢 2-4 倍
