# API 参考手册 (API Reference)

## 概述 / Overview

本文档提供了 `yishape-math` 包中所有公共类和方法的详细API参考。yishape-math是一个功能强大的Java数学计算库，提供线性代数、统计分析、机器学习、数据可视化、优化算法、降维算法、时间序列分析等全面的数学计算功能。

This document provides detailed API reference for all public classes and methods in the `yishape-math` package. yishape-math is a powerful Java mathematical computing library that provides comprehensive mathematical computing capabilities including linear algebra, statistical analysis, machine learning, data visualization, optimization algorithms, dimensionality reduction algorithms, and time series analysis.

## 用户指南 / User Guide

### 🎯 推荐使用的公开API / Recommended Public APIs

#### 核心工厂类 / Core Factory Classes
- **`Linalg`** - 线性代数工厂类，创建矩阵和向量（**推荐使用**） / Linear algebra factory for matrices and vectors (**recommended**)
- **`Stats`** - 统计工厂类，创建概率分布和统计工具（**推荐使用**） / Statistics factory for distributions and tools (**recommended**)
- **`Plots`** - 绘图工厂类，创建各种图表（**推荐使用**） / Plotting factory for charts (**recommended**)
- **`RereMathUtil`** - 数学工具类，提供常用数学函数 / Math utilities
- **`TimeSeriesAnalyzer`** - 时间序列分析工具类（**推荐使用**） / Time series analysis utilities (**recommended**)

#### 核心接口 / Core Interfaces
- **`IMatrix<T>`** - 矩阵操作接口，用户操作矩阵的主要接口 / Matrix operations; primary matrix API
- **`IVector<T>`** - 向量操作接口，用户操作向量的主要接口 / Vector operations; primary vector API
- **`IPlot`** - 绘图接口，用户创建图表的主要接口 / Plotting API
- **`IRegression`** - 回归算法接口 / Regression interface
- **`IClassifier`** - 分类算法接口（`com.yishape.lab.math.ml.cls`） / Classification interface (`com.yishape.lab.math.ml.cls`)

#### 算法实现类 / Algorithm Implementation Classes
- **`RereLinearRegression`** - 线性回归 / Linear regression
- **`RereLogisticRegression`** - 逻辑回归 / Logistic regression
- **`RereLBFGS`** - LBFGS优化器 / L-BFGS optimizer
- **`RerePCA`** - 主成分分析 / PCA
- **`RereSVD`** - 奇异值分解 / SVD
- **`RereTSNE`** - t-SNE降维 / t-SNE
- **`RereUMAP`** - UMAP降维 / UMAP
- **`TimeSeriesForecasting`** - 时间序列预测 / Time series forecasting
- **`TimeSeriesFiltering`** - 时间序列滤波 / Time series filtering
- **`TimeSeriesDecomposition`** - 时间序列分解 / Time series decomposition

#### 数据结构 / Data Structures
- **`DataFrame`** - 数据框类 / Data frame
- **`Column`** - 列数据结构 / Column
- **`Tuple2` 到 `Tuple9`** - 元组类 / Tuple types
- **`TimeSeriesData`** - 时间序列数据类 / Time series data

#### 可视化类 / Visualization Classes
- **`RerePlot`** - 绘图实现类 / Plot implementation
- **`ColorPalette`** - 颜色调色板 / Color palette
- **`ThemeManager`** - 主题管理器 / Theme manager

### ⚠️ 内部实现类 / Internal Implementation Classes

以下类为内部实现类，用户通常不需要直接使用：

The following classes are internal implementation classes that users typically don't need to use directly:

- **对外契约 / Public contract**：矩阵/向量以 **`IMatrix` / `IVector`** 及精度特化 **`IFloatMatrix` / `IDoubleMatrix` / `IFloatVector` / `IDoubleVector`** 为准；`linalg` 包内另有具体实现类，替换实现时勿依赖具体类名。 / Matrices and vectors: **`IMatrix` / `IVector`** and precision types **`IFloatMatrix` / `IDoubleMatrix` / `IFloatVector` / `IDoubleVector`**; do not depend on concrete classes when swapping implementations.
- **计算工具类 / Compute helpers**: `GPUComputeFloatUtils`, `GPUComputeDoubleUtils`, `CPUComputeFloatUtils`, `CPUComputeDoubleUtils`, `GPUConfig`
- **分布实现类 / Distribution impls**: `NormalDistribution`, `StudentDistribution`, `UniformDistribution`, etc.
- **内部工具类 / Internal helpers**: `SliceExpressionParser`, `StyleConverter`, `UniversalStyleApplier`, etc.

这些类由框架内部自动调用，用户通过公开API使用功能时，框架会自动选择合适的实现。

These classes are automatically called by the framework internally. When users use features through public APIs, the framework automatically selects the appropriate implementation.

## 主要功能模块 / Main Functional Modules

### 1. 线性代数 (Linear Algebra)
- **Linalg**: 线性代数工厂类，提供矩阵和向量的创建方法（推荐使用） / Factory for matrix and vector creation (**recommended**)
- **IMatrix/IVector**: 矩阵和向量的核心接口，用户操作的主要接口 / Core matrix and vector APIs

### 2. 统计分析 (Statistical Analysis)
- **Stats**: 统计工厂类，提供概率分布创建方法（推荐使用） / Statistics factory (**recommended**)
- **概率分布**: 支持14种重要概率分布（正态、t分布、卡方、F分布、Beta、Gamma、泊松等） / 14+ distributions (normal, t, chi-squared, F, Beta, Gamma, Poisson, …)
- **假设检验**: 参数估计和假设检验功能 / Hypothesis testing and estimation
- **方差分析**: 支持常见的方差分析功能 / ANOVA

### 3. 机器学习 (Machine Learning)
- **RereLinearRegression**: 线性回归实现，支持L1/L2/ElasticNet正则化 / Linear regression with L1/L2/ElasticNet
- **RereLogisticRegression**: 逻辑回归实现，支持二分类和多分类 / Logistic regression (binary & multiclass)
- **`ML`**: 机器学习工厂与交叉验证、指标等入口（**推荐使用**） / ML factory, CV, metrics (**recommended**)
- **IClassifier / IRegression**: 分类与回归模型接口 / Classification and regression interfaces

### 4. 数据可视化 (Data Visualization)
- **Plots**: 绘图静态工厂类（推荐使用） / Plotting factory (**recommended**)
- **RerePlot**: 基于ECharts的绘图实现类 / ECharts-based plot impl
- **IPlot**: 绘图接口，支持20+种图表类型 / Plot API (20+ chart types)
- **ColorPalette**: 颜色调色板管理 / Color palettes
- **ThemeManager**: 主题管理器 / Themes

### 5. 优化算法 (Optimization Algorithms)
- **RereLBFGS**: 有限内存BFGS优化算法 / L-BFGS
- **IOptimizer**: 优化器接口 / Optimizer interface

### 6. 降维算法 (Dimensionality Reduction)
- **RerePCA**: 主成分分析 / PCA
- **RereSVD**: 奇异值分解 / SVD
- **RereTSNE**: t分布随机邻域嵌入 / t-SNE
- **RereUMAP**: 均匀流形近似和投影 / UMAP

### 7. 数据结构 (Data Structures)
- **DataFrame**: 数据框类，支持CSV读写和矩阵转换 / Data frame with CSV I/O and matrix conversion
- **Column**: 列数据结构 / Column
- **Tuple系列**: 泛型元组类（Tuple2-Tuple9） / Generic tuples

### 8. 数学工具 (Mathematical Utilities)
- **RereMathUtil**: 数学工具类，提供类型转换、随机数生成、统计函数等 / Conversions, RNG, math helpers

### 9. 时间序列分析 (Time Series Analysis)
- **TimeSeriesData**: 时间序列数据类，支持单变量和多变量时间序列 / Uni- and multivariate series
- **TimeSeriesAnalyzer**: 统一时间序列分析工具类，提供完整分析功能 / Unified analysis
- **TimeSeriesForecasting**: 时间序列预测类，支持多种预测方法 / Forecasting
- **TimeSeriesFiltering**: 时间序列滤波类，提供多种滤波方法 / Filtering
- **TimeSeriesDecomposition**: 时间序列分解类，支持趋势、季节性分解 / Decomposition (trend, seasonal)

### 10. 信号处理 (Signal Processing)
- **SignalGeneration**: 信号生成类，提供各种波形和噪声信号生成 / Waveforms and noise
- **SignalFiltering**: 信号滤波类，提供多种数字滤波器 / Digital filters
- **SignalAnalysis**: 信号分析类，提供频谱分析、功率谱密度等功能 / Spectral analysis, PSD
- **SignalUtilities**: 信号工具类，提供窗函数、重采样、信号检测等工具 / Windows, resampling, detection
- **WaveletAnalysis**: 小波分析类，提供小波变换和小波分析功能 / Wavelets
- **RereFFT**: 快速傅里叶变换类，提供FFT和IFFT功能 / FFT / IFFT
- **RereDCT**: 离散余弦变换类，提供DCT变换和压缩功能 / DCT
- **RereHilbert**: 希尔伯特变换类，提供解析信号分析功能 / Hilbert
- **SignalVisualizer**: 信号可视化类，提供信号数据可视化功能 / Signal visualization
- **Complex**: 复数类，提供复数运算功能 / Complex numbers

### 11. 广播与稠密数组相关 API / Broadcasting and dense-array-related APIs
- **`IMatrix`**：`broadcastShape` / `broadcastTo` / `broadcastElementWise`（`double[][]` / `IMatrix<Double>` / `float[][]` 等）；两个操作数均为 **`IMatrix<Float>`** 时用 **`IFloatMatrix.broadcastElementWise`**（泛型擦除） / **`IMatrix`** broadcasting; for two **`IMatrix<Float>`** operands use **`IFloatMatrix.broadcastElementWise`** (type erasure).
- **`IDoubleMatrix`** / **`IFloatMatrix`**：与 `IMatrix` 对齐的 `broadcast*`；`IFloatMatrix.broadcastElementWise(IMatrix<Float>, IMatrix<Float>, …)` / `broadcast*` aligned with **`IMatrix`**; **`IFloatMatrix.broadcastElementWise`** for float–float element-wise.
- **`IMatrix` / `IVector`**：矩阵乘、转置、行和；向量 `fancyGet` / `booleanGet` / `repeat` 等 / Matrix ops; vector indexing helpers.
- **`IVector`**：`histogram`、`digitize`、`polyfit`、`where`（原始数组 `double[]`/`float[]`；向量样本用 **`IVector<? extends Number>`**；**`polyfit`（两向量）** 返回 **`IVector<? extends Number>`**）；**`IVector.HistogramResult`** / Static helpers; **`polyfit`** on two vectors returns **`IVector<? extends Number>`**; **`IVector.HistogramResult`**.
- **`IDoubleVector`** / **`IFloatVector`**：与 **`IVector`** 相同签名的静态便捷方法 / Convenience statics mirroring **`IVector`**.
- **`NpyArrayIO`**：`.npy` 读写；可与 **`Linalg.matrix`** / **`IMatrix`** 配合 / `.npy` I/O with **`Linalg.matrix`** / **`IMatrix`**.
- 实数 FFT：**`RereFFT`** / Real FFT: **`RereFFT`**.
- 示例见 [`examples/Matrix-Examples.md`](examples/Matrix-Examples.md) 文首「广播与相关 API 速查」/ Examples: opening **Quick reference: broadcasting and related APIs** in [`examples/Matrix-Examples.md`](examples/Matrix-Examples.md).

## 核心工厂类 / Core Factory Classes

### Linalg 线性代数工厂类 / Linalg Linear Algebra Factory Class

线性代数工厂类，提供创建各种矩阵和向量的静态工厂方法，通过委托给IMatrix和IVector的统一接口实现。

Linear algebra factory class providing static factory methods for creating matrices and vectors, implemented by delegating to unified IMatrix and IVector interfaces.

```java
public class Linalg {
    // ========== 矩阵创建方法 / Matrix Creation Methods ==========
    
    // 从数组创建矩阵 / Create matrix from arrays
    static IMatrix matrix(float[][] array);      // Create matrix from 2D float array
    static IMatrix matrix(double[][] array);     // Create matrix from 2D double array
    // 无 int[][] 重载：请转为 float[][] / double[][] 或使用包装类数组
    
    // 特殊矩阵创建 / Special matrix creation
    static IMatrix zeros(int rows, int cols);    // Create zero matrix
    static IMatrix ones(int rows, int cols);     // Create matrix of ones
    static IMatrix eye(int size);                // Create identity matrix
    static IMatrix rand(int rows, int cols);     // Create random matrix
    static IMatrix randn(int rows, int cols);    // Create normal random matrix
    static IMatrix diag(IVector diagonal);       // Create diagonal matrix
    
    // ========== 向量创建方法 / Vector Creation Methods ==========
    
    // 从数组创建向量 / Create vector from arrays
    static IVector vector(float[] array);        // Create vector from float array
    static IVector vector(double[] array);       // Create vector from double array
    // 无 int[] 重载：请转为 float[] / double[] 或使用 IVector.of(int[]) 等工厂
    
    // 特殊向量创建 / Special vector creation
    static IVector zeros(int length);            // Create zero vector
    static IVector ones(int length);             // Create vector of ones
    static IVector rand(int length);             // Create random vector
    static IVector randn(int length);            // Create normal random vector
    static IVector range(int start, int end);    // Create range vector
    static IVector range(int start, int end, int step); // Create range vector with step
    static IVector linspace(float start, float stop, int num); // Create linear space vector
    static IVector logspace(float start, float stop, int num); // Create logarithmic space vector

    // Kronecker 积，等价于 a.kron(b) / NumPy numpy.kron
    static <T extends Number> IMatrix<T> kron(IMatrix<T> a, IMatrix<T> b);
}
```

### Stats 统计工厂类 / Stats Statistical Factory Class

统计工厂类，提供创建各种常用概率分布对象的静态工厂方法。该类是统计学计算的核心入口点，支持连续型分布和离散型分布两大类。

Statistical factory class providing static factory methods for creating various common probability distribution objects. This class is the core entry point for statistical calculations, supporting continuous and discrete distributions.

```java
public class Stats {
    // ========== 连续型分布 / Continuous Distributions ==========
    
    // 正态分布 / Normal distribution
    static NormalDistribution norm();                    // Standard normal distribution (mean=0, stdDev=1)
    static NormalDistribution norm(double mean, double stdDev); // Custom normal distribution
    
    // t分布 / Student's t-distribution
    static StudentDistribution t(double degreesOfFreedom); // t-distribution
    
    // 均匀分布 / Uniform distribution
    static UniformDistribution uniform(double min, double max); // Uniform distribution
    
    // 指数分布 / Exponential distribution
    static ExponentialDistribution exponential(double lambda); // Exponential distribution
    
    // 卡方分布 / Chi-squared distribution
    static Chi2Distribution chi2(double degreesOfFreedom); // Chi-squared distribution
    
    // F分布 / F-distribution
    static FDistribution f(double df1, double df2); // F-distribution
    
    // Beta分布 / Beta distribution
    static BetaDistribution beta(double alpha, double beta); // Beta distribution
    
    // Gamma分布 / Gamma distribution
    static GammaDistribution gamma(double shape, double scale); // Gamma distribution
    
    // ========== 离散型分布 / Discrete Distributions ==========
    
    // 伯努利分布 / Bernoulli distribution
    static BernoulliDistribution bernoulli(double p); // Bernoulli distribution
    
    // 二项分布 / Binomial distribution
    static BinomialDistribution binomial(int n, double p); // Binomial distribution
    
    // 离散均匀分布 / Discrete uniform distribution
    static DiscreteUniformDistribution discreteUniform(int min, int max); // Discrete uniform distribution
    
    // 几何分布 / Geometric distribution
    static GeometricDistribution geometric(double p); // Geometric distribution
    
    // 负二项分布 / Negative binomial distribution
    static NegativeBinomialDistribution negativeBinomial(int r, double p); // Negative binomial distribution
    
    // 泊松分布 / Poisson distribution
    static PoissonDistribution poisson(double lambda); // Poisson distribution
    
    // ========== 统计工具 / Statistical Tools ==========
    
    // 参数估计 / Parameter estimation
    static ParameterEstimation estimator; // Parameter estimation tool
    
    // 假设检验 / Hypothesis testing
    static HypothesisTesting testor; // Hypothesis testing tool

    // 方差分析/ Analysis of variance
    static ANOVA anova; // ANOVA testing tool
}
```

### Plots 绘图工厂类 / Plots Visualization Factory Class

绘图静态工厂类，提供创建各种图表类型的静态方法。

Static factory class for creating various chart types.

```java
public final class Plots {
    // ========== 基础工厂方法 / Basic Factory Methods ==========
    
    // 创建绘图对象 / Create plot objects
    static RerePlot of();                           // Create default plot
    static RerePlot of(int width, int height);      // Create plot with specified size
    static RerePlot of(int width, int height, String theme); // Create plot with size and theme
    
    // ========== 图表类型专用工厂方法 / Chart Type Specific Factory Methods ==========
    
    // 线图 / Line charts
    static RerePlot line(IDoubleVector x, IDoubleVector y); // Line chart
    static RerePlot line(IDoubleVector x);                  // Single vector line chart
    static RerePlot line(IDoubleVector x, IDoubleVector y, List<String> hue); // Multi-line chart
    
    // 散点图 / Scatter plots
    static RerePlot scatter(IDoubleVector x, IDoubleVector y); // Scatter plot
    static RerePlot scatter(IDoubleVector x, IDoubleVector y, List<String> hue); // Multi-group scatter plot
    
    // 饼图 / Pie charts
    static RerePlot pie(IDoubleVector x); // Pie chart
    
    // 柱状图 / Bar charts
    static RerePlot bar(IDoubleVector x); // Bar chart
    static RerePlot bar(IDoubleVector x, List<String> hue); // Grouped bar chart
    
    // 直方图 / Histograms
    static RerePlot hist(IDoubleVector x, boolean fittingLine); // Histogram
    
    // 极坐标图 / Polar charts
    static RerePlot polarBar(IDoubleVector data, List<String> categories); // Polar bar chart
    static RerePlot polarLine(IDoubleVector data, List<String> categories); // Polar line chart
    static RerePlot polarScatter(IDoubleVector data, List<String> categories); // Polar scatter chart
    
    // 高级图表 / Advanced charts
    static RerePlot heatmap(IDoubleMatrix data, List<String> xLabels, List<String> yLabels); // Heatmap
    static RerePlot radar(IDoubleVector data, List<String> indicators); // Radar chart
    static RerePlot gauge(double value, double max, double min); // Gauge chart
    static RerePlot boxplot(IDoubleVector data); // Box plot
    static RerePlot violin(IDoubleVector data); // Violin plot
    static RerePlot candlestick(List<Map<String, Object>> data); // Candlestick chart
    static RerePlot funnel(List<Map<String, Object>> data); // Funnel chart
    static RerePlot sankey(List<Map<String, Object>> data); // Sankey diagram
    static RerePlot sunburst(List<Map<String, Object>> data); // Sunburst chart
    static RerePlot treemap(List<Map<String, Object>> data); // Treemap chart
    static RerePlot tree(List<Map<String, Object>> data); // Tree chart
    static RerePlot graph(List<Map<String, Object>> nodes, List<Map<String, Object>> links); // Graph chart
    static RerePlot parallel(IDoubleMatrix data, List<String> dimensions); // Parallel coordinates
    static RerePlot themeRiver(List<Map<String, Object>> data, List<String> categories); // Theme river chart
}
```

## 核心接口 / Core Interfaces

### IVector 接口 / IVector Interface

The core interface for vector operations, providing comprehensive mathematical operations for one-dimensional data structures.

向量操作的核心接口，为一维数据结构提供全面的数学运算。

```java
public interface IVector {
    // 工厂方法 / Factory methods
    static IVector of(float[] array);        // Create vector from float array
    static IVector of(double[] array);       // Create vector from double array
    static IVector of(int[] array);          // Create vector from int array
    static IVector zeros(int length);        // Create zero vector
    static IVector ones(int length);         // Create vector of ones
    static IVector rand(int length);         // Create random vector
    static IVector randn(int length);        // Create normal random vector
    
    // 基本属性 / Basic properties
    int length();                            // Get vector length
    float[] toArray();                       // Convert to float array
    double[] toDoubleArray();                // Convert to double array
    int[] toIntArray();                      // Convert to int array
    
    // 元素访问 / Element access
    float get(int index);                    // Get element at index
    IVector set(int index, float value);        // Set element at index
    IVector slice(int start, int end);       // Get slice of vector
    IVector slice(int[] indices);            // Get elements at specific indices
    
    // 基本数学运算 / Basic mathematical operations
    IVector add(IVector other);              // Vector addition
    IVector add(float scalar);               // Scalar addition
    IVector subtract(IVector other);         // Vector subtraction
    IVector subtract(float scalar);          // Scalar subtraction
    IVector multiply(IVector other);         // Element-wise multiplication
    IVector multiply(float scalar);          // Scalar multiplication
    IVector divide(IVector other);           // Element-wise division
    IVector divide(float scalar);            // Scalar division
    
    // 线性代数运算 / Linear algebra operations
    float dot(IVector other);                // Dot product
    float norm1();                           // L1 norm
    float norm2();                           // L2 norm
    float normInf();                         // Infinity norm
    IVector normalize();                     // Normalize vector
    float cosine(IVector other);             // Cosine similarity
    float euclideanDistance(IVector other);  // Euclidean distance
    float manhattanDistance(IVector other);  // Manhattan distance
    
    // 统计函数 / Statistical functions
    float sum();                             // Sum of elements
    float mean();                            // Mean of elements
    float variance();                        // Variance of elements
    float std();                             // Standard deviation
    float min();                             // Minimum value
    float max();                             // Maximum value
    int argmin();                            // Index of minimum
    int argmax();                            // Index of maximum
    
    // 通用函数 / Universal functions
    IVector abs();                           // Absolute value
    IVector sqrt();                          // Square root
    IVector exp();                           // Exponential
    IVector log();                           // Natural logarithm
    IVector sin();                           // Sine
    IVector cos();                           // Cosine
    IVector tan();                           // Tangent
    IVector pow(float exponent);             // Power
    
    // 比较操作 / Comparison operations
    IVector equals(IVector other);           // Element-wise equality
    IVector greater(IVector other);          // Element-wise greater than
    IVector less(IVector other);             // Element-wise less than
    IVector greaterEqual(IVector other);     // Element-wise greater or equal
    IVector lessEqual(IVector other);        // Element-wise less or equal
    
    // 排序和查找 / Sorting and searching
    IVector sort();                          // Sort in ascending order
    IVector sortDescending();                // Sort in descending order
    int[] argsort();                         // Get sort indices (ascending)
    int[] argsortDescending();               // Get sort indices (descending)
    boolean contains(float value);           // Check if value exists
    int indexOf(float value);                // Get index of value
}
```

### IMatrix 接口 / IMatrix Interface

The core interface for matrix operations, providing comprehensive mathematical operations for two-dimensional data structures.

矩阵操作的核心接口，为二维数据结构提供全面的数学运算。

```java
public interface IMatrix {
    // 工厂方法 / Factory methods
    static IMatrix of(float[][] array);      // Create matrix from 2D float array
    static IMatrix of(double[][] array);     // Create matrix from 2D double array
    static IMatrix of(int[][] array);        // Create matrix from 2D int array
    static IMatrix zeros(int rows, int cols); // Create zero matrix
    static IMatrix ones(int rows, int cols);  // Create matrix of ones
    static IMatrix eye(int size);            // Create identity matrix
    static IMatrix rand(int rows, int cols); // Create random matrix
    static IMatrix randn(int rows, int cols); // Create normal random matrix
    static IMatrix diag(IVector diagonal);   // Create diagonal matrix
    
    // 基本属性 / Basic properties
    int getRowNum();                         // Get number of rows
    int getColNum();                         // Get number of columns
    float[][] toArray();                     // Convert to 2D float array
    double[][] toDoubleArray();              // Convert to 2D double array
    int[][] toIntArray();                    // Convert to 2D int array
    
    // 元素访问 / Element access
    float get(int row, int col);             // Get element at (row, col)
    IVector set(int row, int col, float value); // Set element at (row, col)
    IVector getRow(int row);                 // Get row vector
    IVector getColumn(int col);              // Get column vector
    IVector setRow(int row, IVector vector);    // Set row vector
    IVector setColumn(int col, IVector vector); // Set column vector
    IMatrix slice(int startRow, int endRow, int startCol, int endCol); // Get submatrix
    
    // 基本数学运算 / Basic mathematical operations
    IMatrix add(IMatrix other);              // Matrix addition
    IMatrix add(float scalar);               // Scalar addition
    IMatrix subtract(IMatrix other);         // Matrix subtraction
    IMatrix subtract(float scalar);          // Scalar subtraction
    IMatrix multiply(IMatrix other);         // Element-wise multiplication
    IMatrix multiply(float scalar);          // Scalar multiplication
    IMatrix divide(IMatrix other);           // Element-wise division
    IMatrix divide(float scalar);            // Scalar division
    
    // 矩阵运算 / Matrix operations
    IMatrix mmul(IMatrix other);             // Matrix multiplication
    IVector mmul(IVector vector);            // Matrix-vector multiplication
    IMatrix transpose();                     // Matrix transpose
    IMatrix inverse();                       // Matrix inverse
    float determinant();                     // Matrix determinant
    float trace();                           // Matrix trace
    int rank();                              // Matrix rank
    
    // 线性代数 / Linear algebra
    Tuple2<IVector, IMatrix> eigenDecomposition(); // Eigenvalue decomposition
    Tuple3<IMatrix, IVector, IMatrix> svd();       // Singular value decomposition
    IMatrix pseudoInverse();                 // Moore-Penrose pseudo-inverse
    float condition();                       // Condition number
    float norm1();                           // L1 norm
    float norm2();                           // L2 norm
    float normInf();                         // Infinity norm
    float normFrobenius();                   // Frobenius norm
    
    // 统计函数 / Statistical functions
    float sum();                             // Sum of all elements
    float mean();                            // Mean of all elements
    float variance();                        // Variance of all elements
    float std();                             // Standard deviation
    IVector rowSum();                        // Sum of each row
    IVector rowMean();                       // Mean of each row
    IVector rowVariance();                   // Variance of each row
    IVector rowStd();                        // Standard deviation of each row
    IVector columnSum();                     // Sum of each column
    IVector columnMean();                    // Mean of each column
    IVector columnVariance();                // Variance of each column
    IVector columnStd();                     // Standard deviation of each column
    
    // 数据变换 / Data transformation
    IMatrix center();                        // Center matrix (subtract column means)
    IMatrix standardize();                   // Standardize matrix
    IMatrix normalize();                     // Normalize matrix
    IMatrix covariance();                    // Compute covariance matrix
    
    // 通用函数 / Universal functions
    IMatrix abs();                           // Absolute value
    IMatrix sqrt();                          // Square root
    IMatrix exp();                           // Exponential
    IMatrix log();                           // Natural logarithm
    IMatrix sin();                           // Sine
    IMatrix cos();                           // Cosine
    IMatrix tan();                           // Tangent
    IMatrix pow(float exponent);             // Power

    // 与 NumPy / 常见数值库名称对齐的补充（实际签名为泛型 IMatrix&lt;T&gt;）/ NumPy-aligned extras
    IMatrix mmul(Number scalar);             // 同 multiplyScalar；mmul(IMatrix) 仍为矩阵乘 / like A*s in NumPy
    IMatrix outer(IMatrix other);            // np.outer(A.ravel(), B.ravel()) 行优先展平 / row-major flatten
    IMatrix kron(IMatrix other);             // np.kron(A, B)
    Tuple2<IVector, IMatrix> qrEigen(); // 与 eigen() 相同；短名 / same as eigen(), short name
    Tuple2<IVector, IMatrix> qrEigenDecomposition(); // 与 qrEigen()、eigen() 相同 / same as qrEigen()
}
```

## 优化算法 / Optimization Algorithms

### IOptimizer 接口 / IOptimizer Interface

The core interface for optimization algorithms, providing methods for solving mathematical optimization problems.

优化算法的核心接口，提供求解数学最优化问题的方法。

```java
public interface IOptimizer {
    // 优化方法 / Optimization methods
    Tuple2<Float, IVector> optimize(IVector initialPoint, 
                                   IObjectiveFunction objectiveFunction, 
                                   IGradientFunction gradientFunction);
    // Solve optimization problem and return optimal value and point
    
    // 参数设置 / Parameter settings
    void setMaxIterations(int maxIterations); // Set maximum number of iterations
    void setTolerance(float tolerance);       // Set convergence tolerance
    void setVerbose(boolean verbose);         // Set verbose output mode
    
    // 状态查询 / Status queries
    int getIterationCount();                  // Get current iteration count
    float getCurrentTolerance();              // Get current tolerance value
    boolean hasConverged();                   // Check if optimization has converged
}
```

### RereLBFGS 类 / RereLBFGS Class

Implementation of the Limited-memory BFGS optimization algorithm, suitable for large-scale optimization problems.

有限内存BFGS优化算法的实现，适用于大规模最优化问题。

```java
public class RereLBFGS implements IOptimizer {
    // 构造函数 / Constructors
    public RereLBFGS();                    // Create L-BFGS optimizer with default settings
    public RereLBFGS(int memorySize);      // Create L-BFGS optimizer with specified memory size
    
    // L-BFGS特有方法 / L-BFGS specific methods
    void setMemorySize(int memorySize);    // Set memory size for limited memory
    void setLineSearchType(LineSearchType type); // Set line search algorithm type
    void setC1(float c1);                  // Set Armijo condition parameter
    void setC2(float c2);                  // Set Wolfe condition parameter
    void setMaxLineSearchIterations(int maxIterations); // Set max line search iterations
    
    // 状态查询 / Status queries
    int getMemorySize();                   // Get current memory size
    LineSearchType getLineSearchType();    // Get current line search type
    float getC1();                         // Get current C1 parameter
    float getC2();                         // Get current C2 parameter
    
    // 优化历史 / Optimization history
    List<Float> getObjectiveHistory();     // Get objective function value history
    List<Float> getGradientNormHistory();  // Get gradient norm history
    List<IVector> getPointHistory();       // Get optimization point history
}
```

### 目标函数接口 / Objective Function Interfaces

Functional interfaces for defining objective functions and their gradients in optimization problems.

用于定义最优化问题中目标函数及其梯度的函数式接口。

```java
@FunctionalInterface
public interface IObjectiveFunction {
    float compute(IVector x);  // Compute objective function value at point x
}

@FunctionalInterface  
public interface IGradientFunction {
    IVector compute(IVector x); // Compute gradient vector at point x
}
```

## 降维算法 / Dimensionality Reduction Algorithms

### RerePCA 类 / RerePCA Class

Implementation of Principal Component Analysis for dimensionality reduction and data visualization.

主成分分析算法的实现，用于降维和数据可视化。

```java
public class RerePCA {
    // 构造函数 / Constructors
    public RerePCA();                        // Create PCA with default settings
    public RerePCA(int components);          // Create PCA with specified number of components
    
    // 参数设置 / Parameter settings
    void setComponents(int components);      // Set number of principal components
    void setWhiten(boolean whiten);         // Set whether to whiten the data
    void setRandomState(long seed);         // Set random seed for reproducibility
    
    // 训练和变换 / Training and transformation
    void fit(IMatrix data);                 // Fit PCA to the data
    IMatrix transform(IMatrix data);        // Transform data using fitted PCA
    IMatrix fitTransform(IMatrix data);     // Fit and transform in one step
    IMatrix inverseTransform(IMatrix transformedData); // Inverse transform
    
    // 结果查询 / Result queries
    IVector getExplainedVariance();         // Get explained variance for each component
    IVector getExplainedVarianceRatio();    // Get explained variance ratio
    float getExplainedVarianceSum();        // Get total explained variance
    IMatrix getComponents();                // Get principal component vectors
    IVector getMean();                      // Get mean vector used for centering
    int getOptimalComponents(float varianceThreshold); // Find optimal number of components
}
```

### RereTSNE 类 / RereTSNE Class

Implementation of t-Distributed Stochastic Neighbor Embedding for nonlinear dimensionality reduction and visualization.

t分布随机邻域嵌入算法的实现，用于非线性降维和可视化。

```java
public class RereTSNE {
    // 构造函数 / Constructors
    public RereTSNE();                        // Create t-SNE with default settings
    public RereTSNE(int targetDimensions);    // Create t-SNE with target dimensions
    
    // 参数设置 / Parameter settings
    void setTargetDimensions(int dimensions); // Set target dimensionality
    void setPerplexity(float perplexity);     // Set perplexity parameter
    void setLearningRate(float learningRate); // Set learning rate
    void setMaxIterations(int maxIterations); // Set maximum iterations
    void setEarlyExaggeration(float earlyExaggeration); // Set early exaggeration factor
    void setRandomState(long seed);           // Set random seed
    void setVerbose(boolean verbose);         // Set verbose output
    
    // 降维方法 / Dimensionality reduction methods
    IMatrix dimensionReduction(IMatrix data); // Perform t-SNE dimensionality reduction
    
    // 状态查询 / Status queries
    int getIterationCount();                  // Get current iteration count
    float getCurrentCost();                   // Get current cost value
    List<Float> getCostHistory();             // Get cost history during optimization
}
```

### RereUMAP 类 / RereUMAP Class

Implementation of Uniform Manifold Approximation and Projection for manifold learning and dimensionality reduction.

均匀流形近似和投影算法的实现，用于流形学习和降维。

```java
public class RereUMAP {
    // 构造函数 / Constructors
    public RereUMAP();                        // Create UMAP with default settings
    public RereUMAP(int targetDimensions);    // Create UMAP with target dimensions
    
    // 参数设置 / Parameter settings
    void setTargetDimensions(int dimensions); // Set target dimensionality
    void setNeighbors(int neighbors);         // Set number of neighbors
    void setMinDist(float minDist);           // Set minimum distance parameter
    void setSpread(float spread);             // Set spread parameter
    void setLearningRate(float learningRate); // Set learning rate
    void setMaxIterations(int maxIterations); // Set maximum iterations
    void setRandomState(long seed);           // Set random seed
    void setVerbose(boolean verbose);         // Set verbose output
    
    // 降维方法 / Dimensionality reduction methods
    IMatrix dimensionReduction(IMatrix data); // Perform UMAP dimensionality reduction
    IMatrix transform(IMatrix newData);       // Transform new data using fitted UMAP
    
    // 状态查询 / Status queries
    int getIterationCount();                  // Get current iteration count
    float getCurrentCost();                   // Get current cost value
    IMatrix getEmbedding();                   // Get current embedding
}
```

## 机器学习算法 / Machine Learning Algorithms

### IRegression 接口 / IRegression Interface

The core interface for regression algorithms, providing methods for training and prediction.

回归算法的核心接口，提供训练和预测方法。

```java
public interface IRegression {
    // 训练方法 / Training methods
    RegressionResult fit(IMatrix features, IVector labels); // Train regression model
    
    // 预测方法 / Prediction methods
    float predict(IVector features);         // Predict single sample
    IVector predict(IMatrix features);       // Predict multiple samples
    
    // 评估方法 / Evaluation methods
    float score(IMatrix features, IVector labels); // Calculate R² score
    RegressionMetrics evaluate(IMatrix features, IVector labels); // Comprehensive evaluation
}
```

### IClassifier 接口 / IClassifier Interface

The core interface for classification algorithms: string labels, probability outputs, and batch APIs aligned with the rest of the library.

分类算法的核心接口：类别为字符串标签，支持概率预测与批量预测，与 `ML` 工厂类配合使用。

```java
public interface IClassifier extends ISerializableModel {
    ClassificationResult fit(IMatrix feature, String[] labels);
    String predict(IVector x);
    Map<String, Double> predictProb(IVector x);
    String[] predictBatch(IMatrix features);
    BatchPredictionResult predictBatchWithProbs(IMatrix features);
    boolean isTrained();
    ClassificationMetrics getMetrics();
    void setMetrics(ClassificationMetrics metrics);
}
```

### RereLinearRegression 类 / RereLinearRegression Class

Implementation of linear regression with support for various regularization techniques.

支持多种正则化技术的线性回归实现。

```java
public class RereLinearRegression implements IRegression {
    RegressionResult fit(IMatrix feature, IVector labels);
    double predict(IVector x);
    double r2ScoreOn(IMatrix features, IVector labels);
    /** λ₁、λ₂ 同时设置，内部根据符号自动推断 RegularizationType */
    void setRegularization(double lambda1, double lambda2);
    void setRegularization(RegularizationType type, double lambda1, double lambda2);
    void setLambda1(double lambda1);
    void setLambda2(double lambda2);
    RegularizationType getRegularizationType();
    double getLambda1();
    double getLambda2();
    String getRegularizationDescription();
}
```

### RereLogisticRegression 类 / RereLogisticRegression Class

Implementation of logistic regression supporting both binary and multi-class classification.

支持二分类和多分类的逻辑回归实现。

```java
public class RereLogisticRegression implements IClassifier, IGradientFunction, IObjectiveFunction, ISerializableModel {
    public enum RegularizationType { NONE, L1, L2, ELASTIC_NET }
    void setRegularization(double lambda1, double lambda2); // 推断规则与 RereLinearRegression 一致（见主文档）
    void setLambda1(double lambda1);
    void setLambda2(double lambda2);
    RegularizationType getRegularizationType();
    double getLambda1();
    double getLambda2();
    String getRegularizationDescription();
    // 训练、predict、predictProb 等见 IClassifier 与源码
}
```

### RegressionResult 类 / RegressionResult Class

Container class for regression analysis results, including model parameters and evaluation metrics.

回归分析结果的容器类，包含模型参数和评估指标。

```java
/** 线性回归训练结果（以源码 com.yishape.lab.math.ml.lr.RegressionResult 为准） */
public class RegressionResult {
    IVector getWeights();
    IVector getBias();
    double getLoss();
    double getR2Score(); // 训练集决定系数 R²，由 fit 填充
}
```

验证集 R² 使用 `RereLinearRegression.r2ScoreOn(X_val, y_val)`（需另行传入，因未参与训练）。

### ClassificationResult 类 / ClassificationResult Class

Container class for classification analysis results, including model parameters and evaluation metrics.

分类分析结果的容器类，包含模型参数和评估指标。

```java
public class ClassificationResult {
    // 构造函数 / Constructors
    public ClassificationResult(IMatrix weights, IVector bias, double loss); // Create result with parameters
    
    // 结果访问 / Result access
    IMatrix getWeights();                            // Get weight matrix
    IVector getBias();                               // Get bias vector
    double getLoss();                                // Get training loss
    double getAccuracy();                            // Get accuracy score
    double getPrecision();                           // Get precision score
    double getRecall();                              // Get recall score
    double getF1Score();                            // Get F1 score
    
    // 分类信息 / Classification information
    boolean isBinaryClassification();               // Check if binary classification
    int getNumClasses();                            // Get number of classes
    Map<String, Integer> getLabelMapping();         // Get label mapping
    String[] getClassLabels();                      // Get class labels
    
    // 混淆矩阵 / Confusion matrix
    int[][] getConfusionMatrix();                   // Get confusion matrix
    double[][] getConfusionMatrixNormalized();      // Get normalized confusion matrix
}
```

## 数据结构类 / Data Structure Classes

### DataFrame 类 / DataFrame Class

数据框类，用于处理结构化数据，支持从CSV文件读取数据并与IMatrix进行转换。

DataFrame class for handling structured data, supports reading from CSV files and conversion with IMatrix.

```java
public class DataFrame implements Serializable {
    // 构造函数 / Constructors
    public DataFrame();                                    // 创建空DataFrame / Create empty DataFrame
    public DataFrame(List<Column> columns);               // 从列列表创建 / Create from column list
    
    // 静态工厂方法 / Static factory methods
    static DataFrame readCsv(String filePath, String separator, boolean ifHasHead); // 从CSV读取 / Read from CSV
    
    // 数据访问 / Data access
    Column get(int position);                             // 按位置获取列 / Get column by position
    Column getColumnByName(String columnName);           // 按名称获取列 / Get column by name
    List<Column> getColumns();                           // 获取所有列 / Get all columns
    List<String> getColumnNames();                       // 获取列名列表 / Get column names
    List<ColumnType> getColumnTypes();                   // 获取列类型列表 / Get column types
    
    // 数据操作 / Data operations
    void addColumn(Column column);                        // 添加列 / Add column
    Column removeColumn(int position);                    // 删除列 / Remove column
    void setColumns(List<Column> columns);               // 设置列列表 / Set columns list
    void clear();                                         // 清空DataFrame / Clear DataFrame
    DataFrame copy();                                     // 复制DataFrame / Copy DataFrame
    
    // 切片操作 / Slicing operations
    DataFrame sliceColumn(int start, int end, int step); // 列切片（带步长）/ Column slicing with step
    DataFrame sliceColumn(int start, int end);           // 列切片（步长为1）/ Column slicing step=1
    DataFrame sliceColumn(int start);                    // 列切片（到末尾）/ Column slicing to end
    DataFrame slice(String rowExp, String colExp);       // 通用切片 / General slicing
    
    // 数据转换 / Data conversion
    IMatrix toMatrix();                                   // 转换为IMatrix / Convert to IMatrix
    void toCsv(String filePath);                         // 保存为CSV / Save to CSV
    
    // 基本属性 / Basic properties
    int getRowCount();                                    // 获取行数 / Get row count
    int getColumnCount();                                 // 获取列数 / Get column count
    int[] shape();                                        // 获取形状 / Get shape
    boolean isEmpty();                                    // 是否为空 / Check if empty
}
```

### Column 类 / Column Class

列类，表示数据框中的一列，包含列名、数据类型和数据。

Column class representing a column in the data frame, containing column name, data type, and data.

```java
public class Column implements Serializable {
    // 构造函数 / Constructors
    public Column();                                      // 默认构造函数 / Default constructor
    
    // 属性访问 / Property access
    String getName();                                     // 获取列名 / Get column name
    void setName(String name);                           // 设置列名 / Set column name
    ColumnType getColumnType();                          // 获取列类型 / Get column type
    void setColumnType(ColumnType columnType);           // 设置列类型 / Set column type
    List<Object> getData();                              // 获取数据 / Get data
    void setData(List<Object> data);                     // 设置数据 / Set data
    
    // 数据转换 / Data conversion
    IVector toVec();                                     // 转换为向量（仅Float类型）/ Convert to vector (Float only)
    List<String> toStringList();                         // 转换为字符串列表 / Convert to string list
}
```

### ColumnType 枚举 / ColumnType Enum

列类型枚举，定义支持的数据类型。

Column type enum defining supported data types.

```java
public enum ColumnType {
    String,    // 字符串类型 / String type
    Float      // 浮点数类型 / Float type
}
```

## 高性能计算 / High-Performance Computing

> **注意**: 以下GPU/CPU计算工具类为内部实现类，用户通常不需要直接使用。这些类由框架内部自动调用，以提供GPU加速和CPU回退功能。

> **Note**: The following GPU/CPU computing utility classes are internal implementation classes that users typically don't need to use directly. These classes are automatically called by the framework internally to provide GPU acceleration and CPU fallback functionality.

### GPU计算工具类 / GPU Computing Utility Classes

- **GPUComputeFloatUtils**: Float类型GPU计算工具（内部实现）
- **GPUComputeDoubleUtils**: Double类型GPU计算工具（内部实现）
- **CPUComputeFloatUtils**: Float类型CPU计算工具（内部实现）
- **CPUComputeDoubleUtils**: Double类型CPU计算工具（内部实现）
- **GPUConfig**: GPU配置管理（内部实现）

这些类提供了基于Aparapi框架的OpenCL GPU加速计算和CPU回退功能，支持矩阵运算、向量运算、统计运算和通用函数。用户通过`IMatrix`和`IVector`接口使用这些功能时，框架会自动选择合适的计算方式。

These classes provide OpenCL GPU acceleration based on Aparapi framework and CPU fallback functionality, supporting matrix operations, vector operations, statistical operations, and universal functions. When users use these features through `IMatrix` and `IVector` interfaces, the framework automatically selects the appropriate computation method.

## 数据可视化 / Data Visualization

### RerePlot 类 / RerePlot Class

基于ECharts的绘图实现类，提供丰富的数据可视化功能。

ECharts-based plotting implementation class providing rich data visualization capabilities.

```java
public class RerePlot implements IPlot {
    // ========== 构造函数 / Constructors ==========
    
    // 基础构造函数 / Basic constructors
    public RerePlot();                               // Create default plot
    public RerePlot(int width, int height);          // Create plot with specified size
    public RerePlot(int width, int height, String theme); // Create plot with size and theme
    
    // ========== 基础图表方法 / Basic Chart Methods ==========
    
    // 线图 / Line charts
    IPlot line(IVector x, IVector y);                // Line chart
    IPlot line(IVector x);                           // Single vector line chart
    IPlot line(IVector x, IVector y, List<String> hue); // Multi-line chart
    
    // 散点图 / Scatter plots
    IPlot scatter(IVector x, IVector y);             // Scatter plot
    IPlot scatter(IVector x, IVector y, List<String> hue); // Multi-group scatter plot
    
    // 饼图 / Pie charts
    IPlot pie(IVector x);                            // Pie chart
    
    // 柱状图 / Bar charts
    IPlot bar(IVector x);                            // Bar chart
    IPlot bar(IVector x, List<String> hue);          // Grouped bar chart
    
    // 直方图 / Histograms
    IPlot hist(IVector x, boolean fittingLine);     // Histogram
    
    // ========== 极坐标图表方法 / Polar Chart Methods ==========
    
    // 极坐标图 / Polar charts
    IPlot polarBar(IVector data, List<String> categories); // Polar bar chart
    IPlot polarLine(IVector data, List<String> categories); // Polar line chart
    IPlot polarScatter(IVector data, List<String> categories); // Polar scatter chart
    
    // ========== 高级图表方法 / Advanced Chart Methods ==========
    
    // 热力图 / Heatmap
    IPlot heatmap(IMatrix data, List<String> xLabels, List<String> yLabels); // Heatmap
    
    // 雷达图 / Radar chart
    IPlot radar(IVector data, List<String> indicators); // Radar chart
    
    // 仪表盘 / Gauge chart
    IPlot gauge(double value, double max, double min); // Gauge chart
    
    // 箱线图 / Box plot
    IPlot boxplot(IVector data);                     // Box plot
    IPlot violin(IVector data);                      // Violin plot
    
    // 金融图表 / Financial charts
    IPlot candlestick(List<Map<String, Object>> data); // Candlestick chart
    
    // 漏斗图 / Funnel chart
    IPlot funnel(List<Map<String, Object>> data);     // Funnel chart
    
    // 桑基图 / Sankey diagram
    IPlot sankey(List<Map<String, Object>> data);    // Sankey diagram
    
    // 旭日图 / Sunburst chart
    IPlot sunburst(List<Map<String, Object>> data);   // Sunburst chart
    
    // 矩形树图 / Treemap chart
    IPlot treemap(List<Map<String, Object>> data);    // Treemap chart
    
    // 树图 / Tree chart
    IPlot tree(List<Map<String, Object>> data);       // Tree chart
    
    // 关系图 / Graph chart
    IPlot graph(List<Map<String, Object>> nodes, List<Map<String, Object>> links); // Graph chart
    
    // 平行坐标图 / Parallel coordinates
    IPlot parallel(IMatrix data, List<String> dimensions); // Parallel coordinates
    
    // 主题河流图 / Theme river chart
    IPlot themeRiver(List<Map<String, Object>> data, List<String> categories); // Theme river chart
    
    // ========== 流式API方法 / Fluent API Methods ==========
    
    // 图表设置 / Chart settings
    IPlot title(String titleText);                   // Set chart title
    IPlot title(String titleText, String subtitleText); // Set title and subtitle
    IPlot xlabel(String name);                       // Set X-axis label
    IPlot ylabel(String name);                       // Set Y-axis label
    
    // 样式设置 / Style settings
    IPlot style(String styleName);                   // Set chart style
    IPlot theme(String themeName);                    // Set chart theme
    IPlot color(String colorScheme);                 // Set color scheme
    
    // 输出方法 / Output methods
    String toHtml();                                 // Export to HTML
    void saveToFile(String filePath);                // Save to file
    void show();                                      // Display chart
}
```

### ColorPalette 类 / ColorPalette Class

颜色调色板管理类，提供丰富的颜色方案。

Color palette management class providing rich color schemes.

```java
public class ColorPalette {
    // ========== 预定义调色板 / Predefined Palettes ==========
    
    // 基础调色板 / Basic palettes
    static String[] getDefaultPalette();             // Get default color palette
    static String[] getPrimaryPalette();             // Get primary color palette
    static String[] getSecondaryPalette();           // Get secondary color palette
    
    // 专业调色板 / Professional palettes
    static String[] getSeabornPalette();             // Get Seaborn-style palette
    static String[] getMatplotlibPalette();           // Get Matplotlib-style palette
    static String[] getTableauPalette();             // Get Tableau-style palette
    
    // 主题调色板 / Thematic palettes
    static String[] getWarmPalette();                 // Get warm color palette
    static String[] getCoolPalette();                // Get cool color palette
    static String[] getPastelPalette();              // Get pastel color palette
    static String[] getVibrantPalette();             // Get vibrant color palette
    
    // ========== 调色板生成 / Palette Generation ==========
    
    // 自定义调色板 / Custom palette generation
    static String[] generatePalette(int numColors);   // Generate palette with specified number of colors
    static String[] generatePalette(String baseColor, int numColors); // Generate palette from base color
    static String[] generateGradientPalette(String startColor, String endColor, int numColors); // Generate gradient palette
    
    // ========== 调色板管理 / Palette Management ==========
    
    // 调色板注册 / Palette registration
    static void registerPalette(String name, String[] colors); // Register custom palette
    static String[] getPalette(String name);          // Get palette by name
    static List<String> getAvailablePalettes();      // Get list of available palettes
    
    // 调色板验证 / Palette validation
    static boolean isValidColor(String color);        // Validate color format
    static boolean isValidPalette(String[] colors);   // Validate palette
}
```

### ThemeManager 类 / ThemeManager Class

主题管理器，提供图表主题的统一管理。

Theme manager providing unified management of chart themes.

```java
public class ThemeManager {
    // ========== 预定义主题 / Predefined Themes ==========
    
    // 基础主题 / Basic themes
    static String getDefaultTheme();                 // Get default theme
    static String getLightTheme();                   // Get light theme
    static String getDarkTheme();                    // Get dark theme
    
    // 专业主题 / Professional themes
    static String getSeabornTheme();                 // Get Seaborn-style theme
    static String getMatplotlibTheme();              // Get Matplotlib-style theme
    static String getTableauTheme();                 // Get Tableau-style theme
    
    // ========== 主题管理 / Theme Management ==========
    
    // 主题设置 / Theme settings
    static void setDefaultTheme(String themeName);   // Set default theme
    static String getCurrentTheme();                 // Get current theme
    static void applyTheme(String themeName);        // Apply theme
    
    // 主题注册 / Theme registration
    static void registerTheme(String name, Map<String, Object> themeConfig); // Register custom theme
    static Map<String, Object> getTheme(String name); // Get theme configuration
    static List<String> getAvailableThemes();        // Get list of available themes
    
    // ========== 主题配置 / Theme Configuration ==========
    
    // 主题属性 / Theme properties
    static void setThemeProperty(String themeName, String property, Object value); // Set theme property
    static Object getThemeProperty(String themeName, String property); // Get theme property
    static Map<String, Object> getThemeProperties(String themeName); // Get all theme properties
}
```

## 数学工具类 / Math Utilities

### RereMathUtil 类 / RereMathUtil Class

Utility class providing mathematical functions, type conversions, and statistical operations.

提供数学函数、类型转换和统计运算的工具类。

```java
public final class RereMathUtil {
    // ========== 类型转换 / Type Conversion ==========
    
    // 数组类型转换 / Array type conversion
    static double[] floatToDouble(float[] array);     // Convert float array to double array
    static float[] doubleToFloat(double[] array);     // Convert double array to float array
    static float[] intToFloat(int[] array);           // Convert int array to float array
    static int[] floatToInt(float[] array);           // Convert float array to int array
    static double[] intToDouble(int[] array);         // Convert int array to double array
    static int[] doubleToInt(double[] array);         // Convert double array to int array
    
    // 包装类转换 / Wrapper class conversion
    static float[] toPrimitive(Float[] array);        // Convert Float array to float array
    static double[] toPrimitive(Double[] array);      // Convert Double array to double array
    static int[] toPrimitive(Integer[] array);        // Convert Integer array to int array
    static Float[] toClassArray(float[] array);       // Convert float array to Float array
    static Double[] toClassArray(double[] array);     // Convert double array to Double array
    static Integer[] toClassArray(int[] array);       // Convert int array to Integer array
    
    // ========== 随机数生成 / Random Number Generation ==========
    
    // 随机整数生成 / Random integer generation
    static int[] generateRandomInts(int start, int end, int num); // Generate random integers
    static int[] generateRandomInts(int seed, int start, int end, int num); // Generate random integers with seed
    
    // 随机浮点数生成 / Random float generation
    static float[] generateRandomFloats(int n);       // Generate random floats
    static float[] generateRandomFloats(int seed, int n); // Generate random floats with seed
    
    // 正态分布随机数 / Normal distribution random numbers
    static double normalSample(double mean, double stdDev); // Generate normal random number
    
    // ========== 概率分布相关数学函数 / Probability Distribution Mathematical Functions ==========
    
    // 特殊函数 / Special functions
    static double gamma(double x);                    // Gamma function
    static double beta(double a, double b);          // Beta function
    static double incompleteBeta(double a, double b, double x); // Incomplete beta function
    static double incompleteGamma(double a, double x); // Incomplete gamma function
    static double regularizedIncompleteBeta(double a, double b, double x); // Regularized incomplete beta function
    static double regularizedIncompleteGamma(double a, double x); // Regularized incomplete gamma function
    
    // 误差函数 / Error functions
    static double erf(double x);                      // Error function
    static double inverseNormalCDF(double p);         // Inverse normal CDF
    
    // ========== 组合数学和统计函数 / Combinatorics and Statistical Functions ==========
    
    // 组合数学 / Combinatorics
    static long combination(int n, int k);            // Combination C(n,k)
    static double logCombination(int n, int k);      // Logarithm of combination
    static long factorial(int n);                     // Factorial
    static double logFactorial(int n);                // Logarithm of factorial
    static double stirlingNumber2(int n, int k);     // Stirling number of second kind
    
    // ========== 常用常数 / Common Constants ==========
    
    // 数学常数 / Mathematical constants
    static final double PI = Math.PI;                // Pi constant
    static final double E = Math.E;                  // Euler's number
    static final double EPS = 1e-8;                  // Machine epsilon
    static final double INF = Double.POSITIVE_INFINITY; // Positive infinity
    static final double NEG_INF = Double.NEGATIVE_INFINITY; // Negative infinity
    static final double NAN = Double.NaN;            // Not a number
}
```

## 枚举类型 / Enum Types

### RegularizationType

Enumeration defining different types of regularization for linear regression.

定义线性回归中不同类型正则化的枚举。

```java
public enum RegularizationType {
    NONE,        // 无正则化 / No regularization
    L1,          // L1正则化 (Lasso) / L1 regularization (Lasso)
    L2,          // L2正则化 (Ridge) / L2 regularization (Ridge)
    ELASTIC_NET  // 弹性网络 / Elastic Net
}
```

### LineSearchType

Enumeration defining different types of line search algorithms for optimization.

定义优化中不同类型线搜索算法的枚举。

```java
public enum LineSearchType {
    ARMIJO,      // Armijo线搜索 / Armijo line search
    WOLFE,       // Wolfe线搜索 / Wolfe line search
    STRONG_WOLFE // 强Wolfe线搜索 / Strong Wolfe line search
}
```

## 数据类型 / Data Types

### Tuple2

Generic tuple class for holding two values of different types.

用于保存两个不同类型值的泛型元组类。

```java
public class Tuple2<T1, T2> {
    public final T1 _1;                    // First element
    public final T2 _2;                    // Second element
    
    public Tuple2(T1 first, T2 second);    // Constructor
    
    // 访问方法 / Access methods
    public T1 getFirst();                  // Get first element
    public T2 getSecond();                 // Get second element
    
    // 实用方法 / Utility methods
    public String toString();              // String representation
    public boolean equals(Object obj);     // Equality comparison
    public int hashCode();                 // Hash code
}
```

### Tuple3

Generic tuple class for holding three values of different types.

用于保存三个不同类型值的泛型元组类。

```java
public class Tuple3<T1, T2, T3> {
    public final T1 _1;                    // First element
    public final T2 _2;                    // Second element
    public final T3 _3;                    // Third element
    
    public Tuple3(T1 first, T2 second, T3 third); // Constructor
    
    // 访问方法 / Access methods
    public T1 getFirst();                  // Get first element
    public T2 getSecond();                 // Get second element
    public T3 getThird();                  // Get third element
    
    // 实用方法 / Utility methods
    public String toString();              // String representation
    public boolean equals(Object obj);     // Equality comparison
    public int hashCode();                 // Hash code
}
```

## 使用指南 / Usage Guidelines

### 异常处理 / Exception Handling

Common exceptions that may be thrown by the API methods:

API方法可能抛出的常见异常：

- `IllegalArgumentException`: 参数无效时抛出 / Thrown when invalid parameters are provided
- `UnsupportedOperationException`: 不支持的操作时抛出 / Thrown when unsupported operations are called
- `ArithmeticException`: 数学计算错误时抛出 / Thrown when mathematical computation errors occur
- `IndexOutOfBoundsException`: 数组越界时抛出 / Thrown when array index is out of bounds

### 性能建议 / Performance Recommendations

Guidelines for optimal performance when using the library:

使用库时的性能优化建议：

1. **内存管理**: 避免创建过多临时对象 / **Memory Management**: Avoid creating too many temporary objects
2. **批量操作**: 优先使用向量化操作 / **Batch Operations**: Prefer vectorized operations over loops
3. **数据类型**: 根据精度需求选择float或double / **Data Types**: Choose float or double based on precision requirements
4. **并行化**: 大规模计算时考虑并行处理 / **Parallelization**: Consider parallel processing for large-scale computations

### 最佳实践 / Best Practices

Recommended practices for effective use of the library:

有效使用库的推荐实践：

1. **数据预处理**: 训练前对数据进行标准化 / **Data Preprocessing**: Standardize data before training
2. **参数调优**: 根据具体问题调整算法参数 / **Parameter Tuning**: Adjust algorithm parameters based on specific problems
3. **结果验证**: 使用交叉验证评估模型性能 / **Result Validation**: Use cross-validation to evaluate model performance
4. **错误处理**: 检查输入数据的有效性 / **Error Handling**: Validate input data for correctness

## 快速开始指南 / Quick Start Guide

### 基本使用示例 / Basic Usage Examples

#### 1. 线性代数 / Linear Algebra
```java
// 创建矩阵和向量 / Create matrices and vectors
IMatrix<Double> matrix = Linalg.matrix(new double[][]{{1, 2}, {3, 4}});
IVector<Double> vector = Linalg.vector(new double[]{1, 2, 3});

// 矩阵运算 / Matrix operations
IMatrix<Double> result = matrix.add(matrix).multiplyScalar(2.0);
Double norm = vector.norm2();
```

#### 2. 统计分析 / Statistical Analysis
```java
// 创建概率分布 / Create probability distributions
NormalDistribution normal = Stats.norm(0, 1);  // 标准正态分布
StudentDistribution tDist = Stats.t(10);       // t分布
PoissonDistribution poisson = Stats.poisson(2.5); // 泊松分布

// 分布操作 / Distribution operations
double prob = normal.pdf(0);           // 概率密度
double cdf = normal.cdf(1.96);         // 累积分布
double sample = normal.sample();       // 随机采样
```

#### 3. 机器学习 / Machine Learning
```java
// 线性回归 / Linear regression
RereLinearRegression lr = new RereLinearRegression();
RegressionResult result = lr.fit(X, y);
IVector<Double> predictions = lr.predict(X_test);

// 逻辑回归 / Logistic regression
RereLogisticRegression logReg = new RereLogisticRegression();
ClassificationResult result = logReg.fit(X, labels);
String prediction = logReg.predict(newSample);
```

#### 4. 数据可视化 / Data Visualization
```java
// 创建图表 / Create plots
RerePlot plot = Plots.of(800, 600);
plot.line(x, y).title("Line Chart").show();

// 多种图表类型 / Multiple chart types
plot.scatter(x, y).title("Scatter Plot");
plot.bar(categories, values).title("Bar Chart");
plot.pie(data).title("Pie Chart");
```

#### 5. 降维算法 / Dimensionality Reduction
```java
// 主成分分析 / Principal Component Analysis
RerePCA pca = new RerePCA();
IMatrix<Double> reduced = pca.fitTransform(data, 2); // 降到2维

// t-SNE降维 / t-SNE dimensionality reduction
RereTSNE tsne = new RereTSNE();
IMatrix<Double> embedded = tsne.fitTransform(data);
```

### 最佳实践 / Best Practices

1. **使用工厂类**: 优先使用 `Linalg`、`Stats`、`Plots` 工厂类创建对象
2. **接口编程**: 使用 `IMatrix`、`IVector`、`IPlot` 接口进行编程
3. **错误处理**: 检查输入数据的有效性，处理可能的异常
4. **性能优化**: 对于大规模数据，框架会自动选择GPU加速
5. **内存管理**: 及时释放不需要的大型矩阵和向量对象

## 时间序列分析 / Time Series Analysis

### TimeSeriesData 类 / TimeSeriesData Class

时间序列数据容器类，支持单变量和多变量时间序列，提供基本的数据操作和访问方法。

Time series data container class supporting univariate and multivariate time series with basic data operations and access methods.

```java
public class TimeSeriesData {
    // ========== 构造函数 / Constructors ==========
    
    // 从时间戳和向量数据创建 / Create from timestamps and vector data
    public TimeSeriesData(List<LocalDateTime> timestamps, IVector<Double> values, String columnName);
    
    // 从时间戳和矩阵数据创建 / Create from timestamps and matrix data
    public TimeSeriesData(List<LocalDateTime> timestamps, IMatrix<Double> data, String[] columnNames);
    
    // 从向量和采样率创建 / Create from vector and sampling rate
    public TimeSeriesData(IVector<Double> values, double samplingRate, String columnName, LocalDateTime startTime);
    
    // ========== 数据访问 / Data Access ==========
    
    // 获取时间戳列表 / Get timestamps list
    List<LocalDateTime> getTimestamps();
    
    // 获取数据矩阵 / Get data matrix
    IMatrix<Double> getData();
    
    // 获取列名数组 / Get column names array
    String[] getColumnNames();
    
    // 获取采样率 / Get sampling rate
    double getSamplingRate();
    
    // 获取时间序列长度 / Get time series length
    int getLength();
    
    // 获取变量数量 / Get number of variables
    int getNumVariables();
    
    // 是否为单变量时间序列 / Check if univariate time series
    boolean isUnivariate();
    
    // 是否为多变量时间序列 / Check if multivariate time series
    boolean isMultivariate();
    
    // 获取指定变量的数据 / Get data for specified variable
    IVector<Double> getVariable(int variableIndex);
    IVector<Double> getVariable(String columnName);
    
    // 获取变量索引 / Get variable index
    int getVariableIndex(String columnName);
    
    // ========== 数据操作 / Data Operations ==========
    
    // 切片时间序列 / Slice time series
    TimeSeriesData slice(int startIndex, int endIndex);
    TimeSeriesData slice(LocalDateTime startTime, LocalDateTime endTime);
    
    // 重采样时间序列 / Resample time series
    TimeSeriesData resample(double newSamplingRate);
    
    // 添加噪声 / Add noise
    TimeSeriesData addNoise(double noiseLevel);
    
    // 转换为单变量时间序列 / Convert to univariate time series
    TimeSeriesData toUnivariate(int variableIndex);
    TimeSeriesData toUnivariate(String columnName);
    
    // 计算基本统计信息 / Calculate basic statistics
    IMatrix<Double> getStatistics();
}
```

### TimeSeriesAnalyzer 类 / TimeSeriesAnalyzer Class

统一时间序列分析工具类，提供完整的时间序列分析功能，包括数据预处理、模型选择、预测、诊断等。

Unified time series analysis tool class providing comprehensive time series analysis functionality including data preprocessing, model selection, forecasting, and diagnostics.

```java
public class TimeSeriesAnalyzer {
    // ========== 构造函数 / Constructors ==========
    
    // 从向量数据创建 / Create from vector data
    public TimeSeriesAnalyzer(IVector<Double> data, String name);
    
    // 从向量数据和时间戳创建 / Create from vector data and timestamps
    public TimeSeriesAnalyzer(IVector<Double> data, String name, LocalDateTime[] timestamps);
    
    // ========== 分析配置 / Analysis Configuration ==========
    
    // 分析配置类 / Analysis configuration class
    public static class AnalysisConfig {
        public final ITimeSeriesModel.ModelType[] candidateModels;
        public final TimeSeriesModelFactory.SelectionCriterion selectionCriterion;
        public final int forecastSteps;
        public final double confidenceLevel;
        public final boolean performDiagnostics;
        public final boolean autoPreprocess;
        public final Map<String, Object> modelParameters;
        
        // 配置构建器 / Configuration builder
        public static class Builder {
            public Builder setCandidateModels(ITimeSeriesModel.ModelType[] candidateModels);
            public Builder setSelectionCriterion(TimeSeriesModelFactory.SelectionCriterion selectionCriterion);
            public Builder setForecastSteps(int forecastSteps);
            public Builder setConfidenceLevel(double confidenceLevel);
            public Builder setPerformDiagnostics(boolean performDiagnostics);
            public Builder setAutoPreprocess(boolean autoPreprocess);
            public Builder addModelParameter(String key, Object value);
            public AnalysisConfig build();
        }
    }
    
    // ========== 分析结果 / Analysis Results ==========
    
    // 分析结果类 / Analysis result class
    public static class AnalysisResult {
        public final ITimeSeriesModel bestModel;
        public final ITimeSeriesForecastResult forecast;
        public final ITimeSeriesDiagnostics diagnostics;
        public final String summary;
        public final Map<String, Object> metadata;
    }
    
    // ========== 主要方法 / Main Methods ==========
    
    // 执行完整分析 / Perform complete analysis
    AnalysisResult analyze(AnalysisConfig config);
    
    // 快速分析 / Quick analysis
    AnalysisResult quickAnalyze();
    
    // 预测 / Forecast
    ITimeSeriesForecastResult forecast(int steps, double confidenceLevel);
    
    // 诊断 / Diagnose
    ITimeSeriesDiagnostics diagnose();
    
    // 设置模型 / Set model
    void setModel(ITimeSeriesModel model);
    
    // ========== 特征分析 / Feature Analysis ==========
    
    // 获取数据统计信息 / Get data statistics
    Map<String, Object> getDataStatistics();
    
    // 获取趋势分析 / Get trend analysis
    Map<String, Object> getTrendAnalysis();
    
    // 获取季节性分析 / Get seasonal analysis
    Map<String, Object> getSeasonalAnalysis(int period);
    
    // ========== 报告生成 / Report Generation ==========
    
    // 导出分析报告 / Export analysis report
    String exportReport(String format);
    
    // ========== Getter方法 / Getter Methods ==========
    
    IVector<Double> getData();
    String getName();
    LocalDateTime[] getTimestamps();
    ITimeSeriesModel getCurrentModel();
    ITimeSeriesForecastResult getLastForecast();
    ITimeSeriesDiagnostics getLastDiagnostics();
}
```

### TimeSeriesForecasting 类 / TimeSeriesForecasting Class

时间序列预测类，提供多种预测方法，包括ARIMA模型、指数平滑、线性回归预测等。

Time series forecasting class providing various forecasting methods including ARIMA models, exponential smoothing, and linear regression forecasting.

```java
public class TimeSeriesForecasting {
    // ========== 预测结果类 / Forecast Result Class ==========
    
    public static class ForecastResult {
        public final IVector<Double> forecast;
        public final IVector<Double> lowerBound;
        public final IVector<Double> upperBound;
        public final double mse;
        public final double mae;
        public final double mape;
        public final String modelType;
        public final double confidenceLevel;
    }
    
    // ========== 简单预测方法 / Simple Forecasting Methods ==========
    
    // 简单移动平均预测 / Simple moving average forecasting
    static ForecastResult simpleMovingAverage(TimeSeriesData timeSeries, int variableIndex, 
                                           int windowSize, int forecastSteps, double confidenceLevel);
    static ForecastResult simpleMovingAverage(TimeSeriesData timeSeries, String columnName, 
                                           int windowSize, int forecastSteps, double confidenceLevel);
    
    // 指数平滑预测 / Exponential smoothing forecasting
    static ForecastResult exponentialSmoothing(TimeSeriesData timeSeries, int variableIndex, 
                                            double alpha, int forecastSteps, double confidenceLevel);
    static ForecastResult exponentialSmoothing(TimeSeriesData timeSeries, String columnName, 
                                            double alpha, int forecastSteps, double confidenceLevel);
    
    // 线性回归预测 / Linear regression forecasting
    static ForecastResult linearRegression(TimeSeriesData timeSeries, int variableIndex, 
                                        int forecastSteps, double confidenceLevel);
    static ForecastResult linearRegression(TimeSeriesData timeSeries, String columnName, 
                                        int forecastSteps, double confidenceLevel);
    
    // ========== 高级预测方法 / Advanced Forecasting Methods ==========
    
    // ARIMA模型预测 / ARIMA model forecasting
    static ForecastResult arimaForecast(TimeSeriesData timeSeries, int variableIndex, 
                                     int p, int d, int q, int forecastSteps, double confidenceLevel);
    static ForecastResult arimaForecast(TimeSeriesData timeSeries, String columnName, 
                                     int p, int d, int q, int forecastSteps, double confidenceLevel);
    
    // 季节性预测 / Seasonal forecasting
    static ForecastResult seasonalForecast(TimeSeriesData timeSeries, int variableIndex, 
                                        int period, int forecastSteps, double confidenceLevel);
    static ForecastResult seasonalForecast(TimeSeriesData timeSeries, String columnName, 
                                        int period, int forecastSteps, double confidenceLevel);
    
    // Holt-Winters预测 / Holt-Winters forecasting
    static ForecastResult holtWintersForecast(TimeSeriesData timeSeries, int variableIndex,
                                           double alpha, double beta, double gamma, int period,
                                           int forecastSteps, double confidenceLevel);
    
    // GARCH预测 / GARCH forecasting
    static ForecastResult garchForecast(TimeSeriesData timeSeries, int variableIndex,
                                     int p, int q, int forecastSteps, double confidenceLevel);
    
    // 状态空间模型预测 / State space model forecasting
    static ForecastResult stateSpaceForecast(TimeSeriesData timeSeries, int variableIndex,
                                          double sigmaEta, double sigmaZeta, double sigmaEpsilon,
                                          int forecastSteps, double confidenceLevel);
    
    // 自动模型选择预测 / Automatic model selection forecasting
    static ForecastResult autoForecast(TimeSeriesData timeSeries, int variableIndex,
                                    int forecastSteps, double confidenceLevel);
}
```

### TimeSeriesFiltering 类 / TimeSeriesFiltering Class

时间序列滤波类，提供多种滤波方法，包括移动平均、指数平滑、卡尔曼滤波、小波滤波等。

Time series filtering class providing various filtering methods including moving average, exponential smoothing, Kalman filtering, and wavelet filtering.

```java
public class TimeSeriesFiltering {
    // ========== 滤波结果类 / Filter Result Class ==========
    
    public static class FilterResult {
        public final TimeSeriesData filtered;
        public final TimeSeriesData noise;
        public final double snr;
        public final String filterType;
    }
    
    // ========== 基础滤波方法 / Basic Filtering Methods ==========
    
    // 移动平均滤波 / Moving average filtering
    static FilterResult movingAverage(TimeSeriesData timeSeries, int variableIndex, int windowSize);
    static FilterResult movingAverage(TimeSeriesData timeSeries, String columnName, int windowSize);
    
    // 指数平滑滤波 / Exponential smoothing filtering
    static FilterResult exponentialSmoothing(TimeSeriesData timeSeries, int variableIndex, double alpha);
    static FilterResult exponentialSmoothing(TimeSeriesData timeSeries, String columnName, double alpha);
    
    // 高斯滤波 / Gaussian filtering
    static FilterResult gaussianFilter(TimeSeriesData timeSeries, int variableIndex, double sigma);
    static FilterResult gaussianFilter(TimeSeriesData timeSeries, String columnName, double sigma);
    
    // 中值滤波 / Median filtering
    static FilterResult medianFilter(TimeSeriesData timeSeries, int variableIndex, int windowSize);
    static FilterResult medianFilter(TimeSeriesData timeSeries, String columnName, int windowSize);
    
    // ========== 频域滤波方法 / Frequency Domain Filtering Methods ==========
    
    // 低通滤波 / Low pass filtering
    static FilterResult lowPassFilter(TimeSeriesData timeSeries, int variableIndex, 
                                    double cutoffFreq, int order);
    static FilterResult lowPassFilter(TimeSeriesData timeSeries, String columnName, 
                                    double cutoffFreq, int order);
    
    // 高通滤波 / High pass filtering
    static FilterResult highPassFilter(TimeSeriesData timeSeries, int variableIndex, 
                                     double cutoffFreq, int order);
    static FilterResult highPassFilter(TimeSeriesData timeSeries, String columnName, 
                                     double cutoffFreq, int order);
    
    // 带通滤波 / Band pass filtering
    static FilterResult bandPassFilter(TimeSeriesData timeSeries, int variableIndex, 
                                     double lowFreq, double highFreq, int order);
    static FilterResult bandPassFilter(TimeSeriesData timeSeries, String columnName, 
                                     double lowFreq, double highFreq, int order);
    
    // 自适应滤波 / Adaptive filtering
    static FilterResult adaptiveFilter(TimeSeriesData timeSeries, int variableIndex, double learningRate);
    static FilterResult adaptiveFilter(TimeSeriesData timeSeries, String columnName, double learningRate);
}
```

### TimeSeriesDecomposition 类 / TimeSeriesDecomposition Class

时间序列分解类，提供时间序列分解功能，包括趋势、季节性、周期性成分的分离。

Time series decomposition class providing time series decomposition functionality including trend, seasonal, and cyclical component separation.

```java
public class TimeSeriesDecomposition {
    // ========== 分解模型类型枚举 / Decomposition Model Type Enum ==========
    
    public enum DecompositionModel {
        ADDITIVE,      // 加法模型 / Additive model
        MULTIPLICATIVE // 乘法模型 / Multiplicative model
    }
    
    // ========== 分解结果类 / Decomposition Result Class ==========
    
    public static class DecompositionResult {
        public final IVector<Double> trend;
        public final IVector<Double> seasonal;
        public final IVector<Double> residual;
        public final IVector<Double> original;
        public final DecompositionModel model;
        public final int period;
        public final double trendStrength;
        public final double seasonalStrength;
        public final double residualStrength;
    }
    
    // ========== 分解方法 / Decomposition Methods ==========
    
    // 经典分解 / Classical decomposition
    static DecompositionResult classicalDecomposition(TimeSeriesData timeSeries, int variableIndex, 
                                                    int period, DecompositionModel model);
    static DecompositionResult classicalDecomposition(TimeSeriesData timeSeries, String columnName, 
                                                    int period, DecompositionModel model);
    
    // X-13ARIMA-SEATS分解 / X-13ARIMA-SEATS decomposition
    static DecompositionResult x13Decomposition(TimeSeriesData timeSeries, int variableIndex, int period);
    static DecompositionResult x13Decomposition(TimeSeriesData timeSeries, String columnName, int period);
    
    // STL分解 / STL decomposition
    static DecompositionResult stlDecomposition(TimeSeriesData timeSeries, int variableIndex, 
                                              int period, int seasonalWindow, int trendWindow);
    static DecompositionResult stlDecomposition(TimeSeriesData timeSeries, String columnName, 
                                              int period, int seasonalWindow, int trendWindow);
    
    // 小波分解 / Wavelet decomposition
    static DecompositionResult waveletDecomposition(TimeSeriesData timeSeries, int variableIndex, 
                                                  String wavelet, int levels);
    static DecompositionResult waveletDecomposition(TimeSeriesData timeSeries, String columnName, 
                                                  String wavelet, int levels);
}
```

---

**API参考手册** - 完整的类和方法参考，助您快速开发！

**API Reference Manual** - Complete class and method reference to help you develop quickly!
