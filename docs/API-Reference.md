# API 参考手册 (API Reference)

## 概述 / Overview

本文档提供了 `yishape-math` 包中所有公共类和方法的详细API参考。

This document provides detailed API reference for all public classes and methods in the `yishape-math` package.

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
    void set(int index, float value);        // Set element at index
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
    void set(int row, int col, float value); // Set element at (row, col)
    IVector getRow(int row);                 // Get row vector
    IVector getColumn(int col);              // Get column vector
    void setRow(int row, IVector vector);    // Set row vector
    void setColumn(int col, IVector vector); // Set column vector
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

### RereLinearRegression 类 / RereLinearRegression Class

Implementation of linear regression with support for various regularization techniques.

支持多种正则化技术的线性回归实现。

```java
public class RereLinearRegression implements IRegression {
    // 构造函数 / Constructors
    public RereLinearRegression();                    // Create linear regression with default settings
    public RereLinearRegression(RegularizationType regularization); // Create with regularization type
    
    // 参数设置 / Parameter settings
    void setRegularizationType(RegularizationType type); // Set regularization type (L1, L2, ElasticNet)
    void setLambda1(float lambda1);                  // Set L1 regularization strength
    void setLambda2(float lambda2);                  // Set L2 regularization strength
    void setFitIntercept(boolean fitIntercept);      // Set whether to fit intercept
    void setNormalize(boolean normalize);            // Set whether to normalize features
    void setMaxIterations(int maxIterations);        // Set maximum iterations
    void setTolerance(float tolerance);              // Set convergence tolerance
    
    // 优化器设置 / Optimizer settings
    void setOptimizer(IOptimizer optimizer);         // Set optimization algorithm
    
    // 模型参数 / Model parameters
    IVector getWeights();                            // Get regression coefficients
    float getIntercept();                            // Get intercept term
    RegressionResult getLastResult();                // Get last training result
    
    // 特征重要性 / Feature importance
    IVector getFeatureImportance();                  // Get feature importance scores
    IVector getFeatureImportanceRank();              // Get feature importance ranking
}
```

### RegressionResult 类 / RegressionResult Class

Container class for regression analysis results, including model parameters and evaluation metrics.

回归分析结果的容器类，包含模型参数和评估指标。

```java
public class RegressionResult {
    // 构造函数 / Constructors
    public RegressionResult(IVector weights, float intercept, float loss); // Create result with parameters
    
    // 结果访问 / Result access
    IVector getWeights();                            // Get regression coefficients
    float getIntercept();                            // Get intercept term
    float getLoss();                                 // Get training loss
    float getR2Score();                             // Get R² score
    float getMSE();                                  // Get mean squared error
    float getRMSE();                                 // Get root mean squared error
    float getMAE();                                  // Get mean absolute error
    
    // 统计信息 / Statistical information
    float getExplainedVariance();                    // Get explained variance
    float getUnexplainedVariance();                  // Get unexplained variance
    float getTotalVariance();                        // Get total variance
    int getDegreesOfFreedom();                       // Get degrees of freedom
    
    // 置信区间 / Confidence intervals
    Tuple2<Float, Float> getWeightConfidenceInterval(int index, float alpha); // Get weight confidence interval
    Tuple2<Float, Float> getInterceptConfidenceInterval(float alpha);         // Get intercept confidence interval
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

## 数学工具类 / Math Utilities

### RereMathUtil 类 / RereMathUtil Class

Utility class providing mathematical functions, type conversions, and statistical operations.

提供数学函数、类型转换和统计运算的工具类。

```java
public final class RereMathUtil {
    // 类型转换 / Type conversion
    static float[] doubleToFloat(double[] array);     // Convert double array to float array
    static double[] floatToDouble(float[] array);     // Convert float array to double array
    static float[] intToFloat(int[] array);           // Convert int array to float array
    static int[] floatToInt(float[] array);           // Convert float array to int array
    static double[] vectorToDoubleArray(IVector vector); // Convert vector to double array
    static int[] vectorToIntArray(IVector vector);    // Convert vector to int array
    static double[][] matrixToDoubleArray(IMatrix matrix); // Convert matrix to double array
    static int[][] matrixToIntArray(IMatrix matrix);  // Convert matrix to int array
    
    // 随机数生成 / Random number generation
    static void setSeed(long seed);                   // Set random seed
    static float rand();                              // Generate random float [0,1)
    static float rand(float min, float max);          // Generate random float in range
    static IVector rand(int length);                  // Generate random vector
    static IVector rand(int length, float min, float max); // Generate random vector in range
    static IMatrix rand(int rows, int cols);          // Generate random matrix
    static IMatrix rand(int rows, int cols, float min, float max); // Generate random matrix in range
    static float randn();                             // Generate normal random float
    static IVector randn(int length);                 // Generate normal random vector
    static IMatrix randn(int rows, int cols);         // Generate normal random matrix
    
    // 基本数学函数 / Basic mathematical functions
    static float sin(float x);                        // Sine function
    static float cos(float x);                        // Cosine function
    static float tan(float x);                        // Tangent function
    static float asin(float x);                       // Arcsine function
    static float acos(float x);                       // Arccosine function
    static float atan(float x);                       // Arctangent function
    static float atan2(float y, float x);             // Two-argument arctangent
    static float exp(float x);                        // Exponential function
    static float log(float x);                        // Natural logarithm
    static float log10(float x);                      // Base-10 logarithm
    static float pow(float base, float exponent);     // Power function
    static float sqrt(float x);                       // Square root function
    static float abs(float x);                        // Absolute value
    static float floor(float x);                      // Floor function
    static float ceil(float x);                       // Ceiling function
    static float round(float x);                      // Round function
    static float min(float a, float b);               // Minimum of two values
    static float max(float a, float b);               // Maximum of two values
    
    // 统计函数 / Statistical functions
    static float sum(float[] array);                  // Sum of array elements
    static float mean(float[] array);                 // Mean of array elements
    static float variance(float[] array);             // Variance of array elements
    static float std(float[] array);                  // Standard deviation of array elements
    static float min(float[] array);                  // Minimum value in array
    static float max(float[] array);                  // Maximum value in array
    static int argmin(float[] array);                 // Index of minimum value
    static int argmax(float[] array);                 // Index of maximum value
    static float median(float[] array);               // Median of array elements
    static float quantile(float[] array, float q);    // Quantile of array elements
    static float skewness(float[] array);             // Skewness of array elements
    static float kurtosis(float[] array);             // Kurtosis of array elements
    static float correlation(float[] x, float[] y);   // Correlation coefficient
    static float covariance(float[] x, float[] y);    // Covariance
    
    // 数组操作 / Array operations
    static float[] sort(float[] array, boolean ascending); // Sort array
    static int[] argsort(float[] array, boolean ascending); // Get sort indices
    static int indexOf(float[] array, float value);   // Find index of value
    static boolean contains(float[] array, float value); // Check if array contains value
    static float[] standardize(float[] array);        // Standardize array (z-score)
    static float[] normalize(float[] array);          // Normalize array to [0,1]
    static float[] linspace(float start, float stop, int num); // Generate linear space
    static float[] logspace(float start, float stop, int num); // Generate logarithmic space
    static Tuple2<float[][], float[][]> meshgrid(float[] x, float[] y); // Generate meshgrid
    static float[] sample(float[] array, int sampleSize); // Random sampling
    static float[] stratifiedSample(float[] array, int strata, int sampleSize); // Stratified sampling
    
    // 常用常数 / Common constants
    static final float PI = (float) Math.PI;          // Pi constant
    static final float E = (float) Math.E;            // Euler's number
    static final float EPS = 1e-8f;                   // Machine epsilon
    static final float INF = Float.POSITIVE_INFINITY; // Positive infinity
    static final float NEG_INF = Float.NEGATIVE_INFINITY; // Negative infinity
    static final float NAN = Float.NaN;               // Not a number
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

---

**API参考手册** - 完整的类和方法参考，助您快速开发！

**API Reference Manual** - Complete class and method reference to help you develop quickly!
