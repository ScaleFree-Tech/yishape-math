# API 参考手册 (API Reference)

## 概述 / Overview

本文档提供了 `yishape-math` 包中所有公共类和方法的详细API参考。

## 核心接口 / Core Interfaces

### IVector 接口

```java
public interface IVector {
    // 工厂方法 / Factory methods
    static IVector of(float[] array);
    static IVector of(double[] array);
    static IVector of(int[] array);
    static IVector zeros(int length);
    static IVector ones(int length);
    static IVector rand(int length);
    static IVector randn(int length);
    
    // 基本属性 / Basic properties
    int length();
    float[] toArray();
    double[] toDoubleArray();
    int[] toIntArray();
    
    // 元素访问 / Element access
    float get(int index);
    void set(int index, float value);
    IVector slice(int start, int end);
    IVector slice(int[] indices);
    
    // 基本数学运算 / Basic mathematical operations
    IVector add(IVector other);
    IVector add(float scalar);
    IVector subtract(IVector other);
    IVector subtract(float scalar);
    IVector multiply(IVector other);
    IVector multiply(float scalar);
    IVector divide(IVector other);
    IVector divide(float scalar);
    
    // 线性代数运算 / Linear algebra operations
    float dot(IVector other);
    float norm1();
    float norm2();
    float normInf();
    IVector normalize();
    float cosine(IVector other);
    float euclideanDistance(IVector other);
    float manhattanDistance(IVector other);
    
    // 统计函数 / Statistical functions
    float sum();
    float mean();
    float variance();
    float std();
    float min();
    float max();
    int argmin();
    int argmax();
    
    // 通用函数 / Universal functions
    IVector abs();
    IVector sqrt();
    IVector exp();
    IVector log();
    IVector sin();
    IVector cos();
    IVector tan();
    IVector pow(float exponent);
    
    // 比较操作 / Comparison operations
    IVector equals(IVector other);
    IVector greater(IVector other);
    IVector less(IVector other);
    IVector greaterEqual(IVector other);
    IVector lessEqual(IVector other);
    
    // 排序和查找 / Sorting and searching
    IVector sort();
    IVector sortDescending();
    int[] argsort();
    int[] argsortDescending();
    boolean contains(float value);
    int indexOf(float value);
}
```

### IMatrix 接口

```java
public interface IMatrix {
    // 工厂方法 / Factory methods
    static IMatrix of(float[][] array);
    static IMatrix of(double[][] array);
    static IMatrix of(int[][] array);
    static IMatrix zeros(int rows, int cols);
    static IMatrix ones(int rows, int cols);
    static IMatrix eye(int size);
    static IMatrix rand(int rows, int cols);
    static IMatrix randn(int rows, int cols);
    static IMatrix diag(IVector diagonal);
    
    // 基本属性 / Basic properties
    int getRowNum();
    int getColNum();
    float[][] toArray();
    double[][] toDoubleArray();
    int[][] toIntArray();
    
    // 元素访问 / Element access
    float get(int row, int col);
    void set(int row, int col, float value);
    IVector getRow(int row);
    IVector getColumn(int col);
    void setRow(int row, IVector vector);
    void setColumn(int col, IVector vector);
    IMatrix slice(int startRow, int endRow, int startCol, int endCol);
    
    // 基本数学运算 / Basic mathematical operations
    IMatrix add(IMatrix other);
    IMatrix add(float scalar);
    IMatrix subtract(IMatrix other);
    IMatrix subtract(float scalar);
    IMatrix multiply(IMatrix other);
    IMatrix multiply(float scalar);
    IMatrix divide(IMatrix other);
    IMatrix divide(float scalar);
    
    // 矩阵运算 / Matrix operations
    IMatrix mmul(IMatrix other);
    IVector mmul(IVector vector);
    IMatrix transpose();
    IMatrix inverse();
    float determinant();
    float trace();
    int rank();
    
    // 线性代数 / Linear algebra
    Tuple2<IVector, IMatrix> eigenDecomposition();
    Tuple3<IMatrix, IVector, IMatrix> svd();
    IMatrix pseudoInverse();
    float condition();
    float norm1();
    float norm2();
    float normInf();
    float normFrobenius();
    
    // 统计函数 / Statistical functions
    float sum();
    float mean();
    float variance();
    float std();
    IVector rowSum();
    IVector rowMean();
    IVector rowVariance();
    IVector rowStd();
    IVector columnSum();
    IVector columnMean();
    IVector columnVariance();
    IVector columnStd();
    
    // 数据变换 / Data transformation
    IMatrix center();
    IMatrix standardize();
    IMatrix normalize();
    IMatrix covariance();
    
    // 通用函数 / Universal functions
    IMatrix abs();
    IMatrix sqrt();
    IMatrix exp();
    IMatrix log();
    IMatrix sin();
    IMatrix cos();
    IMatrix tan();
    IMatrix pow(float exponent);
}
```

## 优化算法 / Optimization Algorithms

### IOptimizer 接口

```java
public interface IOptimizer {
    // 优化方法 / Optimization methods
    Tuple2<Float, IVector> optimize(IVector initialPoint, 
                                   IObjectiveFunction objectiveFunction, 
                                   IGradientFunction gradientFunction);
    
    // 参数设置 / Parameter settings
    void setMaxIterations(int maxIterations);
    void setTolerance(float tolerance);
    void setVerbose(boolean verbose);
    
    // 状态查询 / Status queries
    int getIterationCount();
    float getCurrentTolerance();
    boolean hasConverged();
}
```

### RereLBFGS 类

```java
public class RereLBFGS implements IOptimizer {
    // 构造函数 / Constructors
    public RereLBFGS();
    public RereLBFGS(int memorySize);
    
    // L-BFGS特有方法 / L-BFGS specific methods
    void setMemorySize(int memorySize);
    void setLineSearchType(LineSearchType type);
    void setC1(float c1);
    void setC2(float c2);
    void setMaxLineSearchIterations(int maxIterations);
    
    // 状态查询 / Status queries
    int getMemorySize();
    LineSearchType getLineSearchType();
    float getC1();
    float getC2();
    
    // 优化历史 / Optimization history
    List<Float> getObjectiveHistory();
    List<Float> getGradientNormHistory();
    List<IVector> getPointHistory();
}
```

### 目标函数接口 / Objective Function Interfaces

```java
@FunctionalInterface
public interface IObjectiveFunction {
    float compute(IVector x);
}

@FunctionalInterface  
public interface IGradientFunction {
    IVector compute(IVector x);
}
```

## 降维算法 / Dimensionality Reduction Algorithms

### RerePCA 类

```java
public class RerePCA {
    // 构造函数 / Constructors
    public RerePCA();
    public RerePCA(int components);
    
    // 参数设置 / Parameter settings
    void setComponents(int components);
    void setWhiten(boolean whiten);
    void setRandomState(long seed);
    
    // 训练和变换 / Training and transformation
    void fit(IMatrix data);
    IMatrix transform(IMatrix data);
    IMatrix fitTransform(IMatrix data);
    IMatrix inverseTransform(IMatrix transformedData);
    
    // 结果查询 / Result queries
    IVector getExplainedVariance();
    IVector getExplainedVarianceRatio();
    float getExplainedVarianceSum();
    IMatrix getComponents();
    IVector getMean();
    int getOptimalComponents(float varianceThreshold);
}
```

### RereTSNE 类

```java
public class RereTSNE {
    // 构造函数 / Constructors
    public RereTSNE();
    public RereTSNE(int targetDimensions);
    
    // 参数设置 / Parameter settings
    void setTargetDimensions(int dimensions);
    void setPerplexity(float perplexity);
    void setLearningRate(float learningRate);
    void setMaxIterations(int maxIterations);
    void setEarlyExaggeration(float earlyExaggeration);
    void setRandomState(long seed);
    void setVerbose(boolean verbose);
    
    // 降维方法 / Dimensionality reduction methods
    IMatrix dimensionReduction(IMatrix data);
    
    // 状态查询 / Status queries
    int getIterationCount();
    float getCurrentCost();
    List<Float> getCostHistory();
}
```

### RereUMAP 类

```java
public class RereUMAP {
    // 构造函数 / Constructors
    public RereUMAP();
    public RereUMAP(int targetDimensions);
    
    // 参数设置 / Parameter settings
    void setTargetDimensions(int dimensions);
    void setNeighbors(int neighbors);
    void setMinDist(float minDist);
    void setSpread(float spread);
    void setLearningRate(float learningRate);
    void setMaxIterations(int maxIterations);
    void setRandomState(long seed);
    void setVerbose(boolean verbose);
    
    // 降维方法 / Dimensionality reduction methods
    IMatrix dimensionReduction(IMatrix data);
    IMatrix transform(IMatrix newData);
    
    // 状态查询 / Status queries
    int getIterationCount();
    float getCurrentCost();
    IMatrix getEmbedding();
}
```

## 机器学习算法 / Machine Learning Algorithms

### IRegression 接口

```java
public interface IRegression {
    // 训练方法 / Training methods
    RegressionResult fit(IMatrix features, IVector labels);
    
    // 预测方法 / Prediction methods
    float predict(IVector features);
    IVector predict(IMatrix features);
    
    // 评估方法 / Evaluation methods
    float score(IMatrix features, IVector labels);
    RegressionMetrics evaluate(IMatrix features, IVector labels);
}
```

### RereLinearRegression 类

```java
public class RereLinearRegression implements IRegression {
    // 构造函数 / Constructors
    public RereLinearRegression();
    public RereLinearRegression(RegularizationType regularization);
    
    // 参数设置 / Parameter settings
    void setRegularizationType(RegularizationType type);
    void setLambda1(float lambda1);
    void setLambda2(float lambda2);
    void setFitIntercept(boolean fitIntercept);
    void setNormalize(boolean normalize);
    void setMaxIterations(int maxIterations);
    void setTolerance(float tolerance);
    
    // 优化器设置 / Optimizer settings
    void setOptimizer(IOptimizer optimizer);
    
    // 模型参数 / Model parameters
    IVector getWeights();
    float getIntercept();
    RegressionResult getLastResult();
    
    // 特征重要性 / Feature importance
    IVector getFeatureImportance();
    IVector getFeatureImportanceRank();
}
```

### RegressionResult 类

```java
public class RegressionResult {
    // 构造函数 / Constructors
    public RegressionResult(IVector weights, float intercept, float loss);
    
    // 结果访问 / Result access
    IVector getWeights();
    float getIntercept();
    float getLoss();
    float getR2Score();
    float getMSE();
    float getRMSE();
    float getMAE();
    
    // 统计信息 / Statistical information
    float getExplainedVariance();
    float getUnexplainedVariance();
    float getTotalVariance();
    int getDegreesOfFreedom();
    
    // 置信区间 / Confidence intervals
    Tuple2<Float, Float> getWeightConfidenceInterval(int index, float alpha);
    Tuple2<Float, Float> getInterceptConfidenceInterval(float alpha);
}
```

## 数学工具类 / Math Utilities

### RereMathUtil 类

```java
public final class RereMathUtil {
    // 类型转换 / Type conversion
    static float[] doubleToFloat(double[] array);
    static double[] floatToDouble(float[] array);
    static float[] intToFloat(int[] array);
    static int[] floatToInt(float[] array);
    static double[] vectorToDoubleArray(IVector vector);
    static int[] vectorToIntArray(IVector vector);
    static double[][] matrixToDoubleArray(IMatrix matrix);
    static int[][] matrixToIntArray(IMatrix matrix);
    
    // 随机数生成 / Random number generation
    static void setSeed(long seed);
    static float rand();
    static float rand(float min, float max);
    static IVector rand(int length);
    static IVector rand(int length, float min, float max);
    static IMatrix rand(int rows, int cols);
    static IMatrix rand(int rows, int cols, float min, float max);
    static float randn();
    static IVector randn(int length);
    static IMatrix randn(int rows, int cols);
    
    // 基本数学函数 / Basic mathematical functions
    static float sin(float x);
    static float cos(float x);
    static float tan(float x);
    static float asin(float x);
    static float acos(float x);
    static float atan(float x);
    static float atan2(float y, float x);
    static float exp(float x);
    static float log(float x);
    static float log10(float x);
    static float pow(float base, float exponent);
    static float sqrt(float x);
    static float abs(float x);
    static float floor(float x);
    static float ceil(float x);
    static float round(float x);
    static float min(float a, float b);
    static float max(float a, float b);
    
    // 统计函数 / Statistical functions
    static float sum(float[] array);
    static float mean(float[] array);
    static float variance(float[] array);
    static float std(float[] array);
    static float min(float[] array);
    static float max(float[] array);
    static int argmin(float[] array);
    static int argmax(float[] array);
    static float median(float[] array);
    static float quantile(float[] array, float q);
    static float skewness(float[] array);
    static float kurtosis(float[] array);
    static float correlation(float[] x, float[] y);
    static float covariance(float[] x, float[] y);
    
    // 数组操作 / Array operations
    static float[] sort(float[] array, boolean ascending);
    static int[] argsort(float[] array, boolean ascending);
    static int indexOf(float[] array, float value);
    static boolean contains(float[] array, float value);
    static float[] standardize(float[] array);
    static float[] normalize(float[] array);
    static float[] linspace(float start, float stop, int num);
    static float[] logspace(float start, float stop, int num);
    static Tuple2<float[][], float[][]> meshgrid(float[] x, float[] y);
    static float[] sample(float[] array, int sampleSize);
    static float[] stratifiedSample(float[] array, int strata, int sampleSize);
    
    // 常用常数 / Common constants
    static final float PI = (float) Math.PI;
    static final float E = (float) Math.E;
    static final float EPS = 1e-8f;
    static final float INF = Float.POSITIVE_INFINITY;
    static final float NEG_INF = Float.NEGATIVE_INFINITY;
    static final float NAN = Float.NaN;
}
```

## 枚举类型 / Enum Types

### RegularizationType

```java
public enum RegularizationType {
    NONE,        // 无正则化 / No regularization
    L1,          // L1正则化 (Lasso) / L1 regularization (Lasso)
    L2,          // L2正则化 (Ridge) / L2 regularization (Ridge)
    ELASTIC_NET  // 弹性网络 / Elastic Net
}
```

### LineSearchType

```java
public enum LineSearchType {
    ARMIJO,      // Armijo线搜索 / Armijo line search
    WOLFE,       // Wolfe线搜索 / Wolfe line search
    STRONG_WOLFE // 强Wolfe线搜索 / Strong Wolfe line search
}
```

## 数据类型 / Data Types

### Tuple2

```java
public class Tuple2<T1, T2> {
    public final T1 _1;
    public final T2 _2;
    
    public Tuple2(T1 first, T2 second);
    
    // 访问方法 / Access methods
    public T1 getFirst();
    public T2 getSecond();
    
    // 实用方法 / Utility methods
    public String toString();
    public boolean equals(Object obj);
    public int hashCode();
}
```

### Tuple3

```java
public class Tuple3<T1, T2, T3> {
    public final T1 _1;
    public final T2 _2;
    public final T3 _3;
    
    public Tuple3(T1 first, T2 second, T3 third);
    
    // 访问方法 / Access methods
    public T1 getFirst();
    public T2 getSecond();
    public T3 getThird();
    
    // 实用方法 / Utility methods
    public String toString();
    public boolean equals(Object obj);
    public int hashCode();
}
```

## 使用指南 / Usage Guidelines

### 异常处理 / Exception Handling

- `IllegalArgumentException`: 参数无效时抛出
- `UnsupportedOperationException`: 不支持的操作时抛出
- `ArithmeticException`: 数学计算错误时抛出
- `IndexOutOfBoundsException`: 数组越界时抛出

### 性能建议 / Performance Recommendations

1. **内存管理**: 避免创建过多临时对象
2. **批量操作**: 优先使用向量化操作
3. **数据类型**: 根据精度需求选择float或double
4. **并行化**: 大规模计算时考虑并行处理

### 最佳实践 / Best Practices

1. **数据预处理**: 训练前对数据进行标准化
2. **参数调优**: 根据具体问题调整算法参数
3. **结果验证**: 使用交叉验证评估模型性能
4. **错误处理**: 检查输入数据的有效性

---

**API参考手册** - 完整的类和方法参考，助您快速开发！
