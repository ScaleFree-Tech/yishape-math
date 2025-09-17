# 数学工具类 (Math Utilities)

## 概述 / Overview

`RereMathUtil` 类提供了各种数学相关的实用方法，包括类型转换、随机数生成、数学函数等功能。这些工具方法为向量和矩阵操作提供了基础支持，确保数据类型的一致性和操作的便利性。

The `RereMathUtil` class provides various mathematical utility methods including type conversion, random number generation, mathematical functions, and more. These utility methods provide foundational support for vector and matrix operations, ensuring data type consistency and operational convenience.

## 核心类 / Core Class

### RereMathUtil 类 / RereMathUtil Class

`RereMathUtil` 是一个静态工具类，包含所有数学相关的实用方法。

`RereMathUtil` is a static utility class containing all mathematical utility methods.

## 主要功能 / Main Features

### 1. 类型转换 / Type Conversion

#### 基本类型转换 / Primitive Type Conversion

```java
// float <-> double 转换 / float <-> double conversion
double[] doubleArray = RereMathUtil.floatToDouble(new float[]{1.0f, 2.0f, 3.0f});
float[] floatArray = RereMathUtil.doubleToFloat(new double[]{1.0, 2.0, 3.0});

// int -> float 转换 / int -> float conversion
float[] floatFromInt = RereMathUtil.intToFloat(new int[]{1, 2, 3, 4});

// float -> int 转换（截断小数部分）/ float -> int conversion (truncate decimal part)
int[] intFromFloat = RereMathUtil.floatToInt(new float[]{1.5f, 2.7f, 3.2f, 4.9f});

// int <-> double 转换 / int <-> double conversion
double[] doubleFromInt = RereMathUtil.intToDouble(new int[]{1, 2, 3, 4});
int[] intFromDouble = RereMathUtil.doubleToInt(new double[]{1.5, 2.7, 3.2, 4.9});
```

#### 包装类转换 / Wrapper Class Conversion

```java
// Float[] -> float[] 转换 / Float[] -> float[] conversion
float[] primitiveFloat = RereMathUtil.toPrimitive(new Float[]{1.0f, 2.0f, 3.0f});

// Double[] -> double[] 转换 / Double[] -> double[] conversion
double[] primitiveDouble = RereMathUtil.toPrimitive(new Double[]{1.0, 2.0, 3.0});

// Integer[] -> int[] 转换 / Integer[] -> int[] conversion
int[] primitiveInt = RereMathUtil.toPrimitive(new Integer[]{1, 2, 3, 4});

// 二维数组转换 / 2D array conversion
float[][] primitiveFloat2D = RereMathUtil.toPrimitive(new Float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}});
double[][] primitiveDouble2D = RereMathUtil.toPrimitive(new Double[][]{{1.0, 2.0}, {3.0, 4.0}});

// 基本类型转包装类 / Primitive to wrapper class conversion
Float[] wrapperFloat = RereMathUtil.toClassArray(new float[]{1.0f, 2.0f, 3.0f});
Double[] wrapperDouble = RereMathUtil.toClassArray(new double[]{1.0, 2.0, 3.0});
Integer[] wrapperInt = RereMathUtil.toClassArray(new int[]{1, 2, 3, 4});
```

### 2. 随机数生成 / Random Number Generation

#### 基本随机数 / Basic Random Numbers

```java
// 生成随机float数组 / Generate random float array
float[] randomFloats = RereMathUtil.generateRandomFloats(10);        // 10个随机float数 / 10 random float numbers

// 生成随机float数组（带种子）/ Generate random float array with seed
float[] randomFloatsSeeded = RereMathUtil.generateRandomFloats(42, 10);  // 使用种子42 / Using seed 42

// 生成随机int数组 / Generate random int array
int[] randomInts = RereMathUtil.generateRandomInts(1, 100, 10);      // 1到100之间的10个随机整数 / 10 random integers between 1 and 100

// 生成随机int数组（带种子）/ Generate random int array with seed
int[] randomIntsSeeded = RereMathUtil.generateRandomInts(42, 1, 100, 10);  // 使用种子42 / Using seed 42
```

#### 正态分布随机数 / Normal Distribution Random Numbers

```java
// 生成正态分布随机数 / Generate normal distribution random number
double normalSample = RereMathUtil.normalSample(0.0, 1.0);  // 标准正态分布 / Standard normal distribution
double normalSampleCustom = RereMathUtil.normalSample(5.0, 2.0);  // 均值5，标准差2 / Mean 5, std dev 2
```

### 3. 数学函数 / Mathematical Functions

#### 基本数学函数 / Basic Mathematical Functions

```java
// 平方 / Square
float squared = RereMathUtil.square(5.0f);                // 25.0

// 平方根 / Square root
float sqrt = RereMathUtil.sqrt(16.0f);                    // 4.0

// 绝对值 / Absolute value
float abs = RereMathUtil.abs(-5.0f);                      // 5.0

// 最大值 / Maximum
float max = RereMathUtil.max(1.0f, 2.0f, 3.0f);         // 3.0

// 最小值 / Minimum
float min = RereMathUtil.min(1.0f, 2.0f, 3.0f);         // 1.0
```

#### 三角函数 / Trigonometric Functions

```java
// 正弦 / Sine
float sin = RereMathUtil.sin(Math.PI / 2);                // 1.0

// 余弦 / Cosine
float cos = RereMathUtil.cos(0);                          // 1.0

// 正切 / Tangent
float tan = RereMathUtil.tan(Math.PI / 4);                // 1.0
```

#### 指数和对数函数 / Exponential and Logarithmic Functions

```java
// 指数 / Exponential
float exp = RereMathUtil.exp(1.0f);                       // 2.7182818

// 自然对数 / Natural logarithm
float log = RereMathUtil.log(Math.E);                     // 1.0

// 以10为底的对数 / Base-10 logarithm
float log10 = RereMathUtil.log10(100.0f);                 // 2.0
```

### 4. 概率分布相关数学函数 / Probability Distribution Mathematical Functions

#### 伽马函数和贝塔函数 / Gamma and Beta Functions

```java
// 伽马函数 / Gamma function
double gamma = RereMathUtil.gamma(5.0);                    // Γ(5) = 24

// 贝塔函数 / Beta function
double beta = RereMathUtil.beta(2.0, 3.0);                // B(2,3)

// 不完全伽马函数 / Incomplete gamma function
double incompleteGamma = RereMathUtil.incompleteGamma(2.0, 1.0);

// 不完全贝塔函数 / Incomplete beta function
double incompleteBeta = RereMathUtil.incompleteBeta(2.0, 3.0, 0.5);

// 正则化不完全贝塔函数 / Regularized incomplete beta function
double regIncompleteBeta = RereMathUtil.regularizedIncompleteBeta(2.0, 3.0, 0.5);

// 正则化不完全伽马函数 / Regularized incomplete gamma function
double regIncompleteGamma = RereMathUtil.regularizedIncompleteGamma(2, 1.0);
```

#### 误差函数和逆正态分布 / Error Function and Inverse Normal Distribution

```java
// 误差函数 / Error function
double erf = RereMathUtil.erf(1.0);                        // erf(1.0)

// 逆正态累积分布函数 / Inverse normal CDF
double invNormalCDF = RereMathUtil.inverseNormalCDF(0.95); // 95%分位数 / 95th percentile
```

### 5. 组合数学和统计函数 / Combinatorics and Statistical Functions

#### 组合数学 / Combinatorics

```java
// 组合数 / Combination
long combination = RereMathUtil.combination(10, 3);        // C(10,3) = 120

// 组合数的对数（避免溢出）/ Logarithm of combination (avoid overflow)
double logCombination = RereMathUtil.logCombination(100, 50);

// 阶乘 / Factorial
long factorial = RereMathUtil.factorial(10);               // 10! = 3628800

// 阶乘的对数 / Logarithm of factorial
double logFactorial = RereMathUtil.logFactorial(100);

// Stirling数 / Stirling number
double stirling = RereMathUtil.stirlingNumber2(5, 3);      // S(5,3)
```

### 6. 统计函数 / Statistical Functions

#### 基本统计 / Basic Statistics

```java
// 求和 / Sum
float sum = RereMathUtil.sum(new float[]{1, 2, 3, 4, 5}); // 15.0

// 均值 / Mean
float mean = RereMathUtil.mean(new float[]{1, 2, 3, 4, 5}); // 3.0

// 方差 / Variance
float variance = RereMathUtil.variance(new float[]{1, 2, 3, 4, 5}); // 2.0

// 标准差 / Standard deviation
float std = RereMathUtil.std(new float[]{1, 2, 3, 4, 5}); // 1.4142135
```

#### 最值函数 / Min/Max Functions

```java
// 数组最大值 / Array maximum
float max = RereMathUtil.max(new float[]{1, 5, 3, 9, 2}); // 9.0

// 数组最小值 / Array minimum
float min = RereMathUtil.min(new float[]{1, 5, 3, 9, 2}); // 1.0

// 最大值索引 / Index of maximum
int maxIndex = RereMathUtil.argmax(new float[]{1, 5, 3, 9, 2}); // 3

// 最小值索引 / Index of minimum
int minIndex = RereMathUtil.argmin(new float[]{1, 5, 3, 9, 2}); // 0
```

### 7. 数组操作 / Array Operations

#### 数组复制和填充 / Array Copy and Fill

```java
// 复制数组 / Copy array
float[] original = {1, 2, 3, 4, 5};
float[] copied = RereMathUtil.copy(original);

// 填充数组 / Fill array
float[] filled = RereMathUtil.fill(5, 10.0f);            // [10.0, 10.0, 10.0, 10.0, 10.0]

// 范围数组 / Range array
float[] range = RereMathUtil.range(5);                    // [0.0, 1.0, 2.0, 3.0, 4.0]
float[] rangeWithStep = RereMathUtil.range(0, 10, 2);    // [0.0, 2.0, 4.0, 6.0, 8.0]
```

#### 数组变换 / Array Transformations

```java
// 数组转置 / Array transpose
float[][] matrix = {{1, 2, 3}, {4, 5, 6}};
float[][] transposed = RereMathUtil.transpose(matrix);

// 数组重塑 / Array reshape
float[] flat = {1, 2, 3, 4, 5, 6};
float[][] reshaped = RereMathUtil.reshape(flat, 2, 3);  // 2x3矩阵 / 2x3 matrix
```

## 使用示例 / Usage Examples

### 示例1：类型转换 / Example 1: Type Conversion

```java
// 在不同数值类型间转换 / Convert between different numeric types
float[] floatData = {1.5f, 2.7f, 3.2f};

// 转换为double / Convert to double
double[] doubleData = RereMathUtil.floatToDouble(floatData);
// 结果: [1.5, 2.7, 3.2]

// 转换为int（截断小数）/ Convert to int (truncate decimal)
int[] intData = RereMathUtil.floatToInt(floatData);
// 结果: [1, 2, 3]

// 包装类转换 / Wrapper class conversion
Float[] wrapperFloats = {1.0f, 2.0f, 3.0f};
float[] primitiveFloats = RereMathUtil.toPrimitive(wrapperFloats);
```

### 示例2：随机数生成 / Example 2: Random Number Generation

```java
// 生成随机数据 / Generate random data
int size = 1000;

// 生成随机float数组 / Generate random float array
float[] randomFloats = RereMathUtil.generateRandomFloats(size);

// 生成带种子的随机float数组 / Generate random float array with seed
float[] randomFloatsSeeded = RereMathUtil.generateRandomFloats(42, size);

// 生成随机int数组 / Generate random int array
int[] randomInts = RereMathUtil.generateRandomInts(1, 100, size);

// 生成带种子的随机int数组 / Generate random int array with seed
int[] randomIntsSeeded = RereMathUtil.generateRandomInts(42, 1, 100, size);

// 生成正态分布随机数 / Generate normal distribution random numbers
double[] normalSamples = new double[size];
for (int i = 0; i < size; i++) {
    normalSamples[i] = RereMathUtil.normalSample(0.0, 1.0);
}
```

### 示例3：数学函数应用 / Example 3: Mathematical Function Application

```java
// 数学函数计算 / Mathematical function calculations
float x = 2.0f;

// 基本函数 / Basic functions
float xSquared = RereMathUtil.square(x);                  // 4.0
float xSqrt = RereMathUtil.sqrt(x);                       // 1.4142135
float xAbs = RereMathUtil.abs(-x);                        // 2.0

// 三角函数 / Trigonometric functions
float xSin = RereMathUtil.sin(x);                         // 0.9092974
float xCos = RereMathUtil.cos(x);                         // -0.4161468
float xTan = RereMathUtil.tan(x);                         // -2.1850398

// 指数和对数 / Exponential and logarithmic
float xExp = RereMathUtil.exp(x);                         // 7.389056
float xLog = RereMathUtil.log(x);                         // 0.6931472
```

### 示例4：概率分布和组合数学 / Example 4: Probability Distribution and Combinatorics

```java
// 概率分布函数 / Probability distribution functions
double gammaValue = RereMathUtil.gamma(5.0);                    // Γ(5) = 24
double betaValue = RereMathUtil.beta(2.0, 3.0);                // B(2,3)
double erfValue = RereMathUtil.erf(1.0);                       // erf(1.0)
double invNormal = RereMathUtil.inverseNormalCDF(0.95);        // 95%分位数

// 组合数学 / Combinatorics
long combination = RereMathUtil.combination(10, 3);            // C(10,3) = 120
long factorial = RereMathUtil.factorial(10);                   // 10! = 3628800
double logCombination = RereMathUtil.logCombination(100, 50);  // 避免溢出
double stirling = RereMathUtil.stirlingNumber2(5, 3);          // S(5,3)

// 不完全函数 / Incomplete functions
double incompleteGamma = RereMathUtil.incompleteGamma(2.0, 1.0);
double incompleteBeta = RereMathUtil.incompleteBeta(2.0, 3.0, 0.5);
double regIncompleteBeta = RereMathUtil.regularizedIncompleteBeta(2.0, 3.0, 0.5);
```

### 示例5：统计计算 / Example 5: Statistical Calculations

```java
// 数据集 / Dataset
float[] data = {12.5f, 15.2f, 18.7f, 14.3f, 16.8f, 13.9f, 17.1f, 19.4f};

// 基本统计 / Basic statistics
float totalSum = RereMathUtil.sum(data);                  // 127.9
float average = RereMathUtil.mean(data);                  // 15.9875
float dataVariance = RereMathUtil.variance(data);         // 6.234375
float dataStd = RereMathUtil.std(data);                   // 2.496873

// 最值 / Min/Max
float maxValue = RereMathUtil.max(data);                  // 19.4
float minValue = RereMathUtil.min(data);                  // 12.5
int maxIndex = RereMathUtil.argmax(data);                 // 7
int minIndex = RereMathUtil.argmin(data);                 // 0
```

### 示例6：数组操作 / Example 6: Array Operations

```java
// 数组操作 / Array operations
float[] original = {1, 2, 3, 4, 5, 6};

// 复制数组 / Copy array
float[] copied = RereMathUtil.copy(original);

// 填充数组 / Fill array
float[] filled = RereMathUtil.fill(5, 10.0f);            // [10.0, 10.0, 10.0, 10.0, 10.0]

// 范围数组 / Range array
float[] range = RereMathUtil.range(0, 10, 2);            // [0.0, 2.0, 4.0, 6.0, 8.0]

// 数组重塑 / Array reshape
float[][] matrix = RereMathUtil.reshape(original, 2, 3); // [[1, 2, 3], [4, 5, 6]]

// 矩阵转置 / Matrix transpose
float[][] transposed = RereMathUtil.transpose(matrix);    // [[1, 4], [2, 5], [3, 6]]
```

### 示例7：综合应用 / Example 7: Comprehensive Application

```java
// 数据预处理流程 / Data preprocessing pipeline
int sampleSize = 1000;

// 1. 生成随机数据 / Generate random data
float[] rawData = RereMathUtil.generateRandomFloats(sampleSize);
// 将数据缩放到[-100, 100]范围 / Scale data to [-100, 100] range
for (int i = 0; i < sampleSize; i++) {
    rawData[i] = (rawData[i] - 0.5f) * 200.0f;
}

// 2. 计算统计信息 / Calculate statistics
float dataMean = RereMathUtil.mean(rawData);
float dataStd = RereMathUtil.std(rawData);

// 3. 标准化数据 / Standardize data
float[] standardized = new float[sampleSize];
for (int i = 0; i < sampleSize; i++) {
    standardized[i] = (rawData[i] - dataMean) / dataStd;
}

// 4. 验证标准化结果 / Verify standardization results
float stdMean = RereMathUtil.mean(standardized);          // 应该接近0 / Should be close to 0
float stdStd = RereMathUtil.std(standardized);            // 应该接近1 / Should be close to 1

System.out.println("标准化后均值: " + stdMean);            // Standardized mean
System.out.println("标准化后标准差: " + stdStd);          // Standardized standard deviation
```

## 性能特性 / Performance Features

### 内存优化 / Memory Optimization
- 高效的数组操作，最小化内存分配
- 智能的类型转换，避免不必要的装箱拆箱
- 优化的随机数生成算法

### 算法优化 / Algorithm Optimization
- 高效的数学函数实现
- 优化的统计计算算法
- 快速的范围和填充操作

### 类型安全 / Type Safety
- 强类型系统，避免运行时错误
- 参数验证和边界检查
- 清晰的错误提示

## 注意事项 / Notes

1. **精度** / **Precision**: float类型有精度限制，注意数值精度
2. **内存** / **Memory**: 大数组操作时注意内存使用
3. **随机性** / **Randomness**: 随机数生成使用系统默认随机数生成器
4. **异常处理** / **Exception Handling**: 注意处理可能的异常情况

## 扩展性 / Extensibility

`RereMathUtil` 类设计支持扩展，可以轻松添加新的数学工具方法：
- 更多数学函数 / More mathematical functions
- 统计函数扩展 / Statistical function extensions
- 数值优化算法 / Numerical optimization algorithms
- 特殊数学函数 / Special mathematical functions

## 应用场景 / Application Scenarios

### 数据预处理 / Data Preprocessing
- 数据类型转换
- 数据标准化
- 随机数据生成

### 科学计算 / Scientific Computing
- 数学函数计算
- 统计分析
- 数值计算
- 概率分布计算
- 组合数学计算

### 机器学习 / Machine Learning
- 特征工程
- 数据增强
- 模型验证

---

**数学工具类** - 数学计算的基础工具，让复杂操作更简单！

**Math Utilities** - The foundational tools for mathematical computing, making complex operations simpler!
