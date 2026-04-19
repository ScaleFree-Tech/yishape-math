# 数学工具类 (Math Utilities)

## 概述 / Overview

`RereMathUtil` 类提供了各种数学相关的实用方法，包括类型转换、随机数生成、数学函数等功能。这些工具方法为向量和矩阵操作提供了基础支持，确保数据类型的一致性和操作的便利性。

The `RereMathUtil` class provides various mathematical utility methods including type conversion, random number generation, mathematical functions, and more. These utility methods provide foundational support for vector and matrix operations, ensuring data type consistency and operational convenience.

## 快速上手 / Quick Start

```java
// 类型转换 / Type conversion
float[] farr = RereMathUtil.floatToDouble(new float[]{1.0f, 2.0f}); // → double[]
double[] darr = RereMathUtil.intToDouble(new int[]{1, 2, 3});      // → double[]

// 生成随机数 / Random numbers
float[] randoms = RereMathUtil.generateRandomFloats(100);          // 100个随机float
double sample = RereMathUtil.normalSample(0.0, 1.0);               // 正态分布采样

// 特殊数学函数 / Special math functions
double gamma5 = RereMathUtil.gamma(5.0);            // Γ(5) = 24
double sigmoid0 = RereMathUtil.sigmoid(0.0);         // sigmoid(0) = 0.5
boolean close = RereMathUtil.isClose(1.0, 1.0001);  // 浮点比较
```

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

### 3. 组合数学 / Combinatorics

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

### 示例3：概率分布与特殊函数 / Example 3: Probability Distribution and Special Functions

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

// 不完全函数 / Incomplete functions
double incompleteBeta = RereMathUtil.incompleteBeta(2.0, 3.0, 0.5);
double incompleteGamma = RereMathUtil.incompleteGamma(2.0, 1.0);
```

### 示例4：特殊函数应用 / Example 4: Special Function Applications

```java
// 计算 stirling 数 / Calculate Stirling numbers
double stirling = RereMathUtil.stirlingNumber2(5, 3);          // S(5,3)

// 概率分布采样 / Probability distribution sampling
double sample = RereMathUtil.normalSample(0.0, 1.0);        // 标准正态采样

// sigmoid 函数 / Sigmoid function
double sig = RereMathUtil.sigmoid(0.0);                       // 0.5

// 安全类型转换 / Safe type conversion
double safe = RereMathUtil.safeDoubleValue(null, 0.0);       // 0.0 (默认值)

// 浮点近似比较 / Floating point approximate comparison
boolean close = RereMathUtil.isClose(1.0, 1.00000001);      // true
```

// 最值 / Min/Max
### 示例5：类型转换综合 / Example 5: Type Conversion Comprehensive

```java
// 类型转换示例 / Type conversion examples
double[] fromFloat = RereMathUtil.floatToDouble(new float[]{1.0f, 2.0f, 3.0f});
float[] fromDouble = RereMathUtil.doubleToFloat(new double[]{1.0, 2.0, 3.0});

// 包装类与基本类型互转 / Wrapper to primitive and back
Float[] wrappers = {1.0f, 2.0f, 3.0f};
float[] primitives = RereMathUtil.toPrimitive(wrappers);
Float[] backToWrappers = RereMathUtil.toClassArray(primitives);
```

### 示例6：随机数与组合数学 / Example 6: Random Numbers and Combinatorics

```java
// 生成随机整数数组（指定范围）/ Generate random int array with range
int[] randomInts = RereMathUtil.generateRandomInts(42, 1, 100, 10); // 种子42，1-100之间10个数

// 生成随机 float 数组（带种子）/ Generate random float array with seed
float[] randomFloats = RereMathUtil.generateRandomFloats(42, 100); // 种子42，100个数

// 组合数 / Combination
long comb = RereMathUtil.combination(100, 50); // C(100,50)，注意溢出风险
double logComb = RereMathUtil.logCombination(100, 50); // 使用对数避免溢出

// 阶乘与 Stirling 数 / Factorial and Stirling numbers
long fact10 = RereMathUtil.factorial(10); // 10! = 3628800
double stirling = RereMathUtil.stirlingNumber2(5, 3); // 第二类 Stirling 数 S(5,3)
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

## 常见问题 / FAQ

### Q1: `RereMathUtil` 和 `Stats` 有什么区别？

| 类 | 职责 | 示例 |
|----|------|------|
| `RereMathUtil` | 低级数学函数（数学库级别） | `gamma()`, `sigmoid()`, `isClose()` |
| `Stats` | 概率分布和统计推断 | `Stats.norm()`, `Stats.tester` |

简单记忆：**有「分布」「检验」「估计」需求的用 `Stats`，其余底层数学用 `RereMathUtil`**。

### Q2: `isClose(a, b)` 和 `a == b` 有什么区别？

```java
// == 比较精确相等，对浮点数几乎永不成立
boolean eq = (0.1 + 0.2 == 0.3);        // → false（浮点误差）

// isClose 允许一定相对/绝对误差
boolean close = RereMathUtil.isClose(0.1 + 0.2, 0.3);  // → true
boolean close2 = RereMathUtil.isClose(1.0, 1.0001, 0.001, 0.0);  // 自定义容差
```

### Q3: 已有 `Stats.norm()`，`RereMathUtil.randn()` 和 `Stats.norm().sample()` 该用哪个？

| 需求 | 方法 | 说明 |
|------|------|------|
| 生成 n 个标准正态随机数数组 | `Linalg.randn(n)` | 最简洁 |
| 生成 n 个一般正态随机数数组 | `Linalg.randn(n, mean, std)` | 最简洁 |
| 单次采样（概率分布接口）| `Stats.norm().sample()` | 可复用分布对象 |
| 底层 float 数组 | `RereMathUtil.randn(n)` 或 `RereMathUtil.randn(seed, n)` | 确定性（可复现）|

### Q4: `normalSample` 和 `sigmoid` 属于同一类工具吗？

不属于。`normalSample` 是**统计学采样**（依赖概率分布），`sigmoid` 是**确定性数学函数**。它们分别对应：
- `RereMathUtil.normalSample` → 正态分布随机采样
- `RereMathUtil.sigmoid` → 激活函数 `1/(1+e^(-x))`
