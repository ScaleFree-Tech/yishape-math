# 数学工具类示例 (Math Utilities Examples)

## 概述 / Overview

本文档提供了 `RereMathUtil` 类的详细使用示例，展示各种数学工具函数的使用方法。

## 基础数学函数示例 / Basic Mathematical Function Examples

### 类型转换函数 / Type Conversion Functions

```java
import com.yishape.lab.math.RereMathUtil;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;

import java.util.Arrays;

public class TypeConversionExample {
    public static void main(String[] args) {
        System.out.println("=== 类型转换函数示例 / Type Conversion Functions Example ===");

        // 1. 数组类型转换 / Array type conversion
        System.out.println("1. 数组类型转换 / Array Type Conversion");

        double[] doubleArray = {1.1, 2.2, 3.3, 4.4, 5.5};
        int[] intArray = {1, 2, 3, 4, 5};
        float[] floatArray = {1.5f, 2.7f, 3.2f, 4.9f, 5.1f};

        // double[] 转 float[] / Convert double[] to float[]
        float[] doubleToFloatArray = RereMathUtil.doubleToFloat(doubleArray);
        System.out.println("double[] -> float[]: " + Arrays.toString(doubleToFloatArray));

        // int[] 转 float[] / Convert int[] to float[]
        float[] intToFloatArray = RereMathUtil.intToFloat(intArray);
        System.out.println("int[] -> float[]: " + Arrays.toString(intToFloatArray));

        // float[] 转 double[] / Convert float[] to double[]
        double[] floatToDoubleArray = RereMathUtil.floatToDouble(floatArray);
        System.out.println("float[] -> double[]: " + Arrays.toString(floatToDoubleArray));

        // int[] 转 double[] / Convert int[] to double[]
        double[] intToDoubleArray = RereMathUtil.intToDouble(intArray);
        System.out.println("int[] -> double[]: " + Arrays.toString(intToDoubleArray));

        // double[] 转 int[] / Convert double[] to int[]
        int[] doubleToIntArray = RereMathUtil.doubleToInt(doubleArray);
        System.out.println("double[] -> int[]: " + Arrays.toString(doubleToIntArray));

        // float[] 转 int[] / Convert float[] to int[]
        int[] floatToIntArray = RereMathUtil.floatToInt(floatArray);
        System.out.println("float[] -> int[]: " + Arrays.toString(floatToIntArray));

        // 2. 包装类转换 / Wrapper class conversion
        System.out.println("\n2. 包装类转换 / Wrapper Class Conversion");

        // 一维数组转换 / 1D array conversion
        Float[] wrapperFloatArray = {1.1f, 2.2f, 3.3f, 4.4f};
        Double[] wrapperDoubleArray = {1.1, 2.2, 3.3, 4.4};
        Integer[] wrapperIntArray = {1, 2, 3, 4};

        // 包装类转基本类型 / Wrapper to primitive
        float[] primitiveFloat = RereMathUtil.toPrimitive(wrapperFloatArray);
        double[] primitiveDouble = RereMathUtil.toPrimitive(wrapperDoubleArray);
        int[] primitiveInt = RereMathUtil.toPrimitive(wrapperIntArray);

        System.out.println("Float[] -> float[]: " + Arrays.toString(primitiveFloat));
        System.out.println("Double[] -> double[]: " + Arrays.toString(primitiveDouble));
        System.out.println("Integer[] -> int[]: " + Arrays.toString(primitiveInt));

        // 基本类型转包装类 / Primitive to wrapper
        Float[] backToWrapperFloat = RereMathUtil.toClassArray(primitiveFloat);
        Double[] backToWrapperDouble = RereMathUtil.toClassArray(primitiveDouble);
        Integer[] backToWrapperInt = RereMathUtil.toClassArray(primitiveInt);

        System.out.println("float[] -> Float[]: " + Arrays.toString(backToWrapperFloat));
        System.out.println("double[] -> Double[]: " + Arrays.toString(backToWrapperDouble));
        System.out.println("int[] -> Integer[]: " + Arrays.toString(backToWrapperInt));

        // 3. 二维数组转换 / 2D array conversion
        System.out.println("\n3. 二维数组转换 / 2D Array Conversion");

        Float[][] wrapperFloat2D = {{1.1f, 2.2f}, {3.3f, 4.4f}};
        Double[][] wrapperDouble2D = {{1.1, 2.2}, {3.3, 4.4}};

        // 二维包装类转基本类型 / 2D wrapper to primitive
        float[][] primitiveFloat2D = RereMathUtil.toPrimitive(wrapperFloat2D);
        double[][] primitiveDouble2D = RereMathUtil.toPrimitive(wrapperDouble2D);

        System.out.println("Float[][] -> float[][]:");
        for (float[] row : primitiveFloat2D) {
            System.out.println("  " + Arrays.toString(row));
        }

        System.out.println("Double[][] -> double[][]:");
        for (double[] row : primitiveDouble2D) {
            System.out.println("  " + Arrays.toString(row));
        }
    }
}
```

### 随机数生成 / Random Number Generation

```java
import com.yishape.lab.math.RereMathUtil;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;

public class RandomNumberExample {
    public static void main(String[] args) {
        System.out.println("=== 随机数生成示例 / Random Number Generation Example ===");

        // 1. 生成随机数组 / Generate random arrays
        System.out.println("1. 随机数组生成 / Random Array Generation");

        // 生成随机float数组 / Generate random float array
        float[] randomFloats = RereMathUtil.generateRandomFloats(10);
        System.out.println("随机float数组 (长度10): " + Arrays.toString(randomFloats));

        // 生成带种子的随机float数组 / Generate random float array with seed
        float[] randomFloatsSeeded = RereMathUtil.generateRandomFloats(42, 10);
        System.out.println("种子42的随机float数组: " + Arrays.toString(randomFloatsSeeded));

        // 生成随机int数组 / Generate random int array
        int[] randomInts = RereMathUtil.generateRandomInts(1, 100, 10);
        System.out.println("范围[1,100]随机int数组: " + Arrays.toString(randomInts));

        // 生成带种子的随机int数组 / Generate random int array with seed
        int[] randomIntsSeeded = RereMathUtil.generateRandomInts(42, 1, 100, 10);
        System.out.println("种子42的随机int数组: " + Arrays.toString(randomIntsSeeded));

        // 2. 正态分布随机数 / Normal distribution random numbers
        System.out.println("\n2. 正态分布随机数 / Normal Distribution Random Numbers");

        // 生成标准正态分布随机数 / Generate standard normal random numbers
        double[] normalSamples = new double[10];
        for (int i = 0; i < 10; i++) {
            normalSamples[i] = RereMathUtil.normalSample(0.0, 1.0);
        }
        System.out.println("标准正态分布随机数: " + Arrays.toString(normalSamples));

        // 生成自定义参数的正态分布随机数 / Generate normal random numbers with custom parameters
        double[] customNormalSamples = new double[10];
        for (int i = 0; i < 10; i++) {
            customNormalSamples[i] = RereMathUtil.normalSample(5.0, 2.0);
        }
        System.out.println("均值5，标准差2的正态分布随机数: " + Arrays.toString(customNormalSamples));

        // 3. 随机数种子验证 / Random number seed verification
        System.out.println("\n3. 随机数种子验证 / Random Number Seed Verification");

        // 使用相同种子生成两组随机数 / Generate two sets of random numbers with same seed
        int[] randomInts1 = RereMathUtil.generateRandomInts(42, 1, 100, 5);
        int[] randomInts2 = RereMathUtil.generateRandomInts(42, 1, 100, 5);
        System.out.println("种子42的随机int数组1: " + Arrays.toString(randomInts1));
        System.out.println("种子42的随机int数组2: " + Arrays.toString(randomInts2));
        System.out.println("两组数组是否相同: " + Arrays.equals(randomInts1, randomInts2));

        // 使用不同种子生成随机数 / Generate random numbers with different seeds
        int[] randomInts3 = RereMathUtil.generateRandomInts(123, 1, 100, 5);
        System.out.println("种子123的随机int数组: " + Arrays.toString(randomInts3));
        System.out.println("不同种子的数组是否相同: " + Arrays.equals(randomInts1, randomInts3));
    }
}
```

## 概率分布和组合数学示例 / Probability Distribution and Combinatorics Examples

### 概率分布函数 / Probability Distribution Functions

```java
import com.yishape.lab.math.RereMathUtil;

public class ProbabilityDistributionExample {
    public static void main(String[] args) {
        System.out.println("=== 概率分布函数示例 / Probability Distribution Functions Example ===");

        // 1. 伽马函数和贝塔函数 / Gamma and Beta functions
        System.out.println("1. 伽马函数和贝塔函数 / Gamma and Beta Functions");

        // 伽马函数 / Gamma function
        double gamma5 = RereMathUtil.gamma(5.0);
        double gamma10 = RereMathUtil.gamma(10.0);
        System.out.println("Γ(5) = " + gamma5);
        System.out.println("Γ(10) = " + gamma10);

        // 贝塔函数 / Beta function
        double beta23 = RereMathUtil.beta(2.0, 3.0);
        double beta55 = RereMathUtil.beta(5.0, 5.0);
        System.out.println("B(2,3) = " + beta23);
        System.out.println("B(5,5) = " + beta55);

        // 2. 不完全函数 / Incomplete functions
        System.out.println("\n2. 不完全函数 / Incomplete Functions");

        // 不完全伽马函数 / Incomplete gamma function
        double incompleteGamma21 = RereMathUtil.incompleteGamma(2.0, 1.0);
        double incompleteGamma52 = RereMathUtil.incompleteGamma(5.0, 2.0);
        System.out.println("γ(2,1) = " + incompleteGamma21);
        System.out.println("γ(5,2) = " + incompleteGamma52);

        // 不完全贝塔函数 / Incomplete beta function
        double incompleteBeta235 = RereMathUtil.incompleteBeta(2.0, 3.0, 0.5);
        double incompleteBeta551 = RereMathUtil.incompleteBeta(5.0, 5.0, 0.1);
        System.out.println("B(2,3,0.5) = " + incompleteBeta235);
        System.out.println("B(5,5,0.1) = " + incompleteBeta551);

        // 正则化不完全函数 / Regularized incomplete functions
        double regIncompleteBeta235 = RereMathUtil.regularizedIncompleteBeta(2.0, 3.0, 0.5);
        double regIncompleteGamma21 = RereMathUtil.regularizedIncompleteGamma(2, 1.0);
        System.out.println("I(2,3,0.5) = " + regIncompleteBeta235);
        System.out.println("P(2,1) = " + regIncompleteGamma21);

        // 3. 误差函数和逆正态分布 / Error function and inverse normal distribution
        System.out.println("\n3. 误差函数和逆正态分布 / Error Function and Inverse Normal Distribution");

        // 误差函数 / Error function
        double erf1 = RereMathUtil.erf(1.0);
        double erf2 = RereMathUtil.erf(2.0);
        System.out.println("erf(1.0) = " + erf1);
        System.out.println("erf(2.0) = " + erf2);

        // 逆正态累积分布函数 / Inverse normal CDF
        double invNormal95 = RereMathUtil.inverseNormalCDF(0.95);
        double invNormal99 = RereMathUtil.inverseNormalCDF(0.99);
        System.out.println("Φ⁻¹(0.95) = " + invNormal95);
        System.out.println("Φ⁻¹(0.99) = " + invNormal99);
    }
}
```

### 组合数学函数 / Combinatorics Functions

```java
import com.yishape.lab.math.RereMathUtil;

public class CombinatoricsExample {
    public static void main(String[] args) {
        System.out.println("=== 组合数学函数示例 / Combinatorics Functions Example ===");

        // 1. 组合数 / Combinations
        System.out.println("1. 组合数 / Combinations");

        // 基本组合数 / Basic combinations
        long c10_3 = RereMathUtil.combination(10, 3);
        long c20_5 = RereMathUtil.combination(20, 5);
        System.out.println("C(10,3) = " + c10_3);
        System.out.println("C(20,5) = " + c20_5);

        // 大数组合数（使用对数避免溢出）/ Large combinations (using log to avoid overflow)
        double logC100_50 = RereMathUtil.logCombination(100, 50);
        double logC200_100 = RereMathUtil.logCombination(200, 100);
        System.out.println("ln C(100,50) = " + logC100_50);
        System.out.println("ln C(200,100) = " + logC200_100);

        // 2. 阶乘 / Factorials
        System.out.println("\n2. 阶乘 / Factorials");

        // 基本阶乘 / Basic factorials
        long fact10 = RereMathUtil.factorial(10);
        long fact15 = RereMathUtil.factorial(15);
        System.out.println("10! = " + fact10);
        System.out.println("15! = " + fact15);

        // 大数阶乘（使用对数）/ Large factorials (using log)
        double logFact100 = RereMathUtil.logFactorial(100);
        double logFact1000 = RereMathUtil.logFactorial(1000);
        System.out.println("ln 100! = " + logFact100);
        System.out.println("ln 1000! = " + logFact1000);

        // 3. Stirling数 / Stirling numbers
        System.out.println("\n3. Stirling数 / Stirling Numbers");

        // 第二类Stirling数 / Stirling numbers of the second kind
        double s52 = RereMathUtil.stirlingNumber2(5, 2);
        double s73 = RereMathUtil.stirlingNumber2(7, 3);
        double s84 = RereMathUtil.stirlingNumber2(8, 4);
        System.out.println("S(5,2) = " + s52);
        System.out.println("S(7,3) = " + s73);
        System.out.println("S(8,4) = " + s84);

        // 4. 实际应用示例 / Practical application examples
        System.out.println("\n4. 实际应用示例 / Practical Application Examples");

        // 计算概率 / Calculate probabilities
        int n = 20, k = 5;
        double p = 0.3;

        // 二项分布概率 / Binomial distribution probability
        double logProb = RereMathUtil.logCombination(n, k) + k * Math.log(p) + (n - k) * Math.log(1 - p);
        double prob = Math.exp(logProb);
        System.out.println("二项分布 P(X=" + k + ") = " + prob);

        // 泊松分布近似 / Poisson distribution approximation
        double lambda = n * p;
        double poissonProb = Math.exp(-lambda + k * Math.log(lambda) - RereMathUtil.logFactorial(k));
        System.out.println("泊松分布近似 P(X=" + k + ") = " + poissonProb);
    }
}
```

## 数学函数示例 / Mathematical Function Examples

### 特殊数学函数 / Special Mathematical Functions

```java
import com.yishape.lab.math.RereMathUtil;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;

public class SpecialFunctionsExample {
    public static void main(String[] args) {
        System.out.println("=== 特殊数学函数示例 / Special Functions Example ===");

        // 1. 伽马函数和贝塔函数 / Gamma and Beta functions
        System.out.println("1. 伽马函数和贝塔函数 / Gamma and Beta Functions");
        double gamma5 = RereMathUtil.gamma(5.0);               // Γ(5) = 24
        double beta23 = RereMathUtil.beta(2.0, 3.0);          // B(2,3)
        System.out.println("Γ(5) = " + gamma5);
        System.out.println("B(2,3) = " + beta23);

        // 不完全伽马/贝塔函数 / Incomplete gamma/beta functions
        double igamma = RereMathUtil.incompleteGamma(2.0, 1.0);
        double ibeta = RereMathUtil.incompleteBeta(2.0, 3.0, 0.5);
        System.out.println("不完全伽马(2,1) = " + igamma);
        System.out.println("不完全贝塔(2,3,0.5) = " + ibeta);

        // 2. 误差函数和逆正态分布 / Error function and inverse normal
        System.out.println("\n2. 误差函数和逆正态分布 / Error Function and Inverse Normal");
        double erf1 = RereMathUtil.erf(1.0);                  // erf(1.0)
        double invNormal95 = RereMathUtil.inverseNormalCDF(0.95); // 95%分位数
        System.out.println("erf(1.0) = " + erf1);
        System.out.println("正态分布 95% 分位数 = " + invNormal95);

        // 3. Sigmoid 函数 / Sigmoid function
        System.out.println("\n3. Sigmoid 函数 / Sigmoid Function");
        double[] xs = {-2.0, -1.0, 0.0, 1.0, 2.0};
        for (double x : xs) {
            System.out.printf("sigmoid(%.1f) = %.4f%n", x, RereMathUtil.sigmoid(x));
        }

        // 4. 浮点近似比较 / Floating point comparison
        System.out.println("\n4. 浮点近似比较 / Floating Point Comparison");
        System.out.println("isClose(1.0, 1.000001): " + RereMathUtil.isClose(1.0, 1.000001));
        System.out.println("isClose(1.0, 1.1): " + RereMathUtil.isClose(1.0, 1.1));
    }
}
```

### 统计函数 / Statistical Functions

```java
import com.yishape.lab.math.RereMathUtil;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;

public class StatisticalFunctionsExample {
    public static void main(String[] args) {
        System.out.println("=== 统计函数示例 / Statistical Functions Example ===");

        // 1. 向量统计 / Vector statistics
        System.out.println("1. 向量统计 / Vector Statistics");

        float[] data = {1.2f, 3.4f, 5.6f, 7.8f, 9.0f, 2.1f, 4.3f, 6.5f, 8.7f, 0.9f};
        System.out.println("数据: " + Arrays.toString(data));

        IVector dataVector = IVector.of(data);
        System.out.println("数据向量: " + dataVector);

        float vectorSum = dataVector.sum();
        float vectorMean = dataVector.mean();
        float vectorVariance = dataVector.variance();
        float vectorStdDev = dataVector.std();

        System.out.println("向量总和: " + vectorSum);
        System.out.println("向量均值: " + vectorMean);
        System.out.println("向量方差: " + vectorVariance);
        System.out.println("向量标准差: " + vectorStdDev);

        // 2. 矩阵统计 / Matrix statistics
        System.out.println("\n2. 矩阵统计 / Matrix Statistics");

        float[][] matrixData = {
                {1.1f, 2.2f, 3.3f},
                {4.4f, 5.5f, 6.6f},
                {7.7f, 8.8f, 9.9f}
        };
        IMatrix dataMatrix = IMatrix.of(matrixData);
        System.out.println("数据矩阵: " + dataMatrix);

        float matrixSum = dataMatrix.sum();
        float matrixMean = dataMatrix.mean();

        System.out.println("矩阵总和: " + matrixSum);
        System.out.println("矩阵均值: " + matrixMean);

        // 3. 偏度和峰度 / Skewness and kurtosis
        System.out.println("\n3. 偏度和峰度 / Skewness and Kurtosis");

        float[] randomData = RereMathUtil.generateRandomFloats(100);
        System.out.println("生成100个随机数");

        IVector randomVec = IVector.of(randomData);
        float dataSkew = randomVec.skewness();
        float dataKurt = randomVec.kurtosis();

        System.out.println("偏度: " + dataSkew);
        System.out.println("峰度: " + dataKurt);
    }
}
```

## 数组操作示例 / Array Operation Examples

### 数组变换和操作 / Array Transformation and Operations

```java
import com.yishape.lab.math.RereMathUtil;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;

public class ArrayOperationsExample {
    public static void main(String[] args) {
        System.out.println("=== 数组操作示例 / Array Operations Example ===");

        // 1. 类型转换 / Type conversion
        System.out.println("1. 类型转换 / Type Conversion");

        Float[] boxedArray = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float[] primitiveArray = RereMathUtil.toPrimitive(boxedArray);
        System.out.println("Float[] → float[]: " + Arrays.toString(primitiveArray));

        double[] doubleArray = RereMathUtil.floatToDouble(primitiveArray);
        System.out.println("float[] → double[]: " + Arrays.toString(doubleArray));

        // 2. 随机数生成 / Random number generation
        System.out.println("\n2. 随机数生成 / Random Number Generation");

        float[] randomFloats = RereMathUtil.generateRandomFloats(10);
        IVector randomVec = IVector.of(randomFloats);
        System.out.println("生成10个随机数: " + randomVec);
        System.out.println("均值: " + randomVec.mean());
        System.out.println("标准差: " + randomVec.std());

        // 3. 数组统计 / Array statistics
        System.out.println("\n3. 数组统计 / Array Statistics");

        float[] data = {1.2f, 3.4f, 5.6f, 7.8f, 9.0f};
        IVector dataVector = IVector.of(data);
        System.out.println("数据: " + dataVector);
        System.out.println("总和: " + dataVector.sum());
        System.out.println("均值: " + dataVector.mean());
        System.out.println("标准差: " + dataVector.std());
        System.out.println("偏度: " + dataVector.skewness());
        System.out.println("峰度: " + dataVector.kurtosis());
        System.out.println("排序后: " + dataVector.sort());
    }
}
```

## 实际应用示例 / Practical Application Examples

### 数据预处理 / Data Preprocessing

```java
import com.yishape.lab.math.RereMathUtil;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;

public class DataPreprocessingExample {
    public static void main(String[] args) {
        System.out.println("=== 数据预处理示例 / Data Preprocessing Example ===");

        // 1. 模拟真实数据集 / Simulate real dataset
        System.out.println("1. 模拟真实数据集 / Simulate Real Dataset");

        // 生成包含噪声的数据 / Generate data with noise
        int samples = 1000;
        float[] trueValues = new float[samples];
        float[] noise = RereMathUtil.generateRandomFloats(samples);
        float[] noisyData = new float[samples];

        // 生成线性趋势数据 / Generate linear trend data
        for (int i = 0; i < samples; i++) {
            trueValues[i] = (float) i / samples * 10.0f;
            // 添加正态分布噪声 / Add normal distribution noise
            noise[i] = (float) RereMathUtil.normalSample(0.0, 0.5);
            noisyData[i] = trueValues[i] + noise[i];
        }

        System.out.println("生成 " + samples + " 个样本");
        IVector noisyVec = IVector.of(noisyData);
        System.out.println("数据范围: [" + noisyVec.min() + ", " + noisyVec.max() + "]");

        // 2. 数据清洗 / Data cleaning
        System.out.println("\n2. 数据清洗 / Data Cleaning");

        // 检测异常值 / Detect outliers
        float mean = noisyVec.mean();
        float std = noisyVec.std();
        float threshold = 3.0f; // 3σ原则 / 3-sigma rule

        int outlierCount = 0;
        for (float value : noisyData) {
            if (Math.abs(value - mean) > threshold * std) {
                outlierCount++;
            }
        }

        System.out.println("检测到的异常值数量: " + outlierCount);
        System.out.println("异常值比例: " + (outlierCount * 100.0f / samples) + "%");

        // 3. 数据标准化 / Data standardization
        System.out.println("\n3. 数据标准化 / Data Standardization");

        // Z-score标准化 / Z-score standardization
        float[] zScoreData = new float[samples];
        float zMean = noisyVec.mean();
        float zStd = noisyVec.std();
        for (int i = 0; i < samples; i++) {
            zScoreData[i] = (noisyData[i] - zMean) / zStd;
        }
        float zScoreMean = IVector.of(zScoreData).mean();
        float zScoreStd = IVector.of(zScoreData).std();

        System.out.println("Z-score标准化后:");
        System.out.println("  均值: " + zScoreMean);
        System.out.println("  标准差: " + zScoreStd);

        // Min-Max归一化 / Min-Max normalization
        float[] minMaxData = new float[samples];
        float minVal = noisyVec.min();
        float maxVal = noisyVec.max();
        for (int i = 0; i < samples; i++) {
            minMaxData[i] = (noisyData[i] - minVal) / (maxVal - minVal);
        }
        float minMaxMin = IVector.of(minMaxData).min();
        float minMaxMax = IVector.of(minMaxData).max();

        System.out.println("Min-Max归一化后:");
        System.out.println("  最小值: " + minMaxMin);
        System.out.println("  最大值: " + minMaxMax);

        // 4. 特征工程 / Feature engineering
        System.out.println("\n4. 特征工程 / Feature Engineering");

        // 创建多项式特征 / Create polynomial features
        float[] x = new float[100];
        for (int i = 0; i < 100; i++) {
            x[i] = -2.0f + 4.0f * i / 99.0f;  // 从-2到2的线性空间
        }
        float[] xSquared = new float[x.length];
        float[] xCubed = new float[x.length];

        for (int i = 0; i < x.length; i++) {
            xSquared[i] = x[i] * x[i];
            xCubed[i] = x[i] * x[i] * x[i];
        }

        // 计算特征间的相关性 / Calculate correlation between features
        // 注意：这里需要实现correlation方法，或者使用其他方式计算
        System.out.println("特征工程完成:");
        System.out.println("  原始特征x: 长度 " + x.length);
        System.out.println("  二次特征x²: 长度 " + xSquared.length);
        System.out.println("  三次特征x³: 长度 " + xCubed.length);

        // 5. 数据采样 / Data sampling
        System.out.println("\n5. 数据采样 / Data Sampling");

        // 随机采样 / Random sampling
        int sampleSize = 100;
        int[] randomIndices = RereMathUtil.generateRandomInts(0, samples, sampleSize);
        float[] sampledData = new float[sampleSize];
        for (int i = 0; i < sampleSize; i++) {
            sampledData[i] = noisyData[randomIndices[i]];
        }

        System.out.println("随机采样 " + sampleSize + " 个样本");
        IVector sampledVec = IVector.of(sampledData);
        System.out.println("采样后数据范围: [" + sampledVec.min() + ", " + sampledVec.max() + "]");
    }
}
```

## 总结 / Summary

本文档展示了数学工具类的各种使用方法。建议在实际使用中：

1. **合理选择数据类型** / **Choose appropriate data types**
2. **注意数值精度** / **Pay attention to numerical precision**
3. **合理使用随机数** / **Use random numbers reasonably**
4. **注意内存使用** / **Pay attention to memory usage**

---

**数学工具类示例** - 从基础到应用，掌握数学工具的精髓！
