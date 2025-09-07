# Stat类使用示例 / Stat Class Usage Examples

## 概述 / Overview

Stat类提供了创建各种概率分布对象的静态工厂方法，是统计学计算的核心入口点。本文档提供了详细的使用示例，展示如何使用Stat类进行各种统计计算。

The Stat class provides static factory methods for creating various probability distribution objects, serving as the core entry point for statistical calculations. This document provides detailed usage examples demonstrating how to use the Stat class for various statistical computations.

## 目录 / Table of Contents

1. [基本用法 / Basic Usage](#基本用法--basic-usage)
2. [正态分布示例 / Normal Distribution Examples](#正态分布示例--normal-distribution-examples)
3. [t分布示例 / t-Distribution Examples](#t分布示例--t-distribution-examples)
4. [均匀分布示例 / Uniform Distribution Examples](#均匀分布示例--uniform-distribution-examples)
5. [指数分布示例 / Exponential Distribution Examples](#指数分布示例--exponential-distribution-examples)
6. [卡方分布示例 / Chi-Squared Distribution Examples](#卡方分布示例--chi-squared-distribution-examples)
7. [F分布示例 / F-Distribution Examples](#f分布示例--f-distribution-examples)
8. [综合应用示例 / Comprehensive Application Examples](#综合应用示例--comprehensive-application-examples)

## 基本用法 / Basic Usage

### 导入Stat类 / Import Stat Class

```java
import com.reremouse.lab.math.stat.Stat;
import com.reremouse.lab.math.stat.distribution.*;
```

### 创建分布对象 / Creating Distribution Objects

```java
// 创建各种分布对象
// Create various distribution objects
NormalDistribution normal = Stat.norm(5.0f, 2.0f);           // 正态分布
StudentDistribution tDist = Stat.t(10.0f);                   // t分布
UniformDistribution uniform = Stat.uniform(0.0f, 10.0f);     // 均匀分布
ExponentialDistribution exp = Stat.exponential(1.5f);        // 指数分布
Chi2Distribution chi2 = Stat.chi2(5.0f);                     // 卡方分布
FDistribution fDist = Stat.f(3.0f, 10.0f);                   // F分布
```

## 正态分布示例 / Normal Distribution Examples

### 基本操作 / Basic Operations

```java
// 创建标准正态分布（均值为0，标准差为1）
// Create standard normal distribution (mean=0, stdDev=1)
NormalDistribution stdNormal = Stat.norm();

// 创建自定义正态分布
// Create custom normal distribution
NormalDistribution normal = Stat.norm(5.0f, 2.0f);

// 计算概率密度函数值
// Calculate probability density function value
float pdfValue = normal.pdf(5.0f);  // 在均值处的概率密度
System.out.println("PDF at mean: " + pdfValue);

// 计算累积分布函数值
// Calculate cumulative distribution function value
float cdfValue = normal.cdf(7.0f);  // P(X ≤ 7)
System.out.println("CDF at 7: " + cdfValue);

// 计算百分点函数值（分位数）
// Calculate percent point function value (quantile)
float ppfValue = normal.ppf(0.95f);  // 95%分位数
System.out.println("95th percentile: " + ppfValue);
```

### 统计量计算 / Statistical Measures

```java
NormalDistribution normal = Stat.norm(10.0f, 3.0f);

// 基本统计量
// Basic statistical measures
System.out.println("Mean: " + normal.mean());           // 均值
System.out.println("Variance: " + normal.var());        // 方差
System.out.println("Standard Deviation: " + normal.std()); // 标准差
System.out.println("Median: " + normal.median());       // 中位数
System.out.println("Mode: " + normal.mode());           // 众数

// 四分位数
// Quartiles
System.out.println("Q1 (25th percentile): " + normal.q1());
System.out.println("Q3 (75th percentile): " + normal.q3());

// 形状统计量
// Shape statistics
System.out.println("Skewness: " + normal.skewness());   // 偏度
System.out.println("Kurtosis: " + normal.kurtosis());   // 峰度
```

### 随机采样 / Random Sampling

```java
NormalDistribution normal = Stat.norm(0.0f, 1.0f);

// 生成单个随机样本
// Generate single random sample
float sample = normal.sample();
System.out.println("Random sample: " + sample);

// 生成多个随机样本
// Generate multiple random samples
float[] samples = normal.sample(1000);
System.out.println("Generated " + samples.length + " samples");

// 计算样本统计量
// Calculate sample statistics
float sum = 0.0f;
for (float s : samples) {
    sum += s;
}
float sampleMean = sum / samples.length;
System.out.println("Sample mean: " + sampleMean);
```

### 置信区间计算 / Confidence Interval Calculation

```java
// 计算95%置信区间
// Calculate 95% confidence interval
NormalDistribution normal = Stat.norm(0.0f, 1.0f);
float alpha = 0.05f;
float zScore = normal.ppf(1.0f - alpha/2.0f);  // 1.96 for 95% CI

float sampleMean = 5.0f;
float sampleStd = 2.0f;
float sampleSize = 100.0f;
float marginOfError = zScore * sampleStd / (float) Math.sqrt(sampleSize);

float lowerBound = sampleMean - marginOfError;
float upperBound = sampleMean + marginOfError;

System.out.println("95% Confidence Interval: [" + lowerBound + ", " + upperBound + "]");
```

## t分布示例 / t-Distribution Examples

### 小样本推断 / Small Sample Inference

```java
// 创建t分布（自由度为10）
// Create t-distribution (degrees of freedom = 10)
StudentDistribution tDist = Stat.t(10.0f);

// 计算t统计量的临界值
// Calculate critical values for t-statistic
float alpha = 0.05f;
float tCritical = tDist.ppf(1.0f - alpha/2.0f);
System.out.println("t-critical (α=0.05, df=10): " + tCritical);

// 计算p值
// Calculate p-value
float tStatistic = 2.5f;
float pValue = 2.0f * (1.0f - tDist.cdf(Math.abs(tStatistic)));
System.out.println("p-value for t=" + tStatistic + ": " + pValue);
```

### 置信区间估计 / Confidence Interval Estimation

```java
// 小样本置信区间估计
// Small sample confidence interval estimation
StudentDistribution tDist = Stat.t(15.0f);  // 自由度 = 15

float sampleMean = 25.0f;
float sampleStd = 5.0f;
float sampleSize = 16.0f;
float confidenceLevel = 0.95f;

float alpha = 1.0f - confidenceLevel;
float tValue = tDist.ppf(1.0f - alpha/2.0f);
float marginOfError = tValue * sampleStd / (float) Math.sqrt(sampleSize);

float lowerBound = sampleMean - marginOfError;
float upperBound = sampleMean + marginOfError;

System.out.println("95% Confidence Interval (t-distribution): [" + 
                  lowerBound + ", " + upperBound + "]");
```

## 均匀分布示例 / Uniform Distribution Examples

### 基本操作 / Basic Operations

```java
// 创建均匀分布 [0, 10]
// Create uniform distribution [0, 10]
UniformDistribution uniform = Stat.uniform(0.0f, 10.0f);

// 计算概率密度
// Calculate probability density
float pdfValue = uniform.pdf(5.0f);  // 在区间内的概率密度
System.out.println("PDF at 5: " + pdfValue);

// 计算累积概率
// Calculate cumulative probability
float cdfValue = uniform.cdf(7.0f);  // P(X ≤ 7)
System.out.println("CDF at 7: " + cdfValue);

// 计算分位数
// Calculate quantile
float quantile = uniform.ppf(0.8f);  // 80%分位数
System.out.println("80th percentile: " + quantile);
```

### 蒙特卡洛模拟 / Monte Carlo Simulation

```java
// 使用均匀分布进行蒙特卡洛模拟
// Monte Carlo simulation using uniform distribution
UniformDistribution uniform = Stat.uniform(0.0f, 1.0f);

// 估计π值
// Estimate π value
int nSamples = 1000000;
int insideCircle = 0;

for (int i = 0; i < nSamples; i++) {
    float x = uniform.sample();
    float y = uniform.sample();
    
    if (x * x + y * y <= 1.0f) {
        insideCircle++;
    }
}

float piEstimate = 4.0f * insideCircle / nSamples;
System.out.println("π estimate: " + piEstimate);
System.out.println("Error: " + Math.abs(piEstimate - Math.PI));
```

## 指数分布示例 / Exponential Distribution Examples

### 可靠性分析 / Reliability Analysis

```java
// 创建指数分布（故障率λ = 0.1）
// Create exponential distribution (failure rate λ = 0.1)
ExponentialDistribution exp = Stat.exponential(0.1f);

// 计算生存概率
// Calculate survival probability
float time = 10.0f;
float survivalProb = exp.sf(time);  // P(T > 10)
System.out.println("Survival probability at t=10: " + survivalProb);

// 计算平均故障时间
// Calculate mean time to failure
float mttf = exp.mean();
System.out.println("Mean time to failure: " + mttf);

// 计算中位数
// Calculate median
float median = exp.median();
System.out.println("Median time to failure: " + median);
```

### 排队论应用 / Queueing Theory Application

```java
// 模拟M/M/1排队系统
// Simulate M/M/1 queueing system
ExponentialDistribution arrivalRate = Stat.exponential(2.0f);  // 到达率
ExponentialDistribution serviceRate = Stat.exponential(3.0f);  // 服务率

// 模拟100个顾客
// Simulate 100 customers
int nCustomers = 100;
float totalWaitTime = 0.0f;

for (int i = 0; i < nCustomers; i++) {
    float interArrivalTime = arrivalRate.sample();
    float serviceTime = serviceRate.sample();
    // 简化的等待时间计算
    // Simplified waiting time calculation
    totalWaitTime += serviceTime;
}

float avgWaitTime = totalWaitTime / nCustomers;
System.out.println("Average waiting time: " + avgWaitTime);
```

## 卡方分布示例 / Chi-Squared Distribution Examples

### 拟合优度检验 / Goodness of Fit Test

```java
// 创建卡方分布（自由度为5）
// Create chi-squared distribution (degrees of freedom = 5)
Chi2Distribution chi2 = Stat.chi2(5.0f);

// 计算卡方统计量的临界值
// Calculate critical values for chi-squared statistic
float alpha = 0.05f;
float chi2Critical = chi2.ppf(1.0f - alpha);
System.out.println("Chi-squared critical (α=0.05, df=5): " + chi2Critical);

// 计算p值
// Calculate p-value
float chi2Statistic = 8.5f;
float pValue = 1.0f - chi2.cdf(chi2Statistic);
System.out.println("p-value for χ²=" + chi2Statistic + ": " + pValue);
```

### 方差检验 / Variance Test

```java
// 单样本方差检验
// One-sample variance test
Chi2Distribution chi2 = Stat.chi2(9.0f);  // 自由度 = n-1 = 10-1 = 9

float sampleVariance = 4.0f;
float populationVariance = 3.0f;
float sampleSize = 10.0f;

float chi2Statistic = (sampleSize - 1) * sampleVariance / populationVariance;
float pValue = 1.0f - chi2.cdf(chi2Statistic);

System.out.println("Chi-squared statistic: " + chi2Statistic);
System.out.println("p-value: " + pValue);

if (pValue < 0.05f) {
    System.out.println("Reject null hypothesis: variance is significantly different");
} else {
    System.out.println("Fail to reject null hypothesis: variance is not significantly different");
}
```

## F分布示例 / F-Distribution Examples

### 方差分析 / Analysis of Variance

```java
// 创建F分布（分子自由度=3，分母自由度=20）
// Create F-distribution (numerator df=3, denominator df=20)
FDistribution fDist = Stat.f(3.0f, 20.0f);

// 计算F统计量的临界值
// Calculate critical values for F-statistic
float alpha = 0.05f;
float fCritical = fDist.ppf(1.0f - alpha);
System.out.println("F-critical (α=0.05, df1=3, df2=20): " + fCritical);

// 计算p值
// Calculate p-value
float fStatistic = 4.5f;
float pValue = 1.0f - fDist.cdf(fStatistic);
System.out.println("p-value for F=" + fStatistic + ": " + pValue);
```

### 方差齐性检验 / Homogeneity of Variance Test

```java
// 两组样本的方差比较
// Compare variances of two groups
FDistribution fDist = Stat.f(9.0f, 9.0f);  // 两组样本大小都是10

float variance1 = 4.0f;  // 第一组方差
float variance2 = 6.0f;  // 第二组方差

// F统计量 = 较大方差 / 较小方差
// F-statistic = larger variance / smaller variance
float fStatistic = Math.max(variance1, variance2) / Math.min(variance1, variance2);
float pValue = 2.0f * (1.0f - fDist.cdf(fStatistic));  // 双侧检验

System.out.println("F-statistic: " + fStatistic);
System.out.println("p-value (two-tailed): " + pValue);

if (pValue < 0.05f) {
    System.out.println("Reject null hypothesis: variances are significantly different");
} else {
    System.out.println("Fail to reject null hypothesis: variances are not significantly different");
}
```

## 综合应用示例 / Comprehensive Application Examples

### 假设检验框架 / Hypothesis Testing Framework

```java
public class HypothesisTest {
    
    public static void performTTest(float[] sample, float populationMean, float alpha) {
        int n = sample.length;
        float sampleMean = calculateMean(sample);
        float sampleStd = calculateStd(sample);
        
        // 计算t统计量
        // Calculate t-statistic
        float tStatistic = (sampleMean - populationMean) / (sampleStd / (float) Math.sqrt(n));
        
        // 创建t分布
        // Create t-distribution
        StudentDistribution tDist = Stat.t(n - 1);
        
        // 计算p值
        // Calculate p-value
        float pValue = 2.0f * (1.0f - tDist.cdf(Math.abs(tStatistic)));
        
        System.out.println("t-statistic: " + tStatistic);
        System.out.println("p-value: " + pValue);
        
        if (pValue < alpha) {
            System.out.println("Reject null hypothesis at α=" + alpha);
        } else {
            System.out.println("Fail to reject null hypothesis at α=" + alpha);
        }
    }
    
    private static float calculateMean(float[] data) {
        float sum = 0.0f;
        for (float value : data) {
            sum += value;
        }
        return sum / data.length;
    }
    
    private static float calculateStd(float[] data) {
        float mean = calculateMean(data);
        float sumSquaredDiff = 0.0f;
        for (float value : data) {
            float diff = value - mean;
            sumSquaredDiff += diff * diff;
        }
        return (float) Math.sqrt(sumSquaredDiff / (data.length - 1));
    }
}
```

### 置信区间计算器 / Confidence Interval Calculator

```java
public class ConfidenceIntervalCalculator {
    
    public static void calculateNormalCI(float[] sample, float confidenceLevel) {
        int n = sample.length;
        float sampleMean = calculateMean(sample);
        float sampleStd = calculateStd(sample);
        
        // 使用正态分布（大样本）
        // Use normal distribution (large sample)
        NormalDistribution normal = Stat.norm(0.0f, 1.0f);
        
        float alpha = 1.0f - confidenceLevel;
        float zScore = normal.ppf(1.0f - alpha/2.0f);
        float marginOfError = zScore * sampleStd / (float) Math.sqrt(n);
        
        float lowerBound = sampleMean - marginOfError;
        float upperBound = sampleMean + marginOfError;
        
        System.out.println(confidenceLevel * 100 + "% Confidence Interval: [" + 
                          lowerBound + ", " + upperBound + "]");
    }
    
    public static void calculateTDistributionCI(float[] sample, float confidenceLevel) {
        int n = sample.length;
        float sampleMean = calculateMean(sample);
        float sampleStd = calculateStd(sample);
        
        // 使用t分布（小样本）
        // Use t-distribution (small sample)
        StudentDistribution tDist = Stat.t(n - 1);
        
        float alpha = 1.0f - confidenceLevel;
        float tValue = tDist.ppf(1.0f - alpha/2.0f);
        float marginOfError = tValue * sampleStd / (float) Math.sqrt(n);
        
        float lowerBound = sampleMean - marginOfError;
        float upperBound = sampleMean + marginOfError;
        
        System.out.println(confidenceLevel * 100 + "% Confidence Interval (t-distribution): [" + 
                          lowerBound + ", " + upperBound + "]");
    }
}
```

### 蒙特卡洛模拟 / Monte Carlo Simulation

```java
public class MonteCarloSimulation {
    
    public static float estimateIntegral(int nSamples) {
        // 估计 ∫₀¹ x² dx = 1/3
        // Estimate ∫₀¹ x² dx = 1/3
        UniformDistribution uniform = Stat.uniform(0.0f, 1.0f);
        
        float sum = 0.0f;
        for (int i = 0; i < nSamples; i++) {
            float x = uniform.sample();
            sum += x * x;  // f(x) = x²
        }
        
        return sum / nSamples;
    }
    
    public static float estimatePi(int nSamples) {
        // 估计π值
        // Estimate π value
        UniformDistribution uniform = Stat.uniform(0.0f, 1.0f);
        
        int insideCircle = 0;
        for (int i = 0; i < nSamples; i++) {
            float x = uniform.sample();
            float y = uniform.sample();
            
            if (x * x + y * y <= 1.0f) {
                insideCircle++;
            }
        }
        
        return 4.0f * insideCircle / nSamples;
    }
    
    public static void main(String[] args) {
        int nSamples = 1000000;
        
        float integralEstimate = estimateIntegral(nSamples);
        System.out.println("Integral estimate: " + integralEstimate);
        System.out.println("True value: " + (1.0f/3.0f));
        System.out.println("Error: " + Math.abs(integralEstimate - 1.0f/3.0f));
        
        float piEstimate = estimatePi(nSamples);
        System.out.println("π estimate: " + piEstimate);
        System.out.println("True π: " + Math.PI);
        System.out.println("Error: " + Math.abs(piEstimate - (float) Math.PI));
    }
}
```

## 性能考虑 / Performance Considerations

### 内存使用 / Memory Usage

```java
// 对于大量随机样本，考虑批量生成
// For large numbers of random samples, consider batch generation
NormalDistribution normal = Stat.norm(0.0f, 1.0f);

// 一次性生成大量样本比多次调用更高效
// Generating large batches is more efficient than multiple calls
float[] samples = normal.sample(1000000);  // 推荐
// vs
// for (int i = 0; i < 1000000; i++) {
//     float sample = normal.sample();  // 不推荐
// }
```

### 精度考虑 / Precision Considerations

```java
// 注意float类型的精度限制
// Be aware of float precision limitations
NormalDistribution normal = Stat.norm(0.0f, 1.0f);

// 对于高精度要求，可能需要使用double类型
// For high precision requirements, consider using double type
float pdfValue = normal.pdf(0.0f);
System.out.println("PDF value: " + pdfValue);
```

## 错误处理 / Error Handling

```java
public class StatErrorHandling {
    
    public static void safeDistributionCreation() {
        try {
            // 尝试创建无效参数的正态分布
            // Try to create normal distribution with invalid parameters
            NormalDistribution normal = Stat.norm(0.0f, -1.0f);  // 负标准差
        } catch (IllegalArgumentException e) {
            System.out.println("Error: " + e.getMessage());
        }
        
        try {
            // 尝试创建无效参数的t分布
            // Try to create t-distribution with invalid parameters
            StudentDistribution tDist = Stat.t(0.0f);  // 零自由度
        } catch (IllegalArgumentException e) {
            System.out.println("Error: " + e.getMessage());
        }
    }
    
    public static void safeProbabilityCalculation() {
        NormalDistribution normal = Stat.norm(0.0f, 1.0f);
        
        try {
            // 尝试计算无效概率的分位数
            // Try to calculate quantile for invalid probability
            float quantile = normal.ppf(1.5f);  // 概率 > 1
        } catch (IllegalArgumentException e) {
            System.out.println("Error: " + e.getMessage());
        }
    }
}
```

## 总结 / Summary

Stat类提供了强大而灵活的统计分布功能，支持六种重要的概率分布。通过本文档的示例，您可以：

The Stat class provides powerful and flexible statistical distribution functionality, supporting six important probability distributions. Through the examples in this document, you can:

1. **创建各种分布对象** - 使用静态工厂方法轻松创建分布实例
2. **进行统计计算** - 计算PDF、CDF、分位数等统计量
3. **执行假设检验** - 使用t分布、卡方分布、F分布进行统计检验
4. **估计置信区间** - 使用正态分布和t分布计算置信区间
5. **进行蒙特卡洛模拟** - 使用各种分布生成随机样本
6. **处理错误情况** - 正确处理参数验证和边界情况

1. **Create various distribution objects** - Easily create distribution instances using static factory methods
2. **Perform statistical calculations** - Calculate PDF, CDF, quantiles, and other statistics
3. **Conduct hypothesis tests** - Use t-distribution, chi-squared, and F-distribution for statistical tests
4. **Estimate confidence intervals** - Calculate confidence intervals using normal and t-distributions
5. **Perform Monte Carlo simulations** - Generate random samples using various distributions
6. **Handle error cases** - Properly handle parameter validation and edge cases

通过合理使用Stat类，您可以构建强大的统计分析应用程序。

By using the Stat class appropriately, you can build powerful statistical analysis applications.
