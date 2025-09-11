# 统计操作示例 (Statistics Examples)

## 概述 / Overview

本文档按照从简单到复杂的顺序，系统性地编排了统计操作包的详细使用示例。每个级别都包含相应的理论背景、实践示例和进阶指导。

This document systematically organizes detailed usage examples for the statistics package in order from simple to complex. Each level includes corresponding theoretical background, practical examples, and advanced guidance.

---

## 第一部分：入门基础 (Level 1 - 基础入门) / Part 1: Beginner Level (Level 1 - Basic Introduction)

### 1.1 环境准备和基本概念 / Environment Setup and Basic Concepts

#### 导入必要的类 / Import Required Classes

```java
import com.reremouse.lab.math.stat.Stat;
import com.reremouse.lab.math.stat.distribution.*;
import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.RereVector;
import java.util.Arrays;
import java.util.List;
```

#### 基本统计量计算 / Basic Statistics Calculation

```java
public class BasicStatisticsExample {
    public static void main(String[] args) {
        // 创建示例数据 / Create sample data
        float[] data = {1.2f, 2.3f, 1.8f, 3.1f, 2.7f, 1.5f, 2.9f, 3.2f, 2.1f, 2.8f};
        IVector vector = IVector.of(data);
        
        // 计算基本统计量 / Calculate basic statistics
        System.out.println("=== 基本统计量 / Basic Statistics ===");
        System.out.println("数据: " + Arrays.toString(data) + " / Data: " + Arrays.toString(data));
        System.out.println("均值: " + vector.mean() + " / Mean: " + vector.mean());
        System.out.println("中位数: " + vector.median() + " / Median: " + vector.median());
        System.out.println("标准差: " + vector.std() + " / Standard deviation: " + vector.std());
        System.out.println("方差: " + vector.var() + " / Variance: " + vector.var());
        System.out.println("最小值: " + vector.min() + " / Minimum: " + vector.min());
        System.out.println("最大值: " + vector.max() + " / Maximum: " + vector.max());
        System.out.println("数据个数: " + vector.length() + " / Count: " + vector.length());
    }
}
```

### 1.2 正态分布基础 / Normal Distribution Basics

```java
public class NormalDistributionBasicExample {
    public static void main(String[] args) {
        // 创建标准正态分布（均值为0，标准差为1） / Create standard normal distribution (mean=0, std=1)
        NormalDistribution standardNormal = Stat.norm();
        System.out.println("标准正态分布: " + standardNormal + " / Standard normal distribution: " + standardNormal);
        
        // 基本统计量 / Basic statistics
        System.out.println("均值: " + standardNormal.mean() + " / Mean: " + standardNormal.mean());
        System.out.println("标准差: " + standardNormal.std() + " / Standard deviation: " + standardNormal.std());
        System.out.println("方差: " + standardNormal.var() + " / Variance: " + standardNormal.var());
        
        // 概率密度函数 / Probability density function
        System.out.println("\n概率密度函数值: / Probability density function values:");
        float[] xValues = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        for (float x : xValues) {
            System.out.printf("PDF(%.1f) = %.4f%n", x, standardNormal.pdf(x));
        }
        
        // 累积分布函数 / Cumulative distribution function
        System.out.println("\n累积分布函数值: / Cumulative distribution function values:");
        for (float x : xValues) {
            System.out.printf("CDF(%.1f) = %.4f%n", x, standardNormal.cdf(x));
        }
    }
}
```

### 1.3 均匀分布基础 / Uniform Distribution Basics

```java
public class UniformDistributionBasicExample {
    public static void main(String[] args) {
        // 创建均匀分布 [0, 1] / Create uniform distribution [0, 1]
        UniformDistribution uniform = Stat.uniform(0.0f, 1.0f);
        System.out.println("均匀分布[0,1]: " + uniform + " / Uniform distribution [0,1]: " + uniform);
        
        // 基本统计量 / Basic statistics
        System.out.println("均值: " + uniform.mean() + " / Mean: " + uniform.mean());
        System.out.println("标准差: " + uniform.std() + " / Standard deviation: " + uniform.std());
        System.out.println("方差: " + uniform.var() + " / Variance: " + uniform.var());
        
        // 概率计算 / Probability calculation
        System.out.println("\n概率计算: / Probability calculation:");
        System.out.println("P(X ≤ 0.5) = " + uniform.cdf(0.5f));
        System.out.println("P(0.3 ≤ X ≤ 0.7) = " + (uniform.cdf(0.7f) - uniform.cdf(0.3f)));
        
        // 分位数 / Quantiles
        System.out.println("\n分位数: / Quantiles:");
        System.out.println("50%分位数: " + uniform.ppf(0.5f) + " / 50th percentile: " + uniform.ppf(0.5f));
        System.out.println("90%分位数: " + uniform.ppf(0.9f) + " / 90th percentile: " + uniform.ppf(0.9f));
    }
}
```

---

## 第二部分：基础应用 (Level 2 - 基础应用) / Part 2: Basic Applications (Level 2 - Basic Applications)

### 2.1 正态分布应用 / Normal Distribution Applications

```java
public class NormalDistributionApplicationExample {
    public static void main(String[] args) {
        // 创建自定义正态分布（均值=10，标准差=2） / Create custom normal distribution (mean=10, std=2)
        NormalDistribution normal = Stat.norm(10.0f, 2.0f);
        System.out.println("正态分布(μ=10, σ=2): " + normal + " / Normal distribution (μ=10, σ=2): " + normal);
        
        // 基本统计量 / Basic statistics
        System.out.println("均值: " + normal.mean() + " / Mean: " + normal.mean());
        System.out.println("标准差: " + normal.std() + " / Standard deviation: " + normal.std());
        System.out.println("方差: " + normal.var() + " / Variance: " + normal.var());
        
        // 概率计算 / Probability calculation
        System.out.println("\n概率计算: / Probability calculation:");
        System.out.println("P(X ≤ 12) = " + normal.cdf(12.0f));
        System.out.println("P(X > 8) = " + normal.sf(8.0f));
        System.out.println("P(8 ≤ X ≤ 12) = " + (normal.cdf(12.0f) - normal.cdf(8.0f)));
        
        // 分位数计算 / Quantile calculation
        System.out.println("\n分位数: / Quantiles:");
        float[] probabilities = {0.01f, 0.05f, 0.1f, 0.25f, 0.5f, 0.75f, 0.9f, 0.95f, 0.99f};
        for (float p : probabilities) {
            System.out.printf("P=%.2f, 分位数=%.4f%n", p, normal.ppf(p));
        }
    }
}
```

### 2.2 随机采样基础 / Random Sampling Basics

```java
public class RandomSamplingBasicExample {
    public static void main(String[] args) {
        // 正态分布采样 / Normal distribution sampling
        NormalDistribution normal = Stat.norm(5.0f, 2.0f);
        
        // 生成单个随机样本 / Generate single random sample
        float sample = normal.sample();
        System.out.println("单个随机样本: " + sample + " / Single random sample: " + sample);
        
        // 生成多个随机样本 / Generate multiple random samples
        float[] samples = normal.sample(1000);
        System.out.println("生成了 " + samples.length + " 个随机样本 / Generated " + samples.length + " random samples");
        
        // 计算样本统计量 / Calculate sample statistics
        IVector sampleVector = IVector.of(samples);
        System.out.println("\n样本统计量: / Sample statistics:");
        System.out.printf("样本均值: %.4f (理论值: %.4f) / Sample mean: %.4f (theoretical: %.4f)%n", sampleVector.mean(), normal.mean(), sampleVector.mean(), normal.mean());
        System.out.printf("样本标准差: %.4f (理论值: %.4f) / Sample std: %.4f (theoretical: %.4f)%n", sampleVector.std(), normal.std(), sampleVector.std(), normal.std());
        System.out.printf("样本方差: %.4f (理论值: %.4f) / Sample variance: %.4f (theoretical: %.4f)%n", sampleVector.var(), normal.var(), sampleVector.var(), normal.var());
    }
}
```

### 2.3 指数分布基础 / Exponential Distribution Basics

```java
public class ExponentialDistributionBasicExample {
    public static void main(String[] args) {
        // 创建指数分布（λ = 0.5） / Create exponential distribution (λ = 0.5)
        ExponentialDistribution exp = Stat.exponential(0.5f);
        System.out.println("指数分布(λ=0.5): " + exp + " / Exponential distribution (λ=0.5): " + exp);
        
        // 基本统计量 / Basic statistics
        System.out.println("均值: " + exp.mean() + " / Mean: " + exp.mean());
        System.out.println("标准差: " + exp.std() + " / Standard deviation: " + exp.std());
        System.out.println("方差: " + exp.var() + " / Variance: " + exp.var());
        
        // 概率计算 / Probability calculation
        System.out.println("\n概率计算: / Probability calculation:");
        float[] times = {1.0f, 2.0f, 3.0f, 5.0f, 10.0f};
        for (float t : times) {
            System.out.printf("P(X ≤ %.1f) = %.4f%n", t, exp.cdf(t));
            System.out.printf("P(X > %.1f) = %.4f%n", t, exp.sf(t));
        }
        
        // 生存函数（可靠性分析） / Survival function (reliability analysis)
        System.out.println("\n生存函数（设备可靠性）: / Survival function (equipment reliability):");
        for (float t : times) {
            System.out.printf("R(%.1f) = P(寿命 > %.1f) = %.4f%n", t, t, exp.sf(t));
        }
    }
}
```

---

## 第三部分：中级应用 (Level 3 - 中级应用) / Part 3: Intermediate Applications (Level 3 - Intermediate Applications)

### 3.1 t分布应用 / t-Distribution Applications

```java
public class TDistributionApplicationExample {
    public static void main(String[] args) {
        // 创建不同自由度的t分布 / Create t-distributions with different degrees of freedom
        StudentDistribution t5 = Stat.t(5.0f);
        StudentDistribution t10 = Stat.t(10.0f);
        StudentDistribution t30 = Stat.t(30.0f);
        
        System.out.println("=== t分布比较 / t-Distribution Comparison ===");
        System.out.println("t分布(5自由度): " + t5 + " / t-distribution (5 df): " + t5);
        System.out.println("t分布(10自由度): " + t10 + " / t-distribution (10 df): " + t10);
        System.out.println("t分布(30自由度): " + t30 + " / t-distribution (30 df): " + t30);
        
        // 统计量比较 / Statistics comparison
        System.out.println("\n统计量比较: / Statistics comparison:");
        System.out.printf("自由度5 - 均值: %.4f, 标准差: %.4f / df=5 - Mean: %.4f, Std: %.4f%n", t5.mean(), t5.std(), t5.mean(), t5.std());
        System.out.printf("自由度10 - 均值: %.4f, 标准差: %.4f / df=10 - Mean: %.4f, Std: %.4f%n", t10.mean(), t10.std(), t10.mean(), t10.std());
        System.out.printf("自由度30 - 均值: %.4f, 标准差: %.4f / df=30 - Mean: %.4f, Std: %.4f%n", t30.mean(), t30.std(), t30.mean(), t30.std());
        
        // 临界值计算（用于假设检验） / Critical value calculation (for hypothesis testing)
        System.out.println("\n临界值计算（95%置信水平）: / Critical value calculation (95% confidence level):");
        float alpha = 0.05f;
        System.out.printf("t(5)临界值: %.4f / t(5) critical value: %.4f%n", t5.ppf(1.0f - alpha/2.0f), t5.ppf(1.0f - alpha/2.0f));
        System.out.printf("t(10)临界值: %.4f / t(10) critical value: %.4f%n", t10.ppf(1.0f - alpha/2.0f), t10.ppf(1.0f - alpha/2.0f));
        System.out.printf("t(30)临界值: %.4f / t(30) critical value: %.4f%n", t30.ppf(1.0f - alpha/2.0f), t30.ppf(1.0f - alpha/2.0f));
    }
}
```

### 3.2 卡方分布应用 / Chi-Squared Distribution Applications

```java
public class ChiSquaredDistributionApplicationExample {
    public static void main(String[] args) {
        // 创建不同自由度的卡方分布 / Create chi-squared distributions with different degrees of freedom
        Chi2Distribution chi2_1 = Stat.chi2(1.0f);
        Chi2Distribution chi2_5 = Stat.chi2(5.0f);
        Chi2Distribution chi2_10 = Stat.chi2(10.0f);
        
        System.out.println("=== 卡方分布比较 / Chi-Squared Distribution Comparison ===");
        System.out.println("卡方分布(1自由度): " + chi2_1 + " / Chi-squared distribution (1 df): " + chi2_1);
        System.out.println("卡方分布(5自由度): " + chi2_5 + " / Chi-squared distribution (5 df): " + chi2_5);
        System.out.println("卡方分布(10自由度): " + chi2_10 + " / Chi-squared distribution (10 df): " + chi2_10);
        
        // 统计量比较 / Statistics comparison
        System.out.println("\n统计量比较: / Statistics comparison:");
        System.out.printf("自由度1 - 均值: %.4f, 标准差: %.4f / df=1 - Mean: %.4f, Std: %.4f%n", chi2_1.mean(), chi2_1.std(), chi2_1.mean(), chi2_1.std());
        System.out.printf("自由度5 - 均值: %.4f, 标准差: %.4f / df=5 - Mean: %.4f, Std: %.4f%n", chi2_5.mean(), chi2_5.std(), chi2_5.mean(), chi2_5.std());
        System.out.printf("自由度10 - 均值: %.4f, 标准差: %.4f / df=10 - Mean: %.4f, Std: %.4f%n", chi2_10.mean(), chi2_10.std(), chi2_10.mean(), chi2_10.std());
        
        // 临界值计算 / Critical value calculation
        System.out.println("\n临界值计算（95%置信水平）: / Critical value calculation (95% confidence level):");
        float alpha = 0.05f;
        System.out.printf("χ²(1)临界值: %.4f / χ²(1) critical value: %.4f%n", chi2_1.ppf(1.0f - alpha), chi2_1.ppf(1.0f - alpha));
        System.out.printf("χ²(5)临界值: %.4f / χ²(5) critical value: %.4f%n", chi2_5.ppf(1.0f - alpha), chi2_5.ppf(1.0f - alpha));
        System.out.printf("χ²(10)临界值: %.4f / χ²(10) critical value: %.4f%n", chi2_10.ppf(1.0f - alpha), chi2_10.ppf(1.0f - alpha));
    }
}
```

### 3.3 F分布应用 / F-Distribution Applications

```java
public class FDistributionApplicationExample {
    public static void main(String[] args) {
        // 创建不同自由度的F分布 / Create F-distributions with different degrees of freedom
        FDistribution f1 = Stat.f(5.0f, 10.0f);
        FDistribution f2 = Stat.f(10.0f, 5.0f);
        FDistribution f3 = Stat.f(20.0f, 20.0f);
        
        System.out.println("=== F分布比较 / F-Distribution Comparison ===");
        System.out.println("F分布(5,10自由度): " + f1 + " / F-distribution (5,10 df): " + f1);
        System.out.println("F分布(10,5自由度): " + f2 + " / F-distribution (10,5 df): " + f2);
        System.out.println("F分布(20,20自由度): " + f3 + " / F-distribution (20,20 df): " + f3);
        
        // 统计量比较 / Statistics comparison
        System.out.println("\n统计量比较: / Statistics comparison:");
        System.out.printf("F(5,10) - 均值: %.4f / F(5,10) - Mean: %.4f%n", f1.mean(), f1.mean());
        System.out.printf("F(10,5) - 均值: %.4f / F(10,5) - Mean: %.4f%n", f2.mean(), f2.mean());
        System.out.printf("F(20,20) - 均值: %.4f / F(20,20) - Mean: %.4f%n", f3.mean(), f3.mean());
        
        // 临界值计算 / Critical value calculation
        System.out.println("\n临界值计算（95%置信水平）: / Critical value calculation (95% confidence level):");
        float alpha = 0.05f;
        System.out.printf("F(5,10)临界值: %.4f / F(5,10) critical value: %.4f%n", f1.ppf(1.0f - alpha), f1.ppf(1.0f - alpha));
        System.out.printf("F(10,5)临界值: %.4f / F(10,5) critical value: %.4f%n", f2.ppf(1.0f - alpha), f2.ppf(1.0f - alpha));
        System.out.printf("F(20,20)临界值: %.4f / F(20,20) critical value: %.4f%n", f3.ppf(1.0f - alpha), f3.ppf(1.0f - alpha));
    }
}
```

### 3.4 泊松分布应用 / Poisson Distribution Applications

```java
public class PoissonDistributionApplicationExample {
    public static void main(String[] args) {
        // 创建泊松分布（λ = 3） / Create Poisson distribution (λ = 3)
        PoissonDistribution poisson = Stat.poisson(3.0f);
        System.out.println("泊松分布(λ=3): " + poisson + " / Poisson distribution (λ=3): " + poisson);
        
        // 基本统计量 / Basic statistics
        System.out.println("均值: " + poisson.mean() + " / Mean: " + poisson.mean());
        System.out.println("方差: " + poisson.var() + " / Variance: " + poisson.var());
        System.out.println("标准差: " + poisson.std() + " / Standard deviation: " + poisson.std());
        
        // 概率质量函数 / Probability mass function
        System.out.println("\n概率质量函数: / Probability mass function:");
        for (int k = 0; k <= 10; k++) {
            System.out.printf("P(X = %d) = %.4f%n", k, poisson.pmf(k));
        }
        
        // 累积分布函数 / Cumulative distribution function
        System.out.println("\n累积分布函数: / Cumulative distribution function:");
        for (int k = 0; k <= 10; k++) {
            System.out.printf("P(X ≤ %d) = %.4f%n", k, poisson.cdf(k));
        }
        
        // 随机采样 / Random sampling
        System.out.println("\n随机采样: / Random sampling:");
        int[] samples = new int[1000];
        for (int i = 0; i < 1000; i++) {
            samples[i] = (int) poisson.sample();
        }
        
        // 计算样本统计量 / Calculate sample statistics
        float sampleMean = 0;
        for (int sample : samples) {
            sampleMean += sample;
        }
        sampleMean /= samples.length;
        
        System.out.printf("样本均值: %.4f (理论值: %.4f) / Sample mean: %.4f (theoretical: %.4f)%n", sampleMean, poisson.mean(), sampleMean, poisson.mean());
    }
}
```

---

## 第四部分：高级应用 (Level 4 - 高级应用) / Part 4: Advanced Applications (Level 4 - Advanced Applications)

### 4.1 参数估计 / Parameter Estimation

```java
public class ParameterEstimationExample {
    public static void main(String[] args) {
        // 生成样本数据 / Generate sample data
        NormalDistribution trueDist = Stat.norm(10.0f, 2.0f);
        float[] samples = trueDist.sample(100);
        IVector sampleVector = IVector.of(samples);
        
        System.out.println("=== 参数估计示例 / Parameter Estimation Example ===");
        System.out.println("真实参数: μ=10.0, σ=2.0 / True parameters: μ=10.0, σ=2.0");
        System.out.println("样本大小: " + sampleVector.length() + " / Sample size: " + sampleVector.length());
        System.out.println("样本均值: " + sampleVector.mean() + " / Sample mean: " + sampleVector.mean());
        System.out.println("样本标准差: " + sampleVector.std() + " / Sample standard deviation: " + sampleVector.std());
        
        // 参数估计 / Parameter estimation
        ParameterEstimation estimator = new ParameterEstimation();
        
        // 均值置信区间估计（使用t分布） / Mean confidence interval estimation (using t-distribution)
        Tuple2<Float, Float> meanCI = estimator.estimateMeanIntevalWithT(sampleVector, 0.95f);
        System.out.println("\n均值95%置信区间: [" + meanCI._1 + ", " + meanCI._2 + "] / Mean 95% confidence interval: [" + meanCI._1 + ", " + meanCI._2 + "]");
        
        // 方差置信区间估计（使用卡方分布） / Variance confidence interval estimation (using chi-squared distribution)
        Tuple2<Float, Float> varCI = estimator.estimateVarIntevalWithChi2(sampleVector, 0.95f);
        System.out.println("方差95%置信区间: [" + varCI._1 + ", " + varCI._2 + "] / Variance 95% confidence interval: [" + varCI._1 + ", " + varCI._2 + "]");
        
        // 检查真实参数是否在置信区间内 / Check if true parameters are within confidence intervals
        System.out.println("\n参数检验: / Parameter validation:");
        System.out.println("真实均值在置信区间内: " + (meanCI._1 <= 10.0f && 10.0f <= meanCI._2) + " / True mean in confidence interval: " + (meanCI._1 <= 10.0f && 10.0f <= meanCI._2));
        System.out.println("真实方差在置信区间内: " + (varCI._1 <= 4.0f && 4.0f <= varCI._2) + " / True variance in confidence interval: " + (varCI._1 <= 4.0f && 4.0f <= varCI._2));
    }
}
```

### 4.2 假设检验 / Hypothesis Testing

```java
public class HypothesisTestingExample {
    public static void main(String[] args) {
        // 生成样本数据 / Generate sample data
        NormalDistribution trueDist = Stat.norm(10.0f, 2.0f);
        float[] samples = trueDist.sample(50);
        IVector sampleVector = IVector.of(samples);
        
        System.out.println("=== 假设检验示例 / Hypothesis Testing Example ===");
        System.out.println("样本数据: / Sample data:");
        System.out.println("  样本均值: " + sampleVector.mean() + " / Sample mean: " + sampleVector.mean());
        System.out.println("  样本标准差: " + sampleVector.std() + " / Sample standard deviation: " + sampleVector.std());
        
        // 假设检验 / Hypothesis testing
        HypothesisTesting tester = new HypothesisTesting();
        
        // 检验均值是否等于10.0 / Test if mean equals 10.0
        TestingResult meanTest = tester.testMeanEqualWithT(10.0f, sampleVector, 0.95f);
        System.out.println("\n均值检验 (H0: μ = 10.0): / Mean test (H0: μ = 10.0):");
        System.out.println("  检验结果: " + (meanTest.pass ? "接受原假设" : "拒绝原假设") + " / Test result: " + (meanTest.pass ? "Accept null hypothesis" : "Reject null hypothesis"));
        System.out.println("  p值: " + meanTest.p + " / p-value: " + meanTest.p);
        System.out.println("  置信区间: [" + meanTest.criticalInteval._1 + ", " + meanTest.criticalInteval._2 + "] / Confidence interval: [" + meanTest.criticalInteval._1 + ", " + meanTest.criticalInteval._2 + "]");
        
        // 检验方差是否等于4.0 / Test if variance equals 4.0
        TestingResult varTest = tester.testVarEqualWithChi2(4.0f, sampleVector, 0.95f);
        System.out.println("\n方差检验 (H0: σ² = 4.0): / Variance test (H0: σ² = 4.0):");
        System.out.println("  检验结果: " + (varTest.pass ? "接受原假设" : "拒绝原假设") + " / Test result: " + (varTest.pass ? "Accept null hypothesis" : "Reject null hypothesis"));
        System.out.println("  p值: " + varTest.p + " / p-value: " + varTest.p);
        System.out.println("  置信区间: [" + varTest.criticalInteval._1 + ", " + varTest.criticalInteval._2 + "] / Confidence interval: [" + varTest.criticalInteval._1 + ", " + varTest.criticalInteval._2 + "]");
    }
}
```

### 4.3 二项分布应用 / Binomial Distribution Applications

```java
public class BinomialDistributionApplicationExample {
    public static void main(String[] args) {
        // 创建二项分布（n=100, p=0.3） / Create binomial distribution (n=100, p=0.3)
        BinomialDistribution binomial = Stat.binomial(100, 0.3f);
        System.out.println("二项分布(n=100, p=0.3): " + binomial + " / Binomial distribution (n=100, p=0.3): " + binomial);
        
        // 基本统计量 / Basic statistics
        System.out.println("均值: " + binomial.mean() + " / Mean: " + binomial.mean());
        System.out.println("方差: " + binomial.var() + " / Variance: " + binomial.var());
        System.out.println("标准差: " + binomial.std() + " / Standard deviation: " + binomial.std());
        
        // 概率质量函数 / Probability mass function
        System.out.println("\n概率质量函数: / Probability mass function:");
        for (int k = 20; k <= 40; k += 5) {
            System.out.printf("P(X = %d) = %.4f%n", k, binomial.pmf(k));
        }
        
        // 累积分布函数 / Cumulative distribution function
        System.out.println("\n累积分布函数: / Cumulative distribution function:");
        for (int k = 20; k <= 40; k += 5) {
            System.out.printf("P(X ≤ %d) = %.4f%n", k, binomial.cdf(k));
        }
        
        // 随机采样 / Random sampling
        System.out.println("\n随机采样: / Random sampling:");
        int[] samples = new int[1000];
        for (int i = 0; i < 1000; i++) {
            samples[i] = (int) binomial.sample();
        }
        
        // 计算样本统计量 / Calculate sample statistics
        float sampleMean = 0;
        for (int sample : samples) {
            sampleMean += sample;
        }
        sampleMean /= samples.length;
        
        System.out.printf("样本均值: %.4f (理论值: %.4f) / Sample mean: %.4f (theoretical: %.4f)%n", sampleMean, binomial.mean(), sampleMean, binomial.mean());
    }
}
```

---

## 第五部分：专业应用 (Level 5 - 专业应用) / Part 5: Professional Applications (Level 5 - Professional Applications)

### 5.1 质量控制分析 / Quality Control Analysis

```java
public class QualityControlAnalysisExample {
    public static void main(String[] args) {
        System.out.println("=== 质量控制统计分析示例 / Quality Control Statistical Analysis Example ===");
        
        // 模拟生产过程中的产品重量数据 / Simulate product weight data in production process
        // 假设产品重量应该服从正态分布，均值为100g，标准差为2g / Assume product weight follows normal distribution with mean=100g, std=2g
        NormalDistribution targetDist = Stat.norm(100.0f, 2.0f);
        
        // 生成样本数据（模拟实际测量结果） / Generate sample data (simulate actual measurement results)
        float[] measurements = targetDist.sample(50);
        IVector sample = IVector.of(measurements);
        
        System.out.println("产品重量测量数据: / Product weight measurement data:");
        System.out.println("  样本大小: " + sample.length() + " / Sample size: " + sample.length());
        System.out.println("  样本均值: " + sample.mean() + " / Sample mean: " + sample.mean());
        System.out.println("  样本标准差: " + sample.std() + " / Sample standard deviation: " + sample.std());
        System.out.println("  最小值: " + sample.min() + " / Minimum: " + sample.min());
        System.out.println("  最大值: " + sample.max() + " / Maximum: " + sample.max());
        
        // 参数估计 / Parameter estimation
        ParameterEstimation estimator = new ParameterEstimation();
        Tuple2<Float, Float> meanCI = estimator.estimateMeanIntevalWithT(sample, 0.95f);
        Tuple2<Float, Float> varCI = estimator.estimateVarIntevalWithChi2(sample, 0.95f);
        
        System.out.println("\n参数估计结果: / Parameter estimation results:");
        System.out.println("  均值95%置信区间: [" + meanCI._1 + ", " + meanCI._2 + "] / Mean 95% confidence interval: [" + meanCI._1 + ", " + meanCI._2 + "]");
        System.out.println("  方差95%置信区间: [" + varCI._1 + ", " + varCI._2 + "] / Variance 95% confidence interval: [" + varCI._1 + ", " + varCI._2 + "]");
        
        // 假设检验 / Hypothesis testing
        HypothesisTesting tester = new HypothesisTesting();
        
        // 检验均值是否等于目标值100g / Test if mean equals target value 100g
        TestingResult meanTest = tester.testMeanEqualWithT(100.0f, sample, 0.95f);
        System.out.println("\n均值检验 (H0: μ = 100g): / Mean test (H0: μ = 100g):");
        System.out.println("  检验结果: " + (meanTest.pass ? "✓ 生产过程正常" : "✗ 生产过程异常") + " / Test result: " + (meanTest.pass ? "✓ Production process normal" : "✗ Production process abnormal"));
        System.out.println("  p值: " + meanTest.p + " / p-value: " + meanTest.p);
        
        // 检验方差是否等于目标值4.0 / Test if variance equals target value 4.0
        TestingResult varTest = tester.testVarEqualWithChi2(4.0f, sample, 0.95f);
        System.out.println("\n方差检验 (H0: σ² = 4.0): / Variance test (H0: σ² = 4.0):");
        System.out.println("  检验结果: " + (varTest.pass ? "✓ 过程稳定性正常" : "✗ 过程稳定性异常") + " / Test result: " + (varTest.pass ? "✓ Process stability normal" : "✗ Process stability abnormal"));
        System.out.println("  p值: " + varTest.p + " / p-value: " + varTest.p);
        
        // 计算不合格品概率 / Calculate defective product probability
        NormalDistribution estimatedDist = Stat.norm(sample.mean(), sample.std());
        float lowerLimit = 95.0f;  // 下限
        float upperLimit = 105.0f; // 上限
        
        float probBelowLower = estimatedDist.cdf(lowerLimit);
        float probAboveUpper = 1.0f - estimatedDist.cdf(upperLimit);
        float defectProb = probBelowLower + probAboveUpper;
        
        System.out.println("\n质量分析: / Quality analysis:");
        System.out.println("  不合格品概率: " + (defectProb * 100) + "% / Defect probability: " + (defectProb * 100) + "%");
        System.out.println("  低于下限概率: " + (probBelowLower * 100) + "% / Below lower limit probability: " + (probBelowLower * 100) + "%");
        System.out.println("  高于上限概率: " + (probAboveUpper * 100) + "% / Above upper limit probability: " + (probAboveUpper * 100) + "%");
    }
}
```

### 5.2 金融风险评估 / Financial Risk Assessment

```java
public class FinancialRiskAssessmentExample {
    public static void main(String[] args) {
        System.out.println("=== 金融风险评估示例 / Financial Risk Assessment Example ===");
        
        // 模拟股票收益率数据（假设服从正态分布） / Simulate stock return data (assume normal distribution)
        NormalDistribution returnDist = Stat.norm(0.001f, 0.02f); // 日收益率：均值0.1%，标准差2% / Daily return: mean 0.1%, std 2%
        float[] dailyReturns = returnDist.sample(252); // 一年的交易日数据 / One year of trading day data
        IVector returns = IVector.of(dailyReturns);
        
        System.out.println("股票收益率统计: / Stock return statistics:");
        System.out.println("  样本大小: " + returns.length() + " / Sample size: " + returns.length());
        System.out.println("  平均日收益率: " + (returns.mean() * 100) + "% / Average daily return: " + (returns.mean() * 100) + "%");
        System.out.println("  收益率标准差: " + (returns.std() * 100) + "% / Return standard deviation: " + (returns.std() * 100) + "%");
        System.out.println("  最小日收益率: " + (returns.min() * 100) + "% / Minimum daily return: " + (returns.min() * 100) + "%");
        System.out.println("  最大日收益率: " + (returns.max() * 100) + "% / Maximum daily return: " + (returns.max() * 100) + "%");
        
        // 计算VaR (Value at Risk) / Calculate VaR (Value at Risk)
        float[] sortedReturns = dailyReturns.clone();
        Arrays.sort(sortedReturns);
        
        int var95Index = (int) (sortedReturns.length * 0.05);
        int var99Index = (int) (sortedReturns.length * 0.01);
        
        float var95 = sortedReturns[var95Index];
        float var99 = sortedReturns[var99Index];
        
        System.out.println("\n风险度量: / Risk measures:");
        System.out.println("  VaR 95%: " + (var95 * 100) + "% / VaR 95%: " + (var95 * 100) + "%");
        System.out.println("  VaR 99%: " + (var99 * 100) + "% / VaR 99%: " + (var99 * 100) + "%");
        
        // 使用正态分布计算理论VaR / Calculate theoretical VaR using normal distribution
        NormalDistribution estimatedDist = Stat.norm(returns.mean(), returns.std());
        float theoreticalVar95 = estimatedDist.ppf(0.05f);
        float theoreticalVar99 = estimatedDist.ppf(0.01f);
        
        System.out.println("\n理论VaR (基于正态分布): / Theoretical VaR (based on normal distribution):");
        System.out.println("  VaR 95%: " + (theoreticalVar95 * 100) + "% / VaR 95%: " + (theoreticalVar95 * 100) + "%");
        System.out.println("  VaR 99%: " + (theoreticalVar99 * 100) + "% / VaR 99%: " + (theoreticalVar99 * 100) + "%");
        
        // 计算最大回撤 / Calculate maximum drawdown
        float maxDrawdown = calculateMaxDrawdown(dailyReturns);
        System.out.println("\n最大回撤: " + (maxDrawdown * 100) + "% / Maximum drawdown: " + (maxDrawdown * 100) + "%");
        
        // 计算夏普比率 / Calculate Sharpe ratio
        float riskFreeRate = 0.03f / 252; // 年化无风险利率3%，转换为日利率 / Annual risk-free rate 3%, converted to daily rate
        float sharpeRatio = (returns.mean() - riskFreeRate) / returns.std();
        System.out.println("夏普比率: " + sharpeRatio + " / Sharpe ratio: " + sharpeRatio);
    }
    
    private static float calculateMaxDrawdown(float[] returns) {
        float maxDrawdown = 0.0f;
        float peak = 0.0f;
        float cumulative = 0.0f;
        
        for (float ret : returns) {
            cumulative += ret;
            if (cumulative > peak) {
                peak = cumulative;
            }
            float drawdown = peak - cumulative;
            if (drawdown > maxDrawdown) {
                maxDrawdown = drawdown;
            }
        }
        
        return maxDrawdown;
    }
}
```

### 5.3 可靠性分析 / Reliability Analysis

```java
public class ReliabilityAnalysisExample {
    public static void main(String[] args) {
        System.out.println("=== 可靠性分析示例 / Reliability Analysis Example ===");
        
        // 模拟设备故障时间数据（假设服从指数分布） / Simulate equipment failure time data (assume exponential distribution)
        ExponentialDistribution failureDist = Stat.exponential(0.01f); // 故障率λ=0.01/小时 / Failure rate λ=0.01/hour
        float[] failureTimes = failureDist.sample(100); // 100个故障时间样本 / 100 failure time samples
        IVector failures = IVector.of(failureTimes);
        
        System.out.println("设备故障时间统计: / Equipment failure time statistics:");
        System.out.println("  样本大小: " + failures.length() + " / Sample size: " + failures.length());
        System.out.println("  平均故障时间: " + failures.mean() + " 小时 / Average failure time: " + failures.mean() + " hours");
        System.out.println("  故障时间标准差: " + failures.std() + " 小时 / Failure time std dev: " + failures.std() + " hours");
        System.out.println("  最短故障时间: " + failures.min() + " 小时 / Shortest failure time: " + failures.min() + " hours");
        System.out.println("  最长故障时间: " + failures.max() + " 小时 / Longest failure time: " + failures.max() + " hours");
        
        // 计算可靠性指标 / Calculate reliability indicators
        float estimatedLambda = 1.0f / failures.mean();
        System.out.println("\n可靠性指标: / Reliability indicators:");
        System.out.println("  估计故障率: " + estimatedLambda + " /小时 / Estimated failure rate: " + estimatedLambda + " /hour");
        System.out.println("  平均故障间隔时间(MTBF): " + failures.mean() + " 小时 / Mean Time Between Failures (MTBF): " + failures.mean() + " hours");
        
        // 计算不同时间点的可靠性 / Calculate reliability at different time points
        float[] timePoints = {10, 50, 100, 200, 500};
        System.out.println("\n可靠性函数 R(t) = e^(-λt): / Reliability function R(t) = e^(-λt):");
        for (float t : timePoints) {
            float reliability = (float) Math.exp(-estimatedLambda * t);
            System.out.println("  R(" + t + ") = " + reliability);
        }
        
        // 计算故障概率密度 / Calculate failure probability density
        System.out.println("\n故障概率密度 f(t) = λe^(-λt): / Failure probability density f(t) = λe^(-λt):");
        for (float t : timePoints) {
            float density = estimatedLambda * (float) Math.exp(-estimatedLambda * t);
            System.out.println("  f(" + t + ") = " + density);
        }
        
        // 计算累积故障概率 / Calculate cumulative failure probability
        System.out.println("\n累积故障概率 F(t) = 1 - e^(-λt): / Cumulative failure probability F(t) = 1 - e^(-λt):");
        for (float t : timePoints) {
            float cumulativeProb = 1.0f - (float) Math.exp(-estimatedLambda * t);
            System.out.println("  F(" + t + ") = " + cumulativeProb);
        }
        
        // 计算中位故障时间 / Calculate median failure time
        float medianTime = (float) (Math.log(2) / estimatedLambda);
        System.out.println("\n中位故障时间: " + medianTime + " 小时 / Median failure time: " + medianTime + " hours");
    }
}
```

### 5.4 实验设计分析 / Experimental Design Analysis

```java
public class ExperimentalDesignAnalysisExample {
    public static void main(String[] args) {
        System.out.println("=== 实验设计分析示例 / Experimental Design Analysis Example ===");
        
        // 模拟A/B测试数据 / Simulate A/B test data
        // 对照组：转化率5% / Control group: 5% conversion rate
        BinomialDistribution controlDist = Stat.binomial(1000, 0.05f);
        int controlConversions = (int) controlDist.sample();
        
        // 实验组：转化率6% / Treatment group: 6% conversion rate
        BinomialDistribution treatmentDist = Stat.binomial(1000, 0.06f);
        int treatmentConversions = (int) treatmentDist.sample();
        
        System.out.println("A/B测试结果: / A/B test results:");
        System.out.println("  对照组转化数: " + controlConversions + "/1000 / Control group conversions: " + controlConversions + "/1000");
        System.out.println("  实验组转化数: " + treatmentConversions + "/1000 / Treatment group conversions: " + treatmentConversions + "/1000");
        System.out.println("  对照组转化率: " + (controlConversions / 1000.0f * 100) + "% / Control group conversion rate: " + (controlConversions / 1000.0f * 100) + "%");
        System.out.println("  实验组转化率: " + (treatmentConversions / 1000.0f * 100) + "% / Treatment group conversion rate: " + (treatmentConversions / 1000.0f * 100) + "%");
        
        // 计算转化率差异 / Calculate conversion rate difference
        float controlRate = controlConversions / 1000.0f;
        float treatmentRate = treatmentConversions / 1000.0f;
        float rateDifference = treatmentRate - controlRate;
        
        System.out.println("\n转化率差异: " + (rateDifference * 100) + "% / Conversion rate difference: " + (rateDifference * 100) + "%");
        
        // 计算置信区间 / Calculate confidence interval
        float controlStdError = (float) Math.sqrt(controlRate * (1 - controlRate) / 1000);
        float treatmentStdError = (float) Math.sqrt(treatmentRate * (1 - treatmentRate) / 1000);
        float diffStdError = (float) Math.sqrt(controlStdError * controlStdError + treatmentStdError * treatmentStdError);
        
        // 使用正态分布近似 / Use normal distribution approximation
        NormalDistribution diffDist = Stat.norm(rateDifference, diffStdError);
        float lowerBound = diffDist.ppf(0.025f);
        float upperBound = diffDist.ppf(0.975f);
        
        System.out.println("\n转化率差异95%置信区间: / Conversion rate difference 95% confidence interval:");
        System.out.println("  [" + (lowerBound * 100) + "%, " + (upperBound * 100) + "%] / [" + (lowerBound * 100) + "%, " + (upperBound * 100) + "%]");
        
        // 假设检验 / Hypothesis testing
        HypothesisTesting tester = new HypothesisTesting();
        
        // 创建样本数据用于检验 / Create sample data for testing
        float[] controlData = new float[1000];
        float[] treatmentData = new float[1000];
        
        for (int i = 0; i < 1000; i++) {
            controlData[i] = controlDist.sample();
            treatmentData[i] = treatmentDist.sample();
        }
        
        IVector controlVector = IVector.of(controlData);
        IVector treatmentVector = IVector.of(treatmentData);
        
        // 检验两组均值是否相等 / Test if two group means are equal
        TestingResult meanTest = tester.testMeanEqualWithT(0.0f, 
            IVector.of(new float[]{rateDifference}), 0.95f);
        
        System.out.println("\n假设检验 (H0: 转化率差异 = 0): / Hypothesis test (H0: conversion rate difference = 0):");
        System.out.println("  检验结果: " + (meanTest.pass ? "接受原假设" : "拒绝原假设") + " / Test result: " + (meanTest.pass ? "Accept null hypothesis" : "Reject null hypothesis"));
        System.out.println("  p值: " + meanTest.p + " / p-value: " + meanTest.p);
        
        // 计算统计功效 / Calculate statistical power
        float effectSize = Math.abs(rateDifference) / diffStdError;
        System.out.println("\n效应量 (Cohen's d): " + effectSize + " / Effect size (Cohen's d): " + effectSize);
        
        if (effectSize < 0.2) {
            System.out.println("  效应量: 小 / Effect size: Small");
        } else if (effectSize < 0.5) {
            System.out.println("  效应量: 中等 / Effect size: Medium");
        } else {
            System.out.println("  效应量: 大 / Effect size: Large");
        }
    }
}
```

### 5.5 方差分析应用 / Analysis of Variance (ANOVA) Applications

```java
public class ANOVAApplicationExample {
    public static void main(String[] args) {
        System.out.println("=== 方差分析应用示例 / Analysis of Variance (ANOVA) Application Example ===");
        
        // 单因素方差分析示例 / One-way ANOVA example
        demonstrateOneWayANOVA();
        
        // 两因素方差分析示例 / Two-way ANOVA example
        demonstrateTwoWayANOVA();
        
        // 重复测量方差分析示例 / Repeated measures ANOVA example
        demonstrateRepeatedMeasuresANOVA();
        
        // 假设检验示例 / Assumption tests example
        demonstrateAssumptionTests();
    }
    
    public static void demonstrateOneWayANOVA() {
        System.out.println("\n--- 单因素方差分析示例 / One-Way ANOVA Example ---");
        
        // 创建三个组的数据 / Create data for three groups
        // 假设研究不同教学方法对学生成绩的影响 / Assume studying the impact of different teaching methods on student performance
        IVector traditionalMethod = IVector.of(new float[]{75, 78, 82, 85, 88, 90, 92, 95});
        IVector onlineMethod = IVector.of(new float[]{70, 73, 76, 79, 82, 85, 87, 90});
        IVector hybridMethod = IVector.of(new float[]{80, 83, 86, 89, 92, 95, 97, 100});
        
        System.out.println("教学方法数据: / Teaching method data:");
        System.out.println("  传统方法: " + traditionalMethod.mean() + " ± " + traditionalMethod.std() + " / Traditional method: " + traditionalMethod.mean() + " ± " + traditionalMethod.std());
        System.out.println("  在线方法: " + onlineMethod.mean() + " ± " + onlineMethod.std() + " / Online method: " + onlineMethod.mean() + " ± " + onlineMethod.std());
        System.out.println("  混合方法: " + hybridMethod.mean() + " ± " + hybridMethod.std() + " / Hybrid method: " + hybridMethod.mean() + " ± " + hybridMethod.std());
        
        // 执行单因素方差分析 / Perform one-way ANOVA
        ANOVAResult result = ANOVA.performOneWayANOVA(traditionalMethod, onlineMethod, hybridMethod);
        
        System.out.println("\n单因素方差分析结果: / One-way ANOVA results:");
        System.out.println("  F统计量: " + result.fStatistic + " / F-statistic: " + result.fStatistic);
        System.out.println("  p值: " + result.pValue + " / p-value: " + result.pValue);
        System.out.println("  组间平方和: " + result.ssBetween + " / Sum of squares between groups: " + result.ssBetween);
        System.out.println("  组内平方和: " + result.ssWithin + " / Sum of squares within groups: " + result.ssWithin);
        System.out.println("  总平方和: " + result.ssTotal + " / Total sum of squares: " + result.ssTotal);
        
        // 解释结果 / Interpret results
        if (result.pValue < 0.05) {
            System.out.println("  结论: 不同教学方法对学生成绩有显著影响 (p < 0.05) / Conclusion: Different teaching methods have significant effect on student performance (p < 0.05)");
        } else {
            System.out.println("  结论: 不同教学方法对学生成绩无显著影响 (p ≥ 0.05) / Conclusion: Different teaching methods have no significant effect on student performance (p ≥ 0.05)");
        }
        
        // 执行Tukey HSD多重比较 / Perform Tukey HSD multiple comparisons
        ANOVA.performTukeyHSD(traditionalMethod, onlineMethod, hybridMethod);
    }
    
    public static void demonstrateTwoWayANOVA() {
        System.out.println("\n--- 两因素方差分析示例 / Two-Way ANOVA Example ---");
        
        // 创建两因素数据 [教学方法][学习风格] / Create two-factor data [teaching method][learning style]
        // 因素A: 教学方法 (传统 vs 在线) / Factor A: Teaching method (Traditional vs Online)
        // 因素B: 学习风格 (视觉型 vs 听觉型) / Factor B: Learning style (Visual vs Auditory)
        float[][][] data = {
            // 传统方法 / Traditional method
            {{85, 87, 89, 91}, {78, 80, 82, 84}},  // 视觉型, 听觉型 / Visual, Auditory
            // 在线方法 / Online method
            {{88, 90, 92, 94}, {81, 83, 85, 87}}   // 视觉型, 听觉型 / Visual, Auditory
        };
        
        System.out.println("两因素实验设计: / Two-factor experimental design:");
        System.out.println("  因素A: 教学方法 (传统 vs 在线) / Factor A: Teaching method (Traditional vs Online)");
        System.out.println("  因素B: 学习风格 (视觉型 vs 听觉型) / Factor B: Learning style (Visual vs Auditory)");
        
        // 执行两因素方差分析 / Perform two-way ANOVA
        TwoWayANOVAResult result = ANOVA.performTwoWayANOVA(data);
        
        System.out.println("\n两因素方差分析结果: / Two-way ANOVA results:");
        System.out.println("  教学方法主效应: F=" + result.factorAF + ", p=" + result.factorAP + " / Teaching method main effect: F=" + result.factorAF + ", p=" + result.factorAP);
        System.out.println("  学习风格主效应: F=" + result.factorBF + ", p=" + result.factorBP + " / Learning style main effect: F=" + result.factorBF + ", p=" + result.factorBP);
        System.out.println("  交互效应: F=" + result.interactionF + ", p=" + result.interactionP + " / Interaction effect: F=" + result.interactionF + ", p=" + result.interactionP);
        
        // 解释结果 / Interpret results
        System.out.println("\n结果解释: / Result interpretation:");
        if (result.factorAP < 0.05) {
            System.out.println("  教学方法有显著主效应 / Teaching method has significant main effect");
        }
        if (result.factorBP < 0.05) {
            System.out.println("  学习风格有显著主效应 / Learning style has significant main effect");
        }
        if (result.interactionP < 0.05) {
            System.out.println("  教学方法与学习风格存在显著交互效应 / Significant interaction between teaching method and learning style");
        }
    }
    
    public static void demonstrateRepeatedMeasuresANOVA() {
        System.out.println("\n--- 重复测量方差分析示例 / Repeated Measures ANOVA Example ---");
        
        // 创建重复测量数据 [被试][时间点] / Create repeated measures data [subjects][time points]
        // 研究训练前后和训练后的技能水平变化 / Study skill level changes before, during, and after training
        float[][] data = {
            {60, 65, 70, 75},  // 被试1: 训练前, 训练中, 训练后, 随访 / Subject 1: Pre-training, During training, Post-training, Follow-up
            {55, 62, 68, 72},  // 被试2 / Subject 2
            {58, 64, 69, 74},  // 被试3 / Subject 3
            {62, 67, 72, 77},  // 被试4 / Subject 4
            {57, 63, 68, 73}   // 被试5 / Subject 5
        };
        
        System.out.println("重复测量实验设计: / Repeated measures experimental design:");
        System.out.println("  被试数: " + data.length + " / Number of subjects: " + data.length);
        System.out.println("  测量时间点: " + data[0].length + " / Number of time points: " + data[0].length);
        System.out.println("  时间点: 训练前, 训练中, 训练后, 随访 / Time points: Pre-training, During training, Post-training, Follow-up");
        
        // 执行重复测量方差分析 / Perform repeated measures ANOVA
        RepeatedMeasuresANOVAResult result = ANOVA.performRepeatedMeasuresANOVA(data);
        
        System.out.println("\n重复测量方差分析结果: / Repeated measures ANOVA results:");
        System.out.println("  时间效应: F=" + result.timeF + ", p=" + result.timeP + " / Time effect: F=" + result.timeF + ", p=" + result.timeP);
        System.out.println("  被试效应: F=" + result.subjectF + ", p=" + result.subjectP + " / Subject effect: F=" + result.subjectF + ", p=" + result.subjectP);
        
        // 解释结果 / Interpret results
        System.out.println("\n结果解释: / Result interpretation:");
        if (result.timeP < 0.05) {
            System.out.println("  时间因素有显著效应，技能水平随时间显著变化 / Time factor has significant effect, skill level changes significantly over time");
        } else {
            System.out.println("  时间因素无显著效应，技能水平随时间无显著变化 / Time factor has no significant effect, skill level does not change significantly over time");
        }
        if (result.subjectP < 0.05) {
            System.out.println("  被试间存在显著差异 / Significant differences exist between subjects");
        }
    }
    
    public static void demonstrateAssumptionTests() {
        System.out.println("\n--- 假设检验示例 / Assumption Tests Example ---");
        
        // 创建示例数据 / Create sample data
        IVector sample1 = IVector.of(new float[]{1.2f, 2.3f, 1.8f, 3.1f, 2.7f});
        IVector sample2 = IVector.of(new float[]{2.1f, 3.2f, 2.8f, 4.1f, 3.5f});
        IVector sample3 = IVector.of(new float[]{3.2f, 4.1f, 3.8f, 5.2f, 4.6f});
        
        System.out.println("假设检验: / Assumption tests:");
        
        // 正态性检验 / Normality test
        System.out.println("\n1. 正态性检验 / Normality Test:");
        boolean isNormal1 = ANOVA.testNormality(sample1);
        boolean isNormal2 = ANOVA.testNormality(sample2);
        boolean isNormal3 = ANOVA.testNormality(sample3);
        
        System.out.println("  样本1正态性: " + (isNormal1 ? "通过" : "未通过") + " / Sample 1 normality: " + (isNormal1 ? "Pass" : "Fail"));
        System.out.println("  样本2正态性: " + (isNormal2 ? "通过" : "未通过") + " / Sample 2 normality: " + (isNormal2 ? "Pass" : "Fail"));
        System.out.println("  样本3正态性: " + (isNormal3 ? "通过" : "未通过") + " / Sample 3 normality: " + (isNormal3 ? "Pass" : "Fail"));
        
        // 方差齐性检验 / Homogeneity of variance test
        System.out.println("\n2. 方差齐性检验 / Homogeneity of Variance Test:");
        boolean isHomogeneous = ANOVA.testHomogeneityOfVariance(sample1, sample2, sample3);
        System.out.println("  方差齐性: " + (isHomogeneous ? "通过" : "未通过") + " / Variance homogeneity: " + (isHomogeneous ? "Pass" : "Fail"));
        
        // 给出建议 / Provide recommendations
        System.out.println("\n3. 分析建议 / Analysis Recommendations:");
        if (isNormal1 && isNormal2 && isNormal3 && isHomogeneous) {
            System.out.println("  所有假设均满足，可以使用参数检验方法 / All assumptions met, parametric tests can be used");
        } else {
            System.out.println("  部分假设未满足，建议使用非参数检验方法 / Some assumptions not met, non-parametric tests recommended");
        }
    }
}
```

### 5.6 蒙特卡洛模拟 / Monte Carlo Simulation

```java
public class MonteCarloSimulationExample {
    public static void main(String[] args) {
        System.out.println("=== 蒙特卡洛模拟示例 / Monte Carlo Simulation Example ===");
        
        // 期权定价模拟 / Option pricing simulation
        simulateOptionPricing();
        
        // 风险评估模拟 / Risk assessment simulation
        simulateRiskAssessment();
        
        // 质量控制模拟 / Quality control simulation
        simulateQualityControl();
    }
    
    public static void simulateOptionPricing() {
        System.out.println("\n--- 期权定价蒙特卡洛模拟 / Option Pricing Monte Carlo Simulation ---");
        
        // 参数设置 / Parameter settings
        float S0 = 100.0f;      // 当前股价 / Current stock price
        float K = 105.0f;       // 执行价格 / Strike price
        float r = 0.05f;        // 无风险利率 / Risk-free rate
        float sigma = 0.2f;     // 波动率 / Volatility
        float T = 1.0f;         // 到期时间 / Time to maturity
        int nSimulations = 100000;
        
        // 创建正态分布 / Create normal distribution
        NormalDistribution normal = Stat.norm();
        
        float[] payoffs = new float[nSimulations];
        
        // 蒙特卡洛模拟 / Monte Carlo simulation
        for (int i = 0; i < nSimulations; i++) {
            float z = normal.sample();
            float ST = S0 * (float) Math.exp((r - 0.5f * sigma * sigma) * T + sigma * (float) Math.sqrt(T) * z);
            payoffs[i] = Math.max(ST - K, 0.0f); // 看涨期权 / Call option
        }
        
        // 计算期权价格 / Calculate option price
        float optionPrice = calculateMean(payoffs) * (float) Math.exp(-r * T);
        
        System.out.printf("模拟次数: %d / Number of simulations: %d%n", nSimulations, nSimulations);
        System.out.printf("期权价格: %.4f / Option price: %.4f%n", optionPrice, optionPrice);
        System.out.printf("标准差: %.4f / Standard deviation: %.4f%n", calculateStd(payoffs), calculateStd(payoffs));
    }
    
    public static void simulateRiskAssessment() {
        System.out.println("\n--- 风险评估蒙特卡洛模拟 / Risk Assessment Monte Carlo Simulation ---");
        
        // 投资组合参数 / Portfolio parameters
        float[] weights = {0.4f, 0.3f, 0.3f};  // 权重 / Weights
        float[] means = {0.08f, 0.12f, 0.06f}; // 预期收益率 / Expected returns
        float[] stds = {0.15f, 0.25f, 0.10f};  // 标准差 / Standard deviations
        int nSimulations = 50000;
        int timeHorizon = 252; // 交易日 / Trading days
        
        // 创建正态分布 / Create normal distribution
        NormalDistribution[] distributions = new NormalDistribution[3];
        for (int i = 0; i < 3; i++) {
            distributions[i] = Stat.norm(means[i], stds[i]);
        }
        
        float[] portfolioReturns = new float[nSimulations];
        
        // 蒙特卡洛模拟 / Monte Carlo simulation
        for (int sim = 0; sim < nSimulations; sim++) {
            float portfolioReturn = 0.0f;
            for (int i = 0; i < 3; i++) {
                float dailyReturn = distributions[i].sample() / (float) Math.sqrt(timeHorizon);
                portfolioReturn += weights[i] * dailyReturn;
            }
            portfolioReturns[sim] = portfolioReturn;
        }
        
        // 计算风险指标 / Calculate risk metrics
        float meanReturn = calculateMean(portfolioReturns);
        float stdReturn = calculateStd(portfolioReturns);
        float var95 = calculatePercentile(portfolioReturns, 0.05f);
        float var99 = calculatePercentile(portfolioReturns, 0.01f);
        
        System.out.printf("预期收益率: %.4f / Expected return: %.4f%n", meanReturn, meanReturn);
        System.out.printf("收益率标准差: %.4f / Return standard deviation: %.4f%n", stdReturn, stdReturn);
        System.out.printf("95%% VaR: %.4f / 95%% VaR: %.4f%n", var95, var95);
        System.out.printf("99%% VaR: %.4f / 99%% VaR: %.4f%n", var99, var99);
    }
    
    public static void simulateQualityControl() {
        System.out.println("\n--- 质量控制蒙特卡洛模拟 / Quality Control Monte Carlo Simulation ---");
        
        // 质量控制参数 / Quality control parameters
        float targetMean = 100.0f;     // 目标均值 / Target mean
        float targetStd = 5.0f;        // 目标标准差 / Target standard deviation
        float specLower = 90.0f;       // 规格下限 / Specification lower limit
        float specUpper = 110.0f;      // 规格上限 / Specification upper limit
        int nSimulations = 100000;
        int sampleSize = 30;
        
        // 创建正态分布 / Create normal distribution
        NormalDistribution normal = Stat.norm(targetMean, targetStd);
        
        int defectCount = 0;
        
        // 蒙特卡洛模拟 / Monte Carlo simulation
        for (int sim = 0; sim < nSimulations; sim++) {
            float[] sample = normal.sample(sampleSize);
            
            // 检查是否有不合格品 / Check for defective products
            boolean hasDefect = false;
            for (float value : sample) {
                if (value < specLower || value > specUpper) {
                    hasDefect = true;
                    break;
                }
            }
            
            if (hasDefect) {
                defectCount++;
            }
        }
        
        float defectRate = (float) defectCount / nSimulations;
        System.out.printf("模拟次数: %d / Number of simulations: %d%n", nSimulations, nSimulations);
        System.out.printf("样本大小: %d / Sample size: %d%n", sampleSize, sampleSize);
        System.out.printf("不合格品率: %.4f%% / Defect rate: %.4f%%%n", defectRate * 100, defectRate * 100);
    }
    
    // 辅助方法 / Helper methods
    private static float calculateMean(float[] data) {
        float sum = 0;
        for (float value : data) {
            sum += value;
        }
        return sum / data.length;
    }
    
    private static float calculateStd(float[] data) {
        float mean = calculateMean(data);
        float sumSquaredDiff = 0;
        for (float value : data) {
            float diff = value - mean;
            sumSquaredDiff += diff * diff;
        }
        return (float) Math.sqrt(sumSquaredDiff / data.length);
    }
    
    private static float calculatePercentile(float[] data, float percentile) {
        float[] sorted = data.clone();
        Arrays.sort(sorted);
        int index = (int) (sorted.length * percentile);
        return sorted[index];
    }
}
```

---

## 学习路径建议 / Learning Path Recommendations

### 初学者路径 / Beginner Path
1. 从第一部分开始，掌握基本统计量计算 / Start with Part 1, master basic statistical calculations
2. 理解正态分布和均匀分布的基本概念 / Understand basic concepts of normal and uniform distributions
3. 学习简单的随机采样和概率计算 / Learn simple random sampling and probability calculations

### 中级用户路径 / Intermediate Path
1. 掌握t分布、卡方分布、F分布等统计分布 / Master statistical distributions like t-distribution, chi-squared, F-distribution
2. 学习参数估计和假设检验方法 / Learn parameter estimation and hypothesis testing methods
3. 理解不同分布的应用场景 / Understand application scenarios of different distributions

### 高级用户路径 / Advanced Path
1. 掌握复杂的统计分析方法 / Master complex statistical analysis methods
2. 学习实际业务场景的应用 / Learn applications in real business scenarios
3. 理解统计推断的原理和实践 / Understand principles and practice of statistical inference

### 专业用户路径 / Professional Path
1. 掌握蒙特卡洛模拟等高级方法 / Master advanced methods like Monte Carlo simulation
2. 能够设计复杂的统计分析方案 / Be able to design complex statistical analysis solutions
3. 能够处理实际业务中的复杂统计问题 / Be able to handle complex statistical problems in real business

---

## 总结 / Summary

本文档按照从简单到复杂的顺序，系统性地介绍了统计操作包的各种功能。通过循序渐进的学习，您可以：

This document systematically introduces various functions of the statistics package in order from simple to complex. Through progressive learning, you can:

- **掌握基础**：从基本统计量开始，逐步建立统计基础 / **Master the basics**: Start with basic statistics and gradually build statistical foundations
- **应用实践**：通过实际案例学习不同统计方法的使用场景 / **Apply in practice**: Learn usage scenarios of different statistical methods through real cases
- **进阶提升**：掌握高级统计方法和专业应用 / **Advance and improve**: Master advanced statistical methods and professional applications
- **灵活运用**：根据实际需求选择合适的统计分析方法 / **Use flexibly**: Choose appropriate statistical analysis methods based on actual needs

### 新增功能亮点 / New Feature Highlights

- **方差分析 (ANOVA)**：完整的单因素、两因素和重复测量方差分析功能 / **Analysis of Variance (ANOVA)**: Complete one-way, two-way, and repeated measures ANOVA functionality
- **假设检验**：正态性检验和方差齐性检验 / **Assumption Testing**: Normality tests and homogeneity of variance tests
- **多重比较**：Tukey HSD多重比较方法 / **Multiple Comparisons**: Tukey HSD multiple comparison methods
- **中英文对照**：完整的双语文档支持 / **Bilingual Support**: Complete bilingual documentation support

---

**统计操作示例** - 让统计分析更简单！

**Statistics Examples** - Make statistical analysis simpler!
