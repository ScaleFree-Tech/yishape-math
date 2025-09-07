# 统计操作示例 (Statistics Examples)

## 概述 / Overview

本文档提供了 `Stat` 类和概率分布类的详细使用示例，包括各种统计分布的基本操作、概率计算、随机采样等。

This document provides detailed usage examples for the `Stat` class and probability distribution classes, including basic operations, probability calculations, random sampling for various statistical distributions.

## 基本使用 / Basic Usage

### 导入必要的类 / Import Required Classes

```java
import com.reremouse.lab.math.stat.Stat;
import com.reremouse.lab.math.stat.distribution.*;
```

## 正态分布示例 / Normal Distribution Examples

### 1. 创建正态分布 / Creating Normal Distribution

```java
// 标准正态分布（均值为0，标准差为1）/ Standard normal distribution (mean=0, stdDev=1)
NormalDistribution standardNormal = Stat.norm();
System.out.println("标准正态分布: " + standardNormal);

// 自定义参数的正态分布 / Custom parameter normal distribution
NormalDistribution normal = Stat.norm(5.0f, 2.0f); // 均值=5，标准差=2
System.out.println("正态分布(μ=5, σ=2): " + normal);
```

### 2. 基本统计量 / Basic Statistics

```java
NormalDistribution normal = Stat.norm(10.0f, 3.0f);

// 基本统计量 / Basic statistics
System.out.println("均值 / Mean: " + normal.mean());
System.out.println("方差 / Variance: " + normal.var());
System.out.println("标准差 / Standard deviation: " + normal.std());
System.out.println("中位数 / Median: " + normal.median());
System.out.println("众数 / Mode: " + normal.mode());
System.out.println("偏度 / Skewness: " + normal.skewness());
System.out.println("峰度 / Kurtosis: " + normal.kurtosis());

// 四分位数 / Quartiles
System.out.println("第一四分位数 / Q1: " + normal.q1());
System.out.println("第三四分位数 / Q3: " + normal.q3());
```

### 3. 概率密度函数 / Probability Density Function

```java
NormalDistribution normal = Stat.norm(0.0f, 1.0f);

// 计算不同点的PDF值 / Calculate PDF values at different points
float[] xValues = {-3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
System.out.println("正态分布PDF值 / Normal Distribution PDF Values:");
for (float x : xValues) {
    float pdf = normal.pdf(x);
    System.out.printf("x=%.1f, PDF=%.4f%n", x, pdf);
}
```

### 4. 累积分布函数 / Cumulative Distribution Function

```java
NormalDistribution normal = Stat.norm(0.0f, 1.0f);

// 计算不同点的CDF值 / Calculate CDF values at different points
float[] xValues = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
System.out.println("正态分布CDF值 / Normal Distribution CDF Values:");
for (float x : xValues) {
    float cdf = normal.cdf(x);
    System.out.printf("x=%.1f, CDF=%.4f%n", x, cdf);
}
```

### 5. 百分点函数（分位数）/ Percent Point Function (Quantiles)

```java
NormalDistribution normal = Stat.norm(0.0f, 1.0f);

// 计算不同概率的分位数 / Calculate quantiles for different probabilities
float[] probabilities = {0.01f, 0.05f, 0.1f, 0.25f, 0.5f, 0.75f, 0.9f, 0.95f, 0.99f};
System.out.println("正态分布分位数 / Normal Distribution Quantiles:");
for (float p : probabilities) {
    float quantile = normal.ppf(p);
    System.out.printf("P=%.2f, 分位数=%.4f%n", p, quantile);
}
```

### 6. 随机采样 / Random Sampling

```java
NormalDistribution normal = Stat.norm(5.0f, 2.0f);

// 生成单个随机样本 / Generate single random sample
float sample = normal.sample();
System.out.println("随机样本 / Random sample: " + sample);

// 生成多个随机样本 / Generate multiple random samples
float[] samples = normal.sample(1000);
System.out.println("生成了 " + samples.length + " 个随机样本 / Generated " + samples.length + " random samples");

// 计算样本统计量 / Calculate sample statistics
float sampleMean = calculateMean(samples);
float sampleVar = calculateVariance(samples, sampleMean);
System.out.printf("样本均值 / Sample mean: %.4f%n", sampleMean);
System.out.printf("样本方差 / Sample variance: %.4f%n", sampleVar);
System.out.printf("理论均值 / Theoretical mean: %.4f%n", normal.mean());
System.out.printf("理论方差 / Theoretical variance: %.4f%n", normal.var());
```

## t分布示例 / Student's t-Distribution Examples

### 1. 创建t分布 / Creating t-Distribution

```java
// 不同自由度的t分布 / t-distributions with different degrees of freedom
StudentDistribution t5 = Stat.t(5.0f);
StudentDistribution t10 = Stat.t(10.0f);
StudentDistribution t30 = Stat.t(30.0f);

System.out.println("t分布(5自由度) / t-distribution(5 df): " + t5);
System.out.println("t分布(10自由度) / t-distribution(10 df): " + t10);
System.out.println("t分布(30自由度) / t-distribution(30 df): " + t30);
```

### 2. t分布统计量 / t-Distribution Statistics

```java
StudentDistribution tDist = Stat.t(10.0f);

System.out.println("t分布统计量 / t-Distribution Statistics:");
System.out.println("均值 / Mean: " + tDist.mean());
System.out.println("方差 / Variance: " + tDist.var());
System.out.println("标准差 / Standard deviation: " + tDist.std());
System.out.println("偏度 / Skewness: " + tDist.skewness());
System.out.println("峰度 / Kurtosis: " + tDist.kurtosis());
```

### 3. t分布概率计算 / t-Distribution Probability Calculations

```java
StudentDistribution tDist = Stat.t(10.0f);

// 计算t检验的临界值 / Calculate critical values for t-test
float alpha = 0.05f;
float criticalValue = tDist.ppf(1.0f - alpha/2.0f);
System.out.printf("95%%置信区间的临界值 / Critical value for 95%% confidence: %.4f%n", criticalValue);

// 计算p值 / Calculate p-value
float tStatistic = 2.5f;
float pValue = 2.0f * (1.0f - tDist.cdf(Math.abs(tStatistic)));
System.out.printf("t统计量=%.2f的p值 / p-value for t-statistic=%.2f: %.4f%n", tStatistic, tStatistic, pValue);
```

## 均匀分布示例 / Uniform Distribution Examples

### 1. 创建均匀分布 / Creating Uniform Distribution

```java
// 不同区间的均匀分布 / Uniform distributions with different intervals
UniformDistribution uniform1 = Stat.uniform(0.0f, 1.0f);    // [0, 1]
UniformDistribution uniform2 = Stat.uniform(-5.0f, 5.0f);   // [-5, 5]
UniformDistribution uniform3 = Stat.uniform(10.0f, 20.0f);  // [10, 20]

System.out.println("均匀分布[0,1] / Uniform[0,1]: " + uniform1);
System.out.println("均匀分布[-5,5] / Uniform[-5,5]: " + uniform2);
System.out.println("均匀分布[10,20] / Uniform[10,20]: " + uniform3);
```

### 2. 均匀分布概率计算 / Uniform Distribution Probability Calculations

```java
UniformDistribution uniform = Stat.uniform(0.0f, 10.0f);

// 计算区间内外的概率 / Calculate probabilities inside and outside interval
System.out.println("均匀分布[0,10]概率计算 / Uniform[0,10] Probability Calculations:");
System.out.println("P(X ≤ 5) = " + uniform.cdf(5.0f));
System.out.println("P(X ≤ 2) = " + uniform.cdf(2.0f));
System.out.println("P(X ≤ 8) = " + uniform.cdf(8.0f));
System.out.println("P(3 ≤ X ≤ 7) = " + (uniform.cdf(7.0f) - uniform.cdf(3.0f)));

// 计算分位数 / Calculate quantiles
System.out.println("50%分位数 / 50th percentile: " + uniform.ppf(0.5f));
System.out.println("90%分位数 / 90th percentile: " + uniform.ppf(0.9f));
```

### 3. 均匀分布随机采样 / Uniform Distribution Random Sampling

```java
UniformDistribution uniform = Stat.uniform(0.0f, 100.0f);

// 生成随机样本并分析 / Generate random samples and analyze
float[] samples = uniform.sample(10000);
System.out.println("生成了 " + samples.length + " 个均匀分布样本 / Generated " + samples.length + " uniform samples");

// 计算样本统计量 / Calculate sample statistics
float sampleMean = calculateMean(samples);
float sampleVar = calculateVariance(samples, sampleMean);
float sampleMin = calculateMin(samples);
float sampleMax = calculateMax(samples);

System.out.printf("样本均值 / Sample mean: %.4f (理论值 / Theoretical: %.4f)%n", sampleMean, uniform.mean());
System.out.printf("样本方差 / Sample variance: %.4f (理论值 / Theoretical: %.4f)%n", sampleVar, uniform.var());
System.out.printf("样本最小值 / Sample min: %.4f%n", sampleMin);
System.out.printf("样本最大值 / Sample max: %.4f%n", sampleMax);
```

## 指数分布示例 / Exponential Distribution Examples

### 1. 创建指数分布 / Creating Exponential Distribution

```java
// 不同速率参数的指数分布 / Exponential distributions with different rate parameters
ExponentialDistribution exp1 = Stat.exponential(1.0f);   // λ = 1
ExponentialDistribution exp2 = Stat.exponential(2.0f);   // λ = 2
ExponentialDistribution exp3 = Stat.exponential(0.5f);   // λ = 0.5

System.out.println("指数分布(λ=1) / Exponential(λ=1): " + exp1);
System.out.println("指数分布(λ=2) / Exponential(λ=2): " + exp2);
System.out.println("指数分布(λ=0.5) / Exponential(λ=0.5): " + exp3);
```

### 2. 指数分布概率计算 / Exponential Distribution Probability Calculations

```java
ExponentialDistribution exp = Stat.exponential(2.0f); // λ = 2

System.out.println("指数分布(λ=2)概率计算 / Exponential(λ=2) Probability Calculations:");
System.out.println("P(X ≤ 1) = " + exp.cdf(1.0f));
System.out.println("P(X ≤ 2) = " + exp.cdf(2.0f));
System.out.println("P(X > 3) = " + exp.sf(3.0f));
System.out.println("P(1 ≤ X ≤ 3) = " + (exp.cdf(3.0f) - exp.cdf(1.0f)));

// 计算生存函数 / Calculate survival function
System.out.println("P(X > 1) = " + exp.sf(1.0f));
System.out.println("P(X > 5) = " + exp.sf(5.0f));
```

### 3. 指数分布应用示例 / Exponential Distribution Application Examples

```java
ExponentialDistribution exp = Stat.exponential(0.1f); // 平均寿命 = 10

System.out.println("设备寿命分析 / Equipment Lifetime Analysis:");
System.out.println("平均寿命 / Mean lifetime: " + exp.mean() + " 年 / years");
System.out.println("标准差 / Standard deviation: " + exp.std() + " 年 / years");

// 计算不同时间点的生存概率 / Calculate survival probabilities at different times
float[] times = {1.0f, 5.0f, 10.0f, 15.0f, 20.0f};
System.out.println("生存概率 / Survival Probabilities:");
for (float time : times) {
    float survivalProb = exp.sf(time);
    System.out.printf("P(寿命 > %.0f年) = %.4f%n", time, survivalProb);
}
```

## 卡方分布示例 / Chi-Squared Distribution Examples

### 1. 创建卡方分布 / Creating Chi-Squared Distribution

```java
// 不同自由度的卡方分布 / Chi-squared distributions with different degrees of freedom
Chi2Distribution chi2_1 = Stat.chi2(1.0f);
Chi2Distribution chi2_5 = Stat.chi2(5.0f);
Chi2Distribution chi2_10 = Stat.chi2(10.0f);

System.out.println("卡方分布(1自由度) / Chi-squared(1 df): " + chi2_1);
System.out.println("卡方分布(5自由度) / Chi-squared(5 df): " + chi2_5);
System.out.println("卡方分布(10自由度) / Chi-squared(10 df): " + chi2_10);
```

### 2. 卡方检验示例 / Chi-Squared Test Examples

```java
Chi2Distribution chi2 = Stat.chi2(5.0f);

// 计算卡方检验的临界值 / Calculate critical values for chi-squared test
float alpha = 0.05f;
float criticalValue = chi2.ppf(1.0f - alpha);
System.out.printf("95%%置信水平的临界值 / Critical value at 95%% confidence: %.4f%n", criticalValue);

// 计算p值 / Calculate p-value
float chi2Statistic = 8.5f;
float pValue = 1.0f - chi2.cdf(chi2Statistic);
System.out.printf("卡方统计量=%.2f的p值 / p-value for chi-squared=%.2f: %.4f%n", chi2Statistic, chi2Statistic, pValue);
```

## F分布示例 / F-Distribution Examples

### 1. 创建F分布 / Creating F-Distribution

```java
// 不同自由度的F分布 / F-distributions with different degrees of freedom
FDistribution f1 = Stat.f(5.0f, 10.0f);
FDistribution f2 = Stat.f(10.0f, 5.0f);
FDistribution f3 = Stat.f(20.0f, 20.0f);

System.out.println("F分布(5,10自由度) / F-distribution(5,10 df): " + f1);
System.out.println("F分布(10,5自由度) / F-distribution(10,5 df): " + f2);
System.out.println("F分布(20,20自由度) / F-distribution(20,20 df): " + f3);
```

### 2. F检验示例 / F-Test Examples

```java
FDistribution fDist = Stat.f(5.0f, 10.0f);

// 计算F检验的临界值 / Calculate critical values for F-test
float alpha = 0.05f;
float criticalValue = fDist.ppf(1.0f - alpha);
System.out.printf("95%%置信水平的F临界值 / F critical value at 95%% confidence: %.4f%n", criticalValue);

// 计算p值 / Calculate p-value
float fStatistic = 3.0f;
float pValue = 1.0f - fDist.cdf(fStatistic);
System.out.printf("F统计量=%.2f的p值 / p-value for F-statistic=%.2f: %.4f%n", fStatistic, fStatistic, pValue);
```

## 综合应用示例 / Comprehensive Application Examples

### 1. 假设检验 / Hypothesis Testing

```java
public class HypothesisTesting {
    
    public static void main(String[] args) {
        // 单样本t检验 / One-sample t-test
        performOneSampleTTest();
        
        // 双样本t检验 / Two-sample t-test
        performTwoSampleTTest();
        
        // 卡方检验 / Chi-squared test
        performChiSquaredTest();
        
        // F检验 / F-test
        performFTest();
    }
    
    public static void performOneSampleTTest() {
        System.out.println("=== 单样本t检验 / One-Sample t-Test ===");
        
        // 假设总体均值为0，样本数据 / Assume population mean is 0, sample data
        float[] sample = {1.2f, -0.5f, 0.8f, -1.1f, 0.3f, 0.9f, -0.2f, 1.5f, -0.8f, 0.1f};
        float sampleMean = calculateMean(sample);
        float sampleStd = (float) Math.sqrt(calculateVariance(sample, sampleMean));
        int n = sample.length;
        
        // 计算t统计量 / Calculate t-statistic
        float tStatistic = sampleMean / (sampleStd / (float) Math.sqrt(n));
        
        // 创建t分布 / Create t-distribution
        StudentDistribution tDist = Stat.t(n - 1);
        
        // 计算p值 / Calculate p-value
        float pValue = 2.0f * (1.0f - tDist.cdf(Math.abs(tStatistic)));
        
        System.out.printf("样本均值 / Sample mean: %.4f%n", sampleMean);
        System.out.printf("样本标准差 / Sample std: %.4f%n", sampleStd);
        System.out.printf("t统计量 / t-statistic: %.4f%n", tStatistic);
        System.out.printf("p值 / p-value: %.4f%n", pValue);
        System.out.printf("显著性水平α=0.05: %s%n", pValue < 0.05 ? "拒绝原假设 / Reject H0" : "接受原假设 / Accept H0");
    }
    
    public static void performTwoSampleTTest() {
        System.out.println("\n=== 双样本t检验 / Two-Sample t-Test ===");
        
        // 两组样本数据 / Two groups of sample data
        float[] group1 = {10.2f, 9.8f, 10.5f, 9.9f, 10.1f, 10.3f, 9.7f, 10.0f};
        float[] group2 = {9.5f, 9.2f, 9.8f, 9.1f, 9.7f, 9.4f, 9.6f, 9.3f};
        
        float mean1 = calculateMean(group1);
        float mean2 = calculateMean(group2);
        float var1 = calculateVariance(group1, mean1);
        float var2 = calculateVariance(group2, mean2);
        int n1 = group1.length;
        int n2 = group2.length;
        
        // 计算合并方差 / Calculate pooled variance
        float pooledVar = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);
        
        // 计算t统计量 / Calculate t-statistic
        float tStatistic = (mean1 - mean2) / (float) Math.sqrt(pooledVar * (1.0f/n1 + 1.0f/n2));
        
        // 创建t分布 / Create t-distribution
        StudentDistribution tDist = Stat.t(n1 + n2 - 2);
        
        // 计算p值 / Calculate p-value
        float pValue = 2.0f * (1.0f - tDist.cdf(Math.abs(tStatistic)));
        
        System.out.printf("组1均值 / Group 1 mean: %.4f%n", mean1);
        System.out.printf("组2均值 / Group 2 mean: %.4f%n", mean2);
        System.out.printf("t统计量 / t-statistic: %.4f%n", tStatistic);
        System.out.printf("p值 / p-value: %.4f%n", pValue);
        System.out.printf("显著性水平α=0.05: %s%n", pValue < 0.05 ? "拒绝原假设 / Reject H0" : "接受原假设 / Accept H0");
    }
    
    public static void performChiSquaredTest() {
        System.out.println("\n=== 卡方检验 / Chi-Squared Test ===");
        
        // 观察频数和期望频数 / Observed and expected frequencies
        float[] observed = {25.0f, 30.0f, 20.0f, 25.0f};
        float[] expected = {25.0f, 25.0f, 25.0f, 25.0f};
        
        // 计算卡方统计量 / Calculate chi-squared statistic
        float chi2Statistic = 0.0f;
        for (int i = 0; i < observed.length; i++) {
            float diff = observed[i] - expected[i];
            chi2Statistic += (diff * diff) / expected[i];
        }
        
        // 创建卡方分布 / Create chi-squared distribution
        Chi2Distribution chi2 = Stat.chi2(observed.length - 1);
        
        // 计算p值 / Calculate p-value
        float pValue = 1.0f - chi2.cdf(chi2Statistic);
        
        System.out.printf("卡方统计量 / Chi-squared statistic: %.4f%n", chi2Statistic);
        System.out.printf("自由度 / Degrees of freedom: %d%n", observed.length - 1);
        System.out.printf("p值 / p-value: %.4f%n", pValue);
        System.out.printf("显著性水平α=0.05: %s%n", pValue < 0.05 ? "拒绝原假设 / Reject H0" : "接受原假设 / Accept H0");
    }
    
    public static void performFTest() {
        System.out.println("\n=== F检验 / F-Test ===");
        
        // 两组样本的方差 / Variances of two groups
        float[] group1 = {10.2f, 9.8f, 10.5f, 9.9f, 10.1f};
        float[] group2 = {9.5f, 9.2f, 9.8f, 9.1f, 9.7f, 9.4f};
        
        float mean1 = calculateMean(group1);
        float mean2 = calculateMean(group2);
        float var1 = calculateVariance(group1, mean1);
        float var2 = calculateVariance(group2, mean2);
        
        // 计算F统计量 / Calculate F-statistic
        float fStatistic = Math.max(var1, var2) / Math.min(var1, var2);
        
        // 创建F分布 / Create F-distribution
        FDistribution fDist = Stat.f(group1.length - 1, group2.length - 1);
        
        // 计算p值 / Calculate p-value
        float pValue = 2.0f * (1.0f - fDist.cdf(fStatistic));
        
        System.out.printf("组1方差 / Group 1 variance: %.4f%n", var1);
        System.out.printf("组2方差 / Group 2 variance: %.4f%n", var2);
        System.out.printf("F统计量 / F-statistic: %.4f%n", fStatistic);
        System.out.printf("p值 / p-value: %.4f%n", pValue);
        System.out.printf("显著性水平α=0.05: %s%n", pValue < 0.05 ? "拒绝原假设 / Reject H0" : "接受原假设 / Accept H0");
    }
}
```

### 2. 蒙特卡洛模拟 / Monte Carlo Simulation

```java
public class MonteCarloSimulation {
    
    public static void main(String[] args) {
        // 期权定价模拟 / Option pricing simulation
        simulateOptionPricing();
        
        // 风险评估模拟 / Risk assessment simulation
        simulateRiskAssessment();
        
        // 质量控制模拟 / Quality control simulation
        simulateQualityControl();
    }
    
    public static void simulateOptionPricing() {
        System.out.println("=== 期权定价蒙特卡洛模拟 / Option Pricing Monte Carlo Simulation ===");
        
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
        
        System.out.printf("模拟次数 / Number of simulations: %d%n", nSimulations);
        System.out.printf("期权价格 / Option price: %.4f%n", optionPrice);
        System.out.printf("标准差 / Standard deviation: %.4f%n", calculateStd(payoffs));
    }
    
    public static void simulateRiskAssessment() {
        System.out.println("\n=== 风险评估蒙特卡洛模拟 / Risk Assessment Monte Carlo Simulation ===");
        
        // 投资组合参数 / Portfolio parameters
        float[] weights = {0.4f, 0.3f, 0.3f};  // 权重 / Weights
        float[] means = {0.08f, 0.12f, 0.06f}; // 预期收益率 / Expected returns
        float[] stds = {0.15f, 0.25f, 0.10f};  // 标准差 / Standard deviations
        int nSimulations = 50000;
        int timeHorizon = 252; // 交易日 / Trading days
        
        // 创建正态分布 / Create normal distributions
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
        
        System.out.printf("预期收益率 / Expected return: %.4f%n", meanReturn);
        System.out.printf("收益率标准差 / Return std: %.4f%n", stdReturn);
        System.out.printf("95%% VaR / 95%% VaR: %.4f%n", var95);
        System.out.printf("99%% VaR / 99%% VaR: %.4f%n", var99);
    }
    
    public static void simulateQualityControl() {
        System.out.println("\n=== 质量控制蒙特卡洛模拟 / Quality Control Monte Carlo Simulation ===");
        
        // 质量控制参数 / Quality control parameters
        float targetMean = 100.0f;     // 目标均值 / Target mean
        float targetStd = 5.0f;        // 目标标准差 / Target std
        float specLower = 90.0f;       // 规格下限 / Specification lower limit
        float specUpper = 110.0f;      // 规格上限 / Specification upper limit
        int nSimulations = 100000;
        int sampleSize = 30;
        
        // 创建正态分布 / Create normal distribution
        NormalDistribution normal = Stat.norm(targetMean, targetStd);
        
        int defectiveCount = 0;
        float[] sampleMeans = new float[nSimulations];
        
        // 蒙特卡洛模拟 / Monte Carlo simulation
        for (int sim = 0; sim < nSimulations; sim++) {
            float[] sample = normal.sample(sampleSize);
            float sampleMean = calculateMean(sample);
            sampleMeans[sim] = sampleMean;
            
            // 检查是否有缺陷品 / Check for defective items
            for (float value : sample) {
                if (value < specLower || value > specUpper) {
                    defectiveCount++;
                    break;
                }
            }
        }
        
        float defectRate = (float) defectiveCount / nSimulations;
        float meanOfMeans = calculateMean(sampleMeans);
        float stdOfMeans = calculateStd(sampleMeans);
        
        System.out.printf("模拟次数 / Number of simulations: %d%n", nSimulations);
        System.out.printf("样本大小 / Sample size: %d%n", sampleSize);
        System.out.printf("缺陷率 / Defect rate: %.4f%n", defectRate);
        System.out.printf("样本均值分布均值 / Mean of sample means: %.4f%n", meanOfMeans);
        System.out.printf("样本均值分布标准差 / Std of sample means: %.4f%n", stdOfMeans);
    }
}
```

### 3. 数据拟合和模型验证 / Data Fitting and Model Validation

```java
public class DataFitting {
    
    public static void main(String[] args) {
        // 正态性检验 / Normality test
        testNormality();
        
        // 分布拟合 / Distribution fitting
        fitDistributions();
        
        // 模型验证 / Model validation
        validateModel();
    }
    
    public static void testNormality() {
        System.out.println("=== 正态性检验 / Normality Test ===");
        
        // 生成测试数据 / Generate test data
        NormalDistribution normal = Stat.norm(50.0f, 10.0f);
        float[] data = normal.sample(1000);
        
        // 计算样本统计量 / Calculate sample statistics
        float sampleMean = calculateMean(data);
        float sampleStd = calculateStd(data);
        float sampleSkewness = calculateSkewness(data, sampleMean);
        float sampleKurtosis = calculateKurtosis(data, sampleMean);
        
        System.out.printf("样本大小 / Sample size: %d%n", data.length);
        System.out.printf("样本均值 / Sample mean: %.4f%n", sampleMean);
        System.out.printf("样本标准差 / Sample std: %.4f%n", sampleStd);
        System.out.printf("样本偏度 / Sample skewness: %.4f%n", sampleSkewness);
        System.out.printf("样本峰度 / Sample kurtosis: %.4f%n", sampleKurtosis);
        
        // 正态性判断 / Normality assessment
        boolean isNormal = Math.abs(sampleSkewness) < 0.5f && Math.abs(sampleKurtosis) < 0.5f;
        System.out.printf("正态性判断 / Normality assessment: %s%n", 
            isNormal ? "近似正态 / Approximately normal" : "非正态 / Not normal");
    }
    
    public static void fitDistributions() {
        System.out.println("\n=== 分布拟合 / Distribution Fitting ===");
        
        // 生成混合分布数据 / Generate mixed distribution data
        float[] data = generateMixedData(1000);
        
        // 尝试拟合不同分布 / Try fitting different distributions
        System.out.println("尝试拟合不同分布 / Trying to fit different distributions:");
        
        // 正态分布拟合 / Normal distribution fitting
        float mean = calculateMean(data);
        float std = calculateStd(data);
        NormalDistribution normal = Stat.norm(mean, std);
        float normalLogLikelihood = calculateLogLikelihood(data, normal);
        System.out.printf("正态分布对数似然 / Normal distribution log-likelihood: %.4f%n", normalLogLikelihood);
        
        // 指数分布拟合 / Exponential distribution fitting
        float rate = 1.0f / mean;
        ExponentialDistribution exp = Stat.exponential(rate);
        float expLogLikelihood = calculateLogLikelihood(data, exp);
        System.out.printf("指数分布对数似然 / Exponential distribution log-likelihood: %.4f%n", expLogLikelihood);
        
        // 选择最佳拟合 / Choose best fit
        String bestFit = normalLogLikelihood > expLogLikelihood ? "正态分布 / Normal" : "指数分布 / Exponential";
        System.out.printf("最佳拟合分布 / Best fitting distribution: %s%n", bestFit);
    }
    
    public static void validateModel() {
        System.out.println("\n=== 模型验证 / Model Validation ===");
        
        // 生成训练和测试数据 / Generate training and test data
        NormalDistribution trueModel = Stat.norm(100.0f, 15.0f);
        float[] trainingData = trueModel.sample(500);
        float[] testData = trueModel.sample(200);
        
        // 拟合模型 / Fit model
        float fittedMean = calculateMean(trainingData);
        float fittedStd = calculateStd(trainingData);
        NormalDistribution fittedModel = Stat.norm(fittedMean, fittedStd);
        
        // 模型验证 / Model validation
        float[] predictedCDF = new float[testData.length];
        float[] actualCDF = new float[testData.length];
        
        for (int i = 0; i < testData.length; i++) {
            predictedCDF[i] = fittedModel.cdf(testData[i]);
            actualCDF[i] = trueModel.cdf(testData[i]);
        }
        
        // 计算验证指标 / Calculate validation metrics
        float mse = calculateMSE(predictedCDF, actualCDF);
        float mae = calculateMAE(predictedCDF, actualCDF);
        
        System.out.printf("训练数据大小 / Training data size: %d%n", trainingData.length);
        System.out.printf("测试数据大小 / Test data size: %d%n", testData.length);
        System.out.printf("拟合均值 / Fitted mean: %.4f (真实值 / True: %.4f)%n", fittedMean, trueModel.mean());
        System.out.printf("拟合标准差 / Fitted std: %.4f (真实值 / True: %.4f)%n", fittedStd, trueModel.std());
        System.out.printf("CDF均方误差 / CDF MSE: %.6f%n", mse);
        System.out.printf("CDF平均绝对误差 / CDF MAE: %.6f%n", mae);
    }
}
```

## 辅助方法 / Helper Methods

```java
// 计算均值 / Calculate mean
public static float calculateMean(float[] values) {
    float sum = 0.0f;
    for (float value : values) {
        sum += value;
    }
    return sum / values.length;
}

// 计算方差 / Calculate variance
public static float calculateVariance(float[] values, float mean) {
    float sumSquaredDiff = 0.0f;
    for (float value : values) {
        float diff = value - mean;
        sumSquaredDiff += diff * diff;
    }
    return sumSquaredDiff / values.length;
}

// 计算标准差 / Calculate standard deviation
public static float calculateStd(float[] values) {
    float mean = calculateMean(values);
    return (float) Math.sqrt(calculateVariance(values, mean));
}

// 计算最小值 / Calculate minimum
public static float calculateMin(float[] values) {
    float min = Float.MAX_VALUE;
    for (float value : values) {
        if (value < min) {
            min = value;
        }
    }
    return min;
}

// 计算最大值 / Calculate maximum
public static float calculateMax(float[] values) {
    float max = Float.MIN_VALUE;
    for (float value : values) {
        if (value > max) {
            max = value;
        }
    }
    return max;
}

// 计算偏度 / Calculate skewness
public static float calculateSkewness(float[] values, float mean) {
    float sumCubedDiff = 0.0f;
    float sumSquaredDiff = 0.0f;
    
    for (float value : values) {
        float diff = value - mean;
        sumCubedDiff += diff * diff * diff;
        sumSquaredDiff += diff * diff;
    }
    
    float variance = sumSquaredDiff / values.length;
    float std = (float) Math.sqrt(variance);
    
    return (sumCubedDiff / values.length) / (std * std * std);
}

// 计算峰度 / Calculate kurtosis
public static float calculateKurtosis(float[] values, float mean) {
    float sumFourthDiff = 0.0f;
    float sumSquaredDiff = 0.0f;
    
    for (float value : values) {
        float diff = value - mean;
        sumFourthDiff += diff * diff * diff * diff;
        sumSquaredDiff += diff * diff;
    }
    
    float variance = sumSquaredDiff / values.length;
    
    return (sumFourthDiff / values.length) / (variance * variance) - 3.0f;
}

// 计算分位数 / Calculate percentile
public static float calculatePercentile(float[] values, float percentile) {
    float[] sorted = values.clone();
    Arrays.sort(sorted);
    int index = (int) (percentile * (sorted.length - 1));
    return sorted[index];
}

// 计算对数似然 / Calculate log-likelihood
public static float calculateLogLikelihood(float[] data, IContinuousDistribution distribution) {
    float logLikelihood = 0.0f;
    for (float value : data) {
        logLikelihood += (float) Math.log(distribution.pdf(value));
    }
    return logLikelihood;
}

// 计算均方误差 / Calculate mean squared error
public static float calculateMSE(float[] predicted, float[] actual) {
    float sumSquaredError = 0.0f;
    for (int i = 0; i < predicted.length; i++) {
        float error = predicted[i] - actual[i];
        sumSquaredError += error * error;
    }
    return sumSquaredError / predicted.length;
}

// 计算平均绝对误差 / Calculate mean absolute error
public static float calculateMAE(float[] predicted, float[] actual) {
    float sumAbsoluteError = 0.0f;
    for (int i = 0; i < predicted.length; i++) {
        sumAbsoluteError += Math.abs(predicted[i] - actual[i]);
    }
    return sumAbsoluteError / predicted.length;
}

// 生成混合分布数据 / Generate mixed distribution data
public static float[] generateMixedData(int n) {
    float[] data = new float[n];
    NormalDistribution normal1 = Stat.norm(50.0f, 10.0f);
    NormalDistribution normal2 = Stat.norm(80.0f, 15.0f);
    
    for (int i = 0; i < n; i++) {
        if (Math.random() < 0.7) {
            data[i] = normal1.sample();
        } else {
            data[i] = normal2.sample();
        }
    }
    return data;
}
```

## 性能优化建议 / Performance Optimization Recommendations

### 1. 批量采样 / Batch Sampling

```java
// 推荐：批量生成样本 / Recommended: Batch generate samples
float[] samples = distribution.sample(10000);

// 不推荐：逐个生成样本 / Not recommended: Generate samples one by one
float[] samples = new float[10000];
for (int i = 0; i < 10000; i++) {
    samples[i] = distribution.sample();
}
```

### 2. 缓存分布对象 / Cache Distribution Objects

```java
// 推荐：缓存分布对象 / Recommended: Cache distribution objects
private static final NormalDistribution STANDARD_NORMAL = Stat.norm();

// 不推荐：重复创建分布对象 / Not recommended: Repeatedly create distribution objects
for (int i = 0; i < 1000; i++) {
    NormalDistribution normal = Stat.norm(); // 重复创建 / Repeated creation
    float sample = normal.sample();
}
```

### 3. 预计算常量 / Precompute Constants

```java
// 推荐：预计算常量 / Recommended: Precompute constants
private static final float SQRT_2PI = (float) Math.sqrt(2.0 * Math.PI);
private static final float INV_SQRT_2PI = 1.0f / SQRT_2PI;

// 在计算中使用预计算常量 / Use precomputed constants in calculations
float pdf = INV_SQRT_2PI / stdDev * (float) Math.exp(exponent);
```

## 常见问题和解决方案 / Common Issues and Solutions

### 1. 数值精度问题 / Numerical Precision Issues

```java
// 问题：float精度限制 / Issue: float precision limitations
float result = 0.1f + 0.2f; // 可能不等于0.3 / May not equal 0.3

// 解决方案：使用适当的容差 / Solution: Use appropriate tolerance
float tolerance = 1e-6f;
boolean isEqual = Math.abs(result - 0.3f) < tolerance;
```

### 2. 边界情况处理 / Edge Case Handling

```java
// 问题：极值可能导致数值不稳定 / Issue: Extreme values may cause numerical instability
float x = Float.MAX_VALUE;
float pdf = distribution.pdf(x); // 可能返回0或NaN / May return 0 or NaN

// 解决方案：检查边界情况 / Solution: Check edge cases
if (Float.isInfinite(x) || Float.isNaN(x)) {
    return 0.0f; // 或适当的默认值 / or appropriate default value
}
```

### 3. 参数验证 / Parameter Validation

```java
// 问题：无效参数可能导致异常 / Issue: Invalid parameters may cause exceptions
NormalDistribution normal = new NormalDistribution(0.0f, -1.0f); // 负标准差 / Negative std

// 解决方案：添加参数验证 / Solution: Add parameter validation
if (stdDev <= 0) {
    throw new IllegalArgumentException("标准差必须大于0 / Standard deviation must be greater than 0");
}
```

---

**统计操作示例** - 从基础到高级，掌握概率分布的实际应用！

**Statistics Examples** - From basic to advanced, master the practical applications of probability distributions!
