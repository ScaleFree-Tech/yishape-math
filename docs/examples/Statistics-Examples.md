# 统计操作示例 - 系统性编排版 (Statistics Examples - Systematic Organization)

## 概述 / Overview

本文档按照从简单到复杂的顺序，系统性地编排了统计操作包的详细使用示例。每个级别都包含相应的理论背景、实践示例和进阶指导。

This document systematically organizes detailed usage examples for the statistics package in order from simple to complex. Each level includes corresponding theoretical background, practical examples, and advanced guidance.

---

## 第一部分：入门基础 (Level 1 - 基础入门)

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
        // 创建示例数据
        float[] data = {1.2f, 2.3f, 1.8f, 3.1f, 2.7f, 1.5f, 2.9f, 3.2f, 2.1f, 2.8f};
        IVector vector = IVector.of(data);
        
        // 计算基本统计量
        System.out.println("=== 基本统计量 ===");
        System.out.println("数据: " + Arrays.toString(data));
        System.out.println("均值: " + vector.mean());
        System.out.println("中位数: " + vector.median());
        System.out.println("标准差: " + vector.std());
        System.out.println("方差: " + vector.var());
        System.out.println("最小值: " + vector.min());
        System.out.println("最大值: " + vector.max());
        System.out.println("数据个数: " + vector.length());
    }
}
```

### 1.2 正态分布基础 / Normal Distribution Basics

```java
public class NormalDistributionBasicExample {
    public static void main(String[] args) {
        // 创建标准正态分布（均值为0，标准差为1）
        NormalDistribution standardNormal = Stat.norm();
        System.out.println("标准正态分布: " + standardNormal);
        
        // 基本统计量
        System.out.println("均值: " + standardNormal.mean());
        System.out.println("标准差: " + standardNormal.std());
        System.out.println("方差: " + standardNormal.var());
        
        // 概率密度函数
        System.out.println("\n概率密度函数值:");
        float[] xValues = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        for (float x : xValues) {
            System.out.printf("PDF(%.1f) = %.4f%n", x, standardNormal.pdf(x));
        }
        
        // 累积分布函数
        System.out.println("\n累积分布函数值:");
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
        // 创建均匀分布 [0, 1]
        UniformDistribution uniform = Stat.uniform(0.0f, 1.0f);
        System.out.println("均匀分布[0,1]: " + uniform);
        
        // 基本统计量
        System.out.println("均值: " + uniform.mean());
        System.out.println("标准差: " + uniform.std());
        System.out.println("方差: " + uniform.var());
        
        // 概率计算
        System.out.println("\n概率计算:");
        System.out.println("P(X ≤ 0.5) = " + uniform.cdf(0.5f));
        System.out.println("P(0.3 ≤ X ≤ 0.7) = " + (uniform.cdf(0.7f) - uniform.cdf(0.3f)));
        
        // 分位数
        System.out.println("\n分位数:");
        System.out.println("50%分位数: " + uniform.ppf(0.5f));
        System.out.println("90%分位数: " + uniform.ppf(0.9f));
    }
}
```

---

## 第二部分：基础应用 (Level 2 - 基础应用)

### 2.1 正态分布应用 / Normal Distribution Applications

```java
public class NormalDistributionApplicationExample {
    public static void main(String[] args) {
        // 创建自定义正态分布（均值=10，标准差=2）
        NormalDistribution normal = Stat.norm(10.0f, 2.0f);
        System.out.println("正态分布(μ=10, σ=2): " + normal);
        
        // 基本统计量
        System.out.println("均值: " + normal.mean());
        System.out.println("标准差: " + normal.std());
        System.out.println("方差: " + normal.var());
        
        // 概率计算
        System.out.println("\n概率计算:");
        System.out.println("P(X ≤ 12) = " + normal.cdf(12.0f));
        System.out.println("P(X > 8) = " + normal.sf(8.0f));
        System.out.println("P(8 ≤ X ≤ 12) = " + (normal.cdf(12.0f) - normal.cdf(8.0f)));
        
        // 分位数计算
        System.out.println("\n分位数:");
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
        // 正态分布采样
        NormalDistribution normal = Stat.norm(5.0f, 2.0f);
        
        // 生成单个随机样本
        float sample = normal.sample();
        System.out.println("单个随机样本: " + sample);
        
        // 生成多个随机样本
        float[] samples = normal.sample(1000);
        System.out.println("生成了 " + samples.length + " 个随机样本");
        
        // 计算样本统计量
        IVector sampleVector = IVector.of(samples);
        System.out.println("\n样本统计量:");
        System.out.printf("样本均值: %.4f (理论值: %.4f)%n", sampleVector.mean(), normal.mean());
        System.out.printf("样本标准差: %.4f (理论值: %.4f)%n", sampleVector.std(), normal.std());
        System.out.printf("样本方差: %.4f (理论值: %.4f)%n", sampleVector.var(), normal.var());
    }
}
```

### 2.3 指数分布基础 / Exponential Distribution Basics

```java
public class ExponentialDistributionBasicExample {
    public static void main(String[] args) {
        // 创建指数分布（λ = 0.5）
        ExponentialDistribution exp = Stat.exponential(0.5f);
        System.out.println("指数分布(λ=0.5): " + exp);
        
        // 基本统计量
        System.out.println("均值: " + exp.mean());
        System.out.println("标准差: " + exp.std());
        System.out.println("方差: " + exp.var());
        
        // 概率计算
        System.out.println("\n概率计算:");
        float[] times = {1.0f, 2.0f, 3.0f, 5.0f, 10.0f};
        for (float t : times) {
            System.out.printf("P(X ≤ %.1f) = %.4f%n", t, exp.cdf(t));
            System.out.printf("P(X > %.1f) = %.4f%n", t, exp.sf(t));
        }
        
        // 生存函数（可靠性分析）
        System.out.println("\n生存函数（设备可靠性）:");
        for (float t : times) {
            System.out.printf("R(%.1f) = P(寿命 > %.1f) = %.4f%n", t, t, exp.sf(t));
        }
    }
}
```

---

## 第三部分：中级应用 (Level 3 - 中级应用)

### 3.1 t分布应用 / t-Distribution Applications

```java
public class TDistributionApplicationExample {
    public static void main(String[] args) {
        // 创建不同自由度的t分布
        StudentDistribution t5 = Stat.t(5.0f);
        StudentDistribution t10 = Stat.t(10.0f);
        StudentDistribution t30 = Stat.t(30.0f);
        
        System.out.println("=== t分布比较 ===");
        System.out.println("t分布(5自由度): " + t5);
        System.out.println("t分布(10自由度): " + t10);
        System.out.println("t分布(30自由度): " + t30);
        
        // 统计量比较
        System.out.println("\n统计量比较:");
        System.out.printf("自由度5 - 均值: %.4f, 标准差: %.4f%n", t5.mean(), t5.std());
        System.out.printf("自由度10 - 均值: %.4f, 标准差: %.4f%n", t10.mean(), t10.std());
        System.out.printf("自由度30 - 均值: %.4f, 标准差: %.4f%n", t30.mean(), t30.std());
        
        // 临界值计算（用于假设检验）
        System.out.println("\n临界值计算（95%置信水平）:");
        float alpha = 0.05f;
        System.out.printf("t(5)临界值: %.4f%n", t5.ppf(1.0f - alpha/2.0f));
        System.out.printf("t(10)临界值: %.4f%n", t10.ppf(1.0f - alpha/2.0f));
        System.out.printf("t(30)临界值: %.4f%n", t30.ppf(1.0f - alpha/2.0f));
    }
}
```

### 3.2 卡方分布应用 / Chi-Squared Distribution Applications

```java
public class ChiSquaredDistributionApplicationExample {
    public static void main(String[] args) {
        // 创建不同自由度的卡方分布
        Chi2Distribution chi2_1 = Stat.chi2(1.0f);
        Chi2Distribution chi2_5 = Stat.chi2(5.0f);
        Chi2Distribution chi2_10 = Stat.chi2(10.0f);
        
        System.out.println("=== 卡方分布比较 ===");
        System.out.println("卡方分布(1自由度): " + chi2_1);
        System.out.println("卡方分布(5自由度): " + chi2_5);
        System.out.println("卡方分布(10自由度): " + chi2_10);
        
        // 统计量比较
        System.out.println("\n统计量比较:");
        System.out.printf("自由度1 - 均值: %.4f, 标准差: %.4f%n", chi2_1.mean(), chi2_1.std());
        System.out.printf("自由度5 - 均值: %.4f, 标准差: %.4f%n", chi2_5.mean(), chi2_5.std());
        System.out.printf("自由度10 - 均值: %.4f, 标准差: %.4f%n", chi2_10.mean(), chi2_10.std());
        
        // 临界值计算
        System.out.println("\n临界值计算（95%置信水平）:");
        float alpha = 0.05f;
        System.out.printf("χ²(1)临界值: %.4f%n", chi2_1.ppf(1.0f - alpha));
        System.out.printf("χ²(5)临界值: %.4f%n", chi2_5.ppf(1.0f - alpha));
        System.out.printf("χ²(10)临界值: %.4f%n", chi2_10.ppf(1.0f - alpha));
    }
}
```

### 3.3 F分布应用 / F-Distribution Applications

```java
public class FDistributionApplicationExample {
    public static void main(String[] args) {
        // 创建不同自由度的F分布
        FDistribution f1 = Stat.f(5.0f, 10.0f);
        FDistribution f2 = Stat.f(10.0f, 5.0f);
        FDistribution f3 = Stat.f(20.0f, 20.0f);
        
        System.out.println("=== F分布比较 ===");
        System.out.println("F分布(5,10自由度): " + f1);
        System.out.println("F分布(10,5自由度): " + f2);
        System.out.println("F分布(20,20自由度): " + f3);
        
        // 统计量比较
        System.out.println("\n统计量比较:");
        System.out.printf("F(5,10) - 均值: %.4f%n", f1.mean());
        System.out.printf("F(10,5) - 均值: %.4f%n", f2.mean());
        System.out.printf("F(20,20) - 均值: %.4f%n", f3.mean());
        
        // 临界值计算
        System.out.println("\n临界值计算（95%置信水平）:");
        float alpha = 0.05f;
        System.out.printf("F(5,10)临界值: %.4f%n", f1.ppf(1.0f - alpha));
        System.out.printf("F(10,5)临界值: %.4f%n", f2.ppf(1.0f - alpha));
        System.out.printf("F(20,20)临界值: %.4f%n", f3.ppf(1.0f - alpha));
    }
}
```

### 3.4 泊松分布应用 / Poisson Distribution Applications

```java
public class PoissonDistributionApplicationExample {
    public static void main(String[] args) {
        // 创建泊松分布（λ = 3）
        PoissonDistribution poisson = Stat.poisson(3.0f);
        System.out.println("泊松分布(λ=3): " + poisson);
        
        // 基本统计量
        System.out.println("均值: " + poisson.mean());
        System.out.println("方差: " + poisson.var());
        System.out.println("标准差: " + poisson.std());
        
        // 概率质量函数
        System.out.println("\n概率质量函数:");
        for (int k = 0; k <= 10; k++) {
            System.out.printf("P(X = %d) = %.4f%n", k, poisson.pmf(k));
        }
        
        // 累积分布函数
        System.out.println("\n累积分布函数:");
        for (int k = 0; k <= 10; k++) {
            System.out.printf("P(X ≤ %d) = %.4f%n", k, poisson.cdf(k));
        }
        
        // 随机采样
        System.out.println("\n随机采样:");
        int[] samples = new int[1000];
        for (int i = 0; i < 1000; i++) {
            samples[i] = (int) poisson.sample();
        }
        
        // 计算样本统计量
        float sampleMean = 0;
        for (int sample : samples) {
            sampleMean += sample;
        }
        sampleMean /= samples.length;
        
        System.out.printf("样本均值: %.4f (理论值: %.4f)%n", sampleMean, poisson.mean());
    }
}
```

---

## 第四部分：高级应用 (Level 4 - 高级应用)

### 4.1 参数估计 / Parameter Estimation

```java
public class ParameterEstimationExample {
    public static void main(String[] args) {
        // 生成样本数据
        NormalDistribution trueDist = Stat.norm(10.0f, 2.0f);
        float[] samples = trueDist.sample(100);
        IVector sampleVector = IVector.of(samples);
        
        System.out.println("=== 参数估计示例 ===");
        System.out.println("真实参数: μ=10.0, σ=2.0");
        System.out.println("样本大小: " + sampleVector.length());
        System.out.println("样本均值: " + sampleVector.mean());
        System.out.println("样本标准差: " + sampleVector.std());
        
        // 参数估计
        ParameterEstimation estimator = new ParameterEstimation();
        
        // 均值置信区间估计（使用t分布）
        Tuple2<Float, Float> meanCI = estimator.estimateMeanIntevalWithT(sampleVector, 0.95f);
        System.out.println("\n均值95%置信区间: [" + meanCI._1 + ", " + meanCI._2 + "]");
        
        // 方差置信区间估计（使用卡方分布）
        Tuple2<Float, Float> varCI = estimator.estimateVarIntevalWithChi2(sampleVector, 0.95f);
        System.out.println("方差95%置信区间: [" + varCI._1 + ", " + varCI._2 + "]");
        
        // 检查真实参数是否在置信区间内
        System.out.println("\n参数检验:");
        System.out.println("真实均值在置信区间内: " + (meanCI._1 <= 10.0f && 10.0f <= meanCI._2));
        System.out.println("真实方差在置信区间内: " + (varCI._1 <= 4.0f && 4.0f <= varCI._2));
    }
}
```

### 4.2 假设检验 / Hypothesis Testing

```java
public class HypothesisTestingExample {
    public static void main(String[] args) {
        // 生成样本数据
        NormalDistribution trueDist = Stat.norm(10.0f, 2.0f);
        float[] samples = trueDist.sample(50);
        IVector sampleVector = IVector.of(samples);
        
        System.out.println("=== 假设检验示例 ===");
        System.out.println("样本数据:");
        System.out.println("  样本均值: " + sampleVector.mean());
        System.out.println("  样本标准差: " + sampleVector.std());
        
        // 假设检验
        HypothesisTesting tester = new HypothesisTesting();
        
        // 检验均值是否等于10.0
        TestingResult meanTest = tester.testMeanEqualWithT(10.0f, sampleVector, 0.95f);
        System.out.println("\n均值检验 (H0: μ = 10.0):");
        System.out.println("  检验结果: " + (meanTest.pass ? "接受原假设" : "拒绝原假设"));
        System.out.println("  p值: " + meanTest.p);
        System.out.println("  置信区间: [" + meanTest.criticalInteval._1 + ", " + meanTest.criticalInteval._2 + "]");
        
        // 检验方差是否等于4.0
        TestingResult varTest = tester.testVarEqualWithChi2(4.0f, sampleVector, 0.95f);
        System.out.println("\n方差检验 (H0: σ² = 4.0):");
        System.out.println("  检验结果: " + (varTest.pass ? "接受原假设" : "拒绝原假设"));
        System.out.println("  p值: " + varTest.p);
        System.out.println("  置信区间: [" + varTest.criticalInteval._1 + ", " + varTest.criticalInteval._2 + "]");
    }
}
```

### 4.3 二项分布应用 / Binomial Distribution Applications

```java
public class BinomialDistributionApplicationExample {
    public static void main(String[] args) {
        // 创建二项分布（n=100, p=0.3）
        BinomialDistribution binomial = Stat.binomial(100, 0.3f);
        System.out.println("二项分布(n=100, p=0.3): " + binomial);
        
        // 基本统计量
        System.out.println("均值: " + binomial.mean());
        System.out.println("方差: " + binomial.var());
        System.out.println("标准差: " + binomial.std());
        
        // 概率质量函数
        System.out.println("\n概率质量函数:");
        for (int k = 20; k <= 40; k += 5) {
            System.out.printf("P(X = %d) = %.4f%n", k, binomial.pmf(k));
        }
        
        // 累积分布函数
        System.out.println("\n累积分布函数:");
        for (int k = 20; k <= 40; k += 5) {
            System.out.printf("P(X ≤ %d) = %.4f%n", k, binomial.cdf(k));
        }
        
        // 随机采样
        System.out.println("\n随机采样:");
        int[] samples = new int[1000];
        for (int i = 0; i < 1000; i++) {
            samples[i] = (int) binomial.sample();
        }
        
        // 计算样本统计量
        float sampleMean = 0;
        for (int sample : samples) {
            sampleMean += sample;
        }
        sampleMean /= samples.length;
        
        System.out.printf("样本均值: %.4f (理论值: %.4f)%n", sampleMean, binomial.mean());
    }
}
```

---

## 第五部分：专业应用 (Level 5 - 专业应用)

### 5.1 质量控制分析 / Quality Control Analysis

```java
public class QualityControlAnalysisExample {
    public static void main(String[] args) {
        System.out.println("=== 质量控制统计分析示例 ===");
        
        // 模拟生产过程中的产品重量数据
        // 假设产品重量应该服从正态分布，均值为100g，标准差为2g
        NormalDistribution targetDist = Stat.norm(100.0f, 2.0f);
        
        // 生成样本数据（模拟实际测量结果）
        float[] measurements = targetDist.sample(50);
        IVector sample = IVector.of(measurements);
        
        System.out.println("产品重量测量数据:");
        System.out.println("  样本大小: " + sample.length());
        System.out.println("  样本均值: " + sample.mean());
        System.out.println("  样本标准差: " + sample.std());
        System.out.println("  最小值: " + sample.min());
        System.out.println("  最大值: " + sample.max());
        
        // 参数估计
        ParameterEstimation estimator = new ParameterEstimation();
        Tuple2<Float, Float> meanCI = estimator.estimateMeanIntevalWithT(sample, 0.95f);
        Tuple2<Float, Float> varCI = estimator.estimateVarIntevalWithChi2(sample, 0.95f);
        
        System.out.println("\n参数估计结果:");
        System.out.println("  均值95%置信区间: [" + meanCI._1 + ", " + meanCI._2 + "]");
        System.out.println("  方差95%置信区间: [" + varCI._1 + ", " + varCI._2 + "]");
        
        // 假设检验
        HypothesisTesting tester = new HypothesisTesting();
        
        // 检验均值是否等于目标值100g
        TestingResult meanTest = tester.testMeanEqualWithT(100.0f, sample, 0.95f);
        System.out.println("\n均值检验 (H0: μ = 100g):");
        System.out.println("  检验结果: " + (meanTest.pass ? "✓ 生产过程正常" : "✗ 生产过程异常"));
        System.out.println("  p值: " + meanTest.p);
        
        // 检验方差是否等于目标值4.0
        TestingResult varTest = tester.testVarEqualWithChi2(4.0f, sample, 0.95f);
        System.out.println("\n方差检验 (H0: σ² = 4.0):");
        System.out.println("  检验结果: " + (varTest.pass ? "✓ 过程稳定性正常" : "✗ 过程稳定性异常"));
        System.out.println("  p值: " + varTest.p);
        
        // 计算不合格品概率
        NormalDistribution estimatedDist = Stat.norm(sample.mean(), sample.std());
        float lowerLimit = 95.0f;  // 下限
        float upperLimit = 105.0f; // 上限
        
        float probBelowLower = estimatedDist.cdf(lowerLimit);
        float probAboveUpper = 1.0f - estimatedDist.cdf(upperLimit);
        float defectProb = probBelowLower + probAboveUpper;
        
        System.out.println("\n质量分析:");
        System.out.println("  不合格品概率: " + (defectProb * 100) + "%");
        System.out.println("  低于下限概率: " + (probBelowLower * 100) + "%");
        System.out.println("  高于上限概率: " + (probAboveUpper * 100) + "%");
    }
}
```

### 5.2 金融风险评估 / Financial Risk Assessment

```java
public class FinancialRiskAssessmentExample {
    public static void main(String[] args) {
        System.out.println("=== 金融风险评估示例 ===");
        
        // 模拟股票收益率数据（假设服从正态分布）
        NormalDistribution returnDist = Stat.norm(0.001f, 0.02f); // 日收益率：均值0.1%，标准差2%
        float[] dailyReturns = returnDist.sample(252); // 一年的交易日数据
        IVector returns = IVector.of(dailyReturns);
        
        System.out.println("股票收益率统计:");
        System.out.println("  样本大小: " + returns.length());
        System.out.println("  平均日收益率: " + (returns.mean() * 100) + "%");
        System.out.println("  收益率标准差: " + (returns.std() * 100) + "%");
        System.out.println("  最小日收益率: " + (returns.min() * 100) + "%");
        System.out.println("  最大日收益率: " + (returns.max() * 100) + "%");
        
        // 计算VaR (Value at Risk)
        float[] sortedReturns = dailyReturns.clone();
        Arrays.sort(sortedReturns);
        
        int var95Index = (int) (sortedReturns.length * 0.05);
        int var99Index = (int) (sortedReturns.length * 0.01);
        
        float var95 = sortedReturns[var95Index];
        float var99 = sortedReturns[var99Index];
        
        System.out.println("\n风险度量:");
        System.out.println("  VaR 95%: " + (var95 * 100) + "%");
        System.out.println("  VaR 99%: " + (var99 * 100) + "%");
        
        // 使用正态分布计算理论VaR
        NormalDistribution estimatedDist = Stat.norm(returns.mean(), returns.std());
        float theoreticalVar95 = estimatedDist.ppf(0.05f);
        float theoreticalVar99 = estimatedDist.ppf(0.01f);
        
        System.out.println("\n理论VaR (基于正态分布):");
        System.out.println("  VaR 95%: " + (theoreticalVar95 * 100) + "%");
        System.out.println("  VaR 99%: " + (theoreticalVar99 * 100) + "%");
        
        // 计算最大回撤
        float maxDrawdown = calculateMaxDrawdown(dailyReturns);
        System.out.println("\n最大回撤: " + (maxDrawdown * 100) + "%");
        
        // 计算夏普比率
        float riskFreeRate = 0.03f / 252; // 年化无风险利率3%，转换为日利率
        float sharpeRatio = (returns.mean() - riskFreeRate) / returns.std();
        System.out.println("夏普比率: " + sharpeRatio);
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
        System.out.println("=== 可靠性分析示例 ===");
        
        // 模拟设备故障时间数据（假设服从指数分布）
        ExponentialDistribution failureDist = Stat.exponential(0.01f); // 故障率λ=0.01/小时
        float[] failureTimes = failureDist.sample(100); // 100个故障时间样本
        IVector failures = IVector.of(failureTimes);
        
        System.out.println("设备故障时间统计:");
        System.out.println("  样本大小: " + failures.length());
        System.out.println("  平均故障时间: " + failures.mean() + " 小时");
        System.out.println("  故障时间标准差: " + failures.std() + " 小时");
        System.out.println("  最短故障时间: " + failures.min() + " 小时");
        System.out.println("  最长故障时间: " + failures.max() + " 小时");
        
        // 计算可靠性指标
        float estimatedLambda = 1.0f / failures.mean();
        System.out.println("\n可靠性指标:");
        System.out.println("  估计故障率: " + estimatedLambda + " /小时");
        System.out.println("  平均故障间隔时间(MTBF): " + failures.mean() + " 小时");
        
        // 计算不同时间点的可靠性
        float[] timePoints = {10, 50, 100, 200, 500};
        System.out.println("\n可靠性函数 R(t) = e^(-λt):");
        for (float t : timePoints) {
            float reliability = (float) Math.exp(-estimatedLambda * t);
            System.out.println("  R(" + t + ") = " + reliability);
        }
        
        // 计算故障概率密度
        System.out.println("\n故障概率密度 f(t) = λe^(-λt):");
        for (float t : timePoints) {
            float density = estimatedLambda * (float) Math.exp(-estimatedLambda * t);
            System.out.println("  f(" + t + ") = " + density);
        }
        
        // 计算累积故障概率
        System.out.println("\n累积故障概率 F(t) = 1 - e^(-λt):");
        for (float t : timePoints) {
            float cumulativeProb = 1.0f - (float) Math.exp(-estimatedLambda * t);
            System.out.println("  F(" + t + ") = " + cumulativeProb);
        }
        
        // 计算中位故障时间
        float medianTime = (float) (Math.log(2) / estimatedLambda);
        System.out.println("\n中位故障时间: " + medianTime + " 小时");
    }
}
```

### 5.4 实验设计分析 / Experimental Design Analysis

```java
public class ExperimentalDesignAnalysisExample {
    public static void main(String[] args) {
        System.out.println("=== 实验设计分析示例 ===");
        
        // 模拟A/B测试数据
        // 对照组：转化率5%
        BinomialDistribution controlDist = Stat.binomial(1000, 0.05f);
        int controlConversions = (int) controlDist.sample();
        
        // 实验组：转化率6%
        BinomialDistribution treatmentDist = Stat.binomial(1000, 0.06f);
        int treatmentConversions = (int) treatmentDist.sample();
        
        System.out.println("A/B测试结果:");
        System.out.println("  对照组转化数: " + controlConversions + "/1000");
        System.out.println("  实验组转化数: " + treatmentConversions + "/1000");
        System.out.println("  对照组转化率: " + (controlConversions / 1000.0f * 100) + "%");
        System.out.println("  实验组转化率: " + (treatmentConversions / 1000.0f * 100) + "%");
        
        // 计算转化率差异
        float controlRate = controlConversions / 1000.0f;
        float treatmentRate = treatmentConversions / 1000.0f;
        float rateDifference = treatmentRate - controlRate;
        
        System.out.println("\n转化率差异: " + (rateDifference * 100) + "%");
        
        // 计算置信区间
        float controlStdError = (float) Math.sqrt(controlRate * (1 - controlRate) / 1000);
        float treatmentStdError = (float) Math.sqrt(treatmentRate * (1 - treatmentRate) / 1000);
        float diffStdError = (float) Math.sqrt(controlStdError * controlStdError + treatmentStdError * treatmentStdError);
        
        // 使用正态分布近似
        NormalDistribution diffDist = Stat.norm(rateDifference, diffStdError);
        float lowerBound = diffDist.ppf(0.025f);
        float upperBound = diffDist.ppf(0.975f);
        
        System.out.println("\n转化率差异95%置信区间:");
        System.out.println("  [" + (lowerBound * 100) + "%, " + (upperBound * 100) + "%]");
        
        // 假设检验
        HypothesisTesting tester = new HypothesisTesting();
        
        // 创建样本数据用于检验
        float[] controlData = new float[1000];
        float[] treatmentData = new float[1000];
        
        for (int i = 0; i < 1000; i++) {
            controlData[i] = controlDist.sample();
            treatmentData[i] = treatmentDist.sample();
        }
        
        IVector controlVector = IVector.of(controlData);
        IVector treatmentVector = IVector.of(treatmentData);
        
        // 检验两组均值是否相等
        TestingResult meanTest = tester.testMeanEqualWithT(0.0f, 
            IVector.of(new float[]{rateDifference}), 0.95f);
        
        System.out.println("\n假设检验 (H0: 转化率差异 = 0):");
        System.out.println("  检验结果: " + (meanTest.pass ? "接受原假设" : "拒绝原假设"));
        System.out.println("  p值: " + meanTest.p);
        
        // 计算统计功效
        float effectSize = Math.abs(rateDifference) / diffStdError;
        System.out.println("\n效应量 (Cohen's d): " + effectSize);
        
        if (effectSize < 0.2) {
            System.out.println("  效应量: 小");
        } else if (effectSize < 0.5) {
            System.out.println("  效应量: 中等");
        } else {
            System.out.println("  效应量: 大");
        }
    }
}
```

### 5.5 蒙特卡洛模拟 / Monte Carlo Simulation

```java
public class MonteCarloSimulationExample {
    public static void main(String[] args) {
        System.out.println("=== 蒙特卡洛模拟示例 ===");
        
        // 期权定价模拟
        simulateOptionPricing();
        
        // 风险评估模拟
        simulateRiskAssessment();
        
        // 质量控制模拟
        simulateQualityControl();
    }
    
    public static void simulateOptionPricing() {
        System.out.println("\n--- 期权定价蒙特卡洛模拟 ---");
        
        // 参数设置
        float S0 = 100.0f;      // 当前股价
        float K = 105.0f;       // 执行价格
        float r = 0.05f;        // 无风险利率
        float sigma = 0.2f;     // 波动率
        float T = 1.0f;         // 到期时间
        int nSimulations = 100000;
        
        // 创建正态分布
        NormalDistribution normal = Stat.norm();
        
        float[] payoffs = new float[nSimulations];
        
        // 蒙特卡洛模拟
        for (int i = 0; i < nSimulations; i++) {
            float z = normal.sample();
            float ST = S0 * (float) Math.exp((r - 0.5f * sigma * sigma) * T + sigma * (float) Math.sqrt(T) * z);
            payoffs[i] = Math.max(ST - K, 0.0f); // 看涨期权
        }
        
        // 计算期权价格
        float optionPrice = calculateMean(payoffs) * (float) Math.exp(-r * T);
        
        System.out.printf("模拟次数: %d%n", nSimulations);
        System.out.printf("期权价格: %.4f%n", optionPrice);
        System.out.printf("标准差: %.4f%n", calculateStd(payoffs));
    }
    
    public static void simulateRiskAssessment() {
        System.out.println("\n--- 风险评估蒙特卡洛模拟 ---");
        
        // 投资组合参数
        float[] weights = {0.4f, 0.3f, 0.3f};  // 权重
        float[] means = {0.08f, 0.12f, 0.06f}; // 预期收益率
        float[] stds = {0.15f, 0.25f, 0.10f};  // 标准差
        int nSimulations = 50000;
        int timeHorizon = 252; // 交易日
        
        // 创建正态分布
        NormalDistribution[] distributions = new NormalDistribution[3];
        for (int i = 0; i < 3; i++) {
            distributions[i] = Stat.norm(means[i], stds[i]);
        }
        
        float[] portfolioReturns = new float[nSimulations];
        
        // 蒙特卡洛模拟
        for (int sim = 0; sim < nSimulations; sim++) {
            float portfolioReturn = 0.0f;
            for (int i = 0; i < 3; i++) {
                float dailyReturn = distributions[i].sample() / (float) Math.sqrt(timeHorizon);
                portfolioReturn += weights[i] * dailyReturn;
            }
            portfolioReturns[sim] = portfolioReturn;
        }
        
        // 计算风险指标
        float meanReturn = calculateMean(portfolioReturns);
        float stdReturn = calculateStd(portfolioReturns);
        float var95 = calculatePercentile(portfolioReturns, 0.05f);
        float var99 = calculatePercentile(portfolioReturns, 0.01f);
        
        System.out.printf("预期收益率: %.4f%n", meanReturn);
        System.out.printf("收益率标准差: %.4f%n", stdReturn);
        System.out.printf("95%% VaR: %.4f%n", var95);
        System.out.printf("99%% VaR: %.4f%n", var99);
    }
    
    public static void simulateQualityControl() {
        System.out.println("\n--- 质量控制蒙特卡洛模拟 ---");
        
        // 质量控制参数
        float targetMean = 100.0f;     // 目标均值
        float targetStd = 5.0f;        // 目标标准差
        float specLower = 90.0f;       // 规格下限
        float specUpper = 110.0f;      // 规格上限
        int nSimulations = 100000;
        int sampleSize = 30;
        
        // 创建正态分布
        NormalDistribution normal = Stat.norm(targetMean, targetStd);
        
        int defectCount = 0;
        
        // 蒙特卡洛模拟
        for (int sim = 0; sim < nSimulations; sim++) {
            float[] sample = normal.sample(sampleSize);
            
            // 检查是否有不合格品
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
        System.out.printf("模拟次数: %d%n", nSimulations);
        System.out.printf("样本大小: %d%n", sampleSize);
        System.out.printf("不合格品率: %.4f%%%n", defectRate * 100);
    }
    
    // 辅助方法
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
1. 从第一部分开始，掌握基本统计量计算
2. 理解正态分布和均匀分布的基本概念
3. 学习简单的随机采样和概率计算

### 中级用户路径 / Intermediate Path
1. 掌握t分布、卡方分布、F分布等统计分布
2. 学习参数估计和假设检验方法
3. 理解不同分布的应用场景

### 高级用户路径 / Advanced Path
1. 掌握复杂的统计分析方法
2. 学习实际业务场景的应用
3. 理解统计推断的原理和实践

### 专业用户路径 / Professional Path
1. 掌握蒙特卡洛模拟等高级方法
2. 能够设计复杂的统计分析方案
3. 能够处理实际业务中的复杂统计问题

---

## 总结 / Summary

本文档按照从简单到复杂的顺序，系统性地介绍了统计操作包的各种功能。通过循序渐进的学习，您可以：

- **掌握基础**：从基本统计量开始，逐步建立统计基础
- **应用实践**：通过实际案例学习不同统计方法的使用场景
- **进阶提升**：掌握高级统计方法和专业应用
- **灵活运用**：根据实际需求选择合适的统计分析方法

This document systematically introduces various functions of the statistics package in order from simple to complex. Through progressive learning, you can:

- **Master the basics**: Start with basic statistics and gradually build statistical foundations
- **Apply in practice**: Learn usage scenarios of different statistical methods through real cases
- **Advance and improve**: Master advanced statistical methods and professional applications
- **Use flexibly**: Choose appropriate statistical analysis methods based on actual needs

---

**统计操作示例** - 让统计分析更简单！

**Statistics Examples** - Make statistical analysis simpler!
