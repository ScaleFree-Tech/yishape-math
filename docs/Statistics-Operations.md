# 统计操作 (Statistics Operations)

## 概述 / Overview

`Stat` 类和相关的概率分布类提供了完整的统计计算功能，包括各种概率分布的概率密度函数、累积分布函数、随机采样等。该模块支持正态分布、t分布、均匀分布、指数分布、卡方分布、F分布等常用统计分布。

The `Stat` class and related probability distribution classes provide comprehensive statistical computation functionality, including probability density functions, cumulative distribution functions, random sampling for various probability distributions. This module supports common statistical distributions such as normal distribution, t-distribution, uniform distribution, exponential distribution, chi-squared distribution, F-distribution, etc.

## 核心类 / Core Classes

### Stat 类 / Stat Class

`Stat` 是统计操作的核心工厂类，提供了创建各种概率分布对象的静态方法。

`Stat` is the core factory class for statistical operations, providing static methods to create various probability distribution objects.

## 主要功能 / Main Features

### 1. 概率分布创建 / Probability Distribution Creation

#### 正态分布 / Normal Distribution

```java
// 标准正态分布（均值为0，标准差为1）/ Standard normal distribution (mean=0, stdDev=1)
NormalDistribution standardNormal = Stat.norm();

// 自定义参数的正态分布 / Custom parameter normal distribution
NormalDistribution normal = Stat.norm(5.0f, 2.0f); // 均值=5，标准差=2
```

#### t分布 / Student's t-Distribution

```java
// 创建t分布 / Create t-distribution
StudentDistribution tDist = Stat.t(10.0f); // 自由度为10
```

#### 均匀分布 / Uniform Distribution

```java
// 创建均匀分布 / Create uniform distribution
UniformDistribution uniform = Stat.uniform(0.0f, 10.0f); // 区间[0, 10]
```

#### 指数分布 / Exponential Distribution

```java
// 创建指数分布 / Create exponential distribution
ExponentialDistribution exp = Stat.exponential(2.0f); // 速率参数=2
```

#### 卡方分布 / Chi-Squared Distribution

```java
// 创建卡方分布 / Create chi-squared distribution
Chi2Distribution chi2 = Stat.chi2(5.0f); // 自由度为5
```

#### F分布 / F-Distribution

```java
// 创建F分布 / Create F-distribution
FDistribution fDist = Stat.f(5.0f, 10.0f); // 分子自由度=5，分母自由度=10
```

### 2. 概率分布接口 / Probability Distribution Interface

所有连续概率分布都实现了 `IContinuousDistribution` 接口，提供统一的统计计算方法。

All continuous probability distributions implement the `IContinuousDistribution` interface, providing unified statistical computation methods.

#### 基本统计量 / Basic Statistics

```java
// 获取分布的基本统计量 / Get basic statistics of distribution
float mean = distribution.mean();           // 均值 / Mean
float variance = distribution.var();        // 方差 / Variance
float stdDev = distribution.std();          // 标准差 / Standard deviation
float median = distribution.median();       // 中位数 / Median
float mode = distribution.mode();           // 众数 / Mode
float q1 = distribution.q1();               // 第一四分位数 / First quartile
float q3 = distribution.q3();               // 第三四分位数 / Third quartile
float skewness = distribution.skewness();   // 偏度 / Skewness
float kurtosis = distribution.kurtosis();   // 峰度 / Kurtosis
```

#### 概率函数 / Probability Functions

```java
// 概率密度函数 / Probability Density Function
float pdfValue = distribution.pdf(x);

// 累积分布函数 / Cumulative Distribution Function
float cdfValue = distribution.cdf(x);

// 百分点函数（分位数函数）/ Percent Point Function (Quantile Function)
float ppfValue = distribution.ppf(probability);

// 生存函数 / Survival Function
float sfValue = distribution.sf(x);

// 逆生存函数 / Inverse Survival Function
float isfValue = distribution.isf(probability);
```

#### 随机采样 / Random Sampling

```java
// 生成单个随机样本 / Generate single random sample
float sample = distribution.sample();

// 生成多个随机样本 / Generate multiple random samples
float[] samples = distribution.sample(1000); // 生成1000个样本
```

### 3. 具体分布类详解 / Detailed Distribution Classes

#### 正态分布 (NormalDistribution)

正态分布是统计学中最重要的连续概率分布之一，也称为高斯分布。

Normal distribution is one of the most important continuous probability distributions in statistics, also known as Gaussian distribution.

**概率密度函数** / **Probability Density Function**:
```
f(x) = (1/σ√(2π)) * e^(-(x-μ)²/(2σ²))
```

**特性** / **Properties**:
- 均值: μ
- 方差: σ²
- 标准差: σ
- 偏度: 0（对称分布）
- 峰度: 0（超额峰度）

**应用场景** / **Application Scenarios**:
- 中心极限定理
- 假设检验
- 质量控制
- 金融建模

#### t分布 (StudentDistribution)

t分布用于小样本的统计推断，当总体方差未知时特别有用。

t-distribution is used for statistical inference with small samples, especially useful when population variance is unknown.

**特性** / **Properties**:
- 均值: 0（当自由度 > 1）
- 方差: df/(df-2)（当自由度 > 2）
- 自由度越大，越接近正态分布

**应用场景** / **Application Scenarios**:
- t检验
- 置信区间估计
- 回归分析

#### 均匀分布 (UniformDistribution)

均匀分布在指定区间内所有值出现的概率相等。

Uniform distribution has equal probability for all values within a specified interval.

**概率密度函数** / **Probability Density Function**:
```
f(x) = 1/(b-a) 当 a ≤ x ≤ b，否则为0
```

**特性** / **Properties**:
- 均值: (a+b)/2
- 方差: (b-a)²/12
- 偏度: 0（对称分布）
- 峰度: -1.2

**应用场景** / **Application Scenarios**:
- 随机数生成
- 蒙特卡洛模拟
- 质量控制

#### 指数分布 (ExponentialDistribution)

指数分布描述泊松过程中事件间的时间间隔。

Exponential distribution describes the time intervals between events in a Poisson process.

**概率密度函数** / **Probability Density Function**:
```
f(x) = λe^(-λx) 当 x ≥ 0，否则为0
```

**特性** / **Properties**:
- 均值: 1/λ
- 方差: 1/λ²
- 标准差: 1/λ
- 偏度: 2
- 峰度: 6

**应用场景** / **Application Scenarios**:
- 可靠性分析
- 排队论
- 生存分析

#### 卡方分布 (Chi2Distribution)

卡方分布是正态分布随机变量平方和的分布。

Chi-squared distribution is the distribution of the sum of squares of normal random variables.

**特性** / **Properties**:
- 均值: df（自由度）
- 方差: 2df
- 偏度: √(8/df)
- 峰度: 12/df

**应用场景** / **Application Scenarios**:
- 卡方检验
- 方差分析
- 拟合优度检验

#### F分布 (FDistribution)

F分布是两个独立卡方分布随机变量比值的分布。

F-distribution is the distribution of the ratio of two independent chi-squared random variables.

**特性** / **Properties**:
- 均值: d2/(d2-2)（当d2 > 2）
- 方差: 2d2²(d1+d2-2)/(d1(d2-2)²(d2-4))（当d2 > 4）

**应用场景** / **Application Scenarios**:
- F检验
- 方差分析
- 回归分析

## 使用示例 / Usage Examples

详细的代码示例请参考 [Statistics-Examples.md](examples/Statistics-Examples.md) 文档。

For detailed code examples, please refer to the [Statistics-Examples.md](examples/Statistics-Examples.md) document.

## 数值精度和算法 / Numerical Precision and Algorithms

### 正态分布算法 / Normal Distribution Algorithms

- **PDF计算** / **PDF Calculation**: 直接使用数学公式
- **CDF计算** / **CDF Calculation**: 使用误差函数的近似公式（Abramowitz and Stegun）
- **PPF计算** / **PPF Calculation**: 使用Beasley-Springer-Moro算法近似
- **随机采样** / **Random Sampling**: 使用Box-Muller变换

### 其他分布算法 / Other Distribution Algorithms

- **t分布** / **t-Distribution**: 使用数值积分和近似方法
- **均匀分布** / **Uniform Distribution**: 直接使用数学公式
- **指数分布** / **Exponential Distribution**: 使用逆变换采样
- **卡方分布** / **Chi-Squared Distribution**: 使用Gamma函数近似
- **F分布** / **F-Distribution**: 基于Beta分布实现

## 性能特性 / Performance Features

### 计算效率 / Computational Efficiency

- 高效的数值算法实现
- 预计算常量和查找表
- 优化的随机数生成算法

### 内存使用 / Memory Usage

- 最小化临时对象创建
- 高效的数组操作
- 智能的缓存策略

### 数值稳定性 / Numerical Stability

- 稳定的数值算法
- 边界情况处理
- 精度控制

## 注意事项 / Notes

1. **参数验证** / **Parameter Validation**: 所有分布类都会验证输入参数的有效性
2. **数值精度** / **Numerical Precision**: 使用float类型，注意精度限制
3. **边界情况** / **Edge Cases**: 特殊值（如无穷大、NaN）的处理
4. **随机性** / **Randomness**: 随机采样使用Java的Math.random()，可考虑使用更高质量的随机数生成器
5. **异常处理** / **Exception Handling**: 无效参数会抛出IllegalArgumentException

## 扩展性 / Extensibility

### 添加新分布 / Adding New Distributions

1. 实现 `IContinuousDistribution` 接口
2. 在 `Stat` 类中添加工厂方法
3. 添加相应的测试用例

### 自定义分布 / Custom Distributions

```java
public class CustomDistribution implements IContinuousDistribution {
    // 实现接口方法
    // Implement interface methods
}
```

## 应用场景 / Application Scenarios

### 统计分析 / Statistical Analysis

- 描述性统计
- 假设检验
- 置信区间估计
- 回归分析

### 数据科学 / Data Science

- 数据建模
- 蒙特卡洛模拟
- 风险评估
- 质量控制

### 机器学习 / Machine Learning

- 特征分布分析
- 数据预处理
- 模型验证
- 不确定性量化

### 金融工程 / Financial Engineering

- 风险建模
- 期权定价
- 投资组合优化
- 压力测试

## 与SciPy功能对照表 / SciPy Functionality Comparison Table

| 功能类别 / Function Category | Stat/分布类 / Stat/Distribution Classes | SciPy | 说明 / Description |
|---------|-------------------|-------|------|
| **分布创建 / Distribution Creation** | | | |
| 正态分布 / Normal distribution | `Stat.norm(mean, std)` | `scipy.stats.norm(mean, std)` | 创建正态分布 / Create normal distribution |
| t分布 / t-distribution | `Stat.t(df)` | `scipy.stats.t(df)` | 创建t分布 / Create t-distribution |
| 均匀分布 / Uniform distribution | `Stat.uniform(a, b)` | `scipy.stats.uniform(a, b-a)` | 创建均匀分布 / Create uniform distribution |
| 指数分布 / Exponential distribution | `Stat.exponential(rate)` | `scipy.stats.expon(scale=1/rate)` | 创建指数分布 / Create exponential distribution |
| 卡方分布 / Chi-squared distribution | `Stat.chi2(df)` | `scipy.stats.chi2(df)` | 创建卡方分布 / Create chi-squared distribution |
| F分布 / F-distribution | `Stat.f(df1, df2)` | `scipy.stats.f(df1, df2)` | 创建F分布 / Create F-distribution |
| **基本统计量 / Basic Statistics** | | | |
| 均值 / Mean | `dist.mean()` | `dist.mean()` | 分布均值 / Distribution mean |
| 方差 / Variance | `dist.var()` | `dist.var()` | 分布方差 / Distribution variance |
| 标准差 / Standard deviation | `dist.std()` | `dist.std()` | 分布标准差 / Distribution standard deviation |
| 中位数 / Median | `dist.median()` | `dist.median()` | 分布中位数 / Distribution median |
| 众数 / Mode | `dist.mode()` | `dist.mode()` | 分布众数 / Distribution mode |
| 偏度 / Skewness | `dist.skewness()` | `dist.stats(moments='s')` | 分布偏度 / Distribution skewness |
| 峰度 / Kurtosis | `dist.kurtosis()` | `dist.stats(moments='k')` | 分布峰度 / Distribution kurtosis |
| **概率函数 / Probability Functions** | | | |
| 概率密度函数 / PDF | `dist.pdf(x)` | `dist.pdf(x)` | 概率密度函数 / Probability density function |
| 累积分布函数 / CDF | `dist.cdf(x)` | `dist.cdf(x)` | 累积分布函数 / Cumulative distribution function |
| 百分点函数 / PPF | `dist.ppf(p)` | `dist.ppf(p)` | 百分点函数 / Percent point function |
| 生存函数 / Survival function | `dist.sf(x)` | `dist.sf(x)` | 生存函数 / Survival function |
| 逆生存函数 / Inverse survival function | `dist.isf(p)` | `dist.isf(p)` | 逆生存函数 / Inverse survival function |
| **随机采样 / Random Sampling** | | | |
| 单个样本 / Single sample | `dist.sample()` | `dist.rvs()` | 生成单个随机样本 / Generate single random sample |
| 多个样本 / Multiple samples | `dist.sample(n)` | `dist.rvs(n)` | 生成多个随机样本 / Generate multiple random samples |
| **分位数 / Quantiles** | | | |
| 第一四分位数 / First quartile | `dist.q1()` | `dist.ppf(0.25)` | 25%分位数 / 25th percentile |
| 第三四分位数 / Third quartile | `dist.q3()` | `dist.ppf(0.75)` | 75%分位数 / 75th percentile |
| 自定义分位数 / Custom quantile | `dist.ppf(p)` | `dist.ppf(p)` | 自定义概率的分位数 / Quantile for custom probability |

## 最佳实践建议 / Best Practices Recommendations

### 选择合适分布 / Choosing Appropriate Distributions

1. **正态分布** / **Normal Distribution**: 适用于中心极限定理、对称数据
2. **t分布** / **t-Distribution**: 适用于小样本、未知方差
3. **均匀分布** / **Uniform Distribution**: 适用于等概率事件
4. **指数分布** / **Exponential Distribution**: 适用于等待时间、寿命数据
5. **卡方分布** / **Chi-Squared Distribution**: 适用于方差检验、拟合优度
6. **F分布** / **F-Distribution**: 适用于方差比较、回归分析

### 数值计算建议 / Numerical Computation Recommendations

1. **精度控制** / **Precision Control**: 注意float类型的精度限制
2. **边界处理** / **Boundary Handling**: 正确处理极值和特殊值
3. **参数验证** / **Parameter Validation**: 确保输入参数的有效性
4. **异常处理** / **Exception Handling**: 适当处理可能的异常情况

### 性能优化建议 / Performance Optimization Recommendations

1. **批量采样** / **Batch Sampling**: 使用`sample(n)`而不是多次调用`sample()`
2. **缓存结果** / **Cache Results**: 对于重复计算，考虑缓存结果
3. **内存管理** / **Memory Management**: 及时释放不需要的大数组
4. **算法选择** / **Algorithm Selection**: 根据精度要求选择合适的算法

---

**统计操作** - 概率论与数理统计的Java实现，让数据分析更简单！

**Statistics Operations** - Java implementation of probability theory and mathematical statistics, making data analysis simpler!
