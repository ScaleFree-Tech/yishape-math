# Stat类 - 统计分布工厂 / Stat Class - Statistical Distribution Factory

## 简介 / Introduction

Stat类是YishapeMath库中的核心统计类，提供了创建各种常用概率分布对象的静态工厂方法。该类是统计学计算的主要入口点，支持六种重要的概率分布，为统计分析和数据科学应用提供了强大的基础。

The Stat class is the core statistical class in the YishapeMath library, providing static factory methods for creating various common probability distribution objects. This class serves as the main entry point for statistical calculations, supporting six important probability distributions and providing a powerful foundation for statistical analysis and data science applications.

## 特性 / Features

### 支持的分布类型 / Supported Distribution Types

| 分布类型 / Distribution Type | 中文名称 / Chinese Name | 工厂方法 / Factory Method | 主要应用 / Main Applications |
|---------------------------|----------------------|-------------------------|---------------------------|
| Normal Distribution | 正态分布 | `Stat.norm(mean, stdDev)` | 假设检验、置信区间、质量控制 |
| Student's t-Distribution | t分布 | `Stat.t(degreesOfFreedom)` | 小样本推断、t检验 |
| Uniform Distribution | 均匀分布 | `Stat.uniform(lower, upper)` | 随机数生成、蒙特卡洛模拟 |
| Exponential Distribution | 指数分布 | `Stat.exponential(rate)` | 可靠性分析、排队论 |
| Chi-Squared Distribution | 卡方分布 | `Stat.chi2(degreesOfFreedom)` | 拟合优度检验、独立性检验 |
| F-Distribution | F分布 | `Stat.f(df1, df2)` | 方差分析、F检验 |

### 核心功能 / Core Features

- **静态工厂方法** - 简洁的API设计，易于使用
- **完整的统计功能** - 支持PDF、CDF、分位数、统计量计算
- **随机采样** - 高效的随机数生成
- **数值稳定性** - 使用经过验证的数值算法
- **中英文文档** - 完整的中英文JavaDoc注释
- **类型安全** - 强类型参数验证

- **Static Factory Methods** - Clean API design, easy to use
- **Complete Statistical Functions** - Support for PDF, CDF, quantiles, statistical measures
- **Random Sampling** - Efficient random number generation
- **Numerical Stability** - Using proven numerical algorithms
- **Bilingual Documentation** - Complete Chinese and English JavaDoc comments
- **Type Safety** - Strong type parameter validation

## 快速开始 / Quick Start

### 基本用法 / Basic Usage

```java
import com.reremouse.lab.math.stat.Stat;
import com.reremouse.lab.math.stat.distribution.*;

public class StatExample {
    public static void main(String[] args) {
        // 创建标准正态分布
        // Create standard normal distribution
        NormalDistribution stdNormal = Stat.norm();
        
        // 创建自定义正态分布
        // Create custom normal distribution
        NormalDistribution normal = Stat.norm(5.0f, 2.0f);
        
        // 创建t分布
        // Create t-distribution
        StudentDistribution tDist = Stat.t(10.0f);
        
        // 创建均匀分布
        // Create uniform distribution
        UniformDistribution uniform = Stat.uniform(0.0f, 10.0f);
        
        // 创建指数分布
        // Create exponential distribution
        ExponentialDistribution exp = Stat.exponential(1.5f);
        
        // 创建卡方分布
        // Create chi-squared distribution
        Chi2Distribution chi2 = Stat.chi2(5.0f);
        
        // 创建F分布
        // Create F-distribution
        FDistribution fDist = Stat.f(3.0f, 10.0f);
    }
}
```

### 统计计算示例 / Statistical Calculation Examples

```java
// 正态分布计算
// Normal distribution calculations
NormalDistribution normal = Stat.norm(10.0f, 3.0f);

// 概率密度函数
// Probability density function
float pdf = normal.pdf(10.0f);

// 累积分布函数
// Cumulative distribution function
float cdf = normal.cdf(12.0f);

// 分位数函数
// Quantile function
float quantile = normal.ppf(0.95f);

// 统计量
// Statistical measures
float mean = normal.mean();
float variance = normal.var();
float stdDev = normal.std();

// 随机采样
// Random sampling
float sample = normal.sample();
float[] samples = normal.sample(1000);
```

## API参考 / API Reference

### 正态分布 / Normal Distribution

#### `Stat.norm(float mean, float stdDev)`
创建指定均值和标准差的正态分布。

Creates a normal distribution with specified mean and standard deviation.

**参数 / Parameters:**
- `mean` - 均值 / Mean
- `stdDev` - 标准差，必须大于0 / Standard deviation, must be greater than 0

**返回 / Returns:**
- `NormalDistribution` - 正态分布对象 / Normal distribution object

**异常 / Throws:**
- `IllegalArgumentException` - 如果标准差小于等于0 / If standard deviation ≤ 0

#### `Stat.norm()`
创建标准正态分布（均值为0，标准差为1）。

Creates a standard normal distribution (mean=0, stdDev=1).

**返回 / Returns:**
- `NormalDistribution` - 标准正态分布对象 / Standard normal distribution object

### t分布 / t-Distribution

#### `Stat.t(float degreesOfFreedom)`
创建指定自由度的t分布。

Creates a t-distribution with specified degrees of freedom.

**参数 / Parameters:**
- `degreesOfFreedom` - 自由度，必须大于0 / Degrees of freedom, must be greater than 0

**返回 / Returns:**
- `StudentDistribution` - t分布对象 / t-distribution object

**异常 / Throws:**
- `IllegalArgumentException` - 如果自由度小于等于0 / If degrees of freedom ≤ 0

### 均匀分布 / Uniform Distribution

#### `Stat.uniform(float lowerBound, float upperBound)`
创建指定区间的均匀分布。

Creates a uniform distribution over the specified interval.

**参数 / Parameters:**
- `lowerBound` - 下界 / Lower bound
- `upperBound` - 上界，必须大于下界 / Upper bound, must be greater than lower bound

**返回 / Returns:**
- `UniformDistribution` - 均匀分布对象 / Uniform distribution object

**异常 / Throws:**
- `IllegalArgumentException` - 如果上界小于等于下界 / If upper bound ≤ lower bound

### 指数分布 / Exponential Distribution

#### `Stat.exponential(float rate)`
创建指定速率参数的指数分布。

Creates an exponential distribution with specified rate parameter.

**参数 / Parameters:**
- `rate` - 速率参数（λ），必须大于0 / Rate parameter (λ), must be greater than 0

**返回 / Returns:**
- `ExponentialDistribution` - 指数分布对象 / Exponential distribution object

**异常 / Throws:**
- `IllegalArgumentException` - 如果速率参数小于等于0 / If rate parameter ≤ 0

### 卡方分布 / Chi-Squared Distribution

#### `Stat.chi2(float degreesOfFreedom)`
创建指定自由度的卡方分布。

Creates a chi-squared distribution with specified degrees of freedom.

**参数 / Parameters:**
- `degreesOfFreedom` - 自由度，必须大于0 / Degrees of freedom, must be greater than 0

**返回 / Returns:**
- `Chi2Distribution` - 卡方分布对象 / Chi-squared distribution object

**异常 / Throws:**
- `IllegalArgumentException` - 如果自由度小于等于0 / If degrees of freedom ≤ 0

### F分布 / F-Distribution

#### `Stat.f(float degreesOfFreedom1, float degreesOfFreedom2)`
创建指定自由度的F分布。

Creates an F-distribution with specified degrees of freedom.

**参数 / Parameters:**
- `degreesOfFreedom1` - 分子自由度，必须大于0 / Numerator degrees of freedom, must be greater than 0
- `degreesOfFreedom2` - 分母自由度，必须大于0 / Denominator degrees of freedom, must be greater than 0

**返回 / Returns:**
- `FDistribution` - F分布对象 / F-distribution object

**异常 / Throws:**
- `IllegalArgumentException` - 如果任一自由度小于等于0 / If any degrees of freedom ≤ 0

## 分布对象通用方法 / Common Distribution Object Methods

所有分布对象都实现了`IContinuousDistribution`接口，提供以下通用方法：

All distribution objects implement the `IContinuousDistribution` interface, providing the following common methods:

### 概率函数 / Probability Functions

| 方法 / Method | 描述 / Description | 参数 / Parameters | 返回 / Returns |
|-------------|------------------|-----------------|---------------|
| `pdf(float x)` | 概率密度函数 / Probability density function | `x` - 输入值 / Input value | `float` - PDF值 / PDF value |
| `cdf(float x)` | 累积分布函数 / Cumulative distribution function | `x` - 输入值 / Input value | `float` - CDF值 / CDF value |
| `ppf(float p)` | 百分点函数（分位数）/ Percent point function (quantile) | `p` - 概率值[0,1] / Probability value [0,1] | `float` - 分位数 / Quantile |
| `sf(float x)` | 生存函数 / Survival function | `x` - 输入值 / Input value | `float` - 生存概率 / Survival probability |
| `isf(float p)` | 逆生存函数 / Inverse survival function | `p` - 概率值[0,1] / Probability value [0,1] | `float` - 逆生存值 / Inverse survival value |

### 统计量 / Statistical Measures

| 方法 / Method | 描述 / Description | 返回 / Returns |
|-------------|------------------|---------------|
| `mean()` | 均值 / Mean | `float` |
| `var()` | 方差 / Variance | `float` |
| `std()` | 标准差 / Standard deviation | `float` |
| `median()` | 中位数 / Median | `float` |
| `mode()` | 众数 / Mode | `float` |
| `q1()` | 第一四分位数 / First quartile | `float` |
| `q3()` | 第三四分位数 / Third quartile | `float` |
| `skewness()` | 偏度 / Skewness | `float` |
| `kurtosis()` | 峰度 / Kurtosis | `float` |

### 随机采样 / Random Sampling

| 方法 / Method | 描述 / Description | 参数 / Parameters | 返回 / Returns |
|-------------|------------------|-----------------|---------------|
| `sample()` | 生成单个随机样本 / Generate single random sample | 无 / None | `float` - 随机样本 / Random sample |
| `sample(int n)` | 生成n个随机样本 / Generate n random samples | `n` - 样本数量 / Sample count | `float[]` - 随机样本数组 / Random sample array |

## 应用场景 / Application Scenarios

### 1. 假设检验 / Hypothesis Testing

```java
// t检验示例
// t-test example
StudentDistribution tDist = Stat.t(15.0f);
float tStatistic = 2.5f;
float pValue = 2.0f * (1.0f - tDist.cdf(Math.abs(tStatistic)));
```

### 2. 置信区间估计 / Confidence Interval Estimation

```java
// 正态分布置信区间
// Normal distribution confidence interval
NormalDistribution normal = Stat.norm(0.0f, 1.0f);
float zScore = normal.ppf(0.975f);  // 95%置信区间
```

### 3. 蒙特卡洛模拟 / Monte Carlo Simulation

```java
// 使用均匀分布估计π
// Estimate π using uniform distribution
UniformDistribution uniform = Stat.uniform(0.0f, 1.0f);
float[] samples = uniform.sample(1000000);
// 计算π估计值...
```

### 4. 可靠性分析 / Reliability Analysis

```java
// 指数分布可靠性分析
// Exponential distribution reliability analysis
ExponentialDistribution exp = Stat.exponential(0.1f);
float survivalProb = exp.sf(10.0f);  // 10时间单位的生存概率
```

## 性能特性 / Performance Characteristics

### 计算复杂度 / Computational Complexity

| 操作 / Operation | 时间复杂度 / Time Complexity | 空间复杂度 / Space Complexity |
|-----------------|---------------------------|----------------------------|
| 分布创建 / Distribution creation | O(1) | O(1) |
| PDF计算 / PDF calculation | O(1) | O(1) |
| CDF计算 / CDF calculation | O(1) | O(1) |
| 分位数计算 / Quantile calculation | O(1) | O(1) |
| 随机采样 / Random sampling | O(1) | O(1) |
| 批量采样 / Batch sampling | O(n) | O(n) |

### 数值精度 / Numerical Precision

- 使用`float`类型确保计算效率
- 采用经过验证的数值算法
- 对于极端值可能精度有限
- 建议在关键应用中验证结果

- Uses `float` type for computational efficiency
- Employs proven numerical algorithms
- Limited precision for extreme values
- Recommend validation in critical applications

## 最佳实践 / Best Practices

### 1. 参数验证 / Parameter Validation

```java
// 总是检查参数有效性
// Always validate parameter validity
try {
    NormalDistribution normal = Stat.norm(0.0f, -1.0f);
} catch (IllegalArgumentException e) {
    System.err.println("Invalid parameters: " + e.getMessage());
}
```

### 2. 批量采样 / Batch Sampling

```java
// 对于大量样本，使用批量方法
// For large samples, use batch methods
NormalDistribution normal = Stat.norm(0.0f, 1.0f);
float[] samples = normal.sample(1000000);  // 推荐 / Recommended
```

### 3. 内存管理 / Memory Management

```java
// 及时释放不需要的分布对象
// Release unused distribution objects promptly
NormalDistribution normal = Stat.norm(0.0f, 1.0f);
// 使用分布对象...
// Use distribution object...
normal = null;  // 帮助垃圾回收 / Help garbage collection
```

## 错误处理 / Error Handling

### 常见异常 / Common Exceptions

| 异常类型 / Exception Type | 原因 / Cause | 解决方案 / Solution |
|-------------------------|-------------|-------------------|
| `IllegalArgumentException` | 参数无效 / Invalid parameters | 检查参数范围 / Check parameter ranges |
| `ArithmeticException` | 数值溢出 / Numerical overflow | 使用更小的参数值 / Use smaller parameter values |
| `OutOfMemoryError` | 内存不足 / Insufficient memory | 减少批量采样大小 / Reduce batch sample size |

### 错误处理示例 / Error Handling Example

```java
public class SafeStatUsage {
    public static NormalDistribution createNormalDistribution(float mean, float stdDev) {
        try {
            return Stat.norm(mean, stdDev);
        } catch (IllegalArgumentException e) {
            System.err.println("Failed to create normal distribution: " + e.getMessage());
            return null;
        }
    }
    
    public static float[] safeSample(NormalDistribution dist, int n) {
        if (dist == null) {
            return new float[0];
        }
        
        try {
            return dist.sample(n);
        } catch (OutOfMemoryError e) {
            System.err.println("Insufficient memory for sampling: " + e.getMessage());
            return new float[0];
        }
    }
}
```

## 扩展性 / Extensibility

### 自定义分布 / Custom Distributions

虽然Stat类提供了常用的分布，但您也可以直接使用分布类：

While the Stat class provides common distributions, you can also use distribution classes directly:

```java
// 直接使用分布类
// Use distribution classes directly
NormalDistribution normal = new NormalDistribution(0.0f, 1.0f);
StudentDistribution tDist = new StudentDistribution(10.0f);
```

### 接口实现 / Interface Implementation

所有分布都实现了`IContinuousDistribution`接口，可以统一处理：

All distributions implement the `IContinuousDistribution` interface, allowing unified handling:

```java
public void processDistribution(IContinuousDistribution dist) {
    float pdf = dist.pdf(0.0f);
    float cdf = dist.cdf(1.0f);
    float sample = dist.sample();
    // 处理分布...
}
```

## 版本历史 / Version History

| 版本 / Version | 日期 / Date | 变更 / Changes |
|---------------|------------|---------------|
| 1.0 | 2024-01-01 | 初始版本，支持六种基本分布 / Initial version with six basic distributions |

## 许可证 / License

本项目采用MIT许可证。详情请参阅LICENSE文件。

This project is licensed under the MIT License. See the LICENSE file for details.

## 贡献 / Contributing

欢迎贡献代码、报告问题或提出改进建议。请遵循以下步骤：

Contributions are welcome! Please follow these steps:

1. Fork项目 / Fork the project
2. 创建特性分支 / Create a feature branch
3. 提交更改 / Commit your changes
4. 推送到分支 / Push to the branch
5. 创建Pull Request / Create a Pull Request

## 联系方式 / Contact

如有问题或建议，请通过以下方式联系：

For questions or suggestions, please contact us through:

- 项目Issues / Project Issues
- 邮箱 / Email: [your-email@example.com]
- 文档 / Documentation: [project-docs-url]

---

**注意 / Note:** 本文档基于YishapeMath库v1.0版本编写。API可能会在后续版本中发生变化。

**Note:** This documentation is based on YishapeMath library v1.0. The API may change in future versions.
