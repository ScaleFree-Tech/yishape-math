# YiShape-Math 易形数学

[![Java](https://img.shields.io/badge/Java-24+-blue.svg)](https://www.oracle.com/java/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-0.1-blue.svg)]()

## 项目简介 / Project Introduction

**易形数学（YiShape-Math）** 是一个基于Java开发的数学计算库，提供向量运算、矩阵运算、机器学习算法、优化算法、统计学方法和降维算法等核心功能，其API设计最大程度拟合了Python NumPy和SciPy的API。本库的初始目的是用于 电子科技大学《商务统计》、《商业大数据》、《数据分析与决策》、《工程项目管理》等课程的实验教学，通过亲自动手编程以学习线性代数、统计学、最优化、机器学习等领域算法的底层计算原理。本库当前也是 [易形空间 向量数据库管理系统（YiShape VecDB）](https://github.com/ScaleFree-Tech/YiShape-VecDB) 的底层数学基础设施。本库使用简单、性能优异，适用于科学计算、数据分析和机器学习等领域的各类应用。

**YiShape-Math** is a Java-based mathematical computing library that provides core functionalities including vector and matrix operations, machine learning algorithms, optimization algorithms, statistical methods, and dimensionality reduction techniques, and its API design closely mirrors that of the Python NumPy and SciPy API. The initial purpose of the library is to be used for the experimental teaching of courses such as "Business Statistics", "Big Data in Business", "Data Analysis and Decision Making", and "Project Management" at UESTC. Through hands-on programming, students can learn the underlying computational principles of algorithms in fields such as linear algebra, statistics, optimization, and machine learning. The library now also serves as the underlying mathematical infrastructure for the [YiShape Vector DataBase](https://github.com/ScaleFree-Tech/YiShape-VecDB). The library offers ease of use and excellent performance, making it suitable for scientific computing, data analysis, machine learning applications, etc.

## 主要特性 / Key Features

### 🚀 核心数学运算 / Core Mathematical Operations
- **向量运算** / **Vector Operations**: 完整的向量数学运算、统计运算、切片索引
  - *Complete vector mathematical operations, statistical operations, and slicing/indexing*
- **矩阵运算** / **Matrix Operations**: 矩阵变换、线性代数运算、特征分解
  - *Matrix transformations, linear algebra operations, and eigendecomposition*
- **数学工具** / **Math Utilities**: 类型转换、随机数生成、数学函数
  - *Type conversion, random number generation, and mathematical functions*

### 📊 统计学运算 / Statistical Methods
- **分布函数** / **Statistical Distributions**: 正态分布、t分布、卡方分布、F分布、均匀分布、指数分布等
  - *Normal, t, Chi-squared, F, Uniform, Exponential distributions and more*
- **概率密度函数** / **Probability Density Functions**: 完整的PDF和CDF计算
  - *Complete PDF and CDF calculations*
- **随机数生成** / **Random Number Generation**: 各种分布的随机数生成器
  - *Random number generators for various distributions*
- **统计描述** / **Statistical Descriptions**: 均值、方差、标准差、中位数、众数等
  - *Mean, variance, standard deviation, median, mode, and more*
- **假设检验** / **Hypothesis Testing**: 假设检验、参数估计 （待实现）
  - *Hypothesis testing and parameter estimation (to be implemented)*
- **方差分析** / **ANOVA**: Analysis of Variance （待实现）
  - *Analysis of Variance (to be implemented)*

### 🤖 机器学习算法 / Machine Learning Algorithms
- **线性回归** / **Linear Regression**: 支持L1、L2、ElasticNet正则化，LBFGS优化
  - *Support for L1, L2, ElasticNet regularization with LBFGS optimization*
- **逻辑回归** / **Logistic Regression**: 二分类和多分类逻辑回归
  - *Binary and multi-class logistic regression*
- **分类算法** / **Classification Algorithms**: 完整的分类器接口和实现
  - *Complete classifier interfaces and implementations*
- **模型评估** / **Model Evaluation**: 回归结果分析和分类结果分析
  - *Regression result analysis and classification result analysis*

### 🔧 优化算法 / Optimization Algorithms
- **L-BFGS优化器** / **L-BFGS Optimizer**: 拟牛顿法优化算法
  - *Quasi-Newton optimization algorithm*
- **线搜索** / **Line Search**: 一维搜索优化方法
  - *One-dimensional search optimization methods*
- **目标函数接口** / **Objective Function Interface**: 灵活的优化目标定义
  - *Flexible optimization objective definition*

### 📊 降维算法 / Dimensionality Reduction
- **PCA** / **Principal Component Analysis**: 主成分分析降维
  - *Principal Component Analysis for dimensionality reduction*
- **SVD** / **Singular Value Decomposition**: 奇异值分解
  - *Singular Value Decomposition*
- **t-SNE** / **t-Distributed Stochastic Neighbor Embedding**: 非线性降维
  - *Non-linear dimensionality reduction*
- **UMAP** / **Uniform Manifold Approximation and Projection**: 流形学习降维
  - *Manifold learning for dimensionality reduction*

## 快速开始 / Quick Start

### 环境要求 / Requirements
- Java 21 或更高版本 / Java 21 or higher
- Maven 3.6+ / Maven 3.6+

### 安装依赖 / Installation

**Jar:**

直接从右侧的Releases中下载最新的[Jar包](https://github.com/ScaleFree-Tech/yishape-math/releases)。Directly download the latest [Jar package](https://github.com/ScaleFree-Tech/yishape-math/releases) from the Releases on the right.


**Maven:**

```xml
<dependency>
    <groupId>com.reremouse.lab</groupId>
    <artifactId>yishape-math</artifactId>
    <version>0.1</version>
</dependency>
```



### 基本使用示例 / Basic Usage Examples

#### 向量运算 / Vector Operations
```java
// 创建向量 / Create vectors
IVector v1 = IVector.of(new float[]{1, 2, 3, 4});
IVector v2 = IVector.range(10);

// 基本运算 / Basic operations
IVector sum = v1.add(v2.slice(4));
float dotProduct = v1.innerProduct(v2.slice(4));

// 统计运算 / Statistical operations
float mean = v1.mean();
float std = v1.std();
```

#### 矩阵运算 / Matrix Operations
```java
// 创建矩阵 / Create matrices
IMatrix matrix1 = IMatrix.ones(3, 3);
IMatrix matrix2 = IMatrix.rand(3, 3);

// 矩阵运算 / Matrix operations
IMatrix result = matrix1.add(matrix2).mmul(2.0f);
IMatrix transposed = matrix2.transpose();
```

#### 线性回归 / Linear Regression
```java
// 创建线性回归模型 / Create linear regression model
RereLinearRegression lr = new RereLinearRegression();
lr.setRegularizationType(RegularizationType.L2);
lr.setLambda2(0.1f);

// 训练模型 / Train model
RegressionResult result = lr.fit(featureMatrix, labelVector);

// 预测 / Predict
float prediction = lr.predict(newFeatureVector);
```

#### 统计学分布 / Statistical Distributions
```java
// 创建正态分布 / Create normal distribution
NormalDistribution normal = Stat.norm(0, 1);  // 均值0，标准差1
NormalDistribution standardNormal = Stat.norm();  // 标准正态分布

// 计算概率密度和累积分布函数 / Calculate PDF and CDF
float pdf = normal.pdf(1.0f);  // 概率密度函数
float cdf = normal.cdf(1.0f);  // 累积分布函数

// 生成随机数 / Generate random numbers
float[] randomSamples = normal.sample(1000);  // 生成1000个随机样本

// 其他分布 / Other distributions
StudentDistribution tDist = Stat.t(10);  // t分布，自由度10
Chi2Distribution chi2Dist = Stat.chi2(5);  // 卡方分布，自由度5
FDistribution fDist = Stat.f(3, 7);  // F分布，自由度(3,7)
UniformDistribution uniform = Stat.uniform(0, 1);  // 均匀分布[0,1]
ExponentialDistribution exp = Stat.exponential(2.0f);  // 指数分布，参数2
```

#### PCA降维 / PCA Dimensionality Reduction
```java
// 创建PCA降维器 / Create PCA reducer
RerePCA pca = new RerePCA();

// 执行降维 / Perform dimensionality reduction
IMatrix reducedData = pca.dimensionReduction(originalData, 2);
```

#### 逻辑回归分类 / Logistic Regression Classification
```java
// 创建逻辑回归分类器 / Create logistic regression classifier
RereLogisticRegression lr = new RereLogisticRegression();

// 训练模型 / Train model
ClassificationResult result = lr.fit(featureMatrix, labelVector);

// 预测分类 / Predict classification
int prediction = lr.predict(newFeatureVector);
float[] probabilities = lr.predictProbabilities(newFeatureVector);
```

## 项目结构 / Project Structure

```
src/main/java/com/reremouse/lab/math/
├── IVector.java              # 向量操作接口 / Vector Operations Interface
├── RereVector.java           # 向量实现类 / Vector Implementation Class
├── IMatrix.java              # 矩阵操作接口 / Matrix Operations Interface
├── RereMatrix.java           # 矩阵实现类 / Matrix Implementation Class
├── RereMathUtil.java         # 数学工具类 / Math Utilities Class
├── YishapeMath.java          # 主入口类 / Main Entry Class
├── stat/                     # 统计学模块 / Statistics Module
│   ├── Stat.java             # 统计分布工厂类 / Statistical Distribution Factory Class
│   └── distribution/         # 概率分布实现 / Probability Distribution Implementations
│       ├── NormalDistribution.java      # 正态分布 / Normal Distribution
│       ├── StudentDistribution.java     # t分布 / Student's t-Distribution
│       ├── Chi2Distribution.java        # 卡方分布 / Chi-squared Distribution
│       ├── FDistribution.java           # F分布 / F-Distribution
│       ├── UniformDistribution.java     # 均匀分布 / Uniform Distribution
│       ├── ExponentialDistribution.java # 指数分布 / Exponential Distribution
│       ├── IContinuousDistribution.java # 连续分布接口 / Continuous Distribution Interface
│       └── IDiscreteDistribution.java   # 离散分布接口 / Discrete Distribution Interface
├── ml/                       # 机器学习算法 / Machine Learning Algorithms
│   ├── lr/                   # 线性回归 / Linear Regression
│   │   ├── IRegression.java             # 回归接口 / Regression Interface
│   │   ├── RereLinearRegression.java    # 线性回归实现 / Linear Regression Implementation
│   │   └── RegressionResult.java        # 回归结果 / Regression Result
│   └── cls/                  # 分类算法 / Classification Algorithms
│       ├── IClassification.java         # 分类接口 / Classification Interface
│       ├── RereLogisticRegression.java  # 逻辑回归实现 / Logistic Regression Implementation
│       ├── ClassificationResult.java    # 分类结果 / Classification Result
│       └── LogisticRegressionResult.java # 逻辑回归结果 / Logistic Regression Result
├── optimize/                 # 优化算法 / Optimization Algorithms
│   ├── IOptimizer.java       # 优化器接口 / Optimizer Interface
│   ├── IObjectiveFunction.java # 目标函数接口 / Objective Function Interface
│   ├── IGradientFunction.java  # 梯度函数接口 / Gradient Function Interface
│   ├── RereLBFGS.java        # L-BFGS优化器 / L-BFGS Optimizer
│   └── RereLineSearch.java   # 线搜索 / Line Search
└── dimreduce/                # 降维算法 / Dimensionality Reduction Algorithms
    ├── RerePCA.java          # PCA降维 / PCA Dimensionality Reduction
    ├── RereSVD.java          # SVD降维 / SVD Dimensionality Reduction
    ├── RereTSNE.java         # t-SNE降维 / t-SNE Dimensionality Reduction
    └── RereUMAP.java         # UMAP降维 / UMAP Dimensionality Reduction
```

## 核心类文档 / Core Classes Documentation

- [向量操作 (Vector Operations)](./docs/Vector-Operations.md) / [Vector Operations Documentation](./docs/Vector-Operations.md)
- [矩阵操作 (Matrix Operations)](./docs/Matrix-Operations.md) / [Matrix Operations Documentation](./docs/Matrix-Operations.md)
- [数学工具类 (Math Utilities)](./docs/Math-Utilities.md) / [Math Utilities Documentation](./docs/Math-Utilities.md)
- [线性回归 (Linear Regression)](./docs/Linear-Regression.md) / [Linear Regression Documentation](./docs/Linear-Regression.md)
- [优化算法 (Optimization Algorithms)](./docs/Optimization-Algorithms.md) / [Optimization Algorithms Documentation](./docs/Optimization-Algorithms.md)
- [降维算法 (Dimensionality Reduction)](./docs/Dimensionality-Reduction.md) / [Dimensionality Reduction Documentation](./docs/Dimensionality-Reduction.md)
- [API参考手册 (API Reference)](./docs/API-Reference.md) / [API Reference Manual](./docs/API-Reference.md)

### 文档说明 / Documentation Overview

这些文档提供了详细的API参考和使用指南：

These documents provide detailed API references and usage guides:

- **API参考** / **API Reference**: 完整的类和方法文档 / Complete class and method documentation
- **使用指南** / **Usage Guides**: 详细的代码示例和最佳实践 / Detailed code examples and best practices
- **算法说明** / **Algorithm Descriptions**: 数学原理和实现细节 / Mathematical principles and implementation details

## 统计学功能 / Statistical Functions

### 概率分布 / Probability Distributions

本库提供了完整的概率分布实现，支持以下分布：

This library provides comprehensive probability distribution implementations supporting the following distributions:

- **正态分布 (Normal Distribution)**: `Stat.norm(mean, std)` 或 `Stat.norm()` 标准正态分布
  - *Normal distribution with specified mean and standard deviation, or standard normal distribution*
- **t分布 (Student's t-Distribution)**: `Stat.t(degreesOfFreedom)`
  - *Student's t-distribution with specified degrees of freedom*
- **卡方分布 (Chi-squared Distribution)**: `Stat.chi2(degreesOfFreedom)`
  - *Chi-squared distribution with specified degrees of freedom*
- **F分布 (F-Distribution)**: `Stat.f(df1, df2)`
  - *F-distribution with specified degrees of freedom*
- **均匀分布 (Uniform Distribution)**: `Stat.uniform(lower, upper)`
  - *Uniform distribution over the interval [lower, upper]*
- **指数分布 (Exponential Distribution)**: `Stat.exponential(rate)`
  - *Exponential distribution with specified rate parameter*

### 分布功能 / Distribution Features

每个分布都提供以下功能：

Each distribution provides the following functionalities:

- `pdf(x)`: 概率密度函数 / Probability Density Function
- `cdf(x)`: 累积分布函数 / Cumulative Distribution Function
- `quantile(p)`: 分位数函数 / Quantile Function
- `sample(n)`: 生成n个随机样本 / Generate n random samples
- `mean()`, `variance()`, `stdDev()`: 统计量计算 / Statistical measures calculation

## 使用示例 / Usage Examples

- [向量运算示例](./docs/examples/Vector-Examples.md) / [Vector Operations Examples](./docs/examples/Vector-Examples.md)
- [矩阵运算示例](./docs/examples/Matrix-Examples.md) / [Matrix Operations Examples](./docs/examples/Matrix-Examples.md)
- [数学工具类示例](./docs/examples/Math-Utilities-Examples.md) / [Math Utilities Examples](./docs/examples/Math-Utilities-Examples.md)
- [机器学习示例](./docs/examples/Machine-Learning-Examples.md) / [Machine Learning Examples](./docs/examples/Machine-Learning-Examples.md)
- [优化算法示例](./docs/examples/Optimization-Examples.md) / [Optimization Algorithms Examples](./docs/examples/Optimization-Examples.md)
- [降维算法示例](./docs/examples/Dimensionality-Reduction-Examples.md) / [Dimensionality Reduction Examples](./docs/examples/Dimensionality-Reduction-Examples.md)

### 快速入门指南 / Quick Start Guide

这些示例文档提供了详细的使用指南和代码示例，帮助您快速上手：

These example documents provide detailed usage guides and code examples to help you get started quickly:

- **基础数学运算** / **Basic Mathematical Operations**: 向量和矩阵的基本操作 / Basic vector and matrix operations
- **统计学应用** / **Statistical Applications**: 概率分布和统计分析的实践 / Practical probability distributions and statistical analysis
- **机器学习实践** / **Machine Learning Practice**: 回归和分类算法的实际应用 / Real-world applications of regression and classification algorithms
- **高级功能** / **Advanced Features**: 优化算法和降维技术的使用 / Usage of optimization algorithms and dimensionality reduction techniques

## 性能特性 / Performance Features

- **内存优化** / **Memory Optimization**: 高效的数组操作和内存管理，支持大矩阵运算
- **算法优化** / **Algorithm Optimization**: 优化的数学算法实现，包括LBFGS优化器
- **类型安全** / **Type Safety**: 强类型系统，避免运行时错误，提供完整的类型检查
- **接口设计** / **Interface Design**: 清晰的接口设计，易于扩展和自定义实现
- **数值稳定性** / **Numerical Stability**: 采用数值稳定的算法实现，确保计算精度
- **并行计算支持** / **Parallel Computing Support**: 支持多线程并行计算，提高大规模数据处理效率

## 贡献指南 / Contributing

我们欢迎社区贡献！请遵循以下步骤：

We welcome community contributions! Please follow these steps:

1. Fork 本项目 / Fork this project
2. 创建特性分支 / Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 提交更改 / Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 / Push to the branch (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request / Open a Pull Request

### 贡献指南 / Contribution Guidelines

- 请确保代码符合项目的编码规范 / Please ensure your code follows the project's coding standards
- 添加适当的测试用例 / Add appropriate test cases
- 更新相关文档 / Update relevant documentation
- 确保所有测试通过 / Ensure all tests pass

## 许可证 / License

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE.txt) 文件了解详情。

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

### MIT 许可证条款 / MIT License Terms

MIT 许可证是一个宽松的开源许可证，允许您自由使用、修改、分发和销售软件，只要您保留版权声明和许可证文本。

The MIT License is a permissive open-source license that allows you to freely use, modify, distribute, and sell the software, as long as you retain the copyright notice and license text.

## 联系方式 / Contact

- 项目维护者 / Project Maintainer: Big Data and Decision Analytics Research Center of UESTC, Scale-Free Tech.
- 项目地址 / Project URL: [https://github.com/ScaleFree-Tech/yishape-math](https://github.com/ScaleFree-Tech/yishape-math)
- 问题反馈 / Issues: [https://github.com/ScaleFree-Tech/yishape-math/issues](https://github.com/ScaleFree-Tech/yishape-math/issues)

### 获取帮助 / Getting Help

如果您在使用过程中遇到问题，可以通过以下方式获取帮助：

If you encounter any issues while using the library, you can get help through the following channels:

- **GitHub Issues**: 报告bug或提出功能请求 / Report bugs or request features
- **文档**: 查看详细的API文档和示例 / Check detailed API documentation and examples
- **社区**: 参与讨论和分享经验 / Participate in discussions and share experiences

## 更新日志 / Changelog

### v0.1 (2025-08)
- ✨ 初始版本发布 / Initial release
- 🚀 核心向量和矩阵运算功能 / Core vector and matrix operations
- 📊 完整的统计学分布函数库（正态、t、卡方、F、均匀、指数分布） / Complete statistical distribution library (Normal, t, Chi-squared, F, Uniform, Exponential distributions)
- 🤖 线性回归和逻辑回归机器学习算法 / Linear regression and logistic regression machine learning algorithms
- 🔧 L-BFGS优化器和线搜索算法 / L-BFGS optimizer and line search algorithms
- 📈 PCA、SVD、t-SNE、UMAP降维算法 / PCA, SVD, t-SNE, UMAP dimensionality reduction algorithms
- 🎯 支持L1、L2、ElasticNet正则化 / Support for L1, L2, ElasticNet regularization
- 🔢 完整的概率密度函数和累积分布函数 / Complete probability density and cumulative distribution functions
- 📋 丰富的统计描述功能（均值、方差、中位数、众数等） / Rich statistical description functions (mean, variance, median, mode, etc.)

---

**YiShape-Math** - 让Java应用中的数学计算更简单、更高效！

**YiShape-Math** - Making mathematical computing simpler and more efficient for Java applications!
