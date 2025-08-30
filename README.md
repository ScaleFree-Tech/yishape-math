# YiShape-Math 易形数学

[![Java](https://img.shields.io/badge/Java-24+-blue.svg)](https://www.oracle.com/java/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-0.1-blue.svg)]()

## 项目简介 / Project Introduction

**易形数学（YiShape-Math）** 是一个基于Java开发的数学计算库，提供向量运算、矩阵运算、机器学习算法、优化算法、统计学方法和降维算法等核心功能。本库的API设计最大程度拟合了Python NumPy API，使用简单、性能优异，适用于科学计算、数据分析和机器学习应用。本库也是 [易形空间 向量数据库管理系统（YiShape VecDB）](https://github.com/ScaleFree-Tech/yishape) 的底层数学基础设施。

**YiShape-Math** is a Java-based mathematical computing library that provides core functionalities including vector and matrix operations, machine learning algorithms, optimization algorithms, statistical methods, and dimensionality reduction techniques. The API design of this library closely mirrors that of the Python NumPy API, offering ease of use and excellent performance, making it suitable for scientific computing, data analysis, and machine learning applications. This library also serves as the underlying mathematical infrastructure for the [YiShape Vector DataBase](https://github.com/ScaleFree-Tech/yishape).

## 主要特性 / Key Features

### 🚀 核心数学运算 / Core Mathematical Operations
- **向量运算** / **Vector Operations**: 完整的向量数学运算、统计运算、切片索引
- **矩阵运算** / **Matrix Operations**: 矩阵变换、线性代数运算、特征分解
- **数学工具** / **Math Utilities**: 类型转换、随机数生成、数学函数

### 🤖 机器学习算法 / Machine Learning Algorithms
- **线性回归** / **Linear Regression**: 支持L1、L2、ElasticNet正则化
- **分类算法** / **Classification Algorithms**: 逻辑回归等分类器
- **模型评估** / **Model Evaluation**: 回归结果分析和模型性能评估

### 🔧 优化算法 / Optimization Algorithms
- **L-BFGS优化器** / **L-BFGS Optimizer**: 拟牛顿法优化算法
- **线搜索** / **Line Search**: 一维搜索优化方法
- **目标函数接口** / **Objective Function Interface**: 灵活的优化目标定义

### 📊 降维算法 / Dimensionality Reduction
- **PCA** / **Principal Component Analysis**: 主成分分析降维
- **SVD** / **Singular Value Decomposition**: 奇异值分解
- **t-SNE** / **t-Distributed Stochastic Neighbor Embedding**: 非线性降维
- **UMAP** / **Uniform Manifold Approximation and Projection**: 流形学习降维

## 快速开始 / Quick Start

### 环境要求 / Requirements
- Java 24 或更高版本 / Java 24 or higher
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

#### PCA降维 / PCA Dimensionality Reduction
```java
// 创建PCA降维器 / Create PCA reducer
RerePCA pca = new RerePCA();

// 执行降维 / Perform dimensionality reduction
IMatrix reducedData = pca.dimensionReduction(originalData, 2);
```

## 项目结构 / Project Structure

```
src/main/java/com/reremouse/lab/math/
├── IVector.java              # 向量操作接口
├── RereVector.java           # 向量实现类
├── IMatrix.java              # 矩阵操作接口
├── RereMatrix.java           # 矩阵实现类
├── RereMathUtil.java         # 数学工具类
├── ml/                       # 机器学习算法
│   ├── lr/                   # 线性回归
│   └── cls/                  # 分类算法
├── optimize/                 # 优化算法
│   ├── IOptimizer.java       # 优化器接口
│   ├── RereLBFGS.java        # L-BFGS优化器
│   └── RereLineSearch.java   # 线搜索
└── dimreduce/                # 降维算法
    ├── RerePCA.java          # PCA降维
    ├── RereSVD.java          # SVD降维
    ├── RereTSNE.java         # t-SNE降维
    └── RereUMAP.java         # UMAP降维
```

## 核心类文档 / Core Classes Documentation

- [向量操作 (Vector Operations)](./docs/Vector-Operations.md)
- [矩阵操作 (Matrix Operations)](./docs/Matrix-Operations.md)
- [数学工具类 (Math Utilities)](./docs/Math-Utilities.md)
- [线性回归 (Linear Regression)](./docs/Linear-Regression.md)
- [优化算法 (Optimization Algorithms)](./docs/Optimization-Algorithms.md)
- [降维算法 (Dimensionality Reduction)](./docs/Dimensionality-Reduction.md)
- [API参考手册 (API Reference)](./docs/API-Reference.md)

## 使用示例 / Usage Examples

- [向量运算示例](./docs/examples/Vector-Examples.md)
- [矩阵运算示例](./docs/examples/Matrix-Examples.md)
- [数学工具类示例](./docs/examples/Math-Utilities-Examples.md)
- [机器学习示例](./docs/examples/Machine-Learning-Examples.md)
- [优化算法示例](./docs/examples/Optimization-Examples.md)
- [降维算法示例](./docs/examples/Dimensionality-Reduction-Examples.md)

## 性能特性 / Performance Features

- **内存优化** / **Memory Optimization**: 高效的数组操作和内存管理
- **算法优化** / **Algorithm Optimization**: 优化的数学算法实现
- **类型安全** / **Type Safety**: 强类型系统，避免运行时错误
- **接口设计** / **Interface Design**: 清晰的接口设计，易于扩展

## 贡献指南 / Contributing

我们欢迎社区贡献！请遵循以下步骤：

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证 / License

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE.txt) 文件了解详情。

## 联系方式 / Contact

- 项目维护者 / Project Maintainer: Big Data and Decision Analytics Research Center of UESTC, Scale-Free Tech.
- 项目地址 / Project URL: [https://github.com/ScaleFree-Tech/yishape-math](https://github.com/ScaleFree-Tech/yishape-math)
- 问题反馈 / Issues: [https://github.com/ScaleFree-Tech/yishape-math/issues](https://github.com/ScaleFree-Tech/yishape-math/issues)

## 更新日志 / Changelog

### v0.1 (2025-08-30)
- ✨ 初始版本发布
- 🚀 核心向量和矩阵运算功能
- 🤖 线性回归机器学习算法
- 🔧 L-BFGS优化器
- 📊 PCA、SVD、t-SNE、UMAP降维算法

---

**YiShape-Math** - 让Java应用中的数学计算更简单、更高效！

**YiShape-Math** - Making mathematical computing simpler and more efficient for Java applications!
