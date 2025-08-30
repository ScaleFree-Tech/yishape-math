# 线性回归 (Linear Regression)

## 概述 / Overview

`RereLinearRegression` 类实现了标准的线性回归算法，使用最小二乘法优化目标函数。该实现支持多种正则化选项，包括L1（Lasso）、L2（Ridge）和ElasticNet正则化，并提供了灵活的模型配置选项。

The `RereLinearRegression` class implements the standard linear regression algorithm using least squares optimization. This implementation supports multiple regularization options including L1 (Lasso), L2 (Ridge), and ElasticNet regularization, and provides flexible model configuration options.

## 核心类 / Core Classes

### RereLinearRegression 类 / RereLinearRegression Class

主要的线性回归实现类，实现了以下接口：
- `IRegression`: 回归模型接口
- `IGradientFunction`: 梯度计算接口
- `IObjectiveFunction`: 目标函数接口

### IRegression 接口 / IRegression Interface

```java
public interface IRegression {
    /**
     * 训练回归模型 / Train regression model
     * @param features 特征矩阵 / Feature matrix
     * @param labels 标签向量 / Label vector
     * @return 回归结果 / Regression result
     */
    RegressionResult fit(IMatrix features, IVector labels);
    
    /**
     * 预测新样本 / Predict new samples
     * @param features 特征向量 / Feature vector
     * @return 预测值 / Prediction value
     */
    float predict(IVector features);
}
```

### RegressionResult 类 / RegressionResult Class

```java
public class RegressionResult {
    private IVector weights;      // 权重向量 / Weight vector
    private float loss;           // 损失值 / Loss value
    private float r2Score;        // R²分数 / R² score
    private float mse;            // 均方误差 / Mean squared error
    private float mae;            // 平均绝对误差 / Mean absolute error
    
    // getters and setters
}
```

## 算法原理 / Algorithm Principles

### 数学模型 / Mathematical Model

线性回归模型的形式为：
```
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

其中：
- `wᵢ` 是第i个特征的权重系数
- `xᵢ` 是第i个特征值
- `b` 是偏置项（截距）
- `y` 是预测值

### 目标函数 / Objective Function

使用均方误差损失函数加正则化项：
```
J(w) = (1/2n) * ||Xw - y||² + R(w)
```

其中R(w)是正则化项：

#### L1正则化（Lasso）
```
R(w) = λ₁ * ||w||₁ = λ₁ * Σ|wᵢ|
```
- 特点：产生稀疏解，有助于特征选择
- 适用场景：特征数量多，需要特征选择
- 参数：λ₁ > 0

#### L2正则化（Ridge）
```
R(w) = (λ₂/2) * ||w||² = (λ₂/2) * Σwᵢ²
```
- 特点：防止过拟合，权重衰减
- 适用场景：防止过拟合，提高泛化能力
- 参数：λ₂ > 0

#### ElasticNet正则化
```
R(w) = λ₁ * ||w||₁ + (λ₂/2) * ||w||²
```
- 特点：结合L1和L2的优点
- 适用场景：需要特征选择的同时防止过拟合
- 参数：λ₁ > 0, λ₂ > 0

### 梯度计算 / Gradient Calculation

目标函数的梯度为：
```
∇J(w) = (1/n) * X^T * (Xw - y) + ∇R(w)
```

其中∇R(w)是正则化项的梯度：

#### L1正则化梯度
```
∇||w||₁ = sign(w)
```
- sign(wᵢ) = 1 if wᵢ > 0
- sign(wᵢ) = -1 if wᵢ < 0  
- sign(wᵢ) = 0 if wᵢ = 0

#### L2正则化梯度
```
∇||w||² = 2w
```

#### ElasticNet梯度
```
∇R(w) = λ₁ * sign(w) + λ₂ * w
```

## 主要特性 / Main Features

### 1. 灵活的模型配置 / Flexible Model Configuration

```java
// 创建线性回归模型 / Create linear regression model
RereLinearRegression lr = new RereLinearRegression();

// 配置正则化 / Configure regularization
lr.setRegularizationType(RegularizationType.L2);
lr.setLambda2(0.1f);

// 配置偏置项 / Configure bias term
lr.setIncludeBias(true);

// 配置优化器 / Configure optimizer
lr.setOptimizer(new RereLBFGS());
```

### 2. 自动特征增广 / Automatic Feature Augmentation

```java
// 自动在特征矩阵中添加偏置列 / Automatically add bias column to feature matrix
// 如果 includeBias = true，特征矩阵会从 [n_samples, n_features] 变为 [n_samples, n_features+1]
// If includeBias = true, feature matrix changes from [n_samples, n_features] to [n_samples, n_features+1]
```

### 3. 多种正则化选项 / Multiple Regularization Options

```java
// 无正则化 / No regularization
lr.setRegularizationType(RegularizationType.NONE);

// L1正则化（Lasso）/ L1 regularization (Lasso)
lr.setRegularizationType(RegularizationType.L1);
lr.setLambda1(0.01f);

// L2正则化（Ridge）/ L2 regularization (Ridge)
lr.setRegularizationType(RegularizationType.L2);
lr.setLambda2(0.1f);

// ElasticNet正则化 / ElasticNet regularization
lr.setRegularizationType(RegularizationType.ELASTIC_NET);
lr.setLambda1(0.01f);
lr.setLambda2(0.1f);
```

## 使用示例 / Usage Examples

### 示例1：基本线性回归 / Example 1: Basic Linear Regression

```java
// 准备数据 / Prepare data
float[][] featureData = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9},
    {10, 11, 12}
};
float[] labelData = {14, 32, 50, 68};

IMatrix features = IMatrix.of(featureData);
IVector labels = IVector.of(labelData);

// 创建和训练模型 / Create and train model
RereLinearRegression lr = new RereLinearRegression();
RegressionResult result = lr.fit(features, labels);

// 获取结果 / Get results
IVector weights = result.getWeights();
float loss = result.getLoss();
float r2Score = result.getR2Score();

System.out.println("权重: " + weights);
System.out.println("损失: " + loss);
System.out.println("R²分数: " + r2Score);

// 预测新样本 / Predict new sample
IVector newFeatures = IVector.of(new float[]{2, 3, 4});
float prediction = lr.predict(newFeatures);
System.out.println("预测值: " + prediction);
```

### 示例2：带正则化的线性回归 / Example 2: Linear Regression with Regularization

```java
// 创建带L2正则化的模型 / Create model with L2 regularization
RereLinearRegression lr = new RereLinearRegression();
lr.setRegularizationType(RegularizationType.L2);
lr.setLambda2(0.1f);

// 训练模型 / Train model
RegressionResult result = lr.fit(features, labels);

// 查看正则化效果 / View regularization effects
System.out.println("L2正则化系数: " + lr.getLambda2());
System.out.println("最终损失: " + result.getLoss());
```

### 示例3：ElasticNet正则化 / Example 3: ElasticNet Regularization

```java
// 创建ElasticNet正则化模型 / Create ElasticNet regularization model
RereLinearRegression lr = new RereLinearRegression();
lr.setRegularizationType(RegularizationType.ELASTIC_NET);
lr.setLambda1(0.01f);  // L1正则化系数 / L1 regularization coefficient
lr.setLambda2(0.1f);   // L2正则化系数 / L2 regularization coefficient

// 训练模型 / Train model
RegressionResult result = lr.fit(features, labels);

// 查看结果 / View results
System.out.println("L1正则化系数: " + lr.getLambda1());
System.out.println("L2正则化系数: " + lr.getLambda2());
System.out.println("最终损失: " + result.getLoss());
```

### 示例4：模型评估 / Example 4: Model Evaluation

```java
// 训练模型 / Train model
RereLinearRegression lr = new RereLinearRegression();
RegressionResult result = lr.fit(features, labels);

// 评估指标 / Evaluation metrics
float mse = result.getMse();           // 均方误差 / Mean squared error
float mae = result.getMae();           // 平均绝对误差 / Mean absolute error
float r2Score = result.getR2Score();   // R²分数 / R² score

System.out.println("均方误差 (MSE): " + mse);
System.out.println("平均绝对误差 (MAE): " + mae);
System.out.println("R²分数: " + r2Score);

// 解释R²分数 / Interpret R² score
if (r2Score > 0.8) {
    System.out.println("模型拟合很好 / Model fits well");
} else if (r2Score > 0.6) {
    System.out.println("模型拟合一般 / Model fits moderately");
} else {
    System.out.println("模型拟合较差 / Model fits poorly");
}
```

### 示例5：特征重要性分析 / Example 5: Feature Importance Analysis

```java
// 训练模型 / Train model
RereLinearRegression lr = new RereLinearRegression();
RegressionResult result = lr.fit(features, labels);

// 获取权重 / Get weights
IVector weights = result.getWeights();

// 分析特征重要性 / Analyze feature importance
System.out.println("特征重要性分析 / Feature Importance Analysis:");
for (int i = 0; i < weights.length(); i++) {
    if (i == weights.length() - 1 && lr.isIncludeBias()) {
        System.out.println("偏置项 (Bias): " + weights.get(i));
    } else {
        System.out.println("特征 " + i + ": " + weights.get(i));
    }
}

// 找出最重要的特征 / Find most important features
float maxWeight = weights.max();
int maxIndex = weights.argmax();
System.out.println("最重要特征索引: " + maxIndex + ", 权重: " + maxWeight);
```

### 示例6：交叉验证 / Example 6: Cross Validation

```java
// 简单的交叉验证 / Simple cross validation
int foldCount = 5;
int sampleCount = features.getRowNum();
int foldSize = sampleCount / foldCount;

float totalMSE = 0.0f;
float totalR2 = 0.0f;

for (int fold = 0; fold < foldCount; fold++) {
    // 划分训练集和验证集 / Split training and validation sets
    int startIdx = fold * foldSize;
    int endIdx = (fold == foldCount - 1) ? sampleCount : (fold + 1) * foldSize;
    
    // 创建训练集 / Create training set
    List<float[]> trainFeatures = new ArrayList<>();
    List<Float> trainLabels = new ArrayList<>();
    
    for (int i = 0; i < sampleCount; i++) {
        if (i < startIdx || i >= endIdx) {
            trainFeatures.add(features.getRow(i).getData());
            trainLabels.add(labels.get(i));
        }
    }
    
    // 创建验证集 / Create validation set
    List<float[]> valFeatures = new ArrayList<>();
    List<Float> valLabels = new ArrayList<>();
    
    for (int i = startIdx; i < endIdx; i++) {
        valFeatures.add(features.getRow(i).getData());
        valLabels.add(labels.get(i));
    }
    
    // 训练模型 / Train model
    IMatrix trainFeatureMatrix = IMatrix.of(trainFeatures);
    IVector trainLabelVector = IVector.of(RereMathUtil.toPrimitive(trainLabels.toArray(new Float[0])));
    
    RereLinearRegression lr = new RereLinearRegression();
    RegressionResult result = lr.fit(trainFeatureMatrix, trainLabelVector);
    
    // 验证模型 / Validate model
    IMatrix valFeatureMatrix = IMatrix.of(valFeatures);
    IVector valLabelVector = IVector.of(RereMathUtil.toPrimitive(valLabels.toArray(new Float[0])));
    
    float foldMSE = 0.0f;
    for (int i = 0; i < valFeatureMatrix.getRowNum(); i++) {
        float prediction = lr.predict(valFeatureMatrix.getRow(i));
        float actual = valLabelVector.get(i);
        foldMSE += (prediction - actual) * (prediction - actual);
    }
    foldMSE /= valFeatureMatrix.getRowNum();
    
    totalMSE += foldMSE;
    totalR2 += result.getR2Score();
    
    System.out.println("Fold " + (fold + 1) + " - MSE: " + foldMSE + ", R²: " + result.getR2Score());
}

System.out.println("平均MSE: " + (totalMSE / foldCount));
System.out.println("平均R²: " + (totalR2 / foldCount));
```

## 性能特性 / Performance Features

### 优化算法 / Optimization Algorithm
- 使用L-BFGS优化器，收敛速度快
- 支持线搜索，提高优化稳定性
- 自动梯度计算，无需手动实现

### 内存优化 / Memory Optimization
- 高效的矩阵运算
- 智能的内存管理
- 支持大规模数据集

### 数值稳定性 / Numerical Stability
- 正则化防止过拟合
- 梯度裁剪避免梯度爆炸
- 条件数检查提高稳定性

## 注意事项 / Notes

1. **数据预处理** / **Data Preprocessing**: 建议对特征进行标准化处理
2. **正则化参数** / **Regularization Parameters**: 根据数据特点选择合适的正则化参数
3. **特征选择** / **Feature Selection**: L1正则化有助于特征选择
4. **过拟合** / **Overfitting**: 使用正则化和交叉验证防止过拟合

## 扩展性 / Extensibility

`RereLinearRegression` 类设计支持扩展：
- 自定义损失函数
- 新的正则化方法
- 不同的优化器
- 在线学习支持

## 应用场景 / Application Scenarios

### 预测分析 / Predictive Analytics
- 房价预测
- 销售预测
- 风险评估

### 科学研究 / Scientific Research
- 实验数据分析
- 统计建模
- 相关性研究

### 机器学习 / Machine Learning
- 基线模型
- 特征工程
- 模型集成

---

**线性回归** - 机器学习的基础，让预测更准确！

**Linear Regression** - The foundation of machine learning, making predictions more accurate!
