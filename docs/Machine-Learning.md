# 机器学习算法 (Machine Learning Algorithms)

## 概述 / Overview

本文档介绍了 `com.reremouse.lab.math.ml` 包中实现的机器学习算法。该包提供了完整的机器学习解决方案，包括线性回归和逻辑回归算法，支持多种正则化选项和灵活的模型配置。

This document introduces the machine learning algorithms implemented in the `com.reremouse.lab.math.ml` package. The package provides a complete machine learning solution, including linear regression and logistic regression algorithms, with support for multiple regularization options and flexible model configuration.

## 算法列表 / Algorithm List

### 1. 线性回归 (Linear Regression)
- **类名 / Class**: `RereLinearRegression`
- **功能 / Function**: 回归预测，支持多种正则化
- **应用 / Application**: 连续值预测，特征重要性分析

### 2. 逻辑回归 (Logistic Regression)  
- **类名 / Class**: `RereLogisticRegression`
- **功能 / Function**: 分类预测，支持二分类和多分类
- **应用 / Application**: 分类问题，概率预测

---

# 线性回归 (Linear Regression)

## 概述 / Overview

`RereLinearRegression` 类实现了标准的线性回归算法，使用最小二乘法优化目标函数。该实现支持多种正则化选项，包括L1（Lasso）、L2（Ridge）和ElasticNet正则化，并提供了灵活的模型配置选项。

The `RereLinearRegression` class implements the standard linear regression algorithm using least squares optimization. This implementation supports multiple regularization options including L1 (Lasso), L2 (Ridge), and ElasticNet regularization, and provides flexible model configuration options.

## 核心类 / Core Classes

### RereLinearRegression 类 / RereLinearRegression Class

主要的线性回归实现类，实现了以下接口：
The main linear regression implementation class that implements the following interfaces:
- `IRegression`: 回归模型接口 / Regression model interface
- `IGradientFunction`: 梯度计算接口 / Gradient calculation interface
- `IObjectiveFunction`: 目标函数接口 / Objective function interface

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
The linear regression model has the form:
```
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

其中：
Where:
- `wᵢ` 是第i个特征的权重系数 / is the weight coefficient for the i-th feature
- `xᵢ` 是第i个特征值 / is the i-th feature value
- `b` 是偏置项（截距）/ is the bias term (intercept)
- `y` 是预测值 / is the predicted value

### 目标函数 / Objective Function

使用均方误差损失函数加正则化项：
Using mean squared error loss function with regularization term:
```
J(w) = (1/2n) * ||Xw - y||² + R(w)
```

其中R(w)是正则化项：
Where R(w) is the regularization term:

#### L1正则化（Lasso）/ L1 Regularization (Lasso)
```
R(w) = λ₁ * ||w||₁ = λ₁ * Σ|wᵢ|
```
- 特点：产生稀疏解，有助于特征选择 / Characteristics: produces sparse solutions, helps with feature selection
- 适用场景：特征数量多，需要特征选择 / Use cases: many features, need feature selection
- 参数：λ₁ > 0 / Parameters: λ₁ > 0

#### L2正则化（Ridge）/ L2 Regularization (Ridge)
```
R(w) = (λ₂/2) * ||w||² = (λ₂/2) * Σwᵢ²
```
- 特点：防止过拟合，权重衰减 / Characteristics: prevents overfitting, weight decay
- 适用场景：防止过拟合，提高泛化能力 / Use cases: prevent overfitting, improve generalization
- 参数：λ₂ > 0 / Parameters: λ₂ > 0

#### ElasticNet正则化 / ElasticNet Regularization
```
R(w) = λ₁ * ||w||₁ + (λ₂/2) * ||w||²
```
- 特点：结合L1和L2的优点 / Characteristics: combines advantages of L1 and L2
- 适用场景：需要特征选择的同时防止过拟合 / Use cases: need feature selection while preventing overfitting
- 参数：λ₁ > 0, λ₂ > 0 / Parameters: λ₁ > 0, λ₂ > 0

### 梯度计算 / Gradient Calculation

目标函数的梯度为：
The gradient of the objective function is:
```
∇J(w) = (1/n) * X^T * (Xw - y) + ∇R(w)
```

其中∇R(w)是正则化项的梯度：
Where ∇R(w) is the gradient of the regularization term:

#### L1正则化梯度 / L1 Regularization Gradient
```
∇||w||₁ = sign(w)
```
- sign(wᵢ) = 1 if wᵢ > 0
- sign(wᵢ) = -1 if wᵢ < 0  
- sign(wᵢ) = 0 if wᵢ = 0

#### L2正则化梯度 / L2 Regularization Gradient
```
∇||w||² = 2w
```

#### ElasticNet梯度 / ElasticNet Gradient
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

        System.out.println("权重: " + weights); // Weights
        System.out.println("损失: " + loss); // Loss
        System.out.println("R²分数: " + r2Score); // R² score

// 预测新样本 / Predict new sample
IVector newFeatures = IVector.of(new float[]{2, 3, 4});
float prediction = lr.predict(newFeatures);
System.out.println("预测值: " + prediction); // Prediction
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
        System.out.println("L2正则化系数: " + lr.getLambda2()); // L2 regularization coefficient
        System.out.println("最终损失: " + result.getLoss()); // Final loss
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
        System.out.println("L1正则化系数: " + lr.getLambda1()); // L1 regularization coefficient
        System.out.println("L2正则化系数: " + lr.getLambda2()); // L2 regularization coefficient
        System.out.println("最终损失: " + result.getLoss()); // Final loss
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
- 使用L-BFGS优化器，收敛速度快 / Uses L-BFGS optimizer with fast convergence
- 支持线搜索，提高优化稳定性 / Supports line search to improve optimization stability
- 自动梯度计算，无需手动实现 / Automatic gradient calculation, no manual implementation needed

### 内存优化 / Memory Optimization
- 高效的矩阵运算 / Efficient matrix operations
- 智能的内存管理 / Smart memory management
- 支持大规模数据集 / Supports large-scale datasets

### 数值稳定性 / Numerical Stability
- 正则化防止过拟合 / Regularization prevents overfitting
- 梯度裁剪避免梯度爆炸 / Gradient clipping prevents gradient explosion
- 条件数检查提高稳定性 / Condition number checking improves stability

## 注意事项 / Notes

1. **数据预处理** / **Data Preprocessing**: 建议对特征进行标准化处理
2. **正则化参数** / **Regularization Parameters**: 根据数据特点选择合适的正则化参数
3. **特征选择** / **Feature Selection**: L1正则化有助于特征选择
4. **过拟合** / **Overfitting**: 使用正则化和交叉验证防止过拟合

## 扩展性 / Extensibility

`RereLinearRegression` 类设计支持扩展：
The `RereLinearRegression` class is designed to support extensions:
- 自定义损失函数 / Custom loss functions
- 新的正则化方法 / New regularization methods
- 不同的优化器 / Different optimizers
- 在线学习支持 / Online learning support

## 应用场景 / Application Scenarios

### 预测分析 / Predictive Analytics
- 房价预测 / House price prediction
- 销售预测 / Sales forecasting
- 风险评估 / Risk assessment

### 科学研究 / Scientific Research
- 实验数据分析 / Experimental data analysis
- 统计建模 / Statistical modeling
- 相关性研究 / Correlation studies

### 机器学习 / Machine Learning
- 基线模型 / Baseline models
- 特征工程 / Feature engineering
- 模型集成 / Model ensemble

---

# 逻辑回归 (Logistic Regression)

## 概述 / Overview

`RereLogisticRegression` 类实现了统一的逻辑回归算法，自动检测并支持二分类和多分类问题。该实现使用sigmoid函数进行二分类，使用softmax函数进行多分类，支持多种正则化选项，并提供了灵活的模型配置。

The `RereLogisticRegression` class implements a unified logistic regression algorithm that automatically detects and supports both binary and multiclass classification problems. This implementation uses sigmoid function for binary classification and softmax function for multiclass classification, supports multiple regularization options, and provides flexible model configuration.

## 核心类 / Core Classes

### RereLogisticRegression 类 / RereLogisticRegression Class

主要的逻辑回归实现类，实现了以下接口：
The main logistic regression implementation class that implements the following interfaces:
- `IClassification`: 分类模型接口 / Classification model interface
- `IGradientFunction`: 梯度计算接口 / Gradient calculation interface
- `IObjectiveFunction`: 目标函数接口 / Objective function interface

### IClassification 接口 / IClassification Interface

```java
public interface IClassification {
    /**
     * 训练分类模型 / Train classification model
     * @param features 特征矩阵 / Feature matrix
     * @param labels 标签数组 / Label array
     * @return 分类结果 / Classification result
     */
    ClassificationResult fit(IMatrix features, String[] labels);
    
    /**
     * 预测新样本 / Predict new sample
     * @param features 特征向量 / Feature vector
     * @return 预测类别 / Predicted class
     */
    String predict(IVector features);
}
```

### LogisticRegressionResult 类 / LogisticRegressionResult Class

```java
public class LogisticRegressionResult extends ClassificationResult {
    private IVector weights;      // 权重向量 / Weight vector
    private IVector bias;         // 偏置向量 / Bias vector
    private float loss;           // 损失值 / Loss value
    
    // getters and setters
}
```

## 算法原理 / Algorithm Principles

### 数学模型 / Mathematical Model

#### 二分类模型 / Binary Classification Model

对于二分类问题，逻辑回归使用sigmoid函数：
For binary classification problems, logistic regression uses the sigmoid function:

```
P(y=1|x) = 1 / (1 + e^(-z))
```

其中：
Where:
```
z = w^T * x + b
```

- `w` 是权重向量 / is the weight vector
- `x` 是输入特征向量 / is the input feature vector
- `b` 是偏置项 / is the bias term
- `P(y=1|x)` 是样本属于正类的概率 / is the probability that the sample belongs to the positive class

#### 多分类模型 / Multiclass Classification Model

对于多分类问题，逻辑回归使用softmax函数：
For multiclass classification problems, logistic regression uses the softmax function:

```
P(y=k|x) = e^(z_k) / Σ(e^(z_j)) for j=1 to K
```

其中：
Where:
```
z_k = w_k^T * x + b_k
```

- `w_k` 是第k个类别的权重向量 / is the weight vector for the k-th class
- `b_k` 是第k个类别的偏置项 / is the bias term for the k-th class
- `K` 是类别总数 / is the total number of classes
- `P(y=k|x)` 是样本属于第k个类别的概率 / is the probability that the sample belongs to the k-th class

### 目标函数 / Objective Function

#### 二分类损失函数 / Binary Classification Loss Function

使用交叉熵损失函数：
Using cross-entropy loss function:

```
J(w,b) = -(1/m) * Σ[y_i * log(p_i) + (1-y_i) * log(1-p_i)] + R(w)
```

其中：
Where:
- `m` 是样本数量 / is the number of samples
- `y_i` 是真实标签（0或1）/ is the true label (0 or 1)
- `p_i` 是预测概率 / is the predicted probability
- `R(w)` 是正则化项 / is the regularization term

#### 多分类损失函数 / Multiclass Classification Loss Function

使用多类交叉熵损失函数：
Using multiclass cross-entropy loss function:

```
J(W,B) = -(1/m) * Σ Σ[y_ik * log(p_ik)] + R(W)
```

其中：
Where:
- `W` 是权重矩阵 / is the weight matrix
- `B` 是偏置向量 / is the bias vector
- `y_ik` 是one-hot编码的真实标签 / is the one-hot encoded true label
- `p_ik` 是预测概率 / is the predicted probability

### 梯度计算 / Gradient Calculation

#### 二分类梯度 / Binary Classification Gradient

权重梯度：
Weight gradient:
```
∂J/∂w = (1/m) * X^T * (p - y) + ∇R(w)
```

偏置梯度：
Bias gradient:
```
∂J/∂b = (1/m) * Σ(p - y)
```

#### 多分类梯度 / Multiclass Classification Gradient

权重梯度：
Weight gradient:
```
∂J/∂w_k = (1/m) * X^T * (p_k - y_k) + ∇R(w_k)
```

偏置梯度：
Bias gradient:
```
∂J/∂b_k = (1/m) * Σ(p_k - y_k)
```

## 主要特性 / Main Features

### 1. 自动分类类型检测 / Automatic Classification Type Detection

```java
// 自动检测二分类或多分类 / Automatically detect binary or multiclass
RereLogisticRegression lr = new RereLogisticRegression();

// 二分类标签 / Binary classification labels
String[] binaryLabels = {"正类", "负类"}; // {"Positive", "Negative"}

// 多分类标签 / Multiclass labels  
String[] multiclassLabels = {"类别A", "类别B", "类别C"}; // {"Class A", "Class B", "Class C"}
```

### 2. 灵活的模型配置 / Flexible Model Configuration

```java
// 创建逻辑回归模型 / Create logistic regression model
RereLogisticRegression lr = new RereLogisticRegression();

// 配置学习率 / Configure learning rate
lr.setLearningRate(0.01f);

// 配置最大迭代次数 / Configure maximum iterations
lr.setMaxIterations(1000);

// 配置收敛阈值 / Configure convergence tolerance
lr.setTolerance(1e-6f);

// 配置正则化 / Configure regularization
lr.setRegularization(0.01f, 0.1f); // L1=0.01, L2=0.1
```

### 3. 多种正则化选项 / Multiple Regularization Options

```java
// 无正则化 / No regularization
lr.setRegularization(0.0f, 0.0f);

// L1正则化（Lasso）/ L1 regularization (Lasso)
lr.setLambda1(0.01f);

// L2正则化（Ridge）/ L2 regularization (Ridge)
lr.setLambda2(0.1f);

// ElasticNet正则化 / ElasticNet regularization
lr.setRegularization(0.01f, 0.1f);
```

## 使用示例 / Usage Examples

### 示例1：二分类逻辑回归 / Example 1: Binary Classification

```java
import com.reremouse.lab.math.IMatrix;
import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.ml.cls.RereLogisticRegression;
import com.reremouse.lab.math.ml.cls.LogisticRegressionResult;

public class BinaryClassificationExample {
    public static void main(String[] args) {
        // 准备训练数据 / Prepare training data
        float[][] featureData = {
            {1, 2}, {2, 3}, {3, 4}, {4, 5},
            {5, 6}, {6, 7}, {7, 8}, {8, 9}
        };
        String[] labelData = {"正类", "正类", "正类", "正类", 
                             "负类", "负类", "负类", "负类"};
        
        IMatrix features = IMatrix.of(featureData);
        
        // 创建和训练模型 / Create and train model
        RereLogisticRegression lr = new RereLogisticRegression();
        LogisticRegressionResult result = lr.fit(features, labelData);
        
        // 获取结果 / Get results
        IVector weights = result.getWeights();
        IVector bias = result.getBias();
        float loss = result.getLoss();
        
        System.out.println("权重: " + weights); // Weights
        System.out.println("偏置: " + bias); // Bias
        System.out.println("损失: " + loss); // Loss
        
        // 预测新样本 / Predict new sample
        IVector newFeatures = IVector.of(new float[]{2.5f, 3.5f});
        String prediction = lr.predict(newFeatures);
        System.out.println("预测类别: " + prediction); // Predicted class
        
        // 预测概率 / Predict probability
        float probability = lr.predictProbability(newFeatures);
        System.out.println("正类概率: " + probability); // Positive class probability
    }
}
```

### 示例2：多分类逻辑回归 / Example 2: Multiclass Classification

```java
public class MulticlassClassificationExample {
    public static void main(String[] args) {
        // 准备训练数据 / Prepare training data
        float[][] featureData = {
            {1, 2}, {2, 3}, {3, 4}, {4, 5},
            {5, 6}, {6, 7}, {7, 8}, {8, 9},
            {9, 10}, {10, 11}, {11, 12}, {12, 13}
        };
        String[] labelData = {"类别A", "类别A", "类别A", "类别A",
                             "类别B", "类别B", "类别B", "类别B", 
                             "类别C", "类别C", "类别C", "类别C"};
        
        IMatrix features = IMatrix.of(featureData);
        
        // 创建和训练模型 / Create and train model
        RereLogisticRegression lr = new RereLogisticRegression();
        LogisticRegressionResult result = lr.fit(features, labelData);
        
        // 检查模型类型 / Check model type
        System.out.println("模型类型: " + lr.getModelTypeDescription()); // Model type
        System.out.println("类别数量: " + lr.getNumClasses()); // Number of classes
        
        // 预测新样本 / Predict new sample
        IVector newFeatures = IVector.of(new float[]{2.5f, 3.5f});
        String prediction = lr.predict(newFeatures);
        System.out.println("预测类别: " + prediction); // Predicted class
        
        // 预测所有类别的概率 / Predict probabilities for all classes
        float[] probabilities = lr.predictProbabilities(newFeatures);
        System.out.println("各类别概率: " + Arrays.toString(probabilities)); // Class probabilities
    }
}
```

### 示例3：带正则化的逻辑回归 / Example 3: Logistic Regression with Regularization

```java
public class RegularizedLogisticRegressionExample {
    public static void main(String[] args) {
        // 创建带正则化的模型 / Create model with regularization
        RereLogisticRegression lr = new RereLogisticRegression();
        lr.setRegularization(0.01f, 0.1f); // L1=0.01, L2=0.1
        
        // 训练模型 / Train model
        LogisticRegressionResult result = lr.fit(features, labels);
        
        // 查看正则化效果 / View regularization effects
        System.out.println("正则化类型: " + lr.getRegularizationDescription()); // Regularization type
        System.out.println("最终损失: " + result.getLoss()); // Final loss
    }
}
```

### 示例4：批量预测 / Example 4: Batch Prediction

```java
public class BatchPredictionExample {
    public static void main(String[] args) {
        // 训练模型 / Train model
        RereLogisticRegression lr = new RereLogisticRegression();
        lr.fit(features, labels);
        
        // 准备测试数据 / Prepare test data
        float[][] testData = {
            {1.5f, 2.5f}, {2.5f, 3.5f}, {3.5f, 4.5f}
        };
        IMatrix testFeatures = IMatrix.of(testData);
        
        // 批量预测 / Batch prediction
        String[] predictions = lr.predictBatch(testFeatures);
        
        System.out.println("批量预测结果: " + Arrays.toString(predictions)); // Batch prediction results
    }
}
```

### 示例5：模型评估 / Example 5: Model Evaluation

```java
public class ModelEvaluationExample {
    public static void main(String[] args) {
        // 训练模型 / Train model
        RereLogisticRegression lr = new RereLogisticRegression();
        LogisticRegressionResult result = lr.fit(features, labels);
        
        // 评估指标 / Evaluation metrics
        float loss = result.getLoss();
        
        System.out.println("训练损失: " + loss); // Training loss
        
        // 在测试集上评估 / Evaluate on test set
        String[] testPredictions = lr.predictBatch(testFeatures);
        
        // 计算准确率 / Calculate accuracy
        int correct = 0;
        for (int i = 0; i < testLabels.length; i++) {
            if (testPredictions[i].equals(testLabels[i])) {
                correct++;
            }
        }
        float accuracy = (float) correct / testLabels.length;
        System.out.println("测试准确率: " + accuracy); // Test accuracy
    }
}
```

## 性能特性 / Performance Features

### 优化算法 / Optimization Algorithm
- 使用L-BFGS优化器，收敛速度快 / Uses L-BFGS optimizer with fast convergence
- 支持线搜索，提高优化稳定性 / Supports line search to improve optimization stability
- 自动梯度计算，无需手动实现 / Automatic gradient calculation, no manual implementation needed

### 数值稳定性 / Numerical Stability
- Sigmoid和Softmax函数的数值稳定实现 / Numerically stable implementation of sigmoid and softmax functions
- 梯度裁剪避免梯度爆炸 / Gradient clipping prevents gradient explosion
- 正则化防止过拟合 / Regularization prevents overfitting

### 内存优化 / Memory Optimization
- 高效的矩阵运算 / Efficient matrix operations
- 智能的内存管理 / Smart memory management
- 支持大规模数据集 / Supports large-scale datasets

## 注意事项 / Notes

1. **数据预处理** / **Data Preprocessing**: 建议对特征进行标准化处理
2. **正则化参数** / **Regularization Parameters**: 根据数据特点选择合适的正则化参数
3. **特征选择** / **Feature Selection**: L1正则化有助于特征选择
4. **过拟合** / **Overfitting**: 使用正则化和交叉验证防止过拟合
5. **分类类型** / **Classification Type**: 模型会自动检测二分类或多分类

## 扩展性 / Extensibility

`RereLogisticRegression` 类设计支持扩展：
The `RereLogisticRegression` class is designed to support extensions:
- 自定义损失函数 / Custom loss functions
- 新的正则化方法 / New regularization methods
- 不同的优化器 / Different optimizers
- 在线学习支持 / Online learning support

## 应用场景 / Application Scenarios

### 二分类应用 / Binary Classification Applications
- 垃圾邮件检测 / Spam detection
- 医疗诊断 / Medical diagnosis
- 信用评估 / Credit assessment
- 用户行为预测 / User behavior prediction

### 多分类应用 / Multiclass Classification Applications
- 图像分类 / Image classification
- 文本分类 / Text classification
- 情感分析 / Sentiment analysis
- 产品推荐 / Product recommendation

---

**逻辑回归** - 分类问题的经典解决方案，让预测更准确！

**Logistic Regression** - The classic solution for classification problems, making predictions more accurate!

---

**线性回归** - 机器学习的基础，让预测更准确！

**Linear Regression** - The foundation of machine learning, making predictions more accurate!
