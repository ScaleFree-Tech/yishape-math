# 机器学习示例 (Machine Learning Examples)

## 概述 / Overview

本文档提供了 `com.reremouse.lab.math` 包中机器学习算法的详细使用示例。包括线性回归和逻辑回归的完整示例代码，涵盖从基础使用到高级特性的各个方面。

This document provides detailed usage examples for machine learning algorithms in the `com.reremouse.lab.math` package. It includes complete example code for linear regression and logistic regression, covering everything from basic usage to advanced features.

## 目录 / Table of Contents

1. [线性回归示例 / Linear Regression Examples](#线性回归示例--linear-regression-examples)
2. [逻辑回归示例 / Logistic Regression Examples](#逻辑回归示例--logistic-regression-examples)
3. [优化算法示例 / Optimization Algorithm Examples](#优化算法示例--optimization-algorithm-examples)
4. [总结 / Summary](#总结--summary)

## 线性回归示例 / Linear Regression Examples

### 基本线性回归 / Basic Linear Regression

```java
import com.reremouse.lab.math.IMatrix;
import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.ml.lr.RereLinearRegression;
import com.reremouse.lab.math.ml.lr.RegressionResult;

public class BasicLinearRegressionExample {
    public static void main(String[] args) {
        // 准备训练数据 / Prepare training data
        float[][] featureData = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };
        float[] labelData = {14, 32, 50};
        
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
    }
}
```

### 带正则化的线性回归 / Linear Regression with Regularization

```java
public class RegularizedLinearRegressionExample {
    public static void main(String[] args) {
        // 创建带L2正则化的模型 / Create model with L2 regularization
        RereLinearRegression lr = new RereLinearRegression();
        lr.setRegularizationType(RegularizationType.L2);
        lr.setLambda2(0.1f);
        
        // 训练模型 / Train model
        RegressionResult result = lr.fit(features, labels);
        
        // 查看正则化效果 / View regularization effects
        System.out.println("L2正则化系数: " + lr.getLambda2());
        System.out.println("最终损失: " + result.getLoss());
    }
}
```

## 优化算法示例 / Optimization Algorithm Examples

### L-BFGS优化器应用 / L-BFGS Optimizer Application

```java
import com.reremouse.lab.math.optimize.RereLBFGS;
import com.reremouse.lab.math.optimize.IObjectiveFunction;
import com.reremouse.lab.math.optimize.IGradientFunction;

public class LBFGSExample {
    public static void main(String[] args) {
        // 创建L-BFGS优化器 / Create L-BFGS optimizer
        RereLBFGS optimizer = new RereLBFGS();
        optimizer.setMaxIterations(1000);
        optimizer.setTolerance(1e-6f);
        
        // 定义目标函数 / Define objective function
        IObjectiveFunction objFun = new IObjectiveFunction() {
            @Override
            public float compute(IVector x) {
                float x1 = x.get(0);
                float x2 = x.get(1);
                return (1 - x1) * (1 - x1) + 100 * (x2 - x1 * x1) * (x2 - x1 * x1);
            }
        };
        
        // 定义梯度函数 / Define gradient function
        IGradientFunction grdFun = new IGradientFunction() {
            @Override
            public IVector compute(IVector x) {
                float x1 = x.get(0);
                float x2 = x.get(1);
                
                float[] grad = new float[2];
                grad[0] = -2 * (1 - x1) - 400 * x1 * (x2 - x1 * x1);
                grad[1] = 200 * (x2 - x1 * x1);
                
                return IVector.of(grad);
            }
        };
        
        // 执行优化 / Execute optimization
        IVector initX = IVector.of(new float[]{-1.0f, -1.0f});
        Tuple2<Float, IVector> result = optimizer.optimize(initX, objFun, grdFun);
        
        float optimalValue = result._1;
        IVector optimalPoint = result._2;
        
        System.out.println("最优值: " + optimalValue);
        System.out.println("最优点: " + optimalPoint);
    }
}
```

## 逻辑回归示例 / Logistic Regression Examples

### 基本二分类逻辑回归 / Basic Binary Classification

```java
import com.reremouse.lab.math.IMatrix;
import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.ml.cls.RereLogisticRegression;
import com.reremouse.lab.math.ml.cls.LogisticRegressionResult;

public class BasicBinaryClassificationExample {
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

### 多分类逻辑回归 / Multiclass Classification

```java
import java.util.Arrays;

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

### 带正则化的逻辑回归 / Logistic Regression with Regularization

```java
public class RegularizedLogisticRegressionExample {
    public static void main(String[] args) {
        // 准备数据 / Prepare data
        float[][] featureData = {
            {1, 2}, {2, 3}, {3, 4}, {4, 5},
            {5, 6}, {6, 7}, {7, 8}, {8, 9}
        };
        String[] labelData = {"正类", "正类", "正类", "正类", 
                             "负类", "负类", "负类", "负类"};
        
        IMatrix features = IMatrix.of(featureData);
        
        // 创建带正则化的模型 / Create model with regularization
        RereLogisticRegression lr = new RereLogisticRegression();
        lr.setRegularization(0.01f, 0.1f); // L1=0.01, L2=0.1
        
        // 训练模型 / Train model
        LogisticRegressionResult result = lr.fit(features, labelData);
        
        // 查看正则化效果 / View regularization effects
        System.out.println("正则化类型: " + lr.getRegularizationDescription()); // Regularization type
        System.out.println("最终损失: " + result.getLoss()); // Final loss
    }
}
```

### 批量预测示例 / Batch Prediction Example

```java
import java.util.Arrays;

public class BatchPredictionExample {
    public static void main(String[] args) {
        // 准备训练数据 / Prepare training data
        float[][] trainData = {
            {1, 2}, {2, 3}, {3, 4}, {4, 5},
            {5, 6}, {6, 7}, {7, 8}, {8, 9}
        };
        String[] trainLabels = {"正类", "正类", "正类", "正类", 
                               "负类", "负类", "负类", "负类"};
        
        IMatrix trainFeatures = IMatrix.of(trainData);
        
        // 训练模型 / Train model
        RereLogisticRegression lr = new RereLogisticRegression();
        lr.fit(trainFeatures, trainLabels);
        
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

### 模型评估示例 / Model Evaluation Example

```java
public class ModelEvaluationExample {
    public static void main(String[] args) {
        // 准备训练数据 / Prepare training data
        float[][] trainData = {
            {1, 2}, {2, 3}, {3, 4}, {4, 5},
            {5, 6}, {6, 7}, {7, 8}, {8, 9}
        };
        String[] trainLabels = {"正类", "正类", "正类", "正类", 
                               "负类", "负类", "负类", "负类"};
        
        IMatrix trainFeatures = IMatrix.of(trainData);
        
        // 训练模型 / Train model
        RereLogisticRegression lr = new RereLogisticRegression();
        LogisticRegressionResult result = lr.fit(trainFeatures, trainLabels);
        
        // 评估指标 / Evaluation metrics
        float loss = result.getLoss();
        System.out.println("训练损失: " + loss); // Training loss
        
        // 准备测试数据 / Prepare test data
        float[][] testData = {
            {1.5f, 2.5f}, {2.5f, 3.5f}, {3.5f, 4.5f}, {4.5f, 5.5f}
        };
        String[] testLabels = {"正类", "正类", "负类", "负类"};
        
        IMatrix testFeatures = IMatrix.of(testData);
        
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
        
        // 显示详细预测结果 / Show detailed prediction results
        System.out.println("详细预测结果:"); // Detailed prediction results
        for (int i = 0; i < testLabels.length; i++) {
            System.out.println("样本 " + i + ": 真实=" + testLabels[i] + 
                             ", 预测=" + testPredictions[i] + 
                             ", 正确=" + (testLabels[i].equals(testPredictions[i])));
        }
    }
}
```

### 高级配置示例 / Advanced Configuration Example

```java
public class AdvancedConfigurationExample {
    public static void main(String[] args) {
        // 准备数据 / Prepare data
        float[][] featureData = {
            {1, 2}, {2, 3}, {3, 4}, {4, 5},
            {5, 6}, {6, 7}, {7, 8}, {8, 9}
        };
        String[] labelData = {"正类", "正类", "正类", "正类", 
                             "负类", "负类", "负类", "负类"};
        
        IMatrix features = IMatrix.of(featureData);
        
        // 创建模型并配置参数 / Create model and configure parameters
        RereLogisticRegression lr = new RereLogisticRegression();
        
        // 配置学习率 / Configure learning rate
        lr.setLearningRate(0.01f);
        
        // 配置最大迭代次数 / Configure maximum iterations
        lr.setMaxIterations(1000);
        
        // 配置收敛阈值 / Configure convergence tolerance
        lr.setTolerance(1e-6f);
        
        // 配置正则化 / Configure regularization
        lr.setRegularization(0.01f, 0.1f); // L1=0.01, L2=0.1
        
        // 训练模型 / Train model
        LogisticRegressionResult result = lr.fit(features, labelData);
        
        // 显示配置信息 / Show configuration information
        System.out.println("模型配置:"); // Model configuration
        System.out.println("学习率: " + lr.getLearningRate()); // Learning rate
        System.out.println("最大迭代次数: " + lr.getMaxIterations()); // Max iterations
        System.out.println("收敛阈值: " + lr.getTolerance()); // Tolerance
        System.out.println("正则化类型: " + lr.getRegularizationDescription()); // Regularization type
        System.out.println("最终损失: " + result.getLoss()); // Final loss
    }
}
```

## 总结 / Summary

本文档展示了机器学习算法的基本使用方法。建议在实际使用中：

1. 根据数据特点选择合适的正则化方法
2. 使用交叉验证评估模型性能
3. 合理设置优化算法参数
4. 注意数据预处理和特征工程
5. 对于分类问题，根据类别数量选择合适的模型类型
6. 使用概率预测来获得更详细的信息

---

**机器学习示例** - 从理论到实践，掌握机器学习的精髓！
