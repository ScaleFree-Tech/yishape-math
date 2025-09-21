# 机器学习示例 (Machine Learning Examples)

## 概述 / Overview

本文档提供了 `com.reremouse.lab.math` 包中机器学习算法的详细使用示例。包括线性回归和逻辑回归的完整示例代码，涵盖从基础使用到高级特性的各个方面。

This document provides detailed usage examples for machine learning algorithms in the `com.reremouse.lab.math` package. It includes complete example code for linear regression and logistic regression, covering everything from basic usage to advanced features.

## 目录 / Table of Contents

1. [线性回归示例 / Linear Regression Examples](#线性回归示例--linear-regression-examples)
2. [逻辑回归示例 / Logistic Regression Examples](#逻辑回归示例--logistic-regression-examples)
3. [聚类算法示例 / Clustering Algorithm Examples](#聚类算法示例--clustering-algorithm-examples)
4. [降维算法示例 / Dimensionality Reduction Examples](#降维算法示例--dimensionality-reduction-examples)
5. [总结 / Summary](#总结--summary)

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

## 聚类算法示例 / Clustering Algorithm Examples

### K-Means++聚类示例 / K-Means++ Clustering Example

```java
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.ml.clustering.KMeansPlusPlus;
import com.reremouse.lab.math.ml.clustering.ClusteringMetrics;

public class KMeansExample {
    public static void main(String[] args) {
        // 准备数据 / Prepare data
        double[][] data = {
            {1.0, 2.0}, {1.5, 1.8}, {5.0, 8.0}, {8.0, 8.0}, 
            {1.0, 0.6}, {9.0, 11.0}, {8.0, 2.0}, {10.0, 2.0}
        };
        
        IMatrix dataMatrix = Linalg.matrix(data);
        
        // 创建K-Means++聚类器 / Create K-Means++ clusterer
        KMeansPlusPlus kmeans = new KMeansPlusPlus();
        
        // 设置聚类数量 / Set number of clusters
        kmeans.setParameters(Map.of("numClusters", 3));
        
        // 训练模型 / Train model
        kmeans.fit(dataMatrix);
        
        // 获取聚类结果 / Get clustering results
        int[] labels = kmeans.getLabels();
        List<IVector<Double>> centers = kmeans.getClusterCenters();
        
        System.out.println("聚类标签: " + Arrays.toString(labels));
        System.out.println("聚类中心数量: " + centers.size());
        System.out.println("是否收敛: " + kmeans.isConverged());
        System.out.println("迭代次数: " + kmeans.getIterations());
        
        // 评估聚类质量 / Evaluate clustering quality
        ClusteringMetrics metrics = kmeans.evaluateQuality(dataMatrix);
        System.out.println("轮廓系数: " + metrics.getSilhouetteScore());
        System.out.println("惯性: " + metrics.getInertia());
    }
}
```

### 高斯混合模型聚类示例 / Gaussian Mixture Model Clustering Example

```java
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.ml.clustering.GMMClustering;

public class GMMExample {
    public static void main(String[] args) {
        // 准备数据 / Prepare data
        double[][] data = {
            {1.0, 2.0}, {1.5, 1.8}, {5.0, 8.0}, {8.0, 8.0}, 
            {1.0, 0.6}, {9.0, 11.0}, {8.0, 2.0}, {10.0, 2.0}
        };
        
        IMatrix dataMatrix = Linalg.matrix(data);
        
        // 创建GMM聚类器 / Create GMM clusterer
        GMMClustering gmm = new GMMClustering();
        
        // 设置参数 / Set parameters
        gmm.setParameters(Map.of(
            "numClusters", 3,
            "maxIterations", 100,
            "tolerance", 1e-6,
            "verbose", true
        ));
        
        // 训练模型 / Train model
        gmm.fit(dataMatrix);
        
        // 获取聚类结果 / Get clustering results
        int[] labels = gmm.getLabels();
        List<IVector<Double>> centers = gmm.getClusterCenters();
        
        System.out.println("聚类标签: " + Arrays.toString(labels));
        System.out.println("聚类中心数量: " + centers.size());
        
        // 计算后验概率 / Compute posterior probabilities
        List<IVector<Double>> posteriors = gmm.computePosteriorProbabilities(dataMatrix);
        System.out.println("后验概率矩阵大小: " + posteriors.size() + " x " + posteriors.get(0).size());
        
        // 计算对数似然 / Compute log-likelihood
        double logLikelihood = gmm.computeLogLikelihood(dataMatrix);
        System.out.println("对数似然: " + logLikelihood);
        
        // 从模型采样 / Sample from model
        List<IVector<Double>> samples = gmm.sample(10);
        System.out.println("采样数据点数量: " + samples.size());
    }
}
```

## 降维算法示例 / Dimensionality Reduction Examples

### PCA降维示例 / PCA Dimensionality Reduction Example

```java
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.ml.dimreduce.RerePCA;

public class PCAExample {
    public static void main(String[] args) {
        // 准备高维数据 / Prepare high-dimensional data
        double[][] data = {
            {1.0, 2.0, 3.0, 4.0, 5.0},
            {2.0, 3.0, 4.0, 5.0, 6.0},
            {3.0, 4.0, 5.0, 6.0, 7.0},
            {4.0, 5.0, 6.0, 7.0, 8.0},
            {5.0, 6.0, 7.0, 8.0, 9.0}
        };
        
        IMatrix originalData = Linalg.matrix(data);
        System.out.println("原始数据维度: " + originalData.getRowNum() + " x " + originalData.getColNum());
        
        // 创建PCA降维器 / Create PCA reducer
        RerePCA pca = new RerePCA();
        
        // 降维到2维 / Reduce to 2 dimensions
        IMatrix reducedData = pca.dimensionReduction(originalData, 2);
        System.out.println("降维后数据维度: " + reducedData.getRowNum() + " x " + reducedData.getColNum());
        
        // 显示降维结果 / Display reduction results
        System.out.println("降维后的数据:");
        for (int i = 0; i < reducedData.getRowNum(); i++) {
            System.out.println("样本 " + i + ": [" + 
                reducedData.get(i, 0) + ", " + reducedData.get(i, 1) + "]");
        }
    }
}
```

### SVD降维示例 / SVD Dimensionality Reduction Example

```java
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.ml.dimreduce.RereSVD;

public class SVDExample {
    public static void main(String[] args) {
        // 准备数据矩阵 / Prepare data matrix
        double[][] data = {
            {1.0, 2.0, 3.0, 4.0},
            {5.0, 6.0, 7.0, 8.0},
            {9.0, 10.0, 11.0, 12.0}
        };
        
        IMatrix originalData = Linalg.matrix(data);
        System.out.println("原始数据维度: " + originalData.getRowNum() + " x " + originalData.getColNum());
        
        // 创建SVD降维器 / Create SVD reducer
        RereSVD svd = new RereSVD();
        
        // 降维到2维 / Reduce to 2 dimensions
        IMatrix reducedData = svd.dimensionReduction(originalData, 2);
        System.out.println("降维后数据维度: " + reducedData.getRowNum() + " x " + reducedData.getColNum());
        
        // 显示降维结果 / Display reduction results
        System.out.println("SVD降维后的数据:");
        for (int i = 0; i < reducedData.getRowNum(); i++) {
            System.out.println("样本 " + i + ": [" + 
                reducedData.get(i, 0) + ", " + reducedData.get(i, 1) + "]");
        }
    }
}
```

### t-SNE降维示例 / t-SNE Dimensionality Reduction Example

```java
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.ml.dimreduce.RereTSNE;

public class TSNEExample {
    public static void main(String[] args) {
        // 准备高维数据 / Prepare high-dimensional data
        double[][] data = {
            {1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
            {2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
            {3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
            {4.0, 5.0, 6.0, 7.0, 8.0, 9.0},
            {5.0, 6.0, 7.0, 8.0, 9.0, 10.0}
        };
        
        IMatrix originalData = Linalg.matrix(data);
        System.out.println("原始数据维度: " + originalData.getRowNum() + " x " + originalData.getColNum());
        
        // 创建t-SNE降维器 / Create t-SNE reducer
        RereTSNE tsne = new RereTSNE();
        
        // 设置t-SNE参数 / Set t-SNE parameters
        tsne.setParameters(Map.of(
            "perplexity", 30.0,
            "learningRate", 200.0,
            "maxIterations", 1000
        ));
        
        // 降维到2维 / Reduce to 2 dimensions
        IMatrix reducedData = tsne.dimensionReduction(originalData, 2);
        System.out.println("降维后数据维度: " + reducedData.getRowNum() + " x " + reducedData.getColNum());
        
        // 显示降维结果 / Display reduction results
        System.out.println("t-SNE降维后的数据:");
        for (int i = 0; i < reducedData.getRowNum(); i++) {
            System.out.println("样本 " + i + ": [" + 
                reducedData.get(i, 0) + ", " + reducedData.get(i, 1) + "]");
        }
    }
}
```

### UMAP降维示例 / UMAP Dimensionality Reduction Example

```java
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.ml.dimreduce.RereUMAP;

public class UMAPExample {
    public static void main(String[] args) {
        // 准备高维数据 / Prepare high-dimensional data
        double[][] data = {
            {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
            {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0},
            {3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
            {4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0},
            {5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}
        };
        
        IMatrix originalData = Linalg.matrix(data);
        System.out.println("原始数据维度: " + originalData.getRowNum() + " x " + originalData.getColNum());
        
        // 创建UMAP降维器 / Create UMAP reducer
        RereUMAP umap = new RereUMAP();
        
        // 设置UMAP参数 / Set UMAP parameters
        umap.setParameters(Map.of(
            "nNeighbors", 15,
            "minDist", 0.1,
            "nComponents", 2,
            "metric", "euclidean"
        ));
        
        // 降维到2维 / Reduce to 2 dimensions
        IMatrix reducedData = umap.dimensionReduction(originalData, 2);
        System.out.println("降维后数据维度: " + reducedData.getRowNum() + " x " + reducedData.getColNum());
        
        // 显示降维结果 / Display reduction results
        System.out.println("UMAP降维后的数据:");
        for (int i = 0; i < reducedData.getRowNum(); i++) {
            System.out.println("样本 " + i + ": [" + 
                reducedData.get(i, 0) + ", " + reducedData.get(i, 1) + "]");
        }
    }
}
```

## 总结 / Summary

本文档展示了机器学习算法的基本使用方法。建议在实际使用中：

1. 根据数据特点选择合适的正则化方法
2. 使用交叉验证评估模型性能
3. 注意数据预处理和特征工程
4. 对于分类问题，根据类别数量选择合适的模型类型
5. 使用概率预测来获得更详细的信息
6. 对于聚类问题，选择合适的聚类算法和参数
7. 对于降维问题，根据数据特点选择线性或非线性降维方法
8. 使用聚类质量评估指标来验证聚类效果
9. 降维后可以用于数据可视化或进一步的特征学习

---

**机器学习示例** - 从理论到实践，掌握机器学习的精髓！
