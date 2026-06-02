# 距离度量学习使用示例 / Distance Metric Learning Examples

## 概述 / Overview

本文档提供距离度量学习（DML）算法的详细使用示例。矩阵与向量推荐通过 `Linalg.matrix` / `Linalg.vector` 创建，与库内其余文档一致。

This document provides detailed usage examples for Distance Metric Learning (DML) algorithms. Prefer `Linalg.matrix` / `Linalg.vector` for matrix/vector construction, consistent with other library documentation.

---

## ⭐ DDML 示例（自主研发）/ DDML Examples (Proprietary)

### 信用评估 — 特征选择 / Credit Evaluation — Feature Selection

```java
import com.yishape.lab.math.ml.dml.DmlMetric;
import com.yishape.lab.math.ml.ML;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;

// 学习对角度量 / Learn diagonal metric
DmlMetric metric = ML.dml.diagDml(0.1, 0.01).fit(creditFeatures, labels);

// 分析特征重要性 / Analyze feature importance
IMatrix<Double> diagMatrix = metric.transformMatrix();
System.out.println("=== Feature Importance Analysis / 特征重要性分析 ===");
for (int i = 0; i < diagMatrix.getRowNum(); i++) {
    double weight = Math.sqrt(diagMatrix.get(i, i));
    System.out.printf("Feature %d: weight=%.4f %s%n",
        i, weight, weight < 0.01 ? "(selected / 已选择)" : "");
}
```

### 弹性网正则化 — 平衡特征选择与稳定性 / Elastic Net — Balance Feature Selection and Stability

```java
// 纯 L1 正则化（激进的特征选择）/ Pure L1 regularization (aggressive feature selection)
DmlMetric metricL1 = ML.dml.diagDml(0.1, 0.0).fit(features, labels);

// 纯 L2 正则化（稳定的缩放学习）/ Pure L2 regularization (stable scaling)
DmlMetric metricL2 = ML.dml.diagDml(0.0, 0.01).fit(features, labels);

// 弹性网（平衡）/ Elastic Net (balanced)
DmlMetric metricElastic = ML.dml.diagDml(0.05, 0.05).fit(features, labels);
```

### 高级配置 / Advanced Configuration

```java
import com.yishape.lab.math.ml.dml.ddml.RereDiagDml;

RereDiagDml ddml = new RereDiagDml()
    .setL1Weight(0.1)              // L1 正则化强度 / L1 regularization strength
    .setL2Weight(0.01)             // L2 正则化强度 / L2 regularization strength
    .setTau(256.0)                 // 约束尺度 / Constraint scale
    .setDistance("huber")           // Huber 损失 / Huber loss
    .setUseAdmm(true)              // 使用 ADMM 求解器 / Use ADMM solver
    .setMaxAdmmIterations(50)     // ADMM 迭代次数 / ADMM iterations
    .setAdmmErrorTol(1e-5);       // ADMM 误差容忍度 / ADMM error tolerance

DmlMetric metric = ddml.fit(features, labels);
```

---

## 完整工作流示例 / Complete Workflow Examples

### 1. 分类任务：鸢尾花数据集 / Classification: Iris Dataset

```java
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.ML;
import com.yishape.lab.math.ml.dml.DmlMetric;

// 1. 准备数据 / Prepare data
double[][] irisFeatures = {
    {5.1, 3.5, 1.4, 0.2},
    {4.9, 3.0, 1.4, 0.2},
    // ... complete data
};
String[] irisLabels = {
    "setosa", "setosa", ..., "virginica"
};

IMatrix<Double> features = Linalg.matrix(irisFeatures);

// 2. 学习度量 / Learn metric
DmlMetric metric = ML.dml.nca().fit(features, irisLabels);

// 3. 变换数据 / Transform data
IMatrix<Double> transformed = metric.transform(features);

// 4. 应用到 k-NN 分类 / Apply to k-NN classification
int k = 5;
// 在变换后的数据上做 k-NN / k-NN on transformed data
int predicted = knnPredict(transformed, testPoint, labels, k);
```

### 2. 聚类任务 / Clustering Task

```java
import com.yishape.lab.math.ml.clustering.KMeansPlusPlus;

// 学习度量 / Learn metric
DmlMetric metric = ML.dml.lmnn().fit(features, labels);

// 变换特征 / Transform features
IMatrix<Double> transformed = metric.transform(features);

// 在变换空间中进行聚类 / Cluster in transformed space
var kmeans = ML.clu.kMeans(3);
kmeans.fit(transformed);
int cluster = kmeans.predict(testPoint);
```

### 3. 相似性检索 / Similarity Retrieval

```java
// 学习度量 / Learn metric
DmlMetric metric = ML.dml.ldmlPairwise().fit(features, labels);

// 计算查询点到数据库中所有点的距离 / Compute distance from query to all database points
double[] distances = new double[dataSize];
for (int i = 0; i < dataSize; i++) {
    distances[i] = metric.squaredDistance(query, database.getRow(i));
}

// 找最相似的 k 个 / Find top-k most similar
int[] topK = argsort(distances, k);
```

---

## 监督学习算法示例 / Supervised Learning Algorithm Examples

### NCA — 邻域成分分析 / Neighborhood Components Analysis

```java
import com.yishape.lab.math.ml.dml.nca.NcaDml;

// 基本用法 / Basic usage
DmlMetric metric = ML.dml.nca().fit(features, labels);

// 高级配置 / Advanced configuration
NcaDml nca = new NcaDml()
    .setRank(3)                    // 嵌入维度 / Embedding dimension
    .setMaxIter(500)               // 增加迭代次数 / More iterations
    .setTolerance(1e-8)            // 更严格的收敛 / Tighter convergence
    .setMaxBfgsIter(500)          // L-BFGS 迭代 / L-BFGS iterations
    .setGradClip(2.0)             // 梯度裁剪 / Gradient clipping
    .setConvergenceTol(1e-4)      // 启用早停 / Enable early stopping
    .setPatience(20);              // 20 轮无改善则停止 / Stop after 20 rounds

DmlMetric metricNca = nca.fit(features, labels);
```

### LMNN — 大间隔最近邻 / Large Margin Nearest Neighbors

```java
import com.yishape.lab.math.ml.dml.lmnn.LmnnDml;

// 基本用法 / Basic usage
DmlMetric metric = ML.dml.lmnn().fit(features, labels);

// 高级配置 / Advanced configuration
LmnnDml lmnn = new LmnnDml()
    .setRank(2)                    // 2 维嵌入 / 2D embedding
    .setTargetNeighbors(3)        // 每个样本 3 个目标邻居 / 3 target neighbors per sample
    .setMargin(2.0)               // 更大的间隔 / Larger margin
    .setMaxSteps(50)              // 更多优化步 / More optimization steps
    .setGradClip(3.0);

DmlMetric metricLmnn = lmnn.fit(features, labels);
```

### ITML — 信息理论度量学习 / Information-Theoretic Metric Learning

```java
import com.yishape.lab.math.ml.dml.itml.ItmlDml;

// 基本用法 / Basic usage
DmlMetric metric = ML.dml.itml().fit(features, labels);

// 高级配置 / Advanced configuration
ItmlDml itml = new ItmlDml()
    .setGamma(0.5)                             // 约束松弛参数 / Constraint relaxation
    .setMaxIter(100)                          // 最大迭代 / Max iterations
    .setPriorKind(ItmlDml.PriorKind.IDENTITY) // 先验类型 / Prior type
    .setPriorL2Weight(1e-3);                 // 先验正则化 / Prior regularization

DmlMetric metricItml = itml.fit(features, labels);
```

### DML-eig — 特征值优化度量学习 / Eigenvalue Optimization DML

```java
import com.yishape.lab.math.ml.dml.dmleig.DmleigDml;

// 基本用法 / Basic usage
DmlMetric metric = ML.dml.dmleig().fit(features, labels);

// 高级配置 / Advanced configuration
DmleigDml dmleig = new DmleigDml()
    .setMu(1e-5)              // 更小的平滑参数 / Smaller smoothing parameter
    .setMaxIter(50)            // 更多迭代 / More iterations
    .setTolerance(1e-7)      // 更严格的收敛 / Tighter convergence
    .setMaxBfgsIter(300);     // L-BFGS 迭代 / L-BFGS iterations

DmlMetric metricDmleig = dmleig.fit(features, labels);
```

---

## 核方法示例 / Kernel Method Examples

### KLDA — 核线性判别分析 / Kernel Linear Discriminant Analysis

```java
import com.yishape.lab.math.ml.dml.KernelDmlUtils.KernelType;
import com.yishape.lab.math.ml.dml.kllda.KLldaDml;

// KLDA - 核线性判别分析 / Kernel LDA
KLldaDml klda = ML.dml.kllda()
    .setKernelType(KernelType.RBF)
    .setGamma(0.1)
    .setNComponents(2)      // 投影维度 / Projection dimension
    .setReg(1e-6);         // 正则化 / Regularization
DmlMetric kldaMetric = klda.fit(features, labels);
```

### KANMM — 核化近邻度量学习 / Kernelized Adjacent Neighbor Metric Learning

```java
import com.yishape.lab.math.ml.dml.anmm.KanmmDml;

// KANMM - 核近邻度量 / Kernel neighbor metric
KanmmDml kanmm = ML.dml.kanmm()
    .setKernelType(KernelType.RBF)
    .setGamma(0.1)
    .setK(2)
    .setNumDims(2);
DmlMetric kanmmMetric = kanmm.fit(features, labels);
```

### KDMLMJ — 核 Jeffrey 散度度量学习 / Kernel Jeffrey Divergence DML

```java
import com.yishape.lab.math.ml.dml.dmlmj.KDmlmjDml;

// KDMLMJ - 核 Jeffrey 散度 / Kernel Jeffrey divergence
KDmlmjDml kdmlmj = ML.dml.kdmlmj()
    .setKernelType(KernelType.RBF)
    .setGamma(0.1)
    .setK(3)
    .setNumDims(2);
DmlMetric kdmlmjMetric = kdmlmj.fit(features, labels);
```

---

## 性能优化示例 / Performance Optimization Examples

### 大数据集处理 / Large Dataset Processing

```java
// 对于大数据集，使用采样策略 / For large datasets, use sampling strategy
public DmlMetric learnWithSampling(IMatrix<Double> features, String[] labels, int sampleSize) {
    int n = features.getRowNum();

    // 随机采样 / Random sampling
    int[] indices = new int[n];
    for (int i = 0; i < n; i++) indices[i] = i;
    shuffle(indices);

    // 采样子集 / Sample subset
    double[][] sampledFeatures = new double[sampleSize][];
    String[] sampledLabels = new String[sampleSize];
    for (int i = 0; i < sampleSize; i++) {
        sampledFeatures[i] = features.getRowPtr(indices[i]);
        sampledLabels[i] = labels[indices[i]];
    }

    // 在子集上学习度量 / Learn metric on subset
    return ML.dml.nca().fit(Linalg.matrix(sampledFeatures), sampledLabels);
}
```

### 增量学习 / Incremental Learning

```java
import com.yishape.lab.math.ml.dml.odml.OdmlDml;

// 使用 ODML 进行增量学习 / Use ODML for incremental learning
OdmlDml odml = new OdmlDml()
    .setLearningRate(0.001)
    .setAggression(1.0)
    .setMaxIter(10);

// 分批处理数据 / Process data in batches
for (DataBatch batch : dataStream) {
    // 增量更新 / Incremental update
    odml.partialFit(batch.features, batch.labels);
}

// 获取当前度量 / Get current metric
DmlMetric metric = odml.getMetric();
```

---

## 评估与可视化 / Evaluation and Visualization

### Silhouette 分数 / Silhouette Score

```java
import com.yishape.lab.math.ml.dml.MetricTransforms;
import com.yishape.lab.math.linalg.IMatrix;

// 学习度量 / Learn metric
DmlMetric metric = ML.dml.nca().fit(features, labels);

// 变换数据 / Transform data
IMatrix<Double> transformed = metric.transform(features);

// 计算 silhouette 分数 / Compute silhouette score
double silhouette = computeSilhouette(transformed, labels);
```

### k-NN 分类验证 / k-NN Classification Verification

```java
public double knnAccuracy(DmlMetric metric, IMatrix<Double> trainFeatures,
                         String[] trainLabels, IMatrix<Double> testFeatures,
                         String[] testLabels, int k) {
    IMatrix<Double> transformedTrain = metric.transform(trainFeatures);
    IMatrix<Double> transformedTest = metric.transform(testFeatures);

    int correct = 0;
    for (int i = 0; i < transformedTest.getRowNum(); i++) {
        int predicted = knnPredict(transformedTrain, trainLabels,
                                  transformedTest.getRow(i), k);
        if (predicted == testLabels[i].hashCode()) {
            correct++;
        }
    }

    return (double) correct / testLabels.length;
}
```

---

## 常见问题处理 / Troubleshooting

### 1. 收敛困难 / Convergence Difficulties

```java
// 问题：算法不收敛 / Problem: algorithm not converging
// 解决方案 / Solution:

// 1. 放宽收敛容忍度 / Relax convergence tolerance
NcaDml nca = new NcaDml()
    .setTolerance(1e-3)     // 从 1e-6 放宽 / Relax from 1e-6
    .setConvergenceTol(1e-2);

// 2. 增加迭代次数 / Increase iterations
nca.setMaxIter(500);
nca.setMaxBfgsIter(500);

// 3. 调整学习率或梯度裁剪 / Adjust learning rate or gradient clipping
nca.setGradClip(10.0);  // 放宽裁剪 / Relax clipping
```

### 2. 过拟合 / Overfitting

```java
// 问题：训练集表现好，测试集差 / Problem: train good, test bad
// 解决方案 / Solution:

// 1. 使用早停 / Use early stopping
NcaDml nca = new NcaDml()
    .setConvergenceTol(1e-4)
    .setPatience(5);

// 2. 增加正则化 / Increase regularization
// 3. 使用更小的嵌入维度 / Use smaller embedding dimension
nca.setRank(Math.min(nca.getRank(), 2));
```

### 3. 数值不稳定 / Numerical Instability

```java
// 问题：梯度爆炸或 NaN / Problem: gradient explosion or NaN
// 解决方案 / Solution:

// 1. 启用梯度裁剪 / Enable gradient clipping
NcaDml nca = new NcaDml()
    .setGradClip(1.0);  // 更小的裁剪值 / Smaller clip value

// 2. 数据标准化 / Standardize data
IMatrix<Double> normalized = standardize(features);

// 3. 检查异常值 / Check outliers
double maxVal = features.elementMax();
double minVal = features.elementMin();
if (maxVal - minVal > 1e6) {
    features = standardize(features);
}
```

---

## 相关文档 / Related Documents

- [DML 总览](../DML/README.md)
- [机器学习总览](../Machine-Learning.md)
- [降维算法](../Machine-Learning.md#降维算法-dimensionality-reduction-algorithms)