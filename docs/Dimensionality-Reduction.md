# 降维算法 (Dimensionality Reduction)

## 概述 / Overview

降维算法包提供了多种降维技术的实现，包括PCA（主成分分析）、SVD（奇异值分解）、t-SNE（t分布随机邻域嵌入）和UMAP（均匀流形近似和投影）。这些算法能够有效地减少数据维度，保留数据的主要特征，提高计算效率和可视化效果。

The dimensionality reduction algorithms package provides implementations of various dimensionality reduction techniques including PCA (Principal Component Analysis), SVD (Singular Value Decomposition), t-SNE (t-Distributed Stochastic Neighbor Embedding), and UMAP (Uniform Manifold Approximation and Projection). These algorithms can effectively reduce data dimensions while preserving main data characteristics, improving computational efficiency and visualization effects.

## 核心算法 / Core Algorithms

### 1. PCA (主成分分析) / Principal Component Analysis

#### 算法原理 / Algorithm Principles

PCA是一种线性降维技术，通过找到数据的主要变化方向来减少维度。PCA的核心思想是：

PCA is a linear dimensionality reduction technique that reduces dimensions by finding the main directions of data variation. The core idea of PCA is:

- **数据中心化** / **Data Centering**: 减去每列的均值，使数据以原点为中心 / Subtract the mean of each column to center the data around the origin
- **协方差矩阵** / **Covariance Matrix**: 计算特征间的协方差关系 / Calculate covariance relationships between features
- **特征分解** / **Eigendecomposition**: 对协方差矩阵进行特征值分解 / Perform eigendecomposition on the covariance matrix
- **主成分选择** / **Principal Component Selection**: 选择前k个最大特征值对应的特征向量 / Select the eigenvectors corresponding to the k largest eigenvalues
- **数据投影** / **Data Projection**: 将原始数据投影到主成分空间 / Project original data onto the principal component space

#### 数学原理 / Mathematical Principles

1. **数据中心化** / **Data Centering**:
   ```
   X_centered = X - mean(X)
   ```

2. **协方差矩阵计算** / **Covariance Matrix Calculation**:
   ```
   C = (X_centered^T * X_centered) / (n-1)
   ```

3. **特征分解** / **Eigendecomposition**:
   ```
   C = V * Λ * V^T
   ```
   其中V是特征向量矩阵，Λ是对角特征值矩阵
   Where V is the eigenvector matrix and Λ is the diagonal eigenvalue matrix

4. **降维投影** / **Dimensionality Reduction Projection**:
   ```
   X_reduced = X_centered * V[:, :k]
   ```

#### 使用示例 / Usage Examples

```java
// 创建PCA降维器 / Create PCA reducer
RerePCA pca = new RerePCA();

// 准备数据 / Prepare data
float[][] data = {
    {1, 2, 3, 4, 5},
    {2, 4, 6, 8, 10},
    {3, 6, 9, 12, 15},
    {4, 8, 12, 16, 20},
    {5, 10, 15, 20, 25}
};
IMatrix originalData = IMatrix.of(data);

// 执行降维 / Perform dimensionality reduction
int targetDim = 2;
IMatrix reducedData = pca.dimensionReduction(originalData, targetDim);

System.out.println("原始数据维度: " + originalData.getRowNum() + " x " + originalData.getColNum());
System.out.println("降维后数据维度: " + reducedData.getRowNum() + " x " + reducedData.getColNum());
System.out.println("Original data dimensions: " + originalData.getRowNum() + " x " + originalData.getColNum());
System.out.println("Reduced data dimensions: " + reducedData.getRowNum() + " x " + reducedData.getColNum());

// 获取主成分 / Get principal components
IMatrix principalComponents = pca.getPrincipalComponents();
IVector explainedVariance = pca.getExplainedVariance();
IVector explainedVarianceRatio = pca.getExplainedVarianceRatio();

System.out.println("主成分: " + principalComponents);
System.out.println("解释方差: " + explainedVariance);
System.out.println("解释方差比例: " + explainedVarianceRatio);
System.out.println("Principal components: " + principalComponents);
System.out.println("Explained variance: " + explainedVariance);
System.out.println("Explained variance ratio: " + explainedVarianceRatio);
```

### 2. SVD (奇异值分解) / Singular Value Decomposition

#### 算法原理 / Algorithm Principles

SVD是一种矩阵分解技术，能够将任意矩阵分解为三个矩阵的乘积：
SVD is a matrix decomposition technique that can decompose any matrix into the product of three matrices:

```
A = U * Σ * V^T
```

其中：
Where:
- U：左奇异向量矩阵 / U: Left singular vector matrix
- Σ：奇异值对角矩阵 / Σ: Diagonal singular value matrix
- V：右奇异向量矩阵 / V: Right singular vector matrix

SVD在降维中的应用：
SVD applications in dimensionality reduction:
- 保留前k个最大奇异值 / Keep the first k largest singular values
- 截断U和V矩阵 / Truncate U and V matrices
- 重构降维后的数据 / Reconstruct dimensionality-reduced data

#### 使用示例 / Usage Examples

```java
// 创建SVD降维器 / Create SVD reducer
RereSVD svd = new RereSVD();

// 准备数据 / Prepare data
float[][] data = {
    {1, 2, 3, 4},
    {5, 6, 7, 8},
    {9, 10, 11, 12}
};
IMatrix originalData = IMatrix.of(data);

// 执行SVD分解 / Perform SVD decomposition
Tuple3<IMatrix, IVector, IMatrix> svdResult = svd.decompose(originalData);
IMatrix U = svdResult._1;        // 左奇异向量 / Left singular vectors
IVector S = svdResult._2;        // 奇异值 / Singular values
IMatrix V = svdResult._3;        // 右奇异向量 / Right singular vectors

// 降维重构 / Dimensionality reduction reconstruction
int targetDim = 2;
IMatrix reducedData = svd.reconstruct(U, S, V, targetDim);

System.out.println("奇异值: " + S);
System.out.println("降维后数据: " + reducedData);
System.out.println("Singular values: " + S);
System.out.println("Reduced data: " + reducedData);

// 计算重构误差 / Calculate reconstruction error
float reconstructionError = svd.calculateReconstructionError(originalData, reducedData);
System.out.println("重构误差: " + reconstructionError);
System.out.println("Reconstruction error: " + reconstructionError);
```

### 3. t-SNE (t分布随机邻域嵌入) / t-Distributed Stochastic Neighbor Embedding

#### 算法原理 / Algorithm Principles

t-SNE是一种非线性降维技术，特别适合高维数据的可视化。t-SNE的核心思想是：
t-SNE is a non-linear dimensionality reduction technique, particularly suitable for high-dimensional data visualization. The core idea of t-SNE is:

- **相似性保持** / **Similarity Preservation**: 在高维空间中相似的样本在低维空间中应该保持相似 / Similar samples in high-dimensional space should remain similar in low-dimensional space
- **概率分布** / **Probability Distribution**: 使用高斯分布表示高维空间的相似性 / Use Gaussian distribution to represent similarity in high-dimensional space
- **t分布** / **t-Distribution**: 使用t分布表示低维空间的相似性 / Use t-distribution to represent similarity in low-dimensional space
- **KL散度最小化** / **KL Divergence Minimization**: 最小化高维和低维概率分布的KL散度 / Minimize KL divergence between high-dimensional and low-dimensional probability distributions

#### 数学原理 / Mathematical Principles

1. **高维空间相似性** / **High-dimensional Space Similarity**:
   ```
   p_{j|i} = exp(-||x_i - x_j||² / (2σ_i²)) / Σ_{k≠i} exp(-||x_i - x_k||² / (2σ_i²))
   ```

2. **低维空间相似性** / **Low-dimensional Space Similarity**:
   ```
   q_{ij} = (1 + ||y_i - y_j||²)^(-1) / Σ_{k≠l} (1 + ||y_k - y_l||²)^(-1)
   ```

3. **KL散度** / **KL Divergence**:
   ```
   KL(P||Q) = Σ_i Σ_j p_{ij} log(p_{ij} / q_{ij})
   ```

#### 使用示例 / Usage Examples

```java
// 创建t-SNE降维器 / Create t-SNE reducer
RereTSNE tsne = new RereTSNE();

// 设置参数 / Set parameters
tsne.setPerplexity(30.0f);           // 困惑度 / Perplexity
tsne.setLearningRate(200.0f);        // 学习率 / Learning rate
tsne.setMaxIterations(1000);         // 最大迭代次数 / Maximum iterations
tsne.setEarlyExaggeration(12.0f);    // 早期夸大因子 / Early exaggeration factor

// 准备数据 / Prepare data
float[][] data = {
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
    {2, 4, 6, 8, 10, 12, 14, 16, 18, 20},
    {3, 6, 9, 12, 15, 18, 21, 24, 27, 30},
    {4, 8, 12, 16, 20, 24, 28, 32, 36, 40},
    {5, 10, 15, 20, 25, 30, 35, 40, 45, 50}
};
IMatrix originalData = IMatrix.of(data);

// 执行t-SNE降维 / Perform t-SNE dimensionality reduction
int targetDim = 2;
IMatrix reducedData = tsne.fitTransform(originalData, targetDim);

System.out.println("t-SNE降维完成 / t-SNE dimensionality reduction completed");
System.out.println("降维后数据: " + reducedData);
System.out.println("Reduced data: " + reducedData);

// 获取优化历史 / Get optimization history
List<Float> lossHistory = tsne.getLossHistory();
System.out.println("损失历史: " + lossHistory);
System.out.println("Loss history: " + lossHistory);
```

### 4. UMAP (均匀流形近似和投影) / Uniform Manifold Approximation and Projection

#### 算法原理 / Algorithm Principles

UMAP是一种基于流形学习的降维技术，具有以下特点：
UMAP is a dimensionality reduction technique based on manifold learning with the following characteristics:

- **流形学习** / **Manifold Learning**: 假设数据位于低维流形上 / Assume data lies on low-dimensional manifolds
- **图构建** / **Graph Construction**: 构建高维数据的邻域图 / Construct neighborhood graphs of high-dimensional data
- **图优化** / **Graph Optimization**: 优化低维图以保持拓扑结构 / Optimize low-dimensional graphs to preserve topological structure
- **快速收敛** / **Fast Convergence**: 比t-SNE收敛更快 / Converges faster than t-SNE

#### 数学原理 / Mathematical Principles

1. **局部距离计算** / **Local Distance Calculation**:
   ```
   d(x_i, x_j) = ||x_i - x_j|| / ρ_i
   ```
   其中ρ_i是第i个点的局部密度
   Where ρ_i is the local density of the i-th point

2. **相似性计算** / **Similarity Calculation**:
   ```
   sim(x_i, x_j) = exp(-d(x_i, x_j))
   ```

3. **图优化** / **Graph Optimization**:
   最小化高维和低维图之间的交叉熵
   Minimize cross-entropy between high-dimensional and low-dimensional graphs

#### 使用示例 / Usage Examples

```java
// 创建UMAP降维器 / Create UMAP reducer
RereUMAP umap = new RereUMAP();

// 设置参数 / Set parameters
umap.setNNeighbors(15);               // 邻居数量 / Number of neighbors
umap.setMinDist(0.1f);                // 最小距离 / Minimum distance
umap.setNComponents(2);               // 目标维度 / Target dimensions
umap.setMetric("euclidean");          // 距离度量 / Distance metric
umap.setSpread(1.0f);                 // 扩散参数 / Spread parameter

// 准备数据 / Prepare data
float[][] data = {
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
    {2, 4, 6, 8, 10, 12, 14, 16, 18, 20},
    {3, 6, 9, 12, 15, 18, 21, 24, 27, 30},
    {4, 8, 12, 16, 20, 24, 28, 32, 36, 40},
    {5, 10, 15, 20, 25, 30, 35, 40, 45, 50}
};
IMatrix originalData = IMatrix.of(data);

// 执行UMAP降维 / Perform UMAP dimensionality reduction
IMatrix reducedData = umap.fitTransform(originalData);

System.out.println("UMAP降维完成 / UMAP dimensionality reduction completed");
System.out.println("降维后数据: " + reducedData);
System.out.println("Reduced data: " + reducedData);

// 获取嵌入质量指标 / Get embedding quality metrics
float trustworthiness = umap.getTrustworthiness();
float continuity = umap.getContinuity();
System.out.println("可信度: " + trustworthiness);
System.out.println("连续性: " + continuity);
System.out.println("Trustworthiness: " + trustworthiness);
System.out.println("Continuity: " + continuity);
```

## 高级特性 / Advanced Features

### 1. 参数调优 / Parameter Tuning

#### PCA参数调优 / PCA Parameter Tuning

```java
// 自动选择最优维度 / Automatically select optimal dimensions
RerePCA pca = new RerePCA();

// 基于解释方差比例选择维度 / Select dimensions based on explained variance ratio
float varianceThreshold = 0.95f;  // 保留95%的方差 / Preserve 95% variance
int optimalDim = pca.selectOptimalDimensions(originalData, varianceThreshold);

System.out.println("最优维度: " + optimalDim);
System.out.println("Optimal dimensions: " + optimalDim);

// 执行降维 / Perform dimensionality reduction
IMatrix reducedData = pca.dimensionReduction(originalData, optimalDim);
```

#### t-SNE参数调优 / t-SNE Parameter Tuning

```java
// 困惑度选择 / Perplexity selection
RereTSNE tsne = new RereTSNE();

// 自动选择困惑度 / Automatically select perplexity
float optimalPerplexity = tsne.selectOptimalPerplexity(originalData);
tsne.setPerplexity(optimalPerplexity);

System.out.println("最优困惑度: " + optimalPerplexity);
System.out.println("Optimal perplexity: " + optimalPerplexity);

// 学习率调整 / Learning rate adjustment
float optimalLearningRate = tsne.selectOptimalLearningRate(originalData);
tsne.setLearningRate(optimalLearningRate);

System.out.println("最优学习率: " + optimalLearningRate);
System.out.println("Optimal learning rate: " + optimalLearningRate);
```

### 2. 降维质量评估 / Dimensionality Reduction Quality Assessment

```java
// 创建评估器 / Create evaluator
DimensionalityReductionEvaluator evaluator = new DimensionalityReductionEvaluator();

// 评估PCA / Evaluate PCA
RerePCA pca = new RerePCA();
IMatrix pcaResult = pca.dimensionReduction(originalData, 2);

float pcaTrustworthiness = evaluator.calculateTrustworthiness(originalData, pcaResult);
float pcaContinuity = evaluator.calculateContinuity(originalData, pcaResult);
float pcaStress = evaluator.calculateStress(originalData, pcaResult);

System.out.println("PCA评估结果 / PCA Evaluation Results:");
System.out.println("可信度: " + pcaTrustworthiness);
System.out.println("连续性: " + pcaContinuity);
System.out.println("应力: " + pcaStress);
System.out.println("Trustworthiness: " + pcaTrustworthiness);
System.out.println("Continuity: " + pcaContinuity);
System.out.println("Stress: " + pcaStress);

// 评估t-SNE / Evaluate t-SNE
RereTSNE tsne = new RereTSNE();
IMatrix tsneResult = tsne.fitTransform(originalData, 2);

float tsneTrustworthiness = evaluator.calculateTrustworthiness(originalData, tsneResult);
float tsneContinuity = evaluator.calculateContinuity(originalData, tsneResult);
float tsneStress = evaluator.calculateStress(originalData, tsneResult);

System.out.println("t-SNE评估结果 / t-SNE Evaluation Results:");
System.out.println("可信度: " + tsneTrustworthiness);
System.out.println("连续性: " + tsneContinuity);
System.out.println("应力: " + tsneStress);
System.out.println("Trustworthiness: " + tsneTrustworthiness);
System.out.println("Continuity: " + tsneContinuity);
System.out.println("Stress: " + tsneStress);
```

### 3. 增量降维 / Incremental Dimensionality Reduction

```java
// 创建增量PCA / Create incremental PCA
RerePCA incrementalPCA = new RerePCA();
incrementalPCA.setIncremental(true);

// 分批处理数据 / Process data in batches
int batchSize = 100;
for (int i = 0; i < data.length; i += batchSize) {
    int endIdx = Math.min(i + batchSize, data.length);
    float[][] batch = Arrays.copyOfRange(data, i, endIdx);
    IMatrix batchMatrix = IMatrix.of(batch);
    
    // 增量更新 / Incremental update
    incrementalPCA.partialFit(batchMatrix);
}

// 获取最终结果 / Get final results
IMatrix finalResult = incrementalPCA.transform(originalData);
System.out.println("增量PCA结果: " + finalResult);
System.out.println("Incremental PCA result: " + finalResult);
```

## 性能特性 / Performance Features

### 计算复杂度 / Computational Complexity
- **PCA**: O(n²d + d³)，其中n是样本数，d是特征数 / **PCA**: O(n²d + d³), where n is the number of samples and d is the number of features
- **SVD**: O(nd²)，适合大规模数据 / **SVD**: O(nd²), suitable for large-scale data
- **t-SNE**: O(n²)，计算密集但质量高 / **t-SNE**: O(n²), computationally intensive but high quality
- **UMAP**: O(n log n)，比t-SNE快 / **UMAP**: O(n log n), faster than t-SNE

### 内存使用 / Memory Usage
- 高效的矩阵运算 / Efficient matrix operations
- 智能的内存管理 / Smart memory management
- 支持大规模数据集 / Support for large-scale datasets

### 并行化支持 / Parallelization Support
- 矩阵运算并行化 / Matrix operation parallelization
- 多线程支持 / Multi-threading support
- GPU加速（未来版本）/ GPU acceleration (future version)

## 注意事项 / Notes

1. **数据预处理** / **Data Preprocessing**: 建议对数据进行标准化处理 / Recommend standardizing data before processing
2. **参数选择** / **Parameter Selection**: 根据数据特点选择合适的参数 / Choose appropriate parameters based on data characteristics
3. **计算资源** / **Computational Resources**: t-SNE和UMAP计算密集，注意资源消耗 / t-SNE and UMAP are computationally intensive, pay attention to resource consumption
4. **结果解释** / **Result Interpretation**: 降维结果需要结合领域知识解释 / Dimensionality reduction results need to be interpreted with domain knowledge

## 扩展性 / Extensibility

降维算法包设计支持扩展：
The dimensionality reduction algorithms package is designed to support extensions:
- 新的降维算法 / New dimensionality reduction algorithms
- 自定义距离度量 / Custom distance metrics
- 增量学习支持 / Incremental learning support
- 分布式计算支持 / Distributed computing support

## 应用场景 / Application Scenarios

### 数据可视化 / Data Visualization
- 高维数据探索 / High-dimensional data exploration
- 聚类结果展示 / Clustering result visualization
- 异常检测可视化 / Anomaly detection visualization

### 特征工程 / Feature Engineering
- 特征选择 / Feature selection
- 特征提取 / Feature extraction
- 噪声过滤 / Noise filtering

### 机器学习 / Machine Learning
- 预处理步骤 / Preprocessing steps
- 模型训练加速 / Model training acceleration
- 过拟合防止 / Overfitting prevention

---

**降维算法** - 数据探索的利器，让高维数据更清晰！
**Dimensionality Reduction** - A powerful tool for data exploration, making high-dimensional data clearer!
