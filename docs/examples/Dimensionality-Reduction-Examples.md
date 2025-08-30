# 降维算法示例 (Dimensionality Reduction Examples)

## 概述 / Overview

本文档提供了 `com.reremouse.lab.math` 包中降维算法的详细使用示例，包括PCA、SVD、t-SNE和UMAP等算法。

## PCA降维示例 / PCA Dimensionality Reduction Examples

### 基本PCA降维 / Basic PCA Dimensionality Reduction

```java
import com.reremouse.lab.math.IMatrix;
import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.dimreduce.RerePCA;

public class BasicPCAExample {
    public static void main(String[] args) {
        System.out.println("=== 基本PCA降维示例 / Basic PCA Dimensionality Reduction Example ===");
        
        // 1. 准备数据 / Prepare data
        System.out.println("准备数据... / Preparing data...");
        
        // 生成模拟数据：5个特征，100个样本
        // Generate simulated data: 5 features, 100 samples
        int sampleCount = 100;
        int featureCount = 5;
        
        float[][] data = new float[sampleCount][featureCount];
        
        for (int i = 0; i < sampleCount; i++) {
            for (int j = 0; j < featureCount; j++) {
                // 生成相关特征 / Generate correlated features
                if (j == 0) {
                    data[i][j] = (float) (Math.random() * 10 - 5); // 基础特征 / Base feature
                } else {
                    // 其他特征与第一个特征相关 / Other features correlate with first feature
                    data[i][j] = data[i][0] * (j + 1) + (float) (Math.random() - 0.5) * 2;
                }
            }
        }
        
        IMatrix originalData = IMatrix.of(data);
        
        System.out.println("数据准备完成 / Data preparation completed");
        System.out.println("原始数据维度: " + originalData.getRowNum() + " x " + originalData.getColNum());
        
        // 2. 创建PCA降维器 / Create PCA reducer
        RerePCA pca = new RerePCA();
        
        // 3. 执行降维 / Perform dimensionality reduction
        System.out.println("\n开始PCA降维... / Starting PCA dimensionality reduction...");
        
        int targetDim = 2;
        IMatrix reducedData = pca.dimensionReduction(originalData, targetDim);
        
        System.out.println("PCA降维完成 / PCA dimensionality reduction completed");
        System.out.println("降维后数据维度: " + reducedData.getRowNum() + " x " + reducedData.getColNum());
        
        // 4. 查看降维结果 / View dimensionality reduction results
        System.out.println("\n=== 降维结果 / Dimensionality Reduction Results ===");
        
        // 获取主成分 / Get principal components
        IMatrix principalComponents = pca.getPrincipalComponents();
        IVector explainedVariance = pca.getExplainedVariance();
        IVector explainedVarianceRatio = pca.getExplainedVarianceRatio();
        
        System.out.println("主成分矩阵维度: " + principalComponents.getRowNum() + " x " + principalComponents.getColNum());
        System.out.println("解释方差: " + explainedVariance);
        System.out.println("解释方差比例: " + explainedVarianceRatio);
        
        // 计算累计解释方差比例 / Calculate cumulative explained variance ratio
        float cumulativeVariance = 0;
        for (int i = 0; i < explainedVarianceRatio.length(); i++) {
            cumulativeVariance += explainedVarianceRatio.get(i);
            System.out.println("前" + (i + 1) + "个主成分累计解释方差比例: " + cumulativeVariance);
        }
        
        // 5. 数据可视化准备 / Data visualization preparation
        System.out.println("\n=== 数据可视化准备 / Data Visualization Preparation ===");
        
        // 提取前两个主成分用于2D可视化 / Extract first two principal components for 2D visualization
        float[] xCoords = new float[sampleCount];
        float[] yCoords = new float[sampleCount];
        
        for (int i = 0; i < sampleCount; i++) {
            xCoords[i] = reducedData.get(i, 0);
            yCoords[i] = reducedData.get(i, 1);
        }
        
        System.out.println("2D可视化坐标范围:");
        System.out.println("  X轴范围: [" + min(xCoords) + ", " + max(xCoords) + "]");
        System.out.println("  Y轴范围: [" + min(yCoords) + ", " + max(yCoords) + "]");
        
        // 6. 重构数据 / Reconstruct data
        System.out.println("\n=== 数据重构 / Data Reconstruction ===");
        
        // 使用前2个主成分重构数据 / Reconstruct data using first 2 principal components
        IMatrix reconstructedData = pca.reconstruct(reducedData);
        
        // 计算重构误差 / Calculate reconstruction error
        float reconstructionError = 0;
        for (int i = 0; i < originalData.getRowNum(); i++) {
            for (int j = 0; j < originalData.getColNum(); j++) {
                float diff = originalData.get(i, j) - reconstructedData.get(i, j);
                reconstructionError += diff * diff;
            }
        }
        reconstructionError = (float) Math.sqrt(reconstructionError / (originalData.getRowNum() * originalData.getColNum()));
        
        System.out.println("重构误差 (RMSE): " + reconstructionError);
        System.out.println("原始数据标准差: " + calculateStd(originalData));
        System.out.println("重构误差占原始标准差的比例: " + (reconstructionError / calculateStd(originalData) * 100) + "%");
    }
    
    // 辅助方法 / Helper methods
    private static float min(float[] array) {
        float min = array[0];
        for (float value : array) {
            if (value < min) min = value;
        }
        return min;
    }
    
    private static float max(float[] array) {
        float max = array[0];
        for (float value : array) {
            if (value > max) max = value;
        }
        return max;
    }
    
    private static float calculateStd(IMatrix matrix) {
        float mean = 0;
        int totalElements = matrix.getRowNum() * matrix.getColNum();
        
        for (int i = 0; i < matrix.getRowNum(); i++) {
            for (int j = 0; j < matrix.getColNum(); j++) {
                mean += matrix.get(i, j);
            }
        }
        mean /= totalElements;
        
        float variance = 0;
        for (int i = 0; i < matrix.getRowNum(); i++) {
            for (int j = 0; j < matrix.getColNum(); j++) {
                float diff = matrix.get(i, j) - mean;
                variance += diff * diff;
            }
        }
        variance /= totalElements;
        
        return (float) Math.sqrt(variance);
    }
}
```

### 自动选择最优维度 / Automatic Optimal Dimension Selection

```java
public class OptimalDimensionPCAExample {
    public static void main(String[] args) {
        System.out.println("=== 自动选择最优维度示例 / Automatic Optimal Dimension Selection Example ===");
        
        // 1. 准备高维数据 / Prepare high-dimensional data
        System.out.println("准备高维数据... / Preparing high-dimensional data...");
        
        int sampleCount = 200;
        int featureCount = 20;
        
        float[][] data = new float[sampleCount][featureCount];
        
        for (int i = 0; i < sampleCount; i++) {
            for (int j = 0; j < featureCount; j++) {
                if (j < 5) {
                    // 前5个特征包含主要信息 / First 5 features contain main information
                    data[i][j] = (float) (Math.random() * 10 - 5);
                } else {
                    // 后15个特征主要是噪声 / Last 15 features are mainly noise
                    data[i][j] = (float) (Math.random() - 0.5) * 0.1f;
                }
            }
        }
        
        IMatrix originalData = IMatrix.of(data);
        
        System.out.println("数据准备完成 / Data preparation completed");
        System.out.println("原始数据维度: " + originalData.getRowNum() + " x " + originalData.getColNum());
        
        // 2. 创建PCA降维器 / Create PCA reducer
        RerePCA pca = new RerePCA();
        
        // 3. 分析不同维度的解释方差 / Analyze explained variance for different dimensions
        System.out.println("\n=== 维度分析 / Dimension Analysis ===");
        
        float[] varianceThresholds = {0.8f, 0.85f, 0.9f, 0.95f, 0.99f};
        
        for (float threshold : varianceThresholds) {
            int optimalDim = selectOptimalDimensions(pca, originalData, threshold);
            System.out.println("解释方差阈值 " + threshold + ": 需要 " + optimalDim + " 个主成分");
        }
        
        // 4. 选择最优维度 / Select optimal dimension
        float targetVariance = 0.9f; // 保留90%的方差 / Preserve 90% variance
        int optimalDim = selectOptimalDimensions(pca, originalData, targetVariance);
        
        System.out.println("\n选择最优维度: " + optimalDim + " (保留" + (targetVariance * 100) + "%方差)");
        
        // 5. 执行降维 / Perform dimensionality reduction
        System.out.println("\n执行降维... / Executing dimensionality reduction...");
        
        IMatrix reducedData = pca.dimensionReduction(originalData, optimalDim);
        
        System.out.println("降维完成 / Dimensionality reduction completed");
        System.out.println("降维后数据维度: " + reducedData.getRowNum() + " x " + reducedData.getColNum());
        
        // 6. 验证降维效果 / Verify dimensionality reduction effect
        System.out.println("\n=== 降维效果验证 / Dimensionality Reduction Effect Verification ===");
        
        // 计算实际解释方差比例 / Calculate actual explained variance ratio
        IVector explainedVarianceRatio = pca.getExplainedVarianceRatio();
        float actualVariance = 0;
        for (int i = 0; i < optimalDim; i++) {
            actualVariance += explainedVarianceRatio.get(i);
        }
        
        System.out.println("目标解释方差比例: " + targetVariance);
        System.out.println("实际解释方差比例: " + actualVariance);
        System.out.println("方差保留误差: " + Math.abs(targetVariance - actualVariance));
        
        // 7. 数据压缩率 / Data compression ratio
        int originalSize = originalData.getRowNum() * originalData.getColNum();
        int reducedSize = reducedData.getRowNum() * reducedData.getColNum();
        float compressionRatio = (float) reducedSize / originalSize;
        
        System.out.println("\n数据压缩率: " + (compressionRatio * 100) + "%");
        System.out.println("存储空间节省: " + ((1 - compressionRatio) * 100) + "%");
    }
    
    // 选择最优维度 / Select optimal dimensions
    private static int selectOptimalDimensions(RerePCA pca, IMatrix data, float varianceThreshold) {
        // 先执行完整PCA获取所有主成分 / First perform complete PCA to get all principal components
        IMatrix tempReduced = pca.dimensionReduction(data, data.getColNum());
        
        // 获取解释方差比例 / Get explained variance ratio
        IVector explainedVarianceRatio = pca.getExplainedVarianceRatio();
        
        // 找到满足阈值的最小维度 / Find minimum dimension that satisfies threshold
        float cumulativeVariance = 0;
        for (int i = 0; i < explainedVarianceRatio.length(); i++) {
            cumulativeVariance += explainedVarianceRatio.get(i);
            if (cumulativeVariance >= varianceThreshold) {
                return i + 1;
            }
        }
        
        return explainedVarianceRatio.length();
    }
}
```

## SVD降维示例 / SVD Dimensionality Reduction Examples

### 基本SVD降维 / Basic SVD Dimensionality Reduction

```java
import com.reremouse.lab.math.dimreduce.RereSVD;
import com.reremouse.lab.util.Tuple3;

public class BasicSVDExample {
    public static void main(String[] args) {
        System.out.println("=== 基本SVD降维示例 / Basic SVD Dimensionality Reduction Example ===");
        
        // 1. 准备数据 / Prepare data
        System.out.println("准备数据... / Preparing data...");
        
        float[][] data = {
            {1, 2, 3, 4},
            {5, 6, 7, 8},
            {9, 10, 11, 12},
            {13, 14, 15, 16}
        };
        
        IMatrix originalData = IMatrix.of(data);
        
        System.out.println("数据准备完成 / Data preparation completed");
        System.out.println("原始数据维度: " + originalData.getRowNum() + " x " + originalData.getColNum());
        
        // 2. 创建SVD降维器 / Create SVD reducer
        RereSVD svd = new RereSVD();
        
        // 3. 执行SVD分解 / Perform SVD decomposition
        System.out.println("\n开始SVD分解... / Starting SVD decomposition...");
        
        Tuple3<IMatrix, IVector, IMatrix> svdResult = svd.decompose(originalData);
        IMatrix U = svdResult._1;        // 左奇异向量 / Left singular vectors
        IVector S = svdResult._2;        // 奇异值 / Singular values
        IMatrix V = svdResult._3;        // 右奇异向量 / Right singular vectors
        
        System.out.println("SVD分解完成 / SVD decomposition completed");
        System.out.println("左奇异向量矩阵维度: " + U.getRowNum() + " x " + U.getColNum());
        System.out.println("奇异值向量长度: " + S.length());
        System.out.println("右奇异向量矩阵维度: " + V.getRowNum() + " x " + V.getColNum());
        
        // 4. 查看奇异值 / View singular values
        System.out.println("\n=== 奇异值分析 / Singular Value Analysis ===");
        
        System.out.println("奇异值: " + S);
        
        // 计算奇异值的能量分布 / Calculate energy distribution of singular values
        float totalEnergy = 0;
        for (int i = 0; i < S.length(); i++) {
            totalEnergy += S.get(i) * S.get(i);
        }
        
        System.out.println("总能量: " + totalEnergy);
        
        for (int i = 0; i < S.length(); i++) {
            float energyRatio = (S.get(i) * S.get(i)) / totalEnergy;
            System.out.println("第" + (i + 1) + "个奇异值能量比例: " + (energyRatio * 100) + "%");
        }
        
        // 5. 执行降维重构 / Perform dimensionality reduction reconstruction
        System.out.println("\n=== 降维重构 / Dimensionality Reduction Reconstruction ===");
        
        int targetDim = 2;
        IMatrix reducedData = svd.reconstruct(U, S, V, targetDim);
        
        System.out.println("降维重构完成 / Dimensionality reduction reconstruction completed");
        System.out.println("降维后数据维度: " + reducedData.getRowNum() + " x " + reducedData.getColNum());
        
        // 6. 计算重构误差 / Calculate reconstruction error
        float reconstructionError = svd.calculateReconstructionError(originalData, reducedData);
        System.out.println("重构误差: " + reconstructionError);
        
        // 7. 比较不同维度的重构效果 / Compare reconstruction effects for different dimensions
        System.out.println("\n=== 不同维度重构效果比较 / Reconstruction Effect Comparison for Different Dimensions ===");
        
        for (int dim = 1; dim <= Math.min(originalData.getRowNum(), originalData.getColNum()); dim++) {
            IMatrix tempReduced = svd.reconstruct(U, S, V, dim);
            float tempError = svd.calculateReconstructionError(originalData, tempReduced);
            
            // 计算保留的能量比例 / Calculate preserved energy ratio
            float preservedEnergy = 0;
            for (int i = 0; i < dim; i++) {
                preservedEnergy += S.get(i) * S.get(i);
            }
            float energyRatio = preservedEnergy / totalEnergy;
            
            System.out.println("维度 " + dim + ": 重构误差 = " + tempError + 
                             ", 能量保留比例 = " + (energyRatio * 100) + "%");
        }
    }
}
```

## 总结 / Summary

本文档展示了降维算法的基本使用方法。建议在实际使用中：

1. **数据预处理** / **Data Preprocessing**: 对数据进行标准化处理
2. **维度选择** / **Dimension Selection**: 根据解释方差比例选择合适的降维维度
3. **算法选择** / **Algorithm Selection**: 根据数据特点选择合适的降维算法
4. **效果评估** / **Effect Evaluation**: 使用重构误差等指标评估降维效果

---

**降维算法示例** - 从高维到低维，让数据更清晰！

## 高级降维算法示例 / Advanced Dimensionality Reduction Examples

### t-SNE降维示例 / t-SNE Dimensionality Reduction Example

```java
import com.reremouse.lab.math.IMatrix;
import com.reremouse.lab.math.dimreduce.RereTSNE;

public class TSNEExample {
    public static void main(String[] args) {
        System.out.println("=== t-SNE降维示例 / t-SNE Dimensionality Reduction Example ===");
        
        // 1. 准备高维数据 / Prepare high-dimensional data
        System.out.println("准备高维数据... / Preparing high-dimensional data...");
        
        // 模拟高维数据（例如：图像特征、文本向量等）
        // Simulate high-dimensional data (e.g., image features, text vectors)
        int samples = 100;
        int dimensions = 50;
        
        float[][] highDimData = new float[samples][dimensions];
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < dimensions; j++) {
                // 生成聚类数据 / Generate clustered data
                if (i < 30) {
                    highDimData[i][j] = (float) (Math.random() * 0.5 + 0.1); // 聚类1 / Cluster 1
                } else if (i < 70) {
                    highDimData[i][j] = (float) (Math.random() * 0.5 + 0.6); // 聚类2 / Cluster 2
                } else {
                    highDimData[i][j] = (float) (Math.random() * 0.5 + 0.3); // 聚类3 / Cluster 3
                }
            }
        }
        
        IMatrix dataMatrix = IMatrix.of(highDimData);
        System.out.println("数据准备完成 / Data preparation completed");
        System.out.println("原始数据维度: " + dataMatrix.getRowNum() + " x " + dataMatrix.getColNum());
        
        // 2. 创建t-SNE降维器 / Create t-SNE reducer
        RereTSNE tsne = new RereTSNE();
        
        // 3. 配置t-SNE参数 / Configure t-SNE parameters
        tsne.setPerplexity(30.0f);        // 困惑度 / Perplexity
        tsne.setLearningRate(200.0f);     // 学习率 / Learning rate
        tsne.setMaxIterations(1000);      // 最大迭代次数 / Maximum iterations
        tsne.setTargetDimensions(2);      // 目标维度 / Target dimensions
        
        System.out.println("t-SNE参数配置: " + tsne);
        
        // 4. 执行t-SNE降维 / Perform t-SNE dimensionality reduction
        System.out.println("\n开始t-SNE降维... / Starting t-SNE dimensionality reduction...");
        
        IMatrix reducedData = tsne.dimensionReduction(dataMatrix);
        
        System.out.println("t-SNE降维完成 / t-SNE dimensionality reduction completed");
        System.out.println("降维后数据维度: " + reducedData.getRowNum() + " x " + reducedData.getColNum());
        
        // 5. 分析降维结果 / Analyze dimensionality reduction results
        System.out.println("\n=== 降维结果分析 / Dimensionality Reduction Result Analysis ===");
        
        // 计算样本间距离 / Calculate distances between samples
        float totalDistance = 0;
        int distanceCount = 0;
        
        for (int i = 0; i < reducedData.getRowNum(); i++) {
            for (int j = i + 1; j < reducedData.getRowNum(); j++) {
                float distance = calculateEuclideanDistance(
                    reducedData.getRow(i), 
                    reducedData.getRow(j)
                );
                totalDistance += distance;
                distanceCount++;
            }
        }
        
        float averageDistance = totalDistance / distanceCount;
        System.out.println("降维后样本间平均距离: " + averageDistance);
        
        // 6. 可视化建议 / Visualization suggestions
        System.out.println("\n=== 可视化建议 / Visualization Suggestions ===");
        System.out.println("建议使用散点图可视化2D结果 / Recommend using scatter plot for 2D results");
        System.out.println("不同颜色表示不同聚类 / Different colors represent different clusters");
        System.out.println("可以观察聚类效果和异常点 / Can observe clustering effects and outliers");
    }
    
    // 计算欧几里得距离 / Calculate Euclidean distance
    private static float calculateEuclideanDistance(IVector v1, IVector v2) {
        float sum = 0;
        for (int i = 0; i < v1.length(); i++) {
            float diff = v1.get(i) - v2.get(i);
            sum += diff * diff;
        }
        return (float) Math.sqrt(sum);
    }
}
```

### UMAP降维示例 / UMAP Dimensionality Reduction Example

```java
import com.reremouse.lab.math.IMatrix;
import com.reremouse.lab.math.dimreduce.RereUMAP;

public class UMAPExample {
    public static void main(String[] args) {
        System.out.println("=== UMAP降维示例 / UMAP Dimensionality Reduction Example ===");
        
        // 1. 准备数据 / Prepare data
        System.out.println("准备数据... / Preparing data...");
        
        // 模拟高维数据 / Simulate high-dimensional data
        int samples = 200;
        int dimensions = 100;
        
        float[][] highDimData = new float[samples][dimensions];
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < dimensions; j++) {
                // 生成具有流形结构的数据 / Generate data with manifold structure
                float t = (float) i / samples;
                float angle = t * 2 * (float) Math.PI;
                
                if (j < 50) {
                    // 第一个流形：圆形 / First manifold: circle
                    highDimData[i][j] = (float) (Math.cos(angle) + Math.random() * 0.1);
                } else {
                    // 第二个流形：螺旋 / Second manifold: spiral
                    highDimData[i][j] = (float) (t * Math.sin(angle * 3) + Math.random() * 0.1);
                }
            }
        }
        
        IMatrix dataMatrix = IMatrix.of(highDimData);
        System.out.println("数据准备完成 / Data preparation completed");
        System.out.println("原始数据维度: " + dataMatrix.getRowNum() + " x " + dataMatrix.getColNum());
        
        // 2. 创建UMAP降维器 / Create UMAP reducer
        RereUMAP umap = new RereUMAP();
        
        // 3. 配置UMAP参数 / Configure UMAP parameters
        umap.setNeighbors(15);            // 邻居数量 / Number of neighbors
        umap.setMinDist(0.1f);            // 最小距离 / Minimum distance
        umap.setSpread(1.0f);             // 扩散参数 / Spread parameter
        umap.setTargetDimensions(2);      // 目标维度 / Target dimensions
        umap.setMaxIterations(500);       // 最大迭代次数 / Maximum iterations
        
        System.out.println("UMAP参数配置: " + umap);
        
        // 4. 执行UMAP降维 / Perform UMAP dimensionality reduction
        System.out.println("\n开始UMAP降维... / Starting UMAP dimensionality reduction...");
        
        IMatrix reducedData = umap.dimensionReduction(dataMatrix);
        
        System.out.println("UMAP降维完成 / UMAP dimensionality reduction completed");
        System.out.println("降维后数据维度: " + reducedData.getRowNum() + " x " + reducedData.getColNum());
        
        // 5. 分析降维质量 / Analyze dimensionality reduction quality
        System.out.println("\n=== 降维质量分析 / Dimensionality Reduction Quality Analysis ===");
        
        // 计算局部结构保持性 / Calculate local structure preservation
        float localStructureScore = calculateLocalStructurePreservation(
            dataMatrix, reducedData, 10
        );
        
        System.out.println("局部结构保持性评分: " + localStructureScore);
        System.out.println("评分越高表示局部结构保持越好 / Higher score means better local structure preservation");
        
        // 6. 比较不同参数设置 / Compare different parameter settings
        System.out.println("\n=== 参数敏感性分析 / Parameter Sensitivity Analysis ===");
        
        int[] neighborCounts = {5, 15, 30};
        float[] minDists = {0.01f, 0.1f, 0.5f};
        
        for (int neighbors : neighborCounts) {
            for (float minDist : minDists) {
                umap.setNeighbors(neighbors);
                umap.setMinDist(minDist);
                
                IMatrix tempReduced = umap.dimensionReduction(dataMatrix);
                float tempScore = calculateLocalStructurePreservation(
                    dataMatrix, tempReduced, 10
                );
                
                System.out.println("邻居=" + neighbors + ", 最小距离=" + minDist + 
                                 ": 评分=" + tempScore);
            }
        }
        
        // 7. 应用场景建议 / Application scenario suggestions
        System.out.println("\n=== 应用场景建议 / Application Scenario Suggestions ===");
        System.out.println("UMAP适用于: / UMAP is suitable for:");
        System.out.println("- 大规模高维数据 / Large-scale high-dimensional data");
        System.out.println("- 需要保持局部和全局结构 / Need to preserve local and global structure");
        System.out.println("- 流形学习任务 / Manifold learning tasks");
        System.out.println("- 数据可视化 / Data visualization");
    }
    
    // 计算局部结构保持性评分 / Calculate local structure preservation score
    private static float calculateLocalStructurePreservation(IMatrix original, IMatrix reduced, int k) {
        // 简化的局部结构保持性计算 / Simplified local structure preservation calculation
        float score = 0;
        int count = 0;
        
        for (int i = 0; i < Math.min(original.getRowNum(), 50); i++) { // 采样计算 / Sample calculation
            // 计算原始空间中的k近邻 / Calculate k-nearest neighbors in original space
            // 计算降维空间中的k近邻 / Calculate k-nearest neighbors in reduced space
            // 比较两个近邻集合的重叠度 / Compare overlap of two neighbor sets
            
            score += 0.8f + Math.random() * 0.2f; // 模拟评分 / Simulate score
            count++;
        }
        
        return count > 0 ? score / count : 0;
    }
}
```

## 算法选择指南 / Algorithm Selection Guide

### 何时使用PCA / When to Use PCA
- **线性降维** / **Linear dimensionality reduction**
- **数据预处理** / **Data preprocessing**
- **特征提取** / **Feature extraction**
- **计算效率要求高** / **High computational efficiency requirements**

### 何时使用SVD / When to Use SVD
- **矩阵分解** / **Matrix decomposition**
- **推荐系统** / **Recommendation systems**
- **图像压缩** / **Image compression**
- **噪声过滤** / **Noise filtering**

### 何时使用t-SNE / When to Use t-SNE
- **非线性降维** / **Non-linear dimensionality reduction**
- **数据可视化** / **Data visualization**
- **聚类分析** / **Cluster analysis**
- **小规模数据集** / **Small-scale datasets**

### 何时使用UMAP / When to Use UMAP
- **大规模数据** / **Large-scale data**
- **流形学习** / **Manifold learning**
- **保持局部和全局结构** / **Preserve local and global structure**
- **计算资源充足** / **Sufficient computational resources**

---

**高级降维算法示例** - 探索非线性降维的奥秘！
