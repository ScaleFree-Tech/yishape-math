package com.yishape.lab.math.ml.cls.knn;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.cls.BatchPredictionResult;
import com.yishape.lab.math.ml.cls.ClassificationResult;
import com.yishape.lab.math.ml.cls.IClassification;

import java.io.*;
import java.util.*;

/**
 * k近邻分类器实现
 * 参考weka的IBk算法，支持距离加权、交叉验证选择k值等功能
 * @author lteb2
 */
public class RereKnn implements IClassification {

    private static final long serialVersionUID = 1L;

    // ==================== 模型参数 ====================
    /** 训练特征矩阵 */
    private IMatrix trainingFeatures;

    /** 训练标签数组 */
    private String[] trainingLabels;

    /** 标签映射：将字符串标签映射为数值 */
    private Map<String, Integer> labelMapping;

    /** 反向标签映射：将数值映射回字符串标签 */
    private Map<Integer, String> reverseLabelMapping;

    /** 类别数量 */
    private int numClasses;

    /** 特征维度 */
    private int featureDimension;

    /** k值（近邻数量） */
    private int k = 1;

    /** 距离加权类型 */
    public enum DistanceWeighting {
        NONE,       // 无权重
        INVERSE,    // 1/距离
        SIMILARITY  // 1-距离
    }

    private DistanceWeighting distanceWeighting = DistanceWeighting.NONE;

    /** 是否启用交叉验证选择k值 */
    private boolean crossValidate = false;

    /** 交叉验证时k的最大值 */
    private int maxK = 1;

    /** 窗口大小（训练实例的最大数量，0表示无限制） */
    private int windowSize = 0;

    /** 是否已训练 */
    private boolean isTrained = false;

    // ==================== 构造函数 ====================
    public RereKnn() {
        this.labelMapping = new HashMap<>();
        this.reverseLabelMapping = new HashMap<>();
    }

    public RereKnn(int k) {
        this();
        this.k = k;
        this.maxK = k;
    }

    // ==================== 核心方法 ====================
    @Override
    public ClassificationResult fit(IMatrix feature, String[] labels) {
        if (feature == null || labels == null) {
            throw new IllegalArgumentException("特征矩阵和标签数组不能为null");
        }

        if (feature.getRowNum() != labels.length) {
            throw new IllegalArgumentException("特征矩阵行数与标签数组长度不匹配");
        }

        if (feature.getRowNum() == 0) {
            throw new IllegalArgumentException("训练数据不能为空");
        }

        // 检查特征矩阵是否包含无效值
        for (int i = 0; i < feature.getRowNum(); i++) {
            for (int j = 0; j < feature.getColNum(); j++) {
                double val = (double) feature.get(i, j);
                if (Double.isNaN(val) || Double.isInfinite(val)) {
                    throw new IllegalArgumentException(
                        String.format("训练特征矩阵包含无效值：行%d，列%d，值%s", i, j, val));
                }
            }
        }

        // 保存训练数据
        this.trainingFeatures = feature;
        this.featureDimension = feature.getColNum();

        // 标签预处理
        preprocessLabels(labels);

        // 应用窗口大小限制
        if (windowSize > 0 && trainingFeatures.getRowNum() > windowSize) {
            applyWindowSize();
        }

        // 交叉验证选择最佳k值
        if (crossValidate && maxK > 1) {
            selectBestK();
        }

        this.isTrained = true;

        System.out.println("=== k近邻分类器训练完成 ===");
        System.out.println("k值: " + k);
        System.out.println("距离加权: " + distanceWeighting);
        System.out.println("交叉验证: " + (crossValidate ? "启用" : "禁用"));
        System.out.println("窗口大小: " + (windowSize > 0 ? windowSize : "无限制"));
        System.out.println("训练样本数: " + trainingFeatures.getRowNum());
        System.out.println("特征维度: " + featureDimension);
        System.out.println("类别数量: " + numClasses);

        // kNN没有具体的模型参数，返回基本的分类结果
        return new ClassificationResult() {
            @Override
            public String getModelTypeDescription() {
                return "k近邻分类器";
            }

            @Override
            public String getModelSummary() {
                return "k=" + k + ", 距离加权=" + distanceWeighting +
                       ", 交叉验证=" + crossValidate + ", 窗口大小=" + windowSize;
            }

            @Override
            public boolean isTrained() {
                return true;
            }
        };
    }

    @Override
    public String predict(IVector x) {
        if (!isTrained) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }

        if (x == null) {
            throw new IllegalArgumentException("输入特征向量不能为null");
        }

        if (x.length() != featureDimension) {
            throw new IllegalArgumentException(
                String.format("输入特征维度不匹配：期望%d维，实际%d维",
                    featureDimension, x.length()));
        }

        // 检查输入向量是否包含无效值
        for (int i = 0; i < x.length(); i++) {
            double val = (double) x.get(i);
            if (Double.isNaN(val) || Double.isInfinite(val)) {
                throw new IllegalArgumentException(
                    String.format("输入特征向量包含无效值：位置%d，值%s", i, val));
            }
        }

        // 找到k个最近邻
        List<Neighbor> neighbors = findKNearestNeighbors(x, k);

        // 根据距离加权进行投票
        return voteByDistance(neighbors);
    }

    @Override
    public String[] predictBatch(IMatrix features) {
        if (!isTrained) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }

        if (features == null) {
            throw new IllegalArgumentException("特征矩阵不能为null");
        }

        if (features.getColNum() != featureDimension) {
            throw new IllegalArgumentException("特征维度与训练特征维度不匹配");
        }

        String[] predictions = new String[features.getRowNum()];
        for (int i = 0; i < features.getRowNum(); i++) {
            IVector instance = features.getRow(i);
            predictions[i] = predict(instance);
        }

        return predictions;
    }

    @Override
    public BatchPredictionResult predictBatchWithProbabilities(IMatrix features) {
        if (!isTrained) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }

        if (features == null) {
            throw new IllegalArgumentException("特征矩阵不能为null");
        }

        if (features.getColNum() != featureDimension) {
            throw new IllegalArgumentException("特征维度与训练特征维度不匹配");
        }

        int numSamples = features.getRowNum();
        String[] predictions = new String[numSamples];
        double[][] classProbabilities = new double[numSamples][numClasses];

        for (int i = 0; i < numSamples; i++) {
            IVector instance = features.getRow(i);
            predictions[i] = predict(instance);
            
            // 计算每个样本的概率分布
            List<Neighbor> neighbors = findKNearestNeighbors(instance, k);
            double[] votes = new double[numClasses];
            double totalWeight = 0.0;

            for (Neighbor neighbor : neighbors) {
                String label = trainingLabels[neighbor.index];
                int classIndex = labelMapping.get(label);

                double weight = 1.0;
                switch (distanceWeighting) {
                    case INVERSE:
                        weight = 1.0 / (neighbor.distance + 0.001);
                        break;
                    case SIMILARITY:
                        weight = 1.0 - neighbor.distance;
                        break;
                    case NONE:
                    default:
                        weight = 1.0;
                        break;
                }

                votes[classIndex] += weight;
                totalWeight += weight;
            }

            // 归一化得到概率
            for (int j = 0; j < numClasses; j++) {
                classProbabilities[i][j] = totalWeight > 0 ? votes[j] / totalWeight : 1.0 / numClasses;
            }
        }

        return new BatchPredictionResult(predictions, classProbabilities);
    }
    
    
    
    

    @Override
    public void save(String path) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
            oos.writeObject(this);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // ==================== 辅助方法 ====================
    /**
     * 标签预处理
     */
    private void preprocessLabels(String[] labels) {
        labelMapping.clear();
        reverseLabelMapping.clear();

        int nextLabel = 0;
        for (String label : labels) {
            if (!labelMapping.containsKey(label)) {
                labelMapping.put(label, nextLabel);
                reverseLabelMapping.put(nextLabel, label);
                nextLabel++;
            }
        }

        this.numClasses = labelMapping.size();
        this.trainingLabels = Arrays.copyOf(labels, labels.length);
    }

    /**
     * 应用窗口大小限制
     */
    private void applyWindowSize() {
        int startIndex = Math.max(0, trainingFeatures.getRowNum() - windowSize);

        // 创建新的特征矩阵和标签数组
        double[][] newFeatures = new double[windowSize][featureDimension];
        String[] newLabels = new String[windowSize];

        for (int i = 0; i < windowSize; i++) {
            int srcIndex = startIndex + i;
            for (int j = 0; j < featureDimension; j++) {
                newFeatures[i][j] = (double) trainingFeatures.get(srcIndex, j);
            }
            newLabels[i] = trainingLabels[srcIndex];
        }

        this.trainingFeatures = Linalg.matrix(newFeatures);
        this.trainingLabels = newLabels;
    }

    /**
     * 计算两个向量之间的欧几里得距离
     */
    private double euclideanDistance(IVector a, IVector b) {
        double sum = 0.0;
        for (int i = 0; i < a.length(); i++) {
            double diff = (double) a.get(i) - (double) b.get(i);
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    /**
     * 找到k个最近邻
     */
    private List<Neighbor> findKNearestNeighbors(IVector target, int k) {
        List<Neighbor> neighbors = new ArrayList<>();

        for (int i = 0; i < trainingFeatures.getRowNum(); i++) {
            IVector instance = trainingFeatures.getRow(i);
            double distance = euclideanDistance(target, instance);
            neighbors.add(new Neighbor(i, distance));
        }

        // 按距离排序
        neighbors.sort(Comparator.comparingDouble(n -> n.distance));

        // 返回前k个
        return neighbors.subList(0, Math.min(k, neighbors.size()));
    }

    /**
     * 根据距离加权进行投票
     */
    private String voteByDistance(List<Neighbor> neighbors) {
        double[] votes = new double[numClasses];

        for (Neighbor neighbor : neighbors) {
            String label = trainingLabels[neighbor.index];
            int classIndex = labelMapping.get(label);

            double weight = 1.0;
            switch (distanceWeighting) {
                case INVERSE:
                    weight = 1.0 / (neighbor.distance + 0.001); // 避免除零
                    break;
                case SIMILARITY:
                    weight = 1.0 - neighbor.distance;
                    break;
                case NONE:
                default:
                    weight = 1.0;
                    break;
            }

            votes[classIndex] += weight;
        }

        // 找到得票最多的类别
        int bestClass = 0;
        double maxVotes = votes[0];
        for (int i = 1; i < votes.length; i++) {
            if (votes[i] > maxVotes) {
                maxVotes = votes[i];
                bestClass = i;
            }
        }

        return reverseLabelMapping.get(bestClass);
    }

    /**
     * 交叉验证选择最佳k值
     */
    private void selectBestK() {
        double[] errors = new double[maxK];

        // 对每个k值进行交叉验证
        for (int currentK = 1; currentK <= maxK; currentK++) {
            double totalError = 0.0;

            // 留一法交叉验证
            for (int i = 0; i < trainingFeatures.getRowNum(); i++) {
                // 构造测试实例
                IVector testInstance = trainingFeatures.getRow(i);
                String trueLabel = trainingLabels[i];

                // 从训练集中移除当前实例（模拟）
                List<Integer> trainIndices = new ArrayList<>();
                for (int j = 0; j < trainingFeatures.getRowNum(); j++) {
                    if (j != i) {
                        trainIndices.add(j);
                    }
                }

                // 找到k个最近邻（排除自己）
                List<Neighbor> neighbors = new ArrayList<>();
                for (int idx : trainIndices) {
                    IVector instance = trainingFeatures.getRow(idx);
                    double distance = euclideanDistance(testInstance, instance);
                    neighbors.add(new Neighbor(idx, distance));
                }

                neighbors.sort(Comparator.comparingDouble(n -> n.distance));
                List<Neighbor> kNeighbors = neighbors.subList(0, Math.min(currentK, neighbors.size()));

                // 预测
                String predictedLabel = voteByDistance(kNeighbors);

                // 计算错误
                if (!predictedLabel.equals(trueLabel)) {
                    totalError += 1.0;
                }
            }

            errors[currentK - 1] = totalError / trainingFeatures.getRowNum();
        }

        // 选择错误率最低的k
        double minError = Double.MAX_VALUE;
        int bestK = 1;
        for (int i = 0; i < errors.length; i++) {
            if (errors[i] < minError) {
                minError = errors[i];
                bestK = i + 1;
            }
        }

        this.k = bestK;
        System.out.println("交叉验证选择的最佳k值: " + bestK + ", 错误率: " + String.format("%.4f", minError));
    }

    // ==================== Getter和Setter ====================
    public int getK() {
        return k;
    }

    public void setK(int k) {
        this.k = k;
        this.maxK = k;
    }

    public DistanceWeighting getDistanceWeighting() {
        return distanceWeighting;
    }

    public void setDistanceWeighting(DistanceWeighting distanceWeighting) {
        this.distanceWeighting = distanceWeighting;
    }

    public boolean isCrossValidate() {
        return crossValidate;
    }

    public void setCrossValidate(boolean crossValidate) {
        this.crossValidate = crossValidate;
    }

    public int getWindowSize() {
        return windowSize;
    }

    public void setWindowSize(int windowSize) {
        this.windowSize = windowSize;
    }

    public boolean isTrained() {
        return isTrained;
    }

    // ==================== 内部类 ====================
    /**
     * 近邻类
     */
    private static class Neighbor {
        int index;
        double distance;

        Neighbor(int index, double distance) {
            this.index = index;
            this.distance = distance;
        }
    }
}
