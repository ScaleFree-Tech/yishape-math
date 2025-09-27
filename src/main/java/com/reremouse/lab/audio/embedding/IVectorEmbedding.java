package com.reremouse.lab.audio.embedding;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.stats.model.GaussianMixtureModel;
import com.reremouse.lab.math.stats.model.EMAlgorithm;
import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Map;
import java.util.HashMap;
import java.io.Serializable;

/**
 * 基于i-vector模型将MFCC特征转换为定长的向量表征
 * 
 * i-vector模型是一种用于语音识别和说话人识别的特征提取技术，
 * 它通过通用背景模型(UBM)和全变异性矩阵(T矩阵)将变长的音频特征
 * 转换为固定长度的向量表示。
 * 
 * I-vector model is a feature extraction technique for speech recognition 
 * and speaker recognition. It converts variable-length audio features into 
 * fixed-length vector representations through Universal Background Model (UBM) 
 * and Total Variability Matrix (T-matrix).
 * 
 * @author lteb2
 */
public class IVectorEmbedding implements IAudioEmbedding, Serializable {
    
    private static final long serialVersionUID = 1L;
    
    /** i-vector的维度 / Dimension of i-vector */
    private final int len;
    
    /** UBM的高斯分量数 / Number of Gaussian components in UBM */
    private final int numComponents;
    
    /** MFCC特征维度 / MFCC feature dimension */
    private final int featureDim;
    
    /** 通用背景模型 (UBM) / Universal Background Model */
    private GaussianMixtureModel ubm;
    
    /** 全变异性矩阵 T (mfccDim*numComponents x len) / Total variability matrix T */
    private IMatrix<Double> tMatrix;
    
    /** UBM协方差矩阵的逆 / Inverse of UBM covariance matrices */
    private List<IMatrix<Double>> ubmInvCovariances;
    
    /** 是否已训练 / Whether the model is trained */
    private boolean isTrained = false;
    
    /** 随机数生成器 / Random number generator */
    private final Random random;
    
    /** 训练参数 / Training parameters */
    private final int maxIterations = 100;
    private final double convergenceThreshold = 1e-6;
    private double previousLogLikelihood = Double.NEGATIVE_INFINITY;
    
    /** 模型参数 / Model parameters */
    private Map<String, Object> parameters;
    
    // i-vector相关参数
    private final int supervectorDim;
    private final double relevanceFactor = 16.0;

    /**
     * 构造函数
     * @param len i-vector维度 / i-vector dimension
     */
    public IVectorEmbedding(int len) {
        this(len, 512, 13); // 默认512个高斯分量，13维MFCC / Default 512 Gaussian components, 13-dim MFCC
    }
    
    /**
     * 构造函数
     * @param len i-vector维度 / i-vector dimension
     * @param numComponents UBM高斯分量数 / Number of UBM Gaussian components
     * @param mfccDim MFCC特征维度 / MFCC feature dimension
     */
    public IVectorEmbedding(int len, int numComponents, int mfccDim) {
        this.len = len;
        this.numComponents = numComponents;
        this.featureDim = mfccDim;
        this.supervectorDim = numComponents * mfccDim;
        this.random = new Random(42); // 固定种子以保证可重现性 / Fixed seed for reproducibility
        this.ubmInvCovariances = new ArrayList<>();
        this.parameters = new HashMap<>();
    }

    /**
     * 训练i-vector模型
     * @param trainingData 训练数据列表，每个元素是一个MFCC特征矩阵 / Training data list, each element is an MFCC feature matrix
     */
    public void train(List<IMatrix<Double>> trainingData) {
        if (trainingData == null || trainingData.isEmpty()) {
            throw new IllegalArgumentException("训练数据不能为空 / Training data cannot be empty");
        }
        
        System.out.println("开始训练i-vector模型... / Starting i-vector model training...");
        
        // 第一步：训练UBM / Step 1: Train UBM
        System.out.println("训练UBM... / Training UBM...");
        trainUBM(trainingData);
        
        // 预计算UBM协方差矩阵的逆
        precomputeUBMInvCovariances();
        
        // 第二步：计算T矩阵 / Step 2: Compute T-matrix
        System.out.println("计算T矩阵... / Computing T-matrix...");
        List<IVector<Double>> allFeatures = new ArrayList<>();
        for (IMatrix<Double> mfcc : trainingData) {
            for (int i = 0; i < mfcc.rows(); i++) {
                allFeatures.add(mfcc.getRow(i));
            }
        }
        computeTMatrix(allFeatures);
        
        this.isTrained = true;
        System.out.println("i-vector模型训练完成 / i-vector model training completed");
    }

    /**
     * 训练通用背景模型(UBM)
     * 使用EM算法训练高斯混合模型
     */
    private void trainUBM(List<IMatrix<Double>> trainingData) {
        // 收集所有训练特征 / Collect all training features
        List<IVector<Double>> allFeatures = new ArrayList<>();
        for (IMatrix<Double> mfcc : trainingData) {
            for (int t = 0; t < mfcc.getRowNum(); t++) {
                allFeatures.add(mfcc.getRow(t));
            }
        }
        
        int totalFrames = allFeatures.size();
        System.out.println("总训练帧数: " + totalFrames + " / Total training frames: " + totalFrames);
        
        // 使用新的GaussianMixtureModel训练UBM / Train UBM using new GaussianMixtureModel
        ubm = new GaussianMixtureModel(numComponents, featureDim);
        EMAlgorithm emAlgorithm = new EMAlgorithm(20, 1e-6, true);
        ubm.fit(allFeatures, emAlgorithm); // 使用EM算法训练
        
        System.out.println("UBM训练完成 / UBM training completed");
    }

    /**
     * 预计算UBM协方差矩阵的逆
     */
    private void precomputeUBMInvCovariances() {
        ubmInvCovariances.clear();
        for (int i = 0; i < numComponents; i++) {
            IMatrix<Double> cov = ubm.getComponents().get(i).getCovariance();
            try {
                ubmInvCovariances.add(cov.inv());
            } catch (Exception e) {
                // 如果协方差矩阵不可逆，使用伪逆
                ubmInvCovariances.add(cov.pinv());
            }
        }
    }

    /**
     * 计算T矩阵
     * T矩阵是i-vector模型中的关键参数，用于将超向量映射到i-vector空间
     */
    private void computeTMatrix(List<IVector<Double>> features) {
        System.out.println("计算总变异性矩阵... / Computing total variability matrix...");
        // 计算总变异性矩阵 S
        IMatrix<Double> S = computeTotalVariabilityMatrix(features);
        
        System.out.println("执行PCA降维... / Performing PCA dimensionality reduction...");
        // 使用PCA方法计算T矩阵，避免直接对大矩阵进行特征值分解
        Tuple2<IVector<Double>, IMatrix<Double>> pcaResult = computeTMatrixUsingPCA(S);
        IVector<Double> singularValues = pcaResult._1;
        IMatrix<Double> U = pcaResult._2;
        
        System.out.println("构建T矩阵... / Building T matrix...");
        // 选择前len个主成分
        int numComponents = Math.min(len, singularValues.length());
        
        // 构建T矩阵 (supervectorDim x len)
        double[][] tData = new double[supervectorDim][len];
        for (int i = 0; i < supervectorDim; i++) {
            for (int j = 0; j < len; j++) {
                if (j < numComponents) {
                    // 添加数值稳定性检查
                    double singularValue = Math.max(singularValues.get(j), 0.0);
                    tData[i][j] = U.get(i, j) * Math.sqrt(singularValue);
                } else {
                    tData[i][j] = 0.0;
                }
            }
        }
        
        this.tMatrix = Linalg.matrix(tData);
        System.out.println("T矩阵计算完成 / T-matrix computation completed");
    }

    /**
     * 使用PCA方法计算T矩阵
     * 通过随机投影方法避免直接对大矩阵进行特征值分解
     */
    private Tuple2<IVector<Double>, IMatrix<Double>> computeTMatrixUsingPCA(IMatrix<Double> S) {
        int targetDimensions = Math.min(len * 2, supervectorDim); // 目标维度
        System.out.println("目标维度: " + targetDimensions + " / Target dimensions: " + targetDimensions);
        
        // 生成随机投影矩阵
        IMatrix<Double> randomProjection = Linalg.rand(supervectorDim, targetDimensions, 42L);
        
        // 计算 Y = S * R
        IMatrix<Double> Y = S.mmul(randomProjection);
        
        // QR分解 Y = Q * R
        Tuple2<IMatrix<Double>, IMatrix<Double>> qr = Y.qr();
        IMatrix<Double> Q = qr._1;
        
        // 计算 B = Q^T * S * Q
        IMatrix<Double> B = Q.transpose().mmul(S).mmul(Q);
        
        // 对B进行特征值分解
        Tuple2<IVector<Double>, IMatrix<Double>> eigen = B.eigen();
        IVector<Double> eigenvalues = eigen._1;
        IMatrix<Double> eigenvectors = eigen._2;
        
        // 计算最终的特征向量 U = Q * V
        IMatrix<Double> U = Q.mmul(eigenvectors);
        
        return new Tuple2<>(eigenvalues, U);
    }

    /**
     * 计算Baum-Welch统计量
     * 这些统计量用于UBM训练和i-vector提取
     */
    private Tuple2<IVector<Double>, IVector<Double>> computeBaumWelchStats(IVector<Double> features) {
        // 将特征向量重塑为帧序列
        int numFrames = features.length() / featureDim;
        
        // 初始化统计量
        IVector<Double> N = Linalg.zeros(numComponents);  // 零阶统计量
        IVector<Double> F = Linalg.zeros(numComponents * featureDim);  // 一阶统计量
        
        // 对每一帧计算后验概率
        for (int t = 0; t < numFrames; t++) {
            // 提取当前帧
            IVector<Double> frame = features.slice(t * featureDim, (t + 1) * featureDim);
            
            // 使用GaussianMixtureModel计算后验概率
            IVector<Double> posteriors = ubm.computePosteriors(frame);
            
            // 累积统计量
            for (int c = 0; c < numComponents; c++) {
                double posterior = posteriors.get(c);
                N.set(c, N.get(c) + posterior);
                
                for (int d = 0; d < featureDim; d++) {
                    int fidx = c * featureDim + d;
                    F.set(fidx, F.get(fidx) + posterior * frame.get(d));
                }
            }
        }
        
        return new Tuple2<>(N, F);
    }

    /**
     * 计算总变异性矩阵
     */
    private IMatrix<Double> computeTotalVariabilityMatrix(List<IVector<Double>> features) {
        IMatrix<Double> S = Linalg.zeros(supervectorDim, supervectorDim);
        
        int totalFeatures = features.size();
        System.out.println("处理 " + totalFeatures + " 个特征向量 / Processing " + totalFeatures + " feature vectors");
        
        // 为了提高效率，我们只使用一部分特征来计算总变异性矩阵
        // 这在实际应用中是常见的做法
        int sampleSize = Math.min(features.size(), 100); // 进一步减少样本数量
        int step = Math.max(1, features.size() / sampleSize);
        
        int processed = 0;
        for (int i = 0; i < features.size(); i += step) {
            IVector<Double> feature = features.get(i);
            
            // 计算统计量
            Tuple2<IVector<Double>, IVector<Double>> stats = computeBaumWelchStats(feature);
            IVector<Double> N = stats._1;  // 零阶统计量
            IVector<Double> F = stats._2;  // 一阶统计量
            
            // 计算超向量差异
            IVector<Double> diff = computeSupervectorDiff(N, F);
            
            // 计算外积并累加
            IMatrix<Double> outer = diff.outer(diff);
            S = S.add(outer);
            
            processed++;
            // 显示进度
            if (processed % 20 == 0 || processed == sampleSize) {
                System.out.println("进度: " + processed + "/" + sampleSize);
            }
        }
        
        // 归一化
        if (processed > 0) {
            S = S.divideByScalar((double) processed);
        }
        
        // 添加小的正则化项以提高数值稳定性
        double regularization = 1e-8; // 稍微增加正则化
        for (int i = 0; i < supervectorDim; i++) {
            S.set(i, i, S.get(i, i) + regularization);
        }
        
        return S;
    }

    /**
     * 从Baum-Welch统计量计算超向量差异
     */
    private IVector<Double> computeSupervectorDiff(IVector<Double> N, IVector<Double> F) {
        // 计算自适应均值与UBM均值的差异
        double[] diffData = new double[supervectorDim];
        int idx = 0;
        
        for (int i = 0; i < numComponents; i++) {
            IVector<Double> ubmMean = ubm.getComponents().get(i).getMean();
            
            // 计算自适应因子
            double alpha = N.get(i) / (N.get(i) + relevanceFactor);
            
            // 计算自适应均值
            IVector<Double> adaptedMean;
            if (N.get(i) > 1e-10) {
                IVector<Double> empiricalMean = F.slice(i * featureDim, (i + 1) * featureDim).divideByScalar(N.get(i));
                adaptedMean = ubmMean.multiplyScalar(1.0 - alpha).add(empiricalMean.multiplyScalar(alpha));
            } else {
                adaptedMean = ubmMean.copy();
            }
            
            // 计算差异向量 (适应均值 - UBM均值)
            IVector<Double> diff = adaptedMean.sub(ubmMean);
            double[] diffArray = diff.toDoubleArray();
            System.arraycopy(diffArray, 0, diffData, idx, diffArray.length);
            idx += diffArray.length;
        }
        
        return Linalg.vector(diffData);
    }

    @Override
    public IVector<Double> embed(IMatrix<Double> mfcc) {
        if (!isTrained) {
            throw new IllegalStateException("模型尚未训练，请先调用train()方法 / Model not trained, please call train() method first");
        }
        
        if (mfcc == null || mfcc.getRowNum() == 0) {
            throw new IllegalArgumentException("MFCC特征不能为空 / MFCC features cannot be empty");
        }
        
        if (mfcc.getColNum() != featureDim) {
            throw new IllegalArgumentException("MFCC特征维度不匹配，期望: " + featureDim + ", 实际: " + mfcc.getColNum() + 
                " / MFCC feature dimension mismatch, expected: " + featureDim + ", actual: " + mfcc.getColNum());
        }
        
        // 计算Baum-Welch统计量（逐帧处理）
        IVector<Double> N = Linalg.zeros(numComponents);  // 零阶统计量
        IVector<Double> F = Linalg.zeros(numComponents * featureDim);  // 一阶统计量
        
        // 对每一帧计算后验概率并累积统计量
        for (int t = 0; t < mfcc.getRowNum(); t++) {
            IVector<Double> frame = mfcc.getRow(t);
            
            // 使用GaussianMixtureModel计算后验概率
            IVector<Double> posteriors = ubm.computePosteriors(frame);
            
            // 累积统计量
            for (int c = 0; c < numComponents; c++) {
                double posterior = posteriors.get(c);
                N.set(c, N.get(c) + posterior);
                
                for (int d = 0; d < featureDim; d++) {
                    int fidx = c * featureDim + d;
                    F.set(fidx, F.get(fidx) + posterior * frame.get(d));
                }
            }
        }
        
        // 计算超向量差异 / Compute supervector difference
        IVector<Double> supervectorDiff = computeSupervectorDiff(N, F);
        
        // 计算i-vector: w = (I + T^T Σ^{-1} N T)^{-1} T^T Σ^{-1} m
        // 简化版本：w = T^T (Σ^{-1} m)
        // 进一步简化：w = T^T m (当前实现)
        IVector<Double> ivector = tMatrix.transpose().mmul(supervectorDiff);
        
        // 长度归一化 / Length normalization
        double norm = ivector.norm2();
        if (norm > 1e-10) {
            ivector = ivector.divideByScalar(norm);
        } else {
            ivector = Linalg.zeros(len); // Use Linalg factory method
        }
        
        return ivector;
    }
    
    @Override
    public IVector<Double> embed(AudioData audioData) {
        try { 
            var mat = AudioUtil.calculateMFCCMatrix(audioData);
            return this.embed(mat);
        } catch (Exception e) { 
            e.printStackTrace();
        }
        return null;
    }
    
    @Override
    public IVector<Double> embed(IVector<Double> samples, int sampleRate) {
        // This is a simplified implementation
        // In a real implementation, you would extract features from the audio samples
        throw new UnsupportedOperationException("Method not yet implemented");
    }
    
    @Override
    public IMatrix<Double> embedBatch(IMatrix<Double>[] mfccBatch) {
        if (mfccBatch == null || mfccBatch.length == 0) {
            throw new IllegalArgumentException("MFCC批次不能为空 / MFCC batch cannot be empty");
        }
        
        // 创建结果矩阵
        double[][] result = new double[mfccBatch.length][];
        
        // 对每个MFCC矩阵进行处理
        for (int i = 0; i < mfccBatch.length; i++) {
            IVector<Double> embedding = embed(mfccBatch[i]);
            result[i] = embedding.toDoubleArray();
        }
        
        return Linalg.matrix(result);
    }
    
    @Override
    public int getEmbeddingDimension() {
        return len;
    }
    
    @Override
    public FeatureType[] getSupportedFeatureTypes() {
        return new FeatureType[] { FeatureType.MFCC };
    }
    
    @Override
    public boolean supportsFeatureType(FeatureType featureType) {
        return featureType == FeatureType.MFCC;
    }
    
    @Override
    public void setParameters(Map<String, Object> parameters) {
        this.parameters = parameters;
    }
    
    @Override
    public Map<String, Object> getParameters() {
        return parameters;
    }
    
    @Override
    public double calculateSimilarity(IVector<Double> embedding1, IVector<Double> embedding2) {
        if (embedding1 == null || embedding2 == null) {
            throw new IllegalArgumentException("嵌入向量不能为空 / Embedding vectors cannot be null");
        }
        
        if (embedding1.length() != embedding2.length()) {
            throw new IllegalArgumentException("嵌入向量维度必须相同 / Embedding vector dimensions must be the same");
        }
        
        // 使用余弦相似度
        double dotProduct = embedding1.dot(embedding2);
        double norm1 = embedding1.norm2();
        double norm2 = embedding2.norm2();
        
        if (norm1 < 1e-10 || norm2 < 1e-10) {
            return 0.0;
        }
        
        return dotProduct / (norm1 * norm2);
    }
    
    @Override
    public double calculateDistance(IVector<Double> embedding1, IVector<Double> embedding2, DistanceType distanceType) {
        if (embedding1 == null || embedding2 == null) {
            throw new IllegalArgumentException("嵌入向量不能为空 / Embedding vectors cannot be null");
        }
        
        if (embedding1.length() != embedding2.length()) {
            throw new IllegalArgumentException("嵌入向量维度必须相同 / Embedding vector dimensions must be the same");
        }
        
        switch (distanceType) {
            case EUCLIDEAN:
                return embedding1.sub(embedding2).norm2();
            case COSINE:
                return 1.0 - calculateSimilarity(embedding1, embedding2);
            case MANHATTAN:
                return embedding1.sub(embedding2).norm1();
            default:
                throw new UnsupportedOperationException("不支持的距离类型: " + distanceType + " / Unsupported distance type: " + distanceType);
        }
    }
    
    @Override
    public IVector<Double> normalize(IVector<Double> embedding) {
        if (embedding == null) {
            throw new IllegalArgumentException("嵌入向量不能为空 / Embedding vector cannot be null");
        }
        
        double norm = embedding.norm2();
        if (norm < 1e-10) {
            return Linalg.zeros(embedding.length()); // Return zero vector using Linalg factory method
        }
        
        return embedding.divideByScalar(norm);
    }

    // 辅助方法 / Helper methods

    // Getter and setter methods for serialization
    public boolean isTrained() {
        return isTrained;
    }
    
    public void setTrained(boolean trained) {
        isTrained = trained;
    }
    
    public GaussianMixtureModel getUbm() {
        return ubm;
    }
    
    public void setUbm(GaussianMixtureModel ubm) {
        this.ubm = ubm;
    }
    
    public IMatrix<Double> getTMatrix() {
        return tMatrix;
    }
    
    public void setTMatrix(IMatrix<Double> tMatrix) {
        this.tMatrix = tMatrix;
    }
    
    public List<IMatrix<Double>> getUbmInvCovariances() {
        return ubmInvCovariances;
    }
    
    public void setUbmInvCovariances(List<IMatrix<Double>> ubmInvCovariances) {
        this.ubmInvCovariances = ubmInvCovariances;
    }
    
    public int getLen() {
        return len;
    }
    
    public int getNumComponents() {
        return numComponents;
    }
    
    public int getMfccDim() {
        return featureDim;
    }
    
    public int getSupervectorDim() {
        return supervectorDim;
    }
    
    public double getRelevanceFactor() {
        return relevanceFactor;
    }
    
    @Override
    public IAudioEmbedding save(String path) {
        try {
            // Create a map to store all model parameters
            java.util.Map<String, Object> modelData = new java.util.HashMap<>();
            
            // Store basic parameters
            modelData.put("len", this.len);
            modelData.put("numComponents", this.numComponents);
            modelData.put("featureDim", this.featureDim);
            modelData.put("supervectorDim", this.supervectorDim);
            modelData.put("isTrained", this.isTrained);
            modelData.put("relevanceFactor", this.relevanceFactor);
            modelData.put("modelType", "IVectorEmbedding"); // Add model type identifier
            
            // Store UBM if trained
            if (this.ubm != null && this.isTrained) {
                modelData.put("ubm", this.ubm);
            }
            
            // Store T-matrix if trained
            if (this.tMatrix != null && this.isTrained) {
                modelData.put("tMatrix", this.tMatrix);
            }
            
            // Store UBM inverse covariances if available
            if (this.ubmInvCovariances != null && !this.ubmInvCovariances.isEmpty()) {
                modelData.put("ubmInvCovariances", this.ubmInvCovariances);
            }
            
            // Serialize to file using Java serialization
            java.io.FileOutputStream fileOut = new java.io.FileOutputStream(path);
            java.io.ObjectOutputStream out = new java.io.ObjectOutputStream(fileOut);
            out.writeObject(modelData);
            out.close();
            fileOut.close();
            
            return this;
        } catch (Exception e) {
            throw new RuntimeException("Failed to save IVectorEmbedding model to " + path, e);
        }
    }
}
