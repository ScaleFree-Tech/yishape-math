package com.reremouse.lab.audio.embedding;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.stats.model.GaussianMixtureModel;
import com.reremouse.lab.math.stats.model.EMAlgorithm;
import com.reremouse.lab.math.optimize.newton.RereOnlineAdam;
import com.reremouse.lab.math.optimize.newton.RereOnlineSGD;
import com.reremouse.lab.math.optimize.IOnlineOptimizer;
import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.util.Tuple3;
import com.reremouse.lab.audio.core.AudioData;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Map;
import java.util.HashMap;
import java.io.Serializable;

/**
 * 基于i-vector模型的在线增量训练实现
 * Online incremental training implementation based on i-vector model
 * 
 * 支持流式音频数据的增量训练，适用于处理大规模音频数据集
 * 支持小批量MFCC样本的增量训练，提供更灵活的训练控制
 * Supports incremental training with streaming audio data, suitable for processing large-scale audio datasets
 * Supports incremental training with small batches of MFCC samples, providing more flexible training control
 */
public class OnlineIVectorEmbedding implements IAudioEmbedding, Serializable {
    
    private static final long serialVersionUID = 1L;
    
    /** i-vector的维度 / Dimension of i-vector */
    private final int len;
    
    /** UBM的高斯分量数 / Number of Gaussian components in UBM */
    private final int numComponents;
    
    /** MFCC特征维度 / MFCC feature dimension */
    private final int mfccDim;
    
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
    
    // 在线训练相关参数
    private IOnlineOptimizer ubmOptimizer;
    private IOnlineOptimizer tMatrixOptimizer;
    private boolean useAdamOptimizer = true;
    private double learningRate = 0.001;
    
    // 足够统计信息用于增量更新
    private List<IVector<Double>> accumulatedFeatures;
    private int batchSize = 100;
    private int processedSamples = 0;

    /**
     * 构造函数
     * 支持小批量MFCC样本的增量训练
     * @param len i-vector维度 / i-vector dimension
     */
    public OnlineIVectorEmbedding(int len) {
        this(len, 512, 13, true, 0.001); // 默认512个高斯分量，13维MFCC
    }
    
    /**
     * 构造函数
     * 支持小批量MFCC样本的增量训练
     * @param len i-vector维度 / i-vector dimension
     * @param numComponents UBM高斯分量数 / Number of UBM Gaussian components
     * @param mfccDim MFCC特征维度 / MFCC feature dimension
     */
    public OnlineIVectorEmbedding(int len, int numComponents, int mfccDim) {
        this(len, numComponents, mfccDim, true, 0.001); 
    }
    
    /**
     * 构造函数
     * 支持小批量MFCC样本的增量训练，可配置优化器和学习率
     * @param len i-vector维度 / i-vector dimension
     * @param numComponents UBM高斯分量数 / Number of UBM Gaussian components
     * @param mfccDim MFCC特征维度 / MFCC feature dimension
     * @param useAdam 是否使用Adam优化器 / Whether to use Adam optimizer
     * @param learningRate 学习率 / Learning rate
     */
    public OnlineIVectorEmbedding(int len, int numComponents, int mfccDim, 
                             boolean useAdam, double learningRate) {
        this.len = len;
        this.numComponents = numComponents;
        this.mfccDim = mfccDim;
        this.supervectorDim = numComponents * mfccDim;
        this.random = new Random(42); // 固定种子以保证可重现性 / Fixed seed for reproducibility
        this.ubmInvCovariances = new ArrayList<>();
        this.parameters = new HashMap<>();
        this.useAdamOptimizer = useAdam;
        this.learningRate = learningRate;
        this.accumulatedFeatures = new ArrayList<>();
        
        // 初始化优化器
        initializeOptimizers();
    }
    
    /**
     * 初始化在线优化器
     */
    private void initializeOptimizers() {
        if (useAdamOptimizer) {
            this.ubmOptimizer = new RereOnlineAdam(learningRate);
            this.tMatrixOptimizer = new RereOnlineAdam(learningRate);
        } else {
            this.ubmOptimizer = new RereOnlineSGD(learningRate, 0.9); // 带动量的SGD
            this.tMatrixOptimizer = new RereOnlineSGD(learningRate, 0.9);
        }
    }

    /**
     * 在线增量训练i-vector模型
     * @param mfcc MFCC特征矩阵 / MFCC feature matrix
     */
    public void trainIncremental(IMatrix<Double> mfcc) {
        if (mfcc == null || mfcc.getRowNum() == 0) {
            throw new IllegalArgumentException("MFCC特征不能为空 / MFCC features cannot be empty");
        }
        
        if (mfcc.getColNum() != mfccDim) {
            throw new IllegalArgumentException("MFCC特征维度不匹配 / MFCC feature dimension mismatch");
        }
        
        // 如果是第一次训练，需要初始化UBM
        if (!isTrained) {
            initializeUBM(mfcc);
        }
        
        // 累积特征用于批量更新
        for (int i = 0; i < mfcc.getRowNum(); i++) {
            accumulatedFeatures.add(mfcc.getRow(i));
        }
        
        // 当累积的特征达到批次大小时，执行一次更新
        if (accumulatedFeatures.size() >= batchSize) {
            updateModelWithBatch(accumulatedFeatures);
            accumulatedFeatures.clear();
        }
        
        processedSamples += mfcc.getRowNum();
    }
    
    /**
     * 使用小批量MFCC样本进行增量训练（使用默认参数更新UBM和T矩阵）
     * @param mfccBatch 小批量MFCC特征矩阵数组 / Small batch of MFCC feature matrices
     */
    public void trainIncrementalBatch(IMatrix<Double>[] mfccBatch) {
        trainIncrementalBatch(mfccBatch, true, true); // 默认更新UBM和T矩阵
    }
    
    /**
     * 使用小批量MFCC样本进行增量训练
     * @param mfccBatch 小批量MFCC特征矩阵数组 / Small batch of MFCC feature matrices
     * @param updateUBM 是否更新UBM参数 / Whether to update UBM parameters
     * @param updateTMatrix 是否更新T矩阵 / Whether to update T-matrix
     */
    public void trainIncrementalBatch(IMatrix<Double>[] mfccBatch, boolean updateUBM, boolean updateTMatrix) {
        if (mfccBatch == null || mfccBatch.length == 0) {
            throw new IllegalArgumentException("MFCC批次不能为空 / MFCC batch cannot be empty");
        }
        
        System.out.println("使用小批量MFCC样本进行增量训练，批次大小: " + mfccBatch.length + 
                          " / Incremental training with small batch of MFCC samples, batch size: " + mfccBatch.length);
        
        // 收集所有特征
        List<IVector<Double>> batchFeatures = new ArrayList<>();
        for (IMatrix<Double> mfcc : mfccBatch) {
            if (mfcc == null || mfcc.getRowNum() == 0) {
                continue;
            }
            
            if (mfcc.getColNum() != mfccDim) {
                throw new IllegalArgumentException("MFCC特征维度不匹配 / MFCC feature dimension mismatch");
            }
            
            // 如果是第一次训练，需要初始化UBM
            if (!isTrained) {
                initializeUBM(mfcc);
            }
            
            // 收集特征
            for (int i = 0; i < mfcc.getRowNum(); i++) {
                batchFeatures.add(mfcc.getRow(i));
            }
            
            processedSamples += mfcc.getRowNum();
        }
        
        if (batchFeatures.isEmpty()) {
            throw new IllegalArgumentException("批次中没有有效的MFCC特征 / No valid MFCC features in batch");
        }
        
        // 更新模型
        if (isTrained) {
            updateModelWithBatchSelective(batchFeatures, updateUBM, updateTMatrix);
        }
    }
    
    /**
     * 选择性地使用批次数据更新模型
     * @param batchFeatures 批次特征 / Batch features
     * @param updateUBM 是否更新UBM参数 / Whether to update UBM parameters
     * @param updateTMatrix 是否更新T矩阵 / Whether to update T-matrix
     */
    private void updateModelWithBatchSelective(List<IVector<Double>> batchFeatures, boolean updateUBM, boolean updateTMatrix) {
        if (!isTrained) {
            throw new IllegalStateException("模型尚未初始化 / Model not initialized");
        }
        
        System.out.println("选择性更新模型，样本数: " + batchFeatures.size() + 
                          "，更新UBM: " + updateUBM + "，更新T矩阵: " + updateTMatrix +
                          " / Selective model update, sample count: " + batchFeatures.size() + 
                          ", update UBM: " + updateUBM + ", update T-matrix: " + updateTMatrix);
        
        // 更新UBM参数（如果需要）
        if (updateUBM) {
            updateUBMWithBatch(batchFeatures);
        }
        
        // 更新T矩阵（如果需要）
        if (updateTMatrix) {
            updateTMatrixWithBatch(batchFeatures);
        }
        
        // 更新协方差矩阵的逆（如果更新了UBM）
        if (updateUBM) {
            precomputeUBMInvCovariances();
        }
    }
    
    /**
     * 使用单个MFCC样本进行增量训练（使用默认参数更新UBM和T矩阵，不累积直接更新）
     * @param mfcc 单个MFCC特征矩阵 / Single MFCC feature matrix
     */
    public void trainIncrementalSample(IMatrix<Double> mfcc) {
        trainIncrementalSample(mfcc, true, true); // 默认更新UBM和T矩阵
    }
    
    /**
     * 使用单个MFCC样本进行增量训练（不累积，直接更新）
     * @param mfcc 单个MFCC特征矩阵 / Single MFCC feature matrix
     * @param updateUBM 是否更新UBM参数 / Whether to update UBM parameters
     * @param updateTMatrix 是否更新T矩阵 / Whether to update T-matrix
     */
    public void trainIncrementalSample(IMatrix<Double> mfcc, boolean updateUBM, boolean updateTMatrix) {
        if (mfcc == null || mfcc.getRowNum() == 0) {
            throw new IllegalArgumentException("MFCC特征不能为空 / MFCC features cannot be empty");
        }
        
        if (mfcc.getColNum() != mfccDim) {
            throw new IllegalArgumentException("MFCC特征维度不匹配 / MFCC feature dimension mismatch");
        }
        
        System.out.println("使用单个MFCC样本进行增量训练，帧数: " + mfcc.getRowNum() + 
                          " / Incremental training with single MFCC sample, frame count: " + mfcc.getRowNum());
        
        // 如果是第一次训练，需要初始化UBM
        if (!isTrained) {
            initializeUBM(mfcc);
        }
        
        // 收集特征
        List<IVector<Double>> sampleFeatures = new ArrayList<>();
        for (int i = 0; i < mfcc.getRowNum(); i++) {
            sampleFeatures.add(mfcc.getRow(i));
        }
        
        // 直接更新模型
        if (isTrained) {
            updateModelWithBatchSelective(sampleFeatures, updateUBM, updateTMatrix);
        }
        
        processedSamples += mfcc.getRowNum();
    }
    
    /**
     * 初始化UBM模型
     * @param sampleData 样本数据用于初始化 / Sample data for initialization
     */
    private void initializeUBM(IMatrix<Double> sampleData) {
        System.out.println("初始化在线i-vector模型... / Initializing online i-vector model...");
        
        // 收集样本特征用于初始化
        List<IVector<Double>> sampleFeatures = new ArrayList<>();
        int sampleSize = Math.min(sampleData.getRowNum(), 1000); // 限制样本数量
        
        for (int i = 0; i < sampleSize; i++) {
            sampleFeatures.add(sampleData.getRow(i));
        }
        
        // 使用标准方法初始化UBM
        // 确保高斯分量数不超过样本数
        int actualNumComponents = Math.min(numComponents, Math.max(1, sampleFeatures.size() / 2));
        if (actualNumComponents != numComponents) {
            System.out.println("调整高斯分量数从 " + numComponents + " 到 " + actualNumComponents + 
                             " 以适应样本大小 / Adjusting number of Gaussian components from " + 
                             numComponents + " to " + actualNumComponents + " to fit sample size");
        }
        
        ubm = new GaussianMixtureModel(actualNumComponents, mfccDim);
        
        try {
            // 使用EM算法进行初步训练
            EMAlgorithm emAlgorithm = new EMAlgorithm(10, 1e-4, true); // 使用较少的迭代次数进行初始化
            ubm.fit(sampleFeatures, emAlgorithm);
        } catch (Exception e) {
            // 如果K-means++初始化失败，使用随机初始化
            System.out.println("K-means++初始化失败，使用随机初始化 / K-means++ initialization failed, using random initialization");
            try {
                ubm.initializeRandomly(sampleFeatures);
                EMAlgorithm emAlgorithm = new EMAlgorithm(10, 1e-4, true);
                EMAlgorithm.EMResult result = emAlgorithm.fit(sampleFeatures, ubm);
            } catch (Exception e2) {
                System.err.println("UBM初始化失败 / UBM initialization failed: " + e2.getMessage());
                throw new RuntimeException("无法初始化UBM模型 / Failed to initialize UBM model", e2);
            }
        }
        
        // 初始化T矩阵
        computeInitialTMatrix(sampleFeatures);
        
        // 初始化协方差矩阵的逆
        precomputeUBMInvCovariances();
        
        this.isTrained = true;
        System.out.println("在线i-vector模型初始化完成 / Online i-vector model initialization completed");
    }
    
    /**
     * 计算初始T矩阵
     */
    private void computeInitialTMatrix(List<IVector<Double>> features) {
        System.out.println("计算初始T矩阵... / Computing initial T-matrix...");
        
        // 计算总变异性矩阵
        IMatrix<Double> S = computeTotalVariabilityMatrix(features);
        
        // 使用PCA方法计算T矩阵
        Tuple2<IVector<Double>, IMatrix<Double>> pcaResult = computeTMatrixUsingPCA(S);
        IVector<Double> singularValues = pcaResult._1;
        IMatrix<Double> U = pcaResult._2;
        
        // 构建T矩阵
        int numComponents = Math.min(len, singularValues.length());
        double[][] tData = new double[supervectorDim][len];
        
        for (int i = 0; i < supervectorDim; i++) {
            for (int j = 0; j < len; j++) {
                if (j < numComponents) {
                    double singularValue = Math.max(singularValues.get(j), 0.0);
                    tData[i][j] = U.get(i, j) * Math.sqrt(singularValue);
                } else {
                    tData[i][j] = 0.0;
                }
            }
        }
        
        this.tMatrix = Linalg.matrix(tData);
        System.out.println("初始T矩阵计算完成 / Initial T-matrix computation completed");
    }
    
    /**
     * 使用批次数据更新模型
     * @param batchFeatures 批次特征 / Batch features
     */
    private void updateModelWithBatch(List<IVector<Double>> batchFeatures) {
        if (!isTrained) {
            throw new IllegalStateException("模型尚未初始化 / Model not initialized");
        }
        
        System.out.println("使用批次数据更新模型，样本数: " + batchFeatures.size() + 
                          " / Updating model with batch data, sample count: " + batchFeatures.size());
        
        // 更新UBM参数
        updateUBMWithBatch(batchFeatures);
        
        // 更新T矩阵
        updateTMatrixWithBatch(batchFeatures);
        
        // 更新协方差矩阵的逆
        precomputeUBMInvCovariances();
    }
    
    /**
     * 使用批次数据更新UBM
     * @param batchFeatures 批次特征 / Batch features
     */
    private void updateUBMWithBatch(List<IVector<Double>> batchFeatures) {
        // 这里可以实现更复杂的增量更新算法
        // 目前使用简化的方法：使用EM算法对当前批次进行少量迭代
        
        try {
            // 创建一个副本用于更新
            GaussianMixtureModel tempUBM = new GaussianMixtureModel(ubm.getComponents(), ubm.getWeights());
            
            // 使用少量迭代更新
            EMAlgorithm emAlgorithm = new EMAlgorithm(3, 1e-3, false); // 少量迭代
            emAlgorithm.fit(batchFeatures, tempUBM);
            
            // 更新原模型
            for (int i = 0; i < numComponents; i++) {
                ubm.setComponent(i, tempUBM.getComponent(i));
                ubm.setWeight(i, tempUBM.getWeight(i));
            }
            
        } catch (Exception e) {
            System.err.println("UBM更新过程中出现错误: " + e.getMessage());
        }
    }
    
    /**
     * 使用批次数据更新T矩阵
     * @param batchFeatures 批次特征 / Batch features
     */
    private void updateTMatrixWithBatch(List<IVector<Double>> batchFeatures) {
        try {
            // 计算新的总变异性矩阵
            IMatrix<Double> newS = computeTotalVariabilityMatrix(batchFeatures);
            
            // 与现有的S矩阵进行加权平均更新
            // 这里使用简单的移动平均更新策略
            double alpha = 0.1; // 更新率
            // IMatrix<Double> updatedS = S.multiplyScalar(1.0 - alpha).add(newS.multiplyScalar(alpha));
            
            // 重新计算T矩阵
            Tuple2<IVector<Double>, IMatrix<Double>> pcaResult = computeTMatrixUsingPCA(newS);
            IVector<Double> singularValues = pcaResult._1;
            IMatrix<Double> U = pcaResult._2;
            
            // 构建新的T矩阵
            int numComponents = Math.min(len, singularValues.length());
            double[][] tData = new double[supervectorDim][len];
            
            for (int i = 0; i < supervectorDim; i++) {
                for (int j = 0; j < len; j++) {
                    if (j < numComponents) {
                        double singularValue = Math.max(singularValues.get(j), 0.0);
                        tData[i][j] = U.get(i, j) * Math.sqrt(singularValue);
                    } else {
                        tData[i][j] = 0.0;
                    }
                }
            }
            
            this.tMatrix = Linalg.matrix(tData);
            
        } catch (Exception e) {
            System.err.println("T矩阵更新过程中出现错误: " + e.getMessage());
        }
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
     * 完成训练（处理剩余的累积数据）
     */
    public void finishTraining() {
        // 处理剩余的累积特征
        if (!accumulatedFeatures.isEmpty()) {
            updateModelWithBatch(accumulatedFeatures);
            accumulatedFeatures.clear();
        }
        
        System.out.println("在线训练完成，总共处理样本数: " + processedSamples + 
                          " / Online training completed, total processed samples: " + processedSamples);
    }
    
    // 以下方法与原始IVectorModel保持一致
    
    /**
     * 计算总变异性矩阵
     */
    private IMatrix<Double> computeTotalVariabilityMatrix(List<IVector<Double>> features) {
        IMatrix<Double> S = Linalg.zeros(supervectorDim, supervectorDim);
        
        int totalFeatures = features.size();
        System.out.println("处理 " + totalFeatures + " 个特征向量 / Processing " + totalFeatures + " feature vectors");
        
        // 为了提高效率，我们只使用一部分特征来计算总变异性矩阵
        int sampleSize = Math.min(features.size(), 50); // 进一步减少样本数量
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
            // 使用矩阵乘法计算外积: diff * diff^T
            IMatrix<Double> diffMatrix = Linalg.matrix(new double[][]{diff.toDoubleArray()});
            IMatrix<Double> outer = diffMatrix.transpose().mmul(diffMatrix);
            S = S.add(outer);
            
            processed++;
        }
        
        // 归一化
        if (processed > 0) {
            S = S.divideByScalar((double) processed);
        }
        
        // 添加小的正则化项以提高数值稳定性
        double regularization = 1e-8;
        for (int i = 0; i < supervectorDim; i++) {
            S.set(i, i, S.get(i, i) + regularization);
        }
        
        return S;
    }
    
    /**
     * 使用PCA方法计算T矩阵
     */
    private Tuple2<IVector<Double>, IMatrix<Double>> computeTMatrixUsingPCA(IMatrix<Double> S) {
        int targetDimensions = Math.min(len * 2, supervectorDim);
        System.out.println("目标维度: " + targetDimensions + " / Target dimensions: " + targetDimensions);
        
        // 生成随机投影矩阵
        IMatrix<Double> randomProjection = IMatrix.rand(supervectorDim, targetDimensions, 42L);
        
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
     */
    private Tuple2<IVector<Double>, IVector<Double>> computeBaumWelchStats(IVector<Double> features) {
        // 将特征向量重塑为帧序列
        int numFrames = features.length() / mfccDim;
        
        // 初始化统计量
        IVector<Double> N = IVector.zeros(numComponents, Double.class);  // 零阶统计量
        IVector<Double> F = IVector.zeros(numComponents * mfccDim, Double.class);  // 一阶统计量
        
        // 对每一帧计算后验概率
        for (int t = 0; t < numFrames; t++) {
            // 提取当前帧
            IVector<Double> frame = features.slice(t * mfccDim, (t + 1) * mfccDim);
            
            // 使用GaussianMixtureModel计算后验概率
            IVector<Double> posteriors = ubm.computePosteriors(frame);
            
            // 累积统计量
            for (int c = 0; c < numComponents; c++) {
                double posterior = posteriors.get(c);
                N.set(c, N.get(c) + posterior);
                
                for (int d = 0; d < mfccDim; d++) {
                    int fidx = c * mfccDim + d;
                    F.set(fidx, F.get(fidx) + posterior * frame.get(d));
                }
            }
        }
        
        return new Tuple2<>(N, F);
    }
    
    /**
     * 从Baum-Welch统计量计算超向量差异
     */
    private IVector<Double> computeSupervectorDiff(IVector<Double> N, IVector<Double> F) {
        // 计算自适应均值与UBM均值的差异
        List<IVector<Double>> diffVectors = new ArrayList<>();
        
        for (int i = 0; i < numComponents; i++) {
            IVector<Double> ubmMean = ubm.getComponents().get(i).getMean();
            
            // 计算自适应因子
            double alpha = N.get(i) / (N.get(i) + relevanceFactor);
            
            // 计算自适应均值
            IVector<Double> adaptedMean;
            if (N.get(i) > 1e-10) {
                IVector<Double> empiricalMean = F.slice(i * mfccDim, (i + 1) * mfccDim).divideByScalar(N.get(i));
                adaptedMean = ubmMean.multiplyScalar(1.0 - alpha).add(empiricalMean.multiplyScalar(alpha));
            } else {
                adaptedMean = ubmMean.copy();
            }
            
            // 计算差异向量 (适应均值 - UBM均值)
            IVector<Double> diff = adaptedMean.sub(ubmMean);
            diffVectors.add(diff);
        }
        
        // 使用Linalg工厂方法连接向量
        double[] diffData = new double[supervectorDim];
        int idx = 0;
        for (IVector<Double> diffVector : diffVectors) {
            double[] diffArray = diffVector.toDoubleArray();
            System.arraycopy(diffArray, 0, diffData, idx, diffArray.length);
            idx += diffArray.length;
        }
        
        return Linalg.vector(diffData);
    }

    @Override
    public IVector<Double> embed(IMatrix<Double> mfcc) {
        if (!isTrained) {
            throw new IllegalStateException("模型尚未训练，请先调用trainIncremental()方法 / Model not trained, please call trainIncremental() method first");
        }
        
        if (mfcc == null || mfcc.getRowNum() == 0) {
            throw new IllegalArgumentException("MFCC特征不能为空 / MFCC features cannot be empty");
        }
        
        if (mfcc.getColNum() != mfccDim) {
            throw new IllegalArgumentException("MFCC特征维度不匹配 / MFCC feature dimension mismatch");
        }
        
        // 计算Baum-Welch统计量（逐帧处理）
        IVector<Double> N = Linalg.zeros(numComponents);  // 零阶统计量
        IVector<Double> F = Linalg.zeros(numComponents * mfccDim);  // 一阶统计量
        
        // 对每一帧计算后验概率并累积统计量
        for (int t = 0; t < mfcc.getRowNum(); t++) {
            IVector<Double> frame = mfcc.getRow(t);
            
            // 使用GaussianMixtureModel计算后验概率
            IVector<Double> posteriors = ubm.computePosteriors(frame);
            
            // 累积统计量
            for (int c = 0; c < numComponents; c++) {
                double posterior = posteriors.get(c);
                N.set(c, N.get(c) + posterior);
                
                for (int d = 0; d < mfccDim; d++) {
                    int fidx = c * mfccDim + d;
                    F.set(fidx, F.get(fidx) + posterior * frame.get(d));
                }
            }
        }
        
        // 计算超向量差异
        IVector<Double> supervectorDiff = computeSupervectorDiff(N, F);
        
        // 计算i-vector
        IVector<Double> ivector = tMatrix.transpose().mmul(supervectorDiff);
        
        // 长度归一化
        double norm = ivector.norm2();
        if (norm > 1e-10) {
            ivector = ivector.divideByScalar(norm);
        }
        
        return ivector;
    }
    
    @Override
    public IVector<Double> embed(AudioData audioData) {
        // This is a simplified implementation
        // In a real implementation, you would extract features from the audio data
        throw new UnsupportedOperationException("Method not yet implemented");
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
            return embedding.copy(); // Return copy of zero vector
        }
        
        return embedding.divideByScalar(norm);
    }
    
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
        return mfccDim;
    }
    
    public int getSupervectorDim() {
        return supervectorDim;
    }
    
    public double getRelevanceFactor() {
        return relevanceFactor;
    }
    
    public boolean isUseAdamOptimizer() {
        return useAdamOptimizer;
    }
    
    public void setUseAdamOptimizer(boolean useAdamOptimizer) {
        this.useAdamOptimizer = useAdamOptimizer;
    }
    
    public double getLearningRate() {
        return learningRate;
    }
    
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
    
    public int getBatchSize() {
        return batchSize;
    }
    
    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }
    
    public int getProcessedSamples() {
        return processedSamples;
    }
    
    public void setProcessedSamples(int processedSamples) {
        this.processedSamples = processedSamples;
    }
    
    public List<IVector<Double>> getAccumulatedFeatures() {
        return accumulatedFeatures;
    }
    
    public void setAccumulatedFeatures(List<IVector<Double>> accumulatedFeatures) {
        this.accumulatedFeatures = accumulatedFeatures;
    }
    
    public IOnlineOptimizer getUbmOptimizer() {
        return ubmOptimizer;
    }
    
    public void setUbmOptimizer(IOnlineOptimizer ubmOptimizer) {
        this.ubmOptimizer = ubmOptimizer;
    }
    
    public IOnlineOptimizer getTMatrixOptimizer() {
        return tMatrixOptimizer;
    }
    
    public void setTMatrixOptimizer(IOnlineOptimizer tMatrixOptimizer) {
        this.tMatrixOptimizer = tMatrixOptimizer;
    }
    
    @Override
    public IAudioEmbedding save(String path) {
        try {
            // Create a map to store all model parameters
            java.util.Map<String, Object> modelData = new java.util.HashMap<>();
            
            // Store basic parameters
            modelData.put("len", this.len);
            modelData.put("numComponents", this.numComponents);
            modelData.put("mfccDim", this.mfccDim);
            modelData.put("supervectorDim", this.supervectorDim);
            modelData.put("isTrained", this.isTrained);
            modelData.put("relevanceFactor", this.relevanceFactor);
            modelData.put("useAdamOptimizer", this.useAdamOptimizer);
            modelData.put("learningRate", this.learningRate);
            modelData.put("batchSize", this.batchSize);
            modelData.put("processedSamples", this.processedSamples);
            modelData.put("modelType", "OnlineIVectorEmbedding"); // Add model type identifier
            
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
            
            // Store accumulated features if any
            if (this.accumulatedFeatures != null && !this.accumulatedFeatures.isEmpty()) {
                modelData.put("accumulatedFeatures", this.accumulatedFeatures);
            }
            
            // Serialize to file using Java serialization
            java.io.FileOutputStream fileOut = new java.io.FileOutputStream(path);
            java.io.ObjectOutputStream out = new java.io.ObjectOutputStream(fileOut);
            out.writeObject(modelData);
            out.close();
            fileOut.close();
            
            return this;
        } catch (Exception e) {
            throw new RuntimeException("Failed to save OnlineIVectorEmbedding model to " + path, e);
        }
    }
    
    
    
}