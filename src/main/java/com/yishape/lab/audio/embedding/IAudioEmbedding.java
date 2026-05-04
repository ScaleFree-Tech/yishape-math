package com.yishape.lab.audio.embedding;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.math.stats.model.GaussianMixtureModel;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;

/**
 * 音频嵌入接口 / Audio Embedding Interface
 * <p>
 * 定义音频嵌入向量生成的接口，支持将音频特征转换为固定长度的向量表示。
 * Defines interface for audio embedding vector generation, supporting conversion of audio features to fixed-length vector representations.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IAudioEmbedding {
    
    /**
     * 基于MFCC特征生成定长的向量表示 / Generate fixed-length vector representation based on MFCC features
     *
     * @param mfcc MFCC特征矩阵 / MFCC feature matrix
     * @return 嵌入向量 / Embedding vector
     */
    public IVector<Double> embed(IMatrix<Double> mfcc);

    /**
     * 基于音频数据生成嵌入向量 / Generate embedding vector based on audio data
     *
     * @param audioData 音频数据 / Audio data
     * @return 嵌入向量 / Embedding vector
     */
    public IVector<Double> embed(AudioData audioData);

    /**
     * 基于原始音频样本生成嵌入向量 / Generate embedding vector based on raw audio samples
     *
     * @param samples 音频样本 / Audio samples
     * @param sampleRate 采样率 (Hz) / Sample rate (Hz)
     * @return 嵌入向量 / Embedding vector
     */
    public IVector<Double> embed(IVector<Double> samples, int sampleRate);

    /**
     * 批量生成嵌入向量 / Batch generate embedding vectors
     *
     * @param mfccBatch MFCC特征矩阵批次 / Batch of MFCC feature matrices
     * @return 嵌入向量批次 / Batch of embedding vectors
     */
    public IMatrix<Double> embedBatch(IMatrix<Double>[] mfccBatch);

    /**
     * 保存模型 / Save model
     *
     * @param path 模型文件路径 / Model file path
     * @return 保存后的音频嵌入模型 / Saved audio embedding model
     */
    public IAudioEmbedding save(String path);
    
    /**
     * 获取嵌入向量的维度 / Get dimension of embedding vector
     *
     * @return 嵌入向量维度 / Embedding vector dimension
     */
    public int getEmbeddingDimension();
    
    /**
     * 获取支持的输入特征类型 / Get supported input feature types
     *
     * @return 支持的特征类型数组 / Array of supported feature types
     */
    public FeatureType[] getSupportedFeatureTypes();
    
    /**
     * 检查是否支持指定的特征类型 / Check if specified feature type is supported
     *
     * @param featureType 特征类型 / Feature type
     * @return 如果支持返回true / True if supported
     */
    public boolean supportsFeatureType(FeatureType featureType);
    
    /**
     * 设置嵌入参数 / Set embedding parameters
     *
     * @param parameters 参数映射 / Parameter map
     */
    public void setParameters(java.util.Map<String, Object> parameters);
    
    /**
     * 获取嵌入参数 / Get embedding parameters
     *
     * @return 参数映射 / Parameter map
     */
    public java.util.Map<String, Object> getParameters();
    
    /**
     * 计算两个嵌入向量之间的相似度 / Calculate similarity between two embedding vectors
     *
     * @param embedding1 第一个嵌入向量 / First embedding vector
     * @param embedding2 第二个嵌入向量 / Second embedding vector
     * @return 相似度分数 (0-1) / Similarity score (0-1)
     */
    public double calculateSimilarity(IVector<Double> embedding1, IVector<Double> embedding2);
    
    /**
     * 计算嵌入向量的距离 / Calculate distance between embedding vectors
     *
     * @param embedding1 第一个嵌入向量 / First embedding vector
     * @param embedding2 第二个嵌入向量 / Second embedding vector
     * @param distanceType 距离类型 / Distance type
     * @return 距离值 / Distance value
     */
    public double calculateDistance(IVector<Double> embedding1, IVector<Double> embedding2, DistanceType distanceType);
    
    /**
     * 从本地加载模型 / Load model from local storage
     *
     * @param path 模型文件路径 / Model file path
     * @return 加载的音频嵌入模型 / Loaded audio embedding model
     */
    public static IAudioEmbedding load(String path) {
        try {
            // Deserialize from file using Java serialization
            java.io.FileInputStream fileIn = new java.io.FileInputStream(path);
            java.io.ObjectInputStream in = new java.io.ObjectInputStream(fileIn);
            java.util.Map<String, Object> modelData = (java.util.Map<String, Object>) in.readObject();
            in.close();
            fileIn.close();
            
            // Check model type
            String modelType = (String) modelData.get("modelType");
            
            if ("IVectorEmbedding".equals(modelType)) {
                // This is an IVectorEmbedding model
                int len = (Integer) modelData.get("len");
                int numComponents = (Integer) modelData.get("numComponents");
                int featureDim = (Integer) modelData.get("featureDim");
                
                // Create new IVectorEmbedding instance
                IVectorEmbedding embedding = new IVectorEmbedding(len, numComponents, featureDim);
                
                // Restore model state using setter methods
                embedding.setTrained((Boolean) modelData.get("isTrained"));
                
                if (modelData.containsKey("ubm")) {
                    embedding.setUbm((GaussianMixtureModel) modelData.get("ubm"));
                }
                
                if (modelData.containsKey("tMatrix")) {
                    embedding.setTMatrix((IMatrix<Double>) modelData.get("tMatrix"));
                }
                
                if (modelData.containsKey("ubmInvCovariances")) {
                    embedding.setUbmInvCovariances((List<IMatrix<Double>>) modelData.get("ubmInvCovariances"));
                }
                
                return embedding;
            } 
            else if ("OnlineIVectorEmbedding".equals(modelType)) {
                // This is an OnlineIVectorEmbedding model
                int len = (Integer) modelData.get("len");
                int numComponents = (Integer) modelData.get("numComponents");
                int mfccDim = (Integer) modelData.get("mfccDim");
                
                // Create new OnlineIVectorEmbedding instance
                OnlineIVectorEmbedding embedding = new OnlineIVectorEmbedding(len, numComponents, mfccDim);
                
                // Restore model state using setter methods
                embedding.setTrained((Boolean) modelData.get("isTrained"));
                
                if (modelData.containsKey("ubm")) {
                    embedding.setUbm((GaussianMixtureModel) modelData.get("ubm"));
                }
                
                if (modelData.containsKey("tMatrix")) {
                    embedding.setTMatrix((IMatrix<Double>) modelData.get("tMatrix"));
                }
                
                if (modelData.containsKey("ubmInvCovariances")) {
                    embedding.setUbmInvCovariances((List<IMatrix<Double>>) modelData.get("ubmInvCovariances"));
                }
                
                // Restore online-specific parameters
                if (modelData.containsKey("useAdamOptimizer")) {
                    embedding.setUseAdamOptimizer((Boolean) modelData.get("useAdamOptimizer"));
                }
                
                if (modelData.containsKey("learningRate")) {
                    embedding.setLearningRate((Double) modelData.get("learningRate"));
                }
                
                if (modelData.containsKey("batchSize")) {
                    embedding.setBatchSize((Integer) modelData.get("batchSize"));
                }
                
                if (modelData.containsKey("processedSamples")) {
                    embedding.setProcessedSamples((Integer) modelData.get("processedSamples"));
                }
                
                if (modelData.containsKey("accumulatedFeatures")) {
                    embedding.setAccumulatedFeatures((List<IVector<Double>>) modelData.get("accumulatedFeatures"));
                }
                
                return embedding;
            }
            
            return null;
        } catch (Exception e) {
            throw new RuntimeException("Failed to load IAudioEmbedding model from " + path, e);
        }
    }
    
    /**
     * 对嵌入向量进行归一化 / Normalize embedding vector
     *
     * @param embedding 嵌入向量 / Embedding vector
     * @return 归一化后的嵌入向量 / Normalized embedding vector
     */
    public IVector<Double> normalize(IVector<Double> embedding);
    
    /**
     * 特征类型枚举 / Feature Type Enum
     */
    public enum FeatureType {
        /** MFCC特征 / MFCC features */
        MFCC("MFCC", "Mel-frequency cepstral coefficients"),
        
        /** 频谱特征 / Spectral features */
        SPECTRAL("SPECTRAL", "Spectral features"),
        
        /** 时域特征 / Time domain features */
        TIME_DOMAIN("TIME_DOMAIN", "Time domain features"),
        
        /** 色度特征 / Chroma features */
        CHROMA("CHROMA", "Chroma features"),
        
        /** 节拍特征 / Tempo features */
        TEMPO("TEMPO", "Tempo features"),
        
        /** 零交叉率 / Zero crossing rate */
        ZCR("ZCR", "Zero crossing rate"),
        
        /** 原始音频 / Raw audio */
        RAW_AUDIO("RAW_AUDIO", "Raw audio samples");
        
        private final String name;
        private final String description;
        
        FeatureType(String name, String description) {
            this.name = name;
            this.description = description;
        }
        
        public String getName() {
            return name;
        }
        
        public String getDescription() {
            return description;
        }
        
        @Override
        public String toString() {
            return String.format("%s (%s)", name, description);
        }
    }
    
    /**
     * 距离类型枚举 / Distance Type Enum
     */
    enum DistanceType {
        /** 欧几里得距离 / Euclidean distance */
        EUCLIDEAN("EUCLIDEAN", "Euclidean distance"),
        
        /** 曼哈顿距离 / Manhattan distance */
        MANHATTAN("MANHATTAN", "Manhattan distance"),
        
        /** 余弦距离 / Cosine distance */
        COSINE("COSINE", "Cosine distance"),
        
        /** 切比雪夫距离 / Chebyshev distance */
        CHEBYSHEV("CHEBYSHEV", "Chebyshev distance"),
        
        /** 汉明距离 / Hamming distance */
        HAMMING("HAMMING", "Hamming distance");
        
        private final String name;
        private final String description;
        
        DistanceType(String name, String description) {
            this.name = name;
            this.description = description;
        }
        
        public String getName() {
            return name;
        }
        
        public String getDescription() {
            return description;
        }
        
        @Override
        public String toString() {
            return String.format("%s (%s)", name, description);
        }
    }
}