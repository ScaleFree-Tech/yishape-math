package com.reremouse.lab.math.audio;

/**
 * 音频特征类 / Audio Features Class
 * <p>
 * 封装音频的各种特征，包括频谱特征、时域特征等。
 * Encapsulates various audio features including spectral features, temporal features, etc.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class AudioFeatures {
    
    /** 频谱质心 (Hz) / Spectral centroid (Hz) */
    private final double spectralCentroid;
    
    /** 频谱带宽 (Hz) / Spectral bandwidth (Hz) */
    private final double spectralBandwidth;
    
    /** 频谱滚降点 (Hz) / Spectral rolloff (Hz) */
    private final double spectralRolloff;
    
    /** 零交叉率 / Zero crossing rate */
    private final double zeroCrossingRate;
    
    /** MFCC特征 / MFCC features */
    private final double[] mfcc;
    
    /** 频谱对比度 / Spectral contrast */
    private final double[] spectralContrast;
    
    /** 采样率 (Hz) / Sample rate (Hz) */
    private final double sampleRate;
    
    /**
     * 构造函数 / Constructor
     *
     * @param spectralCentroid 频谱质心 / Spectral centroid
     * @param spectralBandwidth 频谱带宽 / Spectral bandwidth
     * @param spectralRolloff 频谱滚降点 / Spectral rolloff
     * @param zeroCrossingRate 零交叉率 / Zero crossing rate
     * @param mfcc MFCC特征 / MFCC features
     * @param spectralContrast 频谱对比度 / Spectral contrast
     * @param sampleRate 采样率 / Sample rate
     */
    public AudioFeatures(double spectralCentroid, double spectralBandwidth, double spectralRolloff,
                        double zeroCrossingRate, double[] mfcc, double[] spectralContrast, double sampleRate) {
        this.spectralCentroid = spectralCentroid;
        this.spectralBandwidth = spectralBandwidth;
        this.spectralRolloff = spectralRolloff;
        this.zeroCrossingRate = zeroCrossingRate;
        this.mfcc = mfcc.clone();
        this.spectralContrast = spectralContrast.clone();
        this.sampleRate = sampleRate;
    }
    
    /**
     * 获取频谱质心 / Get spectral centroid
     *
     * @return 频谱质心 (Hz) / Spectral centroid (Hz)
     */
    public double getSpectralCentroid() {
        return spectralCentroid;
    }
    
    /**
     * 获取频谱带宽 / Get spectral bandwidth
     *
     * @return 频谱带宽 (Hz) / Spectral bandwidth (Hz)
     */
    public double getSpectralBandwidth() {
        return spectralBandwidth;
    }
    
    /**
     * 获取频谱滚降点 / Get spectral rolloff
     *
     * @return 频谱滚降点 (Hz) / Spectral rolloff (Hz)
     */
    public double getSpectralRolloff() {
        return spectralRolloff;
    }
    
    /**
     * 获取零交叉率 / Get zero crossing rate
     *
     * @return 零交叉率 / Zero crossing rate
     */
    public double getZeroCrossingRate() {
        return zeroCrossingRate;
    }
    
    /**
     * 获取MFCC特征 / Get MFCC features
     *
     * @return MFCC特征数组 / MFCC feature array
     */
    public double[] getMfcc() {
        return mfcc.clone();
    }
    
    /**
     * 获取指定索引的MFCC特征 / Get MFCC feature at specified index
     *
     * @param index 特征索引 / Feature index
     * @return MFCC特征值 / MFCC feature value
     * @throws IndexOutOfBoundsException 如果索引超出范围 / If index is out of bounds
     */
    public double getMfcc(int index) {
        if (index < 0 || index >= mfcc.length) {
            throw new IndexOutOfBoundsException("MFCC index out of bounds: " + index);
        }
        return mfcc[index];
    }
    
    /**
     * 获取MFCC特征数量 / Get number of MFCC features
     *
     * @return MFCC特征数量 / Number of MFCC features
     */
    public int getMfccCount() {
        return mfcc.length;
    }
    
    /**
     * 获取频谱对比度 / Get spectral contrast
     *
     * @return 频谱对比度数组 / Spectral contrast array
     */
    public double[] getSpectralContrast() {
        return spectralContrast.clone();
    }
    
    /**
     * 获取指定索引的频谱对比度 / Get spectral contrast at specified index
     *
     * @param index 特征索引 / Feature index
     * @return 频谱对比度值 / Spectral contrast value
     * @throws IndexOutOfBoundsException 如果索引超出范围 / If index is out of bounds
     */
    public double getSpectralContrast(int index) {
        if (index < 0 || index >= spectralContrast.length) {
            throw new IndexOutOfBoundsException("Spectral contrast index out of bounds: " + index);
        }
        return spectralContrast[index];
    }
    
    /**
     * 获取频谱对比度数量 / Get number of spectral contrast features
     *
     * @return 频谱对比度数量 / Number of spectral contrast features
     */
    public int getSpectralContrastCount() {
        return spectralContrast.length;
    }
    
    /**
     * 获取采样率 / Get sample rate
     *
     * @return 采样率 (Hz) / Sample rate (Hz)
     */
    public double getSampleRate() {
        return sampleRate;
    }
    
    /**
     * 获取所有特征作为数组 / Get all features as array
     *
     * @return 特征数组 / Feature array
     */
    public double[] getAllFeatures() {
        int totalFeatures = 4 + mfcc.length + spectralContrast.length; // 4个基本特征 + MFCC + 频谱对比度
        double[] allFeatures = new double[totalFeatures];
        
        int index = 0;
        allFeatures[index++] = spectralCentroid;
        allFeatures[index++] = spectralBandwidth;
        allFeatures[index++] = spectralRolloff;
        allFeatures[index++] = zeroCrossingRate;
        
        System.arraycopy(mfcc, 0, allFeatures, index, mfcc.length);
        index += mfcc.length;
        
        System.arraycopy(spectralContrast, 0, allFeatures, index, spectralContrast.length);
        
        return allFeatures;
    }
    
    /**
     * 获取特征数量 / Get number of features
     *
     * @return 总特征数量 / Total number of features
     */
    public int getFeatureCount() {
        return 4 + mfcc.length + spectralContrast.length;
    }
    
    /**
     * 计算特征向量之间的欧几里得距离 / Calculate Euclidean distance between feature vectors
     *
     * @param other 另一个特征向量 / Another feature vector
     * @return 欧几里得距离 / Euclidean distance
     */
    public double distanceTo(AudioFeatures other) {
        double[] thisFeatures = getAllFeatures();
        double[] otherFeatures = other.getAllFeatures();
        
        if (thisFeatures.length != otherFeatures.length) {
            throw new IllegalArgumentException("Feature vectors must have the same length");
        }
        
        double sumSquaredDiff = 0;
        for (int i = 0; i < thisFeatures.length; i++) {
            double diff = thisFeatures[i] - otherFeatures[i];
            sumSquaredDiff += diff * diff;
        }
        
        return Math.sqrt(sumSquaredDiff);
    }
    
    /**
     * 计算特征向量之间的余弦相似度 / Calculate cosine similarity between feature vectors
     *
     * @param other 另一个特征向量 / Another feature vector
     * @return 余弦相似度 / Cosine similarity
     */
    public double cosineSimilarityTo(AudioFeatures other) {
        double[] thisFeatures = getAllFeatures();
        double[] otherFeatures = other.getAllFeatures();
        
        if (thisFeatures.length != otherFeatures.length) {
            throw new IllegalArgumentException("Feature vectors must have the same length");
        }
        
        double dotProduct = 0;
        double normThis = 0;
        double normOther = 0;
        
        for (int i = 0; i < thisFeatures.length; i++) {
            dotProduct += thisFeatures[i] * otherFeatures[i];
            normThis += thisFeatures[i] * thisFeatures[i];
            normOther += otherFeatures[i] * otherFeatures[i];
        }
        
        if (normThis == 0 || normOther == 0) {
            return 0;
        }
        
        return dotProduct / (Math.sqrt(normThis) * Math.sqrt(normOther));
    }
    
    /**
     * 获取特征摘要 / Get feature summary
     *
     * @return 特征摘要字符串 / Feature summary string
     */
    public String getSummary() {
        return String.format(
            "AudioFeatures{spectralCentroid=%.2fHz, spectralBandwidth=%.2fHz, " +
            "spectralRolloff=%.2fHz, zeroCrossingRate=%.4f, mfccCount=%d, " +
            "spectralContrastCount=%d, totalFeatures=%d}",
            spectralCentroid, spectralBandwidth, spectralRolloff, zeroCrossingRate,
            mfcc.length, spectralContrast.length, getFeatureCount()
        );
    }
    
    @Override
    public String toString() {
        return getSummary();
    }
}
