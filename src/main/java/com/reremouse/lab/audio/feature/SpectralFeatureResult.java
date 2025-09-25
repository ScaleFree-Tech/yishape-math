package com.reremouse.lab.audio.feature;

/**
 * 谱特征结果 / Spectral Feature Result
 * <p>
 * 封装音频的谱相关特征数据，如MFCC、色度特征等。
 * Encapsulates spectral feature data of audio, such as MFCC, chroma features, etc.
 * </p>
 *
 * @author RereMouse
 * @version 2.0
 * @since 2.0
 */
public class SpectralFeatureResult {
    
    private final double[] mfccMeans;                   // MFCC特征（均值）
    private final double[] chroma;                 // 色度特征
    private final double[] spectralFeatures;       // 其他谱特征
    private final double[] mfccDelta;              // MFCC一阶差分
    private final double[] mfccDeltaDelta;         // MFCC二阶差分
    private final double confidence;               // 置信度
    
    /**
     * 构造函数 / Constructor
     */
    public SpectralFeatureResult(double[] mfccMeans, double[] chroma, double[] spectralFeatures,
                               double[] mfccDelta, double[] mfccDeltaDelta, double confidence) {
        this.mfccMeans = mfccMeans != null ? mfccMeans.clone() : new double[0];
        this.chroma = chroma != null ? chroma.clone() : new double[0];
        this.spectralFeatures = spectralFeatures != null ? spectralFeatures.clone() : new double[0];
        this.mfccDelta = mfccDelta != null ? mfccDelta.clone() : new double[0];
        this.mfccDeltaDelta = mfccDeltaDelta != null ? mfccDeltaDelta.clone() : new double[0];
        this.confidence = confidence;
    }
    
    /**
     * 获取MFCC特征（均值） / Get MFCC features (means)
     */
    public double[] getMfcc() {
        return mfccMeans.clone();
    }
    
    /**
     * 获取指定索引的MFCC特征（均值） / Get MFCC feature (mean) at specified index
     */
    public double getMfcc(int index) {
        if (index < 0 || index >= mfccMeans.length) {
            throw new IndexOutOfBoundsException("MFCC index out of bounds: " + index);
        }
        return mfccMeans[index];
    }
    
    /**
     * 获取MFCC特征数量 / Get number of MFCC features
     */
    public int getMfccCount() {
        return mfccMeans.length;
    }
    
    /**
     * 获取色度特征 / Get chroma features
     */
    public double[] getChroma() {
        return chroma.clone();
    }
    
    /**
     * 获取其他谱特征 / Get other spectral features
     */
    public double[] getSpectralFeatures() {
        return spectralFeatures.clone();
    }
    
    /**
     * 获取MFCC一阶差分 / Get MFCC delta
     */
    public double[] getMfccDelta() {
        return mfccDelta.clone();
    }
    
    /**
     * 获取MFCC二阶差分 / Get MFCC delta-delta
     */
    public double[] getMfccDeltaDelta() {
        return mfccDeltaDelta.clone();
    }
    
    /**
     * 获取置信度 / Get confidence
     */
    public double getConfidence() {
        return confidence;
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("SpectralFeatureResult{");
        sb.append("mfccCount=").append(mfccMeans.length);
        sb.append(", chromaLength=").append(chroma.length);
        sb.append(", spectralFeaturesLength=").append(spectralFeatures.length);
        sb.append(", confidence=").append(String.format("%.3f", confidence));
        if (mfccDelta.length > 0) {
            sb.append(", mfccDeltaLength=").append(mfccDelta.length);
        }
        if (mfccDeltaDelta.length > 0) {
            sb.append(", mfccDeltaDeltaLength=").append(mfccDeltaDelta.length);
        }
        sb.append('}');
        return sb.toString();
    }
}