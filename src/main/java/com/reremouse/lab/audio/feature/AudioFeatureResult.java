package com.reremouse.lab.audio.feature;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.util.Tuple2;
import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;

/**
 * 音频特征类 / Audio Features Class
 * <p>
 * 封装音频的各种特征，包括频谱特征、时域特征等 Encapsulates various audio features including spectral
 * features, temporal features, etc.
 * </p>
 *
 * @author lteb2
 * @version 2.0
 * @since 1.0
 */
public class AudioFeatureResult {

    private final TimeDomainFeatureResult timeDomainFeatures;
    private final FrequencyDomainFeatureResult frequencyDomainFeatures;
    private final SpectralFeatureResult spectralFeatures;
    private final Map<String, Object> metadata;

    /**
     * 频谱质心 (Hz) / Spectral centroid (Hz)
     */
    private final double spectralCentroid;

    /**
     * 频谱带宽 (Hz) / Spectral bandwidth (Hz)
     */
    private final double spectralBandwidth;

    /**
     * 频谱滚降点 (Hz) / Spectral rolloff (Hz)
     */
    private final double spectralRolloff;

    /**
     * 零交叉率 / Zero crossing rate
     */
    private final double zeroCrossingRate;

    /**
     * MFCC特征（均值） / MFCC features (means)
     */
    private final double[] mfccMeans;

    /**
     * 频谱对比度 / Spectral contrast
     */
    private final double[] spectralContrast;

    /**
     * 采样率 (Hz) / Sample rate (Hz)
     */
    private final double sampleRate;

    /**
     * 构造函数 / Constructor
     *
     * @param timeDomainFeatures 时域特征 / Time-domain features
     * @param frequencyDomainFeatures 频域特征 / Frequency-domain features
     * @param spectralFeatures 谱特征 / Spectral features
     */
    public AudioFeatureResult(TimeDomainFeatureResult timeDomainFeatures,
            FrequencyDomainFeatureResult frequencyDomainFeatures,
            SpectralFeatureResult spectralFeatures) {
        this.timeDomainFeatures = timeDomainFeatures;
        this.frequencyDomainFeatures = frequencyDomainFeatures;
        this.spectralFeatures = spectralFeatures;
        this.metadata = new HashMap<>();
        this.spectralCentroid = 0.0;
        this.spectralBandwidth = 0.0;
        this.spectralRolloff = 0.0;
        this.zeroCrossingRate = 0.0;
        this.mfccMeans = new double[0];
        this.spectralContrast = new double[0];
        this.sampleRate = 0.0; // Will be set via metadata
    }

    /**
     * 构造函数 / Constructor
     *
     * @param spectralCentroid 频谱质心 / Spectral centroid
     * @param spectralBandwidth 频谱带宽 / Spectral bandwidth
     * @param spectralRolloff 频谱滚降点 / Spectral rolloff
     * @param zeroCrossingRate 零交叉率 / Zero crossing rate
     * @param mfcc MFCC特征（均值） / MFCC features (means)
     * @param spectralContrast 频谱对比度 / Spectral contrast
     * @param sampleRate 采样率 / Sample rate
     */
    public AudioFeatureResult(double spectralCentroid, double spectralBandwidth, double spectralRolloff,
            double zeroCrossingRate, double[] mfccMeans, double[] spectralContrast, double sampleRate) {
        this.timeDomainFeatures = null;
        this.frequencyDomainFeatures = null;
        this.spectralFeatures = null;
        this.metadata = new HashMap<>();
        this.spectralCentroid = spectralCentroid;
        this.spectralBandwidth = spectralBandwidth;
        this.spectralRolloff = spectralRolloff;
        this.zeroCrossingRate = zeroCrossingRate;
        this.mfccMeans = mfccMeans.clone();
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
        return mfccMeans.clone();
    }

    /**
     * 获取指定索引的MFCC特征 / Get MFCC feature at specified index
     *
     * @param index 特征索引 / Feature index
     * @return MFCC特征值 / MFCC feature value
     * @throws IndexOutOfBoundsException 如果索引超出范围 / If index is out of bounds
     */
    public double getMfcc(int index) {
        if (index < 0 || index >= mfccMeans.length) {
            throw new IndexOutOfBoundsException("MFCC index out of bounds: " + index);
        }
        return mfccMeans[index];
    }

    /**
     * 获取MFCC特征数量 / Get number of MFCC features
     *
     * @return MFCC特征数量 / Number of MFCC features
     */
    public int getMfccCount() {
        return mfccMeans.length;
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
     * 获取元数据 / Get metadata
     */
    public Map<String, Object> getMetadata() {
        return metadata;
    }

    /**
     * 添加元数据 / Add metadata
     */
    public void addMetadata(String key, Object value) {
        metadata.put(key, value);
    }

    /**
     * 获取所有特征作为数组 / Get all features as array
     *
     * @return 特征数组 / Feature array
     */
    public double[] getAllFeatures() {
        int totalFeatures = 4 + mfccMeans.length + spectralContrast.length; // 4个基本特征 + MFCC + 频谱对比度
        double[] allFeatures = new double[totalFeatures];

        int index = 0;
        allFeatures[index++] = spectralCentroid;
        allFeatures[index++] = spectralBandwidth;
        allFeatures[index++] = spectralRolloff;
        allFeatures[index++] = zeroCrossingRate;

        System.arraycopy(mfccMeans, 0, allFeatures, index, mfccMeans.length);
        index += mfccMeans.length;

        System.arraycopy(spectralContrast, 0, allFeatures, index, spectralContrast.length);

        return allFeatures;
    }

    /**
     * 获取特征数量 / Get number of features
     *
     * @return 总特征数量 / Total number of features
     */
    public int getFeatureCount() {
        return 4 + mfccMeans.length + spectralContrast.length;
    }

    /**
     * 计算特征向量之间的欧几里得距离 / Calculate Euclidean distance between feature vectors
     *
     * @param other 另一个特征向量 / Another feature vector
     * @return 欧几里得距离 / Euclidean distance
     */
    public double distanceTo(AudioFeatureResult other) {
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
    public double cosineSimilarityTo(AudioFeatureResult other) {
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
                "AudioFeatures{spectralCentroid=%.2fHz, spectralBandwidth=%.2fHz, "
                + "spectralRolloff=%.2fHz, zeroCrossingRate=%.4f, mfccCount=%d, "
                + "spectralContrastCount=%d, totalFeatures=%d}",
                spectralCentroid, spectralBandwidth, spectralRolloff, zeroCrossingRate,
                mfccMeans.length, spectralContrast.length, getFeatureCount()
        );
    }

    /**
     * 将提取的特征全部转换为数值特征，其中List<String>中记录了特征名称，IVector<Double>中记录了特征的值。
     * @return
     */
    public Tuple2<List<String>, IVector<Double>> toNumericalFeatures() {
        // 创建特征名称列表
        List<String> featureNames = new ArrayList<>();
        
        // 创建特征值列表
        List<Double> featureValues = new ArrayList<>();
        
        // 添加时域特征
        if (timeDomainFeatures != null) {
            featureNames.add("timeDomain.rmsEnergy");
            featureValues.add(timeDomainFeatures.getRmsEnergy());
            
            featureNames.add("timeDomain.zeroCrossingRate");
            featureValues.add(timeDomainFeatures.getZeroCrossingRate());
            
            featureNames.add("timeDomain.energy");
            featureValues.add(timeDomainFeatures.getEnergy());
            
            featureNames.add("timeDomain.confidence");
            featureValues.add(timeDomainFeatures.getConfidence());
        }
        
        // 添加频域特征
        if (frequencyDomainFeatures != null) {
            featureNames.add("frequencyDomain.spectralCentroid");
            featureValues.add(frequencyDomainFeatures.getSpectralCentroid());
            
            featureNames.add("frequencyDomain.spectralBandwidth");
            featureValues.add(frequencyDomainFeatures.getSpectralBandwidth());
            
            featureNames.add("frequencyDomain.spectralRolloff");
            featureValues.add(frequencyDomainFeatures.getSpectralRolloff());
            
            featureNames.add("frequencyDomain.spectralFlatness");
            featureValues.add(frequencyDomainFeatures.getSpectralFlatness());
            
            featureNames.add("frequencyDomain.spectralFlux");
            featureValues.add(frequencyDomainFeatures.getSpectralFlux());
            
            featureNames.add("frequencyDomain.confidence");
            featureValues.add(frequencyDomainFeatures.getConfidence());
            
            // 添加频谱对比度特征
            double[] spectralContrast = frequencyDomainFeatures.getSpectralContrast();
            for (int i = 0; i < spectralContrast.length; i++) {
                featureNames.add("frequencyDomain.spectralContrast_" + i);
                featureValues.add(spectralContrast[i]);
            }
        }
        
        // 添加谱特征
        if (spectralFeatures != null) {
            featureNames.add("spectral.confidence");
            featureValues.add(spectralFeatures.getConfidence());
            
            // 添加MFCC特征（均值）
            double[] mfccMeans = spectralFeatures.getMfcc();
            for (int i = 0; i < mfccMeans.length; i++) {
                featureNames.add("spectral.mfccMeans_" + i);
                featureValues.add(mfccMeans[i]);
            }
            
            // 添加色度特征
            double[] chroma = spectralFeatures.getChroma();
            for (int i = 0; i < chroma.length; i++) {
                featureNames.add("spectral.chroma_" + i);
                featureValues.add(chroma[i]);
            }
            
            // 添加其他谱特征
            double[] spectralFeaturesArray = spectralFeatures.getSpectralFeatures();
            for (int i = 0; i < spectralFeaturesArray.length; i++) {
                featureNames.add("spectral.spectralFeature_" + i);
                featureValues.add(spectralFeaturesArray[i]);
            }
            
            // 添加MFCC一阶差分
            double[] mfccDelta = spectralFeatures.getMfccDelta();
            for (int i = 0; i < mfccDelta.length; i++) {
                featureNames.add("spectral.mfccDelta_" + i);
                featureValues.add(mfccDelta[i]);
            }
            
            // 添加MFCC二阶差分
            double[] mfccDeltaDelta = spectralFeatures.getMfccDeltaDelta();
            for (int i = 0; i < mfccDeltaDelta.length; i++) {
                featureNames.add("spectral.mfccDeltaDelta_" + i);
                featureValues.add(mfccDeltaDelta[i]);
            }
        }
        
        // 添加直接存储的特征（向后兼容）
        if (timeDomainFeatures == null && frequencyDomainFeatures == null && spectralFeatures == null) {
            featureNames.add("spectralCentroid");
            featureValues.add(spectralCentroid);
            
            featureNames.add("spectralBandwidth");
            featureValues.add(spectralBandwidth);
            
            featureNames.add("spectralRolloff");
            featureValues.add(spectralRolloff);
            
            featureNames.add("zeroCrossingRate");
            featureValues.add(zeroCrossingRate);
            
            // 添加MFCC特征（均值）
            for (int i = 0; i < mfccMeans.length; i++) {
                featureNames.add("mfccMeans_" + i);
                featureValues.add(mfccMeans[i]);
            }
            
            // 添加频谱对比度特征
            for (int i = 0; i < spectralContrast.length; i++) {
                featureNames.add("spectralContrast_" + i);
                featureValues.add(spectralContrast[i]);
            }
        }
        
        // 转换为数组
        double[] values = new double[featureValues.size()];
        for (int i = 0; i < values.length; i++) {
            values[i] = featureValues.get(i);
        }
        
        // 创建向量
        IVector<Double> featureVector = Linalg.vector(values);
        
        return new Tuple2<>(featureNames, featureVector);
    }

    @Override
    public String toString() {
        return getSummary();
    }
}
