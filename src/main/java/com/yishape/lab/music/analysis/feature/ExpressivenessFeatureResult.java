package com.yishape.lab.music.analysis.feature;

import java.util.HashMap;
import java.util.Map;

/**
 * 表现力特征结果 / Expressiveness Feature Result
 * <p>
 * 封装音乐的表现力、情感和风格相关特征数据。
 * Encapsulates expressiveness, emotional and stylistic feature data of music.
 * </p>
 *
 * @author lteb2
 * @version 2.0
 * @since 2.0
 */
public class ExpressivenessFeatureResult {

    // 情感维度特征 / Emotional dimension features
    /** 效价 (-1.0 ~ 1.0) / Valence (-1.0 ~ 1.0) */
    private final double valence;
    /** 唤醒度 (0.0 ~ 1.0) / Arousal (0.0 ~ 1.0) */
    private final double arousal;
    /** 支配度 (0.0 ~ 1.0) / Dominance (0.0 ~ 1.0) */
    private final double dominance;

    // 音乐风格特征 / Musical style features
    /** 音乐能量强度 (0.0 ~ 1.0) / Energy (0.0 ~ 1.0) */
    private final double energy;
    /** 可舞性 (0.0 ~ 1.0) / Danceability (0.0 ~ 1.0) */
    private final double danceability;
    /** 原声性 (0.0 ~ 1.0) / Acousticness (0.0 ~ 1.0) */
    private final double acousticness;
    /** 器乐性 (0.0 ~ 1.0) / Instrumentalness (0.0 ~ 1.0) */
    private final double instrumentalness;
    /** 现场感 (0.0 ~ 1.0) / Liveness (0.0 ~ 1.0) */
    private final double liveness;
    /** 语音性 (0.0 ~ 1.0) / Speechiness (0.0 ~ 1.0) */
    private final double speechiness;

    // 动态特征 / Dynamic features
    /** 动态范围变化 / Dynamic range */
    private final double[] dynamicRange;
    /** 频谱重心演化 / Spectral centroid evolution */
    private final double[] spectralCentroidEvolution;
    /** 情感强度 / Emotional intensity */
    private final double emotionalIntensity;
    /** 预测情绪标签 / Predicted mood label */
    private final String predictedMood;
    /** 置信度 (0.0 ~ 1.0) / Confidence (0.0 ~ 1.0) */
    private final double confidence;

    /**
     * 构造函数 / Constructor
     *
     * @param valence 效价 / Valence
     * @param arousal 唤醒度 / Arousal
     * @param dominance 支配度 / Dominance
     * @param energy 音乐能量强度 / Energy
     * @param danceability 可舞性 / Danceability
     * @param acousticness 原声性 / Acousticness
     * @param instrumentalness 器乐性 / Instrumentalness
     * @param liveness 现场感 / Liveness
     * @param speechiness 语音性 / Speechiness
     * @param dynamicRange 动态范围变化 / Dynamic range
     * @param spectralCentroidEvolution 频谱重心演化 / Spectral centroid evolution
     * @param emotionalIntensity 情感强度 / Emotional intensity
     * @param predictedMood 预测情绪标签 / Predicted mood label
     * @param confidence 置信度 / Confidence
     */
    public ExpressivenessFeatureResult(double valence, double arousal, double dominance,
                                     double energy, double danceability, double acousticness,
                                     double instrumentalness, double liveness, double speechiness,
                                     double[] dynamicRange, double[] spectralCentroidEvolution,
                                     double emotionalIntensity, String predictedMood, double confidence) {
        this.valence = valence;
        this.arousal = arousal;
        this.dominance = dominance;
        this.energy = energy;
        this.danceability = danceability;
        this.acousticness = acousticness;
        this.confidence = confidence;
        this.instrumentalness = instrumentalness;
        this.liveness = liveness;
        this.speechiness = speechiness;
        this.dynamicRange = dynamicRange != null ? dynamicRange.clone() : new double[0];
        this.spectralCentroidEvolution = spectralCentroidEvolution != null ? spectralCentroidEvolution.clone() : new double[0];
        this.emotionalIntensity = emotionalIntensity;
        this.predictedMood = predictedMood != null ? predictedMood : "Unknown";
    }

    // 情感维度特征的getter方法

    /**
     * 获取效价 / Get valence
     */
    public double getValence() {
        return valence;
    }

    /**
     * 获取唤醒度 / Get arousal
     */
    public double getArousal() {
        return arousal;
    }

    /**
     * 获取支配度 / Get dominance
     */
    public double getDominance() {
        return dominance;
    }

    // 音乐风格特征的getter方法

    /**
     * 获取音乐能量强度 / Get energy
     */
    public double getEnergy() {
        return energy;
    }

    /**
     * 获取可舞性 / Get danceability
     */
    public double getDanceability() {
        return danceability;
    }

    /**
     * 获取原声性 / Get acousticness
     */
    public double getAcousticness() {
        return acousticness;
    }

    /**
     * 获取器乐性 / Get instrumentalness
     */
    public double getInstrumentalness() {
        return instrumentalness;
    }

    /**
     * 获取现场感 / Get liveness
     */
    public double getLiveness() {
        return liveness;
    }

    /**
     * 获取语音性 / Get speechiness
     */
    public double getSpeechiness() {
        return speechiness;
    }

    // 动态特征的getter方法

    /**
     * 获取动态范围变化 / Get dynamic range
     */
    public double[] getDynamicRange() {
        return dynamicRange.clone();
    }

    /**
     * 获取频谱重心演化 / Get spectral centroid evolution
     */
    public double[] getSpectralCentroidEvolution() {
        return spectralCentroidEvolution.clone();
    }

    /**
     * 获取情感强度 / Get emotional intensity
     */
    public double getEmotionalIntensity() {
        return emotionalIntensity;
    }

    /**
     * 获取预测情绪标签 / Get predicted mood
     */
    public String getPredictedMood() {
        return predictedMood;
    }

    // 辅助方法

    /**
     * 判断是否为正面情感 / Check if positive emotion
     *
     * @return 效价大于0时返回true / True if valence > 0
     */
    public boolean isPositiveEmotion() {
        return valence > 0.0;
    }

    /**
     * 判断是否为高能量音乐 / Check if high energy music
     *
     * @return 能量大于0.7时返回true / True if energy > 0.7
     */
    public boolean isHighEnergy() {
        return energy > 0.7;
    }

    /**
     * 判断是否适合跳舞 / Check if suitable for dancing
     *
     * @return 可舞性大于0.6时返回true / True if danceability > 0.6
     */
    public boolean isSuitableForDancing() {
        return danceability > 0.6;
    }

    /**
     * 判断是否为原声音乐 / Check if acoustic music
     *
     * @return 原声性大于0.5时返回true / True if acousticness > 0.5
     */
    public boolean isAcousticMusic() {
        return acousticness > 0.5;
    }

    /**
     * 判断是否为器乐音乐 / Check if instrumental music
     *
     * @return 器乐性大于0.5时返回true / True if instrumentalness > 0.5
     */
    public boolean isInstrumentalMusic() {
        return instrumentalness > 0.5;
    }

    /**
     * 获取情感象限 / Get emotional quadrant
     */
    public String getEmotionalQuadrant() {
        if (valence > 0.0 && arousal > 0.5) {
            return "Excited/Happy";
        } else if (valence < 0.0 && arousal > 0.5) {
            return "Angry/Tense";
        } else if (valence > 0.0 && arousal <= 0.5) {
            return "Calm/Content";
        } else {
            return "Sad/Melancholy";
        }
    }

    /**
     * 获取置信度 / Get confidence
     *
     * @return 置信度 / Confidence
     */
    public double getConfidence() {
        return confidence;
    }

    /**
     * 计算综合表现力评分 / Calculate overall expressiveness score
     */
    public double getOverallExpressivenessScore() {
        // 基于多个维度计算综合评分
        double emotionalScore = (Math.abs(valence) + arousal + dominance) / 3.0;
        double styleScore = (energy + danceability + (1.0 - acousticness) + (1.0 - instrumentalness)) / 4.0;
        double intensityScore = emotionalIntensity;
        
        return (emotionalScore * 0.4 + styleScore * 0.4 + intensityScore * 0.2);
    }

    /**
     * 获取音色变化 / Get timbre variation
     */
    public double getTimbreVariation() {
        // For now, return a calculated value based on existing features
        // In a real implementation, this would be a specific calculation
        return (acousticness + instrumentalness) / 2.0;
    }
    
    /**
     * 获取发音清晰度 / Get articulation clarity
     */
    public double getArticulationClarity() {
        // For now, return a calculated value based on existing features
        return (1.0 - speechiness) * energy;
    }
    
    /**
     * 获取表现强度 / Get expressive intensity
     */
    public double getExpressiveIntensity() {
        return emotionalIntensity;
    }
    
    /**
     * 获取微时值 / Get microtiming
     */
    public double getMicrotiming() {
        // For now, return a calculated value based on existing features
        return (danceability + energy) / 2.0;
    }
    
    /**
     * 获取颤音 / Get vibrato
     */
    public double getVibrato() {
        // For now, return a calculated value based on existing features
        return acousticness * 0.3;
    }
    
    /**
     * 获取自由速度 / Get rubato
     */
    public double getRubato() {
        // For now, return a calculated value based on existing features
        return (1.0 - danceability) * 0.5;
    }
    
    /**
     * 获取附加特征 / Get additional features
     *
     * @return 附加特征映射 / Map of additional features
     */
    public Map<String, Double> getAdditionalFeatures() {
        Map<String, Double> additionalFeatures = new HashMap<>();
        // Add some placeholder features
        additionalFeatures.put("brightness", energy);
        additionalFeatures.put("depth", acousticness);
        additionalFeatures.put("warmth", instrumentalness);
        return additionalFeatures;
    }
    
    /**
     * 获取特定附加特征 / Get specific additional feature
     *
     * @param name 特征名称 / Feature name
     * @return 特征值，如果不存在则返回null / Feature value, or null if not found
     */
    public Double getAdditionalFeature(String name) {
        Map<String, Double> additionalFeatures = getAdditionalFeatures();
        return additionalFeatures.get(name);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("ExpressivenessFeatureResult{");
        
        // 情感维度
        sb.append("emotion={");
        sb.append("valence=").append(String.format("%.3f", valence));
        sb.append(", arousal=").append(String.format("%.3f", arousal));
        sb.append(", dominance=").append(String.format("%.3f", dominance));
        sb.append(", quadrant=").append(getEmotionalQuadrant());
        sb.append("}");
        
        // 音乐风格
        sb.append(", style={");
        sb.append("energy=").append(String.format("%.3f", energy));
        sb.append(", danceability=").append(String.format("%.3f", danceability));
        sb.append(", acousticness=").append(String.format("%.3f", acousticness));
        sb.append(", instrumentalness=").append(String.format("%.3f", instrumentalness));
        sb.append(", liveness=").append(String.format("%.3f", liveness));
        sb.append(", speechiness=").append(String.format("%.3f", speechiness));
        sb.append("}");
        
        // 动态特征
        sb.append(", dynamics={");
        sb.append("intensity=").append(String.format("%.3f", emotionalIntensity));
        sb.append(", mood=").append(predictedMood);
        if (dynamicRange.length > 0) {
            sb.append(", dynamicRangeLength=").append(dynamicRange.length);
        }
        if (spectralCentroidEvolution.length > 0) {
            sb.append(", spectralEvolutionLength=").append(spectralCentroidEvolution.length);
        }
        sb.append("}");
        
        sb.append(", overallScore=").append(String.format("%.3f", getOverallExpressivenessScore()));
        sb.append('}');
        
        return sb.toString();
    }
}