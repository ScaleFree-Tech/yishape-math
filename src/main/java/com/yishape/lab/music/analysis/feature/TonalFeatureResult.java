package com.yishape.lab.music.analysis.feature;

import java.util.Arrays;

/**
 * 调性特征结果 / Tonal Feature Result
 * <p>
 * 封装音乐的调性、和声相关特征数据。
 * Encapsulates tonal and harmonic feature data of music.
 * </p>
 *
 * @author lteb2
 * @version 2.0
 * @since 2.0
 */
public class TonalFeatureResult {

    /** 调性 (如"C major", "A minor") / Key (e.g., "C major", "A minor") */
    private final String key;
    /** 调式 (major/minor) / Mode (major/minor) */
    private final String mode;
    /** 色度向量 (12维) / Chroma vector (12 dimensions) */
    private final double[] chromaVector;
    /** 调性强度 / Key strength */
    private final double keyStrength;
    /** 调性重心 / Tonal centroid */
    private final double tonalCentroid;
    /** 和声复杂度 / Harmonic complexity */
    private final double harmonicComplexity;
    /** 检测到的和弦 / Detected chords */
    private final String[] detectedChords;
    /** 和弦进行强度 / Chord progression strength */
    private final double[] chordProgression;
    /** 置信度 / Confidence */
    private final double confidence;
    /** Tonnetz特征 (6维) / Tonnetz features (6 dimensions) */
    private final double[] tonnetzFeatures;
    /** HPCP特征 (12维) / HPCP features (12 dimensions) */
    private final double[] hpcpFeatures;
    /** 音调稳定性特征 (4维) / Pitch stability features (4 dimensions) */
    private final double[] pitchStabilityFeatures;
    /** 调性稳定性 / Tonal stability */
    private final double tonalStability;

    /**
     * 完整构造函数 / Full constructor
     *
     * @param key 调性 / Key
     * @param mode 调式 / Mode
     * @param chromaVector 色度向量 (12维) / Chroma vector (12 dimensions)
     * @param keyStrength 调性强度 / Key strength
     * @param tonalCentroid 调性重心 / Tonal centroid
     * @param harmonicComplexity 和声复杂度 / Harmonic complexity
     * @param detectedChords 检测到的和弦 / Detected chords
     * @param chordProgression 和弦进行强度 / Chord progression strength
     * @param confidence 置信度 / Confidence
     * @param tonnetzFeatures Tonnetz特征 (6维) / Tonnetz features (6 dimensions)
     * @param hpcpFeatures HPCP特征 (12维) / HPCP features (12 dimensions)
     * @param pitchStabilityFeatures 音调稳定性特征 (4维) / Pitch stability features (4 dimensions)
     * @param tonalStability 调性稳定性 / Tonal stability
     */
    public TonalFeatureResult(String key, String mode, double[] chromaVector,
                             double keyStrength, double tonalCentroid, double harmonicComplexity,
                             String[] detectedChords, double[] chordProgression, double confidence,
                             double[] tonnetzFeatures, double[] hpcpFeatures,
                             double[] pitchStabilityFeatures, double tonalStability) {
        this.key = key;
        this.mode = mode;
        this.chromaVector = chromaVector != null ? chromaVector.clone() : new double[12];
        this.keyStrength = keyStrength;
        this.tonalCentroid = tonalCentroid;
        this.harmonicComplexity = harmonicComplexity;
        this.detectedChords = detectedChords != null ? detectedChords.clone() : new String[0];
        this.chordProgression = chordProgression != null ? chordProgression.clone() : new double[0];
        this.confidence = confidence;
        this.tonnetzFeatures = tonnetzFeatures != null ? tonnetzFeatures.clone() : new double[6];
        this.hpcpFeatures = hpcpFeatures != null ? hpcpFeatures.clone() : new double[12];
        this.pitchStabilityFeatures = pitchStabilityFeatures != null ? pitchStabilityFeatures.clone() : new double[4];
        this.tonalStability = tonalStability;
    }

    /**
     * 带置信度的简化构造函数 / Simplified constructor with confidence
     *
     * @param key 调性 / Key
     * @param mode 调式 / Mode
     * @param chromaVector 色度向量 (12维) / Chroma vector (12 dimensions)
     * @param keyStrength 调性强度 / Key strength
     * @param tonalStability 调性稳定性 / Tonal stability
     * @param confidence 置信度 / Confidence
     */
    public TonalFeatureResult(String key, String mode, double[] chromaVector,
                             double keyStrength, double tonalStability, double confidence) {
        this(key, mode, chromaVector, keyStrength, 0.0, 0.0,
             new String[0], new double[0], confidence,
             new double[6], new double[12], new double[4], tonalStability);
    }

    /**
     * 简化构造函数 (向后兼容) / Simplified constructor (backward compatible)
     *
     * @param key 调性 / Key
     * @param mode 调式 / Mode
     * @param chromaVector 色度向量 (12维) / Chroma vector (12 dimensions)
     * @param keyStrength 调性强度 / Key strength
     * @param tonalStability 调性稳定性 / Tonal stability
     */
    public TonalFeatureResult(String key, String mode, double[] chromaVector,
                             double keyStrength, double tonalStability) {
        this(key, mode, chromaVector, keyStrength, 0.0, 0.0,
             new String[0], new double[0], 0.0,
             new double[6], new double[12], new double[4], tonalStability);
    }

    /**
     * 获取调性 / Get key
     */
    public String getKey() {
        return key;
    }

    /**
     * 获取调式 / Get mode
     */
    public String getMode() {
        return mode;
    }

    /**
     * 获取完整调性描述 / Get full key description
     */
    public String getFullKey() {
        return key + " " + mode;
    }

    /**
     * 获取色度向量 / Get chroma vector
     */
    public double[] getChromaVector() {
        return chromaVector.clone();
    }

    /**
     * 获取调性强度 / Get key strength
     */
    public double getKeyStrength() {
        return keyStrength;
    }

    /**
     * 获取调性重心 / Get tonal centroid
     */
    public double getTonalCentroid() {
        return tonalCentroid;
    }

    /**
     * 获取和声复杂度 / Get harmonic complexity
     */
    public double getHarmonicComplexity() {
        return harmonicComplexity;
    }

    /**
     * 获取检测到的和弦 / Get detected chords
     */
    public String[] getDetectedChords() {
        return detectedChords.clone();
    }

    /**
     * 获取和弦进行强度 / Get chord progression
     */
    public double[] getChordProgression() {
        return chordProgression.clone();
    }

    /**
     * 获取置信度 / Get confidence
     */
    public double getConfidence() {
        return confidence;
    }

    /**
     * 获取Tonnetz特征 / Get Tonnetz features
     */
    public double[] getTonnetzFeatures() {
        return tonnetzFeatures.clone();
    }

    /**
     * 获取HPCP特征 / Get HPCP features
     */
    public double[] getHpcpFeatures() {
        return hpcpFeatures.clone();
    }

    /**
     * 获取音调稳定性特征 / Get pitch stability features
     */
    public double[] getPitchStabilityFeatures() {
        return pitchStabilityFeatures.clone();
    }

    /**
     * 获取调性稳定性 / Get tonal stability
     */
    public double getTonalStability() {
        return tonalStability;
    }

    /**
     * 判断是否为大调 / Check if major key
     *
     * @return 如果调式为大调返回true / True if mode is major
     */
    public boolean isMajorKey() {
        return "major".equalsIgnoreCase(mode);
    }

    /**
     * 判断是否为小调 / Check if minor key
     *
     * @return 如果调式为小调返回true / True if mode is minor
     */
    public boolean isMinorKey() {
        return "minor".equalsIgnoreCase(mode);
    }

    /**
     * 获取主要色度 / Get dominant chroma
     *
     * @return 主要色度的索引 (0-11) / Index of dominant chroma (0-11)
     */
    public int getDominantChroma() {
        int maxIndex = 0;
        double maxValue = chromaVector[0];
        for (int i = 1; i < chromaVector.length; i++) {
            if (chromaVector[i] > maxValue) {
                maxValue = chromaVector[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * 获取色度音名 / Get chroma name
     */
    public String getDominantChromaName() {
        String[] chromaNames = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
        return chromaNames[getDominantChroma()];
    }

    /**
     * 判断是否为强调性 / Check if key is strong
     *
     * @return 如果调性强度大于0.7返回true / True if key strength > 0.7
     */
    public boolean isStrongKey() {
        return keyStrength > 0.7;
    }

    /**
     * 判断调性是否稳定 / Check if tonality is stable
     *
     * @return 如果调性稳定性大于0.6返回true / True if tonal stability > 0.6
     */
    public boolean isStableTonality() {
        return tonalStability > 0.6;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("TonalFeatureResult{");
        sb.append("key=").append(getFullKey());
        sb.append(", keyStrength=").append(String.format("%.3f", keyStrength));
        sb.append(", tonalCentroid=").append(String.format("%.3f", tonalCentroid));
        sb.append(", harmonicComplexity=").append(String.format("%.3f", harmonicComplexity));
        sb.append(", tonalStability=").append(String.format("%.3f", tonalStability));
        sb.append(", confidence=").append(String.format("%.3f", confidence));
        if (detectedChords.length > 0) {
            sb.append(", chords=").append(detectedChords.length);
            if (detectedChords.length > 0 && detectedChords[0] != null) {
                sb.append(", firstChord=").append(detectedChords[0]);
            }
        }
        if (chordProgression.length > 0) {
            sb.append(", progressionLength=").append(chordProgression.length);
        }
        if (chromaVector.length >= 12) {
            sb.append(", dominantChroma=").append(getDominantChromaName());
        }
        sb.append('}');
        return sb.toString();
    }
}