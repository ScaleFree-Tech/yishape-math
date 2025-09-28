package com.reremouse.lab.music.analysis.feature;

import java.util.Arrays;

/**
 * 节拍特征结果 / Rhythm Feature Result
 * <p>
 * 封装音乐的节拍、节奏相关特征数据。
 * Encapsulates rhythm and beat-related feature data of music.
 * </p>
 *
 * @author lteb2
 * @version 2.0
 * @since 2.0
 */
public class RhythmFeatureResult {
    
    private final double tempo;                    // 节拍速度 (BPM)
    private final double beatStrength;             // 节拍强度
    private final double[] rhythmPattern;          // 节奏模式
    private final double rhythmRegularity;        // 节奏规律性
    private final double syncopation;             // 切分音程度
    private final double[] onsetTimes;            // 起始时间点
    private final double confidence;              // 置信度
    
    /**
     * 构造函数 / Constructor
     */
    public RhythmFeatureResult(double tempo, double beatStrength, double[] rhythmPattern,
                              double rhythmRegularity, double syncopation, double[] onsetTimes,
                              double confidence) {
        this.tempo = tempo;
        this.beatStrength = beatStrength;
        this.rhythmPattern = rhythmPattern != null ? rhythmPattern.clone() : new double[0];
        this.rhythmRegularity = rhythmRegularity;
        this.syncopation = syncopation;
        this.onsetTimes = onsetTimes != null ? onsetTimes.clone() : new double[0];
        this.confidence = confidence;
    }
    
    /**
     * 获取节拍速度 / Get tempo
     */
    public double getTempo() {
        return tempo;
    }
    
    /**
     * 获取节拍强度 / Get beat strength
     */
    public double getBeatStrength() {
        return beatStrength;
    }
    
    /**
     * 获取节奏模式 / Get rhythm pattern
     */
    public double[] getRhythmPattern() {
        return rhythmPattern.clone();
    }
    
    /**
     * 获取节奏规律性 / Get rhythm regularity
     */
    public double getRhythmRegularity() {
        return rhythmRegularity;
    }
    
    /**
     * 获取切分音程度 / Get syncopation level
     */
    public double getSyncopation() {
        return syncopation;
    }
    
    /**
     * 获取起始时间点 / Get onset times
     */
    public double[] getOnsetTimes() {
        return onsetTimes.clone();
    }
    
    /**
     * 获取置信度 / Get confidence
     */
    public double getConfidence() {
        return confidence;
    }
    
    /**
     * 判断是否为快节拍 / Check if fast tempo
     */
    public boolean isFastTempo() {
        return tempo > 120.0;
    }
    
    /**
     * 判断是否为慢节拍 / Check if slow tempo
     */
    public boolean isSlowTempo() {
        return tempo < 80.0;
    }
    
    /**
     * 获取节拍类型描述 / Get tempo type description
     */
    public String getTempoType() {
        if (tempo < 60) return "Largo";
        else if (tempo < 80) return "Adagio";
        else if (tempo < 100) return "Andante";
        else if (tempo < 120) return "Moderato";
        else if (tempo < 140) return "Allegro";
        else if (tempo < 180) return "Presto";
        else return "Prestissimo";
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("RhythmFeatureResult{");
        sb.append("tempo=").append(String.format("%.1f", tempo)).append(" BPM");
        sb.append(", type=").append(getTempoType());
        sb.append(", beatStrength=").append(String.format("%.3f", beatStrength));
        sb.append(", rhythmRegularity=").append(String.format("%.3f", rhythmRegularity));
        sb.append(", syncopation=").append(String.format("%.3f", syncopation));
        sb.append(", confidence=").append(String.format("%.3f", confidence));
        if (onsetTimes.length > 0) {
            sb.append(", onsets=").append(onsetTimes.length);
            sb.append(", firstOnset=").append(String.format("%.3f", onsetTimes[0]));
            sb.append(", lastOnset=").append(String.format("%.3f", onsetTimes[onsetTimes.length - 1]));
        }
        if (rhythmPattern.length > 0) {
            sb.append(", patternLength=").append(rhythmPattern.length);
        }
        sb.append('}');
        return sb.toString();
    }
}