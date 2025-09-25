package com.reremouse.lab.audio.feature;

/**
 * 时域特征结果 / Time-Domain Feature Result
 * <p>
 * 封装音频的时域相关特征数据，如均方根能量、过零率等。
 * Encapsulates time-domain feature data of audio, such as RMS energy, zero-crossing rate, etc.
 * </p>
 *
 * @author RereMouse
 * @version 2.0
 * @since 2.0
 */
public class TimeDomainFeatureResult {
    
    private final double rmsEnergy;                // 均方根能量
    private final double zeroCrossingRate;         // 过零率
    private final double energy;                   // 能量
    private final double[] amplitudeEnvelope;      // 幅度包络
    private final double confidence;               // 置信度
    
    /**
     * 构造函数 / Constructor
     */
    public TimeDomainFeatureResult(double rmsEnergy, double zeroCrossingRate, double energy,
                                 double[] amplitudeEnvelope, double confidence) {
        this.rmsEnergy = rmsEnergy;
        this.zeroCrossingRate = zeroCrossingRate;
        this.energy = energy;
        this.amplitudeEnvelope = amplitudeEnvelope != null ? amplitudeEnvelope.clone() : new double[0];
        this.confidence = confidence;
    }
    
    /**
     * 获取均方根能量 / Get RMS energy
     */
    public double getRmsEnergy() {
        return rmsEnergy;
    }
    
    /**
     * 获取过零率 / Get zero crossing rate
     */
    public double getZeroCrossingRate() {
        return zeroCrossingRate;
    }
    
    /**
     * 获取能量 / Get energy
     */
    public double getEnergy() {
        return energy;
    }
    
    /**
     * 获取幅度包络 / Get amplitude envelope
     */
    public double[] getAmplitudeEnvelope() {
        return amplitudeEnvelope.clone();
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
        sb.append("TimeDomainFeatureResult{");
        sb.append("rmsEnergy=").append(String.format("%.3f", rmsEnergy));
        sb.append(", zeroCrossingRate=").append(String.format("%.3f", zeroCrossingRate));
        sb.append(", energy=").append(String.format("%.3f", energy));
        sb.append(", confidence=").append(String.format("%.3f", confidence));
        if (amplitudeEnvelope.length > 0) {
            sb.append(", envelopeLength=").append(amplitudeEnvelope.length);
        }
        sb.append('}');
        return sb.toString();
    }
}