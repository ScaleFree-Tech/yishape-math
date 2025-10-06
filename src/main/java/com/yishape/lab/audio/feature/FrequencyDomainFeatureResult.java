package com.yishape.lab.audio.feature;

/**
 * 频域特征结果 / Frequency-Domain Feature Result
 * <p>
 * 封装音频的频域相关特征数据，如频谱质心、频谱带宽等。
 * Encapsulates frequency-domain feature data of audio, such as spectral centroid, spectral bandwidth, etc.
 * </p>
 *
 * @author RereMouse
 * @version 2.0
 * @since 2.0
 */
public class FrequencyDomainFeatureResult {
    
    private final double spectralCentroid;         // 频谱质心 (Hz)
    private final double spectralBandwidth;        // 频谱带宽 (Hz)
    private final double spectralRolloff;          // 频谱滚降点 (Hz)
    private final double[] spectralContrast;       // 频谱对比度
    private final double spectralFlatness;         // 频谱平坦度
    private final double spectralFlux;             // 频谱通量
    private final double confidence;               // 置信度
    
    /**
     * 构造函数 / Constructor
     */
    public FrequencyDomainFeatureResult(double spectralCentroid, double spectralBandwidth, 
                                     double spectralRolloff, double[] spectralContrast,
                                     double spectralFlatness, double spectralFlux, 
                                     double confidence) {
        this.spectralCentroid = spectralCentroid;
        this.spectralBandwidth = spectralBandwidth;
        this.spectralRolloff = spectralRolloff;
        this.spectralContrast = spectralContrast != null ? spectralContrast.clone() : new double[0];
        this.spectralFlatness = spectralFlatness;
        this.spectralFlux = spectralFlux;
        this.confidence = confidence;
    }
    
    /**
     * 获取频谱质心 / Get spectral centroid
     */
    public double getSpectralCentroid() {
        return spectralCentroid;
    }
    
    /**
     * 获取频谱带宽 / Get spectral bandwidth
     */
    public double getSpectralBandwidth() {
        return spectralBandwidth;
    }
    
    /**
     * 获取频谱滚降点 / Get spectral rolloff
     */
    public double getSpectralRolloff() {
        return spectralRolloff;
    }
    
    /**
     * 获取频谱对比度 / Get spectral contrast
     */
    public double[] getSpectralContrast() {
        return spectralContrast.clone();
    }
    
    /**
     * 获取频谱平坦度 / Get spectral flatness
     */
    public double getSpectralFlatness() {
        return spectralFlatness;
    }
    
    /**
     * 获取频谱通量 / Get spectral flux
     */
    public double getSpectralFlux() {
        return spectralFlux;
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
        sb.append("FrequencyDomainFeatureResult{");
        sb.append("spectralCentroid=").append(String.format("%.2fHz", spectralCentroid));
        sb.append(", spectralBandwidth=").append(String.format("%.2fHz", spectralBandwidth));
        sb.append(", spectralRolloff=").append(String.format("%.2fHz", spectralRolloff));
        sb.append(", spectralFlatness=").append(String.format("%.3f", spectralFlatness));
        sb.append(", spectralFlux=").append(String.format("%.3f", spectralFlux));
        sb.append(", confidence=").append(String.format("%.3f", confidence));
        if (spectralContrast.length > 0) {
            sb.append(", contrastLength=").append(spectralContrast.length);
        }
        sb.append('}');
        return sb.toString();
    }
}