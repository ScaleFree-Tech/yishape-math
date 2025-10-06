package com.yishape.lab.audio.preprocessing;

/**
 * 音频预处理选项 / Audio Preprocessing Options
 * <p>
 * 配置音频预处理的各种选项和参数。
 * Configure various options and parameters for audio preprocessing.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class AudioPreprocessingOptions {
    
    /** 是否标准化采样率 / Whether to normalize sample rate */
    private boolean normalizeSampleRate = true;
    
    /** 目标采样率 / Target sample rate */
    private double targetSampleRate = 44100.0;
    
    /** 是否标准化音量 / Whether to normalize volume */
    private boolean normalizeVolume = true;
    
    /** 归一化目标值 / Normalization target value */
    private double normalizationTarget = 0.95;
    
    /** 是否应用降噪 / Whether to apply noise reduction */
    private boolean applyNoiseReduction = false;
    
    /** 降噪阈值 / Noise reduction threshold */
    private double noiseThreshold = 0.01;
    
    /** 是否转换为单声道 / Whether to convert to mono */
    private boolean convertToMono = false;
    
    /** 是否去除直流分量 / Whether to remove DC offset */
    private boolean removeDCOffset = true;
    
    /** 是否应用预加重滤波器 / Whether to apply pre-emphasis filter */
    private boolean applyPreEmphasis = false;
    
    /** 预加重系数 / Pre-emphasis coefficient */
    private double preEmphasisCoefficient = 0.97;
    
    /**
     * 默认构造函数 / Default constructor
     */
    public AudioPreprocessingOptions() {
    }
    
    /**
     * 获取默认选项 / Get default options
     */
    public static AudioPreprocessingOptions getDefault() {
        return new AudioPreprocessingOptions();
    }
    
    /**
     * 获取高质量选项 / Get high quality options
     */
    public static AudioPreprocessingOptions getHighQuality() {
        AudioPreprocessingOptions options = new AudioPreprocessingOptions();
        options.setNormalizeSampleRate(true);
        options.setNormalizeVolume(true);
        options.setApplyNoiseReduction(true);
        options.setRemoveDCOffset(true);
        options.setApplyPreEmphasis(true);
        return options;
    }
    
    /**
     * 获取快速处理选项 / Get fast processing options
     */
    public static AudioPreprocessingOptions getFast() {
        AudioPreprocessingOptions options = new AudioPreprocessingOptions();
        options.setNormalizeSampleRate(false);
        options.setNormalizeVolume(true);
        options.setApplyNoiseReduction(false);
        options.setRemoveDCOffset(false);
        options.setApplyPreEmphasis(false);
        return options;
    }
    
    // Getters and setters
    
    public boolean isNormalizeSampleRate() {
        return normalizeSampleRate;
    }
    
    public void setNormalizeSampleRate(boolean normalizeSampleRate) {
        this.normalizeSampleRate = normalizeSampleRate;
    }
    
    public double getTargetSampleRate() {
        return targetSampleRate;
    }
    
    public void setTargetSampleRate(double targetSampleRate) {
        this.targetSampleRate = targetSampleRate;
    }
    
    public boolean isNormalizeVolume() {
        return normalizeVolume;
    }
    
    public void setNormalizeVolume(boolean normalizeVolume) {
        this.normalizeVolume = normalizeVolume;
    }
    
    public double getNormalizationTarget() {
        return normalizationTarget;
    }
    
    public void setNormalizationTarget(double normalizationTarget) {
        this.normalizationTarget = Math.max(0.1, Math.min(1.0, normalizationTarget));
    }
    
    public boolean isApplyNoiseReduction() {
        return applyNoiseReduction;
    }
    
    public void setApplyNoiseReduction(boolean applyNoiseReduction) {
        this.applyNoiseReduction = applyNoiseReduction;
    }
    
    public double getNoiseThreshold() {
        return noiseThreshold;
    }
    
    public void setNoiseThreshold(double noiseThreshold) {
        this.noiseThreshold = Math.max(0.001, Math.min(0.1, noiseThreshold));
    }
    
    public boolean isConvertToMono() {
        return convertToMono;
    }
    
    public void setConvertToMono(boolean convertToMono) {
        this.convertToMono = convertToMono;
    }
    
    public boolean isRemoveDCOffset() {
        return removeDCOffset;
    }
    
    public void setRemoveDCOffset(boolean removeDCOffset) {
        this.removeDCOffset = removeDCOffset;
    }
    
    public boolean isApplyPreEmphasis() {
        return applyPreEmphasis;
    }
    
    public void setApplyPreEmphasis(boolean applyPreEmphasis) {
        this.applyPreEmphasis = applyPreEmphasis;
    }
    
    public double getPreEmphasisCoefficient() {
        return preEmphasisCoefficient;
    }
    
    public void setPreEmphasisCoefficient(double preEmphasisCoefficient) {
        this.preEmphasisCoefficient = Math.max(0.9, Math.min(0.99, preEmphasisCoefficient));
    }
}