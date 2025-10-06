package com.yishape.lab.audio.core;

import com.yishape.lab.math.linalg.IVector;

/**
 * 噪声配置文件类 / Noise Profile Class
 * <p>
 * 存储噪声的统计特性，用于音频降噪处理。
 * Stores noise statistical properties for audio noise reduction processing.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class NoiseProfile {
    
    /** 噪声均值 / Noise mean */
    private final double mean;
    
    /** 噪声标准差 / Noise standard deviation */
    private final double stdDev;
    
    /** 噪声功率 / Noise power */
    private final double power;
    
    /** 噪声方差 / Noise variance */
    private final double variance;
    
    /** 噪声RMS / Noise RMS */
    private final double rms;
    
    /** 噪声频谱特征 / Noise spectral characteristics */
    private final double[] spectralProfile;
    
    /**
     * 构造函数 / Constructor
     *
     * @param mean 噪声均值 / Noise mean
     * @param stdDev 噪声标准差 / Noise standard deviation
     */
    public NoiseProfile(double mean, double stdDev) {
        this(mean, stdDev, null);
    }
    
    /**
     * 构造函数（包含频谱特征）/ Constructor (with spectral characteristics)
     *
     * @param mean 噪声均值 / Noise mean
     * @param stdDev 噪声标准差 / Noise standard deviation
     * @param spectralProfile 噪声频谱特征 / Noise spectral characteristics
     */
    public NoiseProfile(double mean, double stdDev, double[] spectralProfile) {
        this.mean = mean;
        this.stdDev = stdDev;
        this.variance = stdDev * stdDev;
        this.power = mean * mean + variance;
        this.rms = Math.sqrt(power);
        this.spectralProfile = spectralProfile != null ? spectralProfile.clone() : null;
    }
    
    /**
     * 从音频样本创建噪声配置文件 / Create noise profile from audio samples
     *
     * @param noiseSamples 噪声样本 / Noise samples
     * @return 噪声配置文件 / Noise profile
     */
    public static NoiseProfile fromSamples(IVector<Double> noiseSamples) {
        double mean = noiseSamples.mean();
        double variance = noiseSamples.var();
        double stdDev = Math.sqrt(variance);
        
        return new NoiseProfile(mean, stdDev);
    }
    
    /**
     * 从音频样本创建噪声配置文件（包含频谱分析）/ Create noise profile from audio samples (with spectral analysis)
     *
     * @param noiseSamples 噪声样本 / Noise samples
     * @param fftSize FFT大小 / FFT size
     * @return 噪声配置文件 / Noise profile
     */
    public static NoiseProfile fromSamplesWithSpectrum(IVector<Double> noiseSamples, int fftSize) {
        double mean = noiseSamples.mean();
        double variance = noiseSamples.var();
        double stdDev = Math.sqrt(variance);
        
        // 简化的频谱分析（实际应用中需要FFT）
        // Simplified spectral analysis (actual implementation would need FFT)
        double[] spectralProfile = new double[fftSize / 2];
        for (int i = 0; i < spectralProfile.length; i++) {
            spectralProfile[i] = stdDev; // 简化假设噪声在所有频率上均匀分布
        }
        
        return new NoiseProfile(mean, stdDev, spectralProfile);
    }
    
    /**
     * 获取噪声均值 / Get noise mean
     *
     * @return 噪声均值 / Noise mean
     */
    public double getMean() {
        return mean;
    }
    
    /**
     * 获取噪声标准差 / Get noise standard deviation
     *
     * @return 噪声标准差 / Noise standard deviation
     */
    public double getStdDev() {
        return stdDev;
    }
    
    /**
     * 获取噪声方差 / Get noise variance
     *
     * @return 噪声方差 / Noise variance
     */
    public double getVariance() {
        return variance;
    }
    
    /**
     * 获取噪声功率 / Get noise power
     *
     * @return 噪声功率 / Noise power
     */
    public double getPower() {
        return power;
    }
    
    /**
     * 获取噪声RMS / Get noise RMS
     *
     * @return 噪声RMS / Noise RMS
     */
    public double getRms() {
        return rms;
    }
    
    /**
     * 获取噪声频谱特征 / Get noise spectral characteristics
     *
     * @return 噪声频谱特征 / Noise spectral characteristics
     */
    public double[] getSpectralProfile() {
        return spectralProfile != null ? spectralProfile.clone() : null;
    }
    
    /**
     * 检查是否有频谱特征 / Check if has spectral characteristics
     *
     * @return 如果有频谱特征返回true / True if has spectral characteristics
     */
    public boolean hasSpectralProfile() {
        return spectralProfile != null;
    }
    
    /**
     * 获取噪声阈值 / Get noise threshold
     * <p>
     * 基于统计特性计算噪声阈值，用于信号检测。
     * Calculate noise threshold based on statistical properties for signal detection.
     * </p>
     *
     * @param confidenceLevel 置信水平 (默认2.0，即95%置信) / Confidence level (default 2.0, i.e., 95% confidence)
     * @return 噪声阈值 / Noise threshold
     */
    public double getThreshold(double confidenceLevel) {
        return mean + confidenceLevel * stdDev;
    }
    
    /**
     * 获取噪声阈值（默认置信水平）/ Get noise threshold (default confidence level)
     *
     * @return 噪声阈值 / Noise threshold
     */
    public double getThreshold() {
        return getThreshold(2.0);
    }
    
    /**
     * 检查信号是否为噪声 / Check if signal is noise
     *
     * @param signalLevel 信号电平 / Signal level
     * @param confidenceLevel 置信水平 / Confidence level
     * @return 如果是噪声返回true / True if noise
     */
    public boolean isNoise(double signalLevel, double confidenceLevel) {
        return Math.abs(signalLevel - mean) < confidenceLevel * stdDev;
    }
    
    /**
     * 检查信号是否为噪声（默认置信水平）/ Check if signal is noise (default confidence level)
     *
     * @param signalLevel 信号电平 / Signal level
     * @return 如果是噪声返回true / True if noise
     */
    public boolean isNoise(double signalLevel) {
        return isNoise(signalLevel, 2.0);
    }
    
    /**
     * 计算信噪比 / Calculate signal-to-noise ratio
     *
     * @param signalPower 信号功率 / Signal power
     * @return 信噪比(dB) / Signal-to-noise ratio (dB)
     */
    public double calculateSNR(double signalPower) {
        if (power == 0) {
            return Double.POSITIVE_INFINITY;
        }
        return 10 * Math.log10(signalPower / power);
    }
    
    /**
     * 获取噪声抑制因子 / Get noise suppression factor
     *
     * @param signalLevel 信号电平 / Signal level
     * @return 抑制因子 (0-1) / Suppression factor (0-1)
     */
    public double getSuppressionFactor(double signalLevel) {
        double snr = calculateSNR(signalLevel * signalLevel);
        if (snr < 0) {
            return 0.1; // 强抑制 / Strong suppression
        } else if (snr < 6) {
            return 0.3; // 中等抑制 / Moderate suppression
        } else if (snr < 12) {
            return 0.6; // 轻微抑制 / Light suppression
        } else {
            return 1.0; // 无抑制 / No suppression
        }
    }
    
    /**
     * 合并两个噪声配置文件 / Merge two noise profiles
     *
     * @param other 另一个噪声配置文件 / Another noise profile
     * @param weight 权重 (0-1) / Weight (0-1)
     * @return 合并后的噪声配置文件 / Merged noise profile
     */
    public NoiseProfile merge(NoiseProfile other, double weight) {
        double newMean = this.mean * (1 - weight) + other.mean * weight;
        double newVariance = this.variance * (1 - weight) + other.variance * weight;
        double newStdDev = Math.sqrt(newVariance);
        
        double[] newSpectralProfile = null;
        if (this.spectralProfile != null && other.spectralProfile != null) {
            newSpectralProfile = new double[this.spectralProfile.length];
            for (int i = 0; i < newSpectralProfile.length; i++) {
                newSpectralProfile[i] = this.spectralProfile[i] * (1 - weight) + 
                                      other.spectralProfile[i] * weight;
            }
        }
        
        return new NoiseProfile(newMean, newStdDev, newSpectralProfile);
    }
    
    @Override
    public String toString() {
        return String.format("NoiseProfile{mean=%.6f, stdDev=%.6f, power=%.6f, rms=%.6f, hasSpectrum=%s}",
                            mean, stdDev, power, rms, hasSpectralProfile());
    }
}