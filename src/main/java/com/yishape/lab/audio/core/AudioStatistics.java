package com.yishape.lab.audio.core;

import com.yishape.lab.math.linalg.IVector;

/**
 * 音频统计信息类 / Audio Statistics Class
 * <p>
 * 提供音频数据的统计分析功能，包括基本统计量、动态范围、信噪比等。
 * 使用项目现有的Stats类进行统计分析。
 * </p>
 * <p>
 * Provides statistical analysis functionality for audio data, including basic statistics,
 * dynamic range, signal-to-noise ratio, etc. Uses existing Stats class for statistical analysis.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class AudioStatistics {
    
    /** 音频样本数据 / Audio sample data */
    private final IVector<Double> samples;
    
    /** 基本统计量 / Basic statistics */
    private final double mean;
    private final double stdDev;
    private final double variance;
    private final double min;
    private final double max;
    private final double range;
    private final double rms;
    private final double peak;
    
    /** 动态范围(dB) / Dynamic range (dB) */
    private final double dynamicRange;
    
    /** 峰值因子 / Crest factor */
    private final double crestFactor;
    
    /** 信噪比估计(dB) / Signal-to-noise ratio estimate (dB) */
    private final double snr;
    
    /** 零交叉率 / Zero crossing rate */
    private final double zeroCrossingRate;
    
    /**
     * 构造函数 / Constructor
     *
     * @param samples 音频样本数据 / Audio sample data
     */
    public AudioStatistics(IVector<Double> samples) {
        this.samples = samples;
        
        // 计算基本统计量 / Calculate basic statistics
        this.mean = samples.mean();
        this.variance = samples.var();
        this.stdDev = Math.sqrt(variance);
        this.min = samples.min();
        this.max = samples.max();
        this.range = max - min;
        
        // 计算RMS / Calculate RMS
        this.rms = calculateRMS();
        
        // 计算峰值 / Calculate peak
        this.peak = Math.max(Math.abs(min), Math.abs(max));
        
        // 计算动态范围 / Calculate dynamic range
        this.dynamicRange = calculateDynamicRange();
        
        // 计算峰值因子 / Calculate crest factor
        this.crestFactor = peak / rms;
        
        // 计算信噪比估计 / Calculate SNR estimate
        this.snr = calculateSNR();
        
        // 计算零交叉率 / Calculate zero crossing rate
        this.zeroCrossingRate = calculateZeroCrossingRate();
    }
    
    /**
     * 计算RMS值 / Calculate RMS value
     *
     * @return RMS值 / RMS value
     */
    private double calculateRMS() {
        double sumSquares = 0.0;
        for (int i = 0; i < samples.length(); i++) {
            double sample = samples.get(i);
            sumSquares += sample * sample;
        }
        return Math.sqrt(sumSquares / samples.length());
    }
    
    /**
     * 计算动态范围 / Calculate dynamic range
     *
     * @return 动态范围(dB) / Dynamic range (dB)
     */
    private double calculateDynamicRange() {
        if (rms == 0) {
            return 0;
        }
        return 20 * Math.log10(peak / rms);
    }
    
    /**
     * 计算信噪比估计 / Calculate SNR estimate
     * <p>
     * 使用统计方法估计信噪比，假设噪声为高斯分布。
     * Uses statistical method to estimate SNR, assuming noise is Gaussian distributed.
     * </p>
     *
     * @return 信噪比估计(dB) / SNR estimate (dB)
     */
    private double calculateSNR() {
        if (stdDev == 0) {
            return Double.POSITIVE_INFINITY;
        }
        
        // 使用信号功率与噪声功率的比值估计SNR
        // Estimate SNR using ratio of signal power to noise power
        double signalPower = rms * rms;
        double noisePower = variance;
        
        if (noisePower == 0) {
            return Double.POSITIVE_INFINITY;
        }
        
        return 10 * Math.log10(signalPower / noisePower);
    }
    
    /**
     * 计算零交叉率 / Calculate zero crossing rate
     *
     * @return 零交叉率 / Zero crossing rate
     */
    private double calculateZeroCrossingRate() {
        int crossings = 0;
        for (int i = 1; i < samples.length(); i++) {
            if ((samples.get(i) >= 0) != (samples.get(i - 1) >= 0)) {
                crossings++;
            }
        }
        return (double) crossings / (samples.length() - 1);
    }
    
    /**
     * 获取均值 / Get mean
     *
     * @return 均值 / Mean
     */
    public double getMean() {
        return mean;
    }
    
    /**
     * 获取标准差 / Get standard deviation
     *
     * @return 标准差 / Standard deviation
     */
    public double getStdDev() {
        return stdDev;
    }
    
    /**
     * 获取方差 / Get variance
     *
     * @return 方差 / Variance
     */
    public double getVariance() {
        return variance;
    }
    
    /**
     * 获取最小值 / Get minimum value
     *
     * @return 最小值 / Minimum value
     */
    public double getMin() {
        return min;
    }
    
    /**
     * 获取最大值 / Get maximum value
     *
     * @return 最大值 / Maximum value
     */
    public double getMax() {
        return max;
    }
    
    /**
     * 获取范围 / Get range
     *
     * @return 范围 / Range
     */
    public double getRange() {
        return range;
    }
    
    /**
     * 获取RMS值 / Get RMS value
     *
     * @return RMS值 / RMS value
     */
    public double getRms() {
        return rms;
    }
    
    /**
     * 获取峰值 / Get peak value
     *
     * @return 峰值 / Peak value
     */
    public double getPeak() {
        return peak;
    }
    
    /**
     * 获取动态范围 / Get dynamic range
     *
     * @return 动态范围(dB) / Dynamic range (dB)
     */
    public double getDynamicRange() {
        return dynamicRange;
    }
    
    /**
     * 获取峰值因子 / Get crest factor
     *
     * @return 峰值因子 / Crest factor
     */
    public double getCrestFactor() {
        return crestFactor;
    }
    
    /**
     * 获取信噪比估计 / Get SNR estimate
     *
     * @return 信噪比估计(dB) / SNR estimate (dB)
     */
    public double getSnr() {
        return snr;
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
     * 检查音频是否过载 / Check if audio is clipped
     *
     * @return 如果音频过载返回true / True if audio is clipped
     */
    public boolean isClipped() {
        return Math.abs(peak) >= 0.99; // 接近满量程 / Near full scale
    }
    
    /**
     * 检查音频是否静音 / Check if audio is silent
     *
     * @return 如果音频静音返回true / True if audio is silent
     */
    public boolean isSilent() {
        return rms < 1e-6; // 非常小的RMS值 / Very small RMS value
    }
    
    /**
     * 获取音频质量评估 / Get audio quality assessment
     *
     * @return 音频质量等级 / Audio quality level
     */
    public AudioQuality getQuality() {
        if (isSilent()) {
            return AudioQuality.SILENT;
        } else if (isClipped()) {
            return AudioQuality.CLIPPED;
        } else if (snr < 20) {
            return AudioQuality.POOR;
        } else if (snr < 40) {
            return AudioQuality.FAIR;
        } else if (snr < 60) {
            return AudioQuality.GOOD;
        } else {
            return AudioQuality.EXCELLENT;
        }
    }
    
    @Override
    public String toString() {
        return String.format(
            "AudioStatistics{mean=%.4f, stdDev=%.4f, min=%.4f, max=%.4f, rms=%.4f, " +
            "dynamicRange=%.2fdB, snr=%.2fdB, zcr=%.4f, quality=%s}",
            mean, stdDev, min, max, rms, dynamicRange, snr, zeroCrossingRate, getQuality()
        );
    }
}