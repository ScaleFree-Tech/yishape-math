package com.reremouse.lab.audio;

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
    
    /**
     * 构造函数 / Constructor
     *
     * @param mean 噪声均值 / Noise mean
     * @param stdDev 噪声标准差 / Noise standard deviation
     */
    public NoiseProfile(double mean, double stdDev) {
        this.mean = mean;
        this.stdDev = stdDev;
        this.power = mean * mean + stdDev * stdDev;
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
     * 获取噪声功率 / Get noise power
     *
     * @return 噪声功率 / Noise power
     */
    public double getPower() {
        return power;
    }
    
    /**
     * 获取噪声阈值 / Get noise threshold
     * <p>
     * 基于统计特性计算噪声阈值，用于信号检测。
     * Calculate noise threshold based on statistical properties for signal detection.
     * </p>
     *
     * @param confidenceLevel 置信水平 (默认2.0，即95%置信度) / Confidence level (default 2.0, i.e., 95% confidence)
     * @return 噪声阈值 / Noise threshold
     */
    public double getThreshold(double confidenceLevel) {
        return mean + confidenceLevel * stdDev;
    }
    
    /**
     * 获取噪声阈值（默认置信水平） / Get noise threshold (default confidence level)
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
     * 检查信号是否为噪声（默认置信水平） / Check if signal is noise (default confidence level)
     *
     * @param signalLevel 信号电平 / Signal level
     * @return 如果是噪声返回true / True if noise
     */
    public boolean isNoise(double signalLevel) {
        return isNoise(signalLevel, 2.0);
    }
    
    @Override
    public String toString() {
        return String.format("NoiseProfile{mean=%.6f, stdDev=%.6f, power=%.6f}", 
                           mean, stdDev, power);
    }
}
