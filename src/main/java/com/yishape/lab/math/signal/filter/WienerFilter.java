package com.yishape.lab.math.signal.filter;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.signal.core.AbstractSignalProcessor;
import com.yishape.lab.math.signal.core.SignalProcessingException;

/**
 * 维纳滤波器实现类 / Wiener Filter Implementation Class
 * <p>
 * 实现维纳滤波器，用于在已知信号和噪声统计特性的情况下进行最优滤波。
 * 维纳滤波器是一种线性滤波器，在最小均方误差准则下提供最优估计。
 * </p>
 * <p>
 * Implements Wiener filter for optimal filtering when signal and noise statistical characteristics are known.
 * Wiener filter is a linear filter that provides optimal estimation under minimum mean square error criterion.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class WienerFilter extends AbstractSignalProcessor<Double> implements ISignalFilter<Double> {
    
    private ISignalFilter.FilterType filterType;
    private ISignalFilter.FilterImplementation implementationType;
    private double signalPower;  // 信号功率 / Signal power
    private double noisePower;  // 噪声功率 / Noise power
    private int filterLength;  // 滤波器长度 / Filter length
    private double samplingRate; // 采样率 / Sampling rate
    
    /**
     * 构造函数 / Constructor
     * <p>
     * 创建维纳滤波器。
     * Create Wiener filter.
     * </p>
     *
     * @param signalPower 信号功率 / Signal power
     * @param noisePower 噪声功率 / Noise power
     * @param filterLength 滤波器长度 / Filter length
     * @throws SignalProcessingException 参数无效时抛出 / Thrown when parameters are invalid
     */
    public WienerFilter(double signalPower, double noisePower, int filterLength) throws SignalProcessingException {
        super("Wiener Filter", "1.0.0");
        
        validateParameters(signalPower, noisePower, filterLength);
        
        this.filterType = ISignalFilter.FilterType.ADAPTIVE;
        this.implementationType = ISignalFilter.FilterImplementation.FIR;
        this.signalPower = signalPower;
        this.noisePower = noisePower;
        this.filterLength = filterLength;
        this.samplingRate = 1000.0; // Default sampling rate
    }
    
    /**
     * 验证参数 / Validate parameters
     */
    private void validateParameters(double signalPower, double noisePower, int filterLength) throws SignalProcessingException {
        if (signalPower < 0) {
            throw new SignalProcessingException("信号功率必须大于等于0 / Signal power must be greater than or equal to 0");
        }
        if (noisePower <= 0) {
            throw new SignalProcessingException("噪声功率必须大于0 / Noise power must be greater than 0");
        }
        if (filterLength <= 0) {
            throw new SignalProcessingException("滤波器长度必须大于0 / Filter length must be greater than 0");
        }
    }
    
    /**
     * 滤波信号 / Filter signal
     * <p>
     * 对输入信号进行维纳滤波处理。
     * Apply Wiener filtering to input signal.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @return 滤波后的信号 / Filtered signal
     * @throws SignalProcessingException 滤波过程中发生错误时抛出 / Thrown when errors occur during filtering
     */
    @Override
    public IVector<Double> filter(IVector<Double> signal) throws SignalProcessingException {
        if (signal == null || signal.length() == 0) {
            throw new SignalProcessingException("输入信号不能为空 / Input signal cannot be empty");
        }
        
        try {
            int n = signal.length();
            IVector<Double> output = Linalg.zeros(n);
            
            // 简化的维纳滤波实现 / Simplified Wiener filter implementation
            // 计算信噪比 / Calculate signal-to-noise ratio
            double snr = signalPower / noisePower;
            
            // 对于简化实现，使用移动平均作为近似 / For simplified implementation, use moving average as approximation
            int windowSize = Math.min(filterLength, n);
            double[] weights = new double[windowSize];
            
            // 计算维纳滤波权重 / Calculate Wiener filter weights
            for (int i = 0; i < windowSize; i++) {
                // 简化的权重计算 / Simplified weight calculation
                weights[i] = 1.0 / windowSize * (snr / (snr + 1.0));
            }
            
            // 应用滤波 / Apply filtering
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                int count = 0;
                for (int j = 0; j < windowSize && (i - j) >= 0; j++) {
                    sum += signal.get(i - j) * weights[j];
                    count++;
                }
                output.set(i, sum);
            }
            
            return output;
        } catch (Exception e) {
            throw new SignalProcessingException("维纳滤波处理失败 / Wiener filtering failed", e);
        }
    }
    
    @Override
    public ISignalFilter.FilterType getFilterType() {
        return filterType;
    }
    
    @Override
    public ISignalFilter.FilterImplementation getImplementationType() {
        return implementationType;
    }
    
    @Override
    public int getOrder() {
        return filterLength;  // 维纳滤波器的阶数等于滤波器长度 / Wiener filter order equals filter length
    }
    
    @Override
    public double[] getCutoffFrequencies() {
        return new double[0];  // 维纳滤波器没有固定的截止频率 / Wiener filter doesn't have fixed cutoff frequencies
    }
    
    @Override
    public void setCutoffFrequencies(double... frequencies) throws SignalProcessingException {
        // 维纳滤波器不需要设置截止频率 / Wiener filter doesn't need cutoff frequencies
    }
    
    @Override
    public double getSamplingRate() {
        return samplingRate;
    }
    
    @Override
    public void setSamplingRate(double samplingRate) throws SignalProcessingException {
        if (samplingRate <= 0) {
            throw new SignalProcessingException("采样率必须大于0 / Sampling rate must be greater than 0");
        }
        this.samplingRate = samplingRate;
    }
    
    @Override
    public ISignalFilter.FilterCoefficients getCoefficients() {
        // 生成滤波器系数 / Generate filter coefficients
        double[] coefficients = new double[filterLength];
        double snr = signalPower / noisePower;
        for (int i = 0; i < filterLength; i++) {
            coefficients[i] = 1.0 / filterLength * (snr / (snr + 1.0));
        }
        return new ISignalFilter.FilterCoefficients(coefficients);
    }
    
    @Override
    public ISignalFilter.FrequencyResponse getFrequencyResponse(double[] frequencies) throws SignalProcessingException {
        // 维纳滤波器的频率响应 / Wiener filter frequency response
        double[] magnitude = new double[frequencies.length];
        double[] phase = new double[frequencies.length];
        double snr = signalPower / noisePower;
        
        for (int i = 0; i < frequencies.length; i++) {
            // 简化的频率响应计算 / Simplified frequency response calculation
            magnitude[i] = snr / (snr + 1.0);
            phase[i] = 0.0;
        }
        return new ISignalFilter.FrequencyResponse(frequencies, magnitude, phase);
    }
    
    /**
     * 获取信号功率 / Get signal power
     */
    public double getSignalPower() {
        return signalPower;
    }
    
    /**
     * 设置信号功率 / Set signal power
     */
    public void setSignalPower(double signalPower) throws SignalProcessingException {
        validateParameters(signalPower, noisePower, filterLength);
        this.signalPower = signalPower;
    }
    
    /**
     * 获取噪声功率 / Get noise power
     */
    public double getNoisePower() {
        return noisePower;
    }
    
    /**
     * 设置噪声功率 / Set noise power
     */
    public void setNoisePower(double noisePower) throws SignalProcessingException {
        validateParameters(signalPower, noisePower, filterLength);
        this.noisePower = noisePower;
    }
    
    /**
     * 获取滤波器长度 / Get filter length
     */
    public int getFilterLength() {
        return filterLength;
    }
    
    /**
     * 设置滤波器长度 / Set filter length
     */
    public void setFilterLength(int filterLength) throws SignalProcessingException {
        validateParameters(signalPower, noisePower, filterLength);
        this.filterLength = filterLength;
    }
    
    @Override
    protected IVector<Double> doProcess(IVector<Double> input) throws SignalProcessingException {
        return filter(input);
    }
    
    @Override
    public WienerFilter clone() {
        try {
            return new WienerFilter(signalPower, noisePower, filterLength);
        } catch (SignalProcessingException e) {
            // This should not happen as we're cloning valid parameters
            throw new RuntimeException("克隆维纳滤波器失败 / Failed to clone Wiener filter", e);
        }
    }
    
    @Override
    public String toString() {
        return String.format("WienerFilter{signalPower=%f, noisePower=%f, filterLength=%d}", 
                           signalPower, noisePower, filterLength);
    }
}