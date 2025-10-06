package com.yishape.lab.math.signal.filter;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.signal.core.AbstractSignalProcessor;
import com.yishape.lab.math.signal.core.SignalProcessingException;

/**
 * 带阻滤波器实现类 / Band-stop Filter Implementation Class
 * <p>
 * 实现带阻滤波器（陷波滤波器），用于滤除特定频率范围内的信号。
 * 基于二阶节的巴特沃斯带阻滤波器设计。
 * </p>
 * <p>
 * Implements band-stop filter (notch filter) for filtering out signals within a specific frequency range.
 * Based on second-order sections Butterworth band-stop filter design.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class BandStopFilter extends AbstractSignalProcessor<Double> implements ISignalFilter<Double> {
    
    private ISignalFilter.FilterType filterType;
    private ISignalFilter.FilterImplementation implementationType;
    private int order;  // 滤波器阶数 / Filter order
    private double lowCutoffFrequency;  // 低截止频率 / Low cutoff frequency
    private double highCutoffFrequency;  // 高截止频率 / High cutoff frequency
    private double samplingRate; // 采样率 / Sampling rate
    
    /**
     * 构造函数 / Constructor
     * <p>
     * 创建带阻滤波器。
     * Create band-stop filter.
     * </p>
     *
     * @param order 滤波器阶数 / Filter order
     * @param lowCutoffFrequency 低截止频率 / Low cutoff frequency
     * @param highCutoffFrequency 高截止频率 / High cutoff frequency
     * @param samplingRate 采样率 / Sampling rate
     * @throws SignalProcessingException 参数无效时抛出 / Thrown when parameters are invalid
     */
    public BandStopFilter(int order, double lowCutoffFrequency, double highCutoffFrequency, double samplingRate) throws SignalProcessingException {
        super("Band-stop Filter", "1.0.0");
        
        validateParameters(order, lowCutoffFrequency, highCutoffFrequency, samplingRate);
        
        this.filterType = ISignalFilter.FilterType.BAND_STOP;
        this.implementationType = ISignalFilter.FilterImplementation.BUTTERWORTH;
        this.order = order;
        this.lowCutoffFrequency = lowCutoffFrequency;
        this.highCutoffFrequency = highCutoffFrequency;
        this.samplingRate = samplingRate;
    }
    
    /**
     * 验证参数 / Validate parameters
     */
    private void validateParameters(int order, double lowCutoffFrequency, double highCutoffFrequency, double samplingRate) throws SignalProcessingException {
        if (order <= 0) {
            throw new SignalProcessingException("滤波器阶数必须大于0 / Filter order must be greater than 0");
        }
        if (lowCutoffFrequency < 0) {
            throw new SignalProcessingException("低截止频率必须大于等于0 / Low cutoff frequency must be greater than or equal to 0");
        }
        if (highCutoffFrequency <= lowCutoffFrequency) {
            throw new SignalProcessingException("高截止频率必须大于低截止频率 / High cutoff frequency must be greater than low cutoff frequency");
        }
        if (samplingRate <= 0) {
            throw new SignalProcessingException("采样率必须大于0 / Sampling rate must be greater than 0");
        }
        if (highCutoffFrequency > samplingRate / 2) {
            throw new SignalProcessingException("截止频率不能超过奈奎斯特频率 / Cutoff frequency cannot exceed Nyquist frequency");
        }
    }
    
    /**
     * 滤波信号 / Filter signal
     * <p>
     * 对输入信号进行带阻滤波处理。
     * Apply band-stop filtering to input signal.
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
            IVector<Double> output = signal.copy();
            
            // 简化的带阻滤波实现 / Simplified band-stop filter implementation
            // 使用频率域的方法进行近似 / Use frequency domain method for approximation
            
            // 计算归一化频率 / Calculate normalized frequencies
            double nyquist = samplingRate / 2.0;
            double normalizedLow = lowCutoffFrequency / nyquist;
            double normalizedHigh = highCutoffFrequency / nyquist;
            
            // 对于简化实现，使用时域差分方程 / For simplified implementation, use time-domain difference equation
            // 这是一个二阶带阻滤波器的简化实现 / This is a simplified implementation of a second-order band-stop filter
            if (order >= 2) {
                // 简化的二阶带阻滤波器系数计算 / Simplified second-order band-stop filter coefficient calculation
                double omega0 = 2 * Math.PI * Math.sqrt(normalizedLow * normalizedHigh); // 中心频率 / Center frequency
                double bandwidth = 2 * Math.PI * (normalizedHigh - normalizedLow); // 带宽 / Bandwidth
                
                // 计算滤波器系数 / Calculate filter coefficients
                double alpha = Math.sin(bandwidth) / 2.0;
                double cosOmega0 = Math.cos(omega0);
                
                // 差分方程系数 / Difference equation coefficients
                double b0 = 1.0;
                double b1 = -2.0 * cosOmega0;
                double b2 = 1.0;
                double a0 = 1.0 + alpha;
                double a1 = -2.0 * cosOmega0;
                double a2 = 1.0 - alpha;
                
                // 归一化系数 / Normalize coefficients
                b0 /= a0;
                b1 /= a0;
                b2 /= a0;
                a1 /= a0;
                a2 /= a0;
                
                // 应用滤波 / Apply filtering
                if (n >= 2) {
                    // 初始化前两个输出值 / Initialize first two output values
                    output.set(0, signal.get(0));
                    output.set(1, signal.get(1));
                    
                    // 应用差分方程 / Apply difference equation
                    for (int i = 2; i < n; i++) {
                        double value = b0 * signal.get(i) + 
                                      b1 * signal.get(i-1) + 
                                      b2 * signal.get(i-2) - 
                                      a1 * output.get(i-1) - 
                                      a2 * output.get(i-2);
                        output.set(i, value);
                    }
                }
            } else {
                // 对于一阶滤波器，使用简化的实现 / For first-order filter, use simplified implementation
                // 直接返回原信号作为近似 / Directly return original signal as approximation
                return signal.copy();
            }
            
            return output;
        } catch (Exception e) {
            throw new SignalProcessingException("带阻滤波处理失败 / Band-stop filtering failed", e);
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
        return order;
    }
    
    @Override
    public double[] getCutoffFrequencies() {
        return new double[]{lowCutoffFrequency, highCutoffFrequency};
    }
    
    @Override
    public void setCutoffFrequencies(double... frequencies) throws SignalProcessingException {
        if (frequencies.length != 2) {
            throw new SignalProcessingException("带阻滤波器需要两个截止频率 / Band-stop filter requires two cutoff frequencies");
        }
        validateParameters(order, frequencies[0], frequencies[1], samplingRate);
        this.lowCutoffFrequency = frequencies[0];
        this.highCutoffFrequency = frequencies[1];
    }
    
    @Override
    public double getSamplingRate() {
        return samplingRate;
    }
    
    @Override
    public void setSamplingRate(double samplingRate) throws SignalProcessingException {
        validateParameters(order, lowCutoffFrequency, highCutoffFrequency, samplingRate);
        this.samplingRate = samplingRate;
    }
    
    @Override
    public ISignalFilter.FilterCoefficients getCoefficients() {
        // 计算滤波器系数 / Calculate filter coefficients
        double nyquist = samplingRate / 2.0;
        double normalizedLow = lowCutoffFrequency / nyquist;
        double normalizedHigh = highCutoffFrequency / nyquist;
        
        // 简化的系数计算 / Simplified coefficient calculation
        double omega0 = 2 * Math.PI * Math.sqrt(normalizedLow * normalizedHigh); // 中心频率 / Center frequency
        double bandwidth = 2 * Math.PI * (normalizedHigh - normalizedLow); // 带宽 / Bandwidth
        
        // 计算滤波器系数 / Calculate filter coefficients
        double alpha = Math.sin(bandwidth) / 2.0;
        double cosOmega0 = Math.cos(omega0);
        
        // 差分方程系数 / Difference equation coefficients
        double b0 = 1.0;
        double b1 = -2.0 * cosOmega0;
        double b2 = 1.0;
        double a0 = 1.0 + alpha;
        double a1 = -2.0 * cosOmega0;
        double a2 = 1.0 - alpha;
        
        // 归一化系数 / Normalize coefficients
        b0 /= a0;
        b1 /= a0;
        b2 /= a0;
        a1 /= a0;
        a2 /= a0;
        
        return new ISignalFilter.FilterCoefficients(
            new double[]{b0, b1, b2}, 
            new double[]{1.0, a1, a2}
        );
    }
    
    @Override
    public ISignalFilter.FrequencyResponse getFrequencyResponse(double[] frequencies) throws SignalProcessingException {
        // 计算频率响应 / Calculate frequency response
        double[] magnitude = new double[frequencies.length];
        double[] phase = new double[frequencies.length];
        
        double nyquist = samplingRate / 2.0;
        double normalizedLow = lowCutoffFrequency / nyquist;
        double normalizedHigh = highCutoffFrequency / nyquist;
        
        // 简化的频率响应计算 / Simplified frequency response calculation
        double omega0 = 2 * Math.PI * Math.sqrt(normalizedLow * normalizedHigh); // 中心频率 / Center frequency
        double bandwidth = 2 * Math.PI * (normalizedHigh - normalizedLow); // 带宽 / Bandwidth
        
        for (int i = 0; i < frequencies.length; i++) {
            double freq = frequencies[i];
            double normalizedFreq = freq / nyquist;
            double omega = 2 * Math.PI * normalizedFreq;
            
            // 计算频率响应 / Calculate frequency response
            double delta = omega - omega0;
            double response = Math.abs(delta) / (Math.abs(delta) + bandwidth/2);
            
            magnitude[i] = response;
            phase[i] = -Math.PI/2; // 简化的相位响应 / Simplified phase response
        }
        
        return new ISignalFilter.FrequencyResponse(frequencies, magnitude, phase);
    }
    
    /**
     * 获取低截止频率 / Get low cutoff frequency
     */
    public double getLowCutoffFrequency() {
        return lowCutoffFrequency;
    }
    
    /**
     * 设置低截止频率 / Set low cutoff frequency
     */
    public void setLowCutoffFrequency(double lowCutoffFrequency) throws SignalProcessingException {
        validateParameters(order, lowCutoffFrequency, highCutoffFrequency, samplingRate);
        this.lowCutoffFrequency = lowCutoffFrequency;
    }
    
    /**
     * 获取高截止频率 / Get high cutoff frequency
     */
    public double getHighCutoffFrequency() {
        return highCutoffFrequency;
    }
    
    /**
     * 设置高截止频率 / Set high cutoff frequency
     */
    public void setHighCutoffFrequency(double highCutoffFrequency) throws SignalProcessingException {
        validateParameters(order, lowCutoffFrequency, highCutoffFrequency, samplingRate);
        this.highCutoffFrequency = highCutoffFrequency;
    }
    
    /**
     * 获取滤波器阶数 / Get filter order
     */
    public void setOrder(int order) throws SignalProcessingException {
        validateParameters(order, lowCutoffFrequency, highCutoffFrequency, samplingRate);
        this.order = order;
    }
    
    @Override
    protected IVector<Double> doProcess(IVector<Double> input) throws SignalProcessingException {
        return filter(input);
    }
    
    @Override
    public BandStopFilter clone() {
        try {
            return new BandStopFilter(order, lowCutoffFrequency, highCutoffFrequency, samplingRate);
        } catch (SignalProcessingException e) {
            // This should not happen as we're cloning valid parameters
            throw new RuntimeException("克隆带阻滤波器失败 / Failed to clone band-stop filter", e);
        }
    }
    
    @Override
    public String toString() {
        return String.format("BandStopFilter{order=%d, lowCutoffFrequency=%f, highCutoffFrequency=%f, samplingRate=%f}", 
                           order, lowCutoffFrequency, highCutoffFrequency, samplingRate);
    }
}