package com.yishape.lab.math.signal.filter;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.signal.core.AbstractSignalProcessor;
import com.yishape.lab.math.signal.core.SignalProcessingException;
import java.util.Arrays;

/**
 * 中值滤波器实现类 / Median Filter Implementation Class
 * <p>
 * 实现中值滤波器，这是一种非线性滤波器，特别适用于去除脉冲噪声。
 * 中值滤波器通过对滑动窗口内的样本取中值来平滑信号，能有效保持边缘特征。
 * </p>
 * <p>
 * Implements median filter, a nonlinear filter particularly suitable for removing impulse noise.
 * Median filter smooths signals by taking median of samples within a sliding window, effectively preserving edge features.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class MedianFilter extends AbstractSignalProcessor<Double> implements ISignalFilter<Double> {
    
    private FilterType filterType;
    private FilterImplementation implementationType;
    private int windowSize; // 窗口大小 / Window size
    private FilterCoefficients coefficients;
    
    /**
     * 构造函数 / Constructor
     * <p>
     * 创建中值滤波器。
     * Create median filter.
     * </p>
     *
     * @param windowSize 窗口大小 / Window size
     * @throws SignalProcessingException 参数无效时抛出 / Thrown when parameters are invalid
     */
    public MedianFilter(int windowSize) throws SignalProcessingException {
        super("Median Filter", "1.0.0");
        
        validateParameters(windowSize);
        
        this.filterType = FilterType.LOW_PASS; // 中值滤波器通常用作低通滤波器 / Median filter typically used as low-pass filter
        this.implementationType = FilterImplementation.MEDIAN;
        this.windowSize = windowSize;
        
        // 中值滤波器是非线性的，没有传统意义上的系数 / Median filter is nonlinear, has no traditional coefficients
        this.coefficients = new FilterCoefficients(new double[]{1.0});
    }
    
    /**
     * 验证参数 / Validate parameters
     */
    private void validateParameters(int windowSize) throws SignalProcessingException {
        if (windowSize <= 0) {
            throw new SignalProcessingException("窗口大小必须大于0 / Window size must be greater than 0");
        }
        if (windowSize % 2 == 0) {
            throw new SignalProcessingException("窗口大小必须为奇数 / Window size must be odd");
        }
    }
    
    /**
     * 滤波信号 / Filter signal
     * <p>
     * 对输入信号进行滤波处理。
     * Apply filtering to input signal.
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
            
            // 中值滤波 / Median filtering
            for (int i = 0; i < n; i++) {
                // 创建窗口数组 / Create window array
                int windowRadius = windowSize / 2;
                int actualWindowSize = Math.min(windowSize, Math.min(i + windowRadius + 1, n) - Math.max(0, i - windowRadius));
                double[] window = new double[actualWindowSize];
                
                // 填充窗口 / Fill window
                int index = 0;
                for (int j = Math.max(0, i - windowRadius); j <= Math.min(n - 1, i + windowRadius); j++) {
                    window[index++] = signal.get(j);
                }
                
                // 计算中值 / Calculate median
                Arrays.sort(window);
                double median = window[actualWindowSize / 2];
                
                output.set(i, median);
            }
            
            return output;
        } catch (Exception e) {
            throw new SignalProcessingException("滤波处理失败 / Filtering failed", e);
        }
    }
    
    @Override
    public FilterType getFilterType() {
        return filterType;
    }

    /**
     * 获取滤波器实现类型 / Get filter implementation type
     * @return 滤波器实现类型 / Filter implementation type
     */
    @Override
    public FilterImplementation getImplementationType() {
        return implementationType;
    }

    /**
     * 获取滤波器阶数 / Get filter order
     * @return 滤波器阶数 / Filter order
     */
    @Override
    public int getOrder() {
        return windowSize - 1; // 返回窗口大小减1作为阶数 / Return window size minus 1 as order
    }

    /**
     * 获取截止频率 / Get cutoff frequencies
     * @return 截止频率数组 / Cutoff frequency array
     */
    @Override
    public double[] getCutoffFrequencies() {
        // 中值滤波器没有明确定义的截止频率 / Median filter has no clearly defined cutoff frequency
        return new double[]{0.0};
    }

    /**
     * 设置截止频率 / Set cutoff frequencies
     * @param frequencies 截止频率数组 / Cutoff frequency array
     * @throws SignalProcessingException 频率设置无效时抛出 / Thrown when frequency setting is invalid
     */
    @Override
    public void setCutoffFrequencies(double... frequencies) throws SignalProcessingException {
        // 中值滤波器不支持通过截止频率设置参数 / Median filter doesn't support setting parameters via cutoff frequency
        throw new SignalProcessingException("中值滤波器不支持通过截止频率设置参数 / Median filter doesn't support setting parameters via cutoff frequency");
    }
    
    @Override
    public double getSamplingRate() {
        // 中值滤波器不直接使用采样率 / Median filter doesn't directly use sampling rate
        return 1.0;
    }

    /**
     * 设置采样率 / Set sampling rate
     * @param samplingRate 采样率 / Sampling rate (未使用 / Not used)
     * @throws SignalProcessingException 参数无效时抛出 / Thrown when parameters are invalid
     */
    @Override
    public void setSamplingRate(double samplingRate) throws SignalProcessingException {
        // 中值滤波器不直接使用采样率 / Median filter doesn't directly use sampling rate
    }

    /**
     * 获取滤波器系数 / Get filter coefficients
     * @return 滤波器系数 / Filter coefficients
     */
    @Override
    public FilterCoefficients getCoefficients() {
        return coefficients;
    }

    /**
     * 计算滤波器的频率响应 / Calculate frequency response of filter
     * @param frequencies 频率点数组 / Frequency point array
     * @return 频率响应 / Frequency response
     * @throws SignalProcessingException 计算过程中发生错误时抛出 / Thrown when errors occur during calculation
     */
    @Override
    public FrequencyResponse getFrequencyResponse(double[] frequencies) throws SignalProcessingException {
        // 中值滤波器是非线性的，没有传统的频率响应 / Median filter is nonlinear, has no traditional frequency response
        double[] magnitude = new double[frequencies.length];
        double[] phase = new double[frequencies.length];
        
        // 返回单位响应作为近似 / Return unit response as approximation
        for (int i = 0; i < frequencies.length; i++) {
            magnitude[i] = 1.0;
            phase[i] = 0.0;
        }
        
        return new FrequencyResponse(frequencies, magnitude, phase);
    }
    
    /**
     * 获取窗口大小 / Get window size
     * @return 窗口大小 / Window size
     */
    public int getWindowSize() {
        return windowSize;
    }

    /**
     * 设置窗口大小 / Set window size
     * @param windowSize 窗口大小 / Window size
     * @throws SignalProcessingException 参数无效时抛出 / Thrown when parameters are invalid
     */
    public void setWindowSize(int windowSize) throws SignalProcessingException {
        validateParameters(windowSize);
        this.windowSize = windowSize;
    }
    
    @Override
    protected IVector<Double> doProcess(IVector<Double> input) throws SignalProcessingException {
        return filter(input);
    }

    /**
     * 克隆中值滤波器 / Clone Median filter
     * @return 中值滤波器副本 / Median filter copy
     */
    @Override
    public MedianFilter clone() {
        try {
            return new MedianFilter(windowSize);
        } catch (SignalProcessingException e) {
            // 这不应该发生，因为我们已经验证了参数 / This should not happen since we've validated parameters
            throw new RuntimeException("克隆中值滤波器失败 / Failed to clone median filter", e);
        }
    }
}