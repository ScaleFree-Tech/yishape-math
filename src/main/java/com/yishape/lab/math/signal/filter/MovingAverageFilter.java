package com.yishape.lab.math.signal.filter;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.signal.core.AbstractSignalProcessor;
import com.yishape.lab.math.signal.core.SignalProcessingException;

/**
 * 移动平均滤波器实现类 / Moving Average Filter Implementation Class
 * <p>
 * 实现移动平均滤波器，这是一种简单的FIR低通滤波器。
 * 移动平均滤波器通过对滑动窗口内的样本求平均来平滑信号。
 * </p>
 * <p>
 * Implements moving average filter, a simple FIR low-pass filter.
 * Moving average filter smooths signals by averaging samples within a sliding window.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class MovingAverageFilter extends AbstractSignalProcessor<Double> implements ISignalFilter<Double> {
    
    private FilterType filterType;
    private FilterImplementation implementationType;
    private int windowSize; // 窗口大小 / Window size
    private FilterCoefficients coefficients;
    
    /**
     * 构造函数 / Constructor
     * <p>
     * 创建移动平均滤波器。
     * Create moving average filter.
     * </p>
     *
     * @param windowSize 窗口大小 / Window size
     * @throws SignalProcessingException 参数无效时抛出 / Thrown when parameters are invalid
     */
    public MovingAverageFilter(int windowSize) throws SignalProcessingException {
        super("Moving Average Filter", "1.0.0");
        
        validateParameters(windowSize);
        
        this.filterType = FilterType.LOW_PASS;
        this.implementationType = FilterImplementation.MOVING_AVERAGE;
        this.windowSize = windowSize;
        
        computeCoefficients();
    }
    
    /**
     * 验证参数 / Validate parameters
     */
    private void validateParameters(int windowSize) throws SignalProcessingException {
        if (windowSize <= 0) {
            throw new SignalProcessingException("窗口大小必须大于0 / Window size must be greater than 0");
        }
    }
    
    /**
     * 计算移动平均滤波器系数 / Compute moving average filter coefficients
     */
    private void computeCoefficients() throws SignalProcessingException {
        try {
            // 移动平均滤波器系数为常数 / Moving average filter coefficients are constant
            double[] kernel = new double[windowSize];
            double weight = 1.0 / windowSize;
            
            for (int i = 0; i < windowSize; i++) {
                kernel[i] = weight;
            }
            
            // 移动平均滤波器是FIR滤波器 / Moving average filter is FIR filter
            coefficients = new FilterCoefficients(kernel);
        } catch (Exception e) {
            throw new SignalProcessingException("计算滤波器系数失败 / Failed to compute filter coefficients", e);
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
            
            // 移动平均滤波 / Moving average filtering
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                int count = 0;
                
                // 计算窗口内的平均值 / Calculate average within window
                for (int j = Math.max(0, i - windowSize + 1); j <= i; j++) {
                    sum += signal.get(j);
                    count++;
                }
                
                output.set(i, sum / count);
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
        return windowSize - 1; // FIR滤波器的阶数 / Order of FIR filter
    }

    /**
     * 获取截止频率 / Get cutoff frequencies
     * @return 截止频率数组 / Cutoff frequency array
     */
    @Override
    public double[] getCutoffFrequencies() {
        // 移动平均滤波器的等效截止频率 / Equivalent cutoff frequency of moving average filter
        // 近似为 0.44π / windowSize / Approximately 0.44π / windowSize
        double cutoffFreq = 0.44 * Math.PI / windowSize;
        return new double[]{cutoffFreq};
    }

    /**
     * 设置截止频率 / Set cutoff frequencies
     * @param frequencies 截止频率数组 / Cutoff frequency array
     * @throws SignalProcessingException 频率设置无效时抛出 / Thrown when frequency setting is invalid
     */
    @Override
    public void setCutoffFrequencies(double... frequencies) throws SignalProcessingException {
        if (frequencies.length != 1) {
            throw new SignalProcessingException("移动平均滤波器只需要一个截止频率 / Moving average filter requires only one cutoff frequency");
        }
        // 根据截止频率反推窗口大小 / Infer window size from cutoff frequency
        double cutoffFreq = frequencies[0];
        this.windowSize = (int) Math.round(0.44 * Math.PI / cutoffFreq);
        if (windowSize <= 0) windowSize = 1;
        computeCoefficients(); // 重新计算系数 / Recalculate coefficients
    }
    
    @Override
    public double getSamplingRate() {
        // 移动平均滤波器不直接使用采样率 / Moving average filter doesn't directly use sampling rate
        return 1.0;
    }

    /**
     * 设置采样率 / Set sampling rate
     * @param samplingRate 采样率 / Sampling rate (未使用 / Not used)
     * @throws SignalProcessingException 参数无效时抛出 / Thrown when parameters are invalid
     */
    @Override
    public void setSamplingRate(double samplingRate) throws SignalProcessingException {
        // 移动平均滤波器不直接使用采样率 / Moving average filter doesn't directly use sampling rate
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
        // 计算移动平均滤波器的频率响应 / Calculate frequency response of moving average filter
        double[] magnitude = new double[frequencies.length];
        double[] phase = new double[frequencies.length];
        
        for (int i = 0; i < frequencies.length; i++) {
            double omega = 2 * Math.PI * frequencies[i];
            // 移动平均滤波器的频率响应 / Frequency response of moving average filter
            if (Math.abs(omega) < 1e-10) {
                magnitude[i] = 1.0;
            } else {
                magnitude[i] = Math.abs(Math.sin(omega * windowSize / 2) / (windowSize * Math.sin(omega / 2)));
            }
            phase[i] = -omega * (windowSize - 1) / 2;
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
        computeCoefficients(); // 重新计算系数 / Recalculate coefficients
    }
    
    @Override
    protected IVector<Double> doProcess(IVector<Double> input) throws SignalProcessingException {
        return filter(input);
    }

    /**
     * 克隆移动平均滤波器 / Clone Moving Average filter
     * @return 移动平均滤波器副本 / Moving Average filter copy
     */
    @Override
    public MovingAverageFilter clone() {
        try {
            return new MovingAverageFilter(windowSize);
        } catch (SignalProcessingException e) {
            // 这不应该发生，因为我们已经验证了参数 / This should not happen since we've validated parameters
            throw new RuntimeException("克隆移动平均滤波器失败 / Failed to clone moving average filter", e);
        }
    }
}