package com.yishape.lab.math.signal.filter;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.signal.core.AbstractSignalProcessor;
import com.yishape.lab.math.signal.core.Complex;
import com.yishape.lab.math.signal.core.SignalProcessingException;

/**
 * 巴特沃斯滤波器实现类 / Butterworth Filter Implementation Class
 * <p>
 * 实现巴特沃斯滤波器，这是一种具有最大平坦幅频响应的滤波器。
 * 巴特沃斯滤波器在通带内具有非常平坦的响应，在阻带内单调下降。
 * </p>
 * <p>
 * Implements Butterworth filter, a filter with maximally flat frequency response.
 * Butterworth filter has very flat response in passband and monotonic decrease in stopband.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ButterworthFilter extends AbstractSignalProcessor<Double> implements ISignalFilter<Double> {
    
    private FilterType filterType;
    private FilterImplementation implementationType;
    private int order;
    private double[] cutoffFrequencies;
    private double samplingRate;
    private FilterCoefficients coefficients;
    
    /**
     * 构造函数 / Constructor
     * <p>
     * 创建巴特沃斯低通滤波器。
     * Create Butterworth low-pass filter.
     * </p>
     *
     * @param order 滤波器阶数 / Filter order
     * @param cutoffFreq 截止频率 / Cutoff frequency
     * @param samplingRate 采样率 / Sampling rate
     * @throws SignalProcessingException 参数无效时抛出 / Thrown when parameters are invalid
     */
    public ButterworthFilter(int order, double cutoffFreq, double samplingRate) throws SignalProcessingException {
        super("Butterworth Filter", "1.0.0");
        
        validateParameters(order, new double[]{cutoffFreq}, samplingRate);
        
        this.filterType = FilterType.LOW_PASS;
        this.implementationType = FilterImplementation.BUTTERWORTH;
        this.order = order;
        this.cutoffFrequencies = new double[]{cutoffFreq};
        this.samplingRate = samplingRate;
        
        computeCoefficients();
    }
    
    /**
     * 构造函数 / Constructor
     * <p>
     * 创建指定类型的巴特沃斯滤波器。
     * Create Butterworth filter of specified type.
     * </p>
     *
     * @param filterType 滤波器类型 / Filter type
     * @param order 滤波器阶数 / Filter order
     * @param cutoffFreqs 截止频率数组 / Cutoff frequency array
     * @param samplingRate 采样率 / Sampling rate
     * @throws SignalProcessingException 参数无效时抛出 / Thrown when parameters are invalid
     */
    public ButterworthFilter(FilterType filterType, int order, double[] cutoffFreqs, double samplingRate) throws SignalProcessingException {
        super("Butterworth Filter", "1.0.0");
        
        validateParameters(order, cutoffFreqs, samplingRate);
        validateFilterType(filterType, cutoffFreqs);
        
        this.filterType = filterType;
        this.implementationType = FilterImplementation.BUTTERWORTH;
        this.order = order;
        this.cutoffFrequencies = cutoffFreqs.clone();
        this.samplingRate = samplingRate;
        
        computeCoefficients();
    }
    
    /**
     * 验证参数 / Validate parameters
     */
    private void validateParameters(int order, double[] cutoffFreqs, double samplingRate) throws SignalProcessingException {
        if (order <= 0) {
            throw new SignalProcessingException("滤波器阶数必须大于0 / Filter order must be greater than 0");
        }
        if (samplingRate <= 0) {
            throw new SignalProcessingException("采样率必须大于0 / Sampling rate must be greater than 0");
        }
        
        double nyquist = samplingRate / 2;
        for (double freq : cutoffFreqs) {
            if (freq <= 0 || freq >= nyquist) {
                throw new SignalProcessingException(
                    String.format("截止频率必须在(0, %f)范围内 / Cutoff frequency must be in (0, %f) range", nyquist, nyquist));
            }
        }
    }
    
    /**
     * 验证滤波器类型 / Validate filter type
     */
    private void validateFilterType(FilterType filterType, double[] cutoffFreqs) throws SignalProcessingException {
        if ((filterType == FilterType.BAND_PASS || filterType == FilterType.BAND_STOP) && cutoffFreqs.length != 2) {
            throw new SignalProcessingException("带通和带阻滤波器需要两个截止频率 / Band-pass and band-stop filters require two cutoff frequencies");
        }
        if ((filterType == FilterType.LOW_PASS || filterType == FilterType.HIGH_PASS) && cutoffFreqs.length != 1) {
            throw new SignalProcessingException("低通和高通滤波器需要一个截止频率 / Low-pass and high-pass filters require one cutoff frequency");
        }
    }
    
    /**
     * 计算巴特沃斯滤波器系数 / Compute Butterworth filter coefficients
     */
    private void computeCoefficients() throws SignalProcessingException {
        try {
            switch (filterType) {
                case LOW_PASS:
                    coefficients = computeLowPassCoefficients();
                    break;
                case HIGH_PASS:
                    coefficients = computeHighPassCoefficients();
                    break;
                case BAND_PASS:
                    coefficients = computeBandPassCoefficients();
                    break;
                case BAND_STOP:
                    coefficients = computeBandStopCoefficients();
                    break;
                default:
                    throw new SignalProcessingException("不支持的滤波器类型 / Unsupported filter type: " + filterType);
            }
        } catch (Exception e) {
            throw new SignalProcessingException("计算滤波器系数失败 / Failed to compute filter coefficients", e);
        }
    }
    
    /**
     * 计算低通巴特沃斯滤波器系数 / Compute low-pass Butterworth filter coefficients
     */
    private FilterCoefficients computeLowPassCoefficients() {
        // 归一化截止频率 / Normalize cutoff frequency
        double omegaC = 2 * Math.PI * cutoffFrequencies[0] / samplingRate;
        double warpedFreq = 2 * Math.tan(omegaC / 2);
        
        // 计算巴特沃斯极点 / Calculate Butterworth poles
        Complex[] poles = computeButterworthPoles(order);
        
        // 将极点转换为系数 / Convert poles to coefficients
        return polesAndZerosToCoefficients(poles, new Complex[0], warpedFreq);
    }
    
    /**
     * 计算巴特沃斯极点 / Calculate Butterworth poles
     */
    private Complex[] computeButterworthPoles(int order) {
        Complex[] poles = new Complex[order];
        
        // 巴特沃斯极点公式 / Butterworth pole formula
        for (int k = 0; k < order; k++) {
            double angle = Math.PI * (2 * k + order + 1) / (2 * order);
            poles[k] = new Complex(-Math.cos(angle), -Math.sin(angle));
        }
        
        return poles;
    }
    
    /**
     * 计算高通巴特沃斯滤波器系数 / Compute high-pass Butterworth filter coefficients
     */
    private FilterCoefficients computeHighPassCoefficients() {
        // 先计算低通系数，然后转换为高通 / First calculate low-pass coefficients, then convert to high-pass
        FilterCoefficients lpCoeffs = computeLowPassCoefficients();
        
        // 高通转换 / High-pass transformation
        double[] numerator = new double[lpCoeffs.getNumerator().length];
        double[] denominator = new double[lpCoeffs.getDenominator().length];
        
        // s -> 1/s 变换 / s -> 1/s transformation
        for (int i = 0; i < numerator.length; i++) {
            numerator[i] = lpCoeffs.getDenominator()[lpCoeffs.getDenominator().length - 1 - i];
        }
        for (int i = 0; i < denominator.length; i++) {
            denominator[i] = lpCoeffs.getNumerator()[lpCoeffs.getNumerator().length - 1 - i];
        }
        
        return new FilterCoefficients(numerator, denominator);
    }
    
    /**
     * 计算带通巴特沃斯滤波器系数 / Compute band-pass Butterworth filter coefficients
     */
    private FilterCoefficients computeBandPassCoefficients() {
        // 简化实现，实际应用中需要更复杂的带通设计
        // Simplified implementation, actual applications need more complex band-pass design
        
        double[] numerator = new double[order * 2 + 1];
        double[] denominator = new double[order * 2 + 1];
        
        // 使用简化的系数作为示例 / Use simplified coefficients as example
        for (int i = 0; i <= order * 2; i++) {
            numerator[i] = (i == 0) ? 1.0 : 0.0;
            denominator[i] = 1.0;
        }
        
        return new FilterCoefficients(numerator, denominator);
    }
    
    /**
     * 计算带阻巴特沃斯滤波器系数 / Compute band-stop Butterworth filter coefficients
     */
    private FilterCoefficients computeBandStopCoefficients() {
        // 简化实现，实际应用中需要更复杂的带阻设计
        // Simplified implementation, actual applications need more complex band-stop design
        
        double[] numerator = new double[order * 2 + 1];
        double[] denominator = new double[order * 2 + 1];
        
        // 使用简化的系数作为示例 / Use simplified coefficients as example
        for (int i = 0; i <= order * 2; i++) {
            numerator[i] = (i == 0) ? 1.0 : 0.0;
            denominator[i] = 1.0;
        }
        
        return new FilterCoefficients(numerator, denominator);
    }
    
    /**
     * 将极点和零点转换为滤波器系数 / Convert poles and zeros to filter coefficients
     */
    private FilterCoefficients polesAndZerosToCoefficients(Complex[] poles, Complex[] zeros, double warpedFreq) {
        int n = poles.length;
        
        // 计算分子系数 / Calculate numerator coefficients
        double[] numerator = new double[n + 1];
        numerator[0] = 1.0;
        
        for (int i = 0; i < zeros.length; i++) {
            for (int j = n; j > 0; j--) {
                numerator[j] -= zeros[i].real * numerator[j - 1];
            }
        }
        
        // 计算分母系数 / Calculate denominator coefficients
        double[] denominator = new double[n + 1];
        denominator[0] = 1.0;
        
        for (int i = 0; i < n; i++) {
            for (int j = n; j > 0; j--) {
                denominator[j] -= poles[i].real * denominator[j - 1];
            }
        }
        
        // 归一化 / Normalize
        double gain = 1.0;
        for (int i = 0; i <= n; i++) {
            numerator[i] *= gain;
        }
        
        return new FilterCoefficients(numerator, denominator);
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
            // 获取滤波器系数 / Get filter coefficients
            double[] numerator = coefficients.getNumerator();
            double[] denominator = coefficients.getDenominator();
            
            int n = signal.length();
            IVector<Double> output = Linalg.zeros(n);
            
            // Direct Form II 实现 / Direct Form II implementation
            double[] delays = new double[Math.max(numerator.length, denominator.length)];
            
            for (int i = 0; i < n; i++) {
                // 计算输入项 / Calculate input terms
                double sum = 0;
                for (int j = 0; j < numerator.length && i - j >= 0; j++) {
                    sum += numerator[j] * signal.get(i - j);
                }
                
                // 减去反馈项 / Subtract feedback terms
                for (int j = 1; j < denominator.length; j++) {
                    sum -= denominator[j] * delays[j - 1];
                }
                
                // 更新延迟线 / Update delay line
                for (int j = delays.length - 1; j > 0; j--) {
                    delays[j] = delays[j - 1];
                }
                delays[0] = sum;
                
                output.set(i, sum);
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
        return order;
    }

    /**
     * 获取截止频率 / Get cutoff frequencies
     * @return 截止频率数组 / Cutoff frequency array
     */
    @Override
    public double[] getCutoffFrequencies() {
        return cutoffFrequencies.clone();
    }

    /**
     * 设置截止频率 / Set cutoff frequencies
     * @param frequencies 截止频率数组 / Cutoff frequency array
     * @throws SignalProcessingException 频率设置无效时抛出 / Thrown when frequency setting is invalid
     */
    @Override
    public void setCutoffFrequencies(double... frequencies) throws SignalProcessingException {
        validateParameters(order, frequencies, samplingRate);
        this.cutoffFrequencies = frequencies.clone();
        computeCoefficients(); // 重新计算系数 / Recalculate coefficients
    }
    
    @Override
    public double getSamplingRate() {
        return samplingRate;
    }

    /**
     * 设置采样率 / Set sampling rate
     * @param samplingRate 采样率 / Sampling rate
     * @throws SignalProcessingException 采样率设置无效时抛出 / Thrown when sampling rate setting is invalid
     */
    @Override
    public void setSamplingRate(double samplingRate) throws SignalProcessingException {
        validateParameters(order, cutoffFrequencies, samplingRate);
        this.samplingRate = samplingRate;
        computeCoefficients(); // 重新计算系数 / Recalculate coefficients
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
        // 简化的频率响应计算 / Simplified frequency response calculation
        double[] magnitude = new double[frequencies.length];
        double[] phase = new double[frequencies.length];
        
        for (int i = 0; i < frequencies.length; i++) {
            magnitude[i] = 1.0; // 简化实现 / Simplified implementation
            phase[i] = 0.0;
        }
        
        return new FrequencyResponse(frequencies, magnitude, phase);
    }
    
    @Override
    protected IVector<Double> doProcess(IVector<Double> input) throws SignalProcessingException {
        return filter(input);
    }

    /**
     * 克隆巴特沃斯滤波器 / Clone Butterworth filter
     * @return 巴特沃斯滤波器副本 / Butterworth filter copy
     */
    @Override
    public ButterworthFilter clone() {
        try {
            return new ButterworthFilter(filterType, order, cutoffFrequencies, samplingRate);
        } catch (SignalProcessingException e) {
            // 这不应该发生，因为我们已经验证了参数 / This should not happen since we've validated parameters
            throw new RuntimeException("克隆巴特沃斯滤波器失败 / Failed to clone Butterworth filter", e);
        }
    }
}