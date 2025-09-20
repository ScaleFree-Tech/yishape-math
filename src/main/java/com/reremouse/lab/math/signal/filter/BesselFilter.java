package com.reremouse.lab.math.signal.filter;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.signal.core.AbstractSignalProcessor;
import com.reremouse.lab.math.signal.core.SignalProcessingException;
import com.reremouse.lab.math.signal.Complex;

/**
 * 贝塞尔滤波器实现类 / Bessel Filter Implementation Class
 * <p>
 * 实现贝塞尔滤波器，这是一种具有最大平坦群延迟响应的滤波器。
 * 贝塞尔滤波器在通带内具有非常平坦的群延迟，在时域应用中能保持信号波形。
 * </p>
 * <p>
 * Implements Bessel filter, a filter with maximally flat group delay response.
 * Bessel filter has very flat group delay in passband and preserves signal waveform in time-domain applications.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class BesselFilter extends AbstractSignalProcessor<Double> implements ISignalFilter<Double> {
    
    private FilterType filterType;
    private FilterImplementation implementationType;
    private int order;
    private double[] cutoffFrequencies;
    private double samplingRate;
    private FilterCoefficients coefficients;
    
    /**
     * 构造函数 / Constructor
     * <p>
     * 创建贝塞尔低通滤波器。
     * Create Bessel low-pass filter.
     * </p>
     *
     * @param order 滤波器阶数 / Filter order
     * @param cutoffFreq 截止频率 / Cutoff frequency
     * @param samplingRate 采样率 / Sampling rate
     * @throws SignalProcessingException 参数无效时抛出 / Thrown when parameters are invalid
     */
    public BesselFilter(int order, double cutoffFreq, double samplingRate) throws SignalProcessingException {
        super("Bessel Filter", "1.0.0");
        
        validateParameters(order, new double[]{cutoffFreq}, samplingRate);
        
        this.filterType = FilterType.LOW_PASS;
        this.implementationType = FilterImplementation.BESSEL;
        this.order = order;
        this.cutoffFrequencies = new double[]{cutoffFreq};
        this.samplingRate = samplingRate;
        
        computeCoefficients();
    }
    
    /**
     * 构造函数 / Constructor
     * <p>
     * 创建指定类型的贝塞尔滤波器。
     * Create Bessel filter of specified type.
     * </p>
     *
     * @param filterType 滤波器类型 / Filter type
     * @param order 滤波器阶数 / Filter order
     * @param cutoffFreqs 截止频率数组 / Cutoff frequency array
     * @param samplingRate 采样率 / Sampling rate
     * @throws SignalProcessingException 参数无效时抛出 / Thrown when parameters are invalid
     */
    public BesselFilter(FilterType filterType, int order, double[] cutoffFreqs, double samplingRate) throws SignalProcessingException {
        super("Bessel Filter", "1.0.0");
        
        validateParameters(order, cutoffFreqs, samplingRate);
        validateFilterType(filterType, cutoffFreqs);
        
        this.filterType = filterType;
        this.implementationType = FilterImplementation.BESSEL;
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
     * 计算贝塞尔滤波器系数 / Compute Bessel filter coefficients
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
     * 计算低通贝塞尔滤波器系数 / Compute low-pass Bessel filter coefficients
     */
    private FilterCoefficients computeLowPassCoefficients() {
        // 归一化截止频率 / Normalize cutoff frequency
        double omegaC = 2 * Math.PI * cutoffFrequencies[0] / samplingRate;
        double warpedFreq = 2 * Math.tan(omegaC / 2);
        
        // 计算贝塞尔多项式系数 / Calculate Bessel polynomial coefficients
        double[] besselCoeffs = computeBesselPolynomial(order);
        
        // 构造滤波器系数 / Construct filter coefficients
        double[] numerator = new double[order + 1];
        double[] denominator = new double[order + 1];
        
        // 分子系数为常数 / Numerator coefficients are constant
        numerator[0] = besselCoeffs[0];
        for (int i = 1; i <= order; i++) {
            numerator[i] = 0.0;
        }
        
        // 分母系数为贝塞尔多项式 / Denominator coefficients are Bessel polynomial
        System.arraycopy(besselCoeffs, 0, denominator, 0, order + 1);
        
        return new FilterCoefficients(numerator, denominator);
    }
    
    /**
     * 计算贝塞尔多项式系数 / Calculate Bessel polynomial coefficients
     */
    private double[] computeBesselPolynomial(int order) {
        double[] coeffs = new double[order + 1];
        
        // 贝塞尔多项式递推公式 / Bessel polynomial recurrence formula
        // B_0(x) = 1
        // B_1(x) = x + 1
        // B_n(x) = (2n-1)x*B_{n-1}(x) + B_{n-2}(x)
        
        if (order >= 0) {
            coeffs[0] = 1.0;
        }
        if (order >= 1) {
            coeffs[0] = 1.0;
            coeffs[1] = 1.0;
        }
        
        for (int n = 2; n <= order; n++) {
            double[] prev = coeffs.clone();
            coeffs[0] = prev[0];
            for (int i = 1; i <= n; i++) {
                coeffs[i] = (2 * n - 1) * prev[i - 1];
                if (i <= n - 2) {
                    coeffs[i] += prev[i];
                }
            }
        }
        
        return coeffs;
    }
    
    /**
     * 计算高通贝塞尔滤波器系数 / Compute high-pass Bessel filter coefficients
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
     * 计算带通贝塞尔滤波器系数 / Compute band-pass Bessel filter coefficients
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
     * 计算带阻贝塞尔滤波器系数 / Compute band-stop Bessel filter coefficients
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
    
    @Override
    public FilterImplementation getImplementationType() {
        return implementationType;
    }
    
    @Override
    public int getOrder() {
        return order;
    }
    
    @Override
    public double[] getCutoffFrequencies() {
        return cutoffFrequencies.clone();
    }
    
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
    
    @Override
    public void setSamplingRate(double samplingRate) throws SignalProcessingException {
        validateParameters(order, cutoffFrequencies, samplingRate);
        this.samplingRate = samplingRate;
        computeCoefficients(); // 重新计算系数 / Recalculate coefficients
    }
    
    @Override
    public FilterCoefficients getCoefficients() {
        return coefficients;
    }
    
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
    
    @Override
    public BesselFilter clone() {
        try {
            return new BesselFilter(filterType, order, cutoffFrequencies, samplingRate);
        } catch (SignalProcessingException e) {
            // 这不应该发生，因为我们已经验证了参数 / This should not happen since we've validated parameters
            throw new RuntimeException("克隆贝塞尔滤波器失败 / Failed to clone Bessel filter", e);
        }
    }
}