package com.yishape.lab.math.signal.filter;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.signal.core.ISignalProcessor;
import com.yishape.lab.math.signal.core.SignalProcessingException;

/**
 * 信号滤波器接口 / Signal Filter Interface
 * <p>
 * 定义所有信号滤波操作的基础接口，支持各种滤波器类型。
 * 使用策略模式支持不同的滤波算法实现。
 * </p>
 * <p>
 * Defines the base interface for all signal filtering operations supporting various filter types.
 * Uses Strategy pattern to support different filtering algorithm implementations.
 * </p>
 *
 * @param <T> 信号数据类型 / Signal data type
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface ISignalFilter<T extends Number> extends ISignalProcessor<T> {
    
    /**
     * 滤波器类型枚举 / Filter Type Enum
     */
    enum FilterType {
        LOW_PASS("低通", "Low-pass"),
        HIGH_PASS("高通", "High-pass"),
        BAND_PASS("带通", "Band-pass"),
        BAND_STOP("带阻", "Band-stop"),
        ALL_PASS("全通", "All-pass"),
        NOTCH("陷波", "Notch"),
        ADAPTIVE("自适应", "Adaptive");
        
        private final String chineseName;
        private final String englishName;
        
        FilterType(String chineseName, String englishName) {
            this.chineseName = chineseName;
            this.englishName = englishName;
        }
        
        public String getChineseName() { return chineseName; }
        public String getEnglishName() { return englishName; }
    }
    
    /**
     * 滤波器实现类型枚举 / Filter Implementation Type Enum
     */
    enum FilterImplementation {
        FIR("有限冲激响应", "Finite Impulse Response"),
        IIR("无限冲激响应", "Infinite Impulse Response"),
        BUTTERWORTH("巴特沃斯", "Butterworth"),
        CHEBYSHEV_I("切比雪夫I型", "Chebyshev Type I"),
        CHEBYSHEV_II("切比雪夫II型", "Chebyshev Type II"),
        ELLIPTIC("椭圆", "Elliptic"),
        BESSEL("贝塞尔", "Bessel"),
        GAUSSIAN("高斯", "Gaussian"),
        MOVING_AVERAGE("移动平均", "Moving Average"),
        MEDIAN("中值", "Median");
        
        private final String chineseName;
        private final String englishName;
        
        FilterImplementation(String chineseName, String englishName) {
            this.chineseName = chineseName;
            this.englishName = englishName;
        }
        
        public String getChineseName() { return chineseName; }
        public String getEnglishName() { return englishName; }
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
    IVector<T> filter(IVector<T> signal) throws SignalProcessingException;
    
    /**
     * 获取滤波器类型 / Get filter type
     * <p>
     * 返回当前滤波器的类型。
     * Return the type of current filter.
     * </p>
     *
     * @return 滤波器类型 / Filter type
     */
    FilterType getFilterType();
    
    /**
     * 获取滤波器实现类型 / Get filter implementation type
     * <p>
     * 返回当前滤波器的实现类型。
     * Return the implementation type of current filter.
     * </p>
     *
     * @return 滤波器实现类型 / Filter implementation type
     */
    FilterImplementation getImplementationType();
    
    /**
     * 获取滤波器阶数 / Get filter order
     * <p>
     * 返回滤波器的阶数。
     * Return the order of the filter.
     * </p>
     *
     * @return 滤波器阶数 / Filter order
     */
    int getOrder();
    
    /**
     * 获取截止频率 / Get cutoff frequencies
     * <p>
     * 返回滤波器的截止频率数组。
     * Return array of cutoff frequencies of the filter.
     * </p>
     *
     * @return 截止频率数组 / Cutoff frequency array
     */
    double[] getCutoffFrequencies();
    
    /**
     * 设置截止频率 / Set cutoff frequencies
     * <p>
     * 设置滤波器的截止频率。
     * Set cutoff frequencies of the filter.
     * </p>
     *
     * @param frequencies 截止频率数组 / Cutoff frequency array
     * @throws SignalProcessingException 频率设置无效时抛出 / Thrown when frequency setting is invalid
     */
    void setCutoffFrequencies(double... frequencies) throws SignalProcessingException;
    
    /**
     * 获取采样率 / Get sampling rate
     * <p>
     * 返回滤波器使用的采样率。
     * Return sampling rate used by the filter.
     * </p>
     *
     * @return 采样率 / Sampling rate
     */
    double getSamplingRate();
    
    /**
     * 设置采样率 / Set sampling rate
     * <p>
     * 设置滤波器使用的采样率。
     * Set sampling rate used by the filter.
     * </p>
     *
     * @param samplingRate 采样率 / Sampling rate
     * @throws SignalProcessingException 采样率设置无效时抛出 / Thrown when sampling rate setting is invalid
     */
    void setSamplingRate(double samplingRate) throws SignalProcessingException;
    
    /**
     * 获取滤波器系数 / Get filter coefficients
     * <p>
     * 返回滤波器的系数。对于FIR滤波器返回单个数组，对于IIR滤波器返回分子和分母系数。
     * Return filter coefficients. For FIR filters return single array, for IIR filters return numerator and denominator coefficients.
     * </p>
     *
     * @return 滤波器系数 / Filter coefficients
     */
    FilterCoefficients getCoefficients();
    
    /**
     * 计算滤波器的频率响应 / Calculate frequency response of filter
     * <p>
     * 计算滤波器在指定频率点的频率响应。
     * Calculate frequency response of filter at specified frequency points.
     * </p>
     *
     * @param frequencies 频率点数组 / Frequency point array
     * @return 频率响应 / Frequency response
     * @throws SignalProcessingException 计算过程中发生错误时抛出 / Thrown when errors occur during calculation
     */
    FrequencyResponse getFrequencyResponse(double[] frequencies) throws SignalProcessingException;
    
    /**
     * 默认实现process方法 / Default implementation of process method
     */
    @Override
    default IVector<T> process(IVector<T> input) throws SignalProcessingException {
        return filter(input);
    }
    
    /**
     * 滤波器系数内部类 / Filter Coefficients Inner Class
     */
    class FilterCoefficients {
        private final double[] numerator;
        private final double[] denominator;
        private final boolean isFIR;
        
        /**
         * FIR滤波器系数构造函数 / FIR filter coefficients constructor
         */
        public FilterCoefficients(double[] coefficients) {
            this.numerator = coefficients.clone();
            this.denominator = new double[]{1.0};
            this.isFIR = true;
        }
        
        /**
         * IIR滤波器系数构造函数 / IIR filter coefficients constructor
         */
        public FilterCoefficients(double[] numerator, double[] denominator) {
            this.numerator = numerator.clone();
            this.denominator = denominator.clone();
            this.isFIR = false;
        }
        
        public double[] getNumerator() { return numerator.clone(); }
        public double[] getDenominator() { return denominator.clone(); }
        public boolean isFIR() { return isFIR; }
        public boolean isIIR() { return !isFIR; }
    }
    
    /**
     * 频率响应内部类 / Frequency Response Inner Class
     */
    class FrequencyResponse {
        private final double[] frequencies;
        private final double[] magnitude;
        private final double[] phase;
        
        public FrequencyResponse(double[] frequencies, double[] magnitude, double[] phase) {
            this.frequencies = frequencies.clone();
            this.magnitude = magnitude.clone();
            this.phase = phase.clone();
        }
        
        public double[] getFrequencies() { return frequencies.clone(); }
        public double[] getMagnitude() { return magnitude.clone(); }
        public double[] getPhase() { return phase.clone(); }
        public double[] getMagnitudeDB() {
            double[] magnitudeDB = new double[magnitude.length];
            for (int i = 0; i < magnitude.length; i++) {
                magnitudeDB[i] = 20 * Math.log10(Math.max(magnitude[i], 1e-12));
            }
            return magnitudeDB;
        }
    }
}