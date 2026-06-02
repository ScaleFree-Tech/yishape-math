package com.yishape.lab.math.signal.filter;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.signal.core.AbstractSignalProcessor;
import com.yishape.lab.math.signal.core.Complex;
import com.yishape.lab.math.signal.core.SignalProcessingException;

/**
 * 切比雪夫滤波器实现类 / Chebyshev Filter Implementation Class
 * <p>
 * 实现切比雪夫I型和II型滤波器。
 * I型：通带有等波纹，阻带单调下降。
 * II型：通带单调，阻带有等波纹。
 * </p>
 * <p>
 * Implements Chebyshev Type I and Type II filters.
 * Type I: Equiripple in passband, monotonic in stopband.
 * Type II: Monotonic in passband, equiripple in stopband.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ChebyshevFilter extends AbstractSignalProcessor<Double> implements ISignalFilter<Double> {
    
    /**
     * 切比雪夫滤波器类型 / Chebyshev Filter Type
     */
    public enum ChebyshevType {
        TYPE_I("I型", "Type I"),
        TYPE_II("II型", "Type II");
        
        private final String chineseName;
        private final String englishName;
        
        ChebyshevType(String chineseName, String englishName) {
            this.chineseName = chineseName;
            this.englishName = englishName;
        }
        
        public String getChineseName() { return chineseName; }
        public String getEnglishName() { return englishName; }
    }
    
    private FilterType filterType;
    private FilterImplementation implementationType;
    private ChebyshevType chebyshevType;
    private int order;
    private double[] cutoffFrequencies;
    private double samplingRate;
    private double ripple;  // 波纹大小 (dB) / Ripple magnitude (dB)
    private FilterCoefficients coefficients;
    
    /**
     * 构造函数 / Constructor
     * <p>
     * 创建切比雪夫低通滤波器。
     * Create Chebyshev low-pass filter.
     * </p>
     *
     * @param chebyshevType 切比雪夫类型 / Chebyshev type
     * @param order 滤波器阶数 / Filter order
     * @param cutoffFreq 截止频率 / Cutoff frequency
     * @param samplingRate 采样率 / Sampling rate
     * @param ripple 波纹大小 (dB) / Ripple magnitude (dB)
     * @throws SignalProcessingException 参数无效时抛出 / Thrown when parameters are invalid
     */
    public ChebyshevFilter(ChebyshevType chebyshevType, int order, double cutoffFreq, 
                          double samplingRate, double ripple) throws SignalProcessingException {
        super("Chebyshev Filter", "1.0.0");
        
        validateParameters(order, new double[]{cutoffFreq}, samplingRate, ripple);
        
        this.filterType = FilterType.LOW_PASS;
        this.implementationType = (chebyshevType == ChebyshevType.TYPE_I) ? 
            FilterImplementation.CHEBYSHEV_I : FilterImplementation.CHEBYSHEV_II;
        this.chebyshevType = chebyshevType;
        this.order = order;
        this.cutoffFrequencies = new double[]{cutoffFreq};
        this.samplingRate = samplingRate;
        this.ripple = ripple;
        
        computeCoefficients();
    }
    
    /**
     * 构造函数 / Constructor
     * <p>
     * 创建指定类型的切比雪夫滤波器。
     * Create Chebyshev filter of specified type.
     * </p>
     *
     * @param filterType 滤波器类型 / Filter type
     * @param chebyshevType 切比雪夫类型 / Chebyshev type
     * @param order 滤波器阶数 / Filter order
     * @param cutoffFreqs 截止频率数组 / Cutoff frequency array
     * @param samplingRate 采样率 / Sampling rate
     * @param ripple 波纹大小 (dB) / Ripple magnitude (dB)
     * @throws SignalProcessingException 参数无效时抛出 / Thrown when parameters are invalid
     */
    public ChebyshevFilter(FilterType filterType, ChebyshevType chebyshevType, int order, 
                          double[] cutoffFreqs, double samplingRate, double ripple) throws SignalProcessingException {
        super("Chebyshev Filter", "1.0.0");
        
        validateParameters(order, cutoffFreqs, samplingRate, ripple);
        validateFilterType(filterType, cutoffFreqs);
        
        this.filterType = filterType;
        this.implementationType = (chebyshevType == ChebyshevType.TYPE_I) ? 
            FilterImplementation.CHEBYSHEV_I : FilterImplementation.CHEBYSHEV_II;
        this.chebyshevType = chebyshevType;
        this.order = order;
        this.cutoffFrequencies = cutoffFreqs.clone();
        this.samplingRate = samplingRate;
        this.ripple = ripple;
        
        computeCoefficients();
    }
    
    /**
     * 验证参数 / Validate parameters
     */
    private void validateParameters(int order, double[] cutoffFreqs, double samplingRate, double ripple) 
            throws SignalProcessingException {
        if (order <= 0) {
            throw new SignalProcessingException("滤波器阶数必须大于0 / Filter order must be greater than 0");
        }
        if (samplingRate <= 0) {
            throw new SignalProcessingException("采样率必须大于0 / Sampling rate must be greater than 0");
        }
        if (ripple <= 0) {
            throw new SignalProcessingException("波纹参数必须大于0 / Ripple parameter must be greater than 0");
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
     * 计算切比雪夫滤波器系数 / Compute Chebyshev filter coefficients
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
     * 计算低通切比雪夫滤波器系数 / Compute low-pass Chebyshev filter coefficients
     */
    private FilterCoefficients computeLowPassCoefficients() {
        // 归一化截止频率 / Normalize cutoff frequency
        double omegaC = 2 * Math.PI * cutoffFrequencies[0] / samplingRate;
        double warpedFreq = 2 * Math.tan(omegaC / 2);
        
        // 计算切比雪夫参数 / Calculate Chebyshev parameters
        double epsilon = Math.sqrt(Math.pow(10, ripple / 10) - 1);
        
        // 计算极点 / Calculate poles
        Complex[] poles = computeChebyshevPoles(order, epsilon, chebyshevType);
        
        // 将极点转换为系数 / Convert poles to coefficients
        return polesAndZerosToCoefficients(poles, new Complex[0], warpedFreq);
    }
    
    /**
     * 计算反双曲正弦 / Calculate inverse hyperbolic sine
     */
    private static double asinh(double x) {
        return Math.log(x + Math.sqrt(x * x + 1));
    }
    
    /**
     * 计算切比雪夫极点 / Compute Chebyshev poles
     */
    private Complex[] computeChebyshevPoles(int order, double epsilon, ChebyshevType type) {
        Complex[] poles = new Complex[order];
        
        // 计算辅助参数 / Calculate auxiliary parameters
        double alpha = (1.0 / order) * asinh(1.0 / epsilon);
        double sinhAlpha = Math.sinh(alpha);
        double coshAlpha = Math.cosh(alpha);
        
        for (int k = 0; k < order; k++) {
            double theta = Math.PI * (2 * k + 1) / (2 * order);
            
            double real, imag;
            if (type == ChebyshevType.TYPE_I) {
                // Type I 极点 / Type I poles
                real = -sinhAlpha * Math.sin(theta);
                imag = coshAlpha * Math.cos(theta);
            } else {
                // Type II 极点 / Type II poles
                real = -Math.sin(theta) / sinhAlpha;
                imag = Math.cos(theta) / coshAlpha;
                
                // Type II 需要倒数变换 / Type II needs reciprocal transformation
                Complex pole = new Complex(real, imag);
                pole = new Complex(1, 0).divide(pole);
                real = pole.real;
                imag = pole.imag;
            }
            
            poles[k] = new Complex(real, imag);
        }
        
        return poles;
    }
    
    /**
     * 计算高通切比雪夫滤波器系数 / Compute high-pass Chebyshev filter coefficients
     */
    private FilterCoefficients computeHighPassCoefficients() {
        // 先计算低通原型 / Calculate low-pass prototype first
        FilterCoefficients lowPassCoeffs = computeLowPassCoefficients();
        
        // 进行低通到高通变换 / Apply low-pass to high-pass transformation
        return transformLowPassToHighPass(lowPassCoeffs);
    }
    
    /**
     * 计算带通切比雪夫滤波器系数 / Compute band-pass Chebyshev filter coefficients
     */
    private FilterCoefficients computeBandPassCoefficients() {
        FilterCoefficients lowPassCoeffs = computeLowPassCoefficients();
        return transformLowPassToBandPass(lowPassCoeffs);
    }
    
    /**
     * 计算带阻切比雪夫滤波器系数 / Compute band-stop Chebyshev filter coefficients
     */
    private FilterCoefficients computeBandStopCoefficients() {
        FilterCoefficients lowPassCoeffs = computeLowPassCoefficients();
        return transformLowPassToBandStop(lowPassCoeffs);
    }
    
    /**
     * 将模拟极点和零点转换为数字滤波器系数（通过双线性变换）。
     * Convert analog poles and zeros to digital filter coefficients via bilinear transform.
     *
     * @param poles      模拟域极点 / Analog-domain poles
     * @param zeros      模拟域零点 / Analog-domain zeros
     * @param warpedFreq 预畸变截止频率 / Prewarped cutoff frequency
     */
    private FilterCoefficients polesAndZerosToCoefficients(Complex[] poles, Complex[] zeros, double warpedFreq) {
        int n = poles.length;

        // 1. 去归一化: 将归一化极点/零点乘以预畸变频率
        Complex[] denormPoles = new Complex[n];
        for (int i = 0; i < n; i++) {
            denormPoles[i] = new Complex(poles[i].real * warpedFreq, poles[i].imag * warpedFreq);
        }
        Complex[] denormZeros = new Complex[zeros.length];
        for (int i = 0; i < zeros.length; i++) {
            denormZeros[i] = new Complex(zeros[i].real * warpedFreq, zeros[i].imag * warpedFreq);
        }

        // 2. 双线性变换: z = (2+s)/(2-s)
        Complex[] digitalPoles = new Complex[n];
        for (int i = 0; i < n; i++) {
            digitalPoles[i] = bilinearMap(denormPoles[i]);
        }
        Complex[] digitalZeros = new Complex[zeros.length];
        for (int i = 0; i < zeros.length; i++) {
            digitalZeros[i] = bilinearMap(denormZeros[i]);
        }

        // 3. 从数字极点和零点展开多项式
        // 分母: ∏(1 - dp_i * z^-1) = expandFromRoots(digitalPoles)
        double[] denominator = expandFromRoots(digitalPoles);

        // 分子: 全极点滤波器分子来自 z=-1 映射（零频在无穷远处）
        double[] numerator;
        if (digitalZeros.length == 0) {
            // (1+z^-1)^n
            numerator = new double[n + 1];
            for (int i = 0; i <= n; i++) {
                numerator[i] = binomialCoeff(n, i);
            }
        } else {
            numerator = expandFromRoots(digitalZeros);
        }

        // 4. 归一化 DC 增益
        double denSum = 0, numSum = 0;
        for (int i = 0; i < denominator.length; i++) denSum += denominator[i];
        for (int i = 0; i < numerator.length; i++) numSum += numerator[i];
        double dcGain = denSum / numSum;
        for (int i = 0; i < numerator.length; i++) {
            numerator[i] *= dcGain;
        }

        return new FilterCoefficients(numerator, denominator);
    }

    /** 双线性变换映射: z = (2+s)/(2-s) */
    private static Complex bilinearMap(Complex s) {
        double denRe = 2.0 - s.real;
        double denIm = -s.imag;
        double den2 = denRe * denRe + denIm * denIm;
        return new Complex(
            ((2.0 + s.real) * denRe + s.imag * denIm) / den2,
            (s.imag * denRe - (2.0 + s.real) * denIm) / den2
        );
    }

    /** 从共轭根展开: ∏(1 - r_i * z^-1)，共轭对一起处理保证实系数 */
    private static double[] expandFromRoots(Complex[] roots) {
        int n = roots.length;
        double[] coeffs = new double[n + 1];
        coeffs[0] = 1.0;
        for (int i = 0; i < n; i++) {
            if (Math.abs(roots[i].imag) < 1e-15) {
                for (int j = coeffs.length - 1; j > 0; j--) {
                    coeffs[j] -= roots[i].real * coeffs[j - 1];
                }
            } else if (roots[i].imag > 0) {
                double a1 = -2.0 * roots[i].real;
                double a2 = roots[i].real * roots[i].real + roots[i].imag * roots[i].imag;
                double[] next = new double[n + 1];
                for (int j = 0; j <= n; j++) {
                    next[j] = coeffs[j];
                    if (j >= 1) next[j] += a1 * coeffs[j - 1];
                    if (j >= 2) next[j] += a2 * coeffs[j - 2];
                }
                coeffs = next;
            }
        }
        return coeffs;
    }

    /** 二项式系数 C(n, k) */
    private static double binomialCoeff(int n, int k) {
        if (k < 0 || k > n) return 0;
        if (k > n - k) k = n - k;
        double r = 1.0;
        for (int i = 0; i < k; i++) r = r * (n - i) / (i + 1);
        return r;
    }
    
    /**
     * 低通到高通变换 / Low-pass to high-pass transformation
     */
    private FilterCoefficients transformLowPassToHighPass(FilterCoefficients lowPassCoeffs) {
        double[] num = lowPassCoeffs.getNumerator();
        double[] den = lowPassCoeffs.getDenominator();
        
        // 高通变换：s -> 1/s
        double[] newNum = new double[den.length];
        double[] newDen = new double[num.length];
        
        for (int i = 0; i < den.length; i++) {
            newNum[i] = den[den.length - 1 - i];
        }
        for (int i = 0; i < num.length; i++) {
            newDen[i] = num[num.length - 1 - i];
        }
        
        return new FilterCoefficients(newNum, newDen);
    }
    
    /**
     * 低通到带通变换 / Low-pass to band-pass transformation
     */
    private FilterCoefficients transformLowPassToBandPass(FilterCoefficients lowPassCoeffs) {
        // 简化的带通变换 / Simplified band-pass transformation
        double[] num = lowPassCoeffs.getNumerator();
        double[] den = lowPassCoeffs.getDenominator();
        
        double[] newNum = new double[num.length * 2 - 1];
        double[] newDen = new double[den.length * 2 - 1];
        
        System.arraycopy(num, 0, newNum, 0, num.length);
        System.arraycopy(den, 0, newDen, 0, den.length);
        
        return new FilterCoefficients(newNum, newDen);
    }
    
    /**
     * 低通到带阻变换 / Low-pass to band-stop transformation
     */
    private FilterCoefficients transformLowPassToBandStop(FilterCoefficients lowPassCoeffs) {
        // 简化的带阻变换 / Simplified band-stop transformation
        double[] num = lowPassCoeffs.getNumerator();
        double[] den = lowPassCoeffs.getDenominator();
        
        double[] newNum = new double[den.length * 2 - 1];
        double[] newDen = new double[num.length * 2 - 1];
        
        System.arraycopy(den, 0, newNum, 0, den.length);
        System.arraycopy(num, 0, newDen, 0, num.length);
        
        return new FilterCoefficients(newNum, newDen);
    }
    
    @Override
    public IVector<Double> filter(IVector<Double> signal) throws SignalProcessingException {
        if (coefficients == null) {
            throw new SignalProcessingException("滤波器系数未初始化 / Filter coefficients not initialized");
        }
        
        return applyIIRFilter(signal, coefficients.getNumerator(), coefficients.getDenominator());
    }
    
    /**
     * 应用IIR滤波器 / Apply IIR filter
     */
    private IVector<Double> applyIIRFilter(IVector<Double> signal, double[] b, double[] a) {
        int n = signal.length();
        IVector<Double> filtered = Linalg.zeros(n);
        
        // 使用直接型II结构 / Use Direct Form II structure
        double[] w = new double[Math.max(a.length, b.length)];
        
        for (int i = 0; i < n; i++) {
            // 计算中间变量w[n] / Calculate intermediate variable w[n]
            double wn = signal.get(i);
            for (int j = 1; j < a.length; j++) {
                if (i - j >= 0) {
                    wn -= a[j] * w[j];
                }
            }
            wn /= a[0];
            
            // 更新延迟线 / Update delay line
            for (int j = w.length - 1; j > 0; j--) {
                w[j] = w[j - 1];
            }
            w[0] = wn;
            
            // 计算输出y[n] / Calculate output y[n]
            double yn = 0;
            for (int j = 0; j < Math.min(b.length, w.length); j++) {
                yn += b[j] * w[j];
            }
            
            filtered.set(i, yn);
        }
        
        return filtered;
    }
    
    @Override
    public FrequencyResponse getFrequencyResponse(double[] frequencies) throws SignalProcessingException {
        if (coefficients == null) {
            throw new SignalProcessingException("滤波器系数未初始化 / Filter coefficients not initialized");
        }
        
        double[] magnitude = new double[frequencies.length];
        double[] phase = new double[frequencies.length];
        
        double[] numerator = coefficients.getNumerator();
        double[] denominator = coefficients.getDenominator();
        
        for (int i = 0; i < frequencies.length; i++) {
            double omega = 2 * Math.PI * frequencies[i] / samplingRate;
            
            Complex numValue = evaluatePolynomial(numerator, omega);
            Complex denValue = evaluatePolynomial(denominator, omega);
            Complex H = numValue.divide(denValue);
            
            magnitude[i] = H.magnitude();
            phase[i] = H.phase();
        }
        
        return new FrequencyResponse(frequencies, magnitude, phase);
    }
    
    /**
     * 计算多项式在复频率处的值 / Evaluate polynomial at complex frequency
     */
    private Complex evaluatePolynomial(double[] coeffs, double omega) {
        Complex result = new Complex(0, 0);
        Complex jOmega = new Complex(0, omega);
        
        for (int k = 0; k < coeffs.length; k++) {
            Complex term = jOmega.power(k).scale(coeffs[k]);
            result = result.add(term);
        }
        
        return result;
    }
    
    @Override
    protected IVector<Double> doProcess(IVector<Double> input) throws SignalProcessingException {
        return filter(input);
    }
    
    @Override
    public ChebyshevFilter clone() {
        try {
            return new ChebyshevFilter(filterType, chebyshevType, order, cutoffFrequencies, samplingRate, ripple);
        } catch (SignalProcessingException e) {
            throw new RuntimeException("Failed to clone ChebyshevFilter", e);
        }
    }
    
    // Interface implementations
    /**
     * 获取滤波器类型 / Get filter type
     * @return 滤波器类型 / Filter type
     */
    @Override
    public FilterType getFilterType() { return filterType; }

    /**
     * 获取滤波器实现类型 / Get filter implementation type
     * @return 滤波器实现类型 / Filter implementation type
     */
    @Override
    public FilterImplementation getImplementationType() { return implementationType; }

    /**
     * 获取滤波器阶数 / Get filter order
     * @return 滤波器阶数 / Filter order
     */
    @Override
    public int getOrder() { return order; }

    /**
     * 获取截止频率 / Get cutoff frequencies
     * @return 截止频率数组 / Cutoff frequency array
     */
    @Override
    public double[] getCutoffFrequencies() { return cutoffFrequencies.clone(); }

    /**
     * 设置截止频率 / Set cutoff frequencies
     * @param frequencies 截止频率数组 / Cutoff frequency array
     * @throws SignalProcessingException 频率设置无效时抛出 / Thrown when frequency setting is invalid
     */
    @Override
    public void setCutoffFrequencies(double... frequencies) throws SignalProcessingException {
        validateParameters(order, frequencies, samplingRate, ripple);
        validateFilterType(filterType, frequencies);
        this.cutoffFrequencies = frequencies.clone();
        computeCoefficients();
    }

    @Override
    public double getSamplingRate() { return samplingRate; }

    /**
     * 设置采样率 / Set sampling rate
     * @param samplingRate 采样率 / Sampling rate
     * @throws SignalProcessingException 采样率设置无效时抛出 / Thrown when sampling rate setting is invalid
     */
    @Override
    public void setSamplingRate(double samplingRate) throws SignalProcessingException {
        if (samplingRate <= 0) {
            throw new SignalProcessingException("采样率必须大于0 / Sampling rate must be greater than 0");
        }
        this.samplingRate = samplingRate;
        computeCoefficients();
    }

    /**
     * 获取滤波器系数 / Get filter coefficients
     * @return 滤波器系数 / Filter coefficients
     */
    @Override
    public FilterCoefficients getCoefficients() { return coefficients; }

    // Additional getters and setters
    /**
     * 获取切比雪夫类型 / Get Chebyshev type
     * @return 切比雪夫类型 / Chebyshev type
     */
    public ChebyshevType getChebyshevType() { return chebyshevType; }

    /**
     * 获取波纹大小 / Get ripple magnitude
     * @return 波纹大小 (dB) / Ripple magnitude (dB)
     */
    public double getRipple() { return ripple; }

    /**
     * 设置波纹大小 / Set ripple magnitude
     * @param ripple 波纹大小 (dB) / Ripple magnitude (dB)
     * @throws SignalProcessingException 波纹参数无效时抛出 / Thrown when ripple parameter is invalid
     */
    public void setRipple(double ripple) throws SignalProcessingException {
        if (ripple <= 0) {
            throw new SignalProcessingException("波纹参数必须大于0 / Ripple parameter must be greater than 0");
        }
        this.ripple = ripple;
        computeCoefficients();
    }
}