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
     * <p>
     * 使用双线性变换法设计数字巴特沃斯低通滤波器。
     * Design digital Butterworth low-pass filter using bilinear transform method.
     * </p>
     */
    private FilterCoefficients computeLowPassCoefficients() {
        double omegaC = 2 * Math.PI * cutoffFrequencies[0] / samplingRate;
        // 预畸变: wa = 2*fs*tan(wd/2)，归一化后 warped = 2*tan(omegaC/2)
        double warped = 2.0 * Math.tan(omegaC / 2.0);

        Complex[] analogPoles = computeButterworthPoles(order);
        // 去归一化: 将归一化极点乘以预畸变截止频率
        Complex[] denormPoles = new Complex[order];
        for (int i = 0; i < order; i++) {
            denormPoles[i] = new Complex(analogPoles[i].real * warped, analogPoles[i].imag * warped);
        }

        // 双线性变换: s = 2*(1-z^-1)/(1+z^-1) 的逆映射: z = (2+s)/(2-s)
        Complex[] digitalPoles = new Complex[order];
        for (int i = 0; i < order; i++) {
            digitalPoles[i] = bilinearMap(denormPoles[i]);
        }

        // 分母: 从数字极点展开多项式 ∏(1 - p_i * z^-1)
        double[] denominator = expandFromRoots(digitalPoles);

        // 分子: (1+z^-1)^n，归一化使 DC 增益为 1
        double[] numerator = new double[order + 1];
        for (int i = 0; i <= order; i++) {
            numerator[i] = binomialCoefficient(order, i);
        }

        // 归一化 DC 增益: H(z=1) = sum(num) / sum(den) = 1
        double denSum = 0, numSum = 0;
        for (int i = 0; i <= order; i++) {
            denSum += denominator[i];
            numSum += numerator[i];
        }
        double dcGain = denSum / numSum;
        for (int i = 0; i <= order; i++) {
            numerator[i] *= dcGain;
        }

        return new FilterCoefficients(numerator, denominator);
    }

    /**
     * 计算高通巴特沃斯滤波器系数 / Compute high-pass Butterworth filter coefficients
     * <p>
     * 通过低通到高通变换: s_lp = w_c² / s_hp。
     * Transform low-pass to high-pass: s_lp = w_c² / s_hp.
     * </p>
     */
    private FilterCoefficients computeHighPassCoefficients() {
        double omegaC = 2 * Math.PI * cutoffFrequencies[0] / samplingRate;
        double warped = 2.0 * Math.tan(omegaC / 2.0);

        Complex[] analogLpPoles = computeButterworthPoles(order);
        // 高通变换在模拟域: 每个低通极点 p 变为 warped/p
        Complex[] hpPoles = new Complex[order];
        for (int i = 0; i < order; i++) {
            // s_lp = warped² / s_hp  ⇒  s_hp = warped² / s_lp
            // 极点: p_hp = warped² / p_lp = warped² * conj(p_lp) / |p_lp|²
            // 归一化极点 |p_lp| = 1, 所以 p_hp = warped² * conj(p_lp)
            double re = analogLpPoles[i].real;
            double im = analogLpPoles[i].imag;
            double mag2 = re * re + im * im; // = 1 for normalized Butterworth poles
            double hpRe = warped * warped * re / mag2;
            double hpIm = -warped * warped * im / mag2;
            hpPoles[i] = new Complex(hpRe, hpIm);
        }

        // 双线性变换
        Complex[] digitalPoles = new Complex[order];
        for (int i = 0; i < order; i++) {
            digitalPoles[i] = bilinearMap(hpPoles[i]);
        }

        // 分母
        double[] denominator = expandFromRoots(digitalPoles);

        // 分子: (1 - z^-1)^n
        double[] numerator = new double[order + 1];
        for (int i = 0; i <= order; i++) {
            double sign = (i % 2 == 0) ? 1.0 : -1.0;
            numerator[i] = sign * binomialCoefficient(order, i);
        }

        // 归一化高频增益 (z=-1)
        double numGain = 0, denGain = 0;
        for (int i = 0; i <= order; i++) {
            double sign = (i % 2 == 0) ? 1.0 : -1.0;
            numGain += numerator[i] * sign;
            denGain += denominator[i] * sign;
        }
        double hfGain = denGain / numGain;
        for (int i = 0; i <= order; i++) {
            numerator[i] *= hfGain;
        }

        return new FilterCoefficients(numerator, denominator);
    }

    /**
     * 计算带通巴特沃斯滤波器系数 / Compute band-pass Butterworth filter coefficients
     * <p>
     * 低通→带通变换: s_lp = (s² + w₀²) / (B·s)，其中 w₀ = sqrt(wa1·wa2), B = wa2 - wa1。
     * Low-pass to band-pass: s_lp = (s² + w₀²) / (B·s), w₀ = sqrt(wa1·wa2), B = wa2 - wa1.
     * </p>
     */
    private FilterCoefficients computeBandPassCoefficients() {
        double wc1 = 2.0 * Math.PI * cutoffFrequencies[0] / samplingRate;
        double wc2 = 2.0 * Math.PI * cutoffFrequencies[1] / samplingRate;
        double wa1 = 2.0 * Math.tan(wc1 / 2.0);
        double wa2 = 2.0 * Math.tan(wc2 / 2.0);
        double w0 = Math.sqrt(wa1 * wa2);
        double B = wa2 - wa1;

        Complex[] lpPoles = computeButterworthPoles(order);

        // 每个低通极点产生一对带通极点: s = (p·B ± sqrt(p²·B² - 4·w₀²)) / 2
        Complex[] bpPoles = new Complex[order * 2];
        for (int i = 0; i < order; i++) {
            Complex p = lpPoles[i];
            double a = p.real * B;
            double b = p.imag * B;
            // discriminant = (a+jb)² - 4w₀² = (a²-b²-4w₀²) + j(2ab)
            double discRe = a * a - b * b - 4.0 * w0 * w0;
            double discIm = 2.0 * a * b;
            Complex disc = new Complex(discRe, discIm).sqrt();
            // s = (a+jb ± disc) / 2
            bpPoles[2 * i] = new Complex((a + disc.real) / 2.0, (b + disc.imag) / 2.0);
            bpPoles[2 * i + 1] = new Complex((a - disc.real) / 2.0, (b - disc.imag) / 2.0);
        }

        // 双线性变换后展开为数字域系数
        Complex[] digitalPoles = new Complex[bpPoles.length];
        for (int i = 0; i < bpPoles.length; i++) {
            digitalPoles[i] = bilinearMap(bpPoles[i]);
        }

        double[] denominator = expandFromRoots(digitalPoles);

        // 分子: (1 - z^-2)^order (带通变换将 s^n 映射为 s^n，双线性后得此结果)
        int numOrder = order * 2;
        double[] numerator = new double[numOrder + 1];
        for (int k = 0; k <= order; k++) {
            double sign = (k % 2 == 0) ? 1.0 : -1.0;
            int pos = 2 * k;
            if (pos <= numOrder) {
                numerator[pos] = sign * binomialCoefficient(order, k);
            }
        }

        // 归一化中心频率增益
        double cosW0 = Math.cos((wc1 + wc2) / 2.0);
        Complex z = Complex.unit((wc1 + wc2) / 2.0); // z = e^(j·w₀)
        Complex numEval = evaluatePoly(numerator, z);
        Complex denEval = evaluatePoly(denominator, z);
        double gain = denEval.magnitude() / numEval.magnitude();
        for (int i = 0; i <= numOrder; i++) {
            numerator[i] *= gain;
        }

        return new FilterCoefficients(numerator, denominator);
    }

    /**
     * 计算带阻巴特沃斯滤波器系数 / Compute band-stop Butterworth filter coefficients
     * <p>
     * 低通→带阻变换: s_lp = B·s / (s² + w₀²)。
     * Low-pass to band-stop: s_lp = B·s / (s² + w₀²).
     * </p>
     */
    private FilterCoefficients computeBandStopCoefficients() {
        double wc1 = 2.0 * Math.PI * cutoffFrequencies[0] / samplingRate;
        double wc2 = 2.0 * Math.PI * cutoffFrequencies[1] / samplingRate;
        double wa1 = 2.0 * Math.tan(wc1 / 2.0);
        double wa2 = 2.0 * Math.tan(wc2 / 2.0);
        double w0 = Math.sqrt(wa1 * wa2);
        double B = wa2 - wa1;

        Complex[] lpPoles = computeButterworthPoles(order);

        // 每个低通极点 p 产生: s_lp * (s² + w₀²) = B·s
        // 即 s² - (B/p)·s + w₀² = 0  ⇒  s = (B/p ± sqrt(B²/p² - 4w₀²)) / 2
        Complex[] bsPoles = new Complex[order * 2];
        for (int i = 0; i < order; i++) {
            Complex p = lpPoles[i];
            // B/p = B * conj(p) / |p|² = B * (re - j*im)
            double mag2 = p.real * p.real + p.imag * p.imag;
            double bOverPRe = B * p.real / mag2;
            double bOverPIm = -B * p.imag / mag2;
            // discriminant = (B/p)² - 4w₀²
            double discRe = bOverPRe * bOverPRe - bOverPIm * bOverPIm - 4.0 * w0 * w0;
            double discIm = 2.0 * bOverPRe * bOverPIm;
            Complex disc = new Complex(discRe, discIm).sqrt();
            bsPoles[2 * i] = new Complex((bOverPRe + disc.real) / 2.0, (bOverPIm + disc.imag) / 2.0);
            bsPoles[2 * i + 1] = new Complex((bOverPRe - disc.real) / 2.0, (bOverPIm - disc.imag) / 2.0);
        }

        Complex[] digitalPoles = new Complex[bsPoles.length];
        for (int i = 0; i < bsPoles.length; i++) {
            digitalPoles[i] = bilinearMap(bsPoles[i]);
        }

        double[] denominator = expandFromRoots(digitalPoles);

        // 带阻分子: (1 - 2·cos(w₀)·z^-1 + z^-2)^order
        int numOrder = order * 2;
        double cosW0 = Math.cos((wc1 + wc2) / 2.0);
        double[] numerator = new double[numOrder + 1];
        numerator[0] = 1.0;
        for (int seg = 0; seg < order; seg++) {
            double[] next = new double[numOrder + 1];
            for (int i = 0; i <= numOrder; i++) {
                next[i] = numerator[i];
                if (i >= 1) next[i] += -2.0 * cosW0 * numerator[i - 1];
                if (i >= 2) next[i] += numerator[i - 2];
            }
            numerator = next;
        }

        // 归一化 DC 增益
        double denSum = 0, numSum = 0;
        for (int i = 0; i <= numOrder; i++) {
            denSum += denominator[i];
            numSum += numerator[i];
        }
        double dcGain = denSum / numSum;
        for (int i = 0; i <= numOrder; i++) {
            numerator[i] *= dcGain;
        }

        return new FilterCoefficients(numerator, denominator);
    }

    // ========== 数字滤波器设计工具方法 / Digital Filter Design Utilities ==========

    /**
     * 双线性变换: 模拟 s 域映射到数字 z 域。
     * Bilinear transform: map analog s-domain to digital z-domain.
     * z = (2 + s) / (2 - s)，使用归一化采样周期 T=1.
     */
    private static Complex bilinearMap(Complex s) {
        double denRe = 2.0 - s.real;
        double denIm = -s.imag;
        double den2 = denRe * denRe + denIm * denIm;
        double zRe = ((2.0 + s.real) * denRe + s.imag * denIm) / den2;
        double zIm = (s.imag * denRe - (2.0 + s.real) * denIm) / den2;
        return new Complex(zRe, zIm);
    }

    /**
     * 从共轭根展开实系数多项式。
     * Expand a real-coefficient polynomial from complex-conjugate root pairs.
     * 返回多项式: poly[0] + poly[1]*z^-1 + ... + poly[n]*z^-n
     * Returns: poly[0] + poly[1]*z^-1 + ... + poly[n]*z^-n = ∏(1 - root_i * z^-1)
     */
    private static double[] expandFromRoots(Complex[] roots) {
        int n = roots.length;
        double[] coeffs = new double[n + 1];
        coeffs[0] = 1.0;

        for (int i = 0; i < n; i++) {
            if (Math.abs(roots[i].imag) < 1e-15) {
                // 实根: 乘以 (1 - r*z^-1)
                for (int j = coeffs.length - 1; j > 0; j--) {
                    coeffs[j] -= roots[i].real * coeffs[j - 1];
                }
            } else if (roots[i].imag > 0) {
                // 共轭复根对: (1 - p*z^-1)(1 - conj(p)*z^-1) = 1 - 2·Re(p)·z^-1 + |p|²·z^-2
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
            // 跳过负虚部根（已作为共轭对的一半被处理）
        }
        return coeffs;
    }

    /**
     * 计算复多项式在 z 点的值。
     */
    private static Complex evaluatePoly(double[] coeffs, Complex z) {
        Complex sum = Complex.ZERO;
        Complex zPow = Complex.ONE;
        for (double c : coeffs) {
            sum = sum.add(zPow.scale(c));
            zPow = zPow.multiply(z);
        }
        return sum;
    }

    /**
     * 二项式系数 C(n, k).
     */
    private static double binomialCoefficient(int n, int k) {
        if (k < 0 || k > n) return 0;
        if (k > n - k) k = n - k;
        double result = 1.0;
        for (int i = 0; i < k; i++) {
            result = result * (n - i) / (i + 1);
        }
        return result;
    }

    /**
     * 计算巴特沃斯模拟原型极点（归一化截止频率 = 1 rad/s）。
     * Compute Butterworth analog prototype poles (normalized cutoff = 1 rad/s).
     */
    private Complex[] computeButterworthPoles(int order) {
        Complex[] poles = new Complex[order];
        for (int k = 0; k < order; k++) {
            double angle = Math.PI * (2 * k + order + 1) / (2 * order);
            poles[k] = new Complex(-Math.cos(angle), -Math.sin(angle));
        }
        return poles;
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
     * <p>
     * 通过计算 H(e^{jw}) = sum(b_k·e^{-jwk}) / sum(a_k·e^{-jwk}) 得到幅频和相频响应。
     * Compute magnitude and phase response by evaluating H(e^{jw}) at each frequency.
     * </p>
     *
     * @param frequencies 频率点数组 (Hz) / Frequency point array (Hz)
     * @return 频率响应 / Frequency response
     */
    @Override
    public FrequencyResponse getFrequencyResponse(double[] frequencies) throws SignalProcessingException {
        double[] magnitude = new double[frequencies.length];
        double[] phase = new double[frequencies.length];
        double[] num = coefficients.getNumerator();
        double[] den = coefficients.getDenominator();

        for (int i = 0; i < frequencies.length; i++) {
            double omega = 2.0 * Math.PI * frequencies[i] / samplingRate;
            // H(e^{jw}) = sum(b_k·e^{-jwk}) / sum(a_k·e^{-jwk})
            double numRe = 0, numIm = 0, denRe = 0, denIm = 0;
            for (int k = 0; k < num.length; k++) {
                double cosTerm = Math.cos(-omega * k);
                double sinTerm = Math.sin(-omega * k);
                numRe += num[k] * cosTerm;
                numIm += num[k] * sinTerm;
            }
            for (int k = 0; k < den.length; k++) {
                double cosTerm = Math.cos(-omega * k);
                double sinTerm = Math.sin(-omega * k);
                denRe += den[k] * cosTerm;
                denIm += den[k] * sinTerm;
            }
            Complex h = new Complex(numRe, numIm).divide(new Complex(denRe, denIm));
            magnitude[i] = h.magnitude();
            phase[i] = h.phase();
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