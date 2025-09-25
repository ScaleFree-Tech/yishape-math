package com.reremouse.lab.math.signal.transform;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.signal.core.AbstractSignalProcessor;
import com.reremouse.lab.math.signal.core.Complex;
import com.reremouse.lab.math.signal.core.RereFFT;
import com.reremouse.lab.math.signal.core.SignalProcessingException;
import com.reremouse.lab.util.Tuple2;

/**
 * 希尔伯特变换实现类 / Hilbert Transform Implementation Class
 * <p>
 * 实现希尔伯特变换，用于生成解析信号和计算瞬时特征。
 * 希尔伯特变换将实信号转换为解析信号，可以提取信号的瞬时幅度和相位。
 * </p>
 * <p>
 * Implements Hilbert transform for generating analytic signals and calculating instantaneous features.
 * Hilbert transform converts real signals to analytic signals, enabling extraction of instantaneous amplitude and phase.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class HilbertTransform extends AbstractSignalProcessor<Double> implements ISignalTransform<Double, Tuple2<IVector<Double>, IVector<Double>>> {
    
    /**
     * 构造函数 / Constructor
     */
    public HilbertTransform() {
        super("Hilbert Transform", "1.0.0");
    }
    
    /**
     * 正向希尔伯特变换 / Forward Hilbert transform
     * <p>
     * 计算输入信号的希尔伯特变换，返回解析信号的实部和虚部。
     * Calculate Hilbert transform of input signal, return real and imaginary parts of analytic signal.
     * </p>
     *
     * @param signal 输入时域信号 / Input time domain signal
     * @return 解析信号的实部和虚部 / Real and imaginary parts of analytic signal
     * @throws SignalProcessingException 变换过程中发生错误时抛出 / Thrown when errors occur during transform
     */
    @Override
    public Tuple2<IVector<Double>, IVector<Double>> forward(IVector<Double> signal) throws SignalProcessingException {
        if (signal == null || signal.length() == 0) {
            throw new SignalProcessingException("输入信号不能为空 / Input signal cannot be empty");
        }
        
        try {
            int n = signal.length();
            
            // 将实信号转换为复数数组 / Convert real signal to complex array
            Complex[] complexSignal = new Complex[n];
            for (int i = 0; i < n; i++) {
                complexSignal[i] = new Complex(signal.get(i), 0);
            }
            
            // 零填充确保长度为2的幂
            Complex[] paddedSignal = RereFFT.zeroPadToPowerOfTwo(complexSignal);
            
            // 计算FFT / Calculate FFT
            Complex[] fftSignal = RereFFT.fft(paddedSignal);
            
            // 构造希尔伯特变换滤波器 / Construct Hilbert transform filter
            Complex[] hilbertFilter = createHilbertFilter(n);
            
            // 应用滤波器 / Apply filter
            Complex[] analyticSignal = new Complex[n];
            for (int i = 0; i < n; i++) {
                analyticSignal[i] = fftSignal[i].multiply(hilbertFilter[i]);
            }
            
            // 计算逆FFT得到解析信号 / Calculate inverse FFT to get analytic signal
            Complex[] ifftResult = RereFFT.ifft(analyticSignal);
            
            // 提取实部和虚部 / Extract real and imaginary parts
            IVector<Double> realPart = Linalg.zeros(n);
            IVector<Double> imagPart = Linalg.zeros(n);
            for (int i = 0; i < n; i++) {
                realPart.set(i, ifftResult[i].real);
                imagPart.set(i, ifftResult[i].imag);
            }
            
            return new Tuple2<>(realPart, imagPart);
        } catch (Exception e) {
            throw new SignalProcessingException("希尔伯特变换计算失败 / Hilbert transform calculation failed", e);
        }
    }
    
    /**
     * 构造希尔伯特变换滤波器 / Construct Hilbert transform filter
     */
    private Complex[] createHilbertFilter(int length) {
        Complex[] filter = new Complex[length];
        
        // 对于偶数长度 / For even length
        if (length % 2 == 0) {
            filter[0] = new Complex(1, 0); // DC component
            for (int i = 1; i < length / 2; i++) {
                filter[i] = new Complex(2, 0);
            }
            filter[length / 2] = new Complex(1, 0); // Nyquist frequency
            for (int i = length / 2 + 1; i < length; i++) {
                filter[i] = new Complex(0, 0);
            }
        } else {
            // 对于奇数长度 / For odd length
            filter[0] = new Complex(1, 0); // DC component
            for (int i = 1; i <= (length - 1) / 2; i++) {
                filter[i] = new Complex(2, 0);
            }
            for (int i = (length - 1) / 2 + 1; i < length; i++) {
                filter[i] = new Complex(0, 0);
            }
        }
        
        return filter;
    }
    
    /**
     * 逆变换（不支持） / Inverse transform (not supported)
     * <p>
     * 希尔伯特变换的逆变换没有实际意义。
     * Inverse transform of Hilbert transform has no practical meaning.
     * </p>
     */
    @Override
    public IVector<Double> inverse(Tuple2<IVector<Double>, IVector<Double>> transformed) throws SignalProcessingException {
        throw new UnsupportedOperationException("希尔伯特变换不支持逆变换 / Hilbert transform does not support inverse transform");
    }
    
    /**
     * 计算瞬时幅度 / Calculate instantaneous amplitude
     * <p>
     * 计算解析信号的瞬时幅度（包络）。
     * Calculate instantaneous amplitude (envelope) of analytic signal.
     * </p>
     *
     * @param analyticSignal 解析信号的实部和虚部 / Real and imaginary parts of analytic signal
     * @return 瞬时幅度 / Instantaneous amplitude
     */
    public IVector<Double> calculateInstantaneousAmplitude(Tuple2<IVector<Double>, IVector<Double>> analyticSignal) {
        IVector<Double> realPart = analyticSignal.getFirst();
        IVector<Double> imagPart = analyticSignal.getSecond();
        
        int n = realPart.length();
        IVector<Double> amplitude = Linalg.zeros(n);
        
        for (int i = 0; i < n; i++) {
            double real = realPart.get(i);
            double imag = imagPart.get(i);
            amplitude.set(i, Math.sqrt(real * real + imag * imag));
        }
        
        return amplitude;
    }
    
    /**
     * 计算瞬时相位 / Calculate instantaneous phase
     * <p>
     * 计算解析信号的瞬时相位。
     * Calculate instantaneous phase of analytic signal.
     * </p>
     *
     * @param analyticSignal 解析信号的实部和虚部 / Real and imaginary parts of analytic signal
     * @return 瞬时相位 / Instantaneous phase
     */
    public IVector<Double> calculateInstantaneousPhase(Tuple2<IVector<Double>, IVector<Double>> analyticSignal) {
        IVector<Double> realPart = analyticSignal.getFirst();
        IVector<Double> imagPart = analyticSignal.getSecond();
        
        int n = realPart.length();
        IVector<Double> phase = Linalg.zeros(n);
        
        for (int i = 0; i < n; i++) {
            double real = realPart.get(i);
            double imag = imagPart.get(i);
            phase.set(i, Math.atan2(imag, real));
        }
        
        return phase;
    }
    
    /**
     * 计算瞬时频率 / Calculate instantaneous frequency
     * <p>
     * 计算解析信号的瞬时频率。
     * Calculate instantaneous frequency of analytic signal.
     * </p>
     *
     * @param analyticSignal 解析信号的实部和虚部 / Real and imaginary parts of analytic signal
     * @param samplingRate 采样率 / Sampling rate
     * @return 瞬时频率 / Instantaneous frequency
     */
    public IVector<Double> calculateInstantaneousFrequency(Tuple2<IVector<Double>, IVector<Double>> analyticSignal, double samplingRate) {
        IVector<Double> phase = calculateInstantaneousPhase(analyticSignal);
        
        int n = phase.length();
        IVector<Double> frequency = Linalg.zeros(n);
        
        // 计算相位差分 / Calculate phase difference
        for (int i = 1; i < n; i++) {
            double deltaPhase = phase.get(i) - phase.get(i - 1);
            // 处理相位缠绕 / Handle phase wrapping
            if (deltaPhase > Math.PI) {
                deltaPhase -= 2 * Math.PI;
            } else if (deltaPhase < -Math.PI) {
                deltaPhase += 2 * Math.PI;
            }
            frequency.set(i, deltaPhase * samplingRate / (2 * Math.PI));
        }
        
        // 第一个点使用前向差分 / First point uses forward difference
        if (n > 1) {
            frequency.set(0, frequency.get(1));
        }
        
        return frequency;
    }
    
    @Override
    protected IVector<Double> doProcess(IVector<Double> input) throws SignalProcessingException {
        Tuple2<IVector<Double>, IVector<Double>> result = forward(input);
        // 默认返回解析信号的实部 / Default return real part of analytic signal
        return result.getFirst();
    }
    
    @Override
    public HilbertTransform clone() {
        return new HilbertTransform();
    }
}