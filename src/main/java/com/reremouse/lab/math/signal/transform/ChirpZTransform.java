package com.reremouse.lab.math.signal.transform;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.signal.Complex;
import com.reremouse.lab.math.signal.RereFFT;
import com.reremouse.lab.math.signal.core.AbstractSignalProcessor;
import com.reremouse.lab.math.signal.core.SignalProcessingException;

/**
 * Chirp-Z变换实现类 / Chirp-Z Transform Implementation Class
 * <p>
 * 实现Chirp-Z变换，这是一种高效计算Z变换在任意复平面路径上的算法。
 * 特别适用于高分辨率频谱分析和任意频率范围的DFT计算。
 * </p>
 * <p>
 * Implements Chirp-Z Transform, an efficient algorithm for computing Z-transform along arbitrary paths in complex plane.
 * Particularly useful for high-resolution spectral analysis and DFT computation over arbitrary frequency ranges.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ChirpZTransform extends AbstractSignalProcessor<Double> implements ISignalTransform<Double, Complex[]> {
    
    private Complex startPoint;      // 起始点 A / Starting point A
    private Complex stepFactor;      // 步进因子 W / Step factor W
    private int numOutputPoints;     // 输出点数 M / Number of output points
    private boolean useOptimizedFFT; // 是否使用优化的FFT / Whether to use optimized FFT
    
    /**
     * 构造函数 / Constructor
     * <p>
     * 使用默认参数初始化Chirp-Z变换。
     * Initialize Chirp-Z transform with default parameters.
     * </p>
     */
    public ChirpZTransform() {
        super("Chirp-Z Transform", "1.0.0");
        this.startPoint = new Complex(1.0, 0.0);  // A = 1
        this.stepFactor = new Complex(Math.cos(-2 * Math.PI / 1024), Math.sin(-2 * Math.PI / 1024)); // W = e^(-j2π/1024)
        this.numOutputPoints = 1024;
        this.useOptimizedFFT = true;
    }
    
    /**
     * 构造函数 / Constructor
     * <p>
     * 使用指定参数初始化Chirp-Z变换。
     * Initialize Chirp-Z transform with specified parameters.
     * </p>
     *
     * @param startPoint 起始点 / Starting point
     * @param stepFactor 步进因子 / Step factor
     * @param numOutputPoints 输出点数 / Number of output points
     */
    public ChirpZTransform(Complex startPoint, Complex stepFactor, int numOutputPoints) {
        super("Chirp-Z Transform", "1.0.0");
        this.startPoint = startPoint;
        this.stepFactor = stepFactor;
        this.numOutputPoints = numOutputPoints;
        this.useOptimizedFFT = true;
    }
    
    /**
     * 为频率范围创建Chirp-Z变换 / Create Chirp-Z transform for frequency range
     * <p>
     * 创建用于分析指定频率范围的Chirp-Z变换实例。
     * Create Chirp-Z transform instance for analyzing specified frequency range.
     * </p>
     *
     * @param startFreq 起始频率 (归一化) / Start frequency (normalized)
     * @param endFreq 结束频率 (归一化) / End frequency (normalized)
     * @param numPoints 频率点数 / Number of frequency points
     * @return Chirp-Z变换实例 / Chirp-Z transform instance
     */
    public static ChirpZTransform forFrequencyRange(double startFreq, double endFreq, int numPoints) {
        // A = e^(j*2π*startFreq)
        Complex startPoint = new Complex(Math.cos(2 * Math.PI * startFreq), Math.sin(2 * Math.PI * startFreq));
        
        // W = e^(-j*2π*(endFreq-startFreq)/(numPoints-1))
        double deltaFreq = (endFreq - startFreq) / (numPoints - 1);
        Complex stepFactor = new Complex(Math.cos(-2 * Math.PI * deltaFreq), Math.sin(-2 * Math.PI * deltaFreq));
        
        return new ChirpZTransform(startPoint, stepFactor, numPoints);
    }
    
    /**
     * 为对数频率扫描创建Chirp-Z变换 / Create Chirp-Z transform for logarithmic frequency sweep
     * <p>
     * 创建用于对数频率扫描的Chirp-Z变换实例，适用于宽带信号分析。
     * Create Chirp-Z transform instance for logarithmic frequency sweep, suitable for wideband signal analysis.
     * </p>
     *
     * @param startFreq 起始频率 / Start frequency
     * @param endFreq 结束频率 / End frequency
     * @param numPoints 频率点数 / Number of frequency points
     * @param samplingRate 采样率 / Sampling rate
     * @return Chirp-Z变换实例 / Chirp-Z transform instance
     */
    public static ChirpZTransform forLogFrequencyRange(double startFreq, double endFreq, int numPoints, double samplingRate) {
        double normStartFreq = startFreq / samplingRate;
        double normEndFreq = endFreq / samplingRate;
        
        // 对数频率步进 / Logarithmic frequency step
        double logStep = Math.log(normEndFreq / normStartFreq) / (numPoints - 1);
        
        Complex startPoint = new Complex(Math.cos(2 * Math.PI * normStartFreq), Math.sin(2 * Math.PI * normStartFreq));
        
        // 这里简化为线性步进，实际应用中需要更复杂的对数步进实现
        // Simplified to linear step here, actual implementation needs more complex logarithmic step
        double deltaFreq = (normEndFreq - normStartFreq) / (numPoints - 1);
        Complex stepFactor = new Complex(Math.cos(-2 * Math.PI * deltaFreq), Math.sin(-2 * Math.PI * deltaFreq));
        
        return new ChirpZTransform(startPoint, stepFactor, numPoints);
    }
    
    /**
     * 计算Chirp-Z变换 / Calculate Chirp-Z transform
     * <p>
     * 使用高效的卷积算法计算Chirp-Z变换。
     * Calculate Chirp-Z transform using efficient convolution algorithm.
     * </p>
     *
     * @param signal 输入时域信号 / Input time domain signal
     * @return Chirp-Z变换结果 / Chirp-Z transform result
     * @throws SignalProcessingException 变换过程中发生错误时抛出 / Thrown when errors occur during transform
     */
    @Override
    public Complex[] forward(IVector<Double> signal) throws SignalProcessingException {
        if (signal == null || signal.length() == 0) {
            throw new SignalProcessingException("输入信号不能为空 / Input signal cannot be empty");
        }
        
        int N = signal.length();
        int M = numOutputPoints;
        
        if (useOptimizedFFT) {
            return computeUsingFFT(signal, N, M);
        } else {
            return computeDirectly(signal, N, M);
        }
    }
    
    /**
     * 使用FFT的高效算法 / Efficient algorithm using FFT
     * <p>
     * 使用Rabiner-Schafer-Rader算法，通过FFT和卷积实现高效的Chirp-Z变换。
     * Use Rabiner-Schafer-Rader algorithm to implement efficient Chirp-Z transform via FFT and convolution.
     * </p>
     */
    private Complex[] computeUsingFFT(IVector<Double> signal, int N, int M) throws SignalProcessingException {
        // 第一步：预乘chirp序列 / Step 1: Pre-multiply by chirp sequence
        Complex[] preMult = new Complex[N];
        for (int n = 0; n < N; n++) {
            // z^n = e^(n * ln(z))
            Complex logStartPoint = startPoint.log();
            Complex logStepFactor = stepFactor.log();
            
            Complex chirp = logStartPoint.scale(n).add(logStepFactor.scale(n * n / 2.0)).exp();
            preMult[n] = new Complex(signal.get(n), 0).multiply(chirp);
        }
        
        // 第二步：卷积运算 / Step 2: Convolution operation
        int L = N + M - 1;
        int convSize = nextPowerOfTwo(L);
        
        // 构造卷积核 / Construct convolution kernel
        Complex[] kernel = new Complex[convSize];
        for (int i = 0; i < convSize; i++) {
            kernel[i] = new Complex(0, 0);
        }
        
        for (int k = 0; k < M; k++) {
            // W^(-k*k/2) = e^(-k*k/2 * ln(W))
            Complex logStepFactor = stepFactor.log();
            Complex chirp = logStepFactor.scale(-k * k / 2.0).exp();
            kernel[k] = chirp;
        }
        
        for (int k = 1; k < N; k++) {
            // W^(-k*k/2) = e^(-k*k/2 * ln(W))
            Complex logStepFactor = stepFactor.log();
            Complex chirp = logStepFactor.scale(-k * k / 2.0).exp();
            kernel[convSize - k] = chirp;
        }
        
        // 零填充预乘结果 / Zero-pad pre-multiplied result
        Complex[] paddedSignal = new Complex[convSize];
        for (int i = 0; i < N; i++) {
            paddedSignal[i] = preMult[i];
        }
        for (int i = N; i < convSize; i++) {
            paddedSignal[i] = new Complex(0, 0);
        }
        
        // 进行FFT卷积 / Perform FFT convolution
        Complex[] signalFFT = RereFFT.fft(paddedSignal);
        Complex[] kernelFFT = RereFFT.fft(kernel);
        
        Complex[] convolutionFFT = new Complex[convSize];
        for (int i = 0; i < convSize; i++) {
            convolutionFFT[i] = signalFFT[i].multiply(kernelFFT[i]);
        }
        
        Complex[] convolutionResult = RereFFT.ifft(convolutionFFT);
        
        // 第三步：后乘chirp序列 / Step 3: Post-multiply by chirp sequence
        Complex[] result = new Complex[M];
        for (int k = 0; k < M; k++) {
            // W^(-k*k/2) = e^(-k*k/2 * ln(W))
            Complex logStepFactor = stepFactor.log();
            Complex chirp = logStepFactor.scale(-k * k / 2.0).exp();
            result[k] = convolutionResult[k].multiply(chirp);
        }
        
        return result;
    }
    
    /**
     * 直接计算方法 / Direct computation method
     * <p>
     * 直接根据定义计算Chirp-Z变换，适用于小规模数据。
     * Direct computation according to definition, suitable for small-scale data.
     * </p>
     */
    private Complex[] computeDirectly(IVector<Double> signal, int N, int M) {
        Complex[] result = new Complex[M];
        
        for (int k = 0; k < M; k++) {
            Complex sum = new Complex(0, 0);
            
            for (int n = 0; n < N; n++) {
                // z_k = A * W^(-k)
                Complex logStepFactor = stepFactor.log();
                Complex zk = startPoint.multiply(logStepFactor.scale(-k).exp());
                
                // X(z_k) = ∑_{n=0}^{N-1} x[n] * z_k^(-n)
                Complex zkPowerN = zk.power(-n);
                sum = sum.add(zkPowerN.scale(signal.get(n)));
            }
            
            result[k] = sum;
        }
        
        return result;
    }
    
    /**
     * 计算逆Chirp-Z变换 / Calculate inverse Chirp-Z transform
     * <p>
     * 通过逆向Chirp-Z变换恢复时域信号。
     * Recover time domain signal through inverse Chirp-Z transform.
     * </p>
     *
     * @param transformed Chirp-Z变换结果 / Chirp-Z transform result
     * @return 时域信号 / Time domain signal
     * @throws SignalProcessingException 逆变换过程中发生错误时抛出 / Thrown when errors occur during inverse transform
     */
    @Override
    public IVector<Double> inverse(Complex[] transformed) throws SignalProcessingException {
        // 逆Chirp-Z变换的实现较为复杂，这里提供一个简化版本
        // Implementation of inverse Chirp-Z transform is complex, simplified version provided here
        
        int M = transformed.length;
        IVector<Double> result = Linalg.zeros(M);
        
        // 如果变换参数满足特定条件，可以使用类似的算法
        // If transform parameters satisfy specific conditions, similar algorithm can be used
        
        // 简化实现：假设为DFT的逆变换
        // Simplified implementation: assume inverse of DFT
        for (int n = 0; n < M; n++) {
            Complex sum = new Complex(0, 0);
            for (int k = 0; k < M; k++) {
                double angle = 2 * Math.PI * k * n / M;
                Complex expTerm = new Complex(Math.cos(angle), Math.sin(angle));
                sum = sum.add(transformed[k].multiply(expTerm));
            }
            result.set(n, sum.real / M);
        }
        
        return result;
    }
    
    /**
     * 计算高分辨率频谱 / Calculate high-resolution spectrum
     * <p>
     * 使用Chirp-Z变换计算指定频率范围内的高分辨率频谱。
     * Calculate high-resolution spectrum within specified frequency range using Chirp-Z transform.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param startFreq 起始频率 / Start frequency
     * @param endFreq 结束频率 / End frequency
     * @param numPoints 频率点数 / Number of frequency points
     * @param samplingRate 采样率 / Sampling rate
     * @return 高分辨率频谱 / High-resolution spectrum
     * @throws SignalProcessingException 计算过程中发生错误时抛出 / Thrown when errors occur during calculation
     */
    public static Complex[] computeHighResolutionSpectrum(IVector<Double> signal, double startFreq, double endFreq, 
                                                        int numPoints, double samplingRate) throws SignalProcessingException {
        ChirpZTransform czt = ChirpZTransform.forFrequencyRange(startFreq / samplingRate, endFreq / samplingRate, numPoints);
        return czt.forward(signal);
    }
    
    @Override
    protected IVector<Double> doProcess(IVector<Double> input) throws SignalProcessingException {
        // 默认处理：计算Chirp-Z变换的幅度谱
        Complex[] cztResult = forward(input);
        IVector<Double> magnitude = Linalg.zeros(cztResult.length);
        for (int i = 0; i < cztResult.length; i++) {
            magnitude.set(i, cztResult[i].magnitude());
        }
        return magnitude;
    }
    
    /**
     * 计算大于等于n的最小2的幂 / Calculate smallest power of 2 >= n
     */
    private int nextPowerOfTwo(int n) {
        if (n <= 0) return 1;
        if ((n & (n - 1)) == 0) return n;
        
        int power = 1;
        while (power < n) {
            power <<= 1;
        }
        return power;
    }
    
    @Override
    public ChirpZTransform clone() {
        return new ChirpZTransform(startPoint, stepFactor, numOutputPoints);
    }
    
    // Getters and setters
    public Complex getStartPoint() { return startPoint; }
    public void setStartPoint(Complex startPoint) { this.startPoint = startPoint; }
    
    public Complex getStepFactor() { return stepFactor; }
    public void setStepFactor(Complex stepFactor) { this.stepFactor = stepFactor; }
    
    public int getNumOutputPoints() { return numOutputPoints; }
    public void setNumOutputPoints(int numOutputPoints) { this.numOutputPoints = numOutputPoints; }
    
    public boolean isUseOptimizedFFT() { return useOptimizedFFT; }
    public void setUseOptimizedFFT(boolean useOptimizedFFT) { this.useOptimizedFFT = useOptimizedFFT; }
}