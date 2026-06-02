package com.yishape.lab.math.signal.transform;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.signal.core.AbstractSignalProcessor;
import com.yishape.lab.math.signal.core.SignalProcessingException;
import com.yishape.lab.util.Tuple2;

/**
 * 小波变换实现类 / Wavelet Transform Implementation Class
 * <p>
 * 实现离散小波变换（DWT），支持多种小波基函数。
 * 小波变换具有良好的时频局部化特性，适用于非平稳信号分析。
 * </p>
 * <p>
 * Implements Discrete Wavelet Transform (DWT) with support for multiple wavelet basis functions.
 * Wavelet transform has good time-frequency localization characteristics, suitable for non-stationary signal analysis.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class WaveletTransform extends AbstractSignalProcessor<Double> implements ISignalTransform<Double, Tuple2<IVector<Double>, IVector<Double>[]>> {
    
    /**
     * 小波类型枚举 / Wavelet Type Enum
     */
    public enum WaveletType {
        HAAR("Haar"),
        DAUBECHIES_2("Daubechies 2"),
        DAUBECHIES_4("Daubechies 4"),
        DAUBECHIES_6("Daubechies 6"),
        DAUBECHIES_8("Daubechies 8"),
        COIFLET_1("Coiflet 1"),
        COIFLET_2("Coiflet 2"),
        COIFLET_3("Coiflet 3"),
        COIFLET_4("Coiflet 4"),
        COIFLET_5("Coiflet 5"),
        SYMLET_2("Symlet 2"),
        SYMLET_4("Symlet 4"),
        SYMLET_6("Symlet 6"),
        SYMLET_8("Symlet 8");
        
        private final String name;
        
        WaveletType(String name) {
            this.name = name;
        }
        
        public String getName() {
            return name;
        }
    }
    
    private WaveletType waveletType;
    private int levels;
    
    /**
     * 构造函数 / Constructor
     * <p>
     * 使用默认参数创建小波变换。
     * Create wavelet transform with default parameters.
     * </p>
     */
    public WaveletTransform() {
        super("Wavelet Transform", "1.0.0");
        this.waveletType = WaveletType.HAAR;
        this.levels = 1;
    }
    
    /**
     * 构造函数 / Constructor
     * <p>
     * 使用指定参数创建小波变换。
     * Create wavelet transform with specified parameters.
     * </p>
     *
     * @param waveletType 小波类型 / Wavelet type
     * @param levels 分解层数 / Decomposition levels
     */
    public WaveletTransform(WaveletType waveletType, int levels) {
        super("Wavelet Transform", "1.0.0");
        this.waveletType = waveletType;
        this.levels = levels;
    }
    
    /**
     * 正向小波变换 / Forward wavelet transform
     * <p>
     * 计算输入信号的小波变换。
     * Calculate wavelet transform of input signal.
     * </p>
     *
     * @param signal 输入时域信号 / Input time domain signal
     * @return 小波变换结果（近似系数和细节系数） / Wavelet transform result (approximation and detail coefficients)
     * @throws SignalProcessingException 变换过程中发生错误时抛出 / Thrown when errors occur during transform
     */
    @Override
    public Tuple2<IVector<Double>, IVector<Double>[]> forward(IVector<Double> signal) throws SignalProcessingException {
        if (signal == null || signal.length() == 0) {
            throw new SignalProcessingException("输入信号不能为空 / Input signal cannot be empty");
        }
        
        try {
            // 确保信号长度是2的幂 / Ensure signal length is power of 2
            int n = signal.length();
            int paddedLength = nextPowerOfTwo(n);
            
            // 如果需要，对信号进行零填充 / Zero-pad signal if needed
            IVector<Double> paddedSignal = signal;
            if (paddedLength != n) {
                paddedSignal = Linalg.zeros(paddedLength);
                for (int i = 0; i < n; i++) {
                    paddedSignal.set(i, signal.get(i));
                }
            }
            
            // 执行多层小波分解 / Perform multi-level wavelet decomposition
            return decompose(paddedSignal, levels);
        } catch (Exception e) {
            throw new SignalProcessingException("小波变换计算失败 / Wavelet transform calculation failed", e);
        }
    }
    
    /**
     * 小波分解 / Wavelet decomposition
     */
    private Tuple2<IVector<Double>, IVector<Double>[]> decompose(IVector<Double> signal, int levels) {
        int n = signal.length();
        IVector<Double> approx = signal;
        IVector<Double>[] details = new IVector[levels];
        
        // 逐层分解 / Decompose level by level
        for (int level = 0; level < levels; level++) {
            Tuple2<IVector<Double>, IVector<Double>> result = singleLevelDecompose(approx);
            approx = result.getFirst();  // 近似系数 / Approximation coefficients
            details[level] = result.getSecond(); // 细节系数 / Detail coefficients
        }
        
        return new Tuple2<>(approx, details);
    }
    
    /**
     * 单层小波分解 / Single level wavelet decomposition
     */
    private Tuple2<IVector<Double>, IVector<Double>> singleLevelDecompose(IVector<Double> signal) {
        int n = signal.length();
        int halfN = n / 2;
        
        IVector<Double> approx = Linalg.zeros(halfN);
        IVector<Double> detail = Linalg.zeros(halfN);
        
        // 获取小波系数 / Get wavelet coefficients
        double[] lowPass = getLowPassFilter(waveletType);
        double[] highPass = getHighPassFilter(waveletType);
        
        // 下采样卷积 / Downsampled convolution
        for (int i = 0; i < halfN; i++) {
            double approxSum = 0.0;
            double detailSum = 0.0;
            
            for (int j = 0; j < lowPass.length; j++) {
                int index = 2 * i + j - (lowPass.length - 2);
                if (index >= 0 && index < n) {
                    approxSum += signal.get(index) * lowPass[j];
                    detailSum += signal.get(index) * highPass[j];
                }
            }
            
            approx.set(i, approxSum);
            detail.set(i, detailSum);
        }
        
        return new Tuple2<>(approx, detail);
    }
    
    /**
     * 逆小波变换 / Inverse wavelet transform
     * <p>
     * 计算小波变换结果的逆变换。
     * Calculate inverse transform of wavelet transform result.
     * </p>
     *
     * @param transformed 小波变换结果 / Wavelet transform result
     * @return 时域信号 / Time domain signal
     * @throws SignalProcessingException 逆变换过程中发生错误时抛出 / Thrown when errors occur during inverse transform
     */
    @Override
    public IVector<Double> inverse(Tuple2<IVector<Double>, IVector<Double>[]> transformed) throws SignalProcessingException {
        try {
            IVector<Double> approx = transformed.getFirst();
            IVector<Double>[] details = transformed.getSecond();
            
            // 逐层重构 / Reconstruct level by level
            IVector<Double> reconstructed = approx;
            for (int level = details.length - 1; level >= 0; level--) {
                reconstructed = singleLevelReconstruct(reconstructed, details[level]);
            }
            
            return reconstructed;
        } catch (Exception e) {
            throw new SignalProcessingException("逆小波变换计算失败 / Inverse wavelet transform calculation failed", e);
        }
    }
    
    /**
     * 单层小波重构 / Single level wavelet reconstruction
     */
    private IVector<Double> singleLevelReconstruct(IVector<Double> approx, IVector<Double> detail) {
        int halfN = approx.length();
        int n = halfN * 2;
        
        IVector<Double> result = Linalg.zeros(n);
        
        // 获取重构滤波器 / Get reconstruction filters
        double[] lowPassRec = getLowPassReconstructionFilter(waveletType);
        double[] highPassRec = getHighPassReconstructionFilter(waveletType);
        
        // 上采样和卷积 / Upsample and convolve
        for (int i = 0; i < halfN; i++) {
            for (int j = 0; j < lowPassRec.length; j++) {
                int index = 2 * i + j - (lowPassRec.length / 2 - 1);
                if (index >= 0 && index < n) {
                    result.set(index, result.get(index) + approx.get(i) * lowPassRec[j]);
                    result.set(index, result.get(index) + detail.get(i) * highPassRec[j]);
                }
            }
        }
        
        return result;
    }
    
    /**
     * 获取低通滤波器系数 / Get low-pass filter coefficients
     */
    private double[] getLowPassFilter(WaveletType waveletType) {
        switch (waveletType) {
            case HAAR:
                return new double[]{0.7071067811865476, 0.7071067811865476}; // [1/sqrt(2), 1/sqrt(2)]
            case DAUBECHIES_2:
                return new double[]{0.4829629131445341, 0.8365163037378077, 0.2241438680420134, -0.1294095225512604};
            default:
                // 默认使用Haar小波 / Default to Haar wavelet
                return new double[]{0.7071067811865476, 0.7071067811865476};
        }
    }
    
    /**
     * 获取高通滤波器系数 / Get high-pass filter coefficients
     */
    private double[] getHighPassFilter(WaveletType waveletType) {
        switch (waveletType) {
            case HAAR:
                return new double[]{-0.7071067811865476, 0.7071067811865476}; // [-1/sqrt(2), 1/sqrt(2)]
            case DAUBECHIES_2:
                return new double[]{-0.1294095225512604, -0.2241438680420134, 0.8365163037378077, -0.4829629131445341};
            default:
                // 默认使用Haar小波 / Default to Haar wavelet
                return new double[]{-0.7071067811865476, 0.7071067811865476};
        }
    }
    
    /**
     * 获取低通重构滤波器系数 / Get low-pass reconstruction filter coefficients
     */
    private double[] getLowPassReconstructionFilter(WaveletType waveletType) {
        // 对于正交小波，重构滤波器是分析滤波器的时序反转 / For orthogonal wavelets, reconstruction filter is time-reversed analysis filter
        double[] analysis = getLowPassFilter(waveletType);
        double[] reconstruction = new double[analysis.length];
        for (int i = 0; i < analysis.length; i++) {
            reconstruction[i] = analysis[analysis.length - 1 - i];
        }
        return reconstruction;
    }
    
    /**
     * 获取高通重构滤波器系数 / Get high-pass reconstruction filter coefficients
     */
    private double[] getHighPassReconstructionFilter(WaveletType waveletType) {
        // 对于正交小波，重构滤波器是分析滤波器的时序反转 / For orthogonal wavelets, reconstruction filter is time-reversed analysis filter
        double[] analysis = getHighPassFilter(waveletType);
        double[] reconstruction = new double[analysis.length];
        for (int i = 0; i < analysis.length; i++) {
            reconstruction[i] = analysis[analysis.length - 1 - i];
        }
        return reconstruction;
    }
    
    /**
     * 计算下一个2的幂 / Calculate next power of 2
     */
    private int nextPowerOfTwo(int n) {
        if (n <= 0) return 1;
        int power = 1;
        while (power < n) {
            power <<= 1;
        }
        return power;
    }
    
    /**
     * 获取小波类型 / Get wavelet type
     */
    public WaveletType getWaveletType() {
        return waveletType;
    }
    
    /**
     * 设置小波类型 / Set wavelet type
     */
    public void setWaveletType(WaveletType waveletType) {
        this.waveletType = waveletType;
    }
    
    /**
     * 获取分解层数 / Get decomposition levels
     */
    public int getLevels() {
        return levels;
    }
    
    /**
     * 设置分解层数 / Set decomposition levels
     */
    public void setLevels(int levels) {
        this.levels = levels;
    }
    
    @Override
    protected IVector<Double> doProcess(IVector<Double> input) throws SignalProcessingException {
        Tuple2<IVector<Double>, IVector<Double>[]> result = forward(input);
        // 默认返回近似系数 / Default return approximation coefficients
        return result.getFirst();
    }
    
    @Override
    public WaveletTransform clone() {
        return new WaveletTransform(waveletType, levels);
    }
}