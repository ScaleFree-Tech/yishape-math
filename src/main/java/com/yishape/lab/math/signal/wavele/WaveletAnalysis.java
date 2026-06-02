package com.yishape.lab.math.signal.wavele;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.signal.core.Complex;
import com.yishape.lab.math.signal.core.RereFFT;

/**
 * 小波分析类 / Wavelet Analysis Class
 * <p>
 * 提供各种小波变换功能，包括连续小波变换(CWT)、离散小波变换(DWT)、小波包变换等。
 * 使用IVector和IMatrix接口进行向量和矩阵操作，确保与现有代码库的兼容性。
 * </p>
 * <p>
 * Provides various wavelet transform functionality including Continuous Wavelet Transform (CWT),
 * Discrete Wavelet Transform (DWT), Wavelet Packet Transform, etc. Uses IVector and IMatrix
 * interfaces for vector and matrix operations to ensure compatibility with existing codebase.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class WaveletAnalysis {

    /**
     * 小波类型枚举 / Wavelet Type Enum
     */
    public enum WaveletType {
        HAAR,           // Haar小波 / Haar wavelet
        DAUBECHIES,     // Daubechies小波 / Daubechies wavelet
        COIFLETS,       // Coiflets小波 / Coiflets wavelet
        BIORTHOGONAL,   // 双正交小波 / Biorthogonal wavelet
        MORLET,         // Morlet小波 / Morlet wavelet
        MEXICAN_HAT,    // 墨西哥帽小波 / Mexican hat wavelet
        GAUSSIAN        // 高斯小波 / Gaussian wavelet
    }



    /**
     * 连续小波变换 (CWT) — FFT 加速版 / Continuous Wavelet Transform (CWT) — FFT-accelerated
     * <p>
     * 使用 FFT 卷积替代 O(n²) 时域卷积，每尺度复杂度从 O(n²) 降至 O(n log n)。
     * 对小信号 (n &lt; 256) 回退到时域卷积以避免 FFT 填充开销。
     * </p>
     * <p>
     * Uses FFT convolution replacing O(n²) time-domain convolution, reducing per-scale
     * complexity to O(n log n). Falls back to time-domain for small signals (n &lt; 256).
     * </p>
     *
     * @param signal      输入信号向量 / Input signal vector
     * @param waveletType 小波类型 / Wavelet type
     * @param scales      尺度数组 / Scale array
     * @param param       小波参数 / Wavelet parameter
     * @return 小波系数矩阵 / Wavelet coefficient matrix
     */
    public static IMatrix<Double> continuousWaveletTransform(IVector<Double> signal, WaveletType waveletType,
                                                           IVector<Double> scales, double param) {
        int signalLength = signal.length();
        int numScales = scales.length();
        IMatrix<Double> cwt = Linalg.zeros(numScales, signalLength);

        // 小信号回退到时域卷积
        if (signalLength < 256) {
            return cwtTimeDomain(signal, waveletType, scales, param);
        }

        // FFT 长度 = 最小 2 的幂 >= signalLength（循环卷积用）
        int fftSize = RereFFT.nextPowerOfTwoLength(signalLength);

        // 信号 FFT（只做一次）
        Complex[] signalComplex = new Complex[fftSize];
        for (int i = 0; i < signalLength; i++) {
            signalComplex[i] = new Complex(signal.get(i), 0);
        }
        for (int i = signalLength; i < fftSize; i++) {
            signalComplex[i] = Complex.ZERO;
        }
        Complex[] signalFFT = RereFFT.fft(signalComplex);

        // 复用缓冲区
        Complex[] waveletBuf = new Complex[fftSize];
        Complex[] productBuf = new Complex[fftSize];

        for (int s = 0; s < numScales; s++) {
            double scale = scales.get(s);

            // 生成小波并 FFT
            IVector<Double> wavelet = generateWavelet(fftSize, waveletType, scale, param);
            for (int i = 0; i < fftSize; i++) {
                waveletBuf[i] = new Complex(wavelet.get(i), 0);
            }
            Complex[] waveletFFT = RereFFT.fft(waveletBuf);

            // 频域乘法（循环卷积）
            for (int k = 0; k < fftSize; k++) {
                productBuf[k] = new Complex(
                    signalFFT[k].real * waveletFFT[k].real - signalFFT[k].imag * waveletFFT[k].imag,
                    signalFFT[k].real * waveletFFT[k].imag + signalFFT[k].imag * waveletFFT[k].real
                );
            }

            // 逆 FFT
            Complex[] convResult = RereFFT.ifft(productBuf);
            double scaleFactor = 1.0 / Math.sqrt(scale);
            for (int j = 0; j < signalLength; j++) {
                cwt.set(s, j, convResult[j].real * scaleFactor);
            }
        }
        return cwt;
    }

    /** 时域 CWT（小信号回退） */
    private static IMatrix<Double> cwtTimeDomain(IVector<Double> signal, WaveletType waveletType,
                                                  IVector<Double> scales, double param) {
        int signalLength = signal.length();
        int numScales = scales.length();
        IMatrix<Double> cwt = Linalg.zeros(numScales, signalLength);
        for (int i = 0; i < numScales; i++) {
            double scale = scales.get(i);
            IVector<Double> wavelet = generateWavelet(signalLength, waveletType, scale, param);
            for (int j = 0; j < signalLength; j++) {
                double sum = 0;
                for (int k = 0; k < signalLength; k++) {
                    int index = (j - k + signalLength) % signalLength;
                    sum += signal.get(k) * wavelet.get(index);
                }
                cwt.set(i, j, sum / Math.sqrt(scale));
            }
        }
        return cwt;
    }

    /**
     * 离散小波变换 (DWT) / Discrete Wavelet Transform (DWT)
     * <p>
     * 计算信号的离散小波变换，使用Mallat算法。
     * Calculate discrete wavelet transform of signal using Mallat algorithm.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @param waveletType 小波类型 / Wavelet type
     * @param levels 分解层数 / Number of decomposition levels
     * @param param 小波参数 / Wavelet parameter
     * @return 小波系数 / Wavelet coefficients
     */
    public static WaveletCoefficients discreteWaveletTransform(IVector<Double> signal, WaveletType waveletType, 
                                                             int levels, double param) {
        if (levels <= 0) {
            throw new IllegalArgumentException("分解层数必须大于0");
        }
        
        IVector<Double> currentSignal = signal.copy();
        @SuppressWarnings("unchecked")
        IVector<Double>[] details = new IVector[levels];
        
        // 逐层分解 / Decompose level by level
        for (int level = 0; level < levels; level++) {
            WaveletFilters filters = getWaveletFilters(waveletType, param);
            
            // 滤波和下采样 / Filter and downsample
            IVector<Double> approximation = convolveAndDownsample(currentSignal, filters.lowPass);
            IVector<Double> detail = convolveAndDownsample(currentSignal, filters.highPass);
            
            details[level] = detail;
            currentSignal = approximation;
        }
        
        return new WaveletCoefficients(currentSignal, details, levels);
    }

    /**
     * 小波逆变换 (IDWT) / Inverse Discrete Wavelet Transform (IDWT)
     * <p>
     * 从小波系数重建原始信号。
     * Reconstruct original signal from wavelet coefficients.
     * </p>
     *
     * @param coefficients 小波系数 / Wavelet coefficients
     * @param waveletType 小波类型 / Wavelet type
     * @param param 小波参数 / Wavelet parameter
     * @return 重建的信号 / Reconstructed signal
     */
    public static IVector<Double> inverseDiscreteWaveletTransform(WaveletCoefficients coefficients, 
                                                                WaveletType waveletType, double param) {
        IVector<Double> currentSignal = coefficients.approximation;
        
        // 逐层重建 / Reconstruct level by level
        for (int level = coefficients.levels - 1; level >= 0; level--) {
            WaveletFilters filters = getWaveletFilters(waveletType, param);
            
            // 上采样和滤波 / Upsample and filter
            IVector<Double> upsampledApprox = upsampleAndConvolve(currentSignal, filters.lowPassRecon);
            IVector<Double> upsampledDetail = upsampleAndConvolve(coefficients.details[level], filters.highPassRecon);
            
            // 确保两个向量长度一致 / Ensure both vectors have the same length
            int minLength = Math.min(upsampledApprox.length(), upsampledDetail.length());
            upsampledApprox = upsampledApprox.slice(0, minLength);
            upsampledDetail = upsampledDetail.slice(0, minLength);
            
            // 合并近似和细节系数 / Combine approximation and detail coefficients
            currentSignal = upsampledApprox.add(upsampledDetail);
        }
        
        return currentSignal;
    }

    /**
     * 小波包变换 / Wavelet Packet Transform
     * <p>
     * 计算信号的小波包变换，提供更精细的频域分解。
     * Calculate wavelet packet transform of signal, providing finer frequency domain decomposition.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @param waveletType 小波类型 / Wavelet type
     * @param levels 分解层数 / Number of decomposition levels
     * @param param 小波参数 / Wavelet parameter
     * @return 小波包系数树 / Wavelet packet coefficient tree
     */
    public static WaveletPacketTree waveletPacketTransform(IVector<Double> signal, WaveletType waveletType, 
                                                         int levels, double param) {
        WaveletPacketTree tree = new WaveletPacketTree();
        tree.root = new WaveletPacketNode(signal, 0, 0);
        
        // 递归分解 / Recursive decomposition
        decomposeNode(tree.root, waveletType, levels, param);
        
        return tree;
    }

    /**
     * 小波去噪 / Wavelet Denoising
     * <p>
     * 使用小波变换进行信号去噪。
     * Perform signal denoising using wavelet transform.
     * </p>
     *
     * @param noisySignal 含噪信号 / Noisy signal
     * @param waveletType 小波类型 / Wavelet type
     * @param levels 分解层数 / Number of decomposition levels
     * @param threshold 阈值 / Threshold
     * @param param 小波参数 / Wavelet parameter
     * @return 去噪后的信号 / Denoised signal
     */
    public static IVector<Double> waveletDenoising(IVector<Double> noisySignal, WaveletType waveletType, 
                                                  int levels, double threshold, double param) {
        // 小波分解 / Wavelet decomposition
        WaveletCoefficients coeffs = discreteWaveletTransform(noisySignal, waveletType, levels, param);
        
        // 阈值处理 / Threshold processing
        coeffs.approximation = softThreshold(coeffs.approximation, threshold);
        for (int i = 0; i < coeffs.levels; i++) {
            coeffs.details[i] = softThreshold(coeffs.details[i], threshold);
        }
        
        // 小波重建 / Wavelet reconstruction
        return inverseDiscreteWaveletTransform(coeffs, waveletType, param);
    }

    /**
     * 小波压缩 / Wavelet Compression
     * <p>
     * 使用小波变换进行信号压缩。
     * Perform signal compression using wavelet transform.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param waveletType 小波类型 / Wavelet type
     * @param levels 分解层数 / Number of decomposition levels
     * @param compressionRatio 压缩比 / Compression ratio
     * @param param 小波参数 / Wavelet parameter
     * @return 压缩后的信号 / Compressed signal
     */
    public static IVector<Double> waveletCompression(IVector<Double> signal, WaveletType waveletType, 
                                                   int levels, double compressionRatio, double param) {
        // 小波分解 / Wavelet decomposition
        WaveletCoefficients coeffs = discreteWaveletTransform(signal, waveletType, levels, param);
        
        // 计算阈值 / Calculate threshold
        double threshold = calculateCompressionThreshold(coeffs, compressionRatio);
        
        // 阈值处理 / Threshold processing
        coeffs.approximation = hardThreshold(coeffs.approximation, threshold);
        for (int i = 0; i < coeffs.levels; i++) {
            coeffs.details[i] = hardThreshold(coeffs.details[i], threshold);
        }
        
        // 小波重建 / Wavelet reconstruction
        return inverseDiscreteWaveletTransform(coeffs, waveletType, param);
    }

    /**
     * 小波能量分析 / Wavelet Energy Analysis
     * <p>
     * 分析小波系数的能量分布。
     * Analyze energy distribution of wavelet coefficients.
     * </p>
     *
     * @param coefficients 小波系数 / Wavelet coefficients
     * @return 能量分布向量 / Energy distribution vector
     */
    public static IVector<Double> waveletEnergyAnalysis(WaveletCoefficients coefficients) {
        int totalLevels = coefficients.levels + 1;
        IVector<Double> energy = Linalg.zeros(totalLevels);
        
        // 近似系数能量 / Approximation coefficient energy
        energy.set(0, coefficients.approximation.multiply(coefficients.approximation).sumValue());

        // 细节系数能量 / Detail coefficient energy
        for (int i = 0; i < coefficients.levels; i++) {
            energy.set(i + 1, coefficients.details[i].multiply(coefficients.details[i]).sumValue());
        }
        
        return energy;
    }

    /**
     * 小波特征提取 / Wavelet Feature Extraction
     * <p>
     * 从小波系数中提取特征。
     * Extract features from wavelet coefficients.
     * </p>
     *
     * @param coefficients 小波系数 / Wavelet coefficients
     * @return 特征向量 / Feature vector
     */
    public static IVector<Double> waveletFeatureExtraction(WaveletCoefficients coefficients) {
        java.util.List<Double> features = new java.util.ArrayList<>();
        
        // 近似系数特征 / Approximation coefficient features
        features.add(coefficients.approximation.meanValue());
        features.add(coefficients.approximation.stdValue());
        features.add(coefficients.approximation.maxValue());
        features.add(coefficients.approximation.minValue());

        // 细节系数特征 / Detail coefficient features
        for (int i = 0; i < coefficients.levels; i++) {
            features.add(coefficients.details[i].meanValue());
            features.add(coefficients.details[i].stdValue());
            features.add(coefficients.details[i].maxValue());
            features.add(coefficients.details[i].minValue());
        }
        
        // 转换为向量 / Convert to vector
        double[] featureArray = new double[features.size()];
        for (int i = 0; i < features.size(); i++) {
            featureArray[i] = features.get(i);
        }
        
        return Linalg.vector(featureArray);
    }

    // ========== 辅助方法 / Helper Methods ==========

    /**
     * 生成小波函数 / Generate Wavelet Function
     */
    private static IVector<Double> generateWavelet(int length, WaveletType type, double scale, double param) {
        IVector<Double> wavelet = Linalg.zeros(length);
        
        switch (type) {
            case HAAR:
                return generateHaarWavelet(length, scale);
            case DAUBECHIES:
                return generateDaubechiesWavelet(length, scale, (int) param);
            case MORLET:
                return generateMorletWavelet(length, scale, param);
            case MEXICAN_HAT:
                return generateMexicanHatWavelet(length, scale);
            case GAUSSIAN:
                return generateGaussianWavelet(length, scale, param);
            default:
                throw new IllegalArgumentException("不支持的小波类型");
        }
    }

    /**
     * 生成Haar小波 / Generate Haar Wavelet
     */
    private static IVector<Double> generateHaarWavelet(int length, double scale) {
        IVector<Double> wavelet = Linalg.zeros(length);
        int center = length / 2;
        int halfWidth = (int) (length / (4 * scale));
        
        for (int i = 0; i < halfWidth; i++) {
            if (center + i < length) {
                wavelet.set(center + i, 1.0);
            }
        }
        for (int i = 0; i < halfWidth; i++) {
            if (center - i - 1 >= 0) {
                wavelet.set(center - i - 1, -1.0);
            }
        }
        
        return wavelet;
    }

    /**
     * 生成Daubechies小波 / Generate Daubechies Wavelet
     */
    private static IVector<Double> generateDaubechiesWavelet(int length, double scale, int order) {
        // 简化的Daubechies小波实现 / Simplified Daubechies wavelet implementation
        IVector<Double> wavelet = Linalg.zeros(length);
        int center = length / 2;
        
        // 使用简化的滤波器系数 / Use simplified filter coefficients
        double[] coeffs = getDaubechiesCoefficients(order);
        int filterLength = coeffs.length;
        
        for (int i = 0; i < filterLength && center + i < length; i++) {
            wavelet.set(center + i, coeffs[i]);
        }
        
        return wavelet;
    }

    /**
     * 生成Morlet小波 / Generate Morlet Wavelet
     */
    private static IVector<Double> generateMorletWavelet(int length, double scale, double frequency) {
        IVector<Double> wavelet = Linalg.zeros(length);
        int center = length / 2;
        
        for (int i = 0; i < length; i++) {
            double t = (i - center) / scale;
            double gaussian = Math.exp(-t * t / 2);
            double complex = Math.cos(2 * Math.PI * frequency * t);
            wavelet.set(i, gaussian * complex);
        }
        
        return wavelet;
    }

    /**
     * 生成墨西哥帽小波 / Generate Mexican Hat Wavelet
     */
    private static IVector<Double> generateMexicanHatWavelet(int length, double scale) {
        IVector<Double> wavelet = Linalg.zeros(length);
        int center = length / 2;
        
        for (int i = 0; i < length; i++) {
            double t = (i - center) / scale;
            double gaussian = Math.exp(-t * t / 2);
            wavelet.set(i, (1 - t * t) * gaussian);
        }
        
        return wavelet;
    }

    /**
     * 生成高斯小波 / Generate Gaussian Wavelet
     */
    private static IVector<Double> generateGaussianWavelet(int length, double scale, double order) {
        IVector<Double> wavelet = Linalg.zeros(length);
        int center = length / 2;
        
        for (int i = 0; i < length; i++) {
            double t = (i - center) / scale;
            double gaussian = Math.exp(-t * t / 2);
            wavelet.set(i, Math.pow(t, order) * gaussian);
        }
        
        return wavelet;
    }

    /**
     * 获取Daubechies滤波器系数 / Get Daubechies Filter Coefficients
     */
    private static double[] getDaubechiesCoefficients(int order) {
        return com.yishape.lab.math.signal.wavele.WaveletFilters.getDaubechiesLowPass(order);
    }

    /**
     * 小波滤波器结构 / Wavelet Filters Structure
     */
    private static class WaveletFilters {
        double[] lowPass;
        double[] highPass;
        double[] lowPassRecon;
        double[] highPassRecon;
        
        WaveletFilters(double[] lowPass, double[] highPass, double[] lowPassRecon, double[] highPassRecon) {
            this.lowPass = lowPass;
            this.highPass = highPass;
            this.lowPassRecon = lowPassRecon;
            this.highPassRecon = highPassRecon;
        }
    }

    /**
     * 获取小波滤波器 / Get Wavelet Filters
     */
    private static WaveletFilters getWaveletFilters(WaveletType type, double param) {
        switch (type) {
            case HAAR:
                return getHaarFilters();
            case DAUBECHIES:
                return getDaubechiesFilters((int) param);
            default:
                return getHaarFilters(); // 默认使用Haar小波
        }
    }

    /**
     * 获取Haar小波滤波器 / Get Haar Wavelet Filters
     */
    private static WaveletFilters getHaarFilters() {
        double[] lowPass = {0.7071067811865476, 0.7071067811865476};
        double[] highPass = {-0.7071067811865476, 0.7071067811865476};
        return new WaveletFilters(lowPass, highPass, lowPass, highPass);
    }

    /**
     * 获取Daubechies小波滤波器 / Get Daubechies Wavelet Filters
     */
    private static WaveletFilters getDaubechiesFilters(int order) {
        double[] lowPass = getDaubechiesCoefficients(order);
        double[] highPass = new double[lowPass.length];
        for (int i = 0; i < lowPass.length; i++) {
            highPass[i] = Math.pow(-1, i) * lowPass[lowPass.length - 1 - i];
        }
        return new WaveletFilters(lowPass, highPass, lowPass, highPass);
    }

    /**
     * 卷积和下采样 / Convolve and Downsample
     */
    private static IVector<Double> convolveAndDownsample(IVector<Double> signal, double[] filter) {
        int signalLength = signal.length();
        int filterLength = filter.length;
        int outputLength = (signalLength + filterLength - 1) / 2;
        
        IVector<Double> result = Linalg.zeros(outputLength);
        
        for (int i = 0; i < outputLength; i++) {
            double sum = 0;
            for (int j = 0; j < filterLength; j++) {
                int index = 2 * i - j;
                if (index >= 0 && index < signalLength) {
                    sum += signal.get(index) * filter[j];
                }
            }
            result.set(i, sum);
        }
        
        return result;
    }

    /**
     * 上采样和卷积 / Upsample and Convolve
     */
    private static IVector<Double> upsampleAndConvolve(IVector<Double> signal, double[] filter) {
        int signalLength = signal.length();
        int filterLength = filter.length;
        
        // 上采样：在每两个样本之间插入0 / Upsample: insert zeros between samples
        IVector<Double> upsampled = Linalg.zeros(signalLength * 2);
        for (int i = 0; i < signalLength; i++) {
            upsampled.set(2 * i, signal.get(i));
        }
        
        // 卷积：考虑滤波器长度的影响 / Convolve: consider filter length effect
        int outputLength = signalLength * 2;
        IVector<Double> result = Linalg.zeros(outputLength);
        
        for (int i = 0; i < outputLength; i++) {
            double sum = 0;
            for (int j = 0; j < filterLength; j++) {
                int index = i - j;
                if (index >= 0 && index < upsampled.length()) {
                    sum += upsampled.get(index) * filter[j];
                }
            }
            result.set(i, sum);
        }
        
        return result;
    }

    /**
     * 软阈值处理 / Soft Threshold Processing
     */
    private static IVector<Double> softThreshold(IVector<Double> signal, double threshold) {
        IVector<Double> result = Linalg.zeros(signal.length());
        for (int i = 0; i < signal.length(); i++) {
            double value = signal.get(i);
            if (Math.abs(value) > threshold) {
                result.set(i, Math.signum(value) * (Math.abs(value) - threshold));
            } else {
                result.set(i, 0.0);
            }
        }
        return result;
    }

    /**
     * 硬阈值处理 / Hard Threshold Processing
     */
    private static IVector<Double> hardThreshold(IVector<Double> signal, double threshold) {
        IVector<Double> result = Linalg.zeros(signal.length());
        for (int i = 0; i < signal.length(); i++) {
            double value = signal.get(i);
            result.set(i, Math.abs(value) > threshold ? value : 0.0);
        }
        return result;
    }

    /**
     * 计算压缩阈值 / Calculate Compression Threshold
     */
    private static double calculateCompressionThreshold(WaveletCoefficients coeffs, double compressionRatio) {
        // 收集所有系数 / Collect all coefficients
        java.util.List<Double> allCoeffs = new java.util.ArrayList<>();
        
        for (int i = 0; i < coeffs.approximation.length(); i++) {
            allCoeffs.add(Math.abs(coeffs.approximation.get(i)));
        }
        
        for (int level = 0; level < coeffs.levels; level++) {
            for (int i = 0; i < coeffs.details[level].length(); i++) {
                allCoeffs.add(Math.abs(coeffs.details[level].get(i)));
            }
        }
        
        // 排序 / Sort
        allCoeffs.sort(java.util.Collections.reverseOrder());
        
        // 计算阈值 / Calculate threshold
        int thresholdIndex = (int) (allCoeffs.size() * compressionRatio);
        return thresholdIndex < allCoeffs.size() ? allCoeffs.get(thresholdIndex) : 0.0;
    }

    /**
     * 分解节点 / Decompose Node
     */
    private static void decomposeNode(WaveletPacketNode node, WaveletType type, int maxLevels, double param) {
        if (node.level >= maxLevels || node.signal.length() < 4) {
            return;
        }
        
        WaveletFilters filters = getWaveletFilters(type, param);
        
        // 分解为近似和细节 / Decompose into approximation and detail
        node.approximation = convolveAndDownsample(node.signal, filters.lowPass);
        node.detail = convolveAndDownsample(node.signal, filters.highPass);
        
        // 递归分解子节点 / Recursively decompose child nodes
        if (node.approximation.length() >= 4) {
            node.approximationNode = new WaveletPacketNode(node.approximation, node.level + 1, node.index * 2);
            decomposeNode(node.approximationNode, type, maxLevels, param);
        }
        
        if (node.detail.length() >= 4) {
            node.detailNode = new WaveletPacketNode(node.detail, node.level + 1, node.index * 2 + 1);
            decomposeNode(node.detailNode, type, maxLevels, param);
        }
    }

    /**
     * 小波包节点类 / Wavelet Packet Node Class
     */
    public static class WaveletPacketNode {
        public IVector<Double> signal;
        public IVector<Double> approximation;
        public IVector<Double> detail;
        public WaveletPacketNode approximationNode;
        public WaveletPacketNode detailNode;
        public int level;
        public int index;
        
        public WaveletPacketNode(IVector<Double> signal, int level, int index) {
            this.signal = signal;
            this.level = level;
            this.index = index;
        }
    }

    /**
     * 小波包树类 / Wavelet Packet Tree Class
     */
    public static class WaveletPacketTree {
        public WaveletPacketNode root;
        
        public WaveletPacketTree() {
            this.root = null;
        }
    }
}
