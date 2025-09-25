package com.reremouse.lab.math.signal.wavele;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IMatrix;

/**
 * 小波工具函数类 / Wavelet Utilities Class
 * <p>
 * 提供各种小波处理工具函数，包括去噪、压缩、特征提取、阈值选择等。
 * 使用IVector和IMatrix接口进行向量和矩阵操作，确保与现有代码库的兼容性。
 * </p>
 * <p>
 * Provides various wavelet processing utility functions including denoising, compression,
 * feature extraction, threshold selection, etc. Uses IVector and IMatrix interfaces for
 * vector and matrix operations to ensure compatibility with existing codebase.
 * </p>
 *
 * @author lter2
 * @version 1.0
 * @since 1.0
 */
public class WaveletUtilities {

    /**
     * 阈值选择方法枚举 / Threshold Selection Method Enum
     */
    public enum ThresholdMethod {
        RIGRSURE,       // Rigrsure阈值 / Rigrsure threshold
        SURE,           // SURE阈值 / SURE threshold
        HEURSURE,       // Heursure阈值 / Heursure threshold
        MINIMAX,        // Minimax阈值 / Minimax threshold
        FIXED           // 固定阈值 / Fixed threshold
    }

    /**
     * 阈值处理类型枚举 / Threshold Processing Type Enum
     */
    public enum ThresholdType {
        SOFT,           // 软阈值 / Soft threshold
        HARD            // 硬阈值 / Hard threshold
    }

    /**
     * 小波去噪 / Wavelet Denoising
     * <p>
     * 使用小波变换进行信号去噪，支持多种阈值选择方法。
     * Perform signal denoising using wavelet transform with multiple threshold selection methods.
     * </p>
     *
     * @param noisySignal 含噪信号 / Noisy signal
     * @param waveletType 小波类型 / Wavelet type
     * @param levels 分解层数 / Number of decomposition levels
     * @param thresholdMethod 阈值选择方法 / Threshold selection method
     * @param thresholdType 阈值处理类型 / Threshold processing type
     * @param param 小波参数 / Wavelet parameter
     * @return 去噪后的信号 / Denoised signal
     */
    public static IVector<Double> waveletDenoising(IVector<Double> noisySignal, 
                                                 WaveletAnalysis.WaveletType waveletType,
                                                 int levels, ThresholdMethod thresholdMethod,
                                                 ThresholdType thresholdType, double param) {
        // 小波分解 / Wavelet decomposition
        WaveletCoefficients coeffs = WaveletAnalysis.discreteWaveletTransform(
                noisySignal, waveletType, levels, param);
        
        // 计算阈值 / Calculate threshold
        double threshold = calculateThreshold(coeffs, thresholdMethod);
        
        // 阈值处理 / Threshold processing
        coeffs.approximation = applyThreshold(coeffs.approximation, threshold, thresholdType);
        for (int i = 0; i < coeffs.levels; i++) {
            coeffs.details[i] = applyThreshold(coeffs.details[i], threshold, thresholdType);
        }
        
        // 小波重建 / Wavelet reconstruction
        return WaveletAnalysis.inverseDiscreteWaveletTransform(coeffs, waveletType, param);
    }

    /**
     * 自适应小波去噪 / Adaptive Wavelet Denoising
     * <p>
     * 使用自适应阈值进行小波去噪。
     * Perform adaptive wavelet denoising using adaptive threshold.
     * </p>
     *
     * @param noisySignal 含噪信号 / Noisy signal
     * @param waveletType 小波类型 / Wavelet type
     * @param levels 分解层数 / Number of decomposition levels
     * @param param 小波参数 / Wavelet parameter
     * @return 去噪后的信号 / Denoised signal
     */
    public static IVector<Double> adaptiveWaveletDenoising(IVector<Double> noisySignal,
                                                         WaveletAnalysis.WaveletType waveletType,
                                                         int levels, double param) {
        // 小波分解 / Wavelet decomposition
        WaveletCoefficients coeffs = WaveletAnalysis.discreteWaveletTransform(
                noisySignal, waveletType, levels, param);
        
        // 自适应阈值处理 / Adaptive threshold processing
        coeffs.approximation = adaptiveThreshold(coeffs.approximation);
        for (int i = 0; i < coeffs.levels; i++) {
            coeffs.details[i] = adaptiveThreshold(coeffs.details[i]);
        }
        
        // 小波重建 / Wavelet reconstruction
        return WaveletAnalysis.inverseDiscreteWaveletTransform(coeffs, waveletType, param);
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
    public static IVector<Double> waveletCompression(IVector<Double> signal,
                                                   WaveletAnalysis.WaveletType waveletType,
                                                   int levels, double compressionRatio,
                                                   double param) {
        // 小波分解 / Wavelet decomposition
        WaveletCoefficients coeffs = WaveletAnalysis.discreteWaveletTransform(
                signal, waveletType, levels, param);
        
        // 计算压缩阈值 / Calculate compression threshold
        double threshold = calculateCompressionThreshold(coeffs, compressionRatio);
        
        // 硬阈值处理 / Hard threshold processing
        coeffs.approximation = applyThreshold(coeffs.approximation, threshold, ThresholdType.HARD);
        for (int i = 0; i < coeffs.levels; i++) {
            coeffs.details[i] = applyThreshold(coeffs.details[i], threshold, ThresholdType.HARD);
        }
        
        // 小波重建 / Wavelet reconstruction
        return WaveletAnalysis.inverseDiscreteWaveletTransform(coeffs, waveletType, param);
    }

    /**
     * 小波特征提取 / Wavelet Feature Extraction
     * <p>
     * 从小波系数中提取统计特征。
     * Extract statistical features from wavelet coefficients.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param waveletType 小波类型 / Wavelet type
     * @param levels 分解层数 / Number of decomposition levels
     * @param param 小波参数 / Wavelet parameter
     * @return 特征向量 / Feature vector
     */
    public static IVector<Double> extractWaveletFeatures(IVector<Double> signal,
                                                       WaveletAnalysis.WaveletType waveletType,
                                                       int levels, double param) {
        // 小波分解 / Wavelet decomposition
        WaveletCoefficients coeffs = WaveletAnalysis.discreteWaveletTransform(
                signal, waveletType, levels, param);
        
        java.util.List<Double> features = new java.util.ArrayList<>();
        
        // 近似系数特征 / Approximation coefficient features
        features.addAll(extractCoefficientFeatures(coeffs.approximation, "approx"));
        
        // 细节系数特征 / Detail coefficient features
        for (int i = 0; i < coeffs.levels; i++) {
            features.addAll(extractCoefficientFeatures(coeffs.details[i], "detail" + (i + 1)));
        }
        
        // 能量特征 / Energy features
        features.addAll(extractEnergyFeatures(coeffs));
        
        // 熵特征 / Entropy features
        features.addAll(extractEntropyFeatures(coeffs));
        
        // 转换为向量 / Convert to vector
        double[] featureArray = new double[features.size()];
        for (int i = 0; i < features.size(); i++) {
            featureArray[i] = features.get(i);
        }
        
        return Linalg.vector(featureArray);
    }

    /**
     * 小波能量分析 / Wavelet Energy Analysis
     * <p>
     * 分析小波系数的能量分布。
     * Analyze energy distribution of wavelet coefficients.
     * </p>
     *
     * @param coeffs 小波系数 / Wavelet coefficients
     * @return 能量分布向量 / Energy distribution vector
     */
    public static IVector<Double> analyzeWaveletEnergy(WaveletCoefficients coeffs) {
        int totalLevels = coeffs.levels + 1;
        IVector<Double> energy = Linalg.zeros(totalLevels);
        
        // 近似系数能量 / Approximation coefficient energy
        energy.set(0, coeffs.approximation.multiply(coeffs.approximation).sum());
        
        // 细节系数能量 / Detail coefficient energy
        for (int i = 0; i < coeffs.levels; i++) {
            energy.set(i + 1, coeffs.details[i].multiply(coeffs.details[i]).sum());
        }
        
        return energy;
    }

    /**
     * 小波熵分析 / Wavelet Entropy Analysis
     * <p>
     * 分析小波系数的熵分布。
     * Analyze entropy distribution of wavelet coefficients.
     * </p>
     *
     * @param coeffs 小波系数 / Wavelet coefficients
     * @return 熵分布向量 / Entropy distribution vector
     */
    public static IVector<Double> analyzeWaveletEntropy(WaveletCoefficients coeffs) {
        int totalLevels = coeffs.levels + 1;
        IVector<Double> entropy = Linalg.zeros(totalLevels);
        
        // 近似系数熵 / Approximation coefficient entropy
        entropy.set(0, calculateEntropy(coeffs.approximation));
        
        // 细节系数熵 / Detail coefficient entropy
        for (int i = 0; i < coeffs.levels; i++) {
            entropy.set(i + 1, calculateEntropy(coeffs.details[i]));
        }
        
        return entropy;
    }

    /**
     * 小波尺度图 / Wavelet Scalogram
     * <p>
     * 计算小波尺度图，用于时频分析。
     * Calculate wavelet scalogram for time-frequency analysis.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param waveletType 小波类型 / Wavelet type
     * @param scales 尺度数组 / Scale array
     * @param param 小波参数 / Wavelet parameter
     * @return 尺度图矩阵 / Scalogram matrix
     */
    public static IMatrix<Double> calculateScalogram(IVector<Double> signal,
                                                   WaveletAnalysis.WaveletType waveletType,
                                                   IVector<Double> scales, double param) {
        return WaveletAnalysis.continuousWaveletTransform(signal, waveletType, scales, param);
    }

    /**
     * 小波相干性分析 / Wavelet Coherence Analysis
     * <p>
     * 分析两个信号的小波相干性。
     * Analyze wavelet coherence between two signals.
     * </p>
     *
     * @param signal1 第一个信号 / First signal
     * @param signal2 第二个信号 / Second signal
     * @param waveletType 小波类型 / Wavelet type
     * @param scales 尺度数组 / Scale array
     * @param param 小波参数 / Wavelet parameter
     * @return 相干性矩阵 / Coherence matrix
     */
    public static IMatrix<Double> calculateWaveletCoherence(IVector<Double> signal1, IVector<Double> signal2,
                                                          WaveletAnalysis.WaveletType waveletType,
                                                          IVector<Double> scales, double param) {
        // 计算两个信号的小波变换 / Calculate wavelet transforms of both signals
        IMatrix<Double> cwt1 = WaveletAnalysis.continuousWaveletTransform(signal1, waveletType, scales, param);
        IMatrix<Double> cwt2 = WaveletAnalysis.continuousWaveletTransform(signal2, waveletType, scales, param);
        
        int rows = cwt1.rows();
        int cols = cwt1.cols();
        IMatrix<Double> coherence = Linalg.zeros(rows, cols);
        
        // 计算相干性 / Calculate coherence
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double cwt1Val = cwt1.get(i, j);
                double cwt2Val = cwt2.get(i, j);
                
                // 计算复数的相干性 / Calculate coherence of complex values
                double numerator = Math.abs(cwt1Val * cwt2Val);
                double denominator = Math.sqrt(cwt1Val * cwt1Val + cwt2Val * cwt2Val);
                
                if (denominator > 0) {
                    coherence.set(i, j, numerator / denominator);
                } else {
                    coherence.set(i, j, 0.0);
                }
            }
        }
        
        return coherence;
    }

    /**
     * 小波去趋势 / Wavelet Detrending
     * <p>
     * 使用小波变换去除信号中的趋势。
     * Remove trends from signal using wavelet transform.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param waveletType 小波类型 / Wavelet type
     * @param levels 分解层数 / Number of decomposition levels
     * @param param 小波参数 / Wavelet parameter
     * @return 去趋势后的信号 / Detrended signal
     */
    public static IVector<Double> waveletDetrending(IVector<Double> signal,
                                                  WaveletAnalysis.WaveletType waveletType,
                                                  int levels, double param) {
        // 小波分解 / Wavelet decomposition
        WaveletCoefficients coeffs = WaveletAnalysis.discreteWaveletTransform(
                signal, waveletType, levels, param);
        
        // 只保留细节系数，去除近似系数（趋势） / Keep only detail coefficients, remove approximation (trend)
        coeffs.approximation = Linalg.zeros(coeffs.approximation.length());
        
        // 小波重建 / Wavelet reconstruction
        return WaveletAnalysis.inverseDiscreteWaveletTransform(coeffs, waveletType, param);
    }

    /**
     * 小波平滑 / Wavelet Smoothing
     * <p>
     * 使用小波变换进行信号平滑。
     * Perform signal smoothing using wavelet transform.
     * </p>
     *
     * @param signal 输入信号 / Input signal
     * @param waveletType 小波类型 / Wavelet type
     * @param levels 分解层数 / Number of decomposition levels
     * @param param 小波参数 / Wavelet parameter
     * @return 平滑后的信号 / Smoothed signal
     */
    public static IVector<Double> waveletSmoothing(IVector<Double> signal,
                                                WaveletAnalysis.WaveletType waveletType,
                                                int levels, double param) {
        // 小波分解 / Wavelet decomposition
        WaveletCoefficients coeffs = WaveletAnalysis.discreteWaveletTransform(
                signal, waveletType, levels, param);
        
        // 只保留近似系数，去除细节系数（噪声） / Keep only approximation coefficients, remove details (noise)
        for (int i = 0; i < coeffs.levels; i++) {
            coeffs.details[i] = Linalg.zeros(coeffs.details[i].length());
        }
        
        // 小波重建 / Wavelet reconstruction
        return WaveletAnalysis.inverseDiscreteWaveletTransform(coeffs, waveletType, param);
    }

    // ========== 辅助方法 / Helper Methods ==========

    /**
     * 计算阈值 / Calculate Threshold
     */
    private static double calculateThreshold(WaveletCoefficients coeffs, ThresholdMethod method) {
        switch (method) {
            case RIGRSURE:
                return calculateRigrsureThreshold(coeffs);
            case SURE:
                return calculateSureThreshold(coeffs);
            case HEURSURE:
                return calculateHeursureThreshold(coeffs);
            case MINIMAX:
                return calculateMinimaxThreshold(coeffs);
            case FIXED:
                return 0.1; // 固定阈值 / Fixed threshold
            default:
                return calculateRigrsureThreshold(coeffs);
        }
    }

    /**
     * 计算Rigrsure阈值 / Calculate Rigrsure Threshold
     */
    private static double calculateRigrsureThreshold(WaveletCoefficients coeffs) {
        // 收集所有细节系数 / Collect all detail coefficients
        java.util.List<Double> allDetails = new java.util.ArrayList<>();
        for (int i = 0; i < coeffs.levels; i++) {
            for (int j = 0; j < coeffs.details[i].length(); j++) {
                allDetails.add(Math.abs(coeffs.details[i].get(j)));
            }
        }
        
        if (allDetails.isEmpty()) {
            return 0.0;
        }
        
        // 排序 / Sort
        allDetails.sort(java.util.Collections.reverseOrder());
        
        // 计算Rigrsure阈值 / Calculate Rigrsure threshold
        int n = allDetails.size();
        double[] risks = new double[n];
        
        for (int i = 0; i < n; i++) {
            double threshold = allDetails.get(i);
            double risk = 0;
            
            for (int j = 0; j < n; j++) {
                double coeff = allDetails.get(j);
                if (coeff > threshold) {
                    risk += Math.min(coeff * coeff, threshold * threshold);
                } else {
                    risk += coeff * coeff;
                }
            }
            
            risks[i] = risk;
        }
        
        // 找到最小风险对应的阈值 / Find threshold corresponding to minimum risk
        int minRiskIndex = 0;
        for (int i = 1; i < n; i++) {
            if (risks[i] < risks[minRiskIndex]) {
                minRiskIndex = i;
            }
        }
        
        return allDetails.get(minRiskIndex);
    }

    /**
     * 计算SURE阈值 / Calculate SURE Threshold
     */
    private static double calculateSureThreshold(WaveletCoefficients coeffs) {
        // 收集所有细节系数 / Collect all detail coefficients
        java.util.List<Double> allDetails = new java.util.ArrayList<>();
        for (int i = 0; i < coeffs.levels; i++) {
            for (int j = 0; j < coeffs.details[i].length(); j++) {
                allDetails.add(Math.abs(coeffs.details[i].get(j)));
            }
        }
        
        if (allDetails.isEmpty()) {
            return 0.0;
        }
        
        double sigma = estimateNoiseVariance(allDetails);
        
        // 计算SURE阈值 / Calculate SURE threshold
        double threshold = sigma * Math.sqrt(2 * Math.log(allDetails.size()));
        
        return threshold;
    }

    /**
     * 计算Heursure阈值 / Calculate Heursure Threshold
     */
    private static double calculateHeursureThreshold(WaveletCoefficients coeffs) {
        double rigrsureThreshold = calculateRigrsureThreshold(coeffs);
        double sureThreshold = calculateSureThreshold(coeffs);
        
        // 收集所有细节系数 / Collect all detail coefficients
        java.util.List<Double> allDetails = new java.util.ArrayList<>();
        for (int i = 0; i < coeffs.levels; i++) {
            for (int j = 0; j < coeffs.details[i].length(); j++) {
                allDetails.add(Math.abs(coeffs.details[i].get(j)));
            }
        }
        
        if (allDetails.isEmpty()) {
            return 0.0;
        }
        
        int n = allDetails.size();
        double eta = calculateEta(allDetails);
        
        // 选择阈值 / Select threshold
        if (eta < 0.5) {
            return Math.min(rigrsureThreshold, sureThreshold);
        } else {
            return sureThreshold;
        }
    }

    /**
     * 计算Minimax阈值 / Calculate Minimax Threshold
     */
    private static double calculateMinimaxThreshold(WaveletCoefficients coeffs) {
        // 收集所有细节系数 / Collect all detail coefficients
        java.util.List<Double> allDetails = new java.util.ArrayList<>();
        for (int i = 0; i < coeffs.levels; i++) {
            for (int j = 0; j < coeffs.details[i].length(); j++) {
                allDetails.add(Math.abs(coeffs.details[i].get(j)));
            }
        }
        
        if (allDetails.isEmpty()) {
            return 0.0;
        }
        
        double sigma = estimateNoiseVariance(allDetails);
        
        // Minimax阈值 / Minimax threshold
        double threshold = sigma * 0.3936; // 对于n>32的近似值 / Approximate value for n>32
        
        return threshold;
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
     * 应用阈值 / Apply Threshold
     */
    private static IVector<Double> applyThreshold(IVector<Double> signal, double threshold, ThresholdType type) {
        IVector<Double> result = Linalg.zeros(signal.length());
        
        for (int i = 0; i < signal.length(); i++) {
            double value = signal.get(i);
            
            if (type == ThresholdType.SOFT) {
                if (Math.abs(value) > threshold) {
                    result.set(i, Math.signum(value) * (Math.abs(value) - threshold));
                } else {
                    result.set(i, 0.0);
                }
            } else { // HARD
                result.set(i, Math.abs(value) > threshold ? value : 0.0);
            }
        }
        
        return result;
    }

    /**
     * 自适应阈值处理 / Adaptive Threshold Processing
     */
    private static IVector<Double> adaptiveThreshold(IVector<Double> signal) {
        // 计算局部统计量 / Calculate local statistics
        double mean = signal.mean();
        double std = signal.std();
        
        // 自适应阈值 / Adaptive threshold
        double threshold = mean + 2 * std;
        
        // 软阈值处理 / Soft threshold processing
        return applyThreshold(signal, threshold, ThresholdType.SOFT);
    }

    /**
     * 提取系数特征 / Extract Coefficient Features
     */
    private static java.util.List<Double> extractCoefficientFeatures(IVector<Double> coeffs, String prefix) {
        java.util.List<Double> features = new java.util.ArrayList<>();
        
        // 统计特征 / Statistical features
        features.add(coeffs.mean());
        features.add(coeffs.std());
        features.add(coeffs.max());
        features.add(coeffs.min());
        features.add(coeffs.sum());
        
        // 高阶统计量 / Higher order statistics
        features.add(calculateSkewness(coeffs));
        features.add(calculateKurtosis(coeffs));
        
        return features;
    }

    /**
     * 提取能量特征 / Extract Energy Features
     */
    private static java.util.List<Double> extractEnergyFeatures(WaveletCoefficients coeffs) {
        java.util.List<Double> features = new java.util.ArrayList<>();
        
        // 总能量 / Total energy
        double totalEnergy = coeffs.approximation.multiply(coeffs.approximation).sum();
        for (int i = 0; i < coeffs.levels; i++) {
            totalEnergy += coeffs.details[i].multiply(coeffs.details[i]).sum();
        }
        features.add(totalEnergy);
        
        // 相对能量 / Relative energy
        features.add(coeffs.approximation.multiply(coeffs.approximation).sum() / totalEnergy);
        
        return features;
    }

    /**
     * 提取熵特征 / Extract Entropy Features
     */
    private static java.util.List<Double> extractEntropyFeatures(WaveletCoefficients coeffs) {
        java.util.List<Double> features = new java.util.ArrayList<>();
        
        // 近似系数熵 / Approximation coefficient entropy
        features.add(calculateEntropy(coeffs.approximation));
        
        // 细节系数熵 / Detail coefficient entropy
        for (int i = 0; i < coeffs.levels; i++) {
            features.add(calculateEntropy(coeffs.details[i]));
        }
        
        return features;
    }

    /**
     * 计算熵 / Calculate Entropy
     */
    private static double calculateEntropy(IVector<Double> signal) {
        // 计算概率分布 / Calculate probability distribution
        double sum = signal.multiply(signal).sum();
        if (sum == 0) {
            return 0.0;
        }
        
        double entropy = 0.0;
        for (int i = 0; i < signal.length(); i++) {
            double p = signal.get(i) * signal.get(i) / sum;
            if (p > 0) {
                entropy -= p * Math.log(p);
            }
        }
        
        return entropy;
    }

    /**
     * 计算偏度 / Calculate Skewness
     */
    private static double calculateSkewness(IVector<Double> signal) {
        double mean = signal.mean();
        double std = signal.std();
        
        if (std == 0) {
            return 0.0;
        }
        
        double skewness = 0.0;
        for (int i = 0; i < signal.length(); i++) {
            double normalized = (signal.get(i) - mean) / std;
            skewness += normalized * normalized * normalized;
        }
        
        return skewness / signal.length();
    }

    /**
     * 计算峰度 / Calculate Kurtosis
     */
    private static double calculateKurtosis(IVector<Double> signal) {
        double mean = signal.mean();
        double std = signal.std();
        
        if (std == 0) {
            return 0.0;
        }
        
        double kurtosis = 0.0;
        for (int i = 0; i < signal.length(); i++) {
            double normalized = (signal.get(i) - mean) / std;
            kurtosis += normalized * normalized * normalized * normalized;
        }
        
        return kurtosis / signal.length() - 3.0; // 减去3得到超额峰度 / Subtract 3 to get excess kurtosis
    }

    /**
     * 估计噪声方差 / Estimate Noise Variance
     */
    private static double estimateNoiseVariance(java.util.List<Double> coefficients) {
        if (coefficients.isEmpty()) {
            return 0.0;
        }
        
        // 使用中位数绝对偏差估计噪声方差 / Use median absolute deviation to estimate noise variance
        java.util.List<Double> sorted = new java.util.ArrayList<>(coefficients);
        sorted.sort(java.util.Collections.reverseOrder());
        
        int n = sorted.size();
        double median = n % 2 == 0 ? 
            (sorted.get(n/2-1) + sorted.get(n/2)) / 2 : 
            sorted.get(n/2);
        
        java.util.List<Double> deviations = new java.util.ArrayList<>();
        for (double coeff : coefficients) {
            deviations.add(Math.abs(coeff - median));
        }
        
        deviations.sort(java.util.Collections.reverseOrder());
        double mad = n % 2 == 0 ? 
            (deviations.get(n/2-1) + deviations.get(n/2)) / 2 : 
            deviations.get(n/2);
        
        return mad / 0.6745; // 转换为标准差 / Convert to standard deviation
    }

    /**
     * 计算Eta值 / Calculate Eta Value
     */
    private static double calculateEta(java.util.List<Double> coefficients) {
        if (coefficients.isEmpty()) {
            return 0.0;
        }
        
        double threshold = Math.sqrt(2 * Math.log(coefficients.size()));
        int count = 0;
        for (double coeff : coefficients) {
            if (Math.abs(coeff) > threshold) {
                count++;
            }
        }
        
        return (double) count / coefficients.size();
    }
}
