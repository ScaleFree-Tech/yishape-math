package com.reremouse.lab.math.signal;

import com.reremouse.lab.math.signal.wavele.WaveletUtilities;
import com.reremouse.lab.math.signal.wavele.WaveletFilters;
import com.reremouse.lab.math.signal.wavele.WaveletAnalysis;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.signal.wavele.WaveletCoefficients;

/**
 * 小波分析示例类 / Wavelet Analysis Example Class
 * <p>
 * 展示如何使用小波分析包中的各种功能。
 * Demonstrates how to use various functions in the wavelet analysis package.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class WaveletAnalysisExample2 {

    public static void main(String[] args) {
        System.out.println("=== 小波分析功能示例 / Wavelet Analysis Example ===");
        
        // 生成测试信号 / Generate test signal
        int length = 1024;
        double samplingRate = 1000.0; // 1kHz
        
        // 生成复合信号：多个频率成分 + 噪声 / Generate composite signal: multiple frequency components + noise
        IVector<Double> signal = Signals.sineWave(length, 10.0, samplingRate, 1.0, 0.0)
                .add(Signals.sineWave(length, 50.0, samplingRate, 0.8, 0.0))
                .add(Signals.sineWave(length, 100.0, samplingRate, 0.5, 0.0))
                .add(Signals.whiteNoise(length, 0.2));
        
        System.out.println("原始信号统计:");
        System.out.println("  长度: " + signal.length());
        System.out.println("  均值: " + signal.mean());
        System.out.println("  标准差: " + signal.std());
        System.out.println("  峰值: " + signal.max());
        
        // 小波变换分析 / Wavelet Transform Analysis
        System.out.println("\n=== 小波变换分析 / Wavelet Transform Analysis ===");
        
        // 离散小波变换 / Discrete Wavelet Transform
        WaveletCoefficients dwtCoeffs = WaveletAnalysis.discreteWaveletTransform(
                signal, WaveletAnalysis.WaveletType.DAUBECHIES, 4, 4);
        
        System.out.println("离散小波变换 (DWT):");
        System.out.println("  分解层数: " + dwtCoeffs.levels);
        System.out.println("  近似系数长度: " + dwtCoeffs.approximation.length());
        System.out.println("  细节系数长度: " + dwtCoeffs.details[0].length());
        
        // 小波包变换 / Wavelet Packet Transform
        WaveletAnalysis.WaveletPacketTree packetTree = WaveletAnalysis.waveletPacketTransform(
                signal, WaveletAnalysis.WaveletType.DAUBECHIES, 3, 4);
        
        System.out.println("小波包变换:");
        System.out.println("  根节点信号长度: " + packetTree.root.signal.length());
        System.out.println("  根节点层数: " + packetTree.root.level);
        
        // 连续小波变换 / Continuous Wavelet Transform
        IVector<Double> scales = Linalg.range(1, 65).multiplyScalar(0.5);
        var cwtResult = WaveletAnalysis.continuousWaveletTransform(
                signal, WaveletAnalysis.WaveletType.MORLET, scales, 5.0);
        
        System.out.println("连续小波变换 (CWT):");
        System.out.println("  尺度数: " + scales.length());
        System.out.println("  时频矩阵大小: " + cwtResult.rows() + " x " + cwtResult.cols());
        
        // 小波去噪 / Wavelet Denoising
        System.out.println("\n=== 小波去噪 / Wavelet Denoising ===");
        
        try {
            // 使用不同方法进行去噪 / Denoise using different methods
            IVector<Double> denoisedRigrsure = WaveletUtilities.waveletDenoising(
                    signal, WaveletAnalysis.WaveletType.DAUBECHIES, 4,
                    WaveletUtilities.ThresholdMethod.RIGRSURE, WaveletUtilities.ThresholdType.SOFT, 4);
            
            IVector<Double> denoisedSure = WaveletUtilities.waveletDenoising(
                    signal, WaveletAnalysis.WaveletType.DAUBECHIES, 4,
                    WaveletUtilities.ThresholdMethod.SURE, WaveletUtilities.ThresholdType.SOFT, 4);
            
            IVector<Double> adaptiveDenoised = WaveletUtilities.adaptiveWaveletDenoising(
                    signal, WaveletAnalysis.WaveletType.DAUBECHIES, 4, 4);
            
            System.out.println("去噪结果:");
            System.out.println("  Rigrsure方法 - 均值: " + denoisedRigrsure.mean() + 
                              ", 标准差: " + denoisedRigrsure.std());
            System.out.println("  SURE方法 - 均值: " + denoisedSure.mean() + 
                              ", 标准差: " + denoisedSure.std());
            System.out.println("  自适应方法 - 均值: " + adaptiveDenoised.mean() + 
                              ", 标准差: " + adaptiveDenoised.std());
            
            // 计算去噪效果 / Calculate denoising effect
            double originalSNR = calculateSNR(signal, signal);
            double denoisedSNR = calculateSNR(signal, denoisedRigrsure);
            System.out.println("  信噪比改善: " + (denoisedSNR - originalSNR) + " dB");
            
        } catch (Exception e) {
            System.out.println("去噪过程中出现错误: " + e.getMessage());
            e.printStackTrace();
        }
        
        // 小波压缩 / Wavelet Compression
        System.out.println("\n=== 小波压缩 / Wavelet Compression ===");
        
        try {
            double[] compressionRatios = {0.1, 0.2, 0.5, 0.8};
            for (double ratio : compressionRatios) {
                IVector<Double> compressed = WaveletUtilities.waveletCompression(
                        signal, WaveletAnalysis.WaveletType.DAUBECHIES, 4, ratio, 4);
                
                double compressionError = calculateCompressionError(signal, compressed);
                double actualCompressionRatio = (double) signal.length() / compressed.length();
                
                System.out.println("  压缩比 " + (int)(ratio * 100) + "%:");
                System.out.println("    压缩误差: " + compressionError);
                System.out.println("    实际压缩比: " + actualCompressionRatio);
            }
        } catch (Exception e) {
            System.out.println("压缩过程中出现错误: " + e.getMessage());
            e.printStackTrace();
        }
        
        // 小波特征提取 / Wavelet Feature Extraction
        System.out.println("\n=== 小波特征提取 / Wavelet Feature Extraction ===");
        
        try {
            IVector<Double> features = WaveletUtilities.extractWaveletFeatures(
                    signal, WaveletAnalysis.WaveletType.DAUBECHIES, 4, 4);
            
            System.out.println("提取的特征:");
            System.out.println("  特征数量: " + features.length());
            System.out.println("  前10个特征: ");
            for (int i = 0; i < Math.min(10, features.length()); i++) {
                System.out.printf("    特征[%d] = %.4f\n", i, features.get(i));
            }
        } catch (Exception e) {
            System.out.println("特征提取过程中出现错误: " + e.getMessage());
            e.printStackTrace();
        }
        
        // 小波能量分析 / Wavelet Energy Analysis
        System.out.println("\n=== 小波能量分析 / Wavelet Energy Analysis ===");
        
        try {
            IVector<Double> waveletEnergy = WaveletUtilities.analyzeWaveletEnergy(dwtCoeffs);
            IVector<Double> entropy = WaveletUtilities.analyzeWaveletEntropy(dwtCoeffs);
            
            System.out.println("能量分布:");
            for (int i = 0; i < waveletEnergy.length(); i++) {
                String levelName = (i == 0) ? "近似" : "细节" + i;
                System.out.printf("  %s层能量: %.4f, 熵: %.4f\n", levelName, waveletEnergy.get(i), entropy.get(i));
            }
        } catch (Exception e) {
            System.out.println("能量分析过程中出现错误: " + e.getMessage());
            e.printStackTrace();
        }
        
        // 小波尺度图分析 / Wavelet Scalogram Analysis
        System.out.println("\n=== 小波尺度图分析 / Wavelet Scalogram Analysis ===");
        
        try {
            var scalogram = WaveletUtilities.calculateScalogram(
                    signal, WaveletAnalysis.WaveletType.MORLET, scales, 5.0);
            
            System.out.println("尺度图信息:");
            System.out.println("  尺度图大小: " + scalogram.rows() + " x " + scalogram.cols());
            
            // 找到最大能量位置 / Find maximum energy position
            double maxEnergy = 0;
            int maxScale = 0, maxTime = 0;
            for (int i = 0; i < scalogram.rows(); i++) {
                for (int j = 0; j < scalogram.cols(); j++) {
                    double energy = Math.abs(scalogram.get(i, j));
                    if (energy > maxEnergy) {
                        maxEnergy = energy;
                        maxScale = i;
                        maxTime = j;
                    }
                }
            }
            
            System.out.println("  最大能量: " + maxEnergy);
            System.out.println("  最大能量位置: 尺度=" + scales.get(maxScale) + ", 时间=" + maxTime);
        } catch (Exception e) {
            System.out.println("尺度图分析过程中出现错误: " + e.getMessage());
            e.printStackTrace();
        }
        
        // 小波滤波器测试 / Wavelet Filter Testing
        System.out.println("\n=== 小波滤波器测试 / Wavelet Filter Testing ===");
        
        testWaveletFilters();
        
        System.out.println("\n=== 示例完成 / Example Complete ===");
    }
    
    /**
     * 测试小波滤波器 / Test Wavelet Filters
     */
    private static void testWaveletFilters() {
        System.out.println("小波滤波器测试:");
        
        // 测试不同小波族 / Test different wavelet families
        WaveletFilters.WaveletFamily[] families = {
            WaveletFilters.WaveletFamily.HAAR,
            WaveletFilters.WaveletFamily.DAUBECHIES,
            WaveletFilters.WaveletFamily.COIFLETS,
            WaveletFilters.WaveletFamily.BIORTHOGONAL,
            WaveletFilters.WaveletFamily.SYMLETS
        };
        
        for (WaveletFilters.WaveletFamily family : families) {
            try {
                WaveletFilters.FilterCoefficients coeffs = WaveletFilters.getFilterCoefficients(family, 2);
                boolean isValid = WaveletFilters.validateFilterCoefficients(coeffs);
                int filterLength = WaveletFilters.getFilterLength(coeffs);
                String info = WaveletFilters.getWaveletFamilyInfo(family);
                
                System.out.println("  " + coeffs.name + ":");
                System.out.println("    滤波器长度: " + filterLength);
                System.out.println("    有效性: " + (isValid ? "有效" : "无效"));
                System.out.println("    描述: " + info);
            } catch (Exception e) {
                System.out.println("  " + family + ": 错误 - " + e.getMessage());
            }
        }
    }
    
    /**
     * 计算信噪比 / Calculate Signal-to-Noise Ratio
     */
    private static double calculateSNR(IVector<Double> original, IVector<Double> noisy) {
        if (original.length() != noisy.length()) {
            return 0.0;
        }
        
        IVector<Double> noise = original.sub(noisy);
        double signalPower = original.multiply(original).mean();
        double noisePower = noise.multiply(noise).mean();
        
        if (noisePower == 0) {
            return Double.POSITIVE_INFINITY;
        }
        
        return 10 * Math.log10(signalPower / noisePower);
    }
    
    /**
     * 计算压缩误差 / Calculate Compression Error
     */
    private static double calculateCompressionError(IVector<Double> original, IVector<Double> compressed) {
        if (original.length() != compressed.length()) {
            return Double.POSITIVE_INFINITY;
        }
        
        IVector<Double> error = original.sub(compressed);
        return error.multiply(error).mean();
    }
}
