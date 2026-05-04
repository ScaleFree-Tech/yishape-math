package com.yishape.lab.math.signal.wavele;

import com.yishape.lab.math.linalg.IVector;

/**
 * 小波系数结构 / Wavelet Coefficients Structure
 * <p>
 * 用于存储小波变换的分解结果，包含近似系数和细节系数。
 * Stores the results of wavelet decomposition, including approximation and detail coefficients.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class WaveletCoefficients {
    public IVector<Double> approximation;  // 近似系数 / Approximation coefficients
    public IVector<Double>[] details;      // 细节系数 / Detail coefficients
    public int levels;                     // 分解层数 / Number of decomposition levels

    /**
     * 创建小波系数对象 / Create Wavelet Coefficients Object
     *
     * @param approximation 近似系数 / Approximation coefficients
     * @param details 细节系数数组 / Array of detail coefficients
     * @param levels 分解层数 / Number of decomposition levels
     */
    public WaveletCoefficients(IVector<Double> approximation, IVector<Double>[] details, int levels) {
        this.approximation = approximation;
        this.details = details;
        this.levels = levels;
    }
}
