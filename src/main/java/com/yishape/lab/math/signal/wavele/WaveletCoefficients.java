package com.yishape.lab.math.signal.wavele;

import com.yishape.lab.math.linalg.IVector;

    /**
     * 小波系数结构 / Wavelet Coefficients Structure
     */
public class WaveletCoefficients {
    public IVector<Double> approximation;  // 近似系数 / Approximation coefficients
    public IVector<Double>[] details;      // 细节系数 / Detail coefficients
    public int levels;                     // 分解层数 / Number of decomposition levels

    public WaveletCoefficients(IVector<Double> approximation, IVector<Double>[] details, int levels) {
        this.approximation = approximation;
        this.details = details;
        this.levels = levels;
    }
}
