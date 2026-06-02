package com.yishape.lab.math.signal.filter;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.signal.core.AbstractSignalProcessor;
import com.yishape.lab.math.signal.core.SignalProcessingException;

/**
 * 维纳滤波器实现类 / Wiener Filter Implementation Class
 * <p>
 * 通过求解 Wiener-Hopf 方程 R_xx * w = r_xd 获得最小均方误差意义下的最优 FIR 滤波器系数。
 * 自相关矩阵 R_xx 利用 Toeplitz 结构构建，使用 {@link Linalg#solve} 求解线性系统。
 * </p>
 * <p>
 * Solves the Wiener-Hopf equation R_xx * w = r_xd to obtain optimal FIR filter coefficients
 * under minimum mean-square error criterion. Uses Toeplitz-structured autocorrelation matrix
 * and {@link Linalg#solve} for linear system solution.
 * </p>
 *
 * @author lteb2
 * @version 2.0
 * @since 1.0
 */
public class WienerFilter extends AbstractSignalProcessor<Double> implements ISignalFilter<Double> {

    private ISignalFilter.FilterType filterType;
    private ISignalFilter.FilterImplementation implementationType;
    private double signalPower;   // 信号功率 / Signal power
    private double noisePower;    // 噪声功率 / Noise power
    private int filterLength;     // FIR 滤波器阶数 + 1 / FIR filter order + 1
    private double samplingRate;  // 采样率 / Sampling rate
    /** 缓存的滤波器系数（求解后复用） */
    private double[] cachedWeights;

    /**
     * 构造函数 / Constructor
     *
     * @param signalPower  信号功率 / Signal power
     * @param noisePower   噪声功率 / Noise power
     * @param filterLength FIR 滤波器长度 / FIR filter length
     * @throws SignalProcessingException 参数无效时抛出
     */
    public WienerFilter(double signalPower, double noisePower, int filterLength) throws SignalProcessingException {
        super("Wiener Filter", "2.0.0");
        validateParameters(signalPower, noisePower, filterLength);
        this.filterType = ISignalFilter.FilterType.ADAPTIVE;
        this.implementationType = ISignalFilter.FilterImplementation.FIR;
        this.signalPower = signalPower;
        this.noisePower = noisePower;
        this.filterLength = filterLength;
        this.samplingRate = 1000.0;
        this.cachedWeights = null;
    }

    private void validateParameters(double signalPower, double noisePower, int filterLength) throws SignalProcessingException {
        if (signalPower < 0) {
            throw new SignalProcessingException("信号功率必须大于等于0 / Signal power >= 0");
        }
        if (noisePower <= 0) {
            throw new SignalProcessingException("噪声功率必须大于0 / Noise power > 0");
        }
        if (filterLength <= 0) {
            throw new SignalProcessingException("滤波器长度必须大于0 / Filter length > 0");
        }
    }

    /**
     * 维纳滤波 / Wiener Filtering
     * <p>
     * 1. 估计输入信号的自相关函数
     * 2. 构建 Toeplitz 自相关矩阵 R_xx
     * 3. 估计互相关向量 r_xd（利用 SNR 估计期望信号的成分）
     * 4. 求解 Wiener-Hopf 方程: R_xx * w = r_xd
     * 5. 应用 FIR 滤波
     * </p>
     */
    @Override
    public IVector<Double> filter(IVector<Double> signal) throws SignalProcessingException {
        if (signal == null || signal.length() == 0) {
            throw new SignalProcessingException("输入信号不能为空 / Input signal cannot be empty");
        }
        try {
            int n = signal.length();
            int M = Math.min(filterLength, n / 4); // 确保足够的数据点来估计

            // 1. 估计输入观测信号的自相关 r_yy[lag]
            double[] ryy = autocorrelation(signal, M);

            // 2. 估计信号自相关: r_ss[lag] = r_yy[lag] - noise_floor * delta[lag]
            //    (假设噪声为白噪声，只贡献 lag=0 项)
            double noiseFloor = noisePower;
            double snr = signalPower / noisePower;
            double[] rss = new double[M];
            rss[0] = Math.max(ryy[0] - noiseFloor, ryy[0] * snr / (snr + 1.0));
            for (int lag = 1; lag < M; lag++) {
                rss[lag] = ryy[lag]; // 白噪声不贡献非零 lag 的互相关
            }

            // 3. 构建 Toeplitz 自相关矩阵 R_xx (使用观测自相关或估计的信号自相关)
            IMatrix<Double> R = Linalg.zeros(M, M);
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < M; j++) {
                    int lag = Math.abs(i - j);
                    R.set(i, j, ryy[lag]); // 使用观测自相关确保正定
                }
                // 对角线加小正则化项以确保正定性
                R.set(i, i, R.get(i, i) + 1e-10);
            }

            // 4. 互相关向量 r_xd: 期望信号与观测信号的互相关
            //    对于加性噪声: y = s + n，且 s ⊥ n，有 r_yd = r_ss
            double[] rVec = new double[M];
            for (int i = 0; i < M; i++) {
                rVec[i] = rss[i];
            }
            IVector<Double> r = Linalg.vector(rVec);

            // 5. 求解 Wiener-Hopf 方程: R * w = r
            IVector<Double> w;
            try {
                w = Linalg.solve(R, r);
            } catch (Exception e) {
                // 当 R 病态时回退到最小二乘解
                w = Linalg.lstsq(R, r)._1;
            }

            // 缓存权重
            cachedWeights = new double[w.length()];
            for (int i = 0; i < w.length(); i++) {
                cachedWeights[i] = w.get(i);
            }

            // 6. 应用 FIR 滤波器
            IVector<Double> output = Linalg.zeros(n);
            for (int i = 0; i < n; i++) {
                double sum = 0;
                for (int j = 0; j < M && i - j >= 0; j++) {
                    sum += cachedWeights[j] * signal.get(i - j);
                }
                output.set(i, sum);
            }
            return output;
        } catch (Exception e) {
            throw new SignalProcessingException("维纳滤波处理失败 / Wiener filtering failed", e);
        }
    }

    /**
     * 估计信号的自相关函数 r[lag] = E[x(n)·x(n-lag)]。
     */
    private static double[] autocorrelation(IVector<Double> signal, int maxLag) {
        int n = signal.length();
        double[] r = new double[maxLag];
        for (int lag = 0; lag < maxLag; lag++) {
            double sum = 0;
            int count = n - lag;
            for (int i = lag; i < n; i++) {
                sum += signal.get(i) * signal.get(i - lag);
            }
            r[lag] = sum / count;
        }
        return r;
    }

    // ========== 存取器 / Accessors ==========

    @Override
    public ISignalFilter.FilterType getFilterType() {
        return filterType;
    }

    @Override
    public ISignalFilter.FilterImplementation getImplementationType() {
        return implementationType;
    }

    @Override
    public int getOrder() {
        return filterLength;
    }

    @Override
    public double[] getCutoffFrequencies() {
        return new double[0];
    }

    @Override
    public void setCutoffFrequencies(double... frequencies) throws SignalProcessingException {
        // 维纳滤波器不需要截止频率
    }

    @Override
    public double getSamplingRate() {
        return samplingRate;
    }

    @Override
    public void setSamplingRate(double samplingRate) throws SignalProcessingException {
        if (samplingRate <= 0) {
            throw new SignalProcessingException("采样率必须大于0 / Sampling rate > 0");
        }
        this.samplingRate = samplingRate;
    }

    @Override
    public ISignalFilter.FilterCoefficients getCoefficients() {
        if (cachedWeights != null) {
            return new ISignalFilter.FilterCoefficients(cachedWeights.clone());
        }
        double snr = signalPower / noisePower;
        double[] coeffs = new double[filterLength];
        for (int i = 0; i < filterLength; i++) {
            coeffs[i] = 1.0 / filterLength * (snr / (snr + 1.0));
        }
        return new ISignalFilter.FilterCoefficients(coeffs);
    }

    @Override
    public ISignalFilter.FrequencyResponse getFrequencyResponse(double[] frequencies) throws SignalProcessingException {
        double[] mag = new double[frequencies.length];
        double[] ph = new double[frequencies.length];
        if (cachedWeights != null) {
            for (int i = 0; i < frequencies.length; i++) {
                double omega = 2.0 * Math.PI * frequencies[i] / samplingRate;
                double re = 0, im = 0;
                for (int k = 0; k < cachedWeights.length; k++) {
                    re += cachedWeights[k] * Math.cos(-omega * k);
                    im += cachedWeights[k] * Math.sin(-omega * k);
                }
                mag[i] = Math.hypot(re, im);
                ph[i] = Math.atan2(im, re);
            }
        }
        return new ISignalFilter.FrequencyResponse(frequencies, mag, ph);
    }

    public double getSignalPower() { return signalPower; }
    public void setSignalPower(double signalPower) throws SignalProcessingException {
        validateParameters(signalPower, noisePower, filterLength);
        this.signalPower = signalPower;
        this.cachedWeights = null;
    }
    public double getNoisePower() { return noisePower; }
    public void setNoisePower(double noisePower) throws SignalProcessingException {
        validateParameters(signalPower, noisePower, filterLength);
        this.noisePower = noisePower;
        this.cachedWeights = null;
    }
    public int getFilterLength() { return filterLength; }
    public void setFilterLength(int filterLength) throws SignalProcessingException {
        validateParameters(signalPower, noisePower, filterLength);
        this.filterLength = filterLength;
        this.cachedWeights = null;
    }

    @Override
    protected IVector<Double> doProcess(IVector<Double> input) throws SignalProcessingException {
        return filter(input);
    }

    @Override
    public WienerFilter clone() {
        try {
            return new WienerFilter(signalPower, noisePower, filterLength);
        } catch (SignalProcessingException e) {
            throw new RuntimeException("克隆维纳滤波器失败", e);
        }
    }

    @Override
    public String toString() {
        return String.format("WienerFilter{signalPower=%.3f, noisePower=%.3f, filterLength=%d}",
                signalPower, noisePower, filterLength);
    }
}
