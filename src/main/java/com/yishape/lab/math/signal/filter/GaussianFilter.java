package com.yishape.lab.math.signal.filter;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.signal.core.AbstractSignalProcessor;
import com.yishape.lab.math.signal.core.SignalProcessingException;

/**
 * 高斯滤波器实现类 / Gaussian Filter Implementation Class
 * <p>
 * 实现高斯滤波器，这是一种基于高斯函数的低通滤波器。
 * 高斯滤波器在时域和频域都具有良好的特性，常用于平滑和去噪。
 * </p>
 * <p>
 * Implements Gaussian filter, a low-pass filter based on Gaussian function.
 * Gaussian filter has good characteristics in both time and frequency domains, commonly used for smoothing and denoising.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class GaussianFilter extends AbstractSignalProcessor<Double> implements ISignalFilter<Double> {
    
    private FilterType filterType;
    private FilterImplementation implementationType;
    private double sigma;  // 高斯标准差 / Gaussian standard deviation
    private int kernelSize; // 卷积核大小 / Convolution kernel size
    private FilterCoefficients coefficients;
    
    /**
     * 构造函数 / Constructor
     * <p>
     * 创建高斯滤波器。
     * Create Gaussian filter.
     * </p>
     *
     * @param sigma 高斯标准差 / Gaussian standard deviation
     * @throws SignalProcessingException 参数无效时抛出 / Thrown when parameters are invalid
     */
    public GaussianFilter(double sigma) throws SignalProcessingException {
        super("Gaussian Filter", "1.0.0");
        
        validateParameters(sigma);
        
        this.filterType = FilterType.LOW_PASS;
        this.implementationType = FilterImplementation.GAUSSIAN;
        this.sigma = sigma;
        this.kernelSize = (int) Math.ceil(6 * sigma); // 通常选择6σ作为核大小 / Typically choose 6σ as kernel size
        if (kernelSize % 2 == 0) kernelSize++; // 确保核大小为奇数 / Ensure kernel size is odd
        
        computeCoefficients();
    }
    
    /**
     * 构造函数 / Constructor
     * <p>
     * 创建指定参数的高斯滤波器。
     * Create Gaussian filter with specified parameters.
     * </p>
     *
     * @param sigma 高斯标准差 / Gaussian standard deviation
     * @param kernelSize 卷积核大小 / Convolution kernel size
     * @throws SignalProcessingException 参数无效时抛出 / Thrown when parameters are invalid
     */
    public GaussianFilter(double sigma, int kernelSize) throws SignalProcessingException {
        super("Gaussian Filter", "1.0.0");
        
        validateParameters(sigma);
        if (kernelSize <= 0 || kernelSize % 2 == 0) {
            throw new SignalProcessingException("卷积核大小必须为正奇数 / Convolution kernel size must be positive odd number");
        }
        
        this.filterType = FilterType.LOW_PASS;
        this.implementationType = FilterImplementation.GAUSSIAN;
        this.sigma = sigma;
        this.kernelSize = kernelSize;
        
        computeCoefficients();
    }
    
    /**
     * 验证参数 / Validate parameters
     */
    private void validateParameters(double sigma) throws SignalProcessingException {
        if (sigma <= 0) {
            throw new SignalProcessingException("高斯标准差必须大于0 / Gaussian standard deviation must be greater than 0");
        }
    }
    
    /**
     * 计算高斯滤波器系数 / Compute Gaussian filter coefficients
     */
    private void computeCoefficients() throws SignalProcessingException {
        try {
            // 计算高斯核 / Calculate Gaussian kernel
            double[] kernel = new double[kernelSize];
            int center = kernelSize / 2;
            double sum = 0.0;
            
            // 计算高斯函数值 / Calculate Gaussian function values
            for (int i = 0; i < kernelSize; i++) {
                int x = i - center;
                kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
                sum += kernel[i];
            }
            
            // 归一化 / Normalize
            for (int i = 0; i < kernelSize; i++) {
                kernel[i] /= sum;
            }
            
            // 高斯滤波器是FIR滤波器 / Gaussian filter is FIR filter
            coefficients = new FilterCoefficients(kernel);
        } catch (Exception e) {
            throw new SignalProcessingException("计算滤波器系数失败 / Failed to compute filter coefficients", e);
        }
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
            double[] kernel = coefficients.getNumerator();
            int kernelRadius = kernelSize / 2;
            
            int n = signal.length();
            IVector<Double> output = Linalg.zeros(n);
            
            // 卷积运算 / Convolution operation
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int j = 0; j < kernelSize; j++) {
                    int index = i + j - kernelRadius;
                    // 边界处理：使用镜像延拓 / Boundary handling: use mirror extension
                    if (index < 0) {
                        index = -index;
                    } else if (index >= n) {
                        index = 2 * (n - 1) - index;
                    }
                    sum += signal.get(index) * kernel[j];
                }
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
        return kernelSize - 1; // FIR滤波器的阶数 / Order of FIR filter
    }

    /**
     * 获取截止频率 / Get cutoff frequencies
     * @return 截止频率数组 / Cutoff frequency array
     */
    @Override
    public double[] getCutoffFrequencies() {
        // 高斯滤波器的等效截止频率 / Equivalent cutoff frequency of Gaussian filter
        double cutoffFreq = 1.0 / (2 * Math.PI * sigma);
        return new double[]{cutoffFreq};
    }

    /**
     * 设置截止频率 / Set cutoff frequencies
     * @param frequencies 截止频率数组 / Cutoff frequency array
     * @throws SignalProcessingException 频率设置无效时抛出 / Thrown when frequency setting is invalid
     */
    @Override
    public void setCutoffFrequencies(double... frequencies) throws SignalProcessingException {
        if (frequencies.length != 1) {
            throw new SignalProcessingException("高斯滤波器只需要一个截止频率 / Gaussian filter requires only one cutoff frequency");
        }
        // 根据截止频率反推sigma / Infer sigma from cutoff frequency
        double cutoffFreq = frequencies[0];
        this.sigma = 1.0 / (2 * Math.PI * cutoffFreq);
        computeCoefficients(); // 重新计算系数 / Recalculate coefficients
    }
    
    @Override
    public double getSamplingRate() {
        // 高斯滤波器不直接使用采样率 / Gaussian filter doesn't directly use sampling rate
        return 1.0;
    }

    /**
     * 设置采样率 / Set sampling rate
     * @param samplingRate 采样率 / Sampling rate
     * @throws SignalProcessingException 采样率设置无效时抛出 / Thrown when sampling rate setting is invalid
     */
    @Override
    public void setSamplingRate(double samplingRate) throws SignalProcessingException {
        // 高斯滤波器不直接使用采样率 / Gaussian filter doesn't directly use sampling rate
        // 但可以用于计算截止频率 / But can be used to calculate cutoff frequency
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
     * @param frequencies 频率点数组 / Frequency point array
     * @return 频率响应 / Frequency response
     * @throws SignalProcessingException 计算过程中发生错误时抛出 / Thrown when errors occur during calculation
     */
    @Override
    public FrequencyResponse getFrequencyResponse(double[] frequencies) throws SignalProcessingException {
        // 计算高斯滤波器的频率响应 / Calculate frequency response of Gaussian filter
        double[] magnitude = new double[frequencies.length];
        double[] phase = new double[frequencies.length];
        
        for (int i = 0; i < frequencies.length; i++) {
            double omega = 2 * Math.PI * frequencies[i];
            // 高斯滤波器的频率响应 / Frequency response of Gaussian filter
            magnitude[i] = Math.exp(-0.5 * sigma * sigma * omega * omega);
            phase[i] = 0.0;
        }
        
        return new FrequencyResponse(frequencies, magnitude, phase);
    }
    
    /**
     * 获取高斯标准差 / Get Gaussian standard deviation
     * @return 高斯标准差 / Gaussian standard deviation
     */
    public double getSigma() {
        return sigma;
    }

    /**
     * 设置高斯标准差 / Set Gaussian standard deviation
     * @param sigma 高斯标准差 / Gaussian standard deviation
     * @throws SignalProcessingException 参数无效时抛出 / Thrown when parameters are invalid
     */
    public void setSigma(double sigma) throws SignalProcessingException {
        validateParameters(sigma);
        this.sigma = sigma;
        computeCoefficients(); // 重新计算系数 / Recalculate coefficients
    }

    /**
     * 获取卷积核大小 / Get convolution kernel size
     * @return 卷积核大小 / Convolution kernel size
     */
    public int getKernelSize() {
        return kernelSize;
    }

    /**
     * 设置卷积核大小 / Set convolution kernel size
     * @param kernelSize 卷积核大小 / Convolution kernel size
     * @throws SignalProcessingException 参数无效时抛出 / Thrown when parameters are invalid
     */
    public void setKernelSize(int kernelSize) throws SignalProcessingException {
        if (kernelSize <= 0 || kernelSize % 2 == 0) {
            throw new SignalProcessingException("卷积核大小必须为正奇数 / Convolution kernel size must be positive odd number");
        }
        this.kernelSize = kernelSize;
        computeCoefficients(); // 重新计算系数 / Recalculate coefficients
    }
    
    @Override
    protected IVector<Double> doProcess(IVector<Double> input) throws SignalProcessingException {
        return filter(input);
    }

    /**
     * 克隆高斯滤波器 / Clone Gaussian filter
     * @return 高斯滤波器副本 / Gaussian filter copy
     */
    @Override
    public GaussianFilter clone() {
        try {
            return new GaussianFilter(sigma, kernelSize);
        } catch (SignalProcessingException e) {
            // 这不应该发生，因为我们已经验证了参数 / This should not happen since we've validated parameters
            throw new RuntimeException("克隆高斯滤波器失败 / Failed to clone Gaussian filter", e);
        }
    }
}