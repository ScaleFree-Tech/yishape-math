package com.yishape.lab.math.signal.filter;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.signal.core.AbstractSignalProcessor;
import com.yishape.lab.math.signal.core.SignalProcessingException;

/**
 * 卡尔曼滤波器实现类 / Kalman Filter Implementation Class
 * <p>
 * 实现卡尔曼滤波器，用于从噪声信号中估计真实信号状态。
 * 卡尔曼滤波是一种递归滤波器，能够在线性系统中提供最优估计。
 * </p>
 * <p>
 * Implements Kalman filter for estimating true signal state from noisy signals.
 * Kalman filter is a recursive filter that provides optimal estimation in linear systems.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class KalmanFilter extends AbstractSignalProcessor<Double> implements ISignalFilter<Double> {
    
    private ISignalFilter.FilterType filterType;
    private ISignalFilter.FilterImplementation implementationType;
    private double processNoiseVariance;  // 过程噪声方差 / Process noise variance
    private double measurementNoiseVariance;  // 测量噪声方差 / Measurement noise variance
    private double estimateErrorVariance;  // 估计误差方差 / Estimate error variance
    private double estimate;  // 当前估计值 / Current estimate
    private double samplingRate; // 采样率 / Sampling rate
    
    /**
     * 构造函数 / Constructor
     * <p>
     * 创建卡尔曼滤波器。
     * Create Kalman filter.
     * </p>
     *
     * @param processNoiseVariance 过程噪声方差 / Process noise variance
     * @param measurementNoiseVariance 测量噪声方差 / Measurement noise variance
     * @throws SignalProcessingException 参数无效时抛出 / Thrown when parameters are invalid
     */
    public KalmanFilter(double processNoiseVariance, double measurementNoiseVariance) throws SignalProcessingException {
        super("Kalman Filter", "1.0.0");
        
        validateParameters(processNoiseVariance, measurementNoiseVariance);
        
        this.filterType = ISignalFilter.FilterType.ADAPTIVE;
        this.implementationType = ISignalFilter.FilterImplementation.MOVING_AVERAGE; // Using as approximation
        this.processNoiseVariance = processNoiseVariance;
        this.measurementNoiseVariance = measurementNoiseVariance;
        this.estimateErrorVariance = 1.0;  // 初始估计误差方差 / Initial estimate error variance
        this.estimate = 0.0;  // 初始估计值 / Initial estimate
        this.samplingRate = 1000.0; // Default sampling rate
    }
    
    /**
     * 验证参数 / Validate parameters
     */
    private void validateParameters(double processNoiseVariance, double measurementNoiseVariance) throws SignalProcessingException {
        if (processNoiseVariance < 0) {
            throw new SignalProcessingException("过程噪声方差必须大于等于0 / Process noise variance must be greater than or equal to 0");
        }
        if (measurementNoiseVariance <= 0) {
            throw new SignalProcessingException("测量噪声方差必须大于0 / Measurement noise variance must be greater than 0");
        }
    }
    
    /**
     * 滤波信号 / Filter signal
     * <p>
     * 对输入信号进行卡尔曼滤波处理。
     * Apply Kalman filtering to input signal.
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
            int n = signal.length();
            IVector<Double> output = Linalg.zeros(n);
            
            // 重置状态 / Reset state
            this.estimateErrorVariance = 1.0;
            this.estimate = signal.get(0);  // 使用第一个测量值作为初始估计 / Use first measurement as initial estimate
            
            // 卡尔曼滤波算法 / Kalman filter algorithm
            for (int i = 0; i < n; i++) {
                double measurement = signal.get(i);
                
                // 预测步骤 / Prediction step
                // 对于简单的1D情况，状态转移矩阵为1，控制输入为0
                // For simple 1D case, state transition matrix is 1, control input is 0
                double predictedEstimate = estimate;  // 状态预测 / State prediction
                double predictedErrorVariance = estimateErrorVariance + processNoiseVariance;  // 误差协方差预测 / Error covariance prediction
                
                // 更新步骤 / Update step
                double kalmanGain = predictedErrorVariance / (predictedErrorVariance + measurementNoiseVariance);  // 卡尔曼增益 / Kalman gain
                estimate = predictedEstimate + kalmanGain * (measurement - predictedEstimate);  // 状态更新 / State update
                estimateErrorVariance = (1 - kalmanGain) * predictedErrorVariance;  // 误差协方差更新 / Error covariance update
                
                output.set(i, estimate);
            }
            
            return output;
        } catch (Exception e) {
            throw new SignalProcessingException("卡尔曼滤波处理失败 / Kalman filtering failed", e);
        }
    }
    
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
        return 1;  // 卡尔曼滤波器没有传统意义上的阶数 / Kalman filter doesn't have traditional order
    }
    
    @Override
    public double[] getCutoffFrequencies() {
        return new double[0];  // 卡尔曼滤波器没有截止频率 / Kalman filter doesn't have cutoff frequencies
    }
    
    @Override
    public void setCutoffFrequencies(double... frequencies) throws SignalProcessingException {
        // 卡尔曼滤波器不需要设置截止频率 / Kalman filter doesn't need cutoff frequencies
    }
    
    @Override
    public double getSamplingRate() {
        return samplingRate;
    }
    
    @Override
    public void setSamplingRate(double samplingRate) throws SignalProcessingException {
        if (samplingRate <= 0) {
            throw new SignalProcessingException("采样率必须大于0 / Sampling rate must be greater than 0");
        }
        this.samplingRate = samplingRate;
    }
    
    @Override
    public ISignalFilter.FilterCoefficients getCoefficients() {
        // 卡尔曼滤波器使用状态空间表示，不使用传统的系数 / Kalman filter uses state-space representation, not traditional coefficients
        return new ISignalFilter.FilterCoefficients(new double[]{1.0}, new double[]{1.0});
    }
    
    @Override
    public ISignalFilter.FrequencyResponse getFrequencyResponse(double[] frequencies) throws SignalProcessingException {
        // 卡尔曼滤波器的频率响应不是固定的 / Kalman filter frequency response is not fixed
        double[] magnitude = new double[frequencies.length];
        double[] phase = new double[frequencies.length];
        for (int i = 0; i < frequencies.length; i++) {
            magnitude[i] = 1.0;
            phase[i] = 0.0;
        }
        return new ISignalFilter.FrequencyResponse(frequencies, magnitude, phase);
    }
    
    /**
     * 获取过程噪声方差 / Get process noise variance
     */
    public double getProcessNoiseVariance() {
        return processNoiseVariance;
    }
    
    /**
     * 设置过程噪声方差 / Set process noise variance
     */
    public void setProcessNoiseVariance(double processNoiseVariance) throws SignalProcessingException {
        validateParameters(processNoiseVariance, measurementNoiseVariance);
        this.processNoiseVariance = processNoiseVariance;
    }
    
    /**
     * 获取测量噪声方差 / Get measurement noise variance
     */
    public double getMeasurementNoiseVariance() {
        return measurementNoiseVariance;
    }
    
    /**
     * 设置测量噪声方差 / Set measurement noise variance
     */
    public void setMeasurementNoiseVariance(double measurementNoiseVariance) throws SignalProcessingException {
        validateParameters(processNoiseVariance, measurementNoiseVariance);
        this.measurementNoiseVariance = measurementNoiseVariance;
    }
    
    @Override
    protected IVector<Double> doProcess(IVector<Double> input) throws SignalProcessingException {
        return filter(input);
    }
    
    @Override
    public KalmanFilter clone() {
        try {
            return new KalmanFilter(processNoiseVariance, measurementNoiseVariance);
        } catch (SignalProcessingException e) {
            // This should not happen as we're cloning valid parameters
            throw new RuntimeException("克隆卡尔曼滤波器失败 / Failed to clone Kalman filter", e);
        }
    }
    
    @Override
    public String toString() {
        return String.format("KalmanFilter{processNoiseVariance=%f, measurementNoiseVariance=%f}", 
                           processNoiseVariance, measurementNoiseVariance);
    }
}