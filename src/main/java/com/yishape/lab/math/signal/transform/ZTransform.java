package com.yishape.lab.math.signal.transform;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.signal.core.AbstractSignalProcessor;
import com.yishape.lab.math.signal.core.Complex;
import com.yishape.lab.math.signal.core.SignalProcessingException;

/**
 * Z变换实现类 / Z-Transform Implementation Class
 * <p>
 * 实现离散时间信号的Z变换和逆Z变换。
 * Z变换是离散时间系统分析的重要工具，类似于连续时间系统的拉普拉斯变换。
 * </p>
 * <p>
 * Implements Z-transform and inverse Z-transform for discrete-time signals.
 * Z-transform is an important tool for discrete-time system analysis, analogous to Laplace transform for continuous-time systems.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ZTransform extends AbstractSignalProcessor<Double> implements ISignalTransform<Double, Complex[]> {
    
    private int numEvaluationPoints;
    private double radiusStart;
    private double radiusEnd;
    private boolean useUnitCircle;
    
    /**
     * 构造函数 / Constructor
     * <p>
     * 使用默认参数初始化Z变换。
     * Initialize Z-transform with default parameters.
     * </p>
     */
    public ZTransform() {
        super("Z-Transform", "1.0.0");
        this.numEvaluationPoints = 512;
        this.radiusStart = 0.1;
        this.radiusEnd = 2.0;
        this.useUnitCircle = true;
    }
    
    /**
     * 构造函数 / Constructor
     * <p>
     * 使用指定参数初始化Z变换。
     * Initialize Z-transform with specified parameters.
     * </p>
     *
     * @param numEvaluationPoints 评估点数量 / Number of evaluation points
     * @param radiusStart 起始半径 / Start radius
     * @param radiusEnd 结束半径 / End radius
     * @param useUnitCircle 是否使用单位圆 / Whether to use unit circle
     */
    public ZTransform(int numEvaluationPoints, double radiusStart, double radiusEnd, boolean useUnitCircle) {
        super("Z-Transform", "1.0.0");
        this.numEvaluationPoints = numEvaluationPoints;
        this.radiusStart = radiusStart;
        this.radiusEnd = radiusEnd;
        this.useUnitCircle = useUnitCircle;
    }
    
    /**
     * 计算Z变换 / Calculate Z-transform
     * <p>
     * 计算离散时间序列的Z变换。
     * Calculate Z-transform of discrete-time sequence.
     * </p>
     *
     * @param signal 输入时域信号 / Input time domain signal
     * @return Z变换结果 / Z-transform result
     * @throws SignalProcessingException 变换过程中发生错误时抛出 / Thrown when errors occur during transform
     */
    @Override
    public Complex[] forward(IVector<Double> signal) throws SignalProcessingException {
        if (signal == null || signal.length() == 0) {
            throw new SignalProcessingException("输入信号不能为空 / Input signal cannot be empty");
        }
        
        int N = signal.length();
        Complex[] result = new Complex[numEvaluationPoints];
        
        if (useUnitCircle) {
            // 在单位圆上计算Z变换 / Calculate Z-transform on unit circle
            return calculateOnUnitCircle(signal);
        } else {
            // 在指定半径范围内计算Z变换 / Calculate Z-transform within specified radius range
            return calculateOnRadiusRange(signal);
        }
    }
    
    /**
     * 在单位圆上计算Z变换 / Calculate Z-transform on unit circle
     * <p>
     * 当z在单位圆上时，Z变换等价于DTFT。
     * When z is on unit circle, Z-transform is equivalent to DTFT.
     * </p>
     */
    private Complex[] calculateOnUnitCircle(IVector<Double> signal) {
        int N = signal.length();
        Complex[] result = new Complex[numEvaluationPoints];
        
        for (int k = 0; k < numEvaluationPoints; k++) {
            double omega = 2.0 * Math.PI * k / numEvaluationPoints;
            Complex z = new Complex(Math.cos(omega), Math.sin(omega));
            
            Complex sum = new Complex(0, 0);
            for (int n = 0; n < N; n++) {
                // X(z) = ∑_{n=0}^{N-1} x[n] * z^{-n}
                Complex zPowerN = z.power(-n);
                sum = sum.add(zPowerN.scale(signal.get(n)));
            }
            result[k] = sum;
        }
        
        return result;
    }
    
    /**
     * 在指定半径范围内计算Z变换 / Calculate Z-transform within specified radius range
     */
    private Complex[] calculateOnRadiusRange(IVector<Double> signal) {
        int N = signal.length();
        Complex[] result = new Complex[numEvaluationPoints];
        
        for (int k = 0; k < numEvaluationPoints; k++) {
            // 在复平面上螺旋采样 / Spiral sampling in complex plane
            double radius = radiusStart + (radiusEnd - radiusStart) * k / (numEvaluationPoints - 1);
            double angle = 2.0 * Math.PI * k / numEvaluationPoints;
            Complex z = new Complex(radius * Math.cos(angle), radius * Math.sin(angle));
            
            Complex sum = new Complex(0, 0);
            for (int n = 0; n < N; n++) {
                Complex zPowerN = z.power(-n);
                sum = sum.add(zPowerN.scale(signal.get(n)));
            }
            result[k] = sum;
        }
        
        return result;
    }
    
    /**
     * 计算逆Z变换 / Calculate inverse Z-transform
     * <p>
     * 通过围道积分计算逆Z变换（简化实现）。
     * Calculate inverse Z-transform using contour integration (simplified implementation).
     * </p>
     *
     * @param transformed Z变换结果 / Z-transform result
     * @return 时域信号 / Time domain signal
     * @throws SignalProcessingException 逆变换过程中发生错误时抛出 / Thrown when errors occur during inverse transform
     */
    @Override
    public IVector<Double> inverse(Complex[] transformed) throws SignalProcessingException {
        if (transformed == null || transformed.length == 0) {
            throw new SignalProcessingException("输入变换结果不能为空 / Input transform result cannot be empty");
        }
        
        // 简化的逆Z变换实现：假设变换是在单位圆上进行的
        // Simplified inverse Z-transform implementation: assume transform was performed on unit circle
        int N = transformed.length;
        IVector<Double> result = Linalg.zeros(N);
        
        for (int n = 0; n < N; n++) {
            Complex sum = new Complex(0, 0);
            for (int k = 0; k < N; k++) {
                double omega = 2.0 * Math.PI * k / N;
                Complex z = new Complex(Math.cos(omega), Math.sin(omega));
                Complex zPowerN = z.power(n);
                sum = sum.add(transformed[k].multiply(zPowerN));
            }
            // 除以2πj和积分路径长度
            result.set(n, sum.real / N);
        }
        
        return result;
    }
    
    /**
     * 计算极点和零点 / Calculate poles and zeros
     * <p>
     * 分析Z变换的极点和零点，用于系统稳定性分析。
     * Analyze poles and zeros of Z-transform for system stability analysis.
     * </p>
     *
     * @param numeratorCoeffs 分子系数 / Numerator coefficients
     * @param denominatorCoeffs 分母系数 / Denominator coefficients
     * @return 包含极点和零点的结果 / Result containing poles and zeros
     * @throws SignalProcessingException 计算过程中发生错误时抛出 / Thrown when errors occur during calculation
     */
    public PoleZeroResult calculatePolesAndZeros(double[] numeratorCoeffs, double[] denominatorCoeffs) throws SignalProcessingException {
        // 这里应该实现多项式求根算法来找到极点和零点
        // Polynomial root-finding algorithm should be implemented here to find poles and zeros
        
        // 简化实现：返回空结果
        // Simplified implementation: return empty result
        Complex[] zeros = new Complex[numeratorCoeffs.length - 1];
        Complex[] poles = new Complex[denominatorCoeffs.length - 1];
        
        // 实际实现需要使用数值方法求解多项式根
        // Actual implementation needs numerical methods to solve polynomial roots
        
        return new PoleZeroResult(zeros, poles);
    }
    
    /**
     * 计算频率响应 / Calculate frequency response
     * <p>
     * 在单位圆上计算系统的频率响应。
     * Calculate frequency response of system on unit circle.
     * </p>
     *
     * @param numeratorCoeffs 分子系数 / Numerator coefficients
     * @param denominatorCoeffs 分母系数 / Denominator coefficients
     * @param frequencies 频率点 / Frequency points
     * @return 频率响应 / Frequency response
     * @throws SignalProcessingException 计算过程中发生错误时抛出 / Thrown when errors occur during calculation
     */
    public Complex[] calculateFrequencyResponse(double[] numeratorCoeffs, double[] denominatorCoeffs, double[] frequencies) throws SignalProcessingException {
        Complex[] response = new Complex[frequencies.length];
        
        for (int i = 0; i < frequencies.length; i++) {
            double omega = frequencies[i];
            Complex z = new Complex(Math.cos(omega), Math.sin(omega));
            
            // 计算分子多项式值 / Calculate numerator polynomial value
            Complex numerator = new Complex(0, 0);
            for (int j = 0; j < numeratorCoeffs.length; j++) {
                numerator = numerator.add(z.power(-j).scale(numeratorCoeffs[j]));
            }
            
            // 计算分母多项式值 / Calculate denominator polynomial value
            Complex denominator = new Complex(0, 0);
            for (int j = 0; j < denominatorCoeffs.length; j++) {
                denominator = denominator.add(z.power(-j).scale(denominatorCoeffs[j]));
            }
            
            // H(z) = N(z) / D(z)
            response[i] = numerator.divide(denominator);
        }
        
        return response;
    }
    
    /**
     * 检查系统稳定性 / Check system stability
     * <p>
     * 通过检查极点是否在单位圆内来判断系统稳定性。
     * Check system stability by verifying if poles are inside unit circle.
     * </p>
     *
     * @param poles 系统极点 / System poles
     * @return 系统是否稳定 / Whether system is stable
     */
    public boolean isSystemStable(Complex[] poles) {
        for (Complex pole : poles) {
            if (pole.magnitude() >= 1.0) {
                return false; // 极点在单位圆外或单位圆上，系统不稳定
            }
        }
        return true; // 所有极点在单位圆内，系统稳定
    }
    
    @Override
    protected IVector<Double> doProcess(IVector<Double> input) throws SignalProcessingException {
        // 默认处理：计算Z变换的幅度谱
        Complex[] zTransform = forward(input);
        IVector<Double> magnitude = Linalg.zeros(zTransform.length);
        for (int i = 0; i < zTransform.length; i++) {
            magnitude.set(i, zTransform[i].magnitude());
        }
        return magnitude;
    }
    
    @Override
    public ZTransform clone() {
        return new ZTransform(numEvaluationPoints, radiusStart, radiusEnd, useUnitCircle);
    }
    
    // Getters and setters
    public int getNumEvaluationPoints() { return numEvaluationPoints; }
    public void setNumEvaluationPoints(int numEvaluationPoints) { this.numEvaluationPoints = numEvaluationPoints; }
    
    public double getRadiusStart() { return radiusStart; }
    public void setRadiusStart(double radiusStart) { this.radiusStart = radiusStart; }
    
    public double getRadiusEnd() { return radiusEnd; }
    public void setRadiusEnd(double radiusEnd) { this.radiusEnd = radiusEnd; }
    
    public boolean isUseUnitCircle() { return useUnitCircle; }
    public void setUseUnitCircle(boolean useUnitCircle) { this.useUnitCircle = useUnitCircle; }
    
    /**
     * 极点零点结果内部类 / Pole-Zero Result Inner Class
     */
    public static class PoleZeroResult {
        private final Complex[] zeros;
        private final Complex[] poles;
        
        public PoleZeroResult(Complex[] zeros, Complex[] poles) {
            this.zeros = zeros.clone();
            this.poles = poles.clone();
        }
        
        public Complex[] getZeros() { return zeros.clone(); }
        public Complex[] getPoles() { return poles.clone(); }
        
        public boolean isStable() {
            for (Complex pole : poles) {
                if (pole.magnitude() >= 1.0) {
                    return false;
                }
            }
            return true;
        }
    }
}