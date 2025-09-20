package com.reremouse.lab.math.signal.generation;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.signal.core.SignalProcessingException;

/**
 * 信号生成器接口 / Signal Generator Interface
 * <p>
 * 定义所有信号生成操作的基础接口，支持各种信号类型的生成。
 * 使用工厂模式和策略模式支持不同的信号生成算法。
 * </p>
 * <p>
 * Defines the base interface for all signal generation operations supporting various signal types.
 * Uses Factory and Strategy patterns to support different signal generation algorithms.
 * </p>
 *
 * @param <T> 生成信号的数据类型 / Data type of generated signal
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface ISignalGenerator<T extends Number> {
    
    /**
     * 信号类型枚举 / Signal Type Enum
     */
    enum SignalType {
        SINE("正弦波", "Sine Wave"),
        COSINE("余弦波", "Cosine Wave"),
        SQUARE("方波", "Square Wave"),
        TRIANGLE("三角波", "Triangle Wave"),
        SAWTOOTH("锯齿波", "Sawtooth Wave"),
        PULSE("脉冲", "Pulse"),
        CHIRP("线性调频", "Chirp"),
        WHITE_NOISE("白噪声", "White Noise"),
        PINK_NOISE("粉红噪声", "Pink Noise"),
        BROWN_NOISE("布朗噪声", "Brown Noise"),
        GAUSSIAN_NOISE("高斯噪声", "Gaussian Noise"),
        UNIFORM_NOISE("均匀噪声", "Uniform Noise"),
        EXPONENTIAL("指数信号", "Exponential Signal"),
        STEP("阶跃信号", "Step Signal"),
        RAMP("斜坡信号", "Ramp Signal"),
        SINC("Sinc函数", "Sinc Function"),
        DIRAC_DELTA("狄拉克δ函数", "Dirac Delta Function"),
        KRONECKER_DELTA("克罗内克δ函数", "Kronecker Delta Function");
        
        private final String chineseName;
        private final String englishName;
        
        SignalType(String chineseName, String englishName) {
            this.chineseName = chineseName;
            this.englishName = englishName;
        }
        
        public String getChineseName() { return chineseName; }
        public String getEnglishName() { return englishName; }
    }
    
    /**
     * 信号参数类 / Signal Parameters Class
     */
    class SignalParameters {
        private double amplitude = 1.0;           // 幅度 / Amplitude
        private double frequency = 1.0;           // 频率 / Frequency
        private double phase = 0.0;               // 相位 / Phase
        private double dutyCycle = 0.5;           // 占空比 / Duty cycle
        private double offset = 0.0;              // 直流偏置 / DC offset
        private double samplingRate = 1000.0;     // 采样率 / Sampling rate
        private double noiseVariance = 1.0;       // 噪声方差 / Noise variance
        private double decay = 0.0;               // 衰减系数 / Decay coefficient
        private double startFrequency = 1.0;      // 起始频率 (调频信号) / Start frequency (for chirp)
        private double endFrequency = 10.0;       // 结束频率 (调频信号) / End frequency (for chirp)
        private int pulseWidth = 1;               // 脉冲宽度 / Pulse width
        private double stepTime = 0.0;            // 阶跃时间 / Step time
        
        // Builder pattern methods
        public SignalParameters amplitude(double amplitude) { this.amplitude = amplitude; return this; }
        public SignalParameters frequency(double frequency) { this.frequency = frequency; return this; }
        public SignalParameters phase(double phase) { this.phase = phase; return this; }
        public SignalParameters dutyCycle(double dutyCycle) { this.dutyCycle = dutyCycle; return this; }
        public SignalParameters offset(double offset) { this.offset = offset; return this; }
        public SignalParameters samplingRate(double samplingRate) { this.samplingRate = samplingRate; return this; }
        public SignalParameters noiseVariance(double noiseVariance) { this.noiseVariance = noiseVariance; return this; }
        public SignalParameters decay(double decay) { this.decay = decay; return this; }
        public SignalParameters startFrequency(double startFrequency) { this.startFrequency = startFrequency; return this; }
        public SignalParameters endFrequency(double endFrequency) { this.endFrequency = endFrequency; return this; }
        public SignalParameters pulseWidth(int pulseWidth) { this.pulseWidth = pulseWidth; return this; }
        public SignalParameters stepTime(double stepTime) { this.stepTime = stepTime; return this; }
        
        // Getters
        public double getAmplitude() { return amplitude; }
        public double getFrequency() { return frequency; }
        public double getPhase() { return phase; }
        public double getDutyCycle() { return dutyCycle; }
        public double getOffset() { return offset; }
        public double getSamplingRate() { return samplingRate; }
        public double getNoiseVariance() { return noiseVariance; }
        public double getDecay() { return decay; }
        public double getStartFrequency() { return startFrequency; }
        public double getEndFrequency() { return endFrequency; }
        public int getPulseWidth() { return pulseWidth; }
        public double getStepTime() { return stepTime; }
    }
    
    /**
     * 生成信号 / Generate signal
     * <p>
     * 根据指定的信号类型和参数生成信号。
     * Generate signal according to specified signal type and parameters.
     * </p>
     *
     * @param signalType 信号类型 / Signal type
     * @param length 信号长度 / Signal length
     * @param parameters 信号参数 / Signal parameters
     * @return 生成的信号向量 / Generated signal vector
     * @throws SignalProcessingException 生成过程中发生错误时抛出 / Thrown when errors occur during generation
     */
    IVector<T> generate(SignalType signalType, int length, SignalParameters parameters) throws SignalProcessingException;
    
    /**
     * 生成信号（简化版） / Generate signal (simplified version)
     * <p>
     * 使用默认参数生成指定类型的信号。
     * Generate signal of specified type using default parameters.
     * </p>
     *
     * @param signalType 信号类型 / Signal type
     * @param length 信号长度 / Signal length
     * @return 生成的信号向量 / Generated signal vector
     * @throws SignalProcessingException 生成过程中发生错误时抛出 / Thrown when errors occur during generation
     */
    default IVector<T> generate(SignalType signalType, int length) throws SignalProcessingException {
        return generate(signalType, length, new SignalParameters());
    }
    
    /**
     * 生成多分量信号 / Generate multi-component signal
     * <p>
     * 生成多个信号分量的叠加。
     * Generate superposition of multiple signal components.
     * </p>
     *
     * @param signalTypes 信号类型数组 / Signal type array
     * @param length 信号长度 / Signal length
     * @param parameters 信号参数数组 / Signal parameters array
     * @return 生成的复合信号向量 / Generated composite signal vector
     * @throws SignalProcessingException 生成过程中发生错误时抛出 / Thrown when errors occur during generation
     */
    IVector<T> generateComposite(SignalType[] signalTypes, int length, SignalParameters[] parameters) throws SignalProcessingException;
    
    /**
     * 添加噪声到信号 / Add noise to signal
     * <p>
     * 向现有信号添加指定类型的噪声。
     * Add specified type of noise to existing signal.
     * </p>
     *
     * @param signal 原始信号 / Original signal
     * @param noiseType 噪声类型 / Noise type
     * @param parameters 噪声参数 / Noise parameters
     * @return 添加噪声后的信号 / Signal with added noise
     * @throws SignalProcessingException 添加噪声过程中发生错误时抛出 / Thrown when errors occur during noise addition
     */
    IVector<T> addNoise(IVector<T> signal, SignalType noiseType, SignalParameters parameters) throws SignalProcessingException;
    
    /**
     * 验证参数 / Validate parameters
     * <p>
     * 验证给定的信号参数是否有效。
     * Validate if given signal parameters are valid.
     * </p>
     *
     * @param signalType 信号类型 / Signal type
     * @param length 信号长度 / Signal length
     * @param parameters 信号参数 / Signal parameters
     * @return 验证是否通过 / Whether validation passes
     */
    boolean validateParameters(SignalType signalType, int length, SignalParameters parameters);
    
    /**
     * 获取支持的信号类型 / Get supported signal types
     * <p>
     * 返回当前生成器支持的所有信号类型。
     * Return all signal types supported by current generator.
     * </p>
     *
     * @return 支持的信号类型数组 / Supported signal type array
     */
    SignalType[] getSupportedSignalTypes();
    
    /**
     * 获取生成器名称 / Get generator name
     * <p>
     * 返回生成器的名称。
     * Return generator name.
     * </p>
     *
     * @return 生成器名称 / Generator name
     */
    String getName();
    
    /**
     * 获取生成器版本 / Get generator version
     * <p>
     * 返回生成器的版本信息。
     * Return generator version information.
     * </p>
     *
     * @return 版本信息 / Version information
     */
    default String getVersion() {
        return "1.0.0";
    }
}