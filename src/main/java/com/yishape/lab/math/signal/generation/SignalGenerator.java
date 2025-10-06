package com.yishape.lab.math.signal.generation;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.signal.core.AbstractSignalProcessor;
import com.yishape.lab.math.signal.core.SignalProcessingException;

/**
 * 信号生成器实现类 / Signal Generator Implementation Class
 * <p>
 * 实现各种信号类型的生成功能。
 * Implements generation functions for various signal types.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class SignalGenerator extends AbstractSignalProcessor<Double> implements ISignalGenerator<Double> {
    
    /**
     * 构造函数 / Constructor
     */
    public SignalGenerator() {
        super("Signal Generator", "1.0.0");
    }
    
    @Override
    public IVector<Double> generate(SignalType signalType, int length, SignalParameters parameters) throws SignalProcessingException {
        switch (signalType) {
            case SINE:
                return generateSine(length, parameters);
            case COSINE:
                return generateCosine(length, parameters);
            case SQUARE:
                return generateSquare(length, parameters);
            case TRIANGLE:
                return generateTriangle(length, parameters);
            case SAWTOOTH:
                return generateSawtooth(length, parameters);
            case WHITE_NOISE:
                return generateWhiteNoise(length, parameters);
            case STEP:
                return generateStep(length, parameters);
            case CHIRP:
                return generateChirp(length, parameters);
            case PULSE:
                return generatePulse(length, parameters);
            case DIRAC_DELTA:
            case KRONECKER_DELTA:
                return generateImpulse(length, parameters);
            default:
                throw new SignalProcessingException("不支持的信号类型 / Unsupported signal type: " + signalType);
        }
    }
    
    /**
     * 生成正弦波 / Generate sine wave
     */
    private IVector<Double> generateSine(int length, SignalParameters parameters) {
        double amplitude = parameters.getAmplitude();
        double frequency = parameters.getFrequency();
        double phase = parameters.getPhase();
        double offset = parameters.getOffset();
        double samplingRate = parameters.getSamplingRate();
        
        IVector<Double> signal = Linalg.zeros(length);
        for (int i = 0; i < length; i++) {
            double t = i / samplingRate;
            double value = amplitude * Math.sin(2 * Math.PI * frequency * t + phase) + offset;
            signal.set(i, value);
        }
        return signal;
    }
    
    /**
     * 生成余弦波 / Generate cosine wave
     */
    private IVector<Double> generateCosine(int length, SignalParameters parameters) {
        double amplitude = parameters.getAmplitude();
        double frequency = parameters.getFrequency();
        double phase = parameters.getPhase();
        double offset = parameters.getOffset();
        double samplingRate = parameters.getSamplingRate();
        
        IVector<Double> signal = Linalg.zeros(length);
        for (int i = 0; i < length; i++) {
            double t = i / samplingRate;
            double value = amplitude * Math.cos(2 * Math.PI * frequency * t + phase) + offset;
            signal.set(i, value);
        }
        return signal;
    }
    
    /**
     * 生成方波 / Generate square wave
     */
    private IVector<Double> generateSquare(int length, SignalParameters parameters) {
        double amplitude = parameters.getAmplitude();
        double frequency = parameters.getFrequency();
        double dutyCycle = parameters.getDutyCycle();
        double offset = parameters.getOffset();
        double samplingRate = parameters.getSamplingRate();
        
        IVector<Double> signal = Linalg.zeros(length);
        for (int i = 0; i < length; i++) {
            double t = i / samplingRate;
            double period = 1.0 / frequency;
            double phase = (t % period) / period;
            double value = (phase < dutyCycle) ? amplitude : -amplitude;
            value += offset;
            signal.set(i, value);
        }
        return signal;
    }
    
    /**
     * 生成三角波 / Generate triangle wave
     */
    private IVector<Double> generateTriangle(int length, SignalParameters parameters) {
        double amplitude = parameters.getAmplitude();
        double frequency = parameters.getFrequency();
        double symmetry = parameters.getDutyCycle(); // Using dutyCycle as symmetry parameter
        double offset = parameters.getOffset();
        double samplingRate = parameters.getSamplingRate();
        
        IVector<Double> signal = Linalg.zeros(length);
        for (int i = 0; i < length; i++) {
            double t = i / samplingRate;
            double period = 1.0 / frequency;
            double phase = (t % period) / period;
            
            double value;
            if (phase < symmetry) {
                value = -amplitude + 2 * amplitude * phase / symmetry;
            } else {
                value = amplitude - 2 * amplitude * (phase - symmetry) / (1 - symmetry);
            }
            value += offset;
            signal.set(i, value);
        }
        return signal;
    }
    
    /**
     * 生成锯齿波 / Generate sawtooth wave
     */
    private IVector<Double> generateSawtooth(int length, SignalParameters parameters) {
        double amplitude = parameters.getAmplitude();
        double frequency = parameters.getFrequency();
        double offset = parameters.getOffset();
        double samplingRate = parameters.getSamplingRate();
        
        IVector<Double> signal = Linalg.zeros(length);
        for (int i = 0; i < length; i++) {
            double t = i / samplingRate;
            double period = 1.0 / frequency;
            double phase = (t % period) / period;
            double value = amplitude * (2 * phase - 1);
            value += offset;
            signal.set(i, value);
        }
        return signal;
    }
    
    /**
     * 生成白噪声 / Generate white noise
     */
    private IVector<Double> generateWhiteNoise(int length, SignalParameters parameters) {
        double amplitude = parameters.getAmplitude();
        double offset = parameters.getOffset();
        
        IVector<Double> signal = Linalg.randn(length);
        // Scale and offset the noise
        signal = signal.multiplyScalar(amplitude).addScalar(offset);
        return signal;
    }
    
    /**
     * 生成阶跃信号 / Generate step signal
     */
    private IVector<Double> generateStep(int length, SignalParameters parameters) {
        double amplitude = parameters.getAmplitude();
        double offset = parameters.getOffset();
        double stepTime = parameters.getStepTime();
        double samplingRate = parameters.getSamplingRate();
        
        IVector<Double> signal = Linalg.zeros(length);
        for (int i = 0; i < length; i++) {
            double t = i / samplingRate;
            double value = (t >= stepTime) ? amplitude : 0.0;
            value += offset;
            signal.set(i, value);
        }
        return signal;
    }
    
    /**
     * 生成线性调频信号 / Generate chirp signal
     */
    private IVector<Double> generateChirp(int length, SignalParameters parameters) {
        double amplitude = parameters.getAmplitude();
        double startFreq = parameters.getStartFrequency();
        double endFreq = parameters.getEndFrequency();
        double offset = parameters.getOffset();
        double samplingRate = parameters.getSamplingRate();
        
        IVector<Double> signal = Linalg.zeros(length);
        double k = (endFreq - startFreq) / (length / samplingRate); // Rate of frequency change
        
        for (int i = 0; i < length; i++) {
            double t = i / samplingRate;
            double freq = startFreq + k * t / 2; // Instantaneous frequency
            double phase = 2 * Math.PI * (startFreq * t + k * t * t / 2);
            double value = amplitude * Math.sin(phase) + offset;
            signal.set(i, value);
        }
        return signal;
    }
    
    /**
     * 生成脉冲信号 / Generate pulse signal
     */
    private IVector<Double> generatePulse(int length, SignalParameters parameters) {
        double amplitude = parameters.getAmplitude();
        double offset = parameters.getOffset();
        int pulseWidth = parameters.getPulseWidth();
        double frequency = parameters.getFrequency();
        double samplingRate = parameters.getSamplingRate();
        
        IVector<Double> signal = Linalg.zeros(length);
        double period = samplingRate / frequency; // Samples per period
        
        for (int i = 0; i < length; i++) {
            double t = i % period;
            double value = (t < pulseWidth) ? amplitude : 0.0;
            value += offset;
            signal.set(i, value);
        }
        return signal;
    }
    
    /**
     * 生成脉冲/冲激信号 / Generate impulse signal
     */
    private IVector<Double> generateImpulse(int length, SignalParameters parameters) {
        double amplitude = parameters.getAmplitude();
        double offset = parameters.getOffset();
        
        IVector<Double> signal = Linalg.zeros(length);
        // Place impulse at the beginning (index 0)
        signal.set(0, amplitude + offset);
        return signal;
    }
    
    @Override
    public IVector<Double> generateComposite(SignalType[] signalTypes, int length, SignalParameters[] parameters) throws SignalProcessingException {
        if (signalTypes.length != parameters.length) {
            throw new SignalProcessingException("信号类型和参数数组长度不匹配 / Signal type and parameter array lengths do not match");
        }
        
        // Generate first signal
        IVector<Double> composite = generate(signalTypes[0], length, parameters[0]);
        
        // Add remaining signals
        for (int i = 1; i < signalTypes.length; i++) {
            IVector<Double> signal = generate(signalTypes[i], length, parameters[i]);
            composite = composite.add(signal);
        }
        
        return composite;
    }
    
    @Override
    public IVector<Double> addNoise(IVector<Double> signal, SignalType noiseType, SignalParameters parameters) throws SignalProcessingException {
        IVector<Double> noise = generate(noiseType, signal.length(), parameters);
        return signal.add(noise);
    }
    
    @Override
    public boolean validateParameters(SignalType signalType, int length, SignalParameters parameters) {
        // Basic validation
        if (length <= 0) {
            return false;
        }
        if (parameters.getAmplitude() < 0) {
            return false;
        }
        if (parameters.getFrequency() < 0) {
            return false;
        }
        if (parameters.getSamplingRate() <= 0) {
            return false;
        }
        if (parameters.getDutyCycle() < 0 || parameters.getDutyCycle() > 1) {
            return false;
        }
        return true;
    }
    
    @Override
    public SignalType[] getSupportedSignalTypes() {
        return new SignalType[] {
            SignalType.SINE,
            SignalType.COSINE,
            SignalType.SQUARE,
            SignalType.TRIANGLE,
            SignalType.SAWTOOTH,
            SignalType.WHITE_NOISE,
            SignalType.STEP,
            SignalType.CHIRP,
            SignalType.PULSE,
            SignalType.DIRAC_DELTA,
            SignalType.KRONECKER_DELTA
        };
    }
    
    @Override
    public String getName() {
        return "SignalGenerator";
    }
    
    @Override
    protected IVector<Double> doProcess(IVector<Double> input) throws SignalProcessingException {
        // For a generator, processing an input doesn't make much sense
        // We'll just return the input as-is
        return input;
    }
    
    @Override
    public SignalGenerator clone() {
        return new SignalGenerator();
    }
}