package com.yishape.lab.math.signal;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.signal.factory.SignalProcessorFactory;
import com.yishape.lab.math.signal.generation.ISignalGenerator;

/**
 * 信号生成包装器 / Signal Generation Wrapper.
 * 提供统一的信号生成入口，通过工厂模式创建生成器实例。
 */
public class GenWrapper {

    private static final java.util.Random PINK_NOISE_RANDOM = new java.util.Random();

    public IVector<Double> sineWave(int length, double frequency, double samplingRate,
                                     double amplitude, double phase) {
        try {
            ISignalGenerator<Double> generator = SignalProcessorFactory.getInstance().createGenerator("sine");
            ISignalGenerator.SignalParameters params = new ISignalGenerator.SignalParameters()
                .frequency(frequency).samplingRate(samplingRate).amplitude(amplitude).phase(phase);
            return generator.generate(ISignalGenerator.SignalType.SINE, length, params);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate sine wave", e);
        }
    }

    public IVector<Double> cosineWave(int length, double frequency, double samplingRate,
                                       double amplitude, double phase) {
        try {
            ISignalGenerator<Double> generator = SignalProcessorFactory.getInstance().createGenerator("cosine");
            ISignalGenerator.SignalParameters params = new ISignalGenerator.SignalParameters()
                .frequency(frequency).samplingRate(samplingRate).amplitude(amplitude).phase(phase);
            return generator.generate(ISignalGenerator.SignalType.COSINE, length, params);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate cosine wave", e);
        }
    }

    public IVector<Double> squareWave(int length, double frequency, double samplingRate,
                                       double amplitude, double dutyCycle) {
        try {
            ISignalGenerator<Double> generator = SignalProcessorFactory.getInstance().createGenerator("square");
            ISignalGenerator.SignalParameters params = new ISignalGenerator.SignalParameters()
                .frequency(frequency).samplingRate(samplingRate).amplitude(amplitude).dutyCycle(dutyCycle);
            return generator.generate(ISignalGenerator.SignalType.SQUARE, length, params);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate square wave", e);
        }
    }

    public IVector<Double> triangularWave(int length, double frequency, double samplingRate,
                                           double amplitude, double symmetry) {
        try {
            ISignalGenerator<Double> generator = SignalProcessorFactory.getInstance().createGenerator("triangle");
            ISignalGenerator.SignalParameters params = new ISignalGenerator.SignalParameters()
                .frequency(frequency).samplingRate(samplingRate).amplitude(amplitude).dutyCycle(symmetry);
            return generator.generate(ISignalGenerator.SignalType.TRIANGLE, length, params);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate triangular wave", e);
        }
    }

    public IVector<Double> sawtoothWave(int length, double frequency, double samplingRate, double amplitude) {
        try {
            ISignalGenerator<Double> generator = SignalProcessorFactory.getInstance().createGenerator("sawtooth");
            ISignalGenerator.SignalParameters params = new ISignalGenerator.SignalParameters()
                .frequency(frequency).samplingRate(samplingRate).amplitude(amplitude);
            return generator.generate(ISignalGenerator.SignalType.SAWTOOTH, length, params);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate sawtooth wave", e);
        }
    }

    public IVector<Double> whiteNoise(int length, double power) {
        try {
            ISignalGenerator<Double> generator = SignalProcessorFactory.getInstance().createGenerator("noise");
            ISignalGenerator.SignalParameters params = new ISignalGenerator.SignalParameters()
                .amplitude(Math.sqrt(power));
            return generator.generate(ISignalGenerator.SignalType.WHITE_NOISE, length, params);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate white noise", e);
        }
    }

    public IVector<Double> pinkNoise(int length, double power) {
        int octaves = 16;
        double[] whiteOctaves = new double[octaves];
        double[] pinkNoiseData = new double[length];
        java.util.Random random = PINK_NOISE_RANDOM;
        double targetStd = Math.sqrt(power);

        for (int i = 0; i < length; i++) {
            int mask = i + 1;
            int octave = 0;
            while ((mask & 1) == 0 && octave < octaves - 1) {
                mask >>= 1;
                octave++;
            }
            whiteOctaves[octave] = random.nextGaussian() * targetStd;
            double sum = 0;
            for (int j = 0; j < octaves; j++) {
                sum += whiteOctaves[j];
            }
            sum += random.nextGaussian() * targetStd * 0.5;
            pinkNoiseData[i] = sum;
        }

        double actualPower = 0;
        for (int i = 0; i < length; i++) {
            actualPower += pinkNoiseData[i] * pinkNoiseData[i];
        }
        actualPower /= length;
        double scale = targetStd / Math.sqrt(actualPower);
        for (int i = 0; i < length; i++) {
            pinkNoiseData[i] *= scale;
        }
        return IVector.of(pinkNoiseData);
    }

    public IVector<Double> compositeSignal(ISignalGenerator.SignalType[] signalTypes,
                                            int length,
                                            ISignalGenerator.SignalParameters[] parameters) {
        try {
            ISignalGenerator<Double> generator = SignalProcessorFactory.getInstance().createGenerator("sine");
            return generator.generateComposite(signalTypes, length, parameters);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate composite signal", e);
        }
    }

    public IVector<Double> addNoise(IVector<Double> signal,
                                     ISignalGenerator.SignalType noiseType,
                                     ISignalGenerator.SignalParameters parameters) {
        try {
            ISignalGenerator<Double> generator = SignalProcessorFactory.getInstance().createGenerator("sine");
            return generator.addNoise(signal, noiseType, parameters);
        } catch (Exception e) {
            throw new RuntimeException("Failed to add noise to signal", e);
        }
    }

    public IVector<Double> stepSignal(int length, double amplitude, double stepTime, double samplingRate) {
        try {
            ISignalGenerator<Double> generator = SignalProcessorFactory.getInstance().createGenerator("signal");
            ISignalGenerator.SignalParameters params = new ISignalGenerator.SignalParameters()
                .amplitude(amplitude).stepTime(stepTime).samplingRate(samplingRate);
            return generator.generate(ISignalGenerator.SignalType.STEP, length, params);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate step signal", e);
        }
    }

    public IVector<Double> unitStep(int length, double stepTime, double samplingRate) {
        return stepSignal(length, 1.0, stepTime, samplingRate);
    }

    public IVector<Double> diracDelta(int length, int impulseIndex, double amplitude) {
        try {
            ISignalGenerator<Double> generator = SignalProcessorFactory.getInstance().createGenerator("signal");
            ISignalGenerator.SignalParameters params = new ISignalGenerator.SignalParameters()
                .amplitude(amplitude);
            IVector<Double> signal = Linalg.zeros(length);
            if (impulseIndex >= 0 && impulseIndex < length) {
                signal.set(impulseIndex, amplitude);
            }
            return signal;
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate Dirac delta signal", e);
        }
    }

    public IVector<Double> unitImpulse(int length, int impulseIndex) {
        return diracDelta(length, impulseIndex, 1.0);
    }

    public IVector<Double> chirpSignal(int length, double startFreq, double endFreq,
                                        double samplingRate, double amplitude) {
        try {
            ISignalGenerator<Double> generator = SignalProcessorFactory.getInstance().createGenerator("signal");
            ISignalGenerator.SignalParameters params = new ISignalGenerator.SignalParameters()
                .startFrequency(startFreq).endFrequency(endFreq)
                .samplingRate(samplingRate).amplitude(amplitude);
            return generator.generate(ISignalGenerator.SignalType.CHIRP, length, params);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate chirp signal", e);
        }
    }

    public IVector<Double> pulseSignal(int length, double amplitude, int pulseWidth,
                                        double frequency, double samplingRate) {
        try {
            ISignalGenerator<Double> generator = SignalProcessorFactory.getInstance().createGenerator("signal");
            ISignalGenerator.SignalParameters params = new ISignalGenerator.SignalParameters()
                .amplitude(amplitude).pulseWidth(pulseWidth)
                .frequency(frequency).samplingRate(samplingRate);
            return generator.generate(ISignalGenerator.SignalType.PULSE, length, params);
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate pulse signal", e);
        }
    }
}
