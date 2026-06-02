package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.signal.Signals;

/**
 * 时间序列滤波类 / Time Series Filtering Class
 *
 * @author lteb2
 * @version 2.0
 * @since 1.0
 */
public class TimeSeriesFiltering {

    private static final double DEFAULT_SAMPLING_RATE = 1.0;

    public static FilterResult movingAverage(IVector<Double> data, int windowSize) {
        IVector<Double> filtered = Signals.movingAverage(data, windowSize);
        IVector<Double> noise = data.sub(filtered);
        double signalPower = filtered.multiply(filtered).sumValue();
        double noisePower = noise.multiply(noise).sumValue();
        double snr = signalPower / (noisePower + 1e-10);
        return new FilterResult(filtered, noise, snr, "MovingAverage");
    }

    public static FilterResult exponentialSmoothing(IVector<Double> data, double alpha) {
        if (alpha < 0 || alpha > 1) {
            throw new IllegalArgumentException("平滑参数必须在0-1之间");
        }
        IVector<Double> filtered = exponentialSmoothingImpl(data, alpha);
        IVector<Double> noise = data.sub(filtered);
        double signalPower = filtered.multiply(filtered).sumValue();
        double noisePower = noise.multiply(noise).sumValue();
        double snr = signalPower / (noisePower + 1e-10);
        return new FilterResult(filtered, noise, snr, "ExponentialSmoothing");
    }

    public static FilterResult gaussianFilter(IVector<Double> data, double sigma) {
        IVector<Double> filtered = Signals.gaussianFilter(data, sigma);
        IVector<Double> noise = data.sub(filtered);
        double signalPower = filtered.multiply(filtered).sumValue();
        double noisePower = noise.multiply(noise).sumValue();
        double snr = signalPower / (noisePower + 1e-10);
        return new FilterResult(filtered, noise, snr, "Gaussian");
    }

    public static FilterResult medianFilter(IVector<Double> data, int windowSize) {
        IVector<Double> filtered = Signals.medianFilter(data, windowSize);
        IVector<Double> noise = data.sub(filtered);
        double signalPower = filtered.multiply(filtered).sumValue();
        double noisePower = noise.multiply(noise).sumValue();
        double snr = signalPower / (noisePower + 1e-10);
        return new FilterResult(filtered, noise, snr, "Median");
    }

    public static FilterResult lowPassFilter(IVector<Double> data,
            double cutoffFreq, double samplingRate, int order) {
        IVector<Double> filtered = Signals.butterworthLowPass(data, cutoffFreq, samplingRate, order);
        IVector<Double> noise = data.sub(filtered);
        double signalPower = filtered.multiply(filtered).sumValue();
        double noisePower = noise.multiply(noise).sumValue();
        double snr = signalPower / (noisePower + 1e-10);
        return new FilterResult(filtered, noise, snr, "LowPass");
    }

    public static FilterResult highPassFilter(IVector<Double> data,
            double cutoffFreq, double samplingRate, int order) {
        IVector<Double> filtered = Signals.butterworthHighPass(data, cutoffFreq, samplingRate, order);
        IVector<Double> noise = data.sub(filtered);
        double signalPower = filtered.multiply(filtered).sumValue();
        double noisePower = noise.multiply(noise).sumValue();
        double snr = signalPower / (noisePower + 1e-10);
        return new FilterResult(filtered, noise, snr, "HighPass");
    }

    public static FilterResult bandPassFilter(IVector<Double> data,
            double lowFreq, double highFreq, double samplingRate, int order) {
        IVector<Double> filtered = Signals.bandPass(data, lowFreq, highFreq, samplingRate, order);
        IVector<Double> noise = data.sub(filtered);
        double signalPower = filtered.multiply(filtered).sumValue();
        double noisePower = noise.multiply(noise).sumValue();
        double snr = signalPower / (noisePower + 1e-10);
        return new FilterResult(filtered, noise, snr, "BandPass");
    }

    public static FilterResult adaptiveFilter(IVector<Double> data, double learningRate) {
        IVector<Double> filtered = adaptiveFilterImpl(data, learningRate);
        IVector<Double> noise = data.sub(filtered);
        double signalPower = filtered.multiply(filtered).sumValue();
        double noisePower = noise.multiply(noise).sumValue();
        double snr = signalPower / (noisePower + 1e-10);
        return new FilterResult(filtered, noise, snr, "Adaptive");
    }

    // ========== private helpers ==========

    private static IVector<Double> exponentialSmoothingImpl(IVector<Double> data, double alpha) {
        int length = data.length();
        IVector<Double> smoothed = Linalg.zeros(length);
        smoothed.set(0, data.get(0));
        for (int i = 1; i < length; i++) {
            smoothed.set(i, alpha * data.get(i) + (1 - alpha) * smoothed.get(i - 1));
        }
        return smoothed;
    }

    private static IVector<Double> adaptiveFilterImpl(IVector<Double> data, double learningRate) {
        int length = data.length();
        IVector<Double> filtered = Linalg.zeros(length);
        filtered.set(0, data.get(0));
        for (int i = 1; i < length; i++) {
            double error = data.get(i) - filtered.get(i - 1);
            filtered.set(i, filtered.get(i - 1) + learningRate * error);
        }
        return filtered;
    }
}
