package com.yishape.lab.math.signal;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.signal.factory.SignalProcessorFactory;
import com.yishape.lab.math.signal.filter.ButterworthFilter;
import com.yishape.lab.math.signal.filter.GaussianFilter;
import com.yishape.lab.math.signal.filter.ISignalFilter;

/**
 * 信号滤波包装器 / Signal Filtering Wrapper.
 * 提供统一的信号滤波入口，通过工厂模式或直接构造创建滤波器实例。
 */
public class FiltWrapper {

    public IVector<Double> movingAverage(IVector<Double> signal, int windowSize) {
        try {
            ISignalFilter<Double> filter = SignalProcessorFactory.getInstance().createFilter("movingaverage");
            return filter.process(signal);
        } catch (Exception e) {
            throw new RuntimeException("Failed to apply moving average filter", e);
        }
    }

    public IVector<Double> medianFilter(IVector<Double> signal, int windowSize) {
        try {
            ISignalFilter<Double> filter = SignalProcessorFactory.getInstance().createFilter("median");
            return filter.process(signal);
        } catch (Exception e) {
            throw new RuntimeException("Failed to apply median filter", e);
        }
    }

    public IVector<Double> gaussianFilter(IVector<Double> signal, double sigma) {
        int defaultKernelSize = (int) Math.ceil(6 * sigma);
        if (defaultKernelSize % 2 == 0) {
            defaultKernelSize++;
        }
        return gaussianFilter(signal, sigma, defaultKernelSize);
    }

    public IVector<Double> gaussianFilter(IVector<Double> signal, double sigma, int kernelSize) {
        try {
            GaussianFilter filter = new GaussianFilter(sigma, kernelSize);
            return filter.filter(signal);
        } catch (Exception e) {
            throw new RuntimeException("Failed to apply gaussian filter", e);
        }
    }

    public IVector<Double> butterworthLowPass(IVector<Double> signal, double cutoffFreq, double samplingRate, int order) {
        try {
            ButterworthFilter filter = new ButterworthFilter(
                    ISignalFilter.FilterType.LOW_PASS, order, new double[]{cutoffFreq}, samplingRate);
            return filter.filter(signal);
        } catch (Exception e) {
            throw new RuntimeException("Failed to apply butterworth low-pass filter", e);
        }
    }

    public IVector<Double> butterworthHighPass(IVector<Double> signal, double cutoffFreq, double samplingRate, int order) {
        try {
            ButterworthFilter filter = new ButterworthFilter(
                    ISignalFilter.FilterType.HIGH_PASS, order, new double[]{cutoffFreq}, samplingRate);
            return filter.filter(signal);
        } catch (Exception e) {
            throw new RuntimeException("Failed to apply butterworth high-pass filter", e);
        }
    }

    public IVector<Double> bandPass(IVector<Double> signal, double lowFreq, double highFreq, double samplingRate, int order) {
        try {
            ISignalFilter<Double> filter = SignalProcessorFactory.getInstance().createFilter("bandpass");
            return filter.process(signal);
        } catch (Exception e) {
            throw new RuntimeException("Failed to apply bandpass filter", e);
        }
    }

    public IVector<Double> kalmanFilter(IVector<Double> signal, double processNoiseVariance, double measurementNoiseVariance) {
        try {
            ISignalFilter<Double> filter = SignalProcessorFactory.getInstance().createFilter("kalman");
            return filter.process(signal);
        } catch (Exception e) {
            throw new RuntimeException("Failed to apply Kalman filter", e);
        }
    }

    public IVector<Double> wienerFilter(IVector<Double> signal, double signalPower, double noisePower, int filterLength) {
        try {
            ISignalFilter<Double> filter = SignalProcessorFactory.getInstance().createFilter("wiener");
            return filter.process(signal);
        } catch (Exception e) {
            throw new RuntimeException("Failed to apply Wiener filter", e);
        }
    }

    public IVector<Double> bandStop(IVector<Double> signal, double lowFreq, double highFreq, double samplingRate, int order) {
        try {
            ISignalFilter<Double> filter = SignalProcessorFactory.getInstance().createFilter("bandstop");
            return filter.process(signal);
        } catch (Exception e) {
            throw new RuntimeException("Failed to apply band-stop filter", e);
        }
    }
}
