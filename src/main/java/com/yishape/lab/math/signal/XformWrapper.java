package com.yishape.lab.math.signal;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.signal.core.Complex;
import com.yishape.lab.math.signal.core.RereDCT;
import com.yishape.lab.math.signal.core.RereFFT;
import com.yishape.lab.math.signal.core.RereHilbert;
import com.yishape.lab.math.signal.wavele.WaveletAnalysis;
import com.yishape.lab.math.signal.wavele.WaveletCoefficients;

/**
 * 信号变换包装器 / Signal Transform Wrapper.
 * 提供统一的信号变换入口，包括 FFT、DCT、Hilbert、Wavelet 等变换。
 */
public class XformWrapper {

    public Complex[] fft(Complex[] x) {
        return RereFFT.fft(x);
    }

    public Complex[] ifft(Complex[] x) {
        return RereFFT.ifft(x);
    }

    public double[] magnitudeSpectrum(Complex[] fftResult) {
        return RereFFT.magnitudeSpectrum(fftResult);
    }

    public double[] phaseSpectrum(Complex[] fftResult) {
        return RereFFT.phaseSpectrum(fftResult);
    }

    public double[] powerSpectrum(Complex[] fftResult) {
        return RereFFT.powerSpectrum(fftResult);
    }

    public IVector<Double> dct2(IVector<Double> signal) {
        return RereDCT.dct2(signal);
    }

    public IVector<Double> idct2(IVector<Double> dctSignal) {
        return RereDCT.idct2(dctSignal);
    }

    public IVector<Double> hilbertTransform(IVector<Double> signal) {
        return RereHilbert.hilbertTransform(signal);
    }

    public Complex[] analyticSignal(IVector<Double> signal) {
        return RereHilbert.analyticSignal(signal);
    }

    public IVector<Double> instantaneousAmplitude(IVector<Double> signal) {
        return RereHilbert.instantaneousAmplitude(signal);
    }

    public IVector<Double> instantaneousPhase(IVector<Double> signal) {
        return RereHilbert.instantaneousPhase(signal);
    }

    public IVector<Double> instantaneousFrequency(IVector<Double> signal, double samplingRate) {
        return RereHilbert.instantaneousFrequency(signal, samplingRate);
    }

    public WaveletCoefficients discreteWaveletTransform(IVector<Double> signal,
                                                         WaveletAnalysis.WaveletType waveletType, int levels, double param) {
        return WaveletAnalysis.discreteWaveletTransform(signal, waveletType, levels, param);
    }

    public IVector<Double> inverseDiscreteWaveletTransform(WaveletCoefficients coefficients,
            WaveletAnalysis.WaveletType waveletType, double param) {
        return WaveletAnalysis.inverseDiscreteWaveletTransform(coefficients, waveletType, param);
    }
}
