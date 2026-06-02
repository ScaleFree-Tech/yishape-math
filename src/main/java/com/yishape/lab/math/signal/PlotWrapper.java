package com.yishape.lab.math.signal;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.plot.IPlot;
import com.yishape.lab.math.signal.wavele.WaveletPlots;

/**
 * 信号可视化包装器 / Signal Visualization Wrapper.
 * 提供统一的信号绘图入口。
 */
public class PlotWrapper {

    public IPlot plotWaveform(IVector<Double> signal, double samplingRate, String title) {
        return SignalPlots.plotWaveform(signal, samplingRate, title);
    }

    public IPlot plotSpectrum(IVector<Double> signal, double samplingRate, String title) {
        return SignalPlots.plotSpectrum(signal, samplingRate, title);
    }

    public IPlot plotPowerSpectralDensity(IVector<Double> signal, double samplingRate,
                                           int windowSize, double overlap, String title) {
        return SignalPlots.plotPowerSpectralDensity(signal, samplingRate, windowSize, overlap, title);
    }

    public IPlot plotAutocorrelation(IVector<Double> signal, int maxLag, String title) {
        return SignalPlots.plotAutocorrelation(signal, maxLag, title);
    }

    public IPlot plotCrossCorrelation(IVector<Double> signal1, IVector<Double> signal2,
                                       int maxLag, String title) {
        return SignalPlots.plotCrossCorrelation(signal1, signal2, maxLag, title);
    }

    public IPlot plotWaveletCoefficients(IVector<Double> signal, String waveletType,
                                          int levels, String title) {
        return WaveletPlots.plotWaveletCoefficients(signal, waveletType, levels, title);
    }

    public IPlot plotWaveletEnergyDistribution(IVector<Double> signal, String waveletType,
                                                int levels, String title) {
        return WaveletPlots.plotWaveletEnergyDistribution(signal, waveletType, levels, title);
    }
}
