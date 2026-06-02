package com.yishape.lab.math.signal;

import com.yishape.lab.util.YishapeLogger;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.plot.IPlot;
import com.yishape.lab.math.signal.analysis.ISignalAnalyzer;
import com.yishape.lab.math.signal.core.Complex;
import com.yishape.lab.math.signal.factory.SignalProcessorFactory;
import com.yishape.lab.math.signal.filter.ISignalFilter;
import com.yishape.lab.math.signal.generation.ISignalGenerator;
import com.yishape.lab.math.signal.transform.ISignalTransform;
import com.yishape.lab.math.signal.wavele.WaveletAnalysis;
import com.yishape.lab.math.signal.wavele.WaveletCoefficients;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.util.Tuple3;

/**
 * 信号处理入口门面类 / Signal Processing Entry Facade Class.
 * 采用两级门面模式：顶层类提供 public static 子域 Wrapper 字段。
 * Two-tier facade pattern: top-level class provides public static sub-domain Wrapper fields.
 *
 * <p>使用示例 / Usage:
 * <pre>{@code
 *   IVector<Double> sine = Signals.gen.sineWave(100, 10, 1000, 1.0, 0);
 *   IVector<Double> filtered = Signals.filt.butterworthLowPass(sine, 50, 1000, 4);
 *   Complex[] spectrum = Signals.xform.fft(complexSignal);
 *   Tuple2<IVector<Double>, IVector<Double>> psd = Signals.analyze.powerSpectralDensity(signal, 256, 0.5, 1000);
 *   IPlot plot = Signals.plot.plotWaveform(signal, 1000, "Waveform");
 * }</pre>
 *
 * @author lteb2
 * @version 2.0
 * @since 1.0
 */
public class Signals {

    private static final YishapeLogger log = YishapeLogger.getLogger(Signals.class);

    // ========== 两级门面：子域 Wrapper 字段 / Two-tier Facade: Sub-domain Wrapper Fields ==========

    /** 信号生成 / Signal Generation */
    public static GenWrapper gen = new GenWrapper();

    /** 信号滤波 / Signal Filtering */
    public static FiltWrapper filt = new FiltWrapper();

    /** 信号变换 / Signal Transform */
    public static XformWrapper xform = new XformWrapper();

    /** 信号分析 / Signal Analysis */
    public static AnalyzeWrapper analyze = new AnalyzeWrapper();

    /** 信号可视化 / Signal Visualization */
    public static PlotWrapper plot = new PlotWrapper();

    // ========== @Deprecated 静态方法 — 向后兼容 / Deprecated Static Methods — Backward Compatibility ==========

    /** @deprecated Use {@link GenWrapper#sineWave} via {@code Signals.gen.sineWave(...)} */
    @Deprecated
    public static IVector<Double> sineWave(int length, double frequency, double samplingRate,
                                         double amplitude, double phase) {
        return gen.sineWave(length, frequency, samplingRate, amplitude, phase);
    }

    /** @deprecated Use {@link GenWrapper#cosineWave} via {@code Signals.gen.cosineWave(...)} */
    @Deprecated
    public static IVector<Double> cosineWave(int length, double frequency, double samplingRate,
                                           double amplitude, double phase) {
        return gen.cosineWave(length, frequency, samplingRate, amplitude, phase);
    }

    /** @deprecated Use {@link GenWrapper#squareWave} via {@code Signals.gen.squareWave(...)} */
    @Deprecated
    public static IVector<Double> squareWave(int length, double frequency, double samplingRate,
                                           double amplitude, double dutyCycle) {
        return gen.squareWave(length, frequency, samplingRate, amplitude, dutyCycle);
    }

    /** @deprecated Use {@link GenWrapper#triangularWave} via {@code Signals.gen.triangularWave(...)} */
    @Deprecated
    public static IVector<Double> triangularWave(int length, double frequency, double samplingRate,
                                               double amplitude, double symmetry) {
        return gen.triangularWave(length, frequency, samplingRate, amplitude, symmetry);
    }

    /** @deprecated Use {@link GenWrapper#sawtoothWave} via {@code Signals.gen.sawtoothWave(...)} */
    @Deprecated
    public static IVector<Double> sawtoothWave(int length, double frequency, double samplingRate, double amplitude) {
        return gen.sawtoothWave(length, frequency, samplingRate, amplitude);
    }

    /** @deprecated Use {@link GenWrapper#whiteNoise} via {@code Signals.gen.whiteNoise(...)} */
    @Deprecated
    public static IVector<Double> whiteNoise(int length, double power) {
        return gen.whiteNoise(length, power);
    }

    /** @deprecated Use {@link GenWrapper#pinkNoise} via {@code Signals.gen.pinkNoise(...)} */
    @Deprecated
    public static IVector<Double> pinkNoise(int length, double power) {
        return gen.pinkNoise(length, power);
    }

    /** @deprecated Use {@link GenWrapper#compositeSignal} via {@code Signals.gen.compositeSignal(...)} */
    @Deprecated
    public static IVector<Double> compositeSignal(ISignalGenerator.SignalType[] signalTypes,
                                                int length,
                                                ISignalGenerator.SignalParameters[] parameters) {
        return gen.compositeSignal(signalTypes, length, parameters);
    }

    /** @deprecated Use {@link GenWrapper#addNoise} via {@code Signals.gen.addNoise(...)} */
    @Deprecated
    public static IVector<Double> addNoise(IVector<Double> signal,
                                         ISignalGenerator.SignalType noiseType,
                                         ISignalGenerator.SignalParameters parameters) {
        return gen.addNoise(signal, noiseType, parameters);
    }

    /** @deprecated Use {@link GenWrapper#stepSignal} via {@code Signals.gen.stepSignal(...)} */
    @Deprecated
    public static IVector<Double> stepSignal(int length, double amplitude, double stepTime, double samplingRate) {
        return gen.stepSignal(length, amplitude, stepTime, samplingRate);
    }

    /** @deprecated Use {@link GenWrapper#unitStep} via {@code Signals.gen.unitStep(...)} */
    @Deprecated
    public static IVector<Double> unitStep(int length, double stepTime, double samplingRate) {
        return gen.unitStep(length, stepTime, samplingRate);
    }

    /** @deprecated Use {@link GenWrapper#diracDelta} via {@code Signals.gen.diracDelta(...)} */
    @Deprecated
    public static IVector<Double> diracDelta(int length, int impulseIndex, double amplitude) {
        return gen.diracDelta(length, impulseIndex, amplitude);
    }

    /** @deprecated Use {@link GenWrapper#unitImpulse} via {@code Signals.gen.unitImpulse(...)} */
    @Deprecated
    public static IVector<Double> unitImpulse(int length, int impulseIndex) {
        return gen.unitImpulse(length, impulseIndex);
    }

    /** @deprecated Use {@link GenWrapper#chirpSignal} via {@code Signals.gen.chirpSignal(...)} */
    @Deprecated
    public static IVector<Double> chirpSignal(int length, double startFreq, double endFreq,
                                            double samplingRate, double amplitude) {
        return gen.chirpSignal(length, startFreq, endFreq, samplingRate, amplitude);
    }

    /** @deprecated Use {@link GenWrapper#pulseSignal} via {@code Signals.gen.pulseSignal(...)} */
    @Deprecated
    public static IVector<Double> pulseSignal(int length, double amplitude, int pulseWidth,
                                            double frequency, double samplingRate) {
        return gen.pulseSignal(length, amplitude, pulseWidth, frequency, samplingRate);
    }

    // ========== @Deprecated 滤波方法 / Deprecated Filtering Methods ==========

    /** @deprecated Use {@link FiltWrapper#movingAverage} via {@code Signals.filt.movingAverage(...)} */
    @Deprecated
    public static IVector<Double> movingAverage(IVector<Double> signal, int windowSize) {
        return filt.movingAverage(signal, windowSize);
    }

    /** @deprecated Use {@link FiltWrapper#medianFilter} via {@code Signals.filt.medianFilter(...)} */
    @Deprecated
    public static IVector<Double> medianFilter(IVector<Double> signal, int windowSize) {
        return filt.medianFilter(signal, windowSize);
    }

    /** @deprecated Use {@link FiltWrapper#gaussianFilter(IVector, double)} via {@code Signals.filt.gaussianFilter(...)} */
    @Deprecated
    public static IVector<Double> gaussianFilter(IVector<Double> signal, double sigma) {
        return filt.gaussianFilter(signal, sigma);
    }

    /** @deprecated Use {@link FiltWrapper#gaussianFilter(IVector, double, int)} via {@code Signals.filt.gaussianFilter(...)} */
    @Deprecated
    public static IVector<Double> gaussianFilter(IVector<Double> signal, double sigma, int kernelSize) {
        return filt.gaussianFilter(signal, sigma, kernelSize);
    }

    /** @deprecated Use {@link FiltWrapper#butterworthLowPass} via {@code Signals.filt.butterworthLowPass(...)} */
    @Deprecated
    public static IVector<Double> butterworthLowPass(IVector<Double> signal, double cutoffFreq, double samplingRate, int order) {
        return filt.butterworthLowPass(signal, cutoffFreq, samplingRate, order);
    }

    /** @deprecated Use {@link FiltWrapper#butterworthHighPass} via {@code Signals.filt.butterworthHighPass(...)} */
    @Deprecated
    public static IVector<Double> butterworthHighPass(IVector<Double> signal, double cutoffFreq, double samplingRate, int order) {
        return filt.butterworthHighPass(signal, cutoffFreq, samplingRate, order);
    }

    /** @deprecated Use {@link FiltWrapper#bandPass} via {@code Signals.filt.bandPass(...)} */
    @Deprecated
    public static IVector<Double> bandPass(IVector<Double> signal, double lowFreq, double highFreq, double samplingRate, int order) {
        return filt.bandPass(signal, lowFreq, highFreq, samplingRate, order);
    }

    /** @deprecated Use {@link FiltWrapper#kalmanFilter} via {@code Signals.filt.kalmanFilter(...)} */
    @Deprecated
    public static IVector<Double> kalmanFilter(IVector<Double> signal, double processNoiseVariance, double measurementNoiseVariance) {
        return filt.kalmanFilter(signal, processNoiseVariance, measurementNoiseVariance);
    }

    /** @deprecated Use {@link FiltWrapper#wienerFilter} via {@code Signals.filt.wienerFilter(...)} */
    @Deprecated
    public static IVector<Double> wienerFilter(IVector<Double> signal, double signalPower, double noisePower, int filterLength) {
        return filt.wienerFilter(signal, signalPower, noisePower, filterLength);
    }

    /** @deprecated Use {@link FiltWrapper#bandStop} via {@code Signals.filt.bandStop(...)} */
    @Deprecated
    public static IVector<Double> bandStop(IVector<Double> signal, double lowFreq, double highFreq, double samplingRate, int order) {
        return filt.bandStop(signal, lowFreq, highFreq, samplingRate, order);
    }

    // ========== @Deprecated 分析方法 / Deprecated Analysis Methods ==========

    /** @deprecated Use {@link AnalyzeWrapper#powerSpectralDensity} via {@code Signals.analyze.powerSpectralDensity(...)} */
    @Deprecated
    public static Tuple2<IVector<Double>, IVector<Double>> powerSpectralDensity(
            IVector<Double> signal, int windowSize, double overlap, double samplingRate) {
        return analyze.powerSpectralDensity(signal, windowSize, overlap, samplingRate);
    }

    /** @deprecated Use {@link AnalyzeWrapper#autocorrelation} via {@code Signals.analyze.autocorrelation(...)} */
    @Deprecated
    public static IVector<Double> autocorrelation(IVector<Double> signal) {
        return analyze.autocorrelation(signal);
    }

    /** @deprecated Use {@link AnalyzeWrapper#crossCorrelation(IVector, IVector)} via {@code Signals.analyze.crossCorrelation(...)} */
    @Deprecated
    public static IVector<Double> crossCorrelation(IVector<Double> signal1, IVector<Double> signal2) {
        return analyze.crossCorrelation(signal1, signal2);
    }

    /** @deprecated Use {@link AnalyzeWrapper#crossCorrelation(IVector, IVector, int)} via {@code Signals.analyze.crossCorrelation(...)} */
    @Deprecated
    public static IVector<Double> crossCorrelation(IVector<Double> signal1, IVector<Double> signal2, int maxLag) {
        return analyze.crossCorrelation(signal1, signal2, maxLag);
    }

    /** @deprecated Use {@link AnalyzeWrapper#spectrum} via {@code Signals.analyze.spectrum(...)} */
    @Deprecated
    public static Tuple3<IVector<Double>, IVector<Double>, IVector<Double>> spectrum(
            IVector<Double> signal, double samplingRate) {
        return analyze.spectrum(signal, samplingRate);
    }

    /** @deprecated Use {@link AnalyzeWrapper#shortTimeFourierTransform} via {@code Signals.analyze.shortTimeFourierTransform(...)} */
    @Deprecated
    public static IMatrix<Double> shortTimeFourierTransform(
            IVector<Double> signal, int windowSize, int hopSize, double samplingRate) {
        return analyze.shortTimeFourierTransform(signal, windowSize, hopSize, samplingRate);
    }

    /** @deprecated Use {@link AnalyzeWrapper#signalToNoiseRatio} via {@code Signals.analyze.signalToNoiseRatio(...)} */
    @Deprecated
    public static double signalToNoiseRatio(IVector<Double> signal, IVector<Double> noise) {
        return analyze.signalToNoiseRatio(signal, noise);
    }

    /** @deprecated Use {@link AnalyzeWrapper#peakSignalToNoiseRatio} via {@code Signals.analyze.peakSignalToNoiseRatio(...)} */
    @Deprecated
    public static double peakSignalToNoiseRatio(IVector<Double> original, IVector<Double> reconstructed) {
        return analyze.peakSignalToNoiseRatio(original, reconstructed);
    }

    // ========== @Deprecated 变换方法 / Deprecated Transform Methods ==========

    /** @deprecated Use {@link XformWrapper#fft} via {@code Signals.xform.fft(...)} */
    @Deprecated
    public static Complex[] fft(Complex[] x) {
        return xform.fft(x);
    }

    /** @deprecated Use {@link XformWrapper#ifft} via {@code Signals.xform.ifft(...)} */
    @Deprecated
    public static Complex[] ifft(Complex[] x) {
        return xform.ifft(x);
    }

    /** @deprecated Use {@link XformWrapper#magnitudeSpectrum} via {@code Signals.xform.magnitudeSpectrum(...)} */
    @Deprecated
    public static double[] magnitudeSpectrum(Complex[] fftResult) {
        return xform.magnitudeSpectrum(fftResult);
    }

    /** @deprecated Use {@link XformWrapper#phaseSpectrum} via {@code Signals.xform.phaseSpectrum(...)} */
    @Deprecated
    public static double[] phaseSpectrum(Complex[] fftResult) {
        return xform.phaseSpectrum(fftResult);
    }

    /** @deprecated Use {@link XformWrapper#powerSpectrum} via {@code Signals.xform.powerSpectrum(...)} */
    @Deprecated
    public static double[] powerSpectrum(Complex[] fftResult) {
        return xform.powerSpectrum(fftResult);
    }

    /** @deprecated Use {@link XformWrapper#dct2} via {@code Signals.xform.dct2(...)} */
    @Deprecated
    public static IVector<Double> dct2(IVector<Double> signal) {
        return xform.dct2(signal);
    }

    /** @deprecated Use {@link XformWrapper#idct2} via {@code Signals.xform.idct2(...)} */
    @Deprecated
    public static IVector<Double> idct2(IVector<Double> dctSignal) {
        return xform.idct2(dctSignal);
    }

    /** @deprecated Use {@link XformWrapper#hilbertTransform} via {@code Signals.xform.hilbertTransform(...)} */
    @Deprecated
    public static IVector<Double> hilbertTransform(IVector<Double> signal) {
        return xform.hilbertTransform(signal);
    }

    /** @deprecated Use {@link XformWrapper#analyticSignal} via {@code Signals.xform.analyticSignal(...)} */
    @Deprecated
    public static Complex[] analyticSignal(IVector<Double> signal) {
        return xform.analyticSignal(signal);
    }

    /** @deprecated Use {@link XformWrapper#instantaneousAmplitude} via {@code Signals.xform.instantaneousAmplitude(...)} */
    @Deprecated
    public static IVector<Double> instantaneousAmplitude(IVector<Double> signal) {
        return xform.instantaneousAmplitude(signal);
    }

    /** @deprecated Use {@link XformWrapper#instantaneousPhase} via {@code Signals.xform.instantaneousPhase(...)} */
    @Deprecated
    public static IVector<Double> instantaneousPhase(IVector<Double> signal) {
        return xform.instantaneousPhase(signal);
    }

    /** @deprecated Use {@link XformWrapper#instantaneousFrequency} via {@code Signals.xform.instantaneousFrequency(...)} */
    @Deprecated
    public static IVector<Double> instantaneousFrequency(IVector<Double> signal, double samplingRate) {
        return xform.instantaneousFrequency(signal, samplingRate);
    }

    /** @deprecated Use {@link XformWrapper#discreteWaveletTransform} via {@code Signals.xform.discreteWaveletTransform(...)} */
    @Deprecated
    public static WaveletCoefficients discreteWaveletTransform(IVector<Double> signal,
                                                               WaveletAnalysis.WaveletType waveletType, int levels, double param) {
        return xform.discreteWaveletTransform(signal, waveletType, levels, param);
    }

    /** @deprecated Use {@link XformWrapper#inverseDiscreteWaveletTransform} via {@code Signals.xform.inverseDiscreteWaveletTransform(...)} */
    @Deprecated
    public static IVector<Double> inverseDiscreteWaveletTransform(WaveletCoefficients coefficients,
            WaveletAnalysis.WaveletType waveletType, double param) {
        return xform.inverseDiscreteWaveletTransform(coefficients, waveletType, param);
    }

    // ========== @Deprecated 可视化方法 / Deprecated Visualization Methods ==========

    /** @deprecated Use {@link PlotWrapper#plotWaveform} via {@code Signals.plot.plotWaveform(...)} */
    @Deprecated
    public static IPlot plotWaveform(IVector<Double> signal, double samplingRate, String title) {
        return plot.plotWaveform(signal, samplingRate, title);
    }

    /** @deprecated Use {@link PlotWrapper#plotSpectrum} via {@code Signals.plot.plotSpectrum(...)} */
    @Deprecated
    public static IPlot plotSpectrum(IVector<Double> signal, double samplingRate, String title) {
        return plot.plotSpectrum(signal, samplingRate, title);
    }

    /** @deprecated Use {@link PlotWrapper#plotPowerSpectralDensity} via {@code Signals.plot.plotPowerSpectralDensity(...)} */
    @Deprecated
    public static IPlot plotPowerSpectralDensity(IVector<Double> signal, double samplingRate,
                                               int windowSize, double overlap, String title) {
        return plot.plotPowerSpectralDensity(signal, samplingRate, windowSize, overlap, title);
    }

    /** @deprecated Use {@link PlotWrapper#plotAutocorrelation} via {@code Signals.plot.plotAutocorrelation(...)} */
    @Deprecated
    public static IPlot plotAutocorrelation(IVector<Double> signal, int maxLag, String title) {
        return plot.plotAutocorrelation(signal, maxLag, title);
    }

    /** @deprecated Use {@link PlotWrapper#plotCrossCorrelation} via {@code Signals.plot.plotCrossCorrelation(...)} */
    @Deprecated
    public static IPlot plotCrossCorrelation(IVector<Double> signal1, IVector<Double> signal2,
                                           int maxLag, String title) {
        return plot.plotCrossCorrelation(signal1, signal2, maxLag, title);
    }

    /** @deprecated Use {@link PlotWrapper#plotWaveletCoefficients} via {@code Signals.plot.plotWaveletCoefficients(...)} */
    @Deprecated
    public static IPlot plotWaveletCoefficients(IVector<Double> signal, String waveletType,
                                              int levels, String title) {
        return plot.plotWaveletCoefficients(signal, waveletType, levels, title);
    }

    /** @deprecated Use {@link PlotWrapper#plotWaveletEnergyDistribution} via {@code Signals.plot.plotWaveletEnergyDistribution(...)} */
    @Deprecated
    public static IPlot plotWaveletEnergyDistribution(IVector<Double> signal, String waveletType,
                                                    int levels, String title) {
        return plot.plotWaveletEnergyDistribution(signal, waveletType, levels, title);
    }

    // ========== 工厂方法 / Factory Methods ==========

    public static SignalProcessorFactory getFactory() {
        return SignalProcessorFactory.getInstance();
    }

    public static <T extends Number, R> ISignalTransform<T, R> createTransform(String transformType) throws Exception {
        return SignalProcessorFactory.getInstance().createTransform(transformType);
    }

    public static <T extends Number> ISignalFilter<T> createFilter(String filterType) throws Exception {
        return SignalProcessorFactory.getInstance().createFilter(filterType);
    }

    public static <T extends Number> ISignalAnalyzer<T> createAnalyzer(String analyzerType) throws Exception {
        return SignalProcessorFactory.getInstance().createAnalyzer(analyzerType);
    }

    public static <T extends Number> ISignalGenerator<T> createGenerator(String generatorType) throws Exception {
        return SignalProcessorFactory.getInstance().createGenerator(generatorType);
    }

    // ========== 测试方法 / Test Method ==========

    public static void main(String[] args) {
        try {
            log.debug("Testing signal generation...");
            IVector<Double> sineWave = Signals.gen.sineWave(100, 10, 1000, 1.0, 0);
            log.debug("Generated sine wave with {} samples", sineWave.length());

            IVector<Double> cosineWave = Signals.gen.cosineWave(100, 10, 1000, 1.0, 0);
            log.debug("Generated cosine wave with {} samples", cosineWave.length());

            IVector<Double> squareWave = Signals.gen.squareWave(100, 10, 1000, 1.0, 0.5);
            log.debug("Generated square wave with {} samples", squareWave.length());

            IVector<Double> noise = Signals.gen.whiteNoise(100, 0.1);
            log.debug("Generated white noise with {} samples", noise.length());

            log.debug("\nTesting signal filtering...");
            IVector<Double> filtered = Signals.filt.butterworthLowPass(sineWave, 50, 1000, 4);
            log.debug("Applied Butterworth filter");

            log.debug("\nTesting signal analysis...");
            IVector<Double> autocorr = Signals.analyze.autocorrelation(sineWave);
            log.debug("Calculated autocorrelation with {} samples", autocorr.length());

            log.debug("\nAll tests passed!");
        } catch (Exception e) {
            log.error("Signal test failed", e);
        }
    }
}
