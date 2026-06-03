package com.yishape.lab.math.signal;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.signal.analysis.CohenClassDistribution;
import com.yishape.lab.math.signal.analysis.ISignalAnalyzer;
import com.yishape.lab.math.signal.core.Complex;
import com.yishape.lab.math.signal.core.RereFFT;
import com.yishape.lab.math.signal.core.RereHilbert;
import com.yishape.lab.math.signal.core.SignalUtilities;
import com.yishape.lab.math.signal.factory.SignalProcessorFactory;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.util.Tuple3;

/**
 * 信号分析包装器 / Signal Analysis Wrapper.
 * 提供统一的信号分析入口，包括频谱分析、相关性分析、信噪比等。
 */
public class AnalyzeWrapper {

    public Tuple2<IVector<Double>, IVector<Double>> powerSpectralDensity(
            IVector<Double> signal, int windowSize, double overlap, double samplingRate) {
        try {
            ISignalAnalyzer<Double> analyzer = SignalProcessorFactory.getInstance().createAnalyzer("psd");
            ISignalAnalyzer.AnalysisParameters params = new ISignalAnalyzer.AnalysisParameters()
                .windowSize(windowSize).overlap(overlap).samplingRate(samplingRate);
            ISignalAnalyzer.AnalysisResult<Tuple2<IVector<Double>, IVector<Double>>> result =
                analyzer.analyze(signal, ISignalAnalyzer.AnalysisType.POWER_SPECTRUM, params);
            return result.getResult();
        } catch (Exception e) {
            throw new RuntimeException("Failed to calculate power spectral density", e);
        }
    }

    public IVector<Double> autocorrelation(IVector<Double> signal) {
        try {
            ISignalAnalyzer<Double> analyzer = SignalProcessorFactory.getInstance().createAnalyzer("autocorr");
            ISignalAnalyzer.AnalysisResult<IVector<Double>> result =
                analyzer.analyze(signal, ISignalAnalyzer.AnalysisType.AUTOCORRELATION);
            return result.getResult();
        } catch (Exception e) {
            throw new RuntimeException("Failed to calculate autocorrelation", e);
        }
    }

    public IVector<Double> crossCorrelation(IVector<Double> signal1, IVector<Double> signal2) {
        int n1 = signal1.length();
        int n2 = signal2.length();
        int n = n1 + n2;
        int fftSize = 1;
        while (fftSize < n) fftSize <<= 1;

        Complex[] x = new Complex[fftSize];
        Complex[] y = new Complex[fftSize];
        for (int i = 0; i < fftSize; i++) {
            x[i] = new Complex(i < n1 ? signal1.get(i) : 0, 0);
            y[i] = new Complex(i < n2 ? signal2.get(i) : 0, 0);
        }

        Complex[] fftX = RereFFT.fft(x);
        Complex[] fftY = RereFFT.fft(y);
        for (int i = 0; i < fftSize; i++) {
            fftX[i] = fftX[i].multiply(fftY[i].conjugate());
        }
        Complex[] result = RereFFT.ifft(fftX);

        double[] corr = new double[n - 1];
        for (int i = 0; i < n - 1; i++) {
            int idx = i < n2 ? n2 - 1 - i : i - n2 + 1;
            corr[i] = result[idx % fftSize].real;
        }
        return IVector.of(corr);
    }

    public IVector<Double> crossCorrelation(IVector<Double> signal1, IVector<Double> signal2, int maxLag) {
        IVector<Double> fullCorr = crossCorrelation(signal1, signal2);
        int fullLength = fullCorr.length();
        if (fullLength <= 2 * maxLag + 1) {
            return fullCorr;
        }
        int center = fullLength / 2;
        int halfWindow = maxLag;
        int start = Math.max(0, center - halfWindow);
        int end = Math.min(fullLength, center + halfWindow + 1);
        double[] truncated = new double[2 * maxLag + 1];
        int destStart = start <= center - halfWindow ? 0 : center - halfWindow - start;
        int srcLen = end - start;
        for (int i = 0; i < srcLen && (destStart + i) < truncated.length; i++) {
            truncated[destStart + i] = fullCorr.get(start + i);
        }
        return IVector.of(truncated);
    }

    public Tuple3<IVector<Double>, IVector<Double>, IVector<Double>> spectrum(
            IVector<Double> signal, double samplingRate) {
        try {
            ISignalAnalyzer<Double> analyzer = SignalProcessorFactory.getInstance().createAnalyzer("spectrum");
            ISignalAnalyzer.AnalysisParameters params = new ISignalAnalyzer.AnalysisParameters()
                .samplingRate(samplingRate);
            ISignalAnalyzer.AnalysisResult<Tuple2<IVector<Double>, IVector<Double>>> result =
                analyzer.analyze(signal, ISignalAnalyzer.AnalysisType.SPECTRUM, params);
            Tuple2<IVector<Double>, IVector<Double>> spectrumResult = result.getResult();
            IVector<Double> frequencies = spectrumResult._1;
            IVector<Double> magnitudes = spectrumResult._2;
            IVector<Double> phases = Linalg.zeros(magnitudes.length());
            return new Tuple3<>(frequencies, magnitudes, phases);
        } catch (Exception e) {
            throw new RuntimeException("Failed to calculate spectrum", e);
        }
    }

    public IMatrix<Double> shortTimeFourierTransform(
            IVector<Double> signal, int windowSize, int hopSize, double samplingRate) {
        return shortTimeFourierTransform(signal, windowSize, hopSize, SignalUtilities.WindowType.HANNING);
    }

    public IMatrix<Double> shortTimeFourierTransform(
            IVector<Double> signal, int windowSize, int hopSize, SignalUtilities.WindowType windowType) {
        int signalLength = signal.length();
        int numFrames = (int) Math.ceil((double)(signalLength - windowSize) / hopSize) + 1;
        int numFreqBins = windowSize / 2 + 1;
        double[][] realParts = new double[numFreqBins][numFrames];
        double[][] imagParts = new double[numFreqBins][numFrames];

        IVector<Double> window = SignalUtilities.window(windowSize, windowType);

        for (int frame = 0; frame < numFrames; frame++) {
            int startIdx = frame * hopSize;
            double[] frameData = new double[windowSize];
            for (int i = 0; i < windowSize; i++) {
                int idx = startIdx + i;
                frameData[i] = (idx < signalLength) ? signal.get(idx) * window.get(i) : 0;
            }
            int fftSize = 1;
            while (fftSize < windowSize) fftSize <<= 1;
            Complex[] complexFrame = new Complex[fftSize];
            for (int i = 0; i < windowSize; i++) {
                complexFrame[i] = new Complex(frameData[i], 0);
            }
            for (int i = windowSize; i < fftSize; i++) {
                complexFrame[i] = new Complex(0, 0);
            }
            Complex[] fftResult = RereFFT.fft(complexFrame);
            for (int i = 0; i < numFreqBins; i++) {
                realParts[i][frame] = fftResult[i].real;
                imagParts[i][frame] = fftResult[i].imag;
            }
        }

        double[][] result = new double[numFreqBins * 2][numFrames];
        for (int f = 0; f < numFreqBins; f++) {
            for (int fr = 0; fr < numFrames; fr++) {
                result[2 * f][fr] = realParts[f][fr];
                result[2 * f + 1][fr] = imagParts[f][fr];
            }
        }
        return IMatrix.of(result);
    }

    public IMatrix<Double> spectrogram(
            IVector<Double> signal, int windowSize, int hopSize, SignalUtilities.WindowType window) {
        IMatrix<Double> stft = shortTimeFourierTransform(signal, windowSize, hopSize, window);
        int numFreqBins = stft.rows() / 2;
        int numFrames = stft.cols();
        double[][] mag = new double[numFreqBins][numFrames];
        for (int f = 0; f < numFreqBins; f++) {
            for (int t = 0; t < numFrames; t++) {
                double re = stft.get(2 * f, t);
                double im = stft.get(2 * f + 1, t);
                mag[f][t] = Math.sqrt(re * re + im * im);
            }
        }
        return IMatrix.of(mag);
    }

    public IMatrix<Double> powerSpectrogram(
            IVector<Double> signal, int windowSize, int hopSize, SignalUtilities.WindowType window) {
        IMatrix<Double> stft = shortTimeFourierTransform(signal, windowSize, hopSize, window);
        int numFreqBins = stft.rows() / 2;
        int numFrames = stft.cols();
        double[][] powSpec = new double[numFreqBins][numFrames];
        for (int f = 0; f < numFreqBins; f++) {
            for (int t = 0; t < numFrames; t++) {
                double re = stft.get(2 * f, t);
                double im = stft.get(2 * f + 1, t);
                powSpec[f][t] = re * re + im * im;
            }
        }
        return IMatrix.of(powSpec);
    }

    public IMatrix<Double> logSpectrogram(
            IVector<Double> signal, int windowSize, int hopSize, SignalUtilities.WindowType window, double minDb) {
        IMatrix<Double> stft = shortTimeFourierTransform(signal, windowSize, hopSize, window);
        int numFreqBins = stft.rows() / 2;
        int numFrames = stft.cols();
        double[][] db = new double[numFreqBins][numFrames];
        double minLinear = Math.pow(10, minDb / 20.0);
        for (int f = 0; f < numFreqBins; f++) {
            for (int t = 0; t < numFrames; t++) {
                double re = stft.get(2 * f, t);
                double im = stft.get(2 * f + 1, t);
                double mag = Math.sqrt(re * re + im * im);
                if (mag < minLinear) mag = minLinear;
                db[f][t] = 20 * Math.log10(mag);
            }
        }
        return IMatrix.of(db);
    }

    public IMatrix<Double> logSpectrogram(
            IVector<Double> signal, int windowSize, int hopSize, SignalUtilities.WindowType window) {
        return logSpectrogram(signal, windowSize, hopSize, window, -120.0);
    }

    public IMatrix<Double> wignerVilleDistribution(IVector<Double> signal) {
        return wignerVilleDistribution(signal, 128);
    }

    public IMatrix<Double> wignerVilleDistribution(IVector<Double> signal, int maxFreqBins) {
        int N = signal.length();
        Complex[] z = RereHilbert.analyticSignal(signal);
        int M = Math.min(maxFreqBins, N);
        int fftSize = 1;
        while (fftSize < 2 * M) fftSize <<= 1;

        double[][] realParts = new double[fftSize / 2 + 1][N];
        double[][] imagParts = new double[fftSize / 2 + 1][N];

        Complex[] kernel = new Complex[fftSize];
        for (int n = 0; n < N; n++) {
            for (int i = 0; i < fftSize; i++) kernel[i] = new Complex(0, 0);
            for (int m = -M; m < M; m++) {
                int idx1 = n + m;
                int idx2 = n - m;
                if (idx1 >= 0 && idx1 < N && idx2 >= 0 && idx2 < N) {
                    Complex prod = z[idx1].multiply(z[idx2].conjugate());
                    kernel[m + M] = prod;
                }
            }
            Complex[] fftResult = RereFFT.fft(kernel);
            int numBins = fftSize / 2 + 1;
            for (int f = 0; f < numBins; f++) {
                realParts[f][n] = fftResult[f].real;
                imagParts[f][n] = fftResult[f].imag;
            }
        }

        int numBins = fftSize / 2 + 1;
        double[][] result = new double[numBins * 2][N];
        for (int f = 0; f < numBins; f++) {
            for (int t = 0; t < N; t++) {
                result[2 * f][t] = realParts[f][t];
                result[2 * f + 1][t] = imagParts[f][t];
            }
        }
        return IMatrix.of(result);
    }

    public double signalToNoiseRatio(IVector<Double> signal, IVector<Double> noise) {
        try {
            ISignalAnalyzer<Double> analyzer = SignalProcessorFactory.getInstance().createAnalyzer("spectrum");
            ISignalAnalyzer.AnalysisParameters params = new ISignalAnalyzer.AnalysisParameters();
            ISignalAnalyzer.AnalysisResult<Double> result =
                analyzer.analyze(signal, ISignalAnalyzer.AnalysisType.SNR, params);
            return result.getResult();
        } catch (Exception e) {
            throw new RuntimeException("Failed to calculate SNR", e);
        }
    }

    public double peakSignalToNoiseRatio(IVector<Double> original, IVector<Double> reconstructed) {
        int n = original.length();
        double mse = 0;
        double maxVal = 0;
        for (int i = 0; i < n; i++) {
            double diff = original.get(i) - reconstructed.get(i);
            mse += diff * diff;
            double absVal = Math.abs(original.get(i));
            if (absVal > maxVal) maxVal = absVal;
        }
        mse /= n;
        if (mse < 1e-30) {
            return Double.POSITIVE_INFINITY;
        }
        if (maxVal < 1e-30) {
            return 0;
        }
        return 20 * Math.log10(maxVal / Math.sqrt(mse));
    }

    // ---- Cohen-class distributions ----

    public IMatrix<Double> cohenClass(IVector<Double> signal, CohenClassDistribution.KernelFunction kernel, int maxLag) {
        return CohenClassDistribution.cohenClass(signal, kernel, maxLag);
    }

    public IMatrix<Double> choiWilliams(IVector<Double> signal, double sigma) {
        return CohenClassDistribution.choiWilliams(signal, sigma);
    }

    public IMatrix<Double> choiWilliams(IVector<Double> signal) {
        return CohenClassDistribution.choiWilliams(signal);
    }

    public IMatrix<Double> bornJordan(IVector<Double> signal) {
        return CohenClassDistribution.bornJordan(signal);
    }

    public IMatrix<Double> margenauHill(IVector<Double> signal) {
        return CohenClassDistribution.margenauHill(signal);
    }

    public IMatrix<Double> pseudoWignerVille(IVector<Double> signal, int smoothingWindow) {
        return CohenClassDistribution.pseudoWignerVille(signal, smoothingWindow);
    }
}
