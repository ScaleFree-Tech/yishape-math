package com.yishape.lab.math.signal.analysis;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.signal.core.Complex;
import com.yishape.lab.math.signal.core.RereFFT;
import com.yishape.lab.math.signal.core.RereHilbert;

public class CohenClassDistribution {

    private CohenClassDistribution() {}

    @FunctionalInterface
    public interface KernelFunction {
        double evaluate(double doppler, double lag);
    }

    /**
     * Generic Cohen-class time-frequency distribution.
     *
     * @param signal real-valued input signal
     * @param kernel kernel function Φ(ξ, τ) where ξ = Doppler, τ = lag
     * @param maxLag maximum lag (frequency resolution)
     * @return time-frequency matrix [2*fftSize/2+1 x N]
     */
    public static IMatrix<Double> cohenClass(IVector<Double> signal, KernelFunction kernel, int maxLag) {
        int N = signal.length();
        Complex[] z = RereHilbert.analyticSignal(signal);
        int M = Math.min(maxLag, N / 2);

        int fftSize = 1;
        while (fftSize < 2 * M + 1) fftSize <<= 1;
        int numFreqBins = fftSize / 2 + 1;

        double[][] realParts = new double[numFreqBins][N];
        double[][] imagParts = new double[numFreqBins][N];

        for (int m = -M; m <= M; m++) {
            Complex[] g = new Complex[N];
            for (int n = 0; n < N; n++) {
                int n1 = n + m;
                int n2 = n - m;
                if (n1 >= 0 && n1 < N && n2 >= 0 && n2 < N) {
                    g[n] = z[n1].multiply(z[n2].conjugate());
                } else {
                    g[n] = new Complex(0, 0);
                }
            }

            Complex[] G = RereFFT.fft(zeroPadToPowerOfTwo(g));
            int dopplerLen = G.length;
            for (int k = 0; k < dopplerLen; k++) {
                double xi = (k <= dopplerLen / 2) ? 2.0 * Math.PI * k / dopplerLen
                        : 2.0 * Math.PI * (k - dopplerLen) / dopplerLen;
                double phi = kernel.evaluate(xi, m);
                G[k] = new Complex(G[k].real * phi, G[k].imag * phi);
            }
            Complex[] gFiltered = RereFFT.ifft(G);

            for (int n = 0; n < N; n++) {
                int bin = m + M;
                int wrappedBin = ((bin % numFreqBins) + numFreqBins) % numFreqBins;
                double re = gFiltered[n].real / gFiltered.length;
                double im = gFiltered[n].imag / gFiltered.length;
                realParts[wrappedBin][n] += re;
                imagParts[wrappedBin][n] += im;
            }
        }

        // FFT along lag dimension for each time
        double[][] resultReal = new double[numFreqBins][N];
        double[][] resultImag = new double[numFreqBins][N];

        for (int n = 0; n < N; n++) {
            Complex[] lagSlice = new Complex[fftSize];
            for (int i = 0; i < fftSize; i++) {
                if (i < 2 * M + 1) {
                    lagSlice[i] = new Complex(realParts[i][n], imagParts[i][n]);
                } else {
                    lagSlice[i] = new Complex(0, 0);
                }
            }
            Complex[] spec = RereFFT.fft(lagSlice);
            for (int f = 0; f < numFreqBins; f++) {
                resultReal[f][n] = spec[f].real;
                resultImag[f][n] = spec[f].imag;
            }
        }

        double[][] result = new double[numFreqBins * 2][N];
        for (int f = 0; f < numFreqBins; f++) {
            for (int n = 0; n < N; n++) {
                result[2 * f][n] = resultReal[f][n];
                result[2 * f + 1][n] = resultImag[f][n];
            }
        }
        return IMatrix.of(result);
    }

    /** Choi-Williams distribution: exponential kernel suppresses cross-terms. */
    public static IMatrix<Double> choiWilliams(IVector<Double> signal, double sigma) {
        return cohenClass(signal, (xi, tau) -> Math.exp(-(xi * xi * tau * tau) / sigma), 128);
    }

    public static IMatrix<Double> choiWilliams(IVector<Double> signal) {
        return choiWilliams(signal, 1.0);
    }

    /** Born-Jordan distribution: sinc kernel. */
    public static IMatrix<Double> bornJordan(IVector<Double> signal) {
        return cohenClass(signal, (xi, tau) -> {
            double arg = xi * tau / 2.0;
            if (Math.abs(arg) < 1e-12) return 1.0;
            return Math.sin(arg) / arg;
        }, 128);
    }

    /** Margenau-Hill distribution. */
    public static IMatrix<Double> margenauHill(IVector<Double> signal) {
        return cohenClass(signal, (xi, tau) -> Math.cos(xi * tau / 2.0), 128);
    }

    /**
     * Pseudo Wigner-Ville distribution: frequency smoothing via lag windowing.
     */
    public static IMatrix<Double> pseudoWignerVille(IVector<Double> signal, int smoothingWindow) {
        int M = Math.min(smoothingWindow, signal.length() / 2);
        double[] hannWindow = new double[2 * M + 1];
        for (int i = 0; i < 2 * M + 1; i++) {
            hannWindow[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (2 * M)));
        }
        return cohenClass(signal, (xi, tau) -> {
            int idx = (int) tau + M;
            if (idx < 0 || idx >= hannWindow.length) return 0;
            return hannWindow[idx];
        }, M);
    }

    private static Complex[] zeroPadToPowerOfTwo(Complex[] signal) {
        int n = 1;
        while (n < signal.length) n <<= 1;
        if (n == signal.length) return signal;
        Complex[] padded = new Complex[n];
        System.arraycopy(signal, 0, padded, 0, signal.length);
        for (int i = signal.length; i < n; i++) {
            padded[i] = new Complex(0, 0);
        }
        return padded;
    }

    /** Compute the magnitude (envelope) of a complex time-frequency matrix. */
    public static IMatrix<Double> magnitude(IMatrix<Double> tfMatrix) {
        int rows = tfMatrix.rows() / 2;
        int cols = tfMatrix.cols();
        double[][] mag = new double[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double re = tfMatrix.get(2 * r, c);
                double im = tfMatrix.get(2 * r + 1, c);
                mag[r][c] = Math.sqrt(re * re + im * im);
            }
        }
        return IMatrix.of(mag);
    }
}
