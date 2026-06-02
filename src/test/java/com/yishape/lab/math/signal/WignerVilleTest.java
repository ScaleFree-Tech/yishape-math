package com.yishape.lab.math.signal;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.signal.analysis.CohenClassDistribution;
import com.yishape.lab.math.signal.core.SignalUtilities;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class WignerVilleTest {

    @Test
    public void wignerVille_singleFrequency_returnsValidMatrix() {
        int N = 128;
        IVector<Double> signal = Signals.gen.sineWave(N, 10, 100, 1.0, 0);
        IMatrix<Double> wvd = Signals.analyze.wignerVilleDistribution(signal);

        assertNotNull(wvd);
        assertTrue(wvd.rows() > 0);
        assertEquals(N, wvd.cols());
    }

    @Test
    public void wignerVille_withCustomBins() {
        int N = 64;
        IVector<Double> signal = Signals.gen.sineWave(N, 5, 100, 1.0, 0);
        IMatrix<Double> wvd = Signals.analyze.wignerVilleDistribution(signal, 64);

        assertNotNull(wvd);
        assertEquals(N, wvd.cols());
    }

    @Test
    public void spectrogram_returnsValidMagnitudes() {
        int N = 256;
        IVector<Double> signal = Signals.gen.chirpSignal(N, 5, 50, 100, 1.0);
        IMatrix<Double> spec = Signals.analyze.spectrogram(signal, 64, 32, SignalUtilities.WindowType.HANNING);

        assertNotNull(spec);
        assertTrue(spec.rows() > 0);
        assertTrue(spec.cols() > 0);
        // Magnitudes should be non-negative
        for (int i = 0; i < spec.rows(); i++) {
            for (int j = 0; j < spec.cols(); j++) {
                assertTrue(spec.get(i, j) >= 0, "Spectrogram magnitudes should be non-negative");
            }
        }
    }

    @Test
    public void powerSpectrogram_returnsValidPowers() {
        int N = 128;
        IVector<Double> signal = Signals.gen.sineWave(N, 10, 100, 1.0, 0);
        IMatrix<Double> pspec = Signals.analyze.powerSpectrogram(signal, 64, 32, SignalUtilities.WindowType.HAMMING);

        assertNotNull(pspec);
        for (int i = 0; i < pspec.rows(); i++) {
            for (int j = 0; j < pspec.cols(); j++) {
                assertTrue(pspec.get(i, j) >= 0, "Power spectrogram should be non-negative");
            }
        }
    }

    @Test
    public void logSpectrogram_returnsFiniteValues() {
        int N = 128;
        IVector<Double> signal = Signals.gen.sineWave(N, 10, 100, 1.0, 0);
        IMatrix<Double> logSpec = Signals.analyze.logSpectrogram(signal, 64, 32, SignalUtilities.WindowType.BLACKMAN, -120.0);

        assertNotNull(logSpec);
        for (int i = 0; i < logSpec.rows(); i++) {
            for (int j = 0; j < logSpec.cols(); j++) {
                double val = logSpec.get(i, j);
                assertFalse(Double.isNaN(val), "Log spectrogram should not have NaN");
                assertTrue(Double.isFinite(val), "Log spectrogram should be finite");
                assertTrue(val >= -120.0, "Log spectrogram should be >= minDb");
            }
        }
    }

    @Test
    public void stft_withWindowType_producesValidOutput() {
        int N = 64;
        IVector<Double> signal = Signals.gen.sineWave(N, 5, 100, 1.0, 0);
        IMatrix<Double> stftHann = Signals.analyze.shortTimeFourierTransform(signal, 32, 16, SignalUtilities.WindowType.HANNING);
        IMatrix<Double> stftRect = Signals.analyze.shortTimeFourierTransform(signal, 32, 16, SignalUtilities.WindowType.RECTANGULAR);

        assertEquals(stftHann.rows(), stftRect.rows());
        assertEquals(stftHann.cols(), stftRect.cols());
    }

    @Test
    public void choiWilliams_producesValidOutput() {
        int N = 64;
        IVector<Double> signal = Signals.gen.sineWave(N, 10, 100, 1.0, 0);
        IMatrix<Double> cw = Signals.analyze.choiWilliams(signal, 1.0);

        assertNotNull(cw);
        assertTrue(cw.rows() > 0);
        assertEquals(N, cw.cols());

        IMatrix<Double> mag = CohenClassDistribution.magnitude(cw);
        for (int i = 0; i < mag.rows(); i++) {
            for (int j = 0; j < mag.cols(); j++) {
                assertTrue(mag.get(i, j) >= 0, "Magnitudes should be non-negative");
            }
        }
    }

    @Test
    public void bornJordan_producesValidOutput() {
        int N = 64;
        IVector<Double> signal = Signals.gen.sineWave(N, 10, 100, 1.0, 0);
        IMatrix<Double> bj = Signals.analyze.bornJordan(signal);

        assertNotNull(bj);
        assertEquals(N, bj.cols());
    }

    @Test
    public void margenauHill_producesValidOutput() {
        int N = 64;
        IVector<Double> signal = Signals.gen.sineWave(N, 10, 100, 1.0, 0);
        IMatrix<Double> mh = Signals.analyze.margenauHill(signal);

        assertNotNull(mh);
        assertEquals(N, mh.cols());
    }

    @Test
    public void pseudoWignerVille_producesValidOutput() {
        int N = 64;
        IVector<Double> signal = Signals.gen.sineWave(N, 10, 100, 1.0, 0);
        IMatrix<Double> pwvd = Signals.analyze.pseudoWignerVille(signal, 32);

        assertNotNull(pwvd);
        assertEquals(N, pwvd.cols());
    }
}
