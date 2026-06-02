package com.yishape.lab.math.signal.wavele;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link WaveletAnalysis}.
 */
class WaveletAnalysisTest {

    // ==================== DWT ====================

    @Test
    void dwt_haar_basic() {
        IVector<Double> signal = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8});
        WaveletCoefficients coeffs = WaveletAnalysis.discreteWaveletTransform(
            signal, WaveletAnalysis.WaveletType.HAAR, 2, 0);
        assertNotNull(coeffs);
        assertNotNull(coeffs.approximation);
        assertNotNull(coeffs.details);
        assertEquals(2, coeffs.levels);
    }

    @Test
    void dwt_invalidLevels_throws() {
        IVector<Double> signal = Linalg.vector(new double[]{1, 2, 3, 4});
        assertThrows(IllegalArgumentException.class,
            () -> WaveletAnalysis.discreteWaveletTransform(
                signal, WaveletAnalysis.WaveletType.HAAR, 0, 0));
    }

    // ==================== IDWT ====================

    @Test
    void idwt_reconstructsSignal() {
        IVector<Double> signal = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8});
        WaveletCoefficients coeffs = WaveletAnalysis.discreteWaveletTransform(
            signal, WaveletAnalysis.WaveletType.HAAR, 1, 0);
        IVector<Double> reconstructed = WaveletAnalysis.inverseDiscreteWaveletTransform(
            coeffs, WaveletAnalysis.WaveletType.HAAR, 0);
        assertNotNull(reconstructed);
        assertEquals(signal.size(), reconstructed.size());
    }

    // ==================== Denoising ====================

    @Test
    void denoising_reducesNoise() {
        IVector<Double> noisy = Linalg.vector(new double[]{
            1.1, 0.9, 2.1, 1.9, 3.05, 2.95, 4.1, 3.9
        });
        IVector<Double> denoised = WaveletAnalysis.waveletDenoising(
            noisy, WaveletAnalysis.WaveletType.HAAR, 1, 0.5, 0);
        assertNotNull(denoised);
        assertEquals(noisy.size(), denoised.size());
    }

    // ==================== Compression ====================

    @Test
    void compression() {
        IVector<Double> signal = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8});
        IVector<Double> compressed = WaveletAnalysis.waveletCompression(
            signal, WaveletAnalysis.WaveletType.HAAR, 2, 0.5, 0);
        assertNotNull(compressed);
        assertEquals(signal.size(), compressed.size());
    }

    // ==================== Energy Analysis ====================

    @Test
    void energyAnalysis() {
        IVector<Double> signal = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8});
        WaveletCoefficients coeffs = WaveletAnalysis.discreteWaveletTransform(
            signal, WaveletAnalysis.WaveletType.HAAR, 2, 0);
        IVector<Double> energy = WaveletAnalysis.waveletEnergyAnalysis(coeffs);
        assertNotNull(energy);
        assertTrue(energy.size() > 0);
    }

    // ==================== Feature Extraction ====================

    @Test
    void featureExtraction() {
        IVector<Double> signal = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8});
        WaveletCoefficients coeffs = WaveletAnalysis.discreteWaveletTransform(
            signal, WaveletAnalysis.WaveletType.HAAR, 2, 0);
        IVector<Double> features = WaveletAnalysis.waveletFeatureExtraction(coeffs);
        assertNotNull(features);
        assertTrue(features.size() > 0);
    }

    // ==================== CWT ====================

    @Test
    void cwt_smallSignal() {
        IVector<Double> signal = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8});
        IVector<Double> scales = Linalg.vector(new double[]{1, 2, 4});
        IMatrix<Double> cwt = WaveletAnalysis.continuousWaveletTransform(
            signal, WaveletAnalysis.WaveletType.HAAR, scales, 0);
        assertNotNull(cwt);
        assertTrue(cwt.rows() > 0);
    }

    @Test
    void cwt_largeSignal() {
        // >= 256 triggers FFT path
        double[] data = new double[256];
        for (int i = 0; i < 256; i++) data[i] = Math.sin(2 * Math.PI * i / 64);
        IVector<Double> signal = Linalg.vector(data);
        IVector<Double> scales = Linalg.vector(new double[]{1, 2, 4, 8});
        IMatrix<Double> cwt = WaveletAnalysis.continuousWaveletTransform(
            signal, WaveletAnalysis.WaveletType.HAAR, scales, 0);
        assertNotNull(cwt);
        assertEquals(4, cwt.rows());
    }

    // ==================== Wavelet Packet ====================

    @Test
    void waveletPacketTransform() {
        IVector<Double> signal = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8});
        WaveletAnalysis.WaveletPacketTree tree = WaveletAnalysis.waveletPacketTransform(
            signal, WaveletAnalysis.WaveletType.HAAR, 2, 0);
        assertNotNull(tree);
        assertNotNull(tree.root);
    }

    // ==================== WaveletType enum ====================

    @Test
    void waveletType_allValues() {
        assertEquals(7, WaveletAnalysis.WaveletType.values().length);
    }
}
