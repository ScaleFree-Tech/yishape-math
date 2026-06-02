package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.timeseries.TimeSeriesDecomposition.DecompositionModel;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link TimeSeriesDecomposition}.
 */
class TimeSeriesDecompositionTest {

    private static IVector<Double> seasonalData(int period, int cycles) {
        int n = period * cycles;
        double[] data = new double[n];
        for (int i = 0; i < n; i++) {
            data[i] = 10 + 5 * Math.sin(2 * Math.PI * i / period) + i * 0.1;
        }
        return Linalg.vector(data);
    }

    // ==================== Classical Decomposition ====================

    @Test
    void classicalDecomposition_additive() {
        IVector<Double> data = seasonalData(12, 5);
        DecompositionResult result = TimeSeriesDecomposition.classicalDecomposition(
            data, 12, DecompositionModel.ADDITIVE);
        assertNotNull(result);
        assertNotNull(result.trend);
        assertNotNull(result.seasonal);
        assertNotNull(result.residual);
        assertEquals(data.size(), result.trend.size());
        assertEquals(data.size(), result.seasonal.size());
        assertEquals(data.size(), result.residual.size());
    }

    @Test
    void classicalDecomposition_multiplicative() {
        IVector<Double> data = seasonalData(12, 5);
        DecompositionResult result = TimeSeriesDecomposition.classicalDecomposition(
            data, 12, DecompositionModel.MULTIPLICATIVE);
        assertNotNull(result);
        assertEquals(DecompositionModel.MULTIPLICATIVE, result.model);
    }

    @Test
    void classicalDecomposition_periodTooLarge_throws() {
        IVector<Double> data = Linalg.vector(new double[]{1, 2, 3});
        assertThrows(IllegalArgumentException.class,
            () -> TimeSeriesDecomposition.classicalDecomposition(
                data, 5, DecompositionModel.ADDITIVE));
    }

    // ==================== STL Decomposition ====================

    @Test
    void stlDecomposition() {
        IVector<Double> data = seasonalData(12, 5);
        DecompositionResult result = TimeSeriesDecomposition.stlDecomposition(
            data, 12, 7, 15);
        assertNotNull(result);
        assertNotNull(result.trend);
        assertNotNull(result.seasonal);
        assertNotNull(result.residual);
    }

    @Test
    void stlDecomposition_periodTooLarge_throws() {
        IVector<Double> data = Linalg.vector(new double[]{1, 2, 3});
        assertThrows(IllegalArgumentException.class,
            () -> TimeSeriesDecomposition.stlDecomposition(data, 5, 7, 15));
    }

    // ==================== X13 Decomposition ====================

    @Test
    void x13Decomposition() {
        IVector<Double> data = seasonalData(12, 5);
        DecompositionResult result = TimeSeriesDecomposition.x13Decomposition(data, 12);
        assertNotNull(result);
        assertNotNull(result.trend);
    }

    @Test
    void x13Decomposition_periodTooLarge_throws() {
        IVector<Double> data = Linalg.vector(new double[]{1, 2, 3});
        assertThrows(IllegalArgumentException.class,
            () -> TimeSeriesDecomposition.x13Decomposition(data, 5));
    }

    // ==================== Wavelet Decomposition ====================

    @Test
    void waveletDecomposition() {
        IVector<Double> data = seasonalData(8, 4);
        DecompositionResult result = TimeSeriesDecomposition.waveletDecomposition(
            data, "haar", 3);
        assertNotNull(result);
        assertNotNull(result.trend);
    }

    // ==================== Result Properties ====================

    @Test
    void result_hasPeriod() {
        IVector<Double> data = seasonalData(12, 5);
        DecompositionResult result = TimeSeriesDecomposition.classicalDecomposition(
            data, 12, DecompositionModel.ADDITIVE);
        assertEquals(12, result.period);
    }

    @Test
    void result_hasOriginal() {
        IVector<Double> data = seasonalData(12, 5);
        DecompositionResult result = TimeSeriesDecomposition.classicalDecomposition(
            data, 12, DecompositionModel.ADDITIVE);
        assertEquals(data.size(), result.original.size());
    }
}
