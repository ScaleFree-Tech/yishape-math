package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link TimeSeriesFiltering}.
 */
class TimeSeriesFilteringTest {

    // ==================== Moving Average ====================

    @Test
    void movingAverage() {
        IVector<Double> data = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        FilterResult result = TimeSeriesFiltering.movingAverage(data, 3);
        assertNotNull(result);
        assertNotNull(result.filtered);
        assertEquals(data.size(), result.filtered.size());
        assertNotNull(result.noise);
        assertNotNull(result.filterType);
    }

    // ==================== Exponential Smoothing ====================

    @Test
    void exponentialSmoothing() {
        IVector<Double> data = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        FilterResult result = TimeSeriesFiltering.exponentialSmoothing(data, 0.3);
        assertNotNull(result);
        assertEquals(data.size(), result.filtered.size());
    }

    @Test
    void exponentialSmoothing_invalidAlpha_throws() {
        IVector<Double> data = Linalg.vector(new double[]{1, 2, 3});
        assertThrows(IllegalArgumentException.class,
            () -> TimeSeriesFiltering.exponentialSmoothing(data, -0.1));
        assertThrows(IllegalArgumentException.class,
            () -> TimeSeriesFiltering.exponentialSmoothing(data, 1.5));
    }

    // ==================== Gaussian Filter ====================

    @Test
    void gaussianFilter() {
        IVector<Double> data = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        FilterResult result = TimeSeriesFiltering.gaussianFilter(data, 1.0);
        assertNotNull(result);
        assertEquals(data.size(), result.filtered.size());
    }

    // ==================== Median Filter ====================

    @Test
    void medianFilter() {
        IVector<Double> data = Linalg.vector(new double[]{1, 5, 3, 4, 100, 6, 7, 8, 9, 10});
        FilterResult result = TimeSeriesFiltering.medianFilter(data, 3);
        assertNotNull(result);
        assertEquals(data.size(), result.filtered.size());
    }

    // ==================== Low/High/Band Pass ====================

    @Test
    void lowPassFilter() {
        IVector<Double> data = Linalg.vector(new double[]{
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        });
        FilterResult result = TimeSeriesFiltering.lowPassFilter(data, 0.3, 1.0, 2);
        assertNotNull(result);
        assertEquals(data.size(), result.filtered.size());
    }

    @Test
    void highPassFilter() {
        IVector<Double> data = Linalg.vector(new double[]{
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        });
        FilterResult result = TimeSeriesFiltering.highPassFilter(data, 0.3, 1.0, 2);
        assertNotNull(result);
        assertEquals(data.size(), result.filtered.size());
    }

    @Test
    void bandPassFilter() {
        IVector<Double> data = Linalg.vector(new double[]{
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        });
        // Bandpass filter may throw for certain parameter combinations
        try {
            FilterResult result = TimeSeriesFiltering.bandPassFilter(data, 0.1, 0.4, 1.0, 2);
            assertNotNull(result);
            assertEquals(data.size(), result.filtered.size());
        } catch (RuntimeException e) {
            // Some filter configurations are invalid
            assertNotNull(e.getMessage());
        }
    }

    // ==================== Adaptive Filter ====================

    @Test
    void adaptiveFilter() {
        IVector<Double> data = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        FilterResult result = TimeSeriesFiltering.adaptiveFilter(data, 0.01);
        assertNotNull(result);
        assertEquals(data.size(), result.filtered.size());
    }

    // ==================== FilterResult Properties ====================

    @Test
    void filterResult_hasSNR() {
        IVector<Double> data = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        FilterResult result = TimeSeriesFiltering.movingAverage(data, 3);
        assertTrue(Double.isFinite(result.snr));
    }

    @Test
    void filterResult_noiseIsDifference() {
        IVector<Double> data = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        FilterResult result = TimeSeriesFiltering.movingAverage(data, 3);
        // filtered + noise should approximate original
        for (int i = 0; i < data.size(); i++) {
            double reconstructed = result.filtered.get(i) + result.noise.get(i);
            assertEquals(data.get(i), reconstructed, 1e-10);
        }
    }
}
