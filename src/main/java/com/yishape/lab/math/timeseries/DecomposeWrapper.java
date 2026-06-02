package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.linalg.IVector;

/**
 * 时间序列分解门面 / Time series decomposition facade.
 *
 * @author lteb2
 * @version 2.0
 * @since 1.0
 */
public class DecomposeWrapper {

    public DecompositionResult classical(IVector<Double> data, int period,
            TimeSeriesDecomposition.DecompositionModel type) {
        return TimeSeriesDecomposition.classicalDecomposition(data, period, type);
    }

    public DecompositionResult x13(IVector<Double> data, int period) {
        return TimeSeriesDecomposition.x13Decomposition(data, period);
    }

    public DecompositionResult stl(IVector<Double> data, int period,
            int seasonalWindow, int trendWindow) {
        return TimeSeriesDecomposition.stlDecomposition(data, period, seasonalWindow, trendWindow);
    }

    public DecompositionResult wavelet(IVector<Double> data, String waveletType, int levels) {
        return TimeSeriesDecomposition.waveletDecomposition(data, waveletType, levels);
    }
}
