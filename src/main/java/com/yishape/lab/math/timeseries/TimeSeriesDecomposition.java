package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;

/**
 * 时间序列分解类 / Time Series Decomposition Class
 *
 * @author lteb2
 * @version 2.0
 * @since 1.0
 */
public class TimeSeriesDecomposition {

    public enum DecompositionModel {
        ADDITIVE,
        MULTIPLICATIVE
    }

    public static DecompositionResult classicalDecomposition(IVector<Double> data,
            int period, DecompositionModel model) {
        int length = data.length();
        if (period >= length) {
            throw new IllegalArgumentException("季节周期不能大于等于数据长度");
        }
        IVector<Double> trend = calculateTrend(data, period);
        IVector<Double> detrended = detrend(data, trend, model);
        IVector<Double> seasonal = calculateSeasonal(detrended, period);
        IVector<Double> residual = calculateResidual(data, trend, seasonal, model);

        double trendStrength = calculateComponentStrength(trend, data);
        double seasonalStrength = calculateComponentStrength(seasonal, data);
        double residualStrength = calculateComponentStrength(residual, data);

        return new DecompositionResult(trend, seasonal, residual, data, model, period,
                trendStrength, seasonalStrength, residualStrength);
    }

    public static DecompositionResult x13Decomposition(IVector<Double> data, int period) {
        int length = data.length();
        if (period >= length) {
            throw new IllegalArgumentException("季节周期不能大于等于数据长度");
        }
        IVector<Double> prewhitened = prewhiten(data);
        IVector<Double> trend = estimateTrend(prewhitened, period);
        IVector<Double> seasonal = estimateSeasonal(prewhitened, trend, period);
        IVector<Double> residual = data.sub(trend).sub(seasonal);

        double trendStrength = calculateComponentStrength(trend, data);
        double seasonalStrength = calculateComponentStrength(seasonal, data);
        double residualStrength = calculateComponentStrength(residual, data);

        return new DecompositionResult(trend, seasonal, residual, data, DecompositionModel.ADDITIVE, period,
                trendStrength, seasonalStrength, residualStrength);
    }

    public static DecompositionResult stlDecomposition(IVector<Double> data,
            int period, int seasonalWindow, int trendWindow) {
        int length = data.length();
        if (period >= length) {
            throw new IllegalArgumentException("季节周期不能大于等于数据长度");
        }
        IVector<Double> trend = Linalg.zeros(length);
        IVector<Double> seasonal = Linalg.zeros(length);
        IVector<Double> residual = data.copy();

        for (int iter = 0; iter < 10; iter++) {
            IVector<Double> detrended = residual.sub(trend);
            seasonal = smoothSeasonal(detrended, period, seasonalWindow);
            IVector<Double> deseasonalized = residual.sub(seasonal);
            trend = smoothTrend(deseasonalized, trendWindow);
            residual = data.sub(trend).sub(seasonal);
        }

        double trendStrength = calculateComponentStrength(trend, data);
        double seasonalStrength = calculateComponentStrength(seasonal, data);
        double residualStrength = calculateComponentStrength(residual, data);

        return new DecompositionResult(trend, seasonal, residual, data, DecompositionModel.ADDITIVE, period,
                trendStrength, seasonalStrength, residualStrength);
    }

    public static DecompositionResult waveletDecomposition(IVector<Double> data,
            String wavelet, int levels) {
        IMatrix<Double> coeffs = waveletDecompose(data, wavelet, levels);
        IVector<Double> trend = reconstructTrend(coeffs, levels);
        IVector<Double> seasonal = reconstructSeasonal(coeffs, levels);
        IVector<Double> residual = data.sub(trend).sub(seasonal);

        double trendStrength = calculateComponentStrength(trend, data);
        double seasonalStrength = calculateComponentStrength(seasonal, data);
        double residualStrength = calculateComponentStrength(residual, data);

        return new DecompositionResult(trend, seasonal, residual, data, DecompositionModel.ADDITIVE, 0,
                trendStrength, seasonalStrength, residualStrength);
    }

    // ========== private helpers ==========

    private static IVector<Double> calculateTrend(IVector<Double> data, int period) {
        int length = data.length();
        IVector<Double> trend = Linalg.zeros(length);
        int windowSize = Math.max(period, 3);
        for (int i = 0; i < length; i++) {
            int start = Math.max(0, i - windowSize / 2);
            int end = Math.min(length, i + windowSize / 2 + 1);
            trend.set(i, data.slice(start, end).meanValue());
        }
        return trend;
    }

    private static IVector<Double> detrend(IVector<Double> data, IVector<Double> trend, DecompositionModel model) {
        return model == DecompositionModel.ADDITIVE ? data.sub(trend) : data.divide(trend);
    }

    private static IVector<Double> calculateSeasonal(IVector<Double> detrended, int period) {
        int length = detrended.length();
        IVector<Double> seasonal = Linalg.zeros(length);
        for (int i = 0; i < length; i++) {
            int seasonalIndex = i % period;
            double seasonalValue = 0.0;
            int count = 0;
            for (int j = seasonalIndex; j < length; j += period) {
                seasonalValue += detrended.get(j);
                count++;
            }
            if (count > 0) {
                seasonal.set(i, seasonalValue / count);
            }
        }
        double mean = seasonal.meanValue();
        return seasonal.subScalar(mean);
    }

    private static IVector<Double> calculateResidual(IVector<Double> data, IVector<Double> trend,
            IVector<Double> seasonal, DecompositionModel model) {
        return model == DecompositionModel.ADDITIVE
                ? data.sub(trend).sub(seasonal)
                : data.divide(trend).divide(seasonal);
    }

    private static double calculateComponentStrength(IVector<Double> component, IVector<Double> original) {
        double componentVar = component.varValue();
        double originalVar = original.varValue();
        return originalVar == 0 ? 0.0 : Math.max(0, 1 - componentVar / originalVar);
    }

    private static IVector<Double> prewhiten(IVector<Double> data) {
        double mean = data.meanValue();
        double std = data.stdValue();
        return std == 0 ? data : data.subScalar(mean).divideByScalar(std);
    }

    private static IVector<Double> estimateTrend(IVector<Double> data, int period) {
        return calculateTrend(data, period);
    }

    private static IVector<Double> estimateSeasonal(IVector<Double> data, IVector<Double> trend, int period) {
        return calculateSeasonal(data.sub(trend), period);
    }

    private static IVector<Double> smoothSeasonal(IVector<Double> data, int period, int window) {
        int length = data.length();
        IVector<Double> smoothed = Linalg.zeros(length);
        for (int i = 0; i < length; i++) {
            int seasonalIndex = i % period;
            double value = 0.0;
            int count = 0;
            for (int j = seasonalIndex; j < length; j += period) {
                int start = Math.max(0, j - window / 2);
                int end = Math.min(length, j + window / 2 + 1);
                value += data.slice(start, end).meanValue();
                count++;
            }
            if (count > 0) {
                smoothed.set(i, value / count);
            }
        }
        return smoothed;
    }

    private static IVector<Double> smoothTrend(IVector<Double> data, int window) {
        int length = data.length();
        IVector<Double> smoothed = Linalg.zeros(length);
        for (int i = 0; i < length; i++) {
            int start = Math.max(0, i - window / 2);
            int end = Math.min(length, i + window / 2 + 1);
            smoothed.set(i, data.slice(start, end).meanValue());
        }
        return smoothed;
    }

    private static IMatrix<Double> waveletDecompose(IVector<Double> data, String wavelet, int levels) {
        int length = data.length();
        IMatrix<Double> coeffs = Linalg.zeros(levels + 1, length);
        coeffs.setRow(0, data);
        for (int level = 1; level <= levels; level++) {
            int step = 1 << level;
            IVector<Double> detail = Linalg.zeros(length);
            for (int i = 0; i < length - step; i += step) {
                detail.set(i, (data.get(i + step) - data.get(i)) / 2.0);
            }
            coeffs.setRow(level, detail);
        }
        return coeffs;
    }

    private static IVector<Double> reconstructTrend(IMatrix<Double> coeffs, int levels) {
        return coeffs.getRow(0);
    }

    private static IVector<Double> reconstructSeasonal(IMatrix<Double> coeffs, int levels) {
        int length = coeffs.getColNum();
        IVector<Double> seasonal = Linalg.zeros(length);
        for (int level = 1; level <= levels; level++) {
            seasonal = seasonal.add(coeffs.getRow(level));
        }
        return seasonal;
    }
}
