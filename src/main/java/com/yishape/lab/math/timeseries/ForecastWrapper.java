package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.linalg.IVector;

/**
 * 时间序列预测门面 / Time series forecasting facade.
 *
 * <p>单变量示例:
 * <pre>{@code
 *   IVector<Double> y = ...;
 *   ForecastResult f = TSA.forecast.arima(y, 1, 0, 1, 10);
 *   ForecastResult f2 = TSA.forecast.expSmooth(y, 0.3, 10);
 * }</pre>
 *
 * @author lteb2
 * @version 2.0
 * @since 1.0
 */
public class ForecastWrapper {

    // --- movingAverage ---

    public ForecastResult movingAverage(IVector<Double> data, int windowSize, int steps) {
        return TimeSeriesForecasting.movingAverage(data, windowSize, steps, TimeSeriesForecasting.DEFAULT_CONFIDENCE);
    }

    public ForecastResult movingAverage(IVector<Double> data, int windowSize, int steps, double confidence) {
        return TimeSeriesForecasting.movingAverage(data, windowSize, steps, confidence);
    }

    // --- expSmooth ---

    public ForecastResult expSmooth(IVector<Double> data, double alpha, int steps) {
        return TimeSeriesForecasting.exponentialSmoothing(data, alpha, steps, TimeSeriesForecasting.DEFAULT_CONFIDENCE);
    }

    public ForecastResult expSmooth(IVector<Double> data, double alpha, int steps, double confidence) {
        return TimeSeriesForecasting.exponentialSmoothing(data, alpha, steps, confidence);
    }

    // --- linearTrend ---

    public ForecastResult linearTrend(IVector<Double> data, int steps) {
        return TimeSeriesForecasting.linearRegression(data, steps, TimeSeriesForecasting.DEFAULT_CONFIDENCE);
    }

    public ForecastResult linearTrend(IVector<Double> data, int steps, double confidence) {
        return TimeSeriesForecasting.linearRegression(data, steps, confidence);
    }

    // --- arima ---

    public ForecastResult arima(IVector<Double> data, int p, int d, int q, int steps) {
        return TimeSeriesForecasting.arimaForecast(data, p, d, q, steps, TimeSeriesForecasting.DEFAULT_CONFIDENCE);
    }

    public ForecastResult arima(IVector<Double> data, int p, int d, int q, int steps, double confidence) {
        return TimeSeriesForecasting.arimaForecast(data, p, d, q, steps, confidence);
    }

    // --- seasonal ---

    public ForecastResult seasonal(IVector<Double> data, int period, int steps) {
        return TimeSeriesForecasting.seasonalForecast(data, period, steps, TimeSeriesForecasting.DEFAULT_CONFIDENCE);
    }

    public ForecastResult seasonal(IVector<Double> data, int period, int steps, double confidence) {
        return TimeSeriesForecasting.seasonalForecast(data, period, steps, confidence);
    }

    // --- holtWinters ---

    public ForecastResult holtWinters(IVector<Double> data,
            double alpha, double beta, double gamma, int period, int steps) {
        return TimeSeriesForecasting.holtWintersForecast(data, alpha, beta, gamma, period, steps,
                TimeSeriesForecasting.DEFAULT_CONFIDENCE);
    }

    public ForecastResult holtWinters(IVector<Double> data,
            double alpha, double beta, double gamma, int period, int steps, double confidence) {
        return TimeSeriesForecasting.holtWintersForecast(data, alpha, beta, gamma, period, steps, confidence);
    }

    // --- garch ---

    public ForecastResult garch(IVector<Double> data, int p, int q, int steps) {
        return TimeSeriesForecasting.garchForecast(data, p, q, steps, TimeSeriesForecasting.DEFAULT_CONFIDENCE);
    }

    public ForecastResult garch(IVector<Double> data, int p, int q, int steps, double confidence) {
        return TimeSeriesForecasting.garchForecast(data, p, q, steps, confidence);
    }

    // --- stateSpace ---

    public ForecastResult stateSpace(IVector<Double> data,
            double sigmaEta, double sigmaZeta, double sigmaEpsilon, int steps) {
        return TimeSeriesForecasting.stateSpaceForecast(data, sigmaEta, sigmaZeta, sigmaEpsilon, steps,
                TimeSeriesForecasting.DEFAULT_CONFIDENCE);
    }

    public ForecastResult stateSpace(IVector<Double> data,
            double sigmaEta, double sigmaZeta, double sigmaEpsilon, int steps, double confidence) {
        return TimeSeriesForecasting.stateSpaceForecast(data, sigmaEta, sigmaZeta, sigmaEpsilon, steps, confidence);
    }

    // --- auto ---

    public ForecastResult auto(IVector<Double> data, int steps) {
        return TimeSeriesForecasting.autoForecast(data, steps, TimeSeriesForecasting.DEFAULT_CONFIDENCE);
    }

    public ForecastResult auto(IVector<Double> data, int steps, double confidence) {
        return TimeSeriesForecasting.autoForecast(data, steps, confidence);
    }
}
