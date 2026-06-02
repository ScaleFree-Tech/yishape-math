package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.timeseries.model.StateSpaceModel;
import com.yishape.lab.math.timeseries.model.ITimeSeriesForecastResult;
import com.yishape.lab.math.timeseries.model.ExponentialSmoothingModels;
import com.yishape.lab.math.timeseries.model.GARCHModel;
import com.yishape.lab.math.timeseries.model.UnifiedARIMAModel;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.util.Tuple2;

/**
 * 时间序列预测类 / Time Series Forecasting Class
 *
 * @author lteb2
 * @version 2.0
 * @since 1.0
 */
public class TimeSeriesForecasting {

    static final double DEFAULT_CONFIDENCE = 0.95;

    public static ForecastResult movingAverage(IVector<Double> data,
            int windowSize, int forecastSteps, double confidenceLevel) {
        int length = data.length();
        IVector<Double> movingAvg = calculateMovingAverage(data, windowSize);

        IVector<Double> forecast = Linalg.zeros(forecastSteps);
        double lastValue = movingAvg.get(length - 1);
        for (int i = 0; i < forecastSteps; i++) {
            forecast.set(i, lastValue);
        }

        double std = data.slice(length - windowSize, length).stdValue();
        double zScore = getZScore(confidenceLevel);
        IVector<Double> lowerBound = forecast.subScalar(zScore * std);
        IVector<Double> upperBound = forecast.addScalar(zScore * std);

        double mse = calculateMSE(data, movingAvg);
        double mae = calculateMAE(data, movingAvg);
        double mape = calculateMAPE(data, movingAvg);

        return new ForecastResult(forecast, lowerBound, upperBound, mse, mae, mape, "SMA", confidenceLevel);
    }

    public static ForecastResult exponentialSmoothing(IVector<Double> data,
            double alpha, int forecastSteps, double confidenceLevel) {
        int length = data.length();
        IVector<Double> smoothed = exponentialSmoothingImpl(data, alpha);

        IVector<Double> forecast = Linalg.zeros(forecastSteps);
        double lastValue = smoothed.get(length - 1);
        for (int i = 0; i < forecastSteps; i++) {
            forecast.set(i, lastValue);
        }

        double std = calculateForecastStd(data, smoothed, alpha);
        double zScore = getZScore(confidenceLevel);
        IVector<Double> lowerBound = forecast.subScalar(zScore * std);
        IVector<Double> upperBound = forecast.addScalar(zScore * std);

        double mse = calculateMSE(data, smoothed);
        double mae = calculateMAE(data, smoothed);
        double mape = calculateMAPE(data, smoothed);

        return new ForecastResult(forecast, lowerBound, upperBound, mse, mae, mape, "ES", confidenceLevel);
    }

    public static ForecastResult linearRegression(IVector<Double> data,
            int forecastSteps, double confidenceLevel) {
        int length = data.length();
        Tuple2<Double, Double> regression = linearRegressionImpl(data);
        double slope = regression._1;
        double intercept = regression._2;

        IVector<Double> forecast = Linalg.zeros(forecastSteps);
        for (int i = 0; i < forecastSteps; i++) {
            forecast.set(i, slope * (length + i) + intercept);
        }

        double std = calculateRegressionStd(data, slope, intercept);
        double zScore = getZScore(confidenceLevel);
        IVector<Double> lowerBound = forecast.subScalar(zScore * std);
        IVector<Double> upperBound = forecast.addScalar(zScore * std);

        IVector<Double> fitted = Linalg.range(length).multiplyByScalar(slope).addScalar(intercept);
        double mse = calculateMSE(data, fitted);
        double mae = calculateMAE(data, fitted);
        double mape = calculateMAPE(data, fitted);

        return new ForecastResult(forecast, lowerBound, upperBound, mse, mae, mape, "LR", confidenceLevel);
    }

    public static ForecastResult arimaForecast(IVector<Double> data,
            int p, int d, int q, int forecastSteps, double confidenceLevel) {
        UnifiedARIMAModel model = UnifiedARIMAModel.fit(data, p, d, q);
        ITimeSeriesForecastResult forecastResult = model.forecastWithConfidence(forecastSteps, confidenceLevel);
        IVector<Double> forecast = forecastResult.getForecastVector();
        IVector<Double> lowerBound = forecastResult.getLowerBoundsVector();
        IVector<Double> upperBound = forecastResult.getUpperBoundsVector();
        double[] errorMetrics = forecastResult.getErrorMetrics();
        return new ForecastResult(forecast, lowerBound, upperBound,
                errorMetrics[0], errorMetrics[1], errorMetrics[2], "ARIMA", confidenceLevel);
    }

    public static ForecastResult seasonalForecast(IVector<Double> data,
            int period, int forecastSteps, double confidenceLevel) {
        int length = data.length();
        IVector<Double> seasonal = calculateSeasonalComponent(data, period);
        IVector<Double> deseasonalized = data.sub(seasonal);

        ForecastResult baseForecast = linearRegression(deseasonalized, forecastSteps, confidenceLevel);
        IVector<Double> extendedSeasonal = extendSeasonal(seasonal, period, forecastSteps);

        return new ForecastResult(
                baseForecast.forecast.add(extendedSeasonal),
                baseForecast.lowerBound.add(extendedSeasonal),
                baseForecast.upperBound.add(extendedSeasonal),
                baseForecast.mse, baseForecast.mae, baseForecast.mape, "Seasonal", confidenceLevel);
    }

    public static ForecastResult holtWintersForecast(IVector<Double> data,
            double alpha, double beta, double gamma, int period,
            int forecastSteps, double confidenceLevel) {
        ExponentialSmoothingModels.HoltWintersSmoothing model
                = ExponentialSmoothingModels.HoltWintersSmoothing.fit(data, alpha, beta, gamma, period);

        IVector<Double> forecast = model.forecast(forecastSteps);
        double std = Math.sqrt(model.getMse());
        double zScore = getZScore(confidenceLevel);

        return new ForecastResult(forecast,
                forecast.subScalar(zScore * std),
                forecast.addScalar(zScore * std),
                model.getMse(), model.getMae(), 0.0, "HoltWinters", confidenceLevel);
    }

    public static ForecastResult garchForecast(IVector<Double> data,
            int p, int q, int forecastSteps, double confidenceLevel) {
        GARCHModel model = GARCHModel.fit(data, p, q);
        GARCHModel.GARCHForecastResult garchResult = model.forecastReturns(forecastSteps, confidenceLevel);
        return new ForecastResult(garchResult.meanForecast, garchResult.lowerBound, garchResult.upperBound,
                model.getVariance().get(model.getVariance().length() - 1), 0.0, 0.0, "GARCH", confidenceLevel);
    }

    public static ForecastResult stateSpaceForecast(IVector<Double> data,
            double sigmaEta, double sigmaZeta, double sigmaEpsilon,
            int forecastSteps, double confidenceLevel) {
        StateSpaceModel model = StateSpaceModel.createLocalLinearTrend(sigmaEta, sigmaZeta, sigmaEpsilon);
        model.runKalmanFilter(data);
        StateSpaceModel.StateSpaceForecastResult forecastResult = model.forecast(forecastSteps);

        IVector<Double> forecast = Linalg.zeros(forecastSteps);
        IVector<Double> lowerBound = Linalg.zeros(forecastSteps);
        IVector<Double> upperBound = Linalg.zeros(forecastSteps);

        for (int i = 0; i < forecastSteps; i++) {
            forecast.set(i, forecastResult.forecastObservations.get(i).get(0));
            double std = forecastResult.forecastStd.get(i).get(0);
            double zScore = getZScore(confidenceLevel);
            lowerBound.set(i, forecast.get(i) - zScore * std);
            upperBound.set(i, forecast.get(i) + zScore * std);
        }

        return new ForecastResult(forecast, lowerBound, upperBound, 0.0, 0.0, 0.0, "StateSpace", confidenceLevel);
    }

    public static ForecastResult autoForecast(IVector<Double> data,
            int forecastSteps, double confidenceLevel) {
        ForecastResult bestResult = null;
        double bestMSE = Double.POSITIVE_INFINITY;

        try {
            ExponentialSmoothingModels.SimpleExponentialSmoothing sesModel
                    = ExponentialSmoothingModels.SimpleExponentialSmoothing.fit(data, 0.3);
            IVector<Double> sesForecast = sesModel.forecast(forecastSteps);
            double std = Math.sqrt(sesModel.getMse());
            double zScore = getZScore(confidenceLevel);
            ForecastResult sesResult = new ForecastResult(sesForecast,
                    sesForecast.subScalar(zScore * std),
                    sesForecast.addScalar(zScore * std),
                    sesModel.getMse(), sesModel.getMae(), 0.0, "SES", confidenceLevel);
            if (sesModel.getMse() < bestMSE) {
                bestMSE = sesModel.getMse();
                bestResult = sesResult;
            }
        } catch (Exception e) { /* skip */ }

        try {
            ForecastResult arimaResult = arimaForecast(data, 1, 1, 1, forecastSteps, confidenceLevel);
            if (arimaResult.mse < bestMSE) {
                bestMSE = arimaResult.mse;
                bestResult = arimaResult;
            }
        } catch (Exception e) { /* skip */ }

        try {
            ForecastResult lrResult = linearRegression(data, forecastSteps, confidenceLevel);
            if (lrResult.mse < bestMSE) {
                bestMSE = lrResult.mse;
                bestResult = lrResult;
            }
        } catch (Exception e) { /* skip */ }

        if (bestResult == null) {
            throw new RuntimeException("无法找到合适的预测模型");
        }
        return bestResult;
    }

    // ========== private helpers ==========

    private static IVector<Double> calculateMovingAverage(IVector<Double> data, int windowSize) {
        int length = data.length();
        IVector<Double> movingAvg = Linalg.zeros(length);
        for (int i = 0; i < length; i++) {
            int start = Math.max(0, i - windowSize + 1);
            movingAvg.set(i, data.slice(start, i + 1).meanValue());
        }
        return movingAvg;
    }

    private static IVector<Double> exponentialSmoothingImpl(IVector<Double> data, double alpha) {
        return TimeSeriesUtils.exponentialSmoothing(data, alpha);
    }

    private static Tuple2<Double, Double> linearRegressionImpl(IVector<Double> data) {
        int length = data.length();
        IVector<Double> x = Linalg.range(length);
        double sumX = x.sumValue();
        double sumY = data.sumValue();
        double sumXY = x.multiply(data).sumValue();
        double sumXX = x.multiply(x).sumValue();
        double slope = (length * sumXY - sumX * sumY) / (length * sumXX - sumX * sumX);
        double intercept = (sumY - slope * sumX) / length;
        return new Tuple2<>(slope, intercept);
    }

    private static IVector<Double> calculateSeasonalComponent(IVector<Double> data, int period) {
        return TimeSeriesUtils.calculateSeasonalComponent(data, period);
    }

    private static IVector<Double> extendSeasonal(IVector<Double> seasonal, int period, int steps) {
        IVector<Double> extended = Linalg.zeros(steps);
        for (int i = 0; i < steps; i++) {
            extended.set(i, seasonal.get(i % period));
        }
        return extended;
    }

    private static double calculateForecastStd(IVector<Double> data, IVector<Double> fitted, double alpha) {
        return data.sub(fitted).stdValue() * Math.sqrt(1 + alpha * alpha);
    }

    private static double calculateRegressionStd(IVector<Double> data, double slope, double intercept) {
        int length = data.length();
        IVector<Double> fitted = Linalg.range(length).multiplyByScalar(slope).addScalar(intercept);
        return data.sub(fitted).stdValue();
    }

    private static double getZScore(double confidenceLevel) {
        if (confidenceLevel == 0.95) return 1.96;
        if (confidenceLevel == 0.90) return 1.645;
        if (confidenceLevel == 0.99) return 2.576;
        return 1.96;
    }

    private static double calculateMSE(IVector<Double> actual, IVector<Double> predicted) {
        IVector<Double> errors = actual.sub(predicted);
        return errors.multiply(errors).meanValue();
    }

    private static double calculateMAE(IVector<Double> actual, IVector<Double> predicted) {
        return actual.sub(predicted).apply(Math::abs).meanValue();
    }

    private static double calculateMAPE(IVector<Double> actual, IVector<Double> predicted) {
        return actual.sub(predicted).divide(actual).apply(Math::abs).meanValue() * 100;
    }
}
