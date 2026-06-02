package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.timeseries.model.ITimeSeriesForecastResult;
import com.yishape.lab.math.timeseries.model.ITimeSeriesModel;

/**
 * 预测结果类 / Forecasting Result Class
 * <p>
 * 存储时间序列预测的结果，包括预测值、置信区间和误差指标。
 * 用于TimeSeriesForecasting类返回预测分析的完整结果。
 * </p>
 * <p>
 * Stores the results of time series forecasting, including forecast values,
 * confidence intervals, and error metrics. Used by TimeSeriesForecasting class
 * to return complete results of forecast analysis.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ForecastResult implements ITimeSeriesForecastResult {

    /** 预测值 / Forecast values */
    public final IVector<Double> forecast;
    /** 预测下界 / Forecast lower bound */
    public final IVector<Double> lowerBound;
    /** 预测上界 / Forecast upper bound */
    public final IVector<Double> upperBound;
    /** 均方误差 / Mean squared error */
    public final double mse;
    /** 平均绝对误差 / Mean absolute error */
    public final double mae;
    /** 平均绝对百分比误差 / Mean absolute percentage error */
    public final double mape;
    /** 模型类型 / Model type */
    public final String modelType;
    /** 置信水平 / Confidence level */
    public final double confidenceLevel;

    /**
     * 构造函数 / Constructor
     *
     * @param forecast 预测值 / Forecast values
     * @param lowerBound 预测下界 / Forecast lower bound
     * @param upperBound 预测上界 / Forecast upper bound
     * @param mse 均方误差 / Mean squared error
     * @param mae 平均绝对误差 / Mean absolute error
     * @param mape 平均绝对百分比误差 / Mean absolute percentage error
     * @param modelType 模型类型 / Model type
     * @param confidenceLevel 置信水平 / Confidence level
     */
    public ForecastResult(IVector<Double> forecast, IVector<Double> lowerBound, IVector<Double> upperBound,
            double mse, double mae, double mape, String modelType, double confidenceLevel) {
        this.forecast = forecast;
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
        this.mse = mse;
        this.mae = mae;
        this.mape = mape;
        this.modelType = modelType;
        this.confidenceLevel = confidenceLevel;
    }

    @Override
    public Object getForecast() {
        return forecast;
    }

    @Override
    public IVector<Double> getForecastVector() {
        return forecast;
    }

    @Override
    public IMatrix<Double> getForecastMatrix() {
        throw new UnsupportedOperationException("Not a multivariate forecast");
    }

    @Override
    public Object getLowerBounds() {
        return lowerBound;
    }

    @Override
    public IVector<Double> getLowerBoundsVector() {
        return lowerBound;
    }

    @Override
    public IMatrix<Double> getLowerBoundsMatrix() {
        throw new UnsupportedOperationException("Not a multivariate forecast");
    }

    @Override
    public Object getUpperBounds() {
        return upperBound;
    }

    @Override
    public IVector<Double> getUpperBoundsVector() {
        return upperBound;
    }

    @Override
    public IMatrix<Double> getUpperBoundsMatrix() {
        throw new UnsupportedOperationException("Not a multivariate forecast");
    }

    @Override
    public Object getStandardDeviations() {
        if (lowerBound == null || upperBound == null) return null;
        double z = confidenceLevel >= 0.99 ? 2.576 : confidenceLevel >= 0.95 ? 1.96 : 1.645;
        IVector<Double> std = IVector.of(new double[forecast.length()]);
        for (int i = 0; i < forecast.length(); i++) {
            std.set(i, (upperBound.get(i) - lowerBound.get(i)) / (2 * z));
        }
        return std;
    }

    @Override
    public IVector<Double> getStandardDeviationsVector() {
        return (IVector<Double>) getStandardDeviations();
    }

    @Override
    public IMatrix<Double> getStandardDeviationsMatrix() {
        throw new UnsupportedOperationException("Not a multivariate forecast");
    }

    @Override
    public double getConfidenceLevel() {
        return confidenceLevel;
    }

    @Override
    public int getForecastSteps() {
        return forecast.length();
    }

    @Override
    public int getVariableCount() {
        return 1;
    }

    @Override
    public String[] getVariableNames() {
        return new String[]{"value"};
    }

    @Override
    public double[] getErrorMetrics() {
        double rmse = Math.sqrt(mse);
        return new double[]{mse, mae, mape, rmse};
    }

    @Override
    public ITimeSeriesModel.ModelType getModelType() {
        try {
            return ITimeSeriesModel.ModelType.valueOf(modelType);
        } catch (IllegalArgumentException e) {
            return ITimeSeriesModel.ModelType.ARIMA;
        }
    }

    @Override
    public Object getTimePoints() {
        return null;
    }

    @Override
    public boolean hasConfidenceIntervals() {
        return lowerBound != null && upperBound != null;
    }

    @Override
    public boolean isMultivariate() {
        return false;
    }

    @Override
    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("Forecast Result (%s, %.0f%% CI)\n", modelType, confidenceLevel * 100));
        sb.append(String.format("Steps: %d, MSE=%.4f, MAE=%.4f, MAPE=%.2f%%\n", getForecastSteps(), mse, mae, mape * 100));
        sb.append("Forecast values: ");
        for (int i = 0; i < Math.min(5, forecast.length()); i++) {
            sb.append(String.format("%.4f", forecast.get(i)));
            if (i < Math.min(5, forecast.length()) - 1) sb.append(", ");
        }
        if (forecast.length() > 5) sb.append("...");
        return sb.toString();
    }

    @Override
    public String export(String format) {
        if ("CSV".equalsIgnoreCase(format) || "csv".equalsIgnoreCase(format)) {
            StringBuilder sb = new StringBuilder("step,forecast,lower,upper\n");
            for (int i = 0; i < forecast.length(); i++) {
                sb.append(String.format("%d,%.6f,%.6f,%.6f\n", i + 1, forecast.get(i),
                        lowerBound != null ? lowerBound.get(i) : Double.NaN,
                        upperBound != null ? upperBound.get(i) : Double.NaN));
            }
            return sb.toString();
        }
        return getSummary();
    }
}
