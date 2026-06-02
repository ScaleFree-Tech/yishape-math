package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.plot.IPlot;

import java.util.List;

/**
 * 时间序列可视化包装器 / Time Series Visualization Wrapper.
 * 提供统一的时间序列绘图入口。
 */
public class PlotWrapper {

    public IPlot plotTimeSeries(TimeSeriesData timeSeriesData, String title) {
        return TimeSeriesPlots.plotTimeSeries(timeSeriesData, title);
    }

    public IPlot plotTrendAnalysis(TimeSeriesData timeSeriesData, String title) {
        return TimeSeriesPlots.plotTrendAnalysis(timeSeriesData, title);
    }

    public IPlot plotSeasonalDecomposition(TimeSeriesData timeSeriesData, int period, String title) {
        return TimeSeriesPlots.plotSeasonalDecomposition(timeSeriesData, period, title);
    }

    public IPlot plotAutocorrelation(TimeSeriesData timeSeriesData, int maxLag, String title) {
        return TimeSeriesPlots.plotAutocorrelation(timeSeriesData, maxLag, title);
    }

    public IPlot plotPartialAutocorrelation(TimeSeriesData timeSeriesData, int maxLag, String title) {
        return TimeSeriesPlots.plotPartialAutocorrelation(timeSeriesData, maxLag, title);
    }

    public IPlot plotForecasting(TimeSeriesData timeSeriesData, int forecastSteps, String title) {
        return TimeSeriesPlots.plotForecasting(timeSeriesData, forecastSteps, title);
    }

    public IPlot plotTimeSeriesStatistics(TimeSeriesData timeSeriesData, String title) {
        return TimeSeriesPlots.plotTimeSeriesStatistics(timeSeriesData, title);
    }

    public IPlot plotMultivariateTimeSeries(TimeSeriesData timeSeriesData, String title) {
        return TimeSeriesPlots.plotMultivariateTimeSeries(timeSeriesData, title);
    }

    public IPlot plotTimeSeriesFeatures(TimeSeriesData timeSeriesData, String title) {
        return TimeSeriesPlots.plotTimeSeriesFeatures(timeSeriesData, title);
    }

    public List<IPlot> createTimeSeriesDashboard(TimeSeriesData timeSeriesData, String title) {
        return TimeSeriesPlots.createTimeSeriesDashboard(timeSeriesData, title);
    }
}
