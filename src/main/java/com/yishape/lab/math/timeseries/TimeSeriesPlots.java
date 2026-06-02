package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.plot.Plots;
import com.yishape.lab.math.plot.IPlot;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * 时间序列可视化器类 / Time Series Visualizer Class
 * <p>
 * 提供时间序列数据的可视化功能，包括时间序列图、趋势分析图、季节性分解图、
 * 自相关图、预测图等。使用项目现有的viz包功能进行时间序列可视化。
 * </p>
 * <p>
 * Provides time series data visualization functionality including time series plots, 
 * trend analysis plots, seasonal decomposition plots, autocorrelation plots, 
 * forecasting plots, etc. Uses existing viz package functionality for time series visualization.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class TimeSeriesPlots {
    
    /**
     * 绘制时间序列图 / Plot time series
     * <p>
     * 显示时间序列的基本图形。
     * Display basic time series plot.
     * </p>
     *
     * @param timeSeriesData 时间序列数据 / Time series data
     * @param title 图表标题 / Plot title
     * @return 时间序列图对象 / Time series plot object
     */
    public static IPlot plotTimeSeries(TimeSeriesData timeSeriesData, String title) {
        IPlot plot = Plots.of();
        plot.title(title);
        plot.xlabel("时间 / Time");
        plot.ylabel("数值 / Value");
        
        // 获取时间戳和数值 / Get timestamps and values
        List<LocalDateTime> timestamps = timeSeriesData.getTimestamps();
        IMatrix<Double> data = timeSeriesData.getData();
        
        // 创建时间轴（使用索引） / Create time axis (using indices)
        double[] timeArray = new double[timestamps.size()];
        for (int i = 0; i < timestamps.size(); i++) {
            timeArray[i] = i;
        }
        IVector<Double> time = Linalg.vector(timeArray);
        
        // 绘制每个变量 / Plot each variable
        for (int col = 0; col < data.cols(); col++) {
            IVector<Double> values = data.getColumn(col);
            String seriesName = timeSeriesData.getVariableNames()[col];
            List<String> labels = new ArrayList<>();
            labels.add(seriesName);
            plot.line(time, values, labels);
        }
        
        return plot;
    }
    
    /**
     * 绘制趋势分析图 / Plot trend analysis
     * <p>
     * 显示时间序列的趋势分析结果。
     * Display trend analysis results of time series.
     * </p>
     *
     * @param timeSeriesData 时间序列数据 / Time series data
     * @param title 图表标题 / Plot title
     * @return 趋势分析图对象 / Trend analysis plot object
     */
    public static IPlot plotTrendAnalysis(TimeSeriesData timeSeriesData, String title) {
        IPlot plot = Plots.of();
        plot.title(title);
        plot.xlabel("时间 / Time");
        plot.ylabel("数值 / Value");
        
        // 获取第一个变量的数据 / Get first variable data
        IVector<Double> values = timeSeriesData.getData().getColumn(0);
        
        // 进行趋势分析 / Perform trend analysis
        TimeSeriesUtils.TrendResult trendResult = TimeSeriesUtils.analyzeTrend(values);
        IVector<Double> trend = trendResult.trend;
        
        // 创建时间轴 / Create time axis
        double[] timeArray = new double[values.length()];
        for (int i = 0; i < values.length(); i++) {
            timeArray[i] = i;
        }
        IVector<Double> time = Linalg.vector(timeArray);
        
        // 绘制原始数据和趋势 / Plot original data and trend
        List<String> labels1 = new ArrayList<>();
        labels1.add("原始数据 / Original Data");
        List<String> labels2 = new ArrayList<>();
        labels2.add("趋势 / Trend");
        plot.line(time, values, labels1);
        plot.line(time, trend, labels2);
        
        return plot;
    }
    
    /**
     * 绘制季节性分解图 / Plot seasonal decomposition
     * <p>
     * 显示时间序列的季节性分解结果。
     * Display seasonal decomposition results of time series.
     * </p>
     *
     * @param timeSeriesData 时间序列数据 / Time series data
     * @param period 季节周期 / Seasonal period
     * @param title 图表标题 / Plot title
     * @return 季节性分解图对象 / Seasonal decomposition plot object
     */
    public static IPlot plotSeasonalDecomposition(TimeSeriesData timeSeriesData, int period, String title) {
        IPlot plot = Plots.of();
        plot.title(title);
        plot.xlabel("时间 / Time");
        plot.ylabel("数值 / Value");
        
        // 获取第一个变量的数据 / Get first variable data
        IVector<Double> values = timeSeriesData.getData().getColumn(0);
        
        // 进行季节性分解 / Perform seasonal decomposition
        DecompositionResult decomposition = 
            TimeSeriesDecomposition.classicalDecomposition(values, period, TimeSeriesDecomposition.DecompositionModel.ADDITIVE);
        
        // 创建时间轴 / Create time axis
        double[] timeArray = new double[values.length()];
        for (int i = 0; i < values.length(); i++) {
            timeArray[i] = i;
        }
        IVector<Double> time = Linalg.vector(timeArray);
        
        // 绘制分解结果 / Plot decomposition results
        List<String> labels1 = new ArrayList<>();
        labels1.add("原始数据 / Original Data");
        List<String> labels2 = new ArrayList<>();
        labels2.add("趋势 / Trend");
        List<String> labels3 = new ArrayList<>();
        labels3.add("季节性 / Seasonal");
        List<String> labels4 = new ArrayList<>();
        labels4.add("残差 / Residual");
        plot.line(time, values, labels1);
        plot.line(time, decomposition.trend, labels2);
        plot.line(time, decomposition.seasonal, labels3);
        plot.line(time, decomposition.residual, labels4);
        
        return plot;
    }
    
    /**
     * 绘制自相关图 / Plot autocorrelation
     * <p>
     * 显示时间序列的自相关函数。
     * Display autocorrelation function of time series.
     * </p>
     *
     * @param timeSeriesData 时间序列数据 / Time series data
     * @param maxLag 最大滞后 / Maximum lag
     * @param title 图表标题 / Plot title
     * @return 自相关图对象 / Autocorrelation plot object
     */
    public static IPlot plotAutocorrelation(TimeSeriesData timeSeriesData, int maxLag, String title) {
        IPlot plot = Plots.of();
        plot.title(title);
        plot.xlabel("滞后 / Lag");
        plot.ylabel("自相关系数 / Autocorrelation");
        
        // 计算自相关 / Calculate autocorrelation
        IVector<Double> values = timeSeriesData.getData().getColumn(0);
        IVector<Double> autocorr = TimeSeriesUtils.calculateAutocorrelation(values, maxLag);
        
        // 创建滞后轴 / Create lag axis
        double[] lagArray = new double[maxLag + 1];
        for (int i = 0; i <= maxLag; i++) {
            lagArray[i] = i;
        }
        IVector<Double> lags = Linalg.vector(lagArray);
        
        plot.line(lags, autocorr);
        // 添加零线 / Add zero line
        // Note: axhline method not available in current IPlot interface
        
        return plot;
    }
    
    /**
     * 绘制偏自相关图 / Plot partial autocorrelation
     * <p>
     * 显示时间序列的偏自相关函数。
     * Display partial autocorrelation function of time series.
     * </p>
     *
     * @param timeSeriesData 时间序列数据 / Time series data
     * @param maxLag 最大滞后 / Maximum lag
     * @param title 图表标题 / Plot title
     * @return 偏自相关图对象 / Partial autocorrelation plot object
     */
    public static IPlot plotPartialAutocorrelation(TimeSeriesData timeSeriesData, int maxLag, String title) {
        IPlot plot = Plots.of();
        plot.title(title);
        plot.xlabel("滞后 / Lag");
        plot.ylabel("偏自相关系数 / Partial Autocorrelation");
        
        // 计算偏自相关 / Calculate partial autocorrelation
        IVector<Double> values = timeSeriesData.getData().getColumn(0);
        IVector<Double> pacf = TimeSeriesUtils.calculatePartialAutocorrelation(values, maxLag);
        
        // 创建滞后轴 / Create lag axis
        double[] lagArray = new double[maxLag + 1];
        for (int i = 0; i <= maxLag; i++) {
            lagArray[i] = i;
        }
        IVector<Double> lags = Linalg.vector(lagArray);
        
        plot.line(lags, pacf);
        // 添加零线 / Add zero line
        // Note: axhline method not available in current IPlot interface
        
        return plot;
    }
    
    /**
     * 绘制预测图 / Plot forecasting
     * <p>
     * 显示时间序列的预测结果。
     * Display forecasting results of time series.
     * </p>
     *
     * @param timeSeriesData 时间序列数据 / Time series data
     * @param forecastSteps 预测步数 / Forecast steps
     * @param title 图表标题 / Plot title
     * @return 预测图对象 / Forecasting plot object
     */
    public static IPlot plotForecasting(TimeSeriesData timeSeriesData, int forecastSteps, String title) {
        IPlot plot = Plots.of();
        plot.title(title);
        plot.xlabel("时间 / Time");
        plot.ylabel("数值 / Value");
        
        // 获取第一个变量的数据 / Get first variable data
        IVector<Double> values = timeSeriesData.getData().getColumn(0);
        
        // 进行简单预测（使用移动平均） / Perform simple forecasting (using moving average)
        int windowSize = Math.min(5, values.length() / 4); // 自适应窗口大小 / Adaptive window size
        IVector<Double> forecast = TimeSeriesUtils.movingAverageForecast(values, forecastSteps, windowSize);
        
        // 创建时间轴 / Create time axis
        int totalLength = values.length() + forecastSteps;
        double[] timeArray = new double[totalLength];
        for (int i = 0; i < totalLength; i++) {
            timeArray[i] = i;
        }
        IVector<Double> time = Linalg.vector(timeArray);
        
        // 创建完整数据（历史+预测） / Create complete data (history + forecast)
        double[] fullDataArray = new double[totalLength];
        for (int i = 0; i < values.length(); i++) {
            fullDataArray[i] = values.get(i);
        }
        for (int i = 0; i < forecastSteps; i++) {
            fullDataArray[values.length() + i] = forecast.get(i);
        }
        IVector<Double> fullData = Linalg.vector(fullDataArray);
        
        // 绘制历史数据和预测 / Plot historical data and forecast
        List<String> labels = new ArrayList<>();
        labels.add("历史+预测 / History+Forecast");
        plot.line(time, fullData, labels);
        
        // 标记预测开始点 / Mark forecast start point
        // Note: axvline method not available in current IPlot interface
        
        return plot;
    }
    
    /**
     * 绘制时间序列统计图 / Plot time series statistics
     * <p>
     * 显示时间序列的基本统计信息。
     * Display basic statistics of time series.
     * </p>
     *
     * @param timeSeriesData 时间序列数据 / Time series data
     * @param title 图表标题 / Plot title
     * @return 统计图对象 / Statistics plot object
     */
    public static IPlot plotTimeSeriesStatistics(TimeSeriesData timeSeriesData, String title) {
        IPlot plot = Plots.of();
        plot.title(title);
        plot.xlabel("统计量 / Statistics");
        plot.ylabel("数值 / Value");
        
        // 获取第一个变量的数据 / Get first variable data
        IVector<Double> values = timeSeriesData.getData().getColumn(0);
        
        // 计算统计量 / Calculate statistics
        double mean = values.meanValue();
        double std = values.stdValue();
        double min = values.minValue();
        double max = values.maxValue();
        double skewness = values.skewness();
        double kurtosis = values.kurtosis();
        
        String[] statNames = {"均值 / Mean", "标准差 / Std", "最小值 / Min", 
                            "最大值 / Max", "偏度 / Skewness", "峰度 / Kurtosis"};
        double[] statValues = {mean, std, min, max, skewness, kurtosis};
        
        IVector<Double> statValuesVector = Linalg.vector(statValues);
        List<String> labels = new ArrayList<>();
        for (String name : statNames) {
            labels.add(name);
        }
        
        plot.bar(labels,statValuesVector);
        
        return plot;
    }
    
    /**
     * 绘制多变量时间序列图 / Plot multivariate time series
     * <p>
     * 显示多变量时间序列的相关性。
     * Display correlation of multivariate time series.
     * </p>
     *
     * @param timeSeriesData 时间序列数据 / Time series data
     * @param title 图表标题 / Plot title
     * @return 多变量时间序列图对象 / Multivariate time series plot object
     */
    public static IPlot plotMultivariateTimeSeries(TimeSeriesData timeSeriesData, String title) {
        IPlot plot = Plots.of();
        plot.title(title);
        plot.xlabel("时间 / Time");
        plot.ylabel("数值 / Value");
        
        // 获取数据 / Get data
        IMatrix<Double> data = timeSeriesData.getData();
        String[] variableNames = timeSeriesData.getVariableNames();
        
        // 创建时间轴 / Create time axis
        double[] timeArray = new double[data.rows()];
        for (int i = 0; i < data.rows(); i++) {
            timeArray[i] = i;
        }
        IVector<Double> time = Linalg.vector(timeArray);
        
        // 绘制每个变量 / Plot each variable
        for (int col = 0; col < data.cols(); col++) {
            IVector<Double> values = data.getColumn(col);
            List<String> labels = new ArrayList<>();
            labels.add(variableNames[col]);
            plot.line(time, values, labels);
        }
        
        return plot;
    }
    
    /**
     * 绘制时间序列特征图 / Plot time series features
     * <p>
     * 显示时间序列的特征分析结果。
     * Display time series feature analysis results.
     * </p>
     *
     * @param timeSeriesData 时间序列数据 / Time series data
     * @param title 图表标题 / Plot title
     * @return 特征图对象 / Features plot object
     */
    public static IPlot plotTimeSeriesFeatures(TimeSeriesData timeSeriesData, String title) {
        IPlot plot = Plots.of();
        plot.title(title);
        plot.xlabel("特征 / Features");
        plot.ylabel("数值 / Value");
        
        // 获取第一个变量的数据 / Get first variable data
        IVector<Double> values = timeSeriesData.getData().getColumn(0);
        
        // 计算特征 / Calculate features
        String[] featureNames = {"均值 / Mean", "方差 / Variance", "偏度 / Skewness", 
                               "峰度 / Kurtosis", "自相关 / Autocorr", "趋势强度 / Trend"};
        double[] featureValues = {
            values.meanValue(),
            values.varValue(),
            TimeSeriesUtils.calculateSkewness(values),
            TimeSeriesUtils.calculateKurtosis(values),
            TimeSeriesUtils.calculateAutocorrelation(values, 1).get(1),
            TimeSeriesUtils.detectTrendStrength(values)
        };
        
        IVector<Double> featureValuesVector = Linalg.vector(featureValues);
        List<String> labels = new ArrayList<>();
        for (String name : featureNames) {
            labels.add(name);
        }
        
        plot.bar(labels,featureValuesVector);
        
        return plot;
    }
    
    /**
     * 创建时间序列可视化仪表板 / Create time series visualization dashboard
     * <p>
     * 创建包含多个图表的时间序列可视化仪表板。
     * Create time series visualization dashboard with multiple charts.
     * </p>
     *
     * @param timeSeriesData 时间序列数据 / Time series data
     * @param title 仪表板标题 / Dashboard title
     * @return 可视化图表列表 / List of visualization plots
     */
    public static List<IPlot> createTimeSeriesDashboard(TimeSeriesData timeSeriesData, String title) {
        List<IPlot> plots = new ArrayList<>();
        
        // 添加各种图表 / Add various plots
        plots.add(plotTimeSeries(timeSeriesData, title + " - 时间序列图 / Time Series"));
        plots.add(plotTrendAnalysis(timeSeriesData, title + " - 趋势分析 / Trend Analysis"));
        plots.add(plotSeasonalDecomposition(timeSeriesData, 12, title + " - 季节性分解 / Seasonal Decomposition"));
        plots.add(plotAutocorrelation(timeSeriesData, Math.min(50, timeSeriesData.getData().rows() / 4), 
                                    title + " - 自相关 / Autocorrelation"));
        plots.add(plotPartialAutocorrelation(timeSeriesData, Math.min(20, timeSeriesData.getData().rows() / 8), 
                                           title + " - 偏自相关 / Partial Autocorrelation"));
        plots.add(plotForecasting(timeSeriesData, 10, title + " - 预测 / Forecasting"));
        plots.add(plotTimeSeriesStatistics(timeSeriesData, title + " - 统计信息 / Statistics"));
        plots.add(plotTimeSeriesFeatures(timeSeriesData, title + " - 特征分析 / Feature Analysis"));
        
        return plots;
    }
    
}
