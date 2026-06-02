package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.timeseries.model.ITimeSeriesForecastResult;
import com.yishape.lab.math.timeseries.model.ITimeSeriesDiagnostics;
import com.yishape.lab.math.timeseries.model.TimeSeriesModelFactory;
import com.yishape.lab.math.timeseries.model.ITimeSeriesModel;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 统一时间序列分析工具类 / Unified Time Series Analysis Tool Class
 * <p>
 * 提供统一的时间序列分析功能，包括数据预处理、模型选择、预测、诊断等。 整合各种时间序列模型，提供一致的分析接口。
 * </p>
 * <p>
 * Provides unified time series analysis functionality including data
 * preprocessing, model selection, forecasting, diagnostics, etc. Integrates
 * various time series models and provides consistent analysis interface.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class TimeSeriesAnalyzer {

    private final IVector<Double> data;
    private final String name;
    private final LocalDateTime[] timestamps;
    private ITimeSeriesModel currentModel;
    private ITimeSeriesForecastResult lastForecast;
    private ITimeSeriesDiagnostics lastDiagnostics;

    /**
     * 分析配置类 / Analysis Configuration Class
     */
    public static class AnalysisConfig {

        public final ITimeSeriesModel.ModelType[] candidateModels;
        public final TimeSeriesModelFactory.SelectionCriterion selectionCriterion;
        public final int forecastSteps;
        public final double confidenceLevel;
        public final boolean performDiagnostics;
        public final boolean autoPreprocess;
        public final Map<String, Object> modelParameters;

        public AnalysisConfig(ITimeSeriesModel.ModelType[] candidateModels,
                TimeSeriesModelFactory.SelectionCriterion selectionCriterion,
                int forecastSteps, double confidenceLevel, boolean performDiagnostics,
                boolean autoPreprocess, Map<String, Object> modelParameters) {
            this.candidateModels = candidateModels;
            this.selectionCriterion = selectionCriterion;
            this.forecastSteps = forecastSteps;
            this.confidenceLevel = confidenceLevel;
            this.performDiagnostics = performDiagnostics;
            this.autoPreprocess = autoPreprocess;
            this.modelParameters = modelParameters;
        }

        public static class Builder {

            private ITimeSeriesModel.ModelType[] candidateModels = {
                ITimeSeriesModel.ModelType.ARIMA,
                ITimeSeriesModel.ModelType.EXPONENTIAL_SMOOTHING
            };
            private TimeSeriesModelFactory.SelectionCriterion selectionCriterion
                    = TimeSeriesModelFactory.SelectionCriterion.AIC;
            private int forecastSteps = 10;
            private double confidenceLevel = 0.95;
            private boolean performDiagnostics = true;
            private boolean autoPreprocess = true;
            private Map<String, Object> modelParameters = new java.util.HashMap<>();

            public Builder setCandidateModels(ITimeSeriesModel.ModelType[] candidateModels) {
                this.candidateModels = candidateModels;
                return this;
            }

            public Builder setSelectionCriterion(TimeSeriesModelFactory.SelectionCriterion selectionCriterion) {
                this.selectionCriterion = selectionCriterion;
                return this;
            }

            public Builder setForecastSteps(int forecastSteps) {
                this.forecastSteps = forecastSteps;
                return this;
            }

            public Builder setConfidenceLevel(double confidenceLevel) {
                this.confidenceLevel = confidenceLevel;
                return this;
            }

            public Builder setPerformDiagnostics(boolean performDiagnostics) {
                this.performDiagnostics = performDiagnostics;
                return this;
            }

            public Builder setAutoPreprocess(boolean autoPreprocess) {
                this.autoPreprocess = autoPreprocess;
                return this;
            }

            public Builder addModelParameter(String key, Object value) {
                this.modelParameters.put(key, value);
                return this;
            }

            public AnalysisConfig build() {
                return new AnalysisConfig(candidateModels, selectionCriterion, forecastSteps,
                        confidenceLevel, performDiagnostics, autoPreprocess, modelParameters);
            }
        }
    }

    /**
     * 构造函数 / Constructor
     *
     * @param data 时间序列数据 / Time series data
     * @param name 序列名称 / Series name
     */
    public TimeSeriesAnalyzer(IVector<Double> data, String name) {
        this.data = data;
        this.name = name;
        this.timestamps = generateDefaultTimestamps(data.length());
    }

    /**
     * 构造函数 / Constructor
     *
     * @param data 时间序列数据 / Time series data
     * @param name 序列名称 / Series name
     * @param timestamps 时间戳 / Timestamps
     */
    public TimeSeriesAnalyzer(IVector<Double> data, String name, LocalDateTime[] timestamps) {
        this.data = data;
        this.name = name;
        this.timestamps = timestamps;
    }

    /**
     * 执行完整分析 / Perform Complete Analysis
     * <p>
     * 执行完整的时间序列分析，包括模型选择、拟合、预测和诊断。 Perform complete time series analysis
     * including model selection, fitting, forecasting, and diagnostics.
     * </p>
     *
     * @param config 分析配置 / Analysis configuration
     * @return 分析结果 / Analysis result
     */
    public AnalysisResult analyze(AnalysisConfig config) {
        if (config == null) {
            config = new AnalysisConfig.Builder().build();
        }

        // 数据预处理 / Data preprocessing
        IVector<Double> processedData = preprocessData(data, config.autoPreprocess);

        // 模型选择 / Model selection
        ITimeSeriesModel bestModel = selectBestModel(processedData, config);
        this.currentModel = bestModel;

        // 预测 / Forecasting
        ITimeSeriesForecastResult forecast = null;
        if (config.forecastSteps > 0) {
            forecast = bestModel.forecastWithConfidence(config.forecastSteps, config.confidenceLevel);
            this.lastForecast = forecast;
        }

        // 诊断 / Diagnostics
        ITimeSeriesDiagnostics diagnostics = null;
        if (config.performDiagnostics) {
            diagnostics = bestModel.diagnose();
            this.lastDiagnostics = diagnostics;
        }

        // 生成摘要 / Generate summary
        String summary = generateAnalysisSummary(bestModel, forecast, diagnostics, config);

        // 元数据 / Metadata
        Map<String, Object> metadata = generateMetadata(processedData, bestModel, config);

        return new AnalysisResult(bestModel, forecast, diagnostics, summary, metadata);
    }

    /**
     * 快速分析 / Quick Analysis
     * <p>
     * 使用默认配置进行快速分析。 Perform quick analysis using default configuration.
     * </p>
     *
     * @return 分析结果 / Analysis result
     */
    public AnalysisResult quickAnalyze() {
        AnalysisConfig config = new AnalysisConfig.Builder()
                .setForecastSteps(10)
                .setConfidenceLevel(0.95)
                .setPerformDiagnostics(true)
                .build();

        return analyze(config);
    }

    /**
     * 预测 / Forecast
     * <p>
     * 使用当前模型进行预测。 Forecast using current model.
     * </p>
     *
     * @param steps 预测步数 / Forecast steps
     * @param confidenceLevel 置信水平 / Confidence level
     * @return 预测结果 / Forecast result
     */
    public ITimeSeriesForecastResult forecast(int steps, double confidenceLevel) {
        if (currentModel == null) {
            throw new IllegalStateException("请先执行分析或设置模型");
        }

        ITimeSeriesForecastResult result = currentModel.forecastWithConfidence(steps, confidenceLevel);
        this.lastForecast = result;
        return result;
    }

    /**
     * 诊断 / Diagnose
     * <p>
     * 对当前模型进行诊断。 Diagnose current model.
     * </p>
     *
     * @return 诊断结果 / Diagnostic result
     */
    public ITimeSeriesDiagnostics diagnose() {
        if (currentModel == null) {
            throw new IllegalStateException("请先执行分析或设置模型");
        }

        ITimeSeriesDiagnostics result = currentModel.diagnose();
        this.lastDiagnostics = result;
        return result;
    }

    /**
     * 设置模型 / Set Model
     * <p>
     * 设置特定的时间序列模型。 Set specific time series model.
     * </p>
     *
     * @param model 时间序列模型 / Time series model
     */
    public void setModel(ITimeSeriesModel model) {
        this.currentModel = model;
    }

    /**
     * 获取数据统计信息 / Get Data Statistics
     * <p>
     * 获取时间序列数据的基本统计信息。 Get basic statistics of time series data.
     * </p>
     *
     * @return 统计信息 / Statistics
     */
    public Map<String, Object> getDataStatistics() {
        Map<String, Object> stats = new java.util.HashMap<>();

        stats.put("length", data.length());
        stats.put("mean", data.meanValue());
        stats.put("std", data.stdValue());
        stats.put("min", data.minValue());
        stats.put("max", data.maxValue());
        stats.put("skewness", calculateSkewness(data));
        stats.put("kurtosis", calculateKurtosis(data));
        stats.put("isStationary", checkStationarity(data));

        return stats;
    }

    /**
     * 获取趋势分析 / Get Trend Analysis
     * <p>
     * 分析时间序列的趋势特征。 Analyze trend characteristics of time series.
     * </p>
     *
     * @return 趋势分析结果 / Trend analysis result
     */
    public Map<String, Object> getTrendAnalysis() {
        Map<String, Object> trend = new java.util.HashMap<>();

        // 线性趋势 / Linear trend
        double[] linearTrend = calculateLinearTrend(data);
        trend.put("slope", linearTrend[0]);
        trend.put("intercept", linearTrend[1]);
        trend.put("rSquared", linearTrend[2]);

        // 移动平均趋势 / Moving average trend
        IVector<Double> maTrend = calculateMovingAverageTrend(data, 5);
        trend.put("movingAverage", maTrend);

        // 趋势强度 / Trend strength
        double trendStrength = calculateTrendStrength(data);
        trend.put("strength", trendStrength);

        return trend;
    }

    /**
     * 获取季节性分析 / Get Seasonal Analysis
     * <p>
     * 分析时间序列的季节性特征。 Analyze seasonal characteristics of time series.
     * </p>
     *
     * @param period 季节周期 / Seasonal period
     * @return 季节性分析结果 / Seasonal analysis result
     */
    public Map<String, Object> getSeasonalAnalysis(int period) {
        Map<String, Object> seasonal = new java.util.HashMap<>();

        if (data.length() < 2 * period) {
            seasonal.put("error", "数据长度不足以进行季节性分析");
            return seasonal;
        }

        // 季节性强度 / Seasonal strength
        double seasonalStrength = calculateSeasonalStrength(data, period);
        seasonal.put("strength", seasonalStrength);

        // 季节性模式 / Seasonal pattern
        IVector<Double> seasonalPattern = calculateSeasonalPattern(data, period);
        seasonal.put("pattern", seasonalPattern);

        // 季节性检验 / Seasonal test
        boolean hasSeasonality = testSeasonality(data, period);
        seasonal.put("hasSeasonality", hasSeasonality);

        return seasonal;
    }

    /**
     * 导出分析报告 / Export Analysis Report
     * <p>
     * 导出完整的分析报告。 Export complete analysis report.
     * </p>
     *
     * @param format 导出格式 / Export format
     * @return 分析报告 / Analysis report
     */
    public String exportReport(String format) {
        StringBuilder report = new StringBuilder();

        report.append("时间序列分析报告 / Time Series Analysis Report\n");
        report.append("==========================================\n\n");

        report.append("数据信息 / Data Information:\n");
        report.append("名称 / Name: ").append(name).append("\n");
        report.append("长度 / Length: ").append(data.length()).append("\n");
        report.append("时间范围 / Time Range: ");
        if (timestamps != null && timestamps.length > 0) {
            report.append(timestamps[0]).append(" - ").append(timestamps[timestamps.length - 1]);
        } else {
            report.append("未指定 / Not specified");
        }
        report.append("\n\n");

        // 数据统计 / Data statistics
        Map<String, Object> stats = getDataStatistics();
        report.append("数据统计 / Data Statistics:\n");
        for (Map.Entry<String, Object> entry : stats.entrySet()) {
            report.append("  ").append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
        }
        report.append("\n");

        // 模型信息 / Model information
        if (currentModel != null) {
            report.append("模型信息 / Model Information:\n");
            report.append(currentModel.getSummary()).append("\n");
        }

        // 预测结果 / Forecast results
        if (lastForecast != null) {
            report.append("预测结果 / Forecast Results:\n");
            report.append(lastForecast.getSummary()).append("\n");
        }

        // 诊断结果 / Diagnostic results
        if (lastDiagnostics != null) {
            report.append("诊断结果 / Diagnostic Results:\n");
            report.append(lastDiagnostics.getSummary()).append("\n");
        }

        return report.toString();
    }

    // ========== Getter方法 / Getter Methods ==========
    public IVector<Double> getData() {
        return data.copy();
    }

    public String getName() {
        return name;
    }

    public LocalDateTime[] getTimestamps() {
        return timestamps.clone();
    }

    public ITimeSeriesModel getCurrentModel() {
        return currentModel;
    }

    public ITimeSeriesForecastResult getLastForecast() {
        return lastForecast;
    }

    public ITimeSeriesDiagnostics getLastDiagnostics() {
        return lastDiagnostics;
    }

    // ========== 私有辅助方法 / Private Helper Methods ==========
    /**
     * 数据预处理 / Data Preprocessing
     */
    private IVector<Double> preprocessData(IVector<Double> data, boolean autoPreprocess) {
        if (!autoPreprocess) {
            return data;
        }

        // 检查缺失值 / Check for missing values
        IVector<Double> processed = data.copy();

        // 检查异常值 / Check for outliers
        processed = removeOutliers(processed);

        // 检查平稳性 / Check stationarity
        if (!checkStationarity(processed)) {
            // 进行差分 / Perform differencing
            processed = difference(processed);
        }

        return processed;
    }

    /**
     * 选择最优模型 / Select Best Model
     */
    private ITimeSeriesModel selectBestModel(IVector<Double> data, AnalysisConfig config) {
        return TimeSeriesModelFactory.selectBestModel(data, config.candidateModels, config.selectionCriterion);
    }

    /**
     * 生成分析摘要 / Generate Analysis Summary
     */
    private String generateAnalysisSummary(ITimeSeriesModel model, ITimeSeriesForecastResult forecast,
            ITimeSeriesDiagnostics diagnostics, AnalysisConfig config) {
        StringBuilder summary = new StringBuilder();

        summary.append("时间序列分析摘要 / Time Series Analysis Summary\n");
        summary.append("==========================================\n");
        summary.append("序列名称 / Series Name: ").append(name).append("\n");
        summary.append("数据长度 / Data Length: ").append(data.length()).append("\n");
        summary.append("最优模型 / Best Model: ").append(model.getModelName()).append("\n");
        summary.append("模型类型 / Model Type: ").append(model.getModelType()).append("\n");
        summary.append("模型有效性 / Model Validity: ").append(model.isValid() ? "有效 / Valid" : "无效 / Invalid").append("\n");

        if (forecast != null) {
            summary.append("预测步数 / Forecast Steps: ").append(forecast.getForecastSteps()).append("\n");
            summary.append("置信水平 / Confidence Level: ").append(forecast.getConfidenceLevel()).append("\n");
        }

        if (diagnostics != null) {
            summary.append("诊断完成 / Diagnostics Completed: 是 / Yes\n");
        }

        return summary.toString();
    }

    /**
     * 生成元数据 / Generate Metadata
     */
    private Map<String, Object> generateMetadata(IVector<Double> processedData, ITimeSeriesModel model, AnalysisConfig config) {
        Map<String, Object> metadata = new java.util.HashMap<>();

        metadata.put("originalLength", data.length());
        metadata.put("processedLength", processedData.length());
        metadata.put("preprocessingApplied", config.autoPreprocess);
        metadata.put("modelType", model.getModelType());
        metadata.put("modelName", model.getModelName());
        metadata.put("selectionCriterion", config.selectionCriterion);
        metadata.put("forecastSteps", config.forecastSteps);
        metadata.put("confidenceLevel", config.confidenceLevel);
        metadata.put("analysisTimestamp", LocalDateTime.now());

        return metadata;
    }

    /**
     * 生成默认时间戳 / Generate Default Timestamps
     */
    private LocalDateTime[] generateDefaultTimestamps(int length) {
        LocalDateTime[] timestamps = new LocalDateTime[length];
        LocalDateTime start = LocalDateTime.now().minusDays(length - 1);

        for (int i = 0; i < length; i++) {
            timestamps[i] = start.plusDays(i);
        }

        return timestamps;
    }

    /**
     * 计算偏度 / Calculate Skewness
     */
    private double calculateSkewness(IVector<Double> data) {
        double mean = data.meanValue();
        double std = data.stdValue();

        if (std == 0) {
            return 0.0;
        }

        IVector<Double> centered = data.subScalar(mean);
        IVector<Double> cubed = centered.apply(x -> x * x * x);

        return cubed.meanValue() / (std * std * std);
    }

    /**
     * 计算峰度 / Calculate Kurtosis
     */
    private double calculateKurtosis(IVector<Double> data) {
        double mean = data.meanValue();
        double std = data.stdValue();

        if (std == 0) {
            return 0.0;
        }

        IVector<Double> centered = data.subScalar(mean);
        IVector<Double> fourth = centered.apply(x -> x * x * x * x);

        return fourth.meanValue() / (std * std * std * std) - 3.0;
    }

    /**
     * 检查平稳性 / Check Stationarity
     */
    private boolean checkStationarity(IVector<Double> data) {
        // 简化的平稳性检验 / Simplified stationarity test
        double std = data.stdValue();

        // 检查均值是否稳定 / Check if mean is stable
        int n = data.length();
        int half = n / 2;
        double mean1 = data.slice(0, half).meanValue();
        double mean2 = data.slice(half, n).meanValue();

        return Math.abs(mean1 - mean2) < 0.1 * std;
    }

    /**
     * 计算线性趋势 / Calculate Linear Trend
     */
    private double[] calculateLinearTrend(IVector<Double> data) {
        int n = data.length();
        double sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;

        for (int i = 0; i < n; i++) {
            double x = i;
            double y = data.get(i);
            sumX += x;
            sumY += y;
            sumXY += x * y;
            sumXX += x * x;
        }

        double slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        double intercept = (sumY - slope * sumX) / n;

        // 计算R² / Calculate R²
        double yMean = sumY / n;
        double ssTot = 0, ssRes = 0;
        for (int i = 0; i < n; i++) {
            double y = data.get(i);
            double yPred = slope * i + intercept;
            ssTot += (y - yMean) * (y - yMean);
            ssRes += (y - yPred) * (y - yPred);
        }
        double rSquared = 1.0 - (ssRes / ssTot);

        return new double[]{slope, intercept, rSquared};
    }

    /**
     * 计算移动平均趋势 / Calculate Moving Average Trend
     */
    private IVector<Double> calculateMovingAverageTrend(IVector<Double> data, int window) {
        int n = data.length();
        IVector<Double> trend = Linalg.zeros(n);

        for (int i = 0; i < n; i++) {
            int start = Math.max(0, i - window + 1);
            int end = i + 1;
            double sum = 0;
            int count = 0;

            for (int j = start; j < end; j++) {
                sum += data.get(j);
                count++;
            }

            trend.set(i, sum / count);
        }

        return trend;
    }

    /**
     * 计算趋势强度 / Calculate Trend Strength
     */
    private double calculateTrendStrength(IVector<Double> data) {
        double[] linearTrend = calculateLinearTrend(data);
        return Math.abs(linearTrend[0]); // 使用斜率的绝对值 / Use absolute value of slope
    }

    /**
     * 计算季节性强度 / Calculate Seasonal Strength
     */
    private double calculateSeasonalStrength(IVector<Double> data, int period) {
        int n = data.length();
        int cycles = n / period;

        if (cycles < 2) {
            return 0.0;
        }

        // 计算季节性模式 / Calculate seasonal pattern
        IVector<Double> seasonalPattern = calculateSeasonalPattern(data, period);

        // 计算季节性强度 / Calculate seasonal strength
        double seasonalVar = seasonalPattern.varValue();
        double totalVar = data.varValue();

        return totalVar > 0 ? seasonalVar / totalVar : 0.0;
    }

    /**
     * 计算季节性模式 / Calculate Seasonal Pattern
     */
    private IVector<Double> calculateSeasonalPattern(IVector<Double> data, int period) {
        IVector<Double> pattern = Linalg.zeros(period);

        for (int i = 0; i < period; i++) {
            double sum = 0;
            int count = 0;

            for (int j = i; j < data.length(); j += period) {
                sum += data.get(j);
                count++;
            }

            pattern.set(i, count > 0 ? sum / count : 0.0);
        }

        return pattern;
    }

    /**
     * 检验季节性 / Test Seasonality
     */
    private boolean testSeasonality(IVector<Double> data, int period) {
        double seasonalStrength = calculateSeasonalStrength(data, period);
        return seasonalStrength > 0.1; // 阈值可调整 / Threshold can be adjusted
    }

    /**
     * 移除异常值 / Remove Outliers
     */
    private IVector<Double> removeOutliers(IVector<Double> data) {
        double mean = data.meanValue();
        double std = data.stdValue();
        double threshold = 3.0; // 3-sigma规则 / 3-sigma rule

        List<Double> filtered = new ArrayList<>();
        for (int i = 0; i < data.length(); i++) {
            double value = data.get(i);
            if (Math.abs(value - mean) <= threshold * std) {
                filtered.add(value);
            }
        }

        return Linalg.vector(filtered.stream().mapToDouble(Double::doubleValue).toArray());
    }

    /**
     * 差分 / Differencing
     */
    private IVector<Double> difference(IVector<Double> data) {
        int length = data.length();
        if (length < 2) {
            return Linalg.zeros(0);
        }

        IVector<Double> diff = Linalg.zeros(length - 1);
        for (int i = 1; i < length; i++) {
            diff.set(i - 1, data.get(i) - data.get(i - 1));
        }

        return diff;
    }
}
