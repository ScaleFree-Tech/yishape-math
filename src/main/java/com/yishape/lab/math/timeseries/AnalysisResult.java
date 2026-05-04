package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.timeseries.model.ITimeSeriesDiagnostics;
import com.yishape.lab.math.timeseries.model.ITimeSeriesForecastResult;
import com.yishape.lab.math.timeseries.model.ITimeSeriesModel;

import java.util.Map;

/**
 * 分析结果类 / Analysis Result Class
 * <p>
 * 存储时间序列分析的结果，包括最佳模型、预测结果和诊断信息。
 * 用于TimeSeriesAnalyzer类返回完整的分析报告。
 * </p>
 * <p>
 * Stores the results of time series analysis, including the best model,
 * forecast results, and diagnostic information. Used by TimeSeriesAnalyzer
 * class to return a complete analysis report.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class AnalysisResult {

    /** 最佳模型 / Best model */
    public final ITimeSeriesModel bestModel;
    /** 预测结果 / Forecast result */
    public final ITimeSeriesForecastResult forecast;
    /** 诊断信息 / Diagnostics information */
    public final ITimeSeriesDiagnostics diagnostics;
    /** 分析摘要 / Analysis summary */
    public final String summary;
    /** 元数据 / Metadata */
    public final Map<String, Object> metadata;

    /**
     * 构造函数 / Constructor
     *
     * @param bestModel 最佳模型 / Best model
     * @param forecast 预测结果 / Forecast result
     * @param diagnostics 诊断信息 / Diagnostics information
     * @param summary 分析摘要 / Analysis summary
     * @param metadata 元数据 / Metadata
     */
    public AnalysisResult(ITimeSeriesModel bestModel, ITimeSeriesForecastResult forecast,
            ITimeSeriesDiagnostics diagnostics, String summary, Map<String, Object> metadata) {
        this.bestModel = bestModel;
        this.forecast = forecast;
        this.diagnostics = diagnostics;
        this.summary = summary;
        this.metadata = metadata;
    }
}
