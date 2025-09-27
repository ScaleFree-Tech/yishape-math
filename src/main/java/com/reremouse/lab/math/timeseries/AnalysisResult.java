package com.reremouse.lab.math.timeseries;

import com.reremouse.lab.math.timeseries.model.ITimeSeriesDiagnostics;
import com.reremouse.lab.math.timeseries.model.ITimeSeriesForecastResult;
import com.reremouse.lab.math.timeseries.model.ITimeSeriesModel;
import java.util.Map;

/**
 *
 * @author lteb2
 */
/**
 * 分析结果类 / Analysis Result Class
 */
public class AnalysisResult {

    public final ITimeSeriesModel bestModel;
    public final ITimeSeriesForecastResult forecast;
    public final ITimeSeriesDiagnostics diagnostics;
    public final String summary;
    public final Map<String, Object> metadata;

    public AnalysisResult(ITimeSeriesModel bestModel, ITimeSeriesForecastResult forecast,
            ITimeSeriesDiagnostics diagnostics, String summary, Map<String, Object> metadata) {
        this.bestModel = bestModel;
        this.forecast = forecast;
        this.diagnostics = diagnostics;
        this.summary = summary;
        this.metadata = metadata;
    }
}
