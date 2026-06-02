package com.yishape.lab.math.timeseries;

import com.yishape.lab.util.YishapeLogger;

import com.yishape.lab.math.timeseries.model.ITimeSeriesForecastResult;
import com.yishape.lab.math.timeseries.model.ITimeSeriesDiagnostics;
import com.yishape.lab.math.timeseries.model.TimeSeriesModelFactory;
import com.yishape.lab.math.timeseries.model.ITimeSeriesModel;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

/**
 * 统一时间序列接口使用示例 / Unified Time Series Interface Usage Example
 * <p>
 * 展示如何使用统一的时间序列接口进行时间序列分析。
 * Demonstrates how to use unified time series interface for time series analysis.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class TimeSeriesUnifiedExample {

    private static final YishapeLogger log = YishapeLogger.getLogger(TimeSeriesUnifiedExample.class);

    
    /**
     * 主方法 / Main Method
     */
    public static void main(String[] args) {
        log.debug("=== 统一时间序列接口使用示例 / Unified Time Series Interface Example ===\n");
        
        // 1. 创建示例数据 / Create sample data
        IVector<Double> data = createSampleData();
        log.debug("1. 创建示例数据 / Created sample data");
        log.debug("   数据长度 / Data length: " + data.length());
        log.debug("   数据范围 / Data range: [" + data.min() + ", " + data.max() + "]");
        log.debug("");
        
        // 2. 使用统一分析器 / Use unified analyzer
        log.debug("2. 使用统一分析器 / Using unified analyzer");
        TimeSeriesAnalyzer analyzer = new TimeSeriesAnalyzer(data, "示例时间序列 / Sample Time Series");
        
        // 快速分析 / Quick analysis
        AnalysisResult result = analyzer.quickAnalyze();
        log.debug("   分析完成 / Analysis completed");
        log.debug("   最优模型 / Best model: " + result.bestModel.getModelName());
        log.debug("   模型类型 / Model type: " + result.bestModel.getModelType());
        log.debug("");
        
        // 3. 模型信息 / Model information
        log.debug("3. 模型信息 / Model information");
        log.debug(result.bestModel.getSummary());
        
        // 4. 预测 / Forecasting
        log.debug("4. 预测 / Forecasting");
        ITimeSeriesForecastResult forecast = analyzer.forecast(5, 0.95);
        log.debug("   预测步数 / Forecast steps: " + forecast.getForecastSteps());
        log.debug("   置信水平 / Confidence level: " + forecast.getConfidenceLevel());
        log.debug("   预测值 / Forecast values: " + forecast.getForecastVector());
        log.debug("");
        
        // 5. 诊断 / Diagnostics
        log.debug("5. 诊断 / Diagnostics");
        ITimeSeriesDiagnostics diagnostics = analyzer.diagnose();
        log.debug("   诊断完成 / Diagnostics completed");
        
        // 显示关键检验结果 / Show key test results
        ITimeSeriesDiagnostics.TestResult normalityTest = diagnostics.performNormalityTest();
        log.debug("   正态性检验 / Normality test: " + normalityTest.testName + 
                          " (p值 / p-value: " + String.format("%.4f", normalityTest.pValue) + 
                          ", 结论 / conclusion: " + normalityTest.conclusion + ")");
        
        ITimeSeriesDiagnostics.TestResult autocorrTest = diagnostics.performAutocorrelationTest(5);
        log.debug("   自相关检验 / Autocorrelation test: " + autocorrTest.testName + 
                          " (p值 / p-value: " + String.format("%.4f", autocorrTest.pValue) + 
                          ", 结论 / conclusion: " + autocorrTest.conclusion + ")");
        log.debug("");
        
        // 6. 数据统计 / Data statistics
        log.debug("6. 数据统计 / Data statistics");
        var stats = analyzer.getDataStatistics();
        stats.forEach((key, value) -> 
            log.debug("   " + key + ": " + value));
        log.debug("");
        
        // 7. 趋势分析 / Trend analysis
        log.debug("7. 趋势分析 / Trend analysis");
        var trend = analyzer.getTrendAnalysis();
        log.debug("   趋势斜率 / Trend slope: " + trend.get("slope"));
        log.debug("   趋势强度 / Trend strength: " + trend.get("strength"));
        log.debug("");
        
        // 8. 使用模型工厂 / Using model factory
        log.debug("8. 使用模型工厂 / Using model factory");
        demonstrateModelFactory(data);
        log.debug("");
        
        // 9. 导出报告 / Export report
        log.debug("9. 导出报告 / Export report");
        String report = analyzer.exportReport("TEXT");
        log.debug("   报告长度 / Report length: " + report.length() + " 字符 / characters");
        log.debug("   报告预览 / Report preview:");
        log.debug(report.substring(0, Math.min(500, report.length())) + "...");
        log.debug("");
        
        // 10. 演示预测结果使用 / Demonstrate forecast result usage
        log.debug("10. 演示预测结果使用 / Demonstrate forecast result usage");
        demonstrateForecastResult(forecast);
        log.debug("");
        
        // 11. 演示诊断结果使用 / Demonstrate diagnostics usage
        log.debug("11. 演示诊断结果使用 / Demonstrate diagnostics usage");
        demonstrateDiagnostics(diagnostics);
        log.debug("");
        
        log.debug("=== 示例完成 / Example completed ===");
    }
    
    /**
     * 创建示例数据 / Create Sample Data
     */
    private static IVector<Double> createSampleData() {
        // 创建一个包含趋势和噪声的时间序列 / Create a time series with trend and noise
        int length = 100;
        IVector<Double> data = Linalg.zeros(length);
        
        for (int i = 0; i < length; i++) {
            // 线性趋势 + 季节性 + 噪声 / Linear trend + seasonality + noise
            double trend = 0.1 * i;
            double seasonality = 2.0 * Math.sin(2 * Math.PI * i / 12.0); // 12期季节性 / 12-period seasonality
            double noise = 0.5 * (Math.random() - 0.5);
            data.set(i, trend + seasonality + noise);
        }
        
        return data;
    }
    
    /**
     * 演示模型工厂使用 / Demonstrate Model Factory Usage
     */
    private static void demonstrateModelFactory(IVector<Double> data) {
        log.debug("   使用模型工厂创建ARIMA模型 / Using model factory to create ARIMA model");
        
        // 创建ARIMA模型 / Create ARIMA model
        ITimeSeriesModel arimaModel = TimeSeriesModelFactory.createARIMAModel(data, 1, 0, 1);
        log.debug("   ARIMA模型创建成功 / ARIMA model created successfully");
        log.debug("   模型名称 / Model name: " + arimaModel.getModelName());
        log.debug("   模型有效性 / Model validity: " + arimaModel.isValid());
        
        // 自动选择ARIMA模型 / Auto-select ARIMA model
        log.debug("   自动选择ARIMA模型 / Auto-selecting ARIMA model");
        ITimeSeriesModel bestARIMA = TimeSeriesModelFactory.createARIMAModel(
            data, 3, 2, 3, TimeSeriesModelFactory.SelectionCriterion.AIC);
        log.debug("   最优ARIMA模型 / Best ARIMA model: " + bestARIMA.getModelName());
        
        // 使用配置创建模型 / Create model using configuration
        log.debug("   使用配置创建模型 / Create model using configuration");
        TimeSeriesModelFactory.ModelConfig config = new TimeSeriesModelFactory.ModelConfig.Builder()
            .setModelType(ITimeSeriesModel.ModelType.ARIMA)
            .addParameter("p", 2)
            .addParameter("d", 1)
            .addParameter("q", 1)
            .setName("自定义ARIMA模型 / Custom ARIMA Model")
            .build();
        
        ITimeSeriesModel customModel = TimeSeriesModelFactory.createModel(data, config);
        log.debug("   自定义模型创建成功 / Custom model created successfully");
        log.debug("   模型名称 / Model name: " + customModel.getModelName());
        
        // 模型选择 / Model selection
        log.debug("   模型选择 / Model selection");
        ITimeSeriesModel.ModelType[] candidateTypes = {
            ITimeSeriesModel.ModelType.ARIMA,
            ITimeSeriesModel.ModelType.EXPONENTIAL_SMOOTHING
        };
        
        try {
            ITimeSeriesModel bestModel = TimeSeriesModelFactory.selectBestModel(
                data, candidateTypes, TimeSeriesModelFactory.SelectionCriterion.AIC);
            log.debug("   最优模型 / Best model: " + bestModel.getModelName());
        } catch (Exception e) {
            log.debug("   模型选择失败 / Model selection failed: " + e.getMessage());
        }
    }
    
    /**
     * 演示预测结果使用 / Demonstrate Forecast Result Usage
     */
    private static void demonstrateForecastResult(ITimeSeriesForecastResult forecast) {
        log.debug("   预测结果演示 / Forecast result demonstration");
        log.debug("   预测值 / Forecast values: " + forecast.getForecastVector());
        log.debug("   置信区间下界 / Lower bounds: " + forecast.getLowerBoundsVector());
        log.debug("   置信区间上界 / Upper bounds: " + forecast.getUpperBoundsVector());
        log.debug("   预测标准差 / Standard deviations: " + forecast.getStandardDeviationsVector());
        
        // 导出预测结果 / Export forecast results
        String csvExport = forecast.export("CSV");
        log.debug("   CSV导出长度 / CSV export length: " + csvExport.length());
        
        String jsonExport = forecast.export("JSON");
        log.debug("   JSON导出长度 / JSON export length: " + jsonExport.length());
    }
    
    /**
     * 演示诊断结果使用 / Demonstrate Diagnostics Usage
     */
    private static void demonstrateDiagnostics(ITimeSeriesDiagnostics diagnostics) {
        log.debug("   诊断结果演示 / Diagnostics demonstration");
        
        // 残差统计 / Residual statistics
        double[] resStats = diagnostics.getResidualStatistics();
        log.debug("   残差统计 / Residual statistics:");
        log.debug("     均值 / Mean: " + resStats[0]);
        log.debug("     标准差 / Std Dev: " + resStats[1]);
        log.debug("     偏度 / Skewness: " + resStats[2]);
        log.debug("     峰度 / Kurtosis: " + resStats[3]);
        
        // 拟合优度 / Goodness of fit
        double[] gofMetrics = diagnostics.getGoodnessOfFitMetrics();
        log.debug("   拟合优度 / Goodness of fit:");
        log.debug("     R²: " + gofMetrics[0]);
        log.debug("     调整R² / Adj R²: " + gofMetrics[1]);
        log.debug("     AIC: " + gofMetrics[2]);
        log.debug("     BIC: " + gofMetrics[3]);
        
        // 自相关函数 / Autocorrelation function
        IVector<Double> acf = diagnostics.getAutocorrelationFunction(10);
        log.debug("   自相关函数 / Autocorrelation function: " + acf);
        
        // 导出诊断结果 / Export diagnostics
        String diagnosticsReport = diagnostics.export("TEXT");
        log.debug("   诊断报告长度 / Diagnostics report length: " + diagnosticsReport.length());
    }
}
