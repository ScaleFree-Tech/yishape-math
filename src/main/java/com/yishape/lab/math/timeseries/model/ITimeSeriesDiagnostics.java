package com.yishape.lab.math.timeseries.model;

import com.yishape.lab.math.linalg.IVector;

/**
 * 时间序列模型诊断统一接口 / Unified Time Series Model Diagnostics Interface
 * <p>
 * 定义时间序列模型诊断结果的标准接口，提供统一的诊断信息访问方式。
 * 包括残差分析、拟合优度检验、模型假设检验等。
 * </p>
 * <p>
 * Defines the standard interface for time series model diagnostics results,
 * providing unified access to diagnostic information. Includes residual analysis,
 * goodness of fit tests, model assumption tests, etc.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface ITimeSeriesDiagnostics {
    
    /**
     * 诊断类型枚举 / Diagnostic Type Enum
     */
    enum DiagnosticType {
        RESIDUAL_ANALYSIS,      // 残差分析 / Residual Analysis
        NORMALITY_TEST,         // 正态性检验 / Normality Test
        AUTOCORRELATION_TEST,   // 自相关检验 / Autocorrelation Test
        HETEROSCEDASTICITY_TEST, // 异方差检验 / Heteroscedasticity Test
        STATIONARITY_TEST,      // 平稳性检验 / Stationarity Test
        COINTEGRATION_TEST,     // 协整检验 / Cointegration Test
        ARCH_EFFECT_TEST,       // ARCH效应检验 / ARCH Effect Test
        LJUNG_BOX_TEST,         // Ljung-Box检验 / Ljung-Box Test
        SHAPIRO_WILK_TEST,      // Shapiro-Wilk检验 / Shapiro-Wilk Test
        ADF_TEST,               // ADF检验 / ADF Test
        KPSS_TEST,              // KPSS检验 / KPSS Test
        JOHANSEN_TEST           // Johansen检验 / Johansen Test
    }
    
    /**
     * 检验结果类 / Test Result Class
     */
    class TestResult {
        public final String testName;
        public final double statistic;
        public final double pValue;
        public final boolean isSignificant;
        public final String conclusion;
        
        public TestResult(String testName, double statistic, double pValue, 
                         boolean isSignificant, String conclusion) {
            this.testName = testName;
            this.statistic = statistic;
            this.pValue = pValue;
            this.isSignificant = isSignificant;
            this.conclusion = conclusion;
        }
    }
    
    /**
     * 获取残差 / Get Residuals
     * <p>
     * 获取模型的残差序列。
     * Get the residual sequence of the model.
     * </p>
     *
     * @return 残差向量 / Residual vector
     */
    IVector<Double> getResiduals();
    
    /**
     * 获取标准化残差 / Get Standardized Residuals
     * <p>
     * 获取标准化后的残差序列。
     * Get standardized residual sequence.
     * </p>
     *
     * @return 标准化残差向量 / Standardized residual vector
     */
    IVector<Double> getStandardizedResiduals();
    
    /**
     * 获取残差统计信息 / Get Residual Statistics
     * <p>
     * 获取残差的基本统计信息。
     * Get basic statistics of residuals.
     * </p>
     *
     * @return 残差统计信息 [均值, 标准差, 偏度, 峰度] / Residual statistics [mean, std, skewness, kurtosis]
     */
    double[] getResidualStatistics();
    
    /**
     * 获取拟合值 / Get Fitted Values
     * <p>
     * 获取模型的拟合值序列。
     * Get the fitted values sequence of the model.
     * </p>
     *
     * @return 拟合值向量 / Fitted values vector
     */
    IVector<Double> getFittedValues();
    
    /**
     * 获取拟合优度指标 / Get Goodness of Fit Metrics
     * <p>
     * 获取模型的拟合优度指标。
     * Get goodness of fit metrics of the model.
     * </p>
     *
     * @return 拟合优度指标 [R², 调整R², AIC, BIC, 对数似然] / Goodness of fit metrics [R², Adj R², AIC, BIC, LogLik]
     */
    double[] getGoodnessOfFitMetrics();
    
    /**
     * 执行正态性检验 / Perform Normality Test
     * <p>
     * 对残差进行正态性检验。
     * Perform normality test on residuals.
     * </p>
     *
     * @return 正态性检验结果 / Normality test result
     */
    TestResult performNormalityTest();
    
    /**
     * 执行自相关检验 / Perform Autocorrelation Test
     * <p>
     * 对残差进行自相关检验。
     * Perform autocorrelation test on residuals.
     * </p>
     *
     * @param maxLag 最大滞后阶数 / Maximum lag order
     * @return 自相关检验结果 / Autocorrelation test result
     */
    TestResult performAutocorrelationTest(int maxLag);
    
    /**
     * 执行Ljung-Box检验 / Perform Ljung-Box Test
     * <p>
     * 对残差进行Ljung-Box检验。
     * Perform Ljung-Box test on residuals.
     * </p>
     *
     * @param maxLag 最大滞后阶数 / Maximum lag order
     * @return Ljung-Box检验结果 / Ljung-Box test result
     */
    TestResult performLjungBoxTest(int maxLag);
    
    /**
     * 执行异方差检验 / Perform Heteroscedasticity Test
     * <p>
     * 对残差进行异方差检验。
     * Perform heteroscedasticity test on residuals.
     * </p>
     *
     * @return 异方差检验结果 / Heteroscedasticity test result
     */
    TestResult performHeteroscedasticityTest();
    
    /**
     * 执行ARCH效应检验 / Perform ARCH Effect Test
     * <p>
     * 对残差进行ARCH效应检验。
     * Perform ARCH effect test on residuals.
     * </p>
     *
     * @param maxLag 最大滞后阶数 / Maximum lag order
     * @return ARCH效应检验结果 / ARCH effect test result
     */
    TestResult performARCHEffectTest(int maxLag);
    
    /**
     * 执行平稳性检验 / Perform Stationarity Test
     * <p>
     * 对原始数据进行平稳性检验。
     * Perform stationarity test on original data.
     * </p>
     *
     * @return 平稳性检验结果 / Stationarity test result
     */
    TestResult performStationarityTest();
    
    /**
     * 获取自相关函数 / Get Autocorrelation Function
     * <p>
     * 获取残差的自相关函数值。
     * Get autocorrelation function values of residuals.
     * </p>
     *
     * @param maxLag 最大滞后阶数 / Maximum lag order
     * @return 自相关函数值向量 / Autocorrelation function values vector
     */
    IVector<Double> getAutocorrelationFunction(int maxLag);
    
    /**
     * 获取偏自相关函数 / Get Partial Autocorrelation Function
     * <p>
     * 获取残差的偏自相关函数值。
     * Get partial autocorrelation function values of residuals.
     * </p>
     *
     * @param maxLag 最大滞后阶数 / Maximum lag order
     * @return 偏自相关函数值向量 / Partial autocorrelation function values vector
     */
    IVector<Double> getPartialAutocorrelationFunction(int maxLag);
    
    /**
     * 获取Q-Q图数据 / Get Q-Q Plot Data
     * <p>
     * 获取用于绘制Q-Q图的数据。
     * Get data for plotting Q-Q plot.
     * </p>
     *
     * @return Q-Q图数据 [理论分位数, 样本分位数] / Q-Q plot data [theoretical quantiles, sample quantiles]
     */
    double[][] getQQPlotData();
    
    /**
     * 获取残差图数据 / Get Residual Plot Data
     * <p>
     * 获取用于绘制残差图的数据。
     * Get data for plotting residual plots.
     * </p>
     *
     * @return 残差图数据 [时间, 残差, 拟合值] / Residual plot data [time, residuals, fitted values]
     */
    double[][] getResidualPlotData();
    
    /**
     * 获取所有检验结果 / Get All Test Results
     * <p>
     * 获取所有已执行的检验结果。
     * Get all executed test results.
     * </p>
     *
     * @return 检验结果映射 / Test results map
     */
    java.util.Map<DiagnosticType, TestResult> getAllTestResults();
    
    /**
     * 检查模型假设 / Check Model Assumptions
     * <p>
     * 检查模型的基本假设是否满足。
     * Check if basic model assumptions are satisfied.
     * </p>
     *
     * @return 假设检验结果映射 / Assumption test results map
     */
    java.util.Map<String, Boolean> checkModelAssumptions();
    
    /**
     * 获取诊断摘要 / Get Diagnostics Summary
     * <p>
     * 获取模型诊断的文本摘要。
     * Get text summary of model diagnostics.
     * </p>
     *
     * @return 诊断摘要 / Diagnostics summary
     */
    String getSummary();
    
    /**
     * 获取诊断报告 / Get Diagnostics Report
     * <p>
     * 获取详细的诊断报告。
     * Get detailed diagnostics report.
     * </p>
     *
     * @return 诊断报告 / Diagnostics report
     */
    String getReport();
    
    /**
     * 导出诊断结果 / Export Diagnostics Results
     * <p>
     * 将诊断结果导出为指定格式。
     * Export diagnostics results in specified format.
     * </p>
     *
     * @param format 导出格式 / Export format
     * @return 导出的字符串 / Exported string
     */
    String export(String format);
    
    /**
     * 导出格式枚举 / Export Format Enum
     */
    enum ExportFormat {
        CSV,        // CSV格式 / CSV format
        JSON,       // JSON格式 / JSON format
        XML,        // XML格式 / XML format
        HTML,       // HTML格式 / HTML format
        TEXT        // 文本格式 / Text format
    }
}
