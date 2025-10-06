package com.yishape.lab.math.timeseries.model;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;

/**
 * 时间序列预测结果统一接口 / Unified Time Series Forecast Result Interface
 * <p>
 * 定义时间序列预测结果的标准接口，提供统一的预测结果访问方式。
 * 支持单变量和多变量时间序列的预测结果。
 * </p>
 * <p>
 * Defines the standard interface for time series forecast results, providing
 * unified access to forecast results. Supports both univariate and multivariate
 * time series forecast results.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface ITimeSeriesForecastResult {
    
    /**
     * 获取预测值 / Get Forecast Values
     * <p>
     * 对于单变量时间序列，返回一维向量；对于多变量时间序列，返回矩阵。
     * For univariate time series, returns a 1D vector; for multivariate time series, returns a matrix.
     * </p>
     *
     * @return 预测值 / Forecast values
     */
    Object getForecast();
    
    /**
     * 获取单变量预测值 / Get Univariate Forecast Values
     * <p>
     * 获取单变量时间序列的预测值向量。
     * Get forecast values vector for univariate time series.
     * </p>
     *
     * @return 预测值向量 / Forecast values vector
     * @throws UnsupportedOperationException 如果是多变量时间序列 / If multivariate time series
     */
    IVector<Double> getForecastVector() throws UnsupportedOperationException;
    
    /**
     * 获取多变量预测值 / Get Multivariate Forecast Values
     * <p>
     * 获取多变量时间序列的预测值矩阵。
     * Get forecast values matrix for multivariate time series.
     * </p>
     *
     * @return 预测值矩阵 / Forecast values matrix
     * @throws UnsupportedOperationException 如果是单变量时间序列 / If univariate time series
     */
    IMatrix<Double> getForecastMatrix() throws UnsupportedOperationException;
    
    /**
     * 获取置信区间下界 / Get Confidence Interval Lower Bounds
     * <p>
     * 获取预测值的置信区间下界。
     * Get lower bounds of confidence intervals for forecast values.
     * </p>
     *
     * @return 置信区间下界 / Confidence interval lower bounds
     */
    Object getLowerBounds();
    
    /**
     * 获取单变量置信区间下界 / Get Univariate Confidence Interval Lower Bounds
     *
     * @return 置信区间下界向量 / Lower bounds vector
     * @throws UnsupportedOperationException 如果是多变量时间序列 / If multivariate time series
     */
    IVector<Double> getLowerBoundsVector() throws UnsupportedOperationException;
    
    /**
     * 获取多变量置信区间下界 / Get Multivariate Confidence Interval Lower Bounds
     *
     * @return 置信区间下界矩阵 / Lower bounds matrix
     * @throws UnsupportedOperationException 如果是单变量时间序列 / If univariate time series
     */
    IMatrix<Double> getLowerBoundsMatrix() throws UnsupportedOperationException;
    
    /**
     * 获取置信区间上界 / Get Confidence Interval Upper Bounds
     * <p>
     * 获取预测值的置信区间上界。
     * Get upper bounds of confidence intervals for forecast values.
     * </p>
     *
     * @return 置信区间上界 / Confidence interval upper bounds
     */
    Object getUpperBounds();
    
    /**
     * 获取单变量置信区间上界 / Get Univariate Confidence Interval Upper Bounds
     *
     * @return 置信区间上界向量 / Upper bounds vector
     * @throws UnsupportedOperationException 如果是多变量时间序列 / If multivariate time series
     */
    IVector<Double> getUpperBoundsVector() throws UnsupportedOperationException;
    
    /**
     * 获取多变量置信区间上界 / Get Multivariate Confidence Interval Upper Bounds
     *
     * @return 置信区间上界矩阵 / Upper bounds matrix
     * @throws UnsupportedOperationException 如果是单变量时间序列 / If univariate time series
     */
    IMatrix<Double> getUpperBoundsMatrix() throws UnsupportedOperationException;
    
    /**
     * 获取预测标准差 / Get Forecast Standard Deviations
     * <p>
     * 获取预测值的标准差。
     * Get standard deviations of forecast values.
     * </p>
     *
     * @return 预测标准差 / Forecast standard deviations
     */
    Object getStandardDeviations();
    
    /**
     * 获取单变量预测标准差 / Get Univariate Forecast Standard Deviations
     *
     * @return 预测标准差向量 / Standard deviations vector
     * @throws UnsupportedOperationException 如果是多变量时间序列 / If multivariate time series
     */
    IVector<Double> getStandardDeviationsVector() throws UnsupportedOperationException;
    
    /**
     * 获取多变量预测标准差 / Get Multivariate Forecast Standard Deviations
     *
     * @return 预测标准差矩阵 / Standard deviations matrix
     * @throws UnsupportedOperationException 如果是单变量时间序列 / If univariate time series
     */
    IMatrix<Double> getStandardDeviationsMatrix() throws UnsupportedOperationException;
    
    /**
     * 获取置信水平 / Get Confidence Level
     *
     * @return 置信水平 / Confidence level
     */
    double getConfidenceLevel();
    
    /**
     * 获取预测步数 / Get Number of Forecast Steps
     *
     * @return 预测步数 / Number of forecast steps
     */
    int getForecastSteps();
    
    /**
     * 获取变量数量 / Get Number of Variables
     *
     * @return 变量数量 / Number of variables
     */
    int getVariableCount();
    
    /**
     * 获取变量名称 / Get Variable Names
     *
     * @return 变量名称数组 / Variable names array
     */
    String[] getVariableNames();
    
    /**
     * 获取预测误差指标 / Get Forecast Error Metrics
     * <p>
     * 获取预测的误差指标，包括MSE、MAE、MAPE等。
     * Get forecast error metrics including MSE, MAE, MAPE, etc.
     * </p>
     *
     * @return 误差指标数组 [MSE, MAE, MAPE, RMSE] / Error metrics array [MSE, MAE, MAPE, RMSE]
     */
    double[] getErrorMetrics();
    
    /**
     * 获取模型类型 / Get Model Type
     *
     * @return 模型类型 / Model type
     */
    ITimeSeriesModel.ModelType getModelType();
    
    /**
     * 获取预测时间点 / Get Forecast Time Points
     * <p>
     * 获取预测对应的时间点。
     * Get time points corresponding to forecasts.
     * </p>
     *
     * @return 时间点数组 / Time points array
     */
    Object getTimePoints();
    
    /**
     * 检查是否有置信区间 / Check if Has Confidence Intervals
     *
     * @return 是否有置信区间 / Whether has confidence intervals
     */
    boolean hasConfidenceIntervals();
    
    /**
     * 检查是否为多变量 / Check if Multivariate
     *
     * @return 是否为多变量 / Whether multivariate
     */
    boolean isMultivariate();
    
    /**
     * 获取预测摘要 / Get Forecast Summary
     * <p>
     * 获取预测结果的文本摘要。
     * Get text summary of forecast results.
     * </p>
     *
     * @return 预测摘要 / Forecast summary
     */
    String getSummary();
    
    /**
     * 导出预测结果 / Export Forecast Results
     * <p>
     * 将预测结果导出为指定格式。
     * Export forecast results in specified format.
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
        TABLE       // 表格格式 / Table format
    }
}
