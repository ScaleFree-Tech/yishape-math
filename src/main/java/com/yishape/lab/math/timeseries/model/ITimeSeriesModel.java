package com.yishape.lab.math.timeseries.model;

import com.yishape.lab.math.linalg.IVector;

/**
 * 时间序列模型统一接口 / Unified Time Series Model Interface
 * <p>
 * 定义所有时间序列模型必须实现的标准接口，包括单变量和多变量时间序列模型。
 * 提供统一的模型拟合、预测、诊断等功能。
 * </p>
 * <p>
 * Defines the standard interface that all time series models must implement,
 * including univariate and multivariate time series models. Provides unified
 * model fitting, forecasting, and diagnostics functionality.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface ITimeSeriesModel {
    
    /**
     * 模型类型枚举 / Model Type Enum
     */
    enum ModelType {
        ARIMA,              // ARIMA模型 / ARIMA Model
        EXPONENTIAL_SMOOTHING, // 指数平滑模型 / Exponential Smoothing Model
        GARCH,              // GARCH模型 / GARCH Model
        STATE_SPACE,        // 状态空间模型 / State Space Model
        VAR,                // 向量自回归模型 / Vector Autoregression Model
        HOLT_WINTERS,       // Holt-Winters模型 / Holt-Winters Model
        SIMPLE_MA,          // 简单移动平均 / Simple Moving Average
        LINEAR_TREND        // 线性趋势模型 / Linear Trend Model
    }
    
    /**
     * 模型状态枚举 / Model State Enum
     */
    enum ModelState {
        UNFITTED,           // 未拟合 / Unfitted
        FITTED,             // 已拟合 / Fitted
        INVALID             // 无效 / Invalid
    }
    
    /**
     * 获取模型类型 / Get Model Type
     *
     * @return 模型类型 / Model type
     */
    ModelType getModelType();
    
    /**
     * 获取模型状态 / Get Model State
     *
     * @return 模型状态 / Model state
     */
    ModelState getModelState();
    
    /**
     * 获取模型名称 / Get Model Name
     *
     * @return 模型名称 / Model name
     */
    String getModelName();
    
    /**
     * 获取模型参数数量 / Get Number of Parameters
     *
     * @return 参数数量 / Number of parameters
     */
    int getParameterCount();
    
    /**
     * 获取模型信息准则 / Get Model Information Criteria
     *
     * @return 信息准则数组 [AIC, BIC, LogLikelihood] / Information criteria array [AIC, BIC, LogLikelihood]
     */
    double[] getInformationCriteria();
    
    /**
     * 单步预测 / Single Step Forecast
     * <p>
     * 预测下一个时间点的值。
     * Forecast the value at the next time point.
     * </p>
     *
     * @return 预测值 / Forecast value
     * @throws IllegalStateException 如果模型未拟合 / If model is not fitted
     */
    double forecastOneStep() throws IllegalStateException;
    
    /**
     * 多步预测 / Multi-Step Forecast
     * <p>
     * 预测未来多个时间点的值。
     * Forecast values at multiple future time points.
     * </p>
     *
     * @param steps 预测步数 / Number of forecast steps
     * @return 预测值向量 / Forecast values vector
     * @throws IllegalStateException 如果模型未拟合 / If model is not fitted
     * @throws IllegalArgumentException 如果步数无效 / If steps is invalid
     */
    IVector<Double> forecast(int steps) throws IllegalStateException, IllegalArgumentException;
    
    /**
     * 带置信区间的预测 / Forecast with Confidence Intervals
     * <p>
     * 预测未来值并计算置信区间。
     * Forecast future values and calculate confidence intervals.
     * </p>
     *
     * @param steps 预测步数 / Number of forecast steps
     * @param confidenceLevel 置信水平 / Confidence level (0.0-1.0)
     * @return 预测结果 / Forecast result
     * @throws IllegalStateException 如果模型未拟合 / If model is not fitted
     * @throws IllegalArgumentException 如果参数无效 / If parameters are invalid
     */
    ITimeSeriesForecastResult forecastWithConfidence(int steps, double confidenceLevel) 
            throws IllegalStateException, IllegalArgumentException;
    
    /**
     * 模型诊断 / Model Diagnostics
     * <p>
     * 进行模型诊断，包括残差分析、拟合优度检验等。
     * Perform model diagnostics including residual analysis, goodness of fit tests, etc.
     * </p>
     *
     * @return 诊断结果 / Diagnostic results
     * @throws IllegalStateException 如果模型未拟合 / If model is not fitted
     */
    ITimeSeriesDiagnostics diagnose() throws IllegalStateException;
    
    /**
     * 获取残差 / Get Residuals
     * <p>
     * 获取模型的残差序列。
     * Get the residual sequence of the model.
     * </p>
     *
     * @return 残差向量 / Residual vector
     * @throws IllegalStateException 如果模型未拟合 / If model is not fitted
     */
    IVector<Double> getResiduals() throws IllegalStateException;
    
    /**
     * 获取拟合值 / Get Fitted Values
     * <p>
     * 获取模型的拟合值序列。
     * Get the fitted values sequence of the model.
     * </p>
     *
     * @return 拟合值向量 / Fitted values vector
     * @throws IllegalStateException 如果模型未拟合 / If model is not fitted
     */
    IVector<Double> getFittedValues() throws IllegalStateException;
    
    /**
     * 计算预测误差 / Calculate Forecast Error
     * <p>
     * 计算预测误差指标（MSE, MAE, MAPE等）。
     * Calculate forecast error metrics (MSE, MAE, MAPE, etc.).
     * </p>
     *
     * @param actual 实际值 / Actual values
     * @param forecast 预测值 / Forecast values
     * @return 误差指标数组 [MSE, MAE, MAPE, RMSE] / Error metrics array [MSE, MAE, MAPE, RMSE]
     */
    double[] calculateForecastError(IVector<Double> actual, IVector<Double> forecast);
    
    /**
     * 检查模型有效性 / Check Model Validity
     * <p>
     * 检查模型是否有效，包括参数合理性、收敛性等。
     * Check if the model is valid, including parameter reasonableness, convergence, etc.
     * </p>
     *
     * @return 是否有效 / Whether valid
     */
    boolean isValid();
    
    /**
     * 获取模型摘要 / Get Model Summary
     * <p>
     * 获取模型的详细摘要信息。
     * Get detailed summary information of the model.
     * </p>
     *
     * @return 模型摘要字符串 / Model summary string
     */
    String getSummary();
    
    /**
     * 重置模型 / Reset Model
     * <p>
     * 重置模型到初始状态。
     * Reset the model to initial state.
     * </p>
     */
    void reset();
    
    /**
     * 克隆模型 / Clone Model
     * <p>
     * 创建模型的深拷贝。
     * Create a deep copy of the model.
     * </p>
     *
     * @return 模型副本 / Model copy
     */
    ITimeSeriesModel clone();
}
