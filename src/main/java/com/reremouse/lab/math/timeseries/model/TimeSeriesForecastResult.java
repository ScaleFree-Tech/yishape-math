package com.reremouse.lab.math.timeseries.model;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;

import java.time.LocalDateTime;
import java.util.Arrays;

/**
 * 时间序列预测结果实现类 / Time Series Forecast Result Implementation
 * <p>
 * 实现ITimeSeriesForecastResult接口，提供统一的时间序列预测结果访问方式。
 * 支持单变量和多变量时间序列的预测结果。
 * </p>
 * <p>
 * Implements ITimeSeriesForecastResult interface, providing unified access to
 * time series forecast results. Supports both univariate and multivariate time series forecast results.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class TimeSeriesForecastResult implements ITimeSeriesForecastResult {
    
    private final Object forecast;
    private final Object lowerBounds;
    private final Object upperBounds;
    private final Object standardDeviations;
    private final double confidenceLevel;
    private final int forecastSteps;
    private final int variableCount;
    private final String[] variableNames;
    private final double[] errorMetrics;
    private final ITimeSeriesModel.ModelType modelType;
    private final Object timePoints;
    private final boolean hasConfidenceIntervals;
    private final boolean isMultivariate;
    
    /**
     * 单变量预测结果构造函数 / Univariate Forecast Result Constructor
     *
     * @param forecast 预测值向量 / Forecast values vector
     * @param lowerBounds 置信区间下界向量 / Lower bounds vector
     * @param upperBounds 置信区间上界向量 / Upper bounds vector
     * @param standardDeviations 预测标准差向量 / Standard deviations vector
     * @param confidenceLevel 置信水平 / Confidence level
     * @param modelType 模型类型 / Model type
     * @param timePoints 时间点数组 / Time points array
     * @param errorMetrics 误差指标 / Error metrics
     */
    public TimeSeriesForecastResult(IVector<Double> forecast, IVector<Double> lowerBounds, 
                                  IVector<Double> upperBounds, IVector<Double> standardDeviations,
                                  double confidenceLevel, ITimeSeriesModel.ModelType modelType,
                                  LocalDateTime[] timePoints, double[] errorMetrics) {
        this.forecast = forecast;
        this.lowerBounds = lowerBounds;
        this.upperBounds = upperBounds;
        this.standardDeviations = standardDeviations;
        this.confidenceLevel = confidenceLevel;
        this.forecastSteps = forecast.length();
        this.variableCount = 1;
        this.variableNames = new String[]{"Value"};
        this.errorMetrics = errorMetrics != null ? errorMetrics : new double[4];
        this.modelType = modelType;
        this.timePoints = timePoints;
        this.hasConfidenceIntervals = lowerBounds != null && upperBounds != null;
        this.isMultivariate = false;
    }
    
    /**
     * 多变量预测结果构造函数 / Multivariate Forecast Result Constructor
     *
     * @param forecast 预测值矩阵 / Forecast values matrix
     * @param lowerBounds 置信区间下界矩阵 / Lower bounds matrix
     * @param upperBounds 置信区间上界矩阵 / Upper bounds matrix
     * @param standardDeviations 预测标准差矩阵 / Standard deviations matrix
     * @param confidenceLevel 置信水平 / Confidence level
     * @param modelType 模型类型 / Model type
     * @param variableNames 变量名称 / Variable names
     * @param timePoints 时间点数组 / Time points array
     * @param errorMetrics 误差指标 / Error metrics
     */
    public TimeSeriesForecastResult(IMatrix<Double> forecast, IMatrix<Double> lowerBounds,
                                  IMatrix<Double> upperBounds, IMatrix<Double> standardDeviations,
                                  double confidenceLevel, ITimeSeriesModel.ModelType modelType,
                                  String[] variableNames, LocalDateTime[] timePoints, double[] errorMetrics) {
        this.forecast = forecast;
        this.lowerBounds = lowerBounds;
        this.upperBounds = upperBounds;
        this.standardDeviations = standardDeviations;
        this.confidenceLevel = confidenceLevel;
        this.forecastSteps = forecast.getRowNum();
        this.variableCount = forecast.getColNum();
        this.variableNames = variableNames != null ? variableNames : generateDefaultVariableNames(variableCount);
        this.errorMetrics = errorMetrics != null ? errorMetrics : new double[4];
        this.modelType = modelType;
        this.timePoints = timePoints;
        this.hasConfidenceIntervals = lowerBounds != null && upperBounds != null;
        this.isMultivariate = true;
    }
    
    @Override
    public Object getForecast() {
        return forecast;
    }
    
    @Override
    public IVector<Double> getForecastVector() throws UnsupportedOperationException {
        if (isMultivariate) {
            throw new UnsupportedOperationException("多变量时间序列不支持向量访问");
        }
        return (IVector<Double>) forecast;
    }
    
    @Override
    public IMatrix<Double> getForecastMatrix() throws UnsupportedOperationException {
        if (!isMultivariate) {
            throw new UnsupportedOperationException("单变量时间序列不支持矩阵访问");
        }
        return (IMatrix<Double>) forecast;
    }
    
    @Override
    public Object getLowerBounds() {
        return lowerBounds;
    }
    
    @Override
    public IVector<Double> getLowerBoundsVector() throws UnsupportedOperationException {
        if (isMultivariate) {
            throw new UnsupportedOperationException("多变量时间序列不支持向量访问");
        }
        return (IVector<Double>) lowerBounds;
    }
    
    @Override
    public IMatrix<Double> getLowerBoundsMatrix() throws UnsupportedOperationException {
        if (!isMultivariate) {
            throw new UnsupportedOperationException("单变量时间序列不支持矩阵访问");
        }
        return (IMatrix<Double>) lowerBounds;
    }
    
    @Override
    public Object getUpperBounds() {
        return upperBounds;
    }
    
    @Override
    public IVector<Double> getUpperBoundsVector() throws UnsupportedOperationException {
        if (isMultivariate) {
            throw new UnsupportedOperationException("多变量时间序列不支持向量访问");
        }
        return (IVector<Double>) upperBounds;
    }
    
    @Override
    public IMatrix<Double> getUpperBoundsMatrix() throws UnsupportedOperationException {
        if (!isMultivariate) {
            throw new UnsupportedOperationException("单变量时间序列不支持矩阵访问");
        }
        return (IMatrix<Double>) upperBounds;
    }
    
    @Override
    public Object getStandardDeviations() {
        return standardDeviations;
    }
    
    @Override
    public IVector<Double> getStandardDeviationsVector() throws UnsupportedOperationException {
        if (isMultivariate) {
            throw new UnsupportedOperationException("多变量时间序列不支持向量访问");
        }
        return (IVector<Double>) standardDeviations;
    }
    
    @Override
    public IMatrix<Double> getStandardDeviationsMatrix() throws UnsupportedOperationException {
        if (!isMultivariate) {
            throw new UnsupportedOperationException("单变量时间序列不支持矩阵访问");
        }
        return (IMatrix<Double>) standardDeviations;
    }
    
    @Override
    public double getConfidenceLevel() {
        return confidenceLevel;
    }
    
    @Override
    public int getForecastSteps() {
        return forecastSteps;
    }
    
    @Override
    public int getVariableCount() {
        return variableCount;
    }
    
    @Override
    public String[] getVariableNames() {
        return variableNames.clone();
    }
    
    @Override
    public double[] getErrorMetrics() {
        return errorMetrics.clone();
    }
    
    @Override
    public ITimeSeriesModel.ModelType getModelType() {
        return modelType;
    }
    
    @Override
    public Object getTimePoints() {
        return timePoints;
    }
    
    @Override
    public boolean hasConfidenceIntervals() {
        return hasConfidenceIntervals;
    }
    
    @Override
    public boolean isMultivariate() {
        return isMultivariate;
    }
    
    @Override
    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("时间序列预测结果摘要 / Time Series Forecast Summary\n");
        sb.append("=====================================\n");
        sb.append(String.format("模型类型 / Model Type: %s\n", modelType));
        sb.append(String.format("预测步数 / Forecast Steps: %d\n", forecastSteps));
        sb.append(String.format("变量数量 / Variable Count: %d\n", variableCount));
        sb.append(String.format("置信水平 / Confidence Level: %.2f%%\n", confidenceLevel * 100));
        sb.append(String.format("是否多变量 / Multivariate: %s\n", isMultivariate ? "是 / Yes" : "否 / No"));
        sb.append(String.format("是否有置信区间 / Has Confidence Intervals: %s\n", 
                               hasConfidenceIntervals ? "是 / Yes" : "否 / No"));
        
        if (errorMetrics.length >= 4) {
            sb.append("\n误差指标 / Error Metrics:\n");
            sb.append(String.format("  MSE: %.6f\n", errorMetrics[0]));
            sb.append(String.format("  MAE: %.6f\n", errorMetrics[1]));
            sb.append(String.format("  MAPE: %.2f%%\n", errorMetrics[2]));
            sb.append(String.format("  RMSE: %.6f\n", errorMetrics[3]));
        }
        
        if (variableNames.length > 0) {
            sb.append("\n变量名称 / Variable Names: ");
            sb.append(Arrays.toString(variableNames));
            sb.append("\n");
        }
        
        return sb.toString();
    }
    
    @Override
    public String export(String format) {
        if (format == null || format.isEmpty()) {
            format = "CSV";
        }
        
        ExportFormat exportFormat;
        try {
            exportFormat = ExportFormat.valueOf(format.toUpperCase());
        } catch (IllegalArgumentException e) {
            exportFormat = ExportFormat.CSV;
        }
        
        switch (exportFormat) {
            case CSV:
                return exportToCSV();
            case JSON:
                return exportToJSON();
            case XML:
                return exportToXML();
            case TABLE:
                return exportToTable();
            default:
                return exportToCSV();
        }
    }
    
    /**
     * 导出为CSV格式 / Export to CSV Format
     */
    private String exportToCSV() {
        StringBuilder sb = new StringBuilder();
        
        // 表头 / Header
        sb.append("Step,Time");
        if (isMultivariate) {
            for (String varName : variableNames) {
                sb.append(",Forecast_").append(varName);
                if (hasConfidenceIntervals) {
                    sb.append(",Lower_").append(varName).append(",Upper_").append(varName);
                }
            }
        } else {
            sb.append(",Forecast");
            if (hasConfidenceIntervals) {
                sb.append(",Lower_Bound,Upper_Bound");
            }
        }
        sb.append("\n");
        
        // 数据行 / Data rows
        LocalDateTime[] timePoints = (LocalDateTime[]) this.timePoints;
        for (int i = 0; i < forecastSteps; i++) {
            sb.append(i + 1).append(",");
            if (timePoints != null && i < timePoints.length) {
                sb.append(timePoints[i]);
            } else {
                sb.append("T+").append(i + 1);
            }
            
            if (isMultivariate) {
                IMatrix<Double> forecastMatrix = (IMatrix<Double>) forecast;
                IMatrix<Double> lowerMatrix = hasConfidenceIntervals ? (IMatrix<Double>) lowerBounds : null;
                IMatrix<Double> upperMatrix = hasConfidenceIntervals ? (IMatrix<Double>) upperBounds : null;
                
                for (int j = 0; j < variableCount; j++) {
                    sb.append(",").append(forecastMatrix.get(i, j));
                    if (hasConfidenceIntervals) {
                        sb.append(",").append(lowerMatrix.get(i, j));
                        sb.append(",").append(upperMatrix.get(i, j));
                    }
                }
            } else {
                IVector<Double> forecastVector = (IVector<Double>) forecast;
                sb.append(",").append(forecastVector.get(i));
                if (hasConfidenceIntervals) {
                    IVector<Double> lowerVector = (IVector<Double>) lowerBounds;
                    IVector<Double> upperVector = (IVector<Double>) upperBounds;
                    sb.append(",").append(lowerVector.get(i));
                    sb.append(",").append(upperVector.get(i));
                }
            }
            sb.append("\n");
        }
        
        return sb.toString();
    }
    
    /**
     * 导出为JSON格式 / Export to JSON Format
     */
    private String exportToJSON() {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"modelType\": \"").append(modelType).append("\",\n");
        sb.append("  \"forecastSteps\": ").append(forecastSteps).append(",\n");
        sb.append("  \"variableCount\": ").append(variableCount).append(",\n");
        sb.append("  \"confidenceLevel\": ").append(confidenceLevel).append(",\n");
        sb.append("  \"isMultivariate\": ").append(isMultivariate).append(",\n");
        sb.append("  \"hasConfidenceIntervals\": ").append(hasConfidenceIntervals).append(",\n");
        
        if (variableNames.length > 0) {
            sb.append("  \"variableNames\": [");
            for (int i = 0; i < variableNames.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append("\"").append(variableNames[i]).append("\"");
            }
            sb.append("],\n");
        }
        
        if (errorMetrics.length >= 4) {
            sb.append("  \"errorMetrics\": {\n");
            sb.append("    \"MSE\": ").append(errorMetrics[0]).append(",\n");
            sb.append("    \"MAE\": ").append(errorMetrics[1]).append(",\n");
            sb.append("    \"MAPE\": ").append(errorMetrics[2]).append(",\n");
            sb.append("    \"RMSE\": ").append(errorMetrics[3]).append("\n");
            sb.append("  },\n");
        }
        
        sb.append("  \"forecastData\": [\n");
        // 这里可以添加具体的预测数据
        sb.append("  ]\n");
        sb.append("}\n");
        
        return sb.toString();
    }
    
    /**
     * 导出为XML格式 / Export to XML Format
     */
    private String exportToXML() {
        StringBuilder sb = new StringBuilder();
        sb.append("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        sb.append("<TimeSeriesForecast>\n");
        sb.append("  <ModelType>").append(modelType).append("</ModelType>\n");
        sb.append("  <ForecastSteps>").append(forecastSteps).append("</ForecastSteps>\n");
        sb.append("  <VariableCount>").append(variableCount).append("</VariableCount>\n");
        sb.append("  <ConfidenceLevel>").append(confidenceLevel).append("</ConfidenceLevel>\n");
        sb.append("  <IsMultivariate>").append(isMultivariate).append("</IsMultivariate>\n");
        sb.append("  <HasConfidenceIntervals>").append(hasConfidenceIntervals).append("</HasConfidenceIntervals>\n");
        sb.append("</TimeSeriesForecast>\n");
        return sb.toString();
    }
    
    /**
     * 导出为表格格式 / Export to Table Format
     */
    private String exportToTable() {
        return getSummary();
    }
    
    /**
     * 生成默认变量名称 / Generate Default Variable Names
     */
    private String[] generateDefaultVariableNames(int count) {
        String[] names = new String[count];
        for (int i = 0; i < count; i++) {
            names[i] = "Var" + (i + 1);
        }
        return names;
    }
}
