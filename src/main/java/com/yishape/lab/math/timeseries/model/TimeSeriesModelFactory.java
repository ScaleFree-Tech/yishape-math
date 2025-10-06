package com.yishape.lab.math.timeseries.model;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;

import java.util.HashMap;
import java.util.Map;

/**
 * 时间序列模型工厂类 / Time Series Model Factory Class
 * <p>
 * 提供统一的时间序列模型创建和管理功能。
 * 支持各种时间序列模型的创建、配置和生命周期管理。
 * </p>
 * <p>
 * Provides unified time series model creation and management functionality.
 * Supports creation, configuration, and lifecycle management of various time series models.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class TimeSeriesModelFactory {
    
    private static final Map<ITimeSeriesModel.ModelType, Class<? extends ITimeSeriesModel>> modelClasses = new HashMap<>();
    
    static {
        // 注册模型类型 / Register model types
        modelClasses.put(ITimeSeriesModel.ModelType.ARIMA, UnifiedARIMAModel.class);
        // 其他模型类型将在后续实现中注册 / Other model types will be registered in subsequent implementations
    }
    
    /**
     * 模型配置类 / Model Configuration Class
     */
    public static class ModelConfig {
        private final ITimeSeriesModel.ModelType modelType;
        private final Map<String, Object> parameters;
        private final String name;
        private final boolean autoFit;
        
        public ModelConfig(ITimeSeriesModel.ModelType modelType, Map<String, Object> parameters, 
                          String name, boolean autoFit) {
            this.modelType = modelType;
            this.parameters = parameters != null ? new HashMap<>(parameters) : new HashMap<>();
            this.name = name;
            this.autoFit = autoFit;
        }
        
        public ITimeSeriesModel.ModelType getModelType() { return modelType; }
        public Map<String, Object> getParameters() { return new HashMap<>(parameters); }
        public String getName() { return name; }
        public boolean isAutoFit() { return autoFit; }
        
        public static class Builder {
            private ITimeSeriesModel.ModelType modelType;
            private Map<String, Object> parameters = new HashMap<>();
            private String name;
            private boolean autoFit = false;
            
            public Builder setModelType(ITimeSeriesModel.ModelType modelType) {
                this.modelType = modelType;
                return this;
            }
            
            public Builder addParameter(String key, Object value) {
                this.parameters.put(key, value);
                return this;
            }
            
            public Builder setName(String name) {
                this.name = name;
                return this;
            }
            
            public Builder setAutoFit(boolean autoFit) {
                this.autoFit = autoFit;
                return this;
            }
            
            public ModelConfig build() {
                return new ModelConfig(modelType, parameters, name, autoFit);
            }
        }
    }
    
    /**
     * 创建ARIMA模型 / Create ARIMA Model
     *
     * @param data 时间序列数据 / Time series data
     * @param p AR阶数 / AR order
     * @param d 差分阶数 / Differencing order
     * @param q MA阶数 / MA order
     * @return ARIMA模型 / ARIMA model
     */
    public static ITimeSeriesModel createARIMAModel(IVector<Double> data, int p, int d, int q) {
        return UnifiedARIMAModel.fit(data, p, d, q);
    }
    
    /**
     * 自动选择ARIMA模型 / Auto-select ARIMA Model
     *
     * @param data 时间序列数据 / Time series data
     * @param maxP 最大AR阶数 / Maximum AR order
     * @param maxD 最大差分阶数 / Maximum differencing order
     * @param maxQ 最大MA阶数 / Maximum MA order
     * @param criterion 选择准则 / Selection criterion
     * @return 最优ARIMA模型 / Optimal ARIMA model
     */
    public static ITimeSeriesModel createARIMAModel(IVector<Double> data, int maxP, int maxD, int maxQ, 
                                                   SelectionCriterion criterion) {
        ITimeSeriesModel bestModel = null;
        double bestCriterion = Double.POSITIVE_INFINITY;
        
        for (int p = 0; p <= maxP; p++) {
            for (int d = 0; d <= maxD; d++) {
                for (int q = 0; q <= maxQ; q++) {
                    try {
                        ITimeSeriesModel model = createARIMAModel(data, p, d, q);
                        double[] criteria = model.getInformationCriteria();
                        double criterionValue = (criterion == SelectionCriterion.AIC) ? criteria[0] : criteria[1];
                        
                        if (criterionValue < bestCriterion) {
                            bestCriterion = criterionValue;
                            bestModel = model;
                        }
                    } catch (Exception e) {
                        // 跳过无效的模型参数组合 / Skip invalid model parameter combinations
                        continue;
                    }
                }
            }
        }
        
        if (bestModel == null) {
            throw new RuntimeException("无法找到合适的ARIMA模型");
        }
        
        return bestModel;
    }
    
    /**
     * 创建指数平滑模型 / Create Exponential Smoothing Model
     *
     * @param data 时间序列数据 / Time series data
     * @param alpha 平滑参数 / Smoothing parameter
     * @return 指数平滑模型 / Exponential smoothing model
     */
    public static ITimeSeriesModel createExponentialSmoothingModel(IVector<Double> data, double alpha) {
        // 这里将在后续实现中完成 / This will be completed in subsequent implementations
        throw new UnsupportedOperationException("指数平滑模型将在后续版本中实现");
    }
    
    /**
     * 创建GARCH模型 / Create GARCH Model
     *
     * @param data 时间序列数据 / Time series data
     * @param p ARCH阶数 / ARCH order
     * @param q GARCH阶数 / GARCH order
     * @return GARCH模型 / GARCH model
     */
    public static ITimeSeriesModel createGARCHModel(IVector<Double> data, int p, int q) {
        // 这里将在后续实现中完成 / This will be completed in subsequent implementations
        throw new UnsupportedOperationException("GARCH模型将在后续版本中实现");
    }
    
    /**
     * 创建状态空间模型 / Create State Space Model
     *
     * @param data 时间序列数据 / Time series data
     * @param config 模型配置 / Model configuration
     * @return 状态空间模型 / State space model
     */
    public static ITimeSeriesModel createStateSpaceModel(IVector<Double> data, ModelConfig config) {
        // 这里将在后续实现中完成 / This will be completed in subsequent implementations
        throw new UnsupportedOperationException("状态空间模型将在后续版本中实现");
    }
    
    /**
     * 创建VAR模型 / Create VAR Model
     *
     * @param data 多变量时间序列数据 / Multivariate time series data
     * @param p VAR阶数 / VAR order
     * @param variableNames 变量名称 / Variable names
     * @return VAR模型 / VAR model
     */
    public static ITimeSeriesModel createVARModel(IMatrix<Double> data, int p, String[] variableNames) {
        // 这里将在后续实现中完成 / This will be completed in subsequent implementations
        throw new UnsupportedOperationException("VAR模型将在后续版本中实现");
    }
    
    /**
     * 根据配置创建模型 / Create Model from Configuration
     *
     * @param data 时间序列数据 / Time series data
     * @param config 模型配置 / Model configuration
     * @return 时间序列模型 / Time series model
     */
    public static ITimeSeriesModel createModel(IVector<Double> data, ModelConfig config) {
        if (data == null) {
            throw new IllegalArgumentException("数据不能为空");
        }
        if (config == null) {
            throw new IllegalArgumentException("配置不能为空");
        }
        
        switch (config.getModelType()) {
            case ARIMA:
                return createARIMAModelFromConfig(data, config);
            case EXPONENTIAL_SMOOTHING:
                return createExponentialSmoothingModelFromConfig(data, config);
            case GARCH:
                return createGARCHModelFromConfig(data, config);
            case STATE_SPACE:
                return createStateSpaceModelFromConfig(data, config);
            case VAR:
                throw new UnsupportedOperationException("VAR模型需要多变量数据");
            default:
                throw new UnsupportedOperationException("不支持的模型类型: " + config.getModelType());
        }
    }
    
    /**
     * 根据配置创建多变量模型 / Create Multivariate Model from Configuration
     *
     * @param data 多变量时间序列数据 / Multivariate time series data
     * @param config 模型配置 / Model configuration
     * @return 时间序列模型 / Time series model
     */
    public static ITimeSeriesModel createModel(IMatrix<Double> data, ModelConfig config) {
        if (data == null) {
            throw new IllegalArgumentException("数据不能为空");
        }
        if (config == null) {
            throw new IllegalArgumentException("配置不能为空");
        }
        
        switch (config.getModelType()) {
            case VAR:
                return createVARModelFromConfig(data, config);
            default:
                throw new UnsupportedOperationException("不支持的模型类型: " + config.getModelType());
        }
    }
    
    /**
     * 模型选择器 / Model Selector
     * <p>
     * 自动选择最优的时间序列模型。
     * Automatically select optimal time series model.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @param candidateTypes 候选模型类型 / Candidate model types
     * @param criterion 选择准则 / Selection criterion
     * @return 最优模型 / Optimal model
     */
    public static ITimeSeriesModel selectBestModel(IVector<Double> data, 
                                                  ITimeSeriesModel.ModelType[] candidateTypes,
                                                  SelectionCriterion criterion) {
        if (data == null || candidateTypes == null || candidateTypes.length == 0) {
            throw new IllegalArgumentException("参数不能为空");
        }
        
        ITimeSeriesModel bestModel = null;
        double bestCriterion = Double.POSITIVE_INFINITY;
        
        for (ITimeSeriesModel.ModelType modelType : candidateTypes) {
            try {
                ITimeSeriesModel model = createModelByType(data, modelType);
                if (model != null && model.isValid()) {
                    double[] criteria = model.getInformationCriteria();
                    double criterionValue = (criterion == SelectionCriterion.AIC) ? criteria[0] : criteria[1];
                    
                    if (criterionValue < bestCriterion) {
                        bestCriterion = criterionValue;
                        bestModel = model;
                    }
                }
            } catch (Exception e) {
                // 跳过无法创建的模型 / Skip models that cannot be created
                continue;
            }
        }
        
        if (bestModel == null) {
            throw new RuntimeException("无法找到合适的时间序列模型");
        }
        
        return bestModel;
    }
    
    /**
     * 获取支持的模型类型 / Get Supported Model Types
     *
     * @return 支持的模型类型数组 / Supported model types array
     */
    public static ITimeSeriesModel.ModelType[] getSupportedModelTypes() {
        return modelClasses.keySet().toArray(new ITimeSeriesModel.ModelType[0]);
    }
    
    /**
     * 检查模型类型是否支持 / Check if Model Type is Supported
     *
     * @param modelType 模型类型 / Model type
     * @return 是否支持 / Whether supported
     */
    public static boolean isModelTypeSupported(ITimeSeriesModel.ModelType modelType) {
        return modelClasses.containsKey(modelType);
    }
    
    /**
     * 注册模型类型 / Register Model Type
     *
     * @param modelType 模型类型 / Model type
     * @param modelClass 模型类 / Model class
     */
    public static void registerModelType(ITimeSeriesModel.ModelType modelType, 
                                       Class<? extends ITimeSeriesModel> modelClass) {
        modelClasses.put(modelType, modelClass);
    }
    
    // ========== 私有辅助方法 / Private Helper Methods ==========
    
    /**
     * 根据配置创建ARIMA模型 / Create ARIMA Model from Configuration
     */
    private static ITimeSeriesModel createARIMAModelFromConfig(IVector<Double> data, ModelConfig config) {
        Map<String, Object> params = config.getParameters();
        
        int p = (Integer) params.getOrDefault("p", 1);
        int d = (Integer) params.getOrDefault("d", 0);
        int q = (Integer) params.getOrDefault("q", 1);
        
        if (config.isAutoFit()) {
            int maxP = (Integer) params.getOrDefault("maxP", 3);
            int maxD = (Integer) params.getOrDefault("maxD", 2);
            int maxQ = (Integer) params.getOrDefault("maxQ", 3);
            SelectionCriterion criterion = (SelectionCriterion) params.getOrDefault("criterion", SelectionCriterion.AIC);
            
            return createARIMAModel(data, maxP, maxD, maxQ, criterion);
        } else {
            return createARIMAModel(data, p, d, q);
        }
    }
    
    /**
     * 根据配置创建指数平滑模型 / Create Exponential Smoothing Model from Configuration
     */
    private static ITimeSeriesModel createExponentialSmoothingModelFromConfig(IVector<Double> data, ModelConfig config) {
        // 这里将在后续实现中完成 / This will be completed in subsequent implementations
        throw new UnsupportedOperationException("指数平滑模型将在后续版本中实现");
    }
    
    /**
     * 根据配置创建GARCH模型 / Create GARCH Model from Configuration
     */
    private static ITimeSeriesModel createGARCHModelFromConfig(IVector<Double> data, ModelConfig config) {
        // 这里将在后续实现中完成 / This will be completed in subsequent implementations
        throw new UnsupportedOperationException("GARCH模型将在后续版本中实现");
    }
    
    /**
     * 根据配置创建状态空间模型 / Create State Space Model from Configuration
     */
    private static ITimeSeriesModel createStateSpaceModelFromConfig(IVector<Double> data, ModelConfig config) {
        // 这里将在后续实现中完成 / This will be completed in subsequent implementations
        throw new UnsupportedOperationException("状态空间模型将在后续版本中实现");
    }
    
    /**
     * 根据配置创建VAR模型 / Create VAR Model from Configuration
     */
    private static ITimeSeriesModel createVARModelFromConfig(IMatrix<Double> data, ModelConfig config) {
        // 这里将在后续实现中完成 / This will be completed in subsequent implementations
        throw new UnsupportedOperationException("VAR模型将在后续版本中实现");
    }
    
    /**
     * 根据类型创建模型 / Create Model by Type
     */
    private static ITimeSeriesModel createModelByType(IVector<Double> data, ITimeSeriesModel.ModelType modelType) {
        switch (modelType) {
            case ARIMA:
                return createARIMAModel(data, 1, 0, 1); // 默认参数 / Default parameters
            case EXPONENTIAL_SMOOTHING:
                return createExponentialSmoothingModel(data, 0.3); // 默认参数 / Default parameters
            case GARCH:
                return createGARCHModel(data, 1, 1); // 默认参数 / Default parameters
            case STATE_SPACE:
                return createStateSpaceModel(data, new ModelConfig.Builder()
                    .setModelType(modelType)
                    .build());
            default:
                return null;
        }
    }
    
    /**
     * 选择准则枚举 / Selection Criterion Enum
     */
    public enum SelectionCriterion {
        AIC,    // Akaike Information Criterion
        BIC     // Bayesian Information Criterion
    }
}
