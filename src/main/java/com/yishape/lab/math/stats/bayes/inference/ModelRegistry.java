package com.yishape.lab.math.stats.bayes.inference;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

/**
 * 模型注册机制
 * Model Registry
 * 
 * <p>提供贝叶斯模型和推断引擎的注册、发现和管理功能。
 * 支持动态注册和插件式架构。</p>
 * 
 * <p>Provides registration, discovery, and management functionality 
 * for Bayesian models and inference engines. Supports dynamic 
 * registration and plugin architecture.</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ModelRegistry {
    
    private static final ModelRegistry INSTANCE = new ModelRegistry();
    
    private final Map<String, ModelFactory> modelFactories;
    private final Map<String, InferenceEngineFactory> engineFactories;
    private final Map<String, ModelMetadata> modelMetadata;
    private final Map<String, EngineMetadata> engineMetadata;
    private final Map<Class<? extends BayesianModel>, Set<String>> modelTypeToEngines;
    
    /**
     * 模型工厂接口
     * Model Factory Interface
     */
    @FunctionalInterface
    public interface ModelFactory {
        /**
         * 创建模型实例
         * Create model instance
         * 
         * @param config 配置参数
         * @return 模型实例
         */
        BayesianModel createModel(Map<String, Object> config);
    }
    
    /**
     * 推断引擎工厂接口
     * Inference Engine Factory Interface
     */
    @FunctionalInterface
    public interface InferenceEngineFactory {
        /**
         * 创建推断引擎实例
         * Create inference engine instance
         * 
         * @param config 配置参数
         * @return 推断引擎实例
         */
        InferenceEngine createEngine(Map<String, Object> config);
    }
    
    /**
     * 模型元数据
     * Model Metadata
     */
    public static class ModelMetadata {
        private final String name;
        private final String description;
        private final String version;
        private final BayesianModel.ModelType type;
        private final Set<String> supportedEngines;
        private final Map<String, Object> defaultConfig;
        private final String author;
        private final Date createdDate;
        
        public ModelMetadata(String name, String description, String version,
                           BayesianModel.ModelType type, Set<String> supportedEngines,
                           Map<String, Object> defaultConfig, String author) {
            this.name = name;
            this.description = description;
            this.version = version;
            this.type = type;
            this.supportedEngines = new HashSet<>(supportedEngines);
            this.defaultConfig = new HashMap<>(defaultConfig);
            this.author = author;
            this.createdDate = new Date();
        }
        
        // Getters
        public String getName() { return name; }
        public String getDescription() { return description; }
        public String getVersion() { return version; }
        public BayesianModel.ModelType getType() { return type; }
        public Set<String> getSupportedEngines() { return new HashSet<>(supportedEngines); }
        public Map<String, Object> getDefaultConfig() { return new HashMap<>(defaultConfig); }
        public String getAuthor() { return author; }
        public Date getCreatedDate() { return new Date(createdDate.getTime()); }
    }
    
    /**
     * 推断引擎元数据
     * Inference Engine Metadata
     */
    public static class EngineMetadata {
        private final String name;
        private final String description;
        private final String version;
        private final Set<BayesianModel.ModelType> supportedModelTypes;
        private final Map<String, Object> defaultConfig;
        private final boolean supportsParallel;
        private final boolean supportsOnline;
        private final String author;
        private final Date createdDate;
        
        public EngineMetadata(String name, String description, String version,
                            Set<BayesianModel.ModelType> supportedModelTypes,
                            Map<String, Object> defaultConfig,
                            boolean supportsParallel, boolean supportsOnline,
                            String author) {
            this.name = name;
            this.description = description;
            this.version = version;
            this.supportedModelTypes = new HashSet<>(supportedModelTypes);
            this.defaultConfig = new HashMap<>(defaultConfig);
            this.supportsParallel = supportsParallel;
            this.supportsOnline = supportsOnline;
            this.author = author;
            this.createdDate = new Date();
        }
        
        // Getters
        public String getName() { return name; }
        public String getDescription() { return description; }
        public String getVersion() { return version; }
        public Set<BayesianModel.ModelType> getSupportedModelTypes() { 
            return new HashSet<>(supportedModelTypes); 
        }
        public Map<String, Object> getDefaultConfig() { return new HashMap<>(defaultConfig); }
        public boolean supportsParallel() { return supportsParallel; }
        public boolean supportsOnline() { return supportsOnline; }
        public String getAuthor() { return author; }
        public Date getCreatedDate() { return new Date(createdDate.getTime()); }
    }
    
    /**
     * 私有构造函数
     */
    private ModelRegistry() {
        this.modelFactories = new ConcurrentHashMap<>();
        this.engineFactories = new ConcurrentHashMap<>();
        this.modelMetadata = new ConcurrentHashMap<>();
        this.engineMetadata = new ConcurrentHashMap<>();
        this.modelTypeToEngines = new ConcurrentHashMap<>();
        
        // 注册默认模型和引擎
        registerDefaultModelsAndEngines();
    }
    
    /**
     * 获取单例实例
     * Get singleton instance
     * 
     * @return 注册表实例
     */
    public static ModelRegistry getInstance() {
        return INSTANCE;
    }
    
    /**
     * 注册模型
     * Register model
     * 
     * @param name 模型名称
     * @param factory 模型工厂
     * @param metadata 模型元数据
     */
    public void registerModel(String name, ModelFactory factory, ModelMetadata metadata) {
        if (name == null || factory == null || metadata == null) {
            throw new IllegalArgumentException("Name, factory, and metadata cannot be null");
        }
        
        modelFactories.put(name, factory);
        modelMetadata.put(name, metadata);
        
        // 更新模型类型到引擎的映射
        modelTypeToEngines.computeIfAbsent(
            getModelClass(metadata.getType()), 
            k -> new HashSet<>()
        ).addAll(metadata.getSupportedEngines());
    }
    
    /**
     * 注册推断引擎
     * Register inference engine
     * 
     * @param name 引擎名称
     * @param factory 引擎工厂
     * @param metadata 引擎元数据
     */
    public void registerInferenceEngine(String name, InferenceEngineFactory factory, 
                                      EngineMetadata metadata) {
        if (name == null || factory == null || metadata == null) {
            throw new IllegalArgumentException("Name, factory, and metadata cannot be null");
        }
        
        engineFactories.put(name, factory);
        engineMetadata.put(name, metadata);
        
        // 更新模型类型到引擎的映射
        for (BayesianModel.ModelType type : metadata.getSupportedModelTypes()) {
            modelTypeToEngines.computeIfAbsent(
                getModelClass(type), 
                k -> new HashSet<>()
            ).add(name);
        }
    }
    
    /**
     * 创建模型
     * Create model
     * 
     * @param name 模型名称
     * @param config 配置参数
     * @return 模型实例
     */
    public BayesianModel createModel(String name, Map<String, Object> config) {
        ModelFactory factory = modelFactories.get(name);
        if (factory == null) {
            throw new IllegalArgumentException("Unknown model: " + name);
        }
        
        // 合并默认配置
        Map<String, Object> mergedConfig = new HashMap<>();
        ModelMetadata metadata = modelMetadata.get(name);
        if (metadata != null) {
            mergedConfig.putAll(metadata.getDefaultConfig());
        }
        if (config != null) {
            mergedConfig.putAll(config);
        }
        
        return factory.createModel(mergedConfig);
    }
    
    /**
     * 创建模型（使用默认配置）
     * Create model with default configuration
     * 
     * @param name 模型名称
     * @return 模型实例
     */
    public BayesianModel createModel(String name) {
        return createModel(name, null);
    }
    
    /**
     * 创建推断引擎
     * Create inference engine
     * 
     * @param name 引擎名称
     * @param config 配置参数
     * @return 推断引擎实例
     */
    public InferenceEngine createInferenceEngine(String name, Map<String, Object> config) {
        InferenceEngineFactory factory = engineFactories.get(name);
        if (factory == null) {
            throw new IllegalArgumentException("Unknown inference engine: " + name);
        }
        
        // 合并默认配置
        Map<String, Object> mergedConfig = new HashMap<>();
        EngineMetadata metadata = engineMetadata.get(name);
        if (metadata != null) {
            mergedConfig.putAll(metadata.getDefaultConfig());
        }
        if (config != null) {
            mergedConfig.putAll(config);
        }
        
        return factory.createEngine(mergedConfig);
    }
    
    /**
     * 创建推断引擎（使用默认配置）
     * Create inference engine with default configuration
     * 
     * @param name 引擎名称
     * @return 推断引擎实例
     */
    public InferenceEngine createInferenceEngine(String name) {
        return createInferenceEngine(name, null);
    }
    
    /**
     * 获取所有注册的模型名称
     * Get all registered model names
     * 
     * @return 模型名称集合
     */
    public Set<String> getRegisteredModels() {
        return new HashSet<>(modelFactories.keySet());
    }
    
    /**
     * 获取所有注册的推断引擎名称
     * Get all registered inference engine names
     * 
     * @return 引擎名称集合
     */
    public Set<String> getRegisteredInferenceEngines() {
        return new HashSet<>(engineFactories.keySet());
    }
    
    /**
     * 获取模型元数据
     * Get model metadata
     * 
     * @param name 模型名称
     * @return 模型元数据
     */
    public ModelMetadata getModelMetadata(String name) {
        return modelMetadata.get(name);
    }
    
    /**
     * 获取推断引擎元数据
     * Get inference engine metadata
     * 
     * @param name 引擎名称
     * @return 引擎元数据
     */
    public EngineMetadata getInferenceEngineMetadata(String name) {
        return engineMetadata.get(name);
    }
    
    /**
     * 查找兼容的推断引擎
     * Find compatible inference engines
     * 
     * @param model 贝叶斯模型
     * @return 兼容的引擎名称集合
     */
    public Set<String> findCompatibleEngines(BayesianModel model) {
        Set<String> compatibleEngines = new HashSet<>();
        
        for (Map.Entry<String, EngineMetadata> entry : engineMetadata.entrySet()) {
            EngineMetadata metadata = entry.getValue();
            if (metadata.getSupportedModelTypes().contains(model.getModelType())) {
                compatibleEngines.add(entry.getKey());
            }
        }
        
        return compatibleEngines;
    }
    
    /**
     * 查找兼容的推断引擎（按模型类型）
     * Find compatible inference engines by model type
     * 
     * @param modelType 模型类型
     * @return 兼容的引擎名称集合
     */
    public Set<String> findCompatibleEngines(BayesianModel.ModelType modelType) {
        Set<String> compatibleEngines = new HashSet<>();
        
        for (Map.Entry<String, EngineMetadata> entry : engineMetadata.entrySet()) {
            EngineMetadata metadata = entry.getValue();
            if (metadata.getSupportedModelTypes().contains(modelType)) {
                compatibleEngines.add(entry.getKey());
            }
        }
        
        return compatibleEngines;
    }
    
    /**
     * 推荐最佳推断引擎
     * Recommend best inference engine
     * 
     * @param model 贝叶斯模型
     * @param requirements 需求（如并行、在线等）
     * @return 推荐的引擎名称
     */
    public String recommendEngine(BayesianModel model, Map<String, Boolean> requirements) {
        Set<String> compatibleEngines = findCompatibleEngines(model);
        
        if (compatibleEngines.isEmpty()) {
            return null;
        }
        
        // 根据需求筛选
        if (requirements != null) {
            compatibleEngines = compatibleEngines.stream()
                .filter(engineName -> {
                    EngineMetadata metadata = engineMetadata.get(engineName);
                    if (metadata == null) return false;
                    
                    Boolean needsParallel = requirements.get("parallel");
                    if (needsParallel != null && needsParallel && !metadata.supportsParallel()) {
                        return false;
                    }
                    
                    Boolean needsOnline = requirements.get("online");
                    if (needsOnline != null && needsOnline && !metadata.supportsOnline()) {
                        return false;
                    }
                    
                    return true;
                })
                .collect(java.util.stream.Collectors.toSet());
        }
        
        // 返回第一个匹配的引擎（可以添加更复杂的选择逻辑）
        return compatibleEngines.iterator().next();
    }
    
    /**
     * 注销模型
     * Unregister model
     * 
     * @param name 模型名称
     */
    public void unregisterModel(String name) {
        modelFactories.remove(name);
        ModelMetadata metadata = modelMetadata.remove(name);
        
        if (metadata != null) {
            // 更新模型类型到引擎的映射
            Class<? extends BayesianModel> modelClass = getModelClass(metadata.getType());
            Set<String> engines = modelTypeToEngines.get(modelClass);
            if (engines != null) {
                engines.removeAll(metadata.getSupportedEngines());
                if (engines.isEmpty()) {
                    modelTypeToEngines.remove(modelClass);
                }
            }
        }
    }
    
    /**
     * 注销推断引擎
     * Unregister inference engine
     * 
     * @param name 引擎名称
     */
    public void unregisterInferenceEngine(String name) {
        engineFactories.remove(name);
        EngineMetadata metadata = engineMetadata.remove(name);
        
        if (metadata != null) {
            // 更新模型类型到引擎的映射
            for (BayesianModel.ModelType type : metadata.getSupportedModelTypes()) {
                Class<? extends BayesianModel> modelClass = getModelClass(type);
                Set<String> engines = modelTypeToEngines.get(modelClass);
                if (engines != null) {
                    engines.remove(name);
                    if (engines.isEmpty()) {
                        modelTypeToEngines.remove(modelClass);
                    }
                }
            }
        }
    }
    
    /**
     * 清空注册表
     * Clear registry
     */
    public void clear() {
        modelFactories.clear();
        engineFactories.clear();
        modelMetadata.clear();
        engineMetadata.clear();
        modelTypeToEngines.clear();
    }
    
    /**
     * 注册默认模型和引擎
     * Register default models and engines
     */
    private void registerDefaultModelsAndEngines() {
        // 这里可以注册一些默认的模型和引擎
        // 实际实现中可以从配置文件或注解中自动发现
    }
    
    /**
     * 获取模型类型对应的类
     * Get model class for model type
     */
    private Class<? extends BayesianModel> getModelClass(BayesianModel.ModelType type) {
        // 简化实现，实际中可能需要更复杂的映射
        return BayesianModel.class;
    }
}