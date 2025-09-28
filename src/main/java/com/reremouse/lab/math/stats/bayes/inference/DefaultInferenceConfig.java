package com.reremouse.lab.math.stats.bayes.inference;

import java.util.HashMap;
import java.util.Map;

/**
 * 默认推断配置实现
 * Default Inference Configuration Implementation
 * 
 * <p>提供推断配置的默认实现，包括常用的配置参数和合理的默认值。
 * 支持链式配置和参数验证。</p>
 * 
 * <p>Provides default implementation of inference configuration, 
 * including common configuration parameters and reasonable default values. 
 * Supports fluent configuration and parameter validation.</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class DefaultInferenceConfig implements InferenceEngine.InferenceConfig {
    
    private int numSamples;
    private int numWarmup;
    private int numChains;
    private long seed;
    private Map<String, Object> algorithmParams;
    private boolean parallel;
    private double convergenceTolerance;
    private boolean saveIntermediateResults;
    
    /**
     * 默认构造函数
     */
    public DefaultInferenceConfig() {
        // 设置默认值
        this.numSamples = 1000;
        this.numWarmup = 500;
        this.numChains = 4;
        this.seed = System.currentTimeMillis();
        this.algorithmParams = new HashMap<>();
        this.parallel = true;
        this.convergenceTolerance = 1e-6;
        this.saveIntermediateResults = false;
    }
    
    /**
     * 复制构造函数
     * 
     * @param other 其他配置
     */
    public DefaultInferenceConfig(InferenceEngine.InferenceConfig other) {
        this.numSamples = other.getNumSamples();
        this.numWarmup = other.getNumWarmup();
        this.numChains = other.getNumChains();
        this.seed = other.getSeed();
        this.algorithmParams = new HashMap<>(other.getAlgorithmParams());
        this.parallel = other.isParallel();
        this.convergenceTolerance = other.getConvergenceTolerance();
        this.saveIntermediateResults = other.isSaveIntermediateResults();
    }
    
    @Override
    public int getNumSamples() {
        return numSamples;
    }
    
    @Override
    public int getNumWarmup() {
        return numWarmup;
    }
    
    @Override
    public int getNumChains() {
        return numChains;
    }
    
    @Override
    public long getSeed() {
        return seed;
    }
    
    @Override
    public Map<String, Object> getAlgorithmParams() {
        return new HashMap<>(algorithmParams);
    }
    
    @Override
    public boolean isParallel() {
        return parallel;
    }
    
    @Override
    public double getConvergenceTolerance() {
        return convergenceTolerance;
    }
    
    @Override
    public boolean isSaveIntermediateResults() {
        return saveIntermediateResults;
    }
    
    /**
     * 设置样本数量
     * Set number of samples
     * 
     * @param numSamples 样本数量
     * @return 配置对象（支持链式调用）
     */
    public DefaultInferenceConfig setNumSamples(int numSamples) {
        if (numSamples <= 0) {
            throw new IllegalArgumentException("Number of samples must be positive");
        }
        this.numSamples = numSamples;
        return this;
    }
    
    /**
     * 设置预热样本数量
     * Set number of warmup samples
     * 
     * @param numWarmup 预热样本数量
     * @return 配置对象（支持链式调用）
     */
    public DefaultInferenceConfig setNumWarmup(int numWarmup) {
        if (numWarmup < 0) {
            throw new IllegalArgumentException("Number of warmup samples cannot be negative");
        }
        this.numWarmup = numWarmup;
        return this;
    }
    
    /**
     * 设置链数量
     * Set number of chains
     * 
     * @param numChains 链数量
     * @return 配置对象（支持链式调用）
     */
    public DefaultInferenceConfig setNumChains(int numChains) {
        if (numChains <= 0) {
            throw new IllegalArgumentException("Number of chains must be positive");
        }
        this.numChains = numChains;
        return this;
    }
    
    /**
     * 设置随机种子
     * Set random seed
     * 
     * @param seed 随机种子
     * @return 配置对象（支持链式调用）
     */
    public DefaultInferenceConfig setSeed(long seed) {
        this.seed = seed;
        return this;
    }
    
    /**
     * 设置算法参数
     * Set algorithm parameters
     * 
     * @param algorithmParams 算法参数
     * @return 配置对象（支持链式调用）
     */
    public DefaultInferenceConfig setAlgorithmParams(Map<String, Object> algorithmParams) {
        this.algorithmParams = new HashMap<>(algorithmParams != null ? algorithmParams : new HashMap<>());
        return this;
    }
    
    /**
     * 添加算法参数
     * Add algorithm parameter
     * 
     * @param key 参数名
     * @param value 参数值
     * @return 配置对象（支持链式调用）
     */
    public DefaultInferenceConfig addAlgorithmParam(String key, Object value) {
        this.algorithmParams.put(key, value);
        return this;
    }
    
    /**
     * 设置是否并行
     * Set parallel flag
     * 
     * @param parallel 是否并行
     * @return 配置对象（支持链式调用）
     */
    public DefaultInferenceConfig setParallel(boolean parallel) {
        this.parallel = parallel;
        return this;
    }
    
    /**
     * 设置收敛容忍度
     * Set convergence tolerance
     * 
     * @param convergenceTolerance 收敛容忍度
     * @return 配置对象（支持链式调用）
     */
    public DefaultInferenceConfig setConvergenceTolerance(double convergenceTolerance) {
        if (convergenceTolerance <= 0) {
            throw new IllegalArgumentException("Convergence tolerance must be positive");
        }
        this.convergenceTolerance = convergenceTolerance;
        return this;
    }
    
    /**
     * 设置是否保存中间结果
     * Set save intermediate results flag
     * 
     * @param saveIntermediateResults 是否保存中间结果
     * @return 配置对象（支持链式调用）
     */
    public DefaultInferenceConfig setSaveIntermediateResults(boolean saveIntermediateResults) {
        this.saveIntermediateResults = saveIntermediateResults;
        return this;
    }
    
    /**
     * 验证配置
     * Validate configuration
     * 
     * @throws IllegalStateException 如果配置无效
     */
    public void validate() {
        if (numSamples <= 0) {
            throw new IllegalStateException("Number of samples must be positive");
        }
        if (numWarmup < 0) {
            throw new IllegalStateException("Number of warmup samples cannot be negative");
        }
        if (numChains <= 0) {
            throw new IllegalStateException("Number of chains must be positive");
        }
        if (convergenceTolerance <= 0) {
            throw new IllegalStateException("Convergence tolerance must be positive");
        }
    }
    
    /**
     * 创建构建器
     * Create builder
     * 
     * @return 配置构建器
     */
    public static Builder builder() {
        return new Builder();
    }
    
    /**
     * 配置构建器
     * Configuration Builder
     */
    public static class Builder {
        private final DefaultInferenceConfig config;
        
        private Builder() {
            this.config = new DefaultInferenceConfig();
        }
        
        public Builder numSamples(int numSamples) {
            config.setNumSamples(numSamples);
            return this;
        }
        
        public Builder numWarmup(int numWarmup) {
            config.setNumWarmup(numWarmup);
            return this;
        }
        
        public Builder numChains(int numChains) {
            config.setNumChains(numChains);
            return this;
        }
        
        public Builder seed(long seed) {
            config.setSeed(seed);
            return this;
        }
        
        public Builder algorithmParams(Map<String, Object> algorithmParams) {
            config.setAlgorithmParams(algorithmParams);
            return this;
        }
        
        public Builder algorithmParam(String key, Object value) {
            config.addAlgorithmParam(key, value);
            return this;
        }
        
        public Builder parallel(boolean parallel) {
            config.setParallel(parallel);
            return this;
        }
        
        public Builder convergenceTolerance(double convergenceTolerance) {
            config.setConvergenceTolerance(convergenceTolerance);
            return this;
        }
        
        public Builder saveIntermediateResults(boolean saveIntermediateResults) {
            config.setSaveIntermediateResults(saveIntermediateResults);
            return this;
        }
        
        public DefaultInferenceConfig build() {
            config.validate();
            return new DefaultInferenceConfig(config);
        }
    }
    
    /**
     * 创建快速配置
     * Create quick configurations
     */
    public static class QuickConfigs {
        
        /**
         * 快速原型配置
         * Quick prototype configuration
         */
        public static DefaultInferenceConfig quickPrototype() {
            return new DefaultInferenceConfig()
                .setNumSamples(100)
                .setNumWarmup(50)
                .setNumChains(1)
                .setParallel(false);
        }
        
        /**
         * 标准配置
         * Standard configuration
         */
        public static DefaultInferenceConfig standard() {
            return new DefaultInferenceConfig()
                .setNumSamples(1000)
                .setNumWarmup(500)
                .setNumChains(4)
                .setParallel(true);
        }
        
        /**
         * 高精度配置
         * High precision configuration
         */
        public static DefaultInferenceConfig highPrecision() {
            return new DefaultInferenceConfig()
                .setNumSamples(5000)
                .setNumWarmup(2000)
                .setNumChains(8)
                .setParallel(true)
                .setConvergenceTolerance(1e-8);
        }
        
        /**
         * 快速配置
         * Fast configuration
         */
        public static DefaultInferenceConfig fast() {
            return new DefaultInferenceConfig()
                .setNumSamples(500)
                .setNumWarmup(200)
                .setNumChains(2)
                .setParallel(true);
        }
        
        /**
         * 调试配置
         * Debug configuration
         */
        public static DefaultInferenceConfig debug() {
            return new DefaultInferenceConfig()
                .setNumSamples(100)
                .setNumWarmup(50)
                .setNumChains(1)
                .setParallel(false)
                .setSaveIntermediateResults(true);
        }
    }
    
    @Override
    public String toString() {
        return String.format(
            "DefaultInferenceConfig{numSamples=%d, numWarmup=%d, numChains=%d, " +
            "seed=%d, parallel=%s, convergenceTolerance=%g, saveIntermediateResults=%s, " +
            "algorithmParams=%s}",
            numSamples, numWarmup, numChains, seed, parallel, 
            convergenceTolerance, saveIntermediateResults, algorithmParams
        );
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        DefaultInferenceConfig that = (DefaultInferenceConfig) obj;
        return numSamples == that.numSamples &&
               numWarmup == that.numWarmup &&
               numChains == that.numChains &&
               seed == that.seed &&
               parallel == that.parallel &&
               Double.compare(that.convergenceTolerance, convergenceTolerance) == 0 &&
               saveIntermediateResults == that.saveIntermediateResults &&
               algorithmParams.equals(that.algorithmParams);
    }
    
    @Override
    public int hashCode() {
        return java.util.Objects.hash(numSamples, numWarmup, numChains, seed, 
                                    algorithmParams, parallel, convergenceTolerance, 
                                    saveIntermediateResults);
    }
}