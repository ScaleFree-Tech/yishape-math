package com.reremouse.lab.math.stats.bayes.inference;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;

import java.util.Map;

/**
 * 推断引擎接口
 * Inference Engine Interface
 * 
 * <p>定义了贝叶斯推断的统一接口，支持多种推断算法和模型类型。
 * 提供了标准化的推断流程和结果格式。</p>
 * 
 * <p>Defines a unified interface for Bayesian inference, supporting 
 * multiple inference algorithms and model types. Provides standardized 
 * inference workflow and result format.</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface InferenceEngine {
    
    /**
     * 推断结果
     * Inference Result
     */
    interface InferenceResult {
        /**
         * 获取后验样本
         * Get posterior samples
         * 
         * @return 后验样本矩阵，每行为一个样本
         */
        IMatrix getPosteriorSamples();
        
        /**
         * 获取后验均值
         * Get posterior mean
         * 
         * @return 后验均值向量
         */
        IVector getPosteriorMean();
        
        /**
         * 获取后验协方差
         * Get posterior covariance
         * 
         * @return 后验协方差矩阵
         */
        IMatrix getPosteriorCovariance();
        
        /**
         * 获取对数边际似然
         * Get log marginal likelihood
         * 
         * @return 对数边际似然
         */
        double getLogMarginalLikelihood();
        
        /**
         * 获取有效样本大小
         * Get effective sample size
         * 
         * @return 有效样本大小
         */
        double getEffectiveSampleSize();
        
        /**
         * 获取收敛诊断信息
         * Get convergence diagnostics
         * 
         * @return 收敛诊断信息
         */
        Map<String, Double> getConvergenceDiagnostics();
        
        /**
         * 获取推断算法名称
         * Get inference algorithm name
         * 
         * @return 算法名称
         */
        String getAlgorithmName();
        
        /**
         * 获取计算时间（毫秒）
         * Get computation time in milliseconds
         * 
         * @return 计算时间
         */
        long getComputationTime();
        
        /**
         * 获取额外信息
         * Get additional information
         * 
         * @return 额外信息映射
         */
        Map<String, Object> getAdditionalInfo();
    }
    
    /**
     * 推断配置
     * Inference Configuration
     */
    interface InferenceConfig {
        /**
         * 获取样本数量
         * Get number of samples
         * 
         * @return 样本数量
         */
        int getNumSamples();
        
        /**
         * 获取预热样本数量
         * Get number of warmup samples
         * 
         * @return 预热样本数量
         */
        int getNumWarmup();
        
        /**
         * 获取链数量
         * Get number of chains
         * 
         * @return 链数量
         */
        int getNumChains();
        
        /**
         * 获取随机种子
         * Get random seed
         * 
         * @return 随机种子
         */
        long getSeed();
        
        /**
         * 获取算法特定参数
         * Get algorithm-specific parameters
         * 
         * @return 参数映射
         */
        Map<String, Object> getAlgorithmParams();
        
        /**
         * 是否启用并行计算
         * Whether to enable parallel computation
         * 
         * @return 是否并行
         */
        boolean isParallel();
        
        /**
         * 获取收敛容忍度
         * Get convergence tolerance
         * 
         * @return 收敛容忍度
         */
        double getConvergenceTolerance();
        
        /**
         * 是否保存中间结果
         * Whether to save intermediate results
         * 
         * @return 是否保存
         */
        boolean isSaveIntermediateResults();
    }
    
    /**
     * 执行推断
     * Perform inference
     * 
     * @param model 贝叶斯模型
     * @param data 观测数据
     * @param config 推断配置
     * @return 推断结果
     */
    InferenceResult infer(BayesianModel model, IVector data, InferenceConfig config);
    
    /**
     * 执行推断（使用默认配置）
     * Perform inference with default configuration
     * 
     * @param model 贝叶斯模型
     * @param data 观测数据
     * @return 推断结果
     */
    default InferenceResult infer(BayesianModel model, IVector data) {
        return infer(model, data, getDefaultConfig());
    }
    
    /**
     * 获取默认配置
     * Get default configuration
     * 
     * @return 默认配置
     */
    InferenceConfig getDefaultConfig();
    
    /**
     * 获取算法名称
     * Get algorithm name
     * 
     * @return 算法名称
     */
    String getAlgorithmName();
    
    /**
     * 是否支持并行计算
     * Whether parallel computation is supported
     * 
     * @return 是否支持并行
     */
    boolean supportsParallel();
    
    /**
     * 是否支持在线推断
     * Whether online inference is supported
     * 
     * @return 是否支持在线推断
     */
    boolean supportsOnlineInference();
    
    /**
     * 获取支持的模型类型
     * Get supported model types
     * 
     * @return 支持的模型类型集合
     */
    java.util.Set<Class<? extends BayesianModel>> getSupportedModelTypes();
    
    /**
     * 验证模型兼容性
     * Validate model compatibility
     * 
     * @param model 贝叶斯模型
     * @return 是否兼容
     */
    boolean isModelSupported(BayesianModel model);
    
    /**
     * 预测
     * Predict
     * 
     * @param result 推断结果
     * @param newData 新数据
     * @return 预测结果
     */
    IVector predict(InferenceResult result, IVector newData);
    
    /**
     * 计算预测分布
     * Compute predictive distribution
     * 
     * @param result 推断结果
     * @param newData 新数据
     * @return 预测分布参数
     */
    Map<String, IVector> predictiveDistribution(InferenceResult result, IVector newData);
}