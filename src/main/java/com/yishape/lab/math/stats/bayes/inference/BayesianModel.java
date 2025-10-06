package com.yishape.lab.math.stats.bayes.inference;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.util.Map;

/**
 * 贝叶斯模型接口
 * Bayesian Model Interface
 * 
 * <p>定义了贝叶斯模型的统一接口，包括似然函数、先验分布、
 * 后验分布等核心组件。支持不同类型的贝叶斯模型。</p>
 * 
 * <p>Defines a unified interface for Bayesian models, including 
 * likelihood functions, prior distributions, posterior distributions, 
 * and other core components. Supports different types of Bayesian models.</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface BayesianModel {
    
    /**
     * 模型类型枚举
     * Model Type Enumeration
     */
    enum ModelType {
        /** 线性回归 Linear Regression */
        LINEAR_REGRESSION,
        /** 逻辑回归 Logistic Regression */
        LOGISTIC_REGRESSION,
        /** 高斯过程 Gaussian Process */
        GAUSSIAN_PROCESS,
        /** 混合模型 Mixture Model */
        MIXTURE_MODEL,
        /** 层次模型 Hierarchical Model */
        HIERARCHICAL_MODEL,
        /** 时间序列 Time Series */
        TIME_SERIES,
        /** 自定义 Custom */
        CUSTOM
    }
    
    /**
     * 参数类型枚举
     * Parameter Type Enumeration
     */
    enum ParameterType {
        /** 连续参数 Continuous Parameter */
        CONTINUOUS,
        /** 离散参数 Discrete Parameter */
        DISCRETE,
        /** 矩阵参数 Matrix Parameter */
        MATRIX,
        /** 向量参数 Vector Parameter */
        VECTOR
    }
    
    /**
     * 参数信息
     * Parameter Information
     */
    interface ParameterInfo {
        /**
         * 获取参数名称
         * Get parameter name
         */
        String getName();
        
        /**
         * 获取参数类型
         * Get parameter type
         */
        ParameterType getType();
        
        /**
         * 获取参数维度
         * Get parameter dimension
         */
        int getDimension();
        
        /**
         * 获取参数约束
         * Get parameter constraints
         */
        Map<String, Object> getConstraints();
        
        /**
         * 获取参数描述
         * Get parameter description
         */
        String getDescription();
    }
    
    /**
     * 计算对数似然
     * Compute log likelihood
     * 
     * @param parameters 模型参数
     * @param data 观测数据
     * @return 对数似然值
     */
    double logLikelihood(IVector parameters, IVector data);
    
    /**
     * 计算对数先验
     * Compute log prior
     * 
     * @param parameters 模型参数
     * @return 对数先验值
     */
    double logPrior(IVector parameters);
    
    /**
     * 计算对数后验（未归一化）
     * Compute log posterior (unnormalized)
     * 
     * @param parameters 模型参数
     * @param data 观测数据
     * @return 对数后验值
     */
    default double logPosterior(IVector parameters, IVector data) {
        return logLikelihood(parameters, data) + logPrior(parameters);
    }
    
    /**
     * 计算似然梯度
     * Compute likelihood gradient
     * 
     * @param parameters 模型参数
     * @param data 观测数据
     * @return 似然梯度
     */
    IVector likelihoodGradient(IVector parameters, IVector data);
    
    /**
     * 计算先验梯度
     * Compute prior gradient
     * 
     * @param parameters 模型参数
     * @return 先验梯度
     */
    IVector priorGradient(IVector parameters);
    
    /**
     * 计算后验梯度
     * Compute posterior gradient
     * 
     * @param parameters 模型参数
     * @param data 观测数据
     * @return 后验梯度
     */
    default IVector posteriorGradient(IVector parameters, IVector data) {
        IVector likelihoodGrad = likelihoodGradient(parameters, data);
        IVector priorGrad = priorGradient(parameters);
        
        // 简单的向量加法
        IVector result = Linalg.vector(likelihoodGrad.size());
        for (int i = 0; i < likelihoodGrad.size(); i++) {
            result.set(i, likelihoodGrad.get(i).doubleValue() + priorGrad.get(i).doubleValue());
        }
        return result;
    }
    
    /**
     * 从先验分布采样
     * Sample from prior distribution
     * 
     * @return 先验样本
     */
    IVector sampleFromPrior();
    
    /**
     * 从似然分布采样
     * Sample from likelihood
     * 
     * @param parameters 模型参数
     * @return 似然样本
     */
    IVector sampleFromLikelihood(IVector parameters);
    
    /**
     * 预测
     * Predict
     * 
     * @param parameters 模型参数
     * @param newData 新数据
     * @return 预测结果
     */
    IVector predict(IVector parameters, IVector newData);
    
    /**
     * 计算预测分布
     * Compute predictive distribution
     * 
     * @param parameters 模型参数
     * @param newData 新数据
     * @return 预测分布参数
     */
    Map<String, IVector> predictiveDistribution(IVector parameters, IVector newData);
    
    /**
     * 获取模型类型
     * Get model type
     * 
     * @return 模型类型
     */
    ModelType getModelType();
    
    /**
     * 获取参数数量
     * Get number of parameters
     * 
     * @return 参数数量
     */
    int getNumParameters();
    
    /**
     * 获取参数信息
     * Get parameter information
     * 
     * @return 参数信息列表
     */
    java.util.List<ParameterInfo> getParameterInfo();
    
    /**
     * 获取模型名称
     * Get model name
     * 
     * @return 模型名称
     */
    String getModelName();
    
    /**
     * 获取模型描述
     * Get model description
     * 
     * @return 模型描述
     */
    String getModelDescription();
    
    /**
     * 验证参数
     * Validate parameters
     * 
     * @param parameters 模型参数
     * @return 是否有效
     */
    boolean validateParameters(IVector parameters);
    
    /**
     * 获取参数约束
     * Get parameter constraints
     * 
     * @return 参数约束映射
     */
    Map<String, Object> getParameterConstraints();
    
    /**
     * 初始化参数
     * Initialize parameters
     * 
     * @param data 观测数据
     * @return 初始参数
     */
    IVector initializeParameters(IVector data);
    
    /**
     * 计算信息矩阵
     * Compute information matrix
     * 
     * @param parameters 模型参数
     * @param data 观测数据
     * @return 信息矩阵
     */
    IMatrix informationMatrix(IVector parameters, IVector data);
    
    /**
     * 计算Hessian矩阵
     * Compute Hessian matrix
     * 
     * @param parameters 模型参数
     * @param data 观测数据
     * @return Hessian矩阵
     */
    IMatrix hessianMatrix(IVector parameters, IVector data);
    
    /**
     * 是否支持解析梯度
     * Whether analytical gradient is supported
     * 
     * @return 是否支持
     */
    boolean supportsAnalyticalGradient();
    
    /**
     * 是否支持解析Hessian
     * Whether analytical Hessian is supported
     * 
     * @return 是否支持
     */
    boolean supportsAnalyticalHessian();
    
    /**
     * 获取模型超参数
     * Get model hyperparameters
     * 
     * @return 超参数映射
     */
    Map<String, Object> getHyperparameters();
    
    /**
     * 设置模型超参数
     * Set model hyperparameters
     * 
     * @param hyperparameters 超参数映射
     */
    void setHyperparameters(Map<String, Object> hyperparameters);
    
    /**
     * 克隆模型
     * Clone model
     * 
     * @return 模型副本
     */
    BayesianModel clone();
}