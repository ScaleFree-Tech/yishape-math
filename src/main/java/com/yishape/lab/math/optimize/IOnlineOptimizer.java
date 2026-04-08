package com.yishape.lab.math.optimize;

import com.yishape.lab.math.linalg.IVector;
import java.io.Serializable;
import java.util.function.BiFunction;

/**
 * 在线优化器接口
 * Online Optimizer Interface
 * 
 * 专门用于在线学习和增量优化的接口，支持逐步接收数据样本并更新模型参数。
 * 与批量优化器(IOptimizer)不同，在线优化器维护内部状态，支持流式数据处理。
 * 
 * Interface specifically designed for online learning and incremental optimization,
 * supporting step-by-step data sample reception and model parameter updates.
 * Unlike batch optimizers (IOptimizer), online optimizers maintain internal state
 * and support streaming data processing.
 * 
 * @author lteb2
 */
public interface IOnlineOptimizer  extends Serializable{
    
    /**
     * 初始化优化器
     * Initialize the optimizer
     * 
     * @param initialParams 初始参数向量 / Initial parameter vector
     * @throws IllegalArgumentException 如果初始参数为null / if initial parameters are null
     */
    void initialize(IVector initialParams);
    
    /**
     * 执行一步优化更新
     * Perform one step of optimization update
     * 
     * @param gradient 当前梯度向量 / Current gradient vector
     * @return 更新后的参数向量 / Updated parameter vector
     * @throws IllegalStateException 如果优化器未初始化 / if optimizer is not initialized
     * @throws IllegalArgumentException 如果梯度为null / if gradient is null
     */
    IVector step(IVector gradient);
    
    /**
     * 执行一步优化更新（带损失值）
     * Perform one step of optimization update (with loss value)
     * 
     * @param gradient 当前梯度向量 / Current gradient vector
     * @param loss 当前损失值 / Current loss value
     * @return 更新后的参数向量 / Updated parameter vector
     * @throws IllegalStateException 如果优化器未初始化 / if optimizer is not initialized
     * @throws IllegalArgumentException 如果梯度为null / if gradient is null
     */
    default IVector step(IVector gradient, double loss) {
        return step(gradient);
    }
    
    /**
     * 使用数据样本和损失函数执行一步优化更新
     * Perform one step of optimization update using data sample and loss function
     * 
     * @param sample 数据样本 / Data sample
     * @param lossFunction 损失函数，接收参数向量和数据样本，返回损失值 / Loss function that takes parameter vector and data sample, returns loss value
     * @return 更新后的参数向量 / Updated parameter vector
     * @throws IllegalStateException 如果优化器未初始化 / if optimizer is not initialized
     * @throws IllegalArgumentException 如果样本或损失函数为null / if sample or loss function is null
     */
    default <T> IVector step(T sample, BiFunction<IVector, T, Double> lossFunction) {
        if (sample == null) {
            throw new IllegalArgumentException("数据样本不能为null / Data sample cannot be null");
        }
        if (lossFunction == null) {
            throw new IllegalArgumentException("损失函数不能为null / Loss function cannot be null");
        }
        
        // 计算当前参数的损失值
        IVector currentParams = getCurrentParams();
        double loss = lossFunction.apply(currentParams, sample);
        
        // 计算梯度（数值梯度）
        IVector gradient = computeNumericalGradient(currentParams, sample, lossFunction);
        
        // 执行优化步骤
        return step(gradient, loss);
    }
    
    /**
     * 使用小批量数据样本和损失函数执行一步优化更新
     * Perform one step of optimization update using mini-batch data samples and loss function
     * 
     * @param samples 小批量数据样本 / Mini-batch data samples
     * @param lossFunction 损失函数，接收参数向量和数据样本，返回损失值 / Loss function that takes parameter vector and data sample, returns loss value
     * @return 更新后的参数向量 / Updated parameter vector
     * @throws IllegalStateException 如果优化器未初始化 / if optimizer is not initialized
     * @throws IllegalArgumentException 如果样本数组或损失函数为null / if sample array or loss function is null
     */
    default <T> IVector step(T[] samples, BiFunction<IVector, T, Double> lossFunction) {
        if (samples == null || samples.length == 0) {
            throw new IllegalArgumentException("数据样本数组不能为空 / Data sample array cannot be null or empty");
        }
        if (lossFunction == null) {
            throw new IllegalArgumentException("损失函数不能为null / Loss function cannot be null");
        }
        
        // 计算当前参数
        IVector currentParams = getCurrentParams();
        
        // 计算小批量的平均损失和平均梯度
        double totalLoss = 0.0;
        IVector totalGradient = null;
        
        for (int i = 0; i < samples.length; i++) {
            T sample = samples[i];
            if (sample == null) {
                throw new IllegalArgumentException("数据样本数组中不能包含null元素 / Data sample array cannot contain null elements");
            }
            
            // 计算当前样本的损失值
            double loss = lossFunction.apply(currentParams, sample);
            totalLoss += loss;
            
            // 计算当前样本的梯度
            IVector gradient = computeNumericalGradient(currentParams, sample, lossFunction);
            
            // 累加梯度
            if (totalGradient == null) {
                totalGradient = gradient.copy();
            } else {
                totalGradient = totalGradient.add(gradient);
            }
        }
        
        // 计算平均损失和平均梯度
        double averageLoss = totalLoss / samples.length;
        IVector averageGradient = totalGradient.multiplyScalar(1.0 / samples.length);
        
        // 执行优化步骤
        return step(averageGradient, averageLoss);
    }
    
    /**
     * 数值梯度计算
     * Numerical gradient computation
     * 
     * @param params 当前参数向量 / Current parameter vector
     * @param sample 数据样本 / Data sample
     * @param lossFunction 损失函数 / Loss function
     * @return 梯度向量 / Gradient vector
     */
    default <T> IVector computeNumericalGradient(IVector params, T sample, BiFunction<IVector, T, Double> lossFunction) {
        double epsilon = 1e-6;
        IVector gradient = params.copy().multiplyScalar(0.0);
        
        // 对每个参数计算偏导数
        for (int i = 0; i < params.size(); i++) {
            // 保存原始参数值
            double originalValue = params.get(i).doubleValue();
            
            // 计算 f(x + ε)
            IVector paramsPlus = params.copy();
            paramsPlus.set(i, originalValue + epsilon);
            double lossPlus = lossFunction.apply(paramsPlus, sample);
            
            // 计算 f(x - ε)
            IVector paramsMinus = params.copy();
            paramsMinus.set(i, originalValue - epsilon);
            double lossMinus = lossFunction.apply(paramsMinus, sample);
            
            // 计算梯度分量: (f(x + ε) - f(x - ε)) / (2 * ε)
            double grad = (lossPlus - lossMinus) / (2.0 * epsilon);
            gradient.set(i, grad);
        }
        
        return gradient;
    }
    
    /**
     * 获取当前参数
     * Get current parameters
     * 
     * @return 当前参数向量的副本 / Copy of current parameter vector
     * @throws IllegalStateException 如果优化器未初始化 / if optimizer is not initialized
     */
    IVector getCurrentParams();
    
    /**
     * 设置当前参数
     * Set current parameters
     * 
     * @param params 新的参数向量 / New parameter vector
     * @throws IllegalArgumentException 如果参数为null / if parameters are null
     */
    void setCurrentParams(IVector params);
    
    /**
     * 获取当前学习率
     * Get current learning rate
     * 
     * @return 当前学习率 / Current learning rate
     */
    double getCurrentLearningRate();
    
    /**
     * 设置学习率
     * Set learning rate
     * 
     * @param learningRate 新的学习率 / New learning rate
     * @throws IllegalArgumentException 如果学习率小于等于0 / if learning rate is &lt;= 0
     */
    void setLearningRate(double learningRate);
    
    /**
     * 获取当前迭代步数
     * Get current iteration step
     * 
     * @return 当前迭代步数 / Current iteration step
     */
    int getCurrentStep();
    
    /**
     * 重置优化器状态
     * Reset optimizer state
     * 
     * 清除所有内部状态，但保留配置参数（如学习率、动量等）
     * Clears all internal state but retains configuration parameters (like learning rate, momentum, etc.)
     */
    void reset();
    
    /**
     * 检查优化器是否已初始化
     * Check if optimizer is initialized
     * 
     * @return true如果已初始化，false否则 / true if initialized, false otherwise
     */
    boolean isInitialized();
    
    /**
     * 获取优化器状态信息
     * Get optimizer state information
     * 
     * @return 包含优化器状态的字符串 / String containing optimizer state information
     */
    default String getStateInfo() {
        return String.format("Step: %d, LR: %.6f, Initialized: %s", 
                           getCurrentStep(), getCurrentLearningRate(), isInitialized());
    }
    
    /**
     * 克隆优化器（深拷贝）
     * Clone optimizer (deep copy)
     * 
     * @return 优化器的深拷贝 / Deep copy of the optimizer
     */
    IOnlineOptimizer clone();
}