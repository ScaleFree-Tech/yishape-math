package com.reremouse.lab.math.optimize.newton;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.optimize.IOnlineOptimizer;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * 在线Adam优化器 (Adaptive Moment Estimation)
 * Online Adam Optimizer - Adaptive Moment Estimation
 * 
 * 实现在线学习版本的Adam算法，支持逐步接收梯度并更新参数。
 * 结合了动量和自适应学习率的优势，适用于流式数据处理和增量学习场景。
 * 
 * @author lteb2
 */
public class RereOnlineAdam implements IOnlineOptimizer {
    
    private double learningRate = 0.001;     // 学习率 (α)
    private double beta1 = 0.9;              // 一阶矩估计的指数衰减率
    private double beta2 = 0.999;            // 二阶矩估计的指数衰减率
    private double epsilon = 1e-8;           // 数值稳定性常数
    private double weightDecay = 0.0;        // 权重衰减
    private boolean verbose = false;         // 是否输出详细信息
    private boolean amsgrad = false;         // 是否使用AMSGrad变体
    private double lrDecayRate = 0.0;        // 学习率衰减率
    private int lrDecayStep = 0;             // 学习率衰减步长
    
    // Adam状态变量
    private IVector currentParams = null;    // 当前参数
    private IVector m = null;                // 一阶矩估计 (梯度的指数移动平均)
    private IVector v = null;                // 二阶矩估计 (梯度平方的指数移动平均)
    private IVector vMax = null;             // AMSGrad的最大二阶矩
    private int currentStep = 0;             // 当前步数
    private boolean initialized = false;     // 是否已初始化
    
    /**
     * 默认构造函数
     */
    public RereOnlineAdam() {
    }
    
    /**
     * 构造函数
     * @param learningRate 学习率
     */
    public RereOnlineAdam(double learningRate) {
        this.setLearningRate(learningRate);
    }
    
    /**
     * 构造函数
     * @param learningRate 学习率
     * @param beta1 一阶矩衰减率
     * @param beta2 二阶矩衰减率
     */
    public RereOnlineAdam(double learningRate, double beta1, double beta2) {
        this.setLearningRate(learningRate);
        this.setBeta1(beta1);
        this.setBeta2(beta2);
    }
    
    /**
     * 完整参数构造函数
     * @param learningRate 学习率
     * @param beta1 一阶矩衰减率
     * @param beta2 二阶矩衰减率
     * @param epsilon 数值稳定性常数
     * @param weightDecay 权重衰减
     */
    public RereOnlineAdam(double learningRate, double beta1, double beta2, 
                          double epsilon, double weightDecay) {
        this.setLearningRate(learningRate);
        this.setBeta1(beta1);
        this.setBeta2(beta2);
        this.setEpsilon(epsilon);
        this.setWeightDecay(weightDecay);
    }

    @Override
    public void initialize(IVector initialParams) {
        if (initialParams == null) {
            throw new IllegalArgumentException("初始参数不能为null");
        }
        
        this.currentParams = initialParams.copy();
        
        // 初始化Adam状态变量
        this.m = initialParams.copy().multiplyScalar(0.0);  // 一阶矩估计，初始化为零向量
        this.v = initialParams.copy().multiplyScalar(0.0);  // 二阶矩估计，初始化为零向量
        if (amsgrad) {
            this.vMax = initialParams.copy().multiplyScalar(0.0);  // AMSGrad的最大二阶矩
        }
        
        this.currentStep = 0;
        this.initialized = true;
        
        if (verbose) {
            System.out.println("在线Adam优化器已初始化，参数维度: " + initialParams.size());
        }
    }

    @Override
    public IVector step(IVector gradient) {
        if (!initialized) {
            throw new IllegalStateException("优化器未初始化，请先调用initialize()方法");
        }
        if (gradient == null) {
            throw new IllegalArgumentException("梯度不能为null");
        }
        if (gradient.size() != currentParams.size()) {
            throw new IllegalArgumentException("梯度维度与参数维度不匹配");
        }
        
        // 检查梯度是否包含无效值
        for (int i = 0; i < gradient.size(); i++) {
            double val = gradient.get(i).doubleValue();
            if (Double.isNaN(val) || Double.isInfinite(val)) {
                throw new IllegalArgumentException("梯度包含无效值 (NaN 或 Infinity)");
            }
        }
        
        currentStep++;
        
        // 更新学习率（如果设置了衰减）
        updateLearningRate();
        
        // 复制梯度以避免修改原始数据
        IVector grad = gradient.copy();
        
        // 添加权重衰减项 (L2正则化)
        if (weightDecay > 0.0) {
            grad = grad.add(currentParams.multiplyScalar(weightDecay));
        }
        
        // 更新一阶矩估计: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
        m = m.multiplyScalar(beta1).add(grad.multiplyScalar(1.0 - beta1));
        
        // 更新二阶矩估计: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
        IVector gradientSquared = grad.multiply(grad);  // 元素级平方
        v = v.multiplyScalar(beta2).add(gradientSquared.multiplyScalar(1.0 - beta2));
        
        // 偏差修正
        double beta1Correction = 1.0 - Math.pow(beta1, currentStep);
        double beta2Correction = 1.0 - Math.pow(beta2, currentStep);
        
        // 防止除零错误
        if (Math.abs(beta1Correction) < 1e-12) beta1Correction = 1e-12;
        if (Math.abs(beta2Correction) < 1e-12) beta2Correction = 1e-12;
        
        IVector mHat = m.multiplyScalar(1.0 / beta1Correction);  // 偏差修正的一阶矩
        IVector vHat = v.multiplyScalar(1.0 / beta2Correction);  // 偏差修正的二阶矩
        
        // AMSGrad变体：使用历史最大二阶矩
        IVector vForUpdate = vHat;
        if (amsgrad) {
            // vMax_t = max(vMax_{t-1}, v_t)
            if (vMax == null) {
                vMax = vHat.copy();
            } else {
                vMax = elementwiseMax(vMax, vHat);
            }
            vForUpdate = vMax;
        }
        
        // 计算更新步长: Δx = -α * m̂_t / (√v̂_t + ε)
        IVector denominator = elementwiseSqrt(vForUpdate).addScalar(epsilon);
        IVector update = mHat.divide(denominator).multiplyScalar(learningRate);
        
        // 更新参数: params = params - update
        currentParams = currentParams.sub(update);
        
        if (verbose && currentStep % 1000 == 0) {
            System.out.printf("在线Adam步骤 %d: 学习率 = %.6f\n", currentStep, learningRate);
        }
        
        return currentParams.copy();
    }

    @Override
    public IVector step(IVector gradient, double loss) {
        IVector result = step(gradient);
        
        if (verbose && currentStep % 1000 == 0) {
            System.out.printf("在线Adam步骤 %d: 损失 = %.6f, 学习率 = %.6f\n", 
                             currentStep, loss, learningRate);
        }
        
        return result;
    }

    @Override
    public IVector getCurrentParams() {
        if (!initialized) {
            throw new IllegalStateException("优化器未初始化");
        }
        return currentParams.copy();
    }

    @Override
    public void setCurrentParams(IVector params) {
        if (params == null) {
            throw new IllegalArgumentException("参数不能为null");
        }
        
        if (initialized && params.size() != currentParams.size()) {
            throw new IllegalArgumentException("参数维度不匹配");
        }
        
        this.currentParams = params.copy();
        
        // 如果优化器已初始化，需要重新初始化Adam状态
        if (initialized) {
            this.m = params.copy().multiplyScalar(0.0);
            this.v = params.copy().multiplyScalar(0.0);
            if (amsgrad) {
                this.vMax = params.copy().multiplyScalar(0.0);
            }
        }
        
        if (!initialized) {
            initialize(params);
        }
    }

    @Override
    public double getCurrentLearningRate() {
        return learningRate;
    }

    @Override
    public void setLearningRate(double learningRate) {
        if (learningRate <= 0.0 || Double.isNaN(learningRate) || Double.isInfinite(learningRate)) {
            throw new IllegalArgumentException("学习率必须大于0且为有效数值");
        }
        this.learningRate = learningRate;
    }

    @Override
    public int getCurrentStep() {
        return currentStep;
    }

    @Override
    public void reset() {
        this.currentParams = null;
        this.m = null;
        this.v = null;
        this.vMax = null;
        this.currentStep = 0;
        this.initialized = false;
        
        if (verbose) {
            System.out.println("在线Adam优化器状态已重置");
        }
    }

    @Override
    public boolean isInitialized() {
        return initialized;
    }

    @Override
    public IOnlineOptimizer clone() {
        RereOnlineAdam clone = new RereOnlineAdam(learningRate, beta1, beta2, epsilon, weightDecay);
        clone.verbose = this.verbose;
        clone.amsgrad = this.amsgrad;
        clone.lrDecayRate = this.lrDecayRate;
        clone.lrDecayStep = this.lrDecayStep;
        
        if (initialized) {
            clone.initialize(this.currentParams);
            clone.currentStep = this.currentStep;
            clone.m = this.m.copy();
            clone.v = this.v.copy();
            if (this.vMax != null) {
                clone.vMax = this.vMax.copy();
            }
        }
        
        return clone;
    }
    
    /**
     * 元素级平方根运算
     * @param vector 输入向量
     * @return 每个元素的平方根
     */
    private IVector elementwiseSqrt(IVector vector) {
        // 使用向量的通用函数接口进行元素级平方根运算
        return vector.sqrt();
    }
    
    /**
     * 元素级最大值运算
     * @param v1 向量1
     * @param v2 向量2
     * @return 每个位置的最大值
     */
    private IVector elementwiseMax(IVector v1, IVector v2) {
        // 使用循环实现元素级最大值运算
        IVector result = v1.copy();
        for (int i = 0; i < v1.size(); i++) {
            double val1 = v1.get(i).doubleValue();
            double val2 = v2.get(i).doubleValue();
            result.set(i, Math.max(val1, val2));
        }
        return result;
    }
    
    // Getter和Setter方法
    public double getBeta1() {
        return beta1;
    }
    
    public RereOnlineAdam setBeta1(double beta1) {
        if (beta1 < 0.0 || beta1 >= 1.0 || Double.isNaN(beta1) || Double.isInfinite(beta1)) {
            throw new IllegalArgumentException("beta1必须在[0, 1)范围内且为有效数值");
        }
        this.beta1 = beta1;
        return this;
    }
    
    public double getBeta2() {
        return beta2;
    }
    
    public RereOnlineAdam setBeta2(double beta2) {
        if (beta2 < 0.0 || beta2 >= 1.0 || Double.isNaN(beta2) || Double.isInfinite(beta2)) {
            throw new IllegalArgumentException("beta2必须在[0, 1)范围内且为有效数值");
        }
        this.beta2 = beta2;
        return this;
    }
    
    public double getEpsilon() {
        return epsilon;
    }
    
    public RereOnlineAdam setEpsilon(double epsilon) {
        if (epsilon <= 0.0 || Double.isNaN(epsilon) || Double.isInfinite(epsilon)) {
            throw new IllegalArgumentException("epsilon必须大于0且为有效数值");
        }
        this.epsilon = epsilon;
        return this;
    }
    
    public double getWeightDecay() {
        return weightDecay;
    }
    
    public RereOnlineAdam setWeightDecay(double weightDecay) {
        if (weightDecay < 0.0 || Double.isNaN(weightDecay) || Double.isInfinite(weightDecay)) {
            throw new IllegalArgumentException("权重衰减必须大于等于0且为有效数值");
        }
        this.weightDecay = weightDecay;
        return this;
    }
    
    public boolean isVerbose() {
        return verbose;
    }
    
    public RereOnlineAdam setVerbose(boolean verbose) {
        this.verbose = verbose;
        return this;
    }
    
    public boolean isAmsgrad() {
        return amsgrad;
    }
    
    public RereOnlineAdam setAmsgrad(boolean amsgrad) {
        this.amsgrad = amsgrad;
        // 如果已初始化且AMSGrad设置改变，需要相应调整vMax
        if (initialized) {
            if (amsgrad && this.vMax == null) {
                this.vMax = currentParams.copy().multiplyScalar(0.0);
            } else if (!amsgrad) {
                this.vMax = null;
            }
        }
        return this;
    }
    
    // 新增学习率调度方法
    public double getLrDecayRate() {
        return lrDecayRate;
    }
    
    public RereOnlineAdam setLrDecayRate(double lrDecayRate) {
        if (lrDecayRate < 0.0 || Double.isNaN(lrDecayRate) || Double.isInfinite(lrDecayRate)) {
            throw new IllegalArgumentException("学习率衰减率必须大于等于0且为有效数值");
        }
        this.lrDecayRate = lrDecayRate;
        return this;
    }
    
    public int getLrDecayStep() {
        return lrDecayStep;
    }
    
    public RereOnlineAdam setLrDecayStep(int lrDecayStep) {
        if (lrDecayStep < 0) {
            throw new IllegalArgumentException("学习率衰减步长必须大于等于0");
        }
        this.lrDecayStep = lrDecayStep;
        return this;
    }
    
    /**
     * 更新学习率（指数衰减）
     */
    private void updateLearningRate() {
        if (lrDecayRate > 0.0 && lrDecayStep > 0 && currentStep % lrDecayStep == 0) {
            learningRate *= (1.0 - lrDecayRate);
            // 确保学习率不会过小
            if (learningRate < 1e-10) {
                learningRate = 1e-10;
            }
        }
    }
}