package com.yishape.lab.math.optimize.newton;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.IOnlineOptimizer;
import com.yishape.lab.math.util.RerePrecision;
import java.util.function.BiFunction;

/**
 * 在线随机梯度下降优化器
 * Online Stochastic Gradient Descent Optimizer
 * 
 * 实现在线学习版本的SGD算法，支持逐步接收梯度并更新参数。
 * 适用于流式数据处理和增量学习场景。
 * 
 * @author lteb2
 */
public class RereOnlineSGD implements IOnlineOptimizer {

    private static final Logger log = LoggerFactory.getLogger(RereOnlineSGD.class);

    
    private double learningRate = 0.01;      // 学习率
    private double momentum = 0.0;           // 动量系数
    private double weightDecay = 0.0;        // 权重衰减
    private boolean verbose = false;         // 是否输出详细信息
    private double lrDecayRate = 0.0;        // 学习率衰减率
    private int lrDecayStep = 0;             // 学习率衰减步长
    
    // 状态变量
    private IVector currentParams = null;    // 当前参数
    private IVector momentumBuffer = null;   // 动量缓存
    private int currentStep = 0;             // 当前步数
    private boolean initialized = false;     // 是否已初始化
    
    /**
     * 默认构造函数
     */
    public RereOnlineSGD() {
    }
    
    /**
     * 构造函数
     * @param learningRate 学习率
     */
    public RereOnlineSGD(double learningRate) {
        this.setLearningRate(learningRate);
    }
    
    /**
     * 构造函数
     * @param learningRate 学习率
     * @param momentum 动量系数
     */
    public RereOnlineSGD(double learningRate, double momentum) {
        this.setLearningRate(learningRate);
        this.setMomentum(momentum);
    }
    
    /**
     * 完整参数构造函数
     * @param learningRate 学习率
     * @param momentum 动量系数
     * @param weightDecay 权重衰减
     */
    public RereOnlineSGD(double learningRate, double momentum, double weightDecay) {
        this.setLearningRate(learningRate);
        this.setMomentum(momentum);
        this.setWeightDecay(weightDecay);
    }

    @Override
    public void initialize(IVector initialParams) {
        if (initialParams == null) {
            throw new IllegalArgumentException("初始参数不能为null");
        }
        
        this.currentParams = initialParams.copy();
        
        // 初始化动量缓存
        if (momentum > 0.0) {
            this.momentumBuffer = initialParams.copy().multiplyScalar(0.0);
        }
        
        this.currentStep = 0;
        this.initialized = true;
        
        if (verbose) {
            log.debug("在线SGD优化器已初始化，参数维度: " + initialParams.size());
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
        
        // 添加权重衰减项
        if (weightDecay > 0.0) {
            grad = grad.add(currentParams.multiplyScalar(weightDecay));
        }
        
        // 应用动量
        IVector update;
        if (momentum > 0.0 && momentumBuffer != null) {
            // v = momentum * v + gradient
            momentumBuffer = momentumBuffer.multiplyScalar(momentum).add(grad);
            update = momentumBuffer.copy();
        } else {
            update = grad;
        }
        
        // 更新参数: params = params - learningRate * update
        currentParams = currentParams.sub(update.multiplyScalar(learningRate));
        
        if (verbose && currentStep % 1000 == 0) {
            log.debug(String.format("在线SGD步骤 %d: 学习率 = %.6f\n", currentStep, learningRate));
        }
        
        return currentParams.copy();
    }

    @Override
    public IVector step(IVector gradient, double loss) {
        IVector result = step(gradient);
        
        if (verbose && currentStep % 1000 == 0) {
            log.debug(String.format("在线SGD步骤 %d: 损失 = %.6f, 学习率 = %.6f\n",
                             currentStep, loss, learningRate));
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
        
        // 如果使用动量且优化器已初始化，需要重新初始化动量缓存
        if (initialized && momentum > 0.0) {
            this.momentumBuffer = params.copy().multiplyScalar(0.0);
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
        this.momentumBuffer = null;
        this.currentStep = 0;
        this.initialized = false;
        
        if (verbose) {
            log.debug("在线SGD优化器状态已重置");
        }
    }

    @Override
    public boolean isInitialized() {
        return initialized;
    }

    @Override
    public IOnlineOptimizer clone() {
        RereOnlineSGD clone = new RereOnlineSGD(learningRate, momentum, weightDecay);
        clone.verbose = this.verbose;
        clone.lrDecayRate = this.lrDecayRate;
        clone.lrDecayStep = this.lrDecayStep;
        
        if (initialized) {
            clone.initialize(this.currentParams);
            clone.currentStep = this.currentStep;
            if (this.momentumBuffer != null) {
                clone.momentumBuffer = this.momentumBuffer.copy();
            }
        }
        
        return clone;
    }
    
    // Getter和Setter方法
    public double getMomentum() {
        return momentum;
    }
    
    public RereOnlineSGD setMomentum(double momentum) {
        if (Double.isNaN(momentum) || Double.isInfinite(momentum)) {
            throw new IllegalArgumentException("动量必须为有效数值");
        }
        this.momentum = momentum;
        // 如果已初始化且动量参数改变，需要重新初始化动量缓存
        if (initialized) {
            if (momentum > 0.0 && this.momentumBuffer == null) {
                this.momentumBuffer = currentParams.copy().multiplyScalar(0.0);
            } else if (momentum <= 0.0) {
                this.momentumBuffer = null;
            }
        }
        return this;
    }
    
    public double getWeightDecay() {
        return weightDecay;
    }
    
    public RereOnlineSGD setWeightDecay(double weightDecay) {
        if (weightDecay < 0.0 || Double.isNaN(weightDecay) || Double.isInfinite(weightDecay)) {
            throw new IllegalArgumentException("权重衰减必须大于等于0且为有效数值");
        }
        this.weightDecay = weightDecay;
        return this;
    }
    
    public boolean isVerbose() {
        return verbose;
    }
    
    public RereOnlineSGD setVerbose(boolean verbose) {
        this.verbose = verbose;
        return this;
    }
    
    // 新增学习率调度方法
    public double getLrDecayRate() {
        return lrDecayRate;
    }
    
    public RereOnlineSGD setLrDecayRate(double lrDecayRate) {
        if (lrDecayRate < 0.0 || Double.isNaN(lrDecayRate) || Double.isInfinite(lrDecayRate)) {
            throw new IllegalArgumentException("学习率衰减率必须大于等于0且为有效数值");
        }
        this.lrDecayRate = lrDecayRate;
        return this;
    }
    
    public int getLrDecayStep() {
        return lrDecayStep;
    }
    
    public RereOnlineSGD setLrDecayStep(int lrDecayStep) {
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
        if (RerePrecision.compareTo(lrDecayRate, 0.0, 1e-12) > 0 && lrDecayStep > 0 && currentStep % lrDecayStep == 0) {
            learningRate *= (1.0 - lrDecayRate);
            // 确保学习率不会过小
            if (RerePrecision.compareTo(learningRate, 1e-10, 1e-12) < 0) {
                learningRate = 1e-10;
            }
        }
    }
}