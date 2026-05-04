package com.yishape.lab.math.optimize.newton;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.IOnlineOptimizer;
import com.yishape.lab.math.util.RerePrecision;
import java.util.function.BiFunction;

/**
 * 在线随机梯度下降优化器，支持流式数据和增量学习
 * Online Stochastic Gradient Descent Optimizer for Streaming Data and Incremental Learning
 *
 * <p>实现在线学习版本的SGD算法，支持逐步接收梯度并更新参数。
 * 适用于流式数据处理、在线学习和增量学习场景。
 * Implements the online learning version of SGD algorithm, supporting gradual gradient
 * reception and parameter updates. Suitable for streaming data processing, online learning,
 * and incremental learning scenarios.</p>
 *
 * <h3>算法描述 / Algorithm Description:</h3>
 * <pre>
 * v_t = momentum * v_{t-1} + gradient
 * x_t = x_{t-1} - learningRate * v_t
 * </pre>
 *
 * <h3>特点 / Features:</h3>
 * <ul>
 *   <li>支持动量: 加速收敛，减少震荡 / Momentum support: accelerates convergence, reduces oscillation</li>
 *   <li>支持权重衰减: 防止过拟合 / Weight decay support: prevents overfitting</li>
 *   <li>支持学习率衰减: 自适应调整学习率 / Learning rate decay: adaptive learning rate adjustment</li>
 *   <li>流式更新: 无需保存完整数据集 / Streaming updates: no need to store complete dataset</li>
 * </ul>
 *
 * @author lteb2
 * @see IOnlineOptimizer
 */
public class RereOnlineSGD implements IOnlineOptimizer {

    private static final Logger log = LoggerFactory.getLogger(RereOnlineSGD.class);


    private double learningRate = 0.01;      // 学习率 / Learning rate
    private double momentum = 0.0;           // 动量系数 / Momentum coefficient
    private double weightDecay = 0.0;        // 权重衰减 / Weight decay
    private boolean verbose = false;         // 是否输出详细信息 / Whether to output detailed information
    private double lrDecayRate = 0.0;        // 学习率衰减率 / Learning rate decay rate
    private int lrDecayStep = 0;             // 学习率衰减步长 / Learning rate decay step

    // 状态变量 / State variables
    private IVector currentParams = null;    // 当前参数 / Current parameters
    private IVector momentumBuffer = null;   // 动量缓存 / Momentum buffer
    private int currentStep = 0;             // 当前步数 / Current step
    private boolean initialized = false;     // 是否已初始化 / Whether initialized

    /**
     * 默认构造函数，使用标准参数
     * Default constructor with standard parameters
     *
     * <p>使用默认参数: learningRate=0.01, momentum=0.0, weightDecay=0.0
     * Uses default parameters: learningRate=0.01, momentum=0.0, weightDecay=0.0</p>
     */
    public RereOnlineSGD() {
    }

    /**
     * 构造函数，仅设置学习率
     * Constructor with only learning rate
     *
     * @param learningRate 学习率，必须大于0 / Learning rate, must be greater than 0
     * @throws IllegalArgumentException 如果学习率无效 / If learning rate is invalid
     */
    public RereOnlineSGD(double learningRate) {
        this.setLearningRate(learningRate);
    }

    /**
     * 构造函数，设置学习率和动量
     * Constructor with learning rate and momentum
     *
     * @param learningRate 学习率，必须大于0 / Learning rate, must be greater than 0
     * @param momentum 动量系数，默认为0 / Momentum coefficient, default 0
     * @throws IllegalArgumentException 如果任何参数无效 / If any parameter is invalid
     */
    public RereOnlineSGD(double learningRate, double momentum) {
        this.setLearningRate(learningRate);
        this.setMomentum(momentum);
    }

    /**
     * 完整参数构造函数
     * Full parameter constructor
     *
     * @param learningRate 学习率，必须大于0 / Learning rate, must be greater than 0
     * @param momentum 动量系数 / Momentum coefficient
     * @param weightDecay 权重衰减 / Weight decay
     * @throws IllegalArgumentException 如果任何参数无效 / If any parameter is invalid
     */
    public RereOnlineSGD(double learningRate, double momentum, double weightDecay) {
        this.setLearningRate(learningRate);
        this.setMomentum(momentum);
        this.setWeightDecay(weightDecay);
    }

    /**
     * 初始化优化器状态
     * Initialize optimizer state
     *
     * @param initialParams 初始参数向量 / Initial parameter vector
     * @throws IllegalArgumentException 如果 initialParams 为 null / If initialParams is null
     */
    @Override
    public void initialize(IVector initialParams) {
        if (initialParams == null) {
            throw new IllegalArgumentException("初始参数不能为null / Initial parameters cannot be null");
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

    /**
     * 执行一步优化更新
     * Perform one optimization step
     *
     * @param gradient 梯度向量 / Gradient vector
     * @return 更新后的参数向量 / Updated parameter vector
     * @throws IllegalStateException 如果优化器未初始化 / If optimizer is not initialized
     * @throws IllegalArgumentException 如果梯度为 null 或维度不匹配 / If gradient is null or dimension mismatch
     */
    @Override
    public IVector step(IVector gradient) {
        if (!initialized) {
            throw new IllegalStateException("优化器未初始化，请先调用initialize()方法 / Optimizer not initialized, please call initialize() first");
        }
        if (gradient == null) {
            throw new IllegalArgumentException("梯度不能为null / Gradient cannot be null");
        }
        if (gradient.size() != currentParams.size()) {
            throw new IllegalArgumentException("梯度维度与参数维度不匹配 / Gradient dimension does not match parameter dimension");
        }

        // 检查梯度是否包含无效值
        for (int i = 0; i < gradient.size(); i++) {
            double val = gradient.get(i).doubleValue();
            if (Double.isNaN(val) || Double.isInfinite(val)) {
                throw new IllegalArgumentException("梯度包含无效值 (NaN 或 Infinity) / Gradient contains invalid values (NaN or Infinity)");
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

    /**
     * 执行一步优化更新并记录损失值
     * Perform one optimization step with loss value recording
     *
     * @param gradient 梯度向量 / Gradient vector
     * @param loss 当前损失值（仅用于日志记录）/ Current loss value (for logging only)
     * @return 更新后的参数向量 / Updated parameter vector
     * @throws IllegalStateException 如果优化器未初始化 / If optimizer is not initialized
     * @throws IllegalArgumentException 如果梯度为 null / If gradient is null
     */
    @Override
    public IVector step(IVector gradient, double loss) {
        IVector result = step(gradient);

        if (verbose && currentStep % 1000 == 0) {
            log.debug(String.format("在线SGD步骤 %d: 损失 = %.6f, 学习率 = %.6f\n",
                             currentStep, loss, learningRate));
        }

        return result;
    }

    /**
     * 获取当前参数向量
     * Get current parameter vector
     *
     * @return 当前参数向量的副本 / Copy of current parameter vector
     * @throws IllegalStateException 如果优化器未初始化 / If optimizer is not initialized
     */
    @Override
    public IVector getCurrentParams() {
        if (!initialized) {
            throw new IllegalStateException("优化器未初始化 / Optimizer not initialized");
        }
        return currentParams.copy();
    }

    /**
     * 设置当前参数向量
     * Set current parameter vector
     *
     * @param params 新的参数向量 / New parameter vector
     * @throws IllegalArgumentException 如果 params 为 null 或维度不匹配 / If params is null or dimension mismatch
     */
    @Override
    public void setCurrentParams(IVector params) {
        if (params == null) {
            throw new IllegalArgumentException("参数不能为null / Parameters cannot be null");
        }

        if (initialized && params.size() != currentParams.size()) {
            throw new IllegalArgumentException("参数维度不匹配 / Parameter dimension mismatch");
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

    /**
     * 获取当前学习率
     * Get current learning rate
     *
     * @return 当前学习率 / Current learning rate
     */
    @Override
    public double getCurrentLearningRate() {
        return learningRate;
    }

    /**
     * 设置学习率
     * Set learning rate
     *
     * @param learningRate 学习率，必须大于0且为有效数值 / Learning rate, must be greater than 0 and a valid number
     * @throws IllegalArgumentException 如果学习率无效 / If learning rate is invalid
     */
    @Override
    public void setLearningRate(double learningRate) {
        if (learningRate <= 0.0 || Double.isNaN(learningRate) || Double.isInfinite(learningRate)) {
            throw new IllegalArgumentException("学习率必须大于0且为有效数值 / Learning rate must be greater than 0 and a valid number");
        }
        this.learningRate = learningRate;
    }

    /**
     * 获取当前步数
     * Get current step number
     *
     * @return 当前步数 / Current step number
     */
    @Override
    public int getCurrentStep() {
        return currentStep;
    }

    /**
     * 重置优化器状态
     * Reset optimizer state
     *
     * <p>调用此方法后，需要重新调用 initialize() 才能继续使用优化器。
     * After calling this method, initialize() must be called again to continue using the optimizer.</p>
     */
    @Override
    public void reset() {
        this.currentParams = null;
        this.momentumBuffer = null;
        this.currentStep = 0;
        this.initialized = false;

        if (verbose) {
            log.debug("在线SGD优化器状态已重置 / Online SGD optimizer state has been reset");
        }
    }

    /**
     * 检查优化器是否已初始化
     * Check if optimizer is initialized
     *
     * @return 如果已初始化返回 true，否则返回 false / Returns true if initialized, false otherwise
     */
    @Override
    public boolean isInitialized() {
        return initialized;
    }

    /**
     * 创建优化器的深拷贝
     * Create a deep copy of the optimizer
     *
     * @return 优化器的深拷贝 / Deep copy of the optimizer
     */
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

    // Getter和Setter方法 / Getter and Setter methods

    /**
     * 获取动量系数
     * Get momentum coefficient
     *
     * @return 动量系数 / Momentum coefficient
     */
    public double getMomentum() {
        return momentum;
    }

    /**
     * 设置动量系数
     * Set momentum coefficient
     *
     * @param momentum 动量系数，必须是有效数值 / Momentum coefficient, must be a valid number
     * @throws IllegalArgumentException 如果动量系数无效 / If momentum coefficient is invalid
     * @return this / 返回自身以支持链式调用 / Returns this for method chaining
     */
    public RereOnlineSGD setMomentum(double momentum) {
        if (Double.isNaN(momentum) || Double.isInfinite(momentum)) {
            throw new IllegalArgumentException("动量必须为有效数值 / Momentum must be a valid number");
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

    /**
     * 获取权重衰减系数
     * Get weight decay coefficient
     *
     * @return 权重衰减系数 / Weight decay coefficient
     */
    public double getWeightDecay() {
        return weightDecay;
    }

    /**
     * 设置权重衰减系数
     * Set weight decay coefficient
     *
     * @param weightDecay 权重衰减系数，必须大于等于0 / Weight decay coefficient, must be greater than or equal to 0
     * @throws IllegalArgumentException 如果权重衰减系数无效 / If weight decay coefficient is invalid
     * @return this / 返回自身以支持链式调用 / Returns this for method chaining
     */
    public RereOnlineSGD setWeightDecay(double weightDecay) {
        if (weightDecay < 0.0 || Double.isNaN(weightDecay) || Double.isInfinite(weightDecay)) {
            throw new IllegalArgumentException("权重衰减必须大于等于0且为有效数值 / Weight decay must be greater than or equal to 0 and a valid number");
        }
        this.weightDecay = weightDecay;
        return this;
    }

    /**
     * 检查是否启用详细输出模式
     * Check if verbose mode is enabled
     *
     * @return 是否启用详细输出 / Whether verbose output is enabled
     */
    public boolean isVerbose() {
        return verbose;
    }

    /**
     * 设置详细输出模式
     * Set verbose mode
     *
     * @param verbose 是否启用详细输出 / Whether to enable verbose output
     * @return this / 返回自身以支持链式调用 / Returns this for method chaining
     */
    public RereOnlineSGD setVerbose(boolean verbose) {
        this.verbose = verbose;
        return this;
    }

    /**
     * 获取学习率衰减率
     * Get learning rate decay rate
     *
     * @return 学习率衰减率 / Learning rate decay rate
     */
    public double getLrDecayRate() {
        return lrDecayRate;
    }

    /**
     * 设置学习率衰减率
     * Set learning rate decay rate
     *
     * @param lrDecayRate 学习率衰减率，必须大于等于0 / Decay rate, must be greater than or equal to 0
     * @throws IllegalArgumentException 如果衰减率无效 / If decay rate is invalid
     * @return this / 返回自身以支持链式调用 / Returns this for method chaining
     */
    public RereOnlineSGD setLrDecayRate(double lrDecayRate) {
        if (lrDecayRate < 0.0 || Double.isNaN(lrDecayRate) || Double.isInfinite(lrDecayRate)) {
            throw new IllegalArgumentException("学习率衰减率必须大于等于0且为有效数值 / Learning rate decay rate must be greater than or equal to 0 and a valid number");
        }
        this.lrDecayRate = lrDecayRate;
        return this;
    }

    /**
     * 获取学习率衰减步长
     * Get learning rate decay step
     *
     * @return 学习率衰减步长 / Learning rate decay step
     */
    public int getLrDecayStep() {
        return lrDecayStep;
    }

    /**
     * 设置学习率衰减步长
     * Set learning rate decay step
     *
     * @param lrDecayStep 衰减步长，必须大于等于0 / Decay step, must be greater than or equal to 0
     * @throws IllegalArgumentException 如果步长无效 / If step is invalid
     * @return this / 返回自身以支持链式调用 / Returns this for method chaining
     */
    public RereOnlineSGD setLrDecayStep(int lrDecayStep) {
        if (lrDecayStep < 0) {
            throw new IllegalArgumentException("学习率衰减步长必须大于等于0 / Learning rate decay step must be greater than or equal to 0");
        }
        this.lrDecayStep = lrDecayStep;
        return this;
    }

    /**
     * 更新学习率（指数衰减）
     * Update learning rate with exponential decay
     *
     * <p>当设置的学习率衰减率和步长都大于0时，每隔 lrDecayStep 步学习率会乘以 (1 - lrDecayRate)。
     * When both lrDecayRate and lrDecayStep are set to values greater than 0,
     * the learning rate is multiplied by (1 - lrDecayRate) every lrDecayStep steps.</p>
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