package com.yishape.lab.math.optimize.newton;

import com.yishape.lab.util.YishapeLogger;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.IOnlineOptimizer;
import com.yishape.lab.math.util.RerePrecision;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 在线Adam优化器实现（Adaptive Moment Estimation）
 * Online Adam Optimizer Implementation - Adaptive Moment Estimation
 *
 * <p>Adam算法结合了动量法和RMSProp的优点，通过计算梯度的一阶和二阶矩估计来自适应学习率。
 * 适用于流式数据处理和增量学习场景。
 * Adam algorithm combines the advantages of momentum method and RMSProp, adaptively
 * adjusting learning rates by computing first and second moment estimates of gradients.
 * Suitable for streaming data processing and incremental learning scenarios.</p>
 *
 * <h3>算法描述 / Algorithm Description:</h3>
 * <pre>
 * m_t = β₁ * m_{t-1} + (1 - β₁) * g_t        # 一阶矩估计 (mean)
 * v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²       # 二阶矩估计 (variance)
 * m̂_t = m_t / (1 - β₁^t)                     # 偏差修正
 * v̂_t = v_t / (1 - β₂^t)                     # 偏差修正
 * x_t = x_{t-1} - α * m̂_t / (√v̂_t + ε)       # 参数更新
 * </pre>
 *
 * <h3>特点 / Features:</h3>
 * <ul>
 *   <li>自适应学习率: 对每个参数使用单独的学习率 / Adaptive learning rates: separate rate for each parameter</li>
 *   <li>动量支持: 加速收敛，减少震荡 / Momentum support: accelerates convergence, reduces oscillation</li>
 *   <li>偏差修正: 抵消初始迭代时的矩估计偏差 / Bias correction: compensates for moment estimate bias in early iterations</li>
 *   <li>AMSGrad变体: 可选的更稳定版本 / AMSGrad variant: optional more stable version</li>
 * </ul>
 *
 * @author lteb2
 * @see IOnlineOptimizer
 */
public class RereOnlineAdam implements IOnlineOptimizer {

    private static final YishapeLogger log = YishapeLogger.getLogger(RereOnlineAdam.class);


    private double learningRate = 0.001;     // 学习率 (α) / Learning rate (α)
    private double beta1 = 0.9;              // 一阶矩估计的指数衰减率 / Exponential decay rate for first moment estimate
    private double beta2 = 0.999;            // 二阶矩估计的指数衰减率 / Exponential decay rate for second moment estimate
    private double epsilon = 1e-8;           // 数值稳定性常数 / Numerical stability constant
    private double weightDecay = 0.0;        // 权重衰减 / Weight decay
    private boolean verbose = false;         // 是否输出详细信息 / Whether to output detailed information
    private boolean amsgrad = false;         // 是否使用AMSGrad变体 / Whether to use AMSGrad variant
    private boolean skipGradientValidation = false; // 是否跳过梯度有效性检查 / Whether to skip gradient validation
    private double lrDecayRate = 0.0;        // 学习率衰减率 / Learning rate decay rate
    private int lrDecayStep = 0;             // 学习率衰减步长 / Learning rate decay step

    // Adam状态变量 / Adam state variables
    private IVector currentParams = null;    // 当前参数 / Current parameters
    private IVector m = null;                // 一阶矩估计 (梯度的指数移动平均) / First moment estimate (exponential moving average of gradient)
    private IVector v = null;                // 二阶矩估计 (梯度平方的指数移动平均) / Second moment estimate (exponential moving average of gradient squared)
    private IVector vMax = null;             // AMSGrad的最大二阶矩 / Maximum second moment for AMSGrad
    private int currentStep = 0;             // 当前步数 / Current step
    private boolean initialized = false;     // 是否已初始化 / Whether initialized

    /**
     * 默认构造函数，使用标准参数
     * Default constructor with standard parameters
     *
     * <p>使用默认参数: learningRate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
     * Uses default parameters: learningRate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8</p>
     */
    public RereOnlineAdam() {
    }

    /**
     * 构造函数，仅设置学习率
     * Constructor with only learning rate
     *
     * @param learningRate 学习率，必须大于0 / Learning rate, must be greater than 0
     * @throws IllegalArgumentException 如果学习率无效 / If learning rate is invalid
     */
    public RereOnlineAdam(double learningRate) {
        this.setLearningRate(learningRate);
    }

    /**
     * 构造函数，设置学习率和衰减率
     * Constructor with learning rate and decay rates
     *
     * @param learningRate 学习率 / Learning rate
     * @param beta1 一阶矩衰减率，必须在[0, 1)范围内 / First moment decay rate, must be in [0, 1)
     * @param beta2 二阶矩衰减率，必须在[0, 1)范围内 / Second moment decay rate, must be in [0, 1)
     * @throws IllegalArgumentException 如果任何参数无效 / If any parameter is invalid
     */
    public RereOnlineAdam(double learningRate, double beta1, double beta2) {
        this.setLearningRate(learningRate);
        this.setBeta1(beta1);
        this.setBeta2(beta2);
    }

    /**
     * 完整参数构造函数
     * Full parameter constructor
     *
     * @param learningRate 学习率 / Learning rate
     * @param beta1 一阶矩衰减率 / First moment decay rate
     * @param beta2 二阶矩衰减率 / Second moment decay rate
     * @param epsilon 数值稳定性常数 / Numerical stability constant
     * @param weightDecay 权重衰减 / Weight decay
     * @throws IllegalArgumentException 如果任何参数无效 / If any parameter is invalid
     */
    public RereOnlineAdam(double learningRate, double beta1, double beta2,
                          double epsilon, double weightDecay) {
        this.setLearningRate(learningRate);
        this.setBeta1(beta1);
        this.setBeta2(beta2);
        this.setEpsilon(epsilon);
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

        // 初始化Adam状态变量
        this.m = initialParams.copy().multiplyByScalar(0.0);  // 一阶矩估计，初始化为零向量
        this.v = initialParams.copy().multiplyByScalar(0.0);  // 二阶矩估计，初始化为零向量
        if (amsgrad) {
            this.vMax = initialParams.copy().multiplyByScalar(0.0);  // AMSGrad的最大二阶矩
        }

        this.currentStep = 0;
        this.initialized = true;

        if (verbose) {
            log.debug("在线Adam优化器已初始化，参数维度: " + initialParams.size());
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

        // 检查梯度是否包含无效值（可跳过以提升性能）
        if (!skipGradientValidation) {
            for (int i = 0; i < gradient.size(); i++) {
                double val = gradient.get(i);
                if (Double.isNaN(val) || Double.isInfinite(val)) {
                    throw new IllegalArgumentException("梯度包含无效值 (NaN 或 Infinity) / Gradient contains invalid values (NaN or Infinity)");
                }
            }
        }

        currentStep++;

        // 更新学习率（如果设置了衰减）
        updateLearningRate();

        // 获取原始数组，一次性完成所有逐元素计算
        double[] p = ((com.yishape.lab.math.linalg.IDoubleVector) currentParams).getData();
        double[] mArr = ((com.yishape.lab.math.linalg.IDoubleVector) m).getData();
        double[] vArr = ((com.yishape.lab.math.linalg.IDoubleVector) v).getData();
        double[] g = ((com.yishape.lab.math.linalg.IDoubleVector) gradient).getData();
        int n = p.length;

        double[] newM = new double[n];
        double[] newV = new double[n];
        double[] newP = new double[n];
        double[] newVMax = amsgrad ? new double[n] : null;
        double[] vMaxArr = (amsgrad && vMax != null) ? ((com.yishape.lab.math.linalg.IDoubleVector) vMax).getData() : null;

        double beta1Correction = Math.max(1.0 - Math.pow(beta1, currentStep), 1e-8);
        double beta2Correction = Math.max(1.0 - Math.pow(beta2, currentStep), 1e-8);
        double invBeta1Corr = 1.0 / beta1Correction;
        double invBeta2Corr = 1.0 / beta2Correction;
        double oneMinusBeta1 = 1.0 - beta1;
        double oneMinusBeta2 = 1.0 - beta2;
        double wd = weightDecay;

        for (int i = 0; i < n; i++) {
            double gi = g[i] + wd * p[i];
            double mi = beta1 * mArr[i] + oneMinusBeta1 * gi;
            newM[i] = mi;
            double vi = beta2 * vArr[i] + oneMinusBeta2 * gi * gi;
            newV[i] = vi;
            double vForUpdateI = vi * invBeta2Corr;
            if (amsgrad) {
                double vMaxI = (vMaxArr != null) ? Math.max(vMaxArr[i], vForUpdateI) : vForUpdateI;
                newVMax[i] = vMaxI;
                vForUpdateI = vMaxI;
            }
            newP[i] = p[i] - learningRate * (mi * invBeta1Corr) / (Math.sqrt(vForUpdateI) + epsilon);
        }

        currentParams = com.yishape.lab.math.linalg.IDoubleVector.of(newP);
        m = com.yishape.lab.math.linalg.IDoubleVector.of(newM);
        v = com.yishape.lab.math.linalg.IDoubleVector.of(newV);
        if (amsgrad) {
            vMax = com.yishape.lab.math.linalg.IDoubleVector.of(newVMax);
        }

        if (verbose && currentStep % 1000 == 0) {
            log.debug(String.format("在线Adam步骤 %d: 学习率 = %.6f\n", currentStep, learningRate));
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
            log.debug(String.format("在线Adam步骤 %d: 损失 = %.6f, 学习率 = %.6f\n",
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

        // 如果优化器已初始化，需要重新初始化Adam状态
        if (initialized) {
            this.m = params.copy().multiplyByScalar(0.0);
            this.v = params.copy().multiplyByScalar(0.0);
            if (amsgrad) {
                this.vMax = params.copy().multiplyByScalar(0.0);
            }
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
        this.m = null;
        this.v = null;
        this.vMax = null;
        this.currentStep = 0;
        this.initialized = false;

        if (verbose) {
            log.debug("在线Adam优化器状态已重置 / Online Adam optimizer state has been reset");
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

    // Getter和Setter方法 / Getter and Setter methods

    /**
     * 获取一阶矩衰减率 beta1
     * Get first moment decay rate beta1
     *
     * @return 一阶矩衰减率 / First moment decay rate
     */
    public double getBeta1() {
        return beta1;
    }

    /**
     * 设置一阶矩衰减率 beta1
     * Set first moment decay rate beta1
     *
     * @param beta1 必须在[0, 1)范围内 / Must be in [0, 1)
     * @throws IllegalArgumentException 如果 beta1 不在有效范围内 / If beta1 is not in valid range
     * @return this / 返回自身以支持链式调用 / Returns this for method chaining
     */
    public RereOnlineAdam setBeta1(double beta1) {
        if (beta1 < 0.0 || beta1 >= 1.0 || Double.isNaN(beta1) || Double.isInfinite(beta1)) {
            throw new IllegalArgumentException("beta1必须在[0, 1)范围内且为有效数值 / beta1 must be in [0, 1) and be a valid number");
        }
        this.beta1 = beta1;
        return this;
    }

    /**
     * 获取二阶矩衰减率 beta2
     * Get second moment decay rate beta2
     *
     * @return 二阶矩衰减率 / Second moment decay rate
     */
    public double getBeta2() {
        return beta2;
    }

    /**
     * 设置二阶矩衰减率 beta2
     * Set second moment decay rate beta2
     *
     * @param beta2 必须在[0, 1)范围内 / Must be in [0, 1)
     * @throws IllegalArgumentException 如果 beta2 不在有效范围内 / If beta2 is not in valid range
     * @return this / 返回自身以支持链式调用 / Returns this for method chaining
     */
    public RereOnlineAdam setBeta2(double beta2) {
        if (beta2 < 0.0 || beta2 >= 1.0 || Double.isNaN(beta2) || Double.isInfinite(beta2)) {
            throw new IllegalArgumentException("beta2必须在[0, 1)范围内且为有效数值 / beta2 must be in [0, 1) and be a valid number");
        }
        this.beta2 = beta2;
        return this;
    }

    /**
     * 获取数值稳定性常数 epsilon
     * Get numerical stability constant epsilon
     *
     * @return 数值稳定性常数 / Numerical stability constant
     */
    public double getEpsilon() {
        return epsilon;
    }

    /**
     * 设置数值稳定性常数 epsilon
     * Set numerical stability constant epsilon
     *
     * @param epsilon 必须大于0 / Must be greater than 0
     * @throws IllegalArgumentException 如果 epsilon 无效 / If epsilon is invalid
     * @return this / 返回自身以支持链式调用 / Returns this for method chaining
     */
    public RereOnlineAdam setEpsilon(double epsilon) {
        if (epsilon <= 0.0 || Double.isNaN(epsilon) || Double.isInfinite(epsilon)) {
            throw new IllegalArgumentException("epsilon必须大于0且为有效数值 / epsilon must be greater than 0 and a valid number");
        }
        this.epsilon = epsilon;
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
     * @param weightDecay 必须大于等于0 / Must be greater than or equal to 0
     * @throws IllegalArgumentException 如果 weightDecay 无效 / If weightDecay is invalid
     * @return this / 返回自身以支持链式调用 / Returns this for method chaining
     */
    public RereOnlineAdam setWeightDecay(double weightDecay) {
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
    public RereOnlineAdam setVerbose(boolean verbose) {
        this.verbose = verbose;
        return this;
    }

    /**
     * 检查是否使用AMSGrad变体
     * Check if AMSGrad variant is enabled
     *
     * @return 是否使用AMSGrad / Whether AMSGrad is used
     */
    public boolean isAmsgrad() {
        return amsgrad;
    }

    /**
     * 设置是否使用AMSGrad变体
     * Set whether to use AMSGrad variant
     *
     * <p>AMSGrad是一种更稳定的Adam变体，使用历史最大二阶矩而不是当前二阶矩。
     * AMSGrad is a more stable variant of Adam that uses the maximum of historical
     * second moments rather than the current second moment.</p>
     *
     * @param amsgrad 是否使用AMSGrad / Whether to use AMSGrad
     * @return this / 返回自身以支持链式调用 / Returns this for method chaining
     */
    public RereOnlineAdam setAmsgrad(boolean amsgrad) {
        this.amsgrad = amsgrad;
        // 如果已初始化且AMSGrad设置改变，需要相应调整vMax
        if (initialized) {
            if (amsgrad && this.vMax == null) {
                this.vMax = currentParams.copy().multiplyByScalar(0.0);
            } else if (!amsgrad) {
                this.vMax = null;
            }
        }
        return this;
    }

    /**
     * 获取是否跳过梯度有效性检查
     * Get whether to skip gradient validation
     *
     * @return 是否跳过梯度检查 / Whether to skip gradient validation
     */
    public boolean isSkipGradientValidation() {
        return skipGradientValidation;
    }

    /**
     * 设置是否跳过梯度有效性检查
     * Set whether to skip gradient validation
     *
     * <p>跳过检查可提升大规模场景下的性能，但需确保输入梯度可靠。
     * Skipping validation improves performance in large-scale scenarios,
     * but input gradients must be reliable.</p>
     *
     * @param skipGradientValidation 是否跳过梯度检查 / Whether to skip gradient validation
     * @return this / 返回自身以支持链式调用 / Returns this for method chaining
     */
    public RereOnlineAdam setSkipGradientValidation(boolean skipGradientValidation) {
        this.skipGradientValidation = skipGradientValidation;
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
     * @param lrDecayRate 必须大于等于0 / Must be greater than or equal to 0
     * @throws IllegalArgumentException 如果衰减率无效 / If decay rate is invalid
     * @return this / 返回自身以支持链式调用 / Returns this for method chaining
     */
    public RereOnlineAdam setLrDecayRate(double lrDecayRate) {
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
     * @param lrDecayStep 必须大于等于0 / Must be greater than or equal to 0
     * @throws IllegalArgumentException 如果步长无效 / If step is invalid
     * @return this / 返回自身以支持链式调用 / Returns this for method chaining
     */
    public RereOnlineAdam setLrDecayStep(int lrDecayStep) {
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

    @Override
    public Map<String, double[]> optimizerStateDict() {
        Map<String, double[]> state = new LinkedHashMap<>();
        state.put("m", toPrimitiveArray(m));
        state.put("v", toPrimitiveArray(v));
        state.put("step", new double[]{currentStep});
        if (amsgrad && vMax != null) {
            state.put("v_max", toPrimitiveArray(vMax));
        }
        return state;
    }

    @Override
    public void loadOptimizerStateDict(Map<String, double[]> stateDict) {
        double[] mData = stateDict.get("m");
        double[] vData = stateDict.get("v");
        double[] stepData = stateDict.get("step");
        double[] vMaxData = stateDict.get("v_max");
        if (mData != null && m == null) m = currentParams.copy().multiplyByScalar(0.0);
        if (mData != null) setVectorData(m, mData);
        if (vData != null) setVectorData(v, vData);
        if (stepData != null && stepData.length > 0) this.currentStep = (int) stepData[0];
        if (amsgrad && vMaxData != null) {
            if (vMax == null) vMax = currentParams.copy().multiplyByScalar(0.0);
            setVectorData(vMax, vMaxData);
        }
    }

    private static double[] toPrimitiveArray(IVector vec) {
        if (vec == null) return new double[0];
        double[] arr = new double[vec.size()];
        for (int i = 0; i < vec.size(); i++) arr[i] = vec.get(i);
        return arr;
    }

    private static void setVectorData(IVector vec, double[] data) {
        if (vec == null || data == null) return;
        int n = Math.min(vec.size(), data.length);
        for (int i = 0; i < n; i++) vec.set(i, data[i]);
    }
}