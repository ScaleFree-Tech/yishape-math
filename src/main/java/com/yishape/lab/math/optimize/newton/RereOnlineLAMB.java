package com.yishape.lab.math.optimize.newton;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.IOnlineOptimizer;
import com.yishape.lab.util.YishapeLogger;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Online LAMB optimizer (Layer-wise Adaptive Moments optimizer for Batch training).
 *
 * <p>LAMB extends Adam with layer-wise trust ratio scaling, enabling stable training
 * with very large batch sizes (up to 65536). Each layer's update is scaled by the
 * ratio of the parameter norm to the update norm.</p>
 *
 * <pre>
 * m_t = β₁ * m_{t-1} + (1-β₁) * g_t                    # first moment
 * v_t = β₂ * v_{t-1} + (1-β₂) * g_t²                   # second moment
 * m̂_t = m_t / (1-β₁^t)                                 # bias correction
 * v̂_t = v_t / (1-β₂^t)                                 # bias correction
 * u_t = m̂_t / (√v̂_t + ε) + λ * θ_t                    # Adam update + weight decay
 * r_t = ‖θ_t‖₂ / ‖u_t‖₂                               # layer-wise trust ratio
 * θ_t = θ_{t-1} - α * r_t * u_t                         # parameter update
 * </pre>
 *
 * <h3>Features:</h3>
 * <ul>
 *   <li>Layer-wise adaptive learning rates via trust ratio</li>
 *   <li>Stable with large batch sizes (65536+)</li>
 *   <li>Decoupled weight decay (like AdamW)</li>
 *   <li>Backward-compatible with Adam when trust ratio = 1</li>
 * </ul>
 *
 * @see IOnlineOptimizer
 */
public class RereOnlineLAMB implements IOnlineOptimizer {

    private static final YishapeLogger log = YishapeLogger.getLogger(RereOnlineLAMB.class);

    private double learningRate = 0.001;
    private double beta1 = 0.9;
    private double beta2 = 0.999;
    private double epsilon = 1e-6;
    private double weightDecay = 0.01;
    private boolean verbose = false;
    private boolean skipGradientValidation = false;

    private IVector currentParams = null;
    private IVector m = null;
    private IVector v = null;
    private int currentStep = 0;
    private boolean initialized = false;

    public RereOnlineLAMB() {}

    public RereOnlineLAMB(double learningRate) {
        setLearningRate(learningRate);
    }

    public RereOnlineLAMB(double learningRate, double beta1, double beta2) {
        setLearningRate(learningRate);
        setBeta1(beta1);
        setBeta2(beta2);
    }

    public RereOnlineLAMB(double learningRate, double beta1, double beta2, double weightDecay) {
        setLearningRate(learningRate);
        setBeta1(beta1);
        setBeta2(beta2);
        setWeightDecay(weightDecay);
    }

    public RereOnlineLAMB(double learningRate, double beta1, double beta2, double epsilon, double weightDecay) {
        setLearningRate(learningRate);
        setBeta1(beta1);
        setBeta2(beta2);
        setEpsilon(epsilon);
        setWeightDecay(weightDecay);
    }

    @Override
    public void initialize(IVector initialParams) {
        if (initialParams == null) {
            throw new IllegalArgumentException("初始参数不能为null");
        }
        this.currentParams = initialParams.copy();
        this.m = initialParams.copy().multiplyByScalar(0.0);
        this.v = initialParams.copy().multiplyByScalar(0.0);
        this.currentStep = 0;
        this.initialized = true;
        if (verbose) {
            log.debug("LAMB 已初始化，参数维度: {}", initialParams.size());
        }
    }

    @Override
    public IVector step(IVector gradient) {
        if (!initialized) {
            throw new IllegalStateException("优化器未初始化，请先调用 initialize()");
        }
        if (gradient == null) {
            throw new IllegalArgumentException("梯度不能为null");
        }
        if (gradient.size() != currentParams.size()) {
            throw new IllegalArgumentException("梯度维度不匹配");
        }
        if (!skipGradientValidation) {
            for (int i = 0; i < gradient.size(); i++) {
                double val = gradient.get(i);
                if (Double.isNaN(val) || Double.isInfinite(val)) {
                    throw new IllegalArgumentException("梯度包含无效值 (NaN 或 Infinity)");
                }
            }
        }

        currentStep++;

        double[] p = ((com.yishape.lab.math.linalg.IDoubleVector) currentParams).getData();
        double[] mArr = ((com.yishape.lab.math.linalg.IDoubleVector) m).getData();
        double[] vArr = ((com.yishape.lab.math.linalg.IDoubleVector) v).getData();
        double[] g = ((com.yishape.lab.math.linalg.IDoubleVector) gradient).getData();
        int n = p.length;

        double[] newM = new double[n];
        double[] newV = new double[n];
        double[] uArr = new double[n];
        double[] newP = new double[n];

        double bc1 = Math.max(1.0 - Math.pow(beta1, currentStep), 1e-12);
        double bc2 = Math.max(1.0 - Math.pow(beta2, currentStep), 1e-12);
        double invBc1 = 1.0 / bc1;
        double invBc2 = 1.0 / bc2;
        double oneMinusBeta1 = 1.0 - beta1;
        double oneMinusBeta2 = 1.0 - beta2;
        double wd = weightDecay;
        double eps = epsilon;
        double lr = learningRate;

        double paramNormSq = 0.0;
        double updateNormSq = 0.0;
        for (int i = 0; i < n; i++) {
            double mi = beta1 * mArr[i] + oneMinusBeta1 * g[i];
            newM[i] = mi;
            double vi = beta2 * vArr[i] + oneMinusBeta2 * g[i] * g[i];
            newV[i] = vi;
            double ui = mi * invBc1 / (Math.sqrt(vi * invBc2) + eps) + wd * p[i];
            uArr[i] = ui;
            paramNormSq += p[i] * p[i];
            updateNormSq += ui * ui;
        }

        double trustRatio = Math.sqrt(paramNormSq) / (Math.sqrt(updateNormSq) + 1e-12);

        for (int i = 0; i < n; i++) {
            newP[i] = p[i] - lr * trustRatio * uArr[i];
        }

        currentParams = com.yishape.lab.math.linalg.IDoubleVector.of(newP);
        m = com.yishape.lab.math.linalg.IDoubleVector.of(newM);
        v = com.yishape.lab.math.linalg.IDoubleVector.of(newV);

        if (verbose && currentStep % 1000 == 0) {
            log.debug("LAMB step {}: lr={}, trustRatio={}", currentStep, learningRate, trustRatio);
        }

        return currentParams.copy();
    }

    @Override
    public IVector step(IVector gradient, double loss) {
        return step(gradient);
    }

    @Override
    public IVector getCurrentParams() {
        if (!initialized) throw new IllegalStateException("优化器未初始化");
        return currentParams.copy();
    }

    @Override
    public void setCurrentParams(IVector params) {
        if (params == null) throw new IllegalArgumentException("参数不能为null");
        this.currentParams = params.copy();
        if (initialized) {
            this.m = params.copy().multiplyByScalar(0.0);
            this.v = params.copy().multiplyByScalar(0.0);
        }
        if (!initialized) initialize(params);
    }

    @Override
    public double getCurrentLearningRate() { return learningRate; }

    @Override
    public void setLearningRate(double learningRate) {
        if (learningRate <= 0.0 || Double.isNaN(learningRate) || Double.isInfinite(learningRate)) {
            throw new IllegalArgumentException("学习率必须大于0");
        }
        this.learningRate = learningRate;
    }

    @Override
    public int getCurrentStep() { return currentStep; }

    @Override
    public void reset() {
        this.currentParams = null;
        this.m = null;
        this.v = null;
        this.currentStep = 0;
        this.initialized = false;
    }

    @Override
    public boolean isInitialized() { return initialized; }

    @Override
    public IOnlineOptimizer clone() {
        RereOnlineLAMB c = new RereOnlineLAMB(learningRate, beta1, beta2, epsilon, weightDecay);
        c.verbose = this.verbose;
        c.skipGradientValidation = this.skipGradientValidation;
        if (initialized) {
            c.currentParams = this.currentParams.copy();
            c.m = this.m.copy();
            c.v = this.v.copy();
            c.currentStep = this.currentStep;
            c.initialized = true;
        }
        return c;
    }

    // ---- getters/setters ----

    public double getBeta1() { return beta1; }
    public RereOnlineLAMB setBeta1(double beta1) {
        if (beta1 < 0.0 || beta1 >= 1.0) throw new IllegalArgumentException("beta1 必须在 [0, 1)");
        this.beta1 = beta1;
        return this;
    }

    public double getBeta2() { return beta2; }
    public RereOnlineLAMB setBeta2(double beta2) {
        if (beta2 < 0.0 || beta2 >= 1.0) throw new IllegalArgumentException("beta2 必须在 [0, 1)");
        this.beta2 = beta2;
        return this;
    }

    public double getEpsilon() { return epsilon; }
    public RereOnlineLAMB setEpsilon(double epsilon) {
        if (epsilon <= 0.0) throw new IllegalArgumentException("epsilon 必须大于0");
        this.epsilon = epsilon;
        return this;
    }

    public double getWeightDecay() { return weightDecay; }
    public RereOnlineLAMB setWeightDecay(double weightDecay) {
        if (weightDecay < 0.0) throw new IllegalArgumentException("权重衰减必须 >= 0");
        this.weightDecay = weightDecay;
        return this;
    }

    public boolean isVerbose() { return verbose; }
    public RereOnlineLAMB setVerbose(boolean verbose) { this.verbose = verbose; return this; }

    public boolean isSkipGradientValidation() { return skipGradientValidation; }
    public RereOnlineLAMB setSkipGradientValidation(boolean skip) { this.skipGradientValidation = skip; return this; }

    @Override
    public Map<String, double[]> optimizerStateDict() {
        Map<String, double[]> state = new LinkedHashMap<>();
        state.put("m", toPrimitiveArray(m));
        state.put("v", toPrimitiveArray(v));
        state.put("step", new double[]{currentStep});
        return state;
    }

    @Override
    public void loadOptimizerStateDict(Map<String, double[]> stateDict) {
        double[] mData = stateDict.get("m");
        double[] vData = stateDict.get("v");
        double[] stepData = stateDict.get("step");
        if (mData != null && m == null) m = currentParams.copy().multiplyByScalar(0.0);
        if (mData != null) setVectorData(m, mData);
        if (vData != null) setVectorData(v, vData);
        if (stepData != null && stepData.length > 0) this.currentStep = (int) stepData[0];
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
