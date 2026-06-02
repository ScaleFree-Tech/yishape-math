package com.yishape.lab.math.optimize.newton;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.IOnlineOptimizer;
import com.yishape.lab.util.YishapeLogger;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Online Adagrad optimizer (Adaptive Gradient Algorithm).
 *
 * <p>Adagrad adapts the learning rate for each parameter by dividing by the
 * square root of the sum of all historical squared gradients. Parameters that
 * receive large gradients get smaller updates, while parameters with small
 * gradients get larger updates.</p>
 *
 * <pre>
 * G_t = G_{t-1} + g_t²                          # accumulate squared gradients
 * θ_t = θ_{t-1} - lr * g_t / (√G_t + ε)         # parameter update
 * </pre>
 *
 * <h3>Features:</h3>
 * <ul>
 *   <li>Per-parameter adaptive learning rates</li>
 *   <li>Good for sparse features / infrequent parameters</li>
 *   <li>No manual learning rate tuning needed</li>
 *   <li>Learning rate monotonically decreases (can become too small)</li>
 * </ul>
 *
 * @see IOnlineOptimizer
 */
public class RereOnlineAdagrad implements IOnlineOptimizer {

    private static final YishapeLogger log = YishapeLogger.getLogger(RereOnlineAdagrad.class);

    private double learningRate = 0.01;
    private double epsilon = 1e-8;
    private double weightDecay = 0.0;
    private boolean verbose = false;
    private boolean skipGradientValidation = false;

    private IVector currentParams = null;
    private IVector G = null;  // sum of squared gradients
    private int currentStep = 0;
    private boolean initialized = false;

    public RereOnlineAdagrad() {}

    public RereOnlineAdagrad(double learningRate) {
        setLearningRate(learningRate);
    }

    public RereOnlineAdagrad(double learningRate, double epsilon) {
        setLearningRate(learningRate);
        setEpsilon(epsilon);
    }

    public RereOnlineAdagrad(double learningRate, double epsilon, double weightDecay) {
        setLearningRate(learningRate);
        setEpsilon(epsilon);
        setWeightDecay(weightDecay);
    }

    @Override
    public void initialize(IVector initialParams) {
        if (initialParams == null) {
            throw new IllegalArgumentException("初始参数不能为null");
        }
        this.currentParams = initialParams.copy();
        this.G = initialParams.copy().multiplyByScalar(0.0);
        this.currentStep = 0;
        this.initialized = true;
        if (verbose) {
            log.debug("Adagrad 已初始化，参数维度: {}", initialParams.size());
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
        double[] GArr = ((com.yishape.lab.math.linalg.IDoubleVector) G).getData();
        double[] g = ((com.yishape.lab.math.linalg.IDoubleVector) gradient).getData();
        int n = p.length;

        double[] newG = new double[n];
        double[] newP = new double[n];
        double lr = learningRate;
        double eps = epsilon;
        double wd = weightDecay;

        for (int i = 0; i < n; i++) {
            double gi = g[i] + wd * p[i];
            double newGi = GArr[i] + gi * gi;
            newG[i] = newGi;
            newP[i] = p[i] - lr * gi / (Math.sqrt(newGi) + eps);
        }

        currentParams = com.yishape.lab.math.linalg.IDoubleVector.of(newP);
        G = com.yishape.lab.math.linalg.IDoubleVector.of(newG);

        if (verbose && currentStep % 1000 == 0) {
            log.debug("Adagrad step {}: lr={}", currentStep, learningRate);
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
            this.G = params.copy().multiplyByScalar(0.0);
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
        this.G = null;
        this.currentStep = 0;
        this.initialized = false;
    }

    @Override
    public boolean isInitialized() { return initialized; }

    @Override
    public IOnlineOptimizer clone() {
        RereOnlineAdagrad c = new RereOnlineAdagrad(learningRate, epsilon, weightDecay);
        c.verbose = this.verbose;
        c.skipGradientValidation = this.skipGradientValidation;
        if (initialized) {
            c.currentParams = this.currentParams.copy();
            c.G = this.G.copy();
            c.currentStep = this.currentStep;
            c.initialized = true;
        }
        return c;
    }

    // ---- getters/setters ----

    public double getEpsilon() { return epsilon; }
    public RereOnlineAdagrad setEpsilon(double epsilon) {
        if (epsilon <= 0.0) throw new IllegalArgumentException("epsilon 必须大于0");
        this.epsilon = epsilon;
        return this;
    }

    public double getWeightDecay() { return weightDecay; }
    public RereOnlineAdagrad setWeightDecay(double weightDecay) {
        if (weightDecay < 0.0) throw new IllegalArgumentException("权重衰减必须 >= 0");
        this.weightDecay = weightDecay;
        return this;
    }

    public boolean isVerbose() { return verbose; }
    public RereOnlineAdagrad setVerbose(boolean verbose) { this.verbose = verbose; return this; }

    public boolean isSkipGradientValidation() { return skipGradientValidation; }
    public RereOnlineAdagrad setSkipGradientValidation(boolean skip) { this.skipGradientValidation = skip; return this; }

    @Override
    public Map<String, double[]> optimizerStateDict() {
        Map<String, double[]> state = new LinkedHashMap<>();
        state.put("G", toPrimitiveArray(G));
        state.put("step", new double[]{currentStep});
        return state;
    }

    @Override
    public void loadOptimizerStateDict(Map<String, double[]> stateDict) {
        double[] gData = stateDict.get("G");
        double[] stepData = stateDict.get("step");
        if (gData != null && G == null) G = currentParams.copy().multiplyByScalar(0.0);
        if (gData != null) setVectorData(G, gData);
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
