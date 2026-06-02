package com.yishape.lab.math.optimize.newton;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.IOnlineOptimizer;
import com.yishape.lab.util.YishapeLogger;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Online Adadelta optimizer (Adaptive Learning Rate).
 *
 * <p>Adadelta is an extension of Adagrad that reduces its aggressive monotonic
 * learning rate decay. It uses a running average of parameter updates (not just
 * gradients) to scale the learning rate, and requires no initial learning rate.</p>
 *
 * <pre>
 * E[g²]_t = ρ * E[g²]_{t-1} + (1-ρ) * g_t²        # gradient running average
 * Δθ_t    = -√(E[Δθ²]_{t-1} + ε) / √(E[g²]_t + ε) * g_t   # parameter update
 * E[Δθ²]_t = ρ * E[Δθ²]_{t-1} + (1-ρ) * Δθ_t²     # update running average
 * θ_t      = θ_{t-1} + Δθ_t                         # apply update
 * </pre>
 *
 * <h3>Features:</h3>
 * <ul>
 *   <li>No initial learning rate required (self-adaptive)</li>
 *   <li>Fixes Adagrad's aggressive learning rate decay</li>
 *   <li>Uses units of "parameter change" for update scaling</li>
 *   <li>Good for recurrent neural networks</li>
 * </ul>
 *
 * @see IOnlineOptimizer
 */
public class RereOnlineAdadelta implements IOnlineOptimizer {

    private static final YishapeLogger log = YishapeLogger.getLogger(RereOnlineAdadelta.class);

    private double rho = 0.95;
    private double epsilon = 1e-6;
    private double weightDecay = 0.0;
    private boolean verbose = false;
    private boolean skipGradientValidation = false;

    private IVector currentParams = null;
    private IVector EGradSq = null;   // E[g²] — running average of squared gradients
    private IVector EDeltaSq = null;  // E[Δθ²] — running average of squared updates
    private int currentStep = 0;
    private boolean initialized = false;

    public RereOnlineAdadelta() {}

    public RereOnlineAdadelta(double rho) {
        setRho(rho);
    }

    public RereOnlineAdadelta(double rho, double epsilon) {
        setRho(rho);
        setEpsilon(epsilon);
    }

    public RereOnlineAdadelta(double rho, double epsilon, double weightDecay) {
        setRho(rho);
        setEpsilon(epsilon);
        setWeightDecay(weightDecay);
    }

    @Override
    public void initialize(IVector initialParams) {
        if (initialParams == null) {
            throw new IllegalArgumentException("初始参数不能为null");
        }
        this.currentParams = initialParams.copy();
        this.EGradSq = initialParams.copy().multiplyByScalar(0.0);
        this.EDeltaSq = initialParams.copy().multiplyByScalar(0.0);
        this.currentStep = 0;
        this.initialized = true;
        if (verbose) {
            log.debug("Adadelta 已初始化，参数维度: {}", initialParams.size());
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
        double[] egArr = ((com.yishape.lab.math.linalg.IDoubleVector) EGradSq).getData();
        double[] edArr = ((com.yishape.lab.math.linalg.IDoubleVector) EDeltaSq).getData();
        double[] g = ((com.yishape.lab.math.linalg.IDoubleVector) gradient).getData();
        int n = p.length;

        double[] newEGradSq = new double[n];
        double[] newEDeltaSq = new double[n];
        double[] newP = new double[n];
        double r = this.rho;
        double oneMinusRho = 1.0 - r;
        double eps = epsilon;
        double wd = weightDecay;

        for (int i = 0; i < n; i++) {
            double gi = g[i] + wd * p[i];
            double eg2i = r * egArr[i] + oneMinusRho * gi * gi;
            newEGradSq[i] = eg2i;
            double deltaI = -Math.sqrt(edArr[i] + eps) / Math.sqrt(eg2i + eps) * gi;
            double ed2i = r * edArr[i] + oneMinusRho * deltaI * deltaI;
            newEDeltaSq[i] = ed2i;
            newP[i] = p[i] + deltaI;
        }

        currentParams = com.yishape.lab.math.linalg.IDoubleVector.of(newP);
        EGradSq = com.yishape.lab.math.linalg.IDoubleVector.of(newEGradSq);
        EDeltaSq = com.yishape.lab.math.linalg.IDoubleVector.of(newEDeltaSq);

        if (verbose && currentStep % 1000 == 0) {
            log.debug("Adadelta step {}", currentStep);
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
            this.EGradSq = params.copy().multiplyByScalar(0.0);
            this.EDeltaSq = params.copy().multiplyByScalar(0.0);
        }
        if (!initialized) initialize(params);
    }

    @Override
    public double getCurrentLearningRate() { return 1.0; } // Adadelta is self-adaptive, no explicit lr

    @Override
    public void setLearningRate(double learningRate) {
        // Adadelta ignores explicit learning rate; this is a no-op
    }

    @Override
    public int getCurrentStep() { return currentStep; }

    @Override
    public void reset() {
        this.currentParams = null;
        this.EGradSq = null;
        this.EDeltaSq = null;
        this.currentStep = 0;
        this.initialized = false;
    }

    @Override
    public boolean isInitialized() { return initialized; }

    @Override
    public IOnlineOptimizer clone() {
        RereOnlineAdadelta c = new RereOnlineAdadelta(rho, epsilon, weightDecay);
        c.verbose = this.verbose;
        c.skipGradientValidation = this.skipGradientValidation;
        if (initialized) {
            c.currentParams = this.currentParams.copy();
            c.EGradSq = this.EGradSq.copy();
            c.EDeltaSq = this.EDeltaSq.copy();
            c.currentStep = this.currentStep;
            c.initialized = true;
        }
        return c;
    }

    // ---- getters/setters ----

    public double getRho() { return rho; }
    public RereOnlineAdadelta setRho(double rho) {
        if (rho < 0.0 || rho >= 1.0) throw new IllegalArgumentException("rho 必须在 [0, 1)");
        this.rho = rho;
        return this;
    }

    public double getEpsilon() { return epsilon; }
    public RereOnlineAdadelta setEpsilon(double epsilon) {
        if (epsilon <= 0.0) throw new IllegalArgumentException("epsilon 必须大于0");
        this.epsilon = epsilon;
        return this;
    }

    public double getWeightDecay() { return weightDecay; }
    public RereOnlineAdadelta setWeightDecay(double weightDecay) {
        if (weightDecay < 0.0) throw new IllegalArgumentException("权重衰减必须 >= 0");
        this.weightDecay = weightDecay;
        return this;
    }

    public boolean isVerbose() { return verbose; }
    public RereOnlineAdadelta setVerbose(boolean verbose) { this.verbose = verbose; return this; }

    public boolean isSkipGradientValidation() { return skipGradientValidation; }
    public RereOnlineAdadelta setSkipGradientValidation(boolean skip) { this.skipGradientValidation = skip; return this; }

    @Override
    public Map<String, double[]> optimizerStateDict() {
        Map<String, double[]> state = new LinkedHashMap<>();
        state.put("EGradSq", toPrimitiveArray(EGradSq));
        state.put("EDeltaSq", toPrimitiveArray(EDeltaSq));
        state.put("step", new double[]{currentStep});
        return state;
    }

    @Override
    public void loadOptimizerStateDict(Map<String, double[]> stateDict) {
        double[] egData = stateDict.get("EGradSq");
        double[] edData = stateDict.get("EDeltaSq");
        double[] stepData = stateDict.get("step");
        if (egData != null && EGradSq == null) EGradSq = currentParams.copy().multiplyByScalar(0.0);
        if (egData != null) setVectorData(EGradSq, egData);
        if (edData != null && EDeltaSq == null) EDeltaSq = currentParams.copy().multiplyByScalar(0.0);
        if (edData != null) setVectorData(EDeltaSq, edData);
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
