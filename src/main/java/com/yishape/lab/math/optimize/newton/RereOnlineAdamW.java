package com.yishape.lab.math.optimize.newton;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.IOnlineOptimizer;
import com.yishape.lab.util.YishapeLogger;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 在线 AdamW 优化器实现（Adam + Decoupled Weight Decay）。
 *
 * <p>与 Adam 的关键区别：权重衰减不进入梯度累积，而是在参数更新后直接作用在参数上。
 * params = params - lr * m̂/(√v̂+ε) - lr * wd * params</p>
 */
public class RereOnlineAdamW implements IOnlineOptimizer {

    private static final YishapeLogger log = YishapeLogger.getLogger(RereOnlineAdamW.class);

    private double learningRate = 0.001;
    private double beta1 = 0.9;
    private double beta2 = 0.999;
    private double epsilon = 1e-8;
    private double weightDecay = 0.01;    // AdamW 默认启用解耦权重衰减
    private boolean verbose = false;
    private boolean amsgrad = false;
    private boolean skipGradientValidation = false;

    private IVector currentParams = null;
    private IVector m = null;
    private IVector v = null;
    private IVector vMax = null;
    private int currentStep = 0;
    private boolean initialized = false;

    public RereOnlineAdamW() {}

    public RereOnlineAdamW(double learningRate) {
        setLearningRate(learningRate);
    }

    public RereOnlineAdamW(double learningRate, double beta1, double beta2) {
        setLearningRate(learningRate);
        setBeta1(beta1);
        setBeta2(beta2);
    }

    public RereOnlineAdamW(double learningRate, double beta1, double beta2, double weightDecay) {
        setLearningRate(learningRate);
        setBeta1(beta1);
        setBeta2(beta2);
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
        if (amsgrad) {
            this.vMax = initialParams.copy().multiplyByScalar(0.0);
        }
        this.currentStep = 0;
        this.initialized = true;
        if (verbose) {
            log.debug("AdamW 已初始化，参数维度: " + initialParams.size());
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
        double[] newP = new double[n];
        double[] newVMax = amsgrad ? new double[n] : null;
        double[] vMaxArr = (amsgrad && vMax != null) ? ((com.yishape.lab.math.linalg.IDoubleVector) vMax).getData() : null;

        double beta1Correction = Math.max(1.0 - Math.pow(beta1, currentStep), 1e-8);
        double beta2Correction = Math.max(1.0 - Math.pow(beta2, currentStep), 1e-8);
        double invBeta1Corr = 1.0 / beta1Correction;
        double invBeta2Corr = 1.0 / beta2Correction;
        double oneMinusBeta1 = 1.0 - beta1;
        double oneMinusBeta2 = 1.0 - beta2;
        double lr = learningRate;
        double lrWd = learningRate * weightDecay;

        for (int i = 0; i < n; i++) {
            double gi = g[i];
            double mi = beta1 * mArr[i] + oneMinusBeta1 * gi;
            newM[i] = mi;
            double vi = beta2 * vArr[i] + oneMinusBeta2 * gi * gi;
            newV[i] = vi;
            double mHatI = mi * invBeta1Corr;
            double vHatI = vi * invBeta2Corr;
            if (amsgrad) {
                double vMaxI = (vMaxArr != null) ? Math.max(vMaxArr[i], vHatI) : vHatI;
                newVMax[i] = vMaxI;
                vHatI = vMaxI;
            }
            double adamI = mHatI / (Math.sqrt(vHatI) + epsilon);
            newP[i] = p[i] - lr * adamI - lrWd * p[i];
        }

        currentParams = com.yishape.lab.math.linalg.IDoubleVector.of(newP);
        m = com.yishape.lab.math.linalg.IDoubleVector.of(newM);
        v = com.yishape.lab.math.linalg.IDoubleVector.of(newV);
        if (amsgrad) {
            vMax = com.yishape.lab.math.linalg.IDoubleVector.of(newVMax);
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
            if (amsgrad) this.vMax = params.copy().multiplyByScalar(0.0);
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
        this.vMax = null;
        this.currentStep = 0;
        this.initialized = false;
    }

    @Override
    public boolean isInitialized() { return initialized; }

    @Override
    public IOnlineOptimizer clone() {
        RereOnlineAdamW clone = new RereOnlineAdamW(learningRate, beta1, beta2, weightDecay);
        clone.epsilon = this.epsilon;
        clone.amsgrad = this.amsgrad;
        clone.verbose = this.verbose;
        if (initialized) {
            clone.currentParams = this.currentParams.copy();
            clone.m = this.m.copy();
            clone.v = this.v.copy();
            if (this.vMax != null) clone.vMax = this.vMax.copy();
            clone.currentStep = this.currentStep;
            clone.initialized = true;
        }
        return clone;
    }

    // ---- getters/setters ----

    public double getBeta1() { return beta1; }
    public RereOnlineAdamW setBeta1(double beta1) {
        if (beta1 < 0.0 || beta1 >= 1.0) throw new IllegalArgumentException("beta1 必须在 [0, 1)");
        this.beta1 = beta1;
        return this;
    }

    public double getBeta2() { return beta2; }
    public RereOnlineAdamW setBeta2(double beta2) {
        if (beta2 < 0.0 || beta2 >= 1.0) throw new IllegalArgumentException("beta2 必须在 [0, 1)");
        this.beta2 = beta2;
        return this;
    }

    public double getEpsilon() { return epsilon; }
    public RereOnlineAdamW setEpsilon(double epsilon) {
        if (epsilon <= 0.0) throw new IllegalArgumentException("epsilon 必须大于0");
        this.epsilon = epsilon;
        return this;
    }

    public double getWeightDecay() { return weightDecay; }
    public RereOnlineAdamW setWeightDecay(double weightDecay) {
        if (weightDecay < 0.0) throw new IllegalArgumentException("权重衰减必须 >= 0");
        this.weightDecay = weightDecay;
        return this;
    }

    public boolean isAmsgrad() { return amsgrad; }
    public RereOnlineAdamW setAmsgrad(boolean amsgrad) {
        this.amsgrad = amsgrad;
        if (initialized) {
            if (amsgrad && vMax == null) vMax = currentParams.copy().multiplyByScalar(0.0);
            else if (!amsgrad) vMax = null;
        }
        return this;
    }

    public boolean isVerbose() { return verbose; }
    public RereOnlineAdamW setVerbose(boolean verbose) {
        this.verbose = verbose;
        return this;
    }

    public boolean isSkipGradientValidation() { return skipGradientValidation; }
    public RereOnlineAdamW setSkipGradientValidation(boolean skip) {
        this.skipGradientValidation = skip;
        return this;
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
