package com.yishape.lab.math.optimize.newton;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.IOnlineOptimizer;
import com.yishape.lab.util.YishapeLogger;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 在线 RMSprop 优化器实现。
 * RMSprop 通过梯度平方的指数移动平均来调整学习率，适合非平稳目标和非凸优化。
 */
public class RereOnlineRMSprop implements IOnlineOptimizer {

    private static final YishapeLogger log = YishapeLogger.getLogger(RereOnlineRMSprop.class);

    private double learningRate = 0.001;
    private double alpha = 0.99;          // 梯度平方的衰减率
    private double epsilon = 1e-8;
    private double weightDecay = 0.0;
    private double momentum = 0.0;         // 可选的动量因子
    private boolean centered = false;      // 是否使用中心化版本（减去均值）
    private boolean verbose = false;
    private boolean skipGradientValidation = false;

    private IVector currentParams = null;
    private IVector v = null;              // 梯度平方的指数移动平均
    private IVector mg = null;             // 中心化版本的梯度均值
    private IVector momentumBuf = null;    // 动量缓冲区
    private int currentStep = 0;
    private boolean initialized = false;

    public RereOnlineRMSprop() {}

    public RereOnlineRMSprop(double learningRate) {
        setLearningRate(learningRate);
    }

    public RereOnlineRMSprop(double learningRate, double alpha) {
        setLearningRate(learningRate);
        setAlpha(alpha);
    }

    @Override
    public void initialize(IVector initialParams) {
        if (initialParams == null) {
            throw new IllegalArgumentException("初始参数不能为null");
        }
        this.currentParams = initialParams.copy();
        this.v = initialParams.copy().multiplyByScalar(0.0);
        if (centered) {
            this.mg = initialParams.copy().multiplyByScalar(0.0);
        }
        if (momentum > 0.0) {
            this.momentumBuf = initialParams.copy().multiplyByScalar(0.0);
        }
        this.currentStep = 0;
        this.initialized = true;
        if (verbose) {
            log.debug("RMSprop 已初始化，参数维度: " + initialParams.size());
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
        double[] vArr = ((com.yishape.lab.math.linalg.IDoubleVector) v).getData();
        double[] g = ((com.yishape.lab.math.linalg.IDoubleVector) gradient).getData();
        int n = p.length;

        double[] newV = new double[n];
        double[] newMg = centered ? new double[n] : null;
        double[] mgArr = (centered && mg != null) ? ((com.yishape.lab.math.linalg.IDoubleVector) mg).getData() : null;
        double[] newMomBuf = (momentum > 0.0) ? new double[n] : null;
        double[] momArr = (momentum > 0.0 && momentumBuf != null) ? ((com.yishape.lab.math.linalg.IDoubleVector) momentumBuf).getData() : null;
        double[] newP = new double[n];

        double alp = this.alpha;
        double oneMinusAlpha = 1.0 - alp;
        double eps = epsilon;
        double wd = weightDecay;
        double lr = learningRate;
        double mom = momentum;

        for (int i = 0; i < n; i++) {
            double gi = g[i] + wd * p[i];
            double vi = alp * vArr[i] + oneMinusAlpha * gi * gi;
            newV[i] = vi;
            double denomI;
            if (centered) {
                double mgi = alp * mgArr[i] + oneMinusAlpha * gi;
                newMg[i] = mgi;
                denomI = Math.sqrt(Math.max(0, vi - mgi * mgi)) + eps;
            } else {
                denomI = Math.sqrt(vi) + eps;
            }
            double updateI = lr * gi / denomI;
            if (mom > 0.0) {
                double newMomI = mom * momArr[i] + updateI;
                newMomBuf[i] = newMomI;
                newP[i] = p[i] - newMomI;
            } else {
                newP[i] = p[i] - updateI;
            }
        }

        currentParams = com.yishape.lab.math.linalg.IDoubleVector.of(newP);
        v = com.yishape.lab.math.linalg.IDoubleVector.of(newV);
        if (centered) {
            mg = com.yishape.lab.math.linalg.IDoubleVector.of(newMg);
        }
        if (momentum > 0.0) {
            momentumBuf = com.yishape.lab.math.linalg.IDoubleVector.of(newMomBuf);
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
            this.v = params.copy().multiplyByScalar(0.0);
            if (centered) this.mg = params.copy().multiplyByScalar(0.0);
            if (momentum > 0.0) this.momentumBuf = params.copy().multiplyByScalar(0.0);
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
        this.v = null;
        this.mg = null;
        this.momentumBuf = null;
        this.currentStep = 0;
        this.initialized = false;
    }

    @Override
    public boolean isInitialized() { return initialized; }

    @Override
    public IOnlineOptimizer clone() {
        RereOnlineRMSprop clone = new RereOnlineRMSprop(learningRate, alpha);
        clone.epsilon = this.epsilon;
        clone.weightDecay = this.weightDecay;
        clone.momentum = this.momentum;
        clone.centered = this.centered;
        clone.verbose = this.verbose;
        if (initialized) {
            clone.currentParams = this.currentParams.copy();
            clone.v = this.v.copy();
            if (this.mg != null) clone.mg = this.mg.copy();
            if (this.momentumBuf != null) clone.momentumBuf = this.momentumBuf.copy();
            clone.currentStep = this.currentStep;
            clone.initialized = true;
        }
        return clone;
    }

    // ---- getters/setters ----

    public double getAlpha() { return alpha; }
    public RereOnlineRMSprop setAlpha(double alpha) {
        if (alpha < 0.0 || alpha >= 1.0) throw new IllegalArgumentException("alpha 必须在 [0, 1) 范围内");
        this.alpha = alpha;
        return this;
    }

    public double getEpsilon() { return epsilon; }
    public RereOnlineRMSprop setEpsilon(double epsilon) {
        if (epsilon <= 0.0) throw new IllegalArgumentException("epsilon 必须大于0");
        this.epsilon = epsilon;
        return this;
    }

    public double getWeightDecay() { return weightDecay; }
    public RereOnlineRMSprop setWeightDecay(double weightDecay) {
        if (weightDecay < 0.0) throw new IllegalArgumentException("权重衰减必须 >= 0");
        this.weightDecay = weightDecay;
        return this;
    }

    public double getMomentum() { return momentum; }
    public RereOnlineRMSprop setMomentum(double momentum) {
        if (momentum < 0.0) throw new IllegalArgumentException("动量必须 >= 0");
        this.momentum = momentum;
        return this;
    }

    public boolean isCentered() { return centered; }
    public RereOnlineRMSprop setCentered(boolean centered) {
        this.centered = centered;
        return this;
    }

    public boolean isVerbose() { return verbose; }
    public RereOnlineRMSprop setVerbose(boolean verbose) {
        this.verbose = verbose;
        return this;
    }

    public boolean isSkipGradientValidation() { return skipGradientValidation; }
    public RereOnlineRMSprop setSkipGradientValidation(boolean skip) {
        this.skipGradientValidation = skip;
        return this;
    }

    @Override
    public Map<String, double[]> optimizerStateDict() {
        Map<String, double[]> state = new LinkedHashMap<>();
        state.put("v", toPrimitiveArray(v));
        state.put("step", new double[]{currentStep});
        if (centered && mg != null) {
            state.put("mg", toPrimitiveArray(mg));
        }
        if (momentum > 0.0 && momentumBuf != null) {
            state.put("momentum_buffer", toPrimitiveArray(momentumBuf));
        }
        return state;
    }

    @Override
    public void loadOptimizerStateDict(Map<String, double[]> stateDict) {
        double[] vData = stateDict.get("v");
        double[] stepData = stateDict.get("step");
        double[] mgData = stateDict.get("mg");
        double[] momData = stateDict.get("momentum_buffer");
        if (vData != null) setVectorData(v, vData);
        if (stepData != null && stepData.length > 0) this.currentStep = (int) stepData[0];
        if (mgData != null && mg != null) setVectorData(mg, mgData);
        if (momData != null && momentumBuf != null) setVectorData(momentumBuf, momData);
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
