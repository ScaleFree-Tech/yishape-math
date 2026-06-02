package com.yishape.lab.math.ml.clf.svm;

import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.ml.ISerializableModel;
import com.yishape.lab.math.ml.clf.BatchPredResult;
import com.yishape.lab.math.ml.clf.ClfResult;
import com.yishape.lab.math.ml.clf.IClassifier;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.newton.RereLBFGS;
import com.yishape.lab.math.ml.metric.ClassificationMetrics;
import com.yishape.lab.math.optimize.Opts;
import com.yishape.lab.util.YishapeLogger;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.TreeSet;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.autodiff.IDiffMatrix;

/**
 * 线性支持向量机（原始问题 + 平方铰链损失 + L2），多分类采用 One-vs-Rest。
 * <p>
 * 目标：{@code 0.5 * ||w||^2 + C * mean_i( max(0, 1 - y_i (w^T x_i + b))^2 )}，{@code y_i ∈ {-1,+1}}。<br>
 * 适合连续特征；可选训练集标准化（每列减均值除标准差）。
 * </p>
 */
public class RereLinearSVM implements IClassifier {

    private static final YishapeLogger log = YishapeLogger.getLogger(RereLinearSVM.class);

    private IOptimizer optimizer = Opts.lbfgs();

    /** 铰链项权重（越大越强调分类间隔） */
    private double C = 1.0;

    private boolean standardizeFeatures = true;

    private Map<String, Integer> labelToIndex = new HashMap<>();
    private Map<Integer, String> indexToLabel = new HashMap<>();
    private int numClasses;
    private int numFeatures;

    /** 二分类：单一超平面 */
    private boolean binaryTask;
    private double[] w;
    private double b;

    /** 多分类 OvR：每类一行权、一偏置 */
    private double[][] wOvR;
    private double[] bOvR;

    private double[] mean;
    private double[] std;

    private boolean trained;
    private ClassificationMetrics metrics;
    private LinearSvmResult result;

    public RereLinearSVM() {
    }

    public RereLinearSVM(double C, boolean standardizeFeatures) {
        this.C = C > 0 ? C : 1.0;
        this.standardizeFeatures = standardizeFeatures;
    }

    public double getC() {
        return C;
    }

    public void setC(double c) {
        this.C = c > 0 ? c : 1.0;
    }

    public boolean isStandardizeFeatures() {
        return standardizeFeatures;
    }

    public void setStandardizeFeatures(boolean standardizeFeatures) {
        this.standardizeFeatures = standardizeFeatures;
    }

    /**
     * 二分类：超平面系数 {@code w}（不含偏置），定义在训练时的<strong>标准化后</strong>特征空间上；
     * 命名与 {@link com.yishape.lab.math.ml.reg.RereLinearRegression#getFeatureWeights()} 对齐。
     * <p>多分类（OvR）请使用 {@link #getFeatureWeightsOvR()}。</p>
     *
     * @return 未训练时为 null
     */
    public IVector<Double> getFeatureWeights() {
        if (!trained || !binaryTask || w == null) {
            return null;
        }
        return Linalg.vector(Arrays.copyOf(w, numFeatures));
    }

    /**
     * 二分类偏置 {@code b}（标准化后空间）。多分类请用 {@link #getBiasOvR()}。
     *
     * @throws IllegalStateException 未训练或非二分类
     */
    public double getBias() {
        if (!trained) {
            throw new IllegalStateException("模型尚未训练");
        }
        if (!binaryTask) {
            throw new IllegalStateException("多分类请使用 getBiasOvR()");
        }
        return b;
    }

    /**
     * OvR 各二分类子问题的 {@code w_k}；二分类时为长度 1 的外层数组。
     *
     * @return 未训练时为 null
     */
    public double[][] getFeatureWeightsOvR() {
        if (!trained || numFeatures <= 0) {
            return null;
        }
        if (binaryTask) {
            return new double[][]{Arrays.copyOf(w, numFeatures)};
        }
        double[][] out = new double[numClasses][numFeatures];
        for (int k = 0; k < numClasses; k++) {
            System.arraycopy(wOvR[k], 0, out[k], 0, numFeatures);
        }
        return out;
    }

    /**
     * OvR 各子问题的偏置 {@code b_k}；二分类时为单元素数组。
     *
     * @return 未训练时为 null
     */
    public double[] getBiasOvR() {
        if (!trained) {
            return null;
        }
        if (binaryTask) {
            return new double[]{b};
        }
        return Arrays.copyOf(bOvR, numClasses);
    }

    /** 训练完成后是否为二分类任务。 */
    public boolean isBinaryTask() {
        return trained && binaryTask;
    }

    @Deprecated
    public IClassifier fitWithManualGradient(IMatrix feature, String[] labels) {
        validate(feature, labels);
        buildLabelMapping(labels);
        numFeatures = feature.getColNum();
        int n = feature.getRowNum();

        IMatrix Xwork = feature;
        mean = new double[numFeatures];
        std = new double[numFeatures];
        if (standardizeFeatures) {
            computeMeanStd(feature, mean, std);
            Xwork = applyStandardize(feature, mean, std);
        } else {
            Arrays.fill(std, 1.0);
        }

        binaryTask = numClasses == 2;
        if (binaryTask) {
            double[] ypm = new double[n];
            for (int i = 0; i < n; i++) {
                int idx = labelToIndex.get(labels[i]);
                ypm[i] = idx == 1 ? 1.0 : -1.0;
            }
            double[] theta = trainBinaryPlane(Xwork, ypm);
            w = Arrays.copyOfRange(theta, 0, numFeatures);
            b = theta[numFeatures];
        } else {
            wOvR = new double[numClasses][numFeatures];
            bOvR = new double[numClasses];
            for (int k = 0; k < numClasses; k++) {
                double[] ypm = new double[n];
                for (int i = 0; i < n; i++) {
                    int idx = labelToIndex.get(labels[i]);
                    ypm[i] = idx == k ? 1.0 : -1.0;
                }
                double[] theta = trainBinaryPlane(Xwork, ypm);
                System.arraycopy(theta, 0, wOvR[k], 0, numFeatures);
                bOvR[k] = theta[numFeatures];
            }
        }

        trained = true;
        metrics = ClassificationMetrics.compute(this, feature, labels);

        this.result = new LinearSvmResult();
        result.setCParam(C);
        result.setStandardized(standardizeFeatures);
        result.setTrained(true);
        result.setNumClasses(numClasses);
        result.setNumFeatures(numFeatures);
        result.setLabelMapping(new HashMap<>(labelToIndex));
        result.setReverseLabelMapping(new HashMap<>(indexToLabel));
        result.setTrainAccuracy(metrics.getAccuracy());

        log.debug("LinearSVM fit: binary={}, acc={}", binaryTask, metrics.getAccuracy());
        return this;
    }

    @Override
    public String[] fitPredict(IMatrix feature, String[] labels) {
        fit(feature, labels);
        return predictBatch(feature);
    }

    @Override
    public IClassifier fit(IMatrix feature, String[] labels) {
        validate(feature, labels);
        buildLabelMapping(labels);
        numFeatures = feature.getColNum();
        int n = feature.getRowNum();
        binaryTask = numClasses == 2;

        IMatrix Xwork = feature;
        mean = new double[numFeatures];
        std = new double[numFeatures];
        if (standardizeFeatures) {
            computeMeanStd(feature, mean, std);
            Xwork = applyStandardize(feature, mean, std);
        } else {
            Arrays.fill(std, 1.0);
        }

        if (binaryTask) {
            // ---- 二分类 autodiff 路径 ----
            double[] ypm = new double[n];
            for (int i = 0; i < n; i++) {
                ypm[i] = labelToIndex.get(labels[i]) == 1 ? 1.0 : -1.0;
            }
            double[] theta = trainBinaryAutodiff(Xwork, ypm);
            w = Arrays.copyOfRange(theta, 0, numFeatures);
            b = theta[numFeatures];
        } else {
            // ---- 多分类 OvR autodiff 路径 ----
            wOvR = new double[numClasses][numFeatures];
            bOvR = new double[numClasses];
            for (int k = 0; k < numClasses; k++) {
                double[] ypm = new double[n];
                for (int i = 0; i < n; i++) {
                    int idx = labelToIndex.get(labels[i]);
                    ypm[i] = idx == k ? 1.0 : -1.0;
                }
                double[] theta = trainBinaryAutodiff(Xwork, ypm);
                System.arraycopy(theta, 0, wOvR[k], 0, numFeatures);
                bOvR[k] = theta[numFeatures];
            }
        }

        trained = true;
        metrics = ClassificationMetrics.compute(this, feature, labels);

        this.result = new LinearSvmResult();
        result.setCParam(C);
        result.setStandardized(standardizeFeatures);
        result.setTrained(true);
        result.setNumClasses(numClasses);
        result.setNumFeatures(numFeatures);
        result.setLabelMapping(new HashMap<>(labelToIndex));
        result.setReverseLabelMapping(new HashMap<>(indexToLabel));
        result.setTrainAccuracy(metrics.getAccuracy());

        log.debug("LinearSVM fit: binary={}, acc={}", binaryTask, metrics.getAccuracy());
        return this;
    }

    private double[] trainBinaryAutodiff(IMatrix Xwork, double[] ypm) {
        int n = Xwork.getRowNum();
        int d = Xwork.getColNum();
        double[][] augData = new double[n][d + 1];
        double[][] xData = ((IDoubleMatrix) Xwork).getData();
        for (int i = 0; i < n; i++) {
            System.arraycopy(xData[i], 0, augData[i], 0, d);
            augData[i][d] = 1.0;
        }
        IDiffMatrix Xc = AD.matrix(augData);
        IDiffVector yc = AD.vector(ypm);
        final double Cparam = C;

        IVector w0 = IVector.zeros(d + 1);
        var res = AD.optimize(w0, w -> {
            IDiffVector margin = yc.mul(Xc.matmul(w));
            IDiffVector hinge = margin.rsub(1.0).relu();
            return hinge.square().mean().mul(Cparam).add(w.square().sum().mul(0.5));
        }, this.optimizer);

        return res.getOptimalPoint().toDoubleArray().clone();
    }

    @Override
    public LinearSvmResult getResult() {
        return result;
    }

    private double[] trainBinaryPlane(IMatrix X, double[] ypm1) {
        int d = X.getColNum();
        int m = X.getRowNum();
        SquaredHingeSvmProblem prob = new SquaredHingeSvmProblem(X, ypm1, d, m, C);
        IVector init = Linalg.zeros(d + 1);
        var res = optimizer.optimize(init, prob, prob);
        double[] theta = res.getOptimalPoint().toDoubleArray().clone();
        return theta;
    }

    private void validate(IMatrix feature, String[] labels) {
        if (feature == null || labels == null) {
            throw new IllegalArgumentException("特征与标签不能为 null");
        }
        if (feature.getRowNum() != labels.length || feature.getRowNum() == 0) {
            throw new IllegalArgumentException("无效训练样本数");
        }
        if (feature.getColNum() == 0) {
            throw new IllegalArgumentException("特征维度不能为 0");
        }
        for (int i = 0; i < feature.getRowNum(); i++) {
            for (int j = 0; j < feature.getColNum(); j++) {
                double v = feature.get(i, j);
                if (Double.isNaN(v) || Double.isInfinite(v)) {
                    throw new IllegalArgumentException("无效特征 (" + i + "," + j + ")");
                }
            }
        }
    }

    private void buildLabelMapping(String[] labels) {
        TreeSet<String> sorted = new TreeSet<>(Arrays.asList(labels));
        labelToIndex.clear();
        indexToLabel.clear();
        int id = 0;
        for (String s : sorted) {
            labelToIndex.put(s, id);
            indexToLabel.put(id, s);
            id++;
        }
        numClasses = sorted.size();
        if (numClasses < 2) {
            throw new IllegalArgumentException("至少需要 2 个类别");
        }
    }

    private static void computeMeanStd(IMatrix X, double[] mean, double[] std) {
        int n = X.getRowNum();
        int d = X.getColNum();
        Arrays.fill(mean, 0);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                mean[j] += X.get(i, j);
            }
        }
        for (int j = 0; j < d; j++) {
            mean[j] /= n;
        }
        Arrays.fill(std, 0);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                double t = X.get(i, j) - mean[j];
                std[j] += t * t;
            }
        }
        for (int j = 0; j < d; j++) {
            std[j] = Math.sqrt(std[j] / Math.max(1, n));
            if (std[j] < 1e-12) {
                std[j] = 1.0;
            }
        }
    }

    private static IMatrix applyStandardize(IMatrix X, double[] mean, double[] std) {
        int n = X.getRowNum();
        int d = X.getColNum();
        double[][] buf = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                buf[i][j] = (X.get(i, j) - mean[j]) / std[j];
            }
        }
        return Linalg.matrix(buf);
    }

    private IVector transform(IVector x) {
        if (!standardizeFeatures) {
            return x;
        }
        double[] z = new double[numFeatures];
        for (int j = 0; j < numFeatures; j++) {
            z[j] = (x.get(j) - mean[j]) / std[j];
        }
        return Linalg.vector(z);
    }

    private double decisionBinary(IVector xz) {
        double s = b;
        for (int j = 0; j < numFeatures; j++) {
            s += w[j] * xz.get(j);
        }
        return s;
    }

    private double decisionOvR(IVector xz, int k) {
        double s = bOvR[k];
        for (int j = 0; j < numFeatures; j++) {
            s += wOvR[k][j] * xz.get(j);
        }
        return s;
    }

    @Override
    public String predict(IVector x) {
        if (!trained) {
            throw new IllegalStateException("模型尚未训练");
        }
        if (x.length() != numFeatures) {
            throw new IllegalArgumentException("特征维度不匹配");
        }
        IVector xz = transform(x);
        if (binaryTask) {
            double s = decisionBinary(xz);
            return indexToLabel.get(s >= 0 ? 1 : 0);
        }
        int best = 0;
        double bestS = decisionOvR(xz, 0);
        for (int k = 1; k < numClasses; k++) {
            double sk = decisionOvR(xz, k);
            if (sk > bestS) {
                bestS = sk;
                best = k;
            }
        }
        return indexToLabel.get(best);
    }

    @Override
    public Map<String, Double> predictProb(IVector x) {
        if (!trained) {
            throw new IllegalStateException("模型尚未训练");
        }
        IVector xz = transform(x);
        Map<String, Double> out = new LinkedHashMap<>();
        if (binaryTask) {
            double s = decisionBinary(xz);
            double p1 = 1.0 / (1.0 + Math.exp(-s));
            out.put(indexToLabel.get(0), 1.0 - p1);
            out.put(indexToLabel.get(1), p1);
            return out;
        }
        double[] scores = new double[numClasses];
        for (int k = 0; k < numClasses; k++) {
            scores[k] = decisionOvR(xz, k);
        }
        double mx = Arrays.stream(scores).max().orElse(0);
        double sum = 0;
        double[] ex = new double[numClasses];
        for (int k = 0; k < numClasses; k++) {
            ex[k] = Math.exp(scores[k] - mx);
            sum += ex[k];
        }
        for (int k = 0; k < numClasses; k++) {
            out.put(indexToLabel.get(k), ex[k] / sum);
        }
        return out;
    }

    @Override
    public String[] predictBatch(IMatrix features) {
        int m = features.getRowNum();
        String[] y = new String[m];
        for (int i = 0; i < m; i++) {
            y[i] = predict(features.getRow(i));
        }
        return y;
    }

    @Override
    public BatchPredResult predictBatchWithProbs(IMatrix features) {
        int m = features.getRowNum();
        String[] labels = new String[m];
        double[][] probs = new double[m][numClasses];
        for (int i = 0; i < m; i++) {
            IVector row = features.getRow(i);
            labels[i] = predict(row);
            Map<String, Double> pm = predictProb(row);
            for (int c = 0; c < numClasses; c++) {
                probs[i][c] = pm.getOrDefault(indexToLabel.get(c), 0.0);
            }
        }
        return new BatchPredResult(labels, probs);
    }

    @Override
    public boolean isTrained() {
        return trained;
    }

    @Override
    public ClassificationMetrics getMetrics() {
        return metrics;
    }

    @Override
    public void setMetrics(ClassificationMetrics metrics) {
        this.metrics = metrics;
    }

    // ==================== JSON persistence ====================

    @Override
    public Map<String, Object> toParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("C", C);
        p.put("standardizeFeatures", standardizeFeatures);
        p.put("numClasses", numClasses);
        p.put("numFeatures", numFeatures);
        p.put("binaryTask", binaryTask);
        if (binaryTask) {
            p.put("w", w.clone());
            p.put("b", b);
        } else {
            p.put("wOvR", wOvR.clone());
            p.put("bOvR", bOvR.clone());
        }
        if (mean != null) p.put("mean", mean.clone());
        if (std != null) p.put("std", std.clone());
        p.put("trained", trained);
        p.put("labelToIndex", new HashMap<>(labelToIndex));
        p.put("indexToLabel", indexToLabelToString());
        return p;
    }

    private Map<String, String> indexToLabelToString() {
        Map<String, String> m = new LinkedHashMap<>();
        for (Map.Entry<Integer, String> e : indexToLabel.entrySet()) {
            m.put(String.valueOf(e.getKey()), e.getValue());
        }
        return m;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void fromParams(Map<String, Object> p) {
        this.C = ((Number) p.get("C")).doubleValue();
        this.standardizeFeatures = (Boolean) p.get("standardizeFeatures");
        this.numClasses = ((Number) p.get("numClasses")).intValue();
        this.numFeatures = ((Number) p.get("numFeatures")).intValue();
        this.binaryTask = (Boolean) p.get("binaryTask");
        if (binaryTask) {
            this.w = (double[]) p.get("w");
            this.b = ((Number) p.get("b")).doubleValue();
        } else {
            this.wOvR = (double[][]) p.get("wOvR");
            this.bOvR = (double[]) p.get("bOvR");
        }
        this.mean = (double[]) p.get("mean");
        this.std = (double[]) p.get("std");
        this.trained = (Boolean) p.get("trained");
        this.labelToIndex = new HashMap<>((Map<String, Integer>) p.get("labelToIndex"));
        this.indexToLabel = indexToLabelFromString((Map<String, String>) p.get("indexToLabel"));
        this.optimizer = Opts.lbfgs();
    }

    private Map<Integer, String> indexToLabelFromString(Map<String, String> m) {
        Map<Integer, String> result = new LinkedHashMap<>();
        for (Map.Entry<String, String> e : m.entrySet()) {
            result.put(Integer.parseInt(e.getKey()), e.getValue());
        }
        return result;
    }

    /** 单二分类子问题的目标与梯度（向量末尾为偏置）。 */
    private static final class SquaredHingeSvmProblem implements IObjectiveFunction, IGradientFunction {
        private final IMatrix X;
        private final double[] y;
        private final int d;
        private final int m;
        private final double C;

        SquaredHingeSvmProblem(IMatrix X, double[] y, int d, int m, double C) {
            this.X = X;
            this.y = y;
            this.d = d;
            this.m = m;
            this.C = C;
        }

        @Override
        public double computeObjective(IVector thetaVec) {
            double[] t = thetaVec.toDoubleArray();
            double reg = 0;
            for (int j = 0; j < d; j++) {
                reg += 0.5 * t[j] * t[j];
            }
            double hinge = 0;
            for (int i = 0; i < m; i++) {
                double f = t[d];
                for (int j = 0; j < d; j++) {
                    f += t[j] * X.get(i, j);
                }
                double z = y[i] * f;
                double h = Math.max(0.0, 1.0 - z);
                hinge += h * h;
            }
            return reg + C * hinge / m;
        }

        @Override
        public IVector computeGradient(IVector thetaVec) {
            double[] t = thetaVec.toDoubleArray();
            double[] g = new double[d + 1];
            for (int j = 0; j < d; j++) {
                g[j] = t[j];
            }
            g[d] = 0;
            for (int i = 0; i < m; i++) {
                double f = t[d];
                for (int j = 0; j < d; j++) {
                    f += t[j] * X.get(i, j);
                }
                double z = y[i] * f;
                if (z >= 1.0) {
                    continue;
                }
                double coeff = C * 2.0 / m * (1.0 - z) * (-y[i]);
                for (int j = 0; j < d; j++) {
                    g[j] += coeff * X.get(i, j);
                }
                g[d] += coeff;
            }
            return Linalg.vector(g);
        }
    }
}
