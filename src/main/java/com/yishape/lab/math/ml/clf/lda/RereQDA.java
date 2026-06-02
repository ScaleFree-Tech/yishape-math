package com.yishape.lab.math.ml.clf.lda;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.ml.clf.BatchPredResult;
import com.yishape.lab.math.ml.clf.ClfResult;
import com.yishape.lab.math.ml.clf.IClassifier;
import com.yishape.lab.math.ml.metric.ClassificationMetrics;
import com.yishape.lab.util.Tuple2;

import com.yishape.lab.util.YishapeLogger;

import java.util.*;

/**
 * 二次判别分析 (Quadratic Discriminant Analysis)
 *
 * @author lteb2
 */
public class RereQDA implements IClassifier {

    private static final YishapeLogger log = YishapeLogger.getLogger(RereQDA.class);

    private double regParam = 0.0;
    private double tol = 1e-4;
    private boolean trained = false;
    private String[] classes;
    private int nClasses;
    private int nFeatures;
    private double[] priors;
    private double[][] means;
    private double[][][] rotations;
    private double[][] scalings;
    private ClfResult result;

    public RereQDA() {
    }

    public RereQDA(double regParam) {
        this.regParam = regParam;
    }

    @Override
    public IClassifier fit(IMatrix feature, String[] labels) {
        if (feature == null || labels == null) {
            throw new IllegalArgumentException("特征矩阵和标签数组不能为空");
        }

        int nSamples = feature.getRowNum();
        this.nFeatures = feature.getColNum();

        Set<String> uniqueClasses = new LinkedHashSet<>(Arrays.asList(labels));
        this.classes = uniqueClasses.toArray(new String[0]);
        Arrays.sort(this.classes);
        this.nClasses = this.classes.length;

        if (nClasses < 2) {
            throw new IllegalArgumentException("类别数量必须至少为2");
        }

        Map<String, Integer> labelToIndex = new HashMap<>();
        for (int i = 0; i < classes.length; i++) {
            labelToIndex.put(classes[i], i);
        }
        int[] y = new int[nSamples];
        for (int i = 0; i < nSamples; i++) {
            y[i] = labelToIndex.get(labels[i]);
        }

        this.priors = new double[nClasses];
        int[] classCounts = new int[nClasses];
        for (int i = 0; i < nSamples; i++) {
            classCounts[y[i]]++;
        }
        for (int i = 0; i < nClasses; i++) {
            priors[i] = (double) classCounts[i] / nSamples;
        }

        this.means = new double[nClasses][nFeatures];
        for (int i = 0; i < nSamples; i++) {
            int clsIdx = y[i];
            for (int j = 0; j < nFeatures; j++) {
                means[clsIdx][j] += (Double) feature.get(i, j);
            }
        }
        for (int i = 0; i < nClasses; i++) {
            for (int j = 0; j < nFeatures; j++) {
                means[i][j] /= classCounts[i];
            }
        }

        this.rotations = new double[nClasses][nFeatures][nFeatures];
        this.scalings = new double[nClasses][nFeatures];

        for (int c = 0; c < nClasses; c++) {
            List<double[]> classRows = new ArrayList<>();
            for (int i = 0; i < nSamples; i++) {
                if (y[i] == c) {
                    double[] row = new double[nFeatures];
                    for (int j = 0; j < nFeatures; j++) {
                        row[j] = (Double) feature.get(i, j) - means[c][j];
                    }
                    classRows.add(row);
                }
            }

            if (classRows.size() == 1) {
                throw new IllegalArgumentException("类别 " + classes[c] + " 只有1个样本");
            }

            int nClassSamples = classRows.size();
            double[][] centered = classRows.toArray(new double[0][]);

            IMatrix<Double> centeredMatrix = IMatrix.of(centered);
            Tuple2<IVector<Double>, IMatrix<Double>> svdResult = centeredMatrix.eigen();

            double[] singularValues = new double[svdResult._1.size()];
            for (int i = 0; i < singularValues.length; i++) {
                singularValues[i] = (Double) svdResult._1.get(i);
            }
            IMatrix<Double> V = svdResult._2;

            double[] s2 = new double[nFeatures];
            for (int i = 0; i < nFeatures; i++) {
                if (i < singularValues.length) {
                    s2[i] = singularValues[i] * singularValues[i] / (nClassSamples - 1);
                }
            }

            for (int i = 0; i < s2.length; i++) {
                s2[i] = (1 - regParam) * s2[i] + regParam;
            }

            scalings[c] = s2;

            for (int i = 0; i < nFeatures; i++) {
                for (int j = 0; j < nFeatures; j++) {
                    rotations[c][i][j] = (Double) V.get(i, j);
                }
            }
        }

        this.trained = true;
        this.result = buildClassificationResult();
        return this;
    }

    @Override
    public String[] fitPredict(IMatrix feature, String[] labels) {
        fit(feature, labels);
        return predictBatch(feature);
    }

    @Override
    public ClfResult getResult() {
        return result;
    }

    @Override
    public String predict(IVector x) {
        if (!trained) {
            throw new IllegalStateException("模型尚未训练");
        }
        double[] probs = predictProbabilities(x);
        int maxIdx = 0;
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > probs[maxIdx]) maxIdx = i;
        }
        return classes[maxIdx];
    }

    @Override
    public Map<String, Double> predictProb(IVector x) {
        if (!trained) {
            throw new IllegalStateException("模型尚未训练");
        }
        double[] probs = predictProbabilities(x);
        Map<String, Double> result = new LinkedHashMap<>();
        for (int i = 0; i < nClasses; i++) {
            result.put(classes[i], probs[i]);
        }
        return result;
    }

    private double[] predictProbabilities(IVector x) {
        double[] decision = new double[nClasses];

        for (int c = 0; c < nClasses; c++) {
            double[] xCentered = new double[nFeatures];
            for (int j = 0; j < nFeatures; j++) {
                xCentered[j] = (Double) x.get(j) - means[c][j];
            }

            double[] tmp = new double[nFeatures];
            for (int i = 0; i < nFeatures; i++) {
                double sum = 0;
                for (int j = 0; j < nFeatures; j++) {
                    sum += xCentered[j] * rotations[c][j][i];
                }
                tmp[i] = sum / Math.sqrt(scalings[c][i]);
            }

            double norm2 = 0;
            for (int i = 0; i < nFeatures; i++) {
                norm2 += tmp[i] * tmp[i];
            }

            double logDet = 0;
            for (int i = 0; i < nFeatures; i++) {
                logDet += Math.log(scalings[c][i]);
            }

            decision[c] = -0.5 * (norm2 + logDet) + Math.log(priors[c]);
        }

        double maxDecision = decision[0];
        for (int i = 1; i < decision.length; i++) {
            if (decision[i] > maxDecision) maxDecision = decision[i];
        }

        double[] expDecision = new double[nClasses];
        double sumExp = 0;
        for (int i = 0; i < decision.length; i++) {
            expDecision[i] = Math.exp(decision[i] - maxDecision);
            sumExp += expDecision[i];
        }

        double[] probs = new double[nClasses];
        for (int i = 0; i < probs.length; i++) {
            probs[i] = expDecision[i] / sumExp;
        }
        return probs;
    }

    @Override
    public String[] predictBatch(IMatrix features) {
        if (!trained) throw new IllegalStateException("模型尚未训练");
        int nSamples = features.getRowNum();
        String[] predictions = new String[nSamples];
        for (int i = 0; i < nSamples; i++) {
            IVector<Double> row = features.getRow(i);
            predictions[i] = predict(row);
        }
        return predictions;
    }

    @Override
    public BatchPredResult predictBatchWithProbs(IMatrix features) {
        if (!trained) throw new IllegalStateException("模型尚未训练");
        int nSamples = features.getRowNum();
        String[] predictions = new String[nSamples];
        double[][] classProbs = new double[nSamples][nClasses];

        for (int i = 0; i < nSamples; i++) {
            IVector<Double> row = features.getRow(i);
            double[] probs = predictProbabilities(row);
            classProbs[i] = probs;
            int maxIdx = 0;
            for (int j = 1; j < probs.length; j++) {
                if (probs[j] > probs[maxIdx]) maxIdx = j;
            }
            predictions[i] = classes[maxIdx];
        }
        return new BatchPredResult(predictions, classProbs);
    }

    @Override
    public boolean isTrained() { return trained; }

    @Override
    public ClassificationMetrics getMetrics() {
        throw new UnsupportedOperationException("请使用 ClassificationMetrics.compute() 方法");
    }

    @Override
    public void setMetrics(ClassificationMetrics metrics) {}

    // ==================== JSON persistence ====================

    @Override
    public Map<String, Object> toParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("regParam", regParam);
        p.put("tol", tol);
        p.put("trained", trained);
        p.put("classes", classes.clone());
        p.put("nClasses", nClasses);
        p.put("nFeatures", nFeatures);
        p.put("priors", priors.clone());
        p.put("means", means.clone());
        p.put("rotations", rotations.clone());
        p.put("scalings", scalings.clone());
        return p;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void fromParams(Map<String, Object> p) {
        this.regParam = ((Number) p.get("regParam")).doubleValue();
        this.tol = ((Number) p.get("tol")).doubleValue();
        this.trained = (Boolean) p.get("trained");
        this.classes = ((List<String>) p.get("classes")).toArray(new String[0]);
        this.nClasses = ((Number) p.get("nClasses")).intValue();
        this.nFeatures = ((Number) p.get("nFeatures")).intValue();
        this.priors = (double[]) p.get("priors");
        this.means = (double[][]) p.get("means");
        this.rotations = (double[][][]) p.get("rotations");
        this.scalings = (double[][]) p.get("scalings");
        this.result = trained ? buildClassificationResult() : null;
    }

    private ClfResult buildClassificationResult() {
        return new ClfResult() {
            {
                setNumClasses(nClasses);
                setBinaryClassification(nClasses == 2);
                Map<String, Integer> mapping = new LinkedHashMap<>();
                for (int i = 0; i < classes.length; i++) mapping.put(classes[i], i);
                setLabelMapping(mapping);
                Map<Integer, String> reverse = new HashMap<>();
                for (Map.Entry<String, Integer> e : mapping.entrySet()) reverse.put(e.getValue(), e.getKey());
                setReverseLabelMapping(reverse);
            }
            @Override
            public String getModelTypeDescription() { return "Quadratic Discriminant Analysis (QDA)"; }
            @Override
            public String getModelSummary() { return "QDA Model: " + nClasses + " classes, " + nFeatures + " features"; }
            @Override
            public boolean isTrained() { return trained; }
        };
    }

    public double getRegParam() { return regParam; }
    public void setRegParam(double regParam) { this.regParam = regParam; }
    public String[] getClasses() { return classes; }
    public double[][] getMeans() { return means; }
}
