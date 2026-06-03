package com.yishape.lab.math.ml.clf.lda;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.RereDoubleVector;
import com.yishape.lab.math.ml.clf.BatchPredResult;
import com.yishape.lab.math.ml.clf.ClfResult;
import com.yishape.lab.math.ml.clf.IClassifier;
import com.yishape.lab.math.ml.metric.ClassificationMetrics;
import com.yishape.lab.util.Tuple2;

import com.yishape.lab.util.YishapeLogger;

import java.util.*;

/**
 * 线性判别分析 (Linear Discriminant Analysis)
 * <p>
 * LDA是一种经典的线性分类方法，通过找到能够最大化类间差异与类内差异比率的方向来进行分类。
 * 本实现参照sklearn的LinearDiscriminantAnalysis，支持SVD求解器。
 * </p>
 *
 * @author lteb2
 */
public class RereLDA implements IClassifier {

    private static final YishapeLogger log = YishapeLogger.getLogger(RereLDA.class);

    public enum Solver {
        SVD
    }

    private Solver solver = Solver.SVD;
    private int nComponents = -1;
    private double[] priors;
    private String[] classes;
    private int nClasses;
    private double[][] means;
    private double[] xbar;
    private double[][] scalings;
    private double[][] eigenvectors;
    private double[] eigenvalues;
    private double[][] coef;
    private double[] intercept;
    private boolean trained = false;
    private int nFeatures;
    private IMatrix<Double> trainingData;
    private ClfResult result;

    public RereLDA() {
    }

    public RereLDA(Solver solver) {
        this.solver = solver;
    }

    public RereLDA(Solver solver, int nComponents) {
        this.solver = solver;
        this.nComponents = nComponents;
    }

    @Override
    public IClassifier fit(IMatrix feature, String[] labels) {
        if (feature == null || labels == null) {
            throw new IllegalArgumentException("特征矩阵和标签数组不能为空");
        }

        int nSamples = feature.getRowNum();
        this.nFeatures = feature.getColNum();

        if (nSamples != labels.length) {
            throw new IllegalArgumentException("样本数量与标签数量不匹配");
        }

        Set<String> uniqueClasses = new LinkedHashSet<>(Arrays.asList(labels));
        this.classes = uniqueClasses.toArray(new String[0]);
        Arrays.sort(this.classes);
        this.nClasses = this.classes.length;

        if (nSamples <= nClasses) {
            throw new IllegalArgumentException("样本数量必须大于类别数量");
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

        this.trainingData = feature;

        int maxComponents = Math.min(nClasses - 1, nFeatures);
        if (nComponents < 0) {
            nComponents = maxComponents;
        } else {
            nComponents = Math.min(nComponents, maxComponents);
        }

        fitSVD(feature, y, classCounts, nSamples);

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

    private void fitSVD(IMatrix<Double> X, int[] y, int[] classCounts, int nSamples) {
        this.xbar = new double[nFeatures];
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                xbar[j] += (Double) X.get(i, j);
            }
        }
        for (int j = 0; j < nFeatures; j++) {
            xbar[j] /= nSamples;
        }

        List<double[]> centeredList = new ArrayList<>();
        for (int c = 0; c < nClasses; c++) {
            for (int i = 0; i < nSamples; i++) {
                if (y[i] == c) {
                    double[] row = new double[nFeatures];
                    for (int j = 0; j < nFeatures; j++) {
                        row[j] = (Double) X.get(i, j) - means[c][j];
                    }
                    centeredList.add(row);
                }
            }
        }
        double[][] allCentered = centeredList.toArray(new double[0][]);

        double[] std = new double[nFeatures];
        for (int j = 0; j < nFeatures; j++) {
            double sum = 0;
            for (double[] row : allCentered) {
                sum += row[j] * row[j];
            }
            std[j] = Math.sqrt(sum / (nSamples - nClasses));
            if (std[j] == 0) std[j] = 1.0;
        }

        double fac = 1.0 / (nSamples - nClasses);
        double[][] scaledX = new double[nSamples][nFeatures];
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                scaledX[i][j] = Math.sqrt(fac) * allCentered[i][j] / std[j];
            }
        }

        IMatrix<Double> scaledXMatrix = IMatrix.of(scaledX);
        Tuple2<IVector<Double>, IMatrix<Double>> svdResult = scaledXMatrix.eigen();
        double[] singularValues = new double[svdResult._1.size()];
        for (int i = 0; i < singularValues.length; i++) {
            singularValues[i] = (Double) svdResult._1.get(i);
        }
        IMatrix<Double> V = svdResult._2;

        int rank = 0;
        double tol = 1e-4;
        for (double sv : singularValues) {
            if (sv > tol) rank++;
        }

        this.scalings = new double[nFeatures][rank];
        for (int j = 0; j < rank; j++) {
            for (int i = 0; i < nFeatures; i++) {
                scalings[i][j] = (Double) V.get(i, j) / std[i] / singularValues[j];
            }
        }

        double[][] meansDiff = new double[nClasses][nFeatures];
        for (int c = 0; c < nClasses; c++) {
            for (int j = 0; j < nFeatures; j++) {
                meansDiff[c][j] = Math.sqrt(nSamples * priors[c] * fac) * (means[c][j] - xbar[j]);
            }
        }

        IMatrix<Double> meansDiffMatrix = IMatrix.of(meansDiff);
        IMatrix<Double> scaledMeansDiff = meansDiffMatrix.mmul(IMatrix.of(scalings));

        Tuple2<IVector<Double>, IMatrix<Double>> svdResult2 = scaledMeansDiff.eigen();
        this.eigenvalues = new double[svdResult2._1.size()];
        for (int i = 0; i < eigenvalues.length; i++) {
            eigenvalues[i] = (Double) svdResult2._1.get(i);
        }
        IMatrix<Double> V2 = svdResult2._2;

        this.eigenvectors = new double[nFeatures][rank];
        for (int i = 0; i < nFeatures; i++) {
            for (int j = 0; j < rank; j++) {
                for (int k = 0; k < nFeatures; k++) {
                    eigenvectors[i][j] += scalings[k][j] * (Double) V2.get(k, j);
                }
            }
        }

        this.coef = new double[nClasses][nFeatures];
        for (int c = 0; c < nClasses; c++) {
            for (int j = 0; j < nFeatures; j++) {
                for (int i = 0; i < rank; i++) {
                    coef[c][j] += (means[c][i] - xbar[i]) * eigenvectors[j][i];
                }
            }
        }

        this.intercept = new double[nClasses];
        for (int c = 0; c < nClasses; c++) {
            double sum = 0;
            for (int j = 0; j < nFeatures; j++) {
                sum += coef[c][j] * coef[c][j];
            }
            intercept[c] = -0.5 * sum + Math.log(priors[c]);
        }

        if (nClasses == 2) {
            double[] coef1 = coef[0];
            coef[0] = new double[nFeatures];
            for (int j = 0; j < nFeatures; j++) {
                coef[0][j] = coef1[j] - coef[1][j];
            }
            intercept[0] = intercept[0] - intercept[1];
        }
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
            decision[c] = intercept[c];
            for (int j = 0; j < nFeatures; j++) {
                decision[c] += coef[c][j] * (Double) x.get(j);
            }
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
        if (!trained) {
            throw new IllegalStateException("模型尚未训练");
        }
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
        if (!trained) {
            throw new IllegalStateException("模型尚未训练");
        }
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
    public boolean isTrained() {
        return trained;
    }

    @Override
    public ClassificationMetrics getMetrics() {
        throw new UnsupportedOperationException("请使用 ClassificationMetrics.compute() 方法");
    }

    @Override
    public void setMetrics(ClassificationMetrics metrics) {
    }

    // ==================== JSON persistence ====================

    @Override
    public Map<String, Object> toParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("solver", solver.name());
        p.put("nComponents", nComponents);
        p.put("priors", priors.clone());
        p.put("classes", classes.clone());
        p.put("nClasses", nClasses);
        p.put("means", means.clone());
        p.put("xbar", xbar != null ? xbar.clone() : null);
        p.put("scalings", scalings.clone());
        if (eigenvectors != null) p.put("eigenvectors", eigenvectors.clone());
        if (eigenvalues != null) p.put("eigenvalues", eigenvalues.clone());
        if (coef != null) p.put("coef", coef.clone());
        if (intercept != null) p.put("intercept", intercept.clone());
        p.put("nFeatures", nFeatures);
        p.put("trained", trained);
        return p;
    }

    @Override
    public void fromParams(Map<String, Object> p) {
        this.solver = Solver.valueOf((String) p.get("solver"));
        this.nComponents = ((Number) p.get("nComponents")).intValue();
        this.priors = (double[]) p.get("priors");
        this.classes = ((List<String>) p.get("classes")).toArray(new String[0]);
        this.nClasses = ((Number) p.get("nClasses")).intValue();
        this.means = (double[][]) p.get("means");
        this.xbar = (double[]) p.get("xbar");
        this.scalings = (double[][]) p.get("scalings");
        this.eigenvectors = (double[][]) p.get("eigenvectors");
        this.eigenvalues = (double[]) p.get("eigenvalues");
        this.coef = (double[][]) p.get("coef");
        this.intercept = (double[]) p.get("intercept");
        this.nFeatures = ((Number) p.get("nFeatures")).intValue();
        this.trained = (Boolean) p.get("trained");
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
            public String getModelTypeDescription() {
                return "Linear Discriminant Analysis (LDA)";
            }

            @Override
            public String getModelSummary() {
                return "LDA Model: " + nClasses + " classes, " + nFeatures + " features";
            }

            @Override
            public boolean isTrained() {
                return trained;
            }
        };
    }

    public double[][] getCoefficients() { return coef; }
    public double[] getIntercepts() { return intercept; }
    public String[] getClasses() { return classes; }
    public double[][] getMeans() { return means; }
    public double[] getEigenvalues() { return eigenvalues; }
    public double[][] getEigenvectors() { return eigenvectors; }
    public int getNComponents() { return nComponents; }
    public void setNComponents(int n) { this.nComponents = n; }
    public Solver getSolver() { return solver; }
    public void setSolver(Solver solver) { this.solver = solver; }
}
