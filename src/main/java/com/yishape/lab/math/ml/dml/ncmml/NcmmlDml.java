package com.yishape.lab.math.ml.dml.ncmml;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.dml.DmlArrays;
import com.yishape.lab.math.ml.dml.DmlMetric;

import java.util.Objects;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import com.yishape.lab.math.ml.dml.ISupervisedDml;

/**
 * Nearest Class Mean Metric Learning (NCMML)：通过梯度下降最大化期望成功率的类均值度量学习。
 *
 * <p>本类实现 {@link ISupervisedDml}。</p>
 *
 * @apiNote 应用代码优先 {@link com.yishape.lab.math.ml.ML#dml}。
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>Mensink, T., et al. (2012). Metric learning for large scale image classification:
 *       Generalizing to new classes at near-zero cost. <em>ECCV</em>.</li>
 * </ul>
 */
public final class NcmmlDml implements ISupervisedDml {

    private Integer numDims;
    private String learningRate = "adaptive";
    private double eta0 = 0.3;
    private String initialTransform = "euclidean";
    private int maxIter = 300;
    private double prec = 1e-15;
    private double tol = 1e-15;
    private String descentMethod = "SGD";
    private double etaThres = 1e-14;
    private double learnInc = 1.01;
    private double learnDec = 0.5;
    private Random random;

    public Integer getNumDims() {
        return numDims;
    }

    public NcmmlDml setNumDims(Integer numDims) {
        this.numDims = numDims;
        return this;
    }

    public String getLearningRate() {
        return learningRate;
    }

    public NcmmlDml setLearningRate(String learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public double getEta0() {
        return eta0;
    }

    public NcmmlDml setEta0(double eta0) {
        this.eta0 = eta0;
        return this;
    }

    public String getInitialTransform() {
        return initialTransform;
    }

    public NcmmlDml setInitialTransform(String initialTransform) {
        this.initialTransform = initialTransform;
        return this;
    }

    public int getMaxIter() {
        return maxIter;
    }

    public NcmmlDml setMaxIter(int maxIter) {
        this.maxIter = maxIter;
        return this;
    }

    public double getPrec() {
        return prec;
    }

    public NcmmlDml setPrec(double prec) {
        this.prec = prec;
        return this;
    }

    public double getTol() {
        return tol;
    }

    public NcmmlDml setTol(double tol) {
        this.tol = tol;
        return this;
    }

    public String getDescentMethod() {
        return descentMethod;
    }

    public NcmmlDml setDescentMethod(String descentMethod) {
        this.descentMethod = descentMethod;
        return this;
    }

    public Random getRandom() {
        return random;
    }

    public NcmmlDml setRandom(Random random) {
        this.random = random;
        return this;
    }

    @Override
    public DmlMetric fit(IMatrix<Double> features, IVector<?> labels) {
        Objects.requireNonNull(labels, "labels");
        return fit(features, DmlArrays.stringLabels(labels));
    }

    @Override
    public DmlMetric fit(IMatrix<Double> features, String[] labels) {
        Objects.requireNonNull(features, "features");
        Objects.requireNonNull(labels, "labels");
        double[][] x = DmlArrays.featureRows(features);
        int[] y = DmlArrays.classIndices(labels);
        return fitFromRows(x, y);
    }

    public static DmlMetric fit(IMatrix<Double> features, String[] labels, NcmmlDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    public static DmlMetric fit(IMatrix<Double> features, IVector<?> labels, NcmmlDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    DmlMetric fitFromRows(double[][] x, int[] y) {
        int n = x.length;
        int d = x[0].length;
        Random rnd = random != null ? random : ThreadLocalRandom.current();

        int nd = (numDims != null) ? Math.min(numDims, d) : d;

        int numClasses = 0;
        for (int yi : y) {
            if (yi + 1 > numClasses) {
                numClasses = yi + 1;
            }
        }

        double[][] centroids = computeClassCentroids(x, y, numClasses);

        double[][] L;
        if (initialTransform == null || "euclidean".equals(initialTransform)) {
            L = new double[nd][d];
            for (int i = 0; i < nd; i++) {
                L[i][i] = 1.0;
            }
        } else if ("scale".equals(initialTransform)) {
            L = new double[nd][d];
            for (int j = 0; j < d; j++) {
                double maxVal = Double.NEGATIVE_INFINITY;
                double minVal = Double.POSITIVE_INFINITY;
                for (int i = 0; i < n; i++) {
                    maxVal = Math.max(maxVal, x[i][j]);
                    minVal = Math.min(minVal, x[i][j]);
                }
                double range = Math.max(maxVal - minVal, 1e-16);
                L[j][j] = 1.0 / range;
            }
        } else {
            throw new IllegalArgumentException("initialTransform 仅支持 null/euclidean/scale");
        }

        boolean adaptive = "adaptive".equals(learningRate);
        double eta = eta0;

        if ("SGD".equals(descentMethod)) {
            L = sgdFit(x, y, L, centroids, eta, adaptive, rnd);
        } else {
            L = bgdFit(x, y, L, centroids, eta, adaptive);
        }

        return DmlMetric.lowRank(Linalg.matrix(L));
    }

    private double[][] sgdFit(double[][] x, int[] y, double[][] L,
            double[][] centroids, double eta, boolean adaptive, Random rnd) {
        int n = x.length;
        int d = x[0].length;
        int nd = L.length;
        int numClasses = centroids.length;

        double prevSucc = 0.0;
        double succ = 0.0;

        for (int iter = 0; iter < maxIter; iter++) {
            int[] perm = rnd.ints(0, n).distinct().limit(n).toArray();

            double[][] gradSum = new double[nd][d];

            for (int idx : perm) {
                double[] lxi = multiply(L, x[idx]);
                double[] muDiff = new double[nd];
                for (int c = 0; c < numClasses; c++) {
                    for (int k = 0; k < nd; k++) {
                        muDiff[k] = lxi[k] - centroids[c][k];
                    }
                }

                double[] dists = new double[numClasses];
                for (int c = 0; c < numClasses; c++) {
                    double sum = 0;
                    for (int k = 0; k < nd; k++) {
                        double diff = lxi[k] - centroids[c][k];
                        sum += diff * diff;
                    }
                    dists[c] = -0.5 * sum;
                }

                double maxDist = Double.NEGATIVE_INFINITY;
                for (int c = 0; c < numClasses; c++) {
                    if (dists[c] > maxDist) {
                        maxDist = dists[c];
                    }
                }

                double sumExp = 0;
                double[] softmax = new double[numClasses];
                for (int c = 0; c < numClasses; c++) {
                    if (Double.isInfinite(maxDist)) {
                        softmax[c] = (c == argmax(dists)) ? 1.0 : 0.0;
                    } else {
                        softmax[c] = Math.exp(dists[c] - maxDist);
                    }
                    sumExp += softmax[c];
                }
                if (sumExp > 0) {
                    for (int c = 0; c < numClasses; c++) {
                        softmax[c] /= sumExp;
                    }
                }

                int yi = y[idx];
                for (int c = 0; c < numClasses; c++) {
                    double mask = (c == yi) ? 1.0 : 0.0;
                    double factor = (softmax[c] - mask);
                    for (int k = 0; k < nd; k++) {
                        double z_k = 0;
                        for (int a = 0; a < d; a++) {
                            z_k += L[k][a] * (x[idx][a] - centroids[c][a]);
                        }
                        for (int j = 0; j < d; j++) {
                            gradSum[k][j] += factor * z_k * (x[idx][j] - centroids[c][j]);
                        }
                    }
                }
            }

            for (int k = 0; k < nd; k++) {
                for (int j = 0; j < d; j++) {
                    L[k][j] += eta * gradSum[k][j];
                }
            }

            if (adaptive) {
                succ = computeExpectedSuccess(L, x, y, centroids);
                if (succ > prevSucc) {
                    eta *= learnInc;
                } else {
                    eta *= learnDec;
                }
                if (eta < etaThres) {
                    break;
                }
                prevSucc = succ;
            }

            double gradNorm = 0;
            for (int k = 0; k < nd; k++) {
                for (int j = 0; j < d; j++) {
                    gradNorm = Math.max(gradNorm, Math.abs(gradSum[k][j]));
                }
            }
            if (gradNorm < prec || eta * gradNorm < tol) {
                break;
            }
        }

        return L;
    }

    private double[][] bgdFit(double[][] x, int[] y, double[][] L,
            double[][] centroids, double eta, boolean adaptive) {
        int n = x.length;
        int d = x[0].length;
        int nd = L.length;
        int numClasses = centroids.length;

        double prevSucc = 0.0;
        double succ = 0.0;

        for (int iter = 0; iter < maxIter; iter++) {
            double[][] gradSum = new double[nd][d];

            for (int i = 0; i < n; i++) {
                double[] lxi = multiply(L, x[i]);
                double[] dists = new double[numClasses];
                for (int c = 0; c < numClasses; c++) {
                    double sum = 0;
                    for (int k = 0; k < nd; k++) {
                        double diff = lxi[k] - centroids[c][k];
                        sum += diff * diff;
                    }
                    dists[c] = -0.5 * sum;
                }

                double sumExp = 0;
                double[] softmax = new double[numClasses];
                for (int c = 0; c < numClasses; c++) {
                    softmax[c] = Math.exp(dists[c]);
                    sumExp += softmax[c];
                }
                if (sumExp > 0) {
                    for (int c = 0; c < numClasses; c++) {
                        softmax[c] /= sumExp;
                    }
                }

                int yi = y[i];
                for (int c = 0; c < numClasses; c++) {
                    double mask = (c == yi) ? 1.0 : 0.0;
                    double factor = (softmax[c] - mask);
                    for (int k = 0; k < nd; k++) {
                        double z_k = 0;
                        for (int a = 0; a < d; a++) {
                            z_k += L[k][a] * (x[i][a] - centroids[c][a]);
                        }
                        for (int j = 0; j < d; j++) {
                            gradSum[k][j] += factor * z_k * (x[i][j] - centroids[c][j]);
                        }
                    }
                }
            }

            for (int k = 0; k < nd; k++) {
                for (int j = 0; j < d; j++) {
                    L[k][j] += (eta / n) * gradSum[k][j];
                }
            }

            if (adaptive) {
                succ = computeExpectedSuccess(L, x, y, centroids);
                if (succ > prevSucc) {
                    eta *= learnInc;
                } else {
                    eta *= learnDec;
                }
                if (eta < etaThres) {
                    break;
                }
                prevSucc = succ;
            }

            double gradNorm = 0;
            for (int k = 0; k < nd; k++) {
                for (int j = 0; j < d; j++) {
                    gradNorm = Math.max(gradNorm, Math.abs(gradSum[k][j]));
                }
            }
            if (gradNorm < prec || eta * gradNorm < tol) {
                break;
            }
        }

        return L;
    }

    private double[][] computeClassCentroids(double[][] x, int[] y, int numClasses) {
        int n = x.length;
        int d = x[0].length;
        int nd = (numDims != null) ? Math.min(numDims, d) : d;

        double[][] centroids = new double[numClasses][d];
        int[] counts = new int[numClasses];

        for (int i = 0; i < n; i++) {
            int c = y[i];
            counts[c]++;
            for (int j = 0; j < d; j++) {
                centroids[c][j] += x[i][j];
            }
        }

        for (int c = 0; c < numClasses; c++) {
            if (counts[c] > 0) {
                for (int j = 0; j < d; j++) {
                    centroids[c][j] /= counts[c];
                }
            }
        }

        return centroids;
    }

    private double computeExpectedSuccess(double[][] L, double[][] x, int[] y, double[][] centroids) {
        int n = x.length;
        int numClasses = centroids.length;
        double success = 0.0;

        for (int i = 0; i < n; i++) {
            double[] lxi = multiply(L, x[i]);
            double[] dists = new double[numClasses];
            for (int c = 0; c < numClasses; c++) {
                double sum = 0;
                for (int k = 0; k < L.length; k++) {
                    double diff = lxi[k] - centroids[c][k];
                    sum += diff * diff;
                }
                dists[c] = -0.5 * sum;
            }

            double sumExp = 0;
            double[] softmax = new double[numClasses];
            for (int c = 0; c < numClasses; c++) {
                softmax[c] = Math.exp(dists[c]);
                sumExp += softmax[c];
            }
            if (sumExp > 0) {
                for (int c = 0; c < numClasses; c++) {
                    softmax[c] /= sumExp;
                }
            }

            int yi = y[i];
            if (softmax[yi] > 0) {
                success += Math.log(softmax[yi]);
            }
        }

        return success / n;
    }

    private static double[] multiply(double[][] A, double[] v) {
        int rows = A.length;
        int cols = v.length;
        double[] result = new double[rows];
        for (int i = 0; i < rows; i++) {
            double sum = 0;
            for (int j = 0; j < cols; j++) {
                sum += A[i][j] * v[j];
            }
            result[i] = sum;
        }
        return result;
    }

    private static int argmax(double[] arr) {
        int idx = 0;
        double max = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
                idx = i;
            }
        }
        return idx;
    }
}
