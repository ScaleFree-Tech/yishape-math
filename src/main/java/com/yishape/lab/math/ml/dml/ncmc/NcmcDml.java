package com.yishape.lab.math.ml.dml.ncmc;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.clu.KMeansPlusPlus;
import com.yishape.lab.math.ml.dml.DmlArrays;
import com.yishape.lab.math.ml.dml.DmlMetric;
import com.yishape.lab.util.Tuple2;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import com.yishape.lab.math.ml.dml.ISupervisedDml;

/**
 * Nearest Class with Multiple Centroids (NCMC): 通过学习线性变换 L 最大化期望成功率。
 * 每个类使用 KMeans 聚类得到多个中心，变换后样本到各类中心的 softmax 距离用于分类。
 *
 * <p>本类实现 {@link ISupervisedDml}。</p>
 *
 * @apiNote 应用代码优先 {@link com.yishape.lab.math.ml.ML#dml}。
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>Thomas Mensink et al. "Metric learning for large scale image classification:
 *       Generalizing to new classes at near-zero cost". <em>ECCV 2012</em>, pp. 488-501.</li>
 * </ul>
 */
public final class NcmcDml implements ISupervisedDml {

    private Integer numDims = null;
    private int centroidsNum = 3;
    private String learningRate = "adaptive";
    private double eta0 = 0.3;
    private Object initialTransform = null;
    private int maxIter = 300;
    private double tol = 1e-15;
    private double prec = 1e-15;
    private String descentMethod = "SGD";
    private double etaThres = 1e-14;
    private double learnInc = 1.01;
    private double learnDec = 0.5;
    private Random random;

    public NcmcDml setNumDims(Integer numDims) {
        this.numDims = numDims;
        return this;
    }

    public NcmcDml setCentroidsNum(int centroidsNum) {
        this.centroidsNum = centroidsNum;
        return this;
    }

    public NcmcDml setLearningRate(String learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public NcmcDml setEta0(double eta0) {
        this.eta0 = eta0;
        return this;
    }

    public NcmcDml setInitialTransform(Object initialTransform) {
        this.initialTransform = initialTransform;
        return this;
    }

    public NcmcDml setMaxIter(int maxIter) {
        this.maxIter = maxIter;
        return this;
    }

    public NcmcDml setDescentMethod(String descentMethod) {
        this.descentMethod = descentMethod;
        return this;
    }

    public NcmcDml setRandom(Random random) {
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

    public static DmlMetric fit(IMatrix<Double> features, String[] labels, NcmcDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    DmlMetric fitFromRows(double[][] x, int[] y) {
        int n = x.length;
        int d = x[0].length;
        Random rnd = random != null ? random : ThreadLocalRandom.current();

        int nd = (numDims != null) ? Math.min(numDims, d) : d;

        // 初始化 L
        double[][] L = initTransform(x, d, nd);

        // 计算每类 KMeans 聚类中心
        int[] classes = distinctIntArray(y);
        int numClasses = classes.length;
        int[] cn = new int[numClasses];
        Arrays.fill(cn, centroidsNum);

        List<double[]> centroidsList = new ArrayList<>();
        List<Integer> classStartList = new ArrayList<>();
        classStartList.add(0);

        for (int c = 0; c < numClasses; c++) {
            int cls = classes[c];
            List<double[]> classPoints = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (y[i] == cls) {
                    classPoints.add(x[i]);
                }
            }
            double[][] classData = classPoints.toArray(new double[0][]);

            KMeansPlusPlus kmeans = new KMeansPlusPlus(rnd.nextLong());
            kmeans.setParameters(Map.of("numClusters", cn[c]));
            kmeans.fit(IMatrix.of(classData));
            List<IVector<Double>> centers = kmeans.getClusterCenters();
            for (IVector<Double> center : centers) {
                double[] cArr = new double[d];
                for (int j = 0; j < d; j++) {
                    cArr[j] = center.get(j);
                }
                centroidsList.add(cArr);
            }
            cn[c] = centers.size();
            classStartList.add(centroidsList.size());
        }

        int totalCentroids = centroidsList.size();
        double[][] centroids = centroidsList.toArray(new double[0][]);
        int[] cs = new int[classStartList.size()];
        for (int i = 0; i < classStartList.size(); i++) {
            cs[i] = classStartList.get(i);
        }

        // 外积预计算
        double[][][] outers = computeOuters(x, centroids, n, totalCentroids, d);

        double eta = eta0;
        boolean adaptive = "adaptive".equals(learningRate);
        boolean sgd = "SGD".equals(descentMethod);

        double prevSucc = 0.0;
        int numIts = 0;
        boolean stop = false;

        while (!stop && numIts < maxIter) {
            if (sgd) {
                // SGD
                int[] perm = randomPermutation(n, rnd);
                for (int idx : perm) {
                    double[] grad = computeGradSgd(L, x, y, centroids, cs, cn, numClasses, outers, idx, d, nd);
                    if (!Double.isNaN(gradSum(grad, nd * d))) {
                        for (int a = 0; a < nd; a++) {
                            for (int b = 0; b < d; b++) {
                                L[a][b] += eta * grad[a * d + b];
                            }
                        }
                    }
                }
            } else {
                // BGD
                double[] grad = computeGradBgd(L, x, y, centroids, cs, cn, numClasses, outers, n, d, nd);
                for (int a = 0; a < nd; a++) {
                    for (int b = 0; b < d; b++) {
                        L[a][b] += eta * grad[a * d + b];
                    }
                }
            }

            if (adaptive) {
                double succ = computeExpectedSuccess(L, x, y, centroids, cs, n, nd, d);
                if (succ > prevSucc) {
                    eta *= learnInc;
                } else {
                    eta *= learnDec;
                }
                if (eta < etaThres) {
                    stop = true;
                }
                prevSucc = succ;
            }

            double gradNorm = sgd ? 0 : maxAbs(computeGradBgd(L, x, y, centroids, cs, cn, numClasses, outers, n, d, nd), nd * d);
            if (gradNorm < prec || eta * gradNorm < tol) {
                stop = true;
            }

            numIts++;
        }

        return DmlMetric.lowRank(Linalg.matrix(L));
    }

    private double[][] initTransform(double[][] x, int d, int nd) {
        double[][] L = new double[nd][d];
        if (initialTransform == null || "euclidean".equals(initialTransform)) {
            for (int i = 0; i < Math.min(nd, d); i++) {
                L[i][i] = 1.0;
            }
        } else if ("scale".equals(initialTransform)) {
            double[] maxVals = new double[d];
            double[] minVals = new double[d];
            Arrays.fill(maxVals, Double.NEGATIVE_INFINITY);
            Arrays.fill(minVals, Double.POSITIVE_INFINITY);
            for (double[] row : x) {
                for (int j = 0; j < d; j++) {
                    if (row[j] > maxVals[j]) maxVals[j] = row[j];
                    if (row[j] < minVals[j]) minVals[j] = row[j];
                }
            }
            for (int j = 0; j < Math.min(nd, d); j++) {
                double range = Math.max(maxVals[j] - minVals[j], 1e-16);
                L[j][j] = 1.0 / range;
            }
        }
        return L;
    }

    private double[][][] computeOuters(double[][] x, double[][] centroids, int n, int m, int d) {
        double[][][] outers = new double[n][m][d * d];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < m; k++) {
                for (int a = 0; a < d; a++) {
                    for (int b = 0; b < d; b++) {
                        outers[i][k][a * d + b] = x[i][a] * centroids[k][b];
                    }
                }
            }
        }
        return outers;
    }

    private double[] computeGradSgd(double[][] L, double[][] x, int[] y, double[][] centroids,
            int[] cs, int[] cn, int numClasses, double[][][] outers, int idx, int d, int nd) {
        // Lx_i
        double[] lxi = new double[nd];
        for (int a = 0; a < nd; a++) {
            double s = 0;
            for (int b = 0; b < d; b++) {
                s += L[a][b] * x[idx][b];
            }
            lxi[a] = s;
        }

        // Lm = L @ centroids^T
        double[][] lm = new double[centroids.length][nd];
        for (int k = 0; k < centroids.length; k++) {
            for (int a = 0; a < nd; a++) {
                double s = 0;
                for (int b = 0; b < d; b++) {
                    s += L[a][b] * centroids[k][b];
                }
                lm[k][a] = s;
            }
        }

        // dists_i = -0.5 * ||lxi - lm_k||^2
        double[] dists = new double[centroids.length];
        for (int k = 0; k < centroids.length; k++) {
            double s = 0;
            for (int a = 0; a < nd; a++) {
                double diff = lxi[a] - lm[k][a];
                s += diff * diff;
            }
            dists[k] = -0.5 * s;
        }

        int iMax = argMax(dists);
        double cmax = dists[iMax];
        double[] softmax = new double[centroids.length];
        double sumExp = 0;
        for (int k = 0; k < centroids.length; k++) {
            softmax[k] = Math.exp(dists[k] - cmax);
            sumExp += softmax[k];
        }
        softmax[iMax] = 1.0;
        sumExp = 1.0 + (sumExp - softmax[iMax]);

        for (int k = 0; k < centroids.length; k++) {
            softmax[k] /= sumExp;
        }

        // 计算 grad_sum
        double[] gradSum = new double[nd * d];
        int yi = y[idx];

        for (int c = 0; c < numClasses; c++) {
            int start = cs[c];
            int end = cs[c + 1];
            double clsSum = 0;
            for (int k = start; k < end; k++) {
                clsSum += softmax[k];
            }

            for (int k = start; k < end; k++) {
                double mask = (c == yi) ? softmax[k] / clsSum : 0.0;
                double coef = softmax[k] - mask;
                double[] outer_i_k = outers[idx][k];
                for (int a = 0; a < nd; a++) {
                    for (int b = 0; b < d; b++) {
                        // 修复 B008: 索引应为 [a * d + b]，与 outers 的存储格式一致
                        gradSum[a * d + b] += coef * outer_i_k[a * d + b];
                    }
                }
            }
        }

        // grad = L @ gradSum
        double[] grad = new double[nd * d];
        for (int a = 0; a < nd; a++) {
            for (int b = 0; b < d; b++) {
                double s = 0;
                for (int c = 0; c < nd; c++) {
                    s += L[c][b] * gradSum[c * d + a];
                }
                grad[a * d + b] = s;
            }
        }
        return grad;
    }

    private double[] computeGradBgd(double[][] L, double[][] x, int[] y, double[][] centroids,
            int[] cs, int[] cn, int numClasses, double[][][] outers, int n, int d, int nd) {
        double[] gradSum = new double[nd * d];
        for (int idx = 0; idx < n; idx++) {
            double[] grad = computeGradSgd(L, x, y, centroids, cs, cn, numClasses, outers, idx, d, nd);
            for (int i = 0; i < gradSum.length; i++) {
                gradSum[i] += grad[i];
            }
        }
        double[] grad = new double[nd * d];
        for (int i = 0; i < nd * d; i++) {
            grad[i] = gradSum[i] / n;
        }
        return grad;
    }

    private double computeExpectedSuccess(double[][] L, double[][] x, int[] y, double[][] centroids, int[] cs, int n, int nd, int d) {
        double success = 0.0;

        // Lx = X @ L^T
        double[][] lx = new double[n][nd];
        for (int i = 0; i < n; i++) {
            for (int a = 0; a < nd; a++) {
                double s = 0;
                for (int b = 0; b < d; b++) {
                    s += x[i][b] * L[a][b];
                }
                lx[i][a] = s;
            }
        }

        // Lm = centroids @ L^T
        double[][] lm = new double[centroids.length][nd];
        for (int k = 0; k < centroids.length; k++) {
            for (int a = 0; a < nd; a++) {
                double s = 0;
                for (int b = 0; b < d; b++) {
                    s += centroids[k][b] * L[a][b];
                }
                lm[k][a] = s;
            }
        }

        for (int i = 0; i < n; i++) {
            double[] dists = new double[centroids.length];
            for (int k = 0; k < centroids.length; k++) {
                double s = 0;
                for (int a = 0; a < nd; a++) {
                    double diff = lx[i][a] - lm[k][a];
                    s += diff * diff;
                }
                dists[k] = -0.5 * s;
            }

            double[] softmax = new double[centroids.length];
            double sumExp = 0;
            for (int k = 0; k < centroids.length; k++) {
                softmax[k] = Math.exp(dists[k]);
                sumExp += softmax[k];
            }

            int yi = y[i];
            double clsSum = 0;
            for (int k = cs[yi]; k < cs[yi + 1]; k++) {
                clsSum += softmax[k];
            }
            success += Math.log(Math.max(clsSum / sumExp, 1e-15));
        }

        return success / n;
    }

    private static int[] distinctIntArray(int[] arr) {
        IntHashSet set = new IntHashSet();
        for (int v : arr) set.add(v);
        int[] result = set.toArray();
        Arrays.sort(result);
        return result;
    }

    private static int[] randomPermutation(int n, Random rnd) {
        int[] perm = new int[n];
        for (int i = 0; i < n; i++) perm[i] = i;
        for (int i = n - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int tmp = perm[i];
            perm[i] = perm[j];
            perm[j] = tmp;
        }
        return perm;
    }

    private static int argMax(double[] arr) {
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

    private static double gradSum(double[] arr, int len) {
        double s = 0;
        for (int i = 0; i < len; i++) s += arr[i];
        return s;
    }

    private static double maxAbs(double[] arr, int len) {
        double m = 0;
        for (int i = 0; i < len; i++) {
            double a = Math.abs(arr[i]);
            if (a > m) m = a;
        }
        return m;
    }

    private static class IntHashSet {
        private IntArrayList list = new IntArrayList();

        void add(int v) {
            for (int i = 0; i < list.size(); i++) {
                if (list.get(i) == v) return;
            }
            list.add(v);
        }

        int[] toArray() {
            return list.toArray();
        }
    }

    private static class IntArrayList {
        private int[] arr = new int[16];
        private int sz = 0;

        void add(int v) {
            if (sz == arr.length) {
                int[] newArr = new int[arr.length * 2];
                System.arraycopy(arr, 0, newArr, 0, sz);
                arr = newArr;
            }
            arr[sz++] = v;
        }

        int get(int i) {
            return arr[i];
        }

        int size() {
            return sz;
        }

        int[] toArray() {
            int[] result = new int[sz];
            System.arraycopy(arr, 0, result, 0, sz);
            return result;
        }
    }
}
