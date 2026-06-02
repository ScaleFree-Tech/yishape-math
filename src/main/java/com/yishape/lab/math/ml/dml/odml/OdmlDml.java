package com.yishape.lab.math.ml.dml.odml;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.dml.DmlArrays;
import com.yishape.lab.math.ml.dml.DmlMetric;

import java.util.*;

import com.yishape.lab.util.YishapeLogger;

import com.yishape.lab.math.ml.dml.ISupervisedDml;

/**
 * Online Distance Metric Learning (ODML): 在线距离度量学习。
 *
 * <p>ODML 通过在线方式学习马氏度量，每次接收一个样本并更新度量矩阵。
 * 使用被动攻击算法 (Passive-Aggressive) 风格的更新规则。</p>
 *
 * <p>本类实现 {@link ISupervisedDml}。</p>
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>Shalev-Shwartz, S., et al. "Online metric learning and fast similarity search".
 *       <em>NIPS</em>, 2006.</li>
 * </ul>
 */
public final class OdmlDml implements ISupervisedDml {

    private static final YishapeLogger log = YishapeLogger.getLogger(OdmlDml.class);

    private int dims = -1;
    private double learningRate = 0.01;
    private double aggression = 1.0;
    private int maxIter = 100;
    private boolean verbose = false;

    // 在线学习的度量矩阵 M (PSD)
    private IMatrix<Double> M;
    private Random random;

    public OdmlDml setLearningRate(double lr) {
        this.learningRate = lr;
        return this;
    }

    public OdmlDml setAggression(double a) {
        this.aggression = a;
        return this;
    }

    public OdmlDml setMaxIter(int maxIter) {
        this.maxIter = maxIter;
        return this;
    }

    public OdmlDml setVerbose(boolean verbose) {
        this.verbose = verbose;
        return this;
    }

    public OdmlDml setRandom(Random random) {
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

    public static DmlMetric fit(IMatrix<Double> features, String[] labels, OdmlDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    DmlMetric fitFromRows(double[][] x, int[] y) {
        int n = x.length;
        int d = x[0].length;
        this.dims = d;
        Random rnd = random != null ? random : new Random(42);

        // 初始化 M 为单位矩阵
        M = Linalg.eye(d);

        // 在线学习迭代
        for (int iter = 0; iter < maxIter; iter++) {
            // 随机打乱顺序
            int[] order = new int[n];
            for (int i = 0; i < n; i++) order[i] = i;
            shuffle(order, rnd);

            boolean updated = false;
            for (int idx : order) {
                double[] xi = x[idx];
                int yi = y[idx];

                // 找到最近的同类和最近的异类样本
                double[] nearestSameClass = null;
                double[] nearestDiffClass = null;
                double minSameDist = Double.MAX_VALUE;
                double minDiffDist = Double.MAX_VALUE;

                for (int j = 0; j < n; j++) {
                    if (j == idx) continue;
                    double dist = mahalDist(xi, x[j]);
                    if (y[j] == yi && dist < minSameDist) {
                        minSameDist = dist;
                        nearestSameClass = x[j];
                    } else if (y[j] != yi && dist < minDiffDist) {
                        minDiffDist = dist;
                        nearestDiffClass = x[j];
                    }
                }

                if (nearestSameClass == null || nearestDiffClass == null) continue;

                // 计算距离
                double distSame = minSameDist;
                double distDiff = minDiffDist;

                // 铰链损失
                double loss = Math.max(0, 1 + distSame - distDiff);

                if (loss > 0) {
                    // 计算梯度
                    double[] diff1 = new double[d];
                    double[] diff2 = new double[d];
                    for (int k = 0; k < d; k++) {
                        diff1[k] = xi[k] - nearestSameClass[k];
                        diff2[k] = xi[k] - nearestDiffClass[k];
                    }

                    // 更新 M
                    double eta = Math.min(aggression, loss / (distDiff - distSame + 1e-10));
                    for (int a = 0; a < d; a++) {
                        for (int b = 0; b < d; b++) {
                            double grad = eta * (diff2[a] * diff2[b] - diff1[a] * diff1[b]);
                            double current = M.get(a, b);
                            M.set(a, b, current + learningRate * grad);
                        }
                    }

                    // 投影到 PSD
                    projectPSD();
                    updated = true;
                }
            }

            if (!updated && verbose) {
                log.info("ODML converged at iteration {}", iter);
                break;
            }
        }

        return DmlMetric.fullWhitening(M);
    }

    private double mahalDist(double[] a, double[] b) {
        double[] diff = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            diff[i] = a[i] - b[i];
        }

        // d^2 = (a-b)^T * M * (a-b)
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a.length; j++) {
                sum += diff[i] * M.get(i, j) * diff[j];
            }
        }
        return Math.sqrt(Math.max(0, sum));
    }

    private void projectPSD() {
        // 特征分解并投影负特征值
        try {
            var eigenResult = M.eigen();
            IVector<Double> evals = eigenResult._1;
            IMatrix<Double> evecs = eigenResult._2;

            double[][] M_new = new double[dims][dims];
            for (int i = 0; i < dims; i++) {
                double eval_i = Math.max(1e-10, (Double) evals.get(i));
                for (int j = 0; j < dims; j++) {
                    for (int k = 0; k < dims; k++) {
                        M_new[j][k] += eval_i * (Double) evecs.get(j, i) * (Double) evecs.get(k, i);
                    }
                }
            }

            for (int i = 0; i < dims; i++) {
                for (int j = 0; j < dims; j++) {
                    M.set(i, j, M_new[i][j]);
                }
            }
        } catch (Exception e) {
            // 如果分解失败，保持当前 M
        }
    }

    private static void shuffle(int[] arr, Random rnd) {
        for (int i = arr.length - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }
}
