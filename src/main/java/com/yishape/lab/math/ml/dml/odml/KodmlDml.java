package com.yishape.lab.math.ml.dml.odml;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.dml.DmlArrays;
import com.yishape.lab.math.ml.dml.DmlMetric;
import com.yishape.lab.math.ml.dml.KernelDmlUtils;
import com.yishape.lab.math.ml.dml.KernelDmlUtils.KernelType;
import com.yishape.lab.util.Tuple2;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import com.yishape.lab.math.ml.dml.ISupervisedDml;

/**
 * 核化开集距离度量学习 (KODML): ODML 的核化版本。
 *
 * <p><strong>注意</strong>：此类与 pyDML 的 KDMLMJ 算法<strong>完全不同</strong>。
 * pyDML KDMLMJ 基于 Jeffrey 散度最大化，而本类使用三元组铰链损失梯度下降。
 * 类名保留是为了历史兼容性。</p>
 *
 * <p>本类实现 {@link ISupervisedDml}。</p>
 */
public final class KodmlDml implements ISupervisedDml {

    private KernelType kernelType = KernelType.RBF;
    private double gamma = 1.0;
    private int degree = 3;
    private double coef0 = 1.0;
    private double alpha = 1.0;
    private int maxIter = 100;
    private double learningRate = 0.01;
    private Random random;

    public KodmlDml setKernelType(KernelType kernelType) {
        this.kernelType = kernelType;
        return this;
    }

    public KodmlDml setGamma(double gamma) {
        this.gamma = gamma;
        return this;
    }

    public KodmlDml setAlpha(double alpha) {
        this.alpha = alpha;
        return this;
    }

    public KodmlDml setLearningRate(double lr) {
        this.learningRate = lr;
        return this;
    }

    public KodmlDml setRandom(Random random) {
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

    public static DmlMetric fit(IMatrix<Double> features, String[] labels, KodmlDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    DmlMetric fitFromRows(double[][] x, int[] y) {
        int n = x.length;
        int d = x[0].length;
        Random rnd = random != null ? random : ThreadLocalRandom.current();

        // 核矩阵
        double[][] K = KernelDmlUtils.kernelMatrix(x, x, kernelType, gamma, degree, coef0);
        double[][] Kc = KernelDmlUtils.centerKernelMatrix(K);

        int nClasses = distinctCount(y);

        // 学习核空间中的度量 A
        double[][] A = new double[n][n];
        for (int i = 0; i < n; i++) {
            A[i][i] = 1.0;
        }

        for (int iter = 0; iter < maxIter; iter++) {
            // 生成三元组
            List<int[]> triplets = generateTriplets(x, y, n, rnd);

            for (int[] t : triplets) {
                int i = t[0], j = t[1], k = t[2];

                // 计算核空间中的距离: d_A(phi(i), phi(j)) = (Kc_i - Kc_j)^T A (Kc_i - Kc_j)
                double dist_j = 0, dist_k = 0;
                for (int a = 0; a < n; a++) {
                    for (int b = 0; b < n; b++) {
                        double d_ij_a = Kc[i][a] - Kc[j][a];
                        double d_ij_b = Kc[i][b] - Kc[j][b];
                        double d_ik_a = Kc[i][a] - Kc[k][a];
                        double d_ik_b = Kc[i][b] - Kc[k][b];
                        dist_j += d_ij_a * A[a][b] * d_ij_b;
                        dist_k += d_ik_a * A[a][b] * d_ik_b;
                    }
                }

                // 梯度
                double margin = alpha * (dist_j - dist_k);
                if (margin > 0) {
                    for (int a = 0; a < n; a++) {
                        for (int b = 0; b < n; b++) {
                            double d_ij_ab = (Kc[i][a] - Kc[j][a]) * (Kc[i][b] - Kc[j][b]);
                            double d_ik_ab = (Kc[i][a] - Kc[k][a]) * (Kc[i][b] - Kc[k][b]);
                            A[a][b] -= learningRate * (d_ij_ab - d_ik_ab);
                        }
                    }
                }
            }

            // 投影到 PSD
            A = projectPSD(A, n);
        }

        return DmlMetric.fullWhitening(IMatrix.of(A));
    }

    private List<int[]> generateTriplets(double[][] x, int[] y, int n, Random rnd) {
        List<int[]> triplets = new ArrayList<>();
        Set<Integer> seen = new HashSet<>();

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (y[i] == y[j] && i != j) {
                    int k = rnd.nextInt(n);
                    if (y[i] != y[k]) {
                        int[] t = new int[]{i, j, k};
                        int key = i * n * n + j * n + k;
                        if (!seen.contains(key)) {
                            triplets.add(t);
                            seen.add(key);
                        }
                    }
                }
            }
        }
        return triplets;
    }

    private double[][] projectPSD(double[][] A, int n) {
        IMatrix<Double> M = IMatrix.of(A);
        try {
            Tuple2<IVector<Double>, IMatrix<Double>> result = M.eigen();
            IVector<Double> evals = result._1;
            IMatrix<Double> evecs = result._2;

            double[][] result_mat = new double[n][n];
            for (int i = 0; i < n; i++) {
                double eval_i = Math.max(0, (Double) evals.get(i));
                for (int j = 0; j < n; j++) {
                    for (int k = 0; k < n; k++) {
                        result_mat[j][k] += eval_i * (Double) evecs.get(j, i) * (Double) evecs.get(k, i);
                    }
                }
            }
            return result_mat;
        } catch (Exception e) {
            return A;
        }
    }

    private static int distinctCount(int[] arr) {
        Set<Integer> set = new HashSet<>();
        for (int v : arr) set.add(v);
        return set.size();
    }
}
