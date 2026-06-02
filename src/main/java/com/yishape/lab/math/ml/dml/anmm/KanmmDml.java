package com.yishape.lab.math.ml.dml.anmm;

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
 * 核化平均近邻间隔最大化 (KANMM): ANMM 的核化版本。
 *
 * <p>本类实现 {@link ISupervisedDml}。</p>
 */
public final class KanmmDml implements ISupervisedDml {

    private KernelType kernelType = KernelType.RBF;
    private double gamma = 1.0;
    private int degree = 3;
    private double coef0 = 1.0;
    private int k = 3;
    private double mu = 1.0;
    private Integer numDims = null; // 输出维度
    private Random random;

    public KanmmDml setKernelType(KernelType kernelType) {
        this.kernelType = kernelType;
        return this;
    }

    public KanmmDml setGamma(double gamma) {
        this.gamma = gamma;
        return this;
    }

    public KanmmDml setK(int k) {
        this.k = k;
        return this;
    }

    public KanmmDml setMu(double mu) {
        this.mu = mu;
        return this;
    }

    public KanmmDml setNumDims(Integer numDims) {
        this.numDims = numDims;
        return this;
    }

    public KanmmDml setRandom(Random random) {
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

    public static DmlMetric fit(IMatrix<Double> features, String[] labels, KanmmDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    DmlMetric fitFromRows(double[][] x, int[] y) {
        int n = x.length;
        int d = x[0].length;
        Random rnd = random != null ? random : ThreadLocalRandom.current();

        // 核矩阵
        double[][] K = KernelDmlUtils.kernelMatrix(x, x, kernelType, gamma, degree, coef0);
        double[][] Kc = KernelDmlUtils.centerKernelMatrix(K);

        // 计算散点矩阵
        double[][] S = new double[n][n];
        double[][] D = new double[n][n];

        for (int i = 0; i < n; i++) {
            // 找同类最近邻和异类最近邻
            List<double[]> friends = new ArrayList<>();
            List<double[]> enemies = new ArrayList<>();

            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                double dist = euclideanDist(x[i], x[j]);
                if (y[i] == y[j]) {
                    friends.add(new double[]{j, dist});
                } else {
                    enemies.add(new double[]{j, dist});
                }
            }

            friends.sort(Comparator.comparingDouble(a -> a[1]));
            enemies.sort(Comparator.comparingDouble(a -> a[1]));

            // 同类近邻贡献到 S
            for (int t = 0; t < Math.min(k, friends.size()); t++) {
                int j = (int) friends.get(t)[0];
                for (int a = 0; a < n; a++) {
                    for (int b = 0; b < n; b++) {
                        S[a][b] += (Kc[i][a] - Kc[j][a]) * (Kc[i][b] - Kc[j][b]) / n;
                    }
                }
            }

            // 异类近邻贡献到 D
            for (int t = 0; t < Math.min(k, enemies.size()); t++) {
                int j = (int) enemies.get(t)[0];
                for (int a = 0; a < n; a++) {
                    for (int b = 0; b < n; b++) {
                        D[a][b] += (Kc[i][a] - Kc[j][a]) * (Kc[i][b] - Kc[j][b]) / n;
                    }
                }
            }
        }

        // 广义特征分解: D^-1 * S
        IMatrix<Double> S_mat = IMatrix.of(S);
        IMatrix<Double> D_mat = IMatrix.of(D);

        try {
            IMatrix<Double> D_inv = D_mat.inv();
            IMatrix<Double> M = D_inv.mmul(S_mat);

            Tuple2<IVector<Double>, IMatrix<Double>> eigenResult = M.eigen();
            IVector<Double> evals = eigenResult._1;
            IMatrix<Double> evecs = eigenResult._2;

            // 排序特征值 (降序)，取前 nd 个
            int nd = Math.min(numDims != null ? numDims : d, d);
            Integer[] indices = new Integer[evals.size()];
            for (int i = 0; i < indices.length; i++) indices[i] = i;
            Arrays.sort(indices, (a, b) -> Double.compare((Double) evals.get(b), (Double) evals.get(a)));

            // 构建 L (nd × d) 矩阵，每个特征向量 pre-image 投影回原始空间
            double[][] L = new double[nd][d];
            for (int i = 0; i < nd && i < indices.length; i++) {
                int idx = indices[i];
                for (int j = 0; j < n; j++) {
                    double vj = (Double) evecs.get(j, idx);
                    for (int k = 0; k < d; k++) {
                        L[i][k] += vj * x[j][k];
                    }
                }
            }

            return DmlMetric.lowRank(Linalg.matrix(L));
        } catch (Exception e) {
            return DmlMetric.fullWhitening(Linalg.eye(d));
        }
    }

    private static double euclideanDist(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }
}
