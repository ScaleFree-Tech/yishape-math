package com.yishape.lab.math.ml.dml.lmnn;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.dml.DmlArrays;
import com.yishape.lab.math.ml.dml.DmlMetric;
import com.yishape.lab.math.ml.dml.KernelDmlUtils;
import com.yishape.lab.math.ml.dml.KernelDmlUtils.KernelType;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import com.yishape.lab.math.ml.dml.ISupervisedDml;

/**
 * 核化大间隔最近邻 (KLMNN): LMNN 的核化版本，在核诱导特征空间中学习度量。
 *
 * <p>本类实现 {@link ISupervisedDml}。</p>
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li> Weinberger, K. Q., &amp; Saul, L. K. (2009). Distance metric learning for large margin nearest
 *       neighbor classification. <em>JMLR</em>, 10, 207–244.</li>
 * </ul>
 */
public final class KlmmnDml implements ISupervisedDml {

    private KernelType kernelType = KernelType.RBF;
    private double gamma = 1.0;
    private int degree = 3;
    private double coef0 = 1.0;
    private int rank = 2;
    private int targetNeighbors = 1;
    private double margin = 1.0;
    private int maxSteps = 500;
    private double learningRate = 0.05;
    private Random random;

    public KlmmnDml setKernelType(KernelType kernelType) {
        this.kernelType = kernelType;
        return this;
    }

    public KlmmnDml setGamma(double gamma) {
        this.gamma = gamma;
        return this;
    }

    public KlmmnDml setRank(int rank) {
        this.rank = rank;
        return this;
    }

    public KlmmnDml setTargetNeighbors(int k) {
        this.targetNeighbors = k;
        return this;
    }

    public KlmmnDml setMargin(double margin) {
        this.margin = margin;
        return this;
    }

    public KlmmnDml setMaxSteps(int maxSteps) {
        this.maxSteps = maxSteps;
        return this;
    }

    public KlmmnDml setLearningRate(double lr) {
        this.learningRate = lr;
        return this;
    }

    public KlmmnDml setRandom(Random random) {
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

    public static DmlMetric fit(IMatrix<Double> features, String[] labels, KlmmnDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    DmlMetric fitFromRows(double[][] x, int[] y) {
        int n = x.length;
        int d = x[0].length;
        Random rnd = random != null ? random : ThreadLocalRandom.current();

        // 计算核矩阵
        double[][] K = KernelDmlUtils.kernelMatrix(x, x, kernelType, gamma, degree, coef0);

        // 中心化核矩阵
        double[][] Kc = KernelDmlUtils.centerKernelMatrix(K);

        // 目标近邻
        List<int[]>[] targets = buildTargetNeighbors(x, y, n, targetNeighbors);

        int r = Math.min(Math.max(1, rank), n);

        // 在核空间中学习低秩变换 A (r x n)
        double[][] A = new double[r][n];
        for (int i = 0; i < r; i++) {
            A[i][i] = 1.0;
        }

        for (int step = 0; step < maxSteps; step++) {
            int i = rnd.nextInt(n);
            List<int[]> ti = targets[i];
            if (ti.isEmpty()) continue;
            int j = ti.get(rnd.nextInt(ti.size()))[0];
            int k = rnd.nextInt(n);
            int guard = 0;
            while (y[k] == y[i] && guard++ < n) {
                k = rnd.nextInt(n);
            }
            if (y[k] == y[i]) continue;

            // 核空间中的嵌入: z[a] = sum_b A[a][b] * Kc[b][sample]
            double[] zi = new double[r];
            double[] zj = new double[r];
            double[] zk = new double[r];
            for (int a = 0; a < r; a++) {
                for (int b = 0; b < n; b++) {
                    zi[a] += A[a][b] * Kc[b][i];
                    zj[a] += A[a][b] * Kc[b][j];
                    zk[a] += A[a][b] * Kc[b][k];
                }
            }

            // 计算 ||z_i - z_j||^2 和 ||z_i - z_k||^2
            double d1 = 0, d2 = 0;
            for (int a = 0; a < r; a++) {
                double diff1 = zi[a] - zj[a];
                double diff2 = zi[a] - zk[a];
                d1 += diff1 * diff1;
                d2 += diff2 * diff2;
            }

            double hinge = margin + d1 - d2;
            if (hinge <= 0) continue;

            // 梯度: d(||z_i-z_j||^2)/dA[a][b] = 2*(z_i[a]-z_j[a])*(Kc[b][i]-Kc[b][j])
            double lr2 = learningRate * 2.0;
            for (int a = 0; a < r; a++) {
                double diff_ij = zi[a] - zj[a];
                double diff_ik = zi[a] - zk[a];
                for (int b = 0; b < n; b++) {
                    double dK_ij = Kc[b][i] - Kc[b][j];
                    double dK_ik = Kc[b][i] - Kc[b][k];
                    A[a][b] -= lr2 * (diff_ij * dK_ij - diff_ik * dK_ik);
                }
            }
        }

        // 构建度量: M = A^T K A (简化为返回低秩表示)
        return DmlMetric.lowRank(Linalg.matrix(A));
    }

    @SuppressWarnings("unchecked")
    private static List<int[]>[] buildTargetNeighbors(double[][] x, int[] y, int n, int k) {
        List<int[]>[] targets = new List[n];
        for (int i = 0; i < n; i++) {
            targets[i] = new ArrayList<>();
        }
        for (int i = 0; i < n; i++) {
            List<double[]> cand = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                if (i == j || y[j] != y[i]) continue;
                double dist = 0;
                for (int t = 0; t < x[i].length; t++) {
                    double u = x[i][t] - x[j][t];
                    dist += u * u;
                }
                cand.add(new double[]{j, dist});
            }
            cand.sort((a, b) -> Double.compare(a[1], b[1]));
            for (int t = 0; t < Math.min(k, cand.size()); t++) {
                targets[i].add(new int[]{(int) cand.get(t)[0]});
            }
        }
        return targets;
    }
}
