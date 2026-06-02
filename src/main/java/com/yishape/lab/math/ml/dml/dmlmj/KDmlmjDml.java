package com.yishape.lab.math.ml.dml.dmlmj;

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
 * 核化 DML-MJ (KDMLMJ): DMLMJ 的核化版本。
 *
 * <p>本类实现 {@link ISupervisedDml}。</p>
 */
public final class KDmlmjDml implements ISupervisedDml {

    private KernelType kernelType = KernelType.RBF;
    private double gamma = 1.0;
    private int degree = 3;
    private double coef0 = 1.0;
    private double alpha = 1.0;
    private int numDims = -1; // -1 means use all
    private int maxIter = 100;
    private Random random;

    public KDmlmjDml setKernelType(KernelType kernelType) {
        this.kernelType = kernelType;
        return this;
    }

    public KDmlmjDml setGamma(double gamma) {
        this.gamma = gamma;
        return this;
    }

    public KDmlmjDml setAlpha(double alpha) {
        this.alpha = alpha;
        return this;
    }

    public KDmlmjDml setNumDims(int numDims) {
        this.numDims = numDims;
        return this;
    }

    public KDmlmjDml setRandom(Random random) {
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

    public static DmlMetric fit(IMatrix<Double> features, String[] labels, KDmlmjDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    DmlMetric fitFromRows(double[][] x, int[] y) {
        int n = x.length;
        int d = x[0].length;
        Random rnd = random != null ? random : ThreadLocalRandom.current();

        // 核矩阵
        double[][] K = KernelDmlUtils.kernelMatrix(x, x, kernelType, gamma, degree, coef0);
        double[][] Kc = KernelDmlUtils.centerKernelMatrix(K);

        // 核空间散点矩阵: S[a][b] = sum_{y_i=y_j} (Kc[i][a]-Kc[j][a])*(Kc[i][b]-Kc[j][b])
        double[][] S = new double[n][n];
        double[][] D = new double[n][n];
        int pairCount = 0;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (y[i] == y[j]) {
                    for (int a = 0; a < n; a++) {
                        for (int b = 0; b < n; b++) {
                            double outer = (Kc[i][a] - Kc[j][a]) * (Kc[i][b] - Kc[j][b]);
                            S[a][b] += outer;
                        }
                    }
                } else {
                    for (int a = 0; a < n; a++) {
                        for (int b = 0; b < n; b++) {
                            double outer = (Kc[i][a] - Kc[j][a]) * (Kc[i][b] - Kc[j][b]);
                            D[a][b] += outer;
                        }
                    }
                }
                pairCount++;
            }
        }

        if (pairCount > 0) {
            for (int a = 0; a < n; a++) {
                for (int b = 0; b < n; b++) {
                    S[a][b] /= pairCount;
                    D[a][b] /= pairCount;
                }
            }
        }

        IMatrix<Double> S_mat = IMatrix.of(S);
        IMatrix<Double> D_mat = IMatrix.of(D);

        try {
            IMatrix<Double> D_inv = D_mat.inv();
            IMatrix<Double> M = D_inv.mmul(S_mat);

            Tuple2<IVector<Double>, IMatrix<Double>> eigenResult = M.eigen();
            IVector<Double> evals = eigenResult._1;
            IMatrix<Double> evecs = eigenResult._2;

            // 排序特征值 (降序)
            int numEigen = Math.min(n, numDims > 0 ? numDims : n);
            Integer[] indices = new Integer[evals.size()];
            for (int i = 0; i < indices.length; i++) indices[i] = i;
            Arrays.sort(indices, (a, b) -> Double.compare((Double) evals.get(b), (Double) evals.get(a)));

            // Pre-image: 将核空间特征向量投影回 d 维原始特征空间
            // 构建 L (nd × d) 矩阵
            double[][] L = new double[numEigen][d];
            for (int i = 0; i < numEigen; i++) {
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
