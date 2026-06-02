package com.yishape.lab.math.ml.dml.kda;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.dml.DmlArrays;
import com.yishape.lab.math.ml.dml.DmlMetric;
import com.yishape.lab.math.ml.dml.KernelDmlUtils;
import com.yishape.lab.math.ml.dml.KernelDmlUtils.KernelType;
import com.yishape.lab.util.Tuple2;

import java.util.*;
import com.yishape.lab.math.ml.dml.ISupervisedDml;

/**
 * Kernel Discriminant Analysis (KDA): 核判别分析。
 * 在核诱导特征空间中执行 LDA。
 *
 * <p>本类实现 {@link ISupervisedDml}。</p>
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>Mika, S., et al. "Fisher discriminant analysis with kernels".
 *       <em>Neural Networks for Signal Processing IX</em>, 1999.</li>
 * </ul>
 */
public final class KdaDml implements ISupervisedDml {

    private Integer nComponents = null;
    private double tol = 1e-4;
    private double alpha = 1e-3;
    private KernelType kernelType = KernelType.RBF;
    private double gamma = 1.0;
    private int degree = 3;
    private double coef0 = 1.0;

    public KdaDml setNComponents(Integer n) {
        this.nComponents = n;
        return this;
    }

    public KdaDml setKernelType(KernelType kernelType) {
        this.kernelType = kernelType;
        return this;
    }

    public KdaDml setGamma(double gamma) {
        this.gamma = gamma;
        return this;
    }

    public KdaDml setDegree(int degree) {
        this.degree = degree;
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

    public static DmlMetric fit(IMatrix<Double> features, String[] labels, KdaDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    DmlMetric fitFromRows(double[][] x, int[] y) {
        int n = x.length;
        int d = x[0].length;

        int[] classes = distinctArray(y);
        int numClasses = classes.length;

        int nd = (nComponents != null)
            ? Math.min(nComponents, numClasses - 1)
            : numClasses - 1;

        // 核矩阵
        double[][] K = KernelDmlUtils.kernelMatrix(x, x, kernelType, gamma, degree, coef0);
        double[][] Kc = KernelDmlUtils.centerKernelMatrix(K);

        // M 和 N 矩阵
        double[] M_avg = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                M_avg[i] += Kc[i][j];
            }
            M_avg[i] /= n;
        }

        double[][] M = new double[n][n];
        double[][] N = new double[n][n];

        for (int c = 0; c < numClasses; c++) {
            int cls = classes[c];
            List<Integer> classIndices = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (y[i] == cls) classIndices.add(i);
            }
            int nc = classIndices.size();

            double[] M_i = new double[n];
            for (int i : classIndices) {
                for (int j = 0; j < n; j++) {
                    M_i[i] += Kc[i][j];
                }
                M_i[i] /= nc;
            }

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    M[i][j] += nc * (M_i[i] - M_avg[i]) * (M_i[j] - M_avg[j]);
                }
            }

            // N 矩阵
            for (int i : classIndices) {
                for (int j : classIndices) {
                    N[i][j] += 1.0 - 1.0 / nc;
                }
            }
        }

        IMatrix<Double> M_mat = IMatrix.of(M);
        IMatrix<Double> N_mat = IMatrix.of(N);

        // 始终添加正则化以确保数值稳定性
        N_mat = N_mat.add(Linalg.eye(n).multiplyByScalar(alpha));

        // 广义特征分解: N^-1 * M
        IMatrix<Double> N_inv = N_mat.inv();
        IMatrix<Double> A = N_inv.mmul(M_mat);

        Tuple2<IVector<Double>, IMatrix<Double>> eigenResult = A.eigen();
        IVector<Double> evals = eigenResult._1;
        IMatrix<Double> evecs = eigenResult._2;

        // 排序
        Integer[] indices = new Integer[evals.size()];
        for (int i = 0; i < indices.length; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> Double.compare((Double) evals.get(b), (Double) evals.get(a)));

        // 取前 nd 个特征向量，通过 pre-image 投影回原始特征空间
        double[][] L = new double[nd][d];
        for (int i = 0; i < nd && i < indices.length; i++) {
            int idx = indices[i];
            for (int j = 0; j < d; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += (Double) evecs.get(k, idx) * x[k][j];
                }
                L[i][j] = sum;
            }
        }

        return DmlMetric.lowRank(Linalg.matrix(L));
    }

    private static int[] distinctArray(int[] arr) {
        IntHashSet set = new IntHashSet();
        for (int v : arr) set.add(v);
        return set.toArray();
    }

    private static class IntHashSet {
        private int[] arr = new int[16];
        private int sz = 0;
        void add(int v) {
            for (int i = 0; i < sz; i++) if (arr[i] == v) return;
            if (sz == arr.length) {
                int[] newArr = new int[arr.length * 2];
                System.arraycopy(arr, 0, newArr, 0, sz);
                arr = newArr;
            }
            arr[sz++] = v;
        }
        int[] toArray() {
            int[] result = new int[sz];
            System.arraycopy(arr, 0, result, 0, sz);
            return result;
        }
    }
}
