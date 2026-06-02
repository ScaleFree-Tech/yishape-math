package com.yishape.lab.math.ml.dml.llda;

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
 * 核化局部 Fisher 判别分析 (KLLDA): LLDA 的核化版本。
 *
 * <p>本类实现 {@link ISupervisedDml}。</p>
 */
public final class KLldaDml implements ISupervisedDml {

    public enum AffinityType {
        NEIGHBORS,
        LOCAL_SCALING
    }

    private Integer nComponents = null;
    private AffinityType affinity = AffinityType.NEIGHBORS;
    private int nNeighbors = 7;
    private double tol = 1e-4;
    private double alpha = 1e-3;
    private KernelType kernelType = KernelType.RBF;
    private double gamma = 1.0;
    private int degree = 3;
    private double coef0 = 1.0;

    public KLldaDml setNComponents(Integer n) {
        this.nComponents = n;
        return this;
    }

    public KLldaDml setAffinity(AffinityType affinity) {
        this.affinity = affinity;
        return this;
    }

    public KLldaDml setNNeighbors(int n) {
        this.nNeighbors = n;
        return this;
    }

    public KLldaDml setKernelType(KernelType kernelType) {
        this.kernelType = kernelType;
        return this;
    }

    public KLldaDml setGamma(double gamma) {
        this.gamma = gamma;
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

    public static DmlMetric fit(IMatrix<Double> features, String[] labels, KLldaDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    DmlMetric fitFromRows(double[][] x, int[] y) {
        int n = x.length;
        int d = x[0].length;

        int nd = (nComponents != null) ? Math.min(nComponents, n - 1) : n - 1;

        // 核矩阵
        double[][] K = KernelDmlUtils.kernelMatrix(x, x, kernelType, gamma, degree, coef0);
        double[][] Kc = KernelDmlUtils.centerKernelMatrix(K);

        // 亲和矩阵
        double[][] A = buildAffinityMatrix(x, y, n);

        int[] classes = distinctArray(y);
        int numClasses = classes.length;

        int[] classCounts = new int[numClasses];
        for (int yi : y) {
            for (int c = 0; c < numClasses; c++) {
                if (yi == classes[c]) {
                    classCounts[c]++;
                    break;
                }
            }
        }

        // 核空间中的散点矩阵
        double[][] S_b = new double[n][n];
        double[][] S_w = new double[n][n];

        // 计算 S_b
        double[] KSum = new double[n];
        double totalK = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                KSum[i] += Kc[i][j];
                totalK += Kc[i][j];
            }
        }

        for (int c = 0; c < numClasses; c++) {
            int cls = classes[c];
            List<Integer> classIndices = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (y[i] == cls) classIndices.add(i);
            }
            int nc = classIndices.size();

            double[] KcSum = new double[n];
            for (int i : classIndices) {
                for (int j = 0; j < n; j++) {
                    KcSum[i] += Kc[i][j];
                }
            }

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    S_b[i][j] += nc * Math.pow(KSum[i] / n - KcSum[i] / nc, 2) / n;
                }
            }
        }

        // 计算 S_w
        for (int c = 0; c < numClasses; c++) {
            int cls = classes[c];
            List<Integer> classIndices = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (y[i] == cls) classIndices.add(i);
            }
            int nc = classIndices.size();

            for (int i : classIndices) {
                for (int j : classIndices) {
                    S_w[i][j] += A[i][j] / nc;
                }
            }
        }

        // 广义特征分解
        IMatrix<Double> S_b_mat = IMatrix.of(S_b);
        IMatrix<Double> S_w_mat = IMatrix.of(S_w);

        double det = S_w_mat.det();
        if (Math.abs(det) < tol) {
            for (int i = 0; i < n; i++) {
                S_w[i][i] += alpha;
            }
            S_w_mat = IMatrix.of(S_w);
        }

        IMatrix<Double> S_w_inv = S_w_mat.inv();
        IMatrix<Double> M = S_w_inv.mmul(S_b_mat);

        Tuple2<IVector<Double>, IMatrix<Double>> eigenResult = M.eigen();
        IVector<Double> evals = eigenResult._1;
        IMatrix<Double> evecs = eigenResult._2;

        // 排序
        Integer[] indices = new Integer[evals.size()];
        for (int i = 0; i < indices.length; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> Double.compare((Double) evals.get(b), (Double) evals.get(a)));

        // 注意: pyDML KLLDA 直接返回核空间特征向量 (nd × n)
        // Java 尝试 pre-image 投影回原始空间 (nd × d)，这是一个近似
        double[][] L = new double[nd][d];
        for (int i = 0; i < nd && i < indices.length; i++) {
            int idx = indices[i];
            for (int j = 0; j < n; j++) {
                // 修复: 累加核特征向量在原始特征空间的投影
                // 这是一个近似：L[i] ≈ Σ_j V[j,i] * x[j]
                double vj = (Double) evecs.get(j, idx);
                for (int k = 0; k < d; k++) {
                    L[i][k] += vj * x[j][k];
                }
            }
        }

        return DmlMetric.lowRank(Linalg.matrix(L));
    }

    private double[][] buildAffinityMatrix(double[][] x, int[] y, int n) {
        double[][] A = new double[n][n];

        if (affinity == AffinityType.NEIGHBORS) {
            for (int i = 0; i < n; i++) {
                List<double[]> neighbors = new ArrayList<>();
                for (int j = 0; j < n; j++) {
                    if (i == j) continue;
                    double dist = euclideanDist(x[i], x[j]);
                    neighbors.add(new double[]{j, dist});
                }
                neighbors.sort(Comparator.comparingDouble(a -> a[1]));
                for (int k = 0; k < Math.min(nNeighbors, neighbors.size()); k++) {
                    A[i][(int) neighbors.get(k)[0]] = 1.0;
                    A[(int) neighbors.get(k)[0]][i] = 1.0;
                }
            }
        }
        return A;
    }

    private static double euclideanDist(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
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
