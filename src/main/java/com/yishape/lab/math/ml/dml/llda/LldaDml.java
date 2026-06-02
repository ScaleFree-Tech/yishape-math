package com.yishape.lab.math.ml.dml.llda;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.dml.DmlArrays;
import com.yishape.lab.math.ml.dml.DmlMetric;
import com.yishape.lab.util.Tuple2;

import java.util.*;
import com.yishape.lab.math.ml.dml.ISupervisedDml;

/**
 * Local Linear Discriminant Analysis (LLDA): 局部 Fisher 判别分析。
 * 结合 LDA 与局部保持投影，在保持类间判别信息的同时保留数据的局部结构。
 *
 * <p>本类实现 {@link ISupervisedDml}。</p>
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>Masashi Sugiyama, "Dimensionality reduction of multimodal labeled data by local fisher
 *       discriminant analysis". <em>JMLR</em>, 2007.</li>
 * </ul>
 */
public final class LldaDml implements ISupervisedDml {

    public enum AffinityType {
        NEIGHBORS,
        LOCAL_SCALING
    }

    public enum SolverType {
        CLASSIC,
        SUGIYAMA
    }

    private Integer nComponents = null;
    private AffinityType affinity = AffinityType.NEIGHBORS;
    private int nNeighbors = 1;
    private double tol = 1e-4;
    private double alpha = 1e-3;
    private SolverType solver = SolverType.SUGIYAMA;

    public LldaDml setNComponents(Integer nComponents) {
        this.nComponents = nComponents;
        return this;
    }

    public LldaDml setAffinity(AffinityType affinity) {
        this.affinity = affinity;
        return this;
    }

    public LldaDml setNNeighbors(int n) {
        this.nNeighbors = n;
        return this;
    }

    public LldaDml setSolver(SolverType solver) {
        this.solver = solver;
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

    public static DmlMetric fit(IMatrix<Double> features, String[] labels, LldaDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    DmlMetric fitFromRows(double[][] x, int[] y) {
        int n = x.length;
        int d = x[0].length;

        int nd = (nComponents != null) ? Math.min(nComponents, d) : d;

        // 构建亲和矩阵
        double[][] A = buildAffinityMatrix(x, y, n);

        // 计算类内/类间散点矩阵
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

        double[][] S_b = new double[d][d];
        double[][] S_w = new double[d][d];

        if (solver == SolverType.SUGIYAMA) {
            // SUGIYAMA 求解器
            for (int c = 0; c < numClasses; c++) {
                int cls = classes[c];
                List<double[]> classPoints = new ArrayList<>();
                List<Integer> classIndices = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    if (y[i] == cls) {
                        classPoints.add(x[i]);
                        classIndices.add(i);
                    }
                }
                double[][] Xc = classPoints.toArray(new double[0][]);
                int nc = Xc.length;

                // G = Xc^T D Xc - Xc^T A Xc
                double[] AcSum = new double[n];
                for (int idx : classIndices) {
                    for (int j = 0; j < n; j++) {
                        AcSum[idx] += A[idx][j];
                    }
                }

                for (int a = 0; a < d; a++) {
                    for (int b = 0; b < d; b++) {
                        double G_ab = 0;
                        for (int p = 0; p < nc; p++) {
                            int i = classIndices.get(p);
                            for (int q = 0; q < nc; q++) {
                                int j = classIndices.get(q);
                                double lap = (p == q ? AcSum[i] : 0.0) - A[i][j];
                                G_ab += Xc[p][a] * lap * Xc[q][b];
                            }
                        }
                        S_w[a][b] += G_ab / nc;
                    }
                }
            }

            // 全局 S_b
            double[] xBar = new double[d];
            for (double[] xi : x) {
                for (int j = 0; j < d; j++) {
                    xBar[j] += xi[j];
                }
            }
            for (int j = 0; j < d; j++) xBar[j] /= n;

            for (int c = 0; c < numClasses; c++) {
                int cls = classes[c];
                List<double[]> classPoints = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    if (y[i] == cls) {
                        classPoints.add(x[i]);
                    }
                }
                double[][] Xc = classPoints.toArray(new double[0][]);
                int nc = Xc.length;

                double[] xcBar = new double[d];
                for (double[] xi : Xc) {
                    for (int j = 0; j < d; j++) {
                        xcBar[j] += xi[j];
                    }
                }
                for (int j = 0; j < d; j++) xcBar[j] /= nc;

                for (int a = 0; a < d; a++) {
                    for (int b = 0; b < d; b++) {
                        S_b[a][b] += (xcBar[a] - xBar[a]) * (xcBar[b] - xBar[b]) * nc / n;
                    }
                }
            }
        } else {
            // CLASSIC 求解器
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double[] xij = new double[d];
                    for (int k = 0; k < d; k++) {
                        xij[k] = x[i][k] - x[j][k];
                    }

                    int ci = -1, cj = -1;
                    for (int c = 0; c < numClasses; c++) {
                        if (y[i] == classes[c]) ci = c;
                        if (y[j] == classes[c]) cj = c;
                    }

                    double A_w_ij, A_b_ij;
                    if (ci == cj) {
                        A_w_ij = A[i][j] / classCounts[ci];
                        A_b_ij = A[i][j] * (1.0 / n - 1.0 / classCounts[ci]);
                    } else {
                        A_w_ij = 0;
                        A_b_ij = 1.0 / n;
                    }

                    for (int a = 0; a < d; a++) {
                        for (int b = 0; b < d; b++) {
                            S_w[a][b] += A_w_ij * xij[a] * xij[b];
                            S_b[a][b] += A_b_ij * xij[a] * xij[b];
                        }
                    }
                }
            }
        }

        // 正则化
        double detSw = determinant(IMatrix.of(S_w).toDoubleArray());
        if (Math.abs(detSw) < tol) {
            for (int i = 0; i < d; i++) {
                S_w[i][i] += alpha;
            }
        }

        // 广义特征分解: S_b * v = λ * S_w * v
        IMatrix<Double> S_b_mat = IMatrix.of(S_b);
        IMatrix<Double> S_w_mat = IMatrix.of(S_w);
        IMatrix<Double> S_w_inv = S_w_mat.inv();
        IMatrix<Double> M = S_w_inv.mmul(S_b_mat);

        Tuple2<IVector<Double>, IMatrix<Double>> eigenResult = M.eigen();
        IVector<Double> evals = eigenResult._1;
        IMatrix<Double> evecs = eigenResult._2;

        // 取最大特征值对应的特征向量
        Integer[] indices = new Integer[evals.size()];
        for (int i = 0; i < indices.length; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> Double.compare((Double) evals.get(b), (Double) evals.get(a)));

        double[][] L = new double[nd][d];
        for (int i = 0; i < nd && i < indices.length; i++) {
            int idx = indices[i];
            for (int j = 0; j < d; j++) {
                L[i][j] = (Double) evecs.get(j, idx);
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
        } else if (affinity == AffinityType.LOCAL_SCALING) {
            for (int i = 0; i < n; i++) {
                List<double[]> dists = new ArrayList<>();
                for (int j = 0; j < n; j++) {
                    if (i == j) continue;
                    double dist = euclideanDist(x[i], x[j]);
                    dists.add(new double[]{j, dist});
                }
                dists.sort(Comparator.comparingDouble(a -> a[1]));
                double sigma = (nNeighbors < dists.size())
                    ? dists.get(nNeighbors)[1]
                    : dists.get(dists.size() - 1)[1];

                for (double[] dj : dists) {
                    int j = (int) dj[0];
                    A[i][j] = Math.exp(-dj[1] * dj[1] / (2 * sigma * sigma));
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

    private static double determinant(double[][] a) {
        int n = a.length;
        if (n == 1) return a[0][0];
        double det = 0;
        for (int j = 0; j < n; j++) {
            double[][] sub = new double[n - 1][n - 1];
            for (int i = 1; i < n; i++) {
                for (int k = 0; k < n; k++) {
                    if (k < j) sub[i - 1][k] = a[i][k];
                    else if (k > j) sub[i - 1][k - 1] = a[i][k];
                }
            }
            det += (j % 2 == 0 ? 1 : -1) * a[0][j] * determinant(sub);
        }
        return det;
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
