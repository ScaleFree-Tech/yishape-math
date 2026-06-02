package com.yishape.lab.math.ml.dml.lsi;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.dml.DmlArrays;
import com.yishape.lab.math.ml.dml.DmlMetric;
import com.yishape.lab.math.ml.dml.IUnsupervisedDml;
import com.yishape.lab.util.Tuple2;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Locality Sensitive Indexing (LSI)：局部敏感索引/保持投影。
 *
 * <p><strong>命名说明</strong>：此类实现的是局部保持投影 (LPP)，
 * 与 pyDML 的 LSI（实为 MMC/Mahalanobis Metric for Clustering）<strong>完全不同</strong>。
 * pyDML 的 LSI 是监督式算法，而本类是无监督降维方法。</p>
 *
 * <p>LSI 是一种保持局部结构的度量/投影学习方法，通过保持数据的邻域结构来学习投影矩阵。
 * 类似于局部保持投影 (LPP)，通过热核相似度构建邻域图。</p>
 *
 * <p>本类实现 {@link IUnsupervisedDml}（无需标签，fit 仅消费特征）。</p>
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>He, X., et al. "Locality preserving indexing (LPI)". <em>CIKM</em>, 2005.</li>
 * </ul>
 */
public final class LsiDml implements IUnsupervisedDml {

    private Integer nComponents = null;
    private int nNeighbors = 5;
    private double sigma = 1.0;
    private boolean autoSigma = true;

    public LsiDml setNComponents(Integer n) {
        this.nComponents = n;
        return this;
    }

    public LsiDml setNNeighbors(int k) {
        this.nNeighbors = k;
        return this;
    }

    public LsiDml setSigma(double sigma) {
        this.sigma = sigma;
        this.autoSigma = false;
        return this;
    }

    @Override
    public DmlMetric fit(IMatrix<Double> features) {
        double[][] x = DmlArrays.featureRows(features);
        return fitFromRows(x);
    }

    /**
     * @deprecated 仅保留签名兼容，请使用 {@link #fit(IMatrix)}。
     *             标签参数被忽略（LSI 为无监督算法）。
     */
    @Deprecated
    public DmlMetric fit(IMatrix<Double> features, IVector<?> labels) {
        return fit(features);
    }

    /**
     * @deprecated 仅保留签名兼容，请使用 {@link #fit(IMatrix)}。
     *             标签参数被忽略（LSI 为无监督算法）。
     */
    @Deprecated
    public DmlMetric fit(IMatrix<Double> features, String[] labels) {
        return fit(features);
    }

    public static DmlMetric fit(IMatrix<Double> features, LsiDml hyper) {
        return Objects.requireNonNull(hyper).fit(features);
    }

    DmlMetric fitFromRows(double[][] x) {
        int n = x.length;
        int d = x[0].length;

        int nd = (nComponents != null) ? Math.min(nComponents, d - 1) : d - 1;

        // 构建邻域图并计算相似度矩阵
        double[][] W = buildSimilarityMatrix(x, n);

        // 自动设置 sigma 如果需要
        if (autoSigma) {
            sigma = estimateSigma(x, n);
        }

        // 计算拉普拉斯矩阵 L = D - W
        double[][] D = new double[n][n];
        for (int i = 0; i < n; i++) {
            double rowSum = 0;
            for (int j = 0; j < n; j++) {
                rowSum += W[i][j];
            }
            D[i][i] = rowSum;
        }

        double[][] L = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                L[i][j] = D[i][j] - W[i][j];
            }
        }

        // 计算 X^T * L * X 和 X^T * D * X
        IMatrix<Double> X_mat = IMatrix.of(x);

        IMatrix<Double> XtLX = X_mat.transpose().mmul(IMatrix.of(L)).mmul(X_mat);
        IMatrix<Double> XtDX = X_mat.transpose().mmul(IMatrix.of(D)).mmul(X_mat);

        // 正则化
        double trace;
        try {
            trace = XtDX.trace();
        } catch (Exception e) {
            trace = 1.0;
        }
        if (Math.abs(trace) < 1e-10) {
            XtDX = XtDX.add(Linalg.eye(d).multiplyByScalar(1e-6));
        }

        // 广义特征分解: XtDX^-1 * XtLX
        IMatrix<Double> XtDX_inv;
        try {
            XtDX_inv = XtDX.inv();
        } catch (Exception e) {
            XtDX_inv = XtDX.pinv();
        }

        IMatrix<Double> M = XtDX_inv.mmul(XtLX);

        Tuple2<IVector<Double>, IMatrix<Double>> eigenResult = M.eigen();
        IVector<Double> evals = eigenResult._1;
        IMatrix<Double> evecs = eigenResult._2;

        // 升序排列（小特征值在前，保留局部结构）
        Integer[] indices = new Integer[evals.size()];
        for (int i = 0; i < indices.length; i++) indices[i] = i;
        Arrays.sort(indices, Comparator.comparingDouble(a -> (Double) evals.get(a)));

        // 取前 nd 个特征向量
        double[][] L_proj = new double[nd][d];
        for (int i = 0; i < nd; i++) {
            int idx = indices[i];
            for (int j = 0; j < d; j++) {
                L_proj[i][j] = (Double) evecs.get(j, idx);
            }
        }

        return DmlMetric.lowRank(Linalg.matrix(L_proj));
    }

    private double[][] buildSimilarityMatrix(double[][] x, int n) {
        double[][] W = new double[n][n];

        for (int i = 0; i < n; i++) {
            // 找到 k 个最近邻（按欧氏距离）
            List<double[]> neighbors = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                double dist = euclideanDist(x[i], x[j]);
                neighbors.add(new double[]{j, dist});
            }
            neighbors.sort(Comparator.comparingDouble(a -> a[1]));

            int k = Math.min(nNeighbors, neighbors.size());

            // 自动 sigma：取第 k 个近邻的距离作为 sigma
            if (autoSigma && k > 0) {
                sigma = neighbors.get(k - 1)[1];
                if (sigma < 1e-10) sigma = 1.0;
            }

            // 计算热核权重
            for (int t = 0; t < k; t++) {
                int j = (int) neighbors.get(t)[0];
                double dist = neighbors.get(t)[1];
                double w = Math.exp(-dist * dist / (2 * sigma * sigma));
                W[i][j] = w;
                W[j][i] = w;
            }
        }

        return W;
    }

    private double estimateSigma(double[][] x, int n) {
        // 使用所有点对距离的中位数估计 sigma
        List<Double> distances = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                distances.add(euclideanDist(x[i], x[j]));
            }
        }
        Collections.sort(distances);
        int midIdx = distances.size() / 2;
        return distances.get(midIdx) + 1e-10;
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
