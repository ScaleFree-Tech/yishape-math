package com.yishape.lab.math.ml.dml.rca;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.ml.dml.DmlArrays;
import com.yishape.lab.math.ml.dml.DmlMetric;
import com.yishape.lab.math.ml.dml.MetricTransforms;

import java.util.Objects;
import com.yishape.lab.math.ml.dml.ISupervisedDml;

/**
 * Relevant Component Analysis（RCA）：将每个类别视为一个 chunklet，在去类均值后估计 chunk 内协方差，
 * 取 {@code (Cov + λ I)⁻¹} 型精度并白化，得到 {@link DmlMetric}。
 *
 * <p>本类实现 {@link ISupervisedDml}；chunk 内协方差对角的 L2 岭 {@link #setL2Weight(double)} 须为正。</p>
 *
 * @apiNote 应用代码优先 {@link com.yishape.lab.math.ml.ML#dml}；需未在
 * {@link com.yishape.lab.math.ml.DmlWrapper} 暴露的调参时再直接使用本类。
 *
 * <p>刻画「块内」二阶结构；与 Fisher 类方法相比更依赖 chunk 语义标签。</p>
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>Bar-Hillel, A., Hertz, T., Shental, N., &amp; Weinshall, D. (2005). Learning a Mahalanobis
 *       metric from equivalence constraints. <em>Journal of Machine Learning Research</em>, 6, 937–965.</li>
 * </ul>
 */
public final class RcaDml implements ISupervisedDml {

    private double l2Weight = 1e-2;

    public double getL2Weight() {
        return l2Weight;
    }

    public RcaDml setL2Weight(double l2Weight) {
        this.l2Weight = l2Weight;
        return this;
    }

    @Override
    public DmlMetric fit(IMatrix<Double> features, IVector<?> labels) {
        Objects.requireNonNull(labels, "labels");
        return fit(features, DmlArrays.stringLabels(labels));
    }

    @Override
    public DmlMetric fit(IMatrix<Double> features, String[] labels) {
        Objects.requireNonNull(labels, "labels");
        double[][] x = DmlArrays.featureRows(features);
        int[] y = DmlArrays.classIndices(labels);
        return fitFromRows(x, y, l2Weight);
    }

    public static DmlMetric fit(IMatrix<Double> features, String[] labels, double l2Weight) {
        return new RcaDml().setL2Weight(l2Weight).fit(features, labels);
    }

    public static DmlMetric fit(IMatrix<Double> features, IVector<?> labels, double l2Weight) {
        return new RcaDml().setL2Weight(l2Weight).fit(features, labels);
    }

    static DmlMetric fitFromRows(double[][] x, int[] y, double l2Weight) {
        if (l2Weight <= 0.0) {
            throw new IllegalArgumentException("l2Weight 须为正");
        }
        if (x == null || x.length == 0 || y.length != x.length) {
            throw new IllegalArgumentException("样本与标签行数须一致且非空");
        }
        int n = x.length;
        int d = x[0].length;
        for (int i = 1; i < n; i++) {
            if (x[i].length != d) {
                throw new IllegalArgumentException("各特征行维数须一致");
            }
        }

        int numClasses = 0;
        for (int yi : y) {
            if (yi < 0) {
                throw new IllegalArgumentException("类别索引非负");
            }
            if (yi + 1 > numClasses) {
                numClasses = yi + 1;
            }
        }

        int[] cnt = new int[numClasses];
        double[][] mu = new double[numClasses][d];
        for (int i = 0; i < n; i++) {
            int c = y[i];
            cnt[c]++;
            for (int j = 0; j < d; j++) {
                mu[c][j] += x[i][j];
            }
        }
        for (int c = 0; c < numClasses; c++) {
            if (cnt[c] == 0) {
                continue;
            }
            for (int j = 0; j < d; j++) {
                mu[c][j] /= cnt[c];
            }
        }

        double[][] cov = new double[d][d];
        for (int i = 0; i < n; i++) {
            int c = y[i];
            for (int a = 0; a < d; a++) {
                double za = x[i][a] - mu[c][a];
                for (int b = 0; b < d; b++) {
                    double zb = x[i][b] - mu[c][b];
                    cov[a][b] += za * zb;
                }
            }
        }
        double scale = n > 1 ? 1.0 / (n - 1) : 1.0;
        for (int a = 0; a < d; a++) {
            for (int b = 0; b < d; b++) {
                cov[a][b] *= scale;
            }
        }
        for (int i = 0; i < d; i++) {
            cov[i][i] += l2Weight;
        }

        IMatrix<Double> covM = IMatrix.of(cov);
        IMatrix<Double> precision = covM.inv();
        precision = MetricTransforms.symmetrize(precision);
        IMatrix<Double> w = MetricTransforms.whitenerFromPrecision(precision);
        return DmlMetric.fullWhitening(w);
    }
}
