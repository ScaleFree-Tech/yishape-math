package com.yishape.lab.math.ml.dml.withinclass;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.dml.DmlArrays;
import com.yishape.lab.math.ml.dml.DmlMetric;

import java.util.Objects;
import com.yishape.lab.math.ml.dml.ISupervisedDml;

/**
 * 类内方差的<strong>对角</strong>近似：按维估计类内方差，缩放取 {@code 1/√(方差+λ)}，
 * 等价 Mahalanobis 仅保留对角精度。{@link DmlMetric#form()} 为
 * {@link com.yishape.lab.math.ml.dml.MetricForm#DIAGONAL}。
 *
 * <p>本类实现 {@link ISupervisedDml}；{@link #setL2Weight(double)} 须为正。</p>
 *
 * @apiNote 应用优先 {@link com.yishape.lab.math.ml.ML#dml}；需额外调参时再 {@code new} 本类。
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>Friedman, J. H. (1989). Regularized discriminant analysis.
 *       <em>Journal of the American Statistical Association</em>, 84(405), 165–175.</li>
 *   <li>Bickel, P. J., &amp; Levina, E. (2004). Some theory for Fisher's linear discriminant analysis,
 *       'naive Bayes', … <em>Bernoulli</em>, 10(6), 989–1010.</li>
 * </ul>
 */
public final class WithinClassDml implements ISupervisedDml {

    private double l2Weight = 1.0;

    /** 对角岭强度 {@code λ}（加在每维合并类内方差上），须为正。 */
    public double getL2Weight() {
        return l2Weight;
    }

    public WithinClassDml setL2Weight(double l2Weight) {
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

    /**
     * 便捷静态入口：{@code l2Weight} 为对角岭强度 {@code λ}。
     */
    public static DmlMetric fit(IMatrix<Double> features, String[] labels, double l2Weight) {
        return new WithinClassDml().setL2Weight(l2Weight).fit(features, labels);
    }

    /**
     * 标签为向量时的静态 {@link #fit(IMatrix, String[])} 包装。
     */
    public static DmlMetric fit(IMatrix<Double> features, IVector<?> labels, double l2Weight) {
        return new WithinClassDml().setL2Weight(l2Weight).fit(features, labels);
    }

    /**
     * 由已物化样本行与类别索引拟合；供测试或内部复用。
     *
     * @param x 行样本，每行维数一致
     * @param y 与行对齐的非负类别索引
     * @param l2Weight 加到每维合并类内方差上的岭 {@code λ}，须为正
     */
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

        double[] var = new double[d];
        for (int i = 0; i < n; i++) {
            int c = y[i];
            for (int j = 0; j < d; j++) {
                double t = x[i][j] - mu[c][j];
                var[j] += t * t;
            }
        }

        double[] scaling = new double[d];
        for (int j = 0; j < d; j++) {
            scaling[j] = 1.0 / Math.sqrt(var[j] + l2Weight);
        }

        IVector<Double> diag = Linalg.vector(scaling);
        return DmlMetric.diagonal(diag, 0, 0);
    }
}
