package com.yishape.lab.math.ml.dml.fisher;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.ml.dml.DmlArrays;
import com.yishape.lab.math.ml.dml.DmlMetric;
import com.yishape.lab.math.ml.dml.MetricTransforms;

import java.util.Objects;
import com.yishape.lab.math.ml.dml.ISupervisedDml;

/**
 * Fisher / LDA 视角下的监督马氏度量：用合并<strong>类内散度</strong> {@code S_w} 估计度量形状，
 * 取精度 {@code (S_w + λ I)⁻¹}，对称化后经 {@link MetricTransforms#whitenerFromPrecision}
 * 得到白化矩阵，封装为 {@link DmlMetric#fullWhitening}。
 *
 * <p>本类实现 {@link ISupervisedDml}；L2 岭强度 {@link #setL2Weight(double)}（加在 {@code S_w} 对角）须为正。</p>
 *
 * @apiNote 应用代码优先 {@link com.yishape.lab.math.ml.ML#dml}；需未在
 * {@link com.yishape.lab.math.ml.DmlWrapper} 暴露的调参时再直接使用本类。
 *
 * <p>强调「类内协方差逆」而非类间项；岭参数 {@code λ &gt; 0} 保证数值正定。适合作为强基线与可解释预处理。</p>
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems.
 *       <em>Annals of Eugenics</em>, 7(2), 179–188.</li>
 *   <li>Rao, C. R. (1948). The utilization of multiple measurements in problems of biological
 *       classification. <em>Journal of the Royal Statistical Society. Series B</em>, 10(2), 159–193.</li>
 * </ul>
 */
public final class FisherDml implements ISupervisedDml {

    /** 加至合并类内散度对角上的 L2 岭系数，须为正（与全库其它模块中的 {@code l2Weight} / Ridge 口径一致）。 */
    private double l2Weight = 1e-3;

    public double getL2Weight() {
        return l2Weight;
    }

    public FisherDml setL2Weight(double l2Weight) {
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
        return new FisherDml().setL2Weight(l2Weight).fit(features, labels);
    }

    public static DmlMetric fit(IMatrix<Double> features, IVector<?> labels, double l2Weight) {
        return new FisherDml().setL2Weight(l2Weight).fit(features, labels);
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

        double[][] sw = new double[d][d];
        for (int i = 0; i < n; i++) {
            int c = y[i];
            for (int a = 0; a < d; a++) {
                double da = x[i][a] - mu[c][a];
                for (int b = 0; b < d; b++) {
                    sw[a][b] += da * (x[i][b] - mu[c][b]);
                }
            }
        }
        for (int i = 0; i < d; i++) {
            sw[i][i] += l2Weight;
        }

        IMatrix<Double> swm = IMatrix.of(sw);
        IMatrix<Double> precision = swm.inv();
        precision = MetricTransforms.symmetrize(precision);
        IMatrix<Double> w = MetricTransforms.whitenerFromPrecision(precision);
        return DmlMetric.fullWhitening(w);
    }
}
