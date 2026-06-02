package com.yishape.lab.math.ml.dml.cmoml;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.dml.DmlArrays;
import com.yishape.lab.math.ml.dml.DmlMetric;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import com.yishape.lab.math.ml.dml.ISupervisedDml;

/**
 * Class Mean Metric Learning (CMOML)：通过类方式均值学习距离度量。
 * 在嵌入空间中最大化类间距离与类内距离的比值。
 *
 * <p><strong>注意</strong>：此类与 pyDML 的 DMLMJ 算法<strong>完全不同</strong>。
 * pyDML DMLMJ 基于 Jeffrey 散度最大化，而本类使用类均值差异的特征值分解。
 * 类名保留是为了历史兼容性。</p>
 *
 * <p>本类实现 {@link ISupervisedDml}。</p>
 *
 * @apiNote 应用代码优先 {@link com.yishape.lab.math.ml.ML#dml}。
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>Wang, X., et al. (2012). Class-wise analysis of distance metric learning.
 *       <em>Pattern Recognition</em>.</li>
 * </ul>
 */
public final class CmomlDml implements ISupervisedDml {

    private Integer numDims;
    private double reg = 1e-6;

    public Integer getNumDims() {
        return numDims;
    }

    public CmomlDml setNumDims(Integer numDims) {
        this.numDims = numDims;
        return this;
    }

    public double getReg() {
        return reg;
    }

    public CmomlDml setReg(double reg) {
        this.reg = reg;
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

    public static DmlMetric fit(IMatrix<Double> features, String[] labels, CmomlDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    public static DmlMetric fit(IMatrix<Double> features, IVector<?> labels, CmomlDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    DmlMetric fitFromRows(double[][] x, int[] y) {
        int n = x.length;
        int d = x[0].length;

        int numClasses = 0;
        for (int yi : y) {
            if (yi + 1 > numClasses) {
                numClasses = yi + 1;
            }
        }

        double[][] centroids = computeClassCentroids(x, y, numClasses);
        double[][] S_b = computeBetweenClassScatter(x, y, centroids, numClasses);
        double[][] S_w = computeWithinClassScatter(x, y, centroids, numClasses);

        for (int i = 0; i < d; i++) {
            S_w[i][i] += reg;
        }

        IMatrix<Double> sWMat = IMatrix.of(S_w);
        IMatrix<Double> sBMat = IMatrix.of(S_b);

        IMatrix<Double> sWInv = sWMat.inv();
        IMatrix<Double> generalized = sWInv.mmul(sBMat);

        var eigResult = generalized.eigen();
        IVector<Double> eigVals = eigResult._1;
        IMatrix<Double> eigVecs = eigResult._2;

        int numDimsFinal = (numDims != null) ? Math.min(numDims, d) : d;

        Integer[] indices = new Integer[d];
        for (int i = 0; i < d; i++) {
            indices[i] = i;
        }
        java.util.Arrays.sort(indices, (i, j) -> Double.compare(eigVals.get(j), eigVals.get(i)));

        double[][] L = new double[numDimsFinal][d];
        for (int i = 0; i < numDimsFinal; i++) {
            int origIdx = indices[i];
            for (int j = 0; j < d; j++) {
                L[i][j] = eigVecs.get(j, origIdx);
            }
        }

        return DmlMetric.lowRank(Linalg.matrix(L));
    }

    private double[][] computeClassCentroids(double[][] x, int[] y, int numClasses) {
        int n = x.length;
        int d = x[0].length;

        double[][] centroids = new double[numClasses][d];
        int[] counts = new int[numClasses];

        for (int i = 0; i < n; i++) {
            int c = y[i];
            counts[c]++;
            for (int j = 0; j < d; j++) {
                centroids[c][j] += x[i][j];
            }
        }

        for (int c = 0; c < numClasses; c++) {
            if (counts[c] > 0) {
                for (int j = 0; j < d; j++) {
                    centroids[c][j] /= counts[c];
                }
            }
        }

        return centroids;
    }

    private double[][] computeBetweenClassScatter(double[][] x, int[] y, double[][] centroids, int numClasses) {
        int n = x.length;
        int d = x[0].length;

        double[] globalMean = new double[d];
        int totalCount = 0;
        for (int c = 0; c < numClasses; c++) {
            List<Integer> classIndices = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (y[i] == c) {
                    classIndices.add(i);
                }
            }
            for (int idx : classIndices) {
                totalCount++;
                for (int j = 0; j < d; j++) {
                    globalMean[j] += x[idx][j];
                }
            }
        }

        if (totalCount > 0) {
            for (int j = 0; j < d; j++) {
                globalMean[j] /= totalCount;
            }
        }

        double[][] S_b = new double[d][d];

        for (int c = 0; c < numClasses; c++) {
            List<Integer> classIndices = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (y[i] == c) {
                    classIndices.add(i);
                }
            }
            int count = classIndices.size();
            if (count == 0) {
                continue;
            }

            double[] meanDiff = new double[d];
            for (int j = 0; j < d; j++) {
                meanDiff[j] = centroids[c][j] - globalMean[j];
            }

            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    S_b[a][b] += count * meanDiff[a] * meanDiff[b];
                }
            }
        }

        return S_b;
    }

    private double[][] computeWithinClassScatter(double[][] x, int[] y, double[][] centroids, int numClasses) {
        int n = x.length;
        int d = x[0].length;

        double[][] S_w = new double[d][d];

        for (int c = 0; c < numClasses; c++) {
            List<Integer> classIndices = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (y[i] == c) {
                    classIndices.add(i);
                }
            }

            for (int idx : classIndices) {
                for (int a = 0; a < d; a++) {
                    double diffA = x[idx][a] - centroids[c][a];
                    for (int b = 0; b < d; b++) {
                        double diffB = x[idx][b] - centroids[c][b];
                        S_w[a][b] += diffA * diffB;
                    }
                }
            }
        }

        return S_w;
    }
}
