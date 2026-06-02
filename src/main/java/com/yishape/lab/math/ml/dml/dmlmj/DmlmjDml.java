package com.yishape.lab.math.ml.dml.dmlmj;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.dml.DmlArrays;
import com.yishape.lab.math.ml.dml.DmlMetric;

import java.util.Objects;
import com.yishape.lab.math.ml.dml.ISupervisedDml;

/**
 * DMLMJ：通过最大化 Jeffrey 散度学习距离度量。
 * 最大化同类近邻差分分布与异类近邻差分分布之间的 Jeffrey 散度。
 *
 * <p>本类实现 {@link ISupervisedDml}。</p>
 *
 * @apiNote 应用代码优先 {@link com.yishape.lab.math.ml.ML#dml}。
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>Nguyen, B., Morell, C., &amp; De Baets, B. (2017). Supervised distance metric learning through
 *       maximization of the Jeffrey divergence. <em>Pattern Recognition</em>, 64, 215–225.</li>
 * </ul>
 */
public final class DmlmjDml implements ISupervisedDml {

    private Integer numDims;
    private int nNeighbors = 3;
    private double alpha = 0.001;
    private double regTol = 1e-10;

    public Integer getNumDims() {
        return numDims;
    }

    public DmlmjDml setNumDims(Integer numDims) {
        this.numDims = numDims;
        return this;
    }

    public int getNNeighbors() {
        return nNeighbors;
    }

    public DmlmjDml setNNeighbors(int nNeighbors) {
        this.nNeighbors = nNeighbors;
        return this;
    }

    public double getAlpha() {
        return alpha;
    }

    public DmlmjDml setAlpha(double alpha) {
        this.alpha = alpha;
        return this;
    }

    public double getRegTol() {
        return regTol;
    }

    public DmlmjDml setRegTol(double regTol) {
        this.regTol = regTol;
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

    public static DmlMetric fit(IMatrix<Double> features, String[] labels, DmlmjDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    public static DmlMetric fit(IMatrix<Double> features, IVector<?> labels, DmlmjDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    DmlMetric fitFromRows(double[][] x, int[] y) {
        int n = x.length;
        int d = x[0].length;

        int[][] hetNeighs = new int[n][nNeighbors];
        int[][] homNeighs = new int[n][nNeighbors];

        double[][] distMatrix = computeDistanceMatrix(x);

        for (int i = 0; i < n; i++) {
            int curClass = y[i];

            Integer[] heteroIndices = new Integer[n];
            int heteroCount = 0;
            for (int j = 0; j < n; j++) {
                if (y[j] != curClass) {
                    heteroIndices[heteroCount++] = j;
                }
            }

            Double[] heteroDists = new Double[heteroCount];
            for (int j = 0; j < heteroCount; j++) {
                heteroDists[j] = distMatrix[i][heteroIndices[j]];
            }

            Integer[] heteroOrder = new Integer[heteroCount];
            for (int j = 0; j < heteroCount; j++) {
                heteroOrder[j] = j;
            }
            java.util.Arrays.sort(heteroOrder, (a, b) -> Double.compare(heteroDists[a], heteroDists[b]));

            for (int j = 0; j < Math.min(nNeighbors, heteroCount); j++) {
                hetNeighs[i][j] = heteroIndices[heteroOrder[j]];
            }

            Integer[] homoIndices = new Integer[n];
            int homoCount = 0;
            for (int j = 0; j < n; j++) {
                if (j != i && y[j] == curClass) {
                    homoIndices[homoCount++] = j;
                }
            }

            // 收集同类样本的距离和索引，并按距离升序排序
            java.util.List<java.util.Map.Entry<Double, Integer>> homoList =
                new java.util.ArrayList<>();
            for (int j = 0; j < homoCount; j++) {
                homoList.add(new java.util.AbstractMap.SimpleEntry<>(
                    distMatrix[i][homoIndices[j]], homoIndices[j]));
            }
            homoList.sort(java.util.Comparator.comparingDouble(java.util.Map.Entry::getKey));

            for (int j = 0; j < Math.min(nNeighbors, homoCount); j++) {
                homNeighs[i][j] = homoList.get(j).getValue();
            }
        }

        double[][] S = new double[d][d];
        double[][] D = new double[d][d];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nNeighbors; j++) {
                int homJ = homNeighs[i][j];
                for (int a = 0; a < d; a++) {
                    for (int b = 0; b < d; b++) {
                        S[a][b] += (x[i][a] - x[homJ][a]) * (x[i][b] - x[homJ][b]);
                    }
                }

                int hetJ = hetNeighs[i][j];
                for (int a = 0; a < d; a++) {
                    for (int b = 0; b < d; b++) {
                        D[a][b] += (x[i][a] - x[hetJ][a]) * (x[i][b] - x[hetJ][b]);
                    }
                }
            }
        }

        int dSize = n * nNeighbors;
        for (int a = 0; a < d; a++) {
            for (int b = 0; b < d; b++) {
                S[a][b] /= dSize;
                D[a][b] /= dSize;
            }
        }

        IMatrix<Double> sMat = IMatrix.of(S);
        IMatrix<Double> dMat = IMatrix.of(D);

        double detS = sMat.det();
        if (Math.abs(detS) < regTol) {
            sMat = sMat.multiplyByScalar(1 - alpha).add(Linalg.diag(new double[d]).multiplyByScalar(alpha));
        }

        double detD = dMat.det();
        if (Math.abs(detD) < regTol) {
            dMat = dMat.multiplyByScalar(1 - alpha).add(Linalg.diag(new double[d]).multiplyByScalar(alpha));
        }

        IMatrix<Double> sInv = sMat.inv();
        IMatrix<Double> generalized = sInv.mmul(dMat);

        var eigResult = generalized.eigen();
        IVector<Double> eigVals = eigResult._1;
        IMatrix<Double> eigVecs = eigResult._2;

        int numDimsFinal = (numDims != null) ? Math.min(numDims, d) : d;

        Integer[] indices = new Integer[d];
        for (int i = 0; i < d; i++) {
            indices[i] = i;
        }
        java.util.Arrays.sort(indices, (i, j) -> {
            double scoreI = eigVals.get(i) + 1.0 / Math.max(eigVals.get(i), 1e-10);
            double scoreJ = eigVals.get(j) + 1.0 / Math.max(eigVals.get(j), 1e-10);
            return Double.compare(scoreJ, scoreI);
        });

        double[][] L = new double[numDimsFinal][d];
        for (int i = 0; i < numDimsFinal; i++) {
            int origIdx = indices[i];
            for (int j = 0; j < d; j++) {
                L[i][j] = eigVecs.get(j, origIdx);
            }
        }

        return DmlMetric.lowRank(Linalg.matrix(L));
    }

    private double[][] computeDistanceMatrix(double[][] x) {
        int n = x.length;
        double[][] dist = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < x[i].length; k++) {
                    double diff = x[i][k] - x[j][k];
                    sum += diff * diff;
                }
                dist[i][j] = sum;
                dist[j][i] = sum;
            }
        }
        return dist;
    }
}
