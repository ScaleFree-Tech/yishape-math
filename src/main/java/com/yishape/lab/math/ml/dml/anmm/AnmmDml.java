package com.yishape.lab.math.ml.dml.anmm;

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
 * Average Neighborhood Margin Maximization (ANMM)：通过最大化最近同类近邻与异类近邻之间的间隔来学习度量。
 *
 * <p>本类实现 {@link ISupervisedDml}。</p>
 *
 * @apiNote 应用代码优先 {@link com.yishape.lab.math.ml.ML#dml}。
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>Wang, F., &amp; Zhang, C. (2007). Feature extraction by maximizing the average neighborhood margin.
 *       <em>CVPR</em>.</li>
 * </ul>
 */
public final class AnmmDml implements ISupervisedDml {

    private Integer numDims;
    private int nFriends = 3;
    private int nEnemies = 1;

    public Integer getNumDims() {
        return numDims;
    }

    public AnmmDml setNumDims(Integer numDims) {
        this.numDims = numDims;
        return this;
    }

    public int getNFriends() {
        return nFriends;
    }

    public AnmmDml setNFriends(int nFriends) {
        this.nFriends = nFriends;
        return this;
    }

    public int getNEnemies() {
        return nEnemies;
    }

    public AnmmDml setNEnemies(int nEnemies) {
        this.nEnemies = nEnemies;
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

    public static DmlMetric fit(IMatrix<Double> features, String[] labels, AnmmDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    public static DmlMetric fit(IMatrix<Double> features, IVector<?> labels, AnmmDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    DmlMetric fitFromRows(double[][] x, int[] y) {
        int n = x.length;
        int d = x[0].length;

        double[][] distMatrix = computeDistanceMatrix(x);

        int[][] hetNeighs = computeHeterogeneousNeighborhood(x, y, distMatrix);
        int[][] homNeighs = computeHomogeneousNeighborhood(x, y, distMatrix);

        double[][] S = new double[d][d];
        double[][] C = new double[d][d];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nEnemies; j++) {
                int hetJ = hetNeighs[i][j];
                for (int a = 0; a < d; a++) {
                    for (int b = 0; b < d; b++) {
                        S[a][b] += (x[i][a] - x[hetJ][a]) * (x[i][b] - x[hetJ][b]);
                    }
                }
            }

            for (int j = 0; j < nFriends; j++) {
                int homJ = homNeighs[i][j];
                for (int a = 0; a < d; a++) {
                    for (int b = 0; b < d; b++) {
                        C[a][b] += (x[i][a] - x[homJ][a]) * (x[i][b] - x[homJ][b]);
                    }
                }
            }
        }

        for (int a = 0; a < d; a++) {
            for (int b = 0; b < d; b++) {
                S[a][b] /= (n * nEnemies);
                C[a][b] /= (n * nFriends);
            }
        }

        IMatrix<Double> sMat = IMatrix.of(S);
        IMatrix<Double> cMat = IMatrix.of(C);
        double[][] diffData = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                diffData[i][j] = S[i][j] - C[i][j];
            }
        }
        IMatrix<Double> diff = IMatrix.of(diffData);

        var eigResult = diff.eigen();
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

    private int[][] computeHeterogeneousNeighborhood(double[][] x, int[] y, double[][] distMatrix) {
        int n = x.length;
        int[][] hetNeighs = new int[n][nEnemies];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nEnemies; j++) {
                hetNeighs[i][j] = i;
            }
        }

        for (int i = 0; i < n; i++) {
            List<Integer> candidates = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                if (y[j] != y[i]) {
                    candidates.add(j);
                }
            }

            final int row = i;
            candidates.sort((a, b) -> Double.compare(distMatrix[row][a], distMatrix[row][b]));

            for (int j = 0; j < Math.min(nEnemies, candidates.size()); j++) {
                hetNeighs[i][j] = candidates.get(j);
            }
        }

        return hetNeighs;
    }

    private int[][] computeHomogeneousNeighborhood(double[][] x, int[] y, double[][] distMatrix) {
        int n = x.length;
        int[][] homNeighs = new int[n][nFriends];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nFriends; j++) {
                homNeighs[i][j] = i;
            }
        }

        for (int i = 0; i < n; i++) {
            List<Integer> candidates = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                if (j != i && y[j] == y[i]) {
                    candidates.add(j);
                }
            }

            final int row = i;
            candidates.sort((a, b) -> Double.compare(distMatrix[row][a], distMatrix[row][b]));

            for (int j = 0; j < Math.min(nFriends, candidates.size()); j++) {
                homNeighs[i][j] = candidates.get(j);
            }
        }

        return homNeighs;
    }
}
