package com.yishape.lab.math.ml.dml.knn;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.dml.DmlArrays;
import com.yishape.lab.math.ml.dml.DmlMetric;

import java.util.*;
import com.yishape.lab.math.ml.dml.ISupervisedDml;

/**
 * Reduced Nearest Neighbors (RNN): 在 CNN 基础上进一步精简，移除不影响分类的样本。
 *
 * <p>本类实现 {@link ISupervisedDml}。</p>
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>G. W. Gates, "The reduced nearest neighbor rule". <em>IEEE Trans. Electron. Comput.</em>, 1972.</li>
 * </ul>
 */
public final class ReducedNearestNeighbors implements ISupervisedDml {

    private CondensedNearestNeighbors cnn = new CondensedNearestNeighbors();
    private Set<Integer> reducedSet = new LinkedHashSet<>();
    private int[] yStore;
    private double[][] xStore;

    @Override
    public DmlMetric fit(IMatrix<Double> features, IVector<?> labels) {
        Objects.requireNonNull(features, "features");
        Objects.requireNonNull(labels, "labels");
        double[][] x = toDoubleArray(features);
        int[] y = toIntArray(labels);
        fitFromRows(x, y);
        return DmlMetric.lowRank(Linalg.ones(x[0].length, x[0].length));
    }

    @Override
    public DmlMetric fit(IMatrix<Double> features, String[] labels) {
        Objects.requireNonNull(features, "features");
        Objects.requireNonNull(labels, "labels");
        double[][] x = toDoubleArray(features);
        int[] y = DmlArrays.classIndices(labels);
        fitFromRows(x, y);
        return DmlMetric.lowRank(Linalg.ones(x[0].length, x[0].length));
    }

    public static DmlMetric fit(IMatrix<Double> features, IVector<?> labels, ReducedNearestNeighbors hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    public void fitFromRows(double[][] x, int[] y) {
        int n = x.length;
        if (n == 0) return;

        this.xStore = x;
        this.yStore = y;

        // Step 1: CNN
        cnn.fitFromRows(x, y);
        Set<Integer> cnnSet = new LinkedHashSet<>();
        for (int idx : cnn.getCondensedIndexes()) {
            cnnSet.add(idx);
        }

        // Step 2: Remove redundant samples
        reducedSet.clear();
        reducedSet.addAll(cnnSet);

        List<Integer> rnnList = new ArrayList<>(reducedSet);
        for (Integer idx : rnnList) {
            Set<Integer> candidate = new LinkedHashSet<>(reducedSet);
            candidate.remove(idx);

            if (candidate.isEmpty()) continue;

            int[] candArr = candidate.stream().mapToInt(Integer::intValue).toArray();
            double[][] candX = new double[candArr.length][];
            int[] candY = new int[candArr.length];
            for (int j = 0; j < candArr.length; j++) {
                candX[j] = x[candArr[j]];
                candY[j] = y[candArr[j]];
            }

            // 验证移除后仍能正确分类所有样本
            boolean allCorrect = true;
            for (int i = 0; i < n; i++) {
                int nearest = -1;
                double minDist = Double.MAX_VALUE;
                for (int j = 0; j < candArr.length; j++) {
                    double dist = euclideanDist(x[i], candX[j]);
                    if (dist < minDist) {
                        minDist = dist;
                        nearest = candY[j];
                    }
                }
                if (nearest != y[i]) {
                    allCorrect = false;
                    break;
                }
            }

            if (allCorrect) {
                reducedSet.remove(idx);
            }
        }
    }

    public int[] getReducedIndexes() {
        return reducedSet.stream().mapToInt(Integer::intValue).toArray();
    }

    public double[][] getReducedSamples() {
        int[] idx = getReducedIndexes();
        double[][] result = new double[idx.length][];
        for (int i = 0; i < idx.length; i++) {
            result[i] = xStore[idx[i]];
        }
        return result;
    }

    public int[] getReducedLabels() {
        int[] idx = getReducedIndexes();
        int[] result = new int[idx.length];
        for (int i = 0; i < idx.length; i++) {
            result[i] = yStore[idx[i]];
        }
        return result;
    }

    private static double euclideanDist(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }

    private static double[][] toDoubleArray(IMatrix<Double> m) {
        int rows = m.getRowNum();
        int cols = m.getColNum();
        double[][] arr = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                arr[i][j] = (Double) m.get(i, j);
            }
        }
        return arr;
    }

    private static int[] toIntArray(IVector<?> v) {
        int n = v.size();
        int[] arr = new int[n];
        for (int i = 0; i < n; i++) {
            Object val = v.get(i);
            if (val instanceof Number) {
                arr[i] = ((Number) val).intValue();
            }
        }
        return arr;
    }
}
