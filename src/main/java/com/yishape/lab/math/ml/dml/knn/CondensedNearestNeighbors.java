package com.yishape.lab.math.ml.dml.knn;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.dml.DmlArrays;
import com.yishape.lab.math.ml.dml.DmlMetric;

import java.util.*;
import com.yishape.lab.math.ml.dml.ISupervisedDml;

/**
 * Condensed Nearest Neighbors (CNN): 用于欠采样的最近邻压缩算法。
 * 选择最小的子集来保持分类准确性。
 *
 * <p>本类实现 {@link ISupervisedDml}。</p>
 *
 * @apiNote 应用代码优先 {@link com.yishape.lab.math.ml.ML#dml}。
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>P. E. Hart, "The condensed nearest neighbor rule". <em>IEEE Trans. Inf. Theory</em>, 1968.</li>
 * </ul>
 */
public final class CondensedNearestNeighbors implements ISupervisedDml {

    private Set<Integer> condensedSet = new LinkedHashSet<>();
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

    public static DmlMetric fit(IMatrix<Double> features, IVector<?> labels, CondensedNearestNeighbors hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    public void fitFromRows(double[][] x, int[] y) {
        int n = x.length;
        if (n == 0) return;

        this.xStore = x;
        this.yStore = y;

        condensedSet.clear();
        condensedSet.add(0);

        boolean added = true;
        while (added) {
            added = false;
            for (int i = 1; i < n; i++) {
                if (condensedSet.contains(i)) continue;

                int[] cnnArr = condensedSet.stream().mapToInt(Integer::intValue).toArray();
                double[][] cnnX = new double[cnnArr.length][];
                int[] cnnY = new int[cnnArr.length];
                for (int j = 0; j < cnnArr.length; j++) {
                    cnnX[j] = x[cnnArr[j]];
                    cnnY[j] = y[cnnArr[j]];
                }

                // 找到最近邻
                int nearest = -1;
                double minDist = Double.MAX_VALUE;
                for (int j = 0; j < cnnArr.length; j++) {
                    double dist = euclideanDist(x[i], cnnX[j]);
                    if (dist < minDist) {
                        minDist = dist;
                        nearest = cnnY[j];
                    }
                }

                if (nearest != y[i]) {
                    condensedSet.add(i);
                    added = true;
                }
            }
        }
    }

    /**
     * 获取压缩后的样本索引。
     */
    public int[] getCondensedIndexes() {
        return condensedSet.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * 获取压缩后的样本。
     */
    public double[][] getCondensedSamples() {
        int[] idx = getCondensedIndexes();
        double[][] result = new double[idx.length][];
        for (int i = 0; i < idx.length; i++) {
            result[i] = xStore[idx[i]];
        }
        return result;
    }

    /**
     * 获取压缩后的标签。
     */
    public int[] getCondensedLabels() {
        int[] idx = getCondensedIndexes();
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
