package com.yishape.lab.math.ml.dml.multidml;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.clf.knn.RereKnn;
import com.yishape.lab.math.ml.dml.DmlArrays;
import com.yishape.lab.math.ml.dml.DmlMetric;

import java.util.*;
import com.yishape.lab.math.ml.dml.ISupervisedDml;

/**
 * Multiple DML k-Nearest Neighbors: 集成多个距离度量学习算法进行 kNN 分类。
 * 每个 DML 算法独立训练，最后通过投票或概率平均进行预测。
 *
 * <p>本类实现 {@link ISupervisedDml}。</p>
 */
public final class MultiDmlKnn implements ISupervisedDml {

    private int nNeighbors = 3;
    private List<ISupervisedDml> dmls = new ArrayList<>();
    private List<double[][]> transformedData = new ArrayList<>();
    private int[] yStore;
    private double[][] xStore;

    public MultiDmlKnn() {
    }

    public MultiDmlKnn(int nNeighbors) {
        this.nNeighbors = nNeighbors;
    }

    public MultiDmlKnn addDml(ISupervisedDml dml) {
        this.dmls.add(dml);
        return this;
    }

    public MultiDmlKnn setNNeighbors(int k) {
        this.nNeighbors = k;
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

    public static DmlMetric fit(IMatrix<Double> features, String[] labels, MultiDmlKnn hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    DmlMetric fitFromRows(double[][] x, int[] y) {
        this.xStore = x;
        this.yStore = y;
        int n = x.length;

        transformedData.clear();

        // 始终包含欧氏距离（作为基准）
        transformedData.add(x);

        // 将 int[] y 转换为 String[]
        String[] yStr = new String[y.length];
        for (int i = 0; i < y.length; i++) {
            yStr[i] = String.valueOf(y[i]);
        }

        // 训练每个 DML 并存储变换后的数据
        for (ISupervisedDml dml : dmls) {
            DmlMetric metric = dml.fit(IMatrix.of(x), yStr);
            double[][] transformed = applyMetric(x, metric);
            transformedData.add(transformed);
        }

        return DmlMetric.lowRank(Linalg.ones(x.length, x.length));
    }

    /**
     * 使用所有集成的度量进行预测（投票）。
     */
    public int[] predict(int[] queryIndices) {
        int n = queryIndices.length;
        int[] predictions = new int[n];

        for (int q = 0; q < n; q++) {
            int queryIdx = queryIndices[q];
            double[] query = xStore[queryIdx];

            // 存储每个 DML 的投票
            Map<Integer, Integer> votes = new HashMap<>();

            for (int d = 0; d < transformedData.size(); d++) {
                double[][] data = transformedData.get(d);
                int[] knnIndices = findKnn(data, query, nNeighbors);

                for (int idx : knnIndices) {
                    int label = yStore[idx];
                    votes.put(label, votes.getOrDefault(label, 0) + 1);
                }
            }

            // 取票数最多的类别
            int maxVotes = 0;
            int bestLabel = -1;
            for (Map.Entry<Integer, Integer> e : votes.entrySet()) {
                if (e.getValue() > maxVotes) {
                    maxVotes = e.getValue();
                    bestLabel = e.getKey();
                }
            }
            predictions[q] = bestLabel;
        }

        return predictions;
    }

    /**
     * 使用指定 DML 索引的度量找到 k 个最近邻。
     */
    public int[] findKnnForDml(int dmlIndex, double[] query) {
        if (dmlIndex < 0 || dmlIndex >= transformedData.size()) {
            throw new IllegalArgumentException("DML 索引超出范围");
        }
        return findKnn(transformedData.get(dmlIndex), query, nNeighbors);
    }

    private int[] findKnn(double[][] data, double[] query, int k) {
        int n = data.length;
        double[] distances = new double[n];
        for (int i = 0; i < n; i++) {
            distances[i] = euclideanDist(data[i], query);
        }

        // 找到 k 个最近邻的索引
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> Double.compare(distances[a], distances[b]));

        int[] knn = new int[Math.min(k, n)];
        for (int i = 0; i < knn.length; i++) {
            knn[i] = indices[i];
        }
        return knn;
    }

    private double[][] applyMetric(double[][] x, DmlMetric metric) {
        IMatrix<Double> xMat = IMatrix.of(x);
        IMatrix<Double> transformed = metric.transform(xMat);
        int n = transformed.getRowNum();
        int d = transformed.getColNum();
        double[][] result = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                result[i][j] = transformed.get(i, j);
            }
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
}
