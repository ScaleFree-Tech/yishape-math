package com.yishape.lab.math.vecidx;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.vecidx.distance.EuclideanMetric;
import com.yishape.lab.math.vecidx.impl.*;

import java.util.List;
import java.util.Objects;

/**
 * 向量索引的静态工厂入口。
 *
 * <p>调用方通过 {@link IdxType} 指定索引类型，或使用 {@link IdxType#AUTO}
 * 让工厂根据数据规模自动选择。</p>
 */
public final class VI {

    private VI() {
    }

    // ==================== AUTO 路由 ====================

    private static IdxType resolveAuto(int n, int dim, IDisMetric<?> metric) {
        if (!isSupportedBuiltIn(metric)) return IdxType.BRUTE_FORCE;
        if (n < 1_000) return IdxType.BRUTE_FORCE;
        if (dim <= 20) return IdxType.KDTree;
        if (n >= 50_000) return IdxType.HNSW;
        if (n >= 5_000) return IdxType.LSH;
        return IdxType.BRUTE_FORCE;
    }

    private static boolean isSupportedBuiltIn(IDisMetric<?> metric) {
        MetricType mt = metric.type();
        return mt == MetricType.EUCLIDEAN || mt == MetricType.SQUARED_EUCLIDEAN || mt == MetricType.COSINE;
    }

    private static IdxType effectiveType(VecSearchOption options, int n, int dim, IDisMetric<?> metric) {
        IdxType t = options.indexType();
        if (t == IdxType.AUTO) {
            return resolveAuto(n, dim, metric);
        }
        return t;
    }

    // ==================== double[][] ====================

    public static IDoubleVecIdx buildDouble(double[][] data, String[] ids,
            IDisMetric<Double> metric, VecSearchOption options) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(ids, "ids");
        Objects.requireNonNull(metric, "metric");
        Objects.requireNonNull(options, "options");
        if (ids.length == 0) throw new IllegalArgumentException("ids 不能为空");
        int dim = data.length > 0 && data[0] != null ? data[0].length : 0;
        IdxType t = effectiveType(options, ids.length, dim, metric);
        return buildDoubleImpl(data, ids, metric, options, t);
    }

    public static IDoubleVecIdx buildDouble(double[][] data, String[] ids, VecSearchOption options) {
        return buildDouble(data, ids, EuclideanMetric.DOUBLE, options);
    }

    public static IDoubleVecIdx buildDoubleSubset(double[][] fullMatrix, String[] subsetIds,
            IDisMetric<Double> metric, VecSearchOption options) {
        Objects.requireNonNull(fullMatrix, "fullMatrix");
        Objects.requireNonNull(subsetIds, "subsetIds");
        Objects.requireNonNull(metric, "metric");
        Objects.requireNonNull(options, "options");
        if (subsetIds.length == 0) throw new IllegalArgumentException("ids 不能为空");
        int dim = fullMatrix.length > 0 && fullMatrix[0] != null ? fullMatrix[0].length : 0;
        // Extract subset rows from full matrix
        double[][] subsetData = new double[subsetIds.length][dim];
        for (int i = 0; i < subsetIds.length; i++) {
            int rowIdx = Integer.parseInt(subsetIds[i]);
            if (rowIdx < 0 || rowIdx >= fullMatrix.length) {
                throw new IllegalArgumentException("Subset index " + rowIdx + " out of range [0, " + fullMatrix.length + ")");
            }
            subsetData[i] = fullMatrix[rowIdx].clone();
        }
        IdxType t = effectiveType(options, subsetIds.length, dim, metric);
        return buildDoubleImpl(subsetData, subsetIds, metric, options, t);
    }

    public static IDoubleVecIdx buildDoubleSubset(double[][] fullMatrix, String[] subsetIds,
            VecSearchOption options) {
        return buildDoubleSubset(fullMatrix, subsetIds, EuclideanMetric.DOUBLE, options);
    }

    private static IDoubleVecIdx buildDoubleImpl(double[][] data, String[] ids,
            IDisMetric<Double> metric, VecSearchOption options, IdxType t) {
        return switch (t) {
            case HNSW -> {
                try { yield new RereHnswDoubleVecIdx(data, ids, metric, options); }
                catch (Exception e) { yield new BruteForceDoubleVecIdx(data, ids, metric); }
            }
            case KDTree -> new KdTreeDoubleVecIdx(data, ids, metric, options);
            case LSH -> new LshDoubleVecIdx(data, ids, metric, options);
            case PQ -> new PqDoubleVecIdx(data, ids, metric, options, false);
            case PQ_HNSW -> new PqHnswDoubleVecIdx(data, ids, metric, options, false);
            case AUTO, BRUTE_FORCE -> new BruteForceDoubleVecIdx(data, ids, metric);
        };
    }

    // ==================== float[][] ====================

    public static IFloatVecIdx buildFloat(float[][] data, String[] ids,
            IDisMetric<Float> metric, VecSearchOption options) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(ids, "ids");
        Objects.requireNonNull(metric, "metric");
        Objects.requireNonNull(options, "options");
        if (ids.length == 0) throw new IllegalArgumentException("ids 不能为空");
        int dim = data.length > 0 && data[0] != null ? data[0].length : 0;
        IdxType t = effectiveType(options, ids.length, dim, metric);
        return buildFloatImpl(data, ids, metric, options, t);
    }

    public static IFloatVecIdx buildFloat(float[][] data, String[] ids, VecSearchOption options) {
        return buildFloat(data, ids, EuclideanMetric.FLOAT, options);
    }

    public static IFloatVecIdx buildFloatSubset(float[][] fullMatrix, String[] subsetIds,
            IDisMetric<Float> metric, VecSearchOption options) {
        Objects.requireNonNull(fullMatrix, "fullMatrix");
        Objects.requireNonNull(subsetIds, "subsetIds");
        Objects.requireNonNull(metric, "metric");
        Objects.requireNonNull(options, "options");
        if (subsetIds.length == 0) throw new IllegalArgumentException("ids 不能为空");
        int dim = fullMatrix.length > 0 && fullMatrix[0] != null ? fullMatrix[0].length : 0;
        // Extract subset rows from full matrix
        float[][] subsetData = new float[subsetIds.length][dim];
        for (int i = 0; i < subsetIds.length; i++) {
            int rowIdx = Integer.parseInt(subsetIds[i]);
            if (rowIdx < 0 || rowIdx >= fullMatrix.length) {
                throw new IllegalArgumentException("Subset index " + rowIdx + " out of range [0, " + fullMatrix.length + ")");
            }
            subsetData[i] = fullMatrix[rowIdx].clone();
        }
        IdxType t = effectiveType(options, subsetIds.length, dim, metric);
        return buildFloatImpl(subsetData, subsetIds, metric, options, t);
    }

    public static IFloatVecIdx buildFloatSubset(float[][] fullMatrix, String[] subsetIds,
            VecSearchOption options) {
        return buildFloatSubset(fullMatrix, subsetIds, EuclideanMetric.FLOAT, options);
    }

    private static IFloatVecIdx buildFloatImpl(float[][] data, String[] ids,
            IDisMetric<Float> metric, VecSearchOption options, IdxType t) {
        return switch (t) {
            case HNSW -> {
                try { yield new RustHnswFloatVecIdx(data, ids, metric, options); }
                catch (Exception e) { yield new BruteForceFloatVecIdx(data, ids, metric); }
            }
            case KDTree -> new KdTreeFloatVecIdx(data, ids, metric, options);
            case LSH -> new LshFloatVecIdx(data, ids, metric, options);
            case PQ -> new PqFloatVecIdx(data, ids, metric, options, false);
            case PQ_HNSW -> new PqHnswFloatVecIdx(data, ids, metric, options, false);
            case AUTO, BRUTE_FORCE -> new BruteForceFloatVecIdx(data, ids, metric);
        };
    }

    // ==================== 显式构建 ====================

    public static IDoubleVecIdx buildLshDouble(double[][] data, String[] ids,
            IDisMetric<Double> metric, VecSearchOption options) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(ids, "ids");
        Objects.requireNonNull(metric, "metric");
        return new LshDoubleVecIdx(data, ids, metric, options);
    }

    public static IFloatVecIdx buildLshFloat(float[][] data, String[] ids,
            IDisMetric<Float> metric, VecSearchOption options) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(ids, "ids");
        Objects.requireNonNull(metric, "metric");
        return new LshFloatVecIdx(data, ids, metric, options);
    }

    // ==================== Learning LSH ====================

    public static IFloatVecIdx buildLearningLshFloat(float[][] data, String[] ids,
            IDisMetric<Float> metric, VecSearchOption options,
            List<LearningLshFloatVecIdx.SimilarPair> similarPairs) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(ids, "ids");
        Objects.requireNonNull(metric, "metric");
        Objects.requireNonNull(similarPairs, "similarPairs");
        return new LearningLshFloatVecIdx(data, ids, metric, options, similarPairs);
    }

    public static IFloatVecIdx buildLearningLshFloat(float[][] data, String[] ids,
            IDisMetric<Float> metric, VecSearchOption options,
            List<LearningLshFloatVecIdx.SimilarPair> similarPairs,
            LearningLshFloatVecIdx.OptimizeMode optimizeMode,
            LearningLshFloatVecIdx.OptimizerType optimizerType,
            double learningRate, int epochs, double l2Regularization,
            int batchSize, double epsilon, double stopThreshold, long seed) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(ids, "ids");
        Objects.requireNonNull(metric, "metric");
        Objects.requireNonNull(similarPairs, "similarPairs");
        return new LearningLshFloatVecIdx(data, ids, metric, options, similarPairs,
                optimizeMode, optimizerType, learningRate, epochs,
                l2Regularization, batchSize, epsilon, stopThreshold, seed);
    }

    public static IDoubleVecIdx buildLearningLshDouble(double[][] data, String[] ids,
            IDisMetric<Double> metric, VecSearchOption options,
            List<LearningLshDoubleVecIdx.SimilarPair> similarPairs) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(ids, "ids");
        Objects.requireNonNull(metric, "metric");
        Objects.requireNonNull(similarPairs, "similarPairs");
        return new LearningLshDoubleVecIdx(data, ids, metric, options, similarPairs);
    }

    public static IDoubleVecIdx buildLearningLshDouble(double[][] data, String[] ids,
            IDisMetric<Double> metric, VecSearchOption options,
            List<LearningLshDoubleVecIdx.SimilarPair> similarPairs,
            LearningLshFloatVecIdx.OptimizeMode optimizeMode,
            LearningLshFloatVecIdx.OptimizerType optimizerType,
            double learningRate, int epochs, double l2Regularization,
            int batchSize, double epsilon, double stopThreshold, long seed) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(ids, "ids");
        Objects.requireNonNull(metric, "metric");
        Objects.requireNonNull(similarPairs, "similarPairs");
        return new LearningLshDoubleVecIdx(data, ids, metric, options, similarPairs,
                optimizeMode, optimizerType, learningRate, epochs,
                l2Regularization, batchSize, epsilon, stopThreshold, seed);
    }

    public static IDoubleVecIdx buildPqDouble(double[][] data, String[] ids,
            IDisMetric<Double> metric, VecSearchOption options) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(ids, "ids");
        Objects.requireNonNull(metric, "metric");
        return new PqDoubleVecIdx(data, ids, metric, options, false);
    }

    public static IFloatVecIdx buildPqFloat(float[][] data, String[] ids,
            IDisMetric<Float> metric, VecSearchOption options) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(ids, "ids");
        Objects.requireNonNull(metric, "metric");
        return new PqFloatVecIdx(data, ids, metric, options, false);
    }

    public static IDoubleVecIdx buildOpqDouble(double[][] data, String[] ids,
            IDisMetric<Double> metric, VecSearchOption options) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(ids, "ids");
        Objects.requireNonNull(metric, "metric");
        return new PqDoubleVecIdx(data, ids, metric, options, true);
    }

    public static IFloatVecIdx buildOpqFloat(float[][] data, String[] ids,
            IDisMetric<Float> metric, VecSearchOption options) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(ids, "ids");
        Objects.requireNonNull(metric, "metric");
        return new PqFloatVecIdx(data, ids, metric, options, true);
    }

    public static IDoubleVecIdx buildPqHnswDouble(double[][] data, String[] ids,
            IDisMetric<Double> metric, VecSearchOption options) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(ids, "ids");
        Objects.requireNonNull(metric, "metric");
        return new PqHnswDoubleVecIdx(data, ids, metric, options, false);
    }

    public static IFloatVecIdx buildPqHnswFloat(float[][] data, String[] ids,
            IDisMetric<Float> metric, VecSearchOption options) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(ids, "ids");
        Objects.requireNonNull(metric, "metric");
        return new PqHnswFloatVecIdx(data, ids, metric, options, false);
    }

    public static IDoubleVecIdx buildOpqHnswDouble(double[][] data, String[] ids,
            IDisMetric<Double> metric, VecSearchOption options) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(ids, "ids");
        Objects.requireNonNull(metric, "metric");
        return new PqHnswDoubleVecIdx(data, ids, metric, options, true);
    }

    public static IFloatVecIdx buildOpqHnswFloat(float[][] data, String[] ids,
            IDisMetric<Float> metric, VecSearchOption options) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(ids, "ids");
        Objects.requireNonNull(metric, "metric");
        return new PqHnswFloatVecIdx(data, ids, metric, options, true);
    }

    // ==================== IMatrix ====================

    @SuppressWarnings("unchecked")
    public static <T extends Number> IVecIdx<T> build(IMatrix<T> data, String[] ids,
            IDisMetric<T> metric, VecSearchOption options) {
        Objects.requireNonNull(data, "data");
        if (ids == null || ids.length == 0) throw new IllegalArgumentException("ids 必须非空");
        double[][] d = data.toDoubleArray();
        return (IVecIdx<T>) buildDouble(d, ids, (IDisMetric<Double>) metric, options);
    }

    // ==================== 可变索引 ====================

    public static IMutableVecIdx<Double> newMutableDouble(int dimensions,
            IDisMetric<Double> metric, VecSearchOption options) {
        Objects.requireNonNull(metric, "metric");
        return new BruteForceDoubleVecIdx(dimensions, metric);
    }

    public static IMutableVecIdx<Float> newMutableFloat(int dimensions,
            IDisMetric<Float> metric, VecSearchOption options) {
        Objects.requireNonNull(metric, "metric");
        return new BruteForceFloatVecIdx(dimensions, metric);
    }
}
