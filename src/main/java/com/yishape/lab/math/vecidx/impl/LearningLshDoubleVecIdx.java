package com.yishape.lab.math.vecidx.impl;

import com.yishape.lab.math.vecidx.VecSearchOption;
import com.yishape.lab.math.vecidx.IDisMetric;

import java.util.ArrayList;
import java.util.List;

/**
 * 可学习的 LSH 双精度向量索引，内部委托给 {@link LearningLshFloatVecIdx}。
 *
 * <p>将 {@code double[]} 转换为 {@code float[]} 后使用父类全部 ANN 引擎能力，
 * 并提供 {@link #train(List)} 进行监督学习优化。</p>
 */
public final class LearningLshDoubleVecIdx extends AbstractDoubleVecIdx<LearningLshFloatVecIdx> {

    private static final long serialVersionUID = 1L;

    // ==================== SimilarPair ====================

    /**
     * 相似向量对（双精度版）。
     *
     * @param vector1    向量 1
     * @param vector2    向量 2
     * @param similarity 目标碰撞概率 [0, 1]
     */
    public record SimilarPair(double[] vector1, double[] vector2, double similarity) {
        public SimilarPair {
            if (vector1 == null || vector2 == null) {
                throw new IllegalArgumentException("vectors must not be null");
            }
            if (vector1.length != vector2.length) {
                throw new IllegalArgumentException(
                        "vectors must have the same dimension, got "
                                + vector1.length + " vs " + vector2.length);
            }
            if (similarity < 0 || similarity > 1) {
                throw new IllegalArgumentException(
                        "similarity must be in [0, 1], got " + similarity);
            }
        }
    }

    // ==================== 构造函数 ====================

    /** 从数据构建（ADAM, BIAS_WIDTH_ONLY, 默认超参）。 */
    public LearningLshDoubleVecIdx(double[][] data, String[] ids,
            IDisMetric<Double> metric, VecSearchOption options) {
        super(validate(data, ids), metric,
                new LearningLshFloatVecIdx(toFloat(data), ids, floatMetric(metric), options));
    }

    /** 从数据构建（完整配置，直接透传给 {@link LearningLshFloatVecIdx}）。 */
    public LearningLshDoubleVecIdx(double[][] data, String[] ids,
            IDisMetric<Double> metric, VecSearchOption options,
            LearningLshFloatVecIdx.OptimizeMode optimizeMode,
            LearningLshFloatVecIdx.OptimizerType optimizerType,
            double learningRate, int epochs, double l2Regularization,
            int batchSize, double epsilon, double stopThreshold, long seed) {
        super(validate(data, ids), metric,
                new LearningLshFloatVecIdx(toFloat(data), ids, floatMetric(metric), options,
                        optimizeMode, optimizerType, learningRate, epochs,
                        l2Regularization, batchSize, epsilon, stopThreshold, seed));
    }

    /** 从数据构建并自动训练（ADAM, BIAS_WIDTH_ONLY, 默认超参）。 */
    public LearningLshDoubleVecIdx(double[][] data, String[] ids,
            IDisMetric<Double> metric, VecSearchOption options,
            List<SimilarPair> similarPairs) {
        this(data, ids, metric, options);
        train(similarPairs);
    }

    /** 从数据构建并自动训练（完整配置）。 */
    public LearningLshDoubleVecIdx(double[][] data, String[] ids,
            IDisMetric<Double> metric, VecSearchOption options,
            List<SimilarPair> similarPairs,
            LearningLshFloatVecIdx.OptimizeMode optimizeMode,
            LearningLshFloatVecIdx.OptimizerType optimizerType,
            double learningRate, int epochs, double l2Regularization,
            int batchSize, double epsilon, double stopThreshold, long seed) {
        this(data, ids, metric, options, optimizeMode, optimizerType,
                learningRate, epochs, l2Regularization, batchSize, epsilon, stopThreshold, seed);
        train(similarPairs);
    }

    // ==================== 训练委托 ====================

    /**
     * 使用相似对训练 LSH 参数，自动重建索引。
     *
     * <p>将输入 {@code double[]} 转换为 {@code float[]} 后委托给内部的
     * {@link LearningLshFloatVecIdx#train(List)}。</p>
     *
     * @param similarPairs 相似对列表（不可为空）
     */
    public void train(List<SimilarPair> similarPairs) {
        List<LearningLshFloatVecIdx.SimilarPair> floatPairs = new ArrayList<>(similarPairs.size());
        for (SimilarPair p : similarPairs) {
            floatPairs.add(new LearningLshFloatVecIdx.SimilarPair(
                    toFloat(p.vector1()), toFloat(p.vector2()), p.similarity()));
        }
        inner.train(floatPairs);
    }

    public boolean isTrained() {
        return inner.isTrained();
    }

    public double finalLoss() {
        return inner.finalLoss();
    }
}
