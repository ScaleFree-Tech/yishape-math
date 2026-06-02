package com.yishape.lab.math.vecidx.impl;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.IOnlineOptimizer;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.Opts;
import com.yishape.lab.math.optimize.newton.RereOnlineAdam;
import com.yishape.lab.math.vecidx.IDisMetric;
import com.yishape.lab.math.vecidx.VecSearchOption;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import com.yishape.lab.util.YishapeLogger;

/**
 * 可学习的 Multi-Probe LSH 单精度向量索引。
 *
 * <p>继承 {@link LshFloatVecIdx} 的完整 ANN 引擎（多探针、并发、动态增删），
 * 并通过监督学习优化 LSH 参数（投影方向、偏置、桶宽），使已知相似对
 * 具有更高的哈希碰撞概率。</p>
 *
 * <p>学习机制参考 {@code com.reremouse.lab.lsh.ImprovedLearningLSH} 的优化器
 * + 正则化框架，但保留了本库 LshFloatVecIdx 的全部索引基础设施。</p>
 *
 * <h3>使用示例</h3>
 * <pre>{@code
 * // 1. 构建索引（power iteration 初始化投影方向）
 * LearningLshFloatVecIdx idx = new LearningLshFloatVecIdx(data, ids, metric, options);
 *
 * // 2. 准备相似对训练数据
 * List<LearningLshFloatVecIdx.SimilarPair> pairs = List.of(
 *     new LearningLshFloatVecIdx.SimilarPair(data[0], data[1], 0.95),
 *     new LearningLshFloatVecIdx.SimilarPair(data[2], data[3], 0.80),
 *     ...
 * );
 *
 * // 3. 训练（ADAM 优化投影方向/偏置/桶宽，训练后自动重建索引）
 * idx.train(pairs);
 *
 * // 4. 训练后的索引可直接查询
 * List<SearchHit> hits = idx.search(query, 10);
 * }</pre>
 *
 * @author lteb2
 * @since 2.8
 */
public class LearningLshFloatVecIdx extends LshFloatVecIdx {

    private static final long serialVersionUID = 1L;
    private static final YishapeLogger log = YishapeLogger.getLogger(LearningLshFloatVecIdx.class);

    /** 优化参数范围 */
    public enum OptimizeMode {
        /** 只优化偏置和桶宽（参数量 = L×K×2，训练快，适合大多数场景） */
        BIAS_WIDTH_ONLY,
        /** 优化投影方向 + 偏置 + 桶宽（参数量 = L×K×(D+2)，训练慢，理论上效果更好） */
        FULL
    }

    /** 优化器类型 */
    public enum OptimizerType {
        LBFGS, ADAM
    }

    // ==================== 训练配置 ====================

    private final OptimizeMode optimizeMode;
    private final OptimizerType optimizerType;
    private final double learningRate;
    private final int epochs;
    private final double l2Regularization;
    private final int batchSize;
    private final double epsilon;
    private final double stopThreshold;
    private final transient Random random;

    // ==================== 训练状态 ====================

    private boolean isTrained = false;
    private double finalLoss = Double.NaN;

    // ==================== SimilarPair ====================

    /**
     * 相似向量对 —— 监督学习的训练样本。
     *
     * @param vector1   向量 1
     * @param vector2   向量 2
     * @param similarity 目标碰撞概率 [0, 1]，1 表示应始终落在同一桶
     */
    public record SimilarPair(float[] vector1, float[] vector2, double similarity) {
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
    public LearningLshFloatVecIdx(float[][] data, String[] ids,
            IDisMetric<Float> metric, VecSearchOption options) {
        this(data, ids, metric, options, OptimizeMode.BIAS_WIDTH_ONLY, OptimizerType.ADAM,
                0.01, 100, 1e-4, 32, 1e-5, 1e-6, 42L);
    }

    /** 从数据构建（完整配置）。 */
    public LearningLshFloatVecIdx(float[][] data, String[] ids,
            IDisMetric<Float> metric, VecSearchOption options,
            OptimizeMode optimizeMode, OptimizerType optimizerType,
            double learningRate, int epochs, double l2Regularization,
            int batchSize, double epsilon, double stopThreshold, long seed) {
        super(data, ids, metric, options);
        this.optimizeMode = optimizeMode;
        this.optimizerType = optimizerType;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.l2Regularization = l2Regularization;
        this.batchSize = batchSize;
        this.epsilon = epsilon;
        this.stopThreshold = stopThreshold;
        this.random = new Random(seed);

        if (learningRate <= 0 || epochs <= 0 || batchSize <= 0 || epsilon <= 0 || stopThreshold <= 0) {
            throw new IllegalArgumentException("learningRate, epochs, batchSize, epsilon, stopThreshold must be > 0");
        }
    }

    /** 从数据构建并自动训练（ADAM, BIAS_WIDTH_ONLY, 默认超参）。 */
    public LearningLshFloatVecIdx(float[][] data, String[] ids,
            IDisMetric<Float> metric, VecSearchOption options,
            List<SimilarPair> similarPairs) {
        this(data, ids, metric, options);
        train(similarPairs);
    }

    /** 从数据构建并自动训练（完整配置）。 */
    public LearningLshFloatVecIdx(float[][] data, String[] ids,
            IDisMetric<Float> metric, VecSearchOption options,
            List<SimilarPair> similarPairs,
            OptimizeMode optimizeMode, OptimizerType optimizerType,
            double learningRate, int epochs, double l2Regularization,
            int batchSize, double epsilon, double stopThreshold, long seed) {
        this(data, ids, metric, options, optimizeMode, optimizerType,
                learningRate, epochs, l2Regularization, batchSize, epsilon, stopThreshold, seed);
        train(similarPairs);
    }

    // ==================== 训练 API ====================

    /**
     * 使用相似对训练 LSH 参数，自动重建索引。
     *
     * <p>训练期间索引处于不一致状态；训练应在离线阶段完成，
     * 完成后方可进行并发查询。</p>
     *
     * @param similarPairs 相似对列表（不可为空）
     */
    public void train(List<SimilarPair> similarPairs) {
        if (similarPairs == null || similarPairs.isEmpty()) {
            throw new IllegalArgumentException("similarPairs must not be null or empty");
        }

        int totalParams = countParameters();
        log.info("[LearningLsh] Starting training: mode={}, optimizer={}, params={}, pairs={}",
                optimizeMode, optimizerType, totalParams, similarPairs.size());

        double initLoss = evaluateLoss(similarPairs);
        log.info(String.format("[LearningLsh] Initial loss: %.6f", initLoss));

        long start = System.currentTimeMillis();
        switch (optimizerType) {
            case LBFGS -> trainWithLBFGS(similarPairs);
            case ADAM -> trainWithAdam(similarPairs);
        }
        long elapsed = System.currentTimeMillis() - start;

        rebuildIndex();

        finalLoss = evaluateLoss(similarPairs);
        isTrained = true;
        log.info(String.format("[LearningLsh] Done in %d ms. Final loss: %.6f (improved %.6f)",
                elapsed, finalLoss, initLoss - finalLoss));
    }

    public boolean isTrained() {
        return isTrained;
    }

    public double finalLoss() {
        return finalLoss;
    }

    // ==================== 训练实现 ====================

    private void trainWithLBFGS(List<SimilarPair> pairs) {
        IVector initParams = packParameters();
        IOptimizer optimizer = Opts.lbfgs(stopThreshold, epochs);
        OptResult result = optimizer.optimize(initParams,
                new LbfgsObjective(pairs),
                new LbfgsGradient(pairs));
        unpackParameters(result.getOptimalPoint());
    }

    private void trainWithAdam(List<SimilarPair> pairs) {
        IVector params = packParameters();
        IOnlineOptimizer adam = Opts.onlineAdam(learningRate, 0.9, 0.999);
//        adam.setEpsilon(1e-8);
        adam.initialize(params);

        double bestLoss = evaluateLossWithParams(pairs, params);
        int patience = 0;
        final int maxPatience = 10;

        for (int epoch = 0; epoch < epochs; epoch++) {
            List<SimilarPair> batch = sampleBatch(pairs);
            IVector grad = computeNumericalGradient(batch, params);
            params = adam.step(grad);

            double fullLoss = evaluateLossWithParams(pairs, params);
            if (fullLoss < bestLoss) {
                bestLoss = fullLoss;
                patience = 0;
            } else {
                patience++;
            }

            if ((epoch + 1) % 10 == 0 || epoch == 0) {
                log.debug(String.format("[LearningLsh] Epoch %d/%d, loss=%.6f, best=%.6f, patience=%d",
                        epoch + 1, epochs, fullLoss, bestLoss, patience));
            }

            if (patience >= maxPatience || fullLoss < stopThreshold) {
                log.info("[LearningLsh] Early stopping at epoch {}", epoch + 1);
                break;
            }
        }

        unpackParameters(params);
    }

    // ==================== 参数 Pack / Unpack ====================

    private int countParameters() {
        int perHash = optimizeMode == OptimizeMode.FULL ? dimensions + 2 : 2;
        return numTables * numBits * perHash;
    }

    /**
     * 将 projections / biases / bucketWidths 打包为参数向量。
     * 参数顺序：[A 矩阵 (可选)] B 向量 W 向量。
     */
    private IVector packParameters() {
        double[] p = new double[countParameters()];
        int idx = 0;
        for (int t = 0; t < numTables; t++) {
            for (int b = 0; b < numBits; b++) {
                if (optimizeMode == OptimizeMode.FULL) {
                    for (int d = 0; d < dimensions; d++) {
                        p[idx++] = projections[t][b][d];
                    }
                }
                p[idx++] = biases[t][b];
                p[idx++] = bucketWidths[t][b];
            }
        }
        return Linalg.vector(p);
    }

    /** 从参数向量恢复 projections / biases / bucketWidths。 */
    private void unpackParameters(IVector params) {
        int idx = 0;
        for (int t = 0; t < numTables; t++) {
            for (int b = 0; b < numBits; b++) {
                if (optimizeMode == OptimizeMode.FULL) {
                    for (int d = 0; d < dimensions; d++) {
                        projections[t][b][d] = (float) params.get(idx++);
                    }
                }
                biases[t][b] = (float) params.get(idx++);
                double w = params.get(idx++);
                bucketWidths[t][b] = (float) Math.max(w, 1e-6);
            }
        }
    }

    // ==================== 损失函数 ====================

    /**
     * 对当前父类参数评估损失。
     * Loss = MSE(target, actual_collision_prob) + L2 * ||params||²
     */
    private double evaluateLoss(List<SimilarPair> pairs) {
        double total = 0.0;
        int[] buf1 = new int[numBits];
        int[] buf2 = new int[numBits];

        for (SimilarPair pair : pairs) {
            double actual = computeCollisionProbability(pair.vector1(), pair.vector2(), buf1, buf2);
            double diff = pair.similarity() - actual;
            total += diff * diff;
        }
        double mse = total / pairs.size();
        return mse + l2Regularization * l2NormSq();
    }

    /**
     * 先 unpack 新参数到父类字段 → 评估损失 → 恢复旧参数。
     * 用于梯度计算中反复切换参数。
     */
    private double evaluateLossWithParams(List<SimilarPair> pairs, IVector params) {
        IVector saved = packParameters();
        unpackParameters(params);
        double loss = evaluateLoss(pairs);
        unpackParameters(saved);
        return loss;
    }

    /** 计算两个向量在每张表上的碰撞概率 = 碰撞表数 / L。 */
    private double computeCollisionProbability(float[] v1, float[] v2, int[] buf1, int[] buf2) {
        int collisions = 0;
        for (int t = 0; t < numTables; t++) {
            computeBucketsInto(v1, t, buf1);
            computeBucketsInto(v2, t, buf2);
            boolean match = true;
            for (int b = 0; b < numBits; b++) {
                if (buf1[b] != buf2[b]) {
                    match = false;
                    break;
                }
            }
            if (match) collisions++;
        }
        return (double) collisions / numTables;
    }

    /** 计算当前父类参数的 L2 范数平方（仅被优化的参数）。 */
    private double l2NormSq() {
        double l2 = 0.0;
        for (int t = 0; t < numTables; t++) {
            for (int b = 0; b < numBits; b++) {
                if (optimizeMode == OptimizeMode.FULL) {
                    for (int d = 0; d < dimensions; d++) {
                        l2 += projections[t][b][d] * projections[t][b][d];
                    }
                }
                l2 += biases[t][b] * biases[t][b];
                l2 += bucketWidths[t][b] * bucketWidths[t][b];
            }
        }
        return l2;
    }

    // ==================== 数值梯度 ====================

    /** 前向有限差分计算梯度，包含 L2 正则项。 */
    private IVector computeNumericalGradient(List<SimilarPair> pairs, IVector params) {
        int n = countParameters();
        double[] grad = new double[n];
        double baseLoss = evaluateLossWithParams(pairs, params);

        for (int i = 0; i < n; i++) {
            double orig = params.get(i);
            params.set(i, orig + epsilon);
            double lossPlus = evaluateLossWithParams(pairs, params);
            params.set(i, orig);
            grad[i] = (lossPlus - baseLoss) / epsilon;
        }

        for (int i = 0; i < n; i++) {
            grad[i] += 2.0 * l2Regularization * params.get(i);
        }

        return Linalg.vector(grad);
    }

    // ==================== 小批量采样 ====================

    private List<SimilarPair> sampleBatch(List<SimilarPair> all) {
        if (all.size() <= batchSize) {
            return all;
        }
        List<SimilarPair> shuffled = new ArrayList<>(all);
        Collections.shuffle(shuffled, random);
        return shuffled.subList(0, batchSize);
    }

    // ==================== LBFGS 适配器 ====================

    private class LbfgsObjective implements IObjectiveFunction {
        private final List<SimilarPair> pairs;

        LbfgsObjective(List<SimilarPair> pairs) {
            this.pairs = pairs;
        }

        @Override
        public double computeObjective(IVector x) {
            return evaluateLossWithParams(pairs, x);
        }
    }

    private class LbfgsGradient implements IGradientFunction {
        private final List<SimilarPair> pairs;

        LbfgsGradient(List<SimilarPair> pairs) {
            this.pairs = pairs;
        }

        @Override
        public IVector computeGradient(IVector x) {
            return computeNumericalGradient(pairs, x);
        }
    }
}
