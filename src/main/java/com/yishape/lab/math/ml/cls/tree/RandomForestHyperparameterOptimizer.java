package com.yishape.lab.math.ml.cls.tree;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.newton.RereLBFGS;
import com.yishape.lab.math.optimize.newton.RereOnlineAdam;

/**
 * 随机森林超参数优化器
 * <p>
 * 使用优化算法自动调优随机森林的超参数，包括：
 * - 决策树数量 (nEstimators)
 * - 最大深度 (maxDepth)
 * - 最小分裂样本数 (minSamplesSplit)
 * - 最小叶子样本数 (minSamplesLeaf)
 * - 最大特征数 (maxFeatures)
 * </p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class RandomForestHyperparameterOptimizer implements IObjectiveFunction {
    
    /** 训练特征矩阵 */
    private IMatrix trainFeatures;
    
    /** 训练标签 */
    private String[] trainLabels;
    
    /** 验证特征矩阵 */
    private IMatrix validFeatures;
    
    /** 验证标签 */
    private String[] validLabels;
    
    /** 超参数边界 */
    private HyperparameterBounds bounds;
    
    /** 随机种子 */
    private long randomSeed;
    
    /** 交叉验证折数 */
    private int cvFolds;
    
    /**
     * 超参数边界类
     */
    public static class HyperparameterBounds {
        public int minEstimators = 10;
        public int maxEstimators = 200;
        public int minDepth = 3;
        public int maxDepth = 20;
        public int minSamplesSplit = 2;
        public int maxSamplesSplit = 20;
        public int minSamplesLeaf = 1;
        public int maxSamplesLeaf = 10;
        public double minMaxFeaturesRatio = 0.1;
        public double maxMaxFeaturesRatio = 1.0;
        
        public HyperparameterBounds() {}
        
        public HyperparameterBounds(int minEstimators, int maxEstimators, 
                                  int minDepth, int maxDepth,
                                  int minSamplesSplit, int maxSamplesSplit,
                                  int minSamplesLeaf, int maxSamplesLeaf,
                                  double minMaxFeaturesRatio, double maxMaxFeaturesRatio) {
            this.minEstimators = minEstimators;
            this.maxEstimators = maxEstimators;
            this.minDepth = minDepth;
            this.maxDepth = maxDepth;
            this.minSamplesSplit = minSamplesSplit;
            this.maxSamplesSplit = maxSamplesSplit;
            this.minSamplesLeaf = minSamplesLeaf;
            this.maxSamplesLeaf = maxSamplesLeaf;
            this.minMaxFeaturesRatio = minMaxFeaturesRatio;
            this.maxMaxFeaturesRatio = maxMaxFeaturesRatio;
        }
    }
    
    /**
     * 优化结果类
     */
    public static class OptimizationResult {
        public final int bestNEstimators;
        public final int bestMaxDepth;
        public final int bestMinSamplesSplit;
        public final int bestMinSamplesLeaf;
        public final int bestMaxFeatures;
        public final double bestScore;
        public final RereRandomForest bestModel;
        
        public OptimizationResult(int nEstimators, int maxDepth, int minSamplesSplit,
                                int minSamplesLeaf, int maxFeatures, double score,
                                RereRandomForest model) {
            this.bestNEstimators = nEstimators;
            this.bestMaxDepth = maxDepth;
            this.bestMinSamplesSplit = minSamplesSplit;
            this.bestMinSamplesLeaf = minSamplesLeaf;
            this.bestMaxFeatures = maxFeatures;
            this.bestScore = score;
            this.bestModel = model;
        }
    }
    
    /**
     * 构造函数
     * @param trainFeatures 训练特征矩阵
     * @param trainLabels 训练标签
     * @param validFeatures 验证特征矩阵
     * @param validLabels 验证标签
     * @param bounds 超参数边界
     * @param randomSeed 随机种子
     * @param cvFolds 交叉验证折数
     */
    public RandomForestHyperparameterOptimizer(IMatrix trainFeatures, String[] trainLabels,
                                             IMatrix validFeatures, String[] validLabels,
                                             HyperparameterBounds bounds, long randomSeed, int cvFolds) {
        this.trainFeatures = trainFeatures;
        this.trainLabels = trainLabels;
        this.validFeatures = validFeatures;
        this.validLabels = validLabels;
        this.bounds = bounds != null ? bounds : new HyperparameterBounds();
        this.randomSeed = randomSeed;
        this.cvFolds = cvFolds;
    }
    
    /**
     * 使用Adam优化器进行超参数优化
     * @param maxIterations 最大迭代次数
     * @param learningRate 学习率
     * @return 优化结果
     */
    public OptimizationResult optimizeWithAdam(int maxIterations, double learningRate) {
        // 初始化超参数向量（归一化到[0,1]）
        IVector initialParams = Linalg.vector(new double[]{0.5, 0.5, 0.5, 0.5, 0.5});
        
        // 创建Adam优化器
        RereOnlineAdam adam = new RereOnlineAdam(learningRate, 0.9, 0.999, 1e-8, 0.0);
        
        // 初始化优化器
        adam.initialize(initialParams);
        
        IVector bestParams = initialParams.copy();
        double bestObjective = computeObjective(bestParams);
        
        // 优化循环
        for (int iter = 0; iter < maxIterations; iter++) {
            // 计算梯度（使用数值梯度）
            IVector gradient = computeNumericalGradient(bestParams);
            
            // Adam更新
            bestParams = adam.step(gradient);
            
            // 确保参数在[0,1]范围内
            bestParams = clampParameters(bestParams);
            
            // 计算新的目标函数值
            double currentObjective = computeObjective(bestParams);
            
            // 更新最佳参数（目标是最小化，所以寻找更小的值）
            if (currentObjective < bestObjective) {
                bestObjective = currentObjective;
            }
        }
        
        // 转换回实际超参数
        return convertToOptimizationResult(bestParams, bestObjective);
    }
    
    /**
     * 使用LBFGS优化器进行超参数优化
     * @param maxIterations 最大迭代次数
     * @return 优化结果
     */
    public OptimizationResult optimizeWithLBFGS(int maxIterations) {
        // 初始化超参数向量（归一化到[0,1]）
        IVector initialParams = Linalg.vector(new double[]{0.5, 0.5, 0.5, 0.5, 0.5});
        
        // 创建LBFGS优化器
        RereLBFGS lbfgs = new RereLBFGS(maxIterations, 1e-6, 10);
        
        // 定义目标函数和梯度函数
        IObjectiveFunction objFunc = this::computeObjective;
        IGradientFunction gradFunc = this::computeNumericalGradient;
        
        // 执行优化
        OptResult result = lbfgs.optimize(initialParams, objFunc, gradFunc);
        
        // 获取最优参数和目标值
        double bestObjective = result.getOptimalValue();
        IVector bestParams = clampParameters(result.getOptimalPoint());
        
        // 转换回实际超参数
        return convertToOptimizationResult(bestParams, bestObjective);
    }
    
    /**
     * 使用网格搜索进行超参数优化
     * @param gridSize 每个维度的网格大小
     * @return 优化结果
     */
    public OptimizationResult optimizeWithGridSearch(int gridSize) {
        double bestScore = Double.POSITIVE_INFINITY;
        IVector bestParams = null;
        
        // 生成网格点
        for (int i = 0; i < gridSize; i++) {
            for (int j = 0; j < gridSize; j++) {
                for (int k = 0; k < gridSize; k++) {
                    for (int l = 0; l < gridSize; l++) {
                        for (int m = 0; m < gridSize; m++) {
                            double[] params = {
                                (double) i / (gridSize - 1),
                                (double) j / (gridSize - 1),
                                (double) k / (gridSize - 1),
                                (double) l / (gridSize - 1),
                                (double) m / (gridSize - 1)
                            };
                            
                            IVector paramVector = Linalg.vector(params);
                            double score = computeObjective(paramVector);
                            
                            if (score < bestScore) {
                                bestScore = score;
                                bestParams = paramVector;
                            }
                        }
                    }
                }
            }
        }
        
        return convertToOptimizationResult(bestParams, bestScore);
    }
    
    @Override
    public double computeObjective(IVector x) {
        // 将归一化参数转换为实际超参数
        int nEstimators = (int) (bounds.minEstimators + 
                                x.get(0).doubleValue() * (bounds.maxEstimators - bounds.minEstimators));
        int maxDepth = (int) (bounds.minDepth + 
                             x.get(1).doubleValue() * (bounds.maxDepth - bounds.minDepth));
        int minSamplesSplit = (int) (bounds.minSamplesSplit + 
                                    x.get(2).doubleValue() * (bounds.maxSamplesSplit - bounds.minSamplesSplit));
        int minSamplesLeaf = (int) (bounds.minSamplesLeaf + 
                                   x.get(3).doubleValue() * (bounds.maxSamplesLeaf - bounds.minSamplesLeaf));
        int maxFeatures = (int) (trainFeatures.cols() * 
                                (bounds.minMaxFeaturesRatio + 
                                 x.get(4).doubleValue() * (bounds.maxMaxFeaturesRatio - bounds.minMaxFeaturesRatio)));
        
        // 确保参数合理性
        maxFeatures = Math.max(1, Math.min(maxFeatures, trainFeatures.cols()));
        minSamplesLeaf = Math.max(1, minSamplesLeaf);
        minSamplesSplit = Math.max(minSamplesLeaf * 2, minSamplesSplit);
        
        try {
            // 创建随机森林模型
            RereRandomForest rf = new RereRandomForest(
                nEstimators, maxDepth, minSamplesSplit, minSamplesLeaf, maxFeatures,
                true, RFTree.SplitCriterion.GINI, randomSeed
            );
            
            // 训练模型
            rf.fit(trainFeatures, trainLabels);
            
            // 在验证集上评估
            String[] predictions = rf.predictBatch(validFeatures);
            
            // 计算错误率（目标是最小化）
            int errors = 0;
            for (int i = 0; i < validLabels.length; i++) {
                if (!validLabels[i].equals(predictions[i])) {
                    errors++;
                }
            }
            
            return (double) errors / validLabels.length;
            
        } catch (Exception e) {
            // 如果参数组合导致错误，返回一个很大的值
            return 1.0;
        }
    }
    
    /**
     * 计算数值梯度
     * @param x 参数向量
     * @return 梯度向量
     */
    private IVector computeNumericalGradient(IVector x) {
        double epsilon = 1e-6;
        double[] gradient = new double[x.size()];
        double fx = computeObjective(x);
        
        for (int i = 0; i < x.size(); i++) {
            IVector xPlusEpsilon = x.copy();
            xPlusEpsilon.set(i, x.get(i).doubleValue() + epsilon);
            
            double fxPlusEpsilon = computeObjective(xPlusEpsilon);
            gradient[i] = (fxPlusEpsilon - fx) / epsilon;
        }
        
        return Linalg.vector(gradient);
    }
    
    /**
     * 将参数限制在[0,1]范围内
     * @param params 参数向量
     * @return 限制后的参数向量
     */
    private IVector clampParameters(IVector params) {
        double[] clampedParams = new double[params.size()];
        for (int i = 0; i < params.size(); i++) {
            clampedParams[i] = Math.max(0.0, Math.min(1.0, params.get(i).doubleValue()));
        }
        return Linalg.vector(clampedParams);
    }
    
    /**
     * 将归一化参数转换为优化结果
     * @param params 归一化参数向量
     * @param score 目标函数值
     * @return 优化结果
     */
    private OptimizationResult convertToOptimizationResult(IVector params, double score) {
        int nEstimators = (int) (bounds.minEstimators + 
                                params.get(0).doubleValue() * (bounds.maxEstimators - bounds.minEstimators));
        int maxDepth = (int) (bounds.minDepth + 
                             params.get(1).doubleValue() * (bounds.maxDepth - bounds.minDepth));
        int minSamplesSplit = (int) (bounds.minSamplesSplit + 
                                    params.get(2).doubleValue() * (bounds.maxSamplesSplit - bounds.minSamplesSplit));
        int minSamplesLeaf = (int) (bounds.minSamplesLeaf + 
                                   params.get(3).doubleValue() * (bounds.maxSamplesLeaf - bounds.minSamplesLeaf));
        int maxFeatures = (int) (trainFeatures.cols() * 
                                (bounds.minMaxFeaturesRatio + 
                                 params.get(4).doubleValue() * (bounds.maxMaxFeaturesRatio - bounds.minMaxFeaturesRatio)));
        
        // 确保参数合理性
        maxFeatures = Math.max(1, Math.min(maxFeatures, trainFeatures.cols()));
        minSamplesLeaf = Math.max(1, minSamplesLeaf);
        minSamplesSplit = Math.max(minSamplesLeaf * 2, minSamplesSplit);
        
        // 创建最佳模型
        RereRandomForest bestModel = new RereRandomForest(
            nEstimators, maxDepth, minSamplesSplit, minSamplesLeaf, maxFeatures,
            true, RFTree.SplitCriterion.GINI, randomSeed
        );
        
        // 训练最佳模型
        bestModel.fit(trainFeatures, trainLabels);
        
        return new OptimizationResult(nEstimators, maxDepth, minSamplesSplit, 
                                    minSamplesLeaf, maxFeatures, 1.0 - score, bestModel);
    }
}