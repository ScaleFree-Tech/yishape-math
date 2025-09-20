package com.reremouse.lab.math.stats.model;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.stats.distribution.multiv.MultivariateNormalDistribution;

import java.util.List;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicReference;

/**
 * 期望最大化(EM)算法实现
 * Expectation-Maximization (EM) Algorithm Implementation
 * 
 * 用于高斯混合模型的参数估计
 * Used for parameter estimation in Gaussian Mixture Models
 */
public class EMAlgorithm {
    
    /** 最大迭代次数 / Maximum number of iterations */
    private final int maxIterations;
    
    /** 收敛阈值 / Convergence threshold */
    private final double tolerance;
    
    /** 是否输出详细信息 / Whether to output verbose information */
    private final boolean verbose;
    
    /** 是否启用并行计算 / Whether to enable parallel computation */
    private final boolean enableParallel;
    
    /** 线程池大小 / Thread pool size */
    private final int threadPoolSize;
    
    /** 缓存的对数似然矩阵 / Cached log-likelihood matrix */
    private IMatrix<Double> cachedLogLikelihoods;
    
    /** 缓存的后验概率矩阵 / Cached posterior probability matrix */
    private IMatrix<Double> cachedPosteriors;
    
    /** 缓存是否有效 / Whether cache is valid */
    private boolean cacheValid = false;
    
    /** 向量对象池 / Vector object pool */
    private final ConcurrentLinkedQueue<IVector<Double>> vectorPool;
    
    /** 矩阵对象池 / Matrix object pool */
    private final ConcurrentLinkedQueue<IMatrix<Double>> matrixPool;
    
    /** 收敛历史记录 / Convergence history for early stopping */
    private final List<Double> convergenceHistory;
    
    /** 最小改进阈值，用于检测收敛停滞 */
    private static final double MIN_IMPROVEMENT_THRESHOLD = 1e-8;
    
    /** 随机数生成器 / Random number generator */
    private final Random random;
    
    /** 连续下降次数阈值 */
    private static final int MAX_CONSECUTIVE_DECREASES = 5; // K-means++初始化的连续下降次数
    private static final int MAX_CONSECUTIVE_DECREASES_RANDOM = 10; // 随机初始化的连续下降次数
    
    /** 数值稳定性正则化参数 */
    private static final double NUMERICAL_STABILITY_REG = 1e-3; // 进一步增强基础正则化
    private static final double ADAPTIVE_REG_FACTOR = 10.0; // 自适应正则化倍数
    private static final double MAX_REGULARIZATION = 1e-1; // 增加最大正则化强度
    private static final double DAMPING_FACTOR = 0.7; // 参数更新阻尼因子
    private static final double MIN_WEIGHT_THRESHOLD = 1e-6; // 最小权重阈值
    
    /** 改进的数值稳定性检测参数 */
    private static final double INSTABILITY_THRESHOLD_BASE = 0.08; // 基础不稳定阈值 (8%)
    private static final double INSTABILITY_THRESHOLD_ADAPTIVE = 0.20; // 自适应阈值 (20%)
    private static final int STABILITY_WINDOW_SIZE = 5; // 稳定性检测窗口大小
    private static final double MIN_ABSOLUTE_DECREASE = 8.0; // 最小绝对下降阈值
    
    /**
     * EM算法结果类
     */
    public static class EMResult {
        /** 收敛的对数似然值 / Converged log-likelihood */
        public final double logLikelihood;
        
        /** 实际迭代次数 / Actual number of iterations */
        public final int iterations;
        
        /** 是否收敛 / Whether converged */
        public final boolean converged;
        
        /** 最终的后验概率矩阵 / Final posterior probability matrix */
        public final IMatrix<Double> posteriors;
        
        public EMResult(double logLikelihood, int iterations, boolean converged, IMatrix<Double> posteriors) {
            this.logLikelihood = logLikelihood;
            this.iterations = iterations;
            this.converged = converged;
            this.posteriors = posteriors;
        }
        
        public EMResult(double logLikelihood, int iterations, boolean converged, String errorMessage) {
            this.logLikelihood = logLikelihood;
            this.iterations = iterations;
            this.converged = converged;
            this.posteriors = null; // 失败情况下没有后验概率矩阵
        }
    }
    
    /**
     * 构造函数
     * @param maxIterations 最大迭代次数
     * @param tolerance 收敛阈值
     * @param verbose 是否输出详细信息
     * @param enableParallel 是否启用并行计算
     * @param threadPoolSize 线程池大小
     */
    public EMAlgorithm(int maxIterations, double tolerance, boolean verbose, boolean enableParallel, int threadPoolSize) {
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.verbose = verbose;
        this.enableParallel = enableParallel;
        this.threadPoolSize = threadPoolSize > 0 ? threadPoolSize : Runtime.getRuntime().availableProcessors();
        this.vectorPool = new ConcurrentLinkedQueue<>();
        this.matrixPool = new ConcurrentLinkedQueue<>();
        this.convergenceHistory = new ArrayList<>();
        this.random = new Random(42);
    }
    
    /**
     * 构造函数
     * @param maxIterations 最大迭代次数
     * @param tolerance 收敛阈值
     * @param verbose 是否输出详细信息
     */
    public EMAlgorithm(int maxIterations, double tolerance, boolean verbose) {
        this(maxIterations, tolerance, verbose, true, Runtime.getRuntime().availableProcessors());
    }
    
    /**
     * 默认构造函数
     */
    public EMAlgorithm() {
        this(100, 1e-6, false, true, Runtime.getRuntime().availableProcessors());
    }
    
    /**
     * 使缓存失效
     */
    private void invalidateCache() {
        cacheValid = false;
    }
    
    /**
     * 从对象池获取向量，如果池为空则创建新的
     */
    private IVector<Double> borrowVector(int dimension) {
        IVector<Double> vector = vectorPool.poll();
        if (vector != null && vector.size() == dimension) {
            // 重置向量值以避免残留数据
            for (int i = 0; i < dimension; i++) {
                vector.set(i, 0.0);
            }
            return vector;
        } else {
            // 如果池中的向量大小不匹配，归还到池中并创建新的
            if (vector != null) {
                vectorPool.offer(vector);
            }
            return Linalg.zeros(dimension);
        }
    }
    
    /**
     * 归还向量到对象池
     */
    private void returnVector(IVector<Double> vector) {
        if (vectorPool.size() < 100) { // 限制池大小
            // 重置向量值以避免残留数据
            for (int i = 0; i < vector.size(); i++) {
                vector.set(i, 0.0);
            }
            vectorPool.offer(vector);
        }
        // 如果池已满，让对象被垃圾回收
    }
    
    /**
     * 从对象池获取矩阵，如果池为空则创建新的
     */
    private IMatrix<Double> borrowMatrix(int rows, int cols) {
        IMatrix<Double> matrix = matrixPool.poll();
        if (matrix != null && matrix.getRowNum() == rows && matrix.getColNum() == cols) {
            // 重置矩阵值以避免残留数据
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    matrix.set(i, j, 0.0);
                }
            }
            return matrix;
        } else {
            // 如果池中的矩阵大小不匹配，归还到池中并创建新的
            if (matrix != null) {
                matrixPool.offer(matrix);
            }
            return Linalg.zeros(rows, cols);
        }
    }
    
    /**
     * 归还矩阵到对象池
     */
    private void returnMatrix(IMatrix<Double> matrix) {
        if (matrixPool.size() < 50) { // 限制池大小
            // 重置矩阵值以避免残留数据
            for (int i = 0; i < matrix.getRowNum(); i++) {
                for (int j = 0; j < matrix.getColNum(); j++) {
                    matrix.set(i, j, 0.0);
                }
            }
            matrixPool.offer(matrix);
        }
        // 如果池已满，让对象被垃圾回收
    }
    
    /**
     * 批量计算对数似然矩阵（改进的并行处理）
     */
    private IMatrix<Double> computeBatchLogLikelihoods(List<IVector<Double>> data, GaussianMixtureModel gmm) {
        int numSamples = data.size();
        int numComponents = gmm.getNumComponents();
        IMatrix<Double> logLikelihoods = borrowMatrix(numSamples, numComponents);
        
        // 根据数据大小决定是否使用并行处理
        boolean useParallel = enableParallel && numSamples > 1000;
        
        if (useParallel) {
            // 并行计算
            IntStream.range(0, numSamples).parallel().forEach(i -> {
                IVector<Double> sample = data.get(i);
                for (int k = 0; k < numComponents; k++) {
                    double logLikelihood = gmm.getComponent(k).logPdf(sample);
                    logLikelihoods.set(i, k, logLikelihood);
                }
            });
        } else {
            // 串行计算（对于小数据集更高效）
            for (int i = 0; i < numSamples; i++) {
                IVector<Double> sample = data.get(i);
                for (int k = 0; k < numComponents; k++) {
                    double logLikelihood = gmm.getComponent(k).logPdf(sample);
                    logLikelihoods.set(i, k, logLikelihood);
                }
            }
        }
        
        return logLikelihoods;
    }
    
    /**
     * 检查早期收敛
     */
    private boolean checkEarlyConvergence(double currentLogLikelihood) {
        convergenceHistory.add(currentLogLikelihood);
        
        // 需要至少3个历史记录才能判断
        if (convergenceHistory.size() < 3) {
            return false;
        }
        
        // 检查最近3次迭代是否都满足收敛条件
        int size = convergenceHistory.size();
        for (int i = 1; i < 3; i++) {
            double diff = Math.abs(convergenceHistory.get(size - i) - convergenceHistory.get(size - i - 1));
            if (diff >= tolerance) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * 训练高斯混合模型（使用K-means++初始化）
     * @param data 训练数据
     * @param numComponents 分量数量
     * @param useKMeansPlusPlus 是否使用K-means++初始化
     * @return EM算法结果
     */
    public EMResult fit(List<IVector<Double>> data, int numComponents, boolean useKMeansPlusPlus) {
        if (data.isEmpty()) {
            throw new IllegalArgumentException("训练数据不能为空");
        }
        
        int dimension = data.get(0).size();
        GaussianMixtureModel gmm = new GaussianMixtureModel(numComponents, dimension);
        
        if (useKMeansPlusPlus) {
            if (verbose) {
                System.out.println("使用K-means++初始化GMM分量...");
            }
            gmm.initializeWithKMeansPlusPlus(data);
        } else {
            if (verbose) {
                System.out.println("使用随机初始化GMM分量...");
            }
            // 默认的随机初始化已在构造函数中完成
        }
        
        return fit(data, gmm);
    }
    
    /**
     * 使用多次随机初始化训练高斯混合模型，选择最佳结果
     * @param data 训练数据
     * @param numComponents 分量数量
     * @param numInitializations 初始化次数
     * @param useKMeansPlusPlus 是否使用K-means++初始化
     * @return 最佳EM算法结果
     */
    public EMResult fitWithMultipleInitializations(List<IVector<Double>> data, int numComponents, 
                                                  int numInitializations, boolean useKMeansPlusPlus) {
        if (data.isEmpty()) {
            throw new IllegalArgumentException("训练数据不能为空");
        }
        
        if (numInitializations <= 0) {
            throw new IllegalArgumentException("初始化次数必须大于0");
        }
        
        EMResult bestResult = null;
        double bestLogLikelihood = Double.NEGATIVE_INFINITY;
        GaussianMixtureModel bestGMM = null;
        
        if (verbose) {
            System.out.printf("开始多次初始化训练，初始化次数: %d\n", numInitializations);
        }
        
        for (int init = 0; init < numInitializations; init++) {
            if (verbose) {
                System.out.printf("\n=== 初始化 %d/%d ===\n", init + 1, numInitializations);
            }
            
            try {
                // 创建新的GMM实例
                int dimension = data.get(0).size();
                GaussianMixtureModel gmm = new GaussianMixtureModel(numComponents, dimension);
                
                if (useKMeansPlusPlus) {
                    gmm.initializeWithKMeansPlusPlus(data);
                } else {
                    // 使用智能随机初始化
                    gmm.initializeWithSmartRandom(data);
                }
                
                // 训练模型
                EMResult result = fit(data, gmm);
                
                if (verbose) {
                    System.out.printf("初始化 %d 结果: 对数似然=%.6f, 收敛=%s, 迭代次数=%d\n", 
                                    init + 1, result.logLikelihood, result.converged ? "是" : "否", result.iterations);
                }
                
                // 选择最佳结果（基于对数似然值）
                if (result.logLikelihood > bestLogLikelihood) {
                    bestLogLikelihood = result.logLikelihood;
                    bestResult = result;
                    bestGMM = gmm;
                    
                    if (verbose) {
                        System.out.printf("✅ 发现更好的结果，对数似然: %.6f\n", bestLogLikelihood);
                    }
                }
                
            } catch (Exception e) {
                if (verbose) {
                    System.out.printf("❌ 初始化 %d 失败: %s\n", init + 1, e.getMessage());
                }
                // 继续尝试下一个初始化
            }
        }
        
        if (bestResult == null) {
            throw new RuntimeException("所有初始化都失败了");
        }
        
        if (verbose) {
            System.out.printf("\n=== 多次初始化完成 ===\n");
            System.out.printf("最佳结果: 对数似然=%.6f, 收敛=%s, 迭代次数=%d\n", 
                            bestResult.logLikelihood, bestResult.converged ? "是" : "否", bestResult.iterations);
            
            // 计算BIC和AIC
            double bic = computeBIC(data, bestGMM);
            double aic = computeAIC(data, bestGMM);
            System.out.printf("最佳模型 BIC: %.6f, AIC: %.6f\n", bic, aic);
        }
        
        return bestResult;
    }
    
    /**
     * 使用多重启动策略的EM算法拟合（改进版本）
     * @param data 训练数据
     * @param numComponents 混合分量数量
     * @param numRestarts 重启次数
     * @param useKMeansPlusPlus 是否使用K-means++初始化
     * @return 最佳的EM算法结果
     */
    public EMResult fitWithMultipleRestarts(List<IVector<Double>> data, int numComponents, 
                                          int numRestarts, boolean useKMeansPlusPlus) {
        if (numRestarts <= 0) {
            return fit(data, numComponents, useKMeansPlusPlus);
        }
        
        EMResult bestResult = null;
        double bestLogLikelihood = Double.NEGATIVE_INFINITY;
        int successfulRuns = 0;
        
        if (verbose) {
            System.out.printf("开始多重启动策略，计划进行 %d 次重启...\n", numRestarts);
        }
        
        for (int restart = 0; restart < numRestarts; restart++) {
            try {
                // 为每次重启创建新的随机种子
                Random restartRandom = new Random(random.nextLong());
                
                // 创建新的GMM实例
                GaussianMixtureModel gmm = new GaussianMixtureModel(numComponents, 
                                                                  data.get(0).size(), 
                                                                  restartRandom);
                
                // 使用指定的初始化方法，不再使用混合策略
                boolean useKMeansForThisRestart = useKMeansPlusPlus;
                
                // 执行单次EM算法
                EMResult result = performSingleEM(data, gmm, useKMeansForThisRestart, restart);
                
                if (result != null && Double.isFinite(result.logLikelihood)) {
                    successfulRuns++;
                    
                    if (result.logLikelihood > bestLogLikelihood) {
                        bestLogLikelihood = result.logLikelihood;
                        bestResult = result;
                        
                        if (verbose) {
                            System.out.printf("重启 %d: 找到更好的解，对数似然 = %.6f\n", 
                                            restart + 1, result.logLikelihood);
                        }
                    } else if (verbose) {
                        System.out.printf("重启 %d: 对数似然 = %.6f (当前最佳: %.6f)\n", 
                                        restart + 1, result.logLikelihood, bestLogLikelihood);
                    }
                }
            } catch (Exception e) {
                if (verbose) {
                    System.out.printf("重启 %d 失败: %s\n", restart + 1, e.getMessage());
                }
            }
        }
        
        if (verbose) {
            System.out.printf("多重启动完成: %d/%d 次成功，最佳对数似然 = %.6f\n", 
                            successfulRuns, numRestarts, bestLogLikelihood);
        }
        
        return bestResult != null ? bestResult : 
               new EMResult(Double.NEGATIVE_INFINITY, 0, false, "所有重启都失败");
    }
    
    /**
     * 执行单次EM算法
     */
    private EMResult performSingleEM(List<IVector<Double>> data, GaussianMixtureModel gmm, 
                                   boolean useKMeansPlusPlus, int restartIndex) {
        try {
            // 清空收敛历史
            convergenceHistory.clear();
            
            // 初始化模型
            if (useKMeansPlusPlus) {
                gmm.initializeWithKMeansPlusPlus(data);
            } else {
                gmm.initializeRandomly(data);
            }
            
            // 执行EM算法
            return performEMIterations(data, gmm, useKMeansPlusPlus);
            
        } catch (Exception e) {
            if (verbose) {
                System.out.printf("重启 %d 执行失败: %s\n", restartIndex + 1, e.getMessage());
            }
            return null;
        }
     }
     
     /**
      * 执行EM算法迭代
      */
     private EMResult performEMIterations(List<IVector<Double>> data, GaussianMixtureModel gmm, boolean useKMeansPlusPlus) {
         int numSamples = data.size();
         int numComponents = gmm.getNumComponents();
         
         double previousLogLikelihood = Double.NEGATIVE_INFINITY;
         IMatrix<Double> posteriors = null;
         
         for (int iteration = 0; iteration < maxIterations; iteration++) {
             // E步：计算后验概率
             posteriors = eStep(data, gmm);
             
             // M步：更新模型参数
             mStep(data, posteriors, gmm);
             
             // 使缓存失效（因为模型参数已更新）
             invalidateCache();
             
             // 计算对数似然
             double currentLogLikelihood = computeLogLikelihood(data, gmm);
             
             // 检测数值不稳定性
             if (detectNumericalInstability(currentLogLikelihood, previousLogLikelihood)) {
                 return new EMResult(previousLogLikelihood, iteration + 1, false, posteriors);
             }
             
             // 检查连续下降趋势
             if (checkConsecutiveDecreases(useKMeansPlusPlus)) {
                 // 继续执行但记录警告
             }
             
             // 使用改进的收敛检查机制
             if (checkImprovedConvergence(currentLogLikelihood, previousLogLikelihood, iteration)) {
                 return new EMResult(currentLogLikelihood, iteration + 1, true, posteriors);
             }
             
             previousLogLikelihood = currentLogLikelihood;
         }
         
         return new EMResult(previousLogLikelihood, maxIterations, false, posteriors);
     }
     
     /**
      * 训练高斯混合模型
      * @param data 训练数据
      * @param gmm 初始化的高斯混合模型
      * @return EM算法结果
      */
    public EMResult fit(List<IVector<Double>> data, GaussianMixtureModel gmm) {
        if (data.isEmpty()) {
            throw new IllegalArgumentException("训练数据不能为空");
        }
        
        // 清空收敛历史和缓存
        convergenceHistory.clear();
        invalidateCache();
        
        int numSamples = data.size();
        int numComponents = gmm.getNumComponents();
        double previousLogLikelihood = Double.NEGATIVE_INFINITY;
        IMatrix<Double> posteriors = null;
        
        if (verbose) {
            System.out.println("开始EM算法训练，样本数: " + numSamples + ", 分量数: " + numComponents);
            System.out.println("并行计算: " + (enableParallel ? "启用" : "禁用") + ", 线程数: " + threadPoolSize);
        }
        
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            // E步：计算后验概率
            posteriors = eStep(data, gmm);
            
            // M步：更新模型参数
            mStep(data, posteriors, gmm);
            
            // 使缓存失效（因为模型参数已更新）
            invalidateCache();
            
            // 计算对数似然
            double currentLogLikelihood = computeLogLikelihood(data, gmm);
            
            if (verbose) {
                System.out.printf("迭代 %d/%d, 对数似然: %.6f%n", 
                                iteration + 1, maxIterations, currentLogLikelihood);
            }
            
            // 检测数值不稳定性
            if (detectNumericalInstability(currentLogLikelihood, previousLogLikelihood)) {
                if (verbose) {
                    System.out.println("检测到数值不稳定，提前终止算法");
                }
                return new EMResult(previousLogLikelihood, iteration + 1, false, posteriors);
            }
            
            // 检查连续下降趋势
            if (checkConsecutiveDecreases(true)) { // 默认使用K-means++的阈值
                if (verbose) {
                    System.out.println("检测到连续下降趋势，可能陷入局部最优");
                }
                // 继续执行但记录警告
            }
            
            // 使用改进的收敛检查机制
            if (checkImprovedConvergence(currentLogLikelihood, previousLogLikelihood, iteration)) {
                if (verbose) {
                    System.out.println("EM算法收敛（改进收敛检测），迭代次数: " + (iteration + 1));
                }
                return new EMResult(currentLogLikelihood, iteration + 1, true, posteriors);
            }
            
            previousLogLikelihood = currentLogLikelihood;
        }
        
        if (verbose) {
            System.out.println("EM算法达到最大迭代次数，未完全收敛");
        }
        
        return new EMResult(previousLogLikelihood, maxIterations, false, posteriors);
    }
    
    /**
     * E步：计算后验概率（优化版本）
     * @param data 训练数据
     * @param gmm 当前的高斯混合模型
     * @return 后验概率矩阵 [numSamples x numComponents]
     */
    private IMatrix<Double> eStep(List<IVector<Double>> data, GaussianMixtureModel gmm) {
        int numSamples = data.size();
        int numComponents = gmm.getNumComponents();
        
        // 使用缓存的对数似然矩阵或重新计算
        IMatrix<Double> logLikelihoods;
        if (cacheValid && cachedLogLikelihoods != null) {
            logLikelihoods = cachedLogLikelihoods;
        } else {
            logLikelihoods = computeBatchLogLikelihoods(data, gmm);
            cachedLogLikelihoods = logLikelihoods;
            cacheValid = true;
        }
        
        // 初始化后验概率矩阵
        IMatrix<Double> posteriors = borrowMatrix(numSamples, numComponents);
        
        // 预计算对数权重
        double[] logWeights = new double[numComponents];
        for (int k = 0; k < numComponents; k++) {
            logWeights[k] = Math.log(Math.max(gmm.getWeight(k), 1e-10)); // 避免log(0)
        }
        
        if (enableParallel) {
            // 并行计算后验概率
            IntStream.range(0, numSamples).parallel().forEach(i -> {
                computePosteriorForSample(i, logLikelihoods, logWeights, posteriors, numComponents);
            });
        } else {
            // 串行计算后验概率
            for (int i = 0; i < numSamples; i++) {
                computePosteriorForSample(i, logLikelihoods, logWeights, posteriors, numComponents);
            }
        }
        
        return posteriors;
    }
    
    /**
     * 为单个样本计算后验概率
     */
    private void computePosteriorForSample(int sampleIndex, IMatrix<Double> logLikelihoods, 
                                         double[] logWeights, IMatrix<Double> posteriors, int numComponents) {
        // 计算加权对数似然
        double[] logWeightedLikelihoods = new double[numComponents];
        double maxLogWeightedLikelihood = Double.NEGATIVE_INFINITY;
        
        for (int k = 0; k < numComponents; k++) {
            logWeightedLikelihoods[k] = logWeights[k] + logLikelihoods.get(sampleIndex, k);
            maxLogWeightedLikelihood = Math.max(maxLogWeightedLikelihood, logWeightedLikelihoods[k]);
        }
        
        // 使用log-sum-exp技巧计算归一化常数
        double sumExp = 0.0;
        for (int k = 0; k < numComponents; k++) {
            sumExp += Math.exp(logWeightedLikelihoods[k] - maxLogWeightedLikelihood);
        }
        
        if (sumExp > 0 && !Double.isInfinite(maxLogWeightedLikelihood)) {
            double logSumExp = maxLogWeightedLikelihood + Math.log(sumExp);
            for (int k = 0; k < numComponents; k++) {
                double logPosterior = logWeightedLikelihoods[k] - logSumExp;
                posteriors.set(sampleIndex, k, Math.exp(logPosterior));
            }
        } else {
            // 处理数值问题，使用均匀分布
            double uniformPosterior = 1.0 / numComponents;
            for (int k = 0; k < numComponents; k++) {
                posteriors.set(sampleIndex, k, uniformPosterior);
            }
        }
    }
    
    /**
     * M步：更新模型参数（带阻尼机制）
     * @param data 训练数据
     * @param posteriors 后验概率矩阵
     * @param gmm 要更新的高斯混合模型
     */
    private void mStep(List<IVector<Double>> data, IMatrix<Double> posteriors, GaussianMixtureModel gmm) {
        int numSamples = data.size();
        int numComponents = gmm.getNumComponents();
        int dimension = data.get(0).size();
        
        // 保存当前参数用于阻尼更新
        List<IVector<Double>> oldMeans = new ArrayList<>();
        List<IMatrix<Double>> oldCovariances = new ArrayList<>();
        List<Double> oldWeights = new ArrayList<>();
        
        for (int c = 0; c < numComponents; c++) {
            oldMeans.add(gmm.getComponent(c).getMean().copy());
            oldCovariances.add(gmm.getComponent(c).getCovariance().copy());
            oldWeights.add(gmm.getWeight(c));
        }
        
        if (enableParallel) {
            // 并行更新每个分量
            IntStream.range(0, numComponents).parallel().forEach(k -> {
                updateComponent(k, data, posteriors, gmm, numSamples, dimension);
            });
        } else {
            // 串行更新每个分量
            for (int k = 0; k < numComponents; k++) {
                updateComponent(k, data, posteriors, gmm, numSamples, dimension);
            }
        }
        
        // 应用阻尼更新
        applyDampedUpdate(gmm, oldMeans, oldCovariances, oldWeights);
        
        // 归一化权重，确保所有权重之和为1
        normalizeWeights(gmm);
    }
    
    /**
     * 更新单个分量的参数
     */
    private void updateComponent(int componentIndex, List<IVector<Double>> data, 
                               IMatrix<Double> posteriors, GaussianMixtureModel gmm, 
                               int numSamples, int dimension) {
        // 计算有效样本数（软计数）
        double effectiveSampleCount = 0.0;
        for (int i = 0; i < numSamples; i++) {
            effectiveSampleCount += posteriors.get(i, componentIndex);
        }
        
        // 防止分量塌陷：确保最小权重
        double minWeight = 1e-6; // 最小权重阈值
        if (effectiveSampleCount < minWeight * numSamples) {
            effectiveSampleCount = minWeight * numSamples;
        }
        
        // 更新权重
        double newWeight = effectiveSampleCount / numSamples;
        synchronized (gmm) { // 确保线程安全
            gmm.setWeight(componentIndex, newWeight);
        }
        
        // 更新均值 - 向量化计算
        IVector<Double> newMean = computeWeightedMean(data, posteriors, componentIndex, 
                                                    effectiveSampleCount, dimension);
        
        // 更新协方差矩阵 - 向量化计算
        IMatrix<Double> newCovariance = computeWeightedCovariance(data, posteriors, componentIndex, 
                                                                newMean, effectiveSampleCount, dimension);
        
        // 添加正则化项以确保正定性
        addRegularization(newCovariance, dimension);
        
        // 检查协方差矩阵健康状况
        if (!checkCovarianceHealth(newCovariance, dimension)) {
            if (verbose) {
                System.out.printf("警告: 分量%d的协方差矩阵不健康，使用单位矩阵替代\n", componentIndex);
            }
            // 使用单位矩阵作为备选方案
            newCovariance = createIdentityMatrix(dimension);
        }
        
        // 更新分量参数
        MultivariateNormalDistribution newComponent = new MultivariateNormalDistribution(newMean, newCovariance);
        synchronized (gmm) { // 确保线程安全
            gmm.setComponent(componentIndex, newComponent);
        }
    }
    
    /**
     * 计算加权均值（向量化）
     */
    private IVector<Double> computeWeightedMean(List<IVector<Double>> data, IMatrix<Double> posteriors, 
                                              int componentIndex, double effectiveSampleCount, int dimension) {
        IVector<Double> weightedSum = borrowVector(dimension);
        
        // 向量化累加
        for (int i = 0; i < data.size(); i++) {
            double posterior = posteriors.get(i, componentIndex);
            if (posterior > 1e-10) { // 跳过极小的权重
                IVector<Double> weightedSample = data.get(i).multiplyScalar(posterior);
                weightedSum = weightedSum.add(weightedSample);
            }
        }
        
        IVector<Double> result = weightedSum.divideByScalar(effectiveSampleCount);
        returnVector(weightedSum); // 归还到对象池
        return result;
    }
    
    /**
     * 计算加权协方差矩阵（向量化）
     */
    private IMatrix<Double> computeWeightedCovariance(List<IVector<Double>> data, IMatrix<Double> posteriors, 
                                                    int componentIndex, IVector<Double> mean, 
                                                    double effectiveSampleCount, int dimension) {
        IMatrix<Double> covarianceSum = borrowMatrix(dimension, dimension);
        
        // 向量化外积计算
        for (int i = 0; i < data.size(); i++) {
            double posterior = posteriors.get(i, componentIndex);
            if (posterior > 1e-10) { // 跳过极小的权重
                IVector<Double> diff = data.get(i).sub(mean);
                
                // 使用向量化外积（如果可用）
                IMatrix<Double> outerProduct = computeOuterProduct(diff, posterior);
                covarianceSum = covarianceSum.add(outerProduct);
                
                returnMatrix(outerProduct); // 归还到对象池
            }
        }
        
        IMatrix<Double> result = covarianceSum.divideByScalar(effectiveSampleCount);
        
        // 确保协方差矩阵对称
        ensureSymmetric(result);
        
        returnMatrix(covarianceSum); // 归还到对象池
        return result;
    }
    
    /**
     * 计算加权外积
     */
    private IMatrix<Double> computeOuterProduct(IVector<Double> vector, double weight) {
        int dimension = vector.size();
        IMatrix<Double> result = borrowMatrix(dimension, dimension);
        
        // 优化的外积计算
        for (int i = 0; i < dimension; i++) {
            double vi = vector.get(i);
            for (int j = 0; j < dimension; j++) {
                result.set(i, j, weight * vi * vector.get(j));
            }
        }
        
        return result;
    }
    
    /**
     * 增强的协方差矩阵正则化，确保数值稳定性和正定性
     */
    private void addRegularization(IMatrix<Double> covariance, int dimension) {
        // 1. 全面的矩阵健康检查
        MatrixHealthStatus health = checkMatrixHealth(covariance, dimension);
        
        // 2. 根据健康状态确定正则化策略
        double regularization = NUMERICAL_STABILITY_REG;
        
        if (health.isPathological) {
            // 病态矩阵需要强正则化
            regularization = MAX_REGULARIZATION;
            if (verbose) {
                System.out.printf("检测到病态协方差矩阵: %s\n", health.issues);
            }
        } else if (health.conditionNumber > 1e12) {
            // 严重病态矩阵，使用强正则化
            regularization = NUMERICAL_STABILITY_REG * ADAPTIVE_REG_FACTOR * 2;
            if (verbose) {
                System.out.printf("应用强正则化: 条件数 %.2e\n", health.conditionNumber);
            }
        } else if (health.conditionNumber > 1e9) {
            // 中度病态矩阵，使用中等正则化
            regularization = NUMERICAL_STABILITY_REG * ADAPTIVE_REG_FACTOR;
        } else if (health.conditionNumber > 1e6) {
            // 轻度病态矩阵，使用轻度正则化
            regularization = NUMERICAL_STABILITY_REG * 5.0;
        }
        
        // 根据最小对角元素调整正则化
        if (health.minDiagonal < regularization) {
            double adaptiveReg = Math.abs(health.minDiagonal) * 100;
            regularization = Math.max(regularization, adaptiveReg);
            regularization = Math.min(regularization, MAX_REGULARIZATION);
        }
        
        // 3. 应用自适应正则化到对角线（改进版本）
        double trace = 0.0;
        for (int d = 0; d < dimension; d++) {
            trace += covariance.get(d, d);
        }
        double averageDiagonal = trace / dimension;
        
        // 使用迹来调整正则化强度
        double traceBasedRegularization = Math.max(averageDiagonal * 1e-6, 1e-10);
        regularization = Math.max(regularization, traceBasedRegularization);
        
        for (int d = 0; d < dimension; d++) {
            double currentValue = covariance.get(d, d);
            double elementAdaptiveReg = Math.max(regularization, Math.abs(currentValue) * 1e-6);
            covariance.set(d, d, currentValue + elementAdaptiveReg);
        }
        
        // 4. 验证正则化效果
        MatrixHealthStatus newHealth = checkMatrixHealth(covariance, dimension);
        if (newHealth.isPathological) {
            // 如果仍然病态，使用单位矩阵替代
            double identityRegularization = Math.max(averageDiagonal * 0.01, 1e-3);
            for (int i = 0; i < dimension; i++) {
                for (int j = 0; j < dimension; j++) {
                    covariance.set(i, j, i == j ? identityRegularization : 0.0);
                }
            }
            if (verbose) {
                System.out.println("正则化失败，使用单位矩阵替代");
            }
        }
        
        if (verbose && regularization > NUMERICAL_STABILITY_REG) {
            System.out.printf("协方差正则化: 条件数=%.2e->%.2e, 正则化强度=%.2e\n", 
                            health.conditionNumber, newHealth.conditionNumber, regularization);
        }
        
        // 5. 确保矩阵对称性（数值误差可能破坏对称性）
        ensureSymmetric(covariance);
    }
    
    /**
     * 矩阵健康状态类
     */
    private static class MatrixHealthStatus {
        final double conditionNumber;
        final double minDiagonal;
        final double maxDiagonal;
        final boolean isPathological;
        final String issues;
        
        MatrixHealthStatus(double conditionNumber, double minDiagonal, double maxDiagonal, 
                          boolean isPathological, String issues) {
            this.conditionNumber = conditionNumber;
            this.minDiagonal = minDiagonal;
            this.maxDiagonal = maxDiagonal;
            this.isPathological = isPathological;
            this.issues = issues;
        }
    }
    
    /**
     * 全面的矩阵健康检查，包含严格的病态检测
     */
    private MatrixHealthStatus checkMatrixHealth(IMatrix<Double> covariance, int dimension) {
        double maxDiag = Double.NEGATIVE_INFINITY;
        double minDiag = Double.POSITIVE_INFINITY;
        boolean hasNaN = false;
        boolean hasInfinite = false;
        boolean hasNegativeDiagonal = false;
        int zeroCount = 0;
        
        // 检查对角线元素
        for (int d = 0; d < dimension; d++) {
            double diagValue = covariance.get(d, d);
            
            if (Double.isNaN(diagValue)) {
                hasNaN = true;
            } else if (Double.isInfinite(diagValue)) {
                hasInfinite = true;
            } else {
                maxDiag = Math.max(maxDiag, diagValue);
                minDiag = Math.min(minDiag, diagValue);
                
                if (diagValue <= 0) {
                    hasNegativeDiagonal = true;
                }
                if (Math.abs(diagValue) < 1e-15) {
                    zeroCount++;
                }
            }
        }
        
        // 检查非对角线元素的异常值
        boolean hasOffDiagonalIssues = false;
        for (int i = 0; i < dimension && !hasOffDiagonalIssues; i++) {
            for (int j = 0; j < dimension && !hasOffDiagonalIssues; j++) {
                if (i != j) {
                    double value = covariance.get(i, j);
                    if (!Double.isFinite(value)) {
                        hasOffDiagonalIssues = true;
                    }
                }
            }
        }
        
        // 计算条件数
        double conditionNumber = (minDiag > 0 && maxDiag < Double.POSITIVE_INFINITY) ? 
                                maxDiag / minDiag : Double.POSITIVE_INFINITY;
        
        // 判断是否病态
        boolean isPathological = hasNaN || hasInfinite || hasNegativeDiagonal || 
                                hasOffDiagonalIssues || conditionNumber > 1e15 || 
                                zeroCount > dimension / 2;
        
        // 构建问题描述
        StringBuilder issues = new StringBuilder();
        if (hasNaN) issues.append("NaN值; ");
        if (hasInfinite) issues.append("无穷大值; ");
        if (hasNegativeDiagonal) issues.append("负对角元素; ");
        if (hasOffDiagonalIssues) issues.append("非对角异常值; ");
        if (conditionNumber > 1e15) issues.append("极大条件数; ");
        if (zeroCount > dimension / 2) issues.append("过多零元素; ");
        
        return new MatrixHealthStatus(conditionNumber, minDiag, maxDiag, 
                                    isPathological, issues.toString());
    }
    
    /**
     * 计算协方差矩阵对角线的最小值
     */
    private double computeMinDiagonalValue(IMatrix<Double> covariance, int dimension) {
        double minValue = Double.MAX_VALUE;
        for (int d = 0; d < dimension; d++) {
            minValue = Math.min(minValue, covariance.get(d, d));
        }
        return minValue;
    }
    
    /**
     * 估算条件数（使用对角线元素的比值作为近似）
     */
    private double estimateConditionNumber(IMatrix<Double> covariance, int dimension) {
        double maxDiag = Double.NEGATIVE_INFINITY;
        double minDiag = Double.POSITIVE_INFINITY;
        
        for (int d = 0; d < dimension; d++) {
            double diagValue = covariance.get(d, d);
            maxDiag = Math.max(maxDiag, diagValue);
            minDiag = Math.min(minDiag, diagValue);
        }
        
        return minDiag > 0 ? maxDiag / minDiag : Double.POSITIVE_INFINITY;
    }
    
    /**
     * 检查协方差矩阵的数值健康状况（增强版）
     */
    private boolean checkCovarianceHealth(IMatrix<Double> covariance, int dimension) {
        MatrixHealthStatus health = checkMatrixHealth(covariance, dimension);
        
        if (health.isPathological) {
            if (verbose) {
                System.out.printf("警告: 协方差矩阵不健康: %s\n", health.issues);
            }
            return false;
        }
        
        // 额外检查：条件数阈值
        if (health.conditionNumber > 1e12) {
            if (verbose) {
                System.out.printf("警告: 协方差矩阵条件数过大: %.2e\n", health.conditionNumber);
            }
            return false;
        }
        
        // 检查最小对角元素
        if (health.minDiagonal < 1e-12) {
            if (verbose) {
                System.out.printf("警告: 协方差矩阵最小对角元素过小: %.2e\n", health.minDiagonal);
            }
            return false;
        }
        
        return true;
    }
    
    /**
     * 创建单位矩阵作为协方差矩阵的备选方案
     */
    private IMatrix<Double> createIdentityMatrix(int dimension) {
        IMatrix<Double> identity = borrowMatrix(dimension, dimension);
        
        // 初始化为零矩阵
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                identity.set(i, j, 0.0);
            }
        }
        
        // 设置对角线元素为1
        for (int d = 0; d < dimension; d++) {
            identity.set(d, d, 1.0);
        }
        
        return identity;
    }
    
    /**
     * 应用强正则化以改善条件数
     */
    private void applyStrongRegularization(IMatrix<Double> covariance, int dimension, double conditionNumber) {
        if (verbose) {
            System.out.printf("警告: 协方差矩阵条件数过大 (%.2e)，应用强正则化\n", conditionNumber);
        }
        
        // 计算矩阵的迹（对角线元素之和）
        double trace = 0.0;
        for (int d = 0; d < dimension; d++) {
            trace += covariance.get(d, d);
        }
        
        // 添加与迹成比例的正则化项
        double strongRegularization = trace / dimension * 1e-3;
        
        for (int d = 0; d < dimension; d++) {
            double currentValue = covariance.get(d, d);
            covariance.set(d, d, currentValue + strongRegularization);
        }
    }
    
    /**
     * 归一化权重，确保所有权重之和为1，并应用最小权重阈值保护
     */
    private void normalizeWeights(GaussianMixtureModel gmm) {
        int numComponents = gmm.getNumComponents();
        
        // 应用最小权重阈值保护
        double weightSum = 0.0;
        for (int k = 0; k < numComponents; k++) {
            double weight = Math.max(gmm.getWeight(k), MIN_WEIGHT_THRESHOLD);
            gmm.setWeight(k, weight);
            weightSum += weight;
        }
        
        // 归一化权重
        if (weightSum > 0) {
            for (int k = 0; k < numComponents; k++) {
                double normalizedWeight = gmm.getWeight(k) / weightSum;
                gmm.setWeight(k, normalizedWeight);
            }
        } else {
            // 如果所有权重都为0，使用均匀分布
            double uniformWeight = 1.0 / numComponents;
            for (int k = 0; k < numComponents; k++) {
                gmm.setWeight(k, uniformWeight);
            }
        }
        
        // 最终验证权重有效性
        for (int k = 0; k < numComponents; k++) {
            double weight = gmm.getWeight(k);
            if (Double.isNaN(weight) || Double.isInfinite(weight) || weight <= 0) {
                // 重置为均匀分布
                double uniformWeight = 1.0 / numComponents;
                for (int c = 0; c < numComponents; c++) {
                    gmm.setWeight(c, uniformWeight);
                }
                break;
            }
        }
    }
    
    /**
     * 确保矩阵对称
     */
    private void ensureSymmetric(IMatrix<Double> matrix) {
        int rows = matrix.getRowNum();
        int cols = matrix.getColNum();
        
        // 对于每个上三角元素，确保下三角元素相同
        for (int i = 0; i < rows; i++) {
            for (int j = i + 1; j < cols; j++) {
                double value = (matrix.get(i, j) + matrix.get(j, i)) / 2.0;
                matrix.set(i, j, value);
                matrix.set(j, i, value);
            }
        }
    }
    
    /**
     * 计算对数似然（优化版本）
     * @param data 训练数据
     * @param gmm 高斯混合模型
     * @return 对数似然值
     */
    private double computeLogLikelihood(List<IVector<Double>> data, GaussianMixtureModel gmm) {
        // 使用缓存的对数似然矩阵
        IMatrix<Double> logLikelihoods;
        if (cacheValid && cachedLogLikelihoods != null) {
            logLikelihoods = cachedLogLikelihoods;
        } else {
            logLikelihoods = computeBatchLogLikelihoods(data, gmm);
            cachedLogLikelihoods = logLikelihoods;
            cacheValid = true;
        }
        
        int numSamples = data.size();
        int numComponents = gmm.getNumComponents();
        
        // 预计算对数权重
        double[] logWeights = new double[numComponents];
        for (int k = 0; k < numComponents; k++) {
            logWeights[k] = Math.log(Math.max(gmm.getWeight(k), 1e-10));
        }
        
        // 并行或串行计算总对数似然
        if (enableParallel) {
            return IntStream.range(0, numSamples)
                    .parallel()
                    .mapToDouble(i -> computeSampleLogLikelihood(i, logLikelihoods, logWeights, numComponents))
                    .sum();
        } else {
            double totalLogLikelihood = 0.0;
            for (int i = 0; i < numSamples; i++) {
                totalLogLikelihood += computeSampleLogLikelihood(i, logLikelihoods, logWeights, numComponents);
            }
            return totalLogLikelihood;
        }
    }
    
    /**
     * 计算单个样本的对数似然
     */
    private double computeSampleLogLikelihood(int sampleIndex, IMatrix<Double> logLikelihoods, 
                                            double[] logWeights, int numComponents) {
        // 计算加权对数似然
        double[] logWeightedLikelihoods = new double[numComponents];
        double maxLogWeightedLikelihood = Double.NEGATIVE_INFINITY;
        
        for (int k = 0; k < numComponents; k++) {
            logWeightedLikelihoods[k] = logWeights[k] + logLikelihoods.get(sampleIndex, k);
            maxLogWeightedLikelihood = Math.max(maxLogWeightedLikelihood, logWeightedLikelihoods[k]);
        }
        
        // 使用log-sum-exp技巧
        double sumExp = 0.0;
        for (int k = 0; k < numComponents; k++) {
            sumExp += Math.exp(logWeightedLikelihoods[k] - maxLogWeightedLikelihood);
        }
        
        if (sumExp > 0 && !Double.isInfinite(maxLogWeightedLikelihood)) {
            return maxLogWeightedLikelihood + Math.log(sumExp);
        } else {
            // 处理数值问题，使用一个合理的下界
            return -700; // 接近Double的最小值的对数
        }
    }
    
    /**
     * 计算贝叶斯信息准则(BIC)
     * @param data 训练数据
     * @param gmm 高斯混合模型
     * @return BIC值
     */
    public double computeBIC(List<IVector<Double>> data, GaussianMixtureModel gmm) {
        double logLikelihood = computeLogLikelihood(data, gmm);
        int numSamples = data.size();
        int dimension = data.get(0).size();
        int numComponents = gmm.getNumComponents();
        
        // 参数数量：每个分量有 dimension 个均值参数 + dimension*(dimension+1)/2 个协方差参数 + 1 个权重参数
        int numParameters = numComponents * (dimension + dimension * (dimension + 1) / 2 + 1) - 1; // 减1因为权重和为1
        
        return -2 * logLikelihood + numParameters * Math.log(numSamples);
    }
    
    /**
     * 计算阿卡信息准则(AIC)
     * @param data 训练数据
     * @param gmm 高斯混合模型
     * @return AIC值
     */
    public double computeAIC(List<IVector<Double>> data, GaussianMixtureModel gmm) {
        double logLikelihood = computeLogLikelihood(data, gmm);
        int dimension = data.get(0).size();
        int numComponents = gmm.getNumComponents();
        
        int numParameters = numComponents * (dimension + dimension * (dimension + 1) / 2 + 1) - 1;
        
        return -2 * logLikelihood + 2 * numParameters;
    }
    
    /**
     * 改进的收敛检查机制
     * 包括相对变化检查、数值稳定性检查和趋势分析
     */
    private boolean checkImprovedConvergence(double currentLogLikelihood, double previousLogLikelihood, int iteration) {
        convergenceHistory.add(currentLogLikelihood);
        
        // 基本的相对变化检查
        if (previousLogLikelihood != Double.NEGATIVE_INFINITY) {
            double relativeChange = Math.abs(currentLogLikelihood - previousLogLikelihood) / 
                                  Math.max(Math.abs(previousLogLikelihood), 1e-10);
            
            // 使用自适应容忍度 - 早期迭代使用更严格的标准
            double adaptiveTolerance = tolerance;
            if (iteration < 10) {
                adaptiveTolerance = tolerance * 0.1; // 早期迭代更严格
            } else if (iteration > 50) {
                adaptiveTolerance = tolerance * 2.0; // 后期迭代更宽松
            }
            
            // 添加绝对变化检查以处理对数似然值非常大的情况
            double absoluteChange = Math.abs(currentLogLikelihood - previousLogLikelihood);
            
            if (relativeChange < adaptiveTolerance || absoluteChange < tolerance) {
                if (verbose) {
                    System.out.printf("收敛检测: 相对变化 %.2e < 自适应阈值 %.2e 或绝对变化 %.2e < 阈值 %.2e (迭代%d)\n", 
                                    relativeChange, adaptiveTolerance, absoluteChange, tolerance, iteration);
                }
                return true;
            }
        }
        
        // 检查是否有足够的历史记录进行趋势分析
        if (convergenceHistory.size() < 5) {
            return false;
        }
        
        // 多重收敛检查策略
        boolean trendConverged = checkConvergenceTrend();
        boolean stagnationDetected = checkStagnation();
        boolean plateauDetected = checkPlateau();
        
        // 如果多个指标都表明收敛，则认为已收敛
        int convergenceIndicators = 0;
        if (trendConverged) convergenceIndicators++;
        if (stagnationDetected) convergenceIndicators++;
        if (plateauDetected) convergenceIndicators++;
        
        if (convergenceIndicators >= 2) {
            if (verbose) {
                System.out.printf("收敛检测: 多重指标收敛 (趋势:%s, 停滞:%s, 平台:%s)\n", 
                                trendConverged, stagnationDetected, plateauDetected);
            }
            return true;
        }
        
        return false;
    }
    
    /**
     * 检查收敛趋势
     */
    private boolean checkConvergenceTrend() {
        int size = convergenceHistory.size();
        if (size < 5) return false;
        
        // 检查最近5次迭代的相对变化是否都很小
        boolean allSmallChanges = true;
        for (int i = size - 4; i < size; i++) {
            double relativeChange = Math.abs(convergenceHistory.get(i) - convergenceHistory.get(i - 1)) / 
                                  Math.max(Math.abs(convergenceHistory.get(i - 1)), 1e-10);
            if (relativeChange >= tolerance * 10) { // 使用更宽松的阈值
                allSmallChanges = false;
                break;
            }
        }
        
        if (allSmallChanges && verbose) {
            System.out.println("收敛检测: 最近5次迭代变化都很小");
        }
        
        return allSmallChanges;
    }
    
    /**
     * 检查算法是否停滞
     */
    private boolean checkStagnation() {
        int size = convergenceHistory.size();
        if (size < 10) return false;
        
        // 检查最近10次迭代的最大改进
        double maxImprovement = 0.0;
        for (int i = size - 9; i < size; i++) {
            double improvement = convergenceHistory.get(i) - convergenceHistory.get(i - 1);
            maxImprovement = Math.max(maxImprovement, improvement);
        }
        
        boolean isStagnant = maxImprovement < MIN_IMPROVEMENT_THRESHOLD;
        if (isStagnant && verbose) {
            System.out.printf("收敛检测: 算法停滞，最近10次迭代最大改进 %.2e\n", maxImprovement);
        }
        
        return isStagnant;
    }
    
    /**
     * 检查是否达到平台期
     * 平台期是指对数似然值在一个小范围内波动，但总体趋势平稳
     */
    private boolean checkPlateau() {
        int size = convergenceHistory.size();
        if (size < 15) return false;
        
        // 计算最近15次迭代的统计信息
        double sum = 0.0;
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        
        for (int i = size - 15; i < size; i++) {
            double value = convergenceHistory.get(i);
            sum += value;
            min = Math.min(min, value);
            max = Math.max(max, value);
        }
        
        double mean = sum / 15.0;
        double range = max - min;
        double relativeRange = Math.abs(mean) > 1e-10 ? range / Math.abs(mean) : range;
        
        // 计算方差
        double variance = 0.0;
        for (int i = size - 15; i < size; i++) {
            double diff = convergenceHistory.get(i) - mean;
            variance += diff * diff;
        }
        variance /= 14.0; // 样本方差
        double stddev = Math.sqrt(variance);
        double relativeStddev = Math.abs(mean) > 1e-10 ? stddev / Math.abs(mean) : stddev;
        
        // 平台期判断条件：
        // 1. 相对范围小于容忍度的10倍
        // 2. 相对标准差小于容忍度的5倍
        // 3. 最近5次迭代没有显著趋势
        boolean smallRange = relativeRange < tolerance * 10;
        boolean smallVariance = relativeStddev < tolerance * 5;
        boolean noTrend = checkNoSignificantTrend();
        
        boolean isPlateau = smallRange && smallVariance && noTrend;
        
        if (isPlateau && verbose) {
            System.out.printf("收敛检测: 检测到平台期，相对范围=%.2e, 相对标准差=%.2e\n", 
                            relativeRange, relativeStddev);
        }
        
        return isPlateau;
    }
    
    /**
     * 检查最近几次迭代是否没有显著趋势
     */
    private boolean checkNoSignificantTrend() {
        int size = convergenceHistory.size();
        if (size < 8) return false;
        
        // 使用简单的线性回归检查趋势
        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        int n = 8;
        
        for (int i = 0; i < n; i++) {
            double x = i;
            double y = convergenceHistory.get(size - n + i);
            sumX += x;
            sumY += y;
            sumXY += x * y;
            sumX2 += x * x;
        }
        
        // 计算斜率
        double slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        double meanY = sumY / n;
        
        // 相对斜率
        double relativeSlope = Math.abs(meanY) > 1e-10 ? Math.abs(slope) / Math.abs(meanY) : Math.abs(slope);
        
        // 如果相对斜率很小，认为没有显著趋势
        return relativeSlope < tolerance * 2;
    }
    
    /**
     * 检测数值不稳定性 - 改进的自适应检测
     */
    private boolean detectNumericalInstability(double currentLogLikelihood, double previousLogLikelihood) {
        // 检查对数似然值是否为无穷大或NaN
        if (!Double.isFinite(currentLogLikelihood)) {
            if (verbose) {
                System.out.println("警告: 对数似然值为无穷大或NaN");
            }
            return true;
        }
        
        // 检查对数似然值是否过小（可能的数值下溢）
        if (currentLogLikelihood < -1e10) {
            if (verbose) {
                System.out.printf("警告: 对数似然值过小 %.6e，可能存在数值下溢\n", currentLogLikelihood);
            }
            return true;
        }
        
        // 检查对数似然值是否出现异常下降
        if (previousLogLikelihood != Double.NEGATIVE_INFINITY && Double.isFinite(previousLogLikelihood)) {
            double decrease = previousLogLikelihood - currentLogLikelihood;
            double relativeDecrease = Math.abs(previousLogLikelihood) > 1e-10 ? 
                                    decrease / Math.abs(previousLogLikelihood) : decrease;
            
            // 更宽松的自适应阈值
            double threshold;
            if (Math.abs(previousLogLikelihood) > 1000) {
                threshold = 1.0; // 大对数似然值，使用更大阈值
            } else if (Math.abs(previousLogLikelihood) > 100) {
                threshold = 0.8; // 中等对数似然值
            } else {
                threshold = 0.5; // 小对数似然值
            }
            
            // 如果收敛历史显示算法已经稳定运行了一段时间，显著放宽阈值
            if (convergenceHistory.size() > 10) {
                threshold *= 2.0; // 放宽100%
            } else if (convergenceHistory.size() > 5) {
                threshold *= 1.5; // 放宽50%
            }
            
            // 检查连续大幅下降
            boolean significantAbsoluteDecrease = decrease > MIN_ABSOLUTE_DECREASE;
            boolean significantRelativeDecrease = relativeDecrease > threshold;
            
            // 检查是否有连续的严重异常下降
            boolean hasConsecutiveAbnormalDecreases = false;
            if (convergenceHistory.size() >= 5) {
                int abnormalCount = 0;
                for (int i = convergenceHistory.size() - 5; i < convergenceHistory.size() - 1; i++) {
                    double prevDecrease = convergenceHistory.get(i) - convergenceHistory.get(i + 1);
                    // 使用更严格的阈值，只有真正严重的下降才算异常
                    if (prevDecrease > MIN_ABSOLUTE_DECREASE * 2.0) {
                        abnormalCount++;
                    }
                }
                hasConsecutiveAbnormalDecreases = abnormalCount >= 3; // 需要至少3次严重下降
            }
            
            if ((significantAbsoluteDecrease && significantRelativeDecrease) || hasConsecutiveAbnormalDecreases) {
                if (verbose) {
                    System.out.printf("警告: 检测到数值不稳定，对数似然下降 %.6f (相对: %.2f%%, 阈值: %.2f%%)\n", 
                                    decrease, relativeDecrease * 100, threshold * 100);
                }
                return true;
            }
        }
        
        return false;
    }
    
    /**
     * 检查连续下降趋势
     */
    private boolean checkConsecutiveDecreases(boolean useKMeansPlusPlus) {
        int size = convergenceHistory.size();
        int maxDecreases = useKMeansPlusPlus ? MAX_CONSECUTIVE_DECREASES : MAX_CONSECUTIVE_DECREASES_RANDOM;
        
        if (size < maxDecreases + 1) return false;
        
        int consecutiveDecreases = 0;
        for (int i = size - maxDecreases; i < size; i++) {
            if (convergenceHistory.get(i) < convergenceHistory.get(i - 1)) {
                consecutiveDecreases++;
            } else {
                break;
            }
        }
        
        boolean hasConsecutiveDecreases = consecutiveDecreases >= maxDecreases;
        if (hasConsecutiveDecreases && verbose) {
            System.out.printf("警告: 检测到连续%d次下降\n", consecutiveDecreases);
        }
        
        return hasConsecutiveDecreases;
    }
    
    /**
     * 应用阻尼更新机制，防止参数变化过大
     */
    private void applyDampedUpdate(GaussianMixtureModel gmm, List<IVector<Double>> oldMeans, 
                                 List<IMatrix<Double>> oldCovariances, List<Double> oldWeights) {
        int numComponents = gmm.getNumComponents();
        
        for (int c = 0; c < numComponents; c++) {
            // 阻尼更新均值
            IVector<Double> currentMean = gmm.getComponent(c).getMean();
            IVector<Double> dampedMean = oldMeans.get(c).multiplyScalar(1.0 - DAMPING_FACTOR)
                                                      .add(currentMean.multiplyScalar(DAMPING_FACTOR));
            
            // 阻尼更新协方差矩阵
            IMatrix<Double> currentCov = gmm.getComponent(c).getCovariance();
            IMatrix<Double> dampedCov = oldCovariances.get(c).multiplyScalar(1.0 - DAMPING_FACTOR)
                                                            .add(currentCov.multiplyScalar(DAMPING_FACTOR));
            
            // 阻尼更新权重
            double currentWeight = gmm.getWeight(c);
            double dampedWeight = oldWeights.get(c) * (1.0 - DAMPING_FACTOR) + currentWeight * DAMPING_FACTOR;
            
            // 应用阻尼后的参数
            MultivariateNormalDistribution dampedComponent = new MultivariateNormalDistribution(dampedMean, dampedCov);
            gmm.setComponent(c, dampedComponent);
            gmm.setWeight(c, dampedWeight);
        }
    }
    
    // Getters
    public int getMaxIterations() {
        return maxIterations;
    }
    
    public double getTolerance() {
        return tolerance;
    }
    
    public boolean isVerbose() {
        return verbose;
    }
}