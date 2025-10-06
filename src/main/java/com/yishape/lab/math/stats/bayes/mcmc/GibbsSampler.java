package com.yishape.lab.math.stats.bayes.mcmc;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;

import java.util.Random;
import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;

/**
 * Gibbs采样器
 * Gibbs Sampler
 * 
 * <p>实现了Gibbs采样算法，通过逐个更新每个参数的条件分布来进行采样。
 * 适用于已知条件分布的多变量概率分布采样。</p>
 * <p>Implements the Gibbs sampling algorithm, which samples by updating each 
 * parameter's conditional distribution one at a time. Suitable for sampling 
 * from multivariate probability distributions with known conditional distributions.</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class GibbsSampler implements IMCMCSampler {
    
    private Random random;
    private int blockSize;
    private boolean useRandomScan;
    
    /**
     * 条件分布采样器接口
     * Conditional distribution sampler interface
     */
    @FunctionalInterface
    public interface ConditionalSampler {
        /**
         * 从条件分布中采样
         * Sample from conditional distribution
         * 
         * @param currentState 当前状态 / Current state
         * @param parameterIndex 要更新的参数索引 / Parameter index to update
         * @return 新的参数值 / New parameter value
         */
        double sampleConditional(IVector currentState, int parameterIndex);
    }
    
    /**
     * 多变量条件分布采样器接口
     * Multivariate conditional distribution sampler interface
     */
    @FunctionalInterface
    public interface BlockConditionalSampler {
        /**
         * 从多变量条件分布中采样
         * Sample from multivariate conditional distribution
         * 
         * @param currentState 当前状态 / Current state
         * @param parameterIndices 要更新的参数索引数组 / Parameter indices to update
         * @return 新的参数值向量 / New parameter values vector
         */
        IVector sampleBlockConditional(IVector currentState, int[] parameterIndices);
    }
    
    private List<ConditionalSampler> conditionalSamplers;
    private List<BlockConditionalSampler> blockSamplers;
    private List<int[]> blockIndices;
    
    /**
     * 默认构造函数
     * Default constructor
     */
    public GibbsSampler() {
        this(new Random(), 1, false);
    }
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param random 随机数生成器 / Random number generator
     * @param blockSize 块大小（用于块Gibbs采样）/ Block size (for block Gibbs sampling)
     * @param useRandomScan 是否使用随机扫描 / Whether to use random scan
     */
    public GibbsSampler(Random random, int blockSize, boolean useRandomScan) {
        this.random = random;
        this.blockSize = blockSize;
        this.useRandomScan = useRandomScan;
        this.conditionalSamplers = new ArrayList<>();
        this.blockSamplers = new ArrayList<>();
        this.blockIndices = new ArrayList<>();
    }
    
    /**
     * 添加条件分布采样器
     * Add conditional distribution sampler
     */
    public void addConditionalSampler(ConditionalSampler sampler) {
        conditionalSamplers.add(sampler);
    }
    
    /**
     * 添加块条件分布采样器
     * Add block conditional distribution sampler
     */
    public void addBlockConditionalSampler(BlockConditionalSampler sampler, int[] indices) {
        blockSamplers.add(sampler);
        blockIndices.add(indices.clone());
    }
    
    /**
     * 设置条件分布采样器列表
     * Set conditional distribution samplers list
     */
    public void setConditionalSamplers(List<ConditionalSampler> samplers) {
        this.conditionalSamplers = new ArrayList<>(samplers);
    }
    
    @Override
    public SamplingResult sample(TargetDistribution targetDistribution, 
                               IVector initialState, 
                               int numSamples, 
                               int burnIn) {
        if (conditionalSamplers.isEmpty() && blockSamplers.isEmpty()) {
            throw new IllegalStateException("No conditional samplers provided. " +
                "Gibbs sampling requires conditional distribution samplers.");
        }
        
        int totalSamples = numSamples + burnIn;
        int dimension = initialState.size();
        
        // 存储样本和对数概率
        List<IVector> samplesList = new ArrayList<>();
        List<Double> logProbsList = new ArrayList<>();
        
        IVector currentState = initialState.copy();
        double currentLogProb = targetDistribution.logPdf(currentState);
        
        for (int i = 0; i < totalSamples; i++) {
            // 执行一次Gibbs扫描
            currentState = performGibbsScan(currentState);
            currentLogProb = targetDistribution.logPdf(currentState);
            
            // 跳过预热期
            if (i >= burnIn) {
                samplesList.add(currentState.copy());
                logProbsList.add(currentLogProb);
            }
        }
        
        // 转换为矩阵格式
        double[][] samplesData = new double[numSamples][dimension];
        double[] logProbs = new double[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            IVector sample = samplesList.get(i);
            for (int j = 0; j < dimension; j++) {
                samplesData[i][j] = sample.get(j).doubleValue();
            }
            logProbs[i] = logProbsList.get(i);
        }
        
        IMatrix samples = Linalg.matrix(samplesData);
        
        // 计算统计信息
        double acceptanceRate = 1.0; // Gibbs采样总是接受
        int effectiveSampleSize = calculateEffectiveSampleSize(samples);
        boolean converged = checkConvergence(samples, logProbs);
        
        return new SamplingResult(samples, logProbs, acceptanceRate, 
                                effectiveSampleSize, converged);
    }
    
    @Override
    public SamplingResult sample(TargetDistribution targetDistribution,
                               ProposalDistribution proposalDistribution,
                               IVector initialState,
                               int numSamples,
                               int burnIn) {
        // Gibbs采样不使用提议分布，直接调用主要的采样方法
        return sample(targetDistribution, initialState, numSamples, burnIn);
    }
    
    /**
     * 执行一次Gibbs扫描
     * Perform one Gibbs scan
     */
    private IVector performGibbsScan(IVector currentState) {
        IVector newState = currentState.copy();
        
        if (!blockSamplers.isEmpty()) {
            // 块Gibbs采样
            performBlockGibbsScan(newState);
        } else {
            // 标准Gibbs采样
            performStandardGibbsScan(newState);
        }
        
        return newState;
    }
    
    /**
     * 执行标准Gibbs扫描
     * Perform standard Gibbs scan
     */
    private void performStandardGibbsScan(IVector state) {
        int dimension = state.size();
        
        if (useRandomScan) {
            // 随机扫描：随机选择参数更新顺序
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < dimension; i++) {
                indices.add(i);
            }
            java.util.Collections.shuffle(indices, random);
            
            for (int idx : indices) {
                if (idx < conditionalSamplers.size()) {
                    double newValue = conditionalSamplers.get(idx).sampleConditional(state, idx);
                    state.set(idx, newValue);
                }
            }
        } else {
            // 系统扫描：按顺序更新参数
            for (int i = 0; i < Math.min(dimension, conditionalSamplers.size()); i++) {
                double newValue = conditionalSamplers.get(i).sampleConditional(state, i);
                state.set(i, newValue);
            }
        }
    }
    
    /**
     * 执行块Gibbs扫描
     * Perform block Gibbs scan
     */
    private void performBlockGibbsScan(IVector state) {
        for (int i = 0; i < blockSamplers.size(); i++) {
            BlockConditionalSampler sampler = blockSamplers.get(i);
            int[] indices = blockIndices.get(i);
            
            IVector newValues = sampler.sampleBlockConditional(state, indices);
            
            // 更新对应的参数
            for (int j = 0; j < indices.length && j < newValues.size(); j++) {
                state.set(indices[j], newValues.get(j));
            }
        }
    }
    
    /**
     * 计算有效样本大小
     * Calculate effective sample size
     */
    private int calculateEffectiveSampleSize(IMatrix samples) {
        int numSamples = samples.rows();
        int dimension = samples.cols();
        
        double minESS = Double.MAX_VALUE;
        
        for (int d = 0; d < dimension; d++) {
            IVector paramSamples = samples.getColumn(d);
            double[] autocorr = calculateAutocorrelation(paramSamples);
            
            // 计算积分自相关时间
            double integratedTime = 1.0;
            for (int lag = 1; lag < autocorr.length && autocorr[lag] > 0.05; lag++) {
                integratedTime += 2 * autocorr[lag];
            }
            
            double ess = numSamples / integratedTime;
            minESS = Math.min(minESS, ess);
        }
        
        return (int) Math.max(1, minESS);
    }
    
    /**
     * 计算自相关函数
     * Calculate autocorrelation function
     */
    private double[] calculateAutocorrelation(IVector samples) {
        int n = samples.size();
        int maxLag = Math.min(n / 4, 100);
        double[] autocorr = new double[maxLag];
        
        double mean = samples.mean().doubleValue();
        double variance = 0.0;
        
        // 计算方差
        for (int i = 0; i < n; i++) {
            double diff = samples.get(i).doubleValue() - mean;
            variance += diff * diff;
        }
        variance /= (n - 1);
        
        // 计算自相关
        for (int lag = 0; lag < maxLag; lag++) {
            double covariance = 0.0;
            int count = 0;
            
            for (int i = 0; i < n - lag; i++) {
                covariance += (samples.get(i).doubleValue() - mean) * 
                             (samples.get(i + lag).doubleValue() - mean);
                count++;
            }
            
            if (count > 0) {
                covariance /= count;
                autocorr[lag] = covariance / variance;
            }
        }
        
        return autocorr;
    }
    
    /**
     * 检查收敛性
     * Check convergence
     */
    private boolean checkConvergence(IMatrix samples, double[] logProbs) {
        int numSamples = samples.rows();
        if (numSamples < 100) return false;
        
        // 检查对数概率的稳定性
        int windowSize = Math.min(numSamples / 4, 500);
        double[] recentLogProbs = new double[windowSize];
        System.arraycopy(logProbs, numSamples - windowSize, recentLogProbs, 0, windowSize);
        
        double mean = 0.0;
        for (double logProb : recentLogProbs) {
            mean += logProb;
        }
        mean /= windowSize;
        
        double variance = 0.0;
        for (double logProb : recentLogProbs) {
            variance += (logProb - mean) * (logProb - mean);
        }
        variance /= (windowSize - 1);
        
        double cv = Math.sqrt(variance) / Math.abs(mean);
        return cv < 0.1;
    }
    
    @Override
    public void setParameters(double stepSize, int adaptationPeriod) {
        // Gibbs采样不使用步长，但可以设置块大小
        this.blockSize = (int) stepSize;
    }
    
    /**
     * 设置是否使用随机扫描
     * Set whether to use random scan
     */
    public void setUseRandomScan(boolean useRandomScan) {
        this.useRandomScan = useRandomScan;
    }
    
    /**
     * 设置块大小
     * Set block size
     */
    public void setBlockSize(int blockSize) {
        this.blockSize = blockSize;
    }
    
    /**
     * 设置随机数生成器
     * Set random number generator
     */
    public void setRandom(Random random) {
        this.random = random;
    }
    
    @Override
    public String getSamplerName() {
        return "Gibbs";
    }
    
    @Override
    public boolean supportsAdaptation() {
        return false; // Gibbs采样通常不需要自适应
    }
    
    /**
     * 获取条件分布采样器数量
     * Get number of conditional samplers
     */
    public int getNumConditionalSamplers() {
        return conditionalSamplers.size();
    }
    
    /**
     * 获取块采样器数量
     * Get number of block samplers
     */
    public int getNumBlockSamplers() {
        return blockSamplers.size();
    }
}