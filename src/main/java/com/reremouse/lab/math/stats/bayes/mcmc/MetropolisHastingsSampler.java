package com.reremouse.lab.math.stats.bayes.mcmc;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.stats.distribution.NormalDistribution;
import java.util.Random;
import java.util.ArrayList;
import java.util.List;

/**
 * Metropolis-Hastings采样器
 * Metropolis-Hastings Sampler
 * 
 * <p>实现了经典的Metropolis-Hastings算法，用于从任意概率分布中采样。
 * 支持自适应步长调整和多种提议分布。</p>
 * <p>Implements the classic Metropolis-Hastings algorithm for sampling from 
 * arbitrary probability distributions. Supports adaptive step size tuning 
 * and various proposal distributions.</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class MetropolisHastingsSampler implements IMCMCSampler {
    
    private double stepSize;
    private int adaptationPeriod;
    private double targetAcceptanceRate;
    private Random random;
    private boolean adaptiveStepSize;
    
    /**
     * 默认构造函数
     * Default constructor
     */
    public MetropolisHastingsSampler() {
        this(0.5, 1000, 0.44, new Random());
    }
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param stepSize 初始步长 / Initial step size
     * @param adaptationPeriod 自适应期 / Adaptation period
     * @param targetAcceptanceRate 目标接受率 / Target acceptance rate
     * @param random 随机数生成器 / Random number generator
     */
    public MetropolisHastingsSampler(double stepSize, int adaptationPeriod, 
                                   double targetAcceptanceRate, Random random) {
        this.stepSize = stepSize;
        this.adaptationPeriod = adaptationPeriod;
        this.targetAcceptanceRate = targetAcceptanceRate;
        this.random = random;
        this.adaptiveStepSize = true;
    }
    
    @Override
    public SamplingResult sample(TargetDistribution targetDistribution, 
                               IVector initialState, 
                               int numSamples, 
                               int burnIn) {
        // 使用默认的多元正态提议分布
        ProposalDistribution defaultProposal = currentState -> {
            double[] proposalData = new double[currentState.size()];
            for (int i = 0; i < currentState.size(); i++) {
                proposalData[i] = currentState.get(i).doubleValue() + 
                                new NormalDistribution(0, stepSize).sample(1)[0];
            }
            return Linalg.vector(proposalData);
        };
        
        return sample(targetDistribution, defaultProposal, initialState, numSamples, burnIn);
    }
    
    @Override
    public SamplingResult sample(TargetDistribution targetDistribution,
                               ProposalDistribution proposalDistribution,
                               IVector initialState,
                               int numSamples,
                               int burnIn) {
        
        int totalSamples = numSamples + burnIn;
        int dimension = initialState.size();
        
        // 存储样本和对数概率
        List<IVector> samplesList = new ArrayList<>();
        List<Double> logProbsList = new ArrayList<>();
        
        IVector currentState = initialState.copy();
        double currentLogProb = targetDistribution.logPdf(currentState);
        
        int acceptedCount = 0;
        double currentStepSize = stepSize;
        
        for (int i = 0; i < totalSamples; i++) {
            // 生成提议状态
            IVector proposedState = proposalDistribution.propose(currentState);
            double proposedLogProb = targetDistribution.logPdf(proposedState);
            
            // 计算接受概率
            double logAcceptanceProb = Math.min(0, proposedLogProb - currentLogProb);
            double acceptanceProb = Math.exp(logAcceptanceProb);
            
            // 决定是否接受
            if (random.nextDouble() < acceptanceProb) {
                currentState = proposedState;
                currentLogProb = proposedLogProb;
                acceptedCount++;
            }
            
            // 自适应步长调整
            if (adaptiveStepSize && i < adaptationPeriod && i > 0 && i % 50 == 0) {
                double currentAcceptanceRate = (double) acceptedCount / (i + 1);
                if (currentAcceptanceRate > targetAcceptanceRate) {
                    currentStepSize *= 1.1;
                } else {
                    currentStepSize *= 0.9;
                }
                stepSize = currentStepSize;
            }
            
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
        double acceptanceRate = (double) acceptedCount / totalSamples;
        int effectiveSampleSize = calculateEffectiveSampleSize(samples);
        boolean converged = checkConvergence(samples, logProbs);
        
        return new SamplingResult(samples, logProbs, acceptanceRate, 
                                effectiveSampleSize, converged);
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
        int maxLag = Math.min(n / 4, 100); // 限制最大滞后
        double[] autocorr = new double[maxLag];
        
        double mean = (double)samples.mean();
        double variance = 0.0;
        
        // 计算方差
        for (int i = 0; i < n; i++) {
            double diff = (double)samples.get(i) - mean;
            variance += diff * diff;
        }
        variance /= (n - 1);
        
        // 计算自相关
        for (int lag = 0; lag < maxLag; lag++) {
            double covariance = 0.0;
            int count = 0;
            
            for (int i = 0; i < n - lag; i++) {
                covariance += ((double)samples.get(i) - mean) * ((double)samples.get(i + lag) - mean);
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
        
        // 简单的收敛检查：检查对数概率的稳定性
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
        return cv < 0.1; // 变异系数小于0.1认为收敛
    }
    
    @Override
    public void setParameters(double stepSize, int adaptationPeriod) {
        this.stepSize = stepSize;
        this.adaptationPeriod = adaptationPeriod;
    }
    
    /**
     * 设置目标接受率
     * Set target acceptance rate
     */
    public void setTargetAcceptanceRate(double targetAcceptanceRate) {
        this.targetAcceptanceRate = targetAcceptanceRate;
    }
    
    /**
     * 设置是否使用自适应步长
     * Set whether to use adaptive step size
     */
    public void setAdaptiveStepSize(boolean adaptiveStepSize) {
        this.adaptiveStepSize = adaptiveStepSize;
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
        return "Metropolis-Hastings";
    }
    
    @Override
    public boolean supportsAdaptation() {
        return true;
    }
    
    /**
     * 获取当前步长
     * Get current step size
     */
    public double getStepSize() {
        return stepSize;
    }
    
    /**
     * 获取目标接受率
     * Get target acceptance rate
     */
    public double getTargetAcceptanceRate() {
        return targetAcceptanceRate;
    }
}