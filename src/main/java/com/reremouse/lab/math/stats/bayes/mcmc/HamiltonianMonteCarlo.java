package com.reremouse.lab.math.stats.bayes.mcmc;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.stats.distribution.NormalDistribution;
import com.reremouse.lab.math.optimize.IGradientFunction;
import java.util.Random;
import java.util.ArrayList;
import java.util.List;

/**
 * Hamiltonian Monte Carlo (HMC) 采样器
 * Hamiltonian Monte Carlo (HMC) Sampler
 * 
 * <p>实现了Hamiltonian Monte Carlo算法，通过引入动量变量和哈密顿动力学来提高采样效率。
 * 特别适用于高维连续参数空间的采样。</p>
 * <p>Implements the Hamiltonian Monte Carlo algorithm, which introduces momentum 
 * variables and Hamiltonian dynamics to improve sampling efficiency. 
 * Particularly suitable for sampling in high-dimensional continuous parameter spaces.</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class HamiltonianMonteCarlo implements IMCMCSampler {
    
    private double stepSize;
    private int numLeapfrogSteps;
    private Random random;
    private IMatrix massMatrix;
    private boolean adaptStepSize;
    private double targetAcceptanceRate;
    private int adaptationPeriod;
    
    /**
     * 梯度函数接口（用于HMC）
     * Gradient function interface (for HMC)
     */
    @FunctionalInterface
    public interface GradientFunction {
        /**
         * 计算负对数概率密度的梯度
         * Calculate gradient of negative log probability density
         * 
         * @param parameters 参数向量 / Parameter vector
         * @return 梯度向量 / Gradient vector
         */
        IVector computeGradient(IVector parameters);
    }
    
    /**
     * 默认构造函数
     * Default constructor
     */
    public HamiltonianMonteCarlo() {
        this(0.1, 10, new Random());
    }
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param stepSize 步长 / Step size
     * @param numLeapfrogSteps Leapfrog积分步数 / Number of leapfrog steps
     * @param random 随机数生成器 / Random number generator
     */
    public HamiltonianMonteCarlo(double stepSize, int numLeapfrogSteps, Random random) {
        this.stepSize = stepSize;
        this.numLeapfrogSteps = numLeapfrogSteps;
        this.random = random;
        this.adaptStepSize = true;
        this.targetAcceptanceRate = 0.65;
        this.adaptationPeriod = 1000;
    }
    
    /**
     * 设置质量矩阵
     * Set mass matrix
     */
    public void setMassMatrix(IMatrix massMatrix) {
        this.massMatrix = massMatrix;
    }
    
    /**
     * 设置Leapfrog步数
     * Set number of leapfrog steps
     */
    public void setNumLeapfrogSteps(int numLeapfrogSteps) {
        this.numLeapfrogSteps = numLeapfrogSteps;
    }
    
    @Override
    public SamplingResult sample(TargetDistribution targetDistribution, 
                               IVector initialState, 
                               int numSamples, 
                               int burnIn) {
        throw new UnsupportedOperationException(
            "HMC requires gradient information. Use sample() with GradientFunction.");
    }
    
    @Override
    public SamplingResult sample(TargetDistribution targetDistribution,
                               ProposalDistribution proposalDistribution,
                               IVector initialState,
                               int numSamples,
                               int burnIn) {
        throw new UnsupportedOperationException(
            "HMC requires gradient information. Use sample() with GradientFunction.");
    }
    
    /**
     * 执行HMC采样（带梯度函数）
     * Perform HMC sampling with gradient function
     * 
     * @param targetDistribution 目标分布 / Target distribution
     * @param gradientFunction 梯度函数 / Gradient function
     * @param initialState 初始状态 / Initial state
     * @param numSamples 采样数量 / Number of samples
     * @param burnIn 预热期样本数 / Number of burn-in samples
     * @return 采样结果 / Sampling result
     */
    public SamplingResult sample(TargetDistribution targetDistribution,
                               GradientFunction gradientFunction,
                               IVector initialState,
                               int numSamples,
                               int burnIn) {
        
        int totalSamples = numSamples + burnIn;
        int dimension = initialState.size();
        
        // 初始化质量矩阵（如果未设置）
        if (massMatrix == null) {
            massMatrix = Linalg.eye(dimension);
        }
        
        // 存储样本和对数概率
        List<IVector> samplesList = new ArrayList<>();
        List<Double> logProbsList = new ArrayList<>();
        
        IVector currentState = initialState.copy();
        double currentLogProb = targetDistribution.logPdf(currentState);
        
        int acceptedCount = 0;
        double currentStepSize = stepSize;
        
        for (int i = 0; i < totalSamples; i++) {
            // 生成动量
            IVector momentum = generateMomentum(dimension);
            
            // 计算当前哈密顿量
            double currentHamiltonian = calculateHamiltonian(currentState, momentum, 
                                                           currentLogProb);
            
            // Leapfrog积分
            LeapfrogResult leapfrogResult = leapfrog(currentState, momentum, 
                                                   gradientFunction, currentStepSize);
            
            // 计算提议状态的对数概率和哈密顿量
            double proposedLogProb = targetDistribution.logPdf(leapfrogResult.position);
            double proposedHamiltonian = calculateHamiltonian(leapfrogResult.position, 
                                                            leapfrogResult.momentum, 
                                                            proposedLogProb);
            
            // Metropolis接受/拒绝步骤
            double acceptanceProb = Math.min(1.0, 
                Math.exp(currentHamiltonian - proposedHamiltonian));
            
            if (random.nextDouble() < acceptanceProb) {
                currentState = leapfrogResult.position;
                currentLogProb = proposedLogProb;
                acceptedCount++;
            }
            
            // 自适应步长调整
            if (adaptStepSize && i < adaptationPeriod && i > 0 && i % 50 == 0) {
                double currentAcceptanceRate = (double) acceptedCount / (i + 1);
                if (currentAcceptanceRate > targetAcceptanceRate) {
                    currentStepSize *= 1.02;
                } else {
                    currentStepSize *= 0.98;
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
     * Leapfrog积分结果
     * Leapfrog integration result
     */
    private static class LeapfrogResult {
        final IVector position;
        final IVector momentum;
        
        LeapfrogResult(IVector position, IVector momentum) {
            this.position = position;
            this.momentum = momentum;
        }
    }
    
    /**
     * 生成动量向量
     * Generate momentum vector
     */
    private IVector generateMomentum(int dimension) {
        double[] momentumData = new double[dimension];
        NormalDistribution normal = new NormalDistribution(0, 1);
        
        for (int i = 0; i < dimension; i++) {
            momentumData[i] = normal.sample(1)[0];
        }
        
        return Linalg.vector(momentumData);
    }
    
    /**
     * 计算哈密顿量
     * Calculate Hamiltonian
     */
    private double calculateHamiltonian(IVector position, IVector momentum, double logProb) {
        // H = -log p(q) + 0.5 * p^T * M^(-1) * p
        // 其中 q 是位置，p 是动量，M 是质量矩阵
        
        double potentialEnergy = -logProb;
        
        // 计算动能：0.5 * p^T * M^(-1) * p
        // 简化情况：假设质量矩阵是单位矩阵
        double kineticEnergy = 0.0;
        for (int i = 0; i < momentum.size(); i++) {
            double p = momentum.get(i).doubleValue();
            kineticEnergy += 0.5 * p * p;
        }
        
        return potentialEnergy + kineticEnergy;
    }
    
    /**
     * Leapfrog积分
     * Leapfrog integration
     */
    private LeapfrogResult leapfrog(IVector initialPosition, IVector initialMomentum,
                                  GradientFunction gradientFunction, double stepSize) {
        
        IVector position = initialPosition.copy();
        IVector momentum = initialMomentum.copy();
        
        // 第一步：更新动量的一半
        IVector gradient = gradientFunction.computeGradient(position);
        for (int i = 0; i < momentum.size(); i++) {
            double newMomentum = momentum.get(i).doubleValue() - 
                               0.5 * stepSize * gradient.get(i).doubleValue();
            momentum.set(i, newMomentum);
        }
        
        // 主要的Leapfrog步骤
        for (int step = 0; step < numLeapfrogSteps; step++) {
            // 更新位置
            for (int i = 0; i < position.size(); i++) {
                double newPosition = position.get(i).doubleValue() + 
                                   stepSize * momentum.get(i).doubleValue();
                position.set(i, newPosition);
            }
            
            // 更新动量（除了最后一步）
            if (step < numLeapfrogSteps - 1) {
                gradient = gradientFunction.computeGradient(position);
                for (int i = 0; i < momentum.size(); i++) {
                    double newMomentum = momentum.get(i).doubleValue() - 
                                       stepSize * gradient.get(i).doubleValue();
                    momentum.set(i, newMomentum);
                }
            }
        }
        
        // 最后一步：更新动量的另一半
        gradient = gradientFunction.computeGradient(position);
        for (int i = 0; i < momentum.size(); i++) {
            double newMomentum = momentum.get(i).doubleValue() - 
                               0.5 * stepSize * gradient.get(i).doubleValue();
            momentum.set(i, newMomentum);
        }
        
        // 翻转动量（保持可逆性）
        for (int i = 0; i < momentum.size(); i++) {
            momentum.set(i, -momentum.get(i).doubleValue());
        }
        
        return new LeapfrogResult(position, momentum);
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
        
        for (int i = 0; i < n; i++) {
            double diff = samples.get(i).doubleValue() - mean;
            variance += diff * diff;
        }
        variance /= (n - 1);
        
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
     * 设置是否自适应步长
     * Set whether to adapt step size
     */
    public void setAdaptStepSize(boolean adaptStepSize) {
        this.adaptStepSize = adaptStepSize;
    }
    
    @Override
    public String getSamplerName() {
        return "Hamiltonian Monte Carlo";
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
     * 获取Leapfrog步数
     * Get number of leapfrog steps
     */
    public int getNumLeapfrogSteps() {
        return numLeapfrogSteps;
    }
}