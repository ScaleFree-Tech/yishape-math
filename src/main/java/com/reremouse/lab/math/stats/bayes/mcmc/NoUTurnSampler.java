package com.reremouse.lab.math.stats.bayes.mcmc;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.stats.distribution.NormalDistribution;
import java.util.Random;
import java.util.ArrayList;
import java.util.List;

/**
 * No-U-Turn Sampler (NUTS) 采样器
 * No-U-Turn Sampler (NUTS)
 * 
 * <p>NUTS是HMC的自适应版本，自动调整轨迹长度以避免U-turn现象，
 * 提供更高效的采样性能。</p>
 * <p>NUTS is an adaptive version of HMC that automatically tunes the trajectory 
 * length to avoid U-turns, providing more efficient sampling performance.</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class NoUTurnSampler implements IMCMCSampler {
    
    private double stepSize;
    private Random random;
    private IMatrix massMatrix;
    private double targetAcceptanceRate;
    private int adaptationPeriod;
    private boolean adaptStepSize;
    private double stepSizeAdaptationRate;
    private double maxTreeDepth;
    
    /**
     * 梯度函数接口（用于NUTS）
     * Gradient function interface (for NUTS)
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
     * 树节点类
     * Tree node class
     */
    private static class TreeNode {
        IVector position;
        IVector momentum;
        IVector positionMinus;
        IVector momentumMinus;
        IVector positionPlus;
        IVector momentumPlus;
        IVector positionPrime;
        double logProbPrime;
        int numNodes;
        boolean valid;
        double sumMetropolisProb;
        
        TreeNode(IVector position, IVector momentum, double logProb) {
            this.position = position.copy();
            this.momentum = momentum.copy();
            this.positionMinus = position.copy();
            this.momentumMinus = momentum.copy();
            this.positionPlus = position.copy();
            this.momentumPlus = momentum.copy();
            this.positionPrime = position.copy();
            this.logProbPrime = logProb;
            this.numNodes = 1;
            this.valid = true;
            this.sumMetropolisProb = 1.0;
        }
    }
    
    /**
     * 默认构造函数
     * Default constructor
     */
    public NoUTurnSampler() {
        this(0.1, new Random());
    }
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param stepSize 初始步长 / Initial step size
     * @param random 随机数生成器 / Random number generator
     */
    public NoUTurnSampler(double stepSize, Random random) {
        this.stepSize = stepSize;
        this.random = random;
        this.targetAcceptanceRate = 0.65;
        this.adaptationPeriod = 1000;
        this.adaptStepSize = true;
        this.stepSizeAdaptationRate = 0.75;
        this.maxTreeDepth = 10;
    }
    
    /**
     * 设置质量矩阵
     * Set mass matrix
     */
    public void setMassMatrix(IMatrix massMatrix) {
        this.massMatrix = massMatrix;
    }
    
    /**
     * 设置最大树深度
     * Set maximum tree depth
     */
    public void setMaxTreeDepth(double maxTreeDepth) {
        this.maxTreeDepth = maxTreeDepth;
    }
    
    @Override
    public SamplingResult sample(TargetDistribution targetDistribution, 
                               IVector initialState, 
                               int numSamples, 
                               int burnIn) {
        throw new UnsupportedOperationException(
            "NUTS requires gradient information. Use sample() with GradientFunction.");
    }
    
    @Override
    public SamplingResult sample(TargetDistribution targetDistribution,
                               ProposalDistribution proposalDistribution,
                               IVector initialState,
                               int numSamples,
                               int burnIn) {
        throw new UnsupportedOperationException(
            "NUTS requires gradient information. Use sample() with GradientFunction.");
    }
    
    /**
     * 执行NUTS采样（带梯度函数）
     * Perform NUTS sampling with gradient function
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
        double logStepSize = Math.log(stepSize);
        double hBar = 0.0; // 用于步长自适应
        
        for (int i = 0; i < totalSamples; i++) {
            // 生成动量
            IVector momentum = generateMomentum(dimension);
            
            // 计算当前哈密顿量
            double currentHamiltonian = calculateHamiltonian(currentState, momentum, 
                                                           currentLogProb);
            
            // 构建NUTS树
            TreeNode tree = buildTree(currentState, momentum, currentLogProb,
                                    gradientFunction, targetDistribution, 
                                    currentStepSize, currentHamiltonian);
            
            // 选择新状态
            if (tree.valid && tree.numNodes > 0) {
                double acceptanceProb = Math.min(1.0, tree.sumMetropolisProb / tree.numNodes);
                
                if (random.nextDouble() < acceptanceProb) {
                    currentState = tree.positionPrime.copy();
                    currentLogProb = tree.logProbPrime;
                    acceptedCount++;
                }
                
                // 自适应步长调整（Dual Averaging）
                if (adaptStepSize && i < adaptationPeriod) {
                    double alpha = Math.min(1.0, tree.sumMetropolisProb / tree.numNodes);
                    hBar = (1.0 - 1.0 / (i + 10.0)) * hBar + 
                           (targetAcceptanceRate - alpha) / (i + 10.0);
                    
                    logStepSize = Math.log(10.0) - Math.sqrt(i + 1) / 10.0 * hBar;
                    currentStepSize = Math.exp(logStepSize);
                    
                    if (i == adaptationPeriod - 1) {
                        stepSize = currentStepSize;
                    }
                }
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
     * 构建NUTS树
     * Build NUTS tree
     */
    private TreeNode buildTree(IVector position, IVector momentum, double logProb,
                             GradientFunction gradientFunction,
                             TargetDistribution targetDistribution,
                             double stepSize, double hamiltonian0) {
        
        TreeNode tree = new TreeNode(position, momentum, logProb);
        
        for (int depth = 0; depth < maxTreeDepth; depth++) {
            // 随机选择方向
            int direction = random.nextBoolean() ? 1 : -1;
            
            // 扩展树
            TreeNode subtree;
            if (direction == 1) {
                subtree = buildSubtree(tree.positionPlus, tree.momentumPlus,
                                     gradientFunction, targetDistribution,
                                     stepSize, depth, hamiltonian0, direction);
                if (subtree.valid) {
                    tree.positionPlus = subtree.positionPlus.copy();
                    tree.momentumPlus = subtree.momentumPlus.copy();
                }
            } else {
                subtree = buildSubtree(tree.positionMinus, tree.momentumMinus,
                                     gradientFunction, targetDistribution,
                                     stepSize, depth, hamiltonian0, direction);
                if (subtree.valid) {
                    tree.positionMinus = subtree.positionMinus.copy();
                    tree.momentumMinus = subtree.momentumMinus.copy();
                }
            }
            
            if (!subtree.valid) {
                tree.valid = false;
                break;
            }
            
            // 更新树的统计信息
            double acceptanceProb = subtree.sumMetropolisProb / 
                                  (tree.sumMetropolisProb + subtree.sumMetropolisProb);
            
            if (random.nextDouble() < acceptanceProb) {
                tree.positionPrime = subtree.positionPrime.copy();
                tree.logProbPrime = subtree.logProbPrime;
            }
            
            tree.sumMetropolisProb += subtree.sumMetropolisProb;
            tree.numNodes += subtree.numNodes;
            
            // 检查U-turn条件
            if (checkUTurn(tree.positionMinus, tree.positionPlus,
                          tree.momentumMinus, tree.momentumPlus)) {
                break;
            }
        }
        
        return tree;
    }
    
    /**
     * 构建子树
     * Build subtree
     */
    private TreeNode buildSubtree(IVector position, IVector momentum,
                                GradientFunction gradientFunction,
                                TargetDistribution targetDistribution,
                                double stepSize, int depth, double hamiltonian0,
                                int direction) {
        
        if (depth == 0) {
            // 基础情况：执行一步Leapfrog
            LeapfrogResult result = leapfrog(position, momentum, gradientFunction, 
                                           stepSize * direction);
            
            double newLogProb = targetDistribution.logPdf(result.position);
            double newHamiltonian = calculateHamiltonian(result.position, 
                                                       result.momentum, newLogProb);
            
            TreeNode node = new TreeNode(result.position, result.momentum, newLogProb);
            
            // 检查能量约束
            double deltaH = newHamiltonian - hamiltonian0;
            if (deltaH > 1000) { // 防止数值溢出
                node.valid = false;
                node.sumMetropolisProb = 0.0;
            } else {
                node.sumMetropolisProb = Math.min(1.0, Math.exp(-deltaH));
            }
            
            return node;
        } else {
            // 递归情况：构建更深的子树
            TreeNode subtree1 = buildSubtree(position, momentum, gradientFunction,
                                           targetDistribution, stepSize, depth - 1,
                                           hamiltonian0, direction);
            
            if (!subtree1.valid) {
                return subtree1;
            }
            
            IVector nextPosition = direction == 1 ? subtree1.positionPlus : subtree1.positionMinus;
            IVector nextMomentum = direction == 1 ? subtree1.momentumPlus : subtree1.momentumMinus;
            
            TreeNode subtree2 = buildSubtree(nextPosition, nextMomentum, gradientFunction,
                                           targetDistribution, stepSize, depth - 1,
                                           hamiltonian0, direction);
            
            if (!subtree2.valid) {
                subtree1.valid = false;
                return subtree1;
            }
            
            // 合并子树
            TreeNode combined = new TreeNode(subtree1.position, subtree1.momentum, 
                                           subtree1.logProbPrime);
            
            if (direction == 1) {
                combined.positionMinus = subtree1.positionMinus.copy();
                combined.momentumMinus = subtree1.momentumMinus.copy();
                combined.positionPlus = subtree2.positionPlus.copy();
                combined.momentumPlus = subtree2.momentumPlus.copy();
            } else {
                combined.positionMinus = subtree2.positionMinus.copy();
                combined.momentumMinus = subtree2.momentumMinus.copy();
                combined.positionPlus = subtree1.positionPlus.copy();
                combined.momentumPlus = subtree1.momentumPlus.copy();
            }
            
            // 选择候选状态
            double totalWeight = subtree1.sumMetropolisProb + subtree2.sumMetropolisProb;
            if (totalWeight > 0) {
                double prob = subtree2.sumMetropolisProb / totalWeight;
                if (random.nextDouble() < prob) {
                    combined.positionPrime = subtree2.positionPrime.copy();
                    combined.logProbPrime = subtree2.logProbPrime;
                } else {
                    combined.positionPrime = subtree1.positionPrime.copy();
                    combined.logProbPrime = subtree1.logProbPrime;
                }
            }
            
            combined.sumMetropolisProb = totalWeight;
            combined.numNodes = subtree1.numNodes + subtree2.numNodes;
            
            // 检查U-turn条件
            combined.valid = !checkUTurn(combined.positionMinus, combined.positionPlus,
                                       combined.momentumMinus, combined.momentumPlus);
            
            return combined;
        }
    }
    
    /**
     * 检查U-turn条件
     * Check U-turn condition
     */
    private boolean checkUTurn(IVector positionMinus, IVector positionPlus,
                             IVector momentumMinus, IVector momentumPlus) {
        
        IVector deltaPosition = positionPlus.sub(positionMinus);
        
        double dotMinus = 0.0;
        double dotPlus = 0.0;
        
        for (int i = 0; i < deltaPosition.size(); i++) {
            double delta = deltaPosition.get(i).doubleValue();
            dotMinus += delta * momentumMinus.get(i).doubleValue();
            dotPlus += delta * momentumPlus.get(i).doubleValue();
        }
        
        return dotMinus <= 0 || dotPlus <= 0;
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
     * 单步Leapfrog积分
     * Single step leapfrog integration
     */
    private LeapfrogResult leapfrog(IVector position, IVector momentum,
                                  GradientFunction gradientFunction, double stepSize) {
        
        IVector newPosition = position.copy();
        IVector newMomentum = momentum.copy();
        
        // 第一步：更新动量的一半
        IVector gradient = gradientFunction.computeGradient(newPosition);
        for (int i = 0; i < newMomentum.size(); i++) {
            double updatedMomentum = newMomentum.get(i).doubleValue() - 
                                   0.5 * stepSize * gradient.get(i).doubleValue();
            newMomentum.set(i, updatedMomentum);
        }
        
        // 第二步：更新位置
        for (int i = 0; i < newPosition.size(); i++) {
            double updatedPosition = newPosition.get(i).doubleValue() + 
                                   stepSize * newMomentum.get(i).doubleValue();
            newPosition.set(i, updatedPosition);
        }
        
        // 第三步：更新动量的另一半
        gradient = gradientFunction.computeGradient(newPosition);
        for (int i = 0; i < newMomentum.size(); i++) {
            double updatedMomentum = newMomentum.get(i).doubleValue() - 
                                   0.5 * stepSize * gradient.get(i).doubleValue();
            newMomentum.set(i, updatedMomentum);
        }
        
        return new LeapfrogResult(newPosition, newMomentum);
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
        double potentialEnergy = -logProb;
        
        double kineticEnergy = 0.0;
        for (int i = 0; i < momentum.size(); i++) {
            double p = momentum.get(i).doubleValue();
            kineticEnergy += 0.5 * p * p;
        }
        
        return potentialEnergy + kineticEnergy;
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
        return "No-U-Turn Sampler (NUTS)";
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
}