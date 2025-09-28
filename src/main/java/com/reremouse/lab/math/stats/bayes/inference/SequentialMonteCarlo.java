package com.reremouse.lab.math.stats.bayes.inference;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

/**
 * 序贯蒙特卡洛方法
 * Sequential Monte Carlo (SMC)
 * 
 * <p>序贯蒙特卡洛是一类基于粒子的推断方法，用于处理序列数据和动态模型。
 * 它包括粒子滤波、SMC采样器等多种变体，适用于状态空间模型和参数估计。</p>
 * 
 * <p>Sequential Monte Carlo is a class of particle-based inference methods 
 * for handling sequential data and dynamic models.
 * It includes particle filters, SMC samplers and other variants, 
 * suitable for state space models and parameter estimation.</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class SequentialMonteCarlo {
    
    private final int numParticles;
    private final Random random;
    private final double resampleThreshold;
    
    /**
     * 构造函数
     * 
     * @param numParticles 粒子数量
     */
    public SequentialMonteCarlo(int numParticles) {
        this(numParticles, 0.5, new Random());
    }
    
    /**
     * 构造函数
     * 
     * @param numParticles 粒子数量
     * @param resampleThreshold 重采样阈值
     * @param random 随机数生成器
     */
    public SequentialMonteCarlo(int numParticles, double resampleThreshold, Random random) {
        this.numParticles = numParticles;
        this.resampleThreshold = resampleThreshold;
        this.random = random;
    }
    
    /**
     * SMC粒子
     * SMC Particle
     */
    public static class SMCParticle {
        private IVector parameter;
        private IVector state;
        private double weight;
        private double logWeight;
        private List<IVector> trajectory;
        
        public SMCParticle(IVector parameter, IVector state, double weight) {
            this.parameter = parameter;
            this.state = state;
            this.weight = weight;
            this.logWeight = Math.log(weight);
            this.trajectory = new ArrayList<>();
            if (state != null) {
                this.trajectory.add(copyVector(state));
            }
        }
        
        public IVector getParameter() { return parameter; }
        public void setParameter(IVector parameter) { this.parameter = parameter; }
        public IVector getState() { return state; }
        public void setState(IVector state) { 
            this.state = state; 
            if (state != null) {
                this.trajectory.add(copyVector(state));
            }
        }
        public double getWeight() { return weight; }
        public void setWeight(double weight) { 
            this.weight = weight; 
            this.logWeight = Math.log(Math.max(weight, 1e-300));
        }
        public double getLogWeight() { return logWeight; }
        public void setLogWeight(double logWeight) { 
            this.logWeight = logWeight; 
            this.weight = Math.exp(logWeight);
        }
        public List<IVector> getTrajectory() { return trajectory; }
        
        public SMCParticle copy() {
            SMCParticle copy = new SMCParticle(
                parameter != null ? copyVector(parameter) : null,
                state != null ? copyVector(state) : null,
                weight
            );
            copy.logWeight = logWeight;
            copy.trajectory = new ArrayList<>();
            for (IVector state : trajectory) {
                copy.trajectory.add(copyVector(state));
            }
            return copy;
        }
        
        private IVector copyVector(IVector vector) {
            IVector copy = Linalg.vector(vector.size());
            for (int i = 0; i < vector.size(); i++) {
                copy.set(i, vector.get(i));
            }
            return copy;
        }
    }
    
    /**
     * SMC结果
     * SMC Result
     */
    public static class SMCResult {
        private final List<SMCParticle> particles;
        private final IVector parameterEstimate;
        private final IVector stateEstimate;
        private final double logMarginalLikelihood;
        private final double effectiveSampleSize;
        private final List<Double> logLikelihoods;
        
        public SMCResult(List<SMCParticle> particles, IVector parameterEstimate,
                        IVector stateEstimate, double logMarginalLikelihood,
                        double effectiveSampleSize, List<Double> logLikelihoods) {
            this.particles = particles;
            this.parameterEstimate = parameterEstimate;
            this.stateEstimate = stateEstimate;
            this.logMarginalLikelihood = logMarginalLikelihood;
            this.effectiveSampleSize = effectiveSampleSize;
            this.logLikelihoods = logLikelihoods;
        }
        
        public List<SMCParticle> getParticles() { return particles; }
        public IVector getParameterEstimate() { return parameterEstimate; }
        public IVector getStateEstimate() { return stateEstimate; }
        public double getLogMarginalLikelihood() { return logMarginalLikelihood; }
        public double getEffectiveSampleSize() { return effectiveSampleSize; }
        public List<Double> getLogLikelihoods() { return logLikelihoods; }
    }
    
    /**
     * 参数转移核接口
     * Parameter transition kernel interface
     */
    @FunctionalInterface
    public interface ParameterKernel {
        /**
         * 参数转移
         * Parameter transition
         * 
         * @param currentParameter 当前参数
         * @param timeStep 时间步长
         * @return 新参数
         */
        IVector transition(IVector currentParameter, double timeStep);
    }
    
    /**
     * 状态转移核接口
     * State transition kernel interface
     */
    @FunctionalInterface
    public interface StateKernel {
        /**
         * 状态转移
         * State transition
         * 
         * @param currentState 当前状态
         * @param parameter 参数
         * @param timeStep 时间步长
         * @return 新状态
         */
        IVector transition(IVector currentState, IVector parameter, double timeStep);
    }
    
    /**
     * 似然函数接口
     * Likelihood function interface
     */
    @FunctionalInterface
    public interface LikelihoodFunction {
        /**
         * 计算似然
         * Calculate likelihood
         * 
         * @param observation 观测值
         * @param state 状态
         * @param parameter 参数
         * @return 对数似然值
         */
        double logLikelihood(IVector observation, IVector state, IVector parameter);
    }
    
    /**
     * SMC采样器 - 用于参数估计
     * SMC Sampler - for parameter estimation
     * 
     * @param observations 观测序列
     * @param parameterKernel 参数转移核
     * @param stateKernel 状态转移核
     * @param likelihoodFunction 似然函数
     * @param initialParameterSampler 初始参数采样器
     * @param initialStateSampler 初始状态采样器
     * @return SMC结果
     */
    public SMCResult smcSampler(List<IVector> observations,
                               ParameterKernel parameterKernel,
                               StateKernel stateKernel,
                               LikelihoodFunction likelihoodFunction,
                               Function<Random, IVector> initialParameterSampler,
                               Function<IVector, IVector> initialStateSampler) {
        
        // 初始化粒子
        List<SMCParticle> particles = initializeParticles(
            initialParameterSampler, initialStateSampler);
        
        List<Double> logLikelihoods = new ArrayList<>();
        double cumulativeLogLikelihood = 0.0;
        
        // 序贯处理观测
        for (int t = 0; t < observations.size(); t++) {
            IVector observation = observations.get(t);
            
            // 参数传播
            if (t > 0) {
                propagateParameters(particles, parameterKernel, 1.0);
            }
            
            // 状态传播
            propagateStates(particles, stateKernel, 1.0);
            
            // 权重更新
            double stepLogLikelihood = updateWeights(particles, observation, likelihoodFunction);
            logLikelihoods.add(stepLogLikelihood);
            cumulativeLogLikelihood += stepLogLikelihood;
            
            // 重采样
            double ess = calculateEffectiveSampleSize(particles);
            if (ess < resampleThreshold * numParticles) {
                resampleParticles(particles);
            }
        }
        
        // 计算最终估计
        IVector parameterEstimate = calculateParameterEstimate(particles);
        IVector stateEstimate = calculateStateEstimate(particles);
        double finalESS = calculateEffectiveSampleSize(particles);
        
        return new SMCResult(particles, parameterEstimate, stateEstimate,
                           cumulativeLogLikelihood, finalESS, logLikelihoods);
    }
    
    /**
     * 自适应SMC - 自动调整粒子数和重采样策略
     * Adaptive SMC - automatically adjust particle count and resampling strategy
     * 
     * @param observations 观测序列
     * @param parameterKernel 参数转移核
     * @param stateKernel 状态转移核
     * @param likelihoodFunction 似然函数
     * @param initialParameterSampler 初始参数采样器
     * @param initialStateSampler 初始状态采样器
     * @param targetESS 目标有效样本大小
     * @return SMC结果
     */
    public SMCResult adaptiveSMC(List<IVector> observations,
                                ParameterKernel parameterKernel,
                                StateKernel stateKernel,
                                LikelihoodFunction likelihoodFunction,
                                Function<Random, IVector> initialParameterSampler,
                                Function<IVector, IVector> initialStateSampler,
                                double targetESS) {
        
        List<SMCParticle> particles = initializeParticles(
            initialParameterSampler, initialStateSampler);
        
        List<Double> logLikelihoods = new ArrayList<>();
        double cumulativeLogLikelihood = 0.0;
        
        for (int t = 0; t < observations.size(); t++) {
            IVector observation = observations.get(t);
            
            // 自适应参数传播
            if (t > 0) {
                double adaptiveStepSize = calculateAdaptiveStepSize(particles, t);
                propagateParameters(particles, parameterKernel, adaptiveStepSize);
            }
            
            // 状态传播
            propagateStates(particles, stateKernel, 1.0);
            
            // 权重更新
            double stepLogLikelihood = updateWeights(particles, observation, likelihoodFunction);
            logLikelihoods.add(stepLogLikelihood);
            cumulativeLogLikelihood += stepLogLikelihood;
            
            // 自适应重采样
            double ess = calculateEffectiveSampleSize(particles);
            if (ess < targetESS) {
                resampleParticles(particles);
                
                // 如果ESS仍然太低，考虑增加粒子数（简化实现）
                if (ess < targetESS * 0.5) {
                    particles = augmentParticles(particles, (int)(numParticles * 0.2));
                }
            }
        }
        
        IVector parameterEstimate = calculateParameterEstimate(particles);
        IVector stateEstimate = calculateStateEstimate(particles);
        double finalESS = calculateEffectiveSampleSize(particles);
        
        return new SMCResult(particles, parameterEstimate, stateEstimate,
                           cumulativeLogLikelihood, finalESS, logLikelihoods);
    }
    
    /**
     * 粒子MCMC - 结合MCMC的SMC方法
     * Particle MCMC - SMC method combined with MCMC
     * 
     * @param observations 观测序列
     * @param parameterKernel 参数转移核
     * @param stateKernel 状态转移核
     * @param likelihoodFunction 似然函数
     * @param initialParameterSampler 初始参数采样器
     * @param initialStateSampler 初始状态采样器
     * @param mcmcSteps MCMC步数
     * @return SMC结果
     */
    public SMCResult particleMCMC(List<IVector> observations,
                                 ParameterKernel parameterKernel,
                                 StateKernel stateKernel,
                                 LikelihoodFunction likelihoodFunction,
                                 Function<Random, IVector> initialParameterSampler,
                                 Function<IVector, IVector> initialStateSampler,
                                 int mcmcSteps) {
        
        List<SMCParticle> particles = initializeParticles(
            initialParameterSampler, initialStateSampler);
        
        List<Double> logLikelihoods = new ArrayList<>();
        double cumulativeLogLikelihood = 0.0;
        
        for (int t = 0; t < observations.size(); t++) {
            IVector observation = observations.get(t);
            
            // 标准SMC步骤
            if (t > 0) {
                propagateParameters(particles, parameterKernel, 1.0);
            }
            propagateStates(particles, stateKernel, 1.0);
            double stepLogLikelihood = updateWeights(particles, observation, likelihoodFunction);
            logLikelihoods.add(stepLogLikelihood);
            cumulativeLogLikelihood += stepLogLikelihood;
            
            // MCMC改进步骤
            if (t % 5 == 0) { // 每5步执行一次MCMC
                performMCMCStep(particles, observation, likelihoodFunction, mcmcSteps);
            }
            
            // 重采样
            double ess = calculateEffectiveSampleSize(particles);
            if (ess < resampleThreshold * numParticles) {
                resampleParticles(particles);
            }
        }
        
        IVector parameterEstimate = calculateParameterEstimate(particles);
        IVector stateEstimate = calculateStateEstimate(particles);
        double finalESS = calculateEffectiveSampleSize(particles);
        
        return new SMCResult(particles, parameterEstimate, stateEstimate,
                           cumulativeLogLikelihood, finalESS, logLikelihoods);
    }
    
    /**
     * 初始化粒子
     * Initialize particles
     */
    private List<SMCParticle> initializeParticles(
            Function<Random, IVector> parameterSampler,
            Function<IVector, IVector> stateSampler) {
        
        List<SMCParticle> particles = new ArrayList<>();
        double uniformWeight = 1.0 / numParticles;
        
        for (int i = 0; i < numParticles; i++) {
            IVector parameter = parameterSampler.apply(random);
            IVector state = stateSampler.apply(parameter);
            particles.add(new SMCParticle(parameter, state, uniformWeight));
        }
        
        return particles;
    }
    
    /**
     * 传播参数
     * Propagate parameters
     */
    private void propagateParameters(List<SMCParticle> particles, 
                                   ParameterKernel kernel, 
                                   double stepSize) {
        for (SMCParticle particle : particles) {
            IVector newParameter = kernel.transition(particle.getParameter(), stepSize);
            particle.setParameter(newParameter);
        }
    }
    
    /**
     * 传播状态
     * Propagate states
     */
    private void propagateStates(List<SMCParticle> particles, 
                               StateKernel kernel, 
                               double stepSize) {
        for (SMCParticle particle : particles) {
            IVector newState = kernel.transition(
                particle.getState(), particle.getParameter(), stepSize);
            particle.setState(newState);
        }
    }
    
    /**
     * 更新权重
     * Update weights
     */
    private double updateWeights(List<SMCParticle> particles, 
                               IVector observation, 
                               LikelihoodFunction likelihoodFunction) {
        
        double maxLogWeight = Double.NEGATIVE_INFINITY;
        
        // 计算对数权重
        for (SMCParticle particle : particles) {
            double logLikelihood = likelihoodFunction.logLikelihood(
                observation, particle.getState(), particle.getParameter());
            double newLogWeight = particle.getLogWeight() + logLikelihood;
            particle.setLogWeight(newLogWeight);
            maxLogWeight = Math.max(maxLogWeight, newLogWeight);
        }
        
        // 归一化权重（数值稳定）
        double totalWeight = 0.0;
        for (SMCParticle particle : particles) {
            double normalizedLogWeight = particle.getLogWeight() - maxLogWeight;
            double weight = Math.exp(normalizedLogWeight);
            particle.setWeight(weight);
            totalWeight += weight;
        }
        
        // 最终归一化
        for (SMCParticle particle : particles) {
            particle.setWeight(particle.getWeight() / totalWeight);
        }
        
        // 返回步骤对数似然
        return maxLogWeight + Math.log(totalWeight / numParticles);
    }
    
    /**
     * 重采样粒子
     * Resample particles
     */
    private void resampleParticles(List<SMCParticle> particles) {
        List<SMCParticle> newParticles = new ArrayList<>();
        double[] cumulativeWeights = new double[particles.size()];
        
        // 计算累积权重
        cumulativeWeights[0] = particles.get(0).getWeight();
        for (int i = 1; i < particles.size(); i++) {
            cumulativeWeights[i] = cumulativeWeights[i-1] + particles.get(i).getWeight();
        }
        
        // 系统重采样
        double u0 = random.nextDouble() / numParticles;
        for (int i = 0; i < numParticles; i++) {
            double u = u0 + (double) i / numParticles;
            int index = findIndex(cumulativeWeights, u);
            SMCParticle selectedParticle = particles.get(index).copy();
            selectedParticle.setWeight(1.0 / numParticles);
            newParticles.add(selectedParticle);
        }
        
        particles.clear();
        particles.addAll(newParticles);
    }
    
    /**
     * 执行MCMC步骤
     * Perform MCMC step
     */
    private void performMCMCStep(List<SMCParticle> particles, 
                               IVector observation, 
                               LikelihoodFunction likelihoodFunction,
                               int mcmcSteps) {
        
        for (SMCParticle particle : particles) {
            for (int step = 0; step < mcmcSteps; step++) {
                // 简单的随机游走Metropolis步骤
                IVector currentParameter = particle.getParameter();
                IVector proposedParameter = addNoise(currentParameter, 0.1);
                
                double currentLogLikelihood = likelihoodFunction.logLikelihood(
                    observation, particle.getState(), currentParameter);
                double proposedLogLikelihood = likelihoodFunction.logLikelihood(
                    observation, particle.getState(), proposedParameter);
                
                double acceptanceRatio = Math.exp(proposedLogLikelihood - currentLogLikelihood);
                
                if (random.nextDouble() < acceptanceRatio) {
                    particle.setParameter(proposedParameter);
                }
            }
        }
    }
    
    /**
     * 计算自适应步长
     * Calculate adaptive step size
     */
    private double calculateAdaptiveStepSize(List<SMCParticle> particles, int timeStep) {
        // 简单的自适应策略：基于粒子分散度
        double variance = calculateParameterVariance(particles);
        double baseStepSize = 1.0;
        double adaptationFactor = Math.exp(-variance * 0.1);
        return baseStepSize * adaptationFactor;
    }
    
    /**
     * 增强粒子集
     * Augment particle set
     */
    private List<SMCParticle> augmentParticles(List<SMCParticle> particles, int additionalParticles) {
        List<SMCParticle> augmentedParticles = new ArrayList<>(particles);
        
        for (int i = 0; i < additionalParticles; i++) {
            // 从现有粒子中随机选择一个进行复制和扰动
            int index = random.nextInt(particles.size());
            SMCParticle original = particles.get(index);
            SMCParticle augmented = original.copy();
            
            // 添加小的随机扰动
            IVector noisyParameter = addNoise(augmented.getParameter(), 0.05);
            augmented.setParameter(noisyParameter);
            augmented.setWeight(1.0 / (particles.size() + additionalParticles));
            
            augmentedParticles.add(augmented);
        }
        
        // 重新归一化权重
        double totalWeight = augmentedParticles.size();
        for (SMCParticle particle : augmentedParticles) {
            particle.setWeight(1.0 / totalWeight);
        }
        
        return augmentedParticles;
    }
    
    /**
     * 添加噪声
     * Add noise
     */
    private IVector addNoise(IVector vector, double noiseLevel) {
        IVector noisyVector = Linalg.vector(vector.size());
        for (int i = 0; i < vector.size(); i++) {
            double noise = noiseLevel * random.nextGaussian();
            noisyVector.set(i, vector.get(i).doubleValue() + noise);
        }
        return noisyVector;
    }
    
    /**
     * 计算参数方差
     * Calculate parameter variance
     */
    private double calculateParameterVariance(List<SMCParticle> particles) {
        if (particles.isEmpty()) return 0.0;
        
        IVector mean = calculateParameterEstimate(particles);
        double variance = 0.0;
        
        for (SMCParticle particle : particles) {
            IVector param = particle.getParameter();
            for (int i = 0; i < param.size(); i++) {
                double diff = param.get(i).doubleValue() - mean.get(i).doubleValue();
                variance += particle.getWeight() * diff * diff;
            }
        }
        
        return variance / mean.size();
    }
    
    /**
     * 计算参数估计
     * Calculate parameter estimate
     */
    private IVector calculateParameterEstimate(List<SMCParticle> particles) {
        if (particles.isEmpty()) return Linalg.vector(0);
        
        int paramDim = particles.get(0).getParameter().size();
        IVector estimate = Linalg.vector(paramDim);
        
        for (int i = 0; i < paramDim; i++) {
            double sum = 0.0;
            for (SMCParticle particle : particles) {
                sum += particle.getWeight() * particle.getParameter().get(i).doubleValue();
            }
            estimate.set(i, sum);
        }
        
        return estimate;
    }
    
    /**
     * 计算状态估计
     * Calculate state estimate
     */
    private IVector calculateStateEstimate(List<SMCParticle> particles) {
        if (particles.isEmpty()) return Linalg.vector(0);
        
        int stateDim = particles.get(0).getState().size();
        IVector estimate = Linalg.vector(stateDim);
        
        for (int i = 0; i < stateDim; i++) {
            double sum = 0.0;
            for (SMCParticle particle : particles) {
                sum += particle.getWeight() * particle.getState().get(i).doubleValue();
            }
            estimate.set(i, sum);
        }
        
        return estimate;
    }
    
    /**
     * 计算有效样本大小
     * Calculate effective sample size
     */
    private double calculateEffectiveSampleSize(List<SMCParticle> particles) {
        double sumSquaredWeights = 0.0;
        
        for (SMCParticle particle : particles) {
            double weight = particle.getWeight();
            sumSquaredWeights += weight * weight;
        }
        
        return 1.0 / sumSquaredWeights;
    }
    
    /**
     * 查找索引
     * Find index
     */
    private int findIndex(double[] cumulativeWeights, double u) {
        for (int i = 0; i < cumulativeWeights.length; i++) {
            if (u <= cumulativeWeights[i]) {
                return i;
            }
        }
        return cumulativeWeights.length - 1;
    }
}