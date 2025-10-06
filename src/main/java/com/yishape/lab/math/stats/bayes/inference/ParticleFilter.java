package com.yishape.lab.math.stats.bayes.inference;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * 粒子滤波
 * Particle Filter
 * 
 * <p>粒子滤波是一种序贯蒙特卡洛方法，用于估计动态系统的状态。
 * 它使用一组加权粒子来近似后验分布，特别适用于非线性、非高斯系统。</p>
 * 
 * <p>Particle filter is a sequential Monte Carlo method for estimating states of dynamic systems.
 * It uses a set of weighted particles to approximate the posterior distribution, 
 * particularly suitable for nonlinear, non-Gaussian systems.</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ParticleFilter {
    
    private final int numParticles;
    private final Random random;
    private final ResamplingStrategy resamplingStrategy;
    
    /**
     * 重采样策略枚举
     * Resampling strategy enumeration
     */
    public enum ResamplingStrategy {
        MULTINOMIAL,
        SYSTEMATIC,
        STRATIFIED,
        RESIDUAL
    }
    
    /**
     * 构造函数
     * 
     * @param numParticles 粒子数量
     */
    public ParticleFilter(int numParticles) {
        this(numParticles, ResamplingStrategy.SYSTEMATIC, new Random());
    }
    
    /**
     * 构造函数
     * 
     * @param numParticles 粒子数量
     * @param resamplingStrategy 重采样策略
     * @param random 随机数生成器
     */
    public ParticleFilter(int numParticles, ResamplingStrategy resamplingStrategy, Random random) {
        this.numParticles = numParticles;
        this.resamplingStrategy = resamplingStrategy;
        this.random = random;
    }
    
    /**
     * 粒子
     * Particle
     */
    public static class Particle {
        private IVector state;
        private double weight;
        
        public Particle(IVector state, double weight) {
            this.state = state;
            this.weight = weight;
        }
        
        public IVector getState() { return state; }
        public void setState(IVector state) { this.state = state; }
        public double getWeight() { return weight; }
        public void setWeight(double weight) { this.weight = weight; }
        
        public Particle copy() {
            IVector stateCopy = Linalg.vector(state.size());
            for (int i = 0; i < state.size(); i++) {
                stateCopy.set(i, state.get(i));
            }
            return new Particle(stateCopy, weight);
        }
    }
    
    /**
     * 粒子滤波结果
     * Particle filter result
     */
    public static class FilterResult {
        private final List<Particle> particles;
        private final IVector stateEstimate;
        private final IVector stateCovariance;
        private final double effectiveSampleSize;
        private final double logLikelihood;
        
        public FilterResult(List<Particle> particles, IVector stateEstimate, 
                          IVector stateCovariance, double effectiveSampleSize, 
                          double logLikelihood) {
            this.particles = particles;
            this.stateEstimate = stateEstimate;
            this.stateCovariance = stateCovariance;
            this.effectiveSampleSize = effectiveSampleSize;
            this.logLikelihood = logLikelihood;
        }
        
        public List<Particle> getParticles() { return particles; }
        public IVector getStateEstimate() { return stateEstimate; }
        public IVector getStateCovariance() { return stateCovariance; }
        public double getEffectiveSampleSize() { return effectiveSampleSize; }
        public double getLogLikelihood() { return logLikelihood; }
    }
    
    /**
     * 状态转移模型接口
     * State transition model interface
     */
    @FunctionalInterface
    public interface StateTransitionModel {
        /**
         * 状态转移
         * State transition
         * 
         * @param previousState 前一状态
         * @param timeStep 时间步长
         * @return 新状态
         */
        IVector transition(IVector previousState, double timeStep);
    }
    
    /**
     * 观测模型接口
     * Observation model interface
     */
    @FunctionalInterface
    public interface ObservationModel {
        /**
         * 计算观测似然
         * Calculate observation likelihood
         * 
         * @param state 状态
         * @param observation 观测值
         * @return 似然值
         */
        double likelihood(IVector state, IVector observation);
    }
    
    /**
     * 初始化粒子
     * Initialize particles
     * 
     * @param initialDistribution 初始分布采样函数
     * @return 初始化的粒子列表
     */
    public List<Particle> initializeParticles(Function<Random, IVector> initialDistribution) {
        List<Particle> particles = new ArrayList<>();
        double uniformWeight = 1.0 / numParticles;
        
        for (int i = 0; i < numParticles; i++) {
            IVector state = initialDistribution.apply(random);
            particles.add(new Particle(state, uniformWeight));
        }
        
        return particles;
    }
    
    /**
     * 预测步骤
     * Prediction step
     * 
     * @param particles 当前粒子
     * @param transitionModel 状态转移模型
     * @param timeStep 时间步长
     * @return 预测后的粒子
     */
    public List<Particle> predict(List<Particle> particles, 
                                StateTransitionModel transitionModel, 
                                double timeStep) {
        List<Particle> predictedParticles = new ArrayList<>();
        
        for (Particle particle : particles) {
            IVector newState = transitionModel.transition(particle.getState(), timeStep);
            predictedParticles.add(new Particle(newState, particle.getWeight()));
        }
        
        return predictedParticles;
    }
    
    /**
     * 更新步骤
     * Update step
     * 
     * @param particles 预测的粒子
     * @param observation 观测值
     * @param observationModel 观测模型
     * @return 更新后的粒子
     */
    public List<Particle> update(List<Particle> particles, 
                               IVector observation, 
                               ObservationModel observationModel) {
        List<Particle> updatedParticles = new ArrayList<>();
        double totalWeight = 0.0;
        
        // 计算似然权重
        for (Particle particle : particles) {
            double likelihood = observationModel.likelihood(particle.getState(), observation);
            double newWeight = particle.getWeight() * likelihood;
            updatedParticles.add(new Particle(particle.getState(), newWeight));
            totalWeight += newWeight;
        }
        
        // 归一化权重
        if (totalWeight > 0) {
            for (Particle particle : updatedParticles) {
                particle.setWeight(particle.getWeight() / totalWeight);
            }
        } else {
            // 如果所有权重为0，重置为均匀权重
            double uniformWeight = 1.0 / numParticles;
            for (Particle particle : updatedParticles) {
                particle.setWeight(uniformWeight);
            }
        }
        
        return updatedParticles;
    }
    
    /**
     * 重采样
     * Resampling
     * 
     * @param particles 当前粒子
     * @return 重采样后的粒子
     */
    public List<Particle> resample(List<Particle> particles) {
        double ess = calculateEffectiveSampleSize(particles);
        
        // 只有当有效样本大小低于阈值时才重采样
        if (ess < numParticles / 2.0) {
            switch (resamplingStrategy) {
                case MULTINOMIAL:
                    return multinomialResampling(particles);
                case SYSTEMATIC:
                    return systematicResampling(particles);
                case STRATIFIED:
                    return stratifiedResampling(particles);
                case RESIDUAL:
                    return residualResampling(particles);
                default:
                    return systematicResampling(particles);
            }
        } else {
            // 不需要重采样，返回原粒子的副本
            List<Particle> resampledParticles = new ArrayList<>();
            for (Particle particle : particles) {
                resampledParticles.add(particle.copy());
            }
            return resampledParticles;
        }
    }
    
    /**
     * 执行完整的滤波步骤
     * Perform complete filtering step
     * 
     * @param particles 当前粒子
     * @param observation 观测值
     * @param transitionModel 状态转移模型
     * @param observationModel 观测模型
     * @param timeStep 时间步长
     * @return 滤波结果
     */
    public FilterResult filter(List<Particle> particles, 
                             IVector observation,
                             StateTransitionModel transitionModel,
                             ObservationModel observationModel,
                             double timeStep) {
        
        // 预测步骤
        List<Particle> predictedParticles = predict(particles, transitionModel, timeStep);
        
        // 更新步骤
        List<Particle> updatedParticles = update(predictedParticles, observation, observationModel);
        
        // 计算状态估计和协方差
        IVector stateEstimate = calculateStateEstimate(updatedParticles);
        IVector stateCovariance = calculateStateCovariance(updatedParticles, stateEstimate);
        
        // 计算有效样本大小
        double ess = calculateEffectiveSampleSize(updatedParticles);
        
        // 计算对数似然
        double logLikelihood = calculateLogLikelihood(updatedParticles);
        
        // 重采样
        List<Particle> resampledParticles = resample(updatedParticles);
        
        return new FilterResult(resampledParticles, stateEstimate, stateCovariance, 
                              ess, logLikelihood);
    }
    
    /**
     * 多项式重采样
     * Multinomial resampling
     */
    private List<Particle> multinomialResampling(List<Particle> particles) {
        List<Particle> resampledParticles = new ArrayList<>();
        double[] cumulativeWeights = calculateCumulativeWeights(particles);
        
        for (int i = 0; i < numParticles; i++) {
            double u = random.nextDouble();
            int index = findIndex(cumulativeWeights, u);
            Particle selectedParticle = particles.get(index).copy();
            selectedParticle.setWeight(1.0 / numParticles);
            resampledParticles.add(selectedParticle);
        }
        
        return resampledParticles;
    }
    
    /**
     * 系统重采样
     * Systematic resampling
     */
    private List<Particle> systematicResampling(List<Particle> particles) {
        List<Particle> resampledParticles = new ArrayList<>();
        double[] cumulativeWeights = calculateCumulativeWeights(particles);
        
        double u0 = random.nextDouble() / numParticles;
        
        for (int i = 0; i < numParticles; i++) {
            double u = u0 + (double) i / numParticles;
            int index = findIndex(cumulativeWeights, u);
            Particle selectedParticle = particles.get(index).copy();
            selectedParticle.setWeight(1.0 / numParticles);
            resampledParticles.add(selectedParticle);
        }
        
        return resampledParticles;
    }
    
    /**
     * 分层重采样
     * Stratified resampling
     */
    private List<Particle> stratifiedResampling(List<Particle> particles) {
        List<Particle> resampledParticles = new ArrayList<>();
        double[] cumulativeWeights = calculateCumulativeWeights(particles);
        
        for (int i = 0; i < numParticles; i++) {
            double u = (i + random.nextDouble()) / numParticles;
            int index = findIndex(cumulativeWeights, u);
            Particle selectedParticle = particles.get(index).copy();
            selectedParticle.setWeight(1.0 / numParticles);
            resampledParticles.add(selectedParticle);
        }
        
        return resampledParticles;
    }
    
    /**
     * 残差重采样
     * Residual resampling
     */
    private List<Particle> residualResampling(List<Particle> particles) {
        List<Particle> resampledParticles = new ArrayList<>();
        
        // 计算每个粒子的复制次数
        int[] copies = new int[particles.size()];
        double[] residualWeights = new double[particles.size()];
        int totalCopies = 0;
        
        for (int i = 0; i < particles.size(); i++) {
            double weight = particles.get(i).getWeight();
            copies[i] = (int) Math.floor(numParticles * weight);
            residualWeights[i] = numParticles * weight - copies[i];
            totalCopies += copies[i];
        }
        
        // 确定性复制
        for (int i = 0; i < particles.size(); i++) {
            for (int j = 0; j < copies[i]; j++) {
                Particle copiedParticle = particles.get(i).copy();
                copiedParticle.setWeight(1.0 / numParticles);
                resampledParticles.add(copiedParticle);
            }
        }
        
        // 随机复制剩余部分
        int remainingCopies = numParticles - totalCopies;
        if (remainingCopies > 0) {
            // 归一化残差权重
            double totalResidualWeight = Arrays.stream(residualWeights).sum();
            if (totalResidualWeight > 0) {
                for (int i = 0; i < residualWeights.length; i++) {
                    residualWeights[i] /= totalResidualWeight;
                }
            }
            
            // 使用多项式重采样处理残差
            double[] cumulativeResidualWeights = new double[residualWeights.length];
            cumulativeResidualWeights[0] = residualWeights[0];
            for (int i = 1; i < residualWeights.length; i++) {
                cumulativeResidualWeights[i] = cumulativeResidualWeights[i-1] + residualWeights[i];
            }
            
            for (int i = 0; i < remainingCopies; i++) {
                double u = random.nextDouble();
                int index = findIndex(cumulativeResidualWeights, u);
                Particle selectedParticle = particles.get(index).copy();
                selectedParticle.setWeight(1.0 / numParticles);
                resampledParticles.add(selectedParticle);
            }
        }
        
        return resampledParticles;
    }
    
    /**
     * 计算累积权重
     * Calculate cumulative weights
     */
    private double[] calculateCumulativeWeights(List<Particle> particles) {
        double[] cumulativeWeights = new double[particles.size()];
        cumulativeWeights[0] = particles.get(0).getWeight();
        
        for (int i = 1; i < particles.size(); i++) {
            cumulativeWeights[i] = cumulativeWeights[i-1] + particles.get(i).getWeight();
        }
        
        return cumulativeWeights;
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
    
    /**
     * 计算状态估计
     * Calculate state estimate
     */
    private IVector calculateStateEstimate(List<Particle> particles) {
        if (particles.isEmpty()) {
            return Linalg.vector(0);
        }
        
        int stateDim = particles.get(0).getState().size();
        IVector estimate = Linalg.vector(stateDim);
        
        for (int i = 0; i < stateDim; i++) {
            double sum = 0.0;
            for (Particle particle : particles) {
                sum += particle.getWeight() * particle.getState().get(i).doubleValue();
            }
            estimate.set(i, sum);
        }
        
        return estimate;
    }
    
    /**
     * 计算状态协方差
     * Calculate state covariance
     */
    private IVector calculateStateCovariance(List<Particle> particles, IVector mean) {
        if (particles.isEmpty()) {
            return Linalg.vector(0);
        }
        
        int stateDim = particles.get(0).getState().size();
        IVector covariance = Linalg.vector(stateDim);
        
        for (int i = 0; i < stateDim; i++) {
            double variance = 0.0;
            for (Particle particle : particles) {
                double diff = particle.getState().get(i).doubleValue() - mean.get(i).doubleValue();
                variance += particle.getWeight() * diff * diff;
            }
            covariance.set(i, variance);
        }
        
        return covariance;
    }
    
    /**
     * 计算有效样本大小
     * Calculate effective sample size
     */
    private double calculateEffectiveSampleSize(List<Particle> particles) {
        double sumSquaredWeights = 0.0;
        
        for (Particle particle : particles) {
            double weight = particle.getWeight();
            sumSquaredWeights += weight * weight;
        }
        
        return 1.0 / sumSquaredWeights;
    }
    
    /**
     * 计算对数似然
     * Calculate log likelihood
     */
    private double calculateLogLikelihood(List<Particle> particles) {
        double totalWeight = 0.0;
        
        for (Particle particle : particles) {
            totalWeight += particle.getWeight();
        }
        
        return Math.log(totalWeight / numParticles);
    }
}