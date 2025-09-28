package com.reremouse.lab.math.stats.bayes.inference;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

/**
 * 近似贝叶斯计算
 * Approximate Bayesian Computation (ABC)
 * 
 * <p>近似贝叶斯计算是一种无似然推断方法，通过比较观测数据和模拟数据的摘要统计量
 * 来近似后验分布。特别适用于似然函数难以计算或不可用的复杂模型。</p>
 * 
 * <p>Approximate Bayesian Computation is a likelihood-free inference method 
 * that approximates posterior distributions by comparing summary statistics 
 * of observed and simulated data. Particularly suitable for complex models 
 * where likelihood functions are intractable or unavailable.</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ApproximateBayesianComputation {
    
    private final Random random;
    
    /**
     * 构造函数
     */
    public ApproximateBayesianComputation() {
        this(new Random());
    }
    
    /**
     * 构造函数
     * 
     * @param random 随机数生成器
     */
    public ApproximateBayesianComputation(Random random) {
        this.random = random;
    }
    
    /**
     * ABC样本
     * ABC Sample
     */
    public static class ABCSample {
        private final IVector parameter;
        private final IVector summaryStatistics;
        private final double distance;
        private double weight;
        
        public ABCSample(IVector parameter, IVector summaryStatistics, double distance) {
            this.parameter = parameter;
            this.summaryStatistics = summaryStatistics;
            this.distance = distance;
            this.weight = 1.0;
        }
        
        public IVector getParameter() { return parameter; }
        public IVector getSummaryStatistics() { return summaryStatistics; }
        public double getDistance() { return distance; }
        public double getWeight() { return weight; }
        public void setWeight(double weight) { this.weight = weight; }
    }
    
    /**
     * ABC结果
     * ABC Result
     */
    public static class ABCResult {
        private final List<ABCSample> acceptedSamples;
        private final IVector posteriorMean;
        private final IVector posteriorVariance;
        private final double acceptanceRate;
        private final double threshold;
        private final int totalSimulations;
        
        public ABCResult(List<ABCSample> acceptedSamples, IVector posteriorMean,
                        IVector posteriorVariance, double acceptanceRate,
                        double threshold, int totalSimulations) {
            this.acceptedSamples = acceptedSamples;
            this.posteriorMean = posteriorMean;
            this.posteriorVariance = posteriorVariance;
            this.acceptanceRate = acceptanceRate;
            this.threshold = threshold;
            this.totalSimulations = totalSimulations;
        }
        
        public List<ABCSample> getAcceptedSamples() { return acceptedSamples; }
        public IVector getPosteriorMean() { return posteriorMean; }
        public IVector getPosteriorVariance() { return posteriorVariance; }
        public double getAcceptanceRate() { return acceptanceRate; }
        public double getThreshold() { return threshold; }
        public int getTotalSimulations() { return totalSimulations; }
        
        public IVector getPosteriorQuantiles(double[] quantiles) {
            if (acceptedSamples.isEmpty()) {
                return Linalg.vector(0);
            }
            
            int paramDim = acceptedSamples.get(0).getParameter().size();
            IVector result = Linalg.vector(paramDim * quantiles.length);
            
            for (int dim = 0; dim < paramDim; dim++) {
                List<Double> values = new ArrayList<>();
                for (ABCSample sample : acceptedSamples) {
                    values.add(sample.getParameter().get(dim).doubleValue());
                }
                Collections.sort(values);
                
                for (int q = 0; q < quantiles.length; q++) {
                    int index = (int) Math.floor(quantiles[q] * (values.size() - 1));
                    result.set(dim * quantiles.length + q, values.get(index));
                }
            }
            
            return result;
        }
    }
    
    /**
     * 先验分布接口
     * Prior distribution interface
     */
    @FunctionalInterface
    public interface PriorSampler {
        /**
         * 从先验分布采样
         * Sample from prior distribution
         * 
         * @return 参数样本
         */
        IVector sample();
    }
    
    /**
     * 模拟器接口
     * Simulator interface
     */
    @FunctionalInterface
    public interface Simulator {
        /**
         * 模拟数据
         * Simulate data
         * 
         * @param parameter 参数
         * @return 模拟数据
         */
        IVector simulate(IVector parameter);
    }
    
    /**
     * 摘要统计量接口
     * Summary statistics interface
     */
    @FunctionalInterface
    public interface SummaryStatistics {
        /**
         * 计算摘要统计量
         * Calculate summary statistics
         * 
         * @param data 数据
         * @return 摘要统计量
         */
        IVector calculate(IVector data);
    }
    
    /**
     * 距离函数接口
     * Distance function interface
     */
    @FunctionalInterface
    public interface DistanceFunction {
        /**
         * 计算距离
         * Calculate distance
         * 
         * @param observed 观测摘要统计量
         * @param simulated 模拟摘要统计量
         * @return 距离
         */
        double distance(IVector observed, IVector simulated);
    }
    
    /**
     * ABC拒绝采样
     * ABC Rejection Sampling
     * 
     * @param observedData 观测数据
     * @param priorSampler 先验采样器
     * @param simulator 模拟器
     * @param summaryStats 摘要统计量函数
     * @param distanceFunction 距离函数
     * @param threshold 接受阈值
     * @param maxSimulations 最大模拟次数
     * @return ABC结果
     */
    public ABCResult rejectionSampling(IVector observedData,
                                     PriorSampler priorSampler,
                                     Simulator simulator,
                                     SummaryStatistics summaryStats,
                                     DistanceFunction distanceFunction,
                                     double threshold,
                                     int maxSimulations) {
        
        IVector observedSummary = summaryStats.calculate(observedData);
        List<ABCSample> acceptedSamples = new ArrayList<>();
        
        for (int i = 0; i < maxSimulations; i++) {
            // 从先验采样参数
            IVector parameter = priorSampler.sample();
            
            // 模拟数据
            IVector simulatedData = simulator.simulate(parameter);
            
            // 计算摘要统计量
            IVector simulatedSummary = summaryStats.calculate(simulatedData);
            
            // 计算距离
            double distance = distanceFunction.distance(observedSummary, simulatedSummary);
            
            // 接受/拒绝
            if (distance <= threshold) {
                acceptedSamples.add(new ABCSample(parameter, simulatedSummary, distance));
            }
        }
        
        // 计算后验统计量
        IVector posteriorMean = calculatePosteriorMean(acceptedSamples);
        IVector posteriorVariance = calculatePosteriorVariance(acceptedSamples, posteriorMean);
        double acceptanceRate = (double) acceptedSamples.size() / maxSimulations;
        
        return new ABCResult(acceptedSamples, posteriorMean, posteriorVariance,
                           acceptanceRate, threshold, maxSimulations);
    }
    
    /**
     * ABC-SMC (序贯蒙特卡洛ABC)
     * ABC-SMC (Sequential Monte Carlo ABC)
     * 
     * @param observedData 观测数据
     * @param priorSampler 先验采样器
     * @param simulator 模拟器
     * @param summaryStats 摘要统计量函数
     * @param distanceFunction 距离函数
     * @param initialThreshold 初始阈值
     * @param thresholdSchedule 阈值调度
     * @param numParticles 粒子数量
     * @param maxIterations 最大迭代次数
     * @return ABC结果
     */
    public ABCResult sequentialMonteCarlo(IVector observedData,
                                        PriorSampler priorSampler,
                                        Simulator simulator,
                                        SummaryStatistics summaryStats,
                                        DistanceFunction distanceFunction,
                                        double initialThreshold,
                                        Function<Integer, Double> thresholdSchedule,
                                        int numParticles,
                                        int maxIterations) {
        
        IVector observedSummary = summaryStats.calculate(observedData);
        List<ABCSample> particles = new ArrayList<>();
        
        // 初始化：使用拒绝采样
        double currentThreshold = initialThreshold;
        int totalSimulations = 0;
        
        while (particles.size() < numParticles && totalSimulations < maxIterations * 1000) {
            IVector parameter = priorSampler.sample();
            IVector simulatedData = simulator.simulate(parameter);
            IVector simulatedSummary = summaryStats.calculate(simulatedData);
            double distance = distanceFunction.distance(observedSummary, simulatedSummary);
            
            if (distance <= currentThreshold) {
                particles.add(new ABCSample(parameter, simulatedSummary, distance));
            }
            totalSimulations++;
        }
        
        // 序贯更新
        for (int iteration = 1; iteration < maxIterations; iteration++) {
            currentThreshold = thresholdSchedule.apply(iteration);
            
            // 计算权重
            updateWeights(particles, currentThreshold);
            
            // 重采样
            particles = resampleParticles(particles, numParticles);
            
            // 扰动和模拟
            List<ABCSample> newParticles = new ArrayList<>();
            for (ABCSample particle : particles) {
                IVector perturbedParameter = perturbParameter(particle.getParameter());
                IVector simulatedData = simulator.simulate(perturbedParameter);
                IVector simulatedSummary = summaryStats.calculate(simulatedData);
                double distance = distanceFunction.distance(observedSummary, simulatedSummary);
                
                if (distance <= currentThreshold) {
                    newParticles.add(new ABCSample(perturbedParameter, simulatedSummary, distance));
                }
                totalSimulations++;
            }
            
            particles = newParticles;
            
            // 检查收敛
            if (particles.size() < numParticles * 0.1) {
                break;
            }
        }
        
        IVector posteriorMean = calculatePosteriorMean(particles);
        IVector posteriorVariance = calculatePosteriorVariance(particles, posteriorMean);
        double acceptanceRate = (double) particles.size() / totalSimulations;
        
        return new ABCResult(particles, posteriorMean, posteriorVariance,
                           acceptanceRate, currentThreshold, totalSimulations);
    }
    
    /**
     * 自适应ABC
     * Adaptive ABC
     * 
     * @param observedData 观测数据
     * @param priorSampler 先验采样器
     * @param simulator 模拟器
     * @param summaryStats 摘要统计量函数
     * @param distanceFunction 距离函数
     * @param targetAcceptanceRate 目标接受率
     * @param numSamples 目标样本数
     * @param maxSimulations 最大模拟次数
     * @return ABC结果
     */
    public ABCResult adaptiveABC(IVector observedData,
                               PriorSampler priorSampler,
                               Simulator simulator,
                               SummaryStatistics summaryStats,
                               DistanceFunction distanceFunction,
                               double targetAcceptanceRate,
                               int numSamples,
                               int maxSimulations) {
        
        IVector observedSummary = summaryStats.calculate(observedData);
        List<ABCSample> allSamples = new ArrayList<>();
        
        // 初始阶段：收集样本以估计距离分布
        int initialSamples = Math.min(1000, maxSimulations / 10);
        for (int i = 0; i < initialSamples; i++) {
            IVector parameter = priorSampler.sample();
            IVector simulatedData = simulator.simulate(parameter);
            IVector simulatedSummary = summaryStats.calculate(simulatedData);
            double distance = distanceFunction.distance(observedSummary, simulatedSummary);
            allSamples.add(new ABCSample(parameter, simulatedSummary, distance));
        }
        
        // 估计自适应阈值
        List<Double> distances = new ArrayList<>();
        for (ABCSample sample : allSamples) {
            distances.add(sample.getDistance());
        }
        Collections.sort(distances);
        
        int targetIndex = (int) (targetAcceptanceRate * distances.size());
        double adaptiveThreshold = distances.get(Math.min(targetIndex, distances.size() - 1));
        
        // 继续采样直到达到目标样本数
        int totalSimulations = initialSamples;
        while (totalSimulations < maxSimulations) {
            IVector parameter = priorSampler.sample();
            IVector simulatedData = simulator.simulate(parameter);
            IVector simulatedSummary = summaryStats.calculate(simulatedData);
            double distance = distanceFunction.distance(observedSummary, simulatedSummary);
            
            allSamples.add(new ABCSample(parameter, simulatedSummary, distance));
            totalSimulations++;
            
            // 定期更新阈值
            if (totalSimulations % 100 == 0) {
                distances.clear();
                for (ABCSample sample : allSamples) {
                    distances.add(sample.getDistance());
                }
                Collections.sort(distances);
                targetIndex = (int) (targetAcceptanceRate * distances.size());
                adaptiveThreshold = distances.get(Math.min(targetIndex, distances.size() - 1));
            }
        }
        
        // 筛选接受的样本
        List<ABCSample> acceptedSamples = new ArrayList<>();
        for (ABCSample sample : allSamples) {
            if (sample.getDistance() <= adaptiveThreshold) {
                acceptedSamples.add(sample);
            }
        }
        
        // 如果样本太多，随机选择
        if (acceptedSamples.size() > numSamples) {
            Collections.shuffle(acceptedSamples, random);
            acceptedSamples = acceptedSamples.subList(0, numSamples);
        }
        
        IVector posteriorMean = calculatePosteriorMean(acceptedSamples);
        IVector posteriorVariance = calculatePosteriorVariance(acceptedSamples, posteriorMean);
        double acceptanceRate = (double) acceptedSamples.size() / totalSimulations;
        
        return new ABCResult(acceptedSamples, posteriorMean, posteriorVariance,
                           acceptanceRate, adaptiveThreshold, totalSimulations);
    }
    
    /**
     * 更新权重
     * Update weights
     */
    private void updateWeights(List<ABCSample> particles, double threshold) {
        for (ABCSample particle : particles) {
            if (particle.getDistance() <= threshold) {
                // 简单的核权重：距离越小权重越大
                double weight = Math.exp(-particle.getDistance() / threshold);
                particle.setWeight(weight);
            } else {
                particle.setWeight(0.0);
            }
        }
        
        // 归一化权重
        double totalWeight = 0.0;
        for (ABCSample particle : particles) {
            totalWeight += particle.getWeight();
        }
        
        if (totalWeight > 0) {
            for (ABCSample particle : particles) {
                particle.setWeight(particle.getWeight() / totalWeight);
            }
        }
    }
    
    /**
     * 重采样粒子
     * Resample particles
     */
    private List<ABCSample> resampleParticles(List<ABCSample> particles, int numParticles) {
        List<ABCSample> resampledParticles = new ArrayList<>();
        
        // 计算累积权重
        double[] cumulativeWeights = new double[particles.size()];
        cumulativeWeights[0] = particles.get(0).getWeight();
        for (int i = 1; i < particles.size(); i++) {
            cumulativeWeights[i] = cumulativeWeights[i-1] + particles.get(i).getWeight();
        }
        
        // 系统重采样
        double u0 = random.nextDouble() / numParticles;
        for (int i = 0; i < numParticles; i++) {
            double u = u0 + (double) i / numParticles;
            int index = findIndex(cumulativeWeights, u);
            ABCSample selected = particles.get(index);
            resampledParticles.add(new ABCSample(
                copyVector(selected.getParameter()),
                copyVector(selected.getSummaryStatistics()),
                selected.getDistance()
            ));
        }
        
        return resampledParticles;
    }
    
    /**
     * 扰动参数
     * Perturb parameter
     */
    private IVector perturbParameter(IVector parameter) {
        IVector perturbedParameter = Linalg.vector(parameter.size());
        double perturbationScale = 0.1; // 可以自适应调整
        
        for (int i = 0; i < parameter.size(); i++) {
            double noise = perturbationScale * random.nextGaussian();
            perturbedParameter.set(i, parameter.get(i).doubleValue() + noise);
        }
        
        return perturbedParameter;
    }
    
    /**
     * 计算后验均值
     * Calculate posterior mean
     */
    private IVector calculatePosteriorMean(List<ABCSample> samples) {
        if (samples.isEmpty()) {
            return Linalg.vector(0);
        }
        
        int paramDim = samples.get(0).getParameter().size();
        IVector mean = Linalg.vector(paramDim);
        
        for (int i = 0; i < paramDim; i++) {
            double sum = 0.0;
            double totalWeight = 0.0;
            
            for (ABCSample sample : samples) {
                sum += sample.getWeight() * sample.getParameter().get(i).doubleValue();
                totalWeight += sample.getWeight();
            }
            
            mean.set(i, totalWeight > 0 ? sum / totalWeight : 0.0);
        }
        
        return mean;
    }
    
    /**
     * 计算后验方差
     * Calculate posterior variance
     */
    private IVector calculatePosteriorVariance(List<ABCSample> samples, IVector mean) {
        if (samples.isEmpty()) {
            return Linalg.vector(0);
        }
        
        int paramDim = samples.get(0).getParameter().size();
        IVector variance = Linalg.vector(paramDim);
        
        for (int i = 0; i < paramDim; i++) {
            double sumSquaredDiff = 0.0;
            double totalWeight = 0.0;
            
            for (ABCSample sample : samples) {
                double diff = sample.getParameter().get(i).doubleValue() - mean.get(i).doubleValue();
                sumSquaredDiff += sample.getWeight() * diff * diff;
                totalWeight += sample.getWeight();
            }
            
            variance.set(i, totalWeight > 0 ? sumSquaredDiff / totalWeight : 0.0);
        }
        
        return variance;
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
     * 复制向量
     * Copy vector
     */
    private IVector copyVector(IVector vector) {
        IVector copy = Linalg.vector(vector.size());
        for (int i = 0; i < vector.size(); i++) {
            copy.set(i, vector.get(i));
        }
        return copy;
    }
    
    /**
     * 常用距离函数
     * Common distance functions
     */
    public static class DistanceFunctions {
        
        /**
         * 欧几里得距离
         * Euclidean distance
         */
        public static final DistanceFunction EUCLIDEAN = (observed, simulated) -> {
            double sum = 0.0;
            for (int i = 0; i < observed.size(); i++) {
                double diff = observed.get(i).doubleValue() - simulated.get(i).doubleValue();
                sum += diff * diff;
            }
            return Math.sqrt(sum);
        };
        
        /**
         * 曼哈顿距离
         * Manhattan distance
         */
        public static final DistanceFunction MANHATTAN = (observed, simulated) -> {
            double sum = 0.0;
            for (int i = 0; i < observed.size(); i++) {
                double diff = Math.abs(observed.get(i).doubleValue() - simulated.get(i).doubleValue());
                sum += diff;
            }
            return sum;
        };
        
        /**
         * 最大距离
         * Maximum distance
         */
        public static final DistanceFunction MAXIMUM = (observed, simulated) -> {
            double max = 0.0;
            for (int i = 0; i < observed.size(); i++) {
                double diff = Math.abs(observed.get(i).doubleValue() - simulated.get(i).doubleValue());
                max = Math.max(max, diff);
            }
            return max;
        };
        
        /**
         * 加权欧几里得距离
         * Weighted Euclidean distance
         */
        public static DistanceFunction weightedEuclidean(IVector weights) {
            return (observed, simulated) -> {
                double sum = 0.0;
                for (int i = 0; i < observed.size(); i++) {
                    double diff = observed.get(i).doubleValue() - simulated.get(i).doubleValue();
                    double weight = i < weights.size() ? weights.get(i).doubleValue() : 1.0;
                    sum += weight * diff * diff;
                }
                return Math.sqrt(sum);
            };
        }
    }
}