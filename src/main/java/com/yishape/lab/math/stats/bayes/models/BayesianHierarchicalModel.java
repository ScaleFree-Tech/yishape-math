package com.yishape.lab.math.stats.bayes.models;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.stats.bayes.mcmc.GibbsSampler;
import com.yishape.lab.math.stats.distribution.NormalDistribution;
import com.yishape.lab.math.stats.distribution.GammaDistribution;

import java.util.Random;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;

/**
 * 贝叶斯层次模型
 * Bayesian Hierarchical Model
 * 
 * <p>实现贝叶斯层次模型，支持多层次的参数结构和组间变异性建模。
 * 适用于具有分组结构的数据，如多中心研究、重复测量数据等。</p>
 * <p>Implements Bayesian hierarchical model with support for multi-level parameter structures 
 * and between-group variability modeling. Suitable for grouped data such as multi-center studies, 
 * repeated measures data, etc.</p>
 * 
 * <p>模型结构：
 * Level 1 (观测层): y_ij | θ_i, σ² ~ N(θ_i, σ²)
 * Level 2 (组层): θ_i | μ, τ² ~ N(μ, τ²)
 * Level 3 (超参数层): μ ~ N(μ₀, σ₀²), τ² ~ InvGamma(α, β), σ² ~ InvGamma(α', β')</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class BayesianHierarchicalModel {
    
    // 数据结构
    private Map<Integer, List<Double>> groupData;  // 分组数据
    private int numGroups;
    private int totalObservations;
    
    // 模型参数
    private IVector groupMeans;      // θ_i: 组均值
    private double populationMean;   // μ: 总体均值
    private double betweenGroupVar;  // τ²: 组间方差
    private double withinGroupVar;   // σ²: 组内方差
    
    // 超参数
    private double priorMean;        // μ₀
    private double priorMeanVar;     // σ₀²
    private double priorBetweenAlpha; // α (for τ²)
    private double priorBetweenBeta;  // β (for τ²)
    private double priorWithinAlpha;  // α' (for σ²)
    private double priorWithinBeta;   // β' (for σ²)
    
    // 推断结果
    private List<HierarchicalSample> posteriorSamples;
    private boolean inferenceCompleted;
    
    /**
     * 构造函数
     * Constructor
     */
    public BayesianHierarchicalModel() {
        this.groupData = new HashMap<>();
        this.numGroups = 0;
        this.totalObservations = 0;
        this.inferenceCompleted = false;
        
        // 设置默认超参数
        setDefaultHyperparameters();
    }
    
    /**
     * 设置默认超参数
     * Set default hyperparameters
     */
    private void setDefaultHyperparameters() {
        this.priorMean = 0.0;
        this.priorMeanVar = 100.0;
        this.priorBetweenAlpha = 1.0;
        this.priorBetweenBeta = 1.0;
        this.priorWithinAlpha = 1.0;
        this.priorWithinBeta = 1.0;
    }
    
    /**
     * 设置超参数
     * Set hyperparameters
     * 
     * @param priorMean 总体均值的先验均值 / Prior mean for population mean
     * @param priorMeanVar 总体均值的先验方差 / Prior variance for population mean
     * @param priorBetweenAlpha 组间方差的逆伽马先验形状参数 / Inverse gamma prior shape for between-group variance
     * @param priorBetweenBeta 组间方差的逆伽马先验尺度参数 / Inverse gamma prior scale for between-group variance
     * @param priorWithinAlpha 组内方差的逆伽马先验形状参数 / Inverse gamma prior shape for within-group variance
     * @param priorWithinBeta 组内方差的逆伽马先验尺度参数 / Inverse gamma prior scale for within-group variance
     */
    public void setHyperparameters(double priorMean, double priorMeanVar,
                                  double priorBetweenAlpha, double priorBetweenBeta,
                                  double priorWithinAlpha, double priorWithinBeta) {
        if (priorMeanVar <= 0 || priorBetweenAlpha <= 0 || priorBetweenBeta <= 0 ||
            priorWithinAlpha <= 0 || priorWithinBeta <= 0) {
            throw new IllegalArgumentException("Variance and gamma parameters must be positive");
        }
        
        this.priorMean = priorMean;
        this.priorMeanVar = priorMeanVar;
        this.priorBetweenAlpha = priorBetweenAlpha;
        this.priorBetweenBeta = priorBetweenBeta;
        this.priorWithinAlpha = priorWithinAlpha;
        this.priorWithinBeta = priorWithinBeta;
    }
    
    /**
     * 添加组数据
     * Add group data
     * 
     * @param groupId 组ID / Group ID
     * @param observations 观测值 / Observations
     */
    public void addGroupData(int groupId, double[] observations) {
        if (observations == null || observations.length == 0) {
            throw new IllegalArgumentException("Observations cannot be null or empty");
        }
        
        List<Double> groupObs = new ArrayList<>();
        for (double obs : observations) {
            groupObs.add(obs);
        }
        
        groupData.put(groupId, groupObs);
        updateDataStatistics();
        this.inferenceCompleted = false;
    }
    
    /**
     * 添加组数据
     * Add group data
     * 
     * @param groupId 组ID / Group ID
     * @param observations 观测值列表 / List of observations
     */
    public void addGroupData(int groupId, List<Double> observations) {
        if (observations == null || observations.isEmpty()) {
            throw new IllegalArgumentException("Observations cannot be null or empty");
        }
        
        groupData.put(groupId, new ArrayList<>(observations));
        updateDataStatistics();
        this.inferenceCompleted = false;
    }
    
    /**
     * 更新数据统计量
     * Update data statistics
     */
    private void updateDataStatistics() {
        this.numGroups = groupData.size();
        this.totalObservations = groupData.values().stream()
            .mapToInt(List::size)
            .sum();
    }
    
    /**
     * 使用Gibbs采样进行推断
     * Perform inference using Gibbs sampling
     * 
     * @param numSamples 采样数量 / Number of samples
     * @param burnIn 预热期 / Burn-in period
     * @param random 随机数生成器 / Random number generator
     */
    public void inferGibbs(int numSamples, int burnIn, Random random) {
        if (groupData.isEmpty()) {
            throw new IllegalStateException("No data has been added");
        }
        
        // 初始化参数
        initializeParameters(random);
        
        // 创建Gibbs采样器
        GibbsSampler gibbs = new GibbsSampler();
        
        // 创建条件采样器
        List<GibbsSampler.ConditionalSampler> samplers = createConditionalSamplers();
        
        // 执行Gibbs采样
        this.posteriorSamples = new ArrayList<>();
        
        for (int iter = 0; iter < numSamples + burnIn; iter++) {
            // 采样组均值
            sampleGroupMeans(random);
            
            // 采样总体均值
            samplePopulationMean(random);
            
            // 采样组间方差
            sampleBetweenGroupVariance(random);
            
            // 采样组内方差
            sampleWithinGroupVariance(random);
            
            // 保存样本（跳过预热期）
            if (iter >= burnIn) {
                posteriorSamples.add(new HierarchicalSample(
                    groupMeans.copy(),
                    populationMean,
                    betweenGroupVar,
                    withinGroupVar
                ));
            }
        }
        
        this.inferenceCompleted = true;
    }
    
    /**
     * 初始化参数
     * Initialize parameters
     */
    private void initializeParameters(Random random) {
        // 初始化组均值
        this.groupMeans = Linalg.vector(numGroups);
        int groupIndex = 0;
        for (Map.Entry<Integer, List<Double>> entry : groupData.entrySet()) {
            List<Double> observations = entry.getValue();
            double groupMean = observations.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            groupMeans.set(groupIndex++, groupMean);
        }
        
        // 初始化总体均值
        this.populationMean = (double)groupMeans.mean();
        
        // 初始化方差
        this.betweenGroupVar = 1.0;
        this.withinGroupVar = 1.0;
    }
    
    /**
     * 创建条件采样器
     * Create conditional samplers
     */
    private List<GibbsSampler.ConditionalSampler> createConditionalSamplers() {
        List<GibbsSampler.ConditionalSampler> samplers = new ArrayList<>();
        
        // 组均值采样器
        samplers.add(new GroupMeansSampler());
        
        // 总体均值采样器
        samplers.add(new PopulationMeanSampler());
        
        // 组间方差采样器
        samplers.add(new BetweenGroupVarianceSampler());
        
        // 组内方差采样器
        samplers.add(new WithinGroupVarianceSampler());
        
        return samplers;
    }
    
    /**
     * 采样组均值
     * Sample group means
     */
    private void sampleGroupMeans(Random random) {
        int groupIndex = 0;
        for (Map.Entry<Integer, List<Double>> entry : groupData.entrySet()) {
            List<Double> observations = entry.getValue();
            int n = observations.size();
            double sumObs = observations.stream().mapToDouble(Double::doubleValue).sum();
            
            // 后验参数
            double precision = n / withinGroupVar + 1.0 / betweenGroupVar;
            double mean = (sumObs / withinGroupVar + populationMean / betweenGroupVar) / precision;
            double variance = 1.0 / precision;
            
            // 采样
            NormalDistribution posterior = new NormalDistribution(mean, Math.sqrt(variance));
            groupMeans.set(groupIndex++, posterior.sample(1)[0]);
        }
    }
    
    /**
     * 采样总体均值
     * Sample population mean
     */
    private void samplePopulationMean(Random random) {
        double sumGroupMeans = (double)groupMeans.sum();
        
        // 后验参数
        double precision = numGroups / betweenGroupVar + 1.0 / priorMeanVar;
        double mean = (sumGroupMeans / betweenGroupVar + priorMean / priorMeanVar) / precision;
        double variance = 1.0 / precision;
        
        // 采样
        NormalDistribution posterior = new NormalDistribution(mean, Math.sqrt(variance));
        this.populationMean = posterior.sample(1)[0];
    }
    
    /**
     * 采样组间方差
     * Sample between-group variance
     */
    private void sampleBetweenGroupVariance(Random random) {
        double sumSquaredDeviations = 0.0;
        for (int i = 0; i < numGroups; i++) {
            double deviation = groupMeans.get(i).doubleValue() - populationMean;
            sumSquaredDeviations += deviation * deviation;
        }
        
        // 后验参数
        double shape = priorBetweenAlpha + numGroups / 2.0;
        double rate = priorBetweenBeta + sumSquaredDeviations / 2.0;
        
        // 采样（逆伽马分布 = 1/伽马分布）
        GammaDistribution gamma = new GammaDistribution(shape, 1.0 / rate);
        this.betweenGroupVar = 1.0 / gamma.sample(1)[0];
    }
    
    /**
     * 采样组内方差
     * Sample within-group variance
     */
    private void sampleWithinGroupVariance(Random random) {
        double sumSquaredResiduals = 0.0;
        int totalN = 0;
        
        int groupIndex = 0;
        for (Map.Entry<Integer, List<Double>> entry : groupData.entrySet()) {
            List<Double> observations = entry.getValue();
            double groupMean = groupMeans.get(groupIndex++).doubleValue();
            
            for (double obs : observations) {
                double residual = obs - groupMean;
                sumSquaredResiduals += residual * residual;
            }
            totalN += observations.size();
        }
        
        // 后验参数
        double shape = priorWithinAlpha + totalN / 2.0;
        double rate = priorWithinBeta + sumSquaredResiduals / 2.0;
        
        // 采样（逆伽马分布 = 1/伽马分布）
        GammaDistribution gamma = new GammaDistribution(shape, 1.0 / rate);
        this.withinGroupVar = 1.0 / gamma.sample(1)[0];
    }
    
    /**
     * 预测新组的观测值
     * Predict observations for a new group
     * 
     * @param numPredictions 预测数量 / Number of predictions
     * @param random 随机数生成器 / Random number generator
     * @return 预测样本 / Prediction samples
     */
    public IMatrix predictNewGroup(int numPredictions, Random random) {
        if (!inferenceCompleted) {
            throw new IllegalStateException("Inference must be completed before prediction");
        }
        
        int numSamples = posteriorSamples.size();
        IMatrix predictions = Linalg.zeros(numSamples, numPredictions);
        
        for (int s = 0; s < numSamples; s++) {
            HierarchicalSample sample = posteriorSamples.get(s);
            
            // 从先验预测分布中采样新组的均值
            NormalDistribution groupMeanDist = new NormalDistribution(
                sample.populationMean, Math.sqrt(sample.betweenGroupVar));
            double newGroupMean = groupMeanDist.sample(1)[0];
            
            // 从新组中采样观测值
            NormalDistribution obsDist = new NormalDistribution(
                newGroupMean, Math.sqrt(sample.withinGroupVar));
            
            for (int p = 0; p < numPredictions; p++) {
                predictions.set(s, p, obsDist.sample(1)[0]);
            }
        }
        
        return predictions;
    }
    
    /**
     * 预测现有组的新观测值
     * Predict new observations for existing group
     * 
     * @param groupId 组ID / Group ID
     * @param numPredictions 预测数量 / Number of predictions
     * @param random 随机数生成器 / Random number generator
     * @return 预测样本 / Prediction samples
     */
    public IMatrix predictExistingGroup(int groupId, int numPredictions, Random random) {
        if (!inferenceCompleted) {
            throw new IllegalStateException("Inference must be completed before prediction");
        }
        
        if (!groupData.containsKey(groupId)) {
            throw new IllegalArgumentException("Group ID not found in data");
        }
        
        // 找到组索引
        int groupIndex = 0;
        for (int id : groupData.keySet()) {
            if (id == groupId) break;
            groupIndex++;
        }
        
        int numSamples = posteriorSamples.size();
        IMatrix predictions = Linalg.zeros(numSamples, numPredictions);
        
        for (int s = 0; s < numSamples; s++) {
            HierarchicalSample sample = posteriorSamples.get(s);
            double groupMean = sample.groupMeans.get(groupIndex).doubleValue();
            
            // 从组分布中采样观测值
            NormalDistribution obsDist = new NormalDistribution(
                groupMean, Math.sqrt(sample.withinGroupVar));
            
            for (int p = 0; p < numPredictions; p++) {
                predictions.set(s, p, obsDist.sample(1)[0]);
            }
        }
        
        return predictions;
    }
    
    /**
     * 计算组收缩因子
     * Calculate group shrinkage factors
     * 
     * @return 收缩因子 / Shrinkage factors
     */
    public IVector calculateShrinkageFactors() {
        if (!inferenceCompleted) {
            throw new IllegalStateException("Inference must be completed before calculating shrinkage");
        }
        
        IVector shrinkageFactors = Linalg.vector(numGroups);
        
        // 计算后验均值
        double avgBetweenVar = posteriorSamples.stream()
            .mapToDouble(s -> s.betweenGroupVar)
            .average().orElse(1.0);
        double avgWithinVar = posteriorSamples.stream()
            .mapToDouble(s -> s.withinGroupVar)
            .average().orElse(1.0);
        
        int groupIndex = 0;
        for (Map.Entry<Integer, List<Double>> entry : groupData.entrySet()) {
            int n = entry.getValue().size();
            double shrinkage = avgBetweenVar / (avgBetweenVar + avgWithinVar / n);
            shrinkageFactors.set(groupIndex++, shrinkage);
        }
        
        return shrinkageFactors;
    }
    
    /**
     * 获取后验统计量
     * Get posterior statistics
     * 
     * @return 后验统计量 / Posterior statistics
     */
    public HierarchicalPosteriorStatistics getPosteriorStatistics() {
        if (!inferenceCompleted) {
            throw new IllegalStateException("Inference must be completed before getting statistics");
        }
        
        return new HierarchicalPosteriorStatistics(posteriorSamples, groupData.keySet().toArray(new Integer[0]));
    }
    
    /**
     * 计算DIC（偏差信息准则）
     * Calculate DIC (Deviance Information Criterion)
     * 
     * @return DIC值 / DIC value
     */
    public double calculateDIC() {
        if (!inferenceCompleted) {
            throw new IllegalStateException("Inference must be completed before calculating DIC");
        }
        
        // 计算后验均值参数
        IVector meanGroupMeans = Linalg.vector(numGroups);
        for (int i = 0; i < numGroups; i++) {
            final int index = i;
            double mean = posteriorSamples.stream()
                .mapToDouble(s -> s.groupMeans.get(index).doubleValue())
                .average().orElse(0.0);
            meanGroupMeans.set(i, mean);
        }
        
        double meanWithinVar = posteriorSamples.stream()
            .mapToDouble(s -> s.withinGroupVar)
            .average().orElse(1.0);
        
        // 计算偏差
        double meanDeviance = posteriorSamples.stream()
            .mapToDouble(this::calculateDeviance)
            .average().orElse(0.0);
        
        double devianceAtMean = calculateDeviance(meanGroupMeans, meanWithinVar);
        
        double pD = meanDeviance - devianceAtMean;
        return meanDeviance + pD;
    }
    
    /**
     * 计算偏差
     * Calculate deviance
     */
    private double calculateDeviance(HierarchicalSample sample) {
        return calculateDeviance(sample.groupMeans, sample.withinGroupVar);
    }
    
    /**
     * 计算偏差
     * Calculate deviance
     */
    private double calculateDeviance(IVector groupMeans, double withinVar) {
        double deviance = 0.0;
        
        int groupIndex = 0;
        for (Map.Entry<Integer, List<Double>> entry : groupData.entrySet()) {
            List<Double> observations = entry.getValue();
            double groupMean = groupMeans.get(groupIndex++).doubleValue();
            
            for (double obs : observations) {
                double residual = obs - groupMean;
                deviance += residual * residual / withinVar + Math.log(2 * Math.PI * withinVar);
            }
        }
        
        return deviance;
    }
    
    /**
     * 层次模型样本类
     * Hierarchical model sample class
     */
    public static class HierarchicalSample {
        public final IVector groupMeans;
        public final double populationMean;
        public final double betweenGroupVar;
        public final double withinGroupVar;
        
        public HierarchicalSample(IVector groupMeans, double populationMean, 
                                 double betweenGroupVar, double withinGroupVar) {
            this.groupMeans = groupMeans;
            this.populationMean = populationMean;
            this.betweenGroupVar = betweenGroupVar;
            this.withinGroupVar = withinGroupVar;
        }
    }
    
    /**
     * 层次模型后验统计量类
     * Hierarchical model posterior statistics class
     */
    public static class HierarchicalPosteriorStatistics {
        private final List<HierarchicalSample> samples;
        private final Integer[] groupIds;
        
        public HierarchicalPosteriorStatistics(List<HierarchicalSample> samples, Integer[] groupIds) {
            this.samples = samples;
            this.groupIds = groupIds;
        }
        
        public double getPopulationMeanEstimate() {
            return samples.stream().mapToDouble(s -> s.populationMean).average().orElse(0.0);
        }
        
        public double getPopulationMeanStd() {
            double mean = getPopulationMeanEstimate();
            double variance = samples.stream()
                .mapToDouble(s -> Math.pow(s.populationMean - mean, 2))
                .average().orElse(0.0);
            return Math.sqrt(variance);
        }
        
        public double getBetweenGroupVarianceEstimate() {
            return samples.stream().mapToDouble(s -> s.betweenGroupVar).average().orElse(0.0);
        }
        
        public double getWithinGroupVarianceEstimate() {
            return samples.stream().mapToDouble(s -> s.withinGroupVar).average().orElse(0.0);
        }
        
        public IVector getGroupMeanEstimates() {
            int numGroups = samples.get(0).groupMeans.size();
            IVector estimates = Linalg.vector(numGroups);
            
            for (int i = 0; i < numGroups; i++) {
                final int index = i;
                double mean = samples.stream()
                    .mapToDouble(s -> s.groupMeans.get(index).doubleValue())
                    .average().orElse(0.0);
                estimates.set(i, mean);
            }
            
            return estimates;
        }
        
        public double[] getCredibleInterval(String parameter, double alpha) {
            List<Double> values = new ArrayList<>();
            
            switch (parameter.toLowerCase()) {
                case "population_mean":
                    values = samples.stream().map(s -> s.populationMean).collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
                    break;
                case "between_group_var":
                    values = samples.stream().map(s -> s.betweenGroupVar).collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
                    break;
                case "within_group_var":
                    values = samples.stream().map(s -> s.withinGroupVar).collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
                    break;
                default:
                    throw new IllegalArgumentException("Unknown parameter: " + parameter);
            }
            
            values.sort(Double::compareTo);
            int n = values.size();
            int lowerIndex = (int) Math.floor(alpha / 2 * n);
            int upperIndex = (int) Math.ceil((1 - alpha / 2) * n) - 1;
            
            return new double[] { values.get(lowerIndex), values.get(upperIndex) };
        }
        
        public double getIntraclassCorrelation() {
            double betweenVar = getBetweenGroupVarianceEstimate();
            double withinVar = getWithinGroupVarianceEstimate();
            return betweenVar / (betweenVar + withinVar);
        }
    }
    
    // 条件采样器实现
    private class GroupMeansSampler implements GibbsSampler.ConditionalSampler {
        @Override
        public double sampleConditional(IVector currentState, int parameterIndex) {
            // 重新采样所有组均值
            sampleGroupMeans(new Random());
            return groupMeans.get(parameterIndex).doubleValue();
        }
    }
    
    private class PopulationMeanSampler implements GibbsSampler.ConditionalSampler {
        @Override
        public double sampleConditional(IVector currentState, int parameterIndex) {
            // 重新采样总体均值
            samplePopulationMean(new Random());
            return populationMean;
        }
    }
    
    private class BetweenGroupVarianceSampler implements GibbsSampler.ConditionalSampler {
        @Override
        public double sampleConditional(IVector currentState, int parameterIndex) {
            // 重新采样组间方差
            sampleBetweenGroupVariance(new Random());
            return betweenGroupVar;
        }
    }
    
    private class WithinGroupVarianceSampler implements GibbsSampler.ConditionalSampler {
        @Override
        public double sampleConditional(IVector currentState, int parameterIndex) {
            // 重新采样组内方差
            sampleWithinGroupVariance(new Random());
            return withinGroupVar;
        }
    }
}