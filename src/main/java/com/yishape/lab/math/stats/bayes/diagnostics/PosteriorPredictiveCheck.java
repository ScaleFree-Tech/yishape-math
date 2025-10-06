package com.yishape.lab.math.stats.bayes.diagnostics;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.util.Random;
import java.util.function.Function;

/**
 * 后验预测检验
 * Posterior Predictive Check
 * 
 * <p>后验预测检验是评估贝叶斯模型拟合质量的重要工具，通过比较观测数据和模型预测数据的统计特征来检验模型。</p>
 * <p>Posterior predictive check is an important tool for evaluating Bayesian model fit 
 * by comparing statistical features of observed data and model-predicted data.</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class PosteriorPredictiveCheck {
    
    /**
     * 后验预测检验结果
     * Posterior predictive check result
     */
    public static class PPCResult {
        private final double observedStatistic;
        private final IVector predictiveStatistics;
        private final double pValue;
        private final double bayesianPValue;
        private final boolean modelAdequate;
        
        public PPCResult(double observedStatistic, IVector predictiveStatistics, 
                        double pValue, double bayesianPValue, boolean modelAdequate) {
            this.observedStatistic = observedStatistic;
            this.predictiveStatistics = predictiveStatistics;
            this.pValue = pValue;
            this.bayesianPValue = bayesianPValue;
            this.modelAdequate = modelAdequate;
        }
        
        public double getObservedStatistic() { return observedStatistic; }
        public IVector getPredictiveStatistics() { return predictiveStatistics; }
        public double getPValue() { return pValue; }
        public double getBayesianPValue() { return bayesianPValue; }
        public boolean isModelAdequate() { return modelAdequate; }
        
        @Override
        public String toString() {
            return String.format("PPCResult{observed=%.4f, p-value=%.4f, Bayesian p-value=%.4f, adequate=%s}", 
                               observedStatistic, pValue, bayesianPValue, modelAdequate);
        }
    }
    
    /**
     * 测试统计量接口
     * Test statistic interface
     */
    @FunctionalInterface
    public interface TestStatistic {
        double calculate(IVector data);
    }
    
    /**
     * 数据生成器接口
     * Data generator interface
     */
    @FunctionalInterface
    public interface DataGenerator {
        IVector generate(IVector parameters, int sampleSize, Random random);
    }
    
    /**
     * 执行后验预测检验
     * Perform posterior predictive check
     * 
     * @param observedData 观测数据
     * @param posteriorSamples 后验样本
     * @param dataGenerator 数据生成器
     * @param testStatistic 测试统计量
     * @param random 随机数生成器
     * @return 后验预测检验结果
     */
    public static PPCResult performCheck(IVector observedData, 
                                       IMatrix posteriorSamples,
                                       DataGenerator dataGenerator,
                                       TestStatistic testStatistic,
                                       Random random) {
        
        int numSamples = posteriorSamples.rows();
        int dataSize = observedData.size();
        
        // 计算观测数据的测试统计量
        double observedStatistic = testStatistic.calculate(observedData);
        
        // 生成后验预测数据并计算测试统计量
        IVector predictiveStatistics = Linalg.vector(numSamples);
        
        for (int i = 0; i < numSamples; i++) {
            // 获取第i个后验样本
            IVector parameters = Linalg.vector(posteriorSamples.cols());
            for (int j = 0; j < posteriorSamples.cols(); j++) {
                parameters.set(j, posteriorSamples.get(i, j));
            }
            
            // 生成预测数据
            IVector predictiveData = dataGenerator.generate(parameters, dataSize, random);
            
            // 计算预测数据的测试统计量
            double predictiveStatistic = testStatistic.calculate(predictiveData);
            predictiveStatistics.set(i, predictiveStatistic);
        }
        
        // 计算p值
        double pValue = calculatePValue(observedStatistic, predictiveStatistics);
        
        // 计算贝叶斯p值
        double bayesianPValue = calculateBayesianPValue(observedStatistic, predictiveStatistics);
        
        // 判断模型是否充分（通常p值在0.05-0.95之间认为模型充分）
        boolean modelAdequate = pValue >= 0.05 && pValue <= 0.95;
        
        return new PPCResult(observedStatistic, predictiveStatistics, pValue, bayesianPValue, modelAdequate);
    }
    
    /**
     * 执行多个测试统计量的后验预测检验
     * Perform posterior predictive check with multiple test statistics
     */
    public static PPCResult[] performMultipleChecks(IVector observedData,
                                                   IMatrix posteriorSamples,
                                                   DataGenerator dataGenerator,
                                                   TestStatistic[] testStatistics,
                                                   Random random) {
        
        PPCResult[] results = new PPCResult[testStatistics.length];
        
        for (int i = 0; i < testStatistics.length; i++) {
            results[i] = performCheck(observedData, posteriorSamples, dataGenerator, 
                                    testStatistics[i], random);
        }
        
        return results;
    }
    
    /**
     * 计算经典p值
     * Calculate classical p-value
     */
    private static double calculatePValue(double observedStatistic, IVector predictiveStatistics) {
        int count = 0;
        int total = predictiveStatistics.size();
        
        for (int i = 0; i < total; i++) {
            if (predictiveStatistics.get(i).doubleValue() >= observedStatistic) {
                count++;
            }
        }
        
        return (double) count / total;
    }
    
    /**
     * 计算贝叶斯p值
     * Calculate Bayesian p-value
     */
    private static double calculateBayesianPValue(double observedStatistic, IVector predictiveStatistics) {
        // 贝叶斯p值是双侧的
        int total = predictiveStatistics.size();
        double mean = calculateMean(predictiveStatistics);
        
        int count = 0;
        if (observedStatistic >= mean) {
            // 右侧
            for (int i = 0; i < total; i++) {
                if (predictiveStatistics.get(i).doubleValue() >= observedStatistic) {
                    count++;
                }
            }
        } else {
            // 左侧
            for (int i = 0; i < total; i++) {
                if (predictiveStatistics.get(i).doubleValue() <= observedStatistic) {
                    count++;
                }
            }
        }
        
        return 2.0 * Math.min((double) count / total, 1.0 - (double) count / total);
    }
    
    /**
     * 常用测试统计量
     * Common test statistics
     */
    public static class CommonTestStatistics {
        
        /**
         * 均值
         */
        public static final TestStatistic MEAN = data -> {
            double sum = 0.0;
            for (int i = 0; i < data.size(); i++) {
                sum += data.get(i).doubleValue();
            }
            return sum / data.size();
        };
        
        /**
         * 方差
         */
        public static final TestStatistic VARIANCE = data -> {
            double mean = MEAN.calculate(data);
            double sumSquaredDiff = 0.0;
            for (int i = 0; i < data.size(); i++) {
                double diff = data.get(i).doubleValue() - mean;
                sumSquaredDiff += diff * diff;
            }
            return sumSquaredDiff / (data.size() - 1);
        };
        
        /**
         * 最小值
         */
        public static final TestStatistic MIN = data -> {
            double min = Double.POSITIVE_INFINITY;
            for (int i = 0; i < data.size(); i++) {
                min = Math.min(min, data.get(i).doubleValue());
            }
            return min;
        };
        
        /**
         * 最大值
         */
        public static final TestStatistic MAX = data -> {
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < data.size(); i++) {
                max = Math.max(max, data.get(i).doubleValue());
            }
            return max;
        };
        
        /**
         * 偏度
         */
        public static final TestStatistic SKEWNESS = data -> {
            double mean = MEAN.calculate(data);
            double variance = VARIANCE.calculate(data);
            double std = Math.sqrt(variance);
            
            double sumCubedDiff = 0.0;
            for (int i = 0; i < data.size(); i++) {
                double standardized = (data.get(i).doubleValue() - mean) / std;
                sumCubedDiff += standardized * standardized * standardized;
            }
            
            return sumCubedDiff / data.size();
        };
        
        /**
         * 峰度
         */
        public static final TestStatistic KURTOSIS = data -> {
            double mean = MEAN.calculate(data);
            double variance = VARIANCE.calculate(data);
            double std = Math.sqrt(variance);
            
            double sumFourthPower = 0.0;
            for (int i = 0; i < data.size(); i++) {
                double standardized = (data.get(i).doubleValue() - mean) / std;
                double fourthPower = standardized * standardized * standardized * standardized;
                sumFourthPower += fourthPower;
            }
            
            return sumFourthPower / data.size() - 3.0; // 减去3得到超额峰度
        };
    }
    
    /**
     * 计算向量均值
     */
    private static double calculateMean(IVector vector) {
        double sum = 0.0;
        for (int i = 0; i < vector.size(); i++) {
            sum += vector.get(i).doubleValue();
        }
        return sum / vector.size();
    }
}