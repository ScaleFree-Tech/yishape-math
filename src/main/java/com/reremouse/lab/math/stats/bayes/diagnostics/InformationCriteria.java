package com.reremouse.lab.math.stats.bayes.diagnostics;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;

import java.util.function.BiFunction;

/**
 * 信息准则计算
 * Information Criteria Calculation
 * 
 * <p>实现多种贝叶斯信息准则，包括DIC、WAIC和LOO-CV，用于模型选择和比较。</p>
 * <p>Implements various Bayesian information criteria including DIC, WAIC, and LOO-CV 
 * for model selection and comparison.</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class InformationCriteria {
    
    /**
     * 信息准则结果
     * Information criteria result
     */
    public static class ICResult {
        private final double dic;
        private final double waic;
        private final double looCV;
        private final double effectiveParametersDIC;
        private final double effectiveParametersWAIC;
        private final IVector pointwiseLOO;
        
        public ICResult(double dic, double waic, double looCV, 
                       double effectiveParametersDIC, double effectiveParametersWAIC,
                       IVector pointwiseLOO) {
            this.dic = dic;
            this.waic = waic;
            this.looCV = looCV;
            this.effectiveParametersDIC = effectiveParametersDIC;
            this.effectiveParametersWAIC = effectiveParametersWAIC;
            this.pointwiseLOO = pointwiseLOO;
        }
        
        public double getDIC() { return dic; }
        public double getWAIC() { return waic; }
        public double getLOOCV() { return looCV; }
        public double getEffectiveParametersDIC() { return effectiveParametersDIC; }
        public double getEffectiveParametersWAIC() { return effectiveParametersWAIC; }
        public IVector getPointwiseLOO() { return pointwiseLOO; }
        
        @Override
        public String toString() {
            return String.format("ICResult{DIC=%.2f, WAIC=%.2f, LOO-CV=%.2f, pDIC=%.2f, pWAIC=%.2f}", 
                               dic, waic, looCV, effectiveParametersDIC, effectiveParametersWAIC);
        }
    }
    
    /**
     * 对数似然函数接口
     * Log-likelihood function interface
     */
    @FunctionalInterface
    public interface LogLikelihoodFunction {
        double calculate(IVector data, IVector parameters);
    }
    
    /**
     * 点对数似然函数接口
     * Pointwise log-likelihood function interface
     */
    @FunctionalInterface
    public interface PointwiseLogLikelihoodFunction {
        double calculate(double dataPoint, IVector parameters);
    }
    
    /**
     * 计算所有信息准则
     * Calculate all information criteria
     * 
     * @param data 观测数据
     * @param posteriorSamples 后验样本
     * @param logLikelihood 对数似然函数
     * @param pointwiseLogLikelihood 点对数似然函数
     * @return 信息准则结果
     */
    public static ICResult calculateAll(IVector data, 
                                      IMatrix posteriorSamples,
                                      LogLikelihoodFunction logLikelihood,
                                      PointwiseLogLikelihoodFunction pointwiseLogLikelihood) {
        
        double dic = calculateDIC(data, posteriorSamples, logLikelihood);
        double waic = calculateWAIC(data, posteriorSamples, pointwiseLogLikelihood);
        double looCV = calculateLOOCV(data, posteriorSamples, pointwiseLogLikelihood);
        
        double pDIC = calculateEffectiveParametersDIC(data, posteriorSamples, logLikelihood);
        double pWAIC = calculateEffectiveParametersWAIC(data, posteriorSamples, pointwiseLogLikelihood);
        
        IVector pointwiseLOO = calculatePointwiseLOO(data, posteriorSamples, pointwiseLogLikelihood);
        
        return new ICResult(dic, waic, looCV, pDIC, pWAIC, pointwiseLOO);
    }
    
    /**
     * 计算偏差信息准则（DIC）
     * Calculate Deviance Information Criterion (DIC)
     * 
     * @param data 观测数据
     * @param posteriorSamples 后验样本
     * @param logLikelihood 对数似然函数
     * @return DIC值
     */
    public static double calculateDIC(IVector data, 
                                    IMatrix posteriorSamples,
                                    LogLikelihoodFunction logLikelihood) {
        
        int numSamples = posteriorSamples.rows();
        
        // 计算后验均值参数
        IVector posteriorMean = calculatePosteriorMean(posteriorSamples);
        
        // 计算平均偏差
        double meanDeviance = 0.0;
        for (int i = 0; i < numSamples; i++) {
            IVector parameters = getRow(posteriorSamples, i);
            double logLik = logLikelihood.calculate(data, parameters);
            meanDeviance += -2.0 * logLik;
        }
        meanDeviance /= numSamples;
        
        // 计算后验均值处的偏差
        double devianceAtMean = -2.0 * logLikelihood.calculate(data, posteriorMean);
        
        // 有效参数数量
        double effectiveParameters = meanDeviance - devianceAtMean;
        
        // DIC = 平均偏差 + 有效参数数量
        return meanDeviance + effectiveParameters;
    }
    
    /**
     * 计算Watanabe-Akaike信息准则（WAIC）
     * Calculate Watanabe-Akaike Information Criterion (WAIC)
     * 
     * @param data 观测数据
     * @param posteriorSamples 后验样本
     * @param pointwiseLogLikelihood 点对数似然函数
     * @return WAIC值
     */
    public static double calculateWAIC(IVector data,
                                     IMatrix posteriorSamples,
                                     PointwiseLogLikelihoodFunction pointwiseLogLikelihood) {
        
        int numSamples = posteriorSamples.rows();
        int dataSize = data.size();
        
        double lppd = 0.0; // log pointwise predictive density
        double pWAIC = 0.0; // effective number of parameters
        
        for (int i = 0; i < dataSize; i++) {
            double dataPoint = data.get(i).doubleValue();
            
            // 计算每个数据点的对数预测密度
            IVector logLikelihoods = Linalg.vector(numSamples);
            for (int j = 0; j < numSamples; j++) {
                IVector parameters = getRow(posteriorSamples, j);
                double logLik = pointwiseLogLikelihood.calculate(dataPoint, parameters);
                logLikelihoods.set(j, logLik);
            }
            
            // 计算log-sum-exp
            double maxLogLik = findMax(logLikelihoods);
            double sumExp = 0.0;
            for (int j = 0; j < numSamples; j++) {
                sumExp += Math.exp(logLikelihoods.get(j).doubleValue() - maxLogLik);
            }
            double logMeanLikelihood = maxLogLik + Math.log(sumExp / numSamples);
            lppd += logMeanLikelihood;
            
            // 计算方差项
            double meanLogLik = calculateMean(logLikelihoods);
            double variance = 0.0;
            for (int j = 0; j < numSamples; j++) {
                double diff = logLikelihoods.get(j).doubleValue() - meanLogLik;
                variance += diff * diff;
            }
            variance /= (numSamples - 1);
            pWAIC += variance;
        }
        
        return -2.0 * (lppd - pWAIC);
    }
    
    /**
     * 计算留一交叉验证（LOO-CV）
     * Calculate Leave-One-Out Cross-Validation (LOO-CV)
     * 
     * @param data 观测数据
     * @param posteriorSamples 后验样本
     * @param pointwiseLogLikelihood 点对数似然函数
     * @return LOO-CV值
     */
    public static double calculateLOOCV(IVector data,
                                      IMatrix posteriorSamples,
                                      PointwiseLogLikelihoodFunction pointwiseLogLikelihood) {
        
        IVector pointwiseLOO = calculatePointwiseLOO(data, posteriorSamples, pointwiseLogLikelihood);
        
        double sum = 0.0;
        for (int i = 0; i < pointwiseLOO.size(); i++) {
            sum += pointwiseLOO.get(i).doubleValue();
        }
        
        return -2.0 * sum;
    }
    
    /**
     * 计算点对点LOO
     * Calculate pointwise LOO
     */
    public static IVector calculatePointwiseLOO(IVector data,
                                              IMatrix posteriorSamples,
                                              PointwiseLogLikelihoodFunction pointwiseLogLikelihood) {
        
        int numSamples = posteriorSamples.rows();
        int dataSize = data.size();
        
        IVector pointwiseLOO = Linalg.vector(dataSize);
        
        for (int i = 0; i < dataSize; i++) {
            double dataPoint = data.get(i).doubleValue();
            
            // 计算重要性权重（简化的Pareto平滑重要性采样）
            IVector logLikelihoods = Linalg.vector(numSamples);
            for (int j = 0; j < numSamples; j++) {
                IVector parameters = getRow(posteriorSamples, j);
                double logLik = pointwiseLogLikelihood.calculate(dataPoint, parameters);
                logLikelihoods.set(j, logLik);
            }
            
            // 简化的LOO估计（使用重要性采样近似）
            double maxLogLik = findMax(logLikelihoods);
            double sumWeightedLik = 0.0;
            double sumWeights = 0.0;
            
            for (int j = 0; j < numSamples; j++) {
                double logLik = logLikelihoods.get(j).doubleValue();
                double weight = 1.0 / Math.exp(logLik - maxLogLik + 1e-8); // 重要性权重
                sumWeightedLik += Math.exp(logLik) * weight;
                sumWeights += weight;
            }
            
            double looEstimate = Math.log(sumWeightedLik / sumWeights);
            pointwiseLOO.set(i, looEstimate);
        }
        
        return pointwiseLOO;
    }
    
    /**
     * 计算DIC的有效参数数量
     * Calculate effective number of parameters for DIC
     */
    public static double calculateEffectiveParametersDIC(IVector data,
                                                        IMatrix posteriorSamples,
                                                        LogLikelihoodFunction logLikelihood) {
        
        int numSamples = posteriorSamples.rows();
        
        // 计算后验均值参数
        IVector posteriorMean = calculatePosteriorMean(posteriorSamples);
        
        // 计算平均偏差
        double meanDeviance = 0.0;
        for (int i = 0; i < numSamples; i++) {
            IVector parameters = getRow(posteriorSamples, i);
            double logLik = logLikelihood.calculate(data, parameters);
            meanDeviance += -2.0 * logLik;
        }
        meanDeviance /= numSamples;
        
        // 计算后验均值处的偏差
        double devianceAtMean = -2.0 * logLikelihood.calculate(data, posteriorMean);
        
        return meanDeviance - devianceAtMean;
    }
    
    /**
     * 计算WAIC的有效参数数量
     * Calculate effective number of parameters for WAIC
     */
    public static double calculateEffectiveParametersWAIC(IVector data,
                                                         IMatrix posteriorSamples,
                                                         PointwiseLogLikelihoodFunction pointwiseLogLikelihood) {
        
        int numSamples = posteriorSamples.rows();
        int dataSize = data.size();
        
        double pWAIC = 0.0;
        
        for (int i = 0; i < dataSize; i++) {
            double dataPoint = data.get(i).doubleValue();
            
            IVector logLikelihoods = Linalg.vector(numSamples);
            for (int j = 0; j < numSamples; j++) {
                IVector parameters = getRow(posteriorSamples, j);
                double logLik = pointwiseLogLikelihood.calculate(dataPoint, parameters);
                logLikelihoods.set(j, logLik);
            }
            
            double meanLogLik = calculateMean(logLikelihoods);
            double variance = 0.0;
            for (int j = 0; j < numSamples; j++) {
                double diff = logLikelihoods.get(j).doubleValue() - meanLogLik;
                variance += diff * diff;
            }
            variance /= (numSamples - 1);
            pWAIC += variance;
        }
        
        return pWAIC;
    }
    
    /**
     * 计算后验均值
     */
    private static IVector calculatePosteriorMean(IMatrix posteriorSamples) {
        int numParams = posteriorSamples.cols();
        int numSamples = posteriorSamples.rows();
        
        IVector mean = Linalg.vector(numParams);
        
        for (int j = 0; j < numParams; j++) {
            double sum = 0.0;
            for (int i = 0; i < numSamples; i++) {
                sum += posteriorSamples.get(i, j).doubleValue();
            }
            mean.set(j, sum / numSamples);
        }
        
        return mean;
    }
    
    /**
     * 获取矩阵的某一行
     */
    private static IVector getRow(IMatrix matrix, int row) {
        int cols = matrix.cols();
        IVector rowVector = Linalg.vector(cols);
        
        for (int j = 0; j < cols; j++) {
            rowVector.set(j, matrix.get(row, j));
        }
        
        return rowVector;
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
    
    /**
     * 找到向量中的最大值
     */
    private static double findMax(IVector vector) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < vector.size(); i++) {
            max = Math.max(max, vector.get(i).doubleValue());
        }
        return max;
    }
}