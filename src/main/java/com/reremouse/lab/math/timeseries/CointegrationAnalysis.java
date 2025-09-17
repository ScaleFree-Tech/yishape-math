package com.reremouse.lab.math.timeseries;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.util.Tuple3;

import java.util.ArrayList;
import java.util.List;

/**
 * 协整分析类 / Cointegration Analysis Class
 * <p>
 * 提供协整检验和协整关系分析功能，用于多变量时间序列的长期关系分析。
 * 使用项目现有的linalg包和stats包功能进行数值计算。
 * </p>
 * <p>
 * Provides cointegration testing and cointegration relationship analysis functionality
 * for long-term relationship analysis of multivariate time series. Uses existing linalg
 * and stats package functionality for numerical computation.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class CointegrationAnalysis {
    
    /**
     * Engle-Granger协整检验 / Engle-Granger Cointegration Test
     * <p>
     * 使用Engle-Granger两步法进行协整检验。
     * Use Engle-Granger two-step method for cointegration testing.
     * </p>
     *
     * @param y 因变量序列 / Dependent variable series
     * @param x 自变量序列 / Independent variable series
     * @param maxLags 最大滞后阶数 / Maximum lag order
     * @return 协整检验结果 / Cointegration test result
     */
    public static EngleGrangerResult engleGrangerTest(IVector<Double> y, IVector<Double> x, int maxLags) {
        if (y.length() != x.length()) {
            throw new IllegalArgumentException("序列长度必须相等");
        }
        
        int n = y.length();
        if (n < maxLags + 10) {
            throw new IllegalArgumentException("数据长度不足以进行协整检验");
        }
        
        // 第一步：估计长期关系 / Step 1: Estimate long-run relationship
        Tuple2<Double, Double> coef = estimateLongRunRelationship(y, x);
        double alpha = coef._1; // 截距 / Intercept
        double beta = coef._2;  // 斜率 / Slope
        
        // 计算残差 / Calculate residuals
        IVector<Double> residuals = y.sub(x.multiplyScalar(beta).addScalar(alpha));
        
        // 第二步：检验残差的平稳性 / Step 2: Test stationarity of residuals
        ADFTestResult adfResult = performADFTest(residuals, maxLags);
        
        // 计算协整向量 / Calculate cointegrating vector
        IVector<Double> cointegratingVector = Linalg.vector(new double[]{1, -beta});
        
        return new EngleGrangerResult(alpha, beta, cointegratingVector, residuals, adfResult);
    }
    
    /**
     * Johansen协整检验 / Johansen Cointegration Test
     * <p>
     * 使用Johansen方法进行多变量协整检验。
     * Use Johansen method for multivariate cointegration testing.
     * </p>
     *
     * @param data 多变量时间序列数据 / Multivariate time series data
     * @param maxLags 最大滞后阶数 / Maximum lag order
     * @param trendType 趋势类型 / Trend type
     * @return Johansen协整检验结果 / Johansen cointegration test result
     */
    public static JohansenResult johansenTest(IMatrix<Double> data, int maxLags, TrendType trendType) {
        int n = data.getRowNum();
        int k = data.getColNum();
        
        if (n < maxLags + 10) {
            throw new IllegalArgumentException("数据长度不足以进行Johansen协整检验");
        }
        
        // 差分数据 / Differenced data
        IMatrix<Double> diffData = Linalg.zeros(n - 1, k);
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < k; j++) {
                diffData.set(i - 1, j, data.get(i, j) - data.get(i - 1, j));
            }
        }
        
        // 滞后数据 / Lagged data
        IMatrix<Double> lagData = Linalg.zeros(n - maxLags, k);
        for (int i = maxLags; i < n; i++) {
            for (int j = 0; j < k; j++) {
                lagData.set(i - maxLags, j, data.get(i - 1, j));
            }
        }
        
        // 计算协整关系 / Calculate cointegrating relationships
        List<CointegratingVector> cointegratingVectors = calculateCointegratingVectors(diffData, lagData, trendType);
        
        // 计算特征值和统计量 / Calculate eigenvalues and statistics
        List<Double> eigenvalues = calculateEigenvalues(diffData, lagData, trendType);
        List<Double> traceStatistics = calculateTraceStatistics(eigenvalues, n - maxLags);
        List<Double> maxEigenvalueStatistics = calculateMaxEigenvalueStatistics(eigenvalues, n - maxLags);
        
        // 计算临界值 / Calculate critical values
        List<Double> traceCriticalValues = calculateTraceCriticalValues(k, trendType);
        List<Double> maxEigenvalueCriticalValues = calculateMaxEigenvalueCriticalValues(k, trendType);
        
        return new JohansenResult(cointegratingVectors, eigenvalues, traceStatistics, maxEigenvalueStatistics,
                                traceCriticalValues, maxEigenvalueCriticalValues);
    }
    
    /**
     * 协整关系估计 / Cointegrating Relationship Estimation
     * <p>
     * 估计协整关系参数。
     * Estimate cointegrating relationship parameters.
     * </p>
     *
     * @param y 因变量序列 / Dependent variable series
     * @param x 自变量序列 / Independent variable series
     * @return 协整关系参数 / Cointegrating relationship parameters
     */
    public static CointegratingRelationship estimateCointegratingRelationship(IVector<Double> y, IVector<Double> x) {
        Tuple2<Double, Double> coef = estimateLongRunRelationship(y, x);
        double alpha = coef._1;
        double beta = coef._2;
        
        // 计算误差修正项 / Calculate error correction term
        IVector<Double> ect = y.sub(x.multiplyScalar(beta).addScalar(alpha));
        
        // 计算统计量 / Calculate statistics
        double rSquared = calculateRSquared(y, x, alpha, beta);
        double durbinWatson = calculateDurbinWatson(ect);
        
        return new CointegratingRelationship(alpha, beta, ect, rSquared, durbinWatson);
    }
    
    /**
     * 误差修正模型 / Error Correction Model
     * <p>
     * 估计误差修正模型。
     * Estimate error correction model.
     * </p>
     *
     * @param deltaY 因变量差分 / Dependent variable differences
     * @param deltaX 自变量差分 / Independent variable differences
     * @param ect 误差修正项 / Error correction term
     * @param lags 滞后阶数 / Lag order
     * @return 误差修正模型结果 / Error correction model result
     */
    public static ErrorCorrectionModel estimateECM(IVector<Double> deltaY, IVector<Double> deltaX, 
                                                  IVector<Double> ect, int lags) {
        int n = deltaY.length();
        if (n < lags + 5) {
            throw new IllegalArgumentException("数据长度不足以估计误差修正模型");
        }
        
        // 构建回归矩阵 / Build regression matrix
        int regressors = 2 + 2 * lags; // 常数项 + ECT + 滞后项 / Constant + ECT + lagged terms
        IMatrix<Double> X = Linalg.zeros(n - lags, regressors);
        IVector<Double> Y = deltaY.slice(lags, n);
        
        for (int i = 0; i < n - lags; i++) {
            int row = i;
            X.set(row, 0, 1.0); // 常数项 / Constant term
            X.set(row, 1, ect.get(i + lags - 1)); // ECT项 / ECT term
            
            // 滞后项 / Lagged terms
            for (int j = 0; j < lags; j++) {
                X.set(row, 2 + j, deltaY.get(i + lags - 1 - j));
                X.set(row, 2 + lags + j, deltaX.get(i + lags - 1 - j));
            }
        }
        
        // 最小二乘估计 / Least squares estimation
        IMatrix<Double> XTX = X.transpose().mmul(X);
        IVector<Double> XTY = X.transpose().mmul(Y);
        IVector<Double> coefficients = XTX.pinv().mmul(XTY);
        
        // 计算拟合值 / Calculate fitted values
        IVector<Double> fitted = X.mmul(coefficients);
        IVector<Double> residuals = Y.sub(fitted);
        
        // 计算统计量 / Calculate statistics
        double rSquared = calculateRSquared(Y, fitted);
        double durbinWatson = calculateDurbinWatson(residuals);
        
        return new ErrorCorrectionModel(coefficients, fitted, residuals, rSquared, durbinWatson);
    }
    
    // ========== 枚举类型 / Enum Types ==========
    
    /**
     * 趋势类型枚举 / Trend Type Enum
     */
    public enum TrendType {
        NONE,       // 无趋势 / No trend
        CONSTANT,   // 常数项 / Constant term
        LINEAR      // 线性趋势 / Linear trend
    }
    
    // ========== 结果类 / Result Classes ==========
    
    /**
     * Engle-Granger检验结果类 / Engle-Granger Test Result Class
     */
    public static class EngleGrangerResult {
        public final double alpha;
        public final double beta;
        public final IVector<Double> cointegratingVector;
        public final IVector<Double> residuals;
        public final ADFTestResult adfResult;
        
        public EngleGrangerResult(double alpha, double beta, IVector<Double> cointegratingVector,
                                IVector<Double> residuals, ADFTestResult adfResult) {
            this.alpha = alpha;
            this.beta = beta;
            this.cointegratingVector = cointegratingVector;
            this.residuals = residuals;
            this.adfResult = adfResult;
        }
    }
    
    /**
     * Johansen检验结果类 / Johansen Test Result Class
     */
    public static class JohansenResult {
        public final List<CointegratingVector> cointegratingVectors;
        public final List<Double> eigenvalues;
        public final List<Double> traceStatistics;
        public final List<Double> maxEigenvalueStatistics;
        public final List<Double> traceCriticalValues;
        public final List<Double> maxEigenvalueCriticalValues;
        
        public JohansenResult(List<CointegratingVector> cointegratingVectors, List<Double> eigenvalues,
                            List<Double> traceStatistics, List<Double> maxEigenvalueStatistics,
                            List<Double> traceCriticalValues, List<Double> maxEigenvalueCriticalValues) {
            this.cointegratingVectors = cointegratingVectors;
            this.eigenvalues = eigenvalues;
            this.traceStatistics = traceStatistics;
            this.maxEigenvalueStatistics = maxEigenvalueStatistics;
            this.traceCriticalValues = traceCriticalValues;
            this.maxEigenvalueCriticalValues = maxEigenvalueCriticalValues;
        }
    }
    
    /**
     * 协整向量类 / Cointegrating Vector Class
     */
    public static class CointegratingVector {
        public final IVector<Double> vector;
        public final double eigenvalue;
        public final double traceStatistic;
        public final double maxEigenvalueStatistic;
        
        public CointegratingVector(IVector<Double> vector, double eigenvalue, 
                                 double traceStatistic, double maxEigenvalueStatistic) {
            this.vector = vector;
            this.eigenvalue = eigenvalue;
            this.traceStatistic = traceStatistic;
            this.maxEigenvalueStatistic = maxEigenvalueStatistic;
        }
    }
    
    /**
     * 协整关系类 / Cointegrating Relationship Class
     */
    public static class CointegratingRelationship {
        public final double alpha;
        public final double beta;
        public final IVector<Double> ect;
        public final double rSquared;
        public final double durbinWatson;
        
        public CointegratingRelationship(double alpha, double beta, IVector<Double> ect,
                                       double rSquared, double durbinWatson) {
            this.alpha = alpha;
            this.beta = beta;
            this.ect = ect;
            this.rSquared = rSquared;
            this.durbinWatson = durbinWatson;
        }
    }
    
    /**
     * 误差修正模型类 / Error Correction Model Class
     */
    public static class ErrorCorrectionModel {
        public final IVector<Double> coefficients;
        public final IVector<Double> fitted;
        public final IVector<Double> residuals;
        public final double rSquared;
        public final double durbinWatson;
        
        public ErrorCorrectionModel(IVector<Double> coefficients, IVector<Double> fitted,
                                  IVector<Double> residuals, double rSquared, double durbinWatson) {
            this.coefficients = coefficients;
            this.fitted = fitted;
            this.residuals = residuals;
            this.rSquared = rSquared;
            this.durbinWatson = durbinWatson;
        }
    }
    
    /**
     * ADF检验结果类 / ADF Test Result Class
     */
    public static class ADFTestResult {
        public final double statistic;
        public final double pValue;
        public final double criticalValue1;
        public final double criticalValue5;
        public final double criticalValue10;
        public final boolean isStationary;
        
        public ADFTestResult(double statistic, double pValue, double criticalValue1,
                           double criticalValue5, double criticalValue10, boolean isStationary) {
            this.statistic = statistic;
            this.pValue = pValue;
            this.criticalValue1 = criticalValue1;
            this.criticalValue5 = criticalValue5;
            this.criticalValue10 = criticalValue10;
            this.isStationary = isStationary;
        }
    }
    
    // ========== 私有辅助方法 / Private Helper Methods ==========
    
    /**
     * 估计长期关系 / Estimate long-run relationship
     */
    private static Tuple2<Double, Double> estimateLongRunRelationship(IVector<Double> y, IVector<Double> x) {
        int n = y.length();
        double sumX = x.sum();
        double sumY = y.sum();
        double sumXY = x.multiply(y).sum();
        double sumXX = x.multiply(x).sum();
        
        double beta = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        double alpha = (sumY - beta * sumX) / n;
        
        return new Tuple2<>(alpha, beta);
    }
    
    /**
     * 执行ADF检验 / Perform ADF test
     */
    private static ADFTestResult performADFTest(IVector<Double> data, int maxLags) {
        // 简化的ADF检验实现 / Simplified ADF test implementation
        double adfStatistic = calculateADFStatistic(data);
        
        // 临界值 / Critical values
        double criticalValue1 = -3.43;
        double criticalValue5 = -2.86;
        double criticalValue10 = -2.57;
        
        boolean isStationary = adfStatistic < criticalValue5;
        double pValue = estimatePValue(adfStatistic);
        
        return new ADFTestResult(adfStatistic, pValue, criticalValue1, criticalValue5, criticalValue10, isStationary);
    }
    
    /**
     * 计算ADF统计量 / Calculate ADF statistic
     */
    private static double calculateADFStatistic(IVector<Double> data) {
        int n = data.length();
        if (n < 3) return 0.0;
        
        IVector<Double> diff = data.slice(1, n).sub(data.slice(0, n - 1));
        IVector<Double> lagged = data.slice(0, n - 1);
        
        double sumXY = diff.multiply(lagged).sum();
        double sumXX = lagged.multiply(lagged).sum();
        
        if (sumXX == 0) return 0.0;
        
        double beta = sumXY / sumXX;
        IVector<Double> residuals = diff.sub(lagged.multiplyScalar(beta));
        double mse = residuals.multiply(residuals).sum() / (n - 2);
        double se = Math.sqrt(mse / sumXX);
        
        return beta / se;
    }
    
    /**
     * 估计p值 / Estimate p-value
     */
    private static double estimatePValue(double adfStatistic) {
        if (adfStatistic < -3.43) return 0.01;
        if (adfStatistic < -2.86) return 0.05;
        if (adfStatistic < -2.57) return 0.10;
        return 0.20;
    }
    
    /**
     * 计算协整向量 / Calculate cointegrating vectors
     */
    private static List<CointegratingVector> calculateCointegratingVectors(IMatrix<Double> diffData, 
                                                                          IMatrix<Double> lagData, 
                                                                          TrendType trendType) {
        // 简化的协整向量计算 / Simplified cointegrating vector calculation
        List<CointegratingVector> vectors = new ArrayList<>();
        
        // 这里应该实现完整的Johansen方法
        // Complete Johansen method should be implemented here
        
        return vectors;
    }
    
    /**
     * 计算特征值 / Calculate eigenvalues
     */
    private static List<Double> calculateEigenvalues(IMatrix<Double> diffData, IMatrix<Double> lagData, 
                                                    TrendType trendType) {
        // 简化的特征值计算 / Simplified eigenvalue calculation
        List<Double> eigenvalues = new ArrayList<>();
        
        // 这里应该实现完整的特征值计算
        // Complete eigenvalue calculation should be implemented here
        
        return eigenvalues;
    }
    
    /**
     * 计算迹统计量 / Calculate trace statistics
     */
    private static List<Double> calculateTraceStatistics(List<Double> eigenvalues, int n) {
        List<Double> traceStats = new ArrayList<>();
        
        for (int i = 0; i < eigenvalues.size(); i++) {
            double trace = 0.0;
            for (int j = i; j < eigenvalues.size(); j++) {
                trace += -n * Math.log(1 - eigenvalues.get(j));
            }
            traceStats.add(trace);
        }
        
        return traceStats;
    }
    
    /**
     * 计算最大特征值统计量 / Calculate max eigenvalue statistics
     */
    private static List<Double> calculateMaxEigenvalueStatistics(List<Double> eigenvalues, int n) {
        List<Double> maxEigenvalueStats = new ArrayList<>();
        
        for (int i = 0; i < eigenvalues.size(); i++) {
            double maxEigenvalue = -n * Math.log(1 - eigenvalues.get(i));
            maxEigenvalueStats.add(maxEigenvalue);
        }
        
        return maxEigenvalueStats;
    }
    
    /**
     * 计算迹临界值 / Calculate trace critical values
     */
    private static List<Double> calculateTraceCriticalValues(int k, TrendType trendType) {
        // 简化的临界值计算 / Simplified critical value calculation
        List<Double> criticalValues = new ArrayList<>();
        
        // 这里应该使用实际的临界值表
        // Actual critical value tables should be used here
        
        return criticalValues;
    }
    
    /**
     * 计算最大特征值临界值 / Calculate max eigenvalue critical values
     */
    private static List<Double> calculateMaxEigenvalueCriticalValues(int k, TrendType trendType) {
        // 简化的临界值计算 / Simplified critical value calculation
        List<Double> criticalValues = new ArrayList<>();
        
        // 这里应该使用实际的临界值表
        // Actual critical value tables should be used here
        
        return criticalValues;
    }
    
    /**
     * 计算R² / Calculate R-squared
     */
    private static double calculateRSquared(IVector<Double> y, IVector<Double> fitted) {
        double ssRes = y.sub(fitted).multiply(y.sub(fitted)).sum();
        double ssTot = y.subScalar(y.mean()).multiply(y.subScalar(y.mean())).sum();
        
        if (ssTot == 0) return 0.0;
        return 1 - ssRes / ssTot;
    }
    
    /**
     * 计算R²（重载） / Calculate R-squared (overloaded)
     */
    private static double calculateRSquared(IVector<Double> y, IVector<Double> x, double alpha, double beta) {
        IVector<Double> fitted = x.multiplyScalar(beta).addScalar(alpha);
        return calculateRSquared(y, fitted);
    }
    
    /**
     * 计算Durbin-Watson统计量 / Calculate Durbin-Watson statistic
     */
    private static double calculateDurbinWatson(IVector<Double> residuals) {
        int n = residuals.length();
        if (n < 2) return 0.0;
        
        double numerator = 0.0;
        for (int i = 1; i < n; i++) {
            double diff = residuals.get(i) - residuals.get(i - 1);
            numerator += diff * diff;
        }
        
        double denominator = residuals.multiply(residuals).sum();
        
        if (denominator == 0) return 0.0;
        return numerator / denominator;
    }
}
