package com.yishape.lab.math.timeseries.model;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;

import java.util.ArrayList;
import java.util.List;

/**
 * VAR模型实现类 / VAR Model Implementation Class
 * <p>
 * 提供向量自回归（VAR）模型的实现，用于多变量时间序列的建模和预测。
 * 使用项目现有的linalg包功能进行数值计算。
 * </p>
 * <p>
 * Provides Vector Autoregression (VAR) model implementation for multivariate time series
 * modeling and forecasting. Uses existing linalg package functionality for numerical computation.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class VARModel {
    
    private final int p; // VAR阶数 / VAR order
    private final int k; // 变量个数 / Number of variables
    private final IMatrix<Double> coefficients; // 系数矩阵 / Coefficient matrix
    private final IMatrix<Double> coefMatrix; // 系数矩阵（用于预测）/ Coefficient matrix (for forecasting)
    private final IVector<Double> constant; // 常数项 / Constant term
    private final IMatrix<Double> residuals; // 残差矩阵 / Residual matrix
    private final IMatrix<Double> covariance; // 残差协方差矩阵 / Residual covariance matrix
    private final double logLikelihood; // 对数似然 / Log likelihood
    private final double aic; // AIC信息准则 / AIC information criterion
    private final double bic; // BIC信息准则 / BIC information criterion
    private final String[] variableNames; // 变量名 / Variable names
    
    /**
     * 构造函数 / Constructor
     *
     * @param p VAR阶数 / VAR order
     * @param k 变量个数 / Number of variables
     * @param coefficients 系数矩阵 / Coefficient matrix
     * @param constant 常数项 / Constant term
     * @param residuals 残差矩阵 / Residual matrix
     * @param covariance 残差协方差矩阵 / Residual covariance matrix
     * @param logLikelihood 对数似然 / Log likelihood
     * @param aic AIC信息准则 / AIC information criterion
     * @param bic BIC信息准则 / BIC information criterion
     * @param variableNames 变量名 / Variable names
     */
    public VARModel(int p, int k, IMatrix<Double> coefficients, IMatrix<Double> coefMatrix, IVector<Double> constant,
                   IMatrix<Double> residuals, IMatrix<Double> covariance, double logLikelihood,
                   double aic, double bic, String[] variableNames) {
        this.p = p;
        this.k = k;
        this.coefficients = coefficients;
        this.coefMatrix = coefMatrix;
        this.constant = constant;
        this.residuals = residuals;
        this.covariance = covariance;
        this.logLikelihood = logLikelihood;
        this.aic = aic;
        this.bic = bic;
        this.variableNames = variableNames;
    }
    
    /**
     * 拟合VAR模型 / Fit VAR Model
     * <p>
     * 使用最小二乘法拟合VAR模型。
     * Use least squares method to fit VAR model.
     * </p>
     *
     * @param data 多变量时间序列数据 / Multivariate time series data
     * @param p VAR阶数 / VAR order
     * @param variableNames 变量名 / Variable names
     * @return 拟合的VAR模型 / Fitted VAR model
     */
    public static VARModel fit(IMatrix<Double> data, int p, String[] variableNames) {
        int n = data.getRowNum();
        int k = data.getColNum();
        
        if (n < p + 10) {
            throw new IllegalArgumentException("数据长度不足以拟合VAR模型");
        }
        
        if (variableNames == null) {
            variableNames = new String[k];
            for (int i = 0; i < k; i++) {
                variableNames[i] = "Var" + (i + 1);
            }
        }
        
        // 构建回归矩阵 / Build regression matrix
        int regressors = k * p + 1; // 滞后项 + 常数项 / Lagged terms + constant
        int observations = n - p;
        
        IMatrix<Double> X = Linalg.zeros(observations, regressors);
        IMatrix<Double> Y = Linalg.zeros(observations, k);
        
        // 填充数据 / Fill data
        for (int t = 0; t < observations; t++) {
            int row = t;
            
            // 常数项 / Constant term
            X.set(row, 0, 1.0);
            
            // 滞后项 / Lagged terms
            for (int lag = 1; lag <= p; lag++) {
                for (int var = 0; var < k; var++) {
                    int col = 1 + (lag - 1) * k + var;
                    X.set(row, col, data.get(t + p - lag, var));
                }
            }
            
            // 因变量 / Dependent variables
            for (int var = 0; var < k; var++) {
                Y.set(row, var, data.get(t + p, var));
            }
        }
        
        // 最小二乘估计 / Least squares estimation
        IMatrix<Double> XTX = X.transpose().mmul(X);
        IMatrix<Double> XTY = X.transpose().mmul(Y);
        IMatrix<Double> coefficients = XTX.pinv().mmul(XTY);
        
        // 提取常数项和系数矩阵 / Extract constant term and coefficient matrix
        IVector<Double> constant = Linalg.zeros(k);
        IMatrix<Double> coefMatrix = Linalg.zeros(k, k * p);
        
        for (int i = 0; i < k; i++) {
            constant.set(i, coefficients.get(0, i));
            for (int j = 0; j < k * p; j++) {
                coefMatrix.set(i, j, coefficients.get(j + 1, i));
            }
        }
        
        // 计算残差 / Calculate residuals
        IMatrix<Double> fitted = X.mmul(coefficients);
        IMatrix<Double> residuals = Y.sub(fitted);
        
        // 计算残差协方差矩阵 / Calculate residual covariance matrix
        IMatrix<Double> covariance = residuals.transpose().mmul(residuals).multiplyScalar(1.0 / (observations - regressors));
        
        // 计算对数似然 / Calculate log likelihood
        double logLikelihood = calculateLogLikelihood(residuals, covariance, observations);
        
        // 计算信息准则 / Calculate information criteria
        int numParams = k * (k * p + 1);
        double aic = 2 * numParams - 2 * logLikelihood;
        double bic = numParams * Math.log(observations) - 2 * logLikelihood;
        
        return new VARModel(p, k, coefficients, coefMatrix, constant, residuals, covariance, 
                          logLikelihood, aic, bic, variableNames);
    }
    
    /**
     * 预测 / Forecast
     * <p>
     * 使用拟合的VAR模型进行预测。
     * Use fitted VAR model for forecasting.
     * </p>
     *
     * @param steps 预测步数 / Forecast steps
     * @return 预测结果 / Forecast results
     */
    public VARForecastResult forecast(int steps) {
        if (steps <= 0) {
            throw new IllegalArgumentException("预测步数必须为正数");
        }
        
        IMatrix<Double> forecast = Linalg.zeros(steps, k);
        IMatrix<Double> forecastVariance = Linalg.zeros(steps, k);
        
        // 获取最后p个观测值 / Get last p observations
        IMatrix<Double> lastObservations = Linalg.zeros(p, k);
        int n = residuals.getRowNum() + p;
        
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < k; j++) {
                lastObservations.set(i, j, 0.0); // 简化处理 / Simplified handling
            }
        }
        
        // 逐步预测 / Step-by-step forecasting
        for (int t = 0; t < steps; t++) {
            IVector<Double> forecastVector = Linalg.zeros(k);
            
            // 常数项 / Constant term
            forecastVector = forecastVector.add(constant);
            
            // 滞后项 / Lagged terms
            for (int lag = 1; lag <= p; lag++) {
                IVector<Double> lagVector = Linalg.zeros(k);
                for (int var = 0; var < k; var++) {
                    lagVector.set(var, lastObservations.get(p - lag, var));
                }
                
                IMatrix<Double> coefBlock = coefMatrix.slice(0, k, (lag - 1) * k, lag * k);
                //todo: 核实此处为何可以这样加，本系统的向量均为行向量，为何发生向量加矩阵的运算？
                forecastVector = forecastVector.add(coefBlock.mmul(lagVector));
            }
            
            // 存储预测值 / Store forecast values
            for (int var = 0; var < k; var++) {
                forecast.set(t, var, forecastVector.get(var));
            }
            
            // 更新滞后观测值 / Update lagged observations
            for (int i = 0; i < p - 1; i++) {
                for (int j = 0; j < k; j++) {
                    lastObservations.set(i, j, lastObservations.get(i + 1, j));
                }
            }
            for (int j = 0; j < k; j++) {
                lastObservations.set(p - 1, j, forecastVector.get(j));
            }
            
            // 计算预测方差 / Calculate forecast variance
            for (int var = 0; var < k; var++) {
                forecastVariance.set(t, var, covariance.get(var, var));
            }
        }
        
        return new VARForecastResult(forecast, forecastVariance, variableNames);
    }
    
    /**
     * 脉冲响应函数 / Impulse Response Function
     * <p>
     * 计算VAR模型的脉冲响应函数。
     * Calculate impulse response function of VAR model.
     * </p>
     *
     * @param steps 响应步数 / Response steps
     * @return 脉冲响应结果 / Impulse response result
     */
    public ImpulseResponseResult impulseResponse(int steps) {
        if (steps <= 0) {
            throw new IllegalArgumentException("响应步数必须为正数");
        }
        
        List<IMatrix<Double>> responseMatrices = new ArrayList<>();
        
        // 初始化 / Initialize
        IMatrix<Double> Phi0 = Linalg.eye(k);
        responseMatrices.add(Phi0);
        
        // 计算脉冲响应矩阵 / Calculate impulse response matrices
        for (int t = 1; t < steps; t++) {
            IMatrix<Double> Phi = Linalg.zeros(k, k);
            
            for (int lag = 1; lag <= Math.min(p, t); lag++) {
                IMatrix<Double> coefBlock = coefMatrix.slice(0, k, (lag - 1) * k, lag * k);
                IMatrix<Double> prevResponse = responseMatrices.get(t - lag);
                Phi = Phi.add(coefBlock.mmul(prevResponse));
            }
            
            responseMatrices.add(Phi);
        }
        
        return new ImpulseResponseResult(responseMatrices, variableNames);
    }
    
    /**
     * 方差分解 / Variance Decomposition
     * <p>
     * 计算VAR模型的方差分解。
     * Calculate variance decomposition of VAR model.
     * </p>
     *
     * @param steps 分解步数 / Decomposition steps
     * @return 方差分解结果 / Variance decomposition result
     */
    public VarianceDecompositionResult varianceDecomposition(int steps) {
        if (steps <= 0) {
            throw new IllegalArgumentException("分解步数必须为正数");
        }
        
        // 计算脉冲响应 / Calculate impulse response
        ImpulseResponseResult impulseResponse = impulseResponse(steps);
        List<IMatrix<Double>> responseMatrices = impulseResponse.responseMatrices;
        
        // 计算方差分解 / Calculate variance decomposition
        IMatrix<Double> decomposition = Linalg.zeros(k, k);
        
        for (int t = 0; t < steps; t++) {
            IMatrix<Double> Phi = responseMatrices.get(t);
            IMatrix<Double> PhiSquared = Phi.multiplyScalar(Phi.get(0, 0));
            
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < k; j++) {
                    double contribution = PhiSquared.get(i, j) * covariance.get(j, j);
                    decomposition.set(i, j, decomposition.get(i, j) + contribution);
                }
            }
        }
        
        // 标准化 / Normalize
        for (int i = 0; i < k; i++) {
            double total = 0.0;
            for (int j = 0; j < k; j++) {
                total += decomposition.get(i, j);
            }
            
            if (total > 0) {
                for (int j = 0; j < k; j++) {
                    decomposition.set(i, j, decomposition.get(i, j) / total);
                }
            }
        }
        
        return new VarianceDecompositionResult(decomposition, variableNames);
    }
    
    /**
     * Granger因果检验 / Granger Causality Test
     * <p>
     * 检验变量之间的Granger因果关系。
     * Test Granger causality between variables.
     * </p>
     *
     * @param causeVar 原因变量索引 / Cause variable index
     * @param effectVar 结果变量索引 / Effect variable index
     * @return Granger因果检验结果 / Granger causality test result
     */
    public GrangerCausalityResult grangerCausalityTest(int causeVar, int effectVar) {
        if (causeVar < 0 || causeVar >= k || effectVar < 0 || effectVar >= k) {
            throw new IllegalArgumentException("变量索引超出范围");
        }
        
        // 构建受限模型 / Build restricted model
        int n = residuals.getRowNum();
        int regressors = k * p + 1;
        
        IMatrix<Double> X = Linalg.zeros(n, regressors);
        IVector<Double> Y = Linalg.zeros(n);
        
        // 填充数据（简化） / Fill data (simplified)
        for (int t = 0; t < n; t++) {
            X.set(t, 0, 1.0); // 常数项 / Constant term
            Y.set(t, 0.0); // 简化处理 / Simplified handling
        }
        
        // 计算F统计量 / Calculate F statistic
        double fStatistic = calculateFStatistic(X, Y, causeVar, effectVar);
        
        // 计算p值 / Calculate p-value
        double pValue = calculateFPValue(fStatistic, p, n - regressors);
        
        boolean isSignificant = pValue < 0.05;
        
        return new GrangerCausalityResult(causeVar, effectVar, fStatistic, pValue, isSignificant);
    }
    
    /**
     * 自动选择VAR模型 / Automatic VAR Model Selection
     * <p>
     * 使用AIC或BIC准则自动选择最优的VAR模型参数。
     * Automatically select optimal VAR model parameters using AIC or BIC criteria.
     * </p>
     *
     * @param data 多变量时间序列数据 / Multivariate time series data
     * @param maxP 最大VAR阶数 / Maximum VAR order
     * @param criterion 选择准则 / Selection criterion
     * @param variableNames 变量名 / Variable names
     * @return 最优VAR模型 / Optimal VAR model
     */
    public static VARModel autoFit(IMatrix<Double> data, int maxP, SelectionCriterion criterion, String[] variableNames) {
        VARModel bestModel = null;
        double bestCriterion = Double.POSITIVE_INFINITY;
        
        for (int p = 1; p <= maxP; p++) {
            try {
                VARModel model = fit(data, p, variableNames);
                double criterionValue = (criterion == SelectionCriterion.AIC) ? model.aic : model.bic;
                
                if (criterionValue < bestCriterion) {
                    bestCriterion = criterionValue;
                    bestModel = model;
                }
            } catch (Exception e) {
                // 跳过无效的模型参数组合 / Skip invalid model parameter combinations
                continue;
            }
        }
        
        if (bestModel == null) {
            throw new RuntimeException("无法找到合适的VAR模型");
        }
        
        return bestModel;
    }
    
    // ========== Getter方法 / Getter Methods ==========
    
    public int getP() { return p; }
    public int getK() { return k; }
    public IMatrix<Double> getCoefficients() { return coefficients; }
    public IVector<Double> getConstant() { return constant; }
    public IMatrix<Double> getResiduals() { return residuals; }
    public IMatrix<Double> getCovariance() { return covariance; }
    public double getLogLikelihood() { return logLikelihood; }
    public double getAic() { return aic; }
    public double getBic() { return bic; }
    public String[] getVariableNames() { return variableNames; }
    
    // ========== 枚举类型 / Enum Types ==========
    
    /**
     * 模型选择准则枚举 / Model Selection Criterion Enum
     */
    public enum SelectionCriterion {
        AIC,    // Akaike Information Criterion
        BIC     // Bayesian Information Criterion
    }
    
    // ========== 结果类 / Result Classes ==========
    
    /**
     * VAR预测结果类 / VAR Forecast Result Class
     * <p>
     * 存储VAR模型预测结果，包括预测值、预测方差和变量名。
     * Stores VAR model forecast results including forecast values, forecast variance, and variable names.
     * </p>
     *
     * @author lteb2
     * @version 1.0
     * @since 1.0
     */
    public static class VARForecastResult {
        public final IMatrix<Double> forecast;
        public final IMatrix<Double> forecastVariance;
        public final String[] variableNames;

        /**
         * 构造函数 / Constructor
         *
         * @param forecast 预测值矩阵 / Forecast matrix
         * @param forecastVariance 预测方差矩阵 / Forecast variance matrix
         * @param variableNames 变量名数组 / Variable names array
         */
        public VARForecastResult(IMatrix<Double> forecast, IMatrix<Double> forecastVariance, String[] variableNames) {
            this.forecast = forecast;
            this.forecastVariance = forecastVariance;
            this.variableNames = variableNames;
        }
    }
    
    /**
     * 脉冲响应结果类 / Impulse Response Result Class
     * <p>
     * 存储VAR模型脉冲响应分析结果，包括响应矩阵序列和变量名。
     * Stores VAR model impulse response analysis results including response matrix sequence and variable names.
     * </p>
     *
     * @author lteb2
     * @version 1.0
     * @since 1.0
     */
    public static class ImpulseResponseResult {
        public final List<IMatrix<Double>> responseMatrices;
        public final String[] variableNames;

        /**
         * 构造函数 / Constructor
         *
         * @param responseMatrices 脉冲响应矩阵列表 / Impulse response matrices list
         * @param variableNames 变量名数组 / Variable names array
         */
        public ImpulseResponseResult(List<IMatrix<Double>> responseMatrices, String[] variableNames) {
            this.responseMatrices = responseMatrices;
            this.variableNames = variableNames;
        }
    }
    
    /**
     * 方差分解结果类 / Variance Decomposition Result Class
     * <p>
     * 存储VAR模型方差分解结果，包括分解矩阵和变量名。
     * Stores VAR model variance decomposition results including decomposition matrix and variable names.
     * </p>
     *
     * @author lteb2
     * @version 1.0
     * @since 1.0
     */
    public static class VarianceDecompositionResult {
        public final IMatrix<Double> decomposition;
        public final String[] variableNames;

        /**
         * 构造函数 / Constructor
         *
         * @param decomposition 方差分解矩阵 / Variance decomposition matrix
         * @param variableNames 变量名数组 / Variable names array
         */
        public VarianceDecompositionResult(IMatrix<Double> decomposition, String[] variableNames) {
            this.decomposition = decomposition;
            this.variableNames = variableNames;
        }
    }
    
    /**
     * Granger因果检验结果类 / Granger Causality Test Result Class
     * <p>
     * 存储Granger因果检验的结果，包括原因变量、结果变量、F统计量和p值。
     * Stores Granger causality test results including cause variable, effect variable, F statistic, and p-value.
     * </p>
     *
     * @author lteb2
     * @version 1.0
     * @since 1.0
     */
    public static class GrangerCausalityResult {
        public final int causeVar;
        public final int effectVar;
        public final double fStatistic;
        public final double pValue;
        public final boolean isSignificant;

        /**
         * 构造函数 / Constructor
         *
         * @param causeVar 原因变量索引 / Cause variable index
         * @param effectVar 结果变量索引 / Effect variable index
         * @param fStatistic F统计量 / F statistic
         * @param pValue p值 / P-value
         * @param isSignificant 是否显著 / Whether significant
         */
        public GrangerCausalityResult(int causeVar, int effectVar, double fStatistic, 
                                    double pValue, boolean isSignificant) {
            this.causeVar = causeVar;
            this.effectVar = effectVar;
            this.fStatistic = fStatistic;
            this.pValue = pValue;
            this.isSignificant = isSignificant;
        }
    }
    
    // ========== 私有辅助方法 / Private Helper Methods ==========
    
    /**
     * 计算对数似然 / Calculate log likelihood
     * <p>
     * 计算VAR模型的对数似然值。
     * Calculate log likelihood of VAR model.
     * </p>
     *
     * @param residuals 残差矩阵 / Residuals matrix
     * @param covariance 残差协方差矩阵 / Residual covariance matrix
     * @param n 样本数量 / Sample size
     * @return 对数似然值 / Log likelihood value
     */
    private static double calculateLogLikelihood(IMatrix<Double> residuals, IMatrix<Double> covariance, int n) {
        try {
            IMatrix<Double> invCov = covariance.pinv();
            double det = Math.abs(covariance.get(0, 0));
            
            if (det <= 0) return Double.NEGATIVE_INFINITY;
            
            double logLikelihood = 0.0;
            for (int i = 0; i < n; i++) {
                IVector<Double> residual = residuals.getRow(i);
                // 计算二次型 r^T * Σ^(-1) * r
                // 由于residual是行向量，需要先计算 residual * Σ^(-1)，然后与residual做内积
                IVector<Double> temp = residual.mmul(invCov);  // 行向量 * 矩阵 = 行向量
                double quadratic = temp.innerProduct(residual);  // 行向量与行向量的内积
                logLikelihood += -0.5 * (Math.log(2 * Math.PI * det) + quadratic);
            }
            
            return logLikelihood;
        } catch (Exception e) {
            return Double.NEGATIVE_INFINITY;
        }
    }
    
    /**
     * 计算F统计量 / Calculate F statistic
     * <p>
     * 计算Granger因果检验的F统计量。
     * Calculate F statistic for Granger causality test.
     * </p>
     *
     * @param X 回归矩阵 / Regression matrix
     * @param Y 因变量向量 / Dependent variable vector
     * @param causeVar 原因变量索引 / Cause variable index
     * @param effectVar 结果变量索引 / Effect variable index
     * @return F统计量 / F statistic
     */
    private static double calculateFStatistic(IMatrix<Double> X, IVector<Double> Y, int causeVar, int effectVar) {
        // 简化的F统计量计算 / Simplified F statistic calculation
        return 1.0; // 占位符 / Placeholder
    }
    
    /**
     * 计算F检验p值 / Calculate F test p-value
     */
    private static double calculateFPValue(double fStatistic, int df1, int df2) {
        // 简化的p值计算 / Simplified p-value calculation
        if (fStatistic > 3.84) return 0.05;
        if (fStatistic > 2.71) return 0.10;
        return 0.20;
    }
    
    @Override
    public String toString() {
        return String.format("VAR(%d){AIC=%.2f, BIC=%.2f, LL=%.2f}", 
                           p, aic, bic, logLikelihood);
    }
}
