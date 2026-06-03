package com.yishape.lab.math.timeseries.model;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.Opts;
import com.yishape.lab.math.stats.Stats;

import java.util.ArrayList;
import java.util.List;

/**
 * GARCH模型实现类 / GARCH Model Implementation Class
 * <p>
 * 提供GARCH（广义自回归条件异方差）模型的实现，用于金融时间序列的波动率建模。
 * 使用项目现有的linalg包和stats包功能进行数值计算。
 * </p>
 * <p>
 * Provides GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model implementation
 * for volatility modeling of financial time series. Uses existing linalg and stats package functionality
 * for numerical computation.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class GARCHModel {
    
    private final int p; // ARCH阶数 / ARCH order
    private final int q; // GARCH阶数 / GARCH order
    private final double omega; // 常数项 / Constant term
    private final IVector<Double> alpha; // ARCH系数 / ARCH coefficients
    private final IVector<Double> beta;  // GARCH系数 / GARCH coefficients
    private final IVector<Double> variance; // 条件方差序列 / Conditional variance sequence
    private final IVector<Double> residuals; // 残差序列 / Residual sequence
    private final double logLikelihood; // 对数似然 / Log likelihood
    private final double aic; // AIC信息准则 / AIC information criterion
    private final double bic; // BIC信息准则 / BIC information criterion
    
    /**
     * 构造函数 / Constructor
     *
     * @param p ARCH阶数 / ARCH order
     * @param q GARCH阶数 / GARCH order
     * @param omega 常数项 / Constant term
     * @param alpha ARCH系数 / ARCH coefficients
     * @param beta GARCH系数 / GARCH coefficients
     * @param variance 条件方差序列 / Conditional variance sequence
     * @param residuals 残差序列 / Residual sequence
     * @param logLikelihood 对数似然 / Log likelihood
     * @param aic AIC信息准则 / AIC information criterion
     * @param bic BIC信息准则 / BIC information criterion
     */
    public GARCHModel(int p, int q, double omega, IVector<Double> alpha, IVector<Double> beta,
                     IVector<Double> variance, IVector<Double> residuals, double logLikelihood,
                     double aic, double bic) {
        this.p = p;
        this.q = q;
        this.omega = omega;
        this.alpha = alpha;
        this.beta = beta;
        this.variance = variance;
        this.residuals = residuals;
        this.logLikelihood = logLikelihood;
        this.aic = aic;
        this.bic = bic;
    }
    
    /**
     * 拟合GARCH模型 / Fit GARCH Model
     * <p>
     * 使用最大似然估计方法拟合GARCH模型。
     * Use maximum likelihood estimation to fit GARCH model.
     * </p>
     *
     * @param returns 收益率序列 / Returns series
     * @param p ARCH阶数 / ARCH order
     * @param q GARCH阶数 / GARCH order
     * @return 拟合的GARCH模型 / Fitted GARCH model
     */
    public static GARCHModel fit(IVector<Double> returns, int p, int q) {
        int n = returns.length();
        int maxLag = Math.max(p, q);
        if (n < maxLag + 10) {
            throw new IllegalArgumentException("数据长度不足以拟合GARCH模型");
        }

        double[] r = new double[n];
        for (int i = 0; i < n; i++) r[i] = returns.get(i);

        int numParams = 1 + p + q;
        IVector<Double> initTheta = Linalg.zeros(numParams);
        double initVar = returns.varValue();
        initTheta.set(0, Math.log(Math.max(initVar * 0.1, 1e-8)));
        for (int i = 0; i < p; i++) initTheta.set(1 + i, Math.log(0.1 / Math.max(p, 1)));
        for (int j = 0; j < q; j++) initTheta.set(1 + p + j, Math.log(0.8 / Math.max(q, 1)));

        IObjectiveFunction objective = theta -> {
            double[] params = unconstrainedToGarchParams(theta, p, q);
            return garchNegLogLikelihood(r, params[0],
                java.util.Arrays.copyOfRange(params, 1, 1 + p),
                java.util.Arrays.copyOfRange(params, 1 + p, params.length));
        };

        IGradientFunction gradient = theta -> numericalGradient(theta, objective);

        IOptimizer optimizer = Opts.lbfgs(1e-8, 500);
        OptResult result = optimizer.optimize(initTheta, objective, gradient);

        IVector<Double> optTheta = result.getOptimalPoint();
        double[] optParams = unconstrainedToGarchParams(optTheta, p, q);
        double optOmega = optParams[0];
        IVector<Double> optAlpha = Linalg.zeros(p);
        IVector<Double> optBeta = Linalg.zeros(q);
        for (int i = 0; i < p; i++) optAlpha.set(i, optParams[1 + i]);
        for (int j = 0; j < q; j++) optBeta.set(j, optParams[1 + p + j]);

        IVector<Double> variance = calculateConditionalVariance(returns, optOmega, optAlpha, optBeta);
        IVector<Double> residuals = returns.divide(variance.apply(Math::sqrt));
        double logLikelihood = calculateLogLikelihood(returns, variance);

        int k = 1 + p + q;
        double aic = 2 * k - 2 * logLikelihood;
        double bic = k * Math.log(n) - 2 * logLikelihood;

        return new GARCHModel(p, q, optOmega, optAlpha, optBeta, variance, residuals, logLikelihood, aic, bic);
    }
    
    /**
     * 预测条件方差 / Forecast Conditional Variance
     * <p>
     * 使用拟合的GARCH模型预测未来的条件方差。
     * Use fitted GARCH model to forecast future conditional variance.
     * </p>
     *
     * @param steps 预测步数 / Forecast steps
     * @return 条件方差预测值 / Conditional variance forecast
     */
    public IVector<Double> forecastVariance(int steps) {
        if (steps <= 0) {
            throw new IllegalArgumentException("预测步数必须为正数");
        }
        
        IVector<Double> forecast = Linalg.zeros(steps);
        int n = variance.length();
        
        for (int i = 0; i < steps; i++) {
            double varianceForecast = omega;
            
            // ARCH部分 / ARCH part
            for (int j = 0; j < Math.min(p, n - i); j++) {
                double returnValue = residuals.get(n - 1 - j);
                varianceForecast += alpha.get(j) * returnValue * returnValue;
            }
            
            // GARCH部分 / GARCH part
            for (int j = 0; j < Math.min(q, n - i); j++) {
                varianceForecast += beta.get(j) * variance.get(n - 1 - j);
            }
            
            forecast.set(i, varianceForecast);
        }
        
        return forecast;
    }
    
    /**
     * 预测收益率 / Forecast Returns
     * <p>
     * 使用GARCH模型预测未来的收益率分布。
     * Use GARCH model to forecast future returns distribution.
     * </p>
     *
     * @param steps 预测步数 / Forecast steps
     * @param confidenceLevel 置信水平 / Confidence level
     * @return 收益率预测结果 / Returns forecast result
     */
    public GARCHForecastResult forecastReturns(int steps, double confidenceLevel) {
        IVector<Double> varianceForecast = forecastVariance(steps);
        IVector<Double> meanForecast = Linalg.zeros(steps); // 假设均值为0 / Assume mean is 0
        
        // 计算置信区间 / Calculate confidence intervals
        double zScore = getZScore(confidenceLevel);
        IVector<Double> stdForecast = varianceForecast.apply(Math::sqrt);
        
        IVector<Double> lowerBound = meanForecast.sub(stdForecast.multiplyByScalar(zScore));
        IVector<Double> upperBound = meanForecast.add(stdForecast.multiplyByScalar(zScore));
        
        return new GARCHForecastResult(meanForecast, varianceForecast, lowerBound, upperBound, confidenceLevel);
    }
    
    /**
     * 计算VaR（风险价值） / Calculate VaR (Value at Risk)
     * <p>
     * 使用GARCH模型计算风险价值。
     * Use GARCH model to calculate Value at Risk.
     * </p>
     *
     * @param confidenceLevel 置信水平 / Confidence level
     * @param horizon 时间期限 / Time horizon
     * @return VaR值 / VaR value
     */
    public double calculateVaR(double confidenceLevel, int horizon) {
        IVector<Double> varianceForecast = forecastVariance(horizon);
        double totalVariance = varianceForecast.sumValue();
        double totalStd = Math.sqrt(totalVariance);
        
        double zScore = getZScore(confidenceLevel);
        return -zScore * totalStd; // VaR通常为负值 / VaR is usually negative
    }
    
    /**
     * 计算ES（期望损失） / Calculate ES (Expected Shortfall)
     * <p>
     * 使用GARCH模型计算期望损失。
     * Use GARCH model to calculate Expected Shortfall.
     * </p>
     *
     * @param confidenceLevel 置信水平 / Confidence level
     * @param horizon 时间期限 / Time horizon
     * @return ES值 / ES value
     */
    public double calculateES(double confidenceLevel, int horizon) {
        double var = calculateVaR(confidenceLevel, horizon);
        IVector<Double> varianceForecast = forecastVariance(horizon);
        double totalVariance = varianceForecast.sumValue();
        double totalStd = Math.sqrt(totalVariance);
        
        // 简化的ES计算 / Simplified ES calculation
        double zScore = getZScore(confidenceLevel);
        double phi = Stats.norm().pdf(zScore);
        double es = -totalStd * phi / (1 - confidenceLevel);
        
        return es;
    }
    
    /**
     * 模型诊断 / Model Diagnostics
     * <p>
     * 进行GARCH模型诊断，包括残差分析、ARCH效应检验等。
     * Perform GARCH model diagnostics including residual analysis, ARCH effect tests, etc.
     * </p>
     *
     * @return 诊断结果 / Diagnostic results
     */
    public GARCHDiagnostics diagnose() {
        // 标准化残差 / Standardized residuals
        IVector<Double> standardizedResiduals = residuals.divide(variance.apply(Math::sqrt));
        
        // 残差统计 / Residual statistics
        double residualMean = standardizedResiduals.meanValue();
        double residualStd = standardizedResiduals.stdValue();
        double residualSkewness = calculateSkewness(standardizedResiduals);
        double residualKurtosis = calculateKurtosis(standardizedResiduals);
        
        // ARCH效应检验 / ARCH effect test
        double archStatistic = calculateARCHStatistic(standardizedResiduals, p);
        double archPValue = calculateARCHPValue(archStatistic, p, standardizedResiduals.length());
        
        // Ljung-Box检验 / Ljung-Box test
        double ljungBoxStatistic = calculateLjungBoxStatistic(standardizedResiduals, p + q);
        double ljungBoxPValue = calculateLjungBoxPValue(ljungBoxStatistic, p + q, standardizedResiduals.length());
        
        // 正态性检验 / Normality test
        double shapiroWilkStatistic = calculateShapiroWilkStatistic(standardizedResiduals);
        double shapiroWilkPValue = calculateShapiroWilkPValue(shapiroWilkStatistic, standardizedResiduals.length());
        
        return new GARCHDiagnostics(
            standardizedResiduals, residualMean, residualStd, residualSkewness, residualKurtosis,
            archStatistic, archPValue, ljungBoxStatistic, ljungBoxPValue,
            shapiroWilkStatistic, shapiroWilkPValue
        );
    }
    
    /**
     * 自动选择GARCH模型 / Automatic GARCH Model Selection
     * <p>
     * 使用AIC或BIC准则自动选择最优的GARCH模型参数。
     * Automatically select optimal GARCH model parameters using AIC or BIC criteria.
     * </p>
     *
     * @param returns 收益率序列 / Returns series
     * @param maxP 最大ARCH阶数 / Maximum ARCH order
     * @param maxQ 最大GARCH阶数 / Maximum GARCH order
     * @param criterion 选择准则 / Selection criterion
     * @return 最优GARCH模型 / Optimal GARCH model
     */
    public static GARCHModel autoFit(IVector<Double> returns, int maxP, int maxQ, TimeSeriesModelFactory.SelectionCriterion criterion) {
        GARCHModel bestModel = null;
        double bestCriterion = Double.POSITIVE_INFINITY;

        for (int p = 1; p <= maxP; p++) {
            for (int q = 1; q <= maxQ; q++) {
                try {
                    GARCHModel model = fit(returns, p, q);
                    double criterionValue = (criterion == TimeSeriesModelFactory.SelectionCriterion.AIC) ? model.aic : model.bic;

                    if (criterionValue < bestCriterion) {
                        bestCriterion = criterionValue;
                        bestModel = model;
                    }
                } catch (Exception e) {
                    // 跳过无效的模型参数组合 / Skip invalid model parameter combinations
                    continue;
                }
            }
        }

        if (bestModel == null) {
            throw new RuntimeException("无法找到合适的GARCH模型");
        }

        return bestModel;
    }
    
    // ========== Getter方法 / Getter Methods ==========
    
    public int getP() { return p; }
    public int getQ() { return q; }
    public double getOmega() { return omega; }
    public IVector<Double> getAlpha() { return alpha; }
    public IVector<Double> getBeta() { return beta; }
    public IVector<Double> getVariance() { return variance; }
    public IVector<Double> getResiduals() { return residuals; }
    public double getLogLikelihood() { return logLikelihood; }
    public double getAic() { return aic; }
    public double getBic() { return bic; }
    
    // ========== 结果类 / Result Classes ==========
    
    /**
     * GARCH预测结果类 / GARCH Forecast Result Class
     */
    public static class GARCHForecastResult {
        public final IVector<Double> meanForecast;
        public final IVector<Double> varianceForecast;
        public final IVector<Double> lowerBound;
        public final IVector<Double> upperBound;
        public final double confidenceLevel;
        
        public GARCHForecastResult(IVector<Double> meanForecast, IVector<Double> varianceForecast,
                                 IVector<Double> lowerBound, IVector<Double> upperBound, double confidenceLevel) {
            this.meanForecast = meanForecast;
            this.varianceForecast = varianceForecast;
            this.lowerBound = lowerBound;
            this.upperBound = upperBound;
            this.confidenceLevel = confidenceLevel;
        }
    }
    
    /**
     * GARCH诊断结果类 / GARCH Diagnostics Result Class
     * <p>
     * 存储GARCH模型诊断的结果，包括标准化残差统计量和各种检验统计量。
     * Stores GARCH model diagnostic results including standardized residual statistics and various test statistics.
     * </p>
     *
     * @author lteb2
     * @version 1.0
     * @since 1.0
     */
    public static class GARCHDiagnostics {
        public final IVector<Double> standardizedResiduals;
        public final double residualMean;
        public final double residualStd;
        public final double residualSkewness;
        public final double residualKurtosis;
        public final double archStatistic;
        public final double archPValue;
        public final double ljungBoxStatistic;
        public final double ljungBoxPValue;
        public final double shapiroWilkStatistic;
        public final double shapiroWilkPValue;

        /**
         * 构造函数 / Constructor
         *
         * @param standardizedResiduals 标准化残差序列 / Standardized residuals series
         * @param residualMean 残差均值 / Residual mean
         * @param residualStd 残差标准差 / Residual standard deviation
         * @param residualSkewness 残差偏度 / Residual skewness
         * @param residualKurtosis 残差峰度 / Residual kurtosis
         * @param archStatistic ARCH效应统计量 / ARCH effect statistic
         * @param archPValue ARCH效应p值 / ARCH effect p-value
         * @param ljungBoxStatistic Ljung-Box统计量 / Ljung-Box statistic
         * @param ljungBoxPValue Ljung-Box p值 / Ljung-Box p-value
         * @param shapiroWilkStatistic Shapiro-Wilk统计量 / Shapiro-Wilk statistic
         * @param shapiroWilkPValue Shapiro-Wilk p值 / Shapiro-Wilk p-value
         */
        public GARCHDiagnostics(IVector<Double> standardizedResiduals, double residualMean, double residualStd,
                              double residualSkewness, double residualKurtosis, double archStatistic,
                              double archPValue, double ljungBoxStatistic, double ljungBoxPValue,
                              double shapiroWilkStatistic, double shapiroWilkPValue) {
            this.standardizedResiduals = standardizedResiduals;
            this.residualMean = residualMean;
            this.residualStd = residualStd;
            this.residualSkewness = residualSkewness;
            this.residualKurtosis = residualKurtosis;
            this.archStatistic = archStatistic;
            this.archPValue = archPValue;
            this.ljungBoxStatistic = ljungBoxStatistic;
            this.ljungBoxPValue = ljungBoxPValue;
            this.shapiroWilkStatistic = shapiroWilkStatistic;
            this.shapiroWilkPValue = shapiroWilkPValue;
        }
    }
    
    // ========== 私有辅助方法 / Private Helper Methods ==========

    /**
     * 将无约束参数变换为满足 GARCH 约束的参数。
     * Transform unconstrained parameters to GARCH-constrained parameters.
     * omega = exp(theta_0), alpha_i = exp(theta_i) / S, beta_j = exp(theta_j) / S,
     * where S = 1 + sum(exp(theta_alpha)) + sum(exp(theta_beta)).
     */
    private static double[] unconstrainedToGarchParams(IVector<Double> theta, int p, int q) {
        int n = theta.length();
        double[] params = new double[n];
        params[0] = Math.exp(theta.get(0));
        double sum = 1.0;
        for (int i = 1; i < n; i++) {
            sum += Math.exp(theta.get(i));
        }
        for (int i = 1; i < n; i++) {
            params[i] = Math.exp(theta.get(i)) / sum;
        }
        return params;
    }

    /**
     * GARCH 负对数似然（使用原始 double[] 以提升性能）。
     * GARCH negative log-likelihood (using raw double[] for performance).
     */
    private static double garchNegLogLikelihood(double[] returns, double omega, double[] alpha, double[] beta) {
        int n = returns.length;
        int p = alpha.length;
        int q = beta.length;
        int maxLag = Math.max(p, q);

        double[] sigma2 = new double[n];
        double initVar = 0.0;
        for (int i = 0; i < n; i++) initVar += returns[i] * returns[i];
        initVar /= n;
        for (int i = 0; i < maxLag; i++) sigma2[i] = initVar;

        double nll = 0.0;
        for (int t = maxLag; t < n; t++) {
            double s2 = omega;
            for (int i = 0; i < p; i++) {
                s2 += alpha[i] * returns[t - i - 1] * returns[t - i - 1];
            }
            for (int j = 0; j < q; j++) {
                s2 += beta[j] * sigma2[t - j - 1];
            }
            sigma2[t] = s2;
            nll += 0.5 * (Math.log(2.0 * Math.PI * s2) + returns[t] * returns[t] / s2);
        }
        return nll;
    }

    /** 中心差分数值梯度 / Central-difference numerical gradient */
    private static IVector<Double> numericalGradient(IVector<Double> theta, IObjectiveFunction f) {
        int n = theta.length();
        IVector<Double> grad = Linalg.zeros(n);
        double h = 1e-6;
        for (int i = 0; i < n; i++) {
            double orig = theta.get(i);
            theta.set(i, orig + h);
            double fp = f.computeObjective(theta);
            theta.set(i, orig - h);
            double fm = f.computeObjective(theta);
            theta.set(i, orig);
            grad.set(i, (fp - fm) / (2.0 * h));
        }
        return grad;
    }
    
    /**
     * 计算条件方差 / Calculate conditional variance
     * <p>
     * 根据GARCH模型公式递归计算条件方差序列。
     * Recursively calculate conditional variance series according to GARCH model formula.
     * </p>
     *
     * @param returns 收益率序列 / Returns series
     * @param omega 常数项 / Constant term
     * @param alpha ARCH系数向量 / ARCH coefficients vector
     * @param beta GARCH系数向量 / GARCH coefficients vector
     * @return 条件方差序列 / Conditional variance series
     */
    private static IVector<Double> calculateConditionalVariance(IVector<Double> returns, double omega,
                                                               IVector<Double> alpha, IVector<Double> beta) {
        int n = returns.length();
        int p = alpha.length();
        int q = beta.length();
        int maxLag = Math.max(p, q);
        
        IVector<Double> variance = Linalg.zeros(n);
        
        // 初始化 / Initialize
        double initialVariance = returns.varValue();
        for (int i = 0; i < maxLag; i++) {
            variance.set(i, initialVariance);
        }
        
        // 递归计算条件方差 / Recursively calculate conditional variance
        for (int i = maxLag; i < n; i++) {
            double varianceValue = omega;
            
            // ARCH部分 / ARCH part
            for (int j = 0; j < p; j++) {
                double returnValue = returns.get(i - j - 1);
                varianceValue += alpha.get(j) * returnValue * returnValue;
            }
            
            // GARCH部分 / GARCH part
            for (int j = 0; j < q; j++) {
                varianceValue += beta.get(j) * variance.get(i - j - 1);
            }
            
            variance.set(i, varianceValue);
        }
        
        return variance;
    }
    
    /**
     * 计算对数似然 / Calculate log likelihood
     * <p>
     * 计算GARCH模型的对数似然值。
     * Calculate log likelihood of GARCH model.
     * </p>
     *
     * @param returns 收益率序列 / Returns series
     * @param variance 条件方差序列 / Conditional variance series
     * @return 对数似然值 / Log likelihood value
     */
    private static double calculateLogLikelihood(IVector<Double> returns, IVector<Double> variance) {
        int n = returns.length();
        double logLikelihood = 0.0;
        
        for (int i = 0; i < n; i++) {
            double var = variance.get(i);
            if (var > 0) {
                double residual = returns.get(i);
                logLikelihood += -0.5 * Math.log(2 * Math.PI * var) - 0.5 * residual * residual / var;
            }
        }
        
        return logLikelihood;
    }
    
    /**
     * 计算偏度 / Calculate skewness
     * <p>
     * 计算时间序列数据的偏度（三阶中心矩标准化）。
     * Calculate skewness of time series data (standardized third central moment).
     * </p>
     *
     * @param data 输入数据 / Input data
     * @return 偏度值 / Skewness value
     */
    private static double calculateSkewness(IVector<Double> data) {
        double mean = data.meanValue();
        double std = data.stdValue();

        if (std == 0) return 0.0;

        IVector<Double> centered = data.subScalar(mean);
        IVector<Double> cubed = centered.apply(x -> x * x * x);

        return cubed.meanValue() / (std * std * std);
    }
    
    /**
     * 计算峰度 / Calculate kurtosis
     * <p>
     * 计算时间序列数据的峰度（四阶中心矩标准化，减去正态分布的峰度）。
     * Calculate kurtosis of time series data (standardized fourth central moment minus normal distribution kurtosis).
     * </p>
     *
     * @param data 输入数据 / Input data
     * @return 峰度值 / Kurtosis value
     */
    private static double calculateKurtosis(IVector<Double> data) {
        double mean = data.meanValue();
        double std = data.stdValue();

        if (std == 0) return 0.0;

        IVector<Double> centered = data.subScalar(mean);
        IVector<Double> fourth = centered.apply(x -> x * x * x * x);

        return fourth.meanValue() / (std * std * std * std) - 3.0;
    }
    
    /**
     * 计算ARCH统计量 / Calculate ARCH statistic
     * <p>
     * 计算ARCH效应检验的统计量。
     * Calculate statistic for ARCH effect test.
     * </p>
     *
     * @param residuals 残差序列 / Residuals series
     * @param lag 滞后期数 / Lag order
     * @return ARCH统计量 / ARCH statistic
     */
    private static double calculateARCHStatistic(IVector<Double> residuals, int lag) {
        int n = residuals.length();
        double statistic = 0.0;
        
        for (int i = 1; i <= lag; i++) {
            double autocorr = autocorrelation(residuals, i);
            statistic += (n - i) * autocorr * autocorr / (n * (n + 2));
        }
        
        return n * (n + 2) * statistic;
    }
    
    /**
     * 计算ARCH p值 / Calculate ARCH p-value
     * <p>
     * 根据ARCH统计量计算p值。
     * Calculate p-value based on ARCH statistic.
     * </p>
     *
     * @param statistic ARCH统计量 / ARCH statistic
     * @param df 自由度 / Degrees of freedom
     * @param n 样本数量 / Sample size
     * @return p值 / P-value
     */
    private static double calculateARCHPValue(double statistic, int df, int n) {
        // 简化的p值计算 / Simplified p-value calculation
        if (statistic < 3.84) return 0.95;
        if (statistic < 6.63) return 0.90;
        if (statistic < 7.81) return 0.80;
        return 0.50;
    }
    
    /**
     * 计算自相关函数 / Calculate autocorrelation function
     * <p>
     * 计算时间序列指定滞后阶数的自相关系数。
     * Calculate autocorrelation coefficient at specified lag order for time series.
     * </p>
     *
     * @param data 输入数据 / Input data
     * @param lag 滞后期数 / Lag order
     * @return 自相关系数 / Autocorrelation coefficient
     */
    private static double autocorrelation(IVector<Double> data, int lag) {
        int n = data.length();
        if (lag >= n) return 0.0;
        
        double mean = data.meanValue();
        double numerator = 0.0;
        double denominator = 0.0;
        
        for (int i = 0; i < n - lag; i++) {
            numerator += (data.get(i) - mean) * (data.get(i + lag) - mean);
        }
        
        for (int i = 0; i < n; i++) {
            denominator += (data.get(i) - mean) * (data.get(i) - mean);
        }
        
        if (denominator == 0) return 0.0;
        return numerator / denominator;
    }
    
    /**
     * 计算Ljung-Box统计量 / Calculate Ljung-Box statistic
     * <p>
     * 计算Ljung-Box检验的统计量，用于检验残差是否存在自相关。
     * Calculate Ljung-Box test statistic for testing autocorrelation in residuals.
     * </p>
     *
     * @param residuals 残差序列 / Residuals series
     * @param lag 滞后期数 / Lag order
     * @return Ljung-Box统计量 / Ljung-Box statistic
     */
    private static double calculateLjungBoxStatistic(IVector<Double> residuals, int lag) {
        int n = residuals.length();
        double statistic = 0.0;
        
        for (int i = 1; i <= lag; i++) {
            double autocorr = autocorrelation(residuals, i);
            statistic += (n - i) * autocorr * autocorr / (n * (n + 2));
        }
        
        return n * (n + 2) * statistic;
    }
    
    /**
     * 计算Ljung-Box p值 / Calculate Ljung-Box p-value
     * <p>
     * 根据Ljung-Box统计量计算p值。
     * Calculate p-value based on Ljung-Box statistic.
     * </p>
     *
     * @param statistic Ljung-Box统计量 / Ljung-Box statistic
     * @param df 自由度 / Degrees of freedom
     * @param n 样本数量 / Sample size
     * @return p值 / P-value
     */
    private static double calculateLjungBoxPValue(double statistic, int df, int n) {
        // 简化的p值计算 / Simplified p-value calculation
        if (statistic < 3.84) return 0.95;
        if (statistic < 6.63) return 0.90;
        if (statistic < 7.81) return 0.80;
        return 0.50;
    }
    
    /**
     * 计算Shapiro-Wilk统计量 / Calculate Shapiro-Wilk statistic
     * <p>
     * 计算Shapiro-Wilk正态性检验的统计量。
     * Calculate Shapiro-Wilk normality test statistic.
     * </p>
     *
     * @param data 输入数据 / Input data
     * @return Shapiro-Wilk统计量 / Shapiro-Wilk statistic
     */
    private static double calculateShapiroWilkStatistic(IVector<Double> data) {
        // 简化的Shapiro-Wilk检验实现 / Simplified Shapiro-Wilk test implementation
        int n = data.length();
        if (n < 3) return 1.0;
        
        // 排序数据 / Sort data
        IVector<Double> sorted = data.sort();
        
        // 计算统计量 / Calculate statistic
        double numerator = 0.0;
        double denominator = 0.0;
        
        for (int i = 0; i < n; i++) {
            double expected = Stats.norm().ppf((i + 1) / (double) (n + 1));
            numerator += sorted.get(i) * expected;
            denominator += sorted.get(i) * sorted.get(i);
        }
        
        if (denominator == 0) return 0.0;
        return (numerator * numerator) / denominator;
    }
    
    /**
     * 计算Shapiro-Wilk p值 / Calculate Shapiro-Wilk p-value
     * <p>
     * 根据Shapiro-Wilk统计量计算p值。
     * Calculate p-value based on Shapiro-Wilk statistic.
     * </p>
     *
     * @param statistic Shapiro-Wilk统计量 / Shapiro-Wilk statistic
     * @param n 样本数量 / Sample size
     * @return p值 / P-value
     */
    private static double calculateShapiroWilkPValue(double statistic, int n) {
        // 简化的p值计算 / Simplified p-value calculation
        if (statistic > 0.95) return 0.90;
        if (statistic > 0.90) return 0.80;
        if (statistic > 0.85) return 0.70;
        return 0.50;
    }
    
    /**
     * 获取Z分数 / Get Z-score
     */
    private static double getZScore(double confidenceLevel) {
        if (confidenceLevel == 0.95) return 1.96;
        if (confidenceLevel == 0.90) return 1.645;
        if (confidenceLevel == 0.99) return 2.576;
        return 1.96; // 默认95%置信水平 / Default 95% confidence level
    }
    
    @Override
    public String toString() {
        return String.format("GARCH(%d,%d){AIC=%.2f, BIC=%.2f, LL=%.2f}", 
                           p, q, aic, bic, logLikelihood);
    }
}
