package com.yishape.lab.math.timeseries.model;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;

import java.time.LocalDateTime;

/**
 * 统一ARIMA模型实现类 / Unified ARIMA Model Implementation
 * <p>
 * 实现ITimeSeriesModel接口的ARIMA模型，提供统一的ARIMA模型功能。
 * 整合原有的ARIMAModel类功能，并实现统一的接口规范。
 * </p>
 * <p>
 * ARIMA model implementation that implements ITimeSeriesModel interface,
 * providing unified ARIMA model functionality. Integrates original ARIMAModel
 * class functionality and implements unified interface specifications.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class UnifiedARIMAModel implements ITimeSeriesModel {
    
    private final int p; // AR阶数 / AR order
    private final int d; // 差分阶数 / Differencing order
    private final int q; // MA阶数 / MA order
    private final IVector<Double> arCoeffs; // AR系数 / AR coefficients
    private final IVector<Double> maCoeffs; // MA系数 / MA coefficients
    private final double sigma2; // 噪声方差 / Noise variance
    private final double logLikelihood; // 对数似然 / Log likelihood
    private final double aic; // AIC信息准则 / AIC information criterion
    private final double bic; // BIC信息准则 / BIC information criterion
    
    private ModelState state = ModelState.UNFITTED;
    private IVector<Double> originalData;
    private IVector<Double> fittedValues;
    private IVector<Double> residuals;
    private String modelName;
    
    /**
     * 构造函数 / Constructor
     *
     * @param p AR阶数 / AR order
     * @param d 差分阶数 / Differencing order
     * @param q MA阶数 / MA order
     * @param arCoeffs AR系数 / AR coefficients
     * @param maCoeffs MA系数 / MA coefficients
     * @param sigma2 噪声方差 / Noise variance
     * @param logLikelihood 对数似然 / Log likelihood
     * @param aic AIC信息准则 / AIC information criterion
     * @param bic BIC信息准则 / BIC information criterion
     */
    public UnifiedARIMAModel(int p, int d, int q, IVector<Double> arCoeffs, IVector<Double> maCoeffs,
                           double sigma2, double logLikelihood, double aic, double bic) {
        this.p = p;
        this.d = d;
        this.q = q;
        this.arCoeffs = arCoeffs;
        this.maCoeffs = maCoeffs;
        this.sigma2 = sigma2;
        this.logLikelihood = logLikelihood;
        this.aic = aic;
        this.bic = bic;
        this.modelName = String.format("ARIMA(%d,%d,%d)", p, d, q);
    }
    
    /**
     * 拟合ARIMA模型 / Fit ARIMA Model
     *
     * @param data 输入时间序列 / Input time series
     * @param p AR阶数 / AR order
     * @param d 差分阶数 / Differencing order
     * @param q MA阶数 / MA order
     * @return 拟合的ARIMA模型 / Fitted ARIMA model
     */
    public static UnifiedARIMAModel fit(IVector<Double> data, int p, int d, int q) {
        if (data.length() < Math.max(p, q) + 10) {
            throw new IllegalArgumentException("数据长度不足以拟合ARIMA模型");
        }
        
        // 差分处理 / Differencing
        IVector<Double> diffData = data;
        for (int i = 0; i < d; i++) {
            diffData = difference(diffData);
        }
        
        // 初始化参数 / Initialize parameters
        IVector<Double> arCoeffs = Linalg.zeros(p);
        IVector<Double> maCoeffs = Linalg.zeros(q);
        
        // 使用Yule-Walker方程估计AR参数 / Estimate AR parameters using Yule-Walker equations
        if (p > 0) {
            arCoeffs = estimateARParameters(diffData, p);
        }
        
        // 使用矩估计方法估计MA参数 / Estimate MA parameters using method of moments
        if (q > 0) {
            maCoeffs = estimateMAParameters(diffData, q, arCoeffs);
        }
        
        // 计算残差和方差 / Calculate residuals and variance
        IVector<Double> residuals = calculateResiduals(diffData, arCoeffs, maCoeffs);
        double sigma2 = residuals.multiply(residuals).meanValue();
        
        // 计算信息准则 / Calculate information criteria
        double logLikelihood = calculateLogLikelihood(residuals, sigma2);
        int n = diffData.length();
        int k = p + q + 1; // 参数个数 / Number of parameters
        double aic = 2 * k - 2 * logLikelihood;
        double bic = k * Math.log(n) - 2 * logLikelihood;
        
        UnifiedARIMAModel model = new UnifiedARIMAModel(p, d, q, arCoeffs, maCoeffs, sigma2, logLikelihood, aic, bic);
        model.originalData = data;
        model.state = ModelState.FITTED;
        model.fittedValues = calculateFittedValues(data, arCoeffs, maCoeffs, d);
        model.residuals = data.sub(model.fittedValues);
        
        return model;
    }
    
    @Override
    public ModelType getModelType() {
        return ModelType.ARIMA;
    }
    
    @Override
    public ModelState getModelState() {
        return state;
    }
    
    @Override
    public String getModelName() {
        return modelName;
    }
    
    @Override
    public int getParameterCount() {
        return p + q + 1; // AR + MA + 常数项 / AR + MA + constant
    }
    
    @Override
    public double[] getInformationCriteria() {
        return new double[]{aic, bic, logLikelihood};
    }
    
    @Override
    public double forecastOneStep() throws IllegalStateException {
        if (state != ModelState.FITTED) {
            throw new IllegalStateException("模型未拟合");
        }

        if (p == 0 && q == 0) {
            if (d == 0) {
                return originalData.meanValue();
            }
            return originalData.get(originalData.length() - 1);
        }
        
        // 简化的单步预测实现 / Simplified one-step forecast implementation
        double value = 0.0;
        
        // AR部分 / AR part
        for (int j = 0; j < Math.min(p, originalData.length()); j++) {
            value += arCoeffs.get(j) * originalData.get(originalData.length() - j - 1);
        }
        
        // MA部分（简化） / MA part (simplified)
        for (int j = 0; j < Math.min(q, residuals.length()); j++) {
            value += maCoeffs.get(j) * residuals.get(residuals.length() - j - 1);
        }
        
        return value;
    }
    
    @Override
    public IVector<Double> forecast(int steps) throws IllegalStateException, IllegalArgumentException {
        if (state != ModelState.FITTED) {
            throw new IllegalStateException("模型未拟合");
        }
        if (steps <= 0) {
            throw new IllegalArgumentException("预测步数必须为正数");
        }
        
        IVector<Double> forecast = Linalg.zeros(steps);

        // 白噪声 / 纯季节性差分：最优点预测为序列均值（水平）或末值（含积分项时常用末值外推）
        if (p == 0 && q == 0) {
            double level;
            if (d == 0) {
                level = originalData.meanValue();
            } else {
                level = originalData.get(originalData.length() - 1);
            }
            for (int i = 0; i < steps; i++) {
                forecast.set(i, level);
            }
            return forecast;
        }
        
        // 简化的多步预测：MA 未来新息置 0；仅用已得预测值递推 AR 部分 / Simplified: zero future innovations
        for (int i = 0; i < steps; i++) {
            double value = 0.0;
            for (int j = 0; j < Math.min(p, i); j++) {
                value += arCoeffs.get(j) * forecast.get(i - j - 1);
            }
            forecast.set(i, value);
        }
        
        return forecast;
    }
    
    @Override
    public ITimeSeriesForecastResult forecastWithConfidence(int steps, double confidenceLevel) 
            throws IllegalStateException, IllegalArgumentException {
        if (state != ModelState.FITTED) {
            throw new IllegalStateException("模型未拟合");
        }
        if (steps <= 0) {
            throw new IllegalArgumentException("预测步数必须为正数");
        }
        if (confidenceLevel <= 0 || confidenceLevel >= 1) {
            throw new IllegalArgumentException("置信水平必须在0和1之间");
        }
        
        IVector<Double> forecast = forecast(steps);
        
        // 计算预测误差的标准差 / Calculate forecast error standard deviation
        IVector<Double> forecastStd = Linalg.zeros(steps);
        for (int i = 0; i < steps; i++) {
            double variance = sigma2 * (1 + i); // 简化的方差计算 / Simplified variance calculation
            forecastStd.set(i, Math.sqrt(variance));
        }
        
        // 计算Z分数 / Calculate Z-score
        double zScore = getZScore(confidenceLevel);
        
        // 计算置信区间 / Calculate confidence intervals
        IVector<Double> lowerBound = forecast.sub(forecastStd.multiplyByScalar(zScore));
        IVector<Double> upperBound = forecast.add(forecastStd.multiplyByScalar(zScore));
        
        // 生成时间点 / Generate time points
        LocalDateTime[] timePoints = generateTimePoints(steps);
        
        // 计算误差指标 / Calculate error metrics
        double[] errorMetrics = new double[4]; // MSE, MAE, MAPE, RMSE
        // 这里可以添加实际的误差计算逻辑
        
        return new TimeSeriesForecastResult(forecast, lowerBound, upperBound, forecastStd,
                                          confidenceLevel, ModelType.ARIMA, timePoints, errorMetrics);
    }
    
    @Override
    public ITimeSeriesDiagnostics diagnose() throws IllegalStateException {
        if (state != ModelState.FITTED) {
            throw new IllegalStateException("模型未拟合");
        }
        
        return new ARIMADiagnostics(residuals, originalData, fittedValues, 
                                  arCoeffs, maCoeffs, sigma2, aic, bic, logLikelihood);
    }
    
    @Override
    public IVector<Double> getResiduals() throws IllegalStateException {
        if (state != ModelState.FITTED) {
            throw new IllegalStateException("模型未拟合");
        }
        return residuals.copy();
    }
    
    @Override
    public IVector<Double> getFittedValues() throws IllegalStateException {
        if (state != ModelState.FITTED) {
            throw new IllegalStateException("模型未拟合");
        }
        return fittedValues.copy();
    }
    
    @Override
    public double[] calculateForecastError(IVector<Double> actual, IVector<Double> forecast) {
        if (actual.length() != forecast.length()) {
            throw new IllegalArgumentException("实际值和预测值长度不匹配");
        }
        
        IVector<Double> errors = actual.sub(forecast);
        double mse = errors.multiply(errors).meanValue();
        double mae = errors.apply(Math::abs).meanValue();
        double mape = 0.0;
        double rmse = Math.sqrt(mse);
        
        // 计算MAPE / Calculate MAPE
        for (int i = 0; i < actual.length(); i++) {
            if (Math.abs(actual.get(i)) > 1e-10) {
                mape += Math.abs((actual.get(i) - forecast.get(i)) / actual.get(i));
            }
        }
        mape = mape / actual.length() * 100;
        
        return new double[]{mse, mae, mape, rmse};
    }
    
    @Override
    public boolean isValid() {
        return state == ModelState.FITTED && 
               !Double.isNaN(aic) && !Double.isNaN(bic) && 
               !Double.isInfinite(aic) && !Double.isInfinite(bic);
    }
    
    @Override
    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("ARIMA模型摘要 / ARIMA Model Summary\n");
        sb.append("=====================================\n");
        sb.append(String.format("模型名称 / Model Name: %s\n", modelName));
        sb.append(String.format("模型状态 / Model State: %s\n", state));
        sb.append(String.format("AR阶数 / AR Order (p): %d\n", p));
        sb.append(String.format("差分阶数 / Differencing Order (d): %d\n", d));
        sb.append(String.format("MA阶数 / MA Order (q): %d\n", q));
        sb.append(String.format("参数数量 / Parameter Count: %d\n", getParameterCount()));
        sb.append(String.format("AIC: %.4f\n", aic));
        sb.append(String.format("BIC: %.4f\n", bic));
        sb.append(String.format("对数似然 / Log Likelihood: %.4f\n", logLikelihood));
        sb.append(String.format("噪声方差 / Noise Variance: %.6f\n", sigma2));
        sb.append(String.format("模型有效性 / Model Validity: %s\n", isValid() ? "有效 / Valid" : "无效 / Invalid"));
        
        if (state == ModelState.FITTED) {
            sb.append("\n拟合信息 / Fitting Information:\n");
            sb.append(String.format("数据长度 / Data Length: %d\n", originalData.length()));
            sb.append(String.format("残差均值 / Residual Mean: %.6f\n", residuals.meanValue()));
            sb.append(String.format("残差标准差 / Residual Std: %.6f\n", residuals.stdValue()));
        }
        
        return sb.toString();
    }
    
    @Override
    public void reset() {
        state = ModelState.UNFITTED;
        originalData = null;
        fittedValues = null;
        residuals = null;
    }
    
    @Override
    public ITimeSeriesModel clone() {
        UnifiedARIMAModel cloned = new UnifiedARIMAModel(p, d, q, arCoeffs.copy(), maCoeffs.copy(),
                                                        sigma2, logLikelihood, aic, bic);
        cloned.state = this.state;
        cloned.originalData = this.originalData != null ? this.originalData.copy() : null;
        cloned.fittedValues = this.fittedValues != null ? this.fittedValues.copy() : null;
        cloned.residuals = this.residuals != null ? this.residuals.copy() : null;
        cloned.modelName = this.modelName;
        return cloned;
    }
    
    // ========== Getter方法 / Getter Methods ==========
    
    public int getP() { return p; }
    public int getD() { return d; }
    public int getQ() { return q; }
    public IVector<Double> getArCoeffs() { return arCoeffs; }
    public IVector<Double> getMaCoeffs() { return maCoeffs; }
    public double getSigma2() { return sigma2; }
    public double getLogLikelihood() { return logLikelihood; }
    public double getAic() { return aic; }
    public double getBic() { return bic; }
    
    // ========== 私有辅助方法 / Private Helper Methods ==========
    
    /**
     * 差分 / Differencing
     */
    private static IVector<Double> difference(IVector<Double> data) {
        int length = data.length();
        if (length < 2) return Linalg.zeros(0);
        
        IVector<Double> diff = Linalg.zeros(length - 1);
        for (int i = 1; i < length; i++) {
            diff.set(i - 1, data.get(i) - data.get(i - 1));
        }
        
        return diff;
    }
    
    /**
     * 估计AR参数 / Estimate AR parameters
     */
    private static IVector<Double> estimateARParameters(IVector<Double> data, int p) {
        int n = data.length();
        if (n < p + 1) return Linalg.zeros(p);
        
        // 构建Yule-Walker方程 / Build Yule-Walker equations
        IMatrix<Double> R = Linalg.zeros(p, p);
        IVector<Double> r = Linalg.zeros(p);
        
        // 计算自相关函数 / Calculate autocorrelation function
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                R.set(i, j, autocorrelation(data, Math.abs(i - j)));
            }
            r.set(i, autocorrelation(data, i + 1));
        }
        
        // 求解线性方程组 / Solve linear system
        try {
            IMatrix<Double> RInv = R.pinv();
            IVector<Double> arCoeffs = RInv.mmul(r);
            return arCoeffs;
        } catch (Exception e) {
            // 如果矩阵不可逆，返回零向量 / If matrix is not invertible, return zero vector
            return Linalg.zeros(p);
        }
    }
    
    /**
     * 估计MA参数 / Estimate MA parameters
     */
    private static IVector<Double> estimateMAParameters(IVector<Double> data, int q, IVector<Double> arCoeffs) {
        // 简化的MA参数估计 / Simplified MA parameter estimation
        IVector<Double> maCoeffs = Linalg.zeros(q);
        
        // 使用矩估计方法 / Use method of moments
        for (int i = 0; i < q; i++) {
            double autocorr = autocorrelation(data, i + 1);
            maCoeffs.set(i, autocorr * 0.5); // 简化的估计 / Simplified estimation
        }
        
        return maCoeffs;
    }
    
    /**
     * 计算自相关函数 / Calculate autocorrelation function
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
     * 计算残差 / Calculate residuals
     */
    private static IVector<Double> calculateResiduals(IVector<Double> data, IVector<Double> arCoeffs, IVector<Double> maCoeffs) {
        int n = data.length();
        int p = arCoeffs.length();
        int q = maCoeffs.length();
        int maxLag = Math.max(p, q);
        
        if (n <= maxLag) return Linalg.zeros(0);
        
        IVector<Double> residuals = Linalg.zeros(n - maxLag);
        
        for (int i = maxLag; i < n; i++) {
            double predicted = 0.0;
            
            // AR部分 / AR part
            for (int j = 0; j < p; j++) {
                predicted += arCoeffs.get(j) * data.get(i - j - 1);
            }
            
            // MA部分（简化） / MA part (simplified)
            for (int j = 0; j < q; j++) {
                predicted += maCoeffs.get(j) * residuals.get(i - maxLag - j - 1);
            }
            
            residuals.set(i - maxLag, data.get(i) - predicted);
        }
        
        return residuals;
    }
    
    /**
     * 计算拟合值 / Calculate fitted values
     */
    private static IVector<Double> calculateFittedValues(IVector<Double> data, IVector<Double> arCoeffs,
                                                        IVector<Double> maCoeffs, int d) {
        int n = data.length();
        int p = arCoeffs.length();
        int q = maCoeffs.length();

        // 差分处理 / Differencing
        IVector<Double> diffData = data;
        for (int i = 0; i < d; i++) {
            diffData = difference(diffData);
        }
        int dn = diffData.length();
        int maxLag = Math.max(p, q);

        // 计算差分序列的拟合值 / Compute fitted values on differenced series
        IVector<Double> fittedDiff = Linalg.zeros(dn);
        IVector<Double> residDiff = Linalg.zeros(Math.max(0, dn - maxLag));

        for (int i = 0; i < Math.min(maxLag, dn); i++) {
            fittedDiff.set(i, diffData.get(i));
        }

        for (int i = maxLag; i < dn; i++) {
            double predicted = 0.0;
            for (int j = 0; j < p; j++) {
                predicted += arCoeffs.get(j) * diffData.get(i - j - 1);
            }
            for (int j = 0; j < q; j++) {
                predicted += maCoeffs.get(j) * residDiff.get(i - maxLag - j - 1);
            }
            fittedDiff.set(i, predicted);
            residDiff.set(i - maxLag, diffData.get(i) - predicted);
        }

        if (d == 0) {
            return fittedDiff;
        }

        // 逆差分：从差分拟合值重建原始尺度的拟合值 / Inverse differencing
        IVector<Double> fitted = Linalg.zeros(n);
        for (int i = 0; i < d; i++) {
            fitted.set(i, data.get(i));
        }

        if (d == 1) {
            for (int i = d; i < n; i++) {
                fitted.set(i, data.get(i - 1) + fittedDiff.get(i - 1));
            }
        } else {
            // 高阶差分：递归逆差分 / Higher-order differencing: recursive inverse
            IVector<Double> current = fittedDiff;
            for (int level = 0; level < d; level++) {
                int curLen = current.length();
                IVector<Double> integrated = Linalg.zeros(curLen + 1);
                integrated.set(0, data.get(d - 1 - level));
                for (int i = 0; i < curLen; i++) {
                    integrated.set(i + 1, integrated.get(i) + current.get(i));
                }
                current = integrated;
            }
            for (int i = 0; i < Math.min(n, current.length()); i++) {
                fitted.set(i, current.get(i));
            }
        }

        return fitted;
    }
    
    /**
     * 计算对数似然 / Calculate log likelihood
     */
    private static double calculateLogLikelihood(IVector<Double> residuals, double sigma2) {
        int n = residuals.length();
        double sumSquares = residuals.multiply(residuals).sumValue();
        return -0.5 * n * Math.log(2 * Math.PI * sigma2) - sumSquares / (2 * sigma2);
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
    
    /**
     * 生成时间点 / Generate time points
     */
    private LocalDateTime[] generateTimePoints(int steps) {
        LocalDateTime[] timePoints = new LocalDateTime[steps];
        LocalDateTime now = LocalDateTime.now();
        for (int i = 0; i < steps; i++) {
            timePoints[i] = now.plusDays(i + 1);
        }
        return timePoints;
    }
    
    @Override
    public String toString() {
        return String.format("ARIMA(%d,%d,%d){AIC=%.2f, BIC=%.2f, LL=%.2f}", 
                           p, d, q, aic, bic, logLikelihood);
    }
}
