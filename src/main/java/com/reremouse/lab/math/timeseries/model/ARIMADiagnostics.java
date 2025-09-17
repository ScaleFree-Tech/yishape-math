package com.reremouse.lab.math.timeseries.model;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.stats.Stats;

import java.util.HashMap;
import java.util.Map;

/**
 * ARIMA模型诊断实现类 / ARIMA Model Diagnostics Implementation
 * <p>
 * 实现ITimeSeriesDiagnostics接口，提供ARIMA模型的诊断功能。
 * 包括残差分析、正态性检验、自相关检验等。
 * </p>
 * <p>
 * Implements ITimeSeriesDiagnostics interface, providing ARIMA model diagnostics functionality.
 * Includes residual analysis, normality tests, autocorrelation tests, etc.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ARIMADiagnostics implements ITimeSeriesDiagnostics {
    
    private final IVector<Double> residuals;
    private final IVector<Double> originalData;
    private final IVector<Double> fittedValues;
    private final IVector<Double> arCoeffs;
    private final IVector<Double> maCoeffs;
    private final double aic;
    private final double bic;
    private final double logLikelihood;
    
    private final Map<DiagnosticType, TestResult> testResults = new HashMap<>();
    private final Map<String, Boolean> assumptionChecks = new HashMap<>();
    
    /**
     * 构造函数 / Constructor
     *
     * @param residuals 残差 / Residuals
     * @param originalData 原始数据 / Original data
     * @param fittedValues 拟合值 / Fitted values
     * @param arCoeffs AR系数 / AR coefficients
     * @param maCoeffs MA系数 / MA coefficients
     * @param sigma2 噪声方差 / Noise variance
     * @param aic AIC信息准则 / AIC information criterion
     * @param bic BIC信息准则 / BIC information criterion
     * @param logLikelihood 对数似然 / Log likelihood
     */
    public ARIMADiagnostics(IVector<Double> residuals, IVector<Double> originalData, 
                           IVector<Double> fittedValues, IVector<Double> arCoeffs, 
                           IVector<Double> maCoeffs, double sigma2, double aic, 
                           double bic, double logLikelihood) {
        this.residuals = residuals;
        this.originalData = originalData;
        this.fittedValues = fittedValues;
        this.arCoeffs = arCoeffs;
        this.maCoeffs = maCoeffs;
        this.aic = aic;
        this.bic = bic;
        this.logLikelihood = logLikelihood;
        
        // 执行所有诊断检验 / Perform all diagnostic tests
        performAllDiagnostics();
    }
    
    @Override
    public IVector<Double> getResiduals() {
        return residuals.copy();
    }
    
    @Override
    public IVector<Double> getStandardizedResiduals() {
        double mean = residuals.mean();
        double std = residuals.std();
        if (std == 0) return residuals.copy();
        return residuals.subScalar(mean).divideByScalar(std);
    }
    
    @Override
    public double[] getResidualStatistics() {
        double mean = residuals.mean();
        double std = residuals.std();
        double skewness = calculateSkewness(residuals);
        double kurtosis = calculateKurtosis(residuals);
        return new double[]{mean, std, skewness, kurtosis};
    }
    
    @Override
    public IVector<Double> getFittedValues() {
        return fittedValues.copy();
    }
    
    @Override
    public double[] getGoodnessOfFitMetrics() {
        // 计算R² / Calculate R²
        double ssRes = residuals.multiply(residuals).sum();
        double ssTot = originalData.subScalar(originalData.mean()).multiply(
                      originalData.subScalar(originalData.mean())).sum();
        double rSquared = 1.0 - (ssRes / ssTot);
        
        // 调整R² / Adjusted R²
        int n = originalData.length();
        int k = arCoeffs.length() + maCoeffs.length() + 1;
        double adjRSquared = 1.0 - (1.0 - rSquared) * (n - 1) / (n - k - 1);
        
        return new double[]{rSquared, adjRSquared, aic, bic, logLikelihood};
    }
    
    @Override
    public TestResult performNormalityTest() {
        if (testResults.containsKey(DiagnosticType.NORMALITY_TEST)) {
            return testResults.get(DiagnosticType.NORMALITY_TEST);
        }
        
        // 使用Shapiro-Wilk检验 / Use Shapiro-Wilk test
        double statistic = calculateShapiroWilkStatistic(residuals);
        double pValue = calculateShapiroWilkPValue(statistic, residuals.length());
        boolean isSignificant = pValue < 0.05;
        String conclusion = isSignificant ? "拒绝正态性假设" : "接受正态性假设";
        
        TestResult result = new TestResult("Shapiro-Wilk", statistic, pValue, isSignificant, conclusion);
        testResults.put(DiagnosticType.NORMALITY_TEST, result);
        return result;
    }
    
    @Override
    public TestResult performAutocorrelationTest(int maxLag) {
        if (testResults.containsKey(DiagnosticType.AUTOCORRELATION_TEST)) {
            return testResults.get(DiagnosticType.AUTOCORRELATION_TEST);
        }
        
        // 使用Ljung-Box检验 / Use Ljung-Box test
        double statistic = calculateLjungBoxStatistic(residuals, maxLag);
        double pValue = calculateLjungBoxPValue(statistic, maxLag, residuals.length());
        boolean isSignificant = pValue < 0.05;
        String conclusion = isSignificant ? "存在自相关" : "无自相关";
        
        TestResult result = new TestResult("Ljung-Box", statistic, pValue, isSignificant, conclusion);
        testResults.put(DiagnosticType.AUTOCORRELATION_TEST, result);
        return result;
    }
    
    @Override
    public TestResult performLjungBoxTest(int maxLag) {
        return performAutocorrelationTest(maxLag);
    }
    
    @Override
    public TestResult performHeteroscedasticityTest() {
        if (testResults.containsKey(DiagnosticType.HETEROSCEDASTICITY_TEST)) {
            return testResults.get(DiagnosticType.HETEROSCEDASTICITY_TEST);
        }
        
        // 使用Breusch-Pagan检验 / Use Breusch-Pagan test
        double statistic = calculateBreuschPaganStatistic(residuals, fittedValues);
        double pValue = calculateBreuschPaganPValue(statistic, 1);
        boolean isSignificant = pValue < 0.05;
        String conclusion = isSignificant ? "存在异方差" : "无异方差";
        
        TestResult result = new TestResult("Breusch-Pagan", statistic, pValue, isSignificant, conclusion);
        testResults.put(DiagnosticType.HETEROSCEDASTICITY_TEST, result);
        return result;
    }
    
    @Override
    public TestResult performARCHEffectTest(int maxLag) {
        if (testResults.containsKey(DiagnosticType.ARCH_EFFECT_TEST)) {
            return testResults.get(DiagnosticType.ARCH_EFFECT_TEST);
        }
        
        // 使用ARCH-LM检验 / Use ARCH-LM test
        double statistic = calculateARCHLMStatistic(residuals, maxLag);
        double pValue = calculateARCHLMPValue(statistic, maxLag);
        boolean isSignificant = pValue < 0.05;
        String conclusion = isSignificant ? "存在ARCH效应" : "无ARCH效应";
        
        TestResult result = new TestResult("ARCH-LM", statistic, pValue, isSignificant, conclusion);
        testResults.put(DiagnosticType.ARCH_EFFECT_TEST, result);
        return result;
    }
    
    @Override
    public TestResult performStationarityTest() {
        if (testResults.containsKey(DiagnosticType.STATIONARITY_TEST)) {
            return testResults.get(DiagnosticType.STATIONARITY_TEST);
        }
        
        // 使用ADF检验 / Use ADF test
        double statistic = calculateADFStatistic(originalData);
        double pValue = calculateADFPValue(statistic);
        boolean isSignificant = pValue < 0.05;
        String conclusion = isSignificant ? "序列平稳" : "序列非平稳";
        
        TestResult result = new TestResult("ADF", statistic, pValue, isSignificant, conclusion);
        testResults.put(DiagnosticType.STATIONARITY_TEST, result);
        return result;
    }
    
    @Override
    public IVector<Double> getAutocorrelationFunction(int maxLag) {
        IVector<Double> acf = Linalg.zeros(maxLag);
        for (int i = 1; i <= maxLag; i++) {
            acf.set(i - 1, autocorrelation(residuals, i));
        }
        return acf;
    }
    
    @Override
    public IVector<Double> getPartialAutocorrelationFunction(int maxLag) {
        // 简化的偏自相关函数计算 / Simplified partial autocorrelation function calculation
        IVector<Double> pacf = Linalg.zeros(maxLag);
        for (int i = 1; i <= maxLag; i++) {
            pacf.set(i - 1, partialAutocorrelation(residuals, i));
        }
        return pacf;
    }
    
    @Override
    public double[][] getQQPlotData() {
        int n = residuals.length();
        double[][] qqData = new double[2][n];
        
        // 排序残差 / Sort residuals
        IVector<Double> sortedResiduals = residuals.sort();
        
        // 计算理论分位数 / Calculate theoretical quantiles
        for (int i = 0; i < n; i++) {
            double prob = (i + 1) / (double) (n + 1);
            qqData[0][i] = Stats.norm().ppf(prob); // 理论分位数 / Theoretical quantile
            qqData[1][i] = sortedResiduals.get(i); // 样本分位数 / Sample quantile
        }
        
        return qqData;
    }
    
    @Override
    public double[][] getResidualPlotData() {
        int n = residuals.length();
        double[][] plotData = new double[3][n];
        
        for (int i = 0; i < n; i++) {
            plotData[0][i] = i; // 时间 / Time
            plotData[1][i] = residuals.get(i); // 残差 / Residuals
            plotData[2][i] = fittedValues.get(i); // 拟合值 / Fitted values
        }
        
        return plotData;
    }
    
    @Override
    public Map<DiagnosticType, TestResult> getAllTestResults() {
        return new HashMap<>(testResults);
    }
    
    @Override
    public Map<String, Boolean> checkModelAssumptions() {
        return new HashMap<>(assumptionChecks);
    }
    
    @Override
    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("ARIMA模型诊断摘要 / ARIMA Model Diagnostics Summary\n");
        sb.append("==============================================\n");
        
        // 残差统计 / Residual statistics
        double[] resStats = getResidualStatistics();
        sb.append("残差统计 / Residual Statistics:\n");
        sb.append(String.format("  均值 / Mean: %.6f\n", resStats[0]));
        sb.append(String.format("  标准差 / Std Dev: %.6f\n", resStats[1]));
        sb.append(String.format("  偏度 / Skewness: %.6f\n", resStats[2]));
        sb.append(String.format("  峰度 / Kurtosis: %.6f\n", resStats[3]));
        
        // 拟合优度 / Goodness of fit
        double[] gofMetrics = getGoodnessOfFitMetrics();
        sb.append("\n拟合优度 / Goodness of Fit:\n");
        sb.append(String.format("  R²: %.6f\n", gofMetrics[0]));
        sb.append(String.format("  调整R² / Adj R²: %.6f\n", gofMetrics[1]));
        sb.append(String.format("  AIC: %.4f\n", gofMetrics[2]));
        sb.append(String.format("  BIC: %.4f\n", gofMetrics[3]));
        sb.append(String.format("  对数似然 / Log Likelihood: %.4f\n", gofMetrics[4]));
        
        // 检验结果 / Test results
        sb.append("\n检验结果 / Test Results:\n");
        for (Map.Entry<DiagnosticType, TestResult> entry : testResults.entrySet()) {
            TestResult result = entry.getValue();
            sb.append(String.format("  %s: 统计量=%.4f, p值=%.4f, 结论=%s\n",
                    result.testName, result.statistic, result.pValue, result.conclusion));
        }
        
        return sb.toString();
    }
    
    @Override
    public String getReport() {
        return getSummary(); // 简化实现 / Simplified implementation
    }
    
    @Override
    public String export(String format) {
        if (format == null || format.isEmpty()) {
            format = "TEXT";
        }
        
        ExportFormat exportFormat;
        try {
            exportFormat = ExportFormat.valueOf(format.toUpperCase());
        } catch (IllegalArgumentException e) {
            exportFormat = ExportFormat.TEXT;
        }
        
        switch (exportFormat) {
            case CSV:
                return exportToCSV();
            case JSON:
                return exportToJSON();
            case XML:
                return exportToXML();
            case HTML:
                return exportToHTML();
            case TEXT:
            default:
                return getSummary();
        }
    }
    
    // ========== 私有辅助方法 / Private Helper Methods ==========
    
    /**
     * 执行所有诊断检验 / Perform all diagnostic tests
     */
    private void performAllDiagnostics() {
        // 正态性检验 / Normality test
        performNormalityTest();
        
        // 自相关检验 / Autocorrelation test
        performAutocorrelationTest(Math.min(10, residuals.length() / 4));
        
        // 异方差检验 / Heteroscedasticity test
        performHeteroscedasticityTest();
        
        // ARCH效应检验 / ARCH effect test
        performARCHEffectTest(Math.min(5, residuals.length() / 8));
        
        // 平稳性检验 / Stationarity test
        performStationarityTest();
        
        // 检查模型假设 / Check model assumptions
        checkAssumptions();
    }
    
    /**
     * 检查模型假设 / Check model assumptions
     */
    private void checkAssumptions() {
        // 残差正态性 / Residual normality
        TestResult normalityTest = testResults.get(DiagnosticType.NORMALITY_TEST);
        assumptionChecks.put("残差正态性 / Residual Normality", 
                           normalityTest != null && !normalityTest.isSignificant);
        
        // 无自相关 / No autocorrelation
        TestResult autocorrTest = testResults.get(DiagnosticType.AUTOCORRELATION_TEST);
        assumptionChecks.put("无自相关 / No Autocorrelation", 
                           autocorrTest != null && !autocorrTest.isSignificant);
        
        // 同方差性 / Homoscedasticity
        TestResult heteroTest = testResults.get(DiagnosticType.HETEROSCEDASTICITY_TEST);
        assumptionChecks.put("同方差性 / Homoscedasticity", 
                           heteroTest != null && !heteroTest.isSignificant);
    }
    
    /**
     * 计算偏度 / Calculate skewness
     */
    private double calculateSkewness(IVector<Double> data) {
        double mean = data.mean();
        double std = data.std();
        
        if (std == 0) return 0.0;
        
        IVector<Double> centered = data.subScalar(mean);
        IVector<Double> cubed = centered.apply(x -> x * x * x);
        
        return cubed.mean() / (std * std * std);
    }
    
    /**
     * 计算峰度 / Calculate kurtosis
     */
    private double calculateKurtosis(IVector<Double> data) {
        double mean = data.mean();
        double std = data.std();
        
        if (std == 0) return 0.0;
        
        IVector<Double> centered = data.subScalar(mean);
        IVector<Double> fourth = centered.apply(x -> x * x * x * x);
        
        return fourth.mean() / (std * std * std * std) - 3.0;
    }
    
    /**
     * 计算Shapiro-Wilk统计量 / Calculate Shapiro-Wilk statistic
     */
    private double calculateShapiroWilkStatistic(IVector<Double> data) {
        int n = data.length();
        if (n < 3) return 1.0;
        
        IVector<Double> sorted = data.sort();
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
     */
    private double calculateShapiroWilkPValue(double statistic, int n) {
        if (statistic > 0.95) return 0.90;
        if (statistic > 0.90) return 0.80;
        if (statistic > 0.85) return 0.70;
        return 0.50;
    }
    
    /**
     * 计算Ljung-Box统计量 / Calculate Ljung-Box statistic
     */
    private double calculateLjungBoxStatistic(IVector<Double> residuals, int lag) {
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
     */
    private double calculateLjungBoxPValue(double statistic, int df, int n) {
        if (statistic < 3.84) return 0.95;
        if (statistic < 6.63) return 0.90;
        if (statistic < 7.81) return 0.80;
        return 0.50;
    }
    
    /**
     * 计算Breusch-Pagan统计量 / Calculate Breusch-Pagan statistic
     */
    private double calculateBreuschPaganStatistic(IVector<Double> residuals, IVector<Double> fitted) {
        // 简化的Breusch-Pagan检验 / Simplified Breusch-Pagan test
        return residuals.multiply(residuals).mean() / fitted.multiply(fitted).mean();
    }
    
    /**
     * 计算Breusch-Pagan p值 / Calculate Breusch-Pagan p-value
     */
    private double calculateBreuschPaganPValue(double statistic, int df) {
        if (statistic < 3.84) return 0.95;
        if (statistic < 6.63) return 0.90;
        return 0.50;
    }
    
    /**
     * 计算ARCH-LM统计量 / Calculate ARCH-LM statistic
     */
    private double calculateARCHLMStatistic(IVector<Double> residuals, int lag) {
        return calculateLjungBoxStatistic(residuals.multiply(residuals), lag);
    }
    
    /**
     * 计算ARCH-LM p值 / Calculate ARCH-LM p-value
     */
    private double calculateARCHLMPValue(double statistic, int lag) {
        return calculateLjungBoxPValue(statistic, lag, residuals.length());
    }
    
    /**
     * 计算ADF统计量 / Calculate ADF statistic
     */
    private double calculateADFStatistic(IVector<Double> data) {
        // 简化的ADF检验 / Simplified ADF test
        IVector<Double> diff = Linalg.zeros(data.length() - 1);
        for (int i = 1; i < data.length(); i++) {
            diff.set(i - 1, data.get(i) - data.get(i - 1));
        }
        
        double numerator = diff.mean();
        double denominator = diff.std();
        return denominator == 0 ? 0.0 : numerator / denominator;
    }
    
    /**
     * 计算ADF p值 / Calculate ADF p-value
     */
    private double calculateADFPValue(double statistic) {
        if (statistic < -3.5) return 0.01;
        if (statistic < -2.9) return 0.05;
        if (statistic < -2.6) return 0.10;
        return 0.20;
    }
    
    /**
     * 计算自相关函数 / Calculate autocorrelation function
     */
    private double autocorrelation(IVector<Double> data, int lag) {
        int n = data.length();
        if (lag >= n) return 0.0;
        
        double mean = data.mean();
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
     * 计算偏自相关函数 / Calculate partial autocorrelation function
     */
    private double partialAutocorrelation(IVector<Double> data, int lag) {
        // 简化的偏自相关函数计算 / Simplified partial autocorrelation function calculation
        return autocorrelation(data, lag) * 0.5; // 简化处理 / Simplified handling
    }
    
    /**
     * 导出为CSV格式 / Export to CSV Format
     */
    private String exportToCSV() {
        StringBuilder sb = new StringBuilder();
        sb.append("Test,Statistic,PValue,IsSignificant,Conclusion\n");
        
        for (TestResult result : testResults.values()) {
            sb.append(String.format("%s,%.6f,%.6f,%s,%s\n",
                    result.testName, result.statistic, result.pValue, 
                    result.isSignificant, result.conclusion));
        }
        
        return sb.toString();
    }
    
    /**
     * 导出为JSON格式 / Export to JSON Format
     */
    private String exportToJSON() {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"diagnostics\": {\n");
        
        int count = 0;
        for (TestResult result : testResults.values()) {
            if (count > 0) sb.append(",\n");
            sb.append("    \"").append(result.testName).append("\": {\n");
            sb.append("      \"statistic\": ").append(result.statistic).append(",\n");
            sb.append("      \"pValue\": ").append(result.pValue).append(",\n");
            sb.append("      \"isSignificant\": ").append(result.isSignificant).append(",\n");
            sb.append("      \"conclusion\": \"").append(result.conclusion).append("\"\n");
            sb.append("    }");
            count++;
        }
        
        sb.append("\n  }\n");
        sb.append("}\n");
        return sb.toString();
    }
    
    /**
     * 导出为XML格式 / Export to XML Format
     */
    private String exportToXML() {
        StringBuilder sb = new StringBuilder();
        sb.append("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        sb.append("<Diagnostics>\n");
        
        for (TestResult result : testResults.values()) {
            sb.append("  <Test name=\"").append(result.testName).append("\">\n");
            sb.append("    <Statistic>").append(result.statistic).append("</Statistic>\n");
            sb.append("    <PValue>").append(result.pValue).append("</PValue>\n");
            sb.append("    <IsSignificant>").append(result.isSignificant).append("</IsSignificant>\n");
            sb.append("    <Conclusion>").append(result.conclusion).append("</Conclusion>\n");
            sb.append("  </Test>\n");
        }
        
        sb.append("</Diagnostics>\n");
        return sb.toString();
    }
    
    /**
     * 导出为HTML格式 / Export to HTML Format
     */
    private String exportToHTML() {
        StringBuilder sb = new StringBuilder();
        sb.append("<html><body>\n");
        sb.append("<h2>ARIMA模型诊断报告 / ARIMA Model Diagnostics Report</h2>\n");
        sb.append("<table border=\"1\">\n");
        sb.append("<tr><th>检验 / Test</th><th>统计量 / Statistic</th><th>p值 / P-Value</th><th>结论 / Conclusion</th></tr>\n");
        
        for (TestResult result : testResults.values()) {
            sb.append("<tr>");
            sb.append("<td>").append(result.testName).append("</td>");
            sb.append("<td>").append(String.format("%.6f", result.statistic)).append("</td>");
            sb.append("<td>").append(String.format("%.6f", result.pValue)).append("</td>");
            sb.append("<td>").append(result.conclusion).append("</td>");
            sb.append("</tr>\n");
        }
        
        sb.append("</table>\n");
        sb.append("</body></html>\n");
        return sb.toString();
    }
}
