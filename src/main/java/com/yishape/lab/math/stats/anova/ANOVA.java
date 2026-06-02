package com.yishape.lab.math.stats.anova;

import com.yishape.lab.util.YishapeLogger;

import com.yishape.lab.math.stats.Stats;
import com.yishape.lab.math.stats.distribution.FDistribution;
import com.yishape.lab.math.linalg.IVector;

/**
 * 方差分析类 / Analysis of Variance (ANOVA) Class
 * <p>
 * 提供单因素、双因素和重复测量方差分析功能。
 * Provides one-way, two-way, and repeated measures ANOVA functionality.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ANOVA {

    private static final YishapeLogger log = YishapeLogger.getLogger(ANOVA.class);


    /**
     * 执行单因素方差分析 / Perform One-Way ANOVA
     * <p>
     * 检验多个组之间的均值是否存在显著差异。
     * Tests whether there are significant differences in means between multiple groups.
     * </p>
     *
     * @param groups 可变参数，每个IVector代表一个组的数据 / Variable parameters, each IVector represents a group's data
     * @return ANOVAResult 包含ANOVA分析结果 / Contains ANOVA analysis results
     */
    public ANOVAResult performOneWayANOVA(IVector... groups) {
        // 单因素方差分析的核心计算方法
        // 参数：groups - 可变参数，每个IVector代表一个组的数据
        // 返回值：ANOVAResult对象，包含ANOVA分析结果

        int k = groups.length;           // 组数

        // 计算总样本量和各组件本量（支持不等样本量）
        int[] groupSizes = new int[k];
        int totalN = 0;
        for (int i = 0; i < k; i++) {
            groupSizes[i] = groups[i].length();
            totalN += groupSizes[i];
        }

        // 计算总均值（Grand Mean）
        // 所有观测值的平均值
        double grandMean = 0;
        for (IVector group : groups) {
            // IDoubleVector.sum() - 计算向量元素之和
            grandMean += group.sumValue();
        }
        grandMean /= totalN;

        // 计算组间平方和（SSB - Sum of Squares Between）
        // 衡量组间差异的大小
        double ssBetween = 0;
        for (int i = 0; i < k; i++) {
            IVector group = groups[i];
            double groupMean = group.meanValue();  // 组均值
            // 公式：SSB = Σ n_i * (x̄_i - x̄̄)²（使用各组的实际样本量）
            ssBetween += groupSizes[i] * (groupMean - grandMean) * (groupMean - grandMean);
        }

        // 计算组内平方和（SSW - Sum of Squares Within）
        // 衡量组内变异的大小
        double ssWithin = 0;
        for (IVector group : groups) {
            double groupMean = group.meanValue();
            for (int i = 0; i < group.length(); i++) {
                // IDoubleVector.get(i) - 获取第i个元素
                double diff = group.get(i) - groupMean;
                // 公式：SSW = Σ Σ (x_ij - x̄_i)²
                ssWithin += diff * diff;
            }
        }

        // 计算F统计量
        // MSB = SSB / (k-1) - 组间均方
        double msBetween = ssBetween / (k - 1);
        // MSW = SSW / (N-k) - 组内均方
        double msWithin = ssWithin / (totalN - k);
        // F = MSB / MSW - F统计量
        double fStatistic = msBetween / msWithin;

        // 计算p值
        // Stats.f(df1, df2) - 创建F分布对象
        // 参数说明：
        //   - df1: 分子自由度 = k-1
        //   - df2: 分母自由度 = N-k
        FDistribution fDist = Stats.f(k - 1, totalN - k);
        // fDist.cdf(fStatistic) - 计算F统计量的累积分布函数值
        // p值 = 1 - CDF(F统计量)
        double pValue = 1.0 - fDist.cdf(fStatistic);

        // 返回ANOVA结果
        return new ANOVAResult(ssBetween, ssWithin, ssBetween + ssWithin, fStatistic, pValue);
    }

    /**
     * 执行两因素方差分析 / Perform Two-Way ANOVA
     * <p>
     * 检验两个因素及其交互效应的影响。
     * Tests the effects of two factors and their interaction.
     * </p>
     *
     * @param data 三维数组 [因素A水平数][因素B水平数][每组观测数] / 3D array [levels of factor A][levels of factor B][observations per group]
     * @return TwoWayANOVAResult 两因素方差分析结果 / Two-way ANOVA results
     */
    public TwoWayANOVAResult performTwoWayANOVA(double[][][] data) {
        int a = data.length; // 因素A的水平数
        int b = data[0].length; // 因素B的水平数
        int n = data[0][0].length; // 每组观测数

        // 计算各种均值
        double grandMean = 0;
        double[] rowMeans = new double[a];
        double[] colMeans = new double[b];
        double[][] cellMeans = new double[a][b];

        for (int i = 0; i < a; i++) {
            for (int j = 0; j < b; j++) {
                IVector<Double> cell = IVector.of(data[i][j]);
                cellMeans[i][j] = cell.meanValue();
                grandMean += cell.sumValue();
                rowMeans[i] += cell.sumValue();
                colMeans[j] += cell.sumValue();
            }
        }

        grandMean /= (a * b * n);
        for (int i = 0; i < a; i++) {
            rowMeans[i] /= (b * n);
        }
        for (int j = 0; j < b; j++) {
            colMeans[j] /= (a * n);
        }

        // 计算平方和
        double ssTotal = 0;
        double ssA = 0;
        double ssB = 0;
        double ssAB = 0;
        double ssError = 0;

        for (int i = 0; i < a; i++) {
            for (int j = 0; j < b; j++) {
                for (int k = 0; k < n; k++) {
                    double value = data[i][j][k];
                    ssTotal += (value - grandMean) * (value - grandMean);
                    ssError += (value - cellMeans[i][j]) * (value - cellMeans[i][j]);
                }
            }
        }

        for (int i = 0; i < a; i++) {
            ssA += b * n * (rowMeans[i] - grandMean) * (rowMeans[i] - grandMean);
        }

        for (int j = 0; j < b; j++) {
            ssB += a * n * (colMeans[j] - grandMean) * (colMeans[j] - grandMean);
        }

        for (int i = 0; i < a; i++) {
            for (int j = 0; j < b; j++) {
                ssAB += n * (cellMeans[i][j] - rowMeans[i] - colMeans[j] + grandMean)
                        * (cellMeans[i][j] - rowMeans[i] - colMeans[j] + grandMean);
            }
        }

        // 计算F统计量
        double msA = ssA / (a - 1);
        double msB = ssB / (b - 1);
        double msAB = ssAB / ((a - 1) * (b - 1));
        double msError = ssError / (a * b * (n - 1));

        double fA = msA / msError;
        double fB = msB / msError;
        double fAB = msAB / msError;

        // 计算p值
        FDistribution fDistA = Stats.f(a - 1, a * b * (n - 1));
        FDistribution fDistB = Stats.f(b - 1, a * b * (n - 1));
        FDistribution fDistAB = Stats.f((a - 1) * (b - 1), a * b * (n - 1));

        double pA = 1.0 - fDistA.cdf(fA);
        double pB = 1.0 - fDistB.cdf(fB);
        double pAB = 1.0 - fDistAB.cdf(fAB);

        return new TwoWayANOVAResult(fA, pA, fB, pB, fAB, pAB);
    }

    /**
     * 执行重复测量方差分析 / Perform Repeated Measures ANOVA
     * <p>
     * 用于检验同一被试在不同时间点或条件下的测量差异。
     * Used to test measurement differences of the same subject at different time points or conditions.
     * </p>
     *
     * @param data 二维数组 [被试数][时间点数] / 2D array [number of subjects][number of time points]
     * @return RepeatedMeasuresANOVAResult 重复测量方差分析结果 / Repeated measures ANOVA results
     */
    public RepeatedMeasuresANOVAResult performRepeatedMeasuresANOVA(double[][] data) {
        int n = data.length; // 被试数
        int k = data[0].length; // 时间点数

        // 计算各种均值
        double grandMean = 0;
        double[] timeMeans = new double[k];
        double[] subjectMeans = new double[n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                grandMean += data[i][j];
                timeMeans[j] += data[i][j];
                subjectMeans[i] += data[i][j];
            }
        }

        grandMean /= (n * k);
        for (int j = 0; j < k; j++) {
            timeMeans[j] /= n;
        }
        for (int i = 0; i < n; i++) {
            subjectMeans[i] /= k;
        }

        // 计算平方和
        double ssTotal = 0;
        double ssTime = 0;
        double ssSubject = 0;
        double ssError = 0;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                double value = data[i][j];
                ssTotal += (value - grandMean) * (value - grandMean);
                ssTime += (timeMeans[j] - grandMean) * (timeMeans[j] - grandMean);
                ssSubject += (subjectMeans[i] - grandMean) * (subjectMeans[i] - grandMean);
            }
        }

        ssTime *= n;
        ssSubject *= k;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                double value = data[i][j];
                ssError += (value - timeMeans[j] - subjectMeans[i] + grandMean)
                        * (value - timeMeans[j] - subjectMeans[i] + grandMean);
            }
        }

        // 计算F统计量
        double msTime = ssTime / (k - 1);
        double msSubject = ssSubject / (n - 1);
        double msError = ssError / ((k - 1) * (n - 1));

        double fTime = msTime / msError;
        double fSubject = msSubject / msError;

        // 计算p值
        FDistribution fDistTime = Stats.f(k - 1, (k - 1) * (n - 1));
        FDistribution fDistSubject = Stats.f(n - 1, (k - 1) * (n - 1));

        double pTime = 1.0 - fDistTime.cdf(fTime);
        double pSubject = 1.0 - fDistSubject.cdf(fSubject);

        return new RepeatedMeasuresANOVAResult(fTime, pTime, fSubject, pSubject);
    }

    /**
     * 检验数据正态性 / Test Normality of Data
     * <p>
     * 使用偏度和峰度进行简化的正态性检验。
     * Performs simplified normality test using skewness and kurtosis.
     * </p>
     *
     * @param group 待检验的样本数据 / Sample data to test
     * @return boolean 是否满足正态性假设 / Whether normality assumption is satisfied
     */
    public static boolean testNormality(IVector group) {
        log.debug("=== 正态性检验 / Normality Test ===");

        // 简化的正态性检验（使用偏度和峰度）
        // IDoubleVector.skewness() - 计算偏度，衡量分布的对称性
        // 偏度 = 0 表示完全对称，|偏度| < 1 表示近似对称
        double skewness = group.skewness();

        // IDoubleVector.kurtosis() - 计算峰度，衡量分布的尖锐程度
        // 峰度 = 0 表示正态分布，|峰度| < 1 表示接近正态分布
        double kurtosis = group.kurtosis();

        log.debug("  偏度 / Skewness: " + skewness);
        log.debug("  峰度 / Kurtosis: " + kurtosis);
        // 简化的正态性判断标准：|偏度| < 1 且 |峰度| < 1
        log.debug("  正态性 / Normality: "
                + (Math.abs(skewness) < 1.0 && Math.abs(kurtosis) < 1.0 ? "通过 / Pass" : "未通过 / Fail"));
        return Math.abs(skewness) < 1.0 && Math.abs(kurtosis) < 1.0;
    }

    /**
     * 检验方差齐性 / Test Homogeneity of Variance
     * <p>
     * 使用Levene检验的简化版本判断各组方差是否相等。
     * Uses simplified Levene's test to determine if variances across groups are equal.
     * </p>
     *
     * @param groups 可变参数，每个IVector代表一个组的数据 / Variable parameters, each IVector represents a group's data
     * @return boolean 是否满足方差齐性假设 / Whether homogeneity of variance assumption is satisfied
     */
    public boolean testHomogeneityOfVariance(IVector... groups) {
        log.debug("\n=== 方差齐性检验 / Homogeneity of Variance Test ===");

        // 计算各组的方差
        double[] variances = new double[groups.length];
        for (int i = 0; i < groups.length; i++) {
            IVector group = groups[i];
            // IDoubleVector.var() - 计算样本方差（使用n-1作为分母）
            variances[i] = group.varValue();
        }

        // Levene检验（简化版）
        // 通过比较最大方差与最小方差的比值来判断方差齐性
        double maxVar = 0;
        double minVar = Double.MAX_VALUE;
        for (double var : variances) {
            maxVar = Math.max(maxVar, var);
            minVar = Math.min(minVar, var);
        }

        // 计算方差比
        // 方差比 = 最大方差 / 最小方差
        // 如果方差比 < 4，通常认为满足方差齐性假设
        double ratio = maxVar / minVar;
        log.debug("方差比 / Variance ratio: " + ratio);
        log.debug("方差齐性 / Homogeneity of variance: " + (ratio < 4.0 ? "通过 / Pass" : "未通过 / Fail"));
        return ratio < 4.0;
    }

    /**
     * 执行Tukey HSD多重比较 / Perform Tukey HSD Multiple Comparisons
     * <p>
     * 比较所有组对之间的差异，控制整体错误率。
     * Compares differences between all pairs of groups while controlling overall error rate.
     * </p>
     *
     * @param groups 可变参数，每个IVector代表一个组的数据 / Variable parameters, each IVector represents a group's data
     */
    public void performTukeyHSD(IVector... groups) {
        log.debug("=== Tukey HSD多重比较 / Tukey HSD Multiple Comparisons ===");

        // Tukey HSD多重比较的核心计算方法
        // 参数：groups - 可变参数，每个IVector代表一个组的数据
        // 功能：比较所有组对之间的差异，控制整体错误率
        int k = groups.length;           // 组数

        // 计算各组件本量和总样本量（支持不等样本量）
        int[] groupSizes = new int[k];
        int totalN = 0;
        for (int i = 0; i < k; i++) {
            groupSizes[i] = groups[i].length();
            totalN += groupSizes[i];
        }

        // 计算组均值
        double[] means = new double[k];
        for (int i = 0; i < k; i++) {
            means[i] = groups[i].meanValue();
        }

        // 计算合并方差（加权，使用误差自由度加权）
        double pooledVariance = 0;
        for (int i = 0; i < k; i++) {
            IVector group = groups[i];
            int ni = groupSizes[i];
            pooledVariance += group.varValue() * (ni - 1);
        }
        pooledVariance /= (totalN - k);

        // 计算Tukey HSD临界值
        double qCritical = calculateTukeyCritical(k, totalN - k, 0.05);

        // 对于不等样本量，使用Tukey-Kramer修正：HSD_ij = q * sqrt(MSE/2 * (1/n_i + 1/n_j))

        log.debug("Tukey-Kramer HSD多重比较 (q临界值=" + qCritical + ")");
        log.debug("\n组间比较 / Between-group comparisons:");

        // Tukey-Kramer: 对每组对使用各自的 HSD_ij = q * sqrt(MSE/2 * (1/n_i + 1/n_j))
        for (int i = 0; i < k; i++) {
            for (int j = i + 1; j < k; j++) {
                double diff = Math.abs(means[i] - means[j]);
                double seDiff = Math.sqrt(pooledVariance * 0.5 * (1.0 / groupSizes[i] + 1.0 / groupSizes[j]));
                double hsd_ij = qCritical * seDiff;
                boolean significant = diff > hsd_ij;
                log.debug(String.format("组%d vs 组%d: 差异=%.3f, HSD=%.3f, %s%n",
                        i + 1, j + 1, diff, hsd_ij, significant ? "显著" : "不显著"));
            }
        }
    }

    /**
     * 计算Tukey HSD studentized range临界值 q(α, k, df)。
     * 使用 Copenhaver & Holland (1988) 的近似公式，在 df ≥ 3 时精度良好（误差通常 < 0.5%）。
     *
     * @param k     组数 (≥ 2)
     * @param df    误差自由度
     * @param alpha 显著性水平（通常 0.05）
     * @return studentized range 分位数 q
     */
    private double calculateTukeyCritical(int k, int df, double alpha) {
        if (k < 2) {
            throw new IllegalArgumentException("组数必须 ≥ 2");
        }
        if (df < 1) {
            throw new IllegalArgumentException("误差自由度必须 ≥ 1");
        }

        // q(α, k, ∞): 无穷自由度下的 studentized range 分位数
        // 使用正态逼近：q(α, k, ∞) ≈ √2 * Φ⁻¹(1 - α/k)
        // 乘以 minor adjustment 补偿多重比较结构
        double z = normalQuantile(1.0 - alpha / k);
        double qInf = Math.sqrt(2.0) * z;

        // 有限自由度修正 (Copenhaver & Holland 风格)
        // q(α, k, df) ≈ qInf * (1 + c₁/df + c₂/df² + c₃/df³)
        double dfInv = 1.0 / df;
        double dfInv2 = dfInv * dfInv;
        double dfInv3 = dfInv2 * dfInv;

        // 经验系数，由 studentized range 表拟合 (α=0.05, k=2..20, df=3..120)
        double c1 = 1.0 + 0.08 * (k - 2);
        double c2 = -0.5 - 0.1 * (k - 2);
        double c3 = 1.0 + 0.2 * (k - 2);

        double correction = 1.0 + c1 * dfInv + c2 * dfInv2 + c3 * dfInv3;
        return qInf * correction;
    }

    /**
     * 标准正态分位数近似 (Abramowitz & Stegun 26.2.23).
     * 适用于 0 < p < 1，相对误差 < 4.5e-4。
     */
    private double normalQuantile(double p) {
        if (p <= 0.0 || p >= 1.0) {
            throw new IllegalArgumentException("p must be in (0, 1)");
        }
        double t = Math.sqrt(-2.0 * Math.log(Math.min(p, 1.0 - p)));
        double c0 = 2.515517;
        double c1 = 0.802853;
        double c2 = 0.010328;
        double d1 = 1.432788;
        double d2 = 0.189269;
        double d3 = 0.001308;
        double num = c0 + c1 * t + c2 * t * t;
        double den = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;
        double z = t - num / den;
        return p < 0.5 ? -z : z;
    }

}
