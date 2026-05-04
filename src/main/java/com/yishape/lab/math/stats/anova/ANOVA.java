package com.yishape.lab.math.stats.anova;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

    private static final Logger log = LoggerFactory.getLogger(ANOVA.class);


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
        int n = groups[0].length();      // 每组样本量（假设各组样本量相等）
        int totalN = k * n;              // 总样本量

        // 计算总均值（Grand Mean）
        // 所有观测值的平均值
        double grandMean = 0;
        for (IVector group : groups) {
            // IDoubleVector.sum() - 计算向量元素之和
            grandMean += group.sum().doubleValue();
        }
        grandMean /= totalN;

        // 计算组间平方和（SSB - Sum of Squares Between）
        // 衡量组间差异的大小
        double ssBetween = 0;
        for (IVector group : groups) {
            double groupMean = group.mean().doubleValue();  // 组均值
            // 公式：SSB = Σ n_i * (x̄_i - x̄̄)²
            ssBetween += n * (groupMean - grandMean) * (groupMean - grandMean);
        }

        // 计算组内平方和（SSW - Sum of Squares Within）
        // 衡量组内变异的大小
        double ssWithin = 0;
        for (IVector group : groups) {
            double groupMean = group.mean().doubleValue();
            for (int i = 0; i < group.length(); i++) {
                // IDoubleVector.get(i) - 获取第i个元素
                double diff = group.get(i).doubleValue() - groupMean;
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
        double pValue = 1.0f - fDist.cdf(fStatistic);

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
                cellMeans[i][j] = cell.mean();
                grandMean += cell.sum();
                rowMeans[i] += cell.sum();
                colMeans[j] += cell.sum();
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

        double pA = 1.0f - fDistA.cdf(fA);
        double pB = 1.0f - fDistB.cdf(fB);
        double pAB = 1.0f - fDistAB.cdf(fAB);

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

        double pTime = 1.0f - fDistTime.cdf(fTime);
        double pSubject = 1.0f - fDistSubject.cdf(fSubject);

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
        double skewness = group.skewness().doubleValue();

        // IDoubleVector.kurtosis() - 计算峰度，衡量分布的尖锐程度
        // 峰度 = 0 表示正态分布，|峰度| < 1 表示接近正态分布
        double kurtosis = group.kurtosis().doubleValue();

        log.debug("  偏度 / Skewness: " + skewness);
        log.debug("  峰度 / Kurtosis: " + kurtosis);
        // 简化的正态性判断标准：|偏度| < 1 且 |峰度| < 1
        log.debug("  正态性 / Normality: "
                + (Math.abs(skewness) < 1.0f && Math.abs(kurtosis) < 1.0f ? "通过 / Pass" : "未通过 / Fail"));
        return Math.abs(skewness) < 1.0f && Math.abs(kurtosis) < 1.0f;
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
            variances[i] = group.var().doubleValue();
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
        log.debug("方差齐性 / Homogeneity of variance: " + (ratio < 4.0f ? "通过 / Pass" : "未通过 / Fail"));
        return ratio < 4.0f;
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
        // 参数：groups - 可变参数，每个double[]代表一个组的数据
        // 功能：比较所有组对之间的差异，控制整体错误率
        int k = groups.length;           // 组数
        int n = groups[0].length();        // 每组样本量（假设各组样本量相等）

        // 计算组均值
        // 计算每个组的样本均值
        double[] means = new double[k];
        for (int i = 0; i < k; i++) {
            IVector group = groups[i];
            // IDoubleVector.mean() - 计算组均值
            means[i] = group.mean().doubleValue();
        }

        // 计算合并方差
        // 使用所有组的方差计算合并方差，作为误差方差的估计
        double pooledVariance = 0;
        for (int i = 0; i < k; i++) {
            IVector group = groups[i];
            // IDoubleVector.var() - 计算组方差
            pooledVariance += group.var().doubleValue();
        }
        pooledVariance /= k;  // 合并方差 = 各组方差的平均值

        // 计算Tukey HSD临界值
        // calculateTukeyCritical(k, df, alpha) - 计算Tukey临界值
        // 参数说明：
        //   - k: 组数
        //   - df: 误差自由度 = N - k
        //   - alpha: 显著性水平（如0.05）
        double qCritical = calculateTukeyCritical(k, n * k - k, 0.05f);

        // 计算HSD临界值
        // HSD = q * sqrt(MSE / n)
        // 其中：q是Tukey临界值，MSE是合并方差，n是每组样本量
        double hsd = qCritical * (double) Math.sqrt(pooledVariance / (double) n);

        log.debug("Tukey HSD临界值 / Tukey HSD critical value: " + hsd);
        log.debug("\n组间比较 / Between-group comparisons:");

        // 进行所有组对之间的比较
        for (int i = 0; i < k; i++) {
            for (int j = i + 1; j < k; j++) {
                // 计算组间均值差异
                double diff = Math.abs(means[i] - means[j]);
                // 判断差异是否显著
                // 如果|均值差异| > HSD，则认为差异显著
                boolean significant = diff > hsd;
                log.debug(String.format("组%d vs 组%d: 差异=%.3f, %s%n",
                        i + 1, j + 1, diff, significant ? "显著" : "不显著"));
            }
        }
    }

    private double calculateTukeyCritical(int k, int df, double alpha) {
        // 简化的Tukey临界值计算
        // 参数说明：
        //   - k: 组数
        //   - df: 误差自由度
        //   - alpha: 显著性水平
        // 返回值：Tukey临界值q
        // 注意：这是简化版本，实际应用中应使用查表或精确计算
        // 真实的Tukey临界值需要通过查表或数值计算获得
        return 2.5f + (k - 2) * 0.1f;
    }

}
