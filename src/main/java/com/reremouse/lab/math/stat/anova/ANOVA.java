package com.reremouse.lab.math.stat.anova;

import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.stat.Stat;
import com.reremouse.lab.math.stat.distribution.FDistribution;

/**
 *
 * @author lteb2
 */
public class ANOVA {

    public static ANOVAResult performOneWayANOVA(IVector... groups) {
        // 单因素方差分析的核心计算方法
        // 参数：groups - 可变参数，每个IVector代表一个组的数据
        // 返回值：ANOVAResult对象，包含ANOVA分析结果

        int k = groups.length;           // 组数
        int n = groups[0].length();      // 每组样本量（假设各组样本量相等）
        int totalN = k * n;              // 总样本量

        // 计算总均值（Grand Mean）
        // 所有观测值的平均值
        float grandMean = 0;
        for (IVector group : groups) {
            // IVector.sum() - 计算向量元素之和
            grandMean += group.sum();
        }
        grandMean /= totalN;

        // 计算组间平方和（SSB - Sum of Squares Between）
        // 衡量组间差异的大小
        float ssBetween = 0;
        for (IVector group : groups) {
            float groupMean = group.mean();  // 组均值
            // 公式：SSB = Σ n_i * (x̄_i - x̄̄)²
            ssBetween += n * (groupMean - grandMean) * (groupMean - grandMean);
        }

        // 计算组内平方和（SSW - Sum of Squares Within）
        // 衡量组内变异的大小
        float ssWithin = 0;
        for (IVector group : groups) {
            float groupMean = group.mean();
            for (int i = 0; i < group.length(); i++) {
                // IVector.get(i) - 获取第i个元素
                float diff = group.get(i) - groupMean;
                // 公式：SSW = Σ Σ (x_ij - x̄_i)²
                ssWithin += diff * diff;
            }
        }

        // 计算F统计量
        // MSB = SSB / (k-1) - 组间均方
        float msBetween = ssBetween / (k - 1);
        // MSW = SSW / (N-k) - 组内均方
        float msWithin = ssWithin / (totalN - k);
        // F = MSB / MSW - F统计量
        float fStatistic = msBetween / msWithin;

        // 计算p值
        // Stat.f(df1, df2) - 创建F分布对象
        // 参数说明：
        //   - df1: 分子自由度 = k-1
        //   - df2: 分母自由度 = N-k
        FDistribution fDist = Stat.f(k - 1, totalN - k);
        // fDist.cdf(fStatistic) - 计算F统计量的累积分布函数值
        // p值 = 1 - CDF(F统计量)
        float pValue = 1.0f - fDist.cdf(fStatistic);

        // 返回ANOVA结果
        return new ANOVAResult(ssBetween, ssWithin, ssBetween + ssWithin, fStatistic, pValue);
    }

    public static TwoWayANOVAResult performTwoWayANOVA(float[][][] data) {
        int a = data.length; // 因素A的水平数
        int b = data[0].length; // 因素B的水平数
        int n = data[0][0].length; // 每组观测数

        // 计算各种均值
        float grandMean = 0;
        float[] rowMeans = new float[a];
        float[] colMeans = new float[b];
        float[][] cellMeans = new float[a][b];

        for (int i = 0; i < a; i++) {
            for (int j = 0; j < b; j++) {
                IVector cell = IVector.of(data[i][j]);
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
        float ssTotal = 0;
        float ssA = 0;
        float ssB = 0;
        float ssAB = 0;
        float ssError = 0;

        for (int i = 0; i < a; i++) {
            for (int j = 0; j < b; j++) {
                for (int k = 0; k < n; k++) {
                    float value = data[i][j][k];
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
        float msA = ssA / (a - 1);
        float msB = ssB / (b - 1);
        float msAB = ssAB / ((a - 1) * (b - 1));
        float msError = ssError / (a * b * (n - 1));

        float fA = msA / msError;
        float fB = msB / msError;
        float fAB = msAB / msError;

        // 计算p值
        FDistribution fDistA = Stat.f(a - 1, a * b * (n - 1));
        FDistribution fDistB = Stat.f(b - 1, a * b * (n - 1));
        FDistribution fDistAB = Stat.f((a - 1) * (b - 1), a * b * (n - 1));

        float pA = 1.0f - fDistA.cdf(fA);
        float pB = 1.0f - fDistB.cdf(fB);
        float pAB = 1.0f - fDistAB.cdf(fAB);

        return new TwoWayANOVAResult(fA, pA, fB, pB, fAB, pAB);
    }

    public static RepeatedMeasuresANOVAResult performRepeatedMeasuresANOVA(float[][] data) {
        int n = data.length; // 被试数
        int k = data[0].length; // 时间点数

        // 计算各种均值
        float grandMean = 0;
        float[] timeMeans = new float[k];
        float[] subjectMeans = new float[n];

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
        float ssTotal = 0;
        float ssTime = 0;
        float ssSubject = 0;
        float ssError = 0;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                float value = data[i][j];
                ssTotal += (value - grandMean) * (value - grandMean);
                ssTime += (timeMeans[j] - grandMean) * (timeMeans[j] - grandMean);
                ssSubject += (subjectMeans[i] - grandMean) * (subjectMeans[i] - grandMean);
            }
        }

        ssTime *= n;
        ssSubject *= k;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                float value = data[i][j];
                ssError += (value - timeMeans[j] - subjectMeans[i] + grandMean)
                        * (value - timeMeans[j] - subjectMeans[i] + grandMean);
            }
        }

        // 计算F统计量
        float msTime = ssTime / (k - 1);
        float msSubject = ssSubject / (n - 1);
        float msError = ssError / ((k - 1) * (n - 1));

        float fTime = msTime / msError;
        float fSubject = msSubject / msError;

        // 计算p值
        FDistribution fDistTime = Stat.f(k - 1, (k - 1) * (n - 1));
        FDistribution fDistSubject = Stat.f(n - 1, (k - 1) * (n - 1));

        float pTime = 1.0f - fDistTime.cdf(fTime);
        float pSubject = 1.0f - fDistSubject.cdf(fSubject);

        return new RepeatedMeasuresANOVAResult(fTime, pTime, fSubject, pSubject);
    }

    public static boolean testNormality(IVector group) {
        System.out.println("=== 正态性检验 / Normality Test ===");

        // 简化的正态性检验（使用偏度和峰度）
        // IVector.skewness() - 计算偏度，衡量分布的对称性
        // 偏度 = 0 表示完全对称，|偏度| < 1 表示近似对称
        float skewness = group.skewness();

        // IVector.kurtosis() - 计算峰度，衡量分布的尖锐程度
        // 峰度 = 0 表示正态分布，|峰度| < 1 表示接近正态分布
        float kurtosis = group.kurtosis();

        System.out.println("  偏度 / Skewness: " + skewness);
        System.out.println("  峰度 / Kurtosis: " + kurtosis);
        // 简化的正态性判断标准：|偏度| < 1 且 |峰度| < 1
        System.out.println("  正态性 / Normality: "
                + (Math.abs(skewness) < 1.0f && Math.abs(kurtosis) < 1.0f ? "通过 / Pass" : "未通过 / Fail"));
        return Math.abs(skewness) < 1.0f && Math.abs(kurtosis) < 1.0f;
    }

    public static boolean testHomogeneityOfVariance(IVector... groups) {
        System.out.println("\n=== 方差齐性检验 / Homogeneity of Variance Test ===");

        // 计算各组的方差
        float[] variances = new float[groups.length];
        for (int i = 0; i < groups.length; i++) {
            IVector group = groups[i];
            // IVector.var() - 计算样本方差（使用n-1作为分母）
            variances[i] = group.var();
        }

        // Levene检验（简化版）
        // 通过比较最大方差与最小方差的比值来判断方差齐性
        float maxVar = 0;
        float minVar = Float.MAX_VALUE;
        for (float var : variances) {
            maxVar = Math.max(maxVar, var);
            minVar = Math.min(minVar, var);
        }

        // 计算方差比
        // 方差比 = 最大方差 / 最小方差
        // 如果方差比 < 4，通常认为满足方差齐性假设
        float ratio = maxVar / minVar;
        System.out.println("方差比 / Variance ratio: " + ratio);
        System.out.println("方差齐性 / Homogeneity of variance: " + (ratio < 4.0f ? "通过 / Pass" : "未通过 / Fail"));
        return ratio < 4.0f;
    }

    public static void performTukeyHSD(IVector... groups) {
        System.out.println("=== Tukey HSD多重比较 / Tukey HSD Multiple Comparisons ===");

        // Tukey HSD多重比较的核心计算方法
        // 参数：groups - 可变参数，每个float[]代表一个组的数据
        // 功能：比较所有组对之间的差异，控制整体错误率
        int k = groups.length;           // 组数
        int n = groups[0].length();        // 每组样本量（假设各组样本量相等）

        // 计算组均值
        // 计算每个组的样本均值
        float[] means = new float[k];
        for (int i = 0; i < k; i++) {
            IVector group = groups[i];
            // IVector.mean() - 计算组均值
            means[i] = group.mean();
        }

        // 计算合并方差
        // 使用所有组的方差计算合并方差，作为误差方差的估计
        float pooledVariance = 0;
        for (int i = 0; i < k; i++) {
            IVector group = groups[i];
            // IVector.var() - 计算组方差
            pooledVariance += group.var();
        }
        pooledVariance /= k;  // 合并方差 = 各组方差的平均值

        // 计算Tukey HSD临界值
        // calculateTukeyCritical(k, df, alpha) - 计算Tukey临界值
        // 参数说明：
        //   - k: 组数
        //   - df: 误差自由度 = N - k
        //   - alpha: 显著性水平（如0.05）
        float qCritical = calculateTukeyCritical(k, n * k - k, 0.05f);

        // 计算HSD临界值
        // HSD = q * sqrt(MSE / n)
        // 其中：q是Tukey临界值，MSE是合并方差，n是每组样本量
        float hsd = qCritical * (float) Math.sqrt(pooledVariance / (float) n);

        System.out.println("Tukey HSD临界值 / Tukey HSD critical value: " + hsd);
        System.out.println("\n组间比较 / Between-group comparisons:");

        // 进行所有组对之间的比较
        for (int i = 0; i < k; i++) {
            for (int j = i + 1; j < k; j++) {
                // 计算组间均值差异
                float diff = Math.abs(means[i] - means[j]);
                // 判断差异是否显著
                // 如果|均值差异| > HSD，则认为差异显著
                boolean significant = diff > hsd;
                System.out.printf("组%d vs 组%d: 差异=%.3f, %s%n",
                        i + 1, j + 1, diff, significant ? "显著" : "不显著");
            }
        }
    }

    private static float calculateTukeyCritical(int k, int df, float alpha) {
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
