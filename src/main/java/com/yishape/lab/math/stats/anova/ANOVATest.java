package com.yishape.lab.math.stats.anova;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.yishape.lab.math.stats.Stats;
import com.yishape.lab.math.stats.distribution.NormalDistribution;
import com.yishape.lab.math.linalg.IDoubleVector;

/**
 *
 * @author lteb2
 */
public class ANOVATest {

    private static final Logger log = LoggerFactory.getLogger(ANOVATest.class);


    public static void demonstrateOneWayANOVA() {
        log.debug("=== 单因素方差分析 / One-way ANOVA ===");

        // 生成三个组的数据
        // 创建三个不同均值的正态分布，模拟不同处理组
        // Stats.norm(mean, std) - 创建正态分布
        // 参数说明：
        //   - mean: 分布均值（组间差异体现在均值上）
        //   - std: 分布标准差（假设各组方差相等）
        NormalDistribution group1 = Stats.norm(100.0f, 15.0f);  // 对照组
        NormalDistribution group2 = Stats.norm(105.0f, 15.0f);  // 处理组1
        NormalDistribution group3 = Stats.norm(110.0f, 15.0f);  // 处理组2

        // 从每个组抽取样本
        // sample(n) - 从分布中抽取n个随机样本
        // 参数：n - 样本容量（每组样本量相等）
        double[] data1 = group1.sample(20);
        double[] data2 = group2.sample(20);
        double[] data3 = group3.sample(20);

        // 将样本数据转换为IVector对象
        // IDoubleVector.of(array) - 将double数组转换为IVector对象
        IDoubleVector vector1 = IDoubleVector.of(data1);
        IDoubleVector vector2 = IDoubleVector.of(data2);
        IDoubleVector vector3 = IDoubleVector.of(data3);

        // 计算ANOVA统计量
        // performOneWayANOVA(vectors...) - 执行单因素方差分析
        // 参数：可变参数，每个IVector代表一个组的数据
        // 返回值：ANOVAResult对象，包含ANOVA分析结果
        ANOVAResult result = Stats.anova.performOneWayANOVA(vector1, vector2, vector3);

        log.debug("组别统计量 / Group statistics:");
        // IDoubleVector.mean() - 计算组均值
        // IDoubleVector.std() - 计算组标准差
        log.debug("  组1 / Group 1: 均值=" + vector1.mean() + ", 标准差=" + vector1.std());
        log.debug("  组2 / Group 2: 均值=" + vector2.mean() + ", 标准差=" + vector2.std());
        log.debug("  组3 / Group 3: 均值=" + vector3.mean() + ", 标准差=" + vector3.std());

        log.debug("\nANOVA结果 / ANOVA results:");
        // ANOVAResult.ssBetween - 组间平方和
        log.debug("  组间平方和 / Between-group sum of squares: " + result.ssBetween);
        // ANOVAResult.ssWithin - 组内平方和
        log.debug("  组内平方和 / Within-group sum of squares: " + result.ssWithin);
        // ANOVAResult.ssTotal - 总平方和
        log.debug("  总平方和 / Total sum of squares: " + result.ssTotal);
        // ANOVAResult.fStatistic - F统计量
        log.debug("  F统计量 / F-statistic: " + result.fStatistic);
        // ANOVAResult.pValue - p值
        log.debug("  p值 / p-value: " + result.pValue);
        log.debug("  结论 / Conclusion: " + (result.pValue < 0.05f ? "拒绝等均值假设 / Reject equal means hypothesis" : "接受等均值假设 / Accept equal means hypothesis"));
    }

    public static void demonstrateTwoWayANOVA() {
        log.debug("\n=== 两因素方差分析 / Two-way ANOVA ===");

        // 模拟2×3设计的数据
        double[][][] data = new double[2][3][10]; // 2个因素，3个水平，每组10个观测

        // 生成数据
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                double mean = 100 + i * 5 + j * 3; // 主效应
                NormalDistribution dist = Stats.norm(mean, 10.0);
                data[i][j] = dist.sample(10);
            }
        }

        // 计算两因素ANOVA / Calculate two-way ANOVA
        TwoWayANOVAResult result = Stats.anova.performTwoWayANOVA(data);

        log.debug("两因素ANOVA结果 / Two-way ANOVA results:");
        log.debug("  因素A主效应F值 / Factor A main effect F-value: " + result.factorAF);
        log.debug("  因素A p值 / Factor A p-value: " + result.factorAP);
        log.debug("  因素B主效应F值 / Factor B main effect F-value: " + result.factorBF);
        log.debug("  因素B p值 / Factor B p-value: " + result.factorBP);
        log.debug("  交互效应F值 / Interaction effect F-value: " + result.interactionF);
        log.debug("  交互效应p值 / Interaction effect p-value: " + result.interactionP);
    }

    public static void demonstrateRepeatedMeasuresANOVA() {
        log.debug("\n=== 重复测量ANOVA / Repeated Measures ANOVA ===");

        // 模拟重复测量数据（3个时间点，10个被试）
        double[][] repeatedData = new double[10][3];

        for (int subject = 0; subject < 10; subject++) {
            for (int time = 0; time < 3; time++) {
                double mean = 100 + time * 5; // 时间效应
                NormalDistribution dist = Stats.norm(mean, 15.0);
                repeatedData[subject][time] = dist.sample();
            }
        }

        // 计算重复测量ANOVA
        RepeatedMeasuresANOVAResult result = Stats.anova.performRepeatedMeasuresANOVA(repeatedData);

        log.debug("重复测量ANOVA结果 / Repeated Measures ANOVA results:");
        log.debug("  时间效应F值 / Time effect F-value: " + result.timeF);
        log.debug("  时间效应p值 / Time effect p-value: " + result.timeP);
        log.debug("  被试效应F值 / Subject effect F-value: " + result.subjectF);
        log.debug("  被试效应p值 / Subject effect p-value: " + result.subjectP);
    }
}
