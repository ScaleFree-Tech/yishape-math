package com.reremouse.lab.math.stat.anova;

import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.stat.Stat;
import com.reremouse.lab.math.stat.distribution.NormalDistribution;

/**
 *
 * @author lteb2
 */
public class ANOVATest {

    public static void demonstrateOneWayANOVA() {
        System.out.println("=== 单因素方差分析 / One-way ANOVA ===");

        // 生成三个组的数据
        // 创建三个不同均值的正态分布，模拟不同处理组
        // Stat.norm(mean, std) - 创建正态分布
        // 参数说明：
        //   - mean: 分布均值（组间差异体现在均值上）
        //   - std: 分布标准差（假设各组方差相等）
        NormalDistribution group1 = Stat.norm(100.0f, 15.0f);  // 对照组
        NormalDistribution group2 = Stat.norm(105.0f, 15.0f);  // 处理组1
        NormalDistribution group3 = Stat.norm(110.0f, 15.0f);  // 处理组2

        // 从每个组抽取样本
        // sample(n) - 从分布中抽取n个随机样本
        // 参数：n - 样本容量（每组样本量相等）
        float[] data1 = group1.sample(20);
        float[] data2 = group2.sample(20);
        float[] data3 = group3.sample(20);

        // 将样本数据转换为IVector对象
        // IVector.of(array) - 将float数组转换为IVector对象
        IVector vector1 = IVector.of(data1);
        IVector vector2 = IVector.of(data2);
        IVector vector3 = IVector.of(data3);

        // 计算ANOVA统计量
        // performOneWayANOVA(vectors...) - 执行单因素方差分析
        // 参数：可变参数，每个IVector代表一个组的数据
        // 返回值：ANOVAResult对象，包含ANOVA分析结果
        ANOVAResult result = ANOVA.performOneWayANOVA(vector1, vector2, vector3);

        System.out.println("组别统计量 / Group statistics:");
        // IVector.mean() - 计算组均值
        // IVector.std() - 计算组标准差
        System.out.println("  组1 / Group 1: 均值=" + vector1.mean() + ", 标准差=" + vector1.std());
        System.out.println("  组2 / Group 2: 均值=" + vector2.mean() + ", 标准差=" + vector2.std());
        System.out.println("  组3 / Group 3: 均值=" + vector3.mean() + ", 标准差=" + vector3.std());

        System.out.println("\nANOVA结果 / ANOVA results:");
        // ANOVAResult.ssBetween - 组间平方和
        System.out.println("  组间平方和 / Between-group sum of squares: " + result.ssBetween);
        // ANOVAResult.ssWithin - 组内平方和
        System.out.println("  组内平方和 / Within-group sum of squares: " + result.ssWithin);
        // ANOVAResult.ssTotal - 总平方和
        System.out.println("  总平方和 / Total sum of squares: " + result.ssTotal);
        // ANOVAResult.fStatistic - F统计量
        System.out.println("  F统计量 / F-statistic: " + result.fStatistic);
        // ANOVAResult.pValue - p值
        System.out.println("  p值 / p-value: " + result.pValue);
        System.out.println("  结论 / Conclusion: " + (result.pValue < 0.05f ? "拒绝等均值假设 / Reject equal means hypothesis" : "接受等均值假设 / Accept equal means hypothesis"));
    }

    public static void demonstrateTwoWayANOVA() {
        System.out.println("\n=== 两因素方差分析 / Two-way ANOVA ===");

        // 模拟2×3设计的数据
        float[][][] data = new float[2][3][10]; // 2个因素，3个水平，每组10个观测

        // 生成数据
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                float mean = 100 + i * 5 + j * 3; // 主效应
                NormalDistribution dist = Stat.norm(mean, 10.0f);
                data[i][j] = dist.sample(10);
            }
        }

        // 计算两因素ANOVA / Calculate two-way ANOVA
        TwoWayANOVAResult result = ANOVA.performTwoWayANOVA(data);

        System.out.println("两因素ANOVA结果 / Two-way ANOVA results:");
        System.out.println("  因素A主效应F值 / Factor A main effect F-value: " + result.factorAF);
        System.out.println("  因素A p值 / Factor A p-value: " + result.factorAP);
        System.out.println("  因素B主效应F值 / Factor B main effect F-value: " + result.factorBF);
        System.out.println("  因素B p值 / Factor B p-value: " + result.factorBP);
        System.out.println("  交互效应F值 / Interaction effect F-value: " + result.interactionF);
        System.out.println("  交互效应p值 / Interaction effect p-value: " + result.interactionP);
    }

    public static void demonstrateRepeatedMeasuresANOVA() {
        System.out.println("\n=== 重复测量ANOVA / Repeated Measures ANOVA ===");

        // 模拟重复测量数据（3个时间点，10个被试）
        float[][] repeatedData = new float[10][3];

        for (int subject = 0; subject < 10; subject++) {
            for (int time = 0; time < 3; time++) {
                float mean = 100 + time * 5; // 时间效应
                NormalDistribution dist = Stat.norm(mean, 15.0f);
                repeatedData[subject][time] = dist.sample();
            }
        }

        // 计算重复测量ANOVA
        RepeatedMeasuresANOVAResult result = ANOVA.performRepeatedMeasuresANOVA(repeatedData);

        System.out.println("重复测量ANOVA结果 / Repeated Measures ANOVA results:");
        System.out.println("  时间效应F值 / Time effect F-value: " + result.timeF);
        System.out.println("  时间效应p值 / Time effect p-value: " + result.timeP);
        System.out.println("  被试效应F值 / Subject effect F-value: " + result.subjectF);
        System.out.println("  被试效应p值 / Subject effect p-value: " + result.subjectP);
    }
}
