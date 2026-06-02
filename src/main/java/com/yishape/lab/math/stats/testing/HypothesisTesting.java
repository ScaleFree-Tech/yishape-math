package com.yishape.lab.math.stats.testing;

import com.yishape.lab.math.stats.Stats;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.math.linalg.IVector;

/**
 * 假设检验类 / Hypothesis Testing Class
 * <p>
 * 提供多种假设检验方法，包括t检验和卡方检验。
 * Provides various hypothesis testing methods including t-test and chi-square test.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class HypothesisTesting {

    /** 参数估计器 / Parameter estimator */
    ParameterEstimation estimator = new ParameterEstimation();

    /**
     * 使用t检验检验均值是否等于假设值
     * Test if mean equals hypothesized value using t-test
     *
     * @param h0 假设的均值 / Hypothesized mean value
     * @param sample 样本数据 / Sample data
     * @param confidence 置信水平 / Confidence level
     * @return 检验结果 / Test result
     */
    public TestingResult testMeanEqualWithT(double h0, IVector sample, double confidence) {
        int n = sample.length();
        double sampleMean = sample.meanValue();
        double sampleStd = Math.sqrt(sample.varValue()); // 样本标准差
        double standardError = sampleStd / Math.sqrt(n);

        // 计算 t 统计量: t = (x̄ − μ₀) / (s / √n)
        double tStatistic = (sampleMean - h0) / standardError;

        var tDist = Stats.t(n - 1);
        Tuple2<Double, Double> tp = estimator.estimateMeanIntevalWithT(sample, confidence);

        // 双尾检验: p = 2 * P(T > |t|)
        double p = 2.0 * (1.0 - tDist.cdf(Math.abs(tStatistic)));
        if (p > 1.0) {
            p = 1.0;
        }

        boolean pass = (h0 >= tp._1 && h0 <= tp._2);
        return new TestingResult(pass, p, tp);
    }

    /**
     * 使用卡方检验检验方差是否等于假设值
     * Test if variance equals hypothesized value using chi-square test
     *
     * @param h0 假设的方差 / Hypothesized variance value
     * @param sample 样本数据 / Sample data
     * @param confidence 置信水平 / Confidence level
     * @return 检验结果 / Test result
     */
    public TestingResult testVarEqualWithChi2(double h0, IVector sample, double confidence) {
        var chi2 = Stats.chi2(sample.length() - 1);
        Tuple2<Double, Double> tp = estimator.estimateVarIntevalWithChi2(sample, confidence);
        double p = 1 - chi2.cdf(h0);
        if (p > 0.5) {
            p = 1 - p;
        }
        boolean pass = (h0 >= tp._1 && h0 <= tp._2);
        return new TestingResult(pass, p, tp);
    }

}
