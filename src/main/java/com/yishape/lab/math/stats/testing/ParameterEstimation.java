package com.yishape.lab.math.stats.testing;

import com.yishape.lab.math.stats.Stats;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.math.linalg.IVector;

/**
 * 统计参数估计类 / Statistical Parameter Estimation Class
 * <p>
 * 提供均值和方差的区间估计功能。
 * Provides interval estimation functionality for mean and variance.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ParameterEstimation {

    /**
     * 使用Z分布估计均值的置信区间（默认置信水平95%）
     * Estimate mean confidence interval using Z-distribution (default 95% confidence level)
     *
     * @param sample 样本数据 / Sample data
     * @param sigma 已知标准差 / Known standard deviation
     * @return 置信区间 (下界, 上界)，类型为 {@link Tuple2} / Confidence interval as Tuple2
     */
    public Tuple2<Double,Double> estimateMeanIntevalWithZ(IVector sample, double sigma) {
        return estimateMeanIntevalWithZ(sample, 0.95f, sigma);
    }

    /**
     * 使用Z分布估计均值的置信区间
     * Estimate mean confidence interval using Z-distribution
     *
     * @param sample 样本数据 / Sample data
     * @param confidence 置信水平 / Confidence level
     * @param sigma 已知标准差 / Known standard deviation
     * @return 置信区间 (下界, 上界)，类型为 {@link Tuple2} / Confidence interval as Tuple2
     */
    public Tuple2<Double,Double> estimateMeanIntevalWithZ(IVector sample, double confidence, double sigma) {
        var norm = Stats.norm();
        double left = (1.0 - confidence) / 2.0;
        double right = 1 - left;
        double zLowBound = norm.ppf(left);
        double zUpperBound = norm.ppf(right);
        double mean = sample.meanValue();
        int n = sample.length();
        // 修正：使用正确的公式
        double margin = zUpperBound * sigma / Math.sqrt(n);
        double mLowBound = mean - margin;
        double mUpperBound = mean + margin;
        return new Tuple2(mLowBound, mUpperBound);
    }

    /**
     * 使用t分布估计均值的置信区间（默认置信水平95%）
     * Estimate mean confidence interval using t-distribution (default 95% confidence level)
     *
     * @param sample 样本数据 / Sample data
     * @return 置信区间 (下界, 上界)，类型为 {@link Tuple2} / Confidence interval as Tuple2
     */
    public Tuple2<Double,Double> estimateMeanIntevalWithT(IVector sample) {
        return estimateMeanIntevalWithT(sample, 0.95f);
    }

    /**
     * 使用t分布估计均值的置信区间
     * Estimate mean confidence interval using t-distribution
     *
     * @param sample 样本数据 / Sample data
     * @param confidence 置信水平 / Confidence level
     * @return 置信区间 (下界, 上界)，类型为 {@link Tuple2} / Confidence interval as Tuple2
     */
    public Tuple2<Double,Double> estimateMeanIntevalWithT(IVector sample, double confidence) {
        var t = Stats.t(sample.length() - 1);
        double left = (1.0 - confidence) / 2.0;
        double right = 1 - left;
        double tLowBound = t.ppf(left);
        double tUpperBound = t.ppf(right);
        double mean = sample.meanValue();
        int n = sample.length();
        double s = sample.stdValue(1);
        // 修正：使用正确的公式
        double margin = tUpperBound * s / Math.sqrt(n);
        double mLowBound = mean - margin;
        double mUpperBound = mean + margin;
        return new Tuple2(mLowBound, mUpperBound);
    }

    /**
     * 使用卡方分布估计方差的置信区间（默认置信水平95%）
     * Estimate variance confidence interval using chi-square distribution (default 95% confidence level)
     *
     * @param sample 样本数据 / Sample data
     * @return 置信区间 (下界, 上界)，类型为 {@link Tuple2} / Confidence interval as Tuple2
     */
    public Tuple2<Double,Double> estimateVarIntevalWithChi2(IVector sample) {
        return estimateVarIntevalWithChi2(sample, 0.95f);
    }

    /**
     * 使用卡方分布估计方差的置信区间
     * Estimate variance confidence interval using chi-square distribution
     *
     * @param sample 样本数据 / Sample data
     * @param confidence 置信水平 / Confidence level
     * @return 置信区间 (下界, 上界)，类型为 {@link Tuple2} / Confidence interval as Tuple2
     */
    public Tuple2<Double,Double> estimateVarIntevalWithChi2(IVector sample, double confidence) {
        var chi2 = Stats.chi2(sample.length() - 1);
        double left = (float) (1.0f - confidence) / 2.0f;
        double right = 1 - left;
        double chi2LowBound = chi2.ppf(left);
        double chi2UpperBound = chi2.ppf(right);
        int n = sample.length();
        double s2 = sample.varValue(1);
        double varLowBound = (n - 1) * s2 / chi2UpperBound;
        double varUpperBound = (n - 1) * s2 / chi2LowBound;
        return new Tuple2(varLowBound, varUpperBound);
    }

}
