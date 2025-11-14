package com.yishape.lab.math.stats.testing;

import com.yishape.lab.math.stats.Stats;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.math.linalg.IVector;

/**
 * 统计参数估计
 *
 * @author lteb2
 */
public class ParameterEstimation {

    public Tuple2<Double,Double> estimateMeanIntevalWithZ(IVector sample, double sigma) {
        return estimateMeanIntevalWithZ(sample, 0.95f, sigma);
    }

    public Tuple2<Double,Double> estimateMeanIntevalWithZ(IVector sample, double confidence, double sigma) {
        var norm = Stats.norm();
        double left = (1.0 - confidence) / 2.0;
        double right = 1 - left;
        double zLowBound = norm.ppf(left);
        double zUpperBound = norm.ppf(right);
        double mean = (double)sample.mean();
        int n = sample.length();
        // 修正：使用正确的公式
        double margin = zUpperBound * sigma / Math.sqrt(n);
        double mLowBound = mean - margin;
        double mUpperBound = mean + margin;
        return new Tuple2(mLowBound, mUpperBound);
    }

    public Tuple2<Double,Double> estimateMeanIntevalWithT(IVector sample) {
        return estimateMeanIntevalWithT(sample, 0.95f);
    }

    public Tuple2<Double,Double> estimateMeanIntevalWithT(IVector sample, double confidence) {
        var t = Stats.t(sample.length() - 1);
        double left = (1.0 - confidence) / 2.0;
        double right = 1 - left;
        double tLowBound = t.ppf(left);
        double tUpperBound = t.ppf(right);
        double mean = (double)sample.mean();
        int n = sample.length();
        double s = (double)sample.std(1);
        // 修正：使用正确的公式
        double margin = tUpperBound * s / Math.sqrt(n);
        double mLowBound = mean - margin;
        double mUpperBound = mean + margin;
        return new Tuple2(mLowBound, mUpperBound);
    }

    public Tuple2<Double,Double> estimateVarIntevalWithChi2(IVector sample) {
        return estimateVarIntevalWithChi2(sample, 0.95f);
    }

    public Tuple2<Double,Double> estimateVarIntevalWithChi2(IVector sample, double confidence) {
        var chi2 = Stats.chi2(sample.length() - 1);
        double left = (float) (1.0f - confidence) / 2.0f;
        double right = 1 - left;
        double chi2LowBound = chi2.ppf(left);
        double chi2UpperBound = chi2.ppf(right);
        int n = sample.length();
        double s2 = (double)sample.var(1);
        double varLowBound = (n - 1) * s2 / chi2UpperBound;
        double varUpperBound = (n - 1) * s2 / chi2LowBound;
        return new Tuple2(varLowBound, varUpperBound);
    }

}
