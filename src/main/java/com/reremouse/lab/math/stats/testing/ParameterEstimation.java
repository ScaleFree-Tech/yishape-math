package com.reremouse.lab.math.stats.testing;

import com.reremouse.lab.math.stats.Stats;
import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.math.linalg.IDoubleVector;

/**
 * 统计参数估计
 *
 * @author lteb2
 */
public class ParameterEstimation {

    public Tuple2<Double,Double> estimateMeanIntevalWithZ(IDoubleVector sample, double sigma) {
        return estimateMeanIntevalWithZ(sample, 0.95f, sigma);
    }

    public Tuple2<Double,Double> estimateMeanIntevalWithZ(IDoubleVector sample, double confidence, double sigma) {
        var norm = Stats.norm();
        double left = (float) (1.0f - confidence) / 2.0f;
        double right = 1 - left;
        double zLowBound = norm.ppf(left);
        double zUpperBound = norm.ppf(right);
        double mean = sample.mean();
        int n = sample.length();
        double mLowBound = mean - zUpperBound * sigma / (float) n;
        double mUpperBound = mean - zLowBound * sigma / (float) n;
        return new Tuple2(mLowBound, mUpperBound);
    }

    public Tuple2<Double,Double> estimateMeanIntevalWithT(IDoubleVector sample) {
        return estimateMeanIntevalWithT(sample, 0.95f);
    }

    public Tuple2<Double,Double> estimateMeanIntevalWithT(IDoubleVector sample, double confidence) {
        var t = Stats.t(sample.length() - 1);
        double left = (float) (1.0f - confidence) / 2.0f;
        double right = 1 - left;
        double tLowBound = t.ppf(left);
        double tUpperBound = t.ppf(right);
        double mean = sample.mean();
        int n = sample.length();
        double s = sample.std(1);
        double mLowBound = mean - tUpperBound * s / (float) n;
        double mUpperBound = mean - tLowBound * s / (float) n;
        return new Tuple2(mLowBound, mUpperBound);
    }

    public Tuple2<Double,Double> estimateVarIntevalWithChi2(IDoubleVector sample) {
        return estimateVarIntevalWithChi2(sample, 0.95f);
    }

    public Tuple2<Double,Double> estimateVarIntevalWithChi2(IDoubleVector sample, double confidence) {
        var chi2 = Stats.chi2(sample.length() - 1);
        double left = (float) (1.0f - confidence) / 2.0f;
        double right = 1 - left;
        double chi2LowBound = chi2.ppf(left);
        double chi2UpperBound = chi2.ppf(right);
        int n = sample.length();
        double s2 = sample.var(1);
        double varLowBound = (n - 1) * s2 / chi2UpperBound;
        double varUpperBound = (n - 1) * s2 / chi2LowBound;
        return new Tuple2(varLowBound, varUpperBound);
    }

}
