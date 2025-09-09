package com.reremouse.lab.math.stat.testing;

import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.stat.Stat;
import com.reremouse.lab.util.Tuple2;

/**
 * 统计参数估计
 *
 * @author lteb2
 */
public class ParameterEstimation {

    public Tuple2<Float,Float> estimateMeanIntevalWithZ(IVector sample, float sigma) {
        return estimateMeanIntevalWithZ(sample, 0.95f, sigma);
    }

    public Tuple2<Float,Float> estimateMeanIntevalWithZ(IVector sample, float confidence, float sigma) {
        var norm = Stat.norm();
        float left = (float) (1.0f - confidence) / 2.0f;
        float right = 1 - left;
        float zLowBound = norm.ppf(left);
        float zUpperBound = norm.ppf(right);
        float mean = sample.mean();
        int n = sample.length();
        float mLowBound = mean - zUpperBound * sigma / (float) n;
        float mUpperBound = mean - zLowBound * sigma / (float) n;
        return new Tuple2(mLowBound, mUpperBound);
    }

    public Tuple2<Float,Float> estimateMeanIntevalWithT(IVector sample) {
        return estimateMeanIntevalWithT(sample, 0.95f);
    }

    public Tuple2<Float,Float> estimateMeanIntevalWithT(IVector sample, float confidence) {
        var t = Stat.t(sample.length() - 1);
        float left = (float) (1.0f - confidence) / 2.0f;
        float right = 1 - left;
        float tLowBound = t.ppf(left);
        float tUpperBound = t.ppf(right);
        float mean = sample.mean();
        int n = sample.length();
        float s = sample.std(1);
        float mLowBound = mean - tUpperBound * s / (float) n;
        float mUpperBound = mean - tLowBound * s / (float) n;
        return new Tuple2(mLowBound, mUpperBound);
    }

    public Tuple2<Float,Float> estimateVarIntevalWithChi2(IVector sample) {
        return estimateVarIntevalWithChi2(sample, 0.95f);
    }

    public Tuple2<Float,Float> estimateVarIntevalWithChi2(IVector sample, float confidence) {
        var chi2 = Stat.chi2(sample.length() - 1);
        float left = (float) (1.0f - confidence) / 2.0f;
        float right = 1 - left;
        float chi2LowBound = chi2.ppf(left);
        float chi2UpperBound = chi2.ppf(right);
        int n = sample.length();
        float s2 = sample.var(1);
        float varLowBound = (n - 1) * s2 / chi2UpperBound;
        float varUpperBound = (n - 1) * s2 / chi2LowBound;
        return new Tuple2(varLowBound, varUpperBound);
    }

}
