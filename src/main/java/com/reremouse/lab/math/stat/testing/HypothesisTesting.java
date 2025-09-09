package com.reremouse.lab.math.stat.testing;

import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.stat.Stat;
import com.reremouse.lab.util.Tuple2;

/**
 *
 * @author lteb2
 */
public class HypothesisTesting {

    ParameterEstimation estimator = new ParameterEstimation();

    public TestingResult testMeanEqualWithT(float h0, IVector sample, float confidence) {
        var t = Stat.t(sample.length() - 1);
        Tuple2<Float, Float> tp = estimator.estimateMeanIntevalWithT(sample, confidence);
        float p = 1 - t.cdf(h0);
        if (p > 0.5) {
            p = 1 - p;
        }
        boolean pass = (h0 >= tp._1 && h0 <= tp._2);
        return new TestingResult(pass, p, tp);
    }

    public TestingResult testVarEqualWithChi2(float h0, IVector sample, float confidence) {
        var chi2 = Stat.chi2(sample.length() - 1);
        Tuple2<Float, Float> tp = estimator.estimateVarIntevalWithChi2(sample, confidence);
        float p = 1 - chi2.cdf(h0);
        if (p > 0.5) {
            p = 1 - p;
        }
        boolean pass = (h0 >= tp._1 && h0 <= tp._2);
        return new TestingResult(pass, p, tp);
    }

}
