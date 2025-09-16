package com.reremouse.lab.math.stats.testing;

import com.reremouse.lab.math.stats.Stats;
import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.math.linalg.IDoubleVector;

/**
 *
 * @author lteb2
 */
public class HypothesisTesting {

    ParameterEstimation estimator = new ParameterEstimation();

    public TestingResult testMeanEqualWithT(double h0, IDoubleVector sample, double confidence) {
        var t = Stats.t(sample.length() - 1);
        Tuple2<Double, Double> tp = estimator.estimateMeanIntevalWithT(sample, confidence);
        double p = 1 - t.cdf(h0);
        if (p > 0.5) {
            p = 1 - p;
        }
        boolean pass = (h0 >= tp._1 && h0 <= tp._2);
        return new TestingResult(pass, p, tp);
    }

    public TestingResult testVarEqualWithChi2(double h0, IDoubleVector sample, double confidence) {
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
