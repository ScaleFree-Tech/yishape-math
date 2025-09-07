package com.reremouse.lab.math.stat;

import com.reremouse.lab.math.stat.distribution.NormalDistribution;
import com.reremouse.lab.math.stat.distribution.StudentDistribution;
import com.reremouse.lab.math.stat.distribution.UniformDistribution;
import com.reremouse.lab.math.stat.distribution.ExponentialDistribution;
import com.reremouse.lab.math.stat.distribution.Chi2Distribution;
import com.reremouse.lab.math.stat.distribution.FDistribution;

/**
 *
 * @author lteb2
 */
public class Stat {


    public static NormalDistribution norm(float mean, float stdDev) {
        return new NormalDistribution(mean, stdDev);
    }
    
    public static NormalDistribution norm() {
        return new NormalDistribution(0,1);
    }

    public static StudentDistribution t(float degreesOfFreedom) {
        return new StudentDistribution(degreesOfFreedom);
    }

    public static UniformDistribution uniform(float lowerBound, float upperBound) {
        return new UniformDistribution(lowerBound, upperBound);
    }

    public static ExponentialDistribution exponential(float rate) {
        return new ExponentialDistribution(rate);
    }

    public static Chi2Distribution chi2(float degreesOfFreedom) {
        return new Chi2Distribution(degreesOfFreedom);
    }

    public static FDistribution f(float degreesOfFreedom1, float degreesOfFreedom2) {
        return new FDistribution(degreesOfFreedom1, degreesOfFreedom2);
    }
}
