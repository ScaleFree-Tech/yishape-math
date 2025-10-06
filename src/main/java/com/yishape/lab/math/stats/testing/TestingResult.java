package com.yishape.lab.math.stats.testing;

import com.yishape.lab.util.Tuple2;

/**
 *
 * @author lteb2
 */
public class TestingResult {

    public Boolean pass = false;
    public double p = 0.0f;
    public Tuple2<Double, Double> criticalInteval;

    public TestingResult(Boolean pass, double p, Tuple2 criticalInteval) {
        this.pass = pass;
        this.p = p;
        this.criticalInteval = criticalInteval;
    }

}
