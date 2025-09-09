package com.reremouse.lab.math.stat.testing;

import com.reremouse.lab.util.Tuple2;

/**
 *
 * @author lteb2
 */
public class TestingResult {

    public Boolean pass = false;
    public float p = 0.0f;
    public Tuple2<Float, Float> criticalInteval;

    public TestingResult(Boolean pass, float p, Tuple2 criticalInteval) {
        this.pass = pass;
        this.p = p;
        this.criticalInteval = criticalInteval;
    }

}
