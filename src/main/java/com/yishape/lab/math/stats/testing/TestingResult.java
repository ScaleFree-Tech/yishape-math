package com.yishape.lab.math.stats.testing;

import com.yishape.lab.util.Tuple2;

/**
 * 检验结果类 / Testing Result Class
 * <p>
 * 用于封装假设检验的结果，包括是否通过、p值和置信区间。
 * Encapsulates hypothesis testing results including pass/fail status, p-value, and confidence interval.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class TestingResult {

    /** 是否通过检验 / Whether the test passed */
    public Boolean pass = false;
    /** p值 / p-value */
    public double p = 0.0f;
    /** 置信区间 / Confidence interval */
    public Tuple2<Double, Double> criticalInteval;

    /**
     * 创建检验结果对象 / Create Testing Result Object
     *
     * @param pass 是否通过检验 / Whether the test passed
     * @param p p值 / p-value
     * @param criticalInteval 置信区间 / Critical interval
     */
    public TestingResult(Boolean pass, double p, Tuple2 criticalInteval) {
        this.pass = pass;
        this.p = p;
        this.criticalInteval = criticalInteval;
    }

}
