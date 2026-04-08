package com.yishape.lab.math.ml.cls;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * 逻辑回归正则类型推断与线性回归一致。
 */
class RereLogisticRegressionRegularizationTest {

    @Test
    void regularizationInferenceMatchesLinearStyle() {
        assertType(new RereLogisticRegression(0, 0), RereLogisticRegression.RegularizationType.NONE);
        assertType(new RereLogisticRegression(0.1, 0), RereLogisticRegression.RegularizationType.L1);
        assertType(new RereLogisticRegression(0, 0.1), RereLogisticRegression.RegularizationType.L2);
        assertType(new RereLogisticRegression(0.01, 0.1), RereLogisticRegression.RegularizationType.ELASTIC_NET);
    }

    private static void assertType(RereLogisticRegression lr, RereLogisticRegression.RegularizationType expected) {
        assertEquals(expected, lr.getRegularizationType());
    }
}
