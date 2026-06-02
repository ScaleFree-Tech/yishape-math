package com.yishape.lab.math.optimize.regularization;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link RereL2Regularization}.
 */
class RereL2RegularizationTest {

    // ==================== Gradient ====================

    @Test
    void gradient_is2x() {
        RereL2Regularization l2 = new RereL2Regularization();
        IVector<Double> x = Linalg.vector(new double[]{1.0, 2.0, 3.0});
        IVector<Double> grad = l2.computeGradient(x);
        assertEquals(2.0, grad.get(0), 1e-10);
        assertEquals(4.0, grad.get(1), 1e-10);
        assertEquals(6.0, grad.get(2), 1e-10);
    }

    @Test
    void gradient_negativeValues() {
        RereL2Regularization l2 = new RereL2Regularization();
        IVector<Double> x = Linalg.vector(new double[]{-1.0, -2.0});
        IVector<Double> grad = l2.computeGradient(x);
        assertEquals(-2.0, grad.get(0), 1e-10);
        assertEquals(-4.0, grad.get(1), 1e-10);
    }

    @Test
    void gradient_atZero() {
        RereL2Regularization l2 = new RereL2Regularization();
        IVector<Double> x = Linalg.vector(new double[]{0.0, 0.0});
        IVector<Double> grad = l2.computeGradient(x);
        assertEquals(0, grad.get(0), 1e-10);
        assertEquals(0, grad.get(1), 1e-10);
    }

    // ==================== Objective ====================

    @Test
    void objective_isSumOfSquares() {
        RereL2Regularization l2 = new RereL2Regularization();
        IVector<Double> x = Linalg.vector(new double[]{1.0, 2.0, 3.0});
        double obj = l2.computeObjective(x);
        // 1^2 + 2^2 + 3^2 = 14
        assertEquals(14.0, obj, 1e-10);
    }

    @Test
    void objective_zeroVector() {
        RereL2Regularization l2 = new RereL2Regularization();
        IVector<Double> x = Linalg.vector(new double[]{0.0, 0.0});
        assertEquals(0.0, l2.computeObjective(x), 1e-10);
    }

    @Test
    void objective_singleElement() {
        RereL2Regularization l2 = new RereL2Regularization();
        IVector<Double> x = Linalg.vector(new double[]{5.0});
        assertEquals(25.0, l2.computeObjective(x), 1e-10);
    }

    @Test
    void objective_nonNegative() {
        RereL2Regularization l2 = new RereL2Regularization();
        IVector<Double> x = Linalg.vector(new double[]{-3.0, 4.0});
        assertTrue(l2.computeObjective(x) >= 0);
    }

    @Test
    void objective_negativeValues() {
        RereL2Regularization l2 = new RereL2Regularization();
        IVector<Double> x = Linalg.vector(new double[]{-1.0, -2.0, -3.0});
        assertEquals(14.0, l2.computeObjective(x), 1e-10);
    }
}
