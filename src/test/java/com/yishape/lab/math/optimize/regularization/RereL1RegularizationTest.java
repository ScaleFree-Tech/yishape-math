package com.yishape.lab.math.optimize.regularization;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link RereL1Regularization}.
 */
class RereL1RegularizationTest {

    // ==================== Gradient ====================

    @Test
    void gradient_positiveValues() {
        RereL1Regularization l1 = new RereL1Regularization();
        IVector<Double> x = Linalg.vector(new double[]{1.0, 2.0, 3.0});
        IVector<Double> grad = l1.computeGradient(x);
        // For |x| >> c, gradient ≈ sign(x) = 1
        assertEquals(1.0, grad.get(0), 1e-5);
        assertEquals(1.0, grad.get(1), 1e-5);
        assertEquals(1.0, grad.get(2), 1e-5);
    }

    @Test
    void gradient_negativeValues() {
        RereL1Regularization l1 = new RereL1Regularization();
        IVector<Double> x = Linalg.vector(new double[]{-1.0, -2.0, -3.0});
        IVector<Double> grad = l1.computeGradient(x);
        assertEquals(-1.0, grad.get(0), 1e-5);
        assertEquals(-1.0, grad.get(1), 1e-5);
        assertEquals(-1.0, grad.get(2), 1e-5);
    }

    @Test
    void gradient_atZero() {
        RereL1Regularization l1 = new RereL1Regularization();
        IVector<Double> x = Linalg.vector(new double[]{0.0});
        IVector<Double> grad = l1.computeGradient(x);
        // At zero, gradient of Huber is 0/c = 0
        assertEquals(0, grad.get(0), 1e-5);
    }

    // ==================== Objective ====================

    @Test
    void objective_positiveValues() {
        RereL1Regularization l1 = new RereL1Regularization();
        IVector<Double> x = Linalg.vector(new double[]{3.0, 4.0});
        double obj = l1.computeObjective(x);
        // For |x| >> c, objective ≈ |x|
        assertEquals(7.0, obj, 1e-5);
    }

    @Test
    void objective_negativeValues() {
        RereL1Regularization l1 = new RereL1Regularization();
        IVector<Double> x = Linalg.vector(new double[]{-3.0, -4.0});
        double obj = l1.computeObjective(x);
        assertEquals(7.0, obj, 1e-5);
    }

    @Test
    void objective_atZero() {
        RereL1Regularization l1 = new RereL1Regularization();
        IVector<Double> x = Linalg.vector(new double[]{0.0});
        double obj = l1.computeObjective(x);
        // c/2 where c = 1e-10
        assertEquals(0.5e-10, obj, 1e-15);
    }

    @Test
    void objective_nonNegative() {
        RereL1Regularization l1 = new RereL1Regularization();
        IVector<Double> x = Linalg.vector(new double[]{-5.0, 3.0, 0.0, 1.0});
        assertTrue(l1.computeObjective(x) >= 0);
    }

    @Test
    void objective_singleElement() {
        RereL1Regularization l1 = new RereL1Regularization();
        IVector<Double> x = Linalg.vector(new double[]{5.0});
        assertEquals(5.0, l1.computeObjective(x), 1e-5);
    }
}
