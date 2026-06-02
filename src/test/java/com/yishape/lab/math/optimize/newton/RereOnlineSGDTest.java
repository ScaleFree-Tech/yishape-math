package com.yishape.lab.math.optimize.newton;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link RereOnlineSGD}.
 */
class RereOnlineSGDTest {

    @Test
    void defaultConstructor() {
        RereOnlineSGD sgd = new RereOnlineSGD();
        assertEquals(0.01, sgd.getCurrentLearningRate(), 1e-10);
        assertEquals(0, sgd.getCurrentStep());
        assertFalse(sgd.isInitialized());
    }

    @Test
    void constructor_withLR() {
        RereOnlineSGD sgd = new RereOnlineSGD(0.1);
        assertEquals(0.1, sgd.getCurrentLearningRate(), 1e-10);
    }

    @Test
    void constructor_withMomentum() {
        RereOnlineSGD sgd = new RereOnlineSGD(0.1, 0.9);
        assertNotNull(sgd);
    }

    @Test
    void initialize() {
        RereOnlineSGD sgd = new RereOnlineSGD();
        IVector<Double> params = Linalg.vector(new double[]{1.0, 2.0, 3.0});
        sgd.initialize(params);
        assertTrue(sgd.isInitialized());
        IVector<Double> current = sgd.getCurrentParams();
        assertEquals(3, current.size());
    }

    @Test
    void step_updatesParams() {
        RereOnlineSGD sgd = new RereOnlineSGD(0.1);
        IVector<Double> params = Linalg.vector(new double[]{1.0, 2.0});
        sgd.initialize(params);

        IVector<Double> grad = Linalg.vector(new double[]{0.5, 0.3});
        IVector<Double> newParams = sgd.step(grad);

        assertNotNull(newParams);
        assertEquals(1, sgd.getCurrentStep());
        // params should have moved in negative gradient direction
        assertTrue(newParams.get(0) < 1.0);
    }

    @Test
    void step_withLoss() {
        RereOnlineSGD sgd = new RereOnlineSGD(0.1);
        IVector<Double> params = Linalg.vector(new double[]{1.0, 2.0});
        sgd.initialize(params);

        IVector<Double> grad = Linalg.vector(new double[]{0.5, 0.3});
        IVector<Double> newParams = sgd.step(grad, 1.5);
        assertNotNull(newParams);
    }

    @Test
    void reset() {
        RereOnlineSGD sgd = new RereOnlineSGD(0.1);
        IVector<Double> params = Linalg.vector(new double[]{1.0, 2.0});
        sgd.initialize(params);

        sgd.step(Linalg.vector(new double[]{0.1, 0.1}));
        sgd.step(Linalg.vector(new double[]{0.1, 0.1}));
        assertEquals(2, sgd.getCurrentStep());

        sgd.reset();
        assertEquals(0, sgd.getCurrentStep());
        assertFalse(sgd.isInitialized());
    }

    @Test
    void clone_createsCopy() {
        RereOnlineSGD sgd = new RereOnlineSGD(0.1, 0.9);
        IVector<Double> params = Linalg.vector(new double[]{1.0, 2.0});
        sgd.initialize(params);
        sgd.step(Linalg.vector(new double[]{0.1, 0.1}));

        var clone = sgd.clone();
        assertEquals(sgd.getCurrentStep(), clone.getCurrentStep());
        assertEquals(sgd.getCurrentLearningRate(), clone.getCurrentLearningRate());
    }

    @Test
    void setLearningRate() {
        RereOnlineSGD sgd = new RereOnlineSGD();
        sgd.setLearningRate(0.5);
        assertEquals(0.5, sgd.getCurrentLearningRate(), 1e-10);
    }

    @Test
    void setCurrentParams() {
        RereOnlineSGD sgd = new RereOnlineSGD();
        IVector<Double> params = Linalg.vector(new double[]{5.0, 6.0});
        sgd.initialize(params);

        IVector<Double> newParams = Linalg.vector(new double[]{7.0, 8.0});
        sgd.setCurrentParams(newParams);
        assertEquals(7.0, sgd.getCurrentParams().get(0), 1e-10);
    }

    @Test
    void stateDict() {
        RereOnlineSGD sgd = new RereOnlineSGD(0.1);
        IVector<Double> params = Linalg.vector(new double[]{1.0, 2.0});
        sgd.initialize(params);
        sgd.step(Linalg.vector(new double[]{0.1, 0.1}));

        java.util.Map<String, double[]> state = sgd.optimizerStateDict();
        assertNotNull(state);
        assertFalse(state.isEmpty());
    }

    @Test
    void convergenceOnSimpleFunction() {
        // Minimize f(x) = x^2, gradient = 2x
        RereOnlineSGD sgd = new RereOnlineSGD(0.01);
        IVector<Double> params = Linalg.vector(new double[]{5.0});
        sgd.initialize(params);

        for (int i = 0; i < 500; i++) {
            IVector<Double> p = sgd.getCurrentParams();
            IVector<Double> grad = Linalg.vector(new double[]{2 * p.get(0)});
            sgd.step(grad);
        }

        IVector<Double> finalParams = sgd.getCurrentParams();
        assertEquals(0.0, finalParams.get(0), 0.5);
    }
}
