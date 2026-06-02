package com.yishape.lab.math.optimize.newton;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link RereOnlineAdam}.
 */
class RereOnlineAdamTest {

    @Test
    void defaultConstructor() {
        RereOnlineAdam adam = new RereOnlineAdam();
        assertEquals(0.001, adam.getCurrentLearningRate(), 1e-10);
        assertEquals(0, adam.getCurrentStep());
        assertFalse(adam.isInitialized());
    }

    @Test
    void constructor_withLR() {
        RereOnlineAdam adam = new RereOnlineAdam(0.01);
        assertEquals(0.01, adam.getCurrentLearningRate(), 1e-10);
    }

    @Test
    void constructor_fullParams() {
        RereOnlineAdam adam = new RereOnlineAdam(0.001, 0.9, 0.999);
        assertNotNull(adam);
    }

    @Test
    void initialize() {
        RereOnlineAdam adam = new RereOnlineAdam();
        IVector<Double> params = Linalg.vector(new double[]{1.0, 2.0, 3.0});
        adam.initialize(params);
        assertTrue(adam.isInitialized());
    }

    @Test
    void step_updatesParams() {
        RereOnlineAdam adam = new RereOnlineAdam(0.01);
        IVector<Double> params = Linalg.vector(new double[]{1.0, 2.0});
        adam.initialize(params);

        IVector<Double> grad = Linalg.vector(new double[]{0.5, 0.3});
        IVector<Double> newParams = adam.step(grad);

        assertNotNull(newParams);
        assertEquals(1, adam.getCurrentStep());
    }

    @Test
    void reset() {
        RereOnlineAdam adam = new RereOnlineAdam();
        IVector<Double> params = Linalg.vector(new double[]{1.0, 2.0});
        adam.initialize(params);
        adam.step(Linalg.vector(new double[]{0.1, 0.1}));

        adam.reset();
        assertEquals(0, adam.getCurrentStep());
        assertFalse(adam.isInitialized());
    }

    @Test
    void clone_createsDeepCopy() {
        RereOnlineAdam adam = new RereOnlineAdam();
        IVector<Double> params = Linalg.vector(new double[]{1.0, 2.0});
        adam.initialize(params);
        adam.step(Linalg.vector(new double[]{0.1, 0.1}));

        var clone = adam.clone();
        assertEquals(adam.getCurrentStep(), clone.getCurrentStep());
        assertEquals(adam.getCurrentLearningRate(), clone.getCurrentLearningRate());
    }

    @Test
    void amsgrad() {
        RereOnlineAdam adam = new RereOnlineAdam();
        adam.setAmsgrad(true);
        IVector<Double> params = Linalg.vector(new double[]{1.0, 2.0});
        adam.initialize(params);

        IVector<Double> grad = Linalg.vector(new double[]{0.5, 0.3});
        IVector<Double> result = adam.step(grad);
        assertNotNull(result);
    }

    @Test
    void setBeta1Beta2() {
        RereOnlineAdam adam = new RereOnlineAdam();
        adam.setBeta1(0.8).setBeta2(0.99);
        assertNotNull(adam);
    }

    @Test
    void convergenceOnSimpleFunction() {
        // Minimize f(x) = x^2, gradient = 2x
        RereOnlineAdam adam = new RereOnlineAdam(0.01);
        IVector<Double> params = Linalg.vector(new double[]{5.0});
        adam.initialize(params);

        for (int i = 0; i < 1000; i++) {
            IVector<Double> p = adam.getCurrentParams();
            IVector<Double> grad = Linalg.vector(new double[]{2 * p.get(0)});
            adam.step(grad);
        }

        IVector<Double> finalParams = adam.getCurrentParams();
        assertEquals(0.0, finalParams.get(0), 0.5);
    }

    @Test
    void stateDict() {
        RereOnlineAdam adam = new RereOnlineAdam();
        IVector<Double> params = Linalg.vector(new double[]{1.0, 2.0});
        adam.initialize(params);
        adam.step(Linalg.vector(new double[]{0.1, 0.1}));

        java.util.Map<String, double[]> state = adam.optimizerStateDict();
        assertNotNull(state);
        assertTrue(state.containsKey("step"));
    }

    @Test
    void setters_returnThis() {
        RereOnlineAdam adam = new RereOnlineAdam();
        RereOnlineAdam result = adam.setEpsilon(1e-6)
            .setWeightDecay(0.001)
            .setVerbose(false)
            .setSkipGradientValidation(true)
            .setLrDecayRate(0.01)
            .setLrDecayStep(100);
        assertSame(adam, result);
    }
}
