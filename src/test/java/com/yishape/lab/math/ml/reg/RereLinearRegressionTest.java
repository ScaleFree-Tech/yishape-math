package com.yishape.lab.math.ml.reg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link RereLinearRegression}.
 */
class RereLinearRegressionTest {

    private IMatrix<Double> features;
    private IVector<Double> labels;

    @BeforeEach
    void setUp() {
        // Simple linear relationship: y = 2*x1 + 3*x2 + 1
        features = Linalg.matrix(new double[][]{
            {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6},
            {6, 7}, {7, 8}, {8, 9}, {9, 10}, {10, 11}
        });
        labels = Linalg.vector(new double[]{
            9, 13, 17, 21, 25, 29, 33, 37, 41, 45
        });
    }

    // ==================== Basic Training ====================

    @Test
    void fit_basicLinear() {
        RereLinearRegression model = new RereLinearRegression();
        IRegression result = model.fit(features, labels);
        assertNotNull(result);
        assertTrue(model.isTrained());
    }

    @Test
    void fit_returnsSelf() {
        RereLinearRegression model = new RereLinearRegression();
        IRegression result = model.fit(features, labels);
        assertSame(model, result, "fit() should return this for chaining");
    }

    @Test
    void predict_afterTraining() {
        RereLinearRegression model = new RereLinearRegression();
        model.fit(features, labels);
        double pred = model.predict(Linalg.vector(new double[]{11, 12}));
        // Model should predict close to the linear relationship
        assertTrue(pred > 40, "Prediction should be reasonable for extrapolation");
    }

    @Test
    void predictBatch() {
        RereLinearRegression model = new RereLinearRegression();
        model.fit(features, labels);
        IMatrix<Double> testFeatures = Linalg.matrix(new double[][]{{11, 12}, {12, 13}});
        double[] preds = model.predictBatch(testFeatures);
        assertEquals(2, preds.length);
    }

    @Test
    void fitPredict() {
        RereLinearRegression model = new RereLinearRegression();
        double[] preds = model.fitPredict(features, labels);
        assertEquals(10, preds.length);
    }

    // ==================== Weights & Bias ====================

    @Test
    void getWeights_afterTraining() {
        RereLinearRegression model = new RereLinearRegression();
        model.fit(features, labels);
        IVector<Double> weights = model.getWeights();
        assertNotNull(weights);
        assertTrue(weights.size() > 0);
    }

    @Test
    void getBias_afterTraining() {
        RereLinearRegression model = new RereLinearRegression();
        model.fit(features, labels);
        double bias = model.getBias();
        // Bias should be finite and model should be trained
        assertTrue(Double.isFinite(bias));
    }

    @Test
    void getFeatureWeights() {
        RereLinearRegression model = new RereLinearRegression();
        model.fit(features, labels);
        IVector<Double> fw = model.getFeatureWeights();
        assertNotNull(fw);
        assertEquals(2, fw.size());
    }

    // ==================== No Bias ====================

    @Test
    void fit_noBias() {
        RereLinearRegression model = new RereLinearRegression(false, 0, 0);
        model.fit(features, labels);
        assertTrue(model.isTrained());
        assertEquals(0, model.getBias(), 1e-10);
    }

    // ==================== Regularization ====================

    @Test
    void fit_withL2Regularization() {
        RereLinearRegression model = new RereLinearRegression(
            true, RereLinearRegression.RegularizationType.L2, 0, 0.1);
        model.fit(features, labels);
        assertTrue(model.isTrained());
        assertEquals(RereLinearRegression.RegularizationType.L2, model.getRegularizationType());
    }

    @Test
    void fit_withElasticNet() {
        RereLinearRegression model = new RereLinearRegression(
            true, RereLinearRegression.RegularizationType.ELASTIC_NET, 0.1, 0.1);
        model.fit(features, labels);
        assertTrue(model.isTrained());
    }

    @Test
    void regularizationDescription() {
        RereLinearRegression model = new RereLinearRegression(
            true, RereLinearRegression.RegularizationType.L2, 0, 0.1);
        assertNotNull(model.getRegularizationDescription());
        assertTrue(model.getRegularizationDescription().contains("L2"));
    }

    // ==================== Evaluation ====================

    @Test
    void r2ScoreOn() {
        RereLinearRegression model = new RereLinearRegression();
        model.fit(features, labels);
        double r2 = model.r2ScoreOn(features, labels);
        assertTrue(r2 > 0.9, "R2 should be high for well-fit model");
    }

    @Test
    void rmseOn() {
        RereLinearRegression model = new RereLinearRegression();
        model.fit(features, labels);
        double rmse = model.rmseOn(features, labels);
        assertTrue(rmse < 1.0, "RMSE should be small for well-fit model");
    }

    @Test
    void getResult() {
        RereLinearRegression model = new RereLinearRegression();
        model.fit(features, labels);
        RegressionResult result = model.getResult();
        assertNotNull(result);
    }

    // ==================== Error Conditions ====================

    @Test
    void predict_notTrained_throws() {
        RereLinearRegression model = new RereLinearRegression();
        assertThrows(IllegalStateException.class,
            () -> model.predict(Linalg.vector(new double[]{1, 2})));
    }

    @Test
    void predict_wrongDimension_throws() {
        RereLinearRegression model = new RereLinearRegression();
        model.fit(features, labels);
        assertThrows(IllegalArgumentException.class,
            () -> model.predict(Linalg.vector(new double[]{1})));
    }

    // ==================== Dimension Getters ====================

    @Test
    void getFeatureCount() {
        RereLinearRegression model = new RereLinearRegression();
        model.fit(features, labels);
        assertEquals(2, model.getFeatureCount());
    }

    @Test
    void getSampleCount() {
        RereLinearRegression model = new RereLinearRegression();
        model.fit(features, labels);
        assertEquals(10, model.getSampleCount());
    }

    // ==================== Gradient & Objective ====================

    @Test
    void computeGradient() {
        RereLinearRegression model = new RereLinearRegression();
        model.fit(features, labels);
        IVector<Double> grad = model.computeGradient(model.getWeights());
        assertNotNull(grad);
        assertEquals(model.getWeights().size(), grad.size());
    }

    @Test
    void computeObjective() {
        RereLinearRegression model = new RereLinearRegression();
        model.fit(features, labels);
        double obj = model.computeObjective(model.getWeights());
        assertTrue(obj >= 0, "Objective should be non-negative");
    }

    // ==================== Serialization ====================

    @Test
    void toParams_fromParams() {
        RereLinearRegression model = new RereLinearRegression(
            true, RereLinearRegression.RegularizationType.L2, 0.1, 0.05);
        model.fit(features, labels);

        java.util.Map<String, Object> params = model.toParams();
        assertNotNull(params);

        RereLinearRegression model2 = new RereLinearRegression();
        model2.fromParams(params);
        assertEquals(model.getLambda1(), model2.getLambda1(), EPS);
    }

    private static final double EPS = 1e-10;
}
