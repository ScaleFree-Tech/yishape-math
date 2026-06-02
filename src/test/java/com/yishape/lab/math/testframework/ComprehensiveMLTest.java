package com.yishape.lab.math.testframework;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.ML;
import com.yishape.lab.math.ml.clf.IClassifier;
import com.yishape.lab.math.ml.preprocessing.ITransform;
import com.yishape.lab.math.ml.reg.IRegression;
import com.yishape.lab.math.ml.reg.RegressionResult;
import com.yishape.lab.math.ml.metric.ClassificationMetrics;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive correctness tests for com.yishape.lab.math.ml package.
 *
 * <p>Tests all factory methods in {@link ML} using small, controlled datasets
 * to ensure each algorithm behaves correctly on well-understood problems.
 * All tests run in &lt; 30 seconds total.</p>
 *
 * <p>Test coverage:</p>
 * <ul>
 *   <li><b>Classifiers</b>: logistic regression, linear SVM, k-NN, decision tree,
 *       random forest, XGBoost</li>
 *   <li><b>Regressors</b>: linear regression (OLS, L1, L2, ElasticNet)</li>
 *   <li><b>Dimensionality reduction</b>: PCA, SVD</li>
 *   <li><b>Metrics</b>: ClassificationMetrics accuracy via {@link ML#classificationMetrics}</li>
 * </ul>
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class ComprehensiveMLTest {

    private static final double ACCURACY_THRESHOLD = 0.85;
    private static final double REGRESSION_R2_THRESHOLD = 0.90;
    private static final Random RANDOM = new Random(42);

    private TestResult.Recorder recorder;

    // ========== Shared test datasets ==========

    /** Linearly separable binary data for LR / SVM. */
    private IMatrix<Double> linearBinaryFeatures;
    private String[] linearBinaryLabels;

    /** XOR-like non-linear data for tree-based methods. */
    private IMatrix<Double> xorFeatures;
    private String[] xorLabels;

    /** Simple three-class data for multi-class tests. */
    private IMatrix<Double> multiClassFeatures;
    private String[] multiClassLabels;

    /** Regression data: y = 2*x + 1 + small noise. */
    private IMatrix<Double> regressionFeatures;
    private IVector<Double> regressionLabels;

    @BeforeAll
    void setUp() {
        recorder = new TestResult.Recorder("ml", "test_docs/results");
        buildDatasets();
    }

    private void buildDatasets() {
        // ---- Linear binary classification (2 features) ----
        // Class A: around (1,1), Class B: around (-1,-1)
        double[][] linearData = new double[20][2];
        String[] linearLbl = new String[20];
        for (int i = 0; i < 10; i++) {
            linearData[i][0] = 1.0 + RANDOM.nextGaussian() * 0.3;
            linearData[i][1] = 1.0 + RANDOM.nextGaussian() * 0.3;
            linearLbl[i] = "A";
        }
        for (int i = 10; i < 20; i++) {
            linearData[i][0] = -1.0 + RANDOM.nextGaussian() * 0.3;
            linearData[i][1] = -1.0 + RANDOM.nextGaussian() * 0.3;
            linearLbl[i] = "B";
        }
        linearBinaryFeatures = Linalg.matrix(linearData);
        linearBinaryLabels = linearLbl;

        // ---- XOR pattern (decision tree / ensemble) ----
        // Four quadrants: (+,+)=A, (+,-)=B, (-,+)=B, (-,-)=A
        // Generate more samples so trees can split properly
        int xorPerQuad = 8;
        double[][] xorData = new double[xorPerQuad * 4][2];
        String[] xorLbl = new String[xorPerQuad * 4];
        int idx = 0;
        // Quadrant 1: (+,+)=A
        for (int i = 0; i < xorPerQuad; i++) {
            xorData[idx][0] = 2.0 + RANDOM.nextGaussian() * 0.3;
            xorData[idx][1] = 2.0 + RANDOM.nextGaussian() * 0.3;
            xorLbl[idx] = "A";
            idx++;
        }
        // Quadrant 2: (-,-)=A
        for (int i = 0; i < xorPerQuad; i++) {
            xorData[idx][0] = -2.0 + RANDOM.nextGaussian() * 0.3;
            xorData[idx][1] = -2.0 + RANDOM.nextGaussian() * 0.3;
            xorLbl[idx] = "A";
            idx++;
        }
        // Quadrant 3: (+,-)=B
        for (int i = 0; i < xorPerQuad; i++) {
            xorData[idx][0] = 2.0 + RANDOM.nextGaussian() * 0.3;
            xorData[idx][1] = -2.0 + RANDOM.nextGaussian() * 0.3;
            xorLbl[idx] = "B";
            idx++;
        }
        // Quadrant 4: (-,+)=B
        for (int i = 0; i < xorPerQuad; i++) {
            xorData[idx][0] = -2.0 + RANDOM.nextGaussian() * 0.3;
            xorData[idx][1] = 2.0 + RANDOM.nextGaussian() * 0.3;
            xorLbl[idx] = "B";
            idx++;
        }
        xorFeatures = Linalg.matrix(xorData);
        xorLabels = xorLbl;

        // ---- Three-class data (for k-NN and multi-class) ----
        double[][] mcData = new double[30][2];
        String[] mcLbl = new String[30];
        // Class X around (3,3)
        for (int i = 0; i < 10; i++) {
            mcData[i][0] = 3.0 + RANDOM.nextGaussian() * 0.4;
            mcData[i][1] = 3.0 + RANDOM.nextGaussian() * 0.4;
            mcLbl[i] = "X";
        }
        // Class Y around (0,0)
        for (int i = 10; i < 20; i++) {
            mcData[i][0] = RANDOM.nextGaussian() * 0.4;
            mcData[i][1] = RANDOM.nextGaussian() * 0.4;
            mcLbl[i] = "Y";
        }
        // Class Z around (-3,3)
        for (int i = 20; i < 30; i++) {
            mcData[i][0] = -3.0 + RANDOM.nextGaussian() * 0.4;
            mcData[i][1] = 3.0 + RANDOM.nextGaussian() * 0.4;
            mcLbl[i] = "Z";
        }
        multiClassFeatures = Linalg.matrix(mcData);
        multiClassLabels = mcLbl;

        // ---- Regression: y = 2*x + 1 + noise ----
        double[][] regData = new double[30][1];
        double[] regLbl = new double[30];
        for (int i = 0; i < 30; i++) {
            double x = i * 0.5 - 7.5;  // x from -7.5 to 7.0
            regData[i][0] = x;
            regLbl[i] = 2.0 * x + 1.0 + RANDOM.nextGaussian() * 0.5;
        }
        regressionFeatures = Linalg.matrix(regData);
        regressionLabels = Linalg.vector(regLbl);
    }

    // ============================================================
    //  Classification Tests
    // ============================================================

    @Test
    void testLogisticRegressionDefault() {
        TestResult r = recorder.record("classifier", "logistic_default");
        long t0 = System.currentTimeMillis();
        try {
            IClassifier clf = ML.clf.logisticRegression();
            clf.fit(linearBinaryFeatures, linearBinaryLabels);

            String[] preds = clf.predictBatch(linearBinaryFeatures);
            double acc = computeAccuracy(linearBinaryLabels, preds);

            long dt = System.currentTimeMillis() - t0;
            r.timeMs = dt;

            if (acc >= ACCURACY_THRESHOLD) {
                r.pass("accuracy=" + String.format("%.3f", acc));
            } else {
                r.fail("accuracy too low: " + String.format("%.3f", acc));
            }
            assertTrue(acc >= ACCURACY_THRESHOLD,
                "Logistic regression default accuracy should be >= " + ACCURACY_THRESHOLD);
        } catch (Exception e) {
            r.fail(e.getMessage());
            fail("Logistic regression default failed: " + e.getMessage());
        }
    }

    @Test
    void testLogisticRegressionRegularized() {
        TestResult r = recorder.record("classifier", "logistic_l1l2");
        long t0 = System.currentTimeMillis();
        try {
            IClassifier clf = ML.clf.logisticRegression(0.01, 0.01);
            clf.fit(linearBinaryFeatures, linearBinaryLabels);

            String[] preds = clf.predictBatch(linearBinaryFeatures);
            double acc = computeAccuracy(linearBinaryLabels, preds);

            long dt = System.currentTimeMillis() - t0;
            r.timeMs = dt;

            if (acc >= ACCURACY_THRESHOLD) {
                r.pass("accuracy=" + String.format("%.3f", acc));
            } else {
                r.fail("accuracy too low: " + String.format("%.3f", acc));
            }
            assertTrue(acc >= ACCURACY_THRESHOLD,
                "Regularized logistic regression accuracy should be >= " + ACCURACY_THRESHOLD);
        } catch (Exception e) {
            r.fail(e.getMessage());
            fail("Logistic regression regularized failed: " + e.getMessage());
        }
    }

    @Test
    void testLinearSvmDefault() {
        TestResult r = recorder.record("classifier", "svm_default");
        long t0 = System.currentTimeMillis();
        try {
            IClassifier clf = ML.clf.linearSvm();
            clf.fit(linearBinaryFeatures, linearBinaryLabels);

            String[] preds = clf.predictBatch(linearBinaryFeatures);
            double acc = computeAccuracy(linearBinaryLabels, preds);

            long dt = System.currentTimeMillis() - t0;
            r.timeMs = dt;

            if (acc >= ACCURACY_THRESHOLD) {
                r.pass("accuracy=" + String.format("%.3f", acc));
            } else {
                r.fail("accuracy too low: " + String.format("%.3f", acc));
            }
            assertTrue(acc >= ACCURACY_THRESHOLD,
                "Linear SVM default accuracy should be >= " + ACCURACY_THRESHOLD);
        } catch (Exception e) {
            r.fail(e.getMessage());
            fail("Linear SVM default failed: " + e.getMessage());
        }
    }

    @Test
    void testLinearSvmParameterized() {
        TestResult r = recorder.record("classifier", "svm_param");
        long t0 = System.currentTimeMillis();
        try {
            IClassifier clf = ML.clf.linearSvm(1.0, true);
            clf.fit(linearBinaryFeatures, linearBinaryLabels);

            String[] preds = clf.predictBatch(linearBinaryFeatures);
            double acc = computeAccuracy(linearBinaryLabels, preds);

            long dt = System.currentTimeMillis() - t0;
            r.timeMs = dt;

            if (acc >= ACCURACY_THRESHOLD) {
                r.pass("accuracy=" + String.format("%.3f", acc));
            } else {
                r.fail("accuracy too low: " + String.format("%.3f", acc));
            }
            assertTrue(acc >= ACCURACY_THRESHOLD,
                "Linear SVM parameterized accuracy should be >= " + ACCURACY_THRESHOLD);
        } catch (Exception e) {
            r.fail(e.getMessage());
            fail("Linear SVM parameterized failed: " + e.getMessage());
        }
    }

    @Test
    void testKnnBinary() {
        TestResult r = recorder.record("classifier", "knn_binary");
        long t0 = System.currentTimeMillis();
        try {
            IClassifier clf = ML.clf.kNN(3);
            clf.fit(linearBinaryFeatures, linearBinaryLabels);

            String[] preds = clf.predictBatch(linearBinaryFeatures);
            double acc = computeAccuracy(linearBinaryLabels, preds);

            long dt = System.currentTimeMillis() - t0;
            r.timeMs = dt;

            if (acc >= ACCURACY_THRESHOLD) {
                r.pass("accuracy=" + String.format("%.3f", acc));
            } else {
                r.fail("accuracy too low: " + String.format("%.3f", acc));
            }
            assertTrue(acc >= ACCURACY_THRESHOLD,
                "k-NN binary accuracy should be >= " + ACCURACY_THRESHOLD);
        } catch (Exception e) {
            r.fail(e.getMessage());
            fail("k-NN binary failed: " + e.getMessage());
        }
    }

    @Test
    void testKnnMulticlass() {
        TestResult r = recorder.record("classifier", "knn_multiclass");
        long t0 = System.currentTimeMillis();
        try {
            IClassifier clf = ML.clf.kNN(3);
            clf.fit(multiClassFeatures, multiClassLabels);

            String[] preds = clf.predictBatch(multiClassFeatures);
            double acc = computeAccuracy(multiClassLabels, preds);

            long dt = System.currentTimeMillis() - t0;
            r.timeMs = dt;

            if (acc >= ACCURACY_THRESHOLD) {
                r.pass("accuracy=" + String.format("%.3f", acc));
            } else {
                r.fail("accuracy too low: " + String.format("%.3f", acc));
            }
            assertTrue(acc >= ACCURACY_THRESHOLD,
                "k-NN multiclass accuracy should be >= " + ACCURACY_THRESHOLD);
        } catch (Exception e) {
            r.fail(e.getMessage());
            fail("k-NN multiclass failed: " + e.getMessage());
        }
    }

    @Test
    void testDecisionTree() {
        TestResult r = recorder.record("classifier", "decision_tree");
        long t0 = System.currentTimeMillis();
        try {
            IClassifier clf = ML.clf.decisionTree();
            clf.fit(xorFeatures, xorLabels);

            String[] preds = clf.predictBatch(xorFeatures);
            double acc = computeAccuracy(xorLabels, preds);

            long dt = System.currentTimeMillis() - t0;
            r.timeMs = dt;

            // Decision tree should perfectly fit XOR with enough depth
            if (acc >= 0.95) {
                r.pass("accuracy=" + String.format("%.3f", acc));
            } else {
                r.fail("accuracy too low: " + String.format("%.3f", acc));
            }
            assertTrue(acc >= 0.95,
                "Decision tree should fit XOR-like data with >= 0.95 accuracy");
        } catch (Exception e) {
            r.fail(e.getMessage());
            fail("Decision tree failed: " + e.getMessage());
        }
    }

    @Test
    void testRandomForest() {
        TestResult r = recorder.record("classifier", "random_forest");
        long t0 = System.currentTimeMillis();
        try {
            IClassifier clf = ML.clf.randomForest();
            clf.fit(xorFeatures, xorLabels);

            String[] preds = clf.predictBatch(xorFeatures);
            double acc = computeAccuracy(xorLabels, preds);

            long dt = System.currentTimeMillis() - t0;
            r.timeMs = dt;

            if (acc >= 0.95) {
                r.pass("accuracy=" + String.format("%.3f", acc));
            } else {
                r.fail("accuracy too low: " + String.format("%.3f", acc));
            }
            assertTrue(acc >= 0.95,
                "Random forest should fit XOR-like data with >= 0.95 accuracy");
        } catch (Exception e) {
            r.fail(e.getMessage());
            fail("Random forest failed: " + e.getMessage());
        }
    }

    @Test
    void testXGboost() {
        TestResult r = recorder.record("classifier", "xgboost");
        long t0 = System.currentTimeMillis();
        try {
            IClassifier clf = ML.clf.xGboost();
            clf.fit(xorFeatures, xorLabels);

            String[] preds = clf.predictBatch(xorFeatures);
            double acc = computeAccuracy(xorLabels, preds);

            long dt = System.currentTimeMillis() - t0;
            r.timeMs = dt;

            if (acc >= 0.80) {
                r.pass("accuracy=" + String.format("%.3f", acc));
            } else {
                r.fail("accuracy too low: " + String.format("%.3f", acc));
            }
            assertTrue(acc >= 0.80,
                "XGBoost should fit XOR-like data with >= 0.80 accuracy");
        } catch (Exception e) {
            r.fail(e.getMessage());
            fail("XGBoost failed: " + e.getMessage());
        }
    }

    // ============================================================
    //  Regression Tests
    // ============================================================

    @Test
    void testLinearRegressionOls() {
        TestResult r = recorder.record("regression", "linear_ols");
        long t0 = System.currentTimeMillis();
        try {
            IRegression reg = ML.reg.linear();
            reg.fit(regressionFeatures, regressionLabels);
            RegressionResult result = reg.getResult();

            double r2 = result.getR2Score();
            long dt = System.currentTimeMillis() - t0;
            r.timeMs = dt;

            if (r2 >= REGRESSION_R2_THRESHOLD) {
                r.pass("R2=" + String.format("%.3f", r2));
            } else {
                r.fail("R2 too low: " + String.format("%.3f", r2));
            }
            assertTrue(r2 >= REGRESSION_R2_THRESHOLD,
                "Linear regression OLS R2 should be >= " + REGRESSION_R2_THRESHOLD);
        } catch (Exception e) {
            r.fail(e.getMessage());
            fail("Linear regression OLS failed: " + e.getMessage());
        }
    }

    @Test
    void testLinearRegressionL1() {
        TestResult r = recorder.record("regression", "linear_l1");
        long t0 = System.currentTimeMillis();
        try {
            IRegression reg = ML.reg.linear(0.1, 0.0);
            reg.fit(regressionFeatures, regressionLabels);
            RegressionResult result = reg.getResult();

            double r2 = result.getR2Score();
            long dt = System.currentTimeMillis() - t0;
            r.timeMs = dt;

            if (r2 >= REGRESSION_R2_THRESHOLD) {
                r.pass("R2=" + String.format("%.3f", r2));
            } else {
                r.fail("R2 too low: " + String.format("%.3f", r2));
            }
            assertTrue(r2 >= REGRESSION_R2_THRESHOLD,
                "Linear regression L1 R2 should be >= " + REGRESSION_R2_THRESHOLD);
        } catch (Exception e) {
            r.fail(e.getMessage());
            fail("Linear regression L1 failed: " + e.getMessage());
        }
    }

    @Test
    void testLinearRegressionL2() {
        TestResult r = recorder.record("regression", "linear_l2");
        long t0 = System.currentTimeMillis();
        try {
            IRegression reg = ML.reg.linear(0.0, 0.1);
            reg.fit(regressionFeatures, regressionLabels);
            RegressionResult result = reg.getResult();

            double r2 = result.getR2Score();
            long dt = System.currentTimeMillis() - t0;
            r.timeMs = dt;

            if (r2 >= REGRESSION_R2_THRESHOLD) {
                r.pass("R2=" + String.format("%.3f", r2));
            } else {
                r.fail("R2 too low: " + String.format("%.3f", r2));
            }
            assertTrue(r2 >= REGRESSION_R2_THRESHOLD,
                "Linear regression L2 R2 should be >= " + REGRESSION_R2_THRESHOLD);
        } catch (Exception e) {
            r.fail(e.getMessage());
            fail("Linear regression L2 failed: " + e.getMessage());
        }
    }

    @Test
    void testLinearRegressionElasticNet() {
        TestResult r = recorder.record("regression", "linear_elasticnet");
        long t0 = System.currentTimeMillis();
        try {
            IRegression reg = ML.reg.linear(0.05, 0.05);
            reg.fit(regressionFeatures, regressionLabels);
            RegressionResult result = reg.getResult();

            double r2 = result.getR2Score();
            long dt = System.currentTimeMillis() - t0;
            r.timeMs = dt;

            if (r2 >= REGRESSION_R2_THRESHOLD) {
                r.pass("R2=" + String.format("%.3f", r2));
            } else {
                r.fail("R2 too low: " + String.format("%.3f", r2));
            }
            assertTrue(r2 >= REGRESSION_R2_THRESHOLD,
                "Linear regression ElasticNet R2 should be >= " + REGRESSION_R2_THRESHOLD);
        } catch (Exception e) {
            r.fail(e.getMessage());
            fail("Linear regression ElasticNet failed: " + e.getMessage());
        }
    }

    // ============================================================
    //  Dimensionality Reduction Tests
    // ============================================================

    @Test
    void testPcaDimReducer() {
        TestResult r = recorder.record("dimreduce", "pca");
        long t0 = System.currentTimeMillis();
        try {
            ITransform<Double> reducer = ML.dr.pca(1);

            // Use multi-class data (2D -> 1D)
            var reduced = reducer.fitTransform(multiClassFeatures);

            long dt = System.currentTimeMillis() - t0;
            r.timeMs = dt;

            assertNotNull(reduced, "PCA reduced matrix should not be null");
            assertEquals(multiClassFeatures.getRowNum(), reduced.getRowNum(),
                "PCA should preserve row count");
            assertEquals(1, reduced.getColNum(),
                "PCA should reduce to requested dimension");

            r.pass("shape=" + reduced.getRowNum() + "x" + reduced.getColNum());
        } catch (Exception e) {
            r.fail(e.getMessage());
            fail("PCA dimension reduction failed: " + e.getMessage());
        }
    }

    @Test
    void testSvdDimReducer() {
        TestResult r = recorder.record("dimreduce", "svd");
        long t0 = System.currentTimeMillis();
        try {
            ITransform<Double> reducer = ML.dr.svd(1);

            // Use multi-class data (2D -> 1D)
            var reduced = reducer.fitTransform(multiClassFeatures);

            long dt = System.currentTimeMillis() - t0;
            r.timeMs = dt;

            assertNotNull(reduced, "SVD reduced matrix should not be null");
            assertEquals(multiClassFeatures.getRowNum(), reduced.getRowNum(),
                "SVD should preserve row count");
            assertEquals(1, reduced.getColNum(),
                "SVD should reduce to requested dimension");

            r.pass("shape=" + reduced.getRowNum() + "x" + reduced.getColNum());
        } catch (Exception e) {
            r.fail(e.getMessage());
            fail("SVD dimension reduction failed: " + e.getMessage());
        }
    }

    // ============================================================
    //  ClassificationMetrics Test
    // ============================================================

    @Test
    void testClassificationMetrics() {
        TestResult r = recorder.record("metrics", "classification_metrics");
        long t0 = System.currentTimeMillis();
        try {
            IClassifier clf = ML.clf.logisticRegression();
            clf.fit(linearBinaryFeatures, linearBinaryLabels);

            ClassificationMetrics metrics = ML.clf.classificationMetrics(clf, linearBinaryFeatures, linearBinaryLabels);

            long dt = System.currentTimeMillis() - t0;
            r.timeMs = dt;

            assertNotNull(metrics, "ClassificationMetrics should not be null");
            assertTrue(metrics.getAccuracy() >= 0.0 && metrics.getAccuracy() <= 1.0,
                "Accuracy should be in [0,1]");
            assertTrue(metrics.getMacroF1() >= 0.0 && metrics.getMacroF1() <= 1.0,
                "Macro F1 should be in [0,1]");
            assertNotNull(metrics.getConfusionMatrix(), "Confusion matrix should not be null");
            assertNotNull(metrics.getClassificationReport(), "Classification report should not be null");

            r.pass("accuracy=" + String.format("%.3f", metrics.getAccuracy()));
        } catch (Exception e) {
            r.fail(e.getMessage());
            fail("ClassificationMetrics computation failed: " + e.getMessage());
        }
    }

    // ============================================================
    //  Interface-level behaviour tests
    // ============================================================

    @Test
    void testClassifierPredictProb() {
        TestResult r = recorder.record("classifier", "predict_prob");
        long t0 = System.currentTimeMillis();
        try {
            IClassifier clf = ML.clf.logisticRegression();
            clf.fit(linearBinaryFeatures, linearBinaryLabels);

            IVector<Double> sample = Linalg.vector(new double[]{1.0, 1.0});
            Map<String, Double> probs = clf.predictProb(sample);

            long dt = System.currentTimeMillis() - t0;
            r.timeMs = dt;

            assertNotNull(probs, "predictProb should return non-null map");
            assertFalse(probs.isEmpty(), "predictProb should return non-empty map");

            double sum = probs.values().stream().mapToDouble(Double::doubleValue).sum();
            assertTrue(Math.abs(sum - 1.0) < 0.01,
                "Probabilities should sum to approximately 1.0, got " + sum);

            r.pass("probs=" + probs);
        } catch (Exception e) {
            r.fail(e.getMessage());
            fail("predictProb test failed: " + e.getMessage());
        }
    }

    @Test
    void testClassifierPredictSingle() {
        TestResult r = recorder.record("classifier", "predict_single");
        long t0 = System.currentTimeMillis();
        try {
            IClassifier clf = ML.clf.logisticRegression();
            clf.fit(linearBinaryFeatures, linearBinaryLabels);

            IVector<Double> sample = Linalg.vector(new double[]{1.0, 1.0});
            String pred = clf.predict(sample);

            long dt = System.currentTimeMillis() - t0;
            r.timeMs = dt;

            assertNotNull(pred, "predict should return non-null label");
            assertTrue(pred.equals("A") || pred.equals("B"),
                "Prediction should be one of the trained labels");

            r.pass("pred=" + pred);
        } catch (Exception e) {
            r.fail(e.getMessage());
            fail("predict single test failed: " + e.getMessage());
        }
    }

    @Test
    void testRegressionPredict() {
        TestResult r = recorder.record("regression", "predict_single");
        long t0 = System.currentTimeMillis();
        try {
            IRegression reg = ML.reg.linear();
            reg.fit(regressionFeatures, regressionLabels);

            // Predict at x = 0, expected around y = 1
            IVector<Double> sample = Linalg.vector(new double[]{0.0});
            double pred = reg.predict(sample);

            long dt = System.currentTimeMillis() - t0;
            r.timeMs = dt;

            // y = 2*0 + 1 = 1, allow some tolerance
            assertTrue(Math.abs(pred - 1.0) < 2.0,
                "Prediction at x=0 should be near 1.0, got " + pred);

            r.pass("pred=" + String.format("%.3f", pred));
        } catch (Exception e) {
            r.fail(e.getMessage());
            fail("Regression predict test failed: " + e.getMessage());
        }
    }

    // ============================================================
    //  Cross-validation sanity check
    // ============================================================

    @Test
    void testKFoldCrossValidation() {
        TestResult r = recorder.record("cv", "kfold");
        long t0 = System.currentTimeMillis();
        try {
            IClassifier clf = ML.clf.logisticRegression();
            var cvResult = ML.clf.kFoldCrossValidation(clf, linearBinaryFeatures, linearBinaryLabels, 5);

            long dt = System.currentTimeMillis() - t0;
            r.timeMs = dt;

            assertNotNull(cvResult, "Cross-validation result should not be null");
            assertTrue(cvResult.getMeanAccuracy() >= 0.0,
                "Mean accuracy should be non-negative");

            r.pass("avg_acc=" + String.format("%.3f", cvResult.getMeanAccuracy()));
        } catch (Exception e) {
            r.fail(e.getMessage());
            fail("k-fold cross-validation failed: " + e.getMessage());
        }
    }

    // ============================================================
    //  Helper methods
    // ============================================================

    private static double computeAccuracy(String[] trueLabels, String[] predLabels) {
        if (trueLabels == null || predLabels == null || trueLabels.length != predLabels.length) {
            return 0.0;
        }
        int correct = 0;
        for (int i = 0; i < trueLabels.length; i++) {
            if (trueLabels[i].equals(predLabels[i])) {
                correct++;
            }
        }
        return (double) correct / trueLabels.length;
    }

    @AfterAll
    void tearDown() {
        if (recorder != null) {
            recorder.writeToFile();
        }
    }
}
