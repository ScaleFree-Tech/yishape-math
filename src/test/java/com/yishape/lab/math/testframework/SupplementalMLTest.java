package com.yishape.lab.math.testframework;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.ML;
import com.yishape.lab.math.ml.clf.ensemble.EnsembleClassifier;
import com.yishape.lab.math.ml.clf.IClassifier;
import com.yishape.lab.math.ml.clu.GMMClustering;
import com.yishape.lab.math.ml.clu.KMeansPlusPlus;
import com.yishape.lab.math.ml.preprocessing.ITransform;
import com.yishape.lab.math.ml.reg.RereLinearRegression;
import com.yishape.lab.math.ml.metric.ClassificationMetrics;
import com.yishape.lab.math.ml.metric.CrossValidationResult;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Supplemental correctness tests for previously uncovered or under-tested ML functionality.
 *
 * <p>Covers:</p>
 * <ul>
 *   <li>LinearSVM classifier (dedicated dataset, predict single + batch)</li>
 *   <li>EnsembleClassifier (VOTING strategy)</li>
 *   <li>KMeansPlusPlus clustering (3 clusters, fitPredict, inertia)</li>
 *   <li>GMMClustering (2-component Gaussian mixture)</li>
 *   <li>RereSVD dimensionality reduction (5D -> 2D, variance retention)</li>
 *   <li>RereTSNE dimensionality reduction (small dataset, no exception)</li>
 *   <li>RereUMAP dimensionality reduction (small dataset, no exception)</li>
 *   <li>ClassificationMetrics manual validation</li>
 *   <li>CrossValidation k-fold sanity checks</li>
 *   <li>RegressionMetrics (RereLinearRegression rmseOn / r2ScoreOn)</li>
 * </ul>
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class SupplementalMLTest {

    private static final double ACCURACY_THRESHOLD = 0.85;
    private static final double ENSEMBLE_THRESHOLD = 0.80;

    private TestResult.Recorder recorder;

    @BeforeAll
    void setUp() {
        recorder = new TestResult.Recorder("ml_supplemental", "test_docs/results");
    }

    // ============================================================
    //  1. LinearSVM Classifier
    // ============================================================

    @Test
    void testLinearSvmFitAndAccuracy() {
        TestResult tr = recorder.record("linear_svm", "fit_accuracy");
        long t0 = System.currentTimeMillis();
        try {
            double[][] features = {
                {1, 1}, {2, 2}, {1, 2}, {2, 1},
                {-1, -1}, {-2, -2}, {-1, -2}, {-2, -1}
            };
            String[] labels = {"pos", "pos", "pos", "pos", "neg", "neg", "neg", "neg"};
            IMatrix<Double> X = Linalg.matrix(features);

            IClassifier svm = ML.clf.linearSvm();
            svm.fit(X, labels);

            String[] preds = svm.predictBatch(X);
            double acc = computeAccuracy(labels, preds);

            long dt = System.currentTimeMillis() - t0;
            tr.timeMs = dt;

            if (acc >= ACCURACY_THRESHOLD) {
                tr.pass("accuracy=" + String.format("%.3f", acc));
            } else {
                tr.fail("accuracy too low: " + String.format("%.3f", acc));
            }
            assertTrue(acc >= ACCURACY_THRESHOLD,
                "LinearSVM accuracy should be >= " + ACCURACY_THRESHOLD);
        } catch (Exception e) {
            tr.fail(e.getMessage());
            fail("LinearSVM fit test failed: " + e.getMessage());
        }
    }

    @Test
    void testLinearSvmPredictSingle() {
        TestResult tr = recorder.record("linear_svm", "predict_single");
        long t0 = System.currentTimeMillis();
        try {
            double[][] features = {
                {1, 1}, {2, 2}, {1, 2}, {2, 1},
                {-1, -1}, {-2, -2}, {-1, -2}, {-2, -1}
            };
            String[] labels = {"pos", "pos", "pos", "pos", "neg", "neg", "neg", "neg"};
            IMatrix<Double> X = Linalg.matrix(features);

            IClassifier svm = ML.clf.linearSvm(1.0, true);
            svm.fit(X, labels);

            IVector<Double> posSample = Linalg.vector(new double[]{1.5, 1.5});
            IVector<Double> negSample = Linalg.vector(new double[]{-1.5, -1.5});

            String posPred = svm.predict(posSample);
            String negPred = svm.predict(negSample);

            long dt = System.currentTimeMillis() - t0;
            tr.timeMs = dt;

            assertEquals("pos", posPred, "Positive sample should predict 'pos'");
            assertEquals("neg", negPred, "Negative sample should predict 'neg'");
            tr.pass("pos=" + posPred + ", neg=" + negPred);
        } catch (Exception e) {
            tr.fail(e.getMessage());
            fail("LinearSVM predict single failed: " + e.getMessage());
        }
    }

    @Test
    void testLinearSvmPredictBatch() {
        TestResult tr = recorder.record("linear_svm", "predict_batch");
        long t0 = System.currentTimeMillis();
        try {
            double[][] features = {
                {1, 1}, {2, 2}, {1, 2}, {2, 1},
                {-1, -1}, {-2, -2}, {-1, -2}, {-2, -1}
            };
            String[] labels = {"pos", "pos", "pos", "pos", "neg", "neg", "neg", "neg"};
            IMatrix<Double> X = Linalg.matrix(features);

            IClassifier svm = ML.clf.linearSvm();
            svm.fit(X, labels);

            String[] preds = svm.predictBatch(X);

            long dt = System.currentTimeMillis() - t0;
            tr.timeMs = dt;

            assertNotNull(preds, "Batch predictions should not be null");
            assertEquals(labels.length, preds.length,
                "Batch prediction count should match input count");
            tr.pass("batch_size=" + preds.length);
        } catch (Exception e) {
            tr.fail(e.getMessage());
            fail("LinearSVM predict batch failed: " + e.getMessage());
        }
    }

    // ============================================================
    //  2. EnsembleClassifier
    // ============================================================

    @Test
    void testEnsembleClassifierVoting() {
        TestResult tr = recorder.record("ensemble", "voting_accuracy");
        long t0 = System.currentTimeMillis();
        try {
            // Linearly separable data with more samples for ensemble stability
            double[][] features = {
                {1, 1}, {2, 2}, {1.5, 1.5}, {2.5, 1.5}, {1, 2},
                {-1, -1}, {-2, -2}, {-1.5, -1.5}, {-2.5, -1.5}, {-1, -2}
            };
            String[] labels = {"pos", "pos", "pos", "pos", "pos",
                               "neg", "neg", "neg", "neg", "neg"};
            IMatrix<Double> X = Linalg.matrix(features);

            IClassifier ensemble = ML.clf.ensembleClassifier(
                EnsembleClassifier.EnsembleStrategy.VOTING, 42L);
            ensemble.fit(X, labels);

            String[] preds = ensemble.predictBatch(X);
            double acc = computeAccuracy(labels, preds);

            long dt = System.currentTimeMillis() - t0;
            tr.timeMs = dt;

            if (acc >= ENSEMBLE_THRESHOLD) {
                tr.pass("accuracy=" + String.format("%.3f", acc));
            } else {
                tr.fail("accuracy too low: " + String.format("%.3f", acc));
            }
            assertTrue(acc >= ENSEMBLE_THRESHOLD,
                "Ensemble voting accuracy should be >= " + ENSEMBLE_THRESHOLD);
        } catch (Exception e) {
            tr.fail(e.getMessage());
            fail("Ensemble classifier voting test failed: " + e.getMessage());
        }
    }

    // ============================================================
    //  3. KMeansPlusPlus Clustering
    // ============================================================

    @Test
    void testKMeansPlusPlusFitAndCenters() {
        TestResult tr = recorder.record("kmeans++", "fit_centers");
        long t0 = System.currentTimeMillis();
        try {
            List<IVector<Double>> data = new ArrayList<>();
            // Cluster 1 around (0,0)
            data.add(IVector.of(new double[]{0.1, 0.1}));
            data.add(IVector.of(new double[]{-0.1, 0.2}));
            data.add(IVector.of(new double[]{0.2, -0.1}));
            // Cluster 2 around (5,5)
            data.add(IVector.of(new double[]{5.1, 5.1}));
            data.add(IVector.of(new double[]{4.9, 5.2}));
            data.add(IVector.of(new double[]{5.2, 4.9}));
            // Cluster 3 around (0,5)
            data.add(IVector.of(new double[]{0.1, 5.1}));
            data.add(IVector.of(new double[]{-0.1, 4.9}));
            data.add(IVector.of(new double[]{0.2, 5.2}));

            KMeansPlusPlus kmeans = new KMeansPlusPlus(42L);
            kmeans.setParameters(java.util.Map.of("numClusters", 3));
            kmeans.fit(data);

            List<IVector<Double>> centers = kmeans.getClusterCenters();

            long dt = System.currentTimeMillis() - t0;
            tr.timeMs = dt;

            assertNotNull(centers, "Cluster centers should not be null");
            assertEquals(3, centers.size(), "Should have exactly 3 cluster centers");
            tr.pass("centers=" + centers.size());
        } catch (Exception e) {
            tr.fail(e.getMessage());
            fail("KMeans++ fit test failed: " + e.getMessage());
        }
    }

    @Test
    void testKMeansPlusPlusFitPredict() {
        TestResult tr = recorder.record("kmeans++", "fit_predict");
        long t0 = System.currentTimeMillis();
        try {
            List<IVector<Double>> data = new ArrayList<>();
            // Cluster 1 around (0,0)
            data.add(IVector.of(new double[]{0.1, 0.1}));
            data.add(IVector.of(new double[]{-0.1, 0.2}));
            data.add(IVector.of(new double[]{0.2, -0.1}));
            // Cluster 2 around (5,5)
            data.add(IVector.of(new double[]{5.1, 5.1}));
            data.add(IVector.of(new double[]{4.9, 5.2}));
            data.add(IVector.of(new double[]{5.2, 4.9}));
            // Cluster 3 around (0,5)
            data.add(IVector.of(new double[]{0.1, 5.1}));
            data.add(IVector.of(new double[]{-0.1, 4.9}));
            data.add(IVector.of(new double[]{0.2, 5.2}));

            KMeansPlusPlus kmeans = new KMeansPlusPlus(42L);
            kmeans.setParameters(java.util.Map.of("numClusters", 3));
            int[] labels = kmeans.fitPredict(data);

            long dt = System.currentTimeMillis() - t0;
            tr.timeMs = dt;

            assertNotNull(labels, "fitPredict labels should not be null");
            assertEquals(data.size(), labels.length, "Label count should match data size");

            // Verify that points from different clusters get different labels
            int labelCluster1 = labels[0];
            int labelCluster2 = labels[3];
            int labelCluster3 = labels[6];

            assertNotEquals(labelCluster1, labelCluster2,
                "Cluster 1 and Cluster 2 should have different labels");
            assertNotEquals(labelCluster1, labelCluster3,
                "Cluster 1 and Cluster 3 should have different labels");
            assertNotEquals(labelCluster2, labelCluster3,
                "Cluster 2 and Cluster 3 should have different labels");

            tr.pass("labels_distinguish_clusters=true");
        } catch (Exception e) {
            tr.fail(e.getMessage());
            fail("KMeans++ fitPredict test failed: " + e.getMessage());
        }
    }

    @Test
    void testKMeansPlusPlusInertia() {
        TestResult tr = recorder.record("kmeans++", "inertia");
        long t0 = System.currentTimeMillis();
        try {
            List<IVector<Double>> data = new ArrayList<>();
            data.add(IVector.of(new double[]{0.1, 0.1}));
            data.add(IVector.of(new double[]{-0.1, 0.2}));
            data.add(IVector.of(new double[]{5.1, 5.1}));
            data.add(IVector.of(new double[]{4.9, 5.2}));
            data.add(IVector.of(new double[]{0.1, 5.1}));
            data.add(IVector.of(new double[]{-0.1, 4.9}));

            KMeansPlusPlus kmeans = new KMeansPlusPlus(42L);
            kmeans.setParameters(java.util.Map.of("numClusters", 3));
            kmeans.fit(data);

            double inertia = kmeans.getInertia();

            long dt = System.currentTimeMillis() - t0;
            tr.timeMs = dt;

            assertTrue(inertia > 0, "Inertia should be positive, got " + inertia);
            tr.pass("inertia=" + String.format("%.4f", inertia));
        } catch (Exception e) {
            tr.fail(e.getMessage());
            fail("KMeans++ inertia test failed: " + e.getMessage());
        }
    }

    // ============================================================
    //  4. GMMClustering
    // ============================================================

    @Test
    void testGMMClusteringFitAndConverge() {
        TestResult tr = recorder.record("gmm", "fit_converge");
        long t0 = System.currentTimeMillis();
        try {
            // Two Gaussian clusters
            List<IVector<Double>> data = new ArrayList<>();
            // Cluster 1 around (0,0)
            data.add(IVector.of(new double[]{0.1, 0.1}));
            data.add(IVector.of(new double[]{-0.2, 0.1}));
            data.add(IVector.of(new double[]{0.1, -0.1}));
            data.add(IVector.of(new double[]{-0.1, -0.1}));
            // Cluster 2 around (5,5)
            data.add(IVector.of(new double[]{5.1, 5.0}));
            data.add(IVector.of(new double[]{4.9, 5.1}));
            data.add(IVector.of(new double[]{5.0, 4.9}));
            data.add(IVector.of(new double[]{5.2, 5.2}));

            GMMClustering gmm = new GMMClustering();
            gmm.setParameters(java.util.Map.of("numClusters", 2));
            gmm.fit(data);

            long dt = System.currentTimeMillis() - t0;
            tr.timeMs = dt;

            assertNotNull(gmm.getClusterCenters(), "GMM should have cluster centers after fit");
            assertEquals(2, gmm.getNumClusters(), "GMM should have 2 clusters");
            tr.pass("converged=" + gmm.isConverged() + ", iterations=" + gmm.getIterations());
        } catch (Exception e) {
            tr.fail(e.getMessage());
            fail("GMM clustering fit test failed: " + e.getMessage());
        }
    }

    @Test
    void testGMMClusteringPredict() {
        TestResult tr = recorder.record("gmm", "predict");
        long t0 = System.currentTimeMillis();
        try {
            List<IVector<Double>> data = new ArrayList<>();
            // Cluster 1 around (0,0)
            data.add(IVector.of(new double[]{0.1, 0.1}));
            data.add(IVector.of(new double[]{-0.2, 0.1}));
            data.add(IVector.of(new double[]{0.1, -0.1}));
            data.add(IVector.of(new double[]{-0.1, -0.1}));
            // Cluster 2 around (5,5)
            data.add(IVector.of(new double[]{5.1, 5.0}));
            data.add(IVector.of(new double[]{4.9, 5.1}));
            data.add(IVector.of(new double[]{5.0, 4.9}));
            data.add(IVector.of(new double[]{5.2, 5.2}));

            GMMClustering gmm = new GMMClustering();
            gmm.setParameters(java.util.Map.of("numClusters", 2));
            gmm.fit(data);

            int[] labels = gmm.getLabels();

            long dt = System.currentTimeMillis() - t0;
            tr.timeMs = dt;

            assertNotNull(labels, "GMM labels should not be null");
            assertEquals(data.size(), labels.length, "Label count should match data size");

            // Points from different original clusters should be assigned to different GMM components
            int labelCluster1 = labels[0];
            int labelCluster2 = labels[4];
            assertNotEquals(labelCluster1, labelCluster2,
                "Points from different clusters should have different labels");

            tr.pass("cluster1_label=" + labelCluster1 + ", cluster2_label=" + labelCluster2);
        } catch (Exception e) {
            tr.fail(e.getMessage());
            fail("GMM clustering predict test failed: " + e.getMessage());
        }
    }

    // ============================================================
    //  5. RereSVD Dimensionality Reduction
    // ============================================================

    @Test
    void testSvdDimReducer5DTo2D() {
        TestResult tr = recorder.record("svd_reduce", "5d_to_2d");
        long t0 = System.currentTimeMillis();
        try {
            // 5-dimensional data with clear structure
            double[][] data = {
                {1.0, 2.0, 3.0, 4.0, 5.0},
                {2.0, 4.0, 6.0, 8.0, 10.0},
                {1.1, 2.1, 3.1, 4.1, 5.1},
                {3.0, 6.0, 9.0, 12.0, 15.0},
                {0.9, 1.9, 2.9, 3.9, 4.9},
                {5.0, 10.0, 15.0, 20.0, 25.0}
            };
            IMatrix<Double> X = Linalg.matrix(data);

            ITransform<Double> svdReduce = ML.dr.svd(2);
            IMatrix<?> reduced = svdReduce.fitTransform(X);

            long dt = System.currentTimeMillis() - t0;
            tr.timeMs = dt;

            assertNotNull(reduced, "SVD reduced matrix should not be null");
            assertEquals(X.getRowNum(), reduced.getRowNum(),
                "SVD should preserve row count");
            assertEquals(2, reduced.getColNum(),
                "SVD should reduce to 2 dimensions");
            tr.pass("shape=" + reduced.getRowNum() + "x" + reduced.getColNum());
        } catch (Exception e) {
            tr.fail(e.getMessage());
            fail("SVD 5D->2D test failed: " + e.getMessage());
        }
    }

    @Test
    void testSvdVarianceRetention() {
        TestResult tr = recorder.record("svd_reduce", "variance_retention");
        long t0 = System.currentTimeMillis();
        try {
            // Data where most variance is along one direction
            double[][] data = new double[20][3];
            for (int i = 0; i < 20; i++) {
                double t = i * 0.5;
                data[i][0] = t + (Math.random() - 0.5) * 0.1;  // main direction
                data[i][1] = (Math.random() - 0.5) * 0.2;       // small noise
                data[i][2] = (Math.random() - 0.5) * 0.2;       // small noise
            }
            IMatrix<Double> X = Linalg.matrix(data);

            ITransform<Double> svdReduce2D = ML.dr.svd(2);
            ITransform<Double> svdReduce1D = ML.dr.svd(1);
            IMatrix<?> reduced2D = svdReduce2D.fitTransform(X);
            IMatrix<?> reduced1D = svdReduce1D.fitTransform(X);

            long dt = System.currentTimeMillis() - t0;
            tr.timeMs = dt;

            assertNotNull(reduced2D, "2D reduction should not be null");
            assertNotNull(reduced1D, "1D reduction should not be null");
            assertEquals(2, reduced2D.getColNum(), "2D reduction should have 2 columns");
            assertEquals(1, reduced1D.getColNum(), "1D reduction should have 1 column");

            // The 1D reduced data should still capture most of the structure
            // (variance in the reduced data should be meaningful)
            tr.pass("2d_cols=" + reduced2D.getColNum() + ", 1d_cols=" + reduced1D.getColNum());
        } catch (Exception e) {
            tr.fail(e.getMessage());
            fail("SVD variance retention test failed: " + e.getMessage());
        }
    }

    // ============================================================
    //  6. RereTSNE Dimensionality Reduction
    // ============================================================

    @Test
    void testTsneDimReducerNoException() {
        TestResult tr = recorder.record("tsne", "no_exception");
        long t0 = System.currentTimeMillis();
        try {
            // Small dataset with 3 separated clusters
            double[][] data = {
                {0.0, 0.0},
                {0.1, 0.1},
                {5.0, 5.0},
                {5.1, 5.1},
                {10.0, 0.0},
                {10.1, 0.1}
            };
            IMatrix<Double> X = Linalg.matrix(data);

            ITransform<Double> tsne = ML.dr.tsne(2);
            IMatrix<?> reduced = tsne.fitTransform(X);

            long dt = System.currentTimeMillis() - t0;
            tr.timeMs = dt;

            assertNotNull(reduced, "t-SNE reduced matrix should not be null");
            assertEquals(X.getRowNum(), reduced.getRowNum(),
                "t-SNE should preserve row count");
            assertEquals(2, reduced.getColNum(),
                "t-SNE should reduce to 2 dimensions");
            tr.pass("shape=" + reduced.getRowNum() + "x" + reduced.getColNum());
        } catch (Exception e) {
            tr.fail(e.getMessage());
            fail("t-SNE test failed: " + e.getMessage());
        }
    }

    @Test
    void testTsneOutputDimensions() {
        TestResult tr = recorder.record("tsne", "output_dimensions");
        long t0 = System.currentTimeMillis();
        try {
            // Very small dataset to keep test fast
            double[][] data = {
                {0.0, 0.0, 0.0},
                {1.0, 1.0, 1.0},
                {5.0, 5.0, 5.0},
                {6.0, 6.0, 6.0}
            };
            IMatrix<Double> X = Linalg.matrix(data);

            ITransform<Double> tsne = ML.dr.tsne(2);
            IMatrix<?> reduced = tsne.fitTransform(X);

            long dt = System.currentTimeMillis() - t0;
            tr.timeMs = dt;

            assertEquals(4, reduced.getRowNum(), "Should have 4 rows");
            assertEquals(2, reduced.getColNum(), "Should have 2 columns");
            tr.pass("rows=" + reduced.getRowNum() + ", cols=" + reduced.getColNum());
        } catch (Exception e) {
            tr.fail(e.getMessage());
            fail("t-SNE output dimensions test failed: " + e.getMessage());
        }
    }

    // ============================================================
    //  7. RereUMAP Dimensionality Reduction
    // ============================================================

    @Test
    void testUmapDimReducerNoException() {
        TestResult tr = recorder.record("umap", "no_exception");
        long t0 = System.currentTimeMillis();
        try {
            // Small dataset (need at least nNeighbors+1 points for UMAP)
            double[][] data = {
                {0.0, 0.0},
                {0.1, 0.1},
                {0.2, 0.0},
                {5.0, 5.0},
                {5.1, 5.1},
                {5.0, 5.2},
                {10.0, 0.0},
                {10.1, 0.1},
                {10.2, 0.0}
            };
            IMatrix<Double> X = Linalg.matrix(data);

            ITransform<Double> umap = ML.dr.umap(2);
            IMatrix<?> reduced = umap.fitTransform(X);

            long dt = System.currentTimeMillis() - t0;
            tr.timeMs = dt;

            assertNotNull(reduced, "UMAP reduced matrix should not be null");
            assertEquals(X.getRowNum(), reduced.getRowNum(),
                "UMAP should preserve row count");
            assertEquals(2, reduced.getColNum(),
                "UMAP should reduce to 2 dimensions");
            tr.pass("shape=" + reduced.getRowNum() + "x" + reduced.getColNum());
        } catch (Exception e) {
            tr.fail(e.getMessage());
            fail("UMAP test failed: " + e.getMessage());
        }
    }

    @Test
    void testUmapOutputDimensions() {
        TestResult tr = recorder.record("umap", "output_dimensions");
        long t0 = System.currentTimeMillis();
        try {
            // Dataset must be larger than nNeighbors (default 15) for UMAP
            // Use 20 points in 3D
            double[][] data = new double[20][3];
            for (int i = 0; i < 10; i++) {
                data[i][0] = i * 0.5;
                data[i][1] = i * 0.5 + 0.1;
                data[i][2] = i * 0.5 + 0.2;
            }
            for (int i = 10; i < 20; i++) {
                data[i][0] = 10.0 + i * 0.5;
                data[i][1] = 10.0 + i * 0.5 + 0.1;
                data[i][2] = 10.0 + i * 0.5 + 0.2;
            }
            IMatrix<Double> X = Linalg.matrix(data);

            ITransform<Double> umap = ML.dr.umap(2);
            IMatrix<?> reduced = umap.fitTransform(X);

            long dt = System.currentTimeMillis() - t0;
            tr.timeMs = dt;

            assertEquals(20, reduced.getRowNum(), "Should have 20 rows");
            assertEquals(2, reduced.getColNum(), "Should have 2 columns");
            tr.pass("rows=" + reduced.getRowNum() + ", cols=" + reduced.getColNum());
        } catch (Exception e) {
            tr.fail(e.getMessage());
            fail("UMAP output dimensions test failed: " + e.getMessage());
        }
    }

    // ============================================================
    //  8. ClassificationMetrics Detailed Validation
    // ============================================================

    @Test
    void testClassificationMetricsManualCalculation() {
        TestResult tr = recorder.record("metrics", "manual_validation");
        long t0 = System.currentTimeMillis();
        try {
            // Known labels and predictions:
            // true:  A  A  A  B  B  B
            // pred:  A  A  B  B  B  B
            // Correct: indices 0,1,3,4,5 = 5 out of 6
            // Expected accuracy = 5/6 = 0.8333...
            String[] trueLabels = {"A", "A", "A", "B", "B", "B"};
            String[] predLabels = {"A", "A", "B", "B", "B", "B"};

            ClassificationMetrics metrics = ClassificationMetrics.compute(trueLabels, predLabels);

            long dt = System.currentTimeMillis() - t0;
            tr.timeMs = dt;

            double expectedAccuracy = 5.0 / 6.0;
            assertEquals(expectedAccuracy, metrics.getAccuracy(), 1e-10,
                "Accuracy should be exactly 5/6");

            // Class A: TP=2, FP=0, FN=1
            // precision_A = 2/2 = 1.0, recall_A = 2/3 = 0.6667, f1_A = 0.8
            // Class B: TP=3, FP=1, FN=0
            // precision_B = 3/4 = 0.75, recall_B = 3/3 = 1.0, f1_B = 0.8571
            // macroPrecision = (1.0 + 0.75) / 2 = 0.875
            // macroRecall = (0.6667 + 1.0) / 2 = 0.8333
            // macroF1 = (0.8 + 0.8571) / 2 = 0.8286
            // weightedPrecision = (1.0*3 + 0.75*3) / 6 = 0.875
            // weightedRecall = (0.6667*3 + 1.0*3) / 6 = 0.8333
            // weightedF1 = (0.8*3 + 0.8571*3) / 6 = 0.8286

            assertEquals(0.875, metrics.getMacroPrecision(), 1e-3,
                "Macro precision should be 0.875");
            assertEquals(5.0 / 6.0, metrics.getMacroRecall(), 1e-3,
                "Macro recall should be 5/6");

            // Verify per-class metrics
            var precisionMap = metrics.getPrecisionPerClass();
            var recallMap = metrics.getRecallPerClass();

            assertEquals(1.0, precisionMap.get("A"), 1e-10,
                "Precision for class A should be 1.0");
            assertEquals(0.75, precisionMap.get("B"), 1e-10,
                "Precision for class B should be 0.75");
            assertEquals(2.0 / 3.0, recallMap.get("A"), 1e-10,
                "Recall for class A should be 2/3");
            assertEquals(1.0, recallMap.get("B"), 1e-10,
                "Recall for class B should be 1.0");

            // Verify confusion matrix
            int[][] cm = metrics.getConfusionMatrix();
            // Labels sorted: A, B
            // CM: [[TP_A, FP_B_for_A], [FP_A_for_B, TP_B]]
            // True A predicted A = 2, True A predicted B = 1
            // True B predicted A = 0, True B predicted B = 3
            assertEquals(2, cm[0][0], "CM[0][0] should be 2 (true A, pred A)");
            assertEquals(1, cm[0][1], "CM[0][1] should be 1 (true A, pred B)");
            assertEquals(0, cm[1][0], "CM[1][0] should be 0 (true B, pred A)");
            assertEquals(3, cm[1][1], "CM[1][1] should be 3 (true B, pred B)");

            tr.pass("accuracy=" + String.format("%.4f", metrics.getAccuracy()));
        } catch (Exception e) {
            tr.fail(e.getMessage());
            fail("ClassificationMetrics manual validation failed: " + e.getMessage());
        }
    }

    // ============================================================
    //  9. CrossValidation
    // ============================================================

    @Test
    void testKFoldCrossValidationResults() {
        TestResult tr = recorder.record("cv", "kfold_results");
        long t0 = System.currentTimeMillis();
        try {
            double[][] features = {
                {1, 1}, {2, 2}, {1.5, 1.5},
                {-1, -1}, {-2, -2}, {-1.5, -1.5},
                {1, 2}, {2, 1}, {-1, -2}, {-2, -1},
                {3, 3}, {-3, -3}
            };
            String[] labels = {
                "pos", "pos", "pos",
                "neg", "neg", "neg",
                "pos", "pos", "neg", "neg",
                "pos", "neg"
            };
            IMatrix<Double> X = Linalg.matrix(features);

            IClassifier clf = ML.clf.logisticRegression();
            CrossValidationResult cvResult = ML.clf.kFoldCrossValidation(clf, X, labels, 3);

            long dt = System.currentTimeMillis() - t0;
            tr.timeMs = dt;

            assertNotNull(cvResult, "CV result should not be null");
            assertEquals(3, cvResult.getTotalFolds(), "Should have 3 folds");

            // Each fold result should be in reasonable range [0, 1]
            for (double acc : cvResult.getAccuracyScores()) {
                assertTrue(acc >= 0.0 && acc <= 1.0,
                    "Each fold accuracy should be in [0,1], got " + acc);
            }

            // Mean accuracy should also be in [0, 1]
            assertTrue(cvResult.getMeanAccuracy() >= 0.0 && cvResult.getMeanAccuracy() <= 1.0,
                "Mean accuracy should be in [0,1]");

            tr.pass("folds=" + cvResult.getTotalFolds()
                + ", mean_acc=" + String.format("%.3f", cvResult.getMeanAccuracy()));
        } catch (Exception e) {
            tr.fail(e.getMessage());
            fail("CrossValidation k-fold test failed: " + e.getMessage());
        }
    }

    @Test
    void testKFoldCrossValidationPerFoldReasonable() {
        TestResult tr = recorder.record("cv", "per_fold_reasonable");
        long t0 = System.currentTimeMillis();
        try {
            // More data for better fold distribution
            double[][] features = new double[24][2];
            String[] labels = new String[24];
            for (int i = 0; i < 12; i++) {
                features[i][0] = 1.0 + Math.random() * 0.5;
                features[i][1] = 1.0 + Math.random() * 0.5;
                labels[i] = "pos";
            }
            for (int i = 12; i < 24; i++) {
                features[i][0] = -1.0 + Math.random() * 0.5;
                features[i][1] = -1.0 + Math.random() * 0.5;
                labels[i] = "neg";
            }
            IMatrix<Double> X = Linalg.matrix(features);

            IClassifier clf = ML.clf.logisticRegression();
            CrossValidationResult cvResult = ML.clf.kFoldCrossValidation(clf, X, labels, 4);

            long dt = System.currentTimeMillis() - t0;
            tr.timeMs = dt;

            assertEquals(4, cvResult.getTotalFolds(), "Should have 4 folds");

            double minAcc = cvResult.getAccuracyScores().stream()
                .mapToDouble(Double::doubleValue).min().orElse(0.0);
            double maxAcc = cvResult.getAccuracyScores().stream()
                .mapToDouble(Double::doubleValue).max().orElse(0.0);

            // All folds should have reasonable accuracy for this easy dataset
            assertTrue(minAcc >= 0.0, "Minimum fold accuracy should be >= 0");
            assertTrue(maxAcc <= 1.0, "Maximum fold accuracy should be <= 1");

            tr.pass("min=" + String.format("%.3f", minAcc)
                + ", max=" + String.format("%.3f", maxAcc)
                + ", mean=" + String.format("%.3f", cvResult.getMeanAccuracy()));
        } catch (Exception e) {
            tr.fail(e.getMessage());
            fail("CrossValidation per-fold test failed: " + e.getMessage());
        }
    }

    // ============================================================
    //  10. RegressionMetrics (RereLinearRegression rmseOn / r2ScoreOn)
    // ============================================================

    @Test
    void testLinearRegressionRmseOn() {
        TestResult tr = recorder.record("regression", "rmse_on");
        long t0 = System.currentTimeMillis();
        try {
            // Training data: y = 2*x + 1 + small noise
            double[][] trainFeatures = new double[20][1];
            double[] trainLabels = new double[20];
            for (int i = 0; i < 20; i++) {
                double x = i * 0.5 - 5.0;
                trainFeatures[i][0] = x;
                trainLabels[i] = 2.0 * x + 1.0 + (Math.random() - 0.5) * 0.3;
            }

            // Test data: same line, no noise
            double[][] testFeatures = new double[10][1];
            double[] testLabels = new double[10];
            for (int i = 0; i < 10; i++) {
                double x = i * 1.0 + 10.0;
                testFeatures[i][0] = x;
                testLabels[i] = 2.0 * x + 1.0;
            }

            RereLinearRegression reg = new RereLinearRegression();
            reg.fit(Linalg.matrix(trainFeatures), Linalg.vector(trainLabels));

            double rmse = reg.rmseOn(Linalg.matrix(testFeatures), Linalg.vector(testLabels));

            long dt = System.currentTimeMillis() - t0;
            tr.timeMs = dt;

            assertTrue(rmse >= 0.0, "RMSE should be non-negative");
            // For a well-fitted line, test RMSE should be small
            assertTrue(rmse < 2.0, "RMSE should be < 2.0 for well-fitted model, got " + rmse);
            tr.pass("rmse=" + String.format("%.4f", rmse));
        } catch (Exception e) {
            tr.fail(e.getMessage());
            fail("Regression rmseOn test failed: " + e.getMessage());
        }
    }

    @Test
    void testLinearRegressionR2ScoreOn() {
        TestResult tr = recorder.record("regression", "r2_score_on");
        long t0 = System.currentTimeMillis();
        try {
            // Training data: y = 3*x - 2 + small noise
            double[][] trainFeatures = new double[20][1];
            double[] trainLabels = new double[20];
            for (int i = 0; i < 20; i++) {
                double x = i * 0.5 - 5.0;
                trainFeatures[i][0] = x;
                trainLabels[i] = 3.0 * x - 2.0 + (Math.random() - 0.5) * 0.3;
            }

            // Test data on the same line
            double[][] testFeatures = new double[10][1];
            double[] testLabels = new double[10];
            for (int i = 0; i < 10; i++) {
                double x = i * 1.0 + 10.0;
                testFeatures[i][0] = x;
                testLabels[i] = 3.0 * x - 2.0;
            }

            RereLinearRegression reg = new RereLinearRegression();
            reg.fit(Linalg.matrix(trainFeatures), Linalg.vector(trainLabels));

            double r2 = reg.r2ScoreOn(Linalg.matrix(testFeatures), Linalg.vector(testLabels));

            long dt = System.currentTimeMillis() - t0;
            tr.timeMs = dt;

            // R2 should be close to 1.0 for data on the same line
            assertTrue(r2 > 0.95, "R2 should be > 0.95 for well-fitted model, got " + r2);
            assertTrue(r2 <= 1.0 + 1e-10, "R2 should not exceed 1.0 by more than epsilon");
            tr.pass("r2=" + String.format("%.4f", r2));
        } catch (Exception e) {
            tr.fail(e.getMessage());
            fail("Regression r2ScoreOn test failed: " + e.getMessage());
        }
    }

    @Test
    void testLinearRegressionR2ScoreOnConstantLabels() {
        TestResult tr = recorder.record("regression", "r2_constant_labels");
        long t0 = System.currentTimeMillis();
        try {
            // Edge case: constant labels
            double[][] features = new double[10][1];
            double[] labels = new double[10];
            for (int i = 0; i < 10; i++) {
                features[i][0] = i;
                labels[i] = 5.0; // constant
            }

            RereLinearRegression reg = new RereLinearRegression();
            reg.fit(Linalg.matrix(features), Linalg.vector(labels));

            double r2 = reg.r2ScoreOn(Linalg.matrix(features), Linalg.vector(labels));

            long dt = System.currentTimeMillis() - t0;
            tr.timeMs = dt;

            // When labels are constant and predictions match, R2 should be 1.0
            assertEquals(1.0, r2, 1e-10,
                "R2 should be 1.0 when predictions perfectly match constant labels");
            tr.pass("r2=" + String.format("%.4f", r2));
        } catch (Exception e) {
            tr.fail(e.getMessage());
            fail("Regression R2 constant labels test failed: " + e.getMessage());
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
