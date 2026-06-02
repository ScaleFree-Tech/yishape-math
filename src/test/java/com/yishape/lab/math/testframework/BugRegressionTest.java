package com.yishape.lab.math.testframework;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.decomposition.Decomps;
import com.yishape.lab.math.linalg.decomposition.ISVDDecomposition;
import com.yishape.lab.math.linalg.decomposition.IEigenDecomposition;
import com.yishape.lab.math.linalg.decomposition.IHessenbergDecomposition;
import com.yishape.lab.math.linalg.solver.MatrixInversionSolver;
import com.yishape.lab.math.ml.clf.lr.RereLogisticRegression;
import com.yishape.lab.math.stats.distribution.PoissonDistribution;
import com.yishape.lab.math.timeseries.TimeSeriesData;
import com.yishape.lab.math.timeseries.TimeSeriesForecasting;
import com.yishape.lab.math.timeseries.TimeSeriesUtils;
import com.yishape.lab.math.timeseries.model.ExponentialSmoothingModels;
import com.yishape.lab.math.timeseries.model.UnifiedARIMAModel;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.util.Tuple3;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.time.LocalDateTime;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Bug Regression Test Suite using completely different test data from previous tests.
 * Validates that 12 previously discovered bugs have been fixed.
 *
 * Framework: JUnit 5 + TestResult recorder
 */
public class BugRegressionTest {

    private static TestResult.Recorder recorder;

    @BeforeAll
    static void setUp() {
        recorder = new TestResult.Recorder("regression", "src/test/resources/results");
    }

    @AfterAll
    static void tearDown() {
        recorder.writeToFile();
    }

    // ========== Bug #1-2: SVD Reconstruction and Singular Values ==========

    @Test
    void testSVDReconstruction_NewData_3x4() {
        // New data: 3x4 matrix (was 2x3 before)
        double[][] data = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
        IMatrix<Double> A = Linalg.matrix(data);

        ISVDDecomposition svd = Decomps.createSVD();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(A);

        IMatrix<Double> U = result._1;
        IVector<Double> S = result._2;
        IMatrix<Double> VT = result._3;

        // Reconstruct: U * diag(S) * VT
        int m = A.rows();
        int n = A.cols();
        int k = Math.min(m, n);

        IMatrix<Double> reconstructed = Linalg.zeros(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int s = 0; s < k; s++) {
                    sum += U.get(i, s) * S.get(s) * VT.get(s, j);
                }
                reconstructed.put(i, j, sum);
            }
        }

        double maxDiff = 0.0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                maxDiff = Math.max(maxDiff, Math.abs(A.get(i, j) - reconstructed.get(i, j)));
            }
        }

        TestResult r = recorder.record("regression", "svd_reconstruction_newdata");
        if (maxDiff < 1e-6) {
            r.pass("reconstruction OK, diff=" + maxDiff);
        } else {
            r.fail("reconstruction FAIL, diff=" + maxDiff, maxDiff, 1e-6);
        }
        assertTrue(maxDiff < 1e-6, "SVD reconstruction error too large: " + maxDiff);
    }

    @Test
    void testSVDSingularValues_NewData_3x4() {
        // Same 3x4 matrix - verify singular values are non-negative and descending
        double[][] data = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
        IMatrix<Double> A = Linalg.matrix(data);

        ISVDDecomposition svd = Decomps.createSVD();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(A);
        IVector<Double> S = result._2;

        boolean allNonNegative = true;
        boolean descending = true;
        for (int i = 0; i < S.length(); i++) {
            if (S.get(i) < 0) allNonNegative = false;
            if (i > 0 && S.get(i) > S.get(i - 1)) descending = false;
        }

        TestResult r = recorder.record("regression", "svd_singular_values_newdata");
        if (allNonNegative && descending) {
            r.pass("singular values OK: non-negative and descending");
        } else {
            r.fail("singular values FAIL: nonNeg=" + allNonNegative + ", desc=" + descending, 0, 0);
        }
        assertTrue(allNonNegative, "Singular values must be non-negative");
        assertTrue(descending, "Singular values must be in descending order");
    }

    // ========== Bug #3: Eigen Decomposition Reconstruction ==========

    @Test
    void testEigenDecomposition_NewData_3x3() {
        // New data: 3x3 symmetric matrix (was 2x2 before)
        double[][] data = {{2, 1, 0}, {1, 3, 1}, {0, 1, 2}};
        IMatrix<Double> A = Linalg.matrix(data);

        IEigenDecomposition eigen = Decomps.createEigen();
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        IVector<Double> eigenvalues = result._1;
        IMatrix<Double> eigenvectors = result._2;

        // Verify A*v = lambda*v for each eigenpair
        double maxDiff = 0.0;
        int n = A.rows();
        for (int i = 0; i < n; i++) {
            double lambda = eigenvalues.get(i);
            IVector<Double> v = eigenvectors.getColumn(i);
            IVector<Double> Av = A.mmul(v);
            IVector<Double> lambdav = v.multiplyByScalar(lambda);
            for (int j = 0; j < n; j++) {
                maxDiff = Math.max(maxDiff, Math.abs(Av.get(j) - lambdav.get(j)));
            }
        }

        TestResult r = recorder.record("regression", "eigen_decomposition_newdata");
        if (maxDiff < 1e-6) {
            r.pass("eigen decomposition OK, diff=" + maxDiff);
        } else {
            r.fail("eigen decomposition FAIL, diff=" + maxDiff, maxDiff, 1e-6);
        }
        assertTrue(maxDiff < 1e-6, "Eigen decomposition error too large: " + maxDiff);
    }

    /**
     * 对称 50×50 良态三对角（与 {@code LargeScaleLinalgTest#generateSymmetricWellConditioned} 同结构）。
     * 此前在开启 {@code jdk.incubator.vector} 时，矩阵×列向量曾误用有缺陷的 SIMD 点积展开，导致 A·v 与 λv 相对残差约 0.19；非特征分解本身 n=50 特例。
     */
    @Test
    void testSymmetricEigenDecomposition_NewData_50x50_wellConditioned() {
        int n = 50;
        long seed = 42L;
        double[][] data = new double[n][n];
        java.util.Random rng = new java.util.Random(seed);
        for (int i = 0; i < n; i++) {
            data[i][i] = 4.0 + rng.nextDouble() * 0.1;
            if (i + 1 < n) {
                double offDiag = -1.0 + rng.nextDouble() * 0.05;
                data[i][i + 1] = offDiag;
                data[i + 1][i] = offDiag;
            }
        }
        IMatrix<Double> A = Linalg.matrix(data);
        assertTrue(A.isSymmetric(), "fixture must be symmetric");

        IEigenDecomposition eigen = Decomps.createEigen();
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        IVector<Double> eigenvalues = result._1;
        IMatrix<Double> eigenvectors = result._2;

        double maxPairErr = 0.0;
        for (int k = 0; k < eigenvalues.length(); k++) {
            IVector<Double> v = eigenvectors.getColumn(k);
            IVector<Double> av = A.mmul(v);
            IVector<Double> lv = v.multiplyByScalar(eigenvalues.get(k));
            double avNorm = av.norm2Value();
            double lvNorm = lv.norm2Value();
            double denom = Math.max(avNorm, lvNorm);
            double pairAbsErr = 0.0;
            for (int i = 0; i < v.length(); i++) {
                pairAbsErr = Math.max(pairAbsErr, Math.abs(av.get(i) - lv.get(i)));
            }
            double pairRelErr = (denom > 0) ? pairAbsErr / denom : pairAbsErr;
            maxPairErr = Math.max(maxPairErr, pairRelErr);
        }

        TestResult rec = recorder.record("regression", "eigen_symmetric_50x50");
        double tol = 1e-6;
        if (maxPairErr < tol) {
            rec.pass("symmetric eigen 50x50 OK, relErr=" + maxPairErr);
        } else {
            rec.fail("symmetric eigen 50x50 FAIL, relErr=" + maxPairErr, maxPairErr, tol);
        }
        assertTrue(maxPairErr < tol, "50x50 symmetric eigen relative residual too large: " + maxPairErr);
    }

    // ========== Bug #4: Hessenberg Decomposition Reconstruction ==========

    @Test
    void testHessenbergDecomposition_NewData_5x5() {
        // New data: 5x5 matrix (was 4x4 before)
        double[][] data = {
            {5, 4, 3, 2, 1},
            {1, 5, 4, 3, 2},
            {2, 1, 5, 4, 3},
            {3, 2, 1, 5, 4},
            {4, 3, 2, 1, 5}
        };
        IMatrix<Double> A = Linalg.matrix(data);

        IHessenbergDecomposition hess = Decomps.createHessenberg();
        Tuple2<IMatrix<Double>, IMatrix<Double>> result = hess.decompose(A);

        IMatrix<Double> H = result._1;
        IMatrix<Double> Q = result._2;

        // Reconstruct: Q * H * Q^T
        IMatrix<Double> QT = Q.transpose();
        IMatrix<Double> reconstructed = Q.mmul(H).mmul(QT);

        double maxDiff = 0.0;
        int n = A.rows();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                maxDiff = Math.max(maxDiff, Math.abs(A.get(i, j) - reconstructed.get(i, j)));
            }
        }

        TestResult r = recorder.record("regression", "hessenberg_reconstruction_newdata");
        if (maxDiff < 1e-6) {
            r.pass("hessenberg reconstruction OK, diff=" + maxDiff);
        } else {
            r.fail("hessenberg reconstruction FAIL, diff=" + maxDiff, maxDiff, 1e-6);
        }
        assertTrue(maxDiff < 1e-6, "Hessenberg reconstruction error too large: " + maxDiff);
    }

    // ========== Bug #5: Singular Matrix Detection ==========

    @Test
    void testSingularMatrixDetection_NewData_Rank1_3x3() {
        // New data: rank-1 3x3 matrix (was 2x2 before)
        double[][] data = {{3, 6, 9}, {1, 2, 3}, {2, 4, 6}};
        IMatrix<Double> A = Linalg.matrix(data);

        TestResult r = recorder.record("regression", "singular_matrix_detection_newdata");
        try {
            IMatrix<Double> inv = MatrixInversionSolver.invert(A);
            r.fail("Expected exception for singular matrix but got result", 0, 0);
            fail("Expected exception for singular matrix");
        } catch (ArithmeticException e) {
            r.pass("singular matrix correctly detected: " + e.getMessage());
        } catch (Exception e) {
            // Other exceptions are also acceptable if they indicate singularity
            if (e.getMessage() != null && (
                e.getMessage().contains("singular") ||
                e.getMessage().contains("singular") ||
                e.getMessage().contains("inverse failed") ||
                e.getMessage().contains("inverse"))) {
                r.pass("singular matrix correctly detected: " + e.getMessage());
            } else {
                r.fail("Unexpected exception: " + e.getMessage(), 0, 0);
                fail("Unexpected exception type: " + e.getClass().getName() + ": " + e.getMessage());
            }
        }
    }

    // ========== Bug #6: Logistic Regression Default Configuration ==========

    @Test
    void testLogisticRegressionDefaultConfig_NewData_4Quadrants() {
        // New data: 4-quadrant classification (was 2-class before)
        double[][] features = {
            {2, 2}, {3, 3}, {2, 3}, {3, 2},     // class A
            {-2, 2}, {-3, 3}, {-2, 3}, {-3, 2}, // class B
            {2, -2}, {3, -3}, {2, -3}, {3, -2}, // class C
            {-2, -2}, {-3, -3}, {-2, -3}, {-3, -2} // class D
        };
        String[] labels = {"A", "A", "A", "A", "B", "B", "B", "B",
                           "C", "C", "C", "C", "D", "D", "D", "D"};

        IMatrix<Double> X = Linalg.matrix(features);

        RereLogisticRegression lr = new RereLogisticRegression();
        lr.setStandardizeFeatures(false);
        lr.fit(X, labels);

        // Predict on training data
        String[] predictions = lr.predictBatch(X);
        int correct = 0;
        for (int i = 0; i < labels.length; i++) {
            if (predictions[i].equals(labels[i])) correct++;
        }
        double accuracy = (double) correct / labels.length;

        TestResult r = recorder.record("regression", "logistic_regression_default_newdata");
        if (accuracy >= 0.85) {
            r.pass("accuracy OK: " + String.format("%.4f", accuracy));
        } else {
            r.fail("accuracy FAIL: " + String.format("%.4f", accuracy), accuracy, 0.85);
        }
        assertTrue(accuracy >= 0.85, "Logistic regression accuracy too low: " + accuracy);
    }

    // ========== Bug #7: Poisson CDF Inversion ==========

    @Test
    void testPoissonCDF_NewData_Lambda3() {
        // New data: lambda=3 (was lambda=5 before)
        PoissonDistribution poisson = new PoissonDistribution(3);

        double pmf0 = poisson.pmf(0);
        double pmf1 = poisson.pmf(1);
        double pmf2 = poisson.pmf(2);
        double pmf3 = poisson.pmf(3);

        double cdf0 = poisson.cdf(0);
        double cdf1 = poisson.cdf(1);
        double cdf2 = poisson.cdf(2);
        double cdf3 = poisson.cdf(3);

        // Reference values for lambda=3:
        // pmf(0) = e^-3 ≈ 0.0498
        // cdf(0) = pmf(0) ≈ 0.0498
        // cdf(1) = pmf(0)+pmf(1) ≈ 0.0498 + 0.1494 = 0.1991
        double refPmf0 = Math.exp(-3);
        double refCdf0 = refPmf0;
        double refCdf1 = refPmf0 + 3 * Math.exp(-3);

        boolean inRange = cdf0 >= 0 && cdf0 <= 1 && cdf1 >= 0 && cdf1 <= 1
                       && cdf2 >= 0 && cdf2 <= 1 && cdf3 >= 0 && cdf3 <= 1;
        boolean monotonic = cdf0 <= cdf1 && cdf1 <= cdf2 && cdf2 <= cdf3;

        TestResult r = recorder.record("regression", "poisson_cdf_newdata");
        double maxErr = Math.max(
            Math.max(Math.abs(pmf0 - refPmf0), Math.abs(cdf0 - refCdf0)),
            Math.abs(cdf1 - refCdf1)
        );

        if (inRange && monotonic && maxErr < 0.01) {
            r.pass("Poisson CDF OK: inRange=" + inRange + ", monotonic=" + monotonic
                 + ", maxErr=" + maxErr);
        } else {
            r.fail("Poisson CDF FAIL: inRange=" + inRange + ", monotonic=" + monotonic
                 + ", maxErr=" + maxErr, maxErr, 0.01);
        }
        assertTrue(inRange, "CDF values must be in [0,1]");
        assertTrue(monotonic, "CDF must be monotonically increasing");
        assertTrue(maxErr < 0.01, "CDF values deviate too much from reference: " + maxErr);
    }

    // ========== Bug #8: MA Prediction Error ==========

    @Test
    void testMAPrediction_NewData_Constant5() {
        // New data: constant sequence of 5s (was 10s before)
        double[] data = {5, 5, 5, 5, 5, 5, 5, 5, 5};
        IVector<Double> vec = Linalg.vector(data);

        // Use moving average forecast
        IVector<Double> forecast = TimeSeriesUtils.movingAverageForecast(vec, 3, 3);

        double maxDiff = 0.0;
        for (int i = 0; i < forecast.length(); i++) {
            maxDiff = Math.max(maxDiff, Math.abs(forecast.get(i) - 5.0));
        }

        TestResult r = recorder.record("regression", "ma_prediction_newdata");
        if (maxDiff < 0.001) {
            r.pass("MA prediction OK, diff=" + maxDiff);
        } else {
            r.fail("MA prediction FAIL, diff=" + maxDiff, maxDiff, 0.001);
        }
        assertTrue(maxDiff < 0.001, "MA prediction error too large: " + maxDiff);
    }

    // ========== Bug #9: Exponential Smoothing alpha=1 ==========

    @Test
    void testExponentialSmoothingAlpha1_NewData_Constant3() {
        // New data: constant sequence of 3s (was 1,2,3,... before)
        double[] data = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
        IVector<Double> vec = Linalg.vector(data);

        // alpha=1: each smoothed value should equal the data value
        IVector<Double> smoothed = TimeSeriesUtils.exponentialSmoothing(vec, 1.0);

        double maxDiff = 0.0;
        for (int i = 0; i < smoothed.length(); i++) {
            maxDiff = Math.max(maxDiff, Math.abs(smoothed.get(i) - 3.0));
        }

        TestResult r = recorder.record("regression", "exponential_smoothing_alpha1_newdata");
        if (maxDiff < 0.001) {
            r.pass("exponential smoothing OK, diff=" + maxDiff);
        } else {
            r.fail("exponential smoothing FAIL, diff=" + maxDiff, maxDiff, 0.001);
        }
        assertTrue(maxDiff < 0.001, "Exponential smoothing alpha=1 error too large: " + maxDiff);
    }

    // ========== Bug #10: ARIMA(0,0,0) Prediction ==========

    @Test
    void testARIMA000Prediction_NewData_Constant7() {
        // New data: constant sequence of 7s (was 1-10 before)
        double[] data = {7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
        IVector<Double> vec = Linalg.vector(data);

        // ARIMA(0,0,0) should predict the mean for constant series
        UnifiedARIMAModel model = UnifiedARIMAModel.fit(vec, 0, 0, 0);
        IVector<Double> forecast = model.forecast(5);

        double maxDiff = 0.0;
        for (int i = 0; i < forecast.length(); i++) {
            maxDiff = Math.max(maxDiff, Math.abs(forecast.get(i) - 7.0));
        }

        TestResult r = recorder.record("regression", "arima000_prediction_newdata");
        if (maxDiff < 0.1) {
            r.pass("ARIMA(0,0,0) prediction OK, diff=" + maxDiff);
        } else {
            r.fail("ARIMA(0,0,0) prediction FAIL, diff=" + maxDiff, maxDiff, 0.1);
        }
        assertTrue(maxDiff < 0.1, "ARIMA(0,0,0) prediction error too large: " + maxDiff);
    }

    // ========== Bug #11: Standardization ==========

    @Test
    void testStandardization_NewData_DifferentData() {
        // New data: different values (was previous data)
        double[] data = {12, 15, 18, 21, 24, 27, 30, 33, 36, 39};
        IVector<Double> vec = Linalg.vector(data);

        IVector<Double> standardized = TimeSeriesUtils.standardize(vec);

        double mean = standardized.meanValue();
        double std = standardized.stdValue();

        TestResult r = recorder.record("regression", "standardization_newdata");
        if (Math.abs(mean) < 1e-10 && Math.abs(std - 1.0) < 1e-6) {
            r.pass("standardization OK: mean=" + mean + ", std=" + std);
        } else {
            r.fail("standardization FAIL: mean=" + mean + ", std=" + std, mean, 0);
        }
        assertTrue(Math.abs(mean) < 1e-10, "Standardized mean should be 0, got " + mean);
        assertTrue(Math.abs(std - 1.0) < 1e-6, "Standardized std should be 1, got " + std);
    }

    // ========== Bug #12: TimeSeriesData Dimensions ==========

    @Test
    void testTimeSeriesDataDimensions_NewData_DifferentConstructors() {
        TestResult r = recorder.record("regression", "timeseries_dimensions_newdata");

        // Constructor 1: from vector
        double[] values1 = {5, 10, 15, 20, 25};
        IVector<Double> vec = Linalg.vector(values1);
        TimeSeriesData ts1 = TimeSeriesData.of(vec, "TestVar");
        boolean ok1 = ts1.getLength() == 5 && ts1.getNumVariables() == 1;

        // Constructor 2: from arrays with timestamps
        LocalDateTime[] timestamps = new LocalDateTime[5];
        LocalDateTime base = LocalDateTime.of(2024, 1, 1, 0, 0);
        for (int i = 0; i < 5; i++) {
            timestamps[i] = base.plusHours(i);
        }
        double[] values2 = {8, 16, 24, 32, 40};
        TimeSeriesData ts2 = TimeSeriesData.of(timestamps, values2, "Var2");
        boolean ok2 = ts2.getLength() == 5 && ts2.getNumVariables() == 1;

        // Constructor 3: from 2D array (multivariate) - use matching timestamps
        double[][] multiData = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
        String[] colNames = {"X", "Y"};
        LocalDateTime[] timestamps4 = new LocalDateTime[4];
        for (int i = 0; i < 4; i++) {
            timestamps4[i] = base.plusHours(i);
        }
        TimeSeriesData ts3 = TimeSeriesData.of(timestamps4, multiData, colNames);
        boolean ok3 = ts3.getLength() == 4 && ts3.getNumVariables() == 2;

        // Constructor 4: static factory from double[]
        TimeSeriesData ts4 = TimeSeriesData.of(Linalg.vector(new double[]{100, 200, 300, 400}), "Z");
        boolean ok4 = ts4.getLength() == 4 && ts4.getNumVariables() == 1;

        if (ok1 && ok2 && ok3 && ok4) {
            r.pass("all dimension checks passed");
        } else {
            r.fail("dimension check FAIL: ok1=" + ok1 + ", ok2=" + ok2 + ", ok3=" + ok3 + ", ok4=" + ok4, 0, 0);
        }
        assertTrue(ok1, "ts1 dimensions wrong: length=" + ts1.getLength() + ", vars=" + ts1.getNumVariables());
        assertTrue(ok2, "ts2 dimensions wrong: length=" + ts2.getLength() + ", vars=" + ts2.getNumVariables());
        assertTrue(ok3, "ts3 dimensions wrong: length=" + ts3.getLength() + ", vars=" + ts3.getNumVariables());
        assertTrue(ok4, "ts4 dimensions wrong: length=" + ts4.getLength() + ", vars=" + ts4.getNumVariables());
    }
}
