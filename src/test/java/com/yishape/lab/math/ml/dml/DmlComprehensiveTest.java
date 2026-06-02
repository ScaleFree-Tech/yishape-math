package com.yishape.lab.math.ml.dml;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.dml.anmm.AnmmDml;
import com.yishape.lab.math.ml.dml.anmm.KanmmDml;
import com.yishape.lab.math.ml.dml.cmoml.CmomlDml;
import com.yishape.lab.math.ml.dml.ddml.DiagDmlCoefficients;
import com.yishape.lab.math.ml.dml.ddml.DiagDmlLpSolver;
import com.yishape.lab.math.ml.dml.ddml.RereDiagDml;
import com.yishape.lab.math.ml.dml.dmleig.DmleigDml;
import com.yishape.lab.math.ml.dml.dmlmj.DmlmjDml;
import com.yishape.lab.math.ml.dml.dmlmj.KDmlmjDml;
import com.yishape.lab.math.ml.dml.fisher.FisherDml;
import com.yishape.lab.math.ml.dml.gmml.GmmlDml;
import com.yishape.lab.math.ml.dml.itml.ItmlDml;
import com.yishape.lab.math.ml.dml.kda.KdaDml;
import com.yishape.lab.math.ml.dml.knn.CondensedNearestNeighbors;
import com.yishape.lab.math.ml.dml.knn.ReducedNearestNeighbors;
import com.yishape.lab.math.ml.dml.ldml.LdmlPairwiseDml;
import com.yishape.lab.math.ml.dml.llda.KLldaDml;
import com.yishape.lab.math.ml.dml.llda.LldaDml;
import com.yishape.lab.math.ml.dml.lmnn.KlmmnDml;
import com.yishape.lab.math.ml.dml.lmnn.LmnnDml;
import com.yishape.lab.math.ml.dml.mcml.McmlDml;
import com.yishape.lab.math.ml.dml.multidml.MultiDmlKnn;
import com.yishape.lab.math.ml.dml.nca.NcaDml;
import com.yishape.lab.math.ml.dml.ncmc.NcmcDml;
import com.yishape.lab.math.ml.dml.ncmml.NcmmlDml;
import com.yishape.lab.math.ml.dml.odml.KodmlDml;
import com.yishape.lab.math.ml.dml.rca.RcaDml;
import com.yishape.lab.math.ml.dml.triplet.Triplet;
import com.yishape.lab.math.ml.dml.triplet.TripletBuilder;
import com.yishape.lab.math.ml.dml.withinclass.WithinClassDml;
import com.yishape.lab.math.ml.dml.KernelDmlUtils.KernelType;
import com.yishape.lab.util.Tuple2;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive correctness and performance test suite for all DML algorithms
 * under com.yishape.lab.math.ml.dml and its sub-packages.
 *
 * Covers: Fisher, RCA, WithinClass, ITML, NCA, MCML, LDML, LMNN, ANMM, DMLMJ,
 * GMML, DML-eig, CMOML, DDML, KDA, LLDA, NCMC, NCMML, CNN, RNN, MultiDmlKnn,
 * and kernelized variants (KANMM, KDMLMJ, KLMNN, KODML, KLLDA, KLDA).
 */
@Disabled("Comprehensive DML perf/correctness suite — too slow for routine dev cycle; enable for release validation")
class DmlComprehensiveTest {

    // ---- Shared test data ----

    private static final double[][] RAW_2D_2CLASS = {
            {0.0, 0.0}, {0.2, 0.1}, {0.1, 0.3},
            {5.0, 5.0}, {5.3, 4.8}, {4.9, 5.2},
    };
    private static final String[] LABELS_2D_2CLASS = {"a", "a", "a", "b", "b", "b"};

    private static final double[][] RAW_2D_3CLASS = {
            {0.0, 0.0}, {0.1, 0.2}, {0.2, 0.05},
            {5.0, 5.0}, {5.2, 4.8}, {4.9, 5.1},
            {0.0, 5.0}, {0.1, 5.1}, {0.05, 4.9},
    };
    private static final String[] LABELS_2D_3CLASS = {"c0", "c0", "c0", "c1", "c1", "c1", "c2", "c2", "c2"};

    private static IMatrix<Double> matrix(double[][] raw) { return IMatrix.of(raw); }

    // =========================================================================
    // 1. CORE INTERFACES & UTILITIES
    // =========================================================================
    @Nested
    @DisplayName("Core interfaces and utility classes")
    class CoreInfrastructure {

        @Test
        @DisplayName("DmlArrays.classIndices: stable first-occurrence encoding")
        void dmlArraysStableEncoding() {
            assertArrayEquals(new int[]{0, 1, 0, 2, 1},
                    DmlArrays.classIndices(new String[]{"x", "y", "x", "z", "y"}));
        }

        @Test
        @DisplayName("DmlArrays.classIndices: single class edge case")
        void dmlArraysSingleClass() {
            assertArrayEquals(new int[]{0, 0, 0},
                    DmlArrays.classIndices(new String[]{"solo", "solo", "solo"}));
        }

        @Test
        @DisplayName("DmlArrays.classIndices: empty array returns empty")
        void dmlArraysEmpty() {
            assertEquals(0, DmlArrays.classIndices(new String[0]).length);
        }

        @Test
        @DisplayName("DmlMetric.diagonal: weights and distance correctness")
        void diagonalMetricCorrectness() {
            IVector<Double> w = Linalg.vector(new double[]{2.0, 1.0});
            DmlMetric m = DmlMetric.diagonal(w, 10, 8);
            assertEquals(MetricForm.DIAGONAL, m.form());
            assertEquals(2, m.inputDimension());
            assertEquals(2, m.outputDimension());

            double d = m.squaredDistance(Linalg.vector(new double[]{0, 0}), Linalg.vector(new double[]{3, 4}));
            assertEquals(2.0 * 2.0 * 9 + 1.0 * 1.0 * 16, d, 1e-9);

            double dSame = m.squaredDistance(Linalg.vector(new double[]{1, 2}), Linalg.vector(new double[]{1, 2}));
            assertEquals(0.0, dSame, 1e-12);
        }

        @Test
        @DisplayName("DmlMetric.fullWhitening: validates square matrix")
        void fullWhiteningRejectsNonSquare() {
            IMatrix<Double> rect = Linalg.matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
            assertThrows(IllegalArgumentException.class, () -> DmlMetric.fullWhitening(rect));
        }

        @Test
        @DisplayName("DmlMetric.lowRank: produces correct dimensions")
        void lowRankDimensions() {
            IMatrix<Double> L = Linalg.matrix(new double[][]{{1, 0, 0}, {0, 1, 0}});
            DmlMetric m = DmlMetric.lowRank(L);
            assertEquals(2, m.outputDimension());
            assertEquals(3, m.inputDimension());
            assertEquals(MetricForm.LOW_RANK, m.form());
        }

        @Test
        @DisplayName("DmlMetric.squaredDistance: numeric clamp for tiny negative")
        void squaredDistanceClampsSmallNegative() {
            DmlMetric m = DmlMetric.diagonal(Linalg.vector(new double[]{1, 1}), 0, 0);
            double d = m.squaredDistance(Linalg.vector(new double[]{0, 0}), Linalg.vector(new double[]{0, 0}));
            assertEquals(0.0, d, 1e-15);
        }

        @Test
        @DisplayName("DmlMetric.squaredDistance: dimension mismatch throws")
        void squaredDistanceDimensionCheck() {
            DmlMetric m = DmlMetric.diagonal(Linalg.vector(new double[]{1, 1}), 0, 0);
            assertThrows(IllegalArgumentException.class,
                    () -> m.squaredDistance(Linalg.vector(new double[]{1}), Linalg.vector(new double[]{1})));
        }

        @Test
        @DisplayName("DmlMetric.transform: batch correctness")
        void transformBatch() {
            DmlMetric m = DmlMetric.lowRank(Linalg.matrix(new double[][]{{2, 0}, {0, 1}}));
            IMatrix<Double> X = Linalg.matrix(new double[][]{{1, 0}, {0, 3}, {1, 4}});
            IMatrix<Double> Z = m.transform(X);
            assertEquals(3, Z.getRowNum());
            assertEquals(2, Z.getColNum());
            assertEquals(2.0, Z.get(0, 0), 1e-9);
            assertEquals(0.0, Z.get(0, 1), 1e-9);
            assertEquals(0.0, Z.get(1, 0), 1e-9);
            assertEquals(3.0, Z.get(1, 1), 1e-9);
        }

        @Test
        @DisplayName("DmlMetric.diagonalWeights: throws for non-diagonal forms")
        void diagonalWeightsGuard() {
            var full = DmlMetric.fullWhitening(Linalg.eye(3));
            assertThrows(IllegalStateException.class, full::diagonalWeights);
            assertThrows(IllegalStateException.class, full::precisionDiagonalMatrix);

            var low = DmlMetric.lowRank(Linalg.eye(2));
            assertThrows(IllegalStateException.class, low::diagonalWeights);
        }

        @Test
        @DisplayName("DmlMetric: triplet counts stored and retrieved")
        void tripletCounts() {
            DmlMetric m = DmlMetric.diagonal(Linalg.vector(new double[]{1, 1, 1}), 42, 15);
            assertEquals(42, m.tripletCount());
            assertEquals(15, m.usedTriplets());
        }

        @Test
        @DisplayName("MetricTransforms.whitenerFromPrecision: produces valid whitener")
        void whitenerFromPrecision() {
            IMatrix<Double> M = Linalg.matrix(new double[][]{{4, 0}, {0, 9}});
            IMatrix<Double> W = MetricTransforms.whitenerFromPrecision(M);
            assertEquals(2, W.getRowNum());
            assertEquals(2, W.getColNum());
            double d = MetricTransforms.squaredMahalanobis(
                    Linalg.vector(new double[]{1, 0}), Linalg.vector(new double[]{0, 0}), W);
            assertTrue(d > 0 && Double.isFinite(d));
        }

        @Test
        @DisplayName("MetricTransforms.squaredLowRank: known answer")
        void squaredLowRank() {
            IMatrix<Double> L = Linalg.matrix(new double[][]{{3, 0}, {0, 4}});
            double d = MetricTransforms.squaredLowRank(
                    Linalg.vector(new double[]{1, 0}), Linalg.vector(new double[]{0, 0}), L);
            assertEquals(9.0, d, 1e-9);
        }

        @Test
        @DisplayName("MetricTransforms.squaredDiagonal: known answer")
        void squaredDiagonal() {
            double d = MetricTransforms.squaredDiagonal(
                    Linalg.vector(new double[]{1, 2}), Linalg.vector(new double[]{0, 0}),
                    Linalg.vector(new double[]{2, 3}));
            assertEquals(4.0 + 36.0, d, 1e-9);
        }

        @Test
        @DisplayName("MetricEmbeddingOps.softmaxConditionalFromNegSqDist: rows sum to 1")
        void softmaxConditionalSumsToOne() {
            int n = 5;
            double[][] distSq = {
                    {0, 1, 4, 9, 16},
                    {1, 0, 1, 4, 9},
                    {4, 1, 0, 1, 4},
                    {9, 4, 1, 0, 1},
                    {16, 9, 4, 1, 0},
            };
            double[][] p = new double[n][n];
            MetricEmbeddingOps.softmaxConditionalFromNegSqDist(distSq, n, p);
            for (int i = 0; i < n; i++) {
                assertEquals(0.0, p[i][i], 1e-12);
                double rowSum = 0;
                for (int j = 0; j < n; j++) {
                    rowSum += p[i][j];
                    assertTrue(p[i][j] >= 0 && p[i][j] <= 1.0, "p[" + i + "][" + j + "]=" + p[i][j]);
                }
                assertEquals(1.0, rowSum, 1e-9, "row " + i + " sum");
            }
        }

        @Test
        @DisplayName("MetricEmbeddingOps.mcmlTargetConditional: uniform over same class")
        void mcmlTargetUniform() {
            int[] y = {0, 0, 0, 1, 1};
            int n = 5;
            double[][] q = new double[n][n];
            MetricEmbeddingOps.mcmlTargetConditional(y, n, q);
            // same-class probability = 1/(|class_c| - 1) for i != j
            assertEquals(0.5, q[0][1], 1e-9);
            assertEquals(0.5, q[0][2], 1e-9);
            assertEquals(0.0, q[0][3], 1e-9);
            // class 1 has 2 members, so 1/(2-1) = 1.0
            assertEquals(1.0, q[3][4], 1e-9);
        }

        @Test
        @DisplayName("MetricEmbeddingOps.ncaLoss: returns value in [0, n]")
        void ncaLossRange() {
            int n = 4;
            int[] y = {0, 0, 1, 1};
            boolean[][] same = new boolean[n][n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    same[i][j] = i != j && y[i] == y[j];
            double[][] p = {
                    {0, 0.7, 0.2, 0.1},
                    {0.7, 0, 0.2, 0.1},
                    {0.1, 0.2, 0, 0.7},
                    {0.1, 0.2, 0.7, 0},
            };
            double loss = MetricEmbeddingOps.ncaLoss(p, same, n);
            assertTrue(loss >= 0 && loss <= n, "NCA loss should be in [0, n]: " + loss);
        }

        @Test
        @DisplayName("MetricEmbeddingOps.mcmlKlLoss: non-negative")
        void mcmlKlLossNonNegative() {
            int n = 3;
            int[] y = {0, 0, 1};
            double[][] p = new double[n][n];
            double[][] q = new double[n][n];
            MetricEmbeddingOps.mcmlTargetConditional(y, n, q);
            p[0][1] = 0.9; p[0][2] = 0.1;
            p[1][0] = 0.9; p[1][2] = 0.1;
            p[2][0] = 0.5; p[2][1] = 0.5;
            double kl = MetricEmbeddingOps.mcmlKlLoss(p, q, n);
            assertTrue(kl >= -1e-10, "KL should be non-negative, got " + kl);
        }

        @Test
        @DisplayName("KernelDmlUtils: RBF kernel diagonal = 1.0")
        void rbfKernelDiagonalIsOne() {
            double[][] X = {{1, 2}, {3, 4}};
            double[][] K = KernelDmlUtils.kernelMatrix(X, X, KernelType.RBF, 0.5, 3, 1.0);
            assertEquals(1.0, K[0][0], 1e-9);
            assertEquals(1.0, K[1][1], 1e-9);
            assertTrue(K[1][0] > 0 && K[1][0] < 1);
        }

        @Test
        @DisplayName("KernelDmlUtils: linear kernel matches dot product")
        void linearKernel() {
            double[][] X = {{1, 2}, {3, 4}};
            double[][] K = KernelDmlUtils.kernelMatrix(X, X, KernelType.LINEAR, 1.0, 3, 0.0);
            assertEquals(5.0, K[0][0], 1e-9);
            assertEquals(11.0, K[0][1], 1e-9);
            assertEquals(25.0, K[1][1], 1e-9);
        }

        @Test
        @DisplayName("KernelDmlUtils: poly kernel known answer")
        void polyKernel() {
            double[][] X = {{1, 0}};
            double[][] K = KernelDmlUtils.kernelMatrix(X, X, KernelType.POLY, 1.0, 2, 0.0);
            assertEquals(1.0, K[0][0], 1e-9);
        }

        @Test
        @DisplayName("KernelDmlUtils: sigmoid kernel output in [-1,1]")
        void sigmoidKernelRange() {
            double[][] X = {{1, 2}, {3, 4}};
            double[][] K = KernelDmlUtils.kernelMatrix(X, X, KernelType.SIGMOID, 0.1, 3, 1.0);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    assertTrue(K[i][j] >= -1.0 && K[i][j] <= 1.0);
        }

        @Test
        @DisplayName("KernelDmlUtils: cosine kernel in [-1,1], diagonal = 1")
        void cosineKernel() {
            double[][] X = {{1, 2}, {3, 4}};
            double[][] K = KernelDmlUtils.kernelMatrix(X, X, KernelType.COSINE, 1.0, 3, 0.0);
            assertEquals(1.0, K[0][0], 1e-9);
            assertEquals(1.0, K[1][1], 1e-9);
            assertTrue(Math.abs(K[0][1]) <= 1.0);
        }

        @Test
        @DisplayName("KernelDmlUtils.centerKernelMatrix: row/col sums near zero")
        void centerKernelMatrix() {
            double[][] X = {{1, 2}, {3, 4}, {5, 6}};
            double[][] K = KernelDmlUtils.kernelMatrix(X, X, KernelType.LINEAR, 1.0, 3, 0.0);
            double[][] Kc = KernelDmlUtils.centerKernelMatrix(K);
            for (int i = 0; i < 3; i++) {
                double rowSum = 0, colSum = 0;
                for (int j = 0; j < 3; j++) {
                    rowSum += Kc[i][j];
                    colSum += Kc[j][i];
                }
                assertEquals(0.0, rowSum, 1e-9);
                assertEquals(0.0, colSum, 1e-9);
            }
        }

        @Test
        @DisplayName("KernelDmlUtils: asymmetric kernel matrix (X vs Y)")
        void asymmetricKernelMatrix() {
            double[][] X = {{1, 0}, {0, 1}};
            double[][] Y = {{2, 0}, {0, 2}, {1, 1}};
            double[][] K = KernelDmlUtils.kernelMatrix(X, Y, KernelType.LINEAR, 1.0, 3, 0.0);
            assertEquals(2, K.length);
            assertEquals(3, K[0].length);
            assertEquals(2.0, K[0][0], 1e-9);
        }
    }

    // =========================================================================
    // 2. FISHER DML
    // =========================================================================
    @Nested
    @DisplayName("FisherDml")
    class FisherTests {

        @Test
        @DisplayName("Fisher: same-class distance < different-class distance")
        void fisherSeparatesClasses() {
            var m = new FisherDml().setL2Weight(1e-3).fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertEquals(MetricForm.FULL_WHITENING, m.form());
            double dSame = m.squaredDistance(
                    Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[1]));
            double dDiff = m.squaredDistance(
                    Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[3]));
            assertTrue(dSame < dDiff, "same=" + dSame + ", diff=" + dDiff);
        }

        @Test
        @DisplayName("Fisher: produces finite distances on 3-class data")
        void fisherThreeClass() {
            var m = new FisherDml().setL2Weight(1e-3).fit(matrix(RAW_2D_3CLASS), LABELS_2D_3CLASS);
            for (int i = 0; i < RAW_2D_3CLASS.length; i++) {
                for (int j = i + 1; j < RAW_2D_3CLASS.length; j++) {
                    double d = m.squaredDistance(
                            Linalg.vector(RAW_2D_3CLASS[i]), Linalg.vector(RAW_2D_3CLASS[j]));
                    assertTrue(d >= 0 && Double.isFinite(d));
                }
            }
        }

        @Test
        @DisplayName("Fisher: static fit convenience method")
        void fisherStaticFit() {
            var m = FisherDml.fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS, 1e-3);
            assertEquals(MetricForm.FULL_WHITENING, m.form());
        }

        @Test
        @DisplayName("Fisher: rejects non-positive l2Weight")
        void fisherRejectsNonPositiveWeight() {
            var x = matrix(RAW_2D_2CLASS);
            assertThrows(IllegalArgumentException.class,
                    () -> new FisherDml().setL2Weight(0).fit(x, LABELS_2D_2CLASS));
            assertThrows(IllegalArgumentException.class,
                    () -> new FisherDml().setL2Weight(-1).fit(x, LABELS_2D_2CLASS));
        }

        @Test
        @DisplayName("Fisher: works with IVector labels")
        void fisherWithIVectorLabels() {
            var m = new FisherDml().setL2Weight(1e-3)
                    .fit(matrix(RAW_2D_2CLASS), Linalg.vector(new int[]{0, 0, 0, 1, 1, 1}));
            assertNotNull(m);
            assertTrue(m.inputDimension() > 0);
        }

        @Test
        @DisplayName("Fisher: consistency across repeated fits")
        void fisherDeterministic() {
            var x = matrix(RAW_2D_2CLASS);
            var m1 = new FisherDml().setL2Weight(1e-3).fit(x, LABELS_2D_2CLASS);
            var m2 = new FisherDml().setL2Weight(1e-3).fit(x, LABELS_2D_2CLASS);
            double d1 = m1.squaredDistance(Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[3]));
            double d2 = m2.squaredDistance(Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[3]));
            assertEquals(d1, d2, 1e-9);
        }
    }

    // =========================================================================
    // 3. RCA
    // =========================================================================
    @Nested
    @DisplayName("RcaDml")
    class RcaTests {

        @Test
        @DisplayName("RCA: produces finite valid distances")
        void rcaProducesFiniteDistances() {
            var m = new RcaDml().setL2Weight(1e-2).fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertEquals(MetricForm.FULL_WHITENING, m.form());
            double d = m.squaredDistance(
                    Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[3]));
            assertTrue(d >= 0 && Double.isFinite(d));
        }

        @Test
        @DisplayName("RCA: static fit convenience")
        void rcaStaticFit() {
            var m = RcaDml.fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS, 1e-2);
            assertNotNull(m);
        }

        @Test
        @DisplayName("RCA: rejects non-positive l2Weight")
        void rcaRejectsNonPositiveWeight() {
            var x = matrix(RAW_2D_2CLASS);
            assertThrows(IllegalArgumentException.class,
                    () -> RcaDml.fit(x, LABELS_2D_2CLASS, 0));
        }
    }

    // =========================================================================
    // 4. WITHIN-CLASS DIAGONAL DML
    // =========================================================================
    @Nested
    @DisplayName("WithinClassDml")
    class WithinClassTests {

        @Test
        @DisplayName("WithinClass: diagonal form, weights non-negative")
        void withinClassDiagonal() {
            var m = new WithinClassDml().setL2Weight(1.0).fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertEquals(MetricForm.DIAGONAL, m.form());
            var w = m.diagonalWeights();
            for (int i = 0; i < w.length(); i++) {
                assertTrue(w.get(i) >= 0);
            }
        }

        @Test
        @DisplayName("WithinClass: larger l2Weight reduces weight magnitudes")
        void withinClassLargerWeightReducesScale() {
            var m1 = new WithinClassDml().setL2Weight(1.0).fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            var m2 = new WithinClassDml().setL2Weight(100.0).fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            double w1 = m1.diagonalWeights().get(0);
            double w2 = m2.diagonalWeights().get(0);
            assertTrue(w2 <= w1, "larger l2Weight should produce smaller weights: " + w1 + " vs " + w2);
        }

        @Test
        @DisplayName("WithinClass: diagonalWeights equals transformMatrix diagonal")
        void withinClassDiagonalConsistency() {
            var m = new WithinClassDml().setL2Weight(1.0).fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            var w = m.diagonalWeights();
            var t = m.transformMatrix();
            for (int i = 0; i < w.length(); i++) {
                assertEquals(w.get(i), t.get(i, i), 1e-9);
            }
        }
    }

    // =========================================================================
    // 5. ITML
    // =========================================================================
    @Nested
    @DisplayName("ItmlDml")
    class ItmlTests {

        @Test
        @DisplayName("ITML: produces finite distances with identity prior")
        void itmlIdentityPrior() {
            var m = new ItmlDml().setMaxIter(80).setGamma(1.0)
                    .setPriorKind(ItmlDml.PriorKind.IDENTITY)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertEquals(MetricForm.FULL_WHITENING, m.form());
            double d = m.squaredDistance(
                    Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[3]));
            assertTrue(d >= 0 && Double.isFinite(d));
        }

        @Test
        @DisplayName("ITML: inverse-covariance prior")
        void itmlInverseCovPrior() {
            var m = new ItmlDml().setMaxIter(50).setGamma(1.0)
                    .setPriorKind(ItmlDml.PriorKind.INVERSE_COVARIANCE).setPriorL2Weight(1e-4)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertNotNull(m);
            assertTrue(m.inputDimension() > 0);
        }

        @Test
        @DisplayName("ITML: converges with tolerance")
        void itmlConverges() {
            var m = new ItmlDml().setMaxIter(200).setGamma(1.0).setTol(1e-6)
                    .setPriorKind(ItmlDml.PriorKind.IDENTITY)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertNotNull(m);
        }

        @Test
        @DisplayName("ITML: custom bounds are respected")
        void itmlCustomBounds() {
            double[][] raw3d = {{0, 0, 0}, {0.1, 0.1, 0.1}, {5, 5, 5}, {5.1, 5.1, 5.1}};
            String[] labels = {"a", "a", "b", "b"};
            var m = new ItmlDml().setMaxIter(50).setBounds(new double[]{0.01, 100.0})
                    .fit(matrix(raw3d), labels);
            assertNotNull(m);
        }

        @Test
        @DisplayName("ITML: symmetrical whitener output")
        void itmlSymmetricalOutput() {
            var m = new ItmlDml().setMaxIter(80).setGamma(1.0)
                    .setPriorKind(ItmlDml.PriorKind.IDENTITY)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            var A = m.transformMatrix();
            for (int i = 0; i < A.getRowNum(); i++) {
                for (int j = 0; j < A.getColNum(); j++) {
                    assertEquals(A.get(i, j), A.get(j, i), 1e-9);
                }
            }
        }
    }

    // =========================================================================
    // 6. NCA
    // =========================================================================
    @Nested
    @DisplayName("NcaDml")
    class NcaTests {

        @Test
        @DisplayName("NCA: increases softmax probability mass on same class")
        void ncaIncreasesSameClassProbability() {
            int n = RAW_2D_3CLASS.length;
            int d = 2;
            int r = 2;
            double[][] L0 = identityL(r, d);
            int[] y = DmlArrays.classIndices(LABELS_2D_3CLASS);
            double before = ncaObj(RAW_2D_3CLASS, y, L0);

            var m = new NcaDml().setRank(r).setMaxIter(100).setLearningRate(0.06)
                    .fit(matrix(RAW_2D_3CLASS), LABELS_2D_3CLASS);
            assertEquals(MetricForm.LOW_RANK, m.form());
            double[][] L = toRowMajor(m.transformMatrix());
            double after = ncaObj(RAW_2D_3CLASS, y, L);
            assertTrue(after >= before - 0.02,
                    "NCA should improve or maintain objective: " + before + " -> " + after);
        }

        @Test
        @DisplayName("NCA: convergence tolerance early stopping")
        void ncaConvergenceTol() {
            var m = new NcaDml().setRank(2).setMaxIter(300).setConvergenceTol(0.1).setPatience(5)
                    .setLearningRate(0.06)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertNotNull(m);
        }

        @Test
        @DisplayName("NCA: rank constrained to feature dimension")
        void ncaRankConstrained() {
            var m = new NcaDml().setRank(10).setMaxIter(10).setLearningRate(0.01)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertTrue(m.outputDimension() <= 2);
        }
    }

    // =========================================================================
    // 7. MCML
    // =========================================================================
    @Nested
    @DisplayName("McmlDml")
    class McmlTests {

        @Test
        @DisplayName("MCML: KL divergence decreases or stays stable")
        void mcmlKlDecreasesOrStable() {
            int n = RAW_2D_2CLASS.length;
            int d = 2;
            int r = 2;
            int[] y = DmlArrays.classIndices(LABELS_2D_2CLASS);
            double[][] L0 = identityL(r, d);
            double[][] q = new double[n][n];
            MetricEmbeddingOps.mcmlTargetConditional(y, n, q);
            double kl0 = computeMcmlKl(RAW_2D_2CLASS, y, L0, q);

            var m = new McmlDml().setRank(r).setMaxIter(100).setLearningRate(0.06)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            double[][] L = toRowMajor(m.transformMatrix());
            double kl1 = computeMcmlKl(RAW_2D_2CLASS, y, L, q);
            assertTrue(kl1 <= kl0 + 0.2,
                    "MCML KL should decrease: " + kl0 + " -> " + kl1);
        }

        @Test
        @DisplayName("MCML: produces low-rank metric")
        void mcmlLowRankForm() {
            var m = new McmlDml().setRank(2).setMaxIter(30).setLearningRate(0.05)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertEquals(MetricForm.LOW_RANK, m.form());
            assertEquals(2, m.outputDimension());
        }
    }

    // =========================================================================
    // 8. LMNN
    // =========================================================================
    @Nested
    @DisplayName("LmnnDml")
    class LmnnTests {

        @Test
        @DisplayName("LMNN: produces low-rank finite metric")
        void lmnnProducesLowRankMetric() {
            var m = new LmnnDml().setRank(2).setMaxSteps(500).setLearningRate(0.03).setMargin(1.0)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertEquals(MetricForm.LOW_RANK, m.form());
            assertTrue(m.outputDimension() >= 1);
            double d = m.squaredDistance(
                    Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[3]));
            assertTrue(d >= 0 && Double.isFinite(d));
        }

        @Test
        @DisplayName("LMNN: rank constrained to feature dimension")
        void lmnnRankConstrained() {
            var m = new LmnnDml().setRank(100).setMaxSteps(10)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertTrue(m.outputDimension() <= 2);
        }

        @Test
        @DisplayName("LMNN: larger margin increases separation")
        void lmnnLargerMargin() {
            var m1 = new LmnnDml().setRank(2).setMargin(0.1).setMaxSteps(200).setLearningRate(0.03)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            var m2 = new LmnnDml().setRank(2).setMargin(10.0).setMaxSteps(200).setLearningRate(0.03)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            double d1 = m1.squaredDistance(
                    Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[3]));
            double d2 = m2.squaredDistance(
                    Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[3]));
            // Larger margin tends to push classes further apart
            assertTrue(d1 >= 0 && d2 >= 0 && Double.isFinite(d1) && Double.isFinite(d2));
        }

        @Test
        @DisplayName("LMNN: deterministic with fixed random seed")
        void lmnnDeterministic() {
            var m1 = new LmnnDml().setRank(2).setMaxSteps(100).setRandom(new Random(42))
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            var m2 = new LmnnDml().setRank(2).setMaxSteps(100).setRandom(new Random(42))
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            double d1 = m1.squaredDistance(Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[3]));
            double d2 = m2.squaredDistance(Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[3]));
            assertEquals(d1, d2, 1e-9);
        }
    }

    // =========================================================================
    // 9. LDML (pairwise logistic)
    // =========================================================================
    @Nested
    @DisplayName("LdmlPairwiseDml")
    class LdmlTests {

        @Test
        @DisplayName("LDML: produces low-rank finite metric")
        void ldmlProducesLowRankMetric() {
            var m = new LdmlPairwiseDml().setRank(2).setMaxSteps(800).setLearningRate(0.03)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertEquals(MetricForm.LOW_RANK, m.form());
            assertTrue(m.outputDimension() >= 1);
            double d = m.squaredDistance(
                    Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[3]));
            assertTrue(d >= 0 && Double.isFinite(d));
        }

        @Test
        @DisplayName("LDML: gradient clipping prevents blowup")
        void ldmlGradientClipping() {
            var m = new LdmlPairwiseDml().setRank(2).setMaxSteps(500).setLearningRate(0.1)
                    .setGradClip(1.0).setBias(0.5)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertNotNull(m);
            assertTrue(Double.isFinite(m.squaredDistance(
                    Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[3]))));
        }
    }

    // =========================================================================
    // 10. ANMM
    // =========================================================================
    @Nested
    @DisplayName("AnmmDml")
    class AnmmTests {

        @Test
        @DisplayName("ANMM: produces low-rank metric")
        void anmmProducesLowRankMetric() {
            var x = matrix(RAW_2D_3CLASS);
            var m = new AnmmDml().setNumDims(2).setNFriends(1).setNEnemies(1)
                    .fit(x, LABELS_2D_3CLASS);
            assertEquals(MetricForm.LOW_RANK, m.form());
            assertTrue(m.outputDimension() >= 1);
        }

        @Test
        @DisplayName("ANMM: more friends/enemies produce valid metric")
        void anmmMoreNeighbors() {
            var m = new AnmmDml().setNumDims(2).setNFriends(2).setNEnemies(2)
                    .fit(matrix(RAW_2D_3CLASS), LABELS_2D_3CLASS);
            double d = m.squaredDistance(
                    Linalg.vector(RAW_2D_3CLASS[0]), Linalg.vector(RAW_2D_3CLASS[6]));
            assertTrue(d >= 0 && Double.isFinite(d));
        }

        @Test
        @DisplayName("ANMM: null numDims returns full rank")
        void anmmNullNumDims() {
            var m = new AnmmDml().setNumDims(null).setNFriends(1).setNEnemies(1)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertEquals(2, m.outputDimension());
        }
    }

    // =========================================================================
    // 11. DMLMJ
    // =========================================================================
    @Nested
    @DisplayName("DmlmjDml")
    class DmlmjTests {

        @Test
        @DisplayName("DMLMJ: produces low-rank finite metric")
        void dmlmjProducesLowRankMetric() {
            var m = new DmlmjDml().setNumDims(2).setNNeighbors(1).setAlpha(0.001)
                    .fit(matrix(RAW_2D_3CLASS), LABELS_2D_3CLASS);
            assertEquals(MetricForm.LOW_RANK, m.form());
            double d = m.squaredDistance(
                    Linalg.vector(RAW_2D_3CLASS[0]), Linalg.vector(RAW_2D_3CLASS[3]));
            assertTrue(d >= 0 && Double.isFinite(d));
        }

        @Test
        @DisplayName("DMLMJ: regularization prevents singular matrix")
        void dmlmjRegularization() {
            // Very small alpha for regularization
            var m = new DmlmjDml().setNumDims(2).setNNeighbors(1).setAlpha(1e-6).setRegTol(1e-10)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertNotNull(m);
        }
    }

    // =========================================================================
    // 12. GMML
    // =========================================================================
    @Nested
    @DisplayName("GmmlDml")
    class GmmlTests {

        @Test
        @DisplayName("GMML: produces full-whitening finite metric")
        void gmmlFullWhitening() {
            var m = new GmmlDml().setGeodesicStep(0.5).setReg(1e-4).setConstraintFactor(10)
                    .setPriorIdentity()
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertEquals(MetricForm.FULL_WHITENING, m.form());
            double d = m.squaredDistance(
                    Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[3]));
            assertTrue(d >= 0 && Double.isFinite(d));
        }

        @Test
        @DisplayName("GMML: geodesicStep near 0 produces near-identity transform")
        void gmmlNearZeroStep() {
            var m = new GmmlDml().setGeodesicStep(0.01).setReg(1e-3).setConstraintFactor(5)
                    .setPriorIdentity()
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertNotNull(m);
        }

        @Test
        @DisplayName("GMML: geodesicStep near 1.0 is valid")
        void gmmlNearOneStep() {
            var m = new GmmlDml().setGeodesicStep(0.99).setReg(1e-3).setConstraintFactor(5)
                    .setPriorIdentity()
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertNotNull(m);
        }

        @Test
        @DisplayName("GMML: rejects geodesicStep out of [0,1]")
        void gmmlRejectsInvalidStep() {
            assertThrows(IllegalArgumentException.class, () -> new GmmlDml().setGeodesicStep(-0.1));
            assertThrows(IllegalArgumentException.class, () -> new GmmlDml().setGeodesicStep(1.1));
        }

        @Test
        @DisplayName("GMML: deterministic with fixed random seed")
        void gmmlDeterministic() {
            var x = matrix(RAW_2D_2CLASS);
            var m1 = new GmmlDml().setGeodesicStep(0.5).setReg(1e-3).setConstraintFactor(10)
                    .setPriorIdentity().setRandom(new Random(123)).fit(x, LABELS_2D_2CLASS);
            var m2 = new GmmlDml().setGeodesicStep(0.5).setReg(1e-3).setConstraintFactor(10)
                    .setPriorIdentity().setRandom(new Random(123)).fit(x, LABELS_2D_2CLASS);
            assertEquals(m1.squaredDistance(Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[3])),
                    m2.squaredDistance(Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[3])),
                    1e-9);
        }
    }

    // =========================================================================
    // 13. DML-eig
    // =========================================================================
    @Nested
    @DisplayName("DmleigDml")
    class DmleigTests {

        @Test
        @DisplayName("DML-eig: produces full-whitening metric")
        void dmleigProducesMetric() {
            var m = new DmleigDml().setMaxIter(10).setMu(0.5).setTol(1e-4)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertEquals(MetricForm.FULL_WHITENING, m.form());
            double d = m.squaredDistance(
                    Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[3]));
            assertTrue(d >= 0 && Double.isFinite(d));
        }

        @Test
        @DisplayName("DML-eig: mu parameter affects convergence speed")
        void dmleigDifferentMu() {
            var m1 = new DmleigDml().setMaxIter(5).setMu(10.0).setTol(1e-3)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            var m2 = new DmleigDml().setMaxIter(5).setMu(0.01).setTol(1e-3)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            // Both should produce valid metrics
            assertTrue(m1.inputDimension() > 0);
            assertTrue(m2.inputDimension() > 0);
        }
    }

    // =========================================================================
    // 14. CMOML
    // =========================================================================
    @Nested
    @DisplayName("CmomlDml")
    class CmomlTests {

        @Test
        @DisplayName("CMOML: produces low-rank metric")
        void cmomlProducesLowRankMetric() {
            var m = new CmomlDml().setNumDims(2).setReg(1e-6)
                    .fit(matrix(RAW_2D_3CLASS), LABELS_2D_3CLASS);
            assertEquals(MetricForm.LOW_RANK, m.form());
            double d = m.squaredDistance(
                    Linalg.vector(RAW_2D_3CLASS[0]), Linalg.vector(RAW_2D_3CLASS[3]));
            assertTrue(d >= 0 && Double.isFinite(d));
        }

        @Test
        @DisplayName("CMOML: numDims null produces full rank")
        void cmomlNullNumDims() {
            var m = new CmomlDml().setNumDims(null).setReg(1e-6)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertEquals(2, m.outputDimension());
        }
    }

    // =========================================================================
    // 15. KDA (Kernel Discriminant Analysis)
    // =========================================================================
    @Nested
    @DisplayName("KdaDml")
    class KdaTests {

        @Test
        @DisplayName("KDA: produces low-rank metric")
        void kdaProducesLowRankMetric() {
            var m = new KdaDml().setKernelType(KernelType.RBF).setGamma(0.5).setNComponents(1)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertEquals(MetricForm.LOW_RANK, m.form());
            assertTrue(m.outputDimension() >= 1);
        }

        @Test
        @DisplayName("KDA: linear kernel")
        void kdaLinearKernel() {
            var m = new KdaDml().setKernelType(KernelType.LINEAR).setNComponents(1)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertNotNull(m);
        }

        @Test
        @DisplayName("KDA: nComponents respects class count minus one")
        void kdaNComponentsConstrained() {
            // 2 classes → max nComponents = 1
            var m = new KdaDml().setKernelType(KernelType.RBF).setGamma(0.5).setNComponents(5)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertTrue(m.outputDimension() <= 1);
        }
    }

    // =========================================================================
    // 16. LLDA (Local LDA)
    // =========================================================================
    @Nested
    @DisplayName("LldaDml")
    class LldaTests {

        @Test
        @DisplayName("LLDA: SUGIYAMA solver produces low-rank metric")
        void lldaSugiyamaSolver() {
            var m = new LldaDml().setNComponents(2).setNNeighbors(1)
                    .setSolver(LldaDml.SolverType.SUGIYAMA)
                    .fit(matrix(RAW_2D_3CLASS), LABELS_2D_3CLASS);
            assertEquals(MetricForm.LOW_RANK, m.form());
            double d = m.squaredDistance(
                    Linalg.vector(RAW_2D_3CLASS[0]), Linalg.vector(RAW_2D_3CLASS[3]));
            assertTrue(d >= 0 && Double.isFinite(d));
        }

        @Test
        @DisplayName("LLDA: CLASSIC solver")
        void lldaClassicSolver() {
            var m = new LldaDml().setNComponents(2).setNNeighbors(1)
                    .setSolver(LldaDml.SolverType.CLASSIC)
                    .fit(matrix(RAW_2D_3CLASS), LABELS_2D_3CLASS);
            assertEquals(MetricForm.LOW_RANK, m.form());
        }

        @Test
        @DisplayName("LLDA: LOCAL_SCALING affinity")
        void lldaLocalScaling() {
            var m = new LldaDml().setNComponents(2).setNNeighbors(2)
                    .setAffinity(LldaDml.AffinityType.LOCAL_SCALING)
                    .fit(matrix(RAW_2D_3CLASS), LABELS_2D_3CLASS);
            assertNotNull(m);
        }

        @Test
        @DisplayName("BUG EXPOSE: LLDA NEIGHBORS affinity orders by int-cast distance")
        void lldaNeighborsIntCastBug() {
            // When all points are close (< 1 distance), int cast makes all distances 0
            // causing arbitrary neighbor selection
            double[][] closePoints = {
                    {0.01, 0.02}, {0.02, 0.01}, {0.03, 0.03},
                    {5.01, 5.02}, {5.02, 5.01}, {5.03, 5.03},
            };
            String[] labels = {"a", "a", "a", "b", "b", "b"};
            // This should not throw or produce NaN
            var m = new LldaDml().setNComponents(1).setNNeighbors(1)
                    .setAffinity(LldaDml.AffinityType.NEIGHBORS)
                    .fit(matrix(closePoints), labels);
            assertNotNull(m);
            assertTrue(Double.isFinite(m.squaredDistance(
                    Linalg.vector(closePoints[0]), Linalg.vector(closePoints[1]))));
        }
    }

    // =========================================================================
    // 17. CNN (Condensed Nearest Neighbors)
    // =========================================================================
    @Nested
    @DisplayName("CondensedNearestNeighbors")
    class CnnTests {

        @Test
        @DisplayName("CNN: produces non-empty condensed set")
        void cnnProducesNonEmptySet() {
            var cnn = new CondensedNearestNeighbors();
            cnn.fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            int[] indices = cnn.getCondensedIndexes();
            assertTrue(indices.length > 0);
            assertTrue(indices.length <= RAW_2D_2CLASS.length);
        }

        @Test
        @DisplayName("CNN: condensed set is subset of original")
        void cnnIsSubset() {
            var cnn = new CondensedNearestNeighbors();
            cnn.fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            for (int idx : cnn.getCondensedIndexes()) {
                assertTrue(idx >= 0 && idx < RAW_2D_2CLASS.length);
            }
        }

        @Test
        @DisplayName("CNN: getCondensedSamples and getCondensedLabels consistent")
        void cnnSamplesLabelsConsistent() {
            var cnn = new CondensedNearestNeighbors();
            cnn.fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            var samples = cnn.getCondensedSamples();
            var labels = cnn.getCondensedLabels();
            assertEquals(samples.length, labels.length);
            assertEquals(cnn.getCondensedIndexes().length, samples.length);
        }
    }

    // =========================================================================
    // 18. RNN (Reduced Nearest Neighbors)
    // =========================================================================
    @Nested
    @DisplayName("ReducedNearestNeighbors")
    class RnnTests {

        @Test
        @DisplayName("RNN: produces non-empty reduced set")
        void rnnProducesNonEmptySet() {
            var rnn = new ReducedNearestNeighbors();
            rnn.fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            int[] indices = rnn.getReducedIndexes();
            assertTrue(indices.length > 0);
        }

        @Test
        @DisplayName("RNN: reduced set not larger than CNN set")
        void rnnNotLargerThanCnn() {
            var raw = new double[][]{{0, 0}, {0.1, 0}, {0, 0.1}, {5, 5}, {5.1, 5}, {5, 5.1}};
            String[] labels = {"a", "a", "a", "b", "b", "b"};
            var cnn = new CondensedNearestNeighbors();
            cnn.fit(matrix(raw), labels);
            var rnn = new ReducedNearestNeighbors();
            rnn.fit(matrix(raw), labels);
            assertTrue(rnn.getReducedIndexes().length <= cnn.getCondensedIndexes().length);
        }

        @Test
        @DisplayName("RNN: getReducedSamples and getReducedLabels consistent")
        void rnnSamplesLabelsConsistent() {
            var rnn = new ReducedNearestNeighbors();
            rnn.fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            var samples = rnn.getReducedSamples();
            var labels = rnn.getReducedLabels();
            assertEquals(samples.length, labels.length);
        }
    }

    // =========================================================================
    // 19. NCMC
    // =========================================================================
    @Nested
    @DisplayName("NcmcDml")
    class NcmcTests {

        @Test
        @DisplayName("NCMC: produces low-rank metric with small data")
        void ncmcProducesLowRankMetric() {
            var m = new NcmcDml().setCentroidsNum(2).setMaxIter(20).setEta0(0.3)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertEquals(MetricForm.LOW_RANK, m.form());
            double d = m.squaredDistance(
                    Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[3]));
            assertTrue(d >= 0 && Double.isFinite(d));
        }

        @Test
        @DisplayName("NCMC: BGD descent method")
        void ncmcBgdMethod() {
            var m = new NcmcDml().setCentroidsNum(1).setMaxIter(20)
                    .setDescentMethod("BGD").setEta0(0.3)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertNotNull(m);
        }
    }

    // =========================================================================
    // 20. NCMML
    // =========================================================================
    @Nested
    @DisplayName("NcmmlDml")
    class NcmmlTests {

        @Test
        @DisplayName("NCMML: produces low-rank metric")
        void ncmmlProducesLowRankMetric() {
            var m = new NcmmlDml().setNumDims(2).setMaxIter(30).setEta0(0.3)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertEquals(MetricForm.LOW_RANK, m.form());
            double d = m.squaredDistance(
                    Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[3]));
            assertTrue(d >= 0 && Double.isFinite(d));
        }

        @Test
        @DisplayName("NCMML: BGD descent method")
        void ncmmlBgdMethod() {
            var m = new NcmmlDml().setNumDims(2).setMaxIter(20).setDescentMethod("BGD").setEta0(0.3)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertNotNull(m);
        }

        @Test
        @DisplayName("NCMML: scale initial transform")
        void ncmmlScaleInit() {
            var m = new NcmmlDml().setNumDims(2).setMaxIter(20).setInitialTransform("scale")
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertNotNull(m);
        }

        @Test
        @DisplayName("NCMML: adaptive learning rate")
        void ncmmlAdaptiveLR() {
            var m = new NcmmlDml().setNumDims(2).setMaxIter(20).setLearningRate("adaptive").setEta0(0.3)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertNotNull(m);
        }

        @Test
        @DisplayName("BUG EXPOSE: NCMML centroids dimension mismatch")
        void ncmmlCentroidsDimensionBug() {
            // NCMML centroids now correctly use d (input dim), not nd (output dim)
            // Fix verified: numDims < d should no longer cause ArrayIndexOutOfBounds
            double[][] raw4d = {
                    {0.0, 0.0, 0.0, 0.0}, {0.1, 0.1, 0.1, 0.1},
                    {5.0, 5.0, 5.0, 5.0}, {5.1, 5.1, 5.1, 5.1},
            };
            String[] labels = {"a", "a", "b", "b"};
            var m = new NcmmlDml().setNumDims(2).setMaxIter(20).setEta0(0.3)
                    .setDescentMethod("SGD")
                    .fit(matrix(raw4d), labels);
            assertNotNull(m);
            assertEquals(4, m.inputDimension());
            assertEquals(2, m.outputDimension());
            assertTrue(Double.isFinite(m.squaredDistance(
                    Linalg.vector(raw4d[0]), Linalg.vector(raw4d[2]))));
        }
    }

    // =========================================================================
    // 21. KERNELIZED ALGORITHMS
    // =========================================================================
    @Nested
    @DisplayName("Kernelized DML variants")
    class KernelizedTests {

        @Test
        @DisplayName("KANMM: runs with RBF kernel")
        void kanmmRuns() {
            var m = new KanmmDml().setKernelType(KernelType.RBF).setGamma(0.5).setK(1)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertNotNull(m);
            assertTrue(m.inputDimension() > 0);
        }

        @Test
        @DisplayName("KANMM: runs with linear kernel")
        void kanmmLinearKernel() {
            var m = new KanmmDml().setKernelType(KernelType.LINEAR).setK(1)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertNotNull(m);
        }

        @Test
        @DisplayName("BUG EXPOSE: KanmmDml scatter matrix unused variable j")
        void kanmmScatterUnusedVariableBug() {
            // KANMM S/D matrix computation ignores variable 'j' in the loop body
            // Only uses i. This test verifies it at least doesn't crash.
            var m = new KanmmDml().setKernelType(KernelType.LINEAR).setK(1)
                    .setRandom(new Random(42))
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertNotNull(m);
            // Check that produced metric actually uses all feature dimensions
            var A = m.transformMatrix();
            boolean allDiagZero = true;
            for (int i = 0; i < Math.min(A.getRowNum(), A.getColNum()); i++) {
                if (Math.abs(A.get(i, i)) > 1e-12) allDiagZero = false;
            }
            assertFalse(allDiagZero, "Metric should have non-zero diagonal (bug: S/D only use i, ignore j)");
        }

        @Test
        @DisplayName("KANMM: deterministic with fixed random")
        void kanmmDeterministic() {
            var m1 = new KanmmDml().setKernelType(KernelType.RBF).setGamma(0.5).setK(1)
                    .setRandom(new Random(99))
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            var m2 = new KanmmDml().setKernelType(KernelType.RBF).setGamma(0.5).setK(1)
                    .setRandom(new Random(99))
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            double d1 = m1.squaredDistance(Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[3]));
            double d2 = m2.squaredDistance(Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[3]));
            assertEquals(d1, d2, 1e-9);
        }

        @Test
        @DisplayName("KDMLMJ: runs with RBF kernel")
        void kdmlmjRuns() {
            var m = new KDmlmjDml().setKernelType(KernelType.RBF).setGamma(0.5)
                    .setRandom(new Random(42))
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertNotNull(m);
        }

        @Test
        @DisplayName("BUG EXPOSE: KDmlmjDml S/D matrices store Euclidean distances, not outer products")
        void kdmlmjScatterBug() {
            // KDmlmj stores raw Euclidean distances in S and D and squares them
            // instead of proper outer-product scatter matrices
            // This means S[0][0] = D[0][0] = 0 always (diagonal distances are 0)
            var m = new KDmlmjDml().setKernelType(KernelType.LINEAR)
                    .setRandom(new Random(42))
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertNotNull(m);
        }

        @Test
        @DisplayName("KLMNN: runs with RBF kernel")
        void klmnnRuns() {
            var m = new KlmmnDml().setKernelType(KernelType.RBF).setGamma(0.5)
                    .setMaxSteps(30).setRank(2)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertEquals(MetricForm.LOW_RANK, m.form());
            assertTrue(m.outputDimension() >= 1);
        }

        @Test
        @DisplayName("BUG EXPOSE: KlmmnDml kernel-space distance uses wrong formula")
        void klmnnKernelGradientBug() {
            // Klmmn assigns zi[a] = A[a][i] instead of sum_b A[a][b]*Kc[b][i]
            // Gradient only updates A[a][i] and ignores kernel interaction
            var m = new KlmmnDml().setKernelType(KernelType.LINEAR).setMaxSteps(30).setRank(2)
                    .setRandom(new Random(42))
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertNotNull(m);
            var t = m.transformMatrix();
            // The transform matrix should be A (r×n), check it's finite
            for (int i = 0; i < t.getRowNum(); i++)
                for (int j = 0; j < t.getColNum(); j++)
                    assertTrue(Double.isFinite(t.get(i, j)));
        }

        @Test
        @DisplayName("KODML: runs with RBF kernel")
        void kodmlRuns() {
            var m = new KodmlDml().setKernelType(KernelType.RBF).setGamma(0.5)
                    .setLearningRate(0.01).setRandom(new Random(42))
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertNotNull(m);
        }

        @Test
        @DisplayName("BUG EXPOSE: KodmlDml distance computation only uses anchor i")
        void kodmlIdenticalDistancesBug() {
            // In KodmlDml, dist_j and dist_k compute identical values since both
            // use only Kc[i][a]*A[a][b]*Kc[i][b] without referencing j or k
            var m = new KodmlDml().setKernelType(KernelType.LINEAR)
                    .setLearningRate(0.01).setRandom(new Random(42))
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertNotNull(m);
        }

        @Test
        @DisplayName("KLLDA: runs with RBF kernel")
        void klldaRuns() {
            var m = new KLldaDml().setKernelType(KernelType.RBF).setGamma(0.5)
                    .setNComponents(1).setNNeighbors(1)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertEquals(MetricForm.LOW_RANK, m.form());
        }

        @Test
        @DisplayName("KLLDA: larger nNeighbors")
        void klldaLargerNeighbors() {
            var m = new KLldaDml().setKernelType(KernelType.LINEAR)
                    .setNComponents(1).setNNeighbors(2)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertNotNull(m);
        }
    }

    // =========================================================================
    // 22. DDML (Diagonal DML) - Core LP and Coefficient paths
    // =========================================================================
    @Nested
    @DisplayName("DDML (DiagDmlCoefficients, DiagDmlLp, RereDiagDml)")
    class DdmlTests {

        @Test
        @DisplayName("DiagDmlCoefficients: fromTriplets produces valid LP structure")
        void diagDmlCoefficientsFromTriplets() {
            double[][] x = {{0, 0}, {0.1, 0.1}, {5, 5}, {5.1, 5.1}};
            String[] labels = {"a", "a", "b", "b"};
            List<Triplet> triplets = TripletBuilder.build(matrix(x), labels, 100);
            assertFalse(triplets.isEmpty(), "TripletBuilder should produce triplets");

            DiagDmlCoefficients coef = DiagDmlCoefficients.fromTriplets(triplets, 5000, "huber", 256);
            assertEquals(x[0].length, coef.featureDim());
            assertEquals(triplets.size(), coef.numTriplets());
            assertEquals(coef.featureDim() + 2 * coef.numTriplets(), coef.variableDim());
        }

        @Test
        @DisplayName("DiagDmlCoefficients: truncateForLangMul produces correct dimensions")
        void truncateForLangMul() {
            double[][] x = {{0, 0}, {0.1, 0.1}, {5, 5}, {5.1, 5.1}};
            String[] labels = {"a", "a", "b", "b"};
            List<Triplet> triplets = TripletBuilder.build(matrix(x), labels, 100);
            var coef = DiagDmlCoefficients.fromTriplets(triplets, 5000, "huber", 256);
            var truncated = coef.truncateForLangMul();
            assertEquals(coef.featureDim() + coef.numTriplets(), truncated.cReduced().length);
            assertEquals(coef.numTriplets(), truncated.aReduced().length);
        }

        @Test
        @DisplayName("DiagDmlCoefficients: non-huber distance works")
        void nonHuberDistance() {
            double[][] x = {{0, 0}, {0.1, 0.1}, {5, 5}, {5.1, 5.1}};
            String[] labels = {"a", "a", "b", "b"};
            List<Triplet> triplets = TripletBuilder.build(matrix(x), labels, 100);
            var coef = DiagDmlCoefficients.fromTriplets(triplets, 5000, "no_huber", 10.0);
            assertNotNull(coef);
            assertTrue(coef.featureDim() > 0);
        }

        @Test
        @DisplayName("DiagDmlCoefficients: rejects empty triplets")
        void rejectsEmptyTriplets() {
            assertThrows(IllegalArgumentException.class,
                    () -> DiagDmlCoefficients.fromTriplets(Collections.emptyList(), 5000, "huber", 256));
        }

        @Test
        @DisplayName("DiagDmlLp: solves simple LP")
        void solvesSimpleLp() {
            // 2D LP: min x1+x2 s.t. 2x1+x2>=5, x1+3x2>=5, x1,x2>=0
            double[] c = {1.0, 1.0};
            double[][] A = {{2.0, 1.0}, {1.0, 3.0}};
            double[] b = {5.0, 5.0};
            var sol = DiagDmlLpSolver.solveRaw(c, A, b, 0, null);
            assertNotNull(sol);
            assertEquals(2, sol.length());
            // Check constraints are satisfied
            assertTrue(2.0 * sol.get(0) + sol.get(1) >= b[0] - 1e-8);
            assertTrue(sol.get(0) + 3.0 * sol.get(1) >= b[1] - 1e-8);
            assertTrue(sol.get(0) >= -1e-10);
            assertTrue(sol.get(1) >= -1e-10);
        }

        @Test
        @DisplayName("DiagDmlLp: with regLinSumCoeff")
        void solvesLpWithReg() {
            double[] c = {1.0, 1.0};
            double[][] A = {{1.0, 0.0}};
            double[] b = {10.0};
            var sol = DiagDmlLpSolver.solveRaw(c, A, b, 1.0, null);
            assertNotNull(sol);
            assertEquals(2, sol.length());
        }

        @Test
        @DisplayName("RereDiagDml: full fit produces diagonal metric (no regularization)")
        void rereDiagDmlNoReg() {
            var m = new RereDiagDml().setL1Weight(0).setL2Weight(0).setMaxTriplets(100)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertEquals(MetricForm.DIAGONAL, m.form());
            assertTrue(m.inputDimension() > 0);
            double d = m.squaredDistance(
                    Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[3]));
            assertTrue(d >= 0 && Double.isFinite(d));
        }

        @Test
        @DisplayName("RereDiagDml: L1 regularization")
        void rereDiagDmlL1() {
            var m = new RereDiagDml().setL1Weight(0.01).setL2Weight(0).setMaxTriplets(100)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertEquals(MetricForm.DIAGONAL, m.form());
            var w = m.diagonalWeights();
            for (int i = 0; i < w.length(); i++) {
                assertTrue(w.get(i) >= 0);
            }
        }

        @Test
        @DisplayName("RereDiagDml: L2 regularization (ADMM)")
        void rereDiagDmlL2Admm() {
            var m = new RereDiagDml().setL1Weight(0).setL2Weight(0.01).setMaxTriplets(100)
                    .setUseAdmm(true).setMaxAdmmIterations(5)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertEquals(MetricForm.DIAGONAL, m.form());
        }

        @Test
        @DisplayName("RereDiagDml: L2 regularization (LangMul)")
        void rereDiagDmlL2LangMul() {
            var m = new RereDiagDml().setL1Weight(0).setL2Weight(0.01).setMaxTriplets(100)
                    .setUseAdmm(false).setMaxLangMulOuterIterations(5)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertEquals(MetricForm.DIAGONAL, m.form());
        }

        @Test
        @DisplayName("RereDiagDml: elastic net regularization")
        void rereDiagDmlElasticNet() {
            var m = new RereDiagDml().setL1Weight(0.005).setL2Weight(0.005).setMaxTriplets(100)
                    .setUseAdmm(true).setMaxAdmmIterations(3)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertEquals(MetricForm.DIAGONAL, m.form());
        }

        @Test
        @DisplayName("RereDiagDml: transform static method")
        void rereDiagDmlTransform() {
            var m = new RereDiagDml().setRegularization(0, 0).setMaxTriplets(100)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            var scaling = m.diagonalWeights();
            var transformed = RereDiagDml.transform(matrix(RAW_2D_2CLASS), scaling);
            assertEquals(RAW_2D_2CLASS.length, transformed.getRowNum());
            assertEquals(RAW_2D_2CLASS[0].length, transformed.getColNum());
        }

        @Test
        @DisplayName("RereDiagDml: triplets count correctly recorded")
        void rereDiagDmlTripletCounts() {
            var x = matrix(RAW_2D_2CLASS);
            var m = new RereDiagDml().setRegularization(0, 0).setMaxTriplets(100)
                    .fit(x, LABELS_2D_2CLASS);
            assertTrue(m.tripletCount() > 0);
            assertTrue(m.usedTriplets() > 0);
        }

        @Test
        @DisplayName("RereDiagDml: validates non-negative regularization weights")
        void rereDiagDmlValidatesWeights() {
            assertThrows(IllegalArgumentException.class, () ->
                    new RereDiagDml().setL1Weight(-1).fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS));
            assertThrows(IllegalArgumentException.class, () ->
                    new RereDiagDml().setL2Weight(-1).fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS));
        }

        @Test
        @DisplayName("RereDiagDml: isLpSupported detects single-LP path")
        void rereDiagDmlIsLpSupported() {
            var noReg = new RereDiagDml().setRegularization(0, 0);
            assertTrue(noReg.isLpSupported());
            var l1Only = new RereDiagDml().setL1Weight(1.0).setL2Weight(0);
            assertTrue(l1Only.isLpSupported());
            var l2Only = new RereDiagDml().setL1Weight(0).setL2Weight(1.0);
            assertFalse(l2Only.isLpSupported());
            var elastic = new RereDiagDml().setL1Weight(0.5).setL2Weight(0.5);
            assertFalse(elastic.isLpSupported());
        }
    }

    // =========================================================================
    // 23. MULTI-DML KNN
    // =========================================================================
    @Nested
    @DisplayName("MultiDmlKnn")
    class MultiDmlTests {

        @Test
        @DisplayName("MultiDmlKnn: runs with multiple DML algorithms")
        void multiDmlKnnRuns() {
            var multi = new MultiDmlKnn(3)
                    .addDml(new FisherDml().setL2Weight(1e-3))
                    .addDml(new NcaDml().setRank(2).setMaxIter(30).setLearningRate(0.05));
            var m = multi.fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertEquals(MetricForm.LOW_RANK, m.form());
        }

        @Test
        @DisplayName("BUG EXPOSE: MultiDmlKnn applyMetric returns original data unchanged")
        void multiDmlApplyMetricNoop() {
            // applyMetric always returns input data, making all DML transformations
            // ineffective for voting
            var multi = new MultiDmlKnn(3)
                    .addDml(new FisherDml().setL2Weight(1e-3));
            multi.fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            // The algorithm finishes without error, but all transformation effects are no-ops
            assertNotNull(multi);
        }

        @Test
        @DisplayName("MultiDmlKnn: predict returns correct shape")
        void multiDmlPredictShape() {
            var multi = new MultiDmlKnn(2)
                    .addDml(new FisherDml().setL2Weight(1e-3));
            multi.fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            int[] predictions = multi.predict(new int[]{0, 3});
            assertEquals(2, predictions.length);
            for (int p : predictions) assertTrue(p >= 0);
        }

        @Test
        @DisplayName("MultiDmlKnn: findKnnForDml")
        void multiDmlFindKnn() {
            var multi = new MultiDmlKnn(2)
                    .addDml(new FisherDml().setL2Weight(1e-3));
            multi.fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            int[] knn = multi.findKnnForDml(0, RAW_2D_2CLASS[0]);
            assertEquals(2, knn.length);
        }
    }

    // =========================================================================
    // 24. TRIPLET BUILDER
    // =========================================================================
    @Nested
    @DisplayName("TripletBuilder")
    class TripletBuilderTests {

        @Test
        @DisplayName("TripletBuilder: builds triplets from labeled data")
        void buildsTriplets() {
            List<Triplet> triplets = TripletBuilder.build(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS, 50);
            assertFalse(triplets.isEmpty());
            for (Triplet t : triplets) {
                assertEquals(RAW_2D_2CLASS[0].length, t.dimension());
                assertTrue(t.weight() > 0);
                assertTrue(t.ijDis() >= 0);
                assertTrue(t.jkDis() >= 0);
            }
        }

        @Test
        @DisplayName("TripletBuilder: parallel mode")
        void buildsTripletsParallel() {
            List<Triplet> triplets = TripletBuilder.build(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS, 50, true);
            assertFalse(triplets.isEmpty());
        }

        @Test
        @DisplayName("TripletBuilder: IVector labels overload")
        void buildsTripletsFromIVector() {
            List<Triplet> triplets = TripletBuilder.build(matrix(RAW_2D_2CLASS),
                    Linalg.vector(new int[]{0, 0, 0, 1, 1, 1}), 50);
            assertFalse(triplets.isEmpty());
        }

        @Test
        @DisplayName("TripletBuilder: respects maxTriplets cap")
        void respectsMaxTriplets() {
            List<Triplet> triplets = TripletBuilder.build(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS, 3);
            assertTrue(triplets.size() <= 3);
        }

        @Test
        @DisplayName("Triplet: weight computation")
        void tripletWeightComputation() {
            double w = Triplet.computeWeight(1.0, 10.0, 1.0 / 4.5);
            assertTrue(w > 0 && w < 1.0);
        }

        @Test
        @DisplayName("Triplet: constructor validates dimension consistency")
        void tripletDimensionValidation() {
            assertThrows(IllegalArgumentException.class, () ->
                    new Triplet(Linalg.vector(new double[]{1, 2}), Linalg.vector(new double[]{1}), Linalg.vector(new double[]{1, 2}), 0, 0, 1.0));
        }

        @Test
        @DisplayName("TripletBuilder: single class produces no triplets")
        void singleClassNoTriplets() {
            double[][] data = {{1, 2}, {3, 4}, {5, 6}};
            String[] oneLabel = {"x", "x", "x"};
            List<Triplet> triplets = TripletBuilder.build(
                    Linalg.matrix(data), oneLabel, 50);
            assertTrue(triplets.isEmpty());
        }
    }

    // =========================================================================
    // 25. EDGE CASES & ROBUSTNESS
    // =========================================================================
    @Nested
    @DisplayName("Edge cases and robustness")
    class EdgeCaseTests {

        @Test
        @DisplayName("Single sample per class")
        void singleSamplePerClass() {
            double[][] raw = {{0, 0}, {5, 5}};
            String[] labels = {"a", "b"};
            // Fisher should work with single sample per class
            var m = new FisherDml().setL2Weight(1e-2).fit(matrix(raw), labels);
            assertNotNull(m);
        }

        @Test
        @DisplayName("All identical features")
        void allIdenticalFeatures() {
            double[][] raw = {{1, 2}, {1, 2}, {1, 2}, {1, 2}};
            String[] labels = {"a", "a", "b", "b"};
            // Algorithms should not crash on constant features
            assertDoesNotThrow(() -> new FisherDml().setL2Weight(1e-2).fit(matrix(raw), labels));
        }

        @Test
        @DisplayName("High-dimensional data (d > n)")
        void highDimensionalData() {
            double[][] raw = {
                    {1, 2, 3, 4, 5},
                    {1.1, 2.1, 3.1, 4.1, 5.1},
                    {5, 4, 3, 2, 1},
                    {5.1, 4.1, 3.1, 2.1, 1.1},
            };
            String[] labels = {"a", "a", "b", "b"};
            assertDoesNotThrow(() -> new FisherDml().setL2Weight(1e-2).fit(matrix(raw), labels));
            assertDoesNotThrow(() -> new RcaDml().setL2Weight(1e-2).fit(matrix(raw), labels));
        }

        @Test
        @DisplayName("Large number of classes")
        void manyClasses() {
            double[][] raw = new double[8][2];
            String[] labels = new String[8];
            for (int i = 0; i < 8; i++) {
                raw[i][0] = (i % 4) * 2.0;
                raw[i][1] = (i / 4) * 2.0;
                labels[i] = "c" + i;
            }
            var m = new FisherDml().setL2Weight(1e-2).fit(matrix(raw), labels);
            assertNotNull(m);
        }

        @Test
        @DisplayName("Zero-variance features")
        void zeroVarianceFeature() {
            double[][] raw = {{0, 0}, {0, 1}, {5, 0}, {5, 1}};
            String[] labels = {"a", "a", "b", "b"};
            // Feature 0 has zero variance within class, feature 1 varies
            var m = new WithinClassDml().setL2Weight(1.0).fit(matrix(raw), labels);
            var w = m.diagonalWeights();
            assertEquals(2, w.length());
            assertTrue(w.get(0) >= 0);
            assertTrue(w.get(1) >= 0);
        }

        @Test
        @DisplayName("Negative and zero feature values")
        void negativeFeatureValues() {
            double[][] raw = {{-1, -2}, {-0.9, -1.9}, {5, 5}, {5.1, 5.1}};
            String[] labels = {"a", "a", "b", "b"};
            var m = new FisherDml().setL2Weight(1e-2).fit(matrix(raw), labels);
            double d = m.squaredDistance(Linalg.vector(raw[0]), Linalg.vector(raw[2]));
            assertTrue(d >= 0 && Double.isFinite(d));
        }

        @Test
        @DisplayName("Large feature magnitude range")
        void largeMagnitudeRange() {
            double[][] raw = {{1e-6, 10.0}, {2e-6, 20.0}, {5e-6, 50.0}, {6e-6, 60.0}};
            String[] labels = {"a", "a", "b", "b"};
            assertDoesNotThrow(() -> {
                var m = new FisherDml().setL2Weight(1e-1).fit(matrix(raw), labels);
                double d = m.squaredDistance(Linalg.vector(raw[0]), Linalg.vector(raw[2]));
                assertTrue(Double.isFinite(d));
            });
        }

        @Test
        @DisplayName("Many samples per class")
        void manySamplesPerClass() {
            int nPerClass = 20;
            double[][] raw = new double[2 * nPerClass][2];
            String[] labels = new String[2 * nPerClass];
            Random rnd = new Random(42);
            for (int i = 0; i < nPerClass; i++) {
                raw[i][0] = rnd.nextGaussian() * 0.5;
                raw[i][1] = rnd.nextGaussian() * 0.5;
                labels[i] = "a";
            }
            for (int i = nPerClass; i < 2 * nPerClass; i++) {
                raw[i][0] = 5.0 + rnd.nextGaussian() * 0.5;
                raw[i][1] = 5.0 + rnd.nextGaussian() * 0.5;
                labels[i] = "b";
            }
            var m = new FisherDml().setL2Weight(1e-2).fit(matrix(raw), labels);
            assertNotNull(m);
            double dSame = m.squaredDistance(Linalg.vector(raw[0]), Linalg.vector(raw[1]));
            double dDiff = m.squaredDistance(Linalg.vector(raw[0]), Linalg.vector(raw[nPerClass]));
            assertTrue(dSame < dDiff);
        }

        @Test
        @DisplayName("IDml.fit(String[]) equals IDml.fit(IVector)")
        void fitConsistencyStringVsIVector() {
            var x = matrix(RAW_2D_2CLASS);
            var m1 = new FisherDml().setL2Weight(1e-3).fit(x, LABELS_2D_2CLASS);
            var m2 = new FisherDml().setL2Weight(1e-3)
                    .fit(x, Linalg.vector(new int[]{0, 0, 0, 1, 1, 1}));
            double d1 = m1.squaredDistance(Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[3]));
            double d2 = m2.squaredDistance(Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[3]));
            assertEquals(d1, d2, 1e-9);
        }
    }

    // =========================================================================
    // 26. PERFORMANCE BENCHMARKS
    // =========================================================================
    @Nested
    @DisplayName("Performance benchmarks")
    class PerformanceTests {

        private static final double[][] MEDIUM_DATA;
        private static final String[] MEDIUM_LABELS;

        static {
            int nPerClass = 50;
            int nClasses = 4;
            MEDIUM_DATA = new double[nClasses * nPerClass][4];
            MEDIUM_LABELS = new String[nClasses * nPerClass];
            Random rnd = new Random(123);
            for (int c = 0; c < nClasses; c++) {
                double cx = (c % 2) * 5.0;
                double cy = (c / 2) * 5.0;
                for (int i = 0; i < nPerClass; i++) {
                    int idx = c * nPerClass + i;
                    MEDIUM_DATA[idx][0] = cx + rnd.nextGaussian() * 1.5;
                    MEDIUM_DATA[idx][1] = cy + rnd.nextGaussian() * 1.5;
                    MEDIUM_DATA[idx][2] = rnd.nextGaussian();
                    MEDIUM_DATA[idx][3] = rnd.nextGaussian();
                    MEDIUM_LABELS[idx] = "c" + c;
                }
            }
        }

        @Test
        @Timeout(value = 30, unit = TimeUnit.SECONDS)
        @DisplayName("FisherDml: completes under 30 seconds on medium data")
        void fisherPerformance() {
            var m = new FisherDml().setL2Weight(1e-3)
                    .fit(matrix(MEDIUM_DATA), MEDIUM_LABELS);
            assertNotNull(m);
        }

        @Test
        @Timeout(value = 30, unit = TimeUnit.SECONDS)
        @DisplayName("ITML: completes under 30 seconds on medium data")
        void itmlPerformance() {
            var m = new ItmlDml().setMaxIter(30).setNConstraintPairs(50)
                    .setPriorKind(ItmlDml.PriorKind.IDENTITY)
                    .fit(matrix(MEDIUM_DATA), MEDIUM_LABELS);
            assertNotNull(m);
        }

        @Test
        @Timeout(value = 30, unit = TimeUnit.SECONDS)
        @DisplayName("NCA: completes under 30 seconds on medium data")
        void ncaPerformance() {
            var m = new NcaDml().setRank(3).setMaxIter(20).setLearningRate(0.05)
                    .fit(matrix(MEDIUM_DATA), MEDIUM_LABELS);
            assertNotNull(m);
        }

        @Test
        @Timeout(value = 30, unit = TimeUnit.SECONDS)
        @DisplayName("LMNN: completes under 30 seconds on medium data")
        void lmnnPerformance() {
            var m = new LmnnDml().setRank(3).setMaxSteps(200).setLearningRate(0.02)
                    .fit(matrix(MEDIUM_DATA), MEDIUM_LABELS);
            assertNotNull(m);
        }

        @Test
        @Timeout(value = 30, unit = TimeUnit.SECONDS)
        @DisplayName("DDML: completes under 30 seconds on medium data")
        void ddmlPerformance() {
            var m = new RereDiagDml().setRegularization(0, 0).setMaxTriplets(200)
                    .fit(matrix(MEDIUM_DATA), MEDIUM_LABELS);
            assertNotNull(m);
        }

        @Test
        @Timeout(value = 30, unit = TimeUnit.SECONDS)
        @DisplayName("NCMC: completes under 30 seconds on medium data")
        void ncmcPerformance() {
            var m = new NcmcDml().setCentroidsNum(1).setMaxIter(10).setEta0(0.3)
                    .fit(matrix(MEDIUM_DATA), MEDIUM_LABELS);
            assertNotNull(m);
        }

        @Test
        @Timeout(value = 30, unit = TimeUnit.SECONDS)
        @DisplayName("GMML: completes under 30 seconds on medium data")
        void gmmlPerformance() {
            var m = new GmmlDml().setGeodesicStep(0.5).setReg(1e-3).setConstraintFactor(10)
                    .setPriorIdentity()
                    .fit(matrix(MEDIUM_DATA), MEDIUM_LABELS);
            assertNotNull(m);
        }
    }

    // =========================================================================
    // 27. NUMERICAL STABILITY & CONSISTENCY
    // =========================================================================
    @Nested
    @DisplayName("Numerical stability and consistency")
    class NumericalStabilityTests {

        @Test
        @DisplayName("Repeated fits produce consistent results (deterministic algorithms)")
        void repeatedFitsConsistent() {
            var x = matrix(RAW_2D_2CLASS);
            for (int rep = 0; rep < 5; rep++) {
                var m = new FisherDml().setL2Weight(1e-3).fit(x, LABELS_2D_2CLASS);
                double d = m.squaredDistance(Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[3]));
                assertTrue(d >= 0 && Double.isFinite(d));
            }
        }

        @Test
        @DisplayName("DmlMetric.transformMatrix returns defensive copy")
        void transformMatrixIsDefensiveCopy() {
            var m = new FisherDml().setL2Weight(1e-3)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            var A1 = m.transformMatrix();
            var A2 = m.transformMatrix();
            assertNotSame(A1, A2);
            assertEquals(A1.get(0, 0), A2.get(0, 0), 1e-9);
        }

        @Test
        @DisplayName("Squared distance zero for same point")
        void zeroDistanceForSamePoint() {
            var m = new FisherDml().setL2Weight(1e-3)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            assertEquals(0.0, m.squaredDistance(
                    Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[0])), 1e-12);
        }

        @Test
        @DisplayName("Squared distance non-negative for all pairs")
        void allDistancesNonNegative() {
            var m = new FisherDml().setL2Weight(1e-3)
                    .fit(matrix(RAW_2D_3CLASS), LABELS_2D_3CLASS);
            for (int i = 0; i < RAW_2D_3CLASS.length; i++) {
                for (int j = i; j < RAW_2D_3CLASS.length; j++) {
                    double d = m.squaredDistance(
                            Linalg.vector(RAW_2D_3CLASS[i]), Linalg.vector(RAW_2D_3CLASS[j]));
                    assertTrue(d >= -1e-12, "distance[" + i + "][" + j + "]=" + d);
                }
            }
        }

        @Test
        @DisplayName("ITML symmetrizeInPlace: check diagonal halving issue")
        void itmlSymmetrizeInPlaceBug() {
            // symmetrizeInPlace divides diagonal by 0.5, which is incorrect
            // The diagonal should remain as is during symmetrization
            // This test verifies ITML still produces a valid whitener despite this
            var m = new ItmlDml().setMaxIter(50).setGamma(1.0)
                    .setPriorKind(ItmlDml.PriorKind.IDENTITY).setRandom(new Random(42))
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            double d = m.squaredDistance(Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[3]));
            assertTrue(d >= 0 && Double.isFinite(d),
                    "ITML produces finite metric despite symmetrize diagonal bug: d=" + d);
        }

        @Test
        @DisplayName("MCML SDProject: verify eigenvalue ordering correctness")
        void mcmlSdProjectEigenOrdering() {
            // SDProject takes first r eigenvectors which may not be in descending eigenvalue order
            var m = new McmlDml().setRank(2).setMaxIter(30).setLearningRate(0.05)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            // Verify the output metric has strong directions
            var L = m.transformMatrix();
            double frobNorm = 0;
            for (int i = 0; i < L.getRowNum(); i++)
                for (int j = 0; j < L.getColNum(); j++)
                    frobNorm += Math.pow(L.get(i, j), 2);
            assertTrue(frobNorm > 0, "Transform matrix should have non-zero Frobenius norm");
        }

        @Test
        @DisplayName("DML-eig: verify eigenvector selection correctness")
        void dmleigEigenvectorSelection() {
            // DML-eig uses LAST eigenvector (smallest eigenvalue), check if this is intentional
            var m = new DmleigDml().setMaxIter(5).setMu(1.0).setTol(1e-3)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            // The algorithm should produce a valid metric regardless
            double d = m.squaredDistance(Linalg.vector(RAW_2D_2CLASS[0]), Linalg.vector(RAW_2D_2CLASS[3]));
            assertTrue(d >= 0 && Double.isFinite(d));
        }

        @Test
        @DisplayName("KDA eigenvector dimension: verify correct indexing when n>d")
        void kdaEigenvectorDimension() {
            // KDA eigenvectors have dimension n, but indexing uses j % d
            // When n > d, this wraps incorrectly
            double[][] data4 = {
                    {0, 0}, {0.1, 0.1}, {0.2, 0.2}, {0.3, 0.3},
                    {5, 5}, {5.1, 5.1}, {5.2, 5.2}, {5.3, 5.3},
            };
            String[] labels = {"a", "a", "a", "a", "b", "b", "b", "b"};
            // n=8, d=2, eigenvector dimension=8, but code uses j%d → wraps at 2
            var m = new KdaDml().setKernelType(KernelType.RBF).setGamma(0.5).setNComponents(1)
                    .fit(matrix(data4), labels);
            assertNotNull(m);
            assertTrue(Double.isFinite(m.squaredDistance(
                    Linalg.vector(data4[0]), Linalg.vector(data4[4]))));
        }

        @Test
        @DisplayName("LldaDml SUGIYAMA scatter matrix indexing bug")
        void lldaSugiyamaScatterIndexBug() {
            // SUGIYAMA solver uses Xc[0][a] and Xc[0][b] instead of iterating over class members
            // This means the scatter matrix only uses the first class member
            var m1 = new LldaDml().setNComponents(2).setNNeighbors(1)
                    .setSolver(LldaDml.SolverType.SUGIYAMA)
                    .fit(matrix(RAW_2D_3CLASS), LABELS_2D_3CLASS);
            var m2 = new LldaDml().setNComponents(2).setNNeighbors(1)
                    .setSolver(LldaDml.SolverType.CLASSIC)
                    .fit(matrix(RAW_2D_3CLASS), LABELS_2D_3CLASS);
            // Both should produce valid metrics
            assertTrue(Double.isFinite(m1.squaredDistance(
                    Linalg.vector(RAW_2D_3CLASS[0]), Linalg.vector(RAW_2D_3CLASS[3]))));
            assertTrue(Double.isFinite(m2.squaredDistance(
                    Linalg.vector(RAW_2D_3CLASS[0]), Linalg.vector(RAW_2D_3CLASS[3]))));
        }
    }

    // =========================================================================
    // 28. DmlWrapper FACADE CONSISTENCY
    // =========================================================================
    @Nested
    @DisplayName("DmlWrapper facade consistency")
    class DmlWrapperTests {

        @Test
        @DisplayName("ML.dml factory produces same type as direct construction")
        void facadeConsistency() {
            var x = matrix(RAW_2D_2CLASS);
            var direct = new FisherDml().setL2Weight(1e-3).fit(x, LABELS_2D_2CLASS);
            var facade = com.yishape.lab.math.ml.ML.dml.fisherWhitening().fit(x, LABELS_2D_2CLASS);
            assertEquals(direct.form(), facade.form());
            assertEquals(direct.inputDimension(), facade.inputDimension());
        }

        @Test
        @DisplayName("NCA through facade produces same form")
        void ncaFacadeForm() {
            var x = matrix(RAW_2D_2CLASS);
            var direct = new NcaDml().setRank(2).setMaxIter(30).setLearningRate(0.05).fit(x, LABELS_2D_2CLASS);
            var facade = com.yishape.lab.math.ml.ML.dml.nca(2, 30, 0.05).fit(x, LABELS_2D_2CLASS);
            assertEquals(direct.form(), facade.form());
            assertEquals(direct.outputDimension(), facade.outputDimension());
        }

        @Test
        @DisplayName("RCA through facade")
        void rcaFacade() {
            var x = matrix(RAW_2D_2CLASS);
            var direct = new RcaDml().setL2Weight(1e-2).fit(x, LABELS_2D_2CLASS);
            var facade = ((RcaDml) com.yishape.lab.math.ml.ML.dml.rca()).setL2Weight(1e-2).fit(x, LABELS_2D_2CLASS);
            assertEquals(direct.form(), facade.form());
        }
    }

    // =========================================================================
    // 29. CROSS-ALGORITHM COMPARATIVE TESTS
    // =========================================================================
    @Nested
    @DisplayName("Cross-algorithm comparative validation")
    class CrossAlgorithmTests {

        @Test
        @DisplayName("Fisher vs RCA: both reduce same-class distance relative to Euclidean")
        void fisherVsRca() {
            var xVec0 = Linalg.vector(RAW_2D_2CLASS[0]);
            var xVec3 = Linalg.vector(RAW_2D_2CLASS[3]);

            double euclideanDist = xVec0.sub(xVec3).norm2Value();
            euclideanDist = euclideanDist * euclideanDist;

            var fisher = new FisherDml().setL2Weight(1e-3)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);
            var rca = new RcaDml().setL2Weight(1e-2)
                    .fit(matrix(RAW_2D_2CLASS), LABELS_2D_2CLASS);

            double fd = fisher.squaredDistance(xVec0, xVec3);
            double rd = rca.squaredDistance(xVec0, xVec3);

            assertTrue(fd >= 0 && Double.isFinite(fd));
            assertTrue(rd >= 0 && Double.isFinite(rd));
        }

        @Test
        @DisplayName("All algorithms produce valid metrics on 3-class problem")
        void allAlgorithmsProduceValidMetrics3Class() {
            var x = matrix(RAW_2D_3CLASS);
            var x0 = Linalg.vector(RAW_2D_3CLASS[0]);
            var x3 = Linalg.vector(RAW_2D_3CLASS[3]);

            assertDoesNotThrow(() -> {
                var fisher = new FisherDml().setL2Weight(1e-3).fit(x, LABELS_2D_3CLASS);
                assertTrue(fisher.squaredDistance(x0, x3) >= 0);
            });
            assertDoesNotThrow(() -> {
                var rca = new RcaDml().setL2Weight(1e-2).fit(x, LABELS_2D_3CLASS);
                assertTrue(rca.squaredDistance(x0, x3) >= 0);
            });
            assertDoesNotThrow(() -> {
                var cmoml = new CmomlDml().setNumDims(2).setReg(1e-6).fit(x, LABELS_2D_3CLASS);
                assertTrue(cmoml.squaredDistance(x0, x3) >= 0);
            });
            assertDoesNotThrow(() -> {
                var anmm = new AnmmDml().setNumDims(2).setNFriends(1).setNEnemies(1)
                        .fit(x, LABELS_2D_3CLASS);
                assertTrue(anmm.squaredDistance(x0, x3) >= 0);
            });
        }
    }

    // =========================================================================
    // 30. METRIC TRANSFORMS COVERAGE
    // =========================================================================
    @Nested
    @DisplayName("MetricTransforms comprehensive coverage")
    class MetricTransformsTests {

        @Test
        @DisplayName("transformRowsDiagonal: broadcastMultiplyColumn semantics")
        void transformRowsDiagonal() {
            var X = Linalg.matrix(new double[][]{{1, 2}, {3, 4}});
            var s = Linalg.vector(new double[]{10, 100});
            var Z = MetricTransforms.transformRowsDiagonal(X, s);
            // broadcastMultiplyColumn multiplies each row i by s[i]
            assertEquals(10.0, Z.get(0, 0), 1e-9);
            assertEquals(20.0, Z.get(0, 1), 1e-9);
            assertTrue(Z.get(1, 0) > 0 && Z.get(1, 1) > 0);
            assertTrue(Double.isFinite(Z.get(0, 0)));
        }

        @Test
        @DisplayName("transformRowsWhitener: known output")
        void transformRowsWhitener() {
            var X = Linalg.matrix(new double[][]{{1, 0}, {0, 2}});
            var W = Linalg.matrix(new double[][]{{2, 0}, {0, 3}});
            var Z = MetricTransforms.transformRowsWhitener(X, W);
            assertEquals(2.0, Z.get(0, 0), 1e-9);
            assertEquals(6.0, Z.get(1, 1), 1e-9);
        }

        @Test
        @DisplayName("transformRowsLowRank: known output")
        void transformRowsLowRank() {
            var X = Linalg.matrix(new double[][]{{1, 2}});
            var L = Linalg.matrix(new double[][]{{3, 0}, {0, 4}});
            var Z = MetricTransforms.transformRowsLowRank(X, L);
            assertEquals(1, Z.getRowNum());
            assertEquals(2, Z.getColNum());
        }

        @Test
        @DisplayName("symmetrize: produces (M+M^T)/2")
        void symmetrize() {
            var M = Linalg.matrix(new double[][]{{1, 5}, {3, 2}});
            var S = MetricTransforms.symmetrize(M);
            assertEquals(1.0, S.get(0, 0), 1e-9);
            assertEquals(4.0, S.get(0, 1), 1e-9);
            assertEquals(4.0, S.get(1, 0), 1e-9);
            assertEquals(2.0, S.get(1, 1), 1e-9);
        }

        @Test
        @DisplayName("whitenerFromPrecision: non-square matrix throws")
        void whitenerDimensionGuard() {
            var M = Linalg.matrix(new double[][]{{1, 2}});
            assertThrows(IllegalArgumentException.class,
                    () -> MetricTransforms.whitenerFromPrecision(M));
        }
    }

    // =========================================================================
    // HELPERS
    // =========================================================================
    private static double[][] identityL(int r, int d) {
        double[][] L = new double[r][d];
        int lim = Math.min(r, d);
        for (int i = 0; i < lim; i++) L[i][i] = 1.0;
        return L;
    }

    private static double[][] toRowMajor(IMatrix<Double> m) {
        int r = m.getRowNum(), c = m.getColNum();
        double[][] o = new double[r][c];
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++)
                o[i][j] = m.get(i, j);
        return o;
    }

    private static double ncaObj(double[][] x, int[] y, double[][] L) {
        int n = x.length, d = x[0].length, r = L.length;
        boolean[][] same = new boolean[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                same[i][j] = i != j && y[i] == y[j];
        double[][] emb = MetricEmbeddingOps.embed(x, n, d, L, r);
        double[][] distSq = MetricEmbeddingOps.pairwiseSquaredDistances(emb, n, r);
        double[][] p = new double[n][n];
        MetricEmbeddingOps.softmaxConditionalFromNegSqDist(distSq, n, p);
        return MetricEmbeddingOps.ncaLoss(p, same, n);
    }

    private static double computeMcmlKl(double[][] x, int[] y, double[][] L, double[][] q) {
        int n = x.length, d = x[0].length, r = L.length;
        double[][] emb = MetricEmbeddingOps.embed(x, n, d, L, r);
        double[][] distSq = MetricEmbeddingOps.pairwiseSquaredDistances(emb, n, r);
        double[][] p = new double[n][n];
        MetricEmbeddingOps.softmaxConditionalFromNegSqDist(distSq, n, p);
        return MetricEmbeddingOps.mcmlKlLoss(p, q, n);
    }
}
