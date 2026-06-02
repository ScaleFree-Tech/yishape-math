package com.yishape.lab.math.ml.dml;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.dml.fisher.FisherDml;
import com.yishape.lab.math.ml.dml.itml.ItmlDml;
import com.yishape.lab.math.ml.dml.ldml.LdmlPairwiseDml;
import com.yishape.lab.math.ml.dml.lmnn.LmnnDml;
import com.yishape.lab.math.ml.dml.mcml.McmlDml;
import com.yishape.lab.math.ml.dml.nca.NcaDml;
import com.yishape.lab.math.ml.dml.rca.RcaDml;
import com.yishape.lab.math.ml.dml.withinclass.WithinClassDml;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 各子包度量学习算法的烟测与简单正确性检查。
 */
class AllDmlMetricLearningTest {

    @Test
    void supervisedMetricLearnerInterface() {
        double[][] raw = {
                {0.0, 0.0},
                {5.0, 5.0},
        };
        IMatrix<Double> x = IMatrix.of(raw);
        String[] labels = {"a", "b"};
        ISupervisedDml learner = new FisherDml().setL2Weight(1e-3);
        DmlMetric m = learner.fit(x, labels);
        assertEquals(MetricForm.FULL_WHITENING, m.form());
        assertTrue(m.squaredDistance(Linalg.vector(raw[0]), Linalg.vector(raw[1])) > 0);
        ISupervisedDml fromFacade = com.yishape.lab.math.ml.ML.dml.fisherWhitening();
        assertEquals(MetricForm.FULL_WHITENING, fromFacade.fit(x, labels).form());
    }

    @Test
    void fisherSeparatesClasses() {
        double[][] raw = {
                {0.0, 0.0},
                {0.1, 0.05},
                {5.0, 5.0},
                {5.2, 4.9},
        };
        IMatrix<Double> x = IMatrix.of(raw);
        String[] labels = {"a", "a", "b", "b"};
        var m = FisherDml.fit(x, labels, 1e-3);
        assertEquals(MetricForm.FULL_WHITENING, m.form());
        double dIn = m.squaredDistance(Linalg.vector(raw[0]), Linalg.vector(raw[1]));
        double dOut = m.squaredDistance(Linalg.vector(raw[0]), Linalg.vector(raw[2]));
        assertTrue(dIn < dOut);
    }

    @Test
    void rcaRuns() {
        double[][] raw = {{1, 0}, {0, 1}, {2, 0}, {0, 2}};
        IMatrix<Double> x = IMatrix.of(raw);
        String[] labels = {"0", "0", "1", "1"};
        var m = RcaDml.fit(x, labels, 1e-2);
        double d = m.squaredDistance(Linalg.vector(raw[0]), Linalg.vector(raw[1]));
        assertTrue(d >= 0 && Double.isFinite(d));
    }

    @Test
    void diagonalWithinClassScaling() {
        double[][] raw = {{0, 0}, {2, 0}};
        IMatrix<Double> x = IMatrix.of(raw);
        String[] labels = {"a", "b"};
        var m = WithinClassDml.fit(x, labels, 1.0);
        assertEquals(MetricForm.DIAGONAL, m.form());
        double manual = m.squaredDistance(Linalg.vector(raw[0]), Linalg.vector(raw[1]));
        assertTrue(manual > 0);
        assertArrayEquals(new double[] {1.0, 1.0}, new double[] {
                m.diagonalWeights().get(0),
                m.diagonalWeights().get(1),
        }, 1e-9);
    }

    @Test
    void dmlArraysStable() {
        int[] y = DmlArrays.classIndices(new String[] {"b", "a", "b"});
        assertArrayEquals(new int[] {0, 1, 0}, y);
    }

    @Test
    void itmlProducesFiniteDistance() {
        double[][] raw = {{0, 0}, {1, 0}, {0, 3}, {2, 2}, {3, 3}, {4, 1}};
        IMatrix<Double> x = IMatrix.of(raw);
        String[] lb = {"a", "a", "a", "b", "b", "b"};
        var m = new ItmlDml().setMaxIter(80).setGamma(1.0)
                .setPriorKind(ItmlDml.PriorKind.IDENTITY).fit(x, lb);
        assertEquals(2, m.inputDimension());
        assertEquals(MetricForm.FULL_WHITENING, m.form());
        double d1 = m.squaredDistance(Linalg.vector(raw[0]), Linalg.vector(raw[1]));
        assertTrue(d1 >= 0 && Double.isFinite(d1));
    }

    @Test
    void ncaIncreasesSoftmaxMassOnSameClass() {
        double[][] raw = {
                {0.0, 0.0}, {0.2, 0.1}, {5.0, 5.0}, {5.1, 4.9},
        };
        IMatrix<Double> x = IMatrix.of(raw);
        String[] lb = {"0", "0", "1", "1"};
        int[] y = {0, 0, 1, 1};
        double before = ncaObjective(raw, y, identityL(2, 2));
        var fit = new NcaDml().setRank(2).setMaxIter(80).setLearningRate(0.08).fit(x, lb);
        var fitFacade = com.yishape.lab.math.ml.ML.dml.nca(2, 80, 0.08).fit(x, lb);
        assertEquals(fit.outputDimension(), fitFacade.outputDimension());
        double[][] L = toRowMajor(fit.transformMatrix());
        double after = ncaObjective(raw, y, L);
        assertTrue(after >= before - 1e-6);
    }

    @Test
    void mcmlKlDecreasesOrStable() {
        double[][] raw = {
                {0.0, 0.0}, {0.3, 0.2}, {4.0, 4.0}, {4.2, 3.9},
        };
        IMatrix<Double> x = IMatrix.of(raw);
        String[] lb = {"0", "0", "1", "1"};
        int[] y = {0, 0, 1, 1};
        int n = 4;
        double[][] p0 = new double[n][n];
        double[][] q = new double[n][n];
        MetricEmbeddingOps.mcmlTargetConditional(y, n, q);
        double[][] emb0 = MetricEmbeddingOps.embed(raw, n, 2, identityL(2, 2), 2);
        double[][] d0 = MetricEmbeddingOps.pairwiseSquaredDistances(emb0, n, 2);
        MetricEmbeddingOps.softmaxConditionalFromNegSqDist(d0, n, p0);
        double kl0 = MetricEmbeddingOps.mcmlKlLoss(p0, q, n);

        var fit = new McmlDml().setRank(2).setMaxIter(100).setLearningRate(0.06).fit(x, lb);
        double[][] L = toRowMajor(fit.transformMatrix());
        double[][] emb = MetricEmbeddingOps.embed(raw, n, 2, L, 2);
        double[][] d = MetricEmbeddingOps.pairwiseSquaredDistances(emb, n, 2);
        double[][] p = new double[n][n];
        MetricEmbeddingOps.softmaxConditionalFromNegSqDist(d, n, p);
        double kl1 = MetricEmbeddingOps.mcmlKlLoss(p, q, n);
        assertTrue(kl1 <= kl0 + 0.15);
    }

    @Test
    void lmnnAndLdmlRunFinite() {
        double[][] raw = {{0, 0}, {0.1, 0}, {3, 3}, {3, 3.1}};
        IMatrix<Double> x = IMatrix.of(raw);
        String[] lb = {"0", "0", "1", "1"};
        ISupervisedDml lmnnStyle = new LmnnDml()
                .setRank(2).setMaxSteps(400).setLearningRate(0.03);
        var l1 = lmnnStyle.fit(x, lb);
        var l2 = new LdmlPairwiseDml().setRank(2).setMaxSteps(800).fit(x, lb);
        assertTrue(l1.form() == MetricForm.LOW_RANK && l1.outputDimension() >= 1);
        assertTrue(l2.form() == MetricForm.LOW_RANK && l2.outputDimension() >= 1);
    }

    private static double ncaObjective(double[][] x, int[] y, double[][] L) {
        int n = x.length;
        int d = x[0].length;
        int r = L.length;
        boolean[][] same = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                same[i][j] = i != j && y[i] == y[j];
            }
        }
        double[][] emb = MetricEmbeddingOps.embed(x, n, d, L, r);
        double[][] distSq = MetricEmbeddingOps.pairwiseSquaredDistances(emb, n, r);
        double[][] p = new double[n][n];
        MetricEmbeddingOps.softmaxConditionalFromNegSqDist(distSq, n, p);
        return MetricEmbeddingOps.ncaLoss(p, same, n);
    }

    private static double[][] identityL(int r, int d) {
        double[][] L = new double[r][d];
        for (int i = 0; i < r; i++) {
            L[i][i] = 1.0;
        }
        return L;
    }

    private static double[][] toRowMajor(IMatrix<Double> m) {
        int r = m.getRowNum();
        int c = m.getColNum();
        double[][] o = new double[r][c];
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                o[i][j] = m.get(i, j);
            }
        }
        return o;
    }
}
