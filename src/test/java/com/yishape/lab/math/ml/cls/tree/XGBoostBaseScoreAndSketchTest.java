package com.yishape.lab.math.ml.cls.tree;

import com.yishape.lab.math.ml.clf.tree.XgbWeightedQuantileSketch;
import com.yishape.lab.math.ml.clf.tree.XGBoostTreeMethod;
import com.yishape.lab.math.ml.clf.tree.XGBoostHistogramBinning;
import com.yishape.lab.math.ml.clf.tree.XGTree;
import com.yishape.lab.math.ml.clf.tree.RereXGboost;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * {@link RereXGboost#resolveBaseMargins 初始 margin（base_score）} 与
 * {@link XgbWeightedQuantileSketch Hessian 加权分位数直方图} 的行为测试。
 */
public class XGBoostBaseScoreAndSketchTest {

    @Test
    void defaults_histogramSketchQuantileAndHistTreeMethod() {
        RereXGboost m = new RereXGboost();
        assertEquals(XGBoostTreeMethod.HIST, m.getTreeMethod());
        assertEquals(XGBoostHistogramBinning.QUANTILE_WEIGHTED_SKETCH, m.getHistogramBinning());
        assertNull(m.getBaseMarginOverride());
    }

    @Test
    void weightedSketch_edgesAreStrictlyIncreasingPerColumn() {
        double[][] raw = new double[200][2];
        for (int i = 0; i < 200; i++) {
            raw[i][0] = i * 0.01;
            raw[i][1] = (i % 17) * 0.13;
        }
        IMatrix<Double> X = Linalg.matrix(raw);
        int[] rows = new int[200];
        for (int i = 0; i < 200; i++) {
            rows[i] = i;
        }
        IVector hess = Linalg.ones(200);
        int maxBin = 16;
        double[][] edges = XgbWeightedQuantileSketch.buildEdges(X, hess, rows, maxBin);
        assertEquals(2, edges.length);
        assertEquals(maxBin + 1, edges[0].length);
        for (double[] e : edges) {
            for (int b = 1; b < e.length; b++) {
                assertTrue(e[b] > e[b - 1], "edges must increase: " + Arrays.toString(e));
            }
        }
    }

    @Test
    void weightedSketch_heavierTailTowardsLargeValues_shiftsInteriorBins() {
        int n = 100;
        double[][] raw = new double[n][1];
        for (int i = 0; i < n; i++) {
            raw[i][0] = i;
        }
        IMatrix<Double> X = Linalg.matrix(raw);
        int[] rows = new int[n];
        for (int i = 0; i < n; i++) {
            rows[i] = i;
        }
        double[] hw = new double[n];
        Arrays.fill(hw, 1.0);
        for (int i = 80; i < n; i++) {
            hw[i] = 20.0;
        }
        IVector hessHeavyRight = Linalg.vector(hw);
        IVector hessUniform = Linalg.ones(n);

        int maxBin = 8;
        double[][] eUniform = XgbWeightedQuantileSketch.buildEdges(X, hessUniform, rows, maxBin);
        double[][] eHeavy = XgbWeightedQuantileSketch.buildEdges(X, hessHeavyRight, rows, maxBin);

        double midUniform = eUniform[0][maxBin / 2];
        double midHeavy = eHeavy[0][maxBin / 2];
        assertTrue(midHeavy > midUniform,
                "more Hessian mass on large feature values should push quantiles right");
    }

    @Test
    void binaryImbalanced_autoBaseMarginLowersFirstBoostingLossVersusZeroMargin() {
        double[][] data = new double[120][2];
        String[] labels = new String[120];
        for (int i = 0; i < 120; i++) {
            data[i][0] = Math.sin(i * 0.07);
            data[i][1] = Math.cos(i * 0.05);
            labels[i] = i % 10 == 0 ? "N" : "P";
        }
        IMatrix<Double> X = Linalg.matrix(data);

        RereXGboost auto = new RereXGboost();
        auto.setHistogramBinning(XGBoostHistogramBinning.UNIFORM);
        auto.setNumEstimators(1);
        auto.setMaxDepth(3);
        auto.setLearningRate(0.3);
        auto.setLambda(1.0);
        auto.setEarlyStopping(false);
        auto.setValidationFraction(0.0);
        auto.setRandomSeed(1);
        auto.fit(X, labels);

        RereXGboost zeroInit = new RereXGboost();
        zeroInit.setHistogramBinning(XGBoostHistogramBinning.UNIFORM);
        zeroInit.setBaseMargin(0.0);
        zeroInit.setNumEstimators(1);
        zeroInit.setMaxDepth(3);
        zeroInit.setLearningRate(0.3);
        zeroInit.setLambda(1.0);
        zeroInit.setEarlyStopping(false);
        zeroInit.setValidationFraction(0.0);
        zeroInit.setRandomSeed(1);
        zeroInit.fit(X, labels);

        double lAuto = auto.getTrainLossHistory().get(0);
        double lZero = zeroInit.getTrainLossHistory().get(0);
        assertTrue(lAuto < lZero,
                "auto logit prior should reduce initial loss when labels imbalanced: auto=" + lAuto + " zero=" + lZero);
    }

    @Test
    void binaryBalanced_autoMarginNearZeroLossSimilarToManualZero() {
        double[][] data = {
                {0, 0}, {1, 0}, {0, 1}, {1, 1}, {2, 2}, {3, 3}
        };
        String[] labels = {"A", "A", "A", "B", "B", "B"};
        IMatrix<Double> X = Linalg.matrix(data);

        RereXGboost auto = new RereXGboost();
        auto.setHistogramBinning(XGBoostHistogramBinning.UNIFORM);
        auto.setNumEstimators(2);
        auto.setRandomSeed(4);
        auto.setEarlyStopping(false);
        auto.setValidationFraction(0.0);
        auto.fit(X, labels);

        RereXGboost manualZero = new RereXGboost();
        manualZero.setHistogramBinning(XGBoostHistogramBinning.UNIFORM);
        manualZero.setBaseMargin(0.0);
        manualZero.setNumEstimators(2);
        manualZero.setRandomSeed(4);
        manualZero.setEarlyStopping(false);
        manualZero.setValidationFraction(0.0);
        manualZero.fit(X, labels);

        double lAuto = auto.getTrainLossHistory().get(0);
        double lZero = manualZero.getTrainLossHistory().get(0);
        assertEquals(lAuto, lZero, 1e-9);
    }

    @Test
    void baseMarginOverrideWrongLength_throwsOnFit() {
        double[][] data = {{1.0}, {2.0}};
        String[] labels = {"a", "b"};
        IMatrix<Double> X = Linalg.matrix(data);
        RereXGboost m = new RereXGboost();
        m.setBaseMargin(new double[]{0.0, 0.0});
        assertThrows(IllegalArgumentException.class, () -> m.fit(X, labels));
    }

    @Test
    void quantileHistVsUniformHist_bothTrainBinarySeparable() {
        double[][] raw = new double[80][3];
        String[] y = new String[80];
        for (int i = 0; i < 80; i++) {
            raw[i][0] = i * 0.2;
            raw[i][1] = i % 7;
            raw[i][2] = Math.sqrt(i + 1);
            y[i] = raw[i][0] > 7.5 ? "Y" : "X";
        }
        IMatrix<Double> X = Linalg.matrix(raw);

        RereXGboost q = new RereXGboost();
        q.setHistogramBinning(XGBoostHistogramBinning.QUANTILE_WEIGHTED_SKETCH);
        q.setMaxBin(32);
        q.setNumEstimators(40);
        q.setMaxDepth(5);
        q.setLearningRate(0.25);
        q.setLambda(0.5);
        q.setRandomSeed(90);
        q.setEarlyStopping(false);
        q.setValidationFraction(0.0);
        q.fit(X, y);
        assertAccuracyAtLeast(q, X, y, 0.92);

        RereXGboost u = new RereXGboost();
        u.setHistogramBinning(XGBoostHistogramBinning.UNIFORM);
        u.setMaxBin(32);
        u.setNumEstimators(40);
        u.setMaxDepth(5);
        u.setLearningRate(0.25);
        u.setLambda(0.5);
        u.setRandomSeed(90);
        u.setEarlyStopping(false);
        u.setValidationFraction(0.0);
        u.fit(X, y);
        assertAccuracyAtLeast(u, X, y, 0.92);
    }

    @Test
    void multiclassSketch_highTrainAccuracy() {
        double[][] data = new double[60][2];
        String[] labels = new String[60];
        for (int i = 0; i < 60; i++) {
            data[i][0] = (i / 20) * 3.0 + 0.05 * Math.sin(i);
            data[i][1] = (i / 20) * 2.0 + 0.05 * Math.cos(i);
            labels[i] = String.valueOf((char) ('U' + (i / 20)));
        }
        IMatrix<Double> X = Linalg.matrix(data);
        RereXGboost m = new RereXGboost();
        m.setHistogramBinning(XGBoostHistogramBinning.QUANTILE_WEIGHTED_SKETCH);
        m.setMaxBin(48);
        m.setNumEstimators(60);
        m.setMaxDepth(5);
        m.setLearningRate(0.2);
        m.setLambda(0.3);
        m.setRandomSeed(77);
        m.setEarlyStopping(false);
        m.setValidationFraction(0.0);
        m.fit(X, labels);
        assertAccuracyAtLeast(m, X, labels, 0.88);
        IMatrix<Double> prob = m.predictProba(X);
        assertEquals(60, prob.rows());
        assertEquals(3, prob.cols());
        for (int i = 0; i < 60; i++) {
            double s = 0;
            for (int j = 0; j < 3; j++) {
                s += prob.get(i, j);
            }
            assertEquals(1.0, s, 1e-4);
        }
    }

    @Test
    void xgTree_histSketchConstructor_setsBinning() {
        XGTree t = new XGTree(4, 2, 1, 0.0, 1.0, 0.2,
                XGBoostTreeMethod.HIST, 16, 0.0, 0.0, XGBoostHistogramBinning.QUANTILE_WEIGHTED_SKETCH);
        assertEquals(XGBoostHistogramBinning.QUANTILE_WEIGHTED_SKETCH, t.getHistogramBinning());
    }

    private static void assertAccuracyAtLeast(RereXGboost model, IMatrix<Double> X, String[] y, double minAcc) {
        String[] pred = model.predictBatch(X);
        int c = 0;
        for (int i = 0; i < pred.length; i++) {
            if (pred[i].equals(y[i])) {
                c++;
            }
        }
        double acc = (double) c / pred.length;
        assertTrue(acc >= minAcc, "acc=" + acc);
    }
}
