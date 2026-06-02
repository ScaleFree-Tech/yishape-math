package com.yishape.lab.math.ml.cls.tree;

import com.yishape.lab.math.ml.clf.tree.DecisionTreeResult;
import com.yishape.lab.math.ml.clf.tree.DecisionTreeCriterion;
import com.yishape.lab.math.ml.clf.tree.RereDecisionTree;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.ISerializableModel;
import com.yishape.lab.math.ml.clf.BatchPredResult;
import com.yishape.lab.math.ml.clf.IClassifier;
import com.yishape.lab.math.ml.clf.svm.RereLinearSVM;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * {@link RereDecisionTree}、{@link RereLinearSVM} 的边界条件、非线性可分、序列化与概率一致性等扩展测试。
 */
public class DecisionTreeLinearSvmExtendedTest {

    private static double accuracy(IClassifier m, IMatrix<Double> X, String[] y) {
        String[] p = m.predictBatch(X);
        int c = 0;
        for (int i = 0; i < y.length; i++) {
            if (p[i].equals(y[i])) {
                c++;
            }
        }
        return (double) c / y.length;
    }

    /**
     * 横向三色条纹（低 y=A，中 y=B，高 y=A）：轴平行树可分；对称 XOR 在轴平行单次分裂下增益常为 0，不适合用来断言「树必完美拟合 XOR」。
     */
    private static IMatrix<Double> horizontalStripeNine(String[] yOut) {
        assert yOut.length >= 9;
        double[][] raw = {
                {0.0, 0.2}, {1.0, 0.25}, {2.0, 0.22},
                {0.0, 1.5}, {1.0, 1.48}, {2.0, 1.52},
                {0.0, 2.85}, {1.0, 2.88}, {2.0, 2.82}
        };
        for (int i = 0; i < 3; i++) {
            yOut[i] = "A";
        }
        for (int i = 3; i < 6; i++) {
            yOut[i] = "B";
        }
        for (int i = 6; i < 9; i++) {
            yOut[i] = "A";
        }
        return Linalg.matrix(raw);
    }

    @Test
    void decisionTreeStripePerfectAllCriteria() {
        String[] y = new String[9];
        IMatrix<Double> X = horizontalStripeNine(y);
        for (DecisionTreeCriterion crit : DecisionTreeCriterion.values()) {
            RereDecisionTree tree = new RereDecisionTree(crit, 12, 2, 1);
            tree.fit(X, y);
            assertEquals(1.0, accuracy(tree, X, y), 1e-9, crit.name());
            assertTrue(tree.isTrained(), crit.name());
        }
    }

    @Test
    void decisionTreeMaxDepthOneInsufficientForStripe() {
        String[] y = new String[9];
        IMatrix<Double> X = horizontalStripeNine(y);
        RereDecisionTree tree = new RereDecisionTree(DecisionTreeCriterion.CART_GINI, 1, 2, 1);
        tree.fit(X, y);
        assertTrue(accuracy(tree, X, y) < 1.0);
    }

    @Test
    void decisionTreeLargeMinLeafBlocksPerfectStripe() {
        String[] y = new String[9];
        IMatrix<Double> X = horizontalStripeNine(y);
        RereDecisionTree tree = new RereDecisionTree(DecisionTreeCriterion.CART_GINI, 20, 2, 5);
        tree.fit(X, y);
        assertTrue(accuracy(tree, X, y) < 1.0);
    }

    @Test
    void decisionTreeThreeClassStripe() {
        double[][] raw = new double[90][2];
        String[] y = new String[90];
        for (int i = 0; i < 90; i++) {
            int k = i / 30;
            raw[i][0] = k * 6.0 + (i % 30) * 0.02;
            raw[i][1] = k * 0.5 + 0.01 * Math.sin(i);
            y[i] = "c" + k;
        }
        IMatrix<Double> X = Linalg.matrix(raw);
        RereDecisionTree tree = new RereDecisionTree(DecisionTreeCriterion.C45_GAIN_RATIO, 16, 2, 1);
        tree.fit(X, y);
        DecisionTreeResult dr = tree.getResult();
        assertEquals(DecisionTreeCriterion.C45_GAIN_RATIO, dr.getCriterion());
        assertTrue(dr.getTreeDepth() >= 1);
        assertTrue(dr.getLeafCount() >= 2);
        assertEquals(1.0, accuracy(tree, X, y), 1e-9);
    }

    @Test
    void decisionTreePredictProbSumsToOneAndPureLeaf() {
        double[][] raw = {{0}, {1}, {5}, {6}};
        String[] y = {"L", "L", "R", "R"};
        IMatrix<Double> X = Linalg.matrix(raw);
        RereDecisionTree tree = new RereDecisionTree(DecisionTreeCriterion.CART_ENTROPY, 8, 2, 1);
        tree.fit(X, y);
        for (int i = 0; i < 4; i++) {
            Map<String, Double> pm = tree.predictProb(X.getRow(i));
            double s = pm.values().stream().mapToDouble(Double::doubleValue).sum();
            assertEquals(1.0, s, 1e-9);
            assertTrue(pm.get(y[i]) >= 0.99);
        }
    }

    @Test
    void decisionTreeBatchPredictMatchesRowPredict() {
        double[][] raw = {{1, 2}, {2, 1}, {3, 4}};
        String[] y = {"p", "q", "p"};
        IMatrix<Double> X = Linalg.matrix(raw);
        RereDecisionTree tree = new RereDecisionTree(DecisionTreeCriterion.CART_GINI, 6, 2, 1);
        tree.fit(X, y);
        String[] batch = tree.predictBatch(X);
        for (int i = 0; i < 3; i++) {
            assertEquals(batch[i], tree.predict(X.getRow(i)));
        }
        BatchPredResult bpr = tree.predictBatchWithProbs(X);
        assertFalse(bpr.isBinaryClassification());
        assertEquals(3, bpr.getSampleCount());
        for (int i = 0; i < 3; i++) {
            double rowSum = Arrays.stream(bpr.getClassProbabilities()[i]).sum();
            assertEquals(1.0, rowSum, 1e-9);
            assertEquals(batch[i], bpr.getPredictions()[i]);
        }
    }

    @Test
    void decisionTreeFitRejectsSingleClass() {
        double[][] raw = {{1}, {2}};
        String[] y = {"x", "x"};
        IMatrix<Double> X = Linalg.matrix(raw);
        RereDecisionTree tree = new RereDecisionTree();
        assertThrows(IllegalArgumentException.class, () -> tree.fit(X, y));
    }

    @Test
    void decisionTreeFitRejectsNull() {
        assertThrows(IllegalArgumentException.class,
                () -> new RereDecisionTree().fit(null, new String[]{"a", "b"}));
    }

    @Test
    void decisionTreeSerializationRoundTrip(@TempDir Path tmp) {
        double[][] raw = {{0, 0}, {5, 5}};
        String[] y = {"a", "b"};
        IMatrix<Double> X = Linalg.matrix(raw);
        RereDecisionTree tree = new RereDecisionTree(DecisionTreeCriterion.C45_GAIN_RATIO, 5, 2, 1);
        tree.fit(X, y);
        Path f = tmp.resolve("tree.bin");
        tree.save(f.toString());
        ISerializableModel loaded = ISerializableModel.load(f.toString());
        assertInstanceOf(RereDecisionTree.class, loaded);
        RereDecisionTree t2 = (RereDecisionTree) loaded;
        assertTrue(t2.isTrained());
        for (int i = 0; i < 2; i++) {
            assertEquals(tree.predict(X.getRow(i)), t2.predict(X.getRow(i)));
            assertEquals(tree.predictProb(X.getRow(i)), t2.predictProb(X.getRow(i)));
        }
    }

    @Test
    void linearSvmStripeBelowTreeAccuracy() {
        String[] y = new String[9];
        IMatrix<Double> X = horizontalStripeNine(y);
        RereDecisionTree tree = new RereDecisionTree(DecisionTreeCriterion.CART_GINI, 12, 2, 1);
        tree.fit(X, y);
        assertEquals(1.0, accuracy(tree, X, y), 1e-9);
        RereLinearSVM svm = new RereLinearSVM(3.0, true);
        svm.fit(X, y);
        assertTrue(accuracy(svm, X, y) < accuracy(tree, X, y) - 0.05,
                "线性模型难以完美划分 A-B-A 条纹区域");
    }

    @Test
    void linearSvmSeparableHighAccuracyStandardizeOnOff() {
        double[][] raw = new double[24][1];
        String[] ys = new String[24];
        for (int i = 0; i < 24; i++) {
            raw[i][0] = i < 12 ? -5 + i * 0.2 : 5 + i * 0.15;
            ys[i] = i < 12 ? "neg" : "pos";
        }
        IMatrix<Double> X = Linalg.matrix(raw);
        RereLinearSVM on = new RereLinearSVM(1.0, true);
        RereLinearSVM off = new RereLinearSVM(1.0, false);
        on.fit(X, ys);
        off.fit(X, ys);
        assertTrue(accuracy(on, X, ys) >= 0.95);
        assertTrue(accuracy(off, X, ys) >= 0.95);
    }

    @Test
    void linearSvmBinaryProbSumAndBatchMatrix() {
        double[][] raw = {{0, 0}, {0, 1}, {3, 0}, {3, 1}};
        String[] y = {"0", "0", "1", "1"};
        IMatrix<Double> X = Linalg.matrix(raw);
        RereLinearSVM svm = new RereLinearSVM(2.0, false);
        svm.fit(X, y);
        BatchPredResult bpr = svm.predictBatchWithProbs(X);
        assertFalse(bpr.isBinaryClassification());
        assertEquals(2, bpr.getClassProbabilities()[0].length);
        for (int i = 0; i < 4; i++) {
            Map<String, Double> pm = svm.predictProb(X.getRow(i));
            double s = pm.values().stream().mapToDouble(Double::doubleValue).sum();
            assertEquals(1.0, s, 1e-5);
            assertEquals(bpr.getPredictions()[i], svm.predict(X.getRow(i)));
            assertEquals(1.0, Arrays.stream(bpr.getClassProbabilities()[i]).sum(), 1e-5);
        }
    }

    @Test
    void linearSvmOvRProbRowsNormalize() {
        double[][] raw = new double[30][2];
        String[] y = new String[30];
        for (int i = 0; i < 30; i++) {
            int k = i / 10;
            raw[i][0] = k * 10.0 + i * 0.01;
            raw[i][1] = k - 0.5 * i * 0.01;
            y[i] = String.valueOf(k);
        }
        IMatrix<Double> X = Linalg.matrix(raw);
        RereLinearSVM svm = new RereLinearSVM(1.2, true);
        svm.fit(X, y);
        BatchPredResult bpr = svm.predictBatchWithProbs(X);
        assertFalse(bpr.isBinaryClassification());
        for (double[] row : bpr.getClassProbabilities()) {
            assertEquals(1.0, Arrays.stream(row).sum(), 1e-5);
        }
    }

    @Test
    void linearSvmFitRejectsSingleClass() {
        IMatrix<Double> X = Linalg.matrix(new double[][]{{1}, {2}});
        assertThrows(IllegalArgumentException.class,
                () -> new RereLinearSVM().fit(X, new String[]{"a", "a"}));
    }

    @Test
    void linearSvmSerializationRoundTrip(@TempDir Path tmp) {
        double[][] raw = {{0, 0}, {2, 2}, {2, 0}, {0, 2}};
        String[] y = {"A", "B", "B", "B"};
        IMatrix<Double> X = Linalg.matrix(raw);
        RereLinearSVM svm = new RereLinearSVM(1.5, true);
        svm.fit(X, y);
        Path f = tmp.resolve("svm.bin");
        svm.save(f.toString());
        ISerializableModel loaded = ISerializableModel.load(f.toString());
        assertInstanceOf(RereLinearSVM.class, loaded);
        RereLinearSVM s2 = (RereLinearSVM) loaded;
        assertTrue(s2.isTrained());
        for (int i = 0; i < 4; i++) {
            IVector<Double> row = X.getRow(i);
            assertEquals(svm.predict(row), s2.predict(row), "row " + i);
            Map<String, Double> p1 = svm.predictProb(row);
            Map<String, Double> p2 = s2.predictProb(row);
            assertEquals(p1.size(), p2.size());
            for (String k : p1.keySet()) {
                assertEquals(p1.get(k), p2.get(k), 1e-10);
            }
        }
    }

    @Test
    void linearSvmNaNFeatureRejected() {
        IMatrix<Double> X = Linalg.matrix(new double[][]{{Double.NaN}, {1.0}});
        assertThrows(IllegalArgumentException.class,
                () -> new RereLinearSVM().fit(X, new String[]{"a", "b"}));
    }

    @Test
    void decisionTreeFeatureImportanceNonNegativeAfterSplit() {
        double[][] raw = {{0}, {2}, {4}, {6}};
        String[] y = {"a", "a", "b", "b"};
        IMatrix<Double> X = Linalg.matrix(raw);
        RereDecisionTree tree = new RereDecisionTree(DecisionTreeCriterion.CART_GINI, 6, 2, 1);
        tree.fit(X, y);
        DecisionTreeResult dr = tree.getResult();
        assertNotNull(dr.getFeatureImportance());
        assertEquals(1, dr.getFeatureImportance().length());
        assertTrue(dr.getFeatureImportance().normInf() >= 0);
        assertTrue(dr.getLeafCount() >= 2);
    }

    @Test
    void linearSvmOverlappingGaussiansBothReasonableAccuracy() {
        double[][] raw = new double[80][2];
        String[] y = new String[80];
        Random rnd = new Random(42);
        for (int i = 0; i < 80; i++) {
            boolean left = i < 40;
            raw[i][0] = (left ? -1 : 2.5) + rnd.nextGaussian() * 0.8;
            raw[i][1] = rnd.nextGaussian() * 0.5;
            y[i] = left ? "L" : "R";
        }
        IMatrix<Double> X = Linalg.matrix(raw);
        RereLinearSVM loose = new RereLinearSVM(0.05, true);
        RereLinearSVM tight = new RereLinearSVM(50.0, true);
        loose.fit(X, y);
        tight.fit(X, y);
        assertTrue(accuracy(loose, X, y) >= 0.55);
        assertTrue(accuracy(tight, X, y) >= 0.55);
    }

    @Test
    void decisionTreeNoGainSkewsToMajority() {
        double[][] raw = new double[6][1];
        String[] y = {"m", "m", "m", "n", "n", "n"};
        for (int i = 0; i < 6; i++) {
            raw[i][0] = i * 0.01;
        }
        IMatrix<Double> X = Linalg.matrix(raw);
        RereDecisionTree tree = new RereDecisionTree(DecisionTreeCriterion.CART_GINI, 4, 10, 3);
        tree.fit(X, y);
        assertEquals(6, tree.predictBatch(X).length);
        assertTrue(tree.predictProb(X.getRow(0)).get("m") >= tree.predictProb(X.getRow(0)).get("n"));
    }

    @Test
    void linearSvmGetFeatureWeightsAlignedWithRegressionApi() {
        double[][] raw = {{0, 0}, {0, 1}, {3, 0}, {3, 1}};
        String[] y = {"0", "1", "1", "1"};
        IMatrix<Double> X = Linalg.matrix(raw);
        RereLinearSVM svm = new RereLinearSVM(2.0, false);
        svm.fit(X, y);
        assertTrue(svm.isBinaryTask());
        assertNotNull(svm.getFeatureWeights());
        assertEquals(2, svm.getFeatureWeights().length());
        svm.getBias();
        double[][] wOvR = svm.getFeatureWeightsOvR();
        assertEquals(1, wOvR.length);
        assertEquals(2, wOvR[0].length);
        assertEquals(1, svm.getBiasOvR().length);
    }

    @Test
    void linearSvmOvRWeightsShape() {
        double[][] raw = new double[12][2];
        String[] lab = new String[12];
        for (int i = 0; i < 12; i++) {
            int k = i / 4;
            raw[i][0] = k * 4.0 + i * 0.01;
            raw[i][1] = k;
            lab[i] = "c" + k;
        }
        IMatrix<Double> X = Linalg.matrix(raw);
        RereLinearSVM svm = new RereLinearSVM(1.0, true);
        svm.fit(X, lab);
        assertFalse(svm.isBinaryTask());
        assertNull(svm.getFeatureWeights());
        assertThrows(IllegalStateException.class, svm::getBias);
        double[][] w = svm.getFeatureWeightsOvR();
        assertEquals(3, w.length);
        assertEquals(2, w[0].length);
        assertEquals(3, svm.getBiasOvR().length);
    }

    @Test
    void linearSvmUntrainedPredictThrows() {
        RereLinearSVM svm = new RereLinearSVM();
        IVector<Double> x = Linalg.vector(new double[]{1, 2});
        assertThrows(IllegalStateException.class, () -> svm.predict(x));
    }

    @Test
    void decisionTreeUntrainedPredictThrows() {
        RereDecisionTree tree = new RereDecisionTree();
        IVector<Double> x = Linalg.vector(new double[]{1});
        assertThrows(IllegalStateException.class, () -> tree.predict(x));
    }
}
