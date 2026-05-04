package com.yishape.lab.math.ml.cls.tree;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.ML;
import com.yishape.lab.math.ml.cls.IClassifier;
import com.yishape.lab.math.ml.cls.svm.RereLinearSVM;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * {@link RereDecisionTree} 与 {@link RereLinearSVM} 行为测试。
 */
public class DecisionTreeLinearSvmTest {

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

    @Test
    void cartGiniSeparatesTwoBlobs() {
        double[][] raw = {
                {0, 0}, {0.5, 0.2}, {0.2, 0.8},
                {5, 5}, {5.2, 4.8}, {4.7, 5.3}
        };
        String[] y = {"A", "A", "A", "B", "B", "B"};
        IMatrix<Double> X = Linalg.matrix(raw);
        RereDecisionTree tree = new RereDecisionTree(DecisionTreeCriterion.CART_GINI, 8, 2, 1);
        tree.fit(X, y);
        assertEquals(1.0, accuracy(tree, X, y), 1e-9);
        assertTrue(tree.isTrained());
        assertNotNull(tree.predictProb(X.getRow(0)));
    }

    @Test
    void cartEntropyAndC45TrainWithoutThrow() {
        double[][] raw = {{1}, {2}, {5}, {6}};
        String[] y = {"L", "L", "R", "R"};
        IMatrix<Double> X = Linalg.matrix(raw);
        RereDecisionTree t1 = new RereDecisionTree(DecisionTreeCriterion.CART_ENTROPY, 5, 2, 1);
        t1.fit(X, y);
        assertEquals(1.0, accuracy(t1, X, y), 1e-9);

        RereDecisionTree t2 = new RereDecisionTree(DecisionTreeCriterion.C45_GAIN_RATIO, 5, 2, 1);
        t2.fit(X, y);
        assertEquals(1.0, accuracy(t2, X, y), 1e-9);
    }

    @Test
    void mlFactoriesConstructModels() {
        assertTrue(ML.decisionTree() instanceof RereDecisionTree);
        assertTrue(ML.linearSvm() instanceof RereLinearSVM);
        assertTrue(ML.decisionTree(DecisionTreeCriterion.C45_GAIN_RATIO, 4, 2, 1) instanceof RereDecisionTree);
    }

    @Test
    void linearSvmBinaryLinearSeparable() {
        double[][] raw = new double[40][2];
        String[] y = new String[40];
        for (int i = 0; i < 40; i++) {
            raw[i][0] = i < 20 ? 0.1 * i : 3 + 0.1 * (i - 20);
            raw[i][1] = i < 20 ? 0 : 2;
            y[i] = i < 20 ? "N" : "P";
        }
        IMatrix<Double> X = Linalg.matrix(raw);
        RereLinearSVM svm = new RereLinearSVM(2.0, true);
        svm.fit(X, y);
        assertTrue(accuracy(svm, X, y) >= 0.95);
        double s = 0;
        for (double v : svm.predictProb(X.getRow(0)).values()) {
            s += v;
        }
        assertEquals(1.0, s, 1e-5);
    }

    @Test
    void linearSvmThreeClasses() {
        double[][] raw = new double[60][2];
        String[] y = new String[60];
        for (int i = 0; i < 60; i++) {
            int k = i / 20;
            raw[i][0] = k * 4.0 + 0.05 * Math.sin(i);
            raw[i][1] = k * 3.0 + 0.05 * Math.cos(i);
            y[i] = String.valueOf((char) ('a' + k));
        }
        IMatrix<Double> X = Linalg.matrix(raw);
        RereLinearSVM svm = new RereLinearSVM(1.5, true);
        svm.fit(X, y);
        assertTrue(accuracy(svm, X, y) >= 0.85);
    }
}
