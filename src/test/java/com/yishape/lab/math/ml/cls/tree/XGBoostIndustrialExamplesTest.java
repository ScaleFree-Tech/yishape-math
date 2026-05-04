package com.yishape.lab.math.ml.cls.tree;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * XGBoost 工业向特性全覆盖测试：直方图/精确贪心、γ、min_child_weight、行列采样、可复现性与典型数据集场景。
 * <p>
 * 这些用例同时可作为 API 使用示例（工厂参数组合、期望行为）。
 * </p>
 */
public class XGBoostIndustrialExamplesTest {

    private static double accuracy(String[] pred, String[] truth) {
        int c = 0;
        for (int i = 0; i < pred.length; i++) {
            if (pred[i].equals(truth[i])) {
                c++;
            }
        }
        return pred.length == 0 ? 0.0 : (double) c / pred.length;
    }

    @Test
    void defaultTreeMethodIsHistogram() {
        RereXGboost m = new RereXGboost();
        assertEquals(XGBoostTreeMethod.HIST, m.getTreeMethod());
        assertEquals(XGBoostHistogramBinning.QUANTILE_WEIGHTED_SKETCH, m.getHistogramBinning());
        assertEquals(256, m.getMaxBin());
        assertEquals(1.0, m.getSubsample(), 1e-12);
        assertEquals(1.0, m.getColsampleBytree(), 1e-12);
    }

    @Test
    void exactAndHistBothTrainOnLinearSeparableBinary() {
        double[][] data = {
                {0.0, 0.0}, {0.2, 0.1}, {0.1, 0.3},
                {3.0, 3.0}, {3.2, 3.1}, {3.1, 3.4}
        };
        String[] labels = {"N", "N", "N", "P", "P", "P"};
        IMatrix<Double> X = Linalg.matrix(data);

        for (XGBoostTreeMethod method : XGBoostTreeMethod.values()) {
            RereXGboost model = new RereXGboost();
            model.setTreeMethod(method);
            model.setLearningRate(0.3);
            model.setNumEstimators(40);
            model.setMaxDepth(4);
            model.setLambda(1.0);
            model.setGamma(0.0);
            model.setMinChildWeight(0.0);
            model.setRandomSeed(7);
            model.setEarlyStopping(false);
            model.setValidationFraction(0.0);

            model.fit(X, labels);
            String[] pred = model.predictBatch(X);
            assertTrue(accuracy(pred, labels) >= 0.99,
                    "method=" + method + " acc=" + accuracy(pred, labels));
            assertBinaryProbaSumOne(model, X);
        }
    }

    /**
     * 严格 XOR 角点在 logits=0、两类平衡时，单独按 x1/x2 的边际分裂常有 Σg=0 → 增益为 0，γ=0 时不分裂（与 Chen et al. 目标公式一致）。
     * 工业测试使用<strong>边际不平衡</strong>的 XOR 布局（对角重复更多），打破 Σg=0，使 EXACT 与 HIST 均能学到交互。
     */
    @Test
    void xorBiasedReplicas_exactAndHistReachHighTrainAccuracy() {
        double[][] corners = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        String[] cornerLabels = {"A", "B", "B", "A"};
        /* (0,0)(1,1) 为 A，重复更多；(0,1)(1,0) 为 B，较少 → 单侧边际上 A/B 不均，首轮即有正增益分裂 */
        int[] replicasPerCorner = {25, 8, 8, 25};
        int n = 0;
        for (int k = 0; k < 4; k++) {
            n += replicasPerCorner[k];
        }
        double[][] raw = new double[n][2];
        String[] y = new String[n];
        int r = 0;
        for (int k = 0; k < 4; k++) {
            for (int j = 0; j < replicasPerCorner[k]; j++) {
                raw[r][0] = corners[k][0] + 1e-6 * j;
                raw[r][1] = corners[k][1] + 1e-6 * j;
                y[r] = cornerLabels[k];
                r++;
            }
        }
        IMatrix<Double> X = Linalg.matrix(raw);

        for (XGBoostTreeMethod method : XGBoostTreeMethod.values()) {
            RereXGboost model = new RereXGboost();
            model.setTreeMethod(method);
            model.setMaxBin(64);
            model.setLearningRate(0.45);
            model.setNumEstimators(220);
            model.setMaxDepth(10);
            model.setLambda(0.05);
            model.setMinSamplesLeaf(1);
            model.setGamma(0.0);
            model.setMinChildWeight(0.0);
            model.setRandomSeed(123);
            model.setEarlyStopping(false);
            model.setValidationFraction(0.0);

            model.fit(X, y);
            double acc = accuracy(model.predictBatch(X), y);
            assertTrue(acc >= 0.92, "biased XOR train acc too low for method=" + method + ": " + acc);
        }
    }

    /** 边际线性可分：EXACT 与 HIST 均应在首轮即有正增益分裂。 */
    @Test
    void singleThresholdSeparable_exactAndHistReachHighTrainAccuracy() {
        double[][] raw = new double[120][2];
        String[] y = new String[120];
        for (int i = 0; i < 120; i++) {
            raw[i][0] = i * 0.05;
            raw[i][1] = Math.sin(i * 0.1);
            y[i] = raw[i][0] > 3.0 ? "P" : "N";
        }
        IMatrix<Double> X = Linalg.matrix(raw);

        for (XGBoostTreeMethod method : XGBoostTreeMethod.values()) {
            RereXGboost model = new RereXGboost();
            model.setTreeMethod(method);
            model.setMaxBin(64);
            model.setLearningRate(0.3);
            model.setNumEstimators(50);
            model.setMaxDepth(5);
            model.setLambda(1.0);
            model.setGamma(0.0);
            model.setMinChildWeight(0.0);
            model.setRandomSeed(21);
            model.setEarlyStopping(false);
            model.setValidationFraction(0.0);

            model.fit(X, y);
            double acc = accuracy(model.predictBatch(X), y);
            assertTrue(acc >= 0.95, "1D-threshold separable acc too low for method=" + method + ": " + acc);
        }
    }

    @Test
    void gammaVeryLargeProducesMinimalStructure() {
        double[][] data = {{1.0}, {2.0}, {3.0}, {4.0}};
        String[] labels = {"L", "L", "R", "R"};
        IMatrix<Double> X = Linalg.matrix(data);

        RereXGboost model = new RereXGboost();
        model.setTreeMethod(XGBoostTreeMethod.EXACT);
        model.setGamma(1e30);
        model.setLearningRate(1.0);
        model.setNumEstimators(5);
        model.setMaxDepth(8);
        model.setEarlyStopping(false);
        model.setValidationFraction(0.0);
        model.setRandomSeed(0);

        XGBoostResult res = (XGBoostResult) model.fit(X, labels);
        int leaves = 0;
        for (XGTree t : res.getTrees()) {
            leaves += t.getLeafCount();
        }
        assertTrue(leaves <= 5 * 2,
                "Huge gamma should forbid splits → roughly one leaf per tree; leaves=" + leaves);
    }

    @Test
    void rowAndColumnSubsampleTrainingCompletes() {
        double[][] data = new double[80][6];
        String[] labels = new String[80];
        java.util.Random r = new java.util.Random(99);
        for (int i = 0; i < 80; i++) {
            for (int j = 0; j < 6; j++) {
                data[i][j] = r.nextGaussian();
            }
            labels[i] = data[i][0] + data[i][1] > 0 ? "P" : "N";
        }
        IMatrix<Double> X = Linalg.matrix(data);

        RereXGboost model = new RereXGboost();
        model.setTreeMethod(XGBoostTreeMethod.HIST);
        model.setMaxBin(32);
        model.setSubsample(0.7);
        model.setColsampleBytree(0.6);
        model.setNumEstimators(25);
        model.setMaxDepth(4);
        model.setRandomSeed(42);
        model.setEarlyStopping(false);
        model.setValidationFraction(0.0);

        assertDoesNotThrow(() -> model.fit(X, labels));
        assertBinaryProbaSumOne(model, X);
    }

    @Test
    void deterministicPredictionsWithFixedSeedFullBatch() {
        double[][] data = {{1, 2}, {2, 1}, {3, 3}, {2, 2}, {4, 1}};
        String[] labels = {"a", "a", "b", "b", "b"};
        IMatrix<Double> X = Linalg.matrix(data);

        String[] p1 = trainAndPredict(X, labels, 999);
        String[] p2 = trainAndPredict(X, labels, 999);
        assertArrayEquals(p1, p2);
    }

    private static String[] trainAndPredict(IMatrix<Double> X, String[] labels, long seed) {
        RereXGboost model = new RereXGboost();
        model.setRandomSeed(seed);
        model.setTreeMethod(XGBoostTreeMethod.HIST);
        model.setSubsample(0.8);
        model.setColsampleBytree(0.9);
        model.setNumEstimators(30);
        model.setMaxDepth(4);
        model.setEarlyStopping(false);
        model.setValidationFraction(0.0);
        model.fit(X, labels);
        return model.predictBatch(X);
    }

    @Test
    void multiclassThreeClusters_highAccuracy() {
        double[][] data = new double[90][2];
        String[] labels = new String[90];
        int idx = 0;
        for (int k = 0; k < 3; k++) {
            double cx = k * 5.0;
            double cy = k * 4.0;
            for (int i = 0; i < 30; i++, idx++) {
                data[idx][0] = cx + 0.1 * Math.sin(i);
                data[idx][1] = cy + 0.1 * Math.cos(i);
                labels[idx] = String.valueOf((char) ('A' + k));
            }
        }
        IMatrix<Double> X = Linalg.matrix(data);

        RereXGboost model = new RereXGboost();
        model.setTreeMethod(XGBoostTreeMethod.HIST);
        model.setMaxBin(128);
        model.setLearningRate(0.2);
        model.setNumEstimators(80);
        model.setMaxDepth(5);
        model.setRandomSeed(11);
        model.setEarlyStopping(false);
        model.setValidationFraction(0.0);

        model.fit(X, labels);
        assertTrue(accuracy(model.predictBatch(X), labels) >= 0.95);

        IMatrix<Double> prob = model.predictProba(X);
        assertEquals(90, prob.rows());
        assertEquals(3, prob.cols());
        for (int i = 0; i < prob.rows(); i++) {
            double s = 0;
            for (int j = 0; j < prob.cols(); j++) {
                s += prob.get(i, j).doubleValue();
            }
            assertEquals(1.0, s, 1e-5);
        }
    }

    @Test
    void minChildWeightConfigurable_TrainingStable() {
        double[][] data = new double[40][1];
        String[] labels = new String[40];
        for (int i = 0; i < 40; i++) {
            data[i][0] = i * 0.1;
            labels[i] = i < 20 ? "L" : "R";
        }
        IMatrix<Double> X = Linalg.matrix(data);

        RereXGboost constrained = new RereXGboost();
        constrained.setTreeMethod(XGBoostTreeMethod.EXACT);
        constrained.setMinChildWeight(8.0);
        constrained.setGamma(0.0);
        constrained.setNumEstimators(8);
        constrained.setMaxDepth(10);
        constrained.setEarlyStopping(false);
        constrained.setValidationFraction(0.0);
        constrained.setRandomSeed(3);

        assertDoesNotThrow(() -> constrained.fit(X, labels));
        assertNotNull(constrained.predictBatch(X));
    }

    @Test
    void histSmallMaxBinStillRuns() {
        double[][] data = {{0.1}, {0.2}, {5.0}, {5.1}};
        String[] labels = {"a", "a", "b", "b"};
        IMatrix<Double> X = Linalg.matrix(data);

        RereXGboost model = new RereXGboost();
        model.setTreeMethod(XGBoostTreeMethod.HIST);
        model.setMaxBin(8);
        model.setNumEstimators(15);
        model.setMaxDepth(4);
        model.setEarlyStopping(false);
        model.setValidationFraction(0.0);
        assertDoesNotThrow(() -> model.fit(X, labels));
        assertEquals(1.0, accuracy(model.predictBatch(X), labels), 0.01);
    }

    @Test
    void xgTreeConstructorsPreserveBackwardCompat() {
        XGTree legacy = new XGTree(3, 2, 1, 0.0, 1.0, 0.1);
        assertEquals(XGBoostTreeMethod.EXACT, legacy.getTreeMethod());

        XGTree industrial = new XGTree(3, 2, 1, 0.0, 1.0, 0.1,
                XGBoostTreeMethod.HIST, 64, 0.1, 1.0, XGBoostHistogramBinning.UNIFORM);
        assertEquals(XGBoostTreeMethod.HIST, industrial.getTreeMethod());
        assertEquals(64, industrial.getMaxBin());
        assertEquals(0.1, industrial.getGamma(), 1e-12);
        assertEquals(1.0, industrial.getMinChildWeight(), 1e-12);
        assertEquals(XGBoostHistogramBinning.UNIFORM, industrial.getHistogramBinning());
    }

    private static void assertBinaryProbaSumOne(RereXGboost model, IMatrix<Double> X) {
        IMatrix<Double> prob = model.predictProba(X);
        assertEquals(2, prob.cols());
        for (int i = 0; i < prob.rows(); i++) {
            double s = prob.get(i, 0).doubleValue() + prob.get(i, 1).doubleValue();
            assertEquals(1.0, s, 1e-5);
        }
    }
}
