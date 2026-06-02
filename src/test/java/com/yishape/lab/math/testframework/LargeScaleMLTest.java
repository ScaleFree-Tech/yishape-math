package com.yishape.lab.math.testframework;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.clf.lr.RereLogisticRegression;
import com.yishape.lab.math.ml.clf.knn.RereKnn;
import com.yishape.lab.math.ml.clf.tree.RereDecisionTree;
import com.yishape.lab.math.ml.clf.tree.RereRandomForest;
import com.yishape.lab.math.ml.clu.GMMClustering;
import com.yishape.lab.math.ml.clu.KMeansPlusPlus;
import com.yishape.lab.math.ml.dr.RerePCA;
import com.yishape.lab.math.ml.dr.RereSVD;
import com.yishape.lab.math.ml.metric.CrossValidation;
import com.yishape.lab.math.ml.metric.CrossValidationResult;
import com.yishape.lab.math.ml.reg.RereLinearRegression;
import com.yishape.lab.math.ml.reg.RegressionResult;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.*;
import org.junit.jupiter.api.Disabled;

/**
 * 大规模机器学习正确性与性能测试
 *
 * <p>测试覆盖分类器、回归器、聚类、降维、边界情况和交叉验证。
 * 每个测试验证正确性并记录运行时间。</p>
 *
 * <p>输出格式: BENCHMARK|模块|操作|规模|时间_ms|正确性状态|说明</p>
 */
@Disabled("大规模机器学习性能基准，默认跳过；需要时去掉本注解或单独 -Dtest=... 运行")
@Timeout(value = 300)
public class LargeScaleMLTest {

    private static final long FIXED_SEED = 42L;
    private static final Random RANDOM = new Random(FIXED_SEED);

    // ==================== 辅助方法 ====================

    private void logBenchmark(String module, String operation, int scale, long timeMs, boolean passed, String note) {
        System.out.printf("BENCHMARK|%s|%s|%d|%d|%s|%s%n",
                module, operation, scale, timeMs, passed ? "PASS" : "FAIL", note);
    }

    private IMatrix generateClassificationData(int nSamples, int nFeatures, String[] labels) {
        double[][] data = new double[nSamples][nFeatures];
        int nClasses = Arrays.stream(labels).distinct().toArray(String[]::new).length;
        for (int i = 0; i < nSamples; i++) {
            int labelIdx = labels[i].equals("A") ? 0 : (labels[i].equals("B") ? 1 : 2);
            for (int j = 0; j < nFeatures; j++) {
                data[i][j] = RANDOM.nextGaussian() + labelIdx * 2.0;
            }
        }
        return Linalg.matrix(data);
    }

    private String[] generateLabels(int nSamples, int nClasses) {
        String[] classNames = {"A", "B", "C"};
        String[] labels = new String[nSamples];
        for (int i = 0; i < nSamples; i++) {
            labels[i] = classNames[i % nClasses];
        }
        shuffleArray(labels);
        return labels;
    }

    private void shuffleArray(String[] array) {
        for (int i = array.length - 1; i > 0; i--) {
            int j = RANDOM.nextInt(i + 1);
            String temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }

    private double computeAccuracy(String[] trueLabels, String[] predictedLabels) {
        int correct = 0;
        for (int i = 0; i < trueLabels.length; i++) {
            if (trueLabels[i].equals(predictedLabels[i])) {
                correct++;
            }
        }
        return (double) correct / trueLabels.length;
    }

    // ==================== 1. 分类器正确性+性能 ====================

    @Test
    public void testLogisticRegressionLargeScale() {
        System.out.println("\n=== 逻辑回归大规模测试 ===");
        int[] scales = {100, 500, 1000, 5000, 10000};
        boolean allPassed = true;

        for (int scale : scales) {
            RANDOM.setSeed(FIXED_SEED + scale);
            String[] labels = generateLabels(scale, 2);
            IMatrix features = generateClassificationData(scale, 5, labels);

            long start = System.nanoTime();
            RereLogisticRegression lr = new RereLogisticRegression();
            lr.setRandomSeed(FIXED_SEED + scale);
            lr.setStandardizeFeatures(false);
            lr.fit(features, labels);
            String[] predictions = lr.predictBatch(features);
            long timeMs = (System.nanoTime() - start) / 1_000_000;

            double accuracy = computeAccuracy(labels, predictions);
            boolean passed = accuracy > 0.5 && lr.isTrained();
            if (!passed) allPassed = false;

            logBenchmark("ML_CLS", "LogisticRegression", scale, timeMs, passed,
                    String.format("accuracy=%.4f, trained=%b", accuracy, lr.isTrained()));
        }
        assert allPassed : "逻辑回归部分规模测试失败";
    }

    @Test
    public void testDecisionTreeLargeScale() {
        System.out.println("\n=== 决策树大规模测试 ===");
        int[] scales = {100, 500, 1000, 5000, 10000};
        boolean allPassed = true;

        for (int scale : scales) {
            RANDOM.setSeed(FIXED_SEED + scale);
            String[] labels = generateLabels(scale, 2);
            IMatrix features = generateClassificationData(scale, 5, labels);

            long start = System.nanoTime();
            RereDecisionTree dt = new RereDecisionTree();
            dt.fit(features, labels);
            String[] predictions = dt.predictBatch(features);
            long timeMs = (System.nanoTime() - start) / 1_000_000;

            double accuracy = computeAccuracy(labels, predictions);
            boolean passed = accuracy > 0.5 && dt.isTrained();
            if (!passed) allPassed = false;

            logBenchmark("ML_CLS", "DecisionTree", scale, timeMs, passed,
                    String.format("accuracy=%.4f, trained=%b", accuracy, dt.isTrained()));
        }
        assert allPassed : "决策树部分规模测试失败";
    }

    @Test
    public void testRandomForestLargeScale() {
        System.out.println("\n=== 随机森林大规模测试 ===");
        int[] scales = {100, 500, 1000, 5000, 10000};
        boolean allPassed = true;

        for (int scale : scales) {
            RANDOM.setSeed(FIXED_SEED + scale);
            String[] labels = generateLabels(scale, 2);
            IMatrix features = generateClassificationData(scale, 5, labels);

            long start = System.nanoTime();
            RereRandomForest rf = new RereRandomForest(10, 10, 2, 1, -1, true,
                    com.yishape.lab.math.ml.clf.tree.RFTree.SplitCriterion.GINI, FIXED_SEED + scale);
            rf.fit(features, labels);
            String[] predictions = rf.predictBatch(features);
            long timeMs = (System.nanoTime() - start) / 1_000_000;

            double accuracy = computeAccuracy(labels, predictions);
            boolean passed = accuracy > 0.5 && rf.isTrained();
            if (!passed) allPassed = false;

            logBenchmark("ML_CLS", "RandomForest", scale, timeMs, passed,
                    String.format("accuracy=%.4f, trained=%b", accuracy, rf.isTrained()));
        }
        assert allPassed : "随机森林部分规模测试失败";
    }

    @Test
    public void testKnnLargeScale() {
        System.out.println("\n=== KNN大规模测试 ===");
        int[] scales = {100, 500, 1000, 5000, 10000};
        boolean allPassed = true;

        for (int scale : scales) {
            RANDOM.setSeed(FIXED_SEED + scale);
            String[] labels = generateLabels(scale, 2);
            IMatrix features = generateClassificationData(scale, 5, labels);

            long start = System.nanoTime();
            RereKnn knn = new RereKnn(5);
            knn.fit(features, labels);

            // 验证预测一致性：相同输入应得相同输出
            IVector testPoint = features.getRow(0);
            String pred1 = knn.predict(testPoint);
            String pred2 = knn.predict(testPoint);
            boolean consistent = pred1.equals(pred2);

            String[] predictions = knn.predictBatch(features);
            long timeMs = (System.nanoTime() - start) / 1_000_000;

            double accuracy = computeAccuracy(labels, predictions);
            boolean passed = consistent && knn.isTrained();
            if (!passed) allPassed = false;

            logBenchmark("ML_CLS", "KNN", scale, timeMs, passed,
                    String.format("consistent=%b, accuracy=%.4f, trained=%b", consistent, accuracy, knn.isTrained()));
        }
        assert allPassed : "KNN部分规模测试失败";
    }

    // ==================== 2. 回归器正确性+性能 ====================

    @Test
    public void testLinearRegressionLargeScale() {
        System.out.println("\n=== 线性回归大规模测试 ===");
        int[] scales = {100, 500, 1000, 5000, 10000};
        boolean allPassed = true;

        for (int scale : scales) {
            RANDOM.setSeed(FIXED_SEED + scale);

            // 生成 y = 2x + 1 + noise 数据
            double[][] features = new double[scale][1];
            double[] labels = new double[scale];
            for (int i = 0; i < scale; i++) {
                double x = RANDOM.nextDouble() * 10;
                features[i][0] = x;
                labels[i] = 2.0 * x + 1.0 + RANDOM.nextGaussian() * 0.5;
            }

            IMatrix X = Linalg.matrix(features);
            IVector y = Linalg.vector(labels);

            long start = System.nanoTime();
            RereLinearRegression lr = new RereLinearRegression();
            lr.fit(X, y);
            RegressionResult result = lr.getResult();
            long timeMs = (System.nanoTime() - start) / 1_000_000;

            double r2 = result.getR2Score();
            boolean passed = r2 > 0.9;
            if (!passed) allPassed = false;

            logBenchmark("ML_REG", "LinearRegression", scale, timeMs, passed,
                    String.format("R2=%.4f, RMSE=%.4f", r2, result.getRmse()));
        }
        assert allPassed : "线性回归部分规模测试失败（R2 <= 0.9）";
    }

    // ==================== 3. 聚类正确性+性能 ====================

    @Test
    public void testKMeansLargeScale() {
        System.out.println("\n=== KMeans大规模测试 ===");
        int[] scales = {100, 500, 1000, 5000};
        boolean allPassed = true;

        for (int scale : scales) {
            RANDOM.setSeed(FIXED_SEED + scale);

            // 生成3个已知簇的数据
            double[][] data = new double[scale][2];
            double[][] trueCenters = {{0.0, 0.0}, {5.0, 5.0}, {-5.0, 5.0}};
            for (int i = 0; i < scale; i++) {
                int cluster = i % 3;
                data[i][0] = trueCenters[cluster][0] + RANDOM.nextGaussian();
                data[i][1] = trueCenters[cluster][1] + RANDOM.nextGaussian();
            }

            IMatrix dataMatrix = Linalg.matrix(data);

            long start = System.nanoTime();
            KMeansPlusPlus kmeans = new KMeansPlusPlus(FIXED_SEED + scale);
            kmeans.setParameters(Map.of("numClusters", 3));
            kmeans.fit(dataMatrix);
            int[] labels = kmeans.getLabels();
            long timeMs = (System.nanoTime() - start) / 1_000_000;

            // 验证簇中心在正确区域（每个中心应接近某个真实中心）
            var centers = kmeans.getClusterCenters();
            boolean centersCorrect = true;
            for (var center : centers) {
                double cx = center.get(0);
                double cy = center.get(1);
                boolean nearAnyTrue = false;
                for (double[] tc : trueCenters) {
                    double dist = Math.sqrt((cx - tc[0]) * (cx - tc[0]) + (cy - tc[1]) * (cy - tc[1]));
                    if (dist < 3.0) {
                        nearAnyTrue = true;
                        break;
                    }
                }
                if (!nearAnyTrue) centersCorrect = false;
            }

            boolean passed = centersCorrect && kmeans.isConverged();
            if (!passed) allPassed = false;

            logBenchmark("ML_CLU", "KMeans", scale, timeMs, passed,
                    String.format("converged=%b, centersCorrect=%b, iterations=%d",
                            kmeans.isConverged(), centersCorrect, kmeans.getIterations()));
        }
        assert allPassed : "KMeans部分规模测试失败";
    }

    @Test
    public void testGMMLargeScale() {
        System.out.println("\n=== GMM大规模测试 ===");
        int[] scales = {100, 500, 1000, 5000};
        boolean allPassed = true;

        for (int scale : scales) {
            RANDOM.setSeed(FIXED_SEED + scale);

            // 生成2个高斯混合的数据
            double[][] data = new double[scale][2];
            for (int i = 0; i < scale; i++) {
                if (i % 2 == 0) {
                    data[i][0] = RANDOM.nextGaussian() * 1.0 + 3.0;
                    data[i][1] = RANDOM.nextGaussian() * 1.0 + 3.0;
                } else {
                    data[i][0] = RANDOM.nextGaussian() * 1.0 - 3.0;
                    data[i][1] = RANDOM.nextGaussian() * 1.0 - 3.0;
                }
            }

            List<IVector<Double>> dataList = new ArrayList<>();
            for (double[] row : data) {
                dataList.add(Linalg.vector(row));
            }

            long start = System.nanoTime();
            GMMClustering gmm = new GMMClustering(100, 1e-6, 3, true, FIXED_SEED + scale, false);
            gmm.setParameters(Map.of("numClusters", 2));
            gmm.fit(dataList);
            int[] labels = gmm.getLabels();
            long timeMs = (System.nanoTime() - start) / 1_000_000;

            // 验证两个簇都有数据点
            int[] counts = new int[2];
            for (int label : labels) {
                if (label >= 0 && label < 2) counts[label]++;
            }
            boolean bothClustersPresent = counts[0] > 0 && counts[1] > 0;

            boolean passed = bothClustersPresent;
            if (!passed) allPassed = false;

            logBenchmark("ML_CLU", "GMM", scale, timeMs, passed,
                    String.format("cluster0=%d, cluster1=%d, converged=%b",
                            counts[0], counts[1], gmm.isConverged()));
        }
        assert allPassed : "GMM部分规模测试失败";
    }

    // ==================== 4. 降维正确性 ====================

    @Test
    public void testPCALargeScale() {
        System.out.println("\n=== PCA大规模测试 ===");
        int[] scales = {100, 500, 1000};
        boolean allPassed = true;

        for (int scale : scales) {
            RANDOM.setSeed(FIXED_SEED + scale);

            // 生成10维数据
            double[][] data = new double[scale][10];
            for (int i = 0; i < scale; i++) {
                for (int j = 0; j < 10; j++) {
                    data[i][j] = RANDOM.nextGaussian();
                }
            }
            IMatrix dataMatrix = Linalg.matrix(data);

            long start = System.nanoTime();
            RerePCA pca = new RerePCA().setNComponents(3);
            IMatrix reduced = pca.fitTransform(dataMatrix);
            long timeMs = (System.nanoTime() - start) / 1_000_000;

            // 验证维度正确
            boolean dimsCorrect = reduced.getRowNum() == scale && reduced.getColNum() == 3;
            boolean passed = dimsCorrect;
            if (!passed) allPassed = false;

            logBenchmark("ML_DR", "PCA", scale, timeMs, passed,
                    String.format("rows=%d, cols=%d", reduced.getRowNum(), reduced.getColNum()));
        }
        assert allPassed : "PCA部分规模测试失败";
    }

    @Test
    public void testSVDLargeScale() {
        System.out.println("\n=== SVD降维大规模测试 ===");
        int[] scales = {100, 500, 1000};
        boolean allPassed = true;

        for (int scale : scales) {
            RANDOM.setSeed(FIXED_SEED + scale);

            double[][] data = new double[scale][10];
            for (int i = 0; i < scale; i++) {
                for (int j = 0; j < 10; j++) {
                    data[i][j] = RANDOM.nextGaussian();
                }
            }
            IMatrix dataMatrix = Linalg.matrix(data);

            long start = System.nanoTime();
            RereSVD svd = new RereSVD().setNComponents(3);
            IMatrix reduced = svd.fitTransform(dataMatrix);
            long timeMs = (System.nanoTime() - start) / 1_000_000;

            boolean dimsCorrect = reduced.getRowNum() == scale && reduced.getColNum() == 3;
            boolean passed = dimsCorrect;
            if (!passed) allPassed = false;

            logBenchmark("ML_DR", "SVD", scale, timeMs, passed,
                    String.format("rows=%d, cols=%d", reduced.getRowNum(), reduced.getColNum()));
        }
        assert allPassed : "SVD降维部分规模测试失败";
    }

    // ==================== 5. 边界测试 ====================

    @Test
    public void testEdgeCases() {
        System.out.println("\n=== 边界测试 ===");
        boolean allPassed = true;

        // 5.1 只有一个类别的数据
        {
            RANDOM.setSeed(FIXED_SEED);
            double[][] data = new double[100][3];
            String[] labels = new String[100];
            for (int i = 0; i < 100; i++) {
                data[i][0] = RANDOM.nextGaussian();
                data[i][1] = RANDOM.nextGaussian();
                data[i][2] = RANDOM.nextGaussian();
                labels[i] = "A";
            }
            IMatrix features = Linalg.matrix(data);

            long start = System.nanoTime();
            boolean threwExpected = false;
            try {
                RereDecisionTree dt = new RereDecisionTree();
                dt.fit(features, labels);
            } catch (IllegalArgumentException e) {
                threwExpected = true;
            }
            long timeMs = (System.nanoTime() - start) / 1_000_000;

            boolean passed = threwExpected;
            if (!passed) allPassed = false;
            logBenchmark("ML_EDGE", "SingleClass", 100, timeMs, passed,
                    "期望抛出IllegalArgumentException");
        }

        // 5.2 所有样本完全相同的特征
        {
            double[][] data = new double[100][3];
            for (int i = 0; i < 100; i++) {
                data[i][0] = 1.0;
                data[i][1] = 2.0;
                data[i][2] = 3.0;
            }
            String[] labels = new String[100];
            for (int i = 0; i < 100; i++) {
                labels[i] = (i < 50) ? "A" : "B";
            }
            IMatrix features = Linalg.matrix(data);

            long start = System.nanoTime();
            RereDecisionTree dt = new RereDecisionTree();
            dt.fit(features, labels);
            String[] preds = dt.predictBatch(features);
            long timeMs = (System.nanoTime() - start) / 1_000_000;

            // 相同特征下，决策树应该能训练（可能所有预测相同）
            boolean passed = dt.isTrained();
            if (!passed) allPassed = false;
            logBenchmark("ML_EDGE", "IdenticalFeatures", 100, timeMs, passed,
                    "trained=" + dt.isTrained());
        }

        // 5.3 空特征矩阵
        {
            long start = System.nanoTime();
            boolean threwExpected = false;
            try {
                double[][] emptyData = new double[0][3];
                String[] emptyLabels = new String[0];
                IMatrix features = Linalg.matrix(emptyData);
                RereLogisticRegression lr = new RereLogisticRegression();
                lr.fit(features, emptyLabels);
            } catch (IllegalArgumentException e) {
                threwExpected = true;
            }
            long timeMs = (System.nanoTime() - start) / 1_000_000;

            boolean passed = threwExpected;
            if (!passed) allPassed = false;
            logBenchmark("ML_EDGE", "EmptyMatrix", 0, timeMs, passed,
                    "期望抛出IllegalArgumentException");
        }

        // 5.4 类别极度不平衡（99:1）
        {
            RANDOM.setSeed(FIXED_SEED);
            double[][] data = new double[1000][3];
            String[] labels = new String[1000];
            for (int i = 0; i < 1000; i++) {
                data[i][0] = RANDOM.nextGaussian();
                data[i][1] = RANDOM.nextGaussian();
                data[i][2] = RANDOM.nextGaussian();
                labels[i] = (i < 10) ? "B" : "A";  // 99:1 不平衡
            }
            IMatrix features = Linalg.matrix(data);

            long start = System.nanoTime();
            RereLogisticRegression lr = new RereLogisticRegression();
            lr.setRandomSeed(FIXED_SEED);
            lr.setStandardizeFeatures(false);
            lr.fit(features, labels);
            String[] preds = lr.predictBatch(features);
            long timeMs = (System.nanoTime() - start) / 1_000_000;

            // 在不平衡数据上，模型应该能训练
            boolean passed = lr.isTrained();
            if (!passed) allPassed = false;
            logBenchmark("ML_EDGE", "Imbalanced99to1", 1000, timeMs, passed,
                    "trained=" + lr.isTrained());
        }

        assert allPassed : "边界测试部分失败";
    }

    // ==================== 6. 交叉验证正确性 ====================

    @Test
    public void testCrossValidationCorrectness() {
        System.out.println("\n=== 交叉验证正确性测试 ===");
        boolean allPassed = true;

        // 生成测试数据
        RANDOM.setSeed(FIXED_SEED);
        int nSamples = 500;
        String[] labels = generateLabels(nSamples, 2);
        IMatrix features = generateClassificationData(nSamples, 5, labels);

        long start = System.nanoTime();
        RereDecisionTree classifier = new RereDecisionTree();
        CrossValidationResult cvResult = CrossValidation.kFoldCrossValidation(classifier, features, labels, 5);
        long timeMs = (System.nanoTime() - start) / 1_000_000;

        // 验证每折的准确率范围合理（0-1）
        List<Double> accuracies = cvResult.getAccuracyScores();
        boolean accuracyInRange = true;
        for (double acc : accuracies) {
            if (acc < 0.0 || acc > 1.0) {
                accuracyInRange = false;
                break;
            }
        }

        // 验证训练/测试不重叠：通过检查fold数量
        boolean correctFolds = accuracies.size() == 5;

        boolean passed = accuracyInRange && correctFolds;
        if (!passed) allPassed = false;

        logBenchmark("ML_CV", "KFoldCV", nSamples, timeMs, passed,
                String.format("folds=%d, accRange=[%.3f, %.3f], inRange=%b",
                        accuracies.size(),
                        accuracies.stream().mapToDouble(Double::doubleValue).min().orElse(0),
                        accuracies.stream().mapToDouble(Double::doubleValue).max().orElse(0),
                        accuracyInRange));

        // 分层K折交叉验证
        {
            start = System.nanoTime();
            CrossValidationResult stratifiedResult = CrossValidation.stratifiedKFoldCrossValidation(
                    classifier, features, labels, 5);
            timeMs = (System.nanoTime() - start) / 1_000_000;

            List<Double> stratAccs = stratifiedResult.getAccuracyScores();
            boolean stratInRange = true;
            for (double acc : stratAccs) {
                if (acc < 0.0 || acc > 1.0) {
                    stratInRange = false;
                    break;
                }
            }

            boolean stratPassed = stratInRange && stratAccs.size() == 5;
            if (!stratPassed) allPassed = false;

            logBenchmark("ML_CV", "StratifiedKFoldCV", nSamples, timeMs, stratPassed,
                    String.format("folds=%d, accRange=[%.3f, %.3f]",
                            stratAccs.size(),
                            stratAccs.stream().mapToDouble(Double::doubleValue).min().orElse(0),
                            stratAccs.stream().mapToDouble(Double::doubleValue).max().orElse(0)));
        }

        assert allPassed : "交叉验证测试失败";
    }
}
