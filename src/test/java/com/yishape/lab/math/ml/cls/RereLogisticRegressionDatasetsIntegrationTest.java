package com.yishape.lab.math.ml.cls;

import com.yishape.lab.math.data.DataFrame;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.ml.metric.ClassificationMetrics;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 使用仓库 {@code datasets/} 下 CSV 对 {@link RereLogisticRegression} 做端到端校验：
 * 训练、批量预测、概率输出，以及无正则 / L1 / L2 / ElasticNet。
 */
@SuppressWarnings("rawtypes")
class RereLogisticRegressionDatasetsIntegrationTest {

    private static String datasetPath(String name) {
        Path[] tried = {
                Paths.get("datasets", name),
                Paths.get(System.getProperty("user.dir"), "datasets", name),
                Paths.get(System.getProperty("user.dir")).getParent().resolve("datasets").resolve(name)
        };
        Path found = null;
        for (Path p : tried) {
            if (Files.exists(p)) {
                found = p;
                break;
            }
        }
        Path resolved = found != null ? found : tried[0];
        assertTrue(Files.exists(resolved), () -> "未找到数据文件，请从项目根目录运行测试: " + resolved.toAbsolutePath());
        return resolved.toString();
    }

    private static void loadFeaturesAndLabels(String csvName, IMatrix[] outX, String[][] outY) throws IOException {
        DataFrame df = DataFrame.readCsv(datasetPath(csvName));
        IMatrix x = df.sliceColumn(0, -1).toMatrix();
        String[] y = df.get(df.getColumnCount() - 1).toStringArray();
        outX[0] = x;
        outY[0] = y;
    }

    private static RereLogisticRegression trained(IMatrix x, String[] y, double l1, double l2) {
        RereLogisticRegression lr = new RereLogisticRegression(l1, l2);
        lr.setRandomSeed(42L);
        lr.setMaxIterations(2500);
        lr.setTolerance(1e-7);
        lr.setLearningRate(0.05);
        lr.fit(x, y);
        return lr;
    }

    private static double trainAccuracy(RereLogisticRegression lr, IMatrix x, String[] y) {
        String[] pred = lr.predictBatch(x);
        return ClassificationMetrics.compute(y, pred).getAccuracy();
    }

    @Test
    void iris_multiclass_regularization_variants() throws IOException {
        IMatrix[] xb = new IMatrix[1];
        String[][] yb = new String[1][];
        loadFeaturesAndLabels("iris.csv", xb, yb);
        IMatrix x = xb[0];
        String[] y = yb[0];

        RereLogisticRegression none = trained(x, y, 0, 0);
        assertEquals(RereLogisticRegression.RegularizationType.NONE, none.getRegularizationType());
        assertTrue(trainAccuracy(none, x, y) >= 0.92);

        RereLogisticRegression l1 = trained(x, y, 0.02, 0);
        assertEquals(RereLogisticRegression.RegularizationType.L1, l1.getRegularizationType());
        assertTrue(trainAccuracy(l1, x, y) >= 0.88);

        RereLogisticRegression l2 = trained(x, y, 0, 0.03);
        assertEquals(RereLogisticRegression.RegularizationType.L2, l2.getRegularizationType());
        assertTrue(trainAccuracy(l2, x, y) >= 0.90);

        RereLogisticRegression en = trained(x, y, 0.008, 0.06);
        assertEquals(RereLogisticRegression.RegularizationType.ELASTIC_NET, en.getRegularizationType());
        assertTrue(trainAccuracy(en, x, y) >= 0.85);

        IVector row = x.getRow(0);
        Map<String, Double> probs = en.predictProb(row);
        double sum = probs.values().stream().mapToDouble(Double::doubleValue).sum();
        assertTrue(sum > 0.99 && sum < 1.01);

        var batch = en.predictBatchWithProbs(x);
        assertEquals(y.length, batch.getPredictions().length);
    }

    @Test
    void wine_multiclass_regularization_variants() throws IOException {
        IMatrix[] xb = new IMatrix[1];
        String[][] yb = new String[1][];
        loadFeaturesAndLabels("wine.csv", xb, yb);
        IMatrix x = xb[0];
        String[] y = yb[0];

        RereLogisticRegression none = trained(x, y, 0, 0);
        assertTrue(trainAccuracy(none, x, y) >= 0.90);

        RereLogisticRegression l2 = trained(x, y, 0, 0.3);
        assertTrue(trainAccuracy(l2, x, y) >= 0.85);

        RereLogisticRegression en = trained(x, y, 0.005, 0.15);
        assertTrue(trainAccuracy(en, x, y) >= 0.85);
    }

    @Test
    void credit_card_binary_subset_regularization_and_prob() throws IOException {
        IMatrix[] xb = new IMatrix[1];
        String[][] yb = new String[1][];
        loadFeaturesAndLabels("credit_card.csv", xb, yb);
        IMatrix fullX = xb[0];
        String[] fullY = yb[0];

        int n = Math.min(4000, fullX.getRowNum());
        IMatrix x = fullX.sliceRows(0, n);
        String[] y = Arrays.copyOfRange(fullY, 0, n);

        RereLogisticRegression lr = trained(x, y, 0.01, 0.2);
        assertTrue(lr.getRegularizationType() == RereLogisticRegression.RegularizationType.ELASTIC_NET);
        assertTrue(trainAccuracy(lr, x, y) >= 0.62);

        Map<String, Double> p = lr.predictProb(x.getRow(0));
        double s = p.values().stream().mapToDouble(Double::doubleValue).sum();
        assertTrue(s > 0.99 && s < 1.01);
        assertEquals(2, p.size());

        double pos = lr.predictProbability(x.getRow(1));
        assertTrue(pos >= 0.0 && pos <= 1.0);
    }
}
