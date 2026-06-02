package com.yishape.lab.math.ml.dml;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.ml.dml.DmlMetric;
import com.yishape.lab.math.ml.ML;

import java.util.function.Supplier;

/**
 * DML 算法对比测试：对比 Java 实现与 Python pyDML
 */
public class JavaDmlBenchmark {

    private static final double[][] IRIS_FEATURES = {
        {5.1, 3.5, 1.4, 0.2},
        {4.9, 3.0, 1.4, 0.2},
        {4.7, 3.2, 1.3, 0.2},
        {4.6, 3.1, 1.5, 0.2},
        {5.0, 3.6, 1.4, 0.2},
        {5.4, 3.9, 1.7, 0.4},
        {4.6, 3.4, 1.4, 0.3},
        {5.0, 3.4, 1.5, 0.2},
        {4.4, 2.9, 1.4, 0.2},
        {4.9, 3.1, 1.5, 0.1},
        {7.0, 3.2, 4.7, 1.4},
        {6.4, 3.2, 4.5, 1.5},
        {6.9, 3.1, 4.9, 1.5},
        {5.5, 2.3, 4.0, 1.3},
        {6.5, 2.8, 4.6, 1.5},
        {5.7, 2.8, 4.5, 1.3},
        {6.3, 3.3, 4.7, 1.6},
        {4.9, 2.4, 3.3, 1.0},
        {6.6, 2.9, 4.6, 1.3},
        {5.2, 2.7, 3.9, 1.4},
        {5.0, 2.0, 3.5, 1.0},
        {5.9, 3.0, 4.2, 1.5},
        {6.0, 2.2, 4.0, 1.0},
        {6.1, 2.9, 4.7, 1.4},
        {5.6, 2.9, 3.6, 1.3},
        {6.7, 3.1, 4.4, 1.4},
        {5.6, 3.0, 4.5, 1.5},
        {5.8, 2.7, 4.1, 1.0},
        {6.2, 2.2, 4.5, 1.5},
        {5.6, 2.5, 3.9, 1.1},
        {5.0, 3.4, 1.5, 0.2},
        {5.2, 3.5, 1.5, 0.2},
        {5.3, 3.7, 1.5, 0.2},
        {4.8, 3.0, 1.4, 0.3},
        {5.4, 3.4, 1.5, 0.4},
        {5.1, 3.7, 1.4, 0.3},
        {5.7, 3.0, 1.7, 0.3},
        {5.1, 3.4, 1.5, 0.2},
        {5.4, 3.9, 1.7, 0.4},
        {5.1, 3.5, 1.4, 0.3},
        {4.6, 3.6, 1.0, 0.2},
        {5.1, 3.3, 1.7, 0.5},
        {4.8, 3.4, 1.9, 0.2},
        {5.0, 3.0, 1.6, 0.2},
        {5.0, 3.4, 1.6, 0.4},
        {5.2, 3.4, 1.4, 0.2},
        {5.2, 4.1, 1.5, 0.1},
        {4.7, 3.2, 1.6, 0.2},
        {4.8, 3.1, 1.6, 0.2},
        {5.4, 3.4, 1.5, 0.4},
        {5.2, 4.0, 1.2, 0.2},
        {5.5, 4.2, 1.4, 0.2},
        {4.9, 3.1, 1.5, 0.2},
        {5.0, 3.2, 1.2, 0.2},
        {5.5, 3.5, 1.3, 0.2},
        {4.9, 3.6, 1.4, 0.1},
        {4.4, 3.0, 1.3, 0.2},
        {5.1, 3.4, 1.5, 0.2},
        {5.0, 3.5, 1.3, 0.3},
        {4.5, 2.3, 1.3, 0.3},
        {4.4, 3.2, 1.3, 0.2},
        {5.0, 3.5, 1.6, 0.6},
        {5.1, 3.8, 1.9, 0.4},
        {4.8, 3.0, 1.4, 0.3},
        {5.1, 3.8, 1.6, 0.2},
        {5.3, 3.7, 1.5, 0.2},
        {5.0, 3.3, 1.4, 0.2},
        {7.1, 3.0, 5.9, 2.1},
        {6.3, 2.9, 5.6, 1.8},
        {6.5, 3.0, 5.8, 2.2},
        {7.6, 3.0, 6.6, 2.1},
        {4.9, 2.5, 4.5, 1.7},
        {7.3, 2.9, 6.3, 1.8},
        {6.7, 2.5, 5.8, 1.8},
        {7.2, 3.6, 6.1, 2.5},
        {6.5, 3.2, 5.1, 2.0},
        {6.4, 2.7, 5.3, 1.9},
        {6.8, 3.0, 5.5, 2.1},
        {5.7, 2.5, 5.0, 2.0},
        {5.8, 2.8, 5.1, 2.4},
        {6.4, 3.2, 5.3, 2.3},
        {6.5, 3.0, 5.5, 1.8},
        {7.7, 3.8, 6.7, 2.2},
        {7.7, 2.6, 6.9, 2.3},
        {6.0, 2.2, 5.0, 1.5},
        {6.9, 3.2, 5.7, 2.3},
        {5.6, 2.8, 4.9, 2.0},
        {7.7, 2.8, 6.7, 2.0},
        {6.3, 2.7, 4.9, 1.8},
        {6.7, 3.3, 5.7, 2.1},
        {7.2, 3.2, 6.0, 1.8},
        {6.2, 2.8, 4.8, 1.8},
        {6.1, 3.0, 4.9, 1.8},
        {6.4, 2.8, 5.6, 2.1},
        {7.2, 3.0, 5.8, 1.6},
        {7.4, 2.8, 6.1, 1.9},
        {7.9, 3.8, 6.4, 2.0},
        {6.4, 2.8, 5.6, 2.2},
        {6.3, 2.8, 5.1, 1.5},
        {6.1, 2.6, 5.6, 1.4},
        {7.7, 3.0, 6.1, 2.3},
        {6.3, 3.4, 5.6, 2.4},
        {6.4, 3.1, 5.5, 1.8},
        {6.0, 3.0, 4.8, 1.8},
        {6.9, 3.1, 5.4, 2.1},
        {6.7, 3.1, 5.6, 2.4},
        {6.9, 3.1, 5.1, 2.3},
        {5.8, 2.7, 5.1, 1.9},
        {6.8, 3.2, 5.9, 2.3},
        {6.7, 3.3, 5.7, 2.5},
        {6.7, 3.0, 5.2, 2.3},
        {6.3, 2.5, 5.0, 1.9},
        {6.5, 3.0, 5.2, 2.0},
        {6.2, 3.4, 5.4, 2.3},
        {5.9, 3.0, 5.1, 1.8}
    };

    private static final String[] IRIS_LABELS = {
        "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa",
        "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa",
        "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa",
        "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa",
        "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa",
        "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor",
        "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor",
        "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor",
        "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor",
        "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor",
        "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor",
        "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor",
        "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica",
        "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica",
        "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica",
        "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica",
        "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica",
        "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica",
        "virginica", "virginica", "virginica", "virginica", "virginica", "virginica", "virginica",
        "virginica", "virginica", "virginica", "virginica"
    };

    public static void main(String[] args) {
        System.out.println("============================================================");
        System.out.println("Java yishape-math DML Benchmark");
        System.out.println("============================================================");

        IMatrix<Double> X = IMatrix.of(IRIS_FEATURES);

        // Test NCA
        runAndPrint("NCA", () -> {
            try {
                long start = System.nanoTime();
                DmlMetric m = ML.dml.nca().fit(X, IRIS_LABELS);
                long elapsed = System.nanoTime() - start;
                double silhouette = computeSilhouette(m, X, IRIS_LABELS);
                return new Result(true, elapsed / 1_000_000.0, silhouette, null);
            } catch (Exception e) {
                return new Result(false, 0, 0, e.getMessage());
            }
        });

        // Test LMNN
        runAndPrint("LMNN", () -> {
            try {
                long start = System.nanoTime();
                DmlMetric m = ML.dml.lmnn().fit(X, IRIS_LABELS);
                long elapsed = System.nanoTime() - start;
                double silhouette = computeSilhouette(m, X, IRIS_LABELS);
                return new Result(true, elapsed / 1_000_000.0, silhouette, null);
            } catch (Exception e) {
                return new Result(false, 0, 0, e.getMessage());
            }
        });

        // Test LDML (Pairwise)
        runAndPrint("LDML", () -> {
            try {
                long start = System.nanoTime();
                DmlMetric m = ML.dml.ldmlPairwise().fit(X, IRIS_LABELS);
                long elapsed = System.nanoTime() - start;
                double silhouette = computeSilhouette(m, X, IRIS_LABELS);
                return new Result(true, elapsed / 1_000_000.0, silhouette, null);
            } catch (Exception e) {
                return new Result(false, 0, 0, e.getMessage());
            }
        });

        // Test MCML
        runAndPrint("MCML", () -> {
            try {
                long start = System.nanoTime();
                DmlMetric m = ML.dml.mcml().fit(X, IRIS_LABELS);
                long elapsed = System.nanoTime() - start;
                double silhouette = computeSilhouette(m, X, IRIS_LABELS);
                return new Result(true, elapsed / 1_000_000.0, silhouette, null);
            } catch (Exception e) {
                return new Result(false, 0, 0, e.getMessage());
            }
        });

        // Test DML-eig
        runAndPrint("DML-eig", () -> {
            try {
                long start = System.nanoTime();
                DmlMetric m = ML.dml.dmleig().fit(X, IRIS_LABELS);
                long elapsed = System.nanoTime() - start;
                double silhouette = computeSilhouette(m, X, IRIS_LABELS);
                return new Result(true, elapsed / 1_000_000.0, silhouette, null);
            } catch (Exception e) {
                return new Result(false, 0, 0, e.getMessage());
            }
        });

        // Test ANMM
        runAndPrint("ANMM", () -> {
            try {
                long start = System.nanoTime();
                DmlMetric m = ML.dml.anmm().fit(X, IRIS_LABELS);
                long elapsed = System.nanoTime() - start;
                double silhouette = computeSilhouette(m, X, IRIS_LABELS);
                return new Result(true, elapsed / 1_000_000.0, silhouette, null);
            } catch (Exception e) {
                return new Result(false, 0, 0, e.getMessage());
            }
        });

        // Test NCMML
        runAndPrint("NCMML", () -> {
            try {
                long start = System.nanoTime();
                DmlMetric m = ML.dml.ncmml().fit(X, IRIS_LABELS);
                long elapsed = System.nanoTime() - start;
                double silhouette = computeSilhouette(m, X, IRIS_LABELS);
                return new Result(true, elapsed / 1_000_000.0, silhouette, null);
            } catch (Exception e) {
                return new Result(false, 0, 0, e.getMessage());
            }
        });

        // Test NCMC
        runAndPrint("NCMC", () -> {
            try {
                long start = System.nanoTime();
                DmlMetric m = ML.dml.ncmc().fit(X, IRIS_LABELS);
                long elapsed = System.nanoTime() - start;
                double silhouette = computeSilhouette(m, X, IRIS_LABELS);
                return new Result(true, elapsed / 1_000_000.0, silhouette, null);
            } catch (Exception e) {
                return new Result(false, 0, 0, e.getMessage());
            }
        });

        // Test ITML
        runAndPrint("ITML", () -> {
            try {
                long start = System.nanoTime();
                DmlMetric m = ML.dml.itml().fit(X, IRIS_LABELS);
                long elapsed = System.nanoTime() - start;
                double silhouette = computeSilhouette(m, X, IRIS_LABELS);
                return new Result(true, elapsed / 1_000_000.0, silhouette, null);
            } catch (Exception e) {
                return new Result(false, 0, 0, e.getMessage());
            }
        });

        // Test DMLMJ
        runAndPrint("DMLMJ", () -> {
            try {
                long start = System.nanoTime();
                DmlMetric m = ML.dml.dmlmj().fit(X, IRIS_LABELS);
                long elapsed = System.nanoTime() - start;
                double silhouette = computeSilhouette(m, X, IRIS_LABELS);
                return new Result(true, elapsed / 1_000_000.0, silhouette, null);
            } catch (Exception e) {
                return new Result(false, 0, 0, e.getMessage());
            }
        });

        System.out.println("============================================================");
        System.out.println("Benchmark completed!");
    }

    private static void runAndPrint(String name, Supplier<Result> test) {
        Result r = test.get();
        if (r.success) {
            System.out.printf("%-10s: OK, time=%.2fms, silhouette=%.4f%n",
                name, r.timeMs, r.silhouette);
        } else {
            System.out.printf("%-10s: FAILED - %s%n", name, r.error);
        }
    }

    private static double computeSilhouette(DmlMetric m, IMatrix<Double> X, String[] labels) {
        // Transform features
        IMatrix<Double> Xt = m.transform(X);
        int n = Xt.getRowNum();
        int d = Xt.getColNum();

        // Convert to array
        double[][] xTransformed = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                xTransformed[i][j] = Xt.get(i, j);
            }
        }

        // Convert labels to int
        java.util.Map<String, Integer> labelMap = new java.util.LinkedHashMap<>();
        int[] y = new int[n];
        int nextLabel = 0;
        for (int i = 0; i < n; i++) {
            String lbl = labels[i];
            if (!labelMap.containsKey(lbl)) {
                labelMap.put(lbl, nextLabel++);
            }
            y[i] = labelMap.get(lbl);
        }

        // Compute silhouette score using sklearn-like logic
        return silhouetteScore(xTransformed, y);
    }

    private static double silhouetteScore(double[][] X, int[] y) {
        int n = X.length;
        int k = 3; // 3 classes for iris

        // Compute pairwise distances
        double[][] dist = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double d = 0;
                for (int p = 0; p < X[i].length; p++) {
                    double diff = X[i][p] - X[j][p];
                    d += diff * diff;
                }
                dist[i][j] = Math.sqrt(d);
                dist[j][i] = dist[i][j];
            }
        }

        double score = 0;
        for (int i = 0; i < n; i++) {
            // a(i): mean distance to same cluster
            double a = 0;
            int sameCount = 0;
            for (int j = 0; j < n; j++) {
                if (i != j && y[i] == y[j]) {
                    a += dist[i][j];
                    sameCount++;
                }
            }
            a = sameCount > 0 ? a / sameCount : 0;

            // b(i): min mean distance to other clusters
            double b = Double.MAX_VALUE;
            for (int c = 0; c < k; c++) {
                if (c == y[i]) continue;
                double meanDist = 0;
                int count = 0;
                for (int j = 0; j < n; j++) {
                    if (y[j] == c) {
                        meanDist += dist[i][j];
                        count++;
                    }
                }
                if (count > 0) {
                    meanDist /= count;
                    b = Math.min(b, meanDist);
                }
            }
            if (b == Double.MAX_VALUE) b = 0;

            double s = (b - a) / Math.max(a, b);
            score += s;
        }

        return score / n;
    }

    private static class Result {
        boolean success;
        double timeMs;
        double silhouette;
        String error;

        Result(boolean success, double timeMs, double silhouette, String error) {
            this.success = success;
            this.timeMs = timeMs;
            this.silhouette = silhouette;
            this.error = error;
        }
    }
}
