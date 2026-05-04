package com.yishape.lab.math.ml.lr;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RereLinearRegressionR2Test {

    @Test
    void r2ScoreNearOneOnPerfectLinearData() {
        double[][] x = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        double[] y = {14, 32, 50};
        IMatrix<Double> features = Linalg.matrix(x);
        IVector<Double> labels = Linalg.vector(y);
        RereLinearRegression lr = new RereLinearRegression();
        RegressionResult result = lr.fit(features, labels);
        assertTrue(result.getR2Score() > 0.999, "training R² should be ~1 for collinear labels");
        assertTrue(Math.abs(result.getR2Score() - lr.r2ScoreOn(features, labels)) < 1e-12);
        assertTrue(Math.abs(result.getRmse() - lr.rmseOn(features, labels)) < 1e-12,
                "RegressionResult RMSE must match rmseOn on training data");
    }

    @Test
    void r2ScoreRequiresFit() {
        RereLinearRegression lr = new RereLinearRegression();
        IMatrix<Double> f = Linalg.matrix(new double[][]{{1, 2}});
        IVector<Double> lab = Linalg.vector(new double[]{3});
        assertThrows(IllegalStateException.class, () -> lr.r2ScoreOn(f, lab));
        assertThrows(IllegalStateException.class, () -> lr.rmseOn(f, lab));
    }
}
