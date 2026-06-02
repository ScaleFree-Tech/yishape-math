package com.yishape.lab.math.ml.dr;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link RerePCA}.
 */
class RerePCATest {

    // ==================== Basic Usage ====================

    @Test
    void dimensionReduction_reducesDimensions() {
        // 5 samples, 4 features
        IMatrix<Double> data = Linalg.matrix(new double[][]{
            {1, 2, 3, 4},
            {2, 3, 4, 5},
            {3, 4, 5, 6},
            {4, 5, 6, 7},
            {5, 6, 7, 8}
        });

        RerePCA pca = new RerePCA();
        IMatrix<Double> result = pca.dimensionReduction(data, 2);
        assertEquals(5, result.rows());
        assertEquals(2, result.cols());
    }

    @Test
    void setNComponents_chaining() {
        RerePCA pca = new RerePCA();
        RerePCA result = pca.setNComponents(3);
        assertSame(pca, result);
        assertEquals(3, pca.getNComponents());
    }

    // ==================== Fit & Transform ====================

    @Test
    void fit_thenTransform() {
        IMatrix<Double> data = Linalg.matrix(new double[][]{
            {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}
        });

        RerePCA pca = new RerePCA();
        pca.setNComponents(2);
        pca.fit(data);
        assertTrue(pca.ifTrained());

        IMatrix<Double> transformed = pca.transform(data);
        assertEquals(4, transformed.rows());
        assertEquals(2, transformed.cols());
    }

    @Test
    void getFeature_afterFit() {
        IMatrix<Double> data = Linalg.matrix(new double[][]{
            {1, 2}, {3, 4}, {5, 6}
        });

        RerePCA pca = new RerePCA();
        pca.setNComponents(1);
        pca.fit(data);
        assertNotNull(pca.getFeature());
    }

    // ==================== Edge Cases ====================

    @Test
    void fit_sameDimensionAsOriginal() {
        IMatrix<Double> data = Linalg.matrix(new double[][]{
            {1, 2, 3}, {4, 5, 6}, {7, 8, 9}
        });

        RerePCA pca = new RerePCA();
        pca.setNComponents(3);
        IMatrix<Double> result = pca.dimensionReduction(data, 3);
        assertEquals(3, result.rows());
        assertEquals(3, result.cols());
    }

    // ==================== Serialization ====================

    @Test
    void toParams_fromParams() {
        RerePCA pca = new RerePCA();
        pca.setNComponents(2);

        Map<String, Object> params = pca.toParams();
        assertNotNull(params);

        RerePCA pca2 = new RerePCA();
        pca2.fromParams(params);
        assertEquals(2, pca2.getNComponents());
    }

    // ==================== High Dimensional Data ====================

    @Test
    void highDimensional_reduction() {
        // 20 samples, 10 features, reduce to 3
        double[][] data = new double[20][10];
        for (int i = 0; i < 20; i++) {
            for (int j = 0; j < 10; j++) {
                data[i][j] = i + j * 0.5;
            }
        }
        IMatrix<Double> mat = Linalg.matrix(data);

        RerePCA pca = new RerePCA();
        IMatrix<Double> result = pca.dimensionReduction(mat, 3);
        assertEquals(20, result.rows());
        assertEquals(3, result.cols());
    }

    // ==================== ifTrained ====================

    @Test
    void ifTrained_falseByDefault() {
        RerePCA pca = new RerePCA();
        assertFalse(pca.ifTrained());
    }
}
