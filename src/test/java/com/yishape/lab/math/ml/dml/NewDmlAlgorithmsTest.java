package com.yishape.lab.math.ml.dml;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.dml.anmm.KanmmDml;
import com.yishape.lab.math.ml.dml.dmlmj.KDmlmjDml;
import com.yishape.lab.math.ml.dml.kda.KdaDml;
import com.yishape.lab.math.ml.dml.knn.CondensedNearestNeighbors;
import com.yishape.lab.math.ml.dml.knn.ReducedNearestNeighbors;
import com.yishape.lab.math.ml.dml.llda.LldaDml;
import com.yishape.lab.math.ml.dml.llda.KLldaDml;
import com.yishape.lab.math.ml.dml.lmnn.KlmmnDml;
import com.yishape.lab.math.ml.dml.ncmc.NcmcDml;
import com.yishape.lab.math.ml.dml.odml.KodmlDml;
import com.yishape.lab.math.ml.dml.KernelDmlUtils.KernelType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 新增 DML 算法的基本正确性检查。
 */
class NewDmlAlgorithmsTest {

    private static final double[][] RAW = {
            {0.0, 0.0},
            {0.1, 0.05},
            {5.0, 5.0},
            {5.2, 4.9},
    };

    private static final String[] LABELS = {"a", "a", "b", "b"};

    @Test
    void ncmcRuns() {
        IMatrix<Double> x = IMatrix.of(RAW);
        var m = new NcmcDml().setCentroidsNum(2).setMaxIter(20).fit(x, LABELS);
        assertEquals(MetricForm.LOW_RANK, m.form());
        assertTrue(m.squaredDistance(Linalg.vector(RAW[0]), Linalg.vector(RAW[2])) > 0);
    }

    @Test
    void cnnRuns() {
        IMatrix<Double> x = IMatrix.of(RAW);
        CondensedNearestNeighbors cnn = new CondensedNearestNeighbors();
        cnn.fit(x, LABELS);
        assertTrue(cnn.getCondensedIndexes().length > 0);
    }

    @Test
    void rnnRuns() {
        IMatrix<Double> x = IMatrix.of(RAW);
        ReducedNearestNeighbors rnn = new ReducedNearestNeighbors();
        rnn.fit(x, LABELS);
        assertTrue(rnn.getReducedIndexes().length > 0);
    }

    @Test
    void klmmnRuns() {
        IMatrix<Double> x = IMatrix.of(RAW);
        var m = new KlmmnDml().setKernelType(KernelType.RBF).setGamma(0.5)
                .setMaxSteps(10).fit(x, LABELS);
        assertNotNull(m);
        assertEquals(MetricForm.LOW_RANK, m.form());
    }

    @Test
    void kanmmRuns() {
        IMatrix<Double> x = IMatrix.of(RAW);
        var m = new KanmmDml().setKernelType(KernelType.RBF).setGamma(0.5)
                .setK(1).fit(x, LABELS);
        assertNotNull(m);
    }

    @Test
    void kdmlmjRuns() {
        IMatrix<Double> x = IMatrix.of(RAW);
        var m = new KDmlmjDml().setKernelType(KernelType.LINEAR).fit(x, LABELS);
        assertNotNull(m);
    }

    @Test
    void kodmlRuns() {
        IMatrix<Double> x = IMatrix.of(RAW);
        var m = new KodmlDml().setKernelType(KernelType.RBF).setGamma(0.5)
                .setLearningRate(0.01).fit(x, LABELS);
        assertNotNull(m);
    }

    @Test
    void lldaRuns() {
        IMatrix<Double> x = IMatrix.of(RAW);
        var m = new LldaDml().setNComponents(1).fit(x, LABELS);
        assertEquals(MetricForm.LOW_RANK, m.form());
        assertTrue(m.squaredDistance(Linalg.vector(RAW[0]), Linalg.vector(RAW[1])) >= 0);
    }

    @Test
    void klldaRuns() {
        IMatrix<Double> x = IMatrix.of(RAW);
        var m = new KLldaDml().setKernelType(KernelType.RBF).setGamma(0.5)
                .setNComponents(1).fit(x, LABELS);
        assertEquals(MetricForm.LOW_RANK, m.form());
    }

    @Test
    void kdaRuns() {
        IMatrix<Double> x = IMatrix.of(RAW);
        var m = new KdaDml().setKernelType(KernelType.RBF).setGamma(0.5)
                .setNComponents(1).fit(x, LABELS);
        assertEquals(MetricForm.LOW_RANK, m.form());
    }

    @Test
    void kernelUtilsKernelMatrix() {
        double[][] K = KernelDmlUtils.kernelMatrix(RAW, RAW, KernelType.RBF, 0.5, 3, 1.0);
        assertEquals(RAW.length, K.length);
        assertEquals(RAW.length, K[0].length);
        // 对角线元素应为 1 (RBF 核)
        assertEquals(1.0, K[0][0], 1e-9);
        assertEquals(1.0, K[2][2], 1e-9);
    }

    @Test
    void kernelUtilsCenterKernelMatrix() {
        double[][] K = KernelDmlUtils.kernelMatrix(RAW, RAW, KernelType.LINEAR, 1.0, 3, 0.0);
        double[][] Kc = KernelDmlUtils.centerKernelMatrix(K);
        assertEquals(K.length, Kc.length);
        // 中心化后每行和应接近 0
        for (int i = 0; i < Kc.length; i++) {
            double rowSum = 0;
            for (int j = 0; j < Kc[i].length; j++) rowSum += Kc[i][j];
            assertEquals(0.0, rowSum, 1e-9);
        }
    }
}
