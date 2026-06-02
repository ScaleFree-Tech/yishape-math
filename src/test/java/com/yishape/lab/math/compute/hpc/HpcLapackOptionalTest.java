package com.yishape.lab.math.compute.hpc;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.util.Tuple3;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * yishape-math-hpc 分解与 Java 分解对齐的冒烟测试（需可用的 yishape_math_rust 原生库）。
 */
class HpcLapackOptionalTest {

    @Test
    void svAndCholeskyConsistentWithJavaWhenLoaded() {
        assumeTrue(HpcOptionalRuntime.isNativeRuntimeAvailable());

        int nChol = 800;
        double[][] aChol = new double[nChol][nChol];
        java.util.concurrent.ThreadLocalRandom r = java.util.concurrent.ThreadLocalRandom.current();
        for (int i = 0; i < nChol; i++) {
            for (int j = 0; j < nChol; j++) {
                aChol[i][j] = r.nextDouble(-0.5, 0.5) + (i == j ? 2.2 : 0);
            }
        }
        IMatrix<Double> mChol = Linalg.matrix(aChol);

        IMatrix<Double> gram = mChol.mmul(mChol.transpose());
        IMatrix<Double> spd
                = gram.add(gram.transpose()).mmul(0.5);
        IMatrix<Double> cOb = HpcLapackDecomps.tryCholeskyL(spd);
        IMatrix<Double> cJv = com.yishape.lab.math.linalg.decomposition.Decomps.createCholesky().decompose(spd);
        assertTrue(cOb != null);
        assertMatrixClose(cOb, cJv, 1e-6);

        int nSvd = 220;
        double[][] aSvd = new double[nSvd][nSvd];
        for (int i = 0; i < nSvd; i++) {
            for (int j = 0; j < nSvd; j++) {
                aSvd[i][j] = r.nextDouble(-0.5, 0.5) + (i == j ? 2.2 : 0);
            }
        }
        IMatrix<Double> m = Linalg.matrix(aSvd);
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> svdOb = HpcLapackDecomps.trySvd(m);
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> svdJv =
                com.yishape.lab.math.linalg.decomposition.Decomps.createSVD().decompose(m);
        assertTrue(svdOb != null);
        assertSvdUsVtClose(svdOb, svdJv, m, 1e-5);
    }

    private static void assertMatrixClose(IMatrix<Double> x, IMatrix<Double> y, double tol) {
        assertEquals(x.rows(), y.rows());
        assertEquals(x.cols(), y.cols());
        for (int i = 0; i < x.rows(); i++) {
            for (int j = 0; j < x.cols(); j++) {
                assertEquals(x.get(i, j), y.get(i, j), tol, "at " + i + "," + j);
            }
        }
    }

    private static void assertSvdUsVtClose(Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> ob,
                                           Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> jv,
                                           IMatrix<Double> a,
                                           double tol) {
        int k = Math.min(a.rows(), a.cols());
        assertEquals(k, ob._2.length());
        assertEquals(k, jv._2.length());
        for (int i = 0; i < k; i++) {
            assertEquals(jv._2.get(i), ob._2.get(i), Math.max(tol, 1e-4 * Math.abs(jv._2.get(i))));
        }
        IMatrix<Double> recOb = reconstructFromSvd(ob);
        IMatrix<Double> recJv = reconstructFromSvd(jv);
        assertMatrixClose(recOb, recJv, tol * 10);
    }

    private static IMatrix<Double> reconstructFromSvd(Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> t) {
        int k = t._2.length();
        IMatrix<Double> vk = t._3.sliceRows(0, k);
        return t._1.mmul(Linalg.diag(t._2)).mmul(vk);
    }
}
