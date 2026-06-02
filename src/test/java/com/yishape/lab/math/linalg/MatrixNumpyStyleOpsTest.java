package com.yishape.lab.math.linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

class MatrixNumpyStyleOpsTest {

    @Test
    void mmulScalarAliasMatchesMultiplyScalar() {
        IMatrix<Double> a = Linalg.matrix(new double[][]{{1, 2}, {3, 4}});
        IMatrix<Double> s1 = a.multiplyByScalar(2.0);
        IMatrix<Double> s2 = a.mmul(2.0);
        assertEquals(s1.get(0, 0), s2.get(0, 0), 1e-12);
        assertEquals(s1.get(1, 1), s2.get(1, 1), 1e-12);
    }

    @Test
    void kronMatchesNumpyKron2x2() {
        IMatrix<Double> a = Linalg.matrix(new double[][]{{1, 2}, {3, 4}});
        IMatrix<Double> b = Linalg.matrix(new double[][]{{0, 5}, {6, 7}});
        IMatrix<Double> k = Linalg.kron(a, b);
        assertEquals(4, k.rows());
        assertEquals(4, k.cols());
        assertEquals(0.0, k.get(0, 0), 1e-12);
        assertEquals(5.0, k.get(0, 1), 1e-12);
        assertEquals(0.0, k.get(0, 2), 1e-12);
        assertEquals(10.0, k.get(0, 3), 1e-12);
        assertEquals(6.0, k.get(1, 0), 1e-12);
        assertEquals(7.0, k.get(1, 1), 1e-12);
        assertEquals(12.0, k.get(1, 2), 1e-12);
        assertEquals(14.0, k.get(1, 3), 1e-12);
    }

    @Test
    void outerFlattenMatchesVectorOuterOfRavel() {
        IMatrix<Double> a = Linalg.matrix(new double[][]{{1, 2}, {3, 4}});
        IMatrix<Double> b = Linalg.matrix(new double[][]{{10.0}, {20.0}});
        IMatrix<Double> o = a.outer(b);
        assertEquals(4, o.rows());
        assertEquals(2, o.cols());
        assertEquals(10.0, o.get(0, 0), 1e-12);
        assertEquals(20.0, o.get(0, 1), 1e-12);
        assertEquals(30.0, o.get(2, 0), 1e-12);
    }

    @Test
    void qrEigenAndQrEigenDecompositionMatchEigen() {
        IMatrix<Double> a = Linalg.matrix(new double[][]{{4.0, -2.0}, {-2.0, 4.0}});
        var e1 = a.eigen();
        var e2 = a.qrEigenDecomposition();
        var e3 = a.qrEigen();
        assertEquals(e1._1.length(), e2._1.length());
        for (int i = 0; i < e1._1.length(); i++) {
            assertEquals(e1._1.get(i), e2._1.get(i), 1e-9);
            assertEquals(e1._1.get(i), e3._1.get(i), 1e-9);
        }
    }

    @Test
    void qrFactorizationIsNotEigen() {
        IMatrix<Double> a = Linalg.matrix(new double[][]{{4.0, -2.0}, {-2.0, 4.0}});
        var qr = a.qr();
        var ev = a.eigen();
        // QR 得到 Q、R；特征分解得到特征值向量，二者类型与语义均不同
        assertEquals(2, qr._1.rows());
        assertEquals(2, qr._1.cols());
        assertEquals(2, ev._1.length());
    }
}
