package com.yishape.lab.math.linalg.complex;

import com.yishape.lab.util.Tuple2;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ComplexMatrixQRDecompositionTest {

    @Test
    public void qr_smallMatrix_reconstructsOriginal() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{12, -51, 4}, {6, 167, -68}, {-4, 24, -41}},
            new double[3][3]
        );
        Tuple2<IComplexMatrix, IComplexMatrix> qrResult = a.qr();
        IComplexMatrix Q = qrResult._1;
        IComplexMatrix R = qrResult._2;
        IComplexMatrix prod = Q.multiply(R);
        double tol = 1e-6;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(a.get(i, j).real, prod.get(i, j).real, tol);
                assertEquals(a.get(i, j).imag, prod.get(i, j).imag, tol);
            }
        }
    }

    @Test
    public void qr_QisUnitary_identityCheck() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 2}, {3, 4}}, new double[][]{{0, 1}, {2, 0}}
        );
        Tuple2<IComplexMatrix, IComplexMatrix> qrResult = a.qr();
        IComplexMatrix Q = qrResult._1;
        IComplexMatrix QH = Q.conjugateTranspose();
        IComplexMatrix ident = QH.multiply(Q);
        double tol = 1e-8;
        assertEquals(1, ident.get(0, 0).real, tol);
        assertEquals(1, ident.get(1, 1).real, tol);
        assertEquals(0, ident.get(0, 1).real, tol);
        assertEquals(0, ident.get(1, 0).real, tol);
    }

    @Test
    public void qr_RisUpperTriangular() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 2, 3}, {4, 5, 6}}, new double[2][3]
        );
        Tuple2<IComplexMatrix, IComplexMatrix> qrResult = a.qr();
        IComplexMatrix R = qrResult._2;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < i; j++) {
                assertEquals(0, R.get(i, j).real, 1e-10);
                assertEquals(0, R.get(i, j).imag, 1e-10);
            }
        }
    }

    @Test
    public void qr_complexMatrix_reconstructsOriginal() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 2}, {3, 4}},
            new double[][]{{0, 1}, {2, 0}}
        );
        Tuple2<IComplexMatrix, IComplexMatrix> qrResult = a.qr();
        IComplexMatrix Q = qrResult._1;
        IComplexMatrix R = qrResult._2;
        IComplexMatrix prod = Q.multiply(R);
        double tol = 1e-10;
        assertEquals(a.get(0, 0).real, prod.get(0, 0).real, tol);
        assertEquals(a.get(0, 0).imag, prod.get(0, 0).imag, tol);
        assertEquals(a.get(0, 1).real, prod.get(0, 1).real, tol);
        assertEquals(a.get(0, 1).imag, prod.get(0, 1).imag, tol);
        assertEquals(a.get(1, 0).real, prod.get(1, 0).real, tol);
        assertEquals(a.get(1, 0).imag, prod.get(1, 0).imag, tol);
        assertEquals(a.get(1, 1).real, prod.get(1, 1).real, tol);
        assertEquals(a.get(1, 1).imag, prod.get(1, 1).imag, tol);
    }

    @Test
    public void qr_complexMatrix_QisUnitary() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 2}, {3, 4}},
            new double[][]{{0, 1}, {2, 0}}
        );
        Tuple2<IComplexMatrix, IComplexMatrix> qrResult = a.qr();
        IComplexMatrix Q = qrResult._1;
        IComplexMatrix QH = Q.conjugateTranspose();
        IComplexMatrix ident = QH.multiply(Q);
        double tol = 1e-8;
        assertEquals(1, ident.get(0, 0).real, tol);
        assertEquals(1, ident.get(1, 1).real, tol);
        assertEquals(0, ident.get(0, 1).real, tol);
        assertEquals(0, ident.get(1, 0).real, tol);
    }

    @Test
    public void qr_identity_isIdentity() {
        IComplexMatrix eye = IComplexMatrix.fromRealImag(
            new double[][]{{1, 0}, {0, 1}}, new double[2][2]
        );
        Tuple2<IComplexMatrix, IComplexMatrix> qrResult = eye.qr();
        IComplexMatrix Q = qrResult._1;
        IComplexMatrix R = qrResult._2;
        double tol = 1e-8;
        assertEquals(1, Math.abs(Q.get(0, 0).real), tol);
        assertEquals(1, Math.abs(R.get(0, 0).real), tol);
        assertEquals(1, Math.abs(R.get(1, 1).real), tol);
    }
}
