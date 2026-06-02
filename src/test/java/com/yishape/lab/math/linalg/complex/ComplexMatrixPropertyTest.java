package com.yishape.lab.math.linalg.complex;

import com.yishape.lab.math.core.Complex;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ComplexMatrixPropertyTest {

    @Test
    public void isSquare_squareMatrix_returnsTrue() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 2}, {3, 4}}, new double[2][2]
        );
        assertTrue(a.isSquare());
    }

    @Test
    public void isSquare_rectangularMatrix_returnsFalse() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 2, 3}}, new double[1][3]
        );
        assertFalse(a.isSquare());
    }

    @Test
    public void isHermitian_hermitianMatrix_returnsTrue() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 2}, {2, 1}}, new double[][]{{0, -3}, {3, 0}}
        );
        assertTrue(a.isHermitian());
    }

    @Test
    public void isHermitian_nonHermitian_returnsFalse() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 2}, {3, 4}}, new double[][]{{0, 0}, {0, 0}}
        );
        assertFalse(a.isHermitian());
    }

    @Test
    public void trace_squareMatrix_sumOfDiagonal() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 2}, {3, 4}}, new double[][]{{5, 6}, {7, 8}}
        );
        assertEquals(5, a.trace().real, 1e-10);
        assertEquals(13, a.trace().imag, 1e-10);
    }

    @Test
    public void det_2x2_uses2x2Formula() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{3, 0}, {0, 2}}, new double[2][2]
        );
        assertEquals(6, a.det().real, 1e-10);
        assertEquals(0, a.det().imag, 1e-10);
    }

    @Test
    public void rank_fullRank_smallMatrix() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 0}, {0, 1}}, new double[2][2]
        );
        assertEquals(2, a.rank());
    }

    @Test
    public void rank_rankDeficient_returnsOne() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 1}, {1, 1}}, new double[2][2]
        );
        assertEquals(1, a.rank());
    }

    @Test
    public void cond_identity_isOne() {
        IComplexMatrix eye = IComplexMatrix.fromRealImag(
            new double[][]{{1, 0}, {0, 1}}, new double[2][2]
        );
        assertEquals(1.0, eye.cond(), 1e-6);
    }

    @Test
    public void diag_squareMatrix_returnsDiagonal() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 2}, {3, 4}}, new double[][]{{5, 6}, {7, 8}}
        );
        Complex[] d = a.diag();
        assertEquals(1, d[0].real, 1e-10);
        assertEquals(5, d[0].imag, 1e-10);
        assertEquals(4, d[1].real, 1e-10);
        assertEquals(8, d[1].imag, 1e-10);
    }

    @Test
    public void inv_complexMatrix_reconstructsIdentity() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 2}, {3, 4}},
            new double[][]{{0, 1}, {2, 0}}
        );
        IComplexMatrix ai = a.inv();
        IComplexMatrix prod = a.multiply(ai);
        double tol = 1e-8;
        assertEquals(1, prod.get(0, 0).real, tol);
        assertEquals(0, prod.get(0, 0).imag, tol);
        assertEquals(0, prod.get(0, 1).real, tol);
        assertEquals(0, prod.get(0, 1).imag, tol);
        assertEquals(0, prod.get(1, 0).real, tol);
        assertEquals(0, prod.get(1, 0).imag, tol);
        assertEquals(1, prod.get(1, 1).real, tol);
        assertEquals(0, prod.get(1, 1).imag, tol);
    }

    @Test
    public void copy_isIndependent() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 2}}, new double[][]{{3, 4}}
        );
        IComplexMatrix b = a.copy();
        assertEquals(a.get(0, 0).real, b.get(0, 0).real, 1e-10);
        b.put(0, 0, Complex.ZERO);
        assertEquals(1, a.get(0, 0).real, 1e-10, "Original should be unaffected");
    }
}
