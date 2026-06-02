package com.yishape.lab.math.linalg.complex;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ComplexBoundaryTest {

    @Test
    public void zeroMatrix_allOps_safeResults() {
        IComplexMatrix z = IComplexMatrix.fromRealImag(
            new double[][]{{0, 0}, {0, 0}}, new double[2][2]
        );
        assertEquals(0, z.frobeniusNorm(), 1e-10);
        assertEquals(0, z.trace().real, 1e-10);
        IComplexMatrix copy = z.copy();
        assertEquals(0, copy.get(0, 0).real, 1e-10);
    }

    @Test
    public void singleElement_square_worksAsScalar() {
        IComplexMatrix s = IComplexMatrix.fromRealImag(
            new double[][]{{42}}, new double[][]{{17}}
        );
        assertEquals(42, s.get(0, 0).real, 1e-10);
        assertEquals(17, s.get(0, 0).imag, 1e-10);
        assertTrue(s.isSquare());
        assertEquals(42, s.det().real, 1e-10);
        assertEquals(17, s.det().imag, 1e-10);
    }

    @Test
    public void pureRealMatrix_conjugate_isSame() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 2}, {3, 4}}, new double[2][2]
        );
        IComplexMatrix conj = a.conjugate();
        assertEquals(a.get(0, 0).real, conj.get(0, 0).real, 1e-10);
        assertEquals(a.get(0, 0).imag, conj.get(0, 0).imag, 1e-10);
    }

    @Test
    public void pureImaginaryMatrix_conjugate_isNegated() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{0, 0}, {0, 0}}, new double[][]{{1, 2}, {3, 4}}
        );
        IComplexMatrix conj = a.conjugate();
        assertEquals(-1, conj.get(0, 0).imag, 1e-10);
        assertEquals(-2, conj.get(0, 1).imag, 1e-10);
    }

    @Test
    public void dimensionMismatch_multiply_throws() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[2][3], new double[2][3]
        );
        IComplexMatrix b = IComplexMatrix.fromRealImag(
            new double[2][3], new double[2][3]
        );
        assertThrows(Exception.class, () -> a.multiply(b));
    }

    @Test
    public void rectangularMatrix_inverse_throws() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 2, 3}}, new double[1][3]
        );
        assertThrows(IllegalStateException.class, a::inv);
    }
}
