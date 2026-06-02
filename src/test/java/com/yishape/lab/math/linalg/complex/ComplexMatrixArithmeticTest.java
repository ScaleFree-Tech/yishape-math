package com.yishape.lab.math.linalg.complex;

import com.yishape.lab.math.core.Complex;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ComplexMatrixArithmeticTest {

    @Test
    public void add_twoMatrices_producesCorrectSum() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 2}, {3, 4}},
            new double[][]{{0, 1}, {2, 0}}
        );
        IComplexMatrix b = IComplexMatrix.fromRealImag(
            new double[][]{{2, 0}, {1, 3}},
            new double[][]{{1, 0}, {0, 2}}
        );
        IComplexMatrix c = a.add(b);
        assertEquals(3, c.get(0, 0).real, 1e-10);
        assertEquals(1, c.get(0, 0).imag, 1e-10);
        assertEquals(2, c.get(0, 1).real, 1e-10);
        assertEquals(1, c.get(0, 1).imag, 1e-10);
    }

    @Test
    public void sub_twoMatrices_producesCorrectDifference() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{5, 3}}, new double[][]{{1, 2}}
        );
        IComplexMatrix b = IComplexMatrix.fromRealImag(
            new double[][]{{2, 1}}, new double[][]{{0, 1}}
        );
        IComplexMatrix c = a.sub(b);
        assertEquals(3, c.get(0, 0).real, 1e-10);
        assertEquals(1, c.get(0, 0).imag, 1e-10);
        assertEquals(2, c.get(0, 1).real, 1e-10);
        assertEquals(1, c.get(0, 1).imag, 1e-10);
    }

    @Test
    public void scale_realScalar_correctlyScaled() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 2}}, new double[][]{{3, 4}}
        );
        IComplexMatrix s = a.scale(2.0);
        assertEquals(2, s.get(0, 0).real, 1e-10);
        assertEquals(6, s.get(0, 0).imag, 1e-10);
        assertEquals(4, s.get(0, 1).real, 1e-10);
        assertEquals(8, s.get(0, 1).imag, 1e-10);
    }

    @Test
    public void multiply_twoMatrices_complexProduct() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 1}, {0, 1}},
            new double[][]{{0, 1}, {1, 0}}
        );
        IComplexMatrix b = IComplexMatrix.fromRealImag(
            new double[][]{{1, 0}, {1, 0}},
            new double[][]{{1, 0}, {0, 1}}
        );
        IComplexMatrix c = a.multiply(b);
        double tol = 1e-10;
        // row 0: (1+i0)*(1+i1) + (1+i1)*(1+i0) = (1*1-0*1) + i(1*1+0*1) + (1*1-1*0) + i(1*0+1*1)
        assertEquals(2, c.get(0, 0).real, tol);
        assertEquals(2, c.get(0, 0).imag, tol);
    }

    @Test
    public void conjugate_negatesImaginaryPart() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 2}}, new double[][]{{3, 4}}
        );
        IComplexMatrix conj = a.conjugate();
        assertEquals(1, conj.get(0, 0).real, 1e-10);
        assertEquals(-3, conj.get(0, 0).imag, 1e-10);
        assertEquals(2, conj.get(0, 1).real, 1e-10);
        assertEquals(-4, conj.get(0, 1).imag, 1e-10);
    }

    @Test
    public void transpose_swapsDimensions() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 2, 3}}, new double[][]{{4, 5, 6}}
        );
        IComplexMatrix t = a.transpose();
        assertEquals(3, t.rows());
        assertEquals(1, t.cols());
        assertEquals(1, t.get(0, 0).real, 1e-10);
        assertEquals(4, t.get(0, 0).imag, 1e-10);
    }

    @Test
    public void frobeniusNorm_pureReal_equalsRealNorm() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{3, 0}, {0, 4}}, new double[2][2]
        );
        assertEquals(5.0, a.frobeniusNorm(), 1e-10);
    }

    @Test
    public void hadamard_elementWiseProduct() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{2, 3}}, new double[][]{{0, 0}}
        );
        IComplexMatrix b = IComplexMatrix.fromRealImag(
            new double[][]{{4, 5}}, new double[][]{{0, 0}}
        );
        IComplexMatrix h = a.hadamard(b);
        assertEquals(8, h.get(0, 0).real, 1e-10);
        assertEquals(15, h.get(0, 1).real, 1e-10);
    }
}
