package com.yishape.lab.math.linalg.complex;

import com.yishape.lab.math.core.Complex;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ComplexVectorTest {

    @Test
    public void fromRealImag_createsVectorCorrectly() {
        IComplexMatrix.IComplexVector v = IComplexMatrix.IComplexVector.fromRealImag(
            new double[]{1, 2, 3}, new double[]{4, 5, 6}
        );
        assertEquals(3, v.length());
        assertEquals(1, v.get(0).real, 1e-10);
        assertEquals(4, v.get(0).imag, 1e-10);
    }

    @Test
    public void add_twoVectors_correctSum() {
        IComplexMatrix.IComplexVector a = IComplexMatrix.IComplexVector.fromRealImag(
            new double[]{1, 2}, new double[]{3, 4}
        );
        IComplexMatrix.IComplexVector b = IComplexMatrix.IComplexVector.fromRealImag(
            new double[]{5, 6}, new double[]{7, 8}
        );
        IComplexMatrix.IComplexVector c = a.add(b);
        assertEquals(6, c.get(0).real, 1e-10);
        assertEquals(10, c.get(0).imag, 1e-10);
    }

    @Test
    public void innerProduct_orthogonalVectors_isZero() {
        IComplexMatrix.IComplexVector a = IComplexMatrix.IComplexVector.fromRealImag(
            new double[]{1, 0}, new double[]{0, 0}
        );
        IComplexMatrix.IComplexVector b = IComplexMatrix.IComplexVector.fromRealImag(
            new double[]{0, 1}, new double[]{0, 0}
        );
        Complex dot = a.innerProduct(b);
        assertEquals(0, dot.real, 1e-10);
        assertEquals(0, dot.imag, 1e-10);
    }

    @Test
    public void magnitude_unitVector_isOne() {
        IComplexMatrix.IComplexVector v = IComplexMatrix.IComplexVector.fromRealImag(
            new double[]{3, 0}, new double[]{4, 0}
        );
        assertEquals(5, v.magnitude(), 1e-10);
    }

    @Test
    public void normalize_unitVector_hasUnitMagnitude() {
        IComplexMatrix.IComplexVector v = IComplexMatrix.IComplexVector.fromRealImag(
            new double[]{3, 4}, new double[]{0, 0}
        );
        IComplexMatrix.IComplexVector u = v.normalize();
        assertEquals(1.0, u.magnitude(), 1e-8);
    }

    @Test
    public void outerProduct_returnsCorrectMatrix() {
        IComplexMatrix.IComplexVector a = IComplexMatrix.IComplexVector.fromRealImag(
            new double[]{1, 0}, new double[]{0, 1}
        );
        IComplexMatrix.IComplexVector b = IComplexMatrix.IComplexVector.fromRealImag(
            new double[]{1, 0}, new double[]{0, 0}
        );
        IComplexMatrix outer = a.outerProduct(b);
        assertEquals(1, outer.get(0, 0).real, 1e-10);
        assertEquals(0, outer.get(0, 0).imag, 1e-10);
        assertEquals(0, outer.get(1, 0).real, 1e-10);
        assertEquals(1, outer.get(1, 0).imag, 1e-10);
    }
}
