package com.yishape.lab.math.linalg.complex;

import com.yishape.lab.math.core.Complex;
import com.yishape.lab.util.Tuple2;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ComplexMatrixEigenDecompositionTest {

    @Test
    public void eigen_diagonal_returnsDiagonalElements() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{5, 0}, {0, 3}}, new double[][]{{1, 0}, {0, 2}}
        );
        Tuple2<IComplexMatrix.IComplexVector, IComplexMatrix> eig = a.eigen();
        IComplexMatrix.IComplexVector vals = eig._1;
        assertEquals(2, vals.length());
        boolean has5i1 = false, has3i2 = false;
        for (int i = 0; i < 2; i++) {
            if (Math.abs(vals.get(i).real - 5) < 1e-6 && Math.abs(vals.get(i).imag - 1) < 1e-6) has5i1 = true;
            if (Math.abs(vals.get(i).real - 3) < 1e-6 && Math.abs(vals.get(i).imag - 2) < 1e-6) has3i2 = true;
        }
        assertTrue(has5i1, "Should find eigenvalue 5+i");
        assertTrue(has3i2, "Should find eigenvalue 3+2i");
    }

    @Test
    public void eigen_smallRealMatrix_hasExpectedDimensions() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{0, 1}, {-1, 0}}, new double[2][2]
        );
        Tuple2<IComplexMatrix.IComplexVector, IComplexMatrix> eig = a.eigen();
        IComplexMatrix.IComplexVector vals = eig._1;
        assertEquals(2, vals.length());
        IComplexMatrix vecs = eig._2;
        assertEquals(2, vecs.rows());
        assertEquals(2, vecs.cols());
    }

    @Test
    public void eigen_identity_returnsAllOnes() {
        IComplexMatrix eye = IComplexMatrix.fromRealImag(
            new double[][]{{1, 0}, {0, 1}}, new double[2][2]
        );
        Tuple2<IComplexMatrix.IComplexVector, IComplexMatrix> eig = eye.eigen();
        IComplexMatrix.IComplexVector vals = eig._1;
        assertEquals(1, vals.get(0).real, 1e-6);
        assertEquals(1, vals.get(1).real, 1e-6);
        assertEquals(0, vals.get(0).imag, 1e-6);
        assertEquals(0, vals.get(1).imag, 1e-6);
    }

    @Test
    public void eigen_hermitianMatrix_eigenpairResidualSmall() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{3, 1}, {1, 2}},
            new double[][]{{0, 1}, {-1, 0}}
        );
        Tuple2<IComplexMatrix.IComplexVector, IComplexMatrix> eig = a.eigen();
        IComplexMatrix.IComplexVector vals = eig._1;
        IComplexMatrix vecs = eig._2;
        assertEquals(2, vals.length());
        assertEquals(2, vecs.rows());
        assertEquals(2, vecs.cols());
        double tol = 1e-4;
        for (int i = 0; i < 2; i++) {
            Complex lambda = vals.get(i);
            double[] vReal = new double[2], vImag = new double[2];
            for (int j = 0; j < 2; j++) {
                vReal[j] = vecs.get(j, i).real;
                vImag[j] = vecs.get(j, i).imag;
            }
            IComplexMatrix.IComplexVector vi = IComplexMatrix.IComplexVector.fromRealImag(vReal, vImag);
            IComplexMatrix.IComplexVector avi = a.mmul(vi);
            for (int j = 0; j < 2; j++) {
                Complex expected = lambda.multiply(vi.get(j));
                assertEquals(expected.real, avi.get(j).real, tol);
                assertEquals(expected.imag, avi.get(j).imag, tol);
            }
        }
    }

    @Test
    public void eigen_nonSquare_throwsException() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 2}}, new double[][]{{0, 0}}
        );
        assertThrows(IllegalStateException.class, a::eigen);
    }
}
