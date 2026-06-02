package com.yishape.lab.math.linalg.complex;

import com.yishape.lab.util.Tuple2;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ComplexMatrixLUDecompositionTest {

    @Test
    public void lu_diagonal_returnsIdentityLAndDiagonalU() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{2, 0}, {0, 3}},
            new double[][]{{0, 0}, {0, 0}}
        );
        Tuple2<IComplexMatrix, IComplexMatrix> lu = a.lu();
        IComplexMatrix L = lu._1;
        IComplexMatrix U = lu._2;
        assertEquals(1, L.get(0, 0).real, 1e-10);
        assertEquals(0, L.get(0, 0).imag, 1e-10);
        assertEquals(0, L.get(0, 1).real, 1e-10);
        assertEquals(2, U.get(0, 0).real, 1e-10);
        assertEquals(3, U.get(1, 1).real, 1e-10);
    }

    @Test
    public void lu_smallMatrix_reconstructsOriginal() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 2}, {3, 4}},
            new double[][]{{5, 6}, {7, 8}}
        );
        Tuple2<IComplexMatrix, IComplexMatrix> lu = a.lu();
        IComplexMatrix L = lu._1;
        IComplexMatrix U = lu._2;
        IComplexMatrix prod = L.multiply(U);
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
    public void lu_identity_isIdentity() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 0}, {0, 1}}, new double[2][2]
        );
        Tuple2<IComplexMatrix, IComplexMatrix> lu = a.lu();
        assertEquals(1, lu._2.get(0, 0).real, 1e-10);
        assertEquals(1, lu._2.get(1, 1).real, 1e-10);
    }

    @Test
    public void lu_nonSquare_throwsException() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 2, 3}, {4, 5, 6}}, new double[2][3]
        );
        assertThrows(IllegalStateException.class, a::lu);
    }

    @Test
    public void lu_complexMatrix_reconstructsOriginal() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 2}, {3, 4}},
            new double[][]{{5, 6}, {7, 8}}
        );
        Tuple2<IComplexMatrix, IComplexMatrix> lu = a.lu();
        IComplexMatrix L = lu._1;
        IComplexMatrix U = lu._2;
        IComplexMatrix prod = L.multiply(U);
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
    public void lu_3x3_returnsTriangularFactors() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{4, 2, 1}, {2, 5, 3}, {1, 3, 6}},
            new double[3][3]
        );
        Tuple2<IComplexMatrix, IComplexMatrix> lu = a.lu();
        IComplexMatrix L = lu._1;
        IComplexMatrix U = lu._2;
        for (int i = 0; i < 3; i++) {
            assertEquals(1, L.get(i, i).real, 1e-10);
            assertEquals(0, L.get(i, i).imag, 1e-10);
        }
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < i; j++) {
                assertEquals(0, U.get(i, j).real, 1e-10, "U should be upper triangular");
            }
        }
    }
}
