package com.yishape.lab.math.linalg.sparse;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SparseMatrixSpecialTest {

    @Test
    public void diagonalSparseMatrix_hasCorrectDiagonal() {
        ISpecialSparseMatrix.DiagonalSparseMatrix d = new ISpecialSparseMatrix.DiagonalSparseMatrix(new double[]{3, 7, 1});
        assertEquals(3, d.rows());
        assertEquals(3, d.cols());
        assertEquals(3, d.nnz());
        assertEquals(3.0, d.get(0, 0));
        assertEquals(7.0, d.get(1, 1));
        assertEquals(1.0, d.get(2, 2));
        assertEquals(0.0, d.get(0, 1));
    }

    @Test
    public void diagonalSparseMatrix_add_diagonalAddition() {
        ISparseMatrix a = new ISpecialSparseMatrix.DiagonalSparseMatrix(new double[]{1, 2});
        ISparseMatrix b = new ISpecialSparseMatrix.DiagonalSparseMatrix(new double[]{3, 4});
        ISparseMatrix c = a.add(b);
        assertEquals(4.0, c.get(0, 0));
        assertEquals(6.0, c.get(1, 1));
    }

    @Test
    public void tridiagonalSparseMatrix_hasCorrectValues() {
        double[] lower = {1, 2};
        double[] main = {4, 5, 6};
        double[] upper = {3, 4};
        ISpecialSparseMatrix.TridiagonalSparseMatrix t = new ISpecialSparseMatrix.TridiagonalSparseMatrix(lower, main, upper);
        assertEquals(3, t.rows());
        assertEquals(5.0, t.get(1, 1));
        assertEquals(3.0, t.get(0, 1));
        assertEquals(1.0, t.get(1, 0));
        assertEquals(0.0, t.get(0, 2));
    }

    @Test
    public void identitySparseMatrix_hasOnesOnDiagonal() {
        ISpecialSparseMatrix.IdentitySparseMatrix eye = new ISpecialSparseMatrix.IdentitySparseMatrix(4);
        assertEquals(4, eye.rows());
        assertEquals(4, eye.cols());
        assertEquals(4, eye.nnz());
        for (int i = 0; i < 4; i++) {
            assertEquals(1.0, eye.get(i, i));
        }
    }

    @Test
    public void identitySparseMatrix_transpose_isSame() {
        ISpecialSparseMatrix.IdentitySparseMatrix eye = new ISpecialSparseMatrix.IdentitySparseMatrix(3);
        ISparseMatrix t = eye.transpose();
        assertEquals(3, t.nnz());
        assertEquals(1.0, t.get(1, 1));
    }

    @Test
    public void identitySparseMatrix_multiply_vector_isSame() {
        ISpecialSparseMatrix.IdentitySparseMatrix eye = new ISpecialSparseMatrix.IdentitySparseMatrix(3);
        com.yishape.lab.math.linalg.IVector<Double> x = com.yishape.lab.math.linalg.Linalg.vector(new double[]{2, 5, 9});
        com.yishape.lab.math.linalg.IVector<Double> y = eye.multiply(x);
        assertEquals(2.0, y.get(0));
        assertEquals(5.0, y.get(1));
        assertEquals(9.0, y.get(2));
    }

    @Test
    public void zeroSparseMatrix_isAllZeros() {
        ISpecialSparseMatrix.ZeroSparseMatrix z = new ISpecialSparseMatrix.ZeroSparseMatrix(3, 4);
        assertEquals(3, z.rows());
        assertEquals(4, z.cols());
        assertEquals(0, z.nnz());
        assertEquals(0.0, z.get(0, 0));
        assertEquals(0.0, z.get(2, 3));
    }

    @Test
    public void zeroSparseMatrix_add_identity_givesIdentity() {
        ISpecialSparseMatrix.ZeroSparseMatrix z = new ISpecialSparseMatrix.ZeroSparseMatrix(3, 3);
        ISpecialSparseMatrix.IdentitySparseMatrix eye = new ISpecialSparseMatrix.IdentitySparseMatrix(3);
        ISparseMatrix c = z.add(eye);
        assertEquals(1.0, c.get(0, 0));
        assertEquals(1.0, c.get(1, 1));
        assertEquals(1.0, c.get(2, 2));
        assertEquals(0.0, c.get(0, 1));
    }
}
