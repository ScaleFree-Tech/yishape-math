package com.yishape.lab.math.linalg.sparse;

import com.yishape.lab.math.linalg.IVector;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SparseMatrixArithmeticTest {

    @Test
    public void add_twoMatrices_producesCorrectSum() {
        ISparseMatrix a = ISparseMatrix.diag(new double[]{1, 2, 3});
        ISparseMatrix b = ISparseMatrix.diag(new double[]{4, 5, 6});
        ISparseMatrix c = a.add(b);
        assertEquals(5.0, c.get(0, 0));
        assertEquals(7.0, c.get(1, 1));
        assertEquals(9.0, c.get(2, 2));
    }

    @Test
    public void add_dimensionMismatch_throwsException() {
        ISparseMatrix a = ISparseMatrix.eye(3);
        ISparseMatrix b = ISparseMatrix.eye(2);
        assertThrows(IllegalArgumentException.class, () -> a.add(b));
    }

    @Test
    public void sub_twoMatrices_producesCorrectDifference() {
        ISparseMatrix a = ISparseMatrix.diag(new double[]{5, 5, 5});
        ISparseMatrix b = ISparseMatrix.diag(new double[]{2, 3, 4});
        ISparseMatrix c = a.sub(b);
        assertEquals(3.0, c.get(0, 0));
        assertEquals(2.0, c.get(1, 1));
        assertEquals(1.0, c.get(2, 2));
    }

    @Test
    public void scale_twice_equalsOriginal() {
        ISparseMatrix a = ISparseMatrix.diag(new double[]{1, 2, 3});
        ISparseMatrix scaled = a.scale(3.0).scale(1.0 / 3.0);
        assertEquals(1.0, scaled.get(0, 0), 1e-10);
        assertEquals(2.0, scaled.get(1, 1), 1e-10);
        assertEquals(3.0, scaled.get(2, 2), 1e-10);
    }

    @Test
    public void scale_zero_makesEmptySparseMatrix() {
        ISparseMatrix a = ISparseMatrix.diag(new double[]{1, 2, 3});
        ISparseMatrix z = a.scale(0.0);
        assertEquals(0, z.nnz());
        assertEquals(0.0, z.get(0, 0));
    }

    @Test
    public void multiply_sparseByVector_givesCorrectResult() {
        double[][] data = {{0, 2, 0}, {3, 0, 0}, {0, 0, 4}};
        ISparseMatrix a = ISparseMatrix.fromDense(data);
        IVector<Double> x = com.yishape.lab.math.linalg.Linalg.vector(new double[]{1, 2, 3});
        IVector<Double> y = a.multiply(x);
        assertEquals(4.0, y.get(0), 1e-10);
        assertEquals(3.0, y.get(1), 1e-10);
        assertEquals(12.0, y.get(2), 1e-10);
    }

    @Test
    public void multiply_twoSparseMatrices_producesCorrectResult() {
        double[][] aData = {{1, 2, 0}, {0, 3, 0}, {0, 0, 4}};
        double[][] bData = {{0, 1, 0}, {2, 0, 0}, {0, 0, 3}};
        ISparseMatrix a = ISparseMatrix.fromDense(aData);
        ISparseMatrix b = ISparseMatrix.fromDense(bData);
        ISparseMatrix c = a.multiply(b);

        assertEquals(4.0, c.get(0, 0), 1e-10);
        assertEquals(1.0, c.get(0, 1), 1e-10);
        assertEquals(6.0, c.get(1, 0), 1e-10);
        assertEquals(12.0, c.get(2, 2), 1e-10);
    }

    @Test
    public void transpose_squareMatrix_swapsDimensions() {
        double[][] data = {{0, 1, 0}, {0, 0, 2}, {3, 0, 0}};
        ISparseMatrix a = ISparseMatrix.fromDense(data);
        ISparseMatrix t = a.transpose();
        assertEquals(0.0, t.get(0, 0));
        assertEquals(0.0, t.get(0, 1));
        assertEquals(3.0, t.get(0, 2));
        assertEquals(1.0, t.get(1, 0));
        assertEquals(2.0, t.get(2, 1));
    }

    @Test
    public void frobeniusNorm_diagonalMatrix_equalsRootSumSquares() {
        ISparseMatrix a = ISparseMatrix.diag(new double[]{3, 4});
        assertEquals(5.0, a.frobeniusNorm(), 1e-10);
    }

    @Test
    public void hadamard_twoMatrices_elementWiseProduct() {
        ISparseMatrix a = ISparseMatrix.diag(new double[]{2, 3, 4});
        ISparseMatrix b = ISparseMatrix.diag(new double[]{5, 6, 7});
        ISparseMatrix h = a.hadamard(b);
        assertEquals(10.0, h.get(0, 0));
        assertEquals(18.0, h.get(1, 1));
        assertEquals(28.0, h.get(2, 2));
        assertEquals(3, h.nnz());
    }
}
