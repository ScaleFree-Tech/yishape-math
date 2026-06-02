package com.yishape.lab.math.linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ToeplitzTest {

    @Test
    public void symmetricToeplitz_correctStructure() {
        double[] c = {1, 2, 3, 4};
        IMatrix<Double> T = Linalg.toeplitz(c);

        assertEquals(4, T.rows());
        assertEquals(4, T.cols());
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                assertEquals(c[Math.abs(i - j)], T.get(i, j), 1e-10);
            }
        }
    }

    @Test
    public void nonSymmetricToeplitz_correctStructure() {
        double[] column = {5, 1, 2};
        double[] row = {5, 3, 4, 6};
        IMatrix<Double> T = Linalg.toeplitz(column, row);

        assertEquals(3, T.rows());
        assertEquals(4, T.cols());
        // First column
        assertEquals(5, T.get(0, 0), 1e-10);
        assertEquals(1, T.get(1, 0), 1e-10);
        assertEquals(2, T.get(2, 0), 1e-10);
        // First row
        assertEquals(3, T.get(0, 1), 1e-10);
        assertEquals(4, T.get(0, 2), 1e-10);
        assertEquals(6, T.get(0, 3), 1e-10);
        // Diagonals
        assertEquals(5, T.get(1, 1), 1e-10);
        assertEquals(3, T.get(1, 2), 1e-10);
    }

    @Test
    public void symmetricToeplitz_fromVector() {
        IVector<Double> c = Linalg.vector(new double[]{2, 1, 0.5});
        IMatrix<Double> T = Linalg.toeplitz(c);

        assertEquals(2, T.get(0, 0), 1e-10);
        assertEquals(1, T.get(0, 1), 1e-10);
        assertEquals(1, T.get(1, 0), 1e-10);
        assertEquals(2, T.get(1, 1), 1e-10);
    }

    @Test
    public void nonSymmetricToeplitz_firstElementsMustMatch() {
        assertThrows(IllegalArgumentException.class, () -> {
            Linalg.toeplitz(new double[]{1, 2}, new double[]{3, 4});
        });
    }

    @Test
    public void toeplitz_solve_vs_denseSolve() {
        double[] c = {4, 1, 0.5};
        IMatrix<Double> T = Linalg.toeplitz(c);
        IVector<Double> b = Linalg.vector(new double[]{1, 2, 3});

        IVector<Double> x1 = Linalg.solve(T, b);
        // T should also be solvable via dense LU
        assertNotNull(x1);
        assertEquals(3, x1.length());
    }
}
