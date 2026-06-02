package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.decomposition.impl.RereSVDDecompBlas2;
import com.yishape.lab.util.Tuple3;
import com.yishape.lab.math.linalg.IVector;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TraditionalSVDTest {

    @Test
    public void testTraditionalSVD() {
        // Create a simple 3x3 matrix
        double[][] data = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0}
        };
        
        IDoubleMatrix matrix = (IDoubleMatrix) Linalg.matrix(data);
        
        // Create SVD decomposition
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        
        // Force use of traditionalSVD by making the matrix small
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result =
            svd.decompose(matrix, 1e-12, 1000);
        
        IMatrix<Double> U = result.getFirst();
        IVector<Double> singularValues = result.getSecond();
        IMatrix<Double> VT = result.getThird();
        
        // Verify dimensions
        assertEquals(3, U.rows());
        assertEquals(3, U.cols());
        assertEquals(3, VT.rows());
        assertEquals(3, VT.cols());
        assertEquals(3, singularValues.length());
        
        // Verify that U and VT are orthogonal
        IMatrix<Double> UUT = U.mmul(U.transposeNew());
        IMatrix<Double> VVT = VT.transposeNew().mmul(VT);
        
        double tolerance = 1e-10;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i == j) {
                    assertEquals(1.0, UUT.get(i, j), tolerance);
                    assertEquals(1.0, VVT.get(i, j), tolerance);
                } else {
                    assertEquals(0.0, UUT.get(i, j), tolerance);
                    assertEquals(0.0, VVT.get(i, j), tolerance);
                }
            }
        }
        
        // Reconstruct the original matrix
        IMatrix<Double> S = Linalg.diag(singularValues);
        IMatrix<Double> reconstructed = U.mmul(S).mmul(VT);
        
        // Verify reconstruction
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(data[i][j], reconstructed.get(i, j), 1e-10);
            }
        }
    }
    
    @Test
    public void testTraditionalSVDNonSquare() {
        // Create a 4x3 matrix
        double[][] data = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0},
            {10.0, 11.0, 12.0}
        };
        
        IDoubleMatrix matrix = (IDoubleMatrix) Linalg.matrix(data);
        
        // Create SVD decomposition
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        
        // Force use of traditionalSVD by making the matrix small enough
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result =
            svd.decompose(matrix, 1e-12, 1000);
        
        IMatrix<Double> U = result.getFirst();
        IVector<Double> singularValues = result.getSecond();
        IMatrix<Double> VT = result.getThird();
        
        // Verify dimensions
        assertEquals(4, U.rows());
        assertEquals(3, U.cols()); // min(4,3) = 3
        assertEquals(3, VT.rows());
        assertEquals(3, VT.cols());
        assertEquals(3, singularValues.length());
        
        // Verify that U and VT have orthogonal columns/rows respectively
        IMatrix<Double> UUT = U.transposeNew().mmul(U);
        IMatrix<Double> VVT = VT.transposeNew().mmul(VT);
        
        double tolerance = 1e-10;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i == j) {
                    assertEquals(1.0, UUT.get(i, j), tolerance);
                    assertEquals(1.0, VVT.get(i, j), tolerance);
                } else {
                    assertEquals(0.0, UUT.get(i, j), tolerance);
                    assertEquals(0.0, VVT.get(i, j), tolerance);
                }
            }
        }
    }
}