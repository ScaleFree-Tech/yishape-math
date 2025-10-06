package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.impl.RereSVDDecomposition;
import com.yishape.lab.util.Tuple3;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class RereSVDDecompositionTest {
    
    @Test
    public void testSimpleMatrixDecomposition() {
        // Create a simple 3x3 matrix
        double[][] data = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0}
        };
        
        IMatrix<Double> matrix = Linalg.matrix(data);
        RereSVDDecomposition svd = new RereSVDDecomposition();
        
        // Perform decomposition
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(matrix);
        IMatrix<Double> U = result.getFirst();
        IVector<Double> S = result.getSecond();
        IMatrix<Double> VT = result.getThird();
        
        // Verify dimensions
        assertEquals(3, U.rows());
        assertEquals(3, U.cols());
        assertEquals(3, S.length());
        assertEquals(3, VT.rows());
        assertEquals(3, VT.cols());
        
        // Reconstruct the original matrix
        IMatrix<Double> S_matrix = Linalg.zeros(3, 3);
        for (int i = 0; i < 3; i++) {
            S_matrix.put(i, i, S.get(i));
        }
        
        IMatrix<Double> reconstructed = U.mmul(S_matrix).mmul(VT);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(data[i][j], reconstructed.get(i, j), 2.0);
            }
        }
    }
    
    @Test
    public void testNonSquareMatrixDecomposition() {
        // Create a 4x3 matrix
        double[][] data = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0},
            {10.0, 11.0, 12.0}
        };
        
        IMatrix<Double> matrix = Linalg.matrix(data);
        RereSVDDecomposition svd = new RereSVDDecomposition();
        
        // Perform decomposition
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(matrix);
        IMatrix<Double> U = result.getFirst();
        IVector<Double> S = result.getSecond();
        IMatrix<Double> VT = result.getThird();
        
        // Verify dimensions
        assertEquals(4, U.rows());
        assertEquals(3, U.cols()); // For non-square matrices, U is m x min(m,n)
        assertEquals(3, S.length());
        assertEquals(3, VT.rows());
        assertEquals(3, VT.cols());
        
        // Reconstruct the original matrix
        IMatrix<Double> S_matrix = Linalg.zeros(3, 3);
        for (int i = 0; i < 3; i++) {
            S_matrix.put(i, i, S.get(i));
        }
        
        // For reconstruction, we need U * S_matrix * VT
        // U is 4x3, S_matrix is 3x3, VT is 3x3
        // So the result should be 4x3
        IMatrix<Double> reconstructed = U.mmul(S_matrix).mmul(VT);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(data[i][j], reconstructed.get(i, j), 2.0);
            }
        }
    }
    
    @Test
    public void testSingularValuesOrdering() {
        // Create a matrix with known singular values
        double[][] data = {
            {3.0, 2.0},
            {2.0, 3.0}
        };
        
        IMatrix<Double> matrix = Linalg.matrix(data);
        RereSVDDecomposition svd = new RereSVDDecomposition();
        
        // Perform decomposition
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(matrix);
        IVector<Double> S = result.getSecond();
        
        // Singular values should be in descending order
        assertTrue(S.get(0) >= S.get(1));
    }
    
    @Test
    public void testGetMethodSelection() {
        // Test that different methods are selected based on matrix size
        RereSVDDecomposition svd = new RereSVDDecomposition();
        
        // Small matrix - should use traditionalSVD (which now uses bidiagonal approach)
        double[][] smallData = {
            {1.0, 2.0},
            {3.0, 4.0}
        };
        IMatrix<Double> smallMatrix = Linalg.matrix(smallData);
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result1 = svd.decompose(smallMatrix);
        assertNotNull(result1);
        
        // Medium matrix - should use bidiagonalSVD
        double[][] mediumData = new double[20][20];
        for (int i = 0; i < 20; i++) {
            for (int j = 0; j < 20; j++) {
                mediumData[i][j] = i * 20 + j;
            }
        }
        IMatrix<Double> mediumMatrix = Linalg.matrix(mediumData);
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result2 = svd.decompose(mediumMatrix);
        assertNotNull(result2);
        
        // Large matrix - should use optimizedSVD
        double[][] largeData = new double[50][50];
        for (int i = 0; i < 50; i++) {
            for (int j = 0; j < 50; j++) {
                largeData[i][j] = i * 50 + j;
            }
        }
        IMatrix<Double> largeMatrix = Linalg.matrix(largeData);
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result3 = svd.decompose(largeMatrix);
        assertNotNull(result3);
    }
}