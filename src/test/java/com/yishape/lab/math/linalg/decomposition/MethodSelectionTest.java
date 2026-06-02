package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.decomposition.impl.RereSVDDecompBlas2;
import com.yishape.lab.util.Tuple3;
import com.yishape.lab.math.linalg.IVector;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class MethodSelectionTest {

    @Test
    public void testTraditionalSVDMethodSelection() {
        // Create a small 2x2 matrix (size = 4 < 1000, should use traditionalSVD)
        double[][] smallData = {
            {1.0, 2.0},
            {3.0, 4.0}
        };
        
        IMatrix<Double> smallMatrix = Linalg.matrix(smallData);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        
        // Perform decomposition - should use traditionalSVD
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result =
            svd.decompose(smallMatrix);
        
        // Verify that decomposition completed successfully
        assertNotNull(result.getFirst());
        assertNotNull(result.getSecond());
        assertNotNull(result.getThird());
        
        // Verify dimensions
        assertEquals(2, result.getFirst().rows());
        assertEquals(2, result.getFirst().cols());
        assertEquals(2, result.getSecond().length());
        assertEquals(2, result.getThird().rows());
        assertEquals(2, result.getThird().cols());
    }
    
    @Test
    public void testBidiagonalSVDMethodSelection() {
        // Create a medium 15x15 matrix (size = 225, between 1000 and 10000, should use bidiagonalSVD)
        double[][] mediumData = new double[15][15];
        for (int i = 0; i < 15; i++) {
            for (int j = 0; j < 15; j++) {
                mediumData[i][j] = i * 15 + j + 1;
            }
        }
        
        IMatrix<Double> mediumMatrix = Linalg.matrix(mediumData);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        
        // Perform decomposition - should use bidiagonalSVD
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result =
            svd.decompose(mediumMatrix);
        
        // Verify that decomposition completed successfully
        assertNotNull(result.getFirst());
        assertNotNull(result.getSecond());
        assertNotNull(result.getThird());
        
        // Verify dimensions
        assertEquals(15, result.getFirst().rows());
        assertEquals(15, result.getFirst().cols());
        assertEquals(15, result.getSecond().length());
        assertEquals(15, result.getThird().rows());
        assertEquals(15, result.getThird().cols());
    }
    
    @Test
    public void testOptimizedSVDMethodSelection() {
        // Create a large 50x50 matrix (size = 2500 > 10000, should use optimizedSVD)
        double[][] largeData = new double[50][50];
        for (int i = 0; i < 50; i++) {
            for (int j = 0; j < 50; j++) {
                largeData[i][j] = i * 50 + j + 1;
            }
        }
        
        IMatrix<Double> largeMatrix = Linalg.matrix(largeData);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        
        // Perform decomposition - should use optimizedSVD
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result =
            svd.decompose(largeMatrix);
        
        // Verify that decomposition completed successfully
        assertNotNull(result.getFirst());
        assertNotNull(result.getSecond());
        assertNotNull(result.getThird());
        
        // Verify dimensions
        assertEquals(50, result.getFirst().rows());
        assertEquals(50, result.getFirst().cols());
        assertEquals(50, result.getSecond().length());
        assertEquals(50, result.getThird().rows());
        assertEquals(50, result.getThird().cols());
    }
}