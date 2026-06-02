package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.ISVDDecomposition;
import com.yishape.lab.util.Tuple3;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class RereSVDDecompositionTest {

    @Test
    public void testEmptyMatrix() {
        ISVDDecomposition svd = new RereSVDDecompBlas2();
        
        // 测试空矩阵情况
        try {
            IMatrix<Double> emptyMatrix = Linalg.matrix(new double[0][0]);
            svd.decompose(emptyMatrix);
            fail("Should have thrown IllegalArgumentException for empty matrix");
        } catch (IllegalArgumentException e) {
            // 预期的异常
            assertTrue(e.getMessage().contains("Matrix cannot be empty") || 
                      e.getMessage().contains("矩阵不能为空"), 
                      "Exception message should contain 'Matrix cannot be empty' or '矩阵不能为空'");
        }
    }

    @Test
    public void testZeroColumnsMatrix() {
        ISVDDecomposition svd = new RereSVDDecompBlas2();
        IMatrix<Double> z = Linalg.matrix(new double[2][0]);
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> svd.decompose(z));
        assertTrue(ex.getMessage().contains("Matrix cannot be empty"));
    }

    @Test
    public void testBasicSVD() {
        ISVDDecomposition svd = new RereSVDDecompBlas2();
        
        // 创建一个简单的测试矩阵
        double[][] data = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0}
        };
        
        IMatrix<Double> matrix = Linalg.matrix(data);
        Tuple3<IMatrix<Double>, com.yishape.lab.math.linalg.IVector<Double>, IMatrix<Double>> result = 
            svd.decompose(matrix);
        
        assertNotNull(result.getFirst(), "U matrix should not be null");
        assertNotNull(result.getSecond(), "Singular values should not be null");
        assertNotNull(result.getThird(), "V transpose matrix should not be null");
        
        // 验证矩阵维度 (3x3 matrix)
        assertEquals(3, result.getFirst().rows(), "U matrix rows");
        assertEquals(3, result.getFirst().cols(), "U matrix cols");
        assertEquals(3, result.getSecond().length(), "Singular values length");
        assertEquals(3, result.getThird().rows(), "V transpose matrix rows");
        assertEquals(3, result.getThird().cols(), "V transpose matrix cols");
    }

    @Test
    public void testNonSquareMatrix() {
        ISVDDecomposition svd = new RereSVDDecompBlas2();
        
        // 创建一个非方阵 (2x4)
        double[][] data = {
            {1.0, 2.0, 3.0, 4.0},
            {5.0, 6.0, 7.0, 8.0}
        };
        
        IMatrix<Double> matrix = Linalg.matrix(data);
        Tuple3<IMatrix<Double>, com.yishape.lab.math.linalg.IVector<Double>, IMatrix<Double>> result = 
            svd.decompose(matrix);
        
        assertNotNull(result.getFirst(), "U matrix should not be null");
        assertNotNull(result.getSecond(), "Singular values should not be null");
        assertNotNull(result.getThird(), "V transpose matrix should not be null");
        
        // 验证矩阵维度 (2x4 matrix)
        // U: 2x2 (compact form)
        // S: 2 singular values
        // V^T: 4x4 (full form, not compact)
        assertEquals(2, result.getFirst().rows(), "U matrix rows");
        assertEquals(2, result.getFirst().cols(), "U matrix cols");
        assertEquals(2, result.getSecond().length(), "Singular values length");
        assertEquals(4, result.getThird().rows(), "V transpose matrix rows");
        assertEquals(4, result.getThird().cols(), "V transpose matrix cols");
    }

    @Test
    public void testSingularValuesArePositive() {
        ISVDDecomposition svd = new RereSVDDecompBlas2();
        
        // 创建一个测试矩阵
        double[][] data = {
            {2.0, -1.0, 0.0},
            {-1.0, 2.0, -1.0},
            {0.0, -1.0, 2.0}
        };
        
        IMatrix<Double> matrix = Linalg.matrix(data);
        Tuple3<IMatrix<Double>, com.yishape.lab.math.linalg.IVector<Double>, IMatrix<Double>> result = 
            svd.decompose(matrix);
        
        // 验证所有奇异值都是非负的
        for (int i = 0; i < result.getSecond().length(); i++) {
            assertTrue(result.getSecond().get(i) >= 0, "Singular values should be non-negative");
        }
    }

    @Test
    public void testOrthogonality() {
        ISVDDecomposition svd = new RereSVDDecompBlas2();
        
        // 创建一个测试矩阵
        double[][] data = {
            {1.0, 2.0},
            {3.0, 4.0},
            {5.0, 6.0}
        };
        
        IMatrix<Double> matrix = Linalg.matrix(data);
        Tuple3<IMatrix<Double>, com.yishape.lab.math.linalg.IVector<Double>, IMatrix<Double>> result = 
            svd.decompose(matrix);
        
        // 验证U矩阵的正交性: U^T * U = I
        IMatrix<Double> u = result.getFirst();
        IMatrix<Double> utu = u.transposeNew().mmul(u);
        
        for (int i = 0; i < utu.rows(); i++) {
            for (int j = 0; j < utu.cols(); j++) {
                if (i == j) {
                    assertEquals(1.0, utu.get(i, j), 1e-10, "Diagonal elements of U^T*U should be 1");
                } else {
                    assertEquals(0.0, utu.get(i, j), 1e-10, "Off-diagonal elements of U^T*U should be 0");
                }
            }
        }
        
        // 验证V矩阵的正交性: V^T * V = I
        IMatrix<Double> vt = result.getThird();
        IMatrix<Double> v = vt.transposeNew();
        IMatrix<Double> vtv = v.transposeNew().mmul(v);
        
        for (int i = 0; i < vtv.rows(); i++) {
            for (int j = 0; j < vtv.cols(); j++) {
                if (i == j) {
                    assertEquals(1.0, vtv.get(i, j), 1e-10, "Diagonal elements of V^T*V should be 1");
                } else {
                    assertEquals(0.0, vtv.get(i, j), 1e-10, "Off-diagonal elements of V^T*V should be 0");
                }
            }
        }
    }
}