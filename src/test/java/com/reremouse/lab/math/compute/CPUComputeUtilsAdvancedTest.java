package com.reremouse.lab.math.compute;

import com.reremouse.lab.math.linalg.RereDoubleMatrix;
import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.util.Tuple3;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import com.reremouse.lab.math.linalg.IDoubleMatrix;
import com.reremouse.lab.math.linalg.IDoubleVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;

/**
 * CPU计算工具类高级功能测试
 * 验证特征分解和奇异值分解功能
 */
public class CPUComputeUtilsAdvancedTest {
    
    @Test
    public void testEigenDecomposition() {
        // 创建一个简单的2x2对称矩阵
        double[][] data = {{4.0, 1.0}, {1.0, 3.0}};
        IMatrix<Double> matrix = new RereDoubleMatrix(data);
        
        Tuple2<IVector<Double>, IMatrix<Double>> result = CPUComputeDoubleUtils.eigen(matrix);
        IVector<Double> eigenvalues = result._1;
        IMatrix<Double> eigenvectors = result._2;
        
        // 验证特征值数量
        assertEquals(2, eigenvalues.length());
        
        // 验证特征向量矩阵的维度
        assertEquals(2, eigenvectors.rows());
        assertEquals(2, eigenvectors.cols());
        
        // 验证特征值是否为正数（对于正定矩阵）
        assertTrue(eigenvalues.get(0) > 0);
        assertTrue(eigenvalues.get(1) > 0);
    }
    
    @Test
    public void testSVD() {
        // 创建一个简单的2x3矩阵
        double[][] data = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
        IDoubleMatrix matrix = new RereDoubleMatrix(data);
        
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = CPUComputeDoubleUtils.svd(matrix);
        IMatrix<Double> U = result._1;
        IVector<Double> singularValues = result._2;
        IMatrix<Double> VT = result._3;
        
        // 验证U的维度
        assertEquals(2, U.rows());
        assertEquals(2, U.cols());
        
        // 验证奇异值数量
        assertEquals(2, singularValues.length());
        
        // 验证V^T的维度 (对于2x3矩阵，V是3x3，V^T也是3x3)
        assertEquals(3, VT.rows());
        assertEquals(3, VT.cols());
        
        // 验证奇异值是否为非负数（奇异值可以为0）
        assertTrue(singularValues.get(0) >= 0);
        assertTrue(singularValues.get(1) >= 0);
        
        // 至少第一个奇异值应该大于0（除非矩阵为零矩阵）
        assertTrue(singularValues.get(0) > 0);
    }
    
    @Test
    public void testEigenDecompositionWithNonSquareMatrix() {
        // 测试非方阵应该抛出异常
        double[][] data = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
        IDoubleMatrix matrix = new RereDoubleMatrix(data);
        
        assertThrows(IllegalArgumentException.class, () -> {
            CPUComputeDoubleUtils.eigen(matrix);
        });
    }
    
    @Test
    public void testSVDWithSmallMatrix() {
        // 测试小矩阵的SVD
        double[][] data = {{1.0, 2.0}, {3.0, 4.0}};
        IDoubleMatrix matrix = new RereDoubleMatrix(data);
        
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = CPUComputeDoubleUtils.svd(matrix);
        IMatrix<Double> U = result._1;
        IVector<Double> singularValues = result._2;
        IMatrix<Double> VT = result._3;
        
        // 验证结果维度
        assertEquals(2, U.rows());
        assertEquals(2, U.cols());
        assertEquals(2, singularValues.length());
        assertEquals(2, VT.rows());
        assertEquals(2, VT.cols());
    }
}
