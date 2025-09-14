package com.reremouse.lab.math.compute;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.RereMatrix;
import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.util.Tuple3;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * CPU计算工具类高级功能测试
 * 验证特征分解和奇异值分解功能
 */
public class CPUComputeUtilsAdvancedTest {
    
    @Test
    public void testEigenDecomposition() {
        // 创建一个简单的2x2对称矩阵
        float[][] data = {{4, 1}, {1, 3}};
        IMatrix matrix = new RereMatrix(data);
        
        Tuple2<IVector, IMatrix> result = CPUComputeUtils.eigen(matrix);
        IVector eigenvalues = result._1;
        IMatrix eigenvectors = result._2;
        
        // 验证特征值数量
        assertEquals(2, eigenvalues.length());
        
        // 验证特征向量矩阵的维度
        assertEquals(2, eigenvectors.getRows());
        assertEquals(2, eigenvectors.getColumns());
        
        // 验证特征值是否为正数（对于正定矩阵）
        assertTrue(eigenvalues.get(0) > 0);
        assertTrue(eigenvalues.get(1) > 0);
    }
    
    @Test
    public void testSVD() {
        // 创建一个简单的2x3矩阵
        float[][] data = {{1, 2, 3}, {4, 5, 6}};
        IMatrix matrix = new RereMatrix(data);
        
        Tuple3<IMatrix, IVector, IMatrix> result = CPUComputeUtils.svd(matrix);
        IMatrix U = result._1;
        IVector singularValues = result._2;
        IMatrix VT = result._3;
        
        // 验证U的维度
        assertEquals(2, U.getRows());
        assertEquals(2, U.getColumns());
        
        // 验证奇异值数量
        assertEquals(2, singularValues.length());
        
        // 验证V^T的维度 (对于2x3矩阵，V是3x3，V^T也是3x3)
        assertEquals(3, VT.getRows());
        assertEquals(3, VT.getColumns());
        
        // 验证奇异值是否为非负数（奇异值可以为0）
        assertTrue(singularValues.get(0) >= 0);
        assertTrue(singularValues.get(1) >= 0);
        
        // 至少第一个奇异值应该大于0（除非矩阵为零矩阵）
        assertTrue(singularValues.get(0) > 0);
    }
    
    @Test
    public void testEigenDecompositionWithNonSquareMatrix() {
        // 测试非方阵应该抛出异常
        float[][] data = {{1, 2, 3}, {4, 5, 6}};
        IMatrix matrix = new RereMatrix(data);
        
        assertThrows(IllegalArgumentException.class, () -> {
            CPUComputeUtils.eigen(matrix);
        });
    }
    
    @Test
    public void testSVDWithSmallMatrix() {
        // 测试小矩阵的SVD
        float[][] data = {{1, 2}, {3, 4}};
        IMatrix matrix = new RereMatrix(data);
        
        Tuple3<IMatrix, IVector, IMatrix> result = CPUComputeUtils.svd(matrix);
        IMatrix U = result._1;
        IVector singularValues = result._2;
        IMatrix VT = result._3;
        
        // 验证结果维度
        assertEquals(2, U.getRows());
        assertEquals(2, U.getColumns());
        assertEquals(2, singularValues.length());
        assertEquals(2, VT.getRows());
        assertEquals(2, VT.getColumns());
    }
}
