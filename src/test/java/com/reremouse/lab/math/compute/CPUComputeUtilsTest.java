package com.reremouse.lab.math.compute;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.RereMatrix;
import com.reremouse.lab.math.linalg.RereVector;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * CPU计算工具类测试
 * 验证重构后的CPU计算功能是否正常工作
 */
public class CPUComputeUtilsTest {
    
    @Test
    public void testMatrixAdd() {
        float[][] dataA = {{1, 2}, {3, 4}};
        float[][] dataB = {{5, 6}, {7, 8}};
        
        IMatrix result = CPUComputeUtils.matrixAdd(dataA, dataB);
        
        assertEquals(6, result.get(0, 0), 1e-6);
        assertEquals(8, result.get(0, 1), 1e-6);
        assertEquals(10, result.get(1, 0), 1e-6);
        assertEquals(12, result.get(1, 1), 1e-6);
    }
    
    @Test
    public void testMatrixSub() {
        float[][] dataA = {{5, 6}, {7, 8}};
        float[][] dataB = {{1, 2}, {3, 4}};
        
        IMatrix result = CPUComputeUtils.matrixSub(dataA, dataB);
        
        assertEquals(4, result.get(0, 0), 1e-6);
        assertEquals(4, result.get(0, 1), 1e-6);
        assertEquals(4, result.get(1, 0), 1e-6);
        assertEquals(4, result.get(1, 1), 1e-6);
    }
    
    @Test
    public void testMatrixScalarMultiply() {
        float[][] dataA = {{1, 2}, {3, 4}};
        float scalar = 2.0f;
        
        IMatrix result = CPUComputeUtils.matrixScalarMultiply(dataA, scalar);
        
        assertEquals(2, result.get(0, 0), 1e-6);
        assertEquals(4, result.get(0, 1), 1e-6);
        assertEquals(6, result.get(1, 0), 1e-6);
        assertEquals(8, result.get(1, 1), 1e-6);
    }
    
    @Test
    public void testMatrixTranspose() {
        float[][] dataA = {{1, 2, 3}, {4, 5, 6}};
        
        IMatrix result = CPUComputeUtils.matrixTranspose(dataA);
        
        assertEquals(1, result.get(0, 0), 1e-6);
        assertEquals(4, result.get(0, 1), 1e-6);
        assertEquals(2, result.get(1, 0), 1e-6);
        assertEquals(5, result.get(1, 1), 1e-6);
        assertEquals(3, result.get(2, 0), 1e-6);
        assertEquals(6, result.get(2, 1), 1e-6);
    }
    
    @Test
    public void testVectorAdd() {
        IVector a = new RereVector(new float[]{1, 2, 3});
        IVector b = new RereVector(new float[]{4, 5, 6});
        
        IVector result = CPUComputeUtils.vectorAdd(a, b);
        
        assertEquals(5, result.get(0), 1e-6);
        assertEquals(7, result.get(1), 1e-6);
        assertEquals(9, result.get(2), 1e-6);
    }
    
    @Test
    public void testVectorDot() {
        IVector a = new RereVector(new float[]{1, 2, 3});
        IVector b = new RereVector(new float[]{4, 5, 6});
        
        float result = CPUComputeUtils.vectorDot(a, b);
        
        assertEquals(32, result, 1e-6); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    }
    
    @Test
    public void testVectorScalarMultiply() {
        IVector a = new RereVector(new float[]{1, 2, 3});
        float scalar = 2.0f;
        
        IVector result = CPUComputeUtils.vectorScalarMultiply(a, scalar);
        
        assertEquals(2, result.get(0), 1e-6);
        assertEquals(4, result.get(1), 1e-6);
        assertEquals(6, result.get(2), 1e-6);
    }
    
    @Test
    public void testVectorSum() {
        IVector a = new RereVector(new float[]{1, 2, 3, 4});
        
        float result = CPUComputeUtils.vectorSum(a);
        
        assertEquals(10, result, 1e-6);
    }
    
    @Test
    public void testVectorReciprocal() {
        IVector a = new RereVector(new float[]{1, 2, 0.001f, 0});
        float tolerance = 0.01f;
        
        IVector result = CPUComputeUtils.vectorReciprocal(a, tolerance);
        
        assertEquals(1.0f, result.get(0), 1e-6);
        assertEquals(0.5f, result.get(1), 1e-6);
        assertEquals(0.0f, result.get(2), 1e-6); // 小于容差，设为0
        assertEquals(0.0f, result.get(3), 1e-6); // 0值，设为0
    }
}
