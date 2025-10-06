package com.yishape.lab.math.stats;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IFloatVector;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Stats类corr和cov方法的测试类
 * Test class for Stats corr and cov methods
 */
public class StatsCorrCovTest {

    @Test
    public void testCorrBasic() {
        // 测试基本相关系数计算 / Test basic correlation calculation
        float[] data1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float[] data2 = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f};
        
        IVector<Float> v1 = IFloatVector.of(data1);
        IVector<Float> v2 = IFloatVector.of(data2);
        
        double correlation = Stats.corr(v1, v2);
        
        // 完全正相关应该接近1 / Perfect positive correlation should be close to 1
        assertEquals(1.0, correlation, 1e-6, "完全正相关的相关系数应该为1 / Perfect positive correlation should be 1");
    }

    @Test
    public void testCorrNegative() {
        // 测试负相关 / Test negative correlation
        float[] data1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float[] data2 = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        
        IVector<Float> v1 = IFloatVector.of(data1);
        IVector<Float> v2 = IFloatVector.of(data2);
        
        double correlation = Stats.corr(v1, v2);
        
        // 完全负相关应该接近-1 / Perfect negative correlation should be close to -1
        assertEquals(-1.0, correlation, 1e-6, "完全负相关的相关系数应该为-1 / Perfect negative correlation should be -1");
    }

    @Test
    public void testCorrZero() {
        // 测试无相关 / Test no correlation
        float[] data1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float[] data2 = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        
        IVector<Float> v1 = IFloatVector.of(data1);
        IVector<Float> v2 = IFloatVector.of(data2);
        
        double correlation = Stats.corr(v1, v2);
        
        // 常数向量与任何向量的相关系数应该为0 / Correlation with constant vector should be 0
        assertEquals(0.0, correlation, 1e-6, "与常数向量的相关系数应该为0 / Correlation with constant vector should be 0");
    }

    @Test
    public void testCovBasic() {
        // 测试基本协方差计算 / Test basic covariance calculation
        float[] data1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float[] data2 = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f};
        
        IVector<Float> v1 = IFloatVector.of(data1);
        IVector<Float> v2 = IFloatVector.of(data2);
        
        double covariance = Stats.cov(v1, v2);
        
        // 协方差应该为正数（正相关）/ Covariance should be positive (positive correlation)
        assertTrue(covariance > 0, "正相关的协方差应该为正数 / Covariance for positive correlation should be positive");
    }

    @Test
    public void testCovNegative() {
        // 测试负协方差 / Test negative covariance
        float[] data1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float[] data2 = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        
        IVector<Float> v1 = IFloatVector.of(data1);
        IVector<Float> v2 = IFloatVector.of(data2);
        
        double covariance = Stats.cov(v1, v2);
        
        // 协方差应该为负数（负相关）/ Covariance should be negative (negative correlation)
        assertTrue(covariance < 0, "负相关的协方差应该为负数 / Covariance for negative correlation should be negative");
    }

    @Test
    public void testCovZero() {
        // 测试零协方差 / Test zero covariance
        float[] data1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float[] data2 = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        
        IVector<Float> v1 = IFloatVector.of(data1);
        IVector<Float> v2 = IFloatVector.of(data2);
        
        double covariance = Stats.cov(v1, v2);
        
        // 与常数向量的协方差应该为0 / Covariance with constant vector should be 0
        assertEquals(0.0, covariance, 1e-6, "与常数向量的协方差应该为0 / Covariance with constant vector should be 0");
    }

    @Test
    public void testCorrNullInput() {
        // 测试null输入 / Test null input
        float[] data = {1.0f, 2.0f, 3.0f};
        IVector<Float> v = IFloatVector.of(data);
        
        assertThrows(IllegalArgumentException.class, () -> {
            Stats.corr(null, v);
        }, "第一个向量为null应该抛出异常 / First vector being null should throw exception");
        
        assertThrows(IllegalArgumentException.class, () -> {
            Stats.corr(v, null);
        }, "第二个向量为null应该抛出异常 / Second vector being null should throw exception");
    }

    @Test
    public void testCovNullInput() {
        // 测试null输入 / Test null input
        float[] data = {1.0f, 2.0f, 3.0f};
        IVector<Float> v = IFloatVector.of(data);
        
        assertThrows(IllegalArgumentException.class, () -> {
            Stats.cov(null, v);
        }, "第一个向量为null应该抛出异常 / First vector being null should throw exception");
        
        assertThrows(IllegalArgumentException.class, () -> {
            Stats.cov(v, null);
        }, "第二个向量为null应该抛出异常 / Second vector being null should throw exception");
    }

    @Test
    public void testCorrLengthMismatch() {
        // 测试长度不匹配 / Test length mismatch
        float[] data1 = {1.0f, 2.0f, 3.0f};
        float[] data2 = {1.0f, 2.0f};
        
        IVector<Float> v1 = IFloatVector.of(data1);
        IVector<Float> v2 = IFloatVector.of(data2);
        
        assertThrows(IllegalArgumentException.class, () -> {
            Stats.corr(v1, v2);
        }, "向量长度不匹配应该抛出异常 / Vector length mismatch should throw exception");
    }

    @Test
    public void testCovLengthMismatch() {
        // 测试长度不匹配 / Test length mismatch
        float[] data1 = {1.0f, 2.0f, 3.0f};
        float[] data2 = {1.0f, 2.0f};
        
        IVector<Float> v1 = IFloatVector.of(data1);
        IVector<Float> v2 = IFloatVector.of(data2);
        
        assertThrows(IllegalArgumentException.class, () -> {
            Stats.cov(v1, v2);
        }, "向量长度不匹配应该抛出异常 / Vector length mismatch should throw exception");
    }

    @Test
    public void testCorrWithZeroStd() {
        // 测试标准差为0的情况 / Test case with zero standard deviation
        float[] data1 = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        float[] data2 = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f};
        
        IVector<Float> v1 = IFloatVector.of(data1);
        IVector<Float> v2 = IFloatVector.of(data2);
        
        assertThrows(ArithmeticException.class, () -> {
            Stats.corr(v1, v2);
        }, "标准差为0时应该抛出异常 / Should throw exception when standard deviation is 0");
    }

    @Test
    public void testCovWithEmptyVector() {
        // 测试空向量 / Test empty vector
        float[] data1 = {};
        float[] data2 = {1.0f, 2.0f, 3.0f};
        
        IVector<Float> v1 = IFloatVector.of(data1);
        IVector<Float> v2 = IFloatVector.of(data2);
        
        assertThrows(ArithmeticException.class, () -> {
            Stats.cov(v1, v2);
        }, "空向量应该抛出异常 / Empty vector should throw exception");
    }
}
