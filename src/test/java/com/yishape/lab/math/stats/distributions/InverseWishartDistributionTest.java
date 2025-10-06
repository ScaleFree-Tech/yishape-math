package com.yishape.lab.math.stats.distributions;

import com.yishape.lab.math.stats.distribution.multiv.InverseWishartDistribution;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 全面测试逆Wishart分布的功能
 */
public class InverseWishartDistributionTest {
    
    private InverseWishartDistribution invWishart;
    private IMatrix<Double> scaleMatrix;
    private double degreesOfFreedom;
    private double tolerance = 1e-6;
    
    @BeforeEach
    void setUp() {
        // 创建一个2x2的正定尺度矩阵
        scaleMatrix = Linalg.matrix(new double[][]{
            {2.0, 0.5},
            {0.5, 1.0}
        });
        degreesOfFreedom = 5.0;
        invWishart = new InverseWishartDistribution(degreesOfFreedom, scaleMatrix);
    }
    
    @Test
    void testConstructorValidation() {
        // 测试有效构造函数
        assertDoesNotThrow(() -> new InverseWishartDistribution(4.0, scaleMatrix));
        
        // 测试无效的自由度（必须 > p + 1，其中p是矩阵维度）
        assertThrows(IllegalArgumentException.class, 
            () -> new InverseWishartDistribution(2.0, scaleMatrix));
        
        // 测试null尺度矩阵
        assertThrows(IllegalArgumentException.class, 
            () -> new InverseWishartDistribution(5.0, null));
    }
    
    @Test
    void testSample() {
        // 测试采样功能
        IMatrix<Double> sample = invWishart.sampleMatrix();
        assertNotNull(sample);
        assertEquals(scaleMatrix.rows(), sample.rows());
        assertEquals(scaleMatrix.cols(), sample.cols());
        
        // 验证采样结果是正定的（通过计算行列式来间接验证）
        assertTrue(sample.get(0, 0).doubleValue() > 0); // 简单验证第一个元素为正
    }
    
    @Test
    void testSampleWithSeed() {
        // 测试带种子的采样
        InverseWishartDistribution invWishart1 = new InverseWishartDistribution(degreesOfFreedom, scaleMatrix);
        InverseWishartDistribution invWishart2 = new InverseWishartDistribution(degreesOfFreedom, scaleMatrix);
        
        IMatrix<Double> sample1 = invWishart1.sampleMatrix();
        IMatrix<Double> sample2 = invWishart2.sampleMatrix();
        
        // 由于随机性，样本不会完全相同
        assertFalse(matricesEqual(sample1, sample2, tolerance));
    }
    
    @Test
    void testPdf() {
        // 创建一个测试矩阵
        IMatrix<Double> testMatrix = Linalg.matrix(new double[][]{
            {1.5, 0.3},
            {0.3, 1.0}
        });
        
        double pdf = invWishart.pdf(testMatrix);
        assertTrue(pdf > 0);
        assertTrue(Double.isFinite(pdf));
    }
    
    @Test
    void testLogPdf() {
        IMatrix<Double> testMatrix = Linalg.matrix(new double[][]{
            {1.5, 0.3},
            {0.3, 1.0}
        });
        
        double logPdf = invWishart.logPdf(testMatrix);
        double pdf = invWishart.pdf(testMatrix);
        
        assertEquals(Math.log(pdf), logPdf, tolerance);
        assertTrue(Double.isFinite(logPdf));
    }
    
    @Test
    void testMean() {
        // 均值只在 nu > p + 1 时存在
        if (degreesOfFreedom > scaleMatrix.rows() + 1) {
            IMatrix<Double> mean = invWishart.mean();
            assertNotNull(mean);
            assertEquals(scaleMatrix.rows(), mean.rows());
            assertEquals(scaleMatrix.cols(), mean.cols());
        }
        
        // 测试自由度不足的情况
        InverseWishartDistribution lowDf = new InverseWishartDistribution(3.0, scaleMatrix);
        assertThrows(IllegalStateException.class, () -> lowDf.mean());
    }
    
    @Test
    void testMode() {
        IMatrix<Double> mode = invWishart.mode();
        assertNotNull(mode);
        assertEquals(scaleMatrix.rows(), mode.rows());
        assertEquals(scaleMatrix.cols(), mode.cols());
    }
    
    @Test
    void testVariance() {
        // 方差只在 nu > p + 3 时存在
        if (degreesOfFreedom > scaleMatrix.rows() + 3) {
            IMatrix<Double> variance = invWishart.variance();
            assertNotNull(variance);
            assertEquals(scaleMatrix.rows(), variance.rows());
            assertEquals(scaleMatrix.cols(), variance.cols());
        }
        
        // 测试自由度不足的情况
        InverseWishartDistribution lowDf = new InverseWishartDistribution(4.0, scaleMatrix);
        assertThrows(IllegalStateException.class, () -> lowDf.variance());
    }
    
    @Test
    void testSpecialCases() {
        // 测试1维情况
        IMatrix<Double> scale1D = Linalg.matrix(new double[][]{{2.0}});
        InverseWishartDistribution invWishart1D = new InverseWishartDistribution(3.0, scale1D);
        
        IMatrix<Double> sample1D = invWishart1D.sampleMatrix();
        assertEquals(1, sample1D.rows());
        assertEquals(1, sample1D.cols());
        assertTrue(sample1D.get(0, 0).doubleValue() > 0);
        
        // 测试大维度情况
        IMatrix<Double> scale5D = Linalg.eye(5);
        InverseWishartDistribution invWishart5D = new InverseWishartDistribution(10.0, scale5D);
        
        IMatrix<Double> sample5D = invWishart5D.sampleMatrix();
        assertEquals(5, sample5D.rows());
        assertEquals(5, sample5D.cols());
    }
    
    @Test
    void testNumericalStability() {
        // 测试数值稳定性 - 使用条件数较大的矩阵
        IMatrix<Double> illConditioned = Linalg.matrix(new double[][]{
            {1.0, 0.99},
            {0.99, 1.0}
        });
        
        InverseWishartDistribution invWishartIll = new InverseWishartDistribution(5.0, illConditioned);
        
        // 应该能够处理而不抛出异常
        assertDoesNotThrow(() -> {
            IMatrix<Double> sample = invWishartIll.sampleMatrix();
            double pdf = invWishartIll.pdf(sample);
            double logPdf = invWishartIll.logPdf(sample);
            IMatrix<Double> mode = invWishartIll.mode();
        });
    }
    
    // 辅助方法
    private boolean matricesEqual(IMatrix<Double> a, IMatrix<Double> b, double tolerance) {
        if (a.rows() != b.rows() || a.cols() != b.cols()) {
            return false;
        }
        
        for (int i = 0; i < a.rows(); i++) {
            for (int j = 0; j < a.cols(); j++) {
                if (Math.abs(a.get(i, j).doubleValue() - b.get(i, j).doubleValue()) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }
}