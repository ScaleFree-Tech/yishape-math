package com.yishape.lab.math.stats.distributions;

import com.yishape.lab.math.stats.distribution.multiv.WishartDistribution;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 全面测试Wishart分布的功能
 */
public class WishartDistributionTest {
    
    private WishartDistribution wishart;
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
        wishart = new WishartDistribution(degreesOfFreedom, scaleMatrix);
    }
    
    @Test
    void testVectorApiMatchesMatrixOrder() {
        assertEquals(scaleMatrix.rows() * scaleMatrix.cols(), wishart.getDimension());
        assertEquals(scaleMatrix.rows(), wishart.getMatrixOrder());
        assertEquals(wishart.getDimension(), wishart.sample().size());
    }

    @Test
    void testConstructorValidation() {
        // 测试有效构造函数
        assertDoesNotThrow(() -> new WishartDistribution(3.0, scaleMatrix));
        
        // 测试无效的自由度
        assertThrows(IllegalArgumentException.class, 
            () -> new WishartDistribution(1.0, scaleMatrix));
        
        // 测试null尺度矩阵
        assertThrows(IllegalArgumentException.class, 
            () -> new WishartDistribution(3.0, null));
    }
    
    @Test
    void testSample() {
        // 测试采样功能
        IMatrix<Double> sample = wishart.sampleMatrix();
        assertNotNull(sample);
        assertEquals(scaleMatrix.rows(), sample.rows());
        assertEquals(scaleMatrix.cols(), sample.cols());
    }
    
    @Test
    void testSampleWithSeed() {
        // 测试带种子的采样
        WishartDistribution wishart1 = new WishartDistribution(degreesOfFreedom, scaleMatrix);
        WishartDistribution wishart2 = new WishartDistribution(degreesOfFreedom, scaleMatrix);
        
        IMatrix<Double> sample1 = wishart1.sampleMatrix();
        IMatrix<Double> sample2 = wishart2.sampleMatrix();
        
        // 由于随机性，样本不会完全相同
        assertFalse(matricesEqual(sample1, sample2, tolerance));
    }
    
    @Test
    void testPdf() {
        // 创建一个测试矩阵
        IMatrix<Double> testMatrix = Linalg.matrix(new double[][]{
            {3.0, 1.0},
            {1.0, 2.0}
        });
        
        double pdf = wishart.pdfMatrix(testMatrix);
        assertTrue(pdf > 0);
        assertTrue(Double.isFinite(pdf));
    }
    
    @Test
    void testLogPdf() {
        IMatrix<Double> testMatrix = Linalg.matrix(new double[][]{
            {3.0, 1.0},
            {1.0, 2.0}
        });
        
        double logPdf = wishart.logPdf(testMatrix);
        double pdf = wishart.pdfMatrix(testMatrix);
        
        assertEquals(Math.log(pdf), logPdf, tolerance);
        assertTrue(Double.isFinite(logPdf));
    }
    
    @Test
    void testMean() {
        IMatrix<Double> mean = wishart.meanMatrix();
        IMatrix<Double> expectedMean = scaleMatrix.multiplyByScalar(degreesOfFreedom);
        
        assertTrue(matricesEqual(mean, expectedMean, tolerance));
    }
    
    @Test
    void testVariance() {
        IMatrix<Double> variance = wishart.varianceMatrix();
        assertNotNull(variance);
        assertEquals(scaleMatrix.rows(), variance.rows());
        assertEquals(scaleMatrix.cols(), variance.cols());
        
        // 验证方差矩阵的对称性
        for (int i = 0; i < variance.rows(); i++) {
            for (int j = 0; j < variance.cols(); j++) {
                assertEquals(variance.get(i, j), variance.get(j, i), tolerance);
            }
        }
    }
    
    @Test
    void testSpecialCases() {
        // 测试1维情况
        IMatrix<Double> scale1D = Linalg.matrix(new double[][]{{2.0}});
        WishartDistribution wishart1D = new WishartDistribution(3.0, scale1D);
        
        IMatrix<Double> sample1D = wishart1D.sampleMatrix();
        assertEquals(1, sample1D.rows());
        assertEquals(1, sample1D.cols());
        assertTrue(sample1D.get(0, 0) > 0);
        
        // 测试大维度情况
        IMatrix<Double> scale5D = Linalg.eye(5);
        WishartDistribution wishart5D = new WishartDistribution(10.0, scale5D);
        
        IMatrix<Double> sample5D = wishart5D.sampleMatrix();
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
        
        WishartDistribution wishartIll = new WishartDistribution(5.0, illConditioned);
        
        // 应该能够处理而不抛出异常
        assertDoesNotThrow(() -> {
            IMatrix<Double> sample = wishartIll.sampleMatrix();
            double pdf = wishartIll.pdfMatrix(sample);
            double logPdf = wishartIll.logPdf(sample);
            IMatrix<Double> mean = wishartIll.meanMatrix();
            IMatrix<Double> variance = wishartIll.varianceMatrix();
        });
    }
    
    // 辅助方法
    private boolean matricesEqual(IMatrix<Double> a, IMatrix<Double> b, double tolerance) {
        if (a.rows() != b.rows() || a.cols() != b.cols()) {
            return false;
        }
        
        for (int i = 0; i < a.rows(); i++) {
            for (int j = 0; j < a.cols(); j++) {
                if (Math.abs(a.get(i, j) - b.get(i, j)) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }
}