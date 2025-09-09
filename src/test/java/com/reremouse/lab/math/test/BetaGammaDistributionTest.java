package com.reremouse.lab.math.test;

import com.reremouse.lab.math.stat.distribution.BetaDistribution;
import com.reremouse.lab.math.stat.distribution.GammaDistribution;
import com.reremouse.lab.math.stat.distribution.IContinuousDistribution;

/**
 * Beta分布和Gamma分布综合测试类
 * Comprehensive test class for Beta and Gamma distributions
 * 
 * 测试Beta分布和Gamma分布的正确性，包括：
 * - 基本统计量（均值、方差、标准差、中位数、众数）
 * - 分位数（Q1、Q3）
 * - 偏度和峰度
 * - 概率密度函数（PDF）
 * - 累积分布函数（CDF）
 * - 百分点函数（PPF）
 * - 生存函数（SF）
 * - 逆生存函数（ISF）
 * - 随机采样
 * - 边界情况处理
 * - 参数验证
 * 
 * Tests Beta and Gamma distributions for correctness, including:
 * - Basic statistics (mean, variance, standard deviation, median, mode)
 * - Quantiles (Q1, Q3)
 * - Skewness and kurtosis
 * - Probability density function (PDF)
 * - Cumulative distribution function (CDF)
 * - Percent point function (PPF)
 * - Survival function (SF)
 * - Inverse survival function (ISF)
 * - Random sampling
 * - Edge case handling
 * - Parameter validation
 * 
 * @author lteb2
 */
public class BetaGammaDistributionTest {
    
    private static final float TOLERANCE = 1e-3f; // 测试容差 / Test tolerance
    private static final int SAMPLE_SIZE = 10000; // 采样大小 / Sample size
    private static int testCount = 0;
    private static int passedTests = 0;
    private static int failedTests = 0;
    
    public static void main(String[] args) {
        System.out.println("开始Beta分布和Gamma分布测试 / Starting Beta and Gamma Distribution Tests");
        System.out.println("==================================================");
        
        try {
            testBetaDistribution();
            testGammaDistribution();
            testDistributionProperties();
            testEdgeCases();
            testParameterValidation();
            testRandomSampling();
            
            System.out.println("==================================================");
            System.out.println("测试完成 / Tests completed");
            System.out.println("总测试数 / Total tests: " + testCount);
            System.out.println("通过测试 / Passed: " + passedTests);
            System.out.println("失败测试 / Failed: " + failedTests);
            System.out.println("成功率 / Success rate: " + String.format("%.2f%%", (float)passedTests / testCount * 100));
            
            if (failedTests == 0) {
                System.out.println("所有测试通过！/ All tests passed!");
            } else {
                System.out.println("有测试失败，请检查实现 / Some tests failed, please check implementation");
            }
            
        } catch (Exception e) {
            System.err.println("测试过程中发生错误 / Error during testing: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试Beta分布
     * Test Beta Distribution
     */
    public static void testBetaDistribution() {
        System.out.println("\n测试Beta分布 / Testing Beta Distribution");
        
        // 测试不同的参数组合
        // Test different parameter combinations
        testBetaDistributionCase(1.0f, 1.0f, "均匀分布情况 / Uniform case");
        testBetaDistributionCase(2.0f, 2.0f, "对称情况 / Symmetric case");
        testBetaDistributionCase(2.0f, 5.0f, "右偏情况 / Right-skewed case");
        testBetaDistributionCase(5.0f, 2.0f, "左偏情况 / Left-skewed case");
        testBetaDistributionCase(0.5f, 0.5f, "U型情况 / U-shaped case");
        testBetaDistributionCase(10.0f, 10.0f, "钟型情况 / Bell-shaped case");
        
        System.out.println("Beta分布测试通过 / Beta Distribution test passed");
    }
    
    /**
     * 测试特定参数的Beta分布
     * Test Beta distribution with specific parameters
     */
    private static void testBetaDistributionCase(float alpha, float beta, String caseName) {
        System.out.println("\n  " + caseName);
        
        try {
            BetaDistribution betaDist = new BetaDistribution(alpha, beta);
        
        // 基本统计量测试
        // Basic statistics test
        float expectedMean = alpha / (alpha + beta);
        float expectedVar = (alpha * beta) / ((alpha + beta) * (alpha + beta) * (alpha + beta + 1.0f));
        
        assertEqual(expectedMean, betaDist.mean(), "均值计算错误 / Mean calculation error");
        assertEqual(expectedVar, betaDist.var(), "方差计算错误 / Variance calculation error");
        assertEqual((float)Math.sqrt(expectedVar), betaDist.std(), "标准差计算错误 / Standard deviation calculation error");
        
        // 众数测试
        // Mode test
        if (alpha > 1.0f && beta > 1.0f) {
            float expectedMode = (alpha - 1.0f) / (alpha + beta - 2.0f);
            assertEqual(expectedMode, betaDist.mode(), "众数计算错误 / Mode calculation error");
        }
        
        // PDF测试
        // PDF test
        assertEqual(0.0f, betaDist.pdf(-0.1f), "区间外PDF应为0 / PDF outside [0,1] should be 0");
        assertEqual(0.0f, betaDist.pdf(1.1f), "区间外PDF应为0 / PDF outside [0,1] should be 0");
        
        // 在区间内PDF应该为正
        // PDF should be positive within [0,1]
        assertTrue(betaDist.pdf(0.5f) >= 0, "区间内PDF应非负 / PDF within [0,1] should be non-negative");
        
        // CDF测试
        // CDF test
        assertEqual(0.0f, betaDist.cdf(0.0f), "x=0处CDF应为0 / CDF at x=0 should be 0");
        assertEqual(1.0f, betaDist.cdf(1.0f), "x=1处CDF应为1 / CDF at x=1 should be 1");
        assertTrue(betaDist.cdf(0.5f) > 0 && betaDist.cdf(0.5f) < 1, "x=0.5处CDF应在(0,1)内 / CDF at x=0.5 should be in (0,1)");
        
        // PPF测试
        // PPF test
        assertEqual(0.0f, betaDist.ppf(0.0f), "0%分位数应为0 / 0th percentile should be 0");
        assertEqual(1.0f, betaDist.ppf(1.0f), "100%分位数应为1 / 100th percentile should be 1");
        
        // 测试PPF和CDF的逆关系
        // Test inverse relationship between PPF and CDF
        float testP = 0.3f;
        float ppfValue = betaDist.ppf(testP);
        float cdfValue = betaDist.cdf(ppfValue);
        assertEqual(testP, cdfValue, "PPF和CDF应为逆函数 / PPF and CDF should be inverse functions");
        
        // 四分位数测试
        // Quartiles test
        float q1 = betaDist.q1();
        float q3 = betaDist.q3();
        assertTrue(q1 >= 0 && q1 <= 1, "Q1应在[0,1]内 / Q1 should be in [0,1]");
        assertTrue(q3 >= 0 && q3 <= 1, "Q3应在[0,1]内 / Q3 should be in [0,1]");
        assertTrue(q1 <= q3, "Q1应小于等于Q3 / Q1 should be less than or equal to Q3");
        
        // 偏度和峰度测试
        // Skewness and kurtosis test
        float skewness = betaDist.skewness();
        float kurtosis = betaDist.kurtosis();
        assertFalse(Float.isNaN(skewness), "偏度不应为NaN / Skewness should not be NaN");
        assertFalse(Float.isNaN(kurtosis), "峰度不应为NaN / Kurtosis should not be NaN");
        
        // 随机采样测试
        // Random sampling test
        testRandomSampling(betaDist, "Beta分布(α=" + alpha + ", β=" + beta + ")");
        
        } catch (Exception e) {
            System.err.println("测试Beta分布时发生错误 / Error testing Beta distribution: " + e.getMessage());
            e.printStackTrace();
            failedTests++;
            testCount++;
        }
    }
    
    /**
     * 测试Gamma分布
     * Test Gamma Distribution
     */
    public static void testGammaDistribution() {
        System.out.println("\n测试Gamma分布 / Testing Gamma Distribution");
        
        // 测试不同的参数组合
        // Test different parameter combinations
        testGammaDistributionCase(1.0f, 1.0f, "指数分布情况 / Exponential case");
        testGammaDistributionCase(2.0f, 1.0f, "形状参数=2 / Shape parameter = 2");
        testGammaDistributionCase(0.5f, 1.0f, "形状参数=0.5 / Shape parameter = 0.5");
        testGammaDistributionCase(5.0f, 2.0f, "较大形状参数 / Large shape parameter");
        testGammaDistributionCase(10.0f, 0.5f, "高尺度参数 / High scale parameter");
        
        System.out.println("Gamma分布测试通过 / Gamma Distribution test passed");
    }
    
    /**
     * 测试特定参数的Gamma分布
     * Test Gamma distribution with specific parameters
     */
    private static void testGammaDistributionCase(float alpha, float beta, String caseName) {
        System.out.println("\n  " + caseName);
        
        try {
            GammaDistribution gammaDist = new GammaDistribution(alpha, beta);
        
        // 基本统计量测试
        // Basic statistics test
        float expectedMean = alpha / beta;
        float expectedVar = alpha / (beta * beta);
        
        assertEqual(expectedMean, gammaDist.mean(), "均值计算错误 / Mean calculation error");
        assertEqual(expectedVar, gammaDist.var(), "方差计算错误 / Variance calculation error");
        assertEqual((float)Math.sqrt(expectedVar), gammaDist.std(), "标准差计算错误 / Standard deviation calculation error");
        
        // 众数测试
        // Mode test
        if (alpha >= 1.0f) {
            float expectedMode = (alpha - 1.0f) / beta;
            assertEqual(expectedMode, gammaDist.mode(), "众数计算错误 / Mode calculation error");
        } else {
            assertEqual(0.0f, gammaDist.mode(), "当α<1时众数应为0 / Mode should be 0 when α<1");
        }
        
        // PDF测试
        // PDF test
        assertEqual(0.0f, gammaDist.pdf(-1.0f), "负值处PDF应为0 / PDF at negative values should be 0");
        assertEqual(0.0f, gammaDist.pdf(0.0f), "x=0处PDF应为0 / PDF at x=0 should be 0");
        
        // 在正值处PDF应该为正
        // PDF should be positive at positive values
        assertTrue(gammaDist.pdf(1.0f) >= 0, "正值处PDF应非负 / PDF at positive values should be non-negative");
        
        // CDF测试
        // CDF test
        assertEqual(0.0f, gammaDist.cdf(0.0f), "x=0处CDF应为0 / CDF at x=0 should be 0");
        assertTrue(gammaDist.cdf(1.0f) > 0, "x=1处CDF应大于0 / CDF at x=1 should be greater than 0");
        
        // PPF测试
        // PPF test
        assertEqual(0.0f, gammaDist.ppf(0.0f), "0%分位数应为0 / 0th percentile should be 0");
        assertTrue(Float.isInfinite(gammaDist.ppf(1.0f)) && gammaDist.ppf(1.0f) > 0, 
            "100%分位数应为正无穷 / 100th percentile should be positive infinity");
        
        // 测试PPF和CDF的逆关系
        // Test inverse relationship between PPF and CDF
        float testP = 0.3f;
        float ppfValue = gammaDist.ppf(testP);
        float cdfValue = gammaDist.cdf(ppfValue);
        assertEqual(testP, cdfValue, "PPF和CDF应为逆函数 / PPF and CDF should be inverse functions");
        
        // 四分位数测试
        // Quartiles test
        float q1 = gammaDist.q1();
        float q3 = gammaDist.q3();
        assertTrue(q1 >= 0, "Q1应非负 / Q1 should be non-negative");
        assertTrue(q3 >= 0, "Q3应非负 / Q3 should be non-negative");
        assertTrue(q1 <= q3, "Q1应小于等于Q3 / Q1 should be less than or equal to Q3");
        
        // 偏度和峰度测试
        // Skewness and kurtosis test
        float expectedSkewness = 2.0f / (float)Math.sqrt(alpha);
        float expectedKurtosis = 6.0f / alpha;
        
        assertEqual(expectedSkewness, gammaDist.skewness(), "偏度计算错误 / Skewness calculation error");
        assertEqual(expectedKurtosis, gammaDist.kurtosis(), "峰度计算错误 / Kurtosis calculation error");
        
        // 随机采样测试
        // Random sampling test
        testRandomSampling(gammaDist, "Gamma分布(α=" + alpha + ", β=" + beta + ")");
        
        } catch (Exception e) {
            System.err.println("测试Gamma分布时发生错误 / Error testing Gamma distribution: " + e.getMessage());
            e.printStackTrace();
            failedTests++;
            testCount++;
        }
    }
    
    /**
     * 测试分布属性
     * Test distribution properties
     */
    public static void testDistributionProperties() {
        System.out.println("\n测试分布属性 / Testing Distribution Properties");
        
        // 测试Beta分布属性
        // Test Beta distribution properties
        BetaDistribution betaDist = new BetaDistribution(2.0f, 3.0f);
        testDistributionPropertiesForDistribution(betaDist, "Beta分布 / Beta Distribution");
        
        // 测试Gamma分布属性
        // Test Gamma distribution properties
        GammaDistribution gammaDist = new GammaDistribution(2.0f, 1.0f);
        testDistributionPropertiesForDistribution(gammaDist, "Gamma分布 / Gamma Distribution");
    }
    
    /**
     * 测试特定分布的属性
     * Test properties for specific distribution
     */
    private static void testDistributionPropertiesForDistribution(IContinuousDistribution dist, String distName) {
        // 测试CDF和SF的关系
        // Test relationship between CDF and SF
        float x = 0.5f;
        float cdf = dist.cdf(x);
        float sf = dist.sf(x);
        assertEqual(1.0f, cdf + sf, 
            distName + " CDF + SF 应等于1 / CDF + SF should equal 1");
        
        // 测试PPF和ISF的关系
        // Test relationship between PPF and ISF
        float p = 0.3f;
        float ppf = dist.ppf(p);
        float isf = dist.isf(1.0f - p);
        assertEqual(ppf, isf, 
            distName + " PPF和ISF应相等 / PPF and ISF should be equal");
        
        // 测试PPF和CDF的逆关系
        // Test inverse relationship between PPF and CDF
        float ppfValue = dist.ppf(p);
        float cdfValue = dist.cdf(ppfValue);
        assertEqual(p, cdfValue, 
            distName + " PPF和CDF应为逆函数 / PPF and CDF should be inverse functions");
        
        // 测试中位数
        // Test median
        float median = dist.median();
        float medianCdf = dist.cdf(median);
        assertEqual(0.5f, medianCdf, 
            distName + " 中位数的CDF应为0.5 / CDF at median should be 0.5");
    }
    
    /**
     * 测试边界情况
     * Test edge cases
     */
    public static void testEdgeCases() {
        System.out.println("\n测试边界情况 / Testing Edge Cases");
        
        // 测试Beta分布边界情况
        // Test Beta distribution edge cases
        BetaDistribution betaDist = new BetaDistribution(2.0f, 3.0f);
        
        // 测试极值
        // Test extreme values
        assertEqual(0.0f, betaDist.pdf(Float.NEGATIVE_INFINITY), 
            "负无穷处PDF应为0 / PDF at negative infinity should be 0");
        assertEqual(0.0f, betaDist.pdf(Float.POSITIVE_INFINITY), 
            "正无穷处PDF应为0 / PDF at positive infinity should be 0");
        assertEqual(0.0f, betaDist.cdf(Float.NEGATIVE_INFINITY), 
            "负无穷处CDF应为0 / CDF at negative infinity should be 0");
        assertEqual(1.0f, betaDist.cdf(Float.POSITIVE_INFINITY), 
            "正无穷处CDF应为1 / CDF at positive infinity should be 1");
        
        // 测试NaN处理
        // Test NaN handling
        assertTrue(Float.isNaN(betaDist.pdf(Float.NaN)), "NaN处PDF应为NaN / PDF at NaN should be NaN");
        assertTrue(Float.isNaN(betaDist.cdf(Float.NaN)), "NaN处CDF应为NaN / CDF at NaN should be NaN");
        
        // 测试Gamma分布边界情况
        // Test Gamma distribution edge cases
        GammaDistribution gammaDist = new GammaDistribution(2.0f, 1.0f);
        
        // 测试极值
        // Test extreme values
        assertEqual(0.0f, gammaDist.pdf(Float.NEGATIVE_INFINITY), 
            "负无穷处PDF应为0 / PDF at negative infinity should be 0");
        assertEqual(0.0f, gammaDist.pdf(Float.POSITIVE_INFINITY), 
            "正无穷处PDF应为0 / PDF at positive infinity should be 0");
        assertEqual(0.0f, gammaDist.cdf(Float.NEGATIVE_INFINITY), 
            "负无穷处CDF应为0 / CDF at negative infinity should be 0");
        assertEqual(1.0f, gammaDist.cdf(Float.POSITIVE_INFINITY), 
            "正无穷处CDF应为1 / CDF at positive infinity should be 1");
        
        // 测试NaN处理
        // Test NaN handling
        assertTrue(Float.isNaN(gammaDist.pdf(Float.NaN)), "NaN处PDF应为NaN / PDF at NaN should be NaN");
        assertTrue(Float.isNaN(gammaDist.cdf(Float.NaN)), "NaN处CDF应为NaN / CDF at NaN should be NaN");
        
        System.out.println("边界情况测试通过 / Edge cases test passed");
    }
    
    /**
     * 测试参数验证
     * Test parameter validation
     */
    public static void testParameterValidation() {
        System.out.println("\n测试参数验证 / Testing Parameter Validation");
        
        // 测试Beta分布参数验证
        // Test Beta distribution parameter validation
        assertThrows(() -> {
            new BetaDistribution(-1.0f, 2.0f);
        }, "负α参数应抛出异常 / Negative α parameter should throw exception");
        
        assertThrows(() -> {
            new BetaDistribution(2.0f, -1.0f);
        }, "负β参数应抛出异常 / Negative β parameter should throw exception");
        
        assertThrows(() -> {
            new BetaDistribution(0.0f, 2.0f);
        }, "α=0应抛出异常 / α=0 should throw exception");
        
        assertThrows(() -> {
            new BetaDistribution(2.0f, 0.0f);
        }, "β=0应抛出异常 / β=0 should throw exception");
        
        // 测试Gamma分布参数验证
        // Test Gamma distribution parameter validation
        assertThrows(() -> {
            new GammaDistribution(-1.0f, 2.0f);
        }, "负α参数应抛出异常 / Negative α parameter should throw exception");
        
        assertThrows(() -> {
            new GammaDistribution(2.0f, -1.0f);
        }, "负β参数应抛出异常 / Negative β parameter should throw exception");
        
        assertThrows(() -> {
            new GammaDistribution(0.0f, 2.0f);
        }, "α=0应抛出异常 / α=0 should throw exception");
        
        assertThrows(() -> {
            new GammaDistribution(2.0f, 0.0f);
        }, "β=0应抛出异常 / β=0 should throw exception");
        
        System.out.println("参数验证测试通过 / Parameter validation test passed");
    }
    
    /**
     * 测试随机采样
     * Test random sampling
     */
    public static void testRandomSampling() {
        System.out.println("\n测试随机采样 / Testing Random Sampling");
        
        // 测试Beta分布采样
        // Test Beta distribution sampling
        BetaDistribution betaDist = new BetaDistribution(2.0f, 3.0f);
        testRandomSampling(betaDist, "Beta分布 / Beta Distribution");
        
        // 测试Gamma分布采样
        // Test Gamma distribution sampling
        GammaDistribution gammaDist = new GammaDistribution(2.0f, 1.0f);
        testRandomSampling(gammaDist, "Gamma分布 / Gamma Distribution");
        
        System.out.println("随机采样测试通过 / Random sampling test passed");
    }
    
    /**
     * 测试随机采样的统计性质
     * Test statistical properties of random sampling
     * 
     * @param dist 分布对象 / Distribution object
     * @param distName 分布名称 / Distribution name
     */
    private static void testRandomSampling(IContinuousDistribution dist, String distName) {
        float[] samples = dist.sample(SAMPLE_SIZE);
        
        // 检查样本数量
        // Check sample size
        assertEqual(SAMPLE_SIZE, samples.length, "样本数量不正确 / Sample size is incorrect");
        
        // 检查样本是否在合理范围内
        // Check if samples are in reasonable range
        for (float sample : samples) {
            assertFalse(Float.isNaN(sample), "样本不应为NaN / Sample should not be NaN");
            assertFalse(Float.isInfinite(sample), "样本不应为无穷大 / Sample should not be infinite");
        }
        
        // 计算样本统计量
        // Calculate sample statistics
        float sampleMean = calculateMean(samples);
        float sampleVar = calculateVariance(samples, sampleMean);
        
        // 检查样本均值是否接近理论均值
        // Check if sample mean is close to theoretical mean
        if (!Float.isNaN(dist.mean())) {
            float meanError = Math.abs(sampleMean - dist.mean());
            float meanThreshold = Math.max(0.1f, Math.abs(dist.mean()) * 0.2f); // 动态调整容差
            assertTrue(meanError < meanThreshold, 
                distName + " 样本均值与理论均值差异过大 / Sample mean differs too much from theoretical mean");
        }
        
        // 检查样本方差是否接近理论方差
        // Check if sample variance is close to theoretical variance
        if (!Float.isNaN(dist.var())) {
            float varError = Math.abs(sampleVar - dist.var());
            float varThreshold = Math.max(0.5f, Math.abs(dist.var()) * 0.3f); // 动态调整容差
            assertTrue(varError < varThreshold, 
                distName + " 样本方差与理论方差差异过大 / Sample variance differs too much from theoretical variance");
        }
    }
    
    /**
     * 计算数组的均值
     * Calculate mean of array
     * 
     * @param values 数值数组 / Array of values
     * @return 均值 / Mean
     */
    private static float calculateMean(float[] values) {
        float sum = 0.0f;
        for (float value : values) {
            sum += value;
        }
        return sum / values.length;
    }
    
    /**
     * 计算数组的方差
     * Calculate variance of array
     * 
     * @param values 数值数组 / Array of values
     * @param mean 均值 / Mean
     * @return 方差 / Variance
     */
    private static float calculateVariance(float[] values, float mean) {
        float sumSquaredDiff = 0.0f;
        for (float value : values) {
            float diff = value - mean;
            sumSquaredDiff += diff * diff;
        }
        return sumSquaredDiff / values.length;
    }
    
    // 测试辅助方法 / Test helper methods
    private static void assertEqual(float expected, float actual, String message) {
        testCount++;
        if (Math.abs(expected - actual) <= TOLERANCE) {
            passedTests++;
        } else {
            failedTests++;
            System.err.println("测试失败 / Test failed: " + message);
            System.err.println("期望值 / Expected: " + expected + ", 实际值 / Actual: " + actual);
            System.err.println("差异 / Difference: " + Math.abs(expected - actual));
        }
    }
    
    private static void assertEqual(int expected, int actual, String message) {
        testCount++;
        if (expected == actual) {
            passedTests++;
        } else {
            failedTests++;
            System.err.println("测试失败 / Test failed: " + message);
            System.err.println("期望值 / Expected: " + expected + ", 实际值 / Actual: " + actual);
        }
    }
    
    private static void assertTrue(boolean condition, String message) {
        testCount++;
        if (condition) {
            passedTests++;
        } else {
            failedTests++;
            System.err.println("测试失败 / Test failed: " + message);
        }
    }
    
    private static void assertFalse(boolean condition, String message) {
        testCount++;
        if (!condition) {
            passedTests++;
        } else {
            failedTests++;
            System.err.println("测试失败 / Test failed: " + message);
        }
    }
    
    private static void assertThrows(Runnable runnable, String message) {
        testCount++;
        try {
            runnable.run();
            failedTests++;
            System.err.println("测试失败 / Test failed: " + message + " - 应该抛出异常但没有 / Should have thrown exception but didn't");
        } catch (Exception e) {
            passedTests++;
        }
    }
}
