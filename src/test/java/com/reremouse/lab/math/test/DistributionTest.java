package com.reremouse.lab.math.test;

import com.reremouse.lab.math.stat.distribution.*;

/**
 * 概率分布类综合测试
 * Comprehensive test for probability distribution classes
 * 
 * 测试所有分布类的正确性，包括：
 * - 基本统计量（均值、方差、标准差、中位数、众数）
 * - 分位数（Q1、Q3）
 * - 偏度和峰度
 * - 概率密度函数（PDF）
 * - 累积分布函数（CDF）
 * - 百分点函数（PPF）
 * - 生存函数（SF）
 * - 逆生存函数（ISF）
 * - 随机采样
 * 
 * Tests all distribution classes for correctness, including:
 * - Basic statistics (mean, variance, standard deviation, median, mode)
 * - Quantiles (Q1, Q3)
 * - Skewness and kurtosis
 * - Probability density function (PDF)
 * - Cumulative distribution function (CDF)
 * - Percent point function (PPF)
 * - Survival function (SF)
 * - Inverse survival function (ISF)
 * - Random sampling
 * 
 * @author lteb2
 */
public class DistributionTest {
    
    private static final float TOLERANCE = 1e-3f; // 测试容差 / Test tolerance
    private static final int SAMPLE_SIZE = 10000; // 采样大小 / Sample size
    private static int testCount = 0;
    private static int passedTests = 0;
    private static int failedTests = 0;
    
    public static void main(String[] args) {
        System.out.println("开始概率分布类测试 / Starting Probability Distribution Tests");
        System.out.println("==================================================");
        
        try {
            testNormalDistribution();
            testStudentDistribution();
            testUniformDistribution();
            testExponentialDistribution();
            testChi2Distribution();
            testFDistribution();
            testDistributionProperties();
            testEdgeCases();
            testParameterValidation();
            
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
    
    public static void testNormalDistribution() {
        System.out.println("\n测试正态分布 / Testing Normal Distribution");
        
        // 标准正态分布测试
        // Standard normal distribution test
        NormalDistribution normal = new NormalDistribution(0.0f, 1.0f);
        
        // 基本统计量测试
        // Basic statistics test
        assertEqual(0.0f, normal.mean(), "均值应为0 / Mean should be 0");
        assertEqual(1.0f, normal.var(), "方差应为1 / Variance should be 1");
        assertEqual(1.0f, normal.std(), "标准差应为1 / Standard deviation should be 1");
        assertEqual(0.0f, normal.median(), "中位数应为0 / Median should be 0");
        assertEqual(0.0f, normal.mode(), "众数应为0 / Mode should be 0");
        assertEqual(0.0f, normal.skewness(), "偏度应为0 / Skewness should be 0");
        assertEqual(0.0f, normal.kurtosis(), "峰度应为0 / Kurtosis should be 0");
        
        // PDF测试
        // PDF test
        float pdf0 = normal.pdf(0.0f);
        float expectedPdf0 = 0.3989f; // 1/√(2π) ≈ 0.3989
        assertEqual(expectedPdf0, pdf0, "x=0处的PDF值不正确 / PDF at x=0 is incorrect");
        
        // CDF测试
        // CDF test
        assertEqual(0.5f, normal.cdf(0.0f), "x=0处的CDF值应为0.5 / CDF at x=0 should be 0.5");
        assertTrue(normal.cdf(1.0f) > 0.8f, "x=1处的CDF值应大于0.8 / CDF at x=1 should be greater than 0.8");
        
        // PPF测试
        // PPF test
        assertEqual(0.0f, normal.ppf(0.5f), "50%分位数应为0 / 50th percentile should be 0");
        assertTrue(normal.ppf(0.95f) > 1.6f, "95%分位数应大于1.6 / 95th percentile should be greater than 1.6");
        
        // 四分位数测试
        // Quartiles test
        float q1 = normal.q1();
        float q3 = normal.q3();
        assertTrue(q1 < 0, "Q1应小于0 / Q1 should be less than 0");
        assertTrue(q3 > 0, "Q3应大于0 / Q3 should be greater than 0");
        assertEqual(0.0f, (q1 + q3) / 2.0f, "Q1和Q3的平均值应为0 / Average of Q1 and Q3 should be 0");
        
        // 随机采样测试
        // Random sampling test
        testRandomSampling(normal, "正态分布 / Normal Distribution");
        
        System.out.println("正态分布测试通过 / Normal Distribution test passed");
    }
    
    public static void testStudentDistribution() {
        System.out.println("\n测试t分布 / Testing Student's t-Distribution");
        
        StudentDistribution tDist = new StudentDistribution(10.0f);
        
        // 基本统计量测试
        // Basic statistics test
        assertEqual(0.0f, tDist.mean(), "均值应为0 / Mean should be 0");
        assertTrue(tDist.var() > 1.0f, "方差应大于1 / Variance should be greater than 1");
        assertEqual(0.0f, tDist.median(), "中位数应为0 / Median should be 0");
        assertEqual(0.0f, tDist.mode(), "众数应为0 / Mode should be 0");
        assertEqual(0.0f, tDist.skewness(), "偏度应为0 / Skewness should be 0");
        assertTrue(tDist.kurtosis() > 0, "峰度应大于0 / Kurtosis should be greater than 0");
        
        // PDF测试
        // PDF test
        float pdf0 = tDist.pdf(0.0f);
        assertTrue(pdf0 > 0, "x=0处的PDF值应大于0 / PDF at x=0 should be greater than 0");
        
        // CDF测试
        // CDF test
        assertEqual(0.5f, tDist.cdf(0.0f), "x=0处的CDF值应为0.5 / CDF at x=0 should be 0.5");
        
        // 随机采样测试
        // Random sampling test
        testRandomSampling(tDist, "t分布 / Student's t-Distribution");
        
        System.out.println("t分布测试通过 / Student's t-Distribution test passed");
    }
    
    public static void testUniformDistribution() {
        System.out.println("\n测试均匀分布 / Testing Uniform Distribution");
        
        UniformDistribution uniform = new UniformDistribution(0.0f, 10.0f);
        
        // 基本统计量测试
        // Basic statistics test
        assertEqual(5.0f, uniform.mean(), "均值应为5 / Mean should be 5");
        assertEqual(8.333f, uniform.var(), "方差应约为8.333 / Variance should be approximately 8.333");
        assertEqual(5.0f, uniform.median(), "中位数应为5 / Median should be 5");
        assertTrue(Float.isNaN(uniform.mode()), "众数应为NaN / Mode should be NaN");
        assertEqual(0.0f, uniform.skewness(), "偏度应为0 / Skewness should be 0");
        assertEqual(-1.2f, uniform.kurtosis(), "峰度应为-1.2 / Kurtosis should be -1.2");
        
        // PDF测试
        // PDF test
        assertEqual(0.1f, uniform.pdf(5.0f), "区间内的PDF值应为0.1 / PDF within interval should be 0.1");
        assertEqual(0.0f, uniform.pdf(-1.0f), "区间外的PDF值应为0 / PDF outside interval should be 0");
        assertEqual(0.0f, uniform.pdf(11.0f), "区间外的PDF值应为0 / PDF outside interval should be 0");
        
        // CDF测试
        // CDF test
        assertEqual(0.0f, uniform.cdf(0.0f), "x=0处的CDF值应为0 / CDF at x=0 should be 0");
        assertEqual(0.5f, uniform.cdf(5.0f), "x=5处的CDF值应为0.5 / CDF at x=5 should be 0.5");
        assertEqual(1.0f, uniform.cdf(10.0f), "x=10处的CDF值应为1 / CDF at x=10 should be 1");
        
        // PPF测试
        // PPF test
        assertEqual(0.0f, uniform.ppf(0.0f), "0%分位数应为0 / 0th percentile should be 0");
        assertEqual(5.0f, uniform.ppf(0.5f), "50%分位数应为5 / 50th percentile should be 5");
        assertEqual(10.0f, uniform.ppf(1.0f), "100%分位数应为10 / 100th percentile should be 10");
        
        // 随机采样测试
        // Random sampling test
        testRandomSampling(uniform, "均匀分布 / Uniform Distribution");
        
        System.out.println("均匀分布测试通过 / Uniform Distribution test passed");
    }
    
    public static void testExponentialDistribution() {
        System.out.println("\n测试指数分布 / Testing Exponential Distribution");
        
        ExponentialDistribution exp = new ExponentialDistribution(2.0f);
        
        // 基本统计量测试
        // Basic statistics test
        assertEqual(0.5f, exp.mean(), "均值应为0.5 / Mean should be 0.5");
        assertEqual(0.25f, exp.var(), "方差应为0.25 / Variance should be 0.25");
        assertEqual(0.5f, exp.std(), "标准差应为0.5 / Standard deviation should be 0.5");
        assertEqual(0.0f, exp.mode(), "众数应为0 / Mode should be 0");
        assertEqual(2.0f, exp.skewness(), "偏度应为2 / Skewness should be 2");
        assertEqual(6.0f, exp.kurtosis(), "峰度应为6 / Kurtosis should be 6");
        
        // PDF测试
        // PDF test
        assertEqual(2.0f, exp.pdf(0.0f), "x=0处的PDF值应为2 / PDF at x=0 should be 2");
        assertTrue(exp.pdf(1.0f) > 0, "x=1处的PDF值应大于0 / PDF at x=1 should be greater than 0");
        assertEqual(0.0f, exp.pdf(-1.0f), "x=-1处的PDF值应为0 / PDF at x=-1 should be 0");
        
        // CDF测试
        // CDF test
        assertEqual(0.0f, exp.cdf(0.0f), "x=0处的CDF值应为0 / CDF at x=0 should be 0");
        assertTrue(exp.cdf(1.0f) > 0.8f, "x=1处的CDF值应大于0.8 / CDF at x=1 should be greater than 0.8");
        
        // 随机采样测试
        // Random sampling test
        testRandomSampling(exp, "指数分布 / Exponential Distribution");
        
        System.out.println("指数分布测试通过 / Exponential Distribution test passed");
    }
    
    public static void testChi2Distribution() {
        System.out.println("\n测试卡方分布 / Testing Chi-Squared Distribution");
        
        Chi2Distribution chi2 = new Chi2Distribution(5.0f);
        
        // 基本统计量测试
        // Basic statistics test
        assertEqual(5.0f, chi2.mean(), "均值应为5 / Mean should be 5");
        assertEqual(10.0f, chi2.var(), "方差应为10 / Variance should be 10");
        assertTrue(chi2.skewness() > 0, "偏度应大于0 / Skewness should be greater than 0");
        assertTrue(chi2.kurtosis() > 0, "峰度应大于0 / Kurtosis should be greater than 0");
        
        // PDF测试
        // PDF test
        assertEqual(0.0f, chi2.pdf(0.0f), "x=0处的PDF值应为0 / PDF at x=0 should be 0");
        assertTrue(chi2.pdf(1.0f) > 0, "x=1处的PDF值应大于0 / PDF at x=1 should be greater than 0");
        assertEqual(0.0f, chi2.pdf(-1.0f), "x=-1处的PDF值应为0 / PDF at x=-1 should be 0");
        
        // CDF测试
        // CDF test
        assertEqual(0.0f, chi2.cdf(0.0f), "x=0处的CDF值应为0 / CDF at x=0 should be 0");
        assertTrue(chi2.cdf(10.0f) > 0.5f, "x=10处的CDF值应大于0.5 / CDF at x=10 should be greater than 0.5");
        
        // 随机采样测试
        // Random sampling test
        testRandomSampling(chi2, "卡方分布 / Chi-Squared Distribution");
        
        System.out.println("卡方分布测试通过 / Chi-Squared Distribution test passed");
    }
    
    public static void testFDistribution() {
        System.out.println("\n测试F分布 / Testing F-Distribution");
        
        FDistribution fDist = new FDistribution(5.0f, 10.0f);
        
        // 基本统计量测试
        // Basic statistics test
        assertTrue(fDist.mean() > 1.0f, "均值应大于1 / Mean should be greater than 1");
        assertTrue(fDist.var() > 0, "方差应大于0 / Variance should be greater than 0");
        assertTrue(fDist.skewness() > 0, "偏度应大于0 / Skewness should be greater than 0");
        
        // PDF测试
        // PDF test
        assertEqual(0.0f, fDist.pdf(0.0f), "x=0处的PDF值应为0 / PDF at x=0 should be 0");
        assertTrue(fDist.pdf(1.0f) > 0, "x=1处的PDF值应大于0 / PDF at x=1 should be greater than 0");
        assertEqual(0.0f, fDist.pdf(-1.0f), "x=-1处的PDF值应为0 / PDF at x=-1 should be 0");
        
        // CDF测试
        // CDF test
        assertEqual(0.0f, fDist.cdf(0.0f), "x=0处的CDF值应为0 / CDF at x=0 should be 0");
        assertTrue(fDist.cdf(1.0f) > 0.3f, "x=1处的CDF值应大于0.3 / CDF at x=1 should be greater than 0.3");
        
        // 随机采样测试
        // Random sampling test
        testRandomSampling(fDist, "F分布 / F-Distribution");
        
        System.out.println("F分布测试通过 / F-Distribution test passed");
    }
    
    public static void testDistributionProperties() {
        System.out.println("\n测试分布属性 / Testing Distribution Properties");
        
        // 测试所有分布的基本属性
        // Test basic properties of all distributions
        IContinuousDistribution[] distributions = {
            new NormalDistribution(0.0f, 1.0f),
            new StudentDistribution(10.0f),
            new UniformDistribution(0.0f, 1.0f),
            new ExponentialDistribution(1.0f),
            new Chi2Distribution(3.0f),
            new FDistribution(3.0f, 5.0f)
        };
        
        for (IContinuousDistribution dist : distributions) {
            // 测试CDF和SF的关系
            // Test relationship between CDF and SF
            float x = 0.5f;
            float cdf = dist.cdf(x);
            float sf = dist.sf(x);
            assertEqual(1.0f, cdf + sf, 
                dist.getClass().getSimpleName() + " CDF + SF 应等于1 / CDF + SF should equal 1");
            
            // 测试PPF和ISF的关系
            // Test relationship between PPF and ISF
            float p = 0.3f;
            float ppf = dist.ppf(p);
            float isf = dist.isf(1.0f - p);
            assertEqual(ppf, isf, 
                dist.getClass().getSimpleName() + " PPF和ISF应相等 / PPF and ISF should be equal");
            
            // 测试PPF和CDF的逆关系
            // Test inverse relationship between PPF and CDF
            float ppfValue = dist.ppf(p);
            float cdfValue = dist.cdf(ppfValue);
            assertEqual(p, cdfValue, 
                dist.getClass().getSimpleName() + " PPF和CDF应为逆函数 / PPF and CDF should be inverse functions");
        }
        
        System.out.println("分布属性测试通过 / Distribution properties test passed");
    }
    
    public static void testEdgeCases() {
        System.out.println("\n测试边界情况 / Testing Edge Cases");
        
        // 测试正态分布的边界情况
        // Test edge cases for normal distribution
        NormalDistribution normal = new NormalDistribution(0.0f, 1.0f);
        
        // 测试极值
        // Test extreme values
        assertEqual(0.0f, normal.pdf(Float.NEGATIVE_INFINITY), "负无穷处的PDF应为0 / PDF at negative infinity should be 0");
        assertEqual(0.0f, normal.pdf(Float.POSITIVE_INFINITY), "正无穷处的PDF应为0 / PDF at positive infinity should be 0");
        assertEqual(0.0f, normal.cdf(Float.NEGATIVE_INFINITY), "负无穷处的CDF应为0 / CDF at negative infinity should be 0");
        assertEqual(1.0f, normal.cdf(Float.POSITIVE_INFINITY), "正无穷处的CDF应为1 / CDF at positive infinity should be 1");
        
        // 测试PPF的边界情况
        // Test edge cases for PPF
        assertTrue(Float.isInfinite(normal.ppf(0.0f)) && normal.ppf(0.0f) < 0, "0%分位数应为负无穷 / 0th percentile should be negative infinity");
        assertTrue(Float.isInfinite(normal.ppf(1.0f)) && normal.ppf(1.0f) > 0, "100%分位数应为正无穷 / 100th percentile should be positive infinity");
        
        // 测试均匀分布的边界情况
        // Test edge cases for uniform distribution
        UniformDistribution uniform = new UniformDistribution(0.0f, 1.0f);
        assertEqual(0.0f, uniform.pdf(-0.1f), "区间外的PDF应为0 / PDF outside interval should be 0");
        assertEqual(0.0f, uniform.pdf(1.1f), "区间外的PDF应为0 / PDF outside interval should be 0");
        
        System.out.println("边界情况测试通过 / Edge cases test passed");
    }
    
    public static void testParameterValidation() {
        System.out.println("\n测试参数验证 / Testing Parameter Validation");
        
        // 测试正态分布参数验证
        // Test normal distribution parameter validation
        assertThrows(() -> {
            new NormalDistribution(0.0f, -1.0f);
        }, "负标准差应抛出异常 / Negative standard deviation should throw exception");
        
        // 测试t分布参数验证
        // Test t-distribution parameter validation
        assertThrows(() -> {
            new StudentDistribution(-1.0f);
        }, "负自由度应抛出异常 / Negative degrees of freedom should throw exception");
        
        // 测试均匀分布参数验证
        // Test uniform distribution parameter validation
        assertThrows(() -> {
            new UniformDistribution(1.0f, 0.0f);
        }, "上界小于下界应抛出异常 / Upper bound less than lower bound should throw exception");
        
        // 测试指数分布参数验证
        // Test exponential distribution parameter validation
        assertThrows(() -> {
            new ExponentialDistribution(-1.0f);
        }, "负速率参数应抛出异常 / Negative rate parameter should throw exception");
        
        // 测试卡方分布参数验证
        // Test chi-squared distribution parameter validation
        assertThrows(() -> {
            new Chi2Distribution(-1.0f);
        }, "负自由度应抛出异常 / Negative degrees of freedom should throw exception");
        
        // 测试F分布参数验证
        // Test F-distribution parameter validation
        assertThrows(() -> {
            new FDistribution(-1.0f, 5.0f);
        }, "负分子自由度应抛出异常 / Negative numerator degrees of freedom should throw exception");
        
        assertThrows(() -> {
            new FDistribution(5.0f, -1.0f);
        }, "负分母自由度应抛出异常 / Negative denominator degrees of freedom should throw exception");
        
        System.out.println("参数验证测试通过 / Parameter validation test passed");
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