package com.reremouse.lab.math.stat.distribution;

import java.util.ArrayList;
import java.util.List;

/**
 * 离散分布验证测试运行器
 * 不依赖JUnit的独立测试框架
 * 
 * Discrete Distribution Validation Test Runner
 * Independent test framework without JUnit dependency
 * 
 * @author lteb2
 */
public class DiscreteDistributionValidationRunner {
    
    private static final float TOLERANCE = 1e-4f;
    private static final float HIGH_TOLERANCE = 1e-2f;
    private static final int SAMPLE_SIZE = 10000;
    
    private static int totalTests = 0;
    private static int passedTests = 0;
    private static int failedTests = 0;
    private static List<String> failures = new ArrayList<>();
    
    public static void main(String[] args) {
        System.out.println("==========================================");
        System.out.println("离散分布正确性验证测试开始");
        System.out.println("Discrete Distribution Correctness Validation Test Started");
        System.out.println("==========================================");
        
        // 运行所有测试
        runBasicFunctionalityTests();
        runMathematicalValidationTests();
        runStatisticalValidationTests();
        runPerformanceTests();
        runEdgeCaseTests();
        
        // 输出测试结果
        printTestResults();
    }
    
    private static void runBasicFunctionalityTests() {
        System.out.println("\n--- 基本功能测试 / Basic Functionality Tests ---");
        
        testBernoulliDistributionBasic();
        testBinomialDistributionBasic();
        testGeometricDistributionBasic();
        testNegativeBinomialDistributionBasic();
        testPoissonDistributionBasic();
        testDiscreteUniformDistributionBasic();
    }
    
    private static void runMathematicalValidationTests() {
        System.out.println("\n--- 数学公式验证测试 / Mathematical Formula Validation Tests ---");
        
        testBernoulliDistributionMathFormulas();
        testBinomialDistributionMathFormulas();
        testGeometricDistributionMathFormulas();
        testPoissonDistributionMathFormulas();
        testDiscreteUniformDistributionMathFormulas();
        testNegativeBinomialDistributionMathFormulas();
    }
    
    private static void runStatisticalValidationTests() {
        System.out.println("\n--- 统计验证测试 / Statistical Validation Tests ---");
        
        testSamplingStatistics();
        testDistributionProperties();
        testDistributionComparison();
    }
    
    private static void runPerformanceTests() {
        System.out.println("\n--- 性能测试 / Performance Tests ---");
        
        testSamplingPerformance();
        testPMFCalculationPerformance();
        testStatisticalCalculationPerformance();
    }
    
    private static void runEdgeCaseTests() {
        System.out.println("\n--- 边界情况测试 / Edge Case Tests ---");
        
        testExtremeParameterValues();
        testNumericalStability();
        testExceptionHandling();
    }
    
    // ==================== 基本功能测试 / Basic Functionality Tests ====================
    
    private static void testBernoulliDistributionBasic() {
        try {
            BernoulliDistribution dist = new BernoulliDistribution(0.3f);
            
            // 测试基本统计量
            assertEqual(0.3f, dist.mean(), TOLERANCE, "伯努利分布均值");
            assertEqual(0.21f, dist.var(), TOLERANCE, "伯努利分布方差");
            assertEqual(0, dist.getMinSupport(), "伯努利分布最小支持值");
            assertEqual(1, dist.getMaxSupport(), "伯努利分布最大支持值");
            assertTrue(dist.isBounded(), "伯努利分布应该是有界的");
            
            // 测试PMF
            assertEqual(0.7f, dist.pmf(0), TOLERANCE, "P(X=0)");
            assertEqual(0.3f, dist.pmf(1), TOLERANCE, "P(X=1)");
            assertEqual(0.0f, dist.pmf(2), TOLERANCE, "P(X=2)");
            
            // 测试CDF
            assertEqual(0.0f, dist.cdf(-1), TOLERANCE, "CDF(-1)");
            assertEqual(0.7f, dist.cdf(0), TOLERANCE, "CDF(0)");
            assertEqual(1.0f, dist.cdf(1), TOLERANCE, "CDF(1)");
            
            // 测试采样
            int[] samples = dist.sample(1000);
            assertEqual(1000, samples.length, "采样数量");
            for (int sample : samples) {
                assertTrue(sample == 0 || sample == 1, "采样值应该在{0,1}中");
            }
            
            System.out.println("✓ 伯努利分布基本功能测试通过");
        } catch (Exception e) {
            recordFailure("伯努利分布基本功能测试", e.getMessage());
        }
    }
    
    private static void testBinomialDistributionBasic() {
        try {
            BinomialDistribution dist = new BinomialDistribution(10, 0.3f);
            
            // 测试基本统计量
            assertEqual(3.0f, dist.mean(), TOLERANCE, "二项分布均值");
            assertEqual(2.1f, dist.var(), TOLERANCE, "二项分布方差");
            assertEqual(0, dist.getMinSupport(), "二项分布最小支持值");
            assertEqual(10, dist.getMaxSupport(), "二项分布最大支持值");
            assertTrue(dist.isBounded(), "二项分布应该是有界的");
            
            // 测试PMF归一化
            float pmfSum = 0.0f;
            for (int x = 0; x <= 10; x++) {
                pmfSum += dist.pmf(x);
            }
            assertEqual(1.0f, pmfSum, TOLERANCE, "PMF归一化");
            
            // 测试特殊情况：n=1时应该退化为伯努利分布
            BinomialDistribution dist1 = new BinomialDistribution(1, 0.3f);
            BernoulliDistribution bernoulli = new BernoulliDistribution(0.3f);
            assertEqual(bernoulli.pmf(0), dist1.pmf(0), TOLERANCE, "n=1时P(X=0)");
            assertEqual(bernoulli.pmf(1), dist1.pmf(1), TOLERANCE, "n=1时P(X=1)");
            
            System.out.println("✓ 二项分布基本功能测试通过");
        } catch (Exception e) {
            recordFailure("二项分布基本功能测试", e.getMessage());
        }
    }
    
    private static void testGeometricDistributionBasic() {
        try {
            GeometricDistribution dist = new GeometricDistribution(0.3f);
            
            // 测试基本统计量
            assertEqual(1.0f / 0.3f, dist.mean(), TOLERANCE, "几何分布均值");
            assertEqual(0.7f / (0.3f * 0.3f), dist.var(), TOLERANCE, "几何分布方差");
            assertEqual(1, dist.getMinSupport(), "几何分布最小支持值");
            assertFalse(dist.isBounded(), "几何分布应该是无界的");
            assertTrue(dist.isMemoryless(), "几何分布应该具有无记忆性");
            
            // 测试PMF公式：P(X=k) = (1-p)^(k-1) * p
            assertEqual(0.3f, dist.pmf(1), TOLERANCE, "P(X=1)");
            assertEqual(0.3f * 0.7f, dist.pmf(2), TOLERANCE, "P(X=2)");
            assertEqual(0.3f * 0.7f * 0.7f, dist.pmf(3), TOLERANCE, "P(X=3)");
            
            // 测试无记忆性
            int s = 3, t = 2;
            float conditionalProb = dist.sf(s + t) / dist.sf(s);
            float unconditionalProb = dist.sf(t);
            assertEqual(unconditionalProb, conditionalProb, TOLERANCE, "无记忆性");
            
            System.out.println("✓ 几何分布基本功能测试通过");
        } catch (Exception e) {
            recordFailure("几何分布基本功能测试", e.getMessage());
        }
    }
    
    private static void testNegativeBinomialDistributionBasic() {
        try {
            NegativeBinomialDistribution dist = new NegativeBinomialDistribution(3, 0.4f);
            
            // 测试基本统计量
            assertEqual(3.0f / 0.4f, dist.mean(), TOLERANCE, "负二项分布均值");
            assertEqual(3.0f * 0.6f / (0.4f * 0.4f), dist.var(), TOLERANCE, "负二项分布方差");
            assertEqual(3, dist.getMinSupport(), "负二项分布最小支持值");
            assertFalse(dist.isBounded(), "负二项分布应该是无界的");
            
            // 测试特殊情况：r=1时应该退化为几何分布
            NegativeBinomialDistribution dist1 = new NegativeBinomialDistribution(1, 0.3f);
            GeometricDistribution geometric = new GeometricDistribution(0.3f);
            for (int x = 1; x <= 5; x++) {
                assertEqual(geometric.pmf(x), dist1.pmf(x), TOLERANCE, "r=1时PMF");
            }
            
            System.out.println("✓ 负二项分布基本功能测试通过");
        } catch (Exception e) {
            recordFailure("负二项分布基本功能测试", e.getMessage());
        }
    }
    
    private static void testPoissonDistributionBasic() {
        try {
            PoissonDistribution dist = new PoissonDistribution(2.5f);
            
            // 测试基本统计量
            assertEqual(2.5f, dist.mean(), TOLERANCE, "泊松分布均值");
            assertEqual(2.5f, dist.var(), TOLERANCE, "泊松分布方差");
            assertEqual(0, dist.getMinSupport(), "泊松分布最小支持值");
            assertFalse(dist.isBounded(), "泊松分布应该是无界的");
            
            // 测试PMF公式：P(X=k) = (λ^k * e^(-λ)) / k!
            float expectedPMF0 = (float) Math.exp(-2.5f);
            float expectedPMF1 = 2.5f * (float) Math.exp(-2.5f);
            float expectedPMF2 = 2.5f * 2.5f * (float) Math.exp(-2.5f) / 2.0f;
            
            assertEqual(expectedPMF0, dist.pmf(0), TOLERANCE, "P(X=0)");
            assertEqual(expectedPMF1, dist.pmf(1), TOLERANCE, "P(X=1)");
            assertEqual(expectedPMF2, dist.pmf(2), TOLERANCE, "P(X=2)");
            
            System.out.println("✓ 泊松分布基本功能测试通过");
        } catch (Exception e) {
            recordFailure("泊松分布基本功能测试", e.getMessage());
        }
    }
    
    private static void testDiscreteUniformDistributionBasic() {
        try {
            DiscreteUniformDistribution dist = new DiscreteUniformDistribution(2, 7);
            
            // 测试基本统计量
            assertEqual(4.5f, dist.mean(), TOLERANCE, "离散均匀分布均值");
            assertEqual(35.0f / 12.0f, dist.var(), TOLERANCE, "离散均匀分布方差");
            assertEqual(2, dist.getMinSupport(), "离散均匀分布最小支持值");
            assertEqual(7, dist.getMaxSupport(), "离散均匀分布最大支持值");
            assertTrue(dist.isBounded(), "离散均匀分布应该是有界的");
            assertTrue(dist.isSymmetric(), "离散均匀分布应该是对称的");
            
            // 测试PMF：每个值概率相等
            float expectedPMF = 1.0f / 6.0f; // n = 7 - 2 + 1 = 6
            for (int x = 2; x <= 7; x++) {
                assertEqual(expectedPMF, dist.pmf(x), TOLERANCE, "P(X=" + x + ")");
            }
            
            // 测试构造函数
            DiscreteUniformDistribution dist1 = new DiscreteUniformDistribution(5);
            assertEqual(0, dist1.getMinSupport(), "单参数构造函数最小支持值");
            assertEqual(4, dist1.getMaxSupport(), "单参数构造函数最大支持值");
            
            System.out.println("✓ 离散均匀分布基本功能测试通过");
        } catch (Exception e) {
            recordFailure("离散均匀分布基本功能测试", e.getMessage());
        }
    }
    
    // ==================== 数学公式验证测试 / Mathematical Formula Validation Tests ====================
    
    private static void testBernoulliDistributionMathFormulas() {
        try {
            float p = 0.4f;
            BernoulliDistribution dist = new BernoulliDistribution(p);
            float q = 1.0f - p;
            
            // 验证均值公式：E[X] = p
            assertEqual(p, dist.mean(), TOLERANCE, "均值公式 E[X] = p");
            
            // 验证方差公式：Var[X] = p(1-p)
            assertEqual(p * q, dist.var(), TOLERANCE, "方差公式 Var[X] = p(1-p)");
            
            // 验证偏度公式：Skew[X] = (1-2p)/√(p(1-p))
            float expectedSkewness = (1.0f - 2.0f * p) / (float) Math.sqrt(p * q);
            assertEqual(expectedSkewness, dist.skewness(), TOLERANCE, "偏度公式");
            
            System.out.println("✓ 伯努利分布数学公式验证通过");
        } catch (Exception e) {
            recordFailure("伯努利分布数学公式验证", e.getMessage());
        }
    }
    
    private static void testBinomialDistributionMathFormulas() {
        try {
            int n = 10;
            float p = 0.3f;
            BinomialDistribution dist = new BinomialDistribution(n, p);
            float q = 1.0f - p;
            
            // 验证均值公式：E[X] = np
            assertEqual(n * p, dist.mean(), TOLERANCE, "均值公式 E[X] = np");
            
            // 验证方差公式：Var[X] = np(1-p)
            assertEqual(n * p * q, dist.var(), TOLERANCE, "方差公式 Var[X] = np(1-p)");
            
            // 验证偏度公式：Skew[X] = (1-2p)/√(np(1-p))
            float expectedSkewness = (1.0f - 2.0f * p) / (float) Math.sqrt(n * p * q);
            assertEqual(expectedSkewness, dist.skewness(), TOLERANCE, "偏度公式");
            
            System.out.println("✓ 二项分布数学公式验证通过");
        } catch (Exception e) {
            recordFailure("二项分布数学公式验证", e.getMessage());
        }
    }
    
    private static void testGeometricDistributionMathFormulas() {
        try {
            float p = 0.3f;
            GeometricDistribution dist = new GeometricDistribution(p);
            float q = 1.0f - p;
            
            // 验证均值公式：E[X] = 1/p
            assertEqual(1.0f / p, dist.mean(), TOLERANCE, "均值公式 E[X] = 1/p");
            
            // 验证方差公式：Var[X] = (1-p)/p²
            assertEqual(q / (p * p), dist.var(), TOLERANCE, "方差公式 Var[X] = (1-p)/p²");
            
            // 验证偏度公式：Skew[X] = (2-p)/√(1-p)
            float expectedSkewness = (2.0f - p) / (float) Math.sqrt(q);
            assertEqual(expectedSkewness, dist.skewness(), TOLERANCE, "偏度公式");
            
            System.out.println("✓ 几何分布数学公式验证通过");
        } catch (Exception e) {
            recordFailure("几何分布数学公式验证", e.getMessage());
        }
    }
    
    private static void testPoissonDistributionMathFormulas() {
        try {
            float lambda = 2.5f;
            PoissonDistribution dist = new PoissonDistribution(lambda);
            
            // 验证均值公式：E[X] = λ
            assertEqual(lambda, dist.mean(), TOLERANCE, "均值公式 E[X] = λ");
            
            // 验证方差公式：Var[X] = λ
            assertEqual(lambda, dist.var(), TOLERANCE, "方差公式 Var[X] = λ");
            
            // 验证偏度公式：Skew[X] = 1/√λ
            float expectedSkewness = 1.0f / (float) Math.sqrt(lambda);
            assertEqual(expectedSkewness, dist.skewness(), TOLERANCE, "偏度公式");
            
            System.out.println("✓ 泊松分布数学公式验证通过");
        } catch (Exception e) {
            recordFailure("泊松分布数学公式验证", e.getMessage());
        }
    }
    
    private static void testDiscreteUniformDistributionMathFormulas() {
        try {
            int a = 2, b = 7;
            DiscreteUniformDistribution dist = new DiscreteUniformDistribution(a, b);
            int n = b - a + 1; // 6
            
            // 验证均值公式：E[X] = (a+b)/2
            assertEqual((a + b) / 2.0f, dist.mean(), TOLERANCE, "均值公式 E[X] = (a+b)/2");
            
            // 验证方差公式：Var[X] = (n²-1)/12
            float expectedVar = (n * n - 1) / 12.0f;
            assertEqual(expectedVar, dist.var(), TOLERANCE, "方差公式 Var[X] = (n²-1)/12");
            
            // 验证偏度公式：Skew[X] = 0（对称分布）
            assertEqual(0.0f, dist.skewness(), TOLERANCE, "偏度公式 Skew[X] = 0");
            
            System.out.println("✓ 离散均匀分布数学公式验证通过");
        } catch (Exception e) {
            recordFailure("离散均匀分布数学公式验证", e.getMessage());
        }
    }
    
    private static void testNegativeBinomialDistributionMathFormulas() {
        try {
            int r = 3;
            float p = 0.4f;
            NegativeBinomialDistribution dist = new NegativeBinomialDistribution(r, p);
            float q = 1.0f - p;
            
            // 验证均值公式：E[X] = r/p
            assertEqual(r / p, dist.mean(), TOLERANCE, "均值公式 E[X] = r/p");
            
            // 验证方差公式：Var[X] = r(1-p)/p²
            assertEqual(r * q / (p * p), dist.var(), TOLERANCE, "方差公式 Var[X] = r(1-p)/p²");
            
            // 验证偏度公式：Skew[X] = (2-p)/√(r(1-p))
            float expectedSkewness = (2.0f - p) / (float) Math.sqrt(r * q);
            assertEqual(expectedSkewness, dist.skewness(), TOLERANCE, "偏度公式");
            
            System.out.println("✓ 负二项分布数学公式验证通过");
        } catch (Exception e) {
            recordFailure("负二项分布数学公式验证", e.getMessage());
        }
    }
    
    // ==================== 统计验证测试 / Statistical Validation Tests ====================
    
    private static void testSamplingStatistics() {
        try {
            BinomialDistribution dist = new BinomialDistribution(20, 0.3f);
            int[] samples = dist.sample(SAMPLE_SIZE);
            
            // 计算样本均值
            float sampleMean = 0.0f;
            for (int sample : samples) {
                sampleMean += sample;
            }
            sampleMean /= samples.length;
            
            // 验证样本均值接近理论均值
            float meanError = Math.abs(sampleMean - dist.mean()) / dist.mean();
            assertTrue(meanError < HIGH_TOLERANCE, "样本均值误差过大: " + meanError);
            
            System.out.println("✓ 采样统计验证通过");
        } catch (Exception e) {
            recordFailure("采样统计验证", e.getMessage());
        }
    }
    
    private static void testDistributionProperties() {
        try {
            // 测试分布的基本属性
            BernoulliDistribution dist = new BernoulliDistribution(0.5f);
            
            // 验证分布名称
            assertNotNull(dist.getDistributionName(), "分布名称不应为空");
            assertNotNull(dist.getParameterInfo(), "参数信息不应为空");
            
            // 验证支持区间
            assertTrue(dist.getMinSupport() <= dist.getMaxSupport(), "最小支持值应小于等于最大支持值");
            assertTrue(dist.isInSupport(dist.getMinSupport()), "最小支持值应在支持区间内");
            
            System.out.println("✓ 分布属性验证通过");
        } catch (Exception e) {
            recordFailure("分布属性验证", e.getMessage());
        }
    }
    
    private static void testDistributionComparison() {
        try {
            BernoulliDistribution dist1 = new BernoulliDistribution(0.3f);
            BernoulliDistribution dist2 = new BernoulliDistribution(0.7f);
            
            // 测试KL散度
            float kl = dist1.klDivergence(dist2);
            assertTrue(kl > 0, "KL散度应该为正");
            
            // 测试相同分布的KL散度
            float klSame = dist1.klDivergence(dist1);
            assertEqual(0.0f, klSame, TOLERANCE, "相同分布的KL散度应该为0");
            
            // 测试JS散度
            float js = dist1.jsDivergence(dist2);
            assertTrue(js > 0, "JS散度应该为正");
            assertTrue(js <= 1.0f, "JS散度应该小于等于1");
            
            // 测试Wasserstein距离
            float wasserstein = dist1.wassersteinDistance(dist2);
            assertTrue(wasserstein > 0, "Wasserstein距离应该为正");
            
            System.out.println("✓ 分布比较验证通过");
        } catch (Exception e) {
            recordFailure("分布比较验证", e.getMessage());
        }
    }
    
    // ==================== 性能测试 / Performance Tests ====================
    
    private static void testSamplingPerformance() {
        try {
            BinomialDistribution dist = new BinomialDistribution(100, 0.5f);
            
            long startTime = System.currentTimeMillis();
            int[] samples = dist.sample(100000);
            long duration = System.currentTimeMillis() - startTime;
            
            assertTrue(duration < 5000, "采样100000个样本应该在5秒内完成: " + duration + "ms");
            assertEqual(100000, samples.length, "采样数量不正确");
            
            System.out.println("✓ 采样性能测试通过 (" + duration + "ms)");
        } catch (Exception e) {
            recordFailure("采样性能测试", e.getMessage());
        }
    }
    
    private static void testPMFCalculationPerformance() {
        try {
            BinomialDistribution dist = new BinomialDistribution(1000, 0.5f);
            
            long startTime = System.currentTimeMillis();
            for (int i = 0; i < 1000; i++) {
                dist.pmf(i);
            }
            long duration = System.currentTimeMillis() - startTime;
            
            assertTrue(duration < 1000, "计算1000个PMF值应该在1秒内完成: " + duration + "ms");
            
            System.out.println("✓ PMF计算性能测试通过 (" + duration + "ms)");
        } catch (Exception e) {
            recordFailure("PMF计算性能测试", e.getMessage());
        }
    }
    
    private static void testStatisticalCalculationPerformance() {
        try {
            BinomialDistribution dist = new BinomialDistribution(100, 0.5f);
            
            long startTime = System.currentTimeMillis();
            
            // 计算各种统计量
            float mean = dist.mean();
            float var = dist.var();
            float std = dist.std();
            float skewness = dist.skewness();
            float kurtosis = dist.kurtosis();
            float entropy = dist.entropy();
            
            long duration = System.currentTimeMillis() - startTime;
            
            assertTrue(duration < 100, "统计量计算应该在100ms内完成: " + duration + "ms");
            assertFalse(Float.isNaN(mean), "均值计算产生NaN");
            assertFalse(Float.isNaN(var), "方差计算产生NaN");
            
            System.out.println("✓ 统计量计算性能测试通过 (" + duration + "ms)");
        } catch (Exception e) {
            recordFailure("统计量计算性能测试", e.getMessage());
        }
    }
    
    // ==================== 边界情况测试 / Edge Case Tests ====================
    
    private static void testExtremeParameterValues() {
        try {
            // 测试极小的概率值
            BernoulliDistribution dist1 = new BernoulliDistribution(1e-6f);
            assertTrue(dist1.pmf(0) > dist1.pmf(1), "极小概率时P(X=0)应该大于P(X=1)");
            
            // 测试接近1的概率值
            BernoulliDistribution dist2 = new BernoulliDistribution(1.0f - 1e-6f);
            assertTrue(dist2.pmf(1) > dist2.pmf(0), "接近1的概率时P(X=1)应该大于P(X=0)");
            
            // 测试大参数
            BinomialDistribution dist3 = new BinomialDistribution(1000, 0.5f);
            assertTrue(dist3.mean() > 0, "大n值时均值应该为正");
            assertTrue(dist3.var() > 0, "大n值时方差应该为正");
            
            System.out.println("✓ 极值参数测试通过");
        } catch (Exception e) {
            recordFailure("极值参数测试", e.getMessage());
        }
    }
    
    private static void testNumericalStability() {
        try {
            BinomialDistribution dist = new BinomialDistribution(1000, 0.5f);
            
            // 验证PMF值在合理范围内
            for (int x = 0; x <= 1000; x += 100) {
                float pmf = dist.pmf(x);
                assertTrue(pmf >= 0.0f && pmf <= 1.0f, "PMF值应该在[0,1]范围内: P(X=" + x + ")=" + pmf);
                assertFalse(Float.isNaN(pmf), "PMF值不应该是NaN: P(X=" + x + ")=" + pmf);
                assertFalse(Float.isInfinite(pmf), "PMF值不应该是无穷大: P(X=" + x + ")=" + pmf);
            }
            
            System.out.println("✓ 数值稳定性测试通过");
        } catch (Exception e) {
            recordFailure("数值稳定性测试", e.getMessage());
        }
    }
    
    private static void testExceptionHandling() {
        try {
            // 测试伯努利分布异常
            testException(() -> new BernoulliDistribution(-0.1f), "负概率应该抛出异常");
            testException(() -> new BernoulliDistribution(1.1f), "大于1的概率应该抛出异常");
            
            // 测试二项分布异常
            testException(() -> new BinomialDistribution(0, 0.5f), "n=0应该抛出异常");
            testException(() -> new BinomialDistribution(-1, 0.5f), "n<0应该抛出异常");
            testException(() -> new BinomialDistribution(10, -0.1f), "p<0应该抛出异常");
            testException(() -> new BinomialDistribution(10, 1.1f), "p>1应该抛出异常");
            
            // 测试几何分布异常
            testException(() -> new GeometricDistribution(0.0f), "p=0应该抛出异常");
            testException(() -> new GeometricDistribution(-0.1f), "p<0应该抛出异常");
            testException(() -> new GeometricDistribution(1.1f), "p>1应该抛出异常");
            
            // 测试泊松分布异常
            testException(() -> new PoissonDistribution(0.0f), "λ=0应该抛出异常");
            testException(() -> new PoissonDistribution(-1.0f), "λ<0应该抛出异常");
            
            // 测试离散均匀分布异常
            testException(() -> new DiscreteUniformDistribution(5, 3), "上界小于下界应该抛出异常");
            testException(() -> new DiscreteUniformDistribution(0), "n=0应该抛出异常");
            testException(() -> new DiscreteUniformDistribution(-1), "n<0应该抛出异常");
            
            // 测试负二项分布异常
            testException(() -> new NegativeBinomialDistribution(0, 0.5f), "r=0应该抛出异常");
            testException(() -> new NegativeBinomialDistribution(-1, 0.5f), "r<0应该抛出异常");
            testException(() -> new NegativeBinomialDistribution(5, 0.0f), "p=0应该抛出异常");
            testException(() -> new NegativeBinomialDistribution(5, 1.0f), "p=1应该抛出异常");
            
            System.out.println("✓ 异常处理测试通过");
        } catch (Exception e) {
            recordFailure("异常处理测试", e.getMessage());
        }
    }
    
    // ==================== 辅助方法 / Helper Methods ====================
    
    private static void assertEqual(float expected, float actual, float tolerance, String message) {
        totalTests++;
        if (Math.abs(expected - actual) <= tolerance) {
            passedTests++;
        } else {
            failedTests++;
            recordFailure(message, String.format("期望=%.6f, 实际=%.6f, 误差=%.6f", 
                expected, actual, Math.abs(expected - actual)));
        }
    }
    
    private static void assertEqual(int expected, int actual, String message) {
        totalTests++;
        if (expected == actual) {
            passedTests++;
        } else {
            failedTests++;
            recordFailure(message, String.format("期望=%d, 实际=%d", expected, actual));
        }
    }
    
    private static void assertTrue(boolean condition, String message) {
        totalTests++;
        if (condition) {
            passedTests++;
        } else {
            failedTests++;
            recordFailure(message, "条件为false");
        }
    }
    
    private static void assertFalse(boolean condition, String message) {
        totalTests++;
        if (!condition) {
            passedTests++;
        } else {
            failedTests++;
            recordFailure(message, "条件为true");
        }
    }
    
    private static void assertNotNull(Object obj, String message) {
        totalTests++;
        if (obj != null) {
            passedTests++;
        } else {
            failedTests++;
            recordFailure(message, "对象为null");
        }
    }
    
    private static void testException(Runnable code, String message) {
        totalTests++;
        try {
            code.run();
            failedTests++;
            recordFailure(message, "应该抛出异常但没有抛出");
        } catch (Exception e) {
            if (e instanceof IllegalArgumentException) {
                passedTests++;
            } else {
                failedTests++;
                recordFailure(message, "抛出了错误的异常类型: " + e.getClass().getSimpleName());
            }
        }
    }
    
    private static void recordFailure(String testName, String reason) {
        failures.add(testName + ": " + reason);
    }
    
    private static void printTestResults() {
        System.out.println("\n==========================================");
        System.out.println("测试结果汇总 / Test Results Summary");
        System.out.println("==========================================");
        System.out.println("总测试数 / Total Tests: " + totalTests);
        System.out.println("通过测试 / Passed Tests: " + passedTests);
        System.out.println("失败测试 / Failed Tests: " + failedTests);
        System.out.println("成功率 / Success Rate: " + String.format("%.2f%%", 
            (double) passedTests / totalTests * 100));
        
        if (failedTests > 0) {
            System.out.println("\n失败测试详情 / Failed Test Details:");
            for (String failure : failures) {
                System.out.println("  ✗ " + failure);
            }
        } else {
            System.out.println("\n🎉 所有测试通过！/ All tests passed! 🎉");
        }
        
        System.out.println("==========================================");
    }
}
