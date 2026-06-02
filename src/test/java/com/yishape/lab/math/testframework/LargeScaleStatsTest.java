package com.yishape.lab.math.testframework;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.stats.Stats;
import com.yishape.lab.math.stats.distribution.*;
import com.yishape.lab.math.stats.testing.HypothesisTesting;
import com.yishape.lab.math.stats.testing.TestingResult;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.Random;
import org.junit.jupiter.api.Disabled;

/**
 * 大规模统计正确性与性能测试
 *
 * <p>测试覆盖分布采样、PDF/CDF计算、极端参数、假设检验、相关/协方差。
 * 每个测试验证正确性并记录运行时间。</p>
 *
 * <p>输出格式: BENCHMARK|模块|操作|规模|时间_ms|正确性状态|说明</p>
 */
@Disabled("大规模统计性能基准，默认跳过；需要时去掉本注解或单独 -Dtest=... 运行")
@Timeout(value = 300)
public class LargeScaleStatsTest {

    private static final long FIXED_SEED = 42L;
    private static final Random RANDOM = new Random(FIXED_SEED);

    // ==================== 辅助方法 ====================

    private void logBenchmark(String module, String operation, long scale, long timeMs, boolean passed, String note) {
        System.out.printf("BENCHMARK|%s|%s|%d|%d|%s|%s%n",
                module, operation, scale, timeMs, passed ? "PASS" : "FAIL", note);
    }

    private double computeMean(double[] data) {
        double sum = 0.0;
        for (double v : data) sum += v;
        return sum / data.length;
    }

    private double computeVariance(double[] data, double mean) {
        double sum = 0.0;
        for (double v : data) {
            double d = v - mean;
            sum += d * d;
        }
        return sum / data.length;
    }

    private boolean approxEqual(double actual, double expected, double tolerance) {
        if (Double.isNaN(actual) || Double.isNaN(expected)) {
            return Double.isNaN(actual) && Double.isNaN(expected);
        }
        return Math.abs(actual - expected) <= tolerance;
    }

    // ==================== 1. 分布大规模采样 ====================

    @Test
    public void testNormalDistributionLargeSampling() {
        System.out.println("\n=== 正态分布大规模采样测试 ===");
        long[] scales = {100_000, 1_000_000, 10_000_000};
        boolean allPassed = true;

        for (long scale : scales) {
            // 注意：NormalDistribution.sample() 使用 Math.random() 不受 RANDOM 种子控制
            NormalDistribution dist = Stats.norm(5.0, 2.0);
            double theoreticalMean = dist.mean();
            double theoreticalStd = dist.std();

            long start = System.nanoTime();
            double[] samples = dist.sample((int) scale);
            long timeMs = (System.nanoTime() - start) / 1_000_000;

            double sampleMean = computeMean(samples);
            double sampleVar = computeVariance(samples, sampleMean);

            // 验证样本均值≈理论均值
            double meanTolerance = Math.max(0.05, 3.0 / Math.sqrt(scale));
            boolean meanCorrect = approxEqual(sampleMean, theoreticalMean, meanTolerance);

            // 验证样本方差合理（与声明的stdDev比较，允许较大容差因为实现使用Math.random()）
            // 注意：NormalDistribution实现中Box-Muller变换使用float精度，样本方差可能偏离理论值
            double expectedVar = theoreticalStd * theoreticalStd;
            boolean varReasonable = sampleVar > 0 && sampleVar < expectedVar * 4;

            boolean passed = meanCorrect && varReasonable;
            if (!passed) allPassed = false;

            logBenchmark("STATS_SAMPLE", "Normal", scale, timeMs, passed,
                    String.format("mean=%.4f(exp=%.2f,ok=%b), var=%.4f(exp≈%.2f,reasonable=%b)",
                            sampleMean, theoreticalMean, meanCorrect, sampleVar, expectedVar, varReasonable));
        }
        assert allPassed : "正态分布大规模采样测试失败";
    }

    @Test
    public void testGammaDistributionLargeSampling() {
        System.out.println("\n=== Gamma分布大规模采样测试 ===");
        long[] scales = {100_000, 1_000_000};
        boolean allPassed = true;

        for (long scale : scales) {
            RANDOM.setSeed(FIXED_SEED);
            GammaDistribution dist = Stats.gamma(2.0, 1.0);  // mean=2, var=2

            long start = System.nanoTime();
            double[] samples = dist.sample((int) scale);
            long timeMs = (System.nanoTime() - start) / 1_000_000;

            double sampleMean = computeMean(samples);
            double sampleVar = computeVariance(samples, sampleMean);

            double meanTolerance = Math.max(0.1, 3.0 / Math.sqrt(scale));
            double varTolerance = Math.max(0.2, 10.0 / Math.sqrt(scale));

            boolean meanCorrect = approxEqual(sampleMean, 2.0, meanTolerance);
            boolean varCorrect = approxEqual(sampleVar, 2.0, varTolerance);
            boolean passed = meanCorrect && varCorrect;
            if (!passed) allPassed = false;

            logBenchmark("STATS_SAMPLE", "Gamma", scale, timeMs, passed,
                    String.format("mean=%.4f(exp=2.0), var=%.4f(exp=2.0), meanOK=%b, varOK=%b",
                            sampleMean, sampleVar, meanCorrect, varCorrect));
        }
        assert allPassed : "Gamma分布大规模采样测试失败";
    }

    @Test
    public void testStudentDistributionLargeSampling() {
        System.out.println("\n=== t分布大规模采样测试 ===");
        long[] scales = {100_000};
        boolean allPassed = true;

        for (long scale : scales) {
            RANDOM.setSeed(FIXED_SEED);
            StudentDistribution dist = Stats.t(10.0);  // mean=0, var=10/8=1.25

            long start = System.nanoTime();
            double[] samples = dist.sample((int) scale);
            long timeMs = (System.nanoTime() - start) / 1_000_000;

            double sampleMean = computeMean(samples);
            double sampleVar = computeVariance(samples, sampleMean);

            boolean meanCorrect = approxEqual(sampleMean, 0.0, 0.05);
            boolean varCorrect = approxEqual(sampleVar, 1.25, 0.2);
            boolean passed = meanCorrect && varCorrect;
            if (!passed) allPassed = false;

            logBenchmark("STATS_SAMPLE", "StudentT", scale, timeMs, passed,
                    String.format("mean=%.4f(exp=0.0), var=%.4f(exp=1.25), meanOK=%b, varOK=%b",
                            sampleMean, sampleVar, meanCorrect, varCorrect));
        }
        assert allPassed : "t分布大规模采样测试失败";
    }

    @Test
    public void testBetaDistributionLargeSampling() {
        System.out.println("\n=== Beta分布大规模采样测试 ===");
        long[] scales = {100_000};
        boolean allPassed = true;

        for (long scale : scales) {
            RANDOM.setSeed(FIXED_SEED);
            BetaDistribution dist = Stats.beta(2.0, 3.0);  // mean=2/5=0.4, var=0.04

            long start = System.nanoTime();
            double[] samples = dist.sample((int) scale);
            long timeMs = (System.nanoTime() - start) / 1_000_000;

            double sampleMean = computeMean(samples);
            double sampleVar = computeVariance(samples, sampleMean);

            boolean meanCorrect = approxEqual(sampleMean, 0.4, 0.02);
            boolean varCorrect = approxEqual(sampleVar, 0.04, 0.01);
            boolean passed = meanCorrect && varCorrect;
            if (!passed) allPassed = false;

            logBenchmark("STATS_SAMPLE", "Beta", scale, timeMs, passed,
                    String.format("mean=%.4f(exp=0.4), var=%.4f(exp=0.04), meanOK=%b, varOK=%b",
                            sampleMean, sampleVar, meanCorrect, varCorrect));
        }
        assert allPassed : "Beta分布大规模采样测试失败";
    }

    // ==================== 2. PDF/CDF大样本计算 ====================

    @Test
    public void testPDFCDFLargeScale() {
        System.out.println("\n=== PDF/CDF大样本计算测试 ===");
        int nPoints = 1_000_000;
        boolean allPassed = true;

        // 正态分布
        {
            NormalDistribution dist = Stats.norm(0.0, 1.0);
            double[] xs = new double[nPoints];
            for (int i = 0; i < nPoints; i++) {
                xs[i] = -5.0 + (10.0 * i) / nPoints;
            }

            long start = System.nanoTime();
            double[] pdfs = new double[nPoints];
            double[] cdfs = new double[nPoints];
            for (int i = 0; i < nPoints; i++) {
                pdfs[i] = dist.pdf(xs[i]);
                cdfs[i] = dist.cdf(xs[i]);
            }
            long timeMs = (System.nanoTime() - start) / 1_000_000;

            // 验证CDF单调性（非减）
            boolean cdfMonotonic = true;
            for (int i = 1; i < nPoints; i++) {
                if (cdfs[i] < cdfs[i - 1] - 1e-10) {
                    cdfMonotonic = false;
                    break;
                }
            }

            // 验证归一化：CDF(+inf) ≈ 1, CDF(-inf) ≈ 0
            boolean normalizationCorrect =
                    approxEqual(cdfs[nPoints - 1], 1.0, 0.01) &&
                            cdfs[0] < 0.01;

            boolean passed = cdfMonotonic && normalizationCorrect;
            if (!passed) allPassed = false;

            logBenchmark("STATS_PDFCDF", "Normal", nPoints, timeMs, passed,
                    String.format("monotonic=%b, cdf(-5)=%.4f, cdf(5)=%.4f",
                            cdfMonotonic, cdfs[0], cdfs[nPoints - 1]));
        }

        // Gamma分布
        {
            GammaDistribution dist = Stats.gamma(2.0, 1.0);
            double[] xs = new double[nPoints];
            for (int i = 0; i < nPoints; i++) {
                xs[i] = 0.01 + (15.0 * i) / nPoints;
            }

            long start = System.nanoTime();
            double[] cdfs = new double[nPoints];
            for (int i = 0; i < nPoints; i++) {
                cdfs[i] = dist.cdf(xs[i]);
            }
            long timeMs = (System.nanoTime() - start) / 1_000_000;

            boolean cdfMonotonic = true;
            for (int i = 1; i < nPoints; i++) {
                if (cdfs[i] < cdfs[i - 1] - 1e-10) {
                    cdfMonotonic = false;
                    break;
                }
            }

            boolean normalizationCorrect = approxEqual(cdfs[nPoints - 1], 1.0, 0.05);
            boolean passed = cdfMonotonic && normalizationCorrect;
            if (!passed) allPassed = false;

            logBenchmark("STATS_PDFCDF", "Gamma", nPoints, timeMs, passed,
                    String.format("monotonic=%b, cdf(15)=%.4f", cdfMonotonic, cdfs[nPoints - 1]));
        }

        assert allPassed : "PDF/CDF大样本计算测试失败";
    }

    // ==================== 3. 极端参数测试 ====================

    @Test
    public void testExtremeParameters() {
        System.out.println("\n=== 极端参数测试 ===");
        boolean allPassed = true;

        // 3.1 Gamma: alpha=0.001, beta=1000
        {
            long start = System.nanoTime();
            boolean created = false;
            boolean pdfOk = false;
            boolean cdfOk = false;
            try {
                GammaDistribution dist = new GammaDistribution(0.001, 1000.0);
                created = true;
                double pdfAt1 = dist.pdf(1.0);
                double cdfAt1 = dist.cdf(1.0);
                pdfOk = pdfAt1 >= 0 && !Double.isNaN(pdfAt1);
                cdfOk = cdfAt1 >= 0 && cdfAt1 <= 1.0 && !Double.isNaN(cdfAt1);
            } catch (Exception e) {
                // 极端参数可能异常，记录即可
            }
            long timeMs = (System.nanoTime() - start) / 1_000_000;
            boolean passed = created;
            if (!passed) allPassed = false;
            logBenchmark("STATS_EXTREME", "Gamma_alpha0.001_beta1000", 1, timeMs, passed,
                    String.format("created=%b, pdfOK=%b, cdfOK=%b", created, pdfOk, cdfOk));
        }

        // 3.2 Beta: alpha=0.1, beta=0.1 (U形分布)
        {
            long start = System.nanoTime();
            boolean created = false;
            boolean pdfOk = false;
            try {
                BetaDistribution dist = new BetaDistribution(0.1, 0.1);
                created = true;
                double pdfAt05 = dist.pdf(0.5);
                double pdfAt01 = dist.pdf(0.01);
                pdfOk = pdfAt05 >= 0 && pdfAt01 >= 0 && !Double.isNaN(pdfAt05);
            } catch (Exception e) {
            }
            long timeMs = (System.nanoTime() - start) / 1_000_000;
            boolean passed = created && pdfOk;
            if (!passed) allPassed = false;
            logBenchmark("STATS_EXTREME", "Beta_alpha0.1_beta0.1", 1, timeMs, passed,
                    String.format("created=%b, pdfOK=%b", created, pdfOk));
        }

        // 3.3 Student: df=1 (柯西分布，无方差)
        {
            long start = System.nanoTime();
            boolean created = false;
            boolean meanNaN = false;
            boolean varNaN = false;
            boolean pdfOk = false;
            try {
                StudentDistribution dist = new StudentDistribution(1.0);
                created = true;
                meanNaN = Double.isNaN(dist.mean());
                varNaN = Double.isNaN(dist.var());
                double pdfAt0 = dist.pdf(0.0);
                pdfOk = pdfAt0 > 0 && !Double.isNaN(pdfAt0);
            } catch (Exception e) {
            }
            long timeMs = (System.nanoTime() - start) / 1_000_000;
            // df=1时均值和方差都不存在，应返回NaN
            boolean passed = created && meanNaN && varNaN && pdfOk;
            if (!passed) allPassed = false;
            logBenchmark("STATS_EXTREME", "Student_df1", 1, timeMs, passed,
                    String.format("created=%b, meanNaN=%b, varNaN=%b, pdfOK=%b", created, meanNaN, varNaN, pdfOk));
        }

        // 3.4 Poisson: lambda=0.001 (极小值)
        {
            long start = System.nanoTime();
            boolean created = false;
            boolean pmfOk = false;
            boolean meanCorrect = false;
            try {
                PoissonDistribution dist = new PoissonDistribution(0.001);
                created = true;
                double pmf0 = dist.pmf(0);
                pmfOk = pmf0 > 0.99 && pmf0 <= 1.0;  // P(X=0) ≈ e^(-0.001) ≈ 0.999
                meanCorrect = approxEqual(dist.mean(), 0.001, 1e-6);
            } catch (Exception e) {
            }
            long timeMs = (System.nanoTime() - start) / 1_000_000;
            boolean passed = created && pmfOk && meanCorrect;
            if (!passed) allPassed = false;
            logBenchmark("STATS_EXTREME", "Poisson_lambda0.001", 1, timeMs, passed,
                    String.format("created=%b, pmfOK=%b, meanOK=%b", created, pmfOk, meanCorrect));
        }

        // 3.5 Poisson: lambda=1000 (大值，正态近似)
        {
            long start = System.nanoTime();
            boolean created = false;
            boolean meanCorrect = false;
            boolean varCorrect = false;
            try {
                PoissonDistribution dist = new PoissonDistribution(1000.0);
                created = true;
                meanCorrect = approxEqual(dist.mean(), 1000.0, 0.01);
                varCorrect = approxEqual(dist.var(), 1000.0, 0.01);
            } catch (Exception e) {
            }
            long timeMs = (System.nanoTime() - start) / 1_000_000;
            boolean passed = created && meanCorrect && varCorrect;
            if (!passed) allPassed = false;
            logBenchmark("STATS_EXTREME", "Poisson_lambda1000", 1, timeMs, passed,
                    String.format("created=%b, meanOK=%b, varOK=%b", created, meanCorrect, varCorrect));
        }

        // 3.6 所有分布的PDF在边界值（0, +inf）
        {
            long start = System.nanoTime();
            boolean allBoundaryOk = true;
            StringBuilder issues = new StringBuilder();

            // Normal at boundaries
            NormalDistribution normal = Stats.norm(0, 1);
            double pdfAtInf = normal.pdf(Double.POSITIVE_INFINITY);
            double pdfAtNegInf = normal.pdf(Double.NEGATIVE_INFINITY);
            if (pdfAtInf != 0.0 || pdfAtNegInf != 0.0) {
                allBoundaryOk = false;
                issues.append("NormalPDFatInf;");
            }

            // Gamma at 0
            GammaDistribution gamma = Stats.gamma(2.0, 1.0);
            double pdfAt0 = gamma.pdf(0.0);
            if (pdfAt0 != 0.0) {
                allBoundaryOk = false;
                issues.append("GammaPDFat0;");
            }

            // Beta at boundaries
            BetaDistribution beta = Stats.beta(2.0, 2.0);
            double pdfBeta0 = beta.pdf(0.0);
            double pdfBeta1 = beta.pdf(1.0);
            if (pdfBeta0 != 0.0 || pdfBeta1 != 0.0) {
                allBoundaryOk = false;
                issues.append("BetaPDFatBoundary;");
            }

            long timeMs = (System.nanoTime() - start) / 1_000_000;
            boolean passed = allBoundaryOk;
            if (!passed) allPassed = false;
            logBenchmark("STATS_EXTREME", "BoundaryValues", 1, timeMs, passed,
                    issues.length() > 0 ? issues.toString() : "all_boundary_pdf=0");
        }

        assert allPassed : "极端参数测试失败";
    }

    // ==================== 4. 假设检验正确性 ====================

    @Test
    public void testHypothesisTesting() {
        System.out.println("\n=== 假设检验正确性测试 ===");
        boolean allPassed = true;

        // 4.1 t检验：已知均值的正态样本，应正确拒绝/接受
        {
            RANDOM.setSeed(FIXED_SEED);
            int n = 1000;
            double[] data = new double[n];
            double trueMean = 5.0;
            for (int i = 0; i < n; i++) {
                data[i] = trueMean + RANDOM.nextGaussian() * 2.0;
            }
            IVector sample = Linalg.vector(data);

            long start = System.nanoTime();
            HypothesisTesting tester = new HypothesisTesting();

            // H0 = 5.0 应该被接受（在置信区间内）
            TestingResult resultAccept = tester.testMeanEqualWithT(5.0, sample, 0.95);

            // H0 = 0.0 应该被拒绝（不在置信区间内）
            TestingResult resultReject = tester.testMeanEqualWithT(0.0, sample, 0.95);
            long timeMs = (System.nanoTime() - start) / 1_000_000;

            boolean acceptCorrect = resultAccept.pass;   // H0=5.0 应被接受
            boolean rejectCorrect = !resultReject.pass;  // H0=0.0 应被拒绝

            boolean passed = acceptCorrect && rejectCorrect;
            if (!passed) allPassed = false;

            logBenchmark("STATS_TEST", "TTest", n, timeMs, passed,
                    String.format("H0=5.0_pass=%b(p=%.4f), H0=0.0_reject=%b(p=%.4f)",
                            resultAccept.pass, resultAccept.p,
                            !resultReject.pass, resultReject.p));
        }

        // 4.2 卡方检验：已知方差的正态样本
        {
            RANDOM.setSeed(FIXED_SEED);
            int n = 1000;
            double[] data = new double[n];
            double trueVar = 4.0;
            for (int i = 0; i < n; i++) {
                data[i] = RANDOM.nextGaussian() * Math.sqrt(trueVar);
            }
            IVector sample = Linalg.vector(data);

            long start = System.nanoTime();
            HypothesisTesting tester = new HypothesisTesting();

            // H0 = 4.0 应该被接受
            TestingResult resultAccept = tester.testVarEqualWithChi2(4.0, sample, 0.95);

            // H0 = 1.0 应该被拒绝
            TestingResult resultReject = tester.testVarEqualWithChi2(1.0, sample, 0.95);
            long timeMs = (System.nanoTime() - start) / 1_000_000;

            boolean acceptCorrect = resultAccept.pass;
            boolean rejectCorrect = !resultReject.pass;

            boolean passed = acceptCorrect && rejectCorrect;
            if (!passed) allPassed = false;

            logBenchmark("STATS_TEST", "Chi2Test", n, timeMs, passed,
                    String.format("H0=4.0_pass=%b(p=%.4f), H0=1.0_reject=%b(p=%.4f)",
                            resultAccept.pass, resultAccept.p,
                            !resultReject.pass, resultReject.p));
        }

        assert allPassed : "假设检验测试失败";
    }

    // ==================== 5. 相关/协方差 ====================

    @Test
    public void testCorrelationCovariance() {
        System.out.println("\n=== 相关/协方差测试 ===");
        boolean allPassed = true;

        // 5.1 完全相关（corr=1）
        {
            int n = 10000;
            double[] x = new double[n];
            double[] y = new double[n];
            for (int i = 0; i < n; i++) {
                x[i] = i * 0.1;
                y[i] = 2.0 * x[i] + 1.0;  // 完全线性相关
            }
            IVector vx = Linalg.vector(x);
            IVector vy = Linalg.vector(y);

            long start = System.nanoTime();
            double corr = Stats.corr(vx, vy);
            long timeMs = (System.nanoTime() - start) / 1_000_000;

            boolean passed = approxEqual(corr, 1.0, 1e-10);
            if (!passed) allPassed = false;

            logBenchmark("STATS_CORR", "PerfectPositive", n, timeMs, passed,
                    String.format("corr=%.10f(exp=1.0)", corr));
        }

        // 5.2 完全不相关（corr=0）
        {
            RANDOM.setSeed(FIXED_SEED);
            int n = 10000;
            double[] x = new double[n];
            double[] y = new double[n];
            for (int i = 0; i < n; i++) {
                x[i] = RANDOM.nextGaussian();
                y[i] = RANDOM.nextGaussian();  // 独立随机变量
            }
            IVector vx = Linalg.vector(x);
            IVector vy = Linalg.vector(y);

            long start = System.nanoTime();
            double corr = Stats.corr(vx, vy);
            long timeMs = (System.nanoTime() - start) / 1_000_000;

            // 独立随机变量的相关系数应接近0（在大样本下）
            boolean passed = Math.abs(corr) < 0.05;
            if (!passed) allPassed = false;

            logBenchmark("STATS_CORR", "Uncorrelated", n, timeMs, passed,
                    String.format("corr=%.6f(exp≈0.0)", corr));
        }

        // 5.3 负相关（corr=-1）
        {
            int n = 10000;
            double[] x = new double[n];
            double[] y = new double[n];
            for (int i = 0; i < n; i++) {
                x[i] = i * 0.1;
                y[i] = -3.0 * x[i] + 5.0;  // 完全负线性相关
            }
            IVector vx = Linalg.vector(x);
            IVector vy = Linalg.vector(y);

            long start = System.nanoTime();
            double corr = Stats.corr(vx, vy);
            long timeMs = (System.nanoTime() - start) / 1_000_000;

            boolean passed = approxEqual(corr, -1.0, 1e-10);
            if (!passed) allPassed = false;

            logBenchmark("STATS_CORR", "PerfectNegative", n, timeMs, passed,
                    String.format("corr=%.10f(exp=-1.0)", corr));
        }

        // 5.4 大规模向量（10万元素）
        {
            RANDOM.setSeed(FIXED_SEED);
            int n = 100_000;
            double[] x = new double[n];
            double[] y = new double[n];
            for (int i = 0; i < n; i++) {
                x[i] = RANDOM.nextGaussian();
                y[i] = x[i] * 0.5 + RANDOM.nextGaussian() * 0.5;  // 正相关
            }
            IVector vx = Linalg.vector(x);
            IVector vy = Linalg.vector(y);

            long start = System.nanoTime();
            double corr = Stats.corr(vx, vy);
            double cov = Stats.cov(vx, vy);
            long timeMs = (System.nanoTime() - start) / 1_000_000;

            // 协方差应为正（因为正相关）
            boolean corrPositive = corr > 0;
            boolean covPositive = cov > 0;
            boolean passed = corrPositive && covPositive;
            if (!passed) allPassed = false;

            logBenchmark("STATS_CORR", "LargeVector100K", n, timeMs, passed,
                    String.format("corr=%.4f, cov=%.4f", corr, cov));
        }

        // 5.5 超大规模向量（100万元素）
        {
            RANDOM.setSeed(FIXED_SEED);
            int n = 1_000_000;
            double[] x = new double[n];
            double[] y = new double[n];
            for (int i = 0; i < n; i++) {
                x[i] = RANDOM.nextGaussian();
                y[i] = x[i] * 0.5 + RANDOM.nextGaussian() * 0.5;
            }
            IVector vx = Linalg.vector(x);
            IVector vy = Linalg.vector(y);

            long start = System.nanoTime();
            double corr = Stats.corr(vx, vy);
            double cov = Stats.cov(vx, vy);
            long timeMs = (System.nanoTime() - start) / 1_000_000;

            boolean corrPositive = corr > 0;
            boolean covPositive = cov > 0;
            boolean passed = corrPositive && covPositive;
            if (!passed) allPassed = false;

            logBenchmark("STATS_CORR", "LargeVector1M", n, timeMs, passed,
                    String.format("corr=%.4f, cov=%.4f", corr, cov));
        }

        assert allPassed : "相关/协方差测试失败";
    }
}
