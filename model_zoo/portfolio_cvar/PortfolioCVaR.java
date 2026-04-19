package model_zoo.portfolio_cvar;

import com.yishape.lab.math.linalg.*;
import com.yishape.lab.math.stats.distribution.*;
import com.yishape.lab.math.optimize.*;
import com.yishape.lab.math.viz.*;

/**
 * 投资组合优化：均值-CVaR 模型
 *
 * 本示例展示如何在 YiShape Math 中实现均值-CVaR（Conditional Value at Risk）投资组合优化。
 *
 * 背景：
 * - Markowitz 均值-方差模型以方差为风险度量，是诺贝尔奖级别的经典理论
 * - 但方差作为风险度量存在缺陷：它把收益上行波动也视为"风险"
 * - CVaR（条件在险价值）是更合理的风险度量：只关注极端损失
 * - CVaR_α(w) = E[loss | loss > VaR_α]，即超过 VaR 的条件均值损失
 *
 * 流程：
 * 历史收益率 → 多元正态分布参数估计 → Monte Carlo 模拟 → CVaR 优化 → 最优配置
 */
public class PortfolioCVaR {

    // === 全局配置 ===
    private static final int NUM_ASSETS = 5;           // 资产数量
    private static final int HISTORICAL_PERIODS = 252; // 历史数据天数（约 1 年交易日）
    private static final int MONTE_CARLO_SIMS = 5000;  // Monte Carlo 模拟次数
    private static final int RANDOM_SEED = 42;          // 随机种子
    private static final double CONFIDENCE_LEVEL = 0.95; // 95% 置信水平（α = 0.95）

    // 资产名称
    private static final String[] ASSET_NAMES = {
        "股票A", "股票B", "债券C", "股票D", "黄金E"
    };

    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("   投资组合优化：均值-CVaR 模型");
        System.out.println("=".repeat(60));
        System.out.println();

        // === Step 1: 生成模拟历史收益率数据 ===
        System.out.println(">>> Step 1: 生成模拟历史收益率数据...");
        Linalg.setRandomSeed(RANDOM_SEED);

        // 各资产的预期年化收益率和波动率
        double[] expectedReturns = {0.08, 0.12, 0.03, 0.15, 0.05};
        double[] volatilities = {0.20, 0.30, 0.05, 0.35, 0.15};

        // 相关性矩阵（股债低相关、黄金与股市低/负相关）
        double[][] corrData = {
            {1.00,  0.60, -0.10,  0.50,  0.05},
            {0.60,  1.00,  0.05,  0.70,  0.10},
            {-0.10, 0.05,  1.00, -0.05,  0.20},
            {0.50,  0.70, -0.05,  1.00,  0.08},
            {0.05,  0.10,  0.20,  0.08,  1.00}
        };
        IMatrix<Double> correlationMatrix = Linalg.matrix(corrData);

        // 生成多资产相关联收益率序列
        IMatrix<Double> historicalReturns = generateCorrelatedReturns(
            expectedReturns, volatilities, correlationMatrix, HISTORICAL_PERIODS);

        System.out.println("   历史数据: " + HISTORICAL_PERIODS + " 天 × " + NUM_ASSETS + " 资产");
        System.out.println("   资产: " + String.join(", ", ASSET_NAMES));
        System.out.println();

        // === Step 2: 估计多元正态分布参数 ===
        System.out.println(">>> Step 2: 估计多元正态分布参数...");
        IVector<Double> sampleMean = computeColumnMeans(historicalReturns);
        System.out.println("   样本均值（日收益率，年化为 ×252）：");
        for (int i = 0; i < NUM_ASSETS; i++) {
            System.out.printf("   - %s: %.4f%%（年化: %.2f%%）%n",
                ASSET_NAMES[i], sampleMean.get(i) * 100, sampleMean.get(i) * 252 * 100);
        }
        IMatrix<Double> sampleCov = computeSampleCovariance(historicalReturns, sampleMean);
        System.out.printf("   协方差矩阵维度: %d × %d%n", sampleCov.rows(), sampleCov.cols());
        System.out.println();

        // === Step 3: Monte Carlo 模拟未来收益场景 ===
        System.out.println(">>> Step 3: Monte Carlo 模拟未来收益场景...");
        MultivariateNormalDistribution multiNorm =
            MultivariateDistributions.normal(sampleMean, sampleCov);

        double[][] mcReturns = new double[MONTE_CARLO_SIMS][NUM_ASSETS];
        for (int i = 0; i < MONTE_CARLO_SIMS; i++) {
            mcReturns[i] = multiNorm.sample(1)[0];
        }
        IMatrix<Double> mcReturnMatrix = Linalg.matrix(mcReturns);
        System.out.println("   Monte Carlo 场景: " + MONTE_CARLO_SIMS + " 个场景");
        System.out.println();

        // === Step 4: 求解均值-CVaR 优化 ===
        System.out.println(">>> Step 4: 求解均值-CVaR 优化...");

        // 设置最低目标收益（取历史等权组合收益的 80%）
        IVector<Double> meanReturns = computeColumnMeans(mcReturnMatrix);
        IVector<Double> equalW = Linalg.vector(
            java.util.stream.IntStream.range(0, NUM_ASSETS)
                .mapToDouble(i -> 1.0 / NUM_ASSETS).toArray());
        double benchReturn = 0.0;
        for (int i = 0; i < NUM_ASSETS; i++) {
            benchReturn += equalW.get(i) * meanReturns.get(i);
        }
        double minReturn = benchReturn * 0.8;
        System.out.printf("   最低目标收益: %.4f%%（日）/ %.2f%%（年化）%n",
            minReturn * 100, minReturn * 252 * 100);

        // 求解 CVaR 优化
        double gamma = 0.95;
        CVaROptimizer optimizer = new CVaROptimizer(mcReturnMatrix, gamma, minReturn, RANDOM_SEED);
        OptResult result = optimizer.solve();

        IVector<Double> optimalWeights = result.getOptimalPoint();
        double optimalCVaR = -result.getOptimalValue();

        System.out.println("   最优 CVaR: " + String.format("%.4f%%", optimalCVaR * 100));
        System.out.println("   最优资产配置:");
        for (int i = 0; i < NUM_ASSETS; i++) {
            System.out.printf("   - %s: %.2f%%%n", ASSET_NAMES[i], optimalWeights.get(i) * 100);
        }
        System.out.println();

        // 计算组合统计量
        double portfolioReturn = 0.0;
        double portfolioVariance = 0.0;
        for (int i = 0; i < NUM_ASSETS; i++) {
            portfolioReturn += optimalWeights.get(i) * meanReturns.get(i);
            for (int j = 0; j < NUM_ASSETS; j++) {
                portfolioVariance += optimalWeights.get(i) * optimalWeights.get(j) *
                    sampleCov.get(i, j);
            }
        }
        double portfolioVol = Math.sqrt(Math.abs(portfolioVariance));
        System.out.printf("   最优组合预期收益: %.4f%%（日）/ %.2f%%（年化）%n",
            portfolioReturn * 100, portfolioReturn * 252 * 100);
        System.out.printf("   最优组合波动率: %.4f%%（日）/ %.2f%%（年化）%n",
            portfolioVol * 100, portfolioVol * Math.sqrt(252) * 100);
        System.out.printf("   夏普比率（无风险利率=0）: %.3f%n",
            (portfolioReturn * 252) / (portfolioVol * Math.sqrt(252)));
        System.out.println();

        // === Step 5: 有效前沿 ===
        System.out.println(">>> Step 5: 生成有效前沿...");
        int frontierPoints = 10;
        double maxReturn = meanReturns.max();
        double[] frontierReturns = new double[frontierPoints];
        double[] frontierCVaRs = new double[frontierPoints];

        System.out.println("   最低收益约束 → 组合收益 → CVaR");
        System.out.println("   " + "-".repeat(48));
        for (int k = 0; k < frontierPoints; k++) {
            double targetReturn = minReturn + (maxReturn - minReturn) * k / (frontierPoints - 1);
            CVaROptimizer opt = new CVaROptimizer(mcReturnMatrix, gamma, targetReturn, RANDOM_SEED);
            OptResult r = opt.solve();
            IVector<Double> w = r.getOptimalPoint();

            double portR = 0.0;
            for (int i = 0; i < NUM_ASSETS; i++) {
                portR += w.get(i) * meanReturns.get(i);
            }
            frontierReturns[k] = portR;
            frontierCVaRs[k] = -r.getOptimalValue();
            System.out.printf("   %.4f%% → %.4f%%（日）/ %.2f%%（年化）| CVaR %.4f%%%n",
                targetReturn * 100, portR * 100, portR * 252 * 100, frontierCVaRs[k] * 100);
        }
        System.out.println();

        // === Step 6: 可视化 ===
        System.out.println(">>> Step 6: 生成可视化...");
        visualizeEfficientFrontier(frontierReturns, frontierCVaRs);
        visualizeAssetAllocation(optimalWeights);
        System.out.println();

        // === Step 7: 与等权组合对比 ===
        System.out.println(">>> Step 7: 与等权组合（1/N）对比...");
        double ewReturn = 0.0;
        double[] ewLosses = new double[MONTE_CARLO_SIMS];
        for (int i = 0; i < MONTE_CARLO_SIMS; i++) {
            double portR = 0.0;
            for (int j = 0; j < NUM_ASSETS; j++) {
                portR += equalW.get(j) * mcReturnMatrix.get(i, j);
            }
            ewReturn += portR;
            ewLosses[i] = -portR;
        }
        ewReturn /= MONTE_CARLO_SIMS;
        java.util.Arrays.sort(ewLosses);
        int varIdx = (int) ((1 - gamma) * MONTE_CARLO_SIMS);
        double ewVaR = ewLosses[varIdx];
        double ewCVaR = 0.0;
        for (int i = varIdx; i < MONTE_CARLO_SIMS; i++) {
            ewCVaR += ewLosses[i];
        }
        ewCVaR /= (MONTE_CARLO_SIMS - varIdx);

        System.out.printf("   等权组合预期收益: %.2f%%（年化）%n", ewReturn * 252 * 100);
        System.out.printf("   等权组合 CVaR: %.4f%%%n", ewCVaR * 100);
        System.out.printf("   CVaR 改善: %.2f%%（CVaR 优化组合更低 = 更保守）%n",
            (ewCVaR - optimalCVaR) * 100);
        System.out.println();

        System.out.println("=".repeat(60));
        System.out.println("   投资组合均值-CVaR 优化完成！");
        System.out.println("=".repeat(60));
    }

    // === 生成相关联的多资产收益率 ===
    private static IMatrix<Double> generateCorrelatedReturns(
            double[] expectedReturns, double[] volatilities,
            IMatrix<Double> correlationMatrix, int periods) {

        // Cholesky 分解：L * L^T = correlationMatrix
        IMatrix<Double> L = choleskyDecomposition(correlationMatrix);

        double[][] result = new double[periods][NUM_ASSETS];
        for (int t = 0; t < periods; t++) {
            // 独立标准正态随机向量
            double[] z = new double[NUM_ASSETS];
            for (int i = 0; i < NUM_ASSETS; i++) {
                z[i] = RereMathUtil.normalSample(0.0, 1.0);
            }

            // 变换为相关联的正态变量
            double[] correlatedZ = new double[NUM_ASSETS];
            for (int i = 0; i < NUM_ASSETS; i++) {
                correlatedZ[i] = 0;
                for (int j = 0; j < NUM_ASSETS; j++) {
                    correlatedZ[i] += L.get(i, j) * z[j];
                }
            }

            // 转换为日收益率（几何布朗运动近似）
            for (int i = 0; i < NUM_ASSETS; i++) {
                double dailyReturn = expectedReturns[i] / 252 +
                    volatilities[i] / Math.sqrt(252) * correlatedZ[i];
                result[t][i] = dailyReturn;
            }
        }
        return Linalg.matrix(result);
    }

    // === Cholesky 分解 ===
    private static IMatrix<Double> choleskyDecomposition(IMatrix<Double> A) {
        int n = A.rows();
        double[][] L = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double sum = 0.0;
                if (j == i) {
                    for (int k = 0; k < j; k++) {
                        sum += L[j][k] * L[j][k];
                    }
                    L[j][j] = Math.sqrt(Math.max(A.get(j, j) - sum, 0));
                } else {
                    for (int k = 0; k < j; k++) {
                        sum += L[i][k] * L[j][k];
                    }
                    L[i][j] = (A.get(i, j) - sum) / L[j][j];
                }
            }
        }
        return Linalg.matrix(L);
    }

    // === 计算矩阵各列均值 ===
    private static IVector<Double> computeColumnMeans(IMatrix<Double> matrix) {
        int rows = matrix.rows();
        int cols = matrix.cols();
        double[] means = new double[cols];
        for (int j = 0; j < cols; j++) {
            double sum = 0.0;
            for (int i = 0; i < rows; i++) {
                sum += matrix.get(i, j);
            }
            means[j] = sum / rows;
        }
        return Linalg.vector(means);
    }

    // === 计算样本协方差矩阵 ===
    private static IMatrix<Double> computeSampleCovariance(IMatrix<Double> matrix, IVector<Double> means) {
        int rows = matrix.rows();
        int cols = matrix.cols();
        double[][] cov = new double[cols][cols];
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < cols; j++) {
                double sum = 0.0;
                for (int k = 0; k < rows; k++) {
                    sum += (matrix.get(k, i) - means.get(i)) * (matrix.get(k, j) - means.get(j));
                }
                cov[i][j] = sum / (rows - 1);
            }
        }
        return Linalg.matrix(cov);
    }

    // === 可视化：有效前沿 ===
    private static void visualizeEfficientFrontier(double[] returns, double[] cvaRs) {
        IVector<Double> ret = Linalg.vector(returns).map(v -> v * 252 * 100);
        IVector<Double> cvar = Linalg.vector(cvaRs).map(v -> v * 100);

        RerePlot plt = Plots.of(800, 500);
        plt.setTitle("均值-CVaR 有效前沿 / Mean-CVaR Efficient Frontier");
        plt.setXLabel("CVaR (%, 年化) / CVaR (%, annualized)");
        plt.setYLabel("预期收益 (%, 年化) / Expected Return (%, annualized)");
        plt.line(cvar, ret).show();
        plt.saveAsHtml("portfolio_efficient_frontier.html");
        System.out.println("   图表已保存: portfolio_efficient_frontier.html");
    }

    // === 可视化：最优资产配置 ===
    private static void visualizeAssetAllocation(IVector<Double> weights) {
        RerePlot plt = Plots.of(600, 400);
        plt.setTitle("最优资产配置 / Optimal Asset Allocation");
        plt.setYLabel("权重 / Weight (%)");
        plt.bar(weights.map(v -> v * 100), java.util.Arrays.asList(ASSET_NAMES)).show();
        plt.saveAsHtml("portfolio_allocation.html");
        System.out.println("   图表已保存: portfolio_allocation.html");
    }

    // ============================================================
    // 内部类：CVaR 优化器（惩罚函数法 + 投影梯度法）
    // ============================================================
    static class CVaROptimizer {
        private final IMatrix<Double> returnMatrix;
        private final double alpha;       // 置信水平（0.95）
        private final double minReturn;   // 最低目标收益
        private final int M;              // 场景数
        private final int N;              // 资产数
        private final double[][] portReturns; // 预计算的组合收益率
        private final int varIndex;       // VaR 分位数索引
        private final double scale;        // CVaR 缩放因子

        CVaROptimizer(IMatrix<Double> returnMatrix, double alpha,
                       double minReturn, int randomSeed) {
            this.returnMatrix = returnMatrix;
            this.alpha = alpha;
            this.minReturn = minReturn;
            this.M = returnMatrix.rows();
            this.N = returnMatrix.cols();
            this.scale = 1.0 / (M * (1 - alpha));
            this.varIndex = (int) Math.floor((1 - alpha) * M);

            // 预计算：每个场景下各资产的收益率（避免重复计算）
            this.portReturns = new double[M][N];
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    portReturns[m][n] = returnMatrix.get(m, n);
                }
            }
        }

        OptResult solve() {
            // 使用 RereLBFGS + 投影到单纯形（处理 w>=0, Σw=1）
            RereLBFGS optimizer = new RereLBFGS();

            double bestCvar = Double.MAX_VALUE;
            IVector<Double> bestW = null;

            // 多次随机重启，选择最好结果
            for (int restart = 0; restart < 5; restart++) {
                java.util.Random rand = new java.util.Random(42 + restart * 17);
                double[] init = new double[N];
                for (int n = 0; n < N; n++) {
                    init[n] = Math.abs(rand.nextGaussian());
                }
                double sumInit = Linalg.vector(init).sum();
                for (int n = 0; n < N; n++) {
                    init[n] /= sumInit;
                }
                IVector<Double> w0 = Linalg.vector(init);

                IVector<Double> w = w0;
                for (int iter = 0; iter < 100; iter++) {
                    // 计算梯度
                    double[] grad = computeNumericalGradient(w);

                    // L-BFGS 步
                    IVector<Double> gradV = Linalg.vector(grad);
                    OptResult optResult = optimizer.optimize(w,
                        (IObjectiveFunction) x -> computeCVaR(x),
                        (IGradientFunction) x -> gradV);

                    w = projectToSimplex(optResult.getOptimalPoint());

                    // 检查收益约束
                    double portR = portMean(w);
                    if (portR >= minReturn) {
                        double cvar = computeCVaR(w);
                        if (cvar < bestCvar) {
                            bestCvar = cvar;
                            bestW = w;
                        }
                        break;
                    }
                }
            }

            if (bestW == null) {
                // Fallback: 等权配置
                bestW = Linalg.vector(
                    java.util.stream.IntStream.range(0, N)
                        .mapToDouble(i -> 1.0 / N).toArray());
                bestCvar = computeCVaR(bestW);
            }

            return new OptResult(-bestCvar, bestW);
        }

        // === 计算 CVaR ===
        private double computeCVaR(IVector x) {
            // 各场景组合收益率
            double[] r = new double[M];
            for (int m = 0; m < M; m++) {
                r[m] = 0.0;
                for (int n = 0; n < N; n++) {
                    r[m] += portReturns[m][n] * x.get(n).doubleValue();
                }
            }

            // 排序找 VaR（损失视角）
            double[] sortedR = r.clone();
            java.util.Arrays.sort(sortedR);
            double var = -sortedR[varIndex]; // VaR = -sortedR[varIndex]

            // CVaR = VaR + (1/(M(1-α))) * Σ_{r_m < -VaR} (-r_m - VaR)
            double cvar = var;
            for (int m = 0; m < M; m++) {
                double loss = Math.max(-r[m] - var, 0.0);
                cvar += scale * loss;
            }
            return cvar;
        }

        // === 计算数值梯度 ===
        private double[] computeNumericalGradient(IVector x) {
            double eps = 1e-7;
            double f0 = computeCVaR(x);
            double[] grad = new double[N];
            for (int n = 0; n < N; n++) {
                double[] xp = x.toDoubleArray();
                xp[n] += eps;
                IVector<Double> xn = Linalg.vector(xp);
                double fp = computeCVaR(xn);
                grad[n] = (fp - f0) / eps;
            }
            return grad;
        }

        // === 组合收益均值 ===
        private double portMean(IVector w) {
            double sum = 0.0;
            for (int m = 0; m < M; m++) {
                double r = 0.0;
                for (int n = 0; n < N; n++) {
                    r += portReturns[m][n] * w.get(n);
                }
                sum += r;
            }
            return sum / M;
        }

        // === 投影到单纯形: w >= 0, Σw = 1 ===
        // Duchi et al. (2008) 的投影算法，O(N log N)
        private IVector<Double> projectToSimplex(IVector<Double> w) {
            double[] sorted = w.toDoubleArray();
            java.util.Arrays.sort(sorted);
            // 降序
            double[] desc = new double[N];
            for (int i = 0; i < N; i++) {
                desc[i] = sorted[N - 1 - i];
            }

            double sum = 0.0;
            int rho = 0;
            for (int i = 0; i < N; i++) {
                sum += desc[i];
                double t = (sum - 1.0) / (i + 1);
                if (desc[i] - t > 0) {
                    rho = i + 1;
                }
            }
            double theta = (sum - 1.0) / rho;

            double[] proj = new double[N];
            for (int i = 0; i < N; i++) {
                proj[i] = Math.max(w.get(i).doubleValue() - theta, 0.0);
            }
            return Linalg.vector(proj);
        }
    }
}
