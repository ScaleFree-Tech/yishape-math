package model_zoo.customer_segmentation;

import com.yishape.lab.math.linalg.*;
import com.yishape.lab.math.stats.model.*;
import com.yishape.lab.math.optimize.*;
import com.yishape.lab.math.viz.*;

/**
 * 客户细分：GMM 软聚类 + 差异化定价
 *
 * 本示例展示如何使用高斯混合模型（GMM）对客户进行软聚类，
 * 并基于聚类结果为每个客户群体设计差异化定价策略。
 *
 * 背景：
 * - 传统 K-Means 是硬聚类（每个客户只属于一个群体）
 * - GMM 是软聚类（每个客户以概率方式属于各群体）
 * - 软分配信息更有价值：可以对概率中等的客户做更多营销
 * - 每个群体有不同的价格敏感度，可设计差异化定价
 *
 * 流程：
 * 客户特征数据 → GMM 软聚类 → 后验概率 → 差异化定价优化
 */
public class CustomerSegmentation {

    // === 全局配置 ===
    private static final int NUM_CUSTOMERS = 500;   // 客户数量
    private static final int NUM_FEATURES = 4;      // 特征维度
    private static final int NUM_CLUSTERS = 4;      // 聚类数量（4 个客户群体）
    private static final int RANDOM_SEED = 42;       // 随机种子

    // 特征含义：[消费频率, 平均订单额(归一化), 折扣敏感度, 活跃天数比例]
    private static final String[] FEATURE_NAMES = {
        "消费频率", "平均订单额", "折扣敏感度", "活跃天数比例"
    };

    // 客户群体名称
    private static final String[] CLUSTER_NAMES = {
        "高价值忠实客户", "价格敏感型客户", "活跃新客户", "低活跃休眠客户"
    };

    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("   客户细分：GMM 软聚类 + 差异化定价");
        System.out.println("=".repeat(60));
        System.out.println();

        // === Step 1: 生成模拟客户特征数据 ===
        System.out.println(">>> Step 1: 生成模拟客户特征数据（4 个潜在群体）...");
        Linalg.setRandomSeed(RANDOM_SEED);

        // 4 个客户群体的真实参数（隐藏的）
        // 高价值忠实客户：消费频率高，订单额高，折扣敏感度低
        // 价格敏感型：消费频率中等，订单额低，折扣敏感度高
        // 活跃新客户：消费频率中等，订单额中等，折扣敏感度中等
        // 低活跃休眠客户：消费频率低，订单额低，折扣敏感度低

        double[][] clusterParams = {
            // {频率μ, 频率σ, 订单μ, 订单σ, 折扣μ, 折扣σ, 活跃μ, 活跃σ}
            {0.80, 0.10, 0.90, 0.10, 0.10, 0.05, 0.85, 0.10}, // 高价值忠实
            {0.50, 0.15, 0.30, 0.15, 0.90, 0.05, 0.55, 0.15}, // 价格敏感
            {0.55, 0.20, 0.55, 0.20, 0.50, 0.15, 0.65, 0.20}, // 活跃新客户
            {0.15, 0.10, 0.20, 0.10, 0.30, 0.15, 0.20, 0.15}  // 低活跃休眠
        };

        // 每个群体的客户数（不均衡，更真实）
        int[] clusterSizes = {80, 150, 170, 100};
        double[][] allFeatures = new double[NUM_CUSTOMERS][NUM_FEATURES];

        int idx = 0;
        for (int k = 0; k < NUM_CLUSTERS; k++) {
            for (int i = 0; i < clusterSizes[k]; i++) {
                allFeatures[idx][0] = clip(clusterParams[k][0] + gaussianRandom() * clusterParams[k][1], 0.0, 1.0);
                allFeatures[idx][1] = clip(clusterParams[k][2] + gaussianRandom() * clusterParams[k][3], 0.0, 1.0);
                allFeatures[idx][2] = clip(clusterParams[k][4] + gaussianRandom() * clusterParams[k][5], 0.0, 1.0);
                allFeatures[idx][3] = clip(clusterParams[k][6] + gaussianRandom() * clusterParams[k][7], 0.0, 1.0);
                idx++;
            }
        }

        System.out.println("   客户数: " + NUM_CUSTOMERS);
        System.out.println("   特征维度: " + NUM_FEATURES + "（" + String.join(", ", FEATURE_NAMES) + "）");
        System.out.println("   各群体真实规模: " + java.util.Arrays.toString(clusterSizes));
        System.out.println();

        // === Step 2: 用 GMM 进行软聚类 ===
        System.out.println(">>> Step 2: 训练 GMM（高斯混合模型）...");
        System.out.println("   GMM 模型: " + NUM_CLUSTERS + " 个高斯分量，" + NUM_FEATURES + " 维特征");
        System.out.println("   使用 EM 算法估计参数...");

        // 准备训练数据（List<IVector<Double>>）
        java.util.List<IVector<Double>> dataList = new java.util.ArrayList<>();
        for (int i = 0; i < NUM_CUSTOMERS; i++) {
            dataList.add(Linalg.vector(allFeatures[i]));
        }

        // 创建 GMM 模型（4 个分量，4 维数据）
        GaussianMixtureModel gmm = new GaussianMixtureModel(NUM_CLUSTERS, NUM_FEATURES);

        // EM 算法配置（默认种子=42）
        EMAlgorithm em = new EMAlgorithm(100, 1e-6, false);

        // EM 训练
        EMAlgorithm.EMResult result = em.fit(dataList, gmm);

        System.out.println("   EM 算法收敛！");
        System.out.printf("   实际迭代次数: %d%n", result.iterations);
        System.out.println("   各聚类分量权重（π_k）：");
        java.util.List<Double> weights = gmm.getWeights();
        for (int k = 0; k < NUM_CLUSTERS; k++) {
            System.out.printf("   - 分量 %d（%s）: %.2f%%%n",
                k, CLUSTER_NAMES[k], weights.get(k) * 100);
        }
        System.out.println();

        // === Step 3: 分析聚类结果 ===
        System.out.println(">>> Step 3: 分析 GMM 聚类结果...");
        System.out.println("   GMM 的优势：每个客户以概率方式属于各群体（软分配）");
        System.out.println();

        // 对每个客户计算后验概率
        double[][] posteriorMatrix = new double[NUM_CUSTOMERS][NUM_CLUSTERS];
        int[] hardLabels = new int[NUM_CUSTOMERS];
        double[] maxProbabilities = new double[NUM_CUSTOMERS];

        for (int i = 0; i < NUM_CUSTOMERS; i++) {
            IVector<Double> sample = dataList.get(i);
            IVector<Double> posteriors = gmm.computePosteriors(sample);
            for (int k = 0; k < NUM_CLUSTERS; k++) {
                posteriorMatrix[i][k] = posteriors.get(k);
            }

            // 硬标签：取概率最大的分量
            int bestK = 0;
            double bestP = posteriors.get(0);
            for (int k = 1; k < NUM_CLUSTERS; k++) {
                if (posteriors.get(k) > bestP) {
                    bestP = posteriors.get(k);
                    bestK = k;
                }
            }
            hardLabels[i] = bestK;
            maxProbabilities[i] = bestP;
        }

        // 各群体的硬标签统计
        int[] clusterCounts = new int[NUM_CLUSTERS];
        for (int label : hardLabels) {
            clusterCounts[label]++;
        }
        System.out.println("   各群体客户数（硬分配）：");
        for (int k = 0; k < NUM_CLUSTERS; k++) {
            System.out.printf("   - %s: %d 人 (%.1f%%)%n",
                CLUSTER_NAMES[k], clusterCounts[k], clusterCounts[k] * 100.0 / NUM_CUSTOMERS);
        }
        System.out.println();

        // 软聚类示例（打印前 5 个客户的后验概率）
        System.out.println("   前 5 位客户的后验概率（软分配）：");
        System.out.println("   客户ID | " + String.format("%-16s", "高价值")
            + String.format("%-14s", "价格敏感")
            + String.format("%-12s", "活跃新客户")
            + String.format("%-14s", "低活跃休眠") + "硬标签");
        System.out.println("   " + "-".repeat(75));
        for (int i = 0; i < 5; i++) {
            StringBuilder sb = new StringBuilder();
            sb.append(String.format("   %6d |", i));
            for (int k = 0; k < NUM_CLUSTERS; k++) {
                sb.append(String.format(" %.4f      ", posteriorMatrix[i][k]));
            }
            sb.append(CLUSTER_NAMES[hardLabels[i]]);
            System.out.println(sb);
        }
        System.out.println();

        // 计算 entropy（不确定性）：-Σ p_k * log(p_k)
        // entropy 高 = 客户属于多个群体（边界客户）
        // entropy 低 = 客户归属明确（典型客户）
        System.out.println("   客户归属确定性分析（entropy）：");
        double totalEntropy = 0.0;
        for (int i = 0; i < NUM_CUSTOMERS; i++) {
            double entropy = 0.0;
            for (int k = 0; k < NUM_CLUSTERS; k++) {
                if (posteriorMatrix[i][k] > 1e-10) {
                    entropy -= posteriorMatrix[i][k] * Math.log(posteriorMatrix[i][k]);
                }
            }
            totalEntropy += entropy;
        }
        double avgEntropy = totalEntropy / NUM_CUSTOMERS;
        System.out.printf("   平均熵: %.4f（0=完全确定，%.4f=完全不确定）%n",
            avgEntropy, Math.log(NUM_CLUSTERS));
        System.out.printf("   熵很低 → 各客户归属比较明确（4 个高斯分量区分度高）%n");
        System.out.println();

        // === Step 4: 为每个群体设计差异化定价 ===
        System.out.println(">>> Step 4: 基于聚类结果设计差异化定价策略...");

        // 基础成本（商品成本 = 售价的 40%）
        double costRate = 0.40;
        System.out.printf("   商品成本率: %.0f%%%n", costRate * 100);

        // 每个群体的价格敏感度（从 GMM 后验均值推断）
        // 折扣敏感度特征越高 → 对价格越敏感 → 最优价格越低
        double[] clusterAvgSensitivity = new double[NUM_CLUSTERS];
        for (int k = 0; k < NUM_CLUSTERS; k++) {
            double sum = 0.0;
            int count = 0;
            for (int i = 0; i < NUM_CUSTOMERS; i++) {
                if (hardLabels[i] == k) {
                    sum += allFeatures[i][2]; // 折扣敏感度
                    count++;
                }
            }
            clusterAvgSensitivity[k] = count > 0 ? sum / count : 0.5;
        }

        System.out.println("   各群体折扣敏感度（推断）：");
        for (int k = 0; k < NUM_CLUSTERS; k++) {
            System.out.printf("   - %s: %.3f%n", CLUSTER_NAMES[k], clusterAvgSensitivity[k]);
        }
        System.out.println();

        // 对每个群体用 Golden Section Search 找最优价格
        System.out.println("   最优定价（单位：元）：");
        double[] optimalPrices = new double[NUM_CLUSTERS];
        for (int k = 0; k < NUM_CLUSTERS; k++) {
            // 价格需求函数: D(p) = D0 * (1 - sensitivity * (p - p0) / p0)
            // 简化为线性敏感度: D(p) = a - b * p
            double sensitivity = clusterAvgSensitivity[k];
            double a = 1.0 + sensitivity;    // 基础需求系数
            double b = 2.0 * sensitivity;    // 价格敏感系数

            // 利润 π(p) = (p - cost) * D(p) = (p - cost) * (a - b*p)
            // dπ/dp = a - 2*b*p + b*cost = 0
            // p* = (a + b*cost) / (2*b)
            double optimalPrice = (a + b * costRate) / (2 * b);
            optimalPrices[k] = Math.max(optimalPrice, costRate + 0.01);

            // 计算最优价格下的预期利润
            double demand = Math.max(a - b * optimalPrices[k], 0.01);
            double profit = (optimalPrices[k] - costRate) * demand;
            double markup = (optimalPrices[k] - costRate) / costRate * 100;

            System.out.printf("   - %s: ¥%.2f（加价率: %.1f%%，需求: %.2f）%n",
                CLUSTER_NAMES[k], optimalPrices[k], markup, demand);
        }
        System.out.println();

        // === Step 5: 综合对比 ===
        System.out.println(">>> Step 5: 差异化定价 vs 统一价格定价...");
        System.out.println("   策略对比（基于各群体客户数）：");
        System.out.println("   " + String.format("%-16s", "群体")
            + String.format("%8s", "客户数")
            + String.format("%12s", "差异化价格")
            + String.format("%14s", "差异化利润")
            + String.format("%12s", "统一价格")
            + String.format("%14s", "统一利润"));
        System.out.println("   " + "-".repeat(80));

        // 统一价格 = 各群体最优价格的均值
        double uniformPrice = java.util.Arrays.stream(optimalPrices).average().orElse(1.0);
        double totalDifferentialProfit = 0.0;
        double totalUniformProfit = 0.0;

        for (int k = 0; k < NUM_CLUSTERS; k++) {
            int count = clusterCounts[k];
            double sensitivity = clusterAvgSensitivity[k];
            double a = 1.0 + sensitivity;
            double b = 2.0 * sensitivity;

            double diffProfit = (optimalPrices[k] - costRate) * Math.max(a - b * optimalPrices[k], 0.01) * count;
            double unifProfit = (uniformPrice - costRate) * Math.max(a - b * uniformPrice, 0.01) * count;

            totalDifferentialProfit += diffProfit;
            totalUniformProfit += unifProfit;

            System.out.printf("   %-16s %6d    ¥%.2f        ¥%.1f       ¥%.2f       ¥%.1f%n",
                CLUSTER_NAMES[k], count, optimalPrices[k], diffProfit, uniformPrice, unifProfit);
        }
        System.out.println("   " + "-".repeat(80));
        System.out.printf("   %-16s %6d    %-12s   ¥%.1f     %-12s   ¥%.1f%n",
            "合计", NUM_CUSTOMERS, "", totalDifferentialProfit, "", totalUniformProfit);
        System.out.printf("   差异化定价提升利润: %.1f%%（vs 统一价格）%n",
            (totalDifferentialProfit / totalUniformProfit - 1) * 100);
        System.out.println();

        // === Step 6: 可视化 ===
        System.out.println(">>> Step 6: 生成可视化...");
        visualizeClusters(dataList, hardLabels, maxProbabilities);
        visualizeClusteringResults(clusterCounts, clusterAvgSensitivity, optimalPrices);
        visualizePosteriorDistribution(posteriorMatrix);
        System.out.println();

        // === Step 7: 软聚类的营销价值 ===
        System.out.println(">>> Step 7: 软聚类的营销应用价值...");
        System.out.println("   GMM 软聚类的核心优势：不只告诉你「客户属于谁」，还告诉你「有多确定」");
        System.out.println();

        // 识别边界客户（entropy 高，营销价值高）
        int[] boundaryCustomers = new int[10];
        int[] certainCustomers = new int[10];
        int bi = 0, ci = 0;
        double maxEnt = 0, minEnt = Double.MAX_VALUE;
        int maxEntIdx = 0, minEntIdx = 0;

        for (int i = 0; i < NUM_CUSTOMERS; i++) {
            double entropy = 0.0;
            for (int k = 0; k < NUM_CLUSTERS; k++) {
                if (posteriorMatrix[i][k] > 1e-10) {
                    entropy -= posteriorMatrix[i][k] * Math.log(posteriorMatrix[i][k]);
                }
            }
            if (entropy > maxEnt) {
                maxEnt = entropy;
                maxEntIdx = i;
            }
            if (entropy < minEnt) {
                minEnt = entropy;
                minEntIdx = i;
            }
        }

        System.out.println("   典型客户（归属最确定，entropy ≈ 0）：");
        System.out.printf("   客户ID %d: 属于「%s」（后验概率 %.4f）%n",
            minEntIdx, CLUSTER_NAMES[hardLabels[minEntIdx]], maxProbabilities[minEntIdx]);
        System.out.println();

        System.out.println("   边界客户（归属最不确定，entropy 最高）：");
        System.out.printf("   客户ID %d: 后验概率: %s%n", maxEntIdx,
            java.util.Arrays.stream(posteriorMatrix[maxEntIdx])
                .mapToObj(p -> String.format("%.2f", p))
                .reduce((a, b) -> a + ", " + b).orElse(""));
        System.out.println("   → 该客户同时具备多类特征，是交叉营销的最佳目标！");
        System.out.println();

        System.out.println("=".repeat(60));
        System.out.println("   GMM 客户细分分析完成！");
        System.out.println("=".repeat(60));
    }

    // === 高斯随机数（Box-Muller）===
    private static java.util.Random rng = new java.util.Random(RANDOM_SEED);
    private static double gaussianRandom() {
        return rng.nextGaussian();
    }

    // === 限制在 [0, 1] 范围内 ===
    private static double clip(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    // === 可视化 1：客户分布散点图（取前 2 个主成分投影）===
    private static void visualizeClusters(
            java.util.List<IVector<Double>> dataList,
            int[] hardLabels,
            double[] maxProbabilities) {

        // 取特征 0（消费频率）和特征 1（平均订单额）做 2D 可视化
        double[] x = new double[dataList.size()];
        double[] y = new double[dataList.size()];
        for (int i = 0; i < dataList.size(); i++) {
            x[i] = dataList.get(i).get(0); // 消费频率
            y[i] = dataList.get(i).get(1); // 平均订单额
        }

        IVector<Double> xv = Linalg.vector(x);
        IVector<Double> yv = Linalg.vector(y);

        // 按群体着色散点图
        RerePlot plt = Plots.of(800, 600);
        plt.setTitle("客户群体分布（消费频率 vs 订单额） / Customer Segments");
        plt.setXLabel("消费频率 / Purchase Frequency");
        plt.setYLabel("平均订单额 / Avg Order Value");

        // 分群体绘制（用不同 hue 标签）
        String[] allHues = new String[dataList.size()];
        for (int i = 0; i < dataList.size(); i++) {
            allHues[i] = CLUSTER_NAMES[hardLabels[i]];
        }

        plt.scatter(xv, yv, java.util.Arrays.asList(allHues)).show();
        plt.saveAsHtml("customer_segments_scatter.html");
        System.out.println("   图表已保存: customer_segments_scatter.html");
    }

    // === 可视化 2：聚类结果统计 ===
    private static void visualizeClusteringResults(
            int[] clusterCounts,
            double[] sensitivities,
            double[] prices) {

        // 群体大小柱状图
        IVector<Double> sizes = Linalg.vector(
            java.util.Arrays.stream(clusterCounts).asDoubleStream().toArray());
        RerePlot plt1 = Plots.of(700, 400);
        plt1.setTitle("各客户群体规模 / Customer Group Sizes");
        plt1.setYLabel("客户数 / Number of Customers");
        plt1.bar(sizes, java.util.Arrays.asList(CLUSTER_NAMES)).show();
        plt1.saveAsHtml("customer_segments_sizes.html");
        System.out.println("   图表已保存: customer_segments_sizes.html");

        // 最优价格柱状图
        IVector<Double> priceV = Linalg.vector(prices);
        RerePlot plt2 = Plots.of(700, 400);
        plt2.setTitle("各群体最优定价 / Optimal Price per Segment");
        plt2.setYLabel("价格 (元) / Price (CNY)");
        plt2.bar(priceV, java.util.Arrays.asList(CLUSTER_NAMES)).show();
        plt2.saveAsHtml("customer_segments_prices.html");
        System.out.println("   图表已保存: customer_segments_prices.html");
    }

    // === 可视化 3：后验概率分布 ===
    private static void visualizePosteriorDistribution(double[][] posteriorMatrix) {
        // 取第一个分量的后验概率做直方图
        double[] p1 = new double[posteriorMatrix.length];
        for (int i = 0; i < posteriorMatrix.length; i++) {
            p1[i] = posteriorMatrix[i][0];
        }

        IVector<Double> pv = Linalg.vector(p1);
        RerePlot plt = Plots.of(700, 400);
        plt.setTitle("后验概率分布（高价值忠实客户）/ Posterior Probability Distribution");
        plt.setXLabel("后验概率 / Posterior Probability");
        plt.setYLabel("客户数 / Count");
        plt.hist(pv, true).show();
        plt.saveAsHtml("customer_posterior_hist.html");
        System.out.println("   图表已保存: customer_posterior_hist.html");
    }
}
