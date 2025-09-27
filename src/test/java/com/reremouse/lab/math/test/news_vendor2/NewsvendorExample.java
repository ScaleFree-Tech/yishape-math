package com.reremouse.lab.math.test.news_vendor2;

import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.stats.distribution.NormalDistribution;

/**
 * 报童模型完整示例
 * <p>
 * 这个示例展示了如何使用 yishape-math 库中的优化、线性代数和统计功能
 * 来求解经典的报童模型问题。
 * </p>
 * 
 * <h3>问题描述：</h3>
 * 一个报童每天需要决定订购多少份报纸。已知：
 * <ul>
 * <li>每份报纸的采购成本为 c</li>
 * <li>每份报纸的销售价格为 p</li>
 * <li>每份缺货的损失为 s</li>
 * <li>需求服从正态分布 N(μ, σ²)</li>
 * </ul>
 * 
 * <h3>目标：</h3>
 * 找到最优订货量 Q*，使得期望利润最大化。
 * 
 * <h3>解决方案：</h3>
 * <ol>
 * <li>理论解：Q* = μ + σ * Φ⁻¹((p+s-c)/(p+s))</li>
 * <li>数值优化：使用 LBFGS 算法求解</li>
 * <li>蒙特卡洛模拟：验证结果并进行风险分析</li>
 * </ol>
 * 
 * @author lteb2
 */
public class NewsvendorExample {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("           报童模型求解示例");
        System.out.println("=".repeat(60));
        
        // 1. 定义问题参数
        double purchaseCost = 5.0;    // 采购成本
        double sellingPrice = 10.0;   // 销售价格
        double shortageCost = 3.0;    // 缺货损失
        double demandMean = 100.0;    // 需求均值
        double demandStd = 20.0;      // 需求标准差
        
        System.out.println("\n1. 问题参数设置:");
        System.out.printf("   采购成本: %.2f 元/份%n", purchaseCost);
        System.out.printf("   销售价格: %.2f 元/份%n", sellingPrice);
        System.out.printf("   缺货损失: %.2f 元/份%n", shortageCost);
        System.out.printf("   需求分布: N(%.1f, %.1f²)%n", demandMean, demandStd);
        
        // 2. 理论解分析
        System.out.println("\n2. 理论解分析:");
        NormalDistribution demandDist = new NormalDistribution(demandMean, demandStd);
        
        // 计算临界比率
        double criticalRatio = (sellingPrice + shortageCost - purchaseCost) / (sellingPrice + shortageCost);
        System.out.printf("   临界比率: %.4f%n", criticalRatio);
        
        // 理论最优订货量
        double theoreticalOptimal = demandMean + demandStd * demandDist.ppf(criticalRatio);
        System.out.printf("   理论最优订货量: %.2f 份%n", theoreticalOptimal);
        
        // 理论最大期望利润
        NewsvendorObjectiveFunction objFunc = new NewsvendorObjectiveFunction(
            purchaseCost, sellingPrice, shortageCost, demandMean, demandStd);
        double theoreticalMaxProfit = objFunc.computeExpectedProfit(theoreticalOptimal);
        System.out.printf("   理论最大期望利润: %.2f 元%n", theoreticalMaxProfit);
        
        // 3. 数值优化求解
        System.out.println("\n3. 数值优化求解:");
        NewsvendorSolver solver = new NewsvendorSolver();
        solver.setParameters(purchaseCost, sellingPrice, shortageCost, demandMean, demandStd);
        
        // 设置初始猜测值
        IVector<Double> initialGuess = Linalg.vector(new double[]{demandMean});
        
        // 求解
        NewsvendorResult result = solver.solve(initialGuess);
        result.printSummary();
        
        // 4. 敏感性分析
        System.out.println("\n4. 敏感性分析:");
        double[] costVariations = {0.8, 0.9, 1.0, 1.1, 1.2};
        double[] priceVariations = {0.8, 0.9, 1.0, 1.1, 1.2};
        
        System.out.println("\n   采购成本敏感性分析:");
        System.out.printf("   %-12s %-15s %-15s%n", "成本倍数", "最优订货量", "最大利润");
        System.out.println("   " + "-".repeat(45));
        
        for (double factor : costVariations) {
            NewsvendorSolver tempSolver = new NewsvendorSolver();
            tempSolver.setParameters(purchaseCost * factor, sellingPrice, shortageCost, demandMean, demandStd);
            NewsvendorResult tempResult = tempSolver.solve(initialGuess);
            System.out.printf("   %-12.1f %-15.2f %-15.2f%n", 
                factor, tempResult.getOptimalQuantity(), tempResult.getMaxExpectedProfit());
        }
        
        System.out.println("\n   销售价格敏感性分析:");
        System.out.printf("   %-12s %-15s %-15s%n", "价格倍数", "最优订货量", "最大利润");
        System.out.println("   " + "-".repeat(45));
        
        for (double factor : priceVariations) {
            NewsvendorSolver tempSolver = new NewsvendorSolver();
            tempSolver.setParameters(purchaseCost, sellingPrice * factor, shortageCost, demandMean, demandStd);
            NewsvendorResult tempResult = tempSolver.solve(initialGuess);
            System.out.printf("   %-12.1f %-15.2f %-15.2f%n", 
                factor, tempResult.getOptimalQuantity(), tempResult.getMaxExpectedProfit());
        }
        
        // 5. 蒙特卡洛模拟验证
        System.out.println("\n5. 蒙特卡洛模拟验证:");
        NewsvendorSimulation simulation = new NewsvendorSimulation(
            purchaseCost, sellingPrice, shortageCost, demandMean, demandStd, 12345L);
        
        // 模拟最优订货量的表现
        NewsvendorSimulation.SimulationResult simResult = 
            simulation.simulate(result.getOptimalQuantity(), 10000);
        simResult.printResults();
        
        // 比较理论值与模拟值
        System.out.println("\n   理论值 vs 模拟值比较:");
        System.out.printf("   理论期望利润: %.2f 元%n", theoreticalMaxProfit);
        System.out.printf("   模拟平均利润: %.2f 元%n", simResult.getAverageProfit());
        System.out.printf("   差异: %.2f 元 (%.2f%%)%n", 
            Math.abs(theoreticalMaxProfit - simResult.getAverageProfit()),
            Math.abs(theoreticalMaxProfit - simResult.getAverageProfit()) / theoreticalMaxProfit * 100);
        
        // 6. 订货量优化模拟
        System.out.println("\n6. 订货量优化模拟:");
        NewsvendorSimulation.OptimizationSimulationResult optResult = 
            simulation.optimizeBySimulation(50, 150, 21, 5000);
        
        System.out.printf("   模拟最优订货量: %.2f 份%n", optResult.getOptimalQuantity());
        System.out.printf("   模拟最大平均利润: %.2f 元%n", optResult.getMaxProfit());
        System.out.printf("   与理论最优的差异: %.2f 份%n", 
            Math.abs(optResult.getOptimalQuantity() - theoreticalOptimal));
        
        // 7. 风险分析
        System.out.println("\n7. 风险分析:");
        NewsvendorSimulation.RiskAnalysisResult riskResult = 
            simulation.analyzeRisk(result.getOptimalQuantity(), 10000, 0.95);
        riskResult.printResults();
        
        // 8. 不同订货量的风险比较
        System.out.println("\n8. 不同订货量的风险比较:");
        double[] testQuantities = {80, 100, 120, result.getOptimalQuantity()};
        String[] labels = {"保守策略", "均值策略", "激进策略", "最优策略"};
        
        System.out.printf("   %-12s %-10s %-12s %-12s %-12s%n", 
            "策略", "订货量", "平均利润", "VaR(95%)", "亏损概率");
        System.out.println("   " + "-".repeat(65));
        
        for (int i = 0; i < testQuantities.length; i++) {
            NewsvendorSimulation.RiskAnalysisResult risk = 
                simulation.analyzeRisk(testQuantities[i], 5000, 0.95);
            System.out.printf("   %-12s %-10.2f %-12.2f %-12.2f %-12.2f%%%n",
                labels[i], testQuantities[i], risk.getAvgProfit(), 
                risk.getVar(), risk.getLossProbability() * 100);
        }
        
        // 9. 业务建议
        System.out.println("\n9. 业务建议:");
        System.out.printf("   • 建议订货量: %.0f 份%n", result.getOptimalQuantity());
        System.out.printf("   • 预期日利润: %.2f 元%n", result.getMaxExpectedProfit());
        System.out.printf("   • 预期服务水平: %.1f%%%n", simResult.getServiceLevel() * 100);
        
        double roi = (result.getMaxExpectedProfit() / (result.getOptimalQuantity() * purchaseCost)) * 100;
        System.out.printf("   • 投资回报率: %.1f%%%n", roi);
        
        if (riskResult.getLossProbability() > 0.1) {
            System.out.println("   • 风险提醒: 亏损概率较高，建议考虑风险管理措施");
        }
        
        if (simResult.getServiceLevel() < 0.9) {
            System.out.println("   • 服务提醒: 服务水平较低，可能影响客户满意度");
        }
        
        System.out.println("\n" + "=".repeat(60));
        System.out.println("           示例运行完成");
        System.out.println("=".repeat(60));
    }
    
    /**
     * 运行简化示例（用于快速测试）
     */
    public static void runSimpleExample() {
        System.out.println("=== 报童模型简化示例 ===");
        
        // 基本参数
        double cost = 3.0, price = 8.0, shortage = 2.0;
        double mean = 50.0, std = 10.0;
        
        // 求解
        NewsvendorSolver solver = new NewsvendorSolver();
        solver.setParameters(cost, price, shortage, mean, std);
        IVector<Double> initial = Linalg.vector(new double[]{mean});
        NewsvendorResult result = solver.solve(initial);
        
        // 输出结果
        System.out.printf("最优订货量: %.2f%n", result.getOptimalQuantity());
        System.out.printf("最大期望利润: %.2f%n", result.getMaxExpectedProfit());
        
        // 简单模拟验证
        NewsvendorSimulation sim = new NewsvendorSimulation(cost, price, shortage, mean, std);
        NewsvendorSimulation.SimulationResult simResult = 
            sim.simulate(result.getOptimalQuantity(), 1000);
        System.out.printf("模拟平均利润: %.2f%n", simResult.getAverageProfit());
    }
}