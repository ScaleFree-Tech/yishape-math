package model_zoo.naive_news_vendor;

import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.stats.Stats;

/**
 * 报童模型使用示例
 * <p>
 * 这个示例展示了如何使用 NewsvendorModel 类来求解经典的报童问题。
 * </p>
 * 
 * <h3>问题设置：</h3>
 * <ul>
 * <li>每份报纸的采购成本为 5 元</li>
 * <li>每份报纸的销售价格为 10 元</li>
 * <li>每份缺货的损失为 3 元</li>
 * <li>需求服从正态分布 N(100, 20²)</li>
 * </ul>
 * 
 * @author lteb2
 */
public class NewsvendorExample {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("           经典报童模型求解示例");
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
        
        // 2. 创建报童模型实例
        var demandDist = Stats.norm(demandMean,demandStd);
        NewsvendorModel model = new NewsvendorModel(
            purchaseCost, sellingPrice, shortageCost, demandDist);
        
        // 3. 理论解分析
        System.out.println("\n2. 理论解分析:");
        
        // 计算临界比率
        double criticalRatio = (sellingPrice - purchaseCost) / (sellingPrice - purchaseCost + shortageCost);
        System.out.printf("   临界比率: %.4f%n", criticalRatio);
        
        // 理论最优订货量
        double theoreticalOptimal = model.computeTheoreticalOptimalQuantity();
        System.out.printf("   理论最优订货量: %.2f 份%n", theoreticalOptimal);
        
        // 理论最大期望利润
        double theoreticalMaxProfit = model.computeExpectedProfit(theoreticalOptimal);
        System.out.printf("   理论最大期望利润: %.2f 元%n", theoreticalMaxProfit);
        
        // 4. 数值优化求解
        System.out.println("\n3. 数值优化求解:");
        OptResult numericalResult = model.solveNumerically();
        
        double numericalOptimal = numericalResult.getOptimalPoint().get(0);
        double numericalMaxProfit = -numericalResult.getOptimalValue(); // 负号因为目标函数是负利润
        
        System.out.printf("   数值最优订货量: %.2f 份%n", numericalOptimal);
        System.out.printf("   数值最大期望利润: %.2f 元%n", numericalMaxProfit);
        System.out.printf("   优化迭代次数: %d%n", numericalResult.getIterations());
        System.out.printf("   是否收敛: %s%n", numericalResult.isConverged() ? "是" : "否");
        
        // 5. 结果比较
        System.out.println("\n4. 结果比较:");
        double quantityDifference = Math.abs(theoreticalOptimal - numericalOptimal);
        double profitDifference = Math.abs(theoreticalMaxProfit - numericalMaxProfit);
        
        System.out.printf("   订货量差异: %.4f 份%n", quantityDifference);
        System.out.printf("   利润差异: %.4f 元%n", profitDifference);
        
        if (quantityDifference < 1e-3 && profitDifference < 1e-3) {
            System.out.println("   ✓ 理论解与数值解一致");
        } else {
            System.out.println("   ⚠ 理论解与数值解存在差异");
        }
        
        // 6. 敏感性分析示例
        System.out.println("\n5. 敏感性分析示例:");
        System.out.println("   不同参数下的最优订货量:");
        System.out.printf("   %-15s %-15s %-15s%n", "采购成本", "最优订货量", "期望利润");
        System.out.println("   " + "-".repeat(48));
        
        double[] costVariations = {4.0, 5.0, 6.0, 7.0};
        for (double cost : costVariations) {
            NewsvendorModel variantModel = new NewsvendorModel(
                cost, sellingPrice, shortageCost, demandDist);
            double optimalQty = variantModel.computeTheoreticalOptimalQuantity();
            double expectedProfit = variantModel.computeExpectedProfit(optimalQty);
            System.out.printf("   %-15.1f %-15.2f %-15.2f%n", cost, optimalQty, expectedProfit);
        }
        
        // 7. 业务建议
        System.out.println("\n6. 业务建议:");
        System.out.printf("   • 建议订货量: %.0f 份%n", theoreticalOptimal);
        System.out.printf("   • 预期日利润: %.2f 元%n", theoreticalMaxProfit);
        
        double roi = (theoreticalMaxProfit / (theoreticalOptimal * purchaseCost)) * 100;
        System.out.printf("   • 投资回报率: %.1f%%%n", roi);
        
        System.out.println("\n" + "=".repeat(60));
        System.out.println("           示例运行完成");
        System.out.println("=".repeat(60));
    }
}