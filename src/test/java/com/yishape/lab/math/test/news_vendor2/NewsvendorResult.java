package com.yishape.lab.math.test.news_vendor2;

import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.stats.distribution.NormalDistribution;

/**
 * 报童模型求解结果
 * <p>
 * 包含报童模型优化求解的详细结果，包括最优解、理论解、
 * 性能指标和统计分析等信息。
 * </p>
 * 
 * @author lteb2
 */
public class NewsvendorResult {
    
    /** 优化求解结果 */
    private final OptResult optimizationResult;
    
    /** 理论最优解 */
    private final double theoreticalOptimal;
    
    /** 模型参数 */
    private final double purchaseCost;
    private final double sellingPrice;
    private final double shortageCost;
    private final double demandMean;
    private final double demandStd;
    
    /** 需求分布 */
    private final NormalDistribution demandDistribution;
    
    /**
     * 构造函数
     */
    public NewsvendorResult(OptResult optimizationResult, double theoreticalOptimal,
                          double purchaseCost, double sellingPrice, double shortageCost,
                          double demandMean, double demandStd) {
        this.optimizationResult = optimizationResult;
        this.theoreticalOptimal = theoreticalOptimal;
        this.purchaseCost = purchaseCost;
        this.sellingPrice = sellingPrice;
        this.shortageCost = shortageCost;
        this.demandMean = demandMean;
        this.demandStd = demandStd;
        this.demandDistribution = new NormalDistribution(demandMean, demandStd);
    }
    
    /**
     * 获取优化求解的最优订货量
     */
    public double getOptimalQuantity() {
        return optimizationResult.getOptimalPoint().get(0);
    }
    
    /**
     * 获取最大期望利润
     */
    public double getMaxExpectedProfit() {
        return -optimizationResult.getOptimalValue();  // 注意：目标函数是负利润
    }
    
    /**
     * 获取理论最优订货量
     */
    public double getTheoreticalOptimal() {
        return theoreticalOptimal;
    }
    
    /**
     * 计算求解误差
     */
    public double getSolutionError() {
        return Math.abs(getOptimalQuantity() - theoreticalOptimal);
    }
    
    /**
     * 计算相对误差
     */
    public double getRelativeError() {
        return getSolutionError() / Math.abs(theoreticalOptimal);
    }
    
    /**
     * 计算临界比率
     */
    public double getCriticalRatio() {
        return (sellingPrice - purchaseCost + shortageCost) / (sellingPrice + shortageCost);
    }
    
    /**
     * 计算服务水平（满足需求的概率）
     */
    public double getServiceLevel() {
        return demandDistribution.cdf(getOptimalQuantity());
    }
    
    /**
     * 计算期望销售量
     */
    public double getExpectedSales() {
        double Q = getOptimalQuantity();
        double mu = demandMean;
        double sigma = demandStd;
        double z = (Q - mu) / sigma;
        
        NormalDistribution standardNormal = new NormalDistribution(0, 1);
        double phi_z = standardNormal.cdf(z);
        double phi_z_pdf = standardNormal.pdf(z);
        
        return mu * phi_z + sigma * phi_z_pdf - (Q - mu) * (1 - phi_z);
    }
    
    /**
     * 计算期望缺货量
     */
    public double getExpectedShortage() {
        double Q = getOptimalQuantity();
        double mu = demandMean;
        double sigma = demandStd;
        double z = (Q - mu) / sigma;
        
        NormalDistribution standardNormal = new NormalDistribution(0, 1);
        double phi_z = standardNormal.cdf(z);
        double phi_z_pdf = standardNormal.pdf(z);
        
        return (mu - Q) * (1 - phi_z) + sigma * phi_z_pdf;
    }
    
    /**
     * 计算期望剩余库存
     */
    public double getExpectedLeftover() {
        return getOptimalQuantity() - getExpectedSales();
    }
    
    /**
     * 计算缺货概率
     */
    public double getStockoutProbability() {
        return 1 - demandDistribution.cdf(getOptimalQuantity());
    }
    
    /**
     * 计算过量库存概率
     */
    public double getOverstockProbability() {
        return demandDistribution.cdf(getOptimalQuantity());
    }
    
    /**
     * 计算投资回报率
     */
    public double getROI() {
        double totalCost = purchaseCost * getOptimalQuantity();
        return totalCost > 0 ? getMaxExpectedProfit() / totalCost : 0;
    }
    
    /**
     * 计算利润率
     */
    public double getProfitMargin() {
        double revenue = sellingPrice * getExpectedSales();
        return revenue > 0 ? getMaxExpectedProfit() / revenue : 0;
    }
    
    /**
     * 打印详细结果
     */
    public void printDetailedResults() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("                报童模型求解结果");
        System.out.println("=".repeat(60));
        
        // 模型参数
        System.out.println("\n【模型参数】");
        System.out.printf("  采购成本 (c): %.2f%n", purchaseCost);
        System.out.printf("  销售价格 (p): %.2f%n", sellingPrice);
        System.out.printf("  缺货损失 (s): %.2f%n", shortageCost);
        System.out.printf("  需求均值 (μ): %.2f%n", demandMean);
        System.out.printf("  需求标准差 (σ): %.2f%n", demandStd);
        System.out.printf("  临界比率: %.4f%n", getCriticalRatio());
        
        // 求解结果
        System.out.println("\n【求解结果】");
        System.out.printf("  最优订货量: %.4f%n", getOptimalQuantity());
        System.out.printf("  理论最优解: %.4f%n", theoreticalOptimal);
        System.out.printf("  求解误差: %.6f%n", getSolutionError());
        System.out.printf("  相对误差: %.6f%%n", getRelativeError() * 100);
        System.out.printf("  最大期望利润: %.2f%n", getMaxExpectedProfit());
        
        // 优化信息
        System.out.println("\n【优化信息】");
        System.out.printf("  迭代次数: %d%n", optimizationResult.getIterations());
        System.out.printf("  函数评估次数: %d%n", optimizationResult.getFunctionEvaluations());
        System.out.printf("  是否收敛: %s%n", optimizationResult.isConverged() ? "是" : "否");
        
        // 业务指标
        System.out.println("\n【业务指标】");
        System.out.printf("  服务水平: %.2f%%n", getServiceLevel() * 100);
        System.out.printf("  缺货概率: %.2f%%n", getStockoutProbability() * 100);
        System.out.printf("  期望销售量: %.2f%n", getExpectedSales());
        System.out.printf("  期望缺货量: %.2f%n", getExpectedShortage());
        System.out.printf("  期望剩余库存: %.2f%n", getExpectedLeftover());
        
        // 财务指标
        System.out.println("\n【财务指标】");
        System.out.printf("  投资回报率: %.2f%%n", getROI() * 100);
        System.out.printf("  利润率: %.2f%%n", getProfitMargin() * 100);
        System.out.printf("  总投资成本: %.2f%n", purchaseCost * getOptimalQuantity());
        
        System.out.println("=".repeat(60));
    }
    
    /**
     * 打印简要结果
     */
    public void printSummary() {
        System.out.println("\n【报童模型求解摘要】");
        System.out.printf("最优订货量: %.2f, 最大期望利润: %.2f%n", 
            getOptimalQuantity(), getMaxExpectedProfit());
        System.out.printf("服务水平: %.1f%%, 投资回报率: %.1f%%%n", 
            getServiceLevel() * 100, getROI() * 100);
    }
    
    /**
     * 验证解的合理性
     */
    public boolean validateSolution() {
        // 检查订货量是否为正
        if (getOptimalQuantity() < 0) {
            System.out.println("警告：最优订货量为负数");
            return false;
        }
        
        // 检查相对误差是否在合理范围内
        if (getRelativeError() > 0.01) {  // 1%
            System.out.printf("警告：相对误差较大 (%.2f%%)%n", getRelativeError() * 100);
            return false;
        }
        
        // 检查是否收敛
        if (!optimizationResult.isConverged()) {
            System.out.println("警告：优化算法未收敛");
            return false;
        }
        
        return true;
    }
    
    // Getter 方法
    public OptResult getOptimizationResult() { return optimizationResult; }
    public double getPurchaseCost() { return purchaseCost; }
    public double getSellingPrice() { return sellingPrice; }
    public double getShortageCost() { return shortageCost; }
    public double getDemandMean() { return demandMean; }
    public double getDemandStd() { return demandStd; }
    public NormalDistribution getDemandDistribution() { return demandDistribution; }
}