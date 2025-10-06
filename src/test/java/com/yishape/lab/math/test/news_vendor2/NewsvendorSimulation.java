package com.yishape.lab.math.test.news_vendor2;

import com.yishape.lab.math.stats.distribution.NormalDistribution;

import java.util.Arrays;
import java.util.Random;

/**
 * 报童模型蒙特卡洛模拟
 * <p>
 * 通过蒙特卡洛模拟验证报童模型的理论结果，并进行风险分析。
 * 模拟包括：
 * 1. 需求随机生成
 * 2. 利润计算
 * 3. 统计分析
 * 4. 风险评估
 * </p>
 * 
 * @author lteb2
 */
public class NewsvendorSimulation {
    
    /** 随机数生成器 */
    private final Random random;
    
    /** 需求分布 */
    private final NormalDistribution demandDistribution;
    
    /** 模型参数 */
    private final double purchaseCost;
    private final double sellingPrice;
    private final double shortageCost;
    
    /**
     * 构造函数
     */
    public NewsvendorSimulation(double purchaseCost, double sellingPrice, double shortageCost,
                              double demandMean, double demandStd, long seed) {
        this.purchaseCost = purchaseCost;
        this.sellingPrice = sellingPrice;
        this.shortageCost = shortageCost;
        this.demandDistribution = new NormalDistribution(demandMean, demandStd);
        this.random = new Random(seed);
    }
    
    /**
     * 构造函数（使用随机种子）
     */
    public NewsvendorSimulation(double purchaseCost, double sellingPrice, double shortageCost,
                              double demandMean, double demandStd) {
        this(purchaseCost, sellingPrice, shortageCost, demandMean, demandStd, System.currentTimeMillis());
    }
    
    /**
     * 单次模拟：给定订货量，计算单次利润
     * 
     * @param orderQuantity 订货量
     * @param demand 实际需求
     * @return 利润
     */
    public double simulateSingleProfit(double orderQuantity, double demand) {
        // 确保需求非负
        demand = Math.max(0, demand);
        
        // 实际销售量 = min(订货量, 需求)
        double actualSales = Math.min(orderQuantity, demand);
        
        // 缺货量 = max(0, 需求 - 订货量)
        double shortage = Math.max(0, demand - orderQuantity);
        
        // 利润 = 销售收入 - 采购成本 - 缺货损失
        double profit = sellingPrice * actualSales - purchaseCost * orderQuantity - shortageCost * shortage;
        
        return profit;
    }
    
    /**
     * 蒙特卡洛模拟
     * 
     * @param orderQuantity 订货量
     * @param numSimulations 模拟次数
     * @return 模拟结果
     */
    public SimulationResult simulate(double orderQuantity, int numSimulations) {
        double[] profits = new double[numSimulations];
        double[] demands = new double[numSimulations];
        double[] sales = new double[numSimulations];
        double[] shortages = new double[numSimulations];
        double[] leftovers = new double[numSimulations];
        
        int stockoutCount = 0;
        int overstockCount = 0;
        
        for (int i = 0; i < numSimulations; i++) {
            // 生成随机需求
            double demand = demandDistribution.sample();
            demand = Math.max(0, demand);  // 确保需求非负
            demands[i] = demand;
            
            // 计算各项指标
            double actualSales = Math.min(orderQuantity, demand);
            double shortage = Math.max(0, demand - orderQuantity);
            double leftover = Math.max(0, orderQuantity - demand);
            double profit = simulateSingleProfit(orderQuantity, demand);
            
            sales[i] = actualSales;
            shortages[i] = shortage;
            leftovers[i] = leftover;
            profits[i] = profit;
            
            // 统计缺货和过量库存情况
            if (shortage > 0) stockoutCount++;
            if (leftover > 0) overstockCount++;
        }
        
        return new SimulationResult(
            orderQuantity, numSimulations,
            profits, demands, sales, shortages, leftovers,
            stockoutCount, overstockCount
        );
    }
    
    /**
     * 订货量优化模拟：测试不同订货量的表现
     * 
     * @param minQuantity 最小订货量
     * @param maxQuantity 最大订货量
     * @param numPoints 测试点数
     * @param numSimulations 每个点的模拟次数
     * @return 优化结果
     */
    public OptimizationSimulationResult optimizeBySimulation(double minQuantity, double maxQuantity, 
                                                           int numPoints, int numSimulations) {
        double[] quantities = new double[numPoints];
        double[] avgProfits = new double[numPoints];
        double[] profitStds = new double[numPoints];
        double[] servicelevels = new double[numPoints];
        
        double step = (maxQuantity - minQuantity) / (numPoints - 1);
        
        for (int i = 0; i < numPoints; i++) {
            double quantity = minQuantity + i * step;
            quantities[i] = quantity;
            
            SimulationResult result = simulate(quantity, numSimulations);
            avgProfits[i] = result.getAverageProfit();
            profitStds[i] = result.getProfitStd();
            servicelevels[i] = result.getServiceLevel();
        }
        
        // 找到最优订货量
        int bestIndex = 0;
        double bestProfit = avgProfits[0];
        for (int i = 1; i < numPoints; i++) {
            if (avgProfits[i] > bestProfit) {
                bestProfit = avgProfits[i];
                bestIndex = i;
            }
        }
        
        return new OptimizationSimulationResult(
            quantities, avgProfits, profitStds, servicelevels,
            quantities[bestIndex], bestProfit
        );
    }
    
    /**
     * 风险分析：计算VaR和CVaR
     * 
     * @param orderQuantity 订货量
     * @param numSimulations 模拟次数
     * @param confidenceLevel 置信水平（如0.95）
     * @return 风险分析结果
     */
    public RiskAnalysisResult analyzeRisk(double orderQuantity, int numSimulations, double confidenceLevel) {
        SimulationResult simResult = simulate(orderQuantity, numSimulations);
        double[] profits = simResult.getProfits();
        
        // 排序利润数组
        Arrays.sort(profits);
        
        // 计算VaR (Value at Risk)
        int varIndex = (int) Math.floor((1 - confidenceLevel) * numSimulations);
        double var = profits[varIndex];
        
        // 计算CVaR (Conditional Value at Risk)
        double cvarSum = 0;
        for (int i = 0; i <= varIndex; i++) {
            cvarSum += profits[i];
        }
        double cvar = cvarSum / (varIndex + 1);
        
        // 计算其他风险指标
        double avgProfit = simResult.getAverageProfit();
        double profitStd = simResult.getProfitStd();
        double sharpeRatio = avgProfit / profitStd;  // 简化的夏普比率
        double lossProb = 0;
        for (double profit : profits) {
            if (profit < 0) lossProb++;
        }
        lossProb /= numSimulations;
        
        return new RiskAnalysisResult(
            orderQuantity, confidenceLevel, var, cvar,
            avgProfit, profitStd, sharpeRatio, lossProb
        );
    }
    
    /**
     * 模拟结果类
     */
    public static class SimulationResult {
        private final double orderQuantity;
        private final int numSimulations;
        private final double[] profits;
        private final double[] demands;
        private final double[] sales;
        private final double[] shortages;
        private final double[] leftovers;
        private final int stockoutCount;
        private final int overstockCount;
        
        public SimulationResult(double orderQuantity, int numSimulations,
                              double[] profits, double[] demands, double[] sales,
                              double[] shortages, double[] leftovers,
                              int stockoutCount, int overstockCount) {
            this.orderQuantity = orderQuantity;
            this.numSimulations = numSimulations;
            this.profits = profits;
            this.demands = demands;
            this.sales = sales;
            this.shortages = shortages;
            this.leftovers = leftovers;
            this.stockoutCount = stockoutCount;
            this.overstockCount = overstockCount;
        }
        
        public double getAverageProfit() {
            return Arrays.stream(profits).average().orElse(0.0);
        }
        
        public double getProfitStd() {
            double mean = getAverageProfit();
            double variance = Arrays.stream(profits)
                .map(p -> Math.pow(p - mean, 2))
                .average().orElse(0.0);
            return Math.sqrt(variance);
        }
        
        public double getAverageDemand() {
            return Arrays.stream(demands).average().orElse(0.0);
        }
        
        public double getAverageSales() {
            return Arrays.stream(sales).average().orElse(0.0);
        }
        
        public double getAverageShortage() {
            return Arrays.stream(shortages).average().orElse(0.0);
        }
        
        public double getAverageLeftover() {
            return Arrays.stream(leftovers).average().orElse(0.0);
        }
        
        public double getServiceLevel() {
            return 1.0 - (double) stockoutCount / numSimulations;
        }
        
        public double getStockoutProbability() {
            return (double) stockoutCount / numSimulations;
        }
        
        public double getOverstockProbability() {
            return (double) overstockCount / numSimulations;
        }
        
        public void printResults() {
            System.out.println("\n=== 蒙特卡洛模拟结果 ===");
            System.out.printf("订货量: %.2f%n", orderQuantity);
            System.out.printf("模拟次数: %d%n", numSimulations);
            System.out.printf("平均利润: %.2f ± %.2f%n", getAverageProfit(), getProfitStd());
            System.out.printf("平均需求: %.2f%n", getAverageDemand());
            System.out.printf("平均销售量: %.2f%n", getAverageSales());
            System.out.printf("平均缺货量: %.2f%n", getAverageShortage());
            System.out.printf("平均剩余库存: %.2f%n", getAverageLeftover());
            System.out.printf("服务水平: %.2f%%%n", getServiceLevel() * 100);
            System.out.printf("缺货概率: %.2f%%%n", getStockoutProbability() * 100);
        }
        
        // Getter 方法
        public double getOrderQuantity() { return orderQuantity; }
        public int getNumSimulations() { return numSimulations; }
        public double[] getProfits() { return profits; }
        public double[] getDemands() { return demands; }
        public double[] getSales() { return sales; }
        public double[] getShortages() { return shortages; }
        public double[] getLeftovers() { return leftovers; }
        public int getStockoutCount() { return stockoutCount; }
        public int getOverstockCount() { return overstockCount; }
    }
    
    /**
     * 优化模拟结果类
     */
    public static class OptimizationSimulationResult {
        private final double[] quantities;
        private final double[] avgProfits;
        private final double[] profitStds;
        private final double[] serviceLevels;
        private final double optimalQuantity;
        private final double maxProfit;
        
        public OptimizationSimulationResult(double[] quantities, double[] avgProfits, 
                                          double[] profitStds, double[] serviceLevels,
                                          double optimalQuantity, double maxProfit) {
            this.quantities = quantities;
            this.avgProfits = avgProfits;
            this.profitStds = profitStds;
            this.serviceLevels = serviceLevels;
            this.optimalQuantity = optimalQuantity;
            this.maxProfit = maxProfit;
        }
        
        public void printResults() {
            System.out.println("\n=== 订货量优化模拟结果 ===");
            System.out.printf("最优订货量: %.2f%n", optimalQuantity);
            System.out.printf("最大平均利润: %.2f%n", maxProfit);
            
            System.out.println("\n详细结果:");
            System.out.printf("%-10s %-12s %-12s %-12s%n", "订货量", "平均利润", "利润标准差", "服务水平");
            System.out.println("-".repeat(50));
            for (int i = 0; i < quantities.length; i++) {
                System.out.printf("%-10.2f %-12.2f %-12.2f %-12.2f%n", 
                    quantities[i], avgProfits[i], profitStds[i], serviceLevels[i]);
            }
        }
        
        // Getter 方法
        public double[] getQuantities() { return quantities; }
        public double[] getAvgProfits() { return avgProfits; }
        public double[] getProfitStds() { return profitStds; }
        public double[] getServiceLevels() { return serviceLevels; }
        public double getOptimalQuantity() { return optimalQuantity; }
        public double getMaxProfit() { return maxProfit; }
    }
    
    /**
     * 风险分析结果类
     */
    public static class RiskAnalysisResult {
        private final double orderQuantity;
        private final double confidenceLevel;
        private final double var;
        private final double cvar;
        private final double avgProfit;
        private final double profitStd;
        private final double sharpeRatio;
        private final double lossProbability;
        
        public RiskAnalysisResult(double orderQuantity, double confidenceLevel,
                                double var, double cvar, double avgProfit, double profitStd,
                                double sharpeRatio, double lossProbability) {
            this.orderQuantity = orderQuantity;
            this.confidenceLevel = confidenceLevel;
            this.var = var;
            this.cvar = cvar;
            this.avgProfit = avgProfit;
            this.profitStd = profitStd;
            this.sharpeRatio = sharpeRatio;
            this.lossProbability = lossProbability;
        }
        
        public void printResults() {
            System.out.println("\n=== 风险分析结果 ===");
            System.out.printf("订货量: %.2f%n", orderQuantity);
            System.out.printf("置信水平: %.1f%%%n", confidenceLevel * 100);
            System.out.printf("平均利润: %.2f%n", avgProfit);
            System.out.printf("利润标准差: %.2f%n", profitStd);
            System.out.printf("VaR (%.1f%%): %.2f%n", confidenceLevel * 100, var);
            System.out.printf("CVaR (%.1f%%): %.2f%n", confidenceLevel * 100, cvar);
            System.out.printf("夏普比率: %.4f%n", sharpeRatio);
            System.out.printf("亏损概率: %.2f%%%n", lossProbability * 100);
        }
        
        // Getter 方法
        public double getOrderQuantity() { return orderQuantity; }
        public double getConfidenceLevel() { return confidenceLevel; }
        public double getVar() { return var; }
        public double getCvar() { return cvar; }
        public double getAvgProfit() { return avgProfit; }
        public double getProfitStd() { return profitStd; }
        public double getSharpeRatio() { return sharpeRatio; }
        public double getLossProbability() { return lossProbability; }
    }
}