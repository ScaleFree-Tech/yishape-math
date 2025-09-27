package com.reremouse.lab.math.test.news_vendor2;

import com.reremouse.lab.math.optimize.newton.RereLBFGS;
import com.reremouse.lab.math.optimize.OptResult;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.stats.distribution.NormalDistribution;

/**
 * 报童模型求解器
 * <p>
 * 使用LBFGS优化算法求解报童模型的最优订货量问题。
 * 报童模型是一个经典的单周期库存管理问题，目标是在不确定需求下
 * 确定最优订货量以最大化期望利润。
 * </p>
 * 
 * <h3>模型参数：</h3>
 * <ul>
 *   <li>c: 单位采购成本</li>
 *   <li>p: 单位销售价格</li>
 *   <li>s: 单位缺货损失（机会成本）</li>
 *   <li>D ~ N(μ, σ²): 需求服从正态分布</li>
 * </ul>
 * 
 * <h3>决策变量：</h3>
 * <ul>
 *   <li>Q: 订货量</li>
 * </ul>
 * 
 * <h3>目标函数：</h3>
 * <p>最大化期望利润：E[π(Q)] = p * E[min(Q, D)] - c * Q - s * E[max(0, D - Q)]</p>
 * 
 * @author lteb2
 */
public class NewsvendorSolver {
    
    /** LBFGS优化器 */
    private final RereLBFGS optimizer;
    
    /** 目标函数 */
    private NewsvendorObjectiveFunction objectiveFunction;
    
    /** 梯度函数 */
    private NewsvendorGradientFunction gradientFunction;
    
    /** 结果对象 */
    private NewsvendorResult result;
    
    /** 模型参数 */
    private double purchaseCost;
    private double sellingPrice;
    private double shortageCost;
    private double demandMean;
    private double demandStd;
    
    /**
     * 构造函数，使用默认的LBFGS参数
     */
    public NewsvendorSolver() {
        this.optimizer = new RereLBFGS();
    }
    
    /**
     * 构造函数，允许自定义LBFGS参数
     * 
     * @param m 存储的历史信息对数
     * @param tolerance 收敛容差
     * @param maxIterations 最大迭代次数
     */
    public NewsvendorSolver(int m, double tolerance, int maxIterations) {
        this.optimizer = new RereLBFGS(m, tolerance, maxIterations);
    }
    
    /**
     * 求解报童模型
     * 
     * @param purchaseCost 单位采购成本
     * @param sellingPrice 单位销售价格
     * @param shortageCost 单位缺货损失
     * @param demandMean 需求均值
     * @param demandStd 需求标准差
     * @param initialGuess 初始猜测值（可选，默认为需求均值）
     * @return 求解结果
     */
    public NewsvendorResult solve(double purchaseCost, double sellingPrice, double shortageCost,
                                double demandMean, double demandStd, Double initialGuess) {
        
        // 参数验证
        validateParameters(purchaseCost, sellingPrice, shortageCost, demandMean, demandStd);
        
        // 创建目标函数和梯度函数
        this.objectiveFunction = new NewsvendorObjectiveFunction(
            purchaseCost, sellingPrice, shortageCost, demandMean, demandStd);
        this.gradientFunction = new NewsvendorGradientFunction(
            purchaseCost, sellingPrice, shortageCost, demandMean, demandStd);
        
        // 设置初始点
        double initQ = (initialGuess != null) ? initialGuess : demandMean;
        IVector initialPoint = Linalg.vector(new double[]{initQ});
        
        // 验证梯度计算的正确性
        if (!gradientFunction.verifyGradient(initialPoint, objectiveFunction, 1e-6)) {
            System.out.println("警告：梯度验证失败，可能影响优化结果");
        }
        
        // 执行优化
        OptResult optResult = optimizer.optimize(initialPoint, objectiveFunction, gradientFunction);
        
        // 计算理论最优解用于比较
        double theoreticalOptimal = objectiveFunction.getOptimalQuantity();
        
        // 创建结果对象
        this.result = new NewsvendorResult(
            optResult,
            theoreticalOptimal,
            purchaseCost,
            sellingPrice,
            shortageCost,
            demandMean,
            demandStd
        );
        
        return this.result;
    }
    
    /**
     * 求解报童模型（使用默认初始猜测值）
     */
    public NewsvendorResult solve(double purchaseCost, double sellingPrice, double shortageCost,
                                double demandMean, double demandStd) {
        return solve(purchaseCost, sellingPrice, shortageCost, demandMean, demandStd, null);
    }
    
    /**
     * 求解报童模型（使用IVector作为初始猜测值）
     */
    public NewsvendorResult solve(IVector<Double> initialGuess) {
        if (objectiveFunction == null || gradientFunction == null) {
            throw new IllegalStateException("必须先设置模型参数才能求解");
        }
        
        // 验证梯度计算的正确性
        if (!gradientFunction.verifyGradient(initialGuess, objectiveFunction, 1e-6)) {
            System.out.println("警告：梯度验证失败，可能影响优化结果");
        }
        
        // 执行优化
        OptResult optResult = optimizer.optimize(initialGuess, objectiveFunction, gradientFunction);
        
        // 计算理论最优解
        double criticalRatio = (sellingPrice + shortageCost - purchaseCost) / (sellingPrice + shortageCost);
        NormalDistribution demandDist = new NormalDistribution(demandMean, demandStd);
        double theoreticalOptimal = demandMean + demandStd * demandDist.ppf(criticalRatio);
        
        // 创建并返回结果
        this.result = new NewsvendorResult(optResult, theoreticalOptimal, 
                                         purchaseCost, sellingPrice, shortageCost, 
                                         demandMean, demandStd);
        return result;
    }
    
    /**
     * 设置模型参数（用于分离参数设置和求解过程）
     */
    public void setParameters(double purchaseCost, double sellingPrice, double shortageCost,
                            double demandMean, double demandStd) {
        // 参数验证
        validateParameters(purchaseCost, sellingPrice, shortageCost, demandMean, demandStd);
        
        // 保存参数
        this.purchaseCost = purchaseCost;
        this.sellingPrice = sellingPrice;
        this.shortageCost = shortageCost;
        this.demandMean = demandMean;
        this.demandStd = demandStd;
        
        // 创建目标函数和梯度函数
        this.objectiveFunction = new NewsvendorObjectiveFunction(
            purchaseCost, sellingPrice, shortageCost, demandMean, demandStd);
        this.gradientFunction = new NewsvendorGradientFunction(
            purchaseCost, sellingPrice, shortageCost, demandMean, demandStd);
    }
    
    /**
     * 参数验证
     */
    private void validateParameters(double purchaseCost, double sellingPrice, double shortageCost,
                                  double demandMean, double demandStd) {
        if (sellingPrice <= purchaseCost) {
            throw new IllegalArgumentException("销售价格必须大于采购成本");
        }
        if (purchaseCost < 0 || sellingPrice < 0 || shortageCost < 0) {
            throw new IllegalArgumentException("成本和价格必须为非负数");
        }
        if (demandMean < 0) {
            throw new IllegalArgumentException("需求均值必须为非负数");
        }
        if (demandStd <= 0) {
            throw new IllegalArgumentException("需求标准差必须为正数");
        }
    }
    
    /**
     * 敏感性分析：分析参数变化对最优解的影响
     * 
     * @param baseParams 基准参数 [purchaseCost, sellingPrice, shortageCost, demandMean, demandStd]
     * @param paramIndex 要分析的参数索引（0-4）
     * @param changeRatios 变化比率数组
     * @return 敏感性分析结果
     */
    public SensitivityAnalysisResult performSensitivityAnalysis(double[] baseParams, int paramIndex, double[] changeRatios) {
        if (baseParams.length != 5) {
            throw new IllegalArgumentException("基准参数数组长度必须为5");
        }
        if (paramIndex < 0 || paramIndex >= 5) {
            throw new IllegalArgumentException("参数索引必须在0-4之间");
        }
        
        double[] optimalQuantities = new double[changeRatios.length];
        double[] expectedProfits = new double[changeRatios.length];
        
        for (int i = 0; i < changeRatios.length; i++) {
            double[] params = baseParams.clone();
            params[paramIndex] *= changeRatios[i];
            
            try {
                NewsvendorResult result = solve(params[0], params[1], params[2], params[3], params[4]);
                optimalQuantities[i] = result.getOptimalQuantity();
                expectedProfits[i] = result.getMaxExpectedProfit();
            } catch (Exception e) {
                optimalQuantities[i] = Double.NaN;
                expectedProfits[i] = Double.NaN;
            }
        }
        
        return new SensitivityAnalysisResult(changeRatios, optimalQuantities, expectedProfits, paramIndex);
    }
    
    // Getter 方法
    public NewsvendorObjectiveFunction getObjectiveFunction() { return objectiveFunction; }
    public NewsvendorGradientFunction getGradientFunction() { return gradientFunction; }
    public NewsvendorResult getResult() { return result; }
    public RereLBFGS getOptimizer() { return optimizer; }
    
    /**
     * 敏感性分析结果类
     */
    public static class SensitivityAnalysisResult {
        private final double[] changeRatios;
        private final double[] optimalQuantities;
        private final double[] expectedProfits;
        private final int parameterIndex;
        private final String[] parameterNames = {"采购成本", "销售价格", "缺货损失", "需求均值", "需求标准差"};
        
        public SensitivityAnalysisResult(double[] changeRatios, double[] optimalQuantities, 
                                       double[] expectedProfits, int parameterIndex) {
            this.changeRatios = changeRatios;
            this.optimalQuantities = optimalQuantities;
            this.expectedProfits = expectedProfits;
            this.parameterIndex = parameterIndex;
        }
        
        public void printResults() {
            System.out.println("\n=== " + parameterNames[parameterIndex] + " 敏感性分析结果 ===");
            System.out.printf("%-10s %-15s %-15s%n", "变化比率", "最优订货量", "最大期望利润");
            System.out.println("----------------------------------------");
            
            for (int i = 0; i < changeRatios.length; i++) {
                System.out.printf("%-10.2f %-15.2f %-15.2f%n", 
                    changeRatios[i], optimalQuantities[i], expectedProfits[i]);
            }
        }
        
        // Getter 方法
        public double[] getChangeRatios() { return changeRatios; }
        public double[] getOptimalQuantities() { return optimalQuantities; }
        public double[] getExpectedProfits() { return expectedProfits; }
        public int getParameterIndex() { return parameterIndex; }
        public String getParameterName() { return parameterNames[parameterIndex]; }
    }
}