package com.reremouse.lab.math.test.news_vendor2;

import com.reremouse.lab.math.optimize.IObjectiveFunction;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.stats.distribution.NormalDistribution;

/**
 * 报童模型目标函数
 * <p>
 * 报童模型是一个经典的库存管理问题，目标是在不确定需求下确定最优订货量。
 * 该模型考虑以下参数：
 * - 单位采购成本 c
 * - 单位销售价格 p
 * - 单位缺货损失 s（机会成本）
 * - 需求服从正态分布 N(μ, σ²)
 * </p>
 * <p>
 * 期望利润函数为：
 * E[π(Q)] = p * E[min(Q, D)] - c * Q - s * E[max(0, D - Q)]
 * 其中 Q 是订货量，D 是随机需求
 * </p>
 * <p>
 * 由于优化器求解最小值，我们使用期望利润的负值作为目标函数
 * </p>
 * 
 * @author lteb2
 */
public class NewsvendorObjectiveFunction implements IObjectiveFunction {
    
    /** 单位采购成本 */
    private final double purchaseCost;
    
    /** 单位销售价格 */
    private final double sellingPrice;
    
    /** 单位缺货损失（机会成本） */
    private final double shortageCost;
    
    /** 需求分布 */
    private final NormalDistribution demandDistribution;
    
    /**
     * 构造函数
     * 
     * @param purchaseCost 单位采购成本
     * @param sellingPrice 单位销售价格
     * @param shortageCost 单位缺货损失
     * @param demandMean 需求均值
     * @param demandStd 需求标准差
     */
    public NewsvendorObjectiveFunction(double purchaseCost, double sellingPrice, 
                                     double shortageCost, double demandMean, double demandStd) {
        if (sellingPrice <= purchaseCost) {
            throw new IllegalArgumentException("销售价格必须大于采购成本");
        }
        if (purchaseCost < 0 || sellingPrice < 0 || shortageCost < 0) {
            throw new IllegalArgumentException("成本和价格必须为非负数");
        }
        if (demandStd <= 0) {
            throw new IllegalArgumentException("需求标准差必须为正数");
        }
        
        this.purchaseCost = purchaseCost;
        this.sellingPrice = sellingPrice;
        this.shortageCost = shortageCost;
        this.demandDistribution = new NormalDistribution(demandMean, demandStd);
    }
    
    /**
     * 计算目标函数值（期望利润的负值）
     * 
     * @param x 决策变量向量，x[0] 表示订货量 Q
     * @return 期望利润的负值
     */
    @Override
    public double computeObjective(IVector x) {
        if (x.length() != 1) {
            throw new IllegalArgumentException("报童模型只有一个决策变量：订货量");
        }
        
        double Q = ((Number) x.get(0)).doubleValue();
        if (Q < 0) {
            // 对负订货量施加惩罚
            return 1e6 + Q * Q;
        }
        
        // 计算期望利润
        double expectedProfit = computeExpectedProfit(Q);
        
        // 返回负值用于最小化
        return -expectedProfit;
    }
    
    /**
     * /**
     * 计算期望利润
     * @param Q 订货量
     * @return 期望利润
     */
    public double computeExpectedProfit(double Q) {
        double mu = demandDistribution.mean();
        double sigma = demandDistribution.std();
        
        // 标准化变量 z = (Q - μ) / σ
        double z = (Q - mu) / sigma;
        
        // 标准正态分布的累积分布函数和概率密度函数
        NormalDistribution standardNormal = new NormalDistribution(0, 1);
        double phi_z = standardNormal.cdf(z);  // Φ(z)
        double phi_z_pdf = standardNormal.pdf(z);  // φ(z)
        
        // 计算期望销售量 E[min(Q, D)]
        double expectedSales = mu * phi_z + sigma * phi_z_pdf - (Q - mu) * (1 - phi_z);
        
        // 计算期望缺货量 E[max(0, D - Q)]
        double expectedShortage = (mu - Q) * (1 - phi_z) + sigma * phi_z_pdf;
        
        // 期望利润 = 销售收入 - 采购成本 - 缺货损失
        double expectedProfit = sellingPrice * expectedSales - purchaseCost * Q - shortageCost * expectedShortage;
        
        return expectedProfit;
    }
    
    /**
     * 获取理论最优解（用于验证）
     * 
     * @return 理论最优订货量
     */
    public double getOptimalQuantity() {
        // 临界比率 = (p - c + s) / (p + s)
        double criticalRatio = (sellingPrice - purchaseCost + shortageCost) / (sellingPrice + shortageCost);
        
        // 最优订货量 Q* = μ + σ * Φ^(-1)(临界比率)
        double optimalQuantity = demandDistribution.mean() + 
                                demandDistribution.std() * demandDistribution.ppf(criticalRatio);
        
        return optimalQuantity;
    }
    
    // Getter 方法
    public double getPurchaseCost() { return purchaseCost; }
    public double getSellingPrice() { return sellingPrice; }
    public double getShortageCost() { return shortageCost; }
    public NormalDistribution getDemandDistribution() { return demandDistribution; }
}