package com.reremouse.lab.math.test.news_vendor2;

import com.reremouse.lab.math.optimize.IGradientFunction;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.stats.distribution.NormalDistribution;

/**
 * 报童模型梯度函数
 * <p>
 * 计算报童模型目标函数关于订货量的梯度。
 * 期望利润函数的导数为：
 * dE[π(Q)]/dQ = (p - c + s) * [1 - F(Q)] - (p + s)
 * 其中 F(Q) 是需求分布的累积分布函数
 * </p>
 * <p>
 * 由于目标函数是期望利润的负值，梯度也需要取负值
 * </p>
 * 
 * @author lteb2
 */
public class NewsvendorGradientFunction implements IGradientFunction {
    
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
    public NewsvendorGradientFunction(double purchaseCost, double sellingPrice, 
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
     * 计算梯度向量
     * 
     * @param x 决策变量向量，x[0] 表示订货量 Q
     * @return 梯度向量
     */
    @Override
    public IVector computeGradient(IVector x) {
        if (x.length() != 1) {
            throw new IllegalArgumentException("报童模型只有一个决策变量：订货量");
        }
        
        double Q = x.get(0).doubleValue();
        
        // 计算梯度
        double gradient;
        if (Q < 0) {
            // 对负订货量的惩罚项梯度
            gradient = 2 * Q;
        } else {
            gradient = computeProfitGradient(Q);
        }
        
        // 创建梯度向量（注意取负值，因为目标函数是期望利润的负值）
        return Linalg.vector(new double[]{-gradient});
    }
    
    /**
     * 计算期望利润的梯度
     * 
     * @param Q 订货量
     * @return 期望利润关于Q的导数
     */
    private double computeProfitGradient(double Q) {
        // 计算需求分布在Q处的累积分布函数值
        double F_Q = demandDistribution.cdf(Q);
        
        // 期望利润的导数：dE[π(Q)]/dQ = (p - c + s) * [1 - F(Q)] - (p + s)
        // 简化为：dE[π(Q)]/dQ = (p - c + s) * [1 - F(Q)] - (p + s)
        // 进一步简化：dE[π(Q)]/dQ = (p - c + s) - (p - c + s) * F(Q) - (p + s)
        // 最终：dE[π(Q)]/dQ = -c - s - (p - c + s) * F(Q)
        
        double gradient = (sellingPrice - purchaseCost + shortageCost) * (1 - F_Q) - (sellingPrice + shortageCost);
        
        return gradient;
    }
    
    /**
     * 数值梯度计算（用于验证解析梯度的正确性）
     * 
     * @param x 决策变量向量
     * @param objectiveFunction 目标函数
     * @param epsilon 数值微分步长
     * @return 数值梯度向量
     */
    public IVector<Double> computeNumericalGradient(IVector<Double> x, NewsvendorObjectiveFunction objectiveFunction, double epsilon) {
        double Q = x.get(0).doubleValue();
        
        // 前向差分
        IVector xPlus = Linalg.vector(new double[]{Q + epsilon});
        IVector xMinus = Linalg.vector(new double[]{Q - epsilon});
        
        double fPlus = objectiveFunction.computeObjective(xPlus);
        double fMinus = objectiveFunction.computeObjective(xMinus);
        
        double numericalGradient = (fPlus - fMinus) / (2 * epsilon);
        
        return Linalg.vector(new double[]{numericalGradient});
    }
    
    /**
     * 验证解析梯度与数值梯度的一致性
     * 
     * @param x 测试点
     * @param objectiveFunction 目标函数
     * @param tolerance 容差
     * @return 是否一致
     */
    public boolean verifyGradient(IVector x, NewsvendorObjectiveFunction objectiveFunction, double tolerance) {
        IVector analyticalGrad = computeGradient(x);
        IVector numericalGrad = computeNumericalGradient(x, objectiveFunction, 1e-8);
        
        double diff = Math.abs(analyticalGrad.get(0).doubleValue() - numericalGrad.get(0).doubleValue());
        return diff < tolerance;
    }
    
    // Getter 方法
    public double getPurchaseCost() { return purchaseCost; }
    public double getSellingPrice() { return sellingPrice; }
    public double getShortageCost() { return shortageCost; }
    public NormalDistribution getDemandDistribution() { return demandDistribution; }
}