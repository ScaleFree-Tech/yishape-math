package model_zoo.naive_news_vendor;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.newton.RereLBFGS;
import com.yishape.lab.math.stats.distribution.IContinuousDistribution;

/**
 * 经典报童模型实现
 * <p>
 * 报童模型是运营管理中的经典问题，用于确定最优订货量以最大化期望利润。
 * </p>
 * 
 * <h3>问题描述：</h3>
 * <ul>
 * <li>每份报纸的采购成本为 c</li>
 * <li>每份报纸的销售价格为 p</li>
 * <li>每份缺货的损失为 s</li>
 * <li>需求服从某种分布，分布形态可以以参数设计</li>
 * </ul>
 * 
 * <h3>数学模型：</h3>
 * <p>
 * 目标函数（期望利润）：<br>
 * E[Profit(Q)] = p * E[min(Q, D)] - c * Q - s * E[max(0, D - Q)]<br>
 * 其中 Q 是订货量，D 是随机需求
 * </p>
 * 
 * <p>
 * 优化目标：找到最优订货量 Q* 使得期望利润最大化
 * </p>
 * 
 * @author lteb2
 */
public class NewsvendorModel {
    
    // 模型参数
    private final double purchaseCost;    // 采购成本 c
    private final double sellingPrice;    // 销售价格 p
    private final double shortageCost;    // 缺货损失 s
    private final IContinuousDistribution demandDistribution; // 需求分布
    
    /**
     * 构造函数
     * 
     * @param purchaseCost 采购成本
     * @param sellingPrice 销售价格
     * @param shortageCost 缺货损失
     * @param demandDistribution 需求分布
     */
    public NewsvendorModel(double purchaseCost, double sellingPrice, double shortageCost, 
                          IContinuousDistribution demandDistribution) {
        this.purchaseCost = purchaseCost;
        this.sellingPrice = sellingPrice;
        this.shortageCost = shortageCost;
        this.demandDistribution = demandDistribution;
    }
    
    /**
     * 计算理论最优订货量
     * <p>
     * 使用分位数方法计算理论最优解：<br>
     * Q* = F⁻¹((p + s - c) / (p + s))<br>
     * 其中 F 是需求的累积分布函数
     * </p>
     * 
     * @return 理论最优订货量
     */
    public double computeTheoreticalOptimalQuantity() {
        // 计算临界比率 - 使用正确的公式
        double criticalRatio = (sellingPrice + shortageCost - purchaseCost) / (sellingPrice + shortageCost);
        
        // 使用分位数函数计算最优订货量（PPF是CDF的反函数）
        return demandDistribution.ppf(criticalRatio);
    }
    
    /**
     * 计算给定订货量下的期望利润（理论值）
     * 
     * @param quantity 订货量
     * @return 期望利润
     */
    public double computeExpectedProfit(double quantity) {
        // E[Profit(Q)] = p * E[min(Q, D)] - c * Q - s * E[max(0, D - Q)]
        
        double mean = demandDistribution.mean();
        double std = demandDistribution.std();
        double z = (quantity - mean) / std;
        
        // 标准正态分布的概率密度函数和累积分布函数
        double phi = (1.0 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * z * z);
        double Phi = demandDistribution.cdf(quantity);
        
        // E[min(Q, D)] = μ - σ * φ((Q-μ)/σ) - (Q-μ) * (1 - Φ((Q-μ)/σ))
        double eMin = mean - std * phi - (quantity - mean) * (1 - Phi);
        
        // E[max(0, D - Q)] = σ * φ((Q-μ)/σ) + (μ - Q) * (1 - Φ((Q-μ)/σ))
        double eMax = std * phi + (mean - quantity) * (1 - Phi);
        
        return sellingPrice * eMin - purchaseCost * quantity - shortageCost * eMax;
    }
    
    /**
     * 使用数值优化方法求解最优订货量
     * 
     * @return 数值优化结果
     */
    public OptResult solveNumerically() {
        // 创建目标函数（负利润，因为优化器是最小化）
        NewsvendorObjectiveFunction objFunc = new NewsvendorObjectiveFunction(this);
        
        // 创建梯度函数
        NewsvendorGradientFunction gradFunc = new NewsvendorGradientFunction(this);
        
        // 使用理论最优解作为初始猜测，以帮助收敛
        double theoreticalOptimal = computeTheoreticalOptimalQuantity();
        IVector<Double> initialGuess = Linalg.vector(new double[]{theoreticalOptimal});
        
        // 使用共轭梯度优化器，调整参数以获得更好的收敛性
//        IOptimizer optimizer = new RereConjugateGradient(1e-4, 200000, 0.05);
        IOptimizer optimizer = new RereLBFGS();
        
        // 求解
        return optimizer.optimize(initialGuess, objFunc, gradFunc);
    }
    
    /**
     * 获取模型参数
     */
    public double getPurchaseCost() { return purchaseCost; }
    public double getSellingPrice() { return sellingPrice; }
    public double getShortageCost() { return shortageCost; }
    public IContinuousDistribution getDemandDistribution() { return demandDistribution; }
    
    /**
     * 报童模型目标函数（负期望利润）
     */
    static class NewsvendorObjectiveFunction implements IObjectiveFunction {
        private final NewsvendorModel model;
        
        public NewsvendorObjectiveFunction(NewsvendorModel model) {
            this.model = model;
        }
        
        @Override
        public double computeObjective(IVector x) {
            double quantity = x.get(0).doubleValue();
            // 返回负的期望利润，因为优化器是最小化
            return -model.computeExpectedProfit(quantity);
        }
    }
    
    /**
     * 报童模型梯度函数（负期望利润的梯度）
     */
    static class NewsvendorGradientFunction implements IGradientFunction {
        private final NewsvendorModel model;
        
        public NewsvendorGradientFunction(NewsvendorModel model) {
            this.model = model;
        }
        
        @Override
        public IVector computeGradient(IVector x) {
            double quantity = x.get(0).doubleValue();
            
            // 负期望利润关于订货量的导数：
            // E[Profit(Q)] = p * E[min(Q,D)] - c * Q - s * E[max(0,D-Q)]
            // dE[Profit(Q)]/dQ = p * (1 - F(Q)) - c - s * (-(1 - F(Q)))
            //                  = p * (1 - F(Q)) - c + s * (1 - F(Q))
            //                  = (p + s) * (1 - F(Q)) - c
            
            // So d(-E[Profit(Q)])/dQ = -(p + s) * (1 - F(Q)) + c
            //                        = c - (p + s) * (1 - F(Q))
            
            double cdf = model.demandDistribution.cdf(quantity);
            double gradient = model.purchaseCost - (model.sellingPrice + model.shortageCost) * (1 - cdf);
            
            return Linalg.vector(new double[]{gradient});
        }
    }
}