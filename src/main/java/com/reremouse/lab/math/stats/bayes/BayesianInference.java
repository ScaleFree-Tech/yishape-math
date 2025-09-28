package com.reremouse.lab.math.stats.bayes;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.stats.distribution.NormalDistribution;
import com.reremouse.lab.math.optimize.IOptimizer;
import com.reremouse.lab.math.optimize.OptResult;
import com.reremouse.lab.math.optimize.newton.RereLBFGS;

/**
 * 贝叶斯推断核心类
 * Core class for Bayesian inference
 */
public class BayesianInference {
    
    /**
     * 执行贝叶斯更新
     * Perform Bayesian update
     * 
     * @param prior 先验分布 / Prior distribution
     * @param likelihood 似然函数 / Likelihood function
     * @param data 观测数据 / Observed data
     * @return 后验分布 / Posterior distribution
     */
    public static IVector bayesianUpdate(IVector prior, IVector likelihood, IVector data) {
        // 对于离散情况，后验正比于似然乘以先验
        IVector unnormalizedPosterior = likelihood.multiply(prior);
        
        // 归一化
        double normalization = unnormalizedPosterior.sum().doubleValue();
        if (normalization <= 0) {
            throw new BayesException("归一化常数必须大于0 / Normalization constant must be greater than 0");
        }
        
        return unnormalizedPosterior.divideByScalar(normalization);
    }
    
    /**
     * 使用共轭先验进行贝叶斯更新（Beta-Binomial模型）
     * Bayesian update using conjugate prior (Beta-Binomial model)
     * 
     * @param alpha Beta先验的alpha参数 / Alpha parameter of Beta prior
     * @param beta Beta先验的beta参数 / Beta parameter of Beta prior
     * @param successes 成功次数 / Number of successes
     * @param trials 试验总次数 / Total number of trials
     * @return 包含后验Beta分布参数的数组 [alpha_posterior, beta_posterior] / Array containing posterior Beta distribution parameters
     */
    public static double[] betaBinomialUpdate(double alpha, double beta, int successes, int trials) {
        if (alpha <= 0 || beta <= 0) {
            throw new BayesException("Beta分布参数必须大于0 / Beta distribution parameters must be greater than 0");
        }
        
        if (successes < 0 || trials < 0 || successes > trials) {
            throw new BayesException("成功次数和试验次数必须非负且成功次数不能超过试验次数 / Successes and trials must be non-negative and successes cannot exceed trials");
        }
        
        // Beta-Binomial共轭更新
        double alphaPosterior = alpha + successes;
        double betaPosterior = beta + trials - successes;
        
        return new double[]{alphaPosterior, betaPosterior};
    }
    
    /**
     * 使用共轭先验进行贝叶斯更新（Dirichlet-Multinomial模型）
     * Bayesian update using conjugate prior (Dirichlet-Multinomial model)
     * 
     * @param priorAlpha Dirichlet先验的alpha参数向量 / Alpha parameter vector of Dirichlet prior
     * @param observations 各类别观测次数 / Observation counts for each category
     * @return 后验Dirichlet分布的alpha参数向量 / Alpha parameter vector of posterior Dirichlet distribution
     */
    public static IVector dirichletMultinomialUpdate(IVector priorAlpha, IVector observations) {
        if (priorAlpha.length() != observations.length()) {
            throw new BayesException("先验参数和观测数据维度必须相同 / Prior parameters and observations must have the same dimension");
        }
        
        // Dirichlet-Multinomial共轭更新
        return priorAlpha.add(observations);
    }
    
    /**
     * 计算最大后验估计（MAP）
     * Calculate Maximum A Posteriori (MAP) estimate
     * 
     * @param logPriorLogLikelihood 负的对数先验和负的对数似然之和的函数 / Function that returns negative log prior plus negative log likelihood
     * @param initialPoint 初始点 / Initial point
     * @param optimizer 优化器 / Optimizer
     * @return MAP估计结果 / MAP estimation result
     */
    public static OptResult calculateMAP(com.reremouse.lab.math.optimize.IObjectiveFunction logPriorLogLikelihood, 
                                       IVector initialPoint, 
                                       IOptimizer optimizer) {
        if (optimizer == null) {
            optimizer = new RereLBFGS();
        }
        
        // 在贝叶斯上下文中，我们最小化负的对数后验（即最大化后验）
        return optimizer.optimize(initialPoint, logPriorLogLikelihood, null);
    }
    
    /**
     * 计算贝叶斯因子
     * Calculate Bayes factor
     * 
     * @param marginalLikelihood1 模型1的边际似然 / Marginal likelihood of model 1
     * @param marginalLikelihood2 模型2的边际似然 / Marginal likelihood of model 2
     * @return 贝叶斯因子 / Bayes factor
     */
    public static double bayesFactor(double marginalLikelihood1, double marginalLikelihood2) {
        if (marginalLikelihood2 <= 0) {
            throw new BayesException("分母的边际似然必须大于0 / Denominator marginal likelihood must be greater than 0");
        }
        
        return marginalLikelihood1 / marginalLikelihood2;
    }
}