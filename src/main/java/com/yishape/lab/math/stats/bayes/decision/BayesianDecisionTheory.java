package com.yishape.lab.math.stats.bayes.decision;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.util.*;
import java.util.function.Function;

/**
 * 贝叶斯决策理论
 * Bayesian Decision Theory
 * 
 * <p>实现贝叶斯决策理论的核心组件，包括损失函数、效用函数、
 * 贝叶斯风险最小化和决策边界计算。提供了完整的决策分析框架。</p>
 * 
 * <p>Implements core components of Bayesian decision theory, including 
 * loss functions, utility functions, Bayesian risk minimization, and 
 * decision boundary computation. Provides a complete decision analysis framework.</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class BayesianDecisionTheory {
    
    /**
     * 损失函数接口
     * Loss Function Interface
     */
    @FunctionalInterface
    public interface LossFunction {
        /**
         * 计算损失
         * Calculate loss
         * 
         * @param trueState 真实状态
         * @param action 采取的行动
         * @return 损失值
         */
        double loss(int trueState, int action);
    }
    
    /**
     * 效用函数接口
     * Utility Function Interface
     */
    @FunctionalInterface
    public interface UtilityFunction {
        /**
         * 计算效用
         * Calculate utility
         * 
         * @param trueState 真实状态
         * @param action 采取的行动
         * @return 效用值
         */
        double utility(int trueState, int action);
    }
    
    /**
     * 决策规则接口
     * Decision Rule Interface
     */
    @FunctionalInterface
    public interface DecisionRule {
        /**
         * 根据观测做决策
         * Make decision based on observation
         * 
         * @param observation 观测值
         * @return 决策行动
         */
        int decide(IVector observation);
    }
    
    /**
     * 决策结果
     * Decision Result
     */
    public static class DecisionResult {
        private final int optimalAction;
        private final double minimalRisk;
        private final IVector posteriorProbabilities;
        private final IVector expectedLosses;
        private final Map<String, Object> additionalInfo;
        
        public DecisionResult(int optimalAction, double minimalRisk,
                            IVector posteriorProbabilities, IVector expectedLosses,
                            Map<String, Object> additionalInfo) {
            this.optimalAction = optimalAction;
            this.minimalRisk = minimalRisk;
            this.posteriorProbabilities = posteriorProbabilities;
            this.expectedLosses = expectedLosses;
            this.additionalInfo = new HashMap<>(additionalInfo != null ? additionalInfo : new HashMap<>());
        }
        
        public int getOptimalAction() { return optimalAction; }
        public double getMinimalRisk() { return minimalRisk; }
        public IVector getPosteriorProbabilities() { return posteriorProbabilities; }
        public IVector getExpectedLosses() { return expectedLosses; }
        public Map<String, Object> getAdditionalInfo() { return new HashMap<>(additionalInfo); }
    }
    
    /**
     * 决策边界结果
     * Decision Boundary Result
     */
    public static class DecisionBoundaryResult {
        private final List<IVector> boundaryPoints;
        private final List<Integer> regions;
        private final IMatrix decisionSurface;
        private final Map<String, Object> boundaryInfo;
        
        public DecisionBoundaryResult(List<IVector> boundaryPoints, List<Integer> regions,
                                    IMatrix decisionSurface, Map<String, Object> boundaryInfo) {
            this.boundaryPoints = new ArrayList<>(boundaryPoints);
            this.regions = new ArrayList<>(regions);
            this.decisionSurface = decisionSurface;
            this.boundaryInfo = new HashMap<>(boundaryInfo != null ? boundaryInfo : new HashMap<>());
        }
        
        public List<IVector> getBoundaryPoints() { return new ArrayList<>(boundaryPoints); }
        public List<Integer> getRegions() { return new ArrayList<>(regions); }
        public IMatrix getDecisionSurface() { return decisionSurface; }
        public Map<String, Object> getBoundaryInfo() { return new HashMap<>(boundaryInfo); }
    }
    
    /**
     * 贝叶斯风险最小化
     * Bayesian Risk Minimization
     * 
     * @param posteriorProbabilities 后验概率
     * @param lossFunction 损失函数
     * @param numActions 行动数量
     * @return 决策结果
     */
    public static DecisionResult minimizeBayesianRisk(IVector posteriorProbabilities,
                                                    LossFunction lossFunction,
                                                    int numActions) {
        int numStates = posteriorProbabilities.size();
        IVector expectedLosses = Linalg.vector(numActions);
        
        // 计算每个行动的期望损失
        for (int action = 0; action < numActions; action++) {
            double expectedLoss = 0.0;
            for (int state = 0; state < numStates; state++) {
                double loss = lossFunction.loss(state, action);
                double probability = posteriorProbabilities.get(state).doubleValue();
                expectedLoss += loss * probability;
            }
            expectedLosses.set(action, expectedLoss);
        }
        
        // 找到最小期望损失的行动
        int optimalAction = 0;
        double minimalRisk = expectedLosses.get(0).doubleValue();
        
        for (int action = 1; action < numActions; action++) {
            double risk = expectedLosses.get(action).doubleValue();
            if (risk < minimalRisk) {
                minimalRisk = risk;
                optimalAction = action;
            }
        }
        
        Map<String, Object> additionalInfo = new HashMap<>();
        additionalInfo.put("numStates", numStates);
        additionalInfo.put("numActions", numActions);
        
        return new DecisionResult(optimalAction, minimalRisk, posteriorProbabilities,
                                expectedLosses, additionalInfo);
    }
    
    /**
     * 效用最大化
     * Utility Maximization
     * 
     * @param posteriorProbabilities 后验概率
     * @param utilityFunction 效用函数
     * @param numActions 行动数量
     * @return 决策结果
     */
    public static DecisionResult maximizeExpectedUtility(IVector posteriorProbabilities,
                                                       UtilityFunction utilityFunction,
                                                       int numActions) {
        int numStates = posteriorProbabilities.size();
        IVector expectedUtilities = Linalg.vector(numActions);
        
        // 计算每个行动的期望效用
        for (int action = 0; action < numActions; action++) {
            double expectedUtility = 0.0;
            for (int state = 0; state < numStates; state++) {
                double utility = utilityFunction.utility(state, action);
                double probability = posteriorProbabilities.get(state).doubleValue();
                expectedUtility += utility * probability;
            }
            expectedUtilities.set(action, expectedUtility);
        }
        
        // 找到最大期望效用的行动
        int optimalAction = 0;
        double maximalUtility = expectedUtilities.get(0).doubleValue();
        
        for (int action = 1; action < numActions; action++) {
            double utility = expectedUtilities.get(action).doubleValue();
            if (utility > maximalUtility) {
                maximalUtility = utility;
                optimalAction = action;
            }
        }
        
        Map<String, Object> additionalInfo = new HashMap<>();
        additionalInfo.put("numStates", numStates);
        additionalInfo.put("numActions", numActions);
        additionalInfo.put("maximalUtility", maximalUtility);
        
        return new DecisionResult(optimalAction, -maximalUtility, posteriorProbabilities,
                                expectedUtilities, additionalInfo);
    }
    
    /**
     * 计算决策边界
     * Compute Decision Boundary
     * 
     * @param likelihoodFunctions 似然函数列表
     * @param priorProbabilities 先验概率
     * @param lossFunction 损失函数
     * @param observationSpace 观测空间网格
     * @return 决策边界结果
     */
    public static DecisionBoundaryResult computeDecisionBoundary(
            List<Function<IVector, Double>> likelihoodFunctions,
            IVector priorProbabilities,
            LossFunction lossFunction,
            List<IVector> observationSpace) {
        
        int numStates = likelihoodFunctions.size();
        int numActions = numStates; // 假设行动数等于状态数
        
        List<IVector> boundaryPoints = new ArrayList<>();
        List<Integer> regions = new ArrayList<>();
        IMatrix decisionSurface = Linalg.zeros(observationSpace.size(), 1);
        
        // 对每个观测点计算最优决策
        for (int i = 0; i < observationSpace.size(); i++) {
            IVector observation = observationSpace.get(i);
            
            // 计算后验概率
            IVector posteriorProbs = computePosteriorProbabilities(
                observation, likelihoodFunctions, priorProbabilities);
            
            // 最小化贝叶斯风险
            DecisionResult result = minimizeBayesianRisk(posteriorProbs, lossFunction, numActions);
            
            regions.add(result.getOptimalAction());
            decisionSurface.set(i, 0, result.getOptimalAction());
            
            // 检查是否为边界点（相邻点有不同决策）
            if (i > 0 && !regions.get(i).equals(regions.get(i - 1))) {
                boundaryPoints.add(copyVector(observation));
            }
        }
        
        Map<String, Object> boundaryInfo = new HashMap<>();
        boundaryInfo.put("numBoundaryPoints", boundaryPoints.size());
        boundaryInfo.put("numRegions", new HashSet<>(regions).size());
        
        return new DecisionBoundaryResult(boundaryPoints, regions, decisionSurface, boundaryInfo);
    }
    
    /**
     * 序贯决策
     * Sequential Decision Making
     * 
     * @param observations 观测序列
     * @param likelihoodFunctions 似然函数列表
     * @param priorProbabilities 先验概率
     * @param lossFunction 损失函数
     * @param stopCriterion 停止准则
     * @return 决策序列
     */
    public static List<DecisionResult> sequentialDecision(
            List<IVector> observations,
            List<Function<IVector, Double>> likelihoodFunctions,
            IVector priorProbabilities,
            LossFunction lossFunction,
            Function<DecisionResult, Boolean> stopCriterion) {
        
        List<DecisionResult> decisions = new ArrayList<>();
        IVector currentPrior = copyVector(priorProbabilities);
        
        for (IVector observation : observations) {
            // 计算后验概率
            IVector posteriorProbs = computePosteriorProbabilities(
                observation, likelihoodFunctions, currentPrior);
            
            // 做决策
            DecisionResult result = minimizeBayesianRisk(
                posteriorProbs, lossFunction, likelihoodFunctions.size());
            
            decisions.add(result);
            
            // 检查停止准则
            if (stopCriterion.apply(result)) {
                break;
            }
            
            // 更新先验（后验变为下一步的先验）
            currentPrior = posteriorProbs;
        }
        
        return decisions;
    }
    
    /**
     * 计算后验概率
     * Compute Posterior Probabilities
     */
    private static IVector computePosteriorProbabilities(
            IVector observation,
            List<Function<IVector, Double>> likelihoodFunctions,
            IVector priorProbabilities) {
        
        int numStates = likelihoodFunctions.size();
        IVector posteriorProbs = Linalg.vector(numStates);
        double evidence = 0.0;
        
        // 计算未归一化的后验概率
        for (int state = 0; state < numStates; state++) {
            double likelihood = likelihoodFunctions.get(state).apply(observation);
            double prior = priorProbabilities.get(state).doubleValue();
            double unnormalizedPosterior = likelihood * prior;
            posteriorProbs.set(state, unnormalizedPosterior);
            evidence += unnormalizedPosterior;
        }
        
        // 归一化
        if (evidence > 0) {
            for (int state = 0; state < numStates; state++) {
                double normalizedProb = posteriorProbs.get(state).doubleValue() / evidence;
                posteriorProbs.set(state, normalizedProb);
            }
        }
        
        return posteriorProbs;
    }
    
    /**
     * 常用损失函数
     * Common Loss Functions
     */
    public static class LossFunctions {
        
        /**
         * 0-1损失函数
         * 0-1 Loss Function
         */
        public static final LossFunction ZERO_ONE_LOSS = (trueState, action) -> 
            trueState == action ? 0.0 : 1.0;
        
        /**
         * 平方损失函数
         * Squared Loss Function
         */
        public static final LossFunction SQUARED_LOSS = (trueState, action) -> {
            double diff = trueState - action;
            return diff * diff;
        };
        
        /**
         * 绝对损失函数
         * Absolute Loss Function
         */
        public static final LossFunction ABSOLUTE_LOSS = (trueState, action) -> 
            Math.abs(trueState - action);
        
        /**
         * 非对称损失函数
         * Asymmetric Loss Function
         */
        public static LossFunction asymmetricLoss(double underestimationCost, double overestimationCost) {
            return (trueState, action) -> {
                double diff = action - trueState;
                return diff >= 0 ? overestimationCost * diff : underestimationCost * (-diff);
            };
        }
        
        /**
         * 自定义损失矩阵
         * Custom Loss Matrix
         */
        public static LossFunction matrixLoss(IMatrix lossMatrix) {
            return (trueState, action) -> {
                if (trueState >= 0 && trueState < lossMatrix.rows() &&
                    action >= 0 && action < lossMatrix.cols()) {
                    return lossMatrix.get(trueState, action).doubleValue();
                }
                return Double.POSITIVE_INFINITY;
            };
        }
    }
    
    /**
     * 常用效用函数
     * Common Utility Functions
     */
    public static class UtilityFunctions {
        
        /**
         * 线性效用函数
         * Linear Utility Function
         */
        public static UtilityFunction linearUtility(double slope, double intercept) {
            return (trueState, action) -> slope * action + intercept;
        }
        
        /**
         * 对数效用函数
         * Logarithmic Utility Function
         */
        public static UtilityFunction logarithmicUtility(double scale) {
            return (trueState, action) -> {
                if (action <= 0) return Double.NEGATIVE_INFINITY;
                return scale * Math.log(action);
            };
        }
        
        /**
         * 指数效用函数
         * Exponential Utility Function
         */
        public static UtilityFunction exponentialUtility(double riskAversion) {
            return (trueState, action) -> -Math.exp(-riskAversion * action);
        }
        
        /**
         * 幂效用函数
         * Power Utility Function
         */
        public static UtilityFunction powerUtility(double gamma) {
            return (trueState, action) -> {
                if (action <= 0) return Double.NEGATIVE_INFINITY;
                if (gamma == 1.0) return Math.log(action);
                return Math.pow(action, gamma) / gamma;
            };
        }
        
        /**
         * 从损失函数转换为效用函数
         * Convert loss function to utility function
         */
        public static UtilityFunction fromLossFunction(LossFunction lossFunction) {
            return (trueState, action) -> -lossFunction.loss(trueState, action);
        }
    }
    
    /**
     * 决策规则工厂
     * Decision Rule Factory
     */
    public static class DecisionRules {
        
        /**
         * 贝叶斯决策规则
         * Bayesian Decision Rule
         */
        public static DecisionRule bayesianRule(
                List<Function<IVector, Double>> likelihoodFunctions,
                IVector priorProbabilities,
                LossFunction lossFunction) {
            
            return observation -> {
                IVector posteriorProbs = computePosteriorProbabilities(
                    observation, likelihoodFunctions, priorProbabilities);
                DecisionResult result = minimizeBayesianRisk(
                    posteriorProbs, lossFunction, likelihoodFunctions.size());
                return result.getOptimalAction();
            };
        }
        
        /**
         * 最大后验概率决策规则
         * Maximum A Posteriori Decision Rule
         */
        public static DecisionRule mapRule(
                List<Function<IVector, Double>> likelihoodFunctions,
                IVector priorProbabilities) {
            
            return observation -> {
                IVector posteriorProbs = computePosteriorProbabilities(
                    observation, likelihoodFunctions, priorProbabilities);
                
                int maxIndex = 0;
                double maxProb = posteriorProbs.get(0).doubleValue();
                
                for (int i = 1; i < posteriorProbs.size(); i++) {
                    double prob = posteriorProbs.get(i).doubleValue();
                    if (prob > maxProb) {
                        maxProb = prob;
                        maxIndex = i;
                    }
                }
                
                return maxIndex;
            };
        }
        
        /**
         * 最大似然决策规则
         * Maximum Likelihood Decision Rule
         */
        public static DecisionRule mlRule(List<Function<IVector, Double>> likelihoodFunctions) {
            return observation -> {
                int maxIndex = 0;
                double maxLikelihood = likelihoodFunctions.get(0).apply(observation);
                
                for (int i = 1; i < likelihoodFunctions.size(); i++) {
                    double likelihood = likelihoodFunctions.get(i).apply(observation);
                    if (likelihood > maxLikelihood) {
                        maxLikelihood = likelihood;
                        maxIndex = i;
                    }
                }
                
                return maxIndex;
            };
        }
    }
    
    /**
     * 复制向量
     * Copy vector
     */
    private static IVector copyVector(IVector vector) {
        IVector copy = Linalg.vector(vector.size());
        for (int i = 0; i < vector.size(); i++) {
            copy.set(i, vector.get(i));
        }
        return copy;
    }
}