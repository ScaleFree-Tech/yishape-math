package com.yishape.lab.math.ml.cls;

import com.yishape.lab.math.ml.clf.ensemble.EnsembleClassifier;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.clf.tree.RandomForestHyperparameterOptimizer;
import com.yishape.lab.math.ml.clf.tree.RereRandomForest;
import com.yishape.lab.math.ml.clf.tree.RFTree;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 优化后的随机森林测试类
 * 测试超参数优化器和集成分类器的功能
 */
public class OptimizedRandomForestTest {
    
    private IMatrix trainFeatures;
    private String[] trainLabels;
    private IMatrix testFeatures;
    private String[] testLabels;
    
    @BeforeEach
    public void setUp() {
        // 创建简单的测试数据集
        double[][] trainData = {
            {1.0, 2.0, 3.0},
            {2.0, 3.0, 4.0},
            {3.0, 4.0, 5.0},
            {4.0, 5.0, 6.0},
            {5.0, 6.0, 7.0},
            {6.0, 7.0, 8.0},
            {7.0, 8.0, 9.0},
            {8.0, 9.0, 10.0},
            {1.5, 2.5, 3.5},
            {2.5, 3.5, 4.5},
            {3.5, 4.5, 5.5},
            {4.5, 5.5, 6.5},
            {5.5, 6.5, 7.5},
            {6.5, 7.5, 8.5},
            {7.5, 8.5, 9.5},
            {8.5, 9.5, 10.5}
        };
        
        trainFeatures = Linalg.matrix(trainData);
        trainLabels = new String[]{
            "A", "A", "A", "A", "B", "B", "B", "B",
            "A", "A", "A", "A", "B", "B", "B", "B"
        };
        
        double[][] testData = {
            {1.2, 2.2, 3.2},
            {4.2, 5.2, 6.2},
            {7.2, 8.2, 9.2},
            {2.8, 3.8, 4.8}
        };
        
        testFeatures = Linalg.matrix(testData);
        testLabels = new String[]{"A", "A", "B", "A"};
    }
    
    @Test
    public void testHyperparameterOptimizerWithAdam() {
        System.out.println("Testing Hyperparameter Optimizer with Adam...");
        
        // 创建超参数边界
        RandomForestHyperparameterOptimizer.HyperparameterBounds bounds =
            new RandomForestHyperparameterOptimizer.HyperparameterBounds(
                10, 50, 3, 8, 2, 10, 1, 5, 0.3, 1.0
            );
        
        // 创建优化器
        RandomForestHyperparameterOptimizer optimizer = 
            new RandomForestHyperparameterOptimizer(
                trainFeatures, trainLabels, testFeatures, testLabels,
                bounds, 42L, 3
            );
        
        // 使用Adam优化
        RandomForestHyperparameterOptimizer.OptimizationResult result = 
            optimizer.optimizeWithAdam(10, 0.01);
        
        assertNotNull(result);
        assertNotNull(result.bestModel);
        assertTrue(result.bestScore >= 0.0 && result.bestScore <= 1.0);
        assertTrue(result.bestNEstimators >= bounds.minEstimators && 
                  result.bestNEstimators <= bounds.maxEstimators);
        assertTrue(result.bestMaxDepth >= bounds.minDepth && 
                  result.bestMaxDepth <= bounds.maxDepth);
        
        System.out.println("Best parameters: nEstimators=" + result.bestNEstimators +
                          ", maxDepth=" + result.bestMaxDepth +
                          ", score=" + String.format("%.3f", result.bestScore));
    }
    
    @Test
    public void testHyperparameterOptimizerWithLBFGS() {
        System.out.println("Testing Hyperparameter Optimizer with LBFGS...");
        
        // 创建超参数边界
        RandomForestHyperparameterOptimizer.HyperparameterBounds bounds = 
            new RandomForestHyperparameterOptimizer.HyperparameterBounds(
                10, 30, 3, 6, 2, 8, 1, 3, 0.5, 1.0
            );
        
        // 创建优化器
        RandomForestHyperparameterOptimizer optimizer = 
            new RandomForestHyperparameterOptimizer(
                trainFeatures, trainLabels, testFeatures, testLabels,
                bounds, 42L, 3
            );
        
        // 使用LBFGS优化
        RandomForestHyperparameterOptimizer.OptimizationResult result = 
            optimizer.optimizeWithLBFGS(5);
        
        assertNotNull(result);
        assertNotNull(result.bestModel);
        assertTrue(result.bestScore >= 0.0 && result.bestScore <= 1.0);
        
        System.out.println("Best parameters: nEstimators=" + result.bestNEstimators +
                          ", maxDepth=" + result.bestMaxDepth +
                          ", score=" + String.format("%.3f", result.bestScore));
    }
    
    @Test
    public void testHyperparameterOptimizerWithGridSearch() {
        System.out.println("Testing Hyperparameter Optimizer with Grid Search...");
        
        // 创建超参数边界
        RandomForestHyperparameterOptimizer.HyperparameterBounds bounds = 
            new RandomForestHyperparameterOptimizer.HyperparameterBounds(
                10, 20, 3, 5, 2, 4, 1, 2, 0.5, 1.0
            );
        
        // 创建优化器
        RandomForestHyperparameterOptimizer optimizer = 
            new RandomForestHyperparameterOptimizer(
                trainFeatures, trainLabels, testFeatures, testLabels,
                bounds, 42L, 3
            );
        
        // 使用网格搜索优化（使用小网格以节省时间）
        RandomForestHyperparameterOptimizer.OptimizationResult result = 
            optimizer.optimizeWithGridSearch(2);
        
        assertNotNull(result);
        assertNotNull(result.bestModel);
        assertTrue(result.bestScore >= 0.0 && result.bestScore <= 1.0);
        
        System.out.println("Best parameters: nEstimators=" + result.bestNEstimators +
                          ", maxDepth=" + result.bestMaxDepth +
                          ", score=" + String.format("%.3f", result.bestScore));
    }
    
    @Test
    public void testEnsembleClassifierVoting() {
        System.out.println("Testing Ensemble Classifier with Voting...");
        
        // 创建集成分类器
        EnsembleClassifier ensemble = new EnsembleClassifier(
            EnsembleClassifier.EnsembleStrategy.VOTING, 42L
        );
        
        // 训练
        ensemble.fit(trainFeatures, trainLabels);
        assertTrue(ensemble.isTrained());
        
        // 预测
        EnsembleClassifier.EnsembleResult result = ensemble.predict(testFeatures);
        
        assertNotNull(result);
        assertNotNull(result.predictions);
        assertNotNull(result.probabilities);
        assertEquals(testFeatures.rows(), result.predictions.length);
        Assertions.assertEquals(testFeatures.rows(), result.probabilities.rows());
        
        // 验证预测结果
        for (String prediction : result.predictions) {
            assertTrue(prediction.equals("A") || prediction.equals("B"));
        }
        
        System.out.println("Ensemble predictions: " + String.join(", ", result.predictions));
    }
    
    @Test
    public void testEnsembleClassifierWeightedVoting() {
        System.out.println("Testing Ensemble Classifier with Weighted Voting...");
        
        // 创建带权重的集成分类器
        double[] weights = {0.5, 0.3, 0.2}; // RF, LR, XGB
        EnsembleClassifier ensemble = new EnsembleClassifier(
            EnsembleClassifier.EnsembleStrategy.WEIGHTED_VOTING, weights, 42L
        );
        
        // 训练
        ensemble.fit(trainFeatures, trainLabels);
        assertTrue(ensemble.isTrained());
        
        // 验证权重
        assertNotNull(ensemble.getClassifierWeights());
        Assertions.assertEquals(3, ensemble.getClassifierWeights().size());
        
        // 预测
        EnsembleClassifier.EnsembleResult result = ensemble.predict(testFeatures);
        
        assertNotNull(result);
        assertNotNull(result.predictions);
        assertEquals(testFeatures.rows(), result.predictions.length);
        
        System.out.println("Weighted ensemble predictions: " + String.join(", ", result.predictions));
        System.out.println("Classifier weights: " + ensemble.getClassifierWeights());
    }
    
    @Test
    public void testEnsembleClassifierStacking() {
        System.out.println("Testing Ensemble Classifier with Stacking...");
        
        // 创建堆叠集成分类器
        EnsembleClassifier ensemble = new EnsembleClassifier(
            EnsembleClassifier.EnsembleStrategy.STACKING, 42L
        );
        
        // 训练
        ensemble.fit(trainFeatures, trainLabels);
        assertTrue(ensemble.isTrained());
        
        // 预测
        EnsembleClassifier.EnsembleResult result = ensemble.predict(testFeatures);
        
        assertNotNull(result);
        assertNotNull(result.predictions);
        assertNotNull(result.probabilities);
        assertEquals(testFeatures.rows(), result.predictions.length);
        
        System.out.println("Stacking ensemble predictions: " + String.join(", ", result.predictions));
    }
    
    @Test
    public void testOptimizedRandomForestPerformance() {
        System.out.println("Testing Optimized Random Forest Performance...");
        
        // 创建标准随机森林
        RereRandomForest standardRF = new RereRandomForest(
            50, 10, 2, 1, -1, true, 
            RFTree.SplitCriterion.GINI, 42L
        );
        
        // 训练标准模型
        long startTime = System.currentTimeMillis();
        standardRF.fit(trainFeatures, trainLabels);
        long standardTrainTime = System.currentTimeMillis() - startTime;
        
        // 预测
        startTime = System.currentTimeMillis();
        String[] standardPredictions = standardRF.predictBatch(testFeatures);
        long standardPredictTime = System.currentTimeMillis() - startTime;
        
        // 创建优化后的随机森林（通过超参数优化）
        RandomForestHyperparameterOptimizer.HyperparameterBounds bounds = 
            new RandomForestHyperparameterOptimizer.HyperparameterBounds(
                20, 60, 5, 12, 2, 8, 1, 4, 0.3, 1.0
            );
        
        RandomForestHyperparameterOptimizer optimizer = 
            new RandomForestHyperparameterOptimizer(
                trainFeatures, trainLabels, testFeatures, testLabels,
                bounds, 42L, 3
            );
        
        startTime = System.currentTimeMillis();
        RandomForestHyperparameterOptimizer.OptimizationResult optimizedResult = 
            optimizer.optimizeWithAdam(5, 0.01);
        long optimizedTrainTime = System.currentTimeMillis() - startTime;
        
        // 预测
        startTime = System.currentTimeMillis();
        String[] optimizedPredictions = optimizedResult.bestModel.predictBatch(testFeatures);
        long optimizedPredictTime = System.currentTimeMillis() - startTime;
        
        // 输出性能比较
        System.out.println("Standard RF - Train time: " + standardTrainTime + "ms, Predict time: " + standardPredictTime + "ms");
        System.out.println("Optimized RF - Train time: " + optimizedTrainTime + "ms, Predict time: " + optimizedPredictTime + "ms");
        System.out.println("Standard predictions: " + String.join(", ", standardPredictions));
        System.out.println("Optimized predictions: " + String.join(", ", optimizedPredictions));
        System.out.println("Optimized score: " + String.format("%.3f", optimizedResult.bestScore));
        
        // 验证预测结果
        assertNotNull(standardPredictions);
        assertNotNull(optimizedPredictions);
        assertEquals(testFeatures.rows(), standardPredictions.length);
        assertEquals(testFeatures.rows(), optimizedPredictions.length);
    }
    
    @Test
    public void testIntegrationWithAllOptimizations() {
        System.out.println("Testing Integration with All Optimizations...");
        
        // 1. 使用超参数优化器找到最佳参数
        RandomForestHyperparameterOptimizer.HyperparameterBounds bounds = 
            new RandomForestHyperparameterOptimizer.HyperparameterBounds(
                10, 30, 3, 8, 2, 6, 1, 3, 0.5, 1.0
            );
        
        RandomForestHyperparameterOptimizer optimizer = 
            new RandomForestHyperparameterOptimizer(
                trainFeatures, trainLabels, testFeatures, testLabels,
                bounds, 42L, 3
            );
        
        RandomForestHyperparameterOptimizer.OptimizationResult optimizedResult = 
            optimizer.optimizeWithAdam(3, 0.01);
        
        // 2. 使用集成分类器结合多种算法
        EnsembleClassifier ensemble = new EnsembleClassifier(
            EnsembleClassifier.EnsembleStrategy.WEIGHTED_VOTING, 42L
        );
        
        ensemble.fit(trainFeatures, trainLabels);
        EnsembleClassifier.EnsembleResult ensembleResult = ensemble.predict(testFeatures);
        
        // 3. 比较结果
        String[] optimizedPredictions = optimizedResult.bestModel.predictBatch(testFeatures);
        
        System.out.println("Optimized RF predictions: " + String.join(", ", optimizedPredictions));
        System.out.println("Ensemble predictions: " + String.join(", ", ensembleResult.predictions));
        System.out.println("Optimized RF score: " + String.format("%.3f", optimizedResult.bestScore));
        System.out.println("Best hyperparameters: nEstimators=" + optimizedResult.bestNEstimators +
                          ", maxDepth=" + optimizedResult.bestMaxDepth +
                          ", minSamplesSplit=" + optimizedResult.bestMinSamplesSplit);
        
        // 验证所有结果
        assertNotNull(optimizedPredictions);
        assertNotNull(ensembleResult.predictions);
        assertTrue(optimizedResult.bestScore >= 0.0 && optimizedResult.bestScore <= 1.0);
        assertEquals(testFeatures.rows(), optimizedPredictions.length);
        assertEquals(testFeatures.rows(), ensembleResult.predictions.length);
        
        System.out.println("Integration test completed successfully!");
    }
}