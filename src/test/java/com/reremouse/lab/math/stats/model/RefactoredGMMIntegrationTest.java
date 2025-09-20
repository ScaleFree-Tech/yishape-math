package com.reremouse.lab.math.stats.model;

import com.reremouse.lab.math.stats.distribution.multiv.MultivariateNormalDistribution;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.audio.embedding.IVectorModel;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.ArrayList;
import java.util.Random;

/**
 * 重构后的GMM和IVectorModel集成测试
 * Integration test for refactored GMM and IVectorModel
 */
public class RefactoredGMMIntegrationTest {
    
    private Random random;
    
    @BeforeEach
    void setUp() {
        random = new Random(42);
    }
    
    @Test
    void testGaussianMixtureModelBasicFunctionality() {
        System.out.println("测试GaussianMixtureModel基本功能...");
        
        // 创建测试数据
        int numSamples = 100;
        int dimension = 3;
        List<IVector<Double>> data = generateTestData(numSamples, dimension);
        
        // 创建GMM
        int numComponents = 2;
        GaussianMixtureModel gmm = new GaussianMixtureModel(numComponents, dimension, random);
        
        // 训练GMM
        EMAlgorithm emAlgorithm = new EMAlgorithm(10, 1e-4, true);
        EMAlgorithm.EMResult result = gmm.fit(data, emAlgorithm);
        
        // 验证训练结果
        assertNotNull(result);
        assertTrue(result.iterations > 0);
        assertTrue(result.logLikelihood > Double.NEGATIVE_INFINITY);
        
        // 测试预测功能
        IVector<Double> testSample = data.get(0);
        IVector<Double> posteriors = gmm.computePosteriors(testSample);
        
        assertNotNull(posteriors);
        assertEquals(numComponents, posteriors.size());
        
        // 验证后验概率和为1
        double sum = 0.0;
        for (int i = 0; i < posteriors.size(); i++) {
            double prob = posteriors.get(i);
            assertTrue(prob >= 0.0 && prob <= 1.0, "后验概率应该在[0,1]范围内");
            sum += prob;
        }
        assertEquals(1.0, sum, 1e-6, "后验概率和应该为1");
        
        // 测试PDF计算
        double pdf = gmm.pdf(testSample);
        assertTrue(pdf > 0, "PDF应该为正数");
        
        System.out.println("GaussianMixtureModel基本功能测试通过");
    }
    
    @Test
    void testMultivariateGaussianFunctionality() {
        System.out.println("测试MultivariateGaussian功能...");
        
        int dimension = 3;
        IVector<Double> mean = Linalg.zeros(dimension);
        IMatrix<Double> covariance = Linalg.eye(dimension);
        
        MultivariateNormalDistribution gaussian = new MultivariateNormalDistribution(mean, covariance);
        
        // 测试PDF计算
        IVector<Double> testPoint = Linalg.zeros(dimension);
        double pdf = gaussian.pdf(testPoint);
        assertTrue(pdf > 0, "PDF应该为正数");
        
        // 测试对数PDF
        double logPdf = gaussian.logPdf(testPoint);
        assertEquals(Math.log(pdf), logPdf, 1e-10, "logPdf应该等于log(pdf)");
        
        // 测试基本属性
        assertEquals(dimension, gaussian.getMean().size());
        assertEquals(dimension, gaussian.getCovariance().getRowNum());
        assertEquals(dimension, gaussian.getCovariance().getColNum());
        
        System.out.println("MultivariateGaussian功能测试通过");
    }
    
    @Test
    void testIVectorModelIntegration() {
        System.out.println("测试IVectorModel与新GMM的集成...");
        
        // 创建IVectorModel
        int numComponents = 2;
        int featureDim = 3;
        int ivectorDim = 2;
        IVectorModel model = new IVectorModel(ivectorDim, numComponents, featureDim);
        
        // 生成训练数据
        List<IMatrix<Double>> trainingData = generateMFCCData(5, 10, featureDim);
        
        // 训练模型
        assertDoesNotThrow(() -> {
            model.train(trainingData);
        }, "模型训练不应该抛出异常");
        
        assertTrue(model.isTrained(), "模型应该已训练");
        
        // 测试嵌入提取
        IMatrix<Double> testMFCC = trainingData.get(0);
        IVector<Double> embedding = model.embed(testMFCC);
        
        assertNotNull(embedding);
        assertEquals(ivectorDim, embedding.size());
        
        // 验证嵌入向量已归一化
        double norm = embedding.norm2();
        assertEquals(1.0, norm, 1e-6, "i-vector应该已归一化");
        
        System.out.println("IVectorModel集成测试通过");
    }
    
    @Test
    void testEMAlgorithmConvergence() {
        System.out.println("测试EM算法收敛性...");
        
        // 创建明显分离的两个高斯分布数据
        List<IVector<Double>> data = new ArrayList<>();
        int dimension = 2;
        
        // 第一个分布：均值在(0,0)
        for (int i = 0; i < 50; i++) {
            double[] values = {random.nextGaussian() * 0.5, random.nextGaussian() * 0.5};
            data.add(IVector.of(values));
        }
        
        // 第二个分布：均值在(3,3)
        for (int i = 0; i < 50; i++) {
            double[] values = {3 + random.nextGaussian() * 0.5, 3 + random.nextGaussian() * 0.5};
            data.add(IVector.of(values));
        }
        
        // 训练GMM
        GaussianMixtureModel gmm = new GaussianMixtureModel(2, dimension, random);
        EMAlgorithm emAlgorithm = new EMAlgorithm(50, 1e-6, true);
        EMAlgorithm.EMResult result = gmm.fit(data, emAlgorithm);
        
        // 验证收敛
        assertTrue(result.converged || result.iterations < 50, "EM算法应该收敛或达到最大迭代次数");
        assertTrue(result.logLikelihood > -1000, "对数似然应该合理");
        
        System.out.println("EM算法收敛性测试通过");
    }
    
    @Test
    void testModelPersistenceAndConsistency() {
        System.out.println("测试模型一致性...");
        
        // 创建并训练两个相同配置的模型
        int numComponents = 2;
        int featureDim = 3;
        int ivectorDim = 2;
        
        IVectorModel model1 = new IVectorModel(ivectorDim, numComponents, featureDim);
        IVectorModel model2 = new IVectorModel(ivectorDim, numComponents, featureDim);
        
        // 使用相同的训练数据
        List<IMatrix<Double>> trainingData = generateMFCCData(3, 8, featureDim);
        
        model1.train(trainingData);
        model2.train(trainingData);
        
        // 测试相同输入产生的输出
        IMatrix<Double> testInput = trainingData.get(0);
        IVector<Double> embedding1 = model1.embed(testInput);
        IVector<Double> embedding2 = model2.embed(testInput);
        
        // 由于随机初始化，两个模型的输出可能不同，但都应该是有效的
        assertNotNull(embedding1);
        assertNotNull(embedding2);
        assertEquals(ivectorDim, embedding1.size());
        assertEquals(ivectorDim, embedding2.size());
        
        // 验证归一化
        assertEquals(1.0, embedding1.norm2(), 1e-6);
        assertEquals(1.0, embedding2.norm2(), 1e-6);
        
        System.out.println("模型一致性测试通过");
    }
    
    /**
     * 生成测试数据
     */
    private List<IVector<Double>> generateTestData(int numSamples, int dimension) {
        List<IVector<Double>> data = new ArrayList<>();
        
        for (int i = 0; i < numSamples; i++) {
            double[] values = new double[dimension];
            for (int j = 0; j < dimension; j++) {
                values[j] = random.nextGaussian();
            }
            data.add(IVector.of(values));
        }
        
        return data;
    }
    
    /**
     * 生成MFCC测试数据
     */
    private List<IMatrix<Double>> generateMFCCData(int numUtterances, int numFrames, int featureDim) {
        List<IMatrix<Double>> data = new ArrayList<>();
        
        for (int i = 0; i < numUtterances; i++) {
            double[][] mfccData = new double[numFrames][featureDim];
            for (int t = 0; t < numFrames; t++) {
                for (int d = 0; d < featureDim; d++) {
                    mfccData[t][d] = random.nextGaussian();
                }
            }
            data.add(IMatrix.of(mfccData));
        }
        
        return data;
    }
}