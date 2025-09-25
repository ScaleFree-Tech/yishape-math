package com.reremouse.lab.audio.embedding;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * OnlineIVectorEmbedding 测试类
 OnlineIVectorEmbedding Test Class
 */
public class OnlineIVectorModelTest {
    
    private OnlineIVectorEmbedding model;
    private Random random;
    
    @BeforeEach
    public void setUp() {
        // 创建一个小型模型用于测试，使用较少的高斯分量以避免初始化问题
        model = new OnlineIVectorEmbedding(10, 2, 3, true, 0.001); // 10维i-vector，2个高斯分量，3维特征
        random = new Random(42);
    }
    
    @Test
    public void testModelCreation() {
        assertNotNull(model);
        assertEquals(10, model.getIVectorDimension());
        assertEquals(2, model.getNumComponents()); // Updated to match actual number of components
        assertEquals(3, model.getFeatureDimension());
        assertFalse(model.isTrained());
    }
    
    @Test
    public void testIncrementalTraining() {
        // 创建测试数据
        List<IMatrix<Double>> trainingBatches = new ArrayList<>();
        
        // 生成5个批次的训练数据，每个批次有足够的数据点来支持2个高斯分量
        for (int i = 0; i < 5; i++) {
            double[][] batchData = new double[20][3]; // 20帧，3维特征，足够支持2个高斯分量
            for (int r = 0; r < 20; r++) {
                for (int c = 0; c < 3; c++) {
                    batchData[r][c] = random.nextGaussian();
                }
            }
            trainingBatches.add(Linalg.matrix(batchData));
        }
        
        // 进行增量训练
        for (IMatrix<Double> batch : trainingBatches) {
            model.trainIncremental(batch);
        }
        
        // 完成训练
        model.finishTraining();
        
        // 验证模型状态
        assertTrue(model.isTrained());
        assertEquals(100, model.getProcessedSamples()); // 5 batches * 20 samples each
    }
    
    @Test
    public void testEmbeddingGeneration() {
        // 创建测试数据，确保有足够的数据点来初始化2个高斯分量
        double[][] testData = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0},
            {2.0, 3.0, 4.0},
            {5.0, 6.0, 7.0},
            {8.0, 9.0, 10.0}
        };
        IMatrix<Double> mfcc = Linalg.matrix(testData);
        
        // 在未训练的模型上调用embed应该抛出异常
        assertThrows(IllegalStateException.class, () -> {
            model.embed(mfcc);
        });
        
        // 训练模型
        model.trainIncremental(mfcc);
        model.finishTraining();
        
        // 现在应该可以生成嵌入
        IVector<Double> embedding = model.embed(mfcc);
        assertNotNull(embedding);
        assertEquals(10, embedding.length()); // 应该匹配i-vector维度
        
        // 验证嵌入向量已归一化
        double norm = embedding.norm2();
        assertEquals(1.0, norm, 1e-6, "嵌入向量应该已归一化 / Embedding vector should be normalized");
    }
    
    @Test
    public void testBatchEmbedding() {
        // 创建训练数据，确保有足够的数据点
        double[][] trainData = {
            {1.0, 1.0, 1.0},
            {2.0, 2.0, 2.0},
            {3.0, 3.0, 3.0},
            {1.5, 1.5, 1.5},
            {2.5, 2.5, 2.5},
            {3.5, 3.5, 3.5}
        };
        IMatrix<Double> trainMfcc = Linalg.matrix(trainData);
        
        // 训练模型
        model.trainIncremental(trainMfcc);
        model.finishTraining();
        
        // 创建测试批次
        IMatrix<Double>[] testBatch = new IMatrix[3];
        for (int i = 0; i < 3; i++) {
            double[][] testData = {
                {1.0 + i, 2.0 + i, 3.0 + i},
                {4.0 + i, 5.0 + i, 6.0 + i},
                {2.0 + i, 3.0 + i, 4.0 + i}
            };
            testBatch[i] = Linalg.matrix(testData);
        }
        
        // 批量生成嵌入
        IMatrix<Double> embeddings = model.embedBatch(testBatch);
        assertNotNull(embeddings);
        assertEquals(3, embeddings.getRowNum()); // 应该有3个嵌入向量
        assertEquals(10, embeddings.getColNum()); // 每个嵌入向量应该是10维
    }
    
    @Test
    public void testSimilarityCalculation() {
        // 创建训练数据，确保有足够的数据点
        double[][] trainData = {
            {1.0, 1.0, 1.0},
            {2.0, 2.0, 2.0},
            {3.0, 3.0, 3.0},
            {1.5, 1.5, 1.5},
            {2.5, 2.5, 2.5},
            {3.5, 3.5, 3.5}
        };
        IMatrix<Double> trainMfcc = Linalg.matrix(trainData);
        
        // 训练模型
        model.trainIncremental(trainMfcc);
        model.finishTraining();
        
        // 生成两个嵌入向量
        double[][] testData1 = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {2.0, 3.0, 4.0}
        };
        double[][] testData2 = {
            {1.5, 2.5, 3.5},
            {4.5, 5.5, 6.5},
            {2.5, 3.5, 4.5}
        };
        
        IMatrix<Double> mfcc1 = Linalg.matrix(testData1);
        IMatrix<Double> mfcc2 = Linalg.matrix(testData2);
        
        IVector<Double> embedding1 = model.embed(mfcc1);
        IVector<Double> embedding2 = model.embed(mfcc2);
        
        // 计算相似度
        double similarity = model.calculateSimilarity(embedding1, embedding2);
        assertTrue(similarity >= -1.0 && similarity <= 1.0, "相似度应该在[-1, 1]范围内 / Similarity should be in range [-1, 1]");
        
        // 计算距离
        double euclideanDistance = model.calculateDistance(embedding1, embedding2, IAudioEmbedding.DistanceType.EUCLIDEAN);
        assertTrue(euclideanDistance >= 0, "欧几里得距离应该非负 / Euclidean distance should be non-negative");
        
        double cosineDistance = model.calculateDistance(embedding1, embedding2, IAudioEmbedding.DistanceType.COSINE);
        assertTrue(cosineDistance >= 0 && cosineDistance <= 2.0, "余弦距离应该在[0, 2]范围内 / Cosine distance should be in range [0, 2]");
    }
    
    @Test
    public void testParameterConfiguration() {
        // 测试使用SGD优化器
        OnlineIVectorEmbedding sgdModel = new OnlineIVectorEmbedding(5, 2, 2, false, 0.01);
        assertNotNull(sgdModel);
        assertEquals(5, sgdModel.getIVectorDimension());
        
        
        // Note: We can't directly verify the batch size as it's a private field
        
        // 测试参数设置
        java.util.Map<String, Object> params = new java.util.HashMap<>();
        params.put("testParam", "testValue");
        sgdModel.setParameters(params);
        assertEquals(params, sgdModel.getParameters());
    }
    
    // 添加一个示例运行测试
    @Test
    public void testExampleRun() {
        // This is just a simple test to ensure the example code works
        // We won't run the full example, just a small portion of it
        
        // 创建在线i-vector模型
        OnlineIVectorEmbedding model = new OnlineIVectorEmbedding(10, 2, 3, true, 0.001);
        
        // 创建一些测试数据
        double[][] testData = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0},
            {2.0, 3.0, 4.0},
            {5.0, 6.0, 7.0},
            {8.0, 9.0, 10.0}
        };
        com.reremouse.lab.math.linalg.IMatrix<Double> mfcc = com.reremouse.lab.math.linalg.Linalg.matrix(testData);
        
        // 训练模型
        model.trainIncremental(mfcc);
        model.finishTraining();
        
        // 验证模型已训练
        assertTrue(model.isTrained());
        
        // 生成嵌入向量
        com.reremouse.lab.math.linalg.IVector<Double> embedding = model.embed(mfcc);
        assertNotNull(embedding);
        assertEquals(10, embedding.length());
    }
}