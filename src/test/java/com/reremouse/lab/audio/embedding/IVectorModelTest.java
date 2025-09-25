package com.reremouse.lab.audio.embedding;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.List;

/**
 * IVectorEmbedding 基本功能测试
 */
public class IVectorModelTest {
    
    private IVectorEmbedding model;
    
    @BeforeEach
    public void setUp() {
        // 创建一个小的模型用于测试
        model = new IVectorEmbedding(10, 4, 3); // 10维i-vector，4个高斯分量，3维特征
    }
    
    @Test
    public void testModelCreation() {
        assertNotNull(model);
        assertEquals(10, model.getIVectorDimension());
        assertEquals(4, model.getNumComponents());
        assertEquals(3, model.getFeatureDimension());
        assertFalse(model.isTrained());
    }
    
    @Test
    public void testEmbedWithoutTraining() {
        // 创建测试数据
        double[][] testData = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0}
        };
        IMatrix<Double> mfcc = Linalg.matrix(testData);
        
        // 在未训练的模型上调用embed应该抛出异常
        assertThrows(IllegalStateException.class, () -> {
            model.embed(mfcc);
        });
    }
    
    @Test
    public void testTrainingDataPreparation() {
        // 创建训练数据
        List<IMatrix<Double>> trainingData = new ArrayList<>();
        
        // 添加几个MFCC矩阵
        for (int i = 0; i < 3; i++) {
            double[][] data = new double[5][3]; // 5帧，3维特征
            for (int j = 0; j < 5; j++) {
                for (int k = 0; k < 3; k++) {
                    data[j][k] = Math.random();
                }
            }
            trainingData.add(Linalg.matrix(data));
        }
        
        // 训练应该不抛出异常
        assertDoesNotThrow(() -> {
            model.train(trainingData);
        });
        
        // 训练后模型应该标记为已训练
        assertTrue(model.isTrained());
    }
}