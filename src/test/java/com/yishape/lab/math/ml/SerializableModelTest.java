package com.yishape.lab.math.ml;

import com.yishape.lab.math.ml.clu.KMeansPlusPlus;
import com.yishape.lab.math.ml.clf.lr.RereLogisticRegression;
import com.yishape.lab.math.ml.dr.RerePCA;
import com.yishape.lab.math.ml.reg.RereLinearRegression;
import com.yishape.lab.math.ml.clf.ensemble.EnsembleClassifier;
import com.yishape.lab.math.ml.clf.tree.RereRandomForest;
import com.yishape.lab.math.ml.clf.tree.RereXGboost;
import com.yishape.lab.math.ml.dr.RereSVD;
import com.yishape.lab.math.ml.dr.RereUMAP;
import com.yishape.lab.math.ml.dr.RereTSNE;

import org.junit.jupiter.api.Test;

import java.io.File;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * 测试ISerializableModel接口的save和load方法
 */
public class SerializableModelTest {
    
    private static final String TEST_DIR = "test_models";
    
    @Test
    public void testRereLinearRegressionSerialization() {
        // 创建测试目录
        createTestDirectory();
        
        // 创建模型实例
        RereLinearRegression model = new RereLinearRegression();
        
        // 保存路径
        String filePath = TEST_DIR + "/linear_regression_model.ser";
        
        // 测试保存
        model.save(filePath);
        
        // 验证文件存在
        assertTrue(new File(filePath).exists(), "模型文件应该存在");
        
        // 清理测试文件
        new File(filePath).delete();
    }
    
    @Test
    public void testRereLogisticRegressionSerialization() {
        // 创建测试目录
        createTestDirectory();
        
        // 创建模型实例
        RereLogisticRegression model = new RereLogisticRegression();
        
        // 保存路径
        String filePath = TEST_DIR + "/logistic_regression_model.ser";
        
        // 测试保存
        model.save(filePath);
        
        // 验证文件存在
        assertTrue(new File(filePath).exists(), "模型文件应该存在");
        
        // 清理测试文件
        new File(filePath).delete();
    }
    
    @Test
    public void testKMeansPlusPlusSerialization() {
        // 创建测试目录
        createTestDirectory();
        
        // 创建模型实例
        KMeansPlusPlus model = new KMeansPlusPlus();
        
        // 保存路径
        String filePath = TEST_DIR + "/kmeans_model.ser";
        
        // 测试保存
        model.save(filePath);
        
        // 验证文件存在
        assertTrue(new File(filePath).exists(), "模型文件应该存在");
        
        // 清理测试文件
        new File(filePath).delete();
    }
    
    @Test
    public void testRerePCASerialization() {
        // 创建测试目录
        createTestDirectory();
        
        // 创建模型实例
        RerePCA model = new RerePCA();
        
        // 保存路径
        String filePath = TEST_DIR + "/pca_model.ser";
        
        // 测试保存
        model.save(filePath);
        
        // 验证文件存在
        assertTrue(new File(filePath).exists(), "模型文件应该存在");
        
        // 清理测试文件
        new File(filePath).delete();
    }
    
    @Test
    public void testEnsembleClassifierSerialization() {
        // 创建测试目录
        createTestDirectory();
        
        // 创建模型实例
        EnsembleClassifier model = new EnsembleClassifier(
            EnsembleClassifier.EnsembleStrategy.VOTING, 42L);
        
        // 保存路径
        String filePath = TEST_DIR + "/ensemble_classifier_model.ser";
        
        // 测试保存
        model.save(filePath);
        
        // 验证文件存在
        assertTrue(new File(filePath).exists(), "模型文件应该存在");
        
        // 清理测试文件
        new File(filePath).delete();
    }
    
    @Test
    public void testRereRandomForestSerialization() {
        // 创建测试目录
        createTestDirectory();
        
        // 创建模型实例
        RereRandomForest model = new RereRandomForest();
        
        // 保存路径
        String filePath = TEST_DIR + "/random_forest_model.ser";
        
        // 测试保存
        model.save(filePath);
        
        // 验证文件存在
        assertTrue(new File(filePath).exists(), "模型文件应该存在");
        
        // 清理测试文件
        new File(filePath).delete();
    }
    
    @Test
    public void testRereXGboostSerialization() {
        // 创建测试目录
        createTestDirectory();
        
        // 创建模型实例
        RereXGboost model = new RereXGboost();
        
        // 保存路径
        String filePath = TEST_DIR + "/xgboost_model.ser";
        
        // 测试保存
        model.save(filePath);
        
        // 验证文件存在
        assertTrue(new File(filePath).exists(), "模型文件应该存在");
        
        // 清理测试文件
        new File(filePath).delete();
    }
    
    @Test
    public void testRereSVDSerialization() {
        // 创建测试目录
        createTestDirectory();
        
        // 创建模型实例
        RereSVD model = new RereSVD();
        
        // 保存路径
        String filePath = TEST_DIR + "/svd_model.ser";
        
        // 测试保存
        model.save(filePath);
        
        // 验证文件存在
        assertTrue(new File(filePath).exists(), "模型文件应该存在");
        
        // 清理测试文件
        new File(filePath).delete();
    }
    
    @Test
    public void testRereUMAPSerialization() {
        // 创建测试目录
        createTestDirectory();
        
        // 创建模型实例
        RereUMAP model = new RereUMAP();
        
        // 保存路径
        String filePath = TEST_DIR + "/umap_model.ser";
        
        // 测试保存
        model.save(filePath);
        
        // 验证文件存在
        assertTrue(new File(filePath).exists(), "模型文件应该存在");
        
        // 清理测试文件
        new File(filePath).delete();
    }
    
    @Test
    public void testRereTSNESerialization() {
        // 创建测试目录
        createTestDirectory();
        
        // 创建模型实例
        RereTSNE model = new RereTSNE();
        
        // 保存路径
        String filePath = TEST_DIR + "/tsne_model.ser";
        
        // 测试保存
        model.save(filePath);
        
        // 验证文件存在
        assertTrue(new File(filePath).exists(), "模型文件应该存在");
        
        // 清理测试文件
        new File(filePath).delete();
    }
    
    @Test
    public void testISerializableModelLoadMethodExists() {
        try {
            java.lang.reflect.Method loadMethod = ISerializableModel.class.getMethod("load", String.class);
            assertNotNull(loadMethod);
            assertTrue(java.lang.reflect.Modifier.isStatic(loadMethod.getModifiers()));
        } catch (NoSuchMethodException e) {
            fail("应存在 ISerializableModel.load(String): " + e.getMessage());
        }
    }
    
    private void createTestDirectory() {
        File dir = new File(TEST_DIR);
        if (!dir.exists()) {
            dir.mkdirs();
        }
    }
}