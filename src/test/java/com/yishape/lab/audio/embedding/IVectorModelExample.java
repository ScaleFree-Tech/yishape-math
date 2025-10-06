package com.yishape.lab.audio.embedding;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * IVectorModel使用示例
 * 
 * 演示如何使用i-vector模型将MFCC特征转换为定长向量表征
 * 
 * @author lteb2
 */
public class IVectorModelExample {
    
    public static void main(String[] args) {
        System.out.println("=== i-vector模型使用示例 / i-vector Model Usage Example ===\n");
        
        // 创建i-vector模型
        int ivectorDim = 100;      // i-vector维度
        int numComponents = 64;    // UBM高斯分量数（减少以加快示例运行）
        int featureDim = 13;       // MFCC特征维度
        
        IVectorEmbedding model = new IVectorEmbedding(ivectorDim, numComponents, featureDim);
        
        System.out.println("创建i-vector模型:");
        System.out.println("- i-vector维度: " + model.getLen());
        System.out.println("- UBM高斯分量数: " + model.getNumComponents());
        System.out.println("- MFCC特征维度: " + model.getMfccDim());
        System.out.println();
        
        // 生成模拟的MFCC训练数据
        System.out.println("生成模拟训练数据...");
        List<IMatrix<Double>> trainingData = generateSimulatedMFCCData(10, featureDim);
        
        System.out.println("训练数据统计:");
        System.out.println("- 训练样本数: " + trainingData.size());
        System.out.println("- 每个样本帧数: " + trainingData.get(0).getRowNum() + " (示例)");
        System.out.println("- 特征维度: " + trainingData.get(0).getColNum());
        System.out.println();
        
        // 训练模型
        System.out.println("开始训练模型...");
        long startTime = System.currentTimeMillis();
        
        try {
            model.train(trainingData);
            long endTime = System.currentTimeMillis();
            
            System.out.println("模型训练完成!");
            System.out.println("训练时间: " + (endTime - startTime) + " ms");
            System.out.println("模型状态: " + (model.isTrained() ? "已训练" : "未训练"));
            System.out.println();
            
            // 测试i-vector提取
            System.out.println("=== 测试i-vector提取 ===");
            
            // 生成测试数据
            IMatrix<Double> testMFCC = generateSingleMFCCSequence(50, featureDim);
            System.out.println("测试MFCC序列: " + testMFCC.getRowNum() + " 帧 x " + testMFCC.getColNum() + " 维");
            
            // 提取i-vector
            IVector<Double> ivector = model.embed(testMFCC);
            
            System.out.println("提取的i-vector:");
            System.out.println("- 维度: " + ivector.length());
            System.out.println("- L2范数: " + String.format("%.6f", ivector.norm2()));
            System.out.println("- 前10个元素: " + formatVector(ivector, 10));
            System.out.println();
            
            // 测试多个样本的i-vector提取
            System.out.println("=== 批量i-vector提取测试 ===");
            List<IMatrix<Double>> testSamples = generateSimulatedMFCCData(5, featureDim);
            
            for (int i = 0; i < testSamples.size(); i++) {
                IMatrix<Double> sample = testSamples.get(i);
                IVector<Double> sampleIVector = model.embed(sample);
                
                System.out.println("样本 " + (i + 1) + ":");
                System.out.println("  输入: " + sample.getRowNum() + " 帧");
                System.out.println("  i-vector L2范数: " + String.format("%.6f", sampleIVector.norm2()));
                System.out.println("  前5个元素: " + formatVector(sampleIVector, 5));
            }
            
            System.out.println("\n=== 示例完成 ===");
            System.out.println("i-vector模型成功将变长的MFCC特征转换为定长的向量表征!");
            
        } catch (Exception e) {
            System.err.println("训练过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 生成模拟的MFCC训练数据
     */
    private static List<IMatrix<Double>> generateSimulatedMFCCData(int numSamples, int featureDim) {
        List<IMatrix<Double>> data = new ArrayList<>();
        Random random = new Random(42);
        
        for (int i = 0; i < numSamples; i++) {
            // 随机生成帧数（模拟不同长度的音频）
            int numFrames = 30 + random.nextInt(70); // 30-100帧
            
            // 生成MFCC特征矩阵
            IMatrix<Double> mfcc = generateSingleMFCCSequence(numFrames, featureDim);
            data.add(mfcc);
        }
        
        return data;
    }
    
    /**
     * 生成单个MFCC序列
     */
    private static IMatrix<Double> generateSingleMFCCSequence(int numFrames, int featureDim) {
        Random random = new Random();
        double[][] data = new double[numFrames][featureDim];
        
        // 生成具有一定结构的模拟MFCC特征
        for (int t = 0; t < numFrames; t++) {
            for (int d = 0; d < featureDim; d++) {
                // 模拟MFCC特征的典型分布
                double base = (d == 0) ? 10.0 : 0.0; // C0通常较大
                double noise = random.nextGaussian() * 2.0;
                double temporal = Math.sin(2 * Math.PI * t / 20.0) * 0.5; // 添加时间相关性
                
                data[t][d] = base + noise + temporal;
            }
        }
        
        return Linalg.matrix(data);
    }
    
    /**
     * 格式化向量输出
     */
    private static String formatVector(IVector<Double> vector, int maxElements) {
        StringBuilder sb = new StringBuilder("[");
        int limit = Math.min(maxElements, vector.length());
        
        for (int i = 0; i < limit; i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.4f", vector.get(i)));
        }
        
        if (vector.length() > maxElements) {
            sb.append(", ...");
        }
        
        sb.append("]");
        return sb.toString();
    }
}