package com.reremouse.lab.audio.embedding;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * OnlineIVectorEmbedding 使用示例
 OnlineIVectorEmbedding Usage Example
 
 演示如何使用在线i-vector模型处理流式音频数据
 Demonstrates how to use the online i-vector model to process streaming audio data
 */
public class OnlineIVectorModelExample {
    
    public static void main(String[] args) {
        System.out.println("=== 在线i-vector模型使用示例 / Online i-vector Model Usage Example ===\n");
        
        // 创建在线i-vector模型
        int ivectorDim = 50;      // i-vector维度
        int numComponents = 16;    // UBM高斯分量数（减少以加快示例运行）
        int featureDim = 13;      // MFCC特征维度
        boolean useAdam = true;   // 使用Adam优化器
        double learningRate = 0.001; // 学习率
        
        OnlineIVectorEmbedding model = new OnlineIVectorEmbedding(ivectorDim, numComponents, featureDim, useAdam, learningRate);
        
        System.out.println("创建在线i-vector模型:");
        System.out.println("- i-vector维度: " + model.getIVectorDimension());
        System.out.println("- UBM高斯分量数: " + model.getNumComponents());
        System.out.println("- MFCC特征维度: " + model.getFeatureDimension());
        System.out.println("- 使用Adam优化器: " + useAdam);
        System.out.println("- 学习率: " + learningRate);
        System.out.println();
        
        // 生成模拟的流式MFCC数据
        System.out.println("生成模拟流式训练数据...");
        List<IMatrix<Double>> streamingData = generateStreamingMFCCData(20, featureDim, 100); // 20个批次，每批次100帧
        
        System.out.println("流式训练数据统计:");
        System.out.println("- 批次数: " + streamingData.size());
        System.out.println("- 每个批次帧数: " + streamingData.get(0).getRowNum() + " (示例)");
        System.out.println("- 特征维度: " + streamingData.get(0).getColNum());
        System.out.println();
        
        // 模拟流式训练过程
        System.out.println("开始流式训练...");
        long startTime = System.currentTimeMillis();
        
        try {
            // 逐批次进行增量训练
            for (int i = 0; i < streamingData.size(); i++) {
                IMatrix<Double> batch = streamingData.get(i);
                model.trainIncremental(batch);
                
                if ((i + 1) % 5 == 0) {
                    System.out.println("已处理 " + (i + 1) + " 个批次 / Processed " + (i + 1) + " batches");
                }
            }
            
            // 完成训练
            model.finishTraining();
            long endTime = System.currentTimeMillis();
            
            System.out.println("流式训练完成!");
            System.out.println("训练时间: " + (endTime - startTime) + " ms");
            System.out.println("模型状态: " + (model.isTrained() ? "已训练" : "未训练"));
            System.out.println("总处理样本数: " + model.getProcessedSamples());
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
            List<IMatrix<Double>> testSamples = generateStreamingMFCCData(3, featureDim, 75);
            
            for (int i = 0; i < testSamples.size(); i++) {
                IMatrix<Double> sample = testSamples.get(i);
                IVector<Double> sampleIVector = model.embed(sample);
                
                System.out.println("样本 " + (i + 1) + ":");
                System.out.println("  输入: " + sample.getRowNum() + " 帧");
                System.out.println("  i-vector L2范数: " + String.format("%.6f", sampleIVector.norm2()));
                System.out.println("  前5个元素: " + formatVector(sampleIVector, 5));
            }
            
            System.out.println("\n=== 示例完成 ===");
            System.out.println("在线i-vector模型成功将流式的MFCC特征转换为定长的向量表征!");
            
        } catch (Exception e) {
            System.err.println("训练过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 生成模拟的流式MFCC数据
     * @param numBatches 批次数
     * @param featureDim 特征维度
     * @param framesPerBatch 每批次帧数
     * @return 流式MFCC数据列表
     */
    private static List<IMatrix<Double>> generateStreamingMFCCData(int numBatches, int featureDim, int framesPerBatch) {
        List<IMatrix<Double>> batches = new ArrayList<>();
        Random random = new Random(42); // 固定种子以保证可重现性
        
        for (int i = 0; i < numBatches; i++) {
            double[][] batchData = new double[framesPerBatch][featureDim];
            
            // 生成具有某种模式的MFCC数据
            for (int r = 0; r < framesPerBatch; r++) {
                for (int c = 0; c < featureDim; c++) {
                    // 添加一些模式和噪声
                    double pattern = Math.sin(r * 0.1) * Math.cos(c * 0.5);
                    double noise = random.nextGaussian() * 0.1;
                    batchData[r][c] = pattern + noise;
                }
            }
            
            batches.add(Linalg.matrix(batchData));
        }
        
        return batches;
    }
    
    /**
     * 生成单个MFCC序列
     * @param numFrames 帧数
     * @param featureDim 特征维度
     * @return MFCC矩阵
     */
    private static IMatrix<Double> generateSingleMFCCSequence(int numFrames, int featureDim) {
        Random random = new Random(123); // 不同的种子
        double[][] data = new double[numFrames][featureDim];
        
        for (int r = 0; r < numFrames; r++) {
            for (int c = 0; c < featureDim; c++) {
                // 添加一些模式和噪声
                double pattern = Math.cos(r * 0.2) * Math.sin(c * 0.3);
                double noise = random.nextGaussian() * 0.15;
                data[r][c] = pattern + noise;
            }
        }
        
        return Linalg.matrix(data);
    }
    
    /**
     * 格式化向量输出
     * @param vector 向量
     * @param maxElements 最大元素数
     * @return 格式化字符串
     */
    private static String formatVector(IVector<Double> vector, int maxElements) {
        StringBuilder sb = new StringBuilder("[");
        int elementsToShow = Math.min(vector.length(), maxElements);
        
        for (int i = 0; i < elementsToShow; i++) {
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