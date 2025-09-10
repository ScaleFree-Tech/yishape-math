package com.reremouse.lab.math.viz;

import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.RereVector;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 简单的boxplot和violinplot方法测试
 * @author lteb2
 */
public class SimpleBoxplotViolinTest {
    
    public static void main(String[] args) {
        System.out.println("=== 测试RerePlot的boxplot和violinplot方法 ===");
        
        // 生成测试数据
        Random random = new Random(42);
        float[] data1 = generateNormalData(50, 15, 100, random);
        IVector vector1 = new RereVector(data1);
        
        // 测试方法调用（不实际渲染图表）
        try {
            RerePlot plot = new RerePlot();
            
            // 测试boxplot方法
            System.out.println("测试boxplot(IVector data)方法...");
            IPlot result1 = plot.boxplot(vector1);
            System.out.println("✓ boxplot(IVector data)方法调用成功");
            
            // 测试violinplot方法
            System.out.println("测试violinplot(IVector data)方法...");
            IPlot result2 = plot.violinplot(vector1);
            System.out.println("✓ violinplot(IVector data)方法调用成功");
            
            // 测试多组violinplot方法
            List<String> labels = new ArrayList<>();
            for (int i = 0; i < data1.length; i++) {
                labels.add("组" + (i % 3 + 1));
            }
            
            System.out.println("测试violinplot(IVector data, List<String> labels)方法...");
            IPlot result3 = plot.violinplot(vector1, labels);
            System.out.println("✓ violinplot(IVector data, List<String> labels)方法调用成功");
            
            System.out.println("\n=== 所有方法测试通过 ===");
            
        } catch (Exception e) {
            System.err.println("测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 生成正态分布数据
     */
    private static float[] generateNormalData(float mean, float stdDev, int size, Random random) {
        float[] data = new float[size];
        for (int i = 0; i < size; i++) {
            double u1 = random.nextDouble();
            double u2 = random.nextDouble();
            double z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
            data[i] = (float) (z0 * stdDev + mean);
        }
        return data;
    }
}
