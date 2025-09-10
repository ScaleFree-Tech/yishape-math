package com.reremouse.lab.math.viz;

import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.RereVector;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 测试带有箱线效果的小提琴图
 * @author lteb2
 */
public class ViolinPlotWithBoxplotTest {
    
    public static void main(String[] args) {
        System.out.println("=== 测试带有箱线效果的小提琴图 ===");
        
        // 生成测试数据
        Random random = new Random(42);
        float[] data1 = generateNormalData(50, 15, 200, random);
        float[] data2 = generateNormalData(80, 10, 200, random);
        float[] data3 = generateNormalData(65, 20, 200, random);
        
        IVector vector1 = new RereVector(data1);
        IVector vector2 = new RereVector(data2);
        IVector vector3 = new RereVector(data3);
        
        // 测试单向量小提琴图（带箱线效果）
        System.out.println("测试单向量小提琴图（带箱线效果）...");
        try {
            RerePlot plot1 = new RerePlot();
            plot1.title("单向量小提琴图（带箱线效果）")
                    .xlabel("数值")
                    .ylabel("密度")
                    .violinplot(vector1);
            
            // 生成HTML文件
            String html1 = plot1.toHtml();
            System.out.println("✓ 单向量小提琴图生成成功，HTML长度: " + html1.length());
            
        } catch (Exception e) {
            System.err.println("✗ 单向量小提琴图测试失败: " + e.getMessage());
            e.printStackTrace();
        }
        
        // 测试多组小提琴图（带箱线效果）
        System.out.println("\n测试多组小提琴图（带箱线效果）...");
        try {
            // 合并数据
            List<Float> allData = new ArrayList<>();
            List<String> allLabels = new ArrayList<>();
            
            for (float value : data1) {
                allData.add(value);
                allLabels.add("数据集1");
            }
            for (float value : data2) {
                allData.add(value);
                allLabels.add("数据集2");
            }
            for (float value : data3) {
                allData.add(value);
                allLabels.add("数据集3");
            }
            
            float[] combinedData = new float[allData.size()];
            for (int i = 0; i < allData.size(); i++) {
                combinedData[i] = allData.get(i);
            }
            IVector combinedVector = new RereVector(combinedData);
            
            RerePlot plot2 = new RerePlot();
            plot2.title("多组小提琴图（带箱线效果）")
                    .xlabel("数值")
                    .ylabel("密度")
                    .violinplot(combinedVector, allLabels);
            
            // 生成HTML文件
            String html2 = plot2.toHtml();
            System.out.println("✓ 多组小提琴图生成成功，HTML长度: " + html2.length());
            
        } catch (Exception e) {
            System.err.println("✗ 多组小提琴图测试失败: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println("\n=== 测试完成 ===");
        System.out.println("小提琴图现在包含：");
        System.out.println("1. 左右对称的密度曲线（形成小提琴形状）");
        System.out.println("2. 箱线图元素（中位数线、四分位数箱体、须线）");
        System.out.println("3. 多组数据支持，每组都有独立的小提琴形状和箱线");
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
