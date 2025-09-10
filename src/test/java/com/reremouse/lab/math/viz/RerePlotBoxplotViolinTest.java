package com.reremouse.lab.math.viz;

import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.RereVector;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 测试RerePlot的boxplot和violinplot方法
 * @author lteb2
 */
public class RerePlotBoxplotViolinTest {
    
    public static void main(String[] args) {
        // 生成测试数据
        Random random = new Random(42); // 固定种子以便结果可重现
        
        // 生成三组正态分布数据
        float[] data1 = generateNormalData(50, 15, 200, random);
        float[] data2 = generateNormalData(80, 10, 200, random);
        float[] data3 = generateNormalData(65, 20, 200, random);
        
        // 创建向量
        IVector vector1 = new RereVector(data1);
        IVector vector2 = new RereVector(data2);
        IVector vector3 = new RereVector(data3);
        
        // 合并数据用于多组小提琴图
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
        
        // 测试单向量箱线图
        System.out.println("=== 测试单向量箱线图 ===");
        try {
            RerePlot plot1 = new RerePlot();
            plot1.title("单向量箱线图测试")
                    .xlabel("数据")
                    .ylabel("数值")
                    .boxplot(vector1)
                    .saveAsHtml("test_boxplot_single.html");
            System.out.println("单向量箱线图测试成功");
        } catch (Exception e) {
            System.err.println("单向量箱线图测试失败: " + e.getMessage());
        }
        
        // 测试单向量小提琴图
        System.out.println("\n=== 测试单向量小提琴图 ===");
        try {
            RerePlot plot2 = new RerePlot();
            plot2.title("单向量小提琴图测试")
                    .xlabel("数值")
                    .ylabel("密度")
                    .violinplot(vector1)
                    .saveAsHtml("test_violinplot_single.html");
            System.out.println("单向量小提琴图测试成功");
        } catch (Exception e) {
            System.err.println("单向量小提琴图测试失败: " + e.getMessage());
        }
        
        // 测试多组小提琴图
        System.out.println("\n=== 测试多组小提琴图 ===");
        try {
            RerePlot plot3 = new RerePlot();
            plot3.title("多组小提琴图测试")
                    .xlabel("数值")
                    .ylabel("密度")
                    .violinplot(combinedVector, allLabels)
                    .saveAsHtml("test_violinplot_multi.html");
            System.out.println("多组小提琴图测试成功");
        } catch (Exception e) {
            System.err.println("多组小提琴图测试失败: " + e.getMessage());
        }
        
        System.out.println("\n=== 所有测试完成 ===");
    }
    
    /**
     * 生成正态分布数据
     * @param mean 均值
     * @param stdDev 标准差
     * @param size 数据量
     * @param random 随机数生成器
     * @return 生成的数据数组
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
