package com.reremouse.lab.math.test;

import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.RereVector;
import com.reremouse.lab.math.viz.Plots;
import java.util.Random;

/**
 * 直方图功能测试
 * @author lteb2
 */
public class HistogramTest {
    
    public static void main(String[] args) {
        System.out.println("=== 直方图功能测试 ===\n");
        
        try {
            // 测试1: 正态分布数据
            System.out.println("1. 测试正态分布数据...");
            float[] normalData = generateNormalDistribution(1000, 5.0f, 2.0f);
            IVector normalVector = IVector.of(normalData);
            
            var plot1 = Plots.of(800, 600);
            plot1.title("正态分布直方图", "均值=5, 标准差=2");
            plot1.xlabel("数值");
            plot1.ylabel("频次");
            plot1.hist(normalVector, true);
            plot1.saveAsHtml("normal_histogram.html");
            System.out.println("✓ 正态分布直方图已保存为 normal_histogram.html");
            
            // 测试2: 均匀分布数据
            System.out.println("\n2. 测试均匀分布数据...");
            float[] uniformData = generateUniformDistribution(1000, 0.0f, 10.0f);
            IVector uniformVector = IVector.of(uniformData);
            
            var plot2 = Plots.of(800, 600);
            plot2.title("均匀分布直方图", "范围=[0, 10]");
            plot2.xlabel("数值");
            plot2.ylabel("频次");
            plot2.hist(uniformVector, false);
            plot2.saveAsHtml("uniform_histogram.html");
            System.out.println("✓ 均匀分布直方图已保存为 uniform_histogram.html");
            
            // 测试3: 小数据集
            System.out.println("\n3. 测试小数据集...");
            float[] smallData = {1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f, 5.0f};
            IVector smallVector = IVector.of(smallData);
            
            var plot3 = Plots.of(800, 600);
            plot3.title("小数据集直方图", "9个数据点");
            plot3.xlabel("数值");
            plot3.ylabel("频次");
            plot3.hist(smallVector, true);
            plot3.saveAsHtml("small_histogram.html");
            System.out.println("✓ 小数据集直方图已保存为 small_histogram.html");
            
            // 测试4: 相同值数据
            System.out.println("\n4. 测试相同值数据...");
            float[] sameData = new float[50];
            for (int i = 0; i < 50; i++) {
                sameData[i] = 5.0f; // 所有值都相同
            }
            IVector sameVector = IVector.of(sameData);
            
            var plot4 = Plots.of(800, 600);
            plot4.title("相同值直方图", "所有值都是5.0");
            plot4.xlabel("数值");
            plot4.ylabel("频次");
            plot4.hist(sameVector, false);
            plot4.saveAsHtml("same_histogram.html");
            System.out.println("✓ 相同值直方图已保存为 same_histogram.html");
            
            System.out.println("\n=== 所有直方图测试完成！ ===");
            
        } catch (Exception e) {
            System.err.println("测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 生成正态分布数据
     * @param size 数据点数量
     * @param mean 均值
     * @param std 标准差
     * @return 生成的数据数组
     */
    private static float[] generateNormalDistribution(int size, float mean, float std) {
        Random random = new Random(42); // 固定种子以便重现
        float[] data = new float[size];
        
        for (int i = 0; i < size; i++) {
            // Box-Muller变换生成正态分布
            double u1 = random.nextDouble();
            double u2 = random.nextDouble();
            double z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
            data[i] = (float) (mean + std * z0);
        }
        
        return data;
    }
    
    /**
     * 生成均匀分布数据
     * @param size 数据点数量
     * @param min 最小值
     * @param max 最大值
     * @return 生成的数据数组
     */
    private static float[] generateUniformDistribution(int size, float min, float max) {
        Random random = new Random(42); // 固定种子以便重现
        float[] data = new float[size];
        
        for (int i = 0; i < size; i++) {
            data[i] = min + random.nextFloat() * (max - min);
        }
        
        return data;
    }
}
