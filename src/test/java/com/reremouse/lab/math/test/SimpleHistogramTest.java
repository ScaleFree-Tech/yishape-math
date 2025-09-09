package com.reremouse.lab.math.test;

import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.RereVector;
import com.reremouse.lab.math.viz.Plots;

/**
 * 简单直方图测试
 * @author lteb2
 */
public class SimpleHistogramTest {
    
    public static void main(String[] args) {
        System.out.println("=== 简单直方图测试 ===");
        
        try {
            // 创建测试数据
            float[] data = {1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 9, 10};
            IVector vector = IVector.of(data);
            
            System.out.println("数据点数量: " + vector.length());
            System.out.println("数据范围: " + vector.min() + " - " + vector.max());
            System.out.println("数据均值: " + vector.mean());
            System.out.println("数据标准差: " + vector.std());
            
            // 测试直方图（无拟合线）
            System.out.println("\n创建直方图（无拟合线）...");
            var plot1 = Plots.of(800, 600);
            plot1.title("简单直方图测试", "无拟合线");
            plot1.xlabel("数值");
            plot1.ylabel("频次");
            plot1.hist(vector, false);
            plot1.saveAsHtml("simple_histogram.html");
            System.out.println("✓ 直方图已保存为 simple_histogram.html");
            
            // 测试直方图（有拟合线）
            System.out.println("\n创建直方图（有拟合线）...");
            var plot2 = Plots.of(800, 600);
            plot2.title("简单直方图测试", "带正态分布拟合");
            plot2.xlabel("数值");
            plot2.ylabel("频次");
            plot2.hist(vector, true);
            plot2.saveAsHtml("simple_histogram_with_fit.html");
            System.out.println("✓ 带拟合线的直方图已保存为 simple_histogram_with_fit.html");
            
            System.out.println("\n=== 测试完成！ ===");
            
        } catch (Exception e) {
            System.err.println("测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
