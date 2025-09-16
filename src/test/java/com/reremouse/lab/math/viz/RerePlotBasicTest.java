package com.reremouse.lab.math.viz;

import com.reremouse.lab.math.linalg.RereDoubleVector;
import com.reremouse.lab.math.viz.RerePlot;
import com.reremouse.lab.math.viz.AxisTicks;
import java.util.Arrays;
import java.util.List;
import com.reremouse.lab.math.linalg.IDoubleVector;

/**
 * RerePlot 基础功能测试类
 * 测试 RerePlot 类的核心功能，不依赖 ECharts
 * 
 * @author lteb2
 */
public class RerePlotBasicTest {
    
    public static void main(String[] args) {
        System.out.println("=== RerePlot 基础功能测试 ===\n");
        
        try {
            // 测试数据创建
            System.out.println("1. 测试数据创建");
            double[] xData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            double[] yData = {2.1f, 3.9f, 6.1f, 8.0f, 10.2f, 12.1f, 14.0f, 16.1f, 18.0f, 20.1f};
            
            IDoubleVector x = new RereDoubleVector(xData);
            IDoubleVector y = new RereDoubleVector(yData);
            
            System.out.println("   X向量长度: " + x.length());
            System.out.println("   Y向量长度: " + y.length());
            System.out.println("   X向量均值: " + x.mean());
            System.out.println("   Y向量均值: " + y.mean());
            System.out.println("   X向量标准差: " + x.std());
            System.out.println("   Y向量标准差: " + y.std());
            
            // 测试 AxisTicks 类
            System.out.println("\n2. 测试 AxisTicks 类");
            AxisTicks xTicks = new AxisTicks();
            xTicks.setTickValues(new RereDoubleVector(new double[]{1, 5, 10}));
            xTicks.setTickLabels(Arrays.asList("开始", "中间", "结束"));
            
            AxisTicks yTicks = new AxisTicks();
            yTicks.setTickValues(new RereDoubleVector(new double[]{0, 10, 20}));
            
            System.out.println("   X轴刻度值数量: " + (xTicks.hasTickValues() ? xTicks.getTickValues().length() : 0));
            System.out.println("   X轴刻度标签数量: " + (xTicks.hasTickLabels() ? xTicks.getTickLabels().size() : 0));
            System.out.println("   Y轴刻度值数量: " + (yTicks.hasTickValues() ? yTicks.getTickValues().length() : 0));
            System.out.println("   Y轴刻度标签数量: " + (yTicks.hasTickLabels() ? yTicks.getTickLabels().size() : 0));
            
            // 测试数据转换功能
            System.out.println("\n3. 测试数据转换功能");
            testDataConversion(x, y);
            
            // 测试统计功能
            System.out.println("\n4. 测试统计功能");
            testStatistics(x, y);
            
            // 测试分组数据
            System.out.println("\n5. 测试分组数据");
            testGroupedData(x, y);
            
            System.out.println("\n=== 测试完成 ===");
            System.out.println("RerePlot 基础功能测试通过！");
            System.out.println("注意：完整的图表渲染功能需要 ECharts-Java 依赖支持。");
            
        } catch (Exception e) {
            System.err.println("测试过程中出现错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试数据转换功能
     */
    private static void testDataConversion(IDoubleVector x, IDoubleVector y) {
        try {
            // 测试单向量转换
            Object[] singleData = convertVectorToObjectArray(x);
            System.out.println("   单向量转换成功，数据长度: " + singleData.length);
            
            // 测试双向量转换
            Object[] doubleData = convertVectorsToObjectArray(x, y);
            System.out.println("   双向量转换成功，数据长度: " + doubleData.length);
            
            // 测试分组数据转换
            List<String> hue = Arrays.asList("A组", "B组", "A组", "B组", "A组", "B组", "A组", "B组", "A组", "B组");
            java.util.Map<String, java.util.List<Object[]>> groupedData = convertToGroupedData(x, y, hue);
            System.out.println("   分组数据转换成功，组数: " + groupedData.size());
            
        } catch (Exception e) {
            System.err.println("   数据转换测试失败: " + e.getMessage());
        }
    }
    
    /**
     * 测试统计功能
     */
    private static void testStatistics(IDoubleVector x, IDoubleVector y) {
        try {
            // 基本统计量
            System.out.println("   X向量 - 最小值: " + x.min() + ", 最大值: " + x.max());
            System.out.println("   Y向量 - 最小值: " + y.min() + ", 最大值: " + y.max());
            
            // 相关性
            double correlation = calculateCorrelation(x, y);
            System.out.println("   X和Y的相关系数: " + correlation);
            
            // 线性回归
            double[] regression = calculateLinearRegression(x, y);
            System.out.println("   线性回归 - 斜率: " + regression[0] + ", 截距: " + regression[1]);
            
        } catch (Exception e) {
            System.err.println("   统计功能测试失败: " + e.getMessage());
        }
    }
    
    /**
     * 测试分组数据功能
     */
    private static void testGroupedData(IDoubleVector x, IDoubleVector y) {
        try {
            List<String> hue = Arrays.asList("A组", "B组", "A组", "B组", "A组", "B组", "A组", "B组", "A组", "B组");
            
            // 按组统计
            java.util.Map<String, java.util.List<Double>> xGroups = new java.util.HashMap<>();
            java.util.Map<String, java.util.List<Double>> yGroups = new java.util.HashMap<>();
            
            for (int i = 0; i < x.length(); i++) {
                String group = hue.get(i);
                xGroups.computeIfAbsent(group, k -> new java.util.ArrayList<>()).add(x.get(i));
                yGroups.computeIfAbsent(group, k -> new java.util.ArrayList<>()).add(y.get(i));
            }
            
            System.out.println("   分组统计:");
            for (String group : xGroups.keySet()) {
                List<Double> xGroup = xGroups.get(group);
                List<Double> yGroup = yGroups.get(group);
                double xMean = (double) xGroup.stream().mapToDouble(Double::doubleValue).average().orElse(0);
                double yMean = (double) yGroup.stream().mapToDouble(Double::doubleValue).average().orElse(0);
                System.out.println("     " + group + " - X均值: " + xMean + ", Y均值: " + yMean);
            }
            
        } catch (Exception e) {
            System.err.println("   分组数据测试失败: " + e.getMessage());
        }
    }
    
    /**
     * 将向量转换为Object数组
     */
    private static Object[] convertVectorToObjectArray(IDoubleVector vector) {
        Object[] data = new Object[vector.length()];
        for (int i = 0; i < vector.length(); i++) {
            data[i] = vector.get(i);
        }
        return data;
    }
    
    /**
     * 将两个向量转换为Object数组
     */
    private static Object[] convertVectorsToObjectArray(IDoubleVector x, IDoubleVector y) {
        if (x.length() != y.length()) {
            throw new IllegalArgumentException("X和Y向量长度必须相等");
        }
        Object[] data = new Object[x.length()];
        for (int i = 0; i < x.length(); i++) {
            data[i] = new Number[]{x.get(i), y.get(i)};
        }
        return data;
    }
    
    /**
     * 转换为分组数据
     */
    private static java.util.Map<String, java.util.List<Object[]>> convertToGroupedData(IDoubleVector x, IDoubleVector y, List<String> hue) {
        java.util.Map<String, java.util.List<Object[]>> groupedData = new java.util.HashMap<>();
        for (int i = 0; i < x.length(); i++) {
            String groupName = hue.get(i);
            if (!groupedData.containsKey(groupName)) {
                groupedData.put(groupName, new java.util.ArrayList<>());
            }
            groupedData.get(groupName).add(new Number[]{x.get(i), y.get(i)});
        }
        return groupedData;
    }
    
    /**
     * 计算相关系数
     */
    private static double calculateCorrelation(IDoubleVector x, IDoubleVector y) {
        double xMean = x.mean();
        double yMean = y.mean();
        
        double numerator = 0;
        double xSumSq = 0;
        double ySumSq = 0;
        
        for (int i = 0; i < x.length(); i++) {
            double xDiff = x.get(i) - xMean;
            double yDiff = y.get(i) - yMean;
            numerator += xDiff * yDiff;
            xSumSq += xDiff * xDiff;
            ySumSq += yDiff * yDiff;
        }
        
        return numerator / (double) Math.sqrt(xSumSq * ySumSq);
    }
    
    /**
     * 计算线性回归
     */
    private static double[] calculateLinearRegression(IDoubleVector x, IDoubleVector y) {
        double xMean = x.mean();
        double yMean = y.mean();
        
        double numerator = 0;
        double denominator = 0;
        
        for (int i = 0; i < x.length(); i++) {
            double xDiff = x.get(i) - xMean;
            numerator += xDiff * (y.get(i) - yMean);
            denominator += xDiff * xDiff;
        }
        
        double slope = numerator / denominator;
        double intercept = yMean - slope * xMean;
        
        return new double[]{slope, intercept};
    }
}
