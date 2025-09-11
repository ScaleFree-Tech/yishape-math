package com.reremouse.lab.math.viz;

import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.RereVector;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 箱线图诊断测试
 * 用于对比单箱线图和多箱线图的显示效果
 * @author lteb2
 */
public class BoxplotDiagnosticTest {
    
    public static void main(String[] args) {
        System.out.println("=== 箱线图诊断测试 ===\n");
        
        try {
            // 生成测试数据
            Random random = new Random(42);
            
            // 生成三组有明显差异的数据
            float[] data1 = generateNormalData(100, 10, 50, random);  // 均值10，标准差50
            float[] data2 = generateNormalData(100, 30, 20, random);  // 均值30，标准差20
            float[] data3 = generateNormalData(100, 50, 30, random);  // 均值50，标准差30
            
            // 创建向量
            IVector vector1 = new RereVector(data1);
            IVector vector2 = new RereVector(data2);
            IVector vector3 = new RereVector(data3);
            
            // 测试1: 单箱线图（应该正确显示）
            System.out.println("1. 测试单箱线图...");
            testSingleBoxplot(vector1, "single_boxplot_1.html");
            testSingleBoxplot(vector2, "single_boxplot_2.html");
            testSingleBoxplot(vector3, "single_boxplot_3.html");
            
            // 测试2: 多箱线图（问题版本）
            System.out.println("\n2. 测试多箱线图...");
            testMultiBoxplot(data1, data2, data3, "multi_boxplot.html");
            
            // 测试3: 调试多箱线图数据
            System.out.println("\n3. 调试多箱线图数据...");
            debugMultiBoxplotData(data1, data2, data3);
            
            System.out.println("\n=== 诊断测试完成 ===");
            
        } catch (Exception e) {
            System.err.println("测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试单箱线图
     */
    private static void testSingleBoxplot(IVector data, String filename) {
        try {
            RerePlot plot = new RerePlot();
            plot.title("单箱线图测试", "数据量: " + data.length());
            plot.xlabel("数据集");
            plot.ylabel("数值");
            plot.boxplot(data);
            plot.saveAsHtml(filename);
            System.out.println("✓ 单箱线图已保存为 " + filename);
        } catch (Exception e) {
            System.err.println("单箱线图测试失败: " + e.getMessage());
        }
    }
    
    /**
     * 测试多箱线图
     */
    private static void testMultiBoxplot(float[] data1, float[] data2, float[] data3, String filename) {
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
            
            RerePlot plot = new RerePlot();
            plot.title("多箱线图测试", "三组数据对比");
            plot.xlabel("数据集");
            plot.ylabel("数值");
            plot.boxplot(combinedVector, allLabels);
            plot.saveAsHtml(filename);
            System.out.println("✓ 多箱线图已保存为 " + filename);
        } catch (Exception e) {
            System.err.println("多箱线图测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 调试多箱线图数据
     */
    private static void debugMultiBoxplotData(float[] data1, float[] data2, float[] data3) {
        System.out.println("数据集1统计信息:");
        printStatistics(data1, "数据集1");
        
        System.out.println("\n数据集2统计信息:");
        printStatistics(data2, "数据集2");
        
        System.out.println("\n数据集3统计信息:");
        printStatistics(data3, "数据集3");
        
        // 模拟多箱线图的分组逻辑
        System.out.println("\n模拟多箱线图分组:");
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
        
        // 按标签分组
        java.util.Map<String, List<Float>> groupedData = new java.util.HashMap<>();
        for (int i = 0; i < allData.size(); i++) {
            String label = allLabels.get(i);
            if (!groupedData.containsKey(label)) {
                groupedData.put(label, new ArrayList<>());
            }
            groupedData.get(label).add(allData.get(i));
        }
        
        // 计算每个组的箱线图统计量
        for (String label : groupedData.keySet()) {
            List<Float> groupData = groupedData.get(label);
            float[] groupArray = new float[groupData.size()];
            for (int i = 0; i < groupData.size(); i++) {
                groupArray[i] = groupData.get(i);
            }
            java.util.Arrays.sort(groupArray);
            
            System.out.println("\n" + label + " 分组后统计:");
            printBoxplotStatistics(groupArray);
        }
    }
    
    /**
     * 打印基本统计信息
     */
    private static void printStatistics(float[] data, String name) {
        java.util.Arrays.sort(data);
        int n = data.length;
        float min = data[0];
        float max = data[n - 1];
        float mean = 0;
        for (float value : data) {
            mean += value;
        }
        mean /= n;
        
        System.out.println("  长度: " + n);
        System.out.println("  最小值: " + min);
        System.out.println("  最大值: " + max);
        System.out.println("  均值: " + mean);
        System.out.println("  范围: " + (max - min));
    }
    
    /**
     * 打印箱线图统计信息
     */
    private static void printBoxplotStatistics(float[] sortedData) {
        int n = sortedData.length;
        float min = sortedData[0];
        float max = sortedData[n - 1];
        
        // 计算Q1 (25%分位数)
        int q1Index = (int) Math.ceil(n * 0.25) - 1;
        q1Index = Math.max(0, Math.min(q1Index, n - 1));
        float q1 = sortedData[q1Index];
        
        // 计算中位数 (50%分位数)
        int q2Index = (int) Math.ceil(n * 0.5) - 1;
        q2Index = Math.max(0, Math.min(q2Index, n - 1));
        float q2 = sortedData[q2Index];
        
        // 计算Q3 (75%分位数)
        int q3Index = (int) Math.ceil(n * 0.75) - 1;
        q3Index = Math.max(0, Math.min(q3Index, n - 1));
        float q3 = sortedData[q3Index];
        
        System.out.println("  箱线图数据: [" + min + ", " + q1 + ", " + q2 + ", " + q3 + ", " + max + "]");
        System.out.println("  四分位距: " + (q3 - q1));
        System.out.println("  箱体高度: " + (q3 - q1));
    }
    
    /**
     * 生成正态分布数据
     */
    private static float[] generateNormalData(int size, float mean, float std, Random random) {
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
}
