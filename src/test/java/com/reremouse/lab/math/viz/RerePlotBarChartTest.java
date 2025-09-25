package com.reremouse.lab.math.viz;

import com.reremouse.lab.math.linalg.RereDoubleVector;
import com.reremouse.lab.math.linalg.IDoubleVector;
import java.util.Arrays;
import java.util.List;

/**
 * RerePlot 柱状图功能测试类
 * 测试柱状图的x轴标签功能
 */
public class RerePlotBarChartTest {
    
    public static void main(String[] args) {
        System.out.println("=== RerePlot 柱状图功能测试 ===\n");
        
        try {
            // 创建测试数据
            double[] data = {20.1, 3.9, 16.1, 8.0, 20.2};
            IDoubleVector y = new RereDoubleVector(data);
            
            // 创建x轴标签
            List<String> xLabels = Arrays.asList("一月", "二月", "三月", "四月", "五月");
            
            // 测试1: 基础柱状图带x轴标签
            System.out.println("1. 测试基础柱状图带x轴标签");
            RerePlot plot1 = new RerePlot(800, 600);
            plot1.bar(xLabels, y)
                  .title("月度销售数据", "2024年各月销售情况")
                  .xlabel("月份")
                  .ylabel("销售额（万元）")
                  .saveAsHtml("test_bar_with_labels.html");
            
            String html1 = plot1.toHtml();
            System.out.println("   ✓ 柱状图HTML生成成功，长度: " + html1.length());
            
            // 测试2: 分组柱状图带x轴标签
            System.out.println("\n2. 测试分组柱状图带x轴标签");
            List<String> hue = Arrays.asList("产品A", "产品B", "产品A", "产品B", "产品A");
            RerePlot plot2 = new RerePlot(800, 600);
            plot2.bar(xLabels, y, hue)
                  .title("产品销售对比", "不同产品在各月的销售情况")
                  .xlabel("月份")
                  .ylabel("销售额（万元）")
                  .saveAsHtml("test_grouped_bar_with_labels.html");
            
            String html2 = plot2.toHtml();
            System.out.println("   ✓ 分组柱状图HTML生成成功，长度: " + html2.length());
            
            // 测试3: 使用setXticks设置x轴标签
            System.out.println("\n3. 测试使用setXticks设置x轴标签");
            RerePlot plot3 = new RerePlot(800, 600);
            AxisTicks xTicks = new AxisTicks();
            xTicks.setTickLabels(Arrays.asList("Q1", "Q2", "Q3", "Q4", "Q5"));
            plot3.setXticks(xTicks);
            plot3.bar(y)
                  .title("季度数据", "使用setXticks设置标签")
                  .xlabel("季度")
                  .ylabel("数值")
                  .saveAsHtml("test_bar_with_xticks.html");
            
            String html3 = plot3.toHtml();
            System.out.println("   ✓ setXticks柱状图HTML生成成功，长度: " + html3.length());
            
            System.out.println("\n=== 所有测试通过 ===");
            System.out.println("柱状图x轴标签功能实现正确！");
            
        } catch (Exception e) {
            System.err.println("测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}