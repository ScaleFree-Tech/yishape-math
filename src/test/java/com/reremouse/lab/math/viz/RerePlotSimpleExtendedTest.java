package com.reremouse.lab.math.viz;

import com.reremouse.lab.math.RereVector;
import com.reremouse.lab.math.RereMatrix;
import com.reremouse.lab.math.viz.RerePlot;
import java.util.*;

/**
 * RerePlot扩展功能简单测试
 * @author lteb2
 */
public class RerePlotSimpleExtendedTest {
    
    public static void main(String[] args) {
        System.out.println("=== RerePlot扩展功能简单测试 ===");
        
        // 创建测试数据
        RereVector data = new RereVector(new float[]{10, 20, 30, 40, 50});
        List<String> categories = Arrays.asList("类别A", "类别B", "类别C", "类别D", "类别E");
        
        RerePlot plot = new RerePlot(800, 600);
        plot.setTitle("RerePlot扩展功能测试");
        
        try {
            // 测试极坐标柱状图
            System.out.println("测试极坐标柱状图...");
            plot.polarBar(data, categories);
            plot.saveAsHtml("test_polar_bar.html");
            System.out.println("✓ 极坐标柱状图测试成功");
            
            // 测试箱线图
            System.out.println("测试箱线图...");
            plot.boxplot(data, categories);
            plot.saveAsHtml("test_boxplot.html");
            System.out.println("✓ 箱线图测试成功");
            
            // 测试漏斗图
            System.out.println("测试漏斗图...");
            plot.funnel(data, categories);
            plot.saveAsHtml("test_funnel.html");
            System.out.println("✓ 漏斗图测试成功");
            
            // 测试K线图
            System.out.println("测试K线图...");
            float[][] candlestickArray = {
                {100, 110, 95, 115},
                {110, 120, 105, 125},
                {120, 115, 110, 130}
            };
            RereMatrix candlestickData = new RereMatrix(candlestickArray);
            List<String> dates = Arrays.asList("2024-01-01", "2024-01-02", "2024-01-03");
            plot.candlestick(candlestickData, dates);
            plot.saveAsHtml("test_candlestick.html");
            System.out.println("✓ K线图测试成功");
            
            // 测试热力图
            System.out.println("测试热力图...");
            float[][] heatmapArray = {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
            };
            RereMatrix heatmapData = new RereMatrix(heatmapArray);
            List<String> xLabels = Arrays.asList("X1", "X2", "X3");
            List<String> yLabels = Arrays.asList("Y1", "Y2", "Y3");
            plot.heatmap(heatmapData, xLabels, yLabels);
            plot.saveAsHtml("test_heatmap.html");
            System.out.println("✓ 热力图测试成功");
            
            // 测试雷达图
            System.out.println("测试雷达图...");
            List<String> indicators = Arrays.asList("指标1", "指标2", "指标3", "指标4", "指标5");
            plot.radar(data, indicators);
            plot.saveAsHtml("test_radar.html");
            System.out.println("✓ 雷达图测试成功");
            
            // 测试仪表盘
            System.out.println("测试仪表盘...");
            plot.gauge(75, 100, 0);
            plot.saveAsHtml("test_gauge.html");
            System.out.println("✓ 仪表盘测试成功");
            
            System.out.println("\n=== 所有测试完成 ===");
            System.out.println("生成的HTML文件：");
            System.out.println("- test_polar_bar.html");
            System.out.println("- test_boxplot.html");
            System.out.println("- test_funnel.html");
            System.out.println("- test_candlestick.html");
            System.out.println("- test_heatmap.html");
            System.out.println("- test_radar.html");
            System.out.println("- test_gauge.html");
            
        } catch (Exception e) {
            System.err.println("测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
