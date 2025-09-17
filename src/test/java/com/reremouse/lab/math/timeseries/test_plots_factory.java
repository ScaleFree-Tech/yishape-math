package com.reremouse.lab.math.timeseries;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.viz.Plots;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

/**
 * 测试Plots工厂方法
 * @author lteb2
 */
public class test_plots_factory {
    
    public static void main(String[] args) {
        System.out.println("=== 测试Plots工厂方法 ===\n");
        
        try {
            // 创建测试数据
            double[] xData = {1, 2, 3, 4, 5};
            double[] yData = {10, 20, 15, 30, 25};
            double[] pieData = {30, 25, 20, 15, 10};
            double[] histData = {1.2, 2.3, 1.8, 3.1, 2.7, 1.5, 2.9, 3.2, 2.1, 2.8};
            
            IVector<Double> x = Linalg.vector(xData);
            IVector<Double> y = Linalg.vector(yData);
            IVector<Double> pie = Linalg.vector(pieData);
            IVector<Double> hist = Linalg.vector(histData);
            
            List<String> labels = Arrays.asList("A", "B", "C", "D", "E");
            List<String> categories = Arrays.asList("类别1", "类别2", "类别3", "类别4", "类别5");
            
            // 测试基础图表工厂方法
            System.out.println("1. 测试基础图表工厂方法");
            
            // 线图
            Plots.line(x, y)
                .title("线图测试")
                .xlabel("X轴")
                .ylabel("Y轴")
                .saveAsHtml("test_line.html");
            System.out.println("   ✓ 线图创建成功");
            
            // 散点图
            Plots.scatter(x, y)
                .title("散点图测试")
                .saveAsHtml("test_scatter.html");
            System.out.println("   ✓ 散点图创建成功");
            
            // 饼图
            Plots.pie(pie, labels)
                .title("饼图测试")
                .saveAsHtml("test_pie.html");
            System.out.println("   ✓ 饼图创建成功");
            
            // 柱状图
            Plots.bar(pie, labels)
                .title("柱状图测试")
                .saveAsHtml("test_bar.html");
            System.out.println("   ✓ 柱状图创建成功");
            
            // 直方图
            Plots.hist(hist, true)
                .title("直方图测试")
                .saveAsHtml("test_hist.html");
            System.out.println("   ✓ 直方图创建成功");
            
            // 测试极坐标图表工厂方法
            System.out.println("\n2. 测试极坐标图表工厂方法");
            
            Plots.polarBar(pie, categories)
                .title("极坐标柱状图测试")
                .saveAsHtml("test_polar_bar.html");
            System.out.println("   ✓ 极坐标柱状图创建成功");
            
            Plots.polarLine(pie, categories)
                .title("极坐标线图测试")
                .saveAsHtml("test_polar_line.html");
            System.out.println("   ✓ 极坐标线图创建成功");
            
            Plots.polarScatter(pie, categories)
                .title("极坐标散点图测试")
                .saveAsHtml("test_polar_scatter.html");
            System.out.println("   ✓ 极坐标散点图创建成功");
            
            // 测试统计图表工厂方法
            System.out.println("\n3. 测试统计图表工厂方法");
            
            Plots.boxplot(hist, Arrays.asList("数据集"))
                .title("箱线图测试")
                .saveAsHtml("test_boxplot.html");
            System.out.println("   ✓ 箱线图创建成功");
            
            Plots.violinplot(hist)
                .title("小提琴图测试")
                .saveAsHtml("test_violinplot.html");
            System.out.println("   ✓ 小提琴图创建成功");
            
            // K线图测试数据
            double[][] candlestickData = {{100, 110, 95, 115}, {110, 120, 105, 125}, {120, 115, 110, 130}};
            IMatrix<Double> candlestickMatrix = Linalg.matrix(candlestickData);
            List<String> dates = Arrays.asList("2024-01-01", "2024-01-02", "2024-01-03");
            
            Plots.candlestick(candlestickMatrix, dates)
                .title("K线图测试")
                .saveAsHtml("test_candlestick.html");
            System.out.println("   ✓ K线图创建成功");
            
            // 测试特殊图表工厂方法
            System.out.println("\n4. 测试特殊图表工厂方法");
            
            Plots.funnel(pie, labels)
                .title("漏斗图测试")
                .saveAsHtml("test_funnel.html");
            System.out.println("   ✓ 漏斗图创建成功");
            
            // 桑基图测试数据
            List<Map<String, Object>> nodes = Arrays.asList(
                Map.of("name", "节点1"),
                Map.of("name", "节点2"),
                Map.of("name", "节点3")
            );
            List<Map<String, Object>> links = Arrays.asList(
                Map.of("source", "节点1", "target", "节点2", "value", 10),
                Map.of("source", "节点2", "target", "节点3", "value", 5)
            );
            
            Plots.sankey(nodes, links)
                .title("桑基图测试")
                .saveAsHtml("test_sankey.html");
            System.out.println("   ✓ 桑基图创建成功");
            
            // 测试扩展图表工厂方法
            System.out.println("\n5. 测试扩展图表工厂方法");
            
            // 热力图测试数据
            double[][] heatmapData = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
            IMatrix<Double> heatmapMatrix = Linalg.matrix(heatmapData);
            List<String> xLabels = Arrays.asList("X1", "X2", "X3");
            List<String> yLabels = Arrays.asList("Y1", "Y2", "Y3");
            
            Plots.heatmap(heatmapMatrix, xLabels, yLabels)
                .title("热力图测试")
                .saveAsHtml("test_heatmap.html");
            System.out.println("   ✓ 热力图创建成功");
            
            Plots.radar(pie, categories)
                .title("雷达图测试")
                .saveAsHtml("test_radar.html");
            System.out.println("   ✓ 雷达图创建成功");
            
            Plots.gauge(75.5, 100, 0)
                .title("仪表盘测试")
                .saveAsHtml("test_gauge.html");
            System.out.println("   ✓ 仪表盘创建成功");
            
            System.out.println("\n=== 所有测试完成！===");
            System.out.println("生成的HTML文件：");
            System.out.println("- test_line.html");
            System.out.println("- test_scatter.html");
            System.out.println("- test_pie.html");
            System.out.println("- test_bar.html");
            System.out.println("- test_hist.html");
            System.out.println("- test_polar_bar.html");
            System.out.println("- test_polar_line.html");
            System.out.println("- test_polar_scatter.html");
            System.out.println("- test_boxplot.html");
            System.out.println("- test_violinplot.html");
            System.out.println("- test_candlestick.html");
            System.out.println("- test_funnel.html");
            System.out.println("- test_sankey.html");
            System.out.println("- test_heatmap.html");
            System.out.println("- test_radar.html");
            System.out.println("- test_gauge.html");
            
        } catch (Exception e) {
            System.err.println("测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
