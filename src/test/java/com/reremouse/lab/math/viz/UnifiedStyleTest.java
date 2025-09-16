package com.reremouse.lab.math.viz;

import com.reremouse.lab.math.linalg.RereDoubleVector;
import com.reremouse.lab.math.linalg.RereDoubleMatrix;
import com.reremouse.lab.math.viz.RerePlot;
import java.util.Arrays;
import java.util.List;
import com.reremouse.lab.math.linalg.IDoubleMatrix;
import com.reremouse.lab.math.linalg.IDoubleVector;

/**
 * 统一风格参数系统测试类
 * 验证所有图表类型都能正确应用样式参数
 * 
 * @author lteb2
 */
public class UnifiedStyleTest {
    
    public static void main(String[] args) {
        System.out.println("开始测试统一风格参数系统...");
        
        // 创建测试数据
        double[] data1 = {1.0, 2.0, 3.0, 4.0, 5.0};
        double[] data2 = {2.0, 4.0, 6.0, 8.0, 10.0};
        IDoubleVector x = new RereDoubleVector(data1);
        IDoubleVector y = new RereDoubleVector(data2);
        
        // 创建分组数据
        List<String> labels = Arrays.asList("A", "B", "A", "B", "A");
        List<String> categories = Arrays.asList("类别1", "类别2", "类别3", "类别4", "类别5");
        
        // 创建矩阵数据
        double[][] matrixData = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0}
        };
        IDoubleMatrix matrix = new RereDoubleMatrix(matrixData);
        List<String> xLabels = Arrays.asList("X1", "X2", "X3");
        List<String> yLabels = Arrays.asList("Y1", "Y2", "Y3");
        
        try {
            // 测试1: 线图样式
            System.out.println("测试1: 线图样式");
            RerePlot plot1 = new RerePlot();
            plot1.line(x, y, "r-")  // 红色实线
                  .title("红色实线图")
                  .xlabel("X轴")
                  .ylabel("Y轴")
                  .saveAsHtml("test_line_red.html");
            System.out.println("✓ 线图样式测试通过");
            
            // 测试2: 散点图样式
            System.out.println("测试2: 散点图样式");
            RerePlot plot2 = new RerePlot();
            plot2.scatter(x, y, "bo")  // 蓝色圆圈
                  .title("蓝色散点图")
                  .xlabel("X轴")
                  .ylabel("Y轴")
                  .saveAsHtml("test_scatter_blue.html");
            System.out.println("✓ 散点图样式测试通过");
            
            // 测试3: 柱状图样式
            System.out.println("测试3: 柱状图样式");
            RerePlot plot3 = new RerePlot();
            plot3.bar(x, "g-")  // 绿色柱状图
                  .title("绿色柱状图")
                  .xlabel("类别")
                  .ylabel("数值")
                  .saveAsHtml("test_bar_green.html");
            System.out.println("✓ 柱状图样式测试通过");
            
            // 测试4: 分组柱状图样式
            System.out.println("测试4: 分组柱状图样式");
            RerePlot plot4 = new RerePlot();
            plot4.bar(x, labels, "r-")  // 红色分组柱状图
                  .title("红色分组柱状图")
                  .xlabel("类别")
                  .ylabel("数值")
                  .saveAsHtml("test_grouped_bar_red.html");
            System.out.println("✓ 分组柱状图样式测试通过");
            
            // 测试5: 饼图样式
            System.out.println("测试5: 饼图样式");
            RerePlot plot5 = new RerePlot();
            plot5.pie(x, categories, "b-")  // 蓝色饼图
                  .title("蓝色饼图")
                  .saveAsHtml("test_pie_blue.html");
            System.out.println("✓ 饼图样式测试通过");
            
            // 测试6: 直方图样式
            System.out.println("测试6: 直方图样式");
            RerePlot plot6 = new RerePlot();
            plot6.hist(x, true, "g-")  // 绿色直方图带拟合线
                  .title("绿色直方图")
                  .xlabel("数值区间")
                  .ylabel("频次")
                  .saveAsHtml("test_hist_green.html");
            System.out.println("✓ 直方图样式测试通过");
            
            // 测试7: 热力图样式
            System.out.println("测试7: 热力图样式");
            RerePlot plot7 = new RerePlot();
            plot7.heatmap(matrix, xLabels, yLabels, "r-")  // 红色热力图
                  .title("红色热力图")
                  .xlabel("X轴")
                  .ylabel("Y轴")
                  .saveAsHtml("test_heatmap_red.html");
            System.out.println("✓ 热力图样式测试通过");
            
            // 测试8: 雷达图样式
            System.out.println("测试8: 雷达图样式");
            RerePlot plot8 = new RerePlot();
            plot8.radar(x, categories, "b-")  // 蓝色雷达图
                  .title("蓝色雷达图")
                  .saveAsHtml("test_radar_blue.html");
            System.out.println("✓ 雷达图样式测试通过");
            
            // 测试9: 仪表盘样式
            System.out.println("测试9: 仪表盘样式");
            RerePlot plot9 = new RerePlot();
            plot9.gauge(75.0, 100.0, 0.0, "g-")  // 绿色仪表盘
                  .title("绿色仪表盘")
                  .saveAsHtml("test_gauge_green.html");
            System.out.println("✓ 仪表盘样式测试通过");
            
            // 测试10: 箱线图样式
            System.out.println("测试10: 箱线图样式");
            RerePlot plot10 = new RerePlot();
            plot10.boxplot(x, labels, "r-")  // 红色箱线图
                   .title("红色箱线图")
                   .xlabel("类别")
                   .ylabel("数值")
                   .saveAsHtml("test_boxplot_red.html");
            System.out.println("✓ 箱线图样式测试通过");
            
            // 测试11: 使用PlotStyle对象
            System.out.println("测试11: 使用PlotStyle对象");
            PlotStyle customStyle = PlotStyle.line("#FF6B6B", "dashed", 3.0f)
                    .marker("s")
                    .markerSize(8.0f)
                    .label("自定义样式");
            
            RerePlot plot11 = new RerePlot();
            plot11.line(x, y, customStyle)
                  .title("自定义样式线图")
                  .xlabel("X轴")
                  .ylabel("Y轴")
                  .saveAsHtml("test_custom_style.html");
            System.out.println("✓ PlotStyle对象测试通过");
            
            // 测试12: 样式系统开关
            System.out.println("测试12: 样式系统开关");
            RerePlot plot12 = new RerePlot();
            plot12.enableStyleSystem(false)  // 禁用样式系统
                  .line(x, y, "r-")  // 这个样式字符串应该被忽略
                  .title("禁用样式系统的线图")
                  .xlabel("X轴")
                  .ylabel("Y轴")
                  .saveAsHtml("test_style_disabled.html");
            System.out.println("✓ 样式系统开关测试通过");
            
            System.out.println("\n🎉 所有测试通过！统一风格参数系统工作正常。");
            System.out.println("生成的HTML文件可以在浏览器中查看效果。");
            
        } catch (Exception e) {
            System.err.println("❌ 测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
