package com.reremouse.lab.math.viz;

import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.viz.AxisTicks;
import java.util.Arrays;
import java.util.List;
import com.reremouse.lab.math.viz.Plots;
import com.reremouse.lab.math.viz.IPlot;

/**
 * IPlot 演示类
 展示 IPlot 类的各种画图功能
 * 
 * @author lteb2
 */
public class PlotDemo {
    
    public static void main(String[] args) {
        System.out.println("=== IPlot 画图功能演示 ===\n");
        
        // 创建测试数据
        float[] xData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        float[] yData = {2.1f, 3.9f, 6.1f, 8.0f, 10.2f, 12.1f, 14.0f, 16.1f, 18.0f, 20.1f};
        float[] y2Data = {1.5f, 2.8f, 4.2f, 5.9f, 7.1f, 8.8f, 10.5f, 12.2f, 13.9f, 15.6f};
        
        IVector x = IVector.of(xData);
        IVector y = IVector.of(yData);
        IVector y2 = IVector.of(y2Data);
        
        // 创建分组数据
        List<String> hue = Arrays.asList("A组", "B组", "A组", "B组", "A组", "B组", "A组", "B组", "A组", "B组");
        
        // 1. 线图演示
        System.out.println("1. 线图演示");
        IPlot plot1 = Plots.of(800, 600);
        plot1.setTitle("线图演示", "展示数据趋势");
        plot1.setXlabel("X轴");
        plot1.setYlabel("Y轴");
        plot1.line(x, y);
        plot1.saveAsHtml("line_chart.html");
        System.out.println("   线图已保存为 line_chart.html\n");
        
        // 2. 多条线图演示
        System.out.println("2. 多条线图演示");
        IPlot plot2 = Plots.of(800, 600);
        plot2.setTitle("多条线图演示", "对比两组数据");
        plot2.setXlabel("X轴");
        plot2.setYlabel("Y轴");
        plot2.line(x, y, hue);
        plot2.saveAsHtml("multi_line_chart.html");
        System.out.println("   多条线图已保存为 multi_line_chart.html\n");
        
        // 3. 散点图演示
        System.out.println("3. 散点图演示");
        IPlot plot3 = Plots.of(800, 600);
        plot3.setTitle("散点图演示", "展示数据分布");
        plot3.setXlabel("X轴");
        plot3.setYlabel("Y轴");
        plot3.scatter(x, y);
        plot3.saveAsHtml("scatter_chart.html");
        System.out.println("   散点图已保存为 scatter_chart.html\n");
        
        // 4. 多组散点图演示
        System.out.println("4. 多组散点图演示");
        IPlot plot4 = Plots.of(800, 600);
        plot4.setTitle("多组散点图演示", "对比两组数据分布");
        plot4.setXlabel("X轴");
        plot4.setYlabel("Y轴");
        plot4.scatter(x, y, hue);
        plot4.saveAsHtml("multi_scatter_chart.html");
        System.out.println("   多组散点图已保存为 multi_scatter_chart.html\n");
        
        // 5. 饼图演示
        System.out.println("5. 饼图演示");
        IPlot plot5 = Plots.of(800, 600);
        plot5.setTitle("饼图演示", "展示数据占比");
        float[] pieData = {30, 25, 20, 15, 10};
        IVector pieVector = IVector.of(pieData);
        plot5.pie(pieVector);
        plot5.saveAsHtml("pie_chart.html");
        System.out.println("   饼图已保存为 pie_chart.html\n");
        
        // 6. 柱状图演示
        System.out.println("6. 柱状图演示");
        IPlot plot6 = Plots.of(800, 600);
        plot6.setTitle("柱状图演示", "展示分类数据");
        plot6.setXlabel("类别");
        plot6.setYlabel("数值");
        plot6.bar(pieVector);
        plot6.saveAsHtml("bar_chart.html");
        System.out.println("   柱状图已保存为 bar_chart.html\n");
        
        // 7. 分组柱状图演示
        System.out.println("7. 分组柱状图演示");
        IPlot plot7 = Plots.of(800, 600);
        plot7.setTitle("分组柱状图演示", "对比不同组别的数据");
        plot7.setXlabel("类别");
        plot7.setYlabel("数值");
        List<String> barHue = Arrays.asList("组A", "组B", "组A", "组B", "组A");
        plot7.bar(pieVector, barHue);
        plot7.saveAsHtml("grouped_bar_chart.html");
        System.out.println("   分组柱状图已保存为 grouped_bar_chart.html\n");
        
        // 8. 直方图演示
        System.out.println("8. 直方图演示");
        IPlot plot8 = Plots.of(800, 600);
        plot8.setTitle("直方图演示", "展示数据分布和正态分布拟合");
        plot8.setXlabel("数值区间");
        plot8.setYlabel("频次");
        
        // 生成正态分布数据
        float[] histData = new float[100];
        for (int i = 0; i < 100; i++) {
            histData[i] = (float) (Math.random() * 10 + 5); // 均值5，标准差约2.9
        }
        IVector histVector = IVector.of(histData);
        plot8.hist(histVector, true);
        plot8.saveAsHtml("histogram_chart.html");
        System.out.println("   直方图已保存为 histogram_chart.html\n");
        
        // 9. 自定义配置演示
        System.out.println("9. 自定义配置演示");
        IPlot plot9 = Plots.of(1000, 700);
        plot9.setTitle("自定义配置演示", "展示各种配置选项");
        plot9.setXlabel("时间");
        plot9.setYlabel("温度 (°C)");
        
        // 设置坐标轴刻度
        AxisTicks xTicks = new AxisTicks();
        xTicks.setTickValues(IVector.of(new float[]{1, 3, 5, 7, 9}));
        xTicks.setTickLabels(Arrays.asList("周一", "周三", "周五", "周日", "周二"));
        plot9.setXticks(xTicks);
        
        AxisTicks yTicks = new AxisTicks();
        yTicks.setTickValues(IVector.of(new float[]{0, 5, 10, 15, 20, 25}));
        plot9.setYticks(yTicks);
        
        plot9.line(x, y);
        plot9.saveAsHtml("custom_config_chart.html");
        System.out.println("   自定义配置图表已保存为 custom_config_chart.html\n");
        
        // 10. 获取图表信息
        System.out.println("10. 图表信息获取演示");
        System.out.println("    HTML内容长度: " + plot1.toHtml().length());
        System.out.println("    JSON配置长度: " + plot1.toJson().length());
        System.out.println("    图表尺寸: " + plot1.getWidth() + " x " + plot1.getHeight());
        System.out.println("    图表主题: " + plot1.getTheme());
        
        System.out.println("\n=== 演示完成 ===");
        System.out.println("所有图表文件已生成，可以在浏览器中查看。");
    }
}
