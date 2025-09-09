package com.reremouse.lab.math.test;

import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.RereVector;
import com.reremouse.lab.math.viz.RerePlot;
import com.reremouse.lab.math.viz.AxisTicks;
import java.util.Arrays;
import java.util.List;

/**
 * RerePlot 简化演示类
 * 展示 RerePlot 类的基本画图功能，避免复杂依赖
 * 
 * @author lteb2
 */
public class RerePlotSimpleDemo {
    
    public static void main(String[] args) {
        System.out.println("=== RerePlot 简化演示 ===\n");
        
        try {
            // 创建测试数据
            float[] xData = {1, 2, 3, 4, 5};
            float[] yData = {2.1f, 3.9f, 6.1f, 8.0f, 10.2f};
            
            IVector x = IVector.of(xData);
            IVector y = IVector.of(yData);
            
            // 1. 基础线图演示
            System.out.println("1. 基础线图演示");
            RerePlot plot1 = new RerePlot(600, 400);
            plot1.setTitle("基础线图", "简单数据趋势");
            plot1.setXlabel("X轴");
            plot1.setYlabel("Y轴");
            plot1.line(x, y);
            
            // 保存图表
            String html1 = plot1.toHtml();
            System.out.println("   线图HTML长度: " + html1.length());
            System.out.println("   线图JSON长度: " + plot1.toJson().length());
            
            // 2. 散点图演示
            System.out.println("\n2. 散点图演示");
            RerePlot plot2 = new RerePlot(600, 400);
            plot2.setTitle("散点图", "数据点分布");
            plot2.setXlabel("X轴");
            plot2.setYlabel("Y轴");
            plot2.scatter(x, y);
            
            String html2 = plot2.toHtml();
            System.out.println("   散点图HTML长度: " + html2.length());
            
            // 3. 柱状图演示
            System.out.println("\n3. 柱状图演示");
            RerePlot plot3 = new RerePlot(600, 400);
            plot3.setTitle("柱状图", "分类数据");
            plot3.setXlabel("类别");
            plot3.setYlabel("数值");
            plot3.bar(y);
            
            String html3 = plot3.toHtml();
            System.out.println("   柱状图HTML长度: " + html3.length());
            
            // 4. 饼图演示
            System.out.println("\n4. 饼图演示");
            RerePlot plot4 = new RerePlot(600, 400);
            plot4.setTitle("饼图", "数据占比");
            float[] pieData = {30, 25, 20, 15, 10};
            IVector pieVector = IVector.of(pieData);
            plot4.pie(pieVector);
            
            String html4 = plot4.toHtml();
            System.out.println("   饼图HTML长度: " + html4.length());
            
            // 5. 直方图演示
            System.out.println("\n5. 直方图演示");
            RerePlot plot5 = new RerePlot(600, 400);
            plot5.setTitle("直方图", "数据分布");
            plot5.setXlabel("数值区间");
            plot5.setYlabel("频次");
            
            // 生成测试数据
            float[] histData = new float[50];
            for (int i = 0; i < 50; i++) {
                histData[i] = (float) (Math.random() * 10 + 5);
            }
            IVector histVector = IVector.of(histData);
            plot5.hist(histVector, false);
            
            String html5 = plot5.toHtml();
            System.out.println("   直方图HTML长度: " + html5.length());
            
            // 6. 图表配置演示
            System.out.println("\n6. 图表配置演示");
            RerePlot plot6 = new RerePlot(800, 500);
            plot6.setTitle("配置演示", "自定义设置");
            plot6.setXlabel("时间");
            plot6.setYlabel("温度 (°C)");
            
            // 设置坐标轴刻度
            AxisTicks xTicks = new AxisTicks();
            xTicks.setTickValues(IVector.of(new float[]{1, 3, 5}));
            xTicks.setTickLabels(Arrays.asList("周一", "周三", "周五"));
            plot6.setXticks(xTicks);
            
            plot6.line(x, y);
            
            String html6 = plot6.toHtml();
            System.out.println("   配置图表HTML长度: " + html6.length());
            
            // 7. 图表信息
            System.out.println("\n7. 图表信息");
            System.out.println("   图表尺寸: " + plot1.getWidth() + " x " + plot1.getHeight());
            System.out.println("   图表主题: " + plot1.getTheme());
            System.out.println("   X轴标签: " + plot1.getXlabel());
            System.out.println("   Y轴标签: " + plot1.getYlabel());
            
            System.out.println("\n=== 演示完成 ===");
            System.out.println("所有图表功能测试通过！");
            
        } catch (Exception e) {
            System.err.println("演示过程中出现错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
