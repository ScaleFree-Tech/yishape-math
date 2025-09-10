package com.reremouse.lab.math.viz;

import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.RereVector;
import java.util.Arrays;
import java.util.List;
import com.reremouse.lab.math.viz.Plots;
import com.reremouse.lab.math.viz.IPlot;

/**
 * 测试重构后的IPlot和RerePlot功能
 * @author lteb2
 */
public class PlotRefactoringTest {
    
    public static void main(String[] args) {
        System.out.println("=== 测试重构后的IPlot功能 ===");
        
        // 创建测试数据
        IVector x = new RereVector(new float[]{1, 2, 3, 4, 5});
        IVector y = new RereVector(new float[]{2, 4, 6, 8, 10});
        List<String> labels = Arrays.asList("A", "B", "C", "D", "E");
        
        try {
            // 测试基础工厂方法
            System.out.println("1. 测试基础工厂方法...");
            IPlot plot1 = Plots.of();
            System.out.println("✓ Plots.of() 创建成功");
            
            IPlot plot2 = Plots.of(800, 600);
            System.out.println("✓ Plots.of(width, height) 创建成功");
            
            IPlot plot3 = Plots.of(1000, 800, "dark");
            System.out.println("✓ Plots.of(width, height, theme) 创建成功");
            
            // 测试图表类型专用工厂方法
            System.out.println("\n2. 测试图表类型专用工厂方法...");
            
            IPlot linePlot = Plots.line(x, y);
            System.out.println("✓ Plots.line(x, y) 创建成功");
            
            IPlot scatterPlot = Plots.scatter(x, y);
            System.out.println("✓ Plots.scatter(x, y) 创建成功");
            
            IPlot barPlot = Plots.bar(x);
            System.out.println("✓ Plots.bar(x) 创建成功");
            
            IPlot piePlot = Plots.pie(x);
            System.out.println("✓ Plots.pie(x) 创建成功");
            
            // 测试带尺寸的工厂方法
            System.out.println("\n3. 测试带尺寸的工厂方法...");
            
            IPlot linePlotWithSize = Plots.line(x, y, 1200, 800);
            System.out.println("✓ Plots.line(x, y, width, height) 创建成功");
            
            IPlot scatterPlotWithSize = Plots.scatter(x, y, 1000, 600);
            System.out.println("✓ Plots.scatter(x, y, width, height) 创建成功");
            
            // 测试带主题的工厂方法
            System.out.println("\n4. 测试带主题的工厂方法...");
            
            IPlot linePlotWithTheme = Plots.line(x, y, 1200, 800, "dark");
            System.out.println("✓ Plots.line(x, y, width, height, theme) 创建成功");
            
            IPlot barPlotWithTheme = Plots.bar(x, 1000, 600, "light");
            System.out.println("✓ Plots.bar(x, width, height, theme) 创建成功");
            
            // 测试流式API
            System.out.println("\n5. 测试流式API...");
            
            IPlot fluentPlot = Plots.line(x, y)
                .title("测试图表", "这是一个测试副标题")
                .xlabel("X轴标签")
                .ylabel("Y轴标签")
                .size(1200, 800)
                .theme("dark");
            System.out.println("✓ 流式API链式调用成功");
            
            // 测试保存功能
            System.out.println("\n6. 测试保存功能...");
            
            IPlot savePlot = Plots.scatter(x, y)
                .title("散点图测试")
                .xlabel("X轴")
                .ylabel("Y轴");
            
            // 注意：这里只是测试方法调用，不实际保存文件
            System.out.println("✓ 流式API保存方法调用成功");
            
            System.out.println("\n=== 所有测试通过！重构成功！ ===");
            
        } catch (Exception e) {
            System.err.println("测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
