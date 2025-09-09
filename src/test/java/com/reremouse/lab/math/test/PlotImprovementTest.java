package com.reremouse.lab.math.test;

import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.RereVector;
import com.reremouse.lab.math.viz.Plots;
import com.reremouse.lab.math.viz.PlotException;
import java.util.Arrays;
import java.util.List;
import com.reremouse.lab.math.viz.IPlot;

/**
 * 测试改进后的Plot API
 * @author lteb2
 */
public class PlotImprovementTest {
    
    public static void main(String[] args) {
        System.out.println("=== 测试改进后的Plot API ===\n");
        
        // 创建测试数据
        IVector x = new RereVector(new float[]{1, 2, 3, 4, 5});
        IVector y = new RereVector(new float[]{2, 4, 6, 8, 10});
        List<String> labels = Arrays.asList("A", "B", "C", "D", "E");
        
        try {
            // 测试1: 基础工厂方法
            System.out.println("1. 测试基础工厂方法...");
            IPlot plot1 = Plots.of();
            System.out.println("✓ Plots.of() 创建成功");
            
            IPlot plot2 = Plots.of(800, 600);
            System.out.println("✓ Plots.of(width, height) 创建成功");
            
            IPlot plot3 = Plots.of(1000, 800, "dark");
            System.out.println("✓ Plots.of(width, height, theme) 创建成功");
            
            // 测试2: 图表类型专用工厂方法
            System.out.println("\n2. 测试图表类型专用工厂方法...");
            
            IPlot linePlot = Plots.line(x, y);
            System.out.println("✓ Plots.line(x, y) 创建成功");
            
            IPlot scatterPlot = Plots.scatter(x, y);
            System.out.println("✓ Plots.scatter(x, y) 创建成功");
            
            IPlot barPlot = Plots.bar(x);
            System.out.println("✓ Plots.bar(x) 创建成功");
            
            IPlot piePlot = Plots.pie(x);
            System.out.println("✓ Plots.pie(x) 创建成功");
            
            // 测试3: 流式API
            System.out.println("\n3. 测试流式API...");
            
            IPlot fluentPlot = Plots.line(x, y)
                .title("测试图表", "这是一个测试副标题")
                .xlabel("X轴标签")
                .ylabel("Y轴标签")
                .size(1200, 800)
                .theme("dark");
            System.out.println("✓ 流式API链式调用成功");
            
            // 测试4: 异常处理
            System.out.println("\n4. 测试异常处理...");
            
            try {
                IPlot errorPlot = Plots.of();
                errorPlot.line(null, y); // 应该抛出异常
                System.out.println("✗ 异常处理测试失败：应该抛出异常");
            } catch (PlotException e) {
                System.out.println("✓ 异常处理测试成功：" + e.getMessage());
            }
            
            // 测试5: 配置方法
            System.out.println("\n5. 测试配置方法...");
            
            IPlot configPlot = Plots.of(800, 600);
            configPlot.setTitle("配置测试");
            configPlot.setXlabel("X轴");
            configPlot.setYlabel("Y轴");
            configPlot.line(x, y);
            
            System.out.println("✓ 配置方法测试成功");
            System.out.println("  宽度: " + configPlot.getWidth());
            System.out.println("  高度: " + configPlot.getHeight());
            System.out.println("  主题: " + configPlot.getTheme());
            
            // 测试6: 工具方法
            System.out.println("\n6. 测试工具方法...");
            
            IPlot toolPlot = Plots.scatter(x, y);
            String html = toolPlot.toHtml();
            String json = toolPlot.toJson();
            
            System.out.println("✓ 工具方法测试成功");
            System.out.println("  HTML长度: " + html.length());
            System.out.println("  JSON长度: " + json.length());
            
            System.out.println("\n=== 所有测试通过！API改进成功！ ===");
            
        } catch (Exception e) {
            System.err.println("测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
