package com.yishape.lab.math.viz;

import com.yishape.lab.math.plot.Plots;
import com.yishape.lab.math.plot.IPlot;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;

/**
 * 测试RerePlot坐标轴标签功能
 */
public class RerePlotLabelTest {
    public static void main(String[] args) {
        // 创建测试数据
        IVector<Double> x = Linalg.range(10);
        IVector<Double> y = Linalg.range(10).multiplyScalar(2.0);
        
        // 测试1: 先设置数据再设置标签
        System.out.println("测试1: 先设置数据再设置标签");
        IPlot plot1 = Plots.of()
                .line(x, y)
                .xlabel("时间 (秒)")
                .ylabel("幅度");
        // 保存HTML文件进行检查
        plot1.saveAsHtml("test1_before_show.html");
        // 显示图表
        plot1.show();
        
        // 测试2: 先设置标签再设置数据
        System.out.println("测试2: 先设置标签再设置数据");
        IPlot plot2 = Plots.of()
                .xlabel("频率 (Hz)")
                .ylabel("功率")
                .line(x, y);
        // 保存HTML文件进行检查
        plot2.saveAsHtml("test2_before_show.html");
        // 显示图表
        plot2.show();
        
        // 测试3: 在显示前修改标签
        System.out.println("测试3: 在显示前修改标签");
        IPlot plot3 = Plots.of()
                .line(x, y)
                .xlabel("初始X标签")
                .ylabel("初始Y标签");
        // 在显示前修改标签
        plot3.xlabel("修改后的X标签")
             .ylabel("修改后的Y标签");
        // 保存HTML文件进行检查
        plot3.saveAsHtml("test3_before_show.html");
        // 显示图表
        plot3.show();
        
        System.out.println("所有测试完成，请检查生成的HTML文件中的坐标轴标签是否正确显示。");
    }
}