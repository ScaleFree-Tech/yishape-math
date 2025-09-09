package com.reremouse.lab.math.viz;

import com.reremouse.lab.math.IMatrix;
import com.reremouse.lab.math.IVector;
import java.util.List;

/**
 * 绘图静态工厂类，提供创建各种图表类型的静态方法
 * @author lteb2
 */
public final class Plots {
    
    private Plots() {
        // 工具类，防止实例化
    }
    
    // ========== 基础工厂方法 ==========
    
    /**
     * 创建默认绘图对象
     * @return RerePlot实例
     */
    public static RerePlot of() {
        return new RerePlot();
    }

    /**
     * 创建指定尺寸的绘图对象
     * @param width 图表宽度
     * @param height 图表高度
     * @return RerePlot实例
     */
    public static RerePlot of(int width, int height) {
        return new RerePlot(width, height);
    }

    /**
     * 创建指定尺寸和主题的绘图对象
     * @param width 图表宽度
     * @param height 图表高度
     * @param theme 主题名称
     * @return RerePlot实例
     */
    public static RerePlot of(int width, int height, String theme) {
        return new RerePlot(width, height, theme);
    }
    
    // ========== 图表类型专用工厂方法 ==========
    
    /**
     * 创建线图
     * @param x X轴数据
     * @param y Y轴数据
     * @return 配置好的线图实例
     */
    public static RerePlot line(IVector x, IVector y) {
        RerePlot plot = of();
        plot.line(x, y);
        return plot;
    }
    
    /**
     * 创建单向量线图
     * @param x 数据向量
     * @return 配置好的线图实例
     */
    public static RerePlot line(IVector x) {
        RerePlot plot = of();
        plot.line(x);
        return plot;
    }
    
    /**
     * 创建多线图
     * @param x X轴数据
     * @param y Y轴数据
     * @param hue 分组标签
     * @return 配置好的多线图实例
     */
    public static RerePlot line(IVector x, IVector y, List<String> hue) {
        RerePlot plot = of();
        plot.line(x, y, hue);
        return plot;
    }
    
    /**
     * 创建散点图
     * @param x X轴数据
     * @param y Y轴数据
     * @return 配置好的散点图实例
     */
    public static RerePlot scatter(IVector x, IVector y) {
        RerePlot plot = of();
        plot.scatter(x, y);
        return plot;
    }
    
    /**
     * 创建多组散点图
     * @param x X轴数据
     * @param y Y轴数据
     * @param hue 分组标签
     * @return 配置好的多组散点图实例
     */
    public static RerePlot scatter(IVector x, IVector y, List<String> hue) {
        RerePlot plot = of();
        plot.scatter(x, y, hue);
        return plot;
    }
    
    /**
     * 创建饼图
     * @param x 数据向量
     * @return 配置好的饼图实例
     */
    public static RerePlot pie(IVector x) {
        RerePlot plot = of();
        plot.pie(x);
        return plot;
    }
    
    /**
     * 创建柱状图
     * @param x 数据向量
     * @return 配置好的柱状图实例
     */
    public static RerePlot bar(IVector x) {
        RerePlot plot = of();
        plot.bar(x);
        return plot;
    }
    
    /**
     * 创建分组柱状图
     * @param x 数据向量
     * @param hue 分组标签
     * @return 配置好的分组柱状图实例
     */
    public static RerePlot bar(IVector x, List<String> hue) {
        RerePlot plot = of();
        plot.bar(x, hue);
        return plot;
    }
    
    /**
     * 创建直方图
     * @param x 数据向量
     * @param fittingLine 是否显示拟合线
     * @return 配置好的直方图实例
     */
    public static RerePlot hist(IVector x, boolean fittingLine) {
        RerePlot plot = of();
        plot.hist(x, fittingLine);
        return plot;
    }
    
    /**
     * 创建热力图
     * @param data 二维数据矩阵
     * @param xLabels X轴标签
     * @param yLabels Y轴标签
     * @return 配置好的热力图实例
     */
    public static RerePlot heatmap(IMatrix data, List<String> xLabels, List<String> yLabels) {
        RerePlot plot = of();
        plot.heatmap(data, xLabels, yLabels);
        return plot;
    }
    
    /**
     * 创建雷达图
     * @param data 数据向量
     * @param indicators 指标名称
     * @return 配置好的雷达图实例
     */
    public static RerePlot radar(IVector data, List<String> indicators) {
        RerePlot plot = of();
        plot.radar(data, indicators);
        return plot;
    }
    
    /**
     * 创建仪表盘
     * @param value 数值
     * @param max 最大值
     * @param min 最小值
     * @return 配置好的仪表盘实例
     */
    public static RerePlot gauge(float value, float max, float min) {
        RerePlot plot = of();
        plot.gauge(value, max, min);
        return plot;
    }
    
    // ========== 带尺寸的工厂方法 ==========
    
    /**
     * 创建指定尺寸的线图
     * @param x X轴数据
     * @param y Y轴数据
     * @param width 图表宽度
     * @param height 图表高度
     * @return 配置好的线图实例
     */
    public static RerePlot line(IVector x, IVector y, int width, int height) {
        RerePlot plot = of(width, height);
        plot.line(x, y);
        return plot;
    }
    
    /**
     * 创建指定尺寸的散点图
     * @param x X轴数据
     * @param y Y轴数据
     * @param width 图表宽度
     * @param height 图表高度
     * @return 配置好的散点图实例
     */
    public static RerePlot scatter(IVector x, IVector y, int width, int height) {
        RerePlot plot = of(width, height);
        plot.scatter(x, y);
        return plot;
    }
    
    /**
     * 创建指定尺寸的柱状图
     * @param x 数据向量
     * @param width 图表宽度
     * @param height 图表高度
     * @return 配置好的柱状图实例
     */
    public static RerePlot bar(IVector x, int width, int height) {
        RerePlot plot = of(width, height);
        plot.bar(x);
        return plot;
    }
    
    /**
     * 创建指定尺寸的饼图
     * @param x 数据向量
     * @param width 图表宽度
     * @param height 图表高度
     * @return 配置好的饼图实例
     */
    public static RerePlot pie(IVector x, int width, int height) {
        RerePlot plot = of(width, height);
        plot.pie(x);
        return plot;
    }
    
    // ========== 带主题的工厂方法 ==========
    
    /**
     * 创建指定主题的线图
     * @param x X轴数据
     * @param y Y轴数据
     * @param width 图表宽度
     * @param height 图表高度
     * @param theme 主题名称
     * @return 配置好的线图实例
     */
    public static RerePlot line(IVector x, IVector y, int width, int height, String theme) {
        RerePlot plot = of(width, height, theme);
        plot.line(x, y);
        return plot;
    }
    
    /**
     * 创建指定主题的散点图
     * @param x X轴数据
     * @param y Y轴数据
     * @param width 图表宽度
     * @param height 图表高度
     * @param theme 主题名称
     * @return 配置好的散点图实例
     */
    public static RerePlot scatter(IVector x, IVector y, int width, int height, String theme) {
        RerePlot plot = of(width, height, theme);
        plot.scatter(x, y);
        return plot;
    }
    
    /**
     * 创建指定主题的柱状图
     * @param x 数据向量
     * @param width 图表宽度
     * @param height 图表高度
     * @param theme 主题名称
     * @return 配置好的柱状图实例
     */
    public static RerePlot bar(IVector x, int width, int height, String theme) {
        RerePlot plot = of(width, height, theme);
        plot.bar(x);
        return plot;
    }
    
    /**
     * 创建指定主题的饼图
     * @param x 数据向量
     * @param width 图表宽度
     * @param height 图表高度
     * @param theme 主题名称
     * @return 配置好的饼图实例
     */
    public static RerePlot pie(IVector x, int width, int height, String theme) {
        RerePlot plot = of(width, height, theme);
        plot.pie(x);
        return plot;
    }
}
