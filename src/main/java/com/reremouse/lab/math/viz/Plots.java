package com.reremouse.lab.math.viz;

import java.util.List;
import java.util.Map;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;

/**
 * 绘图静态工厂类，提供创建各种图表类型的静态方法
 * @author lteb2
 */
public final class Plots {
    
//    private Plots() {
//        // 工具类，防止实例化
//    }
    
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
    
    // ========== 基础图表工厂方法 ==========
    
    /**
     * 创建线图
     * @param x X轴数据
     * @param y Y轴数据
     * @return RerePlot实例
     */
    public static RerePlot line(IVector<?> x, IVector<?> y) {
        return new RerePlot().line(x, y);
    }
    
    /**
     * 创建线图（支持样式字符串）
     * @param x X轴数据
     * @param y Y轴数据
     * @param styleString 样式字符串
     * @return RerePlot实例
     */
    public static RerePlot line(IVector<?> x, IVector<?> y, String styleString) {
        return new RerePlot().line(x, y, styleString);
    }
    
    /**
     * 创建单向量线图
     * @param y Y轴数据
     * @return RerePlot实例
     */
    public static RerePlot line(IVector<?> y) {
        return new RerePlot().line(y);
    }
    
    /**
     * 创建分组线图
     * @param x X轴数据
     * @param y Y轴数据
     * @param hue 颜色分组
     * @return RerePlot实例
     */
    public static RerePlot line(IVector<?> x, IVector<?> y, List<String> hue) {
        return new RerePlot().line(x, y, hue);
    }
    
    /**
     * 创建散点图
     * @param x X轴数据
     * @param y Y轴数据
     * @return RerePlot实例
     */
    public static RerePlot scatter(IVector<?> x, IVector<?> y) {
        return new RerePlot().scatter(x, y);
    }
    
    /**
     * 创建散点图（支持样式字符串）
     * @param x X轴数据
     * @param y Y轴数据
     * @param styleString 样式字符串
     * @return RerePlot实例
     */
    public static RerePlot scatter(IVector<?> x, IVector<?> y, String styleString) {
        return new RerePlot().scatter(x, y, styleString);
    }
    
    /**
     * 创建分组散点图
     * @param x X轴数据
     * @param y Y轴数据
     * @param hue 颜色分组
     * @return RerePlot实例
     */
    public static RerePlot scatter(IVector<?> x, IVector<?> y, List<String> hue) {
        return new RerePlot().scatter(x, y, hue);
    }
    
    /**
     * 创建饼图
     * @param data 数据
     * @return RerePlot实例
     */
    public static RerePlot pie(IVector<?> data) {
        return new RerePlot().pie(data);
    }
    
    /**
     * 创建饼图（支持标签）
     * @param data 数据
     * @param labels 标签
     * @return RerePlot实例
     */
    public static RerePlot pie(IVector<?> data, List<String> labels) {
        return new RerePlot().pie(data, labels, (PlotStyle) null);
    }
    
    /**
     * 创建柱状图
     * @param data 数据
     * @return RerePlot实例
     */
    public static RerePlot bar(IVector<?> data) {
        return new RerePlot().bar(data);
    }
    
    /**
     * 创建分组柱状图
     * @param data 数据
     * @param hue 分组标签
     * @return RerePlot实例
     */
    public static RerePlot bar(IVector<?> data, List<String> hue) {
        return new RerePlot().bar(data, hue);
    }
    
    /**
     * 创建直方图
     * @param data 数据
     * @param fittingLine 是否显示拟合线
     * @return RerePlot实例
     */
    public static RerePlot hist(IVector<?> data, boolean fittingLine) {
        return new RerePlot().hist(data, fittingLine);
    }
    
    // ========== 极坐标图表工厂方法 ==========
    
    /**
     * 创建极坐标柱状图
     * @param data 数据
     * @param categories 类别标签
     * @return RerePlot实例
     */
    public static RerePlot polarBar(IVector<?> data, List<String> categories) {
        return new RerePlot().polarBar(data, categories);
    }
    
    /**
     * 创建极坐标线图
     * @param data 数据
     * @param categories 类别标签
     * @return RerePlot实例
     */
    public static RerePlot polarLine(IVector<?> data, List<String> categories) {
        return new RerePlot().polarLine(data, categories);
    }
    
    /**
     * 创建极坐标散点图
     * @param data 数据
     * @param categories 类别标签
     * @return RerePlot实例
     */
    public static RerePlot polarScatter(IVector<?> data, List<String> categories) {
        return new RerePlot().polarScatter(data, categories);
    }
    
    // ========== 统计图表工厂方法 ==========
    
    /**
     * 创建箱线图
     * @param data 数据
     * @param labels 标签
     * @return RerePlot实例
     */
    public static RerePlot boxplot(IVector<?> data, List<String> labels) {
        return new RerePlot().boxplot(data, labels);
    }
    
    /**
     * 创建小提琴图
     * @param data 数据
     * @return RerePlot实例
     */
    public static RerePlot violinplot(IVector<?> data) {
        RerePlot plot = new RerePlot();
        plot.violinplot(data);
        return plot;
    }
    
    /**
     * 创建小提琴图（支持标签）
     * @param data 数据
     * @param labels 标签
     * @return RerePlot实例
     */
    public static RerePlot violinplot(IVector<?> data, List<String> labels) {
        RerePlot plot = new RerePlot();
        plot.violinplot(data, labels);
        return plot;
    }
    
    /**
     * 创建K线图
     * @param data K线数据矩阵（每行包含[开盘价, 收盘价, 最低价, 最高价]）
     * @param dates 日期标签
     * @return RerePlot实例
     */
    public static RerePlot candlestick(IMatrix<?> data, List<String> dates) {
        return new RerePlot().candlestick(data, dates);
    }
    
    // ========== 特殊图表工厂方法 ==========
    
    /**
     * 创建漏斗图
     * @param data 数据
     * @param labels 标签
     * @return RerePlot实例
     */
    public static RerePlot funnel(IVector<?> data, List<String> labels) {
        return new RerePlot().funnel(data, labels);
    }
    
    /**
     * 创建桑基图
     * @param nodes 节点数据
     * @param links 连接数据
     * @return RerePlot实例
     */
    public static RerePlot sankey(List<Map<String, Object>> nodes, List<Map<String, Object>> links) {
        return new RerePlot().sankey(nodes, links);
    }
    
    /**
     * 创建旭日图
     * @param data 层次数据
     * @return RerePlot实例
     */
    public static RerePlot sunburst(List<Map<String, Object>> data) {
        return new RerePlot().sunburst(data);
    }
    
    /**
     * 创建主题河流图
     * @param data 时间序列数据
     * @param categories 类别标签
     * @return RerePlot实例
     */
    public static RerePlot themeRiver(List<Map<String, Object>> data, List<String> categories) {
        return new RerePlot().themeRiver(data, categories);
    }
    
    /**
     * 创建树图
     * @param data 树形数据
     * @return RerePlot实例
     */
    public static RerePlot tree(List<Map<String, Object>> data) {
        return new RerePlot().tree(data);
    }
    
    /**
     * 创建矩形树图
     * @param data 层次数据
     * @return RerePlot实例
     */
    public static RerePlot treemap(List<Map<String, Object>> data) {
        return new RerePlot().treemap(data);
    }
    
    /**
     * 创建关系图
     * @param nodes 节点数据
     * @param links 连接数据
     * @return RerePlot实例
     */
    public static RerePlot graph(List<Map<String, Object>> nodes, List<Map<String, Object>> links) {
        return new RerePlot().graph(nodes, links);
    }
    
    /**
     * 创建平行坐标图
     * @param data 多维数据矩阵
     * @param dimensions 维度标签
     * @return RerePlot实例
     */
    public static RerePlot parallel(IMatrix<?> data, List<String> dimensions) {
        return new RerePlot().parallel(data, dimensions);
    }
    
    // ========== 扩展图表工厂方法 ==========
    
    /**
     * 创建热力图
     * @param data 二维数据矩阵
     * @param xLabels X轴标签
     * @param yLabels Y轴标签
     * @return RerePlot实例
     */
    public static RerePlot heatmap(IMatrix<?> data, List<String> xLabels, List<String> yLabels) {
        return new RerePlot().heatmap(data, xLabels, yLabels);
    }
    
    /**
     * 创建雷达图
     * @param data 数据
     * @param indicators 指标标签
     * @return RerePlot实例
     */
    public static RerePlot radar(IVector<?> data, List<String> indicators) {
        return new RerePlot().radar(data, indicators);
    }
    
    /**
     * 创建仪表盘
     * @param value 当前值
     * @param max 最大值
     * @param min 最小值
     * @return RerePlot实例
     */
    public static RerePlot gauge(double value, double max, double min) {
        return new RerePlot().gauge(value, max, min);
    }
    
}
