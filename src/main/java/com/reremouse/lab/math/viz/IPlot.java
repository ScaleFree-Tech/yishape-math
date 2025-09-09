package com.reremouse.lab.math.viz;

import com.reremouse.lab.math.IMatrix;
import com.reremouse.lab.math.IVector;
import java.util.List;
import java.util.Map;

/**
 * 绘图接口，定义绘图实例的基本方法
 * @author lteb2
 */
public interface IPlot {
    
    // ========== 基础图表方法 ==========
    
    /**
     * 绘制线图
     * @param x X轴数据
     * @param y Y轴数据
     */
    void line(IVector x, IVector y);
    
    /**
     * 绘制单向量线图
     * @param x 数据向量
     */
    void line(IVector x);
    
    /**
     * 绘制多线图
     * @param x X轴数据
     * @param y Y轴数据
     * @param hue 分组标签
     */
    void line(IVector x, IVector y, List<String> hue);
    
    /**
     * 绘制散点图
     * @param x X轴数据
     * @param y Y轴数据
     */
    void scatter(IVector x, IVector y);
    
    /**
     * 绘制多组散点图
     * @param x X轴数据
     * @param y Y轴数据
     * @param hue 分组标签
     */
    void scatter(IVector x, IVector y, List<String> hue);
    
    /**
     * 绘制饼图
     * @param x 数据向量
     */
    void pie(IVector x);
    
    /**
     * 绘制柱状图
     * @param x 数据向量
     */
    void bar(IVector x);
    
    /**
     * 绘制分组柱状图
     * @param x 数据向量
     * @param hue 分组标签
     */
    void bar(IVector x, List<String> hue);
    
    /**
     * 绘制直方图
     * @param x 数据向量
     * @param fittingLine 是否显示拟合线
     */
    void hist(IVector x, boolean fittingLine);
    
    // ========== 极坐标图表方法 ==========
    
    /**
     * 绘制极坐标柱状图
     * @param data 数据向量
     * @param categories 类别标签
     */
    void polarBar(IVector data, List<String> categories);
    
    /**
     * 绘制极坐标线图
     * @param data 数据向量
     * @param categories 类别标签
     */
    void polarLine(IVector data, List<String> categories);
    
    /**
     * 绘制极坐标散点图
     * @param data 数据向量
     * @param categories 类别标签
     */
    void polarScatter(IVector data, List<String> categories);
    
    // ========== 统计图表方法 ==========
    
    /**
     * 绘制箱线图
     * @param data 数据向量
     * @param labels 标签
     */
    void boxplot(IVector data, List<String> labels);
    
    /**
     * 绘制K线图
     * @param data 数据矩阵，每行包含[开盘价, 收盘价, 最低价, 最高价]
     * @param dates 日期标签
     */
    void candlestick(IMatrix data, List<String> dates);
    
    // ========== 特殊图表方法 ==========
    
    /**
     * 绘制漏斗图
     * @param data 数据向量
     * @param labels 标签
     */
    void funnel(IVector data, List<String> labels);
    
    /**
     * 绘制桑基图
     * @param nodes 节点数据
     * @param links 连接数据
     */
    void sankey(List<Map<String, Object>> nodes, List<Map<String, Object>> links);
    
    /**
     * 绘制旭日图
     * @param data 层次数据
     */
    void sunburst(List<Map<String, Object>> data);
    
    /**
     * 绘制主题河流图
     * @param data 时间序列数据
     * @param categories 类别
     */
    void themeRiver(List<Map<String, Object>> data, List<String> categories);
    
    /**
     * 绘制树图
     * @param data 树形数据
     */
    void tree(List<Map<String, Object>> data);
    
    /**
     * 绘制矩形树图
     * @param data 层次数据
     */
    void treemap(List<Map<String, Object>> data);
    
    /**
     * 绘制关系图
     * @param nodes 节点数据
     * @param links 连接数据
     */
    void graph(List<Map<String, Object>> nodes, List<Map<String, Object>> links);
    
    /**
     * 绘制平行坐标图
     * @param data 数据矩阵
     * @param dimensions 维度名称
     */
    void parallel(IMatrix data, List<String> dimensions);
    
    // ========== 完善图表方法 ==========
    
    /**
     * 绘制热力图
     * @param data 二维数据矩阵
     * @param xLabels X轴标签
     * @param yLabels Y轴标签
     */
    void heatmap(IMatrix data, List<String> xLabels, List<String> yLabels);
    
    /**
     * 绘制雷达图
     * @param data 数据向量
     * @param indicators 指标名称
     */
    void radar(IVector data, List<String> indicators);
    
    /**
     * 绘制仪表盘
     * @param value 数值
     * @param max 最大值
     * @param min 最小值
     */
    void gauge(float value, float max, float min);
    
    // ========== 流式API方法 ==========
    
    /**
     * 设置图表标题（流式API）
     * @param titleText 标题文本
     * @return 当前实例，支持链式调用
     */
    IPlot title(String titleText);
    
    /**
     * 设置图表标题和副标题（流式API）
     * @param titleText 标题文本
     * @param subtitleText 副标题文本
     * @return 当前实例，支持链式调用
     */
    IPlot title(String titleText, String subtitleText);
    
    /**
     * 设置X轴标签（流式API）
     * @param name X轴标签名称
     * @return 当前实例，支持链式调用
     */
    IPlot xlabel(String name);
    
    /**
     * 设置Y轴标签（流式API）
     * @param name Y轴标签名称
     * @return 当前实例，支持链式调用
     */
    IPlot ylabel(String name);
    
    /**
     * 设置图表尺寸（流式API）
     * @param width 图表宽度
     * @param height 图表高度
     * @return 当前实例，支持链式调用
     */
    IPlot size(int width, int height);
    
    /**
     * 设置图表主题（流式API）
     * @param theme 主题名称
     * @return 当前实例，支持链式调用
     */
    IPlot theme(String theme);
    
    /**
     * 显示图表（流式API）
     * @return 当前实例，支持链式调用
     */
    IPlot show();
    
    /**
     * 保存图表为HTML文件（流式API）
     * @param filename 文件名
     * @return 当前实例，支持链式调用
     */
    IPlot saveAsHtml(String filename);
    
    // ========== 工具方法 ==========
    
    /**
     * 获取图表的HTML内容
     * @return HTML字符串
     */
    String toHtml();
    
    /**
     * 获取图表的JSON配置
     * @return JSON字符串
     */
    String toJson();
    
    // ========== 配置方法 ==========
    
    /**
     * 设置图表标题
     * @param titleText 标题文本
     */
    void setTitle(String titleText);
    
    /**
     * 设置图表标题和副标题
     * @param titleText 标题文本
     * @param subtitleText 副标题文本
     */
    void setTitle(String titleText, String subtitleText);
    
    /**
     * 设置X轴标签
     * @param name X轴标签名称
     */
    void setXlabel(String name);
    
    /**
     * 设置Y轴标签
     * @param name Y轴标签名称
     */
    void setYlabel(String name);
    
    /**
     * 设置X轴刻度
     * @param xticks X轴刻度配置
     */
    void setXticks(com.reremouse.lab.math.viz.AxisTicks xticks);
    
    /**
     * 设置Y轴刻度
     * @param yticks Y轴刻度配置
     */
    void setYticks(com.reremouse.lab.math.viz.AxisTicks yticks);
    
    /**
     * 获取图表宽度
     * @return 图表宽度
     */
    int getWidth();
    
    /**
     * 获取图表高度
     * @return 图表高度
     */
    int getHeight();
    
    /**
     * 获取图表主题
     * @return 主题名称
     */
    String getTheme();
}