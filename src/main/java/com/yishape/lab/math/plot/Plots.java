package com.yishape.lab.math.plot;

import com.yishape.lab.math.plot.echarts.EchartsPlot;
import com.yishape.lab.math.plot.echarts.j3d.Echarts3dPlot;
import com.yishape.lab.math.plot.echarts.EchartsThemeManager;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.plot.I3dPlot;
import com.yishape.lab.math.plot.javafx.j3d.JavaFx3dPlot;
import com.yishape.lab.math.plot.javafx.JavaFxPlot;
import com.yishape.lab.math.plot.svg.SvgPlot;

/**
 * 绘图静态工厂：通过 {@link #of()}、{@link #theme(String)} 等入口创建 {@link IPlot}， 具体渲染后端由
 * {@link PlotProvider} 决定（默认 {@link PlotProvider#JavaFx}，可切换为 ECharts）。
 * <p>
 * Static entry points for charts; each method delegates to {@link IPlot} on a
 * fresh plot instance. Use {@link IPlot} for the complete method list; this
 * class mirrors the most common factories for concise call sites.
 *
 * @author lteb2
 * @see IPlot
 * @see PlotProvider
 * @see I3dPlot
 */
public final class Plots {

    private static PlotProvider provider = PlotProvider.JavaFx;
    private static PlotProvider3d provider3d = PlotProvider3d.EchartsGL; // 默认使用ECharts 3D

    /**
     * 3D绘图后端枚举
     */
    public enum PlotProvider3d {
        /**
         * JavaFX 3D (桌面应用)
         */
        JavaFx,
        /**
         * ECharts GL (Web/交互式)
         */
        EchartsGL
    }

    /**
     * 设置全局默认绘图后端（影响后续所有 {@link #of()} 及基于 {@code of()} 的静态工厂方法）。
     *
     * @param provider 后端枚举，不可为 {@code null}
     */
    public static void setProvider(PlotProvider provider) {
        Plots.provider = provider;
    }

    /**
     * 设置全局默认3D绘图后端（影响后续所有 {@link #of3d()} 方法）。
     *
     * @param provider3d 3D后端枚举，不可为 {@code null}
     */
    public static void setProvider3d(PlotProvider3d provider3d) {
        Plots.provider3d = provider3d;
    }

//    private Plots() {
//        // 工具类，防止实例化
//        // Utility class, prevent instantiation
//    }
    // ========== Basic Factory Methods 基础工厂方法 ==========
    /**
     * Create a default plot object 创建默认绘图对象
     *
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot of() {
        return switch (provider) {
            case PlotProvider.JavaFx ->
                new JavaFxPlot();
            case PlotProvider.Echarts ->
                new EchartsPlot();
            case PlotProvider.Svg ->
                new SvgPlot();
            default ->
                new JavaFxPlot();
        };
    }

    /**
     * Create a plot object with specified dimensions 创建指定尺寸的绘图对象
     * <p>
     * JavaFX 实现：默认使用 {@code seaborn} 报告风主题（muted 调色与浅底网格）， 序列线色/标记色随主题调色板对齐。显式
     * {@link #of(int, int, String)} 可改用其它内置主题。</p>
     *
     * @param width Chart width 图表宽度
     * @param height Chart height 图表高度
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot of(int width, int height) {
        return switch (provider) {
            case PlotProvider.JavaFx ->
                new JavaFxPlot(width, height);
            case PlotProvider.Echarts ->
                new EchartsPlot(width, height);
            case PlotProvider.Svg ->
                new SvgPlot(width, height);
            default ->
                new JavaFxPlot(width, height);
        };
    }

    /**
     * Create a plot object with specified dimensions and theme 创建指定尺寸和主题的绘图对象
     *
     * @param width Chart width 图表宽度
     * @param height Chart height 图表高度
     * @param theme Theme name 主题名称
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot of(int width, int height, String theme) {
        return switch (provider) {
            case PlotProvider.JavaFx ->
                new JavaFxPlot(width, height, theme);
            case PlotProvider.Echarts ->
                new EchartsPlot(width, height, theme);
            case PlotProvider.Svg ->
                new SvgPlot(width, height, theme);
            default ->
                new JavaFxPlot(width, height, theme);
        };
    }

    /**
     * 创建三维绘图实例。与 {@link #of()} 的二维后端选择无关： 3D后端由
     * {@link #setProvider3d(PlotProvider3d)} 决定（默认 ECharts GL）。
     *
     * @return 可链式调用的 {@link I3dPlot}
     */
    public static I3dPlot of3d() {
        return switch (provider3d) {
            case PlotProvider3d.JavaFx ->
                new JavaFx3dPlot();
            case PlotProvider3d.EchartsGL ->
                new Echarts3dPlot();
            default ->
                new Echarts3dPlot();
        };
    }

    /**
     * 创建指定画布尺寸的三维绘图实例。
     *
     * @param width 宽度（像素）
     * @param height 高度（像素）
     * @return 可链式调用的 {@link I3dPlot}
     */
    public static I3dPlot of3d(int width, int height) {
        return switch (provider3d) {
            case PlotProvider3d.JavaFx ->
                new JavaFx3dPlot(width, height);
            case PlotProvider3d.EchartsGL ->
                new Echarts3dPlot(width, height);
            default ->
                new Echarts3dPlot(width, height);
        };
    }

    /**
     * 创建指定尺寸与主题的三维绘图实例。
     *
     * @param width 宽度（像素）
     * @param height 高度（像素）
     * @param theme 主题名
     * @return 可链式调用的 {@link I3dPlot}
     */
    public static I3dPlot of3d(int width, int height, String theme) {
        return switch (provider3d) {
            case PlotProvider3d.JavaFx ->
                new JavaFx3dPlot(width, height, theme);
            case PlotProvider3d.EchartsGL ->
                new Echarts3dPlot(width, height, theme);
            default ->
                new Echarts3dPlot(width, height, theme);
        };
    }

    /**
     * 新建绘图实例并设置主题名，等价 {@code Plots.of().theme(themeName)}。
     *
     * @param themeName 已注册或内置主题名称
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot theme(String themeName) {
        return of().theme(themeName);
    }

    /**
     * 与 {@link #theme(String)} 相同，命名上与 {@link IPlot#theme(String)} 呼应。
     *
     * @param theme 主题名
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot setTheme(String theme) {
        return of().theme(theme);
    }

    /**
     * 新建矢量SVG绘图实例（支持箱线图、直方图等，输出真正矢量SVG）。 与 {@link #of()} 的后端实现无关，始终生成SVG文件。
     *
     * @return 可链式调用的 {@link SvgPlot}
     */
    public static IPlot ofJavaFx() {
        return new JavaFxPlot();
    }

    /**
     * 新建指定尺寸的矢量SVG绘图实例。
     *
     * @param width 宽度
     * @param height 高度
     * @return 可链式调用的 {@link SvgPlot}
     */
    public static IPlot ofJavaFx(int width, int height) {
        return new JavaFxPlot(width, height);
    }

    /**
     * 新建指定尺寸和主题的矢量SVG绘图实例。
     *
     * @param width 宽度
     * @param height 高度
     * @param theme 主题名
     * @return 可链式调用的 {@link SvgPlot}
     */
    public static IPlot ofJavaFx(int width, int height, String theme) {
        return new JavaFxPlot(width, height, theme);
    }

    /**
     * 新建矢量SVG绘图实例（支持箱线图、直方图等，输出真正矢量SVG）。 与 {@link #of()} 的后端实现无关，始终生成SVG文件。
     *
     * @return 可链式调用的 {@link SvgPlot}
     */
    public static IPlot ofEcharts() {
        return new EchartsPlot();
    }

    /**
     * 新建指定尺寸的矢量SVG绘图实例。
     *
     * @param width 宽度
     * @param height 高度
     * @return 可链式调用的 {@link SvgPlot}
     */
    public static IPlot ofEcharts(int width, int height) {
        return new EchartsPlot(width, height);
    }

    /**
     * 新建指定尺寸和主题的矢量SVG绘图实例。
     *
     * @param width 宽度
     * @param height 高度
     * @param theme 主题名
     * @return 可链式调用的 {@link SvgPlot}
     */
    public static IPlot ofEcharts(int width, int height, String theme) {
        return new EchartsPlot(width, height, theme);
    }

    //
    /**
     * 新建矢量 SVG 绘图实例（与 {@link #of()} 的后端无关，始终为 {@link SvgPlot}）。
     *
     * @return 可链式调用的 {@link SvgPlot}（含仅 SVG 后端支持的 API）
     */
    public static SvgPlot ofSvg() {
        return new SvgPlot();
    }

    /**
     * 新建指定尺寸的矢量 SVG 绘图实例。
     *
     * @param width 宽度
     * @param height 高度
     * @return {@link SvgPlot}
     */
    public static SvgPlot ofSvg(int width, int height) {
        return new SvgPlot(width, height);
    }

    /**
     * 新建指定尺寸和主题的矢量 SVG 绘图实例。
     *
     * @param width 宽度
     * @param height 高度
     * @param theme 主题名
     * @return {@link SvgPlot}
     */
    public static SvgPlot ofSvg(int width, int height, String theme) {
        return new SvgPlot(width, height, theme);
    }

    /**
     * 与 {@link #ofSvg(int, int)} 相同，便于与旧示例命名一致。
     */
    public static SvgPlot svg(int width, int height) {
        return ofSvg(width, height);
    }

    // ========== Chart Type Specific Factory Methods 图表类型专用工厂方法 ==========
    // ========== Basic Chart Factory Methods 基础图表工厂方法 ==========
    /**
     * Create a line chart 创建线图
     *
     * @param x X-axis data X轴数据
     * @param y Y-axis data Y轴数据
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot line(IVector<?> x, IVector<?> y) {
        return of().line(x, y);
    }

    /**
     * Create a line chart (with style string support) 创建线图（支持样式字符串）
     *
     * @param x X-axis data X轴数据
     * @param y Y-axis data Y轴数据
     * @param styleString Style string 样式字符串
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot line(IVector<?> x, IVector<?> y, String styleString) {
        return of().line(x, y, styleString);
    }

    /**
     * Create a single vector line chart 创建单向量线图
     *
     * @param y Y-axis data Y轴数据
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot line(IVector<?> y) {
        return of().line(y);
    }

    /**
     * Create a grouped line chart 创建分组线图
     *
     * @param x X-axis data X轴数据
     * @param y Y-axis data Y轴数据
     * @param hue Color grouping 颜色分组
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot line(IVector<?> x, IVector<?> y, List<String> hue) {
        return of().line(x, y, hue);
    }

    /**
     * Create a scatter plot 创建散点图
     *
     * @param x X-axis data X轴数据
     * @param y Y-axis data Y轴数据
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot scatter(IVector<?> x, IVector<?> y) {
        return of().scatter(x, y);
    }

    /**
     * Create a scatter plot (with style string support) 创建散点图（支持样式字符串）
     *
     * @param x X-axis data X轴数据
     * @param y Y-axis data Y轴数据
     * @param styleString Style string 样式字符串
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot scatter(IVector<?> x, IVector<?> y, String styleString) {
        return of().scatter(x, y, styleString);
    }

    /**
     * Create a grouped scatter plot 创建分组散点图
     *
     * @param x X-axis data X轴数据
     * @param y Y-axis data Y轴数据
     * @param hue Color grouping 颜色分组
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot scatter(IVector<?> x, IVector<?> y, List<String> hue) {
        return of().scatter(x, y, hue);
    }

    /**
     * Create a pie chart 创建饼图
     *
     * @param data Data 数据
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot pie(IVector<?> data) {
        return of().pie(data);
    }

    /**
     * Create a pie chart (with label support) 创建饼图（支持标签）
     *
     * @param data Data 数据
     * @param labels Labels 标签
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot pie(IVector<?> data, List<String> labels) {
        return of().pie(data, labels, (PlotStyle) null);
    }

    /**
     * 饼图（样式字符串，解析规则与各 {@link IPlot} 实现一致）。
     *
     * @param data 扇区数值
     * @param styleString 样式表达式
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot pie(IVector<?> data, String styleString) {
        return of().pie(data, styleString);
    }

    /**
     * 饼图（显式 {@link PlotStyle}）。
     */
    public static IPlot pie(IVector<?> data, PlotStyle style) {
        return of().pie(data, style);
    }

    /**
     * 饼图（扇区标签 + 样式字符串）。
     */
    public static IPlot pie(IVector<?> data, List<String> labels, String styleString) {
        return of().pie(data, labels, styleString);
    }

    /**
     * 饼图（扇区标签 + {@link PlotStyle}）。
     */
    public static IPlot pie(IVector<?> data, List<String> labels, PlotStyle style) {
        return of().pie(data, labels, style);
    }

    /**
     * Create a bar chart 创建柱状图
     *
     * @param data Data 数据
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot bar(IVector<?> data) {
        return of().bar(data);
    }

    /**
     * Create a grouped bar chart 创建分组柱状图（与 {@link IPlot#bar(IVector, List)}
     * 相同：可为单向量 + hue 或自动生成类别序）。
     *
     * @param dataOrY Data or Y values 数据或 Y 轴数值
     * @param hue Grouping labels 分组标签
     * @return plot instance 绘图实例
     */
    public static IPlot bar(IVector<?> dataOrY, List<String> hue) {
        return of().bar(dataOrY, hue);
    }

    /**
     * 单向量柱状图（数值向量，类别可由实现默认生成）。
     */
    public static IPlot bar(IVector<?> x, String styleString) {
        return of().bar(x, styleString);
    }

    /**
     * 单向量柱状图（{@link PlotStyle}）。
     */
    public static IPlot bar(IVector<?> x, PlotStyle style) {
        return of().bar(x, style);
    }

    /**
     * 按 hue 分组的柱状图（显式 {@link PlotStyle}）。
     */
    public static IPlot bar(IVector<?> x, List<String> hue, PlotStyle style) {
        return of().bar(x, hue, style);
    }

    /**
     * 按 hue 分组的柱状图（样式字符串）。
     */
    public static IPlot bar(IVector<?> x, List<String> hue, String styleString) {
        return of().bar(x, hue, styleString);
    }

    /**
     * Create a histogram 创建直方图
     *
     * @param data Data 数据
     * @param fittingLine Whether to display the fitting line 是否显示拟合线
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot hist(IVector<?> data, boolean fittingLine) {
        return of().hist(data, fittingLine);
    }

    /**
     * 直方图（样式字符串）。
     *
     * @param data 样本数据
     * @param fittingLine 是否叠加拟合曲线（由实现定义）
     * @param styleString 样式表达式
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot hist(IVector<?> data, boolean fittingLine, String styleString) {
        return of().hist(data, fittingLine, styleString);
    }

    /**
     * 直方图（{@link PlotStyle}）。
     */
    public static IPlot hist(IVector<?> data, boolean fittingLine, PlotStyle style) {
        return of().hist(data, fittingLine, style);
    }

    /**
     * 直方图（显式样式与分箱数；{@code bins} 可为 {@code null} 表示使用实现默认箱数）。
     *
     * @param bins 箱数，可能为 {@code null}
     */
    public static IPlot hist(IVector<?> data, boolean fittingLine, PlotStyle style, Integer bins) {
        return of().hist(data, fittingLine, style, bins);
    }

    /**
     * 柱状图：类别标签 {@code xticks} 与数值向量 {@code y} 一一对应。
     *
     * @param xticks X 轴类别名
     * @param y 柱高
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot bar(List<String> xticks, IVector<?> y) {
        return of().bar(xticks, y);
    }

    /**
     * 分组柱状图：在 {@code xticks}、{@code y} 基础上按 {@code hue} 分组着色/分系列。
     */
    public static IPlot bar(List<String> xticks, IVector<?> y, List<String> hue) {
        return of().bar(xticks, y, hue);
    }

    /**
     * 单向量线图（样式字符串）。
     */
    public static IPlot line(IVector<?> y, String styleString) {
        return of().line(y, styleString);
    }

    /**
     * 单向量线图（PlotStyle）。
     */
    public static IPlot line(IVector<?> y, PlotStyle style) {
        return of().line(y, style);
    }

    /**
     * 线图（PlotStyle）。
     */
    public static IPlot line(IVector<?> x, IVector<?> y, PlotStyle style) {
        return of().line(x, y, style);
    }

    /**
     * 分组线图（hue + 每组样式串）。
     */
    public static IPlot line(IVector<?> x, IVector<?> y, List<String> hue, List<String> styleGroup) {
        return of().line(x, y, hue, styleGroup);
    }

    /**
     * 散点图（PlotStyle）。
     */
    public static IPlot scatter(IVector<?> x, IVector<?> y, PlotStyle style) {
        return of().scatter(x, y, style);
    }

    // ========== Polar Coordinate Chart Factory Methods 极坐标图表工厂方法 ==========
    /**
     * Create a polar bar chart 创建极坐标柱状图
     *
     * @param data Data 数据
     * @param categories Category labels 类别标签
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot polarBar(IVector<?> data, List<String> categories) {
        return of().polarBar(data, categories);
    }

    /**
     * 极坐标柱状图（样式字符串）。
     */
    public static IPlot polarBar(IVector<?> data, List<String> categories, String styleString) {
        return of().polarBar(data, categories, styleString);
    }

    /**
     * 极坐标柱状图（{@link PlotStyle}）。
     */
    public static IPlot polarBar(IVector<?> data, List<String> categories, PlotStyle style) {
        return of().polarBar(data, categories, style);
    }

    /**
     * Create a polar line chart 创建极坐标线图
     *
     * @param data Data 数据
     * @param categories Category labels 类别标签
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot polarLine(IVector<?> data, List<String> categories) {
        return of().polarLine(data, categories);
    }

    /**
     * 极坐标折线图（样式字符串）。
     */
    public static IPlot polarLine(IVector<?> data, List<String> categories, String styleString) {
        return of().polarLine(data, categories, styleString);
    }

    /**
     * 极坐标折线图（{@link PlotStyle}）。
     */
    public static IPlot polarLine(IVector<?> data, List<String> categories, PlotStyle style) {
        return of().polarLine(data, categories, style);
    }

    /**
     * Create a polar scatter plot 创建极坐标散点图
     *
     * @param data Data 数据
     * @param categories Category labels 类别标签
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot polarScatter(IVector<?> data, List<String> categories) {
        return of().polarScatter(data, categories);
    }

    /**
     * 极坐标散点图（样式字符串）。
     */
    public static IPlot polarScatter(IVector<?> data, List<String> categories, String styleString) {
        return of().polarScatter(data, categories, styleString);
    }

    /**
     * 极坐标散点图（{@link PlotStyle}）。
     */
    public static IPlot polarScatter(IVector<?> data, List<String> categories, PlotStyle style) {
        return of().polarScatter(data, categories, style);
    }

    // ========== Statistical Chart Factory Methods 统计图表工厂方法 ==========
    /**
     * 箱线图（单组、无标签）。
     */
    public static IPlot boxplot(IVector<?> data) {
        IPlot p = of();
        p.boxplot(data);
        return p;
    }

    /**
     * 箱线图（带标签）。
     */
    public static IPlot boxplot(IVector<?> data, List<String> labels) {
        return of().boxplot(data, labels);
    }

    /**
     * 箱线图（标签 + 样式字符串）。
     */
    public static IPlot boxplot(IVector<?> data, List<String> labels, String styleString) {
        return of().boxplot(data, labels, styleString);
    }

    /**
     * 箱线图（标签 + {@link PlotStyle}）。
     */
    public static IPlot boxplot(IVector<?> data, List<String> labels, PlotStyle style) {
        return of().boxplot(data, labels, style);
    }

    /**
     * Create a violin plot 创建小提琴图
     *
     * @param data Data 数据
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot violinplot(IVector<?> data) {
        IPlot p = of();
        p.violinplot(data);
        return p;
    }

    /**
     * 小提琴图（样式字符串）。
     */
    public static IPlot violinplot(IVector<?> data, String styleString) {
        return of().violinplot(data, styleString);
    }

    /**
     * 小提琴图（{@link PlotStyle}）。
     */
    public static IPlot violinplot(IVector<?> data, PlotStyle style) {
        return of().violinplot(data, style);
    }

    /**
     * Create a violin plot (with label support) 创建小提琴图（支持标签）
     *
     * @param data Data 数据
     * @param labels Labels 标签
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot violinplot(IVector<?> data, List<String> labels) {
        IPlot p = of();
        p.violinplot(data, labels);
        return p;
    }

    /**
     * 带分组标签的小提琴图（样式字符串）。
     */
    public static IPlot violinplot(IVector<?> data, List<String> labels, String styleString) {
        return of().violinplot(data, labels, styleString);
    }

    /**
     * 带分组标签的小提琴图（{@link PlotStyle}）。
     */
    public static IPlot violinplot(IVector<?> data, List<String> labels, PlotStyle style) {
        return of().violinplot(data, labels, style);
    }

    /**
     * Create a candlestick chart 创建K线图
     *
     * @param data K-line data matrix (each row contains [open price, close
     * price, lowest price, highest price]) K线数据矩阵（每行包含[开盘价, 收盘价, 最低价, 最高价]）
     * @param dates Date labels 日期标签
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot candlestick(IMatrix<?> data, List<String> dates) {
        return of().candlestick(data, dates);
    }

    /**
     * K 线图（样式字符串）。
     */
    public static IPlot candlestick(IMatrix<?> data, List<String> dates, String styleString) {
        return of().candlestick(data, dates, styleString);
    }

    /**
     * K 线图（{@link PlotStyle}）。
     */
    public static IPlot candlestick(IMatrix<?> data, List<String> dates, PlotStyle style) {
        return of().candlestick(data, dates, style);
    }

    /**
     * 漏斗图。
     *
     * @param data 各段数值
     * @param labels 各段标签
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot funnel(IVector<?> data, List<String> labels) {
        return of().funnel(data, labels);
    }

    /**
     * 漏斗图（样式字符串）。
     */
    public static IPlot funnel(IVector<?> data, List<String> labels, String styleString) {
        return of().funnel(data, labels, styleString);
    }

    /**
     * 漏斗图（{@link PlotStyle}）。
     */
    public static IPlot funnel(IVector<?> data, List<String> labels, PlotStyle style) {
        return of().funnel(data, labels, style);
    }

    /**
     * Create a Sankey diagram 创建桑基图
     *
     * @param nodes Node data 节点数据
     * @param links Connection data 连接数据
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot sankey(List<Map<String, Object>> nodes, List<Map<String, Object>> links) {
        return of().sankey(nodes, links);
    }

    /**
     * 桑基图（样式字符串）。
     */
    public static IPlot sankey(List<Map<String, Object>> nodes, List<Map<String, Object>> links,
            String styleString) {
        return of().sankey(nodes, links, styleString);
    }

    /**
     * 桑基图（{@link PlotStyle}）。
     */
    public static IPlot sankey(List<Map<String, Object>> nodes, List<Map<String, Object>> links,
            PlotStyle style) {
        return of().sankey(nodes, links, style);
    }

    /**
     * Create a sunburst chart 创建旭日图
     *
     * @param data Hierarchical data 层次数据
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot sunburst(List<Map<String, Object>> data) {
        return of().sunburst(data);
    }

    /**
     * 旭日图（样式字符串）。
     */
    public static IPlot sunburst(List<Map<String, Object>> data, String styleString) {
        return of().sunburst(data, styleString);
    }

    /**
     * 旭日图（{@link PlotStyle}）。
     */
    public static IPlot sunburst(List<Map<String, Object>> data, PlotStyle style) {
        return of().sunburst(data, style);
    }

    /**
     * Create a theme river chart 创建主题河流图
     *
     * @param data Time series data 时间序列数据
     * @param categories Category labels 类别标签
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot themeRiver(List<Map<String, Object>> data, List<String> categories) {
        return of().themeRiver(data, categories);
    }

    /**
     * 主题河流图（样式字符串）。
     */
    public static IPlot themeRiver(List<Map<String, Object>> data, List<String> categories,
            String styleString) {
        return of().themeRiver(data, categories, styleString);
    }

    /**
     * 主题河流图（{@link PlotStyle}）。
     */
    public static IPlot themeRiver(List<Map<String, Object>> data, List<String> categories, PlotStyle style) {
        return of().themeRiver(data, categories, style);
    }

    /**
     * Create a tree diagram 创建树图
     *
     * @param data Tree data 树形数据
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot tree(List<Map<String, Object>> data) {
        return of().tree(data);
    }

    /**
     * 树图（样式字符串）。
     */
    public static IPlot tree(List<Map<String, Object>> data, String styleString) {
        return of().tree(data, styleString);
    }

    /**
     * 树图（{@link PlotStyle}）。
     */
    public static IPlot tree(List<Map<String, Object>> data, PlotStyle style) {
        return of().tree(data, style);
    }

    /**
     * Create a treemap 创建矩形树图
     *
     * @param data Hierarchical data 层次数据
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot treemap(List<Map<String, Object>> data) {
        return of().treemap(data);
    }

    /**
     * 矩形树图（样式字符串）。
     */
    public static IPlot treemap(List<Map<String, Object>> data, String styleString) {
        return of().treemap(data, styleString);
    }

    /**
     * 矩形树图（{@link PlotStyle}）。
     */
    public static IPlot treemap(List<Map<String, Object>> data, PlotStyle style) {
        return of().treemap(data, style);
    }

    /**
     * Create a relationship graph 创建关系图
     *
     * @param nodes Node data 节点数据
     * @param links Connection data 连接数据
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot graph(List<Map<String, Object>> nodes, List<Map<String, Object>> links) {
        return of().graph(nodes, links);
    }

    /**
     * 关系图（样式字符串）。
     */
    public static IPlot graph(List<Map<String, Object>> nodes, List<Map<String, Object>> links,
            String styleString) {
        return of().graph(nodes, links, styleString);
    }

    /**
     * 关系图（{@link PlotStyle}）。
     */
    public static IPlot graph(List<Map<String, Object>> nodes, List<Map<String, Object>> links,
            PlotStyle style) {
        return of().graph(nodes, links, style);
    }

    /**
     * Create a parallel coordinates plot 创建平行坐标图
     *
     * @param data Multidimensional data matrix 多维数据矩阵
     * @param dimensions Dimension labels 维度标签
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot parallel(IMatrix<?> data, List<String> dimensions) {
        return of().parallel(data, dimensions);
    }

    /**
     * 平行坐标图（样式字符串）。
     */
    public static IPlot parallel(IMatrix<?> data, List<String> dimensions, String styleString) {
        return of().parallel(data, dimensions, styleString);
    }

    /**
     * 平行坐标图（{@link PlotStyle}）。
     */
    public static IPlot parallel(IMatrix<?> data, List<String> dimensions, PlotStyle style) {
        return of().parallel(data, dimensions, style);
    }

    // ========== Extended Chart Factory Methods 扩展图表工厂方法 ==========
    /**
     * Create a heatmap 创建热力图
     *
     * @param data 2D data matrix 二维数据矩阵
     * @param xLabels X-axis labels X轴标签
     * @param yLabels Y-axis labels Y轴标签
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot heatmap(IMatrix<?> data, List<String> xLabels, List<String> yLabels) {
        return of().heatmap(data, xLabels, yLabels);
    }

    /**
     * 热力图（带轴标签 + 样式字符串）。
     */
    public static IPlot heatmap(IMatrix<?> data, List<String> xLabels, List<String> yLabels,
            String styleString) {
        return of().heatmap(data, xLabels, yLabels, styleString);
    }

    /**
     * 热力图（带轴标签 + {@link PlotStyle}）。
     */
    public static IPlot heatmap(IMatrix<?> data, List<String> xLabels, List<String> yLabels,
            PlotStyle style) {
        return of().heatmap(data, xLabels, yLabels, style);
    }

    /**
     * Create a heatmap 创建热力图
     *
     * @param data 2D data matrix 二维数据矩阵
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot heatmap(IMatrix<?> data) {
        return of().heatmap(data);
    }

    /**
     * Create a radar chart 创建雷达图
     *
     * @param data Data 数据
     * @param indicators Indicator labels 指标标签
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot radar(IVector<?> data, List<String> indicators) {
        return of().radar(data, indicators);
    }

    /**
     * 雷达图（样式字符串）。
     */
    public static IPlot radar(IVector<?> data, List<String> indicators, String styleString) {
        return of().radar(data, indicators, styleString);
    }

    /**
     * 雷达图（{@link PlotStyle}）。
     */
    public static IPlot radar(IVector<?> data, List<String> indicators, PlotStyle style) {
        return of().radar(data, indicators, style);
    }

    /**
     * Create a gauge chart 创建仪表盘
     *
     * @param value Current value 当前值
     * @param max Maximum value 最大值
     * @param min Minimum value 最小值
     * @return 可链式调用的 {@link IPlot}（实现由当前 {@link PlotProvider} 决定）
     */
    public static IPlot gauge(double value, double max, double min) {
        return of().gauge(value, max, min);
    }

    /**
     * 仪表盘（样式字符串）。
     */
    public static IPlot gauge(double value, double max, double min, String styleString) {
        return of().gauge(value, max, min, styleString);
    }

    /**
     * 仪表盘（{@link PlotStyle}）。
     */
    public static IPlot gauge(double value, double max, double min, PlotStyle style) {
        return of().gauge(value, max, min, style);
    }

    // ========== Seaborn/Matplotlib 风格笛卡尔扩展（与 {@link IPlot} 对齐）==========
    /**
     * 设置后续笛卡尔图 X 轴刻度类型（线性 / 对数），见 {@link IPlot#xscale(PlotAxisScale)}。
     */
    public static IPlot xscale(PlotAxisScale scale) {
        return of().xscale(scale);
    }

    /**
     * 设置后续笛卡尔图 Y 轴刻度类型。
     */
    public static IPlot yscale(PlotAxisScale scale) {
        return of().yscale(scale);
    }

    /**
     * 双 Y 轴时右侧轴标题，见 {@link IPlot#lineWithSecondaryY}。
     */
    public static IPlot y2label(String label) {
        return of().y2label(label);
    }

    /**
     * 面积图（X–Y 折线下方填充至基线，行为由实现决定）。
     */
    public static IPlot area(IVector<?> x, IVector<?> y) {
        return of().area(x, y);
    }

    /**
     * 阶梯图（先水平后垂直，post-step 语义）。
     */
    public static IPlot step(IVector<?> x, IVector<?> y) {
        return of().step(x, y);
    }

    /**
     * 水平条形图：类别在纵轴，数值在横轴。
     */
    public static IPlot barh(List<String> categories, IVector<?> values) {
        return of().barh(categories, values);
    }

    /**
     * 堆叠柱：矩阵每行一层、每列对应 {@code categories} 中一项。
     */
    public static IPlot barStacked(List<String> categories, IMatrix<?> values, List<String> layerNames) {
        return of().barStacked(categories, values, layerNames);
    }

    /**
     * 误差棒：在每个 (x,y) 处绘制竖向 y±yerr。
     */
    public static IPlot errorbar(IVector<?> x, IVector<?> y, IVector<?> yerr) {
        return of().errorbar(x, y, yerr);
    }

    /**
     * 气泡散点：{@code sizes} 映射为标记面积/半径（由各实现解释）。
     */
    public static IPlot scatter(IVector<?> x, IVector<?> y, IVector<?> sizes) {
        return of().scatter(x, y, sizes);
    }

    /**
     * 散点 + OLS 回归线（无置信带），见 {@link IPlot#regplot(IVector, IVector)}。
     */
    public static IPlot regplot(IVector<?> x, IVector<?> y) {
        return of().regplot(x, y);
    }

    /**
     * 正态 Q–Q 图（理论分位数 vs 有序样本）。
     */
    public static IPlot qqplot(IVector<?> data) {
        IPlot p = of();
        p.qqplot(data);
        return p;
    }

    /**
     * 双 Y 轴折线：共享 X，左 {@code yLeft}、右 {@code yRight}。
     */
    public static IPlot lineWithSecondaryY(IVector<?> x, IVector<?> yLeft, IVector<?> yRight) {
        return of().lineWithSecondaryY(x, yLeft, yRight);
    }

    // ========== 默认样式 / 调色板 / 主题系统（链式起点，委托 {@link IPlot}）==========
    /**
     * 设置启用样式系统时的默认序列样式基底，见 {@link IPlot#setDefaultStyle(PlotStyle)}。
     */
    public static IPlot setDefaultStyle(PlotStyle style) {
        return of().setDefaultStyle(style);
    }

    /**
     * 按名称设置色板（须已在 {@link ColorPalette} 注册）。
     */
    public static IPlot setPalette(String paletteName) {
        return of().setPalette(paletteName);
    }

    /**
     * 启用或关闭序列样式系统（解析 {@link PlotStyle} / 样式字符串等）。
     */
    public static IPlot enableStyleSystem(boolean enabled) {
        return of().enableStyleSystem(enabled);
    }

    /**
     * 在已启用主题系统时应用已注册或内置主题名，见 {@link IPlot#applyTheme(String)}。
     */
    public static IPlot applyTheme(String themeName) {
        return of().applyTheme(themeName);
    }

    /**
     * 启用或关闭主题系统（与 {@link IPlot#enableThemeSystem(boolean)} 对齐）。
     */
    public static IPlot enableThemeSystem(boolean enabled) {
        return of().enableThemeSystem(enabled);
    }

    /**
     * 注册自定义 ECharts 主题配置（部分实现会同步到 JavaFX 主题注册表，见各 {@link IPlot} 实现）。
     *
     * @param themeName 主题名
     * @param theme ECharts 侧主题定义
     */
    public static IPlot registerTheme(String themeName,
            com.yishape.lab.math.plot.echarts.EchartsThemeManager.CustomTheme theme) {
        return of().registerTheme(themeName, theme);
    }

    /**
     * 创建并注册渐变主题（起止色与背景色），见 {@link IPlot#createGradientTheme}。
     */
    public static IPlot createGradientTheme(String themeName, String startColor, String endColor,
            String backgroundColor) {
        return of().createGradientTheme(themeName, startColor, endColor, backgroundColor);
    }

    /**
     * 创建并注册单色主题，见 {@link IPlot#createMonochromeTheme}。
     */
    public static IPlot createMonochromeTheme(String themeName, String baseColor, String backgroundColor) {
        return of().createMonochromeTheme(themeName, baseColor, backgroundColor);
    }

    /**
     * 分面子图网格：创建 {@code rows}×{@code cols} 个格子，后续用
     * {@link IPlot#subplot(int, int)} 选择当前格再绘图。 JavaFX 与 ECharts
     * 实现均实现该流程；未选格时的默认行为见各实现类说明。
     */
    public static IPlot subplots(int rows, int cols) {
        return of().subplots(rows, cols);
    }

    /**
     * 一维核密度曲线（默认网格 256 点、带宽 0 表示 Scott 规则），见
     * {@link IPlot#kdeplot(IVector, int, double)}。
     */
    public static IPlot kdeplot(IVector<?> data) {
        return of().kdeplot(data, 256, 0.0);
    }

    /**
     * 一维核密度曲线（指定网格点数与带宽；{@code bandwidth == 0} 通常表示自动带宽）。
     */
    public static IPlot kdeplot(IVector<?> data, int gridPoints, double bandwidth) {
        return of().kdeplot(data, gridPoints, bandwidth);
    }

    /**
     * 散点 + 回归线；{@code confidenceBand == true} 时由实现绘制均值响应置信带（通常为约 95%）。
     */
    public static IPlot regplot(IVector<?> x, IVector<?> y, boolean confidenceBand) {
        return of().regplot(x, y, confidenceBand);
    }

    /**
     * 变量两两散点矩阵（对角线样式见 {@link PairplotDiagonal}），见
     * {@link IPlot#pairplot(IMatrix, List, PairplotDiagonal)}。
     */
    public static IPlot pairplot(IMatrix<?> data) {
        return of().pairplot(data);
    }

    /**
     * 指定列名与对角线类型的 {@link #pairplot(IMatrix)}。
     */
    public static IPlot pairplot(IMatrix<?> data, List<String> columnNames, PairplotDiagonal diagonal) {
        return of().pairplot(data, columnNames, diagonal);
    }

    /**
     * 主散点图 + 边缘分布（KDE 或直方图），见 {@link IPlot#jointplot}。
     */
    public static IPlot jointplot(IVector<?> x, IVector<?> y, JointplotMarginal marginal) {
        return of().jointplot(x, y, marginal);
    }

    /**
     * 列出当前 {@link PlotProvider} 下二维绘图后端可用的主题名（内置 + 运行时注册）。
     * <p>
     * JavaFX：见
     * {@link com.yishape.lab.math.plot.javafx.JavaFxThemeManager#getRegisteredThemeNames()}；
     * ECharts：见
     * {@link EchartsThemeManager#getRegisteredThemeNames()}。名称按字典序排序、大小写不敏感去重。</p>
     * <p>
     * 三维 {@link #of3d()} 固定为 JavaFX，若仅需 3D 主题列表可使用
     * {@link com.yishape.lab.math.plot.javafx.JavaFxThemeManager#getRegisteredThemeNames()}
     * 或通过 {@link #setProvider} 切换到 JavaFx 后再调用本方法。</p>
     *
     * @return 可用主题名称的不可修改列表（已排序）
     */
    public static List<String> listThemes() {
        List<String> raw = switch (provider) {
            case PlotProvider.JavaFx ->
                JavaFxThemeManager.getRegisteredThemeNames();
            case PlotProvider.Echarts ->
                EchartsThemeManager.getRegisteredThemeNames();
            default ->
                JavaFxThemeManager.getRegisteredThemeNames();
        };
        TreeSet<String> sorted = new TreeSet<>(String.CASE_INSENSITIVE_ORDER);
        sorted.addAll(raw);
        return List.copyOf(new ArrayList<>(sorted));
    }
}
