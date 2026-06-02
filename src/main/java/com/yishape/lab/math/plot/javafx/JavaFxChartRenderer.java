package com.yishape.lab.math.plot.javafx;

import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;

import java.util.List;
import java.util.Map;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.plot.AxisTicks;
import com.yishape.lab.math.plot.PlotAxisScale;
import com.yishape.lab.math.plot.PlotStyle;

/**
 * JavaFX图表渲染器接口
 * 定义所有图表渲染器必须实现的方法
 * 
 * @author lteb2
 */
public interface JavaFxChartRenderer {
    
    /**
     * 渲染图表
     * @param gc GraphicsContext
     * @param data 系列数据列表
     * @param config 图表配置
     */
    void render(GraphicsContext gc, List<SeriesData> data, ChartConfig config);
    
    /**
     * 获取渲染器支持的图表类型
     * @return 图表类型名称
     */
    String getChartType();
    
    /**
     * 检查是否支持动画
     * @return 是否支持动画
     */
    boolean supportsAnimation();
    
    /**
     * 获取动画持续时间(毫秒)
     * @return 动画持续时间
     */
    int getAnimationDuration();
    
    /**
     * 系列数据内部类
     */
    class SeriesData {
        public String name;
        public IVector<Double> x;
        public IVector<Double> y;
        public PlotStyle style;
        public String type;
        /** 0 = primary Y, 1 = secondary (right) axis */
        public int yAxisIndex = 0;
        public List<String> labels;
        public Map<String, Object> extraData;
        /**
         * 为 true 时折线按 y（再按索引）排序连点；用于 joint 右侧边际密度（数据为 x=密度、y=原变量）。
         */
        public boolean sortLineByY;

        public SeriesData(String name, IVector<Double> x, IVector<Double> y, 
                         PlotStyle style, String type) {
            this.name = name;
            this.x = x;
            this.y = y;
            this.style = style;
            this.type = type;
            this.labels = new java.util.ArrayList<>();
            this.extraData = new java.util.HashMap<>();
        }
    }
    
    /**
     * 图表配置类
     */
    class ChartConfig {
        public int width;
        public int height;
        public int paddingLeft = 80;
        public int paddingRight = 50;
        public int paddingTop = 80;
        public int paddingBottom = 80;
        public String title = "";
        public String subtitle = "";
        public String xlabel = "";
        public String ylabel = "";
        public Color backgroundColor = Color.WHITE;
        public String theme = "default";
        public boolean showLegend = true;
        /** Legend position string (see {@link com.yishape.lab.math.plot.LegendPositions}). */
        public String legendPosition = com.yishape.lab.math.plot.LegendPositions.TOP_RIGHT;
        /** 图例相对自动布局位置的像素偏移（可由交互窗口拖动）。 */
        public double legendOffsetX = 0;
        public double legendOffsetY = 0;
        public boolean showGrid = true;
        public boolean enableAnimation = true;
        public int animationDuration = 1000;
        
        // 坐标轴范围
        public double[] xRange;
        public double[] yRange;

        /**
         * jointplot 等与主图对齐：长度=2 时强制笛卡尔轴范围（{@link com.yishape.lab.math.plot.javafx.renderers.CartesianComboRenderer}）。
         */
        public double[] axisLockX;
        public double[] axisLockY;
        /** &gt;0 时限制 X/Y 刻度格数，减轻小窗格标签重叠 */
        public int maxAxisTicks;

        /** 自定义 X 轴刻度；非空时在 {@link com.yishape.lab.math.plot.javafx.JavaFxChartUtils} 中优先使用 */
        public AxisTicks xAxisTicks;
        /**
         * 为 true 时，{@link #xAxisTicks} 的各标签绘在槽位中心 {@code (i+0.5)/n}（如 K 线、分类横轴）。
         */
        public boolean centerCategoryXLabels = false;
        /** 自定义 Y 轴刻度 */
        public AxisTicks yAxisTicks;
        /** 渲染时由 {@link JavaFxPlot} 注入，供渲染器注册悬停/点击命中区域 */
        public JavaFxInteractionHandler hitTestHandler;

        public PlotAxisScale xAxisScale = PlotAxisScale.LINEAR;
        public PlotAxisScale yAxisScale = PlotAxisScale.LINEAR;
        /** Optional label for secondary Y axis (when any series has yAxisIndex == 1) */
        public String y2AxisLabel = "";
        
        public ChartConfig(int width, int height) {
            this.width = width;
            this.height = height;
        }
    }
}
