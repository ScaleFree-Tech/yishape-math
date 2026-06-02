package com.yishape.lab.math.plot;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.plot.echarts.EchartsThemeManager;
import java.io.Serializable;
import java.util.List;
import java.util.Map;

/**
 * 3D plotting interface: chart types and fluent configuration (parallel to {@link IPlot}).
 * 三维绘图接口：图表类型与流式配置（与 {@link IPlot} 风格一致）。
 *
 * <p>Grid conventions / 网格约定：对 {@link #surface3d(IVector, IVector, IMatrix)} 等基于 meshgrid 的方法，
 * {@code z} 为二维采样矩阵，行索引与 {@code x} 对齐、列索引与 {@code y} 对齐（实现类须在同一约定下解释 {@link IMatrix} 下标）。
 *
 * <p>Geospatial point clouds / 地理点集：可将经度、纬度、高程（或抬高后的专题值）作为
 * {@link #scatter3d(IVector, IVector, IVector)} 的 {@code x,y,z}；栅格高程与拉伸专题图可用 {@link #terrain3d}、{@link #heatmap3d}。
 *
 * @author lteb2
 */
public interface I3dPlot extends Serializable {

    /**
     * 柱体在三维条形图中的截面形状（长方体、圆柱、圆锥等视觉变体）。
     */
    enum BarExtrusion3D {
        /** 长方体柱（默认）。 */
        BOX,
        /** 圆柱体。 */
        CYLINDER,
        /** 圆锥体（通常为戏剧化展示）。 */
        CONE
    }

    // ========== 1. Relations & distribution 三维关系与分布 ==========

    /**
     * 3D scatter: one point per ({@code x[i], y[i], z[i]}).
     * 三维散点图：每个样本对应空间中一点。
     */
    I3dPlot scatter3d(IVector x, IVector y, IVector z);

    /**
     * 3D scatter with categorical hue (e.g. cluster or class label per point).
     * 带分组的多系列三维散点。
     */
    I3dPlot scatter3d(IVector x, IVector y, IVector z, List<String> hue);

    /**
     * 3D bubble chart: {@code sizes} encodes a fourth numeric dimension (marker size); optional {@code hue} for category/color.
     * 三维气泡图：{@code sizes} 表示第四维（大小），可选 {@code hue} 作类别/颜色。
     */
    I3dPlot scatterBubble3d(IVector x, IVector y, IVector z, IVector sizes);

    /**
     * 3D bubble chart with hue grouping.
     * 带分组的三维气泡图。
     */
    I3dPlot scatterBubble3d(IVector x, IVector y, IVector z, IVector sizes, List<String> hue);

    /**
     * 3D line / path: consecutive points are connected in index order (trajectory, time series in 3D).
     * 三维折线/路径：按索引顺序连接，用于轨迹或三维序列。
     */
    I3dPlot line3d(IVector x, IVector y, IVector z);

    /**
     * 3D density / point-cloud density: color, opacity, or isosurfaces per implementation (hotspots / outliers context).
     * 三维密度图：用颜色、透明度或等值面表示点密度（热点与离群脉络）。
     */
    I3dPlot density3d(IVector x, IVector y, IVector z);

    /**
     * 3D density with resolution hint (voxel/grid resolution or KDE grids; {@code resolution &lt;= 0} 表示实现类默认).
     * 指定分辨率的三维密度（体素/网格或 KDE 网格；非正数则由实现决定默认）。
     */
    I3dPlot density3d(IVector x, IVector y, IVector z, int resolution);

    // ========== 2. Distribution & statistics 数据分布与统计 ==========

    /**
     * 3D bar chart: categories on one axis, values as bar heights (extrusion shape via {@link BarExtrusion3D} overload).
     * 三维柱状图：类别与柱高；柱体形状见 {@link BarExtrusion3D} 重载。
     */
    I3dPlot bar3d(List<String> categories, IVector values);

    /**
     * 3D bar with extrusion style (box / cylinder / cone).
     * 指定柱体截面形状的三维柱图。
     */
    I3dPlot bar3d(List<String> categories, IVector values, BarExtrusion3D extrusion);

    /**
     * Grouped 3D bars (same idea as {@link IPlot#bar(List, IVector, List)} in a 3D scene).
     * 分组三维柱图。
     */
    I3dPlot bar3d(List<String> xticks, IVector y, List<String> hue);

    /**
     * Grouped 3D bars with extrusion style.
     * 带柱体形状的分组三维柱图。
     */
    I3dPlot bar3d(List<String> xticks, IVector y, List<String> hue, BarExtrusion3D extrusion);

    /**
     * 3D pie: part-to-whole proportions with extruded / perspective pie (implementation-specific styling).
     * 三维饼图：比例组成的三维透视/拉伸效果。
     */
    I3dPlot pie3d(IVector data);

    /**
     * 3D pie with slice labels.
     * 带扇区标签的三维饼图。
     */
    I3dPlot pie3d(IVector data, List<String> labels);

    /**
     * 3D / joint histogram: 2D bins over ({@code x}, {@code y}), bar height = count per cell (bivariate density).
     * 三维直方图（联合直方图）：{@code (x,y)} 平面分格，柱体高度为频数。
     */
    I3dPlot hist3d(IVector x, IVector y);

    /**
     * Joint histogram with explicit bin counts along x and y.
     * 指定 x、y 方向 Bin 数的联合直方图。
     */
    I3dPlot hist3d(IVector x, IVector y, int xBins, int yBins);

    /**
     * 3D box plot: single-sample groups with {@link #boxplot3d(IMatrix, List)} for multi-column layouts.
     * 三维箱线图：单组向量；多组对比宜用矩阵重载。
     */
    I3dPlot boxplot3d(IVector data, List<String> labels);

    /**
     * 3D box plot: each column (or row, per implementation doc) is one distribution / group for side-by-side comparison.
     * 多组三维箱线对比（每列或每行一组分布，以实现类文档为准）。
     */
    I3dPlot boxplot3d(IMatrix data, List<String> labels);

    // ========== 3. Surface, model & fields 曲面、模型与场 ==========

    /**
     * 3D surface: height from {@code z} over the mesh ({@code x}, {@code y}).
     * 三维曲面图：高度由网格矩阵 {@code z} 给定。
     */
    I3dPlot surface3d(IVector x, IVector y, IMatrix z);

    /**
     * Same as {@link #surface3d(IVector, IVector, IMatrix)} with optional contour projection on the base plane (meshc-style).
     * 曲面图并可是否在底面绘制等高线投影（类似 meshc 的底面等高线）。
     */
    I3dPlot surface3d(IVector x, IVector y, IMatrix z, boolean bottomContourProjection);

    /**
     * 3D contour lines on the surface or in volume (contour3-style); levels may follow {@link PlotStyle} or defaults.
     * 三维等高线/等值线。
     */
    I3dPlot contour3d(IVector x, IVector y, IMatrix z);

    /**
     * 3D wireframe only (no face fill).
     * 三维线框图。
     */
    I3dPlot wireframe3d(IVector x, IVector y, IMatrix z);

    /**
     * 3D heatmap: intensity on ({@code x},{@code y}) with height and/or color (same grid as 2D heatmap, extruded in Z).
     * 三维热力图：平面上强度映射为高度与/或颜色。
     */
    I3dPlot heatmap3d(IMatrix z, List<String> xLabels, List<String> yLabels);

    /**
     * 3D waterfall: layers stacked or offset along an axis (e.g. spectra, vibrations); {@code layerHeights} rows = series, cols = {@code x}.
     * 三维瀑布图：多层沿轴向排列；矩阵每行一条序列、列与 {@code x} 对齐。
     */
    I3dPlot waterfall3d(IVector x, IMatrix layerHeights);

    /**
     * Quiver / 3D vector field: arrows at ({@code x,y,z}) with directions ({@code u,v,w}).
     * 三维向量场：在采样点处绘制方向与长度。
     */
    I3dPlot vectorField3d(IVector x, IVector y, IVector z, IVector u, IVector v, IVector w);

    /**
     * 3D streamlines / flow lines traced through a regular grid field (same-length vectors as {@link #vectorField3d}).
     * 三维流线图：在给定矢量场上积分得到的流线（与向量场采样一致）。
     */
    I3dPlot streamlines3d(IVector x, IVector y, IVector z, IVector u, IVector v, IVector w);

    // ========== 4. Geo & structure 地理空间与结构 ==========

    /**
     * Terrain / height field (semantic alias for surface over geographic or DEM grids; axis labels are geographic if set via {@link #xlabel(String)}).
     * 三维地形/高程栅格（语义上与曲面一致；轴名可通过流式 API 写为经纬度等）。
     */
    I3dPlot terrain3d(IVector x, IVector y, IMatrix elevation);

    /**
     * Flight-path alias for {@link #line3d(IVector, IVector, IVector)}.
     * 三维航线语义别名，等价于 {@link #line3d(IVector, IVector, IVector)}。
     *
     * @param x path X
     * @param y path Y
     * @param z path Z
     * @return this instance for chaining
     */
    default I3dPlot route3d(IVector x, IVector y, IVector z) {
        return line3d(x, y, z);
    }

    /**
     * Same as {@link #route3d(IVector, IVector, IVector)} with style string.
     *
     * @param x path X
     * @param y path Y
     * @param z path Z
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    default I3dPlot route3d(IVector x, IVector y, IVector z, String styleString) {
        return line3d(x, y, z, styleString);
    }

    /**
     * Same as {@link #route3d(IVector, IVector, IVector)} with {@link PlotStyle}.
     *
     * @param x path X
     * @param y path Y
     * @param z path Z
     * @param style structured style
     * @return this instance for chaining
     */
    default I3dPlot route3d(IVector x, IVector y, IVector z, PlotStyle style) {
        return line3d(x, y, z, style);
    }

    /**
     * 3D force-directed or fixed-layout graph (nodes + links), analogous to {@link IPlot#graph(List, List)}.
     * 三维网络/关系图。
     */
    I3dPlot graph3d(List<Map<String, Object>> nodes, List<Map<String, Object>> links);

    /**
     * Filled 3D region (e.g. volume under a surface, curtain between two polylines, or cumulative extrusion along an axis; interpretation is renderer-defined).
     * 三维区域填充：如曲面下方体积、两曲线间的帷幕或沿轴向的累积立体区域（具体由实现解释为网格或参数曲面）。
     */
    I3dPlot areaFill3d(IVector x, IVector y, IVector z);

    /**
     * 3D radar / spider: multiple samples (rows of {@code data}) vs {@code indicators} axes in 3D layout.
     * 三维雷达/蛛网图：{@code data} 每行一样本，列对应 {@code indicators}。
     */
    I3dPlot radar3d(IMatrix data, List<String> indicators);

    /**
     * 3D radar with per-series names (for legend).
     * 三维雷达图并指定系列名。
     */
    I3dPlot radar3d(IMatrix data, List<String> indicators, List<String> seriesNames);

    // ========== PlotStyle / style-string overloads（与 IPlot 一致，便于双后端统一） ==========

    /**
     * 3D scatter with style string; same as {@link #scatter3d(IVector, IVector, IVector)} plus style.
     *
     * @param x X coordinates
     * @param y Y coordinates
     * @param z Z coordinates
     * @param styleString backend-specific tokens
     * @return this instance for chaining
     */
    I3dPlot scatter3d(IVector x, IVector y, IVector z, String styleString);

    /**
     * 3D scatter with {@link PlotStyle}; same as {@link #scatter3d(IVector, IVector, IVector)} plus style.
     *
     * @param x X coordinates
     * @param y Y coordinates
     * @param z Z coordinates
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot scatter3d(IVector x, IVector y, IVector z, PlotStyle style);

    /**
     * 3D scatter with hue and style string; same as {@link #scatter3d(IVector, IVector, IVector, List)} plus style.
     *
     * @param x X coordinates
     * @param y Y coordinates
     * @param z Z coordinates
     * @param hue group label per point
     * @param styleString backend-specific tokens
     * @return this instance for chaining
     */
    I3dPlot scatter3d(IVector x, IVector y, IVector z, List<String> hue, String styleString);

    /**
     * 3D scatter with hue and {@link PlotStyle}.
     *
     * @param x X coordinates
     * @param y Y coordinates
     * @param z Z coordinates
     * @param hue group labels
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot scatter3d(IVector x, IVector y, IVector z, List<String> hue, PlotStyle style);

    /**
     * 3D bubble scatter with style string; same as {@link #scatterBubble3d(IVector, IVector, IVector, IVector)} plus style.
     *
     * @param x X coordinates
     * @param y Y coordinates
     * @param z Z coordinates
     * @param sizes marker size channel
     * @param styleString backend-specific tokens
     * @return this instance for chaining
     */
    I3dPlot scatterBubble3d(IVector x, IVector y, IVector z, IVector sizes, String styleString);

    /**
     * 3D bubble scatter with {@link PlotStyle}.
     *
     * @param x X coordinates
     * @param y Y coordinates
     * @param z Z coordinates
     * @param sizes marker size channel
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot scatterBubble3d(IVector x, IVector y, IVector z, IVector sizes, PlotStyle style);

    /**
     * 3D bubble scatter with hue and style string.
     *
     * @param x X coordinates
     * @param y Y coordinates
     * @param z Z coordinates
     * @param sizes marker size channel
     * @param hue group labels
     * @param styleString backend-specific tokens
     * @return this instance for chaining
     */
    I3dPlot scatterBubble3d(IVector x, IVector y, IVector z, IVector sizes, List<String> hue, String styleString);

    /**
     * 3D bubble scatter with hue and {@link PlotStyle}.
     *
     * @param x X coordinates
     * @param y Y coordinates
     * @param z Z coordinates
     * @param sizes marker size channel
     * @param hue group labels
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot scatterBubble3d(IVector x, IVector y, IVector z, IVector sizes, List<String> hue, PlotStyle style);

    /**
     * Same as {@link #line3d(IVector, IVector, IVector)} with a style string.
     * 与无样式 {@link #line3d(IVector, IVector, IVector)} 相同，附带样式字符串。
     *
     * @param x X coordinates
     * @param y Y coordinates
     * @param z Z coordinates
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    I3dPlot line3d(IVector x, IVector y, IVector z, String styleString);

    /**
     * Same as {@link #line3d(IVector, IVector, IVector)} with {@link PlotStyle}.
     *
     * @param x X coordinates
     * @param y Y coordinates
     * @param z Z coordinates
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot line3d(IVector x, IVector y, IVector z, PlotStyle style);

    /**
     * Same as {@link #density3d(IVector, IVector, IVector)} with a style string.
     *
     * @param x sample X
     * @param y sample Y
     * @param z sample Z
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    I3dPlot density3d(IVector x, IVector y, IVector z, String styleString);

    /**
     * Same as {@link #density3d(IVector, IVector, IVector)} with {@link PlotStyle}.
     *
     * @param x sample X
     * @param y sample Y
     * @param z sample Z
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot density3d(IVector x, IVector y, IVector z, PlotStyle style);

    /**
     * Same as {@link #density3d(IVector, IVector, IVector, int)} with a style string.
     *
     * @param x sample X
     * @param y sample Y
     * @param z sample Z
     * @param resolution grid / voxel resolution hint
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    I3dPlot density3d(IVector x, IVector y, IVector z, int resolution, String styleString);

    /**
     * Same as {@link #density3d(IVector, IVector, IVector, int)} with {@link PlotStyle}.
     *
     * @param x sample X
     * @param y sample Y
     * @param z sample Z
     * @param resolution grid / voxel resolution hint
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot density3d(IVector x, IVector y, IVector z, int resolution, PlotStyle style);

    /**
     * Same as {@link #bar3d(List, IVector)} with a style string.
     *
     * @param categories category axis labels
     * @param values bar heights
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    I3dPlot bar3d(List<String> categories, IVector values, String styleString);

    /**
     * Same as {@link #bar3d(List, IVector)} with {@link PlotStyle}.
     *
     * @param categories category axis labels
     * @param values bar heights
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot bar3d(List<String> categories, IVector values, PlotStyle style);

    /**
     * Same as {@link #bar3d(List, IVector, BarExtrusion3D)} with a style string.
     *
     * @param categories category axis labels
     * @param values bar heights
     * @param extrusion bar cross-section shape
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    I3dPlot bar3d(List<String> categories, IVector values, BarExtrusion3D extrusion, String styleString);

    /**
     * Same as {@link #bar3d(List, IVector, BarExtrusion3D)} with {@link PlotStyle}.
     *
     * @param categories category axis labels
     * @param values bar heights
     * @param extrusion bar cross-section shape
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot bar3d(List<String> categories, IVector values, BarExtrusion3D extrusion, PlotStyle style);

    /**
     * Same as {@link #bar3d(List, IVector, List)} with a style string.
     *
     * @param xticks category labels
     * @param y bar heights
     * @param hue group labels
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    I3dPlot bar3d(List<String> xticks, IVector y, List<String> hue, String styleString);

    /**
     * Same as {@link #bar3d(List, IVector, List)} with {@link PlotStyle}.
     *
     * @param xticks category labels
     * @param y bar heights
     * @param hue group labels
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot bar3d(List<String> xticks, IVector y, List<String> hue, PlotStyle style);

    /**
     * Same as {@link #bar3d(List, IVector, List, BarExtrusion3D)} with a style string.
     *
     * @param xticks category labels
     * @param y bar heights
     * @param hue group labels
     * @param extrusion bar cross-section shape
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    I3dPlot bar3d(List<String> xticks, IVector y, List<String> hue, BarExtrusion3D extrusion, String styleString);

    /**
     * Same as {@link #bar3d(List, IVector, List, BarExtrusion3D)} with {@link PlotStyle}.
     *
     * @param xticks category labels
     * @param y bar heights
     * @param hue group labels
     * @param extrusion bar cross-section shape
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot bar3d(List<String> xticks, IVector y, List<String> hue, BarExtrusion3D extrusion, PlotStyle style);

    /**
     * Same as {@link #pie3d(IVector)} with a style string.
     *
     * @param data slice values
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    I3dPlot pie3d(IVector data, String styleString);

    /**
     * Same as {@link #pie3d(IVector)} with {@link PlotStyle}.
     *
     * @param data slice values
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot pie3d(IVector data, PlotStyle style);

    /**
     * Same as {@link #pie3d(IVector, List)} with a style string.
     *
     * @param data slice values
     * @param labels slice labels
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    I3dPlot pie3d(IVector data, List<String> labels, String styleString);

    /**
     * Same as {@link #pie3d(IVector, List)} with {@link PlotStyle}.
     *
     * @param data slice values
     * @param labels slice labels
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot pie3d(IVector data, List<String> labels, PlotStyle style);

    /**
     * Same as {@link #hist3d(IVector, IVector)} with a style string.
     *
     * @param x first marginal sample
     * @param y second marginal sample
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    I3dPlot hist3d(IVector x, IVector y, String styleString);

    /**
     * Same as {@link #hist3d(IVector, IVector)} with {@link PlotStyle}.
     *
     * @param x first marginal sample
     * @param y second marginal sample
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot hist3d(IVector x, IVector y, PlotStyle style);

    /**
     * Same as {@link #hist3d(IVector, IVector, int, int)} with a style string.
     *
     * @param x first marginal sample
     * @param y second marginal sample
     * @param xBins bins along X
     * @param yBins bins along Y
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    I3dPlot hist3d(IVector x, IVector y, int xBins, int yBins, String styleString);

    /**
     * Same as {@link #hist3d(IVector, IVector, int, int)} with {@link PlotStyle}.
     *
     * @param x first marginal sample
     * @param y second marginal sample
     * @param xBins bins along X
     * @param yBins bins along Y
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot hist3d(IVector x, IVector y, int xBins, int yBins, PlotStyle style);

    /**
     * Same as {@link #boxplot3d(IVector, List)} with a style string.
     *
     * @param data sample vector
     * @param labels group labels
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    I3dPlot boxplot3d(IVector data, List<String> labels, String styleString);

    /**
     * Same as {@link #boxplot3d(IVector, List)} with {@link PlotStyle}.
     *
     * @param data sample vector
     * @param labels group labels
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot boxplot3d(IVector data, List<String> labels, PlotStyle style);

    /**
     * Same as {@link #boxplot3d(IMatrix, List)} with a style string.
     *
     * @param data columns (or rows) as groups per implementation docs
     * @param labels group labels
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    I3dPlot boxplot3d(IMatrix data, List<String> labels, String styleString);

    /**
     * Same as {@link #boxplot3d(IMatrix, List)} with {@link PlotStyle}.
     *
     * @param data columns (or rows) as groups per implementation docs
     * @param labels group labels
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot boxplot3d(IMatrix data, List<String> labels, PlotStyle style);

    /**
     * Same as {@link #surface3d(IVector, IVector, IMatrix)} with a style string.
     *
     * @param x surface grid X
     * @param y surface grid Y
     * @param z height field
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    I3dPlot surface3d(IVector x, IVector y, IMatrix z, String styleString);

    /**
     * Same as {@link #surface3d(IVector, IVector, IMatrix)} with {@link PlotStyle}.
     *
     * @param x surface grid X
     * @param y surface grid Y
     * @param z height field
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot surface3d(IVector x, IVector y, IMatrix z, PlotStyle style);

    /**
     * Same as {@link #surface3d(IVector, IVector, IMatrix, boolean)} with a style string.
     *
     * @param x surface grid X
     * @param y surface grid Y
     * @param z height field
     * @param bottomContourProjection draw base contour overlay when {@code true}
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    I3dPlot surface3d(IVector x, IVector y, IMatrix z, boolean bottomContourProjection, String styleString);

    /**
     * Same as {@link #surface3d(IVector, IVector, IMatrix, boolean)} with {@link PlotStyle}.
     *
     * @param x surface grid X
     * @param y surface grid Y
     * @param z height field
     * @param bottomContourProjection draw base contour overlay when {@code true}
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot surface3d(IVector x, IVector y, IMatrix z, boolean bottomContourProjection, PlotStyle style);

    /**
     * Same as {@link #contour3d(IVector, IVector, IMatrix)} with a style string.
     *
     * @param x grid X
     * @param y grid Y
     * @param z scalar field
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    I3dPlot contour3d(IVector x, IVector y, IMatrix z, String styleString);

    /**
     * Same as {@link #contour3d(IVector, IVector, IMatrix)} with {@link PlotStyle}.
     *
     * @param x grid X
     * @param y grid Y
     * @param z scalar field
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot contour3d(IVector x, IVector y, IMatrix z, PlotStyle style);

    /**
     * Same as {@link #wireframe3d(IVector, IVector, IMatrix)} with a style string.
     *
     * @param x grid X
     * @param y grid Y
     * @param z height field
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    I3dPlot wireframe3d(IVector x, IVector y, IMatrix z, String styleString);

    /**
     * Same as {@link #wireframe3d(IVector, IVector, IMatrix)} with {@link PlotStyle}.
     *
     * @param x grid X
     * @param y grid Y
     * @param z height field
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot wireframe3d(IVector x, IVector y, IMatrix z, PlotStyle style);

    /**
     * Same as {@link #heatmap3d(IMatrix, List, List)} with a style string.
     *
     * @param z intensity matrix
     * @param xLabels column labels
     * @param yLabels row labels
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    I3dPlot heatmap3d(IMatrix z, List<String> xLabels, List<String> yLabels, String styleString);

    /**
     * Same as {@link #heatmap3d(IMatrix, List, List)} with {@link PlotStyle}.
     *
     * @param z intensity matrix
     * @param xLabels column labels
     * @param yLabels row labels
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot heatmap3d(IMatrix z, List<String> xLabels, List<String> yLabels, PlotStyle style);

    /**
     * Same as {@link #waterfall3d(IVector, IMatrix)} with a style string.
     *
     * @param x shared axis
     * @param layerHeights one row per series
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    I3dPlot waterfall3d(IVector x, IMatrix layerHeights, String styleString);

    /**
     * Same as {@link #waterfall3d(IVector, IMatrix)} with {@link PlotStyle}.
     *
     * @param x shared axis
     * @param layerHeights one row per series
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot waterfall3d(IVector x, IMatrix layerHeights, PlotStyle style);

    /**
     * Same as {@link #vectorField3d(IVector, IVector, IVector, IVector, IVector, IVector)} with a style string.
     *
     * @param x sample X
     * @param y sample Y
     * @param z sample Z
     * @param u vector X component
     * @param v vector Y component
     * @param w vector Z component
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    I3dPlot vectorField3d(IVector x, IVector y, IVector z, IVector u, IVector v, IVector w, String styleString);

    /**
     * Same as {@link #vectorField3d(IVector, IVector, IVector, IVector, IVector, IVector)} with {@link PlotStyle}.
     *
     * @param x sample X
     * @param y sample Y
     * @param z sample Z
     * @param u vector X component
     * @param v vector Y component
     * @param w vector Z component
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot vectorField3d(IVector x, IVector y, IVector z, IVector u, IVector v, IVector w, PlotStyle style);

    /**
     * Same as {@link #streamlines3d(IVector, IVector, IVector, IVector, IVector, IVector)} with a style string.
     *
     * @param x sample X
     * @param y sample Y
     * @param z sample Z
     * @param u vector X component
     * @param v vector Y component
     * @param w vector Z component
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    I3dPlot streamlines3d(IVector x, IVector y, IVector z, IVector u, IVector v, IVector w, String styleString);

    /**
     * Same as {@link #streamlines3d(IVector, IVector, IVector, IVector, IVector, IVector)} with {@link PlotStyle}.
     *
     * @param x sample X
     * @param y sample Y
     * @param z sample Z
     * @param u vector X component
     * @param v vector Y component
     * @param w vector Z component
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot streamlines3d(IVector x, IVector y, IVector z, IVector u, IVector v, IVector w, PlotStyle style);

    /**
     * Same as {@link #terrain3d(IVector, IVector, IMatrix)} with a style string.
     *
     * @param x terrain grid X
     * @param y terrain grid Y
     * @param elevation height field / DEM samples
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    I3dPlot terrain3d(IVector x, IVector y, IMatrix elevation, String styleString);

    /**
     * Same as {@link #terrain3d(IVector, IVector, IMatrix)} with {@link PlotStyle}.
     *
     * @param x terrain grid X
     * @param y terrain grid Y
     * @param elevation height field / DEM samples
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot terrain3d(IVector x, IVector y, IMatrix elevation, PlotStyle style);

    /**
     * Same as {@link #graph3d(List, List)} with a style string.
     *
     * @param nodes node maps
     * @param links link maps
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    I3dPlot graph3d(List<Map<String, Object>> nodes, List<Map<String, Object>> links, String styleString);

    /**
     * Same as {@link #graph3d(List, List)} with {@link PlotStyle}.
     *
     * @param nodes node maps
     * @param links link maps
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot graph3d(List<Map<String, Object>> nodes, List<Map<String, Object>> links, PlotStyle style);

    /**
     * Same as {@link #areaFill3d(IVector, IVector, IVector)} with a style string.
     *
     * @param x boundary / surface X
     * @param y boundary / surface Y
     * @param z boundary / surface Z
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    I3dPlot areaFill3d(IVector x, IVector y, IVector z, String styleString);

    /**
     * Same as {@link #areaFill3d(IVector, IVector, IVector)} with {@link PlotStyle}.
     *
     * @param x boundary / surface X
     * @param y boundary / surface Y
     * @param z boundary / surface Z
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot areaFill3d(IVector x, IVector y, IVector z, PlotStyle style);

    /**
     * Same as {@link #radar3d(IMatrix, List)} with a style string.
     *
     * @param data one row per series/sample
     * @param indicators axis names
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    I3dPlot radar3d(IMatrix data, List<String> indicators, String styleString);

    /**
     * Same as {@link #radar3d(IMatrix, List)} with {@link PlotStyle}.
     *
     * @param data one row per series/sample
     * @param indicators axis names
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot radar3d(IMatrix data, List<String> indicators, PlotStyle style);

    /**
     * Same as {@link #radar3d(IMatrix, List, List)} with a style string.
     *
     * @param data one row per series
     * @param indicators axis names
     * @param seriesNames legend entries
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    I3dPlot radar3d(IMatrix data, List<String> indicators, List<String> seriesNames, String styleString);

    /**
     * Same as {@link #radar3d(IMatrix, List, List)} with {@link PlotStyle}.
     *
     * @param data one row per series
     * @param indicators axis names
     * @param seriesNames legend entries
     * @param style structured style
     * @return this instance for chaining
     */
    I3dPlot radar3d(IMatrix data, List<String> indicators, List<String> seriesNames, PlotStyle style);

    // ========== Style / theme hooks（对齐 IPlot 可扩展性） ==========

    /**
     * Default plot style when the style system resolves implicit series styling.
     * 默认绘图样式基底（与 {@link IPlot#setDefaultStyle(PlotStyle)} 对齐）。
     *
     * @param style structured defaults
     * @return this instance for chaining
     */
    I3dPlot setDefaultStyle(PlotStyle style);

    /**
     * Active palette key; must exist on {@link ColorPalette}.
     * 调色板名称（须在 {@link ColorPalette} 中注册）。
     *
     * @param paletteName registered palette identifier
     * @return this instance for chaining
     */
    I3dPlot setPalette(String paletteName);

    /**
     * Toggle structured style parsing for subsequent geometry layers.
     * 启用/关闭结构化样式系统。
     *
     * @param enabled {@code true} to honour {@link PlotStyle} pipelines
     * @return this instance for chaining
     */
    I3dPlot enableStyleSystem(boolean enabled);

    /**
     * Toggle bundled/custom theme overlays.
     * 启用/关闭主题系统。
     *
     * @param enabled {@code true} before calling {@link #applyTheme(String)}
     * @return this instance for chaining
     */
    I3dPlot enableThemeSystem(boolean enabled);

    /**
     * Apply a previously registered theme name (or built-ins).
     * 应用已注册主题名。
     *
     * @param themeName palette + chrome bundle key
     * @return this instance for chaining
     */
    I3dPlot applyTheme(String themeName);

    /**
     * Register extra ECharts theme assets for HTML export backends.
     * 注册自定义 ECharts 主题。
     *
     * @param themeName lookup key used by {@link #applyTheme(String)}
     * @param theme serializer-friendly theme blob
     * @return this instance for chaining
     */
    I3dPlot registerTheme(String themeName, EchartsThemeManager.CustomTheme theme);

    /**
     * Create a linear-gradient palette theme and register it for reuse.
     * 创建渐变主题并注册。
     *
     * @param themeName lookup key
     * @param startColor gradient start colour
     * @param endColor gradient end colour
     * @param backgroundColor viewport background colour
     * @return this instance for chaining
     */
    I3dPlot createGradientTheme(String themeName, String startColor, String endColor, String backgroundColor);

    /**
     * Derive theme colours from one accent colour plus background.
     * 由单色推导主题并注册。
     *
     * @param themeName lookup key
     * @param baseColor accent chroma anchor
     * @param backgroundColor viewport background colour
     * @return this instance for chaining
     */
    I3dPlot createMonochromeTheme(String themeName, String baseColor, String backgroundColor);

    // ========== Fluent API 流式 API ==========

    /**
     * Sets the chart title (fluent setter).
     * 设置主标题。
     *
     * @param titleText title string
     * @return this instance for chaining
     */
    I3dPlot title(String titleText);

    /**
     * Sets title plus subtitle captions.
     * 设置主副标题。
     *
     * @param titleText main headline
     * @param subtitleText secondary headline
     * @return this instance for chaining
     */
    I3dPlot title(String titleText, String subtitleText);

    /**
     * Sets the textual label rendered along the projected X dimension.
     * 设置 X 轴标题。
     *
     * @param name axis label
     * @return this instance for chaining
     */
    I3dPlot xlabel(String name);

    /**
     * Sets the textual label rendered along the projected Y dimension.
     * 设置 Y 轴标题。
     *
     * @param name axis label
     * @return this instance for chaining
     */
    I3dPlot ylabel(String name);

    /**
     * Sets the textual label rendered along the depth / vertical emphasis axis.
     * Z 轴标题（三维专用）。
     *
     * @param name axis label
     * @return this instance for chaining
     */
    I3dPlot zlabel(String name);

    /**
     * Overrides the drawable width/height in pixels for raster / HTML exporters.
     * 设置画布像素宽高。
     *
     * @param width pixel width (&gt; 0)
     * @param height pixel height (&gt; 0)
     * @return this instance for chaining
     */
    I3dPlot size(int width, int height);

    /**
     * Shortcut for applying bundled named themes ({@code "dark"}, Seaborn clones, etc.).
     * 按名称应用内置/注册主题简称。
     *
     * @param theme theme mnemonic accepted by concrete implementation
     * @return this instance for chaining
     */
    I3dPlot theme(String theme);

    /**
     * Shows the interactive JavaFX stage or warms the off-screen buffer (implementation defined).
     * 显示或激活交互窗口（依实现而定）。
     *
     * @return this instance for chaining
     */
    I3dPlot show();

    /**
     * Serialises the current scene to HTML (ECharts) or a scene description (JavaFX snapshot host).
     * 导出 HTML。
     *
     * @param filename destination path
     * @return this instance for chaining
     */
    I3dPlot saveAsHtml(String filename);

    /**
     * Serialised HTML snippet or document body ready for embedding.
     * HTML 源码。
     *
     * @return HTML string
     */
    String toHtml();

    /**
     * Serialisable scene/config JSON consumed by scripting bridges.
     * JSON 配置。
     *
     * @return JSON payload
     */
    String toJson();

    // ========== Configuration 配置方法 ==========

    /**
     * Imperative setter for title (non-fluent).
     * 命令式设置主标题。
     *
     * @param titleText headline
     */
    void setTitle(String titleText);

    /**
     * Imperative setter for title + subtitle blocks.
     * 命令式设置主副标题。
     *
     * @param titleText main headline
     * @param subtitleText secondary headline
     */
    void setTitle(String titleText, String subtitleText);

    /**
     * Imperative setter for projected X-axis label text.
     * 命令式设置 X 轴标题。
     *
     * @param name textual label
     */
    void setXlabel(String name);

    /**
     * Imperative setter for projected Y-axis label text.
     * 命令式设置 Y 轴标题。
     *
     * @param name textual label
     */
    void setYlabel(String name);

    /**
     * Imperative setter for projected Z-axis label text.
     * 命令式设置 Z 轴标题。
     *
     * @param name textual label
     */
    void setZlabel(String name);

    /**
     * Replace auto ticks on the projected X axis.
     * 设置 X 轴刻度配置。
     *
     * @param xticks tick specification
     */
    void setXticks(AxisTicks xticks);

    /**
     * Replace auto ticks on the projected Y axis.
     * 设置 Y 轴刻度配置。
     *
     * @param yticks tick specification
     */
    void setYticks(AxisTicks yticks);

    /**
     * Replace auto ticks on the projected Z axis.
     * 设置 Z 轴刻度配置。
     *
     * @param zticks tick specification
     */
    void setZticks(AxisTicks zticks);

    /**
     * Logical canvas width configured on the backing renderer.
     * 画布宽度（像素）。
     *
     * @return width in pixels
     */
    int getWidth();

    /**
     * Logical canvas height configured on the backing renderer.
     * 画布高度（像素）。
     *
     * @return height in pixels
     */
    int getHeight();

    /**
     * Active theme mnemonic after {@link #theme(String)} / {@link #applyTheme(String)}.
     * 当前主题名。
     *
     * @return theme identifier
     */
    String getTheme();
}