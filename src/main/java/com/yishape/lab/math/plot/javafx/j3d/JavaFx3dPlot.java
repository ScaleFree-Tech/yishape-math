package com.yishape.lab.math.plot.javafx.j3d;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.plot.AxisTicks;
import com.yishape.lab.math.plot.ColorPalette;
import com.yishape.lab.math.plot.I3dPlot;
import com.yishape.lab.math.plot.MeshGridHelper;
import com.yishape.lab.math.plot.PlotException;
import com.yishape.lab.math.plot.PlotStyle;
import com.yishape.lab.math.plot.echarts.EchartsThemeManager;
import com.yishape.lab.math.plot.javafx.JavaFxStyleApplier;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;
import javafx.scene.Group;
import javafx.scene.PerspectiveCamera;
import javafx.scene.Scene;
import javafx.scene.SubScene;
import javafx.scene.layout.BorderPane;
import javafx.scene.paint.Color;
import javafx.scene.paint.PhongMaterial;
import javafx.scene.shape.Box;
import javafx.scene.shape.Cylinder;
import javafx.scene.shape.MeshView;
import javafx.scene.shape.Sphere;
import javafx.scene.shape.TriangleMesh;
import javafx.scene.transform.Rotate;
import javafx.scene.transform.Scale;
import javafx.scene.transform.Translate;
import javafx.stage.Stage;

import java.io.File;
import java.io.IOException;
import java.io.Serial;
import java.io.Serializable;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.DoubleBinaryOperator;

/**
 * 基于 JavaFX 三维场景（{@link SubScene}、{@link PerspectiveCamera}、{@link MeshView} 等）的 {@link I3dPlot} 实现。
 * <p>用户传入原始 {@link IVector}/{@link IMatrix}；曲面/热力等网格在实现内按 {@link I3dPlot} 约定与 {@link MeshGridHelper} 自动对齐，无需自行 meshgrid。</p>
 * <p>仅使用 JavaFX 官方 API，不引入第三方 3D 引擎。</p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class JavaFx3dPlot implements I3dPlot {

    private static final long serialVersionUID = 1L;

    transient JavaFxThemeManager themeManager;
    private transient Group contentRaw;
    /** 仅图表数据与几何；轴网在 {@link #axesDecoration}。 */
    private transient Group plotContent;
    private transient Group axesDecoration;
    transient Group pivot;
    private transient Scale worldScale;
    private transient Translate worldShift;
    transient Rotate dragRx;
    transient Rotate dragRy;
    transient SubScene subScene;
    transient PerspectiveCamera camera;
    transient BorderPane rootPane;
    transient Scene scene;
    transient Stage stage;
    /** 用于离屏快照的宿主窗口（JavaFX 3D 需挂到已 show 的场景图才可正确栅格化）。 */
    transient Stage snapshotHostStage;
    transient BorderPane snapshotHostRoot;

    int width;
    int height;
    private String themeName = JavaFxThemeManager.THEME_SEABORN;
    String titleText = "";
    String subtitleText = "";
    private String xLabel = "";
    private String yLabel = "";
    private String zLabel = "";
    private AxisTicks xticks;
    private AxisTicks yticks;
    private AxisTicks zticks;

    private PlotStyle defaultSeriesStyle;
    private boolean useStyleSystem = true;
    private boolean useThemeSystem = true;
    private String paletteName;

    private double minX = Double.NaN, maxX = Double.NaN;
    private double minY = Double.NaN, maxY = Double.NaN;
    private double minZ = Double.NaN, maxZ = Double.NaN;

    double anchorX, anchorY;
    double anchorAngleX;
    double anchorAngleY;

    /** 图例项列表（用于显示系列图例） */
    transient List<JavaFx3dLegend.LegendItem> legendItems;

    /** Tooltip标签（由SceneSupport初始化） */
    transient javafx.scene.control.Label tooltipLabel;

    public JavaFx3dPlot() {
        this(800, 600);
    }

    public JavaFx3dPlot(int width, int height) {
        this(width, height, JavaFxThemeManager.THEME_SEABORN);
    }

    public JavaFx3dPlot(int width, int height, String theme) {
        this.width = width;
        this.height = height;
        this.themeManager = new JavaFxThemeManager(theme != null ? theme : JavaFxThemeManager.THEME_SEABORN);
        this.themeName = themeManager.getCurrentTheme();
        PlotStyle s = new PlotStyle(PlotStyle.defaultStyle());
        String[] pal = themeManager.getColorPalette();
        if (pal != null && pal.length > 0) {
            s.setColor(pal[0]);
            s.setFaceColor(pal[0]);
            s.setMarkerColor(pal[0]);
        }
        this.defaultSeriesStyle = s;
        JavaFx3dFxUtil.runOnFxThreadSync(this::initWorldGraph);
    }

    private static PhongMaterial phongData(Color diffuse) {
        PhongMaterial m = new PhongMaterial(diffuse);
        m.setSpecularColor(Color.color(1, 1, 1, 0.38));
        m.setSpecularPower(42);
        return m;
    }

    private void initWorldGraph() {
        contentRaw = new Group();
        axesDecoration = new Group();
        plotContent = new Group();
        contentRaw.getChildren().addAll(axesDecoration, plotContent);
        worldScale = new Scale(1, 1, 1);
        worldShift = new Translate(0, 0, 0);
        Group scaled = new Group(contentRaw);
        scaled.getTransforms().setAll(worldShift, worldScale);
        // 默认视角：略俯视 + 斜向，避免沿某一数据轴压扁成近似 2D。
        dragRx = new Rotate(-24, Rotate.X_AXIS);
        dragRy = new Rotate(42, Rotate.Y_AXIS);
        pivot = new Group(scaled);
        pivot.getTransforms().setAll(dragRy, dragRx);
    }

    private void ensureWorld() {
        if (contentRaw == null) {
            initWorldGraph();
        }
    }

    private PlotStyle effectiveStyle(PlotStyle overlay, String styleString) {
        return JavaFx3dStyleHelper.effectiveStyle(defaultSeriesStyle, overlay, styleString);
    }

    private static Color fxColor(PlotStyle st, boolean fill) {
        return JavaFx3dStyleHelper.fxColor(st, fill);
    }

    private String[] palette() {
        if (paletteName != null && ColorPalette.hasPalette(paletteName)) {
            String[] p = ColorPalette.getPalette(paletteName);
            if (p != null && p.length > 0) {
                return p;
            }
        }
        return themeManager.getColorPalette();
    }

    private Color paletteColor(int i) {
        String[] pal = palette();
        return JavaFxStyleApplier.parseColor(pal[i % pal.length]);
    }

    private void growBounds(double x, double y, double z) {
        if (Double.isNaN(minX)) {
            minX = maxX = x;
            minY = maxY = y;
            minZ = maxZ = z;
            return;
        }
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
        minZ = Math.min(minZ, z);
        maxZ = Math.max(maxZ, z);
    }

    /** 数据 (x,y,z) → 子节点平移：JavaFX (X,Y,Z)=(x,z,y)。 */
    private void place(javafx.scene.Node node, double xd, double yd, double zd) {
        growBounds(xd, yd, zd);
        double[] trans = CoordinateMapper.dataToTranslation(xd, yd, zd);
        node.setTranslateX(trans[0]);
        node.setTranslateY(trans[1]);
        node.setTranslateZ(trans[2]);
        plotContent.getChildren().add(node);
    }

    /**
     * 统一使用CoordinateMapper进行数据到场景的坐标转换。
     * 这是简化坐标系转换的核心方法。
     */
    private javafx.geometry.Point3D dataToScene(double xd, double yd, double zd) {
        return CoordinateMapper.dataToScene(xd, yd, zd);
    }

    void rebuildWorldTransform() {
        if (Double.isNaN(minX)) {
            worldScale.setX(1);
            worldScale.setY(1);
            worldScale.setZ(1);
            worldShift.setX(0);
            worldShift.setY(0);
            worldShift.setZ(0);
            refreshAxes();
            return;
        }
        double cx = (minX + maxX) / 2;
        double cy = (minY + maxY) / 2;
        double cz = (minZ + maxZ) / 2;
        double ex = Math.max(1e-9, maxX - minX);
        double ey = Math.max(1e-9, maxY - minY);
        double ez = Math.max(1e-9, maxZ - minZ);
        double ext = Math.max(ex, Math.max(ey, ez));
        double s = 220.0 / ext;
        worldScale.setX(s);
        worldScale.setY(s);
        worldScale.setZ(s);
        worldShift.setX(-cx * s);
        worldShift.setY(-cz * s);
        worldShift.setZ(-cy * s);
        refreshAxes();
    }

    private void refreshAxes() {
        if (axesDecoration == null) {
            return;
        }
        if (Double.isNaN(minX)) {
            axesDecoration.getChildren().clear();
            return;
        }
        double ex = Math.max(1e-9, maxX - minX);
        double ey = Math.max(1e-9, maxY - minY);
        double ez = Math.max(1e-9, maxZ - minZ);
        double ext = Math.max(ex, Math.max(ey, ez));
        double sWorld = 220.0 / ext;
        JavaFx3dAxesDecoration.rebuild(
                axesDecoration,
                minX, maxX, minY, maxY, minZ, maxZ,
                xLabel, yLabel, zLabel,
                xticks, yticks, zticks,
                themeManager,
                sWorld);
    }

    /** JavaFX 坐标下连接两点的圆柱（圆柱默认沿 Y）。 */
    private void addCylinderBetweenFx(javafx.geometry.Point3D p0, javafx.geometry.Point3D p1,
            double radius, PhongMaterial mat) {
        growBounds(p0.getX(), p0.getZ(), p0.getY());
        growBounds(p1.getX(), p1.getZ(), p1.getY());
        JavaFx3dMeshGeometry.addCylinderBetweenFx(plotContent, p0, p1, radius, mat);
    }

    private void addCylinderBetweenData(double x0, double y0, double z0, double x1, double y1, double z1,
            double radius, PhongMaterial mat) {
        addCylinderBetweenFx(
                new javafx.geometry.Point3D(x0, z0, y0),
                new javafx.geometry.Point3D(x1, z1, y1),
                radius,
                mat);
    }

    private void addArrowData(double x0, double y0, double z0, double ux, double uy, double uz,
            double shaftLen, double radius, PhongMaterial shaftMat, PhongMaterial headMat) {
        JavaFx3dPlotPrimitives.addArrowData(plotContent, this::addCylinderBetweenFx,
                p -> growBounds(p.getX(), p.getZ(), p.getY()),
                x0, y0, z0, ux, uy, uz, shaftLen, radius, shaftMat, headMat);
    }

    // —— 曲面三角网（ij：Z[i][j] 对应 (x[i], y[j])）——

    private void addSurfaceMesh(IVector<?> xv, IVector<?> yv, IMatrix<?> zmat, PhongMaterial mat,
            boolean wireframe, boolean twoSided) {
        JavaFx3dSurfaceContour.addSurfaceMesh(plotContent, this::growBounds, xv, yv, zmat, mat, wireframe, twoSided);
    }

    @Override
    public I3dPlot scatter3d(IVector x, IVector y, IVector z) {
        return scatter3d(x, y, z, (List<String>) null, (PlotStyle) null);
    }

    @Override
    public I3dPlot scatter3d(IVector x, IVector y, IVector z, List<String> hue) {
        return scatter3d(x, y, z, hue, (PlotStyle) null);
    }

    private void scatter3dImpl(IVector x, IVector y, IVector z, List<String> hue, PlotStyle style) {
        ensureWorld();
        int n = JavaFx3dPlotMath.len(x);
        JavaFx3dPlotMath.requireSameLength(n, x, y, z);

        // 性能优化：数据量大时自动降采样
        boolean needsSampling = DataSamplingUtils.needsSampling(n);
        int maxPoints = DataSamplingUtils.recommendMaxPoints();
        int step = needsSampling ? Math.max(1, n / maxPoints) : 1;
        int renderCount = (n + step - 1) / step;

        PlotStyle st = effectiveStyle(style, null);
        double minx = JavaFx3dPlotMath.minVec(x), maxx = JavaFx3dPlotMath.maxVec(x);
        double miny = JavaFx3dPlotMath.minVec(y), maxy = JavaFx3dPlotMath.maxVec(y);
        double minz = JavaFx3dPlotMath.minVec(z), maxz = JavaFx3dPlotMath.maxVec(z);
        double ex = Math.max(1e-9, maxx - minx);
        double ey = Math.max(1e-9, maxy - miny);
        double ez = Math.max(1e-9, maxz - minz);
        double ext = Math.max(ex, Math.max(ey, ez));
        double ms = Math.max(3, st.getMarkerSize());

        // 数据量大时减小点大小以提高性能
        double sizeScale = needsSampling ? 0.7 : 1.0;
        double rBase = JavaFx3dFxUtil.clamp(ext * 0.015 * (ms / 8.0) * sizeScale, ext * 1e-5, ext * 0.09);

        Map<String, Integer> cat = new HashMap<>();
        AtomicInteger nextHue = new AtomicInteger();

        // 构建图例
        if (hue != null && !hue.isEmpty()) {
            autoBuildLegend(hue);
        }

        // 使用LOD几何体级别（数据量大时使用低多边形球体）
        int sphereDivisions = needsSampling ? 12 : 20;

        for (int i = 0; i < n; i += step) {
            Color col = fxColor(st, true);
            if (hue != null && i < hue.size()) {
                String h = hue.get(i);
                Integer idx = cat.computeIfAbsent(h, k -> nextHue.getAndIncrement());
                col = paletteColor(idx);
            }
            Sphere sp = new Sphere(rBase, sphereDivisions);
            sp.setMaterial(phongData(col));
            double xd = x.get(i).doubleValue();
            double yd = y.get(i).doubleValue();
            double zd = z.get(i).doubleValue();
            place(sp, xd, yd, zd);
        }
        rebuildWorldTransform();

        // 如果进行了降采样，在标题中添加提示
        if (needsSampling && renderCount < n) {
            this.subtitleText = (this.subtitleText != null && !this.subtitleText.isEmpty()
                    ? this.subtitleText + " | " : "") +
                    String.format("显示 %d/%d 点 (%.1f%%)", renderCount, n, 100.0 * renderCount / n);
        }
    }

    @Override
    public I3dPlot scatter3d(IVector x, IVector y, IVector z, String styleString) {
        return scatter3d(x, y, z, null, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot scatter3d(IVector x, IVector y, IVector z, PlotStyle style) {
        return scatter3d(x, y, z, null, style);
    }

    @Override
    public I3dPlot scatter3d(IVector x, IVector y, IVector z, List<String> hue, String styleString) {
        return scatter3d(x, y, z, hue, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot scatter3d(IVector x, IVector y, IVector z, List<String> hue, PlotStyle style) {
        JavaFx3dFxUtil.runOnFxThreadSync(() -> scatter3dImpl(x, y, z, hue, style));
        return this;
    }

    @Override
    public I3dPlot scatterBubble3d(IVector x, IVector y, IVector z, IVector sizes) {
        return scatterBubble3d(x, y, z, sizes, (List<String>) null, (PlotStyle) null);
    }

    @Override
    public I3dPlot scatterBubble3d(IVector x, IVector y, IVector z, IVector sizes, List<String> hue) {
        return scatterBubble3d(x, y, z, sizes, hue, (PlotStyle) null);
    }

    @Override
    public I3dPlot scatterBubble3d(IVector x, IVector y, IVector z, IVector sizes, String styleString) {
        return scatterBubble3d(x, y, z, sizes, null, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot scatterBubble3d(IVector x, IVector y, IVector z, IVector sizes, PlotStyle style) {
        return scatterBubble3d(x, y, z, sizes, null, style);
    }

    @Override
    public I3dPlot scatterBubble3d(IVector x, IVector y, IVector z, IVector sizes, List<String> hue, String styleString) {
        return scatterBubble3d(x, y, z, sizes, hue, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot scatterBubble3d(IVector x, IVector y, IVector z, IVector sizes, List<String> hue, PlotStyle style) {
        JavaFx3dFxUtil.runOnFxThreadSync(() -> scatterBubble3dImpl(x, y, z, sizes, hue, style));
        return this;
    }

    private void scatterBubble3dImpl(IVector x, IVector y, IVector z, IVector sizes, List<String> hue, PlotStyle style) {
        ensureWorld();
        int n = JavaFx3dPlotMath.len(x);
        JavaFx3dPlotMath.requireSameLength(n, x, y, z, sizes);
        PlotStyle st = effectiveStyle(style, null);
        double smin = Double.POSITIVE_INFINITY, smax = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            double s = sizes.get(i).doubleValue();
            if (Double.isFinite(s)) {
                smin = Math.min(smin, s);
                smax = Math.max(smax, s);
            }
        }
        if (!Double.isFinite(smin) || smax <= smin) {
            smin = 0;
            smax = 1;
        }
        double minx = JavaFx3dPlotMath.minVec(x), maxx = JavaFx3dPlotMath.maxVec(x);
        double miny = JavaFx3dPlotMath.minVec(y), maxy = JavaFx3dPlotMath.maxVec(y);
        double minz = JavaFx3dPlotMath.minVec(z), maxz = JavaFx3dPlotMath.maxVec(z);
        double ex = Math.max(1e-9, maxx - minx);
        double ey = Math.max(1e-9, maxy - miny);
        double ez = Math.max(1e-9, maxz - minz);
        double ext = Math.max(ex, Math.max(ey, ez));
        double ms = Math.max(4, st.getMarkerSize());
        double rLo = ext * 0.008 * (ms / 8.0);
        double rHi = ext * 0.06 * (ms / 8.0);
        Map<String, Integer> cat = new HashMap<>();
        AtomicInteger nextHue = new AtomicInteger();
        for (int i = 0; i < n; i++) {
            Color col = fxColor(st, true);
            if (hue != null && i < hue.size()) {
                Integer idx = cat.computeIfAbsent(hue.get(i), k -> nextHue.getAndIncrement());
                col = paletteColor(idx);
            }
            double t = (sizes.get(i).doubleValue() - smin) / (smax - smin);
            double r = JavaFx3dFxUtil.clamp(rLo + t * (rHi - rLo), ext * 1e-5, ext * 0.12);
            Sphere sp = new Sphere(r, 18);
            sp.setMaterial(phongData(col));
            place(sp, x.get(i).doubleValue(), y.get(i).doubleValue(), z.get(i).doubleValue());
        }
        rebuildWorldTransform();
    }

    @Override
    public I3dPlot line3d(IVector x, IVector y, IVector z) {
        return line3d(x, y, z, (PlotStyle) null);
    }

    @Override
    public I3dPlot line3d(IVector x, IVector y, IVector z, String styleString) {
        return line3d(x, y, z, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot line3d(IVector x, IVector y, IVector z, PlotStyle style) {
        JavaFx3dFxUtil.runOnFxThreadSync(() -> {
            ensureWorld();
            int n = JavaFx3dPlotMath.len(x);
            JavaFx3dPlotMath.requireSameLength(n, x, y, z);
            PlotStyle st = effectiveStyle(style, null);
            double minx = JavaFx3dPlotMath.minVec(x), maxx = JavaFx3dPlotMath.maxVec(x);
            double miny = JavaFx3dPlotMath.minVec(y), maxy = JavaFx3dPlotMath.maxVec(y);
            double minz = JavaFx3dPlotMath.minVec(z), maxz = JavaFx3dPlotMath.maxVec(z);
            double ex = Math.max(1e-9, maxx - minx);
            double ey = Math.max(1e-9, maxy - miny);
            double ez = Math.max(1e-9, maxz - minz);
            double ext = Math.max(ex, Math.max(ey, ez));
            double lw = JavaFx3dFxUtil.clamp(
                    ext * 0.0034 * Math.max(1, st.getLineWidth() / 2.2), ext * 7e-4, ext * 0.028);
            Color col = fxColor(st, false);
            PhongMaterial mat = phongData(col);
            for (int i = 0; i < n - 1; i++) {
                addCylinderBetweenData(
                        x.get(i).doubleValue(), y.get(i).doubleValue(), z.get(i).doubleValue(),
                        x.get(i + 1).doubleValue(), y.get(i + 1).doubleValue(), z.get(i + 1).doubleValue(),
                        lw, mat);
            }
            for (int i = 0; i < n; i++) {
                Sphere joint = new Sphere(lw * 1.08, 12);
                joint.setMaterial(mat);
                place(joint, x.get(i).doubleValue(), y.get(i).doubleValue(), z.get(i).doubleValue());
            }
            rebuildWorldTransform();
        });
        return this;
    }

    @Override
    public I3dPlot density3d(IVector x, IVector y, IVector z) {
        return density3d(x, y, z, 24);
    }

    @Override
    public I3dPlot density3d(IVector x, IVector y, IVector z, int resolution) {
        return density3d(x, y, z, resolution, (PlotStyle) null);
    }

    @Override
    public I3dPlot density3d(IVector x, IVector y, IVector z, String styleString) {
        return density3d(x, y, z, 24, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot density3d(IVector x, IVector y, IVector z, PlotStyle style) {
        return density3d(x, y, z, 24, style);
    }

    @Override
    public I3dPlot density3d(IVector x, IVector y, IVector z, int resolution, String styleString) {
        return density3d(x, y, z, resolution, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot density3d(IVector x, IVector y, IVector z, int resolution, PlotStyle style) {
        JavaFx3dFxUtil.runOnFxThreadSync(() -> {
            ensureWorld();
            int n = JavaFx3dPlotMath.len(x);
            JavaFx3dPlotMath.requireSameLength(n, x, y, z);
            int g = resolution <= 0 ? 16 : JavaFx3dPlotMath.clampInt(resolution, 8, 40);
            double minx = JavaFx3dPlotMath.minVec(x), maxx = JavaFx3dPlotMath.maxVec(x);
            double miny = JavaFx3dPlotMath.minVec(y), maxy = JavaFx3dPlotMath.maxVec(y);
            double minz = JavaFx3dPlotMath.minVec(z), maxz = JavaFx3dPlotMath.maxVec(z);
            double[] rx = {minx, maxx};
            double[] ry = {miny, maxy};
            double[] rz = {minz, maxz};
            JavaFx3dPlotMath.padRange(rx);
            JavaFx3dPlotMath.padRange(ry);
            JavaFx3dPlotMath.padRange(rz);
            minx = rx[0];
            maxx = rx[1];
            miny = ry[0];
            maxy = ry[1];
            minz = rz[0];
            maxz = rz[1];
            double[][][] cnt = new double[g][g][g];
            double dx = (maxx - minx) / g;
            double dy = (maxy - miny) / g;
            double dz = (maxz - minz) / g;
            if (dx <= 0) {
                dx = 1;
            }
            if (dy <= 0) {
                dy = 1;
            }
            if (dz <= 0) {
                dz = 1;
            }
            double cmax = 0;
            for (int i = 0; i < n; i++) {
                int ix = JavaFx3dPlotMath.clampBin((x.get(i).doubleValue() - minx) / dx, g);
                int iy = JavaFx3dPlotMath.clampBin((y.get(i).doubleValue() - miny) / dy, g);
                int iz = JavaFx3dPlotMath.clampBin((z.get(i).doubleValue() - minz) / dz, g);
                cnt[ix][iy][iz] += 1;
                cmax = Math.max(cmax, cnt[ix][iy][iz]);
            }
            if (cmax <= 0) {
                cmax = 1;
            }
            PlotStyle st = effectiveStyle(style, null);
            Color base = fxColor(st, true);
            for (int i = 0; i < g; i++) {
                for (int j = 0; j < g; j++) {
                    for (int k = 0; k < g; k++) {
                        double f = cnt[i][j][k] / cmax;
                        if (f < 0.02) {
                            continue;
                        }
                        double cx = minx + (i + 0.5) * dx;
                        double cy = miny + (j + 0.5) * dy;
                        double cz = minz + (k + 0.5) * dz;
                        Box b = new Box(dx * 0.92, dz * 0.92 * f, dy * 0.92);
                        PhongMaterial pm = new PhongMaterial(Color.color(base.getRed(), base.getGreen(), base.getBlue(), 0.15 + 0.55 * f));
                        b.setMaterial(pm);
                        place(b, cx, cy, cz);
                    }
                }
            }
            rebuildWorldTransform();
        });
        return this;
    }

    @Override
    public I3dPlot bar3d(List<String> categories, IVector values) {
        return bar3d(categories, values, BarExtrusion3D.BOX);
    }

    @Override
    public I3dPlot bar3d(List<String> categories, IVector values, BarExtrusion3D extrusion) {
        return bar3d(categories, values, extrusion, (PlotStyle) null);
    }

    @Override
    public I3dPlot bar3d(List<String> xticks, IVector y, List<String> hue) {
        return bar3d(xticks, y, hue, BarExtrusion3D.BOX);
    }

    @Override
    public I3dPlot bar3d(List<String> xticks, IVector y, List<String> hue, BarExtrusion3D extrusion) {
        return bar3d(xticks, y, hue, extrusion, (PlotStyle) null);
    }

    @Override
    public I3dPlot bar3d(List<String> categories, IVector values, String styleString) {
        return bar3d(categories, values, BarExtrusion3D.BOX, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot bar3d(List<String> categories, IVector values, PlotStyle style) {
        return bar3d(categories, values, BarExtrusion3D.BOX, style);
    }

    @Override
    public I3dPlot bar3d(List<String> categories, IVector values, BarExtrusion3D extrusion, String styleString) {
        return bar3d(categories, values, extrusion, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot bar3d(List<String> categories, IVector values, BarExtrusion3D extrusion, PlotStyle style) {
        JavaFx3dFxUtil.runOnFxThreadSync(() -> {
            ensureWorld();
            Objects.requireNonNull(categories);
            Objects.requireNonNull(values);
            int nx = categories.size();
            if (nx == 0 || values.length() == 0) {
                return;
            }
            PlotStyle st = effectiveStyle(style, null);
            double gap = 1.0;
            double w = 0.65 * gap;
            Color col = fxColor(st, true);
            PhongMaterial mat = new PhongMaterial(col);
            for (int i = 0; i < nx && i < values.length(); i++) {
                double h = Math.max(1e-6, values.get(i).doubleValue());
                double xc = i * gap;
                addBarExtrusion(extrusion, w, h, w, mat, xc, 0, h / 2);
            }
            rebuildWorldTransform();
        });
        return this;
    }

    @Override
    public I3dPlot bar3d(List<String> xticks, IVector y, List<String> hue, String styleString) {
        return barGrouped3d(xticks, y, hue, BarExtrusion3D.BOX, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot bar3d(List<String> xticks, IVector y, List<String> hue, PlotStyle style) {
        return barGrouped3d(xticks, y, hue, BarExtrusion3D.BOX, style);
    }

    @Override
    public I3dPlot bar3d(List<String> xticks, IVector y, List<String> hue, BarExtrusion3D extrusion, String styleString) {
        return barGrouped3d(xticks, y, hue, extrusion, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot bar3d(List<String> xticks, IVector y, List<String> hue, BarExtrusion3D extrusion, PlotStyle style) {
        return barGrouped3d(xticks, y, hue, extrusion, style);
    }

    private I3dPlot barGrouped3d(List<String> xticks, IVector y, List<String> hue, BarExtrusion3D extrusion, PlotStyle style) {
        JavaFx3dFxUtil.runOnFxThreadSync(() -> {
            ensureWorld();
            if (xticks == null || hue == null || y == null || xticks.size() != hue.size() || xticks.size() != y.length()) {
                throw new IllegalArgumentException("分组柱须 xticks/hue/y 等长 / grouped bar requires equal lengths");
            }
            LinkedHashSet<String> ux = new LinkedHashSet<>(xticks);
            List<String> xOrder = new ArrayList<>(ux);
            LinkedHashSet<String> uh = new LinkedHashSet<>(hue);
            List<String> hOrder = new ArrayList<>(uh);
            int nx = xOrder.size();
            int nh = hOrder.size();
            double gap = 1.0;
            double bw = gap * 0.7 / nh;
            for (int xi = 0; xi < nx; xi++) {
                String xlk = xOrder.get(xi);
                for (int hg = 0; hg < nh; hg++) {
                    String hk = hOrder.get(hg);
                    double val = 0;
                    for (int k = 0; k < xticks.size(); k++) {
                        if (xticks.get(k).equals(xlk) && hue.get(k).equals(hk)) {
                            val = y.get(k).doubleValue();
                            break;
                        }
                    }
                    if (val <= 0) {
                        continue;
                    }
                    Color col = paletteColor(hg);
                    PhongMaterial mat = new PhongMaterial(col);
                    double xc = xi * gap + (hg - (nh - 1) / 2.0) * bw;
                    addBarExtrusion(extrusion, bw * 0.85, val, bw * 0.85, mat, xc, 0, val / 2);
                }
            }
            rebuildWorldTransform();
        });
        return this;
    }

    private void addBarExtrusion(BarExtrusion3D ex, double wx, double hz, double dz, PhongMaterial mat,
            double cx, double cy, double czCenter) {
        switch (ex) {
            case BOX -> {
                Box b = new Box(wx, hz, dz);
                b.setMaterial(mat);
                place(b, cx, cy, czCenter);
            }
            case CYLINDER -> {
                Cylinder c = new Cylinder(Math.min(wx, dz) / 2, hz);
                c.setMaterial(mat);
                place(c, cx, cy, czCenter);
            }
            case CONE -> {
                MeshView c = JavaFx3dMeshGeometry.coneMeshView(Math.min(wx, dz) / 2, hz, mat);
                place(c, cx, cy, czCenter);
            }
            default -> {
                Box b = new Box(wx, hz, dz);
                b.setMaterial(mat);
                place(b, cx, cy, czCenter);
            }
        }
    }

    @Override
    public I3dPlot pie3d(IVector data) {
        return pie3d(data, (List<String>) null, (PlotStyle) null);
    }

    @Override
    public I3dPlot pie3d(IVector data, List<String> labels) {
        return pie3d(data, labels, (PlotStyle) null);
    }

    @Override
    public I3dPlot pie3d(IVector data, String styleString) {
        return pie3d(data, null, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot pie3d(IVector data, PlotStyle style) {
        return pie3d(data, null, style);
    }

    @Override
    public I3dPlot pie3d(IVector data, List<String> labels, String styleString) {
        return pie3d(data, labels, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot pie3d(IVector data, List<String> labels, PlotStyle style) {
        JavaFx3dFxUtil.runOnFxThreadSync(() -> {
            ensureWorld();
            int n = JavaFx3dPlotMath.len(data);
            if (n == 0) {
                return;
            }
            double sum = 0;
            for (int i = 0; i < n; i++) {
                sum += Math.max(0, data.get(i).doubleValue());
            }
            if (sum <= 0) {
                sum = 1;
            }
            effectiveStyle(style, null);
            double a0 = 0;
            double R = 3.0;
            double thick = 0.35;
            for (int i = 0; i < n; i++) {
                double frac = Math.max(0, data.get(i).doubleValue()) / sum;
                double a1 = a0 + frac * 2 * Math.PI;
                Color col = paletteColor(i);
                addPieWedgePrimitive(R, thick, a0, a1, new PhongMaterial(col));
                a0 = a1;
            }
            rebuildWorldTransform();
        });
        return this;
    }

    private void addPieWedgePrimitive(double r, double thick, double a0, double a1, PhongMaterial mat) {
        JavaFx3dPlotPrimitives.addPieWedge(this::place, r, thick, a0, a1, mat);
    }

    @Override
    public I3dPlot hist3d(IVector x, IVector y) {
        return hist3d(x, y, 12, 12, (PlotStyle) null);
    }

    @Override
    public I3dPlot hist3d(IVector x, IVector y, int xBins, int yBins) {
        return hist3d(x, y, xBins, yBins, (PlotStyle) null);
    }

    @Override
    public I3dPlot hist3d(IVector x, IVector y, String styleString) {
        return hist3d(x, y, 12, 12, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot hist3d(IVector x, IVector y, PlotStyle style) {
        return hist3d(x, y, 12, 12, style);
    }

    @Override
    public I3dPlot hist3d(IVector x, IVector y, int xBins, int yBins, String styleString) {
        return hist3d(x, y, xBins, yBins, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot hist3d(IVector x, IVector y, int xBins, int yBins, PlotStyle style) {
        JavaFx3dFxUtil.runOnFxThreadSync(() -> {
            ensureWorld();
            int n = JavaFx3dPlotMath.len(x);
            JavaFx3dPlotMath.requireSameLength(n, x, y);
            int bx = Math.max(2, xBins);
            int by = Math.max(2, yBins);
            double[] rx = {JavaFx3dPlotMath.minVec(x), JavaFx3dPlotMath.maxVec(x)};
            double[] ry = {JavaFx3dPlotMath.minVec(y), JavaFx3dPlotMath.maxVec(y)};
            JavaFx3dPlotMath.padRange(rx);
            JavaFx3dPlotMath.padRange(ry);
            double dx = (rx[1] - rx[0]) / bx;
            double dy = (ry[1] - ry[0]) / by;
            double[][] cnt = new double[bx][by];
            double cmax = 0;
            for (int i = 0; i < n; i++) {
                int ix = JavaFx3dPlotMath.clampInt((int) Math.floor((x.get(i).doubleValue() - rx[0]) / dx), 0, bx - 1);
                int iy = JavaFx3dPlotMath.clampInt((int) Math.floor((y.get(i).doubleValue() - ry[0]) / dy), 0, by - 1);
                cnt[ix][iy] += 1;
                cmax = Math.max(cmax, cnt[ix][iy]);
            }
            if (cmax <= 0) {
                cmax = 1;
            }
            PlotStyle st = effectiveStyle(style, null);
            Color base = fxColor(st, true);
            for (int i = 0; i < bx; i++) {
                for (int j = 0; j < by; j++) {
                    double h = cnt[i][j];
                    if (h <= 0) {
                        continue;
                    }
                    double cx = rx[0] + (i + 0.5) * dx;
                    double cy = ry[0] + (j + 0.5) * dy;
                    double cz = (h / cmax) * Math.max(1, dx + dy);
                    Box b = new Box(dx * 0.9, cz, dy * 0.9);
                    b.setMaterial(new PhongMaterial(base));
                    place(b, cx, cy, cz / 2);
                }
            }
            rebuildWorldTransform();
        });
        return this;
    }

    @Override
    public I3dPlot boxplot3d(IVector data, List<String> labels) {
        return boxplot3d(data, labels, (PlotStyle) null);
    }

    @Override
    public I3dPlot boxplot3d(IVector data, List<String> labels, String styleString) {
        return boxplot3d(data, labels, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot boxplot3d(IVector data, List<String> labels, PlotStyle style) {
        JavaFx3dFxUtil.runOnFxThreadSync(() -> {
            ensureWorld();
            int n = JavaFx3dPlotMath.len(data);
            if (n == 0) {
                return;
            }
            double[] a = new double[n];
            for (int i = 0; i < n; i++) {
                a[i] = data.get(i).doubleValue();
            }
            Arrays.sort(a);
            addBoxAt(0, a, effectiveStyle(style, null));
            rebuildWorldTransform();
        });
        return this;
    }

    @Override
    public I3dPlot boxplot3d(IMatrix data, List<String> labels) {
        return boxplot3d(data, labels, (PlotStyle) null);
    }

    @Override
    public I3dPlot boxplot3d(IMatrix data, List<String> labels, String styleString) {
        return boxplot3d(data, labels, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot boxplot3d(IMatrix data, List<String> labels, PlotStyle style) {
        JavaFx3dFxUtil.runOnFxThreadSync(() -> {
            ensureWorld();
            int cols = data.cols();
            PlotStyle st = effectiveStyle(style, null);
            for (int c = 0; c < cols; c++) {
                int r = data.rows();
                double[] a = new double[r];
                for (int i = 0; i < r; i++) {
                    a[i] = data.get(i, c).doubleValue();
                }
                Arrays.sort(a);
                addBoxAt(c * 1.4, a, st);
            }
            rebuildWorldTransform();
        });
        return this;
    }

    private void addBoxAt(double x0, double[] sorted, PlotStyle st) {
        int n = sorted.length;
        double q1 = sorted[Math.max(0, n / 4)];
        double q2 = sorted[n / 2];
        double q3 = sorted[Math.min(n - 1, (3 * n) / 4)];
        double iqr = Math.max(1e-9, q3 - q1);
        double low = Math.max(sorted[0], q1 - 1.5 * iqr);
        double high = Math.min(sorted[n - 1], q3 + 1.5 * iqr);
        Color col = fxColor(st, true);
        PhongMaterial mat = new PhongMaterial(col);
        double bodyH = q3 - q1;
        Box body = new Box(0.35, bodyH, 0.35);
        body.setMaterial(mat);
        place(body, x0, 0, (q1 + q3) / 2);
        addCylinderBetweenData(x0, 0, low, x0, 0, q1, 0.05, mat);
        addCylinderBetweenData(x0, 0, q3, x0, 0, high, 0.05, mat);
        Box med = new Box(0.5, 0.07, 0.07);
        med.setMaterial(mat);
        place(med, x0, 0, q2);
    }

    @Override
    public I3dPlot surface3d(IVector x, IVector y, IMatrix z) {
        return surface3d(x, y, z, false, (PlotStyle) null);
    }

    @Override
    public I3dPlot surface3d(IVector x, IVector y, IMatrix z, boolean bottomContourProjection) {
        return surface3d(x, y, z, bottomContourProjection, (PlotStyle) null);
    }

    @Override
    public I3dPlot surface3d(IVector x, IVector y, IMatrix z, String styleString) {
        return surface3d(x, y, z, false, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot surface3d(IVector x, IVector y, IMatrix z, PlotStyle style) {
        return surface3d(x, y, z, false, style);
    }

    @Override
    public I3dPlot surface3d(IVector x, IVector y, IMatrix z, boolean bottomContourProjection, String styleString) {
        return surface3d(x, y, z, bottomContourProjection, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot surface3d(IVector x, IVector y, IMatrix z, boolean bottomContourProjection, PlotStyle style) {
        JavaFx3dFxUtil.runOnFxThreadSync(() -> {
            ensureWorld();
            PlotStyle st = effectiveStyle(style, null);
            PhongMaterial mat = new PhongMaterial(fxColor(st, true));
            addSurfaceMesh(x, y, z, mat, false, true);
            if (bottomContourProjection) {
                double zmin = JavaFx3dPlotMath.zMinMatrix(z);
                JavaFx3dSurfaceContour.addContourLinesOnPlane(this::addCylinderBetweenData, x, y, z,
                        zmin - 1e-3 * (JavaFx3dPlotMath.zMaxMatrix(z) - zmin + 1e-9), 6, fxColor(st, false));
            }
            rebuildWorldTransform();
        });
        return this;
    }

    @Override
    public I3dPlot contour3d(IVector x, IVector y, IMatrix z) {
        return contour3d(x, y, z, (PlotStyle) null);
    }

    @Override
    public I3dPlot contour3d(IVector x, IVector y, IMatrix z, String styleString) {
        return contour3d(x, y, z, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot contour3d(IVector x, IVector y, IMatrix z, PlotStyle style) {
        JavaFx3dFxUtil.runOnFxThreadSync(() -> {
            ensureWorld();
            double zp = JavaFx3dPlotMath.zMinMatrix(z);
            JavaFx3dSurfaceContour.addContourLinesOnPlane(this::addCylinderBetweenData, x, y, z, zp, 8,
                    fxColor(effectiveStyle(style, null), false));
            rebuildWorldTransform();
        });
        return this;
    }

    @Override
    public I3dPlot wireframe3d(IVector x, IVector y, IMatrix z) {
        return wireframe3d(x, y, z, (PlotStyle) null);
    }

    @Override
    public I3dPlot wireframe3d(IVector x, IVector y, IMatrix z, String styleString) {
        return wireframe3d(x, y, z, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot wireframe3d(IVector x, IVector y, IMatrix z, PlotStyle style) {
        JavaFx3dFxUtil.runOnFxThreadSync(() -> {
            ensureWorld();
            PhongMaterial mat = new PhongMaterial(fxColor(effectiveStyle(style, null), false));
            addSurfaceMesh(x, y, z, mat, true, false);
            rebuildWorldTransform();
        });
        return this;
    }

    @Override
    public I3dPlot heatmap3d(IMatrix z, List<String> xLabels, List<String> yLabels) {
        return heatmap3d(z, xLabels, yLabels, (PlotStyle) null);
    }

    @Override
    public I3dPlot heatmap3d(IMatrix z, List<String> xLabels, List<String> yLabels, String styleString) {
        return heatmap3d(z, xLabels, yLabels, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot heatmap3d(IMatrix z, List<String> xLabels, List<String> yLabels, PlotStyle style) {
        JavaFx3dFxUtil.runOnFxThreadSync(() -> {
            ensureWorld();
            int nx = z.rows();
            int ny = z.cols();
            if (xLabels != null && xLabels.size() != nx) {
                throw new IllegalArgumentException("xLabels 须与 Z 行数一致");
            }
            if (yLabels != null && yLabels.size() != ny) {
                throw new IllegalArgumentException("yLabels 须与 Z 列数一致");
            }
            double vmax = JavaFx3dPlotMath.zMaxMatrix(z);
            double vmin = JavaFx3dPlotMath.zMinMatrix(z);
            double span = Math.max(1e-9, vmax - vmin);
            effectiveStyle(style, null);
            for (int i = 0; i < nx; i++) {
                for (int j = 0; j < ny; j++) {
                    double val = z.get(i, j).doubleValue();
                    double h = Math.max(0.04, (val - vmin) / span * 3);
                    Color c = paletteColor((i + j * 3) % Math.max(1, palette().length));
                    Box b = new Box(0.85, h, 0.85);
                    b.setMaterial(new PhongMaterial(c));
                    place(b, i, j, h / 2);
                }
            }
            rebuildWorldTransform();
        });
        return this;
    }

    @Override
    public I3dPlot waterfall3d(IVector x, IMatrix layerHeights) {
        return waterfall3d(x, layerHeights, (PlotStyle) null);
    }

    @Override
    public I3dPlot waterfall3d(IVector x, IMatrix layerHeights, String styleString) {
        return waterfall3d(x, layerHeights, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot waterfall3d(IVector x, IMatrix layerHeights, PlotStyle style) {
        JavaFx3dFxUtil.runOnFxThreadSync(() -> {
            ensureWorld();
            int nr = layerHeights.rows();
            int nc = layerHeights.cols();
            if (JavaFx3dPlotMath.len(x) != nc) {
                throw new IllegalArgumentException("layerHeights 列数须与 x 长度一致");
            }
            PlotStyle st = effectiveStyle(style, null);
            PhongMaterial mat = new PhongMaterial(fxColor(st, false));
            double dyOff = 0.35;
            for (int r = 0; r < nr; r++) {
                double y0 = r * dyOff;
                for (int k = 0; k < nc - 1; k++) {
                    double xa = x.get(k).doubleValue();
                    double xb = x.get(k + 1).doubleValue();
                    double za = layerHeights.get(r, k).doubleValue();
                    double zb = layerHeights.get(r, k + 1).doubleValue();
                    addCylinderBetweenData(xa, y0, za, xb, y0, zb, 0.06, mat);
                }
            }
            rebuildWorldTransform();
        });
        return this;
    }

    @Override
    public I3dPlot vectorField3d(IVector x, IVector y, IVector z, IVector u, IVector v, IVector w) {
        return vectorField3d(x, y, z, u, v, w, (PlotStyle) null);
    }

    @Override
    public I3dPlot vectorField3d(IVector x, IVector y, IVector z, IVector u, IVector v, IVector w, String styleString) {
        return vectorField3d(x, y, z, u, v, w, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot vectorField3d(IVector x, IVector y, IVector z, IVector u, IVector v, IVector w, PlotStyle style) {
        JavaFx3dFxUtil.runOnFxThreadSync(() -> {
            ensureWorld();
            int n = JavaFx3dPlotMath.len(x);
            JavaFx3dPlotMath.requireSameLength(n, x, y, z, u, v, w);
            PlotStyle st = effectiveStyle(style, null);
            double maxS = 0;
            for (int i = 0; i < n; i++) {
                double nrm = Math.sqrt(
                        u.get(i).doubleValue() * u.get(i).doubleValue()
                                + v.get(i).doubleValue() * v.get(i).doubleValue()
                                + w.get(i).doubleValue() * w.get(i).doubleValue());
                maxS = Math.max(maxS, nrm);
            }
            if (maxS <= 0) {
                maxS = 1;
            }
            Color col = fxColor(st, true);
            PhongMaterial shaft = new PhongMaterial(col);
            PhongMaterial head = new PhongMaterial(col);
            double L = Math.max(0.2, JavaFx3dPlotMath.maxVec(x) - JavaFx3dPlotMath.minVec(x)
                    + JavaFx3dPlotMath.maxVec(y) - JavaFx3dPlotMath.minVec(y)) / Math.max(12, n / 2);
            for (int i = 0; i < n; i++) {
                double ux = u.get(i).doubleValue();
                double uy = v.get(i).doubleValue();
                double uz = w.get(i).doubleValue();
                double len = Math.sqrt(ux * ux + uy * uy + uz * uz);
                double scale = L * len / maxS;
                if (len < 1e-12) {
                    continue;
                }
                addArrowData(
                        x.get(i).doubleValue(), y.get(i).doubleValue(), z.get(i).doubleValue(),
                        ux, uy, uz,
                        scale, Math.max(0.03, st.getLineWidth() * 0.04), shaft, head);
            }
            rebuildWorldTransform();
        });
        return this;
    }

    @Override
    public I3dPlot streamlines3d(IVector x, IVector y, IVector z, IVector u, IVector v, IVector w) {
        return streamlines3d(x, y, z, u, v, w, (PlotStyle) null);
    }

    @Override
    public I3dPlot streamlines3d(IVector x, IVector y, IVector z, IVector u, IVector v, IVector w, String styleString) {
        return streamlines3d(x, y, z, u, v, w, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot streamlines3d(IVector x, IVector y, IVector z, IVector u, IVector v, IVector w, PlotStyle style) {
        JavaFx3dFxUtil.runOnFxThreadSync(() -> {
            ensureWorld();
            int n = JavaFx3dPlotMath.len(x);
            JavaFx3dPlotMath.requireSameLength(n, x, y, z, u, v, w);
            PlotStyle st = effectiveStyle(style, null);
            PhongMaterial mat = new PhongMaterial(fxColor(st, false));
            int seeds = Math.min(24, n);
            double step = Math.max(1e-3, (JavaFx3dPlotMath.maxVec(x) - JavaFx3dPlotMath.minVec(x)
                    + JavaFx3dPlotMath.maxVec(y) - JavaFx3dPlotMath.minVec(y)
                    + JavaFx3dPlotMath.maxVec(z) - JavaFx3dPlotMath.minVec(z)) / 80);
            for (int s = 0; s < seeds; s++) {
                int si = s * Math.max(1, n / Math.max(1, seeds));
                double px = x.get(si).doubleValue();
                double py = y.get(si).doubleValue();
                double pz = z.get(si).doubleValue();
                for (int stepi = 0; stepi < 40; stepi++) {
                    double[] vel = JavaFx3dPlotMath.idwVel(px, py, pz, x, y, z, u, v, w);
                    double nrm = Math.sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);
                    if (nrm < 1e-9) {
                        break;
                    }
                    double vx = vel[0] / nrm * step;
                    double vy = vel[1] / nrm * step;
                    double vz = vel[2] / nrm * step;
                    double qx = px + vx;
                    double qy = py + vy;
                    double qz = pz + vz;
                    addCylinderBetweenData(px, py, pz, qx, qy, qz, 0.04, mat);
                    px = qx;
                    py = qy;
                    pz = qz;
                }
            }
            rebuildWorldTransform();
        });
        return this;
    }

    @Override
    public I3dPlot terrain3d(IVector x, IVector y, IMatrix elevation) {
        return surface3d(x, y, elevation, false);
    }

    @Override
    public I3dPlot terrain3d(IVector x, IVector y, IMatrix elevation, String styleString) {
        return surface3d(x, y, elevation, false, styleString);
    }

    @Override
    public I3dPlot terrain3d(IVector x, IVector y, IMatrix elevation, PlotStyle style) {
        return surface3d(x, y, elevation, false, style);
    }

    @Override
    public I3dPlot graph3d(List<Map<String, Object>> nodes, List<Map<String, Object>> links) {
        return graph3d(nodes, links, (PlotStyle) null);
    }

    @Override
    public I3dPlot graph3d(List<Map<String, Object>> nodes, List<Map<String, Object>> links, String styleString) {
        return graph3d(nodes, links, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot graph3d(List<Map<String, Object>> nodes, List<Map<String, Object>> links, PlotStyle style) {
        JavaFx3dFxUtil.runOnFxThreadSync(() -> {
            ensureWorld();
            if (nodes == null || nodes.isEmpty()) {
                return;
            }
            int n = nodes.size();
            double[] px = new double[n];
            double[] py = new double[n];
            double[] pz = new double[n];
            double area = Math.max(400, n * 30);
            for (int i = 0; i < n; i++) {
                px[i] = (Math.random() - 0.5) * Math.sqrt(area);
                py[i] = (Math.random() - 0.5) * Math.sqrt(area);
                pz[i] = (Math.random() - 0.5) * Math.sqrt(area) * 0.6;
                Map<String, Object> nd = nodes.get(i);
                if (nd.get("x") instanceof Number) {
                    px[i] = ((Number) nd.get("x")).doubleValue();
                }
                if (nd.get("y") instanceof Number) {
                    py[i] = ((Number) nd.get("y")).doubleValue();
                }
                if (nd.get("z") instanceof Number) {
                    pz[i] = ((Number) nd.get("z")).doubleValue();
                }
            }
            JavaFx3dPlotMath.springLayout3d(px, py, pz, links, n, area);
            PlotStyle st = effectiveStyle(style, null);
            PhongMaterial nodeMat = new PhongMaterial(fxColor(st, true));
            PhongMaterial edgeMat = new PhongMaterial(Color.GRAY);
            for (int i = 0; i < n; i++) {
                Sphere s = new Sphere(0.35);
                s.setMaterial(nodeMat);
                place(s, px[i], py[i], pz[i]);
            }
            if (links != null) {
                for (Map<String, Object> L : links) {
                    int a = JavaFx3dPlotMath.asInt(L.get("source"), 0);
                    int b = JavaFx3dPlotMath.asInt(L.get("target"), 0);
                    if (a >= 0 && a < n && b >= 0 && b < n) {
                        addCylinderBetweenData(px[a], py[a], pz[a], px[b], py[b], pz[b], 0.05, edgeMat);
                    }
                }
            }
            rebuildWorldTransform();
        });
        return this;
    }

    @Override
    public I3dPlot areaFill3d(IVector x, IVector y, IVector z) {
        return areaFill3d(x, y, z, (PlotStyle) null);
    }

    @Override
    public I3dPlot areaFill3d(IVector x, IVector y, IVector z, String styleString) {
        return areaFill3d(x, y, z, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot areaFill3d(IVector x, IVector y, IVector z, PlotStyle style) {
        JavaFx3dFxUtil.runOnFxThreadSync(() -> {
            ensureWorld();
            int n = JavaFx3dPlotMath.len(x);
            JavaFx3dPlotMath.requireSameLength(n, x, y, z);
            double z0 = JavaFx3dPlotMath.minVec(z);
            PlotStyle st = effectiveStyle(style, null);
            Color col = fxColor(st, true);
            PhongMaterial mat = new PhongMaterial(Color.color(col.getRed(), col.getGreen(), col.getBlue(), 0.45));
            TriangleMesh mesh = new TriangleMesh();
            mesh.getTexCoords().addAll(0, 0);
            int b = 0;
            for (int i = 0; i < n - 1; i++) {
                double x0 = x.get(i).doubleValue();
                double y0 = y.get(i).doubleValue();
                double zA = z.get(i).doubleValue();
                double x1 = x.get(i + 1).doubleValue();
                double y1 = y.get(i + 1).doubleValue();
                double zB = z.get(i + 1).doubleValue();
                mesh.getPoints().addAll((float) x0, (float) zA, (float) y0);
                mesh.getPoints().addAll((float) x1, (float) zB, (float) y1);
                mesh.getPoints().addAll((float) x1, (float) z0, (float) y1);
                mesh.getPoints().addAll((float) x0, (float) z0, (float) y0);
                int o = b;
                mesh.getFaces().addAll(o, 0, o + 1, 0, o + 2, 0);
                mesh.getFaces().addAll(o, 0, o + 2, 0, o + 3, 0);
                b += 4;
                growBounds(x0, y0, zA);
                growBounds(x1, y1, zB);
            }
            MeshView mv = new MeshView(mesh);
            mv.setMaterial(mat);
            plotContent.getChildren().add(mv);
            rebuildWorldTransform();
        });
        return this;
    }

    @Override
    public I3dPlot radar3d(IMatrix data, List<String> indicators) {
        return radar3d(data, indicators, (List<String>) null, (PlotStyle) null);
    }

    @Override
    public I3dPlot radar3d(IMatrix data, List<String> indicators, List<String> seriesNames) {
        return radar3d(data, indicators, seriesNames, (PlotStyle) null);
    }

    @Override
    public I3dPlot radar3d(IMatrix data, List<String> indicators, String styleString) {
        return radar3d(data, indicators, null, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot radar3d(IMatrix data, List<String> indicators, PlotStyle style) {
        return radar3d(data, indicators, null, style);
    }

    @Override
    public I3dPlot radar3d(IMatrix data, List<String> indicators, List<String> seriesNames, String styleString) {
        return radar3d(data, indicators, seriesNames, effectiveStyle(null, styleString));
    }

    @Override
    public I3dPlot radar3d(IMatrix data, List<String> indicators, List<String> seriesNames, PlotStyle style) {
        JavaFx3dFxUtil.runOnFxThreadSync(() -> {
            ensureWorld();
            if (data == null || indicators == null) {
                return;
            }
            int axes = Math.min(indicators.size(), data.cols());
            int rows = data.rows();
            double maxV = 1;
            for (int i = 0; i < data.rows(); i++) {
                for (int j = 0; j < axes; j++) {
                    maxV = Math.max(maxV, Math.abs(data.get(i, j).doubleValue()));
                }
            }
            if (!Double.isFinite(maxV) || maxV <= 0) {
                maxV = 1;
            }
            for (int r = 0; r < rows; r++) {
                double yLayer = r * 1.2;
                Color c = paletteColor(r);
                PhongMaterial rm = new PhongMaterial(c);
                for (int t = 0; t < axes; t++) {
                    double ang = t * 2 * Math.PI / axes;
                    double vr = data.get(r, t).doubleValue() / maxV * 3;
                    double x0 = vr * Math.cos(ang);
                    double z0 = vr * Math.sin(ang);
                    int nt = (t + 1) % axes;
                    double ang2 = nt * 2 * Math.PI / axes;
                    double vr2 = data.get(r, nt).doubleValue() / maxV * 3;
                    double x1 = vr2 * Math.cos(ang2);
                    double z1 = vr2 * Math.sin(ang2);
                    addCylinderBetweenData(x0, yLayer, z0, x1, yLayer, z1, 0.06, rm);
                }
            }
            rebuildWorldTransform();
        });
        return this;
    }

    @Override
    public I3dPlot setDefaultStyle(PlotStyle style) {
        if (style != null) {
            this.defaultSeriesStyle = new PlotStyle(style);
        }
        return this;
    }

    @Override
    public I3dPlot setPalette(String paletteName) {
        this.paletteName = paletteName;
        return this;
    }

    @Override
    public I3dPlot enableStyleSystem(boolean enabled) {
        this.useStyleSystem = enabled;
        return this;
    }

    @Override
    public I3dPlot enableThemeSystem(boolean enabled) {
        this.useThemeSystem = enabled;
        return this;
    }

    @Override
    public I3dPlot applyTheme(String themeName) {
        if (themeName != null && useThemeSystem) {
            themeManager.setTheme(themeName);
            this.themeName = themeManager.getCurrentTheme();
            JavaFx3dFxUtil.runOnFxThreadSync(this::refreshAxes);
        }
        return this;
    }

    /**
     * 设置图例项列表（公开API，供用户自定义图例）。
     *
     * @param items 图例项列表
     * @return this instance for chaining
     */
    public I3dPlot setLegendItems(List<JavaFx3dLegend.LegendItem> items) {
        this.legendItems = items != null ? new ArrayList<>(items) : null;
        return this;
    }

    /**
     * 根据hue分组自动构建图例。
     *
     * @param hueGroups hue分组名称列表
     * @return this instance for chaining
     */
    private I3dPlot autoBuildLegend(List<String> hueGroups) {
        if (hueGroups == null || hueGroups.isEmpty()) return this;
        this.legendItems = JavaFx3dLegend.buildItemsFromHueGroups(
                new ArrayList<>(new LinkedHashSet<>(hueGroups)),
                palette());
        return this;
    }

    /**
     * 设置悬停Tooltip文本。
     *
     * @param text tooltip文本
     */
    public void setTooltipText(String text) {
        if (tooltipLabel != null) {
            tooltipLabel.setText(text);
            tooltipLabel.setVisible(text != null && !text.isEmpty());
        }
    }

    @Override
    public I3dPlot registerTheme(String themeName, EchartsThemeManager.CustomTheme theme) {
        if (themeName != null && theme != null) {
            EchartsThemeManager.registerCustomTheme(themeName, theme);
        }
        return this;
    }

    @Override
    public I3dPlot createGradientTheme(String themeName, String startColor, String endColor, String backgroundColor) {
        if (themeName == null) {
            return this;
        }
        EchartsThemeManager.CustomTheme ec = EchartsThemeManager.createGradientTheme(
                themeName, startColor, endColor, backgroundColor);
        EchartsThemeManager.registerCustomTheme(themeName, ec);
        JavaFxThemeManager.Theme jfx = JavaFxThemeManager.createGradientTheme(themeName, startColor, endColor, backgroundColor);
        JavaFxThemeManager.registerCustomTheme(themeName, jfx);
        return this;
    }

    @Override
    public I3dPlot createMonochromeTheme(String themeName, String baseColor, String backgroundColor) {
        if (themeName == null) {
            return this;
        }
        EchartsThemeManager.CustomTheme ec = EchartsThemeManager.createMonochromeTheme(
                themeName, baseColor, backgroundColor);
        EchartsThemeManager.registerCustomTheme(themeName, ec);
        String[] colors = new String[10];
        Arrays.fill(colors, baseColor);
        Map<String, Object> cfg = new HashMap<>();
        cfg.put("backgroundColor", backgroundColor);
        cfg.put("textColor", ColorPalette.getContrastColor(backgroundColor));
        cfg.put("color", colors);
        JavaFxThemeManager.registerCustomTheme(themeName, new JavaFxThemeManager.Theme(themeName, cfg));
        return this;
    }

    @Override
    public I3dPlot title(String titleText) {
        this.titleText = titleText != null ? titleText : "";
        return this;
    }

    @Override
    public I3dPlot title(String titleText, String subtitleText) {
        this.titleText = titleText != null ? titleText : "";
        this.subtitleText = subtitleText != null ? subtitleText : "";
        return this;
    }

    @Override
    public I3dPlot xlabel(String name) {
        this.xLabel = name != null ? name : "";
        JavaFx3dFxUtil.runOnFxThreadSync(this::refreshAxes);
        return this;
    }

    @Override
    public I3dPlot ylabel(String name) {
        this.yLabel = name != null ? name : "";
        JavaFx3dFxUtil.runOnFxThreadSync(this::refreshAxes);
        return this;
    }

    @Override
    public I3dPlot zlabel(String name) {
        this.zLabel = name != null ? name : "";
        JavaFx3dFxUtil.runOnFxThreadSync(this::refreshAxes);
        return this;
    }

    @Override
    public I3dPlot size(int width, int height) {
        this.width = width;
        this.height = height;
        return this;
    }

    @Override
    public I3dPlot theme(String theme) {
        applyTheme(theme);
        return this;
    }

    @Override
    public I3dPlot show() {
        JavaFx3dSceneSupport.show(this);
        return this;
    }

    /**
     * 在 JavaFX 线程上同步将当前 3D 内容快照为 PNG；无需先 {@link #show()}。
     *
     * @param file 输出路径（父目录不存在时会尝试创建）
     */
    public void writeSnapshotPng(File file) {
        Objects.requireNonNull(file, "file");
        JavaFx3dSceneSupport.writeSnapshotPng(this, file);
    }

    @Override
    public I3dPlot saveAsHtml(String filename) {
        try {
            String pngName = filename.replace(".html", ".png").replace(".HTML", ".png");
            writeSnapshotPng(new File(pngName));
            byte[] raw = Files.readAllBytes(Path.of(pngName));
            String b64 = Base64.getEncoder().encodeToString(raw);
            String html = "<!DOCTYPE html>\n<html><head><meta charset=\"UTF-8\"><title>"
                    + JavaFx3dFxUtil.jsonEscape(titleText.isEmpty() ? "3D" : titleText)
                    + "</title></head><body>\n<img src=\"data:image/png;base64," + b64 + "\"/>\n</body></html>\n";
            Files.writeString(Path.of(filename), html, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new PlotException("保存失败: " + e.getMessage(), e);
        }
        return this;
    }

    @Override
    public String toHtml() {
        return "<html><body><p>JavaFX3dPlot：请使用 show() 或 saveAsHtml()（内嵌快照 PNG）。</p></body></html>";
    }

    @Override
    public String toJson() {
        return "{\n  \"kind\":\"JavaFx3dPlot\",\n  \"title\":\""
                + JavaFx3dFxUtil.jsonEscape(titleText)
                + "\",\n  \"width\":"
                + width
                + ",\n  \"height\":"
                + height
                + "\n}";
    }

    @Override
    public void setTitle(String titleText) {
        this.titleText = titleText != null ? titleText : "";
    }

    @Override
    public void setTitle(String titleText, String subtitleText) {
        this.titleText = titleText != null ? titleText : "";
        this.subtitleText = subtitleText != null ? subtitleText : "";
    }

    @Override
    public void setXlabel(String name) {
        this.xLabel = name != null ? name : "";
        JavaFx3dFxUtil.runOnFxThreadSync(this::refreshAxes);
    }

    @Override
    public void setYlabel(String name) {
        this.yLabel = name != null ? name : "";
        JavaFx3dFxUtil.runOnFxThreadSync(this::refreshAxes);
    }

    @Override
    public void setZlabel(String name) {
        this.zLabel = name != null ? name : "";
        JavaFx3dFxUtil.runOnFxThreadSync(this::refreshAxes);
    }

    @Override
    public void setXticks(AxisTicks xticks) {
        this.xticks = xticks;
        JavaFx3dFxUtil.runOnFxThreadSync(this::refreshAxes);
    }

    @Override
    public void setYticks(AxisTicks yticks) {
        this.yticks = yticks;
        JavaFx3dFxUtil.runOnFxThreadSync(this::refreshAxes);
    }

    @Override
    public void setZticks(AxisTicks zticks) {
        this.zticks = zticks;
        JavaFx3dFxUtil.runOnFxThreadSync(this::refreshAxes);
    }

    @Override
    public int getWidth() {
        return width;
    }

    @Override
    public int getHeight() {
        return height;
    }

    @Override
    public String getTheme() {
        return themeName;
    }

    // ==================== 序列化处理 ====================

    /**
     * 自定义序列化：保存3D场景的配置数据，以便反序列化后重建场景。
     * 这是修复序列化设计缺陷的关键方法。
     */
    @Serial
    private void writeObject(java.io.ObjectOutputStream out) throws IOException {
        // 先写入非transient字段
        out.defaultWriteObject();

        // 序列化场景配置
        out.writeObject(themeName);
        out.writeObject(titleText);
        out.writeObject(subtitleText);
        out.writeObject(xLabel);
        out.writeObject(yLabel);
        out.writeObject(zLabel);
        out.writeInt(width);
        out.writeInt(height);
        out.writeObject(paletteName);
        out.writeBoolean(useStyleSystem);
        out.writeBoolean(useThemeSystem);

        // 序列化数据边界（用于重建场景）
        out.writeDouble(minX);
        out.writeDouble(maxX);
        out.writeDouble(minY);
        out.writeDouble(maxY);
        out.writeDouble(minZ);
        out.writeDouble(maxZ);

        // 序列化默认样式
        out.writeObject(defaultSeriesStyle);
    }

    /**
     * 自定义反序列化：恢复场景配置并在JavaFX线程上重建3D场景。
     */
    @Serial
    @SuppressWarnings("unchecked")
    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        // 先读取非transient字段
        in.defaultReadObject();

        // 恢复场景配置
        this.themeName = (String) in.readObject();
        this.titleText = (String) in.readObject();
        this.subtitleText = (String) in.readObject();
        this.xLabel = (String) in.readObject();
        this.yLabel = (String) in.readObject();
        this.zLabel = (String) in.readObject();
        this.width = in.readInt();
        this.height = in.readInt();
        this.paletteName = (String) in.readObject();
        this.useStyleSystem = in.readBoolean();
        this.useThemeSystem = in.readBoolean();

        // 恢复边界
        this.minX = in.readDouble();
        this.maxX = in.readDouble();
        this.minY = in.readDouble();
        this.maxY = in.readDouble();
        this.minZ = in.readDouble();
        this.maxZ = in.readDouble();

        // 恢复样式
        this.defaultSeriesStyle = (PlotStyle) in.readObject();

        // 在JavaFX线程上重建3D场景
        this.themeManager = new JavaFxThemeManager(themeName != null ? themeName : JavaFxThemeManager.THEME_SEABORN);
        JavaFx3dFxUtil.runOnFxThreadSync(this::initWorldGraph);
    }
}
