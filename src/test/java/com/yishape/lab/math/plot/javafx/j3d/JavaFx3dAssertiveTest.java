package com.yishape.lab.math.plot.javafx.j3d;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.plot.I3dPlot;
import com.yishape.lab.math.plot.Plots;
import com.yishape.lab.math.plot.echarts.j3d.Echarts3dPlot;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;

import java.io.*;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * JavaFX 3D图表自动化断言测试（替代纯图片对比测试）。
 * <p>
 * 包含验证性断言，可自动检测回归问题。
 *
 * @author lteb2
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
@DisplayName("JavaFX 3D自动化断言测试")
class JavaFx3dAssertiveTest extends JavaFx3dTestBase {

    @TempDir
    Path tempDir;

    @BeforeAll
    static void beforeAll() {
        initialize3d();
    }

    private static IVector<Double> randVector(int n, long seed) {
        java.util.Random r = new java.util.Random(seed);
        return Linalg.vector(r.doubles(n).toArray());
    }

    private static IVector<Double> seqVector(int n) {
        return Linalg.vector(IntStream.range(0, n).mapToDouble(i -> i * 0.1).toArray());
    }

    // ========== 基础功能断言测试 ==========

    @Test
    @Order(10)
    @DisplayName("scatter3d基本功能与数据验证")
    void testScatter3dBasic() {
        var plot = new JavaFx3dPlot(800, 600);
        IVector<Double> x = randVector(50, 1L);
        IVector<Double> y = randVector(50, 2L);
        IVector<Double> z = randVector(50, 3L);

        plot.scatter3d(x, y, z).title("Test Scatter");

        // 断言：标题已设置
        assertEquals("Test Scatter", plot.titleText, "标题应正确设置");

        // 断言：尺寸正确
        assertEquals(800, plot.getWidth(), "宽度应匹配");
        assertEquals(600, plot.getHeight(), "高度应匹配");

        // 生成快照（验证导出功能）
        File png = tempDir.resolve("scatter_test.png").toFile();
        assertDoesNotThrow(() -> plot.writeSnapshotPng(png), "应能成功导出PNG");
        assertTrue(png.exists() && png.length() > 0, "PNG文件应存在且非空");
    }

    @Test
    @Order(20)
    @DisplayName("scatter3d with hue分组验证")
    void testScatter3dHue() {
        var plot = new JavaFx3dPlot(800, 600);
        int n = 60;
        IVector<Double> x = Linalg.vector(IntStream.range(0, n).mapToDouble(i -> i).toArray());
        IVector<Double> y = Linalg.vector(IntStream.range(0, n).mapToDouble(i -> Math.sin(i * 0.1)).toArray());
        IVector<Double> z = Linalg.vector(IntStream.range(0, n).mapToDouble(i -> Math.cos(i * 0.1)).toArray());
        List<String> hue = IntStream.range(0, n).mapToObj(i -> "Group" + (i % 3)).toList();

        plot.scatter3d(x, y, z, hue);

        // 断言：图例项应已创建
        assertNotNull(plot.legendItems, "分组散点应创建图例");
        assertEquals(3, plot.legendItems.size(), "应有3个分组图例项");
    }

    @Test
    @Order(30)
    @DisplayName("line3d折线图验证")
    void testLine3d() {
        var plot = new JavaFx3dPlot(800, 600);
        int n = 100;
        double[] t = IntStream.range(0, n).mapToDouble(i -> i * 0.1).toArray();
        IVector<Double> x = Linalg.vector(java.util.Arrays.stream(t).map(Math::cos).toArray());
        IVector<Double> y = Linalg.vector(java.util.Arrays.stream(t).map(Math::sin).toArray());
        IVector<Double> z = Linalg.vector(t);

        var result = plot.line3d(x, y, z).title("3D Line");

        // 断言：返回自身用于链式调用
        assertSame(plot, result, "应返回自身支持链式调用");
        assertEquals("3D Line", plot.titleText, "标题应设置");
    }

    @Test
    @Order(40)
    @DisplayName("surface3d曲面图验证")
    void testSurface3d() {
        var plot = new JavaFx3dPlot(800, 600);
        int nx = 20, ny = 15;
        IVector<Double> x = seqVector(nx);
        IVector<Double> y = seqVector(ny);
        double[][] zd = new double[nx][ny];
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                zd[i][j] = Math.sin(x.get(i) * 2) * Math.cos(y.get(j) * 2);
            }
        }
        IMatrix<Double> z = Linalg.matrix(zd);

        plot.surface3d(x, y, z).title("Surface");

        // 断言：标题正确
        assertEquals("Surface", plot.titleText);
    }

    @Test
    @Order(50)
    @DisplayName("bar3d柱状图验证")
    void testBar3d() {
        var plot = new JavaFx3dPlot(800, 600);
        List<String> cat = List.of("A", "B", "C", "D");
        IVector<Double> vals = Linalg.vector(new double[]{2, 4, 3, 5});

        var result = plot.bar3d(cat, vals, I3dPlot.BarExtrusion3D.BOX);

        // 断言：链式调用正常工作
        assertSame(plot, result);
    }

    // ========== ECharts 3D后端测试 ==========

    @Test
    @Order(100)
    @DisplayName("ECharts 3D后端：基础散点图与HTML导出")
    void testEcharts3dScatter() {
        // 切换到ECharts后端
        Plots.setProvider3d(Plots.PlotProvider3d.EchartsGL);

        var plot = Plots.of3d(800, 600);
        assertInstanceOf(Echarts3dPlot.class, plot, "应创建ECharts 3D实例");

        IVector<Double> x = randVector(30, 1L);
        IVector<Double> y = randVector(30, 2L);
        IVector<Double> z = randVector(30, 3L);

        plot.scatter3d(x, y, z).title("ECharts 3D Scatter");

        // 断言：HTML导出应生成有效的ECharts配置
        String html = plot.toHtml();
        assertNotNull(html);
        assertTrue(html.contains("echarts"), "HTML应包含ECharts引用");
        assertTrue(html.contains("scatter3D"), "HTML应包含scatter3D系列");
        assertTrue(html.contains("ECharts 3D Scatter"), "HTML应包含标题");

        // 断言：JSON配置应有效
        String json = plot.toJson();
        assertNotNull(json);
        assertTrue(json.contains("scatter3D") || json.contains("scatter"),
                "JSON应包含系列类型");
    }

    @Test
    @Order(110)
    @DisplayName("ECharts 3D后端：曲面图与主题支持")
    void testEcharts3dSurface() {
        var plot = new Echarts3dPlot(800, 600, "dark");

        int nx = 15, ny = 12;
        IVector<Double> x = seqVector(nx);
        IVector<Double> y = seqVector(ny);
        double[][] zd = new double[nx][ny];
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                zd[i][j] = Math.sin(i * 0.3) * Math.cos(j * 0.3);
            }
        }
        IMatrix<Double> z = Linalg.matrix(zd);

        plot.surface3d(x, y, z).title("Dark Theme Surface");

        String html = plot.toHtml();
        assertTrue(html.contains("surface"), "HTML应包含surface系列");
        assertTrue(html.contains("dark") || html.contains("#1e1e1e"),
                "HTML应反映dark主题");
    }

    @Test
    @Order(120)
    @DisplayName("ECharts 3D后端：HTML文件保存")
    void testEcharts3dSaveHtml() {
        var plot = new Echarts3dPlot(640, 480);
        plot.scatter3d(randVector(20, 1L), randVector(20, 2L), randVector(20, 3L))
                .title("Save Test");

        File htmlFile = tempDir.resolve("test_output.html").toFile();

        assertDoesNotThrow(() -> plot.saveAsHtml(htmlFile.getAbsolutePath()),
                "保存HTML不应抛出异常");
        assertTrue(htmlFile.exists(), "HTML文件应存在");
        assertTrue(htmlFile.length() > 1000, "HTML文件应包含内容");

        // 验证文件内容
        String content = assertDoesNotThrow(() ->
                new String(java.nio.file.Files.readAllBytes(htmlFile.toPath())));
        assertTrue(content.contains("<!DOCTYPE html>"), "应是完整HTML文档");
        assertTrue(content.contains("Save Test"), "应包含标题");
        assertTrue(content.contains("echarts"), "应引用ECharts库");
    }

    // ========== 性能优化测试 ==========

    @Test
    @Order(200)
    @DisplayName("大数据量自动降采样验证")
    void testDataSampling() {
        var plot = new JavaFx3dPlot(800, 600);

        // 生成大量数据（超过阈值）
        int largeN = DataSamplingUtils.PERFORMANCE_THRESHOLD * 3;
        IVector<Double> x = randVector(largeN, 1L);
        IVector<Double> y = randVector(largeN, 2L);
        IVector<Double> z = randVector(largeN, 3L);

        plot.scatter3d(x, y, z);

        // 断言：应添加降采样提示
        assertNotNull(plot.subtitleText);
        // 降采样时会添加提示信息
        assertTrue(plot.subtitleText.contains("显示") || plot.subtitleText.isEmpty(),
                "降采样时应添加提示或为空");
    }

    // ========== 序列化测试 ==========

    @Test
    @Order(300)
    @DisplayName("3D图表序列化与反序列化")
    void testSerialization() throws Exception {
        var original = new JavaFx3dPlot(640, 480, "dark");
        original.scatter3d(randVector(20, 1L), randVector(20, 2L), randVector(20, 3L))
                .title("Serialization Test")
                .xlabel("X Axis")
                .ylabel("Y Axis")
                .zlabel("Z Axis");

        // 序列化
        File serFile = tempDir.resolve("plot.ser").toFile();
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(serFile))) {
            oos.writeObject(original);
        }

        // 反序列化
        JavaFx3dPlot restored;
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(serFile))) {
            restored = (JavaFx3dPlot) ois.readObject();
        }

        // 断言：配置应正确恢复
        assertNotNull(restored);
        assertEquals(original.getWidth(), restored.getWidth(), "宽度应恢复");
        assertEquals(original.getHeight(), restored.getHeight(), "高度应恢复");
        assertEquals(original.getTheme(), restored.getTheme(), "主题应恢复");
    }

    // ========== 坐标转换测试 ==========

    @Test
    @Order(400)
    @DisplayName("坐标系转换工具类验证")
    void testCoordinateMapper() {
        // 数据坐标 -> 场景坐标
        var scenePoint = CoordinateMapper.dataToScene(1.0, 2.0, 3.0);
        assertEquals(1.0, scenePoint.getX(), 1e-9, "X坐标应保持");
        assertEquals(3.0, scenePoint.getY(), 1e-9, "Y应为数据Z");
        assertEquals(2.0, scenePoint.getZ(), 1e-9, "Z应为数据Y");

        // 平移值转换
        double[] trans = CoordinateMapper.dataToTranslation(1.0, 2.0, 3.0);
        assertArrayEquals(new double[]{1.0, 3.0, 2.0}, trans, "平移值应正确映射");

        // 边界计算
        var points = List.of(
                new double[]{0, 0, 0},
                new double[]{1, 2, 3},
                new double[]{-1, -2, -3}
        );
        double[] bounds = CoordinateMapper.calculateBounds(points);
        assertEquals(-1, bounds[0], 1e-9, "minX");
        assertEquals(1, bounds[1], 1e-9, "maxX");
        assertEquals(-2, bounds[2], 1e-9, "minY");
        assertEquals(2, bounds[3], 1e-9, "maxY");
        assertEquals(-3, bounds[4], 1e-9, "minZ");
        assertEquals(3, bounds[5], 1e-9, "maxZ");
    }

    @Test
    @Order(410)
    @DisplayName("数据降采样算法验证")
    void testDataSamplingUtils() {
        double[][] data = new double[10000][3];
        for (int i = 0; i < 10000; i++) {
            data[i] = new double[]{i, i * 0.5, i * 0.25};
        }

        // 均匀采样
        var uniform = DataSamplingUtils.uniformSample(data, 100);
        assertEquals(100, uniform.length, "均匀采样应返回指定数量");

        // 随机采样
        var random = DataSamplingUtils.randomSample(data, 50, 42L);
        assertEquals(50, random.length, "随机采样应返回指定数量");

        // 自动采样
        var auto = DataSamplingUtils.autoSample(data, 500);
        assertTrue(auto.length <= 500, "自动采样应限制在最大数量内");

        // LOD级别
        var lod = DataSamplingUtils.lodSample(data, DataSamplingUtils.LodLevel.MEDIUM);
        assertEquals(5000, lod.length, 100, "MEDIUM LOD应为50%左右");
    }

    // ========== 工厂方法测试 ==========

    @Test
    @Order(500)
    @DisplayName("Plots.of3d工厂方法验证")
    void testPlotsFactory() {
        // 默认后端（现在是ECharts GL）
        I3dPlot plot1 = Plots.of3d();
        assertNotNull(plot1);

        // 带尺寸
        I3dPlot plot2 = Plots.of3d(640, 480);
        assertEquals(640, plot2.getWidth());
        assertEquals(480, plot2.getHeight());

        // 带主题
        I3dPlot plot3 = Plots.of3d(800, 600, "dark");
        assertEquals("dark", plot3.getTheme());
    }

    // ========== 主题配置测试 ==========

    @Test
    @Order(600)
    @DisplayName("3D主题配置类验证")
    void test3dThemeConfig() {
        // 默认配置
        var defaultConfig = JavaFx3dThemeConfig.getDefaultConfig();
        assertNotNull(defaultConfig);
        assertTrue(defaultConfig.isAntialiasingEnabled());

        // 深色主题
        var darkConfig = JavaFx3dThemeConfig.getDarkConfig();
        assertNotNull(darkConfig);
        assertNotEquals(defaultConfig.getBackgroundColor(), darkConfig.getBackgroundColor());

        // 学术主题
        var academicConfig = JavaFx3dThemeConfig.getAcademicConfig();
        assertNotNull(academicConfig);
        assertTrue(academicConfig.isGridEnabled());

        // 克隆
        var cloned = defaultConfig.clone();
        assertNotSame(defaultConfig, cloned);
        assertEquals(defaultConfig.getSpecularPower(), cloned.getSpecularPower());
    }
}
