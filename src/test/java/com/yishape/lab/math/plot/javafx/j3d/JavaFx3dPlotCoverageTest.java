package com.yishape.lab.math.plot.javafx.j3d;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.plot.I3dPlot;
import com.yishape.lab.math.plot.Plots;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;
import org.junit.jupiter.api.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * {@link JavaFx3dPlot} / {@link I3dPlot} 功能覆盖测试：每种图表类型导出一张 PNG 至 {@code images_javafx/3d}。
 *
 * @author lteb2
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
@DisplayName("JavaFX 3D 绘图全覆盖快照")
class JavaFx3dPlotCoverageTest extends JavaFx3dTestBase {

    @BeforeAll
    static void beforeAll() {
        initialize3d();
        System.out.println("\n========== JavaFX 3D 全覆盖测试 · 输出: " + OUTPUT_DIR + " ==========\n");
    }

    private static JavaFx3dPlot newPlot() {
        return new JavaFx3dPlot(DEFAULT_W, DEFAULT_H, JavaFxThemeManager.THEME_SEABORN);
    }

    private static IMatrix<Double> meshZ(int nx, int ny) {
        double[] xv = IntStream.range(0, nx).mapToDouble(i -> i * 0.35).toArray();
        double[] yv = IntStream.range(0, ny).mapToDouble(j -> j * 0.32).toArray();
        double[][] zd = new double[nx][ny];
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                zd[i][j] = 2.0 * Math.sin(xv[i] * 0.4) * Math.cos(yv[j] * 0.4);
            }
        }
        return Linalg.matrix(zd);
    }

    private static IVector<Double> meshX(int nx) {
        return Linalg.vector(IntStream.range(0, nx).mapToDouble(i -> i * 0.35).toArray());
    }

    private static IVector<Double> meshY(int ny) {
        return Linalg.vector(IntStream.range(0, ny).mapToDouble(j -> j * 0.32).toArray());
    }

    private static IVector<Double> randCloud(int n, long seed) {
        java.util.Random r = new java.util.Random(seed);
        double[] a = new double[n];
        for (int i = 0; i < n; i++) {
            a[i] = r.nextGaussian();
        }
        return Linalg.vector(a);
    }

    @Test
    @Order(5)
    @DisplayName("scatter3d")
    void scatter3d_basic() {
        var p = newPlot();
        int n = 80;
        IVector<Double> x = randCloud(n, 1L);
        IVector<Double> y = randCloud(n, 2L);
        IVector<Double> z = randCloud(n, 3L);
        p.scatter3d(x, y, z).title("Scatter3D", "Gaussian 云");
        assertSnapshot(p, "j3d_01_scatter3d.png");
    }

    @Test
    @Order(10)
    @DisplayName("scatter3d + hue")
    void scatter3d_hue() {
        var p = newPlot();
        int n = 60;
        double[] xt = IntStream.range(0, n).mapToDouble(i -> Math.cos(i * 0.15) * (i + 1)).toArray();
        double[] yt = IntStream.range(0, n).mapToDouble(i -> Math.sin(i * 0.15) * (i + 1)).toArray();
        double[] zt = IntStream.range(0, n).mapToDouble(i -> i * 0.08).toArray();
        List<String> hue = IntStream.range(0, n).mapToObj(i -> "G" + (i % 3)).toList();
        p.scatter3d(Linalg.vector(xt), Linalg.vector(yt), Linalg.vector(zt), hue)
                .title("Scatter3D hue", "螺旋着色");
        assertSnapshot(p, "j3d_02_scatter3d_hue.png");
    }

    @Test
    @Order(15)
    @DisplayName("scatterBubble3d")
    void scatterBubble3d_test() {
        var p = newPlot();
        int n = 45;
        double[] x = IntStream.range(0, n).mapToDouble(i -> i * 0.2).toArray();
        double[] y = IntStream.range(0, n).mapToDouble(i -> Math.sin(i * 0.2)).toArray();
        double[] z = IntStream.range(0, n).mapToDouble(i -> Math.cos(i * 0.18)).toArray();
        double[] s = IntStream.range(0, n).mapToDouble(i -> 50 + i * 3).toArray();
        p.scatterBubble3d(Linalg.vector(x), Linalg.vector(y), Linalg.vector(z), Linalg.vector(s))
                .title("Bubble3D");
        assertSnapshot(p, "j3d_03_scatter_bubble.png");
    }

    @Test
    @Order(20)
    @DisplayName("line3d")
    void line3d_test() {
        var p = newPlot();
        double[] t = IntStream.range(0, 120).mapToDouble(i -> i * 0.08).toArray();
        double[] x = new double[t.length];
        double[] y = new double[t.length];
        double[] z = new double[t.length];
        for (int i = 0; i < t.length; i++) {
            x[i] = Math.cos(t[i]) * (3 + 0.3 * Math.sin(t[i] * 3));
            y[i] = Math.sin(t[i]) * (3 + 0.3 * Math.sin(t[i] * 3));
            z[i] = t[i] * 0.15;
        }
        p.line3d(Linalg.vector(x), Linalg.vector(y), Linalg.vector(z)).title("Line3D", "3D 螺线");
        assertSnapshot(p, "j3d_04_line3d.png");
    }

    @Test
    @Order(25)
    @DisplayName("route3d (default → line3d)")
    void route3d_test() {
        var p = newPlot();
        double[] x = {0, 2, 4, 6, 8};
        double[] y = {0, 1, 0.5, 2, 1};
        double[] z = {1, 2, 3, 2.5, 4};
        p.route3d(Linalg.vector(x), Linalg.vector(y), Linalg.vector(z)).title("Route3D");
        assertSnapshot(p, "j3d_05_route3d.png");
    }

    @Test
    @Order(30)
    @DisplayName("density3d")
    void density3d_test() {
        var p = newPlot();
        int n = 400;
        java.util.Random r = new java.util.Random(42);
        double[] x = new double[n];
        double[] y = new double[n];
        double[] z = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = r.nextGaussian();
            y[i] = r.nextGaussian() * 0.8;
            z[i] = r.nextGaussian() * 0.6;
        }
        p.density3d(Linalg.vector(x), Linalg.vector(y), Linalg.vector(z), 20).title("Density3D");
        assertSnapshot(p, "j3d_06_density3d.png");
    }

    @Test
    @Order(35)
    @DisplayName("bar3d BOX / CYLINDER / CONE + grouped")
    void bar3d_extrusions() {
        List<String> cat = List.of("A", "B", "C", "D");
        double[] v = {2.1, 3.4, 1.2, 4.0};
        var p1 = newPlot();
        p1.bar3d(cat, Linalg.vector(v), I3dPlot.BarExtrusion3D.BOX).title("Bar3D BOX");
        assertSnapshot(p1, "j3d_07_bar3d_box.png");

        var p2 = newPlot();
        p2.bar3d(cat, Linalg.vector(v), I3dPlot.BarExtrusion3D.CYLINDER).title("Bar3D CYLINDER");
        assertSnapshot(p2, "j3d_08_bar3d_cylinder.png");

        var p3 = newPlot();
        p3.bar3d(cat, Linalg.vector(v), I3dPlot.BarExtrusion3D.CONE).title("Bar3D CONE");
        assertSnapshot(p3, "j3d_09_bar3d_cone.png");

        var p4 = newPlot();
        List<String> xt = List.of("Q1", "Q1", "Q2", "Q2");
        List<String> hg = List.of("East", "West", "East", "West");
        double[] yv = {1.2, 1.5, 2.0, 1.8};
        p4.bar3d(xt, Linalg.vector(yv), hg, I3dPlot.BarExtrusion3D.BOX).title("Bar3D grouped");
        assertSnapshot(p4, "j3d_10_bar3d_grouped.png");
    }

    @Test
    @Order(40)
    @DisplayName("pie3d")
    void pie3d_test() {
        var p = newPlot();
        p.pie3d(Linalg.vector(new double[] {30, 25, 20, 15, 10}), List.of("A", "B", "C", "D", "E"))
                .title("Pie3D");
        assertSnapshot(p, "j3d_11_pie3d.png");
    }

    @Test
    @Order(45)
    @DisplayName("hist3d")
    void hist3d_test() {
        var p = newPlot();
        java.util.Random r = new java.util.Random(7);
        int n = 800;
        double[] x = new double[n];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = r.nextGaussian();
            y[i] = 0.6 * x[i] + 0.4 * r.nextGaussian();
        }
        p.hist3d(Linalg.vector(x), Linalg.vector(y), 14, 12).title("Hist3D");
        assertSnapshot(p, "j3d_12_hist3d.png");
    }

    @Test
    @Order(50)
    @DisplayName("boxplot3d vector + matrix")
    void boxplot3d_test() {
        var pv = newPlot();
        IVector<Double> v = Linalg.vector(new double[] {1, 3, 4, 5, 6, 7, 8, 12, 15, 22});
        pv.boxplot3d(v, List.of("All")).title("Boxplot3D 单组");
        assertSnapshot(pv, "j3d_13_boxplot3d_vector.png");

        var pm = newPlot();
        double[][] rect = {
                {2, 4, 1},
                {3, 5, 2},
                {4, 6, 2.5},
                {5, 7, 3},
                {6, 8, 10}
        };
        IMatrix<Double> m = Linalg.matrix(rect);
        pm.boxplot3d(m, List.of("S1", "S2", "S3")).title("Boxplot3D 多组");
        assertSnapshot(pm, "j3d_14_boxplot3d_matrix.png");
    }

    @Test
    @Order(55)
    @DisplayName("surface3d + bottom contour")
    void surface3d_tests() {
        int nx = 28;
        int ny = 22;
        var x = meshX(nx);
        var y = meshY(ny);
        var z = meshZ(nx, ny);
        var p1 = newPlot();
        p1.surface3d(x, y, z).title("Surface3D");
        assertSnapshot(p1, "j3d_15_surface3d.png");

        var p2 = newPlot();
        p2.surface3d(x, y, z, true).title("Surface3D + bottom contour");
        assertSnapshot(p2, "j3d_16_surface3d_meshc.png");
    }

    @Test
    @Order(60)
    @DisplayName("contour3d + wireframe3d")
    void contour_wireframe() {
        int nx = 22;
        int ny = 18;
        var x = meshX(nx);
        var y = meshY(ny);
        var z = meshZ(nx, ny);

        var p1 = newPlot();
        p1.contour3d(x, y, z).title("Contour3D");
        assertSnapshot(p1, "j3d_17_contour3d.png");

        var p2 = newPlot();
        p2.wireframe3d(x, y, z).title("Wireframe3D");
        assertSnapshot(p2, "j3d_18_wireframe3d.png");
    }

    @Test
    @Order(65)
    @DisplayName("heatmap3d + waterfall3d")
    void heatmap_waterfall() {
        double[][] hz = {
            {1, 3, 2, 4},
            {2, 2, 5, 1},
            {4, 1, 2, 3}
        };
        var p1 = newPlot();
        p1.heatmap3d(Linalg.matrix(hz), List.of("r0", "r1", "r2"), List.of("c0", "c1", "c2", "c3"))
                .title("Heatmap3D");
        assertSnapshot(p1, "j3d_19_heatmap3d.png");

        var p2 = newPlot();
        double[] xv = {0, 1, 2, 3, 4};
        double[][] layers = {
            {1.0, 1.2, 1.1, 0.9, 1.0},
            {0.8, 1.0, 1.3, 1.0, 0.95},
            {0.9, 0.85, 1.0, 1.15, 1.05}
        };
        p2.waterfall3d(Linalg.vector(xv), Linalg.matrix(layers)).title("Waterfall3D");
        assertSnapshot(p2, "j3d_20_waterfall3d.png");
    }

    @Test
    @Order(70)
    @DisplayName("vectorField3d + streamlines3d")
    void field_and_stream() {
        int g = 4;
        int n = g * g * g;
        double[] x = new double[n];
        double[] y = new double[n];
        double[] z = new double[n];
        double[] u = new double[n];
        double[] v = new double[n];
        double[] w = new double[n];
        int idx = 0;
        for (int i = 0; i < g; i++) {
            for (int j = 0; j < g; j++) {
                for (int k = 0; k < g; k++) {
                    x[idx] = i * 0.6;
                    y[idx] = j * 0.6;
                    z[idx] = k * 0.5;
                    u[idx] = Math.sin(y[idx]);
                    v[idx] = -Math.cos(x[idx]);
                    w[idx] = 0.15;
                    idx++;
                }
            }
        }
        var p1 = newPlot();
        p1.vectorField3d(Linalg.vector(x), Linalg.vector(y), Linalg.vector(z),
                        Linalg.vector(u), Linalg.vector(v), Linalg.vector(w))
                .title("VectorField3D");
        assertSnapshot(p1, "j3d_21_vector_field.png");

        var p2 = newPlot();
        p2.streamlines3d(Linalg.vector(x), Linalg.vector(y), Linalg.vector(z),
                        Linalg.vector(u), Linalg.vector(v), Linalg.vector(w))
                .title("Streamlines3D");
        assertSnapshot(p2, "j3d_22_streamlines3d.png");
    }

    @Test
    @Order(75)
    @DisplayName("terrain3d")
    void terrain3d_test() {
        int nx = 26;
        int ny = 20;
        var xv = meshX(nx);
        var yv = meshY(ny);
        var el = meshZ(nx, ny);
        var p = newPlot();
        p.terrain3d(xv, yv, el).title("Terrain3D", "DEM 示意");
        assertSnapshot(p, "j3d_23_terrain3d.png");
    }

    @Test
    @Order(80)
    @DisplayName("graph3d")
    void graph3d_test() {
        List<Map<String, Object>> nodes = new ArrayList<>();
        for (int i = 0; i < 6; i++) {
            Map<String, Object> m = new HashMap<>();
            m.put("x", (double) (i % 3) * 1.5);
            m.put("y", (double) (i / 3) * 1.5);
            m.put("z", (double) (i % 2) * 0.5);
            nodes.add(m);
        }
        List<Map<String, Object>> links = new ArrayList<>();
        links.add(Map.of("source", 0, "target", 1));
        links.add(Map.of("source", 1, "target", 2));
        links.add(Map.of("source", 0, "target", 3));
        links.add(Map.of("source", 3, "target", 4));
        links.add(Map.of("source", 2, "target", 5));
        var p = newPlot();
        p.graph3d(nodes, links).title("Graph3D");
        assertSnapshot(p, "j3d_24_graph3d.png");
    }

    @Test
    @Order(85)
    @DisplayName("areaFill3d")
    void areaFill3d_test() {
        double[] x = IntStream.range(0, 40).mapToDouble(i -> i * 0.12).toArray();
        double[] y = new double[x.length];
        double[] z = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = Math.sin(x[i]) * 1.5;
            z[i] = 1.2 + 0.5 * Math.cos(x[i] * 2);
        }
        var p = newPlot();
        p.areaFill3d(Linalg.vector(x), Linalg.vector(y), Linalg.vector(z)).title("AreaFill3D");
        assertSnapshot(p, "j3d_25_area_fill3d.png");
    }

    @Test
    @Order(90)
    @DisplayName("radar3d")
    void radar3d_test() {
        double[][] data = {
            {0.7, 0.5, 0.9, 0.6},
            {0.5, 0.8, 0.4, 0.85}
        };
        var p = newPlot();
        p.radar3d(Linalg.matrix(data), List.of("Cost", "Speed", "Rel.", "UX"),
                List.of("Product A", "Product B")).title("Radar3D");
        assertSnapshot(p, "j3d_26_radar3d.png");
    }

    @Test
    @Order(95)
    @DisplayName("Plots.of3d + dark theme + style string")
    void plots_factory_theme_style() {
        int n = 50;
        java.util.Random rnd = new java.util.Random(12345L);
        double[] x = new double[n];
        double[] y = new double[n];
        double[] z = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = rnd.nextDouble();
            y[i] = rnd.nextDouble();
            z[i] = rnd.nextDouble();
        }
        Assertions.assertInstanceOf(JavaFx3dPlot.class, Plots.of3d(680, 500));
        var p = (JavaFx3dPlot) Plots.of3d(680, 500, JavaFxThemeManager.THEME_DARK);
        p.scatter3d(Linalg.vector(x), Linalg.vector(y), Linalg.vector(z), "ms=9; color=#88c0d0")
                .title("Factory + dark + style", "Plots.of3d");
        assertSnapshot(p, "j3d_27_theme_dark_style.png");
    }
}
