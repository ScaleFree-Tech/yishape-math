package com.yishape.lab.math.testframework;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.plot.IPlot;
import com.yishape.lab.math.plot.PlotProvider;
import com.yishape.lab.math.plot.Plots;
import com.yishape.lab.math.plot.svg.SvgPlot;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test class for plot functionality.
 * Tests SVG backend thoroughly (file-based, no graphics environment needed).
 * JavaFX backend is only tested for factory creation (not rendering, requires graphics env).
 *
 * @author lteb2
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class ComprehensivePlotTest {

    private static final String RESULTS_DIR = "test_docs/results";
    private TestResult.Recorder recorder;

    @BeforeAll
    void setUp() {
        recorder = new TestResult.Recorder("plot", RESULTS_DIR);
        new File(RESULTS_DIR).mkdirs();
    }

    @AfterAll
    void tearDown() {
        recorder.writeToFile();
    }

    // ========== Helper Methods ==========

    /**
     * Verify that an SVG file exists, is non-empty, and contains valid SVG tags.
     */
    private void verifySvgFile(String filepath, TestResult r) {
        Path path = Paths.get(filepath);
        if (!Files.exists(path)) {
            r.fail("SVG file does not exist: " + filepath);
            return;
        }
        try {
            long size = Files.size(path);
            if (size == 0) {
                r.fail("SVG file is empty: " + filepath);
                return;
            }
            String content = Files.readString(path).toLowerCase();
            if (!content.contains("<svg")) {
                r.fail("SVG file missing <svg> tag: " + filepath);
                return;
            }
            if (!content.contains("</svg>") && !content.contains("/>")) {
                r.fail("SVG file appears incomplete: " + filepath);
                return;
            }
            r.pass("SVG file valid: " + filepath + " (" + size + " bytes)");
        } catch (IOException e) {
            r.fail("IOException reading SVG: " + e.getMessage());
        }
    }

    /**
     * Run a test that generates an SVG and verify the file.
     */
    private void runSvgTest(String testName, String subTest, PlotTestAction action) {
        TestResult r = recorder.record(testName, subTest);
        long t0 = System.currentTimeMillis();
        try {
            String filepath = action.run();
            verifySvgFile(filepath, r);
        } catch (Exception e) {
            r.fail("Exception: " + e.getClass().getSimpleName() + " - " + e.getMessage());
        }
        r.timeMs = System.currentTimeMillis() - t0;
    }

    @FunctionalInterface
    interface PlotTestAction {
        String run() throws Exception;
    }

    // ========== Factory Tests ==========

    @Test
    void testPlotsOfSvgFactory() {
        TestResult r = recorder.record("factory", "ofSvg");
        try {
            SvgPlot plot = Plots.ofSvg();
            assertNotNull(plot, "Plots.ofSvg() should return non-null SvgPlot");
            r.pass("Plots.ofSvg() returns non-null SvgPlot");
        } catch (Exception e) {
            r.fail("Exception: " + e.getMessage());
            fail(e);
        }
    }

    @Test
    void testPlotsOfSvgWithSize() {
        TestResult r = recorder.record("factory", "ofSvg_with_size");
        try {
            SvgPlot plot = Plots.ofSvg(800, 600);
            assertNotNull(plot, "Plots.ofSvg(800, 600) should return non-null SvgPlot");
            assertEquals(800, plot.getWidth(), "Width should be 800");
            assertEquals(600, plot.getHeight(), "Height should be 600");
            r.pass("Plots.ofSvg(800, 600) returns correct dimensions");
        } catch (Exception e) {
            r.fail("Exception: " + e.getMessage());
            fail(e);
        }
    }

    @Test
    void testPlotsOfSvgWithTheme() {
        TestResult r = recorder.record("factory", "ofSvg_with_theme");
        try {
            SvgPlot plot = Plots.ofSvg(800, 600, "dark");
            assertNotNull(plot, "Plots.ofSvg(800, 600, \"dark\") should return non-null SvgPlot");
            assertEquals("dark", plot.getTheme(), "Theme should be 'dark'");
            r.pass("Plots.ofSvg(800, 600, \"dark\") returns correct theme");
        } catch (Exception e) {
            r.fail("Exception: " + e.getMessage());
            fail(e);
        }
    }

    @Test
    void testPlotsOfJavaFxFactory() {
        TestResult r = recorder.record("factory", "ofJavaFx");
        try {
            IPlot plot = Plots.ofJavaFx();
            assertNotNull(plot, "Plots.ofJavaFx() should return non-null IPlot");
            r.pass("Plots.ofJavaFx() returns non-null IPlot");
        } catch (Exception e) {
            r.fail("Exception: " + e.getMessage());
            fail(e);
        }
    }

    @Test
    void testPlotsOfJavaFxWithSize() {
        TestResult r = recorder.record("factory", "ofJavaFx_with_size");
        try {
            IPlot plot = Plots.ofJavaFx(1024, 768);
            assertNotNull(plot, "Plots.ofJavaFx(1024, 768) should return non-null IPlot");
            assertEquals(1024, plot.getWidth(), "Width should be 1024");
            assertEquals(768, plot.getHeight(), "Height should be 768");
            r.pass("Plots.ofJavaFx(1024, 768) returns correct dimensions");
        } catch (Exception e) {
            r.fail("Exception: " + e.getMessage());
            fail(e);
        }
    }

    // ========== SVG Chart Type Tests ==========

    @Test
    void testSvgLineChart() {
        runSvgTest("svg", "line_chart", () -> {
            IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
            IVector<Double> y = Linalg.vector(new double[]{1, 4, 9, 16, 25});
            String filepath = RESULTS_DIR + "/test_line.svg";
            Plots.ofSvg()
                .line(x, y)
                .title("Test Line Chart")
                .xlabel("X Axis")
                .ylabel("Y Axis")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    @Test
    void testSvgScatterChart() {
        runSvgTest("svg", "scatter_chart", () -> {
            IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
            IVector<Double> y = Linalg.vector(new double[]{2.3, 4.5, 3.1, 6.7, 5.2, 8.1, 7.4, 9.8, 10.2, 11.5});
            String filepath = RESULTS_DIR + "/test_scatter.svg";
            Plots.ofSvg()
                .scatter(x, y)
                .title("Test Scatter Plot")
                .xlabel("X Values")
                .ylabel("Y Values")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    @Test
    void testSvgBarChart() {
        runSvgTest("svg", "bar_chart", () -> {
            IVector<Double> data = Linalg.vector(new double[]{30, 50, 20, 80, 45});
            String filepath = RESULTS_DIR + "/test_bar.svg";
            Plots.ofSvg()
                .bar(data)
                .title("Test Bar Chart")
                .xlabel("Categories")
                .ylabel("Values")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    @Test
    void testSvgHistogram() {
        runSvgTest("svg", "histogram", () -> {
            // Generate some normally-distributed-looking data
            IVector<Double> data = Linalg.vector(new double[]{
                1.2, 2.1, 1.8, 3.5, 2.9, 2.3, 1.5, 2.7, 3.1, 2.0,
                2.4, 1.9, 2.8, 3.2, 2.1, 1.7, 2.5, 2.3, 1.6, 2.9,
                3.0, 2.2, 1.8, 2.6, 2.4, 3.3, 1.4, 2.7, 2.1, 2.5
            });
            String filepath = RESULTS_DIR + "/test_histogram.svg";
            Plots.ofSvg()
                .hist(data, true)
                .title("Test Histogram with Fitting Line")
                .xlabel("Value")
                .ylabel("Frequency")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    @Test
    void testSvgHeatmap() {
        runSvgTest("svg", "heatmap", () -> {
            double[][] data = {
                {1.0, 0.8, 0.6, 0.4},
                {0.8, 1.0, 0.5, 0.3},
                {0.6, 0.5, 1.0, 0.7},
                {0.4, 0.3, 0.7, 1.0}
            };
            IMatrix<Double> matrix = Linalg.matrix(data);
            String filepath = RESULTS_DIR + "/test_heatmap.svg";
            Plots.ofSvg()
                .heatmap(matrix)
                .title("Test Heatmap")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    @Test
    void testSvgBoxplot() {
        runSvgTest("svg", "boxplot", () -> {
            IVector<Double> data = Linalg.vector(new double[]{
                10, 12, 15, 18, 20, 22, 25, 28, 30, 35,
                40, 42, 45, 48, 50, 55, 60, 65, 70, 80
            });
            String filepath = RESULTS_DIR + "/test_boxplot.svg";
            Plots.ofSvg()
                .boxplot(data)
                .title("Test Boxplot")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    @Test
    void testSvgPieChart() {
        runSvgTest("svg", "pie_chart", () -> {
            IVector<Double> data = Linalg.vector(new double[]{30, 25, 20, 15, 10});
            String filepath = RESULTS_DIR + "/test_pie.svg";
            Plots.ofSvg()
                .pie(data)
                .title("Test Pie Chart")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    @Test
    void testSvgAreaChart() {
        runSvgTest("svg", "area_chart", () -> {
            IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7});
            IVector<Double> y = Linalg.vector(new double[]{3, 5, 4, 7, 6, 8, 9});
            String filepath = RESULTS_DIR + "/test_area.svg";
            Plots.ofSvg()
                .area(x, y)
                .title("Test Area Chart")
                .xlabel("X")
                .ylabel("Y")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    // ========== SVG Subplots Test ==========

    @Test
    void testSvgSubplots() {
        runSvgTest("svg", "subplots", () -> {
            IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
            IVector<Double> y1 = Linalg.vector(new double[]{1, 4, 9, 16, 25});
            IVector<Double> y2 = Linalg.vector(new double[]{25, 16, 9, 4, 1});

            String filepath = RESULTS_DIR + "/test_subplots.svg";
            IPlot plot = Plots.ofSvg();
            plot.subplots(1, 2);

            plot.subplot(0, 0);
            plot.line(x, y1);
            plot.title("Subplot 1: Ascending");

            plot.subplot(0, 1);
            plot.line(x, y2);
            plot.title("Subplot 2: Descending");

            plot.saveAsSvg(filepath);
            return filepath;
        });
    }

    @Test
    void testSvgSubplots2x2() {
        runSvgTest("svg", "subplots_2x2", () -> {
            IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4});
            IVector<Double> y = Linalg.vector(new double[]{2, 4, 3, 5});

            String filepath = RESULTS_DIR + "/test_subplots_2x2.svg";
            IPlot plot = Plots.ofSvg();
            plot.subplots(2, 2);

            plot.subplot(0, 0);
            plot.line(x, y);
            plot.title("Top-Left");

            plot.subplot(0, 1);
            plot.scatter(x, y);
            plot.title("Top-Right");

            plot.subplot(1, 0);
            plot.bar(x);
            plot.title("Bottom-Left");

            plot.subplot(1, 1);
            plot.area(x, y);
            plot.title("Bottom-Right");

            plot.saveAsSvg(filepath);
            return filepath;
        });
    }

    // ========== SVG Theme Tests ==========

    @Test
    void testSvgThemeDark() {
        runSvgTest("svg", "theme_dark", () -> {
            IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
            IVector<Double> y = Linalg.vector(new double[]{2, 5, 3, 8, 7});
            String filepath = RESULTS_DIR + "/test_theme_dark.svg";
            Plots.ofSvg(800, 600, "dark")
                .line(x, y)
                .title("Dark Theme Test")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    @Test
    void testSvgThemeLight() {
        runSvgTest("svg", "theme_light", () -> {
            IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
            IVector<Double> y = Linalg.vector(new double[]{2, 5, 3, 8, 7});
            String filepath = RESULTS_DIR + "/test_theme_light.svg";
            Plots.ofSvg(800, 600, "light")
                .line(x, y)
                .title("Light Theme Test")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    @Test
    void testSvgThemeDefault() {
        runSvgTest("svg", "theme_default", () -> {
            IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
            IVector<Double> y = Linalg.vector(new double[]{2, 5, 3, 8, 7});
            String filepath = RESULTS_DIR + "/test_theme_default.svg";
            Plots.ofSvg(800, 600, "default")
                .line(x, y)
                .title("Default Theme Test")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    @Test
    void testSvgThemeSwitch() {
        runSvgTest("svg", "theme_switch", () -> {
            IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
            IVector<Double> y = Linalg.vector(new double[]{2, 5, 3, 8, 7});
            String filepath = RESULTS_DIR + "/test_theme_switch.svg";
            Plots.ofSvg()
                .line(x, y)
                .theme("dark")
                .title("Theme Switched to Dark")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    // ========== SVG Size/Configuration Tests ==========

    @Test
    void testSvgSizeMethod() {
        runSvgTest("svg", "size_method", () -> {
            IVector<Double> x = Linalg.vector(new double[]{1, 2, 3});
            IVector<Double> y = Linalg.vector(new double[]{1, 2, 3});
            String filepath = RESULTS_DIR + "/test_size.svg";
            Plots.ofSvg()
                .size(1200, 800)
                .line(x, y)
                .title("Custom Size 1200x800")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    @Test
    void testSvgLabels() {
        runSvgTest("svg", "labels", () -> {
            IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
            IVector<Double> y = Linalg.vector(new double[]{10, 20, 15, 25, 30});
            String filepath = RESULTS_DIR + "/test_labels.svg";
            Plots.ofSvg()
                .line(x, y)
                .title("Test Title", "Test Subtitle")
                .xlabel("X Label")
                .ylabel("Y Label")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    // ========== SVG Advanced Chart Tests ==========

    @Test
    void testSvgBoxplotWithLabels() {
        runSvgTest("svg", "boxplot_with_labels", () -> {
            IVector<Double> data = Linalg.vector(new double[]{
                10, 15, 20, 25, 30, 35, 40, 45, 50
            });
            List<String> labels = Arrays.asList("Group A");
            String filepath = RESULTS_DIR + "/test_boxplot_labels.svg";
            Plots.ofSvg()
                .boxplot(data, labels)
                .title("Boxplot with Labels")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    @Test
    void testSvgBarWithLabels() {
        runSvgTest("svg", "bar_with_xticks", () -> {
            List<String> categories = Arrays.asList("A", "B", "C", "D", "E");
            IVector<Double> values = Linalg.vector(new double[]{30, 50, 20, 80, 45});
            String filepath = RESULTS_DIR + "/test_bar_xticks.svg";
            Plots.ofSvg()
                .bar(categories, values)
                .title("Bar Chart with Categories")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    @Test
    void testSvgHeatmapWithLabels() {
        runSvgTest("svg", "heatmap_with_labels", () -> {
            double[][] data = {
                {1.0, 0.5, 0.2},
                {0.5, 1.0, 0.3},
                {0.2, 0.3, 1.0}
            };
            IMatrix<Double> matrix = Linalg.matrix(data);
            List<String> xLabels = Arrays.asList("X1", "X2", "X3");
            List<String> yLabels = Arrays.asList("Y1", "Y2", "Y3");
            String filepath = RESULTS_DIR + "/test_heatmap_labels.svg";
            Plots.ofSvg()
                .heatmap(matrix, xLabels, yLabels)
                .title("Heatmap with Labels")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    @Test
    void testSvgPieWithLabels() {
        runSvgTest("svg", "pie_with_labels", () -> {
            IVector<Double> data = Linalg.vector(new double[]{30, 25, 20, 15, 10});
            List<String> labels = Arrays.asList("Apple", "Banana", "Cherry", "Date", "Elderberry");
            String filepath = RESULTS_DIR + "/test_pie_labels.svg";
            Plots.ofSvg()
                .pie(data, labels, "default")
                .title("Pie Chart with Labels")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    // ========== SVG Grouped/Styled Chart Tests ==========

    @Test
    void testSvgGroupedBar() {
        runSvgTest("svg", "grouped_bar", () -> {
            IVector<Double> data = Linalg.vector(new double[]{30, 50, 20, 80, 45});
            List<String> hue = Arrays.asList("A", "B", "A", "B", "A");
            String filepath = RESULTS_DIR + "/test_grouped_bar.svg";
            Plots.ofSvg()
                .bar(data, hue)
                .title("Grouped Bar Chart")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    @Test
    void testSvgLineWithStyleString() {
        runSvgTest("svg", "line_with_style", () -> {
            IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
            IVector<Double> y = Linalg.vector(new double[]{1, 4, 9, 16, 25});
            String filepath = RESULTS_DIR + "/test_line_style.svg";
            Plots.ofSvg()
                .line(x, y, "r-o")
                .title("Line with Style String")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    @Test
    void testSvgScatterWithStyleString() {
        runSvgTest("svg", "scatter_with_style", () -> {
            IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
            IVector<Double> y = Linalg.vector(new double[]{2, 5, 3, 8, 7});
            String filepath = RESULTS_DIR + "/test_scatter_style.svg";
            Plots.ofSvg()
                .scatter(x, y, "b*")
                .title("Scatter with Style String")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    // ========== SVG Multi-Series Tests ==========

    @Test
    void testSvgMultiLine() {
        runSvgTest("svg", "multi_line", () -> {
            IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
            IVector<Double> y = Linalg.vector(new double[]{1, 4, 9, 16, 25});
            List<String> hue = Arrays.asList("S1", "S1", "S2", "S2", "S2");
            String filepath = RESULTS_DIR + "/test_multi_line.svg";
            Plots.ofSvg()
                .line(x, y, hue)
                .title("Multi-Line Chart")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    @Test
    void testSvgMultiScatter() {
        runSvgTest("svg", "multi_scatter", () -> {
            IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6});
            IVector<Double> y = Linalg.vector(new double[]{2, 4, 3, 6, 5, 7});
            List<String> hue = Arrays.asList("A", "B", "A", "B", "A", "B");
            String filepath = RESULTS_DIR + "/test_multi_scatter.svg";
            Plots.ofSvg()
                .scatter(x, y, hue)
                .title("Multi-Scatter Chart")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    // ========== SVG Chaining/Fluent API Tests ==========

    @Test
    void testSvgChainedCalls() {
        runSvgTest("svg", "chained_calls", () -> {
            IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
            IVector<Double> y = Linalg.vector(new double[]{2, 5, 3, 8, 7});
            String filepath = RESULTS_DIR + "/test_chained.svg";
            IPlot plot = Plots.ofSvg();
            plot.line(x, y)
                .title("Chained API Test")
                .xlabel("X")
                .ylabel("Y")
                .size(900, 700)
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    @Test
    void testSvgMultipleLayers() {
        runSvgTest("svg", "multiple_layers", () -> {
            IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
            IVector<Double> y1 = Linalg.vector(new double[]{1, 4, 9, 16, 25});
            IVector<Double> y2 = Linalg.vector(new double[]{2, 3, 5, 7, 11});
            String filepath = RESULTS_DIR + "/test_layers.svg";
            Plots.ofSvg()
                .line(x, y1)
                .scatter(x, y2)
                .title("Line + Scatter Overlay")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    // ========== SVG Output Format Tests ==========

    @Test
    void testSvgToHtml() {
        TestResult r = recorder.record("svg", "toHtml");
        try {
            IVector<Double> x = Linalg.vector(new double[]{1, 2, 3});
            IVector<Double> y = Linalg.vector(new double[]{1, 2, 3});
            SvgPlot plot = Plots.ofSvg();
            plot.line(x, y);
            String html = plot.toHtml();
            assertNotNull(html, "toHtml() should not return null");
            assertTrue(html.contains("<"), "HTML should contain tags");
            r.pass("toHtml() returns valid HTML content");
        } catch (UnsupportedOperationException e) {
            // SVG backend does not support HTML export - this is expected behavior
            r.pass("toHtml() correctly throws UnsupportedOperationException for SVG backend");
        } catch (Exception e) {
            r.fail("Exception: " + e.getMessage());
            fail(e);
        }
    }

    @Test
    void testSvgToJson() {
        TestResult r = recorder.record("svg", "toJson");
        try {
            IVector<Double> x = Linalg.vector(new double[]{1, 2, 3});
            IVector<Double> y = Linalg.vector(new double[]{1, 2, 3});
            SvgPlot plot = Plots.ofSvg();
            plot.line(x, y);
            String json = plot.toJson();
            assertNotNull(json, "toJson() should not return null");
            r.pass("toJson() returns content");
        } catch (Exception e) {
            r.fail("Exception: " + e.getMessage());
            fail(e);
        }
    }

    @Test
    void testSvgToBase64Svg() {
        TestResult r = recorder.record("svg", "toBase64Svg");
        try {
            IVector<Double> x = Linalg.vector(new double[]{1, 2, 3});
            IVector<Double> y = Linalg.vector(new double[]{1, 2, 3});
            SvgPlot plot = Plots.ofSvg();
            plot.line(x, y);
            String b64 = plot.toBase64Svg();
            assertNotNull(b64, "toBase64Svg() should not return null");
            assertFalse(b64.isEmpty(), "toBase64Svg() should not return empty string");
            r.pass("toBase64Svg() returns non-empty content");
        } catch (Exception e) {
            r.fail("Exception: " + e.getMessage());
            fail(e);
        }
    }

    // ========== SVG Edge Cases ==========

    @Test
    void testSvgSinglePoint() {
        runSvgTest("svg", "single_point", () -> {
            IVector<Double> x = Linalg.vector(new double[]{1});
            IVector<Double> y = Linalg.vector(new double[]{5});
            String filepath = RESULTS_DIR + "/test_single_point.svg";
            Plots.ofSvg()
                .scatter(x, y)
                .title("Single Point")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    @Test
    void testSvgLargeData() {
        runSvgTest("svg", "large_data", () -> {
            int n = 100;
            double[] xData = new double[n];
            double[] yData = new double[n];
            for (int i = 0; i < n; i++) {
                xData[i] = i;
                yData[i] = Math.sin(i * 0.1) * 10;
            }
            IVector<Double> x = Linalg.vector(xData);
            IVector<Double> y = Linalg.vector(yData);
            String filepath = RESULTS_DIR + "/test_large_data.svg";
            Plots.ofSvg()
                .line(x, y)
                .title("Large Dataset (100 points)")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    @Test
    void testSvgNegativeValues() {
        runSvgTest("svg", "negative_values", () -> {
            IVector<Double> x = Linalg.vector(new double[]{-5, -3, 0, 3, 5});
            IVector<Double> y = Linalg.vector(new double[]{-25, -9, 0, 9, 25});
            String filepath = RESULTS_DIR + "/test_negative.svg";
            Plots.ofSvg()
                .line(x, y)
                .title("Negative Values")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    @Test
    void testSvgZeroValues() {
        runSvgTest("svg", "zero_values", () -> {
            IVector<Double> x = Linalg.vector(new double[]{0, 0, 0, 0});
            IVector<Double> y = Linalg.vector(new double[]{0, 0, 0, 0});
            String filepath = RESULTS_DIR + "/test_zero.svg";
            Plots.ofSvg()
                .scatter(x, y)
                .title("All Zeros")
                .saveAsSvg(filepath);
            return filepath;
        });
    }

    // ========== JavaFX Factory-Only Tests (no rendering) ==========

    @Test
    void testJavaFxThemeMethod() {
        TestResult r = recorder.record("javafx", "theme_method");
        try {
            IPlot plot = Plots.ofJavaFx();
            IPlot themed = plot.theme("dark");
            assertNotNull(themed, "theme() should return non-null");
            r.pass("JavaFX theme() method works");
        } catch (Exception e) {
            r.fail("Exception: " + e.getMessage());
            // Do not fail the JUnit test - JavaFX may not be fully available
        }
    }

    @Test
    void testJavaFxSizeMethod() {
        TestResult r = recorder.record("javafx", "size_method");
        try {
            IPlot plot = Plots.ofJavaFx();
            IPlot sized = plot.size(1000, 800);
            assertNotNull(sized, "size() should return non-null");
            r.pass("JavaFX size() method works");
        } catch (Exception e) {
            r.fail("Exception: " + e.getMessage());
        }
    }

    @Test
    void testJavaFxTitleMethod() {
        TestResult r = recorder.record("javafx", "title_method");
        try {
            IPlot plot = Plots.ofJavaFx();
            IPlot titled = plot.title("Test Title");
            assertNotNull(titled, "title() should return non-null");
            r.pass("JavaFX title() method works");
        } catch (Exception e) {
            r.fail("Exception: " + e.getMessage());
        }
    }

    // ========== Plots Static Factory Method Tests ==========

    @Test
    void testPlotsStaticLine() {
        TestResult r = recorder.record("plots_static", "line");
        try {
            // Set provider to SVG for static factory test
            Plots.setProvider(PlotProvider.Svg);
            IVector<Double> x = Linalg.vector(new double[]{1, 2, 3});
            IVector<Double> y = Linalg.vector(new double[]{1, 2, 3});
            IPlot plot = Plots.line(x, y);
            assertNotNull(plot, "Plots.line() should return non-null");
            r.pass("Plots.line() static factory works");
        } catch (Exception e) {
            r.fail("Exception: " + e.getMessage());
            fail(e);
        }
    }

    @Test
    void testPlotsStaticScatter() {
        TestResult r = recorder.record("plots_static", "scatter");
        try {
            Plots.setProvider(PlotProvider.Svg);
            IVector<Double> x = Linalg.vector(new double[]{1, 2, 3});
            IVector<Double> y = Linalg.vector(new double[]{1, 2, 3});
            IPlot plot = Plots.scatter(x, y);
            assertNotNull(plot, "Plots.scatter() should return non-null");
            r.pass("Plots.scatter() static factory works");
        } catch (Exception e) {
            r.fail("Exception: " + e.getMessage());
            fail(e);
        }
    }

    @Test
    void testPlotsStaticBar() {
        TestResult r = recorder.record("plots_static", "bar");
        try {
            Plots.setProvider(PlotProvider.Svg);
            IVector<Double> data = Linalg.vector(new double[]{10, 20, 30});
            IPlot plot = Plots.bar(data);
            assertNotNull(plot, "Plots.bar() should return non-null");
            r.pass("Plots.bar() static factory works");
        } catch (Exception e) {
            r.fail("Exception: " + e.getMessage());
            fail(e);
        }
    }

    @Test
    void testPlotsStaticPie() {
        TestResult r = recorder.record("plots_static", "pie");
        try {
            Plots.setProvider(PlotProvider.Svg);
            IVector<Double> data = Linalg.vector(new double[]{30, 40, 30});
            IPlot plot = Plots.pie(data);
            assertNotNull(plot, "Plots.pie() should return non-null");
            r.pass("Plots.pie() static factory works");
        } catch (Exception e) {
            r.fail("Exception: " + e.getMessage());
            fail(e);
        }
    }

    @Test
    void testPlotsStaticHist() {
        TestResult r = recorder.record("plots_static", "hist");
        try {
            Plots.setProvider(PlotProvider.Svg);
            IVector<Double> data = Linalg.vector(new double[]{1, 2, 2, 3, 3, 3, 4, 4, 5});
            IPlot plot = Plots.hist(data, false);
            assertNotNull(plot, "Plots.hist() should return non-null");
            r.pass("Plots.hist() static factory works");
        } catch (Exception e) {
            r.fail("Exception: " + e.getMessage());
            fail(e);
        }
    }

    @Test
    void testPlotsStaticHeatmap() {
        TestResult r = recorder.record("plots_static", "heatmap");
        try {
            Plots.setProvider(PlotProvider.Svg);
            double[][] data = {{1, 2}, {3, 4}};
            IMatrix<Double> matrix = Linalg.matrix(data);
            IPlot plot = Plots.heatmap(matrix);
            assertNotNull(plot, "Plots.heatmap() should return non-null");
            r.pass("Plots.heatmap() static factory works");
        } catch (Exception e) {
            r.fail("Exception: " + e.getMessage());
            fail(e);
        }
    }

    @Test
    void testPlotsStaticBoxplot() {
        TestResult r = recorder.record("plots_static", "boxplot");
        try {
            Plots.setProvider(PlotProvider.Svg);
            IVector<Double> data = Linalg.vector(new double[]{10, 20, 30, 40, 50});
            IPlot plot = Plots.boxplot(data);
            assertNotNull(plot, "Plots.boxplot() should return non-null");
            r.pass("Plots.boxplot() static factory works");
        } catch (Exception e) {
            r.fail("Exception: " + e.getMessage());
            fail(e);
        }
    }

    @Test
    void testPlotsStaticSubplots() {
        TestResult r = recorder.record("plots_static", "subplots");
        try {
            Plots.setProvider(PlotProvider.Svg);
            IPlot plot = Plots.subplots(2, 2);
            assertNotNull(plot, "Plots.subplots() should return non-null");
            r.pass("Plots.subplots() static factory works");
        } catch (Exception e) {
            r.fail("Exception: " + e.getMessage());
            fail(e);
        }
    }

    @Test
    void testPlotsStaticTheme() {
        TestResult r = recorder.record("plots_static", "theme");
        try {
            Plots.setProvider(PlotProvider.Svg);
            IPlot plot = Plots.theme("dark");
            assertNotNull(plot, "Plots.theme() should return non-null");
            r.pass("Plots.theme() static factory works");
        } catch (Exception e) {
            r.fail("Exception: " + e.getMessage());
            fail(e);
        }
    }

    @Test
    void testPlotsProviderSwitch() {
        TestResult r = recorder.record("plots_static", "provider_switch");
        try {
            // Save current provider
            PlotProvider original = PlotProvider.JavaFx; // default

            // Switch to SVG
            Plots.setProvider(PlotProvider.Svg);
            IPlot svgPlot = Plots.of();
            assertTrue(svgPlot instanceof SvgPlot, "of() should return SvgPlot when provider is Svg");

            // Switch to JavaFx
            Plots.setProvider(PlotProvider.JavaFx);
            IPlot fxPlot = Plots.ofJavaFx();
            assertNotNull(fxPlot, "ofJavaFx() should return non-null");

            // Restore to SVG for other tests
            Plots.setProvider(PlotProvider.Svg);

            r.pass("Provider switching works correctly");
        } catch (Exception e) {
            r.fail("Exception: " + e.getMessage());
            fail(e);
        }
    }
}
