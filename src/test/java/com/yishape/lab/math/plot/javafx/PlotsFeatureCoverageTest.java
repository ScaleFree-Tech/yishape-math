package com.yishape.lab.math.plot.javafx;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.plot.IPlot;
import com.yishape.lab.math.plot.JointplotMarginal;
import com.yishape.lab.math.plot.PairplotDiagonal;
import com.yishape.lab.math.plot.PlotAxisScale;
import com.yishape.lab.math.plot.PlotStyle;
import com.yishape.lab.math.plot.Plots;
import com.yishape.lab.math.plot.javafx.base.JavaFxChartTestBase;
import org.junit.jupiter.api.*;

import java.util.List;
import java.util.Random;

/**
 * 覆盖 {@link Plots} / {@link IPlot} 中扩展笛卡尔能力：KDE、回归、pairplot、jointplot、Q-Q、
 * 面积/阶梯/条形堆叠、误差棒、气泡、双 Y 轴、分面、对数轴，以及主题与样式字符串。
 * 数据尽量贴近常见统计/公开数据集尺度，而非随意常数。
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class PlotsFeatureCoverageTest extends JavaFxChartTestBase {

    /** Iris Setosa：萼片长（cm），UCI 经典集前 27 条，按种内真实观测顺序。 */
    private static final double[] IRIS_SETOSA_SEPAL_LENGTH = {
        5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9, 5.4, 4.8, 4.8, 4.3, 5.8, 5.7, 5.4, 5.1, 5.7, 5.1,
        5.4, 5.1, 4.9, 5.1, 4.8, 5.0, 5.0,
    };

    /**
     * mtcars 前 14 行：mpg、排量(cu.in.)、马力、重量(1000lb)。
     * 来源：Henderson & Velleman, 1981；R 内置 mtcars。
     */
    private static final double[] MTCARS_MPG = {
        21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2,
    };
    private static final double[] MTCARS_DISP = {
        160.0, 160.0, 108.0, 258.0, 360.0, 225.0, 360.0, 146.7, 140.8, 167.6, 167.6, 275.8, 275.8, 275.8,
    };
    private static final double[] MTCARS_HP = {
        110, 110, 93, 110, 175, 105, 245, 62, 95, 123, 123, 180, 180, 180,
    };
    private static final double[] MTCARS_WT = {
        2.620, 2.875, 2.320, 3.215, 3.440, 3.460, 3.570, 3.190, 3.150, 3.440, 3.440, 4.070, 3.730, 3.780,
    };

    /** 德国汉堡 12 个月降水量近似（mm），与 DWD 常年量级一致，用于累计面积示意。 */
    private static final double[] HAMBURG_MONTHLY_RAIN_MM = {
        68, 54, 51, 46, 55, 73, 79, 78, 70, 65, 69, 73,
    };

    /**
     * 美联储联邦基金目标区间中点（%）：2022Q1–2024Q4 代表性时点，阶梯近似公开政策路径。
     */
    private static final double[] FED_EFF_RATE_QUARTER = {
        0.08, 0.83, 2.33, 4.33, 4.83, 5.08, 5.33, 5.33, 5.33, 4.58,
    };

    private static double[] boxMullerGaussian(int n, long seed) {
        Random rnd = new Random(seed);
        double[] z = new double[n];
        for (int i = 0; i < n; i += 2) {
            double u1 = Math.nextUp(rnd.nextDouble());
            double u2 = rnd.nextDouble();
            double r = Math.sqrt(-2.0 * Math.log(u1));
            z[i] = r * Math.cos(2.0 * Math.PI * u2);
            if (i + 1 < n) {
                z[i + 1] = r * Math.sin(2.0 * Math.PI * u2);
            }
        }
        return z;
    }

    @BeforeAll
    static void setUp() {
        initialize();
        resetCounters();
        System.out.println("\n========== Plots 扩展功能覆盖测试 ==========");
    }

    @Test
    @Order(1)
    @DisplayName("KDE：萼片长分布")
    void testKdeIrisSepal() {
        IVector<Double> v = Linalg.vector(IRIS_SETOSA_SEPAL_LENGTH);
        IPlot p = Plots.of(820, 520)
                .kdeplot(v, 256, 0.0)
                .title("KDE：Iris setosa 萼片长", "Scott 带宽")
                .xlabel("Sepal length (cm)")
                .ylabel("密度");
        Assertions.assertTrue(generateImage(p, "ext_01_kde_iris_sepal.png"));
    }

    @Test
    @Order(2)
    @DisplayName("回归：车重 vs 油耗（无置信带）")
    void testRegplotMtcars() {
        IPlot p = Plots.of(820, 520)
                .regplot(Linalg.vector(MTCARS_WT), Linalg.vector(MTCARS_MPG), false)
                .title("OLS：每加仑英里 ~ 车重", "mtcars 前 14 行")
                .xlabel("Weight (1000 lb)")
                .ylabel("MPG");
        Assertions.assertTrue(generateImage(p, "ext_02_regplot_mtcars.png"));
    }

    @Test
    @Order(3)
    @DisplayName("回归：带均值响应置信带")
    void testRegplotWithBand() {
        IPlot p = Plots.of(820, 520)
                .regplot(Linalg.vector(MTCARS_DISP), Linalg.vector(MTCARS_HP), true)
                .title("马力 ~ 排量", "近似 95% 均值带")
                .xlabel("Displacement (cu.in.)")
                .ylabel("HP");
        Assertions.assertTrue(generateImage(p, "ext_03_regplot_confidence.png"));
    }

    @Test
    @Order(4)
    @DisplayName("Pairplot：对角 KDE")
    void testPairplotKdeDiagonal() {
        double[][] m = new double[MTCARS_MPG.length][4];
        for (int i = 0; i < MTCARS_MPG.length; i++) {
            m[i][0] = MTCARS_MPG[i];
            m[i][1] = MTCARS_DISP[i];
            m[i][2] = MTCARS_HP[i];
            m[i][3] = MTCARS_WT[i];
        }
        IMatrix<Double> data = Linalg.matrix(m);
        List<String> cols = List.of("mpg", "disp", "hp", "wt");
        IPlot p = Plots.of(920, 920)
                .pairplot(data, cols, PairplotDiagonal.KDE)
                .title("Pairplot：mtcars 四变量", "对角 KDE");
        Assertions.assertTrue(generateImage(p, "ext_04_pairplot_kde.png"));
    }

    @Test
    @Order(5)
    @DisplayName("Pairplot：对角直方图")
    void testPairplotHistDiagonal() {
        double[][] m = new double[MTCARS_MPG.length][4];
        for (int i = 0; i < MTCARS_MPG.length; i++) {
            m[i][0] = MTCARS_MPG[i];
            m[i][1] = MTCARS_DISP[i];
            m[i][2] = MTCARS_HP[i];
            m[i][3] = MTCARS_WT[i];
        }
        IPlot p = Plots.of(920, 920)
                .pairplot(Linalg.matrix(m), List.of("mpg", "disp", "hp", "wt"), PairplotDiagonal.HIST)
                .title("Pairplot：mtcars", "对角直方图");
        Assertions.assertTrue(generateImage(p, "ext_05_pairplot_hist.png"));
    }

    @Test
    @Order(6)
    @DisplayName("Jointplot：边际 KDE")
    void testJointplotKde() {
        IPlot p = Plots.of(840, 640)
                .jointplot(Linalg.vector(MTCARS_HP), Linalg.vector(MTCARS_MPG), JointplotMarginal.KDE)
                .title("Joint：油耗 vs 马力", "边际 KDE");
        Assertions.assertTrue(generateImage(p, "ext_06_jointplot_kde.png"));
    }

    @Test
    @Order(7)
    @DisplayName("Jointplot：边际直方图")
    void testJointplotHist() {
        IPlot p = Plots.of(840, 640)
                .jointplot(Linalg.vector(MTCARS_WT), Linalg.vector(MTCARS_MPG), JointplotMarginal.HIST)
                .title("Joint：油耗 vs 车重", "边际直方图");
        Assertions.assertTrue(generateImage(p, "ext_07_jointplot_hist.png"));
    }

    @Test
    @Order(8)
    @DisplayName("Q-Q 图：标准正态与随机样本")
    void testQqplotNormal() {
        double[] z = boxMullerGaussian(120, 42L);
        IPlot p = Plots.of(720, 640)
                .qqplot(Linalg.vector(z))
                .title("Q-Q：N(0,1) 蒙特卡洛样本", "n=120");
        Assertions.assertTrue(generateImage(p, "ext_08_qqplot_normal.png"));
    }

    @Test
    @Order(9)
    @DisplayName("面积图：月降水累计")
    void testAreaCumulativeRain() {
        double[] months = new double[12];
        double[] cum = new double[12];
        double s = 0;
        for (int i = 0; i < 12; i++) {
            months[i] = i + 1;
            s += HAMBURG_MONTHLY_RAIN_MM[i];
            cum[i] = s;
        }
        IPlot p = Plots.of(820, 480)
                .area(Linalg.vector(months), Linalg.vector(cum))
                .title("累计降水量（示例）", "Hamburg 量级 mm/年")
                .xlabel("月份")
                .ylabel("累计 mm");
        Assertions.assertTrue(generateImage(p, "ext_09_area_cum_rain.png"));
    }

    @Test
    @Order(10)
    @DisplayName("阶梯图：政策利率路径")
    void testStepFedRate() {
        double[] t = new double[FED_EFF_RATE_QUARTER.length];
        for (int i = 0; i < t.length; i++) {
            t[i] = i;
        }
        IPlot p = Plots.of(820, 420)
                .step(Linalg.vector(t), Linalg.vector(FED_EFF_RATE_QUARTER))
                .title("联邦基金利率（阶梯示意）", "季度索引")
                .xlabel("Quarter index from 2022Q1")
                .ylabel("Rate %");
        Assertions.assertTrue(generateImage(p, "ext_10_step_fed_rate.png"));
    }

    @Test
    @Order(11)
    @DisplayName("水平条形：GDP 量级（万亿美元近似）")
    void testBarhGdp() {
        List<String> countries = List.of("United States", "China", "Germany", "Japan", "India");
        double[] gdpT = {28.2, 18.5, 4.7, 4.1, 4.0};
        IPlot p = Plots.of(820, 400)
                .barh(countries, Linalg.vector(gdpT))
                .title("GDP 名义值（示意）", "世界银行量级 2023–2024")
                .xlabel("Trillion USD (approx)");
        Assertions.assertTrue(generateImage(p, "ext_11_barh_gdp.png"));
    }

    @Test
    @Order(12)
    @DisplayName("堆叠柱：区域电网可再生结构")
    void testBarStackedRenewables() {
        List<String> q = List.of("2022", "2023", "2024");
        double[][] layers = {
            {28, 31, 34},
            {12, 14, 16},
            {35, 34, 33},
        };
        IPlot p = Plots.of(640, 480)
                .barStacked(q, Linalg.matrix(layers), List.of("Wind", "Solar", "Hydro"))
                .title("可再生发电占比示意", "%，非真实统计仅结构演示")
                .ylabel("Share %");
        Assertions.assertTrue(generateImage(p, "ext_12_barstacked_renew.png"));
    }

    @Test
    @Order(13)
    @DisplayName("误差棒：弹簧劲度系数测定")
    void testErrorbarSpring() {
        double[] x = {1, 2, 3, 4, 5};
        double[] k = {21.2, 20.8, 21.5, 20.9, 21.1};
        double[] err = {0.6, 0.5, 0.7, 0.55, 0.5};
        IPlot p = Plots.of(720, 480)
                .errorbar(Linalg.vector(x), Linalg.vector(k), Linalg.vector(err))
                .title("弹簧 k 测量（示意）", "5 次独立试样 N/m")
                .xlabel("Specimen")
                .ylabel("k (N/m)");
        Assertions.assertTrue(generateImage(p, "ext_13_errorbar_spring.png"));
    }

    @Test
    @Order(14)
    @DisplayName("气泡：人口–能耗–GDP")
    void testScatterBubble() {
        double[] gdpB = {2700, 1850, 470, 420, 380, 315, 210};
        double[] toePerCap = {6.8, 2.4, 3.6, 3.3, 0.65, 4.1, 7.5};
        double[] popM = {335, 1410, 84, 124, 1440, 68, 340};
        IPlot p = Plots.of(820, 560)
                .scatter(Linalg.vector(toePerCap), Linalg.vector(gdpB), Linalg.vector(popM))
                .title("能耗强度 vs GDP（气泡~人口）", "示意尺度")
                .xlabel("Energy use (toe/cap, approx)")
                .ylabel("GDP (100B USD approx)");
        Assertions.assertTrue(generateImage(p, "ext_14_scatter_bubble.png"));
    }

    @Test
    @Order(15)
    @DisplayName("双 Y：温度与湿度周序列")
    void testLineSecondaryY() {
        double[] day = {1, 2, 3, 4, 5, 6, 7};
        double[] tempC = {18.2, 19.1, 22.4, 24.0, 21.3, 20.0, 19.5};
        double[] rh = {62, 58, 52, 48, 55, 60, 64};
        IPlot p = Plots.of(820, 480)
                .y2label("相对湿度 %")
                .lineWithSecondaryY(Linalg.vector(day), Linalg.vector(tempC), Linalg.vector(rh))
                .title("夏季一周气象（示意）", "左：气温 右：相对湿度")
                .xlabel("Day index");
        Assertions.assertTrue(generateImage(p, "ext_15_dual_y_weather.png"));
    }

    @Test
    @Order(16)
    @DisplayName("分面 2×2：四城市气温对比")
    void testSubplotsFourCells() {
        double[] xq = {0, 1, 2, 3};
        double[] hamburg = {3.8, 7.2, 11.5, 15.2};
        double[] munich = {1.2, 5.5, 10.0, 14.0};
        double[] vienna = {2.0, 6.0, 10.5, 14.5};
        double[] zurich = {2.5, 6.5, 10.8, 14.2};
        IPlot p = Plots.of(900, 700);
        p.subplots(2, 2).title("分面：季度均温示意（°C）", "Winter–Spring 索引");
        p.subplot(0, 0).line(Linalg.vector(xq), Linalg.vector(hamburg)).ylabel("Hamburg");
        p.subplot(0, 1).line(Linalg.vector(xq), Linalg.vector(munich)).ylabel("Munich");
        p.subplot(1, 0).line(Linalg.vector(xq), Linalg.vector(vienna)).ylabel("Vienna");
        p.subplot(1, 1).line(Linalg.vector(xq), Linalg.vector(zurich)).ylabel("Zurich");
        Assertions.assertTrue(generateImage(p, "ext_16_subplots_2x2.png"));
    }

    @Test
    @Order(17)
    @DisplayName("对数轴：车重 vs 油耗")
    void testLogAxisMtcars() {
        IPlot p = Plots.of(820, 520)
                .xscale(PlotAxisScale.LOG)
                .yscale(PlotAxisScale.LOG)
                .scatter(Linalg.vector(MTCARS_WT), Linalg.vector(MTCARS_MPG))
                .title("双对数：MPG vs Weight", "须为正值")
                .xlabel("Weight (1000 lb, log)")
                .ylabel("MPG (log)");
        Assertions.assertTrue(generateImage(p, "ext_17_loglog_scatter.png"));
    }

    @Test
    @Order(18)
    @DisplayName("主题系统：applyTheme + 暗色")
    void testThemeSystemDark() {
        IPlot p = Plots.of(820, 500)
                .enableThemeSystem(true)
                .applyTheme("dark")
                .line(Linalg.vector(MTCARS_DISP), Linalg.vector(MTCARS_MPG))
                .title("主题 dark", "排量 vs 油耗");
        Assertions.assertTrue(generateImage(p, "ext_18_theme_dark_apply.png"));
    }

    @Test
    @Order(19)
    @DisplayName("调色板：setPalette seaborn")
    void testSetPaletteSeaborn() {
        IPlot p = Plots.of(820, 500)
                .setPalette("seaborn")
                .line(Linalg.vector(MTCARS_HP), Linalg.vector(MTCARS_MPG))
                .title("palette=seaborn", "马力 vs 油耗");
        Assertions.assertTrue(generateImage(p, "ext_19_palette_seaborn.png"));
    }

    @Test
    @Order(20)
    @DisplayName("样式字符串：虚线/点划/标记")
    void testStyleStringLineScatter() {
        double[] x = {0, 1, 2, 3, 4, 5};
        double[] y1 = {2, 3, 2.5, 4, 3.5, 5};
        double[] y2 = {1, 2, 2.2, 3, 4, 4.2};
        IPlot p = Plots.of(820, 500);
        p.enableStyleSystem(true)
                .line(Linalg.vector(x), Linalg.vector(y1), "r--")
                .scatter(Linalg.vector(x), Linalg.vector(y2), "b:^")
                .title("StyleExpression", "r-- 线 + b:^ 散点")
                .xlabel("x")
                .ylabel("y");
        Assertions.assertTrue(generateImage(p, "ext_20_style_strings.png"));
    }

    @Test
    @Order(21)
    @DisplayName("PlotStyle 对象：线宽与颜色")
    void testPlotStyleObject() {
        double[] x = new double[10];
        double[] y = new double[10];
        for (int i = 0; i < 10; i++) {
            x[i] = i;
            y[i] = Math.sin(i * 0.5) * 2 + 5;
        }
        PlotStyle st = PlotStyle.defaultStyle()
                .color("#7c3aed")
                .lineStyle("dashed")
                .lineWidth(2.5);
        IPlot p = Plots.of(820, 460)
                .line(Linalg.vector(x), Linalg.vector(y), st)
                .title("显式 PlotStyle", "紫虚线 2.5px");
        Assertions.assertTrue(generateImage(p, "ext_21_plotstyle_object.png"));
    }

    @Test
    @Order(22)
    @DisplayName("自定义渐变主题注册")
    void testGradientTheme() {
        IPlot p = Plots.of(820, 480)
                .createGradientTheme("jfx_grad_test", "#0f172a", "#38bdf8", "#020617")
                .theme("jfx_grad_test")
                .line(Linalg.vector(MTCARS_WT), Linalg.vector(MTCARS_MPG))
                .title("渐变注册主题", "车重 vs 油耗");
        Assertions.assertTrue(generateImage(p, "ext_22_gradient_theme.png"));
    }

    @Test
    @Order(23)
    @DisplayName("直方图：指定 bins")
    void testHistBins() {
        IPlot p = Plots.of(780, 480)
                .hist(Linalg.vector(IRIS_SETOSA_SEPAL_LENGTH), false, PlotStyle.defaultStyle(), 14)
                .title("直方图 bins=14", "Setosa 萼片长");
        Assertions.assertTrue(generateImage(p, "ext_23_hist_bins14.png"));
    }

    @Test
    @Order(24)
    @DisplayName("分组线：区域 hue + 促销 styleGroup")
    void testLineHueStyleGroup() {
        double[] t = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        double[] rev = {4.2, 4.5, 4.1, 4.8, 5.2, 5.0, 5.4, 5.1, 5.6, 5.9, 6.0, 6.2};
        List<String> hue = List.of(
                "North", "North", "North", "North", "North", "North",
                "South", "South", "South", "South", "South", "South");
        List<String> styleGroup = List.of(
                "off", "off", "promo", "off", "off", "promo",
                "off", "promo", "off", "off", "promo", "off");
        IPlot p = Plots.of(860, 480)
                .line(Linalg.vector(t), Linalg.vector(rev), hue, styleGroup)
                .title("分区营收（百万/月·示意）", "hue=区域 styleGroup=是否促销")
                .xlabel("Month")
                .ylabel("Revenue");
        Assertions.assertTrue(generateImage(p, "ext_24_line_hue_stylegroup.png"));
    }

    @Test
    @Order(25)
    @DisplayName("单色主题注册 + 散点")
    void testMonochromeTheme() {
        IPlot p = Plots.of(780, 480)
                .createMonochromeTheme("jfx_mono_test", "#1d4ed8", "#f1f5f9")
                .theme("jfx_mono_test")
                .scatter(Linalg.vector(MTCARS_DISP), Linalg.vector(MTCARS_MPG))
                .title("单色注册主题", "排量 vs 油耗");
        Assertions.assertTrue(generateImage(p, "ext_25_monochrome_theme.png"));
    }

    @Test
    @Order(26)
    @DisplayName("小提琴图（内嵌箱线，Iris setosa 萼片长）")
    void testViolinplotEmbeddedBox() {
        IPlot p = Plots.of(760, 480)
                .violinplot(Linalg.vector(IRIS_SETOSA_SEPAL_LENGTH))
                .title("Violin (inner box)", "Setosa sepal length cm");
        Assertions.assertTrue(generateImage(p, "ext_26_violin_innerbox.png"));
    }

    @AfterAll
    static void tearDown() {
        printSummary("Plots 扩展功能覆盖");
    }
}
