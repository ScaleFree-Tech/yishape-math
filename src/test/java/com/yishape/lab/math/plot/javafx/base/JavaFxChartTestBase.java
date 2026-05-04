package com.yishape.lab.math.plot.javafx.base;

import com.yishape.lab.math.plot.IPlot;
import com.yishape.lab.math.plot.PlotProvider;
import com.yishape.lab.math.plot.Plots;
import com.yishape.lab.math.plot.javafx.JavaFxPlot;
import javafx.application.Platform;

import java.io.File;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * JavaFX 图表测试基类：通过 {@link Plots#setProvider}（在 {@link #initialize()} 中固定为 JavaFX）
 * 与 {@link Plots#of(int, int)} 等链式构造 {@link IPlot}，再快照 {@link JavaFxPlot#getCanvas()} 输出 PNG。
 *
 * @author lteb2
 */
public abstract class JavaFxChartTestBase {
    
    protected static final String OUTPUT_DIR = "images_javafx_svg";
    protected static int testCount = 0;
    protected static int passCount = 0;
    protected static boolean initialized = false;
    
    /**
     * 初始化JavaFX和输出目录
     */
    protected static void initialize() {
        if (initialized) return;
        
        // 创建输出目录
        File outputDir = new File(OUTPUT_DIR);
        if (!outputDir.exists()) {
            outputDir.mkdirs();
        }

        Plots.setProvider(PlotProvider.JavaFx);

        // 初始化JavaFX
        try {
            Platform.startup(() -> {});
        } catch (IllegalStateException e) {
            // JavaFX已初始化
        }
        
        initialized = true;
    }
    
    /**
     * 根据 {@link IPlot} 具体类型自动选择渲染方式：
     * <ul>
     *   <li>{@link JavaFxPlot} → JavaFX Canvas 快照输出 PNG/SVG</li>
     *   <li>{@link SvgPlot} → 直接生成矢量 SVG 文件（无需 JavaFX）</li>
     * </ul>
     */
    protected boolean generateImage(IPlot plot, String filename) {
        if (plot instanceof JavaFxPlot jfx) {
            return generateImage(jfx, filename);
        }
        if (plot instanceof com.yishape.lab.math.plot.svg.SvgPlot svg) {
            return generateImageSvg(svg, filename);
        }
        throw new IllegalStateException(
                "不支持的 IPlot 实现: " + (plot == null ? "null" : plot.getClass().getName()));
    }

    /**
     * 生成矢量SVG文件（用于 {@link SvgPlot}，无需 JavaFX）。
     */
    protected boolean generateImageSvg(com.yishape.lab.math.plot.svg.SvgPlot plot, String filename) {
        testCount++;
        try {
            File outputFile = new File(OUTPUT_DIR, filename);
            plot.saveAsSvg(outputFile.getAbsolutePath());
            if (outputFile.exists()) {
                System.out.println("  ✓ 已生成: " + filename + " (" + outputFile.length() / 1024 + " KB)");
                passCount++;
                return true;
            } else {
                System.out.println("  ✗ 文件未创建: " + filename);
                return false;
            }
        } catch (Exception e) {
            System.out.println("  ✗ 失败: " + filename + " - " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    /**
     * 生成 JavaFX 图表快照：按扩展名选择格式——{@code .png} 为真 PNG，
     * {@code .svg} 为「SVG 文档内嵌 PNG」（与 {@link JavaFxPlot#saveAsSvg} 一致）。
     */
    protected boolean generateImage(JavaFxPlot plot, String filename) {
        testCount++;
        final boolean[] success = {false};
        CountDownLatch latch = new CountDownLatch(1);

        Platform.runLater(() -> {
            try {
                plot.render();
                File outputFile = new File(OUTPUT_DIR, filename);
                String fn = filename.trim();
                if (fn.length() >= 4 && fn.regionMatches(true, fn.length() - 4, ".png", 0, 4)) {
                    plot.saveAsPng(outputFile.getAbsolutePath());
                } else {
                    plot.saveAsSvg(outputFile.getAbsolutePath());
                }

                if (outputFile.exists()) {
                    System.out.println("  ✓ 已生成: " + filename + " (" + outputFile.length() / 1024 + " KB)");
                    success[0] = true;
                    passCount++;
                } else {
                    System.out.println("  ✗ 文件未创建: " + filename);
                }
            } catch (Exception e) {
                System.out.println("  ✗ 失败: " + filename + " - " + e.getMessage());
                e.printStackTrace();
            } finally {
                latch.countDown();
            }
        });

        try {
            boolean completed = latch.await(5, TimeUnit.SECONDS);
            if (!completed) {
                System.out.println("  ✗ 超时: " + filename);
            }
        } catch (InterruptedException e) {
            System.out.println("  ✗ 中断: " + filename);
        }

        return success[0];
    }
    
    /**
     * 打印测试结果摘要
     */
    protected static void printSummary(String testName) {
        System.out.println("\n=== " + testName + " 测试结果 ===");
        System.out.println("测试数: " + testCount + ", 成功: " + passCount + ", 失败: " + (testCount - passCount));
    }
    
    /**
     * 重置计数器
     */
    protected static void resetCounters() {
        testCount = 0;
        passCount = 0;
    }
}
