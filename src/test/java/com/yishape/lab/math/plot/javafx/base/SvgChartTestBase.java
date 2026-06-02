package com.yishape.lab.math.plot.javafx.base;

import com.yishape.lab.math.plot.svg.SvgPlot;
import org.junit.jupiter.api.*;

import java.io.File;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * SVG图表测试基类：直接使用 {@link SvgPlot} 生成矢量 SVG 文件（无需 JavaFX）。
 */
public abstract class SvgChartTestBase {

    protected static final String OUTPUT_DIR = "images_javafx_svg";
    protected static int testCount = 0;
    protected static int passCount = 0;
    protected static boolean initialized = false;

    protected static void initialize() {
        if (initialized) return;
        File outputDir = new File(OUTPUT_DIR);
        if (!outputDir.exists()) {
            outputDir.mkdirs();
        }
        initialized = true;
    }

    protected boolean generateImage(SvgPlot plot, String filename) {
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

    protected static void printSummary(String testName) {
        System.out.println("\n=== " + testName + " SVG测试结果 ===");
        System.out.println("测试数: " + testCount + ", 成功: " + passCount + ", 失败: " + (testCount - passCount));
    }

    protected static void resetCounters() {
        testCount = 0;
        passCount = 0;
    }
}
