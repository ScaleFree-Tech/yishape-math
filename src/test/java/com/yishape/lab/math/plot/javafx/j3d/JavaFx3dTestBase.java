package com.yishape.lab.math.plot.javafx.j3d;

import javafx.application.Platform;
import org.junit.jupiter.api.Assertions;

import java.io.File;
import java.util.concurrent.TimeUnit;

/**
 * JavaFX 3D 图测试基类：初始化 Toolkit、创建 {@code images_javafx/3d} 目录，
 * 并通过 {@link JavaFx3dPlot#writeSnapshotPng(File)} 无窗口导出 PNG。
 *
 * @author lteb2
 */
public abstract class JavaFx3dTestBase {

    protected static final String OUTPUT_DIR = "images_javafx" + File.separator + "3d";

    protected static final int DEFAULT_W = 720;
    protected static final int DEFAULT_H = 540;

    private static boolean initialized;

    protected static void initialize3d() {
        if (initialized) {
            return;
        }
        File dir = new File(OUTPUT_DIR);
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IllegalStateException("无法创建输出目录: " + dir.getAbsolutePath());
        }
        try {
            Platform.startup(() -> { });
        } catch (IllegalStateException ignored) {
            // already started
        }
        initialized = true;
    }

    /**
     * 写入 PNG 并断言文件非空。
     */
    protected static void assertSnapshot(JavaFx3dPlot plot, String filename) {
        File out = new File(OUTPUT_DIR, filename);
        long t0 = System.nanoTime();
        plot.writeSnapshotPng(out);
        long ms = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - t0);
        Assertions.assertTrue(out.exists(), "应生成文件: " + out.getAbsolutePath());
        Assertions.assertTrue(out.length() > 256, "PNG 过小可能失败: " + filename + " (" + out.length() + " B)");
        System.out.println("  ✓ " + filename + " (" + out.length() / 1024 + " KB, " + ms + " ms)");
    }
}
