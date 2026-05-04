package com.yishape.lab.math.plot.javafx.j3d;

import com.yishape.lab.math.plot.PlotException;
import javafx.application.Platform;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;

/**
 * JavaFX 工具：工具包初始化与 FX 线程同步执行。
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
final class JavaFx3dFxUtil {

    private JavaFx3dFxUtil() {
    }

    static void ensureFxToolkit() {
        try {
            Platform.startup(() -> { });
        } catch (IllegalStateException ignored) {
            // already started
        }
    }

    static void runOnFxThreadSync(Runnable action) {
        ensureFxToolkit();
        if (Platform.isFxApplicationThread()) {
            action.run();
            return;
        }
        CountDownLatch done = new CountDownLatch(1);
        AtomicReference<Throwable> err = new AtomicReference<>();
        Platform.runLater(() -> {
            try {
                action.run();
            } catch (Throwable t) {
                err.set(t);
            } finally {
                done.countDown();
            }
        });
        try {
            done.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new PlotException("JavaFX 3D 操作被中断", e);
        }
        if (err.get() != null) {
            throw new PlotException("JavaFX 3D: " + err.get().getMessage(), err.get());
        }
    }

    static String jsonEscape(String s) {
        if (s == null) {
            return "";
        }
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
