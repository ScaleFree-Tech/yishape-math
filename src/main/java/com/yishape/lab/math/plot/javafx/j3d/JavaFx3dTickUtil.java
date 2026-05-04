package com.yishape.lab.math.plot.javafx.j3d;

import com.yishape.lab.math.plot.AxisTicks;
import com.yishape.lab.math.linalg.IVector;

import java.util.ArrayList;
import java.util.List;

/**
 * 三维场景刻度：优先使用 {@link AxisTicks}，否则生成 human-readable 步长。
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
final class JavaFx3dTickUtil {

    private JavaFx3dTickUtil() {
    }

    static int fallbackDivisions(AxisTicks ticks, int defaultDivisions) {
        if (ticks != null && ticks.hasTickValues()) {
            return Math.max(1, ticks.getTickValues().length() - 1);
        }
        return defaultDivisions;
    }

    static List<Tick> ticksForAxis(AxisTicks cfg, double lo, double hi, int targetSteps) {
        List<Tick> out = new ArrayList<>();
        if (cfg != null && cfg.hasTickValues()) {
            IVector<?> v = cfg.getTickValues();
            List<String> labels = cfg.getTickLabels();
            int n = v.length();
            for (int i = 0; i < n; i++) {
                double val = v.get(i).doubleValue();
                String lab = labels != null && i < labels.size() && labels.get(i) != null && !labels.get(i).isEmpty()
                        ? labels.get(i)
                        : null;
                out.add(new Tick(val, lab));
            }
            return out;
        }
        if (!Double.isFinite(lo) || !Double.isFinite(hi) || hi < lo) {
            return out;
        }
        if (hi - lo <= 1e-15) {
            out.add(new Tick(lo, null));
            return out;
        }
        int steps = Math.max(3, Math.min(12, targetSteps));
        double span = hi - lo;
        double raw = span / steps;
        double step = niceStep(raw);
        double start = Math.ceil(lo / step) * step;
        if (start > hi + step * 1e-9) {
            start = lo;
        }
        for (double x = start; x <= hi + step * 1e-9; x += step) {
            if (x >= lo - 1e-9 && x <= hi + 1e-9) {
                out.add(new Tick(JavaFx3dFxUtil.clamp(x, lo, hi), null));
            }
        }
        if (out.isEmpty() || out.get(out.size() - 1).value < hi - 1e-9 * span) {
            out.add(new Tick(hi, null));
        }
        if (!out.isEmpty() && out.get(0).value > lo + 1e-9 * span) {
            out.add(0, new Tick(lo, null));
        }
        return out;
    }

    private static double niceStep(double raw) {
        if (raw <= 0 || !Double.isFinite(raw)) {
            return 1;
        }
        double exp = Math.floor(Math.log10(raw));
        double f = raw / Math.pow(10, exp);
        double nf;
        if (f <= 1) {
            nf = 1;
        } else if (f <= 2) {
            nf = 2;
        } else if (f <= 5) {
            nf = 5;
        } else {
            nf = 10;
        }
        return nf * Math.pow(10, exp);
    }

    record Tick(double value, String labelOverride) {
    }
}
