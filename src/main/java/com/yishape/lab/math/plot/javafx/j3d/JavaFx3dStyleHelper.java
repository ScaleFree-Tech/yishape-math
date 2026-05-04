package com.yishape.lab.math.plot.javafx.j3d;

import com.yishape.lab.math.plot.PlotStyle;
import com.yishape.lab.math.plot.StyleExpression;
import com.yishape.lab.math.plot.javafx.JavaFxStyleApplier;
import javafx.scene.paint.Color;

/**
 * {@link JavaFx3dPlot} 的样式合并与颜色解析。
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public final class JavaFx3dStyleHelper {

    private JavaFx3dStyleHelper() {
    }

    public static PlotStyle effectiveStyle(PlotStyle defaultSeries, PlotStyle overlay, String styleString) {
        PlotStyle base = defaultSeries != null ? new PlotStyle(defaultSeries) : new PlotStyle();
        PlotStyle o = overlay;
        if (o == null && styleString != null && !styleString.isEmpty()) {
            o = StyleExpression.parse(styleString);
        }
        if (o == null) {
            return base;
        }
        PlotStyle m = new PlotStyle(base);
        if (o.getColor() != null) {
            m.setColor(o.getColor());
        }
        if (o.getFaceColor() != null) {
            m.setFaceColor(o.getFaceColor());
        }
        if (o.getMarkerSize() > 0) {
            m.setMarkerSize(o.getMarkerSize());
        }
        if (o.getLineWidth() > 0) {
            m.setLineWidth(o.getLineWidth());
        }
        if (o.getAlpha() >= 0 && o.getAlpha() <= 1) {
            m.setAlpha(o.getAlpha());
        }
        return m;
    }

    public static Color fxColor(PlotStyle st, boolean fill) {
        String spec = fill ? st.getFaceColor() : st.getColor();
        if (spec == null || spec.isEmpty()) {
            spec = st.getColor();
        }
        return JavaFxStyleApplier.parseColor(spec != null ? spec : "#5470c6");
    }
}
