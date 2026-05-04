package com.yishape.lab.math.plot.javafx.j3d;

import com.yishape.lab.math.plot.javafx.JavaFxStyleApplier;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Group;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Rectangle;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.scene.text.Text;

import java.util.ArrayList;
import java.util.List;

/**
 * 3D图表图例渲染器，在2D叠加层上显示图例。
 * <p>
 * 由于JavaFX 3D场景中的Text节点难以直接阅读（受3D透视影响），
 * 图例作为独立的2D UI元素叠加在3D场景之上。
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public final class JavaFx3dLegend {

    private JavaFx3dLegend() {
        // 工具类
    }

    /**
     * 图例项定义
     */
    public static class LegendItem {
        private final String name;
        private final Color color;
        private final LegendSymbolType symbolType;
        private final double symbolSize;

        public LegendItem(String name, Color color) {
            this(name, color, LegendSymbolType.CIRCLE, 10);
        }

        public LegendItem(String name, Color color, LegendSymbolType symbolType, double symbolSize) {
            this.name = name;
            this.color = color;
            this.symbolType = symbolType;
            this.symbolSize = symbolSize;
        }

        public String getName() { return name; }
        public Color getColor() { return color; }
        public LegendSymbolType getSymbolType() { return symbolType; }
        public double getSymbolSize() { return symbolSize; }
    }

    /**
     * 图例符号类型
     */
    public enum LegendSymbolType {
        CIRCLE,      // 圆形（散点）
        RECTANGLE,   // 矩形（柱状）
        LINE,        // 线段（线图）
        TRIANGLE     // 三角形（特殊）
    }

    /**
     * 创建水平图例栏（适合顶部/底部放置）。
     *
     * @param items 图例项列表
     * @param theme 主题管理器
     * @return 图例容器
     */
    public static HBox createHorizontalLegend(List<LegendItem> items, JavaFxThemeManager theme) {
        HBox legendBox = new HBox(15);
        legendBox.setAlignment(Pos.CENTER);
        legendBox.setPadding(new Insets(8, 15, 8, 15));
        legendBox.setStyle("-fx-background-color: rgba(255,255,255,0.85); -fx-background-radius: 8;");

        Color textColor = theme.getTextColor();

        for (LegendItem item : items) {
            HBox itemBox = new HBox(8);
            itemBox.setAlignment(Pos.CENTER);

            // 符号
            javafx.scene.Node symbol = createSymbol(item, textColor);

            // 文本
            Text label = new Text(item.getName());
            label.setFont(Font.font(theme.getLabelFont().getFamily(), FontWeight.NORMAL, 12));
            label.setFill(textColor);

            itemBox.getChildren().addAll(symbol, label);
            legendBox.getChildren().add(itemBox);
        }

        return legendBox;
    }

    /**
     * 创建垂直图例栏（适合左侧/右侧放置）。
     *
     * @param items 图例项列表
     * @param theme 主题管理器
     * @return 图例容器
     */
    public static VBox createVerticalLegend(List<LegendItem> items, JavaFxThemeManager theme) {
        VBox legendBox = new VBox(10);
        legendBox.setAlignment(Pos.CENTER_LEFT);
        legendBox.setPadding(new Insets(12, 15, 12, 15));
        legendBox.setStyle("-fx-background-color: rgba(255,255,255,0.85); -fx-background-radius: 8;");

        Color textColor = theme.getTextColor();

        for (LegendItem item : items) {
            HBox itemBox = new HBox(10);
            itemBox.setAlignment(Pos.CENTER_LEFT);

            // 符号
            javafx.scene.Node symbol = createSymbol(item, textColor);

            // 文本
            Text label = new Text(item.getName());
            label.setFont(Font.font(theme.getLabelFont().getFamily(), FontWeight.NORMAL, 12));
            label.setFill(textColor);

            itemBox.getChildren().addAll(symbol, label);
            legendBox.getChildren().add(itemBox);
        }

        return legendBox;
    }

    /**
     * 创建浮动图例面板（可放置在任意位置）。
     *
     * @param items 图例项列表
     * @param theme 主题管理器
     * @param orientation 方向："horizontal" 或 "vertical"
     * @return 图例容器
     */
    public static javafx.scene.Node createFloatingLegend(
            List<LegendItem> items,
            JavaFxThemeManager theme,
            String orientation) {

        javafx.scene.Node legend = "vertical".equalsIgnoreCase(orientation)
                ? createVerticalLegend(items, theme)
                : createHorizontalLegend(items, theme);

        // 添加阴影效果
        javafx.scene.effect.DropShadow shadow = new javafx.scene.effect.DropShadow();
        shadow.setRadius(5);
        shadow.setOffsetX(2);
        shadow.setOffsetY(2);
        shadow.setColor(Color.color(0, 0, 0, 0.2));
        legend.setEffect(shadow);

        return legend;
    }

    /**
     * 根据符号类型创建对应图形。
     */
    private static javafx.scene.Node createSymbol(LegendItem item, Color defaultColor) {
        Color color = item.getColor() != null ? item.getColor() : defaultColor;
        double size = item.getSymbolSize();

        return switch (item.getSymbolType()) {
            case CIRCLE -> {
                Circle c = new Circle(size / 2);
                c.setFill(color);
                yield c;
            }
            case RECTANGLE -> {
                Rectangle r = new Rectangle(size, size * 0.7);
                r.setFill(color);
                r.setArcWidth(3);
                r.setArcHeight(3);
                yield r;
            }
            case LINE -> {
                javafx.scene.shape.Line line = new javafx.scene.shape.Line(0, size/2, size * 1.5, size/2);
                line.setStroke(color);
                line.setStrokeWidth(2.5);
                Pane p = new Pane(line);
                p.setPrefSize(size * 1.5, size);
                yield p;
            }
            case TRIANGLE -> {
                javafx.scene.shape.Polygon tri = new javafx.scene.shape.Polygon();
                tri.getPoints().addAll(
                        size / 2, 0.0,
                        0.0, size,
                        size, size
                );
                tri.setFill(color);
                yield tri;
            }
        };
    }

    /**
     * 构建默认图例项列表（根据系列数量和调色板）。
     *
     * @param seriesNames 系列名称列表
     * @param palette 颜色数组
     * @param symbolType 符号类型
     * @return 图例项列表
     */
    public static List<LegendItem> buildDefaultItems(
            List<String> seriesNames,
            String[] palette,
            LegendSymbolType symbolType) {

        List<LegendItem> items = new ArrayList<>();
        if (seriesNames == null || seriesNames.isEmpty()) {
            return items;
        }

        String[] colors = palette != null && palette.length > 0
                ? palette
                : new String[]{"#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de"};

        for (int i = 0; i < seriesNames.size(); i++) {
            String name = seriesNames.get(i);
            if (name == null || name.isEmpty()) continue;

            Color color = JavaFxStyleApplier.parseColor(colors[i % colors.length]);
            items.add(new LegendItem(name, color, symbolType, 10));
        }

        return items;
    }

    /**
     * 根据hue分组自动构建图例项。
     *
     * @param hueGroups hue分组（去重后的分组名称）
     * @param palette 颜色调色板
     * @return 图例项列表
     */
    public static List<LegendItem> buildItemsFromHueGroups(
            List<String> hueGroups,
            String[] palette) {

        return buildDefaultItems(hueGroups, palette, LegendSymbolType.CIRCLE);
    }
}
