package com.yishape.lab.math.plot.javafx;

import javafx.animation.Interpolator;
import javafx.animation.KeyFrame;
import javafx.animation.KeyValue;
import javafx.animation.Timeline;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.geometry.Point2D;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Tooltip;
import javafx.scene.input.MouseEvent;
import javafx.scene.input.ScrollEvent;
import javafx.scene.transform.Affine;
import javafx.util.Duration;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * JavaFX交互处理器
 * 处理鼠标交互、提示框、动画效果
 *
 * @author lteb2
 */
public class JavaFxInteractionHandler {

    private Canvas canvas;

    private List<DataPoint> dataPoints = new ArrayList<>();
    private final Map<String, Consumer<DataPoint>> clickHandlers = new HashMap<>();

    private double scaleX = 1.0;
    private double scaleY = 1.0;
    private double translateX = 0.0;
    private double translateY = 0.0;
    private boolean enableZoom = true;
    private boolean enablePan = true;

    private Timeline animationTimeline;
    private final SimpleDoubleProperty animationProgressProperty = new SimpleDoubleProperty(0);
    private double animationProgress = 0.0;
    private int animationDuration = 1000;
    private boolean animationEnabled = true;

    private Tooltip tooltip;
    private boolean tooltipEnabled = true;

    /** 与 dataPoints 中对象解耦，便于每帧重建命中列表后恢复悬停 */
    private DataPoint hoveredPoint = null;
    private String hoverSeriesName;
    private int hoverSeriesIndex = -1;
    private int hoverPointIndex = -1;

    private Runnable repaintCallback;

    private double dragStartX;
    private double dragStartY;
    private double dragStartTranslateX;
    private double dragStartTranslateY;

    /**
     * 数据点类
     */
    public static class DataPoint {
        public double x;
        public double y;
        public double dataX;
        public double dataY;
        public String seriesName;
        public String label;
        public int seriesIndex;
        public int pointIndex;
        public Object extraData;

        public DataPoint(double x, double y, double dataX, double dataY,
                         String seriesName, int seriesIndex, int pointIndex) {
            this.x = x;
            this.y = y;
            this.dataX = dataX;
            this.dataY = dataY;
            this.seriesName = seriesName;
            this.seriesIndex = seriesIndex;
            this.pointIndex = pointIndex;
        }

        @Override
        public String toString() {
            return String.format("%s: (%.2f, %.2f)", seriesName, dataX, dataY);
        }
    }

    public JavaFxInteractionHandler(Canvas canvas, JavaFxChartRenderer.ChartConfig config) {
        this.canvas = canvas;
        this.animationDuration = config.animationDuration;
        this.animationEnabled = config.enableAnimation;

        animationProgressProperty.addListener((obs, o, n) -> {
            animationProgress = n.doubleValue();
            requestRepaint();
        });

        setupEventHandlers();
        setupTooltip();
    }

    public void setRepaintCallback(Runnable repaintCallback) {
        this.repaintCallback = repaintCallback;
    }

    public void updateCanvas(Canvas canvas, JavaFxChartRenderer.ChartConfig config) {
        this.canvas = canvas;
        this.animationDuration = config.animationDuration;
        this.animationEnabled = config.enableAnimation;
    }

    public void updateConfig(JavaFxChartRenderer.ChartConfig config) {
        this.animationDuration = config.animationDuration;
        this.animationEnabled = config.enableAnimation;
    }

    private void setupEventHandlers() {
        canvas.addEventHandler(MouseEvent.MOUSE_MOVED, this::handleMouseMoved);
        canvas.addEventHandler(MouseEvent.MOUSE_CLICKED, this::handleMouseClicked);
        canvas.addEventHandler(MouseEvent.MOUSE_DRAGGED, this::handleMouseDragged);
        canvas.addEventHandler(MouseEvent.MOUSE_PRESSED, this::handleMousePressed);
        canvas.addEventHandler(ScrollEvent.SCROLL, this::handleScroll);
    }

    private void setupTooltip() {
        tooltip = new Tooltip();
        tooltip.setAutoHide(true);
        tooltip.setHideDelay(Duration.millis(100));
        tooltip.setShowDelay(Duration.millis(100));
    }

    private void handleMousePressed(MouseEvent event) {
        dragStartX = event.getX();
        dragStartY = event.getY();
        dragStartTranslateX = translateX;
        dragStartTranslateY = translateY;
    }

    private void handleMouseMoved(MouseEvent event) {
        if (!tooltipEnabled) {
            return;
        }

        Point2D local = screenToLogical(event.getX(), event.getY());
        DataPoint point = findNearestPoint(local.getX(), local.getY());

        if (point != null) {
            boolean changed = !sameHover(point);
            if (changed) {
                applyHover(point);
            }
            showTooltip(point, event.getScreenX(), event.getScreenY());
            if (changed) {
                requestRepaint();
            }
        } else {
            if (hoverSeriesIndex >= 0) {
                clearHover();
                hideTooltip();
                requestRepaint();
            }
        }
    }

    private boolean sameHover(DataPoint p) {
        return hoverSeriesIndex == p.seriesIndex
            && hoverPointIndex == p.pointIndex
            && Objects.equals(hoverSeriesName, p.seriesName);
    }

    private void applyHover(DataPoint p) {
        hoverSeriesName = p.seriesName;
        hoverSeriesIndex = p.seriesIndex;
        hoverPointIndex = p.pointIndex;
        hoveredPoint = p;
    }

    private void clearHover() {
        hoverSeriesName = null;
        hoverSeriesIndex = -1;
        hoverPointIndex = -1;
        hoveredPoint = null;
    }

    /**
     * 重建 {@link #dataPoints} 后根据悬停身份重新绑定 {@link #hoveredPoint}（坐标随平移缩放更新）
     */
    public void syncHoverAfterRebuild() {
        if (hoverSeriesIndex < 0) {
            hoveredPoint = null;
            return;
        }
        for (DataPoint p : dataPoints) {
            if (p.seriesIndex == hoverSeriesIndex
                && p.pointIndex == hoverPointIndex
                && Objects.equals(p.seriesName, hoverSeriesName)) {
                hoveredPoint = p;
                return;
            }
        }
        clearHover();
    }

    private void handleMouseClicked(MouseEvent event) {
        Point2D local = screenToLogical(event.getX(), event.getY());
        DataPoint point = findNearestPoint(local.getX(), local.getY());
        if (point != null) {
            Consumer<DataPoint> handler = clickHandlers.get("default");
            if (handler != null) {
                handler.accept(point);
            }
        }
    }

    private void handleMouseDragged(MouseEvent event) {
        if (!enablePan) {
            return;
        }

        double dx = event.getX() - dragStartX;
        double dy = event.getY() - dragStartY;

        translateX = dragStartTranslateX + dx;
        translateY = dragStartTranslateY + dy;

        requestRepaint();
    }

    private void handleScroll(ScrollEvent event) {
        if (!enableZoom) {
            return;
        }

        double zoomFactor = event.getDeltaY() > 0 ? 1.1 : 0.9;

        double mouseX = event.getX();
        double mouseY = event.getY();

        translateX = mouseX - (mouseX - translateX) * zoomFactor;
        translateY = mouseY - (mouseY - translateY) * zoomFactor;

        scaleX *= zoomFactor;
        scaleY *= zoomFactor;

        scaleX = Math.max(0.1, Math.min(10.0, scaleX));
        scaleY = Math.max(0.1, Math.min(10.0, scaleY));

        requestRepaint();
    }

    /**
     * 与 {@link #applyTransform} 使用相同的 Affine，将屏幕坐标转换到绘图逻辑坐标
     */
    public Point2D screenToLogical(double screenX, double screenY) {
        Affine a = buildViewAffine();
        try {
            return a.inverseTransform(screenX, screenY);
        } catch (Exception e) {
            return new Point2D(screenX, screenY);
        }
    }

    private Affine buildViewAffine() {
        Affine a = new Affine();
        a.appendTranslation(translateX, translateY);
        a.appendScale(scaleX, scaleY);
        return a;
    }

    private DataPoint findNearestPoint(double logicalX, double logicalY) {
        DataPoint nearest = null;
        double minDistance = Double.MAX_VALUE;
        double threshold = 20;

        for (DataPoint point : dataPoints) {
            double dx = point.x - logicalX;
            double dy = point.y - logicalY;
            double distance = Math.sqrt(dx * dx + dy * dy);

            if (distance < threshold && distance < minDistance) {
                minDistance = distance;
                nearest = point;
            }
        }

        return nearest;
    }

    private void showTooltip(DataPoint point, double screenX, double screenY) {
        String text = String.format("%s\nX: %.2f\nY: %.2f",
            point.seriesName, point.dataX, point.dataY);

        if (point.label != null && !point.label.isEmpty()) {
            text += "\n" + point.label;
        }

        tooltip.setText(text);
        tooltip.show(canvas, screenX + 15, screenY + 10);
    }

    private void hideTooltip() {
        tooltip.hide();
    }

    private void requestRepaint() {
        if (repaintCallback != null) {
            repaintCallback.run();
        }
    }

    public void addDataPoint(DataPoint point) {
        dataPoints.add(point);
    }

    public void clearDataPoints() {
        dataPoints.clear();
    }

    public void setClickHandler(String name, Consumer<DataPoint> handler) {
        clickHandlers.put(name, handler);
    }

    public void startAnimation(Runnable onFinished) {
        stopAnimation();
        if (!animationEnabled) {
            animationProgressProperty.set(1.0);
            animationProgress = 1.0;
            if (onFinished != null) {
                onFinished.run();
            }
            return;
        }

        animationTimeline = new Timeline(
            new KeyFrame(Duration.ZERO, new KeyValue(animationProgressProperty, 0, Interpolator.LINEAR)),
            new KeyFrame(Duration.millis(animationDuration), new KeyValue(animationProgressProperty, 1, Interpolator.EASE_OUT))
        );
        animationTimeline.setOnFinished(e -> {
            animationProgressProperty.set(1.0);
            animationProgress = 1.0;
            if (onFinished != null) {
                onFinished.run();
            }
        });
        animationTimeline.play();
    }

    public void stopAnimation() {
        if (animationTimeline != null) {
            animationTimeline.stop();
        }
    }

    public double getAnimationProgress() {
        return animationProgress;
    }

    public void setAnimationProgress(double progress) {
        this.animationProgress = progress;
        animationProgressProperty.set(progress);
        requestRepaint();
    }

    public DataPoint getHoveredPoint() {
        return hoveredPoint;
    }

    public void applyTransform(GraphicsContext graphicsContext) {
        graphicsContext.translate(translateX, translateY);
        graphicsContext.scale(scaleX, scaleY);
    }

    public void resetTransform() {
        scaleX = 1.0;
        scaleY = 1.0;
        translateX = 0.0;
        translateY = 0.0;
    }

    public void setZoomEnabled(boolean enabled) {
        this.enableZoom = enabled;
    }

    public void setPanEnabled(boolean enabled) {
        this.enablePan = enabled;
    }

    public void setTooltipEnabled(boolean enabled) {
        this.tooltipEnabled = enabled;
    }

    public void setAnimationEnabled(boolean enabled) {
        this.animationEnabled = enabled;
    }

    public void setAnimationDuration(int duration) {
        this.animationDuration = duration;
    }

    public double getScaleX() {
        return scaleX;
    }

    public double getScaleY() {
        return scaleY;
    }

    public double getTranslateX() {
        return translateX;
    }

    public double getTranslateY() {
        return translateY;
    }
}
