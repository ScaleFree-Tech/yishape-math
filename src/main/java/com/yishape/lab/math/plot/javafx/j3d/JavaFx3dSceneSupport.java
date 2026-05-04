package com.yishape.lab.math.plot.javafx.j3d;

import com.yishape.lab.math.plot.PlotException;
import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.AmbientLight;
import javafx.scene.Group;
import javafx.scene.PerspectiveCamera;
import javafx.scene.PointLight;
import javafx.scene.Scene;
import javafx.scene.SceneAntialiasing;
import javafx.scene.SnapshotParameters;
import javafx.scene.SubScene;
import javafx.scene.image.WritableImage;
import javafx.scene.input.ScrollEvent;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.stage.Stage;
import javafx.stage.StageStyle;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/**
 * {@link JavaFx3dPlot} 的 Stage / SubScene / 快照与导航，与数据绘制解耦。
 * <p>访问 plot 的同包字段（包级可见）。</p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
final class JavaFx3dSceneSupport {

    private JavaFx3dSceneSupport() {
    }

    static void show(JavaFx3dPlot plot) {
        JavaFx3dFxUtil.ensureFxToolkit();
        Platform.runLater(() -> {
            plot.rebuildWorldTransform();
            if (plot.stage == null) {
                plot.stage = new Stage();
            }
            Group root3d = new Group();
            Color bg = plot.themeManager.getBackgroundColor();
            AmbientLight am = ambientForBackground(bg);
            PointLight pl = pointLightForBackground(bg);
            PointLight fill = fillLightForBackground(bg);
            root3d.getChildren().addAll(am, pl, fill, plot.pivot);
            plot.camera = new PerspectiveCamera(true);
            plot.camera.setNearClip(0.1);
            plot.camera.setFarClip(8000);
            plot.camera.setFieldOfView(45);
            plot.camera.setTranslateZ(-690);
            plot.subScene = new SubScene(root3d, plot.width, plot.height, true, SceneAntialiasing.BALANCED);
            plot.subScene.setFill(bg);
            plot.subScene.setCamera(plot.camera);
            armNavigation(plot, plot.subScene);

            // 创建根布局（支持图例和Tooltip的叠加）
            plot.rootPane = new BorderPane();

            // 标题区域
            if (!plot.titleText.isEmpty()) {
                VBox tb = new VBox(4);
                tb.setAlignment(Pos.CENTER);
                tb.setPadding(new Insets(10));
                javafx.scene.control.Label t = new javafx.scene.control.Label(plot.titleText);
                t.setFont(Font.font(null, FontWeight.BOLD, 16));
                t.setTextFill(plot.themeManager.getTextColor());
                tb.getChildren().add(t);
                if (plot.subtitleText != null && !plot.subtitleText.isEmpty()) {
                    javafx.scene.control.Label st = new javafx.scene.control.Label(plot.subtitleText);
                    st.setTextFill(plot.themeManager.getMutedTextColor());
                    tb.getChildren().add(st);
                }
                plot.rootPane.setTop(tb);
            }

            // 中心3D场景
            plot.rootPane.setCenter(plot.subScene);

            // 添加图例（如果系列数量>1）
            if (plot.legendItems != null && !plot.legendItems.isEmpty()) {
                javafx.scene.Node legend = JavaFx3dLegend.createFloatingLegend(
                        plot.legendItems, plot.themeManager, "horizontal");
                BorderPane.setAlignment(legend, Pos.TOP_RIGHT);
                BorderPane.setMargin(legend, new Insets(10, 10, 0, 0));
                plot.rootPane.setRight(legend);
            }

            // 添加Tooltip层
            armTooltip(plot, plot.subScene);

            plot.scene = new Scene(plot.rootPane, plot.width, plot.height + (plot.titleText.isEmpty() ? 0 : 70));
            plot.stage.setScene(plot.scene);
            plot.stage.setTitle(plot.titleText.isEmpty() ? "JavaFX 3D" : plot.titleText);
            plot.stage.show();
        });
    }

    static void armNavigation(JavaFx3dPlot plot, SubScene sub) {
        sub.setOnMousePressed(e -> {
            plot.anchorX = e.getSceneX();
            plot.anchorY = e.getSceneY();
            plot.anchorAngleX = plot.dragRx.getAngle();
            plot.anchorAngleY = plot.dragRy.getAngle();
        });
        sub.setOnMouseDragged(e -> {
            double dx = e.getSceneX() - plot.anchorX;
            double dy = e.getSceneY() - plot.anchorY;
            plot.dragRy.setAngle(plot.anchorAngleY + dx * 0.35);
            plot.dragRx.setAngle(plot.anchorAngleX - dy * 0.35);
        });
        sub.addEventHandler(ScrollEvent.SCROLL, e -> {
            double dz = e.getDeltaY();
            plot.camera.setTranslateZ(plot.camera.getTranslateZ() + dz * 0.45);
        });
    }

    /**
     * 添加Tooltip交互支持
     */
    static void armTooltip(JavaFx3dPlot plot, SubScene sub) {
        // 创建Tooltip标签
        javafx.scene.control.Label tooltipLabel = new javafx.scene.control.Label();
        tooltipLabel.setStyle("-fx-background-color: rgba(0,0,0,0.75); -fx-text-fill: white; " +
                "-fx-padding: 8px; -fx-background-radius: 4px; -fx-font-size: 12px;");
        tooltipLabel.setVisible(false);

        if (plot.rootPane != null) {
            plot.rootPane.getChildren().add(tooltipLabel);
            BorderPane.setAlignment(tooltipLabel, Pos.TOP_LEFT);
        }

        // 鼠标移动时更新Tooltip位置
        sub.setOnMouseMoved(e -> {
            if (tooltipLabel.isVisible()) {
                tooltipLabel.setLayoutX(e.getSceneX() + 15);
                tooltipLabel.setLayoutY(e.getSceneY() + 15);
            }
        });

        // 鼠标进入/离开场景
        sub.setOnMouseEntered(e -> tooltipLabel.setVisible(tooltipLabel.getText() != null && !tooltipLabel.getText().isEmpty()));
        sub.setOnMouseExited(e -> tooltipLabel.setVisible(false));

        // 存储引用以便更新
        plot.tooltipLabel = tooltipLabel;
    }

    static void ensureOffscreenSubScene(JavaFx3dPlot plot) {
        if (plot.subScene != null) {
            return;
        }
        Group root3d = new Group();
        Color bg = plot.themeManager.getBackgroundColor();
        AmbientLight am = ambientForBackground(bg);
        PointLight pl = pointLightForBackground(bg);
        PointLight fill = fillLightForBackground(bg);
        root3d.getChildren().addAll(am, pl, fill, plot.pivot);
        plot.camera = new PerspectiveCamera(true);
        plot.camera.setNearClip(0.1);
        plot.camera.setFarClip(8000);
        plot.camera.setFieldOfView(45);
        plot.camera.setTranslateZ(-690);
        plot.subScene = new SubScene(root3d, plot.width, plot.height, true, SceneAntialiasing.BALANCED);
        plot.subScene.setFill(bg);
        plot.subScene.setCamera(plot.camera);
        armNavigation(plot, plot.subScene);
    }

    static void attachSnapshotHost(JavaFx3dPlot plot) {
        ensureOffscreenSubScene(plot);
        if (plot.snapshotHostStage == null) {
            plot.snapshotHostStage = new Stage();
            plot.snapshotHostStage.initStyle(StageStyle.UNDECORATED);
            plot.snapshotHostRoot = new BorderPane();
            Scene sc = new Scene(plot.snapshotHostRoot, plot.width, plot.height);
            plot.snapshotHostStage.setScene(sc);
            plot.snapshotHostStage.setOpacity(0.01);
            plot.snapshotHostStage.setX(-4000);
            plot.snapshotHostStage.setY(-4000);
        }
        plot.snapshotHostRoot.setCenter(plot.subScene);
        plot.snapshotHostStage.setWidth(plot.width);
        plot.snapshotHostStage.setHeight(plot.height);
        plot.snapshotHostStage.show();
    }

    static void writeSnapshotPng(JavaFx3dPlot plot, File file) {
        JavaFx3dFxUtil.ensureFxToolkit();
        CountDownLatch done = new CountDownLatch(1);
        AtomicReference<Throwable> err = new AtomicReference<>();
        Runnable pass1 = () -> {
            try {
                plot.rebuildWorldTransform();
                attachSnapshotHost(plot);
                Platform.runLater(() -> {
                    try {
                        SnapshotParameters sp = new SnapshotParameters();
                        WritableImage wi = new WritableImage(plot.width, plot.height);
                        plot.subScene.snapshot(sp, wi);
                        File parent = file.getParentFile();
                        if (parent != null && !parent.exists()) {
                            parent.mkdirs();
                        }
                        ImageIO.write(SwingFXUtils.fromFXImage(wi, null), "png", file);
                    } catch (Throwable t) {
                        err.set(t);
                    } finally {
                        done.countDown();
                    }
                });
            } catch (Throwable t) {
                err.set(t);
                done.countDown();
            }
        };
        if (Platform.isFxApplicationThread()) {
            pass1.run();
        } else {
            Platform.runLater(pass1);
        }
        try {
            if (!done.await(60, TimeUnit.SECONDS)) {
                throw new PlotException("3D 快照超时");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new PlotException("3D 快照被中断", e);
        }
        Throwable t = err.get();
        if (t != null) {
            if (t instanceof PlotException) {
                throw (PlotException) t;
            }
            if (t instanceof IOException e2) {
                throw new PlotException("3D 快照写入失败: " + e2.getMessage(), e2);
            }
            throw new PlotException("3D 快照失败: " + t.getMessage(), t);
        }
    }

    static double backgroundLuminance(Color bg) {
        return 0.2126 * bg.getRed() + 0.7152 * bg.getGreen() + 0.0722 * bg.getBlue();
    }

    static AmbientLight ambientForBackground(Color bg) {
        if (backgroundLuminance(bg) < 0.42) {
            return new AmbientLight(Color.color(0.55, 0.55, 0.58));
        }
        return new AmbientLight(Color.color(0.42, 0.42, 0.45));
    }

    static PointLight fillLightForBackground(Color bg) {
        if (backgroundLuminance(bg) < 0.42) {
            PointLight f = new PointLight(Color.color(0.5, 0.55, 0.65));
            f.setTranslateX(-480);
            f.setTranslateY(220);
            f.setTranslateZ(380);
            return f;
        }
        PointLight f = new PointLight(Color.color(0.95, 0.96, 1.0));
        f.setTranslateX(-420);
        f.setTranslateY(160);
        f.setTranslateZ(360);
        return f;
    }

    static PointLight pointLightForBackground(Color bg) {
        PointLight pl = new PointLight(
                backgroundLuminance(bg) < 0.42 ? Color.color(0.93, 0.95, 1.0) : Color.WHITE);
        pl.setTranslateX(400);
        pl.setTranslateY(-280);
        pl.setTranslateZ(-340);
        return pl;
    }
}
