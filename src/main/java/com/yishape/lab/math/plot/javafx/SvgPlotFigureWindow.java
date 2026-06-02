package com.yishape.lab.math.plot.javafx;

import com.yishape.lab.math.plot.PlotException;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.svg.SvgPlot;

import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Dialog;
import javafx.scene.control.Label;
import javafx.scene.control.MenuButton;
import javafx.scene.control.MenuItem;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Separator;
import javafx.scene.control.TextField;
import javafx.scene.control.ToolBar;
import javafx.scene.image.ImageView;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import javafx.stage.Window;

import javax.imageio.ImageIO;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.StringReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.apache.batik.transcoder.TranscoderInput;
import org.apache.batik.transcoder.TranscoderOutput;
import org.apache.batik.transcoder.image.PNGTranscoder;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.common.PDRectangle;
import org.apache.pdfbox.pdmodel.graphics.image.PDImageXObject;

/**
 * {@link SvgPlot#show()} 使用的交互窗口：用 Batik 将 SVG 栅格化后在 {@link ImageView} 中预览（不依赖 javafx-web）。
 * 矢量导出仍使用 {@link SvgPlot#saveAsSvg(String)}；PNG/PDF 为与当前图幅同尺寸的栅格。
 */
public final class SvgPlotFigureWindow {

    private static final int TOOLBAR_VSPACE = 40;

    private SvgPlotFigureWindow() {
    }

    public static void open(SvgPlot plot) {
        JavaFxPlot.ensureJavaFxToolkitStarted();
        Platform.runLater(() -> openOnFxThread(plot));
    }

    private static void openOnFxThread(SvgPlot plot) {
        Stage stage = new Stage();
        stage.setTitle(JavaFxChartUtils.APP_PLOT_STAGE_TITLE);

        ImageView imageView = new ImageView();
        imageView.setPreserveRatio(true);
        ScrollPane scroll = new ScrollPane(imageView);
        scroll.setFitToWidth(true);
        scroll.setFitToHeight(true);
        scroll.setPannable(true);
        refreshPreviewImage(plot, imageView);

        ToolBar bar = buildToolBar(plot, imageView, stage);
        BorderPane root = new BorderPane();
        root.setTop(new VBox(bar));
        root.setCenter(scroll);

        Scene scene = new Scene(root, plot.getWidth(), plot.getHeight() + TOOLBAR_VSPACE);
        stage.setScene(scene);
        stage.show();
    }

    private static void refreshPreviewImage(SvgPlot plot, ImageView imageView) {
        try {
            BufferedImage buf = rasterizeSvg(plot.toSvgString(), plot.getWidth(), plot.getHeight());
            imageView.setImage(SwingFXUtils.toFXImage(buf, null));
        } catch (Exception ex) {
            imageView.setImage(null);
            Alert a = new Alert(Alert.AlertType.WARNING);
            a.setTitle("SVG 预览");
            a.setHeaderText("栅格化预览失败（仍可使用「导出 → SVG」保存矢量图）");
            a.setContentText(ex.getMessage() != null ? ex.getMessage() : ex.toString());
            a.show();
        }
    }

    /**
     * 将 SVG 文档光栅化为 {@link BufferedImage}（用于窗口预览与 PNG/PDF）。
     */
    static BufferedImage rasterizeSvg(String svg, int targetW, int targetH) throws Exception {
        PNGTranscoder t = new PNGTranscoder();
        if (targetW > 0) {
            t.addTranscodingHint(PNGTranscoder.KEY_WIDTH, (float) targetW);
        }
        if (targetH > 0) {
            t.addTranscodingHint(PNGTranscoder.KEY_HEIGHT, (float) targetH);
        }
        TranscoderInput input = new TranscoderInput(new StringReader(svg));
        ByteArrayOutputStream o = new ByteArrayOutputStream(Math.max(16_384, targetW * targetH / 2));
        t.transcode(input, new TranscoderOutput(o));
        byte[] png = o.toByteArray();
        BufferedImage decoded = ImageIO.read(new ByteArrayInputStream(png));
        if (decoded == null) {
            throw new IOException("Batik 输出无法被 ImageIO 解析为 PNG");
        }
        if (decoded.getType() != BufferedImage.TYPE_INT_ARGB) {
            BufferedImage argb = new BufferedImage(decoded.getWidth(), decoded.getHeight(), BufferedImage.TYPE_INT_ARGB);
            Graphics2D g = argb.createGraphics();
            try {
                g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
                g.drawImage(decoded, 0, 0, null);
            } finally {
                g.dispose();
            }
            return argb;
        }
        return decoded;
    }

    private static ToolBar buildToolBar(SvgPlot plot, ImageView imageView, Stage stage) {
        MenuButton export = new MenuButton("导出");
        MenuItem png = new MenuItem("PNG（栅格，与图幅同尺寸）…");
        png.setOnAction(e -> exportPngRaster(plot, stage));
        MenuItem svg = new MenuItem("SVG（矢量）…");
        svg.setOnAction(e -> exportVectorSvg(plot, stage));
        MenuItem pdf = new MenuItem("PDF（栅格，与图幅同尺寸）…");
        pdf.setOnAction(e -> exportPdfRaster(plot, stage));
        export.getItems().addAll(png, svg, pdf);

        List<String> themes = JavaFxThemeManager.getRegisteredThemeNames();
        ComboBox<String> themeBox = new ComboBox<>(FXCollections.observableArrayList(themes));
        themeBox.setValue(plot.getTheme());
        themeBox.valueProperty().addListener((o, a, n) -> {
            if (n != null && !n.isBlank()) {
                plot.theme(n);
                refreshPreviewImage(plot, imageView);
            }
        });

        Button props = new Button("标题 / 轴 / 图例");
        props.setOnAction(e -> openPropertiesDialog(plot, imageView, stage));

        return new ToolBar(
            export,
            new Separator(),
            new Label("主题:"),
            themeBox,
            new Separator(),
            props
        );
    }

    private static void exportPngRaster(SvgPlot plot, Window owner) {
        FileChooser ch = new FileChooser();
        ch.setTitle("导出 PNG");
        ch.getExtensionFilters().add(new FileChooser.ExtensionFilter("PNG", "*.png"));
        File f = ch.showSaveDialog(owner);
        if (f == null) {
            return;
        }
        try {
            BufferedImage buf = rasterizeSvg(plot.toSvgString(), plot.getWidth(), plot.getHeight());
            ImageIO.write(buf, "png", f);
        } catch (Exception ex) {
            throw new PlotException("保存 PNG 失败: " + ex.getMessage(), ex);
        }
    }

    private static void exportVectorSvg(SvgPlot plot, Window owner) {
        FileChooser ch = new FileChooser();
        ch.setTitle("导出 SVG");
        ch.getExtensionFilters().add(new FileChooser.ExtensionFilter("SVG", "*.svg"));
        File f = ch.showSaveDialog(owner);
        if (f == null) {
            return;
        }
        plot.saveAsSvg(f.getPath());
    }

    private static void exportPdfRaster(SvgPlot plot, Window owner) {
        FileChooser ch = new FileChooser();
        ch.setTitle("导出 PDF");
        ch.getExtensionFilters().add(new FileChooser.ExtensionFilter("PDF", "*.pdf"));
        File f = ch.showSaveDialog(owner);
        if (f == null) {
            return;
        }
        try {
            BufferedImage buf = rasterizeSvg(plot.toSvgString(), plot.getWidth(), plot.getHeight());
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ImageIO.write(buf, "png", baos);
            int w = buf.getWidth();
            int h = buf.getHeight();
            writePngBytesToPdf(Path.of(f.getPath()), w, h, baos.toByteArray());
        } catch (Exception ex) {
            throw new PlotException("保存 PDF 失败: " + ex.getMessage(), ex);
        }
    }

    private static void writePngBytesToPdf(Path path, int w, int h, byte[] png) throws IOException {
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        try (PDDocument doc = new PDDocument()) {
            PDPage page = new PDPage(new PDRectangle((float) w, (float) h));
            doc.addPage(page);
            PDImageXObject img = PDImageXObject.createFromByteArray(doc, png, "chart");
            try (PDPageContentStream cs = new PDPageContentStream(doc, page)) {
                cs.drawImage(img, 0, 0, w, h);
            }
            doc.save(path.toFile());
        }
    }

    private static void openPropertiesDialog(SvgPlot plot, ImageView imageView, Stage owner) {
        Dialog<ButtonType> dialog = new Dialog<>();
        dialog.setTitle("图表属性");
        dialog.setHeaderText("标题、坐标轴标签与图例文字");
        JavaFxChartRenderer.ChartConfig cfg = plot.getChartConfig();

        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(8);
        grid.setPadding(new Insets(12));

        TextField tfTitle = new TextField(cfg.title);
        TextField tfSub = new TextField(cfg.subtitle != null ? cfg.subtitle : "");
        TextField tfX = new TextField(cfg.xlabel);
        TextField tfY = new TextField(cfg.ylabel);
        int row = 0;
        grid.addRow(row++, new Label("标题"), tfTitle);
        grid.addRow(row++, new Label("副标题"), tfSub);
        grid.addRow(row++, new Label("X 轴"), tfX);
        grid.addRow(row++, new Label("Y 轴"), tfY);

        List<TextField> seriesFields = new ArrayList<>();
        List<SeriesData> series = plot.mutableSeriesListForFigureUi();
        if (!series.isEmpty()) {
            VBox seriesBox = new VBox(6);
            seriesBox.getChildren().add(new Label("序列名称（图例）"));
            for (SeriesData s : series) {
                TextField tf = new TextField(s.name != null ? s.name : "");
                seriesFields.add(tf);
                seriesBox.getChildren().add(tf);
            }
            ScrollPane sp = new ScrollPane(seriesBox);
            sp.setFitToWidth(true);
            sp.setPrefHeight(Math.min(220, 40 + series.size() * 36));
            grid.add(new Label("图例"), 0, row);
            grid.add(sp, 1, row);
        }

        dialog.getDialogPane().setContent(grid);
        dialog.getDialogPane().getButtonTypes().addAll(ButtonType.OK, ButtonType.CANCEL);
        dialog.initOwner(owner);
        dialog.showAndWait().ifPresent(bt -> {
            if (bt != ButtonType.OK) {
                return;
            }
            cfg.title = tfTitle.getText();
            cfg.subtitle = tfSub.getText();
            cfg.xlabel = tfX.getText();
            cfg.ylabel = tfY.getText();
            for (int i = 0; i < seriesFields.size() && i < series.size(); i++) {
                String nn = seriesFields.get(i).getText();
                series.get(i).name = nn != null ? nn : "";
            }
            refreshPreviewImage(plot, imageView);
        });
    }
}
