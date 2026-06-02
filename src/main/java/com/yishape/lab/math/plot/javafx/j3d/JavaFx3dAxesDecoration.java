package com.yishape.lab.math.plot.javafx.j3d;

import com.yishape.lab.math.plot.AxisTicks;
import com.yishape.lab.math.plot.javafx.JavaFxChartUtils;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;
import javafx.geometry.Point3D;
import javafx.scene.Group;
import javafx.scene.paint.Color;
import javafx.scene.paint.PhongMaterial;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.scene.text.Text;
import javafx.scene.transform.Rotate;
import javafx.scene.transform.Scale;
import javafx.scene.transform.Transform;
import javafx.scene.transform.Translate;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * 数据坐标系下的坐标轴、底面网格与刻度标签（与 {@link JavaFx3dPlot#place} 的 FX 映射一致：
 * 数据 (x,y,z) ↔ JavaFX 平移 (x, z, y)）。
 * <p>默认相机自 -Z 朝原点看、Y 轴向上；{@link Text} 躺在局部 XY 平面（朝向 +Z），故铺在 XZ「地面」上的刻度须绕 X 旋转，
 * 否则从默认视角看过去会像一条竖线，所有数字投影重叠。</p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
final class JavaFx3dAxesDecoration {

    private JavaFx3dAxesDecoration() {
    }

    /** 刻度数字贴在哪个竖直面 / 地面，决定旋转让字形可读。 */
    private enum LabelOrient {
        /** 贴在 XZ 平面（Y 恒定，常见：X 轴刻度、X 轴标题）。 */
        FLAT_XZ,
        /** 贴在 YZ 平面（X 恒定，常见：数据 Y → 场景 Z 的刻度与标题）。 */
        FLAT_YZ,
        /** 保持 XY 平面（Z 恒定，常见：数据 Z → 场景 Y 的刻度）。 */
        UPRIGHT_XY
    }

    /**
     * @param worldSceneScale {@link JavaFx3dPlot#rebuildWorldTransform} 中用于 contentRaw 的均匀缩放 s（≈ 220/数据跨度）。
     */
    static void rebuild(Group axesRoot,
            double minX, double maxX,
            double minY, double maxY,
            double minZ, double maxZ,
            String xLabel, String yLabel, String zLabel,
            AxisTicks xticks, AxisTicks yticks, AxisTicks zticks,
            JavaFxThemeManager theme,
            double worldSceneScale) {
        axesRoot.getChildren().clear();

        double ex = Math.max(1e-9, maxX - minX);
        double ey = Math.max(1e-9, maxY - minY);
        double ez = Math.max(1e-9, maxZ - minZ);
        double ext = Math.max(ex, Math.max(ey, ez));
        double invScene = 1.0 / Math.max(1e-12, worldSceneScale);

        double padX = ex * 0.04;
        double padY = ey * 0.04;
        double padZ = ez * 0.04;

        double loX = minX - padX;
        double hiX = maxX + padX;
        double loY = minY - padY;
        double hiY = maxY + padY;
        double loZ = minZ - padZ;
        double hiZ = maxZ + padZ;

        Color bg = theme.getBackgroundColor();
        Color textColor = theme.getTextColor();
        Color muted = theme.getMutedTextColor();

        double bgLum = 0.2126 * bg.getRed() + 0.7152 * bg.getGreen() + 0.0722 * bg.getBlue();
        boolean darkBg = bgLum < 0.42;

        Color gridC = darkBg
                ? Color.color(0.55, 0.58, 0.64, 0.42)
                : new Color(muted.getRed(), muted.getGreen(), muted.getBlue(), 0.38);
        Color boxC = darkBg
                ? Color.color(0.55, 0.58, 0.65, 0.55)
                : new Color(muted.getRed(), muted.getGreen(), muted.getBlue(), 0.48);

        Color labelFill = darkBg
                ? Color.color(0.92, 0.94, 0.96)
                : textColor;

        Color rgbX = darkBg ? Color.color(0.98, 0.48, 0.42) : Color.color(0.78, 0.2, 0.14);
        Color rgbY = darkBg ? Color.color(0.42, 0.92, 0.55) : Color.color(0.08, 0.52, 0.22);
        Color rgbZ = darkBg ? Color.color(0.45, 0.74, 0.98) : Color.color(0.12, 0.38, 0.84);

        double rAxis = JavaFx3dFxUtil.clamp(ext * 0.0019, 0.018, ext * 0.0045);
        double rTick = rAxis * 0.78;
        double tickLen = JavaFx3dFxUtil.clamp(ext * 0.034, 0.075, ext * 0.075);
        double lbl = JavaFx3dFxUtil.clamp(ext * 0.05, 0.11, ext * 0.15);

        PhongMaterial axisMatX = new PhongMaterial(rgbX);
        PhongMaterial axisMatY = new PhongMaterial(rgbY);
        PhongMaterial axisMatZ = new PhongMaterial(rgbZ);
        for (PhongMaterial m : new PhongMaterial[] {axisMatX, axisMatY, axisMatZ}) {
            m.setSpecularColor(Color.color(1, 1, 1, 0.28));
            m.setSpecularPower(32);
        }
        PhongMaterial gridMat = new PhongMaterial(gridC);
        PhongMaterial boxMat = new PhongMaterial(boxC);

        LabelAnchorTracker anchors = new LabelAnchorTracker(ext);

        int xDiv = JavaFx3dTickUtil.fallbackDivisions(xticks, 5);
        int yDiv = JavaFx3dTickUtil.fallbackDivisions(yticks, 5);
        int zDiv = JavaFx3dTickUtil.fallbackDivisions(zticks, 4);

        List<JavaFx3dTickUtil.Tick> txs = JavaFx3dTickUtil.ticksForAxis(xticks, loX, hiX, xDiv);
        List<JavaFx3dTickUtil.Tick> tys = JavaFx3dTickUtil.ticksForAxis(yticks, loY, hiY, yDiv);
        List<JavaFx3dTickUtil.Tick> tzs = JavaFx3dTickUtil.ticksForAxis(zticks, loZ, hiZ, zDiv);

        // 底面网格
        for (JavaFx3dTickUtil.Tick tx : txs) {
            double x = tx.value();
            if (x < minX - 1e-9 || x > maxX + 1e-9) {
                continue;
            }
            JavaFx3dMeshGeometry.addCylinderBetweenFx(axesRoot,
                    new Point3D(x, minZ, loY),
                    new Point3D(x, minZ, hiY),
                    rAxis * 0.28,
                    gridMat);
        }
        for (JavaFx3dTickUtil.Tick ty : tys) {
            double y = ty.value();
            if (y < minY - 1e-9 || y > maxY + 1e-9) {
                continue;
            }
            JavaFx3dMeshGeometry.addCylinderBetweenFx(axesRoot,
                    new Point3D(loX, minZ, y),
                    new Point3D(hiX, minZ, y),
                    rAxis * 0.28,
                    gridMat);
        }

        // 线框包围盒（与数据盒对齐，深色背景下仍可辨）
        addWireBox(axesRoot, new Point3D(minX, minZ, minY), new Point3D(hiX, hiZ, hiY), rAxis * 0.52, boxMat);

        // 三主轴：红/绿/蓝 对应数据 X、Y、Z（展示坐标与 JavaFX Y=数据 Z 的映射关系）。
        Point3D origin = new Point3D(minX, minZ, minY);
        JavaFx3dMeshGeometry.addCylinderBetweenFx(axesRoot, origin, new Point3D(hiX, minZ, minY), rAxis, axisMatX);
        JavaFx3dMeshGeometry.addCylinderBetweenFx(axesRoot, origin, new Point3D(minX, minZ, hiY), rAxis, axisMatY);
        JavaFx3dMeshGeometry.addCylinderBetweenFx(axesRoot, origin, new Point3D(minX, hiZ, minY), rAxis, axisMatZ);

        // X 轴刻度（地面 XZ）
        for (JavaFx3dTickUtil.Tick tx : txs) {
            double x = tx.value();
            if (x < minX - 1e-9 * ex || x > maxX + 1e-9 * ex) {
                continue;
            }
            Point3D t0 = new Point3D(x, minZ, minY);
            Point3D t1 = new Point3D(x, minZ + tickLen, minY);
            JavaFx3dMeshGeometry.addCylinderBetweenFx(axesRoot, t0, t1, rTick, axisMatX);
            String lab = tx.labelOverride() != null ? tx.labelOverride() : JavaFxChartUtils.formatNumber(x);
            double ly = minZ + tickLen + lbl * 0.22;
            double lz = minY - lbl * 0.62;
            if (anchors.claim(x, ly, lz)) {
                Text tg = axisText(lab, theme, labelFill, false);
                axesRoot.getChildren().add(wrapLabel(tg, x, ly, lz, invScene, LabelOrient.FLAT_XZ));
            }
        }

        // 数据 Y → 场景 Z（侧面 YZ）
        for (JavaFx3dTickUtil.Tick ty : tys) {
            double y = ty.value();
            if (y < minY - 1e-9 * ey || y > maxY + 1e-9 * ey) {
                continue;
            }
            Point3D t0 = new Point3D(minX, minZ, y);
            Point3D t1 = new Point3D(minX, minZ + tickLen, y);
            JavaFx3dMeshGeometry.addCylinderBetweenFx(axesRoot, t0, t1, rTick, axisMatY);
            String lab = ty.labelOverride() != null ? ty.labelOverride() : JavaFxChartUtils.formatNumber(y);
            double lx = minX - lbl * 1.42;
            double ly = minZ + tickLen * 0.28;
            if (anchors.claim(lx, ly, y)) {
                Text tg = axisText(lab, theme, labelFill, false);
                axesRoot.getChildren().add(wrapLabel(tg, lx, ly, y, invScene, LabelOrient.FLAT_YZ));
            }
        }

        // 数据 Z → 场景 Y（竖直，XY 平面随高度平移）
        for (JavaFx3dTickUtil.Tick tz : tzs) {
            double z = tz.value();
            if (z < minZ - 1e-9 * ez || z > maxZ + 1e-9 * ez) {
                continue;
            }
            Point3D t0 = new Point3D(minX, z, minY);
            Point3D t1 = new Point3D(minX + tickLen, z, minY);
            JavaFx3dMeshGeometry.addCylinderBetweenFx(axesRoot, t0, t1, rTick, axisMatZ);
            String lab = tz.labelOverride() != null ? tz.labelOverride() : JavaFxChartUtils.formatNumber(z);
            double lx = minX - lbl * 1.38;
            double lz = minY - tickLen - lbl * 0.42;
            if (anchors.claim(lx, z, lz)) {
                Text tg = axisText(lab, theme, labelFill, false);
                axesRoot.getChildren().add(wrapLabel(tg, lx, z, lz, invScene, LabelOrient.UPRIGHT_XY));
            }
        }

        // 轴标题（与对应刻度同朝向）
        if (xLabel != null && !xLabel.isEmpty()) {
            double ly = minZ + tickLen + lbl * 1.05;
            double lz = minY - lbl * 1.48;
            if (anchors.claim((minX + maxX) / 2, ly, lz)) {
                Text t = axisText(xLabel, theme, rgbX, true);
                axesRoot.getChildren().add(wrapLabel(
                        t, (minX + maxX) / 2, ly, lz, invScene, LabelOrient.FLAT_XZ));
            }
        }
        if (yLabel != null && !yLabel.isEmpty()) {
            double lx = minX - lbl * 2.55;
            double ly = minZ + tickLen * 0.42;
            double cz = (minY + maxY) / 2;
            if (anchors.claim(lx, ly, cz)) {
                Text t = axisText(yLabel, theme, rgbY, true);
                axesRoot.getChildren().add(wrapLabel(t, lx, ly, cz, invScene, LabelOrient.FLAT_YZ));
            }
        }
        if (zLabel != null && !zLabel.isEmpty()) {
            double lx = minX - lbl * 2.35;
            double cy = (minZ + maxZ) / 2;
            double lz = minY - lbl * 1.65;
            if (anchors.claim(lx, cy, lz)) {
                Text t = axisText(zLabel, theme, rgbZ, true);
                axesRoot.getChildren().add(wrapLabel(t, lx, cy, lz, invScene, LabelOrient.UPRIGHT_XY));
            }
        }
    }

    /** 立方体 12 条棱（FX 空间 min→max）。 */
    private static void addWireBox(Group root, Point3D c0, Point3D c1, double r, PhongMaterial mat) {
        double x0 = c0.getX();
        double y0 = c0.getY();
        double z0 = c0.getZ();
        double x1 = c1.getX();
        double y1 = c1.getY();
        double z1 = c1.getZ();
        Point3D p000 = new Point3D(x0, y0, z0);
        Point3D p100 = new Point3D(x1, y0, z0);
        Point3D p010 = new Point3D(x0, y1, z0);
        Point3D p110 = new Point3D(x1, y1, z0);
        Point3D p001 = new Point3D(x0, y0, z1);
        Point3D p101 = new Point3D(x1, y0, z1);
        Point3D p011 = new Point3D(x0, y1, z1);
        Point3D p111 = new Point3D(x1, y1, z1);
        JavaFx3dMeshGeometry.addCylinderBetweenFx(root, p000, p100, r, mat);
        JavaFx3dMeshGeometry.addCylinderBetweenFx(root, p000, p010, r, mat);
        JavaFx3dMeshGeometry.addCylinderBetweenFx(root, p000, p001, r, mat);
        JavaFx3dMeshGeometry.addCylinderBetweenFx(root, p100, p110, r, mat);
        JavaFx3dMeshGeometry.addCylinderBetweenFx(root, p100, p101, r, mat);
        JavaFx3dMeshGeometry.addCylinderBetweenFx(root, p010, p110, r, mat);
        JavaFx3dMeshGeometry.addCylinderBetweenFx(root, p010, p011, r, mat);
        JavaFx3dMeshGeometry.addCylinderBetweenFx(root, p001, p101, r, mat);
        JavaFx3dMeshGeometry.addCylinderBetweenFx(root, p001, p011, r, mat);
        JavaFx3dMeshGeometry.addCylinderBetweenFx(root, p110, p111, r, mat);
        JavaFx3dMeshGeometry.addCylinderBetweenFx(root, p101, p111, r, mat);
        JavaFx3dMeshGeometry.addCylinderBetweenFx(root, p011, p111, r, mat);
    }

    private static Group wrapLabel(Text text, double tx, double ty, double tz,
            double invWorldScale, LabelOrient orient) {
        Group g = new Group(text);
        List<Transform> tf = new ArrayList<>(4);
        tf.add(new Scale(invWorldScale, invWorldScale, invWorldScale));
        switch (orient) {
            case FLAT_XZ -> tf.add(new Rotate(90, Rotate.X_AXIS));
            case FLAT_YZ -> tf.add(new Rotate(90, Rotate.Y_AXIS));
            case UPRIGHT_XY -> { /* default Text 平面朝向 +Z */ }
        }
        tf.add(new Translate(tx, ty, tz));
        g.getTransforms().setAll(tf);
        return g;
    }

    private static Text axisText(String s, JavaFxThemeManager theme, Color fill, boolean title) {
        Text t = new Text(s);
        Font base = title ? theme.getTitleFont() : theme.getLabelFont();
        double sz = base.getSize() * (title ? 1.1 : 1.34);
        t.setFont(Font.font(base.getFamily(), title ? FontWeight.BOLD : FontWeight.NORMAL, sz));
        t.setFill(fill);
        return t;
    }

    /** 避免原点附近多轴刻度文字抢占同一 3D 锚点导致叠成黑块。 */
    private static final class LabelAnchorTracker {
        private final double eps;
        private final Set<Long> seen = new HashSet<>();

        LabelAnchorTracker(double dataExtent) {
            this.eps = Math.max(dataExtent * 0.009, 1e-6);
        }

        boolean claim(double fx, double fy, double fz) {
            long k = key(fx, fy, fz);
            if (seen.contains(k)) {
                return false;
            }
            seen.add(k);
            return true;
        }

        private long key(double fx, double fy, double fz) {
            long qx = Math.round(fx / eps);
            long qy = Math.round(fy / eps);
            long qz = Math.round(fz / eps);
            return ((qx & 0x1fffffL) << 42) | ((qy & 0x1fffffL) << 21) | (qz & 0x1fffffL);
        }
    }
}
