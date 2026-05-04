package com.yishape.lab.math.plot.javafx.j3d;

import javafx.geometry.Point3D;
import javafx.scene.Group;
import javafx.scene.Node;
import javafx.scene.paint.PhongMaterial;
import javafx.scene.shape.Box;
import javafx.scene.shape.MeshView;
import javafx.scene.transform.Rotate;

/**
 * 箭头、饼图扇段等组合图元。
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public final class JavaFx3dPlotPrimitives {

    @FunctionalInterface
    public interface FxCylinder {
        void addFx(Point3D p0, Point3D p1, double radius, PhongMaterial mat);
    }

    @FunctionalInterface
    public interface NodePlace {
        void place(Node node, double xd, double yd, double zd);
    }

    private JavaFx3dPlotPrimitives() {
    }

    public static void addArrowData(Group plotContent, FxCylinder cylinderFx,
            java.util.function.Consumer<Point3D> growDataFromFxPoint,
            double x0, double y0, double z0, double ux, double uy, double uz,
            double shaftLen, double radius, PhongMaterial shaftMat, PhongMaterial headMat) {
        Point3D p0 = new Point3D(x0, z0, y0);
        Point3D dirData = new Point3D(ux, uz, uy);
        if (dirData.magnitude() < 1e-12) {
            return;
        }
        Point3D u = dirData.normalize();
        Point3D pTip = p0.add(u.multiply(shaftLen));
        Point3D pShaftEnd = p0.add(u.multiply(shaftLen * 0.82));
        cylinderFx.addFx(p0, pShaftEnd, radius, shaftMat);
        MeshView cone = JavaFx3dMeshGeometry.coneMeshView(radius * 2.4, shaftLen * 0.2, headMat);
        Point3D midCone = pTip.midpoint(pShaftEnd);
        cone.setTranslateX(midCone.getX());
        cone.setTranslateY(midCone.getY());
        cone.setTranslateZ(midCone.getZ());
        Point3D yUp = new Point3D(0, 1, 0);
        Point3D axis = yUp.crossProduct(u);
        double dot = JavaFx3dFxUtil.clamp(yUp.dotProduct(u), -1, 1);
        double ang = Math.toDegrees(Math.acos(dot));
        if (axis.magnitude() > 1e-9) {
            cone.getTransforms().add(new Rotate(ang, axis.normalize()));
        } else if (dot < 0) {
            cone.getTransforms().add(new Rotate(180, Rotate.X_AXIS));
        }
        growDataFromFxPoint.accept(pTip);
        plotContent.getChildren().add(cone);
    }

    /** 近似扇形块：水平面内窄盒 + 绕 Y 旋转（数据 z 为挤出厚度）。 */
    public static void addPieWedge(NodePlace placer, double r, double thick, double a0, double a1, PhongMaterial mat) {
        double mid = (a0 + a1) / 2;
        double span = Math.max(1e-3, a1 - a0);
        double chord = 2 * r * Math.sin(span / 2);
        double radial = r * Math.cos(span / 2);
        Box b = new Box(chord * 1.02, thick, Math.max(0.08, radial * 0.35));
        b.setMaterial(mat);
        b.getTransforms().add(new Rotate(Math.toDegrees(-mid), Rotate.Y_AXIS));
        double xc = radial * 0.45 * Math.cos(mid);
        double yc = radial * 0.45 * Math.sin(mid);
        placer.place(b, xc, yc, thick / 2);
    }
}
