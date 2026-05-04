package com.yishape.lab.math.plot.javafx.j3d;

import javafx.geometry.Point3D;
import javafx.scene.Group;
import javafx.scene.paint.PhongMaterial;
import javafx.scene.shape.Cylinder;
import javafx.scene.shape.MeshView;
import javafx.scene.shape.TriangleMesh;
import javafx.scene.transform.Rotate;

/**
 * JavaFX 3D 几何体：圆柱线段、圆锥箭头等（与数据坐标 → JavaFX 平移约定无关，调用方传入已变换空间中的点）。
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
final class JavaFx3dMeshGeometry {

    private JavaFx3dMeshGeometry() {
    }

    static MeshView coneMeshView(double radius, double height, PhongMaterial mat) {
        int div = 32;
        TriangleMesh mesh = new TriangleMesh();
        mesh.getTexCoords().addAll(0, 0);
        float hb = (float) (-height / 2);
        float ht = (float) (height / 2);
        mesh.getPoints().addAll(0, ht, 0);
        for (int i = 0; i < div; i++) {
            double a = 2 * Math.PI * i / div;
            mesh.getPoints().addAll(
                    (float) (radius * Math.cos(a)), hb, (float) (radius * Math.sin(a)));
        }
        mesh.getPoints().addAll(0, hb, 0);
        int baseCenter = div + 1;
        for (int i = 0; i < div; i++) {
            int b1 = 1 + i;
            int b2 = 1 + ((i + 1) % div);
            mesh.getFaces().addAll(0, 0, b1, 0, b2, 0);
        }
        for (int i = 0; i < div; i++) {
            int b1 = 1 + i;
            int b2 = 1 + ((i + 1) % div);
            mesh.getFaces().addAll(baseCenter, 0, b2, 0, b1, 0);
        }
        MeshView mv = new MeshView(mesh);
        mv.setMaterial(mat);
        return mv;
    }

    /**
     * 在 JavaFX 坐标下连接两点绘制圆柱（圆柱默认沿 +Y）。
     */
    static void addCylinderBetweenFx(Group parent, Point3D p0, Point3D p1, double radius, PhongMaterial mat) {
        Point3D dVec = p1.subtract(p0);
        double len = dVec.magnitude();
        if (len < 1e-9) {
            return;
        }
        Cylinder cy = new Cylinder(radius, len);
        cy.setMaterial(mat);
        Point3D mid = p0.midpoint(p1);
        cy.setTranslateX(mid.getX());
        cy.setTranslateY(mid.getY());
        cy.setTranslateZ(mid.getZ());
        Point3D yAxis = new Point3D(0, 1, 0);
        Point3D dir = dVec.normalize();
        Point3D axis = yAxis.crossProduct(dir);
        double dot = JavaFx3dFxUtil.clamp(yAxis.dotProduct(dir), -1, 1);
        double ang = Math.toDegrees(Math.acos(dot));
        if (axis.magnitude() > 1e-9) {
            cy.getTransforms().add(new Rotate(ang, axis.normalize()));
        } else if (dot < 0) {
            cy.getTransforms().add(new Rotate(180, Rotate.X_AXIS));
        }
        parent.getChildren().add(cy);
    }
}
