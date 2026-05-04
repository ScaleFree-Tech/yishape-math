package com.yishape.lab.math.plot.javafx.j3d;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.plot.MeshGridHelper;
import javafx.scene.Group;
import javafx.scene.paint.Color;
import javafx.scene.paint.PhongMaterial;
import javafx.scene.shape.DrawMode;
import javafx.scene.shape.MeshView;
import javafx.scene.shape.TriangleMesh;

/**
 * 曲面三角网与平面等高线（marching 逐边）。
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public final class JavaFx3dSurfaceContour {

    /** 在数据空间中用圆柱线段连接两点（实现类负责坐标与 bounds）。 */
    @FunctionalInterface
    public interface DataCylinder {
        void add(double x0, double y0, double z0, double x1, double y1, double z1, double radius, PhongMaterial mat);
    }

    @FunctionalInterface
    public interface BoundsGrow {
        void grow(double dataX, double dataY, double dataZ);
    }

    private JavaFx3dSurfaceContour() {
    }

    public static void addSurfaceMesh(Group plotContent, BoundsGrow grow,
            IVector<?> xv, IVector<?> yv, IMatrix<?> zmat, PhongMaterial mat,
            boolean wireframe, boolean twoSided) {
        MeshGridHelper.requireZMatchesMeshIj(xv, yv, zmat);
        int nx = xv.length();
        int ny = yv.length();
        TriangleMesh mesh = new TriangleMesh();
        mesh.getTexCoords().addAll(0, 0);
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                double x = xv.get(i).doubleValue();
                double y = yv.get(j).doubleValue();
                double z = zmat.get(i, j).doubleValue();
                mesh.getPoints().addAll((float) x, (float) z, (float) y);
                grow.grow(x, y, z);
            }
        }
        for (int i = 0; i < nx - 1; i++) {
            for (int j = 0; j < ny - 1; j++) {
                int v00 = i * ny + j;
                int v01 = v00 + 1;
                int v10 = v00 + ny;
                int v11 = v10 + 1;
                mesh.getFaces().addAll(v00, 0, v10, 0, v11, 0);
                mesh.getFaces().addAll(v00, 0, v11, 0, v01, 0);
                if (twoSided) {
                    mesh.getFaces().addAll(v00, 0, v11, 0, v10, 0);
                    mesh.getFaces().addAll(v00, 0, v01, 0, v11, 0);
                }
            }
        }
        MeshView mv = new MeshView(mesh);
        mv.setMaterial(mat);
        mv.setDrawMode(wireframe ? DrawMode.LINE : DrawMode.FILL);
        plotContent.getChildren().add(mv);
    }

    public static void addContourLinesOnPlane(DataCylinder cylinder,
            IVector<?> xv, IVector<?> yv, IMatrix<?> zmat, double zPlane, int nLevel, Color col) {
        MeshGridHelper.requireZMatchesMeshIj(xv, yv, zmat);
        double z0 = JavaFx3dPlotMath.zMinMatrix(zmat);
        double z1 = JavaFx3dPlotMath.zMaxMatrix(zmat);
        PhongMaterial mat = new PhongMaterial(col);
        for (int l = 1; l < nLevel; l++) {
            double lev = z0 + l * (z1 - z0) / nLevel;
            marchSquareSegments(cylinder, xv, yv, zmat, lev, zPlane, mat);
        }
    }

    private static void marchSquareSegments(DataCylinder cylinder,
            IVector<?> xv, IVector<?> yv, IMatrix<?> z, double level, double zPlane,
            PhongMaterial mat) {
        int nx = xv.length();
        int ny = yv.length();
        for (int i = 0; i < nx - 1; i++) {
            for (int j = 0; j < ny - 1; j++) {
                double z00 = z.get(i, j).doubleValue();
                double z10 = z.get(i + 1, j).doubleValue();
                double z01 = z.get(i, j + 1).doubleValue();
                double z11 = z.get(i + 1, j + 1).doubleValue();
                edgeIso(cylinder, xv.get(i).doubleValue(), yv.get(j).doubleValue(), z00, xv.get(i + 1).doubleValue(), yv.get(j).doubleValue(), z10, level, zPlane, mat);
                edgeIso(cylinder, xv.get(i + 1).doubleValue(), yv.get(j).doubleValue(), z10, xv.get(i + 1).doubleValue(), yv.get(j + 1).doubleValue(), z11, level, zPlane, mat);
                edgeIso(cylinder, xv.get(i + 1).doubleValue(), yv.get(j + 1).doubleValue(), z11, xv.get(i).doubleValue(), yv.get(j + 1).doubleValue(), z01, level, zPlane, mat);
                edgeIso(cylinder, xv.get(i).doubleValue(), yv.get(j + 1).doubleValue(), z01, xv.get(i).doubleValue(), yv.get(j).doubleValue(), z00, level, zPlane, mat);
            }
        }
    }

    private static void edgeIso(DataCylinder cylinder,
            double x0, double y0, double z0, double x1, double y1, double z1, double lev, double zPlane,
            PhongMaterial mat) {
        if ((z0 < lev && z1 < lev) || (z0 >= lev && z1 >= lev)) {
            return;
        }
        double t = (lev - z0) / (z1 - z0 + 1e-15);
        t = JavaFx3dFxUtil.clamp(t, 0, 1);
        double xa = x0 + t * (x1 - x0);
        double ya = y0 + t * (y1 - y0);
        cylinder.add(xa, ya, zPlane, xa + 1e-4, ya + 1e-4, zPlane, 0.02, mat);
    }
}
