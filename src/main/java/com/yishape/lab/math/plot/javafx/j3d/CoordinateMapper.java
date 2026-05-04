package com.yishape.lab.math.plot.javafx.j3d;

import javafx.geometry.Point3D;

/**
 * 3D坐标系转换工具类，统一处理数据坐标系到JavaFX场景坐标系的转换。
 * <p>
 * 坐标系约定：
 * <ul>
 *   <li>数据坐标系 (Data): (X, Y, Z) - 用户数据空间</li>
 *   <li>JavaFX坐标系 (Scene): (X, Z, Y) - 3D场景空间，Y轴向上</li>
 *   <li>世界坐标系 (World): 经过缩放和平移后的标准化坐标</li>
 * </ul>
 * <p>
 * 此类提供统一的转换方法，消除代码中分散的坐标转换逻辑。
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public final class CoordinateMapper {

    private CoordinateMapper() {
        // 工具类，禁止实例化
    }

    /**
     * 将数据坐标转换为JavaFX场景坐标。
     * 核心转换：数据 (x, y, z) → 场景 (x, z, y)
     *
     * @param x 数据X坐标
     * @param y 数据Y坐标
     * @param z 数据Z坐标
     * @return JavaFX场景中的Point3D (X, Z, Y)
     */
    public static Point3D dataToScene(double x, double y, double z) {
        return new Point3D(x, z, y);
    }

    /**
     * 将数据坐标转换为JavaFX场景坐标（使用数组）。
     *
     * @param dataPoint 数据点数组 [x, y, z]
     * @return JavaFX场景中的Point3D
     */
    public static Point3D dataToScene(double[] dataPoint) {
        if (dataPoint == null || dataPoint.length < 3) {
            throw new IllegalArgumentException("数据点必须包含至少3个坐标值");
        }
        return new Point3D(dataPoint[0], dataPoint[2], dataPoint[1]);
    }

    /**
     * 将JavaFX场景坐标转换回数据坐标。
     * 逆向转换：场景 (x, z, y) → 数据 (x, y, z)
     *
     * @param scenePoint JavaFX场景坐标
     * @return 数据坐标数组 [x, y, z]
     */
    public static double[] sceneToData(Point3D scenePoint) {
        return new double[]{scenePoint.getX(), scenePoint.getZ(), scenePoint.getY()};
    }

    /**
     * 提取数据坐标并转换为JavaFX场景的平移值。
     * 用于设置Node的translateX, translateY, translateZ。
     *
     * @param x 数据X
     * @param y 数据Y
     * @param z 数据Z
     * @return 平移值数组 [translateX, translateY, translateZ]
     */
    public static double[] dataToTranslation(double x, double y, double z) {
        return new double[]{x, z, y};
    }

    /**
     * 将两点之间的数据坐标连接转换为场景坐标连接。
     * 用于绘制线段/箭头。
     *
     * @param x0 起点数据X
     * @param y0 起点数据Y
     * @param z0 起点数据Z
     * @param x1 终点数据X
     * @param y1 终点数据Y
     * @param z1 终点数据Z
     * @return 场景坐标点对 [p0, p1]
     */
    public static Point3D[] dataLineToScene(double x0, double y0, double z0,
                                               double x1, double y1, double z1) {
        return new Point3D[]{
                dataToScene(x0, y0, z0),
                dataToScene(x1, y1, z1)
        };
    }

    /**
     * 计算数据坐标下的范围（边界框）。
     *
     * @param points 数据点列表，每个点是[x, y, z]
     * @return 边界数组 [minX, maxX, minY, maxY, minZ, maxZ]
     */
    public static double[] calculateBounds(java.util.List<double[]> points) {
        if (points == null || points.isEmpty()) {
            return new double[]{0, 1, 0, 1, 0, 1};
        }

        double minX = Double.POSITIVE_INFINITY, maxX = Double.NEGATIVE_INFINITY;
        double minY = Double.POSITIVE_INFINITY, maxY = Double.NEGATIVE_INFINITY;
        double minZ = Double.POSITIVE_INFINITY, maxZ = Double.NEGATIVE_INFINITY;

        for (double[] p : points) {
            if (p == null || p.length < 3) continue;
            minX = Math.min(minX, p[0]);
            maxX = Math.max(maxX, p[0]);
            minY = Math.min(minY, p[1]);
            maxY = Math.max(maxY, p[1]);
            minZ = Math.min(minZ, p[2]);
            maxZ = Math.max(maxZ, p[2]);
        }

        // 处理退化情况
        if (!Double.isFinite(minX) || maxX <= minX) { minX = 0; maxX = 1; }
        if (!Double.isFinite(minY) || maxY <= minY) { minY = 0; maxY = 1; }
        if (!Double.isFinite(minZ) || maxZ <= minZ) { minZ = 0; maxZ = 1; }

        return new double[]{minX, maxX, minY, maxY, minZ, maxZ};
    }

    /**
     * 计算世界变换参数（统一缩放和中心平移）。
     *
     * @param bounds 边界数组 [minX, maxX, minY, maxY, minZ, maxZ]
     * @param targetExtent 目标场景尺寸（如220）
     * @return 变换参数数组 [centerX, centerY, centerZ, scale]
     */
    public static double[] calculateWorldTransform(double[] bounds, double targetExtent) {
        if (bounds == null || bounds.length < 6) {
            return new double[]{0, 0, 0, 1};
        }

        double cx = (bounds[0] + bounds[1]) / 2;
        double cy = (bounds[2] + bounds[3]) / 2;
        double cz = (bounds[4] + bounds[5]) / 2;

        double ex = Math.max(1e-9, bounds[1] - bounds[0]);
        double ey = Math.max(1e-9, bounds[3] - bounds[2]);
        double ez = Math.max(1e-9, bounds[5] - bounds[4]);
        double ext = Math.max(ex, Math.max(ey, ez));

        double scale = targetExtent / ext;

        return new double[]{cx, cy, cz, scale};
    }

    /**
     * 将数据坐标应用世界变换（用于世界缩放后的坐标）。
     *
     * @param x 数据X
     * @param y 数据Y
     * @param z 数据Z
     * @param centerX 世界中心X
     * @param centerY 世界中心Y
     * @param centerZ 世界中心Z
     * @param scale 世界缩放比例
     * @return 世界坐标数组 [wx, wy, wz]
     */
    public static double[] dataToWorld(double x, double y, double z,
                                          double centerX, double centerY, double centerZ,
                                          double scale) {
        return new double[]{
                (x - centerX) * scale,
                (y - centerY) * scale,
                (z - centerZ) * scale
        };
    }

    /**
     * 安全地进行数值钳制。
     *
     * @param value 输入值
     * @param min 最小值
     * @param max 最大值
     * @return 钳制后的值
     */
    public static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    /**
     * 计算向量长度（数据空间）。
     *
     * @param dx X分量
     * @param dy Y分量
     * @param dz Z分量
     * @return 向量长度
     */
    public static double vectorLength(double dx, double dy, double dz) {
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    /**
     * 归一化向量。
     *
     * @param v 输入向量 [x, y, z]
     * @return 归一化后的向量
     */
    public static double[] normalize(double[] v) {
        if (v == null || v.length < 3) return new double[]{0, 0, 0};
        double len = vectorLength(v[0], v[1], v[2]);
        if (len < 1e-12) return new double[]{0, 0, 0};
        return new double[]{v[0] / len, v[1] / len, v[2] / len};
    }
}
