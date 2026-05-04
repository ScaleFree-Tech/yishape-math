package com.yishape.lab.math.plot;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.util.Tuple3;
import java.util.Objects;
import java.util.function.DoubleBinaryOperator;

/**
 * Meshgrid 与张量采样网格工具，供 {@link I3dPlot} 及曲面/场图实现使用。
 * <p>
 * 优先通过 {@link IVector}、{@link IMatrix} 的 {@code reshape}、{@code asColumnVector}、{@code tile}、
 * {@code flatten}、{@code concat} 等完成构造，避免手写裸循环（标量场采样等少量索引除外）。
 * </p>
 *
 * <p><b>索引约定 / Indexing</b></p>
 * <ul>
 *   <li>{@link MeshIndexing#XY}：与 NumPy {@code meshgrid(..., indexing='xy')} 及 {@code com.yishape.lab.math.linalg.Linq#meshgrid}
 *       一致，网格形状为 {@code y.len × x.len}，点 {@code (i,j)} 对应坐标 {@code (x[j], y[i])}。</li>
 *   <li>{@link MeshIndexing#IJ}：与 {@code indexing='ij'} 一致，且与 {@link I3dPlot} 中 {@code z} 的约定一致：
 *       形状为 {@code x.len × y.len}，{@code z[i][j]} 对应 {@code (x[i], y[j])}。</li>
 * </ul>
 *
 * @author lteb2
 */
public final class MeshGridHelper {

    /** 与 NumPy meshgrid indexing 对齐的两种约定。 */
    public enum MeshIndexing {
        /** 笛卡尔“先横后纵”展示常用；矩阵行数 = len(y)。 */
        XY,
        /** 矩阵行数 = len(x)，列数 = len(y)，与 I3dPlot 曲面 Z 默认布局一致。 */
        IJ
    }

    private MeshGridHelper() {}

    /**
     * 默认 {@link MeshIndexing#XY}，等价于 NumPy {@code meshgrid(x,y)}。
     */
    public static <T extends Number> Tuple2<IMatrix<T>, IMatrix<T>> meshgrid(IVector<T> x, IVector<T> y) {
        return meshgrid(x, y, MeshIndexing.XY);
    }

    public static <T extends Number> Tuple2<IMatrix<T>, IMatrix<T>> meshgrid(
            IVector<T> x, IVector<T> y, MeshIndexing indexing) {
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(y, "y");
        Objects.requireNonNull(indexing, "indexing");
        return indexing == MeshIndexing.XY ? meshgridXy(x, y) : meshgridIj(x, y);
    }

    /**
     * {@code indexing='xy'}：{@code X[i,j]=x[j]}，{@code Y[i,j]=y[i]}，形状 {@code (len(y), len(x))}。
     */
    public static <T extends Number> Tuple2<IMatrix<T>, IMatrix<T>> meshgridXy(IVector<T> x, IVector<T> y) {
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(y, "y");
        int nx = x.length();
        int ny = y.length();
        if (nx == 0 || ny == 0) {
            Class<T> t = combinedElementClass2(x, y);
            return new Tuple2<>(IMatrix.zeros(ny, nx, t), IMatrix.zeros(ny, nx, t));
        }
        IMatrix<T> xx = x.reshape(1, nx).tile(ny, 1);
        IMatrix<T> yy = y.asColumnVector().tile(1, nx);
        return new Tuple2<>(xx, yy);
    }

    /**
     * {@code indexing='ij'}：{@code X[i,j]=x[i]}，{@code Y[i,j]=y[j]}，形状 {@code (len(x), len(y))}。
     */
    public static <T extends Number> Tuple2<IMatrix<T>, IMatrix<T>> meshgridIj(IVector<T> x, IVector<T> y) {
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(y, "y");
        int nx = x.length();
        int ny = y.length();
        if (nx == 0 || ny == 0) {
            Class<T> t = combinedElementClass2(x, y);
            return new Tuple2<>(IMatrix.zeros(nx, ny, t), IMatrix.zeros(nx, ny, t));
        }
        IMatrix<T> xx = x.asColumnVector().tile(1, ny);
        IMatrix<T> yy = y.reshape(1, ny).tile(nx, 1);
        return new Tuple2<>(xx, yy);
    }

    /**
     * 三维规则网格（{@code indexing='ij'}）：返回与 C-order 展平顺序一致的三条坐标向量，长度均为 {@code len(x)*len(y)*len(z)}，
     * 展平顺序为 {@code i(最慢) → j → k(最快)}，即 {@code idx = i*(ny*nz) + j*nz + k}。
     */
    public static <T extends Number> Tuple3<IVector<T>, IVector<T>, IVector<T>> meshgrid3RavelIj(
            IVector<T> x, IVector<T> y, IVector<T> z) {
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(y, "y");
        Objects.requireNonNull(z, "z");
        int nx = x.length();
        int ny = y.length();
        int nz = z.length();
        if (nx == 0 || ny == 0 || nz == 0) {
            Class<T> tc = combinedElementClass(x, y, z);
            IVector<T> ez = IVector.zeros(0, tc);
            return new Tuple3<>(ez, ez, ez);
        }
        Class<T> type = combinedElementClass(x, y, z);
        IVector<T> accX = null;
        IVector<T> accY = null;
        IVector<T> accZ = null;
        for (int k = 0; k < nz; k++) {
            Tuple2<IMatrix<T>, IMatrix<T>> xy = meshgridIj(x, y);
            IVector<T> fx = xy._1.flatten();
            IVector<T> fy = xy._2.flatten();
            IMatrix<T> zk = IMatrix.ones(nx, ny, type).multiplyScalar(z.get(k));
            IVector<T> fz = zk.flatten();
            accX = accX == null ? fx : accX.concat(fx);
            accY = accY == null ? fy : accY.concat(fy);
            accZ = accZ == null ? fz : accZ.concat(fz);
        }
        return new Tuple3<>(accX, accY, accZ);
    }

    /**
     * 将二维 mesh 展平为 {@code N×2} 点集（行优先与 {@link IMatrix#flatten()} 一致）。
     */
    public static <T extends Number> IMatrix<T> toPointCloud2(Tuple2<IMatrix<T>, IMatrix<T>> mesh) {
        Objects.requireNonNull(mesh, "mesh");
        Objects.requireNonNull(mesh._1, "mesh._1");
        Objects.requireNonNull(mesh._2, "mesh._2");
        IMatrix<T> c0 = mesh._1.flatten().asColumnVector();
        IMatrix<T> c1 = mesh._2.flatten().asColumnVector();
        return c0.hstack(c1);
    }

    /**
     * 将三维 ravel 网格转为 {@code N×3} 点集。
     */
    public static <T extends Number> IMatrix<T> toPointCloud3(Tuple3<IVector<T>, IVector<T>, IVector<T>> ravel) {
        Objects.requireNonNull(ravel, "ravel");
        IVector<T> xs = Objects.requireNonNull(ravel._1, "ravel._1");
        IVector<T> ys = Objects.requireNonNull(ravel._2, "ravel._2");
        IVector<T> zs = Objects.requireNonNull(ravel._3, "ravel._3");
        if (xs.length() != ys.length() || xs.length() != zs.length()) {
            throw new IllegalArgumentException(
                    "meshgrid3 ravel 坐标须等长 / meshgrid3 ravel axes must have equal length");
        }
        return xs.asColumnVector().hstack(ys.asColumnVector()).hstack(zs.asColumnVector());
    }

    /**
     * 校验 {@code Z} 是否与 {@link MeshIndexing#IJ} 及 {@link I3dPlot} 约定一致：{@code Z.rows()==len(x)} 且 {@code Z.cols()==len(y)}。
     */
    public static void requireZMatchesMeshIj(IVector<?> x, IVector<?> y, IMatrix<?> z) {
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(y, "y");
        Objects.requireNonNull(z, "z");
        if (z.rows() != x.length() || z.cols() != y.length()) {
            throw new IllegalArgumentException(
                    String.format(
                            "Z 形状须为 %d×%d (ij，与 x、y 对齐)，实际 %d×%d / Z must be %d×%d for ij indexing, got %d×%d",
                            x.length(),
                            y.length(),
                            z.rows(),
                            z.cols(),
                            x.length(),
                            y.length(),
                            z.rows(),
                            z.cols()));
        }
    }

    /**
     * 校验 {@code Z} 是否与 {@link MeshIndexing#XY} 一致：{@code Z.rows()==len(y)} 且 {@code Z.cols()==len(x)}。
     */
    public static void requireZMatchesMeshXy(IVector<?> x, IVector<?> y, IMatrix<?> z) {
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(y, "y");
        Objects.requireNonNull(z, "z");
        if (z.rows() != y.length() || z.cols() != x.length()) {
            throw new IllegalArgumentException(
                    String.format(
                            "Z 形状须为 %d×%d (xy，与 y、x 对齐)，实际 %d×%d / Z must be %d×%d for xy indexing, got %d×%d",
                            y.length(),
                            x.length(),
                            z.rows(),
                            z.cols(),
                            y.length(),
                            x.length(),
                            z.rows(),
                            z.cols()));
        }
    }

    /**
     * 在 mesh 上对二元函数采样得到 {@code Z}（{@link MeshIndexing#IJ} 时形状 {@code len(x)×len(y)}；{@code XY} 时为 {@code len(y)×len(x)}）。
     */
    public static IMatrix<Double> sampleBinary(
            IVector<Double> x, IVector<Double> y, MeshIndexing indexing, DoubleBinaryOperator f) {
        Objects.requireNonNull(f, "f");
        Tuple2<IMatrix<Double>, IMatrix<Double>> g = meshgrid(x, y, indexing);
        IMatrix<Double> gx = g._1;
        IMatrix<Double> gy = g._2;
        int r = gx.rows();
        int c = gx.cols();
        IMatrix<Double> out = IMatrix.zeros(r, c);
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                out.set(i, j, f.applyAsDouble(gx.get(i, j), gy.get(i, j)));
            }
        }
        return out;
    }

    /**
     * 将按 {@code xy} 布局采样的 {@code Z}（{@code len(y)×len(x)}）转为 {@code ij} 布局（{@code len(x)×len(y)}），仅交换轴索引，不修改数值语义。
     * <p>若原数据在几何上按 {@code (x[j],y[i])} 采样，转置后第 {@code i} 行第 {@code j} 列对应原 {@code (j,i)} 位置。</p>
     */
    public static <T extends Number> IMatrix<T> zXyToIjLayout(IMatrix<T> zXy) {
        Objects.requireNonNull(zXy, "zXy");
        return zXy.transpose();
    }

    @SuppressWarnings("unchecked")
    private static <T extends Number> Class<T> elementClass(IVector<T> v) {
        if (v.length() == 0) {
            return (Class<T>) Double.class;
        }
        Number z = v.get(0);
        if (z instanceof Float) {
            return (Class<T>) Float.class;
        }
        if (z instanceof Double) {
            return (Class<T>) Double.class;
        }
        throw new IllegalArgumentException(
                "仅支持 Float/Double meshgrid 元素类型 / Only Float/Double mesh elements supported: "
                        + z.getClass());
    }

    @SuppressWarnings("unchecked")
    private static <T extends Number> Class<T> combinedElementClass(IVector<T> x, IVector<T> y, IVector<T> z) {
        Class<T> cx = elementClass(x);
        Class<T> cy = elementClass(y);
        Class<T> cz = elementClass(z);
        if (cx == Float.class || cy == Float.class || cz == Float.class) {
            return (Class<T>) Float.class;
        }
        return (Class<T>) Double.class;
    }

    @SuppressWarnings("unchecked")
    private static <T extends Number> Class<T> combinedElementClass2(IVector<T> a, IVector<T> b) {
        Class<T> ca = elementClass(a);
        Class<T> cb = elementClass(b);
        if (ca == Float.class || cb == Float.class) {
            return (Class<T>) Float.class;
        }
        return (Class<T>) Double.class;
    }
}
