package com.yishape.lab.math.linalg;

import com.yishape.lab.math.RereMathUtil;
import com.yishape.lab.util.Tuple2;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link Linalg} 与矩阵/向量上数组风格 API 的回归测试（插值、网格、堆叠、逐元素工具等）。
 */
class LinalgArrayOpsTest {

    @Test
    void isCloseAndAllClose() {
        assertTrue(RereMathUtil.isClose(1.0, 1.0 + 1e-10));
        assertTrue(RereMathUtil.allClose(new double[]{1, 2}, new double[]{1 + 1e-10, 2 - 1e-10}));
    }

    @Test
    void interpClampsAndInterpolates() {
        IVector<Double> xp = Linalg.vector(new double[]{0, 1, 2});
        IVector<Double> fp = Linalg.vector(new double[]{10, 20, 30});
        IVector<Double> xq = Linalg.vector(new double[]{-1, 0.5, 1.5, 3});
        IVector<Double> y = Linalg.interp(xq, xp, fp);
        assertEquals(10.0, y.get(0), 1e-12);
        assertEquals(15.0, y.get(1), 1e-12);
        assertEquals(25.0, y.get(2), 1e-12);
        assertEquals(30.0, y.get(3), 1e-12);
    }

    @Test
    void meshgridIndexingXy() {
        IVector<Double> x = Linalg.vector(new double[]{1, 2});
        IVector<Double> y = Linalg.vector(new double[]{10, 20, 30});
        Tuple2<IMatrix<Double>, IMatrix<Double>> g = Linalg.meshgrid(x, y);
        IMatrix<Double> X = g._1;
        IMatrix<Double> Y = g._2;
        assertEquals(3, X.rows());
        assertEquals(2, X.cols());
        assertEquals(1.0, X.get(0, 0), 1e-12);
        assertEquals(2.0, X.get(0, 1), 1e-12);
        assertEquals(10.0, Y.get(0, 0), 1e-12);
        assertEquals(10.0, Y.get(0, 1), 1e-12);
        assertEquals(30.0, Y.get(2, 0), 1e-12);
    }

    @Test
    void stackAxis0And1() {
        IMatrix<Double> a = Linalg.matrix(new double[][]{{1, 2}});
        IMatrix<Double> b = Linalg.matrix(new double[][]{{3, 4}});
        IMatrix<Double> v = Linalg.stack(0, a, b);
        assertEquals(2, v.rows());
        assertEquals(2, v.cols());
        assertEquals(3.0, v.get(1, 0), 1e-12);
        IMatrix<Double> h = Linalg.stack(1, a, b);
        assertEquals(1, h.rows());
        assertEquals(4, h.cols());
        assertEquals(3.0, h.get(0, 2), 1e-12);
    }

    @Test
    void multiDotChainsMmul() {
        IMatrix<Double> a = Linalg.matrix(new double[][]{{1, 2}, {3, 4}});
        IMatrix<Double> b = Linalg.matrix(new double[][]{{2, 0}, {0, 2}});
        IMatrix<Double> c = Linalg.multiDot(a, b, a);
        IMatrix<Double> expected = a.mmul(b).mmul(a);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(expected.get(i, j), c.get(i, j), 1e-10);
            }
        }
    }

    @Test
    void vectorCrossProduct3D() {
        IDoubleVector u = IDoubleVector.of(new double[]{1, 0, 0});
        IDoubleVector v = IDoubleVector.of(new double[]{0, 1, 0});
        IVector<Double> w = u.cross(v);
        assertEquals(0.0, w.get(0), 1e-12);
        assertEquals(0.0, w.get(1), 1e-12);
        assertEquals(1.0, w.get(2), 1e-12);
    }

    @Test
    void vectorSearchSorted() {
        IDoubleVector a = IDoubleVector.of(new double[]{1, 3, 3, 7});
        assertEquals(0, a.searchSorted(0.5));
        assertEquals(1, a.searchSorted(2.0));
        assertEquals(1, a.searchSorted(3.0));
        assertEquals(4, a.searchSorted(9.0));
    }

    @Test
    void matrixClipFlipRot90RollPadTileWhere() {
        IMatrix<Double> m = Linalg.matrix(new double[][]{{-5, 2, 8}, {1, 2, 3}});
        IMatrix<Double> c = m.clip(0.0, 5.0);
        assertEquals(0.0, c.get(0, 0), 1e-12);
        assertEquals(5.0, c.get(0, 2), 1e-12);
        assertEquals(1.0, m.flipAxis(0).get(0, 0), 1e-12);
        assertEquals(8.0, m.flipAxis(1).get(0, 0), 1e-12);
        IMatrix<Double> r = Linalg.matrix(new double[][]{{1, 2}, {3, 4}});
        IMatrix<Double> r90 = r.rot90(1);
        assertEquals(2.0, r90.get(0, 0), 1e-12);
        assertEquals(4.0, r90.get(0, 1), 1e-12);
        assertEquals(3.0, r.roll(1, 0).get(0, 0), 1e-12);
        assertEquals(2.0, r.roll(-1, 1).get(0, 0), 1e-12);
        IMatrix<Double> p = r.pad(1, 0, 1, 0, 0.0);
        assertEquals(3, p.rows());
        assertEquals(3, p.cols());
        assertEquals(0.0, p.get(0, 0), 1e-12);
        assertEquals(1.0, p.get(1, 1), 1e-12);
        IMatrix<Double> t = r.tile(2, 2);
        assertEquals(4, t.rows());
        assertEquals(4, t.cols());
        assertEquals(4.0, t.get(1, 1), 1e-12);
        boolean[][] cond = {{true, false}, {false, true}};
        IMatrix<Double> w = r.where(cond, 9.0, -1.0);
        assertEquals(9.0, w.get(0, 0), 1e-12);
        assertEquals(-1.0, w.get(0, 1), 1e-12);
    }

    @Test
    void rot90EmptyMatrixNoThrow() {
        IMatrix<Double> z = Linalg.zeros(0, 0);
        assertEquals(0, z.rot90(2).rows());
    }

    @Test
    void rot90EmptyFloatMatrixCopyNoThrow() {
        IMatrix<Float> z = IFloatMatrix.zeros(0, 0);
        assertEquals(0, z.copy().rows());
        assertEquals(0, z.rot90(1).rows());
    }

    @Test
    void matrixPowerAndSlogdetAndEigh() {
        IMatrix<Double> a = Linalg.matrix(new double[][]{{2, 0}, {0, 3}});
        IMatrix<Double> a3 = a.matrixPower(3);
        assertEquals(8.0, a3.get(0, 0), 1e-10);
        assertEquals(27.0, a3.get(1, 1), 1e-10);
        Tuple2<Double, Double> sl = a.slogdet();
        assertEquals(1.0, sl._1, 1e-12);
        assertEquals(Math.log(6.0), sl._2, 1e-10);
        IMatrix<Double> sym = Linalg.matrix(new double[][]{{2, 1}, {1, 2}});
        var e1 = sym.eigen();
        var e2 = sym.eigh();
        assertEquals(e1._1.length(), e2._1.length());
        assertThrows(IllegalArgumentException.class, () -> Linalg.matrix(new double[][]{{1, 2}, {3, 4}}).eigh());
    }
}
