package com.yishape.lab.math.linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 统一索引机制测试
 * Tests for unified IndexExpressionParser and the refactored indexing methods
 */
public class IndexExpressionParserTest {

    // ========== IndexExpressionParser 单元测试 ==========

    @Test
    public void testSliceResult_basic() {
        // "1:3" on size=5 → actualStart=1, actualEnd=3
        IndexExpressionParser.SliceResult r = IndexExpressionParser.parse("1:3", 5);
        assertEquals(1, r.start);
        assertEquals(3, r.end);
        assertEquals(1, r.step);
        assertEquals(1, r.actualStart);
        assertEquals(3, r.actualEnd);
    }

    @Test
    public void testSliceResult_negativeIndices() {
        // ":-1" on size=5 → actualStart=0, actualEnd=4
        IndexExpressionParser.SliceResult r = IndexExpressionParser.parse(":-1", 5);
        assertEquals(0, r.actualStart);
        assertEquals(4, r.actualEnd); // exclusive
    }

    @Test
    public void testSliceResult_negativeStart() {
        // "-3:-1" on size=5 → actualStart=2, actualEnd=4
        IndexExpressionParser.SliceResult r = IndexExpressionParser.parse("-3:-1", 5);
        assertEquals(2, r.actualStart);
        assertEquals(4, r.actualEnd);
    }

    @Test
    public void testSliceResult_negativeStep() {
        // "::-1" on size=4 → should resolve to full reverse
        IndexExpressionParser.SliceResult r = IndexExpressionParser.parse("::-1", 4);
        assertEquals(-1, r.step);
        assertEquals(3, r.actualStart); // last element
        assertEquals(-1, r.actualEnd); // sentinel: to beginning
    }

    @Test
    public void testSliceResult_negativeStepWithStart() {
        // "3::-1" on size=4 → start=3, step=-1
        IndexExpressionParser.SliceResult r = IndexExpressionParser.parse("3::-1", 4);
        assertEquals(3, r.actualStart);
        assertEquals(-1, r.actualEnd);
    }

    @Test
    public void testSliceResult_positiveStepEmpty() {
        // "5:2" on size=5 → empty (actualStart > actualEnd after clamping, normalized to equal)
        IndexExpressionParser.SliceResult r = IndexExpressionParser.parse("5:2", 5);
        assertEquals(2, r.actualStart);
        assertEquals(2, r.actualEnd);
    }

    @Test
    public void testGenerateIndices_positiveStep() {
        IndexExpressionParser.SliceResult r = IndexExpressionParser.parse("1:5:2", 10);
        int[] indices = IndexExpressionParser.generateIndices(r);
        assertArrayEquals(new int[]{1, 3}, indices);
    }

    @Test
    public void testGenerateIndices_negativeStep() {
        // "3::-1" on size=4 → [3, 2, 1, 0]
        IndexExpressionParser.SliceResult r = IndexExpressionParser.parse("3::-1", 4);
        int[] indices = IndexExpressionParser.generateIndices(r);
        assertArrayEquals(new int[]{3, 2, 1, 0}, indices);
    }

    @Test
    public void testGenerateIndices_emptySlice() {
        IndexExpressionParser.SliceResult r = IndexExpressionParser.parse("2:2", 5);
        int[] indices = IndexExpressionParser.generateIndices(r);
        assertEquals(0, indices.length);
    }

    @Test
    public void testCalculateSliceSizeLegacy_positiveStep() {
        // "1:5:2" → start=1, end=5, step=1 → 4 elements
        int size = IndexExpressionParser.calculateSliceSizeLegacy(1, 5, 1);
        assertEquals(4, size);
    }

    @Test
    public void testCalculateSliceSizeLegacy_negativeStep() {
        // "::-1" on size=4 → actualStart=3, actualEnd=-1, step=-1 → 4 elements
        int size = IndexExpressionParser.calculateSliceSizeLegacy(3, -1, -1);
        assertEquals(4, size);
    }

    @Test
    public void testCalculateSliceSizeLegacy_negativeStepEmpty() {
        // "2:2" with negative step → empty
        int size = IndexExpressionParser.calculateSliceSizeLegacy(2, 2, -1);
        assertEquals(0, size);
    }

    @Test
    public void testFancyIndex_positive() {
        int[] positions = {0, 2, 4};
        IndexExpressionParser.FancyIndexResult r = IndexExpressionParser.resolveFancyIndex(positions, 5);
        assertArrayEquals(new int[]{0, 2, 4}, r.indices);
    }

    @Test
    public void testFancyIndex_negative() {
        int[] positions = {-1, -2};
        IndexExpressionParser.FancyIndexResult r = IndexExpressionParser.resolveFancyIndex(positions, 5);
        assertArrayEquals(new int[]{4, 3}, r.indices);
    }

    @Test
    public void testFancyIndex_outOfBounds() {
        int[] positions = {5};
        assertThrows(IndexOutOfBoundsException.class, () -> {
            IndexExpressionParser.resolveFancyIndex(positions, 5);
        });
    }

    @Test
    public void testBooleanIndex() {
        boolean[] mask = {true, false, true, false, true};
        IndexExpressionParser.BooleanIndexResult r = IndexExpressionParser.resolveBooleanIndex(mask);
        assertEquals(3, r.count);
        assertArrayEquals(new int[]{0, 2, 4}, r.trueIndices);
    }

    @Test
    public void testBooleanIndex_allFalse() {
        boolean[] mask = {false, false, false};
        IndexExpressionParser.BooleanIndexResult r = IndexExpressionParser.resolveBooleanIndex(mask);
        assertEquals(0, r.count);
    }

    // ========== Vector Slice (String) Tests ==========

    @Test
    public void testVectorSlice_positiveStep() {
        IDoubleVector v = IDoubleVector.of(new double[]{0, 1, 2, 3, 4});
        IVector<Double> result = v.slice("1:4");
        assertEquals(3, result.length());
        assertEquals(1.0, result.get(0));
        assertEquals(3.0, result.get(2));
    }

    @Test
    public void testVectorSlice_negativeStep() {
        // THIS IS THE KEY TEST - previously this would throw, now it should work
        IDoubleVector v = IDoubleVector.of(new double[]{0, 1, 2, 3, 4});
        IVector<Double> result = v.slice("::-1");
        assertEquals(5, result.length());
        assertEquals(4.0, result.get(0));
        assertEquals(0.0, result.get(4));
    }

    @Test
    public void testVectorSlice_negativeStartPositiveEnd() {
        IDoubleVector v = IDoubleVector.of(new double[]{0, 1, 2, 3, 4});
        IVector<Double> result = v.slice("3::-1");
        assertEquals(4, result.length()); // 3,2,1,0
        assertEquals(3.0, result.get(0));
        assertEquals(0.0, result.get(3));
    }

    @Test
    public void testVectorSlice_empty() {
        IDoubleVector v = IDoubleVector.of(new double[]{0, 1, 2, 3, 4});
        IVector<Double> result = v.slice("2:2");
        assertEquals(0, result.length());
    }

    @Test
    public void testVectorSlice_step2() {
        IDoubleVector v = IDoubleVector.of(new double[]{0, 1, 2, 3, 4, 5});
        IVector<Double> result = v.slice("::2");
        assertEquals(3, result.length());
        assertEquals(0.0, result.get(0));
        assertEquals(2.0, result.get(1));
        assertEquals(4.0, result.get(2));
    }

    // ========== Vector Fancy Get/Set Tests ==========

    @Test
    public void testVectorFancyGet_positiveIndices() {
        IDoubleVector v = IDoubleVector.of(new double[]{10, 20, 30, 40, 50});
        IVector<Double> result = v.fancyGet(new int[]{0, 2, 4});
        assertEquals(3, result.length());
        assertEquals(10.0, result.get(0));
        assertEquals(30.0, result.get(1));
        assertEquals(50.0, result.get(2));
    }

    @Test
    public void testVectorFancyGet_negativeIndices() {
        IDoubleVector v = IDoubleVector.of(new double[]{10, 20, 30, 40, 50});
        IVector<Double> result = v.fancyGet(new int[]{-1, -2});
        assertEquals(2, result.length());
        assertEquals(50.0, result.get(0));
        assertEquals(40.0, result.get(1));
    }

    @Test
    public void testVectorFancySet() {
        IDoubleVector v = IDoubleVector.of(new double[]{1, 2, 3, 4, 5});
        v.fancySet(new int[]{0, 2, 4}, new Double[]{10.0, 30.0, 50.0});
        assertEquals(10.0, v.get(0));
        assertEquals(2.0, v.get(1));
        assertEquals(30.0, v.get(2));
        assertEquals(50.0, v.get(4));
    }

    @Test
    public void testVectorFancySetScalar() {
        IDoubleVector v = IDoubleVector.of(new double[]{1, 2, 3, 4, 5});
        v.fancySetScalar(new int[]{1, 3}, 99.0);
        assertEquals(1.0, v.get(0));
        assertEquals(99.0, v.get(1));
        assertEquals(3.0, v.get(2));
        assertEquals(99.0, v.get(3));
    }

    // ========== Vector Boolean Get/Set Tests ==========

    @Test
    public void testVectorBooleanGet() {
        IDoubleVector v = IDoubleVector.of(new double[]{10, 20, 30, 40, 50});
        IVector<Double> result = v.booleanGet(new boolean[]{true, false, true, false, true});
        assertEquals(3, result.length());
        assertEquals(10.0, result.get(0));
        assertEquals(30.0, result.get(1));
        assertEquals(50.0, result.get(2));
    }

    @Test
    public void testVectorBooleanSet() {
        IDoubleVector v = IDoubleVector.of(new double[]{1, 2, 3, 4, 5});
        v.booleanSet(new boolean[]{true, false, true, false, true},
                     new Double[]{10.0, 30.0, 50.0});
        assertEquals(10.0, v.get(0));
        assertEquals(2.0, v.get(1));
        assertEquals(30.0, v.get(2));
        assertEquals(4.0, v.get(3));
        assertEquals(50.0, v.get(4));
    }

    @Test
    public void testVectorBooleanSetScalar() {
        IDoubleVector v = IDoubleVector.of(new double[]{1, 2, 3, 4, 5});
        v.booleanSetScalar(new boolean[]{false, true, false, true, false}, 99.0);
        assertEquals(1.0, v.get(0));
        assertEquals(99.0, v.get(1));
        assertEquals(3.0, v.get(2));
        assertEquals(99.0, v.get(3));
        assertEquals(5.0, v.get(4));
    }

    // ========== Matrix Slice Tests ==========

    @Test
    public void testMatrixSlice_basic() {
        IMatrix<Double> m = IDoubleMatrix.of(new double[][]{
            {1, 2, 3, 4},
            {5, 6, 7, 8},
            {9, 10, 11, 12}
        });
        IMatrix<Double> result = m.slice("1:3", "0:2");
        assertEquals(2, result.rows());
        assertEquals(2, result.cols());
        assertEquals(5.0, result.get(0, 0));
        assertEquals(6.0, result.get(0, 1));
        assertEquals(9.0, result.get(1, 0));
    }

    @Test
    public void testMatrixSlice_negativeStep() {
        IMatrix<Double> m = IDoubleMatrix.of(new double[][]{
            {1, 2, 3, 4},
            {5, 6, 7, 8},
            {9, 10, 11, 12}
        });
        // Reverse both dimensions
        IMatrix<Double> result = m.slice("::-1", "::-1");
        assertEquals(3, result.rows());
        assertEquals(4, result.cols());
        assertEquals(12.0, result.get(0, 0));
        assertEquals(1.0, result.get(2, 3));
    }

    @Test
    public void testMatrixSlice_negativeStepCombined() {
        IMatrix<Double> m = IDoubleMatrix.of(new double[][]{
            {1, 2, 3, 4},
            {5, 6, 7, 8},
            {9, 10, 11, 12}
        });
        // Reverse rows only
        IMatrix<Double> result = m.slice("::-1", ":");
        assertEquals(3, result.rows());
        assertEquals(4, result.cols());
        assertEquals(9.0, result.get(0, 0));
        assertEquals(1.0, result.get(2, 0));
    }

    @Test
    public void testMatrixSlice_empty() {
        IMatrix<Double> m = IDoubleMatrix.of(new double[][]{
            {1, 2, 3},
            {4, 5, 6}
        });
        IMatrix<Double> result = m.slice("2:2", ":");
        assertEquals(0, result.rows());
    }

    @Test
    public void testMatrixSlice_allRows() {
        IMatrix<Double> m = IDoubleMatrix.of(new double[][]{
            {1, 2, 3},
            {4, 5, 6}
        });
        IMatrix<Double> result = m.slice(":", "1:2");
        assertEquals(2, result.rows());
        assertEquals(1, result.cols());
        assertEquals(2.0, result.get(0, 0));
        assertEquals(5.0, result.get(1, 0));
    }

    // ========== Matrix Fancy Get/Set Tests ==========

    @Test
    public void testMatrixFancyGet() {
        IMatrix<Double> m = IDoubleMatrix.of(new double[][]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });
        IMatrix<Double> result = m.fancyGet(new int[]{0, 2}, new int[]{1, 2});
        assertEquals(2, result.rows());
        assertEquals(2, result.cols());
        assertEquals(2.0, result.get(0, 0));
        assertEquals(3.0, result.get(0, 1));
        assertEquals(8.0, result.get(1, 0));
        assertEquals(9.0, result.get(1, 1));
    }

    @Test
    public void testMatrixFancyGet_negativeIndices() {
        IMatrix<Double> m = IDoubleMatrix.of(new double[][]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });
        IMatrix<Double> result = m.fancyGet(new int[]{-1}, new int[]{-1});
        assertEquals(1, result.rows());
        assertEquals(1, result.cols());
        assertEquals(9.0, result.get(0, 0));
    }

    @Test
    public void testMatrixFancySet() {
        IMatrix<Double> m = IDoubleMatrix.of(new double[][]{
            {1, 2, 3},
            {4, 5, 6}
        });
        m.fancySet(new int[]{0, 1}, new int[]{0, 2},
                    new Double[]{99.0, 88.0, 77.0, 66.0});
        assertEquals(99.0, m.get(0, 0));
        assertEquals(2.0, m.get(0, 1));
        assertEquals(88.0, m.get(0, 2));
        assertEquals(77.0, m.get(1, 0));
        assertEquals(66.0, m.get(1, 2));
    }

    @Test
    public void testMatrixFancySetScalar() {
        IMatrix<Double> m = IDoubleMatrix.of(new double[][]{
            {1, 2, 3},
            {4, 5, 6}
        });
        m.fancySetScalar(new int[]{0}, new int[]{1, 2}, 99.0);
        assertEquals(1.0, m.get(0, 0));
        assertEquals(99.0, m.get(0, 1));
        assertEquals(99.0, m.get(0, 2));
    }

    // ========== Matrix Boolean Get Tests ==========

    @Test
    public void testMatrixBooleanGet_rowMask() {
        IMatrix<Double> m = IDoubleMatrix.of(new double[][]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });
        IMatrix<Double> result = m.booleanGet(new boolean[]{true, false, true});
        assertEquals(2, result.rows());
        assertEquals(3, result.cols());
        assertEquals(1.0, result.get(0, 0));
        assertEquals(7.0, result.get(1, 0));
    }

    @Test
    public void testMatrixBooleanGet_rowAndColMask() {
        IMatrix<Double> m = IDoubleMatrix.of(new double[][]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });
        IMatrix<Double> result = m.booleanGet(
            new boolean[]{true, false, true},
            new boolean[]{false, true, false}
        );
        assertEquals(2, result.rows());
        assertEquals(1, result.cols());
        assertEquals(2.0, result.get(0, 0));
        assertEquals(8.0, result.get(1, 0));
    }

    @Test
    public void testMatrixBooleanGet_allFalse() {
        IMatrix<Double> m = IDoubleMatrix.of(new double[][]{
            {1, 2, 3},
            {4, 5, 6}
        });
        IMatrix<Double> result = m.booleanGet(new boolean[]{false, false});
        assertEquals(0, result.rows());
    }

    // ========== Empty Matrix/Vector Edge Cases ==========

    @Test
    public void testEmptyVectorSlice() {
        IDoubleVector v = IDoubleVector.of(new double[]{});
        IVector<Double> result = v.slice(":");
        assertEquals(0, result.length());
    }

    @Test
    public void testEmptyMatrixSlice() {
        IMatrix<Double> m = IDoubleMatrix.of(new double[0][0]);
        IMatrix<Double> result = m.slice(":2", ":2");
        assertEquals(0, result.rows());
    }

    @Test
    public void testVectorSlice_outOfBounds() {
        IDoubleVector v = IDoubleVector.of(new double[]{0, 1, 2, 3, 4});
        // This should return empty slice after clamping, not throw
        IVector<Double> result = v.slice("10:20");
        assertEquals(0, result.length());
    }

    // ========== Float Vector Tests ==========

    @Test
    public void testFloatVectorSlice_negativeStep() {
        IFloatVector v = IFloatVector.of(new float[]{0, 1, 2, 3, 4});
        IVector<Float> result = v.slice("::-1");
        assertEquals(5, result.length());
        assertEquals(4.0f, result.get(0));
        assertEquals(0.0f, result.get(4));
    }

    @Test
    public void testFloatVectorFancyGet() {
        IFloatVector v = IFloatVector.of(new float[]{10, 20, 30, 40, 50});
        IVector<Float> result = v.fancyGet(new int[]{-1, -2});
        assertEquals(2, result.length());
        assertEquals(50.0f, result.get(0));
        assertEquals(40.0f, result.get(1));
    }

    @Test
    public void testFloatVectorBooleanGet() {
        IFloatVector v = IFloatVector.of(new float[]{10, 20, 30, 40, 50});
        IVector<Float> result = v.booleanGet(new boolean[]{true, false, true, false, true});
        assertEquals(3, result.length());
        assertEquals(10.0f, result.get(0));
        assertEquals(30.0f, result.get(1));
        assertEquals(50.0f, result.get(2));
    }

    // ========== Float Matrix Tests ==========

    @Test
    public void testFloatMatrixSlice_negativeStep() {
        IMatrix<Float> m = IFloatMatrix.of(new float[][]{
            {1, 2, 3},
            {4, 5, 6}
        });
        IMatrix<Float> result = m.slice("::-1", "::-1");
        assertEquals(2, result.rows());
        assertEquals(3, result.cols());
        assertEquals(6.0f, result.get(0, 0));
        assertEquals(1.0f, result.get(1, 2));
    }

    @Test
    public void testFloatMatrixBooleanGet() {
        IMatrix<Float> m = IFloatMatrix.of(new float[][]{
            {1, 2, 3},
            {4, 5, 6}
        });
        IMatrix<Float> result = m.booleanGet(new boolean[]{true, false});
        assertEquals(1, result.rows());
        assertEquals(3, result.cols());
        assertEquals(1.0f, result.get(0, 0));
    }

    @Test
    public void testFloatMatrixFancyGet() {
        IMatrix<Float> m = IFloatMatrix.of(new float[][]{
            {1, 2, 3},
            {4, 5, 6}
        });
        IMatrix<Float> result = m.fancyGet(new int[]{0, 1}, new int[]{0, 2});
        assertEquals(2, result.rows());
        assertEquals(2, result.cols());
        assertEquals(1.0f, result.get(0, 0));
        assertEquals(3.0f, result.get(0, 1));
        assertEquals(6.0f, result.get(1, 1));
    }
}
