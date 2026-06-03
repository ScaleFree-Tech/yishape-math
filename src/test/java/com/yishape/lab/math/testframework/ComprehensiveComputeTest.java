package com.yishape.lab.math.testframework;

import com.yishape.lab.math.compute.ComputerConfig;
import com.yishape.lab.math.compute.DoubleVectorComputer;
import com.yishape.lab.math.compute.FloatVectorComputer;
import com.yishape.lab.math.compute.IFloatVectorComputer;
import com.yishape.lab.math.compute.hpc.HpcOptionalRuntime;
import com.yishape.lab.math.compute.ops.BinaryOperation;
import com.yishape.lab.math.compute.ops.BinaryReduceOperation;
import com.yishape.lab.math.compute.ops.LogicalCompare;
import com.yishape.lab.math.compute.ops.ReduceOperation;
import com.yishape.lab.math.compute.ops.UniversalOperation;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;


import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test class for com.yishape.lab.math.compute package.
 * Covers DoubleVectorComputer, FloatVectorComputer, ComputerConfig,
 * HpcOptionalRuntime, and all matrix/vector operations.
 */
public class ComprehensiveComputeTest {

    private final DoubleVectorComputer doubleComputer = new DoubleVectorComputer();
    private final FloatVectorComputer floatComputer = new FloatVectorComputer();

    // ==================== Helper Methods ====================

    private static void assertArrayEqualsDouble(double[] expected, double[] actual, double delta, String message) {
        assertEquals(expected.length, actual.length, message + " - length mismatch");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], delta, message + " - element " + i);
        }
    }

    private static void assertMatrixEqualsDouble(double[][] expected, double[][] actual, double delta, String message) {
        assertEquals(expected.length, actual.length, message + " - row count mismatch");
        for (int i = 0; i < expected.length; i++) {
            assertArrayEqualsDouble(expected[i], actual[i], delta, message + " - row " + i);
        }
    }

    private static void assertArrayEqualsFloat(float[] expected, float[] actual, float delta, String message) {
        assertEquals(expected.length, actual.length, message + " - length mismatch");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], delta, message + " - element " + i);
        }
    }

    private static void assertMatrixEqualsFloat(float[][] expected, float[][] actual, float delta, String message) {
        assertEquals(expected.length, actual.length, message + " - row count mismatch");
        for (int i = 0; i < expected.length; i++) {
            assertArrayEqualsFloat(expected[i], actual[i], delta, message + " - row " + i);
        }
    }

    // ==================== DoubleVectorComputer Tests ====================

    @Test
    @Timeout(value = 10)
    public void testDoubleBinaryOperateAdd() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        double[] x1 = {1.0, 2.0, 3.0};
        double[] x2 = {4.0, 5.0, 6.0};
        double[] result = doubleComputer.binaryOperate(x1, x2, BinaryOperation.ADD);

        TestResult r = recorder.record("simd", "double_binary_add");
        try {
            assertArrayEqualsDouble(new double[]{5.0, 7.0, 9.0}, result, 1e-10, "Double ADD");
            r.pass("Double ADD: [1,2,3]+[4,5,6]=[5,7,9]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDoubleBinaryOperateSubtract() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        double[] x1 = {5.0, 3.0, 1.0};
        double[] x2 = {2.0, 2.0, 2.0};
        double[] result = doubleComputer.binaryOperate(x1, x2, BinaryOperation.SUBTRACT);

        TestResult r = recorder.record("simd", "double_binary_subtract");
        try {
            assertArrayEqualsDouble(new double[]{3.0, 1.0, -1.0}, result, 1e-10, "Double SUBTRACT");
            r.pass("Double SUBTRACT: [5,3,1]-[2,2,2]=[3,1,-1]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDoubleBinaryOperateMultiply() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        double[] x1 = {2.0, 3.0, 4.0};
        double[] x2 = {1.0, 2.0, 3.0};
        double[] result = doubleComputer.binaryOperate(x1, x2, BinaryOperation.MULTIPLY);

        TestResult r = recorder.record("simd", "double_binary_multiply");
        try {
            assertArrayEqualsDouble(new double[]{2.0, 6.0, 12.0}, result, 1e-10, "Double MULTIPLY");
            r.pass("Double MULTIPLY: [2,3,4]*[1,2,3]=[2,6,12]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDoubleBinaryOperateDivide() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        double[] x1 = {10.0, 20.0, 30.0};
        double[] x2 = {2.0, 5.0, 10.0};
        double[] result = doubleComputer.binaryOperate(x1, x2, BinaryOperation.DIVIDE);

        TestResult r = recorder.record("simd", "double_binary_divide");
        try {
            assertArrayEqualsDouble(new double[]{5.0, 4.0, 3.0}, result, 1e-10, "Double DIVIDE");
            r.pass("Double DIVIDE: [10,20,30]/[2,5,10]=[5,4,3]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDoubleScalarOperation() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        double[] x1 = {1.0, 2.0, 3.0};
        double scalar = 2.0;
        double[] result = doubleComputer.binaryOperate(x1, scalar, BinaryOperation.MULTIPLY);

        TestResult r = recorder.record("simd", "double_scalar_multiply");
        try {
            assertArrayEqualsDouble(new double[]{2.0, 4.0, 6.0}, result, 1e-10, "Double scalar multiply");
            r.pass("Double scalar MULTIPLY: [1,2,3]*2=[2,4,6]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDoubleReduceOperateSum() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        double[] x = {1.0, 2.0, 3.0, 4.0, 5.0};
        double result = doubleComputer.reduceOperate(x, ReduceOperation.SUM);

        TestResult r = recorder.record("simd", "double_reduce_sum");
        try {
            assertEquals(15.0, result, 1e-10, "Double SUM");
            r.pass("Double SUM: sum([1,2,3,4,5])=15");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDoubleReduceOperateMean() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        double[] x = {1.0, 2.0, 3.0, 4.0, 5.0};
        double result = doubleComputer.reduceOperate(x, ReduceOperation.MEAN);

        TestResult r = recorder.record("simd", "double_reduce_mean");
        try {
            assertEquals(3.0, result, 1e-10, "Double MEAN");
            r.pass("Double MEAN: mean([1,2,3,4,5])=3");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDoubleReduceOperateMax() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        double[] x = {1.0, 5.0, 3.0};
        double result = doubleComputer.reduceOperate(x, ReduceOperation.MAX);

        TestResult r = recorder.record("simd", "double_reduce_max");
        try {
            assertEquals(5.0, result, 1e-10, "Double MAX");
            r.pass("Double MAX: max([1,5,3])=5");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDoubleReduceOperateMin() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        double[] x = {1.0, 5.0, 3.0};
        double result = doubleComputer.reduceOperate(x, ReduceOperation.MIN);

        TestResult r = recorder.record("simd", "double_reduce_min");
        try {
            assertEquals(1.0, result, 1e-10, "Double MIN");
            r.pass("Double MIN: min([1,5,3])=1");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDoubleUniversalOperateExp() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        double[] x = {0.0, 1.0, 2.0};
        double[] result = doubleComputer.universalOperate(x, UniversalOperation.EXP, 0.0);

        TestResult r = recorder.record("simd", "double_universal_exp");
        try {
            assertEquals(3, result.length, "Length check");
            assertEquals(1.0, result[0], 1e-10, "exp(0)");
            assertEquals(Math.E, result[1], 1e-10, "exp(1)");
            assertEquals(Math.exp(2.0), result[2], 1e-10, "exp(2)");
            r.pass("Double EXP: exp([0,1,2]) approx [1, 2.718, 7.389]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDoubleUniversalOperateSqrt() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        double[] x = {1.0, 4.0, 9.0};
        double[] result = doubleComputer.universalOperate(x, UniversalOperation.SQRT, 0.0);

        TestResult r = recorder.record("simd", "double_universal_sqrt");
        try {
            assertArrayEqualsDouble(new double[]{1.0, 2.0, 3.0}, result, 1e-10, "Double SQRT");
            r.pass("Double SQRT: sqrt([1,4,9])=[1,2,3]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDoubleBinaryReduceOperateDot() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        double[] x1 = {1.0, 2.0, 3.0};
        double[] x2 = {4.0, 5.0, 6.0};
        double result = doubleComputer.binaryReduceOperate(x1, x2, BinaryReduceOperation.DOT);

        TestResult r = recorder.record("simd", "double_binaryreduce_dot");
        try {
            assertEquals(32.0, result, 1e-10, "Double DOT");
            r.pass("Double DOT: dot([1,2,3],[4,5,6])=32");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDoubleMmul() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        double[][] a = {{1.0, 2.0}, {3.0, 4.0}};
        double[][] b = {{2.0, 0.0}, {1.0, 2.0}};
        double[][] result = doubleComputer.mmul(a, b);

        TestResult r = recorder.record("simd", "double_mmul");
        try {
            assertMatrixEqualsDouble(new double[][]{{4.0, 4.0}, {10.0, 8.0}}, result, 1e-10, "Double MMUL");
            r.pass("Double MMUL: [[1,2],[3,4]]*[[2,0],[1,2]]=[[4,4],[10,8]]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDoubleTranspose() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        double[][] matrix = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
        double[][] result = doubleComputer.transpose(matrix);

        TestResult r = recorder.record("simd", "double_transpose");
        try {
            assertMatrixEqualsDouble(new double[][]{{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}}, result, 1e-10, "Double TRANSPOSE");
            r.pass("Double TRANSPOSE: [[1,2,3],[4,5,6]]^T=[[1,4],[2,5],[3,6]]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDoubleSign() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        double[] x = {-3.0, 0.0, 5.0};
        double[] result = doubleComputer.sign(x);

        TestResult r = recorder.record("simd", "double_sign");
        try {
            assertArrayEqualsDouble(new double[]{-1.0, 0.0, 1.0}, result, 1e-10, "Double SIGN");
            r.pass("Double SIGN: sign([-3,0,5])=[-1,0,1]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDoubleLogicalCompare() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        double[] x1 = {1.0, 2.0, 3.0};
        double[] x2 = {1.0, 3.0, 2.0};

        TestResult r = recorder.record("simd", "double_logical_compare");
        try {
            boolean[] eq = doubleComputer.logicalCompare(x1, x2, LogicalCompare.EQUALS);
            assertArrayEquals(new boolean[]{true, false, false}, eq, "EQUALS");

            boolean[] gt = doubleComputer.logicalCompare(x1, x2, LogicalCompare.GREATER_THAN);
            assertArrayEquals(new boolean[]{false, false, true}, gt, "GREATER_THAN");

            boolean[] lt = doubleComputer.logicalCompare(x1, x2, LogicalCompare.LESS_THAN);
            assertArrayEquals(new boolean[]{false, true, false}, lt, "LESS_THAN");

            boolean[] gte = doubleComputer.logicalCompare(x1, x2, LogicalCompare.GREATER_THAN_OR_EQUALS);
            assertArrayEquals(new boolean[]{true, false, true}, gte, "GREATER_THAN_OR_EQUALS");

            boolean[] lte = doubleComputer.logicalCompare(x1, x2, LogicalCompare.LESS_THAN_OR_EQUALS);
            assertArrayEquals(new boolean[]{true, true, false}, lte, "LESS_THAN_OR_EQUALS");

            boolean[] neq = doubleComputer.logicalCompare(x1, x2, LogicalCompare.NOT_EQUALS);
            assertArrayEquals(new boolean[]{false, true, true}, neq, "NOT_EQUALS");

            r.pass("Double logicalCompare: EQUALS, GREATER_THAN, LESS_THAN, etc. all correct");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    // ==================== FloatVectorComputer Tests ====================

    @Test
    @Timeout(value = 10)
    public void testFloatBinaryOperateAdd() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        float[] x1 = {1.0f, 2.0f, 3.0f};
        float[] x2 = {4.0f, 5.0f, 6.0f};
        float[] result = floatComputer.binaryOperate(x1, x2, BinaryOperation.ADD);

        TestResult r = recorder.record("simd", "float_binary_add");
        try {
            assertArrayEqualsFloat(new float[]{5.0f, 7.0f, 9.0f}, result, 1e-5f, "Float ADD");
            r.pass("Float ADD: [1,2,3]+[4,5,6]=[5,7,9]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testFloatBinaryOperateSubtract() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        float[] x1 = {5.0f, 3.0f, 1.0f};
        float[] x2 = {2.0f, 2.0f, 2.0f};
        float[] result = floatComputer.binaryOperate(x1, x2, BinaryOperation.SUBTRACT);

        TestResult r = recorder.record("simd", "float_binary_subtract");
        try {
            assertArrayEqualsFloat(new float[]{3.0f, 1.0f, -1.0f}, result, 1e-5f, "Float SUBTRACT");
            r.pass("Float SUBTRACT: [5,3,1]-[2,2,2]=[3,1,-1]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testFloatBinaryOperateMultiply() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        float[] x1 = {2.0f, 3.0f, 4.0f};
        float[] x2 = {1.0f, 2.0f, 3.0f};
        float[] result = floatComputer.binaryOperate(x1, x2, BinaryOperation.MULTIPLY);

        TestResult r = recorder.record("simd", "float_binary_multiply");
        try {
            assertArrayEqualsFloat(new float[]{2.0f, 6.0f, 12.0f}, result, 1e-5f, "Float MULTIPLY");
            r.pass("Float MULTIPLY: [2,3,4]*[1,2,3]=[2,6,12]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testFloatBinaryOperateDivide() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        float[] x1 = {10.0f, 20.0f, 30.0f};
        float[] x2 = {2.0f, 5.0f, 10.0f};
        float[] result = floatComputer.binaryOperate(x1, x2, BinaryOperation.DIVIDE);

        TestResult r = recorder.record("simd", "float_binary_divide");
        try {
            assertArrayEqualsFloat(new float[]{5.0f, 4.0f, 3.0f}, result, 1e-5f, "Float DIVIDE");
            r.pass("Float DIVIDE: [10,20,30]/[2,5,10]=[5,4,3]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testFloatScalarOperation() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        float[] x1 = {1.0f, 2.0f, 3.0f};
        float scalar = 2.0f;
        float[] result = floatComputer.binaryOperate(x1, scalar, BinaryOperation.MULTIPLY);

        TestResult r = recorder.record("simd", "float_scalar_multiply");
        try {
            assertArrayEqualsFloat(new float[]{2.0f, 4.0f, 6.0f}, result, 1e-5f, "Float scalar multiply");
            r.pass("Float scalar MULTIPLY: [1,2,3]*2=[2,4,6]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testFloatReduceOperateSum() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        float[] x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float result = floatComputer.reduceOperate(x, ReduceOperation.SUM);

        TestResult r = recorder.record("simd", "float_reduce_sum");
        try {
            assertEquals(15.0f, result, 1e-5f, "Float SUM");
            r.pass("Float SUM: sum([1,2,3,4,5])=15");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testFloatReduceOperateMean() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        float[] x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float result = floatComputer.reduceOperate(x, ReduceOperation.MEAN);

        TestResult r = recorder.record("simd", "float_reduce_mean");
        try {
            assertEquals(3.0f, result, 1e-5f, "Float MEAN");
            r.pass("Float MEAN: mean([1,2,3,4,5])=3");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testFloatReduceOperateMax() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        float[] x = {1.0f, 5.0f, 3.0f};
        float result = floatComputer.reduceOperate(x, ReduceOperation.MAX);

        TestResult r = recorder.record("simd", "float_reduce_max");
        try {
            assertEquals(5.0f, result, 1e-5f, "Float MAX");
            r.pass("Float MAX: max([1,5,3])=5");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testFloatReduceOperateMin() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        float[] x = {1.0f, 5.0f, 3.0f};
        float result = floatComputer.reduceOperate(x, ReduceOperation.MIN);

        TestResult r = recorder.record("simd", "float_reduce_min");
        try {
            assertEquals(1.0f, result, 1e-5f, "Float MIN");
            r.pass("Float MIN: min([1,5,3])=1");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testFloatUniversalOperateExp() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        float[] x = {0.0f, 1.0f, 2.0f};
        float[] result = floatComputer.universalOperate(x, UniversalOperation.EXP, 0.0f);

        TestResult r = recorder.record("simd", "float_universal_exp");
        try {
            assertEquals(3, result.length, "Length check");
            assertEquals(1.0f, result[0], 1e-5f, "exp(0)");
            assertEquals((float) Math.E, result[1], 1e-5f, "exp(1)");
            assertEquals((float) Math.exp(2.0), result[2], 1e-5f, "exp(2)");
            r.pass("Float EXP: exp([0,1,2]) approx [1, 2.718, 7.389]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testFloatUniversalOperateSqrt() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        float[] x = {1.0f, 4.0f, 9.0f};
        float[] result = floatComputer.universalOperate(x, UniversalOperation.SQRT, 0.0f);

        TestResult r = recorder.record("simd", "float_universal_sqrt");
        try {
            assertArrayEqualsFloat(new float[]{1.0f, 2.0f, 3.0f}, result, 1e-5f, "Float SQRT");
            r.pass("Float SQRT: sqrt([1,4,9])=[1,2,3]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testFloatBinaryReduceOperateDot() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        float[] x1 = {1.0f, 2.0f, 3.0f};
        float[] x2 = {4.0f, 5.0f, 6.0f};
        float result = floatComputer.binaryReduceOperate(x1, x2, BinaryReduceOperation.DOT);

        TestResult r = recorder.record("simd", "float_binaryreduce_dot");
        try {
            assertEquals(32.0f, result, 1e-5f, "Float DOT");
            r.pass("Float DOT: dot([1,2,3],[4,5,6])=32");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testFloatMmul() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        float[][] a = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        float[][] b = {{2.0f, 0.0f}, {1.0f, 2.0f}};
        float[][] result = floatComputer.mmul(a, b);

        TestResult r = recorder.record("simd", "float_mmul");
        try {
            assertMatrixEqualsFloat(new float[][]{{4.0f, 4.0f}, {10.0f, 8.0f}}, result, 1e-5f, "Float MMUL");
            r.pass("Float MMUL: [[1,2],[3,4]]*[[2,0],[1,2]]=[[4,4],[10,8]]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testFloatTranspose() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        float[][] matrix = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
        float[][] result = floatComputer.transpose(matrix);

        TestResult r = recorder.record("simd", "float_transpose");
        try {
            assertMatrixEqualsFloat(new float[][]{{1.0f, 4.0f}, {2.0f, 5.0f}, {3.0f, 6.0f}}, result, 1e-5f, "Float TRANSPOSE");
            r.pass("Float TRANSPOSE: [[1,2,3],[4,5,6]]^T=[[1,4],[2,5],[3,6]]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testFloatSign() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        float[] x = {-3.0f, 0.0f, 5.0f};
        float[] result = floatComputer.sign(x);

        TestResult r = recorder.record("simd", "float_sign");
        try {
            assertArrayEqualsFloat(new float[]{-1.0f, 0.0f, 1.0f}, result, 1e-5f, "Float SIGN");
            r.pass("Float SIGN: sign([-3,0,5])=[-1,0,1]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testFloatLogicalCompare() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        float[] x1 = {1.0f, 2.0f, 3.0f};
        float[] x2 = {1.0f, 3.0f, 2.0f};

        TestResult r = recorder.record("simd", "float_logical_compare");
        try {
            boolean[] eq = floatComputer.logicalCompare(x1, x2, LogicalCompare.EQUALS);
            assertArrayEquals(new boolean[]{true, false, false}, eq, "Float EQUALS");

            boolean[] gt = floatComputer.logicalCompare(x1, x2, LogicalCompare.GREATER_THAN);
            assertArrayEquals(new boolean[]{false, false, true}, gt, "Float GREATER_THAN");

            boolean[] lt = floatComputer.logicalCompare(x1, x2, LogicalCompare.LESS_THAN);
            assertArrayEquals(new boolean[]{false, true, false}, lt, "Float LESS_THAN");

            r.pass("Float logicalCompare: EQUALS, GREATER_THAN, LESS_THAN all correct");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    // ==================== ComputerConfig Tests ====================

    @Test
    @Timeout(value = 10)
    public void testComputerConfigCheckIfSIMDSupported() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        TestResult r = recorder.record("config", "check_simd_supported");
        try {
            boolean result = ComputerConfig.checkIfSIMDSupported();
            // Only verify it does not throw an exception; do not verify true/false
            r.pass("checkIfSIMDSupported() returned " + result + " without exception");
        } catch (Exception e) {
            r.fail("checkIfSIMDSupported() threw exception: " + e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testComputerConfigCheckIfGPUSupported() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        TestResult r = recorder.record("config", "check_gpu_supported");
        try {
            boolean result = ComputerConfig.checkIfGPUSupported();
            // Only verify it does not throw an exception; do not verify true/false
            r.pass("checkIfGPUSupported() returned " + result + " without exception");
        } catch (Exception e) {
            r.fail("checkIfGPUSupported() threw exception: " + e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    // ==================== HpcOptionalRuntime Tests ====================

    @Test
    @Timeout(value = 10)
    public void testHpcOptionalRuntimeIsExtensionPresent() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        TestResult r = recorder.record("hpc", "is_extension_present");
        try {
            boolean result = HpcOptionalRuntime.isExtensionPresent();
            // Only verify it does not throw an exception
            r.pass("isExtensionPresent() returned " + result + " without exception");
        } catch (Exception e) {
            r.fail("isExtensionPresent() threw exception: " + e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    // ==================== Matrix Operations Tests ====================

    @Test
    @Timeout(value = 10)
    public void testDoubleElementWiseMin() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        double[] x1 = {1.0, 5.0, 3.0};
        double[] x2 = {4.0, 2.0, 6.0};
        double[] result = doubleComputer.elementWiseMin(x1, x2);

        TestResult r = recorder.record("simd", "double_elementwise_min");
        try {
            assertArrayEqualsDouble(new double[]{1.0, 2.0, 3.0}, result, 1e-10, "Double elementWiseMin");
            r.pass("Double elementWiseMin: min([1,5,3],[4,2,6])=[1,2,3]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDoubleElementWiseMax() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        double[] x1 = {1.0, 5.0, 3.0};
        double[] x2 = {4.0, 2.0, 6.0};
        double[] result = doubleComputer.elementWiseMax(x1, x2);

        TestResult r = recorder.record("simd", "double_elementwise_max");
        try {
            assertArrayEqualsDouble(new double[]{4.0, 5.0, 6.0}, result, 1e-10, "Double elementWiseMax");
            r.pass("Double elementWiseMax: max([1,5,3],[4,2,6])=[4,5,6]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDoubleNegate() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        double[] x = {1.0, -2.0, 3.0, 0.0};
        double[] result = doubleComputer.negate(x);

        TestResult r = recorder.record("simd", "double_negate");
        try {
            assertArrayEqualsDouble(new double[]{-1.0, 2.0, -3.0, 0.0}, result, 1e-10, "Double negate");
            r.pass("Double negate: negate([1,-2,3,0])=[-1,2,-3,0]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDoubleOuter() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        double[] a = {1.0, 2.0, 3.0};
        double[] b = {4.0, 5.0};
        double[][] result = doubleComputer.outer(a, b);

        TestResult r = recorder.record("simd", "double_outer");
        try {
            assertMatrixEqualsDouble(new double[][]{{4.0, 5.0}, {8.0, 10.0}, {12.0, 15.0}}, result, 1e-10, "Double outer");
            r.pass("Double outer: outer([1,2,3],[4,5])=[[4,5],[8,10],[12,15]]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testFloatElementWiseMin() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        float[] x1 = {1.0f, 5.0f, 3.0f};
        float[] x2 = {4.0f, 2.0f, 6.0f};
        float[] result = floatComputer.elementWiseMin(x1, x2);

        TestResult r = recorder.record("simd", "float_elementwise_min");
        try {
            assertArrayEqualsFloat(new float[]{1.0f, 2.0f, 3.0f}, result, 1e-5f, "Float elementWiseMin");
            r.pass("Float elementWiseMin: min([1,5,3],[4,2,6])=[1,2,3]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testFloatElementWiseMax() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        float[] x1 = {1.0f, 5.0f, 3.0f};
        float[] x2 = {4.0f, 2.0f, 6.0f};
        float[] result = floatComputer.elementWiseMax(x1, x2);

        TestResult r = recorder.record("simd", "float_elementwise_max");
        try {
            assertArrayEqualsFloat(new float[]{4.0f, 5.0f, 6.0f}, result, 1e-5f, "Float elementWiseMax");
            r.pass("Float elementWiseMax: max([1,5,3],[4,2,6])=[4,5,6]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testFloatNegate() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        float[] x = {1.0f, -2.0f, 3.0f, 0.0f};
        float[] result = floatComputer.negate(x);

        TestResult r = recorder.record("simd", "float_negate");
        try {
            assertArrayEqualsFloat(new float[]{-1.0f, 2.0f, -3.0f, 0.0f}, result, 1e-5f, "Float negate");
            r.pass("Float negate: negate([1,-2,3,0])=[-1,2,-3,0]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testFloatOuter() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        float[] a = {1.0f, 2.0f, 3.0f};
        float[] b = {4.0f, 5.0f};
        float[][] result = floatComputer.outer(a, b);

        TestResult r = recorder.record("simd", "float_outer");
        try {
            assertMatrixEqualsFloat(new float[][]{{4.0f, 5.0f}, {8.0f, 10.0f}, {12.0f, 15.0f}}, result, 1e-5f, "Float outer");
            r.pass("Float outer: outer([1,2,3],[4,5])=[[4,5],[8,10],[12,15]]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    // ==================== Matrix-level elementWiseMin/Max/Negate Tests ====================

    @Test
    @Timeout(value = 10)
    public void testDoubleMatrixElementWiseMin() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        double[][] x1 = {{1.0, 5.0}, {3.0, 7.0}};
        double[][] x2 = {{4.0, 2.0}, {6.0, 1.0}};
        double[][] result = doubleComputer.elementWiseMin(x1, x2);

        TestResult r = recorder.record("simd", "double_matrix_elementwise_min");
        try {
            assertMatrixEqualsDouble(new double[][]{{1.0, 2.0}, {3.0, 1.0}}, result, 1e-10, "Double matrix elementWiseMin");
            r.pass("Double matrix elementWiseMin correct");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDoubleMatrixElementWiseMax() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        double[][] x1 = {{1.0, 5.0}, {3.0, 7.0}};
        double[][] x2 = {{4.0, 2.0}, {6.0, 1.0}};
        double[][] result = doubleComputer.elementWiseMax(x1, x2);

        TestResult r = recorder.record("simd", "double_matrix_elementwise_max");
        try {
            assertMatrixEqualsDouble(new double[][]{{4.0, 5.0}, {6.0, 7.0}}, result, 1e-10, "Double matrix elementWiseMax");
            r.pass("Double matrix elementWiseMax correct");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDoubleMatrixNegate() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        double[][] x = {{1.0, -2.0}, {3.0, 0.0}};
        double[][] result = doubleComputer.negate(x);

        TestResult r = recorder.record("simd", "double_matrix_negate");
        try {
            assertMatrixEqualsDouble(new double[][]{{-1.0, 2.0}, {-3.0, 0.0}}, result, 1e-10, "Double matrix negate");
            r.pass("Double matrix negate correct");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testFloatMatrixElementWiseMin() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        float[][] x1 = {{1.0f, 5.0f}, {3.0f, 7.0f}};
        float[][] x2 = {{4.0f, 2.0f}, {6.0f, 1.0f}};
        float[][] result = floatComputer.elementWiseMin(x1, x2);

        TestResult r = recorder.record("simd", "float_matrix_elementwise_min");
        try {
            assertMatrixEqualsFloat(new float[][]{{1.0f, 2.0f}, {3.0f, 1.0f}}, result, 1e-5f, "Float matrix elementWiseMin");
            r.pass("Float matrix elementWiseMin correct");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testFloatMatrixElementWiseMax() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        float[][] x1 = {{1.0f, 5.0f}, {3.0f, 7.0f}};
        float[][] x2 = {{4.0f, 2.0f}, {6.0f, 1.0f}};
        float[][] result = floatComputer.elementWiseMax(x1, x2);

        TestResult r = recorder.record("simd", "float_matrix_elementwise_max");
        try {
            assertMatrixEqualsFloat(new float[][]{{4.0f, 5.0f}, {6.0f, 7.0f}}, result, 1e-5f, "Float matrix elementWiseMax");
            r.pass("Float matrix elementWiseMax correct");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testFloatMatrixNegate() {
        TestResult.Recorder recorder = new TestResult.Recorder("compute", "test_docs/results");
        float[][] x = {{1.0f, -2.0f}, {3.0f, 0.0f}};
        float[][] result = floatComputer.negate(x);

        TestResult r = recorder.record("simd", "float_matrix_negate");
        try {
            assertMatrixEqualsFloat(new float[][]{{-1.0f, 2.0f}, {-3.0f, 0.0f}}, result, 1e-5f, "Float matrix negate");
            r.pass("Float matrix negate correct");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }
}
