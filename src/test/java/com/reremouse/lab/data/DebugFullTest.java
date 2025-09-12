package com.reremouse.lab.data;

import com.reremouse.lab.math.data.Column;
import com.reremouse.lab.math.data.DataFrame;
import com.reremouse.lab.math.data.ColumnType;
import com.reremouse.lab.math.IMatrix;
import java.util.ArrayList;
import java.util.List;

/**
 * 调试完整测试 / Debug Full Test
 */
public class DebugFullTest {
    
    private DataFrame df;

    public static void main(String[] args) {
        DebugFullTest test = new DebugFullTest();
        test.setUp();
        test.runDebugTests();
    }

    void setUp() {
        // 创建测试数据 / Create test data
        df = new DataFrame();
        
        // 添加Float类型列 / Add Float type column
        Column floatCol1 = new Column();
        floatCol1.setName("feature1");
        floatCol1.setColumnType(ColumnType.Float);
        List<Object> floatData1 = new ArrayList<>();
        floatData1.add(1.0f);
        floatData1.add(2.0f);
        floatData1.add(3.0f);
        floatData1.add(4.0f);
        floatCol1.setData(floatData1);
        df.addColumn(floatCol1);
        
        // 添加另一个Float类型列 / Add another Float type column
        Column floatCol2 = new Column();
        floatCol2.setName("feature2");
        floatCol2.setColumnType(ColumnType.Float);
        List<Object> floatData2 = new ArrayList<>();
        floatData2.add(5.0f);
        floatData2.add(6.0f);
        floatData2.add(7.0f);
        floatData2.add(8.0f);
        floatCol2.setData(floatData2);
        df.addColumn(floatCol2);
        
        // 添加String类型列 / Add String type column
        Column stringCol = new Column();
        stringCol.setName("label");
        stringCol.setColumnType(ColumnType.String);
        List<Object> stringData = new ArrayList<>();
        stringData.add("A");
        stringData.add("B");
        stringData.add("A");
        stringData.add("B");
        stringCol.setData(stringData);
        df.addColumn(stringCol);
    }

    void runDebugTests() {
        System.out.println("开始调试完整测试... / Starting debug full test...");
        
        printDataFrameState("初始状态 / Initial state");
        
        testDataFrameCreation();
        printDataFrameState("DataFrame创建后 / After DataFrame creation");
        
        testGetColumn();
        printDataFrameState("获取列后 / After get column");
        
        testSliceColumn();
        printDataFrameState("列切片后 / After column slicing");
        
        testSlice();
        printDataFrameState("通用切片后 / After general slicing");
        
        testToMatrix();
        printDataFrameState("转换为矩阵后 / After matrix conversion");
        
        System.out.println("调试测试完成！/ Debug test completed!");
    }

    void printDataFrameState(String stage) {
        System.out.println("\n=== " + stage + " ===");
        System.out.println("列数: " + df.getColumnCount());
        System.out.println("行数: " + df.getRowCount());
        for (int i = 0; i < df.getColumnCount(); i++) {
            Column col = df.get(i);
            System.out.println("列 " + i + " (" + col.getName() + "): 类型=" + col.getColumnType() + ", 数据长度=" + col.getData().size());
        }
    }

    void testDataFrameCreation() {
        System.out.println("测试DataFrame创建... / Testing DataFrame creation...");
        // 测试DataFrame创建 / Test DataFrame creation
        assertNotNull(df);
        assertEquals(3, df.getColumnCount());
        assertEquals(4, df.getRowCount());
        assertFalse(df.isEmpty());
        System.out.println("✓ DataFrame创建测试通过 / DataFrame creation test passed");
    }

    void testGetColumn() {
        System.out.println("测试获取列... / Testing column getting...");
        // 测试获取列 / Test getting columns
        Column col1 = df.get(0);
        assertEquals("feature1", col1.getName());
        assertEquals(ColumnType.Float, col1.getColumnType());
        
        // 测试负数索引 / Test negative indexing
        Column lastCol = df.get(-1);
        assertEquals("label", lastCol.getName());
        assertEquals(ColumnType.String, lastCol.getColumnType());
        System.out.println("✓ 获取列测试通过 / Column getting test passed");
    }

    void testSliceColumn() {
        System.out.println("测试列切片... / Testing column slicing...");
        // 测试列切片 / Test column slicing
        DataFrame sliced = df.sliceColumn(0, 2);
        assertEquals(2, sliced.getColumnCount());
        assertEquals("feature1", sliced.get(0).getName());
        assertEquals("feature2", sliced.get(1).getName());
        
        // 测试步长切片 / Test step slicing
        DataFrame stepSliced = df.sliceColumn(0, 3, 2);
        assertEquals(2, stepSliced.getColumnCount());
        assertEquals("feature1", stepSliced.get(0).getName());
        assertEquals("label", stepSliced.get(1).getName());
        System.out.println("✓ 列切片测试通过 / Column slicing test passed");
    }

    void testSlice() {
        System.out.println("测试通用切片... / Testing general slicing...");
        // 测试通用切片 / Test general slicing
        DataFrame sliced = df.slice("1:3", "0:2");
        assertEquals(2, sliced.getColumnCount());
        assertEquals(2, sliced.getRowCount());
        
        // 验证数据正确性 / Verify data correctness
        assertEquals(2.0f, ((Float) sliced.get(0).getData().get(0)).floatValue());
        assertEquals(3.0f, ((Float) sliced.get(0).getData().get(1)).floatValue());
        System.out.println("✓ 通用切片测试通过 / General slicing test passed");
    }

    void testToMatrix() {
        System.out.println("测试转换为矩阵... / Testing matrix conversion...");
        // 测试转换为矩阵 / Test conversion to matrix
        IMatrix matrix = df.toMatrix();
        assertNotNull(matrix);
        System.out.println("实际矩阵形状: " + matrix.getRowNum() + "x" + matrix.getColNum());
        assertEquals(4, matrix.getRowNum());
        assertEquals(2, matrix.getColNum()); // 只有Float类型列被转换 / Only Float type columns are converted
        
        // 验证矩阵数据 / Verify matrix data
        assertEquals(1.0f, matrix.get(0, 0));
        assertEquals(5.0f, matrix.get(0, 1));
        assertEquals(4.0f, matrix.get(3, 0));
        assertEquals(8.0f, matrix.get(3, 1));
        System.out.println("✓ 矩阵转换测试通过 / Matrix conversion test passed");
    }

    // 简单的断言方法 / Simple assertion methods
    private void assertNotNull(Object obj) {
        if (obj == null) {
            throw new AssertionError("Expected not null, but was null");
        }
    }

    private void assertEquals(Object expected, Object actual) {
        if (expected == null && actual == null) return;
        if (expected == null || actual == null || !expected.equals(actual)) {
            throw new AssertionError("Expected: " + expected + ", but was: " + actual);
        }
    }

    private void assertEquals(int expected, int actual) {
        if (expected != actual) {
            throw new AssertionError("Expected: " + expected + ", but was: " + actual);
        }
    }

    private void assertEquals(float expected, float actual) {
        if (Math.abs(expected - actual) > 1e-6f) {
            throw new AssertionError("Expected: " + expected + ", but was: " + actual);
        }
    }

    private void assertFalse(boolean condition) {
        if (condition) {
            throw new AssertionError("Expected false, but was true");
        }
    }
}
