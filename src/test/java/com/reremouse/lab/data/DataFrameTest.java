package com.reremouse.lab.data;

import com.reremouse.lab.math.IMatrix;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * DataFrame测试类 / DataFrame Test Class
 * <p>测试DataFrame的各种功能 / Tests various functionalities of DataFrame</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class DataFrameTest {

    private DataFrame df;

    public static void main(String[] args) {
        DataFrameTest test = new DataFrameTest();
        test.setUp();
        test.runAllTests();
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

    void runAllTests() {
        System.out.println("开始运行DataFrame测试... / Starting DataFrame tests...");
        
        testDataFrameCreation();
        testGetColumn();
        testSliceColumn();
        testSlice();
        testToMatrix();
        testAddRemoveColumn();
        testShape();
        testGetColumnNames();
        testGetColumnByName();
        testCopy();
        testToString();
        testEmptyDataFrame();
        testInvalidOperations();
        testToMatrixWithNoFloatColumns();
        
        System.out.println("所有测试完成！ / All tests completed!");
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
        assertEquals(4, matrix.getRowNum());
        assertEquals(2, matrix.getColNum()); // 只有Float类型列被转换 / Only Float type columns are converted
        
        // 验证矩阵数据 / Verify matrix data
        assertEquals(1.0f, matrix.get(0, 0));
        assertEquals(5.0f, matrix.get(0, 1));
        assertEquals(4.0f, matrix.get(3, 0));
        assertEquals(8.0f, matrix.get(3, 1));
        System.out.println("✓ 矩阵转换测试通过 / Matrix conversion test passed");
    }

    void testAddRemoveColumn() {
        System.out.println("测试添加删除列... / Testing add/remove column...");
        // 测试添加列 / Test adding column
        Column newCol = new Column();
        newCol.setName("newFeature");
        newCol.setColumnType(ColumnType.Float);
        List<Object> newData = new ArrayList<>();
        newData.add(9.0f);
        newData.add(10.0f);
        newData.add(11.0f);
        newData.add(12.0f);
        newCol.setData(newData);
        
        df.addColumn(newCol);
        assertEquals(4, df.getColumnCount());
        assertEquals("newFeature", df.get(3).getName());
        
        // 测试删除列 / Test removing column
        Column removed = df.removeColumn(3);
        assertEquals("newFeature", removed.getName());
        assertEquals(3, df.getColumnCount());
        System.out.println("✓ 添加删除列测试通过 / Add/remove column test passed");
    }

    void testShape() {
        System.out.println("测试形状获取... / Testing shape getting...");
        // 测试形状获取 / Test shape getting
        int[] shape = df.shape();
        assertEquals(4, shape[0]); // 行数 / Row count
        assertEquals(3, shape[1]); // 列数 / Column count
        System.out.println("✓ 形状获取测试通过 / Shape getting test passed");
    }

    void testGetColumnNames() {
        System.out.println("测试获取列名... / Testing column names getting...");
        // 测试获取列名 / Test getting column names
        List<String> names = df.getColumnNames();
        assertEquals(3, names.size());
        assertTrue(names.contains("feature1"));
        assertTrue(names.contains("feature2"));
        assertTrue(names.contains("label"));
        System.out.println("✓ 获取列名测试通过 / Column names getting test passed");
    }

    void testGetColumnByName() {
        System.out.println("测试根据列名获取列... / Testing column getting by name...");
        // 测试根据列名获取列 / Test getting column by name
        Column col = df.getColumnByName("feature1");
        assertNotNull(col);
        assertEquals(ColumnType.Float, col.getColumnType());
        
        Column notFound = df.getColumnByName("nonexistent");
        assertNull(notFound);
        System.out.println("✓ 根据列名获取列测试通过 / Column getting by name test passed");
    }

    void testCopy() {
        System.out.println("测试复制... / Testing copying...");
        // 测试复制 / Test copying
        DataFrame copy = df.copy();
        assertNotSame(df, copy);
        assertEquals(df.getColumnCount(), copy.getColumnCount());
        assertEquals(df.getRowCount(), copy.getRowCount());
        
        // 修改副本不应影响原对象 / Modifying copy should not affect original
        copy.removeColumn(0);
        assertEquals(2, copy.getColumnCount());
        assertEquals(3, df.getColumnCount());
        System.out.println("✓ 复制测试通过 / Copying test passed");
    }

    void testToString() {
        System.out.println("测试字符串表示... / Testing string representation...");
        // 测试字符串表示 / Test string representation
        String str = df.toString();
        assertNotNull(str);
        assertTrue(str.contains("DataFrame"));
        assertTrue(str.contains("4 rows x 3 columns"));
        assertTrue(str.contains("feature1"));
        assertTrue(str.contains("feature2"));
        assertTrue(str.contains("label"));
        System.out.println("✓ 字符串表示测试通过 / String representation test passed");
    }

    void testEmptyDataFrame() {
        System.out.println("测试空DataFrame... / Testing empty DataFrame...");
        // 测试空DataFrame / Test empty DataFrame
        DataFrame empty = new DataFrame();
        assertTrue(empty.isEmpty());
        assertEquals(0, empty.getColumnCount());
        assertEquals(0, empty.getRowCount());
        assertEquals("Empty DataFrame", empty.toString());
        System.out.println("✓ 空DataFrame测试通过 / Empty DataFrame test passed");
    }

    void testInvalidOperations() {
        System.out.println("测试无效操作... / Testing invalid operations...");
        // 测试无效操作 / Test invalid operations
        try {
            df.get(10);
            System.out.println("❌ 应该抛出IndexOutOfBoundsException / Should throw IndexOutOfBoundsException");
        } catch (IndexOutOfBoundsException e) {
            System.out.println("✓ 正确抛出IndexOutOfBoundsException / Correctly threw IndexOutOfBoundsException");
        }
        
        try {
            df.get(-10);
            System.out.println("❌ 应该抛出IndexOutOfBoundsException / Should throw IndexOutOfBoundsException");
        } catch (IndexOutOfBoundsException e) {
            System.out.println("✓ 正确抛出IndexOutOfBoundsException / Correctly threw IndexOutOfBoundsException");
        }
        
        try {
            df.sliceColumn(-1, 2);
            System.out.println("❌ 应该抛出IllegalArgumentException / Should throw IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            System.out.println("✓ 正确抛出IllegalArgumentException / Correctly threw IllegalArgumentException");
        }
        
        try {
            df.addColumn(null);
            System.out.println("❌ 应该抛出IllegalArgumentException / Should throw IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            System.out.println("✓ 正确抛出IllegalArgumentException / Correctly threw IllegalArgumentException");
        }
        
        System.out.println("✓ 无效操作测试通过 / Invalid operations test passed");
    }

    void testToMatrixWithNoFloatColumns() {
        System.out.println("测试没有Float列时的toMatrix... / Testing toMatrix with no Float columns...");
        // 测试没有Float列时的toMatrix / Test toMatrix with no Float columns
        DataFrame stringOnlyDf = new DataFrame();
        Column stringCol = new Column();
        stringCol.setName("text");
        stringCol.setColumnType(ColumnType.String);
        List<Object> stringData = new ArrayList<>();
        stringData.add("hello");
        stringData.add("world");
        stringCol.setData(stringData);
        stringOnlyDf.addColumn(stringCol);
        
        try {
            stringOnlyDf.toMatrix();
            System.out.println("❌ 应该抛出IllegalStateException / Should throw IllegalStateException");
        } catch (IllegalStateException e) {
            System.out.println("✓ 正确抛出IllegalStateException / Correctly threw IllegalStateException");
        }
        
        System.out.println("✓ 没有Float列时的toMatrix测试通过 / toMatrix with no Float columns test passed");
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

    private void assertTrue(boolean condition) {
        if (!condition) {
            throw new AssertionError("Expected true, but was false");
        }
    }

    private void assertFalse(boolean condition) {
        if (condition) {
            throw new AssertionError("Expected false, but was true");
        }
    }

    private void assertNull(Object obj) {
        if (obj != null) {
            throw new AssertionError("Expected null, but was: " + obj);
        }
    }

    private void assertNotSame(Object expected, Object actual) {
        if (expected == actual) {
            throw new AssertionError("Expected different objects, but they are the same");
        }
    }
}
