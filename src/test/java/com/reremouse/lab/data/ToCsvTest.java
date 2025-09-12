package com.reremouse.lab.data;

import com.reremouse.lab.math.data.Column;
import com.reremouse.lab.math.data.DataFrame;
import com.reremouse.lab.math.data.ColumnType;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * 测试DataFrame的toCsv方法
 * Test for DataFrame's toCsv method
 */
public class ToCsvTest {

    private DataFrame testDataFrame;
    private String testFilePath = "test_output.csv";

    public static void main(String[] args) {
        ToCsvTest test = new ToCsvTest();
        test.setUp();
        test.runAllTests();
    }

    void setUp() {
        // 创建测试数据 / Create test data
        testDataFrame = new DataFrame();
        
        // 创建列1：字符串类型 / Create column 1: String type
        Column col1 = new Column();
        col1.setName("姓名");
        col1.setColumnType(ColumnType.String);
        List<Object> col1Data = new ArrayList<>();
        col1Data.add("张三");
        col1Data.add("李四");
        col1Data.add("王五");
        col1.setData(col1Data);
        
        // 创建列2：浮点类型 / Create column 2: Float type
        Column col2 = new Column();
        col2.setName("年龄");
        col2.setColumnType(ColumnType.Float);
        List<Object> col2Data = new ArrayList<>();
        col2Data.add(25.0f);
        col2Data.add(30.0f);
        col2Data.add(35.0f);
        col2.setData(col2Data);
        
        // 创建列3：字符串类型 / Create column 3: String type
        Column col3 = new Column();
        col3.setName("城市");
        col3.setColumnType(ColumnType.String);
        List<Object> col3Data = new ArrayList<>();
        col3Data.add("北京");
        col3Data.add("上海");
        col3Data.add("广州");
        col3.setData(col3Data);
        
        testDataFrame.addColumn(col1);
        testDataFrame.addColumn(col2);
        testDataFrame.addColumn(col3);
    }

    void tearDown() {
        // 清理测试文件 / Clean up test files
        try {
            Files.deleteIfExists(Paths.get(testFilePath));
        } catch (IOException e) {
            // 忽略清理错误 / Ignore cleanup errors
        }
    }

    void runAllTests() {
        System.out.println("开始运行toCsv方法测试 / Starting toCsv method tests");
        
        try {
            testToCsvBasic();
            System.out.println("✓ testToCsvBasic 通过 / passed");
        } catch (Exception e) {
            System.out.println("✗ testToCsvBasic 失败 / failed: " + e.getMessage());
        }
        
        try {
            testToCsvEmptyDataFrame();
            System.out.println("✓ testToCsvEmptyDataFrame 通过 / passed");
        } catch (Exception e) {
            System.out.println("✗ testToCsvEmptyDataFrame 失败 / failed: " + e.getMessage());
        }
        
        try {
            testToCsvWithNullFilePath();
            System.out.println("✓ testToCsvWithNullFilePath 通过 / passed");
        } catch (Exception e) {
            System.out.println("✗ testToCsvWithNullFilePath 失败 / failed: " + e.getMessage());
        }
        
        try {
            testToCsvWithDifferentDataTypes();
            System.out.println("✓ testToCsvWithDifferentDataTypes 通过 / passed");
        } catch (Exception e) {
            System.out.println("✗ testToCsvWithDifferentDataTypes 失败 / failed: " + e.getMessage());
        }
        
        try {
            testToCsvWithUnequalColumnLengths();
            System.out.println("✓ testToCsvWithUnequalColumnLengths 通过 / passed");
        } catch (Exception e) {
            System.out.println("✗ testToCsvWithUnequalColumnLengths 失败 / failed: " + e.getMessage());
        }
        
        System.out.println("所有测试完成 / All tests completed");
    }

    void testToCsvBasic() throws IOException {
        // 测试基本的toCsv功能 / Test basic toCsv functionality
        testDataFrame.toCsv(testFilePath);
        
        // 验证文件是否创建 / Verify file was created
        assertTrue(Files.exists(Paths.get(testFilePath)), "CSV文件应该被创建 / CSV file should be created");
        
        // 读取并验证文件内容 / Read and verify file content
        List<String> lines = Files.readAllLines(Paths.get(testFilePath));
        
        // 验证行数 / Verify number of lines
        assertEquals(4, lines.size(), "应该有4行（1行表头 + 3行数据）/ Should have 4 lines (1 header + 3 data rows)");
        
        // 验证表头 / Verify header
        assertEquals("姓名,年龄,城市", lines.get(0), "表头应该正确 / Header should be correct");
        
        // 验证数据行 / Verify data rows
        assertEquals("张三,25.0,北京", lines.get(1), "第一行数据应该正确 / First data row should be correct");
        assertEquals("李四,30.0,上海", lines.get(2), "第二行数据应该正确 / Second data row should be correct");
        assertEquals("王五,35.0,广州", lines.get(3), "第三行数据应该正确 / Third data row should be correct");
        
        tearDown();
    }

    void testToCsvEmptyDataFrame() throws IOException {
        // 测试空DataFrame / Test empty DataFrame
        DataFrame emptyDf = new DataFrame();
        emptyDf.toCsv(testFilePath);
        
        // 验证文件是否创建 / Verify file was created
        assertTrue(Files.exists(Paths.get(testFilePath)), "空DataFrame也应该创建文件 / Empty DataFrame should also create file");
        
        // 读取并验证文件内容 / Read and verify file content
        List<String> lines = Files.readAllLines(Paths.get(testFilePath));
        
        // 空DataFrame应该只创建空文件或只有表头 / Empty DataFrame should create empty file or only header
        assertTrue(lines.isEmpty() || lines.size() == 1, "空DataFrame应该创建空文件或只有表头 / Empty DataFrame should create empty file or only header");
        
        tearDown();
    }

    void testToCsvWithNullFilePath() {
        // 测试空文件路径 / Test null file path
        try {
            testDataFrame.toCsv(null);
            throw new AssertionError("空文件路径应该抛出异常 / Null file path should throw exception");
        } catch (IllegalArgumentException e) {
            // 期望的异常 / Expected exception
        } catch (Exception e) {
            throw new AssertionError("应该抛出IllegalArgumentException，但抛出了: " + e.getClass().getSimpleName());
        }
        
        try {
            testDataFrame.toCsv("");
            throw new AssertionError("空字符串文件路径应该抛出异常 / Empty string file path should throw exception");
        } catch (IllegalArgumentException e) {
            // 期望的异常 / Expected exception
        } catch (Exception e) {
            throw new AssertionError("应该抛出IllegalArgumentException，但抛出了: " + e.getClass().getSimpleName());
        }
        
        try {
            testDataFrame.toCsv("   ");
            throw new AssertionError("空白文件路径应该抛出异常 / Blank file path should throw exception");
        } catch (IllegalArgumentException e) {
            // 期望的异常 / Expected exception
        } catch (Exception e) {
            throw new AssertionError("应该抛出IllegalArgumentException，但抛出了: " + e.getClass().getSimpleName());
        }
    }

    void testToCsvWithDifferentDataTypes() throws IOException {
        // 创建包含不同数据类型的DataFrame / Create DataFrame with different data types
        DataFrame mixedDf = new DataFrame();
        
        // 字符串列 / String column
        Column stringCol = new Column();
        stringCol.setName("文本");
        stringCol.setColumnType(ColumnType.String);
        List<Object> stringData = new ArrayList<>();
        stringData.add("Hello");
        stringData.add("World");
        stringCol.setData(stringData);
        
        // 浮点列 / Float column
        Column floatCol = new Column();
        floatCol.setName("数值");
        floatCol.setColumnType(ColumnType.Float);
        List<Object> floatData = new ArrayList<>();
        floatData.add(1.5f);
        floatData.add(2.7f);
        floatCol.setData(floatData);
        
        mixedDf.addColumn(stringCol);
        mixedDf.addColumn(floatCol);
        
        mixedDf.toCsv(testFilePath);
        
        // 验证文件内容 / Verify file content
        List<String> lines = Files.readAllLines(Paths.get(testFilePath));
        assertEquals(3, lines.size(), "应该有3行（1行表头 + 2行数据）/ Should have 3 lines (1 header + 2 data rows)");
        assertEquals("文本,数值", lines.get(0), "表头应该正确 / Header should be correct");
        assertEquals("Hello,1.5", lines.get(1), "第一行数据应该正确 / First data row should be correct");
        assertEquals("World,2.7", lines.get(2), "第二行数据应该正确 / Second data row should be correct");
        
        tearDown();
    }

    void testToCsvWithUnequalColumnLengths() throws IOException {
        // 测试列长度不一致的情况 / Test case with unequal column lengths
        DataFrame unequalDf = new DataFrame();
        
        // 第一列：3个元素 / First column: 3 elements
        Column col1 = new Column();
        col1.setName("列1");
        col1.setColumnType(ColumnType.String);
        List<Object> col1Data = new ArrayList<>();
        col1Data.add("A");
        col1Data.add("B");
        col1Data.add("C");
        col1.setData(col1Data);
        
        // 第二列：2个元素 / Second column: 2 elements
        Column col2 = new Column();
        col2.setName("列2");
        col2.setColumnType(ColumnType.String);
        List<Object> col2Data = new ArrayList<>();
        col2Data.add("X");
        col2Data.add("Y");
        col2.setData(col2Data);
        
        // 注意：这应该会抛出异常，因为列长度不一致 / Note: This should throw exception because column lengths are inconsistent
        try {
            unequalDf.addColumn(col1);
            unequalDf.addColumn(col2);
            throw new AssertionError("列长度不一致应该抛出异常 / Unequal column lengths should throw exception");
        } catch (IllegalArgumentException e) {
            // 期望的异常 / Expected exception
        } catch (Exception e) {
            throw new AssertionError("应该抛出IllegalArgumentException，但抛出了: " + e.getClass().getSimpleName());
        }
    }
    
    // 自定义断言方法 / Custom assertion methods
    private void assertTrue(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }
    
    private void assertEquals(Object expected, Object actual, String message) {
        if (expected == null && actual == null) {
            return;
        }
        if (expected == null || !expected.equals(actual)) {
            throw new AssertionError(message + " - 期望: " + expected + ", 实际: " + actual);
        }
    }
}
