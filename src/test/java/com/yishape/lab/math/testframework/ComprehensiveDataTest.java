package com.yishape.lab.math.testframework;

import com.yishape.lab.math.data.Column;
import com.yishape.lab.math.data.ColumnType;
import com.yishape.lab.math.data.DataFrame;
import com.yishape.lab.math.linalg.IMatrix;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test class for com.yishape.lab.math.data package.
 * Covers DataFrame, Column, and CSV read/write operations.
 */
public class ComprehensiveDataTest {

    // ==================== DataFrame Tests ====================

    @Test
    @Timeout(value = 10)
    public void testDataFrameEmptyCreation() {
        TestResult.Recorder recorder = new TestResult.Recorder("data", "test_docs/results");
        TestResult r = recorder.record("dataframe", "empty_creation");
        try {
            DataFrame df = new DataFrame();
            assertEquals(0, df.getColumnCount(), "Empty DataFrame should have 0 columns");
            assertEquals(0, df.getRowCount(), "Empty DataFrame should have 0 rows");
            assertTrue(df.isEmpty(), "Empty DataFrame should be empty");
            r.pass("Empty DataFrame created with 0 columns and 0 rows");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDataFrameAddColumn() {
        TestResult.Recorder recorder = new TestResult.Recorder("data", "test_docs/results");
        TestResult r = recorder.record("dataframe", "add_column");
        try {
            DataFrame df = new DataFrame();
            Column col = new Column();
            col.setName("testCol");
            col.setColumnType(ColumnType.Numeric);
            col.setData(new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0)));

            df.addColumn(col);
            assertEquals(1, df.getColumnCount(), "DataFrame should have 1 column");
            assertEquals(3, df.getRowCount(), "DataFrame should have 3 rows");
            r.pass("addColumn: DataFrame has 1 column and 3 rows");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDataFrameGetColumnNames() {
        TestResult.Recorder recorder = new TestResult.Recorder("data", "test_docs/results");
        TestResult r = recorder.record("dataframe", "get_column_names");
        try {
            DataFrame df = new DataFrame();
            Column col1 = new Column();
            col1.setName("colA");
            col1.setColumnType(ColumnType.Numeric);
            col1.setData(new ArrayList<>(Arrays.asList(1.0, 2.0)));

            Column col2 = new Column();
            col2.setName("colB");
            col2.setColumnType(ColumnType.Numeric);
            col2.setData(new ArrayList<>(Arrays.asList(3.0, 4.0)));

            df.addColumn(col1);
            df.addColumn(col2);

            List<String> names = df.getColumnNames();
            assertEquals(2, names.size(), "Should have 2 column names");
            assertEquals("colA", names.get(0), "First column name should be colA");
            assertEquals("colB", names.get(1), "Second column name should be colB");
            r.pass("getColumnNames returns correct names: [colA, colB]");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDataFrameGetRowCount() {
        TestResult.Recorder recorder = new TestResult.Recorder("data", "test_docs/results");
        TestResult r = recorder.record("dataframe", "get_row_count");
        try {
            DataFrame df = new DataFrame();
            Column col = new Column();
            col.setName("testCol");
            col.setColumnType(ColumnType.Numeric);
            col.setData(new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0)));

            df.addColumn(col);
            assertEquals(5, df.getRowCount(), "DataFrame should have 5 rows");
            r.pass("getRowCount returns 5 for DataFrame with 5 rows");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDataFrameSliceColumn() {
        TestResult.Recorder recorder = new TestResult.Recorder("data", "test_docs/results");
        TestResult r = recorder.record("dataframe", "slice_column");
        try {
            DataFrame df = new DataFrame();
            for (int i = 0; i < 5; i++) {
                Column col = new Column();
                col.setName("col" + i);
                col.setColumnType(ColumnType.Numeric);
                col.setData(new ArrayList<>(Arrays.asList((double) i)));
                df.addColumn(col);
            }

            DataFrame sliced = df.sliceColumn(1, 4);
            assertEquals(3, sliced.getColumnCount(), "Sliced DataFrame should have 3 columns");
            assertEquals("col1", sliced.getColumnNames().get(0), "First sliced column should be col1");
            assertEquals("col2", sliced.getColumnNames().get(1), "Second sliced column should be col2");
            assertEquals("col3", sliced.getColumnNames().get(2), "Third sliced column should be col3");
            r.pass("sliceColumn(1,4) returns DataFrame with 3 columns: col1, col2, col3");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDataFrameCopy() {
        TestResult.Recorder recorder = new TestResult.Recorder("data", "test_docs/results");
        TestResult r = recorder.record("dataframe", "copy");
        try {
            DataFrame df = new DataFrame();
            Column col = new Column();
            col.setName("original");
            col.setColumnType(ColumnType.Numeric);
            col.setData(new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0)));
            df.addColumn(col);

            DataFrame copied = df.copy();
            assertEquals(df.getColumnCount(), copied.getColumnCount(), "Copy should have same column count");
            assertEquals(df.getRowCount(), copied.getRowCount(), "Copy should have same row count");

            // Modify original and verify copy is independent
            col.setName("modified");
            assertEquals("original", copied.getColumnNames().get(0), "Copy should be independent of original");

            r.pass("copy creates independent DataFrame; modifying original does not affect copy");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDataFrameShape() {
        TestResult.Recorder recorder = new TestResult.Recorder("data", "test_docs/results");
        TestResult r = recorder.record("dataframe", "shape");
        try {
            DataFrame df = new DataFrame();
            Column col1 = new Column();
            col1.setName("col1");
            col1.setColumnType(ColumnType.Numeric);
            col1.setData(new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0)));

            Column col2 = new Column();
            col2.setName("col2");
            col2.setColumnType(ColumnType.Numeric);
            col2.setData(new ArrayList<>(Arrays.asList(4.0, 5.0, 6.0)));

            df.addColumn(col1);
            df.addColumn(col2);

            int[] shape = df.shape();
            assertEquals(2, shape.length, "Shape should have 2 dimensions");
            assertEquals(3, shape[0], "Shape row count should be 3");
            assertEquals(2, shape[1], "Shape column count should be 2");
            r.pass("shape returns [3, 2] for 3x2 DataFrame");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDataFrameIsEmpty() {
        TestResult.Recorder recorder = new TestResult.Recorder("data", "test_docs/results");
        TestResult r = recorder.record("dataframe", "is_empty");
        try {
            DataFrame emptyDf = new DataFrame();
            assertTrue(emptyDf.isEmpty(), "Empty DataFrame should be empty");

            DataFrame nonEmptyDf = new DataFrame();
            Column col = new Column();
            col.setName("col");
            col.setColumnType(ColumnType.Numeric);
            col.setData(new ArrayList<>(Arrays.asList(1.0)));
            nonEmptyDf.addColumn(col);
            assertFalse(nonEmptyDf.isEmpty(), "Non-empty DataFrame should not be empty");

            r.pass("isEmpty returns true for empty DataFrame and false for non-empty DataFrame");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDataFrameToMatrix() {
        TestResult.Recorder recorder = new TestResult.Recorder("data", "test_docs/results");
        TestResult r = recorder.record("dataframe", "to_matrix");
        try {
            DataFrame df = new DataFrame();
            Column col1 = new Column();
            col1.setName("x");
            col1.setColumnType(ColumnType.Numeric);
            col1.setData(new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0)));

            Column col2 = new Column();
            col2.setName("y");
            col2.setColumnType(ColumnType.Numeric);
            col2.setData(new ArrayList<>(Arrays.asList(4.0, 5.0, 6.0)));

            df.addColumn(col1);
            df.addColumn(col2);

            IMatrix matrix = df.toMatrix();
            assertNotNull(matrix, "Matrix should not be null");
            assertEquals(3, matrix.rows(), "Matrix should have 3 rows");
            assertEquals(2, matrix.cols(), "Matrix should have 2 columns");
            r.pass("toMatrix converts DataFrame to 3x2 matrix correctly");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDataFrameClear() {
        TestResult.Recorder recorder = new TestResult.Recorder("data", "test_docs/results");
        TestResult r = recorder.record("dataframe", "clear");
        try {
            DataFrame df = new DataFrame();
            Column col = new Column();
            col.setName("col");
            col.setColumnType(ColumnType.Numeric);
            col.setData(new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0)));
            df.addColumn(col);

            assertFalse(df.isEmpty(), "DataFrame should not be empty before clear");
            df.clear();
            assertTrue(df.isEmpty(), "DataFrame should be empty after clear");
            assertEquals(0, df.getColumnCount(), "Column count should be 0 after clear");
            assertEquals(0, df.getRowCount(), "Row count should be 0 after clear");
            r.pass("clear empties DataFrame: columns=0, rows=0, isEmpty=true");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    // ==================== Column Tests ====================

    @Test
    @Timeout(value = 10)
    public void testColumnNumericCreation() {
        TestResult.Recorder recorder = new TestResult.Recorder("data", "test_docs/results");
        TestResult r = recorder.record("column", "numeric_creation");
        try {
            Column col = new Column();
            col.setName("numericCol");
            col.setColumnType(ColumnType.Numeric);
            col.setData(new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0)));

            assertEquals("numericCol", col.getName(), "Column name should be numericCol");
            assertEquals(ColumnType.Numeric, col.getColumnType(), "Column type should be Numeric");
            assertEquals(5, col.getData().size(), "Column should have 5 data elements");
            r.pass("Numeric column created with correct name, type, and 5 elements");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testColumnGetNameSetName() {
        TestResult.Recorder recorder = new TestResult.Recorder("data", "test_docs/results");
        TestResult r = recorder.record("column", "get_set_name");
        try {
            Column col = new Column();
            col.setName("oldName");
            assertEquals("oldName", col.getName(), "Initial name should be oldName");

            col.setName("newName");
            assertEquals("newName", col.getName(), "Name should be updated to newName");
            r.pass("getName/setName work correctly: oldName -> newName");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testColumnGetColumnType() {
        TestResult.Recorder recorder = new TestResult.Recorder("data", "test_docs/results");
        TestResult r = recorder.record("column", "get_column_type");
        try {
            Column numericCol = new Column();
            numericCol.setColumnType(ColumnType.Numeric);
            assertEquals(ColumnType.Numeric, numericCol.getColumnType(), "Type should be Numeric");

            Column stringCol = new Column();
            stringCol.setColumnType(ColumnType.String);
            assertEquals(ColumnType.String, stringCol.getColumnType(), "Type should be String");

            r.pass("getColumnType returns correct type for Numeric and String columns");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testColumnToVec() {
        TestResult.Recorder recorder = new TestResult.Recorder("data", "test_docs/results");
        TestResult r = recorder.record("column", "to_vec");
        try {
            Column col = new Column();
            col.setName("vecCol");
            col.setColumnType(ColumnType.Numeric);
            col.setData(new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0)));

            var vec = col.toVec();
            assertNotNull(vec, "Vector should not be null");
            assertEquals(5, vec.size(), "Vector should have 5 elements");
            r.pass("toVec converts Numeric column to vector with 5 elements");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    // ==================== CSV Read/Write Tests ====================

    @Test
    @Timeout(value = 10)
    public void testCsvWriteAndRead() throws IOException {
        TestResult.Recorder recorder = new TestResult.Recorder("data", "test_docs/results");
        TestResult r = recorder.record("csv", "write_and_read");
        Path tempDir = null;
        try {
            tempDir = Files.createTempDirectory("yishape_test_");
            String csvPath = tempDir.resolve("test_data.csv").toString();

            // Create a DataFrame and write to CSV
            DataFrame df = new DataFrame();
            Column col1 = new Column();
            col1.setName("A");
            col1.setColumnType(ColumnType.Numeric);
            col1.setData(new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0)));

            Column col2 = new Column();
            col2.setName("B");
            col2.setColumnType(ColumnType.Numeric);
            col2.setData(new ArrayList<>(Arrays.asList(4.0, 5.0, 6.0)));

            Column col3 = new Column();
            col3.setName("C");
            col3.setColumnType(ColumnType.String);
            col3.setData(new ArrayList<>(Arrays.asList("x", "y", "z")));

            df.addColumn(col1);
            df.addColumn(col2);
            df.addColumn(col3);

            df.toCsv(csvPath);
            assertTrue(new File(csvPath).exists(), "CSV file should exist after writing");

            // Read back the CSV
            DataFrame readDf = DataFrame.readCsv(csvPath);
            assertEquals(3, readDf.getColumnCount(), "Read DataFrame should have 3 columns");
            assertEquals(3, readDf.getRowCount(), "Read DataFrame should have 3 rows");

            List<String> names = readDf.getColumnNames();
            assertTrue(names.contains("A"), "Column A should exist");
            assertTrue(names.contains("B"), "Column B should exist");
            assertTrue(names.contains("C"), "Column C should exist");

            r.pass("CSV write and read: data consistent, 3 columns and 3 rows preserved");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
            if (tempDir != null) {
                Files.walk(tempDir).sorted((a, b) -> -a.compareTo(b)).forEach(p -> {
                    try {
                        Files.deleteIfExists(p);
                    } catch (IOException ignored) {
                    }
                });
            }
        }
    }

    @Test
    @Timeout(value = 10)
    public void testCsvReadWithHeader() throws IOException {
        TestResult.Recorder recorder = new TestResult.Recorder("data", "test_docs/results");
        TestResult r = recorder.record("csv", "read_with_header");
        Path tempDir = null;
        try {
            tempDir = Files.createTempDirectory("yishape_test_");
            String csvPath = tempDir.resolve("header_test.csv").toString();

            // Write a CSV with header manually
            String csvContent = "Name,Age,Score\nAlice,25,95.5\nBob,30,88.0\nCharlie,35,92.5\n";
            Files.writeString(tempDir.resolve("header_test.csv"), csvContent);

            DataFrame df = DataFrame.readCsv(csvPath);
            assertEquals(3, df.getColumnCount(), "Should have 3 columns");
            assertEquals(3, df.getRowCount(), "Should have 3 rows");

            List<String> names = df.getColumnNames();
            assertEquals("Name", names.get(0), "First column should be Name");
            assertEquals("Age", names.get(1), "Second column should be Age");
            assertEquals("Score", names.get(2), "Third column should be Score");

            r.pass("CSV read with header: 3 columns (Name, Age, Score), 3 rows");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
            if (tempDir != null) {
                Files.walk(tempDir).sorted((a, b) -> -a.compareTo(b)).forEach(p -> {
                    try {
                        Files.deleteIfExists(p);
                    } catch (IOException ignored) {
                    }
                });
            }
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDataFrameGetColumnByName() {
        TestResult.Recorder recorder = new TestResult.Recorder("data", "test_docs/results");
        TestResult r = recorder.record("dataframe", "get_column_by_name");
        try {
            DataFrame df = new DataFrame();
            Column col1 = new Column();
            col1.setName("alpha");
            col1.setColumnType(ColumnType.Numeric);
            col1.setData(new ArrayList<>(Arrays.asList(1.0, 2.0)));

            Column col2 = new Column();
            col2.setName("beta");
            col2.setColumnType(ColumnType.Numeric);
            col2.setData(new ArrayList<>(Arrays.asList(3.0, 4.0)));

            df.addColumn(col1);
            df.addColumn(col2);

            Column found = df.getColumnByName("alpha");
            assertNotNull(found, "Should find column 'alpha'");
            assertEquals("alpha", found.getName(), "Found column name should be alpha");

            Column notFound = df.getColumnByName("gamma");
            assertNull(notFound, "Should return null for non-existent column");

            r.pass("getColumnByName finds existing columns and returns null for missing ones");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDataFrameGetColumnByIndex() {
        TestResult.Recorder recorder = new TestResult.Recorder("data", "test_docs/results");
        TestResult r = recorder.record("dataframe", "get_column_by_index");
        try {
            DataFrame df = new DataFrame();
            Column col1 = new Column();
            col1.setName("first");
            col1.setColumnType(ColumnType.Numeric);
            col1.setData(new ArrayList<>(Arrays.asList(1.0)));

            Column col2 = new Column();
            col2.setName("second");
            col2.setColumnType(ColumnType.Numeric);
            col2.setData(new ArrayList<>(Arrays.asList(2.0)));

            df.addColumn(col1);
            df.addColumn(col2);

            Column at0 = df.getColumn(0);
            assertEquals("first", at0.getName(), "Column at index 0 should be 'first'");

            Column at1 = df.getColumn(1);
            assertEquals("second", at1.getName(), "Column at index 1 should be 'second'");

            // Test negative index
            Column atNeg1 = df.getColumn(-1);
            assertEquals("second", atNeg1.getName(), "Column at index -1 should be 'second'");

            r.pass("getColumn by index works for positive and negative indices");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDataFrameRemoveColumn() {
        TestResult.Recorder recorder = new TestResult.Recorder("data", "test_docs/results");
        TestResult r = recorder.record("dataframe", "remove_column");
        try {
            DataFrame df = new DataFrame();
            Column col1 = new Column();
            col1.setName("A");
            col1.setColumnType(ColumnType.Numeric);
            col1.setData(new ArrayList<>(Arrays.asList(1.0)));

            Column col2 = new Column();
            col2.setName("B");
            col2.setColumnType(ColumnType.Numeric);
            col2.setData(new ArrayList<>(Arrays.asList(2.0)));

            Column col3 = new Column();
            col3.setName("C");
            col3.setColumnType(ColumnType.Numeric);
            col3.setData(new ArrayList<>(Arrays.asList(3.0)));

            df.addColumn(col1);
            df.addColumn(col2);
            df.addColumn(col3);

            Column removed = df.removeColumn(1);
            assertEquals("B", removed.getName(), "Removed column should be B");
            assertEquals(2, df.getColumnCount(), "Should have 2 columns after removal");
            assertEquals("A", df.getColumnNames().get(0), "First column should still be A");
            assertEquals("C", df.getColumnNames().get(1), "Second column should now be C");

            r.pass("removeColumn removes correct column and shifts remaining columns");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testColumnToStringList() {
        TestResult.Recorder recorder = new TestResult.Recorder("data", "test_docs/results");
        TestResult r = recorder.record("column", "to_string_list");
        try {
            Column col = new Column();
            col.setName("mixed");
            col.setColumnType(ColumnType.Numeric);
            col.setData(new ArrayList<>(Arrays.asList(1.0, 2.5, 3.0)));

            List<String> strings = col.toStringList();
            assertEquals(3, strings.size(), "Should have 3 string elements");
            assertEquals("1.0", strings.get(0), "First element should be '1.0'");
            assertEquals("2.5", strings.get(1), "Second element should be '2.5'");

            r.pass("toStringList converts numeric data to string list correctly");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDataFrameGetColumnTypes() {
        TestResult.Recorder recorder = new TestResult.Recorder("data", "test_docs/results");
        TestResult r = recorder.record("dataframe", "get_column_types");
        try {
            DataFrame df = new DataFrame();
            Column col1 = new Column();
            col1.setName("num");
            col1.setColumnType(ColumnType.Numeric);
            col1.setData(new ArrayList<>(Arrays.asList(1.0)));

            Column col2 = new Column();
            col2.setName("str");
            col2.setColumnType(ColumnType.String);
            col2.setData(new ArrayList<>(Arrays.asList("hello")));

            df.addColumn(col1);
            df.addColumn(col2);

            List<ColumnType> types = df.getColumnTypes();
            assertEquals(2, types.size(), "Should have 2 types");
            assertEquals(ColumnType.Numeric, types.get(0), "First type should be Numeric");
            assertEquals(ColumnType.String, types.get(1), "Second type should be String");

            r.pass("getColumnTypes returns [Numeric, String] correctly");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDataFrameSliceColumnWithStep() {
        TestResult.Recorder recorder = new TestResult.Recorder("data", "test_docs/results");
        TestResult r = recorder.record("dataframe", "slice_column_with_step");
        try {
            DataFrame df = new DataFrame();
            for (int i = 0; i < 6; i++) {
                Column col = new Column();
                col.setName("col" + i);
                col.setColumnType(ColumnType.Numeric);
                col.setData(new ArrayList<>(Arrays.asList((double) i)));
                df.addColumn(col);
            }

            DataFrame sliced = df.sliceColumn(0, 6, 2);
            assertEquals(3, sliced.getColumnCount(), "Sliced with step 2 should have 3 columns");
            assertEquals("col0", sliced.getColumnNames().get(0), "First should be col0");
            assertEquals("col2", sliced.getColumnNames().get(1), "Second should be col2");
            assertEquals("col4", sliced.getColumnNames().get(2), "Third should be col4");

            r.pass("sliceColumn with step works: slice(0,6,2) -> col0, col2, col4");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }

    @Test
    @Timeout(value = 10)
    public void testDataFrameToString() {
        TestResult.Recorder recorder = new TestResult.Recorder("data", "test_docs/results");
        TestResult r = recorder.record("dataframe", "to_string");
        try {
            DataFrame df = new DataFrame();
            Column col = new Column();
            col.setName("test");
            col.setColumnType(ColumnType.Numeric);
            col.setData(new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0)));
            df.addColumn(col);

            String str = df.toString();
            assertNotNull(str, "toString should not return null");
            assertTrue(str.contains("3 rows"), "toString should contain row count");
            assertTrue(str.contains("1 columns"), "toString should contain column count");
            assertTrue(str.contains("test"), "toString should contain column name");

            DataFrame emptyDf = new DataFrame();
            String emptyStr = emptyDf.toString();
            assertEquals("Empty DataFrame", emptyStr, "Empty DataFrame toString should be 'Empty DataFrame'");

            r.pass("toString returns meaningful representation for DataFrame");
        } catch (AssertionError e) {
            r.fail(e.getMessage());
            throw e;
        } finally {
            recorder.writeToFile();
        }
    }
}
