package com.yishape.lab.math.data;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for {@link DataFrame} class.
 */
class DataFrameTest {

    // ==================== Construction ====================

    @Test
    void emptyDataFrame() {
        DataFrame df = new DataFrame();
        assertEquals(0, df.getRowCount());
        assertEquals(0, df.getColumnCount());
        assertTrue(df.isEmpty());
    }

    @Test
    void constructFromColumns() {
        Column c1 = new Column();
        c1.setName("x");
        c1.setColumnType(ColumnType.Numeric);
        c1.setData(Arrays.asList(1.0, 2.0, 3.0));

        Column c2 = new Column();
        c2.setName("y");
        c2.setColumnType(ColumnType.Numeric);
        c2.setData(Arrays.asList(4.0, 5.0, 6.0));

        DataFrame df = new DataFrame(Arrays.asList(c1, c2));
        assertEquals(3, df.getRowCount());
        assertEquals(2, df.getColumnCount());
        assertEquals(Arrays.asList("x", "y"), df.getColumnNames());
    }

    @Test
    void shape() {
        Column c = new Column();
        c.setName("a");
        c.setColumnType(ColumnType.Numeric);
        c.setData(Arrays.asList(1.0, 2.0));
        DataFrame df = new DataFrame(List.of(c));
        assertArrayEquals(new int[]{2, 1}, df.shape());
    }

    // ==================== Column Access ====================

    @Test
    void getColumnByName() {
        Column c = new Column();
        c.setName("age");
        c.setColumnType(ColumnType.Numeric);
        c.setData(Arrays.asList(25.0, 30.0));
        DataFrame df = new DataFrame(List.of(c));

        assertNotNull(df.getColumnByName("age"));
        assertNull(df.getColumnByName("nonexistent"));
        assertNull(df.getColumnByName(null));
    }

    @Test
    void get_positiveIndex() {
        Column c1 = new Column();
        c1.setName("a");
        c1.setColumnType(ColumnType.Numeric);
        c1.setData(Arrays.asList(1.0));

        Column c2 = new Column();
        c2.setName("b");
        c2.setColumnType(ColumnType.Numeric);
        c2.setData(Arrays.asList(2.0));

        DataFrame df = new DataFrame(Arrays.asList(c1, c2));
        assertEquals("b", df.get(1).getName());
    }

    @Test
    void get_negativeIndex() {
        Column c1 = new Column();
        c1.setName("a");
        c1.setColumnType(ColumnType.Numeric);
        c1.setData(Arrays.asList(1.0));

        Column c2 = new Column();
        c2.setName("b");
        c2.setColumnType(ColumnType.Numeric);
        c2.setData(Arrays.asList(2.0));

        DataFrame df = new DataFrame(Arrays.asList(c1, c2));
        assertEquals("b", df.get(-1).getName());
    }

    // ==================== Add/Remove Column ====================

    @Test
    void addColumn() {
        DataFrame df = new DataFrame();
        Column c = new Column();
        c.setName("x");
        c.setColumnType(ColumnType.Numeric);
        c.setData(Arrays.asList(1.0, 2.0));
        df.addColumn(c);
        assertEquals(1, df.getColumnCount());
        assertEquals(2, df.getRowCount());
    }

    @Test
    void addColumn_null_throws() {
        DataFrame df = new DataFrame();
        assertThrows(IllegalArgumentException.class, () -> df.addColumn(null));
    }

    @Test
    void addColumn_wrongSize_throws() {
        Column c1 = new Column();
        c1.setName("a");
        c1.setColumnType(ColumnType.Numeric);
        c1.setData(Arrays.asList(1.0, 2.0));
        DataFrame df = new DataFrame(List.of(c1));

        Column c2 = new Column();
        c2.setName("b");
        c2.setColumnType(ColumnType.Numeric);
        c2.setData(Arrays.asList(1.0)); // wrong size
        assertThrows(IllegalArgumentException.class, () -> df.addColumn(c2));
    }

    @Test
    void removeColumn() {
        Column c1 = new Column();
        c1.setName("a");
        c1.setColumnType(ColumnType.Numeric);
        c1.setData(Arrays.asList(1.0));

        Column c2 = new Column();
        c2.setName("b");
        c2.setColumnType(ColumnType.Numeric);
        c2.setData(Arrays.asList(2.0));

        DataFrame df = new DataFrame(Arrays.asList(c1, c2));
        Column removed = df.removeColumn(0);
        assertEquals("a", removed.getName());
        assertEquals(1, df.getColumnCount());
    }

    // ==================== Copy & Clear ====================

    @Test
    void copy_isDeepCopy() {
        Column c = new Column();
        c.setName("x");
        c.setColumnType(ColumnType.Numeric);
        c.setData(new ArrayList<>(Arrays.asList(1.0, 2.0)));
        DataFrame df = new DataFrame(List.of(c));

        DataFrame copy = df.copy();
        assertEquals(1, copy.getColumnCount());
        assertEquals(2, copy.getRowCount());

        // Modify original, copy should be unaffected
        c.getData().set(0, 99.0);
        assertEquals(1.0, copy.get(0).getData().get(0));
    }

    @Test
    void clear_emptiesDataFrame() {
        Column c = new Column();
        c.setName("x");
        c.setColumnType(ColumnType.Numeric);
        c.setData(Arrays.asList(1.0));
        DataFrame df = new DataFrame(List.of(c));

        df.clear();
        assertTrue(df.isEmpty());
        assertEquals(0, df.getRowCount());
        assertEquals(0, df.getColumnCount());
    }

    // ==================== Column Selection ====================

    @Test
    void selectNumeric() {
        Column c1 = new Column();
        c1.setName("num");
        c1.setColumnType(ColumnType.Numeric);
        c1.setData(Arrays.asList(1.0));

        Column c2 = new Column();
        c2.setName("str");
        c2.setColumnType(ColumnType.String);
        c2.setData(Arrays.asList("hello"));

        DataFrame df = new DataFrame(Arrays.asList(c1, c2));
        DataFrame numeric = df.selectNumeric();
        assertEquals(1, numeric.getColumnCount());
        assertEquals("num", numeric.getColumnNames().get(0));
    }

    @Test
    void selectByType() {
        Column c1 = new Column();
        c1.setName("a");
        c1.setColumnType(ColumnType.Numeric);
        c1.setData(Arrays.asList(1.0));

        Column c2 = new Column();
        c2.setName("b");
        c2.setColumnType(ColumnType.String);
        c2.setData(Arrays.asList("x"));

        DataFrame df = new DataFrame(Arrays.asList(c1, c2));
        DataFrame strings = df.selectByType(ColumnType.String);
        assertEquals(1, strings.getColumnCount());
        assertEquals("b", strings.getColumnNames().get(0));
    }

    @Test
    void selectByName_wildcard() {
        Column c1 = new Column();
        c1.setName("train_x");
        c1.setColumnType(ColumnType.Numeric);
        c1.setData(Arrays.asList(1.0));

        Column c2 = new Column();
        c2.setName("train_y");
        c2.setColumnType(ColumnType.Numeric);
        c2.setData(Arrays.asList(2.0));

        Column c3 = new Column();
        c3.setName("test_x");
        c3.setColumnType(ColumnType.Numeric);
        c3.setData(Arrays.asList(3.0));

        DataFrame df = new DataFrame(Arrays.asList(c1, c2, c3));
        DataFrame selected = df.selectByName("train_*");
        assertEquals(2, selected.getColumnCount());
    }

    @Test
    void excludeByName() {
        Column c1 = new Column();
        c1.setName("x");
        c1.setColumnType(ColumnType.Numeric);
        c1.setData(Arrays.asList(1.0));

        Column c2 = new Column();
        c2.setName("y");
        c2.setColumnType(ColumnType.Numeric);
        c2.setData(Arrays.asList(2.0));

        DataFrame df = new DataFrame(Arrays.asList(c1, c2));
        DataFrame excluded = df.excludeByName("y");
        assertEquals(1, excluded.getColumnCount());
        assertEquals("x", excluded.getColumnNames().get(0));
    }

    // ==================== Missing Values ====================

    @Test
    void hasMissingValues_false() {
        Column c = new Column();
        c.setName("x");
        c.setColumnType(ColumnType.Numeric);
        c.setData(Arrays.asList(1.0, 2.0));
        DataFrame df = new DataFrame(List.of(c));
        assertFalse(df.hasMissingValues());
    }

    @Test
    void hasMissingValues_true() {
        Column c = new Column();
        c.setName("x");
        c.setColumnType(ColumnType.Numeric);
        c.setData(Arrays.asList(1.0, null, 3.0));
        DataFrame df = new DataFrame(List.of(c));
        assertTrue(df.hasMissingValues());
    }

    @Test
    void dropNa() {
        Column c = new Column();
        c.setName("x");
        c.setColumnType(ColumnType.Numeric);
        c.setData(new ArrayList<>(Arrays.asList(1.0, null, 3.0)));
        DataFrame df = new DataFrame(List.of(c));

        DataFrame cleaned = df.dropNa();
        assertEquals(2, cleaned.getRowCount());
    }

    @Test
    void fillNa() {
        Column c = new Column();
        c.setName("x");
        c.setColumnType(ColumnType.Numeric);
        c.setData(new ArrayList<>(Arrays.asList(1.0, null, 3.0)));
        DataFrame df = new DataFrame(List.of(c));

        DataFrame filled = df.fillNa(0.0);
        assertEquals(0.0, filled.get(0).getData().get(1));
    }

    @Test
    void missingCount() {
        Column c = new Column();
        c.setName("x");
        c.setColumnType(ColumnType.Numeric);
        c.setData(Arrays.asList(1.0, null, Double.NaN, 4.0));
        DataFrame df = new DataFrame(List.of(c));
        assertEquals(2, df.missingCount());
    }

    // ==================== Matrix Conversion ====================

    @Test
    void toMatrix() {
        Column c1 = new Column();
        c1.setName("x");
        c1.setColumnType(ColumnType.Numeric);
        c1.setData(Arrays.asList(1.0, 2.0));

        Column c2 = new Column();
        c2.setName("y");
        c2.setColumnType(ColumnType.Numeric);
        c2.setData(Arrays.asList(3.0, 4.0));

        DataFrame df = new DataFrame(Arrays.asList(c1, c2));
        IMatrix<Double> mat = df.toMatrix();
        assertEquals(2, mat.rows());
        assertEquals(2, mat.cols());
        assertEquals(1.0, mat.get(0, 0), 1e-10);
        assertEquals(4.0, mat.get(1, 1), 1e-10);
    }

    @Test
    void toMatrix_noNumericColumns_throws() {
        Column c = new Column();
        c.setName("name");
        c.setColumnType(ColumnType.String);
        c.setData(Arrays.asList("Alice", "Bob"));
        DataFrame df = new DataFrame(List.of(c));
        assertThrows(IllegalStateException.class, df::toMatrix);
    }

    @Test
    void toVectors() {
        Column c = new Column();
        c.setName("x");
        c.setColumnType(ColumnType.Numeric);
        c.setData(Arrays.asList(1.0, 2.0));
        DataFrame df = new DataFrame(List.of(c));

        List<IVector<Double>> vecs = df.toVectors();
        assertEquals(1, vecs.size());
        assertEquals(2, vecs.get(0).size());
    }

    // ==================== Describe & Info ====================

    @Test
    void describe_returnsStatisticsMatrix() {
        Column c = new Column();
        c.setName("x");
        c.setColumnType(ColumnType.Numeric);
        c.setData(Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0));
        DataFrame df = new DataFrame(List.of(c));

        IMatrix<Double> desc = df.describe();
        assertNotNull(desc);
        // 8 rows: count, mean, std, min, q1, median, q3, max
        assertEquals(8, desc.rows());
    }

    @Test
    void describeColumns() {
        Column c = new Column();
        c.setName("x");
        c.setColumnType(ColumnType.Numeric);
        c.setData(Arrays.asList(1.0));
        DataFrame df = new DataFrame(List.of(c));
        assertEquals(List.of("x"), df.describeColumns());
    }

    @Test
    void describeIndex() {
        List<String> index = new DataFrame().describeIndex();
        assertEquals(8, index.size());
        assertTrue(index.contains("mean"));
        assertTrue(index.contains("std"));
    }

    @Test
    void info() {
        Column c = new Column();
        c.setName("x");
        c.setColumnType(ColumnType.Numeric);
        c.setData(Arrays.asList(1.0, 2.0));
        DataFrame df = new DataFrame(List.of(c));

        Map<String, Object> info = df.info();
        assertNotNull(info);
        assertFalse(info.isEmpty());
    }

    // ==================== CSV I/O ====================

    @Test
    void csvRoundTrip(@TempDir Path tempDir) throws IOException {
        Column c1 = new Column();
        c1.setName("x");
        c1.setColumnType(ColumnType.Numeric);
        c1.setData(Arrays.asList(1.0, 2.0, 3.0));

        Column c2 = new Column();
        c2.setName("name");
        c2.setColumnType(ColumnType.String);
        c2.setData(Arrays.asList("a", "b", "c"));

        DataFrame original = new DataFrame(Arrays.asList(c1, c2));
        File file = tempDir.resolve("test.csv").toFile();
        original.toCsv(file.getAbsolutePath());

        DataFrame loaded = DataFrame.readCsv(file.getAbsolutePath());
        assertEquals(3, loaded.getRowCount());
        assertEquals(2, loaded.getColumnCount());
        assertEquals(Arrays.asList("x", "name"), loaded.getColumnNames());
    }

    @Test
    void toCsv_nullPath_throws() {
        DataFrame df = new DataFrame();
        assertThrows(IllegalArgumentException.class, () -> df.toCsv(null));
    }

    @Test
    void readCsv_nullPath_throws() {
        assertThrows(IllegalArgumentException.class, () -> DataFrame.readCsv(null));
    }

    // ==================== Slicing ====================

    @Test
    void sliceColumn_range() {
        Column c1 = new Column();
        c1.setName("a");
        c1.setColumnType(ColumnType.Numeric);
        c1.setData(Arrays.asList(1.0));

        Column c2 = new Column();
        c2.setName("b");
        c2.setColumnType(ColumnType.Numeric);
        c2.setData(Arrays.asList(2.0));

        Column c3 = new Column();
        c3.setName("c");
        c3.setColumnType(ColumnType.Numeric);
        c3.setData(Arrays.asList(3.0));

        DataFrame df = new DataFrame(Arrays.asList(c1, c2, c3));
        DataFrame sliced = df.sliceColumn(0, 2);
        assertEquals(2, sliced.getColumnCount());
        assertEquals("a", sliced.getColumnNames().get(0));
        assertEquals("b", sliced.getColumnNames().get(1));
    }

    @Test
    void sliceColumn_invalidStep_throws() {
        Column c = new Column();
        c.setName("a");
        c.setColumnType(ColumnType.Numeric);
        c.setData(Arrays.asList(1.0));
        DataFrame df = new DataFrame(List.of(c));
        assertThrows(IllegalArgumentException.class, () -> df.sliceColumn(0, 1, 0));
    }

    // ==================== Boolean Get ====================

    @Test
    void booleanGet_filterRows() {
        Column c = new Column();
        c.setName("x");
        c.setColumnType(ColumnType.Numeric);
        c.setData(new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0)));
        DataFrame df = new DataFrame(List.of(c));

        DataFrame filtered = df.booleanGet(new boolean[]{true, false, true});
        assertEquals(2, filtered.getRowCount());
    }

    @Test
    void booleanGet_wrongLength_throws() {
        Column c = new Column();
        c.setName("x");
        c.setColumnType(ColumnType.Numeric);
        c.setData(Arrays.asList(1.0));
        DataFrame df = new DataFrame(List.of(c));
        assertThrows(IllegalArgumentException.class,
            () -> df.booleanGet(new boolean[]{true, false}));
    }

    // ==================== Fancy Get ====================

    @Test
    void fancyGet_rows() {
        Column c = new Column();
        c.setName("x");
        c.setColumnType(ColumnType.Numeric);
        c.setData(new ArrayList<>(Arrays.asList(10.0, 20.0, 30.0)));
        DataFrame df = new DataFrame(List.of(c));

        DataFrame fancy = df.fancyGet(new int[]{0, 2});
        assertEquals(2, fancy.getRowCount());
        assertEquals(10.0, fancy.get(0).getData().get(0));
        assertEquals(30.0, fancy.get(0).getData().get(1));
    }

    // ==================== ML Preprocessing ====================

    @Test
    void standardize() {
        Column c = new Column();
        c.setName("x");
        c.setColumnType(ColumnType.Numeric);
        c.setData(new ArrayList<>(Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0)));
        DataFrame df = new DataFrame(List.of(c));

        DataFrame std = df.standardize();
        IMatrix<Double> mat = std.toMatrix();
        // Mean should be ~0
        double sum = 0;
        for (int i = 0; i < mat.rows(); i++) {
            sum += mat.get(i, 0);
        }
        assertEquals(0, sum / mat.rows(), 1e-10);
    }

    @Test
    void normalize() {
        Column c = new Column();
        c.setName("x");
        c.setColumnType(ColumnType.Numeric);
        c.setData(new ArrayList<>(Arrays.asList(10.0, 20.0, 30.0)));
        DataFrame df = new DataFrame(List.of(c));

        DataFrame norm = df.normalize();
        IMatrix<Double> mat = norm.toMatrix();
        assertEquals(0.0, mat.get(0, 0), 1e-10);
        assertEquals(1.0, mat.get(2, 0), 1e-10);
    }

    // ==================== Train Test Split ====================

    @Test
    void trainTestSplit() {
        Column c = new Column();
        c.setName("x");
        c.setColumnType(ColumnType.Numeric);
        List<Object> data = new ArrayList<>();
        for (int i = 0; i < 100; i++) data.add((double) i);
        c.setData(data);
        DataFrame df = new DataFrame(List.of(c));

        DataFrame[] splits = df.trainTestSplit(0.8);
        assertEquals(2, splits.length);
        assertEquals(80, splits[0].getRowCount());
        assertEquals(20, splits[1].getRowCount());
    }

    @Test
    void trainTestSplit_invalidRatio_throws() {
        Column c = new Column();
        c.setName("x");
        c.setColumnType(ColumnType.Numeric);
        c.setData(Arrays.asList(1.0));
        DataFrame df = new DataFrame(List.of(c));
        assertThrows(IllegalArgumentException.class, () -> df.trainTestSplit(0));
        assertThrows(IllegalArgumentException.class, () -> df.trainTestSplit(1.5));
    }

    // ==================== Column Types ====================

    @Test
    void getColumnTypes() {
        Column c1 = new Column();
        c1.setName("num");
        c1.setColumnType(ColumnType.Numeric);
        c1.setData(Arrays.asList(1.0));

        Column c2 = new Column();
        c2.setName("str");
        c2.setColumnType(ColumnType.String);
        c2.setData(Arrays.asList("x"));

        DataFrame df = new DataFrame(Arrays.asList(c1, c2));
        List<ColumnType> types = df.getColumnTypes();
        assertEquals(2, types.size());
        assertEquals(ColumnType.Numeric, types.get(0));
        assertEquals(ColumnType.String, types.get(1));
    }

    @Test
    void getColumns_returnsDefensiveCopy() {
        Column c = new Column();
        c.setName("x");
        c.setColumnType(ColumnType.Numeric);
        c.setData(Arrays.asList(1.0));
        DataFrame df = new DataFrame(List.of(c));

        List<Column> cols = df.getColumns();
        cols.clear(); // Should not affect original
        assertEquals(1, df.getColumnCount());
    }
}
