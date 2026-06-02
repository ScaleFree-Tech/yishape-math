package com.yishape.lab.math.data;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IndexExpressionParser;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * 数据框类，用于处理结构化数据，支持从CSV文件读取数据并与IMatrix进行转换。
 * 所有数值操作依赖 {@link Linalg}/{@link IVector}/{@link IMatrix} 实现。
 *
 * <p>
 * 本类提供了类似pandas DataFrame的功能，包括：
 * </p>
 * <ul>
 * <li>从CSV文件读取数据 / Reading data from CSV files</li>
 * <li>列切片和行切片操作 / Column and row slicing operations</li>
 * <li>数据类型自动推断（String/Numeric/Integer/Boolean） / Data type auto-detection</li>
 * <li>缺失值处理（dropNa/fillNa） / Missing value handling</li>
 * <li>与IMatrix的互转换 / Mutual conversion with IMatrix</li>
 * <li>标准化/归一化 / Standardization/Normalization</li>
 * <li>训练/测试集拆分 / Train/test split</li>
 * <li>统计摘要 / Statistical summary</li>
 * </ul>
 *
 * @author lteb2
 * @version 1.1
 * @since 1.0
 */
public class DataFrame implements Serializable {

    private static final long serialVersionUID = 1L;

    /** 列列表，存储所有列数据 */
    private List<Column> columns = new ArrayList<>();

    /** 行数，用于数据验证 */
    private int rowCount = 0;

    /** 列名到索引的映射，用于 O(1) 的 getColumnByName */
    private Map<String, Integer> nameIndex = new HashMap<>();

    /** 标记 nameIndex 是否需要重建 */
    private boolean nameIndexDirty = true;

    // ==================== 构造函数 ====================

    /**
     * 默认构造函数，创建一个空的DataFrame
     */
    public DataFrame() {
        this.columns = new ArrayList<>();
        this.rowCount = 0;
    }

    /**
     * 带列列表的构造函数
     * @param columns 列列表
     */
    public DataFrame(List<Column> columns) {
        this.columns = new ArrayList<>(columns);
        this.rowCount = columns.isEmpty() ? 0 : columns.get(0).getData().size();
        invalidateNameIndex();
    }

    // ==================== CSV 读写 ====================

    /**
     * 从CSV文件读取数据，默认逗号分隔，有表头
     * @param filePath 文件路径
     * @return 创建的DataFrame对象
     * @throws IOException 如果文件读取失败
     */
    public static DataFrame readCsv(String filePath) throws IOException {
        return readCsv(filePath, ",", true);
    }

    /**
     * 从CSV文件读取数据
     * <p>
     * 数值型数据自动转换为 Double，Boolean 识别 "true"/"false"（不区分大小写）。
     * 空字符串或 "NA"/"NaN"/"null" 被视为缺失值，存储为 Double.NaN。
     * 解析过程为单次遍历，无需二次扫描。
     * </p>
     *
     * @param filePath 文件路径
     * @param separator 分隔符
     * @param ifHasHead 是否有表头
     * @return 创建的DataFrame对象
     * @throws IOException 如果文件读取失败
     */
    public static DataFrame readCsv(String filePath, String separator, boolean ifHasHead) throws IOException {
        if (filePath == null || filePath.trim().isEmpty()) {
            throw new IllegalArgumentException("文件路径不能为空 / File path cannot be empty");
        }
        if (separator == null || separator.trim().isEmpty()) {
            throw new IllegalArgumentException("分隔符不能为空 / Separator cannot be empty");
        }

        DataFrame df = new DataFrame();

        try (FileReader reader = new FileReader(filePath);
             CSVParser parser = new CSVParser(reader,
                     CSVFormat.DEFAULT.builder().setDelimiter(separator.charAt(0)).build())) {

            List<CSVRecord> records = parser.getRecords();
            if (records.isEmpty()) {
                return df;
            }

            int numColumns = records.get(0).size();
            String[] columnNames = new String[numColumns];
            List<CSVRecord> dataRecords;

            // 处理表头
            if (ifHasHead && !records.isEmpty()) {
                CSVRecord headerRecord = records.get(0);
                for (int i = 0; i < numColumns; i++) {
                    columnNames[i] = headerRecord.get(i);
                }
                dataRecords = records.subList(1, records.size());
            } else {
                for (int i = 0; i < numColumns; i++) {
                    columnNames[i] = "col_" + i;
                }
                dataRecords = records;
            }

            // 初始化列（单次遍历完成类型推断和数据填充）
            List<List<Object>> rawData = new ArrayList<>();
            for (int i = 0; i < numColumns; i++) {
                rawData.add(new ArrayList<>());
            }

            // 解析状态：null=未决定, true=数值, false=非数值
            Boolean[] colIsNumeric = new Boolean[numColumns];
            Boolean[] colIsBoolean = new Boolean[numColumns];

            // 单次遍历解析所有数据
            for (CSVRecord record : dataRecords) {
                for (int i = 0; i < numColumns; i++) {
                    String value = record.get(i).trim();
                    List<Object> colData = rawData.get(i);

                    // 检测缺失值标记
                    if (value.isEmpty() || value.equalsIgnoreCase("NA")
                            || value.equalsIgnoreCase("NaN") || value.equalsIgnoreCase("null")) {
                        colData.add(Double.NaN);
                        continue;
                    }

                    // 如果已确定为非数值或布尔，直接存字符串
                    if (Boolean.FALSE.equals(colIsNumeric[i]) || Boolean.FALSE.equals(colIsBoolean[i])) {
                        colData.add(value);
                        continue;
                    }

                    // 尝试解析为布尔值
                    if (colIsBoolean[i] == null || Boolean.TRUE.equals(colIsBoolean[i])) {
                        if (value.equalsIgnoreCase("true") || value.equalsIgnoreCase("false")) {
                            colIsBoolean[i] = true;
                            colData.add(Boolean.parseBoolean(value));
                            continue;
                        } else if (colIsBoolean[i] != null) {
                            // 已经确定是布尔但当前不是，标记为非布尔
                            colIsBoolean[i] = false;
                            colData.add(value);
                            continue;
                        }
                    }

                    // 尝试解析为数值
                    try {
                        double d = Double.parseDouble(value);
                        colData.add(d);
                        colIsNumeric[i] = true;
                    } catch (NumberFormatException e) {
                        colIsNumeric[i] = false;
                        colData.add(value);
                    }
                }
            }

            // 创建 Column 并设置类型
            for (int i = 0; i < numColumns; i++) {
                Column column = new Column();
                column.setName(columnNames[i]);
                column.setData(rawData.get(i));

                if (Boolean.TRUE.equals(colIsBoolean[i])) {
                    column.setColumnType(ColumnType.Boolean);
                } else if (Boolean.TRUE.equals(colIsNumeric[i])) {
                    // 检查是否全为整数
                    List<Object> colData = rawData.get(i);
                    boolean allInteger = true;
                    for (Object v : colData) {
                        if (v instanceof Double d) {
                            if (d != Math.floor(d)) {
                                allInteger = false;
                                break;
                            }
                        } else if (v instanceof String) {
                            allInteger = false;
                            break;
                        }
                    }
                    column.setColumnType(allInteger ? ColumnType.Integer : ColumnType.Numeric);
                } else {
                    column.setColumnType(ColumnType.String);
                }

                df.columns.add(column);
            }

            df.rowCount = dataRecords.size();
            df.invalidateNameIndex();
        }

        return df;
    }

    /**
     * 写入CSV文件
     * @param filePath 文件路径
     * @param separator 分隔符
     * @throws IOException 如果文件写入失败
     */
    public void toCsv(String filePath, String separator) throws IOException {
        if (filePath == null || filePath.trim().isEmpty()) {
            throw new IllegalArgumentException("文件路径不能为空 / File path cannot be empty");
        }
        if (separator == null || separator.trim().isEmpty()) {
            throw new IllegalArgumentException("分隔符不能为空 / Separator cannot be empty");
        }

        try (FileWriter writer = new FileWriter(filePath);
             CSVPrinter printer = new CSVPrinter(writer,
                     CSVFormat.DEFAULT.builder().setDelimiter(separator.charAt(0)).build())) {

            printer.printRecord(columns.stream().map(Column::getName).toArray());

            if (rowCount > 0) {
                for (int i = 0; i < rowCount; i++) {
                    Object[] rowData = new Object[columns.size()];
                    for (int j = 0; j < columns.size(); j++) {
                        List<Object> columnData = columns.get(j).getData();
                        if (i < columnData.size()) {
                            Object v = columnData.get(i);
                            // 将 NaN 转换回空字符串
                            if (v instanceof Double d && Double.isNaN(d)) {
                                rowData[j] = "";
                            } else {
                                rowData[j] = v;
                            }
                        } else {
                            rowData[j] = "";
                        }
                    }
                    printer.printRecord(rowData);
                }
            }
        }
    }

    /**
     * 写入CSV文件，默认逗号分隔
     * @param filePath 文件路径
     * @throws IOException 如果文件写入失败
     */
    public void toCsv(String filePath) throws IOException {
        toCsv(filePath, ",");
    }

    // ==================== 切片操作 ====================

    /**
     * 列切片操作（带步长）
     */
    public DataFrame sliceColumn(int start, int end, int step) {
        if (step <= 0) {
            throw new IllegalArgumentException("步长必须大于0 / Step must be greater than 0");
        }
        if (columns.isEmpty()) {
            return new DataFrame();
        }

        int actualStart = normalizeIndex(start, columns.size(), true);
        int actualEnd = normalizeIndex(end, columns.size(), false);

        if (actualStart >= actualEnd) {
            return new DataFrame();
        }

        DataFrame result = new DataFrame();
        for (int i = actualStart; i < actualEnd && i < columns.size(); i += step) {
            result.addColumn(columns.get(i));
        }
        return result;
    }

    /**
     * 列切片操作（步长为1）
     */
    public DataFrame sliceColumn(int start, int end) {
        return sliceColumn(start, end, 1);
    }

    /**
     * 列切片操作（从指定位置到末尾）
     */
    public DataFrame sliceColumn(int start) {
        return sliceColumn(start, columns.size(), 1);
    }

    /**
     * 通用切片操作，支持 NumPy 风格切片语法
     * @param rowExp 行切片表达式，如 "1:5" 或 "1:5:2"，null 表示所有行
     * @param colExp 列切片表达式，如 "1:5" 或 "1:5:2"，null 表示所有列
     * @return 切片后的DataFrame
     */
    public DataFrame slice(String rowExp, String colExp) {
        DataFrame result = new DataFrame();

        // 解析列切片
        if (colExp != null && !colExp.trim().isEmpty()) {
            int[] colIndices = IndexExpressionParser.generateIndices(
                    IndexExpressionParser.parse(colExp, columns.size()));
            for (int colIndex : colIndices) {
                if (colIndex >= 0 && colIndex < columns.size()) {
                    Column originalColumn = columns.get(colIndex);
                    Column newColumn = createColumnCopy(originalColumn);
                    result.addColumn(newColumn);
                }
            }
        } else {
            for (Column column : columns) {
                result.addColumn(createColumnCopy(column));
            }
        }

        // 解析行切片
        if (rowExp != null && !rowExp.trim().isEmpty()) {
            int[] rowIndices = IndexExpressionParser.generateIndices(
                    IndexExpressionParser.parse(rowExp, rowCount));
            for (Column column : result.columns) {
                List<Object> newData = new ArrayList<>();
                for (int rowIndex : rowIndices) {
                    if (rowIndex >= 0 && rowIndex < column.getData().size()) {
                        newData.add(column.getData().get(rowIndex));
                    }
                }
                column.setData(newData);
            }
            result.rowCount = rowIndices.length;
        }

        return result;
    }

    /**
     * 根据行索引数组进行花式索引获取（行选择），行为与 {@link IMatrix#fancyGet(int[], int[])} 一致。
     * 支持负数索引，委托 {@link IndexExpressionParser#resolveFancyIndex(int[], int)} 处理。
     * @param rowIndices 要选择的行索引数组（支持负数）
     * @return 新的DataFrame
     */
    public DataFrame fancyGet(int[] rowIndices) {
        IndexExpressionParser.FancyIndexResult resolved =
                IndexExpressionParser.resolveFancyIndex(rowIndices, rowCount);
        DataFrame result = new DataFrame();
        for (Column column : columns) {
            List<Object> newData = new ArrayList<>();
            for (int idx : resolved.indices) {
                newData.add(column.getData().get(idx));
            }
            Column newColumn = createColumnCopy(column);
            newColumn.setData(newData);
            result.addColumn(newColumn);
        }
        result.rowCount = resolved.indices.length;
        return result;
    }

    /**
     * 根据行和列索引数组进行花式索引获取，行为与 {@link IMatrix#fancyGet(int[], int[])} 一致。
     * 支持负数索引。
     * @param rowIndices 行索引数组（支持负数）
     * @param colIndices 列索引数组（支持负数）
     * @return 新的DataFrame
     */
    public DataFrame fancyGet(int[] rowIndices, int[] colIndices) {
        IndexExpressionParser.FancyIndexResult resolvedRows =
                IndexExpressionParser.resolveFancyIndex(rowIndices, rowCount);
        IndexExpressionParser.FancyIndexResult resolvedCols =
                IndexExpressionParser.resolveFancyIndex(colIndices, columns.size());

        DataFrame result = new DataFrame();
        for (int ci : resolvedCols.indices) {
            Column originalColumn = columns.get(ci);
            List<Object> newData = new ArrayList<>();
            for (int ri : resolvedRows.indices) {
                newData.add(originalColumn.getData().get(ri));
            }
            Column newColumn = createColumnCopy(originalColumn);
            newColumn.setData(newData);
            result.addColumn(newColumn);
        }
        result.rowCount = resolvedRows.indices.length;
        return result;
    }

    /**
     * 根据布尔数组进行布尔索引获取（行过滤），行为与 {@link IMatrix#booleanGet(boolean[])} 一致。
     * @param rowMask 行布尔索引数组，长度必须等于行数
     * @return 由 mask 为 true 的行组成的新DataFrame
     * @throws IllegalArgumentException 如果数组长度与行数不匹配
     */
    public DataFrame booleanGet(boolean[] rowMask) {
        if (rowMask.length != rowCount) {
            throw new IllegalArgumentException(
                    "布尔掩码长度(" + rowMask.length + ")与行数(" + rowCount + ")不匹配 / Boolean mask length mismatch");
        }
        IndexExpressionParser.BooleanIndexResult resolved =
                IndexExpressionParser.resolveBooleanIndex(rowMask);
        int[] indices = resolved.trueIndices;
        DataFrame result = new DataFrame();
        for (Column column : columns) {
            List<Object> newData = new ArrayList<>();
            for (int idx : indices) {
                newData.add(column.getData().get(idx));
            }
            Column newColumn = createColumnCopy(column);
            newColumn.setData(newData);
            result.addColumn(newColumn);
        }
        result.rowCount = resolved.count;
        return result;
    }

    /**
     * 根据条件数组筛选行，与 {@link IMatrix#booleanGet(boolean[])} 等价。
     * @param condition 行条件数组，与行数等长
     * @return 满足条件的行组成的新DataFrame
     * @throws IllegalArgumentException 如果条件数组长度与行数不匹配
     */
    public DataFrame where(boolean[] condition) {
        return booleanGet(condition);
    }

    // ==================== @Deprecated 别名（兼容旧代码） ====================

    /**
     * @deprecated 使用 {@link #fancyGet(int[])} 代替
     */
    @Deprecated
    public DataFrame fancySlice(int[] rowIndices) {
        return fancyGet(rowIndices);
    }

    /**
     * @deprecated 使用 {@link #booleanGet(boolean[])} 代替
     */
    @Deprecated
    public DataFrame fancySlice(boolean[] rowMask) {
        return booleanGet(rowMask);
    }

    // ==================== Matrix 转换 ====================

    /**
     * 将所有数值列（Numeric/Integer）转换为 IMatrix<Double>。
     * 依赖 {@link Linalg} 创建矩阵。
     * 缺失值（NaN）会被保留在矩阵中。
     *
     * @return 包含所有数值列数据的矩阵，形状为 (行数, 数值列数)
     * @throws IllegalStateException 如果没有数值列
     */
    public IMatrix<Double> toMatrix() {
        List<Column> numericColumns = columns.stream()
                .filter(c -> c.getColumnType().isNumeric())
                .collect(Collectors.toList());

        if (numericColumns.isEmpty()) {
            throw new IllegalStateException("没有数值列可以转换为矩阵 / No numeric columns available for matrix conversion");
        }

        int actualRowCount = numericColumns.get(0).getData().size();
        if (actualRowCount == 0) {
            return Linalg.matrix(new double[0][0]);
        }

        for (Column column : numericColumns) {
            if (column.getData().size() != actualRowCount) {
                throw new IllegalStateException("数值列数据长度不一致 / Numeric column data lengths are inconsistent");
            }
        }

        double[][] matrixData = new double[actualRowCount][numericColumns.size()];
        for (int i = 0; i < actualRowCount; i++) {
            for (int j = 0; j < numericColumns.size(); j++) {
                Object value = numericColumns.get(j).getData().get(i);
                if (value instanceof Number number) {
                    matrixData[i][j] = number.doubleValue();
                } else {
                    matrixData[i][j] = Double.NaN;
                }
            }
        }

        return Linalg.matrix(matrixData);
    }

    /**
     * 将数值列转换为 IVector 列表，便于利用 linalg 的向量操作。
     * @return 数值列对应的 IVector 列表
     */
    public List<IVector<Double>> toVectors() {
        List<IVector<Double>> vectors = new ArrayList<>();
        for (Column column : columns) {
            if (column.getColumnType().isNumeric()) {
                vectors.add(column.toVec());
            }
        }
        return vectors;
    }

    // ==================== 缺失值处理 ====================

    /**
     * 删除包含缺失值的行
     * @return 新的DataFrame，不包含含缺失值的行
     */
    public DataFrame dropNa() {
        if (rowCount == 0) return new DataFrame();

        // 找出所有有效行的索引
        List<Integer> validRowIndices = new ArrayList<>();
        for (int i = 0; i < rowCount; i++) {
            boolean hasMissing = false;
            for (Column col : columns) {
                Object v = col.getData().get(i);
                if (v == null || (v instanceof Double d && Double.isNaN(d))) {
                    hasMissing = true;
                    break;
                }
            }
            if (!hasMissing) {
                validRowIndices.add(i);
            }
        }

        int[] indices = validRowIndices.stream().mapToInt(Integer::intValue).toArray();
        return fancyGet(indices);
    }

    /**
     * 用指定值填充缺失值
     * @param fillValue 填充值
     * @return 新的DataFrame，缺失值被填充
     */
    public DataFrame fillNa(double fillValue) {
        DataFrame result = new DataFrame();
        for (Column col : columns) {
            Column newCol = createColumnCopy(col);
            List<Object> newData = new ArrayList<>();
            for (Object v : col.getData()) {
                if (v == null || (v instanceof Double d && Double.isNaN(d))) {
                    newData.add(fillValue);
                } else {
                    newData.add(v);
                }
            }
            newCol.setData(newData);
            result.addColumn(newCol);
        }
        return result;
    }

    /**
     * 用列均值填充数值列的缺失值
     * @return 新的DataFrame
     */
    public DataFrame fillNaWithMean() {
        DataFrame result = new DataFrame();
        for (Column col : columns) {
            Column newCol = createColumnCopy(col);
            if (col.getColumnType().isNumeric()) {
                IVector<Double> vec = col.toVec();
                double mean = vec.meanValue();
                List<Object> newData = new ArrayList<>();
                for (Object v : col.getData()) {
                    if (v == null || (v instanceof Double d && Double.isNaN(d))) {
                        newData.add(mean);
                    } else {
                        newData.add(v);
                    }
                }
                newCol.setData(newData);
            }
            result.addColumn(newCol);
        }
        return result;
    }

    /**
     * 检查是否包含缺失值
     * @return 是否包含缺失值
     */
    public boolean hasMissingValues() {
        for (Column col : columns) {
            if (col.hasMissingValues()) return true;
        }
        return false;
    }

    /**
     * 返回缺失值总数
     * @return 缺失值数量
     */
    public int missingCount() {
        int count = 0;
        for (Column col : columns) {
            count += col.missingCount();
        }
        return count;
    }

    // ==================== 数据选择器 ====================

    /**
     * 按类型选择列
     * @param type 要选择的列类型
     * @return 包含所有指定类型列的新DataFrame
     */
    public DataFrame selectByType(ColumnType type) {
        DataFrame result = new DataFrame();
        for (Column col : columns) {
            if (col.getColumnType() == type) {
                result.addColumn(createColumnCopy(col));
            }
        }
        return result;
    }

    /**
     * 选择所有数值列
     * @return 包含所有数值列的新DataFrame
     */
    public DataFrame selectNumeric() {
        DataFrame result = new DataFrame();
        for (Column col : columns) {
            if (col.getColumnType().isNumeric()) {
                result.addColumn(createColumnCopy(col));
            }
        }
        return result;
    }

    /**
     * 按列名模式选择列，支持通配符 * 匹配任意字符
     * @param pattern 列名模式，如 "feat_*" 匹配所有以 feat_ 开头的列
     * @return 匹配的新DataFrame
     */
    public DataFrame selectByName(String pattern) {
        String regex = pattern.replace("*", ".*");
        Pattern p = Pattern.compile("^" + regex + "$");
        DataFrame result = new DataFrame();
        for (Column col : columns) {
            if (p.matcher(col.getName()).matches()) {
                result.addColumn(createColumnCopy(col));
            }
        }
        return result;
    }

    /**
     * 按列名列表选择列
     * @param columnNames 要选择的列名列表
     * @return 包含指定列的新DataFrame
     */
    public DataFrame selectColumns(List<String> columnNames) {
        DataFrame result = new DataFrame();
        for (String name : columnNames) {
            Column col = getColumnByName(name);
            if (col != null) {
                result.addColumn(createColumnCopy(col));
            }
        }
        return result;
    }

    /**
     * 排除指定列名模式的列
     * @param pattern 要排除的列名模式，支持 *
     * @return 不包含匹配列的新DataFrame
     */
    public DataFrame excludeByName(String pattern) {
        String regex = pattern.replace("*", ".*");
        Pattern p = Pattern.compile("^" + regex + "$");
        DataFrame result = new DataFrame();
        for (Column col : columns) {
            if (!p.matcher(col.getName()).matches()) {
                result.addColumn(createColumnCopy(col));
            }
        }
        return result;
    }

    // ==================== 统计摘要 ====================

    /**
     * 返回数值列的统计摘要，依赖 {@link IVector} 的统计方法。
     * @return 统计摘要，每行是一个指标，每列是一个数值列
     */
    public IMatrix<Double> describe() {
        List<Column> numericCols = columns.stream()
                .filter(c -> c.getColumnType().isNumeric())
                .collect(Collectors.toList());

        if (numericCols.isEmpty()) {
            return Linalg.matrix(new double[0][0]);
        }

        // 统计指标顺序：count, mean, std, min, q1, median, q3, max
        String[] statsNames = {"count", "mean", "std", "min", "q1", "median", "q3", "max"};
        double[][] statsData = new double[statsNames.length][numericCols.size()];

        for (int j = 0; j < numericCols.size(); j++) {
            IVector<Double> vec = numericCols.get(j).toVec();
            statsData[0][j] = numericCols.get(j).validCount();
            statsData[1][j] = vec.meanValue();
            statsData[2][j] = vec.stdValue();
            statsData[3][j] = vec.minValue();
            statsData[4][j] = vec.q1();
            statsData[5][j] = vec.median();
            statsData[6][j] = vec.q3();
            statsData[7][j] = vec.maxValue();
        }

        return Linalg.matrix(statsData);
    }

    /**
     * 返回数值列统计摘要的列名
     * @return 统计指标名称列表
     */
    public List<String> describeColumns() {
        List<String> names = new ArrayList<>();
        for (Column col : columns) {
            if (col.getColumnType().isNumeric()) {
                names.add(col.getName());
            }
        }
        return names;
    }

    /**
     * 返回统计摘要的行名
     * @return 统计指标名称列表
     */
    public List<String> describeIndex() {
        return List.of("count", "mean", "std", "min", "q1", "median", "q3", "max");
    }

    /**
     * 返回 DataFrame 的基本信息（类型、非空数量、缺失值等）
     * @return 信息映射
     */
    public Map<String, Object> info() {
        Map<String, Object> info = new HashMap<>();
        info.put("rows", rowCount);
        info.put("columns", columns.size());
        info.put("missingCount", missingCount());
        info.put("hasMissing", hasMissingValues());

        List<Map<String, Object>> colInfoList = new ArrayList<>();
        for (Column col : columns) {
            Map<String, Object> colInfo = new HashMap<>();
            colInfo.put("name", col.getName());
            colInfo.put("type", col.getColumnType());
            colInfo.put("count", col.getData().size());
            colInfo.put("missing", col.missingCount());
            colInfo.put("valid", col.validCount());
            colInfoList.add(colInfo);
        }
        info.put("columnInfo", colInfoList);
        return info;
    }

    // ==================== ML 预处理 ====================

    /**
     * 标准化（Z-score）：减去均值，除以标准差。
     * 利用 {@link IVector} 的 mean/std/标准向量运算实现。
     * @return 新的DataFrame，数值列已标准化
     */
    public DataFrame standardize() {
        DataFrame result = new DataFrame();
        for (Column col : columns) {
            Column newCol = createColumnCopy(col);
            if (col.getColumnType().isNumeric()) {
                IVector<Double> vec = col.toVec();
                double mean = vec.meanValue();
                double std = vec.stdValue();
                IVector<Double> standardized = vec.subScalar(mean).divideByScalar(std);
                List<Object> newData = new ArrayList<>();
                for (int i = 0; i < col.getData().size(); i++) {
                    Object v = col.getData().get(i);
                    if (v instanceof Double d && Double.isNaN(d)) {
                        newData.add(Double.NaN);
                    } else {
                        newData.add(standardized.get(i));
                    }
                }
                newCol.setData(newData);
            }
            result.addColumn(newCol);
        }
        return result;
    }

    /**
     * 归一化（Min-Max）：缩放到 [0, 1] 范围。
     * 利用 {@link IVector} 的 min/max 实现。
     * @return 新的DataFrame，数值列已归一化
     */
    public DataFrame normalize() {
        DataFrame result = new DataFrame();
        for (Column col : columns) {
            Column newCol = createColumnCopy(col);
            if (col.getColumnType().isNumeric()) {
                IVector<Double> vec = col.toVec();
                double min = vec.minValue();
                double max = vec.maxValue();
                double range = max - min;
                if (range == 0) {
                    // 如果所有值相同，归一化为 0
                    IVector<Double> normalized = vec.subScalar(min);
                    List<Object> newData = new ArrayList<>();
                    for (int i = 0; i < col.getData().size(); i++) {
                        Object v = col.getData().get(i);
                        if (v instanceof Double d && Double.isNaN(d)) {
                            newData.add(Double.NaN);
                        } else {
                            newData.add(normalized.get(i));
                        }
                    }
                    newCol.setData(newData);
                } else {
                    IVector<Double> normalized = vec.subScalar(min).divideByScalar(range);
                    List<Object> newData = new ArrayList<>();
                    for (int i = 0; i < col.getData().size(); i++) {
                        Object v = col.getData().get(i);
                        if (v instanceof Double d && Double.isNaN(d)) {
                            newData.add(Double.NaN);
                        } else {
                            newData.add(normalized.get(i));
                        }
                    }
                    newCol.setData(newData);
                }
            }
            result.addColumn(newCol);
        }
        return result;
    }

    /**
     * 划分训练集和测试集。
     * 使用固定随机种子确保可重现性，依赖 {@link Linalg#rand(int)} 打乱索引。
     * @param trainRatio 训练集比例，{@code 0 < trainRatio <= 1}
     * @return 长度为 2 的数组，[0] 为训练集，[1] 为测试集
     * @throws IllegalArgumentException 如果比例不合法或数据不足
     */
    public DataFrame[] trainTestSplit(double trainRatio) {
        if (trainRatio <= 0 || trainRatio > 1) {
            throw new IllegalArgumentException("训练集比例必须在 (0, 1] 范围内 / Train ratio must be in (0, 1]");
        }
        if (rowCount == 0) {
            throw new IllegalArgumentException("DataFrame 不能为空 / DataFrame cannot be empty");
        }

        int trainSize = (int) Math.round(rowCount * trainRatio);
        if (trainSize < 1) trainSize = 1;
        if (trainSize > rowCount) trainSize = rowCount;

        // 生成打乱的索引
        int[] allIndices = new int[rowCount];
        for (int i = 0; i < rowCount; i++) allIndices[i] = i;

        // Fisher-Yates shuffle
        IVector<Double> rand = Linalg.rand(rowCount);
        for (int i = rowCount - 1; i > 0; i--) {
            int j = (int) (rand.get(i) * (i + 1));
            int tmp = allIndices[i];
            allIndices[i] = allIndices[j];
            allIndices[j] = tmp;
        }

        int[] trainIndices = new int[trainSize];
        int[] testIndices = new int[rowCount - trainSize];
        System.arraycopy(allIndices, 0, trainIndices, 0, trainSize);
        System.arraycopy(allIndices, trainSize, testIndices, 0, rowCount - trainSize);

        return new DataFrame[]{fancyGet(trainIndices), fancyGet(testIndices)};
    }

    /**
     * 打乱数据行顺序（可重复）
     * @param seed 随机种子
     * @return 新的 DataFrame，行顺序已打乱
     */
    public DataFrame shuffle(long seed) {
        int[] shuffled = new int[rowCount];
        for (int i = 0; i < rowCount; i++) shuffled[i] = i;

        IVector<Double> rand = Linalg.rand(rowCount);
        for (int i = rowCount - 1; i > 0; i--) {
            int j = (int) (rand.get(i) * (i + 1));
            int tmp = shuffled[i];
            shuffled[i] = shuffled[j];
            shuffled[j] = tmp;
        }

        return fancyGet(shuffled);
    }

    // ==================== 列操作 ====================

    /**
     * 获取指定位置的列，支持负数索引
     */
    public Column get(int position) {
        if (columns.isEmpty()) {
            throw new IndexOutOfBoundsException("DataFrame为空，无法获取列 / DataFrame is empty, cannot get column");
        }
        int actualPosition = normalizeIndex(position, columns.size(), true);
        return columns.get(actualPosition);
    }

    /**
     * 获取指定位置的列（别名）
     */
    public Column getColumn(int position) {
        return get(position);
    }

    /**
     * 根据列名获取列，O(1) 复杂度（使用 HashMap 索引）
     * @param columnName 列名
     * @return 对应的列，如果未找到返回null
     */
    public Column getColumnByName(String columnName) {
        if (columnName == null) return null;
        rebuildNameIndexIfNeeded();
        Integer index = nameIndex.get(columnName);
        return index != null ? columns.get(index) : null;
    }

    /**
     * 添加列
     */
    public void addColumn(Column column) {
        if (column == null) {
            throw new IllegalArgumentException("列不能为null / Column cannot be null");
        }
        if (rowCount > 0 && column.getData().size() != rowCount) {
            throw new IllegalArgumentException("列数据长度 " + column.getData().size() + " 与现有行数 " + rowCount + " 不匹配");
        }
        columns.add(column);
        if (rowCount == 0 && !column.getData().isEmpty()) {
            rowCount = column.getData().size();
        }
        invalidateNameIndex();
    }

    /**
     * 删除指定位置的列
     */
    public Column removeColumn(int position) {
        int actualPosition = normalizeIndex(position, columns.size(), true);
        Column removed = columns.remove(actualPosition);
        invalidateNameIndex();
        return removed;
    }

    // ==================== 属性访问 ====================

    public int getColumnCount() { return columns.size(); }
    public int cols() { return getColumnCount(); }
    public int getRowCount() { return rowCount; }
    public int rows() { return getRowCount(); }
    public int[] shape() { return new int[]{rowCount, columns.size()}; }
    public boolean isEmpty() { return columns.isEmpty() || rowCount == 0; }

    public List<String> getColumnNames() {
        List<String> names = new ArrayList<>();
        for (Column column : columns) {
            names.add(column.getName());
        }
        return names;
    }

    public List<ColumnType> getColumnTypes() {
        List<ColumnType> types = new ArrayList<>();
        for (Column column : columns) {
            types.add(column.getColumnType());
        }
        return types;
    }

    public List<Column> getColumns() {
        return new ArrayList<>(columns);
    }

    public void setColumns(List<Column> columns) {
        if (columns == null) {
            throw new IllegalArgumentException("列列表不能为null");
        }
        this.columns = new ArrayList<>(columns);
        this.rowCount = columns.isEmpty() ? 0 : columns.get(0).getData().size();
        for (Column column : columns) {
            if (column.getData().size() != rowCount) {
                throw new IllegalArgumentException("列数据长度不一致 / Column data lengths are inconsistent");
            }
        }
        invalidateNameIndex();
    }

    public void clear() {
        columns.clear();
        rowCount = 0;
        invalidateNameIndex();
    }

    /**
     * 深拷贝
     */
    public DataFrame copy() {
        DataFrame copy = new DataFrame();
        for (Column column : columns) {
            copy.addColumn(createColumnCopy(column));
        }
        return copy;
    }

    // ==================== 内部工具方法 ====================

    /**
     * 创建列的深拷贝
     */
    private Column createColumnCopy(Column original) {
        Column newColumn = new Column();
        newColumn.setName(original.getName());
        newColumn.setColumnType(original.getColumnType());
        newColumn.setData(new ArrayList<>(original.getData()));
        return newColumn;
    }

    /**
     * 规范化索引，支持负数索引
     * @param index 原始索引
     * @param size 有效范围大小
     * @param isStart 是否为起始索引（包含/不包含语义不同）
     */
    private int normalizeIndex(int index, int size, boolean isStart) {
        if (size == 0) return 0;
        int actual;
        if (index < 0) {
            actual = size + index;
        } else {
            actual = index;
        }
        if (isStart) {
            return Math.max(0, Math.min(actual, size));
        } else {
            return Math.max(0, Math.min(actual, size));
        }
    }

    private void invalidateNameIndex() {
        nameIndexDirty = true;
    }

    private void rebuildNameIndexIfNeeded() {
        if (nameIndexDirty) {
            nameIndex.clear();
            for (int i = 0; i < columns.size(); i++) {
                nameIndex.put(columns.get(i).getName(), i);
            }
            nameIndexDirty = false;
        }
    }

    // ==================== toString ====================

    @Override
    public String toString() {
        if (isEmpty()) {
            return "Empty DataFrame";
        }

        StringBuilder sb = new StringBuilder();
        sb.append("DataFrame [").append(rowCount).append(" rows x ").append(columns.size()).append(" columns]\n");
        sb.append("Columns: ");
        for (int i = 0; i < columns.size(); i++) {
            if (i > 0) sb.append(", ");
            Column c = columns.get(i);
            sb.append(c.getName()).append("(").append(c.getColumnType()).append(")");
        }
        sb.append("\n");

        int maxRows = Math.min(5, rowCount);
        for (int i = 0; i < maxRows; i++) {
            sb.append("Row ").append(i).append(": ");
            for (int j = 0; j < columns.size(); j++) {
                if (j > 0) sb.append(", ");
                Object v = columns.get(j).getData().get(i);
                sb.append(v instanceof Double d && Double.isNaN(d) ? "NaN" : v);
            }
            sb.append("\n");
        }

        if (rowCount > 5) {
            sb.append("... (").append(rowCount - 5).append(" more rows)");
        }

        return sb.toString();
    }
}
