package com.reremouse.lab.math.data;

import com.reremouse.lab.math.IMatrix;
import com.reremouse.lab.math.RereMatrix;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import com.reremouse.lab.math.SliceExpressionParser;
import org.apache.commons.csv.CSVPrinter;

/**
 * 数据框类，用于处理结构化数据，支持从CSV文件读取数据并与IMatrix进行转换
 * DataFrame class for handling structured data, supports reading from CSV files and conversion with IMatrix
 * 
 * <p>本类提供了类似pandas DataFrame的功能，包括：</p>
 * <p>This class provides functionality similar to pandas DataFrame, including:</p>
 * <ul>
 * <li>从CSV文件读取数据 / Reading data from CSV files</li>
 * <li>列切片和行切片操作 / Column and row slicing operations</li>
 * <li>数据类型转换（String/Float） / Data type conversion (String/Float)</li>
 * <li>与IMatrix的互转换 / Mutual conversion with IMatrix</li>
 * <li>数据访问和操作 / Data access and manipulation</li>
 * </ul>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class DataFrame implements Serializable {

    private static final long serialVersionUID = 1L;
    
    /** 列列表，存储所有列数据 / List of columns storing all column data */
    private List<Column> columns = new ArrayList<>();
    
    /** 行数，用于数据验证 / Number of rows for data validation */
    private int rowCount = 0;

    /**
     * 默认构造函数 / Default constructor
     * <p>创建一个空的DataFrame / Creates an empty DataFrame</p>
     */
    public DataFrame() {
        this.columns = new ArrayList<>();
        this.rowCount = 0;
    }

    /**
     * 带列列表的构造函数 / Constructor with column list
     * <p>使用指定的列列表创建DataFrame / Creates DataFrame with specified column list</p>
     * 
     * @param columns 列列表 / List of columns
     */
    public DataFrame(List<Column> columns) {
        this.columns = new ArrayList<>(columns);
        this.rowCount = columns.isEmpty() ? 0 : columns.get(0).getData().size();
    }

    /**
     * 从CSV文件读取数据创建DataFrame / Create DataFrame by reading data from CSV file
     * <p>读取CSV文件并创建DataFrame，数值型数据自动转换为Float类型列</p>
     * <p>Reads CSV file and creates DataFrame, automatically converts numeric data to Float type columns</p>
     * 
     * @param filePath 文件路径 / File path
     * @param separator 分隔符 / Separator
     * @param ifHasHead 是否有表头 / Whether has header
     * @return 创建的DataFrame对象 / Created DataFrame object
     * @throws IOException 如果文件读取失败 / if file reading fails
     * @throws IllegalArgumentException 如果参数无效 / if parameters are invalid
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
             CSVParser parser = new CSVParser(reader, CSVFormat.DEFAULT.builder().setDelimiter(separator.charAt(0)).build())) {
            
            List<CSVRecord> records = parser.getRecords();
            if (records.isEmpty()) {
                return df;
            }

            // 确定列数 / Determine number of columns
            int numColumns = records.get(0).size();
            String[] columnNames = new String[numColumns];
            
            // 处理表头 / Handle header
            if (ifHasHead && !records.isEmpty()) {
                CSVRecord headerRecord = records.get(0);
                for (int i = 0; i < numColumns; i++) {
                    columnNames[i] = headerRecord.get(i);
                }
                records = records.subList(1, records.size()); // 移除表头行 / Remove header row
            } else {
                // 如果没有表头，生成默认列名 / If no header, generate default column names
                for (int i = 0; i < numColumns; i++) {
                    columnNames[i] = "col_" + i;
                }
            }

            // 初始化列 / Initialize columns
            for (int i = 0; i < numColumns; i++) {
                Column column = new Column();
                column.setName(columnNames[i]);
                column.setColumnType(ColumnType.String); // 默认为String类型 / Default to String type
                column.setData(new ArrayList<>());
                df.columns.add(column);
            }

            // 解析数据并确定列类型 / Parse data and determine column types
            for (CSVRecord record : records) {
                for (int i = 0; i < numColumns; i++) {
                    String value = record.get(i).trim();
                    df.columns.get(i).getData().add(value);
                }
            }

            // 尝试将数值型数据转换为Float类型 / Try to convert numeric data to Float type
            for (Column column : df.columns) {
                boolean isNumeric = true;
                List<Object> floatData = new ArrayList<>();
                
                for (Object value : column.getData()) {
                    try {
                        float floatValue = Float.parseFloat(value.toString());
                        floatData.add(floatValue);
                    } catch (NumberFormatException e) {
                        isNumeric = false;
                        break;
                    }
                }
                
                if (isNumeric && !floatData.isEmpty()) {
                    column.setColumnType(ColumnType.Float);
                    column.setData(floatData);
                }
            }

            df.rowCount = records.size();
        }
        
        return df;
    }

    public void toCsv(String filePath) throws IOException {
        if (filePath == null || filePath.trim().isEmpty()) {
            throw new IllegalArgumentException("文件路径不能为空 / File path cannot be empty");
        }
        
        try (FileWriter writer = new FileWriter(filePath);
             CSVPrinter printer = new CSVPrinter(writer, CSVFormat.DEFAULT)) {
            
            // 打印列名作为表头 / Print column names as header
            printer.printRecord(columns.stream().map(Column::getName).toArray(String[]::new));
            
            // 按行打印数据 / Print data row by row
            if (rowCount > 0) {
                for (int i = 0; i < rowCount; i++) {
                    Object[] rowData = new Object[columns.size()];
                    for (int j = 0; j < columns.size(); j++) {
                        List<Object> columnData = columns.get(j).getData();
                        if (i < columnData.size()) {
                            rowData[j] = columnData.get(i);
                        } else {
                            rowData[j] = ""; // 空值处理 / Handle empty values
                        }
                    }
                    printer.printRecord(rowData);
                }
            }
        }
    }

    /**
     * 列切片操作（带步长） / Column slicing operation (with step)
     * <p>根据起始位置、结束位置和步长对列进行切片</p>
     * <p>Slices columns based on start position, end position and step</p>
     * 
     * @param start 起始位置（包含） / Start position (inclusive)
     * @param end 结束位置（不包含），支持负数索引 / End position (exclusive), supports negative indices
     * @param step 步长 / Step size
     * @return 切片后的DataFrame / Sliced DataFrame
     * @throws IllegalArgumentException 如果参数无效 / if parameters are invalid
     */
    public DataFrame sliceColumn(int start, int end, int step) {
        if (step <= 0) {
            throw new IllegalArgumentException("步长必须大于0 / Step must be greater than 0");
        }
        if (columns.isEmpty()) {
            return new DataFrame();
        }
        
        // 处理负数索引 / Handle negative indices
        int actualStart = start;
        int actualEnd = end;
        
        if (start < 0) {
            actualStart = columns.size() + start;
        }
        if (end < 0) {
            actualEnd = columns.size() + end;
        }
        
        // 确保索引在有效范围内 / Ensure indices are within valid range
        actualStart = Math.max(0, Math.min(actualStart, columns.size()));
        actualEnd = Math.max(0, Math.min(actualEnd, columns.size()));
        
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
     * 列切片操作（步长为1） / Column slicing operation (step = 1)
     * <p>根据起始位置和结束位置对列进行切片，步长默认为1</p>
     * <p>Slices columns based on start and end positions with default step of 1</p>
     * 
     * @param start 起始位置（包含） / Start position (inclusive)
     * @param end 结束位置（不包含） / End position (exclusive)
     * @return 切片后的DataFrame / Sliced DataFrame
     */
    public DataFrame sliceColumn(int start, int end) {
        return sliceColumn(start, end, 1);
    }

    /**
     * 列切片操作（从指定位置到末尾） / Column slicing operation (from specified position to end)
     * <p>从指定位置开始到最后一列进行切片</p>
     * <p>Slices from specified position to the last column</p>
     * 
     * @param start 起始位置（包含） / Start position (inclusive)
     * @return 切片后的DataFrame / Sliced DataFrame
     */
    public DataFrame sliceColumn(int start) {
        return sliceColumn(start, columns.size(), 1);
    }
    
    /**
     * 通用切片操作 / General slicing operation
     * <p>支持类似NumPy的切片语法，如"1:5:2"表示从1到5，步长为2</p>
     * <p>Supports NumPy-like slicing syntax, e.g., "1:5:2" means from 1 to 5 with step 2</p>
     * 
     * @param rowExp 行切片表达式，如"1:5"或"1:5:2" / Row slicing expression, e.g., "1:5" or "1:5:2"
     * @param colExp 列切片表达式，如"1:5"或"1:5:2" / Column slicing expression, e.g., "1:5" or "1:5:2"
     * @return 切片后的DataFrame / Sliced DataFrame
     * @throws IllegalArgumentException 如果表达式格式无效 / if expression format is invalid
     */
    public DataFrame slice(String rowExp, String colExp) {
        DataFrame result = new DataFrame();
        
        // 解析列切片 / Parse column slicing
        if (colExp != null && !colExp.trim().isEmpty()) {
            int[] colIndices = SliceExpressionParser.generateIndices(
                SliceExpressionParser.parse(colExp, columns.size()));
            for (int colIndex : colIndices) {
                if (colIndex >= 0 && colIndex < columns.size()) {
                    // 创建列的副本而不是直接引用 / Create a copy of the column instead of direct reference
                    Column originalColumn = columns.get(colIndex);
                    Column newColumn = new Column();
                    newColumn.setName(originalColumn.getName());
                    newColumn.setColumnType(originalColumn.getColumnType());
                    newColumn.setData(new ArrayList<>(originalColumn.getData()));
                    result.addColumn(newColumn);
                }
            }
        } else {
            // 如果没有指定列切片，复制所有列 / If no column slicing specified, copy all columns
            for (Column column : columns) {
                // 创建列的副本而不是直接引用 / Create a copy of the column instead of direct reference
                Column newColumn = new Column();
                newColumn.setName(column.getName());
                newColumn.setColumnType(column.getColumnType());
                newColumn.setData(new ArrayList<>(column.getData()));
                result.addColumn(newColumn);
            }
        }
        
        // 解析行切片 / Parse row slicing
        if (rowExp != null && !rowExp.trim().isEmpty()) {
            int[] rowIndices = SliceExpressionParser.generateIndices(
                SliceExpressionParser.parse(rowExp, rowCount));
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
     * 将Float类型的列转换为IMatrix / Convert Float type columns to IMatrix
     * <p>提取所有Float类型的列并合并为一个矩阵</p>
     * <p>Extracts all Float type columns and combines them into a matrix</p>
     * 
     * @return 包含所有Float列数据的矩阵 / Matrix containing all Float column data
     * @throws IllegalStateException 如果没有Float类型的列 / if no Float type columns exist
     */
    public IMatrix toMatrix() {
        List<Column> floatColumns = new ArrayList<>();
        for (Column column : columns) {
            if (column.getColumnType() == ColumnType.Float) {
                floatColumns.add(column);
            }
        }
        
        if (floatColumns.isEmpty()) {
            throw new IllegalStateException("没有Float类型的列可以转换为矩阵 / No Float type columns available for matrix conversion");
        }
        
        // 检查所有Float列的数据长度是否一致 / Check if all Float columns have consistent data length
        int actualRowCount = floatColumns.get(0).getData().size();
        if (actualRowCount == 0) {
            return new RereMatrix(new float[0][0]);
        }
        
        for (Column column : floatColumns) {
            if (column.getData().size() != actualRowCount) {
                throw new IllegalStateException("Float列数据长度不一致: " + column.getName() + " 有 " + column.getData().size() + " 个元素，期望 " + actualRowCount + " 个 / Float column data lengths are inconsistent: " + column.getName() + " has " + column.getData().size() + " elements, expected " + actualRowCount);
            }
        }
        
        // 创建矩阵数据 / Create matrix data
        float[][] matrixData = new float[actualRowCount][floatColumns.size()];
        for (int i = 0; i < actualRowCount; i++) {
            for (int j = 0; j < floatColumns.size(); j++) {
                Object value = floatColumns.get(j).getData().get(i);
                matrixData[i][j] = (Float) value;
            }
        }
        
        return new RereMatrix(matrixData);
    }

    /**
     * 获取指定位置的列 / Get column at specified position
     * <p>支持负数索引，-1表示最后一列，-2表示倒数第二列，以此类推</p>
     * <p>Supports negative indices, -1 means last column, -2 means second to last column, and so on</p>
     * 
     * @param position 列位置，支持负数索引 / Column position, supports negative indices
     * @return 指定位置的列 / Column at specified position
     * @throws IndexOutOfBoundsException 如果位置超出范围 / if position is out of bounds
     */
    public Column get(int position) {
        if (columns.isEmpty()) {
            throw new IndexOutOfBoundsException("DataFrame为空，无法获取列 / DataFrame is empty, cannot get column");
        }
        
        int actualPosition = position;
        if (position < 0) {
            actualPosition = columns.size() + position;
        }
        
        if (actualPosition < 0 || actualPosition >= columns.size()) {
            throw new IndexOutOfBoundsException("列位置 " + position + " 超出范围 [0, " + (columns.size() - 1) + "] / Column position " + position + " out of bounds [0, " + (columns.size() - 1) + "]");
        }
        
        return columns.get(actualPosition);
    }

    /**
     * 添加列 / Add column
     * <p>向DataFrame添加新列</p>
     * <p>Adds a new column to the DataFrame</p>
     * 
     * @param column 要添加的列 / Column to add
     * @throws IllegalArgumentException 如果列数据长度与现有行数不匹配 / if column data length doesn't match existing row count
     */
    public void addColumn(Column column) {
        if (column == null) {
            throw new IllegalArgumentException("列不能为null / Column cannot be null");
        }
        
        if (rowCount > 0 && column.getData().size() != rowCount) {
            throw new IllegalArgumentException("列数据长度 " + column.getData().size() + " 与现有行数 " + rowCount + " 不匹配 / Column data length " + column.getData().size() + " doesn't match existing row count " + rowCount);
        }
        
        columns.add(column);
        if (rowCount == 0 && !column.getData().isEmpty()) {
            // 如果这是第一列且不为空，设置行数 / If this is the first column and not empty, set row count
            rowCount = column.getData().size();
        }
    }

    /**
     * 删除列 / Remove column
     * <p>根据位置删除列</p>
     * <p>Removes column by position</p>
     * 
     * @param position 列位置 / Column position
     * @return 被删除的列 / Removed column
     * @throws IndexOutOfBoundsException 如果位置超出范围 / if position is out of bounds
     */
    public Column removeColumn(int position) {
        int actualPosition = position;
        if (position < 0) {
            actualPosition = columns.size() + position;
        }
        
        if (actualPosition < 0 || actualPosition >= columns.size()) {
            throw new IndexOutOfBoundsException("列位置 " + position + " 超出范围 / Column position " + position + " out of bounds");
        }
        
        return columns.remove(actualPosition);
    }

    /**
     * 获取列数 / Get column count
     * <p>返回DataFrame的列数</p>
     * <p>Returns the number of columns in the DataFrame</p>
     * 
     * @return 列数 / Number of columns
     */
    public int getColumnCount() {
        return columns.size();
    }

    /**
     * 获取行数 / Get row count
     * <p>返回DataFrame的行数</p>
     * <p>Returns the number of rows in the DataFrame</p>
     * 
     * @return 行数 / Number of rows
     */
    public int getRowCount() {
        return rowCount;
    }

    /**
     * 获取DataFrame形状 / Get DataFrame shape
     * <p>返回[行数, 列数]的数组</p>
     * <p>Returns array of [row count, column count]</p>
     * 
     * @return 形状数组 / Shape array
     */
    public int[] shape() {
        return new int[]{rowCount, columns.size()};
    }

    /**
     * 检查DataFrame是否为空 / Check if DataFrame is empty
     * <p>判断DataFrame是否没有数据</p>
     * <p>Determines if DataFrame has no data</p>
     * 
     * @return 是否为空 / Whether empty
     */
    public boolean isEmpty() {
        return columns.isEmpty() || rowCount == 0;
    }

    /**
     * 获取列名列表 / Get column names
     * <p>返回所有列的名称列表</p>
     * <p>Returns list of all column names</p>
     * 
     * @return 列名列表 / List of column names
     */
    public List<String> getColumnNames() {
        List<String> names = new ArrayList<>();
        for (Column column : columns) {
            names.add(column.getName());
        }
        return names;
    }

    /**
     * 根据列名获取列 / Get column by name
     * <p>根据列名查找并返回对应的列</p>
     * <p>Finds and returns column by name</p>
     * 
     * @param columnName 列名 / Column name
     * @return 对应的列，如果未找到返回null / Corresponding column, null if not found
     */
    public Column getColumnByName(String columnName) {
        if (columnName == null) {
            return null;
        }
        
        for (Column column : columns) {
            if (columnName.equals(column.getName())) {
                return column;
            }
        }
        return null;
    }

    /**
     * 获取列类型列表 / Get column types
     * <p>返回所有列的类型列表</p>
     * <p>Returns list of all column types</p>
     * 
     * @return 列类型列表 / List of column types
     */
    public List<ColumnType> getColumnTypes() {
        List<ColumnType> types = new ArrayList<>();
        for (Column column : columns) {
            types.add(column.getColumnType());
        }
        return types;
    }

    /**
     * 清空DataFrame / Clear DataFrame
     * <p>清空所有列和行数据</p>
     * <p>Clears all columns and row data</p>
     */
    public void clear() {
        columns.clear();
        rowCount = 0;
    }

    /**
     * 复制DataFrame / Copy DataFrame
     * <p>创建当前DataFrame的深拷贝</p>
     * <p>Creates a deep copy of current DataFrame</p>
     * 
     * @return DataFrame的副本 / Copy of DataFrame
     */
    public DataFrame copy() {
        DataFrame copy = new DataFrame();
        for (Column column : columns) {
            Column newColumn = new Column();
            newColumn.setName(column.getName());
            newColumn.setColumnType(column.getColumnType());
            newColumn.setData(new ArrayList<>(column.getData()));
            copy.addColumn(newColumn);
        }
        return copy;
    }

    /**
     * 获取列列表 / Get columns list
     * <p>返回列列表的副本</p>
     * <p>Returns a copy of the columns list</p>
     * 
     * @return 列列表副本 / Copy of columns list
     */
    public List<Column> getColumns() {
        return new ArrayList<>(columns);
    }

    /**
     * 设置列列表 / Set columns list
     * <p>设置列列表并更新行数</p>
     * <p>Sets columns list and updates row count</p>
     * 
     * @param columns 新的列列表 / New columns list
     * @throws IllegalArgumentException 如果列数据长度不一致 / if column data lengths are inconsistent
     */
    public void setColumns(List<Column> columns) {
        if (columns == null) {
            throw new IllegalArgumentException("列列表不能为null / Columns list cannot be null");
        }
        
        this.columns = new ArrayList<>(columns);
        this.rowCount = columns.isEmpty() ? 0 : columns.get(0).getData().size();
        
        // 验证所有列的数据长度一致 / Validate all columns have consistent data length
        for (Column column : columns) {
            if (column.getData().size() != rowCount) {
                throw new IllegalArgumentException("列数据长度不一致 / Column data lengths are inconsistent");
            }
        }
    }

    @Override
    public String toString() {
        if (isEmpty()) {
            return "Empty DataFrame";
        }
        
        StringBuilder sb = new StringBuilder();
        sb.append("DataFrame [").append(rowCount).append(" rows x ").append(columns.size()).append(" columns]\n");
        
        // 添加列名 / Add column names
        sb.append("Columns: ");
        for (int i = 0; i < columns.size(); i++) {
            if (i > 0) sb.append(", ");
            sb.append(columns.get(i).getName()).append("(").append(columns.get(i).getColumnType()).append(")");
        }
        sb.append("\n");
        
        // 添加前几行数据 / Add first few rows of data
        int maxRows = Math.min(5, rowCount);
        for (int i = 0; i < maxRows; i++) {
            sb.append("Row ").append(i).append(": ");
            for (int j = 0; j < columns.size(); j++) {
                if (j > 0) sb.append(", ");
                sb.append(columns.get(j).getData().get(i));
            }
            sb.append("\n");
        }
        
        if (rowCount > 5) {
            sb.append("... (").append(rowCount - 5).append(" more rows)");
        }
        
        return sb.toString();
    }
}
