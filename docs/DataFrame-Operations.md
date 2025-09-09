# DataFrame 数据框操作 (DataFrame Operations)

## 概述 / Overview

`DataFrame` 类提供了类似pandas DataFrame的功能，用于处理结构化数据。它支持从CSV文件读取数据、列切片和行切片操作、数据类型转换、与IMatrix的互转换等功能。

The `DataFrame` class provides functionality similar to pandas DataFrame for handling structured data. It supports reading data from CSV files, column and row slicing operations, data type conversion, mutual conversion with IMatrix, and more.

## 核心类 / Core Classes

### DataFrame 类 / DataFrame Class

`DataFrame` 是数据框操作的核心类，提供了完整的数据处理功能。

`DataFrame` is the core class for data frame operations, providing comprehensive data processing functionality.

### Column 类 / Column Class

`Column` 类表示数据框中的一列，包含列名、数据类型和数据。

`Column` class represents a column in the data frame, containing column name, data type, and data.

### ColumnType 枚举 / ColumnType Enum

`ColumnType` 枚举定义了支持的数据类型：String 和 Float。

`ColumnType` enum defines supported data types: String and Float.

## 主要功能 / Main Features

### 1. DataFrame 创建 / DataFrame Creation

#### 构造函数 / Constructors

```java
// 默认构造函数 / Default constructor
DataFrame df1 = new DataFrame();

// 带列列表的构造函数 / Constructor with column list
List<Column> columns = new ArrayList<>();
columns.add(new Column("name", ColumnType.String, Arrays.asList("Alice", "Bob")));
columns.add(new Column("age", ColumnType.Float, Arrays.asList(25.0f, 30.0f)));
DataFrame df2 = new DataFrame(columns);
```

#### 从CSV文件读取 / Reading from CSV Files

```java
// 从CSV文件读取数据 / Read data from CSV file
DataFrame df = DataFrame.readCsv("data.csv", ",", true);  // 文件路径, 分隔符, 是否有表头

// 示例CSV文件内容 / Example CSV file content:
// name,age,salary
// Alice,25,50000
// Bob,30,60000
// Charlie,35,70000
```

### 2. 数据访问和操作 / Data Access and Manipulation

#### 基本属性 / Basic Properties

```java
// 获取形状 / Get shape
int[] shape = df.shape();                    // [行数, 列数]
int rowCount = df.getRowCount();             // 行数
int columnCount = df.getColumnCount();       // 列数

// 检查是否为空 / Check if empty
boolean isEmpty = df.isEmpty();

// 获取列名 / Get column names
List<String> columnNames = df.getColumnNames();

// 获取列类型 / Get column types
List<ColumnType> columnTypes = df.getColumnTypes();
```

#### 列操作 / Column Operations

```java
// 获取列 / Get column
Column column = df.get(0);                   // 按索引获取列
Column columnByName = df.getColumnByName("age"); // 按名称获取列

// 添加列 / Add column
Column newColumn = new Column();
newColumn.setName("department");
newColumn.setColumnType(ColumnType.String);
newColumn.setData(Arrays.asList("IT", "HR", "Finance"));
df.addColumn(newColumn);

// 删除列 / Remove column
Column removedColumn = df.removeColumn(0);   // 按索引删除列
Column removedColumnByName = df.removeColumn(-1); // 按负数索引删除列（最后一列）
```

### 3. 切片操作 / Slicing Operations

#### 列切片 / Column Slicing

```java
// 基本列切片 / Basic column slicing
DataFrame colSlice1 = df.sliceColumn(1, 3);     // 从索引1到3（不包含）
DataFrame colSlice2 = df.sliceColumn(1);        // 从索引1到末尾
DataFrame colSlice3 = df.sliceColumn(0, 4, 2);  // 从索引0到4，步长为2

// 支持负数索引 / Support negative indices
DataFrame colSlice4 = df.sliceColumn(-2, -1);   // 倒数第二列到倒数第一列（不包含）
```

#### 通用切片 / General Slicing

```java
// 通用切片操作 / General slicing operation
DataFrame sliced = df.slice("1:3", "0:2");      // 行切片"1:3"，列切片"0:2"
DataFrame sliced2 = df.slice("0:5:2", "1:4");  // 行切片"0:5:2"（步长为2），列切片"1:4"

// 只进行行切片 / Row slicing only
DataFrame rowSliced = df.slice("1:3", null);

// 只进行列切片 / Column slicing only
DataFrame colSliced = df.slice(null, "0:2");
```

### 4. 数据类型转换 / Data Type Conversion

#### 与IMatrix转换 / Conversion with IMatrix

```java
// 将Float类型列转换为IMatrix / Convert Float type columns to IMatrix
IMatrix matrix = df.toMatrix();

// 注意：只有Float类型的列会被转换 / Note: Only Float type columns will be converted
// String类型的列会被忽略 / String type columns will be ignored
```

#### 列数据类型转换 / Column Data Type Conversion

```java
// 获取列并转换数据类型 / Get column and convert data type
Column ageColumn = df.getColumnByName("age");

// 转换为向量 / Convert to vector (仅限Float类型)
IVector ageVector = ageColumn.toVec();

// 转换为字符串列表 / Convert to string list
List<String> ageStrings = ageColumn.toStringList();
```

### 5. 数据输出 / Data Output

#### 保存为CSV / Save to CSV

```java
// 保存DataFrame为CSV文件 / Save DataFrame to CSV file
df.toCsv("output.csv");
```

### 6. 数据复制和清理 / Data Copying and Cleaning

```java
// 复制DataFrame / Copy DataFrame
DataFrame dfCopy = df.copy();

// 清空DataFrame / Clear DataFrame
df.clear();

// 设置列列表 / Set columns list
List<Column> newColumns = Arrays.asList(/* ... */);
df.setColumns(newColumns);
```

## 使用示例 / Usage Examples

### 基础示例 / Basic Examples

#### 创建和操作DataFrame / Creating and Manipulating DataFrame

```java
import com.reremouse.lab.data.DataFrame;
import com.reremouse.lab.data.Column;
import com.reremouse.lab.data.ColumnType;
import com.reremouse.lab.math.IMatrix;
import java.util.Arrays;
import java.util.List;

public class DataFrameBasicExample {
    public static void main(String[] args) {
        // 创建DataFrame / Create DataFrame
        DataFrame df = new DataFrame();
        
        // 添加列 / Add columns
        Column nameColumn = new Column();
        nameColumn.setName("name");
        nameColumn.setColumnType(ColumnType.String);
        nameColumn.setData(Arrays.asList("Alice", "Bob", "Charlie"));
        df.addColumn(nameColumn);
        
        Column ageColumn = new Column();
        ageColumn.setName("age");
        ageColumn.setColumnType(ColumnType.Float);
        ageColumn.setData(Arrays.asList(25.0f, 30.0f, 35.0f));
        df.addColumn(ageColumn);
        
        Column salaryColumn = new Column();
        salaryColumn.setName("salary");
        salaryColumn.setColumnType(ColumnType.Float);
        salaryColumn.setData(Arrays.asList(50000.0f, 60000.0f, 70000.0f));
        df.addColumn(salaryColumn);
        
        // 显示DataFrame信息 / Display DataFrame information
        System.out.println("DataFrame形状: " + Arrays.toString(df.shape()));
        System.out.println("列名: " + df.getColumnNames());
        System.out.println("列类型: " + df.getColumnTypes());
        System.out.println("DataFrame内容:\n" + df);
    }
}
```

#### 从CSV文件读取 / Reading from CSV File

```java
import com.reremouse.lab.data.DataFrame;
import java.io.IOException;

public class CSVReadingExample {
    public static void main(String[] args) {
        try {
            // 从CSV文件读取 / Read from CSV file
            DataFrame df = DataFrame.readCsv("data.csv", ",", true);
            
            System.out.println("从CSV读取的DataFrame:");
            System.out.println("形状: " + Arrays.toString(df.shape()));
            System.out.println("列名: " + df.getColumnNames());
            System.out.println("内容:\n" + df);
            
        } catch (IOException e) {
            System.err.println("读取CSV文件失败: " + e.getMessage());
        }
    }
}
```

#### 切片操作 / Slicing Operations

```java
public class SlicingExample {
    public static void main(String[] args) {
        // 创建示例DataFrame / Create example DataFrame
        DataFrame df = createSampleDataFrame();
        
        // 列切片 / Column slicing
        DataFrame colSlice1 = df.sliceColumn(1, 3);     // 列1到2
        DataFrame colSlice2 = df.sliceColumn(0, 4, 2);  // 列0,2（步长为2）
        DataFrame colSlice3 = df.sliceColumn(-2);       // 最后两列
        
        // 通用切片 / General slicing
        DataFrame sliced1 = df.slice("1:3", "0:2");     // 行1-2，列0-1
        DataFrame sliced2 = df.slice("0:5:2", "1:4");  // 行0,2,4，列1-3
        
        System.out.println("列切片结果1:\n" + colSlice1);
        System.out.println("通用切片结果1:\n" + sliced1);
    }
    
    private static DataFrame createSampleDataFrame() {
        // 创建示例数据的实现 / Implementation for creating sample data
        // ... (省略实现细节) / ... (omitted implementation details)
    }
}
```

#### 数据类型转换 / Data Type Conversion

```java
public class DataConversionExample {
    public static void main(String[] args) {
        DataFrame df = createSampleDataFrame();
        
        // 转换为矩阵 / Convert to matrix
        try {
            IMatrix matrix = df.toMatrix();
            System.out.println("转换后的矩阵:\n" + matrix);
        } catch (IllegalStateException e) {
            System.out.println("转换失败: " + e.getMessage());
        }
        
        // 列数据转换 / Column data conversion
        Column ageColumn = df.getColumnByName("age");
        if (ageColumn != null) {
            // 转换为向量 / Convert to vector
            IVector ageVector = ageColumn.toVec();
            System.out.println("年龄向量: " + ageVector);
            
            // 转换为字符串列表 / Convert to string list
            List<String> ageStrings = ageColumn.toStringList();
            System.out.println("年龄字符串列表: " + ageStrings);
        }
    }
}
```

### 高级示例 / Advanced Examples

#### 数据预处理管道 / Data Preprocessing Pipeline

```java
public class DataPreprocessingPipeline {
    public static void main(String[] args) {
        try {
            // 1. 从CSV读取数据 / Read data from CSV
            DataFrame df = DataFrame.readCsv("raw_data.csv", ",", true);
            System.out.println("原始数据形状: " + Arrays.toString(df.shape()));
            
            // 2. 数据清洗 / Data cleaning
            // 删除空行或异常值 / Remove empty rows or outliers
            DataFrame cleaned = df.slice("0:100", null); // 只取前100行
            
            // 3. 特征选择 / Feature selection
            // 只选择数值型列 / Select only numeric columns
            DataFrame numericOnly = cleaned.slice(null, "1:4"); // 假设列1-3是数值型
            
            // 4. 转换为矩阵进行进一步处理 / Convert to matrix for further processing
            IMatrix matrix = numericOnly.toMatrix();
            
            // 5. 数据标准化 / Data standardization
            IMatrix standardized = matrix.center(); // 中心化
            
            // 6. 保存处理后的数据 / Save processed data
            DataFrame processedDf = new DataFrame();
            // 将标准化后的矩阵转换回DataFrame
            // Convert standardized matrix back to DataFrame
            processedDf.toCsv("processed_data.csv");
            
        } catch (IOException e) {
            System.err.println("数据处理失败: " + e.getMessage());
        }
    }
}
```

#### 数据分析和统计 / Data Analysis and Statistics

```java
public class DataAnalysisExample {
    public static void main(String[] args) {
        DataFrame df = createSampleDataFrame();
        
        // 基本统计信息 / Basic statistics
        System.out.println("数据形状: " + Arrays.toString(df.shape()));
        System.out.println("列名: " + df.getColumnNames());
        System.out.println("列类型: " + df.getColumnTypes());
        
        // 数值型列的统计 / Statistics for numeric columns
        for (int i = 0; i < df.getColumnCount(); i++) {
            Column col = df.get(i);
            if (col.getColumnType() == ColumnType.Float) {
                System.out.println("\n列 " + col.getName() + " 的统计信息:");
                
                // 转换为向量进行统计 / Convert to vector for statistics
                IVector vector = col.toVec();
                System.out.println("  均值: " + vector.mean());
                System.out.println("  标准差: " + vector.std());
                System.out.println("  最小值: " + vector.min());
                System.out.println("  最大值: " + vector.max());
                System.out.println("  总和: " + vector.sum());
            }
        }
        
        // 数据切片分析 / Data slicing analysis
        DataFrame youngPeople = df.slice("0:2", null); // 前两行
        System.out.println("\n年轻人群数据:\n" + youngPeople);
    }
}
```

#### 机器学习数据准备 / Machine Learning Data Preparation

```java
public class MLDataPreparationExample {
    public static void main(String[] args) {
        try {
            // 读取原始数据 / Read raw data
            DataFrame rawData = DataFrame.readCsv("ml_data.csv", ",", true);
            
            // 分离特征和标签 / Separate features and labels
            DataFrame features = rawData.slice(null, "0:-1"); // 除最后一列外的所有列
            DataFrame labels = rawData.slice(null, "-1:");    // 最后一列
            
            // 转换特征为矩阵 / Convert features to matrix
            IMatrix featureMatrix = features.toMatrix();
            System.out.println("特征矩阵形状: " + Arrays.toString(featureMatrix.shape()));
            
            // 数据预处理 / Data preprocessing
            IMatrix centeredFeatures = featureMatrix.center();        // 中心化
            IMatrix normalizedFeatures = centeredFeatures.normalizeColumns(); // 列归一化
            
            // 计算协方差矩阵 / Compute covariance matrix
            IMatrix covMatrix = normalizedFeatures.covariance();
            
            // 特征选择（例如PCA） / Feature selection (e.g., PCA)
            // 这里可以使用RerePCA类进行主成分分析
            // RerePCA can be used here for principal component analysis
            
            System.out.println("预处理完成，特征矩阵已准备就绪");
            
        } catch (IOException e) {
            System.err.println("数据准备失败: " + e.getMessage());
        }
    }
}
```

## 性能特性 / Performance Features

### 内存优化 / Memory Optimization
- 使用ArrayList存储列数据，支持动态扩展 / Uses ArrayList to store column data, supports dynamic expansion
- 智能的数据类型检测和转换 / Smart data type detection and conversion
- 高效的切片操作实现 / Efficient slicing operation implementation

### 数据处理优化 / Data Processing Optimization
- 优化的CSV解析算法 / Optimized CSV parsing algorithm
- 批量数据操作支持 / Batch data operation support
- 最小化临时对象创建 / Minimize temporary object creation

### 类型安全 / Type Safety
- 强类型系统，避免运行时错误 / Strong type system to avoid runtime errors
- 参数验证和异常处理 / Parameter validation and exception handling
- 清晰的数据类型定义 / Clear data type definitions

## 注意事项 / Notes

1. **数据类型** / **Data Types**: 只支持String和Float两种数据类型 / Only supports String and Float data types
2. **CSV格式** / **CSV Format**: 确保CSV文件格式正确，分隔符一致 / Ensure CSV file format is correct and delimiter is consistent
3. **内存使用** / **Memory Usage**: 大型数据集时注意内存使用 / Pay attention to memory usage for large datasets
4. **索引范围** / **Index Range**: 切片操作支持负数索引 / Slicing operations support negative indices
5. **矩阵转换** / **Matrix Conversion**: 只有Float类型列才能转换为矩阵 / Only Float type columns can be converted to matrix
6. **异常处理** / **Exception Handling**: 文件操作和矩阵转换可能抛出异常 / File operations and matrix conversion may throw exceptions

## 与pandas功能对照表 / Pandas Functionality Comparison Table

| 功能类别 / Function Category | DataFrame | pandas | 说明 / Description |
|---------|-----------|--------|------|
| **数据创建 / Data Creation** | | | |
| 从CSV读取 / Read from CSV | `DataFrame.readCsv(path, sep, header)` | `pd.read_csv(path, sep, header)` | 从CSV文件读取数据 / Read data from CSV file |
| 创建空DataFrame / Create empty DataFrame | `new DataFrame()` | `pd.DataFrame()` | 创建空数据框 / Create empty data frame |
| 从列列表创建 / Create from column list | `new DataFrame(columns)` | `pd.DataFrame(dict)` | 从列数据创建 / Create from column data |
| **数据访问 / Data Access** | | | |
| 获取形状 / Get shape | `df.shape()` | `df.shape` | 获取数据框形状 / Get data frame shape |
| 获取列名 / Get column names | `df.getColumnNames()` | `df.columns` | 获取列名列表 / Get column names list |
| 获取列 / Get column | `df.get(i)`, `df.getColumnByName(name)` | `df[col]`, `df[col_name]` | 获取指定列 / Get specified column |
| 检查是否为空 / Check if empty | `df.isEmpty()` | `df.empty` | 检查数据框是否为空 / Check if data frame is empty |
| **数据操作 / Data Operations** | | | |
| 添加列 / Add column | `df.addColumn(column)` | `df[new_col] = values` | 添加新列 / Add new column |
| 删除列 / Remove column | `df.removeColumn(i)` | `del df[col]`, `df.drop(col)` | 删除指定列 / Remove specified column |
| 列切片 / Column slicing | `df.sliceColumn(start, end)` | `df.iloc[:, start:end]` | 按列切片 / Slice by columns |
| 行切片 / Row slicing | `df.slice(row_exp, null)` | `df.iloc[start:end, :]` | 按行切片 / Slice by rows |
| 通用切片 / General slicing | `df.slice(row_exp, col_exp)` | `df.iloc[row_slice, col_slice]` | 通用切片操作 / General slicing operation |
| **数据类型转换 / Data Type Conversion** | | | |
| 转换为矩阵 / Convert to matrix | `df.toMatrix()` | `df.values`, `df.to_numpy()` | 转换为数值矩阵 / Convert to numeric matrix |
| 列转向量 / Column to vector | `col.toVec()` | `df[col].values` | 将列转换为向量 / Convert column to vector |
| 列转字符串列表 / Column to string list | `col.toStringList()` | `df[col].astype(str).tolist()` | 将列转换为字符串列表 / Convert column to string list |
| **数据输出 / Data Output** | | | |
| 保存为CSV / Save to CSV | `df.toCsv(path)` | `df.to_csv(path)` | 保存为CSV文件 / Save to CSV file |
| **数据复制 / Data Copying** | | | |
| 复制数据框 / Copy data frame | `df.copy()` | `df.copy()` | 深拷贝数据框 / Deep copy data frame |
| 清空数据框 / Clear data frame | `df.clear()` | `df.drop(df.index, inplace=True)` | 清空所有数据 / Clear all data |

## 最佳实践建议 / Best Practices Recommendations

### 数据加载 / Data Loading
1. **文件格式检查** / **File Format Check**: 确保CSV文件格式正确 / Ensure CSV file format is correct
2. **编码处理** / **Encoding Handling**: 注意文件编码，建议使用UTF-8 / Pay attention to file encoding, recommend UTF-8
3. **内存管理** / **Memory Management**: 大型文件考虑分批处理 / Consider batch processing for large files

### 数据处理 / Data Processing
1. **类型转换** / **Type Conversion**: 及时转换数据类型以提高性能 / Convert data types timely to improve performance
2. **切片优化** / **Slicing Optimization**: 使用切片操作减少内存使用 / Use slicing operations to reduce memory usage
3. **异常处理** / **Exception Handling**: 妥善处理文件操作和转换异常 / Properly handle file operation and conversion exceptions

### 性能优化 / Performance Optimization
1. **批量操作** / **Batch Operations**: 优先使用批量操作而不是循环 / Prefer batch operations over loops
2. **内存释放** / **Memory Release**: 及时释放不需要的DataFrame引用 / Release unnecessary DataFrame references promptly
3. **数据类型选择** / **Data Type Selection**: 根据精度需求选择合适的数值类型 / Choose appropriate numeric types based on precision requirements

---

**DataFrame 数据框操作** - 结构化数据处理的核心，让数据分析更简单！

**DataFrame Operations** - The core of structured data processing, making data analysis simpler!
