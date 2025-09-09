# DataFrame 数据框示例 (DataFrame Examples)

## 概述 / Overview

本文档提供了 `DataFrame` 类的详细使用示例，涵盖从基础操作到高级应用的各个方面。

This document provides detailed usage examples for the `DataFrame` class, covering everything from basic operations to advanced applications.

## 基础示例 / Basic Examples

### DataFrame 创建和基本操作 / DataFrame Creation and Basic Operations

```java
import com.reremouse.lab.data.DataFrame;
import com.reremouse.lab.data.Column;
import com.reremouse.lab.data.ColumnType;
import com.reremouse.lab.math.IMatrix;
import com.reremouse.lab.math.IVector;
import java.util.Arrays;
import java.util.List;

public class DataFrameBasicExample {
    public static void main(String[] args) {
        // 创建空DataFrame / Create empty DataFrame
        DataFrame df = new DataFrame();
        
        // 添加字符串列 / Add string column
        Column nameColumn = new Column();
        nameColumn.setName("name");
        nameColumn.setColumnType(ColumnType.String);
        nameColumn.setData(Arrays.asList("Alice", "Bob", "Charlie", "David"));
        df.addColumn(nameColumn);
        
        // 添加数值列 / Add numeric column
        Column ageColumn = new Column();
        ageColumn.setName("age");
        ageColumn.setColumnType(ColumnType.Float);
        ageColumn.setData(Arrays.asList(25.0f, 30.0f, 35.0f, 28.0f));
        df.addColumn(ageColumn);
        
        // 添加薪资列 / Add salary column
        Column salaryColumn = new Column();
        salaryColumn.setName("salary");
        salaryColumn.setColumnType(ColumnType.Float);
        salaryColumn.setData(Arrays.asList(50000.0f, 60000.0f, 70000.0f, 55000.0f));
        df.addColumn(salaryColumn);
        
        // 显示DataFrame信息 / Display DataFrame information
        System.out.println("DataFrame形状: " + Arrays.toString(df.shape()));
        System.out.println("行数: " + df.getRowCount());
        System.out.println("列数: " + df.getColumnCount());
        System.out.println("列名: " + df.getColumnNames());
        System.out.println("列类型: " + df.getColumnTypes());
        System.out.println("是否为空: " + df.isEmpty());
        System.out.println("\nDataFrame内容:\n" + df);
    }
}
```

### 从CSV文件读取数据 / Reading Data from CSV Files

```java
import com.reremouse.lab.data.DataFrame;
import java.io.IOException;

public class CSVReadingExample {
    public static void main(String[] args) {
        try {
            // 从CSV文件读取数据 / Read data from CSV file
            // CSV文件内容示例 / Example CSV file content:
            // name,age,salary,department
            // Alice,25,50000,IT
            // Bob,30,60000,HR
            // Charlie,35,70000,Finance
            // David,28,55000,IT
            
            DataFrame df = DataFrame.readCsv("employees.csv", ",", true);
            
            System.out.println("从CSV读取的DataFrame:");
            System.out.println("形状: " + Arrays.toString(df.shape()));
            System.out.println("列名: " + df.getColumnNames());
            System.out.println("列类型: " + df.getColumnTypes());
            System.out.println("\n内容:\n" + df);
            
        } catch (IOException e) {
            System.err.println("读取CSV文件失败: " + e.getMessage());
        }
    }
}
```

### 数据访问和列操作 / Data Access and Column Operations

```java
public class DataAccessExample {
    public static void main(String[] args) {
        DataFrame df = createSampleDataFrame();
        
        // 获取列 / Get columns
        Column nameColumn = df.get(0);                    // 按索引获取
        Column ageColumn = df.getColumnByName("age");     // 按名称获取
        
        System.out.println("第0列: " + nameColumn.getName());
        System.out.println("年龄列数据: " + ageColumn.getData());
        
        // 添加新列 / Add new column
        Column departmentColumn = new Column();
        departmentColumn.setName("department");
        departmentColumn.setColumnType(ColumnType.String);
        departmentColumn.setData(Arrays.asList("IT", "HR", "Finance", "IT"));
        df.addColumn(departmentColumn);
        
        System.out.println("添加部门列后的列名: " + df.getColumnNames());
        
        // 删除列 / Remove column
        Column removedColumn = df.removeColumn(-1); // 删除最后一列
        System.out.println("删除的列: " + removedColumn.getName());
        System.out.println("删除后的列数: " + df.getColumnCount());
    }
    
    private static DataFrame createSampleDataFrame() {
        DataFrame df = new DataFrame();
        
        Column nameColumn = new Column();
        nameColumn.setName("name");
        nameColumn.setColumnType(ColumnType.String);
        nameColumn.setData(Arrays.asList("Alice", "Bob", "Charlie", "David"));
        df.addColumn(nameColumn);
        
        Column ageColumn = new Column();
        ageColumn.setName("age");
        ageColumn.setColumnType(ColumnType.Float);
        ageColumn.setData(Arrays.asList(25.0f, 30.0f, 35.0f, 28.0f));
        df.addColumn(ageColumn);
        
        return df;
    }
}
```

### 切片操作 / Slicing Operations

```java
public class SlicingExample {
    public static void main(String[] args) {
        DataFrame df = createSampleDataFrame();
        
        // 列切片 / Column slicing
        DataFrame colSlice1 = df.sliceColumn(1, 3);     // 列1到2
        DataFrame colSlice2 = df.sliceColumn(0, 4, 2);  // 列0,2（步长为2）
        DataFrame colSlice3 = df.sliceColumn(-2);       // 最后两列
        
        System.out.println("列切片1 (1:3):\n" + colSlice1);
        System.out.println("列切片2 (0:4:2):\n" + colSlice2);
        System.out.println("列切片3 (-2:):\n" + colSlice3);
        
        // 通用切片 / General slicing
        DataFrame sliced1 = df.slice("1:3", "0:2");     // 行1-2，列0-1
        DataFrame sliced2 = df.slice("0:4:2", "1:3");  // 行0,2，列1-2
        DataFrame rowOnly = df.slice("1:3", null);      // 只进行行切片
        DataFrame colOnly = df.slice(null, "0:2");      // 只进行列切片
        
        System.out.println("通用切片1 (行1:3, 列0:2):\n" + sliced1);
        System.out.println("通用切片2 (行0:4:2, 列1:3):\n" + sliced2);
        System.out.println("行切片 (行1:3):\n" + rowOnly);
        System.out.println("列切片 (列0:2):\n" + colOnly);
    }
    
    private static DataFrame createSampleDataFrame() {
        DataFrame df = new DataFrame();
        
        // 创建示例数据 / Create sample data
        String[] names = {"Alice", "Bob", "Charlie", "David", "Eve"};
        Float[] ages = {25.0f, 30.0f, 35.0f, 28.0f, 32.0f};
        Float[] salaries = {50000.0f, 60000.0f, 70000.0f, 55000.0f, 65000.0f};
        String[] departments = {"IT", "HR", "Finance", "IT", "Marketing"};
        
        Column nameColumn = new Column();
        nameColumn.setName("name");
        nameColumn.setColumnType(ColumnType.String);
        nameColumn.setData(Arrays.asList(names));
        df.addColumn(nameColumn);
        
        Column ageColumn = new Column();
        ageColumn.setName("age");
        ageColumn.setColumnType(ColumnType.Float);
        ageColumn.setData(Arrays.asList(ages));
        df.addColumn(ageColumn);
        
        Column salaryColumn = new Column();
        salaryColumn.setName("salary");
        salaryColumn.setColumnType(ColumnType.Float);
        salaryColumn.setData(Arrays.asList(salaries));
        df.addColumn(salaryColumn);
        
        Column deptColumn = new Column();
        deptColumn.setName("department");
        deptColumn.setColumnType(ColumnType.String);
        deptColumn.setData(Arrays.asList(departments));
        df.addColumn(deptColumn);
        
        return df;
    }
}
```

### 数据类型转换 / Data Type Conversion

```java
public class DataConversionExample {
    public static void main(String[] args) {
        DataFrame df = createSampleDataFrame();
        
        // 转换为矩阵 / Convert to matrix
        try {
            IMatrix matrix = df.toMatrix();
            System.out.println("转换后的矩阵:\n" + matrix);
            System.out.println("矩阵形状: " + Arrays.toString(matrix.shape()));
        } catch (IllegalStateException e) {
            System.out.println("转换失败: " + e.getMessage());
            System.out.println("只有Float类型的列才能转换为矩阵");
        }
        
        // 列数据转换 / Column data conversion
        Column ageColumn = df.getColumnByName("age");
        if (ageColumn != null) {
            // 转换为向量 / Convert to vector
            IVector ageVector = ageColumn.toVec();
            System.out.println("年龄向量: " + ageVector);
            System.out.println("向量长度: " + ageVector.length());
            
            // 转换为字符串列表 / Convert to string list
            List<String> ageStrings = ageColumn.toStringList();
            System.out.println("年龄字符串列表: " + ageStrings);
        }
        
        // 字符串列转换 / String column conversion
        Column nameColumn = df.getColumnByName("name");
        if (nameColumn != null) {
            List<String> nameStrings = nameColumn.toStringList();
            System.out.println("姓名列表: " + nameStrings);
        }
    }
    
    private static DataFrame createSampleDataFrame() {
        DataFrame df = new DataFrame();
        
        // 只添加数值列用于矩阵转换 / Add only numeric columns for matrix conversion
        Column ageColumn = new Column();
        ageColumn.setName("age");
        ageColumn.setColumnType(ColumnType.Float);
        ageColumn.setData(Arrays.asList(25.0f, 30.0f, 35.0f, 28.0f));
        df.addColumn(ageColumn);
        
        Column salaryColumn = new Column();
        salaryColumn.setName("salary");
        salaryColumn.setColumnType(ColumnType.Float);
        salaryColumn.setData(Arrays.asList(50000.0f, 60000.0f, 70000.0f, 55000.0f));
        df.addColumn(salaryColumn);
        
        Column experienceColumn = new Column();
        experienceColumn.setName("experience");
        experienceColumn.setColumnType(ColumnType.Float);
        experienceColumn.setData(Arrays.asList(2.0f, 5.0f, 8.0f, 3.0f));
        df.addColumn(experienceColumn);
        
        // 添加字符串列 / Add string column
        Column nameColumn = new Column();
        nameColumn.setName("name");
        nameColumn.setColumnType(ColumnType.String);
        nameColumn.setData(Arrays.asList("Alice", "Bob", "Charlie", "David"));
        df.addColumn(nameColumn);
        
        return df;
    }
}
```

### 数据保存 / Data Saving

```java
public class DataSavingExample {
    public static void main(String[] args) {
        DataFrame df = createSampleDataFrame();
        
        try {
            // 保存为CSV文件 / Save to CSV file
            df.toCsv("output_data.csv");
            System.out.println("数据已保存到 output_data.csv");
            
            // 验证保存的数据 / Verify saved data
            DataFrame loadedDf = DataFrame.readCsv("output_data.csv", ",", true);
            System.out.println("重新加载的数据:\n" + loadedDf);
            
        } catch (IOException e) {
            System.err.println("保存文件失败: " + e.getMessage());
        }
    }
    
    private static DataFrame createSampleDataFrame() {
        DataFrame df = new DataFrame();
        
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
        
        return df;
    }
}
```

## 高级示例 / Advanced Examples

### 数据预处理管道 / Data Preprocessing Pipeline

```java
public class DataPreprocessingPipeline {
    public static void main(String[] args) {
        try {
            // 1. 从CSV读取原始数据 / Read raw data from CSV
            DataFrame rawData = DataFrame.readCsv("raw_data.csv", ",", true);
            System.out.println("原始数据形状: " + Arrays.toString(rawData.shape()));
            System.out.println("原始数据:\n" + rawData);
            
            // 2. 数据清洗 / Data cleaning
            // 删除异常行 / Remove abnormal rows
            DataFrame cleaned = rawData.slice("0:10", null); // 只取前10行
            System.out.println("清洗后数据形状: " + Arrays.toString(cleaned.shape()));
            
            // 3. 特征选择 / Feature selection
            // 只选择数值型列 / Select only numeric columns
            DataFrame numericOnly = cleaned.slice(null, "1:4"); // 假设列1-3是数值型
            System.out.println("数值型列:\n" + numericOnly);
            
            // 4. 转换为矩阵进行进一步处理 / Convert to matrix for further processing
            IMatrix matrix = numericOnly.toMatrix();
            System.out.println("特征矩阵形状: " + Arrays.toString(matrix.shape()));
            
            // 5. 数据标准化 / Data standardization
            IMatrix centered = matrix.center(); // 中心化
            IMatrix standardized = centered.normalizeColumns(); // 列归一化
            
            System.out.println("标准化后的矩阵:\n" + standardized);
            
            // 6. 保存处理后的数据 / Save processed data
            // 这里可以将标准化后的矩阵转换回DataFrame并保存
            // Here we can convert the standardized matrix back to DataFrame and save
            
        } catch (IOException e) {
            System.err.println("数据处理失败: " + e.getMessage());
        }
    }
}
```

### 数据分析和统计 / Data Analysis and Statistics

```java
public class DataAnalysisExample {
    public static void main(String[] args) {
        DataFrame df = createSampleDataFrame();
        
        // 基本统计信息 / Basic statistics
        System.out.println("=== 基本统计信息 ===");
        System.out.println("数据形状: " + Arrays.toString(df.shape()));
        System.out.println("列名: " + df.getColumnNames());
        System.out.println("列类型: " + df.getColumnTypes());
        
        // 数值型列的统计 / Statistics for numeric columns
        System.out.println("\n=== 数值型列统计 ===");
        for (int i = 0; i < df.getColumnCount(); i++) {
            Column col = df.get(i);
            if (col.getColumnType() == ColumnType.Float) {
                System.out.println("\n列 " + col.getName() + " 的统计信息:");
                
                // 转换为向量进行统计 / Convert to vector for statistics
                IVector vector = col.toVec();
                System.out.println("  均值: " + vector.mean());
                System.out.println("  标准差: " + vector.std());
                System.out.println("  方差: " + vector.variance());
                System.out.println("  最小值: " + vector.min());
                System.out.println("  最大值: " + vector.max());
                System.out.println("  总和: " + vector.sum());
                System.out.println("  中位数: " + vector.median());
            }
        }
        
        // 数据切片分析 / Data slicing analysis
        System.out.println("\n=== 数据切片分析 ===");
        DataFrame youngPeople = df.slice("0:2", null); // 前两行
        System.out.println("年轻人群数据:\n" + youngPeople);
        
        DataFrame highSalary = df.slice("2:4", null); // 第3-4行
        System.out.println("高薪人群数据:\n" + highSalary);
    }
    
    private static DataFrame createSampleDataFrame() {
        DataFrame df = new DataFrame();
        
        // 创建示例数据 / Create sample data
        String[] names = {"Alice", "Bob", "Charlie", "David", "Eve"};
        Float[] ages = {25.0f, 30.0f, 35.0f, 28.0f, 32.0f};
        Float[] salaries = {50000.0f, 60000.0f, 70000.0f, 55000.0f, 65000.0f};
        Float[] experience = {2.0f, 5.0f, 8.0f, 3.0f, 6.0f};
        
        Column nameColumn = new Column();
        nameColumn.setName("name");
        nameColumn.setColumnType(ColumnType.String);
        nameColumn.setData(Arrays.asList(names));
        df.addColumn(nameColumn);
        
        Column ageColumn = new Column();
        ageColumn.setName("age");
        ageColumn.setColumnType(ColumnType.Float);
        ageColumn.setData(Arrays.asList(ages));
        df.addColumn(ageColumn);
        
        Column salaryColumn = new Column();
        salaryColumn.setName("salary");
        salaryColumn.setColumnType(ColumnType.Float);
        salaryColumn.setData(Arrays.asList(salaries));
        df.addColumn(salaryColumn);
        
        Column expColumn = new Column();
        expColumn.setName("experience");
        expColumn.setColumnType(ColumnType.Float);
        expColumn.setData(Arrays.asList(experience));
        df.addColumn(expColumn);
        
        return df;
    }
}
```

### 机器学习数据准备 / Machine Learning Data Preparation

```java
public class MLDataPreparationExample {
    public static void main(String[] args) {
        try {
            // 读取原始数据 / Read raw data
            DataFrame rawData = DataFrame.readCsv("ml_data.csv", ",", true);
            System.out.println("原始数据形状: " + Arrays.toString(rawData.shape()));
            
            // 分离特征和标签 / Separate features and labels
            DataFrame features = rawData.slice(null, "0:-1"); // 除最后一列外的所有列
            DataFrame labels = rawData.slice(null, "-1:");    // 最后一列
            
            System.out.println("特征数据形状: " + Arrays.toString(features.shape()));
            System.out.println("标签数据形状: " + Arrays.toString(labels.shape()));
            
            // 转换特征为矩阵 / Convert features to matrix
            IMatrix featureMatrix = features.toMatrix();
            System.out.println("特征矩阵形状: " + Arrays.toString(featureMatrix.shape()));
            
            // 数据预处理 / Data preprocessing
            IMatrix centeredFeatures = featureMatrix.center();        // 中心化
            IMatrix normalizedFeatures = centeredFeatures.normalizeColumns(); // 列归一化
            
            System.out.println("中心化后特征矩阵:\n" + centeredFeatures);
            System.out.println("归一化后特征矩阵:\n" + normalizedFeatures);
            
            // 计算协方差矩阵 / Compute covariance matrix
            IMatrix covMatrix = normalizedFeatures.covariance();
            System.out.println("协方差矩阵:\n" + covMatrix);
            
            // 特征选择（例如PCA） / Feature selection (e.g., PCA)
            // 这里可以使用RerePCA类进行主成分分析
            // RerePCA can be used here for principal component analysis
            
            System.out.println("数据预处理完成，特征矩阵已准备就绪");
            
        } catch (IOException e) {
            System.err.println("数据准备失败: " + e.getMessage());
        }
    }
}
```

### 数据验证和清理 / Data Validation and Cleaning

```java
public class DataValidationExample {
    public static void main(String[] args) {
        DataFrame df = createSampleDataFrame();
        
        // 数据验证 / Data validation
        System.out.println("=== 数据验证 ===");
        System.out.println("数据形状: " + Arrays.toString(df.shape()));
        System.out.println("列数: " + df.getColumnCount());
        System.out.println("行数: " + df.getRowCount());
        
        // 检查列数据一致性 / Check column data consistency
        boolean isConsistent = true;
        int expectedRows = df.getRowCount();
        for (int i = 0; i < df.getColumnCount(); i++) {
            Column col = df.get(i);
            if (col.getData().size() != expectedRows) {
                System.out.println("警告: 列 " + col.getName() + " 数据长度不一致");
                isConsistent = false;
            }
        }
        
        if (isConsistent) {
            System.out.println("所有列数据长度一致");
        }
        
        // 数据类型检查 / Data type checking
        System.out.println("\n=== 数据类型检查 ===");
        for (int i = 0; i < df.getColumnCount(); i++) {
            Column col = df.get(i);
            System.out.println("列 " + col.getName() + ": " + col.getColumnType());
        }
        
        // 数据清理 / Data cleaning
        System.out.println("\n=== 数据清理 ===");
        // 移除空值或异常值 / Remove null values or outliers
        DataFrame cleaned = df.slice("0:3", null); // 只取前3行
        System.out.println("清理后数据:\n" + cleaned);
        
        // 数据复制 / Data copying
        DataFrame dfCopy = df.copy();
        System.out.println("复制的数据框:\n" + dfCopy);
        
        // 清空数据 / Clear data
        df.clear();
        System.out.println("清空后的数据框是否为空: " + df.isEmpty());
    }
    
    private static DataFrame createSampleDataFrame() {
        DataFrame df = new DataFrame();
        
        Column nameColumn = new Column();
        nameColumn.setName("name");
        nameColumn.setColumnType(ColumnType.String);
        nameColumn.setData(Arrays.asList("Alice", "Bob", "Charlie", "David"));
        df.addColumn(nameColumn);
        
        Column ageColumn = new Column();
        ageColumn.setName("age");
        ageColumn.setColumnType(ColumnType.Float);
        ageColumn.setData(Arrays.asList(25.0f, 30.0f, 35.0f, 28.0f));
        df.addColumn(ageColumn);
        
        return df;
    }
}
```

### 批量数据处理 / Batch Data Processing

```java
public class BatchProcessingExample {
    public static void main(String[] args) {
        // 创建多个DataFrame进行批量处理 / Create multiple DataFrames for batch processing
        DataFrame[] dataFrames = new DataFrame[5];
        
        for (int i = 0; i < dataFrames.length; i++) {
            dataFrames[i] = createSampleDataFrame(i);
        }
        
        // 批量统计 / Batch statistics
        System.out.println("=== 批量统计 ===");
        for (int i = 0; i < dataFrames.length; i++) {
            DataFrame df = dataFrames[i];
            System.out.println("DataFrame " + i + " 形状: " + Arrays.toString(df.shape()));
            
            // 计算数值列的统计信息 / Calculate statistics for numeric columns
            for (int j = 0; j < df.getColumnCount(); j++) {
                Column col = df.get(j);
                if (col.getColumnType() == ColumnType.Float) {
                    IVector vector = col.toVec();
                    System.out.println("  列 " + col.getName() + " 均值: " + vector.mean());
                }
            }
        }
        
        // 批量转换 / Batch conversion
        System.out.println("\n=== 批量转换 ===");
        IMatrix[] matrices = new IMatrix[dataFrames.length];
        for (int i = 0; i < dataFrames.length; i++) {
            try {
                matrices[i] = dataFrames[i].toMatrix();
                System.out.println("DataFrame " + i + " 转换为矩阵成功");
            } catch (IllegalStateException e) {
                System.out.println("DataFrame " + i + " 转换失败: " + e.getMessage());
            }
        }
        
        // 批量保存 / Batch saving
        System.out.println("\n=== 批量保存 ===");
        for (int i = 0; i < dataFrames.length; i++) {
            try {
                dataFrames[i].toCsv("batch_data_" + i + ".csv");
                System.out.println("DataFrame " + i + " 保存成功");
            } catch (IOException e) {
                System.err.println("DataFrame " + i + " 保存失败: " + e.getMessage());
            }
        }
    }
    
    private static DataFrame createSampleDataFrame(int index) {
        DataFrame df = new DataFrame();
        
        // 创建不同的示例数据 / Create different sample data
        String[] names = {"Alice" + index, "Bob" + index, "Charlie" + index};
        Float[] ages = {25.0f + index, 30.0f + index, 35.0f + index};
        Float[] salaries = {50000.0f + index * 1000, 60000.0f + index * 1000, 70000.0f + index * 1000};
        
        Column nameColumn = new Column();
        nameColumn.setName("name");
        nameColumn.setColumnType(ColumnType.String);
        nameColumn.setData(Arrays.asList(names));
        df.addColumn(nameColumn);
        
        Column ageColumn = new Column();
        ageColumn.setName("age");
        ageColumn.setColumnType(ColumnType.Float);
        ageColumn.setData(Arrays.asList(ages));
        df.addColumn(ageColumn);
        
        Column salaryColumn = new Column();
        salaryColumn.setName("salary");
        salaryColumn.setColumnType(ColumnType.Float);
        salaryColumn.setData(Arrays.asList(salaries));
        df.addColumn(salaryColumn);
        
        return df;
    }
}
```

## 错误处理和调试示例 / Error Handling and Debugging Examples

### 异常处理 / Exception Handling

```java
public class ExceptionHandlingExample {
    public static void main(String[] args) {
        // 文件读取异常处理 / File reading exception handling
        try {
            DataFrame df = DataFrame.readCsv("nonexistent.csv", ",", true);
        } catch (IOException e) {
            System.err.println("文件读取失败: " + e.getMessage());
        }
        
        // 矩阵转换异常处理 / Matrix conversion exception handling
        DataFrame df = createStringOnlyDataFrame();
        try {
            IMatrix matrix = df.toMatrix();
        } catch (IllegalStateException e) {
            System.err.println("矩阵转换失败: " + e.getMessage());
            System.out.println("只有Float类型的列才能转换为矩阵");
        }
        
        // 索引越界异常处理 / Index out of bounds exception handling
        try {
            Column col = df.get(10); // 假设只有3列
        } catch (IndexOutOfBoundsException e) {
            System.err.println("索引越界: " + e.getMessage());
        }
        
        // 参数验证异常处理 / Parameter validation exception handling
        try {
            df.sliceColumn(1, 0); // 起始位置大于结束位置
        } catch (IllegalArgumentException e) {
            System.err.println("参数无效: " + e.getMessage());
        }
    }
    
    private static DataFrame createStringOnlyDataFrame() {
        DataFrame df = new DataFrame();
        
        Column nameColumn = new Column();
        nameColumn.setName("name");
        nameColumn.setColumnType(ColumnType.String);
        nameColumn.setData(Arrays.asList("Alice", "Bob", "Charlie"));
        df.addColumn(nameColumn);
        
        return df;
    }
}
```

### 数据验证和调试 / Data Validation and Debugging

```java
public class DataValidationDebuggingExample {
    public static void main(String[] args) {
        DataFrame df = createSampleDataFrame();
        
        // 数据完整性检查 / Data integrity check
        System.out.println("=== 数据完整性检查 ===");
        System.out.println("DataFrame是否为空: " + df.isEmpty());
        System.out.println("行数: " + df.getRowCount());
        System.out.println("列数: " + df.getColumnCount());
        
        // 列数据长度检查 / Column data length check
        System.out.println("\n=== 列数据长度检查 ===");
        for (int i = 0; i < df.getColumnCount(); i++) {
            Column col = df.get(i);
            System.out.println("列 " + col.getName() + " 数据长度: " + col.getData().size());
        }
        
        // 数据类型验证 / Data type validation
        System.out.println("\n=== 数据类型验证 ===");
        for (int i = 0; i < df.getColumnCount(); i++) {
            Column col = df.get(i);
            System.out.println("列 " + col.getName() + " 类型: " + col.getColumnType());
            
            // 验证数据是否与类型匹配 / Verify data matches type
            if (col.getColumnType() == ColumnType.Float) {
                boolean allNumeric = true;
                for (Object value : col.getData()) {
                    try {
                        Float.parseFloat(value.toString());
                    } catch (NumberFormatException e) {
                        allNumeric = false;
                        break;
                    }
                }
                System.out.println("  所有数据都是数值型: " + allNumeric);
            }
        }
        
        // 切片操作验证 / Slicing operation validation
        System.out.println("\n=== 切片操作验证 ===");
        try {
            DataFrame sliced = df.slice("0:2", "0:2");
            System.out.println("切片操作成功，结果形状: " + Arrays.toString(sliced.shape()));
        } catch (Exception e) {
            System.err.println("切片操作失败: " + e.getMessage());
        }
    }
    
    private static DataFrame createSampleDataFrame() {
        DataFrame df = new DataFrame();
        
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
        
        return df;
    }
}
```

## 性能优化示例 / Performance Optimization Examples

### 内存优化 / Memory Optimization

```java
public class MemoryOptimizationExample {
    public static void main(String[] args) {
        // 避免创建过多临时DataFrame / Avoid creating too many temporary DataFrames
        DataFrame df = createLargeDataFrame();
        
        // 不好的做法 / Bad practice
        DataFrame temp1 = df.slice("0:100", null);
        DataFrame temp2 = temp1.slice(null, "0:2");
        DataFrame temp3 = temp2.slice("0:50", null);
        DataFrame result1 = temp3.slice(null, "0:1");
        
        // 好的做法 / Good practice
        DataFrame result2 = df.slice("0:50", "0:1");
        
        // 及时释放引用 / Release references promptly
        temp1 = null;
        temp2 = null;
        temp3 = null;
        
        System.out.println("优化前结果形状: " + Arrays.toString(result1.shape()));
        System.out.println("优化后结果形状: " + Arrays.toString(result2.shape()));
    }
    
    private static DataFrame createLargeDataFrame() {
        DataFrame df = new DataFrame();
        
        // 创建较大的数据集 / Create larger dataset
        List<Object> names = new ArrayList<>();
        List<Object> ages = new ArrayList<>();
        List<Object> salaries = new ArrayList<>();
        
        for (int i = 0; i < 1000; i++) {
            names.add("Person" + i);
            ages.add(20.0f + i % 50);
            salaries.add(30000.0f + i * 100);
        }
        
        Column nameColumn = new Column();
        nameColumn.setName("name");
        nameColumn.setColumnType(ColumnType.String);
        nameColumn.setData(names);
        df.addColumn(nameColumn);
        
        Column ageColumn = new Column();
        ageColumn.setName("age");
        ageColumn.setColumnType(ColumnType.Float);
        ageColumn.setData(ages);
        df.addColumn(ageColumn);
        
        Column salaryColumn = new Column();
        salaryColumn.setName("salary");
        salaryColumn.setColumnType(ColumnType.Float);
        salaryColumn.setData(salaries);
        df.addColumn(salaryColumn);
        
        return df;
    }
}
```

## 总结 / Summary

本文档展示了DataFrame的全面使用方法，从基础操作到高级应用。建议在实际使用中：

1. **合理使用切片操作** / **Use slicing operations reasonably**: 避免创建过多临时DataFrame / Avoid creating too many temporary DataFrames
2. **注意数据类型** / **Pay attention to data types**: 确保数据类型与操作匹配 / Ensure data types match operations
3. **异常处理** / **Exception handling**: 妥善处理文件操作和转换异常 / Properly handle file operation and conversion exceptions
4. **内存管理** / **Memory management**: 及时释放不需要的引用 / Release unnecessary references promptly
5. **数据验证** / **Data validation**: 确保数据完整性和一致性 / Ensure data integrity and consistency
6. **性能优化** / **Performance optimization**: 使用批量操作和链式调用 / Use batch operations and method chaining

---

**DataFrame 数据框示例** - 从基础到高级，掌握数据处理的核心！

**DataFrame Examples** - From basics to advanced, master the core of data processing!
