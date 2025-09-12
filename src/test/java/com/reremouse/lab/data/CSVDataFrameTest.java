package com.reremouse.lab.data;

import com.reremouse.lab.math.data.DataFrame;
import com.reremouse.lab.math.IMatrix;
import java.io.IOException;

/**
 * CSV DataFrame测试 / CSV DataFrame Test
 */
public class CSVDataFrameTest {
    
    public static void main(String[] args) {
        System.out.println("开始CSV DataFrame测试... / Starting CSV DataFrame test...");
        
        try {
            // 测试读取CSV文件 / Test reading CSV file
            DataFrame df = DataFrame.readCsv("test_data.csv", ",", true);
            
            System.out.println("CSV文件读取成功 / CSV file read successfully");
            System.out.println("DataFrame形状: " + df.getRowCount() + "行 x " + df.getColumnCount() + "列 / DataFrame shape: " + df.getRowCount() + " rows x " + df.getColumnCount() + " columns");
            
            // 显示列信息 / Display column information
            System.out.println("列名: " + df.getColumnNames() + " / Column names: " + df.getColumnNames());
            System.out.println("列类型: " + df.getColumnTypes() + " / Column types: " + df.getColumnTypes());
            
            // 显示前几行数据 / Display first few rows
            System.out.println("前3行数据: / First 3 rows:");
            for (int i = 0; i < Math.min(3, df.getRowCount()); i++) {
                System.out.print("行 " + i + ": ");
                for (int j = 0; j < df.getColumnCount(); j++) {
                    System.out.print(df.get(j).getData().get(i) + " ");
                }
                System.out.println();
            }
            
            // 测试转换为矩阵（只包含数值列）/ Test matrix conversion (only numeric columns)
            try {
                IMatrix matrix = df.toMatrix();
                System.out.println("数值列矩阵形状: " + matrix.getRowNum() + "x" + matrix.getColNum() + " / Numeric matrix shape: " + matrix.getRowNum() + "x" + matrix.getColNum());
                System.out.println("数值列矩阵数据: / Numeric matrix data:");
                for (int i = 0; i < matrix.getRowNum(); i++) {
                    for (int j = 0; j < matrix.getColNum(); j++) {
                        System.out.print(matrix.get(i, j) + " ");
                    }
                    System.out.println();
                }
            } catch (IllegalStateException e) {
                System.out.println("没有数值列可以转换为矩阵 / No numeric columns available for matrix conversion");
            }
            
            // 测试切片 / Test slicing
            DataFrame sliced = df.slice("1:3", "1:3"); // 行1-2，列1-2
            System.out.println("切片后形状: " + sliced.getRowCount() + "行 x " + sliced.getColumnCount() + "列 / Sliced shape: " + sliced.getRowCount() + " rows x " + sliced.getColumnCount() + " columns");
            
            System.out.println("CSV测试完成！ / CSV test completed!");
            
        } catch (IOException e) {
            System.err.println("CSV文件读取失败: " + e.getMessage() + " / CSV file reading failed: " + e.getMessage());
            e.printStackTrace();
        } catch (Exception e) {
            System.err.println("测试失败: " + e.getMessage() + " / Test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
