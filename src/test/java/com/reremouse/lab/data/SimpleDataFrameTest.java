package com.reremouse.lab.data;

import com.reremouse.lab.math.data.Column;
import com.reremouse.lab.math.data.DataFrame;
import com.reremouse.lab.math.data.ColumnType;
import com.reremouse.lab.math.IMatrix;
import java.util.ArrayList;
import java.util.List;

/**
 * 简单的DataFrame测试 / Simple DataFrame Test
 */
public class SimpleDataFrameTest {
    
    public static void main(String[] args) {
        System.out.println("开始简单DataFrame测试... / Starting simple DataFrame test...");
        
        try {
            // 创建测试数据 / Create test data
            DataFrame df = new DataFrame();
            
            // 添加Float类型列 / Add Float type column
            Column floatCol1 = new Column();
            floatCol1.setName("feature1");
            floatCol1.setColumnType(ColumnType.Float);
            List<Object> floatData1 = new ArrayList<>();
            floatData1.add(1.0f);
            floatData1.add(2.0f);
            floatData1.add(3.0f);
            floatCol1.setData(floatData1);
            df.addColumn(floatCol1);
            
            // 添加另一个Float类型列 / Add another Float type column
            Column floatCol2 = new Column();
            floatCol2.setName("feature2");
            floatCol2.setColumnType(ColumnType.Float);
            List<Object> floatData2 = new ArrayList<>();
            floatData2.add(4.0f);
            floatData2.add(5.0f);
            floatData2.add(6.0f);
            floatCol2.setData(floatData2);
            df.addColumn(floatCol2);
            
            System.out.println("DataFrame创建成功 / DataFrame created successfully");
            System.out.println("列数: " + df.getColumnCount() + " / Column count: " + df.getColumnCount());
            System.out.println("行数: " + df.getRowCount() + " / Row count: " + df.getRowCount());
            
            // 测试获取列 / Test getting columns
            Column col1 = df.get(0);
            System.out.println("第一列名称: " + col1.getName() + " / First column name: " + col1.getName());
            
            // 测试转换为矩阵 / Test matrix conversion
            IMatrix matrix = df.toMatrix();
            System.out.println("矩阵形状: " + matrix.getRowNum() + "x" + matrix.getColNum() + " / Matrix shape: " + matrix.getRowNum() + "x" + matrix.getColNum());
            System.out.println("矩阵数据: / Matrix data:");
            for (int i = 0; i < matrix.getRowNum(); i++) {
                for (int j = 0; j < matrix.getColNum(); j++) {
                    System.out.print(matrix.get(i, j) + " ");
                }
                System.out.println();
            }
            
            // 测试切片 / Test slicing
            DataFrame sliced = df.sliceColumn(0, 1);
            System.out.println("切片后列数: " + sliced.getColumnCount() + " / Column count after slicing: " + sliced.getColumnCount());
            
            System.out.println("所有测试通过！ / All tests passed!");
            
        } catch (Exception e) {
            System.err.println("测试失败: " + e.getMessage() + " / Test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
