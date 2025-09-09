package com.reremouse.lab.data;

import com.reremouse.lab.math.IMatrix;
import java.util.ArrayList;
import java.util.List;

/**
 * 简单的toMatrix测试 / Simple toMatrix Test
 */
public class SimpleToMatrixTest {
    
    public static void main(String[] args) {
        System.out.println("开始简单toMatrix测试... / Starting simple toMatrix test...");
        
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
            
            System.out.println("DataFrame信息:");
            System.out.println("列数: " + df.getColumnCount());
            System.out.println("行数: " + df.getRowCount());
            
            // 检查每列的数据长度 / Check data length of each column
            for (int i = 0; i < df.getColumnCount(); i++) {
                Column col = df.get(i);
                System.out.println("列 " + i + " (" + col.getName() + "): 数据长度 = " + col.getData().size());
            }
            
            // 测试转换为矩阵 / Test matrix conversion
            System.out.println("开始转换为矩阵... / Starting matrix conversion...");
            IMatrix matrix = df.toMatrix();
            System.out.println("矩阵转换成功！/ Matrix conversion successful!");
            System.out.println("矩阵形状: " + matrix.getRowNum() + "x" + matrix.getColNum());
            
            // 显示矩阵数据 / Display matrix data
            System.out.println("矩阵数据: / Matrix data:");
            for (int i = 0; i < matrix.getRowNum(); i++) {
                for (int j = 0; j < matrix.getColNum(); j++) {
                    System.out.print(matrix.get(i, j) + " ");
                }
                System.out.println();
            }
            
            System.out.println("toMatrix测试完成！/ toMatrix test completed!");
            
        } catch (Exception e) {
            System.err.println("测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
