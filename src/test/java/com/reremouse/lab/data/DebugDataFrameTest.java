package com.reremouse.lab.data;

import com.reremouse.lab.math.data.Column;
import com.reremouse.lab.math.data.DataFrame;
import com.reremouse.lab.math.data.ColumnType;
import java.util.ArrayList;
import java.util.List;

/**
 * 调试DataFrame测试 / Debug DataFrame Test
 */
public class DebugDataFrameTest {
    
    public static void main(String[] args) {
        System.out.println("开始调试DataFrame测试... / Starting debug DataFrame test...");
        
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
            
            System.out.println("添加第一列前 - rowCount: " + df.getRowCount());
            df.addColumn(floatCol1);
            System.out.println("添加第一列后 - rowCount: " + df.getRowCount());
            System.out.println("第一列数据长度: " + floatCol1.getData().size());
            
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
            
            System.out.println("添加第二列前 - rowCount: " + df.getRowCount());
            df.addColumn(floatCol2);
            System.out.println("添加第二列后 - rowCount: " + df.getRowCount());
            System.out.println("第二列数据长度: " + floatCol2.getData().size());
            
            // 添加String类型列 / Add String type column
            Column stringCol = new Column();
            stringCol.setName("label");
            stringCol.setColumnType(ColumnType.String);
            List<Object> stringData = new ArrayList<>();
            stringData.add("A");
            stringData.add("B");
            stringData.add("A");
            stringData.add("B");
            stringCol.setData(stringData);
            
            System.out.println("添加第三列前 - rowCount: " + df.getRowCount());
            df.addColumn(stringCol);
            System.out.println("添加第三列后 - rowCount: " + df.getRowCount());
            System.out.println("第三列数据长度: " + stringCol.getData().size());
            
            System.out.println("最终DataFrame信息:");
            System.out.println("列数: " + df.getColumnCount());
            System.out.println("行数: " + df.getRowCount());
            
            // 检查每列的数据长度 / Check data length of each column
            for (int i = 0; i < df.getColumnCount(); i++) {
                Column col = df.get(i);
                System.out.println("列 " + i + " (" + col.getName() + "): 数据长度 = " + col.getData().size());
            }
            
        } catch (Exception e) {
            System.err.println("调试测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
