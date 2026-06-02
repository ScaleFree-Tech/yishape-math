package com.yishape.lab.math.timeseries;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

public class TestA1Q3 {
    public static void main(String[] args) {
        // 测试Float向量
        System.out.println("测试RereFloatVector的q1()和q3()方法:");
        float[] floatData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
        IVector<Float> floatVector = Linalg.vector(floatData);
        
        try {
            var q1 = floatVector.q1();
            var q3 = floatVector.q3();
            System.out.println("Float向量数据: " + java.util.Arrays.toString(floatData));
            System.out.println("Q1 (25%分位数): " + q1);
            System.out.println("Q3 (75%分位数): " + q3);
        } catch (Exception e) {
            System.out.println("Float向量测试失败: " + e.getMessage());
        }
        
        // 测试Double向量
        System.out.println("\n测试RereDoubleVector的q1()和q3()方法:");
        double[] doubleData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
        IVector<Double> doubleVector = Linalg.vector(doubleData);
        
        try {
            Double q1 = doubleVector.q1();
            Double q3 = doubleVector.q3();
            System.out.println("Double向量数据: " + java.util.Arrays.toString(doubleData));
            System.out.println("Q1 (25%分位数): " + q1);
            System.out.println("Q3 (75%分位数): " + q3);
        } catch (Exception e) {
            System.out.println("Double向量测试失败: " + e.getMessage());
        }
        
        // 测试空向量
        System.out.println("\n测试空向量:");
        try {
            IVector emptyFloatVector = Linalg.vector(new float[0]);
            emptyFloatVector.q1();
        } catch (Exception e) {
            System.out.println("空Float向量测试通过: " + e.getMessage());
        }
        
        try {
            IVector emptyDoubleVector = Linalg.vector(new double[0]);
            emptyDoubleVector.q1();
        } catch (Exception e) {
            System.out.println("空Double向量测试通过: " + e.getMessage());
        }
    }
}
