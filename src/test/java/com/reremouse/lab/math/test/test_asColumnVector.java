package com.reremouse.lab.math.test;
import com.reremouse.lab.math.linalg.RereDoubleVector;
import com.reremouse.lab.math.linalg.IDoubleVector;
import com.reremouse.lab.math.linalg.IMatrix;

public class test_asColumnVector {
    public static void main(String[] args) {
        // 创建测试向量
        double[] data = {1.0, 2.0, 3.0, 4.0, 5.0};
        IDoubleVector vector = new RereDoubleVector(data);
        
        // 转换为列向量矩阵
        IMatrix<Double> columnMatrix = vector.asColumnVector();
        
        // 验证结果
        System.out.println("原始向量长度: " + vector.length());
        System.out.println("列向量矩阵维度: " + columnMatrix.getRowNum() + " x " + columnMatrix.getColNum());
        
        // 打印列向量矩阵
        System.out.println("列向量矩阵内容:");
        for (int i = 0; i < columnMatrix.getRowNum(); i++) {
            System.out.println("[" + i + "][0] = " + columnMatrix.get(i, 0));
        }
        
        // 验证数据一致性
        boolean isConsistent = true;
        for (int i = 0; i < vector.length(); i++) {
            if (Math.abs(vector.get(i) - columnMatrix.get(i, 0)) > 1e-10) {
                isConsistent = false;
                break;
            }
        }
        
        System.out.println("数据一致性检查: " + (isConsistent ? "通过" : "失败"));
    }
}
